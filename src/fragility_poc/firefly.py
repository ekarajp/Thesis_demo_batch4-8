"""Deterministic discrete Firefly search with an exact optimality audit."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FireflyAudit:
    selected_index: int
    firefly_best_index: int
    verified_best_index: int
    firefly_objective: float
    verified_objective: float
    optimality_gap: float
    firefly_found_verified_optimum: bool
    population_size: int
    completed_iterations: int
    stalled_iterations: int
    seed: int
    candidate_count: int
    firefly_search_candidate_count: int
    multi_start_runs: int
    multi_start_success_count: int
    multi_start_success_rate: float
    multi_start_best_gap: float
    multi_start_mean_gap: float
    multi_start_median_gap: float
    multi_start_worst_gap: float
    multi_start_objectives: tuple[float, ...]
    multi_start_gaps: tuple[float, ...]
    multi_start_seeds: tuple[int, ...]
    total_completed_iterations: int


def optimize_discrete_candidates(
    vectors: Sequence[Sequence[float]],
    objectives: Sequence[float],
    *,
    seed: int,
    settings: dict[str, Any],
) -> FireflyAudit:
    """Search a finite feasible pool and certify the result by exact audit.

    Firefly movement is performed in normalized design-variable space and
    snapped to the nearest feasible discrete candidate.  The final exhaustive
    objective scan is intentional: a metaheuristic alone cannot substantiate
    a claim of global optimality.  The certified candidate is returned while
    the Firefly-only gap remains available for thesis reporting.
    """
    matrix = np.asarray(vectors, dtype=float)
    values = np.asarray(objectives, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != values.size:
        raise ValueError("Firefly vectors and objectives have incompatible sizes")
    if not values.size:
        raise ValueError("Firefly optimization requires a non-empty feasible pool")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(values)):
        raise ValueError("Firefly inputs must be finite")
    if np.any(values < 0.0):
        raise ValueError("Firefly objectives cannot be negative")

    verified_index = int(np.argmin(values))
    search_limit = max(
        int(settings.get("maximum_search_candidates", 512)),
        int(settings["population_size"]),
    )
    if values.size > search_limit:
        # Feasibility has already been established upstream.  Candidates
        # outside the least-cost shortlist are price-dominated for the
        # current strength tier and need not make every nearest-neighbour
        # Firefly movement O(N).  The final exact audit still scans all N.
        search_indices = np.argsort(values, kind="stable")[:search_limit]
    else:
        search_indices = np.arange(values.size)
    search_matrix = matrix[search_indices]
    search_values = values[search_indices]

    minima = np.min(search_matrix, axis=0)
    spans = np.max(search_matrix, axis=0) - minima
    spans[spans <= np.finfo(float).eps] = 1.0
    normalized = (search_matrix - minima) / spans
    population_size = min(int(settings["population_size"]), search_values.size)
    maximum_iterations = int(settings["maximum_iterations"])
    stall_limit = int(settings["stall_iterations"])
    beta_zero = float(settings["beta_zero"])
    gamma = float(settings["gamma"])
    alpha = float(settings["alpha_initial"])
    alpha_decay = float(settings["alpha_decay"])
    if (
        population_size < 1
        or maximum_iterations < 1
        or stall_limit < 1
        or beta_zero <= 0.0
        or gamma < 0.0
        or alpha < 0.0
        or not 0.0 < alpha_decay <= 1.0
    ):
        raise ValueError("Invalid Firefly hyperparameters")

    run_count = int(settings.get("multi_start_runs", 1))
    if run_count < 1:
        raise ValueError("Firefly multi_start_runs must be positive")
    seed_sequence = np.random.SeedSequence(int(seed))
    child_sequences = seed_sequence.spawn(run_count)
    run_seeds = tuple(
        int(sequence.generate_state(1, dtype=np.uint32)[0])
        for sequence in child_sequences
    )

    def run_once(run_seed: int) -> tuple[int, int, int]:
        if search_values.size == 1:
            return 0, 0, 0
        rng = np.random.default_rng(run_seed)
        population = rng.choice(
            search_values.size,
            size=population_size,
            replace=search_values.size < population_size,
        ).astype(int)
        best_search_index = int(
            population[np.argmin(search_values[population])]
        )
        stalled = 0
        completed = 0
        current_alpha = alpha
        for iteration in range(maximum_iterations):
            completed = iteration + 1
            previous_best = float(search_values[best_search_index])
            for first in range(population_size):
                first_index = int(population[first])
                for second in range(population_size):
                    second_index = int(population[second])
                    if (
                        search_values[second_index] + 1.0e-12
                        >= search_values[first_index]
                    ):
                        continue
                    delta = (
                        normalized[second_index]
                        - normalized[first_index]
                    )
                    distance_squared = float(np.dot(delta, delta))
                    attractiveness = beta_zero * math.exp(
                        -gamma * distance_squared
                    )
                    random_step = current_alpha * (
                        rng.random(matrix.shape[1]) - 0.5
                    )
                    moved = (
                        normalized[first_index]
                        + attractiveness * delta
                        + random_step
                    )
                    distances = np.sum((normalized - moved) ** 2, axis=1)
                    first_index = int(np.argmin(distances))
                population[first] = first_index
            current = int(
                population[np.argmin(search_values[population])]
            )
            if search_values[current] < search_values[best_search_index]:
                best_search_index = current
            if (
                search_values[best_search_index]
                < previous_best - 1.0e-12
            ):
                stalled = 0
            else:
                stalled += 1
            if stalled >= stall_limit:
                break
            current_alpha *= alpha_decay
        return best_search_index, completed, stalled

    runs = [run_once(run_seed) for run_seed in run_seeds]
    run_objectives = tuple(
        float(search_values[run[0]]) for run in runs
    )
    best_run_index = int(np.argmin(run_objectives))
    best_search_index, completed, stalled = runs[best_run_index]
    best_index = int(search_indices[best_search_index])
    firefly_objective = float(values[best_index])
    verified_objective = float(values[verified_index])
    run_gaps = tuple(
        max(
            (objective - verified_objective)
            / max(abs(verified_objective), 1.0e-12),
            0.0,
        )
        for objective in run_objectives
    )
    gap = run_gaps[best_run_index]
    success_count = sum(value <= 1.0e-12 for value in run_gaps)
    return FireflyAudit(
        # The certified optimum is used by the structural database.
        selected_index=verified_index,
        firefly_best_index=best_index,
        verified_best_index=verified_index,
        firefly_objective=firefly_objective,
        verified_objective=verified_objective,
        optimality_gap=float(max(gap, 0.0)),
        firefly_found_verified_optimum=bool(gap <= 1.0e-12),
        population_size=population_size,
        completed_iterations=completed,
        stalled_iterations=stalled,
        seed=int(seed),
        candidate_count=int(values.size),
        firefly_search_candidate_count=int(search_values.size),
        multi_start_runs=run_count,
        multi_start_success_count=success_count,
        multi_start_success_rate=success_count / run_count,
        multi_start_best_gap=float(min(run_gaps)),
        multi_start_mean_gap=float(np.mean(run_gaps)),
        multi_start_median_gap=float(np.median(run_gaps)),
        multi_start_worst_gap=float(max(run_gaps)),
        multi_start_objectives=run_objectives,
        multi_start_gaps=run_gaps,
        multi_start_seeds=run_seeds,
        total_completed_iterations=sum(run[1] for run in runs),
    )
