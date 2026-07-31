"""Modal and nonlinear static pushover analyses."""

from __future__ import annotations

import copy
import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar

from .db import (
    connect,
    initialize,
    record_pipeline_failure,
    resolve_pipeline_failures,
    transaction,
    upsert_many,
)
from .io_utils import atomic_write_json, read_json, stable_hash
from .structural import (
    _ops,
    build_model,
    modal_properties,
    plastic_hinge_mechanism_snapshot,
    run_gravity,
)

TRILINEAR_IDEALIZATION_METHOD = (
    "literature-two-line-50pct-stiffness-v5-collapse-oriented-spo"
)
SPO_ANALYSIS_SCHEMA_VERSION = (
    "rc-3d-spo-v13-contraflexure-lp-effective-mass"
)


def _spo_analysis_signature(
    building: dict[str, Any],
    config: dict[str, Any],
) -> str:
    return stable_hash(
        {
            "analysis_schema_version": SPO_ANALYSIS_SCHEMA_VERSION,
            "building_id": building["building_id"],
            "model_hash": building.get("model_hash"),
            "model_config": config["model"],
            "idealization_method": TRILINEAR_IDEALIZATION_METHOD,
        }
    )


def _load_buildings(
    database_path: str,
    *,
    building_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    with connect(database_path) as connection:
        parameters: list[Any] = []
        where = "valid=1 AND selected=1"
        if building_ids:
            marks = ",".join("?" for _ in building_ids)
            where += f" AND building_id IN ({marks})"
            parameters.extend(building_ids)
        sql = (
            f"SELECT * FROM building_catalog WHERE {where} "
            "ORDER BY queue_rank"
        )
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        return [dict(row) for row in connection.execute(sql, parameters)]


def _bounded_spo_batch(
    all_buildings: list[dict[str, Any]],
    *,
    current_valid_ids: set[str],
    limit: int | None,
    resume: bool,
) -> list[dict[str, Any]]:
    """Select a deterministic SPO batch that advances when resume is active."""
    if limit is None:
        return all_buildings
    if limit < 1:
        raise ValueError("SPO batch limit must be at least 1")
    candidates = (
        [
            building
            for building in all_buildings
            if str(building["building_id"]) not in current_valid_ids
        ]
        if resume
        else all_buildings
    )
    return candidates[:limit]


def _base_shear(
    base_nodes: tuple[int, ...],
    *,
    load_factor: float,
    dof: int = 1,
) -> float:
    """Return signed base shear using the lateral-pattern load-factor sign."""
    ops = _ops()
    ops.reactions()
    magnitude = abs(
        sum(float(ops.nodeReaction(tag, dof)) for tag in base_nodes)
    )
    if abs(load_factor) <= 1.0e-14:
        return 0.0
    return math.copysign(magnitude, load_factor)


def _displacement_step_ladder(
    requested_increment_m: float,
    minimum_increment_m: float,
) -> tuple[float, ...]:
    """Return an exact halving ladder ending at the configured minimum step."""
    if requested_increment_m <= 0.0 or minimum_increment_m <= 0.0:
        raise ValueError("SPO displacement increments must be positive")
    if minimum_increment_m > requested_increment_m:
        raise ValueError(
            "spo_min_displacement_step_m cannot exceed "
            "spo_displacement_step_m"
        )
    steps = [float(requested_increment_m)]
    tolerance = 1.0e-12 * max(1.0, requested_increment_m)
    while steps[-1] / 2.0 > minimum_increment_m + tolerance:
        steps.append(steps[-1] / 2.0)
    if steps[-1] > minimum_increment_m + tolerance:
        steps.append(float(minimum_increment_m))
    return tuple(steps)


def _has_postpeak_ultimate_evidence(
    shear: list[float] | np.ndarray,
    *,
    ultimate_ratio: float = 0.80,
) -> bool:
    """Return whether the converged path crossed the post-peak U target."""
    values = np.asarray(shear, dtype=float)
    if values.size < 3 or not np.all(np.isfinite(values)):
        return False
    peak_index = int(np.argmax(values))
    peak = float(values[peak_index])
    if peak <= 0.0 or peak_index >= values.size - 1:
        return False
    return bool(
        np.any(values[peak_index + 1 :] <= ultimate_ratio * peak)
    )


def _classify_solver_exhaustion(
    shear: list[float] | np.ndarray,
) -> tuple[int, str, str]:
    """Separate a model endpoint candidate from a numerical failure."""
    if _has_postpeak_ultimate_evidence(shear):
        return (
            1,
            "model_endpoint_postpeak_solver_exhaustion",
            "post-peak flexural-model equilibrium path exhausted after the "
            "complete step/algorithm recovery ladder; excluded brittle "
            "failure modes are not represented",
        )
    return (
        0,
        "unresolved_numerical_failure_before_ultimate",
        "solver exhaustion occurred before a converged post-peak "
        "0.80Vmax crossing and is not classified as a model endpoint",
    )


def _try_static_step(
    control_node: int,
    increment_m: float,
    *,
    minimum_increment_m: float,
    dof: int = 1,
) -> tuple[int, int, float, int, int]:
    """Try one SPO step through step reductions and solution algorithms.

    Returns ``(result, algorithm_level, attempted_step, failed_attempts,
    step_reduction_level)``. A nonzero result means that every combination in
    the recovery ladder failed at the last committed structural state.
    """
    ops = _ops()
    recovery_algorithms = (
        ("Newton", ()),
        ("NewtonLineSearch", (0.8,)),
        ("KrylovNewton", ()),
        ("BFGS", ()),
        ("ModifiedNewton", ("-initial",)),
    )
    failed_attempts = 0
    steps = _displacement_step_ladder(
        increment_m,
        minimum_increment_m,
    )
    for step_level, attempted_step in enumerate(steps):
        for recovery, (name, arguments) in enumerate(recovery_algorithms):
            ops.test("NormDispIncr", 1.0e-7, 100, 0)
            ops.algorithm(name, *arguments)
            ops.integrator(
                "DisplacementControl",
                control_node,
                dof,
                attempted_step,
            )
            result = int(ops.analyze(1))
            if result == 0:
                return (
                    result,
                    recovery,
                    attempted_step,
                    failed_attempts,
                    step_level,
                )
            failed_attempts += 1
    return (
        -1,
        len(recovery_algorithms),
        steps[-1],
        failed_attempts,
        len(steps) - 1,
    )


def _trilinear_response(
    displacement: np.ndarray,
    dy: float,
    vy: float,
    dc: float,
    vc: float,
    du: float,
    vu: float,
) -> np.ndarray:
    response = np.empty_like(displacement)
    ultimate_tolerance = 1.0e-12 * max(1.0, abs(du))
    first = displacement <= dy
    second = (displacement > dy) & (displacement <= dc)
    third = (displacement > dc) & (
        displacement <= du + ultimate_tolerance
    )
    beyond_ultimate = displacement > du + ultimate_tolerance
    response[first] = vy / max(dy, 1.0e-12) * displacement[first]
    response[second] = vy + (vc - vy) * (
        (displacement[second] - dy) / max(dc - dy, 1.0e-12)
    )
    response[third] = vc + (vu - vc) * (
        (displacement[third] - dc) / max(du - dc, 1.0e-12)
    )
    response[beyond_ultimate] = np.nan
    return response


def _slope_through_origin(
    displacement: np.ndarray,
    shear: np.ndarray,
) -> float:
    denominator = float(np.dot(displacement, displacement))
    if denominator <= 0.0:
        raise ValueError("Elastic-fit displacement range is degenerate")
    return float(np.dot(displacement, shear) / denominator)


def _local_tangent_stiffness(
    displacement: np.ndarray,
    shear: np.ndarray,
    *,
    half_window: int = 2,
) -> np.ndarray:
    """Return local least-squares slopes for robust stiffness-loss detection."""
    tangent = np.empty_like(displacement)
    for index in range(displacement.size):
        start = max(0, index - half_window)
        stop = min(displacement.size, index + half_window + 1)
        if stop - start < 3:
            if start == 0:
                stop = min(displacement.size, 3)
            else:
                start = max(0, displacement.size - 3)
        local_x = displacement[start:stop]
        local_y = shear[start:stop]
        centered_x = local_x - float(np.mean(local_x))
        denominator = float(np.dot(centered_x, centered_x))
        tangent[index] = (
            float(
                np.dot(
                    centered_x,
                    local_y - float(np.mean(local_y)),
                )
                / denominator
            )
            if denominator > 0.0
            else 0.0
        )
    return tangent


def _yield_from_postyield_slope(
    initial_stiffness: float,
    postyield_stiffness: float,
    dc: float,
    vc: float,
) -> tuple[float, float]:
    """Intersect the origin-anchored elastic line with the line through capping."""
    denominator = initial_stiffness - postyield_stiffness
    if denominator <= 0.0:
        raise ValueError("Post-yield stiffness must be below initial stiffness")
    dy = (vc - postyield_stiffness * dc) / denominator
    vy = initial_stiffness * dy
    return float(dy), float(vy)


def _trilinear_area(
    dy: float,
    vy: float,
    dc: float,
    vc: float,
    du: float,
    vu: float,
) -> float:
    """Exact area under the connected trilinear curve through the ultimate point."""
    return float(
        0.5 * vy * dy
        + 0.5 * (vy + vc) * (dc - dy)
        + 0.5 * (vc + vu) * (du - dc)
    )


def _curve_segment_area(
    displacement: np.ndarray,
    shear: np.ndarray,
    start_displacement: float,
    end_displacement: float,
) -> float:
    """Integrate a raw SPO segment with interpolated boundary ordinates."""
    if end_displacement <= start_displacement:
        return 0.0
    interior = (
        (displacement > start_displacement)
        & (displacement < end_displacement)
    )
    segment_x = np.concatenate(
        (
            np.asarray([start_displacement]),
            displacement[interior],
            np.asarray([end_displacement]),
        )
    )
    segment_y = np.concatenate(
        (
            np.asarray(
                [np.interp(start_displacement, displacement, shear)]
            ),
            shear[interior],
            np.asarray([np.interp(end_displacement, displacement, shear)]),
        )
    )
    return float(trapezoid(segment_y, segment_x))


def fit_trilinear(
    displacement: np.ndarray,
    shear: np.ndarray,
    *,
    energy_tolerance: float = 0.05,
    pre_capping_energy_tolerance: float = 0.10,
    post_capping_energy_tolerance: float = 0.10,
    normalized_rmse_tolerance: float = 0.10,
) -> dict[str, float | bool | str]:
    """Literature-based trilinear idealisation with post-yield energy adjustment.

    The initial elastic line is fitted through the origin over the response up
    to 20% of peak strength.  The post-yield fitting range starts where the
    local tangent stiffness first falls to 50% of the initial stiffness, and
    its line is constrained to pass through the capping point.  Their
    intersection defines the effective yield point.  The pre-capping
    post-yield slope is adjusted when either the total-energy or
    pre-capping-energy tolerance is exceeded.  The bounded adjustment
    minimizes the worst normalized adjustable quality metric so that fixing
    one energy metric cannot silently degrade the other metric or the NRMSE.
    """
    displacement = np.asarray(displacement, dtype=float)
    shear = np.asarray(shear, dtype=float)
    valid = (
        displacement.size >= 10
        and displacement.size == shear.size
        and np.all(np.diff(displacement) > 0)
        and np.max(shear) > 0
    )
    if not valid:
        raise ValueError("Insufficient or invalid SPO curve for trilinear fit")
    peak_index = int(np.argmax(shear))
    if peak_index < 4 or peak_index >= displacement.size - 2:
        raise ValueError("SPO curve does not contain adequate pre/post-peak ranges")
    vc = float(shear[peak_index])
    dc = float(displacement[peak_index])
    ultimate_target = 0.80 * vc
    after_peak = np.where(shear[peak_index:] <= ultimate_target)[0]
    postpeak_reached = after_peak.size > 0
    ultimate_index = (
        peak_index + int(after_peak[0]) if postpeak_reached else len(shear) - 1
    )
    if postpeak_reached and ultimate_index > peak_index:
        before_index = ultimate_index - 1
        before_shear = float(shear[before_index])
        after_shear = float(shear[ultimate_index])
        if before_shear > ultimate_target and after_shear < ultimate_target:
            crossing_fraction = (
                (before_shear - ultimate_target)
                / max(before_shear - after_shear, 1.0e-12)
            )
            du = float(
                displacement[before_index]
                + crossing_fraction
                * (
                    displacement[ultimate_index]
                    - displacement[before_index]
                )
            )
            vu = ultimate_target
        else:
            du = float(displacement[ultimate_index])
            vu = float(shear[ultimate_index])
    else:
        du = float(displacement[ultimate_index])
        vu = float(shear[ultimate_index])
    if du <= dc:
        ultimate_index = len(shear) - 1
        du = float(displacement[ultimate_index])
        vu = float(shear[ultimate_index])
    curve_area = _curve_segment_area(
        displacement,
        shear,
        0.0,
        du,
    )

    prepeak_shear = shear[: peak_index + 1]
    initial_crossings = np.where(prepeak_shear >= 0.20 * vc)[0]
    initial_end = (
        int(initial_crossings[0])
        if initial_crossings.size
        else max(3, peak_index // 5)
    )
    initial_end = min(max(initial_end, 3), peak_index - 2)
    initial_stiffness = _slope_through_origin(
        displacement[: initial_end + 1],
        shear[: initial_end + 1],
    )
    capping_secant_stiffness = vc / max(dc, 1.0e-12)
    if initial_stiffness <= capping_secant_stiffness:
        raise ValueError(
            "Fitted initial stiffness is not greater than capping secant stiffness"
        )

    tangent = _local_tangent_stiffness(
        displacement[: peak_index + 1],
        shear[: peak_index + 1],
    )
    loss_candidates = np.where(
        tangent[initial_end:peak_index] <= 0.50 * initial_stiffness
    )[0]
    stiffness_loss_index = (
        initial_end + int(loss_candidates[0])
        if loss_candidates.size
        else max(initial_end, peak_index // 2)
    )
    postyield_fit_start = min(
        max(stiffness_loss_index, initial_end + 1),
        peak_index - 2,
    )
    postyield_x = displacement[postyield_fit_start : peak_index + 1]
    postyield_y = shear[postyield_fit_start : peak_index + 1]
    centered_x = postyield_x - dc
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0.0:
        raise ValueError("Post-yield fitting range is degenerate")
    postyield_stiffness_fit = float(
        np.dot(centered_x, postyield_y - vc) / denominator
    )

    minimum_yield_displacement = float(displacement[1])
    maximum_from_minimum_yield = (
        (vc - initial_stiffness * minimum_yield_displacement)
        / max(dc - minimum_yield_displacement, 1.0e-12)
    )
    lower_postyield_stiffness = max(
        1.0e-8,
        1.0e-6 * initial_stiffness,
    )
    upper_postyield_stiffness = min(
        0.999 * initial_stiffness,
        0.999 * capping_secant_stiffness,
        0.999 * maximum_from_minimum_yield,
    )
    if upper_postyield_stiffness <= lower_postyield_stiffness:
        raise ValueError("No feasible positive post-yield stiffness range")

    postyield_stiffness_fit = float(
        np.clip(
            postyield_stiffness_fit,
            lower_postyield_stiffness,
            upper_postyield_stiffness,
        )
    )

    actual_pre_capping_area = _curve_segment_area(
        displacement,
        shear,
        0.0,
        dc,
    )
    actual_post_capping_area = _curve_segment_area(
        displacement,
        shear,
        dc,
        du,
    )
    ideal_post_capping_area = float(
        0.5 * (vc + vu) * (du - dc)
    )
    post_capping_energy_error = abs(
        ideal_post_capping_area - actual_post_capping_area
    ) / max(abs(actual_post_capping_area), 1.0e-12)
    to_ultimate = displacement <= du + 1.0e-12

    def quality_metrics_for_slope(
        postyield_stiffness: float,
    ) -> dict[str, float]:
        candidate_dy, candidate_vy = _yield_from_postyield_slope(
            initial_stiffness,
            postyield_stiffness,
            dc,
            vc,
        )
        candidate_area = _trilinear_area(
            candidate_dy,
            candidate_vy,
            dc,
            vc,
            du,
            vu,
        )
        total_energy_error = abs(candidate_area - curve_area) / max(
            abs(curve_area),
            1.0e-12,
        )
        ideal_pre_capping_area = float(
            0.5 * candidate_vy * candidate_dy
            + 0.5
            * (candidate_vy + vc)
            * (dc - candidate_dy)
        )
        pre_capping_energy_error = abs(
            ideal_pre_capping_area - actual_pre_capping_area
        ) / max(abs(actual_pre_capping_area), 1.0e-12)
        fitted_to_ultimate = _trilinear_response(
            displacement[to_ultimate],
            candidate_dy,
            candidate_vy,
            dc,
            vc,
            du,
            vu,
        )
        normalized_rmse = math.sqrt(
            float(
                np.mean(
                    (
                        fitted_to_ultimate
                        - shear[to_ultimate]
                    )
                    ** 2
                )
            )
        ) / max(abs(vc), 1.0e-12)
        return {
            "energy_error": total_energy_error,
            "pre_capping_energy_error": pre_capping_energy_error,
            "normalized_rmse": normalized_rmse,
        }

    def adjustment_score(metrics: dict[str, float]) -> float:
        ratios = (
            metrics["energy_error"] / max(energy_tolerance, 1.0e-12),
            metrics["pre_capping_energy_error"]
            / max(pre_capping_energy_tolerance, 1.0e-12),
            metrics["normalized_rmse"]
            / max(normalized_rmse_tolerance, 1.0e-12),
        )
        # The maximum ratio targets the controlling QC gate.  The small
        # secondary term provides a smooth preference when two slopes have
        # nearly equal controlling ratios.
        return float(max(ratios) + 1.0e-3 * sum(ratios))

    metrics_before_adjustment = quality_metrics_for_slope(
        postyield_stiffness_fit
    )
    energy_error_before_adjustment = metrics_before_adjustment[
        "energy_error"
    ]
    pre_capping_energy_error_before_adjustment = (
        metrics_before_adjustment["pre_capping_energy_error"]
    )
    normalized_rmse_before_adjustment = metrics_before_adjustment[
        "normalized_rmse"
    ]
    adjustment_score_before = adjustment_score(metrics_before_adjustment)
    postyield_stiffness = postyield_stiffness_fit
    energy_adjusted = False
    adjustment_reasons = []
    if energy_error_before_adjustment > energy_tolerance:
        adjustment_reasons.append("total_energy")
    if (
        pre_capping_energy_error_before_adjustment
        > pre_capping_energy_tolerance
    ):
        adjustment_reasons.append("pre_capping_energy")
    if adjustment_reasons:
        adjustment = minimize_scalar(
            lambda slope: adjustment_score(
                quality_metrics_for_slope(float(slope))
            ),
            bounds=(
                lower_postyield_stiffness,
                upper_postyield_stiffness,
            ),
            method="bounded",
            options={
                "xatol": max(
                    1.0e-10,
                    1.0e-10 * upper_postyield_stiffness,
                )
            },
        )
        if (
            adjustment.success
            and float(adjustment.fun)
            < adjustment_score_before - 1.0e-12
        ):
            postyield_stiffness = float(adjustment.x)
            energy_adjusted = True

    dy, vy = _yield_from_postyield_slope(
        initial_stiffness,
        postyield_stiffness,
        dc,
        vc,
    )
    metrics_after_adjustment = quality_metrics_for_slope(
        postyield_stiffness
    )
    energy_error = metrics_after_adjustment["energy_error"]
    pre_capping_energy_error = metrics_after_adjustment[
        "pre_capping_energy_error"
    ]
    normalized_rmse = metrics_after_adjustment["normalized_rmse"]
    adjustment_score_after = adjustment_score(metrics_after_adjustment)
    trilinear_quality_valid = bool(
        energy_error <= energy_tolerance
        and pre_capping_energy_error <= pre_capping_energy_tolerance
        and post_capping_energy_error <= post_capping_energy_tolerance
        and normalized_rmse <= normalized_rmse_tolerance
    )
    return {
        "vy_kn": vy,
        "dy_m": dy,
        "vc_kn": vc,
        "dc_m": dc,
        "vu_kn": vu,
        "du_m": du,
        "energy_error": energy_error,
        "energy_error_before_adjustment": energy_error_before_adjustment,
        "pre_capping_energy_error": pre_capping_energy_error,
        "pre_capping_energy_error_before_adjustment": (
            pre_capping_energy_error_before_adjustment
        ),
        "post_capping_energy_error": post_capping_energy_error,
        "normalized_rmse": normalized_rmse,
        "normalized_rmse_before_adjustment": (
            normalized_rmse_before_adjustment
        ),
        "trilinear_quality_valid": trilinear_quality_valid,
        "initial_stiffness_kn_m": initial_stiffness,
        "postyield_stiffness_fit_kn_m": postyield_stiffness_fit,
        "postyield_stiffness_final_kn_m": postyield_stiffness,
        "stiffness_loss_displacement_m": float(
            displacement[stiffness_loss_index]
        ),
        "energy_adjusted": energy_adjusted,
        "slope_adjustment_reason": (
            ";".join(adjustment_reasons)
            if adjustment_reasons
            else "not_required"
        ),
        "slope_adjustment_score_before": adjustment_score_before,
        "slope_adjustment_score_after": adjustment_score_after,
        "idealization_method": TRILINEAR_IDEALIZATION_METHOD,
        "postpeak_reached": postpeak_reached,
        "run_end_displacement_m": float(displacement[-1]),
        "run_end_shear_kn": float(shear[-1]),
        "run_end_shear_ratio_vc": float(shear[-1] / vc),
    }


def _run_one_spo_once(
    building: dict[str, Any],
    config: dict[str, Any],
    *,
    direction: str = "X",
) -> dict[str, Any]:
    started = time.perf_counter()
    ops = _ops()
    info = build_model(building, config)
    gravity_error = run_gravity(info)
    modal = modal_properties(
        info,
        mode_count=int(config["model"].get("modal_mode_count", 12)),
        solver=str(config["model"].get("eigen_solver", "fullGenLapack")),
        minimum_cumulative_mass_ratio=float(
            config["model"][
                "minimum_cumulative_translational_mass_ratio"
            ]
        ),
    )

    direction = direction.upper()
    if direction not in {"X", "Y"}:
        raise ValueError("SPO direction must be X or Y")
    dof = 1 if direction == "X" else 2
    ops.timeSeries("Linear", 20)
    ops.pattern("Plain", 20, 20)
    weights = np.asarray(info.floor_elevations_m, dtype=float)
    weights /= np.sum(weights)
    for node, force in zip(info.master_nodes, weights):
        load = [0.0] * 6
        load[dof - 1] = float(force)
        ops.load(node, *load)
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    increment = float(config["model"]["spo_displacement_step_m"])
    minimum_increment = float(
        config["model"].get(
            "spo_min_displacement_step_m",
            increment / 64.0,
        )
    )
    maximum_steps = int(
        config["model"].get("spo_analysis_guard_max_steps", 10000)
    )
    maximum_runtime_s = float(
        config["model"].get(
            "spo_analysis_guard_max_runtime_s",
            600.0,
        )
    )
    if maximum_steps <= 0 or maximum_runtime_s <= 0.0:
        raise ValueError("SPO analysis guards must be positive")
    _displacement_step_ladder(increment, minimum_increment)
    control_node = info.master_nodes[-1]
    ops.test("NormDispIncr", 1.0e-7, 60, 0)
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", control_node, dof, increment)
    ops.analysis("Static")
    displacements = [0.0]
    shears = [0.0]
    load_factors = [0.0]
    recoveries: list[int] = []
    step_reduction_levels: list[int] = []
    failed_step_attempts = 0
    minimum_attempted_step = increment
    current_increment = increment
    collapse_reached = 0
    collapse_classification = "not_reached"
    collapse_interpretation = (
        "analysis has not reached a defensible flexural-model endpoint"
    )
    collapse_displacement_m: float | None = None
    collapse_roof_drift: float | None = None
    analysis_guard_triggered = 0
    termination_reason = "analysis_guard_max_steps"
    analysis_started = time.perf_counter()
    accepted_steps = 0
    mechanism_sampling_interval = int(
        config["model"].get("mechanism_sampling_interval_steps", 10)
    )
    if mechanism_sampling_interval <= 0:
        raise ValueError("mechanism_sampling_interval_steps must be positive")
    mechanism_snapshots: list[dict[str, Any]] = []

    def record_mechanism(
        displacement_m: float,
        base_shear_kn: float,
        *,
        force: bool = False,
    ) -> None:
        prior_peak = max(shears, default=0.0)
        projected_step = accepted_steps + 1
        significant_new_peak = (
            base_shear_kn > 0.0
            and base_shear_kn >= 1.02 * max(prior_peak, 1.0e-12)
        )
        first_ultimate_crossing = (
            prior_peak > 0.0
            and base_shear_kn <= 0.80 * prior_peak
            and not any(
                snapshot.get("postpeak_0p8_crossed")
                for snapshot in mechanism_snapshots
            )
        )
        if not (
            force
            or significant_new_peak
            or first_ultimate_crossing
            or projected_step % mechanism_sampling_interval == 0
        ):
            return
        snapshot = plastic_hinge_mechanism_snapshot(
            info,
            building,
            config,
        )
        snapshot.update(
            {
                "accepted_step": projected_step,
                "roof_displacement_m": float(displacement_m),
                "base_shear_kn": float(base_shear_kn),
                "postpeak_0p8_crossed": bool(first_ultimate_crossing),
            }
        )
        mechanism_snapshots.append(snapshot)

    while True:
        if accepted_steps >= maximum_steps:
            analysis_guard_triggered = 1
            termination_reason = "analysis_guard_max_steps"
            collapse_classification = "analysis_guard_without_collapse"
            collapse_interpretation = (
                "maximum analysis-step guard reached; this is not a "
                "modeled collapse endpoint"
            )
            break
        if time.perf_counter() - analysis_started >= maximum_runtime_s:
            analysis_guard_triggered = 1
            termination_reason = "analysis_guard_max_runtime"
            collapse_classification = "analysis_guard_without_collapse"
            collapse_interpretation = (
                "maximum analysis-runtime guard reached; this is not a "
                "modeled collapse endpoint"
            )
            break
        (
            result,
            recovery,
            attempted_increment,
            failed_attempts,
            step_reduction_level,
        ) = _try_static_step(
            control_node,
            current_increment,
            minimum_increment_m=minimum_increment,
            dof=dof,
        )
        failed_step_attempts += failed_attempts
        minimum_attempted_step = min(
            minimum_attempted_step,
            attempted_increment,
        )
        recoveries.append(recovery)
        step_reduction_levels.append(step_reduction_level)
        if result != 0:
            (
                collapse_reached,
                collapse_classification,
                collapse_interpretation,
            ) = _classify_solver_exhaustion(shears)
            termination_reason = collapse_classification
            collapse_displacement_m = float(displacements[-1])
            collapse_roof_drift = (
                collapse_displacement_m
                / float(info.floor_elevations_m[-1])
            )
            break
        displacement = abs(float(ops.nodeDisp(control_node, dof)))
        load_factor = float(ops.getLoadFactor(20))
        shear = _base_shear(
            info.base_nodes,
            load_factor=load_factor,
            dof=dof,
        )
        if displacement <= displacements[-1] + 1.0e-12:
            (
                collapse_reached,
                collapse_classification,
                collapse_interpretation,
            ) = _classify_solver_exhaustion(shears)
            termination_reason = (
                "model_endpoint_postpeak_no_displacement_progress"
                if collapse_reached
                else "unresolved_no_displacement_progress_before_ultimate"
            )
            collapse_classification = termination_reason
            collapse_displacement_m = float(displacements[-1])
            collapse_roof_drift = (
                collapse_displacement_m
                / float(info.floor_elevations_m[-1])
            )
            break
        if (
            load_factor <= 0.0
            and len(shears) > 2
            and max(shears) > 0.0
        ):
            previous_displacement = float(displacements[-1])
            previous_shear = float(shears[-1])
            if previous_shear > 0.0 and shear <= 0.0:
                crossing_fraction = previous_shear / max(
                    previous_shear - shear,
                    1.0e-12,
                )
                collapse_displacement_m = float(
                    previous_displacement
                    + crossing_fraction
                    * (displacement - previous_displacement)
                )
            else:
                collapse_displacement_m = displacement
            record_mechanism(displacement, max(shear, 0.0), force=True)
            displacements.append(collapse_displacement_m)
            shears.append(0.0)
            load_factors.append(0.0)
            collapse_reached = 1
            collapse_classification = (
                "model_static_instability_zero_lateral_resistance"
            )
            collapse_interpretation = (
                "the converged flexure-controlled post-peak equilibrium path "
                "reached zero positive lateral resistance; this is a model "
                "endpoint, not a claim that every physical failure mode was "
                "simulated"
            )
            collapse_roof_drift = (
                collapse_displacement_m
                / float(info.floor_elevations_m[-1])
            )
            termination_reason = collapse_classification
            accepted_steps += 1
            break
        record_mechanism(displacement, shear)
        displacements.append(displacement)
        shears.append(shear)
        load_factors.append(load_factor)
        accepted_steps += 1
        current_increment = min(
            increment,
            max(minimum_increment, 2.0 * attempted_increment),
        )

    curve_displacement = np.asarray(displacements)
    curve_shear = np.asarray(shears)
    energy_tolerance = float(config["model"]["spo_energy_tolerance"])
    pre_capping_energy_tolerance = float(
        config["model"].get(
            "spo_pre_capping_energy_tolerance",
            0.10,
        )
    )
    post_capping_energy_tolerance = float(
        config["model"].get(
            "spo_post_capping_energy_tolerance",
            0.10,
        )
    )
    normalized_rmse_tolerance = float(
        config["model"].get("spo_normalized_rmse_tolerance", 0.10)
    )
    fit = fit_trilinear(
        curve_displacement,
        curve_shear,
        energy_tolerance=energy_tolerance,
        pre_capping_energy_tolerance=pre_capping_energy_tolerance,
        post_capping_energy_tolerance=post_capping_energy_tolerance,
        normalized_rmse_tolerance=normalized_rmse_tolerance,
    )

    if not mechanism_snapshots and len(displacements) > 1:
        mechanism_snapshot = plastic_hinge_mechanism_snapshot(
            info,
            building,
            config,
        )
        mechanism_snapshot.update(
            {
                "accepted_step": accepted_steps,
                "roof_displacement_m": float(displacements[-1]),
                "base_shear_kn": float(shears[-1]),
                "postpeak_0p8_crossed": bool(fit["postpeak_reached"]),
            }
        )
        mechanism_snapshots.append(mechanism_snapshot)

    def mechanism_nearest(displacement_m: float) -> dict[str, Any]:
        if not mechanism_snapshots:
            return {
                "mechanism_class": "unavailable",
                "beam_yielded_end_fraction": math.nan,
                "column_yielded_end_fraction": math.nan,
                "maximum_story_column_yielded_end_fraction": math.nan,
            }
        return min(
            mechanism_snapshots,
            key=lambda snapshot: abs(
                float(snapshot["roof_displacement_m"])
                - float(displacement_m)
            ),
        )

    mechanism_capping = mechanism_nearest(float(fit["dc_m"]))
    mechanism_ultimate = mechanism_nearest(float(fit["du_m"]))
    mechanism_run_end = mechanism_nearest(float(displacements[-1]))
    output_directory = (
        Path(config["run_dir"]) / "spo" / str(building["building_id"])
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    curve_path = output_directory / f"spo_{direction.lower()}.csv"
    mechanism_history_path = (
        output_directory / f"mechanism_{direction.lower()}.json"
    )
    atomic_write_json(
        mechanism_history_path,
        {
            "building_id": building["building_id"],
            "direction": direction,
            "diagnostic_method": (
                "maximum longitudinal strain evaluated at every actual "
                "member-end reinforcing-fiber coordinate relative to fy/Es"
            ),
            "not_ann_input": True,
            "snapshots": mechanism_snapshots,
        },
    )
    fitted_full = _trilinear_response(
        curve_displacement,
        float(fit["dy_m"]),
        float(fit["vy_kn"]),
        float(fit["dc_m"]),
        float(fit["vc_kn"]),
        float(fit["du_m"]),
        float(fit["vu_kn"]),
    )
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("roof_displacement_m", "base_shear_kn", "trilinear_kn"))
        writer.writerows(zip(curve_displacement, curve_shear, fitted_full))

    valid = (
        gravity_error <= 0.01
        and modal.period_error <= 0.01
        and bool(fit["trilinear_quality_valid"])
        and bool(fit["postpeak_reached"])
        and bool(collapse_reached)
        and not bool(analysis_guard_triggered)
    )
    messages = []
    if gravity_error > 0.01:
        messages.append("gravity equilibrium error exceeds 1%")
    if modal.period_error > 0.01:
        messages.append("X/Y period mismatch exceeds 1%")
    if float(fit["energy_error"]) > energy_tolerance:
        messages.append(
            "trilinear energy error exceeds "
            f"{100.0 * energy_tolerance:.1f}%"
        )
    if (
        float(fit["pre_capping_energy_error"])
        > pre_capping_energy_tolerance
    ):
        messages.append(
            "pre-capping energy error exceeds "
            f"{100.0 * pre_capping_energy_tolerance:.1f}%"
        )
    if (
        float(fit["post_capping_energy_error"])
        > post_capping_energy_tolerance
    ):
        messages.append(
            "post-capping energy error exceeds "
            f"{100.0 * post_capping_energy_tolerance:.1f}%"
        )
    if float(fit["normalized_rmse"]) > normalized_rmse_tolerance:
        messages.append(
            "trilinear normalized RMSE exceeds "
            f"{100.0 * normalized_rmse_tolerance:.1f}%"
        )
    if not bool(fit["postpeak_reached"]):
        messages.append("post-peak 0.8Vmax was not reached")
    if not collapse_reached:
        messages.append(collapse_interpretation)
    else:
        messages.append(
            "SPO endpoint: "
            f"{collapse_classification}; {collapse_interpretation}"
        )
    if not messages:
        messages.append("valid modal and SPO result")
    return {
        "building_id": building["building_id"],
        "t1_s": (modal.t_x_s + modal.t_y_s) / 2.0,
        # t2_s means the second global eigenperiod.  The directional X/Y
        # mode identities and periods are stored separately, so silently
        # skipping the repeated orthogonal mode here would make this field
        # neither a global T2 nor a directional second-mode period.
        "t2_s": float(modal.periods_s[1]) if len(modal.periods_s) > 1 else None,
        "gravity_error": gravity_error,
        "x_y_period_error": modal.period_error,
        "x_mode": modal.x_mode,
        "y_mode": modal.y_mode,
        "x_mode_effective_mass_ratio": (
            modal.x_mode_effective_mass_ratio
        ),
        "y_mode_effective_mass_ratio": (
            modal.y_mode_effective_mass_ratio
        ),
        "cumulative_x_effective_mass_ratio": (
            modal.cumulative_x_effective_mass_ratio
        ),
        "cumulative_y_effective_mass_ratio": (
            modal.cumulative_y_effective_mass_ratio
        ),
        "modal_identification_method": modal.identification_method,
        "plastic_hinge_length_method": (
            info.plastic_hinge_length_method
        ),
        "plastic_hinge_shear_span_method": (
            info.plastic_hinge_shear_span_method
        ),
        "plastic_hinge_reference_fallback_end_count": (
            info.plastic_hinge_reference_fallback_end_count
        ),
        "plastic_hinge_reference_clipped_end_count": (
            info.plastic_hinge_reference_clipped_end_count
        ),
        "column_hinge_length_median_m": info.column_hinge_length_m,
        "beam_hinge_length_median_m": info.beam_hinge_length_m,
        **fit,
        "postpeak_reached": int(bool(fit["postpeak_reached"])),
        "curve_path": str(curve_path.absolute()),
        "mechanism_history_path": str(mechanism_history_path.absolute()),
        "mechanism_class_at_capping": mechanism_capping[
            "mechanism_class"
        ],
        "mechanism_class_at_ultimate": mechanism_ultimate[
            "mechanism_class"
        ],
        "mechanism_class_at_run_end": mechanism_run_end[
            "mechanism_class"
        ],
        "capping_beam_yielded_end_fraction": mechanism_capping[
            "beam_yielded_end_fraction"
        ],
        "capping_column_yielded_end_fraction": mechanism_capping[
            "column_yielded_end_fraction"
        ],
        "capping_max_story_column_yielded_end_fraction": mechanism_capping[
            "maximum_story_column_yielded_end_fraction"
        ],
        "ultimate_beam_yielded_end_fraction": mechanism_ultimate[
            "beam_yielded_end_fraction"
        ],
        "ultimate_column_yielded_end_fraction": mechanism_ultimate[
            "column_yielded_end_fraction"
        ],
        "ultimate_max_story_column_yielded_end_fraction": (
            mechanism_ultimate[
                "maximum_story_column_yielded_end_fraction"
            ]
        ),
        "runtime_s": time.perf_counter() - started,
        "valid": int(valid),
        "validation_message": "; ".join(messages),
        "maximum_recovery_level": max(recoveries, default=0),
        "maximum_step_reduction_level": max(
            step_reduction_levels,
            default=0,
        ),
        "minimum_attempted_step_m": minimum_attempted_step,
        "failed_step_attempts": failed_step_attempts,
        "accepted_step_count": accepted_steps,
        "last_converged_load_factor": float(load_factors[-1]),
        "collapse_reached": int(bool(collapse_reached)),
        "collapse_classification": collapse_classification,
        "collapse_interpretation": collapse_interpretation,
        "collapse_displacement_m": collapse_displacement_m,
        "collapse_roof_drift": collapse_roof_drift,
        "analysis_guard_triggered": int(bool(analysis_guard_triggered)),
        "spo_termination_reason": termination_reason,
        "direction": direction,
        "analysis_signature": _spo_analysis_signature(building, config),
    }


def run_one_spo(
    building: dict[str, Any],
    config: dict[str, Any],
    *,
    direction: str = "X",
) -> dict[str, Any]:
    """Run SPO with deterministic curve-resolution refinement when needed.

    The engineering quality gates are never relaxed.  Refinement only reduces
    the displacement increment when the converged curve is too coarsely
    sampled for the required pre/post-peak trilinear ranges.
    """
    started = time.perf_counter()
    requested_step = float(config["model"]["spo_displacement_step_m"])
    requested_minimum = float(
        config["model"].get(
            "spo_min_displacement_step_m",
            requested_step / 64.0,
        )
    )
    retry_message = "SPO curve does not contain adequate pre/post-peak ranges"
    last_error: ValueError | None = None
    for refinement_level in range(3):
        attempt_config = copy.deepcopy(config)
        divisor = float(2**refinement_level)
        effective_step = requested_step / divisor
        effective_minimum = requested_minimum / divisor
        attempt_config["model"]["spo_displacement_step_m"] = effective_step
        attempt_config["model"][
            "spo_min_displacement_step_m"
        ] = effective_minimum
        try:
            result = _run_one_spo_once(
                building,
                attempt_config,
                direction=direction,
            )
        except ValueError as exc:
            last_error = exc
            if str(exc) == retry_message and refinement_level < 2:
                continue
            raise
        result["runtime_s"] = time.perf_counter() - started
        result["requested_displacement_step_m"] = requested_step
        result["effective_displacement_step_m"] = effective_step
        result["adaptive_refinement_level"] = refinement_level
        result["adaptive_refinement_reason"] = (
            "coarse_curve_pre_post_peak_range"
            if refinement_level
            else "not_required"
        )
        # The signature describes the deterministic adaptive policy and the
        # user's requested base step, not the internally selected retry step.
        result["analysis_signature"] = _spo_analysis_signature(
            building,
            config,
        )
        return result
    if last_error is not None:  # pragma: no cover - loop always raises/returns
        raise last_error
    raise RuntimeError("Adaptive SPO refinement exited unexpectedly")


def _count_ida_capacities_for_buildings(
    connection: Any,
    building_ids: set[str],
) -> int:
    """Count derived IDA labels only for buildings whose SPO must be rerun."""
    if not building_ids:
        return 0
    marks = ",".join("?" for _ in building_ids)
    return int(
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM ida_capacities
            WHERE building_id IN ({marks})
            """,
            sorted(building_ids),
        ).fetchone()[0]
    )


def run_modal_spo(
    config: dict[str, Any],
    *,
    building_ids: list[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
    validate_xy: bool = False,
) -> dict[str, Any]:
    initialize(config["database_path"])
    all_buildings = _load_buildings(
        config["database_path"],
        building_ids=building_ids,
        # Apply a bounded batch after current, valid result signatures have
        # been identified.  Otherwise repeating ``--limit 50`` would keep
        # selecting the first 50 queue rows forever.
        limit=None,
    )
    existing: set[str] = set()
    stale_analysis: set[str] = set()
    if resume:
        with connect(config["database_path"]) as connection:
            stored_rows = {
                str(row["building_id"]): dict(row)
                for row in connection.execute(
                    "SELECT building_id, analysis_signature, valid "
                    "FROM spo_features "
                    "WHERE idealization_method=?",
                    (TRILINEAR_IDEALIZATION_METHOD,),
                )
            }
        existing = {
            str(building["building_id"])
            for building in all_buildings
            if (
                bool(
                    stored_rows.get(
                        str(building["building_id"]), {}
                    ).get("valid")
                )
                and stored_rows[str(building["building_id"])][
                    "analysis_signature"
                ]
                == _spo_analysis_signature(building, config)
            )
        }
        stale_analysis = {
            str(building["building_id"])
            for building in all_buildings
            if (
                str(building["building_id"]) in stored_rows
                and str(building["building_id"]) not in existing
            )
        }
    buildings = _bounded_spo_batch(
        all_buildings,
        current_valid_ids=existing,
        limit=limit,
        resume=resume,
    )
    pending_building_ids = {
        str(building["building_id"])
        for building in buildings
        if str(building["building_id"]) not in existing
    }
    if pending_building_ids:
        with transaction(config["database_path"]) as connection:
            production_capacity_count = _count_ida_capacities_for_buildings(
                connection,
                pending_building_ids,
            )
            if production_capacity_count:
                raise RuntimeError(
                    "SPO definitions cannot be changed for pending buildings "
                    "that already have IDA capacities. Archive/reset those "
                    "derived results before rerunning SPO."
                )
            # A split made before all pending SPO analyses were complete would
            # no longer represent the eligible Building-ID population.
            connection.execute("DELETE FROM ml_runs")
            connection.execute("DELETE FROM ml_split_manifest")
            connection.execute("DELETE FROM ml_split")
    results = []
    failures: list[dict[str, str]] = []
    for building in buildings:
        if building["building_id"] in existing:
            continue
        building_id = str(building["building_id"])
        if not resume or building_id in stale_analysis:
            with transaction(config["database_path"]) as connection:
                connection.execute(
                    "DELETE FROM fragility_targets WHERE building_id=?",
                    (building_id,),
                )
                connection.execute(
                    "DELETE FROM ida_capacities WHERE building_id=?",
                    (building_id,),
                )
                connection.execute(
                    "DELETE FROM ida_runs WHERE building_id=?",
                    (building_id,),
                )
                connection.execute(
                    "DELETE FROM building_ground_motion_selection "
                    "WHERE building_id=?",
                    (building_id,),
                )
                connection.execute(
                    "DELETE FROM spo_features WHERE building_id=?",
                    (building_id,),
                )
        try:
            result = run_one_spo(building, config)
            database_row = {
                key: value
                for key, value in result.items()
                if key
                in {
                    "building_id",
                    "t1_s",
                    "t2_s",
                    "gravity_error",
                    "x_y_period_error",
                    "x_mode",
                    "y_mode",
                    "x_mode_effective_mass_ratio",
                    "y_mode_effective_mass_ratio",
                    "cumulative_x_effective_mass_ratio",
                    "cumulative_y_effective_mass_ratio",
                    "modal_identification_method",
                    "vy_kn",
                    "dy_m",
                    "vc_kn",
                    "dc_m",
                    "vu_kn",
                    "du_m",
                    "energy_error",
                    "energy_error_before_adjustment",
                    "pre_capping_energy_error",
                    "pre_capping_energy_error_before_adjustment",
                    "post_capping_energy_error",
                    "normalized_rmse",
                    "normalized_rmse_before_adjustment",
                    "trilinear_quality_valid",
                    "initial_stiffness_kn_m",
                    "postyield_stiffness_fit_kn_m",
                    "postyield_stiffness_final_kn_m",
                    "stiffness_loss_displacement_m",
                    "energy_adjusted",
                    "slope_adjustment_reason",
                    "slope_adjustment_score_before",
                    "slope_adjustment_score_after",
                    "idealization_method",
                    "postpeak_reached",
                    "run_end_displacement_m",
                    "run_end_shear_kn",
                    "run_end_shear_ratio_vc",
                    "spo_termination_reason",
                    "maximum_recovery_level",
                    "maximum_step_reduction_level",
                    "minimum_attempted_step_m",
                    "requested_displacement_step_m",
                    "effective_displacement_step_m",
                    "adaptive_refinement_level",
                    "adaptive_refinement_reason",
                    "failed_step_attempts",
                    "accepted_step_count",
                    "last_converged_load_factor",
                    "collapse_reached",
                    "collapse_classification",
                    "collapse_interpretation",
                    "collapse_displacement_m",
                    "collapse_roof_drift",
                    "analysis_guard_triggered",
                    "mechanism_history_path",
                    "mechanism_class_at_capping",
                    "mechanism_class_at_ultimate",
                    "mechanism_class_at_run_end",
                    "capping_beam_yielded_end_fraction",
                    "capping_column_yielded_end_fraction",
                    "capping_max_story_column_yielded_end_fraction",
                    "ultimate_beam_yielded_end_fraction",
                    "ultimate_column_yielded_end_fraction",
                    "ultimate_max_story_column_yielded_end_fraction",
                    "analysis_signature",
                    "curve_path",
                    "runtime_s",
                    "valid",
                    "validation_message",
                }
            }
            with transaction(config["database_path"]) as connection:
                upsert_many(
                    connection,
                    "spo_features",
                    [database_row],
                    ("building_id",),
                )
            results.append(result)
            if result["valid"]:
                resolve_pipeline_failures(
                    config["database_path"],
                    stage="SPO",
                    building_id=building_id,
                )
            else:
                record_pipeline_failure(
                    config["database_path"],
                    stage="SPO",
                    error=result["validation_message"] or "SPO quality control failed",
                    building_id=building_id,
                    details={
                        "collapse_classification": result[
                            "collapse_classification"
                        ],
                        "curve_path": result["curve_path"],
                    },
                )
        except Exception as exc:
            record_pipeline_failure(
                config["database_path"],
                stage="SPO",
                error=exc,
                building_id=building_id,
            )
            failures.append(
                {
                    "building_id": building_id,
                    "message": str(exc),
                }
            )
    summary = {
        "requested_count": len(buildings),
        "completed_count": len(results),
        "skipped_count": sum(
            building["building_id"] in existing for building in buildings
        ),
        "failure_count": len(failures),
        "valid_count": sum(int(result["valid"]) for result in results),
        "selected_population_count": len(all_buildings),
        "valid_before_batch": len(existing),
        "remaining_after_batch": max(
            0,
            len(all_buildings)
            - len(existing)
            - sum(int(result["valid"]) for result in results),
        ),
        "bounded_batch_advances_pending_queue": bool(
            resume and limit is not None
        ),
        "failures": failures,
    }
    if (
        len(buildings) >= 3
        and (
            validate_xy
            or (building_ids is None and limit is None)
        )
    ):
        summary["xy_equivalence"] = validate_xy_spo_equivalence(
            config,
            buildings[:3],
            resume=resume,
        )
    return summary


def validate_xy_spo_equivalence(
    config: dict[str, Any],
    buildings: list[dict[str, Any]],
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Run three prescribed Y-direction checks for the symmetric archetype."""
    report_path = Path(config["output_dir"]) / "spo_xy_equivalence.json"
    requested_building_ids = [
        str(building["building_id"]) for building in buildings
    ]
    requested_signatures = {
        str(building["building_id"]): _spo_analysis_signature(building, config)
        for building in buildings
    }
    if resume and report_path.is_file():
        existing_report = read_json(report_path)
        if (
            existing_report.get("idealization_method")
            == TRILINEAR_IDEALIZATION_METHOD
            and existing_report.get("curve_acceptance_scope")
            == "zero_to_common_ultimate_Du"
            and existing_report.get("building_ids")
            == requested_building_ids
            and existing_report.get("analysis_signatures")
            == requested_signatures
        ):
            return existing_report
    feature_names = ("t1_s", "vy_kn", "dy_m", "vc_kn", "dc_m", "vu_kn", "du_m")
    checks = []
    with connect(config["database_path"]) as connection:
        x_rows = {
            row["building_id"]: dict(row)
            for row in connection.execute(
                "SELECT * FROM spo_features WHERE building_id IN ({})".format(
                    ",".join("?" for _ in buildings)
                ),
                [building["building_id"] for building in buildings],
            )
        }
    for building in buildings:
        building_id = str(building["building_id"])
        if building_id not in x_rows:
            continue
        y_result = run_one_spo(building, config, direction="Y")
        x_result = x_rows[building_id]
        errors = {}
        for feature in feature_names:
            x_value = float(x_result[feature])
            y_value = float(y_result[feature])
            errors[feature] = abs(x_value - y_value) / max(
                (abs(x_value) + abs(y_value)) / 2.0,
                1.0e-12,
            )
        x_curve = np.loadtxt(
            x_result["curve_path"], delimiter=",", skiprows=1
        )
        y_curve = np.loadtxt(
            y_result["curve_path"], delimiter=",", skiprows=1
        )
        common_run_end = min(
            float(np.max(x_curve[:, 0])),
            float(np.max(y_curve[:, 0])),
        )
        # Symmetry acceptance is evaluated through the common ultimate point,
        # which is the response range used by the seven ANN features.  The
        # unstable post-ultimate path to zero resistance is still retained as
        # a separate diagnostic: tiny numerical perturbations can select
        # different equilibrium branches near static collapse even in a
        # perfectly symmetric model.
        common_ultimate = min(
            float(x_result["du_m"]),
            float(y_result["du_m"]),
            common_run_end,
        )

        def response_nrmse(maximum_displacement: float) -> tuple[
            float,
            np.ndarray,
            np.ndarray,
        ]:
            displacement_grid = np.linspace(
                0.0,
                maximum_displacement,
                500,
            )
            x_values = np.interp(
                displacement_grid,
                x_curve[:, 0],
                x_curve[:, 1],
            )
            y_values = np.interp(
                displacement_grid,
                y_curve[:, 0],
                y_curve[:, 1],
            )
            value = math.sqrt(
                float(np.mean((x_values - y_values) ** 2))
            ) / max(
                float(np.max(np.abs(x_values))),
                float(np.max(np.abs(y_values))),
                1.0e-12,
            )
            return value, x_values, y_values

        curve_nrmse, x_shear, y_shear = response_nrmse(common_ultimate)
        full_run_end_curve_nrmse, _, _ = response_nrmse(common_run_end)
        peak_strength_error = abs(
            float(np.max(x_shear)) - float(np.max(y_shear))
        ) / max(
            (
                float(np.max(x_shear))
                + float(np.max(y_shear))
            )
            / 2.0,
            1.0e-12,
        )
        maximum_feature_error = max(errors.values())
        primary_symmetry_valid = bool(
            y_result["valid"]
            and errors["t1_s"] <= 0.01
            and curve_nrmse <= 0.01
            and peak_strength_error <= 0.01
        )
        checks.append(
            {
                "building_id": building_id,
                "feature_relative_errors": errors,
                "maximum_feature_relative_error": maximum_feature_error,
                "curve_nrmse": curve_nrmse,
                "curve_nrmse_scope": "zero_to_common_ultimate_Du",
                "common_ultimate_displacement_m": common_ultimate,
                "full_run_end_curve_nrmse_diagnostic": (
                    full_run_end_curve_nrmse
                ),
                "peak_strength_relative_error": peak_strength_error,
                "feature_coordinate_consistent": bool(
                    maximum_feature_error <= 0.10
                ),
                "valid": primary_symmetry_valid,
                "y_curve_path": y_result["curve_path"],
            }
        )
    report = {
        "idealization_method": TRILINEAR_IDEALIZATION_METHOD,
        "building_ids": requested_building_ids,
        "analysis_signatures": requested_signatures,
        "required_sample_count": 3,
        "completed_sample_count": len(checks),
        "curve_and_strength_tolerance": 0.01,
        "curve_acceptance_scope": "zero_to_common_ultimate_Du",
        "post_ultimate_run_end_curve_is_diagnostic_only": True,
        "feature_coordinate_diagnostic_tolerance": 0.10,
        "valid": len(checks) == 3 and all(check["valid"] for check in checks),
        "checks": checks,
    }
    atomic_write_json(report_path, report)
    return report
