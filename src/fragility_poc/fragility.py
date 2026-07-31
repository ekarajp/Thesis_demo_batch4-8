"""Lognormal fragility fitting from complete IDA capacities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .constants import LIMIT_STATES
from .db import connect, initialize, transaction, upsert_many
from .io_utils import atomic_write_json, stable_hash


FRAGILITY_ESTIMATOR = (
    "complete-ida-lognormal-sample-ddof1-v2-qc-warning-nonfatal"
)


def fragility_bootstrap_seed(
    building_id: str,
    config: dict[str, Any],
) -> int:
    """Return a stable per-building seed independent of batch ordering."""
    building_token = int(
        stable_hash(
            {
                "schema": "fragility-bootstrap-seed-v1",
                "building_id": str(building_id),
            }
        )[:16],
        16,
    )
    # NumPy accepts a 32-bit unsigned seed. Keep the configured research seed
    # visible while making each Building ID reproducible whether it is fitted
    # alone, in Batch 001, or in the complete 375-building dataset.
    return (
        int(config["random_seed"]) + building_token
    ) % np.iinfo(np.uint32).max


def capacity_row_has_current_curve_signature(
    row: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    """Return whether a persisted capacity belongs to the current IDA curve."""
    from .ida import ida_curve_analysis_signature

    expected = ida_curve_analysis_signature(
        {
            "building_id": row["building_id"],
            "model_hash": row["current_model_hash"],
            "t1_s": row["current_t1_s"],
        },
        {
            "pair_id": row["pair_id"],
            "sha256_x": row["current_sha256_x"],
            "sha256_y": row["current_sha256_y"],
        },
        config,
    )
    return row.get("analysis_signature") == expected


def fragility_source_signature(
    building_id: str,
    spo_analysis_signature: str,
    capacity_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    bootstrap_count: int | None = None,
    bootstrap_seed: int | None = None,
) -> str:
    """Hash every current input used to derive one building fragility."""
    resolved_bootstrap_count = int(
        bootstrap_count
        if bootstrap_count is not None
        else config["ml"]["bootstrap_resamples"]
    )
    resolved_bootstrap_seed = int(
        bootstrap_seed
        if bootstrap_seed is not None
        else fragility_bootstrap_seed(building_id, config)
    )
    payload = [
        {
            "pair_id": str(row["pair_id"]),
            "limit_state": str(row["limit_state"]),
            "threshold_midr": float(row["threshold_midr"]),
            "capacity_im_g": float(row["capacity_im_g"]),
            "censoring": str(row.get("censoring", "none")),
            "analysis_signature": row.get("analysis_signature"),
        }
        for row in sorted(
            capacity_rows,
            key=lambda item: (
                str(item["pair_id"]),
                str(item["limit_state"]),
            ),
        )
    ]
    return stable_hash(
        {
            "schema": "fragility-source-v5-bootstrap-provenance",
            "estimator": FRAGILITY_ESTIMATOR,
            "building_id": building_id,
            "spo_analysis_signature": spo_analysis_signature,
            "capacity_rows": payload,
            "limit_state_midr": config["ida"]["limit_state_midr"],
            "bootstrap": {
                "requested_resamples": resolved_bootstrap_count,
                "building_seed": resolved_bootstrap_seed,
                "limit_state_seeds": {
                    state: resolved_bootstrap_seed + 1009 * state_index
                    for state_index, state in enumerate(LIMIT_STATES)
                },
            },
        }
    )


def fragility_probability(
    intensity_g: np.ndarray | float,
    theta_g: float,
    beta: float,
) -> np.ndarray:
    intensity = np.asarray(intensity_g, dtype=float)
    probability = np.zeros_like(intensity)
    positive = intensity > 0
    probability[positive] = norm.cdf(
        np.log(intensity[positive] / theta_g) / beta
    )
    return probability


def fit_censored_lognormal(
    capacities_g: np.ndarray,
    censored: np.ndarray,
) -> tuple[float, float]:
    capacities = np.asarray(capacities_g, dtype=float)
    censor_input = np.asarray(censored)
    if censor_input.dtype.kind in {"b", "i", "u", "f"}:
        censoring = np.where(
            censor_input.astype(bool),
            "right",
            "none",
        )
    else:
        censoring = np.char.lower(censor_input.astype(str))
    allowed = {"none", "right", "left"}
    if any(value not in allowed for value in censoring):
        raise ValueError("Censoring must be none, right, or left")
    uncensored_mask = censoring == "none"
    right_mask = censoring == "right"
    left_mask = censoring == "left"
    if capacities.ndim != 1 or capacities.size < 2:
        raise ValueError("At least two capacities are required")
    if np.any(capacities <= 0) or np.any(~np.isfinite(capacities)):
        raise ValueError("Capacities must be positive and finite")
    uncensored = capacities[uncensored_mask]
    if uncensored.size == 0:
        raise ValueError(
            "At least one uncensored capacity is required to identify "
            "the lognormal fit"
        )
    logs = np.log(capacities)
    initial_mu = float(np.mean(np.log(uncensored)))
    initial_sigma = (
        max(float(np.std(np.log(uncensored), ddof=0)), 0.15)
        if uncensored.size > 1
        else 0.40
    )

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        mu = float(parameters[0])
        sigma = math.exp(float(parameters[1]))
        z = (logs - mu) / sigma
        uncensored_ll = (
            norm.logpdf(z[uncensored_mask])
            - math.log(sigma)
            - logs[uncensored_mask]
        ).sum()
        right_ll = norm.logsf(z[right_mask]).sum()
        left_ll = norm.logcdf(z[left_mask]).sum()
        total = float(uncensored_ll + right_ll + left_ll)
        return -total if math.isfinite(total) else 1.0e100

    result = minimize(
        negative_log_likelihood,
        np.asarray([initial_mu, math.log(initial_sigma)]),
        method="L-BFGS-B",
        bounds=((-8.0, 5.0), (math.log(0.02), math.log(3.0))),
    )
    if not result.success:
        raise RuntimeError(f"Censored lognormal MLE failed: {result.message}")
    theta = math.exp(float(result.x[0]))
    beta = math.exp(float(result.x[1]))
    return theta, beta


def fit_complete_lognormal_sample(
    capacities_g: np.ndarray,
) -> tuple[float, float]:
    """Fit median and record-to-record dispersion from complete capacities.

    The median is the geometric mean of the first-crossing IM capacities.
    Beta is the Bessel-corrected sample standard deviation of their natural
    logarithms (ddof=1), matching the final Draft-5 research definition.
    """
    capacities = np.asarray(capacities_g, dtype=float)
    if capacities.ndim != 1 or capacities.size < 2:
        raise ValueError("At least two complete capacities are required")
    if np.any(capacities <= 0) or np.any(~np.isfinite(capacities)):
        raise ValueError("Capacities must be positive and finite")
    logs = np.log(capacities)
    theta = math.exp(float(np.mean(logs)))
    beta = float(np.std(logs, ddof=1))
    if beta <= 0 or not math.isfinite(beta):
        raise ValueError(
            "Complete capacities have zero or non-finite sample dispersion"
        )
    return theta, beta


def _fit_one_limit_state(
    rows: list[dict[str, Any]],
    *,
    bootstrap_count: int,
    seed: int,
) -> dict[str, Any]:
    capacities = np.asarray([float(row["capacity_im_g"]) for row in rows])
    censoring = np.asarray(
        [
            str(row.get("censoring") or (
                "right" if bool(row["censored"]) else "none"
            ))
            for row in rows
        ]
    )
    censored_count = int(np.sum(censoring != "none"))
    if censored_count:
        raise ValueError(
            "Final fragility fitting requires complete uncensored "
            f"first-crossing capacities; found {censored_count} censored rows"
        )
    theta, beta = fit_complete_lognormal_sample(capacities)
    rng = np.random.default_rng(seed)
    bootstrap_theta = []
    bootstrap_beta = []
    for _ in range(bootstrap_count):
        sample = rng.integers(0, len(rows), len(rows))
        sample_capacity = capacities[sample]
        try:
            sampled_theta, sampled_beta = fit_complete_lognormal_sample(
                sample_capacity
            )
            if sampled_beta > 0 and math.isfinite(sampled_beta):
                bootstrap_theta.append(sampled_theta)
                bootstrap_beta.append(sampled_beta)
        except (ValueError, RuntimeError):
            continue
    if len(bootstrap_theta) < max(100, bootstrap_count // 2):
        raise RuntimeError("Too few valid bootstrap resamples")
    return {
        "theta_g": theta,
        "beta": beta,
        "estimator": FRAGILITY_ESTIMATOR,
        "beta_definition": (
            "Bessel-corrected sample standard deviation (ddof=1) of "
            "ln(complete first-crossing IM capacity); record-to-record "
            "dispersion only"
        ),
        "sample_dispersion_warning": bool(beta < 0.02 or beta > 3.0),
        "censored_count": 0,
        "right_censored_count": 0,
        "left_censored_count": 0,
        "theta_ci95_g": np.percentile(bootstrap_theta, [2.5, 97.5]).tolist(),
        "beta_ci95": np.percentile(bootstrap_beta, [2.5, 97.5]).tolist(),
        "requested_bootstrap_count": int(bootstrap_count),
        "bootstrap_seed": int(seed),
        "valid_bootstrap_count": len(bootstrap_theta),
    }


def fit_building_fragility(
    building_id: str,
    capacity_rows: list[dict[str, Any]],
    *,
    bootstrap_count: int,
    seed: int,
    expected_pair_count: int,
) -> dict[str, Any]:
    pair_ids = {str(row["pair_id"]) for row in capacity_rows}
    if len(pair_ids) != expected_pair_count:
        raise ValueError(
            f"{building_id}: {len(pair_ids)} complete pairs; "
            f"{expected_pair_count} required"
        )
    fit: dict[str, dict[str, Any]] = {}
    for state_index, limit_state in enumerate(LIMIT_STATES):
        rows = [
            row
            for row in capacity_rows
            if row["limit_state"] == limit_state
        ]
        if len(rows) != expected_pair_count:
            raise ValueError(
                f"{building_id}/{limit_state}: incomplete capacity rows"
            )
        fit[limit_state] = _fit_one_limit_state(
            rows,
            bootstrap_count=bootstrap_count,
            seed=seed + 1009 * state_index,
        )
    theta_values = [float(fit[state]["theta_g"]) for state in LIMIT_STATES]
    beta_values = [float(fit[state]["beta"]) for state in LIMIT_STATES]
    errors = []
    warnings = []
    if not theta_values[0] < theta_values[1] < theta_values[2]:
        errors.append("theta ordering violation")
    if any(beta <= 0 or not math.isfinite(beta) for beta in beta_values):
        errors.append("non-positive or non-finite beta")
    dispersion_warning_states = [
        state
        for state in LIMIT_STATES
        if bool(fit[state]["sample_dispersion_warning"])
    ]
    if dispersion_warning_states:
        warnings.append(
            "QC warning only--unusually small or large sample "
            "log-dispersion: "
            + ",".join(dispersion_warning_states)
        )
    for state in LIMIT_STATES:
        probability_at_median = float(
            fragility_probability(
                float(fit[state]["theta_g"]),
                float(fit[state]["theta_g"]),
                float(fit[state]["beta"]),
            )
        )
        if not math.isclose(probability_at_median, 0.5, abs_tol=1.0e-10):
            errors.append(f"{state} P(theta) != 0.5")
    valid = not errors
    messages = list(errors)
    if valid:
        messages.append("valid monotonic lognormal fragility")
    messages.extend(warnings)
    return {
        "building_id": building_id,
        "theta_io_g": fit["IO"]["theta_g"],
        "beta_io": fit["IO"]["beta"],
        "theta_ls_g": fit["LS"]["theta_g"],
        "beta_ls": fit["LS"]["beta"],
        "theta_cp_g": fit["CP"]["theta_g"],
        "beta_cp": fit["CP"]["beta"],
        "n_pairs": expected_pair_count,
        "n_censored_io": fit["IO"]["censored_count"],
        "n_censored_ls": fit["LS"]["censored_count"],
        "n_censored_cp": fit["CP"]["censored_count"],
        "bootstrap_json": fit,
        "valid": int(valid),
        "validation_message": "; ".join(messages),
    }


def fit_all_fragilities(
    config: dict[str, Any],
    *,
    building_ids: list[str] | None = None,
    bootstrap_count: int | None = None,
    allow_incomplete_records: bool = False,
) -> dict[str, Any]:
    initialize(config["database_path"])
    bootstrap = int(
        bootstrap_count
        if bootstrap_count is not None
        else config["ml"]["bootstrap_resamples"]
    )
    with connect(config["database_path"]) as connection:
        parameters: list[Any] = []
        sql = (
            "SELECT c.*, sp.analysis_signature AS spo_analysis_signature, "
            "b.model_hash AS current_model_hash, "
            "sp.t1_s AS current_t1_s, "
            "g.source_set AS current_source_set, "
            "g.sha256_x AS current_sha256_x, "
            "g.sha256_y AS current_sha256_y "
            "FROM ida_capacities c "
            "JOIN building_ground_motion_selection s "
            "ON s.building_id=c.building_id AND s.pair_id=c.pair_id "
            "JOIN building_catalog b USING(building_id) "
            "JOIN spo_features sp USING(building_id) "
            "JOIN ground_motion_catalog g USING(pair_id) "
            "WHERE b.selected=1 AND b.valid=1 AND sp.valid=1 AND g.valid=1 "
            "AND s.analysis_role='fragility_primary'"
        )
        if building_ids:
            marks = ",".join("?" for _ in building_ids)
            sql += f" AND c.building_id IN ({marks})"
            parameters.extend(building_ids)
        sql += " ORDER BY c.building_id, c.pair_id, c.limit_state"
        rows = [dict(row) for row in connection.execute(sql, parameters)]
        selection_stats = {
            str(row["building_id"]): dict(row)
            for row in connection.execute(
                """
                SELECT s.building_id, COUNT(*) AS pair_count,
                       COUNT(DISTINCT COALESCE(
                           g.physical_pair_hash, g.pair_id
                       )) AS physical_pair_count,
                       SUM(CASE WHEN g.valid=1
                                     AND g.source_set LIKE 'PWSA%'
                                THEN 1 ELSE 0 END) AS pwsa_count
                FROM building_ground_motion_selection s
                JOIN ground_motion_catalog g USING(pair_id)
                JOIN building_catalog b USING(building_id)
                JOIN spo_features sp USING(building_id)
                WHERE b.selected=1 AND b.valid=1
                  AND sp.valid=1 AND g.valid=1
                  AND s.analysis_role='fragility_primary'
                GROUP BY s.building_id
                """
            )
        }
        run_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT r.building_id, r.pair_id, r.target_im_g,
                       r.analysis_signature, b.model_hash, sp.t1_s,
                       g.sha256_x, g.sha256_y
                FROM ida_runs r
                JOIN building_catalog b USING(building_id)
                JOIN spo_features sp USING(building_id)
                JOIN ground_motion_catalog g USING(pair_id)
                WHERE b.selected=1 AND b.valid=1
                  AND sp.valid=1 AND g.valid=1
                """
            )
        ]
    from .ida import nltha_analysis_signature
    from .spo import _spo_analysis_signature

    current_run_provenance: dict[tuple[str, str], list[bool]] = {}
    for row in run_rows:
        key = (str(row["building_id"]), str(row["pair_id"]))
        expected_signature = nltha_analysis_signature(
            {
                "building_id": row["building_id"],
                "model_hash": row["model_hash"],
                "t1_s": row["t1_s"],
            },
            {
                "pair_id": row["pair_id"],
                "sha256_x": row["sha256_x"],
                "sha256_y": row["sha256_y"],
            },
            float(row["target_im_g"]),
            config,
        )
        current_run_provenance.setdefault(key, []).append(
            row.get("analysis_signature") == expected_signature
        )
    selection_counts = {
        building_id: int(stats["pair_count"])
        for building_id, stats in selection_stats.items()
    }
    target_building_ids = sorted(selection_counts)
    if building_ids:
        requested = set(building_ids)
        target_building_ids = [
            building_id
            for building_id in target_building_ids
            if building_id in requested
        ]
    # Do not erase the whole requested batch before fitting. Each successfully
    # fitted building is atomically upserted below, so an interruption leaves
    # all earlier current results and any not-yet-refitted historical rows
    # recoverable. Scientific consumers still reject historical rows whose
    # source signature is not current.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["building_id"]), []).append(row)
    results = []
    failures = []
    minimum_pairs = int(
        config["ida"].get("minimum_pairs_for_fragility", 4)
    )
    for building_id in target_building_ids:
        capacity_rows = grouped.get(building_id, [])
        try:
            stale_spo_rows = [
                row
                for row in capacity_rows
                if row.get("spo_analysis_signature")
                != _spo_analysis_signature(
                    {
                        "building_id": row["building_id"],
                        "model_hash": row["current_model_hash"],
                    },
                    config,
                )
            ]
            if stale_spo_rows:
                raise ValueError(
                    f"{building_id}: SPO provenance is stale and modal/SPO "
                    "analysis must be rerun before fragility fitting"
                )
            expected_pairs = selection_counts.get(building_id, 0)
            if expected_pairs < minimum_pairs and not allow_incomplete_records:
                raise ValueError(
                    f"{building_id}: {expected_pairs} selected pairs; "
                    f"minimum {minimum_pairs} required"
                )
            if allow_incomplete_records:
                expected_pairs = len(
                    {str(row["pair_id"]) for row in capacity_rows}
                )
            elif (
                int(selection_stats[building_id]["physical_pair_count"])
                != expected_pairs
                and not allow_incomplete_records
            ):
                raise ValueError(
                    f"{building_id}: selected record set contains duplicated "
                    "physical X-Y pairs"
                )
            capacity_pair_ids = {
                str(row["pair_id"]) for row in capacity_rows
            }
            stale_pairs = [
                pair_id
                for pair_id in capacity_pair_ids
                if not current_run_provenance.get((building_id, pair_id))
                or not all(
                    current_run_provenance[(building_id, pair_id)]
                )
            ]
            if stale_pairs:
                raise ValueError(
                    f"{building_id}: {len(stale_pairs)} IDA pairs have "
                    "missing/stale analysis provenance and must be rerun"
                )
            stale_capacity_pairs = sorted(
                {
                    str(row["pair_id"])
                    for row in capacity_rows
                    if not capacity_row_has_current_curve_signature(
                        row,
                        config,
                    )
                }
            )
            if stale_capacity_pairs:
                raise ValueError(
                    f"{building_id}: {len(stale_capacity_pairs)} IDA "
                    "capacity pairs have stale curve provenance and must "
                    "be rerun"
                )
            expected_thresholds = {
                str(state): float(value)
                for state, value in config["ida"]["limit_state_midr"].items()
            }
            wrong_thresholds = [
                row
                for row in capacity_rows
                if not math.isclose(
                    float(row["threshold_midr"]),
                    expected_thresholds[str(row["limit_state"])],
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ]
            if wrong_thresholds:
                raise ValueError(
                    f"{building_id}: stored IDA capacity thresholds do not "
                    "match the current limit-state configuration"
                )
            bootstrap_seed = fragility_bootstrap_seed(
                building_id,
                config,
            )
            result = fit_building_fragility(
                building_id,
                capacity_rows,
                bootstrap_count=bootstrap,
                seed=bootstrap_seed,
                expected_pair_count=expected_pairs,
            )
            spo_signatures = {
                str(row.get("spo_analysis_signature"))
                for row in capacity_rows
            }
            if len(spo_signatures) != 1:
                raise ValueError(
                    f"{building_id}: inconsistent SPO provenance in "
                    "capacity rows"
                )
            result["source_signature"] = fragility_source_signature(
                building_id,
                spo_signatures.pop(),
                capacity_rows,
                config,
                bootstrap_count=bootstrap,
                bootstrap_seed=bootstrap_seed,
            )
            if allow_incomplete_records:
                result["valid"] = 0
                result["validation_message"] += (
                    "; incomplete demonstration result is not ML-eligible"
                )
            with transaction(config["database_path"]) as connection:
                upsert_many(
                    connection,
                    "fragility_targets",
                    [result],
                    ("building_id",),
                )
            results.append(result)
        except (ValueError, RuntimeError) as exc:
            failures.append(
                {"building_id": building_id, "message": str(exc)}
            )
    summary = {
        "available_building_count": len(grouped),
        "fitted_building_count": len(results),
        "valid_building_count": sum(int(result["valid"]) for result in results),
        "failure_count": len(failures),
        "selected_pair_count_range": (
            [
                min(selection_counts.values()),
                max(selection_counts.values()),
            ]
            if selection_counts
            else [0, 0]
        ),
        "minimum_pair_count": minimum_pairs,
        "bootstrap_resamples": bootstrap,
        "allow_incomplete_records": bool(allow_incomplete_records),
        "incomplete_demonstration": bool(allow_incomplete_records),
        "cms_only_demonstration": bool(
            allow_incomplete_records
            and not any(
                str(row.get("current_source_set", "")).startswith("PWSA")
                for row in rows
            )
        ),
        "failures": failures,
    }
    output_path = Path(config["output_dir"]) / "fragility_fit_summary.json"
    atomic_write_json(output_path, summary)
    summary["summary_path"] = str(output_path.absolute())
    return summary
