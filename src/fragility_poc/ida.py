"""Bidirectional NLTHA, adaptive IDA, checkpointing, and compute gating."""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import sqlite3
import tempfile
import time
from collections import deque
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .active_ida import predict_active_capacities
from .catalog import SCWB_RESEARCH_CLASSES
from .constants import G_STD, LIMIT_STATES
from .db import (
    connect,
    initialize,
    record_pipeline_failure,
    resolve_pipeline_failures,
    transaction,
    upsert_many,
)
from .ground_motion import (
    build_ground_motion_selection,
    load_acceleration,
    pair_sa_geomean_g,
)
from .io_utils import atomic_write_json, read_json, stable_hash
from .structural import (
    _ops,
    apply_damping,
    build_model,
    modal_properties,
    run_gravity,
)

NLTHA_ANALYSIS_SCHEMA_VERSION = (
    "bidirectional-nltha-v13-controller-independent-run-cache"
)
NLTHA_RECOVERY_POLICY_VERSION = (
    "local-step-recovery-v7-no-record-restart-consensus"
)
HYBRID_ACTIVE_CONTROLLER_VERSION = (
    "hybrid-active-accuracy-seeker-v4-unbounded-successful-cp-hunt"
)
RECOVERY_CONSENSUS_MIN_ATTEMPTS = 6
RECOVERY_CONSENSUS_MIN_CROSSING_FRACTION = 0.75
RECOVERY_CONSENSUS_MIN_ALGORITHMS = 3
RECOVERY_CONSENSUS_MIN_SUBSTEP_FACTORS = 2
RECOVERY_CONSENSUS_MAX_TIME_SPREAD_RATIO = 0.05
RECOVERY_UPPER_BOUND_MIN_RESPONSE_RATIO = 0.50
RECOVERY_UPPER_BOUND_MAX_RESPONSE_SPREAD_RATIO = 0.25
RECOVERY_UPPER_BOUND_MIN_COMPLETED_FRACTION = 0.01
class ScaleFactorGuardError(ValueError):
    """Raised before analysis when an NLTHA scale factor exceeds its guard."""


def _pair_analysis_role(pair: dict[str, Any]) -> str:
    """Return the explicit research role, with a safe legacy fallback."""
    role = str(pair.get("analysis_role") or "").strip()
    if role:
        return role
    if str(pair.get("source_set", "")).startswith("PWSA"):
        return "event_specific_sensitivity"
    return "fragility_primary"


def _is_primary_fragility_pair(pair: dict[str, Any]) -> bool:
    return _pair_analysis_role(pair) == "fragility_primary"


def _is_pwsa_sf1_sensitivity_pair(pair: dict[str, Any]) -> bool:
    return (
        _pair_analysis_role(pair) == "event_specific_sensitivity"
        and str(
            pair.get("scale_factor_policy")
            or (
                "as_recorded_sf1"
                if str(pair.get("source_set", "")).startswith("PWSA")
                else ""
            )
        )
        == "as_recorded_sf1"
    )


def _validate_scale_factor_value(
    pair_id: str,
    scale_factor: float,
    config: dict[str, Any],
) -> float:
    """Apply the same mandatory guard to new and checkpointed analyses."""
    if not math.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError("NLTHA scale factor must be positive and finite")
    enforce_guard = bool(
        config["ida"].get("enforce_maximum_scale_factor_guard", True)
    )
    maximum_scale_factor = float(
        config["ida"]["maximum_scale_factor_guard"]
    )
    if (
        enforce_guard
        and scale_factor > maximum_scale_factor + 1.0e-12
    ):
        raise ScaleFactorGuardError(
            f"{pair_id}: required scale factor {scale_factor:.8g} "
            f"exceeds mandatory guard {maximum_scale_factor:.8g}"
        )
    return float(scale_factor)


def _checked_scale_factor(
    pair: dict[str, Any],
    period_s: float,
    target_im_g: float,
    config: dict[str, Any],
) -> tuple[float, float]:
    """Return ``(unscaled Sa, scale factor)`` after mandatory guard checks."""
    unscaled_sa = pair_sa_geomean_g(pair, period_s)
    if not math.isfinite(unscaled_sa) or unscaled_sa <= 0.0:
        raise ValueError(
            f"{pair['pair_id']}: unscaled Sa(T1) must be positive and finite"
        )
    scale_factor = _validate_scale_factor_value(
        str(pair["pair_id"]),
        target_im_g / unscaled_sa,
        config,
    )
    return float(unscaled_sa), float(scale_factor)


def _result_path(
    config: dict[str, Any],
    building_id: str,
    pair_id: str,
    target_im_g: float,
) -> Path:
    token = f"{target_im_g:.8f}".rstrip("0").rstrip(".").replace(".", "p")
    return Path(config["run_dir"]) / "ida" / building_id / pair_id / f"im_{token}.json"


def _load_existing_result(
    path: Path,
    expected_signature: str,
    *,
    certified_threshold: float | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    result = read_json(path)
    required = {
        "building_id",
        "pair_id",
        "target_im_g",
        "scale_factor",
        "achieved_im_g",
        "max_midr",
        "status",
        "runtime_s",
        "recovery_level",
        "analysis_signature",
    }
    if not (
        required.issubset(result)
        and result["analysis_signature"] == expected_signature
    ):
        return None
    # A baseline (recovery-level zero) successful or direct-drift-instability
    # run is physically unchanged and remains reusable. Any old result that
    # depended on a fallback, including a consensus instability, must be
    # rerun because local recovery follows a different time-integration path.
    if (
        result.get("recovery_policy_version")
        != NLTHA_RECOVERY_POLICY_VERSION
        and (
            result["status"] == "numerical_failure"
            or int(result.get("recovery_level", 0)) > 0
        )
    ):
        return None
    return result


def _ida_numerical_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return only fields that alter one IDA curve or its saved diagnostics."""
    ida = config["ida"]
    keys = (
        "initial_im_g",
        "minimum_im_g",
        "hunt_multiplier",
        "cp_hunt_soft_warning_im_g",
        "maximum_hunt_steps",
        "scale_factor_review_threshold",
        "maximum_scale_factor_guard",
        "enforce_maximum_scale_factor_guard",
        "continue_scaling_until_cp",
        "maximum_unresolved_failure_targets",
        "require_cp_crossing",
        "bracket_tolerance",
        "limit_state_midr",
    )
    numerical = {key: ida[key] for key in keys}
    numerical["controller"] = ida.get(
        "controller", {"mode": "legacy_hunt_fill"}
    )
    return numerical


def _nltha_run_numerical_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return fields that can alter one physical NLTHA result.

    Point-selection settings such as the active-controller model, hunt
    multiplier, and bracket tolerance do not change the structural response
    at a fixed target IM.  Excluding them lets a stopped batch or a
    controller-tolerance sensitivity reuse its atomic NLTHA checkpoints
    without treating the measured response as a new simulation.
    """
    ida = config["ida"]
    keys = (
        "cp_hunt_soft_warning_im_g",
        "scale_factor_review_threshold",
        "maximum_scale_factor_guard",
        "limit_state_midr",
    )
    return {key: ida[key] for key in keys}


def nltha_analysis_signature(
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
) -> str:
    """Return the provenance signature for one nonlinear time-history run."""
    return stable_hash(
        {
            "analysis_schema_version": NLTHA_ANALYSIS_SCHEMA_VERSION,
            "model_hash": building.get("model_hash"),
            "building_id": building["building_id"],
            "t1_s": float(building["t1_s"]),
            "pair_id": pair["pair_id"],
            "sha256_x": pair.get("sha256_x"),
            "sha256_y": pair.get("sha256_y"),
            "target_im_g": float(target_im_g),
            "model_config": config["model"],
            "nltha_run_numerical_config": _nltha_run_numerical_config(config),
        }
    )


def ida_curve_analysis_signature(
    building: dict[str, Any],
    pair: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """Return one signature shared by all points/capacities of an IDA curve."""
    return stable_hash(
        {
            "analysis_schema_version": NLTHA_ANALYSIS_SCHEMA_VERSION,
            "recovery_policy_version": NLTHA_RECOVERY_POLICY_VERSION,
            "model_hash": building.get("model_hash"),
            "building_id": building["building_id"],
            "t1_s": building.get("t1_s"),
            "pair_id": pair["pair_id"],
            "sha256_x": pair.get("sha256_x"),
            "sha256_y": pair.get("sha256_y"),
            "model_config": config["model"],
            "ida_numerical_config": _ida_numerical_config(config),
        }
    )


def _configure_transient(
    algorithm: str,
    *,
    test_type: str = "EnergyIncr",
    tolerance: float = 1.0e-7,
    max_iterations: int = 30,
) -> None:
    ops = _ops()
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    # The dense band solver was benchmarked on the original PoC model as
    # 27-32% faster than UmfPack while changing MIDR by only 0.0058% at a
    # nonlinear CP-exceeding point. The Draft 5 migration smoke test retains
    # this deterministic solver; the multi-bay runtime is reported separately.
    ops.system("BandGeneral")
    _set_transient_solution_strategy(
        algorithm,
        test_type=test_type,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")


def _set_transient_solution_strategy(
    algorithm: str,
    *,
    test_type: str,
    tolerance: float,
    max_iterations: int,
) -> None:
    """Change convergence strategy without wiping state or recorders."""
    ops = _ops()
    ops.test(test_type, tolerance, max_iterations, 0)
    if algorithm == "NewtonLineSearch":
        ops.algorithm("NewtonLineSearch", 0.8)
    elif algorithm == "ModifiedNewton":
        ops.algorithm("ModifiedNewton", "-initial")
    elif algorithm == "KrylovNewton":
        ops.algorithm("KrylovNewton")
    else:
        ops.algorithm("Newton")


def _maximum_midr_from_recorded_displacements(
    recorder_data: np.ndarray,
    floor_elevations_m: tuple[float, ...] | list[float],
) -> float:
    """Calculate Draft 4 max-component MIDR from a Node-recorder matrix."""
    data = np.asarray(recorder_data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    floor_count = len(floor_elevations_m)
    expected_columns = 1 + 2 * floor_count
    if data.size == 0 or data.shape[1] != expected_columns:
        return math.nan
    if not np.all(np.isfinite(data)):
        return math.nan
    displacements = data[:, 1:].reshape(data.shape[0], floor_count, 2)
    base = np.zeros((data.shape[0], 1, 2), dtype=float)
    storey_displacements = np.diff(
        np.concatenate((base, displacements), axis=1),
        axis=1,
    )
    elevations = np.asarray(floor_elevations_m, dtype=float)
    storey_heights = np.diff(
        np.concatenate((np.asarray([0.0]), elevations))
    )
    if np.any(storey_heights <= 0.0):
        raise ValueError("Floor elevations must increase strictly")
    drift_components = np.abs(storey_displacements) / (
        storey_heights.reshape(1, floor_count, 1)
    )
    return float(np.max(drift_components))


def _run_transient_with_node_recorder(
    info: Any,
    *,
    total_steps: int,
    analysis_dt: float,
) -> tuple[int, float, int]:
    """Run OpenSees in C++ and recover the complete MIDR time envelope.

    The earlier step-by-step Python loop made eight ``nodeDisp`` calls for
    every step. A Node recorder preserves every X/Y floor displacement while
    ``ops.analyze`` advances the identical Newmark/convergence sequence
    internally. The temporary recorder is removed after its envelope is read.
    """
    ops = _ops()
    descriptor, recorder_name = tempfile.mkstemp(
        prefix="fragility_midr_",
        suffix=".txt",
    )
    os.close(descriptor)
    recorder_path = Path(recorder_name)
    try:
        ops.recorder(
            "Node",
            "-file",
            str(recorder_path),
            "-precision",
            16,
            "-time",
            "-node",
            *info.master_nodes,
            "-dof",
            1,
            2,
            "disp",
        )
        result = int(ops.analyze(total_steps, analysis_dt))
        ops.remove("recorders")
        if (
            not recorder_path.is_file()
            or recorder_path.stat().st_size == 0
        ):
            return result, math.nan, 0
        recorder_data = np.loadtxt(recorder_path, dtype=float, ndmin=2)
        completed_steps = int(recorder_data.shape[0])
        maximum_midr = _maximum_midr_from_recorded_displacements(
            recorder_data,
            info.floor_elevations_m,
        )
        return result, maximum_midr, completed_steps
    finally:
        try:
            ops.remove("recorders")
        except Exception:
            pass
        recorder_path.unlink(missing_ok=True)


def _single_nltha_attempt(
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
    *,
    substep_factor: int,
    algorithm: str,
    scale_factor: float,
    test_type: str = "EnergyIncr",
    tolerance: float = 1.0e-7,
    max_iterations: int = 30,
) -> dict[str, Any]:
    # Defense in depth: even internal callers and sensitivity utilities must
    # pass the same mandatory scale-factor guard as the public NLTHA entry
    # point.  This prevents direct/private calls from bypassing the safety
    # contract before acceleration values are assembled.
    scale_factor = _validate_scale_factor_value(
        str(pair["pair_id"]),
        float(scale_factor),
        config,
    )
    ops = _ops()
    info = build_model(building, config)
    gravity_error = run_gravity(info)
    if gravity_error > 0.01:
        raise RuntimeError(
            f"gravity equilibrium error {gravity_error:.3%} exceeds 1%"
        )
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
    if modal.period_error > 0.01:
        raise RuntimeError(
            f"X/Y modal-period mismatch {modal.period_error:.3%} exceeds 1%"
        )
    apply_damping(
        modal,
        float(config["model"]["damping_ratio"]),
        method=str(config["model"].get("damping_model", "modal")),
    )

    time_x, acceleration_x_g = load_acceleration(
        pair["component_x_path"], pair["units"]
    )
    time_y, acceleration_y_g = load_acceleration(
        pair["component_y_path"], pair["units"]
    )
    if not np.allclose(time_x, time_y, rtol=0, atol=1.0e-10):
        raise ValueError(f"{pair['pair_id']}: X/Y time vectors differ")
    record_dt = float(pair["dt_s"])
    observed_dt = float(np.median(np.diff(time_x)))
    if not math.isclose(observed_dt, record_dt, rel_tol=1.0e-6, abs_tol=1.0e-10):
        raise ValueError(
            f"{pair['pair_id']}: catalog dt={record_dt} differs from "
            f"component dt={observed_dt}"
        )
    acceleration_x = acceleration_x_g * G_STD * scale_factor
    acceleration_y = acceleration_y_g * G_STD * scale_factor
    ops.timeSeries("Path", 101, "-dt", record_dt, "-values", *acceleration_x)
    ops.timeSeries("Path", 102, "-dt", record_dt, "-values", *acceleration_y)
    ops.pattern("UniformExcitation", 101, 1, "-accel", 101)
    ops.pattern("UniformExcitation", 102, 2, "-accel", 102)
    _configure_transient(
        algorithm,
        test_type=test_type,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    analysis_dt = record_dt / substep_factor
    total_steps = int((len(time_x) - 1) * substep_factor)
    instability_limit = float(config["model"]["dynamic_instability_drift"])
    result, maximum_midr, completed_steps = (
        _run_transient_with_node_recorder(
            info,
            total_steps=total_steps,
            analysis_dt=analysis_dt,
        )
    )
    if not math.isfinite(maximum_midr):
        # NaN/Inf is not an observed physical drift and must never become a
        # fabricated CP/collapse label.
        maximum_midr = 0.0
        status = "numerical_failure"
    elif maximum_midr >= instability_limit:
        status = "dynamic_instability"
    elif result != 0 or completed_steps < total_steps:
        status = "numerical_failure"
    else:
        status = "success"
    return {
        "status": status,
        "max_midr": maximum_midr,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "analysis_dt_s": analysis_dt,
        "modal_period_x_s": modal.t_x_s,
        "modal_period_y_s": modal.t_y_s,
        "modal_x_mode": modal.x_mode,
        "modal_y_mode": modal.y_mode,
        "modal_x_effective_mass_ratio": (
            modal.x_mode_effective_mass_ratio
        ),
        "modal_y_effective_mass_ratio": (
            modal.y_mode_effective_mass_ratio
        ),
        "modal_cumulative_x_effective_mass_ratio": (
            modal.cumulative_x_effective_mass_ratio
        ),
        "modal_cumulative_y_effective_mass_ratio": (
            modal.cumulative_y_effective_mass_ratio
        ),
        "modal_identification_method": modal.identification_method,
        "modal_eigen_solver_requested": getattr(
            modal, "eigen_solver_requested", "unknown"
        ),
        "modal_eigen_solver_used": getattr(
            modal, "eigen_solver_used", "unknown"
        ),
        "modal_eigen_solver_fallback": bool(
            getattr(modal, "eigen_solver_fallback", False)
        ),
        "dynamic_instability_interpretation": str(
            config["model"]["dynamic_instability_interpretation"]
        ),
        "collapse_scope": str(config["model"]["collapse_scope"]),
        "target_im_g": target_im_g,
    }


def _run_transient_with_local_recovery(
    info: Any,
    *,
    total_duration_s: float,
    base_dt_s: float,
    chunk_steps: int,
    recovery_plan: tuple[
        tuple[int, str, str, float, int], ...
    ],
) -> tuple[int, float, int, list[dict[str, Any]], int, float]:
    """Advance one record once, recovering only at its failing time step.

    A successful part of the nonlinear history is never discarded.  The
    default Newton analysis advances in bounded C++ chunks.  When it stops,
    alternative algorithms and smaller local time steps are tried from the
    last committed OpenSees state.  After one local step succeeds, the
    default integration resumes from that state.
    """
    if total_duration_s <= 0.0 or base_dt_s <= 0.0:
        raise ValueError("NLTHA duration and base time step must be positive")
    if chunk_steps < 1:
        raise ValueError("NLTHA chunk_steps must be at least one")
    ops = _ops()
    descriptor, recorder_name = tempfile.mkstemp(
        prefix="fragility_midr_local_recovery_",
        suffix=".txt",
    )
    os.close(descriptor)
    recorder_path = Path(recorder_name)
    start_time = float(ops.getTime())
    end_time = start_time + float(total_duration_s)
    time_tolerance = max(1.0e-10, base_dt_s * 1.0e-8)
    recovery_log: list[dict[str, Any]] = []
    maximum_recovery_level = 0
    terminal_code = 0
    nominal_total_steps = int(math.ceil(total_duration_s / base_dt_s))

    def progress() -> tuple[float, float]:
        elapsed = max(float(ops.getTime()) - start_time, 0.0)
        fraction = min(elapsed / total_duration_s, 1.0)
        return elapsed, fraction

    _configure_transient(
        "Newton",
        test_type="EnergyIncr",
        tolerance=1.0e-7,
        max_iterations=30,
    )
    try:
        ops.recorder(
            "Node",
            "-file",
            str(recorder_path),
            "-precision",
            16,
            "-time",
            "-node",
            *info.master_nodes,
            "-dof",
            1,
            2,
            "disp",
        )
        while float(ops.getTime()) < end_time - time_tolerance:
            remaining = end_time - float(ops.getTime())
            default_steps = min(
                chunk_steps,
                max(int(math.floor(remaining / base_dt_s + 1.0e-9)), 1),
            )
            default_dt = min(base_dt_s, remaining / default_steps)
            _set_transient_solution_strategy(
                "Newton",
                test_type="EnergyIncr",
                tolerance=1.0e-7,
                max_iterations=30,
            )
            before = float(ops.getTime())
            code = int(ops.analyze(default_steps, default_dt))
            after = float(ops.getTime())
            if code == 0:
                continue

            elapsed, fraction = progress()
            recovery_log.append(
                {
                    "recovery_level": 0,
                    "substep_factor": 1,
                    "algorithm": "Newton",
                    "test_type": "EnergyIncr",
                    "tolerance": 1.0e-7,
                    "max_iterations": 30,
                    "status": "numerical_failure",
                    "termination_time_s": elapsed,
                    "completed_fraction": fraction,
                    "committed_time_increment_s": max(after - before, 0.0),
                    "analysis_dt_s": default_dt,
                    "completed_steps": int(
                        round(elapsed / max(base_dt_s, 1.0e-12))
                    ),
                    "total_steps": nominal_total_steps,
                }
            )

            recovered = False
            for recovery_level, (
                substep_factor,
                algorithm,
                test_type,
                tolerance,
                max_iterations,
            ) in enumerate(recovery_plan, start=1):
                remaining = end_time - float(ops.getTime())
                if remaining <= time_tolerance:
                    recovered = True
                    break
                local_dt = min(base_dt_s / substep_factor, remaining)
                _set_transient_solution_strategy(
                    algorithm,
                    test_type=test_type,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
                before = float(ops.getTime())
                local_code = int(ops.analyze(1, local_dt))
                after = float(ops.getTime())
                elapsed, fraction = progress()
                local_status = (
                    "local_recovery_success"
                    if local_code == 0
                    else "numerical_failure"
                )
                recovery_log.append(
                    {
                        "recovery_level": recovery_level,
                        "substep_factor": substep_factor,
                        "algorithm": algorithm,
                        "test_type": test_type,
                        "tolerance": tolerance,
                        "max_iterations": max_iterations,
                        "status": local_status,
                        "termination_time_s": elapsed,
                        "completed_fraction": fraction,
                        "committed_time_increment_s": max(
                            after - before, 0.0
                        ),
                        "analysis_dt_s": local_dt,
                        "completed_steps": int(
                            round(elapsed / max(base_dt_s, 1.0e-12))
                        ),
                        "total_steps": nominal_total_steps,
                    }
                )
                maximum_recovery_level = max(
                    maximum_recovery_level, recovery_level
                )
                if local_code == 0:
                    recovered = True
                    break
            if not recovered:
                terminal_code = code if code != 0 else -1
                break

        ops.remove("recorders")
        elapsed, _ = progress()
        if (
            not recorder_path.is_file()
            or recorder_path.stat().st_size == 0
        ):
            return (
                terminal_code or -1,
                math.nan,
                0,
                recovery_log,
                maximum_recovery_level,
                elapsed,
            )
        recorder_data = np.loadtxt(recorder_path, dtype=float, ndmin=2)
        maximum_midr = _maximum_midr_from_recorded_displacements(
            recorder_data,
            info.floor_elevations_m,
        )
        completed_samples = int(recorder_data.shape[0])
        if terminal_code == 0 and float(ops.getTime()) < (
            end_time - time_tolerance
        ):
            terminal_code = -1
        return (
            terminal_code,
            maximum_midr,
            completed_samples,
            recovery_log,
            maximum_recovery_level,
            elapsed,
        )
    finally:
        try:
            ops.remove("recorders")
        except Exception:
            pass
        recorder_path.unlink(missing_ok=True)


def _single_nltha_local_recovery(
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
    *,
    scale_factor: float,
) -> dict[str, Any]:
    """Build once and run one complete record with local step recovery."""
    scale_factor = _validate_scale_factor_value(
        str(pair["pair_id"]),
        float(scale_factor),
        config,
    )
    ops = _ops()
    info = build_model(building, config)
    gravity_error = run_gravity(info)
    if gravity_error > 0.01:
        raise RuntimeError(
            f"gravity equilibrium error {gravity_error:.3%} exceeds 1%"
        )
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
    if modal.period_error > 0.01:
        raise RuntimeError(
            f"X/Y modal-period mismatch {modal.period_error:.3%} exceeds 1%"
        )
    apply_damping(
        modal,
        float(config["model"]["damping_ratio"]),
        method=str(config["model"].get("damping_model", "modal")),
    )
    time_x, acceleration_x_g = load_acceleration(
        pair["component_x_path"], pair["units"]
    )
    time_y, acceleration_y_g = load_acceleration(
        pair["component_y_path"], pair["units"]
    )
    if not np.allclose(time_x, time_y, rtol=0, atol=1.0e-10):
        raise ValueError(f"{pair['pair_id']}: X/Y time vectors differ")
    record_dt = float(pair["dt_s"])
    observed_dt = float(np.median(np.diff(time_x)))
    if not math.isclose(
        observed_dt, record_dt, rel_tol=1.0e-6, abs_tol=1.0e-10
    ):
        raise ValueError(
            f"{pair['pair_id']}: catalog dt={record_dt} differs from "
            f"component dt={observed_dt}"
        )
    acceleration_x = acceleration_x_g * G_STD * scale_factor
    acceleration_y = acceleration_y_g * G_STD * scale_factor
    ops.timeSeries("Path", 101, "-dt", record_dt, "-values", *acceleration_x)
    ops.timeSeries("Path", 102, "-dt", record_dt, "-values", *acceleration_y)
    ops.pattern("UniformExcitation", 101, 1, "-accel", 101)
    ops.pattern("UniformExcitation", 102, 2, "-accel", 102)

    base_dt = record_dt
    recovery_plan = (
        (2, "Newton", "EnergyIncr", 1.0e-7, 30),
        (4, "Newton", "EnergyIncr", 1.0e-7, 30),
        (4, "NewtonLineSearch", "EnergyIncr", 1.0e-7, 30),
        (4, "ModifiedNewton", "EnergyIncr", 1.0e-7, 30),
        (4, "KrylovNewton", "EnergyIncr", 1.0e-7, 30),
        (4, "ModifiedNewton", "EnergyIncr", 1.0e-6, 100),
        (4, "KrylovNewton", "NormDispIncr", 1.0e-6, 100),
    )
    total_duration = float(time_x[-1] - time_x[0])
    (
        result_code,
        maximum_midr,
        completed_samples,
        attempt_log,
        recovery_level,
        termination_time_s,
    ) = _run_transient_with_local_recovery(
        info,
        total_duration_s=total_duration,
        base_dt_s=base_dt,
        chunk_steps=int(
            config["ida"].get("local_recovery_chunk_steps", 200)
        ),
        recovery_plan=recovery_plan,
    )
    total_steps = int(math.ceil(total_duration / base_dt))
    completed_fraction = min(
        max(termination_time_s / max(total_duration, 1.0e-12), 0.0),
        1.0,
    )
    if not math.isfinite(maximum_midr):
        maximum_midr = 0.0
        status = "numerical_failure"
    elif maximum_midr >= float(
        config["model"]["dynamic_instability_drift"]
    ):
        status = "dynamic_instability"
    elif result_code != 0 or completed_fraction < 1.0 - 1.0e-8:
        status = "numerical_failure"
    else:
        status = "success"
    if not attempt_log:
        attempt_log.append(
            {
                "recovery_level": 0,
                "substep_factor": 1,
                "algorithm": "Newton",
                "test_type": "EnergyIncr",
                "tolerance": 1.0e-7,
                "max_iterations": 30,
                "status": status,
                "termination_time_s": termination_time_s,
                "completed_fraction": completed_fraction,
                "analysis_dt_s": base_dt,
                "completed_steps": total_steps,
                "total_steps": total_steps,
                "committed_time_increment_s": total_duration,
            }
        )
    for attempt in attempt_log:
        attempt["max_midr"] = maximum_midr
        attempt["target_im_g"] = target_im_g
    return {
        "status": status,
        "max_midr": maximum_midr,
        "completed_steps": int(round(completed_fraction * total_steps)),
        "completed_samples": completed_samples,
        "total_steps": total_steps,
        "analysis_dt_s": base_dt,
        "termination_time_s": termination_time_s,
        "completed_fraction": completed_fraction,
        "recovery_level": recovery_level,
        "attempts": attempt_log,
        "modal_period_x_s": modal.t_x_s,
        "modal_period_y_s": modal.t_y_s,
        "modal_x_mode": modal.x_mode,
        "modal_y_mode": modal.y_mode,
        "modal_x_effective_mass_ratio": (
            modal.x_mode_effective_mass_ratio
        ),
        "modal_y_effective_mass_ratio": (
            modal.y_mode_effective_mass_ratio
        ),
        "modal_cumulative_x_effective_mass_ratio": (
            modal.cumulative_x_effective_mass_ratio
        ),
        "modal_cumulative_y_effective_mass_ratio": (
            modal.cumulative_y_effective_mass_ratio
        ),
        "modal_identification_method": modal.identification_method,
        "modal_eigen_solver_requested": getattr(
            modal, "eigen_solver_requested", "unknown"
        ),
        "modal_eigen_solver_used": getattr(
            modal, "eigen_solver_used", "unknown"
        ),
        "modal_eigen_solver_fallback": bool(
            getattr(modal, "eigen_solver_fallback", False)
        ),
        "dynamic_instability_interpretation": str(
            config["model"]["dynamic_instability_interpretation"]
        ),
        "collapse_scope": str(config["model"]["collapse_scope"]),
        "target_im_g": target_im_g,
    }


def _recovery_consensus_dynamic_instability(
    attempt_log: list[dict[str, Any]],
    cp_threshold: float,
) -> dict[str, Any]:
    """Classify recovery-consensus instability and a strict CP upper bound.

    Observing CP before repeatable solver loss certifies dynamic instability.
    If CP was not numerically observed, a narrower second classification may
    still certify only an *upper bound* on CP capacity. It requires full
    recovery exhaustion, consistent physical termination time and response,
    multiple algorithms/substeps, and substantial pre-failure drift. This is
    intentionally distinct from relabelling generic nonconvergence as collapse.
    """
    # Local recovery may have succeeded earlier in the record.  Only the
    # final exhausted episode can support a termination consensus; including
    # earlier recovered episodes would mix different physical times.
    last_recovery_success = max(
        (
            index
            for index, attempt in enumerate(attempt_log)
            if attempt.get("status") == "local_recovery_success"
        ),
        default=-1,
    )
    consensus_attempts = attempt_log[last_recovery_success + 1 :]
    numerical = [
        attempt
        for attempt in consensus_attempts
        if attempt.get("status") == "numerical_failure"
    ]
    eligible = [
        attempt
        for attempt in numerical
        if math.isfinite(float(attempt.get("max_midr", math.nan)))
        and int(attempt.get("completed_steps", 0))
        < int(attempt.get("total_steps", 0))
        and int(attempt.get("completed_steps", 0)) > 0
        and float(
            attempt.get(
                "completed_fraction",
                int(attempt.get("completed_steps", 0))
                / max(int(attempt.get("total_steps", 0)), 1),
            )
        )
        >= RECOVERY_UPPER_BOUND_MIN_COMPLETED_FRACTION
        and float(attempt.get("analysis_dt_s", 0.0)) > 0.0
    ]
    crossing = [
        attempt
        for attempt in eligible
        if float(attempt["max_midr"]) >= cp_threshold
    ]
    upper_support = [
        attempt
        for attempt in eligible
        if float(attempt["max_midr"])
        >= RECOVERY_UPPER_BOUND_MIN_RESPONSE_RATIO * cp_threshold
    ]
    crossing_fraction = len(crossing) / max(len(consensus_attempts), 1)
    upper_support_fraction = len(upper_support) / max(
        len(consensus_attempts), 1
    )
    algorithms = {str(item.get("algorithm")) for item in eligible}
    substeps = {int(item.get("substep_factor", 0)) for item in eligible}
    termination_times = [
        float(
            item.get(
                "termination_time_s",
                int(item["completed_steps"]) * float(item["analysis_dt_s"]),
            )
        )
        for item in eligible
    ]
    median_time = (
        float(np.median(termination_times)) if termination_times else None
    )
    time_spread_ratio = (
        (max(termination_times) - min(termination_times))
        / max(float(median_time), 1.0e-12)
        if termination_times
        else math.inf
    )
    responses = [float(item["max_midr"]) for item in eligible]
    median_response = (
        float(np.median(responses)) if responses else None
    )
    response_spread_ratio = (
        (max(responses) - min(responses))
        / max(float(median_response), 1.0e-12)
        if responses
        else math.inf
    )
    common_consensus = bool(
        len(consensus_attempts) >= RECOVERY_CONSENSUS_MIN_ATTEMPTS
        and len(numerical) == len(consensus_attempts)
        and len(eligible) >= RECOVERY_CONSENSUS_MIN_ATTEMPTS
        and len(algorithms) >= RECOVERY_CONSENSUS_MIN_ALGORITHMS
        and len(substeps) >= RECOVERY_CONSENSUS_MIN_SUBSTEP_FACTORS
        and time_spread_ratio
        <= RECOVERY_CONSENSUS_MAX_TIME_SPREAD_RATIO
    )
    certified = bool(
        common_consensus
        and crossing_fraction
        >= RECOVERY_CONSENSUS_MIN_CROSSING_FRACTION
    )
    upper_bound_certified = bool(
        common_consensus
        and upper_support_fraction
        >= RECOVERY_CONSENSUS_MIN_CROSSING_FRACTION
        and response_spread_ratio
        <= RECOVERY_UPPER_BOUND_MAX_RESPONSE_SPREAD_RATIO
    )
    return {
        "certified": certified,
        "upper_bound_certified": upper_bound_certified,
        "basis": (
            "multi_algorithm_full_recovery_post_cp_time_consensus"
            if certified
            else (
                "multi_algorithm_full_recovery_pre_cp_upper_bound_consensus"
                if upper_bound_certified
                else "no_recovery_consensus"
            )
        ),
        "attempt_count": len(consensus_attempts),
        "full_record_log_entry_count": len(attempt_log),
        "numerical_failure_count": len(numerical),
        "eligible_failure_count": len(eligible),
        "post_cp_crossing_count": len(crossing),
        "post_cp_crossing_fraction": crossing_fraction,
        "upper_bound_support_count": len(upper_support),
        "upper_bound_support_fraction": upper_support_fraction,
        "upper_bound_minimum_response_ratio": (
            RECOVERY_UPPER_BOUND_MIN_RESPONSE_RATIO
        ),
        "distinct_algorithms": sorted(algorithms),
        "distinct_substep_factors": sorted(substeps),
        "median_termination_time_s": median_time,
        "termination_time_spread_ratio": time_spread_ratio,
        "maximum_allowed_time_spread_ratio": (
            RECOVERY_CONSENSUS_MAX_TIME_SPREAD_RATIO
        ),
        "median_max_midr": median_response,
        "max_midr_spread_ratio": response_spread_ratio,
        "maximum_allowed_midr_spread_ratio": (
            RECOVERY_UPPER_BOUND_MAX_RESPONSE_SPREAD_RATIO
        ),
    }


def run_nltha(
    building: dict[str, Any],
    pair: dict[str, Any],
    target_im_g: float,
    config: dict[str, Any],
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Run one target IM with local recovery and atomic checkpointing."""
    path = _result_path(
        config,
        str(building["building_id"]),
        str(pair["pair_id"]),
        target_im_g,
    )
    analysis_signature = nltha_analysis_signature(
        building,
        pair,
        target_im_g,
        config,
    )
    if resume:
        existing = _load_existing_result(
            path,
            analysis_signature,
            certified_threshold=float(
                config["ida"]["limit_state_midr"]["CP"]
            ),
        )
        if existing is not None:
            _validate_scale_factor_value(
                str(pair["pair_id"]),
                float(existing["scale_factor"]),
                config,
            )
            return existing
    t1 = float(building["t1_s"])
    unscaled_sa, scale_factor = _checked_scale_factor(
        pair,
        t1,
        target_im_g,
        config,
    )
    started = time.perf_counter()
    final_attempt = _single_nltha_local_recovery(
        building,
        pair,
        target_im_g,
        config,
        scale_factor=scale_factor,
    )
    recovery_level = int(final_attempt["recovery_level"])
    attempt_log = list(final_attempt["attempts"])
    recovery_consensus = _recovery_consensus_dynamic_instability(
        attempt_log,
        float(config["ida"]["limit_state_midr"]["CP"]),
    )
    if (
        final_attempt["status"] == "numerical_failure"
        and recovery_consensus["certified"]
    ):
        final_attempt = {
            **final_attempt,
            "status": "dynamic_instability",
            "dynamic_instability_basis": recovery_consensus["basis"],
        }
    achieved = unscaled_sa * scale_factor
    soft_warning_im = float(
        config["ida"]["cp_hunt_soft_warning_im_g"]
    )
    scale_factor_review_threshold = float(
        config["ida"]["scale_factor_review_threshold"]
    )
    result = {
        "building_id": building["building_id"],
        "pair_id": pair["pair_id"],
        "target_im_g": target_im_g,
        "scale_factor": scale_factor,
        "unscaled_sa_g": unscaled_sa,
        "achieved_im_g": achieved,
        "scaling_error": abs(achieved - target_im_g) / target_im_g,
        "max_midr": final_attempt["max_midr"],
        "status": final_attempt["status"],
        "dynamic_instability": int(
            final_attempt["status"] == "dynamic_instability"
        ),
        "runtime_s": time.perf_counter() - started,
        "recovery_level": recovery_level,
        "soft_im_warning_exceeded": int(target_im_g > soft_warning_im),
        "scale_factor_warning_exceeded": int(
            scale_factor > scale_factor_review_threshold
        ),
        "result_path": str(path.absolute()),
        "attempts": attempt_log,
        "recovery_consensus": recovery_consensus,
        "certified_instability_upper_bound": int(
            final_attempt["status"] == "dynamic_instability"
            or recovery_consensus["upper_bound_certified"]
        ),
        "dynamic_instability_basis": final_attempt.get(
            "dynamic_instability_basis",
            final_attempt.get(
                "dynamic_instability_interpretation",
                "not_classified",
            ),
        ),
        "analysis_signature": analysis_signature,
        "recovery_policy_version": NLTHA_RECOVERY_POLICY_VERSION,
        "observed_limit_states_before_termination": [
            name
            for name, threshold in config["ida"]["limit_state_midr"].items()
            if float(final_attempt["max_midr"]) >= float(threshold)
        ],
        "certified_limit_states": (
            list(config["ida"]["limit_state_midr"])
            if final_attempt["status"] == "dynamic_instability"
            else [
                name
                for name, threshold in config["ida"][
                    "limit_state_midr"
                ].items()
                if final_attempt["status"] == "success"
                and float(final_attempt["max_midr"]) >= float(threshold)
            ]
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, result)
    return result


def _response_value(result: dict[str, Any]) -> float | None:
    status = result["status"]
    if status == "dynamic_instability":
        return math.inf
    if status == "success":
        return float(result["max_midr"])
    return None


def _threshold_response(
    result: dict[str, Any],
    threshold: float,
) -> float | None:
    """Return a response only when its threshold relation is certified.

    A successful analysis certifies its complete-record maximum. Dynamic
    instability certifies exceedance of every drift threshold. A generic
    numerical failure is diagnostic only: its partial-record maximum may guide
    the next IM trial, but it can never close a capacity bracket.
    """
    response = _response_value(result)
    if response is not None:
        return response
    return None


def _is_certified_upper(
    result: dict[str, Any],
    threshold: float,
    limit_state: str,
) -> bool:
    """Return whether ``result`` may close the upper side of a state bracket.

    A successful complete-record NLTHA may close any upper bracket. Explicitly
    classified dynamic instability may also close a transition-limited bracket;
    IO/LS then require a nearby successful lower point in the controller. A
    generic numerical failure is never a label.
    """
    if bool(result.get("certified_instability_upper_bound", 0)):
        return True
    response = _threshold_response(result, threshold)
    if response is None or response < threshold:
        return False
    if result.get("status") == "success":
        return True
    return result.get("status") == "dynamic_instability"


def _is_transition_search_guide(
    result: dict[str, Any],
    threshold: float,
) -> bool:
    """Use a non-successful threshold crossing only to choose follow-up IM."""
    if (
        result.get("status") == "dynamic_instability"
        or bool(result.get("certified_instability_upper_bound", 0))
    ):
        return True
    return (
        result.get("status") == "numerical_failure"
        and math.isfinite(float(result.get("max_midr", 0.0)))
        and float(result.get("max_midr", 0.0)) >= threshold
    )


def _relative_bracket_width(lower: float, upper: float) -> float:
    return (upper - lower) / max((upper + lower) / 2.0, 1.0e-12)


def _capacity_error_bound(lower: float, upper: float) -> float:
    """Return the maximum relative error of a log-midpoint capacity.

    The true first crossing is certified to lie inside ``[lower, upper]``.
    The returned value is therefore an auditable worst-case numerical
    localization error, not a statistical uncertainty estimate.
    """
    midpoint = math.sqrt(lower * upper)
    return max(midpoint / lower - 1.0, upper / midpoint - 1.0)


def _manual_request_targets(
    config: dict[str, Any],
    building_id: str,
    pair_id: str,
) -> set[float]:
    try:
        with connect(config["database_path"]) as connection:
            rows = connection.execute(
                """
                SELECT target_im_g
                FROM ida_manual_point_requests
                WHERE building_id=? AND pair_id=?
                """,
                (building_id, pair_id),
            ).fetchall()
    except sqlite3.OperationalError:
        # Direct unit-level curve calls may deliberately bypass database
        # initialization. Production batch/add-point entry points initialize
        # the schema before launching a worker.
        return set()
    return {round(float(row["target_im_g"]), 8) for row in rows}


def _load_curve_checkpoint_results(
    building: dict[str, Any],
    pair: dict[str, Any],
    config: dict[str, Any],
) -> dict[float, dict[str, Any]]:
    """Load every reusable fixed-IM checkpoint for one IDA curve.

    This is what makes later Codex/user-directed point additions incremental:
    changing the point-selection policy or adding a target does not rerun any
    fixed-IM NLTHA whose physical analysis signature is still current.
    """
    directory = (
        Path(config["run_dir"])
        / "ida"
        / str(building["building_id"])
        / str(pair["pair_id"])
    )
    if not directory.is_dir():
        return {}
    cp_threshold = float(config["ida"]["limit_state_midr"]["CP"])
    reusable: dict[float, dict[str, Any]] = {}
    for path in sorted(directory.glob("im_*.json")):
        try:
            candidate = read_json(path)
            target = round(float(candidate["target_im_g"]), 8)
            signature = nltha_analysis_signature(
                building, pair, target, config
            )
            result = _load_existing_result(
                path,
                signature,
                certified_threshold=cp_threshold,
            )
            if result is None:
                continue
            _validate_scale_factor_value(
                str(pair["pair_id"]),
                float(result["scale_factor"]),
                config,
            )
            reusable[target] = result
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            # A malformed/stale file is preserved for audit but cannot enter
            # the current signed curve.
            continue
    return reusable


def _checkpoint_failure_guidance_targets(
    building: dict[str, Any],
    pair: dict[str, Any],
    config: dict[str, Any],
    threshold: float,
) -> set[float]:
    """Return physically current failure IMs for point selection only.

    An older recovery-policy failure must be rerun and can never be a response
    label. Its target IM is nevertheless valuable scheduling evidence: retry
    the lowest known threshold-crossing failure before spending time at a
    higher surrogate seed.
    """
    directory = (
        Path(config["run_dir"])
        / "ida"
        / str(building["building_id"])
        / str(pair["pair_id"])
    )
    targets: set[float] = set()
    if not directory.is_dir():
        return targets
    for path in sorted(directory.glob("im_*.json")):
        try:
            candidate = read_json(path)
            target = round(float(candidate["target_im_g"]), 8)
            if (
                candidate.get("status") != "numerical_failure"
                or not math.isfinite(
                    float(candidate.get("max_midr", math.nan))
                )
                or float(candidate["max_midr"]) < threshold
                or candidate.get("analysis_signature")
                != nltha_analysis_signature(
                    building, pair, target, config
                )
            ):
                continue
            targets.add(target)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return targets


def _curve_review_diagnostics(
    *,
    results: dict[float, dict[str, Any]],
    capacities: list[dict[str, Any]],
    tolerance: float,
    soft_review_point_count: int,
    manual_targets: set[float],
) -> dict[str, Any]:
    """Build deterministic evidence for subsequent review by this Codex.

    These checks do not replace engineering judgement and do not create
    labels. They identify curves for visual/numerical review and suggest
    additional IM targets that can be run without repeating existing points.
    """
    flags: list[dict[str, Any]] = []
    recommended_targets: set[float] = set()
    error_bounds: dict[str, float | None] = {}

    for capacity in capacities:
        state = str(capacity["limit_state"])
        lower_raw = capacity.get("lower_im_g")
        upper_raw = capacity.get("upper_im_g")
        if lower_raw is None or upper_raw is None:
            error_bounds[state] = None
            flags.append(
                {
                    "code": "CENSORED_CAPACITY",
                    "severity": "review",
                    "detail": f"{state} does not have a two-sided bracket",
                }
            )
            continue
        lower = float(lower_raw)
        upper = float(upper_raw)
        width = _relative_bracket_width(lower, upper)
        error_bounds[state] = _capacity_error_bound(lower, upper)
        if width > tolerance + 1.0e-12:
            target = round(math.sqrt(lower * upper), 8)
            recommended_targets.add(target)
            flags.append(
                {
                    "code": "BRACKET_WIDER_THAN_TARGET",
                    "severity": "critical",
                    "limit_state": state,
                    "relative_width": width,
                    "recommended_target_im_g": target,
                }
            )

    ordered = [
        (target, result)
        for target, result in sorted(results.items())
        if result.get("status") == "success"
        and math.isfinite(float(result.get("max_midr", math.nan)))
    ]
    for (lower_im, lower_result), (upper_im, upper_result) in zip(
        ordered, ordered[1:]
    ):
        lower_response = float(lower_result["max_midr"])
        upper_response = float(upper_result["max_midr"])
        drop = lower_response - upper_response
        meaningful_drop = max(0.002, 0.10 * max(lower_response, 1.0e-12))
        if drop > meaningful_drop:
            target = round(math.sqrt(lower_im * upper_im), 8)
            if target not in results:
                recommended_targets.add(target)
            flags.append(
                {
                    "code": "NONMONOTONIC_IDA_RESPONSE",
                    "severity": "review",
                    "lower_im_g": lower_im,
                    "upper_im_g": upper_im,
                    "midr_drop": drop,
                    "recommended_target_im_g": target,
                }
            )

    numerical_failures = [
        result
        for result in results.values()
        if result.get("status") == "numerical_failure"
    ]
    certified_numerical_upper_bounds = [
        result
        for result in numerical_failures
        if bool(result.get("certified_instability_upper_bound", 0))
    ]
    unresolved_numerical_failures = [
        result
        for result in numerical_failures
        if not bool(result.get("certified_instability_upper_bound", 0))
    ]
    if unresolved_numerical_failures:
        flags.append(
            {
                "code": "EXHAUSTED_NUMERICAL_FAILURE_PRESENT",
                "severity": "review",
                "count": len(unresolved_numerical_failures),
                "detail": (
                    "Failures are retained for diagnostics/point selection "
                    "only and were not used to close a capacity bracket."
                ),
            }
        )
    if certified_numerical_upper_bounds:
        flags.append(
            {
                "code": "RECOVERY_CONSENSUS_CP_UPPER_BOUND",
                "severity": "review",
                "count": len(certified_numerical_upper_bounds),
                "detail": (
                    "Full-recovery multi-algorithm/time-step consensus was "
                    "used only as a certified upper side of a measured CP "
                    "bracket; generic nonconvergence was not used."
                ),
            }
        )
    dynamic_instability = sum(
        result.get("status") == "dynamic_instability"
        for result in results.values()
    )
    if dynamic_instability:
        flags.append(
            {
                "code": "DYNAMIC_INSTABILITY_UPPER_BOUND",
                "severity": "review",
                "count": dynamic_instability,
            }
        )
    maximum_recovery = max(
        (int(result.get("recovery_level", 0)) for result in results.values()),
        default=0,
    )
    if maximum_recovery >= 3:
        flags.append(
            {
                "code": "HEAVY_NUMERICAL_RECOVERY",
                "severity": "review",
                "maximum_recovery_level": maximum_recovery,
            }
        )
    high_scale_count = sum(
        bool(result.get("scale_factor_warning_exceeded"))
        for result in results.values()
    )
    if high_scale_count:
        flags.append(
            {
                "code": "HIGH_SCALE_FACTOR_REVIEW",
                "severity": "review",
                "count": high_scale_count,
            }
        )
    if len(results) > soft_review_point_count:
        flags.append(
            {
                "code": "POINT_COUNT_ABOVE_SOFT_REVIEW_LEVEL",
                "severity": "review",
                "point_count": len(results),
                "soft_review_point_count": soft_review_point_count,
                "detail": (
                    "This is a review flag only; it never stops refinement."
                ),
            }
        )

    capacity_values = {
        str(row["limit_state"]): float(row["capacity_im_g"])
        for row in capacities
    }
    if (
        set(capacity_values) == set(LIMIT_STATES)
        and not (
            capacity_values["IO"]
            < capacity_values["LS"]
            < capacity_values["CP"]
        )
    ):
        flags.append(
            {
                "code": "CAPACITY_ORDERING_VIOLATION",
                "severity": "review",
                "capacities_g": capacity_values,
                "detail": (
                    "Nested drift thresholds can share a sharp response jump; "
                    "Codex must inspect the measured brackets before accepting "
                    "the record-specific ordering."
                ),
            }
        )

    review_required = any(
        flag["severity"] in {"review", "critical"} for flag in flags
    )
    critical = any(flag["severity"] == "critical" for flag in flags)
    if critical:
        review_status = "CRITICAL_QC_FAILURE"
    elif review_required:
        review_status = "CODEX_REVIEW_REQUIRED"
    else:
        review_status = "PASS_AUTOMATED_QC"
    maximum_error = max(
        (value for value in error_bounds.values() if value is not None),
        default=None,
    )
    return {
        "review_status": review_status,
        "codex_review_required": int(review_required),
        "review_flags": flags,
        "review_flag_count": len(flags),
        "recommended_additional_targets_g": sorted(recommended_targets),
        "capacity_error_bounds": error_bounds,
        "maximum_capacity_error_bound": maximum_error,
        "soft_review_point_count": soft_review_point_count,
        "manually_requested_targets_g": sorted(manual_targets),
        "manually_requested_point_count": len(
            manual_targets.intersection(results)
        ),
    }


def _run_legacy_hunt_fill_curve(
    building: dict[str, Any],
    pair: dict[str, Any],
    config: dict[str, Any],
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Adaptive hunt-and-fill IDA for one building-record pair."""
    ida_config = config["ida"]
    initial = float(ida_config["initial_im_g"])
    minimum = float(ida_config.get("minimum_im_g", initial / 50.0))
    multiplier = float(ida_config["hunt_multiplier"])
    maximum_hunt_steps = int(ida_config["maximum_hunt_steps"])
    maximum_scale_factor = float(
        ida_config["maximum_scale_factor_guard"]
    )
    tolerance = float(ida_config["bracket_tolerance"])
    thresholds = {
        name: float(value)
        for name, value in ida_config["limit_state_midr"].items()
    }
    results: dict[float, dict[str, Any]] = {}
    curve_signature = ida_curve_analysis_signature(building, pair, config)
    scale_guard_message: str | None = None

    def analyse_target(target_im_g: float) -> dict[str, Any] | None:
        nonlocal scale_guard_message
        try:
            return run_nltha(
                building,
                pair,
                target_im_g,
                config,
                resume=resume,
            )
        except ScaleFactorGuardError as exc:
            scale_guard_message = str(exc)
            return None

    def scale_guard_failure() -> dict[str, Any]:
        return {
            "valid": False,
            "building_id": building["building_id"],
            "pair_id": pair["pair_id"],
            "runs": list(results.values()),
            "capacities": [],
            "message": (
                "CP was not observed before the mandatory scale-factor "
                f"safety guard: {scale_guard_message}"
            ),
        }

    target = initial
    cp_crossed = False
    for _hunt_step in range(maximum_hunt_steps):
        rounded_target = round(target, 8)
        result = analyse_target(rounded_target)
        if result is None:
            return scale_guard_failure()
        results[rounded_target] = result
        response = _threshold_response(result, thresholds["CP"])
        if response is None:
            return {
                "valid": False,
                "building_id": building["building_id"],
                "pair_id": pair["pair_id"],
                "runs": list(results.values()),
                "capacities": [],
                "message": "unresolved numerical failure during hunting",
            }
        if response >= thresholds["CP"]:
            cp_crossed = True
            break
        if float(result.get("scale_factor", 0.0)) >= maximum_scale_factor:
            return {
                "valid": False,
                "building_id": building["building_id"],
                "pair_id": pair["pair_id"],
                "runs": list(results.values()),
                "capacities": [],
                "message": (
                    "CP was not observed before the numerical "
                    "scale-factor safety guard"
                ),
            }
        target *= multiplier
    if not cp_crossed:
        return {
            "valid": False,
            "building_id": building["building_id"],
            "pair_id": pair["pair_id"],
            "runs": list(results.values()),
            "capacities": [],
            "message": (
                "CP was not observed within the maximum hunt-step safety "
                "guard; no censored capacity label was created"
            ),
        }

    capacities = []
    for limit_state in LIMIT_STATES:
        threshold = thresholds[limit_state]
        while True:
            ordered = sorted(results.items())
            below = [
                (im, result)
                for im, result in ordered
                if result["status"] == "success"
                and float(result["max_midr"]) < threshold
            ]
            above = [
                (im, result)
                for im, result in ordered
                if (_threshold_response(result, threshold) is not None)
                and float(_threshold_response(result, threshold)) >= threshold
            ]
            if not above:
                return {
                    "valid": False,
                    "building_id": building["building_id"],
                    "pair_id": pair["pair_id"],
                    "runs": list(results.values()),
                    "capacities": [],
                    "message": (
                        f"{limit_state} has no certified upper bracket even "
                        "though CP hunting completed"
                    ),
                }
            upper = min(item[0] for item in above)
            upper_result = results[upper]
            lower_candidates = [item[0] for item in below if item[0] < upper]
            lower = max(lower_candidates, default=0.0)
            if lower <= 0:
                # Threshold exceeded at every analysed intensity. Continue
                # hunting downward until a genuine below-threshold point is
                # found or the explicit minimum IM is reached.
                downward_target = round(
                    max(minimum, upper / multiplier),
                    8,
                )
                if (
                    downward_target < upper - 1.0e-12
                    and downward_target not in results
                ):
                    result = analyse_target(downward_target)
                    if result is None:
                        return scale_guard_failure()
                    results[downward_target] = result
                    if _threshold_response(result, threshold) is None:
                        return {
                            "valid": False,
                            "building_id": building["building_id"],
                            "pair_id": pair["pair_id"],
                            "runs": list(results.values()),
                            "capacities": [],
                            "message": "unresolved numerical failure during filling",
                        }
                    continue
                capacities.append(
                    {
                        "building_id": building["building_id"],
                        "pair_id": pair["pair_id"],
                        "limit_state": limit_state,
                        "threshold_midr": threshold,
                        "capacity_im_g": upper,
                        "censored": 1,
                        "censoring": "left",
                        "lower_im_g": None,
                        "upper_im_g": upper,
                        "upper_bound_status": upper_result["status"],
                        "upper_bound_certified": 1,
                        "analysis_signature": curve_signature,
                    }
                )
                break
            if _relative_bracket_width(lower, upper) <= tolerance:
                capacities.append(
                    {
                        "building_id": building["building_id"],
                        "pair_id": pair["pair_id"],
                        "limit_state": limit_state,
                        "threshold_midr": threshold,
                        # Log-space midpoint is consistent with lognormal IM.
                        "capacity_im_g": math.sqrt(lower * upper),
                        "censored": 0,
                        "censoring": "none",
                        "lower_im_g": lower,
                        "upper_im_g": upper,
                        "upper_bound_status": upper_result["status"],
                        "upper_bound_certified": 1,
                        "analysis_signature": curve_signature,
                    }
                )
                break
            midpoint = round(math.sqrt(lower * upper), 8)
            if midpoint in results:
                midpoint = round((lower + upper) / 2.0, 8)
            result = analyse_target(midpoint)
            if result is None:
                return scale_guard_failure()
            results[midpoint] = result
            if _threshold_response(result, threshold) is None:
                return {
                    "valid": False,
                    "building_id": building["building_id"],
                    "pair_id": pair["pair_id"],
                    "runs": list(results.values()),
                    "capacities": [],
                    "message": "unresolved numerical failure during filling",
                }

    return {
        "valid": True,
        "building_id": building["building_id"],
        "pair_id": pair["pair_id"],
        "runs": list(results.values()),
        "capacities": capacities,
        "message": "valid CP-bracketed capacities for all limit states",
    }


def _run_hybrid_active_curve(
    building: dict[str, Any],
    pair: dict[str, Any],
    config: dict[str, Any],
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Seek IO/LS/CP capacities with frozen-ML seeds and measured brackets.

    The surrogate never supplies a label.  It proposes only the first IM
    levels.  Every uncensored capacity is the logarithmic midpoint of an
    actual below-threshold and certified above-threshold NLTHA bracket.
    """
    ida_config = config["ida"]
    controller = ida_config["controller"]
    minimum = float(ida_config["minimum_im_g"])
    multiplier = float(ida_config["hunt_multiplier"])
    maximum_hunt_steps = int(ida_config["maximum_hunt_steps"])
    continue_scaling_until_cp = bool(
        ida_config.get("continue_scaling_until_cp", False)
    )
    maximum_unresolved_failure_targets = int(
        ida_config.get("maximum_unresolved_failure_targets", 6)
    )
    tolerance = float(ida_config["bracket_tolerance"])
    minimum_refinements = int(
        controller["minimum_refinements_per_limit_state"]
    )
    transition_lower_response_ratio = float(
        controller.get("transition_lower_response_ratio", 0.90)
    )
    refinement_strategy = str(controller["refinement_strategy"])
    soft_review_point_count = int(
        controller.get("soft_review_point_count", 18)
    )
    thresholds = {
        name: float(value)
        for name, value in ida_config["limit_state_midr"].items()
    }
    curve_signature = ida_curve_analysis_signature(building, pair, config)
    predictions, prediction_diagnostics = predict_active_capacities(
        building, pair, config
    )
    results = _load_curve_checkpoint_results(
        building, pair, config
    )
    failure_guidance_targets = _checkpoint_failure_guidance_targets(
        building,
        pair,
        config,
        thresholds["CP"],
    )
    manual_targets = _manual_request_targets(
        config,
        str(building["building_id"]),
        str(pair["pair_id"]),
    )
    phases: dict[float, str] = {
        target: (
            "codex_or_user_added"
            if target in manual_targets
            else "existing_checkpoint"
        )
        for target in results
    }
    scale_guard_message: str | None = None

    def invalid(message: str) -> dict[str, Any]:
        review = {
            "review_status": "CRITICAL_QC_FAILURE",
            "codex_review_required": 1,
            "review_flags": [
                {
                    "code": "INVALID_ACTIVE_IDA_CURVE",
                    "severity": "critical",
                    "detail": message,
                }
            ],
            "review_flag_count": 1,
            "recommended_additional_targets_g": [],
            "capacity_error_bounds": {},
            "maximum_capacity_error_bound": None,
            "soft_review_point_count": soft_review_point_count,
            "manually_requested_targets_g": sorted(manual_targets),
            "manually_requested_point_count": len(
                manual_targets.intersection(results)
            ),
        }
        return {
            "valid": False,
            "building_id": building["building_id"],
            "pair_id": pair["pair_id"],
            "runs": list(results.values()),
            "capacities": [],
            "message": message,
            "diagnostics": {
                "analysis_signature": curve_signature,
                "controller_mode": "hybrid_active_v1",
                "controller_version": HYBRID_ACTIVE_CONTROLLER_VERSION,
                "refinement_strategy": refinement_strategy,
                "prediction": prediction_diagnostics,
                "predicted_capacities_g": predictions,
                "point_phases": {
                    f"{target:.8g}": phase
                    for target, phase in sorted(phases.items())
                },
                "point_count": len(results),
                **review,
            },
        }

    def analyse_target(
        target_im_g: float,
        phase: str,
    ) -> dict[str, Any] | None:
        nonlocal scale_guard_message
        if not math.isfinite(target_im_g) or target_im_g <= 0.0:
            scale_guard_message = (
                "CP hunt exhausted finite floating-point IM representation"
            )
            return None
        rounded_target = round(max(minimum, target_im_g), 8)
        if rounded_target in results:
            return results[rounded_target]
        try:
            result = run_nltha(
                building,
                pair,
                rounded_target,
                config,
                resume=resume,
            )
        except ScaleFactorGuardError as exc:
            scale_guard_message = str(exc)
            return None
        results[rounded_target] = result
        phases[rounded_target] = phase
        return result

    def failed_analysis_message(stage: str) -> str:
        if scale_guard_message is not None:
            return (
                "CP/capacity bracketing stopped at the mandatory scale-factor "
                f"safety guard during {stage}: {scale_guard_message}"
            )
        return f"Unresolved numerical failure during hybrid active {stage}"

    low_seed = predictions["IO"] * float(
        controller["seed_lower_io_factor"]
    )
    high_seed = predictions["CP"] * float(
        controller["seed_upper_cp_factor"]
    )
    seed_targets = sorted(
        {
            round(max(minimum, low_seed), 8),
            round(max(minimum, predictions["IO"]), 8),
            round(max(minimum, predictions["LS"]), 8),
            round(max(minimum, high_seed), 8),
            *failure_guidance_targets,
        }
    )

    cp_crossed = any(
        _is_certified_upper(result, thresholds["CP"], "CP")
        for result in results.values()
    )
    if not cp_crossed:
        for target in seed_targets:
            result = analyse_target(target, "surrogate_seed")
            if result is None:
                return invalid(failed_analysis_message("seeding"))
            if _is_certified_upper(result, thresholds["CP"], "CP"):
                cp_crossed = True
                break
            if _is_transition_search_guide(result, thresholds["CP"]):
                # Do not waste time at still-higher surrogate seeds. The
                # exhausted point is guidance for a smaller IM only.
                break

    # A successful analysis below CP never stops upward expansion: there is no
    # IM or scale-factor research cap. A full-recovery consensus failure may
    # supply only the upper side of a measured bracket. Generic failures remain
    # guidance and are bounded separately so a broken solver/model cannot loop
    # forever while masquerading as an unusually strong structure.
    hunt_count = 0
    consecutive_unresolved_failures = 0
    while (
        not cp_crossed
        and (
            continue_scaling_until_cp
            or hunt_count < maximum_hunt_steps
        )
    ):
        transition_guides = sorted(
            im
            for im, result in results.items()
            if _is_transition_search_guide(result, thresholds["CP"])
        )
        if transition_guides:
            guide = transition_guides[0]
            lower_candidates = [
                im
                for im, result in results.items()
                if im < guide
                and result.get("status") == "success"
                and float(result["max_midr"]) < thresholds["CP"]
            ]
            if not lower_candidates:
                target = max(minimum, guide / multiplier)
            else:
                target = math.sqrt(max(lower_candidates) * guide)
            phase = "CP_transition_refinement"
        else:
            target = max(results, default=minimum) * multiplier
            phase = "upward_expansion"
        rounded_target = round(max(minimum, target), 8)
        if rounded_target in results:
            if transition_guides and lower_candidates:
                rounded_target = round(
                    (max(lower_candidates) + transition_guides[0]) / 2.0,
                    8,
                )
            if rounded_target in results:
                return invalid(
                    "CP transition refinement exhausted floating-point "
                    "resolution without a successful or dynamic-instability "
                    "upper bracket"
                )
        result = analyse_target(rounded_target, phase)
        if result is None:
            return invalid(failed_analysis_message(phase))
        if _is_certified_upper(result, thresholds["CP"], "CP"):
            cp_crossed = True
            break
        if (
            result.get("status") == "numerical_failure"
            and not _is_transition_search_guide(
                result, thresholds["CP"]
            )
        ):
            consecutive_unresolved_failures += 1
        else:
            consecutive_unresolved_failures = 0
        if (
            consecutive_unresolved_failures
            >= maximum_unresolved_failure_targets
        ):
            return invalid(
                "CP was not observed and repeated full-recovery numerical "
                "failures could not be certified as an instability upper "
                "bound; no capacity value was fabricated"
            )
        hunt_count += 1
    if not cp_crossed:
        return invalid(
            "CP was not observed within the configured bounded test hunt; "
            "no censored capacity label was created"
        )

    capacities = []
    final_widths: dict[str, float | None] = {}
    refinement_count = 0
    for limit_state in LIMIT_STATES:
        threshold = thresholds[limit_state]
        state_refinements = 0
        while True:
            ordered = sorted(results.items())
            below = [
                (im, result)
                for im, result in ordered
                if result["status"] == "success"
                and float(result["max_midr"]) < threshold
            ]
            above = [
                (im, result)
                for im, result in ordered
                if _is_certified_upper(result, threshold, limit_state)
            ]
            if not above:
                transition_guides = [
                    im
                    for im, result in ordered
                    if _is_transition_search_guide(result, threshold)
                ]
                if transition_guides and state_refinements < maximum_hunt_steps:
                    guide = min(transition_guides)
                    lower_candidates = [
                        item[0] for item in below if item[0] < guide
                    ]
                    if lower_candidates:
                        lower = max(lower_candidates)
                        midpoint = round(math.sqrt(lower * guide), 8)
                        if midpoint in results:
                            midpoint = round((lower + guide) / 2.0, 8)
                        if midpoint not in results and lower < midpoint < guide:
                            result = analyse_target(
                                midpoint,
                                f"{limit_state}_transition_refinement",
                            )
                            if result is None:
                                return invalid(
                                    failed_analysis_message(
                                        f"{limit_state} transition refinement"
                                    )
                                )
                            state_refinements += 1
                            refinement_count += 1
                            continue
                return invalid(
                    f"{limit_state} has no successful upper NLTHA bracket"
                    + (
                        " after refining the non-successful transition"
                        if transition_guides
                        else ""
                    )
                )
            upper = min(item[0] for item in above)
            upper_result = results[upper]
            lower_candidates = [item[0] for item in below if item[0] < upper]
            lower = max(lower_candidates, default=0.0)
            if lower <= 0.0:
                downward_target = round(
                    max(minimum, min(results) / multiplier),
                    8,
                )
                if (
                    downward_target < min(results) - 1.0e-12
                    and downward_target not in results
                ):
                    result = analyse_target(
                        downward_target, "downward_expansion"
                    )
                    if result is None:
                        return invalid(
                            failed_analysis_message("downward expansion")
                        )
                    if (
                        _threshold_response(result, threshold) is None
                        and not _is_certified_upper(
                            result, threshold, limit_state
                        )
                    ):
                        return invalid(
                            failed_analysis_message("downward expansion")
                        )
                    continue
                capacities.append(
                    {
                        "building_id": building["building_id"],
                        "pair_id": pair["pair_id"],
                        "limit_state": limit_state,
                        "threshold_midr": threshold,
                        "capacity_im_g": upper,
                        "censored": 1,
                        "censoring": "left",
                        "lower_im_g": None,
                        "upper_im_g": upper,
                        "upper_bound_status": upper_result["status"],
                        "upper_bound_certified": 1,
                        "analysis_signature": curve_signature,
                    }
                )
                final_widths[limit_state] = None
                break

            relative_width = _relative_bracket_width(lower, upper)
            transition_proximity_satisfied = True
            if (
                (
                    upper_result.get("status") == "dynamic_instability"
                    or bool(
                        upper_result.get(
                            "certified_instability_upper_bound", 0
                        )
                    )
                )
                and limit_state in {"IO", "LS"}
            ):
                transition_proximity_satisfied = bool(
                    float(results[lower]["max_midr"])
                    >= transition_lower_response_ratio * threshold
                )
            if (
                state_refinements >= minimum_refinements
                and relative_width <= tolerance
                and transition_proximity_satisfied
            ):
                capacities.append(
                    {
                        "building_id": building["building_id"],
                        "pair_id": pair["pair_id"],
                        "limit_state": limit_state,
                        "threshold_midr": threshold,
                        "capacity_im_g": math.sqrt(lower * upper),
                        "censored": 0,
                        "censoring": "none",
                        "lower_im_g": lower,
                        "upper_im_g": upper,
                        "upper_bound_status": upper_result["status"],
                        "upper_bound_certified": 1,
                        "analysis_signature": curve_signature,
                    }
                )
                final_widths[limit_state] = relative_width
                break
            midpoint = round(math.sqrt(lower * upper), 8)
            if midpoint in results:
                midpoint = round((lower + upper) / 2.0, 8)
            if midpoint in results or not lower < midpoint < upper:
                return invalid(
                    f"{limit_state} bracket refinement exhausted floating-"
                    "point resolution before reaching the requested accuracy"
                )
            result = analyse_target(midpoint, f"{limit_state}_refinement")
            if result is None:
                return invalid(
                    failed_analysis_message(f"{limit_state} refinement")
                )
            if (
                _threshold_response(result, threshold) is None
                and not _is_certified_upper(
                    result, threshold, limit_state
                )
            ):
                return invalid(
                    failed_analysis_message(f"{limit_state} refinement")
                )
            state_refinements += 1
            refinement_count += 1

    prediction_source = str(
        prediction_diagnostics.get("prediction_source", "unknown")
    )
    review = _curve_review_diagnostics(
        results=results,
        capacities=capacities,
        tolerance=tolerance,
        soft_review_point_count=soft_review_point_count,
        manual_targets=manual_targets,
    )
    diagnostics = {
        "analysis_signature": curve_signature,
        "controller_mode": "hybrid_active_v1",
        "controller_version": HYBRID_ACTIVE_CONTROLLER_VERSION,
        "refinement_strategy": refinement_strategy,
        "prediction_source": prediction_source,
        "prediction": prediction_diagnostics,
        "predicted_capacities_g": predictions,
        "point_phases": {
            f"{target:.8g}": phase
            for target, phase in sorted(phases.items())
        },
        "point_count": len(results),
        "seed_point_count": sum(
            phase == "surrogate_seed" for phase in phases.values()
        ),
        "expansion_point_count": sum(
            "expansion" in phase for phase in phases.values()
        ),
        "refinement_point_count": refinement_count,
        "final_bracket_widths": final_widths,
        "maximum_final_bracket_width": max(
            (value for value in final_widths.values() if value is not None),
            default=None,
        ),
        **review,
    }
    valid = review["review_status"] != "CRITICAL_QC_FAILURE"
    return {
        "valid": valid,
        "building_id": building["building_id"],
        "pair_id": pair["pair_id"],
        "runs": list(results.values()),
        "capacities": capacities,
        "message": (
            "valid accuracy-driven hybrid-active CP-bracketed capacities "
            "for all limit states"
            if valid
            else "hybrid-active curve failed critical post-analysis QC"
        ),
        "diagnostics": diagnostics,
    }


def run_ida_curve(
    building: dict[str, Any],
    pair: dict[str, Any],
    config: dict[str, Any],
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Dispatch one curve to the frozen legacy or hybrid active controller."""
    if _is_pwsa_sf1_sensitivity_pair(pair):
        native_im_g = pair_sa_geomean_g(pair, float(building["t1_s"]))
        result = run_nltha(
            building,
            pair,
            native_im_g,
            config,
            resume=resume,
        )
        if not math.isclose(
            float(result["scale_factor"]),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise RuntimeError(
                f"{pair['pair_id']}: event-specific PWSA response violated "
                "the locked as-recorded scale-factor policy"
            )
        valid = result["status"] in {"success", "dynamic_instability"}
        signature = ida_curve_analysis_signature(building, pair, config)
        return {
            "valid": valid,
            "building_id": building["building_id"],
            "pair_id": pair["pair_id"],
            "runs": [result],
            "capacities": [],
            "message": (
                "valid PWSA as-recorded SF=1 event-specific sensitivity"
                if valid
                else "PWSA SF=1 sensitivity ended in unresolved numerical "
                "failure"
            ),
            "diagnostics": {
                "controller_mode": "event_specific_sf1",
                "controller_version": "pwsa-sensitivity-v1",
                "prediction_source": "not_applicable",
                "point_count": 1,
                "seed_point_count": 1,
                "expansion_point_count": 0,
                "refinement_point_count": 0,
                "review_status": (
                    "PASS" if valid else "CRITICAL_QC_FAILURE"
                ),
                "review_flag_count": 0 if valid else 1,
                "codex_review_required": int(not valid),
                "analysis_role": "event_specific_sensitivity",
                "scale_factor_policy": "as_recorded_sf1",
                "included_in_primary_fragility": False,
                "native_im_g": native_im_g,
                "analysis_signature": signature,
            },
        }
    mode = str(
        config["ida"].get("controller", {}).get(
            "mode", "legacy_hunt_fill"
        )
    )
    if mode == "legacy_hunt_fill":
        return _run_legacy_hunt_fill_curve(
            building, pair, config, resume=resume
        )
    if mode == "hybrid_active_v1":
        return _run_hybrid_active_curve(
            building, pair, config, resume=resume
        )
    raise ValueError(f"Unsupported IDA controller mode: {mode}")


def _ida_worker(payload: tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]) -> dict[str, Any]:
    building, pair, config, resume = payload
    return run_ida_curve(building, pair, config, resume=resume)


def _ida_payload_cost_proxy(
    payload: tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool],
) -> float:
    """Estimate relative curve cost without using any response label.

    Ground-motion sample count dominates NLTHA time. The member-count proxy
    accounts for the larger 3D model of multi-bay frames. This value changes
    scheduling only; it never changes IM selection or a structural result.
    """
    building, pair, _config, _resume = payload
    sample_count = pair.get("npts")
    if sample_count is None:
        sample_count = (
            float(pair["duration_s"]) / float(pair["dt_s"]) + 1.0
        )
    bays = max(int(building.get("number_of_bays", 1)), 1)
    members_per_storey_proxy = (bays + 1) * (3 * bays + 1)
    return float(sample_count) * float(members_per_storey_proxy)


def _schedule_ida_payloads(
    payloads: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]
    ],
) -> list[
    tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]
]:
    """Prioritize building completion without ever reserving idle workers.

    Building queue rank is the primary key.  Curves within one building are
    ordered by decreasing estimated cost so its PWSA/long records start early
    and are less likely to become a late straggler.  Once every pending curve
    of the leading building has been submitted, the next available worker
    immediately starts the next building; it never waits for the previous
    building's final active curve to finish.
    """
    return sorted(
        payloads,
        key=lambda payload: (
            payload[0].get("queue_rank") is None,
            int(payload[0].get("queue_rank") or 0),
            str(payload[0]["building_id"]),
            -_ida_payload_cost_proxy(payload),
            str(payload[1]["pair_id"]),
        ),
    )


def _ida_payload_key(
    payload: tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool],
) -> tuple[str, str]:
    return (
        str(payload[0]["building_id"]),
        str(payload[1]["pair_id"]),
    )


def _has_current_spo_signature(
    building: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    """Return whether an IDA candidate has a current, reproducible SPO row."""
    from .spo import _spo_analysis_signature

    return (
        building.get("spo_analysis_signature")
        == _spo_analysis_signature(building, config)
    )


def _database_inputs(
    config: dict[str, Any],
    *,
    building_ids: list[str] | None,
    limit: int | None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    with connect(config["database_path"]) as connection:
        parameters: list[Any] = []
        where = "b.valid=1 AND b.selected=1 AND s.valid=1"
        if building_ids:
            marks = ",".join("?" for _ in building_ids)
            where += f" AND b.building_id IN ({marks})"
            parameters.extend(building_ids)
        sql = (
            "SELECT b.*, s.t1_s, s.vy_kn, s.dy_m, s.vc_kn, s.dc_m, "
            "s.vu_kn, s.du_m, s.x_mode_effective_mass_ratio, "
            "s.analysis_signature AS spo_analysis_signature "
            "FROM building_catalog b "
            "JOIN spo_features s USING(building_id) "
            f"WHERE {where} ORDER BY b.queue_rank"
        )
        buildings = [
            dict(row) for row in connection.execute(sql, parameters)
        ]
    # Filter provenance before applying a limit. Otherwise stale rows early in
    # queue order can consume the SQL LIMIT and produce an empty or undersized
    # IDA batch after the ground-motion selector rejects their SPO signatures.
    buildings = [
        building
        for building in buildings
        if _has_current_spo_signature(building, config)
    ]
    if not buildings:
        return buildings, {}
    selected_building_ids = [
        str(building["building_id"]) for building in buildings
    ]
    build_ground_motion_selection(
        config, building_ids=selected_building_ids
    )
    marks = ",".join("?" for _ in selected_building_ids)
    with connect(config["database_path"]) as connection:
        selected_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT s.building_id, s.analysis_role, "
                "s.scale_factor_policy, s.selection_reason, g.* "
                "FROM building_ground_motion_selection s "
                "JOIN ground_motion_catalog g USING(pair_id) "
                f"WHERE s.building_id IN ({marks}) AND g.valid=1 "
                "ORDER BY s.building_id, g.source_set, g.pair_id",
                selected_building_ids,
            )
        ]
    pairs_by_building: dict[str, list[dict[str, Any]]] = {
        building_id: [] for building_id in selected_building_ids
    }
    for row in selected_rows:
        building_id = str(row.pop("building_id"))
        pairs_by_building[building_id].append(row)
    if limit is not None:
        if limit < 1:
            raise ValueError("IDA batch limit must be at least 1")
        completed = _current_complete_ida_building_ids(
            config,
            buildings=buildings,
            pairs_by_building=pairs_by_building,
        )
        # A bounded production invocation advances through the queue.  A
        # partially completed building remains pending, so every point already
        # checkpointed by run_ida_curve is resumed before a later building is
        # admitted to the batch.
        buildings = [
            building
            for building in buildings
            if str(building["building_id"]) not in completed
        ][:limit]
        pairs_by_building = {
            str(building["building_id"]): pairs_by_building[
                str(building["building_id"])
            ]
            for building in buildings
        }
    return buildings, pairs_by_building


def _current_complete_ida_building_ids(
    config: dict[str, Any],
    *,
    buildings: list[dict[str, Any]],
    pairs_by_building: dict[str, list[dict[str, Any]]],
) -> set[str]:
    """Return buildings with all current IO/LS/CP record capacities.

    Completion is content-addressed: every selected record must contain all
    limit states with the exact current curve signature.  Stale, partial, or
    failed curves therefore remain in the next bounded batch.
    """
    if not buildings:
        return set()
    building_ids = [str(item["building_id"]) for item in buildings]
    marks = ",".join("?" for _ in building_ids)
    with connect(config["database_path"]) as connection:
        capacity_rows = [
            dict(row)
            for row in connection.execute(
                f"""
                SELECT building_id, pair_id, limit_state, analysis_signature
                FROM ida_capacities
                WHERE building_id IN ({marks})
                """,
                building_ids,
            )
        ]
    primary_complete = _complete_ida_building_ids_from_capacity_rows(
        config,
        buildings=buildings,
        pairs_by_building=pairs_by_building,
        capacity_rows=capacity_rows,
    )
    return {
        str(building["building_id"])
        for building in buildings
        if str(building["building_id"]) in primary_complete
        and all(
            _current_pair_is_complete(
                config,
                building=building,
                pair=pair,
            )
            for pair in pairs_by_building.get(
                str(building["building_id"]), []
            )
        )
    }


def _complete_ida_building_ids_from_capacity_rows(
    config: dict[str, Any],
    *,
    buildings: list[dict[str, Any]],
    pairs_by_building: dict[str, list[dict[str, Any]]],
    capacity_rows: list[dict[str, Any]],
) -> set[str]:
    """Pure completion check used by the queue and regression tests."""
    stored = {
        (
            str(row["building_id"]),
            str(row["pair_id"]),
            str(row["limit_state"]),
        ): row.get("analysis_signature")
        for row in capacity_rows
    }
    complete: set[str] = set()
    for building in buildings:
        building_id = str(building["building_id"])
        pairs = [
            pair
            for pair in pairs_by_building.get(building_id, [])
            if _is_primary_fragility_pair(pair)
        ]
        if not pairs:
            continue
        is_complete = True
        for pair in pairs:
            expected_signature = ida_curve_analysis_signature(
                building,
                pair,
                config,
            )
            if any(
                stored.get(
                    (building_id, str(pair["pair_id"]), limit_state)
                )
                != expected_signature
                for limit_state in LIMIT_STATES
            ):
                is_complete = False
                break
        if is_complete:
            complete.add(building_id)
    return complete


def _current_pair_is_complete(
    config: dict[str, Any],
    *,
    building: dict[str, Any],
    pair: dict[str, Any],
) -> bool:
    """Return whether a primary curve or SF=1 sensitivity is current."""
    if _is_pwsa_sf1_sensitivity_pair(pair):
        native_im_g = pair_sa_geomean_g(pair, float(building["t1_s"]))
        expected_signature = nltha_analysis_signature(
            building,
            pair,
            native_im_g,
            config,
        )
        with connect(config["database_path"]) as connection:
            row = connection.execute(
                """
                SELECT target_im_g, scale_factor, status,
                       analysis_signature
                FROM ida_runs
                WHERE building_id=? AND pair_id=?
                ORDER BY ABS(target_im_g-?) LIMIT 1
                """,
                (
                    str(building["building_id"]),
                    str(pair["pair_id"]),
                    native_im_g,
                ),
            ).fetchone()
        return bool(
            row is not None
            and math.isclose(
                float(row["target_im_g"]),
                native_im_g,
                rel_tol=0.0,
                abs_tol=1.0e-8,
            )
            and math.isclose(
                float(row["scale_factor"]),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
            and str(row["status"]) in {"success", "dynamic_instability"}
            and row["analysis_signature"] == expected_signature
        )
    expected_signature = ida_curve_analysis_signature(
        building,
        pair,
        config,
    )
    with connect(config["database_path"]) as connection:
        rows = connection.execute(
            """
            SELECT limit_state, analysis_signature
            FROM ida_capacities
            WHERE building_id=? AND pair_id=?
            """,
            (str(building["building_id"]), str(pair["pair_id"])),
        ).fetchall()
    current_states = {
        str(row["limit_state"])
        for row in rows
        if row["analysis_signature"] == expected_signature
    }
    return current_states == set(LIMIT_STATES)


def _persist_curve(config: dict[str, Any], curve: dict[str, Any]) -> None:
    run_columns = {
        "building_id",
        "pair_id",
        "target_im_g",
        "scale_factor",
        "achieved_im_g",
        "max_midr",
        "status",
        "dynamic_instability",
        "runtime_s",
        "recovery_level",
        "soft_im_warning_exceeded",
        "scale_factor_warning_exceeded",
        "analysis_signature",
        "recovery_policy_version",
        "result_path",
    }
    run_rows = [
        {key: result.get(key) for key in run_columns}
        for result in curve["runs"]
    ]
    with transaction(config["database_path"]) as connection:
        # A rerun is authoritative for this curve. Remove stale points and
        # capacities that could otherwise survive after a changed config or
        # a newly invalid numerical result.
        connection.execute(
            "DELETE FROM ida_runs WHERE building_id=? AND pair_id=?",
            (curve["building_id"], curve["pair_id"]),
        )
        connection.execute(
            "DELETE FROM ida_capacities WHERE building_id=? AND pair_id=?",
            (curve["building_id"], curve["pair_id"]),
        )
        connection.execute(
            "DELETE FROM ida_curve_diagnostics "
            "WHERE building_id=? AND pair_id=?",
            (curve["building_id"], curve["pair_id"]),
        )
        upsert_many(
            connection,
            "ida_runs",
            run_rows,
            ("building_id", "pair_id", "target_im_g"),
        )
        if curve["valid"]:
            upsert_many(
                connection,
                "ida_capacities",
                curve["capacities"],
                ("building_id", "pair_id", "limit_state"),
            )
        diagnostics = curve.get("diagnostics")
        if diagnostics:
            predicted = diagnostics.get("predicted_capacities_g", {})
            prediction = diagnostics.get("prediction", {})
            upsert_many(
                connection,
                "ida_curve_diagnostics",
                [
                    {
                        "building_id": curve["building_id"],
                        "pair_id": curve["pair_id"],
                        "controller_mode": diagnostics.get(
                            "controller_mode", "unknown"
                        ),
                        "controller_version": diagnostics.get(
                            "controller_version", "unknown"
                        ),
                        "prediction_source": diagnostics.get(
                            "prediction_source",
                            prediction.get("prediction_source", "unknown"),
                        ),
                        "predicted_io_g": predicted.get("IO"),
                        "predicted_ls_g": predicted.get("LS"),
                        "predicted_cp_g": predicted.get("CP"),
                        "point_count": int(
                            diagnostics.get(
                                "point_count", len(curve.get("runs", []))
                            )
                        ),
                        "seed_point_count": int(
                            diagnostics.get("seed_point_count", 0)
                        ),
                        "expansion_point_count": int(
                            diagnostics.get("expansion_point_count", 0)
                        ),
                        "refinement_point_count": int(
                            diagnostics.get("refinement_point_count", 0)
                        ),
                        "maximum_final_bracket_width": diagnostics.get(
                            "maximum_final_bracket_width"
                        ),
                        "maximum_capacity_error_bound": diagnostics.get(
                            "maximum_capacity_error_bound"
                        ),
                        "review_status": diagnostics.get("review_status"),
                        "review_flag_count": int(
                            diagnostics.get("review_flag_count", 0)
                        ),
                        "codex_review_required": int(
                            diagnostics.get("codex_review_required", 0)
                        ),
                        "manually_requested_point_count": int(
                            diagnostics.get(
                                "manually_requested_point_count", 0
                            )
                        ),
                        "recommended_targets_json": json.dumps(
                            diagnostics.get(
                                "recommended_additional_targets_g", []
                            ),
                            sort_keys=True,
                        ),
                        "model_path": prediction.get("model_path"),
                        "model_sha256": prediction.get("model_sha256"),
                        "details_json": json.dumps(
                            diagnostics, sort_keys=True
                        ),
                        "analysis_signature": diagnostics[
                            "analysis_signature"
                        ],
                    }
                ],
                ("building_id", "pair_id"),
            )


def add_ida_points(
    config: dict[str, Any],
    *,
    building_id: str,
    pair_id: str,
    target_ims_g: list[float],
    requester: str = "User/Codex",
    reason: str = "Additional IDA accuracy/review point",
) -> dict[str, Any]:
    """Add only requested IM targets and rebuild the affected signed results.

    Existing fixed-IM JSON checkpoints are content-addressed and loaded into
    the rebuilt curve. Consequently this operation never repeats a current
    NLTHA point and never reruns another building or record pair.
    """
    initialize(config["database_path"])
    targets = sorted({round(float(value), 8) for value in target_ims_g})
    if not targets or any(
        not math.isfinite(value) or value <= 0.0 for value in targets
    ):
        raise ValueError("At least one positive finite target IM is required")
    with connect(config["database_path"]) as connection:
        building_row = connection.execute(
            """
            SELECT b.*, s.t1_s, s.vy_kn, s.dy_m, s.vc_kn, s.dc_m,
                   s.vu_kn, s.du_m,
                   s.x_mode_effective_mass_ratio,
                   s.analysis_signature AS spo_analysis_signature
            FROM building_catalog b
            JOIN spo_features s USING(building_id)
            WHERE b.building_id=? AND b.selected=1
              AND b.valid=1 AND s.valid=1
            """,
            (building_id,),
        ).fetchone()
        pair_row = connection.execute(
            """
            SELECT s.analysis_role, s.scale_factor_policy,
                   s.selection_reason, g.*
            FROM building_ground_motion_selection s
            JOIN ground_motion_catalog g USING(pair_id)
            WHERE s.building_id=? AND s.pair_id=? AND g.valid=1
            """,
            (building_id, pair_id),
        ).fetchone()
    if building_row is None:
        raise ValueError(
            f"{building_id}: no selected current valid SPO building"
        )
    if pair_row is None:
        raise ValueError(
            f"{building_id}/{pair_id}: pair is not currently selected"
        )
    building = dict(building_row)
    pair = dict(pair_row)
    if not _has_current_spo_signature(building, config):
        raise ValueError(f"{building_id}: SPO signature is stale")
    if _is_pwsa_sf1_sensitivity_pair(pair):
        raise ValueError(
            f"{building_id}/{pair_id}: PWSA is locked to one as-recorded "
            "SF=1 sensitivity point and cannot accept manual IDA targets"
        )

    requested_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with transaction(config["database_path"]) as connection:
        for target in targets:
            upsert_many(
                connection,
                "ida_manual_point_requests",
                [
                    {
                        "building_id": building_id,
                        "pair_id": pair_id,
                        "target_im_g": target,
                        "requester": requester,
                        "reason": reason,
                        "requested_utc": requested_utc,
                        "status": "pending",
                        "result_status": None,
                        "result_path": None,
                        "completed_utc": None,
                    }
                ],
                ("building_id", "pair_id", "target_im_g"),
            )

    point_results = []
    for target in targets:
        path = _result_path(config, building_id, pair_id, target)
        expected_signature = nltha_analysis_signature(
            building, pair, target, config
        )
        reused = (
            _load_existing_result(
                path,
                expected_signature,
                certified_threshold=float(
                    config["ida"]["limit_state_midr"]["CP"]
                ),
            )
            is not None
        )
        try:
            result = run_nltha(
                building, pair, target, config, resume=True
            )
        except Exception:
            with transaction(config["database_path"]) as connection:
                connection.execute(
                    """
                    UPDATE ida_manual_point_requests
                    SET status='failed',
                        completed_utc=?
                    WHERE building_id=? AND pair_id=? AND target_im_g=?
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                        building_id,
                        pair_id,
                        target,
                    ),
                )
            raise
        with transaction(config["database_path"]) as connection:
            connection.execute(
                """
                UPDATE ida_manual_point_requests
                SET status='completed', result_status=?, result_path=?,
                    completed_utc=?
                WHERE building_id=? AND pair_id=? AND target_im_g=?
                """,
                (
                    result["status"],
                    result["result_path"],
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    building_id,
                    pair_id,
                    target,
                ),
            )
        point_results.append(
            {
                "target_im_g": target,
                "status": result["status"],
                "max_midr": result["max_midr"],
                "reused_existing_checkpoint": reused,
                "result_path": result["result_path"],
            }
        )

    curve = _run_hybrid_active_curve(
        building, pair, config, resume=True
    )
    _persist_curve(config, curve)

    buildings, pairs_by_building = _database_inputs(
        config, building_ids=[building_id], limit=None
    )
    building_complete = bool(
        buildings
        and building_id
        in _current_complete_ida_building_ids(
            config,
            buildings=buildings,
            pairs_by_building=pairs_by_building,
        )
    )
    fragility_refit: dict[str, Any] = {
        "deferred": True,
        "reason": "building does not yet have all selected IDA curves",
    }
    if building_complete and curve["valid"]:
        from .fragility import fit_all_fragilities

        fragility_refit = fit_all_fragilities(
            config,
            building_ids=[building_id],
        )
    return {
        "building_id": building_id,
        "pair_id": pair_id,
        "requested_points": point_results,
        "curve_valid": bool(curve["valid"]),
        "curve_message": curve["message"],
        "curve_point_count": len(curve["runs"]),
        "curve_capacities": curve["capacities"],
        "curve_review": curve.get("diagnostics", {}),
        "building_fragility_refit": fragility_refit,
        "existing_points_reused": True,
        "other_curves_rerun": False,
    }


def run_ida_batch(
    config: dict[str, Any],
    *,
    building_ids: list[str] | None = None,
    pair_ids: list[str] | None = None,
    limit: int | None = None,
    workers: int | None = None,
    resume: bool = True,
    allow_incomplete_records: bool = False,
) -> dict[str, Any]:
    initialize(config["database_path"])
    buildings, pairs_by_building = _database_inputs(
        config, building_ids=building_ids, limit=limit
    )
    if not buildings:
        if limit is not None:
            return {
                "building_count": 0,
                "curve_count": 0,
                "valid_curve_count": 0,
                "invalid_curve_count": 0,
                "failure_count": 0,
                "failures": [],
                "batch_complete": True,
                "message": (
                    "No pending building remains in the bounded IDA queue; "
                    "all current selected record curves are complete."
                ),
            }
        raise RuntimeError("No valid SPO buildings are available for IDA")
    if pair_ids:
        requested_pair_ids = {str(pair_id) for pair_id in pair_ids}
        available_pair_ids = {
            str(pair["pair_id"])
            for pairs in pairs_by_building.values()
            for pair in pairs
        }
        unknown_pair_ids = sorted(requested_pair_ids - available_pair_ids)
        if unknown_pair_ids:
            raise ValueError(
                "Requested IDA pair IDs are not in the current building "
                "selection: " + ", ".join(unknown_pair_ids)
            )
        pairs_by_building = {
            building_id: [
                pair
                for pair in pairs
                if str(pair["pair_id"]) in requested_pair_ids
            ]
            for building_id, pairs in pairs_by_building.items()
        }
        buildings = [
            building
            for building in buildings
            if pairs_by_building[str(building["building_id"])]
        ]
    selected_pairs = [
        pair
        for pairs in pairs_by_building.values()
        for pair in pairs
    ]
    has_pwsa = any(
        _is_pwsa_sf1_sensitivity_pair(pair) for pair in selected_pairs
    )
    require_pwsa = bool(
        config["ground_motion"].get(
            "require_pwsa_sensitivity_for_full_batch", True
        )
    )
    missing_pwsa_buildings = [
        str(building["building_id"])
        for building in buildings
        if not any(
            _is_pwsa_sf1_sensitivity_pair(pair)
            for pair in pairs_by_building[str(building["building_id"])]
        )
    ]
    if (
        require_pwsa
        and missing_pwsa_buildings
        and not allow_incomplete_records
    ):
        raise RuntimeError(
            "Production response generation is gated: one PWSA 2568 "
            "as-recorded SF=1 sensitivity pair is required for every "
            "building; missing for "
            + ", ".join(missing_pwsa_buildings)
            + ". Use --allow-incomplete-records only for an explicitly "
            "labelled CMS smoke run."
        )
    duplicate_physical_pair_buildings = []
    for building in buildings:
        building_id = str(building["building_id"])
        identities = []
        for pair in pairs_by_building[building_id]:
            component_hashes = (
                pair.get("sha256_x"),
                pair.get("sha256_y"),
            )
            identity = pair.get("physical_pair_hash")
            if not identity and all(component_hashes):
                identity = f"{component_hashes[0]}:{component_hashes[1]}"
            identities.append(str(identity or pair["pair_id"]))
        if len(identities) != len(set(identities)):
            duplicate_physical_pair_buildings.append(building_id)
    if duplicate_physical_pair_buildings and not allow_incomplete_records:
        raise RuntimeError(
            "Full IDA is gated: duplicated physical X-Y record pairs were "
            "selected within Building IDs "
            + ", ".join(duplicate_physical_pair_buildings)
        )
    pair_counts = [
        sum(
            _is_primary_fragility_pair(pair)
            for pair in pairs_by_building[str(building["building_id"])]
        )
        for building in buildings
    ]
    minimum_pairs = int(
        config["ida"].get("minimum_pairs_for_fragility", 4)
    )
    if (
        any(count < minimum_pairs for count in pair_counts)
        and not allow_incomplete_records
    ):
        raise RuntimeError(
            f"Each building requires at least {minimum_pairs} selected "
            f"ground-motion pairs; observed range="
            f"{min(pair_counts)}-{max(pair_counts)}"
        )

    requested_payloads = [
        (building, pair, config, resume)
        for building in buildings
        for pair in pairs_by_building[str(building["building_id"])]
    ]
    payloads = [
        payload
        for payload in requested_payloads
        if not _current_pair_is_complete(
            config,
            building=payload[0],
            pair=payload[1],
        )
    ]
    already_complete_curve_count = len(requested_payloads) - len(payloads)
    worker_count = int(workers or config["ida"]["workers"])
    native_worker_restart_limit = int(
        config["ida"].get("native_worker_restart_limit", 2)
    )
    curves: list[dict[str, Any]] = []
    failures = []
    pool_break_count = 0
    native_worker_restart_count = 0
    native_worker_restart_exhausted_count = 0
    isolated_process_launch_count = 0

    def persist_outcome(
        curve: dict[str, Any],
        *,
        building_id: str,
        pair_id: str,
    ) -> None:
        _persist_curve(config, curve)
        curves.append(curve)
        if curve["valid"]:
            resolve_pipeline_failures(
                config["database_path"],
                stage="IDA",
                building_id=building_id,
                pair_id=pair_id,
            )
        else:
            record_pipeline_failure(
                config["database_path"],
                stage="IDA",
                error=curve["message"],
                building_id=building_id,
                pair_id=pair_id,
                details={"run_count": len(curve["runs"])},
            )

    scheduled_payloads = _schedule_ida_payloads(payloads)
    pending = deque((payload, 0) for payload in scheduled_payloads)
    active: dict[
        concurrent.futures.Future,
        tuple[
            tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool],
            concurrent.futures.ProcessPoolExecutor,
            int,
        ],
    ] = {}
    while pending or active:
        while pending and len(active) < worker_count:
            payload, restart_index = pending.popleft()
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
            future = executor.submit(_ida_worker, payload)
            active[future] = (payload, executor, restart_index)
            isolated_process_launch_count += 1
        if not active:
            break
        completed, _ = concurrent.futures.wait(
            active,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        for future in completed:
            payload, executor, restart_index = active.pop(future)
            building_id, pair_id = _ida_payload_key(payload)
            worker_broken = False
            try:
                curve = future.result()
                persist_outcome(
                    curve,
                    building_id=building_id,
                    pair_id=pair_id,
                )
            except BrokenProcessPool as exc:
                worker_broken = True
                pool_break_count += 1
                record_pipeline_failure(
                    config["database_path"],
                    stage="IDA",
                    error=exc,
                    building_id=building_id,
                    pair_id=pair_id,
                    details={
                        "rolling_isolated_worker": True,
                        "isolated_native_restart": restart_index,
                        "isolated_native_restart_limit": (
                            native_worker_restart_limit
                        ),
                    },
                )
                if restart_index < native_worker_restart_limit:
                    native_worker_restart_count += 1
                    pending.appendleft((payload, restart_index + 1))
                else:
                    native_worker_restart_exhausted_count += 1
                    message = (
                        "Native OpenSees worker terminated after "
                        f"{native_worker_restart_limit} isolated fresh-process "
                        f"restarts: {exc}"
                    )
                    failures.append(
                        {
                            "building_id": building_id,
                            "pair_id": pair_id,
                            "message": message,
                            "isolated_native_restart_exhausted": True,
                        }
                    )
            except Exception as exc:
                record_pipeline_failure(
                    config["database_path"],
                    stage="IDA",
                    error=exc,
                    building_id=building_id,
                    pair_id=pair_id,
                    details={
                        "rolling_isolated_worker": True,
                        "isolated_native_restart": restart_index,
                    },
                )
                failures.append(
                    {
                        "building_id": building_id,
                        "pair_id": pair_id,
                        "message": str(exc),
                        "isolated_native_restart": restart_index,
                    }
                )
            finally:
                if worker_broken:
                    future.cancel()
                    executor.shutdown(
                        wait=False,
                        cancel_futures=True,
                    )
                else:
                    executor.shutdown(wait=True)
    valid_curves = sum(bool(curve["valid"]) for curve in curves)
    return {
        "building_count": len(buildings),
        "unique_pair_count": len(
            {pair["pair_id"] for pair in selected_pairs}
        ),
        "pair_count_range_per_building": [
            min(pair_counts),
            max(pair_counts),
        ],
        "curve_count": len(curves),
        "requested_curve_count": len(requested_payloads),
        "already_complete_curve_count": already_complete_curve_count,
        "valid_curve_count": valid_curves,
        "invalid_curve_count": len(curves) - valid_curves,
        "worker_count": worker_count,
        "cms_only": not has_pwsa,
        "failure_count": len(failures),
        "pool_break_count": pool_break_count,
        "native_worker_restart_count": native_worker_restart_count,
        "native_worker_restart_exhausted_count": (
            native_worker_restart_exhausted_count
        ),
        "native_worker_restart_limit": native_worker_restart_limit,
        "payload_scheduling_mode": (
            "building_priority_cost_descending_rolling_isolated_v2"
        ),
        "one_curve_per_native_process": True,
        "maximum_concurrent_native_processes": worker_count,
        "isolated_process_launch_count": isolated_process_launch_count,
        "failures": failures,
        "research_phase": "data_generation_only_ml_deferred",
        "bounded_batch_advances_pending_queue": bool(limit is not None),
        "batch_complete": bool(
            not failures
            and valid_curves + already_complete_curve_count
            == len(requested_payloads)
        ),
    }


def benchmark_ida(
    config: dict[str, Any],
    *,
    resume: bool = True,
    workers: int | None = None,
) -> dict[str, Any]:
    """Benchmark SCWB classes against short/medium/long records."""
    buildings, pairs_by_building = _database_inputs(
        config, building_ids=None, limit=None
    )
    if len(buildings) < 2:
        raise RuntimeError("Benchmark requires at least two SPO buildings and three pairs")
    strength_key = lambda item: (
        float(item["beam_phi_mn_knm"])
        + float(item["column_phi_mn_knm"])
    )
    buildings_by_strength = sorted(
        buildings,
        key=strength_key,
    )
    required_scwb_classes = SCWB_RESEARCH_CLASSES
    by_scwb_class = {
        class_name: sorted(
            [
                building
                for building in buildings
                if building.get("scwb_class") == class_name
            ],
            key=strength_key,
        )
        for class_name in required_scwb_classes
    }
    if all(by_scwb_class.values()):
        # Cover the full accepted research strength-margin range without
        # implying code compliance.
        selected_buildings = [
            by_scwb_class[SCWB_RESEARCH_CLASSES[0]][0],
            by_scwb_class[SCWB_RESEARCH_CLASSES[1]][
                len(by_scwb_class[SCWB_RESEARCH_CLASSES[1]]) // 2
            ],
            by_scwb_class[SCWB_RESEARCH_CLASSES[2]][-1],
        ]
    else:
        # Availability-aware fallback: retain three strength quantiles when
        # the current archetype does not populate every diagnostic band.
        indices = sorted(
            {
                0,
                len(buildings_by_strength) // 2,
                len(buildings_by_strength) - 1,
            }
        )
        selected_buildings = [
            buildings_by_strength[index] for index in indices
        ]
    curves: list[dict[str, Any]] = []
    selected_pair_ids: dict[str, list[str]] = {}
    benchmark_payloads: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]
    ] = []
    for building in selected_buildings:
        eligible_pairs = pairs_by_building[str(building["building_id"])]
        if len(eligible_pairs) < 3:
            raise RuntimeError(
                f"{building['building_id']} has fewer than three selected pairs"
            )
        pairs_by_duration = sorted(
            eligible_pairs,
            key=lambda item: float(item["duration_s"]),
        )
        selected_pairs = [
            pairs_by_duration[0],
            pairs_by_duration[len(pairs_by_duration) // 2],
            pairs_by_duration[-1],
        ]
        selected_pair_ids[str(building["building_id"])] = [
            str(pair["pair_id"]) for pair in selected_pairs
        ]
        for pair in selected_pairs:
            benchmark_payloads.append((building, pair, config, resume))
    # Launch the three longest records first so each SCWB band's PWSA curve
    # occupies one worker immediately. This minimizes benchmark wall time
    # without changing any curve, target IM, or measured worker runtime.
    scwb_order = {
        class_name: index
        for index, class_name in enumerate(SCWB_RESEARCH_CLASSES)
    }
    benchmark_payloads.sort(
        key=lambda payload: (
            -float(payload[1]["duration_s"]),
            scwb_order.get(
                str(payload[0].get("scwb_class")),
                len(scwb_order),
            ),
            str(payload[0]["building_id"]),
            str(payload[1]["pair_id"]),
        )
    )
    benchmark_workers = int(
        workers
        if workers is not None
        else min(3, int(config["ida"]["workers"]))
    )
    if benchmark_workers <= 0:
        raise ValueError("ida.benchmark_workers must be positive")
    benchmark_workers = min(
        benchmark_workers,
        len(benchmark_payloads),
    )
    if benchmark_workers == 1:
        for payload in benchmark_payloads:
            curve = _ida_worker(payload)
            _persist_curve(config, curve)
            curves.append(curve)
    else:
        # One independent process per curve avoids shared OpenSees state.
        # Limiting the benchmark to three concurrent curves covers the three
        # SCWB bands while leaving headroom on the 8-core workstation. The
        # measured per-run runtimes therefore include realistic batch
        # contention rather than an artificially serial execution.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=benchmark_workers
        ) as executor:
            futures = [
                executor.submit(_ida_worker, payload)
                for payload in benchmark_payloads
            ]
            for future in concurrent.futures.as_completed(futures):
                curve = future.result()
                _persist_curve(config, curve)
                curves.append(curve)
    curve_order = {
        (
            str(payload[0]["building_id"]),
            str(payload[1]["pair_id"]),
        ): index
        for index, payload in enumerate(benchmark_payloads)
    }
    curves.sort(
        key=lambda curve: curve_order[
            (
                str(curve["building_id"]),
                str(curve["pair_id"]),
            )
        ]
    )
    invalid_curves = [
        {
            "building_id": curve["building_id"],
            "pair_id": curve["pair_id"],
            "message": curve["message"],
        }
        for curve in curves
        if not curve["valid"]
    ]
    runtimes = [
        float(run["runtime_s"])
        for curve in curves
        for run in curve["runs"]
        if run["status"] in {"success", "dynamic_instability"}
    ]
    run_counts = [len(curve["runs"]) for curve in curves]
    if not runtimes:
        raise RuntimeError("Benchmark produced no valid NLTHA runtimes")
    t90 = float(np.percentile(runtimes, 90))
    r_estimate = int(math.ceil(float(np.percentile(run_counts, 90))))
    ida_config = config["ida"]
    numerator = (
        float(ida_config["compute_days"])
        * 86400.0
        * int(ida_config["workers"])
    )
    denominator = (
        float(
            math.ceil(
                np.percentile(
                    [len(pairs) for pairs in pairs_by_building.values()],
                    90,
                )
            )
        )
        * r_estimate
        * t90
        * float(ida_config["runtime_safety_factor"])
    )
    budget = math.floor(numerator / denominator)
    if invalid_curves:
        # Runtime observations remain useful for diagnosis, but a benchmark
        # with an unresolved curve must never authorize the production batch.
        recommended = 0
        gate = "benchmark_invalid"
    elif budget >= int(ida_config["target_buildings"]):
        recommended = int(ida_config["target_buildings"])
        gate = "full_200"
    elif budget >= 100:
        recommended = int(budget // 10 * 10)
        gate = "budget_limited"
    else:
        recommended = 50
        gate = "pipeline_demo_only"
    report = {
        "benchmark_curve_count": len(curves),
        "valid_curve_count": len(curves) - len(invalid_curves),
        "invalid_curve_count": len(invalid_curves),
        "invalid_curves": invalid_curves,
        "t90_nltha_runtime_s": t90,
        "r_nltha_per_curve_p90": r_estimate,
        "estimated_building_budget": budget,
        "recommended_building_count": recommended,
        "gate": gate,
        "worker_count": int(ida_config["workers"]),
        "benchmark_worker_count": benchmark_workers,
        "record_pairs_per_building_p90": int(
            math.ceil(
                np.percentile(
                    [len(pairs) for pairs in pairs_by_building.values()],
                    90,
                )
            )
        ),
        "building_ids": [item["building_id"] for item in selected_buildings],
        "scwb_classes": [
            item.get("scwb_class", "unclassified")
            for item in selected_buildings
        ],
        "pair_ids_by_building": selected_pair_ids,
        "research_phase": "data_generation_only_ml_deferred",
    }
    output_path = Path(config["output_dir"]) / "ida_benchmark.json"
    atomic_write_json(output_path, report)
    report["report_path"] = str(output_path.absolute())
    return report
