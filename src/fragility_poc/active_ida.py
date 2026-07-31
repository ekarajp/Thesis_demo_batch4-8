"""Frozen surrogate support for capacity-seeking hybrid active IDA.

The surrogate is deliberately limited to *point selection*.  Every stored
IO/LS/CP capacity remains bracketed by actual NLTHA results.  If prediction
loading, feature validation, or inference fails, the controller falls back to
the deterministic SPO-informed seed without changing the acceptance rules.
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .constants import G_STD, LIMIT_STATES

ACTIVE_IDA_PREDICTOR_SCHEMA = "frozen-rf-point-selector-v1"
ACTIVE_IDA_FEATURES = (
    "vy_kn",
    "dy_m",
    "vc_kn",
    "dc_m",
    "vu_kn",
    "du_m",
    "t1_s",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_log_features(
    building: dict[str, Any],
    pair_id: str,
    pair_ids: list[str],
) -> np.ndarray:
    numeric = []
    for name in ACTIVE_IDA_FEATURES:
        value = float(building[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Hybrid IDA feature {name} must be positive and finite"
            )
        numeric.append(math.log(value))
    categorical = [1.0 if pair_id == item else 0.0 for item in pair_ids]
    return np.asarray([numeric + categorical], dtype=float)


def spo_yield_sa_g(building: dict[str, Any]) -> float:
    """Return an SPO-informed modal yield spectral-acceleration proxy."""
    stories = int(building["stories"])
    floor_mass = float(building["floor_mass_kn_s2_m"])
    modal_mass_ratio = float(building["x_mode_effective_mass_ratio"])
    vy_kn = float(building["vy_kn"])
    effective_mass = stories * floor_mass * modal_mass_ratio
    if (
        stories <= 0
        or effective_mass <= 0.0
        or not math.isfinite(effective_mass)
        or vy_kn <= 0.0
        or not math.isfinite(vy_kn)
    ):
        raise ValueError("Cannot form a positive finite SPO yield-Sa proxy")
    return vy_kn / (effective_mass * G_STD)


def heuristic_capacity_prediction(
    building: dict[str, Any],
    controller: dict[str, Any],
) -> dict[str, float]:
    """Return the deterministic prediction used when ML is unavailable."""
    yield_sa = spo_yield_sa_g(building)
    ratios = controller["spo_capacity_ratio_seed"]
    prediction = {
        state: yield_sa * float(ratios[state]) for state in LIMIT_STATES
    }
    return _ordered_positive_prediction(prediction)


def _ordered_positive_prediction(
    prediction: dict[str, float],
    *,
    minimum_ratio: float = 1.02,
) -> dict[str, float]:
    ordered: dict[str, float] = {}
    previous = 0.0
    for state in LIMIT_STATES:
        value = float(prediction[state])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Active IDA capacity predictions must be positive")
        value = max(value, previous * minimum_ratio)
        ordered[state] = value
        previous = value
    return ordered


@lru_cache(maxsize=4)
def _load_frozen_predictor(
    model_path: str,
    expected_sha256: str,
) -> dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Active IDA predictor is missing: {path}")
    actual_sha256 = _file_sha256(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "Active IDA predictor checksum mismatch: "
            f"expected {expected_sha256}, observed {actual_sha256}"
        )
    artifact = joblib.load(path)
    if artifact.get("schema") != ACTIVE_IDA_PREDICTOR_SCHEMA:
        raise ValueError("Unsupported active IDA predictor schema")
    if tuple(artifact.get("feature_names", ())) != ACTIVE_IDA_FEATURES:
        raise ValueError("Active IDA predictor feature schema mismatch")
    if tuple(artifact.get("target_names", ())) != tuple(LIMIT_STATES):
        raise ValueError("Active IDA predictor target schema mismatch")
    return artifact


def predict_active_capacities(
    building: dict[str, Any],
    pair: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return a bounded frozen-ML prediction and auditable diagnostics."""
    controller = config["ida"]["controller"]
    heuristic = heuristic_capacity_prediction(building, controller)
    diagnostics: dict[str, Any] = {
        "prediction_source": "spo_ratio_fallback",
        "heuristic_prediction_g": heuristic,
        "model_path": controller.get("model_path"),
        "model_sha256": controller.get("model_sha256"),
    }
    if not bool(controller.get("use_frozen_surrogate", True)):
        return heuristic, diagnostics

    try:
        artifact = _load_frozen_predictor(
            str(controller["model_path"]),
            str(controller["model_sha256"]),
        )
        pair_id = str(pair["pair_id"])
        features = _positive_log_features(
            building,
            pair_id,
            list(artifact["pair_ids"]),
        )
        raw_log = np.asarray(artifact["estimator"].predict(features))[0]
        raw = {
            state: math.exp(float(value))
            for state, value in zip(LIMIT_STATES, raw_log, strict=True)
        }
        raw = _ordered_positive_prediction(raw)
        blend = float(controller["surrogate_blend_weight"])
        lower_ratio, upper_ratio = [
            float(value) for value in controller["prediction_clamp_ratio"]
        ]
        bounded = {}
        for state in LIMIT_STATES:
            blended = math.exp(
                blend * math.log(raw[state])
                + (1.0 - blend) * math.log(heuristic[state])
            )
            bounded[state] = min(
                max(blended, heuristic[state] * lower_ratio),
                heuristic[state] * upper_ratio,
            )
        bounded = _ordered_positive_prediction(bounded)
        diagnostics.update(
            {
                "prediction_source": "frozen_rf_blended_with_spo",
                "raw_surrogate_prediction_g": raw,
                "bounded_prediction_g": bounded,
                "training_curve_count": int(
                    artifact["training_curve_count"]
                ),
                "training_building_count": int(
                    artifact["training_building_count"]
                ),
            }
        )
        return bounded, diagnostics
    except Exception as exc:
        if bool(controller.get("require_frozen_surrogate", True)):
            raise
        diagnostics["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return heuristic, diagnostics
