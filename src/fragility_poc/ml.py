"""Leakage-safe split, ANN training, baselines, and independent-test evaluation."""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .catalog import SCWB_RESEARCH_CLASSES

from .constants import FEATURE_COLUMNS, LIMIT_STATES, TARGET_COLUMNS
from .db import connect, initialize, transaction, upsert_many
from .io_utils import atomic_write_json, stable_hash

DESIGN_BALANCE_COLUMNS = (
    "fc_ksc",
    "number_of_bays",
    "bay_width_m",
    "sdl_kg_m2",
    "ll_kg_m2",
    "beam_tier",
    "column_tier",
    "scwb_class",
)
ML_SPLIT_SCHEMA = "post-dataset-balanced-split-v7"


def _spo_split_frame(config: dict[str, Any]) -> pd.DataFrame:
    with connect(config["database_path"]) as connection:
        frame = pd.read_sql_query(
            """
            SELECT b.building_id, b.model_hash, b.queue_rank,
                   b.fc_ksc, b.number_of_bays, b.bay_width_m,
                   b.sdl_kg_m2, b.ll_kg_m2, b.beam_tier, b.column_tier,
                   b.scwb_class,
                   s.t1_s, s.analysis_signature
            FROM building_catalog b
            JOIN spo_features s USING(building_id)
            WHERE b.selected=1
              AND b.valid=1
              AND s.valid=1
              AND s.collapse_reached=1
              AND COALESCE(s.analysis_guard_triggered, 0)=0
            ORDER BY b.queue_rank
            """,
            connection,
        )
    from .spo import _spo_analysis_signature

    current = [
        row.analysis_signature
        == _spo_analysis_signature(
            {
                "building_id": row.building_id,
                "model_hash": row.model_hash,
            },
            config,
        )
        for row in frame.itertuples(index=False)
    ]
    return frame.loc[current].drop(
        columns=["model_hash", "analysis_signature"]
    )


def _balance_score(
    frame: pd.DataFrame,
    development_indices: np.ndarray,
    test_indices: np.ndarray,
) -> float:
    score = 0.0
    development = frame.iloc[development_indices]
    test = frame.iloc[test_indices]
    for column in (*DESIGN_BALANCE_COLUMNS, "t1_quartile"):
        all_values = sorted(frame[column].unique())
        overall = frame[column].value_counts(normalize=True)
        dev = development[column].value_counts(normalize=True)
        holdout = test[column].value_counts(normalize=True)
        for value in all_values:
            score += abs(float(dev.get(value, 0.0)) - float(overall.get(value, 0.0)))
            score += abs(float(holdout.get(value, 0.0)) - float(overall.get(value, 0.0)))
    return score


def _supported_strata(
    frame: pd.DataFrame,
    *,
    minimum_count: int,
    maximum_classes: int | None = None,
) -> tuple[pd.Series, str]:
    """Use joint T1/SCWB strata only when every stratum is estimable.

    Small bounded pilots can legitimately contain one member of a particular
    ``T1 quartile × SCWB class`` cell.  scikit-learn cannot stratify such a
    cell, and duplicating the building would create leakage.  In that case the
    split remains stratified by the proposal-prescribed T1 quartile while the
    existing multi-variable balance score still balances SCWB class and all
    design variables.
    """
    joint = (
        frame["t1_quartile"].astype(str)
        + "|"
        + frame["scwb_class"].astype(str)
    )
    joint_counts = joint.value_counts()
    joint_supported = (
        not joint_counts.empty
        and int(joint_counts.min()) >= int(minimum_count)
        and (
            maximum_classes is None
            or int(joint_counts.size) <= int(maximum_classes)
        )
    )
    if joint_supported:
        return joint, "t1_quartile_x_scwb_class"
    quartile = frame["t1_quartile"].astype(str)
    quartile_counts = quartile.value_counts()
    if (
        quartile_counts.empty
        or int(quartile_counts.min()) < int(minimum_count)
        or (
            maximum_classes is not None
            and int(quartile_counts.size) > int(maximum_classes)
        )
    ):
        raise RuntimeError(
            "The available Building IDs cannot support the requested "
            "T1-stratified split/folds without duplicating observations"
        )
    return quartile, "t1_quartile_fallback_for_sparse_joint_cells"


def _constrained_scwb_test_counts(
    frame: pd.DataFrame,
    available: pd.DataFrame,
    *,
    desired_test_count: int,
    minimum_per_class: int,
) -> dict[str, int]:
    """Allocate the holdout while enforcing the reported SCWB-class minimum.

    The target proportions follow the complete label-blind population.  A
    minimum is then imposed for every represented research class, and any
    remaining slots are assigned greedily to the largest proportional
    shortfall.  Buildings with pre-existing IDA results are absent from
    ``available`` and can therefore never be assigned to the holdout.
    """
    observed_classes = [
        class_name
        for class_name in SCWB_RESEARCH_CLASSES
        if bool((frame["scwb_class"] == class_name).any())
    ]
    unexpected = sorted(
        set(frame["scwb_class"].astype(str)) - set(observed_classes)
    )
    if unexpected:
        raise RuntimeError(
            "The label-blind split found unsupported SCWB classes: "
            + ", ".join(unexpected)
        )
    if desired_test_count < minimum_per_class * len(observed_classes):
        raise RuntimeError(
            "The requested independent-test size cannot provide at least "
            f"{minimum_per_class} buildings for each of "
            f"{len(observed_classes)} represented SCWB classes"
        )
    full_counts = frame["scwb_class"].value_counts()
    available_counts = available["scwb_class"].value_counts()
    upper = {
        class_name: int(available_counts.get(class_name, 0))
        for class_name in observed_classes
    }
    unavailable = [
        class_name
        for class_name in observed_classes
        if upper[class_name] < minimum_per_class
    ]
    if unavailable:
        raise RuntimeError(
            "Too many pre-existing IDA buildings prevent the independent "
            f"test set from retaining {minimum_per_class} observations in "
            "these SCWB classes: "
            + ", ".join(unavailable)
        )
    if sum(upper.values()) < desired_test_count:
        raise RuntimeError(
            "The non-exposed population is smaller than the requested "
            "independent test set"
        )

    targets = {
        class_name: (
            desired_test_count
            * float(full_counts[class_name])
            / float(len(frame))
        )
        for class_name in observed_classes
    }
    allocation = {
        class_name: minimum_per_class for class_name in observed_classes
    }
    while sum(allocation.values()) < desired_test_count:
        eligible = [
            class_name
            for class_name in observed_classes
            if allocation[class_name] < upper[class_name]
        ]
        if not eligible:
            raise RuntimeError(
                "The constrained SCWB allocation exhausted all available "
                "holdout candidates"
            )
        chosen = max(
            eligible,
            key=lambda class_name: (
                targets[class_name] - allocation[class_name],
                -SCWB_RESEARCH_CLASSES.index(class_name),
            ),
        )
        allocation[chosen] += 1
    return allocation


def _constrained_holdout_candidates(
    frame: pd.DataFrame,
    *,
    available_indices: np.ndarray,
    exposed_indices: np.ndarray,
    allocation: dict[str, int],
    seed: int,
    candidate_count: int = 256,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate T1-stratified candidates with exact SCWB holdout counts."""
    class_candidates: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    available = frame.iloc[available_indices]
    for class_index, (class_name, test_count) in enumerate(
        allocation.items()
    ):
        class_local = np.flatnonzero(
            (available["scwb_class"].astype(str) == class_name).to_numpy()
        )
        class_frame = available.iloc[class_local]
        quartiles = class_frame["t1_quartile"].astype(str)
        quartile_counts = quartiles.value_counts()
        stratification_supported = (
            not quartile_counts.empty
            and int(quartile_counts.min()) >= 2
            and test_count >= int(quartile_counts.size)
            and len(class_frame) - test_count >= int(quartile_counts.size)
        )
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        if stratification_supported:
            splitter = StratifiedShuffleSplit(
                n_splits=candidate_count,
                test_size=test_count,
                random_state=seed + 104729 * class_index,
            )
            for development_local, test_local in splitter.split(
                class_frame,
                quartiles.to_numpy(),
            ):
                splits.append(
                    (
                        class_local[development_local],
                        class_local[test_local],
                    )
                )
        else:
            rng = np.random.default_rng(seed + 104729 * class_index)
            for _ in range(candidate_count):
                shuffled = rng.permutation(class_local)
                splits.append(
                    (shuffled[test_count:], shuffled[:test_count])
                )
        class_candidates[class_name] = splits

    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for candidate_index in range(candidate_count):
        development_available = np.concatenate(
            [
                class_candidates[class_name][candidate_index][0]
                for class_name in allocation
            ]
        )
        test_available = np.concatenate(
            [
                class_candidates[class_name][candidate_index][1]
                for class_name in allocation
            ]
        )
        candidates.append(
            (
                np.concatenate(
                    (
                        available_indices[development_available],
                        exposed_indices,
                    )
                ),
                available_indices[test_available],
            )
        )
    return candidates


def create_ml_split(
    config: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Create split using SPO/design data only; fragility labels are never queried."""
    initialize(config["database_path"])
    frame = _spo_split_frame(config)
    with connect(config["database_path"]) as connection:
        selected_population_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM building_catalog
                WHERE selected=1 AND valid=1
                """
            ).fetchone()[0]
        )
    if len(frame) != selected_population_count:
        raise RuntimeError(
            "The label-blind split cannot be created from a partial SPO "
            "population: current valid SPO/T1 results exist for "
            f"{len(frame)}/{selected_population_count} selected buildings."
        )
    building_ids = frame["building_id"].astype(str).tolist()
    selection_payload: list[dict[str, Any]] = []
    selection_counts: dict[str, int] = {building_id: 0 for building_id in building_ids}
    pwsa_counts: dict[str, int] = {building_id: 0 for building_id in building_ids}
    if building_ids:
        marks = ",".join("?" for _ in building_ids)
        with connect(config["database_path"]) as connection:
            selection_rows = [
                dict(row)
                for row in connection.execute(
                    f"""
                    SELECT sel.building_id, sel.pair_id, g.source_set,
                           g.sha256_x, g.sha256_y, g.dt_s, g.npts
                    FROM building_ground_motion_selection sel
                    JOIN ground_motion_catalog g USING(pair_id)
                    WHERE g.valid=1 AND sel.building_id IN ({marks})
                    ORDER BY sel.building_id, sel.pair_id
                    """,
                    building_ids,
                )
            ]
        for row in selection_rows:
            building_id = str(row["building_id"])
            selection_counts[building_id] += 1
            if str(row["source_set"]).startswith("PWSA"):
                pwsa_counts[building_id] += 1
            selection_payload.append(row)
    minimum_pairs = int(config["ida"].get("minimum_pairs_for_fragility", 4))
    incomplete = [
        building_id
        for building_id, count in selection_counts.items()
        if count < minimum_pairs
    ]
    if incomplete:
        raise RuntimeError(
            "The label-blind split requires a validated ground-motion "
            f"selection for every eligible SPO building; {len(incomplete)} "
            f"buildings have fewer than {minimum_pairs} pairs."
        )
    if bool(config["ground_motion"].get("require_pwsa_for_full_batch", True)):
        without_pwsa = [
            building_id
            for building_id, count in pwsa_counts.items()
            if count < 1
        ]
        if without_pwsa:
            raise RuntimeError(
                "The label-blind split requires PWSA in every eligible "
                f"record set; missing for {len(without_pwsa)} buildings."
            )
    source_payload = [
        {
            column: (
                str(value)
                if column in {"building_id", "scwb_class"}
                else int(value)
                if column in {
                    "queue_rank",
                    "number_of_bays",
                    "beam_tier",
                    "column_tier",
                }
                else float(value)
            )
            for column, value in row.items()
        }
        for row in frame[
            [
                "building_id",
                "queue_rank",
                *DESIGN_BALANCE_COLUMNS,
                "t1_s",
            ]
        ].to_dict(orient="records")
    ]
    with connect(config["database_path"]) as connection:
        existing_count = int(
            connection.execute("SELECT COUNT(*) FROM ml_split").fetchone()[0]
        )
        manifest_row = connection.execute(
            "SELECT * FROM ml_split_manifest WHERE manifest_key=1"
        ).fetchone()
    exposed_building_ids: list[str] = []
    source_signature = stable_hash(
        {
            "schema": ML_SPLIT_SCHEMA,
            "spo_rows": source_payload,
            "ground_motion_selection": selection_payload,
            "feature_columns": FEATURE_COLUMNS,
            "model_config": config["model"],
            "holdout_policy": {
                "method": "constrained_scwb_minimum_then_t1_balance",
                "minimum_test_buildings_per_class": int(
                    config["ml"]["mechanism_ambiguity_alarm"][
                        "minimum_test_buildings_per_class"
                    ]
                ),
            },
            "ida_label_config": {
                key: config["ida"][key]
                for key in (
                    "initial_im_g",
                    "minimum_im_g",
                    "hunt_multiplier",
                    "cp_hunt_soft_warning_im_g",
                    "maximum_hunt_steps",
                    "scale_factor_review_threshold",
                    "maximum_scale_factor_guard",
                    "require_cp_crossing",
                    "bracket_tolerance",
                    "limit_state_midr",
                    )
            },
        }
    )
    if existing_count and not force:
        if manifest_row is None:
            raise RuntimeError(
                "Existing ML split has no provenance signature. Recreate it "
                "with create-ml-split --force before continuing."
            )
        manifest = dict(manifest_row)
        expected = {
            "source_signature": source_signature,
            "source_count": len(frame),
            "random_seed": int(config["random_seed"]),
            "cv_folds": int(config["ml"]["cv_folds"]),
            "test_fraction": float(config["ml"]["test_fraction"]),
        }
        mismatches = [
            key
            for key, value in expected.items()
            if manifest[key] != value
        ]
        if mismatches:
            raise RuntimeError(
                "Existing ML split is stale for the current SPO/catalog "
                f"inputs ({', '.join(mismatches)}). Recreate it with "
                "create-ml-split --force."
            )
        return {
            "created": False,
            "existing_count": existing_count,
            "source_signature": source_signature,
            "message": "existing split manifest retained",
        }
    folds = int(config["ml"]["cv_folds"])
    if len(frame) < max(20, folds * 4):
        raise RuntimeError(
            f"At least {max(20, folds * 4)} valid SPO buildings are required "
            f"to create the split; found {len(frame)}"
        )
    frame["t1_quartile"] = (
        pd.qcut(frame["t1_s"], 4, labels=False, duplicates="drop").astype(int)
        + 1
    )
    if frame["t1_quartile"].nunique() < 4:
        raise RuntimeError("T1 values do not support four non-empty quartiles")
    seed = int(config["random_seed"])
    exposed_mask = frame["building_id"].astype(str).isin(exposed_building_ids)
    exposed_indices = np.flatnonzero(exposed_mask.to_numpy())
    available_indices = np.flatnonzero((~exposed_mask).to_numpy())
    desired_test_count = int(
        math.ceil(len(frame) * float(config["ml"]["test_fraction"]))
    )
    if len(available_indices) < desired_test_count:
        raise RuntimeError(
            "Too many buildings already have exploratory IDA capacities to "
            "preserve the requested independent-test size."
        )
    available = frame.iloc[available_indices].copy()
    minimum_test_per_class = int(
        config["ml"]["mechanism_ambiguity_alarm"][
            "minimum_test_buildings_per_class"
        ]
    )
    test_class_allocation = _constrained_scwb_test_counts(
        frame,
        available,
        desired_test_count=desired_test_count,
        minimum_per_class=minimum_test_per_class,
    )
    candidate_splits = _constrained_holdout_candidates(
        frame,
        available_indices=available_indices,
        exposed_indices=exposed_indices,
        allocation=test_class_allocation,
        seed=seed,
    )
    holdout_stratification = (
        "constrained_scwb_minimum_within_class_t1_quartiles"
    )
    development_indices, test_indices = min(
        candidate_splits,
        key=lambda item: _balance_score(frame, item[0], item[1]),
    )
    development = frame.iloc[development_indices].copy()
    test = frame.iloc[test_indices].copy()

    development["fold"] = -1
    fold_strata, fold_stratification = _supported_strata(
        development,
        minimum_count=folds,
    )
    fold_splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    for fold, (_, validation_positions) in enumerate(
        fold_splitter.split(
            development,
            fold_strata.to_numpy(),
        )
    ):
        development.iloc[
            validation_positions,
            development.columns.get_loc("fold"),
        ] = fold
    split_rows = []
    for row in development.itertuples(index=False):
        split_rows.append(
            {
                "building_id": row.building_id,
                "split": "development",
                "fold": int(row.fold),
                "t1_quartile": int(row.t1_quartile),
                "random_seed": seed,
            }
        )
    for row in test.itertuples(index=False):
        split_rows.append(
            {
                "building_id": row.building_id,
                "split": "test",
                "fold": None,
                "t1_quartile": int(row.t1_quartile),
                "random_seed": seed,
            }
        )
    with transaction(config["database_path"]) as connection:
        if force:
            connection.execute("DELETE FROM ml_split")
            connection.execute("DELETE FROM ml_split_manifest")
        upsert_many(connection, "ml_split", split_rows, ("building_id",))
        upsert_many(
            connection,
            "ml_split_manifest",
            [
                {
                    "manifest_key": 1,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_signature": source_signature,
                    "source_count": len(frame),
                    "random_seed": seed,
                    "cv_folds": folds,
                    "test_fraction": float(config["ml"]["test_fraction"]),
                }
            ],
            ("manifest_key",),
        )

    output_path = Path(config["output_dir"]) / "ml_split_manifest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(
            sorted(split_rows, key=lambda row: (row["split"], row["building_id"]))
        )
    # These four buildings are selected using T1 only, before fragility labels
    # are joined, so later overlays cannot be cherry-picked for accuracy.
    preselected_demo = []
    for quartile, group in test.groupby("t1_quartile"):
        median_t1 = float(group["t1_s"].median())
        chosen = group.iloc[
            np.argmin(np.abs(group["t1_s"].to_numpy(float) - median_t1))
        ]
        preselected_demo.append(
            {
                "t1_quartile": int(quartile),
                "building_id": str(chosen["building_id"]),
                "t1_s": float(chosen["t1_s"]),
                "selection_rule": "closest to within-quartile median T1",
            }
        )
    demo_path = Path(config["output_dir"]) / "demo_building_selection.csv"
    pd.DataFrame(preselected_demo).to_csv(
        demo_path, index=False, encoding="utf-8-sig"
    )
    balance = {
        column: {
            "development": development[column].value_counts(normalize=True).sort_index().to_dict(),
            "test": test[column].value_counts(normalize=True).sort_index().to_dict(),
        }
        for column in (*DESIGN_BALANCE_COLUMNS, "t1_quartile")
    }
    return {
        "created": True,
        "development_count": len(development),
        "test_count": len(test),
        "fold_count": folds,
        "holdout_stratification": holdout_stratification,
        "minimum_test_buildings_per_scwb_class": minimum_test_per_class,
        "test_scwb_class_allocation": test_class_allocation,
        "fold_stratification": fold_stratification,
        "seed": seed,
        "source_signature": source_signature,
        "balance": balance,
        "manifest_path": str(output_path.absolute()),
        "demo_selection_path": str(demo_path.absolute()),
    }


def _eligible_ml_frame(config: dict[str, Any]) -> pd.DataFrame:
    """Return eligible features without reading independent-test target values."""
    with connect(config["database_path"]) as connection:
        frame = pd.read_sql_query(
            """
            SELECT b.building_id, b.model_hash,
                   b.scwb_strength_ratio, b.scwb_class,
                   m.split, m.fold, m.t1_quartile,
                   s.vy_kn, s.dy_m, s.vc_kn, s.dc_m, s.vu_kn, s.du_m,
                   s.collapse_classification,
                   s.mechanism_class_at_ultimate,
                   s.ultimate_beam_yielded_end_fraction,
                   s.ultimate_column_yielded_end_fraction,
                   s.ultimate_max_story_column_yielded_end_fraction,
                   s.t1_s, s.analysis_signature,
                    f.n_pairs, f.source_signature,
                    stats.selected_pair_count,
                    stats.valid_selected_pair_count,
                    stats.pwsa_pair_count
            FROM ml_split m
            JOIN building_catalog b USING(building_id)
            JOIN spo_features s USING(building_id)
            JOIN fragility_targets f USING(building_id)
            JOIN (
                SELECT sel.building_id,
                       COUNT(*) AS selected_pair_count,
                       SUM(CASE WHEN g.valid=1 THEN 1 ELSE 0 END)
                           AS valid_selected_pair_count,
                       SUM(CASE WHEN g.valid=1
                                     AND g.source_set LIKE 'PWSA%'
                                THEN 1 ELSE 0 END) AS pwsa_pair_count
                FROM building_ground_motion_selection sel
                JOIN ground_motion_catalog g USING(pair_id)
                GROUP BY sel.building_id
            ) stats USING(building_id)
            WHERE b.selected=1
              AND b.valid=1
              AND s.valid=1
              AND s.collapse_reached=1
              AND COALESCE(s.analysis_guard_triggered, 0)=0
              AND f.valid=1
              AND f.n_pairs=stats.selected_pair_count
              AND stats.valid_selected_pair_count=stats.selected_pair_count
            ORDER BY b.queue_rank
            """,
            connection,
        )
    from .fragility import fragility_source_signature
    from .spo import _spo_analysis_signature

    building_ids = frame["building_id"].astype(str).tolist()
    capacity_by_building: dict[str, list[dict[str, Any]]] = {
        building_id: [] for building_id in building_ids
    }
    if building_ids:
        marks = ",".join("?" for _ in building_ids)
        with connect(config["database_path"]) as connection:
            for row in connection.execute(
                f"""
                SELECT c.*
                FROM ida_capacities c
                JOIN building_ground_motion_selection s
                  ON s.building_id=c.building_id
                 AND s.pair_id=c.pair_id
                WHERE c.building_id IN ({marks})
                ORDER BY c.building_id, c.pair_id, c.limit_state
                """,
                building_ids,
            ):
                capacity_by_building[str(row["building_id"])].append(
                    dict(row)
                )

    current = [
        (
            row.analysis_signature
            == _spo_analysis_signature(
                {
                    "building_id": row.building_id,
                    "model_hash": row.model_hash,
                },
                config,
            )
            and row.source_signature
            == fragility_source_signature(
                str(row.building_id),
                str(row.analysis_signature),
                capacity_by_building[str(row.building_id)],
                config,
            )
        )
        for row in frame.itertuples(index=False)
    ]
    return frame.loc[current].drop(
        columns=["model_hash", "analysis_signature", "source_signature"]
    )


def _fragility_labels(
    database_path: str,
    building_ids: list[str],
) -> pd.DataFrame:
    """Read target values only for the explicitly authorized Building IDs."""
    if not building_ids:
        return pd.DataFrame(columns=["building_id", *TARGET_COLUMNS])
    marks = ",".join("?" for _ in building_ids)
    with connect(database_path) as connection:
        labels = pd.read_sql_query(
            f"""
            SELECT building_id, {", ".join(TARGET_COLUMNS)}
            FROM fragility_targets
            WHERE valid=1 AND building_id IN ({marks})
            """,
            connection,
            params=building_ids,
        )
    indexed = labels.set_index("building_id")
    missing = [
        building_id
        for building_id in building_ids
        if building_id not in indexed.index
    ]
    if missing:
        raise RuntimeError(
            f"Missing valid fragility labels for {len(missing)} Building IDs"
        )
    return indexed.loc[building_ids].reset_index()


def _fit_predict_log_model(
    estimator: Any,
    x_train: np.ndarray,
    y_log_train: np.ndarray,
    x_validation: np.ndarray,
) -> tuple[Any, np.ndarray]:
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y_log_train)
    fitted = clone(estimator)
    fitted.fit(x_train, y_scaled)
    predicted_scaled = np.asarray(fitted.predict(x_validation))
    if predicted_scaled.ndim == 1:
        predicted_scaled = predicted_scaled.reshape(-1, 1)
    predicted_log = target_scaler.inverse_transform(predicted_scaled)
    return {
        "estimator": fitted,
        "target_scaler": target_scaler,
    }, predicted_log


def _nrmse(
    actual_log: np.ndarray,
    predicted_log: np.ndarray,
    training_log: np.ndarray,
) -> tuple[float, np.ndarray]:
    rmse = np.sqrt(np.mean((actual_log - predicted_log) ** 2, axis=0))
    scale = np.std(training_log, axis=0, ddof=1)
    normalized = rmse / np.maximum(scale, 1.0e-12)
    return float(np.mean(normalized)), normalized


def _ann_pipeline(
    architecture: tuple[int, ...],
    alpha: float,
    *,
    seed: int,
    max_iter: int,
) -> Pipeline:
    return Pipeline(
        (
            ("feature_scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=architecture,
                    activation="relu",
                    solver="lbfgs",
                    alpha=alpha,
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        )
    )


def _ridge_pipeline() -> Pipeline:
    return Pipeline(
        (
            ("feature_scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        )
    )


def train_ann(config: dict[str, Any]) -> dict[str, Any]:
    """Select the ANN with development data and save untouched test predictions."""
    split_report = create_ml_split(config, force=False)
    frame = _eligible_ml_frame(config)
    minimum_pairs = int(config["ml"].get("minimum_pairs_for_ml", 4))
    frame = frame[frame["n_pairs"] >= minimum_pairs].copy()
    if bool(config["ground_motion"].get("require_pwsa_for_full_batch", True)):
        frame = frame[frame["pwsa_pair_count"] >= 1].copy()
    development = frame[frame["split"] == "development"].copy()
    test = frame[frame["split"] == "test"].copy()
    folds = int(config["ml"]["cv_folds"])
    if len(development) < folds * 4 or len(test) < 5:
        raise RuntimeError(
            "Insufficient complete selected-record fragility labels for ANN training: "
            f"development={len(development)}, test={len(test)}"
        )
    x_dev = development[list(FEATURE_COLUMNS)].to_numpy(float)
    x_test = test[list(FEATURE_COLUMNS)].to_numpy(float)
    development_labels = _fragility_labels(
        config["database_path"],
        development["building_id"].astype(str).tolist(),
    )
    y_dev = development_labels[list(TARGET_COLUMNS)].to_numpy(float)
    if np.any(y_dev <= 0):
        raise ValueError("All development theta and beta targets must be positive")
    if not np.all(np.isfinite(x_dev)) or not np.all(np.isfinite(x_test)):
        raise ValueError("All seven ANN input features must be finite")
    y_log_dev = np.log(y_dev)
    fold_assignments = development["fold"].to_numpy(int)
    architectures = [
        tuple(int(width) for width in candidate)
        for candidate in config["ml"]["architectures"]
    ]
    alphas = [float(value) for value in config["ml"]["alphas"]]
    selection_seed = int(config["random_seed"])
    max_iter = int(config["ml"]["max_iter"])
    cv_results = []
    for architecture in architectures:
        for alpha in alphas:
            fold_scores = []
            per_target = []
            fold_iterations = []
            fold_converged = []
            for fold in range(folds):
                training_mask = fold_assignments != fold
                validation_mask = fold_assignments == fold
                estimator = _ann_pipeline(
                    architecture,
                    alpha,
                    seed=selection_seed,
                    max_iter=max_iter,
                )
                fitted_candidate, predicted_log = _fit_predict_log_model(
                    estimator,
                    x_dev[training_mask],
                    y_log_dev[training_mask],
                    x_dev[validation_mask],
                )
                score, target_scores = _nrmse(
                    y_log_dev[validation_mask],
                    predicted_log,
                    y_log_dev[training_mask],
                )
                fold_scores.append(score)
                per_target.append(target_scores.tolist())
                fitted_mlp = fitted_candidate["estimator"].named_steps[
                    "model"
                ]
                iterations = int(fitted_mlp.n_iter_)
                fold_iterations.append(iterations)
                fold_converged.append(iterations < max_iter)
            cv_results.append(
                {
                    "architecture": list(architecture),
                    "alpha": alpha,
                    "fold_nrmse": fold_scores,
                    "per_target_nrmse": per_target,
                    "fold_iterations": fold_iterations,
                    "fold_converged": fold_converged,
                    "all_folds_converged": all(fold_converged),
                    "mean_nrmse": float(np.mean(fold_scores)),
                    "se_nrmse": float(
                        np.std(fold_scores, ddof=1) / math.sqrt(folds)
                    ),
                    "complexity": int(sum(architecture)),
                }
            )
    best = min(cv_results, key=lambda row: row["mean_nrmse"])
    one_se_limit = float(best["mean_nrmse"] + best["se_nrmse"])
    eligible = [
        result
        for result in cv_results
        if float(result["mean_nrmse"]) <= one_se_limit
    ]
    converged_eligible = [
        result for result in eligible if result["all_folds_converged"]
    ]
    selection_pool = converged_eligible or eligible
    selected = min(
        selection_pool,
        key=lambda row: (
            int(row["complexity"]),
            -float(row["alpha"]),
            float(row["mean_nrmse"]),
        ),
    )
    selected_architecture = tuple(int(value) for value in selected["architecture"])
    selected_alpha = float(selected["alpha"])

    ridge_fold_scores = []
    mean_fold_scores = []
    for fold in range(folds):
        training_mask = fold_assignments != fold
        validation_mask = fold_assignments == fold
        _, ridge_prediction = _fit_predict_log_model(
            _ridge_pipeline(),
            x_dev[training_mask],
            y_log_dev[training_mask],
            x_dev[validation_mask],
        )
        ridge_score, _ = _nrmse(
            y_log_dev[validation_mask],
            ridge_prediction,
            y_log_dev[training_mask],
        )
        ridge_fold_scores.append(ridge_score)
        mean_prediction = np.repeat(
            np.mean(y_log_dev[training_mask], axis=0, keepdims=True),
            np.sum(validation_mask),
            axis=0,
        )
        mean_score, _ = _nrmse(
            y_log_dev[validation_mask],
            mean_prediction,
            y_log_dev[training_mask],
        )
        mean_fold_scores.append(mean_score)

    learning_curve_rows = []
    learning_fractions = [
        float(value) for value in config["ml"]["learning_curve_fractions"]
    ]
    for fraction in learning_fractions:
        ann_scores = []
        ridge_scores = []
        training_counts = []
        for fold in range(folds):
            available = np.flatnonzero(fold_assignments != fold)
            validation = fold_assignments == fold
            rng = np.random.default_rng(selection_seed + 104729 * fold)
            shuffled = rng.permutation(available)
            training_count = min(
                len(shuffled),
                max(
                    8,
                    int(math.ceil(float(fraction) * len(shuffled))),
                ),
            )
            chosen = shuffled[:training_count]
            _, ann_log = _fit_predict_log_model(
                _ann_pipeline(
                    selected_architecture,
                    selected_alpha,
                    seed=selection_seed + fold,
                    max_iter=max_iter,
                ),
                x_dev[chosen],
                y_log_dev[chosen],
                x_dev[validation],
            )
            ann_score, _ = _nrmse(
                y_log_dev[validation],
                ann_log,
                y_log_dev[chosen],
            )
            _, ridge_log = _fit_predict_log_model(
                _ridge_pipeline(),
                x_dev[chosen],
                y_log_dev[chosen],
                x_dev[validation],
            )
            ridge_score, _ = _nrmse(
                y_log_dev[validation],
                ridge_log,
                y_log_dev[chosen],
            )
            ann_scores.append(ann_score)
            ridge_scores.append(ridge_score)
            training_counts.append(training_count)
        learning_curve_rows.append(
            {
                "development_fraction": fraction,
                "mean_training_building_count": float(
                    np.mean(training_counts)
                ),
                "ann_mean_nrmse": float(np.mean(ann_scores)),
                "ann_se_nrmse": float(
                    np.std(ann_scores, ddof=1) / math.sqrt(folds)
                ),
                "ridge_mean_nrmse": float(np.mean(ridge_scores)),
                "ridge_se_nrmse": float(
                    np.std(ridge_scores, ddof=1) / math.sqrt(folds)
                ),
            }
        )

    ensemble = []
    ensemble_fit_audits = []
    for seed in config["ml"]["ensemble_seeds"]:
        estimator = _ann_pipeline(
            selected_architecture,
            selected_alpha,
            seed=int(seed),
            max_iter=max_iter,
        )
        fitted, predicted_log = _fit_predict_log_model(
            estimator,
            x_dev,
            y_log_dev,
            x_test,
        )
        ensemble.append(fitted)
        fitted_mlp = fitted["estimator"].named_steps["model"]
        iterations = int(fitted_mlp.n_iter_)
        ensemble_fit_audits.append(
            {
                "seed": int(seed),
                "iterations": iterations,
                "converged": iterations < max_iter,
                "loss": float(fitted_mlp.loss_),
            }
        )
    inference_started = time.perf_counter()
    ann_test_logs = []
    for fitted in ensemble:
        predicted_scaled = np.asarray(
            fitted["estimator"].predict(x_test)
        )
        if predicted_scaled.ndim == 1:
            predicted_scaled = predicted_scaled.reshape(-1, 1)
        ann_test_logs.append(
            fitted["target_scaler"].inverse_transform(predicted_scaled)
        )
    ann_prediction = np.exp(np.mean(np.stack(ann_test_logs), axis=0))
    inference_runtime_s = time.perf_counter() - inference_started
    ridge_fitted, ridge_log_prediction = _fit_predict_log_model(
        _ridge_pipeline(), x_dev, y_log_dev, x_test
    )
    ridge_prediction = np.exp(ridge_log_prediction)
    dummy = DummyRegressor(strategy="mean")
    dummy_fitted, dummy_log_prediction = _fit_predict_log_model(
        dummy, x_dev, y_log_dev, x_test
    )
    dummy_prediction = np.exp(dummy_log_prediction)

    run_payload = {
        "seed": selection_seed,
        "features": list(FEATURE_COLUMNS),
        "architecture": list(selected_architecture),
        "alpha": selected_alpha,
        "development_ids": development["building_id"].tolist(),
        "test_ids": test["building_id"].tolist(),
        "development_target_hash": stable_hash(y_dev.tolist()),
        "ml_config": config["ml"],
    }
    run_id = f"ANN-{stable_hash(run_payload)[:12]}"
    model_directory = Path(config["output_dir"]) / "ml" / run_id
    model_directory.mkdir(parents=True, exist_ok=True)
    learning_curve_path = model_directory / "learning_curve.csv"
    pd.DataFrame(learning_curve_rows).to_csv(
        learning_curve_path,
        index=False,
        encoding="utf-8-sig",
    )
    model_path = model_directory / "model.joblib"
    joblib.dump(
        {
            "features": FEATURE_COLUMNS,
            "targets": TARGET_COLUMNS,
            "ann_ensemble": ensemble,
            "ridge": ridge_fitted,
            "mean": dummy_fitted,
            "selected_architecture": selected_architecture,
            "selected_alpha": selected_alpha,
            "development_ids": development["building_id"].tolist(),
            "test_ids": test["building_id"].tolist(),
        },
        model_path,
    )
    prediction_rows = []
    for row_index, building_id in enumerate(test["building_id"]):
        row: dict[str, Any] = {"building_id": building_id}
        for target_index, target in enumerate(TARGET_COLUMNS):
            row[f"ann_{target}"] = ann_prediction[row_index, target_index]
            row[f"ridge_{target}"] = ridge_prediction[row_index, target_index]
            row[f"mean_{target}"] = dummy_prediction[row_index, target_index]
        prediction_rows.append(row)
    predictions_path = model_directory / "test_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(
        predictions_path, index=False, encoding="utf-8-sig"
    )
    demo_selection_rows = []
    for quartile, group in test.groupby("t1_quartile"):
        median_t1 = float(group["t1_s"].median())
        chosen = group.iloc[
            np.argmin(np.abs(group["t1_s"].to_numpy(float) - median_t1))
        ]
        demo_selection_rows.append(
            {
                "t1_quartile": int(quartile),
                "building_id": str(chosen["building_id"]),
                "t1_s": float(chosen["t1_s"]),
                "selection_rule": "predefined test building closest to within-quartile median T1",
            }
        )
    run_demo_selection_path = model_directory / "demo_selection.csv"
    pd.DataFrame(demo_selection_rows).to_csv(
        run_demo_selection_path, index=False, encoding="utf-8-sig"
    )
    cv_report = {
        "features": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "development_feature_standard_deviation": {
            feature: float(
                np.std(development[feature].to_numpy(float), ddof=1)
            )
            for feature in FEATURE_COLUMNS
        },
        "constant_development_features": [
            feature
            for feature in FEATURE_COLUMNS
            if np.isclose(
                np.std(
                    development[feature].to_numpy(float),
                    ddof=1,
                ),
                0.0,
            )
        ],
        "selection_metric": "mean six-output log-target NRMSE",
        "one_standard_error_limit": one_se_limit,
        "one_se_converged_candidate_count": len(converged_eligible),
        "selection_required_convergence": bool(converged_eligible),
        "selected": selected,
        "best_raw": best,
        "ann_candidates": cv_results,
        "ridge_fold_nrmse": ridge_fold_scores,
        "ridge_mean_nrmse": float(np.mean(ridge_fold_scores)),
        "mean_predictor_fold_nrmse": mean_fold_scores,
        "mean_predictor_mean_nrmse": float(np.mean(mean_fold_scores)),
        "ann_reduction_vs_ridge": 1.0
        - float(selected["mean_nrmse"]) / float(np.mean(ridge_fold_scores)),
        "selected_cv_all_folds_converged": bool(
            selected["all_folds_converged"]
        ),
        "ensemble_fit_audits": ensemble_fit_audits,
        "ensemble_all_converged": all(
            audit["converged"] for audit in ensemble_fit_audits
        ),
        "ann_ensemble_test_inference_runtime_s": inference_runtime_s,
        "ann_inference_runtime_per_building_s": inference_runtime_s
        / len(test),
        "run_demo_selection_path": str(
            run_demo_selection_path.absolute()
        ),
        "learning_curve": learning_curve_rows,
        "learning_curve_path": str(learning_curve_path.absolute()),
        "split_report": split_report,
    }
    metrics_path = model_directory / "cv_selection.json"
    atomic_write_json(metrics_path, cv_report)
    pending_summary = {
        "run_id": run_id,
        "features": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "development_count": len(development),
        "test_count": len(test),
        "model_path": str(model_path.absolute()),
        "predictions_path": str(predictions_path.absolute()),
        "cv_path": str(metrics_path.absolute()),
        "demo_selection_path": str(run_demo_selection_path.absolute()),
        "learning_curve_path": str(learning_curve_path.absolute()),
        "message": "Model trained; run evaluate to score the independent test once.",
    }
    with transaction(config["database_path"]) as connection:
        upsert_many(
            connection,
            "ml_runs",
            [
                {
                    "run_id": run_id,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "model_path": str(model_path.absolute()),
                    "metrics_path": str(metrics_path.absolute()),
                    "predictions_path": str(predictions_path.absolute()),
                    "selected_architecture": json.dumps(
                        list(selected_architecture)
                    ),
                    "selected_alpha": selected_alpha,
                    "development_count": len(development),
                    "test_count": len(test),
                    "promising": 0,
                    "summary_json": pending_summary,
                    "cv_metrics_path": str(metrics_path.absolute()),
                    "evaluation_path": None,
                    "evaluated_predictions_path": None,
                    "evaluated_utc": None,
                }
            ],
            ("run_id",),
        )
    return pending_summary


def _probability_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    errors = []
    for row in range(actual.shape[0]):
        grid_max = max(
            4.0,
            2.0
            * float(
                np.max(
                    np.concatenate(
                        (actual[row, [0, 2, 4]], predicted[row, [0, 2, 4]])
                    )
                )
            ),
        )
        grid = np.geomspace(0.01, grid_max, 200)
        for state_index, _ in enumerate(LIMIT_STATES):
            theta_index = 2 * state_index
            beta_index = theta_index + 1
            actual_probability = norm.cdf(
                np.log(grid / actual[row, theta_index])
                / actual[row, beta_index]
            )
            predicted_probability = norm.cdf(
                np.log(grid / predicted[row, theta_index])
                / predicted[row, beta_index]
            )
            errors.append(
                float(np.mean(np.abs(actual_probability - predicted_probability)))
            )
    return float(np.mean(errors))


def _ordering_violation_rate(predicted: np.ndarray) -> float:
    ordered = (
        (predicted[:, 0] < predicted[:, 2])
        & (predicted[:, 2] < predicted[:, 4])
    )
    return float(np.mean(~ordered))


def _model_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    if (
        actual.shape != predicted.shape
        or np.any(~np.isfinite(actual))
        or np.any(~np.isfinite(predicted))
        or np.any(actual <= 0.0)
        or np.any(predicted <= 0.0)
    ):
        raise ValueError(
            "Actual and predicted fragility arrays must have matching shapes "
            "with positive finite values"
        )
    targets = {}
    for index, target in enumerate(TARGET_COLUMNS):
        actual_column = actual[:, index]
        predicted_column = predicted[:, index]
        targets[target] = {
            "mae": float(mean_absolute_error(actual_column, predicted_column)),
            "rmse": float(
                mean_squared_error(
                    actual_column, predicted_column
                )
                ** 0.5
            ),
            "r2": float(r2_score(actual_column, predicted_column)),
            "median_absolute_percentage_error": float(
                np.median(
                    np.abs(
                        (predicted_column - actual_column)
                        / actual_column
                    )
                )
            ),
            "mean_absolute_percentage_error": float(
                np.mean(
                    np.abs(
                        (predicted_column - actual_column)
                        / actual_column
                    )
                )
            ),
            "mean_signed_percentage_error": float(
                np.mean(
                    (predicted_column - actual_column)
                    / actual_column
                )
            ),
        }
        if target.startswith("theta"):
            targets[target]["median_predicted_actual_ratio"] = float(
                np.median(predicted_column / actual_column)
            )
        else:
            targets[target]["mean_delta_beta"] = float(
                np.mean(predicted_column - actual_column)
            )
    return {
        "targets": targets,
        "mean_absolute_fragility_probability_error": _probability_error(
            actual, predicted
        ),
        "ordering_violation_rate": _ordering_violation_rate(predicted),
    }


def _cp_probability_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Mean absolute CP-fragility probability error over the locked IM grid."""
    errors = []
    for row in range(actual.shape[0]):
        grid = np.geomspace(
            0.01,
            max(
                4.0,
                2.0 * float(max(actual[row, 4], predicted[row, 4])),
            ),
            200,
        )
        actual_probability = norm.cdf(
            np.log(grid / actual[row, 4]) / actual[row, 5]
        )
        predicted_probability = norm.cdf(
            np.log(grid / predicted[row, 4]) / predicted[row, 5]
        )
        errors.append(
            float(np.mean(np.abs(actual_probability - predicted_probability)))
        )
    return float(np.mean(errors))


def _mechanism_ambiguity_report(
    feature_frame: pd.DataFrame,
    labels: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Flag similar SPO inputs with different mechanisms and CP labels."""
    columns = [
        "building_id",
        "scwb_class",
        "scwb_strength_ratio",
        "mechanism_class_at_ultimate",
        "ultimate_beam_yielded_end_fraction",
        "ultimate_column_yielded_end_fraction",
        "ultimate_max_story_column_yielded_end_fraction",
        *FEATURE_COLUMNS,
    ]
    merged = feature_frame[columns].merge(
        labels[["building_id", "theta_cp_g", "beta_cp"]],
        on="building_id",
        how="inner",
        validate="one_to_one",
    )
    mechanism_series = merged["mechanism_class_at_ultimate"].fillna(
        "unavailable"
    ).astype(str)
    observed_mechanisms = set(mechanism_series) - {"unavailable"}
    coverage_complete = len(observed_mechanisms) >= 2
    if len(merged) < 2 or not coverage_complete:
        return {
            "alarm": True,
            "classification_coverage_complete": coverage_complete,
            "evaluated_building_count": len(merged),
            "cross_class_nearest_pair_count": 0,
            "flagged_pair_count": 0,
            "flagged_building_count": 0,
            "flagged_pairs": [],
            "observed_mechanism_classes": sorted(observed_mechanisms),
            "message": (
                "Mechanism ambiguity could not be assessed because fewer "
                "than two recorded nonlinear mechanism classes have complete "
                "fragility labels."
            ),
        }
    features = merged[list(FEATURE_COLUMNS)].to_numpy(float)
    means = np.mean(features, axis=0)
    scales = np.std(features, axis=0, ddof=0)
    scales[scales <= np.finfo(float).eps] = 1.0
    standardized = (features - means) / scales
    classes = mechanism_series.to_numpy()
    nearest_by_pair: dict[tuple[int, int], float] = {}
    for index in range(len(merged)):
        candidates = np.flatnonzero(classes != classes[index])
        if not len(candidates):
            continue
        distances = np.linalg.norm(
            standardized[candidates] - standardized[index],
            axis=1,
        )
        neighbor = int(candidates[int(np.argmin(distances))])
        pair_key = tuple(sorted((index, neighbor)))
        distance = float(np.min(distances))
        nearest_by_pair[pair_key] = min(
            distance,
            nearest_by_pair.get(pair_key, math.inf),
        )
    maximum_distance = float(
        settings["maximum_standardized_feature_distance"]
    )
    minimum_theta_ratio = float(settings["minimum_cp_theta_ratio"])
    minimum_beta_difference = float(settings["minimum_cp_beta_difference"])
    flagged_pairs = []
    for (first, second), distance in nearest_by_pair.items():
        first_row = merged.iloc[first]
        second_row = merged.iloc[second]
        theta_ratio = float(
            max(first_row["theta_cp_g"], second_row["theta_cp_g"])
            / min(first_row["theta_cp_g"], second_row["theta_cp_g"])
        )
        beta_difference = float(
            abs(first_row["beta_cp"] - second_row["beta_cp"])
        )
        if (
            distance <= maximum_distance
            and (
                theta_ratio >= minimum_theta_ratio
                or beta_difference >= minimum_beta_difference
            )
        ):
            flagged_pairs.append(
                {
                    "building_id_a": str(first_row["building_id"]),
                    "scwb_class_a": str(first_row["scwb_class"]),
                    "scwb_strength_ratio_a": float(
                        first_row["scwb_strength_ratio"]
                    ),
                    "mechanism_class_a": str(
                        first_row["mechanism_class_at_ultimate"]
                    ),
                    "building_id_b": str(second_row["building_id"]),
                    "scwb_class_b": str(second_row["scwb_class"]),
                    "scwb_strength_ratio_b": float(
                        second_row["scwb_strength_ratio"]
                    ),
                    "mechanism_class_b": str(
                        second_row["mechanism_class_at_ultimate"]
                    ),
                    "standardized_feature_distance": distance,
                    "cp_theta_ratio": theta_ratio,
                    "cp_beta_difference": beta_difference,
                }
            )
    flagged_pairs.sort(
        key=lambda row: (
            row["standardized_feature_distance"],
            -row["cp_theta_ratio"],
            -row["cp_beta_difference"],
        )
    )
    flagged_buildings = {
        building_id
        for row in flagged_pairs
        for building_id in (row["building_id_a"], row["building_id_b"])
    }
    alarm = bool(flagged_pairs) or not coverage_complete
    return {
        "alarm": alarm,
        "classification_coverage_complete": coverage_complete,
        "observed_mechanism_classes": sorted(observed_mechanisms),
        "evaluated_building_count": len(merged),
        "cross_class_nearest_pair_count": len(nearest_by_pair),
        "flagged_pair_count": len(flagged_pairs),
        "flagged_building_count": len(flagged_buildings),
        "thresholds": {
            "maximum_standardized_feature_distance": maximum_distance,
            "minimum_cp_theta_ratio": minimum_theta_ratio,
            "minimum_cp_beta_difference": minimum_beta_difference,
        },
        "flagged_pairs": flagged_pairs,
        "message": (
            "ALARM: similar seven-input SPO representations have materially "
            "different recorded mechanisms and CP fragility."
            if alarm
            else "No material cross-SCWB SPO/CP ambiguity was detected."
        ),
    }


def _scwb_class_performance_report(
    classes: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate ANN CP performance separately for every SCWB class."""
    observed = {str(value) for value in classes}
    expected_classes = tuple(
        class_name
        for class_name in SCWB_RESEARCH_CLASSES
        if class_name in observed
    )
    unobserved_classes = [
        class_name
        for class_name in SCWB_RESEARCH_CLASSES
        if class_name not in observed
    ]
    minimum_count = int(settings["minimum_test_buildings_per_class"])
    class_results = {}
    alarm_reasons = []
    for class_name in expected_classes:
        mask = classes == class_name
        count = int(np.sum(mask))
        if count:
            actual_class = actual[mask]
            predicted_class = predicted[mask]
            theta_ape = np.abs(
                (predicted_class[:, 4] - actual_class[:, 4])
                / actual_class[:, 4]
            )
            theta_median_ape = float(np.median(theta_ape))
            beta_mae = float(
                np.mean(
                    np.abs(predicted_class[:, 5] - actual_class[:, 5])
                )
            )
            probability_mae = _cp_probability_error(
                actual_class,
                predicted_class,
            )
            median_ratio = float(
                np.median(predicted_class[:, 4] / actual_class[:, 4])
            )
            median_bias = abs(median_ratio - 1.0)
        else:
            theta_median_ape = None
            beta_mae = None
            probability_mae = None
            median_ratio = None
            median_bias = None
        checks = {
            "sample_count_sufficient": count >= minimum_count,
            "cp_theta_median_ape_acceptable": (
                theta_median_ape is not None
                and theta_median_ape
                <= float(settings["maximum_class_cp_theta_median_ape"])
            ),
            "cp_beta_mae_acceptable": (
                beta_mae is not None
                and beta_mae
                <= float(settings["maximum_class_cp_beta_mae"])
            ),
            "cp_probability_mae_acceptable": (
                probability_mae is not None
                and probability_mae
                <= float(settings["maximum_class_cp_probability_mae"])
            ),
            "cp_median_bias_acceptable": (
                median_bias is not None
                and median_bias
                <= float(settings["maximum_class_cp_median_bias"])
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            alarm_reasons.append(
                {
                    "scwb_class": class_name,
                    "failed_checks": failed,
                }
            )
        class_results[class_name] = {
            "test_building_count": count,
            "cp_theta_median_absolute_percentage_error": theta_median_ape,
            "cp_beta_mean_absolute_error": beta_mae,
            "cp_fragility_probability_mae": probability_mae,
            "cp_theta_median_predicted_actual_ratio": median_ratio,
            "checks": checks,
        }
    return {
        "alarm": bool(alarm_reasons),
        "thresholds": {
            key: settings[key]
            for key in (
                "maximum_class_cp_theta_median_ape",
                "maximum_class_cp_beta_mae",
                "maximum_class_cp_probability_mae",
                "maximum_class_cp_median_bias",
                "minimum_test_buildings_per_class",
            )
        },
        "classes": class_results,
        "unobserved_research_classes": unobserved_classes,
        "coverage_note": (
            "The >=15 rule is evaluated only for research strength-margin "
            "bands present in the independent test set. Missing bands are "
            "reported, not treated as fabricated failed observations."
        ),
        "alarm_reasons": alarm_reasons,
        "message": (
            "ALARM: ANN CP performance is not acceptable in at least one "
            "SCWB class."
            if alarm_reasons
            else "ANN CP performance is acceptable in every SCWB class."
        ),
    }


def evaluate(config: dict[str, Any], *, run_id: str | None = None) -> dict[str, Any]:
    """Score the independent test predictions and apply the PoC decision rule."""
    with connect(config["database_path"]) as connection:
        if run_id:
            row = connection.execute(
                "SELECT * FROM ml_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM ml_runs ORDER BY created_utc DESC LIMIT 1"
            ).fetchone()
    if row is None:
        raise RuntimeError("No trained ML run is available")
    run = dict(row)
    existing_evaluation = run.get("evaluation_path")
    if existing_evaluation and Path(existing_evaluation).is_file():
        result = json.loads(
            Path(existing_evaluation).read_text(encoding="utf-8")
        )
        result["evaluation_reused"] = True
        return result
    predictions = pd.read_csv(run["predictions_path"])
    building_ids = predictions["building_id"].astype(str).tolist()
    labels = _fragility_labels(config["database_path"], building_ids)
    actual = labels[list(TARGET_COLUMNS)].to_numpy(float)
    eligible_frame = _eligible_ml_frame(config)
    eligible_indexed = eligible_frame.set_index("building_id")
    missing_metadata = [
        building_id
        for building_id in building_ids
        if building_id not in eligible_indexed.index
    ]
    if missing_metadata:
        raise RuntimeError(
            "Missing current SCWB/SPO metadata for "
            f"{len(missing_metadata)} independent-test buildings"
        )
    test_metadata = eligible_indexed.loc[building_ids].reset_index()
    evaluated_predictions = predictions.copy()
    evaluated_predictions["scwb_strength_ratio"] = test_metadata[
        "scwb_strength_ratio"
    ].to_numpy(float)
    evaluated_predictions["scwb_class"] = test_metadata[
        "scwb_class"
    ].astype(str).to_numpy()
    for index, target in enumerate(TARGET_COLUMNS):
        evaluated_predictions[f"actual_{target}"] = actual[:, index]
    evaluated_columns = [
        "building_id",
        "scwb_strength_ratio",
        "scwb_class",
    ]
    for target in TARGET_COLUMNS:
        evaluated_columns.extend(
            [
                f"actual_{target}",
                f"ann_{target}",
                f"ridge_{target}",
                f"mean_{target}",
            ]
        )
    evaluated_predictions = evaluated_predictions[evaluated_columns]
    model_metrics = {}
    for model_name in ("ann", "ridge", "mean"):
        predicted = predictions[
            [f"{model_name}_{target}" for target in TARGET_COLUMNS]
        ].to_numpy(float)
        model_metrics[model_name] = _model_metrics(actual, predicted)
    cv_path_value = run.get("cv_metrics_path")
    cv_path = (
        Path(cv_path_value)
        if cv_path_value
        else Path(run["predictions_path"]).parent / "cv_selection.json"
    )
    with cv_path.open("r", encoding="utf-8") as handle:
        cv_report = json.load(handle)
    with connect(config["database_path"]) as connection:
        marks = ",".join("?" for _ in building_ids)
        ida_runtime_s = float(
            connection.execute(
                f"SELECT COALESCE(SUM(runtime_s),0) FROM ida_runs "
                f"WHERE building_id IN ({marks})",
                building_ids,
            ).fetchone()[0]
        )
    ann_runtime_s = float(
        cv_report.get("ann_ensemble_test_inference_runtime_s", math.nan)
    )
    runtime_comparison = {
        "full_ida_accumulated_runtime_s": ida_runtime_s,
        "ann_test_inference_runtime_s": ann_runtime_s,
        "full_ida_runtime_per_test_building_s": ida_runtime_s
        / len(building_ids),
        "ann_inference_runtime_per_test_building_s": ann_runtime_s
        / len(building_ids),
        "compute_time_ratio": (
            ida_runtime_s / ann_runtime_s
            if ann_runtime_s > 0
            else None
        ),
        "note": "Accumulated NLTHA compute time is not parallel wall-clock time.",
    }
    ann_targets = model_metrics["ann"]["targets"]
    theta_targets = [f"theta_{state.lower()}_g" for state in LIMIT_STATES]
    beta_targets = [f"beta_{state.lower()}" for state in LIMIT_STATES]
    theta_indices = [0, 2, 4]
    ann_array = predictions[
        [f"ann_{target}" for target in TARGET_COLUMNS]
    ].to_numpy(float)
    ambiguity_settings = config["ml"]["mechanism_ambiguity_alarm"]
    all_eligible_ids = eligible_frame["building_id"].astype(str).tolist()
    all_labels = _fragility_labels(
        config["database_path"],
        all_eligible_ids,
    )
    mechanism_ambiguity = _mechanism_ambiguity_report(
        eligible_frame,
        all_labels,
        ambiguity_settings,
    )
    scwb_class_performance = _scwb_class_performance_report(
        test_metadata["scwb_class"].astype(str).to_numpy(),
        actual,
        ann_array,
        ambiguity_settings,
    )
    pooled_theta_median_ape = float(
        np.median(
            np.abs(
                (
                    ann_array[:, theta_indices]
                    - actual[:, theta_indices]
                )
                / actual[:, theta_indices]
            )
        )
    )
    criteria = {
        "planned_full_ida_dataset_reached": (
            len(eligible_frame)
            >= int(config["ida"]["planned_full_ida_buildings"])
        ),
        "theta_r2_positive": all(
            ann_targets[target]["r2"] > 0 for target in theta_targets
        ),
        "theta_median_ape_at_most_15_percent": (
            pooled_theta_median_ape <= 0.15
        ),
        "beta_mae_at_most_0p10": float(
            np.mean([ann_targets[target]["mae"] for target in beta_targets])
        )
        <= 0.10,
        "probability_mae_at_most_0p10": model_metrics["ann"][
            "mean_absolute_fragility_probability_error"
        ]
        <= 0.10,
        "ordering_violation_below_5_percent": model_metrics["ann"][
            "ordering_violation_rate"
        ]
        < 0.05,
        "ann_cv_nrmse_reduction_vs_ridge_at_least_5_percent": cv_report[
            "ann_reduction_vs_ridge"
        ]
        >= 0.05,
        "ann_optimization_converged": bool(
            cv_report.get("selected_cv_all_folds_converged", False)
            and cv_report.get("ensemble_all_converged", False)
        ),
        "no_material_cross_scwb_spo_cp_ambiguity": (
            not mechanism_ambiguity["alarm"]
        ),
        "scwb_class_cp_performance_acceptable": (
            not scwb_class_performance["alarm"]
        ),
    }
    promising = all(criteria.values())
    summary = {
        "run_id": run["run_id"],
        "independent_test_count": len(predictions),
        "eligible_labelled_building_count": len(eligible_frame),
        "planned_full_ida_building_count": int(
            config["ida"]["planned_full_ida_buildings"]
        ),
        "metrics": model_metrics,
        "pooled_theta_median_absolute_percentage_error": (
            pooled_theta_median_ape
        ),
        "criteria": criteria,
        "mechanism_ambiguity": mechanism_ambiguity,
        "scwb_class_performance": scwb_class_performance,
        "runtime_comparison": runtime_comparison,
        "promising_poc": promising,
        "conclusion": (
            "The predefined predictive-accuracy criteria were satisfied."
            if promising
            else "The computational pipeline is complete, but the predefined "
            "evidence threshold for a promising ANN PoC was not satisfied."
        ),
        "mechanism_ambiguity_recommendation": (
            [
                "Do not claim that the seven SPO inputs uniquely determine "
                "CP fragility across the retained SCWB range.",
                "Inspect the flagged Building-ID pairs and their beam/story "
                "plastic mechanisms.",
                "Run a documented sensitivity/ablation model adding "
                "scwb_strength_ratio or an explicit mechanism indicator.",
                "Compare the prespecified research-margin bands above the "
                "strict admission boundary R>1.00 without silently deleting "
                "unfavorable cases.",
            ]
            if (
                mechanism_ambiguity["alarm"]
                or scwb_class_performance["alarm"]
            )
            else [
                "Retain the seven-input primary model and continue reporting "
                "performance separately for every SCWB class."
            ]
        ),
    }
    evaluation_path = Path(run["metrics_path"]).parent / "evaluation.json"
    evaluated_predictions_path = (
        Path(run["predictions_path"]).parent
        / "evaluated_test_predictions.csv"
    )
    ambiguity_pairs_path = (
        Path(run["predictions_path"]).parent
        / "mechanism_ambiguity_pairs.csv"
    )
    evaluated_predictions.to_csv(
        evaluated_predictions_path,
        index=False,
        encoding="utf-8-sig",
    )
    ambiguity_columns = [
        "building_id_a",
        "scwb_class_a",
        "scwb_strength_ratio_a",
        "mechanism_class_a",
        "building_id_b",
        "scwb_class_b",
        "scwb_strength_ratio_b",
        "mechanism_class_b",
        "standardized_feature_distance",
        "cp_theta_ratio",
        "cp_beta_difference",
    ]
    pd.DataFrame(
        mechanism_ambiguity["flagged_pairs"],
        columns=ambiguity_columns,
    ).to_csv(
        ambiguity_pairs_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary["evaluated_predictions_path"] = str(
        evaluated_predictions_path.absolute()
    )
    summary["mechanism_ambiguity_pairs_path"] = str(
        ambiguity_pairs_path.absolute()
    )
    atomic_write_json(evaluation_path, summary)
    with transaction(config["database_path"]) as connection:
        connection.execute(
            """
            UPDATE ml_runs
            SET promising=?, summary_json=?, evaluation_path=?,
                evaluated_predictions_path=?, evaluated_utc=?
            WHERE run_id=?
            """,
            (
                int(promising),
                json.dumps(summary, ensure_ascii=False),
                str(evaluation_path.absolute()),
                str(evaluated_predictions_path.absolute()),
                datetime.now(timezone.utc).isoformat(),
                run["run_id"],
            ),
        )
    summary["evaluation_path"] = str(evaluation_path.absolute())
    return summary
