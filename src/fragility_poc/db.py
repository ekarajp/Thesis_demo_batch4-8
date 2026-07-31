"""SQLite persistence for resumable structural analyses."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS catalog_base_cases (
    base_case_id TEXT PRIMARY KEY,
    fc_ksc REAL NOT NULL,
    number_of_bays INTEGER NOT NULL,
    bay_width_m REAL NOT NULL,
    sdl_kg_m2 REAL NOT NULL,
    ll_kg_m2 REAL NOT NULL,
    stories INTEGER NOT NULL,
    story_height_m REAL NOT NULL,
    preliminary_design_valid INTEGER NOT NULL,
    invalid_reason TEXT,
    generated_model_count INTEGER NOT NULL,
    feasible_model_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS catalog_optimizer_audits (
    base_case_id TEXT NOT NULL
        REFERENCES catalog_base_cases(base_case_id) ON DELETE CASCADE,
    member_type TEXT NOT NULL CHECK(member_type IN ('beam', 'column')),
    tier INTEGER NOT NULL,
    strength_multiplier REAL NOT NULL,
    selected_objective_cost_per_m REAL NOT NULL,
    exact_verified_objective REAL NOT NULL,
    firefly_best_objective REAL NOT NULL,
    firefly_best_gap REAL NOT NULL,
    multi_start_runs INTEGER NOT NULL,
    multi_start_success_count INTEGER NOT NULL,
    multi_start_success_rate REAL NOT NULL,
    multi_start_best_gap REAL NOT NULL,
    multi_start_mean_gap REAL NOT NULL,
    multi_start_median_gap REAL NOT NULL,
    multi_start_worst_gap REAL NOT NULL,
    total_completed_iterations INTEGER NOT NULL,
    audit_json TEXT NOT NULL,
    PRIMARY KEY(base_case_id, member_type, tier)
);

CREATE TABLE IF NOT EXISTS catalog_design_cache (
    design_cache_key TEXT PRIMARY KEY,
    model_schema_version TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS building_catalog (
    building_id TEXT PRIMARY KEY,
    base_case_id TEXT NOT NULL,
    model_hash TEXT NOT NULL UNIQUE,
    queue_rank INTEGER,
    selected INTEGER NOT NULL DEFAULT 0,
    valid INTEGER NOT NULL,
    invalid_reason TEXT,
    fc_ksc REAL NOT NULL,
    number_of_bays INTEGER NOT NULL,
    bay_width_m REAL NOT NULL,
    sdl_kg_m2 REAL NOT NULL,
    ll_kg_m2 REAL NOT NULL,
    stories INTEGER NOT NULL,
    story_height_m REAL NOT NULL,
    slab_thickness_m REAL NOT NULL,
    beam_tier INTEGER NOT NULL,
    column_tier INTEGER NOT NULL,
    beam_strength_multiplier REAL NOT NULL,
    column_strength_multiplier REAL NOT NULL,
    beam_b_m REAL NOT NULL,
    beam_h_m REAL NOT NULL,
    beam_bars_per_face INTEGER NOT NULL,
    beam_bar_diameter_m REAL NOT NULL,
    beam_phi_mn_knm REAL NOT NULL,
    beam_phi_vn_kn REAL NOT NULL,
    column_b_m REAL NOT NULL,
    column_h_m REAL NOT NULL,
    column_bar_count INTEGER NOT NULL,
    column_bar_diameter_m REAL NOT NULL,
    column_phi_pn_kn REAL NOT NULL,
    column_phi_mn_knm REAL NOT NULL,
    axial_ratio REAL NOT NULL,
    scwb_strength_ratio REAL,
    scwb_class TEXT,
    dead_load_kn_m2 REAL NOT NULL,
    live_load_kn_m2 REAL NOT NULL,
    floor_mass_kn_s2_m REAL NOT NULL,
    design_metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_building_selected
ON building_catalog(selected, queue_rank);

CREATE TABLE IF NOT EXISTS ground_motion_catalog (
    pair_id TEXT PRIMARY KEY,
    source_set TEXT NOT NULL,
    conditioning_period_s REAL,
    component_x_path TEXT NOT NULL,
    component_y_path TEXT NOT NULL,
    units TEXT NOT NULL,
    dt_s REAL NOT NULL,
    npts INTEGER NOT NULL,
    duration_s REAL NOT NULL,
    pga_x_g REAL NOT NULL,
    pga_y_g REAL NOT NULL,
    sha256_x TEXT NOT NULL,
    sha256_y TEXT NOT NULL,
    physical_pair_hash TEXT,
    raw_source_path TEXT,
    raw_source_sha256 TEXT,
    source_metadata_json TEXT,
    valid INTEGER NOT NULL,
    validation_message TEXT
);

CREATE TABLE IF NOT EXISTS building_ground_motion_selection (
    building_id TEXT NOT NULL REFERENCES building_catalog(building_id),
    pair_id TEXT NOT NULL REFERENCES ground_motion_catalog(pair_id),
    t1_s REAL NOT NULL,
    selected_cms_periods_json TEXT NOT NULL,
    selection_reason TEXT NOT NULL,
    analysis_role TEXT NOT NULL DEFAULT 'fragility_primary',
    scale_factor_policy TEXT NOT NULL DEFAULT 'incremental_ida',
    PRIMARY KEY(building_id, pair_id)
);

CREATE TABLE IF NOT EXISTS spo_features (
    building_id TEXT PRIMARY KEY REFERENCES building_catalog(building_id),
    t1_s REAL NOT NULL,
    t2_s REAL,
    gravity_error REAL NOT NULL,
    x_y_period_error REAL,
    x_mode INTEGER,
    y_mode INTEGER,
    x_mode_effective_mass_ratio REAL,
    y_mode_effective_mass_ratio REAL,
    cumulative_x_effective_mass_ratio REAL,
    cumulative_y_effective_mass_ratio REAL,
    modal_identification_method TEXT,
    vy_kn REAL NOT NULL,
    dy_m REAL NOT NULL,
    vc_kn REAL NOT NULL,
    dc_m REAL NOT NULL,
    vu_kn REAL NOT NULL,
    du_m REAL NOT NULL,
    energy_error REAL NOT NULL,
    energy_error_before_adjustment REAL,
    pre_capping_energy_error REAL,
    pre_capping_energy_error_before_adjustment REAL,
    post_capping_energy_error REAL,
    normalized_rmse REAL,
    normalized_rmse_before_adjustment REAL,
    trilinear_quality_valid INTEGER,
    initial_stiffness_kn_m REAL,
    postyield_stiffness_fit_kn_m REAL,
    postyield_stiffness_final_kn_m REAL,
    stiffness_loss_displacement_m REAL,
    energy_adjusted INTEGER,
    slope_adjustment_reason TEXT,
    slope_adjustment_score_before REAL,
    slope_adjustment_score_after REAL,
    idealization_method TEXT,
    postpeak_reached INTEGER NOT NULL,
    run_end_displacement_m REAL,
    run_end_shear_kn REAL,
    run_end_shear_ratio_vc REAL,
    spo_termination_reason TEXT,
    maximum_recovery_level INTEGER,
    maximum_step_reduction_level INTEGER,
    minimum_attempted_step_m REAL,
    requested_displacement_step_m REAL,
    effective_displacement_step_m REAL,
    adaptive_refinement_level INTEGER,
    adaptive_refinement_reason TEXT,
    failed_step_attempts INTEGER,
    accepted_step_count INTEGER,
    last_converged_load_factor REAL,
    collapse_reached INTEGER,
    collapse_classification TEXT,
    collapse_interpretation TEXT,
    collapse_displacement_m REAL,
    collapse_roof_drift REAL,
    analysis_guard_triggered INTEGER,
    mechanism_history_path TEXT,
    mechanism_class_at_capping TEXT,
    mechanism_class_at_ultimate TEXT,
    mechanism_class_at_run_end TEXT,
    capping_beam_yielded_end_fraction REAL,
    capping_column_yielded_end_fraction REAL,
    capping_max_story_column_yielded_end_fraction REAL,
    ultimate_beam_yielded_end_fraction REAL,
    ultimate_column_yielded_end_fraction REAL,
    ultimate_max_story_column_yielded_end_fraction REAL,
    analysis_signature TEXT,
    curve_path TEXT NOT NULL,
    runtime_s REAL NOT NULL,
    valid INTEGER NOT NULL,
    validation_message TEXT
);

CREATE TABLE IF NOT EXISTS ida_runs (
    building_id TEXT NOT NULL REFERENCES building_catalog(building_id),
    pair_id TEXT NOT NULL REFERENCES ground_motion_catalog(pair_id),
    target_im_g REAL NOT NULL,
    scale_factor REAL NOT NULL,
    achieved_im_g REAL NOT NULL,
    max_midr REAL NOT NULL,
    status TEXT NOT NULL,
    dynamic_instability INTEGER NOT NULL,
    runtime_s REAL NOT NULL,
    recovery_level INTEGER NOT NULL,
    soft_im_warning_exceeded INTEGER NOT NULL DEFAULT 0,
    scale_factor_warning_exceeded INTEGER NOT NULL DEFAULT 0,
    analysis_signature TEXT,
    recovery_policy_version TEXT,
    result_path TEXT NOT NULL,
    PRIMARY KEY(building_id, pair_id, target_im_g)
);

CREATE TABLE IF NOT EXISTS ida_capacities (
    building_id TEXT NOT NULL REFERENCES building_catalog(building_id),
    pair_id TEXT NOT NULL REFERENCES ground_motion_catalog(pair_id),
    limit_state TEXT NOT NULL,
    threshold_midr REAL NOT NULL,
    capacity_im_g REAL NOT NULL,
    censored INTEGER NOT NULL,
    censoring TEXT NOT NULL DEFAULT 'none',
    lower_im_g REAL,
    upper_im_g REAL,
    upper_bound_status TEXT,
    upper_bound_certified INTEGER NOT NULL DEFAULT 0,
    analysis_signature TEXT,
    PRIMARY KEY(building_id, pair_id, limit_state)
);

CREATE TABLE IF NOT EXISTS ida_curve_diagnostics (
    building_id TEXT NOT NULL REFERENCES building_catalog(building_id),
    pair_id TEXT NOT NULL REFERENCES ground_motion_catalog(pair_id),
    controller_mode TEXT NOT NULL,
    controller_version TEXT NOT NULL,
    prediction_source TEXT NOT NULL,
    predicted_io_g REAL,
    predicted_ls_g REAL,
    predicted_cp_g REAL,
    point_count INTEGER NOT NULL,
    seed_point_count INTEGER NOT NULL,
    expansion_point_count INTEGER NOT NULL,
    refinement_point_count INTEGER NOT NULL,
    maximum_final_bracket_width REAL,
    maximum_capacity_error_bound REAL,
    review_status TEXT,
    review_flag_count INTEGER NOT NULL DEFAULT 0,
    codex_review_required INTEGER NOT NULL DEFAULT 0,
    manually_requested_point_count INTEGER NOT NULL DEFAULT 0,
    recommended_targets_json TEXT,
    model_path TEXT,
    model_sha256 TEXT,
    details_json TEXT NOT NULL,
    analysis_signature TEXT NOT NULL,
    PRIMARY KEY(building_id, pair_id)
);

CREATE TABLE IF NOT EXISTS ida_manual_point_requests (
    building_id TEXT NOT NULL REFERENCES building_catalog(building_id),
    pair_id TEXT NOT NULL REFERENCES ground_motion_catalog(pair_id),
    target_im_g REAL NOT NULL,
    requester TEXT NOT NULL,
    reason TEXT NOT NULL,
    requested_utc TEXT NOT NULL,
    status TEXT NOT NULL,
    result_status TEXT,
    result_path TEXT,
    completed_utc TEXT,
    PRIMARY KEY(building_id, pair_id, target_im_g)
);

CREATE TABLE IF NOT EXISTS fragility_targets (
    building_id TEXT PRIMARY KEY REFERENCES building_catalog(building_id),
    theta_io_g REAL NOT NULL,
    beta_io REAL NOT NULL,
    theta_ls_g REAL NOT NULL,
    beta_ls REAL NOT NULL,
    theta_cp_g REAL NOT NULL,
    beta_cp REAL NOT NULL,
    n_pairs INTEGER NOT NULL,
    n_censored_io INTEGER NOT NULL,
    n_censored_ls INTEGER NOT NULL,
    n_censored_cp INTEGER NOT NULL,
    bootstrap_json TEXT NOT NULL,
    valid INTEGER NOT NULL,
    validation_message TEXT,
    source_signature TEXT
);

CREATE TABLE IF NOT EXISTS ml_split (
    building_id TEXT PRIMARY KEY REFERENCES building_catalog(building_id),
    split TEXT NOT NULL,
    fold INTEGER,
    t1_quartile INTEGER NOT NULL,
    random_seed INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ml_split_manifest (
    manifest_key INTEGER PRIMARY KEY CHECK(manifest_key = 1),
    created_utc TEXT NOT NULL,
    source_signature TEXT NOT NULL,
    source_count INTEGER NOT NULL,
    random_seed INTEGER NOT NULL,
    cv_folds INTEGER NOT NULL,
    test_fraction REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS ml_runs (
    run_id TEXT PRIMARY KEY,
    created_utc TEXT NOT NULL,
    model_path TEXT NOT NULL,
    metrics_path TEXT NOT NULL,
    predictions_path TEXT NOT NULL,
    selected_architecture TEXT NOT NULL,
    selected_alpha REAL NOT NULL,
    development_count INTEGER NOT NULL,
    test_count INTEGER NOT NULL,
    promising INTEGER NOT NULL,
    summary_json TEXT NOT NULL,
    cv_metrics_path TEXT,
    evaluation_path TEXT,
    evaluated_predictions_path TEXT,
    evaluated_utc TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_failures (
    failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    building_id TEXT,
    pair_id TEXT,
    created_utc TEXT NOT NULL,
    error_type TEXT NOT NULL,
    message TEXT NOT NULL,
    details_json TEXT,
    resolved INTEGER NOT NULL DEFAULT 0
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=60.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 60000")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize(path: str | Path) -> None:
    with connect(path) as connection:
        connection.executescript(SCHEMA)
        ground_motion_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(ground_motion_catalog)"
            )
        }
        migrations = {
            "raw_source_path": "TEXT",
            "raw_source_sha256": "TEXT",
            "source_metadata_json": "TEXT",
            "physical_pair_hash": "TEXT",
        }
        for column, definition in migrations.items():
            if column not in ground_motion_columns:
                connection.execute(
                    f"ALTER TABLE ground_motion_catalog "
                    f"ADD COLUMN {column} {definition}"
                )
        selection_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(building_ground_motion_selection)"
            )
        }
        selection_migrations = {
            "analysis_role": (
                "TEXT NOT NULL DEFAULT 'fragility_primary'"
            ),
            "scale_factor_policy": (
                "TEXT NOT NULL DEFAULT 'incremental_ida'"
            ),
        }
        for column, definition in selection_migrations.items():
            if column not in selection_columns:
                connection.execute(
                    "ALTER TABLE building_ground_motion_selection "
                    f"ADD COLUMN {column} {definition}"
                )
        connection.execute(
            """
            UPDATE building_ground_motion_selection
            SET analysis_role='event_specific_sensitivity',
                scale_factor_policy='as_recorded_sf1'
            WHERE pair_id IN (
                SELECT pair_id FROM ground_motion_catalog
                WHERE source_set LIKE 'PWSA%'
            )
            """
        )
        building_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(building_catalog)"
            )
        }
        building_migrations = {
            "number_of_bays": "INTEGER NOT NULL DEFAULT 1",
            "scwb_strength_ratio": "REAL",
            "scwb_class": "TEXT",
        }
        for column, definition in building_migrations.items():
            if column not in building_columns:
                connection.execute(
                    f"ALTER TABLE building_catalog "
                    f"ADD COLUMN {column} {definition}"
                )
        base_case_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(catalog_base_cases)"
            )
        }
        if "number_of_bays" not in base_case_columns:
            connection.execute(
                "ALTER TABLE catalog_base_cases "
                "ADD COLUMN number_of_bays INTEGER NOT NULL DEFAULT 1"
            )
        spo_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(spo_features)"
            )
        }
        spo_migrations = {
            "energy_error_before_adjustment": "REAL",
            "x_mode": "INTEGER",
            "y_mode": "INTEGER",
            "x_mode_effective_mass_ratio": "REAL",
            "y_mode_effective_mass_ratio": "REAL",
            "cumulative_x_effective_mass_ratio": "REAL",
            "cumulative_y_effective_mass_ratio": "REAL",
            "modal_identification_method": "TEXT",
            "pre_capping_energy_error": "REAL",
            "pre_capping_energy_error_before_adjustment": "REAL",
            "post_capping_energy_error": "REAL",
            "normalized_rmse": "REAL",
            "normalized_rmse_before_adjustment": "REAL",
            "trilinear_quality_valid": "INTEGER",
            "initial_stiffness_kn_m": "REAL",
            "postyield_stiffness_fit_kn_m": "REAL",
            "postyield_stiffness_final_kn_m": "REAL",
            "stiffness_loss_displacement_m": "REAL",
            "energy_adjusted": "INTEGER",
            "slope_adjustment_reason": "TEXT",
            "slope_adjustment_score_before": "REAL",
            "slope_adjustment_score_after": "REAL",
            "idealization_method": "TEXT",
            "run_end_displacement_m": "REAL",
            "run_end_shear_kn": "REAL",
            "run_end_shear_ratio_vc": "REAL",
            "spo_termination_reason": "TEXT",
            "maximum_recovery_level": "INTEGER",
            "maximum_step_reduction_level": "INTEGER",
            "minimum_attempted_step_m": "REAL",
            "requested_displacement_step_m": "REAL",
            "effective_displacement_step_m": "REAL",
            "adaptive_refinement_level": "INTEGER",
            "adaptive_refinement_reason": "TEXT",
            "failed_step_attempts": "INTEGER",
            "accepted_step_count": "INTEGER",
            "last_converged_load_factor": "REAL",
            "collapse_reached": "INTEGER",
            "collapse_classification": "TEXT",
            "collapse_interpretation": "TEXT",
            "collapse_displacement_m": "REAL",
            "collapse_roof_drift": "REAL",
            "analysis_guard_triggered": "INTEGER",
            "mechanism_history_path": "TEXT",
            "mechanism_class_at_capping": "TEXT",
            "mechanism_class_at_ultimate": "TEXT",
            "mechanism_class_at_run_end": "TEXT",
            "capping_beam_yielded_end_fraction": "REAL",
            "capping_column_yielded_end_fraction": "REAL",
            "capping_max_story_column_yielded_end_fraction": "REAL",
            "ultimate_beam_yielded_end_fraction": "REAL",
            "ultimate_column_yielded_end_fraction": "REAL",
            "ultimate_max_story_column_yielded_end_fraction": "REAL",
            "analysis_signature": "TEXT",
        }
        for column, definition in spo_migrations.items():
            if column not in spo_columns:
                connection.execute(
                    f"ALTER TABLE spo_features "
                    f"ADD COLUMN {column} {definition}"
                )
        ida_capacity_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(ida_capacities)"
            )
        }
        if "censoring" not in ida_capacity_columns:
            connection.execute(
                "ALTER TABLE ida_capacities "
                "ADD COLUMN censoring TEXT NOT NULL DEFAULT 'none'"
            )
            connection.execute(
                "UPDATE ida_capacities SET censoring="
                "CASE WHEN censored=1 THEN 'right' ELSE 'none' END"
            )
        if "upper_bound_status" not in ida_capacity_columns:
            connection.execute(
                "ALTER TABLE ida_capacities "
                "ADD COLUMN upper_bound_status TEXT"
            )
        if "upper_bound_certified" not in ida_capacity_columns:
            connection.execute(
                "ALTER TABLE ida_capacities "
                "ADD COLUMN upper_bound_certified INTEGER NOT NULL DEFAULT 0"
            )
        if "analysis_signature" not in ida_capacity_columns:
            connection.execute(
                "ALTER TABLE ida_capacities "
                "ADD COLUMN analysis_signature TEXT"
            )
        ida_run_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(ida_runs)")
        }
        if "analysis_signature" not in ida_run_columns:
            connection.execute(
                "ALTER TABLE ida_runs ADD COLUMN analysis_signature TEXT"
            )
        if "recovery_policy_version" not in ida_run_columns:
            connection.execute(
                "ALTER TABLE ida_runs ADD COLUMN recovery_policy_version TEXT"
            )
        if "soft_im_warning_exceeded" not in ida_run_columns:
            connection.execute(
                "ALTER TABLE ida_runs ADD COLUMN "
                "soft_im_warning_exceeded INTEGER NOT NULL DEFAULT 0"
            )
        if "scale_factor_warning_exceeded" not in ida_run_columns:
            connection.execute(
                "ALTER TABLE ida_runs ADD COLUMN "
                "scale_factor_warning_exceeded INTEGER NOT NULL DEFAULT 0"
            )
        ida_diagnostic_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(ida_curve_diagnostics)"
            )
        }
        ida_diagnostic_migrations = {
            "maximum_capacity_error_bound": "REAL",
            "review_status": "TEXT",
            "review_flag_count": "INTEGER NOT NULL DEFAULT 0",
            "codex_review_required": "INTEGER NOT NULL DEFAULT 0",
            "manually_requested_point_count": "INTEGER NOT NULL DEFAULT 0",
            "recommended_targets_json": "TEXT",
        }
        for column, definition in ida_diagnostic_migrations.items():
            if column not in ida_diagnostic_columns:
                connection.execute(
                    "ALTER TABLE ida_curve_diagnostics "
                    f"ADD COLUMN {column} {definition}"
                )
        ml_run_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(ml_runs)")
        }
        ml_run_migrations = {
            "cv_metrics_path": "TEXT",
            "evaluation_path": "TEXT",
            "evaluated_predictions_path": "TEXT",
            "evaluated_utc": "TEXT",
        }
        for column, definition in ml_run_migrations.items():
            if column not in ml_run_columns:
                connection.execute(
                    f"ALTER TABLE ml_runs ADD COLUMN {column} {definition}"
                )
        fragility_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(fragility_targets)"
            )
        }
        if "source_signature" not in fragility_columns:
            connection.execute(
                "ALTER TABLE fragility_targets "
                "ADD COLUMN source_signature TEXT"
            )


@contextmanager
def transaction(path: str | Path) -> Iterator[sqlite3.Connection]:
    connection = connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def upsert_many(
    connection: sqlite3.Connection,
    table: str,
    rows: Iterable[dict[str, Any]],
    conflict_columns: tuple[str, ...],
) -> int:
    materialized = list(rows)
    if not materialized:
        return 0
    columns = list(materialized[0])
    placeholders = ", ".join("?" for _ in columns)
    update_columns = [column for column in columns if column not in conflict_columns]
    conflict = ", ".join(conflict_columns)
    updates = ", ".join(f"{column}=excluded.{column}" for column in update_columns)
    sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
    )
    values = [
        tuple(
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
            for value in (row[column] for column in columns)
        )
        for row in materialized
    ]
    connection.executemany(sql, values)
    return len(materialized)


def record_pipeline_failure(
    path: str | Path,
    *,
    stage: str,
    error: BaseException | str,
    building_id: str | None = None,
    pair_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Persist an unresolved pipeline failure so a model cannot disappear silently."""
    error_type = (
        type(error).__name__ if isinstance(error, BaseException) else "QualityControl"
    )
    message = str(error)
    with transaction(path) as connection:
        connection.execute(
            """
            INSERT INTO pipeline_failures
                (stage, building_id, pair_id, created_utc, error_type,
                 message, details_json, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                stage,
                building_id,
                pair_id,
                datetime.now(timezone.utc).isoformat(),
                error_type,
                message,
                json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
            ),
        )


def resolve_pipeline_failures(
    path: str | Path,
    *,
    stage: str,
    building_id: str | None = None,
    pair_id: str | None = None,
) -> int:
    """Mark earlier failures resolved after the same analysis target succeeds."""
    clauses = ["stage=?", "resolved=0"]
    parameters: list[Any] = [stage]
    if building_id is not None:
        clauses.append("building_id=?")
        parameters.append(building_id)
    if pair_id is not None:
        clauses.append("pair_id=?")
        parameters.append(pair_id)
    with transaction(path) as connection:
        cursor = connection.execute(
            f"UPDATE pipeline_failures SET resolved=1 WHERE {' AND '.join(clauses)}",
            parameters,
        )
        return int(cursor.rowcount)
