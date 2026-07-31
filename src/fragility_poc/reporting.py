"""Reproducible consolidated Excel reporting from current signed results."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import connect, initialize
from .ida import ida_curve_analysis_signature, nltha_analysis_signature
from .io_utils import atomic_write_json, stable_hash


def _json_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = []
    for row in rows:
        clean.append(
            {
                key: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
        )
    return clean


def _presentation_safe_endpoint_names(
    row: dict[str, Any],
) -> dict[str, Any]:
    """Avoid presenting flexural-model endpoints as physical collapse."""
    renamed = dict(row)
    if "collapse_roof_drift" in renamed:
        renamed["model_end_roof_drift"] = renamed.pop(
            "collapse_roof_drift"
        )
    if "collapse_classification" in renamed:
        renamed["model_endpoint_classification"] = renamed.pop(
            "collapse_classification"
        )
    return renamed


def _current_sensitivity_rows(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return compact current-schema sensitivity tables for one workbook."""
    from .sensitivity import (
        MODELING_SENSITIVITY_SCHEMA_VERSION,
        SENSITIVITY_SCHEMA_VERSION,
    )

    output_dir = Path(config["output_dir"])
    hinge_path = output_dir / "plastic_hinge_length_sensitivity.json"
    modeling_path = output_dir / "modeling_sensitivity.json"
    hinge_rows: list[dict[str, Any]] = []
    modeling_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    if hinge_path.is_file():
        hinge = json.loads(hinge_path.read_text(encoding="utf-8"))
        if hinge.get("schema_version") == SENSITIVITY_SCHEMA_VERSION:
            keep = (
                "building_id",
                "scwb_class",
                "scwb_strength_ratio",
                "plastic_hinge_length_factor",
                "beam_lp_m",
                "column_lp_m",
                "t1_s",
                "vy_kn",
                "dy_m",
                "vc_kn",
                "dc_m",
                "vu_kn",
                "du_m",
                "collapse_roof_drift",
                "mechanism_class_at_capping",
                "mechanism_class_at_ultimate",
                "mechanism_class_at_run_end",
                "valid",
            )
            hinge_rows = [
                _presentation_safe_endpoint_names(
                    {key: row.get(key) for key in keep}
                )
                for row in hinge.get("results", [])
            ]
            summary["hinge_sensitivity"] = hinge.get("summary", {})
    if modeling_path.is_file():
        modeling = json.loads(modeling_path.read_text(encoding="utf-8"))
        if (
            modeling.get("schema_version")
            == MODELING_SENSITIVITY_SCHEMA_VERSION
        ):
            spo_keep = (
                "building_id",
                "scwb_class",
                "scwb_strength_ratio",
                "scenario_group",
                "scenario_name",
                "t1_s",
                "vy_kn",
                "dy_m",
                "vc_kn",
                "dc_m",
                "vu_kn",
                "du_m",
                "collapse_roof_drift",
                "valid",
            )
            modeling_rows.extend(
                _presentation_safe_endpoint_names(
                    {
                        "analysis_type": "SPO",
                        **{key: row.get(key) for key in spo_keep},
                    }
                )
                for row in modeling.get("spo_results", [])
            )
            damping_keep = (
                "building_id",
                "scwb_class",
                "scwb_strength_ratio",
                "pair_id",
                "damping_ratio",
                "target_im_g",
                "max_midr",
                "max_midr_change_from_5pct_damping_pct",
                "status",
                "valid",
            )
            modeling_rows.extend(
                {
                    "analysis_type": "NLTHA damping",
                    **{key: row.get(key) for key in damping_keep},
                }
                for row in modeling.get("damping_results", [])
            )
            summary["modeling_sensitivity"] = {
                "maximum_absolute_spo_change_percent": modeling.get(
                    "maximum_absolute_spo_change_percent"
                ),
                "scenario_maximum_absolute_spo_change_percent": modeling.get(
                    "scenario_maximum_absolute_spo_change_percent",
                    {},
                ),
                "fine_mesh_convergence_max_change_percent": modeling.get(
                    "fine_mesh_convergence_max_change_percent"
                ),
                "fine_mesh_convergence_tolerance_percent": modeling.get(
                    "fine_mesh_convergence_tolerance_percent"
                ),
                "fine_mesh_convergence_within_tolerance": modeling.get(
                    "fine_mesh_convergence_within_tolerance"
                ),
                "maximum_absolute_damping_midr_change_percent": modeling.get(
                    "maximum_absolute_damping_midr_change_percent"
                ),
                "all_spo_cases_valid": modeling.get("all_spo_cases_valid"),
                "all_damping_cases_valid": modeling.get(
                    "all_damping_cases_valid"
                ),
                "damping_sensitivity_enabled": modeling.get(
                    "damping_sensitivity_enabled"
                ),
                "production_modal_damping_ratio": modeling.get(
                    "production_modal_damping_ratio"
                ),
            }
    return hinge_rows, modeling_rows, summary


def _current_ida_rows(
    config: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    with connect(config["database_path"]) as connection:
        run_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT r.*, b.model_hash, sp.t1_s,
                       g.sha256_x, g.sha256_y
                FROM ida_runs r
                JOIN building_catalog b USING(building_id)
                JOIN spo_features sp USING(building_id)
                JOIN ground_motion_catalog g USING(pair_id)
                WHERE b.selected=1 AND b.valid=1 AND sp.valid=1 AND g.valid=1
                ORDER BY b.queue_rank, r.pair_id, r.target_im_g
                """
            )
        ]
        capacity_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT c.*, b.model_hash, sp.t1_s,
                       g.sha256_x, g.sha256_y
                FROM ida_capacities c
                JOIN building_catalog b USING(building_id)
                JOIN spo_features sp USING(building_id)
                JOIN ground_motion_catalog g USING(pair_id)
                JOIN building_ground_motion_selection s
                  ON s.building_id=c.building_id
                 AND s.pair_id=c.pair_id
                WHERE b.selected=1 AND b.valid=1 AND sp.valid=1 AND g.valid=1
                ORDER BY b.queue_rank, c.pair_id, c.limit_state
                """
            )
        ]
        diagnostic_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT d.*, b.model_hash, sp.t1_s,
                       g.sha256_x, g.sha256_y
                FROM ida_curve_diagnostics d
                JOIN building_catalog b USING(building_id)
                JOIN spo_features sp USING(building_id)
                JOIN ground_motion_catalog g USING(pair_id)
                WHERE b.selected=1 AND b.valid=1 AND sp.valid=1 AND g.valid=1
                ORDER BY b.queue_rank, d.pair_id
                """
            )
        ]
    current_runs = [
        row
        for row in run_rows
        if row.get("analysis_signature")
        == nltha_analysis_signature(
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
    ]
    current_capacities = [
        row
        for row in capacity_rows
        if row.get("analysis_signature")
        == ida_curve_analysis_signature(
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
            config,
        )
    ]
    current_diagnostics = [
        row
        for row in diagnostic_rows
        if row.get("analysis_signature")
        == ida_curve_analysis_signature(
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
            config,
        )
    ]
    removable = {"model_hash", "t1_s", "sha256_x", "sha256_y"}
    return (
        [
            {key: value for key, value in row.items() if key not in removable}
            for row in current_runs
        ],
        [
            {key: value for key, value in row.items() if key not in removable}
            for row in current_capacities
        ],
        [
            {key: value for key, value in row.items() if key not in removable}
            for row in current_diagnostics
        ],
    )


def _migration_smoke_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact, isolated Draft-5 migration evidence for the workbook."""
    database_path = Path(config["database_path"])
    smoke_database = database_path.with_name("migration_smoke.sqlite")
    if not smoke_database.is_file():
        return []
    with connect(smoke_database) as connection:
        selected_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM building_catalog
                WHERE selected=1 AND valid=1
                """
            ).fetchone()[0]
        )
        valid_gm_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM ground_motion_catalog WHERE valid=1
                """
            ).fetchone()[0]
        )
        unique_physical_gm_count = int(
            connection.execute(
                """
                SELECT COUNT(DISTINCT physical_pair_hash)
                FROM ground_motion_catalog
                WHERE valid=1
                """
            ).fetchone()[0]
        )
        spo_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT building_id, t1_s, energy_error,
                       pre_capping_energy_error,
                       post_capping_energy_error, normalized_rmse,
                       postpeak_reached, spo_termination_reason,
                       valid, runtime_s
                FROM spo_features
                ORDER BY building_id
                """
            )
        ]
    rows: list[dict[str, Any]] = [
        {
            "stage": "Catalog migration",
            "case": "Isolated smoke queue",
            "status": "PASS" if selected_count == 30 else "REVIEW",
            "value": selected_count,
            "unit": "buildings",
            "detail": (
                "Draft-5 five-storey migration database; separate from the "
                "375-building production queue."
            ),
        },
        {
            "stage": "Ground motion",
            "case": "Validated catalogue",
            "status": "PASS" if valid_gm_count == 24 else "REVIEW",
            "value": valid_gm_count,
            "unit": "catalogue pairs",
            "detail": (
                f"{unique_physical_gm_count} unique physical X-Y pairs after "
                "checksum deduplication."
            ),
        },
    ]
    for row in spo_rows:
        rows.append(
            {
                "stage": "Modal + SPO",
                "case": row["building_id"],
                "status": (
                    "PASS"
                    if bool(row["valid"]) and bool(row["postpeak_reached"])
                    else "REVIEW"
                ),
                "value": row["t1_s"],
                "unit": "T1 (s)",
                "detail": (
                    "total/pre/post/NRMSE="
                    f"{100.0 * float(row['energy_error']):.3f}%/"
                    f"{100.0 * float(row['pre_capping_energy_error']):.3f}%/"
                    f"{100.0 * float(row['post_capping_energy_error']):.3f}%/"
                    f"{100.0 * float(row['normalized_rmse']):.3f}%; "
                    f"runtime={float(row['runtime_s']):.1f}s; "
                    f"end={row['spo_termination_reason']}"
                ),
            }
        )
    xy_path = (
        Path(config["output_dir"])
        / "migration_smoke"
        / "spo_xy_equivalence.json"
    )
    if xy_path.is_file():
        xy = json.loads(xy_path.read_text(encoding="utf-8"))
        maximum_curve_nrmse = max(
            (
                float(item.get("curve_nrmse", 0.0))
                for item in xy.get("checks", [])
            ),
            default=0.0,
        )
        rows.append(
            {
                "stage": "X-Y symmetry",
                "case": "Three multi-bay topologies",
                "status": "PASS" if xy.get("valid") else "REVIEW",
                "value": maximum_curve_nrmse,
                "unit": "curve NRMSE",
                "detail": (
                    "Directional equivalence check using effective modal mass "
                    "mode identification."
                ),
            }
        )
    # Building IDs are content hashes and therefore change whenever the
    # Draft-5 catalogue/model schema changes. Discover the current isolated
    # checkpoint instead of silently depending on a stale historical ID.
    nltha_candidates = sorted(
        (
            Path(config["run_dir"])
            / "migration_smoke"
            / "ida"
        ).glob("*/*/im_*.json")
    )
    nltha_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in nltha_candidates
    ]
    nltha = next(
        (
            payload
            for payload in nltha_payloads
            if payload.get("status") == "success"
        ),
        nltha_payloads[0] if nltha_payloads else None,
    )
    if nltha is not None:
        rows.append(
            {
                "stage": "NLTHA checkpoint",
                "case": (
                    f"{nltha.get('building_id')} / {nltha.get('pair_id')}"
                ),
                "status": (
                    "PASS" if nltha.get("status") == "success" else "REVIEW"
                ),
                "value": nltha.get("max_midr"),
                "unit": "MIDR",
                "detail": (
                    f"target/achieved={nltha.get('target_im_g')}/"
                    f"{nltha.get('achieved_im_g')} g; "
                    f"scale={float(nltha.get('scale_factor', 0.0)):.6g}; "
                    f"scaling error={float(nltha.get('scaling_error', 0.0)):.3g}; "
                    f"recovery level={nltha.get('recovery_level')}; "
                    f"runtime={float(nltha.get('runtime_s', 0.0)):.1f}s."
                ),
            }
        )
    return rows


def _report_payload(config: dict[str, Any]) -> dict[str, Any]:
    from .catalog import (
        CATALOG_MODEL_SCHEMA_VERSION,
        SCWB_RESEARCH_CLASSES,
    )
    from .demo import _current_fragility_rows, _current_spo_features

    current_spo = _current_spo_features(config)
    current_fragility = _current_fragility_rows(config, current_spo)
    current_ida, current_capacities, current_ida_diagnostics = (
        _current_ida_rows(config)
    )
    migration_smoke = _migration_smoke_rows(config)
    hinge_sensitivity, modeling_sensitivity, sensitivity_summary = (
        _current_sensitivity_rows(config)
    )
    with connect(config["database_path"]) as connection:
        buildings = [
            dict(row)
            for row in connection.execute(
                """
                SELECT building_id, base_case_id, queue_rank, model_hash,
                       fc_ksc, number_of_bays, bay_width_m,
                       sdl_kg_m2, ll_kg_m2,
                       slab_thickness_m, beam_tier, column_tier,
                       beam_b_m, beam_h_m, beam_bars_per_face,
                       beam_bar_diameter_m, beam_phi_vn_kn,
                       column_b_m, column_h_m,
                       column_bar_count, column_bar_diameter_m,
                       axial_ratio, scwb_strength_ratio, scwb_class,
                       dead_load_kn_m2, live_load_kn_m2, floor_mass_kn_s2_m,
                       design_metadata_json
                FROM building_catalog
                WHERE selected=1 AND valid=1
                ORDER BY queue_rank
                """
            )
        ]
        optimizer = [
            dict(row)
            for row in connection.execute(
                """
                SELECT base_case_id, member_type, tier,
                       strength_multiplier,
                       selected_objective_cost_per_m,
                       exact_verified_objective,
                       firefly_best_objective, firefly_best_gap,
                       multi_start_runs, multi_start_success_count,
                       multi_start_success_rate, multi_start_best_gap,
                       multi_start_mean_gap, multi_start_median_gap,
                       multi_start_worst_gap, total_completed_iterations
                FROM catalog_optimizer_audits
                ORDER BY base_case_id, member_type, tier
                """
            )
        ]
        ground_motions = [
            dict(row)
            for row in connection.execute(
                """
                SELECT pair_id, source_set, conditioning_period_s,
                       physical_pair_hash, dt_s, npts, duration_s,
                       pga_x_g, pga_y_g, sha256_x, sha256_y,
                       raw_source_path, raw_source_sha256,
                       valid, validation_message, source_metadata_json
                FROM ground_motion_catalog
                WHERE valid=1
                ORDER BY source_set, conditioning_period_s, pair_id
                """
            )
        ]
        selection = [
            dict(row)
            for row in connection.execute(
                """
                SELECT s.* FROM building_ground_motion_selection s
                JOIN building_catalog b USING(building_id)
                WHERE b.selected=1 AND b.valid=1
                ORDER BY b.queue_rank, s.pair_id
                """
            )
        ]
        manual_ida_points = [
            dict(row)
            for row in connection.execute(
                """
                SELECT r.*
                FROM ida_manual_point_requests r
                JOIN building_catalog b USING(building_id)
                WHERE b.selected=1 AND b.valid=1
                ORDER BY b.queue_rank, r.pair_id, r.target_im_g
                """
            )
        ]
        ml_runs = [
            dict(row)
            for row in connection.execute(
                """
                SELECT run_id, created_utc, selected_architecture,
                       selected_alpha, development_count, test_count,
                       promising, metrics_path, predictions_path,
                       evaluation_path
                FROM ml_runs ORDER BY created_utc DESC
                """
            )
        ]
        unresolved_failures = [
            dict(row)
            for row in connection.execute(
                """
                SELECT stage, building_id, pair_id, created_utc,
                       error_type, message
                FROM pipeline_failures
                WHERE resolved=0
                  AND (
                    building_id IS NULL
                    OR EXISTS (
                        SELECT 1
                        FROM building_catalog b
                        WHERE b.building_id=pipeline_failures.building_id
                          AND b.selected=1 AND b.valid=1
                    )
                  )
                ORDER BY created_utc
                """
            )
        ]
        base_case_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM catalog_base_cases"
            ).fetchone()[0]
        )
    compact_buildings = []
    for row in buildings:
        metadata_raw = row.pop("design_metadata_json", "{}")
        metadata = (
            json.loads(metadata_raw or "{}")
            if isinstance(metadata_raw, str)
            else dict(metadata_raw or {})
        )
        row.update(
            {
                "beam_gravity_tributary_width_m": metadata.get(
                    "beam_gravity_tributary_width_m"
                ),
                "beam_capacity_design_clear_span_m": metadata.get(
                    "beam_capacity_design_clear_span_m"
                ),
                "beam_capacity_design_shear_kn": metadata.get(
                    "beam_capacity_design_shear_kn"
                ),
                "beam_shear_utilization": metadata.get(
                    "beam_shear_target_utilization"
                ),
                "column_capacity_design_shear_kn": metadata.get(
                    "column_capacity_design_shear_kn"
                ),
                "column_phi_vn_kn": metadata.get("column_phi_vn_kn"),
                "column_shear_utilization": metadata.get(
                    "column_capacity_design_shear_utilization"
                ),
            }
        )
        compact_buildings.append(row)
    buildings = compact_buildings
    spo_by_building = {
        str(row["building_id"]): row for row in current_spo
    }
    fragility_by_building = {
        str(row["building_id"]): row for row in current_fragility
    }
    diagnostics_by_building: dict[str, list[dict[str, Any]]] = {}
    for row in current_ida_diagnostics:
        diagnostics_by_building.setdefault(
            str(row["building_id"]), []
        ).append(row)
    building_review = []
    for building in buildings:
        building_id = str(building["building_id"])
        row = dict(building)
        spo = spo_by_building.get(building_id, {})
        fragility = fragility_by_building.get(building_id, {})
        diagnostics = diagnostics_by_building.get(building_id, [])
        for key in (
            "t1_s",
            "vy_kn",
            "dy_m",
            "vc_kn",
            "dc_m",
            "vu_kn",
            "du_m",
            "energy_error",
            "pre_capping_energy_error",
            "post_capping_energy_error",
            "normalized_rmse",
            "mechanism_class_at_capping",
            "mechanism_class_at_ultimate",
            "collapse_classification",
            "collapse_roof_drift",
        ):
            row[f"spo_{key}"] = spo.get(key)
        for key in (
            "n_pairs",
            "theta_io_g",
            "beta_io",
            "theta_ls_g",
            "beta_ls",
            "theta_cp_g",
            "beta_cp",
            "valid",
            "validation_message",
        ):
            row[f"fragility_{key}"] = fragility.get(key)
        row["ida_curve_count"] = len(diagnostics)
        row["ida_total_point_count"] = sum(
            int(item.get("point_count") or 0) for item in diagnostics
        )
        row["ida_codex_review_curve_count"] = sum(
            bool(item.get("codex_review_required"))
            for item in diagnostics
        )
        row["ida_review_status"] = (
            "CRITICAL_QC_FAILURE"
            if any(
                item.get("review_status") == "CRITICAL_QC_FAILURE"
                for item in diagnostics
            )
            else (
                "CODEX_REVIEW_REQUIRED"
                if any(
                    bool(item.get("codex_review_required"))
                    for item in diagnostics
                )
                else (
                    "PASS_AUTOMATED_QC"
                    if diagnostics
                    else "IDA_PENDING"
                )
            )
        )
        row["ida_maximum_bracket_width"] = max(
            (
                float(item["maximum_final_bracket_width"])
                for item in diagnostics
                if item.get("maximum_final_bracket_width") is not None
            ),
            default=None,
        )
        row["ida_maximum_capacity_error_bound"] = max(
            (
                float(item["maximum_capacity_error_bound"])
                for item in diagnostics
                if item.get("maximum_capacity_error_bound") is not None
            ),
            default=None,
        )
        building_review.append(row)
    structural_audit_path = (
        Path(config["output_dir"])
        / "selected_model_structural_audit.json"
    )
    structural_audit: dict[str, Any] = {}
    if structural_audit_path.is_file():
        candidate_audit = json.loads(
            structural_audit_path.read_text(encoding="utf-8")
        )
        current_queue_signature = stable_hash(
            [
                {
                    "building_id": row["building_id"],
                    "model_hash": row["model_hash"],
                }
                for row in buildings
            ]
        )
        if (
            candidate_audit.get("model_schema")
            == CATALOG_MODEL_SCHEMA_VERSION
            and candidate_audit.get("queue_signature")
            == current_queue_signature
        ):
            structural_audit = candidate_audit
    compact_ground_motions = []
    for row in ground_motions:
        metadata_raw = row.pop("source_metadata_json", "{}")
        metadata = (
            json.loads(metadata_raw or "{}")
            if isinstance(metadata_raw, str)
            else dict(metadata_raw or {})
        )
        processing = metadata.get("analysis_preprocessing", {})
        processed_diagnostics = processing.get(
            "processed_baseline_diagnostics",
            {},
        )
        residual_velocities = [
            abs(float(values.get("residual_velocity_m_s", 0.0)))
            for values in processed_diagnostics.values()
        ]
        residual_displacements = [
            abs(float(values.get("residual_displacement_m", 0.0)))
            for values in processed_diagnostics.values()
        ]
        row.update(
            {
                "pairing_authority_sha256": metadata.get(
                    "pairing_authority_sha256"
                ),
                "preprocessing_method": processing.get("method"),
                "raw_component_x_sha256": processing.get(
                    "raw_component_x_sha256"
                ),
                "raw_component_y_sha256": processing.get(
                    "raw_component_y_sha256"
                ),
                "max_processed_residual_velocity_m_s": (
                    max(residual_velocities)
                    if residual_velocities
                    else None
                ),
                "max_processed_residual_displacement_m": (
                    max(residual_displacements)
                    if residual_displacements
                    else None
                ),
            }
        )
        compact_ground_motions.append(row)
    ground_motions = compact_ground_motions
    spo_columns = [
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
        "pre_capping_energy_error",
        "post_capping_energy_error",
        "normalized_rmse",
        "mechanism_class_at_capping",
        "mechanism_class_at_ultimate",
        "capping_beam_yielded_end_fraction",
        "capping_column_yielded_end_fraction",
        "ultimate_beam_yielded_end_fraction",
        "ultimate_column_yielded_end_fraction",
        "collapse_classification",
        "collapse_roof_drift",
        "analysis_signature",
        "curve_path",
        "mechanism_history_path",
        "valid",
        "validation_message",
    ]
    physical_hashes = {
        str(row["physical_pair_hash"])
        for row in ground_motions
        if row.get("physical_pair_hash")
    }
    duplicate_hashes = len(ground_motions) - len(physical_hashes)
    scwb_counts = {
        class_name: sum(
            str(row.get("scwb_class")) == class_name
            for row in buildings
        )
        for class_name in SCWB_RESEARCH_CLASSES
    }
    optimizer_exact_count = sum(
        abs(
            float(row["selected_objective_cost_per_m"])
            - float(row["exact_verified_objective"])
        )
        <= 1.0e-8
        for row in optimizer
    )
    optimizer_at_least_one_success = sum(
        int(row["multi_start_success_count"]) > 0
        for row in optimizer
    )
    optimizer_maximum_best_gap = max(
        (
            float(row["multi_start_best_gap"])
            for row in optimizer
        ),
        default=0.0,
    )
    dashboard = {
        "model_schema": CATALOG_MODEL_SCHEMA_VERSION,
        "generated_utc": (
            "UTC " + datetime.now(timezone.utc).isoformat(timespec="seconds")
        ),
        "selected_buildings": len(buildings),
        "covered_base_cases": len(
            {str(row["base_case_id"]) for row in buildings}
        ),
        "catalog_base_cases": base_case_count,
        "research_low_margin_buildings": scwb_counts[
            SCWB_RESEARCH_CLASSES[0]
        ],
        "research_medium_margin_buildings": scwb_counts[
            SCWB_RESEARCH_CLASSES[1]
        ],
        "research_high_margin_buildings": scwb_counts[
            SCWB_RESEARCH_CLASSES[2]
        ],
        "research_column_beam_acceptance_rule": (
            "minimum non-roof joint sum(Mnc_column)/sum(Mnb_beam) > 1.00; "
            "sampling bands >1.00-1.50, 1.50-3.00 and >=3.00; no "
            "code-compliance claim"
        ),
        "optimizer_audits": len(optimizer),
        "exact_certified_optima": optimizer_exact_count,
        "ffa_audits_with_successful_start": (
            optimizer_at_least_one_success
        ),
        "ffa_audits_without_successful_start": (
            len(optimizer) - optimizer_at_least_one_success
        ),
        "maximum_multistart_best_gap": optimizer_maximum_best_gap,
        "valid_gm_catalog_rows": len(ground_motions),
        "unique_physical_gm_pairs": len(physical_hashes),
        "duplicate_gm_catalog_rows": duplicate_hashes,
        "current_spo": len(current_spo),
        "current_ida_runs": len(current_ida),
        "current_ida_capacities": len(current_capacities),
        "current_ida_curve_diagnostics": len(current_ida_diagnostics),
        "ida_curves_requiring_codex_review": sum(
            bool(row.get("codex_review_required"))
            for row in current_ida_diagnostics
        ),
        "ida_hard_point_cap": "NONE",
        "ida_soft_review_point_count": int(
            config["ida"]["controller"]["soft_review_point_count"]
        ),
        "ida_bracket_tolerance": float(
            config["ida"]["bracket_tolerance"]
        ),
        "current_fragilities": len(current_fragility),
        "ml_runs": len(ml_runs),
        "migration_smoke_checks": len(migration_smoke),
        "migration_smoke_passed": sum(
            row.get("status") == "PASS" for row in migration_smoke
        ),
        "unresolved_pipeline_failures": len(unresolved_failures),
        "ann_feature_count": 7,
        "ann_features": "Vy, Dy, Vc, Dc, Vu, Du, T1",
        "planned_full_ida_buildings": int(
            config["ida"]["planned_full_ida_buildings"]
        ),
        "gm_limitation": (
            "Available CMS families contain only 3-4 pairs; retained as an "
            "explicit research limitation."
        ),
        "ida_scaling_policy": str(config["ida"]["scaling_policy"]),
        "modal_identification_method": str(
            config["model"]["modal_identification_method"]
        ),
        "minimum_cumulative_translational_mass_ratio": float(
            config["model"][
                "minimum_cumulative_translational_mass_ratio"
            ]
        ),
        "structural_queue_audit_passed": structural_audit.get("passed"),
        "structural_queue_audit_failed": structural_audit.get("failed"),
        "structural_queue_max_gravity_error": structural_audit.get(
            "max_gravity_error"
        ),
        "structural_queue_max_xy_period_error": structural_audit.get(
            "max_xy_period_error"
        ),
        "structural_queue_t1_range_s": (
            " to ".join(
                f"{float(value):.6g}"
                for value in structural_audit.get("t1_range_s", [])
            )
            if structural_audit
            else None
        ),
        "structural_queue_selected_effective_mass_ratio_range": (
            " to ".join(
                f"{float(value):.6g}"
                for value in structural_audit.get(
                    "selected_effective_mass_ratio_range",
                    [],
                )
            )
            if structural_audit
            else None
        ),
        "hinge_sensitivity_max_change_percent": (
            sensitivity_summary.get("hinge_sensitivity", {}).get(
                "maximum_absolute_primary_response_change_percent"
            )
        ),
        "hinge_mechanism_class_change_observed": (
            sensitivity_summary.get("hinge_sensitivity", {}).get(
                "any_mechanism_class_change"
            )
        ),
        "modeling_spo_max_change_percent": (
            sensitivity_summary.get("modeling_sensitivity", {}).get(
                "maximum_absolute_spo_change_percent"
            )
        ),
        "fine_mesh_convergence_max_change_percent": (
            sensitivity_summary.get("modeling_sensitivity", {}).get(
                "fine_mesh_convergence_max_change_percent"
            )
        ),
        "fine_mesh_convergence_within_tolerance": (
            sensitivity_summary.get("modeling_sensitivity", {}).get(
                "fine_mesh_convergence_within_tolerance"
            )
        ),
        "damping_midr_max_change_percent": (
            sensitivity_summary.get("modeling_sensitivity", {}).get(
                "maximum_absolute_damping_midr_change_percent"
            )
        ),
        "damping_sensitivity_enabled": (
            sensitivity_summary.get("modeling_sensitivity", {}).get(
                "damping_sensitivity_enabled"
            )
        ),
        "production_modal_damping_ratio": float(
            config["model"]["damping_ratio"]
        ),
        "engineering_scope": (
            "Engineering-screened research archetypes; detailed slab flexure, "
            "nonlinear member shear, joint shear, development/splice, "
            "bond-slip, bar buckling and torsional failure are outside the "
            "declared PoC scope."
        ),
        "beam_section_model": str(config["model"]["beam_section_model"]),
        "collapse_scope": str(config["model"]["collapse_scope"]),
    }
    batch_output_dir = Path(config["output_dir"]) / "batch_001"
    batch_state_path = (
        Path(config["run_dir"]) / "batches" / "batch_001" / "batch_state.json"
    )
    batch_runtime: list[dict[str, Any]] = []
    if batch_state_path.is_file():
        batch_state = json.loads(
            batch_state_path.read_text(encoding="utf-8")
        )
        for stage_name, stage in batch_state.get("stages", {}).items():
            batch_runtime.append(
                {
                    "record_type": "stage",
                    "stage": stage_name,
                    "status": stage.get("status"),
                    "attempt_count": len(stage.get("attempts", [])),
                    "accumulated_wall_seconds": stage.get(
                        "accumulated_wall_seconds"
                    ),
                    "accumulated_wall_minutes": (
                        float(stage.get("accumulated_wall_seconds", 0.0))
                        / 60.0
                    ),
                    "accumulated_wall_hours": (
                        float(stage.get("accumulated_wall_seconds", 0.0))
                        / 3600.0
                    ),
                }
            )
        first_timing = batch_state.get("first_building_timing")
        if isinstance(first_timing, dict):
            batch_runtime.append(
                {
                    "record_type": "first_building",
                    "stage": "end_to_end_timing",
                    **first_timing,
                }
            )
        final_progress = batch_state.get("final_progress")
        if isinstance(final_progress, dict):
            batch_runtime.append(
                {
                    "record_type": "batch_summary",
                    "stage": "batch_001",
                    **final_progress,
                    "total_accumulated_stage_wall_seconds": batch_state.get(
                        "total_accumulated_stage_wall_seconds"
                    ),
                    "average_pipeline_wall_seconds_per_building": (
                        batch_state.get(
                            "average_pipeline_wall_seconds_per_building"
                        )
                    ),
                }
            )
        dashboard["batch_001_status"] = batch_state.get("status")
        dashboard["batch_001_first_building_pipeline_seconds"] = (
            (first_timing or {}).get("pipeline_wall_seconds")
            if isinstance(first_timing, dict)
            else None
        )
        dashboard["batch_001_average_pipeline_seconds_per_building"] = (
            batch_state.get("average_pipeline_wall_seconds_per_building")
        )
    no_pwsa_comparison: list[dict[str, Any]] = []
    no_pwsa_path = batch_output_dir / (
        "Batch01_Fragility_Full_vs_NoPWSA.csv"
    )
    if no_pwsa_path.is_file():
        with no_pwsa_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            for source_row in csv.DictReader(handle):
                row: dict[str, Any] = {}
                for key, value in source_row.items():
                    if value in {None, ""}:
                        row[key] = None
                        continue
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        row[key] = value
                no_pwsa_comparison.append(row)
    qa = [
        {
            "check": "Full selected-queue structural audit",
            "status": (
                "PASS"
                if (
                    structural_audit
                    and int(structural_audit.get("failed", 1)) == 0
                    and int(structural_audit.get("passed", 0))
                    == len(buildings)
                )
                else (
                    "FAIL"
                    if structural_audit
                    else "PENDING PRODUCTION SPO"
                )
            ),
            "detail": (
                f"Build, gravity and 12-mode effective-mass analysis passed "
                f"{structural_audit.get('passed', 0)}/{len(buildings)} active "
                "Building IDs. The isolated migration smoke passed the "
                "representative 1x1, 3x3 and 5x5 topologies; every production "
                "Building ID receives this check during run-modal-spo. The "
                "workbook accepts a full-queue audit only when its model "
                "schema and complete queue signature match."
            ),
            "source": "",
        },
        {
            "check": "Modal direction identification",
            "status": "EFFECTIVE-MASS BASED",
            "detail": (
                "Distinct X/Y modes are selected by directional effective "
                "modal mass using UX, UY and RZ generalized mass. The "
                "extracted set must capture at least 90% cumulative X and Y "
                "translational mass; raw eigenvector sums are not used."
            ),
            "source": "",
        },
        {
            "check": "Plastic-hinge integration",
            "status": "PASS by construction",
            "detail": (
                "HingeMidpoint; every built model validates positive actual "
                "OpenSees integration weights summing to member length."
            ),
            "source": (
                "https://opensees.github.io/OpenSeesDocumentation/user/"
                "manual/model/beamIntegrations/HingeMidpoint.html"
            ),
        },
        {
            "check": "Plastic-hinge length",
            "status": (
                "EMPIRICAL BASELINE; SENSITIVITY RECORDED"
                if hinge_sensitivity
                else "EMPIRICAL BASELINE; SENSITIVITY PENDING"
            ),
            "detail": (
                "Production Lp=max(0.08Ls+0.022fye*db, 0.044fye*db), "
                "with fye=1.25fy. Ls=abs(M/V) is computed at every member "
                "end from the rigid-diaphragm, inverted-triangular elastic "
                "reference frame, matching Draft 5's distance-to-"
                "contraflexure definition. The recorded "
                "0.75x/1.00x/1.25x study quantifies modelling uncertainty "
                "but is not specimen-specific laboratory calibration."
            ),
            "source": (
                "https://web.engr.oregonstate.edu/~mhscott/"
                "Scott-Fenves_JSE_2006.pdf"
            ),
        },
        {
            "check": "Fibre-section mesh convergence",
            "status": (
                "PASS"
                if sensitivity_summary.get(
                    "modeling_sensitivity", {}
                ).get("fine_mesh_convergence_within_tolerance")
                else "PENDING/FAIL"
            ),
            "detail": (
                "Production 24x24 core mesh is compared with a 32x32 fine "
                "mesh across all primary SPO responses. Acceptance requires "
                "the maximum absolute change to remain within the configured "
                "10% numerical-convergence tolerance. The 16x16 case is "
                "retained only as a coarse-mesh sensitivity bound."
            ),
            "source": "",
        },
        {
            "check": "Physical GM deduplication",
            "status": "ACTIVE",
            "detail": (
                "A SHA-256 physical-pair hash prevents a duplicated X-Y pair "
                "from entering one building selection twice."
            ),
            "source": "",
        },
        {
            "check": "Ground-motion baseline processing",
            "status": "ACTIVE",
            "detail": (
                "Raw components are checksum-preserved. Response spectra and "
                "NLTHA use deterministic linear least-squares detrended "
                "analysis copies with residual velocity/displacement QA."
            ),
            "source": "",
        },
        {
            "check": "CP bracketing",
            "status": "ACTIVE",
            "detail": (
                "Accuracy-driven IDA hunts beyond the 4g warning until CP is "
                "observed, then refines measured IO/LS/CP brackets to 5%. "
                "There is no hard point-count cap. Counts above 18 are review "
                "flags only and never stop refinement."
            ),
            "source": "",
        },
        {
            "check": "Codex IDA review and incremental extension",
            "status": "ACTIVE",
            "detail": (
                "Every curve stores bracket widths, worst-case localization "
                "error bounds, numerical recovery, non-monotonic response "
                "flags and recommended extra IM targets. Codex/user-requested "
                "points reuse all signed fixed-IM checkpoints and rebuild only "
                "the affected curve and building fragility."
            ),
            "source": "",
        },
        {
            "check": "Capacity-design shear screen",
            "status": "ACTIVE; FLEXURE-CONTROL SCREEN ONLY",
            "detail": (
                "Every retained beam and column passes phiVn against the "
                "configured probable-flexural-strength shear demand. The "
                "OpenSees fibre elements do not simulate nonlinear shear "
                "degradation or shear failure."
            ),
            "source": (
                "https://opensees.berkeley.edu/wiki/index.php/"
                "Section_Aggregator"
            ),
        },
        {
            "check": "Column P-M and research column/beam rule",
            "status": "STRAIN-COMPATIBLE",
            "detail": (
                "Column flexural strength uses a Whitney block, discrete "
                "perimeter bars, tied-column phi transition and the factored "
                "axial load at each tier/joint. Admission requires the "
                "minimum non-roof joint sum(Mnc_column)/sum(Mnb_beam)>1.00. "
                "The >1.00-1.50, 1.50-3.00 and >=3.00 bands support sampling "
                "and do not claim ACI SCWB code compliance."
            ),
            "source": "",
        },
        {
            "check": "Fragility beta",
            "status": "SAMPLE RTR DEFINITION",
            "detail": (
                "All limit states require complete first-crossing IDA "
                "capacities. Beta_RTR is the Bessel-corrected sample "
                "standard deviation of ln(IM capacity), using ddof=1. "
                "Censored rows are rejected from the final ML dataset."
            ),
            "source": "",
        },
        {
            "check": "ANN inputs",
            "status": "LOCKED",
            "detail": "Exactly 7 inputs; Run End/mechanism are audit data.",
            "source": "",
        },
        {
            "check": "Data-generation/ML phase separation",
            "status": "ML DEFERRED",
            "detail": (
                "The current phase creates SPO, Full IDA and fragility data "
                "for the 375-building research queue only. No development/"
                "test role is assigned until the later ML phase begins."
            ),
            "source": "",
        },
        {
            "check": "Dataset-size claim gate",
            "status": "LOCKED",
            "detail": (
                "The Draft-5 Demo is evaluated only on the frozen completed "
                "population actually analysed, between 100 and 375 unique "
                "Building IDs. No 5,000-building claim is made."
            ),
            "source": "",
        },
        {
            "check": "FFA optimum",
            "status": "EXACT CERTIFIED",
            "detail": (
                "Multi-start FFA is a search accelerator only. Exact discrete "
                f"enumeration selected/certified {optimizer_exact_count}/"
                f"{len(optimizer)} final optima. At least one FFA start found "
                f"the same optimum in {optimizer_at_least_one_success}/"
                f"{len(optimizer)} audits; the maximum best-of-five FFA gap "
                f"was {100.0 * optimizer_maximum_best_gap:.2f}%."
            ),
            "source": "",
        },
    ]
    return {
        "dashboard": dashboard,
        "buildings": _json_safe_rows(buildings),
        "optimizer": _json_safe_rows(optimizer),
        "ground_motions": _json_safe_rows(ground_motions),
        "ground_motion_selection": _json_safe_rows(selection),
        "spo": _json_safe_rows(
            [
                _presentation_safe_endpoint_names(
                    {key: row.get(key) for key in spo_columns}
                )
                for row in current_spo
            ]
        ),
        "ida_runs": _json_safe_rows(current_ida),
        "ida_capacities": _json_safe_rows(current_capacities),
        "ida_curve_diagnostics": _json_safe_rows(current_ida_diagnostics),
        "manual_ida_points": _json_safe_rows(manual_ida_points),
        "building_review": _json_safe_rows(building_review),
        "fragility": _json_safe_rows(current_fragility),
        "ml_runs": _json_safe_rows(ml_runs),
        "hinge_sensitivity": _json_safe_rows(hinge_sensitivity),
        "modeling_sensitivity": _json_safe_rows(modeling_sensitivity),
        "migration_smoke": _json_safe_rows(migration_smoke),
        "pipeline_failures": _json_safe_rows(unresolved_failures),
        "batch_runtime": _json_safe_rows(batch_runtime),
        "no_pwsa_comparison": _json_safe_rows(no_pwsa_comparison),
        "qa": qa,
    }


def _artifact_runtime_paths() -> tuple[Path, Path]:
    configured_node = os.environ.get("FRAGILITY_POC_NODE_EXE")
    configured_modules = os.environ.get("FRAGILITY_POC_NODE_MODULES")
    if configured_node or configured_modules:
        if not configured_node or not configured_modules:
            raise RuntimeError(
                "Set both FRAGILITY_POC_NODE_EXE and "
                "FRAGILITY_POC_NODE_MODULES"
            )
        node = Path(configured_node).resolve()
        modules = Path(configured_modules).resolve()
        if not node.is_file() or not modules.is_dir():
            raise RuntimeError(
                "Configured Node/artifact-tool runtime does not exist"
            )
        return node, modules
    dependency_root = (
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
    )
    node = dependency_root / "node" / "bin" / "node.exe"
    modules = dependency_root / "node" / "node_modules"
    if not node.is_file() or not modules.is_dir():
        raise RuntimeError(
            "Bundled Node/artifact-tool runtime was not found; run inside "
            "the configured Codex workspace runtime"
        )
    return node, modules


def _ensure_node_modules_junction(link: Path, target: Path) -> None:
    if link.exists():
        if link.resolve() != target.resolve():
            raise RuntimeError(
                f"Existing artifact-tool junction points elsewhere: {link}"
            )
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError(
                "Could not create artifact-tool junction: "
                + completed.stderr.decode(
                    "utf-8", errors="replace"
                ).strip()
            )
    else:  # pragma: no cover - Windows is the research workstation
        link.symlink_to(target, target_is_directory=True)


def build_consolidated_workbook(config: dict[str, Any]) -> dict[str, Any]:
    """Create the single current-results workbook through artifact-tool."""
    initialize(config["database_path"])
    # Keep the user-facing ASCII ``phD`` junction in artifact provenance.
    # ``Path.resolve()`` dereferences the jSync-managed junction back to the
    # legacy Thai directory name even though the configured runtime path is
    # intentionally the ASCII alias.
    configured_root = Path(
        str(config.get("_project_root", ""))
    ).absolute()
    source_root = Path(__file__).resolve().parents[2]
    root_candidates = [configured_root, Path.cwd().absolute(), source_root]
    project_root = next(
        (
            root
            for root in root_candidates
            if (root / "scripts" / "build_summary_workbook.mjs").is_file()
        ),
        None,
    )
    if project_root is None:
        raise RuntimeError(
            "Could not locate scripts/build_summary_workbook.mjs from the "
            "configuration, current directory, or source tree"
        )
    runtime = project_root / "tmp" / "artifact_workbook_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    payload_path = runtime / "summary_payload.json"
    atomic_write_json(payload_path, _report_payload(config))
    builder_source = project_root / "scripts" / "build_summary_workbook.mjs"
    builder_runtime = runtime / "build_summary_workbook.mjs"
    shutil.copyfile(builder_source, builder_runtime)
    node, modules = _artifact_runtime_paths()
    _ensure_node_modules_junction(runtime / "node_modules", modules)
    output_path = (
        Path(config["output_dir"]) / "RC_Fragility_PoC_Summary.xlsx"
    ).absolute()
    preview_dir = runtime / "previews"
    completed = subprocess.run(
        [
            str(node),
            str(builder_runtime),
            str(payload_path.absolute()),
            str(output_path),
            str(preview_dir.absolute()),
        ],
        cwd=runtime,
        check=False,
        capture_output=True,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError(
            "Consolidated workbook generation failed: "
            + stderr.strip()[-4000:]
        )
    verification = json.loads(stdout.strip().splitlines()[-1])
    return {
        "workbook_path": str(output_path),
        "sheet_count": int(verification["sheet_count"]),
        "formula_error_count": int(verification["formula_error_count"]),
        "rendered_sheet_count": int(
            verification["rendered_sheet_count"]
        ),
    }
