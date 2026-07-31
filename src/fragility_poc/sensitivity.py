"""Plastic-hinge-length sensitivity analysis for the RC frame SPO model."""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .catalog import SCWB_RESEARCH_CLASSES
from .db import connect
from .ground_motion import select_motions_for_t1
from .ida import ScaleFactorGuardError, run_nltha
from .io_utils import atomic_write_json, read_json, stable_hash
from .spo import _spo_analysis_signature, run_one_spo
from .structural import plastic_hinge_layout


SENSITIVITY_SCHEMA_VERSION = "plastic-hinge-length-sensitivity-v4"
MODELING_SENSITIVITY_SCHEMA_VERSION = "modeling-sensitivity-oat-v4"
DEFAULT_FACTORS = (0.75, 1.00, 1.25)
DEFAULT_BASELINE_FACTOR = 1.00
DEFAULT_REPRESENTATIVE_CLASSES = SCWB_RESEARCH_CLASSES
PRIMARY_RESPONSE_METRICS = (
    "t1_s",
    "vy_kn",
    "dy_m",
    "vc_kn",
    "dc_m",
    "vu_kn",
    "du_m",
    "collapse_roof_drift",
)
REFERENCE_URLS = (
    "https://opensees.github.io/OpenSeesDocumentation/user/manual/"
    "model/beamIntegrations/HingeMidpoint.html",
    "https://repo.nzsee.org.nz/handle/nzsee/2476",
    "https://www.mdpi.com/2075-5309/12/10/1603",
)


def _validate_factors(
    factors: list[float] | tuple[float, ...],
    baseline_factor: float,
) -> tuple[float, ...]:
    values = tuple(sorted({float(value) for value in factors}))
    if len(values) < 3:
        raise ValueError(
            "Plastic-hinge sensitivity requires at least three unique factors"
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(
            "Plastic-hinge sensitivity factors must be positive and finite"
        )
    if not any(
        math.isclose(value, baseline_factor, rel_tol=0.0, abs_tol=1.0e-12)
        for value in values
    ):
        raise ValueError("The baseline plastic-hinge factor must be included")
    return values


def _factor_tag(factor: float) -> str:
    return f"{factor:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _load_selected_buildings(
    config: dict[str, Any],
    *,
    building_ids: list[str] | None,
) -> list[dict[str, Any]]:
    with connect(config["database_path"]) as connection:
        parameters: list[Any] = []
        where = "valid=1 AND selected=1"
        if building_ids:
            marks = ",".join("?" for _ in building_ids)
            where += f" AND building_id IN ({marks})"
            parameters.extend(building_ids)
        rows = [
            dict(row)
            for row in connection.execute(
                f"SELECT * FROM building_catalog WHERE {where} "
                "ORDER BY queue_rank",
                parameters,
            )
        ]
    if building_ids:
        found = {str(row["building_id"]) for row in rows}
        missing = sorted(set(building_ids) - found)
        if missing:
            raise ValueError(
                "Requested sensitivity Building IDs are not selected/valid: "
                + ", ".join(missing)
            )
    return rows


def _representative_buildings(
    buildings: list[dict[str, Any]],
    *,
    configured_ids: list[str] | None,
    representative_classes: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Choose sensitivity cases without inventing unavailable SCWB classes.

    The exact strain-compatible SCWB calculation can legitimately leave an
    archetype with only one observed class.  In that case, retain the requested
    number of representatives by spanning the observed SCWB-ratio range
    instead of failing or fabricating weaker buildings.
    """
    if not buildings:
        raise ValueError("No selected buildings are available for sensitivity")
    if configured_ids:
        by_id = {str(row["building_id"]): row for row in buildings}
        missing = [item for item in configured_ids if item not in by_id]
        if missing:
            raise ValueError(
                "Configured sensitivity Building IDs are unavailable: "
                + ", ".join(missing)
            )
        selected = [by_id[item] for item in configured_ids]
    else:
        selected = []
        selected_ids: set[str] = set()
        for class_name in representative_classes:
            candidates = [
                row
                for row in buildings
                if str(row["scwb_class"]) == class_name
            ]
            if not candidates:
                continue
            ratios = np.asarray(
                [float(row["scwb_strength_ratio"]) for row in candidates]
            )
            median_ratio = float(np.median(ratios))
            representative = min(
                candidates,
                key=lambda row: (
                    abs(
                        float(row["scwb_strength_ratio"])
                        - median_ratio
                    ),
                    int(row["queue_rank"]),
                ),
            )
            selected.append(representative)
            selected_ids.add(str(representative["building_id"]))

        target_count = min(len(representative_classes), len(buildings))
        ranked = sorted(
            buildings,
            key=lambda row: (
                float(row["scwb_strength_ratio"]),
                float(row["beam_phi_mn_knm"])
                + float(row["column_phi_mn_knm"]),
                int(row["queue_rank"]),
            ),
        )
        if len(selected) < target_count:
            selected_rank_indices = {
                index
                for index, row in enumerate(ranked)
                if str(row["building_id"]) in selected_ids
            }
            target_indices = sorted(
                np.linspace(0, len(ranked) - 1, target_count),
                key=lambda value: min(
                    (
                        abs(float(value) - selected_index)
                        for selected_index in selected_rank_indices
                    ),
                    default=math.inf,
                ),
                reverse=True,
            )
            for target_index in target_indices:
                available = [
                    (index, row)
                    for index, row in enumerate(ranked)
                    if str(row["building_id"]) not in selected_ids
                ]
                if not available or len(selected) >= target_count:
                    break
                _, representative = min(
                    available,
                    key=lambda item: (
                        abs(item[0] - float(target_index)),
                        int(item[1]["queue_rank"]),
                    ),
                )
                selected.append(representative)
                selected_ids.add(str(representative["building_id"]))

        if len(selected) < target_count:
            for representative in ranked:
                building_id = str(representative["building_id"])
                if building_id in selected_ids:
                    continue
                selected.append(representative)
                selected_ids.add(building_id)
                if len(selected) >= target_count:
                    break
    return selected


def _case_signature(
    building: dict[str, Any],
    run_config: dict[str, Any],
    factor: float,
) -> str:
    return stable_hash(
        {
            "schema_version": SENSITIVITY_SCHEMA_VERSION,
            "building_id": building["building_id"],
            "model_hash": building.get("model_hash"),
            "plastic_hinge_length_factor": factor,
            "spo_analysis_signature": _spo_analysis_signature(
                building, run_config
            ),
        }
    )


def _checkpoint_is_reusable(
    checkpoint_path: Path,
    *,
    expected_signature: str,
) -> bool:
    if not checkpoint_path.exists():
        return False
    stored = read_json(checkpoint_path)
    if stored.get("sensitivity_signature") != expected_signature:
        return False
    for key in ("curve_path", "mechanism_history_path"):
        if not Path(str(stored.get(key, ""))).exists():
            return False
    return True


def _relative_change_percent(value: float, baseline: float) -> float:
    if not math.isfinite(value) or not math.isfinite(baseline):
        return math.nan
    if abs(baseline) <= 1.0e-12:
        return math.nan
    return 100.0 * (value / baseline - 1.0)


def _attach_baseline_changes(
    results: list[dict[str, Any]],
    *,
    baseline_factor: float,
) -> None:
    baseline_by_building = {
        str(row["building_id"]): row
        for row in results
        if math.isclose(
            float(row["plastic_hinge_length_factor"]),
            baseline_factor,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    }
    for row in results:
        baseline = baseline_by_building[str(row["building_id"])]
        for metric in PRIMARY_RESPONSE_METRICS:
            row[f"{metric}_change_from_baseline_pct"] = (
                _relative_change_percent(
                    float(row[metric]),
                    float(baseline[metric]),
                )
            )


def _sensitivity_summary(
    results: list[dict[str, Any]],
    *,
    baseline_factor: float,
) -> dict[str, Any]:
    by_building: dict[str, Any] = {}
    all_primary_changes: list[float] = []
    for building_id in sorted({str(row["building_id"]) for row in results}):
        rows = [
            row for row in results if str(row["building_id"]) == building_id
        ]
        metric_changes: dict[str, float] = {}
        for metric in PRIMARY_RESPONSE_METRICS:
            changes = [
                abs(float(row[f"{metric}_change_from_baseline_pct"]))
                for row in rows
                if math.isfinite(
                    float(row[f"{metric}_change_from_baseline_pct"])
                )
            ]
            maximum = max(changes, default=math.nan)
            metric_changes[metric] = maximum
            if math.isfinite(maximum):
                all_primary_changes.append(maximum)
        mechanism_changed = any(
            (
                row["mechanism_class_at_capping"],
                row["mechanism_class_at_ultimate"],
                row["mechanism_class_at_run_end"],
            )
            != (
                rows[0]["mechanism_class_at_capping"],
                rows[0]["mechanism_class_at_ultimate"],
                rows[0]["mechanism_class_at_run_end"],
            )
            for row in rows[1:]
        )
        by_building[building_id] = {
            "scwb_class": rows[0]["scwb_class"],
            "maximum_absolute_change_percent": metric_changes,
            "mechanism_class_changed": mechanism_changed,
            "all_cases_valid": all(bool(row["valid"]) for row in rows),
        }
    maximum_change = max(all_primary_changes, default=math.nan)
    if not math.isfinite(maximum_change):
        rating = "indeterminate"
    elif maximum_change <= 5.0 + 1.0e-9:
        rating = "low"
    elif maximum_change <= 10.0 + 1.0e-9:
        rating = "moderate"
    else:
        rating = "high"
    return {
        "baseline_factor": baseline_factor,
        "maximum_absolute_primary_response_change_percent": maximum_change,
        "sensitivity_rating": rating,
        "all_cases_valid": all(bool(row["valid"]) for row in results),
        "any_mechanism_class_change": any(
            item["mechanism_class_changed"]
            for item in by_building.values()
        ),
        "by_building": by_building,
        "interpretation": (
            "This is a parametric modelling-uncertainty study. It does not "
            "calibrate Lp against laboratory data or prove that the baseline "
            "factor is physically correct."
        ),
    }


def _load_curve(result: dict[str, Any]) -> np.ndarray:
    return np.loadtxt(
        result["curve_path"],
        delimiter=",",
        skiprows=1,
        ndmin=2,
    )


def _plot_sensitivity(
    results: list[dict[str, Any]],
    output_path: Path,
    *,
    baseline_factor: float,
) -> None:
    building_ids = list(dict.fromkeys(str(row["building_id"]) for row in results))
    factors = sorted(
        {
            float(row["plastic_hinge_length_factor"])
            for row in results
        }
    )
    palette = ("#2563EB", "#111827", "#D97706", "#DC2626", "#059669")
    colors = {
        factor: palette[index % len(palette)]
        for index, factor in enumerate(factors)
    }
    fig, axes = plt.subplots(
        len(building_ids),
        2,
        figsize=(13.5, 4.2 * len(building_ids)),
        squeeze=False,
    )
    for row_index, building_id in enumerate(building_ids):
        rows = [
            row for row in results if str(row["building_id"]) == building_id
        ]
        rows.sort(key=lambda item: float(item["plastic_hinge_length_factor"]))
        curve_axis = axes[row_index, 0]
        metric_axis = axes[row_index, 1]
        for row in rows:
            factor = float(row["plastic_hinge_length_factor"])
            curve = _load_curve(row)
            color = colors.get(factor, None)
            linewidth = (
                2.4
                if math.isclose(factor, baseline_factor)
                else 1.5
            )
            curve_axis.plot(
                100.0
                * curve[:, 0]
                / (
                    float(row["stories"])
                    * float(row["story_height_m"])
                ),
                curve[:, 1],
                label=(
                    f"{factor:.2f}x empirical Lp "
                    f"(beam {float(row['beam_lp_m']):.3f} m)"
                ),
                color=color,
                linewidth=linewidth,
            )
            curve_axis.scatter(
                [
                    100.0
                    * float(row["dy_m"])
                    / (
                        float(row["stories"])
                        * float(row["story_height_m"])
                    ),
                    100.0
                    * float(row["dc_m"])
                    / (
                        float(row["stories"])
                        * float(row["story_height_m"])
                    ),
                    100.0
                    * float(row["du_m"])
                    / (
                        float(row["stories"])
                        * float(row["story_height_m"])
                    ),
                ],
                [row["vy_kn"], row["vc_kn"], row["vu_kn"]],
                color=color,
                s=18,
                zorder=3,
            )
        scwb_class = str(rows[0]["scwb_class"])
        curve_axis.set_title(f"{building_id} — {scwb_class}")
        curve_axis.set_xlabel("Roof drift (%)")
        curve_axis.set_ylabel("Base shear (kN)")
        curve_axis.grid(alpha=0.22)
        curve_axis.legend(ncol=2, fontsize=8)

        factor_values = [
            float(row["plastic_hinge_length_factor"]) for row in rows
        ]
        metric_styles = (
            ("vc_kn", "Vc", "o"),
            ("dc_m", "Dc", "s"),
            ("du_m", "Du", "^"),
            ("collapse_roof_drift", "Model-end drift", "D"),
        )
        for metric, label, marker in metric_styles:
            metric_axis.plot(
                factor_values,
                [
                    float(row[f"{metric}_change_from_baseline_pct"])
                    for row in rows
                ],
                marker=marker,
                linewidth=1.5,
                label=label,
            )
        metric_axis.axhline(0.0, color="#111827", linewidth=0.8)
        metric_axis.axhspan(-5.0, 5.0, color="#16A34A", alpha=0.08)
        metric_axis.set_title(
            f"Change from empirical baseline x {baseline_factor:.2f}"
        )
        metric_axis.set_xlabel("Multiplier on empirical plastic-hinge length")
        metric_axis.set_ylabel("Change from baseline (%)")
        metric_axis.grid(alpha=0.22)
        metric_axis.legend(ncol=2, fontsize=8)

    fig.suptitle(
        "Plastic-Hinge-Length Sensitivity — SPO Response",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _export_curve_rows(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    columns = (
        "building_id",
        "scwb_class",
        "plastic_hinge_length_factor",
        "beam_lp_m",
        "column_lp_m",
        "roof_displacement_m",
        "roof_drift",
        "base_shear_kn",
        "trilinear_kn",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            curve = _load_curve(result)
            total_height = (
                float(result["stories"])
                * float(result["story_height_m"])
            )
            for displacement, shear, trilinear in curve:
                writer.writerow(
                    {
                        "building_id": result["building_id"],
                        "scwb_class": result["scwb_class"],
                        "plastic_hinge_length_factor": result[
                            "plastic_hinge_length_factor"
                        ],
                        "beam_lp_m": result["beam_lp_m"],
                        "column_lp_m": result["column_lp_m"],
                        "roof_displacement_m": displacement,
                        "roof_drift": displacement / total_height,
                        "base_shear_kn": shear,
                        "trilinear_kn": trilinear,
                    }
                )


def run_plastic_hinge_length_sensitivity(
    config: dict[str, Any],
    *,
    building_ids: list[str] | None = None,
    factors: list[float] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run isolated SPO cases for multipliers on the empirical Priestley Lp."""
    settings = config.get("plastic_hinge_length_sensitivity", {})
    baseline_factor = float(
        settings.get("baseline_factor", DEFAULT_BASELINE_FACTOR)
    )
    factor_values = _validate_factors(
        factors or settings.get("factors", DEFAULT_FACTORS),
        baseline_factor,
    )
    representative_classes = tuple(
        settings.get(
            "representative_classes",
            DEFAULT_REPRESENTATIVE_CLASSES,
        )
    )
    configured_ids = (
        list(settings.get("representative_building_ids", []))
        if building_ids is None
        else list(building_ids)
    )
    all_buildings = _load_selected_buildings(
        config,
        building_ids=configured_ids or None,
    )
    buildings = _representative_buildings(
        all_buildings,
        configured_ids=configured_ids or None,
        representative_classes=representative_classes,
    )

    run_root = (
        Path(config["run_dir"])
        / "sensitivity"
        / "plastic_hinge_length"
    )
    results: list[dict[str, Any]] = []
    for building in buildings:
        for factor in factor_values:
            factor_tag = _factor_tag(factor)
            run_config = copy.deepcopy(config)
            run_config["run_dir"] = str(run_root / f"lp_{factor_tag}")
            run_config["model"]["plastic_hinge_length_method"] = (
                "priestley_1996_contraflexure_expected_strength_scaled"
            )
            run_config["model"]["plastic_hinge_length_factor"] = 1.0
            run_config["model"]["plastic_hinge_length_scale_factor"] = factor
            case_signature = _case_signature(
                building,
                run_config,
                factor,
            )
            checkpoint_path = (
                run_root
                / f"lp_{factor_tag}"
                / str(building["building_id"])
                / "sensitivity_result.json"
            )
            if resume and _checkpoint_is_reusable(
                checkpoint_path,
                expected_signature=case_signature,
            ):
                result = read_json(checkpoint_path)
            else:
                result = run_one_spo(building, run_config, direction="X")
                layout = plastic_hinge_layout(building, run_config)
                beam_values = [
                    value
                    for level in layout.beam_hinge_lengths_m
                    for pair in level
                    for value in pair
                ]
                column_values = [
                    value
                    for level in layout.column_hinge_lengths_m
                    for pair in level
                    for value in pair
                ]
                beam_lp_m = float(np.median(beam_values))
                column_lp_m = float(np.median(column_values))
                result.update(
                    {
                        "sensitivity_schema_version": (
                            SENSITIVITY_SCHEMA_VERSION
                        ),
                        "sensitivity_signature": case_signature,
                        "plastic_hinge_length_factor": factor,
                        "plastic_hinge_length_method": (
                            "priestley_1996_contraflexure_expected_strength_scaled"
                        ),
                        "plastic_hinge_shear_span_method": (
                            layout.shear_span_method
                        ),
                        "baseline_case": math.isclose(
                            factor,
                            baseline_factor,
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        ),
                        "scwb_class": building["scwb_class"],
                        "scwb_strength_ratio": building[
                            "scwb_strength_ratio"
                        ],
                        "model_hash": building["model_hash"],
                        "stories": building["stories"],
                        "story_height_m": building["story_height_m"],
                        "bay_width_m": building["bay_width_m"],
                        "beam_h_m": building["beam_h_m"],
                        "column_h_m": building["column_h_m"],
                        "beam_lp_m": beam_lp_m,
                        "column_lp_m": column_lp_m,
                    }
                )
                atomic_write_json(checkpoint_path, result)
            results.append(result)

    results.sort(
        key=lambda row: (
            configured_ids.index(str(row["building_id"]))
            if configured_ids
            else next(
                index
                for index, building in enumerate(buildings)
                if str(building["building_id"])
                == str(row["building_id"])
            ),
            float(row["plastic_hinge_length_factor"]),
        )
    )
    _attach_baseline_changes(
        results,
        baseline_factor=baseline_factor,
    )
    summary = _sensitivity_summary(
        results,
        baseline_factor=baseline_factor,
    )
    output_root = Path(config["output_dir"])
    report_path = output_root / "plastic_hinge_length_sensitivity.json"
    plot_path = output_root / "plastic_hinge_length_sensitivity.png"
    curve_export_path = (
        run_root / "plastic_hinge_length_sensitivity_curves.csv"
    )
    _plot_sensitivity(
        results,
        plot_path,
        baseline_factor=baseline_factor,
    )
    _export_curve_rows(results, curve_export_path)
    report = {
        "schema_version": SENSITIVITY_SCHEMA_VERSION,
        "study_type": "parametric_plastic_hinge_length_sensitivity",
        "factors_on_empirical_lp": list(factor_values),
        "baseline_factor": baseline_factor,
        "representative_building_ids": [
            str(building["building_id"]) for building in buildings
        ],
        "representative_scwb_classes": [
            str(building["scwb_class"]) for building in buildings
        ],
        "representative_selection_note": (
            "One median-ratio building per available requested SCWB class; "
            "when a class is physically absent, remaining cases span the "
            "observed SCWB-ratio range without fabricating buildings."
        ),
        "summary": summary,
        "results": results,
        "plot_path": str(plot_path.absolute()),
        "curve_export_path": str(curve_export_path.absolute()),
        "reference_urls": list(REFERENCE_URLS),
        "scope_note": (
            "The factor range brackets the baseline to quantify sensitivity. "
            "It is not a substitute for member-test calibration."
        ),
    }
    atomic_write_json(report_path, report)
    return {
        "report_path": str(report_path.absolute()),
        "plot_path": str(plot_path.absolute()),
        "curve_export_path": str(curve_export_path.absolute()),
        "case_count": len(results),
        "valid_case_count": sum(bool(row["valid"]) for row in results),
        "summary": summary,
    }


def _modeling_scenarios(
    config: dict[str, Any],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Return unique one-factor-at-a-time SPO model configurations."""
    settings = config["modeling_sensitivity"]
    scenarios: list[tuple[str, str, dict[str, Any]]] = [
        ("baseline", "baseline", copy.deepcopy(config["model"]))
    ]
    mesh_cases = settings["fiber_mesh_cases"]
    for case_name in ("coarse", "fine"):
        model = copy.deepcopy(config["model"])
        model["fiber_mesh"] = copy.deepcopy(mesh_cases[case_name])
        scenarios.append(("fiber_mesh", case_name, model))
    for factor in settings["concrete_ultimate_strain_factors"]:
        factor = float(factor)
        if math.isclose(factor, 1.0):
            continue
        model = copy.deepcopy(config["model"])
        model["hognestad"]["ultimate_strain"] *= factor
        model["mander_confinement"]["minimum_ultimate_strain"] *= factor
        model["mander_confinement"]["maximum_ultimate_strain"] *= factor
        scenarios.append(
            ("concrete_strain", f"x{factor:.2f}", model)
        )
    for factor in settings["steel_backbone_strain_factors"]:
        factor = float(factor)
        if math.isclose(factor, 1.0):
            continue
        model = copy.deepcopy(config["model"])
        for key in ("peak_strain", "residual_strain", "failure_strain"):
            model["steel_hysteretic"][key] *= factor
        scenarios.append(("steel_strain", f"x{factor:.2f}", model))
    return scenarios


def _modeling_case_signature(
    building: dict[str, Any],
    run_config: dict[str, Any],
    scenario_group: str,
    scenario_name: str,
) -> str:
    return stable_hash(
        {
            "schema": MODELING_SENSITIVITY_SCHEMA_VERSION,
            "building_id": building["building_id"],
            "model_hash": building["model_hash"],
            "scenario_group": scenario_group,
            "scenario_name": scenario_name,
            "spo_signature": _spo_analysis_signature(building, run_config),
        }
    )


def _selected_dynamic_pair(
    config: dict[str, Any],
    building_id: str,
    t1_s: float,
) -> dict[str, Any]:
    with connect(config["database_path"]) as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT *
                FROM ground_motion_catalog
                WHERE valid=1
                ORDER BY source_set, pair_id
                """
            )
        ]
    selected, _periods, _reason, _omissions = select_motions_for_t1(
        rows,
        t1_s=float(t1_s),
        midpoint_blend_fraction=float(
            config["ground_motion"]["cms_midpoint_blend_fraction"]
        ),
        include_pwsa=bool(
            config["ground_motion"].get(
                "include_year_2568_for_every_building",
                True,
            )
        ),
    )
    if not selected:
        raise RuntimeError(
            f"{building_id}: no valid selected ground-motion pair for "
            "damping sensitivity"
        )
    cms_rows = [
        row for row in selected if row["source_set"] == "CMS_ZONE5"
    ]
    candidates = cms_rows or selected
    durations = np.asarray(
        [float(row["duration_s"]) for row in candidates],
        dtype=float,
    )
    median_duration = float(np.median(durations))
    return min(
        candidates,
        key=lambda row: (
            abs(float(row["duration_s"]) - median_duration),
            str(row["pair_id"]),
        ),
    )


def _plot_modeling_sensitivity(
    spo_results: list[dict[str, Any]],
    damping_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    scenario_order = list(
        dict.fromkeys(
            f"{row['scenario_group']}:{row['scenario_name']}"
            for row in spo_results
            if not row["baseline_case"]
        )
    )
    metrics = ("vc_kn", "dc_m", "du_m", "collapse_roof_drift")
    heatmap = np.zeros((len(metrics), len(scenario_order)), dtype=float)
    for column, scenario in enumerate(scenario_order):
        rows = [
            row
            for row in spo_results
            if f"{row['scenario_group']}:{row['scenario_name']}" == scenario
        ]
        for index, metric in enumerate(metrics):
            heatmap[index, column] = max(
                abs(float(row[f"{metric}_change_from_baseline_pct"]))
                for row in rows
            )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    image = axes[0].imshow(heatmap, cmap="YlOrRd", aspect="auto")
    axes[0].set_xticks(range(len(scenario_order)), scenario_order, rotation=35)
    axes[0].set_yticks(
        range(len(metrics)),
        ("Vc", "Dc", "Du", "Model-end drift"),
    )
    axes[0].set_title("Maximum absolute SPO change from baseline (%)")
    for row in range(heatmap.shape[0]):
        for column in range(heatmap.shape[1]):
            axes[0].text(
                column,
                row,
                f"{heatmap[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    if damping_results:
        for building_id in dict.fromkeys(
            str(row["building_id"]) for row in damping_results
        ):
            rows = [
                row
                for row in damping_results
                if str(row["building_id"]) == building_id
            ]
            rows.sort(key=lambda row: float(row["damping_ratio"]))
            axes[1].plot(
                [100.0 * float(row["damping_ratio"]) for row in rows],
                [100.0 * float(row["max_midr"]) for row in rows],
                marker="o",
                label=f"{building_id} ({rows[0]['scwb_class']})",
            )
        axes[1].set_xlabel("Modal damping ratio (%)")
        axes[1].set_ylabel("Maximum interstorey drift ratio (%)")
        axes[1].set_title("Fixed-IM NLTHA damping sensitivity")
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=7)
    else:
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.55,
            "Damping sensitivity disabled",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            transform=axes[1].transAxes,
        )
        axes[1].text(
            0.5,
            0.42,
            "Production modal damping is fixed at 5%",
            ha="center",
            va="center",
            fontsize=11,
            transform=axes[1].transAxes,
        )
    fig.suptitle(
        "RC Frame Modeling Sensitivity",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_modeling_sensitivity(
    config: dict[str, Any],
    *,
    building_ids: list[str] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run OAT SPO mesh/material studies and fixed-IM damping NLTHAs."""
    settings = config["modeling_sensitivity"]
    representative_classes = tuple(
        settings.get(
            "representative_classes",
            DEFAULT_REPRESENTATIVE_CLASSES,
        )
    )
    configured_ids = (
        list(settings.get("representative_building_ids", []))
        if building_ids is None
        else list(building_ids)
    )
    all_buildings = _load_selected_buildings(
        config,
        building_ids=configured_ids or None,
    )
    buildings = _representative_buildings(
        all_buildings,
        configured_ids=configured_ids or None,
        representative_classes=representative_classes,
    )
    run_root = Path(config["run_dir"]) / "sensitivity" / "modeling"
    spo_results: list[dict[str, Any]] = []
    scenarios = _modeling_scenarios(config)
    for building in buildings:
        for scenario_group, scenario_name, model_config in scenarios:
            scenario_tag = (
                f"{scenario_group}_{scenario_name}"
                .replace(".", "p")
                .replace(":", "_")
            )
            run_config = copy.deepcopy(config)
            run_config["model"] = model_config
            run_config["run_dir"] = str(run_root / "spo" / scenario_tag)
            signature = _modeling_case_signature(
                building,
                run_config,
                scenario_group,
                scenario_name,
            )
            checkpoint = (
                run_root
                / "spo"
                / scenario_tag
                / str(building["building_id"])
                / "sensitivity_result.json"
            )
            if resume and _checkpoint_is_reusable(
                checkpoint,
                expected_signature=signature,
            ):
                result = read_json(checkpoint)
            else:
                result = run_one_spo(building, run_config, direction="X")
                result.update(
                    {
                        "sensitivity_schema_version": (
                            MODELING_SENSITIVITY_SCHEMA_VERSION
                        ),
                        "sensitivity_signature": signature,
                        "scenario_group": scenario_group,
                        "scenario_name": scenario_name,
                        "baseline_case": scenario_group == "baseline",
                        "scwb_class": building["scwb_class"],
                        "scwb_strength_ratio": building[
                            "scwb_strength_ratio"
                        ],
                        "model_hash": building["model_hash"],
                        "stories": building["stories"],
                        "story_height_m": building["story_height_m"],
                        "model_settings": model_config,
                    }
                )
                atomic_write_json(checkpoint, result)
            spo_results.append(result)

    baseline_by_building = {
        str(row["building_id"]): row
        for row in spo_results
        if row["baseline_case"]
    }
    for row in spo_results:
        baseline = baseline_by_building[str(row["building_id"])]
        for metric in PRIMARY_RESPONSE_METRICS:
            row[f"{metric}_change_from_baseline_pct"] = (
                _relative_change_percent(
                    float(row[metric]),
                    float(baseline[metric]),
                )
            )

    damping_results: list[dict[str, Any]] = []
    damping_sensitivity_enabled = bool(
        settings.get("damping_sensitivity_enabled", True)
    )
    if damping_sensitivity_enabled:
        target_im_g = float(settings["dynamic_target_im_g"])
        for building in buildings:
            baseline_spo = baseline_by_building[str(building["building_id"])]
            dynamic_building = dict(building)
            dynamic_building["t1_s"] = float(baseline_spo["t1_s"])
            pair = _selected_dynamic_pair(
                config,
                str(building["building_id"]),
                float(dynamic_building["t1_s"]),
            )
            for damping_ratio in settings["damping_ratios"]:
                damping_ratio = float(damping_ratio)
                run_config = copy.deepcopy(config)
                run_config["model"]["damping_ratio"] = damping_ratio
                damping_tag = _factor_tag(damping_ratio)
                run_config["run_dir"] = str(
                    run_root / "damping" / f"zeta_{damping_tag}"
                )
                try:
                    result = run_nltha(
                        dynamic_building,
                        pair,
                        target_im_g,
                        run_config,
                        resume=resume,
                    )
                    result = {
                        **result,
                        "valid": result["status"]
                        in {"success", "dynamic_instability"},
                        "scwb_class": building["scwb_class"],
                        "scwb_strength_ratio": building[
                            "scwb_strength_ratio"
                        ],
                        "damping_ratio": damping_ratio,
                        "target_im_g": target_im_g,
                        "source_pair_id": pair["pair_id"],
                    }
                except ScaleFactorGuardError as exc:
                    result = {
                        "building_id": building["building_id"],
                        "pair_id": pair["pair_id"],
                        "valid": False,
                        "status": "scale_factor_guard",
                        "max_midr": math.nan,
                        "damping_ratio": damping_ratio,
                        "target_im_g": target_im_g,
                        "scwb_class": building["scwb_class"],
                        "scwb_strength_ratio": building[
                            "scwb_strength_ratio"
                        ],
                        "message": str(exc),
                    }
                damping_results.append(result)

    damping_baseline = {
        str(row["building_id"]): row
        for row in damping_results
        if math.isclose(float(row["damping_ratio"]), 0.05)
    }
    if damping_sensitivity_enabled:
        for row in damping_results:
            baseline = damping_baseline[str(row["building_id"])]
            row["max_midr_change_from_5pct_damping_pct"] = (
                _relative_change_percent(
                    float(row["max_midr"]),
                    float(baseline["max_midr"]),
                )
            )

    maximum_spo_change = max(
        (
            abs(float(row[f"{metric}_change_from_baseline_pct"]))
            for row in spo_results
            for metric in PRIMARY_RESPONSE_METRICS
            if not row["baseline_case"]
            and math.isfinite(
                float(row[f"{metric}_change_from_baseline_pct"])
            )
        ),
        default=math.nan,
    )
    scenario_maximum_spo_change: dict[str, float] = {}
    for row in spo_results:
        if row["baseline_case"]:
            continue
        scenario_key = (
            f"{row['scenario_group']}:{row['scenario_name']}"
        )
        scenario_change = max(
            (
                abs(float(row[f"{metric}_change_from_baseline_pct"]))
                for metric in PRIMARY_RESPONSE_METRICS
                if math.isfinite(
                    float(row[f"{metric}_change_from_baseline_pct"])
                )
            ),
            default=math.nan,
        )
        previous_change = scenario_maximum_spo_change.get(
            scenario_key,
            math.nan,
        )
        if math.isnan(previous_change) or scenario_change > previous_change:
            scenario_maximum_spo_change[scenario_key] = scenario_change
    fine_mesh_change = scenario_maximum_spo_change.get(
        "fiber_mesh:fine",
        math.nan,
    )
    fine_mesh_tolerance = float(
        settings["fine_mesh_convergence_tolerance_percent"]
    )
    fine_mesh_converged = (
        math.isfinite(fine_mesh_change)
        and fine_mesh_change <= fine_mesh_tolerance
    )
    maximum_damping_change_raw = max(
        (
            abs(float(row["max_midr_change_from_5pct_damping_pct"]))
            for row in damping_results
            if math.isfinite(
                float(row["max_midr_change_from_5pct_damping_pct"])
            )
        ),
        default=math.nan,
    )
    maximum_damping_change = (
        maximum_damping_change_raw
        if damping_sensitivity_enabled
        else None
    )
    output_root = Path(config["output_dir"])
    report_path = output_root / "modeling_sensitivity.json"
    plot_path = output_root / "modeling_sensitivity.png"
    report = {
        "schema_version": MODELING_SENSITIVITY_SCHEMA_VERSION,
        "study_type": "one_factor_at_a_time_modeling_sensitivity",
        "representative_building_ids": [
            str(building["building_id"]) for building in buildings
        ],
        "representative_scwb_classes": [
            str(building["scwb_class"]) for building in buildings
        ],
        "representative_selection_note": (
            "One median-ratio building per available requested SCWB class; "
            "when a class is physically absent, remaining cases span the "
            "observed SCWB-ratio range without fabricating buildings."
        ),
        "maximum_absolute_spo_change_percent": maximum_spo_change,
        "scenario_maximum_absolute_spo_change_percent": (
            scenario_maximum_spo_change
        ),
        "fine_mesh_convergence_max_change_percent": fine_mesh_change,
        "fine_mesh_convergence_tolerance_percent": fine_mesh_tolerance,
        "fine_mesh_convergence_within_tolerance": fine_mesh_converged,
        "maximum_absolute_damping_midr_change_percent": (
            maximum_damping_change
        ),
        "damping_sensitivity_enabled": damping_sensitivity_enabled,
        "production_modal_damping_ratio": float(
            config["model"]["damping_ratio"]
        ),
        "all_spo_cases_valid": all(
            bool(row["valid"]) for row in spo_results
        ),
        "all_damping_cases_valid": (
            all(bool(row["valid"]) for row in damping_results)
            if damping_sensitivity_enabled
            else None
        ),
        "spo_results": spo_results,
        "damping_results": damping_results,
        "scope_note": settings["scope_note"],
        "interpretation": (
            "Mesh sensitivity is a numerical convergence check. Lp and "
            "material-strain ranges quantify modelling uncertainty; they do "
            "not constitute calibration against physical test data. Modal "
            "damping is fixed at 5% and is not varied in this research."
        ),
        "plot_path": str(plot_path.absolute()),
    }
    _plot_modeling_sensitivity(spo_results, damping_results, plot_path)
    atomic_write_json(report_path, report)
    return {
        "report_path": str(report_path.absolute()),
        "plot_path": str(plot_path.absolute()),
        "spo_case_count": len(spo_results),
        "damping_case_count": len(damping_results),
        "maximum_absolute_spo_change_percent": maximum_spo_change,
        "fine_mesh_convergence_max_change_percent": fine_mesh_change,
        "fine_mesh_convergence_within_tolerance": fine_mesh_converged,
        "maximum_absolute_damping_midr_change_percent": (
            maximum_damping_change
        ),
        "all_spo_cases_valid": report["all_spo_cases_valid"],
        "all_damping_cases_valid": report["all_damping_cases_valid"],
    }
