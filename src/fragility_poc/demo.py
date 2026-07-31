"""Committee-facing figures generated only from persisted analysis results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .catalog import SCWB_RESEARCH_CLASSES
from .constants import (
    FEATURE_COLUMNS,
    KSC_TO_KN_M2,
    LIMIT_STATES,
    TARGET_COLUMNS,
)
from .db import connect
from .io_utils import atomic_write_json
from .structural import (
    _ops,
    define_hysteretic_steel_material,
)


def _save_figure(
    figure: plt.Figure,
    path: Path,
    *,
    dpi: int = 220,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return str(path.absolute())


def _building_figure(config: dict[str, Any], output: Path) -> str:
    building = config["building"]
    bay = float(building["bay_width_m"][1])
    bay_counts = [int(value) for value in building["number_of_bays"]]
    number_of_bays = bay_counts[len(bay_counts) // 2]
    stories = int(building["stories"])
    height = float(building["story_height_m"])
    figure = plt.figure(figsize=(6.5, 5.2))
    axis = figure.add_subplot(111, projection="3d")
    plan_width = number_of_bays * bay
    coordinates = [
        index * bay for index in range(number_of_bays + 1)
    ]
    for x in coordinates:
        for y in coordinates:
            axis.plot(
                [x, x],
                [y, y],
                [0, stories * height],
                color="#30475e",
                lw=1.5,
            )
    for level in range(1, stories + 1):
        z = level * height
        for coordinate in coordinates:
            axis.plot(
                [0, plan_width],
                [coordinate, coordinate],
                [z, z],
                color="#d1495b",
                lw=1.4,
            )
            axis.plot(
                [coordinate, coordinate],
                [0, plan_width],
                [z, z],
                color="#d1495b",
                lw=1.4,
            )
        axis.plot(
            [0, plan_width, plan_width, 0, 0],
            [0, 0, plan_width, plan_width, 0],
            [z] * 5,
            color="#7a8fa6",
            lw=0.8,
        )
    axis.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Elevation (m)")
    axis.set_title(
        f"{stories}-storey, {number_of_bays}×{number_of_bays}-bay "
        "3D RC moment-frame Demo"
    )
    axis.view_init(elev=22, azim=-56)
    return _save_figure(figure, output / "01_archetype_3d.png")


def _catalog_figure(database_path: str, output: Path) -> str | None:
    with connect(database_path) as connection:
        frame = pd.read_sql_query(
            """
            SELECT fc_ksc, number_of_bays, bay_width_m,
                   sdl_kg_m2, ll_kg_m2,
                   beam_tier, column_tier, valid, selected
            FROM building_catalog
            WHERE invalid_reason IS NULL
               OR invalid_reason <> 'superseded by current candidate library'
            """,
            connection,
        )
    if frame.empty:
        return None
    figure, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    columns = (
        ("fc_ksc", "Concrete strength (ksc)"),
        ("number_of_bays", "Bays in each direction"),
        ("bay_width_m", "Bay width (m)"),
        ("sdl_kg_m2", "SDL (kg/m²)"),
        ("ll_kg_m2", "LL (kg/m²)"),
        ("beam_tier", "Beam strength tier"),
        ("column_tier", "Column strength tier"),
    )
    selected = frame[frame["selected"] == 1]
    for axis, (column, label) in zip(axes.flat, columns):
        counts = selected[column].value_counts().sort_index()
        axis.bar([str(value) for value in counts.index], counts.values, color="#3c91e6")
        axis.set_xlabel(label)
        axis.set_ylabel("Selected Building IDs")
    for axis in axes.flat[len(columns):]:
        axis.axis("off")
    figure.suptitle(
        f"Balanced {len(selected)}-building queue from "
        f"{len(frame):,} enumerated models"
    )
    figure.tight_layout()
    return _save_figure(figure, output / "02_catalog_balance.png")


def _steel_material_figure(config: dict[str, Any], output: Path) -> str:
    """Plot the exact monotonic envelope and a cyclic material response."""
    fy_kn_m2 = (
        float(config["building"]["steel_fy_ksc"]) * KSC_TO_KN_M2
    )
    model_config = config["model"]
    ops = _ops()
    ops.wipe()
    points = define_hysteretic_steel_material(
        ops,
        base_material_tag=11,
        material_tag=1,
        fy_kn_m2=fy_kn_m2,
        model_config=model_config,
    )
    ops.testUniaxialMaterial(1)
    targets = (
        0.0,
        0.005,
        0.0,
        -0.005,
        0.0,
        0.015,
        0.0,
        -0.015,
        0.0,
        0.040,
        0.0,
        -0.040,
        0.0,
        0.080,
        0.0,
        -0.080,
        0.0,
    )
    cyclic_strain = [targets[0]]
    cyclic_stress = []
    ops.setStrain(targets[0])
    cyclic_stress.append(float(ops.getStress()))
    for start, end in zip(targets[:-1], targets[1:]):
        for strain in np.linspace(start, end, 81)[1:]:
            ops.setStrain(float(strain))
            cyclic_strain.append(float(strain))
            cyclic_stress.append(float(ops.getStress()))
    ops.wipe()

    (yield_strain, fy), (peak_strain, peak_stress), (
        residual_strain,
        residual_stress,
    ) = points
    positive_strain = np.asarray(
        [0.0, yield_strain, peak_strain, residual_strain]
    )
    positive_stress = np.asarray(
        [0.0, fy, peak_stress, residual_stress]
    )
    envelope_strain = np.concatenate(
        (-positive_strain[:0:-1], positive_strain)
    )
    envelope_stress = np.concatenate(
        (-positive_stress[:0:-1], positive_stress)
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    envelope_axis, cyclic_axis = axes
    envelope_axis.plot(
        100.0 * envelope_strain,
        envelope_stress / 1000.0,
        color="#1f5a7a",
        lw=2.2,
    )
    envelope_axis.scatter(
        100.0 * positive_strain[1:],
        positive_stress[1:] / 1000.0,
        color="#c44e52",
        zorder=4,
    )
    for label, strain, stress, offset in (
        ("Y", yield_strain, fy, (7, 7)),
        ("P", peak_strain, peak_stress, (7, 7)),
        ("R", residual_strain, residual_stress, (-18, 9)),
    ):
        envelope_axis.annotate(
            label,
            (100.0 * strain, stress / 1000.0),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )
    envelope_axis.text(
        0.03,
        0.04,
        (
            f"Y: εy={100.0 * yield_strain:.3f}%, fy={fy / 1000.0:.1f} MPa\n"
            f"P: ε={100.0 * peak_strain:.2f}%, σ=1.05fy\n"
            f"R: ε={100.0 * residual_strain:.1f}%, σ=0.20fy\n"
            "R is the residual-strength point; fracture occurs at ±10%"
        ),
        transform=envelope_axis.transAxes,
        fontsize=8,
        va="bottom",
    )
    envelope_axis.set(
        xlabel="Steel strain (%)",
        ylabel="Steel stress (MPa)",
        title="(a) Symmetric trilinear backbone",
    )

    cyclic_axis.plot(
        100.0 * np.asarray(cyclic_strain),
        np.asarray(cyclic_stress) / 1000.0,
        color="#1f5a7a",
        lw=1.25,
    )
    cyclic_axis.set(
        xlabel="Steel strain (%)",
        ylabel="Steel stress (MPa)",
        title="(b) Representative cyclic response",
    )
    for axis in axes:
        axis.axhline(0.0, color="#777777", lw=0.7)
        axis.axvline(0.0, color="#777777", lw=0.7)
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Post-peak trilinear Hysteretic reinforcement model (SD40)",
        fontsize=13,
    )
    figure.tight_layout()
    return _save_figure(
        figure,
        output / "00_steel_hysteretic_model.png",
        dpi=300,
    )


def _current_spo_features(config: dict[str, Any]) -> list[dict[str, Any]]:
    from .spo import _spo_analysis_signature

    with connect(config["database_path"]) as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT s.*, b.model_hash, b.queue_rank, b.scwb_class,
                       b.scwb_strength_ratio
                FROM spo_features s
                JOIN building_catalog b USING(building_id)
                WHERE s.valid=1 AND b.selected=1 AND b.valid=1
                ORDER BY b.queue_rank
                """
            )
        ]
    return [
        row
        for row in rows
        if row.get("analysis_signature")
        == _spo_analysis_signature(row, config)
    ]


def _spo_figure(
    config: dict[str, Any],
    output: Path,
) -> tuple[str | None, str | None]:
    current = _current_spo_features(config)
    if not current:
        return None, None
    feature = current[0]
    curve = pd.read_csv(feature["curve_path"])
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(
        curve["roof_displacement_m"],
        curve["base_shear_kn"],
        label="OpenSees SPO",
        color="#30475e",
        lw=2,
    )
    axis.plot(
        curve["roof_displacement_m"],
        curve["trilinear_kn"],
        label="Trilinear fit",
        color="#d1495b",
        ls="--",
        lw=2,
    )
    axis.scatter(
        [feature["dy_m"], feature["dc_m"], feature["du_m"]],
        [feature["vy_kn"], feature["vc_kn"], feature["vu_kn"]],
        color="#d1495b",
        zorder=4,
    )
    axis.scatter(
        [feature["run_end_displacement_m"]],
        [feature["run_end_shear_kn"]],
        marker="x",
        s=55,
        color="#222222",
        label="Run End / diagnostic endpoint",
        zorder=5,
    )
    for label, x_value, y_value in (
        ("Y", feature["dy_m"], feature["vy_kn"]),
        ("C", feature["dc_m"], feature["vc_kn"]),
        ("U", feature["du_m"], feature["vu_kn"]),
    ):
        axis.annotate(
            label,
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    endpoint_label = (
        "Model static instability"
        if feature["collapse_classification"]
        == "model_static_instability_zero_lateral_resistance"
        else "Model endpoint"
    )
    axis.annotate(
        endpoint_label,
        (
            feature["run_end_displacement_m"],
            feature["run_end_shear_kn"],
        ),
        xytext=(-6, 7),
        textcoords="offset points",
        ha="right",
        fontsize=8,
    )
    quality_text = (
        f"Energy error: total {100.0 * feature['energy_error']:.2f}% | "
        f"pre-cap {100.0 * feature['pre_capping_energy_error']:.2f}%\n"
        f"post-cap {100.0 * feature['post_capping_energy_error']:.2f}% | "
        f"NRMSE {100.0 * feature['normalized_rmse']:.2f}%\n"
        f"Endpoint: {feature['collapse_classification']}\n"
        f"Roof drift at endpoint: "
        f"{100.0 * feature['collapse_roof_drift']:.2f}%"
    )
    axis.text(
        0.98,
        0.04,
        quality_text,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
        },
    )
    axis.set(
        xlabel="Roof displacement (m)",
        ylabel="Base shear (kN)",
        title=f"SPO and trilinear fit — {feature['building_id']}",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    return (
        _save_figure(figure, output / "03_spo_trilinear.png"),
        str(feature["building_id"]),
    )


def _spo_smoke_panel(
    config: dict[str, Any],
    output: Path,
) -> str | None:
    """Plot up to three current signed SPO smoke checks side by side."""
    available = _current_spo_features(config)
    by_class = {
        class_name: [
            row
            for row in available
            if row.get("scwb_class") == class_name
        ]
        for class_name in SCWB_RESEARCH_CLASSES
    }
    if all(by_class.values()):
        # Show one result from every retained research strength-margin band.
        current = [
            min(
                by_class[SCWB_RESEARCH_CLASSES[0]],
                key=lambda row: float(row["scwb_strength_ratio"]),
            ),
            by_class[SCWB_RESEARCH_CLASSES[1]][0],
            min(
                by_class[SCWB_RESEARCH_CLASSES[2]],
                key=lambda row: abs(
                    float(row["scwb_strength_ratio"]) - 3.00
                ),
            ),
        ]
    else:
        current = available[:3]
    if len(current) < 2:
        return None
    figure, axes = plt.subplots(
        1,
        len(current),
        figsize=(5.2 * len(current), 4.6),
        squeeze=False,
    )
    for axis, feature in zip(axes[0], current):
        curve = pd.read_csv(feature["curve_path"])
        axis.plot(
            curve["roof_displacement_m"],
            curve["base_shear_kn"],
            label="OpenSees SPO",
            color="#30475e",
            lw=1.8,
        )
        axis.plot(
            curve["roof_displacement_m"],
            curve["trilinear_kn"],
            label="Trilinear fit",
            color="#d1495b",
            ls="--",
            lw=1.8,
        )
        axis.scatter(
            [feature["dy_m"], feature["dc_m"], feature["du_m"]],
            [feature["vy_kn"], feature["vc_kn"], feature["vu_kn"]],
            color="#d1495b",
            s=28,
            zorder=4,
        )
        axis.scatter(
            [feature["run_end_displacement_m"]],
            [feature["run_end_shear_kn"]],
            marker="x",
            s=42,
            color="#222222",
            zorder=5,
        )
        axis.text(
            0.98,
            0.04,
            (
                f"Energy {100.0 * feature['energy_error']:.2f}% | "
                f"pre {100.0 * feature['pre_capping_energy_error']:.2f}%\n"
                f"post {100.0 * feature['post_capping_energy_error']:.2f}% | "
                f"NRMSE {100.0 * feature['normalized_rmse']:.2f}%"
            ),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#cccccc",
                "alpha": 0.9,
            },
        )
        axis.set(
            xlabel="Roof displacement (m)",
            ylabel="Base shear (kN)",
            title=(
                f"{feature.get('scwb_class', 'SCWB class unavailable')}\n"
                f"{feature['building_id']}"
            ),
        )
        axis.grid(alpha=0.25)
    axes[0][0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Current signed SPO and trilinear smoke checks",
        fontsize=14,
    )
    figure.tight_layout()
    return _save_figure(
        figure,
        output / "03b_spo_smoke_3_buildings.png",
    )


def _ida_fragility_figures(
    config: dict[str, Any],
    output: Path,
    building_id: str | None,
) -> list[str]:
    if building_id is None:
        return []
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
            WHERE r.building_id=?
            ORDER BY r.pair_id, r.target_im_g
            """,
            (building_id,),
            )
        ]
        fragility_row = connection.execute(
            "SELECT * FROM fragility_targets WHERE building_id=?",
            (building_id,),
        ).fetchone()
        capacity_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT c.* FROM ida_capacities c
                JOIN building_ground_motion_selection s
                  ON s.building_id=c.building_id
                 AND s.pair_id=c.pair_id
                WHERE c.building_id=?
                ORDER BY c.pair_id, c.limit_state
                """,
                (building_id,),
            )
        ]
        spo_signature_row = connection.execute(
            "SELECT analysis_signature FROM spo_features WHERE building_id=?",
            (building_id,),
        ).fetchone()
    from .fragility import fragility_source_signature
    from .ida import nltha_analysis_signature

    current_run_rows = [
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
    runs = pd.DataFrame(current_run_rows)
    if fragility_row is not None and spo_signature_row is not None:
        fragility = dict(fragility_row)
        expected_fragility_signature = fragility_source_signature(
            building_id,
            str(spo_signature_row["analysis_signature"]),
            capacity_rows,
            config,
        )
        if fragility.get("source_signature") != expected_fragility_signature:
            fragility_row = None
    paths = []
    if not runs.empty:
        figure, axis = plt.subplots(figsize=(7.2, 5.0))
        for pair_id, group in runs.groupby("pair_id"):
            response = group["max_midr"].replace([np.inf], 0.15)
            axis.plot(
                response * 100.0,
                group["target_im_g"],
                lw=0.9,
                alpha=0.65,
            )
        for drift, label in ((1.0, "IO"), (2.0, "LS"), (4.0, "CP")):
            axis.axvline(drift, color="#555555", ls=":", lw=0.8)
            axis.text(drift, axis.get_ylim()[1], label, va="top", ha="right")
        axis.set(
            xlabel="Maximum interstorey drift ratio (%)",
            ylabel=r"$S_{a,GM}(T_1,5\%)$ (g)",
            title=f"Bidirectional IDA curves — {building_id}",
        )
        axis.grid(alpha=0.2)
        paths.append(_save_figure(figure, output / "04_ida_curves.png"))
    if fragility_row is not None:
        fragility = dict(fragility_row)
        intensity_max = max(4.0, 1.5 * float(fragility["theta_cp_g"]))
        intensity = np.linspace(0.001, intensity_max, 500)
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        colors = {"IO": "#3c91e6", "LS": "#f49d37", "CP": "#7a5195"}
        for state in LIMIT_STATES:
            lower = state.lower()
            theta = float(fragility[f"theta_{lower}_g"])
            beta = float(fragility[f"beta_{lower}"])
            probability = norm.cdf(np.log(intensity / theta) / beta)
            axis.plot(
                intensity,
                probability,
                label=f"{state}: θ={theta:.3f}g, β={beta:.3f}",
                color=colors[state],
                lw=2,
            )
            axis.scatter([theta], [0.5], color=colors[state], s=25)
        axis.set(
            xlabel=r"$S_{a,GM}(T_1,5\%)$ (g)",
            ylabel="Probability of exceedance",
            ylim=(0, 1),
            title=f"IDA-derived fragility — {building_id}",
        )
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
        paths.append(_save_figure(figure, output / "05_fragility.png"))
    return paths


def _parity_figure(database_path: str, output: Path) -> tuple[str | None, str | None]:
    with connect(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM ml_runs ORDER BY created_utc DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None, None
    run = dict(row)
    evaluated_path = run.get("evaluated_predictions_path")
    if not evaluated_path:
        return None, None
    prediction_path = Path(evaluated_path)
    if not prediction_path.is_file():
        return None, None
    predictions = pd.read_csv(prediction_path)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for axis, target in zip(axes.flat, TARGET_COLUMNS):
        actual = predictions[f"actual_{target}"].to_numpy(float)
        lower = float(np.min(actual))
        upper = float(np.max(actual))
        padding = max((upper - lower) * 0.08, abs(upper) * 0.02, 1.0e-5)
        bounds = (lower - padding, upper + padding)
        axis.plot(bounds, bounds, color="#555555", lw=1, ls=":")
        axis.scatter(
            actual,
            predictions[f"ridge_{target}"],
            facecolors="none",
            edgecolors="#f49d37",
            label="Ridge",
            alpha=0.8,
        )
        axis.scatter(
            actual,
            predictions[f"ann_{target}"],
            color="#3c91e6",
            label="ANN",
            alpha=0.8,
        )
        axis.set(xlabel="Full IDA", ylabel="Prediction", title=target)
        axis.set_xlim(bounds)
        axis.set_ylim(bounds)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False)
    figure.suptitle("Independent-test predictions against full IDA")
    figure.tight_layout()
    return (
        _save_figure(figure, output / "06_predicted_vs_ida.png"),
        str(run["run_id"]),
    )


def _test_fragility_overlay(
    database_path: str,
    output: Path,
) -> str | None:
    with connect(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM ml_runs ORDER BY created_utc DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    run = dict(row)
    evaluated_path = run.get("evaluated_predictions_path")
    if not evaluated_path:
        return None
    prediction_path = Path(evaluated_path)
    if not prediction_path.is_file():
        return None
    predictions = pd.read_csv(prediction_path)
    # After evaluation metrics_path points to evaluation.json; the original CV
    # file remains next to it and contains the preselected test IDs.
    original_cv_path = prediction_path.parent / "cv_selection.json"
    if not original_cv_path.is_file():
        return None
    cv_report = json.loads(original_cv_path.read_text(encoding="utf-8"))
    selection_path = Path(cv_report["run_demo_selection_path"])
    selected_ids = (
        pd.read_csv(selection_path)["building_id"].astype(str).tolist()
        if selection_path.is_file()
        else []
    )
    theta_actual = predictions[
        [f"actual_theta_{state.lower()}_g" for state in LIMIT_STATES]
    ].to_numpy(float)
    theta_ann = predictions[
        [f"ann_theta_{state.lower()}_g" for state in LIMIT_STATES]
    ].to_numpy(float)
    worst_index = int(
        np.argmax(np.mean(np.abs(theta_ann / theta_actual - 1.0), axis=1))
    )
    worst_id = str(predictions.iloc[worst_index]["building_id"])
    display_ids = []
    for building_id in [*selected_ids, worst_id]:
        if building_id not in display_ids:
            display_ids.append(building_id)
    if not display_ids:
        return None
    figure, axes = plt.subplots(
        len(display_ids),
        1,
        figsize=(8.0, 2.6 * len(display_ids)),
        squeeze=False,
    )
    intensity = np.geomspace(0.01, 4.0, 300)
    colors = {"IO": "#3c91e6", "LS": "#f49d37", "CP": "#7a5195"}
    for axis, building_id in zip(axes[:, 0], display_ids):
        result = predictions[predictions["building_id"].astype(str) == building_id]
        if result.empty:
            continue
        result_row = result.iloc[0]
        for state in LIMIT_STATES:
            lower = state.lower()
            actual_theta = float(result_row[f"actual_theta_{lower}_g"])
            actual_beta = float(result_row[f"actual_beta_{lower}"])
            ann_theta = float(result_row[f"ann_theta_{lower}_g"])
            ann_beta = float(result_row[f"ann_beta_{lower}"])
            axis.plot(
                intensity,
                norm.cdf(np.log(intensity / actual_theta) / actual_beta),
                color=colors[state],
                lw=1.8,
                label=f"{state} full IDA",
            )
            axis.plot(
                intensity,
                norm.cdf(np.log(intensity / ann_theta) / ann_beta),
                color=colors[state],
                lw=1.5,
                ls="--",
                label=f"{state} ANN",
            )
        suffix = " — worst θ error" if building_id == worst_id else ""
        axis.set(
            xscale="log",
            xlim=(0.01, 4.0),
            ylim=(0.0, 1.0),
            ylabel="P(exceedance)",
            title=f"{building_id}{suffix}",
        )
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel(r"$S_{a,GM}(T_1,5\%)$ (g)")
    axes[0, 0].legend(ncol=3, frameon=False, fontsize=8)
    figure.suptitle(
        "Preselected T1-quartile test buildings and predefined worst case"
    )
    figure.tight_layout()
    return _save_figure(figure, output / "07_test_fragility_overlays.png")


def _split_figure(database_path: str, output: Path) -> str | None:
    with connect(database_path) as connection:
        split = pd.read_sql_query(
            "SELECT split, fold, COUNT(*) AS count "
            "FROM ml_split GROUP BY split, fold",
            connection,
        )
    if split.empty:
        return None
    development = split[split["split"] == "development"].sort_values("fold")
    test_count = int(split.loc[split["split"] == "test", "count"].sum())
    figure, axis = plt.subplots(figsize=(8.5, 2.7))
    left = 0
    colors = plt.cm.Blues(np.linspace(0.38, 0.82, max(len(development), 1)))
    for color, row in zip(colors, development.itertuples(index=False)):
        axis.barh(
            ["Building IDs"],
            [row.count],
            left=left,
            color=color,
            label=f"Dev fold {int(row.fold) + 1}: {row.count}",
        )
        left += int(row.count)
    axis.barh(
        ["Building IDs"],
        [test_count],
        left=left,
        color="#d1495b",
        label=f"Independent test: {test_count}",
    )
    axis.set(
        xlabel="One Building ID = one ML observation",
        title="80% development with five-fold CV; 20% independent test",
    )
    axis.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    axis.spines[["top", "right", "left"]].set_visible(False)
    return _save_figure(figure, output / "08_ml_split.png")


def _current_fragility_rows(
    config: dict[str, Any],
    current_spo: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from .fragility import fragility_source_signature

    spo_signatures = {
        str(row["building_id"]): str(row["analysis_signature"])
        for row in current_spo
    }
    if not spo_signatures:
        return []
    building_ids = sorted(spo_signatures)
    marks = ",".join("?" for _ in building_ids)
    with connect(config["database_path"]) as connection:
        fragilities = [
            dict(row)
            for row in connection.execute(
                f"""
                SELECT f.* FROM fragility_targets f
                WHERE f.valid=1 AND f.building_id IN ({marks})
                """,
                building_ids,
            )
        ]
        capacities: dict[str, list[dict[str, Any]]] = {
            building_id: [] for building_id in building_ids
        }
        for row in connection.execute(
            f"""
            SELECT c.* FROM ida_capacities c
            JOIN building_ground_motion_selection s
              ON s.building_id=c.building_id
             AND s.pair_id=c.pair_id
            WHERE c.building_id IN ({marks})
            ORDER BY c.building_id, c.pair_id, c.limit_state
            """,
            building_ids,
        ):
            capacities[str(row["building_id"])].append(dict(row))
    return [
        row
        for row in fragilities
        if row.get("source_signature")
        == fragility_source_signature(
            str(row["building_id"]),
            spo_signatures[str(row["building_id"])],
            capacities[str(row["building_id"])],
            config,
        )
    ]


def make_demo(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["output_dir"]) / "demo"
    output.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    figures.append(_steel_material_figure(config, output))
    figures.append(_building_figure(config, output))
    catalog = _catalog_figure(config["database_path"], output)
    if catalog:
        figures.append(catalog)
    spo_path, representative = _spo_figure(config, output)
    if spo_path:
        figures.append(spo_path)
    spo_smoke_panel = _spo_smoke_panel(config, output)
    if spo_smoke_panel:
        figures.append(spo_smoke_panel)
    figures.extend(
        _ida_fragility_figures(
            config, output, representative
        )
    )
    parity, run_id = _parity_figure(config["database_path"], output)
    if parity:
        figures.append(parity)
    overlay = _test_fragility_overlay(config["database_path"], output)
    if overlay:
        figures.append(overlay)
    split = _split_figure(config["database_path"], output)
    if split:
        figures.append(split)

    current_spo = _current_spo_features(config)
    current_fragility = _current_fragility_rows(config, current_spo)
    with connect(config["database_path"]) as connection:
        counts = {
            "building_catalog": int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM building_catalog
                    WHERE invalid_reason IS NULL
                       OR invalid_reason <> 'superseded by current candidate library'
                    """
                ).fetchone()[0]
            ),
            "queue": int(
                connection.execute(
                    "SELECT COUNT(*) FROM building_catalog WHERE selected=1"
                ).fetchone()[0]
            ),
            "ground_motion_pairs": int(
                connection.execute(
                    "SELECT COUNT(DISTINCT physical_pair_hash) "
                    "FROM ground_motion_catalog WHERE valid=1"
                ).fetchone()[0]
            ),
            "valid_spo": len(current_spo),
            "complete_fragility": len(current_fragility),
        }
    current_collapses = [
        row
        for row in current_spo
        if bool(row.get("collapse_reached"))
    ]
    collapse_count = len(current_collapses)
    collapse_drifts = [
        float(row["collapse_roof_drift"])
        for row in current_collapses
        if row.get("collapse_roof_drift") is not None
    ]
    minimum_collapse_drift = (
        min(collapse_drifts) if collapse_drifts else None
    )
    maximum_collapse_drift = (
        max(collapse_drifts) if collapse_drifts else None
    )
    guard_count = sum(
        int(row.get("analysis_guard_triggered") or 0)
        for row in current_spo
    )
    report_path = output / "committee_demo.md"
    figure_lines = "\n".join(
        f"- `{Path(path).name}`" for path in figures
    )
    collapse_range_text = (
        f"{100.0 * minimum_collapse_drift:.2f}% to "
        f"{100.0 * maximum_collapse_drift:.2f}%"
        if minimum_collapse_drift is not None
        and maximum_collapse_drift is not None
        else "not available"
    )
    report_text = f"""# RC Fragility ANN Proof of Concept

## Evidence status

- Enumerated building models: {counts['building_catalog']:,}
- Balanced queue: {counts['queue']:,}
- Valid ground-motion pairs: {counts['ground_motion_pairs']}
- Valid modal/SPO analyses: {counts['valid_spo']}
- Classified flexural-model endpoints: {collapse_count}
- Model-endpoint roof-drift range: {collapse_range_text}
- SPO analysis guards triggered: {guard_count}
- Complete selected-record fragility targets: {counts['complete_fragility']}
- Representative Building ID: {representative or 'not available'}
- ML run: {run_id or 'not available'}

## Generated figures

{figure_lines}

Figures are generated from the SQLite result database. Missing downstream
figures mean the corresponding analysis gate has not yet been completed; no
synthetic IDA or fragility labels are substituted.

The raw SPO analysis does not stop at 75% of Vmax or a prescribed roof drift.
The trilinear U=0.80Vc point is retained as a feature, while the analysis
continues to the classified flexural-model endpoint. This endpoint does not
claim that excluded shear, joint, bond, bar-buckling, splice or torsional
failure modes were simulated.

The ANN input vector contains exactly {len(FEATURE_COLUMNS)} values:
{", ".join(FEATURE_COLUMNS)}. Run End and the recorded beam/story mechanism
remain diagnostic audit data and are not ANN inputs.

The available per-building CMS record count is a declared research-data
limitation. IDA does not stop at 4.0g: 4.0g is a review warning and hunting
continues until CP is observed or a numerical safety guard invalidates the
curve without creating a right-censored CP label.

HingeMidpoint is the locked v28 integration rule and actual OpenSees
integration points/weights are checked at model build. Production
Lp=max(0.08Ls+0.022fye*db, 0.044fye*db), with fye=1.25fy and
Ls=abs(M/V) from a rigid-diaphragm, inverted-triangular elastic reference
frame. Thus Ls is the member-end distance to contraflexure required by Draft
5, not the full bay width or storey height. The recorded
0.75x/1.00x/1.25x sensitivity quantifies modelling uncertainty but is not
specimen-specific laboratory calibration.

The catalogue admission rule is the minimum non-roof joint
sum(Mnc_column)/sum(Mnb_beam)>1.00. The >1.00-1.50, 1.50-3.00 and >=3.00
margin bands are used only to preserve research diversity; they are not an
ACI code-compliance classification.
"""
    report_path.write_text(report_text, encoding="utf-8")
    manifest = {
        "counts": counts,
        "collapse_summary": {
            "collapse_count": collapse_count,
            "minimum_roof_drift": minimum_collapse_drift,
            "maximum_roof_drift": maximum_collapse_drift,
            "analysis_guard_count": guard_count,
        },
        "representative_building_id": representative,
        "ml_run_id": run_id,
        "figures": figures,
        "report_path": str(report_path.absolute()),
    }
    manifest_path = output / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path.absolute())
    from .reporting import build_consolidated_workbook

    manifest["workbook"] = build_consolidated_workbook(config)
    atomic_write_json(manifest_path, manifest)
    return manifest
