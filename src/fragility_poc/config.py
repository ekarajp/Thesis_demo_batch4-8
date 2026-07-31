"""Configuration loading and path resolution."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from fragility_poc.io_utils import stable_hash
from fragility_poc.runtime_storage import configure_runtime_storage


def _resolve_project_path(project_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    # Preserve the user-facing ASCII junction (``phD``) instead of resolving
    # it back to the jSync-managed Thai reparse-point name.  Absolute paths
    # are fully usable by Windows without dereferencing directory junctions.
    return path if path.is_absolute() else (project_root / path).absolute()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).absolute()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    project_root = config_path.parent.parent.absolute()
    result = deepcopy(config)
    result["_config_path"] = str(config_path)
    result["_project_root"] = str(project_root)
    result["database_path"] = str(
        _resolve_project_path(project_root, result["database_path"])
    )
    result["output_dir"] = str(
        _resolve_project_path(project_root, result["output_dir"])
    )
    result["run_dir"] = str(_resolve_project_path(project_root, result["run_dir"]))
    result["runtime_storage_root"] = str(
        _resolve_project_path(
            project_root,
            result.get("runtime_storage_root", "runtime_storage"),
        )
    )
    result["_runtime_storage"] = configure_runtime_storage(
        project_root,
        result["runtime_storage_root"],
    )
    # Schema migration for frozen one-bay configurations created before
    # Draft 5 introduced an explicit bay-count design variable.
    result["building"].setdefault("number_of_bays", [1])
    result["building"]["capacity_design_shear"].setdefault(
        "column_candidate_axial_variation_margin",
        1.0,
    )
    gm = result["ground_motion"]
    gm["cms_root"] = str(_resolve_project_path(project_root, gm["cms_root"]))
    for key in ("pwsa_x", "pwsa_y", "year_2568_workbook"):
        resolved = _resolve_project_path(project_root, gm.get(key))
        gm[key] = str(resolved) if resolved else None
    controller = result.get("ida", {}).get("controller")
    if isinstance(controller, dict) and controller.get("model_path"):
        controller["model_path"] = str(
            _resolve_project_path(project_root, controller["model_path"])
        )
    validate_config(result)
    return result


def _positive_number(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def validate_config(config: dict[str, Any]) -> None:
    """Fail early when a configuration would make the workflow ambiguous."""
    project_root = Path(config["_project_root"]).absolute()
    writable_paths = {
        key: Path(config[key]).absolute()
        for key in (
            "database_path",
            "output_dir",
            "run_dir",
            "runtime_storage_root",
        )
    }
    for name, path in writable_paths.items():
        if path.drive.upper() == "C:":
            raise ValueError(f"{name} must not write to Drive C")
        try:
            inside_project = (
                os.path.commonpath((str(path), str(project_root)))
                == str(project_root)
            )
        except ValueError:
            inside_project = False
        if not inside_project:
            raise ValueError(
                f"{name} must remain inside the Demo Program project"
            )

    for section in ("ground_motion", "building", "model", "ida", "ml"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration section: {section}")

    ground_motion = config["ground_motion"]
    preprocessing = ground_motion.get("analysis_preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError(
            "ground_motion.analysis_preprocessing must be a mapping"
        )
    if preprocessing.get("method") != (
        "linear_least_squares_detrend_v1"
    ):
        raise ValueError(
            "Ground-motion analysis preprocessing must use the locked "
            "linear_least_squares_detrend_v1 method"
        )
    if not bool(preprocessing.get("preserve_raw_files")):
        raise ValueError(
            "Ground-motion analysis preprocessing must preserve raw files"
        )
    _positive_number(
        preprocessing["maximum_abs_residual_velocity_m_s"],
        (
            "ground_motion.analysis_preprocessing."
            "maximum_abs_residual_velocity_m_s"
        ),
    )
    _positive_number(
        preprocessing["maximum_abs_residual_displacement_m"],
        (
            "ground_motion.analysis_preprocessing."
            "maximum_abs_residual_displacement_m"
        ),
    )
    target_dt = preprocessing.get("target_dt_s")
    if target_dt is not None:
        _positive_number(
            target_dt,
            "ground_motion.analysis_preprocessing.target_dt_s",
        )
        if preprocessing.get("resample_method") != (
            "scipy_resample_poly_v1"
        ):
            raise ValueError(
                "Ground-motion resampling must use "
                "scipy_resample_poly_v1"
            )
    pwsa_window = preprocessing.get("pwsa_window_s")
    if pwsa_window is not None:
        if (
            not isinstance(pwsa_window, list)
            or len(pwsa_window) != 2
        ):
            raise ValueError(
                "ground_motion.analysis_preprocessing.pwsa_window_s "
                "must contain [start, end]"
            )
        window_start, window_end = map(float, pwsa_window)
        if window_start < 0.0 or window_end <= window_start:
            raise ValueError(
                "The PWSA processing window must have 0 <= start < end"
            )
        taper_s = float(preprocessing.get("pwsa_taper_s", 0.0))
        if (
            not math.isfinite(taper_s)
            or taper_s < 0.0
            or 2.0 * taper_s >= window_end - window_start
        ):
            raise ValueError(
                "PWSA taper must be finite, nonnegative, and shorter "
                "than half the processing window"
            )
    dataset_id = str(
        preprocessing.get("processed_dataset_id", "analysis_v1")
    )
    if not dataset_id or any(
        character not in (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789_-."
        )
        for character in dataset_id
    ):
        raise ValueError(
            "ground_motion.analysis_preprocessing.processed_dataset_id "
            "must be a filesystem-safe identifier"
        )
    if ground_motion.get("pwsa_analysis_role") != (
        "event_specific_sensitivity"
    ):
        raise ValueError(
            "ground_motion.pwsa_analysis_role must be "
            "'event_specific_sensitivity'"
        )
    if ground_motion.get("pwsa_scale_factor_policy") != "as_recorded_sf1":
        raise ValueError(
            "ground_motion.pwsa_scale_factor_policy must be "
            "'as_recorded_sf1'"
        )
    if bool(ground_motion.get("include_pwsa_in_primary_fragility", True)):
        raise ValueError(
            "PWSA must remain excluded from the primary CMS fragility fit"
        )
    if bool(ground_motion.get("require_pwsa_for_full_batch", False)):
        raise ValueError(
            "Legacy scaled-PWSA full-IDA requirement must remain disabled"
        )
    if not bool(
        ground_motion.get(
            "require_pwsa_sensitivity_for_full_batch", False
        )
    ):
        raise ValueError(
            "Every production building must retain the PWSA SF=1 "
            "event-specific sensitivity response"
        )

    building = config["building"]
    price_policy = config.get("research_price_policy")
    if not isinstance(price_policy, dict):
        raise ValueError(
            "research_price_policy must lock the embedded research prices"
        )
    if price_policy.get("mode") != "fixed_embedded_snapshot":
        raise ValueError(
            "research_price_policy.mode must be 'fixed_embedded_snapshot'"
        )
    if bool(price_policy.get("allow_runtime_download")):
        raise ValueError(
            "Runtime price downloads are forbidden for reproducibility"
        )
    if bool(price_policy.get("allow_automatic_update")):
        raise ValueError(
            "Automatic price updates are forbidden for reproducibility"
        )
    if not bool(price_policy.get("source_urls_are_provenance_only")):
        raise ValueError(
            "Price source URLs must be declared as provenance-only"
        )
    if not str(price_policy.get("snapshot_id", "")).strip():
        raise ValueError("research_price_policy.snapshot_id is required")
    stories = int(building["stories"])
    if stories <= 0:
        raise ValueError("building.stories must be positive")
    _positive_number(building["story_height_m"], "building.story_height_m")
    multipliers = [float(value) for value in building["strength_multipliers"]]
    if len(multipliers) < 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in multipliers
    ):
        raise ValueError(
            "building.strength_multipliers must contain at least two "
            "positive finite values"
        )
    if multipliers != sorted(set(multipliers)):
        raise ValueError(
            "building.strength_multipliers must be unique and increasing"
        )
    slab_design = building.get("slab_design")
    if not isinstance(slab_design, dict):
        raise ValueError("building.slab_design must be a mapping")
    if slab_design.get("selection_rule") != (
        "max_of_aci_minimum_and_one_way_shear_required_thickness"
    ):
        raise ValueError(
            "Slab thickness must be max(ACI minimum, shear-required)"
        )
    if not str(slab_design.get("code_reference", "")).startswith(
        "ACI CODE-318-25"
    ):
        raise ValueError("Slab design must identify ACI CODE-318-25")
    minimum_slab = slab_design.get("minimum_thickness")
    slab_shear = slab_design.get("one_way_shear")
    if not isinstance(minimum_slab, dict) or not isinstance(
        slab_shear, dict
    ):
        raise ValueError(
            "Slab minimum-thickness and one-way-shear settings are required"
        )
    if minimum_slab.get("method") != (
        "ACI_318_25_two_way_exterior_panel_without_drop_panel"
    ):
        raise ValueError("Unexpected ACI slab minimum-thickness method")
    for key in (
        "span_divisor",
        "steel_yield_modifier_denominator_mpa",
        "absolute_minimum_m",
    ):
        _positive_number(
            minimum_slab[key],
            f"building.slab_design.minimum_thickness.{key}",
        )
    if slab_shear.get("method") != (
        "ACI_318_25_member_without_shear_reinforcement_size_effect_SI"
    ):
        raise ValueError("Unexpected ACI slab one-way-shear method")
    if slab_shear.get("reaction_model") != (
        "equal_four_edge_support_reaction_for_square_two_way_panel"
    ):
        raise ValueError(
            "Unexpected square two-way slab reaction model"
        )
    for key in (
        "strength_reduction_factor",
        "normalweight_lambda",
        "two_way_directional_load_fraction",
        "assumed_longitudinal_reinforcement_ratio",
        "strip_width_m",
        "clear_cover_m",
        "assumed_bar_diameter_m",
        "minimum_support_width_m",
    ):
        _positive_number(
            slab_shear[key],
            f"building.slab_design.one_way_shear.{key}",
        )
    if not 0.0 < float(
        slab_shear["strength_reduction_factor"]
    ) <= 1.0:
        raise ValueError("Slab shear phi must be in (0, 1]")
    if not 0.0 < float(
        slab_shear["two_way_directional_load_fraction"]
    ) <= 1.0:
        raise ValueError(
            "Slab two-way directional load fraction must be in (0, 1]"
        )
    if (
        slab_shear.get("critical_section")
        != "distance_d_from_face_of_minimum_width_support"
        or slab_shear.get("slab_shear_reinforcement") != "none"
    ):
        raise ValueError(
            "Slab shear geometry/reinforcement assumptions are ambiguous"
        )
    slab_increment = _positive_number(
        slab_design["construction_increment_m"],
        "building.slab_design.construction_increment_m",
    )
    slab_maximum = _positive_number(
        slab_design["computational_maximum_thickness_m"],
        "building.slab_design.computational_maximum_thickness_m",
    )
    if slab_maximum <= max(
        float(minimum_slab["absolute_minimum_m"]),
        slab_increment,
    ):
        raise ValueError(
            "Slab computational maximum must exceed its minimum/increment"
        )
    beam_search = building["beam_section_search"]
    if str(beam_search.get("method")) != (
        "adaptive_discrete_expansion_without_physical_maximum"
    ):
        raise ValueError(
            "building.beam_section_search.method must use adaptive "
            "expansion without a physical maximum"
        )
    search_values = {
        key: _positive_number(
            beam_search[key],
            f"building.beam_section_search.{key}",
        )
        for key in (
            "minimum_width_m",
            "minimum_depth_m",
            "maximum_depth_to_width_ratio",
            "width_increment_m",
            "depth_increment_m",
            "initial_search_width_m",
            "initial_search_depth_m",
            "expansion_width_m",
            "expansion_depth_m",
        )
    }
    if (
        search_values["initial_search_width_m"]
        < search_values["minimum_width_m"]
        or search_values["initial_search_depth_m"]
        < search_values["minimum_depth_m"]
    ):
        raise ValueError(
            "Initial beam-search envelope cannot be below its geometric "
            "starting dimensions"
        )
    if search_values["maximum_depth_to_width_ratio"] <= 1.0:
        raise ValueError(
            "building.beam_section_search."
            "maximum_depth_to_width_ratio must exceed one"
        )
    if int(beam_search["computational_guard_expansions"]) < 1:
        raise ValueError(
            "building.beam_section_search.computational_guard_expansions "
            "must be at least one"
        )
    bar_diameters = [
        _positive_number(
            value,
            "building.beam_section_search.bar_diameters_m",
        )
        for value in beam_search["bar_diameters_m"]
    ]
    if not bar_diameters or bar_diameters != sorted(set(bar_diameters)):
        raise ValueError(
            "building.beam_section_search.bar_diameters_m must be a "
            "non-empty increasing list of unique positive values"
        )
    maximum_layers = int(
        beam_search["maximum_longitudinal_layers_per_face"]
    )
    minimum_bars_per_layer = int(beam_search["minimum_bars_per_layer"])
    if maximum_layers not in {1, 2, 3}:
        raise ValueError(
            "Beam longitudinal reinforcement currently supports one to three "
            "constructible layers per face"
        )
    if minimum_bars_per_layer < 2:
        raise ValueError(
            "Beam reinforcement layers must contain at least two bars"
        )
    three_layer_minimum_depth = _positive_number(
        beam_search["three_layer_minimum_depth_m"],
        "building.beam_section_search.three_layer_minimum_depth_m",
    )
    if maximum_layers == 3 and three_layer_minimum_depth < 0.50:
        raise ValueError(
            "Three-layer beam reinforcement must be restricted to sections "
            "at least 0.50 m deep"
        )
    if str(beam_search.get("longitudinal_diameter_rule")) != (
        "one_common_diameter_for_top_bottom_and_side_bars"
    ):
        raise ValueError(
            "Beam top, bottom and side bars must use one common standard "
            "diameter per section"
        )
    for key in ("stirrup_diameters_m", "stirrup_spacings_m"):
        values = [
            _positive_number(
                value,
                f"building.beam_section_search.{key}",
            )
            for value in beam_search[key]
        ]
        if not values or values != sorted(set(values)):
            raise ValueError(
                f"building.beam_section_search.{key} must be a non-empty "
                "increasing list of unique positive values"
            )
    stirrup_legs = [int(value) for value in beam_search["stirrup_leg_options"]]
    if (
        not stirrup_legs
        or stirrup_legs != sorted(set(stirrup_legs))
        or any(value < 2 or value % 2 for value in stirrup_legs)
    ):
        raise ValueError(
            "building.beam_section_search.stirrup_leg_options must contain "
            "unique increasing even integers of at least two"
        )
    beam_transverse_fy = beam_search[
        "transverse_fy_ksc_by_diameter_mm"
    ]
    for diameter_m in beam_search["stirrup_diameters_m"]:
        key = str(int(round(float(diameter_m) * 1000.0)))
        _positive_number(
            beam_transverse_fy[key],
            "building.beam_section_search."
            f"transverse_fy_ksc_by_diameter_mm.{key}",
        )

    column_search = building["column_section_search"]
    if str(column_search.get("method")) != (
        "adaptive_discrete_expansion_without_physical_maximum"
    ):
        raise ValueError(
            "building.column_section_search.method must use adaptive "
            "expansion without a physical maximum"
        )
    if str(column_search.get("shape_constraint")) != (
        "square_b_equals_h_for_xy_symmetric_archetype"
    ):
        raise ValueError(
            "The current symmetric multi-bay archetype requires square columns "
            "(b=h); independent rectangular dimensions would invalidate the "
            "locked X-Y equivalence check"
        )
    column_geometry = {
        key: _positive_number(
            column_search[key],
            f"building.column_section_search.{key}",
        )
        for key in (
            "minimum_size_m",
            "size_increment_m",
            "initial_search_maximum_m",
            "expansion_size_m",
        )
    }
    if (
        column_geometry["initial_search_maximum_m"]
        < column_geometry["minimum_size_m"]
    ):
        raise ValueError(
            "Initial column-search maximum cannot be below its minimum size"
        )
    if int(column_search["computational_guard_expansions"]) < 1:
        raise ValueError(
            "Column computational_guard_expansions must be at least one"
        )
    for key in ("bar_diameters_m", "hoop_diameters_m", "hoop_spacings_m"):
        values = [
            _positive_number(
                value,
                f"building.column_section_search.{key}",
            )
            for value in column_search[key]
        ]
        if not values or values != sorted(set(values)):
            raise ValueError(
                f"building.column_section_search.{key} must be a non-empty "
                "increasing list of unique positive values"
            )
    minimum_bar_count = int(column_search["minimum_bar_count"])
    bar_count_increment = int(column_search["bar_count_increment"])
    if (
        minimum_bar_count < 8
        or minimum_bar_count % 4
        or bar_count_increment < 4
        or bar_count_increment % 4
    ):
        raise ValueError(
            "Column minimum bar count and increment must be multiples of "
            "four, with a minimum of at least eight"
        )
    additional_legs = [
        int(value)
        for value in column_search["additional_hoop_leg_options"]
    ]
    if (
        not additional_legs
        or additional_legs != sorted(set(additional_legs))
        or additional_legs[0] != 0
        or any(value < 0 for value in additional_legs)
    ):
        raise ValueError(
            "Column additional hoop-leg options must be unique increasing "
            "nonnegative integers beginning with zero"
        )
    _positive_number(
        column_search["maximum_lateral_support_spacing_m"],
        "building.column_section_search.maximum_lateral_support_spacing_m",
    )
    column_transverse_fy = column_search[
        "transverse_fy_ksc_by_diameter_mm"
    ]
    for diameter_m in column_search["hoop_diameters_m"]:
        key = str(int(round(float(diameter_m) * 1000.0)))
        _positive_number(
            column_transverse_fy[key],
            "building.column_section_search."
            f"transverse_fy_ksc_by_diameter_mm.{key}",
        )
    serviceability = building["beam_serviceability"]
    if str(serviceability.get("method")) != (
        "Branson_effective_inertia_simply_supported_uniform_load"
    ):
        raise ValueError(
            "building.beam_serviceability.method must identify the locked "
            "Branson effective-inertia calculation"
        )
    _positive_number(
        serviceability["immediate_live_deflection_limit_ratio"],
        "building.beam_serviceability."
        "immediate_live_deflection_limit_ratio",
    )
    _positive_number(
        serviceability["total_long_term_deflection_limit_ratio"],
        "building.beam_serviceability."
        "total_long_term_deflection_limit_ratio",
    )
    sustained_fraction = float(
        serviceability["sustained_live_load_fraction"]
    )
    if (
        not math.isfinite(sustained_fraction)
        or not 0.0 <= sustained_fraction <= 1.0
    ):
        raise ValueError(
            "building.beam_serviceability.sustained_live_load_fraction "
            "must be between zero and one"
        )
    long_term_multiplier = float(
        serviceability["long_term_deflection_multiplier"]
    )
    if not math.isfinite(long_term_multiplier) or long_term_multiplier < 0.0:
        raise ValueError(
            "building.beam_serviceability.long_term_deflection_multiplier "
            "cannot be negative"
        )
    moment_coefficients = building["beam_gravity_moment_coefficients"]
    for key in ("negative_support_wl2", "positive_midspan_wl2"):
        _positive_number(
            moment_coefficients[key],
            f"building.beam_gravity_moment_coefficients.{key}",
        )
    if str(moment_coefficients.get("analysis_model")) != (
        "fixed_fixed_uniform_load"
    ):
        raise ValueError(
            "Beam gravity moment coefficients must identify the locked "
            "fixed-fixed uniform-load model"
        )
    tributary_rule = str(
        moment_coefficients.get("tributary_width_rule")
    )
    accepted_tributary_rules = {
        "equal_panel_edge_reactions_perimeter_bay_over_4_"
        "interior_bay_over_2",
    }
    if building.get("number_of_bays", [1]) == [1]:
        accepted_tributary_rules.add(
            "equal_four_edge_share_bay_over_4"
        )
    if tributary_rule not in accepted_tributary_rules:
        raise ValueError(
            "Beam gravity tributary width must match the OpenSees two-way "
            "panel-edge allocation (perimeter bay/4; interior bay/2)"
        )
    firefly = building["firefly_optimization"]
    if not bool(firefly.get("enabled")):
        raise ValueError("building.firefly_optimization.enabled must be true")
    for key in ("population_size", "maximum_iterations", "stall_iterations"):
        if int(firefly[key]) < 2:
            raise ValueError(
                f"building.firefly_optimization.{key} must be at least two"
            )
    if int(firefly["multi_start_runs"]) < 2:
        raise ValueError(
            "building.firefly_optimization.multi_start_runs must be at "
            "least two"
        )
    if int(firefly["maximum_search_candidates"]) < int(
        firefly["population_size"]
    ):
        raise ValueError(
            "Firefly maximum_search_candidates cannot be smaller than "
            "population_size"
        )
    for key in ("beta_zero", "alpha_initial"):
        _positive_number(
            firefly[key],
            f"building.firefly_optimization.{key}",
        )
    gamma = float(firefly["gamma"])
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError(
            "building.firefly_optimization.gamma cannot be negative"
        )
    alpha_decay = float(firefly["alpha_decay"])
    if not 0.0 < alpha_decay <= 1.0:
        raise ValueError(
            "building.firefly_optimization.alpha_decay must be in (0, 1]"
        )
    if not bool(firefly.get("exact_optimality_audit")):
        raise ValueError(
            "Firefly results must retain the exact optimality audit"
        )
    if not str(firefly.get("unit_cost_basis", "")).strip():
        raise ValueError(
            "Firefly unit_cost_basis must identify the cost provenance"
        )
    snapshot = firefly.get("price_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("Firefly price_snapshot must be a mapping")
    for key in (
        "agency",
        "region_code",
        "region_name",
        "retrieved_date",
        "price_definition",
        "portal_url",
    ):
        if not str(snapshot.get(key, "")).strip():
            raise ValueError(f"Firefly price_snapshot.{key} is required")
    if int(snapshot["year_be"]) < 2500 or not 1 <= int(snapshot["month"]) <= 12:
        raise ValueError("Firefly price snapshot period is invalid")
    unit_costs = firefly["unit_costs"]
    locked_price_hash = stable_hash(
        {
            "price_snapshot": snapshot,
            "unit_costs": unit_costs,
        }
    )
    if (
        locked_price_hash
        != price_policy.get("locked_price_payload_sha256")
    ):
        raise ValueError(
            "Embedded research prices do not match the locked checksum. "
            "Do not edit prices in place; create a new snapshot_id and "
            "checksum for a separately reported research scenario."
        )
    for key in (
        "formwork_per_m2",
        "steel_density_kg_m3",
        "transverse_hook_length_bar_diameters",
    ):
        _positive_number(
            unit_costs[key],
            f"building.firefly_optimization.unit_costs.{key}",
        )
    concrete_costs = unit_costs["concrete_per_m3_by_fc_ksc"]
    for fc_ksc in building["fc_ksc"]:
        key = str(int(round(float(fc_ksc))))
        _positive_number(
            concrete_costs[key],
            "building.firefly_optimization.unit_costs."
            f"concrete_per_m3_by_fc_ksc.{key}",
        )
    longitudinal_costs = unit_costs[
        "longitudinal_steel_per_kg_by_diameter_mm"
    ]
    for diameter_m in set(
        beam_search["bar_diameters_m"]
        + column_search["bar_diameters_m"]
    ):
        key = str(int(round(float(diameter_m) * 1000.0)))
        _positive_number(
            longitudinal_costs[key],
            "building.firefly_optimization.unit_costs."
            f"longitudinal_steel_per_kg_by_diameter_mm.{key}",
        )
    transverse_costs = unit_costs[
        "transverse_steel_per_kg_by_diameter_mm"
    ]
    for diameter_m in set(
        beam_search["stirrup_diameters_m"]
        + column_search["hoop_diameters_m"]
    ):
        key = str(int(round(float(diameter_m) * 1000.0)))
        _positive_number(
            transverse_costs[key],
            "building.firefly_optimization.unit_costs."
            f"transverse_steel_per_kg_by_diameter_mm.{key}",
        )
    gravity_combination = building["gravity_strength_combination"]
    dead_load_factor = _positive_number(
        gravity_combination["dead_load_factor"],
        "building.gravity_strength_combination.dead_load_factor",
    )
    live_load_factor = _positive_number(
        gravity_combination["live_load_factor"],
        "building.gravity_strength_combination.live_load_factor",
    )
    if str(gravity_combination.get("method")) != "strength_design":
        raise ValueError(
            "building.gravity_strength_combination.method must be "
            "'strength_design'"
        )
    if not str(gravity_combination.get("standard_reference", "")).strip():
        raise ValueError(
            "building.gravity_strength_combination.standard_reference "
            "must identify the governing standard"
        )
    if dead_load_factor < 1.0 or live_load_factor < 1.0:
        raise ValueError(
            "Gravity strength-design load factors must not be below 1.0"
        )
    capacity_shear = building["capacity_design_shear"]
    if not bool(capacity_shear.get("enabled")):
        raise ValueError(
            "building.capacity_design_shear.enabled must remain true so the "
            "fiber-hinge models are screened as flexure-controlled"
        )
    probable_factor = _positive_number(
        capacity_shear["probable_steel_strength_factor"],
        "building.capacity_design_shear.probable_steel_strength_factor",
    )
    shear_phi = _positive_number(
        capacity_shear["strength_reduction_factor"],
        "building.capacity_design_shear.strength_reduction_factor",
    )
    beam_span_ratio = _positive_number(
        capacity_shear["beam_preselection_clear_span_ratio"],
        "building.capacity_design_shear.beam_preselection_clear_span_ratio",
    )
    column_axial_margin = _positive_number(
        capacity_shear["column_candidate_axial_variation_margin"],
        "building.capacity_design_shear."
        "column_candidate_axial_variation_margin",
    )
    if probable_factor < 1.0:
        raise ValueError(
            "Capacity-design probable steel strength factor cannot be below "
            "one"
        )
    if shear_phi > 1.0:
        raise ValueError(
            "Capacity-design shear strength-reduction factor cannot exceed one"
        )
    if beam_span_ratio > 1.0:
        raise ValueError(
            "Capacity-design beam preselection clear-span ratio cannot exceed "
            "one"
        )
    if column_axial_margin < 1.0:
        raise ValueError(
            "Column candidate axial-variation margin cannot be below one"
        )
    for key in ("beam_demand_rule", "column_demand_rule", "purpose"):
        if not str(capacity_shear.get(key, "")).strip():
            raise ValueError(
                f"building.capacity_design_shear.{key} is required"
            )
    scwb = building["scwb_research_database"]
    scwb_minimum = _positive_number(
        scwb["minimum_ratio"],
        "building.scwb_research_database.minimum_ratio",
    )
    scwb_medium = _positive_number(
        scwb["medium_margin_ratio"],
        "building.scwb_research_database.medium_margin_ratio",
    )
    scwb_high = _positive_number(
        scwb["high_margin_ratio"],
        "building.scwb_research_database.high_margin_ratio",
    )
    if not scwb_minimum < scwb_medium < scwb_high:
        raise ValueError(
            "Research column/beam strength thresholds must satisfy "
            "minimum_ratio < medium_margin_ratio < high_margin_ratio"
        )
    if not math.isclose(scwb_minimum, 1.0, abs_tol=1.0e-12):
        raise ValueError(
            "Research column/beam admission threshold is locked at 1.00"
        )
    if not bool(scwb.get("strictly_greater_than_minimum", False)):
        raise ValueError(
            "Research column/beam admission must require ratio > 1.00"
        )
    if bool(scwb.get("code_compliance_claim", True)):
        raise ValueError(
            "The research column/beam ratio bands must not claim code "
            "compliance"
        )
    if not str(scwb.get("acceptance_rule", "")).strip():
        raise ValueError(
            "building.scwb_research_database.acceptance_rule is required"
        )
    bay_counts = building.get("number_of_bays")
    if not isinstance(bay_counts, list) or not bay_counts:
        raise ValueError(
            "building.number_of_bays must contain at least one integer"
        )
    normalized_bay_counts = [int(value) for value in bay_counts]
    if any(
        isinstance(value, bool)
        or float(value) != int(value)
        or int(value) < 1
        for value in bay_counts
    ):
        raise ValueError(
            "building.number_of_bays must contain positive integers"
        )
    if normalized_bay_counts != sorted(set(normalized_bay_counts)):
        raise ValueError(
            "building.number_of_bays must be unique and increasing"
        )
    for key in ("fc_ksc", "bay_width_m", "sdl_kg_m2", "ll_kg_m2"):
        values = [float(value) for value in building[key]]
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError(f"building.{key} must contain finite values")
        if key in {"fc_ksc", "bay_width_m"} and any(
            value <= 0.0 for value in values
        ):
            raise ValueError(f"building.{key} must contain positive values")
        if key in {"sdl_kg_m2", "ll_kg_m2"} and any(
            value < 0.0 for value in values
        ):
            raise ValueError(f"building.{key} cannot contain negative values")

    model = config["model"]
    damping_ratio = _positive_number(
        model["damping_ratio"], "model.damping_ratio"
    )
    if damping_ratio >= 1.0:
        raise ValueError("model.damping_ratio must be below 1.0")
    if str(model.get("damping_model", "modal")).lower() not in {
        "modal",
        "rayleigh_initial",
    }:
        raise ValueError(
            "model.damping_model must be 'modal' or 'rayleigh_initial'"
        )
    if int(model.get("modal_mode_count", 12)) < 2:
        raise ValueError("model.modal_mode_count must be at least 2")
    if str(model.get("modal_identification_method", "")) != (
        "directional_effective_modal_mass_distinct_assignment_v1"
    ):
        raise ValueError(
            "model.modal_identification_method must use directional "
            "effective modal mass with distinct X/Y assignment"
        )
    modal_mass_capture = float(
        model.get("minimum_cumulative_translational_mass_ratio", 0.0)
    )
    if (
        not math.isfinite(modal_mass_capture)
        or not 0.0 < modal_mass_capture <= 1.0
    ):
        raise ValueError(
            "model.minimum_cumulative_translational_mass_ratio must lie "
            "in (0, 1]"
        )
    if str(model.get("beam_section_model", "")) != (
        "rectangular_no_effective_flange_conservative"
    ):
        raise ValueError(
            "model.beam_section_model must retain the conservative "
            "rectangular no-effective-flange assumption"
        )
    if str(model.get("collapse_scope", "")) != (
        "flexure_axial_material_and_global_instability_only_excludes_"
        "shear_joint_bond_bar_buckling_splice_and_torsional_failure"
    ):
        raise ValueError(
            "model.collapse_scope must explicitly declare the retained and "
            "excluded failure mechanisms"
        )
    if not str(
        model.get("dynamic_instability_interpretation", "")
    ).strip():
        raise ValueError(
            "model.dynamic_instability_interpretation is required"
        )
    fiber_mesh = model.get("fiber_mesh", {})
    for key in (
        "core_y",
        "core_z",
        "cover_thickness",
        "cover_length_y",
        "cover_length_z",
    ):
        if int(fiber_mesh.get(key, 0)) < 1:
            raise ValueError(
                f"model.fiber_mesh.{key} must be a positive integer"
            )
    if not str(fiber_mesh.get("baseline_id", "")).strip():
        raise ValueError("model.fiber_mesh.baseline_id is required")
    if str(model.get("eigen_solver", "fullGenLapack")).removeprefix("-") not in {
        "fullGenLapack",
        "genBandArpack",
    }:
        raise ValueError(
            "model.eigen_solver must be 'fullGenLapack' or 'genBandArpack'"
        )
    if str(model.get("midr_combination", "max_component")) != "max_component":
        raise ValueError(
            "model.midr_combination must be 'max_component' to match "
            "Draft 4 Equation (3.21)"
        )
    if int(config["ida"].get("local_recovery_chunk_steps", 200)) < 1:
        raise ValueError(
            "ida.local_recovery_chunk_steps must be at least one"
        )
    hognestad = model["hognestad"]
    peak_strain = _positive_number(
        hognestad["peak_strain"], "model.hognestad.peak_strain"
    )
    ultimate_strain = _positive_number(
        hognestad["ultimate_strain"],
        "model.hognestad.ultimate_strain",
    )
    if ultimate_strain <= peak_strain:
        raise ValueError(
            "model.hognestad.ultimate_strain must exceed peak_strain"
        )
    confinement = model["mander_confinement"]
    if str(confinement["unequal_pressure_rule"]) != (
        "conservative_minimum_effective_lateral_pressure"
    ):
        raise ValueError(
            "model.mander_confinement.unequal_pressure_rule must use the "
            "documented conservative minimum-pressure rule"
        )
    transverse_ultimate_strain = _positive_number(
        confinement["transverse_ultimate_strain"],
        "model.mander_confinement.transverse_ultimate_strain",
    )
    minimum_confined_strain = _positive_number(
        confinement["minimum_ultimate_strain"],
        "model.mander_confinement.minimum_ultimate_strain",
    )
    maximum_confined_strain = _positive_number(
        confinement["maximum_ultimate_strain"],
        "model.mander_confinement.maximum_ultimate_strain",
    )
    if minimum_confined_strain < ultimate_strain:
        raise ValueError(
            "Confined-concrete minimum ultimate strain cannot be below the "
            "unconfined Hognestad ultimate strain"
        )
    if maximum_confined_strain < minimum_confined_strain:
        raise ValueError(
            "Confined-concrete maximum ultimate strain must exceed its minimum"
        )
    if transverse_ultimate_strain <= maximum_confined_strain:
        raise ValueError(
            "Transverse-steel ultimate strain must exceed the concrete limit"
        )
    detailing = model["material_detailing"]
    for key in (
        "clear_cover_m",
        "beam_minimum_clear_bar_spacing_m",
        "nominal_maximum_coarse_aggregate_size_m",
        "beam_maximum_side_bar_spacing_m",
        "beam_hoop_diameter_m",
        "beam_hoop_spacing_m",
        "column_minimum_clear_bar_spacing_m",
        "column_hoop_diameter_m",
        "column_hoop_spacing_m",
    ):
        _positive_number(detailing[key], f"model.material_detailing.{key}")
    for key in (
        "beam_hoop_legs_each_axis",
        "column_hoop_legs_x",
        "column_hoop_legs_y",
    ):
        if int(detailing[key]) < 2:
            raise ValueError(
                f"model.material_detailing.{key} must be at least 2"
            )
    if (
        float(detailing["beam_maximum_side_bar_spacing_m"])
        <= float(detailing["beam_minimum_clear_bar_spacing_m"])
    ):
        raise ValueError(
            "Beam maximum side-bar spacing must exceed the minimum clear "
            "longitudinal-bar spacing"
        )
    if str(detailing.get("beam_clear_spacing_rule")) != (
        "ACI_CODE_318_25_max_25mm_db_and_4over3_nominal_maximum_aggregate"
    ):
        raise ValueError(
            "Beam clear spacing must follow the locked ACI aggregate-aware "
            "rule"
        )
    steel = model["steel_hysteretic"]
    _positive_number(
        steel["elastic_modulus_mpa"],
        "model.steel_hysteretic.elastic_modulus_mpa",
    )
    if float(steel["peak_strength_ratio"]) < 1.0:
        raise ValueError(
            "model.steel_hysteretic.peak_strength_ratio must be at least 1"
        )
    steel_strains = [
        _positive_number(
            steel["peak_strain"],
            "model.steel_hysteretic.peak_strain",
        ),
        _positive_number(
            steel["residual_strain"],
            "model.steel_hysteretic.residual_strain",
        ),
        _positive_number(
            steel["failure_strain"],
            "model.steel_hysteretic.failure_strain",
        ),
    ]
    if steel_strains != sorted(steel_strains) or len(set(steel_strains)) != 3:
        raise ValueError(
            "Steel strains must satisfy peak < residual < failure"
        )
    residual_ratio = float(steel["residual_strength_ratio"])
    if not 0.0 <= residual_ratio < float(steel["peak_strength_ratio"]):
        raise ValueError(
            "Steel residual-strength ratio must be nonnegative and below peak"
        )
    for key in ("pinch_x", "pinch_y"):
        value = float(steel[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"model.steel_hysteretic.{key} must lie within [0, 1]"
            )
    supported_integrations = {
        "HingeMidpoint",
        "HingeEndpoint",
        "HingeRadauTwo",
        "HingeRadau",
    }
    if str(model.get("beam_integration")) not in supported_integrations:
        raise ValueError(
            "model.beam_integration must be one of "
            + ", ".join(sorted(supported_integrations))
        )
    supported_hinge_length_methods = {
        "priestley_1996_contraflexure_expected_strength",
        "priestley_1996_contraflexure_expected_strength_scaled",
        "priestley_1992_expected_strength",
        "priestley_1992_expected_strength_scaled",
        "section_depth_factor",
        "0.5_section_depth",
        "parametric_section_depth_sensitivity",
    }
    hinge_length_method = str(
        model.get("plastic_hinge_length_method")
    )
    if hinge_length_method not in supported_hinge_length_methods:
        raise ValueError(
            "model.plastic_hinge_length_method must be one of "
            + ", ".join(sorted(supported_hinge_length_methods))
        )
    _positive_number(
        model["plastic_hinge_length_factor"],
        "model.plastic_hinge_length_factor",
    )
    _positive_number(
        model["plastic_hinge_length_scale_factor"],
        "model.plastic_hinge_length_scale_factor",
    )
    _positive_number(
        model["plastic_hinge_expected_strength_factor"],
        "model.plastic_hinge_expected_strength_factor",
    )
    shear_span_method = str(
        model.get(
            "plastic_hinge_shear_span_method",
            "elastic_reference_m_over_v",
        )
    )
    if shear_span_method not in {
        "elastic_reference_m_over_v",
        "legacy_full_member",
    }:
        raise ValueError(
            "model.plastic_hinge_shear_span_method must be "
            "elastic_reference_m_over_v or legacy_full_member"
        )
    if shear_span_method == "elastic_reference_m_over_v":
        if (
            str(model.get("plastic_hinge_reference_load_pattern"))
            != "inverted_triangular"
        ):
            raise ValueError(
                "model.plastic_hinge_reference_load_pattern must be "
                "inverted_triangular"
            )
        if (
            str(
                model.get(
                    "plastic_hinge_column_bidirectional_combination"
                )
            )
            != "geometric_mean"
        ):
            raise ValueError(
                "model.plastic_hinge_column_bidirectional_combination must "
                "be geometric_mean"
            )
        minimum_span_ratio = _positive_number(
            model["plastic_hinge_minimum_shear_span_ratio"],
            "model.plastic_hinge_minimum_shear_span_ratio",
        )
        maximum_span_ratio = _positive_number(
            model["plastic_hinge_maximum_shear_span_ratio"],
            "model.plastic_hinge_maximum_shear_span_ratio",
        )
        if not 0.0 < minimum_span_ratio <= maximum_span_ratio <= 1.0:
            raise ValueError(
                "Plastic-hinge shear-span ratios must satisfy "
                "0 < minimum <= maximum <= 1"
            )
    if int(model["mechanism_sampling_interval_steps"]) <= 0:
        raise ValueError(
            "model.mechanism_sampling_interval_steps must be positive"
        )
    _positive_number(
        model["mechanism_yield_strain_ratio"],
        "model.mechanism_yield_strain_ratio",
    )
    story_fraction = float(
        model["mechanism_story_column_fraction_threshold"]
    )
    if not 0.0 < story_fraction <= 1.0:
        raise ValueError(
            "model.mechanism_story_column_fraction_threshold must lie in "
            "(0, 1]"
        )
    sensitivity = config["modeling_sensitivity"]
    mesh_cases = sensitivity["fiber_mesh_cases"]
    required_mesh_cases = {"coarse", "baseline", "fine"}
    missing_mesh_cases = required_mesh_cases.difference(mesh_cases)
    if missing_mesh_cases:
        raise ValueError(
            "modeling_sensitivity.fiber_mesh_cases is missing required "
            f"cases: {sorted(missing_mesh_cases)}"
        )
    for case_name, mesh_case in mesh_cases.items():
        for key in (
            "core_y",
            "core_z",
            "cover_thickness",
            "cover_length_y",
            "cover_length_z",
        ):
            if int(mesh_case.get(key, 0)) < 1:
                raise ValueError(
                    "modeling_sensitivity.fiber_mesh_cases."
                    f"{case_name}.{key} must be positive"
                )
    baseline_mesh = {
        key: model["fiber_mesh"][key]
        for key in (
            "core_y",
            "core_z",
            "cover_thickness",
            "cover_length_y",
            "cover_length_z",
            "baseline_id",
        )
    }
    if dict(mesh_cases["baseline"]) != baseline_mesh:
        raise ValueError(
            "The modeling-sensitivity baseline mesh must exactly match "
            "model.fiber_mesh"
        )
    fine_mesh_tolerance = _positive_number(
        sensitivity["fine_mesh_convergence_tolerance_percent"],
        "modeling_sensitivity.fine_mesh_convergence_tolerance_percent",
    )
    if fine_mesh_tolerance > 100.0:
        raise ValueError(
            "The fine-mesh convergence tolerance cannot exceed 100 percent"
        )
    for key in (
        "concrete_ultimate_strain_factors",
        "steel_backbone_strain_factors",
    ):
        values = [float(value) for value in sensitivity[key]]
        if len(values) < 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError(
                f"modeling_sensitivity.{key} requires at least three "
                "positive finite values"
            )
    if 1.0 not in {
        float(value)
        for value in sensitivity["concrete_ultimate_strain_factors"]
    } or 1.0 not in {
        float(value)
        for value in sensitivity["steel_backbone_strain_factors"]
    }:
        raise ValueError(
            "Material-strain sensitivity ranges must include factor 1.0"
        )
    damping_values = [
        float(value) for value in sensitivity["damping_ratios"]
    ]
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in damping_values
    ):
        raise ValueError(
            "modeling_sensitivity.damping_ratios must be positive and finite"
        )
    damping_sensitivity_enabled = bool(
        sensitivity.get("damping_sensitivity_enabled", True)
    )
    if damping_sensitivity_enabled:
        if len(damping_values) < 3 or not any(
            math.isclose(value, damping_ratio, abs_tol=1.0e-12)
            for value in damping_values
        ):
            raise ValueError(
                "Enabled damping sensitivity requires at least three ratios "
                "including the configured baseline"
            )
        _positive_number(
            sensitivity["dynamic_target_im_g"],
            "modeling_sensitivity.dynamic_target_im_g",
        )
    elif (
        len(damping_values) != 1
        or not math.isclose(
            damping_values[0],
            damping_ratio,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "When damping sensitivity is disabled, damping_ratios must "
            "contain only the production damping ratio"
        )

    ida = config["ida"]
    minimum_im = _positive_number(ida["minimum_im_g"], "ida.minimum_im_g")
    initial_im = _positive_number(ida["initial_im_g"], "ida.initial_im_g")
    soft_warning_im = _positive_number(
        ida["cp_hunt_soft_warning_im_g"],
        "ida.cp_hunt_soft_warning_im_g",
    )
    if not minimum_im < initial_im < soft_warning_im:
        raise ValueError(
            "IDA intensities must satisfy minimum_im_g < initial_im_g < "
            "cp_hunt_soft_warning_im_g"
        )
    if float(ida["hunt_multiplier"]) <= 1.0:
        raise ValueError("ida.hunt_multiplier must exceed 1.0")
    if not 0.0 < float(ida["bracket_tolerance"]) < 1.0:
        raise ValueError("ida.bracket_tolerance must lie in (0, 1)")
    controller = ida.get("controller", {"mode": "legacy_hunt_fill"})
    if not isinstance(controller, dict):
        raise ValueError("ida.controller must be an object")
    controller_mode = str(controller.get("mode", "legacy_hunt_fill"))
    if controller_mode not in {"legacy_hunt_fill", "hybrid_active_v1"}:
        raise ValueError(
            "ida.controller.mode must be legacy_hunt_fill or "
            "hybrid_active_v1"
        )
    if controller_mode == "hybrid_active_v1":
        required_controller_fields = {
            "model_path",
            "model_sha256",
            "use_frozen_surrogate",
            "require_frozen_surrogate",
            "surrogate_blend_weight",
            "prediction_clamp_ratio",
            "spo_capacity_ratio_seed",
            "seed_lower_io_factor",
            "seed_upper_cp_factor",
            "refinement_strategy",
            "minimum_refinements_per_limit_state",
            "transition_lower_response_ratio",
            "soft_review_point_count",
        }
        missing = sorted(required_controller_fields - set(controller))
        if missing:
            raise ValueError(
                "Hybrid active IDA controller is missing: "
                + ", ".join(missing)
            )
        if not 0.0 <= float(controller["surrogate_blend_weight"]) <= 1.0:
            raise ValueError(
                "ida.controller.surrogate_blend_weight must lie in [0, 1]"
            )
        clamp = [float(value) for value in controller["prediction_clamp_ratio"]]
        if len(clamp) != 2 or not 0.0 < clamp[0] <= 1.0 <= clamp[1]:
            raise ValueError(
                "ida.controller.prediction_clamp_ratio must straddle 1.0"
            )
        ratios = [
            float(controller["spo_capacity_ratio_seed"][state])
            for state in ("IO", "LS", "CP")
        ]
        if ratios != sorted(ratios) or min(ratios) <= 0.0:
            raise ValueError(
                "SPO capacity seed ratios must be positive and ordered"
            )
        lower_factor = float(controller["seed_lower_io_factor"])
        upper_factor = float(controller["seed_upper_cp_factor"])
        if not 0.0 < lower_factor < 1.0:
            raise ValueError(
                "ida.controller.seed_lower_io_factor must lie in (0, 1)"
            )
        if upper_factor <= 1.0:
            raise ValueError(
                "ida.controller.seed_upper_cp_factor must exceed 1.0"
            )
        if controller["refinement_strategy"] != "minimax_log_bisection":
            raise ValueError(
                "ida.controller.refinement_strategy must be "
                "minimax_log_bisection"
            )
        minimum_refinements = int(
            controller["minimum_refinements_per_limit_state"]
        )
        if minimum_refinements < 0:
            raise ValueError(
                "Hybrid active IDA minimum refinement count cannot be negative"
            )
        transition_ratio = float(
            controller["transition_lower_response_ratio"]
        )
        if not 0.0 < transition_ratio < 1.0:
            raise ValueError(
                "ida.controller.transition_lower_response_ratio must lie "
                "in (0, 1)"
            )
        if int(controller["soft_review_point_count"]) < 4:
            raise ValueError(
                "ida.controller.soft_review_point_count must be at least 4"
            )
    thresholds = [
        float(ida["limit_state_midr"][state])
        for state in ("IO", "LS", "CP")
    ]
    if thresholds != sorted(thresholds) or len(set(thresholds)) != 3:
        raise ValueError(
            "IDA limit-state MIDR thresholds must satisfy IO < LS < CP"
        )
    if int(ida["workers"]) <= 0:
        raise ValueError("ida.workers must be positive")
    if int(ida.get("native_worker_restart_limit", 2)) <= 0:
        raise ValueError(
            "ida.native_worker_restart_limit must be positive"
        )
    if int(ida["maximum_hunt_steps"]) < 2:
        raise ValueError("ida.maximum_hunt_steps must be at least 2")
    _positive_number(
        ida["maximum_scale_factor_guard"],
        "ida.maximum_scale_factor_guard",
    )
    review_scale = _positive_number(
        ida["scale_factor_review_threshold"],
        "ida.scale_factor_review_threshold",
    )
    if review_scale >= float(ida["maximum_scale_factor_guard"]):
        raise ValueError(
            "ida.scale_factor_review_threshold must be below the numerical "
            "safety guard"
        )
    if not bool(ida.get("require_cp_crossing", True)):
        raise ValueError(
            "Production configuration must require an observed CP crossing"
        )
    planned = int(ida.get("planned_full_ida_buildings", 0))
    if planned <= 0:
        raise ValueError("ida.planned_full_ida_buildings must be positive")

    ml = config["ml"]
    if not 0.0 < float(ml["test_fraction"]) < 0.5:
        raise ValueError("ml.test_fraction must lie in (0, 0.5)")
    if int(ml["cv_folds"]) < 2:
        raise ValueError("ml.cv_folds must be at least 2")
    if not ml["architectures"] or any(
        not candidate
        or any(int(width) <= 0 for width in candidate)
        for candidate in ml["architectures"]
    ):
        raise ValueError("ml.architectures must contain positive layer widths")
    if not ml["alphas"] or any(float(value) < 0.0 for value in ml["alphas"]):
        raise ValueError("ml.alphas must contain nonnegative values")
    learning_fractions = [
        float(value) for value in ml["learning_curve_fractions"]
    ]
    if (
        not learning_fractions
        or learning_fractions != sorted(set(learning_fractions))
        or learning_fractions[-1] != 1.0
        or any(not 0.0 < value <= 1.0 for value in learning_fractions)
    ):
        raise ValueError(
            "ml.learning_curve_fractions must be unique ascending values in "
            "(0, 1] ending at 1.0"
        )
    ambiguity = ml["mechanism_ambiguity_alarm"]
    for key in (
        "maximum_standardized_feature_distance",
        "minimum_cp_theta_ratio",
        "minimum_cp_beta_difference",
        "maximum_class_cp_theta_median_ape",
        "maximum_class_cp_beta_mae",
        "maximum_class_cp_probability_mae",
        "maximum_class_cp_median_bias",
    ):
        _positive_number(
            ambiguity[key],
            f"ml.mechanism_ambiguity_alarm.{key}",
        )
    if float(ambiguity["minimum_cp_theta_ratio"]) <= 1.0:
        raise ValueError(
            "ml.mechanism_ambiguity_alarm.minimum_cp_theta_ratio "
            "must exceed 1.0"
        )
    if int(ambiguity["minimum_test_buildings_per_class"]) < 2:
        raise ValueError(
            "ml.mechanism_ambiguity_alarm."
            "minimum_test_buildings_per_class must be at least 2"
        )


def ensure_runtime_directories(config: dict[str, Any]) -> None:
    for key in ("output_dir", "run_dir"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)
    Path(config["database_path"]).parent.mkdir(parents=True, exist_ok=True)
