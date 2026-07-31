"""Deterministic building catalogue generation and balanced sampling."""

from __future__ import annotations

import csv
import heapq
import itertools
import json
import math
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import KG_M2_TO_KN_M2, KSC_TO_KN_M2
from .db import connect, initialize, transaction, upsert_many
from .firefly import optimize_discrete_candidates
from .io_utils import stable_hash
from .structural import (
    aci_required_clear_spacing_m,
    beam_bar_positions,
    beam_reinforcement_fibers,
    mander_rectangular_confined_parameters,
    perimeter_bar_positions,
)

_BASE_DESIGN_CACHE: dict[str, dict[str, Any]] = {}
CATALOG_MODEL_SCHEMA_VERSION = (
    "rc3d-hinge-midpoint-hognestad-mander-hysteretic-"
    "firefly-multistart-adaptive-aci-spacing-"
    "capacity-shear-clear-span-multibay-equal-edge-gravity-"
    "strain-compatible-column-pm-research-rgt1-"
    "priestley-contraflexure-lp-mesh24-effective-modal-mass-"
    "column-joint-equilibrium-no-storyheight-proxy-v32"
)
SCWB_RESEARCH_CLASSES = (
    "research-low-margin",
    "research-medium-margin",
    "research-high-margin",
)


def _column_end_preselection_moment_knm(
    beam_count_at_joint: int,
    maximum_beam_design_moment_knm: float,
) -> float:
    """Return the preliminary moment demand assigned to one column end.

    The joint has a column end above and below it.  The exact SCWB screen later
    sums both strain-compatible column strengths, so preliminary member sizing
    assigns one half of the governing directional beam-joint sum to each
    column end.  Assigning the entire beam sum to one column and then summing
    two columns in the exact screen would double count the joint demand.
    """
    if beam_count_at_joint < 1:
        raise ValueError("beam_count_at_joint must be at least one")
    if maximum_beam_design_moment_knm < 0.0:
        raise ValueError(
            "maximum_beam_design_moment_knm must be non-negative"
        )
    return (
        0.5
        * float(beam_count_at_joint)
        * float(maximum_beam_design_moment_knm)
    )


def _capacity_constrained_equal_quotas(
    values: Iterable[Any],
    availability: dict[Any, int],
    total: int,
) -> dict[Any, int]:
    """Allocate an equal quota, saturating scarce groups deterministically."""
    ordered_values = list(values)
    if total < 0:
        raise ValueError("total quota must be non-negative")
    if sum(int(availability.get(value, 0)) for value in ordered_values) < total:
        raise ValueError("available rows cannot fill requested quota")
    quotas = {value: 0 for value in ordered_values}
    active = [
        value
        for value in ordered_values
        if int(availability.get(value, 0)) > 0
    ]
    slots = total
    while active and slots:
        equal_share, extra = divmod(slots, len(active))
        saturated = [
            value
            for value in active
            if int(availability[value]) <= equal_share
        ]
        if saturated:
            for value in saturated:
                quota = int(availability[value])
                quotas[value] = quota
                slots -= quota
                active.remove(value)
            continue
        for index, value in enumerate(active):
            quotas[value] = equal_share + int(index < extra)
        slots = 0
    if slots:
        raise ValueError("quota allocation ended with unfilled slots")
    return quotas


def _serialize_base_designs(designs: dict[str, Any]) -> str:
    """Serialize one expensive exact/FFA result for interruption-safe reuse."""
    payload = dict(designs)
    payload["beams"] = (
        None
        if designs["beams"] is None
        else [asdict(item) for item in designs["beams"]]
    )
    payload["columns"] = (
        None
        if designs["columns"] is None
        else [asdict(item) for item in designs["columns"]]
    )
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _deserialize_base_designs(payload_json: str) -> dict[str, Any]:
    payload = json.loads(payload_json)
    payload["beams"] = (
        None
        if payload["beams"] is None
        else [BeamDesign(**item) for item in payload["beams"]]
    )
    payload["columns"] = (
        None
        if payload["columns"] is None
        else [ColumnDesign(**item) for item in payload["columns"]]
    )
    return payload


@dataclass(frozen=True)
class BeamDesign:
    b_m: float
    h_m: float
    bars_per_face: int
    top_bar_count: int
    bottom_bar_count: int
    side_bar_count_each: int
    top_bar_diameter_m: float
    bottom_bar_diameter_m: float
    side_bar_diameter_m: float
    top_bar_layers: int
    bottom_bar_layers: int
    bar_diameter_m: float
    phi_mn_knm: float
    phi_mn_negative_knm: float
    phi_mn_positive_knm: float
    nominal_mn_negative_knm: float
    nominal_mn_positive_knm: float
    probable_mn_negative_knm: float
    probable_mn_positive_knm: float
    phi_vn_kn: float
    reinforcement_ratio: float
    stirrup_diameter_m: float
    stirrup_legs: int
    stirrup_spacing_m: float
    stirrup_fy_ksc: float
    design_moment_knm: float
    design_negative_moment_knm: float
    design_positive_moment_knm: float
    design_shear_kn: float
    capacity_design_shear_kn: float
    immediate_live_deflection_m: float
    total_long_term_deflection_m: float
    live_deflection_limit_m: float
    total_deflection_limit_m: float
    service_effective_inertia_m4: float
    service_cracked_inertia_m4: float
    objective_cost_per_m: float


@dataclass(frozen=True)
class BeamFaceOption:
    count: int
    diameter_m: float
    layers: int
    area_m2: float
    centroid_from_face_m: float
    effective_depth_m: float
    reinforcement_ratio: float


@dataclass(frozen=True)
class ColumnDesign:
    b_m: float
    h_m: float
    bar_count: int
    bar_diameter_m: float
    phi_pn_kn: float
    phi_mn_knm: float
    phi_vn_kn: float
    capacity_design_shear_kn: float
    reinforcement_ratio: float
    hoop_diameter_m: float
    hoop_spacing_m: float
    hoop_legs_x: int
    hoop_legs_y: int
    hoop_fy_ksc: float
    tier_strength_multipliers: tuple[float, ...]
    tier_axial_demands_kn: tuple[float, ...]
    tier_moment_demands_knm: tuple[float, ...]
    tier_phi_mn_at_axial_knm: tuple[float, ...]
    nominal_mn_at_base_axial_knm: float
    probable_mn_at_base_axial_knm: float
    objective_cost_per_m: float


@dataclass(frozen=True)
class ColumnGeometry:
    """Minimum geometry interface needed by the column-demand callback."""

    b_m: float
    h_m: float


def classify_scwb_ratio(
    ratio: float,
    *,
    minimum_ratio: float,
    medium_margin_ratio: float,
    high_margin_ratio: float,
) -> str | None:
    """Classify accepted research models by column/beam strength margin.

    This is deliberately not an ACI code-compliance classification.  The
    admission rule is only ``sum(Mnc) / sum(Mnb) > 1.0``.  The two higher
    cutoffs are sampling/audit bands that keep low-margin buildings visible
    in the ANN research database.
    """
    if ratio <= minimum_ratio:
        return None
    if ratio < medium_margin_ratio:
        return SCWB_RESEARCH_CLASSES[0]
    if ratio < high_margin_ratio:
        return SCWB_RESEARCH_CLASSES[1]
    return SCWB_RESEARCH_CLASSES[2]


def _round_up_to_increment(value: float, increment: float) -> float:
    return (
        math.ceil((value - 1.0e-12) / increment) * increment
    )


def _slab_one_way_shear_check(
    *,
    thickness_m: float,
    bay_width_m: float,
    fc_ksc: float,
    sdl_kg_m2: float,
    ll_kg_m2: float,
    concrete_density_kn_m3: float,
    dead_load_factor: float,
    live_load_factor: float,
    settings: dict[str, Any],
) -> dict[str, float | bool]:
    """ACI 318 size-effect one-way shear check for a conservative 1 m strip.

    Each PoC slab panel is supported by beams on all four edges, so gravity
    load transfers from slab to beams rather than directly from slab to
    columns. Punching shear is therefore not the relevant thickness
    criterion. For the square two-way panel, half the gravity load is
    assigned to each orthogonal spanning direction. The critical section is
    at ``d`` from the face of the minimum permitted beam width.
    """
    cover_m = float(settings["clear_cover_m"])
    bar_diameter_m = float(settings["assumed_bar_diameter_m"])
    effective_depth_m = thickness_m - cover_m - bar_diameter_m / 2.0
    if effective_depth_m <= 0.0:
        return {
            "valid": False,
            "effective_depth_m": effective_depth_m,
            "factored_load_kn_m2": math.inf,
            "demand_kn": math.inf,
            "phi_vc_kn": 0.0,
            "utilization": math.inf,
            "lambda_s": 0.0,
        }
    slab_dead_kn_m2 = (
        thickness_m * concrete_density_kn_m3
        + sdl_kg_m2 * KG_M2_TO_KN_M2
    )
    live_kn_m2 = ll_kg_m2 * KG_M2_TO_KN_M2
    factored_load_kn_m2 = (
        dead_load_factor * slab_dead_kn_m2
        + live_load_factor * live_kn_m2
    )
    strip_width_m = float(settings["strip_width_m"])
    support_width_m = float(settings["minimum_support_width_m"])
    shear_span_m = max(
        bay_width_m / 2.0
        - support_width_m / 2.0
        - effective_depth_m,
        0.0,
    )
    demand_kn = (
        factored_load_kn_m2
        * float(settings["two_way_directional_load_fraction"])
        * strip_width_m
        * shear_span_m
    )
    effective_depth_mm = effective_depth_m * 1000.0
    strip_width_mm = strip_width_m * 1000.0
    lambda_s = min(
        1.0,
        math.sqrt(2.0 / (1.0 + 0.004 * effective_depth_mm)),
    )
    fc_mpa = fc_ksc * KSC_TO_KN_M2 / 1000.0
    rho_w = float(
        settings["assumed_longitudinal_reinforcement_ratio"]
    )
    vc_stress_mpa = (
        0.66
        * lambda_s
        * float(settings["normalweight_lambda"])
        * rho_w ** (1.0 / 3.0)
        * math.sqrt(fc_mpa)
    )
    vc_kn = (
        vc_stress_mpa
        * strip_width_mm
        * effective_depth_mm
        / 1000.0
    )
    phi_vc_kn = float(settings["strength_reduction_factor"]) * vc_kn
    utilization = demand_kn / max(phi_vc_kn, 1.0e-12)
    return {
        "valid": bool(demand_kn <= phi_vc_kn + 1.0e-9),
        "effective_depth_m": effective_depth_m,
        "factored_load_kn_m2": factored_load_kn_m2,
        "demand_kn": demand_kn,
        "phi_vc_kn": phi_vc_kn,
        "utilization": utilization,
        "lambda_s": lambda_s,
    }


def slab_thickness_design(
    *,
    bay_width_m: float,
    fc_ksc: float,
    fy_ksc: float,
    sdl_kg_m2: float,
    ll_kg_m2: float,
    concrete_density_kn_m3: float,
    dead_load_factor: float,
    live_load_factor: float,
    slab_design: dict[str, Any],
) -> dict[str, float | str]:
    """Select max(ACI minimum thickness, one-way-shear thickness).

    ACI CODE-318-25 Chapter 8 is represented by the exterior two-way-panel
    limit without a drop panel, including the Grade-dependent multiplier.
    The full centreline bay is used as the clear span, which is conservative.
    Chapter 22 one-way shear is checked using the size-effect equation for a
    member without shear reinforcement.  Detailed slab flexural design is
    outside this PoC, so the locked minimum longitudinal ratio is used.
    """
    minimum = slab_design["minimum_thickness"]
    shear = slab_design["one_way_shear"]
    increment_m = float(slab_design["construction_increment_m"])
    maximum_m = float(
        slab_design["computational_maximum_thickness_m"]
    )
    fy_mpa = fy_ksc * KSC_TO_KN_M2 / 1000.0
    steel_modifier = (
        0.8
        + fy_mpa
        / float(
            minimum["steel_yield_modifier_denominator_mpa"]
        )
    )
    aci_raw_m = max(
        float(minimum["absolute_minimum_m"]),
        (
            bay_width_m
            * steel_modifier
            / float(minimum["span_divisor"])
        ),
    )
    aci_minimum_m = _round_up_to_increment(
        aci_raw_m, increment_m
    )

    starting_m = _round_up_to_increment(
        float(shear["clear_cover_m"])
        + float(shear["assumed_bar_diameter_m"]) / 2.0
        + increment_m,
        increment_m,
    )
    shear_required_m: float | None = None
    thickness_m = starting_m
    while thickness_m <= maximum_m + 1.0e-12:
        check = _slab_one_way_shear_check(
            thickness_m=thickness_m,
            bay_width_m=bay_width_m,
            fc_ksc=fc_ksc,
            sdl_kg_m2=sdl_kg_m2,
            ll_kg_m2=ll_kg_m2,
            concrete_density_kn_m3=concrete_density_kn_m3,
            dead_load_factor=dead_load_factor,
            live_load_factor=live_load_factor,
            settings=shear,
        )
        if bool(check["valid"]):
            shear_required_m = thickness_m
            break
        thickness_m = round(thickness_m + increment_m, 10)
    if shear_required_m is None:
        raise RuntimeError(
            "Slab shear sizing reached its computational maximum before "
            "finding a valid thickness; this is not an engineering limit"
        )
    selected_m = max(aci_minimum_m, shear_required_m)
    selected_check = _slab_one_way_shear_check(
        thickness_m=selected_m,
        bay_width_m=bay_width_m,
        fc_ksc=fc_ksc,
        sdl_kg_m2=sdl_kg_m2,
        ll_kg_m2=ll_kg_m2,
        concrete_density_kn_m3=concrete_density_kn_m3,
        dead_load_factor=dead_load_factor,
        live_load_factor=live_load_factor,
        settings=shear,
    )
    if not bool(selected_check["valid"]):
        raise RuntimeError(
            "Selected slab thickness does not pass its final shear check"
        )
    if aci_minimum_m > shear_required_m + 1.0e-12:
        controlling = "ACI minimum thickness"
    elif shear_required_m > aci_minimum_m + 1.0e-12:
        controlling = "one-way shear"
    else:
        controlling = "ACI minimum and one-way shear"
    return {
        "selected_thickness_m": selected_m,
        "aci_minimum_thickness_m": aci_minimum_m,
        "aci_unrounded_thickness_m": aci_raw_m,
        "shear_required_thickness_m": shear_required_m,
        "controlling_criterion": controlling,
        "effective_depth_m": float(
            selected_check["effective_depth_m"]
        ),
        "factored_load_kn_m2": float(
            selected_check["factored_load_kn_m2"]
        ),
        "shear_demand_kn_per_m_strip": float(
            selected_check["demand_kn"]
        ),
        "phi_vc_kn_per_m_strip": float(
            selected_check["phi_vc_kn"]
        ),
        "shear_utilization": float(
            selected_check["utilization"]
        ),
        "lambda_s": float(selected_check["lambda_s"]),
    }


def _bar_area(diameter_m: float) -> float:
    return math.pi * diameter_m**2 / 4.0


def _diameter_key(diameter_m: float) -> str:
    return str(int(round(float(diameter_m) * 1000.0)))


def _fc_key(fc_kn_m2: float) -> str:
    return str(int(round(float(fc_kn_m2) / KSC_TO_KN_M2)))


def _material_unit_price(
    unit_costs: dict[str, Any],
    *,
    material: str,
    diameter_m: float | None = None,
    fc_kn_m2: float | None = None,
) -> float:
    if material == "concrete":
        if fc_kn_m2 is None:
            raise ValueError("Concrete unit-price lookup requires fc")
        return float(
            unit_costs["concrete_per_m3_by_fc_ksc"][_fc_key(fc_kn_m2)]
        )
    if diameter_m is None:
        raise ValueError("Steel unit-price lookup requires bar diameter")
    if material == "longitudinal":
        table = unit_costs[
            "longitudinal_steel_per_kg_by_diameter_mm"
        ]
    elif material == "transverse":
        table = unit_costs[
            "transverse_steel_per_kg_by_diameter_mm"
        ]
    else:
        raise ValueError(f"Unknown material price class: {material}")
    return float(table[_diameter_key(diameter_m)])


def _doubly_reinforced_phi_mn(
    *,
    b_m: float,
    h_m: float,
    tension_area_m2: float,
    compression_area_m2: float,
    tension_depth_m: float,
    compression_depth_m: float,
    fc_kn_m2: float,
    fy_kn_m2: float,
    steel_elastic_modulus_mpa: float,
    apply_strength_reduction: bool = True,
) -> float:
    """Return strain-compatible flexural strength.

    The default result is design strength ``phi*Mn``.  With
    ``apply_strength_reduction=False`` the function returns nominal strength;
    this mode is used with expected steel strength by the capacity-design
    shear screen.
    """
    as_tension = float(tension_area_m2)
    as_compression = float(compression_area_m2)
    d_m = float(tension_depth_m)
    compression_depth_m = float(compression_depth_m)
    if (
        as_tension <= 0.0
        or as_compression <= 0.0
        or not 0.0 < compression_depth_m < d_m < h_m
    ):
        return 0.0
    fc_mpa = fc_kn_m2 / 1000.0
    beta_one = max(0.65, 0.85 - 0.05 * max(fc_mpa - 28.0, 0.0) / 7.0)
    steel_modulus_kn_m2 = steel_elastic_modulus_mpa * 1000.0
    concrete_ultimate_strain = 0.003

    def equilibrium(neutral_axis_m: float) -> float:
        tension_strain = (
            concrete_ultimate_strain
            * (d_m - neutral_axis_m)
            / neutral_axis_m
        )
        tension_stress = min(
            max(steel_modulus_kn_m2 * tension_strain, 0.0),
            fy_kn_m2,
        )
        compression_strain = (
            concrete_ultimate_strain
            * (neutral_axis_m - compression_depth_m)
            / neutral_axis_m
        )
        compression_stress = min(
            max(
                steel_modulus_kn_m2 * compression_strain,
                -fy_kn_m2,
            ),
            fy_kn_m2,
        )
        concrete_force = (
            0.85
            * fc_kn_m2
            * b_m
            * beta_one
            * neutral_axis_m
        )
        return (
            concrete_force
            + as_compression * compression_stress
            - as_tension * tension_stress
        )

    lower = max(1.0e-8, compression_depth_m * 0.05)
    upper = d_m * (1.0 - 1.0e-8)
    lower_force = equilibrium(lower)
    upper_force = equilibrium(upper)
    if lower_force * upper_force > 0.0:
        return 0.0
    # Thirty-two bisections resolve the neutral-axis depth to substantially
    # better than 1e-8 m for every permitted section.  Additional iterations
    # do not affect engineering capacities at the stored precision and only
    # multiply the cost of the exact discrete search.
    for _ in range(32):
        middle = 0.5 * (lower + upper)
        force = equilibrium(middle)
        if force == 0.0:
            lower = upper = middle
            break
        if lower_force * force <= 0.0:
            upper = middle
            upper_force = force
        else:
            lower = middle
            lower_force = force
    neutral_axis = 0.5 * (lower + upper)
    tension_strain = (
        concrete_ultimate_strain
        * (d_m - neutral_axis)
        / neutral_axis
    )
    compression_strain = (
        concrete_ultimate_strain
        * (neutral_axis - compression_depth_m)
        / neutral_axis
    )
    compression_stress = min(
        max(steel_modulus_kn_m2 * compression_strain, -fy_kn_m2),
        fy_kn_m2,
    )
    a_m = beta_one * neutral_axis
    concrete_force = 0.85 * fc_kn_m2 * b_m * a_m
    nominal_moment = (
        concrete_force * (d_m - a_m / 2.0)
        + as_compression
        * compression_stress
        * (d_m - compression_depth_m)
    )
    yield_strain = fy_kn_m2 / steel_modulus_kn_m2
    if tension_strain >= 0.005:
        phi = 0.90
    elif tension_strain <= yield_strain:
        phi = 0.65
    else:
        phi = 0.65 + 0.25 * (
            (tension_strain - yield_strain)
            / (0.005 - yield_strain)
        )
    reduction = phi if apply_strength_reduction else 1.0
    return max(reduction * nominal_moment, 0.0)


def column_pm_capacity_at_axial(
    *,
    width_m: float,
    depth_m: float,
    bar_count: int,
    bar_diameter_m: float,
    clear_cover_m: float,
    hoop_diameter_m: float,
    fc_kn_m2: float,
    fy_kn_m2: float,
    steel_elastic_modulus_mpa: float,
    target_axial_kn: float,
    strength_mode: str = "design",
    probable_steel_strength_factor: float = 1.25,
) -> dict[str, float | bool | str]:
    """Return strain-compatible tied-column flexural capacity at ``Pu``.

    The rectangular Whitney block and every perimeter reinforcing bar are
    included explicitly.  ``strength_mode='design'`` applies the tied-column
    ACI strength-reduction transition and the 0.80 maximum axial-strength
    limit. ``nominal`` uses ``fy`` with ``phi=1`` and ``probable`` uses
    ``1.25fy`` with ``phi=1`` for capacity-design equilibrium.
    """
    positive = {
        "width_m": width_m,
        "depth_m": depth_m,
        "bar_diameter_m": bar_diameter_m,
        "fc_kn_m2": fc_kn_m2,
        "fy_kn_m2": fy_kn_m2,
        "steel_elastic_modulus_mpa": steel_elastic_modulus_mpa,
    }
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in positive.values()
    ):
        raise ValueError("Column P-M section inputs must be positive and finite")
    if target_axial_kn < 0.0 or not math.isfinite(target_axial_kn):
        raise ValueError("Column target axial load must be finite and nonnegative")
    if strength_mode not in {"design", "nominal", "probable"}:
        raise ValueError("strength_mode must be design, nominal, or probable")
    if bar_count < 8 or bar_count % 4:
        raise ValueError("Column bar count must be a multiple of four and at least 8")

    bar_area_m2 = _bar_area(bar_diameter_m)
    steel_area_m2 = bar_count * bar_area_m2
    gross_area_m2 = width_m * depth_m
    if steel_area_m2 >= gross_area_m2:
        raise ValueError("Column steel area must be smaller than gross area")
    bar_half_width_m = (
        width_m / 2.0
        - clear_cover_m
        - hoop_diameter_m
        - bar_diameter_m / 2.0
    )
    bar_half_depth_m = (
        depth_m / 2.0
        - clear_cover_m
        - hoop_diameter_m
        - bar_diameter_m / 2.0
    )
    bar_positions = perimeter_bar_positions(
        bar_count,
        half_width_m=bar_half_width_m,
        half_depth_m=bar_half_depth_m,
    )
    del bar_half_width_m
    # Bars on the same layer have identical strain and stress.  Aggregate
    # those layers once so the exact strain-compatible solution does not
    # repeat identical calculations for every perimeter bar.
    bar_depth_counts = tuple(
        Counter(
            round(depth_m / 2.0 - z_coordinate, 12)
            for _, z_coordinate in bar_positions
        ).items()
    )

    expected_factor = (
        probable_steel_strength_factor
        if strength_mode == "probable"
        else 1.0
    )
    analysis_fy_kn_m2 = fy_kn_m2 * expected_factor
    elastic_modulus_kn_m2 = steel_elastic_modulus_mpa * 1000.0
    yield_strain = analysis_fy_kn_m2 / elastic_modulus_kn_m2
    concrete_ultimate_strain = 0.003
    fc_mpa = fc_kn_m2 / 1000.0
    beta_one = max(
        0.65,
        0.85 - 0.05 * max(fc_mpa - 28.0, 0.0) / 7.0,
    )
    nominal_concentric_kn = (
        0.85 * fc_kn_m2 * (gross_area_m2 - steel_area_m2)
        + analysis_fy_kn_m2 * steel_area_m2
    )
    maximum_axial_kn = (
        0.80 * 0.65 * nominal_concentric_kn
        if strength_mode == "design"
        else nominal_concentric_kn
    )
    if target_axial_kn > maximum_axial_kn + 1.0e-8:
        return {
            "valid": False,
            "strength_mode": strength_mode,
            "target_axial_kn": float(target_axial_kn),
            "maximum_axial_kn": float(maximum_axial_kn),
            "moment_capacity_knm": 0.0,
            "neutral_axis_m": math.nan,
            "phi": 0.0,
            "extreme_tension_strain": math.nan,
        }

    def response(neutral_axis_m: float) -> tuple[float, float, float, float]:
        a_m = min(beta_one * neutral_axis_m, depth_m)
        concrete_force_kn = 0.85 * fc_kn_m2 * width_m * a_m
        concrete_moment_knm = concrete_force_kn * (
            depth_m / 2.0 - a_m / 2.0
        )
        axial_kn = concrete_force_kn
        moment_knm = concrete_moment_knm
        extreme_tension_strain = 0.0
        for bar_depth_m, bars_at_depth in bar_depth_counts:
            strain = concrete_ultimate_strain * (
                neutral_axis_m - bar_depth_m
            ) / neutral_axis_m
            stress_kn_m2 = min(
                max(
                    elastic_modulus_kn_m2 * strain,
                    -analysis_fy_kn_m2,
                ),
                analysis_fy_kn_m2,
            )
            # The Whitney block already contains the gross concrete area.
            # Replace concrete by steel where a bar lies inside that block.
            if bar_depth_m <= a_m + 1.0e-12:
                stress_kn_m2 -= 0.85 * fc_kn_m2
            force_kn = (
                stress_kn_m2 * bar_area_m2 * int(bars_at_depth)
            )
            axial_kn += force_kn
            moment_knm += force_kn * (depth_m / 2.0 - bar_depth_m)
            extreme_tension_strain = max(
                extreme_tension_strain,
                -strain,
            )

        if strength_mode == "design":
            if extreme_tension_strain <= yield_strain:
                phi = 0.65
            elif extreme_tension_strain >= yield_strain + 0.003:
                phi = 0.90
            else:
                phi = 0.65 + 0.25 * (
                    (extreme_tension_strain - yield_strain) / 0.003
                )
            axial_kn = min(phi * axial_kn, maximum_axial_kn)
            moment_knm = phi * moment_knm
        else:
            phi = 1.0
        return (
            float(axial_kn),
            abs(float(moment_knm)),
            float(phi),
            float(extreme_tension_strain),
        )

    lower = max(depth_m * 1.0e-7, 1.0e-9)
    upper = depth_m * 1.0e3
    lower_response = response(lower)
    upper_response = response(upper)
    if (
        lower_response[0] > target_axial_kn + 1.0e-8
        or upper_response[0] < target_axial_kn - 1.0e-8
    ):
        return {
            "valid": False,
            "strength_mode": strength_mode,
            "target_axial_kn": float(target_axial_kn),
            "maximum_axial_kn": float(maximum_axial_kn),
            "moment_capacity_knm": 0.0,
            "neutral_axis_m": math.nan,
            "phi": 0.0,
            "extreme_tension_strain": math.nan,
        }
    # P(c) is monotonic over the compression-controlled branch used here.
    # Thirty-six bisections resolve c/h to better than 1.5e-8 over this
    # bracket, already far below the precision of material and section input.
    # More iterations only repeated floating-point work during catalogue
    # enumeration without changing an engineering decision.
    for _ in range(36):
        middle = 0.5 * (lower + upper)
        middle_response = response(middle)
        if middle_response[0] >= target_axial_kn:
            upper = middle
            upper_response = middle_response
        else:
            lower = middle
            lower_response = middle_response
    neutral_axis_m = 0.5 * (lower + upper)
    axial_kn, moment_knm, phi, tension_strain = response(neutral_axis_m)
    return {
        "valid": True,
        "strength_mode": strength_mode,
        "target_axial_kn": float(target_axial_kn),
        "equilibrium_axial_kn": float(axial_kn),
        "maximum_axial_kn": float(maximum_axial_kn),
        "moment_capacity_knm": max(float(moment_knm), 0.0),
        "neutral_axis_m": float(neutral_axis_m),
        "phi": float(phi),
        "extreme_tension_strain": float(tension_strain),
    }


def _beam_cost_per_m(
    *,
    b_m: float,
    h_m: float,
    top_bar_count: int,
    bottom_bar_count: int,
    side_bar_count_each: int,
    top_bar_diameter_m: float,
    bottom_bar_diameter_m: float,
    side_bar_diameter_m: float,
    stirrup_diameter_m: float,
    stirrup_legs: int,
    stirrup_spacing_m: float,
    clear_cover_m: float,
    fc_kn_m2: float,
    unit_costs: dict[str, Any],
) -> float:
    steel_density = float(unit_costs["steel_density_kg_m3"])
    top_weight = (
        top_bar_count
        * _bar_area(top_bar_diameter_m)
        * steel_density
    )
    bottom_weight = (
        bottom_bar_count
        * _bar_area(bottom_bar_diameter_m)
        * steel_density
    )
    side_weight = (
        2
        * side_bar_count_each
        * _bar_area(side_bar_diameter_m)
        * steel_density
    )
    core_width = max(b_m - 2.0 * clear_cover_m, 0.0)
    core_depth = max(h_m - 2.0 * clear_cover_m, 0.0)
    hook_length = (
        2.0
        * float(unit_costs["transverse_hook_length_bar_diameters"])
        * stirrup_diameter_m
    )
    hoop_centerline_length = (
        2.0 * (core_width + core_depth)
        + max(stirrup_legs - 2, 0) * core_depth
        + hook_length
    )
    transverse_weight = (
        hoop_centerline_length
        * _bar_area(stirrup_diameter_m)
        / stirrup_spacing_m
        * steel_density
    )
    return (
        b_m
        * h_m
        * _material_unit_price(
            unit_costs,
            material="concrete",
            fc_kn_m2=fc_kn_m2,
        )
        + top_weight
        * _material_unit_price(
            unit_costs,
            material="longitudinal",
            diameter_m=top_bar_diameter_m,
        )
        + bottom_weight
        * _material_unit_price(
            unit_costs,
            material="longitudinal",
            diameter_m=bottom_bar_diameter_m,
        )
        + side_weight
        * _material_unit_price(
            unit_costs,
            material="longitudinal",
            diameter_m=side_bar_diameter_m,
        )
        + transverse_weight
        * _material_unit_price(
            unit_costs,
            material="transverse",
            diameter_m=stirrup_diameter_m,
        )
        + (b_m + 2.0 * h_m) * float(unit_costs["formwork_per_m2"])
    )


def _column_cost_per_m(
    *,
    candidate: ColumnDesign | None,
    b_m: float,
    h_m: float,
    bar_count: int,
    bar_diameter_m: float,
    hoop_diameter_m: float,
    hoop_spacing_m: float,
    hoop_legs_x: int,
    hoop_legs_y: int,
    clear_cover_m: float,
    fc_kn_m2: float,
    unit_costs: dict[str, Any],
) -> float:
    del candidate
    steel_density = float(unit_costs["steel_density_kg_m3"])
    longitudinal_weight = (
        bar_count * _bar_area(bar_diameter_m) * steel_density
    )
    core_width = max(b_m - 2.0 * clear_cover_m, 0.0)
    core_depth = max(h_m - 2.0 * clear_cover_m, 0.0)
    hook_length = (
        2.0
        * float(unit_costs["transverse_hook_length_bar_diameters"])
        * hoop_diameter_m
    )
    hoop_length = (
        2.0 * (core_width + core_depth)
        + max(hoop_legs_x - 2, 0) * core_depth
        + max(hoop_legs_y - 2, 0) * core_width
        + hook_length
    )
    transverse_weight = (
        hoop_length
        * _bar_area(hoop_diameter_m)
        / hoop_spacing_m
        * steel_density
    )
    return (
        b_m
        * h_m
        * _material_unit_price(
            unit_costs,
            material="concrete",
            fc_kn_m2=fc_kn_m2,
        )
        + longitudinal_weight
        * _material_unit_price(
            unit_costs,
            material="longitudinal",
            diameter_m=bar_diameter_m,
        )
        + transverse_weight
        * _material_unit_price(
            unit_costs,
            material="transverse",
            diameter_m=hoop_diameter_m,
        )
        + 2.0
        * (b_m + h_m)
        * float(unit_costs["formwork_per_m2"])
    )


def _inclusive_grid(
    minimum: float,
    maximum: float,
    increment: float,
) -> list[float]:
    count = int(math.floor((maximum - minimum) / increment + 1.0e-9))
    return [round(minimum + index * increment, 10) for index in range(count + 1)]


def _beam_effective_inertia(
    *,
    b_m: float,
    h_m: float,
    d_m: float,
    as_tension_m2: float,
    fc_kn_m2: float,
    steel_elastic_modulus_mpa: float,
    service_moment_knm: float,
) -> tuple[float, float]:
    """Return Branson effective inertia and transformed cracked inertia."""
    fc_mpa = fc_kn_m2 / 1000.0
    concrete_modulus_kn_m2 = 4700.0 * math.sqrt(fc_mpa) * 1000.0
    steel_modulus_kn_m2 = steel_elastic_modulus_mpa * 1000.0
    modular_ratio = steel_modulus_kn_m2 / concrete_modulus_kn_m2
    gross_inertia_m4 = b_m * h_m**3 / 12.0
    rupture_modulus_kn_m2 = 0.62 * math.sqrt(fc_mpa) * 1000.0
    cracking_moment_knm = (
        rupture_modulus_kn_m2 * gross_inertia_m4 / (h_m / 2.0)
    )
    transformed_term = modular_ratio * as_tension_m2
    neutral_axis_m = (
        -transformed_term
        + math.sqrt(
            transformed_term**2
            + 2.0 * b_m * transformed_term * d_m
        )
    ) / b_m
    neutral_axis_m = min(max(neutral_axis_m, 0.0), d_m)
    cracked_inertia_m4 = (
        b_m * neutral_axis_m**3 / 3.0
        + transformed_term * (d_m - neutral_axis_m) ** 2
    )
    cracked_inertia_m4 = min(
        max(cracked_inertia_m4, 1.0e-12),
        gross_inertia_m4,
    )
    if service_moment_knm <= cracking_moment_knm:
        return gross_inertia_m4, cracked_inertia_m4
    cracking_ratio = min(cracking_moment_knm / service_moment_knm, 1.0)
    effective_inertia_m4 = (
        cracking_ratio**3 * gross_inertia_m4
        + (1.0 - cracking_ratio**3) * cracked_inertia_m4
    )
    return min(effective_inertia_m4, gross_inertia_m4), cracked_inertia_m4


def _beam_serviceability(
    *,
    b_m: float,
    h_m: float,
    d_m: float,
    as_tension_m2: float,
    bay_width_m: float,
    dead_line_load_kn_m: float,
    live_line_load_kn_m: float,
    fc_kn_m2: float,
    steel_elastic_modulus_mpa: float,
    serviceability: dict[str, Any],
) -> dict[str, float | bool]:
    """Check live and long-term total deflection using cracked-section stiffness."""
    concrete_modulus_kn_m2 = (
        4700.0 * math.sqrt(fc_kn_m2 / 1000.0) * 1000.0
    )

    def deflection(line_load_kn_m: float) -> tuple[float, float, float]:
        service_moment_knm = line_load_kn_m * bay_width_m**2 / 8.0
        effective, cracked = _beam_effective_inertia(
            b_m=b_m,
            h_m=h_m,
            d_m=d_m,
            as_tension_m2=as_tension_m2,
            fc_kn_m2=fc_kn_m2,
            steel_elastic_modulus_mpa=steel_elastic_modulus_mpa,
            service_moment_knm=service_moment_knm,
        )
        value = (
            5.0
            * line_load_kn_m
            * bay_width_m**4
            / (384.0 * concrete_modulus_kn_m2 * effective)
        )
        return value, effective, cracked

    total_line_load = dead_line_load_kn_m + live_line_load_kn_m
    immediate_total, effective_inertia, cracked_inertia = deflection(
        total_line_load
    )
    # Incremental live deflection is evaluated with the stiffness established
    # by the full service-load state, avoiding an unconservative gross-I value.
    immediate_live = (
        5.0
        * live_line_load_kn_m
        * bay_width_m**4
        / (384.0 * concrete_modulus_kn_m2 * effective_inertia)
    )
    sustained_live_fraction = float(
        serviceability["sustained_live_load_fraction"]
    )
    sustained_line_load = (
        dead_line_load_kn_m
        + sustained_live_fraction * live_line_load_kn_m
    )
    immediate_sustained, _, _ = deflection(sustained_line_load)
    long_term_multiplier = float(
        serviceability["long_term_deflection_multiplier"]
    )
    total_long_term = (
        immediate_total + long_term_multiplier * immediate_sustained
    )
    live_limit = bay_width_m / float(
        serviceability["immediate_live_deflection_limit_ratio"]
    )
    total_limit = bay_width_m / float(
        serviceability["total_long_term_deflection_limit_ratio"]
    )
    return {
        "immediate_live_deflection_m": immediate_live,
        "total_long_term_deflection_m": total_long_term,
        "live_deflection_limit_m": live_limit,
        "total_deflection_limit_m": total_limit,
        "service_effective_inertia_m4": effective_inertia,
        "service_cracked_inertia_m4": cracked_inertia,
        "valid": bool(
            immediate_live <= live_limit + 1.0e-12
            and total_long_term <= total_limit + 1.0e-12
        ),
    }


def _beam_candidates(
    fc_kn_m2: float,
    fy_kn_m2: float,
    factored_area_load_kn_m2: float,
    tributary_width_m: float,
    bay_width_m: float,
    concrete_density_kn_m3: float,
    dead_load_factor: float,
    material_detailing: dict[str, Any],
    service_dead_area_load_kn_m2: float,
    service_live_area_load_kn_m2: float,
    steel_elastic_modulus_mpa: float,
    section_search: dict[str, Any],
    serviceability: dict[str, Any],
    strength_multipliers: list[float],
    moment_coefficients: dict[str, Any],
    unit_costs: dict[str, Any],
    capacity_design_shear: dict[str, Any],
) -> list[BeamDesign]:
    """Enumerate constructible common-diameter one/two/three-layer beams.

    Top and bottom faces have independent bar counts but share one standard
    longitudinal diameter with the side bars.  Each face uses the minimum
    number of layers needed, up to three; bars per layer are limited only by
    the actual section geometry and ACI aggregate-aware clear spacing.  A
    third layer is permitted only at the configured minimum section depth.
    Every constructible face pair is checked, and only the exact cheapest
    layout for each target tier and section/stirrup family is retained.
    """
    clear_cover_m = float(material_detailing["clear_cover_m"])
    minimum_clear_spacing_m = float(
        material_detailing.get("beam_minimum_clear_bar_spacing_m", 0.025)
    )
    nominal_maximum_aggregate_size_m = float(
        material_detailing[
            "nominal_maximum_coarse_aggregate_size_m"
        ]
    )
    maximum_side_spacing_m = float(
        material_detailing.get("beam_maximum_side_bar_spacing_m", 0.30)
    )
    minimum_width_m = float(section_search["minimum_width_m"])
    minimum_depth_m = float(section_search["minimum_depth_m"])
    maximum_depth_to_width_ratio = float(
        section_search["maximum_depth_to_width_ratio"]
    )
    width_increment_m = float(section_search["width_increment_m"])
    depth_increment_m = float(section_search["depth_increment_m"])
    maximum_width_m = float(section_search["initial_search_width_m"])
    maximum_depth_m = float(section_search["initial_search_depth_m"])
    expansion_width_m = float(section_search["expansion_width_m"])
    expansion_depth_m = float(section_search["expansion_depth_m"])
    computational_guard = int(
        section_search["computational_guard_expansions"]
    )
    bar_diameters = tuple(
        float(value) for value in section_search["bar_diameters_m"]
    )
    maximum_layers = int(
        section_search["maximum_longitudinal_layers_per_face"]
    )
    minimum_bars_per_layer = int(
        section_search["minimum_bars_per_layer"]
    )
    three_layer_minimum_depth_m = float(
        section_search["three_layer_minimum_depth_m"]
    )
    stirrup_diameters = tuple(
        float(value) for value in section_search["stirrup_diameters_m"]
    )
    stirrup_spacings = tuple(
        float(value) for value in section_search["stirrup_spacings_m"]
    )
    tightest_stirrup_spacing_m = min(stirrup_spacings)
    widest_stirrup_spacing_m = max(stirrup_spacings)
    stirrup_leg_options = tuple(
        int(value) for value in section_search["stirrup_leg_options"]
    )
    transverse_fy_by_diameter = section_search[
        "transverse_fy_ksc_by_diameter_mm"
    ]
    bar_area_by_diameter = {
        diameter_m: _bar_area(diameter_m)
        for diameter_m in bar_diameters
    }
    required_clear_spacing_by_diameter = {
        diameter_m: aci_required_clear_spacing_m(
            bar_diameter_m=diameter_m,
            nominal_maximum_aggregate_size_m=(
                nominal_maximum_aggregate_size_m
            ),
            code_minimum_m=minimum_clear_spacing_m,
        )
        for diameter_m in bar_diameters
    }
    fc_mpa = fc_kn_m2 / 1000.0
    fy_mpa = fy_kn_m2 / 1000.0
    capacity_shear_enabled = bool(capacity_design_shear["enabled"])
    probable_strength_factor = float(
        capacity_design_shear["probable_steel_strength_factor"]
    )
    preselection_clear_span_ratio = float(
        capacity_design_shear["beam_preselection_clear_span_ratio"]
    )
    shear_phi = float(
        capacity_design_shear["strength_reduction_factor"]
    )
    rho_min = max(
        0.25 * math.sqrt(fc_mpa) / fy_mpa,
        1.40 / fy_mpa,
    )
    candidates: list[BeamDesign] = []
    evaluated_sections: set[tuple[float, float]] = set()
    certification_cost_ceiling = math.inf
    concrete_cost_per_m3 = _material_unit_price(
        unit_costs,
        material="concrete",
        fc_kn_m2=fc_kn_m2,
    )
    formwork_cost_per_m2 = float(unit_costs["formwork_per_m2"])

    for _ in range(computational_guard + 1):
        widths = _inclusive_grid(
            minimum_width_m,
            maximum_width_m,
            width_increment_m,
        )
        depths = _inclusive_grid(
            minimum_depth_m,
            maximum_depth_m,
            depth_increment_m,
        )
        for b_m, h_m in itertools.product(widths, depths):
            section_key = (b_m, h_m)
            if section_key in evaluated_sections:
                continue
            evaluated_sections.add(section_key)
            if (
                h_m <= b_m
                or h_m / b_m > maximum_depth_to_width_ratio + 1.0e-12
            ):
                continue
            # Exact adaptive-frontier pruning.  Concrete plus formwork is an
            # admissible lower bound because every real design must add
            # longitudinal and transverse steel.  Once all six tier optima
            # have a finite cost, a newly expanded section whose bare-section
            # lower bound already exceeds the most expensive current optimum
            # cannot improve any tier.  Marking it evaluated is safe because
            # the ceiling can only decrease in later expansions.
            section_cost_lower_bound = (
                b_m * h_m * concrete_cost_per_m3
                + (b_m + 2.0 * h_m) * formwork_cost_per_m2
            )
            if (
                section_cost_lower_bound
                > certification_cost_ceiling + 1.0e-12
            ):
                continue
            line_load = (
                factored_area_load_kn_m2 * tributary_width_m
                + dead_load_factor
                * b_m
                * h_m
                * concrete_density_kn_m3
            )
            design_negative_moment = (
                line_load
                * bay_width_m**2
                * float(moment_coefficients["negative_support_wl2"])
            )
            design_positive_moment = (
                line_load
                * bay_width_m**2
                * float(moment_coefficients["positive_midspan_wl2"])
            )
            design_shear = line_load * bay_width_m / 2.0
            service_dead_line_load = (
                service_dead_area_load_kn_m2 * tributary_width_m
                + b_m * h_m * concrete_density_kn_m3
            )
            service_live_line_load = (
                service_live_area_load_kn_m2 * tributary_width_m
            )
            # The face capacities and constructible layouts are identical for
            # alternative hoop-leg counts at a fixed stirrup diameter.  Keep
            # section-level caches so the exact checks are not repeated for
            # each leg option.
            section_capacity_cache: dict[
                tuple[BeamFaceOption, BeamFaceOption, bool, bool], float
            ] = {}
            section_layout_valid_cache: dict[
                tuple[
                    float,
                    BeamFaceOption,
                    BeamFaceOption,
                ],
                bool,
            ] = {}

            for stirrup_diameter_m, stirrup_legs in itertools.product(
                stirrup_diameters,
                stirrup_leg_options,
            ):
                option_transverse_fy_kn_m2 = (
                    float(
                        transverse_fy_by_diameter[
                            _diameter_key(stirrup_diameter_m)
                        ]
                    )
                    * KSC_TO_KN_M2
                )
                face_options: list[BeamFaceOption] = []
                for diameter_m in bar_diameters:
                    required_clear_spacing = (
                        required_clear_spacing_by_diameter[diameter_m]
                    )
                    geometric_maximum_bars_per_layer = int(
                        math.floor(
                            (
                                b_m
                                - 2.0 * clear_cover_m
                                - 2.0 * stirrup_diameter_m
                                + required_clear_spacing
                            )
                            / (
                                diameter_m
                                + required_clear_spacing
                            )
                            + 1.0e-12
                        )
                    )
                    if (
                        geometric_maximum_bars_per_layer
                        < minimum_bars_per_layer
                    ):
                        continue
                    outer_centroid_m = (
                        clear_cover_m
                        + stirrup_diameter_m
                        + diameter_m / 2.0
                    )
                    vertical_center_spacing_m = (
                        diameter_m + required_clear_spacing
                    )
                    maximum_count = (
                        maximum_layers
                        * geometric_maximum_bars_per_layer
                    )
                    for count in range(
                        minimum_bars_per_layer,
                        maximum_count + 1,
                    ):
                        layers = int(
                            math.ceil(
                                count
                                / geometric_maximum_bars_per_layer
                            )
                        )
                        if (
                            layers < 1
                            or layers > maximum_layers
                            or (
                                layers == 3
                                and h_m + 1.0e-12
                                < three_layer_minimum_depth_m
                            )
                        ):
                            continue
                        base_count, remainder = divmod(count, layers)
                        layer_counts = tuple(
                            base_count + int(index < remainder)
                            for index in range(layers)
                        )
                        if min(layer_counts) < minimum_bars_per_layer:
                            continue
                        if (
                            max(layer_counts)
                            > geometric_maximum_bars_per_layer
                        ):
                            continue
                        area_m2 = (
                            count * bar_area_by_diameter[diameter_m]
                        )
                        centroid_from_face_m = (
                            sum(
                                layer_count
                                * (
                                    outer_centroid_m
                                    + layer_index
                                    * vertical_center_spacing_m
                                )
                                for layer_index, layer_count in enumerate(
                                    layer_counts
                                )
                            )
                            / count
                        )
                        effective_depth_m = (
                            h_m - centroid_from_face_m
                        )
                        if not (
                            0.0
                            < centroid_from_face_m
                            < effective_depth_m
                            < h_m
                        ):
                            continue
                        rho = area_m2 / (b_m * effective_depth_m)
                        if rho_min <= rho <= 0.025:
                            face_options.append(
                                BeamFaceOption(
                                    count=count,
                                    diameter_m=diameter_m,
                                    layers=layers,
                                    area_m2=area_m2,
                                    centroid_from_face_m=(
                                        centroid_from_face_m
                                    ),
                                    effective_depth_m=effective_depth_m,
                                    reinforcement_ratio=rho,
                                )
                            )
                if not face_options:
                    continue
                # There is exactly one face option for each
                # (diameter, count) pair.  Under the locked common-diameter
                # rule, steel cost and area both increase strictly with
                # count, so no same-diameter option can safely dominate
                # another.  Retaining all options is exact and avoids an
                # unnecessary quadratic dominance scan.
                # A yielded tension face cannot contribute more than
                # phi*As*fy*d, even with compression reinforcement, because
                # its lever arm cannot exceed d.  Filter face options that
                # cannot reach even the lowest requested tier before forming
                # the Cartesian product.  This is an admissible capacity
                # bound, so it accelerates exact enumeration without removing
                # any feasible optimum.
                minimum_multiplier = min(strength_multipliers)
                top_options = [
                    option
                    for option in face_options
                    if (
                        0.90
                        * option.area_m2
                        * fy_kn_m2
                        * option.effective_depth_m
                        + 1.0e-9
                        >= minimum_multiplier
                        * design_negative_moment
                    )
                ]
                bottom_capacity_options = [
                    option
                    for option in face_options
                    if (
                        0.90
                        * option.area_m2
                        * fy_kn_m2
                        * option.effective_depth_m
                        + 1.0e-9
                        >= minimum_multiplier
                        * design_positive_moment
                    )
                ]
                if not top_options or not bottom_capacity_options:
                    continue
                service_cache: dict[
                    BeamFaceOption, dict[str, float | bool]
                ] = {
                    option: _beam_serviceability(
                        b_m=b_m,
                        h_m=h_m,
                        d_m=option.effective_depth_m,
                        as_tension_m2=option.area_m2,
                        bay_width_m=bay_width_m,
                        dead_line_load_kn_m=service_dead_line_load,
                        live_line_load_kn_m=service_live_line_load,
                        fc_kn_m2=fc_kn_m2,
                        steel_elastic_modulus_mpa=(
                            steel_elastic_modulus_mpa
                        ),
                        serviceability=serviceability,
                    )
                    for option in bottom_capacity_options
                }
                bottom_options = [
                    option
                    for option in bottom_capacity_options
                    if bool(service_cache[option]["valid"])
                ]
                if not bottom_options:
                    continue
                av_m2 = stirrup_legs * _bar_area(
                    stirrup_diameter_m
                )
                minimum_av_over_s = max(
                    62.0
                    * math.sqrt(fc_mpa)
                    * b_m
                    / option_transverse_fy_kn_m2,
                    350.0
                    * b_m
                    / option_transverse_fy_kn_m2,
                )

                def maximum_phi_vn_for_depth(
                    effective_depth_m: float,
                ) -> float:
                    if tightest_stirrup_spacing_m > min(
                        0.50 * effective_depth_m,
                        0.30,
                    ) + 1.0e-12:
                        return 0.0
                    if (
                        av_m2 / tightest_stirrup_spacing_m + 1.0e-15
                        < minimum_av_over_s
                    ):
                        return 0.0
                    vc_upper = (
                        170.0
                        * math.sqrt(fc_mpa)
                        * b_m
                        * effective_depth_m
                    )
                    vs_upper = (
                        av_m2
                        * option_transverse_fy_kn_m2
                        * effective_depth_m
                        / tightest_stirrup_spacing_m
                    )
                    code_maximum_vn = (
                        660.0
                        * math.sqrt(fc_mpa)
                        * b_m
                        * effective_depth_m
                    )
                    return shear_phi * min(
                        vc_upper + vs_upper,
                        code_maximum_vn,
                    )
                # Admissible section-capacity upper bounds.  Nominal flexural
                # resistance cannot exceed the yielded tension force times
                # its full effective depth; the 0.90 factor is the largest
                # permitted flexural strength-reduction factor.  The shear
                # bound uses the strongest actually permitted spacing for the
                # current stirrup diameter and leg count.  These bounds
                # identify tiers that this family could possibly satisfy,
                # avoiding exact flexural solves for impossible shear tiers.
                top_options_by_diameter = {
                    diameter_m: sorted(
                        (
                            option
                            for option in top_options
                            if math.isclose(
                                option.diameter_m,
                                diameter_m,
                                rel_tol=0.0,
                                abs_tol=1.0e-12,
                            )
                        ),
                        key=lambda option: option.count,
                    )
                    for diameter_m in bar_diameters
                }
                bottom_options_by_diameter = {
                    diameter_m: sorted(
                        (
                            option
                            for option in bottom_options
                            if math.isclose(
                                option.diameter_m,
                                diameter_m,
                                rel_tol=0.0,
                                abs_tol=1.0e-12,
                            )
                        ),
                        key=lambda option: option.count,
                    )
                    for diameter_m in bar_diameters
                }
                diameter_upper_bounds: list[
                    tuple[float, float, float]
                ] = []
                for diameter_m in bar_diameters:
                    top_group = top_options_by_diameter[diameter_m]
                    bottom_group = bottom_options_by_diameter[
                        diameter_m
                    ]
                    if not top_group or not bottom_group:
                        continue
                    negative_upper = max(
                        0.90
                        * option.area_m2
                        * fy_kn_m2
                        * option.effective_depth_m
                        for option in top_group
                    )
                    positive_upper = max(
                        0.90
                        * option.area_m2
                        * fy_kn_m2
                        * option.effective_depth_m
                        for option in bottom_group
                    )
                    shear_depth_upper = min(
                        max(
                            option.effective_depth_m
                            for option in top_group
                        ),
                        max(
                            option.effective_depth_m
                            for option in bottom_group
                        ),
                    )
                    shear_upper = maximum_phi_vn_for_depth(
                        shear_depth_upper
                    )
                    diameter_upper_bounds.append(
                        (negative_upper, positive_upper, shear_upper)
                    )
                potential_tiers = {
                    tier_index
                    for tier_index, multiplier in enumerate(
                        strength_multipliers
                    )
                    if any(
                        negative_upper + 1.0e-9
                        >= multiplier * design_negative_moment
                        and positive_upper + 1.0e-9
                        >= multiplier * design_positive_moment
                        and shear_upper + 1.0e-9
                        >= multiplier * design_shear
                        for (
                            negative_upper,
                            positive_upper,
                            shear_upper,
                        ) in diameter_upper_bounds
                    )
                }
                if not potential_tiers:
                    continue

                def face_pair_capacity(
                    tension: BeamFaceOption,
                    compression: BeamFaceOption,
                    *,
                    probable: bool = False,
                    nominal: bool = False,
                ) -> float:
                    if probable and nominal:
                        raise ValueError(
                            "Beam capacity cannot be both nominal and probable"
                        )
                    key = (tension, compression, probable, nominal)
                    if key not in section_capacity_cache:
                        section_capacity_cache[key] = (
                            _doubly_reinforced_phi_mn(
                                b_m=b_m,
                                h_m=h_m,
                                tension_area_m2=tension.area_m2,
                                compression_area_m2=(
                                    compression.area_m2
                                ),
                                tension_depth_m=(
                                    tension.effective_depth_m
                                ),
                                compression_depth_m=(
                                    compression.centroid_from_face_m
                                ),
                                fc_kn_m2=fc_kn_m2,
                                fy_kn_m2=(
                                    fy_kn_m2 * probable_strength_factor
                                    if probable
                                    else fy_kn_m2
                                ),
                                steel_elastic_modulus_mpa=(
                                    steel_elastic_modulus_mpa
                                ),
                                apply_strength_reduction=not (
                                    probable or nominal
                                ),
                            )
                        )
                    return section_capacity_cache[key]

                best_for_tier: dict[int, BeamDesign] = {}
                side_bar_count_by_diameter: dict[float, int] = {}
                for diameter_m in bar_diameters:
                    half_depth = (
                        h_m / 2.0
                        - clear_cover_m
                        - stirrup_diameter_m
                        - diameter_m / 2.0
                    )
                    if half_depth <= 0.0:
                        side_bar_count_by_diameter[diameter_m] = -1
                    else:
                        side_bar_count_by_diameter[diameter_m] = max(
                            0,
                            math.ceil(
                                2.0
                                * half_depth
                                / maximum_side_spacing_m
                                - 1.0e-12
                            )
                            - 1,
                        )

                # Enumerate common-diameter top/bottom pairs lazily in exact
                # nondecreasing lower-bound cost order.  Materializing and
                # sorting the full Cartesian product became prohibitive once
                # the arbitrary eight-bars-per-layer cap was removed.  For a
                # fixed diameter and stirrup family, the lower-bound cost is
                # a constant plus a positive coefficient times
                # ``top.count + bottom.count``.  Each top-option row is thus
                # sorted by bottom count; a heap merge visits exactly the
                # same pairs in cost order and permits the existing exact
                # stopping certificate without allocating O(N_top*N_bottom).
                pair_groups: dict[
                    float,
                    tuple[list[BeamFaceOption], list[BeamFaceOption]],
                ] = {}
                pair_cost_terms: dict[float, tuple[float, float]] = {}
                longitudinal_density = float(
                    unit_costs["steel_density_kg_m3"]
                )
                for diameter_m in bar_diameters:
                    top_group = top_options_by_diameter[diameter_m]
                    bottom_group = bottom_options_by_diameter[
                        diameter_m
                    ]
                    count_each = side_bar_count_by_diameter[diameter_m]
                    if not top_group or not bottom_group or count_each < 0:
                        continue
                    fixed_cost = _beam_cost_per_m(
                        b_m=b_m,
                        h_m=h_m,
                        top_bar_count=0,
                        bottom_bar_count=0,
                        side_bar_count_each=count_each,
                        top_bar_diameter_m=diameter_m,
                        bottom_bar_diameter_m=diameter_m,
                        side_bar_diameter_m=diameter_m,
                        stirrup_diameter_m=stirrup_diameter_m,
                        stirrup_legs=stirrup_legs,
                        stirrup_spacing_m=widest_stirrup_spacing_m,
                        clear_cover_m=clear_cover_m,
                        fc_kn_m2=fc_kn_m2,
                        unit_costs=unit_costs,
                    )
                    longitudinal_cost_per_bar = (
                        bar_area_by_diameter[diameter_m]
                        * longitudinal_density
                        * _material_unit_price(
                            unit_costs,
                            material="longitudinal",
                            diameter_m=diameter_m,
                        )
                    )
                    pair_groups[diameter_m] = (
                        top_group,
                        bottom_group,
                    )
                    pair_cost_terms[diameter_m] = (
                        fixed_cost,
                        longitudinal_cost_per_bar,
                    )

                pair_heap: list[tuple[float, float, int, int]] = []
                for diameter_m, (
                    top_group,
                    bottom_group,
                ) in pair_groups.items():
                    fixed_cost, cost_per_bar = pair_cost_terms[
                        diameter_m
                    ]
                    first_bottom = bottom_group[0]
                    for top_index, top_option in enumerate(top_group):
                        heapq.heappush(
                            pair_heap,
                            (
                                fixed_cost
                                + cost_per_bar
                                * (
                                    top_option.count
                                    + first_bottom.count
                                ),
                                diameter_m,
                                top_index,
                                0,
                            ),
                        )

                while pair_heap:
                    (
                        pair_cost_lower_bound,
                        common_bar_diameter_m,
                        top_index,
                        bottom_index,
                    ) = heapq.heappop(pair_heap)
                    top_group, bottom_group = pair_groups[
                        common_bar_diameter_m
                    ]
                    top = top_group[top_index]
                    bottom = bottom_group[bottom_index]
                    next_bottom_index = bottom_index + 1
                    if next_bottom_index < len(bottom_group):
                        fixed_cost, cost_per_bar = pair_cost_terms[
                            common_bar_diameter_m
                        ]
                        heapq.heappush(
                            pair_heap,
                            (
                                fixed_cost
                                + cost_per_bar
                                * (
                                    top.count
                                    + bottom_group[
                                        next_bottom_index
                                    ].count
                                ),
                                common_bar_diameter_m,
                                top_index,
                                next_bottom_index,
                            ),
                        )
                    common_bar_diameter_m = top.diameter_m
                    side_bar_count_each = (
                        side_bar_count_by_diameter[
                            common_bar_diameter_m
                        ]
                    )
                    if side_bar_count_each < 0:
                        continue
                    if (
                        potential_tiers.issubset(best_for_tier)
                        and pair_cost_lower_bound
                        >= max(
                            best_for_tier[index].objective_cost_per_m
                            for index in potential_tiers
                        )
                        - 1.0e-12
                    ):
                        break
                    pair_negative_upper = (
                        0.90
                        * top.area_m2
                        * fy_kn_m2
                        * top.effective_depth_m
                    )
                    pair_positive_upper = (
                        0.90
                        * bottom.area_m2
                        * fy_kn_m2
                        * bottom.effective_depth_m
                    )
                    pair_shear_depth = min(
                        top.effective_depth_m,
                        bottom.effective_depth_m,
                    )
                    pair_shear_upper = maximum_phi_vn_for_depth(
                        pair_shear_depth
                    )
                    improvable_tiers = {
                        tier_index
                        for tier_index, multiplier in enumerate(
                            strength_multipliers
                        )
                        if (
                            tier_index in potential_tiers
                            and (
                                tier_index not in best_for_tier
                                or pair_cost_lower_bound
                                < best_for_tier[
                                    tier_index
                                ].objective_cost_per_m
                                - 1.0e-12
                            )
                            and pair_negative_upper + 1.0e-9
                            >= multiplier * design_negative_moment
                            and pair_positive_upper + 1.0e-9
                            >= multiplier * design_positive_moment
                            and pair_shear_upper + 1.0e-9
                            >= multiplier * design_shear
                        )
                    }
                    if not improvable_tiers:
                        continue
                    deflection = service_cache[bottom]
                    negative_capacity = face_pair_capacity(top, bottom)
                    positive_capacity = face_pair_capacity(bottom, top)
                    negative_nominal_capacity = face_pair_capacity(
                        top,
                        bottom,
                        nominal=True,
                    )
                    positive_nominal_capacity = face_pair_capacity(
                        bottom,
                        top,
                        nominal=True,
                    )
                    if capacity_shear_enabled:
                        negative_probable_capacity = face_pair_capacity(
                            top,
                            bottom,
                            probable=True,
                        )
                        positive_probable_capacity = face_pair_capacity(
                            bottom,
                            top,
                            probable=True,
                        )
                        capacity_design_shear_kn = (
                            design_shear
                            + (
                                negative_probable_capacity
                                + positive_probable_capacity
                            )
                            / (
                                preselection_clear_span_ratio
                                * bay_width_m
                            )
                        )
                    else:
                        negative_probable_capacity = negative_capacity
                        positive_probable_capacity = positive_capacity
                        capacity_design_shear_kn = design_shear
                    shear_depth_m = pair_shear_depth
                    vc = (
                        170.0
                        * math.sqrt(fc_mpa)
                        * b_m
                        * shear_depth_m
                    )
                    maximum_vn = (
                        660.0
                        * math.sqrt(fc_mpa)
                        * b_m
                        * shear_depth_m
                    )
                    for tier_index, multiplier in enumerate(
                        strength_multipliers
                    ):
                        if tier_index not in improvable_tiers:
                            continue
                        current = best_for_tier.get(tier_index)
                        if (
                            current is not None
                            and pair_cost_lower_bound
                            >= current.objective_cost_per_m - 1.0e-12
                        ):
                            continue
                        if (
                            negative_capacity + 1.0e-9
                            < multiplier * design_negative_moment
                            or positive_capacity + 1.0e-9
                            < multiplier * design_positive_moment
                        ):
                            continue
                        valid_stirrups: list[
                            tuple[float, float]
                        ] = []
                        for spacing_m in stirrup_spacings:
                            if spacing_m > min(
                                0.50 * shear_depth_m,
                                0.30,
                            ) + 1.0e-12:
                                continue
                            av_over_s = av_m2 / spacing_m
                            if (
                                av_over_s + 1.0e-15
                                < minimum_av_over_s
                            ):
                                continue
                            vs = (
                                av_m2
                                * option_transverse_fy_kn_m2
                                * shear_depth_m
                                / spacing_m
                            )
                            phi_vn = shear_phi * min(
                                vc + vs,
                                maximum_vn,
                            )
                            required_shear = max(
                                multiplier * design_shear,
                                capacity_design_shear_kn,
                            )
                            if phi_vn + 1.0e-9 >= required_shear:
                                valid_stirrups.append(
                                    (spacing_m, phi_vn)
                                )
                        if not valid_stirrups:
                            continue
                        stirrup_spacing_m, phi_vn = max(
                            valid_stirrups,
                            key=lambda item: item[0],
                        )
                        pair_key = (
                            stirrup_diameter_m,
                            top,
                            bottom,
                        )
                        if pair_key not in section_layout_valid_cache:
                            try:
                                beam_reinforcement_fibers(
                                    width_m=b_m,
                                    depth_m=h_m,
                                    clear_cover_m=clear_cover_m,
                                    stirrup_diameter_m=(
                                        stirrup_diameter_m
                                    ),
                                    top_bar_count=top.count,
                                    top_bar_diameter_m=(
                                        top.diameter_m
                                    ),
                                    top_bar_layers=top.layers,
                                    bottom_bar_count=bottom.count,
                                    bottom_bar_diameter_m=(
                                        bottom.diameter_m
                                    ),
                                    bottom_bar_layers=bottom.layers,
                                    side_bar_count_each=(
                                        side_bar_count_each
                                    ),
                                    side_bar_diameter_m=(
                                        common_bar_diameter_m
                                    ),
                                    minimum_clear_spacing_m=(
                                        minimum_clear_spacing_m
                                    ),
                                    nominal_maximum_aggregate_size_m=(
                                        nominal_maximum_aggregate_size_m
                                    ),
                                    minimum_bars_per_layer=(
                                        minimum_bars_per_layer
                                    ),
                                    three_layer_minimum_depth_m=(
                                        three_layer_minimum_depth_m
                                    ),
                                )
                            except ValueError:
                                section_layout_valid_cache[pair_key] = (
                                    False
                                )
                            else:
                                section_layout_valid_cache[pair_key] = (
                                    True
                                )
                        if not section_layout_valid_cache[pair_key]:
                            continue
                        cost = _beam_cost_per_m(
                            b_m=b_m,
                            h_m=h_m,
                            top_bar_count=top.count,
                            bottom_bar_count=bottom.count,
                            side_bar_count_each=side_bar_count_each,
                            top_bar_diameter_m=common_bar_diameter_m,
                            bottom_bar_diameter_m=common_bar_diameter_m,
                            side_bar_diameter_m=common_bar_diameter_m,
                            stirrup_diameter_m=stirrup_diameter_m,
                            stirrup_legs=stirrup_legs,
                            stirrup_spacing_m=stirrup_spacing_m,
                            clear_cover_m=clear_cover_m,
                            fc_kn_m2=fc_kn_m2,
                            unit_costs=unit_costs,
                        )
                        candidate = BeamDesign(
                            b_m=b_m,
                            h_m=h_m,
                            bars_per_face=max(top.count, bottom.count),
                            top_bar_count=top.count,
                            bottom_bar_count=bottom.count,
                            side_bar_count_each=side_bar_count_each,
                            top_bar_diameter_m=top.diameter_m,
                            bottom_bar_diameter_m=bottom.diameter_m,
                            side_bar_diameter_m=common_bar_diameter_m,
                            top_bar_layers=top.layers,
                            bottom_bar_layers=bottom.layers,
                            bar_diameter_m=common_bar_diameter_m,
                            phi_mn_knm=max(
                                negative_capacity,
                                positive_capacity,
                            ),
                            phi_mn_negative_knm=negative_capacity,
                            phi_mn_positive_knm=positive_capacity,
                            nominal_mn_negative_knm=(
                                negative_nominal_capacity
                            ),
                            nominal_mn_positive_knm=(
                                positive_nominal_capacity
                            ),
                            probable_mn_negative_knm=(
                                negative_probable_capacity
                            ),
                            probable_mn_positive_knm=(
                                positive_probable_capacity
                            ),
                            phi_vn_kn=phi_vn,
                            reinforcement_ratio=max(
                                top.reinforcement_ratio,
                                bottom.reinforcement_ratio,
                            ),
                            stirrup_diameter_m=stirrup_diameter_m,
                            stirrup_legs=stirrup_legs,
                            stirrup_spacing_m=stirrup_spacing_m,
                            stirrup_fy_ksc=(
                                option_transverse_fy_kn_m2
                                / KSC_TO_KN_M2
                            ),
                            design_moment_knm=max(
                                design_negative_moment,
                                design_positive_moment,
                            ),
                            design_negative_moment_knm=(
                                design_negative_moment
                            ),
                            design_positive_moment_knm=(
                                design_positive_moment
                            ),
                            design_shear_kn=design_shear,
                            capacity_design_shear_kn=(
                                capacity_design_shear_kn
                            ),
                            immediate_live_deflection_m=float(
                                deflection[
                                    "immediate_live_deflection_m"
                                ]
                            ),
                            total_long_term_deflection_m=float(
                                deflection[
                                    "total_long_term_deflection_m"
                                ]
                            ),
                            live_deflection_limit_m=float(
                                deflection["live_deflection_limit_m"]
                            ),
                            total_deflection_limit_m=float(
                                deflection["total_deflection_limit_m"]
                            ),
                            service_effective_inertia_m4=float(
                                deflection[
                                    "service_effective_inertia_m4"
                                ]
                            ),
                            service_cracked_inertia_m4=float(
                                deflection[
                                    "service_cracked_inertia_m4"
                                ]
                            ),
                            objective_cost_per_m=cost,
                        )
                        if (
                            current is None
                            or (
                                candidate.objective_cost_per_m,
                                candidate.top_bar_layers
                                + candidate.bottom_bar_layers,
                                candidate.top_bar_count
                                + candidate.bottom_bar_count,
                            )
                            < (
                                current.objective_cost_per_m,
                                current.top_bar_layers
                                + current.bottom_bar_layers,
                                current.top_bar_count
                                + current.bottom_bar_count,
                            )
                        ):
                            best_for_tier[tier_index] = candidate
                candidates.extend(best_for_tier.values())

        candidates = list(
            {
                tuple(asdict(candidate).values()): candidate
                for candidate in candidates
            }.values()
        )
        candidates = sorted(
            candidates,
            key=lambda item: (
                item.objective_cost_per_m,
                item.h_m,
                item.b_m,
                item.reinforcement_ratio,
            ),
        )
        selected = _select_beam_designs_exact(
            candidates,
            strength_multipliers,
        )
        if selected:
            maximum_selected_cost = max(
                item.objective_cost_per_m for item in selected
            )
            certification_cost_ceiling = min(
                certification_cost_ceiling,
                maximum_selected_cost,
            )
            next_width = maximum_width_m + width_increment_m
            minimum_depth_for_next_width = next(
                depth
                for depth in _inclusive_grid(
                    minimum_depth_m,
                    max(
                        maximum_depth_m + depth_increment_m,
                        next_width + depth_increment_m,
                    ),
                    depth_increment_m,
                )
                if depth > next_width + 1.0e-12
            )
            next_depth = maximum_depth_m + depth_increment_m
            minimum_width_for_next_depth = next(
                width
                for width in _inclusive_grid(
                    minimum_width_m,
                    max(
                        maximum_width_m + width_increment_m,
                        next_depth,
                    ),
                    width_increment_m,
                )
                if (
                    width + 1.0e-12
                    >= next_depth / maximum_depth_to_width_ratio
                    and width < next_depth - 1.0e-12
                )
            )
            minimum_unseen_cost = min(
                next_width
                * minimum_depth_for_next_width
                * _material_unit_price(
                    unit_costs,
                    material="concrete",
                    fc_kn_m2=fc_kn_m2,
                )
                + (
                    next_width + 2.0 * minimum_depth_for_next_width
                )
                * float(unit_costs["formwork_per_m2"]),
                minimum_width_for_next_depth
                * next_depth
                * _material_unit_price(
                    unit_costs,
                    material="concrete",
                    fc_kn_m2=fc_kn_m2,
                )
                + (minimum_width_for_next_depth + 2.0 * next_depth)
                * float(unit_costs["formwork_per_m2"]),
            )
            if (
                minimum_unseen_cost
                > maximum_selected_cost + 1.0e-12
            ):
                return candidates
        maximum_width_m += expansion_width_m
        maximum_depth_m += expansion_depth_m
    raise RuntimeError(
        "Adaptive common-diameter beam search reached its computational "
        "guard before proving the six exact tier optima. This is an "
        "algorithmic guard, not a maximum permitted beam section."
    )


def _beam_is_eligible(
    candidate: BeamDesign,
    multiplier: float,
) -> bool:
    return bool(
        candidate.phi_mn_negative_knm
        >= multiplier * candidate.design_negative_moment_knm
        and candidate.phi_mn_positive_knm
        >= multiplier * candidate.design_positive_moment_knm
        and candidate.phi_vn_kn
        >= max(
            multiplier * candidate.design_shear_kn,
            candidate.capacity_design_shear_kn,
        )
    )


def _select_beam_designs_exact(
    candidates: Iterable[BeamDesign],
    multipliers: list[float],
) -> list[BeamDesign] | None:
    """Cheap exact selector used to prove the adaptive search frontier."""
    selected: list[BeamDesign] = []
    materialized = list(candidates)
    for multiplier in multipliers:
        eligible = [
            candidate
            for candidate in materialized
            if _beam_is_eligible(candidate, multiplier)
        ]
        if not eligible:
            return None
        chosen = min(
            eligible,
            key=lambda item: (
                item.objective_cost_per_m,
                item.h_m,
                item.b_m,
            ),
        )
        selected.append(chosen)
    return selected


def _select_beam_designs(
    candidates: Iterable[BeamDesign],
    multipliers: list[float],
    *,
    firefly_settings: dict[str, Any],
    seed_prefix: dict[str, Any],
) -> tuple[list[BeamDesign] | None, list[dict[str, Any]]]:
    """Use Firefly per tier and retain an exact finite-pool audit."""
    selected: list[BeamDesign] = []
    audits: list[dict[str, Any]] = []
    materialized = list(candidates)
    for tier, multiplier in enumerate(multipliers, start=1):
        eligible = [
            candidate
            for candidate in materialized
            if _beam_is_eligible(candidate, multiplier)
        ]
        if not eligible:
            return None, audits
        vectors = [
            (
                item.b_m,
                item.h_m,
                item.top_bar_count,
                item.bottom_bar_count,
                item.top_bar_diameter_m,
                item.bottom_bar_diameter_m,
                item.top_bar_layers,
                item.bottom_bar_layers,
                item.side_bar_diameter_m,
                item.stirrup_diameter_m,
                item.stirrup_legs,
                item.stirrup_spacing_m,
                item.side_bar_count_each,
            )
            for item in eligible
        ]
        seed = int(
            stable_hash(
                {
                    **seed_prefix,
                    "member": "beam",
                    "tier": tier,
                    "multiplier": multiplier,
                }
            )[:8],
            16,
        )
        audit = optimize_discrete_candidates(
            vectors,
            [item.objective_cost_per_m for item in eligible],
            seed=seed,
            settings=firefly_settings,
        )
        chosen = eligible[audit.selected_index]
        selected.append(chosen)
        audits.append(
            {
                **asdict(audit),
                "tier": tier,
                "strength_multiplier": multiplier,
                "feasible_candidate_count": len(eligible),
                "selected_objective_cost_per_m": (
                    chosen.objective_cost_per_m
                ),
            }
        )
    return selected, audits


def _column_candidates(
    fc_kn_m2: float,
    fy_kn_m2: float,
    *,
    material_detailing: dict[str, Any],
    model_config: dict[str, Any],
    section_search: dict[str, Any],
    unit_costs: dict[str, Any],
    multipliers: list[float],
    demand_for_candidate: Any,
    minimum_moment_demand_knm: float,
    member_length_m: float,
    capacity_design_shear: dict[str, Any],
) -> list[ColumnDesign]:
    """Adaptively enumerate columns and certify that unseen sizes cost more.

    The square-section equality constraint preserves the X-Y symmetry of the
    one-bay archetype.  Section size has no physical upper bound: the search
    expands on the configured construction grid until a concrete-plus-formwork
    lower bound for every unseen section exceeds the most expensive of the six
    exact tier optima.  Longitudinal bar counts and hoop legs are generated
    from spacing/detailing limits instead of a fixed finite list.
    """
    candidates: list[ColumnDesign] = []
    clear_cover_m = float(material_detailing["clear_cover_m"])
    configured_minimum_clear_spacing_m = float(
        material_detailing.get(
            "column_minimum_clear_bar_spacing_m",
            0.040,
        )
    )
    maximum_lateral_support_spacing_m = float(
        section_search["maximum_lateral_support_spacing_m"]
    )
    transverse_fy_by_diameter = section_search[
        "transverse_fy_ksc_by_diameter_mm"
    ]
    minimum_size_m = float(section_search["minimum_size_m"])
    size_increment_m = float(section_search["size_increment_m"])
    maximum_size_m = float(section_search["initial_search_maximum_m"])
    expansion_size_m = float(section_search["expansion_size_m"])
    computational_guard = int(
        section_search["computational_guard_expansions"]
    )
    minimum_bar_count = int(section_search["minimum_bar_count"])
    bar_count_increment = int(section_search["bar_count_increment"])
    bar_diameters = tuple(
        float(value) for value in section_search["bar_diameters_m"]
    )
    hoop_diameters = tuple(
        float(value) for value in section_search["hoop_diameters_m"]
    )
    hoop_spacings = tuple(
        float(value) for value in section_search["hoop_spacings_m"]
    )
    additional_leg_options = tuple(
        int(value)
        for value in section_search["additional_hoop_leg_options"]
    )
    evaluated_sizes: set[float] = set()
    capacity_shear_enabled = bool(capacity_design_shear["enabled"])
    probable_strength_factor = float(
        capacity_design_shear["probable_steel_strength_factor"]
    )
    column_axial_variation_margin = float(
        capacity_design_shear[
            "column_candidate_axial_variation_margin"
        ]
    )
    shear_phi = float(
        capacity_design_shear["strength_reduction_factor"]
    )

    for _ in range(computational_guard + 1):
        sizes = _inclusive_grid(
            minimum_size_m,
            maximum_size_m,
            size_increment_m,
        )
        for size_m in sizes:
            if size_m in evaluated_sizes:
                continue
            evaluated_sizes.add(size_m)
            ag = size_m**2
            for diameter_m in bar_diameters:
                required_clear_spacing_m = max(
                    configured_minimum_clear_spacing_m,
                    1.5 * diameter_m,
                )
                smallest_hoop = min(hoop_diameters)
                largest_bar_half_dimension = (
                    size_m / 2.0
                    - clear_cover_m
                    - smallest_hoop
                    - diameter_m / 2.0
                )
                if largest_bar_half_dimension <= 0.0:
                    continue
                maximum_intervals_per_edge = int(
                    math.floor(
                        2.0
                        * largest_bar_half_dimension
                        / (diameter_m + required_clear_spacing_m)
                        + 1.0e-12
                    )
                )
                maximum_count_by_spacing = (
                    4 * maximum_intervals_per_edge
                )
                maximum_count_by_ratio = int(
                    math.floor(
                        0.04 * ag / _bar_area(diameter_m) + 1.0e-12
                    )
                )
                maximum_count = min(
                    maximum_count_by_spacing,
                    maximum_count_by_ratio
                    - maximum_count_by_ratio % bar_count_increment,
                )
                required_count = max(
                    minimum_bar_count,
                    int(
                        math.ceil(
                            0.01
                            * ag
                            / _bar_area(diameter_m)
                            - 1.0e-12
                        )
                    ),
                )
                minimum_count = (
                    math.ceil(required_count / bar_count_increment)
                    * bar_count_increment
                )
                for bar_count in range(
                    minimum_count,
                    maximum_count + 1,
                    bar_count_increment,
                ):
                    as_total = bar_count * _bar_area(diameter_m)
                    rho = as_total / ag
                    if not 0.01 <= rho <= 0.04:
                        continue
                    (
                        axial_demand_kn,
                        lateral_moment_demand_knm,
                    ) = demand_for_candidate(
                        ColumnGeometry(b_m=size_m, h_m=size_m)
                    )
                    base_moment_demand_knm = max(
                        lateral_moment_demand_knm,
                        minimum_moment_demand_knm,
                    )
                    tier_axial_demands_kn = tuple(
                        float(multiplier) * axial_demand_kn
                        for multiplier in multipliers
                    )
                    tier_moment_demands_knm = tuple(
                        float(multiplier) * base_moment_demand_knm
                        for multiplier in multipliers
                    )
                    intervals_per_edge = bar_count // 4
                    core_dimension_m = max(
                        size_m - 2.0 * clear_cover_m,
                        0.0,
                    )
                    minimum_legs = max(
                        2,
                        int(
                            math.ceil(
                                core_dimension_m
                                / maximum_lateral_support_spacing_m
                                - 1.0e-12
                            )
                        )
                        + 1,
                    )
                    hoop_options: list[ColumnDesign] = []
                    interaction_by_hoop_diameter: dict[
                        float,
                        dict[str, Any],
                    ] = {}
                    for (
                        hoop_diameter_m,
                        hoop_spacing_m,
                        additional_legs,
                    ) in itertools.product(
                        hoop_diameters,
                        hoop_spacings,
                        additional_leg_options,
                    ):
                        bar_half_dimension = (
                            size_m / 2.0
                            - clear_cover_m
                            - hoop_diameter_m
                            - diameter_m / 2.0
                        )
                        if bar_half_dimension <= 0.0:
                            continue
                        clear_spacing_m = (
                            2.0
                            * bar_half_dimension
                            / intervals_per_edge
                            - diameter_m
                        )
                        if (
                            clear_spacing_m + 1.0e-12
                            < required_clear_spacing_m
                        ):
                            continue
                        hoop_legs = minimum_legs + additional_legs
                        if hoop_spacing_m > min(
                            16.0 * diameter_m,
                            48.0 * hoop_diameter_m,
                            size_m,
                            0.20,
                        ) + 1.0e-12:
                            continue
                        hoop_fy_ksc = float(
                            transverse_fy_by_diameter[
                                _diameter_key(hoop_diameter_m)
                            ]
                        )
                        hoop_fy_kn_m2 = hoop_fy_ksc * KSC_TO_KN_M2
                        if hoop_diameter_m not in interaction_by_hoop_diameter:
                            pure_bending = column_pm_capacity_at_axial(
                                width_m=size_m,
                                depth_m=size_m,
                                bar_count=bar_count,
                                bar_diameter_m=diameter_m,
                                clear_cover_m=clear_cover_m,
                                hoop_diameter_m=hoop_diameter_m,
                                fc_kn_m2=fc_kn_m2,
                                fy_kn_m2=fy_kn_m2,
                                steel_elastic_modulus_mpa=float(
                                    model_config["steel_hysteretic"][
                                        "elastic_modulus_mpa"
                                    ]
                                ),
                                target_axial_kn=0.0,
                                strength_mode="design",
                                probable_steel_strength_factor=(
                                    probable_strength_factor
                                ),
                            )
                            tier_strengths = tuple(
                                column_pm_capacity_at_axial(
                                    width_m=size_m,
                                    depth_m=size_m,
                                    bar_count=bar_count,
                                    bar_diameter_m=diameter_m,
                                    clear_cover_m=clear_cover_m,
                                    hoop_diameter_m=hoop_diameter_m,
                                    fc_kn_m2=fc_kn_m2,
                                    fy_kn_m2=fy_kn_m2,
                                    steel_elastic_modulus_mpa=float(
                                        model_config["steel_hysteretic"][
                                            "elastic_modulus_mpa"
                                        ]
                                    ),
                                    target_axial_kn=target_axial,
                                    strength_mode="design",
                                    probable_steel_strength_factor=(
                                        probable_strength_factor
                                    ),
                                )
                                for target_axial in tier_axial_demands_kn
                            )
                            nominal_at_base = column_pm_capacity_at_axial(
                                width_m=size_m,
                                depth_m=size_m,
                                bar_count=bar_count,
                                bar_diameter_m=diameter_m,
                                clear_cover_m=clear_cover_m,
                                hoop_diameter_m=hoop_diameter_m,
                                fc_kn_m2=fc_kn_m2,
                                fy_kn_m2=fy_kn_m2,
                                steel_elastic_modulus_mpa=float(
                                    model_config["steel_hysteretic"][
                                        "elastic_modulus_mpa"
                                    ]
                                ),
                                target_axial_kn=axial_demand_kn,
                                strength_mode="nominal",
                                probable_steel_strength_factor=(
                                    probable_strength_factor
                                ),
                            )
                            probable_at_base = column_pm_capacity_at_axial(
                                width_m=size_m,
                                depth_m=size_m,
                                bar_count=bar_count,
                                bar_diameter_m=diameter_m,
                                clear_cover_m=clear_cover_m,
                                hoop_diameter_m=hoop_diameter_m,
                                fc_kn_m2=fc_kn_m2,
                                fy_kn_m2=fy_kn_m2,
                                steel_elastic_modulus_mpa=float(
                                    model_config["steel_hysteretic"][
                                        "elastic_modulus_mpa"
                                    ]
                                ),
                                target_axial_kn=axial_demand_kn,
                                strength_mode="probable",
                                probable_steel_strength_factor=(
                                    probable_strength_factor
                                ),
                            )
                            interaction_by_hoop_diameter[hoop_diameter_m] = {
                                "pure_bending": pure_bending,
                                "tier_strengths": tier_strengths,
                                "nominal_at_base": nominal_at_base,
                                "probable_at_base": probable_at_base,
                            }
                        interaction = interaction_by_hoop_diameter[
                            hoop_diameter_m
                        ]
                        pure_bending = interaction["pure_bending"]
                        tier_strengths = interaction["tier_strengths"]
                        nominal_at_base = interaction["nominal_at_base"]
                        probable_at_base = interaction["probable_at_base"]
                        if not (
                            bool(pure_bending["valid"])
                            and all(
                                bool(item["valid"])
                                for item in tier_strengths
                            )
                            and bool(nominal_at_base["valid"])
                            and bool(probable_at_base["valid"])
                        ):
                            continue
                        phi_pn = float(pure_bending["maximum_axial_kn"])
                        phi_mn = float(
                            pure_bending["moment_capacity_knm"]
                        )
                        tier_phi_mn_at_axial_knm = tuple(
                            float(item["moment_capacity_knm"])
                            for item in tier_strengths
                        )
                        nominal_mn_at_base_axial_knm = float(
                            nominal_at_base["moment_capacity_knm"]
                        )
                        probable_mn_at_base_axial_knm = float(
                            probable_at_base["moment_capacity_knm"]
                        )
                        capacity_design_shear_kn = (
                            column_axial_variation_margin
                            * 2.0
                            * probable_mn_at_base_axial_knm
                            / member_length_m
                            if capacity_shear_enabled
                            else 0.0
                        )
                        shear_depth_m = 0.80 * size_m
                        av_m2 = hoop_legs * _bar_area(hoop_diameter_m)
                        vc_kn = (
                            170.0
                            * math.sqrt(fc_kn_m2 / 1000.0)
                            * size_m
                            * shear_depth_m
                        )
                        vs_kn = (
                            av_m2
                            * hoop_fy_kn_m2
                            * shear_depth_m
                            / hoop_spacing_m
                        )
                        maximum_vn_kn = (
                            660.0
                            * math.sqrt(fc_kn_m2 / 1000.0)
                            * size_m
                            * shear_depth_m
                        )
                        phi_vn_kn = shear_phi * min(
                            vc_kn + vs_kn,
                            maximum_vn_kn,
                        )
                        if (
                            capacity_shear_enabled
                            and phi_vn_kn + 1.0e-9
                            < capacity_design_shear_kn
                        ):
                            continue
                        option_cost = _column_cost_per_m(
                            candidate=None,
                            b_m=size_m,
                            h_m=size_m,
                            bar_count=bar_count,
                            bar_diameter_m=diameter_m,
                            hoop_diameter_m=hoop_diameter_m,
                            hoop_spacing_m=hoop_spacing_m,
                            hoop_legs_x=hoop_legs,
                            hoop_legs_y=hoop_legs,
                            clear_cover_m=clear_cover_m,
                            fc_kn_m2=fc_kn_m2,
                            unit_costs=unit_costs,
                        )
                        option = ColumnDesign(
                            b_m=size_m,
                            h_m=size_m,
                            bar_count=bar_count,
                            bar_diameter_m=diameter_m,
                            phi_pn_kn=phi_pn,
                            phi_mn_knm=phi_mn,
                            phi_vn_kn=phi_vn_kn,
                            capacity_design_shear_kn=(
                                capacity_design_shear_kn
                            ),
                            reinforcement_ratio=rho,
                            hoop_diameter_m=hoop_diameter_m,
                            hoop_spacing_m=hoop_spacing_m,
                            hoop_legs_x=hoop_legs,
                            hoop_legs_y=hoop_legs,
                            hoop_fy_ksc=hoop_fy_ksc,
                            tier_strength_multipliers=tuple(
                                float(value) for value in multipliers
                            ),
                            tier_axial_demands_kn=tier_axial_demands_kn,
                            tier_moment_demands_knm=(
                                tier_moment_demands_knm
                            ),
                            tier_phi_mn_at_axial_knm=(
                                tier_phi_mn_at_axial_knm
                            ),
                            nominal_mn_at_base_axial_knm=(
                                nominal_mn_at_base_axial_knm
                            ),
                            probable_mn_at_base_axial_knm=(
                                probable_mn_at_base_axial_knm
                            ),
                            objective_cost_per_m=option_cost,
                        )
                        try:
                            _column_confinement_preflight(
                                option,
                                fc_kn_m2=fc_kn_m2,
                                model_config=model_config,
                            )
                        except ValueError:
                            continue
                        hoop_options.append(option)
                    if hoop_options:
                        # For a fixed section and longitudinal layout the
                        # capacity-design shear demand is fixed. Every hoop
                        # option is checked for shear and confinement first;
                        # retaining the cheapest passing option is exact.
                        candidates.append(
                            min(
                                hoop_options,
                                key=lambda item: (
                                    item.objective_cost_per_m,
                                    item.hoop_spacing_m,
                                    item.hoop_diameter_m,
                                ),
                            )
                        )

        selected = _select_column_designs_exact(
            candidates,
            multipliers,
            demand_for_candidate,
            minimum_moment_demand_knm,
        )
        if selected:
            maximum_selected_cost = max(
                item.objective_cost_per_m for item in selected
            )
            next_size_m = (
                max(evaluated_sizes) + size_increment_m
            )
            minimum_unseen_cost = (
                next_size_m**2
                * _material_unit_price(
                    unit_costs,
                    material="concrete",
                    fc_kn_m2=fc_kn_m2,
                )
                + 4.0
                * next_size_m
                * float(unit_costs["formwork_per_m2"])
                + 0.01
                * next_size_m**2
                * float(unit_costs["steel_density_kg_m3"])
                * min(
                    float(value)
                    for value in unit_costs[
                        "longitudinal_steel_per_kg_by_diameter_mm"
                    ].values()
                )
            )
            if (
                minimum_unseen_cost
                > maximum_selected_cost + 1.0e-12
            ):
                return sorted(
                    candidates,
                    key=lambda item: (
                        item.objective_cost_per_m,
                        item.b_m,
                        item.reinforcement_ratio,
                        item.bar_count,
                    ),
                )
        maximum_size_m += expansion_size_m
    # Some high-load/long-span Draft 5 base cases are physically incompatible
    # with the flexure-controlled capacity-shear screen: increasing a very
    # stocky column raises 2Mpr/L faster than the code-capped shear strength.
    # Return the fully enumerated feasible library so the caller can mark the
    # missing target tier/base case infeasible instead of aborting the entire
    # catalogue. No incomplete tier enters the selected research queue.
    return sorted(
        candidates,
        key=lambda item: (
            item.objective_cost_per_m,
            item.b_m,
            item.reinforcement_ratio,
            item.bar_count,
        ),
    )


def _column_is_eligible(
    candidate: ColumnDesign,
    multiplier: float,
    demand_for_candidate: Any,
    minimum_moment_demand_knm: float,
) -> bool:
    del demand_for_candidate, minimum_moment_demand_knm
    matching = [
        index
        for index, value in enumerate(candidate.tier_strength_multipliers)
        if math.isclose(
            float(value),
            float(multiplier),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]
    if len(matching) != 1:
        raise ValueError(
            "Column candidate does not contain one exact P-M tier for "
            f"multiplier {multiplier}"
        )
    tier_index = matching[0]
    axial_demand_kn = candidate.tier_axial_demands_kn[tier_index]
    moment_demand_knm = candidate.tier_moment_demands_knm[tier_index]
    moment_capacity_knm = candidate.tier_phi_mn_at_axial_knm[tier_index]
    axial_utilization = axial_demand_kn / candidate.phi_pn_kn
    moment_utilization = (
        moment_demand_knm / max(moment_capacity_knm, 1.0e-12)
    )
    return bool(
        axial_utilization <= 0.60
        and moment_utilization <= 1.0
        and candidate.phi_vn_kn + 1.0e-9
        >= candidate.capacity_design_shear_kn
    )


def _select_column_designs_exact(
    candidates: Iterable[ColumnDesign],
    multipliers: list[float],
    demand_for_candidate: Any,
    minimum_moment_demand_knm: float,
) -> list[ColumnDesign] | None:
    """Return each tier's exact cheapest column; repeated designs are valid."""
    materialized = list(candidates)
    selected: list[ColumnDesign] = []
    for multiplier in multipliers:
        eligible = [
            candidate
            for candidate in materialized
            if _column_is_eligible(
                candidate,
                multiplier,
                demand_for_candidate,
                minimum_moment_demand_knm,
            )
        ]
        if not eligible:
            return None
        selected.append(
            min(
                eligible,
                key=lambda item: (
                    item.objective_cost_per_m,
                    item.b_m,
                    item.reinforcement_ratio,
                    item.bar_count,
                ),
            )
        )
    return selected


def _select_column_designs(
    candidates: Iterable[ColumnDesign],
    multipliers: list[float],
    demand_for_candidate: Any,
    minimum_moment_demand_knm: float,
    *,
    firefly_settings: dict[str, Any],
    seed_prefix: dict[str, Any],
) -> tuple[list[ColumnDesign] | None, list[dict[str, Any]]]:
    """Select tiers satisfying axial-flexural interaction including self-weight."""
    selected: list[ColumnDesign] = []
    audits: list[dict[str, Any]] = []
    materialized = list(candidates)
    for multiplier in multipliers:
        eligible = [
            candidate
            for candidate in materialized
            if _column_is_eligible(
                candidate,
                multiplier,
                demand_for_candidate,
                minimum_moment_demand_knm,
            )
        ]
        if not eligible:
            return None, audits
        vectors = [
            (
                item.b_m,
                item.h_m,
                item.bar_count,
                item.bar_diameter_m,
                item.hoop_diameter_m,
                item.hoop_spacing_m,
                item.hoop_legs_x,
                item.hoop_legs_y,
                item.reinforcement_ratio,
            )
            for item in eligible
        ]
        tier = len(selected) + 1
        seed = int(
            stable_hash(
                {
                    **seed_prefix,
                    "member": "column",
                    "tier": tier,
                    "multiplier": multiplier,
                }
            )[:8],
            16,
        )
        audit = optimize_discrete_candidates(
            vectors,
            [item.objective_cost_per_m for item in eligible],
            seed=seed,
            settings=firefly_settings,
        )
        chosen = eligible[audit.selected_index]
        selected.append(chosen)
        audits.append(
            {
                **asdict(audit),
                "tier": tier,
                "strength_multiplier": multiplier,
                "feasible_candidate_count": len(eligible),
                "selected_objective_cost_per_m": (
                    chosen.objective_cost_per_m
                ),
            }
        )
    return selected, audits


def _base_designs(
    *,
    fc_ksc: float,
    number_of_bays: int,
    bay_width_m: float,
    sdl_kg_m2: float,
    ll_kg_m2: float,
    stories: int,
    story_height_m: float,
    strength_multipliers: list[float],
    fy_ksc: float,
    concrete_density_kn_m3: float,
    dead_load_factor: float,
    live_load_factor: float,
    live_load_mass_fraction: float,
    gravity_m_s2: float,
    material_detailing: dict[str, Any],
    slab_design: dict[str, Any],
    beam_section_search: dict[str, Any],
    column_section_search: dict[str, Any],
    beam_serviceability: dict[str, Any],
    beam_moment_coefficients: dict[str, Any],
    firefly_optimization: dict[str, Any],
    steel_elastic_modulus_mpa: float,
    model_config: dict[str, Any],
    capacity_design_shear: dict[str, Any],
) -> dict[str, Any]:
    fc = fc_ksc * KSC_TO_KN_M2
    fy = fy_ksc * KSC_TO_KN_M2
    slab_result = slab_thickness_design(
        bay_width_m=bay_width_m,
        fc_ksc=fc_ksc,
        fy_ksc=fy_ksc,
        sdl_kg_m2=sdl_kg_m2,
        ll_kg_m2=ll_kg_m2,
        concrete_density_kn_m3=concrete_density_kn_m3,
        dead_load_factor=dead_load_factor,
        live_load_factor=live_load_factor,
        slab_design=slab_design,
    )
    slab_m = float(slab_result["selected_thickness_m"])
    sdl = sdl_kg_m2 * KG_M2_TO_KN_M2
    ll = ll_kg_m2 * KG_M2_TO_KN_M2
    dead = slab_m * concrete_density_kn_m3 + sdl
    factored_area_load = (
        dead_load_factor * dead + live_load_factor * ll
    )
    if number_of_bays < 1:
        raise ValueError("number_of_bays must be at least one")
    tributary_rule = str(beam_moment_coefficients["tributary_width_rule"])
    accepted_tributary_rules = {
        "equal_panel_edge_reactions_perimeter_bay_over_4_"
        "interior_bay_over_2",
    }
    if number_of_bays == 1:
        accepted_tributary_rules.add(
            "equal_four_edge_share_bay_over_4"
        )
    if tributary_rule not in accepted_tributary_rules:
        raise ValueError("Unsupported beam gravity tributary-width rule")
    # One common beam section is used throughout.  The governing segment is a
    # perimeter beam for one bay and an interior beam for two or more bays.
    tributary_width = (
        bay_width_m / 4.0
        if number_of_bays == 1
        else bay_width_m / 2.0
    )

    beam_candidates = _beam_candidates(
        fc,
        fy,
        factored_area_load,
        tributary_width,
        bay_width_m,
        concrete_density_kn_m3,
        dead_load_factor,
        material_detailing,
        dead,
        ll,
        steel_elastic_modulus_mpa,
        beam_section_search,
        beam_serviceability,
        strength_multipliers,
        beam_moment_coefficients,
        firefly_optimization["unit_costs"],
        capacity_design_shear,
    )
    beam_candidates = [
        candidate
        for candidate in beam_candidates
        if _beam_confinement_is_valid(
            candidate,
            fc_kn_m2=fc,
            model_config=model_config,
        )
    ]
    seed_prefix = {
        "fc_ksc": fc_ksc,
        "number_of_bays": number_of_bays,
        "bay_width_m": bay_width_m,
        "sdl_kg_m2": sdl_kg_m2,
        "ll_kg_m2": ll_kg_m2,
    }
    beams, beam_firefly_audits = _select_beam_designs(
        beam_candidates,
        strength_multipliers,
        firefly_settings=firefly_optimization,
        seed_prefix=seed_prefix,
    )

    governing_column_tributary_area = (
        bay_width_m**2 / 4.0
        if number_of_bays == 1
        else bay_width_m**2
    )
    governing_beam_length_at_column = (
        bay_width_m if number_of_bays == 1 else 2.0 * bay_width_m
    )
    column_pu = (
        factored_area_load
        * governing_column_tributary_area
        * stories
    )
    maximum_beam_self_weight = (
        max(
            governing_beam_length_at_column
            * beam.b_m
            * beam.h_m
            * concrete_density_kn_m3
            for beam in (beams or [])
        )
        if beams
        else 0.0
    )

    maximum_beam_design_moment = max(
        (beam.design_moment_knm for beam in (beams or [])),
        default=0.0,
    )
    beam_count_at_governing_joint = (
        1 if number_of_bays == 1 else 2
    )
    column_end_preselection_moment_knm = (
        _column_end_preselection_moment_knm(
            beam_count_at_governing_joint,
            maximum_beam_design_moment,
        )
    )
    column_strength_moment = column_end_preselection_moment_knm

    def column_demands(
        candidate: ColumnDesign,
    ) -> tuple[float, float]:
        column_self_weight = (
            story_height_m
            * candidate.b_m
            * candidate.h_m
            * concrete_density_kn_m3
        )
        factored_floor_gravity = (
            factored_area_load * governing_column_tributary_area
            + dead_load_factor
            * (maximum_beam_self_weight + column_self_weight)
        )
        axial = factored_floor_gravity * stories
        return axial, column_end_preselection_moment_knm
    columns_all = _column_candidates(
        fc,
        fy,
        material_detailing=material_detailing,
        model_config=model_config,
        section_search=column_section_search,
        unit_costs=firefly_optimization["unit_costs"],
        multipliers=strength_multipliers,
        demand_for_candidate=column_demands,
        minimum_moment_demand_knm=column_strength_moment,
        member_length_m=story_height_m,
        capacity_design_shear=capacity_design_shear,
    )
    columns, column_firefly_audits = _select_column_designs(
        columns_all,
        strength_multipliers,
        column_demands,
        column_strength_moment,
        firefly_settings=firefly_optimization,
        seed_prefix=seed_prefix,
    )
    column_tier_candidate_counts = {
        str(multiplier): sum(
            _column_is_eligible(
                candidate,
                multiplier,
                column_demands,
                column_strength_moment,
            )
            for candidate in columns_all
        )
        for multiplier in strength_multipliers
    }

    floor_mass = (
        (dead + live_load_mass_fraction * ll)
        * (number_of_bays * bay_width_m) ** 2
        / gravity_m_s2
    )
    return {
        "slab_thickness_m": slab_m,
        "slab_design_result": slab_result,
        "dead_load_kn_m2": dead,
        "live_load_kn_m2": ll,
        "factored_area_load_kn_m2": factored_area_load,
        "beam_gravity_tributary_width_m": tributary_width,
        "beam_gravity_tributary_width_rule": tributary_rule,
        "governing_column_tributary_area_m2": (
            governing_column_tributary_area
        ),
        "governing_beam_length_at_column_m": (
            governing_beam_length_at_column
        ),
        "dead_load_factor": dead_load_factor,
        "live_load_factor": live_load_factor,
        "floor_mass_kn_s2_m": floor_mass,
        "beam_design_moment_knm": (
            max(
                (beam.design_moment_knm for beam in (beams or [])),
                default=math.nan,
            )
        ),
        "beam_design_shear_kn": (
            max(
                (beam.design_shear_kn for beam in (beams or [])),
                default=math.nan,
            )
        ),
        "column_design_axial_kn": column_pu,
        "column_design_moment_knm": column_end_preselection_moment_knm,
        "column_design_moment_method": (
            "one-half governing directional beam-joint design-moment sum "
            "assigned to each column end; no 0.10Pu*storey-height proxy"
        ),
        "column_preselection_joint_beam_count": (
            beam_count_at_governing_joint
        ),
        "column_preselection_joint_beam_sum_knm": (
            beam_count_at_governing_joint
            * maximum_beam_design_moment
        ),
        "column_preselection_per_end_share_factor": 0.5,
        "column_preselection_minimum_per_end_moment_knm": (
            column_end_preselection_moment_knm
        ),
        "beams": beams,
        "columns": columns,
        "beam_firefly_audits": beam_firefly_audits,
        "column_firefly_audits": column_firefly_audits,
        "column_tier_candidate_counts": column_tier_candidate_counts,
        "firefly_optimization": firefly_optimization,
    }


def _confinement_common(model_config: dict[str, Any]) -> dict[str, float]:
    confinement = model_config["mander_confinement"]
    return {
        "clear_cover_m": float(
            model_config["material_detailing"]["clear_cover_m"]
        ),
        "eps_co": float(model_config["hognestad"]["peak_strain"]),
        "transverse_ultimate_strain": float(
            confinement["transverse_ultimate_strain"]
        ),
        "minimum_ultimate_strain": float(
            confinement["minimum_ultimate_strain"]
        ),
        "maximum_ultimate_strain": float(
            confinement["maximum_ultimate_strain"]
        ),
    }


def _beam_confinement_preflight(
    beam: BeamDesign,
    *,
    fc_kn_m2: float,
    model_config: dict[str, Any],
) -> tuple[Any, int]:
    common = _confinement_common(model_config)
    clear_cover_m = common["clear_cover_m"]
    minimum_clear_spacing_m = float(
        model_config["material_detailing"].get(
            "beam_minimum_clear_bar_spacing_m",
            0.025,
        )
    )
    nominal_maximum_aggregate_size_m = float(
        model_config["material_detailing"][
            "nominal_maximum_coarse_aggregate_size_m"
        ]
    )
    actual_fibers = beam_reinforcement_fibers(
        width_m=beam.b_m,
        depth_m=beam.h_m,
        clear_cover_m=clear_cover_m,
        stirrup_diameter_m=beam.stirrup_diameter_m,
        top_bar_count=beam.top_bar_count,
        top_bar_diameter_m=beam.top_bar_diameter_m,
        top_bar_layers=beam.top_bar_layers,
        bottom_bar_count=beam.bottom_bar_count,
        bottom_bar_diameter_m=beam.bottom_bar_diameter_m,
        bottom_bar_layers=beam.bottom_bar_layers,
        side_bar_count_each=beam.side_bar_count_each,
        side_bar_diameter_m=beam.side_bar_diameter_m,
        minimum_clear_spacing_m=minimum_clear_spacing_m,
        nominal_maximum_aggregate_size_m=(
            nominal_maximum_aggregate_size_m
        ),
        minimum_bars_per_layer=2,
    )
    equivalent_diameter_m = max(
        beam.top_bar_diameter_m,
        beam.bottom_bar_diameter_m,
        beam.side_bar_diameter_m,
    )
    bar_half_width_m = (
        beam.b_m / 2.0
        - clear_cover_m
        - beam.stirrup_diameter_m
        - equivalent_diameter_m / 2.0
    )
    bar_half_depth_m = (
        beam.h_m / 2.0
        - clear_cover_m
        - beam.stirrup_diameter_m
        - equivalent_diameter_m / 2.0
    )
    positions = beam_bar_positions(
        (
            beam.top_bar_count
            if beam.top_bar_layers == 1
            else math.ceil(
                beam.top_bar_count / beam.top_bar_layers
            )
        ),
        beam.side_bar_count_each,
        half_width_m=bar_half_width_m,
        half_depth_m=bar_half_depth_m,
        bottom_bar_count=(
            beam.bottom_bar_count
            if beam.bottom_bar_layers == 1
            else math.ceil(
                beam.bottom_bar_count / beam.bottom_bar_layers
            )
        ),
    )
    total_longitudinal_area_m2 = sum(
        fiber[2] for fiber in actual_fibers
    )
    confined = mander_rectangular_confined_parameters(
        fc_kn_m2,
        width_m=beam.b_m,
        depth_m=beam.h_m,
        hoop_diameter_m=beam.stirrup_diameter_m,
        hoop_spacing_m=beam.stirrup_spacing_m,
        transverse_legs_x=beam.stirrup_legs,
        transverse_legs_y=beam.stirrup_legs,
        longitudinal_bar_diameter_m=equivalent_diameter_m,
        longitudinal_bar_area_m2=_bar_area(equivalent_diameter_m),
        bar_positions=positions,
        longitudinal_area_total_m2=total_longitudinal_area_m2,
        transverse_fy_kn_m2=(
            beam.stirrup_fy_ksc * KSC_TO_KN_M2
        ),
        **common,
    )
    return confined, len(actual_fibers)


def _beam_confinement_is_valid(
    beam: BeamDesign,
    *,
    fc_kn_m2: float,
    model_config: dict[str, Any],
) -> bool:
    try:
        _beam_confinement_preflight(
            beam,
            fc_kn_m2=fc_kn_m2,
            model_config=model_config,
        )
    except ValueError:
        return False
    return True


def _column_confinement_preflight(
    column: ColumnDesign,
    *,
    fc_kn_m2: float,
    model_config: dict[str, Any],
) -> Any:
    common = _confinement_common(model_config)
    clear_cover_m = common["clear_cover_m"]
    bar_half_width_m = (
        column.b_m / 2.0
        - clear_cover_m
        - column.hoop_diameter_m
        - column.bar_diameter_m / 2.0
    )
    bar_half_depth_m = (
        column.h_m / 2.0
        - clear_cover_m
        - column.hoop_diameter_m
        - column.bar_diameter_m / 2.0
    )
    positions = perimeter_bar_positions(
        column.bar_count,
        half_width_m=bar_half_width_m,
        half_depth_m=bar_half_depth_m,
    )
    return mander_rectangular_confined_parameters(
        fc_kn_m2,
        width_m=column.b_m,
        depth_m=column.h_m,
        hoop_diameter_m=column.hoop_diameter_m,
        hoop_spacing_m=column.hoop_spacing_m,
        transverse_legs_x=column.hoop_legs_x,
        transverse_legs_y=column.hoop_legs_y,
        longitudinal_bar_diameter_m=column.bar_diameter_m,
        longitudinal_bar_area_m2=_bar_area(column.bar_diameter_m),
        bar_positions=positions,
        transverse_fy_kn_m2=column.hoop_fy_ksc * KSC_TO_KN_M2,
        **common,
    )


def _material_model_preflight(
    *,
    beam: BeamDesign,
    column: ColumnDesign,
    fc_kn_m2: float,
    model_config: dict[str, Any],
) -> dict[str, float | int]:
    """Validate the exact bar/confinement geometry before queue selection.

    This uses the same coordinate helpers and Mander implementation as
    ``build_model``.  A catalogue row may be called feasible only when its
    fibre-section material geometry can actually be created.
    """
    beam_confined, beam_position_count = _beam_confinement_preflight(
        beam,
        fc_kn_m2=fc_kn_m2,
        model_config=model_config,
    )
    column_confined = _column_confinement_preflight(
        column,
        fc_kn_m2=fc_kn_m2,
        model_config=model_config,
    )
    return {
        "beam_side_bar_count_each": beam.side_bar_count_each,
        "beam_total_longitudinal_bar_count": beam_position_count,
        "beam_confinement_effectiveness": (
            beam_confined.confinement_effectiveness
        ),
        "beam_confined_strength_kn_m2": beam_confined.fcc_kn_m2,
        "beam_confined_failure_strain": beam_confined.eps_cu,
        "column_confinement_effectiveness": (
            column_confined.confinement_effectiveness
        ),
        "column_confined_strength_kn_m2": column_confined.fcc_kn_m2,
        "column_confined_failure_strain": column_confined.eps_cu,
    }


def _balanced_sample(
    rows: list[dict[str, Any]],
    queue_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tie_break = {row["building_id"]: rng.random() for row in rows}
    fields = (
        "fc_ksc",
        "number_of_bays",
        "bay_width_m",
        "sdl_kg_m2",
        "ll_kg_m2",
        "beam_tier",
        "column_tier",
        "scwb_class",
        "base_case_id",
    )
    counts: Counter[tuple[str, Any]] = Counter()
    remaining = list(rows)
    selected: list[dict[str, Any]] = []
    scwb_order = SCWB_RESEARCH_CLASSES
    available_scwb_classes = [
        class_name
        for class_name in scwb_order
        if any(row["scwb_class"] == class_name for row in rows)
    ]
    class_availability = {
        class_name: sum(
            row["scwb_class"] == class_name for row in rows
        )
        for class_name in available_scwb_classes
    }
    # Capacity-constrained equal allocation ("water filling"). A strict
    # one-third quota can stop early when a minority SCWB class contains fewer
    # physical models than its nominal share. Saturate every scarce class,
    # then redistribute its unused quota to the remaining classes. This gives
    # the most class-balanced queue possible without duplicating buildings.
    class_quotas = _capacity_constrained_equal_quotas(
        available_scwb_classes,
        class_availability,
        queue_size,
    )
    tier_pairs = sorted(
        {
            (int(row["beam_tier"]), int(row["column_tier"]))
            for row in rows
        }
    )
    tier_pair_availability = {
        pair: sum(
            (
                int(row["beam_tier"]),
                int(row["column_tier"]),
            )
            == pair
            for row in rows
        )
        for pair in tier_pairs
    }
    tier_pair_quotas = _capacity_constrained_equal_quotas(
        tier_pairs,
        tier_pair_availability,
        queue_size,
    )

    # Coverage is a hard sampling requirement: every feasible base case must
    # contribute at least one model before marginal balancing fills the
    # remainder of the queue.  This prevents an extreme load/geometry corner
    # from disappearing even when all one-dimensional marginals look balanced.
    base_case_ids = sorted({str(row["base_case_id"]) for row in rows})
    require_complete_base_case_coverage = len(base_case_ids) <= queue_size

    if not require_complete_base_case_coverage:
        # For a production queue smaller than the feasible base-case set,
        # geometry coverage and all configured beam/column target-strength
        # pairs are hard requirements. SCWB class and the remaining material/
        # load marginals are balanced softly inside those constraints.
        bay_values = sorted(
            {int(row["number_of_bays"]) for row in rows}
        )
        base_quota, remainder = divmod(queue_size, len(bay_values))
        bay_quotas = {
            bay: base_quota + int(index < remainder)
            for index, bay in enumerate(bay_values)
        }
        width_quotas: dict[tuple[int, float], int] = {}
        for bay_index, bay in enumerate(bay_values):
            widths = sorted(
                {
                    float(row["bay_width_m"])
                    for row in rows
                    if int(row["number_of_bays"]) == bay
                }
            )
            width_share, width_extra = divmod(
                bay_quotas[bay],
                len(widths),
            )
            # Rotate the remainder so repeated two-width bay groups do not
            # all favour the same width.
            rotated_widths = (
                widths[bay_index % len(widths) :]
                + widths[: bay_index % len(widths)]
            )
            extra_widths = set(rotated_widths[:width_extra])
            for width in widths:
                width_quota = (
                    width_share + int(width in extra_widths)
                )
                width_quotas[(bay, width)] = width_quota

        def production_score(
            row: dict[str, Any],
        ) -> tuple[float, float, float, float, float, float]:
            bay = int(row["number_of_bays"])
            width = float(row["bay_width_m"])
            class_name = str(row["scwb_class"])
            tier_pair = (
                int(row["beam_tier"]),
                int(row["column_tier"]),
            )
            tier_pair_ratio = (
                counts[("tier_pair", tier_pair)]
                / max(tier_pair_quotas[tier_pair], 1)
            )
            class_ratio = (
                counts[("scwb_class", class_name)]
                / max(class_quotas[class_name], 1)
            )
            width_ratio = (
                counts[("bay_width_cell", bay, width)]
                / max(width_quotas[(bay, width)], 1)
            )
            bay_ratio = (
                counts[("number_of_bays", bay)]
                / max(bay_quotas[bay], 1)
            )
            imbalance = sum(
                counts[(field, row[field])] for field in fields
            )
            return (
                class_ratio,
                bay_ratio,
                width_ratio,
                tier_pair_ratio,
                float(imbalance),
                tie_break[row["building_id"]],
            )

        while len(selected) < queue_size:
            eligible = [
                row
                for row in remaining
                if counts[
                    ("number_of_bays", int(row["number_of_bays"]))
                ]
                < bay_quotas[int(row["number_of_bays"])]
                and counts[
                    (
                        "bay_width_cell",
                        int(row["number_of_bays"]),
                        float(row["bay_width_m"]),
                    )
                ]
                < width_quotas[
                    (
                        int(row["number_of_bays"]),
                        float(row["bay_width_m"]),
                    )
                ]
                and counts[
                    (
                        "tier_pair",
                        (
                            int(row["beam_tier"]),
                            int(row["column_tier"]),
                        ),
                    )
                ]
                < tier_pair_quotas[
                    (
                        int(row["beam_tier"]),
                        int(row["column_tier"]),
                    )
                ]
            ]
            if not eligible:
                raise RuntimeError(
                    "Conditional geometry/tier-pair balancing could not fill "
                    "the "
                    "production queue without duplicate models"
                )
            chosen = min(eligible, key=production_score)
            remaining.remove(chosen)
            selected.append(chosen)
            for field in fields:
                counts[(field, chosen[field])] += 1
            counts[
                (
                    "bay_width_cell",
                    int(chosen["number_of_bays"]),
                    float(chosen["bay_width_m"]),
                )
            ] += 1
            counts[
                (
                    "tier_pair",
                    (
                        int(chosen["beam_tier"]),
                        int(chosen["column_tier"]),
                    ),
                )
            ] += 1
        selected_tier_pair_counts = Counter(
            (
                int(row["beam_tier"]),
                int(row["column_tier"]),
            )
            for row in selected
        )
        if any(
            selected_tier_pair_counts[pair] != quota
            for pair, quota in tier_pair_quotas.items()
        ):
            raise RuntimeError(
                "Production queue did not satisfy target-strength pair quotas"
            )
        return selected

    def score(row: dict[str, Any]) -> tuple[float, float, float]:
        class_name = str(row["scwb_class"])
        class_ratio = (
            counts[("scwb_class", class_name)]
            / max(class_quotas[class_name], 1)
        )
        imbalance = sum(counts[(field, row[field])] for field in fields)
        return class_ratio, float(imbalance), tie_break[row["building_id"]]

    if require_complete_base_case_coverage:
        for base_case_id in base_case_ids:
            candidates = [
                row
                for row in remaining
                if str(row["base_case_id"]) == base_case_id
                and counts[("scwb_class", row["scwb_class"])]
                < class_quotas[row["scwb_class"]]
            ]
            if not candidates:
                candidates = [
                    row
                    for row in remaining
                    if str(row["base_case_id"]) == base_case_id
                ]
            if not candidates:
                raise RuntimeError(
                    f"No remaining feasible candidate for {base_case_id}"
                )
            chosen = min(candidates, key=score)
            remaining.remove(chosen)
            selected.append(chosen)
            for field in fields:
                counts[(field, chosen[field])] += 1

    while remaining and len(selected) < queue_size:
        eligible_remaining = [
            row
            for row in remaining
            if counts[("scwb_class", row["scwb_class"])]
            < class_quotas[row["scwb_class"]]
        ]
        if not eligible_remaining:
            break

        chosen = min(eligible_remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        for field in fields:
            counts[(field, chosen[field])] += 1
    return selected


def build_catalog(config: dict[str, Any]) -> dict[str, Any]:
    database_path = config["database_path"]
    initialize(database_path)
    persistent_cache_loaded = 0
    persistent_cache_invalid = 0
    with connect(database_path) as connection:
        cached_rows = list(
            connection.execute(
                """
                SELECT design_cache_key, payload_json
                FROM catalog_design_cache
                WHERE model_schema_version=?
                """,
                (CATALOG_MODEL_SCHEMA_VERSION,),
            )
        )
    for cached_row in cached_rows:
        cache_key = str(cached_row["design_cache_key"])
        if cache_key in _BASE_DESIGN_CACHE:
            continue
        try:
            _BASE_DESIGN_CACHE[cache_key] = _deserialize_base_designs(
                str(cached_row["payload_json"])
            )
            persistent_cache_loaded += 1
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            # A corrupt/incompatible row is never trusted. The content-addressed
            # key will be recomputed and atomically replaced below.
            persistent_cache_invalid += 1
    building = config["building"]
    model = config["model"]
    multipliers = [float(value) for value in building["strength_multipliers"]]
    gravity_combination = building["gravity_strength_combination"]
    dead_load_factor = float(gravity_combination["dead_load_factor"])
    live_load_factor = float(gravity_combination["live_load_factor"])
    scwb_policy = building["scwb_research_database"]
    scwb_minimum_ratio = float(scwb_policy["minimum_ratio"])
    scwb_medium_margin_ratio = float(
        scwb_policy["medium_margin_ratio"]
    )
    scwb_high_margin_ratio = float(scwb_policy["high_margin_ratio"])
    all_rows: list[dict[str, Any]] = []
    base_case_rows: list[dict[str, Any]] = []
    optimizer_audit_rows: list[dict[str, Any]] = []
    invalid_base_cases = 0
    persistent_cache_computed = 0

    combinations = itertools.product(
        building["fc_ksc"],
        building["number_of_bays"],
        building["bay_width_m"],
        building["sdl_kg_m2"],
        building["ll_kg_m2"],
    )
    for (
        fc_ksc,
        number_of_bays,
        bay_width_m,
        sdl_kg_m2,
        ll_kg_m2,
    ) in combinations:
        base_payload = {
            "fc_ksc": float(fc_ksc),
            "number_of_bays": int(number_of_bays),
            "bay_width_m": float(bay_width_m),
            "sdl_kg_m2": float(sdl_kg_m2),
            "ll_kg_m2": float(ll_kg_m2),
            "stories": int(building["stories"]),
            "story_height_m": float(building["story_height_m"]),
        }
        base_case_signature = {
            **base_payload,
            "gravity_strength_combination": gravity_combination,
            "slab_design": building["slab_design"],
            "beam_section_search": building["beam_section_search"],
            "column_section_search": building["column_section_search"],
            "beam_serviceability": building["beam_serviceability"],
            "capacity_design_shear": building["capacity_design_shear"],
        }
        base_case_id = f"BASE-{stable_hash(base_case_signature)[:12]}"
        # For a common section throughout the frame, the governing member
        # demand has two topology classes: one-bay perimeter members, and
        # multi-bay interior members.  Reuse the expensive exact/FFA search
        # within each mechanically equivalent class; total building mass is
        # restored for the actual bay count below.
        design_payload = {
            **base_payload,
            "number_of_bays": (
                1 if int(number_of_bays) == 1 else 2
            ),
        }
        design_cache_signature = {
            **base_case_signature,
            "number_of_bays": design_payload["number_of_bays"],
        }
        design_cache_key = stable_hash(
            {
                "catalog_model_schema_version": (
                    CATALOG_MODEL_SCHEMA_VERSION
                ),
                **design_cache_signature,
                "strength_multipliers": multipliers,
                "steel_fy_ksc": building["steel_fy_ksc"],
                "transverse_steel_fy_ksc": (
                    building["transverse_steel_fy_ksc"]
                ),
                "model_design_inputs": {
                    "concrete_density_kn_m3": (
                        model["concrete_density_kn_m3"]
                    ),
                    "live_load_mass_fraction": (
                        model["live_load_mass_fraction"]
                    ),
                    "gravity_m_s2": model["gravity_m_s2"],
                    "material_detailing": model["material_detailing"],
                    "hognestad": model["hognestad"],
                    "mander_confinement": model[
                        "mander_confinement"
                    ],
                    "steel_elastic_modulus_mpa": (
                        model["steel_hysteretic"][
                            "elastic_modulus_mpa"
                        ]
                    ),
                },
                "beam_gravity_moment_coefficients": (
                    building["beam_gravity_moment_coefficients"]
                ),
                "firefly_optimization": (
                    building["firefly_optimization"]
                ),
            }
        )
        if design_cache_key not in _BASE_DESIGN_CACHE:
            try:
                _BASE_DESIGN_CACHE[design_cache_key] = _base_designs(
                    **design_payload,
                    strength_multipliers=multipliers,
                    fy_ksc=float(building["steel_fy_ksc"]),
                    concrete_density_kn_m3=float(
                        model["concrete_density_kn_m3"]
                    ),
                    dead_load_factor=dead_load_factor,
                    live_load_factor=live_load_factor,
                    live_load_mass_fraction=float(
                        model["live_load_mass_fraction"]
                    ),
                    gravity_m_s2=float(model["gravity_m_s2"]),
                    material_detailing=dict(model["material_detailing"]),
                    slab_design=dict(building["slab_design"]),
                    beam_section_search=dict(
                        building["beam_section_search"]
                    ),
                    column_section_search=dict(
                        building["column_section_search"]
                    ),
                    beam_serviceability=dict(
                        building["beam_serviceability"]
                    ),
                    beam_moment_coefficients=dict(
                        building["beam_gravity_moment_coefficients"]
                    ),
                    firefly_optimization=dict(
                        building["firefly_optimization"]
                    ),
                    capacity_design_shear=dict(
                        building["capacity_design_shear"]
                    ),
                    steel_elastic_modulus_mpa=float(
                        model["steel_hysteretic"]["elastic_modulus_mpa"]
                    ),
                    model_config=dict(model),
                )
            except Exception as exc:
                raise RuntimeError(
                    "Base-design search failed for "
                    f"{json.dumps(base_payload, sort_keys=True)}: {exc}"
                ) from exc
            with transaction(database_path) as connection:
                upsert_many(
                    connection,
                    "catalog_design_cache",
                    [
                        {
                            "design_cache_key": design_cache_key,
                            "model_schema_version": (
                                CATALOG_MODEL_SCHEMA_VERSION
                            ),
                            "payload_json": _serialize_base_designs(
                                _BASE_DESIGN_CACHE[design_cache_key]
                            ),
                            "created_utc": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }
                    ],
                    ("design_cache_key",),
                )
            persistent_cache_computed += 1
        designs = dict(_BASE_DESIGN_CACHE[design_cache_key])
        designs["floor_mass_kn_s2_m"] = (
            (
                float(designs["dead_load_kn_m2"])
                + float(model["live_load_mass_fraction"])
                * float(designs["live_load_kn_m2"])
            )
            * (int(number_of_bays) * float(bay_width_m)) ** 2
            / float(model["gravity_m_s2"])
        )
        for member_type, audits in (
            ("beam", designs["beam_firefly_audits"]),
            ("column", designs["column_firefly_audits"]),
        ):
            for audit in audits:
                optimizer_audit_rows.append(
                    {
                        "base_case_id": base_case_id,
                        "member_type": member_type,
                        "tier": int(audit["tier"]),
                        "strength_multiplier": float(
                            audit["strength_multiplier"]
                        ),
                        "selected_objective_cost_per_m": float(
                            audit["selected_objective_cost_per_m"]
                        ),
                        "exact_verified_objective": float(
                            audit["verified_objective"]
                        ),
                        "firefly_best_objective": float(
                            audit["firefly_objective"]
                        ),
                        "firefly_best_gap": float(
                            audit["optimality_gap"]
                        ),
                        "multi_start_runs": int(
                            audit["multi_start_runs"]
                        ),
                        "multi_start_success_count": int(
                            audit["multi_start_success_count"]
                        ),
                        "multi_start_success_rate": float(
                            audit["multi_start_success_rate"]
                        ),
                        "multi_start_best_gap": float(
                            audit["multi_start_best_gap"]
                        ),
                        "multi_start_mean_gap": float(
                            audit["multi_start_mean_gap"]
                        ),
                        "multi_start_median_gap": float(
                            audit["multi_start_median_gap"]
                        ),
                        "multi_start_worst_gap": float(
                            audit["multi_start_worst_gap"]
                        ),
                        "total_completed_iterations": int(
                            audit["total_completed_iterations"]
                        ),
                        "audit_json": audit,
                    }
                )
        base_case_row = {
            **base_payload,
            "base_case_id": base_case_id,
            "preliminary_design_valid": 1,
            "invalid_reason": None,
            "generated_model_count": 0,
            "feasible_model_count": 0,
        }
        if designs["beams"] is None or designs["columns"] is None:
            invalid_base_cases += 1
            missing = []
            if designs["beams"] is None:
                missing.append("beam")
            if designs["columns"] is None:
                missing.append("column")
            column_counts = designs.get(
                "column_tier_candidate_counts",
                {},
            )
            unavailable_column_tiers = [
                multiplier
                for multiplier, count in column_counts.items()
                if int(count) == 0
            ]
            base_case_row["preliminary_design_valid"] = 0
            base_case_row["invalid_reason"] = (
                f"{' and '.join(missing)} candidate library cannot provide "
                "all configured target-strength tiers after adaptive exact "
                "expansion and flexure-controlled capacity-shear screening"
                + (
                    f"; unavailable column multipliers="
                    f"{unavailable_column_tiers}"
                    if unavailable_column_tiers
                    else ""
                )
            )
            base_case_rows.append(base_case_row)
            continue
        tier_count = len(multipliers)
        for beam_index, column_index in itertools.product(
            range(tier_count),
            repeat=2,
        ):
            # Draft 5 permits the ten target-strength combinations for which
            # the column target is not below the beam target.
            if column_index < beam_index:
                continue
            beam: BeamDesign = designs["beams"][beam_index]
            column: ColumnDesign = designs["columns"][column_index]
            number_of_bays_int = int(number_of_bays)
            columns_per_storey = (number_of_bays_int + 1) ** 2
            beams_per_storey = (
                2 * number_of_bays_int * (number_of_bays_int + 1)
            )
            frame_self_weight_kn_per_floor = (
                beams_per_storey
                * float(bay_width_m)
                * beam.b_m
                * beam.h_m
                * float(model["concrete_density_kn_m3"])
                + columns_per_storey
                * float(building["story_height_m"])
                * column.b_m
                * column.h_m
                * float(model["concrete_density_kn_m3"])
            )
            total_floor_mass = (
                designs["floor_mass_kn_s2_m"]
                + frame_self_weight_kn_per_floor
                / float(model["gravity_m_s2"])
            )
            factored_column_axial_kn = (
                (
                    (
                        dead_load_factor * designs["dead_load_kn_m2"]
                        + live_load_factor
                        * designs["live_load_kn_m2"]
                    )
                    * float(
                        designs["governing_column_tributary_area_m2"]
                    )
                    + dead_load_factor
                    * (
                        float(
                            designs[
                                "governing_beam_length_at_column_m"
                            ]
                        )
                        * beam.b_m
                        * beam.h_m
                        * float(model["concrete_density_kn_m3"])
                        + float(building["story_height_m"])
                        * column.b_m
                        * column.h_m
                        * float(model["concrete_density_kn_m3"])
                    )
                )
                * int(building["stories"])
            )
            column_moment_demand_knm = (
                float(
                    designs[
                        "column_preselection_minimum_per_end_moment_knm"
                    ]
                )
            )
            axial_ratio = factored_column_axial_kn / column.phi_pn_kn
            column_pm_design = column_pm_capacity_at_axial(
                width_m=column.b_m,
                depth_m=column.h_m,
                bar_count=column.bar_count,
                bar_diameter_m=column.bar_diameter_m,
                clear_cover_m=float(
                    model["material_detailing"]["clear_cover_m"]
                ),
                hoop_diameter_m=column.hoop_diameter_m,
                fc_kn_m2=float(fc_ksc) * KSC_TO_KN_M2,
                fy_kn_m2=float(building["steel_fy_ksc"])
                * KSC_TO_KN_M2,
                steel_elastic_modulus_mpa=float(
                    model["steel_hysteretic"]["elastic_modulus_mpa"]
                ),
                target_axial_kn=factored_column_axial_kn,
                strength_mode="design",
                probable_steel_strength_factor=float(
                    building["capacity_design_shear"][
                        "probable_steel_strength_factor"
                    ]
                ),
            )
            column_pm_probable = column_pm_capacity_at_axial(
                width_m=column.b_m,
                depth_m=column.h_m,
                bar_count=column.bar_count,
                bar_diameter_m=column.bar_diameter_m,
                clear_cover_m=float(
                    model["material_detailing"]["clear_cover_m"]
                ),
                hoop_diameter_m=column.hoop_diameter_m,
                fc_kn_m2=float(fc_ksc) * KSC_TO_KN_M2,
                fy_kn_m2=float(building["steel_fy_ksc"])
                * KSC_TO_KN_M2,
                steel_elastic_modulus_mpa=float(
                    model["steel_hysteretic"]["elastic_modulus_mpa"]
                ),
                target_axial_kn=factored_column_axial_kn,
                strength_mode="probable",
                probable_steel_strength_factor=float(
                    building["capacity_design_shear"][
                        "probable_steel_strength_factor"
                    ]
                ),
            )
            column_available_moment_at_axial_knm = float(
                column_pm_design["moment_capacity_knm"]
            )
            interaction_ratio = (
                column_moment_demand_knm
                / max(column_available_moment_at_axial_knm, 1.0e-12)
            )
            paired_column_capacity_design_shear_kn = (
                2.0
                * float(column_pm_probable["moment_capacity_knm"])
                / float(building["story_height_m"])
            )

            # The research SCWB indicator compares nominal strengths in one
            # directional frame plane.  A one-bay corner/exterior joint has
            # one beam framing into the joint; the governing interior joint
            # of a multi-bay frame has two.  Roof joints are excluded because
            # no upper column segment exists.
            factored_axial_per_floor_kn = (
                factored_column_axial_kn / int(building["stories"])
            )
            beam_nominal_joint_moment_knm = max(
                beam.nominal_mn_negative_knm,
                beam.nominal_mn_positive_knm,
            )
            beam_count_at_governing_joint = (
                1 if number_of_bays_int == 1 else 2
            )
            beam_sum_at_governing_joint_knm = (
                beam_count_at_governing_joint
                * beam_nominal_joint_moment_knm
            )
            scwb_joint_results: list[dict[str, float | int]] = []
            for joint_level in range(1, int(building["stories"])):
                below_floor_count = (
                    int(building["stories"]) - joint_level + 1
                )
                above_floor_count = int(building["stories"]) - joint_level
                below_axial_kn = (
                    below_floor_count * factored_axial_per_floor_kn
                )
                above_axial_kn = (
                    above_floor_count * factored_axial_per_floor_kn
                )
                column_nominal_below = column_pm_capacity_at_axial(
                    width_m=column.b_m,
                    depth_m=column.h_m,
                    bar_count=column.bar_count,
                    bar_diameter_m=column.bar_diameter_m,
                    clear_cover_m=float(
                        model["material_detailing"]["clear_cover_m"]
                    ),
                    hoop_diameter_m=column.hoop_diameter_m,
                    fc_kn_m2=float(fc_ksc) * KSC_TO_KN_M2,
                    fy_kn_m2=float(building["steel_fy_ksc"])
                    * KSC_TO_KN_M2,
                    steel_elastic_modulus_mpa=float(
                        model["steel_hysteretic"]["elastic_modulus_mpa"]
                    ),
                    target_axial_kn=below_axial_kn,
                    strength_mode="nominal",
                )
                column_nominal_above = column_pm_capacity_at_axial(
                    width_m=column.b_m,
                    depth_m=column.h_m,
                    bar_count=column.bar_count,
                    bar_diameter_m=column.bar_diameter_m,
                    clear_cover_m=float(
                        model["material_detailing"]["clear_cover_m"]
                    ),
                    hoop_diameter_m=column.hoop_diameter_m,
                    fc_kn_m2=float(fc_ksc) * KSC_TO_KN_M2,
                    fy_kn_m2=float(building["steel_fy_ksc"])
                    * KSC_TO_KN_M2,
                    steel_elastic_modulus_mpa=float(
                        model["steel_hysteretic"]["elastic_modulus_mpa"]
                    ),
                    target_axial_kn=above_axial_kn,
                    strength_mode="nominal",
                )
                column_sum_knm = float(
                    column_nominal_below["moment_capacity_knm"]
                ) + float(column_nominal_above["moment_capacity_knm"])
                scwb_joint_results.append(
                    {
                        "joint_level": joint_level,
                        "column_below_axial_kn": below_axial_kn,
                        "column_above_axial_kn": above_axial_kn,
                        "sum_column_nominal_mn_knm": column_sum_knm,
                        "sum_beam_nominal_mn_knm": (
                            beam_sum_at_governing_joint_knm
                        ),
                        "beam_count_in_directional_plane": (
                            beam_count_at_governing_joint
                        ),
                        "scwb_strength_ratio": (
                            column_sum_knm
                            / max(
                                beam_sum_at_governing_joint_knm,
                                1.0e-12,
                            )
                        ),
                    }
                )
            critical_scwb_joint = min(
                scwb_joint_results,
                key=lambda item: float(item["scwb_strength_ratio"]),
            )
            scwb_strength_ratio = float(
                critical_scwb_joint["scwb_strength_ratio"]
            )
            scwb_class = classify_scwb_ratio(
                scwb_strength_ratio,
                minimum_ratio=scwb_minimum_ratio,
                medium_margin_ratio=scwb_medium_margin_ratio,
                high_margin_ratio=scwb_high_margin_ratio,
            )
            scwb_in_research_scope = scwb_class is not None
            beam_negative_moment_target_ok = (
                beam.phi_mn_negative_knm
                >= multipliers[beam_index]
                * beam.design_negative_moment_knm
            )
            beam_positive_moment_target_ok = (
                beam.phi_mn_positive_knm
                >= multipliers[beam_index]
                * beam.design_positive_moment_knm
            )
            beam_moment_target_ok = (
                beam_negative_moment_target_ok
                and beam_positive_moment_target_ok
            )
            beam_clear_span_m = (
                float(bay_width_m) - column.b_m
            )
            if beam_clear_span_m <= 0.0:
                raise ValueError(
                    "Column width leaves no positive beam clear span"
                )
            paired_beam_capacity_design_shear_kn = (
                beam.design_shear_kn
                + (
                    beam.probable_mn_negative_knm
                    + beam.probable_mn_positive_knm
                )
                / beam_clear_span_m
            )
            beam_shear_target_ok = (
                beam.phi_vn_kn
                >= max(
                    multipliers[beam_index] * beam.design_shear_kn,
                    paired_beam_capacity_design_shear_kn,
                )
            )
            column_shear_target_ok = (
                column.phi_vn_kn + 1.0e-9
                >= paired_column_capacity_design_shear_kn
            )
            beam_live_deflection_ok = (
                beam.immediate_live_deflection_m
                <= beam.live_deflection_limit_m + 1.0e-12
            )
            beam_total_deflection_ok = (
                beam.total_long_term_deflection_m
                <= beam.total_deflection_limit_m + 1.0e-12
            )
            valid = bool(
                scwb_in_research_scope
                and axial_ratio <= 0.60
                and interaction_ratio <= 1.0
                and beam_moment_target_ok
                and beam_shear_target_ok
                and column_shear_target_ok
                and beam_live_deflection_ok
                and beam_total_deflection_ok
            )
            reason = None
            if not scwb_in_research_scope:
                reason = (
                    "Research column/beam strength ratio does not strictly "
                    f"exceed {scwb_minimum_ratio:.2f}"
                )
            elif axial_ratio > 0.60:
                reason = "column axial ratio exceeds 0.60"
            elif interaction_ratio > 1.0:
                reason = (
                    "column moment demand exceeds strain-compatible "
                    "P-M capacity at factored axial load"
                )
            elif not beam_moment_target_ok:
                reason = "beam flexural capacity misses exact target strength"
            elif not beam_shear_target_ok:
                reason = (
                    "beam shear capacity misses gravity/capacity-design "
                    "demand"
                )
            elif not column_shear_target_ok:
                reason = "column shear capacity misses capacity-design demand"
            elif not beam_live_deflection_ok:
                reason = "beam immediate live-load deflection exceeds limit"
            elif not beam_total_deflection_ok:
                reason = "beam total long-term deflection exceeds limit"
            material_preflight: dict[str, float | int] = {}
            try:
                material_preflight = _material_model_preflight(
                    beam=beam,
                    column=column,
                    fc_kn_m2=float(fc_ksc) * KSC_TO_KN_M2,
                    model_config=model,
                )
            except ValueError as exc:
                valid = False
                reason = f"material-model preflight failed: {exc}"
            selected_clear_spacing_m = aci_required_clear_spacing_m(
                bar_diameter_m=beam.bar_diameter_m,
                nominal_maximum_aggregate_size_m=float(
                    model["material_detailing"][
                        "nominal_maximum_coarse_aggregate_size_m"
                    ]
                ),
                code_minimum_m=float(
                    model["material_detailing"][
                        "beam_minimum_clear_bar_spacing_m"
                    ]
                ),
            )
            selected_geometric_maximum_bars_per_layer = int(
                math.floor(
                    (
                        beam.b_m
                        - 2.0
                        * float(
                            model["material_detailing"]["clear_cover_m"]
                        )
                        - 2.0 * beam.stirrup_diameter_m
                        + selected_clear_spacing_m
                    )
                    / (
                        beam.bar_diameter_m
                        + selected_clear_spacing_m
                    )
                    + 1.0e-12
                )
            )
            model_payload = {
                "model_schema_version": CATALOG_MODEL_SCHEMA_VERSION,
                **base_payload,
                "beam": asdict(beam),
                "column": asdict(column),
                "slab_thickness_m": designs["slab_thickness_m"],
                "slab_design": dict(building["slab_design"]),
                "slab_design_result": dict(
                    designs["slab_design_result"]
                ),
                "dead_load_kn_m2": designs["dead_load_kn_m2"],
                "live_load_kn_m2": designs["live_load_kn_m2"],
                "gravity_strength_combination": dict(
                    gravity_combination
                ),
                "beam_section_search": dict(
                    building["beam_section_search"]
                ),
                "column_section_search": dict(
                    building["column_section_search"]
                ),
                "beam_serviceability": dict(
                    building["beam_serviceability"]
                ),
                "capacity_design_shear": dict(
                    building["capacity_design_shear"]
                ),
                "beam_gravity_moment_coefficients": dict(
                    building["beam_gravity_moment_coefficients"]
                ),
                "firefly_optimization": dict(
                    building["firefly_optimization"]
                ),
                "floor_mass_kn_s2_m": total_floor_mass,
                "steel_fy_ksc": float(building["steel_fy_ksc"]),
                "transverse_steel_fy_ksc": float(
                    building["transverse_steel_fy_ksc"]
                ),
                "concrete_density_kn_m3": float(
                    model["concrete_density_kn_m3"]
                ),
                "live_load_mass_fraction": float(
                    model["live_load_mass_fraction"]
                ),
                "hognestad_peak_strain": float(
                    model["hognestad"]["peak_strain"]
                ),
                "hognestad_ultimate_strain": float(
                    model["hognestad"]["ultimate_strain"]
                ),
                "hognestad_ultimate_strength_ratio": float(
                    model["hognestad"]["ultimate_strength_ratio"]
                ),
                "concrete_failure_at_epsilon_cu": bool(
                    model["concrete_failure_at_epsilon_cu"]
                ),
                "mander_confinement": dict(model["mander_confinement"]),
                "fiber_mesh": dict(model["fiber_mesh"]),
                "material_detailing": dict(model["material_detailing"]),
                "steel_material": str(model["steel_material"]),
                "steel_hysteretic": dict(model["steel_hysteretic"]),
                "beam_integration": str(model["beam_integration"]),
                "plastic_hinge_length_method": str(
                    model["plastic_hinge_length_method"]
                ),
                "plastic_hinge_shear_span_method": str(
                    model["plastic_hinge_shear_span_method"]
                ),
                "plastic_hinge_reference_load_pattern": str(
                    model.get(
                        "plastic_hinge_reference_load_pattern",
                        "legacy_full_member",
                    )
                ),
                "plastic_hinge_minimum_shear_span_ratio": float(
                    model.get(
                        "plastic_hinge_minimum_shear_span_ratio",
                        1.0,
                    )
                ),
                "plastic_hinge_maximum_shear_span_ratio": float(
                    model.get(
                        "plastic_hinge_maximum_shear_span_ratio",
                        1.0,
                    )
                ),
                "plastic_hinge_column_bidirectional_combination": str(
                    model.get(
                        "plastic_hinge_column_bidirectional_combination",
                        "not_applicable",
                    )
                ),
                "plastic_hinge_length_factor": float(
                    model["plastic_hinge_length_factor"]
                ),
                "plastic_hinge_length_scale_factor": float(
                    model["plastic_hinge_length_scale_factor"]
                ),
                "plastic_hinge_expected_strength_factor": float(
                    model["plastic_hinge_expected_strength_factor"]
                ),
                "element_formulation": str(
                    model["element_formulation"]
                ),
                "force_beam_column_max_iterations": int(
                    model["force_beam_column_max_iterations"]
                ),
                "force_beam_column_tolerance": float(
                    model["force_beam_column_tolerance"]
                ),
            }
            model_hash = stable_hash(model_payload)
            all_rows.append(
                {
                    "building_id": f"B-{model_hash[:16]}",
                    "base_case_id": base_case_id,
                    "model_hash": model_hash,
                    "queue_rank": None,
                    "selected": 0,
                    "valid": int(valid),
                    "invalid_reason": reason,
                    **base_payload,
                    "slab_thickness_m": designs["slab_thickness_m"],
                    "beam_tier": beam_index + 1,
                    "column_tier": column_index + 1,
                    "beam_strength_multiplier": multipliers[beam_index],
                    "column_strength_multiplier": multipliers[column_index],
                    "beam_b_m": beam.b_m,
                    "beam_h_m": beam.h_m,
                    "beam_bars_per_face": beam.bars_per_face,
                    "beam_bar_diameter_m": beam.bar_diameter_m,
                    "beam_phi_mn_knm": beam.phi_mn_knm,
                    "beam_phi_vn_kn": beam.phi_vn_kn,
                    "column_b_m": column.b_m,
                    "column_h_m": column.h_m,
                    "column_bar_count": column.bar_count,
                    "column_bar_diameter_m": column.bar_diameter_m,
                    "column_phi_pn_kn": column.phi_pn_kn,
                    "column_phi_mn_knm": column.phi_mn_knm,
                    "axial_ratio": axial_ratio,
                    "scwb_strength_ratio": scwb_strength_ratio,
                    "scwb_class": scwb_class,
                    "dead_load_kn_m2": designs["dead_load_kn_m2"],
                    "live_load_kn_m2": designs["live_load_kn_m2"],
                    "floor_mass_kn_s2_m": total_floor_mass,
                    "design_metadata_json": {
                        "model_schema_version": (
                            CATALOG_MODEL_SCHEMA_VERSION
                        ),
                        "beam_design_moment_knm": designs[
                            "beam_design_moment_knm"
                        ],
                        "beam_design_shear_kn": designs[
                            "beam_design_shear_kn"
                        ],
                        "column_design_axial_kn": designs[
                            "column_design_axial_kn"
                        ],
                        "column_design_moment_knm": designs[
                            "column_design_moment_knm"
                        ],
                        "column_design_moment_method": designs[
                            "column_design_moment_method"
                        ],
                        "column_preselection_joint_beam_count": designs[
                            "column_preselection_joint_beam_count"
                        ],
                        "column_preselection_joint_beam_sum_knm": designs[
                            "column_preselection_joint_beam_sum_knm"
                        ],
                        "column_preselection_per_end_share_factor": designs[
                            "column_preselection_per_end_share_factor"
                        ],
                        "column_preselection_minimum_per_end_moment_knm": (
                            designs[
                                "column_preselection_minimum_per_end_moment_knm"
                            ]
                        ),
                        "gravity_strength_combination": (
                            f"{dead_load_factor:.1f}D"
                            f"+{live_load_factor:.1f}L"
                        ),
                        "gravity_strength_dead_load_factor": (
                            dead_load_factor
                        ),
                        "gravity_strength_live_load_factor": (
                            live_load_factor
                        ),
                        "gravity_strength_standard_reference": str(
                            gravity_combination["standard_reference"]
                        ),
                        "factored_area_load_kn_m2": designs[
                            "factored_area_load_kn_m2"
                        ],
                        "beam_gravity_tributary_width_m": designs[
                            "beam_gravity_tributary_width_m"
                        ],
                        "beam_gravity_tributary_width_rule": designs[
                            "beam_gravity_tributary_width_rule"
                        ],
                        "number_of_bays_each_direction": number_of_bays_int,
                        "plan_width_m": (
                            number_of_bays_int * float(bay_width_m)
                        ),
                        "columns_per_storey": columns_per_storey,
                        "beams_per_storey": beams_per_storey,
                        "governing_column_tributary_area_m2": designs[
                            "governing_column_tributary_area_m2"
                        ],
                        "slab_selection_rule": building["slab_design"][
                            "selection_rule"
                        ],
                        "slab_code_reference": building["slab_design"][
                            "code_reference"
                        ],
                        "slab_selected_thickness_m": designs[
                            "slab_design_result"
                        ]["selected_thickness_m"],
                        "slab_aci_minimum_thickness_m": designs[
                            "slab_design_result"
                        ]["aci_minimum_thickness_m"],
                        "slab_aci_unrounded_thickness_m": designs[
                            "slab_design_result"
                        ]["aci_unrounded_thickness_m"],
                        "slab_shear_required_thickness_m": designs[
                            "slab_design_result"
                        ]["shear_required_thickness_m"],
                        "slab_controlling_criterion": designs[
                            "slab_design_result"
                        ]["controlling_criterion"],
                        "slab_effective_depth_m": designs[
                            "slab_design_result"
                        ]["effective_depth_m"],
                        "slab_factored_load_kn_m2": designs[
                            "slab_design_result"
                        ]["factored_load_kn_m2"],
                        "slab_shear_demand_kn_per_m_strip": designs[
                            "slab_design_result"
                        ]["shear_demand_kn_per_m_strip"],
                        "slab_phi_vc_kn_per_m_strip": designs[
                            "slab_design_result"
                        ]["phi_vc_kn_per_m_strip"],
                        "slab_shear_utilization": designs[
                            "slab_design_result"
                        ]["shear_utilization"],
                        "slab_shear_size_effect_lambda_s": designs[
                            "slab_design_result"
                        ]["lambda_s"],
                        "slab_punching_shear_applicable": False,
                        "slab_punching_shear_note": (
                            "The slab is supported by perimeter beams with "
                            "no direct slab-column gravity transfer; one-way "
                            "slab shear, not punching shear, sizes thickness."
                        ),
                        "nonlinear_gravity_combination": (
                            "1.0D+0.25L plus member self-weight"
                        ),
                        "beam_reinforcement_ratio": beam.reinforcement_ratio,
                        "beam_top_bar_count": beam.top_bar_count,
                        "beam_bottom_bar_count": beam.bottom_bar_count,
                        "beam_top_steel_area_m2": (
                            beam.top_bar_count
                            * _bar_area(beam.top_bar_diameter_m)
                        ),
                        "beam_bottom_steel_area_m2": (
                            beam.bottom_bar_count
                            * _bar_area(beam.bottom_bar_diameter_m)
                        ),
                        "beam_top_bar_diameter_m": (
                            beam.top_bar_diameter_m
                        ),
                        "beam_bottom_bar_diameter_m": (
                            beam.bottom_bar_diameter_m
                        ),
                        "beam_side_bar_diameter_m": (
                            beam.side_bar_diameter_m
                        ),
                        "beam_top_bar_layers": beam.top_bar_layers,
                        "beam_bottom_bar_layers": (
                            beam.bottom_bar_layers
                        ),
                        "beam_longitudinal_diameter_rule": (
                            building["beam_section_search"][
                                "longitudinal_diameter_rule"
                            ]
                        ),
                        "beam_nominal_maximum_coarse_aggregate_size_m": (
                            float(
                                model["material_detailing"][
                                    "nominal_maximum_coarse_aggregate_size_m"
                                ]
                            )
                        ),
                        "beam_required_clear_spacing_m": (
                            selected_clear_spacing_m
                        ),
                        "beam_geometric_maximum_bars_per_layer": (
                            selected_geometric_maximum_bars_per_layer
                        ),
                        "beam_actual_maximum_bars_in_a_layer": max(
                            math.ceil(
                                beam.top_bar_count
                                / beam.top_bar_layers
                            ),
                            math.ceil(
                                beam.bottom_bar_count
                                / beam.bottom_bar_layers
                            ),
                        ),
                        "beam_bars_per_layer_limit_rule": (
                            "No arbitrary bar-count cap; geometric fit using "
                            "ACI clear spacing max(25 mm, db, 4/3 dagg)"
                        ),
                        "beam_multilayer_rule": (
                            "Use the minimum number of layers needed; "
                            "three layers require h >= 0.50 m"
                        ),
                        "beam_phi_mn_negative_knm": (
                            beam.phi_mn_negative_knm
                        ),
                        "beam_phi_mn_positive_knm": (
                            beam.phi_mn_positive_knm
                        ),
                        "beam_design_negative_moment_knm": (
                            beam.design_negative_moment_knm
                        ),
                        "beam_design_positive_moment_knm": (
                            beam.design_positive_moment_knm
                        ),
                        "beam_objective_cost_per_m": (
                            beam.objective_cost_per_m
                        ),
                        "beam_side_bar_count_each": (
                            beam.side_bar_count_each
                        ),
                        "beam_stirrup_diameter_m": (
                            beam.stirrup_diameter_m
                        ),
                        "beam_stirrup_legs": beam.stirrup_legs,
                        "beam_stirrup_spacing_m": (
                            beam.stirrup_spacing_m
                        ),
                        "beam_stirrup_fy_ksc": beam.stirrup_fy_ksc,
                        "beam_exact_design_moment_knm": (
                            beam.design_moment_knm
                        ),
                        "beam_exact_design_shear_kn": (
                            beam.design_shear_kn
                        ),
                        "beam_moment_target_utilization": (
                            multipliers[beam_index]
                            * beam.design_moment_knm
                            / beam.phi_mn_knm
                        ),
                        "beam_negative_moment_target_utilization": (
                            multipliers[beam_index]
                            * beam.design_negative_moment_knm
                            / beam.phi_mn_negative_knm
                        ),
                        "beam_positive_moment_target_utilization": (
                            multipliers[beam_index]
                            * beam.design_positive_moment_knm
                            / beam.phi_mn_positive_knm
                        ),
                        "beam_shear_target_utilization": (
                            max(
                                multipliers[beam_index]
                                * beam.design_shear_kn,
                                paired_beam_capacity_design_shear_kn,
                            )
                            / beam.phi_vn_kn
                        ),
                        "beam_capacity_design_shear_preselection_kn": (
                            beam.capacity_design_shear_kn
                        ),
                        "beam_capacity_design_shear_kn": (
                            paired_beam_capacity_design_shear_kn
                        ),
                        "beam_capacity_design_clear_span_m": (
                            beam_clear_span_m
                        ),
                        "beam_probable_mn_negative_knm": (
                            beam.probable_mn_negative_knm
                        ),
                        "beam_probable_mn_positive_knm": (
                            beam.probable_mn_positive_knm
                        ),
                        "beam_nominal_mn_negative_knm": (
                            beam.nominal_mn_negative_knm
                        ),
                        "beam_nominal_mn_positive_knm": (
                            beam.nominal_mn_positive_knm
                        ),
                        "beam_immediate_live_deflection_m": (
                            beam.immediate_live_deflection_m
                        ),
                        "beam_live_deflection_limit_m": (
                            beam.live_deflection_limit_m
                        ),
                        "beam_live_deflection_utilization": (
                            beam.immediate_live_deflection_m
                            / beam.live_deflection_limit_m
                        ),
                        "beam_total_long_term_deflection_m": (
                            beam.total_long_term_deflection_m
                        ),
                        "beam_total_deflection_limit_m": (
                            beam.total_deflection_limit_m
                        ),
                        "beam_total_deflection_utilization": (
                            beam.total_long_term_deflection_m
                            / beam.total_deflection_limit_m
                        ),
                        "beam_service_effective_inertia_m4": (
                            beam.service_effective_inertia_m4
                        ),
                        "beam_service_cracked_inertia_m4": (
                            beam.service_cracked_inertia_m4
                        ),
                        "beam_serviceability_passed": bool(
                            beam_live_deflection_ok
                            and beam_total_deflection_ok
                        ),
                        "beam_section_search_has_physical_maximum": False,
                        "beam_section_search_method": (
                            "adaptive discrete expansion; computational "
                            "guard raises an error and is not an engineering "
                            "section limit"
                        ),
                        "firefly_unit_cost_basis": (
                            building["firefly_optimization"][
                                "unit_cost_basis"
                            ]
                        ),
                        "firefly_price_snapshot": dict(
                            building["firefly_optimization"][
                                "price_snapshot"
                            ]
                        ),
                        "firefly_unit_costs": dict(
                            building["firefly_optimization"]["unit_costs"]
                        ),
                        "beam_concrete_price_thb_per_m3": (
                            _material_unit_price(
                                building["firefly_optimization"][
                                    "unit_costs"
                                ],
                                material="concrete",
                                fc_kn_m2=float(fc_ksc) * KSC_TO_KN_M2,
                            )
                        ),
                        "beam_longitudinal_steel_price_thb_per_kg": (
                            _material_unit_price(
                                building["firefly_optimization"][
                                    "unit_costs"
                                ],
                                material="longitudinal",
                                diameter_m=beam.bar_diameter_m,
                            )
                        ),
                        "beam_top_steel_price_thb_per_kg": (
                            _material_unit_price(
                                building["firefly_optimization"][
                                    "unit_costs"
                                ],
                                material="longitudinal",
                                diameter_m=beam.top_bar_diameter_m,
                            )
                        ),
                        "beam_bottom_steel_price_thb_per_kg": (
                            _material_unit_price(
                                building["firefly_optimization"][
                                    "unit_costs"
                                ],
                                material="longitudinal",
                                diameter_m=beam.bottom_bar_diameter_m,
                            )
                        ),
                        "beam_side_steel_price_thb_per_kg": (
                            _material_unit_price(
                                building["firefly_optimization"][
                                    "unit_costs"
                                ],
                                material="longitudinal",
                                diameter_m=beam.side_bar_diameter_m,
                            )
                        ),
                        "beam_transverse_steel_price_thb_per_kg": (
                            _material_unit_price(
                                building["firefly_optimization"][
                                    "unit_costs"
                                ],
                                material="transverse",
                                diameter_m=beam.stirrup_diameter_m,
                            )
                        ),
                        "beam_firefly_audit": designs[
                            "beam_firefly_audits"
                        ][beam_index],
                        "column_reinforcement_ratio": column.reinforcement_ratio,
                        "column_objective_cost_per_m": (
                            column.objective_cost_per_m
                        ),
                        "column_firefly_audit": designs[
                            "column_firefly_audits"
                        ][column_index],
                        "column_factored_axial_with_member_weight_kn": (
                            factored_column_axial_kn
                        ),
                        "column_moment_demand_knm": (
                            column_moment_demand_knm
                        ),
                        "column_axial_flexural_interaction_ratio": (
                            interaction_ratio
                        ),
                        "column_pm_method": (
                            "ACI strain compatibility with Whitney block, "
                            "discrete perimeter bars, tied-column phi "
                            "transition, and 0.80 maximum axial cap"
                        ),
                        "column_pm_design_at_factored_axial": (
                            column_pm_design
                        ),
                        "column_pm_probable_at_factored_axial": (
                            column_pm_probable
                        ),
                        "column_phi_vn_kn": column.phi_vn_kn,
                        "column_capacity_design_shear_kn": (
                            paired_column_capacity_design_shear_kn
                        ),
                        "column_capacity_design_shear_utilization": (
                            paired_column_capacity_design_shear_kn
                            / column.phi_vn_kn
                        ),
                        "capacity_design_shear_policy": dict(
                            building["capacity_design_shear"]
                        ),
                        "column_available_phi_mn_at_axial_knm": (
                            column_available_moment_at_axial_knm
                        ),
                        "scwb_strength_ratio_at_factored_axial": (
                            scwb_strength_ratio
                        ),
                        "scwb_method": (
                            "minimum non-roof joint ratio of the sum of "
                            "strain-compatible nominal column strengths at "
                            "their factored axial loads to the sum of nominal "
                            "beam strengths in the governing directional "
                            "frame-plane joint; "
                            "research admission requires ratio > 1.00 and "
                            "does not claim ACI SCWB code compliance"
                        ),
                        "scwb_critical_joint": critical_scwb_joint,
                        "scwb_joint_results": scwb_joint_results,
                        "scwb_class": scwb_class,
                        "scwb_research_minimum_ratio": scwb_minimum_ratio,
                        "scwb_code_compliance_claim": False,
                        "column_hoop_diameter_m": column.hoop_diameter_m,
                        "column_hoop_spacing_m": column.hoop_spacing_m,
                        "column_hoop_legs_x": column.hoop_legs_x,
                        "column_hoop_legs_y": column.hoop_legs_y,
                        "column_hoop_fy_ksc": column.hoop_fy_ksc,
                        "column_section_shape_constraint": (
                            building["column_section_search"][
                                "shape_constraint"
                            ]
                        ),
                        "column_section_search_has_physical_maximum": False,
                        "column_section_search_method": (
                            "adaptive discrete expansion with an admissible "
                            "concrete-plus-formwork lower-bound certificate; "
                            "the computational guard is not an engineering "
                            "section limit"
                        ),
                        "column_transverse_exact_dominance_pruning": (
                            "For each square section and longitudinal-bar "
                            "combination, every permitted hoop diameter, "
                            "spacing and X/Y leg arrangement is checked. "
                            "Each option must pass capacity-design shear and "
                            "confinement; only the least-cost passing hoop "
                            "arrangement is retained before Firefly without "
                            "changing the optimum."
                        ),
                        "column_hinge_tie_assumption": (
                            "Firefly-selected RB6/RB9 or DB12 rectangular "
                            "hoops/crossties; Mander confinement credited "
                            "from the selected detailed core geometry"
                        ),
                        "frame_self_weight_kn_per_floor": (
                            frame_self_weight_kn_per_floor
                        ),
                        "material_model_preflight_passed": bool(
                            material_preflight
                        ),
                        **material_preflight,
                    },
                }
            )
            base_case_row["generated_model_count"] += 1
            if valid:
                base_case_row["feasible_model_count"] += 1
        base_case_rows.append(base_case_row)

    if bool(config.get("_catalog_cache_only", False)):
        return {
            "cache_only": True,
            "processed_base_case_count": len(base_case_rows),
            "invalid_base_case_count": invalid_base_cases,
            "persistent_design_cache_loaded": persistent_cache_loaded,
            "persistent_design_cache_computed": persistent_cache_computed,
            "persistent_design_cache_invalid": persistent_cache_invalid,
        }

    raw_combination_count = len(all_rows)
    rows_by_physical_model: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        rows_by_physical_model.setdefault(str(row["model_hash"]), []).append(row)
    deduplicated_rows: list[dict[str, Any]] = []
    duplicate_group_count = 0
    for equivalent_rows in rows_by_physical_model.values():
        if len(equivalent_rows) > 1:
            duplicate_group_count += 1
        valid_equivalents = [
            row for row in equivalent_rows if bool(row["valid"])
        ]
        candidates = valid_equivalents or equivalent_rows
        representative = max(
            candidates,
            key=lambda row: (
                float(row["beam_strength_multiplier"]),
                float(row["column_strength_multiplier"]),
            ),
        ).copy()
        metadata = dict(representative["design_metadata_json"])
        metadata["equivalent_generation_tiers"] = sorted(
            {
                (
                    int(row["beam_tier"]),
                    int(row["column_tier"]),
                )
                for row in equivalent_rows
                if bool(row["valid"]) == bool(representative["valid"])
            }
        )
        metadata["full_building_duplicate_generation_count"] = len(
            equivalent_rows
        )
        metadata["full_building_deduplication_basis"] = (
            "SHA-256 of the complete model payload including geometry, "
            "loads, mass, materials, member reinforcement, analysis-model "
            "settings, and fixed research-price/design configuration"
        )
        metadata["full_building_duplicate_analyzed_once"] = bool(
            len(equivalent_rows) > 1
        )
        representative["design_metadata_json"] = metadata
        deduplicated_rows.append(representative)
    all_rows = deduplicated_rows
    feasible = [row for row in all_rows if row["valid"]]
    queue_size = int(building["queue_size"])
    if len(feasible) < queue_size:
        invalid_reason_counts = Counter(
            str(row.get("invalid_reason") or "unspecified")
            for row in all_rows
            if not row["valid"]
        )
        raise RuntimeError(
            f"Only {len(feasible)} feasible models were generated; "
            f"{queue_size} are required. Invalid reasons: "
            f"{dict(invalid_reason_counts)}"
        )
    selected = _balanced_sample(feasible, queue_size, int(config["random_seed"]))
    if len(selected) != queue_size:
        raise RuntimeError(
            "Balanced sampling did not fill the requested queue without "
            f"duplicates: selected {len(selected)} of {queue_size}"
        )
    selected_base_case_count = len(
        {str(row["base_case_id"]) for row in selected}
    )
    feasible_base_case_count = len(
        {str(row["base_case_id"]) for row in feasible}
    )
    if (
        feasible_base_case_count <= queue_size
        and selected_base_case_count != feasible_base_case_count
    ):
        raise RuntimeError(
            "Balanced sampling failed the mandatory feasible-base-case "
            "coverage gate"
        )
    selected_ids = {row["building_id"] for row in selected}
    queue_ranks = {
        row["building_id"]: rank for rank, row in enumerate(selected, start=1)
    }
    for row in all_rows:
        if row["building_id"] in selected_ids:
            row["selected"] = 1
            row["queue_rank"] = queue_ranks[row["building_id"]]

    with transaction(database_path) as connection:
        # Preserve historical rows referenced by completed analyses, but make a
        # rerun authoritative for selection and prevent stale queue membership.
        # Any ML split/model is tied to the previous queue and must not remain
        # the active result after the catalogue is regenerated.
        connection.execute("DELETE FROM ml_runs")
        connection.execute("DELETE FROM ml_split_manifest")
        connection.execute("DELETE FROM ml_split")
        connection.execute("DELETE FROM catalog_optimizer_audits")
        connection.execute("DELETE FROM catalog_base_cases")
        upsert_many(
            connection,
            "catalog_base_cases",
            base_case_rows,
            ("base_case_id",),
        )
        upsert_many(
            connection,
            "catalog_optimizer_audits",
            optimizer_audit_rows,
            ("base_case_id", "member_type", "tier"),
        )
        connection.execute(
            """
            UPDATE building_catalog
            SET selected=0, queue_rank=NULL, valid=0,
                invalid_reason='superseded by current candidate library'
            """
        )
        connection.execute(
            """
            DELETE FROM building_ground_motion_selection
            WHERE building_id IN (
                SELECT building_id FROM building_catalog
                WHERE invalid_reason='superseded by current candidate library'
            )
            """
        )
        upsert_many(connection, "building_catalog", all_rows, ("building_id",))
        connection.execute(
            """
            DELETE FROM building_catalog
            WHERE invalid_reason='superseded by current candidate library'
              AND NOT EXISTS (
                  SELECT 1 FROM spo_features s
                  WHERE s.building_id=building_catalog.building_id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM ida_runs r
                  WHERE r.building_id=building_catalog.building_id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM ida_capacities c
                  WHERE c.building_id=building_catalog.building_id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM fragility_targets f
                  WHERE f.building_id=building_catalog.building_id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM ml_split m
                  WHERE m.building_id=building_catalog.building_id
              )
            """
        )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    export_path = output_dir / f"building_queue_{queue_size}.csv"
    export_columns = list(selected[0])
    with export_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=export_columns)
        writer.writeheader()
        for row in sorted(selected, key=lambda item: queue_ranks[item["building_id"]]):
            exported = dict(row)
            exported["queue_rank"] = queue_ranks[row["building_id"]]
            exported["selected"] = 1
            exported["design_metadata_json"] = json.dumps(
                exported["design_metadata_json"],
                ensure_ascii=False,
                sort_keys=True,
            )
            writer.writerow(exported)

    optimizer_by_member = {
        member_type: [
            row
            for row in optimizer_audit_rows
            if row["member_type"] == member_type
        ]
        for member_type in ("beam", "column")
    }
    optimizer_summary = {
        member_type: {
            "audit_count": len(rows),
            "multi_start_runs": sorted(
                {int(row["multi_start_runs"]) for row in rows}
            ),
            "mean_success_rate": (
                sum(
                    float(row["multi_start_success_rate"])
                    for row in rows
                )
                / len(rows)
            ),
            "audits_with_at_least_one_success": sum(
                int(row["multi_start_success_count"]) > 0
                for row in rows
            ),
            "maximum_best_gap": max(
                float(row["multi_start_best_gap"]) for row in rows
            ),
            "maximum_worst_gap": max(
                float(row["multi_start_worst_gap"]) for row in rows
            ),
        }
        for member_type, rows in optimizer_by_member.items()
    }
    return {
        "raw_combination_count": raw_combination_count,
        "unique_physical_model_count": len(all_rows),
        "raw_model_count": raw_combination_count,
        "duplicate_generation_count": (
            raw_combination_count - len(all_rows)
        ),
        "duplicate_physical_model_group_count": duplicate_group_count,
        "feasible_model_count": len(feasible),
        "invalid_model_count": len(all_rows) - len(feasible),
        "invalid_base_case_count": invalid_base_cases,
        "selected_queue_count": len(selected),
        "available_scwb_class_counts": dict(
            Counter(row["scwb_class"] for row in feasible)
        ),
        "selected_scwb_class_counts": dict(
            Counter(row["scwb_class"] for row in selected)
        ),
        "available_strength_tier_pair_counts": dict(
            sorted(
                Counter(
                    (
                        f"{float(row['beam_strength_multiplier']):.1f}"
                        "-"
                        f"{float(row['column_strength_multiplier']):.1f}"
                    )
                    for row in feasible
                ).items()
            )
        ),
        "selected_strength_tier_pair_counts": dict(
            sorted(
                Counter(
                    (
                        f"{float(row['beam_strength_multiplier']):.1f}"
                        "-"
                        f"{float(row['column_strength_multiplier']):.1f}"
                    )
                    for row in selected
                ).items()
            )
        ),
        "selected_base_case_count": selected_base_case_count,
        "feasible_base_case_count": feasible_base_case_count,
        "base_case_coverage_complete": (
            selected_base_case_count == feasible_base_case_count
        ),
        "optimizer_audit_count": len(optimizer_audit_rows),
        "optimizer_summary": optimizer_summary,
        "persistent_design_cache_loaded": persistent_cache_loaded,
        "persistent_design_cache_computed": persistent_cache_computed,
        "persistent_design_cache_invalid": persistent_cache_invalid,
        "queue_csv": str(export_path),
    }
