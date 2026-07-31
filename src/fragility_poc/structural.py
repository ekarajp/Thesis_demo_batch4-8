"""OpenSeesPy model construction and common analysis helpers.

The PoC archetype is a regular, symmetric, three-dimensional RC moment frame.
The storey count and equal X/Y bay count are supplied by the catalogue.
Units are kN, m, and s throughout this module.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import KSC_TO_KN_M2


@dataclass(frozen=True)
class ModelInfo:
    number_of_bays: int
    grid_nodes_per_floor: int
    columns_per_storey: int
    beams_per_storey: int
    base_nodes: tuple[int, ...]
    floor_nodes: tuple[tuple[int, ...], ...]
    master_nodes: tuple[int, ...]
    floor_elevations_m: tuple[float, ...]
    expected_gravity_kn: float
    total_mass_kn_s2_m: float
    column_element_tags_by_storey: tuple[tuple[int, ...], ...]
    beam_element_tags_by_storey: tuple[tuple[int, ...], ...]
    beam_integration: str
    plastic_hinge_length_method: str
    plastic_hinge_shear_span_method: str
    plastic_hinge_reference_fallback_end_count: int
    plastic_hinge_reference_clipped_end_count: int
    column_hinge_length_m: float
    beam_hinge_length_m: float
    column_hinge_lengths_m: tuple[tuple[tuple[float, float], ...], ...]
    beam_hinge_lengths_m: tuple[tuple[tuple[float, float], ...], ...]
    column_shear_spans_m: tuple[tuple[tuple[float, float], ...], ...]
    beam_shear_spans_m: tuple[tuple[tuple[float, float], ...], ...]
    column_integration_points_m: tuple[float, ...]
    column_integration_weights_m: tuple[float, ...]
    beam_integration_points_m: tuple[float, ...]
    beam_integration_weights_m: tuple[float, ...]


@dataclass(frozen=True)
class ModalResult:
    periods_s: tuple[float, ...]
    x_mode: int
    y_mode: int
    t_x_s: float
    t_y_s: float
    period_error: float
    x_mode_effective_mass_ratio: float = math.nan
    y_mode_effective_mass_ratio: float = math.nan
    cumulative_x_effective_mass_ratio: float = math.nan
    cumulative_y_effective_mass_ratio: float = math.nan
    identification_method: str = "directional_effective_modal_mass"
    eigen_solver_requested: str = "fullGenLapack"
    eigen_solver_used: str = "fullGenLapack"
    eigen_solver_fallback: bool = False


@dataclass(frozen=True)
class ManderConcreteParameters:
    """Traceable Mander (1988) parameters for a rectangular RC core.

    Stresses are stored in kN/m2 and strains are positive magnitudes in this
    record.  They are converted to OpenSees' negative compression convention
    only when the uniaxial material is created.
    """

    fco_kn_m2: float
    ec_kn_m2: float
    eps_co: float
    bc_m: float
    dc_m: float
    clear_hoop_spacing_m: float
    longitudinal_ratio_core: float
    confinement_effectiveness: float
    transverse_ratio_x: float
    transverse_ratio_y: float
    effective_lateral_x_kn_m2: float
    effective_lateral_y_kn_m2: float
    equivalent_lateral_kn_m2: float
    fcc_kn_m2: float
    eps_cc: float
    eps_cu: float


@dataclass(frozen=True)
class ElasticReferenceShearSpans:
    """End shear spans from the proposal-consistent elastic reference frame.

    Entries are ordered by storey/floor and then by column grid line or beam
    bay.  Each innermost pair contains the distances from the i and j critical
    sections to the elastic point of contraflexure, evaluated as ``abs(M/V)``.
    """

    column_end_spans_m: tuple[
        tuple[tuple[float, float], ...], ...
    ]
    beam_end_spans_m: tuple[
        tuple[tuple[float, float], ...], ...
    ]
    fallback_end_count: int
    clipped_end_count: int


@dataclass(frozen=True)
class PlasticHingeLayout:
    """Member-end shear spans and physical plastic-hinge lengths."""

    method: str
    shear_span_method: str
    column_shear_spans_m: tuple[
        tuple[tuple[float, float], ...], ...
    ]
    beam_shear_spans_m: tuple[
        tuple[tuple[float, float], ...], ...
    ]
    column_hinge_lengths_m: tuple[
        tuple[tuple[float, float], ...], ...
    ]
    beam_hinge_lengths_m: tuple[
        tuple[tuple[float, float], ...], ...
    ]
    reference_fallback_end_count: int
    reference_clipped_end_count: int


def aci_required_clear_spacing_m(
    *,
    bar_diameter_m: float,
    nominal_maximum_aggregate_size_m: float,
    code_minimum_m: float = 0.025,
) -> float:
    """Return ACI aggregate-aware clear spacing between bar surfaces.

    ACI 318 requires at least the greater of 25 mm and the bar diameter for
    parallel nonprestressed bars.  The nominal maximum coarse aggregate size
    is also limited to three-quarters of the specified clear spacing, which
    gives the equivalent spacing lower bound ``4/3 * aggregate size``.
    """
    values = (
        float(bar_diameter_m),
        float(nominal_maximum_aggregate_size_m),
        float(code_minimum_m),
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("ACI clear-spacing inputs must be positive")
    return max(
        float(code_minimum_m),
        float(bar_diameter_m),
        4.0 * float(nominal_maximum_aggregate_size_m) / 3.0,
    )


def plastic_hinge_elastic_interior_length_m(
    *,
    member_length_m: float,
    hinge_length_i_m: float,
    hinge_length_j_m: float,
    integration: str,
) -> float:
    """Return the integration rule's nominal elastic-interior length.

    ``HingeRadau`` applies its two-point Radau rule over four times each
    user-supplied hinge length.  The other supported plastic-hinge rules apply
    their quadrature over the supplied hinge lengths themselves.  Keeping this
    rule explicit prevents zero or negative integration weights in short RC
    members.
    """
    length = float(member_length_m)
    hinge_i = float(hinge_length_i_m)
    hinge_j = float(hinge_length_j_m)
    if (
        not math.isfinite(length)
        or not math.isfinite(hinge_i)
        or not math.isfinite(hinge_j)
        or length <= 0.0
        or hinge_i <= 0.0
        or hinge_j <= 0.0
    ):
        raise ValueError("Member and plastic-hinge lengths must be positive")
    supported = {
        "HingeMidpoint": 1.0,
        "HingeEndpoint": 1.0,
        "HingeRadauTwo": 1.0,
        "HingeRadau": 4.0,
    }
    if integration not in supported:
        raise ValueError(
            "beam_integration must be one of "
            + ", ".join(sorted(supported))
        )
    return length - supported[integration] * (hinge_i + hinge_j)


def _validate_element_integration(
    ops: Any,
    *,
    element_tag: int,
    member_length_m: float,
    member_label: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Verify the integration object actually created by OpenSees."""
    points = tuple(
        float(value)
        for value in ops.eleResponse(element_tag, "integrationPoints")
    )
    weights = tuple(
        float(value)
        for value in ops.eleResponse(element_tag, "integrationWeights")
    )
    if not points or len(points) != len(weights):
        raise ValueError(
            f"{member_label}: OpenSees returned incomplete integration data"
        )
    tolerance = max(1.0e-12, 1.0e-10 * float(member_length_m))
    if any(
        not math.isfinite(value)
        or value < -tolerance
        or value > float(member_length_m) + tolerance
        for value in points
    ):
        raise ValueError(
            f"{member_label}: integration point lies outside the member"
        )
    if any(not math.isfinite(value) or value <= tolerance for value in weights):
        raise ValueError(
            f"{member_label}: zero/negative OpenSees integration weight"
        )
    if not math.isclose(
        sum(weights),
        float(member_length_m),
        rel_tol=1.0e-9,
        abs_tol=tolerance,
    ):
        raise ValueError(
            f"{member_label}: integration weights do not sum to member length"
        )
    return points, weights


def hognestad_concrete_parameters(
    fc_kn_m2: float,
    *,
    peak_strain: float = 0.0020,
    ultimate_strain: float = 0.0038,
    ultimate_strength_ratio: float = 0.85,
) -> tuple[float, float, float, float]:
    """Return OpenSees Concrete01 inputs for the proposal's Hognestad curve.

    Compression is negative in OpenSees.  Equations (3.7)-(3.8) and Figure
    3.4 of Draft 4 define a parabolic rise to ``f'c`` at ``eps_0`` followed
    by a linear drop of 0.15 ``f'c`` at ``eps_cu``.  Therefore the terminal
    stress ratio is 0.85.  The sentence below Equation (3.8) says 0.75, but
    that conflicts with both the equation and its figure.
    """
    if fc_kn_m2 <= 0.0:
        raise ValueError("Concrete compressive strength must be positive")
    if not 0.0 < peak_strain < ultimate_strain:
        raise ValueError(
            "Hognestad strains must satisfy 0 < peak < ultimate"
        )
    if not 0.0 < ultimate_strength_ratio <= 1.0:
        raise ValueError(
            "Hognestad ultimate strength ratio must be in (0, 1]"
        )
    return (
        -float(fc_kn_m2),
        -float(peak_strain),
        -float(ultimate_strength_ratio) * float(fc_kn_m2),
        -float(ultimate_strain),
    )


def _perimeter_clear_spacing_square_sum(
    bar_positions: tuple[tuple[float, float], ...],
    *,
    bar_diameter_m: float,
) -> float:
    """Return ``sum(w_i'^2)`` around a rectangular longitudinal-bar layout.

    Mander et al. (1988), Eq. (20), defines ``w_i'`` as the clear distance
    between adjacent longitudinal bars along each face of the hoop.  Corner
    bars participate in two faces, while every perimeter interval is counted
    exactly once.
    """
    if len(bar_positions) < 4:
        raise ValueError("At least four perimeter bars are required")
    if bar_diameter_m <= 0.0:
        raise ValueError("Longitudinal bar diameter must be positive")
    y_values = [float(point[0]) for point in bar_positions]
    z_values = [float(point[1]) for point in bar_positions]
    y_min, y_max = min(y_values), max(y_values)
    z_min, z_max = min(z_values), max(z_values)
    tolerance = max(abs(y_max - y_min), abs(z_max - z_min), 1.0) * 1.0e-9

    face_coordinates = (
        sorted(
            y for y, z in bar_positions if abs(float(z) - z_max) <= tolerance
        ),
        sorted(
            y for y, z in bar_positions if abs(float(z) - z_min) <= tolerance
        ),
        sorted(
            z for y, z in bar_positions if abs(float(y) - y_min) <= tolerance
        ),
        sorted(
            z for y, z in bar_positions if abs(float(y) - y_max) <= tolerance
        ),
    )
    square_sum = 0.0
    interval_count = 0
    for coordinates in face_coordinates:
        if len(coordinates) < 2:
            raise ValueError(
                "Each confined-core face must contain at least two bars"
            )
        for first, second in zip(coordinates[:-1], coordinates[1:]):
            clear_distance = float(second - first) - float(bar_diameter_m)
            if clear_distance <= 0.0:
                raise ValueError(
                    "Longitudinal bars overlap in the confinement calculation"
                )
            square_sum += clear_distance**2
            interval_count += 1
    if interval_count != len(bar_positions):
        raise ValueError(
            "Longitudinal bars must lie on a complete rectangular perimeter"
        )
    return square_sum


def mander_rectangular_confined_parameters(
    fco_kn_m2: float,
    *,
    width_m: float,
    depth_m: float,
    clear_cover_m: float,
    hoop_diameter_m: float,
    hoop_spacing_m: float,
    transverse_legs_x: int,
    transverse_legs_y: int,
    transverse_fy_kn_m2: float,
    longitudinal_bar_diameter_m: float,
    longitudinal_bar_area_m2: float,
    bar_positions: tuple[tuple[float, float], ...],
    longitudinal_area_total_m2: float | None = None,
    eps_co: float = 0.0020,
    transverse_ultimate_strain: float = 0.12,
    minimum_ultimate_strain: float = 0.0038,
    maximum_ultimate_strain: float = 0.030,
) -> ManderConcreteParameters:
    """Calculate a conservative rectangular Mander confined-concrete law.

    Equations (5), (20)-(29) of Mander, Priestley, and Park (1988) are used.
    Their closed-form confined-strength equation is for equal biaxial lateral
    pressure.  For a rectangular section with unequal pressures, the smaller
    effective pressure is used here as an explicitly conservative equivalent;
    the two directional pressures are retained for reporting.

    The concrete failure strain uses the widely adopted energy-based
    approximation ``0.004 + 1.4*rho_s*fyh*eps_su/fcc`` and is bounded by the
    configured limits.  In the model it is a true failure limit, not merely
    the last point of a stress-strain curve.
    """
    positive_values = {
        "fco_kn_m2": fco_kn_m2,
        "width_m": width_m,
        "depth_m": depth_m,
        "clear_cover_m": clear_cover_m,
        "hoop_diameter_m": hoop_diameter_m,
        "hoop_spacing_m": hoop_spacing_m,
        "transverse_fy_kn_m2": transverse_fy_kn_m2,
        "longitudinal_bar_diameter_m": longitudinal_bar_diameter_m,
        "longitudinal_bar_area_m2": longitudinal_bar_area_m2,
        "eps_co": eps_co,
        "transverse_ultimate_strain": transverse_ultimate_strain,
        "minimum_ultimate_strain": minimum_ultimate_strain,
        "maximum_ultimate_strain": maximum_ultimate_strain,
    }
    for name, value in positive_values.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if transverse_legs_x < 2 or transverse_legs_y < 2:
        raise ValueError("At least two transverse legs are required per axis")
    if maximum_ultimate_strain < minimum_ultimate_strain:
        raise ValueError(
            "maximum_ultimate_strain must not be below the minimum"
        )

    bc = float(width_m) - 2.0 * (
        float(clear_cover_m) + 0.5 * float(hoop_diameter_m)
    )
    dc = float(depth_m) - 2.0 * (
        float(clear_cover_m) + 0.5 * float(hoop_diameter_m)
    )
    clear_spacing = float(hoop_spacing_m) - float(hoop_diameter_m)
    if bc <= 0.0 or dc <= 0.0 or clear_spacing <= 0.0:
        raise ValueError("Hoop geometry leaves no confined concrete core")

    longitudinal_area = (
        len(bar_positions) * float(longitudinal_bar_area_m2)
        if longitudinal_area_total_m2 is None
        else float(longitudinal_area_total_m2)
    )
    if not math.isfinite(longitudinal_area) or longitudinal_area <= 0.0:
        raise ValueError(
            "Total longitudinal reinforcement area must be positive and finite"
        )
    rho_cc = longitudinal_area / (bc * dc)
    if not 0.0 < rho_cc < 1.0:
        raise ValueError("Confined-core longitudinal ratio must be in (0, 1)")
    spacing_square_sum = _perimeter_clear_spacing_square_sum(
        bar_positions,
        bar_diameter_m=float(longitudinal_bar_diameter_m),
    )
    plan_effectiveness = 1.0 - spacing_square_sum / (6.0 * bc * dc)
    vertical_effectiveness = (
        (1.0 - clear_spacing / (2.0 * bc))
        * (1.0 - clear_spacing / (2.0 * dc))
    )
    ke = plan_effectiveness * vertical_effectiveness / (1.0 - rho_cc)
    ke = min(max(float(ke), 0.0), 1.0)
    if ke <= 0.0:
        raise ValueError("Calculated confinement effectiveness is nonpositive")

    hoop_area = math.pi * float(hoop_diameter_m) ** 2 / 4.0
    rho_x = (
        float(transverse_legs_x)
        * hoop_area
        / (float(hoop_spacing_m) * dc)
    )
    rho_y = (
        float(transverse_legs_y)
        * hoop_area
        / (float(hoop_spacing_m) * bc)
    )
    fl_x = ke * rho_x * float(transverse_fy_kn_m2)
    fl_y = ke * rho_y * float(transverse_fy_kn_m2)
    fl_equivalent = min(fl_x, fl_y)
    pressure_ratio = fl_equivalent / float(fco_kn_m2)
    strength_ratio = (
        -1.254
        + 2.254 * math.sqrt(1.0 + 7.94 * pressure_ratio)
        - 2.0 * pressure_ratio
    )
    fcc = max(float(fco_kn_m2), strength_ratio * float(fco_kn_m2))
    eps_cc = float(eps_co) * (
        1.0 + 5.0 * (fcc / float(fco_kn_m2) - 1.0)
    )
    # Use the effective transverse ratio in the ultimate-strain estimate.
    # Applying the nominal hoop ratio without ``ke`` would overstate ductility
    # in rectangular members whose intermediate longitudinal bars are poorly
    # restrained (particularly the beam layouts in this PoC).
    rho_s_effective = ke * (rho_x + rho_y)
    eps_cu_unbounded = (
        0.004
        + 1.4
        * rho_s_effective
        * float(transverse_fy_kn_m2)
        * float(transverse_ultimate_strain)
        / fcc
    )
    eps_cu = min(
        max(
            eps_cu_unbounded,
            float(minimum_ultimate_strain),
            1.05 * eps_cc,
        ),
        float(maximum_ultimate_strain),
    )
    # Mander et al. Eq. (7), with fco in MPa and Ec returned in kN/m2.
    ec_kn_m2 = 5_000.0 * math.sqrt(float(fco_kn_m2) / 1_000.0) * 1_000.0
    return ManderConcreteParameters(
        fco_kn_m2=float(fco_kn_m2),
        ec_kn_m2=ec_kn_m2,
        eps_co=float(eps_co),
        bc_m=bc,
        dc_m=dc,
        clear_hoop_spacing_m=clear_spacing,
        longitudinal_ratio_core=rho_cc,
        confinement_effectiveness=ke,
        transverse_ratio_x=rho_x,
        transverse_ratio_y=rho_y,
        effective_lateral_x_kn_m2=fl_x,
        effective_lateral_y_kn_m2=fl_y,
        equivalent_lateral_kn_m2=fl_equivalent,
        fcc_kn_m2=fcc,
        eps_cc=eps_cc,
        eps_cu=eps_cu,
    )


def hysteretic_steel_backbone_points(
    fy_kn_m2: float,
    *,
    elastic_modulus_kn_m2: float = 200_000_000.0,
    peak_strength_ratio: float = 1.05,
    peak_strain: float = 0.015,
    residual_strength_ratio: float = 0.20,
    residual_strain: float = 0.080,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return the three positive-envelope points of the steel model.

    The model restores the original PoC reinforcement representation: an
    elastic branch to yield, a short hardening branch, and a descending
    post-peak branch.  The third point is a residual-strength point, not an
    automatic fracture limit.
    """
    if fy_kn_m2 <= 0.0 or elastic_modulus_kn_m2 <= 0.0:
        raise ValueError("Steel strength and elastic modulus must be positive")
    if peak_strength_ratio <= 1.0:
        raise ValueError("Steel peak strength ratio must exceed 1.0")
    if not 0.0 <= residual_strength_ratio < peak_strength_ratio:
        raise ValueError(
            "Steel residual strength ratio must be nonnegative and below peak"
        )
    yield_strain = float(fy_kn_m2) / float(elastic_modulus_kn_m2)
    if not yield_strain < peak_strain < residual_strain:
        raise ValueError(
            "Steel strains must satisfy yield < peak < residual point"
        )
    return (
        (yield_strain, float(fy_kn_m2)),
        (float(peak_strain), peak_strength_ratio * float(fy_kn_m2)),
        (
            float(residual_strain),
            residual_strength_ratio * float(fy_kn_m2),
        ),
    )


def define_hysteretic_steel_material(
    ops: Any,
    *,
    material_tag: int,
    fy_kn_m2: float,
    model_config: dict[str, Any],
    base_material_tag: int | None = None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Create the stable cyclic trilinear steel law used by the PoC.

    The envelope retains an explicit post-peak branch.  When
    ``base_material_tag`` is supplied, the Hysteretic material is wrapped in
    MinMax so the configured fracture strain is a true material failure limit.
    Pinching defaults to unity because bond-slip is not represented inside a
    reinforcing-steel fibre.
    """
    steel_config = model_config.get("steel_hysteretic", {})
    elastic_modulus = (
        float(steel_config.get("elastic_modulus_mpa", 200_000.0)) * 1000.0
    )
    points = hysteretic_steel_backbone_points(
        fy_kn_m2,
        elastic_modulus_kn_m2=elastic_modulus,
        peak_strength_ratio=float(
            steel_config.get("peak_strength_ratio", 1.05)
        ),
        peak_strain=float(steel_config.get("peak_strain", 0.015)),
        residual_strength_ratio=float(
            steel_config.get("residual_strength_ratio", 0.20)
        ),
        residual_strain=float(
            steel_config.get("residual_strain", 0.080)
        ),
    )
    pinch_x = float(steel_config.get("pinch_x", 1.0))
    pinch_y = float(steel_config.get("pinch_y", 1.0))
    damage_ductility = float(
        steel_config.get("damage_ductility", 0.0)
    )
    damage_energy = float(steel_config.get("damage_energy", 0.0))
    unloading_beta = float(
        steel_config.get("unloading_stiffness_beta", 0.0)
    )
    if not 0.0 <= pinch_x <= 1.0 or not 0.0 <= pinch_y <= 1.0:
        raise ValueError("Steel pinching factors must be within [0, 1]")
    if damage_ductility < 0.0 or damage_energy < 0.0:
        raise ValueError("Steel damage factors must be nonnegative")
    if unloading_beta < 0.0:
        raise ValueError("Steel unloading-stiffness beta must be nonnegative")
    (e1, s1), (e2, s2), (e3, s3) = points
    hysteretic_tag = (
        int(base_material_tag)
        if base_material_tag is not None
        else int(material_tag)
    )
    ops.uniaxialMaterial(
        "Hysteretic",
        hysteretic_tag,
        s1,
        e1,
        s2,
        e2,
        s3,
        e3,
        -s1,
        -e1,
        -s2,
        -e2,
        -s3,
        -e3,
        pinch_x,
        pinch_y,
        damage_ductility,
        damage_energy,
        unloading_beta,
    )
    if base_material_tag is not None:
        failure_strain = float(steel_config.get("failure_strain", 0.10))
        if failure_strain <= e3:
            raise ValueError(
                "Steel failure strain must exceed the residual-point strain"
            )
        ops.uniaxialMaterial(
            "MinMax",
            material_tag,
            hysteretic_tag,
            "-min",
            -failure_strain,
            "-max",
            failure_strain,
        )
    return points


def _planar_frame_element_stiffness(
    *,
    elastic_modulus: float,
    area_m2: float,
    inertia_m4: float,
    length_m: float,
    cosine: float,
    sine: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local stiffness and global-to-local transformation matrices."""
    if min(
        float(elastic_modulus),
        float(area_m2),
        float(inertia_m4),
        float(length_m),
    ) <= 0.0:
        raise ValueError("Elastic reference-member properties must be positive")
    ea_l = elastic_modulus * area_m2 / length_m
    ei = elastic_modulus * inertia_m4
    k_local = np.array(
        [
            [ea_l, 0.0, 0.0, -ea_l, 0.0, 0.0],
            [
                0.0,
                12.0 * ei / length_m**3,
                6.0 * ei / length_m**2,
                0.0,
                -12.0 * ei / length_m**3,
                6.0 * ei / length_m**2,
            ],
            [
                0.0,
                6.0 * ei / length_m**2,
                4.0 * ei / length_m,
                0.0,
                -6.0 * ei / length_m**2,
                2.0 * ei / length_m,
            ],
            [-ea_l, 0.0, 0.0, ea_l, 0.0, 0.0],
            [
                0.0,
                -12.0 * ei / length_m**3,
                -6.0 * ei / length_m**2,
                0.0,
                12.0 * ei / length_m**3,
                -6.0 * ei / length_m**2,
            ],
            [
                0.0,
                6.0 * ei / length_m**2,
                2.0 * ei / length_m,
                0.0,
                -6.0 * ei / length_m**2,
                4.0 * ei / length_m,
            ],
        ],
        dtype=float,
    )
    c = float(cosine)
    s = float(sine)
    transformation = np.array(
        [
            [c, s, 0.0, 0.0, 0.0, 0.0],
            [-s, c, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c, s, 0.0],
            [0.0, 0.0, 0.0, -s, c, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return k_local, transformation


def elastic_reference_shear_spans(
    *,
    stories: int,
    number_of_bays: int,
    story_height_m: float,
    bay_width_m: float,
    beam_width_m: float,
    beam_depth_m: float,
    column_width_m: float,
    column_depth_m: float,
    minimum_span_ratio: float = 0.05,
    maximum_span_ratio: float = 1.00,
) -> ElasticReferenceShearSpans:
    """Compute proposal-defined contraflexure distances as ``abs(M/V)``.

    A deterministic, linear-elastic 2D reference frame is subjected to the
    same inverted-triangular lateral-load shape used by SPO.  Floor horizontal
    translations are tied to represent the rigid diaphragm; vertical
    translations and joint rotations remain free.  Gravity fixed-end moments
    are deliberately excluded because ``L`` in the Priestley expression is
    the lateral-response shear span at the member critical section.

    Very small force ratios are numerical, not physical, information.  Such
    ends fall back to half the member length.  All spans are bounded to the
    configured fraction of the member centreline length; clipping is reported
    so it cannot silently contaminate the research database.
    """
    if stories < 1 or number_of_bays < 1:
        raise ValueError("Elastic reference frame needs positive stories/bays")
    dimensions = (
        story_height_m,
        bay_width_m,
        beam_width_m,
        beam_depth_m,
        column_width_m,
        column_depth_m,
    )
    if any(not math.isfinite(float(value)) or value <= 0.0 for value in dimensions):
        raise ValueError("Elastic reference-frame dimensions must be positive")
    if (
        not 0.0 < float(minimum_span_ratio)
        <= float(maximum_span_ratio)
        <= 1.0
    ):
        raise ValueError(
            "Contraflexure span ratios must satisfy 0 < minimum <= maximum <= 1"
        )

    grid_size = number_of_bays + 1
    node_count = (stories + 1) * grid_size
    full_dof_count = 3 * node_count

    def node_index(level: int, grid_index: int) -> int:
        return level * grid_size + grid_index

    def node_dofs(level: int, grid_index: int) -> tuple[int, int, int]:
        first = 3 * node_index(level, grid_index)
        return first, first + 1, first + 2

    stiffness = np.zeros((full_dof_count, full_dof_count), dtype=float)
    elements: list[
        tuple[str, int, int, tuple[int, ...], np.ndarray, np.ndarray, float]
    ] = []
    elastic_modulus = 1.0
    column_area = column_width_m * column_depth_m
    column_inertia = column_width_m * column_depth_m**3 / 12.0
    beam_area = beam_width_m * beam_depth_m
    beam_inertia = beam_width_m * beam_depth_m**3 / 12.0

    def add_element(
        *,
        member_type: str,
        level_index: int,
        position_index: int,
        start: tuple[int, int],
        end: tuple[int, int],
        length_m: float,
        cosine: float,
        sine: float,
        area_m2: float,
        inertia_m4: float,
    ) -> None:
        dofs = (*node_dofs(*start), *node_dofs(*end))
        k_local, transformation = _planar_frame_element_stiffness(
            elastic_modulus=elastic_modulus,
            area_m2=area_m2,
            inertia_m4=inertia_m4,
            length_m=length_m,
            cosine=cosine,
            sine=sine,
        )
        k_global = transformation.T @ k_local @ transformation
        stiffness[np.ix_(dofs, dofs)] += k_global
        elements.append(
            (
                member_type,
                level_index,
                position_index,
                tuple(dofs),
                k_local,
                transformation,
                float(length_m),
            )
        )

    for level in range(1, stories + 1):
        for grid_index in range(grid_size):
            add_element(
                member_type="column",
                level_index=level - 1,
                position_index=grid_index,
                start=(level - 1, grid_index),
                end=(level, grid_index),
                length_m=story_height_m,
                cosine=0.0,
                sine=1.0,
                area_m2=column_area,
                inertia_m4=column_inertia,
            )
        for bay_index in range(number_of_bays):
            add_element(
                member_type="beam",
                level_index=level - 1,
                position_index=bay_index,
                start=(level, bay_index),
                end=(level, bay_index + 1),
                length_m=bay_width_m,
                cosine=1.0,
                sine=0.0,
                area_m2=beam_area,
                inertia_m4=beam_inertia,
            )

    # Transformation from independent generalized coordinates to all nodal
    # DOFs.  Every floor has one diaphragm horizontal translation; vertical
    # translations and rotations remain independent at each joint.
    reduced_indices: dict[tuple[str, int, int], int] = {}

    def reduced_index(key: tuple[str, int, int]) -> int:
        if key not in reduced_indices:
            reduced_indices[key] = len(reduced_indices)
        return reduced_indices[key]

    full_to_reduced: list[int | None] = [None] * full_dof_count
    for level in range(1, stories + 1):
        floor_u = reduced_index(("u", level, 0))
        for grid_index in range(grid_size):
            u_dof, v_dof, rotation_dof = node_dofs(level, grid_index)
            full_to_reduced[u_dof] = floor_u
            full_to_reduced[v_dof] = reduced_index(
                ("v", level, grid_index)
            )
            full_to_reduced[rotation_dof] = reduced_index(
                ("r", level, grid_index)
            )
    transformation = np.zeros(
        (full_dof_count, len(reduced_indices)),
        dtype=float,
    )
    for full_index, reduced in enumerate(full_to_reduced):
        if reduced is not None:
            transformation[full_index, reduced] = 1.0
    reduced_stiffness = transformation.T @ stiffness @ transformation
    reduced_load = np.zeros(len(reduced_indices), dtype=float)
    for level in range(1, stories + 1):
        reduced_load[reduced_indices[("u", level, 0)]] = float(level)
    try:
        reduced_displacement = np.linalg.solve(
            reduced_stiffness,
            reduced_load,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Elastic contraflexure reference frame is singular"
        ) from exc
    full_displacement = transformation @ reduced_displacement

    column_spans: list[list[tuple[float, float] | None]] = [
        [None] * grid_size for _ in range(stories)
    ]
    beam_spans: list[list[tuple[float, float] | None]] = [
        [None] * number_of_bays for _ in range(stories)
    ]
    fallback_count = 0
    clipped_count = 0
    force_scale = max(
        1.0,
        float(np.max(np.abs(reduced_load))),
    )

    def end_span(moment: float, shear: float, member_length: float) -> float:
        nonlocal fallback_count, clipped_count
        shear_tolerance = 1.0e-10 * force_scale
        if (
            not math.isfinite(moment)
            or not math.isfinite(shear)
            or abs(shear) <= shear_tolerance
        ):
            fallback_count += 1
            raw = 0.5 * member_length
        else:
            raw = abs(moment / shear)
            if not math.isfinite(raw) or raw <= 0.0:
                fallback_count += 1
                raw = 0.5 * member_length
        lower = float(minimum_span_ratio) * member_length
        upper = float(maximum_span_ratio) * member_length
        bounded = min(max(raw, lower), upper)
        if not math.isclose(bounded, raw, rel_tol=0.0, abs_tol=1.0e-10):
            clipped_count += 1
        return bounded

    for (
        member_type,
        level_index,
        position_index,
        dofs,
        k_local,
        element_transformation,
        member_length,
    ) in elements:
        local_displacement = (
            element_transformation @ full_displacement[list(dofs)]
        )
        local_force = k_local @ local_displacement
        spans = (
            end_span(local_force[2], local_force[1], member_length),
            end_span(local_force[5], local_force[4], member_length),
        )
        target = column_spans if member_type == "column" else beam_spans
        target[level_index][position_index] = spans

    if any(value is None for level in column_spans for value in level):
        raise AssertionError("Incomplete elastic-reference column spans")
    if any(value is None for level in beam_spans for value in level):
        raise AssertionError("Incomplete elastic-reference beam spans")
    return ElasticReferenceShearSpans(
        column_end_spans_m=tuple(
            tuple(value for value in level if value is not None)
            for level in column_spans
        ),
        beam_end_spans_m=tuple(
            tuple(value for value in level if value is not None)
            for level in beam_spans
        ),
        fallback_end_count=fallback_count,
        clipped_end_count=clipped_count,
    )


def plastic_hinge_length(
    section_depth_m: float,
    *,
    method: str = "section_depth_factor",
    factor: float = 0.50,
    shear_span_m: float | None = None,
    member_length_m: float | None = None,
    longitudinal_bar_diameter_m: float | None = None,
    steel_yield_strength_mpa: float | None = None,
    expected_strength_factor: float = 1.25,
) -> float:
    """Return the configured finite RC plastic-hinge length.

    ``section_depth_factor`` is retained only for the explicit parametric
    sensitivity study.  The production method uses the Priestley expression
    with expected steel strength:

    ``Lp=max(0.08Ls + 0.022*fye*db, 0.044*fye*db)``.

    In accordance with Draft 5, ``Ls`` is the distance from the member-end
    critical section to the point of contraflexure, not automatically the
    complete bay width or storey height.  ``member_length_m`` is accepted only
    as a backwards-compatible alias for isolated legacy calculations.
    """
    if section_depth_m <= 0.0:
        raise ValueError("Section depth must be positive")
    if factor <= 0.0:
        raise ValueError("Plastic-hinge length factor must be positive")
    if method in {
        "section_depth_factor",
        "0.5_section_depth",
        "parametric_section_depth_sensitivity",
    }:
        return float(factor) * float(section_depth_m)
    if method not in {
        "priestley_1996_contraflexure_expected_strength",
        "priestley_1996_contraflexure_expected_strength_scaled",
        "priestley_1992_expected_strength",
        "priestley_1992_expected_strength_scaled",
    }:
        raise ValueError(f"Unsupported plastic-hinge length method: {method}")
    resolved_shear_span = (
        shear_span_m if shear_span_m is not None else member_length_m
    )
    inputs = {
        "shear_span_m": resolved_shear_span,
        "longitudinal_bar_diameter_m": longitudinal_bar_diameter_m,
        "steel_yield_strength_mpa": steel_yield_strength_mpa,
    }
    if any(
        value is None
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in inputs.values()
    ):
        raise ValueError(
            "Priestley plastic-hinge length requires positive finite member "
            "end shear span, longitudinal-bar diameter, and steel yield "
            "strength"
        )
    if (
        not math.isfinite(expected_strength_factor)
        or expected_strength_factor <= 0.0
    ):
        raise ValueError(
            "Expected steel-strength factor must be positive and finite"
        )
    expected_fy_mpa = (
        float(steel_yield_strength_mpa) * float(expected_strength_factor)
    )
    bar_term_m = (
        expected_fy_mpa * float(longitudinal_bar_diameter_m)
    )
    empirical_length_m = max(
        0.08 * float(resolved_shear_span) + 0.022 * bar_term_m,
        0.044 * bar_term_m,
    )
    return float(factor) * empirical_length_m


def plastic_hinge_layout(
    building: dict[str, Any],
    config: dict[str, Any],
) -> PlasticHingeLayout:
    """Return proposal-consistent end-specific ``Ls`` and ``Lp`` values.

    Beam spans are taken directly from the corresponding member in the
    representative X-direction elastic frame.  A 3D column participates in
    both orthogonal frames; because the archetype is square and symmetric, its
    single physical integration length uses the geometric mean of the X- and
    Y-direction end shear spans.  This preserves X/Y symmetry without
    selecting an arbitrary governing component.
    """
    stories = int(building["stories"])
    number_of_bays = int(building.get("number_of_bays", 1))
    story_height = float(building["story_height_m"])
    bay_width = float(building["bay_width_m"])
    beam_width = float(building["beam_b_m"])
    beam_depth = float(building["beam_h_m"])
    column_width = float(building["column_b_m"])
    column_depth = float(building["column_h_m"])
    model_config = config["model"]
    method = str(
        model_config.get(
            "plastic_hinge_length_method",
            "priestley_1996_contraflexure_expected_strength",
        )
    )
    shear_span_method = str(
        model_config.get(
            "plastic_hinge_shear_span_method",
            "elastic_reference_m_over_v",
        )
    )
    scale = float(
        model_config.get(
            "plastic_hinge_length_scale_factor",
            model_config.get("plastic_hinge_length_factor", 1.0),
        )
    )
    expected_strength_factor = float(
        model_config.get("plastic_hinge_expected_strength_factor", 1.25)
    )
    fy_mpa = (
        float(
            building.get(
                "steel_fy_ksc",
                config["building"]["steel_fy_ksc"],
            )
        )
        * KSC_TO_KN_M2
        / 1000.0
    )
    metadata_raw = building.get("design_metadata_json", "{}")
    if isinstance(metadata_raw, str):
        design_metadata = json.loads(metadata_raw or "{}")
    elif isinstance(metadata_raw, dict):
        design_metadata = dict(metadata_raw)
    else:
        raise ValueError("design_metadata_json must be JSON text or a mapping")
    beam_bar_diameter = max(
        float(
            design_metadata.get(
                "beam_top_bar_diameter_m",
                building["beam_bar_diameter_m"],
            )
        ),
        float(
            design_metadata.get(
                "beam_bottom_bar_diameter_m",
                building["beam_bar_diameter_m"],
            )
        ),
        float(
            design_metadata.get(
                "beam_side_bar_diameter_m",
                building["beam_bar_diameter_m"],
            )
        ),
    )
    column_bar_diameter = float(building["column_bar_diameter_m"])

    if shear_span_method == "elastic_reference_m_over_v":
        reference = elastic_reference_shear_spans(
            stories=stories,
            number_of_bays=number_of_bays,
            story_height_m=story_height,
            bay_width_m=bay_width,
            beam_width_m=beam_width,
            beam_depth_m=beam_depth,
            column_width_m=column_width,
            column_depth_m=column_depth,
            minimum_span_ratio=float(
                model_config.get(
                    "plastic_hinge_minimum_shear_span_ratio",
                    0.05,
                )
            ),
            maximum_span_ratio=float(
                model_config.get(
                    "plastic_hinge_maximum_shear_span_ratio",
                    1.0,
                )
            ),
        )
    elif shear_span_method == "legacy_full_member":
        reference = ElasticReferenceShearSpans(
            column_end_spans_m=tuple(
                tuple(
                    (story_height, story_height)
                    for _ in range(number_of_bays + 1)
                )
                for _ in range(stories)
            ),
            beam_end_spans_m=tuple(
                tuple(
                    (bay_width, bay_width)
                    for _ in range(number_of_bays)
                )
                for _ in range(stories)
            ),
            fallback_end_count=0,
            clipped_end_count=0,
        )
    else:
        raise ValueError(
            "plastic_hinge_shear_span_method must be "
            "elastic_reference_m_over_v or legacy_full_member"
        )

    grid_size = number_of_bays + 1
    column_spans_3d: list[tuple[tuple[float, float], ...]] = []
    beam_spans_3d: list[tuple[tuple[float, float], ...]] = []
    for level in range(stories):
        column_level: list[tuple[float, float]] = []
        for ix in range(grid_size):
            for iy in range(grid_size):
                x_spans = reference.column_end_spans_m[level][ix]
                y_spans = reference.column_end_spans_m[level][iy]
                column_level.append(
                    (
                        math.sqrt(x_spans[0] * y_spans[0]),
                        math.sqrt(x_spans[1] * y_spans[1]),
                    )
                )
        column_spans_3d.append(tuple(column_level))

        beam_level: list[tuple[float, float]] = []
        for _iy in range(grid_size):
            for ix in range(number_of_bays):
                beam_level.append(
                    reference.beam_end_spans_m[level][ix]
                )
        for _ix in range(grid_size):
            for iy in range(number_of_bays):
                beam_level.append(
                    reference.beam_end_spans_m[level][iy]
                )
        beam_spans_3d.append(tuple(beam_level))

    def hinge_pair(
        spans: tuple[float, float],
        *,
        section_depth: float,
        bar_diameter: float,
    ) -> tuple[float, float]:
        return tuple(
            plastic_hinge_length(
                section_depth,
                method=method,
                factor=scale,
                shear_span_m=span,
                longitudinal_bar_diameter_m=bar_diameter,
                steel_yield_strength_mpa=fy_mpa,
                expected_strength_factor=expected_strength_factor,
            )
            for span in spans
        )  # type: ignore[return-value]

    column_hinges = tuple(
        tuple(
            hinge_pair(
                spans,
                section_depth=column_depth,
                bar_diameter=column_bar_diameter,
            )
            for spans in level
        )
        for level in column_spans_3d
    )
    beam_hinges = tuple(
        tuple(
            hinge_pair(
                spans,
                section_depth=beam_depth,
                bar_diameter=beam_bar_diameter,
            )
            for spans in level
        )
        for level in beam_spans_3d
    )
    return PlasticHingeLayout(
        method=method,
        shear_span_method=shear_span_method,
        column_shear_spans_m=tuple(column_spans_3d),
        beam_shear_spans_m=tuple(beam_spans_3d),
        column_hinge_lengths_m=column_hinges,
        beam_hinge_lengths_m=beam_hinges,
        reference_fallback_end_count=reference.fallback_end_count,
        reference_clipped_end_count=reference.clipped_end_count,
    )


def _elastic_rectangular_section(
    ops: Any,
    *,
    section_tag: int,
    elastic_modulus_kn_m2: float,
    shear_modulus_kn_m2: float,
    width_m: float,
    depth_m: float,
) -> None:
    """Create the linear-elastic interior section for hinge integration."""
    area = width_m * depth_m
    # The fiber-section coordinates use y across the width and z across the
    # depth, so Iz = integral(y^2)dA and Iy = integral(z^2)dA.
    iz = depth_m * width_m**3 / 12.0
    iy = width_m * depth_m**3 / 12.0
    torsional_constant = rectangular_torsional_constant(width_m, depth_m)
    ops.section(
        "Elastic",
        section_tag,
        elastic_modulus_kn_m2,
        area,
        iz,
        iy,
        shear_modulus_kn_m2,
        torsional_constant,
    )


def _ops() -> Any:
    try:
        import openseespy.opensees as ops
    except ImportError as exc:  # pragma: no cover - exercised in clean installs
        raise RuntimeError(
            "OpenSeesPy is not installed. Install the locked project environment "
            "with: python -m pip install -e .[test]"
        ) from exc
    return ops


def _rectangle_patch(
    ops: Any,
    material: int,
    ny: int,
    nz: int,
    y1: float,
    z1: float,
    y2: float,
    z2: float,
) -> None:
    if y2 > y1 and z2 > z1:
        ops.patch("rect", material, ny, nz, y1, z1, y2, z2)


def rectangular_torsional_constant(width_m: float, depth_m: float) -> float:
    """Return the Saint-Venant torsional constant of a solid rectangle.

    The approximation is symmetric in the two section dimensions and is
    accurate for the aspect ratios used by the PoC.  A bending inertia is not
    an appropriate substitute for ``J`` in a 3D frame model.
    """
    if width_m <= 0.0 or depth_m <= 0.0:
        raise ValueError("Section dimensions must be positive")
    long_side = max(float(width_m), float(depth_m))
    short_side = min(float(width_m), float(depth_m))
    ratio = short_side / long_side
    return (
        long_side
        * short_side**3
        * (
            1.0 / 3.0
            - 0.21 * ratio * (1.0 - ratio**4 / 12.0)
        )
    )


def _rc_section(
    ops: Any,
    *,
    section_tag: int,
    concrete_core_tag: int,
    concrete_cover_tag: int,
    steel_tag: int,
    width_m: float,
    depth_m: float,
    clear_cover_m: float,
    transverse_bar_diameter_m: float,
    bar_area_m2: float,
    top_bar_count: int,
    bottom_bar_count: int,
    side_bar_count_each: int = 0,
    perimeter_bar_count: int | None = None,
    shear_modulus_kn_m2: float,
    longitudinal_fibers: tuple[
        tuple[float, float, float], ...
    ] | None = None,
    fiber_mesh: dict[str, int] | None = None,
) -> None:
    """Create a rectangular biaxial fibre section with Saint-Venant torsion."""
    mesh = fiber_mesh or {}
    core_y = int(mesh.get("core_y", 8))
    core_z = int(mesh.get("core_z", 8))
    cover_thickness = int(mesh.get("cover_thickness", 2))
    cover_length_y = int(mesh.get("cover_length_y", core_y))
    cover_length_z = int(mesh.get("cover_length_z", core_z))
    if min(
        core_y,
        core_z,
        cover_thickness,
        cover_length_y,
        cover_length_z,
    ) < 1:
        raise ValueError("Every RC fiber-mesh subdivision must be at least one")
    y_outer = width_m / 2.0
    z_outer = depth_m / 2.0
    y_core = max(
        y_outer - clear_cover_m - 0.5 * transverse_bar_diameter_m,
        width_m * 0.15,
    )
    z_core = max(
        z_outer - clear_cover_m - 0.5 * transverse_bar_diameter_m,
        depth_m * 0.15,
    )
    torsional_constant = rectangular_torsional_constant(width_m, depth_m)
    ops.section(
        "Fiber",
        section_tag,
        "-GJ",
        max(shear_modulus_kn_m2 * torsional_constant, 1.0),
    )
    _rectangle_patch(
        ops,
        concrete_core_tag,
        core_y,
        core_z,
        -y_core,
        -z_core,
        y_core,
        z_core,
    )
    _rectangle_patch(
        ops,
        concrete_cover_tag,
        cover_thickness,
        cover_length_z,
        -y_outer,
        -z_outer,
        -y_core,
        z_outer,
    )
    _rectangle_patch(
        ops,
        concrete_cover_tag,
        cover_thickness,
        cover_length_z,
        y_core,
        -z_outer,
        y_outer,
        z_outer,
    )
    _rectangle_patch(
        ops,
        concrete_cover_tag,
        cover_length_y,
        cover_thickness,
        -y_core,
        -z_outer,
        y_core,
        -z_core,
    )
    _rectangle_patch(
        ops,
        concrete_cover_tag,
        cover_length_y,
        cover_thickness,
        -y_core,
        z_core,
        y_core,
        z_outer,
    )
    if longitudinal_fibers is not None:
        if not longitudinal_fibers:
            raise ValueError("Explicit longitudinal fibre list cannot be empty")
        for y_coordinate, z_coordinate, fiber_area_m2 in longitudinal_fibers:
            ops.fiber(
                y_coordinate,
                z_coordinate,
                fiber_area_m2,
                steel_tag,
            )
        return
    longitudinal_radius = 0.5 * math.sqrt(4.0 * bar_area_m2 / math.pi)
    bar_y = max(
        y_core - 0.5 * transverse_bar_diameter_m - longitudinal_radius,
        0.0,
    )
    bar_z = max(
        z_core - 0.5 * transverse_bar_diameter_m - longitudinal_radius,
        0.0,
    )
    if perimeter_bar_count is not None:
        for y_coordinate, z_coordinate in perimeter_bar_positions(
            perimeter_bar_count,
            half_width_m=bar_y,
            half_depth_m=bar_z,
        ):
            ops.fiber(
                y_coordinate,
                z_coordinate,
                bar_area_m2,
                steel_tag,
            )
    else:
        ops.layer(
            "straight",
            steel_tag,
            top_bar_count,
            bar_area_m2,
            -bar_y,
            bar_z,
            bar_y,
            bar_z,
        )
        ops.layer(
            "straight",
            steel_tag,
            bottom_bar_count,
            bar_area_m2,
            -bar_y,
            -bar_z,
            bar_y,
            -bar_z,
        )
        if side_bar_count_each > 0:
            # ``side_bar_count_each`` means intermediate bars only.  Creating
            # a straight layer from -bar_z to +bar_z would duplicate the four
            # corner fibres that already belong to the top/bottom layers.
            side_coordinates = np.linspace(
                -bar_z,
                bar_z,
                side_bar_count_each + 2,
            )[1:-1]
            for z_coordinate in side_coordinates:
                ops.fiber(
                    -bar_y,
                    float(z_coordinate),
                    bar_area_m2,
                    steel_tag,
                )
                ops.fiber(
                    bar_y,
                    float(z_coordinate),
                    bar_area_m2,
                    steel_tag,
                )


def beam_bar_positions(
    top_bar_count: int,
    side_bar_count_each: int,
    *,
    half_width_m: float,
    half_depth_m: float,
    bottom_bar_count: int | None = None,
) -> tuple[tuple[float, float], ...]:
    """Return the exact unique beam-bar layout used by the fibre section.

    The top and bottom counts include both corner bars on their face.
    ``bottom_bar_count`` defaults to the top count for backward compatibility.
    ``side_bar_count_each`` contains only intermediate bars on each vertical
    face.  Keeping this geometry in one helper makes the Mander confinement
    preflight and the OpenSees fibre section use identical coordinates.
    """
    if bottom_bar_count is None:
        bottom_bar_count = top_bar_count
    if top_bar_count < 2 or bottom_bar_count < 2:
        raise ValueError("Beam top and bottom faces require at least two bars")
    if side_bar_count_each < 0:
        raise ValueError("Beam side-bar count cannot be negative")
    if half_width_m <= 0.0 or half_depth_m <= 0.0:
        raise ValueError("Beam bar-layout dimensions must be positive")
    top_horizontal = np.linspace(
        -half_width_m,
        half_width_m,
        top_bar_count,
    )
    bottom_horizontal = np.linspace(
        -half_width_m,
        half_width_m,
        bottom_bar_count,
    )
    side_interior = np.linspace(
        -half_depth_m,
        half_depth_m,
        side_bar_count_each + 2,
    )[1:-1]
    positions = [
        *((float(y), half_depth_m) for y in top_horizontal),
        *((float(y), -half_depth_m) for y in bottom_horizontal),
        *((-half_width_m, float(z)) for z in side_interior),
        *((half_width_m, float(z)) for z in side_interior),
    ]
    if len(positions) != len(set(positions)):
        raise AssertionError("Generated beam bar layout is not unique")
    return tuple(positions)


def beam_reinforcement_fibers(
    *,
    width_m: float,
    depth_m: float,
    clear_cover_m: float,
    stirrup_diameter_m: float,
    top_bar_count: int,
    top_bar_diameter_m: float,
    top_bar_layers: int,
    bottom_bar_count: int,
    bottom_bar_diameter_m: float,
    bottom_bar_layers: int,
    side_bar_count_each: int,
    side_bar_diameter_m: float,
    minimum_clear_spacing_m: float,
    nominal_maximum_aggregate_size_m: float,
    minimum_bars_per_layer: int = 2,
    three_layer_minimum_depth_m: float = 0.50,
) -> tuple[tuple[float, float, float, float, str], ...]:
    """Return the locked constructible beam reinforcement layout.

    Top, bottom and side bars use one standard diameter.  Top and bottom bar
    counts may differ.  A face uses another layer only when its bars cannot
    fit in fewer layers.  Bars per layer have no arbitrary numerical cap:
    the maximum is calculated from section width, cover, stirrup diameter,
    bar diameter, and the ACI aggregate-aware clear spacing.  At most three
    layers are supported.  Counts are divided as evenly as possible with the
    larger share in the outer layer.  Side bars are intermediate skin bars
    and are not counted as top or bottom flexural reinforcement.
    """
    positive = (
        width_m,
        depth_m,
        clear_cover_m,
        stirrup_diameter_m,
        top_bar_diameter_m,
        bottom_bar_diameter_m,
        side_bar_diameter_m,
        minimum_clear_spacing_m,
        nominal_maximum_aggregate_size_m,
    )
    if any(not math.isfinite(float(value)) or value <= 0.0 for value in positive):
        raise ValueError("Beam reinforcement geometry values must be positive")
    if top_bar_layers not in {1, 2, 3} or bottom_bar_layers not in {1, 2, 3}:
        raise ValueError("Beam faces currently support one to three layers")
    if (
        3 in {top_bar_layers, bottom_bar_layers}
        and depth_m + 1.0e-12 < three_layer_minimum_depth_m
    ):
        raise ValueError(
            "Three-layer beam reinforcement requires the configured minimum "
            "section depth"
        )
    if minimum_bars_per_layer < 2:
        raise ValueError("Each beam reinforcement layer needs at least two bars")
    if not (
        math.isclose(
            top_bar_diameter_m,
            bottom_bar_diameter_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            top_bar_diameter_m,
            side_bar_diameter_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "Top, bottom and side bars must use one common diameter"
        )
    if side_bar_count_each < 0:
        raise ValueError("Beam side-bar count cannot be negative")

    def layer_counts(count: int, layers: int) -> tuple[int, ...]:
        base_count, remainder = divmod(count, layers)
        result = tuple(
            base_count + int(index < remainder)
            for index in range(layers)
        )
        if any(value < minimum_bars_per_layer for value in result):
            raise ValueError(
                "Beam bar count cannot populate every requested layer"
            )
        return result

    fibers: list[tuple[float, float, float, float, str]] = []

    def add_face(
        *,
        role: str,
        count: int,
        diameter_m: float,
        layers: int,
        sign: float,
    ) -> None:
        half_width = (
            width_m / 2.0
            - clear_cover_m
            - stirrup_diameter_m
            - diameter_m / 2.0
        )
        outer_z = (
            depth_m / 2.0
            - clear_cover_m
            - stirrup_diameter_m
            - diameter_m / 2.0
        )
        if half_width <= 0.0 or outer_z <= 0.0:
            raise ValueError("Beam section cannot contain the requested bars")
        required_horizontal_clear = aci_required_clear_spacing_m(
            bar_diameter_m=diameter_m,
            nominal_maximum_aggregate_size_m=(
                nominal_maximum_aggregate_size_m
            ),
            code_minimum_m=minimum_clear_spacing_m,
        )
        geometric_maximum = int(
            math.floor(
                (
                    2.0 * half_width
                    + required_horizontal_clear
                )
                / (diameter_m + required_horizontal_clear)
                + 1.0e-12
            )
        )
        if geometric_maximum < minimum_bars_per_layer:
            raise ValueError(
                "Beam width cannot contain a constructible bar layer"
            )
        required_layers = int(
            math.ceil(count / geometric_maximum)
        )
        if layers != required_layers:
            raise ValueError(
                "Beam reinforcement must use the minimum number of layers "
                "needed for its bar count"
            )
        vertical_center_spacing = (
            diameter_m + required_horizontal_clear
        )
        for layer_index, layer_count in enumerate(
            layer_counts(count, layers)
        ):
            z_coordinate = sign * (
                outer_z - layer_index * vertical_center_spacing
            )
            if sign * z_coordinate <= 0.0:
                raise ValueError("Beam inner layer crosses the section mid-depth")
            horizontal_step = (
                2.0 * half_width / (layer_count - 1)
            )
            horizontal = tuple(
                -half_width + index * horizontal_step
                for index in range(layer_count)
            )
            if layer_count > 1:
                clear_spacing = (
                    horizontal_step - diameter_m
                )
                if (
                    clear_spacing + 1.0e-12
                    < required_horizontal_clear
                ):
                    raise ValueError(
                        "Beam bars do not satisfy horizontal clear spacing"
                    )
            area = math.pi * diameter_m**2 / 4.0
            fibers.extend(
                (
                    float(y_coordinate),
                    float(z_coordinate),
                    area,
                    diameter_m,
                    role,
                )
                for y_coordinate in horizontal
            )

    add_face(
        role="top",
        count=top_bar_count,
        diameter_m=top_bar_diameter_m,
        layers=top_bar_layers,
        sign=1.0,
    )
    add_face(
        role="bottom",
        count=bottom_bar_count,
        diameter_m=bottom_bar_diameter_m,
        layers=bottom_bar_layers,
        sign=-1.0,
    )
    if side_bar_count_each:
        side_y = (
            width_m / 2.0
            - clear_cover_m
            - stirrup_diameter_m
            - side_bar_diameter_m / 2.0
        )
        side_z = (
            depth_m / 2.0
            - clear_cover_m
            - stirrup_diameter_m
            - side_bar_diameter_m / 2.0
        )
        if side_y <= 0.0 or side_z <= 0.0:
            raise ValueError("Beam section cannot contain its side bars")
        side_vertical_step = (
            2.0 * side_z / (side_bar_count_each + 1)
        )
        vertical = tuple(
            -side_z + index * side_vertical_step
            for index in range(1, side_bar_count_each + 1)
        )
        side_area = math.pi * side_bar_diameter_m**2 / 4.0
        for y_coordinate in (-side_y, side_y):
            fibers.extend(
                (
                    float(y_coordinate),
                    float(z_coordinate),
                    side_area,
                    side_bar_diameter_m,
                    "side",
                )
                for z_coordinate in vertical
            )

    # Check actual two-dimensional clear distance.  The common-diameter rule
    # makes the required centre distance constant.  A sweep sorted by the
    # horizontal coordinate only compares bars whose horizontal separation is
    # small enough to conflict; this is exactly equivalent to the previous
    # all-pairs O(N^2) test but avoids millions of irrelevant distance checks
    # for wide multilayer sections.
    common_diameter_m = top_bar_diameter_m
    pair_clear_spacing = aci_required_clear_spacing_m(
        bar_diameter_m=common_diameter_m,
        nominal_maximum_aggregate_size_m=(
            nominal_maximum_aggregate_size_m
        ),
        code_minimum_m=minimum_clear_spacing_m,
    )
    required_distance = common_diameter_m + pair_clear_spacing
    required_distance_squared = required_distance**2
    ordered_fibers = sorted(
        fibers,
        key=lambda fiber: (fiber[0], fiber[1]),
    )
    for first_index, first in enumerate(ordered_fibers):
        for second in ordered_fibers[first_index + 1 :]:
            delta_y = second[0] - first[0]
            if delta_y + 1.0e-12 >= required_distance:
                break
            delta_z = second[1] - first[1]
            if (
                delta_y**2 + delta_z**2 + 1.0e-12
                < required_distance_squared
            ):
                raise ValueError(
                    "Beam reinforcement layout violates clear spacing"
                )
    return tuple(fibers)


def perimeter_bar_positions(
    bar_count: int,
    *,
    half_width_m: float,
    half_depth_m: float,
) -> tuple[tuple[float, float], ...]:
    """Return a symmetric rectangular perimeter layout without duplicate corners."""
    if bar_count < 8 or bar_count % 4:
        raise ValueError(
            "Column perimeter bar count must be a multiple of four and at least 8"
        )
    if half_width_m <= 0.0 or half_depth_m <= 0.0:
        raise ValueError("Column bar-layout dimensions must be positive")
    intervals_per_edge = bar_count // 4
    horizontal = [
        -half_width_m + 2.0 * half_width_m * index / intervals_per_edge
        for index in range(intervals_per_edge + 1)
    ]
    vertical_interior = [
        -half_depth_m + 2.0 * half_depth_m * index / intervals_per_edge
        for index in range(1, intervals_per_edge)
    ]
    positions = [
        *((coordinate, half_depth_m) for coordinate in horizontal),
        *((coordinate, -half_depth_m) for coordinate in horizontal),
        *((-half_width_m, coordinate) for coordinate in vertical_interior),
        *((half_width_m, coordinate) for coordinate in vertical_interior),
    ]
    if len(positions) != bar_count or len(set(positions)) != bar_count:
        raise AssertionError("Generated column bar layout is not unique")
    return tuple(positions)


def build_model(building: dict[str, Any], config: dict[str, Any]) -> ModelInfo:
    """Build the complete gravity-ready 3D nonlinear archetype."""
    ops = _ops()
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    stories = int(building["stories"])
    height = float(building["story_height_m"])
    bay = float(building["bay_width_m"])
    number_of_bays = int(building.get("number_of_bays", 1))
    if number_of_bays < 1:
        raise ValueError("number_of_bays must be at least one")
    model_config = config["model"]
    g = float(model_config["gravity_m_s2"])

    grid_size = number_of_bays + 1
    plan_width = number_of_bays * bay
    grid_xy = tuple(
        (ix * bay, iy * bay)
        for ix in range(grid_size)
        for iy in range(grid_size)
    )

    def grid_node_tag(level: int, ix: int, iy: int) -> int:
        return level * 1000 + ix * grid_size + iy + 1

    floor_nodes: list[tuple[int, ...]] = []
    master_nodes: list[int] = []
    for level in range(stories + 1):
        z = level * height
        tags = []
        for index, (x, y) in enumerate(grid_xy):
            ix, iy = divmod(index, grid_size)
            tag = grid_node_tag(level, ix, iy)
            ops.node(tag, x, y, z)
            tags.append(tag)
        floor_nodes.append(tuple(tags))
        if level == 0:
            for tag in tags:
                ops.fix(tag, 1, 1, 1, 1, 1, 1)
        else:
            master = 100000 + level
            ops.node(master, plan_width / 2.0, plan_width / 2.0, z)
            # The diaphragm master retains UX, UY and RZ only.
            ops.fix(master, 0, 0, 1, 1, 1, 0)
            ops.rigidDiaphragm(3, master, *tags)
            master_nodes.append(master)

    fc = float(building["fc_ksc"]) * KSC_TO_KN_M2
    fy = float(config["building"]["steel_fy_ksc"]) * KSC_TO_KN_M2
    metadata_raw = building.get("design_metadata_json", "{}")
    if isinstance(metadata_raw, str):
        design_metadata = json.loads(metadata_raw or "{}")
    elif isinstance(metadata_raw, dict):
        design_metadata = dict(metadata_raw)
    else:
        raise ValueError("design_metadata_json must be JSON text or a mapping")

    beam_width = float(building["beam_b_m"])
    beam_depth = float(building["beam_h_m"])
    column_width = float(building["column_b_m"])
    column_depth = float(building["column_h_m"])
    beam_bar_diameter = float(building["beam_bar_diameter_m"])
    column_bar_diameter = float(building["column_bar_diameter_m"])
    beam_bar_area = math.pi * beam_bar_diameter**2 / 4.0
    column_bar_area = math.pi * column_bar_diameter**2 / 4.0
    detailing = model_config.get("material_detailing", {})
    clear_cover = float(detailing.get("clear_cover_m", 0.04))
    beam_hoop_diameter = float(
        design_metadata.get(
            "beam_stirrup_diameter_m",
            detailing.get("beam_hoop_diameter_m", 0.009),
        )
    )
    beam_hoop_spacing = float(
        design_metadata.get(
            "beam_stirrup_spacing_m",
            detailing.get("beam_hoop_spacing_m", 0.10),
        )
    )
    beam_hoop_legs = int(
        design_metadata.get(
            "beam_stirrup_legs",
            detailing.get("beam_hoop_legs_each_axis", 2),
        )
    )
    column_hoop_diameter = float(
        design_metadata.get(
            "column_hoop_diameter_m",
            detailing.get("column_hoop_diameter_m", 0.009),
        )
    )
    column_hoop_spacing = float(
        design_metadata.get(
            "column_hoop_spacing_m",
            detailing.get("column_hoop_spacing_m", 0.10),
        )
    )
    column_hoop_legs_x = int(
        design_metadata.get(
            "column_hoop_legs_x",
            detailing.get("column_hoop_legs_x", 2),
        )
    )
    column_hoop_legs_y = int(
        design_metadata.get(
            "column_hoop_legs_y",
            detailing.get("column_hoop_legs_y", 2),
        )
    )
    beam_transverse_fy = (
        float(
            design_metadata.get(
                "beam_stirrup_fy_ksc",
                config["building"]["transverse_steel_fy_ksc"],
            )
        )
        * KSC_TO_KN_M2
    )
    column_transverse_fy = (
        float(
            design_metadata.get(
                "column_hoop_fy_ksc",
                config["building"]["transverse_steel_fy_ksc"],
            )
        )
        * KSC_TO_KN_M2
    )

    bars_per_face = int(building["beam_bars_per_face"])
    beam_top_bar_count = int(
        design_metadata.get("beam_top_bar_count", bars_per_face)
    )
    beam_bottom_bar_count = int(
        design_metadata.get("beam_bottom_bar_count", bars_per_face)
    )
    beam_side_bar_count_each = int(
        design_metadata.get("beam_side_bar_count_each", 0)
    )
    beam_top_bar_diameter = float(
        design_metadata.get(
            "beam_top_bar_diameter_m",
            beam_bar_diameter,
        )
    )
    beam_bottom_bar_diameter = float(
        design_metadata.get(
            "beam_bottom_bar_diameter_m",
            beam_bar_diameter,
        )
    )
    beam_side_bar_diameter = float(
        design_metadata.get(
            "beam_side_bar_diameter_m",
            beam_bar_diameter,
        )
    )
    beam_top_bar_layers = int(
        design_metadata.get("beam_top_bar_layers", 1)
    )
    beam_bottom_bar_layers = int(
        design_metadata.get("beam_bottom_bar_layers", 1)
    )
    beam_minimum_bars_per_layer = int(
        config["building"]["beam_section_search"].get(
            "minimum_bars_per_layer",
            2,
        )
    )
    beam_actual_fibers = beam_reinforcement_fibers(
        width_m=beam_width,
        depth_m=beam_depth,
        clear_cover_m=clear_cover,
        stirrup_diameter_m=beam_hoop_diameter,
        top_bar_count=beam_top_bar_count,
        top_bar_diameter_m=beam_top_bar_diameter,
        top_bar_layers=beam_top_bar_layers,
        bottom_bar_count=beam_bottom_bar_count,
        bottom_bar_diameter_m=beam_bottom_bar_diameter,
        bottom_bar_layers=beam_bottom_bar_layers,
        side_bar_count_each=beam_side_bar_count_each,
        side_bar_diameter_m=beam_side_bar_diameter,
        minimum_clear_spacing_m=float(
            detailing.get("beam_minimum_clear_bar_spacing_m", 0.025)
        ),
        nominal_maximum_aggregate_size_m=float(
            detailing["nominal_maximum_coarse_aggregate_size_m"]
        ),
        minimum_bars_per_layer=beam_minimum_bars_per_layer,
        three_layer_minimum_depth_m=float(
            config["building"]["beam_section_search"].get(
                "three_layer_minimum_depth_m",
                0.50,
            )
        ),
    )
    beam_equivalent_bar_diameter = max(
        beam_top_bar_diameter,
        beam_bottom_bar_diameter,
        beam_side_bar_diameter,
    )
    beam_bar_half_width = (
        beam_width / 2.0
        - clear_cover
        - beam_hoop_diameter
        - beam_equivalent_bar_diameter / 2.0
    )
    beam_bar_half_depth = (
        beam_depth / 2.0
        - clear_cover
        - beam_hoop_diameter
        - beam_equivalent_bar_diameter / 2.0
    )
    beam_longitudinal_bar_positions = beam_bar_positions(
        (
            beam_top_bar_count
            if beam_top_bar_layers == 1
            else math.ceil(
                beam_top_bar_count / beam_top_bar_layers
            )
        ),
        beam_side_bar_count_each,
        half_width_m=beam_bar_half_width,
        half_depth_m=beam_bar_half_depth,
        bottom_bar_count=(
            beam_bottom_bar_count
            if beam_bottom_bar_layers == 1
            else math.ceil(
                beam_bottom_bar_count / beam_bottom_bar_layers
            )
        ),
    )
    column_bar_half_width = (
        column_width / 2.0
        - clear_cover
        - column_hoop_diameter
        - column_bar_diameter / 2.0
    )
    column_bar_half_depth = (
        column_depth / 2.0
        - clear_cover
        - column_hoop_diameter
        - column_bar_diameter / 2.0
    )
    column_bar_count = int(building["column_bar_count"])
    column_bar_positions = perimeter_bar_positions(
        column_bar_count,
        half_width_m=column_bar_half_width,
        half_depth_m=column_bar_half_depth,
    )

    # Cover concrete retains the proposal's Hognestad envelope.  The beam and
    # column cores use separate Mander (1988) confined-concrete parameters
    # derived from their actual longitudinal and transverse reinforcement.
    cover_concrete_base = 11
    beam_core_concrete_base = 12
    column_core_concrete_base = 13
    cover_concrete = 1
    beam_core_concrete = 2
    column_core_concrete = 4
    beam_steel_base = 31
    beam_steel = 3
    column_steel_base = 51
    column_steel = 5
    hognestad = model_config.get("hognestad", {})
    cover_parameters = hognestad_concrete_parameters(
        fc,
        peak_strain=float(hognestad.get("peak_strain", 0.0020)),
        ultimate_strain=float(
            hognestad.get("ultimate_strain", 0.0038)
        ),
        ultimate_strength_ratio=float(
            hognestad.get("ultimate_strength_ratio", 0.85)
        ),
    )
    confinement = model_config.get("mander_confinement", {})
    confinement_common = {
        "clear_cover_m": clear_cover,
        "eps_co": float(hognestad.get("peak_strain", 0.0020)),
        "transverse_ultimate_strain": float(
            confinement.get("transverse_ultimate_strain", 0.12)
        ),
        "minimum_ultimate_strain": float(
            confinement.get("minimum_ultimate_strain", 0.0038)
        ),
        "maximum_ultimate_strain": float(
            confinement.get("maximum_ultimate_strain", 0.030)
        ),
    }
    beam_confined = mander_rectangular_confined_parameters(
        fc,
        width_m=beam_width,
        depth_m=beam_depth,
        hoop_diameter_m=beam_hoop_diameter,
        hoop_spacing_m=beam_hoop_spacing,
        transverse_legs_x=beam_hoop_legs,
        transverse_legs_y=beam_hoop_legs,
        longitudinal_bar_diameter_m=beam_equivalent_bar_diameter,
        longitudinal_bar_area_m2=(
            math.pi * beam_equivalent_bar_diameter**2 / 4.0
        ),
        bar_positions=beam_longitudinal_bar_positions,
        longitudinal_area_total_m2=sum(
            fiber[2] for fiber in beam_actual_fibers
        ),
        transverse_fy_kn_m2=beam_transverse_fy,
        **confinement_common,
    )
    column_confined = mander_rectangular_confined_parameters(
        fc,
        width_m=column_width,
        depth_m=column_depth,
        hoop_diameter_m=column_hoop_diameter,
        hoop_spacing_m=column_hoop_spacing,
        transverse_legs_x=column_hoop_legs_x,
        transverse_legs_y=column_hoop_legs_y,
        longitudinal_bar_diameter_m=column_bar_diameter,
        longitudinal_bar_area_m2=column_bar_area,
        bar_positions=column_bar_positions,
        transverse_fy_kn_m2=column_transverse_fy,
        **confinement_common,
    )
    # The derivative of the Hognestad ascending parabola at the origin is
    # 2f'c/eps0.  Use the same tangent for the elastic member interior so the
    # Nonlinear end sections and the elastic centre are initially compatible.
    ec = 2.0 * fc / abs(cover_parameters[1])
    poisson = 0.20
    shear_modulus = ec / (2.0 * (1.0 + poisson))
    ops.uniaxialMaterial(
        "Concrete01",
        cover_concrete_base,
        *cover_parameters,
    )
    ops.uniaxialMaterial(
        "Concrete04",
        beam_core_concrete_base,
        -beam_confined.fcc_kn_m2,
        -beam_confined.eps_cc,
        -beam_confined.eps_cu,
        beam_confined.ec_kn_m2,
    )
    ops.uniaxialMaterial(
        "Concrete04",
        column_core_concrete_base,
        -column_confined.fcc_kn_m2,
        -column_confined.eps_cc,
        -column_confined.eps_cu,
        column_confined.ec_kn_m2,
    )
    concrete_failure_at_epsilon_cu = bool(
        model_config.get(
            "concrete_failure_at_epsilon_cu",
            model_config.get("concrete_failure_at_ultimate_strain", True),
        )
    )
    if concrete_failure_at_epsilon_cu:
        # eps_cu is the concrete failure limit. Concrete01 otherwise retains
        # 0.85f'c indefinitely after eps_cu, so MinMax terminates each concrete
        # fibre when its compressive strain reaches that limit.
        ops.uniaxialMaterial(
            "MinMax",
            cover_concrete,
            cover_concrete_base,
            "-min",
            cover_parameters[3],
        )
        ops.uniaxialMaterial(
            "MinMax",
            beam_core_concrete,
            beam_core_concrete_base,
            "-min",
            -beam_confined.eps_cu,
        )
        ops.uniaxialMaterial(
            "MinMax",
            column_core_concrete,
            column_core_concrete_base,
            "-min",
            -column_confined.eps_cu,
        )
    else:
        cover_concrete = cover_concrete_base
        beam_core_concrete = beam_core_concrete_base
        column_core_concrete = column_core_concrete_base
    define_hysteretic_steel_material(
        ops,
        base_material_tag=beam_steel_base,
        material_tag=beam_steel,
        fy_kn_m2=fy,
        model_config=model_config,
    )
    define_hysteretic_steel_material(
        ops,
        base_material_tag=column_steel_base,
        material_tag=column_steel,
        fy_kn_m2=fy,
        model_config=model_config,
    )

    beam_section = 101
    column_section = 102
    beam_elastic_section = 111
    column_elastic_section = 112
    _rc_section(
        ops,
        section_tag=beam_section,
        concrete_core_tag=beam_core_concrete,
        concrete_cover_tag=cover_concrete,
        steel_tag=beam_steel,
        width_m=beam_width,
        depth_m=beam_depth,
        clear_cover_m=clear_cover,
        transverse_bar_diameter_m=beam_hoop_diameter,
        bar_area_m2=beam_bar_area,
        top_bar_count=beam_top_bar_count,
        bottom_bar_count=beam_bottom_bar_count,
        side_bar_count_each=beam_side_bar_count_each,
        shear_modulus_kn_m2=shear_modulus,
        fiber_mesh=dict(model_config.get("fiber_mesh", {})),
        longitudinal_fibers=tuple(
            (fiber[0], fiber[1], fiber[2])
            for fiber in beam_actual_fibers
        ),
    )
    _rc_section(
        ops,
        section_tag=column_section,
        concrete_core_tag=column_core_concrete,
        concrete_cover_tag=cover_concrete,
        steel_tag=column_steel,
        width_m=column_width,
        depth_m=column_depth,
        clear_cover_m=clear_cover,
        transverse_bar_diameter_m=column_hoop_diameter,
        bar_area_m2=column_bar_area,
        top_bar_count=0,
        bottom_bar_count=0,
        perimeter_bar_count=column_bar_count,
        shear_modulus_kn_m2=shear_modulus,
        fiber_mesh=dict(model_config.get("fiber_mesh", {})),
    )
    _elastic_rectangular_section(
        ops,
        section_tag=beam_elastic_section,
        elastic_modulus_kn_m2=ec,
        shear_modulus_kn_m2=shear_modulus,
        width_m=beam_width,
        depth_m=beam_depth,
    )
    _elastic_rectangular_section(
        ops,
        section_tag=column_elastic_section,
        elastic_modulus_kn_m2=ec,
        shear_modulus_kn_m2=shear_modulus,
        width_m=column_width,
        depth_m=column_depth,
    )

    column_transform = 1
    beam_transform = 2
    ops.geomTransf("PDelta", column_transform, 1.0, 0.0, 0.0)
    ops.geomTransf("PDelta", beam_transform, 0.0, 0.0, 1.0)
    hinge_layout = plastic_hinge_layout(building, config)
    hinge_length_method = hinge_layout.method
    shear_span_method = hinge_layout.shear_span_method
    all_column_hinges = [
        value
        for level in hinge_layout.column_hinge_lengths_m
        for pair in level
        for value in pair
    ]
    all_beam_hinges = [
        value
        for level in hinge_layout.beam_hinge_lengths_m
        for pair in level
        for value in pair
    ]
    column_hinge_length = float(np.median(all_column_hinges))
    beam_hinge_length = float(np.median(all_beam_hinges))
    beam_integration = str(
        model_config.get("beam_integration", "HingeMidpoint")
    )
    # Draft 5 Section 3.3.4: nonlinear fibre regions have finite physical
    # lengths at both member ends.  Each end now uses the Priestley Lp based
    # on its elastic-reference distance to contraflexure (abs(M/V)), rather
    # than substituting the complete bay width or storey height for L.
    element_max_iterations = int(
        model_config.get("force_beam_column_max_iterations", 100)
    )
    element_tolerance = float(
        model_config.get("force_beam_column_tolerance", 1.0e-8)
    )
    element_formulation = str(
        model_config.get("element_formulation", "forceBeamColumn")
    )
    if element_formulation not in {"dispBeamColumn", "forceBeamColumn"}:
        raise ValueError(
            "element_formulation must be dispBeamColumn or forceBeamColumn"
        )

    def create_member(
        tag: int,
        start_node: int,
        end_node: int,
        transform_tag: int,
        integration_tag: int,
    ) -> None:
        if element_formulation == "dispBeamColumn":
            ops.element(
                "dispBeamColumn",
                tag,
                start_node,
                end_node,
                transform_tag,
                integration_tag,
            )
        else:
            ops.element(
                "forceBeamColumn",
                tag,
                start_node,
                end_node,
                transform_tag,
                integration_tag,
                "-iter",
                element_max_iterations,
                element_tolerance,
            )

    element_tag = 1
    column_tags_by_storey: list[tuple[int, ...]] = []
    floor_beam_tags: list[tuple[int, ...]] = []
    for level in range(1, stories + 1):
        column_tags: list[int] = []
        for column_index, (lower, upper) in enumerate(
            zip(floor_nodes[level - 1], floor_nodes[level])
        ):
            hinge_i, hinge_j = (
                hinge_layout.column_hinge_lengths_m[level - 1][column_index]
            )
            elastic_interior = plastic_hinge_elastic_interior_length_m(
                member_length_m=height,
                hinge_length_i_m=hinge_i,
                hinge_length_j_m=hinge_j,
                integration=beam_integration,
            )
            if elastic_interior <= 1.0e-10 * height:
                raise ValueError(
                    f"Column {beam_integration} plastic hinges leave no "
                    "positive elastic interior"
                )
            integration_tag = 10000 + element_tag
            ops.beamIntegration(
                beam_integration,
                integration_tag,
                column_section,
                hinge_i,
                column_section,
                hinge_j,
                column_elastic_section,
            )
            column_tags.append(element_tag)
            create_member(
                element_tag,
                lower,
                upper,
                column_transform,
                integration_tag,
            )
            element_tag += 1
        column_tags_by_storey.append(tuple(column_tags))
        grid_nodes = floor_nodes[level]
        beam_tags: list[int] = []
        beam_index = 0
        for iy in range(grid_size):
            for ix in range(number_of_bays):
                start = ix * grid_size + iy
                end = (ix + 1) * grid_size + iy
                hinge_i, hinge_j = (
                    hinge_layout.beam_hinge_lengths_m[level - 1][beam_index]
                )
                elastic_interior = plastic_hinge_elastic_interior_length_m(
                    member_length_m=bay,
                    hinge_length_i_m=hinge_i,
                    hinge_length_j_m=hinge_j,
                    integration=beam_integration,
                )
                if elastic_interior <= 1.0e-10 * bay:
                    raise ValueError(
                        f"Beam {beam_integration} plastic hinges leave no "
                        "positive elastic interior"
                    )
                integration_tag = 10000 + element_tag
                ops.beamIntegration(
                    beam_integration,
                    integration_tag,
                    beam_section,
                    hinge_i,
                    beam_section,
                    hinge_j,
                    beam_elastic_section,
                )
                beam_tags.append(element_tag)
                create_member(
                    element_tag,
                    grid_nodes[start],
                    grid_nodes[end],
                    beam_transform,
                    integration_tag,
                )
                element_tag += 1
                beam_index += 1
        for ix in range(grid_size):
            for iy in range(number_of_bays):
                start = ix * grid_size + iy
                end = ix * grid_size + iy + 1
                hinge_i, hinge_j = (
                    hinge_layout.beam_hinge_lengths_m[level - 1][beam_index]
                )
                elastic_interior = plastic_hinge_elastic_interior_length_m(
                    member_length_m=bay,
                    hinge_length_i_m=hinge_i,
                    hinge_length_j_m=hinge_j,
                    integration=beam_integration,
                )
                if elastic_interior <= 1.0e-10 * bay:
                    raise ValueError(
                        f"Beam {beam_integration} plastic hinges leave no "
                        "positive elastic interior"
                    )
                integration_tag = 10000 + element_tag
                ops.beamIntegration(
                    beam_integration,
                    integration_tag,
                    beam_section,
                    hinge_i,
                    beam_section,
                    hinge_j,
                    beam_elastic_section,
                )
                beam_tags.append(element_tag)
                create_member(
                    element_tag,
                    grid_nodes[start],
                    grid_nodes[end],
                    beam_transform,
                    integration_tag,
                )
                element_tag += 1
                beam_index += 1
        floor_beam_tags.append(tuple(beam_tags))

    column_count = grid_size**2
    beam_count = 2 * number_of_bays * grid_size
    if any(len(tags) != column_count for tags in column_tags_by_storey):
        raise AssertionError("Column topology count is inconsistent")
    if any(len(tags) != beam_count for tags in floor_beam_tags):
        raise AssertionError("Beam topology count is inconsistent")

    column_points: tuple[float, ...] = ()
    column_weights: tuple[float, ...] = ()
    for level_index, tags in enumerate(column_tags_by_storey):
        for member_index, tag in enumerate(tags):
            points, weights = _validate_element_integration(
                ops,
                element_tag=tag,
                member_length_m=height,
                member_label=f"column {tag}",
            )
            if beam_integration == "HingeMidpoint":
                expected_i, expected_j = (
                    hinge_layout.column_hinge_lengths_m[level_index][
                        member_index
                    ]
                )
                if not (
                    math.isclose(weights[0], expected_i, rel_tol=1.0e-9)
                    and math.isclose(
                        weights[-1],
                        expected_j,
                        rel_tol=1.0e-9,
                    )
                ):
                    raise ValueError(
                        f"Column {tag}: OpenSees hinge weights differ from "
                        "the proposal-consistent member-end Lp values"
                    )
            if not column_points:
                column_points, column_weights = points, weights

    beam_points: tuple[float, ...] = ()
    beam_weights: tuple[float, ...] = ()
    for level_index, tags in enumerate(floor_beam_tags):
        for member_index, tag in enumerate(tags):
            points, weights = _validate_element_integration(
                ops,
                element_tag=tag,
                member_length_m=bay,
                member_label=f"beam {tag}",
            )
            if beam_integration == "HingeMidpoint":
                expected_i, expected_j = (
                    hinge_layout.beam_hinge_lengths_m[level_index][
                        member_index
                    ]
                )
                if not (
                    math.isclose(weights[0], expected_i, rel_tol=1.0e-9)
                    and math.isclose(
                        weights[-1],
                        expected_j,
                        rel_tol=1.0e-9,
                    )
                ):
                    raise ValueError(
                        f"Beam {tag}: OpenSees hinge weights differ from "
                        "the proposal-consistent member-end Lp values"
                    )
            if not beam_points:
                beam_points, beam_weights = points, weights

    floor_mass = float(building["floor_mass_kn_s2_m"])
    rotational_mass = floor_mass * plan_width**2 / 6.0
    for master in master_nodes:
        ops.mass(master, floor_mass, floor_mass, 1.0e-12, 1.0e-12, 1.0e-12, rotational_mass)

    # The catalogued floor mass includes slab, SDL, the configured fraction of
    # live load, and member self-weight. Apply area load plus beam self-weight
    # as vertical element loads so the beams enter lateral analysis with their
    # gravity moments; apply each storey's column self-weight at its top nodes.
    # Each two-way panel transfers one quarter of its area load to each edge.
    # A perimeter beam segment therefore receives q*bay/4 and an interior
    # segment, shared by two panels, receives q*bay/2.
    live_fraction = float(model_config["live_load_mass_fraction"])
    concrete_density = float(model_config["concrete_density_kn_m3"])
    area_load_kn_m2 = (
        float(building["dead_load_kn_m2"])
        + live_fraction * float(building["live_load_kn_m2"])
    )
    area_floor_weight = area_load_kn_m2 * plan_width**2
    beam_floor_weight = (
        beam_count * bay * beam_width * beam_depth * concrete_density
    )
    column_floor_weight = (
        column_count * height * column_width * column_depth * concrete_density
    )
    gravity_floor_load = (
        area_floor_weight + beam_floor_weight + column_floor_weight
    )
    catalogued_floor_weight = floor_mass * g
    if not math.isclose(
        gravity_floor_load,
        catalogued_floor_weight,
        rel_tol=1.0e-9,
        abs_tol=1.0e-8,
    ):
        raise ValueError(
            "Catalogued floor mass is inconsistent with area and member "
            "self-weight used by the OpenSees gravity load pattern"
        )
    beam_self_weight_kn_m = (
        beam_width * beam_depth * concrete_density
    )
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for tags, beam_tags in zip(floor_nodes[1:], floor_beam_tags):
        beam_index = 0
        for iy in range(grid_size):
            adjacent_panels = 1 if iy in {0, number_of_bays} else 2
            tributary_width = adjacent_panels * bay / 4.0
            for _ix in range(number_of_bays):
                beam_tag = beam_tags[beam_index]
                beam_index += 1
                uniform_beam_load = (
                    area_load_kn_m2 * tributary_width
                    + beam_self_weight_kn_m
                )
                ops.eleLoad(
                    "-ele",
                    beam_tag,
                    "-type",
                    "-beamUniform",
                    0.0,
                    -uniform_beam_load,
                    0.0,
                )
        for ix in range(grid_size):
            adjacent_panels = 1 if ix in {0, number_of_bays} else 2
            tributary_width = adjacent_panels * bay / 4.0
            for _iy in range(number_of_bays):
                beam_tag = beam_tags[beam_index]
                beam_index += 1
                uniform_beam_load = (
                    area_load_kn_m2 * tributary_width
                    + beam_self_weight_kn_m
                )
                ops.eleLoad(
                    "-ele",
                    beam_tag,
                    "-type",
                    "-beamUniform",
                    0.0,
                    -uniform_beam_load,
                    0.0,
                )
        if beam_index != len(beam_tags):
            raise AssertionError("Gravity beam-load topology is inconsistent")
        column_segment_weight = (
            height * column_width * column_depth * concrete_density
        )
        for tag in tags:
            ops.load(
                tag,
                0.0,
                0.0,
                -column_segment_weight,
                0.0,
                0.0,
                0.0,
            )

    return ModelInfo(
        number_of_bays=number_of_bays,
        grid_nodes_per_floor=column_count,
        columns_per_storey=column_count,
        beams_per_storey=beam_count,
        base_nodes=floor_nodes[0],
        floor_nodes=tuple(floor_nodes[1:]),
        master_nodes=tuple(master_nodes),
        floor_elevations_m=tuple((index + 1) * height for index in range(stories)),
        expected_gravity_kn=gravity_floor_load * stories,
        total_mass_kn_s2_m=floor_mass * stories,
        column_element_tags_by_storey=tuple(column_tags_by_storey),
        beam_element_tags_by_storey=tuple(floor_beam_tags),
        beam_integration=beam_integration,
        plastic_hinge_length_method=hinge_length_method,
        plastic_hinge_shear_span_method=shear_span_method,
        plastic_hinge_reference_fallback_end_count=(
            hinge_layout.reference_fallback_end_count
        ),
        plastic_hinge_reference_clipped_end_count=(
            hinge_layout.reference_clipped_end_count
        ),
        column_hinge_length_m=column_hinge_length,
        beam_hinge_length_m=beam_hinge_length,
        column_hinge_lengths_m=hinge_layout.column_hinge_lengths_m,
        beam_hinge_lengths_m=hinge_layout.beam_hinge_lengths_m,
        column_shear_spans_m=hinge_layout.column_shear_spans_m,
        beam_shear_spans_m=hinge_layout.beam_shear_spans_m,
        column_integration_points_m=column_points,
        column_integration_weights_m=column_weights,
        beam_integration_points_m=beam_points,
        beam_integration_weights_m=beam_weights,
    )


def _configure_static_analysis(ops: Any) -> None:
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-8, 30, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")


def run_gravity(info: ModelInfo) -> float:
    """Run gravity and return the relative base-reaction equilibrium error."""
    ops = _ops()
    _configure_static_analysis(ops)
    if ops.analyze(10) != 0:
        raise RuntimeError("Gravity analysis did not converge")
    ops.reactions()
    reaction = sum(float(ops.nodeReaction(tag, 3)) for tag in info.base_nodes)
    error = abs(abs(reaction) - info.expected_gravity_kn) / max(
        info.expected_gravity_kn, 1.0
    )
    ops.loadConst("-time", 0.0)
    return error


def plastic_hinge_mechanism_snapshot(
    info: ModelInfo,
    building: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Summarize yielded member-end sections from the current OpenSees state.

    The indicator is intentionally diagnostic and is not one of the seven ANN
    inputs. A member end is counted as yielded when the section deformation
    gives a strain of at least ``fy/Es`` at any actual longitudinal
    reinforcing-fibre coordinate. This records the nonlinear mechanism rather
    than inferring it from SCWB strength ratio alone.
    """
    ops = _ops()
    model = config["model"]
    fy_kn_m2 = (
        float(
            building.get(
                "steel_fy_ksc",
                config["building"]["steel_fy_ksc"],
            )
        )
        * KSC_TO_KN_M2
    )
    elastic_modulus_kn_m2 = (
        float(model["steel_hysteretic"]["elastic_modulus_mpa"]) * 1000.0
    )
    yield_strain = fy_kn_m2 / elastic_modulus_kn_m2
    if yield_strain <= 0.0:
        raise ValueError("Steel yield strain must be positive")

    beam_bar_positions, column_bar_positions, layout_audit = (
        _mechanism_longitudinal_bar_positions(building, config)
    )

    def member_end_ratios(
        element_tags_by_storey: tuple[tuple[int, ...], ...],
        *,
        longitudinal_bar_positions: tuple[tuple[float, float], ...],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for storey_index, element_tags in enumerate(
            element_tags_by_storey, start=1
        ):
            for element_tag in element_tags:
                point_count = len(
                    ops.eleResponse(element_tag, "integrationPoints")
                )
                if point_count < 2:
                    raise ValueError(
                        f"Element {element_tag} has fewer than two sections"
                    )
                for end_name, section_number in (
                    ("I", 1),
                    ("J", point_count),
                ):
                    deformation = tuple(
                        float(value)
                        for value in ops.eleResponse(
                            element_tag,
                            "section",
                            section_number,
                            "deformation",
                        )
                    )
                    if len(deformation) < 3:
                        raise ValueError(
                            f"Element {element_tag} section deformation is "
                            "incomplete"
                        )
                    extreme_strain = _maximum_longitudinal_bar_strain(
                        deformation,
                        longitudinal_bar_positions,
                    )
                    rows.append(
                        {
                            "storey": storey_index,
                            "element_tag": int(element_tag),
                            "end": end_name,
                            "yield_strain_ratio": float(
                                extreme_strain / yield_strain
                            ),
                        }
                    )
        return rows

    beam_rows = member_end_ratios(
        info.beam_element_tags_by_storey,
        longitudinal_bar_positions=beam_bar_positions,
    )
    column_rows = member_end_ratios(
        info.column_element_tags_by_storey,
        longitudinal_bar_positions=column_bar_positions,
    )
    yielded_ratio = float(model.get("mechanism_yield_strain_ratio", 1.0))
    beam_fraction = sum(
        row["yield_strain_ratio"] >= yielded_ratio for row in beam_rows
    ) / max(len(beam_rows), 1)
    column_fraction = sum(
        row["yield_strain_ratio"] >= yielded_ratio for row in column_rows
    ) / max(len(column_rows), 1)
    story_fractions = {}
    for storey in range(1, len(info.column_element_tags_by_storey) + 1):
        rows = [row for row in column_rows if row["storey"] == storey]
        story_fractions[str(storey)] = sum(
            row["yield_strain_ratio"] >= yielded_ratio for row in rows
        ) / max(len(rows), 1)
    maximum_story_fraction = max(story_fractions.values(), default=0.0)
    story_threshold = float(
        model.get("mechanism_story_column_fraction_threshold", 0.75)
    )
    if beam_fraction == 0.0 and column_fraction == 0.0:
        mechanism_class = "elastic"
    elif (
        maximum_story_fraction >= story_threshold
        and column_fraction >= beam_fraction
    ):
        mechanism_class = "story-column-dominant"
    elif beam_fraction >= 1.25 * max(column_fraction, 1.0e-12):
        mechanism_class = "beam-dominant"
    else:
        mechanism_class = "mixed"
    return {
        "mechanism_class": mechanism_class,
        "yield_strain": yield_strain,
        "yield_strain_ratio_threshold": yielded_ratio,
        "beam_yielded_end_fraction": float(beam_fraction),
        "column_yielded_end_fraction": float(column_fraction),
        "maximum_story_column_yielded_end_fraction": float(
            maximum_story_fraction
        ),
        "story_column_yielded_end_fractions": story_fractions,
        "maximum_beam_end_yield_strain_ratio": max(
            (row["yield_strain_ratio"] for row in beam_rows),
            default=0.0,
        ),
        "maximum_column_end_yield_strain_ratio": max(
            (row["yield_strain_ratio"] for row in column_rows),
            default=0.0,
        ),
        "bar_strain_evaluation": (
            "exact_longitudinal_fiber_coordinates_"
            "eps_minus_y_kappa_z_plus_z_kappa_y"
        ),
        **layout_audit,
    }


def _mechanism_longitudinal_bar_positions(
    building: dict[str, Any],
    config: dict[str, Any],
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[tuple[float, float], ...],
    dict[str, Any],
]:
    """Recreate the exact longitudinal-fibre coordinates used by the model.

    Mechanism classification is diagnostic, but it must still use each
    building's optimized stirrup/hoop diameter and its actual multilayer beam
    reinforcement.  Reusing the public layout generators prevents the
    diagnostic from silently reverting to the material-detailing defaults.
    """
    model = config["model"]
    detailing = model["material_detailing"]
    metadata_raw = building.get("design_metadata_json", "{}")
    if isinstance(metadata_raw, str):
        design_metadata = json.loads(metadata_raw or "{}")
    elif isinstance(metadata_raw, dict):
        design_metadata = dict(metadata_raw)
    else:
        raise ValueError("design_metadata_json must be JSON text or a mapping")

    clear_cover_m = float(detailing["clear_cover_m"])
    beam_default_diameter_m = float(building["beam_bar_diameter_m"])
    beam_stirrup_diameter_m = float(
        design_metadata.get(
            "beam_stirrup_diameter_m",
            detailing["beam_hoop_diameter_m"],
        )
    )
    bars_per_face = int(building["beam_bars_per_face"])
    top_count = int(
        design_metadata.get("beam_top_bar_count", bars_per_face)
    )
    bottom_count = int(
        design_metadata.get("beam_bottom_bar_count", bars_per_face)
    )
    side_count_each = int(
        design_metadata.get("beam_side_bar_count_each", 0)
    )
    top_diameter_m = float(
        design_metadata.get(
            "beam_top_bar_diameter_m",
            beam_default_diameter_m,
        )
    )
    bottom_diameter_m = float(
        design_metadata.get(
            "beam_bottom_bar_diameter_m",
            beam_default_diameter_m,
        )
    )
    side_diameter_m = float(
        design_metadata.get(
            "beam_side_bar_diameter_m",
            beam_default_diameter_m,
        )
    )
    top_layers = int(design_metadata.get("beam_top_bar_layers", 1))
    bottom_layers = int(
        design_metadata.get("beam_bottom_bar_layers", 1)
    )
    search = config["building"]["beam_section_search"]
    beam_fibers = beam_reinforcement_fibers(
        width_m=float(building["beam_b_m"]),
        depth_m=float(building["beam_h_m"]),
        clear_cover_m=clear_cover_m,
        stirrup_diameter_m=beam_stirrup_diameter_m,
        top_bar_count=top_count,
        top_bar_diameter_m=top_diameter_m,
        top_bar_layers=top_layers,
        bottom_bar_count=bottom_count,
        bottom_bar_diameter_m=bottom_diameter_m,
        bottom_bar_layers=bottom_layers,
        side_bar_count_each=side_count_each,
        side_bar_diameter_m=side_diameter_m,
        minimum_clear_spacing_m=float(
            detailing["beam_minimum_clear_bar_spacing_m"]
        ),
        nominal_maximum_aggregate_size_m=float(
            detailing["nominal_maximum_coarse_aggregate_size_m"]
        ),
        minimum_bars_per_layer=int(
            search.get("minimum_bars_per_layer", 2)
        ),
        three_layer_minimum_depth_m=float(
            search.get("three_layer_minimum_depth_m", 0.50)
        ),
    )
    beam_positions = tuple(
        (float(fiber[0]), float(fiber[1]))
        for fiber in beam_fibers
    )

    column_diameter_m = float(building["column_bar_diameter_m"])
    column_hoop_diameter_m = float(
        design_metadata.get(
            "column_hoop_diameter_m",
            detailing["column_hoop_diameter_m"],
        )
    )
    column_half_width_m = (
        0.5 * float(building["column_b_m"])
        - clear_cover_m
        - column_hoop_diameter_m
        - 0.5 * column_diameter_m
    )
    column_half_depth_m = (
        0.5 * float(building["column_h_m"])
        - clear_cover_m
        - column_hoop_diameter_m
        - 0.5 * column_diameter_m
    )
    column_positions = perimeter_bar_positions(
        int(building["column_bar_count"]),
        half_width_m=column_half_width_m,
        half_depth_m=column_half_depth_m,
    )
    return (
        beam_positions,
        column_positions,
        {
            "beam_longitudinal_fiber_count": len(beam_positions),
            "column_longitudinal_fiber_count": len(column_positions),
            "beam_stirrup_diameter_m": beam_stirrup_diameter_m,
            "column_hoop_diameter_m": column_hoop_diameter_m,
        },
    )


def _maximum_longitudinal_bar_strain(
    section_deformation: tuple[float, ...],
    longitudinal_bar_positions: tuple[tuple[float, float], ...],
) -> float:
    """Return the maximum absolute strain at an actual reinforcing fibre.

    OpenSees orders a three-dimensional fibre-section deformation as axial
    strain, curvature about z, and curvature about y.  For a fibre at (y, z),
    its longitudinal strain is ``eps - y*kappa_z + z*kappa_y``.  Torsional
    deformation, when present as a fourth component, does not create
    longitudinal bar strain in this section model.
    """
    if len(section_deformation) < 3:
        raise ValueError("Section deformation must contain eps, kappa_z, kappa_y")
    if not longitudinal_bar_positions:
        raise ValueError("At least one longitudinal-bar position is required")
    deformation = tuple(float(value) for value in section_deformation[:3])
    if any(not math.isfinite(value) for value in deformation):
        raise ValueError("Section deformation must be finite")
    strains = []
    for y_coordinate, z_coordinate in longitudinal_bar_positions:
        y_value = float(y_coordinate)
        z_value = float(z_coordinate)
        if not math.isfinite(y_value) or not math.isfinite(z_value):
            raise ValueError("Longitudinal-bar coordinates must be finite")
        strains.append(
            abs(
                deformation[0]
                - y_value * deformation[1]
                + z_value * deformation[2]
            )
        )
    return float(max(strains))


def _directional_effective_modal_mass_ratios(
    phi_x: np.ndarray,
    phi_y: np.ndarray,
    phi_rz: np.ndarray,
    mass_x: np.ndarray,
    mass_y: np.ndarray,
    mass_rz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return X/Y effective-mass ratios for mass-normalization-free modes.

    The generalized modal mass includes UX, UY and RZ inertia at every rigid
    diaphragm master node.  The directional participation numerator contains
    the corresponding translational influence vector.  Consequently the
    ratios are invariant to arbitrary eigenvector scaling and remain valid
    when the symmetric X/Y modes are repeated or nearly repeated.
    """
    vectors = tuple(
        np.asarray(value, dtype=float)
        for value in (phi_x, phi_y, phi_rz)
    )
    masses = tuple(
        np.asarray(value, dtype=float)
        for value in (mass_x, mass_y, mass_rz)
    )
    if any(vector.ndim != 2 for vector in vectors):
        raise ValueError("Modal eigenvector arrays must be two-dimensional")
    if not (
        vectors[0].shape == vectors[1].shape == vectors[2].shape
        and masses[0].shape == masses[1].shape == masses[2].shape
        and vectors[0].shape[1] == masses[0].size
    ):
        raise ValueError("Modal eigenvectors and nodal masses are incompatible")
    if any(
        np.any(~np.isfinite(value)) for value in (*vectors, *masses)
    ):
        raise ValueError("Modal eigenvectors and masses must be finite")
    if np.any(masses[0] <= 0.0) or np.any(masses[1] <= 0.0):
        raise ValueError("Every diaphragm must have positive X/Y mass")
    if np.any(masses[2] < 0.0):
        raise ValueError("Diaphragm rotational masses cannot be negative")

    generalized_mass = (
        (vectors[0] ** 2) @ masses[0]
        + (vectors[1] ** 2) @ masses[1]
        + (vectors[2] ** 2) @ masses[2]
    )
    if np.any(~np.isfinite(generalized_mass)) or np.any(
        generalized_mass <= 0.0
    ):
        raise RuntimeError("A mode has non-positive generalized modal mass")
    x_numerator = vectors[0] @ masses[0]
    y_numerator = vectors[1] @ masses[1]
    x_ratio = x_numerator**2 / (
        generalized_mass * float(np.sum(masses[0]))
    )
    y_ratio = y_numerator**2 / (
        generalized_mass * float(np.sum(masses[1]))
    )
    x_ratio = np.maximum(x_ratio, 0.0)
    y_ratio = np.maximum(y_ratio, 0.0)
    # A directional effective-mass fraction cannot exceed unity.  Check the
    # invariant here so a malformed mass assignment/eigenvector extraction
    # cannot still produce a plausible-looking "dominant mode".
    tolerance = 1.0e-8
    if np.any(x_ratio > 1.0 + tolerance) or np.any(y_ratio > 1.0 + tolerance):
        raise RuntimeError(
            "A directional modal effective-mass ratio exceeds unity"
        )
    return x_ratio, y_ratio


def _select_distinct_directional_modes(
    x_effective_mass_ratios: np.ndarray,
    y_effective_mass_ratios: np.ndarray,
) -> tuple[int, int]:
    """Choose distinct zero-based X/Y modes by maximum joint participation."""
    x_ratio = np.asarray(x_effective_mass_ratios, dtype=float)
    y_ratio = np.asarray(y_effective_mass_ratios, dtype=float)
    if (
        x_ratio.ndim != 1
        or y_ratio.ndim != 1
        or x_ratio.shape != y_ratio.shape
        or x_ratio.size < 2
        or np.any(~np.isfinite(x_ratio))
        or np.any(~np.isfinite(y_ratio))
        or np.any(x_ratio < 0.0)
        or np.any(y_ratio < 0.0)
    ):
        raise ValueError(
            "Directional effective-mass ratios must be matching finite "
            "nonnegative vectors containing at least two modes"
        )
    candidates = (
        (
            float(x_ratio[x_index] + y_ratio[y_index]),
            float(min(x_ratio[x_index], y_ratio[y_index])),
            -max(x_index, y_index),
            -x_index,
            -y_index,
            x_index,
            y_index,
        )
        for x_index in range(x_ratio.size)
        for y_index in range(y_ratio.size)
        if x_index != y_index
    )
    *_, x_mode_index, y_mode_index = max(candidates)
    return int(x_mode_index), int(y_mode_index)


def modal_properties(
    info: ModelInfo,
    mode_count: int = 6,
    *,
    solver: str = "fullGenLapack",
    minimum_cumulative_mass_ratio: float = 0.90,
) -> ModalResult:
    """Return periods and identify distinct X/Y modes by effective mass."""
    ops = _ops()
    normalized_solver = solver.removeprefix("-")
    solver_used = normalized_solver
    solver_fallback = False
    if normalized_solver == "fullGenLapack":
        eigenvalues = np.atleast_1d(
            np.asarray(ops.eigen("-fullGenLapack", mode_count), dtype=float)
        )
    elif normalized_solver == "genBandArpack":
        try:
            eigenvalues = np.atleast_1d(
                np.asarray(
                    ops.eigen("-genBandArpack", mode_count),
                    dtype=float,
                )
            )
        except Exception:
            solver_used = "fullGenLapack"
            solver_fallback = True
            eigenvalues = np.atleast_1d(
                np.asarray(
                    ops.eigen("-fullGenLapack", mode_count),
                    dtype=float,
                )
            )
    else:
        raise ValueError(
            "eigen solver must be fullGenLapack or genBandArpack"
        )
    if eigenvalues.size < 2 or np.any(eigenvalues <= 0):
        raise RuntimeError("OpenSees returned invalid eigenvalues")
    periods = 2.0 * math.pi / np.sqrt(eigenvalues)
    if not 0.0 < minimum_cumulative_mass_ratio <= 1.0:
        raise ValueError(
            "minimum_cumulative_mass_ratio must lie in (0, 1]"
        )
    mass_x = np.asarray(
        [float(ops.nodeMass(node, 1)) for node in info.master_nodes]
    )
    mass_y = np.asarray(
        [float(ops.nodeMass(node, 2)) for node in info.master_nodes]
    )
    mass_rz = np.asarray(
        [float(ops.nodeMass(node, 6)) for node in info.master_nodes]
    )
    phi_x = np.asarray(
        [
            [
                float(ops.nodeEigenvector(node, mode, 1))
                for node in info.master_nodes
            ]
            for mode in range(1, len(periods) + 1)
        ]
    )
    phi_y = np.asarray(
        [
            [
                float(ops.nodeEigenvector(node, mode, 2))
                for node in info.master_nodes
            ]
            for mode in range(1, len(periods) + 1)
        ]
    )
    phi_rz = np.asarray(
        [
            [
                float(ops.nodeEigenvector(node, mode, 6))
                for node in info.master_nodes
            ]
            for mode in range(1, len(periods) + 1)
        ]
    )
    x_ratios, y_ratios = _directional_effective_modal_mass_ratios(
        phi_x,
        phi_y,
        phi_rz,
        mass_x,
        mass_y,
        mass_rz,
    )
    cumulative_x = float(np.sum(x_ratios))
    cumulative_y = float(np.sum(y_ratios))
    if cumulative_x > 1.0 + 1.0e-6 or cumulative_y > 1.0 + 1.0e-6:
        raise RuntimeError(
            "Cumulative directional effective modal mass exceeds 100%; "
            "check eigenvector orthogonality and nodal mass assignment"
        )
    if (
        cumulative_x + 1.0e-10 < minimum_cumulative_mass_ratio
        or cumulative_y + 1.0e-10 < minimum_cumulative_mass_ratio
    ):
        raise RuntimeError(
            "Extracted modes capture insufficient translational effective "
            f"mass: X={cumulative_x:.3%}, Y={cumulative_y:.3%}, required="
            f"{minimum_cumulative_mass_ratio:.3%}"
        )
    x_index, y_index = _select_distinct_directional_modes(
        x_ratios,
        y_ratios,
    )
    x_mode = x_index + 1
    y_mode = y_index + 1
    t_x = float(periods[x_mode - 1])
    t_y = float(periods[y_mode - 1])
    period_error = abs(t_x - t_y) / max((t_x + t_y) / 2.0, 1.0e-12)
    return ModalResult(
        periods_s=tuple(float(value) for value in periods),
        x_mode=x_mode,
        y_mode=y_mode,
        t_x_s=t_x,
        t_y_s=t_y,
        period_error=period_error,
        x_mode_effective_mass_ratio=float(x_ratios[x_index]),
        y_mode_effective_mass_ratio=float(y_ratios[y_index]),
        cumulative_x_effective_mass_ratio=cumulative_x,
        cumulative_y_effective_mass_ratio=cumulative_y,
        identification_method=(
            "directional_effective_modal_mass_distinct_assignment_v1"
        ),
        eigen_solver_requested=normalized_solver,
        eigen_solver_used=solver_used,
        eigen_solver_fallback=solver_fallback,
    )


def apply_rayleigh_damping(modal: ModalResult, damping_ratio: float) -> None:
    """Apply initial-stiffness Rayleigh damping using the two X/Y modes."""
    ops = _ops()
    omega_x = 2.0 * math.pi / modal.t_x_s
    omega_y = 2.0 * math.pi / modal.t_y_s
    if math.isclose(omega_x, omega_y, rel_tol=1.0e-6):
        # The symmetric frame has repeated modes; use the next available period.
        alternatives = [
            2.0 * math.pi / period
            for index, period in enumerate(modal.periods_s, start=1)
            if index not in {modal.x_mode, modal.y_mode}
            and period > 0
        ]
        omega_y = alternatives[0] if alternatives else 3.0 * omega_x
    alpha_m = 2.0 * damping_ratio * omega_x * omega_y / (omega_x + omega_y)
    beta_k_init = 2.0 * damping_ratio / (omega_x + omega_y)
    ops.rayleigh(alpha_m, 0.0, beta_k_init, 0.0)


def apply_damping(
    modal: ModalResult,
    damping_ratio: float,
    *,
    method: str = "modal",
) -> None:
    """Apply the explicitly configured damping model after eigen analysis."""
    ops = _ops()
    normalized = method.lower()
    if normalized == "modal":
        # Draft 4 specifies modal damping. OpenSees requires eigen() to have
        # been called first, which modal_properties() has already done. Supply
        # one value for every extracted mode; a single value damps only the
        # corresponding first mode in the OpenSees modal-damping vector.
        ops.modalDamping(
            *[float(damping_ratio) for _ in modal.periods_s]
        )
        return
    if normalized == "rayleigh_initial":
        apply_rayleigh_damping(modal, damping_ratio)
        return
    raise ValueError(f"Unsupported damping method: {method}")


def _story_drift_ratio(
    displacement_x_m: float,
    displacement_y_m: float,
    previous_x_m: float,
    previous_y_m: float,
    storey_height_m: float,
) -> float:
    """Draft 4 Eq. (3.21): maximum component, not vector-resultant drift."""
    if storey_height_m <= 0.0:
        raise ValueError("Storey height must be positive")
    return max(
        abs(displacement_x_m - previous_x_m),
        abs(displacement_y_m - previous_y_m),
    ) / storey_height_m


def maximum_interstorey_drift(info: ModelInfo) -> float:
    """Maximum over time caller, stories, and the two horizontal components."""
    ops = _ops()
    maximum = 0.0
    previous_x = 0.0
    previous_y = 0.0
    previous_z = 0.0
    for node, elevation in zip(info.master_nodes, info.floor_elevations_m):
        x = float(ops.nodeDisp(node, 1))
        y = float(ops.nodeDisp(node, 2))
        storey_height = elevation - previous_z
        drift = _story_drift_ratio(
            x,
            y,
            previous_x,
            previous_y,
            storey_height,
        )
        maximum = max(maximum, drift)
        previous_x = x
        previous_y = y
        previous_z = elevation
    return maximum
