"""Ground-motion discovery, validation, and response spectra."""

from __future__ import annotations

import csv
import json
import math
import re
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
from scipy.signal import resample_poly

from .constants import G_STD
from .db import initialize, transaction, upsert_many
from .io_utils import atomic_write_json, sha256_file, stable_hash


@dataclass(frozen=True)
class GroundMotionPair:
    pair_id: str
    source_set: str
    conditioning_period_s: float | None
    component_x_path: str
    component_y_path: str
    units: str
    dt_s: float
    npts: int
    duration_s: float
    pga_x_g: float
    pga_y_g: float
    sha256_x: str
    sha256_y: str
    physical_pair_hash: str
    raw_source_path: str | None
    raw_source_sha256: str | None
    source_metadata_json: dict[str, Any]
    valid: int
    validation_message: str


def load_acceleration(path: str | Path, units: str = "g") -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path}: expected at least two columns (time, acceleration)")
    time = data[:, 0]
    acceleration = data[:, 1]
    if units.lower() in {"m/s2", "m/s^2", "mps2"}:
        acceleration = acceleration / G_STD
    elif units.lower() in {"gal", "cm/s2", "cm/s^2"}:
        acceleration = acceleration / (G_STD * 100.0)
    elif units.lower() != "g":
        raise ValueError(f"Unsupported acceleration units: {units}")
    return time, acceleration


def acceleration_to_g(
    acceleration: np.ndarray,
    units: str,
) -> np.ndarray:
    """Convert an acceleration vector to g without modifying the raw source."""
    values = np.asarray(acceleration, dtype=float)
    normalized = units.lower()
    if normalized == "g":
        return values
    if normalized in {"m/s2", "m/s^2", "mps2"}:
        return values / G_STD
    if normalized in {"gal", "cm/s2", "cm/s^2"}:
        return values / (G_STD * 100.0)
    raise ValueError(f"Unsupported acceleration units: {units}")


def _validate_component(path: Path, units: str) -> dict[str, Any]:
    time, acceleration = load_acceleration(path, units)
    if len(time) < 2:
        raise ValueError(f"{path}: fewer than two samples")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(acceleration)):
        raise ValueError(f"{path}: non-finite value")
    increments = np.diff(time)
    if np.any(increments <= 0):
        raise ValueError(f"{path}: time values are not strictly increasing")
    dt = float(np.median(increments))
    if not np.allclose(increments, dt, rtol=1e-6, atol=1e-10):
        raise ValueError(f"{path}: non-uniform time step")
    return {
        "time": time,
        "acceleration_g": acceleration,
        "dt_s": dt,
        "npts": len(time),
        "duration_s": float(time[-1] - time[0]),
        "pga_g": float(np.max(np.abs(acceleration))),
        "baseline_diagnostics": _baseline_diagnostics(time, acceleration),
    }


def _baseline_diagnostics(
    time_s: np.ndarray,
    acceleration_g: np.ndarray,
) -> dict[str, float]:
    """Return transparent raw-record drift diagnostics without altering data."""
    time = np.asarray(time_s, dtype=float)
    acceleration = np.asarray(acceleration_g, dtype=float) * G_STD
    increments = np.diff(time)
    average_acceleration = 0.5 * (acceleration[1:] + acceleration[:-1])
    velocity = np.concatenate(
        ([0.0], np.cumsum(average_acceleration * increments))
    )
    average_velocity = 0.5 * (velocity[1:] + velocity[:-1])
    displacement = np.concatenate(
        ([0.0], np.cumsum(average_velocity * increments))
    )
    duration = max(float(time[-1] - time[0]), np.finfo(float).eps)
    return {
        "mean_acceleration_g": float(np.mean(acceleration_g)),
        "linear_trend_acceleration_g_per_s": float(
            np.polyfit(time - time[0], acceleration_g, 1)[0]
        ),
        "residual_velocity_m_s": float(velocity[-1]),
        "residual_displacement_m": float(displacement[-1]),
        "normalized_residual_velocity_g_s": float(
            velocity[-1] / (G_STD * duration)
        ),
    }


def _linear_detrend_acceleration(
    time_s: np.ndarray,
    acceleration_g: np.ndarray,
) -> np.ndarray:
    """Remove the least-squares constant and linear acceleration baseline."""
    time = np.asarray(time_s, dtype=float)
    acceleration = np.asarray(acceleration_g, dtype=float)
    if (
        time.ndim != 1
        or acceleration.ndim != 1
        or len(time) != len(acceleration)
        or len(time) < 2
    ):
        raise ValueError("Time and acceleration must be equal 1D vectors")
    relative_time = time - time[0]
    coefficients = np.polyfit(relative_time, acceleration, 1)
    corrected = acceleration - np.polyval(coefficients, relative_time)
    if not np.all(np.isfinite(corrected)):
        raise ValueError("Baseline correction produced non-finite values")
    return corrected


def _cosine_taper_acceleration(
    time_s: np.ndarray,
    acceleration_g: np.ndarray,
    taper_s: float,
) -> np.ndarray:
    """Apply equal leading/trailing raised-cosine tapers."""
    time = np.asarray(time_s, dtype=float)
    acceleration = np.asarray(acceleration_g, dtype=float)
    duration = float(time[-1] - time[0])
    if taper_s <= 0.0:
        return acceleration.copy()
    if 2.0 * taper_s >= duration:
        raise ValueError("Cosine taper must be shorter than half the record")
    relative_time = time - time[0]
    window = np.ones_like(acceleration)
    leading = relative_time < taper_s
    trailing = relative_time > duration - taper_s
    window[leading] = 0.5 * (
        1.0 - np.cos(np.pi * relative_time[leading] / taper_s)
    )
    window[trailing] = 0.5 * (
        1.0
        - np.cos(
            np.pi
            * (duration - relative_time[trailing])
            / taper_s
        )
    )
    return acceleration * window


def _resample_acceleration(
    time_s: np.ndarray,
    acceleration_g: np.ndarray,
    target_dt_s: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return a deterministic uniform record using polyphase filtering."""
    time = np.asarray(time_s, dtype=float)
    acceleration = np.asarray(acceleration_g, dtype=float)
    source_dt_s = float(np.median(np.diff(time)))
    if target_dt_s is None or math.isclose(
        source_dt_s,
        target_dt_s,
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        return time.copy(), acceleration.copy(), {
            "method": "not_required",
            "source_dt_s": source_dt_s,
            "target_dt_s": source_dt_s,
            "up": 1,
            "down": 1,
        }
    ratio = Fraction(source_dt_s / target_dt_s).limit_denominator(
        10000
    )
    resampled = resample_poly(
        acceleration,
        up=ratio.numerator,
        down=ratio.denominator,
        padtype="line",
    )
    duration_s = float(time[-1] - time[0])
    # Keep only samples whose target-grid time lies inside the source
    # duration.  Rounding can request one artificial endpoint when the
    # original duration is not an exact multiple of the target dt.
    target_count = int(
        math.floor(duration_s / target_dt_s + 1.0e-9)
    ) + 1
    if resampled.size < target_count:
        raise RuntimeError(
            "Polyphase resampling returned fewer values than required"
        )
    resampled = np.asarray(resampled[:target_count], dtype=float)
    resampled_time = (
        np.arange(target_count, dtype=float) * target_dt_s
    )
    return resampled_time, resampled, {
        "method": "scipy_resample_poly_v1",
        "source_dt_s": source_dt_s,
        "target_dt_s": float(target_dt_s),
        "up": int(ratio.numerator),
        "down": int(ratio.denominator),
    }


def _prepare_pair_for_analysis(
    pair: GroundMotionPair,
    config: dict[str, Any],
) -> GroundMotionPair:
    """Write deterministic, analysis-ready copies used by spectra and NLTHA."""
    if not pair.valid:
        return pair
    preprocessing = config["ground_motion"]["analysis_preprocessing"]
    method = str(preprocessing["method"])
    if method != "linear_least_squares_detrend_v1":
        raise ValueError(f"Unsupported ground-motion preprocessing: {method}")
    raw_x_path = Path(pair.component_x_path)
    raw_y_path = Path(pair.component_y_path)
    time_x, raw_x_g = load_acceleration(raw_x_path, pair.units)
    time_y, raw_y_g = load_acceleration(raw_y_path, pair.units)
    if not np.allclose(time_x, time_y, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{pair.pair_id}: raw X/Y time vectors differ")
    target_dt_value = preprocessing.get("target_dt_s")
    target_dt_s = (
        None if target_dt_value is None else float(target_dt_value)
    )
    pwsa_window = preprocessing.get("pwsa_window_s")
    pwsa_taper_s = float(preprocessing.get("pwsa_taper_s", 0.0))
    processing_sequence = ["linear_least_squares_detrend"]
    working_time = time_x
    working_x_g = raw_x_g
    working_y_g = raw_y_g
    applied_window: list[float] | None = None
    if pair.source_set.startswith("PWSA") and pwsa_window is not None:
        window_start_s, window_end_s = map(float, pwsa_window)
        mask = (
            (working_time >= window_start_s)
            & (working_time <= window_end_s)
        )
        if int(np.count_nonzero(mask)) < 2:
            raise ValueError(
                f"{pair.pair_id}: PWSA window contains too few samples"
            )
        working_time = working_time[mask] - window_start_s
        working_x_g = working_x_g[mask]
        working_y_g = working_y_g[mask]
        working_x_g = _cosine_taper_acceleration(
            working_time, working_x_g, pwsa_taper_s
        )
        working_y_g = _cosine_taper_acceleration(
            working_time, working_y_g, pwsa_taper_s
        )
        applied_window = [window_start_s, window_end_s]
        processing_sequence = [
            "pwsa_time_window",
            "raised_cosine_taper",
            "linear_least_squares_detrend",
        ]
    detrended_x_g = _linear_detrend_acceleration(
        working_time, working_x_g
    )
    detrended_y_g = _linear_detrend_acceleration(
        working_time, working_y_g
    )
    processed_time_x, processed_x_g, resampling_x = (
        _resample_acceleration(
            working_time, detrended_x_g, target_dt_s
        )
    )
    processed_time_y, processed_y_g, resampling_y = (
        _resample_acceleration(
            working_time, detrended_y_g, target_dt_s
        )
    )
    if not np.array_equal(processed_time_x, processed_time_y):
        raise ValueError(
            f"{pair.pair_id}: processed X/Y time vectors differ"
        )
    # A final detrend removes the minute endpoint residual introduced by
    # interpolation and guarantees the same baseline QA for every source dt.
    processed_x_g = _linear_detrend_acceleration(
        processed_time_x, processed_x_g
    )
    processed_y_g = _linear_detrend_acceleration(
        processed_time_y, processed_y_g
    )
    if target_dt_s is not None:
        processing_sequence.extend(
            [
                "polyphase_resample",
                "post_resample_linear_detrend",
            ]
        )
    processed_diagnostics = {
        "component_x": _baseline_diagnostics(
            processed_time_x, processed_x_g
        ),
        "component_y": _baseline_diagnostics(
            processed_time_y, processed_y_g
        ),
    }
    velocity_limit = float(
        preprocessing["maximum_abs_residual_velocity_m_s"]
    )
    displacement_limit = float(
        preprocessing["maximum_abs_residual_displacement_m"]
    )
    violations = []
    for component, diagnostics in processed_diagnostics.items():
        if (
            abs(float(diagnostics["residual_velocity_m_s"]))
            > velocity_limit
        ):
            violations.append(f"{component} residual velocity")
        if (
            abs(float(diagnostics["residual_displacement_m"]))
            > displacement_limit
        ):
            violations.append(f"{component} residual displacement")
    if violations:
        raise ValueError(
            f"{pair.pair_id}: processed baseline QA failed: "
            + ", ".join(violations)
        )
    processed_directory = (
        Path(config["_project_root"])
        / "data"
        / "ground_motion"
        / "processed"
        / str(preprocessing.get("processed_dataset_id", "analysis_v1"))
    )
    processed_directory.mkdir(parents=True, exist_ok=True)
    safe_pair_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", pair.pair_id)
    processed_paths = []
    for suffix, time, acceleration in (
        ("X", processed_time_x, processed_x_g),
        ("Y", processed_time_y, processed_y_g),
    ):
        output_path = processed_directory / f"{safe_pair_id}_{suffix}.txt"
        temporary_path = output_path.with_suffix(".txt.tmp")
        np.savetxt(
            temporary_path,
            np.column_stack((time, acceleration)),
            fmt=("%.9f", "%.12e"),
        )
        temporary_path.replace(output_path)
        processed_paths.append(output_path)
    processed_x_path, processed_y_path = processed_paths
    processed_sha_x = sha256_file(processed_x_path)
    processed_sha_y = sha256_file(processed_y_path)
    metadata = dict(pair.source_metadata_json)
    metadata["analysis_preprocessing"] = {
        "method": method,
        "raw_files_preserved": True,
        "raw_component_x_path": str(raw_x_path.absolute()),
        "raw_component_y_path": str(raw_y_path.absolute()),
        "raw_component_x_sha256": sha256_file(raw_x_path),
        "raw_component_y_sha256": sha256_file(raw_y_path),
        "processed_component_x_path": str(processed_x_path.absolute()),
        "processed_component_y_path": str(processed_y_path.absolute()),
        "processed_component_x_sha256": processed_sha_x,
        "processed_component_y_sha256": processed_sha_y,
        "processed_baseline_diagnostics": processed_diagnostics,
        "processing_sequence": processing_sequence,
        "processed_dataset_id": str(
            preprocessing.get("processed_dataset_id", "analysis_v1")
        ),
        "target_dt_s": target_dt_s,
        "pwsa_window_s": applied_window,
        "pwsa_taper_s": (
            pwsa_taper_s if applied_window is not None else None
        ),
        "resampling_component_x": resampling_x,
        "resampling_component_y": resampling_y,
        "original_dt_s": float(pair.dt_s),
        "original_npts": int(pair.npts),
        "original_duration_s": float(pair.duration_s),
        "processed_dt_s": float(
            np.median(np.diff(processed_time_x))
        ),
        "processed_npts": int(processed_time_x.size),
        "processed_duration_s": float(
            processed_time_x[-1] - processed_time_x[0]
        ),
        "maximum_abs_residual_velocity_m_s": velocity_limit,
        "maximum_abs_residual_displacement_m": displacement_limit,
    }
    processed_dt_s = float(np.median(np.diff(processed_time_x)))
    processed_npts = int(processed_time_x.size)
    processed_duration_s = float(
        processed_time_x[-1] - processed_time_x[0]
    )
    physical_pair_hash = stable_hash(
        {
            "sha256_x": processed_sha_x,
            "sha256_y": processed_sha_y,
            "dt_s": processed_dt_s,
            "npts": processed_npts,
            "units": "g",
            "analysis_preprocessing": method,
            "processing_sequence": processing_sequence,
        }
    )
    return replace(
        pair,
        component_x_path=str(processed_x_path.absolute()),
        component_y_path=str(processed_y_path.absolute()),
        units="g",
        pga_x_g=float(np.max(np.abs(processed_x_g))),
        pga_y_g=float(np.max(np.abs(processed_y_g))),
        dt_s=processed_dt_s,
        npts=processed_npts,
        duration_s=processed_duration_s,
        sha256_x=processed_sha_x,
        sha256_y=processed_sha_y,
        physical_pair_hash=physical_pair_hash,
        source_metadata_json=metadata,
        validation_message=(
            pair.validation_message
            + "; analysis copy passed deterministic linear-detrend QA"
        ),
    )


def _make_pair(
    pair_id: str,
    source_set: str,
    period: float | None,
    x_path: Path,
    y_path: Path,
    units: str,
    *,
    raw_source_path: Path | None = None,
    raw_source_sha256: str | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> GroundMotionPair:
    messages: list[str] = []
    valid = 1
    try:
        x = _validate_component(x_path, units)
        y = _validate_component(y_path, units)
        if x["npts"] != y["npts"]:
            raise ValueError("X/Y sample counts differ")
        if not np.allclose(x["time"], y["time"], rtol=0, atol=1e-10):
            raise ValueError("X/Y time vectors differ")
        messages.append("valid paired components")
        if source_metadata and source_metadata.get(
            "pairing_verified_from_source_manifest"
        ):
            messages.append(
                "X/Y identity verified from source ReadMe manifest"
            )
    except Exception as exc:
        valid = 0
        messages.append(str(exc))
        x = {
            "dt_s": math.nan,
            "npts": 0,
            "duration_s": math.nan,
            "pga_g": math.nan,
        }
        y = {"pga_g": math.nan}
    sha256_x = sha256_file(x_path)
    sha256_y = sha256_file(y_path)
    metadata = dict(source_metadata or {})
    metadata["baseline_diagnostics"] = {
        "component_x": x.get("baseline_diagnostics"),
        "component_y": y.get("baseline_diagnostics"),
        "interpretation": (
            "Raw-record screening values only; the source files are never "
            "modified automatically. Large residuals require documented "
            "engineering review before production IDA."
        ),
    }
    physical_pair_hash = stable_hash(
        {
            "sha256_x": sha256_x,
            "sha256_y": sha256_y,
            "dt_s": float(x["dt_s"]),
            "npts": int(x["npts"]),
            "units": str(units).lower(),
        }
    )
    return GroundMotionPair(
        pair_id=pair_id,
        source_set=source_set,
        conditioning_period_s=period,
        component_x_path=str(x_path.absolute()),
        component_y_path=str(y_path.absolute()),
        # ``units`` describes the values stored in the component files.
        # Validation metrics above are converted to g independently.
        units=str(units),
        dt_s=float(x["dt_s"]),
        npts=int(x["npts"]),
        duration_s=float(x["duration_s"]),
        pga_x_g=float(x["pga_g"]),
        pga_y_g=float(y["pga_g"]),
        sha256_x=sha256_x,
        sha256_y=sha256_y,
        physical_pair_hash=physical_pair_hash,
        raw_source_path=(
            str(raw_source_path.absolute()) if raw_source_path else None
        ),
        raw_source_sha256=raw_source_sha256,
        source_metadata_json=metadata,
        valid=valid,
        validation_message="; ".join(messages),
    )


def _excel_column_number(column: str) -> int:
    result = 0
    for character in column.strip().upper():
        if not "A" <= character <= "Z":
            raise ValueError(f"Invalid Excel column: {column}")
        result = result * 26 + ord(character) - 64
    return result


def _xlsx_sheet_target(
    archive: zipfile.ZipFile,
    sheet_name: str,
) -> str:
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    document_relationship = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    )
    package_relationship = (
        "http://schemas.openxmlformats.org/package/2006/relationships"
    )
    relationships_root = ET.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    relationships = {
        relation.attrib["Id"]: relation.attrib["Target"].lstrip("/")
        for relation in relationships_root.findall(
            f"{{{package_relationship}}}Relationship"
        )
    }
    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    for sheet in workbook_root.findall(f".//{{{main}}}sheet"):
        if sheet.attrib["name"] != sheet_name:
            continue
        relationship_id = sheet.attrib[f"{{{document_relationship}}}id"]
        target = relationships[relationship_id]
        return target if target.startswith("xl/") else f"xl/{target}"
    available = [
        sheet.attrib["name"]
        for sheet in workbook_root.findall(f".//{{{main}}}sheet")
    ]
    raise ValueError(
        f"Worksheet {sheet_name!r} not found; available sheets: {available}"
    )


def extract_year_2568_pair(
    config: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    """Extract PWSA EW/NS columns to auditable processed text files."""
    ground_motion = config["ground_motion"]
    workbook_path = Path(ground_motion["year_2568_workbook"])
    if not workbook_path.is_file():
        raise FileNotFoundError(
            f"Ground Motion Data workbook not found: {workbook_path}"
        )
    sheet_name = str(ground_motion["year_2568_sheet"])
    time_column = _excel_column_number(
        str(ground_motion["year_2568_time_column"])
    )
    x_column = _excel_column_number(
        str(ground_motion["year_2568_x_column"])
    )
    y_column = _excel_column_number(
        str(ground_motion["year_2568_y_column"])
    )
    data_start_row = int(ground_motion["year_2568_data_start_row"])
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    time_values: list[float] = []
    x_values: list[float] = []
    y_values: list[float] = []

    with zipfile.ZipFile(workbook_path) as archive:
        sheet_target = _xlsx_sheet_target(archive, sheet_name)
        with archive.open(sheet_target) as stream:
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag != f"{{{main}}}row":
                    continue
                row_number = int(element.attrib.get("r", "0"))
                if row_number < data_start_row:
                    element.clear()
                    continue
                numeric: dict[int, float] = {}
                for cell in element.findall(f"{{{main}}}c"):
                    reference = cell.attrib.get("r", "")
                    match = re.match(r"([A-Z]+)", reference)
                    if not match:
                        continue
                    column_number = _excel_column_number(match.group(1))
                    if column_number not in {
                        time_column,
                        x_column,
                        y_column,
                    }:
                        continue
                    value = cell.find(f"{{{main}}}v")
                    if value is None or value.text is None:
                        continue
                    numeric[column_number] = float(value.text)
                if {
                    time_column,
                    x_column,
                    y_column,
                }.issubset(numeric):
                    time_values.append(numeric[time_column])
                    x_values.append(numeric[x_column])
                    y_values.append(numeric[y_column])
                element.clear()
    if len(time_values) < 2:
        raise ValueError(
            f"{workbook_path}/{sheet_name}: fewer than two numeric rows"
        )
    time = np.asarray(time_values, dtype=float)
    source_units = str(ground_motion["year_2568_units"])
    x_acceleration = acceleration_to_g(
        np.asarray(x_values, dtype=float),
        source_units,
    )
    y_acceleration = acceleration_to_g(
        np.asarray(y_values, dtype=float),
        source_units,
    )
    increments = np.diff(time)
    dt = float(np.median(increments))
    if (
        not np.all(np.isfinite(time))
        or not np.all(np.isfinite(x_acceleration))
        or not np.all(np.isfinite(y_acceleration))
        or np.any(increments <= 0)
        or not np.allclose(increments, dt, rtol=1.0e-6, atol=1.0e-10)
    ):
        raise ValueError(
            f"{workbook_path}/{sheet_name}: invalid numeric values or time step"
        )

    processed_directory = (
        Path(config["_project_root"])
        / "data"
        / "ground_motion"
        / "processed"
    )
    processed_directory.mkdir(parents=True, exist_ok=True)
    x_path = processed_directory / "PWSA_2568_EW.txt"
    y_path = processed_directory / "PWSA_2568_NS.txt"
    x_temporary = x_path.with_suffix(".txt.tmp")
    y_temporary = y_path.with_suffix(".txt.tmp")
    np.savetxt(
        x_temporary,
        np.column_stack((time, x_acceleration)),
        fmt=("%.9f", "%.12e"),
    )
    np.savetxt(
        y_temporary,
        np.column_stack((time, y_acceleration)),
        fmt=("%.9f", "%.12e"),
    )
    x_temporary.replace(x_path)
    y_temporary.replace(y_path)
    metadata = {
        "workbook_path": str(workbook_path.absolute()),
        "workbook_sha256": sha256_file(workbook_path),
        "sheet": sheet_name,
        "time_column": ground_motion["year_2568_time_column"],
        "x_column": ground_motion["year_2568_x_column"],
        "y_column": ground_motion["year_2568_y_column"],
        "data_start_row": data_start_row,
        "source_units": source_units,
        "processed_units": "g",
        "dt_s": dt,
        "npts": len(time),
        "duration_s": float(time[-1] - time[0]),
        "x_label": "East-West",
        "y_label": "North-South",
    }
    return x_path, y_path, metadata


def _read_cms_pairing_manifest(
    directory: Path,
    component_files: list[Path],
) -> tuple[Path, list[dict[str, Any]]]:
    """Read the source-supplied CMS X/Y manifest and return verified pairs.

    Pairing is never inferred from adjacent filenames alone.  Each period
    directory must contain exactly one ReadMe*.txt whose entries explicitly
    assign every numbered component to a logical record and X/Y orientation,
    for example ``EQ_1 = EQ1_X`` and ``EQ_2 = EQ1_Y``.
    """
    readme_candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.name.lower().startswith("readme")
        and path.suffix.lower() == ".txt"
    )
    if len(readme_candidates) != 1:
        raise ValueError(
            f"{directory}: expected exactly one source ReadMe*.txt pairing "
            f"manifest; found {len(readme_candidates)}"
        )
    readme_path = readme_candidates[0]
    text: str | None = None
    decoding_errors: list[str] = []
    for encoding in ("utf-8-sig", "cp874", "cp1252"):
        try:
            text = readme_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            decoding_errors.append(f"{encoding}: {exc}")
    if text is None:
        raise ValueError(
            f"{readme_path}: cannot decode pairing manifest; "
            + "; ".join(decoding_errors)
        )

    entries: dict[int, dict[str, Any]] = {}
    pattern = re.compile(
        r"^\s*EQ_(\d+)\s*=\s*EQ(\d+)_([XY])\s*$",
        flags=re.IGNORECASE,
    )
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = pattern.match(line)
        if not match:
            continue
        component_number = int(match.group(1))
        logical_record_number = int(match.group(2))
        orientation = match.group(3).upper()
        if component_number in entries:
            raise ValueError(
                f"{readme_path}: duplicate EQ_{component_number} assignment"
            )
        entries[component_number] = {
            "component_number": component_number,
            "logical_record_number": logical_record_number,
            "orientation": orientation,
            "source_line_number": line_number,
            "source_entry": line.strip(),
        }

    files_by_number: dict[int, Path] = {}
    for path in component_files:
        match = re.search(r"EQ_(\d+)$", path.stem)
        if not match:
            raise ValueError(f"{path}: component number not found in filename")
        component_number = int(match.group(1))
        if component_number in files_by_number:
            raise ValueError(
                f"{directory}: duplicate component file EQ_{component_number}"
            )
        files_by_number[component_number] = path

    file_numbers = set(files_by_number)
    manifest_numbers = set(entries)
    if file_numbers != manifest_numbers:
        missing_manifest = sorted(file_numbers - manifest_numbers)
        missing_files = sorted(manifest_numbers - file_numbers)
        raise ValueError(
            f"{readme_path}: manifest/file mismatch; components without "
            f"manifest={missing_manifest}, assignments without files="
            f"{missing_files}"
        )

    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for entry in entries.values():
        logical_number = int(entry["logical_record_number"])
        orientation = str(entry["orientation"])
        orientation_entries = grouped.setdefault(logical_number, {})
        if orientation in orientation_entries:
            raise ValueError(
                f"{readme_path}: logical EQ{logical_number} has duplicate "
                f"{orientation} components"
            )
        orientation_entries[orientation] = entry

    verified_pairs: list[dict[str, Any]] = []
    for logical_number in sorted(grouped):
        orientations = grouped[logical_number]
        if set(orientations) != {"X", "Y"}:
            raise ValueError(
                f"{readme_path}: logical EQ{logical_number} must have exactly "
                "one X and one Y component"
            )
        x_entry = orientations["X"]
        y_entry = orientations["Y"]
        x_number = int(x_entry["component_number"])
        y_number = int(y_entry["component_number"])
        if y_number != x_number + 1:
            raise ValueError(
                f"{readme_path}: logical EQ{logical_number} is source-mapped "
                f"to X=EQ_{x_number}, Y=EQ_{y_number}; expected adjacent "
                "X-then-Y component numbers"
            )
        verified_pairs.append(
            {
                "logical_record_number": logical_number,
                "x_number": x_number,
                "y_number": y_number,
                "x_path": files_by_number[x_number],
                "y_path": files_by_number[y_number],
                "x_source_entry": str(x_entry["source_entry"]),
                "y_source_entry": str(y_entry["source_entry"]),
                "x_source_line_number": int(
                    x_entry["source_line_number"]
                ),
                "y_source_line_number": int(
                    y_entry["source_line_number"]
                ),
            }
        )
    return readme_path, verified_pairs


def discover_cms_pairs(cms_root: str | Path) -> list[GroundMotionPair]:
    root = Path(cms_root)
    if not root.is_dir():
        raise FileNotFoundError(f"CMS root not found: {root}")
    result: list[GroundMotionPair] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", directory.name)
        if not match:
            continue
        period = float(match.group(1))
        files = sorted(
            directory.glob("Zone5_*_EQ_*.txt"),
            key=lambda path: int(re.search(r"EQ_(\d+)", path.stem).group(1)),
        )
        if not files:
            raise ValueError(f"{directory}: no CMS component files found")
        readme_path, source_verified_pairs = _read_cms_pairing_manifest(
            directory,
            files,
        )
        readme_sha256 = sha256_file(readme_path)
        for source_pair in source_verified_pairs:
            pair_number = int(source_pair["logical_record_number"])
            pair_id = f"CMS-T{period:0.1f}-P{pair_number:02d}"
            x_number = int(source_pair["x_number"])
            y_number = int(source_pair["y_number"])
            result.append(
                _make_pair(
                    pair_id,
                    "CMS_ZONE5",
                    period,
                    Path(source_pair["x_path"]),
                    Path(source_pair["y_path"]),
                    "g",
                    raw_source_path=readme_path,
                    raw_source_sha256=readme_sha256,
                    source_metadata={
                        "pairing_rule": "source ReadMe mapping plus adjacency",
                        "pairing_verified_from_source_manifest": True,
                        "pairing_authority_path": str(
                            readme_path.absolute()
                        ),
                        "pairing_authority_sha256": readme_sha256,
                        "logical_record_number": pair_number,
                        "component_x_number": x_number,
                        "component_y_number": y_number,
                        "component_x_label": f"EQ{pair_number}_X",
                        "component_y_label": f"EQ{pair_number}_Y",
                        "component_x_source_entry": source_pair[
                            "x_source_entry"
                        ],
                        "component_y_source_entry": source_pair[
                            "y_source_entry"
                        ],
                        "component_x_source_line_number": source_pair[
                            "x_source_line_number"
                        ],
                        "component_y_source_line_number": source_pair[
                            "y_source_line_number"
                        ],
                        "adjacent_x_then_y_verified": (
                            y_number == x_number + 1
                        ),
                    },
                )
            )
    return result


def discover_all_pairs(config: dict[str, Any]) -> list[GroundMotionPair]:
    gm = config["ground_motion"]
    pairs = discover_cms_pairs(gm["cms_root"])
    workbook_path = gm.get("year_2568_workbook")
    if workbook_path:
        x_path, y_path, metadata = extract_year_2568_pair(config)
        pairs.append(
            _make_pair(
                "PWSA-2568-MANDALAY",
                "PWSA_2568_WORKBOOK",
                None,
                x_path,
                y_path,
                "g",
                raw_source_path=Path(workbook_path),
                raw_source_sha256=metadata["workbook_sha256"],
                source_metadata=metadata,
            )
        )
        return pairs
    pwsa_x = gm.get("pwsa_x")
    pwsa_y = gm.get("pwsa_y")
    if bool(pwsa_x) != bool(pwsa_y):
        raise ValueError("Both pwsa_x and pwsa_y must be supplied together")
    if pwsa_x and pwsa_y:
        pairs.append(
            _make_pair(
                "PWSA-MANDALAY-2025",
                "PWSA_MANDALAY",
                None,
                Path(pwsa_x),
                Path(pwsa_y),
                gm.get("pwsa_units", "g"),
                raw_source_path=None,
                source_metadata={
                    "source": "explicit processed component paths"
                },
            )
        )
    return pairs


def validate_records(config: dict[str, Any]) -> dict[str, Any]:
    initialize(config["database_path"])
    pairs = [
        _prepare_pair_for_analysis(pair, config)
        for pair in discover_all_pairs(config)
    ]
    if not pairs:
        raise RuntimeError("No ground-motion pairs were discovered")
    rows = [asdict(pair) for pair in pairs]
    from .db import connect

    with connect(config["database_path"]) as connection:
        existing = {
            str(row["pair_id"]): dict(row)
            for row in connection.execute(
                "SELECT pair_id, sha256_x, sha256_y, raw_source_sha256, "
                "dt_s, npts, valid "
                "FROM ground_motion_catalog"
            )
        }
    current_ids = {str(row["pair_id"]) for row in rows}
    changed_pair_ids = {
        str(row["pair_id"])
        for row in rows
        if str(row["pair_id"]) not in existing
        or any(
            existing[str(row["pair_id"])][key] != row[key]
            for key in (
                "sha256_x",
                "sha256_y",
                "raw_source_sha256",
                "dt_s",
                "npts",
                "valid",
            )
        )
    }
    changed_pair_ids.update(
        pair_id
        for pair_id, row in existing.items()
        if bool(row["valid"]) and pair_id not in current_ids
    )
    with transaction(config["database_path"]) as connection:
        if changed_pair_ids:
            marks = ",".join("?" for _ in changed_pair_ids)
            active_capacity_count = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM ida_capacities c
                    JOIN building_catalog b USING(building_id)
                    WHERE b.selected=1 AND b.valid=1
                      AND c.pair_id IN ({marks})
                    """,
                    sorted(changed_pair_ids),
                ).fetchone()[0]
            )
            if active_capacity_count:
                raise RuntimeError(
                    "Ground-motion files/checksums changed after production "
                    "IDA capacities were created for the active queue. "
                    "Archive/reset the affected production dataset first."
                )
            affected_buildings = [
                str(row[0])
                for row in connection.execute(
                    f"""
                    SELECT DISTINCT building_id
                    FROM building_ground_motion_selection
                    WHERE pair_id IN ({marks})
                    """,
                    sorted(changed_pair_ids),
                )
            ]
            if affected_buildings:
                building_marks = ",".join("?" for _ in affected_buildings)
                connection.execute(
                    f"DELETE FROM fragility_targets "
                    f"WHERE building_id IN ({building_marks})",
                    affected_buildings,
                )
                connection.execute(
                    f"DELETE FROM ida_capacities "
                    f"WHERE building_id IN ({building_marks}) "
                    f"AND pair_id IN ({marks})",
                    [*affected_buildings, *sorted(changed_pair_ids)],
                )
                connection.execute(
                    f"DELETE FROM ida_runs "
                    f"WHERE building_id IN ({building_marks}) "
                    f"AND pair_id IN ({marks})",
                    [*affected_buildings, *sorted(changed_pair_ids)],
                )
            connection.execute("DELETE FROM ml_runs")
            connection.execute("DELETE FROM ml_split_manifest")
            connection.execute("DELETE FROM ml_split")
        # The current config is authoritative. Historical rows may remain for
        # foreign-key provenance, but cannot silently satisfy the active-data gate.
        connection.execute(
            """
            UPDATE ground_motion_catalog
            SET valid=0,
                validation_message='not present in current ground-motion config'
            """
        )
        upsert_many(connection, "ground_motion_catalog", rows, ("pair_id",))

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ground_motion_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid_pairs = sum(pair.valid for pair in pairs)
    cms_pairs = sum(pair.source_set == "CMS_ZONE5" for pair in pairs)
    cms_source_verified_pairs = sum(
        pair.source_set == "CMS_ZONE5"
        and bool(
            pair.source_metadata_json.get(
                "pairing_verified_from_source_manifest"
            )
        )
        and bool(
            pair.source_metadata_json.get("adjacent_x_then_y_verified")
        )
        for pair in pairs
    )
    pwsa_present = any(
        pair.source_set.startswith("PWSA") for pair in pairs
    )
    expected_by_period = {
        float(period): int(count)
        for period, count in config["ground_motion"]
        .get("cms_expected_pairs_by_period", {})
        .items()
    }
    observed_by_period = {
        period: sum(
            pair.source_set == "CMS_ZONE5"
            and pair.conditioning_period_s == period
            and bool(pair.valid)
            for pair in pairs
        )
        for period in expected_by_period
    }
    cms_family_counts_valid = all(
        observed_by_period[period] == expected
        for period, expected in expected_by_period.items()
    )
    expected_pwsa = int(
        config["ground_motion"].get("expected_pwsa_pair_count", 1)
    )
    valid_pwsa_count = sum(
        bool(pair.valid) and pair.source_set.startswith("PWSA")
        for pair in pairs
    )
    duplicate_physical_groups: dict[str, list[str]] = {}
    for pair in pairs:
        if pair.valid:
            duplicate_physical_groups.setdefault(
                pair.physical_pair_hash, []
            ).append(pair.pair_id)
    duplicate_physical_groups = {
        pair_hash: sorted(pair_ids)
        for pair_hash, pair_ids in duplicate_physical_groups.items()
        if len(pair_ids) > 1
    }
    unique_physical_pair_count = len(
        {
            pair.physical_pair_hash
            for pair in pairs
            if pair.valid
        }
    )
    processed_diagnostics = [
        {
            "pair_id": pair.pair_id,
            "component": component,
            **diagnostics,
        }
        for pair in pairs
        if pair.valid
        for component, diagnostics in pair.source_metadata_json[
            "analysis_preprocessing"
        ]["processed_baseline_diagnostics"].items()
    ]
    maximum_processed_residual_velocity = max(
        abs(float(row["residual_velocity_m_s"]))
        for row in processed_diagnostics
    )
    maximum_processed_residual_displacement = max(
        abs(float(row["residual_displacement_m"]))
        for row in processed_diagnostics
    )
    cms_pairing_evidence = [
        {
            "pair_id": pair.pair_id,
            "conditioning_period_s": pair.conditioning_period_s,
            "pairing_authority_path": pair.source_metadata_json.get(
                "pairing_authority_path"
            ),
            "pairing_authority_sha256": pair.source_metadata_json.get(
                "pairing_authority_sha256"
            ),
            "component_x_path": pair.component_x_path,
            "component_x_source_entry": pair.source_metadata_json.get(
                "component_x_source_entry"
            ),
            "component_y_path": pair.component_y_path,
            "component_y_source_entry": pair.source_metadata_json.get(
                "component_y_source_entry"
            ),
            "time_vector_and_sample_count_match": bool(pair.valid),
        }
        for pair in pairs
        if pair.source_set == "CMS_ZONE5"
    ]
    report = {
        "pair_count": len(pairs),
        "valid_pair_count": valid_pairs,
        "unique_physical_pair_count": unique_physical_pair_count,
        "duplicate_physical_pair_group_count": len(
            duplicate_physical_groups
        ),
        "duplicate_physical_pair_groups": duplicate_physical_groups,
        "analysis_preprocessing_method": config["ground_motion"][
            "analysis_preprocessing"
        ]["method"],
        "raw_component_files_preserved": True,
        "maximum_processed_abs_residual_velocity_m_s": (
            maximum_processed_residual_velocity
        ),
        "maximum_processed_abs_residual_displacement_m": (
            maximum_processed_residual_displacement
        ),
        "cms_pair_count": cms_pairs,
        "pwsa_present": pwsa_present,
        "batch_ready": (
            valid_pairs == len(pairs)
            and cms_family_counts_valid
            and cms_source_verified_pairs == cms_pairs
            and valid_pwsa_count == expected_pwsa
        ),
        "cms_source_pairing_verified": (
            cms_source_verified_pairs == cms_pairs
        ),
        "cms_source_verified_pair_count": cms_source_verified_pairs,
        "cms_pairing_authorities": sorted(
            {
                str(
                    pair.source_metadata_json.get(
                        "pairing_authority_path"
                    )
                )
                for pair in pairs
                if pair.source_set == "CMS_ZONE5"
            }
        ),
        "cms_pairing_evidence": cms_pairing_evidence,
        "cms_expected_pairs_by_period": expected_by_period,
        "cms_valid_pairs_by_period": observed_by_period,
        "cms_family_counts_valid": cms_family_counts_valid,
        "expected_pwsa_pair_count": expected_pwsa,
        "valid_pwsa_pair_count": valid_pwsa_count,
        "cms_periods_s": sorted(
            {
                pair.conditioning_period_s
                for pair in pairs
                if pair.source_set == "CMS_ZONE5"
                and pair.conditioning_period_s is not None
            }
        ),
        "dt_s_values": sorted(
            {round(pair.dt_s, 9) for pair in pairs if pair.valid}
        ),
        "npts_values": sorted({pair.npts for pair in pairs if pair.valid}),
        "duration_s_range": [
            min(pair.duration_s for pair in pairs if pair.valid),
            max(pair.duration_s for pair in pairs if pair.valid),
        ],
        "pga_g_range": [
            min(
                min(pair.pga_x_g, pair.pga_y_g)
                for pair in pairs
                if pair.valid
            ),
            max(
                max(pair.pga_x_g, pair.pga_y_g)
                for pair in pairs
                if pair.valid
            ),
        ],
        "csv_path": str(csv_path),
    }
    report_path = output_dir / "ground_motion_validation.json"
    atomic_write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def spectral_acceleration_g(
    acceleration_g: np.ndarray,
    dt_s: float,
    period_s: float,
    damping_ratio: float = 0.05,
) -> float:
    """5%-damped pseudo spectral acceleration via average-acceleration Newmark."""
    if period_s <= 0:
        raise ValueError("period_s must be positive")
    acceleration = np.asarray(acceleration_g, dtype=float) * G_STD
    omega = 2.0 * math.pi / period_s
    stiffness = omega**2
    damping = 2.0 * damping_ratio * omega
    beta = 0.25
    gamma = 0.5
    a0 = 1.0 / (beta * dt_s**2)
    a1 = gamma / (beta * dt_s)
    a2 = 1.0 / (beta * dt_s)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt_s * (gamma / (2.0 * beta) - 1.0)
    effective_stiffness = stiffness + a0 + a1 * damping

    displacement = 0.0
    velocity = 0.0
    relative_acceleration = -acceleration[0] - damping * velocity - stiffness * displacement
    maximum_displacement = abs(displacement)
    for ground_acceleration in acceleration[1:]:
        effective_load = (
            -ground_acceleration
            + a0 * displacement
            + a2 * velocity
            + a3 * relative_acceleration
            + damping
            * (
                a1 * displacement
                + a4 * velocity
                + a5 * relative_acceleration
            )
        )
        new_displacement = effective_load / effective_stiffness
        new_relative_acceleration = (
            a0 * (new_displacement - displacement)
            - a2 * velocity
            - a3 * relative_acceleration
        )
        new_velocity = (
            velocity
            + dt_s
            * (
                (1.0 - gamma) * relative_acceleration
                + gamma * new_relative_acceleration
            )
        )
        displacement = new_displacement
        velocity = new_velocity
        relative_acceleration = new_relative_acceleration
        maximum_displacement = max(maximum_displacement, abs(displacement))
    return omega**2 * maximum_displacement / G_STD


@lru_cache(maxsize=4096)
def _cached_pair_sa_geomean_g(
    component_x_path: str,
    component_y_path: str,
    units: str,
    period_s: float,
    sha256_x: str,
    sha256_y: str,
) -> float:
    """Compute a record-pair Sa once per immutable file/period signature.

    IDA changes only the scale factor between target IM levels. Recomputing
    the unscaled response spectrum for every target is numerically redundant
    and is particularly expensive for the 600-s PWSA record. Component
    checksums are part of the cache key so a changed source file cannot reuse
    an earlier spectral value.
    """
    del sha256_x, sha256_y  # Included in the immutable cache key.
    time_x, accel_x = load_acceleration(component_x_path, units)
    time_y, accel_y = load_acceleration(component_y_path, units)
    dt = float(np.median(np.diff(time_x)))
    if not np.allclose(time_x, time_y, rtol=0, atol=1e-10):
        raise ValueError("X/Y time vectors differ")
    sa_x = spectral_acceleration_g(accel_x, dt, period_s)
    sa_y = spectral_acceleration_g(accel_y, dt, period_s)
    if sa_x <= 0 or sa_y <= 0:
        raise ValueError("non-positive Sa")
    return math.sqrt(sa_x * sa_y)


def pair_sa_geomean_g(pair: dict[str, Any], period_s: float) -> float:
    try:
        return _cached_pair_sa_geomean_g(
            str(Path(pair["component_x_path"]).absolute()),
            str(Path(pair["component_y_path"]).absolute()),
            str(pair["units"]),
            float(period_s),
            str(pair.get("sha256_x") or ""),
            str(pair.get("sha256_y") or ""),
        )
    except ValueError as exc:
        raise ValueError(f"{pair['pair_id']}: {exc}") from exc


def select_cms_periods(
    t1_s: float,
    available_periods_s: Iterable[float],
    midpoint_blend_fraction: float = 0.10,
) -> tuple[list[float], str]:
    """Select the nearest CMS family, or both families near a midpoint.

    A blend fraction of 0.10 reproduces the user's example: when only 1.0 s
    and 2.0 s CMS families bracket the building, T1=1.4–1.6 s selects both.
    """
    periods = sorted({float(period) for period in available_periods_s})
    if not periods:
        raise ValueError("No CMS conditioning periods are available")
    if t1_s <= 0:
        raise ValueError("T1 must be positive")
    if not 0 <= midpoint_blend_fraction < 0.5:
        raise ValueError(
            "midpoint_blend_fraction must be between 0 and 0.5"
        )
    for lower, upper in zip(periods[:-1], periods[1:]):
        midpoint = (lower + upper) / 2.0
        half_band = midpoint_blend_fraction * (upper - lower)
        if midpoint - half_band <= t1_s <= midpoint + half_band:
            return (
                [lower, upper],
                f"T1={t1_s:.6g}s lies in midpoint band "
                f"[{midpoint-half_band:.6g},{midpoint+half_band:.6g}]s "
                f"between CMS T={lower:g}s and T={upper:g}s",
            )
    nearest = min(periods, key=lambda period: (abs(period - t1_s), period))
    return (
        [nearest],
        f"CMS T={nearest:g}s is nearest to T1={t1_s:.6g}s",
    )


def _deduplicate_selected_motions(
    motions: list[dict[str, Any]],
    *,
    t1_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep one deterministic representative of each physical X–Y pair."""
    ranked = sorted(
        motions,
        key=lambda motion: (
            0
            if str(motion["source_set"]).startswith("PWSA")
            else 1,
            (
                abs(float(motion["conditioning_period_s"]) - float(t1_s))
                if motion.get("conditioning_period_s") is not None
                else 0.0
            ),
            str(motion["pair_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    representative_by_hash: dict[str, str] = {}
    for motion in ranked:
        pair_hash = str(
            motion.get("physical_pair_hash")
            or stable_hash(
                {
                    "sha256_x": motion.get("sha256_x"),
                    "sha256_y": motion.get("sha256_y"),
                    "dt_s": motion.get("dt_s"),
                    "npts": motion.get("npts"),
                    "units": motion.get("units"),
                }
            )
        )
        if pair_hash in representative_by_hash:
            omitted.append(
                {
                    "omitted_pair_id": str(motion["pair_id"]),
                    "kept_pair_id": representative_by_hash[pair_hash],
                    "physical_pair_hash": pair_hash,
                }
            )
            continue
        representative_by_hash[pair_hash] = str(motion["pair_id"])
        selected.append(motion)
    return selected, omitted


def select_motions_for_t1(
    motions: list[dict[str, Any]],
    *,
    t1_s: float,
    midpoint_blend_fraction: float,
    include_pwsa: bool,
) -> tuple[list[dict[str, Any]], list[float], str, list[dict[str, Any]]]:
    """Apply the production CMS-family and physical-deduplication policy.

    Keeping this policy in one function prevents supporting studies such as
    damping sensitivity from silently using a different record family than
    the production IDA workflow.
    """
    cms_periods = sorted(
        {
            float(motion["conditioning_period_s"])
            for motion in motions
            if motion["source_set"] == "CMS_ZONE5"
            and motion["conditioning_period_s"] is not None
        }
    )
    if not cms_periods:
        raise RuntimeError("No valid CMS period families are available")
    periods, reason = select_cms_periods(
        float(t1_s),
        cms_periods,
        float(midpoint_blend_fraction),
    )
    cms_motions = [
        motion
        for motion in motions
        if motion["source_set"] == "CMS_ZONE5"
        and float(motion["conditioning_period_s"]) in periods
    ]
    pwsa_motions = [
        motion
        for motion in motions
        if include_pwsa and str(motion["source_set"]).startswith("PWSA")
    ]
    selected_motions, duplicate_omissions = _deduplicate_selected_motions(
        [*cms_motions, *pwsa_motions],
        t1_s=float(t1_s),
    )
    return selected_motions, periods, reason, duplicate_omissions


def build_ground_motion_selection(
    config: dict[str, Any],
    *,
    building_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Persist the record families applicable to each analysed Building ID."""
    initialize(config["database_path"])
    from .db import connect

    with connect(config["database_path"]) as connection:
        parameters: list[Any] = []
        building_sql = (
            "SELECT b.building_id, b.model_hash, b.queue_rank, "
            "s.t1_s, s.analysis_signature "
            "FROM building_catalog b JOIN spo_features s USING(building_id) "
            "WHERE b.selected=1 AND b.valid=1 AND s.valid=1"
        )
        if building_ids:
            marks = ",".join("?" for _ in building_ids)
            building_sql += f" AND b.building_id IN ({marks})"
            parameters.extend(building_ids)
        building_sql += " ORDER BY b.queue_rank"
        buildings = [
            dict(row)
            for row in connection.execute(building_sql, parameters)
        ]
        motions = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM ground_motion_catalog "
                "WHERE valid=1 ORDER BY source_set, pair_id"
            )
        ]
    from .spo import _spo_analysis_signature

    buildings = [
        building
        for building in buildings
        if building.get("analysis_signature")
        == _spo_analysis_signature(building, config)
    ]
    cms_periods = sorted(
        {
            float(motion["conditioning_period_s"])
            for motion in motions
            if motion["source_set"] == "CMS_ZONE5"
            and motion["conditioning_period_s"] is not None
        }
    )
    if not cms_periods:
        raise RuntimeError("No valid CMS period families are available")
    pwsa_motions = [
        motion
        for motion in motions
        if str(motion["source_set"]).startswith("PWSA")
    ]
    include_pwsa = bool(
        config["ground_motion"].get(
            "include_year_2568_for_every_building", True
        )
    )
    blend_fraction = float(
        config["ground_motion"]["cms_midpoint_blend_fraction"]
    )
    rows = []
    selection_summary = []
    for building in buildings:
        selected_motions, periods, reason, duplicate_omissions = (
            select_motions_for_t1(
                motions,
                t1_s=float(building["t1_s"]),
                midpoint_blend_fraction=blend_fraction,
                include_pwsa=include_pwsa,
            )
        )
        cms_motions = [
            motion
            for motion in motions
            if motion["source_set"] == "CMS_ZONE5"
            and float(motion["conditioning_period_s"]) in periods
        ]
        candidate_pair_count = len(cms_motions) + (
            len(pwsa_motions) if include_pwsa else 0
        )
        for motion in selected_motions:
            is_pwsa = str(motion["source_set"]).startswith("PWSA")
            motion_reason = (
                reason
                if not is_pwsa
                else (
                    "PWSA 2568 observed pair retained at as-recorded "
                    "scale factor 1.0 as event-specific sensitivity only"
                )
            )
            rows.append(
                {
                    "building_id": building["building_id"],
                    "pair_id": motion["pair_id"],
                    "t1_s": building["t1_s"],
                    "selected_cms_periods_json": periods,
                    "selection_reason": motion_reason,
                    "analysis_role": (
                        "event_specific_sensitivity"
                        if is_pwsa
                        else "fragility_primary"
                    ),
                    "scale_factor_policy": (
                        "as_recorded_sf1"
                        if is_pwsa
                        else "incremental_ida"
                    ),
                }
            )
        selection_summary.append(
            {
                "building_id": building["building_id"],
                "queue_rank": building["queue_rank"],
                "t1_s": building["t1_s"],
                "selected_cms_periods_s": periods,
                "cms_pair_count": len(cms_motions),
                "year_2568_pair_count": (
                    len(pwsa_motions) if include_pwsa else 0
                ),
                "total_pair_count": len(selected_motions),
                "candidate_pair_count_before_physical_deduplication": (
                    candidate_pair_count
                ),
                "physical_duplicate_omission_count": len(
                    duplicate_omissions
                ),
                "physical_duplicate_omissions_json": duplicate_omissions,
                "selection_reason": reason,
            }
        )
    new_pairs_by_building: dict[str, set[str]] = {
        str(building["building_id"]): set() for building in buildings
    }
    for row in rows:
        new_pairs_by_building[str(row["building_id"])].add(str(row["pair_id"]))
    current_building_ids = sorted(new_pairs_by_building)
    old_pairs_by_building: dict[str, set[str]] = {
        building_id: set() for building_id in current_building_ids
    }
    if current_building_ids:
        marks = ",".join("?" for _ in current_building_ids)
        with connect(config["database_path"]) as connection:
            for row in connection.execute(
                f"""
                SELECT building_id, pair_id
                FROM building_ground_motion_selection
                WHERE building_id IN ({marks})
                """,
                current_building_ids,
            ):
                old_pairs_by_building[str(row["building_id"])].add(
                    str(row["pair_id"])
                )
    changed_buildings = [
        building_id
        for building_id in current_building_ids
        if old_pairs_by_building[building_id]
        != new_pairs_by_building[building_id]
    ]
    with transaction(config["database_path"]) as connection:
        if changed_buildings:
            changed_marks = ",".join("?" for _ in changed_buildings)
            capacity_pairs_by_building: dict[str, set[str]] = {
                building_id: set() for building_id in changed_buildings
            }
            for capacity_row in connection.execute(
                f"""
                SELECT c.building_id, c.pair_id
                FROM ida_capacities c
                JOIN building_catalog b USING(building_id)
                WHERE b.selected=1 AND b.valid=1
                  AND c.building_id IN ({changed_marks})
                """,
                changed_buildings,
            ):
                capacity_pairs_by_building[
                    str(capacity_row["building_id"])
                ].add(str(capacity_row["pair_id"]))
            reconstruction_buildings = {
                building_id
                for building_id in changed_buildings
                if capacity_pairs_by_building[building_id]
                and not old_pairs_by_building[building_id]
                and capacity_pairs_by_building[building_id].issubset(
                    new_pairs_by_building[building_id]
                )
            }
            incompatible_capacity_buildings = [
                building_id
                for building_id in changed_buildings
                if capacity_pairs_by_building[building_id]
                and building_id not in reconstruction_buildings
            ]
            if incompatible_capacity_buildings:
                raise RuntimeError(
                    "Ground-motion selection cannot change after production "
                    "IDA capacities exist for the affected active buildings."
                )
            reset_buildings = [
                building_id
                for building_id in changed_buildings
                if building_id not in reconstruction_buildings
            ]
            if reset_buildings:
                reset_marks = ",".join("?" for _ in reset_buildings)
                connection.execute(
                    f"DELETE FROM fragility_targets "
                    f"WHERE building_id IN ({reset_marks})",
                    reset_buildings,
                )
                connection.execute(
                    f"DELETE FROM ida_capacities "
                    f"WHERE building_id IN ({reset_marks})",
                    reset_buildings,
                )
                connection.execute(
                    f"DELETE FROM ida_runs "
                    f"WHERE building_id IN ({reset_marks})",
                    reset_buildings,
                )
            # Any selection change invalidates ML provenance. Reconstructing a
            # missing selection table is allowed only when every preserved
            # capacity pair remains inside the newly derived selection.
            connection.execute("DELETE FROM ml_runs")
            connection.execute("DELETE FROM ml_split_manifest")
            connection.execute("DELETE FROM ml_split")
        if building_ids:
            marks = ",".join("?" for _ in building_ids)
            connection.execute(
                f"DELETE FROM building_ground_motion_selection "
                f"WHERE building_id IN ({marks})",
                building_ids,
            )
        else:
            connection.execute(
                "DELETE FROM building_ground_motion_selection"
            )
        upsert_many(
            connection,
            "building_ground_motion_selection",
            rows,
            ("building_id", "pair_id"),
        )
    output_path = (
        Path(config["output_dir"]) / "ground_motion_selection.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # This is the canonical selection table, so it must always represent the
    # complete active database state. A building-specific IDA or validation
    # call must not silently replace it with a one-building subset.
    with connect(config["database_path"]) as connection:
        current_selection_summary = [
            {
                "building_id": str(row["building_id"]),
                "queue_rank": int(row["queue_rank"]),
                "t1_s": float(row["t1_s"]),
                "selected_cms_periods_s": json.loads(
                    str(row["selected_cms_periods_json"])
                ),
                "cms_pair_count": int(row["cms_pair_count"]),
                "year_2568_pair_count": int(
                    row["year_2568_pair_count"]
                ),
                "total_pair_count": int(row["total_pair_count"]),
                "physical_unique_pair_count": int(
                    row["physical_unique_pair_count"]
                ),
                "fragility_primary_pair_count": int(
                    row["fragility_primary_pair_count"]
                ),
                "sensitivity_pair_count": int(
                    row["sensitivity_pair_count"]
                ),
                "selection_reasons": str(row["selection_reasons"]),
            }
            for row in connection.execute(
                """
                SELECT s.building_id, b.queue_rank, s.t1_s,
                       MIN(s.selected_cms_periods_json)
                           AS selected_cms_periods_json,
                       SUM(CASE WHEN g.source_set='CMS_ZONE5'
                                THEN 1 ELSE 0 END) AS cms_pair_count,
                       SUM(CASE WHEN g.source_set LIKE 'PWSA%'
                                THEN 1 ELSE 0 END)
                           AS year_2568_pair_count,
                       COUNT(*) AS total_pair_count,
                       COUNT(DISTINCT COALESCE(
                           g.physical_pair_hash, g.pair_id
                       )) AS physical_unique_pair_count,
                       SUM(CASE WHEN s.analysis_role='fragility_primary'
                                THEN 1 ELSE 0 END)
                           AS fragility_primary_pair_count,
                       SUM(CASE WHEN
                                     s.analysis_role=
                                         'event_specific_sensitivity'
                                THEN 1 ELSE 0 END)
                           AS sensitivity_pair_count,
                       GROUP_CONCAT(DISTINCT s.selection_reason)
                           AS selection_reasons
                FROM building_ground_motion_selection s
                JOIN building_catalog b USING(building_id)
                JOIN ground_motion_catalog g USING(pair_id)
                WHERE b.selected=1 AND b.valid=1 AND g.valid=1
                GROUP BY s.building_id, b.queue_rank, s.t1_s
                ORDER BY b.queue_rank
                """
            )
        ]
    if current_selection_summary:
        with output_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(current_selection_summary[0])
            )
            writer.writeheader()
            writer.writerows(current_selection_summary)
    pair_counts = [
        summary["total_pair_count"] for summary in selection_summary
    ]
    return {
        "building_count": len(buildings),
        "available_cms_periods_s": cms_periods,
        "midpoint_blend_fraction": blend_fraction,
        "selection_row_count": len(rows),
        "physical_duplicate_omission_count": sum(
            int(summary["physical_duplicate_omission_count"])
            for summary in selection_summary
        ),
        "pair_count_range": (
            [min(pair_counts), max(pair_counts)]
            if pair_counts
            else [0, 0]
        ),
        "selection_csv_building_count": len(current_selection_summary),
        "selection_csv": str(output_path.absolute()),
    }
