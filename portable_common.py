"""Shared utilities for the portable Batch 004-008 server package."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PACKAGE_SCHEMA = "rc-fragility-portable-batches-004-008-v1"
RESULT_SCHEMA = "rc-fragility-server-result-v1"
BATCHES = {
    "batch_004": (151, 200),
    "batch_005": (201, 250),
    "batch_006": (251, 300),
    "batch_007": (301, 350),
    "batch_008": (351, 375),
}
PRIMARY_ROLE = "fragility_primary"
PWSA_ROLE = "event_specific_sensitivity"


def root_from(script: str | Path) -> Path:
    return Path(script).absolute().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def append_jsonl(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, default=str) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.is_file():
        return default
    return json.loads(target.read_text(encoding="utf-8"))


def connect(path: str | Path, *, readonly: bool = False) -> sqlite3.Connection:
    db_path = Path(path).absolute()
    if readonly:
        connection = sqlite3.connect(
            f"{db_path.as_uri()}?mode=ro", uri=True, timeout=60.0
        )
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path, timeout=60.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=60000")
    return connection


def online_backup(source: str | Path, target: str | Path) -> None:
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    source_connection = sqlite3.connect(Path(source), timeout=60.0)
    target_connection = sqlite3.connect(temporary)
    try:
        source_connection.backup(target_connection)
        integrity = target_connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite backup integrity failed: {integrity}")
    finally:
        target_connection.close()
        source_connection.close()
    os.replace(temporary, destination)


def building_rows(
    connection: sqlite3.Connection,
    rank_start: int = 151,
    rank_end: int = 375,
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in connection.execute(
            """
            SELECT building_id, queue_rank, model_hash, number_of_bays,
                   bay_width_m, fc_ksc, sdl_kg_m2, ll_kg_m2,
                   beam_tier, column_tier, scwb_strength_ratio, scwb_class
            FROM building_catalog
            WHERE selected=1 AND valid=1
              AND queue_rank BETWEEN ? AND ?
            ORDER BY queue_rank
            """,
            (rank_start, rank_end),
        )
    ]


def ids_for_batch(
    connection: sqlite3.Connection, batch_id: str
) -> list[str]:
    slot_table = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type='table' AND name='batch_building_slots'
        """
    ).fetchone()
    if slot_table is not None:
        assigned = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT current_building_id
                FROM batch_building_slots
                WHERE batch_id=?
                ORDER BY slot_rank
                """,
                (batch_id,),
            )
        ]
        if assigned:
            return assigned
    rank_start, rank_end = BATCHES[batch_id]
    return [
        str(row[0])
        for row in connection.execute(
            """
            SELECT building_id FROM building_catalog
            WHERE selected=1 AND valid=1
              AND queue_rank BETWEEN ? AND ?
            ORDER BY queue_rank
            """,
            (rank_start, rank_end),
        )
    ]


def expected_batch_count(batch_id: str) -> int:
    start, end = BATCHES[batch_id]
    return end - start + 1


def table_columns(
    connection: sqlite3.Connection, table: str
) -> list[str]:
    return [
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table})")
    ]


def table_primary_key(
    connection: sqlite3.Connection, table: str
) -> list[str]:
    rows = list(connection.execute(f"PRAGMA table_info({table})"))
    return [
        str(row["name"])
        for row in sorted(rows, key=lambda item: int(item["pk"]))
        if int(row["pk"]) > 0
    ]


def csv_write(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    columns: list[str] | None = None,
) -> int:
    materialized = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(materialized[0]) if materialized else []
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if columns:
            writer.writeheader()
            writer.writerows(materialized)
    return len(materialized)


def manifest_hash_payload(
    root: Path, relative_paths: Iterable[str]
) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(set(relative_paths)):
        path = root / relative
        rows.append(
            {
                "path": relative.replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def scientific_config_signature(path: str | Path) -> str:
    """Hash scientific choices while ignoring machine-specific paths."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    normalized = copy.deepcopy(payload)
    normalized.pop("portable_server_package", None)
    for key in (
        "database_path",
        "output_dir",
        "run_dir",
        "runtime_storage_root",
    ):
        if key in normalized:
            normalized[key] = f"<PORTABLE:{key}>"
    gm = normalized.get("ground_motion", {})
    for key in ("cms_root", "year_2568_workbook", "pwsa_x", "pwsa_y"):
        gm[key] = f"<PORTABLE:ground_motion.{key}>"
    gm.pop("portable_catalogue_mode", None)
    controller = normalized.get("ida", {}).get("controller", {})
    if "model_path" in controller:
        controller["model_path"] = Path(
            str(controller["model_path"])
        ).name
    serialized = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def source_code_signature(source_root: str | Path) -> str:
    source = Path(source_root)
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        relative = path.relative_to(source).as_posix()
        content = path.read_text(encoding="utf-8")
        # The portable server copy deliberately uses SQLite synchronous=FULL
        # for stronger power-loss durability.  This is an operational storage
        # hardening only; normalize that exact line so the scientific source
        # identity remains comparable with the main workstation copy.
        content = content.replace(
            '    connection.execute("PRAGMA synchronous = FULL")\n', ""
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def ground_motion_signature(connection: sqlite3.Connection) -> str:
    rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT pair_id, source_set, conditioning_period_s,
                   units, dt_s, npts, duration_s, sha256_x, sha256_y,
                   physical_pair_hash, valid
            FROM ground_motion_catalog
            ORDER BY pair_id
            """
        )
    ]
    serialized = json.dumps(
        rows, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def process_is_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def progress_snapshot(database: str | Path) -> dict[str, Any]:
    """Return accurate persisted progress; no time-based pseudo-progress."""
    with connect(database, readonly=True) as connection:
        buildings = [
            dict(row)
            for row in connection.execute(
                """
                SELECT building_id, queue_rank, model_hash, number_of_bays,
                       bay_width_m, fc_ksc, sdl_kg_m2, ll_kg_m2,
                       scwb_strength_ratio, scwb_class
                FROM building_catalog
                WHERE selected=1 AND valid=1
                  AND queue_rank BETWEEN 151 AND 375
                ORDER BY queue_rank
                """
            )
        ]
        spo = {
            str(row["building_id"]): dict(row)
            for row in connection.execute(
                """
                SELECT building_id, valid, t1_s, runtime_s,
                       validation_message
                FROM spo_features
                """
            )
        }
        selection: dict[str, dict[str, int]] = {}
        for row in connection.execute(
            """
            SELECT building_id,
                   SUM(CASE WHEN analysis_role=? THEN 1 ELSE 0 END)
                     AS primary_count,
                   SUM(CASE WHEN analysis_role=? THEN 1 ELSE 0 END)
                     AS pwsa_count
            FROM building_ground_motion_selection
            GROUP BY building_id
            """,
            (PRIMARY_ROLE, PWSA_ROLE),
        ):
            selection[str(row["building_id"])] = {
                "primary": int(row["primary_count"] or 0),
                "pwsa": int(row["pwsa_count"] or 0),
            }
        completed_primary: dict[str, int] = {}
        for row in connection.execute(
            """
            SELECT c.building_id, COUNT(*) AS curve_count
            FROM (
              SELECT building_id, pair_id
              FROM ida_capacities
              WHERE censored=0 AND censoring='none'
              GROUP BY building_id, pair_id
              HAVING COUNT(DISTINCT limit_state)=3
            ) c
            JOIN building_ground_motion_selection s
              ON s.building_id=c.building_id AND s.pair_id=c.pair_id
            WHERE s.analysis_role=?
            GROUP BY c.building_id
            """,
            (PRIMARY_ROLE,),
        ):
            completed_primary[str(row["building_id"])] = int(
                row["curve_count"]
            )
        pwsa_complete = {
            str(row["building_id"])
            for row in connection.execute(
                """
                SELECT DISTINCT s.building_id
                FROM building_ground_motion_selection s
                JOIN ida_runs r
                  ON r.building_id=s.building_id AND r.pair_id=s.pair_id
                WHERE s.analysis_role=?
                  AND s.scale_factor_policy='as_recorded_sf1'
                  AND ABS(r.scale_factor-1.0)<=1e-10
                  AND r.status IN ('success','dynamic_instability')
                """,
                (PWSA_ROLE,),
            )
        }
        ida_run_counts = {
            str(row["building_id"]): {
                "count": int(row["run_count"]),
                "runtime_s": float(row["runtime_s"] or 0.0),
            }
            for row in connection.execute(
                """
                SELECT building_id, COUNT(*) AS run_count,
                       SUM(runtime_s) AS runtime_s
                FROM ida_runs GROUP BY building_id
                """
            )
        }
        fragility = {
            str(row["building_id"]): int(row["valid"])
            for row in connection.execute(
                "SELECT building_id, valid FROM fragility_targets"
            )
        }
        unresolved = {
            str(row["building_id"]): int(row["failure_count"])
            for row in connection.execute(
                """
                SELECT building_id, COUNT(*) AS failure_count
                FROM pipeline_failures
                WHERE resolved=0 AND building_id IS NOT NULL
                GROUP BY building_id
                """
            )
        }
        slot_assignments: dict[str, dict[str, Any]] = {}
        quarantine_by_batch: dict[str, int] = {}
        replacement_table = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='batch_building_slots'
            """
        ).fetchone()
        if replacement_table is not None:
            slot_assignments = {
                str(row["current_building_id"]): dict(row)
                for row in connection.execute(
                    """
                    SELECT batch_id,slot_rank,original_building_id,
                           current_building_id,replacement_generation
                    FROM batch_building_slots
                    """
                )
            }
            quarantine_by_batch = {
                str(row["batch_id"]): int(row["quarantine_count"])
                for row in connection.execute(
                    """
                    SELECT batch_id,COUNT(*) AS quarantine_count
                    FROM spo_quarantine GROUP BY batch_id
                    """
                )
            }

    package_root = Path(database).absolute().parent.parent
    ida_checkpoint_root = package_root / "runs" / "ida"
    output_buildings: list[dict[str, Any]] = []
    for building in buildings:
        building_id = str(building["building_id"])
        rank = int(building["queue_rank"])
        batch_id = next(
            key
            for key, (start, end) in BATCHES.items()
            if start <= rank <= end
        )
        spo_valid = bool(
            building_id in spo and int(spo[building_id]["valid"]) == 1
        )
        selected = selection.get(building_id, {"primary": 0, "pwsa": 0})
        selection_complete = (
            selected["primary"] > 0 and selected["pwsa"] == 1
        )
        primary_total = selected["primary"]
        primary_complete = completed_primary.get(building_id, 0)
        primary_fraction = (
            min(primary_complete / primary_total, 1.0)
            if primary_total
            else 0.0
        )
        pwsa_done = building_id in pwsa_complete
        fragility_valid = fragility.get(building_id, 0) == 1
        percent = (
            (10.0 if spo_valid else 0.0)
            + (5.0 if selection_complete else 0.0)
            + 70.0 * primary_fraction
            + (5.0 if pwsa_done else 0.0)
            + (10.0 if fragility_valid else 0.0)
        )
        run_data = ida_run_counts.get(
            building_id, {"count": 0, "runtime_s": 0.0}
        )
        raw_checkpoint_count = 0
        checkpoint_building_dir = ida_checkpoint_root / building_id
        if checkpoint_building_dir.is_dir():
            raw_checkpoint_count = sum(
                1
                for pair_dir in checkpoint_building_dir.iterdir()
                if pair_dir.is_dir()
                for path in pair_dir.glob("im_*.json")
                if path.is_file()
            )
        output_buildings.append(
            {
                **building,
                "batch_id": batch_id,
                "percent": round(min(percent, 100.0), 2),
                "spo": "complete" if spo_valid else "pending",
                "t1_s": (
                    float(spo[building_id]["t1_s"])
                    if spo_valid
                    else None
                ),
                "selection": (
                    "complete" if selection_complete else "pending"
                ),
                "primary_curves_complete": primary_complete,
                "primary_curves_total": primary_total,
                "pwsa": "complete" if pwsa_done else "pending",
                "fragility": (
                    "complete" if fragility_valid else "pending"
                ),
                "ida_run_count": int(run_data["count"]),
                "saved_nltha_checkpoint_count": max(
                    raw_checkpoint_count, int(run_data["count"])
                ),
                "ida_runtime_s": round(float(run_data["runtime_s"]), 3),
                "unresolved_failures": unresolved.get(building_id, 0),
                "original_building_id": (
                    str(slot_assignments[building_id][
                        "original_building_id"
                    ])
                    if building_id in slot_assignments
                    else building_id
                ),
                "replacement_generation": int(
                    slot_assignments.get(
                        building_id, {"replacement_generation": 0}
                    )["replacement_generation"]
                ),
            }
        )

    batch_rows = []
    for batch_id, (rank_start, rank_end) in BATCHES.items():
        items = [
            row
            for row in output_buildings
            if rank_start <= int(row["queue_rank"]) <= rank_end
        ]
        batch_rows.append(
            {
                "batch_id": batch_id,
                "rank_start": rank_start,
                "rank_end": rank_end,
                "building_count": len(items),
                "completed_buildings": sum(
                    1 for row in items if row["percent"] >= 100.0
                ),
                "average_percent": round(
                    sum(float(row["percent"]) for row in items)
                    / max(len(items), 1),
                    2,
                ),
                "spo_complete": sum(
                    1 for row in items if row["spo"] == "complete"
                ),
                "fragility_complete": sum(
                    1
                    for row in items
                    if row["fragility"] == "complete"
                ),
                "unresolved_failures": sum(
                    int(row["unresolved_failures"]) for row in items
                ),
                "quarantined_spo_models": quarantine_by_batch.get(
                    batch_id, 0
                ),
            }
        )
    return {
        "timestamp_utc": utc_now(),
        "total_buildings": len(output_buildings),
        "completed_buildings": sum(
            1 for row in output_buildings if row["percent"] >= 100.0
        ),
        "average_percent": round(
            sum(float(row["percent"]) for row in output_buildings)
            / max(len(output_buildings), 1),
            2,
        ),
        "batches": batch_rows,
        "buildings": output_buildings,
        "quarantined_spo_models": sum(quarantine_by_batch.values()),
    }
