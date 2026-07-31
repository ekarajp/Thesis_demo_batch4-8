"""Fail-closed preflight for the portable Batch 004-008 package."""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import sys
from pathlib import Path

from portable_common import BATCHES, building_rows, read_json, sha256, utc_now


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--build-check", action="store_true")
    args = parser.parse_args()
    root = args.root.absolute()
    database = root / "data" / "server_batches_004_008.sqlite"
    config_path = root / "config" / "poc.json"
    blockers: list[str] = []
    warnings: list[str] = []

    if not database.is_file():
        blockers.append("portable SQLite database is missing")
    if not config_path.is_file():
        blockers.append("portable config is missing")
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    if int(config.get("building", {}).get("stories", -1)) != 5:
        blockers.append("Demo archetype is not locked to five storeys")
    package = config.get("portable_server_package", {})
    if package.get("queue_rank_start") != 151:
        blockers.append("portable queue does not start at rank 151")
    if package.get("queue_rank_end") != 375:
        blockers.append("portable queue does not end at rank 375")
    if package.get("phase") != "data_generation_only_no_ml":
        blockers.append("package phase is not data-generation-only")
    checkpoint_policy = package.get("checkpoint_policy", {})
    if checkpoint_policy.get("sqlite_synchronous") != "FULL":
        blockers.append("SQLite FULL-sync checkpoint policy is not locked")
    db_source = root / "src" / "fragility_poc" / "db.py"
    if not db_source.is_file() or (
        'connection.execute("PRAGMA synchronous = FULL")'
        not in db_source.read_text(encoding="utf-8")
    ):
        blockers.append("Portable database connections do not enforce FULL sync")

    db_summary = {}
    if database.is_file():
        connection = sqlite3.connect(database, timeout=60.0)
        connection.row_factory = sqlite3.Row
        try:
            integrity = connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
            if integrity != "ok":
                blockers.append(f"SQLite integrity check failed: {integrity}")
            buildings = building_rows(connection)
            ranks = [int(row["queue_rank"]) for row in buildings]
            if len(buildings) != 225 or ranks != list(range(151, 376)):
                blockers.append(
                    "database does not contain exactly frozen ranks 151-375"
                )
            if len({row["model_hash"] for row in buildings}) != len(buildings):
                blockers.append("target building model_hash values are duplicated")
            gm_rows = list(
                connection.execute(
                    """
                    SELECT pair_id, component_x_path, component_y_path,
                           sha256_x, sha256_y
                    FROM ground_motion_catalog WHERE valid=1
                    """
                )
            )
            if len(gm_rows) != 24:
                blockers.append(
                    f"expected 24 valid GM pairs, found {len(gm_rows)}"
                )
            for row in gm_rows:
                for key, checksum in (
                    ("component_x_path", "sha256_x"),
                    ("component_y_path", "sha256_y"),
                ):
                    path = Path(str(row[key]))
                    if not path.is_file():
                        blockers.append(
                            f"{row['pair_id']} missing {key}: {path}"
                        )
                    elif sha256(path) != str(row[checksum]):
                        blockers.append(
                            f"{row['pair_id']} checksum mismatch for {key}"
                        )
            foreign = list(connection.execute("PRAGMA foreign_key_check"))
            if foreign:
                blockers.append(
                    f"database contains {len(foreign)} foreign-key violations"
                )
            outside = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                      SELECT building_id FROM spo_features
                      UNION ALL SELECT building_id FROM ida_runs
                      UNION ALL SELECT building_id FROM fragility_targets
                    ) x
                    WHERE building_id NOT IN (
                      SELECT building_id FROM building_catalog
                    )
                    """
                ).fetchone()[0]
            )
            if outside:
                blockers.append("result tables contain non-target buildings")
            db_summary = {
                "building_count": len(buildings),
                "valid_ground_motion_pair_count": len(gm_rows),
                "existing_spo_count": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM spo_features WHERE valid=1"
                    ).fetchone()[0]
                ),
                "existing_ida_run_count": int(
                    connection.execute("SELECT COUNT(*) FROM ida_runs").fetchone()[0]
                ),
                "existing_fragility_count": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM fragility_targets WHERE valid=1"
                    ).fetchone()[0]
                ),
            }
        finally:
            connection.close()

    if not args.build_check:
        if sys.version_info[:2] != (3, 10) or platform.architecture()[0] != "64bit":
            blockers.append(
                "runtime must use 64-bit CPython 3.10 (not another version)"
            )
        try:
            import numpy  # noqa: F401
            import openseespy.opensees  # noqa: F401
            import scipy  # noqa: F401
            import sklearn  # noqa: F401
        except Exception as exc:
            blockers.append(f"locked Python environment import failed: {exc}")
        prepared = read_json(root / "runtime" / "prepared.json")
        if not prepared:
            blockers.append("portable runtime has not been prepared")
        elif prepared.get("package_root") != str(root):
            blockers.append("runtime relocation marker points to another root")
    else:
        warnings.append(
            "Build check does not import the Python 3.10/OpenSees environment."
        )

    report = {
        "schema": "portable-preflight-v1",
        "timestamp_utc": utc_now(),
        "status": "PASS" if not blockers else "BLOCKED",
        "ready": not blockers,
        "build_check": args.build_check,
        "package_root": str(root),
        "batches": BATCHES,
        "database": db_summary,
        "blockers": blockers,
        "warnings": warnings,
    }
    if not args.build_check:
        from portable_common import atomic_json

        atomic_json(root / "runtime" / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
