"""Fail-closed preflight for the portable Batch 004-008 package."""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import sys
import zlib
from pathlib import Path

from portable_common import BATCHES, building_rows, read_json, sha256, utc_now


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--build-check", action="store_true")
    args = parser.parse_args()
    root = args.root.absolute()
    database = root / "data" / "server_batches_004_008.sqlite"
    reserve_database = root / "data" / "spo_replacement_reserve.sqlite"
    config_path = root / "config" / "poc.json"
    blockers: list[str] = []
    warnings: list[str] = []

    if not database.is_file():
        blockers.append("portable SQLite database is missing")
    if not config_path.is_file():
        blockers.append("portable config is missing")
    if not reserve_database.is_file():
        blockers.append("SPO replacement reserve database is missing")
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
    replacement_policy = package.get("spo_replacement_policy", {})
    if (
        replacement_policy.get("schema")
        != "spo-full-catalog-nearest-replacement-v2"
    ):
        blockers.append("SPO full-catalog replacement policy is not locked")
    if (
        replacement_policy.get("invalid_spo_action")
        != "quarantine_and_refill_same_queue_slot"
    ):
        blockers.append("SPO invalid-result quarantine action is not locked")
    if replacement_policy.get(
        "downstream_response_used_for_selection"
    ) is not False:
        blockers.append(
            "SPO replacement selection must not use downstream response"
        )
    if (
        replacement_policy.get("reserve_scope")
        != "complete_five_storey_catalog"
    ):
        blockers.append("complete five-storey replacement catalog is not locked")
    streaming_policy = package.get("streaming_pipeline", {})
    if (
        streaming_policy.get("schema")
        != "streaming-spo-to-fragility-v2"
    ):
        blockers.append("streaming SPO-to-fragility policy is not locked")
    if streaming_policy.get("wait_for_all_spo_before_ida") is not False:
        blockers.append("Full IDA is still gated by batch-wide SPO completion")
    if (
        streaming_policy.get(
            "release_building_to_ida_after_own_spo_and_gm_selection"
        )
        is not True
    ):
        blockers.append("per-building SPO-to-IDA release is not enabled")
    if (
        streaming_policy.get("wait_for_all_ida_before_fragility")
        is not False
    ):
        blockers.append(
            "fragility is still gated by batch-wide IDA completion"
        )
    if (
        streaming_policy.get(
            "release_building_to_fragility_after_own_full_ida"
        )
        is not True
    ):
        blockers.append("per-building IDA-to-fragility release is not enabled")
    orchestrator_source = root / "server_orchestrator.py"
    orchestrator_text = (
        orchestrator_source.read_text(encoding="utf-8")
        if orchestrator_source.is_file()
        else ""
    )
    for required_symbol in (
        "def run_streaming_spo_ida(",
        "def streaming_worksets(",
        "def _run_stream_fragility(",
        "run_streaming_spo_ida(state, batch_id, workers)",
    ):
        if required_symbol not in orchestrator_text:
            blockers.append(
                f"streaming orchestrator implementation missing: "
                f"{required_symbol}"
            )
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
    if reserve_database.is_file():
        reserve_manifest = read_json(
            root / "data" / "spo_replacement_reserve_manifest.json", {}
        )
        reserve_connection = sqlite3.connect(
            reserve_database, timeout=60.0
        )
        try:
            reserve_integrity = reserve_connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
            if reserve_integrity != "ok":
                blockers.append(
                    "SPO replacement reserve integrity check failed: "
                    f"{reserve_integrity}"
                )
            catalogue_count = int(
                reserve_connection.execute(
                    """
                    SELECT COUNT(*) FROM building_catalog
                    WHERE stories=5
                    """
                ).fetchone()[0]
            )
            valid_count = int(
                reserve_connection.execute(
                    """
                    SELECT COUNT(*) FROM building_catalog
                    WHERE stories=5 AND valid=1
                    """
                ).fetchone()[0]
            )
            reserve_count = int(
                reserve_connection.execute(
                    """
                    SELECT COUNT(*) FROM building_catalog
                    WHERE stories=5 AND valid=1 AND selected=0
                    """
                ).fetchone()[0]
            )
            if reserve_count < 225:
                blockers.append(
                    "SPO replacement reserve is too small: "
                    f"{reserve_count} models"
                )
            if (
                reserve_manifest.get("schema")
                != "full-five-storey-catalog-v2"
            ):
                blockers.append("full five-storey catalog manifest is invalid")
            if (
                reserve_manifest.get("design_metadata_storage")
                != "zlib-compressed-utf8-lossless"
            ):
                blockers.append(
                    "full catalog lossless metadata storage is not locked"
                )
            if (
                reserve_manifest.get("target_database_sha256")
                != sha256(reserve_database)
            ):
                blockers.append("full catalog file checksum is invalid")
            corrupt_metadata: list[str] = []
            for row in reserve_connection.execute(
                """
                SELECT building_id,design_metadata_json
                FROM building_catalog
                """
            ):
                try:
                    compressed = row[1]
                    if not isinstance(compressed, bytes):
                        raise TypeError("metadata is not a compressed blob")
                    decoded = json.loads(
                        zlib.decompress(compressed).decode("utf-8")
                    )
                    if not isinstance(decoded, dict):
                        raise TypeError("metadata JSON is not an object")
                except (
                    TypeError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    zlib.error,
                ):
                    corrupt_metadata.append(str(row[0]))
                    if len(corrupt_metadata) >= 5:
                        break
            if corrupt_metadata:
                blockers.append(
                    "full catalog contains corrupt compressed metadata: "
                    + ",".join(corrupt_metadata)
                )
            expected_counts = {
                "catalogue_model_count": catalogue_count,
                "valid_model_count": valid_count,
                "eligible_replacement_model_count": reserve_count,
            }
            for key, actual in expected_counts.items():
                if int(reserve_manifest.get(key, -1)) != actual:
                    blockers.append(
                        f"full catalog manifest count mismatch: {key}"
                    )
            db_summary["five_storey_catalog_count"] = catalogue_count
            db_summary["five_storey_valid_model_count"] = valid_count
            db_summary["spo_replacement_reserve_count"] = reserve_count
        finally:
            reserve_connection.close()

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
