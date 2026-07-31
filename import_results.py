"""Validate and merge server result ZIPs into the main Demo Program.

This script is deliberately fail-closed.  It verifies scientific config,
source-code, model, ground-motion and Building-ID identities; creates an
online SQLite backup; refuses conflicting existing rows; copies raw files
with checksums; and commits all database rows in one transaction.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from portable_common import (
    BATCHES,
    RESULT_SCHEMA,
    atomic_json,
    ground_motion_signature,
    online_backup,
    scientific_config_signature,
    sha256,
    source_code_signature,
    table_columns,
    table_primary_key,
    utc_now,
)


RESULT_TABLES = (
    "building_ground_motion_selection",
    "spo_features",
    "ida_runs",
    "ida_capacities",
    "ida_curve_diagnostics",
    "ida_manual_point_requests",
    "fragility_targets",
)
PATH_COLUMNS = {
    ("spo_features", "curve_path"),
    ("spo_features", "mechanism_history_path"),
    ("ida_runs", "result_path"),
    ("ida_manual_point_requests", "result_path"),
}


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.absolute()
    for member in archive.infolist():
        target = (root / member.filename).absolute()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"ZIP path traversal refused: {member.filename}"
            ) from exc
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)


def locate_manifest(extracted: Path) -> tuple[Path, dict[str, Any]]:
    candidates = list(extracted.rglob("result_manifest.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one result_manifest.json; found {len(candidates)}"
        )
    path = candidates[0]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != RESULT_SCHEMA:
        raise RuntimeError("Result manifest schema is incompatible")
    return path, manifest


def verify_result_files(base: Path, manifest: dict[str, Any]) -> None:
    problems = []
    seen = set()
    for item in manifest.get("files", []):
        relative = str(item["archive_path"])
        if relative in seen:
            problems.append(f"duplicate archive path: {relative}")
            continue
        seen.add(relative)
        path = base / relative
        if not path.is_file():
            problems.append(f"missing: {relative}")
        elif path.stat().st_size != int(item["size_bytes"]):
            problems.append(f"size mismatch: {relative}")
        elif sha256(path) != str(item["sha256"]):
            problems.append(f"checksum mismatch: {relative}")
    if problems:
        raise RuntimeError(
            "Result ZIP file verification failed: " + "; ".join(problems)
        )


def value_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) <= 1.0e-12 * max(
                1.0, abs(float(left)), abs(float(right))
            )
        except (TypeError, ValueError):
            pass
    return left == right


def row_conflicts(
    existing: sqlite3.Row, incoming: dict[str, Any], columns: list[str]
) -> list[str]:
    return [
        column
        for column in columns
        if not value_equal(existing[column], incoming[column])
    ]


def replacement_paths(
    main_root: Path,
    base: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, str], list[tuple[Path, Path, str]]]:
    mapping: dict[str, str] = {}
    copies: list[tuple[Path, Path, str]] = []
    batch_id = str(manifest["batch_id"])
    for item in manifest.get("raw_file_relocation", []):
        source = base / str(item["archive_path"])
        archive_path = Path(str(item["archive_path"]))
        parts = archive_path.parts
        if len(parts) >= 2 and parts[0] == "raw" and parts[1] != "external":
            relative = Path(*parts[1:])
        else:
            relative = (
                Path("runs")
                / "5storey"
                / "server_imports"
                / batch_id
                / "external"
                / source.name
            )
        target = (main_root / relative).absolute()
        mapping[str(item["original_path"])] = str(target)
        copies.append((source, target, str(item["sha256"])))
    return mapping, copies


def verify_scientific_identity(
    main_root: Path,
    main_connection: sqlite3.Connection,
    result_connection: sqlite3.Connection,
    manifest: dict[str, Any],
) -> None:
    config = main_root / "config" / "poc.json"
    if scientific_config_signature(config) != str(
        manifest["scientific_config_signature"]
    ):
        raise RuntimeError(
            "Scientific configuration differs from the server package; "
            "automatic merge is refused."
        )
    if source_code_signature(main_root / "src" / "fragility_poc") != str(
        manifest["source_code_signature"]
    ):
        raise RuntimeError(
            "Structural-analysis source code differs from the server "
            "package; automatic merge is refused."
        )
    controller = main_root / "models" / "active_ida_controller_v1.joblib"
    if sha256(controller) != str(manifest["controller_model_sha256"]):
        raise RuntimeError("Active-IDA controller model differs")
    if ground_motion_signature(main_connection) != str(
        manifest["ground_motion_signature"]
    ):
        raise RuntimeError("Ground-motion catalogue identity differs")

    expected = {
        str(row["building_id"]): (
            int(row["queue_rank"]),
            str(row["model_hash"]),
        )
        for row in manifest["buildings"]
    }
    placeholders = ",".join("?" for _ in expected)
    current = {
        str(row["building_id"]): (
            int(row["queue_rank"]),
            str(row["model_hash"]),
        )
        for row in main_connection.execute(
            f"""
            SELECT building_id,queue_rank,model_hash FROM building_catalog
            WHERE building_id IN ({placeholders})
            """,
            list(expected),
        )
    }
    result = {
        str(row["building_id"]): (
            int(row["queue_rank"]),
            str(row["model_hash"]),
        )
        for row in result_connection.execute(
            "SELECT building_id,queue_rank,model_hash FROM building_catalog"
        )
    }
    if current != expected or result != expected:
        raise RuntimeError(
            "Building-ID/rank/model-hash identity does not match exactly"
        )


def prepare_incoming_rows(
    main_connection: sqlite3.Connection,
    result_connection: sqlite3.Connection,
    path_mapping: dict[str, str],
    main_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    incoming: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, dict[str, int]] = {}
    for table in RESULT_TABLES:
        main_columns = table_columns(main_connection, table)
        result_columns = table_columns(result_connection, table)
        if main_columns != result_columns:
            raise RuntimeError(f"Database schema differs for table {table}")
        primary_key = table_primary_key(main_connection, table)
        if not primary_key:
            raise RuntimeError(f"Table {table} has no primary key")
        rows = [dict(row) for row in result_connection.execute(f"SELECT * FROM {table}")]
        for row in rows:
            for column in main_columns:
                if (table, column) in PATH_COLUMNS and row.get(column):
                    original = str(row[column])
                    if original not in path_mapping:
                        raise RuntimeError(
                            f"No raw-file relocation for {table}.{column}: "
                            f"{original}"
                        )
                    row[column] = path_mapping[original]
            if table == "ida_curve_diagnostics" and row.get("model_path"):
                row["model_path"] = str(
                    main_root
                    / "models"
                    / Path(str(row["model_path"])).name
                )
        insert_count = 0
        identical_count = 0
        for row in rows:
            where = " AND ".join(f"{key}=?" for key in primary_key)
            values = [row[key] for key in primary_key]
            existing = main_connection.execute(
                f"SELECT * FROM {table} WHERE {where}", values
            ).fetchone()
            if existing is None:
                insert_count += 1
                continue
            differences = row_conflicts(existing, row, main_columns)
            if differences:
                identity = ", ".join(
                    f"{key}={row[key]!r}" for key in primary_key
                )
                raise RuntimeError(
                    f"Conflicting existing row in {table} ({identity}); "
                    f"columns={differences}"
                )
            identical_count += 1
        incoming[table] = rows
        summary[table] = {
            "incoming": len(rows),
            "to_insert": insert_count,
            "already_identical": identical_count,
        }
    return incoming, summary


def prepare_failures(
    main_connection: sqlite3.Connection,
    result_connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in result_connection.execute(
            "SELECT * FROM pipeline_failures ORDER BY failure_id"
        )
    ]
    output = []
    for row in rows:
        duplicate = main_connection.execute(
            """
            SELECT 1 FROM pipeline_failures
            WHERE stage=? AND building_id IS ? AND pair_id IS ?
              AND created_utc=? AND error_type=? AND message=?
            """,
            (
                row["stage"],
                row["building_id"],
                row["pair_id"],
                row["created_utc"],
                row["error_type"],
                row["message"],
            ),
        ).fetchone()
        if duplicate is None:
            row.pop("failure_id", None)
            output.append(row)
    return output


def copy_raw_files(copies: list[tuple[Path, Path, str]]) -> dict[str, int]:
    copied = 0
    identical = 0
    for source, target, expected_hash in copies:
        if sha256(source) != expected_hash:
            raise RuntimeError(f"Extracted raw-file checksum failed: {source}")
        if target.exists():
            if sha256(target) != expected_hash:
                raise RuntimeError(
                    f"Main project has conflicting raw file: {target}"
                )
            identical += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".importing")
        shutil.copy2(source, temporary)
        if sha256(temporary) != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Copied raw-file checksum failed: {target}")
        os.replace(temporary, target)
        copied += 1
    return {"copied": copied, "already_identical": identical}


def insert_rows(
    connection: sqlite3.Connection,
    incoming: dict[str, list[dict[str, Any]]],
    failures: list[dict[str, Any]],
) -> dict[str, int]:
    inserted: dict[str, int] = {}
    for table in RESULT_TABLES:
        columns = table_columns(connection, table)
        primary_key = table_primary_key(connection, table)
        count = 0
        for row in incoming[table]:
            where = " AND ".join(f"{key}=?" for key in primary_key)
            if connection.execute(
                f"SELECT 1 FROM {table} WHERE {where}",
                [row[key] for key in primary_key],
            ).fetchone():
                continue
            placeholders = ",".join("?" for _ in columns)
            connection.execute(
                f"INSERT INTO {table} ({','.join(columns)}) "
                f"VALUES ({placeholders})",
                [row[column] for column in columns],
            )
            count += 1
        inserted[table] = count
    failure_columns = [
        column
        for column in table_columns(connection, "pipeline_failures")
        if column != "failure_id"
    ]
    for row in failures:
        connection.execute(
            f"INSERT INTO pipeline_failures ({','.join(failure_columns)}) "
            f"VALUES ({','.join('?' for _ in failure_columns)})",
            [row[column] for column in failure_columns],
        )
    inserted["pipeline_failures"] = len(failures)
    return inserted


def import_one(
    main_root: Path, zip_path: Path, work_root: Path
) -> dict[str, Any]:
    extract_root = work_root / (zip_path.stem + "_extracted")
    extract_root.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(zip_path) as archive:
        safe_extract(archive, extract_root)
    manifest_path, manifest = locate_manifest(extract_root)
    base = manifest_path.parent
    verify_result_files(base, manifest)
    batch_id = str(manifest["batch_id"])
    if batch_id not in BATCHES:
        raise RuntimeError(f"Unexpected batch ID: {batch_id}")
    if list(manifest["queue_rank_range"]) != list(BATCHES[batch_id]):
        raise RuntimeError("Batch rank range is inconsistent")
    result_database = base / "result_database.sqlite"
    main_database = main_root / "data" / "5storey" / "poc.sqlite"
    if not main_database.is_file():
        raise FileNotFoundError(f"Main database not found: {main_database}")

    main_connection = sqlite3.connect(main_database, timeout=60.0)
    main_connection.row_factory = sqlite3.Row
    main_connection.execute("PRAGMA foreign_keys=ON")
    main_connection.execute("PRAGMA busy_timeout=60000")
    result_connection = sqlite3.connect(result_database)
    result_connection.row_factory = sqlite3.Row
    try:
        verify_scientific_identity(
            main_root, main_connection, result_connection, manifest
        )
        path_mapping, copies = replacement_paths(main_root, base, manifest)
        incoming, row_summary = prepare_incoming_rows(
            main_connection,
            result_connection,
            path_mapping,
            main_root,
        )
        failures = prepare_failures(main_connection, result_connection)

        backup_dir = main_root / "backups" / "server_result_imports"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = utc_now().replace(":", "").replace("+", "_")
        backup = backup_dir / f"pre_import_{batch_id}_{stamp}.sqlite"
        online_backup(main_database, backup)
        raw_summary = copy_raw_files(copies)
        try:
            main_connection.execute("BEGIN IMMEDIATE")
            inserted = insert_rows(main_connection, incoming, failures)
            foreign = list(main_connection.execute("PRAGMA foreign_key_check"))
            if foreign:
                raise RuntimeError(
                    f"Import creates {len(foreign)} foreign-key violations"
                )
            main_connection.commit()
        except Exception:
            main_connection.rollback()
            raise

        report = {
            "schema": "server-result-import-report-v1",
            "timestamp_utc": utc_now(),
            "status": "IMPORTED",
            "batch_id": batch_id,
            "queue_rank_range": manifest["queue_rank_range"],
            "result_zip": str(zip_path),
            "result_zip_sha256": sha256(zip_path),
            "database_backup": str(backup),
            "database_backup_sha256": sha256(backup),
            "row_precheck": row_summary,
            "inserted_rows": inserted,
            "raw_files": raw_summary,
            "scientific_identity_verified": True,
        }
        report_dir = (
            main_root / "outputs" / "5storey" / "server_result_imports"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(report_dir / f"{batch_id}_import_report.json", report)
        return report
    finally:
        result_connection.close()
        main_connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-project", type=Path, required=True)
    parser.add_argument("--result-zip", type=Path, required=True)
    args = parser.parse_args()
    main_root = args.main_project.absolute()
    result_zip = args.result_zip.absolute()
    if not result_zip.is_file():
        raise FileNotFoundError(result_zip)
    work_parent = main_root / "runtime_storage" / "server_result_imports"
    work_parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(
        tempfile.mkdtemp(prefix="import_", dir=work_parent)
    ).absolute()
    reports = []
    try:
        with zipfile.ZipFile(result_zip) as archive:
            names = archive.namelist()
            combined = "COMBINED_MANIFEST.json" in names
            if combined:
                safe_extract(archive, work_root / "combined")
                combined_root = work_root / "combined"
                manifest = json.loads(
                    (combined_root / "COMBINED_MANIFEST.json").read_text(
                        encoding="utf-8"
                    )
                )
                for item in manifest["files"]:
                    nested = combined_root / str(item["name"])
                    if sha256(nested) != str(item["sha256"]):
                        raise RuntimeError(
                            f"Nested batch ZIP checksum failed: {nested.name}"
                        )
                    reports.append(import_one(main_root, nested, work_root))
            else:
                reports.append(import_one(main_root, result_zip, work_root))
        print(
            json.dumps(
                {
                    "status": "ok",
                    "imported_batch_count": len(reports),
                    "reports": reports,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        # The target is a freshly created child of the explicit project-local
        # import workspace.  Verify that relationship before recursive cleanup.
        try:
            work_root.relative_to(work_parent.absolute())
        except ValueError:
            raise RuntimeError(
                f"Refusing to remove unexpected temp path: {work_root}"
            )
        shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
