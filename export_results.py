"""Create readable, self-verifying result ZIPs for safe main-PC import."""

from __future__ import annotations

import argparse
import html
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
    connect,
    csv_write,
    ground_motion_signature,
    ids_for_batch,
    online_backup,
    progress_snapshot,
    scientific_config_signature,
    sha256,
    source_code_signature,
    utc_now,
)


def query_dicts(
    connection: sqlite3.Connection, sql: str, values: list[Any]
) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(sql, values)]


def marks(ids: list[str]) -> str:
    return ",".join("?" for _ in ids)


def make_result_database(
    source: Path, target: Path, ids: list[str]
) -> None:
    online_backup(source, target)
    placeholders = marks(ids)
    connection = sqlite3.connect(target, timeout=60.0)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        for table in (
            "ida_manual_point_requests",
            "ida_curve_diagnostics",
            "ida_capacities",
            "ida_runs",
            "fragility_targets",
            "building_ground_motion_selection",
            "spo_features",
            "ml_split",
        ):
            connection.execute(
                f"DELETE FROM {table} WHERE building_id NOT IN ({placeholders})",
                ids,
            )
        connection.execute(
            f"""
            DELETE FROM pipeline_failures
            WHERE building_id IS NULL
               OR building_id NOT IN ({placeholders})
            """,
            ids,
        )
        connection.execute("DELETE FROM ml_runs")
        connection.execute("DELETE FROM ml_split_manifest")
        connection.execute(
            f"DELETE FROM building_catalog WHERE building_id NOT IN ({placeholders})",
            ids,
        )
        connection.commit()
        connection.execute("VACUUM")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"Result SQLite integrity failed: {integrity}")
    finally:
        connection.close()


def report_rows(
    connection: sqlite3.Connection, ids: list[str], batch_id: str
) -> dict[str, list[dict[str, Any]]]:
    placeholders = marks(ids)
    buildings = query_dicts(
        connection,
        f"""
        SELECT b.*, s.t1_s, s.vy_kn, s.dy_m, s.vc_kn, s.dc_m,
               s.vu_kn, s.du_m, s.energy_error,
               s.pre_capping_energy_error, s.post_capping_energy_error,
               s.normalized_rmse, s.run_end_displacement_m,
               s.run_end_shear_kn, s.run_end_shear_ratio_vc,
               s.spo_termination_reason, s.mechanism_class_at_run_end,
               f.theta_io_g, f.beta_io, f.theta_ls_g, f.beta_ls,
               f.theta_cp_g, f.beta_cp, f.n_pairs,
               f.validation_message AS fragility_validation_message
        FROM building_catalog b
        LEFT JOIN spo_features s USING(building_id)
        LEFT JOIN fragility_targets f USING(building_id)
        WHERE b.building_id IN ({placeholders})
        ORDER BY b.queue_rank
        """,
        ids,
    )
    tables: dict[str, list[dict[str, Any]]] = {"building_summary": buildings}
    for name, order in (
        ("spo_features", "building_id"),
        ("building_ground_motion_selection", "building_id,pair_id"),
        ("ida_runs", "building_id,pair_id,target_im_g"),
        ("ida_capacities", "building_id,pair_id,limit_state"),
        ("ida_curve_diagnostics", "building_id,pair_id"),
        ("fragility_targets", "building_id"),
        ("pipeline_failures", "failure_id"),
    ):
        if name == "pipeline_failures":
            sql = (
                f"SELECT * FROM {name} WHERE building_id IN ({placeholders}) "
                f"ORDER BY {order}"
            )
        else:
            sql = (
                f"SELECT * FROM {name} WHERE building_id IN ({placeholders}) "
                f"ORDER BY {order}"
            )
        tables[name] = query_dicts(connection, sql, ids)
    replacement_table = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type='table' AND name='spo_replacement_history'
        """
    ).fetchone()
    if replacement_table is not None:
        tables["spo_quarantine"] = query_dicts(
            connection,
            """
            SELECT * FROM spo_quarantine
            WHERE batch_id=? ORDER BY slot_rank,first_quarantined_utc
            """,
            [batch_id],
        )
        tables["spo_replacement_history"] = query_dicts(
            connection,
            """
            SELECT * FROM spo_replacement_history
            WHERE batch_id=? ORDER BY slot_rank,generation
            """,
            [batch_id],
        )
    return tables


def raw_files(
    connection: sqlite3.Connection,
    ids: list[str],
    root: Path,
    batch_id: str,
) -> list[dict[str, Any]]:
    placeholders = marks(ids)
    rows = []
    for table, key, sql in (
        (
            "spo_features",
            "curve_path",
            f"SELECT building_id,curve_path FROM spo_features "
            f"WHERE building_id IN ({placeholders})",
        ),
        (
            "spo_features",
            "mechanism_history_path",
            f"SELECT building_id,mechanism_history_path FROM spo_features "
            f"WHERE building_id IN ({placeholders})",
        ),
        (
            "ida_runs",
            "result_path",
            f"SELECT building_id,pair_id,target_im_g,result_path FROM ida_runs "
            f"WHERE building_id IN ({placeholders})",
        ),
        (
            "ida_manual_point_requests",
            "result_path",
            f"SELECT building_id,pair_id,target_im_g,result_path "
            f"FROM ida_manual_point_requests "
            f"WHERE result_path IS NOT NULL "
            f"AND building_id IN ({placeholders})",
        ),
    ):
        for row in connection.execute(sql, ids):
            source = Path(str(row[key]))
            if not source.is_file() or source.stat().st_size <= 0:
                raise RuntimeError(f"Raw result file is missing: {source}")
            try:
                relative = source.absolute().relative_to(root.absolute())
                archive_relative = Path("raw") / relative
            except ValueError:
                archive_relative = (
                    Path("raw")
                    / "external"
                    / (sha256(source)[:12] + "_" + source.name)
                )
            rows.append(
                {
                    "table": table,
                    "path_column": key,
                    "building_id": str(row["building_id"]),
                    "original_path": str(source),
                    "archive_path": archive_relative.as_posix(),
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256(source),
                    "_source": source,
                }
            )
    replacement_table = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type='table' AND name='spo_quarantine'
        """
    ).fetchone()
    if replacement_table is not None:
        for row in connection.execute(
            """
            SELECT building_id,curve_path,mechanism_history_path
            FROM spo_quarantine WHERE batch_id=?
            """,
            (batch_id,),
        ):
            for key in ("curve_path", "mechanism_history_path"):
                if not row[key]:
                    continue
                source = Path(str(row[key]))
                if not source.is_file() or source.stat().st_size <= 0:
                    continue
                try:
                    relative = source.absolute().relative_to(root.absolute())
                    archive_relative = Path("raw") / relative
                except ValueError:
                    archive_relative = (
                        Path("raw")
                        / "quarantine"
                        / (sha256(source)[:12] + "_" + source.name)
                    )
                rows.append(
                    {
                        "table": "spo_quarantine",
                        "path_column": key,
                        "building_id": str(row["building_id"]),
                        "original_path": str(source),
                        "archive_path": archive_relative.as_posix(),
                        "size_bytes": source.stat().st_size,
                        "sha256": sha256(source),
                        "_source": source,
                    }
                )
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["original_path"]
        if key in unique and unique[key]["sha256"] != row["sha256"]:
            raise RuntimeError(f"Conflicting raw file identity: {key}")
        unique[key] = row
    return list(unique.values())


def html_index(
    batch_id: str, rank_start: int, rank_end: int, rows: list[dict[str, Any]]
) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{int(row['queue_rank'])}</td>"
            f"<td><code>{html.escape(str(row['building_id']))}</code></td>"
            f"<td>{float(row['t1_s']):.3f}</td>"
            f"<td>{float(row['theta_io_g']):.4f}</td>"
            f"<td>{float(row['beta_io']):.4f}</td>"
            f"<td>{float(row['theta_ls_g']):.4f}</td>"
            f"<td>{float(row['beta_ls']):.4f}</td>"
            f"<td>{float(row['theta_cp_g']):.4f}</td>"
            f"<td>{float(row['beta_cp']):.4f}</td>"
            "</tr>"
        )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{batch_id} results</title><style>
body{{font-family:Segoe UI,Tahoma,sans-serif;margin:25px;color:#172033}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border:1px solid #dce3eb}}
th{{background:#eef3f9}}code{{font-family:Consolas,monospace}}</style></head>
<body><h1>{batch_id} — RC Fragility Results</h1>
<p>Queue ranks {rank_start}–{rank_end}; complete SPO, Full IDA and fragility
data. This is Phase 1 data generation only; no ML role is assigned.</p>
<table><thead><tr><th>Rank</th><th>Building ID</th><th>T1 (s)</th>
<th>θ IO (g)</th><th>β IO</th><th>θ LS (g)</th><th>β LS</th>
<th>θ CP (g)</th><th>β CP</th></tr></thead><tbody>
{''.join(body)}</tbody></table>
<p>See the UTF-8 CSV files and result_database.sqlite for all raw table data.
Raw SPO/mechanism/NLTHA checkpoint files are included in the ZIP.</p></body></html>"""


def export_batch(root: Path, batch_id: str) -> dict[str, Any]:
    rank_start, rank_end = BATCHES[batch_id]
    database = root / "data" / "server_batches_004_008.sqlite"
    output = root / "outputs" / batch_id
    output.mkdir(parents=True, exist_ok=True)
    with connect(database, readonly=True) as connection:
        ids = ids_for_batch(connection, batch_id)
        expected = rank_end - rank_start + 1
        if len(ids) != expected:
            raise RuntimeError(f"{batch_id}: expected {expected} IDs")
        valid_fragility = int(
            connection.execute(
                f"""
                SELECT COUNT(*) FROM fragility_targets
                WHERE valid=1 AND building_id IN ({marks(ids)})
                """,
                ids,
            ).fetchone()[0]
        )
        if valid_fragility != expected:
            raise RuntimeError(
                f"{batch_id}: only {valid_fragility}/{expected} valid fragilities"
            )
        reports = report_rows(connection, ids, batch_id)
        raw = raw_files(connection, ids, root, batch_id)
        gm_signature = ground_motion_signature(connection)
        model_rows = [
            dict(row)
            for row in connection.execute(
                f"""
                SELECT building_id,queue_rank,model_hash
                FROM building_catalog
                WHERE building_id IN ({marks(ids)})
                ORDER BY queue_rank
                """,
                ids,
            )
        ]

    csv_counts = {}
    for name, rows in reports.items():
        csv_counts[name] = csv_write(output / f"{name}.csv", rows)
    progress = progress_snapshot(database)
    batch_progress = [
        row for row in progress["buildings"] if row["batch_id"] == batch_id
    ]
    csv_counts["progress"] = csv_write(
        output / "progress.csv", batch_progress
    )
    (output / "index.html").write_text(
        html_index(
            batch_id, rank_start, rank_end, reports["building_summary"]
        ),
        encoding="utf-8",
    )
    (output / "README_RESULTS_TH.txt").write_text(
        (
            f"{batch_id}: อันดับ {rank_start}-{rank_end}\n"
            "ข้อมูลใน ZIP นี้ประกอบด้วยฐานข้อมูล SQLite, ตาราง CSV, "
            "SPO curves, mechanism histories และ NLTHA checkpoints\n"
            "ให้นำ ZIP ไปใช้กับ import_results.py เพื่อรวมกลับเครื่องหลัก "
            "โปรแกรมจะตรวจ checksum/model/config และสำรองฐานข้อมูลก่อนรวม\n"
        ),
        encoding="utf-8",
    )

    temporary_root = root / "runtime_storage" / "export_temp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    result_database = temporary_root / f"{batch_id}_result.sqlite"
    make_result_database(database, result_database, ids)
    static_files = [
        *sorted(output.glob("*.csv")),
        output / "index.html",
        output / "README_RESULTS_TH.txt",
        result_database,
    ]
    file_entries = [
        {
            "archive_path": (
                "result_database.sqlite"
                if path == result_database
                else f"reports/{path.name}"
            ),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "_source": path,
        }
        for path in static_files
    ]
    controller = root / "models" / "active_ida_controller_v1.joblib"
    manifest = {
        "schema": RESULT_SCHEMA,
        "created_utc": utc_now(),
        "batch_id": batch_id,
        "queue_rank_range": [rank_start, rank_end],
        "building_count": len(ids),
        "buildings": model_rows,
        "spo_replacements": reports.get(
            "spo_replacement_history", []
        ),
        "spo_quarantine_count": len(
            reports.get("spo_quarantine", [])
        ),
        "phase": "data_generation_only_no_ml",
        "scientific_config_signature": scientific_config_signature(
            root / "config" / "poc.json"
        ),
        "source_code_signature": source_code_signature(
            root / "src" / "fragility_poc"
        ),
        "ground_motion_signature": gm_signature,
        "controller_model_sha256": sha256(controller),
        "table_row_counts": csv_counts,
        "raw_file_relocation": [
            {key: value for key, value in row.items() if key != "_source"}
            for row in raw
        ],
        "files": [
            {key: value for key, value in row.items() if key != "_source"}
            for row in file_entries
        ]
        + [
            {
                "archive_path": row["archive_path"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
            for row in raw
        ],
    }
    atomic_json(output / "result_manifest.json", manifest)
    manifest_entry = {
        "archive_path": "result_manifest.json",
        "size_bytes": (output / "result_manifest.json").stat().st_size,
        "sha256": sha256(output / "result_manifest.json"),
        "_source": output / "result_manifest.json",
    }
    archive_prefix = (
        f"Batch_{int(batch_id[-3:]):03d}_Result_Ranks_"
        f"{rank_start}_{rank_end}"
    )
    zip_path = (
        root / "exports" / "READY_TO_COPY" / f"{archive_prefix}.zip"
    )
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip = zip_path.with_suffix(".zip.tmp")
    temporary_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(
        temporary_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for item in [*file_entries, manifest_entry]:
            archive.write(
                item["_source"],
                f"{archive_prefix}/{item['archive_path']}",
            )
        for item in raw:
            archive.write(
                item["_source"],
                f"{archive_prefix}/{item['archive_path']}",
            )
    os.replace(temporary_zip, zip_path)
    result_database.unlink(missing_ok=True)
    manifest["zip_path"] = str(zip_path)
    manifest["zip_size_bytes"] = zip_path.stat().st_size
    manifest["zip_sha256"] = sha256(zip_path)
    atomic_json(output / "result_manifest.json", manifest)
    return manifest


def export_all(root: Path) -> dict[str, Any]:
    zips = []
    for batch_id, (start, end) in BATCHES.items():
        path = (
            root
            / "exports"
            / "READY_TO_COPY"
            / f"Batch_{int(batch_id[-3:]):03d}_Result_Ranks_{start}_{end}.zip"
        )
        if not path.is_file():
            raise RuntimeError(f"Per-batch result ZIP is missing: {path}")
        zips.append(path)
    target = (
        root
        / "exports"
        / "READY_TO_COPY"
        / "Server_Batches_004_008_All_Results.zip"
    )
    temporary = target.with_suffix(".zip.tmp")
    temporary.unlink(missing_ok=True)
    combined = {
        "schema": "rc-fragility-combined-results-v1",
        "created_utc": utc_now(),
        "included_batches": list(BATCHES),
        "files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in zips
        ],
    }
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "COMBINED_MANIFEST.json",
            json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        )
        for path in zips:
            archive.write(path, path.name)
    os.replace(temporary, target)
    combined.update(
        {
            "zip_path": str(target),
            "zip_size_bytes": target.stat().st_size,
            "zip_sha256": sha256(target),
        }
    )
    atomic_json(root / "outputs" / "combined_result_manifest.json", combined)
    print(json.dumps(combined, ensure_ascii=False, indent=2))
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch", choices=list(BATCHES))
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    root = args.root.absolute()
    if args.all:
        export_all(root)
    else:
        result = export_batch(root, str(args.batch))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
