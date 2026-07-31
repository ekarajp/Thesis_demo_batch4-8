"""Relocate paths safely after the portable ZIP is unpacked."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

from portable_common import PACKAGE_SCHEMA, atomic_json, read_json, sha256, utc_now


def verify_initial_package(root: Path) -> dict:
    manifest_path = root / "package_manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != PACKAGE_SCHEMA:
        raise RuntimeError("Missing or incompatible package_manifest.json")
    problems = []
    for item in manifest.get("required_files", []):
        path = root / str(item["path"])
        if not path.is_file():
            problems.append(f"missing: {item['path']}")
            continue
        if path.stat().st_size != int(item["size_bytes"]):
            problems.append(f"size mismatch: {item['path']}")
            continue
        if sha256(path) != str(item["sha256"]):
            problems.append(f"checksum mismatch: {item['path']}")
    if problems:
        raise RuntimeError(
            "Portable package verification failed: " + "; ".join(problems)
        )
    return manifest


def rewrite_ground_motion_paths(root: Path, database: Path) -> int:
    processed = (
        root
        / "data"
        / "ground_motion"
        / "processed"
        / "analysis_v2_dt001_pwsa150_350"
    ).absolute()
    connection = sqlite3.connect(database, timeout=60.0)
    connection.row_factory = sqlite3.Row
    try:
        rows = list(
            connection.execute(
                """
                SELECT pair_id, component_x_path, component_y_path,
                       sha256_x, sha256_y
                FROM ground_motion_catalog
                WHERE valid=1
                ORDER BY pair_id
                """
            )
        )
        if len(rows) != 24:
            raise RuntimeError(
                f"Expected 24 valid ground-motion pairs; found {len(rows)}"
            )
        for row in rows:
            # The SQLite catalog stores absolute Windows paths (backslash
            # separators). Normalize to forward slashes before taking the base
            # name so relocation also works on POSIX, where '\' is a legal path
            # character rather than a separator. No-op on Windows.
            x_path = processed / Path(str(row["component_x_path"]).replace("\\", "/")).name
            y_path = processed / Path(str(row["component_y_path"]).replace("\\", "/")).name
            if not x_path.is_file() or not y_path.is_file():
                raise FileNotFoundError(
                    f"Processed GM files missing for {row['pair_id']}"
                )
            if sha256(x_path) != str(row["sha256_x"]):
                raise RuntimeError(f"X checksum mismatch: {row['pair_id']}")
            if sha256(y_path) != str(row["sha256_y"]):
                raise RuntimeError(f"Y checksum mismatch: {row['pair_id']}")
            connection.execute(
                """
                UPDATE ground_motion_catalog
                SET component_x_path=?, component_y_path=?
                WHERE pair_id=?
                """,
                (str(x_path), str(y_path), str(row["pair_id"])),
            )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity failed: {integrity}")
        return len(rows)
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    root = args.root.absolute()
    marker_path = root / "runtime" / "prepared.json"
    marker = read_json(marker_path, {})
    previous_root = str(marker.get("package_root", ""))

    if not marker:
        manifest = verify_initial_package(root)
    else:
        manifest = read_json(root / "package_manifest.json")
        if not isinstance(manifest, dict):
            raise RuntimeError("package_manifest.json is unreadable")

    for relative in (
        "outputs",
        "runs",
        "exports/READY_TO_COPY",
        "logs",
        "runtime",
        "runtime_storage",
        "backups",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    database = root / "data" / "server_batches_004_008.sqlite"
    pair_count = rewrite_ground_motion_paths(root, database)
    config = json.loads(
        (root / "config" / "poc.json").read_text(encoding="utf-8")
    )
    model = root / str(config["ida"]["controller"]["model_path"])
    expected_model_hash = str(config["ida"]["controller"]["model_sha256"])
    if not model.is_file() or sha256(model) != expected_model_hash:
        raise RuntimeError("Frozen active-IDA controller is missing or altered")

    prepared = {
        "schema": "portable-runtime-prepared-v1",
        "prepared_utc": utc_now(),
        "package_root": str(root),
        "relocated_from": previous_root or None,
        "database_path": str(database),
        "config_path": str(root / "config" / "poc.json"),
        "valid_ground_motion_pairs": pair_count,
        "controller_sha256": expected_model_hash,
        "package_schema": manifest.get("schema"),
        "pid": os.getpid(),
    }
    atomic_json(marker_path, prepared)
    print(json.dumps(prepared, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
