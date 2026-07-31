#!/usr/bin/env python3
"""Verify processed ground-motion files against the SQLite catalog checksums.

Mirrors the integrity check in prepare_portable_runtime.rewrite_ground_motion_paths:
for each of the 24 valid ground-motion pairs, confirms both X and Y component
files exist in data/ground_motion/processed/analysis_v2_dt001_pwsa150_350/ and
match the sha256 recorded in the database.

Prints a per-file report and exits non-zero if anything is missing or mismatched,
so you know the exact moment the package is ready to run (prepare/preflight pass).

Usage:
    python verify_gm.py                 # from the repo root
    python verify_gm.py --root /path    # explicit package root
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path

PROC_REL = Path("data/ground_motion/processed/analysis_v2_dt001_pwsa150_350")
DB_REL = Path("data/server_batches_004_008.sqlite")
EXPECTED_PAIRS = 24


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_name(stored: str) -> str:
    # The catalog stores absolute Windows paths; normalize backslashes so this
    # also resolves on POSIX (see prepare_portable_runtime.py).
    return Path(stored.replace("\\", "/")).name


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    root: Path = args.root.resolve()

    db = root / DB_REL
    proc = root / PROC_REL
    if not db.is_file():
        print(f"ERROR: database not found: {db}", file=sys.stderr)
        return 2
    if not proc.is_dir():
        print(f"ERROR: processed ground-motion dir not found: {proc}", file=sys.stderr)
        return 2

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT pair_id, component_x_path, component_y_path, sha256_x, sha256_y, valid "
        "FROM ground_motion_catalog ORDER BY pair_id"
    ).fetchall()
    con.close()

    valid = [r for r in rows if r["valid"]]
    print(f"DB catalog: {len(rows)} pairs ({len(valid)} valid); expected {EXPECTED_PAIRS} valid")
    print(f"Processed dir: {proc}\n")

    ok_pairs = 0
    missing = bad = 0
    problems = []
    for r in valid:
        pair_ok = True
        for axis, path_col, sha_col in (
            ("X", "component_x_path", "sha256_x"),
            ("Y", "component_y_path", "sha256_y"),
        ):
            f = proc / base_name(str(r[path_col]))
            if not f.is_file():
                problems.append(f"  MISSING   {r['pair_id']}_{axis}  -> {f.name}")
                missing += 1
                pair_ok = False
            elif sha256(f) != str(r[sha_col]):
                problems.append(f"  MISMATCH  {r['pair_id']}_{axis}  -> {f.name}")
                bad += 1
                pair_ok = False
        if pair_ok:
            ok_pairs += 1

    total_files = len(valid) * 2
    print(f"Pairs fully OK : {ok_pairs}/{len(valid)}")
    print(f"Files OK       : {total_files - missing - bad}/{total_files}")
    print(f"Files missing  : {missing}")
    print(f"Files mismatch : {bad}")
    if problems:
        print("\nProblems (drop the correct files into the processed dir, then re-run):")
        for p in problems:
            print(p)

    ready = ok_pairs == len(valid) == EXPECTED_PAIRS
    print(
        f"\n{'✓ READY — prepare/preflight will pass.' if ready else '✗ NOT READY — see problems above.'}"
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
