"""Package the complete five-storey design catalogue for SPO replacement.

Every enumerated five-storey row is retained for research provenance,
including globally selected and engineering-invalid rows. Runtime replacement
is restricted to ``valid=1 AND selected=0`` so a failed SPO can draw from
every feasible model not already allocated to the frozen 375-building study,
without duplicating another production Building ID.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import zlib
from pathlib import Path

from portable_common import atomic_json, sha256, utc_now


COMPRESSED_COLUMN = "design_metadata_json"


def _reserve_value(column: str, value: object) -> object:
    """Losslessly compress the large per-building JSON payload."""
    if column != COMPRESSED_COLUMN or value is None:
        return value
    return sqlite3.Binary(zlib.compress(str(value).encode("utf-8"), level=9))


def build(source: Path, target: Path, report_path: Path) -> dict[str, object]:
    source_connection = sqlite3.connect(source, timeout=60.0)
    source_connection.row_factory = sqlite3.Row
    try:
        targets = [
            dict(row)
            for row in source_connection.execute(
                """
                SELECT * FROM building_catalog
                WHERE selected=1 AND valid=1
                  AND queue_rank BETWEEN 151 AND 375
                ORDER BY queue_rank
                """
            )
        ]
        catalogue = [
            dict(row)
            for row in source_connection.execute(
                """
                SELECT * FROM building_catalog
                WHERE stories=5
                ORDER BY COALESCE(queue_rank,2147483647),building_id
                """
            )
        ]
        if len(targets) != 225:
            raise RuntimeError(
                f"Expected 225 frozen server targets, found {len(targets)}"
            )
        if not catalogue:
            raise RuntimeError("The five-storey catalogue is empty")
        catalogue_ids = {
            str(row["building_id"]) for row in catalogue
        }
        if len(catalogue_ids) != len(catalogue):
            raise RuntimeError("The full catalogue contains duplicate IDs")
        eligible = [
            row
            for row in catalogue
            if int(row["valid"]) == 1 and int(row["selected"]) == 0
        ]
        selected_global = [
            row
            for row in catalogue
            if int(row["valid"]) == 1 and int(row["selected"]) == 1
        ]
        invalid = [row for row in catalogue if int(row["valid"]) != 1]

        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        destination = sqlite3.connect(temporary)
        try:
            table_sql = source_connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type='table' AND name='building_catalog'
                """
            ).fetchone()[0]
            destination.execute(table_sql)
            columns = [
                str(row["name"])
                for row in source_connection.execute(
                    "PRAGMA table_info(building_catalog)"
                )
            ]
            sql = (
                f"INSERT INTO building_catalog({','.join(columns)}) "
                f"VALUES({','.join('?' for _ in columns)})"
            )
            for row in catalogue:
                destination.execute(
                    sql,
                    [
                        _reserve_value(column, row[column])
                        for column in columns
                    ],
                )
            destination.execute(
                """
                CREATE INDEX idx_reserve_strata ON building_catalog(
                    number_of_bays,bay_width_m,fc_ksc,sdl_kg_m2,ll_kg_m2,
                    beam_tier,column_tier,scwb_class
                )
                """
            )
            destination.commit()
            destination.execute("VACUUM")
            integrity = destination.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(
                    f"Reserve SQLite integrity failed: {integrity}"
                )
        finally:
            destination.close()
        os.replace(temporary, target)
    finally:
        source_connection.close()

    report: dict[str, object] = {
        "schema": "full-five-storey-catalog-v2",
        "created_utc": utc_now(),
        "source_database": str(source),
        "source_database_sha256": sha256(source),
        "target_database": str(target),
        "target_database_sha256": sha256(target),
        "server_target_count": len(targets),
        "catalogue_model_count": len(catalogue),
        "valid_model_count": len(catalogue) - len(invalid),
        "invalid_model_count": len(invalid),
        "globally_selected_valid_model_count": len(selected_global),
        "eligible_replacement_model_count": len(eligible),
        "replacement_filter": "stories=5 AND valid=1 AND selected=0",
        "catalogue_is_complete_for_source_database": True,
        "design_metadata_storage": "zlib-compressed-utf8-lossless",
    }
    atomic_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    report = build(
        arguments.source.absolute(),
        arguments.target.absolute(),
        arguments.report.absolute(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
