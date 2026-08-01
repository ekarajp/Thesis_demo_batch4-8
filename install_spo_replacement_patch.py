"""Install the SPO quarantine/replacement policy into an existing server run."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from portable_common import (
    atomic_json,
    online_backup,
    process_is_running,
    read_json,
    sha256,
    utc_now,
)
from replacement_policy import (
    ensure_replacement_schema,
    initialize_batch_slots,
    unresolved_current_spo_failures,
)


ROOT = Path(__file__).absolute().parent
DATABASE = ROOT / "data" / "server_batches_004_008.sqlite"
RESERVE = ROOT / "data" / "spo_replacement_reserve.sqlite"
LOCK = ROOT / "runtime" / "orchestrator.lock.json"


def main() -> int:
    lock = read_json(LOCK, {})
    runner_pid = int(lock.get("pid") or 0)
    if process_is_running(runner_pid):
        raise RuntimeError(
            f"Server orchestrator PID {runner_pid} is still running. "
            "Press the dashboard Pause button and wait for paused status "
            "before installing this patch."
        )
    if not DATABASE.is_file():
        raise FileNotFoundError(DATABASE)
    if not RESERVE.is_file():
        raise FileNotFoundError(RESERVE)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = (
        ROOT
        / "backups"
        / f"pre_spo_replacement_policy_{stamp}.sqlite"
    )
    online_backup(DATABASE, backup)

    connection = sqlite3.connect(DATABASE, timeout=60.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        ensure_replacement_schema(connection)
        initialize_batch_slots(connection)
        connection.commit()
        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
        foreign = list(connection.execute("PRAGMA foreign_key_check"))
        active = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM building_catalog
                WHERE selected=1 AND valid=1
                  AND queue_rank BETWEEN 151 AND 375
                """
            ).fetchone()[0]
        )
        slots = int(
            connection.execute(
                "SELECT COUNT(*) FROM batch_building_slots"
            ).fetchone()[0]
        )
        failures = []
        for batch_id in (
            "batch_004",
            "batch_005",
            "batch_006",
            "batch_007",
            "batch_008",
        ):
            failures.extend(
                {
                    "batch_id": batch_id,
                    "slot_rank": row["slot_rank"],
                    "building_id": row["building_id"],
                }
                for row in unresolved_current_spo_failures(
                    connection, batch_id
                )
            )
    finally:
        connection.close()

    reserve_connection = sqlite3.connect(RESERVE, timeout=60.0)
    try:
        reserve_integrity = reserve_connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
        reserve_count = int(
            reserve_connection.execute(
                """
                SELECT COUNT(*) FROM building_catalog
                WHERE valid=1 AND selected=0
                """
            ).fetchone()[0]
        )
    finally:
        reserve_connection.close()

    if integrity != "ok" or foreign or active != 225 or slots != 225:
        raise RuntimeError(
            "Patch migration validation failed: "
            f"integrity={integrity}, foreign={len(foreign)}, "
            f"active={active}, slots={slots}"
        )
    if reserve_integrity != "ok" or reserve_count < 225:
        raise RuntimeError(
            "SPO replacement reserve validation failed: "
            f"integrity={reserve_integrity}, count={reserve_count}"
        )
    report = {
        "schema": "spo-replacement-patch-install-v1",
        "installed_utc": utc_now(),
        "status": "PASS",
        "database": str(DATABASE),
        "database_sha256_after_schema_migration": sha256(DATABASE),
        "backup": str(backup),
        "backup_sha256": sha256(backup),
        "active_building_count": active,
        "slot_count": slots,
        "reserve_model_count": reserve_count,
        "evidenced_spo_failures_ready_for_quarantine_on_resume": failures,
        "note": (
            "No Building ID was replaced during installation. On Resume, "
            "the orchestrator will atomically quarantine each evidenced "
            "invalid SPO and refill the same slot before running SPO again. "
            "Each other building is released to Full IDA immediately after "
            "its own valid SPO and GM selection; batch-wide SPO completion "
            "is not an IDA gate. Each IDA-complete building is released to "
            "fragility fitting without waiting for the rest of the batch."
        ),
    }
    report_path = ROOT / "runtime" / "spo_replacement_patch_install.json"
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
