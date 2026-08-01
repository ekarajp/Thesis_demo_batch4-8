"""Deterministic SPO quarantine and balanced nearest-replacement policy.

The expensive structural-analysis code remains fail-closed: an invalid SPO is
never relabelled as valid and never proceeds to IDA.  This module changes only
the orchestration population.  A failed Building ID is retained for later
repair, while its frozen queue slot is assigned to the nearest unused valid
model from the reserve design-space catalogue.
"""

from __future__ import annotations

import json
import math
import sqlite3
import zlib
from pathlib import Path
from typing import Any

from portable_common import BATCHES, connect, utc_now


REPLACEMENT_POLICY_SCHEMA = "spo-nearest-replacement-v1"
MAX_REPLACEMENTS_PER_SLOT = 20


def ensure_replacement_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS batch_building_slots (
            batch_id TEXT NOT NULL,
            slot_rank INTEGER NOT NULL,
            original_building_id TEXT NOT NULL,
            current_building_id TEXT NOT NULL,
            replacement_generation INTEGER NOT NULL DEFAULT 0,
            updated_utc TEXT NOT NULL,
            PRIMARY KEY(batch_id, slot_rank),
            UNIQUE(current_building_id)
        );

        CREATE TABLE IF NOT EXISTS spo_quarantine (
            building_id TEXT PRIMARY KEY,
            batch_id TEXT NOT NULL,
            slot_rank INTEGER NOT NULL,
            model_hash TEXT NOT NULL,
            failure_count INTEGER NOT NULL,
            quarantine_reason TEXT NOT NULL,
            spo_termination_reason TEXT,
            validation_message TEXT,
            analysis_signature TEXT,
            curve_path TEXT,
            mechanism_history_path TEXT,
            first_quarantined_utc TEXT NOT NULL,
            last_quarantined_utc TEXT NOT NULL,
            replacement_building_id TEXT,
            replacement_distance REAL,
            details_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS spo_replacement_history (
            batch_id TEXT NOT NULL,
            slot_rank INTEGER NOT NULL,
            generation INTEGER NOT NULL,
            original_building_id TEXT NOT NULL,
            failed_building_id TEXT NOT NULL,
            replacement_building_id TEXT NOT NULL,
            replacement_model_hash TEXT NOT NULL,
            replacement_distance REAL NOT NULL,
            policy_schema TEXT NOT NULL,
            created_utc TEXT NOT NULL,
            details_json TEXT NOT NULL,
            PRIMARY KEY(batch_id, slot_rank, generation),
            UNIQUE(replacement_building_id)
        );

        CREATE INDEX IF NOT EXISTS idx_spo_quarantine_batch
        ON spo_quarantine(batch_id, slot_rank);

        CREATE INDEX IF NOT EXISTS idx_spo_replacement_history_batch
        ON spo_replacement_history(batch_id, slot_rank);
        """
    )


def initialize_batch_slots(connection: sqlite3.Connection) -> None:
    """Create immutable slot identities while preserving migrated assignments."""
    ensure_replacement_schema(connection)
    now = utc_now()
    for batch_id, (rank_start, rank_end) in BATCHES.items():
        for row in connection.execute(
            """
            SELECT building_id, queue_rank
            FROM building_catalog
            WHERE selected=1 AND valid=1
              AND queue_rank BETWEEN ? AND ?
            ORDER BY queue_rank
            """,
            (rank_start, rank_end),
        ):
            rank = int(row["queue_rank"])
            building_id = str(row["building_id"])
            connection.execute(
                """
                INSERT OR IGNORE INTO batch_building_slots(
                    batch_id,slot_rank,original_building_id,
                    current_building_id,replacement_generation,updated_utc
                ) VALUES(?,?,?,?,0,?)
                """,
                (batch_id, rank, building_id, building_id, now),
            )


def ids_for_active_batch(
    connection: sqlite3.Connection, batch_id: str
) -> list[str]:
    initialize_batch_slots(connection)
    rows = list(
        connection.execute(
            """
            SELECT slot_rank,current_building_id
            FROM batch_building_slots
            WHERE batch_id=?
            ORDER BY slot_rank
            """,
            (batch_id,),
        )
    )
    return [str(row["current_building_id"]) for row in rows]


def _latest_failure(
    connection: sqlite3.Connection, building_id: str
) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT failure_id,created_utc,error_type,message,details_json
        FROM pipeline_failures
        WHERE stage='SPO' AND building_id=? AND resolved=0
        ORDER BY failure_id DESC
        LIMIT 1
        """,
        (building_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def unresolved_current_spo_failures(
    connection: sqlite3.Connection, batch_id: str
) -> list[dict[str, Any]]:
    """Return only evidenced failures, never merely unprocessed buildings."""
    initialize_batch_slots(connection)
    failures: list[dict[str, Any]] = []
    for slot in connection.execute(
        """
        SELECT slot_rank,original_building_id,current_building_id,
               replacement_generation
        FROM batch_building_slots
        WHERE batch_id=?
        ORDER BY slot_rank
        """,
        (batch_id,),
    ):
        building_id = str(slot["current_building_id"])
        spo = connection.execute(
            """
            SELECT valid,spo_termination_reason,validation_message,
                   analysis_signature,curve_path,mechanism_history_path,
                   analysis_guard_triggered,collapse_classification,
                   runtime_s
            FROM spo_features WHERE building_id=?
            """,
            (building_id,),
        ).fetchone()
        pipeline_failure = _latest_failure(connection, building_id)
        # A current valid SPO is authoritative. A stale unresolved row from an
        # earlier failed attempt must never replace a building that later
        # converged and passed all SPO QC checks.
        if spo is not None and int(spo["valid"]) == 1:
            continue
        if not (
            (spo is not None and int(spo["valid"]) == 0)
            or pipeline_failure is not None
        ):
            continue
        building = connection.execute(
            "SELECT * FROM building_catalog WHERE building_id=?",
            (building_id,),
        ).fetchone()
        if building is None:
            raise RuntimeError(
                f"Current batch slot references missing building {building_id}"
            )
        failures.append(
            {
                "batch_id": batch_id,
                "slot_rank": int(slot["slot_rank"]),
                "original_building_id": str(slot["original_building_id"]),
                "building_id": building_id,
                "replacement_generation": int(
                    slot["replacement_generation"]
                ),
                "building": dict(building),
                "spo": dict(spo) if spo is not None else None,
                "pipeline_failure": pipeline_failure,
            }
        )
    return failures


def _normalized_difference(
    left: dict[str, Any], right: dict[str, Any], key: str, span: float
) -> float:
    return abs(float(left[key]) - float(right[key])) / float(span)


def replacement_distance(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    """Research-strata distance used for deterministic nearest replacement.

    Geometry and strength tiers receive the greatest weight. Material and
    gravity-load strata follow. SCWB class and two continuous response/design
    indicators break otherwise similar choices.  No SPO/IDA/fragility result
    is used, so replacement cannot leak downstream response information.
    """
    components = {
        "number_of_bays": 2.0
        * _normalized_difference(
            reference, candidate, "number_of_bays", 4.0
        ),
        "bay_width_m": 2.0
        * _normalized_difference(reference, candidate, "bay_width_m", 4.0),
        "fc_ksc": 1.5
        * _normalized_difference(reference, candidate, "fc_ksc", 210.0),
        "sdl_kg_m2": 1.5
        * _normalized_difference(reference, candidate, "sdl_kg_m2", 300.0),
        "ll_kg_m2": 1.5
        * _normalized_difference(reference, candidate, "ll_kg_m2", 400.0),
        "beam_tier": 2.0
        * _normalized_difference(reference, candidate, "beam_tier", 5.0),
        "column_tier": 2.0
        * _normalized_difference(reference, candidate, "column_tier", 5.0),
        "scwb_class": (
            0.0
            if str(reference.get("scwb_class"))
            == str(candidate.get("scwb_class"))
            else 1.5
        ),
        "axial_ratio": 0.5
        * _normalized_difference(
            reference, candidate, "axial_ratio", 0.6
        ),
        "scwb_strength_ratio": 0.5
        * abs(
            math.log(
                max(float(reference["scwb_strength_ratio"]), 1.0e-9)
                / max(float(candidate["scwb_strength_ratio"]), 1.0e-9)
            )
        ),
    }
    return float(sum(components.values())), components


def _reserve_candidates(
    reserve_database: Path,
) -> list[dict[str, Any]]:
    if not reserve_database.is_file():
        raise FileNotFoundError(
            f"SPO replacement reserve database is missing: {reserve_database}"
        )
    connection = sqlite3.connect(reserve_database, timeout=60.0)
    connection.row_factory = sqlite3.Row
    try:
        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(
                f"Replacement reserve integrity failed: {integrity}"
            )
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM building_catalog
                WHERE valid=1 AND selected=0
                ORDER BY building_id
                """
            )
        ]
    finally:
        connection.close()


def _reference_building(
    connection: sqlite3.Connection, failure: dict[str, Any]
) -> dict[str, Any]:
    row = connection.execute(
        "SELECT * FROM building_catalog WHERE building_id=?",
        (failure["original_building_id"],),
    ).fetchone()
    if row is None:
        return dict(failure["building"])
    return dict(row)


def _candidate_is_available(
    connection: sqlite3.Connection, candidate: dict[str, Any]
) -> bool:
    building_id = str(candidate["building_id"])
    model_hash = str(candidate["model_hash"])
    selected = connection.execute(
        """
        SELECT 1 FROM building_catalog
        WHERE selected=1 AND (building_id=? OR model_hash=?)
        LIMIT 1
        """,
        (building_id, model_hash),
    ).fetchone()
    if selected is not None:
        return False
    used = connection.execute(
        """
        SELECT 1 FROM spo_replacement_history
        WHERE replacement_building_id=? OR replacement_model_hash=?
        LIMIT 1
        """,
        (building_id, model_hash),
    ).fetchone()
    return used is None


def _insert_or_reset_reserve_row(
    connection: sqlite3.Connection,
    candidate: dict[str, Any],
    slot_rank: int,
) -> None:
    columns = [
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(building_catalog)")
    ]
    missing = [column for column in columns if column not in candidate]
    if missing:
        raise RuntimeError(
            f"Replacement reserve row lacks building columns: {missing}"
        )
    payload = dict(candidate)
    compressed_metadata = payload.get("design_metadata_json")
    if isinstance(compressed_metadata, bytes):
        try:
            payload["design_metadata_json"] = zlib.decompress(
                compressed_metadata
            ).decode("utf-8")
        except (zlib.error, UnicodeDecodeError) as exc:
            raise RuntimeError(
                "Replacement catalog contains corrupt compressed "
                "design_metadata_json"
            ) from exc
    payload["selected"] = 1
    payload["queue_rank"] = int(slot_rank)
    payload["valid"] = 1
    existing = connection.execute(
        "SELECT selected FROM building_catalog WHERE building_id=?",
        (payload["building_id"],),
    ).fetchone()
    if existing is None:
        connection.execute(
            f"""
            INSERT INTO building_catalog({','.join(columns)})
            VALUES({','.join('?' for _ in columns)})
            """,
            [payload[column] for column in columns],
        )
    else:
        connection.execute(
            """
            UPDATE building_catalog
            SET selected=1,queue_rank=?,valid=1,invalid_reason=NULL
            WHERE building_id=?
            """,
            (slot_rank, payload["building_id"]),
        )


def quarantine_and_replace_failures(
    database: Path,
    reserve_database: Path,
    batch_id: str,
) -> dict[str, Any]:
    """Atomically quarantine every evidenced failure and refill its slot."""
    with connect(database) as preview_connection:
        initialize_batch_slots(preview_connection)
        preview_failures = unresolved_current_spo_failures(
            preview_connection, batch_id
        )
    if not preview_failures:
        return {
            "schema": REPLACEMENT_POLICY_SCHEMA,
            "batch_id": batch_id,
            "replacement_count": 0,
            "replacements": [],
        }
    reserve = _reserve_candidates(reserve_database)
    now = utc_now()
    replacements: list[dict[str, Any]] = []
    with connect(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            initialize_batch_slots(connection)
            failures = unresolved_current_spo_failures(
                connection, batch_id
            )
            for failure in failures:
                generation = int(failure["replacement_generation"]) + 1
                if generation > MAX_REPLACEMENTS_PER_SLOT:
                    raise RuntimeError(
                        f"{batch_id} slot {failure['slot_rank']} exhausted "
                        f"{MAX_REPLACEMENTS_PER_SLOT} replacements"
                    )
                reference = _reference_building(connection, failure)
                ranked: list[
                    tuple[float, str, str, dict[str, Any], dict[str, float]]
                ] = []
                for candidate in reserve:
                    if int(candidate.get("stories") or 0) != int(
                        reference["stories"]
                    ):
                        continue
                    if not _candidate_is_available(connection, candidate):
                        continue
                    distance, components = replacement_distance(
                        reference, candidate
                    )
                    ranked.append(
                        (
                            distance,
                            str(candidate["model_hash"]),
                            str(candidate["building_id"]),
                            candidate,
                            components,
                        )
                    )
                if not ranked:
                    raise RuntimeError(
                        f"No unused valid reserve model remains for "
                        f"{batch_id} slot {failure['slot_rank']}"
                    )
                (
                    distance,
                    _model_hash,
                    _building_id,
                    candidate,
                    components,
                ) = min(ranked)
                failed_id = str(failure["building_id"])
                replacement_id = str(candidate["building_id"])
                spo = failure["spo"] or {}
                pipeline_failure = failure["pipeline_failure"] or {}
                details = {
                    "schema": REPLACEMENT_POLICY_SCHEMA,
                    "reference_building_id": str(
                        failure["original_building_id"]
                    ),
                    "failed_building_id": failed_id,
                    "replacement_building_id": replacement_id,
                    "distance_components": components,
                    "pipeline_failure": pipeline_failure,
                    "spo": spo,
                }

                # Preserve every invalid SPO and raw path. Only population
                # membership and the queue-slot assignment change.
                connection.execute(
                    """
                    UPDATE building_catalog
                    SET selected=0,queue_rank=NULL
                    WHERE building_id=?
                    """,
                    (failed_id,),
                )
                _insert_or_reset_reserve_row(
                    connection, candidate, int(failure["slot_rank"])
                )
                connection.execute(
                    """
                    INSERT INTO spo_quarantine(
                        building_id,batch_id,slot_rank,model_hash,
                        failure_count,quarantine_reason,
                        spo_termination_reason,validation_message,
                        analysis_signature,curve_path,mechanism_history_path,
                        first_quarantined_utc,last_quarantined_utc,
                        replacement_building_id,replacement_distance,
                        details_json
                    ) VALUES(?,?,?,?,1,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(building_id) DO UPDATE SET
                        failure_count=spo_quarantine.failure_count+1,
                        quarantine_reason=excluded.quarantine_reason,
                        spo_termination_reason=
                            excluded.spo_termination_reason,
                        validation_message=excluded.validation_message,
                        analysis_signature=excluded.analysis_signature,
                        curve_path=excluded.curve_path,
                        mechanism_history_path=
                            excluded.mechanism_history_path,
                        last_quarantined_utc=excluded.last_quarantined_utc,
                        replacement_building_id=
                            excluded.replacement_building_id,
                        replacement_distance=
                            excluded.replacement_distance,
                        details_json=excluded.details_json
                    """,
                    (
                        failed_id,
                        batch_id,
                        int(failure["slot_rank"]),
                        str(failure["building"]["model_hash"]),
                        str(
                            pipeline_failure.get("message")
                            or spo.get("validation_message")
                            or "SPO did not produce a valid result"
                        ),
                        spo.get("spo_termination_reason"),
                        spo.get("validation_message"),
                        spo.get("analysis_signature"),
                        spo.get("curve_path"),
                        spo.get("mechanism_history_path"),
                        now,
                        now,
                        replacement_id,
                        distance,
                        json.dumps(
                            details, ensure_ascii=False, default=str
                        ),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO spo_replacement_history(
                        batch_id,slot_rank,generation,
                        original_building_id,failed_building_id,
                        replacement_building_id,replacement_model_hash,
                        replacement_distance,policy_schema,created_utc,
                        details_json
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        batch_id,
                        int(failure["slot_rank"]),
                        generation,
                        str(failure["original_building_id"]),
                        failed_id,
                        replacement_id,
                        str(candidate["model_hash"]),
                        distance,
                        REPLACEMENT_POLICY_SCHEMA,
                        now,
                        json.dumps(
                            details, ensure_ascii=False, default=str
                        ),
                    ),
                )
                connection.execute(
                    """
                    UPDATE batch_building_slots
                    SET current_building_id=?,replacement_generation=?,
                        updated_utc=?
                    WHERE batch_id=? AND slot_rank=?
                    """,
                    (
                        replacement_id,
                        generation,
                        now,
                        batch_id,
                        int(failure["slot_rank"]),
                    ),
                )
                replacements.append(
                    {
                        "batch_id": batch_id,
                        "slot_rank": int(failure["slot_rank"]),
                        "generation": generation,
                        "failed_building_id": failed_id,
                        "replacement_building_id": replacement_id,
                        "replacement_distance": distance,
                        "distance_components": components,
                    }
                )

            active_ids = ids_for_active_batch(connection, batch_id)
            expected = (
                BATCHES[batch_id][1] - BATCHES[batch_id][0] + 1
            )
            if len(active_ids) != expected:
                raise RuntimeError(
                    f"{batch_id} has {len(active_ids)} active slots; "
                    f"expected {expected}"
                )
            duplicate_hashes = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                      SELECT model_hash
                      FROM building_catalog
                      WHERE selected=1
                      GROUP BY model_hash HAVING COUNT(*)>1
                    )
                    """
                ).fetchone()[0]
            )
            if duplicate_hashes:
                raise RuntimeError(
                    "Replacement would duplicate an active model hash"
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return {
        "schema": REPLACEMENT_POLICY_SCHEMA,
        "batch_id": batch_id,
        "replacement_count": len(replacements),
        "replacements": replacements,
    }
