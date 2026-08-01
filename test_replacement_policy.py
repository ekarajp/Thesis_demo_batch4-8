"""Regression tests for checkpoint-safe SPO population replacement."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from contextlib import nullcontext
from pathlib import Path

import pytest

import server_orchestrator
from import_results import verify_scientific_identity
from portable_common import (
    BATCHES,
    ground_motion_signature,
    ids_for_batch,
    scientific_config_signature,
    sha256,
    source_code_signature,
    progress_snapshot,
)
from replacement_policy import (
    ensure_replacement_schema,
    initialize_batch_slots,
    quarantine_and_replace_failures,
)


ROOT = Path(__file__).absolute().parent
MAIN_ROOT = ROOT.parent.parent
SOURCE_DATABASE = ROOT / "data" / "server_batches_004_008.sqlite"
RESERVE_DATABASE = ROOT / "data" / "spo_replacement_reserve.sqlite"
TEMP_ROOT = ROOT / "runtime_storage" / "replacement_policy_tests"


def _working_database() -> tuple[tempfile.TemporaryDirectory[str], Path]:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
    path = Path(temporary.name) / "test.sqlite"
    shutil.copy2(SOURCE_DATABASE, path)
    return temporary, path


def test_invalid_spo_is_retained_and_slot_is_refilled() -> None:
    temporary, database = _working_database()
    try:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            ensure_replacement_schema(connection)
            initialize_batch_slots(connection)
            original = connection.execute(
                """
                SELECT building_id,model_hash
                FROM building_catalog WHERE queue_rank=182
                """
            ).fetchone()
            original_id = str(original["building_id"])
            connection.execute(
                """
                INSERT INTO pipeline_failures(
                    stage,building_id,pair_id,created_utc,error_type,
                    message,details_json,resolved
                ) VALUES('SPO',?,NULL,'2026-08-01T00:00:00+00:00',
                         'RuntimeError','analysis_guard_max_runtime','{}',0)
                """,
                (original_id,),
            )
            connection.commit()
        finally:
            connection.close()

        outcome = quarantine_and_replace_failures(
            database, RESERVE_DATABASE, "batch_004"
        )
        assert outcome["replacement_count"] == 1
        replacement = outcome["replacements"][0]
        assert replacement["slot_rank"] == 182
        assert replacement["failed_building_id"] == original_id
        assert replacement["replacement_building_id"] != original_id

        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            active = ids_for_batch(connection, "batch_004")
            assert len(active) == 50
            assert len(set(active)) == 50
            assert original_id not in active
            assert replacement["replacement_building_id"] in active
            replacement_metadata = connection.execute(
                """
                SELECT design_metadata_json FROM building_catalog
                WHERE building_id=?
                """,
                (replacement["replacement_building_id"],),
            ).fetchone()[0]
            assert isinstance(replacement_metadata, str)
            assert isinstance(json.loads(replacement_metadata), dict)
            failed = connection.execute(
                """
                SELECT selected,queue_rank FROM building_catalog
                WHERE building_id=?
                """,
                (original_id,),
            ).fetchone()
            assert int(failed["selected"]) == 0
            assert failed["queue_rank"] is None
            assert (
                connection.execute(
                    """
                    SELECT COUNT(*) FROM pipeline_failures
                    WHERE building_id=? AND resolved=0
                    """,
                    (original_id,),
                ).fetchone()[0]
                == 1
            )
            quarantine = connection.execute(
                """
                SELECT replacement_building_id,replacement_distance
                FROM spo_quarantine WHERE building_id=?
                """,
                (original_id,),
            ).fetchone()
            assert quarantine is not None
            assert (
                str(quarantine["replacement_building_id"])
                == replacement["replacement_building_id"]
            )
            assert float(quarantine["replacement_distance"]) >= 0.0
            rank_rows = list(
                connection.execute(
                    """
                    SELECT queue_rank,COUNT(*) AS n
                    FROM building_catalog
                    WHERE selected=1 AND queue_rank BETWEEN 151 AND 375
                    GROUP BY queue_rank HAVING COUNT(*)<>1
                    """
                )
            )
            assert rank_rows == []
            progress = progress_snapshot(database)
            assert progress["total_buildings"] == 225
            assert progress["quarantined_spo_models"] == 1
            replacement_row = next(
                row
                for row in progress["buildings"]
                if row["queue_rank"] == 182
            )
            assert replacement_row["replacement_generation"] == 1
            assert (
                replacement_row["original_building_id"] == original_id
            )
        finally:
            connection.close()
    finally:
        temporary.cleanup()


def test_unprocessed_building_is_not_replaced() -> None:
    temporary, database = _working_database()
    try:
        outcome = quarantine_and_replace_failures(
            database, RESERVE_DATABASE, "batch_004"
        )
        assert outcome["replacement_count"] == 0
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            assert len(ids_for_batch(connection, "batch_004")) == (
                BATCHES["batch_004"][1] - BATCHES["batch_004"][0] + 1
            )
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM spo_quarantine"
                ).fetchone()[0]
                == 0
            )
        finally:
            connection.close()
    finally:
        temporary.cleanup()


def test_streaming_worksets_release_each_spo_ready_building_to_ida() -> None:
    ids = ["B-001", "B-002", "B-003"]
    before_selection = server_orchestrator.streaming_worksets(
        ids,
        valid_spo={"B-001"},
        selected=set(),
        ida_complete=set(),
    )
    assert before_selection == {
        "pending_spo": ["B-002", "B-003"],
        "pending_selection": ["B-001"],
        "pending_ida": [],
        "pending_fragility": [],
    }

    after_selection = server_orchestrator.streaming_worksets(
        ids,
        valid_spo={"B-001"},
        selected={"B-001"},
        ida_complete=set(),
    )
    assert after_selection["pending_spo"] == ["B-002", "B-003"]
    assert after_selection["pending_ida"] == ["B-001"]
    assert len(after_selection["pending_ida"]) == 1
    assert len(after_selection["pending_spo"]) == 2


def test_streaming_worksets_resume_only_incomplete_persisted_work() -> None:
    ids = ["B-001", "B-002", "B-003"]
    worksets = server_orchestrator.streaming_worksets(
        ids,
        valid_spo={"B-001", "B-002"},
        selected={"B-001", "B-002"},
        ida_complete={"B-001"},
    )
    assert worksets == {
        "pending_spo": ["B-003"],
        "pending_selection": [],
        "pending_ida": ["B-002"],
        "pending_fragility": ["B-001"],
    }


def test_streaming_releases_each_ida_complete_building_to_fragility() -> None:
    worksets = server_orchestrator.streaming_worksets(
        ["B-001", "B-002", "B-003"],
        valid_spo={"B-001", "B-002", "B-003"},
        selected={"B-001", "B-002", "B-003"},
        ida_complete={"B-001", "B-002"},
        fragility_complete={"B-001"},
    )
    assert worksets["pending_ida"] == ["B-003"]
    assert worksets["pending_fragility"] == ["B-002"]


def test_streaming_orchestrator_starts_ida_while_other_spo_is_running(
    monkeypatch,
    tmp_path: Path,
) -> None:
    ids = ["B-001", "B-002"]
    selected: set[str] = set()
    starts: list[tuple[str, list[str]]] = []

    class _Running:
        returncode = None

        def __init__(self, pid: int) -> None:
            self.pid = pid

        def poll(self):
            return self.returncode

    def _passing(_batch, stage, requested):
        requested_set = set(requested)
        if stage == "modal_spo":
            return {"B-001"} & requested_set
        if stage == "select_records":
            return selected & requested_set
        if stage == "full_ida":
            return set()
        if stage == "fragility":
            return set()
        raise AssertionError(stage)

    def _start(**kwargs):
        role = str(kwargs["role"])
        work_ids = list(kwargs["ids"])
        starts.append((role, work_ids))
        process = _Running(1100 if role == "spo" else 1200)
        return {
            "role": role,
            "wave": kwargs["wave"],
            "ids": work_ids,
            "command": [role],
            "process": process,
            "pid": process.pid,
            "started_utc": "now",
            "started_monotonic": 0.0,
            "stdout_path": "stdout",
            "stderr_path": "stderr",
        }

    def _select(_batch, work_ids, wave):
        del wave
        selected.update(work_ids)
        return {
            "ids": list(work_ids),
            "return_code": 0,
        }

    def _pause_after_first_loop(_seconds):
        server_orchestrator.PAUSE_REQUEST.write_text(
            "{}", encoding="utf-8"
        )

    def _terminate(process):
        process.returncode = -1
        return {"pid": process.pid, "return_code": -1}

    monkeypatch.setattr(
        server_orchestrator, "PAUSE_REQUEST", tmp_path / "pause.json"
    )
    monkeypatch.setattr(
        server_orchestrator, "connect", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(
        server_orchestrator, "ids_for_batch", lambda *_args: list(ids)
    )
    monkeypatch.setattr(
        server_orchestrator, "expected_batch_count", lambda _batch: 2
    )
    monkeypatch.setattr(
        server_orchestrator,
        "_ids_passing_postcondition",
        _passing,
    )
    monkeypatch.setattr(
        server_orchestrator,
        "quarantine_and_replace_failures",
        lambda *_args: {"replacement_count": 0},
    )
    monkeypatch.setattr(
        server_orchestrator, "_start_stream_child", _start
    )
    monkeypatch.setattr(
        server_orchestrator, "_run_stream_selection", _select
    )
    monkeypatch.setattr(
        server_orchestrator, "save_state", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        server_orchestrator, "event", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        server_orchestrator, "online_backup", lambda *_args: None
    )
    monkeypatch.setattr(
        server_orchestrator, "terminate_tree", _terminate
    )
    monkeypatch.setattr(
        server_orchestrator.time, "sleep", _pause_after_first_loop
    )

    with pytest.raises(InterruptedError, match="Pause requested"):
        server_orchestrator.run_streaming_spo_ida(
            server_orchestrator.default_state(),
            "batch_004",
            workers=6,
        )
    assert starts[:2] == [
        ("spo", ["B-002"]),
        ("ida", ["B-001"]),
    ]


def test_streaming_orchestrator_fits_fragility_while_other_ida_is_running(
    monkeypatch,
    tmp_path: Path,
) -> None:
    ids = ["B-001", "B-002"]
    starts: list[tuple[str, list[str]]] = []
    fragility_done: set[str] = set()

    class _Running:
        returncode = None

        def __init__(self, pid: int) -> None:
            self.pid = pid

        def poll(self):
            return self.returncode

    def _passing(_batch, stage, requested):
        requested_set = set(requested)
        values = {
            "modal_spo": set(ids),
            "select_records": set(ids),
            "full_ida": {"B-001"},
            "fragility": fragility_done,
        }
        return values[stage] & requested_set

    def _start(**kwargs):
        role = str(kwargs["role"])
        work_ids = list(kwargs["ids"])
        starts.append((role, work_ids))
        process = _Running(2200)
        return {
            "role": role,
            "wave": kwargs["wave"],
            "ids": work_ids,
            "command": [role],
            "process": process,
            "pid": process.pid,
            "started_utc": "now",
            "started_monotonic": 0.0,
            "stdout_path": "stdout",
            "stderr_path": "stderr",
        }

    def _fit(_batch, work_ids, wave):
        del wave
        assert starts == [("ida", ["B-002"])]
        fragility_done.update(work_ids)
        return {
            "ids": list(work_ids),
            "return_code": 0,
        }

    def _pause_after_first_loop(_seconds):
        server_orchestrator.PAUSE_REQUEST.write_text(
            "{}", encoding="utf-8"
        )

    def _terminate(process):
        process.returncode = -1
        return {"pid": process.pid, "return_code": -1}

    monkeypatch.setattr(
        server_orchestrator, "PAUSE_REQUEST", tmp_path / "pause.json"
    )
    monkeypatch.setattr(
        server_orchestrator, "connect", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(
        server_orchestrator, "ids_for_batch", lambda *_args: list(ids)
    )
    monkeypatch.setattr(
        server_orchestrator, "expected_batch_count", lambda _batch: 2
    )
    monkeypatch.setattr(
        server_orchestrator, "_ids_passing_postcondition", _passing
    )
    monkeypatch.setattr(
        server_orchestrator,
        "quarantine_and_replace_failures",
        lambda *_args: {"replacement_count": 0},
    )
    monkeypatch.setattr(
        server_orchestrator, "_start_stream_child", _start
    )
    monkeypatch.setattr(
        server_orchestrator, "_run_stream_fragility", _fit
    )
    monkeypatch.setattr(
        server_orchestrator, "save_state", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        server_orchestrator, "event", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        server_orchestrator, "online_backup", lambda *_args: None
    )
    monkeypatch.setattr(
        server_orchestrator, "terminate_tree", _terminate
    )
    monkeypatch.setattr(
        server_orchestrator.time, "sleep", _pause_after_first_loop
    )

    with pytest.raises(InterruptedError, match="Pause requested"):
        server_orchestrator.run_streaming_spo_ida(
            server_orchestrator.default_state(),
            "batch_004",
            workers=6,
        )
    assert fragility_done == {"B-001"}
    assert starts == [("ida", ["B-002"])]


def test_orchestrator_replaces_evidenced_failure_before_identical_retry(
    monkeypatch,
) -> None:
    temporary, database = _working_database()
    test_root = Path(temporary.name)
    try:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            original_id = str(
                connection.execute(
                    """
                    SELECT building_id FROM building_catalog
                    WHERE queue_rank=182
                    """
                ).fetchone()[0]
            )
            connection.execute(
                """
                INSERT INTO pipeline_failures(
                    stage,building_id,pair_id,created_utc,error_type,
                    message,details_json,resolved
                ) VALUES('SPO',?,NULL,'2026-08-01T00:00:00+00:00',
                         'RuntimeError','analysis_guard_max_runtime','{}',0)
                """,
                (original_id,),
            )
            connection.commit()
            ids = ids_for_batch(connection, "batch_004")
        finally:
            connection.close()

        monkeypatch.setattr(server_orchestrator, "DATABASE", database)
        monkeypatch.setattr(
            server_orchestrator, "RESERVE_DATABASE", RESERVE_DATABASE
        )
        monkeypatch.setattr(
            server_orchestrator, "BACKUP", test_root / "backup.sqlite"
        )
        monkeypatch.setattr(
            server_orchestrator, "STATE", test_root / "state.json"
        )
        monkeypatch.setattr(
            server_orchestrator, "EVENTS", test_root / "events.jsonl"
        )
        monkeypatch.setattr(
            server_orchestrator,
            "PAUSE_REQUEST",
            test_root / "pause.request.json",
        )

        def _unexpected_process(*_args, **_kwargs):
            raise AssertionError(
                "Orchestrator repeated a known failed SPO before replacement"
            )

        monkeypatch.setattr(
            server_orchestrator.subprocess, "Popen", _unexpected_process
        )
        state = server_orchestrator.default_state()
        refresh = server_orchestrator.run_stage(
            state, "batch_004", "modal_spo", ids, workers=6
        )
        assert refresh is True
        assert "nearest replacement" in str(state["message"])
    finally:
        temporary.cleanup()


def test_failed_replacement_advances_same_slot_to_next_nearest_model() -> None:
    temporary, database = _working_database()
    try:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            original_id = str(
                connection.execute(
                    """
                    SELECT building_id FROM building_catalog
                    WHERE queue_rank=182
                    """
                ).fetchone()[0]
            )
            connection.execute(
                """
                INSERT INTO pipeline_failures(
                    stage,building_id,pair_id,created_utc,error_type,
                    message,details_json,resolved
                ) VALUES('SPO',?,NULL,'2026-08-01T00:00:00+00:00',
                         'RuntimeError','first failure','{}',0)
                """,
                (original_id,),
            )
            connection.commit()
        finally:
            connection.close()
        first = quarantine_and_replace_failures(
            database, RESERVE_DATABASE, "batch_004"
        )["replacements"][0]

        connection = sqlite3.connect(database)
        try:
            connection.execute(
                """
                INSERT INTO pipeline_failures(
                    stage,building_id,pair_id,created_utc,error_type,
                    message,details_json,resolved
                ) VALUES('SPO',?,NULL,'2026-08-01T00:01:00+00:00',
                         'RuntimeError','replacement failure','{}',0)
                """,
                (first["replacement_building_id"],),
            )
            connection.commit()
        finally:
            connection.close()
        second = quarantine_and_replace_failures(
            database, RESERVE_DATABASE, "batch_004"
        )["replacements"][0]

        assert second["slot_rank"] == 182
        assert second["generation"] == 2
        assert (
            second["failed_building_id"]
            == first["replacement_building_id"]
        )
        assert second["replacement_building_id"] not in {
            original_id,
            first["replacement_building_id"],
        }
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            slot = connection.execute(
                """
                SELECT original_building_id,current_building_id,
                       replacement_generation
                FROM batch_building_slots
                WHERE batch_id='batch_004' AND slot_rank=182
                """
            ).fetchone()
            assert str(slot["original_building_id"]) == original_id
            assert (
                str(slot["current_building_id"])
                == second["replacement_building_id"]
            )
            assert int(slot["replacement_generation"]) == 2
        finally:
            connection.close()
    finally:
        temporary.cleanup()


def test_main_import_accepts_only_audited_replacement_chain() -> None:
    temporary, database = _working_database()
    try:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        try:
            original_id = str(
                connection.execute(
                    """
                    SELECT building_id FROM building_catalog
                    WHERE queue_rank=182
                    """
                ).fetchone()[0]
            )
            connection.execute(
                """
                INSERT INTO pipeline_failures(
                    stage,building_id,pair_id,created_utc,error_type,
                    message,details_json,resolved
                ) VALUES('SPO',?,NULL,'2026-08-01T00:00:00+00:00',
                         'RuntimeError','failure','{}',0)
                """,
                (original_id,),
            )
            connection.commit()
        finally:
            connection.close()
        quarantine_and_replace_failures(
            database, RESERVE_DATABASE, "batch_004"
        )
        connection = sqlite3.connect(database)
        try:
            active_ids = [
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT building_id FROM building_catalog
                    WHERE selected=1 AND queue_rank BETWEEN 151 AND 200
                    """
                )
            ]
            placeholders = ",".join("?" for _ in active_ids)
            connection.execute(
                f"""
                DELETE FROM building_catalog
                WHERE building_id NOT IN ({placeholders})
                """,
                active_ids,
            )
            connection.commit()
        finally:
            connection.close()

        result_connection = sqlite3.connect(database)
        result_connection.row_factory = sqlite3.Row
        main_database = MAIN_ROOT / "data" / "5storey" / "poc.sqlite"
        main_connection = sqlite3.connect(main_database)
        main_connection.row_factory = sqlite3.Row
        try:
            buildings = [
                dict(row)
                for row in result_connection.execute(
                    """
                    SELECT building_id,queue_rank,model_hash
                    FROM building_catalog
                    WHERE selected=1 AND queue_rank BETWEEN 151 AND 200
                    ORDER BY queue_rank
                    """
                )
            ]
            history = [
                dict(row)
                for row in result_connection.execute(
                    """
                    SELECT * FROM spo_replacement_history
                    WHERE batch_id='batch_004'
                    ORDER BY slot_rank,generation
                    """
                )
            ]
            manifest = {
                "scientific_config_signature":
                    scientific_config_signature(
                        MAIN_ROOT / "config" / "poc.json"
                    ),
                "source_code_signature": source_code_signature(
                    MAIN_ROOT / "src" / "fragility_poc"
                ),
                "controller_model_sha256": sha256(
                    MAIN_ROOT
                    / "models"
                    / "active_ida_controller_v1.joblib"
                ),
                "ground_motion_signature": ground_motion_signature(
                    main_connection
                ),
                "buildings": buildings,
                "spo_replacements": history,
            }
            plans = verify_scientific_identity(
                MAIN_ROOT,
                main_connection,
                result_connection,
                manifest,
            )
            assert len(plans) == 1
            assert plans[0]["slot_rank"] == 182
            assert plans[0]["previous_building_id"] == original_id
            assert (
                plans[0]["replacement_building_id"]
                == history[-1]["replacement_building_id"]
            )
        finally:
            main_connection.close()
            result_connection.close()
    finally:
        temporary.cleanup()
