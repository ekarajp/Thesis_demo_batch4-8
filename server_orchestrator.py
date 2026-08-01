"""One-click, restartable orchestrator for server Batches 004-008."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from portable_common import (
    BATCHES,
    append_jsonl,
    atomic_json,
    connect,
    expected_batch_count,
    ids_for_batch,
    online_backup,
    process_is_running,
    progress_snapshot,
    read_json,
    sha256,
    utc_now,
)
from replacement_policy import (
    ensure_replacement_schema,
    initialize_batch_slots,
    quarantine_and_replace_failures,
)


ROOT = Path(__file__).absolute().parent
CONFIG = ROOT / "config" / "poc.json"
DATABASE = ROOT / "data" / "server_batches_004_008.sqlite"
RUNTIME = ROOT / "runtime"
STATE = RUNTIME / "server_state.json"
EVENTS = RUNTIME / "events.jsonl"
LOCK = RUNTIME / "orchestrator.lock.json"
PAUSE_REQUEST = RUNTIME / "pause.request.json"
SETTINGS = RUNTIME / "server_settings.json"
BACKUP = ROOT / "backups" / "server_latest.sqlite"
RESERVE_DATABASE = ROOT / "data" / "spo_replacement_reserve.sqlite"


def event(name: str, **details: Any) -> None:
    append_jsonl(
        EVENTS, {"timestamp_utc": utc_now(), "event": name, **details}
    )


def default_state() -> dict[str, Any]:
    return {
        "schema": "server-orchestrator-state-v1",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "status": "ready",
        "runner_pid": None,
        "current_batch": None,
        "current_stage": None,
        "current_child_pid": None,
        "active_child_pids": {},
        "stages": {},
        "message": "Ready to start Batch 004-008",
    }


def load_state() -> dict[str, Any]:
    return read_json(STATE, default_state())


def save_state(state: dict[str, Any], **updates: Any) -> None:
    state.update(updates)
    state["updated_utc"] = utc_now()
    state["progress_summary"] = {
        key: value
        for key, value in progress_snapshot(DATABASE).items()
        if key != "buildings"
    }
    atomic_json(STATE, state)


def acquire_lock() -> None:
    existing = read_json(LOCK, {})
    existing_pid = int(existing.get("pid") or 0)
    if process_is_running(existing_pid):
        raise RuntimeError(
            f"Another server orchestrator is already running (PID {existing_pid})"
        )
    atomic_json(
        LOCK,
        {
            "pid": os.getpid(),
            "created_utc": utc_now(),
            "root": str(ROOT),
        },
    )


def release_lock() -> None:
    existing = read_json(LOCK, {})
    if int(existing.get("pid") or 0) == os.getpid():
        LOCK.unlink(missing_ok=True)


def building_arguments(ids: list[str]) -> list[str]:
    result: list[str] = []
    for building_id in ids:
        result.extend(["--building-id", building_id])
    return result


def cli(*arguments: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "fragility_poc.cli",
        "--config",
        str(CONFIG),
        *arguments,
    ]


def terminate_tree(process: subprocess.Popen[Any]) -> dict[str, Any]:
    result = {"pid": process.pid, "graceful": False, "forced": False}
    if process.poll() is not None:
        result["return_code"] = process.returncode
        return result
    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            result["graceful"] = True
            process.wait(timeout=20)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            completed = subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
            result["forced"] = True
            result["taskkill_return_code"] = completed.returncode
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            result["graceful"] = True
            process.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            os.killpg(process.pid, signal.SIGKILL)
            result["forced"] = True
    try:
        result["return_code"] = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        result["return_code"] = process.poll()
    return result


def terminate_pid_tree(pid: int) -> dict[str, Any]:
    """Best-effort emergency cleanup when only a persisted child PID remains."""
    result: dict[str, Any] = {
        "pid": int(pid),
        "already_finished": not process_is_running(int(pid)),
        "forced": False,
    }
    if result["already_finished"]:
        return result
    if os.name == "nt":
        completed = subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        result.update(
            {
                "forced": True,
                "return_code": int(completed.returncode),
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
    else:
        try:
            os.killpg(int(pid), signal.SIGKILL)
            result["forced"] = True
            result["return_code"] = 0
        except OSError as exc:
            result["return_code"] = 1
            result["error"] = str(exc)
    return result


def batch_postcondition(
    batch_id: str, stage_name: str, ids: list[str]
) -> dict[str, Any]:
    marks = ",".join("?" for _ in ids)
    with connect(DATABASE, readonly=True) as connection:
        if stage_name == "modal_spo":
            count = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) FROM spo_features
                    WHERE valid=1 AND building_id IN ({marks})
                    """,
                    ids,
                ).fetchone()[0]
            )
            if count != len(ids):
                raise RuntimeError(
                    f"SPO valid for {count}/{len(ids)} buildings"
                )
            return {"valid_spo_buildings": count}

        if stage_name == "select_records":
            rows = list(
                connection.execute(
                    f"""
                    SELECT s.building_id,
                           SUM(CASE WHEN s.analysis_role='fragility_primary'
                                    AND g.valid=1 THEN 1 ELSE 0 END)
                             AS primary_count,
                           SUM(CASE WHEN s.analysis_role=
                                          'event_specific_sensitivity'
                                    AND s.scale_factor_policy='as_recorded_sf1'
                                    AND g.valid=1 THEN 1 ELSE 0 END)
                             AS pwsa_count,
                           COUNT(*) AS pair_count,
                           COUNT(DISTINCT COALESCE(
                               g.physical_pair_hash,g.pair_id
                           )) AS physical_count
                    FROM building_ground_motion_selection s
                    JOIN ground_motion_catalog g ON g.pair_id=s.pair_id
                    WHERE s.building_id IN ({marks})
                    GROUP BY s.building_id
                    """,
                    ids,
                )
            )
            config = json.loads(CONFIG.read_text(encoding="utf-8"))
            minimum = int(config["ida"]["minimum_pairs_for_fragility"])
            invalid = [
                str(row["building_id"])
                for row in rows
                if int(row["primary_count"] or 0) < minimum
                or int(row["pwsa_count"] or 0) != 1
                or int(row["pair_count"]) != int(row["physical_count"])
            ]
            if len(rows) != len(ids) or invalid:
                raise RuntimeError(
                    f"GM selection invalid; rows={len(rows)}/{len(ids)}, "
                    f"invalid={invalid[:10]}"
                )
            return {
                "selected_buildings": len(rows),
                "minimum_primary_pairs": min(
                    int(row["primary_count"]) for row in rows
                ),
                "exactly_one_pwsa_per_building": True,
            }

        if stage_name == "full_ida":
            primary_total = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) FROM building_ground_motion_selection
                    WHERE analysis_role='fragility_primary'
                      AND building_id IN ({marks})
                    """,
                    ids,
                ).fetchone()[0]
            )
            primary_complete = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) FROM (
                      SELECT c.building_id,c.pair_id
                      FROM ida_capacities c
                      JOIN building_ground_motion_selection s
                        ON s.building_id=c.building_id
                       AND s.pair_id=c.pair_id
                      WHERE c.building_id IN ({marks})
                        AND s.analysis_role='fragility_primary'
                        AND c.censored=0 AND c.censoring='none'
                      GROUP BY c.building_id,c.pair_id
                      HAVING COUNT(DISTINCT c.limit_state)=3
                    )
                    """,
                    ids,
                ).fetchone()[0]
            )
            pwsa_complete = int(
                connection.execute(
                    f"""
                    SELECT COUNT(DISTINCT s.building_id)
                    FROM building_ground_motion_selection s
                    JOIN ida_runs r
                      ON r.building_id=s.building_id
                     AND r.pair_id=s.pair_id
                    WHERE s.building_id IN ({marks})
                      AND s.analysis_role='event_specific_sensitivity'
                      AND s.scale_factor_policy='as_recorded_sf1'
                      AND ABS(r.scale_factor-1.0)<=1e-10
                      AND r.status IN ('success','dynamic_instability')
                    """,
                    ids,
                ).fetchone()[0]
            )
            critical = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) FROM ida_curve_diagnostics
                    WHERE building_id IN ({marks})
                      AND review_status='CRITICAL_QC_FAILURE'
                    """,
                    ids,
                ).fetchone()[0]
            )
            if (
                primary_total <= 0
                or primary_complete != primary_total
                or pwsa_complete != len(ids)
                or critical
            ):
                raise RuntimeError(
                    "IDA postcondition failed: "
                    f"primary={primary_complete}/{primary_total}, "
                    f"PWSA={pwsa_complete}/{len(ids)}, critical={critical}"
                )
            return {
                "complete_primary_curves": primary_complete,
                "complete_pwsa_buildings": pwsa_complete,
                "critical_qc_failures": 0,
            }

        if stage_name == "fragility":
            rows = list(
                connection.execute(
                    f"""
                    SELECT building_id, theta_io_g,theta_ls_g,theta_cp_g,
                           beta_io,beta_ls,beta_cp,
                           n_censored_io,n_censored_ls,n_censored_cp
                    FROM fragility_targets
                    WHERE valid=1 AND building_id IN ({marks})
                    """,
                    ids,
                )
            )
            invalid = [
                str(row["building_id"])
                for row in rows
                if not (
                    0 < float(row["theta_io_g"])
                    < float(row["theta_ls_g"])
                    < float(row["theta_cp_g"])
                    and float(row["beta_io"]) > 0
                    and float(row["beta_ls"]) > 0
                    and float(row["beta_cp"]) > 0
                    and int(row["n_censored_io"]) == 0
                    and int(row["n_censored_ls"]) == 0
                    and int(row["n_censored_cp"]) == 0
                )
            ]
            if len(rows) != len(ids) or invalid:
                raise RuntimeError(
                    f"Fragility valid for {len(rows)}/{len(ids)}; "
                    f"invalid={invalid[:10]}"
                )
            return {"valid_fragility_buildings": len(rows)}
    if stage_name == "export":
        path = (
            ROOT
            / "exports"
            / "READY_TO_COPY"
            / (
                f"{batch_id.title().replace('_', '_')}_Result_"
                f"Ranks_{BATCHES[batch_id][0]}_{BATCHES[batch_id][1]}.zip"
            )
        )
        # Exporter writes a simpler lower-case alias in its JSON; validate
        # through the latest export manifest instead of relying on casing.
        manifest = read_json(
            ROOT / "outputs" / batch_id / "result_manifest.json", {}
        )
        actual = Path(str(manifest.get("zip_path", "")))
        if not actual.is_file() or actual.stat().st_size <= 0:
            raise RuntimeError(f"Result ZIP is missing for {batch_id}: {actual}")
        return {"zip_path": str(actual), "zip_sha256": sha256(actual)}
    raise ValueError(f"Unknown stage: {stage_name}")


def stage_command(
    batch_id: str, stage_name: str, ids: list[str], workers: int
) -> list[str]:
    ids_args = building_arguments(ids)
    if stage_name == "modal_spo":
        return cli("run-modal-spo", *ids_args)
    if stage_name == "select_records":
        return cli("select-records", *ids_args)
    if stage_name == "full_ida":
        return cli("run-ida", *ids_args, "--workers", str(workers))
    if stage_name == "fragility":
        return cli(
            "fit-fragility",
            *ids_args,
            "--bootstrap-count",
            "2000",
        )
    if stage_name == "export":
        return [
            sys.executable,
            str(ROOT / "export_results.py"),
            "--root",
            str(ROOT),
            "--batch",
            batch_id,
        ]
    raise ValueError(stage_name)


def _ids_passing_postcondition(
    batch_id: str,
    stage_name: str,
    ids: list[str],
) -> set[str]:
    """Return IDs with a complete persisted result for one stage."""
    if not ids:
        return set()
    if stage_name == "modal_spo":
        marks = ",".join("?" for _ in ids)
        with connect(DATABASE, readonly=True) as connection:
            return {
                str(row["building_id"])
                for row in connection.execute(
                    f"""
                    SELECT building_id FROM spo_features
                    WHERE valid=1 AND building_id IN ({marks})
                    """,
                    ids,
                )
            }
    marks = ",".join("?" for _ in ids)
    if stage_name == "select_records":
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        minimum = int(config["ida"]["minimum_pairs_for_fragility"])
        with connect(DATABASE, readonly=True) as connection:
            completed = set()
            for row in connection.execute(
                f"""
                SELECT s.building_id,
                       SUM(CASE WHEN s.analysis_role='fragility_primary'
                                AND g.valid=1 THEN 1 ELSE 0 END)
                         AS primary_count,
                       SUM(CASE WHEN s.analysis_role=
                                      'event_specific_sensitivity'
                                AND s.scale_factor_policy='as_recorded_sf1'
                                AND g.valid=1 THEN 1 ELSE 0 END)
                         AS pwsa_count,
                       COUNT(*) AS pair_count,
                       COUNT(DISTINCT COALESCE(
                           g.physical_pair_hash,g.pair_id
                       )) AS physical_count
                FROM building_ground_motion_selection s
                JOIN ground_motion_catalog g ON g.pair_id=s.pair_id
                WHERE s.building_id IN ({marks})
                GROUP BY s.building_id
                """,
                ids,
            ):
                if (
                    int(row["primary_count"] or 0) >= minimum
                    and int(row["pwsa_count"] or 0) == 1
                    and int(row["pair_count"])
                    == int(row["physical_count"])
                ):
                    completed.add(str(row["building_id"]))
            return completed
    if stage_name == "full_ida":
        with connect(DATABASE, readonly=True) as connection:
            primary_totals = {
                str(row["building_id"]): int(row["curve_count"])
                for row in connection.execute(
                    f"""
                    SELECT building_id,COUNT(*) AS curve_count
                    FROM building_ground_motion_selection
                    WHERE analysis_role='fragility_primary'
                      AND building_id IN ({marks})
                    GROUP BY building_id
                    """,
                    ids,
                )
            }
            primary_complete = {
                str(row["building_id"]): int(row["curve_count"])
                for row in connection.execute(
                    f"""
                    SELECT x.building_id,COUNT(*) AS curve_count
                    FROM (
                      SELECT c.building_id,c.pair_id
                      FROM ida_capacities c
                      JOIN building_ground_motion_selection s
                        ON s.building_id=c.building_id
                       AND s.pair_id=c.pair_id
                      WHERE c.building_id IN ({marks})
                        AND s.analysis_role='fragility_primary'
                        AND c.censored=0 AND c.censoring='none'
                      GROUP BY c.building_id,c.pair_id
                      HAVING COUNT(DISTINCT c.limit_state)=3
                    ) x GROUP BY x.building_id
                    """,
                    ids,
                )
            }
            pwsa_complete = {
                str(row["building_id"])
                for row in connection.execute(
                    f"""
                    SELECT DISTINCT s.building_id
                    FROM building_ground_motion_selection s
                    JOIN ida_runs r
                      ON r.building_id=s.building_id
                     AND r.pair_id=s.pair_id
                    WHERE s.building_id IN ({marks})
                      AND s.analysis_role='event_specific_sensitivity'
                      AND s.scale_factor_policy='as_recorded_sf1'
                      AND ABS(r.scale_factor-1.0)<=1e-10
                      AND r.status IN ('success','dynamic_instability')
                    """,
                    ids,
                )
            }
            critical = {
                str(row["building_id"])
                for row in connection.execute(
                    f"""
                    SELECT DISTINCT building_id
                    FROM ida_curve_diagnostics
                    WHERE building_id IN ({marks})
                      AND review_status='CRITICAL_QC_FAILURE'
                    """,
                    ids,
                )
            }
        return {
            building_id
            for building_id in ids
            if primary_totals.get(building_id, 0) > 0
            and primary_complete.get(building_id, 0)
            == primary_totals[building_id]
            and building_id in pwsa_complete
            and building_id not in critical
        }
    if stage_name == "fragility":
        with connect(DATABASE, readonly=True) as connection:
            return {
                str(row["building_id"])
                for row in connection.execute(
                    f"""
                    SELECT building_id
                    FROM fragility_targets
                    WHERE valid=1
                      AND theta_io_g>0
                      AND theta_io_g<theta_ls_g
                      AND theta_ls_g<theta_cp_g
                      AND beta_io>0 AND beta_ls>0 AND beta_cp>0
                      AND n_censored_io=0
                      AND n_censored_ls=0
                      AND n_censored_cp=0
                      AND building_id IN ({marks})
                    """,
                    ids,
                )
            }
    completed: set[str] = set()
    for building_id in ids:
        try:
            batch_postcondition(
                batch_id, stage_name, [building_id]
            )
        except Exception:
            continue
        completed.add(building_id)
    return completed


def _stream_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _start_stream_child(
    *,
    batch_id: str,
    role: str,
    ids: list[str],
    workers: int,
    wave: int,
) -> dict[str, Any]:
    if role == "spo":
        command = stage_command(
            batch_id, "modal_spo", ids, workers
        )
    elif role == "ida":
        command = stage_command(
            batch_id, "full_ida", ids, workers
        )
    else:
        raise ValueError(role)
    log_root = ROOT / "logs" / batch_id
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"streaming_{role}.stdout.log"
    stderr_path = log_root / f"streaming_{role}.stderr.log"
    stdout = stdout_path.open("a", encoding="utf-8", newline="\n")
    stderr = stderr_path.open("a", encoding="utf-8", newline="\n")
    header = (
        f"\n--- streaming {role} wave {wave} started {utc_now()} "
        f"for {len(ids)} building(s) ---\n"
    )
    stdout.write(header)
    stdout.flush()
    os.fsync(stdout.fileno())
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=stdout,
        stderr=stderr,
        env=_stream_environment(),
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        ),
        start_new_session=os.name != "nt",
    )
    # Popen duplicates the handles for the child. The orchestrator does not
    # need to hold them open while the independent SPO/IDA wave is running.
    stdout.close()
    stderr.close()
    return {
        "role": role,
        "wave": wave,
        "ids": list(ids),
        "command": command,
        "process": process,
        "pid": int(process.pid),
        "started_utc": utc_now(),
        "started_monotonic": time.perf_counter(),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _run_stream_selection(
    batch_id: str,
    ids: list[str],
    wave: int,
) -> dict[str, Any]:
    """Select records for newly valid SPOs while both producers keep running."""
    command = stage_command(batch_id, "select_records", ids, workers=1)
    log_root = ROOT / "logs" / batch_id
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / "streaming_select_records.stdout.log"
    stderr_path = log_root / "streaming_select_records.stderr.log"
    started = time.perf_counter()
    with stdout_path.open("a", encoding="utf-8", newline="\n") as stdout, (
        stderr_path.open("a", encoding="utf-8", newline="\n")
    ) as stderr:
        stdout.write(
            f"\n--- selection wave {wave} started {utc_now()} "
            f"for {len(ids)} building(s) ---\n"
        )
        stdout.flush()
        os.fsync(stdout.fileno())
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            env=_stream_environment(),
            check=False,
        )
    return {
        "wave": wave,
        "ids": list(ids),
        "command": command,
        "return_code": int(completed.returncode),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _run_stream_fragility(
    batch_id: str,
    ids: list[str],
    wave: int,
) -> dict[str, Any]:
    """Fit newly IDA-complete buildings while other analyses keep running."""
    command = stage_command(batch_id, "fragility", ids, workers=1)
    log_root = ROOT / "logs" / batch_id
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / "streaming_fragility.stdout.log"
    stderr_path = log_root / "streaming_fragility.stderr.log"
    started = time.perf_counter()
    with stdout_path.open("a", encoding="utf-8", newline="\n") as stdout, (
        stderr_path.open("a", encoding="utf-8", newline="\n")
    ) as stderr:
        stdout.write(
            f"\n--- fragility wave {wave} started {utc_now()} "
            f"for {len(ids)} building(s) ---\n"
        )
        stdout.flush()
        os.fsync(stdout.fileno())
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            env=_stream_environment(),
            check=False,
        )
    return {
        "wave": wave,
        "ids": list(ids),
        "command": command,
        "return_code": int(completed.returncode),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _record_completed_legacy_stage(
    state: dict[str, Any],
    batch_id: str,
    stage_name: str,
    ids: list[str],
) -> None:
    stage_key = f"{batch_id}:{stage_name}"
    postcondition = batch_postcondition(batch_id, stage_name, ids)
    stage = state["stages"].setdefault(
        stage_key, {"status": "pending", "attempts": []}
    )
    stage["status"] = "completed"
    stage["postcondition"] = postcondition


def streaming_worksets(
    ids: list[str],
    valid_spo: set[str],
    selected: set[str],
    ida_complete: set[str],
    fragility_complete: set[str] | None = None,
) -> dict[str, list[str]]:
    """Compute persisted queues without any batch-wide stage gate."""
    ordered = list(ids)
    completed_fragility = fragility_complete or set()
    return {
        "pending_spo": [
            building_id
            for building_id in ordered
            if building_id not in valid_spo
        ],
        "pending_selection": [
            building_id
            for building_id in ordered
            if building_id in valid_spo and building_id not in selected
        ],
        "pending_ida": [
            building_id
            for building_id in ordered
            if building_id in selected and building_id not in ida_complete
        ],
        "pending_fragility": [
            building_id
            for building_id in ordered
            if building_id in ida_complete
            and building_id not in completed_fragility
        ],
    }


def run_streaming_spo_ida(
    state: dict[str, Any],
    batch_id: str,
    workers: int,
) -> None:
    """Stream each building through SPO, selection, Full IDA, and fragility.

    One serial SPO producer may overlap one six-worker IDA consumer. New valid
    SPOs accumulate while the current IDA wave runs, then enter the next wave.
    Every IDA-complete building is fitted immediately without waiting for the
    remaining IDA curves. No building crosses a stage unless its own persisted
    postcondition passes.
    """
    stream_key = f"{batch_id}:streaming_spo_to_fragility"
    stream = state["stages"].setdefault(
        stream_key,
        {
            "status": "pending",
            "spo_waves": [],
            "selection_waves": [],
            "ida_waves": [],
            "fragility_waves": [],
        },
    )
    with connect(DATABASE, readonly=True) as connection:
        initial_ids = ids_for_batch(connection, batch_id)
    if len(initial_ids) != expected_batch_count(batch_id):
        raise RuntimeError(
            f"{batch_id} has {len(initial_ids)} active IDs, expected "
            f"{expected_batch_count(batch_id)}"
        )
    if (
        len(
            _ids_passing_postcondition(
                batch_id, "fragility", initial_ids
            )
        )
        == len(initial_ids)
    ):
        for stage_name in (
            "modal_spo",
            "select_records",
            "full_ida",
            "fragility",
        ):
            _record_completed_legacy_stage(
                state, batch_id, stage_name, initial_ids
            )
        stream["status"] = "completed"
        save_state(state)
        return

    stream["status"] = "running"
    stream["started_utc"] = stream.get("started_utc") or utc_now()
    active: dict[str, dict[str, Any] | None] = {
        "spo": None,
        "ida": None,
    }
    selection_wave = int(stream.get("selection_wave_count", 0))
    fragility_wave = int(stream.get("fragility_wave_count", 0))
    last_backup = time.perf_counter()

    def update_running_state(message: str) -> None:
        pids = {
            role: int(child["pid"])
            for role, child in active.items()
            if child is not None
            and child["process"].poll() is None
        }
        save_state(
            state,
            status="running",
            current_batch=batch_id,
            current_stage="streaming_spo_to_fragility",
            current_child_pid=(
                next(iter(pids.values())) if pids else None
            ),
            active_child_pids=pids,
            message=message,
        )

    update_running_state(
        f"Streaming {batch_id}: SPO → IDA → Fragility per building"
    )
    event("streaming_stage_started", stage=stream_key)

    while True:
        if PAUSE_REQUEST.exists():
            terminations = {}
            for role, child in active.items():
                if child is not None:
                    terminations[role] = terminate_tree(
                        child["process"]
                    )
            online_backup(DATABASE, BACKUP)
            PAUSE_REQUEST.unlink(missing_ok=True)
            stream["status"] = "paused"
            stream["pause_terminations"] = terminations
            save_state(
                state,
                status="paused",
                current_child_pid=None,
                active_child_pids={},
                message=(
                    f"Paused streaming {batch_id}; completed SPO, NLTHA, "
                    "IDA-capacity, and fragility results will be reused."
                ),
            )
            event(
                "streaming_stage_paused",
                stage=stream_key,
                terminations=terminations,
            )
            raise InterruptedError("Pause requested")

        # Complete and classify an SPO producer wave before assigning any
        # failed slot to a replacement model.
        spo_child = active["spo"]
        if (
            spo_child is not None
            and spo_child["process"].poll() is not None
        ):
            return_code = int(spo_child["process"].returncode)
            finished = {
                key: value
                for key, value in spo_child.items()
                if key not in {"process", "started_monotonic"}
            }
            finished["return_code"] = return_code
            finished["finished_utc"] = utc_now()
            finished["elapsed_wall_seconds"] = (
                time.perf_counter()
                - float(spo_child["started_monotonic"])
            )
            stream["spo_waves"].append(finished)
            active["spo"] = None
            if return_code != 0:
                recovery = quarantine_and_replace_failures(
                    DATABASE, RESERVE_DATABASE, batch_id
                )
                if int(recovery["replacement_count"]) == 0:
                    stream["status"] = "needs_attention"
                    update_running_state(
                        f"{batch_id} SPO wave failed without an evidenced "
                        "replaceable building; review streaming_spo.stderr.log"
                    )
                    raise RuntimeError(
                        f"{batch_id} streaming SPO failed with code "
                        f"{return_code} and no quarantine candidate"
                    )
                stream.setdefault("replacement_recoveries", []).append(
                    recovery
                )
                event(
                    "streaming_spo_quarantined_and_replaced",
                    stage=stream_key,
                    recovery=recovery,
                )

        # Finish one IDA consumer wave. A nonzero result remains fail-closed;
        # every completed curve/checkpoint from the wave is still retained.
        ida_child = active["ida"]
        if (
            ida_child is not None
            and ida_child["process"].poll() is not None
        ):
            return_code = int(ida_child["process"].returncode)
            finished = {
                key: value
                for key, value in ida_child.items()
                if key not in {"process", "started_monotonic"}
            }
            finished["return_code"] = return_code
            finished["finished_utc"] = utc_now()
            finished["elapsed_wall_seconds"] = (
                time.perf_counter()
                - float(ida_child["started_monotonic"])
            )
            stream["ida_waves"].append(finished)
            active["ida"] = None
            completed_wave_ids = _ids_passing_postcondition(
                batch_id, "full_ida", list(ida_child["ids"])
            )
            if (
                return_code != 0
                or len(completed_wave_ids) != len(ida_child["ids"])
            ):
                incomplete = sorted(
                    set(ida_child["ids"]) - completed_wave_ids
                )
                stream["status"] = "needs_attention"
                update_running_state(
                    f"{batch_id} IDA wave needs attention; incomplete "
                    f"buildings={incomplete[:5]}"
                )
                raise RuntimeError(
                    f"{batch_id} streaming IDA wave failed; incomplete "
                    f"buildings={incomplete}"
                )
            event(
                "streaming_ida_wave_completed",
                stage=stream_key,
                building_count=len(completed_wave_ids),
            )

        # Refresh active slots because an invalid SPO may just have been
        # replaced. Previously valid IDs remain untouched and reusable.
        with connect(DATABASE, readonly=True) as connection:
            ids = ids_for_batch(connection, batch_id)
        if len(ids) != expected_batch_count(batch_id):
            raise RuntimeError(
                f"{batch_id} active population changed to {len(ids)} IDs"
            )
        valid_spo = _ids_passing_postcondition(
            batch_id, "modal_spo", ids
        )
        selected = _ids_passing_postcondition(
            batch_id, "select_records", sorted(valid_spo)
        )
        ida_complete = _ids_passing_postcondition(
            batch_id, "full_ida", sorted(selected)
        )
        fragility_complete = _ids_passing_postcondition(
            batch_id, "fragility", sorted(ida_complete)
        )

        # Start/continue the SPO producer independently of the IDA consumer.
        if active["spo"] is None:
            recovery = quarantine_and_replace_failures(
                DATABASE, RESERVE_DATABASE, batch_id
            )
            if int(recovery["replacement_count"]) > 0:
                stream.setdefault("replacement_recoveries", []).append(
                    recovery
                )
                with connect(DATABASE, readonly=True) as connection:
                    ids = ids_for_batch(connection, batch_id)
                valid_spo = _ids_passing_postcondition(
                    batch_id, "modal_spo", ids
                )
            worksets = streaming_worksets(
                ids,
                valid_spo,
                selected,
                ida_complete,
                fragility_complete,
            )
            pending_spo = worksets["pending_spo"]
            if pending_spo:
                wave = len(stream["spo_waves"]) + 1
                active["spo"] = _start_stream_child(
                    batch_id=batch_id,
                    role="spo",
                    ids=pending_spo,
                    workers=workers,
                    wave=wave,
                )
                event(
                    "streaming_spo_wave_started",
                    stage=stream_key,
                    wave=wave,
                    building_count=len(pending_spo),
                    pid=active["spo"]["pid"],
                )
                update_running_state(
                    f"{batch_id}: SPO producer started for "
                    f"{len(pending_spo)} pending building(s)"
                )

        # A newly valid SPO gets its own GM selection immediately, even while
        # the current SPO and IDA processes continue on other buildings.
        worksets = streaming_worksets(
            ids,
            valid_spo,
            selected,
            ida_complete,
            fragility_complete,
        )
        pending_selection = worksets["pending_selection"]
        if pending_selection:
            selection_wave += 1
            selection_result = _run_stream_selection(
                batch_id, pending_selection, selection_wave
            )
            stream["selection_waves"].append(selection_result)
            stream["selection_wave_count"] = selection_wave
            if int(selection_result["return_code"]) != 0:
                stream["status"] = "needs_attention"
                update_running_state(
                    f"{batch_id} record selection failed; review "
                    "streaming_select_records.stderr.log"
                )
                raise RuntimeError(
                    f"{batch_id} streaming record selection failed"
                )
            selected = _ids_passing_postcondition(
                batch_id, "select_records", sorted(valid_spo)
            )

        # One work-conserving IDA process uses the configured worker pool.
        # SPO keeps one independent core; newly selected buildings queue for
        # the next IDA wave instead of waiting for all SPOs in the batch.
        if active["ida"] is None:
            ida_complete = _ids_passing_postcondition(
                batch_id, "full_ida", sorted(selected)
            )
            worksets = streaming_worksets(
                ids,
                valid_spo,
                selected,
                ida_complete,
                fragility_complete,
            )
            pending_ida = worksets["pending_ida"]
            if pending_ida:
                wave = len(stream["ida_waves"]) + 1
                active["ida"] = _start_stream_child(
                    batch_id=batch_id,
                    role="ida",
                    ids=pending_ida,
                    workers=workers,
                    wave=wave,
                )
                event(
                    "streaming_ida_wave_started",
                    stage=stream_key,
                    wave=wave,
                    building_count=len(pending_ida),
                    pid=active["ida"]["pid"],
                )
                update_running_state(
                    f"{batch_id}: Full IDA consumer started immediately for "
                    f"{len(pending_ida)} SPO-ready building(s)"
                )

        # A building enters fragility fitting as soon as all of its own
        # primary IDA capacities and PWSA sensitivity result are complete.
        # The IDA worker pool may continue processing other buildings.
        ida_complete = _ids_passing_postcondition(
            batch_id, "full_ida", ids
        )
        fragility_complete = _ids_passing_postcondition(
            batch_id, "fragility", sorted(ida_complete)
        )
        worksets = streaming_worksets(
            ids,
            valid_spo,
            selected,
            ida_complete,
            fragility_complete,
        )
        pending_fragility = worksets["pending_fragility"]
        if pending_fragility:
            fragility_wave += 1
            fragility_result = _run_stream_fragility(
                batch_id, pending_fragility, fragility_wave
            )
            stream["fragility_waves"].append(fragility_result)
            stream["fragility_wave_count"] = fragility_wave
            fragility_complete = _ids_passing_postcondition(
                batch_id, "fragility", sorted(ida_complete)
            )
            missing_fragility = sorted(
                set(pending_fragility) - fragility_complete
            )
            if (
                int(fragility_result["return_code"]) != 0
                or missing_fragility
            ):
                stream["status"] = "needs_attention"
                update_running_state(
                    f"{batch_id} fragility wave needs attention; "
                    f"incomplete buildings={missing_fragility[:5]}"
                )
                raise RuntimeError(
                    f"{batch_id} streaming fragility failed; "
                    f"incomplete buildings={missing_fragility}"
                )
            event(
                "streaming_fragility_wave_completed",
                stage=stream_key,
                wave=fragility_wave,
                building_count=len(pending_fragility),
            )

        # Batch completion is based solely on persisted fragility
        # postconditions, not on a stage-status flag.
        if (
            len(fragility_complete) == len(ids)
            and active["spo"] is None
            and active["ida"] is None
        ):
            for stage_name in (
                "modal_spo",
                "select_records",
                "full_ida",
                "fragility",
            ):
                _record_completed_legacy_stage(
                    state, batch_id, stage_name, ids
                )
            stream["status"] = "completed"
            stream["finished_utc"] = utc_now()
            online_backup(DATABASE, BACKUP)
            save_state(
                state,
                status="running",
                current_child_pid=None,
                active_child_pids={},
                message=f"{batch_id} streaming SPO→IDA→Fragility completed",
            )
            event(
                "streaming_stage_completed",
                stage=stream_key,
                building_count=len(ids),
            )
            return

        if time.perf_counter() - last_backup >= 600.0:
            online_backup(DATABASE, BACKUP)
            last_backup = time.perf_counter()
            event(
                "streaming_rolling_backup_completed",
                stage=stream_key,
            )
        update_running_state(
            f"{batch_id}: SPO {len(valid_spo)}/{len(ids)}, "
            f"selected {len(selected)}/{len(ids)}, "
            f"Full IDA {len(ida_complete)}/{len(ids)}, "
            f"Fragility {len(fragility_complete)}/{len(ids)}"
        )
        time.sleep(2.0)


def run_stage(
    state: dict[str, Any],
    batch_id: str,
    stage_name: str,
    ids: list[str],
    workers: int,
) -> bool:
    """Run one stage.

    Return ``True`` only when an invalid SPO was quarantined and the caller
    must refresh the batch IDs before retrying this same stage.
    """
    stage_key = f"{batch_id}:{stage_name}"
    if PAUSE_REQUEST.exists():
        PAUSE_REQUEST.unlink(missing_ok=True)
        online_backup(DATABASE, BACKUP)
        save_state(
            state,
            status="paused",
            current_batch=batch_id,
            current_stage=stage_name,
            current_child_pid=None,
            message=(
                f"Paused safely before {batch_id} / {stage_name}; "
                "press Resume when ready."
            ),
        )
        event("paused_between_stages", next_stage=stage_key)
        raise InterruptedError("Pause requested between stages")

    # A previous process may have completed with an invalid SPO before this
    # orchestrator version was installed. Quarantine that evidenced failure
    # before launching another identical, expensive retry.
    if stage_name == "modal_spo":
        recovery = quarantine_and_replace_failures(
            DATABASE, RESERVE_DATABASE, batch_id
        )
        if int(recovery["replacement_count"]) > 0:
            online_backup(DATABASE, BACKUP)
            save_state(
                state,
                status="running",
                current_batch=batch_id,
                current_stage=stage_name,
                current_child_pid=None,
                message=(
                    f"Quarantined {recovery['replacement_count']} invalid "
                    f"SPO model(s) in {batch_id}; retrying nearest "
                    "replacement model(s)."
                ),
            )
            event(
                "spo_failures_quarantined_and_replaced",
                stage=stage_key,
                recovery=recovery,
            )
            return True
    stage = state["stages"].setdefault(
        stage_key, {"status": "pending", "attempts": []}
    )
    if stage.get("status") == "completed":
        try:
            postcondition = batch_postcondition(batch_id, stage_name, ids)
        except Exception as exc:
            stage["status"] = "stale"
            stage["stale_reason"] = str(exc)
            event("completed_stage_stale", stage=stage_key, error=str(exc))
        else:
            stage["postcondition"] = postcondition
            save_state(state)
            return False

    command = stage_command(batch_id, stage_name, ids, workers)
    attempt = {
        "attempt": len(stage["attempts"]) + 1,
        "started_utc": utc_now(),
        "command": command,
        "status": "running",
    }
    stage["attempts"].append(attempt)
    stage["status"] = "running"
    save_state(
        state,
        status="running",
        current_batch=batch_id,
        current_stage=stage_name,
        message=f"Running {batch_id} / {stage_name}",
    )
    event("stage_started", stage=stage_key, attempt=attempt["attempt"])
    log_root = ROOT / "logs" / batch_id
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{stage_name}.stdout.log"
    stderr_path = log_root / f"{stage_name}.stderr.log"
    started = time.perf_counter()
    last_snapshot = 0.0
    last_backup = 0.0
    with stdout_path.open("a", encoding="utf-8", newline="\n") as stdout, (
        stderr_path.open("a", encoding="utf-8", newline="\n")
    ) as stderr:
        environment = os.environ.copy()
        source_path = str(ROOT / "src")
        environment["PYTHONPATH"] = source_path + (
            os.pathsep + environment["PYTHONPATH"]
            if environment.get("PYTHONPATH")
            else ""
        )
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
            start_new_session=os.name != "nt",
        )
        state["current_child_pid"] = process.pid
        save_state(state)
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            attempt["elapsed_wall_seconds"] = round(elapsed, 3)
            if PAUSE_REQUEST.exists():
                termination = terminate_tree(process)
                elapsed = time.perf_counter() - started
                attempt.update(
                    {
                        "status": "paused",
                        "finished_utc": utc_now(),
                        "elapsed_wall_seconds": elapsed,
                        "termination": termination,
                    }
                )
                stage["status"] = "paused"
                online_backup(DATABASE, BACKUP)
                PAUSE_REQUEST.unlink(missing_ok=True)
                save_state(
                    state,
                    status="paused",
                    current_child_pid=None,
                    message=(
                        f"Paused safely at {batch_id} / {stage_name}; "
                        "saved checkpoints will be reused."
                    ),
                )
                event("stage_paused", stage=stage_key, termination=termination)
                raise InterruptedError("Pause requested")
            if elapsed - last_snapshot >= 10.0:
                save_state(state)
                last_snapshot = elapsed
            if elapsed - last_backup >= 600.0:
                online_backup(DATABASE, BACKUP)
                last_backup = elapsed
                event(
                    "rolling_database_backup_completed",
                    stage=stage_key,
                    elapsed_wall_seconds=round(elapsed, 3),
                )
            time.sleep(2.0)
        return_code = int(process.returncode)

    elapsed = time.perf_counter() - started
    attempt.update(
        {
            "finished_utc": utc_now(),
            "elapsed_wall_seconds": elapsed,
            "return_code": return_code,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
    )
    state["current_child_pid"] = None
    if return_code != 0:
        if stage_name == "modal_spo":
            recovery = quarantine_and_replace_failures(
                DATABASE, RESERVE_DATABASE, batch_id
            )
            if int(recovery["replacement_count"]) > 0:
                attempt["status"] = "quarantined_and_replaced"
                attempt["replacement_recovery"] = recovery
                stage["status"] = "pending_replacement_spo"
                online_backup(DATABASE, BACKUP)
                save_state(
                    state,
                    status="running",
                    current_child_pid=None,
                    message=(
                        f"{stage_key}: retained invalid SPO evidence and "
                        f"assigned {recovery['replacement_count']} nearest "
                        "replacement model(s)."
                    ),
                )
                event(
                    "spo_failures_quarantined_and_replaced",
                    stage=stage_key,
                    return_code=return_code,
                    recovery=recovery,
                )
                return True
        attempt["status"] = "failed"
        stage["status"] = "failed"
        save_state(
            state,
            status="needs_attention",
            message=(
                f"{stage_key} stopped with code {return_code}. "
                "Press Resume after reviewing the log."
            ),
        )
        event("stage_failed", stage=stage_key, return_code=return_code)
        raise RuntimeError(
            f"{stage_key} failed with code {return_code}; see {stderr_path}"
        )
    postcondition = batch_postcondition(batch_id, stage_name, ids)
    attempt["status"] = "completed"
    attempt["postcondition"] = postcondition
    stage["status"] = "completed"
    stage["postcondition"] = postcondition
    online_backup(DATABASE, BACKUP)
    save_state(state)
    event("stage_completed", stage=stage_key, postcondition=postcondition)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if not args.start:
        parser.error("--start is required")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    acquire_lock()
    state = load_state()
    try:
        settings = read_json(SETTINGS, {})
        requested = args.workers or int(settings.get("workers") or 6)
        maximum = max(1, (os.cpu_count() or 2) - 1)
        workers = max(1, min(int(requested), maximum))
        atomic_json(
            SETTINGS,
            {
                "workers": workers,
                "last_started_utc": utc_now(),
                "cpu_count": os.cpu_count(),
            },
        )
        PAUSE_REQUEST.unlink(missing_ok=True)
        save_state(
            state,
            status="running",
            runner_pid=os.getpid(),
            workers=workers,
            message="Starting or resuming Batch 004-008",
        )
        event("orchestrator_started", pid=os.getpid(), workers=workers)

        with connect(DATABASE) as connection:
            ensure_replacement_schema(connection)
            initialize_batch_slots(connection)

        for batch_id in BATCHES:
            run_streaming_spo_ida(state, batch_id, workers)
            for stage_name in ("export",):
                while True:
                    with connect(DATABASE, readonly=True) as connection:
                        ids = ids_for_batch(connection, batch_id)
                    if len(ids) != expected_batch_count(batch_id):
                        raise RuntimeError(
                            f"{batch_id} has {len(ids)} active IDs, expected "
                            f"{expected_batch_count(batch_id)}"
                        )
                    refresh_ids = run_stage(
                        state, batch_id, stage_name, ids, workers
                    )
                    if not refresh_ids:
                        break

        combined = subprocess.run(
            [
                sys.executable,
                str(ROOT / "export_results.py"),
                "--root",
                str(ROOT),
                "--all",
            ],
            cwd=ROOT,
            check=False,
        )
        if combined.returncode != 0:
            raise RuntimeError("Combined result export failed")
        save_state(
            state,
            status="completed",
            current_batch=None,
            current_stage=None,
            current_child_pid=None,
            active_child_pids={},
            runner_pid=None,
            completed_utc=utc_now(),
            message=(
                "All Batch 004-008 scientific results and copy-ready ZIPs "
                "are complete."
            ),
        )
        event("orchestrator_completed")
        return 0
    except InterruptedError:
        return 2
    except Exception as exc:
        emergency_cleanup = {
            str(role): terminate_pid_tree(int(pid))
            for role, pid in dict(
                state.get("active_child_pids") or {}
            ).items()
            if int(pid) > 0
        }
        save_state(
            state,
            status="needs_attention",
            runner_pid=None,
            current_child_pid=None,
            active_child_pids={},
            message=str(exc),
            last_error={
                "timestamp_utc": utc_now(),
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "emergency_child_cleanup": emergency_cleanup,
            },
        )
        event(
            "orchestrator_failed",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        return 1
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
