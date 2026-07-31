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


def run_stage(
    state: dict[str, Any],
    batch_id: str,
    stage_name: str,
    ids: list[str],
    workers: int,
) -> None:
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
            return

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

        for batch_id in BATCHES:
            with connect(DATABASE, readonly=True) as connection:
                ids = ids_for_batch(connection, batch_id)
            if len(ids) != expected_batch_count(batch_id):
                raise RuntimeError(
                    f"{batch_id} has {len(ids)} frozen IDs, expected "
                    f"{expected_batch_count(batch_id)}"
                )
            for stage_name in (
                "modal_spo",
                "select_records",
                "full_ida",
                "fragility",
                "export",
            ):
                run_stage(
                    state, batch_id, stage_name, ids, workers
                )

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
        save_state(
            state,
            status="needs_attention",
            runner_pid=None,
            current_child_pid=None,
            message=str(exc),
            last_error={
                "timestamp_utc": utc_now(),
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
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
