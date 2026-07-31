"""Command-line interface for the complete PoC pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .catalog import build_catalog
from .config import ensure_runtime_directories, load_config
from .demo import make_demo
from .fragility import fit_all_fragilities
from .ground_motion import (
    build_ground_motion_selection,
    validate_records,
)
from .ida import add_ida_points, benchmark_ida, run_ida_batch
from .ml import create_ml_split, evaluate, train_ann
from .reporting import build_consolidated_workbook
from .sensitivity import (
    run_modeling_sensitivity,
    run_plastic_hinge_length_sensitivity,
)
from .spo import run_modal_spo


def _configure_utf8_console() -> None:
    """Keep Thai Windows paths printable without requiring shell variables."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fragility-poc",
        description=(
            "End-to-end 5-storey multi-bay RC "
            "SPO–IDA–fragility–ANN demonstration"
        ),
    )
    parser.add_argument(
        "--config",
        default="config/poc.json",
        help="Path to the JSON configuration (default: config/poc.json)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "validate-records",
        help="Discover, validate, checksum, and catalogue X–Y records",
    )
    subparsers.add_parser(
        "build-catalog",
        help=(
            "Enumerate Draft 5 engineered archetypes and build the balanced "
            "100–375 Building-ID queue"
        ),
    )
    selection = subparsers.add_parser(
        "select-records",
        help="Select nearest CMS period family/families from each building T1",
    )
    selection.add_argument("--building-id", action="append")
    spo = subparsers.add_parser(
        "run-modal-spo",
        help="Run gravity, modal analysis, and X-direction SPO",
    )
    spo.add_argument("--building-id", action="append")
    spo.add_argument("--limit", type=int)
    spo.add_argument("--no-resume", action="store_true")
    spo.add_argument(
        "--validate-xy",
        action="store_true",
        help="Run Y-direction SPO for the first three symmetric models",
    )
    hinge_sensitivity = subparsers.add_parser(
        "plastic-hinge-sensitivity",
        help=(
            "Run isolated SPO sensitivity cases for alternative plastic-hinge "
            "length factors"
        ),
    )
    hinge_sensitivity.add_argument("--building-id", action="append")
    hinge_sensitivity.add_argument(
        "--factor",
        action="append",
        type=float,
        help=(
            "Multiplier on the empirical Priestley Lp; repeat for at least "
            "three values and include the configured baseline"
        ),
    )
    hinge_sensitivity.add_argument("--no-resume", action="store_true")
    modeling_sensitivity = subparsers.add_parser(
        "modeling-sensitivity",
        help=(
            "Run fiber-mesh/material-strain SPO sensitivity and fixed-IM "
            "damping NLTHA sensitivity"
        ),
    )
    modeling_sensitivity.add_argument("--building-id", action="append")
    modeling_sensitivity.add_argument("--no-resume", action="store_true")
    subparsers.add_parser(
        "benchmark-ida",
        help=(
            "Benchmark 3 research column/beam margin bands x 3 "
            "duration-ranked pairs and set compute gate"
        ),
    )
    split = subparsers.add_parser(
        "create-ml-split",
        help="Create the label-blind 80/20 and five-fold Building-ID manifest",
    )
    split.add_argument(
        "--force",
        action="store_true",
        help="Replace a stale split after the SPO/catalog inputs change",
    )
    ida = subparsers.add_parser(
        "run-ida",
        help="Run checkpointed bidirectional adaptive IDA",
    )
    ida.add_argument("--building-id", action="append")
    ida.add_argument(
        "--pair-id",
        action="append",
        help=(
            "Run/resume only the named already-selected GM pair; intended "
            "for isolated recovery or diagnostic reruns"
        ),
    )
    ida.add_argument("--limit", type=int)
    ida.add_argument("--workers", type=int)
    ida.add_argument("--no-resume", action="store_true")
    ida.add_argument(
        "--allow-incomplete-records",
        action="store_true",
        help="Permit a clearly labelled CMS-only smoke run before PWSA arrives",
    )
    add_points = subparsers.add_parser(
        "add-ida-points",
        help=(
            "Run only explicitly requested IM targets, reuse every existing "
            "checkpoint, and rebuild only the affected curve/fragility"
        ),
    )
    add_points.add_argument("--building-id", required=True)
    add_points.add_argument("--pair-id", required=True)
    add_points.add_argument(
        "--target-im-g",
        action="append",
        type=float,
        required=True,
        help="Target Sa,GM(T1,5%%) in g; repeat for multiple points",
    )
    add_points.add_argument(
        "--requester",
        default="User/Codex",
        help="Audit-trail name for the point request",
    )
    add_points.add_argument(
        "--reason",
        default="Additional IDA accuracy/review point",
    )
    fragility = subparsers.add_parser(
        "fit-fragility",
        help=(
            "Fit complete-capacity lognormal IO/LS/CP fragilities with "
            "sample RTR beta and bootstrap CIs"
        ),
    )
    fragility.add_argument("--building-id", action="append")
    fragility.add_argument("--bootstrap-count", type=int)
    fragility.add_argument(
        "--allow-incomplete-records",
        action="store_true",
        help="Fit a CMS-only demonstration; it will not enter the ML dataset",
    )
    subparsers.add_parser(
        "train-ann",
        help="Create leakage-safe split, select ANN, fit baselines, predict test",
    )
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Score the independent test once against predefined PoC criteria",
    )
    evaluate_parser.add_argument("--run-id")
    subparsers.add_parser(
        "make-demo",
        help="Generate committee figures and an evidence-status report",
    )
    subparsers.add_parser(
        "make-workbook",
        help="Generate the single consolidated current-results Excel workbook",
    )
    return parser


def _dispatch(arguments: argparse.Namespace, config: dict[str, Any]) -> Any:
    command = arguments.command
    if command == "validate-records":
        return validate_records(config)
    if command == "build-catalog":
        return build_catalog(config)
    if command == "select-records":
        return build_ground_motion_selection(
            config, building_ids=arguments.building_id
        )
    if command == "run-modal-spo":
        return run_modal_spo(
            config,
            building_ids=arguments.building_id,
            limit=arguments.limit,
            resume=not arguments.no_resume,
            validate_xy=arguments.validate_xy,
        )
    if command == "plastic-hinge-sensitivity":
        return run_plastic_hinge_length_sensitivity(
            config,
            building_ids=arguments.building_id,
            factors=arguments.factor,
            resume=not arguments.no_resume,
        )
    if command == "modeling-sensitivity":
        return run_modeling_sensitivity(
            config,
            building_ids=arguments.building_id,
            resume=not arguments.no_resume,
        )
    if command == "benchmark-ida":
        return benchmark_ida(config)
    if command == "create-ml-split":
        return create_ml_split(config, force=arguments.force)
    if command == "run-ida":
        return run_ida_batch(
            config,
            building_ids=arguments.building_id,
            pair_ids=arguments.pair_id,
            limit=arguments.limit,
            workers=arguments.workers,
            resume=not arguments.no_resume,
            allow_incomplete_records=arguments.allow_incomplete_records,
        )
    if command == "add-ida-points":
        return add_ida_points(
            config,
            building_id=arguments.building_id,
            pair_id=arguments.pair_id,
            target_ims_g=arguments.target_im_g,
            requester=arguments.requester,
            reason=arguments.reason,
        )
    if command == "fit-fragility":
        return fit_all_fragilities(
            config,
            building_ids=arguments.building_id,
            bootstrap_count=arguments.bootstrap_count,
            allow_incomplete_records=arguments.allow_incomplete_records,
        )
    if command == "train-ann":
        return train_ann(config)
    if command == "evaluate":
        return evaluate(config, run_id=arguments.run_id)
    if command == "make-demo":
        return make_demo(config)
    if command == "make-workbook":
        return build_consolidated_workbook(config)
    raise AssertionError(f"Unhandled command: {command}")


def _command_result_failures(
    arguments: argparse.Namespace,
    result: object,
) -> list[str]:
    """Translate structured pipeline summaries into a failing CLI status.

    The pipeline functions return detailed summaries so callers can inspect
    partial work. A production CLI is fail-closed: exit code zero means the
    requested scientific result is complete and valid, not merely that the
    Python function returned normally.
    """
    if not isinstance(result, dict):
        return []
    command = str(arguments.command)
    failures: list[str] = []
    failure_count = int(result.get("failure_count", 0) or 0)
    if failure_count:
        failures.append(f"failure_count={failure_count}")

    if command == "run-modal-spo":
        completed = int(result.get("completed_count", 0) or 0)
        valid = int(result.get("valid_count", 0) or 0)
        if valid != completed:
            failures.append(
                f"valid_count={valid} differs from completed_count={completed}"
            )
        xy = result.get("xy_equivalence")
        if isinstance(xy, dict) and xy.get("valid") is False:
            failures.append("X-Y SPO equivalence validation failed")
    elif command == "run-ida":
        invalid = int(result.get("invalid_curve_count", 0) or 0)
        if invalid:
            failures.append(f"invalid_curve_count={invalid}")
        if result.get("batch_complete") is not True:
            failures.append("batch_complete is not true")
    elif command == "fit-fragility":
        fitted = int(result.get("fitted_building_count", 0) or 0)
        valid = int(result.get("valid_building_count", 0) or 0)
        requested_buildings = {
            str(value)
            for value in (getattr(arguments, "building_id", None) or [])
        }
        if requested_buildings and fitted != len(requested_buildings):
            failures.append(
                f"fitted_building_count={fitted} differs from explicitly "
                f"requested building count={len(requested_buildings)}"
            )
        if not bool(result.get("allow_incomplete_records", False)) and (
            fitted != valid
        ):
            failures.append(
                f"valid_building_count={valid} differs from "
                f"fitted_building_count={fitted}"
            )
    return failures


def main() -> int:
    _configure_utf8_console()
    parser = _common_parser()
    arguments = parser.parse_args()
    try:
        config = load_config(arguments.config)
        ensure_runtime_directories(config)
        result = _dispatch(arguments, config)
        result_failures = _command_result_failures(arguments, result)
        if result_failures:
            raise RuntimeError(
                "Pipeline result failed completion/validity checks: "
                + "; ".join(result_failures)
            )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "command": arguments.command,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {"status": "ok", "command": arguments.command, "result": result},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
