"""Sequential independent retraining for a promising architecture."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.training_client import (
    SUPPORTED_REQUESTED_DEVICES,
    WorkerError,
    run_worker_job,
)
from common.training_config import PROFILES, TrainingSeedBundle, get_training_profile
from common.evaluation_profiles import EvaluationLayer, resolve_evaluation_plan
from common.public_evaluation import (
    PUBLIC_LAYER_A_SOURCE_ID,
    PUBLIC_LAYER_A_SOURCE_SHA256,
)


def _safe_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"output directory is non-empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(PROFILES)),
        required=True,
    )
    parser.add_argument("--seeds", required=True)
    parser.add_argument(
        "--device",
        choices=SUPPORTED_REQUESTED_DEVICES,
        required=True,
        help="strict requested backend; CUDA never falls back to CPU",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-cpu-for-tests", action="store_true")
    parser.add_argument("--layer-a-cases", type=int, default=10_000)
    parser.add_argument("--evaluation-profile")
    parser.add_argument("--scientific-decision-record")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    output_root = Path(args.output_dir).resolve()
    _safe_root(output_root)
    profile = get_training_profile(args.profile)
    evaluation_profile = args.evaluation_profile or (
        "scientific_layer_a_v1" if profile.scientific else "smoke_eval_v1"
    )
    evaluation_plan = resolve_evaluation_plan(
        evaluation_profile,
        layer=EvaluationLayer.SEARCH,
        case_source_id=PUBLIC_LAYER_A_SOURCE_ID,
        case_source_sha256=PUBLIC_LAYER_A_SOURCE_SHA256,
        case_count=args.layer_a_cases,
        pi_decision_record_id=args.scientific_decision_record,
    )
    runs: list[dict] = []

    for run_seed in seeds:
        try:
            response = run_worker_job(
                mode="evaluate",
                candidate_path=args.candidate,
                output_dir=output_root / f"seed_{run_seed}",
                profile=profile,
                seeds=TrainingSeedBundle.from_run_seed(run_seed),
                requested_device=args.device,
                allow_cpu_for_tests=args.allow_cpu_for_tests,
                evaluation_plan={
                    **evaluation_plan.__dict__,
                    "layer": evaluation_plan.layer.value,
                },
                evaluation_context={
                    "study_id": "independent-retraining",
                    "block_id": "retraining",
                    "run_id": f"retraining-seed-{run_seed}",
                    "condition_id": "retraining",
                },
            )
        except WorkerError as error:
            response = {
                "kind": "worker_failure",
                "failure_stage": "worker_infrastructure",
                "error": str(error),
            }
        if response.get("kind") == "search_evaluation":
            evaluation = response["evaluation"]
            runs.append(
                {
                    "seed": run_seed,
                    "success": bool(evaluation.get("execution_ok")),
                    "public_accuracy": float(
                        evaluation.get("public_accuracy", 0.0)
                    ),
                    "eligible_for_parent": bool(
                        evaluation.get("eligible_for_parent")
                    ),
                    "failure_stage": evaluation.get("failure_stage", ""),
                }
            )
        else:
            runs.append(
                {
                    "seed": run_seed,
                    "success": False,
                    "public_accuracy": 0.0,
                    "eligible_for_parent": False,
                    "failure_stage": response.get(
                        "failure_stage", "worker_infrastructure"
                    ),
                }
            )

    accuracies = [run["public_accuracy"] for run in runs]
    aggregate = {
        "candidate": str(Path(args.candidate).resolve()),
        "profile": profile.name,
        "profile_version": profile.version,
        "profile_hash": profile.profile_hash,
        "requested_device": args.device,
        "device": args.device,
        "sequential": True,
        "success_count": sum(run["success"] for run in runs),
        "layer_a_eligibility_rate": sum(
            run["eligible_for_parent"] for run in runs
        ) / len(runs),
        "mean_public_accuracy": statistics.fmean(accuracies),
        "population_stddev_public_accuracy": (
            statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0
        ),
        "sealed_qualification_performed": False,
        "runs": runs,
    }
    destination = output_root / "aggregate_retraining_report.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_root,
        prefix=".aggregate_retraining_report.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
