"""Sequential independent retraining for a promising architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.candidate_artifact import inspect_candidate_artifact  # noqa: E402
from common.evaluation_profiles import (  # noqa: E402
    EvaluationLayer,
    resolve_evaluation_plan,
)
from common.public_evaluation import (  # noqa: E402
    PUBLIC_LAYER_A_SOURCE_ID,
    PUBLIC_LAYER_A_SOURCE_SHA256,
)
from common.training_client import WorkerError, run_worker_job  # noqa: E402
from common.training_config import (  # noqa: E402
    PROFILES,
    TrainingSeedBundle,
    get_training_profile,
)


def _safe_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"output directory is non-empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_aggregate_report(
    *,
    candidate: Path,
    profile,
    evaluation_plan,
    requested_device: str,
    seeds: list[int],
    runs: list[dict],
) -> dict:
    """Build a portable v2 aggregate while preserving the historical v1 shape."""

    accuracies = [run["public_accuracy"] for run in runs]
    shared = {
        "profile": profile.name,
        "profile_version": profile.version,
        "profile_hash": profile.profile_hash,
        "device": requested_device,
        "sequential": True,
        "success_count": sum(run["success"] for run in runs),
        "layer_a_eligibility_rate": sum(
            run["eligible_for_parent"] for run in runs
        )
        / len(runs),
        "mean_public_accuracy": statistics.fmean(accuracies),
        "population_stddev_public_accuracy": (
            statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0
        ),
        "sealed_qualification_performed": False,
        "runs": runs,
    }
    if profile.version == "1":
        return {
            "candidate": str(candidate.resolve()),
            **shared,
        }

    inspection = inspect_candidate_artifact(candidate)
    if not inspection.valid:
        raise ValueError(
            "candidate contract failed before aggregate recording: "
            + "; ".join(inspection.reasons)
        )
    immutable_name = (
        "candidate_graph.json"
        if inspection.candidate_format.value == "architecture_ir"
        else "candidate_source.py"
    )
    plan = asdict(evaluation_plan)
    plan["layer"] = evaluation_plan.layer.value
    plan["plan_hash"] = evaluation_plan.plan_hash
    return {
        "schema_name": "AggregateRetrainingReport",
        "schema_version": "2.0",
        "candidate_format": inspection.candidate_format.value,
        "candidate_source_hash": _sha256_file(candidate),
        "candidate_graph_hash": inspection.graph_hash,
        "candidate_artifact_paths": [
            f"seed_{run_seed}/{immutable_name}" for run_seed in seeds
        ],
        "evaluation_plan": plan,
        **shared,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(PROFILES)),
        required=True,
    )
    parser.add_argument("--seeds", required=True)
    parser.add_argument(
        "--device", choices=("cuda", "mps", "cpu"), required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-cpu-for-tests", action="store_true")
    parser.add_argument("--layer-a-cases", type=int, default=10_000)
    parser.add_argument("--evaluation-profile")
    parser.add_argument("--scientific-decision-record")
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    candidate = Path(args.candidate).resolve()
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
                candidate_path=candidate,
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
            run = {
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
            if profile.version == "2":
                # Retain the complete typed Layer A record so a downloaded
                # Modal smoke can revalidate more than aggregate scalars.
                run["evaluation_record"] = evaluation
            runs.append(run)
        else:
            run = {
                "seed": run_seed,
                "success": False,
                "public_accuracy": 0.0,
                "eligible_for_parent": False,
                "failure_stage": response.get(
                    "failure_stage", "worker_infrastructure"
                ),
            }
            if profile.version == "2":
                run["evaluation_record"] = None
            runs.append(run)

    aggregate = _build_aggregate_report(
        candidate=candidate,
        profile=profile,
        evaluation_plan=evaluation_plan,
        requested_device=args.device,
        seeds=seeds,
        runs=runs,
    )
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
