"""Direct entrypoint for one evaluator-owned candidate training run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.training_client import WorkerError, run_worker_job
from common.training_config import TrainingSeedBundle, get_training_profile
from common.trainer import validate_training_request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--profile",
        choices=("full_train_v1", "smoke_train_v1"),
        required=True,
    )
    parser.add_argument("--device", choices=("mps", "cpu"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--allow-cpu-for-tests", action="store_true")
    args = parser.parse_args()

    profile = get_training_profile(args.profile)
    seeds = TrainingSeedBundle.from_run_seed(args.seed)
    if args.dry_run:
        try:
            resolved = validate_training_request(
                candidate_path=args.candidate,
                profile=profile,
                seeds=seeds,
                requested_device=args.device,
                allow_cpu_for_tests=args.allow_cpu_for_tests,
                output_dir=args.output_dir,
                resume=args.resume,
            )
        except Exception as error:
            raise SystemExit(
                f"dry-run validation failed: {type(error).__name__}: {error}"
            ) from error
        resolved["dry_run"] = True
        resolved["training_updates"] = 0
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return

    try:
        response = run_worker_job(
            mode="train",
            candidate_path=args.candidate,
            output_dir=args.output_dir,
            profile=profile,
            seeds=seeds,
            requested_device=args.device,
            allow_cpu_for_tests=args.allow_cpu_for_tests,
            resume=args.resume,
        )
    except WorkerError as error:
        raise SystemExit(f"training worker launch failed: {error}") from error
    if response.get("kind") != "training_result":
        print(json.dumps(response, indent=2, sort_keys=True))
        raise SystemExit("training worker failed")
    training = response["training"]
    print(json.dumps(training, indent=2, sort_keys=True))
    if not training["success"]:
        raise SystemExit(
            f"candidate training failed at {training['failure_stage']}"
        )


if __name__ == "__main__":
    main()
