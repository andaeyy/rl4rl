"""Single-incumbent discovery controller."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from openai import OpenAI
from openevolve.utils.code_utils import apply_diff, extract_diffs, format_diff_summary

from common.evaluator import SearchEvaluationContext, evaluate_candidate, file_hash
from common.gpt56_sol import GPT56SolProfile
from common.lineage_schema import CandidateRecord, append_record, text_hash, utc_now
from common.task_adapter import DEFAULT_TASK
from common.training_config import TrainingSeedBundle, get_training_profile


AGENT_DIR = Path(__file__).resolve().parent


def _provider_values() -> tuple[str, str, str]:
    values = {
        "DISCOVERY_API_KEY": os.environ.get("DISCOVERY_API_KEY"),
        "DISCOVERY_API_BASE": os.environ.get("DISCOVERY_API_BASE"),
        "DISCOVERY_MODEL": os.environ.get("DISCOVERY_MODEL"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SystemExit("missing provider configuration: " + ", ".join(missing))
    return (
        str(values["DISCOVERY_API_KEY"]),
        str(values["DISCOVERY_API_BASE"]),
        str(values["DISCOVERY_MODEL"]),
    )


def _git(lineage_repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=lineage_repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _request_completion(
    client: OpenAI,
    generation: GPT56SolProfile,
    messages: list[dict[str, str]],
):
    """Use the same explicit retry count and delay as OpenEvolve."""

    for attempt in range(generation.retries + 1):
        try:
            return client.chat.completions.create(
                **generation.chat_completion_request(messages)
            )
        except Exception as exc:
            if attempt >= generation.retries:
                raise
            print(
                "provider request failed; retrying "
                f"({attempt + 1}/{generation.retries}): {exc}",
                file=sys.stderr,
            )
            time.sleep(generation.retry_delay_seconds)

    raise RuntimeError("unreachable provider retry state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = yaml.safe_load((AGENT_DIR / "config.yaml").read_text())
    training_config = config["training"]
    training_profile = get_training_profile(training_config["profile"])
    if training_profile.version != str(training_config["profile_version"]):
        raise SystemExit("greedy training profile version mismatch")
    training_device = os.environ.get(
        "DISCOVERY_TRAIN_DEVICE", training_config["device"]
    )
    allow_cpu_for_tests = bool(training_config["allow_cpu_for_tests"])
    training_seeds = TrainingSeedBundle.from_run_seed(args.seed)
    api_key, api_base, model_name = _provider_values()
    try:
        generation = GPT56SolProfile.resolve(
            model=model_name,
            seed=args.seed,
            default_reasoning_effort=str(config["reasoning_effort"]),
            default_max_completion_tokens=int(config["max_tokens"]),
            default_timeout_seconds=int(config["timeout_seconds"]),
            default_retries=int(config["retries"]),
            default_retry_delay_seconds=int(config["retry_delay_seconds"]),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=generation.timeout_seconds,
        max_retries=0,
    )
    run_id = f"greedy-seed-{args.seed}-{uuid.uuid4().hex[:8]}"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else ROOT / "outputs" / "native_replications" / run_id
    )
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    ledger = output_dir / "lineage.jsonl"
    incumbent = output_dir / "incumbent.py"
    shutil.copy2(ROOT / "common" / "initial_candidate.py", incumbent)
    lineage_repo = output_dir / "accepted_lineage"
    lineage_repo.mkdir(parents=True, exist_ok=True)
    _git(lineage_repo, "init", "-q")
    _git(lineage_repo, "config", "user.name", "Architecture Discovery Controller")
    _git(lineage_repo, "config", "user.email", "discovery-controller@localhost")
    shutil.copy2(incumbent, lineage_repo / "candidate.py")
    _git(lineage_repo, "add", "candidate.py")
    _git(lineage_repo, "commit", "-q", "-m", "initial validated candidate")

    run_manifest = {
        "run_id": run_id,
        "condition": "greedy_autoresearch",
        "seed": args.seed,
        "candidate_budget": args.iterations,
        "mutation_budget": args.iterations,
        "candidate_training_budget": args.iterations,
        "generator": {
            **generation.manifest_fields(),
            "api_base_configured": True,
        },
        "initial_candidate_hash": file_hash(ROOT / "common" / "initial_candidate.py"),
        "evaluator_hash": file_hash(ROOT / "common" / "evaluator.py"),
        "config_hash": file_hash(AGENT_DIR / "config.yaml"),
        "training": {
            "profile": training_profile.name,
            "profile_version": training_profile.version,
            "profile_hash": training_profile.profile_hash,
            "task_adapter": DEFAULT_TASK.version,
            "task_adapter_hash": DEFAULT_TASK.config_hash,
            "seed_bundle": training_seeds.__dict__,
            "seed_bundle_hash": training_seeds.bundle_hash,
            "device": training_device,
            "allow_cpu_for_tests": allow_cpu_for_tests,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )

    system = "\n\n".join(
        [
            (ROOT / "common" / "prompts" / "shared_system.md").read_text(),
            (ROOT / "common" / "prompts" / "shared_task.md").read_text(),
            (AGENT_DIR / "program.md").read_text(),
        ]
    )
    parent_id = file_hash(incumbent)

    for iteration in range(args.iterations):
        proposal_time = utc_now()
        code = incumbent.read_text()
        user_prompt = (
            f"Iteration {iteration + 1}. Current candidate:\n\n"
            f"```python\n{code}\n```\n\n"
            "Return one hypothesis and SEARCH/REPLACE blocks."
        )
        response = _request_completion(
            client,
            generation,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        )
        response_text = response.choices[0].message.content or ""
        artifact_base = artifacts / f"{iteration + 1:04d}"
        artifact_base.with_suffix(".prompt.md").write_text(user_prompt)
        artifact_base.with_suffix(".response.md").write_text(response_text)
        diff_blocks = extract_diffs(response_text)
        if not diff_blocks:
            candidate_id = text_hash(response_text)
            usage = response.usage
            append_record(
                ledger,
                CandidateRecord(
                    run_id=run_id,
                    condition="greedy_autoresearch",
                    seed=args.seed,
                    candidate_id=candidate_id,
                    parent_id=parent_id,
                    proposal_text=response_text,
                    mechanism_hypothesis=response_text,
                    prompt_hash=text_hash(user_prompt),
                    response_hash=text_hash(response_text),
                    code_hash="",
                    proposal_timestamp=proposal_time,
                    completion_timestamp=utc_now(),
                    evaluation={"failure_stage": "mutation_format"},
                    retention_decision="crash",
                    rollback_target=parent_id,
                    input_tokens=getattr(usage, "prompt_tokens", 0),
                    output_tokens=getattr(usage, "completion_tokens", 0),
                ),
            )
            print(
                json.dumps(
                    {
                        "iteration": iteration + 1,
                        "candidate_id": candidate_id,
                        "decision": "crash",
                        "failure_stage": "mutation_format",
                    }
                )
            )
            continue
        child_code = apply_diff(code, response_text)
        candidate_id = text_hash(child_code)
        child_path = artifacts / f"{iteration + 1:04d}_{candidate_id[:12]}.py"
        child_path.write_text(child_code)
        artifact_base.with_suffix(".diff.txt").write_text(format_diff_summary(diff_blocks))
        candidate_training_dir = (
            output_dir
            / "candidate_training"
            / f"{iteration + 1:04d}_{candidate_id[:12]}"
        )
        evaluation = evaluate_candidate(
            child_path,
            training_profile=training_profile.name,
            training_seed=args.seed,
            training_output_dir=candidate_training_dir,
            device=training_device,
            allow_cpu_for_tests=allow_cpu_for_tests,
            evaluation_profile=os.environ.get("DISCOVERY_LAYER_A_PROFILE"),
            evaluation_case_count=(
                int(os.environ["DISCOVERY_LAYER_A_CASES"])
                if os.environ.get("DISCOVERY_LAYER_A_CASES")
                else None
            ),
            pi_decision_record_id=os.environ.get(
                "DISCOVERY_SCIENTIFIC_DECISION_RECORD"
            ),
            context=SearchEvaluationContext(
                study_id="native-replication",
                block_id="native-greedy",
                run_id=run_id,
                condition_id="native-greedy",
            ),
        )
        controller_evaluation = evaluation.controller_view()
        accepted = controller_evaluation.eligible_for_parent
        decision = "accept" if accepted else "reject"
        if accepted:
            shutil.copy2(child_path, incumbent)
            shutil.copy2(child_path, lineage_repo / "candidate.py")
            _git(lineage_repo, "add", "candidate.py")
            _git(
                lineage_repo,
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                f"accept iteration {iteration + 1}: {candidate_id[:12]}",
            )
            next_parent = candidate_id
            rollback_target = None
        else:
            next_parent = parent_id
            rollback_target = parent_id

        usage = response.usage
        record = CandidateRecord(
            run_id=run_id,
            condition="greedy_autoresearch",
            seed=args.seed,
            candidate_id=candidate_id,
            parent_id=parent_id,
            proposal_text=response_text.split("<<<<<<< SEARCH", 1)[0].strip(),
            mechanism_hypothesis=response_text.split("<<<<<<< SEARCH", 1)[0].strip(),
            prompt_hash=text_hash(user_prompt),
            response_hash=text_hash(response_text),
            code_hash=candidate_id,
            diff=format_diff_summary(diff_blocks),
            proposal_timestamp=proposal_time,
            completion_timestamp=utc_now(),
            evaluation=dict(controller_evaluation.as_dict()),
            retention_decision=decision,
            rollback_target=rollback_target,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
        )
        append_record(ledger, record)
        parent_id = next_parent
        print(
            json.dumps(
                {
                    "iteration": iteration + 1,
                    "candidate_id": candidate_id,
                    "decision": decision,
                    "eligible_for_parent": (
                        controller_evaluation.eligible_for_parent
                    ),
                    "search_score": controller_evaluation.search_score,
                }
            )
        )


if __name__ == "__main__":
    main()
