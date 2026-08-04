"""Shared launcher for generic and semantic OpenEvolve configurations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path

import yaml
from openevolve.config import LLMModelConfig, load_config
from openevolve.controller import OpenEvolve

from common.evaluator import file_hash
from common.gpt56_sol import GPT56SolProfile
from common.openevolve_policy import install_validity_first_policy
from common.task_adapter import DEFAULT_TASK
from common.training_client import resolve_harness_training_selection
from common.training_config import TrainingSeedBundle


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--profile",
        help=(
            "training profile override; defaults to DISCOVERY_TRAINING_PROFILE "
            "or the unchanged harness configuration"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cuda"),
        help=(
            "training accelerator override; defaults to DISCOVERY_TRAIN_DEVICE "
            "or the unchanged harness configuration"
        ),
    )
    return parser


def _build_model_config(
    generation: GPT56SolProfile, *, api_base: str, api_key: str
) -> LLMModelConfig:
    """Translate the shared profile into OpenEvolve's model configuration."""

    return LLMModelConfig(
        name=generation.model,
        weight=1.0,
        api_base=api_base,
        api_key=api_key,
        temperature=None,
        top_p=None,
        max_tokens=generation.max_completion_tokens,
        timeout=generation.timeout_seconds,
        retries=generation.retries,
        retry_delay=generation.retry_delay_seconds,
        random_seed=generation.seed,
        reasoning_effort=generation.reasoning_effort,
    )


def run_controller(kind: str) -> None:
    args = build_parser().parse_args()

    api_key = os.environ.get("DISCOVERY_API_KEY")
    api_base = os.environ.get("DISCOVERY_API_BASE")
    model_name = os.environ.get("DISCOVERY_MODEL")
    missing = [
        name
        for name, value in (
            ("DISCOVERY_API_KEY", api_key),
            ("DISCOVERY_API_BASE", api_base),
            ("DISCOVERY_MODEL", model_name),
        )
        if not value
    ]
    if missing:
        raise SystemExit("missing provider configuration: " + ", ".join(missing))

    agent_dir = ROOT / "agents" / f"openevolve_{kind}"
    run_id = f"openevolve-{kind}-seed-{args.seed}-{uuid.uuid4().hex[:8]}"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else ROOT / "outputs" / "native_replications" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_config = yaml.safe_load((agent_dir / "config.yaml").read_text())
    training_config = raw_config["training"]
    try:
        training_selection = resolve_harness_training_selection(
            training_config,
            profile_override=args.profile,
            device_override=args.device,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid OpenEvolve training selection: {exc}") from exc
    training_profile = training_selection.profile
    training_device = training_selection.requested_device
    allow_cpu_for_tests = training_selection.allow_cpu_for_tests
    training_seeds = TrainingSeedBundle.from_run_seed(args.seed)
    os.environ["DISCOVERY_TRAINING_PROFILE"] = training_profile.name
    os.environ["DISCOVERY_TRAINING_SEED"] = str(args.seed)
    os.environ["DISCOVERY_TRAIN_DEVICE"] = training_device
    os.environ["DISCOVERY_ALLOW_CPU_TRAINING"] = (
        "1" if allow_cpu_for_tests else "0"
    )
    os.environ["DISCOVERY_TRAINING_OUTPUT_ROOT"] = str(
        output_dir / "candidate_training"
    )
    os.environ["DISCOVERY_STUDY_ID"] = "native-replication"
    os.environ["DISCOVERY_BLOCK_ID"] = f"native-openevolve-{kind}"
    os.environ["DISCOVERY_RUN_ID"] = run_id
    os.environ["DISCOVERY_CONDITION_ID"] = f"native-openevolve-{kind}"

    config = load_config(agent_dir / "config.yaml")
    try:
        generation = GPT56SolProfile.resolve(
            model=str(model_name),
            seed=args.seed,
            default_reasoning_effort=str(config.llm.reasoning_effort),
            default_max_completion_tokens=int(config.llm.max_tokens),
            default_timeout_seconds=int(config.llm.timeout),
            default_retries=int(config.llm.retries),
            default_retry_delay_seconds=int(config.llm.retry_delay),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config.max_iterations = args.iterations
    config.random_seed = args.seed
    config.database.random_seed = args.seed
    config.database.db_path = str(output_dir / "database")
    config.database.in_memory = False
    config.log_dir = str(output_dir / "logs")
    config.prompt.system_message = "\n\n".join(
        [
            (ROOT / "common" / "prompts" / "shared_system.md").read_text(),
            (ROOT / "common" / "prompts" / "shared_task.md").read_text(),
            (agent_dir / "system_prompt.md").read_text(),
        ]
    )
    config.evolution_trace.output_path = str(output_dir / "evolution_trace.jsonl")
    config.llm.temperature = None
    config.llm.top_p = None
    config.llm.max_tokens = generation.max_completion_tokens
    config.llm.timeout = generation.timeout_seconds
    config.llm.retries = generation.retries
    config.llm.retry_delay = generation.retry_delay_seconds
    config.llm.reasoning_effort = generation.reasoning_effort
    model_config = _build_model_config(
        generation,
        api_base=str(api_base),
        api_key=str(api_key),
    )
    config.llm.models = [model_config]
    config.llm.evaluator_models = [model_config]

    install_validity_first_policy()
    if kind == "semantic":
        from agents.openevolve_semantic.semantic_archive import install_semantic_archive

        install_semantic_archive()

    run_manifest = {
        "run_id": run_id,
        "condition": f"openevolve_{kind}",
        "seed": args.seed,
        "candidate_budget": args.iterations + 1,
        "mutation_budget": args.iterations,
        "candidate_training_budget": args.iterations + 1,
        "initial_program_is_evaluated": True,
        "generator": {
            **generation.manifest_fields(),
            "api_base_configured": bool(api_base),
        },
        "initial_candidate_hash": file_hash(ROOT / "common" / "initial_candidate.py"),
        "evaluator_hash": file_hash(ROOT / "common" / "evaluator.py"),
        "config_hash": file_hash(agent_dir / "config.yaml"),
        "training": {
            **training_selection.manifest_fields(),
            "task_adapter": DEFAULT_TASK.version,
            "task_adapter_hash": DEFAULT_TASK.config_hash,
            "seed_bundle": training_seeds.__dict__,
            "seed_bundle_hash": training_seeds.bundle_hash,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )

    controller = OpenEvolve(
        initial_program_path=str(ROOT / "common" / "initial_candidate.py"),
        evaluation_file=str(agent_dir / "evaluator_adapter.py"),
        config=config,
        output_dir=str(output_dir),
    )
    asyncio.run(controller.run(iterations=args.iterations))
