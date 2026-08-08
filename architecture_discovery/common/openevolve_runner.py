"""Shared launcher for generic and semantic OpenEvolve configurations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import yaml
from openevolve.config import LLMModelConfig, load_config
from openevolve.controller import OpenEvolve
from openevolve.process_parallel import (
    SerializableResult,
    _run_iteration_worker as _VENDOR_RUN_ITERATION_WORKER,
)

from architecture_ir.interpreter import (
    validate_ir_candidate_json,
    validate_ir_candidate_path,
)
from architecture_ir.graph import ARCHITECTURE_HASH_SCHEMA
from common.architecture_dedup import ArchitectureHashRegistry
from common.evaluator import file_hash
from common.evaluation_profiles import (
    EVALUATION_PROFILES,
    EvaluationLayer,
    EvaluationPlan,
    resolve_evaluation_plan,
)
from common.gpt56_sol import GPT56SolProfile, resolve_provider_endpoint
from common.openevolve_policy import install_validity_first_policy
from common.public_evaluation import (
    PUBLIC_LAYER_A_SOURCE_ID,
    PUBLIC_LAYER_A_SOURCE_SHA256,
)
from common.task_adapter import DEFAULT_TASK
from common.trainer import (
    trusted_component_hashes,
    trusted_component_set_sha256,
    validate_training_request,
)
from common.training_config import PROFILES, TrainingSeedBundle, get_training_profile


ROOT = Path(__file__).resolve().parents[1]
INITIAL_IR_CANDIDATE = ROOT / "common" / "initial_candidate.ir.json"
ENGINEERING_TRAINING_PROFILE = "smoke_train_v1"
ENGINEERING_EVALUATION_PROFILE = "smoke_eval_v1"
_PARENT_CHANGE_ENFORCEMENT_ENV = "DISCOVERY_ENFORCE_PARENT_ARCHITECTURE_CHANGE"
_PARENT_ARCHITECTURE_HASH_ENV = "DISCOVERY_PARENT_ARCHITECTURE_HASH"
_INITIAL_ARCHITECTURE_HASH_ENV = "DISCOVERY_INITIAL_ARCHITECTURE_HASH"
_INITIAL_EVALUATION_AUTH_ENV = "DISCOVERY_OPENEVOLVE_INITIAL_EVALUATION_AUTH"
_ARCHITECTURE_REGISTRY_ENV = "DISCOVERY_ARCHITECTURE_HASH_REGISTRY"
_TERMINAL_LEDGER_ENV = "DISCOVERY_OPENEVOLVE_TERMINAL_LEDGER"


def _architecture_hash_for_ir_text(text: str) -> str:
    validation = validate_ir_candidate_json(text)
    if not validation.valid or validation.graph is None:
        reasons = "; ".join(
            f"{issue.code}: {issue.message}" for issue in validation.issues
        )
        raise ValueError(f"invalid parent Architecture IR: {reasons}")
    return validation.graph.architecture_hash


def _record_terminal_outcome(iteration: int, result: SerializableResult) -> None:
    """Append one non-secret terminal record from the iteration worker."""

    ledger_value = os.environ.get(_TERMINAL_LEDGER_ENV)
    if not ledger_value:
        return
    status = "error" if result.error else "candidate"
    payload = {
        "iteration": iteration,
        "status": status,
        "candidate_produced": bool(result.child_program_dict),
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(Path(ledger_value), flags, 0o600)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def _terminal_outcome_summary(path: Path, requested: int) -> dict[str, object]:
    """Return strict, unique per-iteration completion evidence."""

    outcomes: dict[int, str] = {}
    accounting_errors: list[str] = []
    if path.exists():
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                payload = json.loads(raw_line)
                iteration = payload["iteration"]
                status = payload["status"]
                if (
                    isinstance(iteration, bool)
                    or not isinstance(iteration, int)
                    or not 1 <= iteration <= requested
                ):
                    raise ValueError("iteration is outside the requested range")
                if status not in {"candidate", "error"}:
                    raise ValueError("status is not a recognized terminal outcome")
                if iteration in outcomes:
                    raise ValueError("iteration has more than one terminal outcome")
                outcomes[iteration] = status
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                accounting_errors.append(f"line {line_number}: {error}")
    counts = Counter(outcomes.values())
    return {
        "terminal_count": len(outcomes),
        "terminal_iterations": sorted(outcomes),
        "terminal_status_counts": dict(sorted(counts.items())),
        "accounting_errors": accounting_errors,
        "all_requested_terminal": (
            len(outcomes) == requested and not accounting_errors
        ),
    }


def _parent_bound_iteration_worker(
    iteration: int,
    db_snapshot: dict,
    parent_id: str,
    inspiration_ids: list[str],
) -> SerializableResult:
    """Bind a selected parent's executable identity into child evaluation.

    This top-level wrapper is picklable under macOS ``spawn``.  The vendor
    worker still owns proposal generation and bookkeeping; the only added
    behavior is a trusted parent hash that the evaluator adapter checks before
    entering candidate training.
    """

    try:
        parent_record = db_snapshot["programs"][parent_id]
        parent_code = parent_record["code"]
        if not isinstance(parent_code, str):
            raise ValueError("parent program code must be Architecture IR text")
        parent_architecture_hash = _architecture_hash_for_ir_text(parent_code)
    except (KeyError, TypeError, ValueError) as error:
        result = SerializableResult(
            error=f"parent architecture binding failed: {error}",
            iteration=iteration,
        )
        _record_terminal_outcome(iteration, result)
        return result

    previous_hash = os.environ.get(_PARENT_ARCHITECTURE_HASH_ENV)
    os.environ[_PARENT_ARCHITECTURE_HASH_ENV] = parent_architecture_hash
    try:
        try:
            result = _VENDOR_RUN_ITERATION_WORKER(
                iteration,
                db_snapshot,
                parent_id,
                inspiration_ids,
            )
        except Exception as error:  # defensive boundary around vendor worker
            result = SerializableResult(
                error=f"iteration worker failed: {type(error).__name__}: {error}",
                iteration=iteration,
            )
    finally:
        if previous_hash is None:
            os.environ.pop(_PARENT_ARCHITECTURE_HASH_ENV, None)
        else:
            os.environ[_PARENT_ARCHITECTURE_HASH_ENV] = previous_hash
    _record_terminal_outcome(iteration, result)
    return result


def _install_parent_bound_worker():
    """Install the wrapper for this run and return the previous worker."""

    import openevolve.process_parallel as process_parallel

    previous = process_parallel._run_iteration_worker
    process_parallel._run_iteration_worker = _parent_bound_iteration_worker
    return previous


def _restore_iteration_worker(previous) -> None:
    import openevolve.process_parallel as process_parallel

    process_parallel._run_iteration_worker = previous


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _environment_case_count() -> int | None:
    value = os.environ.get("DISCOVERY_LAYER_A_CASES")
    if value is None or not value.strip():
        return None
    try:
        return _positive_int(value)
    except argparse.ArgumentTypeError as error:
        raise SystemExit(f"invalid DISCOVERY_LAYER_A_CASES: {error}") from error


def _preflight_runtime(
    *,
    candidate_path: Path,
    training_profile,
    training_seeds: TrainingSeedBundle,
    training_device: str,
    allow_cpu_for_tests: bool,
    evaluation_profile_name: str,
    evaluation_case_count: int | None,
    pi_decision_record_id: str | None,
) -> EvaluationPlan:
    """Validate the complete non-provider execution plan without training."""

    try:
        plan = resolve_evaluation_plan(
            evaluation_profile_name,
            layer=EvaluationLayer.SEARCH,
            case_source_id=PUBLIC_LAYER_A_SOURCE_ID,
            case_source_sha256=PUBLIC_LAYER_A_SOURCE_SHA256,
            case_count=evaluation_case_count,
            pi_decision_record_id=pi_decision_record_id,
        )
        if plan.scientific != training_profile.scientific:
            raise ValueError(
                "training and evaluation profiles must have matching "
                "scientific status"
            )
        validate_training_request(
            candidate_path=candidate_path,
            profile=training_profile,
            seeds=training_seeds,
            requested_device=training_device,
            allow_cpu_for_tests=allow_cpu_for_tests,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(
            f"OpenEvolve preflight failed before provider initialization: {error}"
        ) from error
    return plan


def _require_fresh_output(path: Path) -> None:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise SystemExit(f"OpenEvolve output directory may not be a symlink: {expanded}")
    if expanded.exists() and not expanded.is_dir():
        raise SystemExit(f"OpenEvolve output path is not a directory: {expanded}")
    if expanded.exists() and any(expanded.iterdir()):
        raise SystemExit(
            f"OpenEvolve output directory is non-empty: {expanded}. "
            "Safe native-controller resume is not implemented; use a fresh directory."
        )


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


def run_controller(kind: str, argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=_positive_int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--engineering-pilot",
        action="store_true",
        help=(
            "run the explicitly non-authoritative IR canary with the frozen "
            "smoke training and evaluation profiles"
        ),
    )
    parser.add_argument("--training-profile", choices=sorted(PROFILES))
    parser.add_argument(
        "--evaluation-profile",
        choices=sorted(EVALUATION_PROFILES),
    )
    parser.add_argument("--evaluation-cases", type=_positive_int)
    parser.add_argument("--device", choices=("mps", "cpu"))
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args(argv)
    if args.resume_checkpoint:
        raise SystemExit(
            "Safe native-controller resume is not implemented; omit "
            "--resume-checkpoint and use a fresh output directory."
        )

    agent_dir = ROOT / "agents" / f"openevolve_{kind}"
    if kind not in {"generic", "semantic"} or not agent_dir.is_dir():
        raise SystemExit(f"unsupported OpenEvolve controller kind: {kind!r}")
    run_id = f"openevolve-{kind}-seed-{args.seed}-{uuid.uuid4().hex[:8]}"
    raw_output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else ROOT / "outputs" / "native_replications" / run_id
    )
    _require_fresh_output(raw_output_dir)
    output_dir = raw_output_dir.resolve()

    raw_config = yaml.safe_load((agent_dir / "config.yaml").read_text())
    training_config = raw_config["training"]
    configured_profile_name = str(training_config["profile"])
    if args.engineering_pilot:
        if args.training_profile not in (None, ENGINEERING_TRAINING_PROFILE):
            raise SystemExit(
                "--engineering-pilot fixes --training-profile to "
                f"{ENGINEERING_TRAINING_PROFILE}"
            )
        if args.evaluation_profile not in (None, ENGINEERING_EVALUATION_PROFILE):
            raise SystemExit(
                "--engineering-pilot fixes --evaluation-profile to "
                f"{ENGINEERING_EVALUATION_PROFILE}"
            )
        requested_training_profile = ENGINEERING_TRAINING_PROFILE
        requested_evaluation_profile = ENGINEERING_EVALUATION_PROFILE
    else:
        requested_training_profile = args.training_profile or configured_profile_name
        requested_evaluation_profile = args.evaluation_profile
    try:
        training_profile = get_training_profile(requested_training_profile)
    except ValueError as error:
        raise SystemExit(f"invalid OpenEvolve training profile: {error}") from error
    if not args.engineering_pilot and not training_profile.scientific:
        raise SystemExit(
            "non-scientific OpenEvolve runs require the explicit "
            "--engineering-pilot flag"
        )
    if (
        not args.engineering_pilot
        and args.training_profile is None
        and training_profile.version != str(training_config["profile_version"])
    ):
        raise SystemExit("OpenEvolve training profile version mismatch")
    training_device = str(
        args.device
        or os.environ.get("DISCOVERY_TRAIN_DEVICE")
        or training_config["device"]
    ).lower()
    if args.engineering_pilot and training_device != "mps":
        raise SystemExit(
            "provider-backed OpenEvolve engineering pilots require device='mps'"
        )
    allow_cpu_for_tests = bool(training_config["allow_cpu_for_tests"])
    training_seeds = TrainingSeedBundle.from_run_seed(args.seed)
    evaluation_profile_name = str(
        requested_evaluation_profile
        or (
            None
            if args.engineering_pilot
            else os.environ.get("DISCOVERY_LAYER_A_PROFILE")
        )
        or (
            "scientific_layer_a_v1"
            if training_profile.scientific
            else ENGINEERING_EVALUATION_PROFILE
        )
    )
    evaluation_case_count = (
        args.evaluation_cases
        if args.evaluation_cases is not None
        else (None if args.engineering_pilot else _environment_case_count())
    )
    evaluation_plan = _preflight_runtime(
        candidate_path=INITIAL_IR_CANDIDATE,
        training_profile=training_profile,
        training_seeds=training_seeds,
        training_device=training_device,
        allow_cpu_for_tests=allow_cpu_for_tests,
        evaluation_profile_name=evaluation_profile_name,
        evaluation_case_count=evaluation_case_count,
        pi_decision_record_id=(
            None
            if args.engineering_pilot
            else os.environ.get("DISCOVERY_SCIENTIFIC_DECISION_RECORD")
        ),
    )
    initial_validation = validate_ir_candidate_path(INITIAL_IR_CANDIDATE)
    if not initial_validation.valid or initial_validation.graph is None:
        reasons = "; ".join(
            f"{issue.code}: {issue.message}" for issue in initial_validation.issues
        )
        raise SystemExit(
            "OpenEvolve initial Architecture IR is invalid before provider "
            f"initialization: {reasons}"
        )
    initial_architecture_hash = initial_validation.graph.architecture_hash

    # Provider configuration is deliberately read only after the execution
    # plan, device, candidate contract, and scientific containment gate pass.
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
    try:
        provider_endpoint = resolve_provider_endpoint(
            str(api_base),
            scientific=training_profile.scientific,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    # An explicitly supplied empty directory is permitted; non-empty paths were
    # rejected above before any provider object could be constructed.
    output_dir.mkdir(parents=True, exist_ok=True)
    architecture_registry = ArchitectureHashRegistry(
        output_dir / "architecture_hash_registry"
    )
    if not architecture_registry.claim(initial_architecture_hash):
        raise SystemExit("fresh architecture registry rejected the initial seed")
    terminal_ledger = output_dir / "proposal_terminal_outcomes.jsonl"
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
    os.environ["DISCOVERY_LAYER_A_PROFILE"] = evaluation_plan.profile_name
    os.environ["DISCOVERY_LAYER_A_CASES"] = str(evaluation_plan.case_count)
    os.environ["DISCOVERY_ENGINEERING_PILOT"] = (
        "1" if args.engineering_pilot else "0"
    )
    os.environ[_PARENT_CHANGE_ENFORCEMENT_ENV] = "1"
    os.environ[_INITIAL_ARCHITECTURE_HASH_ENV] = initial_architecture_hash
    os.environ[_ARCHITECTURE_REGISTRY_ENV] = str(architecture_registry.directory)
    os.environ[_TERMINAL_LEDGER_ENV] = str(terminal_ledger)
    # The controller evaluates the initial candidate in the main process.  A
    # parent hash is valid only inside the per-iteration worker wrapper.
    os.environ.pop(_PARENT_ARCHITECTURE_HASH_ENV, None)
    eligibility_threshold = 0.0 if args.engineering_pilot else 0.99
    os.environ["DISCOVERY_ELIGIBILITY_THRESHOLD"] = str(eligibility_threshold)
    if args.engineering_pilot:
        os.environ.pop("DISCOVERY_SCIENTIFIC_DECISION_RECORD", None)

    config = load_config(agent_dir / "config.yaml")
    if (
        config.diff_based_evolution
        or config.language != "json"
        or config.file_suffix != ".json"
    ):
        raise SystemExit(
            "OpenEvolve candidates must use strict full-document JSON IR "
            "proposals (diff_based_evolution=false, language=json, "
            "file_suffix=.json)"
        )
    try:
        generation = GPT56SolProfile.resolve(
            model=str(model_name),
            seed=args.seed,
            default_reasoning_effort=str(config.llm.reasoning_effort),
            default_max_completion_tokens=int(config.llm.max_tokens),
            default_timeout_seconds=int(config.llm.timeout),
            default_retries=int(config.llm.retries),
            default_retry_delay_seconds=int(config.llm.retry_delay),
            allow_environment_overrides=not training_profile.scientific,
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
        (
            (agent_dir / "system_prompt.md").read_text(),
            (ROOT / "common" / "prompts" / "architecture_ir_contract.md").read_text(),
        )
    )
    config.prompt.template_dir = str(agent_dir / "templates")
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
        api_base=provider_endpoint.base_url,
        api_key=str(api_key),
    )
    config.llm.models = [model_config]
    config.llm.evaluator_models = [model_config]

    install_validity_first_policy()
    if kind == "semantic":
        from agents.openevolve_semantic.semantic_archive import install_semantic_archive

        install_semantic_archive()

    component_hashes = trusted_component_hashes()
    run_manifest = {
        "run_id": run_id,
        "condition": f"openevolve_{kind}",
        "seed": args.seed,
        "candidate_budget": args.iterations + 1,
        "mutation_budget": args.iterations,
        "proposal_opportunities": args.iterations,
        "maximum_provider_attempts": args.iterations * (generation.retries + 1),
        "candidate_training_budget": args.iterations + 1,
        "initial_program_is_evaluated": True,
        "engineering_pilot": args.engineering_pilot,
        "candidate_format": "architecture_ir_json",
        "proposal_format": "strict_full_document_json",
        "generated_python_execution": False,
        "containment_bypass": False,
        "generator": {
            **generation.manifest_fields(),
            "api_base_configured": bool(api_base),
            **provider_endpoint.manifest_fields(),
        },
        "initial_candidate_hash": file_hash(INITIAL_IR_CANDIDATE),
        "initial_architecture_hash": initial_architecture_hash,
        "architecture_hash_schema": ARCHITECTURE_HASH_SCHEMA,
        "parent_relative_architecture_change_required": True,
        "architecture_deduplication": {
            "scope": "run",
            "identity": "normalized_executable_architecture_hash",
            "duplicate_proposals_train": False,
            "duplicate_proposals_consume_opportunity": True,
        },
        "proposal_terminal_ledger": str(terminal_ledger),
        "evaluator_hash": file_hash(ROOT / "common" / "evaluator.py"),
        "trusted_executable_component_hashes": component_hashes,
        "trusted_component_set_sha256": trusted_component_set_sha256(
            component_hashes
        ),
        "config_hash": file_hash(agent_dir / "config.yaml"),
        "evidence_scope": (
            "exploratory_engineering_pilot"
            if args.engineering_pilot
            else "secondary_native_replication"
        ),
        "authoritative_scientific_evidence": False,
        "eligibility_threshold": eligibility_threshold,
        "limitations": (
            [
                "Engineering smoke profiles are non-authoritative and cannot "
                "support architecture rankings or scientific conclusions."
            ]
            if args.engineering_pilot
            else [
                "Native OpenEvolve results are secondary replication evidence, "
                "not the primary randomized study."
            ]
        ),
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
        "evaluation": {
            "profile": evaluation_plan.profile_name,
            "profile_version": evaluation_plan.profile_version,
            "profile_hash": evaluation_plan.profile_hash,
            "plan_hash": evaluation_plan.plan_hash,
            "case_count": evaluation_plan.case_count,
            "case_source_id": evaluation_plan.case_source_id,
            "case_source_sha256": evaluation_plan.case_source_sha256,
            "scientific": evaluation_plan.scientific,
            "pi_decision_record_id": evaluation_plan.pi_decision_record_id,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )

    controller = OpenEvolve(
        initial_program_path=str(INITIAL_IR_CANDIDATE),
        evaluation_file=str(agent_dir / "evaluator_adapter.py"),
        config=config,
        output_dir=str(output_dir),
    )
    # The adapter consumes this authorization exactly once during the initial
    # program evaluation.  Evolved candidates require a worker-bound parent.
    os.environ[_INITIAL_EVALUATION_AUTH_ENV] = initial_architecture_hash
    previous_iteration_worker = _install_parent_bound_worker()
    try:
        best_program = asyncio.run(controller.run(iterations=args.iterations))
    finally:
        _restore_iteration_worker(previous_iteration_worker)
        os.environ.pop(_INITIAL_EVALUATION_AUTH_ENV, None)
    terminal_summary = _terminal_outcome_summary(terminal_ledger, args.iterations)
    terminal_count = int(terminal_summary["terminal_count"])
    all_requested_terminal = bool(terminal_summary["all_requested_terminal"])
    eligible_best_found = best_program is not None
    completed = all_requested_terminal and eligible_best_found
    if not eligible_best_found:
        failure_stage = "no_eligible_candidate"
    elif not all_requested_terminal:
        failure_stage = "incomplete_proposal_opportunities"
    else:
        failure_stage = ""
    run_result = {
        "run_id": run_id,
        "condition": f"openevolve_{kind}",
        "completed": completed,
        "eligible_best_program_found": eligible_best_found,
        "best_program_id": (
            str(getattr(best_program, "id", "")) if best_program is not None else None
        ),
        "proposal_opportunities_requested": args.iterations,
        "proposal_opportunities_completed": terminal_count,
        "proposal_terminal_iterations": terminal_summary["terminal_iterations"],
        "proposal_terminal_status_counts": terminal_summary[
            "terminal_status_counts"
        ],
        "proposal_accounting_errors": terminal_summary["accounting_errors"],
        "engineering_pilot": args.engineering_pilot,
        "authoritative_scientific_evidence": False,
        "failure_stage": failure_stage,
    }
    (output_dir / "run_result.json").write_text(
        json.dumps(run_result, indent=2, sort_keys=True) + "\n"
    )
    if best_program is None:
        raise SystemExit(
            "OpenEvolve completed without an eligible candidate; no best-program "
            "artifact is valid"
        )
    if not all_requested_terminal:
        raise SystemExit(
            "OpenEvolve stopped before every requested proposal opportunity had "
            f"a terminal outcome ({terminal_count}/{args.iterations})"
        )
