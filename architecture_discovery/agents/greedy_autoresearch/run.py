"""Single-incumbent, declarative-IR architecture discovery controller."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from openai import OpenAI

from architecture_ir import encode_graph_json, validate_ir_candidate_json
from architecture_ir.codec import MAX_IR_JSON_BYTES
from architecture_ir.graph import ARCHITECTURE_HASH_SCHEMA
from common.architecture_dedup import ArchitectureHashRegistry
from common.evaluation_profiles import (
    EVALUATION_PROFILES,
    EvaluationLayer,
    get_evaluation_profile,
    resolve_evaluation_plan,
)
from common.evaluator import (
    SearchEvaluationContext,
    evaluate_candidate,
    file_hash,
    preflight_candidate_evaluation,
    validate_controller_view_binding,
)
from common.gpt56_sol import (
    GPT56SolProfile,
    ProviderEndpoint,
    resolve_provider_endpoint,
)
from common.lineage_schema import CandidateRecord, append_record, text_hash, utc_now
from common.public_evaluation import (
    PUBLIC_LAYER_A_SOURCE_ID,
    PUBLIC_LAYER_A_SOURCE_SHA256,
)
from common.provider_attempts import (
    PROVIDER_ATTEMPT_LEDGER_FILENAME,
    PROVIDER_ATTEMPT_SCHEMA,
    ProviderAttemptLedger,
)
from common.task_adapter import DEFAULT_TASK
from common.trainer import trusted_component_hashes, trusted_component_set_sha256
from common.training_config import PROFILES, TrainingSeedBundle, get_training_profile
from evaluation.records import ControllerSearchView


AGENT_DIR = Path(__file__).resolve().parent
CONDITION = "greedy_autoresearch"
DEFAULT_INITIAL_CANDIDATE = ROOT / "common" / "initial_candidate.ir.json"
DEFAULT_MAX_IR_BYTES = 40_000
_JSON_FENCE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*)\r?\n```[ \t]*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


class PilotPreflightError(RuntimeError):
    """Local configuration or the initial seed cannot support a run."""


class IRProposalError(ValueError):
    """A provider response is not a bounded, valid declarative architecture."""


@dataclass(frozen=True)
class ProposalResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class ProposalProvider(Protocol):
    def preflight(self) -> None: ...

    def generate(self, messages: Sequence[Mapping[str, str]]) -> ProposalResponse: ...

    def manifest_fields(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class EvaluationRequest:
    candidate_path: Path
    training_output_dir: Path
    training_profile: str
    evaluation_profile: str
    evaluation_case_count: int
    training_seed: int
    device: str
    allow_cpu_for_tests: bool
    pi_decision_record_id: str | None
    eligibility_threshold: float
    context: SearchEvaluationContext


CandidateEvaluator = Callable[[EvaluationRequest], ControllerSearchView]


@dataclass(frozen=True)
class RunOptions:
    iterations: int
    seed: int
    output_dir: Path | None
    initial_candidate: Path
    training_profile: str
    evaluation_profile: str
    evaluation_case_count: int | None
    device: str
    allow_cpu_for_tests: bool
    pi_decision_record_id: str | None
    eligibility_threshold: float
    engineering_pilot: bool = False
    max_ir_bytes: int = DEFAULT_MAX_IR_BYTES
    accept_valid_plateau_moves: bool = True


class OpenAIProposalProvider:
    """GPT proposal provider. It returns text but never executes that text."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: ProviderEndpoint,
        generation: GPT56SolProfile,
    ) -> None:
        if generation.retries != 0 or generation.retry_delay_seconds != 0:
            raise PilotPreflightError(
                "provider attempts are single-shot; retries and retry delay must be zero"
            )
        self.endpoint = endpoint
        self.api_base = endpoint.base_url
        self.generation = generation
        self._attempt_ledger: ProviderAttemptLedger | None = None
        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint.base_url,
            timeout=generation.timeout_seconds,
            max_retries=0,
        )

    def preflight(self) -> None:
        if not self.api_base.strip():
            raise PilotPreflightError("provider API base is empty")
        if not self.generation.model.strip():
            raise PilotPreflightError("provider model is empty")

    def bind_attempt_ledger(
        self,
        output_dir: Path,
        *,
        run_id: str,
        action: str,
    ) -> None:
        """Create the run's immutable attempt ledger before any request."""

        if self._attempt_ledger is not None:
            raise PilotPreflightError("provider attempt ledger is already bound")
        self._attempt_ledger = ProviderAttemptLedger.create(
            output_dir / PROVIDER_ATTEMPT_LEDGER_FILENAME,
            harness=CONDITION,
            action=action,
            controller_run_id=run_id,
            api_endpoint=self.endpoint.base_url,
            model=self.generation.model,
        )

    def generate(self, messages: Sequence[Mapping[str, str]]) -> ProposalResponse:
        if self._attempt_ledger is None:
            raise PilotPreflightError(
                "provider attempt ledger must be bound before generation"
            )
        request = self.generation.chat_completion_request(messages)
        response = self._attempt_ledger.record_call(
            request,
            lambda: self.client.chat.completions.create(**request),
        )
        usage = response.usage
        return ProposalResponse(
            text=response.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=(
                getattr(usage, "completion_tokens", 0) if usage else 0
            ),
        )

    def manifest_fields(self) -> Mapping[str, object]:
        return {
            **self.generation.manifest_fields(),
            "api_base_configured": True,
            **self.endpoint.manifest_fields(),
        }


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _environment_positive_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return _positive_int(value)
    except argparse.ArgumentTypeError as error:
        raise SystemExit(f"invalid {name}: {error}") from error


def _validate_fresh_output(path: Path) -> None:
    raw = path.expanduser()
    if raw.is_symlink():
        raise FileExistsError(f"output directory may not be a symlink: {raw}")
    if raw.exists() and (not raw.is_dir() or any(raw.iterdir())):
        raise FileExistsError(f"output directory is not fresh: {raw}")


def _prepare_fresh_output(path: Path) -> Path:
    _validate_fresh_output(path)
    raw = path.expanduser()
    raw.mkdir(parents=True, exist_ok=True)
    return raw.resolve()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _validated_ir_text(
    text: str,
    *,
    max_ir_bytes: int,
    require_hypothesis: bool,
) -> tuple[str, str]:
    """Decode, statically validate, and canonicalize an untrusted proposal."""

    if not isinstance(text, str):
        raise IRProposalError("architecture proposal must be text")
    raw_size = len(text.encode("utf-8"))
    if raw_size > max_ir_bytes:
        raise IRProposalError(
            f"architecture proposal exceeds {max_ir_bytes} byte controller limit"
        )
    stripped = text.strip()
    match = _JSON_FENCE.fullmatch(stripped)
    if stripped.startswith("```"):
        if match is None:
            raise IRProposalError(
                "fenced architecture proposal must contain exactly one JSON block"
            )
        stripped = match.group("body").strip()
    elif "```" in stripped:
        raise IRProposalError("architecture proposal contains a malformed JSON fence")

    encoded_size = len(stripped.encode("utf-8"))
    if encoded_size > max_ir_bytes:
        raise IRProposalError(
            f"architecture proposal exceeds {max_ir_bytes} byte controller limit"
        )
    if encoded_size > MAX_IR_JSON_BYTES:
        raise IRProposalError(
            f"architecture proposal exceeds {MAX_IR_JSON_BYTES} byte codec limit"
        )
    validation = validate_ir_candidate_json(stripped)
    if not validation.valid:
        issue_codes = ", ".join(issue.code for issue in validation.issues[:8])
        raise IRProposalError(
            "invalid architecture IR; interpreter validation failed: " + issue_codes
        )
    if validation.graph is None:
        raise IRProposalError("validated architecture IR is missing its graph")
    graph = validation.graph
    canonical = encode_graph_json(graph)
    if len(canonical.encode("utf-8")) > max_ir_bytes:
        raise IRProposalError(
            f"canonical architecture exceeds {max_ir_bytes} byte controller limit"
        )
    raw_hypothesis = graph.metadata.get("mechanism_hypothesis", "")
    if raw_hypothesis is None:
        raw_hypothesis = ""
    if not isinstance(raw_hypothesis, str):
        raise IRProposalError("metadata.mechanism_hypothesis must be a string")
    hypothesis = raw_hypothesis.strip()
    if require_hypothesis and not hypothesis:
        raise IRProposalError(
            "provider IR must declare metadata.mechanism_hypothesis"
        )
    return canonical, hypothesis


def _load_initial_ir(path: Path, *, max_ir_bytes: int) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PilotPreflightError(f"could not read initial IR: {error}") from error
    canonical, _ = _validated_ir_text(
        text,
        max_ir_bytes=max_ir_bytes,
        require_hypothesis=False,
    )
    return canonical


def _architecture_hash(ir_text: str) -> str:
    validation = validate_ir_candidate_json(ir_text)
    if not validation.valid or validation.graph is None:
        raise IRProposalError("cannot fingerprint invalid architecture IR")
    return validation.graph.architecture_hash


def _lineage_evaluation(view: ControllerSearchView) -> dict[str, object]:
    payload = dict(view.as_dict())
    payload["evaluation_run_id"] = payload.pop("run_id")
    payload["evaluation_candidate_id"] = payload.pop("candidate_id")
    return payload


def _load_prompt_protocol() -> tuple[str, dict[str, str], dict[str, object]]:
    sources = (
        ("shared_system", ROOT / "common" / "prompts" / "shared_system.md"),
        ("shared_task", ROOT / "common" / "prompts" / "shared_task.md"),
        (
            "architecture_ir_contract",
            ROOT / "common" / "prompts" / "architecture_ir_contract.md",
        ),
        ("greedy_autoresearch_program", AGENT_DIR / "program.md"),
    )
    contents: dict[str, str] = {}
    components: list[dict[str, str]] = []
    for name, path in sources:
        content = path.read_text(encoding="utf-8")
        contents[name] = content
        components.append(
            {
                "name": name,
                "source_path": str(path.relative_to(ROOT)),
                "sha256": text_hash(content),
            }
        )
    system_prompt = "\n\n".join(contents[name] for name, _ in sources)
    return system_prompt, contents, {
        "components": components,
        "combined_system_prompt_sha256": text_hash(system_prompt),
        "message_hash": "sha256_canonical_json_v1",
        "snapshot_directory": "prompt_snapshot",
    }


def _snapshot_prompt_protocol(
    output_dir: Path,
    *,
    system_prompt: str,
    components: Mapping[str, str],
) -> None:
    destination = output_dir / "prompt_snapshot"
    destination.mkdir()
    for name, content in components.items():
        (destination / f"{name}.md").write_text(content, encoding="utf-8")
    (destination / "combined_system_prompt.md").write_text(
        system_prompt,
        encoding="utf-8",
    )


def _canonical_messages_hash(messages: Sequence[Mapping[str, str]]) -> str:
    canonical = json.dumps(
        [dict(message) for message in messages],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return text_hash(canonical)


def _validate_options(
    options: RunOptions,
    *,
    allow_injected_cpu_test: bool = False,
) -> tuple[object, object, int]:
    if options.iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if options.max_ir_bytes <= 0 or options.max_ir_bytes > MAX_IR_JSON_BYTES:
        raise ValueError(f"max_ir_bytes must be in [1, {MAX_IR_JSON_BYTES}]")
    if not 0.0 <= options.eligibility_threshold <= 1.0:
        raise ValueError("eligibility_threshold must be in [0, 1]")
    training = get_training_profile(options.training_profile)
    evaluation = get_evaluation_profile(options.evaluation_profile)
    if training.scientific != evaluation.scientific:
        raise ValueError(
            "training and evaluation profiles must both be scientific or both be engineering"
        )
    if options.engineering_pilot:
        if (
            training.name != "smoke_train_cuda_v2"
            or evaluation.name != "smoke_eval_v1"
        ):
            raise ValueError(
                "engineering pilot requires smoke_train_cuda_v2 and smoke_eval_v1"
            )
        if options.pi_decision_record_id is not None:
            raise ValueError("engineering pilot cannot claim a PI decision record")
        if options.eligibility_threshold != 0.0:
            raise ValueError(
                "engineering pilot eligibility_threshold must be 0.0 "
                "(mechanics-only transformer-validity selection)"
            )
    elif not training.scientific:
        raise ValueError(
            "engineering profiles require the explicit --engineering-pilot mode"
        )
    if options.device not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be 'cuda', 'mps', or engineering-test 'cpu'")
    if (
        options.engineering_pilot
        and options.device != training.device_requirement
        and not allow_injected_cpu_test
    ):
        raise ValueError(
            "provider-backed engineering pilots require device='cuda'; CPU is "
            "reserved for explicit injected unit-test evaluators"
        )
    if training.scientific and options.device != training.device_requirement:
        raise ValueError(
            f"scientific training requires device={training.device_requirement!r}"
        )
    if options.device == "cpu" and not options.allow_cpu_for_tests:
        raise ValueError("CPU requires allow_cpu_for_tests")
    if training.scientific and options.allow_cpu_for_tests:
        raise ValueError("scientific training cannot enable the CPU test flag")
    count = (
        evaluation.default_case_count
        if options.evaluation_case_count is None
        else options.evaluation_case_count
    )
    plan = resolve_evaluation_plan(
        evaluation.name,
        layer=EvaluationLayer.SEARCH,
        case_source_id=PUBLIC_LAYER_A_SOURCE_ID,
        case_source_sha256=PUBLIC_LAYER_A_SOURCE_SHA256,
        case_count=count,
        pi_decision_record_id=(
            options.pi_decision_record_id if evaluation.scientific else None
        ),
    )
    return training, plan, plan.case_count


def _default_evaluator(request: EvaluationRequest) -> ControllerSearchView:
    evaluation = evaluate_candidate(
        request.candidate_path,
        training_profile=request.training_profile,
        training_seed=request.training_seed,
        training_output_dir=request.training_output_dir,
        device=request.device,
        allow_cpu_for_tests=request.allow_cpu_for_tests,
        evaluation_profile=request.evaluation_profile,
        evaluation_case_count=request.evaluation_case_count,
        pi_decision_record_id=request.pi_decision_record_id,
        eligibility_threshold=request.eligibility_threshold,
        context=request.context,
    )
    return evaluation.controller_view()


def _preflight_default_evaluator(
    options: RunOptions,
    *,
    planned_output: Path | None = None,
) -> dict[str, object]:
    training, plan, case_count = _validate_options(options)
    initial_candidate = options.initial_candidate.resolve()
    if not initial_candidate.is_file():
        raise FileNotFoundError(f"initial candidate does not exist: {initial_candidate}")
    _load_initial_ir(initial_candidate, max_ir_bytes=options.max_ir_bytes)
    output = planned_output or options.output_dir
    if output is not None:
        _validate_fresh_output(output)
        resolved_output = output.expanduser().resolve()
    else:
        resolved_output = (
            ROOT
            / "outputs"
            / "native_replications"
            / f".greedy-autoresearch-preflight-seed-{options.seed}"
        )
    _load_prompt_protocol()
    candidate_preflight = preflight_candidate_evaluation(
        initial_candidate,
        training_profile=training.name,
        training_seed=options.seed,
        training_output_dir=resolved_output / "candidate_training" / "0000_seed",
        device=options.device,
        allow_cpu_for_tests=options.allow_cpu_for_tests,
        evaluation_profile=plan.profile_name,
        evaluation_case_count=case_count,
        pi_decision_record_id=options.pi_decision_record_id,
    )
    return {
        "candidate_evaluation": candidate_preflight,
        "candidate_boundary": {
            "kind": "trusted_declarative_architecture_ir",
            "schema_name": "architecture_tensor_graph",
            "schema_version": "1.0",
            "arbitrary_python_executed": False,
        },
    }


def _prompt_for_incumbent(
    *,
    system_prompt: str,
    incumbent_ir: str,
    incumbent_score: float,
    opportunity: int,
) -> list[dict[str, str]]:
    user_prompt = (
        f"Proposal opportunity {opportunity}.\n"
        f"Current public search score: {incumbent_score:.6f}.\n\n"
        "Return exactly one complete replacement architecture IR JSON object. "
        "Put one testable hypothesis in metadata.mechanism_hypothesis. Do not "
        "return Python, a diff, markdown prose, or executable content. A single "
        "json fence is allowed. Parameter count is metadata only.\n\n"
        f"Current architecture IR:\n```json\n{incumbent_ir}\n```"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _failure_evaluation(stage: str, error: BaseException | str) -> dict[str, object]:
    detail = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else error
    return {
        "execution_ok": False,
        "transformer_valid": False,
        "public_accuracy": 0.0,
        "search_score": 0.0,
        "eligible_for_parent": False,
        "failure_stage": stage,
        "infrastructure_failure": stage in {
            "provider_failure",
            "evaluator_binding",
            "evaluator_exception",
            "seed_evaluation",
        },
        "error": detail[:1_000],
    }


def _provider_error(provider: ProposalProvider, error: BaseException) -> str:
    if isinstance(provider, OpenAIProposalProvider):
        return (
            f"{type(error).__name__}: provider request failed after configured "
            "retries; raw provider exception text was not persisted"
        )
    return f"{type(error).__name__}: {error}"[:1_000]


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def run_greedy_autoresearch(
    options: RunOptions,
    *,
    provider: ProposalProvider,
    evaluator: CandidateEvaluator = _default_evaluator,
) -> dict[str, object]:
    """Run one IR-only greedy replication with exactly one call per opportunity."""

    injected_controller_test = evaluator is not _default_evaluator
    if injected_controller_test and not (
        getattr(evaluator, "controller_only_test_double", False) is True
        and not isinstance(provider, OpenAIProposalProvider)
    ):
        raise PilotPreflightError(
            "an injected evaluator is permitted only with an explicit "
            "controller-only test double and a non-production provider"
        )
    training, evaluation_plan, evaluation_case_count = _validate_options(
        options,
        allow_injected_cpu_test=injected_controller_test,
    )
    initial_candidate = options.initial_candidate.resolve()
    if not initial_candidate.is_file():
        raise FileNotFoundError(f"initial candidate does not exist: {initial_candidate}")
    initial_ir = _load_initial_ir(initial_candidate, max_ir_bytes=options.max_ir_bytes)
    initial_architecture_hash = _architecture_hash(initial_ir)
    run_id = f"greedy-autoresearch-seed-{options.seed}-{uuid.uuid4().hex[:8]}"
    requested_output = options.output_dir or (
        ROOT / "outputs" / "native_replications" / run_id
    )
    _validate_fresh_output(requested_output)
    system_prompt, prompt_components, prompt_manifest = _load_prompt_protocol()

    default_preflight = None
    if not injected_controller_test:
        default_preflight = _preflight_default_evaluator(
            options,
            planned_output=requested_output,
        )
    provider.preflight()

    output_dir = _prepare_fresh_output(requested_output)
    if isinstance(provider, OpenAIProposalProvider):
        provider.bind_attempt_ledger(
            output_dir,
            run_id=run_id,
            action=(
                "one_opportunity_engineering_canary"
                if options.engineering_pilot
                else "scientific_replication"
            ),
        )
    artifacts = output_dir / "artifacts"
    artifacts.mkdir()
    training_root = output_dir / "candidate_training"
    ledger = output_dir / "lineage.jsonl"
    architecture_registry = ArchitectureHashRegistry(
        output_dir / "architecture_hash_registry"
    )
    if not architecture_registry.claim(initial_architecture_hash):
        raise PilotPreflightError("fresh architecture registry rejected the initial seed")
    _snapshot_prompt_protocol(
        output_dir,
        system_prompt=system_prompt,
        components=prompt_components,
    )
    incumbent_path = output_dir / "incumbent.ir.json"
    incumbent_path.write_text(initial_ir, encoding="utf-8")
    seed_id = file_hash(incumbent_path)
    seeds = TrainingSeedBundle.from_run_seed(options.seed)
    context = SearchEvaluationContext(
        study_id="native-replication",
        block_id="native-greedy-autoresearch",
        run_id=run_id,
        condition_id="native-greedy-autoresearch",
    )
    manifest_evaluation = (
        default_preflight["candidate_evaluation"]["evaluation"]
        if default_preflight is not None
        else {
            "profile": evaluation_plan.profile_name,
            "profile_version": evaluation_plan.profile_version,
            "profile_hash": evaluation_plan.profile_hash,
            "plan_hash": evaluation_plan.plan_hash,
            "case_count": evaluation_case_count,
            "case_source_id": evaluation_plan.case_source_id,
            "case_source_sha256": evaluation_plan.case_source_sha256,
            "scientific": evaluation_plan.scientific,
            "synthetic": evaluation_plan.synthetic,
            "controller_visible": evaluation_plan.controller_visible,
            "sealed": evaluation_plan.sealed,
            "pi_decision_record_id": evaluation_plan.pi_decision_record_id,
        }
    )
    component_hashes = trusted_component_hashes()
    manifest = {
        "run_id": run_id,
        "condition": CONDITION,
        "seed": options.seed,
        "candidate_budget": options.iterations + 1,
        "mutation_budget": options.iterations,
        "maximum_provider_attempts": options.iterations,
        "candidate_training_budget": options.iterations + 1,
        "initial_candidate_is_evaluated": True,
        "candidate_format": "architecture_tensor_graph@1.0",
        "max_ir_bytes": options.max_ir_bytes,
        "run_mode": (
            "engineering_pilot" if options.engineering_pilot else "scientific_replication"
        ),
        "exploratory_only": options.engineering_pilot,
        "selection_semantics": (
            "mechanics_only_transformer_validity"
            if options.engineering_pilot
            else "frozen_scientific_parent_eligibility"
        ),
        "greedy_retention": {
            "requires_parent_eligibility": True,
            "rejects_search_score_regressions": True,
            "accept_valid_plateau_moves": options.accept_valid_plateau_moves,
        },
        "generator": dict(provider.manifest_fields()),
        "initial_candidate_hash": seed_id,
        "initial_architecture_hash": initial_architecture_hash,
        "architecture_hash_schema": ARCHITECTURE_HASH_SCHEMA,
        "architecture_deduplication": {
            "scope": "run",
            "identity": "normalized_executable_architecture_hash",
            "duplicate_proposals_train": False,
            "duplicate_proposals_consume_opportunity": True,
        },
        "evaluator_hash": file_hash(ROOT / "common" / "evaluator.py"),
        "trusted_executable_component_hashes": component_hashes,
        "trusted_component_set_sha256": trusted_component_set_sha256(
            component_hashes
        ),
        "config_hash": file_hash(AGENT_DIR / "config.yaml"),
        "prompt_protocol": prompt_manifest,
        "training": {
            "profile": training.name,
            "profile_version": training.version,
            "profile_hash": training.profile_hash,
            "task_adapter": DEFAULT_TASK.version,
            "task_adapter_hash": DEFAULT_TASK.config_hash,
            "seed_bundle": asdict(seeds),
            "seed_bundle_hash": seeds.bundle_hash,
            "device": options.device,
            "allow_cpu_for_tests": options.allow_cpu_for_tests,
        },
        "evaluation": manifest_evaluation,
        "preflight": default_preflight or {
            "mode": "injected_controller_test_double",
            "generated_candidate_execution_by_controller": False,
            "explicit_controller_only_test_double": True,
            "candidate_boundary": "strict_validated_architecture_ir",
        },
        "evidence_scope": "secondary_native_replication",
        "authoritative_scientific_evidence": False,
    }
    if isinstance(provider, OpenAIProposalProvider):
        manifest.update(
            {
                "provider_attempt_ledger": PROVIDER_ATTEMPT_LEDGER_FILENAME,
                "provider_attempt_schema": PROVIDER_ATTEMPT_SCHEMA,
            }
        )
    if training.version == "2":
        manifest.update(
            {"schema_name": "ControllerRunManifest", "schema_version": "2.0"}
        )
    _atomic_json(output_dir / "run_manifest.json", manifest)

    seed_lineage_id = "lineage-" + text_hash(f"{run_id}|lineage|0")
    seed_started = utc_now()
    seed_request = EvaluationRequest(
        candidate_path=incumbent_path,
        training_output_dir=training_root / f"0000_{seed_id[:12]}",
        training_profile=training.name,
        evaluation_profile=evaluation_plan.profile_name,
        evaluation_case_count=evaluation_case_count,
        training_seed=options.seed,
        device=options.device,
        allow_cpu_for_tests=options.allow_cpu_for_tests,
        pi_decision_record_id=options.pi_decision_record_id,
        eligibility_threshold=options.eligibility_threshold,
        context=context,
    )
    try:
        seed_view = evaluator(seed_request)
        validate_controller_view_binding(
            seed_view,
            candidate_source_hash=seed_id,
            context=context,
        )
    except Exception as error:
        append_record(
            ledger,
            CandidateRecord(
                run_id=run_id,
                condition=CONDITION,
                seed=options.seed,
                candidate_id=seed_id,
                parent_id=None,
                lineage_record_id=seed_lineage_id,
                proposal_text="checked-in initial architecture IR",
                code_hash=seed_id,
                proposal_timestamp=seed_started,
                completion_timestamp=utc_now(),
                evaluation={
                    "proposal_opportunity": 0,
                    **_failure_evaluation("seed_evaluation", error),
                },
                retention_decision="seed_rejected",
            ),
        )
        raise PilotPreflightError(
            "initial candidate evaluation failed before any provider call: "
            f"{type(error).__name__}: {error}"
        ) from error
    seed_evaluation = _lineage_evaluation(seed_view)
    seed_evaluation.update(
        {"proposal_opportunity": 0, "candidate_role": "initial_seed"}
    )
    append_record(
        ledger,
        CandidateRecord(
            run_id=run_id,
            condition=CONDITION,
            seed=options.seed,
            candidate_id=seed_id,
            parent_id=None,
            lineage_record_id=seed_lineage_id,
            proposal_text="checked-in initial architecture IR",
            mechanism_hypothesis="shared starting architecture",
            code_hash=seed_id,
            proposal_timestamp=seed_started,
            completion_timestamp=utc_now(),
            evaluation=seed_evaluation,
            retention_decision="seed_parent" if seed_view.eligible_for_parent else "seed_rejected",
        ),
    )
    if not seed_view.eligible_for_parent:
        raise PilotPreflightError(
            "initial candidate is not eligible to be a parent; no provider call was made"
        )

    accepted_lineage = output_dir / "accepted_lineage"
    accepted_lineage.mkdir()
    _git(accepted_lineage, "init", "-q")
    _git(accepted_lineage, "config", "user.name", "Architecture Discovery Controller")
    _git(accepted_lineage, "config", "user.email", "discovery-controller@localhost")
    shutil.copy2(incumbent_path, accepted_lineage / "candidate.ir.json")
    _git(accepted_lineage, "add", "candidate.ir.json")
    _git(accepted_lineage, "commit", "-q", "-m", "accept initial validated IR")

    parent_id = seed_id
    parent_lineage_id = seed_lineage_id
    incumbent_score = float(seed_view.search_score)
    incumbent_architecture_hash = initial_architecture_hash
    completed = 0
    for opportunity in range(1, options.iterations + 1):
        proposal_time = utc_now()
        proposal_id = "proposal-" + text_hash(f"{run_id}|proposal|{opportunity}")
        lineage_id = "lineage-" + text_hash(f"{run_id}|lineage|{opportunity}")
        incumbent_ir = incumbent_path.read_text(encoding="utf-8")
        messages = _prompt_for_incumbent(
            system_prompt=system_prompt,
            incumbent_ir=incumbent_ir,
            incumbent_score=incumbent_score,
            opportunity=opportunity,
        )
        prompt_hash = _canonical_messages_hash(messages)
        artifact_base = artifacts / f"{opportunity:04d}"
        artifact_base.with_suffix(".prompt.md").write_text(
            messages[-1]["content"],
            encoding="utf-8",
        )
        artifact_base.with_suffix(".messages.json").write_text(
            json.dumps(messages, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            response = provider.generate(messages)
        except Exception as error:
            safe_error = _provider_error(provider, error)
            append_record(
                ledger,
                CandidateRecord(
                    run_id=run_id,
                    condition=CONDITION,
                    seed=options.seed,
                    candidate_id="failed-" + text_hash(f"{proposal_id}|provider"),
                    parent_id=parent_id,
                    lineage_record_id=lineage_id,
                    proposal_id=proposal_id,
                    parent_lineage_record_id=parent_lineage_id,
                    prompt_hash=prompt_hash,
                    proposal_timestamp=proposal_time,
                    completion_timestamp=utc_now(),
                    evaluation={
                        "proposal_opportunity": opportunity,
                        **_failure_evaluation("provider_failure", safe_error),
                    },
                    retention_decision="crash",
                    rollback_target=parent_id,
                ),
            )
            completed += 1
            continue

        response_text = response.text
        artifact_base.with_suffix(".response.txt").write_text(
            response_text,
            encoding="utf-8",
        )
        base_record = {
            "run_id": run_id,
            "condition": CONDITION,
            "seed": options.seed,
            "parent_id": parent_id,
            "lineage_record_id": lineage_id,
            "proposal_id": proposal_id,
            "parent_lineage_record_id": parent_lineage_id,
            "prompt_hash": prompt_hash,
            "response_hash": text_hash(response_text),
            "proposal_timestamp": proposal_time,
            "rollback_target": parent_id,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }
        try:
            child_ir, hypothesis = _validated_ir_text(
                response_text,
                max_ir_bytes=options.max_ir_bytes,
                require_hypothesis=True,
            )
        except IRProposalError as error:
            append_record(
                ledger,
                CandidateRecord(
                    **base_record,
                    candidate_id="failed-" + text_hash(f"{proposal_id}|ir_validation"),
                    proposal_text=str(error),
                    completion_timestamp=utc_now(),
                    evaluation={
                        "proposal_opportunity": opportunity,
                        **_failure_evaluation("ir_validation", error),
                    },
                    retention_decision="crash",
                ),
            )
            completed += 1
            continue
        child_architecture_hash = _architecture_hash(child_ir)
        if child_architecture_hash == incumbent_architecture_hash:
            append_record(
                ledger,
                CandidateRecord(
                    **base_record,
                    candidate_id="failed-" + text_hash(f"{proposal_id}|no_change"),
                    proposal_text=hypothesis,
                    mechanism_hypothesis=hypothesis,
                    completion_timestamp=utc_now(),
                    code_hash=parent_id,
                    evaluation={
                        "proposal_opportunity": opportunity,
                        **_failure_evaluation(
                            "mutation_no_change",
                            "replacement IR changes only document identity or metadata, "
                            "not executable architecture",
                        ),
                    },
                    retention_decision="crash",
                ),
            )
            completed += 1
            continue

        candidate_id = text_hash(child_ir)
        if not architecture_registry.claim(child_architecture_hash):
            append_record(
                ledger,
                CandidateRecord(
                    **base_record,
                    candidate_id="failed-"
                    + text_hash(f"{proposal_id}|duplicate_architecture"),
                    proposal_text=hypothesis,
                    mechanism_hypothesis=hypothesis,
                    completion_timestamp=utc_now(),
                    code_hash=candidate_id,
                    evaluation={
                        "proposal_opportunity": opportunity,
                        "duplicate_architecture_hash": child_architecture_hash,
                        **_failure_evaluation(
                            "duplicate_architecture",
                            "normalized executable architecture was already proposed "
                            "or evaluated in this run",
                        ),
                    },
                    retention_decision="duplicate_rejected",
                ),
            )
            completed += 1
            continue

        child_path = artifacts / f"{opportunity:04d}_{candidate_id[:12]}.ir.json"
        child_path.write_text(child_ir, encoding="utf-8")
        request = EvaluationRequest(
            candidate_path=child_path,
            training_output_dir=training_root / f"{opportunity:04d}_{candidate_id[:12]}",
            training_profile=training.name,
            evaluation_profile=evaluation_plan.profile_name,
            evaluation_case_count=evaluation_case_count,
            training_seed=options.seed,
            device=options.device,
            allow_cpu_for_tests=options.allow_cpu_for_tests,
            pi_decision_record_id=options.pi_decision_record_id,
            eligibility_threshold=options.eligibility_threshold,
            context=context,
        )
        try:
            if not injected_controller_test:
                preflight_candidate_evaluation(
                    child_path,
                    training_profile=training.name,
                    training_seed=options.seed,
                    training_output_dir=request.training_output_dir,
                    device=options.device,
                    allow_cpu_for_tests=options.allow_cpu_for_tests,
                    evaluation_profile=evaluation_plan.profile_name,
                    evaluation_case_count=evaluation_case_count,
                    pi_decision_record_id=options.pi_decision_record_id,
                )
            view = evaluator(request)
            validate_controller_view_binding(
                view,
                candidate_source_hash=candidate_id,
                context=context,
            )
        except Exception as error:
            stage = (
                "evaluator_binding"
                if type(error).__name__ == "EvaluationBindingError"
                else "evaluator_exception"
            )
            append_record(
                ledger,
                CandidateRecord(
                    **base_record,
                    candidate_id=candidate_id,
                    proposal_text=hypothesis,
                    mechanism_hypothesis=hypothesis,
                    completion_timestamp=utc_now(),
                    code_hash=candidate_id,
                    evaluation={
                        "proposal_opportunity": opportunity,
                        **_failure_evaluation(stage, error),
                    },
                    retention_decision="crash",
                ),
            )
            completed += 1
            continue

        child_score = float(view.search_score)
        score_improved = child_score > incumbent_score
        score_tied = child_score == incumbent_score
        accepted = bool(
            view.eligible_for_parent
            and (
                score_improved
                or (options.accept_valid_plateau_moves and score_tied)
            )
        )
        if accepted:
            decision = "accept"
        elif not view.eligible_for_parent:
            decision = "reject"
        elif score_tied:
            decision = "reject_score_tie"
        else:
            decision = "reject_score_regression"
        rollback = None if accepted else parent_id
        evaluation = _lineage_evaluation(view)
        evaluation.update(
            {"proposal_opportunity": opportunity, "candidate_role": "proposed_ir"}
        )
        append_record(
            ledger,
            CandidateRecord(
                **{
                    **base_record,
                    "rollback_target": rollback,
                },
                candidate_id=candidate_id,
                proposal_text=hypothesis,
                mechanism_hypothesis=hypothesis,
                completion_timestamp=utc_now(),
                code_hash=candidate_id,
                evaluation=evaluation,
                retention_decision=decision,
            ),
        )
        if accepted:
            shutil.copy2(child_path, incumbent_path)
            shutil.copy2(child_path, accepted_lineage / "candidate.ir.json")
            _git(accepted_lineage, "add", "candidate.ir.json")
            _git(
                accepted_lineage,
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                f"accept opportunity {opportunity}: {candidate_id[:12]}",
            )
            parent_id = candidate_id
            parent_lineage_id = lineage_id
            incumbent_score = child_score
            incumbent_architecture_hash = child_architecture_hash
        completed += 1

    summary = {
        "run_id": run_id,
        "condition": CONDITION,
        "proposal_opportunities_requested": options.iterations,
        "proposal_opportunities_terminal": completed,
        "lineage_path": (
            str(ledger) if training.version == "1" else ledger.name
        ),
        "incumbent_path": (
            str(incumbent_path)
            if training.version == "1"
            else incumbent_path.name
        ),
        "authoritative_scientific_evidence": False,
    }
    if training.version == "2":
        summary.update(
            {"schema_name": "ControllerRunSummary", "schema_version": "2.0"}
        )
    _atomic_json(output_dir / "run_summary.json", summary)
    return summary


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


def _provider_from_environment(
    config: Mapping[str, object],
    seed: int,
    *,
    scientific: bool,
) -> ProposalProvider:
    api_key, api_base, model = _provider_values()
    try:
        endpoint = resolve_provider_endpoint(api_base, scientific=scientific)
        generation = GPT56SolProfile.resolve(
            model=model,
            seed=seed,
            default_reasoning_effort=str(config["reasoning_effort"]),
            default_max_completion_tokens=int(config["max_tokens"]),
            default_timeout_seconds=int(config["timeout_seconds"]),
            default_retries=int(config["retries"]),
            default_retry_delay_seconds=int(config["retry_delay_seconds"]),
            allow_environment_overrides=not scientific,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return OpenAIProposalProvider(
        api_key=api_key,
        endpoint=endpoint,
        generation=generation,
    )


def main() -> None:
    config = yaml.safe_load((AGENT_DIR / "config.yaml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(
        description="Run Greedy Autoresearch over trusted declarative architecture IR."
    )
    parser.add_argument("--iterations", type=_positive_int, default=config["iterations"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--engineering-pilot",
        action="store_true",
        help=(
            "run an exploratory IR-only CUDA canary; forces smoke_train_cuda_v2 and "
            "smoke_eval_v1 and never creates authoritative scientific evidence"
        ),
    )
    parser.add_argument("--training-profile", choices=tuple(sorted(PROFILES)))
    parser.add_argument(
        "--evaluation-profile",
        choices=tuple(sorted(EVALUATION_PROFILES)),
    )
    parser.add_argument("--evaluation-cases", type=_positive_int)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    parser.add_argument("--allow-cpu-for-tests", action="store_true")
    parser.add_argument("--scientific-decision-record")
    arguments = parser.parse_args()

    if arguments.engineering_pilot and arguments.training_profile not in {
        None,
        "smoke_train_cuda_v2",
    }:
        parser.error(
            "--engineering-pilot forces --training-profile smoke_train_cuda_v2"
        )
    if arguments.engineering_pilot and arguments.evaluation_profile not in {
        None,
        "smoke_eval_v1",
    }:
        parser.error("--engineering-pilot forces --evaluation-profile smoke_eval_v1")
    if arguments.engineering_pilot and arguments.scientific_decision_record:
        parser.error("--engineering-pilot cannot claim a scientific decision record")

    configured_candidate = (ROOT / str(config["candidate_path"])).resolve()
    if configured_candidate != DEFAULT_INITIAL_CANDIDATE.resolve():
        parser.error("configured candidate_path differs from the trusted initial IR")
    training_profile = (
        "smoke_train_cuda_v2"
        if arguments.engineering_pilot
        else arguments.training_profile or str(config["training"]["profile"])
    )
    evaluation_profile = (
        "smoke_eval_v1"
        if arguments.engineering_pilot
        else arguments.evaluation_profile or "scientific_layer_a_v1"
    )
    case_count = arguments.evaluation_cases
    if case_count is None and not arguments.engineering_pilot:
        case_count = _environment_positive_int("DISCOVERY_LAYER_A_CASES")
    decision = None if arguments.engineering_pilot else (
        arguments.scientific_decision_record
        or os.environ.get("DISCOVERY_SCIENTIFIC_DECISION_RECORD")
    )
    device = (
        arguments.device
        or os.environ.get("DISCOVERY_TRAIN_DEVICE")
        or str(config["training"]["device"])
    ).lower()
    options = RunOptions(
        iterations=arguments.iterations,
        seed=arguments.seed,
        output_dir=arguments.output_dir,
        initial_candidate=DEFAULT_INITIAL_CANDIDATE,
        training_profile=training_profile,
        evaluation_profile=evaluation_profile,
        evaluation_case_count=case_count,
        device=device,
        allow_cpu_for_tests=(
            arguments.allow_cpu_for_tests
            or bool(config["training"]["allow_cpu_for_tests"])
        ),
        pi_decision_record_id=decision,
        eligibility_threshold=(
            0.0
            if arguments.engineering_pilot
            else float(config["robustness_floor"])
        ),
        engineering_pilot=arguments.engineering_pilot,
        accept_valid_plateau_moves=bool(
            config["acceptance"]["accept_valid_plateau_moves"]
        ),
    )
    try:
        _preflight_default_evaluator(options)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(f"local preflight failed before provider initialization: {error}")
    provider = _provider_from_environment(
        config,
        arguments.seed,
        scientific=get_training_profile(training_profile).scientific,
    )
    try:
        summary = run_greedy_autoresearch(options, provider=provider)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
