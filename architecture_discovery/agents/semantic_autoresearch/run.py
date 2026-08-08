"""Semantic-coverage single-controller Autoresearch.

The controller never imports or executes generated candidate source.  Candidate
execution remains evaluator-owned, and tests can inject a provider and an
evaluator to exercise controller plumbing without network access or arbitrary
code execution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
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

from common.descriptor_schema import CATEGORY_CODES, SEMANTIC_METRIC_NAMES
from common.evaluation_profiles import (
    EvaluationLayer,
    get_evaluation_profile,
    resolve_evaluation_plan,
)
from common.evaluator import (
    SearchEvaluationContext,
    evaluate_candidate,
    file_hash,
    preflight_candidate_evaluation,
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
from common.task_adapter import DEFAULT_TASK
from common.trainer import trusted_component_hashes, trusted_component_set_sha256
from common.training_config import PROFILES, TrainingSeedBundle, get_training_profile
from evaluation.records import SCHEMA_VERSION, ControllerSearchView


AGENT_DIR = Path(__file__).resolve().parent
CONDITION = "semantic_autoresearch"
DEFAULT_INITIAL_CANDIDATE = ROOT / "common" / "initial_candidate.ir.json"
FROZEN_DESCRIPTOR_AXES = tuple(
    SEMANTIC_METRIC_NAMES[axis] for axis in CATEGORY_CODES
)
_METRIC_TO_AXIS = {
    metric: axis for axis, metric in SEMANTIC_METRIC_NAMES.items()
}
CONTROLLER_EVALUATION_PROFILES = (
    "unit_eval_v1",
    "smoke_eval_v1",
    "development_eval_v1",
    "scientific_layer_a_v1",
)


class PilotPreflightError(RuntimeError):
    """The initial seed or local run configuration cannot support a pilot."""


class EvaluationBindingError(RuntimeError):
    """An evaluator response is not bound to the requested candidate and run."""


class IRProposalError(ValueError):
    """A provider response is not a bounded, valid declarative architecture."""


@dataclass(frozen=True)
class ProposalResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class ProposalProvider(Protocol):
    """Minimal injectable provider boundary used by the controller."""

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
    max_ir_bytes: int = 40_000


@dataclass
class ArchiveCandidate:
    candidate_id: str
    lineage_record_id: str
    source_path: Path
    signature: tuple[int, ...]
    search_score: float
    public_accuracy: float
    discovered_opportunity: int
    parent_uses: int = 0


class FrozenSemanticArchive:
    """Categorical coverage archive whose codes are never treated as fitness."""

    def __init__(self, axes: Sequence[str] = FROZEN_DESCRIPTOR_AXES):
        frozen = tuple(axes)
        if frozen != FROZEN_DESCRIPTOR_AXES:
            raise ValueError("semantic archive axes differ from the frozen descriptor order")
        self.axes = frozen
        self._cells: dict[tuple[int, ...], ArchiveCandidate] = {}

    @property
    def coverage(self) -> int:
        return len(self._cells)

    def signature_from(self, view: ControllerSearchView) -> tuple[int, ...]:
        values: dict[str, float] = {}
        for name, raw_value in view.online_descriptor_codes:
            if name in values:
                raise ValueError(f"duplicate semantic descriptor: {name}")
            values[name] = raw_value
        missing = [name for name in self.axes if name not in values]
        if missing:
            raise ValueError("eligible candidate lacks semantic descriptors: " + ", ".join(missing))

        signature: list[int] = []
        for metric in self.axes:
            raw_value = values[metric]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"semantic descriptor {metric} is not numeric")
            numeric = float(raw_value)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"semantic descriptor {metric} is not a finite category code")
            code = int(numeric)
            axis = _METRIC_TO_AXIS[metric]
            if code not in set(CATEGORY_CODES[axis].values()):
                raise ValueError(f"semantic descriptor {metric} has unknown code {code}")
            signature.append(code)
        return tuple(signature)

    def cell_label(self, signature: Sequence[int]) -> str:
        return "|".join(
            f"{metric.removeprefix('semantic_')}={code}"
            for metric, code in zip(self.axes, signature)
        )

    def consider(
        self,
        *,
        candidate_id: str,
        lineage_record_id: str,
        source_path: Path,
        view: ControllerSearchView,
        opportunity: int,
    ) -> tuple[str, str]:
        if not (
            view.execution_ok
            and view.transformer_valid
            and view.eligible_for_parent
        ):
            raise ValueError("only valid, eligible candidates may enter the parent archive")
        signature = self.signature_from(view)
        cell = self.cell_label(signature)
        candidate = ArchiveCandidate(
            candidate_id=candidate_id,
            lineage_record_id=lineage_record_id,
            source_path=source_path,
            signature=signature,
            search_score=view.search_score,
            public_accuracy=view.public_accuracy,
            discovered_opportunity=opportunity,
        )
        incumbent = self._cells.get(signature)
        if incumbent is None:
            self._cells[signature] = candidate
            return "archive_new_cell", cell
        if candidate.public_accuracy > incumbent.public_accuracy:
            candidate.parent_uses = incumbent.parent_uses
            self._cells[signature] = candidate
            return "archive_replace", cell
        return "eligible_cell_incumbent_preserved", cell

    def select_parent(self) -> ArchiveCandidate:
        if not self._cells:
            raise PilotPreflightError("semantic archive contains no eligible parent")
        # Category numbers are deliberately absent from this ordering.  The
        # archive uses coverage frequency, public accuracy, age, and an opaque hash.
        parent = min(
            self._cells.values(),
            key=lambda item: (
                item.parent_uses,
                -item.public_accuracy,
                item.discovered_opportunity,
                item.candidate_id,
            ),
        )
        parent.parent_uses += 1
        return parent

    def to_dict(self) -> dict[str, object]:
        cells = []
        for candidate in sorted(
            self._cells.values(), key=lambda item: item.candidate_id
        ):
            cells.append(
                {
                    "cell": self.cell_label(candidate.signature),
                    "signature": list(candidate.signature),
                    "candidate_id": candidate.candidate_id,
                    "lineage_record_id": candidate.lineage_record_id,
                    "source_path": str(candidate.source_path),
                    "search_score": candidate.search_score,
                    "public_accuracy": candidate.public_accuracy,
                    "discovered_opportunity": candidate.discovered_opportunity,
                    "parent_uses": candidate.parent_uses,
                }
            )
        return {
            "schema_name": "semantic_autoresearch_archive",
            "schema_version": "1",
            "axes": list(self.axes),
            "coverage_cells": len(cells),
            "novelty_role": "exploratory_coverage_tiebreak_only",
            "scientific_novelty_claim": False,
            "cells": cells,
        }


class OpenAIProposalProvider:
    """GPT-5.6 proposal provider; candidate execution is not performed here."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: ProviderEndpoint,
        generation: GPT56SolProfile,
    ):
        self.endpoint = endpoint
        self.api_base = endpoint.base_url
        self.generation = generation
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

    def generate(self, messages: Sequence[Mapping[str, str]]) -> ProposalResponse:
        for attempt in range(self.generation.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    **self.generation.chat_completion_request(messages)
                )
                usage = response.usage
                return ProposalResponse(
                    text=response.choices[0].message.content or "",
                    input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    output_tokens=(
                        getattr(usage, "completion_tokens", 0) if usage else 0
                    ),
                )
            except Exception:
                if attempt >= self.generation.retries:
                    raise
                time.sleep(self.generation.retry_delay_seconds)
        raise RuntimeError("unreachable provider retry state")

    def manifest_fields(self) -> Mapping[str, object]:
        return {
            **self.generation.manifest_fields(),
            "api_base_configured": True,
            **self.endpoint.manifest_fields(),
        }


def _default_evaluator(request: EvaluationRequest) -> ControllerSearchView:
    result = evaluate_candidate(
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
    return result.controller_view()


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


def _validate_fresh_output_path(path: Path) -> None:
    raw_path = path.expanduser()
    if raw_path.is_symlink():
        raise FileExistsError(f"output directory may not be a symlink: {raw_path}")
    if raw_path.exists():
        if not raw_path.is_dir() or any(raw_path.iterdir()):
            raise FileExistsError(f"output directory is not fresh: {raw_path}")


def _prepare_fresh_output(path: Path) -> Path:
    _validate_fresh_output_path(path)
    raw_path = path.expanduser()
    raw_path.mkdir(parents=True, exist_ok=True)
    return raw_path.resolve()


def _sanitized_exception(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:1_000]


def _provider_failure_detail(
    provider: ProposalProvider,
    error: BaseException,
) -> str:
    """Never persist production-provider exception text that may echo secrets."""

    if isinstance(provider, OpenAIProposalProvider):
        return (
            f"{type(error).__name__}: provider request failed after configured "
            "retries; raw provider exception text was not persisted"
        )
    return _sanitized_exception(error)


def _failure_fields(stage: str, error: BaseException | str) -> dict[str, object]:
    detail = _sanitized_exception(error) if isinstance(error, BaseException) else error
    return {
        "execution_ok": False,
        "transformer_valid": False,
        "public_accuracy": 0.0,
        "search_score": 0.0,
        "eligible_for_parent": False,
        "failure_stage": stage,
        "infrastructure_failure": stage in {
            "evaluator_binding",
            "evaluator_exception",
            "provider_failure",
            "seed_evaluation_binding",
            "seed_preflight_exception",
        },
        "error": detail,
    }


def _lineage_evaluation(view: ControllerSearchView) -> dict[str, object]:
    """Keep evaluator identity without overwriting lineage identity fields."""

    payload = dict(view.as_dict())
    payload["evaluation_run_id"] = payload.pop("run_id")
    payload["evaluation_candidate_id"] = payload.pop("candidate_id")
    return payload


def _validate_evaluator_binding(
    view: ControllerSearchView,
    *,
    expected_source_hash: str,
    context: SearchEvaluationContext,
    seen_record_ids: set[str],
) -> None:
    """Reject stale or cross-run evaluator output before it can affect search."""

    if not isinstance(view, ControllerSearchView):
        raise EvaluationBindingError("evaluator did not return ControllerSearchView")
    # Calling as_dict also enforces the controller-field allowlist.
    view.as_dict()
    expected_candidate_id = f"candidate-{expected_source_hash}"
    mismatches: list[str] = []
    if view.schema_name != "search_evaluation" or view.schema_version != SCHEMA_VERSION:
        mismatches.append("search-evaluation schema")
    if view.candidate_id != expected_candidate_id:
        mismatches.append("candidate source hash")
    if view.run_id != context.run_id:
        mismatches.append("run_id")
    if view.condition_id != context.condition_id:
        mismatches.append("condition_id")
    if not view.record_id.strip():
        mismatches.append("record_id")
    elif view.record_id in seen_record_ids:
        mismatches.append("duplicate record_id")
    for field_name in (
        "execution_ok",
        "transformer_valid",
        "eligible_for_parent",
        "infrastructure_failure",
    ):
        if type(getattr(view, field_name)) is not bool:
            mismatches.append(f"{field_name} type")
    for field_name in ("public_accuracy", "search_score"):
        value = getattr(view, field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            mismatches.append(f"{field_name} type")
        elif not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            mismatches.append(f"{field_name} range")
    if view.eligible_for_parent and not (
        view.execution_ok and view.transformer_valid
    ):
        mismatches.append("eligibility invariant")
    if mismatches:
        raise EvaluationBindingError(
            "evaluator result is not bound to this request: "
            + ", ".join(dict.fromkeys(mismatches))
        )
    seen_record_ids.add(view.record_id)


def _lineage_record_id(run_id: str, opportunity: int) -> str:
    return "lineage-" + text_hash(f"{run_id}|lineage|{opportunity}")


def _proposal_id(run_id: str, opportunity: int) -> str:
    return "proposal-" + text_hash(f"{run_id}|proposal|{opportunity}")


def _failed_candidate_id(proposal_id: str, stage: str) -> str:
    return "failed-" + text_hash(f"{proposal_id}|{stage}")


def _canonical_messages_hash(messages: Sequence[Mapping[str, str]]) -> str:
    canonical = json.dumps(
        [dict(message) for message in messages],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return text_hash(canonical)


def _load_prompt_protocol() -> tuple[str, dict[str, str], dict[str, object]]:
    sources = (
        ("shared_system", ROOT / "common" / "prompts" / "shared_system.md"),
        ("shared_task", ROOT / "common" / "prompts" / "shared_task.md"),
        (
            "architecture_ir_contract",
            ROOT / "common" / "prompts" / "architecture_ir_contract.md",
        ),
        ("semantic_autoresearch_program", AGENT_DIR / "program.md"),
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
    manifest = {
        "components": components,
        "combined_system_prompt_sha256": text_hash(system_prompt),
        "message_hash": "sha256_canonical_json_v1",
        "snapshot_directory": "prompt_snapshot",
    }
    return system_prompt, contents, manifest


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
        system_prompt, encoding="utf-8"
    )


_JSON_FENCE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*)\r?\n```[ \t]*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def _validated_ir_text(
    text: str,
    *,
    max_ir_bytes: int,
    require_hypothesis: bool,
) -> tuple[str, str]:
    """Return canonical validated IR and its declared mechanism hypothesis.

    Provider output is data, never source.  Only a bare JSON document or one
    complete JSON fence is accepted; surrounding prose cannot be smuggled into
    an artifact that later stages might interpret differently.
    """

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
        source = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PilotPreflightError(f"could not read initial IR: {error}") from error
    canonical, _ = _validated_ir_text(
        source,
        max_ir_bytes=max_ir_bytes,
        require_hypothesis=False,
    )
    return canonical


def _architecture_hash(ir_text: str) -> str:
    validation = validate_ir_candidate_json(ir_text)
    if not validation.valid or validation.graph is None:
        raise IRProposalError("cannot fingerprint invalid architecture IR")
    return validation.graph.architecture_hash


def _validate_options(
    options: RunOptions,
    *,
    allow_injected_cpu_test: bool = False,
) -> tuple[object, object, int]:
    if options.iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if options.max_ir_bytes <= 0 or options.max_ir_bytes > MAX_IR_JSON_BYTES:
        raise ValueError(
            f"max_ir_bytes must be in [1, {MAX_IR_JSON_BYTES}]"
        )
    if not 0.0 <= options.eligibility_threshold <= 1.0:
        raise ValueError("eligibility_threshold must be in [0, 1]")
    training = get_training_profile(options.training_profile)
    evaluation = get_evaluation_profile(options.evaluation_profile)
    if training.scientific != evaluation.scientific:
        raise ValueError(
            "training and evaluation profiles must both be scientific or both be engineering"
        )
    if options.engineering_pilot:
        if training.name != "smoke_train_v1" or evaluation.name != "smoke_eval_v1":
            raise ValueError(
                "engineering pilot requires smoke_train_v1 and smoke_eval_v1"
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
    if options.device not in {"mps", "cpu"}:
        raise ValueError("device must be 'mps' or explicit engineering-test 'cpu'")
    if (
        options.engineering_pilot
        and options.device != "mps"
        and not allow_injected_cpu_test
    ):
        raise ValueError(
            "provider-backed engineering pilots require device='mps'; CPU is "
            "reserved for explicit injected unit-test evaluators"
        )
    if training.scientific and options.device != "mps":
        raise ValueError("scientific training requires device='mps'")
    if options.device == "cpu" and not options.allow_cpu_for_tests:
        raise ValueError("CPU requires allow_cpu_for_tests")
    if training.scientific and options.allow_cpu_for_tests:
        raise ValueError("scientific training cannot enable the CPU test flag")

    count = (
        evaluation.default_case_count
        if options.evaluation_case_count is None
        else options.evaluation_case_count
    )
    # Resolving the complete plan is the evaluator-configuration preflight.  It
    # happens before provider.generate can make a paid request.
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


def _preflight_default_evaluator(
    options: RunOptions,
    *,
    planned_output: Path | None = None,
) -> dict[str, object]:
    """Complete all non-executing local gates before provider construction."""

    training, evaluation_plan, evaluation_case_count = _validate_options(options)
    initial_candidate = options.initial_candidate.resolve()
    if not initial_candidate.is_file():
        raise FileNotFoundError(f"initial candidate does not exist: {initial_candidate}")
    _load_initial_ir(initial_candidate, max_ir_bytes=options.max_ir_bytes)
    if planned_output is not None:
        _validate_fresh_output_path(planned_output)
        resolved_output = planned_output.expanduser().resolve()
    elif options.output_dir is not None:
        _validate_fresh_output_path(options.output_dir)
        resolved_output = options.output_dir.expanduser().resolve()
    else:
        resolved_output = (
            ROOT
            / "outputs"
            / "native_replications"
            / f".semantic-autoresearch-preflight-seed-{options.seed}"
        )
    # Prompt files are part of generation and must be readable before credentials
    # are consulted or a provider client is initialized.
    _load_prompt_protocol()
    candidate_preflight = preflight_candidate_evaluation(
        initial_candidate,
        training_profile=training.name,
        training_seed=options.seed,
        training_output_dir=resolved_output / "candidate_training" / "0000_seed",
        device=options.device,
        allow_cpu_for_tests=options.allow_cpu_for_tests,
        evaluation_profile=evaluation_plan.profile_name,
        evaluation_case_count=evaluation_case_count,
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


def _prompt_for_parent(
    *,
    system_prompt: str,
    parent: ArchiveCandidate,
    parent_ir: str,
    archive: FrozenSemanticArchive,
    opportunity: int,
) -> list[dict[str, str]]:
    signature = archive.cell_label(parent.signature)
    user_prompt = (
        f"Proposal opportunity {opportunity}.\n"
        f"Current parent public accuracy: {parent.public_accuracy:.6f}.\n"
        f"Current categorical signature: {signature}.\n"
        f"Occupied semantic cells: {archive.coverage}.\n\n"
        "Propose one testable architectural mechanism. Preserving public "
        "eligibility takes priority; an uncovered signature is useful only as "
        "exploratory coverage. Category numbers are unordered labels and are "
        "never fitness. Return exactly one complete replacement architecture IR "
        "JSON object. Put the testable hypothesis in "
        "metadata.mechanism_hypothesis. Do not return Python, a diff, markdown "
        "prose, or executable content. A single json fence is allowed.\n\n"
        f"Current architecture IR:\n```json\n{parent_ir}\n```"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def run_semantic_autoresearch(
    options: RunOptions,
    *,
    provider: ProposalProvider,
    evaluator: CandidateEvaluator = _default_evaluator,
) -> dict[str, object]:
    """Run one semantic Autoresearch replication and return its summary."""

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
    initial_ir = _load_initial_ir(
        initial_candidate,
        max_ir_bytes=options.max_ir_bytes,
    )
    initial_architecture_hash = _architecture_hash(initial_ir)

    run_id = f"semantic-autoresearch-seed-{options.seed}-{uuid.uuid4().hex[:8]}"
    requested_output = (
        options.output_dir
        if options.output_dir is not None
        else ROOT / "outputs" / "native_replications" / run_id
    )
    _validate_fresh_output_path(requested_output)
    system_prompt, prompt_components, prompt_manifest = _load_prompt_protocol()
    default_preflight: dict[str, object] | None = None
    if not injected_controller_test:
        default_preflight = _preflight_default_evaluator(
            options,
            planned_output=requested_output,
        )
    # This method may validate provider configuration, but must not issue a
    # generation request. All local execution gates above have already passed.
    provider.preflight()

    output_dir = _prepare_fresh_output(requested_output)
    artifacts = output_dir / "artifacts"
    artifacts.mkdir()
    _snapshot_prompt_protocol(
        output_dir,
        system_prompt=system_prompt,
        components=prompt_components,
    )
    training_root = output_dir / "candidate_training"
    ledger = output_dir / "lineage.jsonl"
    archive_path = output_dir / "semantic_archive.json"
    architecture_registry = ArchitectureHashRegistry(
        output_dir / "architecture_hash_registry"
    )
    if not architecture_registry.claim(initial_architecture_hash):
        raise PilotPreflightError("fresh architecture registry rejected the initial seed")

    seed_path = artifacts / "0000_seed.ir.json"
    seed_path.write_text(initial_ir, encoding="utf-8")
    seed_id = file_hash(seed_path)
    seed_lineage_record_id = _lineage_record_id(run_id, 0)
    seeds = TrainingSeedBundle.from_run_seed(options.seed)
    context = SearchEvaluationContext(
        study_id="native-replication",
        block_id="native-semantic-autoresearch",
        run_id=run_id,
        condition_id="native-semantic-autoresearch",
    )
    seen_evaluation_record_ids: set[str] = set()
    archive = FrozenSemanticArchive()
    component_hashes = trusted_component_hashes()
    manifest = {
        "run_id": run_id,
        "condition": CONDITION,
        "seed": options.seed,
        "candidate_budget": options.iterations + 1,
        "mutation_budget": options.iterations,
        "candidate_training_budget": options.iterations + 1,
        "max_ir_bytes": options.max_ir_bytes,
        "candidate_format": "architecture_tensor_graph@1.0",
        "run_mode": (
            "engineering_pilot" if options.engineering_pilot else "scientific_replication"
        ),
        "exploratory_only": options.engineering_pilot,
        "selection_semantics": (
            "mechanics_only_transformer_validity"
            if options.engineering_pilot
            else "frozen_scientific_parent_eligibility"
        ),
        "initial_candidate_is_evaluated": True,
        "generator": dict(provider.manifest_fields()),
        "prompt_protocol": prompt_manifest,
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
        "evidence_scope": "secondary_native_replication",
        "authoritative_scientific_evidence": False,
        "preflight": (
            default_preflight
            if default_preflight is not None
            else {
                "mode": "injected_controller_test_double",
                "generated_candidate_execution_by_controller": False,
                "explicit_controller_only_test_double": True,
                "candidate_boundary": "strict_validated_architecture_ir",
            }
        ),
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
        "evaluation": {
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
            "eligibility_threshold": options.eligibility_threshold,
        },
        "semantic_archive": {
            "axes": list(FROZEN_DESCRIPTOR_AXES),
            "parent_policy": "least_used_cell_then_accuracy",
            "novelty_role": "exploratory_coverage_tiebreak_only",
            "scientific_novelty_claim": False,
            "parameter_count_role": "descriptive_metadata_only",
        },
    }
    _atomic_json(output_dir / "run_manifest.json", manifest)
    _atomic_json(archive_path, archive.to_dict())

    # A complete seed evaluation must pass before the first provider request.
    seed_started = utc_now()
    try:
        seed_view = evaluator(
            EvaluationRequest(
                candidate_path=seed_path,
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
        )
        _validate_evaluator_binding(
            seed_view,
            expected_source_hash=seed_id,
            context=context,
            seen_record_ids=seen_evaluation_record_ids,
        )
    except Exception as error:
        stage = (
            "seed_evaluation_binding"
            if isinstance(error, EvaluationBindingError)
            else "seed_preflight_exception"
        )
        append_record(
            ledger,
            CandidateRecord(
                run_id=run_id,
                condition=CONDITION,
                seed=options.seed,
                candidate_id=seed_id,
                parent_id=None,
                lineage_record_id=seed_lineage_record_id,
                proposal_text="Initial candidate evaluator preflight.",
                code_hash=seed_id,
                proposal_timestamp=seed_started,
                completion_timestamp=utc_now(),
                evaluation={
                    "opportunity_index": 0,
                    **_failure_fields(stage, error),
                },
                retention_decision="seed_rejected",
            ),
        )
        raise PilotPreflightError(
            "initial candidate evaluation raised before any paid proposal call: "
            + _sanitized_exception(error)
        ) from error

    seed_evaluation = _lineage_evaluation(seed_view)
    seed_evaluation["opportunity_index"] = 0
    seed_decision = "seed_rejected"
    seed_cells: list[str] = []
    if seed_view.eligible_for_parent:
        try:
            seed_decision, seed_cell = archive.consider(
                candidate_id=seed_id,
                lineage_record_id=seed_lineage_record_id,
                source_path=seed_path,
                view=seed_view,
                opportunity=0,
            )
            seed_decision = "seed_parent"
            seed_cells = [seed_cell]
        except ValueError as error:
            seed_evaluation.update(
                _failure_fields("seed_semantic_descriptor", error)
            )
    append_record(
        ledger,
        CandidateRecord(
            run_id=run_id,
            condition=CONDITION,
            seed=options.seed,
            candidate_id=seed_id,
            parent_id=None,
            lineage_record_id=seed_lineage_record_id,
            proposal_text="Initial candidate evaluator preflight.",
            code_hash=seed_id,
            proposal_timestamp=seed_started,
            completion_timestamp=utc_now(),
            evaluation=seed_evaluation,
            retention_decision=seed_decision,
            archive_cells=seed_cells,
        ),
    )
    _atomic_json(archive_path, archive.to_dict())
    if seed_decision != "seed_parent":
        raise PilotPreflightError(
            "initial candidate is not a valid, eligible semantic parent; "
            "no paid proposal call was made"
        )

    completed = 0
    for opportunity in range(1, options.iterations + 1):
        proposal_time = utc_now()
        proposal_id = _proposal_id(run_id, opportunity)
        lineage_record_id = _lineage_record_id(run_id, opportunity)
        parent = archive.select_parent()
        parent_ir = parent.source_path.read_text(encoding="utf-8")
        messages = _prompt_for_parent(
            system_prompt=system_prompt,
            parent=parent,
            parent_ir=parent_ir,
            archive=archive,
            opportunity=opportunity,
        )
        user_prompt = messages[-1]["content"]
        prompt_hash = _canonical_messages_hash(messages)
        artifact_base = artifacts / f"{opportunity:04d}"
        artifact_base.with_suffix(".prompt.md").write_text(
            user_prompt, encoding="utf-8"
        )
        artifact_base.with_suffix(".messages.json").write_text(
            json.dumps(messages, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        try:
            response = provider.generate(messages)
        except Exception as error:
            safe_error = _provider_failure_detail(provider, error)
            append_record(
                ledger,
                CandidateRecord(
                    run_id=run_id,
                    condition=CONDITION,
                    seed=options.seed,
                    candidate_id=_failed_candidate_id(
                        proposal_id, "provider_failure"
                    ),
                    parent_id=parent.candidate_id,
                    lineage_record_id=lineage_record_id,
                    proposal_id=proposal_id,
                    parent_lineage_record_id=parent.lineage_record_id,
                    prompt_hash=prompt_hash,
                    proposal_timestamp=proposal_time,
                    completion_timestamp=utc_now(),
                    evaluation={
                        "opportunity_index": opportunity,
                        **_failure_fields("provider_failure", safe_error),
                    },
                    retention_decision="crash",
                    rollback_target=parent.candidate_id,
                ),
            )
            _atomic_json(archive_path, archive.to_dict())
            completed += 1
            continue

        response_text = response.text
        response_artifact = artifact_base.with_suffix(".response.txt")
        response_artifact.write_text(
            response_text, encoding="utf-8"
        )
        base_record = {
            "run_id": run_id,
            "condition": CONDITION,
            "seed": options.seed,
            "parent_id": parent.candidate_id,
            "lineage_record_id": lineage_record_id,
            "proposal_id": proposal_id,
            "parent_lineage_record_id": parent.lineage_record_id,
            "proposal_text": "declarative architecture IR proposal",
            "mechanism_hypothesis": "",
            "prompt_hash": prompt_hash,
            "response_hash": text_hash(response_text),
            "proposal_timestamp": proposal_time,
            "rollback_target": parent.candidate_id,
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
                    **{
                        **base_record,
                        "proposal_text": str(error),
                    },
                    candidate_id=_failed_candidate_id(
                        proposal_id, "ir_validation"
                    ),
                    completion_timestamp=utc_now(),
                    evaluation={
                        "opportunity_index": opportunity,
                        **_failure_fields(
                            "ir_validation", error
                        ),
                    },
                    retention_decision="crash",
                ),
            )
            _atomic_json(archive_path, archive.to_dict())
            completed += 1
            continue
        base_record["proposal_text"] = hypothesis
        base_record["mechanism_hypothesis"] = hypothesis
        parent_canonical, _ = _validated_ir_text(
            parent_ir,
            max_ir_bytes=options.max_ir_bytes,
            require_hypothesis=False,
        )
        if _architecture_hash(child_ir) == _architecture_hash(parent_canonical):
            failure_id = _failed_candidate_id(proposal_id, "mutation_no_change")
            append_record(
                ledger,
                CandidateRecord(
                    **base_record,
                    candidate_id=failure_id,
                    completion_timestamp=utc_now(),
                    code_hash=parent.candidate_id,
                    evaluation={
                        "opportunity_index": opportunity,
                        "failed_candidate_fingerprint": parent.candidate_id,
                        **_failure_fields(
                            "mutation_no_change",
                            "replacement IR changes only document identity or metadata, "
                            "not executable architecture",
                        ),
                    },
                    retention_decision="crash",
                ),
            )
            _atomic_json(archive_path, archive.to_dict())
            completed += 1
            continue

        candidate_id = text_hash(child_ir)
        child_architecture_hash = _architecture_hash(child_ir)
        if not architecture_registry.claim(child_architecture_hash):
            append_record(
                ledger,
                CandidateRecord(
                    **base_record,
                    candidate_id=_failed_candidate_id(
                        proposal_id, "duplicate_architecture"
                    ),
                    completion_timestamp=utc_now(),
                    code_hash=candidate_id,
                    evaluation={
                        "opportunity_index": opportunity,
                        "duplicate_architecture_hash": child_architecture_hash,
                        **_failure_fields(
                            "duplicate_architecture",
                            "normalized executable architecture was already proposed "
                            "or evaluated in this run",
                        ),
                    },
                    retention_decision="duplicate_rejected",
                ),
            )
            _atomic_json(archive_path, archive.to_dict())
            completed += 1
            continue

        child_path = artifacts / f"{opportunity:04d}_{candidate_id[:12]}.ir.json"
        child_path.write_text(child_ir, encoding="utf-8")
        request = EvaluationRequest(
            candidate_path=child_path,
            training_output_dir=(
                training_root / f"{opportunity:04d}_{candidate_id[:12]}"
            ),
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
            _validate_evaluator_binding(
                view,
                expected_source_hash=candidate_id,
                context=context,
                seen_record_ids=seen_evaluation_record_ids,
            )
        except Exception as error:
            if isinstance(error, EvaluationBindingError):
                stage = "evaluator_binding"
            else:
                stage = "evaluator_exception"
            append_record(
                ledger,
                CandidateRecord(
                    **base_record,
                    candidate_id=candidate_id,
                    completion_timestamp=utc_now(),
                    code_hash=candidate_id,
                    evaluation={
                        "opportunity_index": opportunity,
                        **_failure_fields(stage, error),
                    },
                    retention_decision="crash",
                ),
            )
            _atomic_json(archive_path, archive.to_dict())
            completed += 1
            continue

        evaluation = _lineage_evaluation(view)
        evaluation["opportunity_index"] = opportunity
        cells: list[str] = []
        if view.eligible_for_parent:
            try:
                decision, cell = archive.consider(
                    candidate_id=candidate_id,
                    lineage_record_id=lineage_record_id,
                    source_path=child_path,
                    view=view,
                    opportunity=opportunity,
                )
                cells = [cell]
                rollback = None if decision in {"archive_new_cell", "archive_replace"} else parent.candidate_id
            except ValueError as error:
                decision = "crash"
                rollback = parent.candidate_id
                evaluation.update(_failure_fields("semantic_descriptor", error))
        else:
            decision = "reject"
            rollback = parent.candidate_id

        append_record(
            ledger,
            CandidateRecord(
                **{
                    **base_record,
                    "rollback_target": rollback,
                },
                candidate_id=candidate_id,
                completion_timestamp=utc_now(),
                code_hash=candidate_id,
                evaluation=evaluation,
                retention_decision=decision,
                archive_cells=cells,
            ),
        )
        _atomic_json(archive_path, archive.to_dict())
        completed += 1

    summary = {
        "run_id": run_id,
        "condition": CONDITION,
        "proposal_opportunities_requested": options.iterations,
        "proposal_opportunities_terminal": completed,
        "semantic_archive_cells": archive.coverage,
        "lineage_path": str(ledger),
        "archive_path": str(archive_path),
        "scientific_novelty_claim": False,
    }
    _atomic_json(output_dir / "run_summary.json", summary)
    return summary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _provider_from_environment(
    config: Mapping[str, object],
    seed: int,
    *,
    scientific: bool,
) -> ProposalProvider:
    values = {
        "DISCOVERY_API_KEY": os.environ.get("DISCOVERY_API_KEY"),
        "DISCOVERY_API_BASE": os.environ.get("DISCOVERY_API_BASE"),
        "DISCOVERY_MODEL": os.environ.get("DISCOVERY_MODEL"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SystemExit("missing provider configuration: " + ", ".join(missing))
    try:
        endpoint = resolve_provider_endpoint(
            str(values["DISCOVERY_API_BASE"]),
            scientific=scientific,
        )
        generation = GPT56SolProfile.resolve(
            model=str(values["DISCOVERY_MODEL"]),
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
        api_key=str(values["DISCOVERY_API_KEY"]),
        endpoint=endpoint,
        generation=generation,
    )


def main() -> None:
    config = yaml.safe_load((AGENT_DIR / "config.yaml").read_text())
    parser = argparse.ArgumentParser(
        description="Run semantic-coverage Autoresearch with evaluator-owned training."
    )
    parser.add_argument("--iterations", type=_positive_int, default=config["iterations"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--engineering-pilot",
        action="store_true",
        help=(
            "run an exploratory IR-only MPS canary; forces smoke_train_v1 and "
            "smoke_eval_v1 and never creates authoritative scientific evidence"
        ),
    )
    parser.add_argument(
        "--training-profile",
        choices=tuple(sorted(PROFILES)),
    )
    parser.add_argument(
        "--evaluation-profile",
        choices=CONTROLLER_EVALUATION_PROFILES,
    )
    parser.add_argument("--evaluation-cases", type=_positive_int)
    parser.add_argument("--device", choices=("mps", "cpu"))
    parser.add_argument("--allow-cpu-for-tests", action="store_true")
    parser.add_argument("--pi-decision-record-id")
    arguments = parser.parse_args()

    if arguments.engineering_pilot and arguments.training_profile not in {
        None,
        "smoke_train_v1",
    }:
        parser.error("--engineering-pilot forces --training-profile smoke_train_v1")
    if arguments.engineering_pilot and arguments.evaluation_profile not in {
        None,
        "smoke_eval_v1",
    }:
        parser.error("--engineering-pilot forces --evaluation-profile smoke_eval_v1")
    if arguments.engineering_pilot and arguments.pi_decision_record_id:
        parser.error("--engineering-pilot cannot claim a PI decision record")

    training_profile = (
        "smoke_train_v1"
        if arguments.engineering_pilot
        else arguments.training_profile or config["training"]["profile"]
    )
    evaluation_profile = (
        "smoke_eval_v1"
        if arguments.engineering_pilot
        else arguments.evaluation_profile or config["evaluation"]["profile"]
    )
    device = arguments.device or str(config["training"]["device"])

    configured_candidate = (ROOT / str(config["candidate_path"])).resolve()
    if configured_candidate != DEFAULT_INITIAL_CANDIDATE.resolve():
        parser.error("configured candidate_path differs from the trusted initial seed")

    case_count = arguments.evaluation_cases
    if (
        case_count is None
        and not arguments.engineering_pilot
        and os.environ.get("DISCOVERY_LAYER_A_CASES")
    ):
        try:
            case_count = _positive_int(os.environ["DISCOVERY_LAYER_A_CASES"])
        except (ValueError, argparse.ArgumentTypeError) as error:
            parser.error(f"DISCOVERY_LAYER_A_CASES {error}")
    decision = None if arguments.engineering_pilot else (
        arguments.pi_decision_record_id
        or os.environ.get("DISCOVERY_SCIENTIFIC_DECISION_RECORD")
    )
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
            else float(config["evaluation"]["eligibility_threshold"])
        ),
        engineering_pilot=arguments.engineering_pilot,
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
        summary = run_semantic_autoresearch(options, provider=provider)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
