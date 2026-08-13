"""Real-provider and evaluator adapters for the project-owned causal engine.

Importing this module performs no provider request and starts no training. The
offline smoke uses ``study.fakes`` instead. Scientific execution remains
subject to the explicit readiness, evaluation-profile, accelerator, and
containment gates in the called evaluator.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from architecture_ir import encode_graph_json, validate_ir_candidate_json
from architecture_ir.codec import MAX_IR_JSON_BYTES
from common.evaluation_profiles import EvaluationLayer
from common.evaluator import (
    SearchEvaluationContext,
    evaluate_candidate,
    file_hash,
    validate_controller_view_binding,
)
from common.gpt56_sol import GPT56SolProfile
from evaluation.artifacts import EvaluationArtifactRoots, JsonEvaluationArtifactStore
from evaluation.records import SearchEvaluationRecord
from study.budget import OpportunityOutcome
from study.interfaces import (
    EvaluationResult,
    ProposalContext,
    ProposalResult,
    RetryableProviderError,
)
from study.serialization import content_hash, require_bool, require_int


class ScientificReadinessBlocked(RuntimeError):
    """A launch gate failed and the study must stop rather than charge a candidate."""


class ArchitectureIRProposalError(ValueError):
    """A provider response is not one complete, valid Architecture IR document."""


_JSON_FENCE = re.compile(r"\A```json[ \t]*\r?\n(?P<body>.*)\r?\n```\Z", re.DOTALL)


def canonicalize_architecture_ir(
    text: str,
    *,
    require_hypothesis: bool,
    allow_json_fence: bool = True,
) -> str:
    """Validate untrusted IR text and return the one canonical JSON encoding.

    A provider may wrap its document in one exact ``json`` fence for transport
    compatibility. No surrounding prose, additional fences, partial objects,
    patches, or executable source are accepted.
    """

    if not isinstance(text, str):
        raise TypeError("architecture IR proposal must be text")
    if len(text.encode("utf-8")) > MAX_IR_JSON_BYTES:
        raise ArchitectureIRProposalError(
            f"architecture IR proposal exceeds {MAX_IR_JSON_BYTES} bytes"
        )
    stripped = text.strip()
    if stripped.startswith("```") or stripped.endswith("```"):
        match = _JSON_FENCE.fullmatch(stripped) if allow_json_fence else None
        if match is None:
            raise ArchitectureIRProposalError(
                "architecture IR may use at most one exact json fence"
            )
        stripped = match.group("body")
    elif "```" in stripped:
        raise ArchitectureIRProposalError(
            "architecture IR response contains an unexpected Markdown fence"
        )
    validation = validate_ir_candidate_json(stripped)
    if not validation.valid or validation.graph is None:
        issue_codes = sorted({issue.code for issue in validation.issues})
        detail = ", ".join(issue_codes) or "unknown_validation_failure"
        raise ArchitectureIRProposalError(
            f"architecture IR candidate failed validation: {detail}"
        )
    if require_hypothesis:
        hypothesis = validation.graph.metadata.get("mechanism_hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ArchitectureIRProposalError(
                "architecture IR metadata.mechanism_hypothesis must be non-empty"
            )
    canonical = encode_graph_json(validation.graph)
    if len(canonical.encode("utf-8")) > MAX_IR_JSON_BYTES:
        raise ArchitectureIRProposalError(
            f"canonical architecture IR exceeds {MAX_IR_JSON_BYTES} bytes"
        )
    return canonical


class CandidateSourceStore:
    """Immutable canonical IR objects addressed by the engine's content hash."""

    def __init__(self, root: str | Path) -> None:
        requested_root = Path(root).expanduser()
        if requested_root.is_symlink():
            raise ValueError("candidate source-store root cannot be a symlink")
        requested_root.mkdir(parents=True, exist_ok=True)
        if requested_root.is_symlink() or not requested_root.is_dir():
            raise ValueError("candidate source-store root must be a directory")
        self.root = requested_root.resolve()

    def register(self, candidate_id: str, source: str) -> Path:
        if not candidate_id or any(character in candidate_id for character in "/\\\x00"):
            raise ValueError("candidate_id is not a safe source-store identifier")
        canonical = canonicalize_architecture_ir(
            source,
            require_hypothesis=False,
        )
        if candidate_id != content_hash(canonical):
            raise ValueError(
                "candidate ID does not match the canonical architecture IR"
            )
        destination = self.root / f"{candidate_id}.json"
        payload = canonical.encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(
                destination,
                flags,
                0o400,
            )
        except FileExistsError:
            if destination.is_symlink():
                raise ValueError("candidate source object cannot be a symlink")
            if destination.read_bytes() != payload:
                raise ValueError("candidate ID collided with different source")
            return destination
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return destination

    def put(self, source: str) -> tuple[str, Path]:
        canonical = canonicalize_architecture_ir(
            source,
            require_hypothesis=True,
        )
        candidate_id = content_hash(canonical)
        return candidate_id, self.register(candidate_id, canonical)

    def path(self, candidate_id: str) -> Path:
        path = self.root / f"{candidate_id}.json"
        if path.is_symlink() or not path.is_file():
            raise KeyError(f"candidate source {candidate_id!r} is unavailable")
        return path

    def read(self, candidate_id: str) -> str:
        return self.path(candidate_id).read_text(encoding="utf-8")


@dataclass
class MatchedCausalProposalGenerator:
    """One prompt schema shared by C0-C3, with fixed parent slots."""

    client: Any
    generation: GPT56SolProfile
    source_store: CandidateSourceStore
    portfolio_size: int
    system_prompt: str
    request_log_root: Path
    neutral_slot_text: str = "[NEUTRAL EMPTY PARENT SLOT]"
    calls: list[ProposalContext] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.portfolio_size < 2:
            raise ValueError("portfolio_size must be at least two")
        self.request_log_root = Path(self.request_log_root).resolve()
        self.request_log_root.mkdir(parents=True, exist_ok=True)

    def _parent_slots(self, context: ProposalContext) -> tuple[str, ...]:
        sources = [self.source_store.read(identifier) for identifier in context.parent_ids]
        if len(sources) > self.portfolio_size:
            raise ValueError("proposal context exceeds the frozen parent-slot count")
        return tuple(sources) + (self.neutral_slot_text,) * (
            self.portfolio_size - len(sources)
        )

    def build_user_prompt(self, context: ProposalContext) -> str:
        slots = self._parent_slots(context)
        rendered: list[str] = []
        for index, source in enumerate(slots):
            if source == self.neutral_slot_text:
                rendered.append(f"PARENT SLOT {index + 1}:\n{source}")
            else:
                rendered.append(
                    f"PARENT SLOT {index + 1}:\n```json\n{source}\n```"
                )
        rendered_slots = "\n\n".join(rendered)
        designated_base = len(context.parent_ids)
        treatment_directive = (
            "Reconsider one abstract architectural assumption before proposing "
            "one testable architecture mutation."
            if context.transition_active
            else "Propose one testable architecture mutation using ordinary "
            "mechanism-based reasoning."
        )
        if context.repair:
            if context.previous_response is None:
                raise ValueError("repair context lacks the failed response")
            proposal_directive = (
                f"{treatment_directive} The previous response could not be parsed. "
                "Repair only its complete Architecture IR JSON format without using "
                "evaluation feedback. Previous response follows:\n"
                f"{context.previous_response}"
            )
        else:
            proposal_directive = treatment_directive
        return (
            f"Opportunity {context.opportunity_index}.\n"
            f"Designated mutation base: active parent slot {designated_base}.\n"
            f"Proposal directive: {proposal_directive}\n\n"
            f"{rendered_slots}\n\n"
            "Return exactly one complete replacement architecture_tensor_graph "
            "version 1.0 JSON object. Put a short falsifiable mechanism hypothesis "
            "in metadata.mechanism_hypothesis. Return no Python, source diff, "
            "Markdown prose, or executable content."
        )

    def generate(self, context: ProposalContext) -> ProposalResult:
        self.calls.append(context)
        prompt = self.build_user_prompt(context)
        messages = (
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        )
        try:
            response = self.client.chat.completions.create(
                **self.generation.chat_completion_request(messages)
            )
        except Exception as error:
            raise RetryableProviderError(
                f"provider attempt failed: {type(error).__name__}"
            ) from error
        response_text = response.choices[0].message.content or ""
        usage = response.usage
        candidate_source: str | None = None
        try:
            candidate_source = canonicalize_architecture_ir(
                response_text,
                require_hypothesis=True,
            )
        except ArchitectureIRProposalError:
            candidate_source = None
        if candidate_source is not None:
            generated_id, _ = self.source_store.put(candidate_source)
            if generated_id != content_hash(candidate_source):
                raise RuntimeError("candidate source store changed the engine identity")
        record = {
            "study_id": context.study_id,
            "block_id": context.block_id,
            "run_id": context.run_id,
            "opportunity_index": context.opportunity_index,
            "provider_attempt": context.provider_attempt,
            "condition_id": context.condition.condition_id.value,
            "prompt": prompt,
            "response": response_text,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }
        path = self.request_log_root / (
            f"{context.opportunity_index:06d}-attempt-{context.provider_attempt}.json"
        )
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return ProposalResult(
            response_text=response_text,
            candidate_source=candidate_source,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )


EvaluateFunction = Callable[..., SearchEvaluationRecord]


@dataclass
class LayerACandidateEvaluator:
    """Map trusted Layer A records into the narrow study-engine result."""

    study_id: str
    block_id: str
    run_id: str
    condition_id: str
    initial_candidate_id: str
    source_store: CandidateSourceStore
    output_root: Path
    training_profile: str
    device: str
    allow_cpu_for_tests: bool
    evaluation_profile: str
    evaluation_case_count: int
    pi_decision_record_id: str | None
    eligibility_threshold: float
    evaluate_function: EvaluateFunction = evaluate_candidate

    def __post_init__(self) -> None:
        require_bool(self.allow_cpu_for_tests, "allow_cpu_for_tests")
        self.output_root = Path(self.output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        roots = EvaluationArtifactRoots.under(self.output_root / "evaluations")
        roots.prepare()
        self._artifact_store = JsonEvaluationArtifactStore(
            roots, layer=EvaluationLayer.SEARCH
        )

    def _context(self) -> SearchEvaluationContext:
        return SearchEvaluationContext(
            study_id=self.study_id,
            block_id=self.block_id,
            run_id=self.run_id,
            condition_id=self.condition_id,
        )

    @staticmethod
    def _training_resources(
        output_dir: Path,
        *,
        expected_candidate_hash: str,
        expected_profile_name: str,
    ) -> tuple[int, int, int, float]:
        summary = output_dir / "training_summary.json"
        failure = output_dir / "failure.json"
        source = summary if summary.is_file() else failure
        if not source.is_file():
            return 0, 0, 0, 0.0
        payload = json.loads(source.read_text(encoding="utf-8"))
        if source == summary:
            if payload.get("candidate_source_hash") != expected_candidate_hash:
                raise ValueError("training summary belongs to a different candidate")
            if payload.get("profile_name") != expected_profile_name:
                raise ValueError("training summary belongs to a different profile")
        steps = payload.get("steps_completed", 0)
        examples = payload.get("examples_processed", 0)
        seconds = payload.get("train_seconds", 0.0)
        require_int(steps, "steps_completed")
        require_int(examples, "examples_processed")
        if steps < 0 or examples < 0:
            raise ValueError("training resource counters cannot be negative")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("train_seconds must be a finite non-negative number")
        return (
            1,
            steps,
            examples,
            float(seconds),
        )

    def _run(
        self,
        *,
        candidate_id: str,
        source_path: Path,
        run_seed: int,
        label: str,
    ) -> EvaluationResult:
        training_dir = self.output_root / "candidate_training" / label
        artifact_digest = file_hash(source_path)
        context = self._context()
        record = self.evaluate_function(
            source_path,
            training_profile=self.training_profile,
            training_seed=run_seed,
            training_output_dir=training_dir,
            device=self.device,
            allow_cpu_for_tests=self.allow_cpu_for_tests,
            evaluation_profile=self.evaluation_profile,
            evaluation_case_count=self.evaluation_case_count,
            pi_decision_record_id=self.pi_decision_record_id,
            eligibility_threshold=self.eligibility_threshold,
            context=context,
        )
        validate_controller_view_binding(
            record.controller_view(),
            candidate_source_hash=artifact_digest,
            context=context,
        )
        self._artifact_store.write_json(record.envelope.record_id, record.to_dict())
        attempts, steps, examples, seconds = self._training_resources(
            training_dir,
            expected_candidate_hash=artifact_digest,
            expected_profile_name=self.training_profile,
        )
        if record.failure_stage in {
            "containment_unproven",
            "device_unavailable",
        }:
            raise ScientificReadinessBlocked(
                f"study readiness gate failed: {record.failure_stage}"
            )
        if record.infrastructure_failure:
            outcome = OpportunityOutcome.INFRASTRUCTURE_FAILURE
        elif record.eligible_for_parent:
            outcome = OpportunityOutcome.ACCEPTED
        elif not record.execution_ok or not record.transformer_valid:
            outcome = OpportunityOutcome.SCIENTIFIC_FAILURE
        else:
            outcome = OpportunityOutcome.REJECTED
        return EvaluationResult(
            outcome=outcome,
            score=record.search_score,
            training_attempts=attempts,
            training_steps=steps,
            training_examples=examples,
            accelerator_kind=self.device,
            accelerator_seconds=seconds,
            evaluation_cases=(self.evaluation_case_count if record.execution_ok else 0),
            failure_stage=record.failure_stage,
        )

    def evaluate_seed(self, initial_candidate_id: str, run_seed: int) -> EvaluationResult:
        if initial_candidate_id != self.initial_candidate_id:
            raise ValueError("engine requested a different initial candidate")
        return self._run(
            candidate_id=initial_candidate_id,
            source_path=self.source_store.path(initial_candidate_id),
            run_seed=run_seed,
            label="seed",
        )

    def evaluate_candidate(
        self,
        candidate_source: str,
        *,
        candidate_id: str,
        opportunity_index: int,
        run_seed: int,
    ) -> EvaluationResult:
        canonical = canonicalize_architecture_ir(
            candidate_source,
            require_hypothesis=True,
        )
        if candidate_source != canonical:
            raise ValueError("engine candidate source must already be canonical IR")
        expected = content_hash(canonical)
        if candidate_id != expected:
            raise ValueError("engine candidate ID does not match candidate source")
        source_path = self.source_store.register(candidate_id, canonical)
        return self._run(
            candidate_id=candidate_id,
            source_path=source_path,
            run_seed=run_seed,
            label=f"opportunity-{opportunity_index:06d}-{candidate_id[:12]}",
        )
