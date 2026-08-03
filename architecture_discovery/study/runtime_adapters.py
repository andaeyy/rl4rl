"""Real-provider and evaluator adapters for the project-owned causal engine.

Importing this module performs no provider request and starts no training. The
offline smoke uses ``study.fakes`` instead. Scientific execution remains
subject to the explicit readiness, evaluation-profile, MPS, and containment
gates in the called evaluator.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openevolve.utils.code_utils import apply_diff, extract_diffs

from common.evaluation_profiles import EvaluationLayer
from common.evaluator import SearchEvaluationContext, evaluate_candidate
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


class CandidateSourceStore:
    """Immutable local source objects addressed by the engine's content hash."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, candidate_id: str, source: str) -> Path:
        if not candidate_id or any(character in candidate_id for character in "/\\\x00"):
            raise ValueError("candidate_id is not a safe source-store identifier")
        destination = self.root / f"{candidate_id}.py"
        payload = source.encode("utf-8")
        try:
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o400,
            )
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise ValueError("candidate ID collided with different source")
            return destination
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return destination

    def put(self, source: str) -> tuple[str, Path]:
        candidate_id = content_hash(source)
        return candidate_id, self.register(candidate_id, source)

    def path(self, candidate_id: str) -> Path:
        path = self.root / f"{candidate_id}.py"
        if not path.is_file():
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
        rendered_slots = "\n\n".join(
            f"PARENT SLOT {index + 1}:\n```python\n{source}\n```"
            for index, source in enumerate(slots)
        )
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
                "Repair only its SEARCH/REPLACE format without using evaluation "
                "feedback. Previous response follows:\n"
                f"{context.previous_response}"
            )
        else:
            proposal_directive = treatment_directive
        return (
            f"Opportunity {context.opportunity_index}.\n"
            f"Designated mutation base: active parent slot {designated_base}.\n"
            f"Proposal directive: {proposal_directive}\n\n"
            f"{rendered_slots}\n\n"
            "Return a short falsifiable mechanism hypothesis followed by "
            "SEARCH/REPLACE blocks for the designated base."
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
        base_id = context.parent_ids[-1]
        base_source = self.source_store.read(base_id)
        candidate_source: str | None = None
        if extract_diffs(response_text):
            candidate_source = apply_diff(base_source, response_text)
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
            context=self._context(),
        )
        self._artifact_store.write_json(record.envelope.record_id, record.to_dict())
        attempts, steps, examples, seconds = self._training_resources(
            training_dir,
            expected_candidate_hash=candidate_id,
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
            mps_seconds=seconds,
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
        expected = content_hash(candidate_source)
        if candidate_id != expected:
            raise ValueError("engine candidate ID does not match candidate source")
        source_path = self.source_store.register(candidate_id, candidate_source)
        return self._run(
            candidate_id=candidate_id,
            source_path=source_path,
            run_seed=run_seed,
            label=f"opportunity-{opportunity_index:06d}-{candidate_id[:12]}",
        )
