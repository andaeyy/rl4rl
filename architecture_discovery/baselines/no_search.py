"""Independent proposal baseline with no adaptive-search information channel.

This module intentionally contains no OpenAI client.  A future provider adapter may
implement :class:`IndependentProposalBackend`, but the scientific request passed to
that adapter is fixed by ``NoSearchSpec`` and contains no parent, score, ancestry,
candidate-history, transition, repair, or evaluation fields.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from study.budget import BudgetSpec
from study.interfaces import ProposalContext, ProposalResult, RetryableProviderError
from study.serialization import content_hash


TARGET_MODEL = "gpt-5.6-sol"


class GeneratorBudgetMismatch(ValueError):
    """The baseline and comparison arm have unequal generator opportunity."""


@dataclass(frozen=True)
class NoSearchSpec:
    """Frozen model input shared by every independent proposal opportunity."""

    system_prompt: str
    task_prompt: str
    max_completion_tokens: int
    model: str = TARGET_MODEL
    reasoning_effort: str = "high"
    schema_name: str = field(default="NoSearchSpec", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        if self.model != TARGET_MODEL:
            raise ValueError(f"no-search model must be pinned to {TARGET_MODEL!r}")
        if not self.system_prompt.strip() or not self.task_prompt.strip():
            raise ValueError("no-search prompts cannot be empty")
        if self.max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be positive")
        if self.reasoning_effort not in {"high", "xhigh"}:
            raise ValueError("reasoning_effort must be 'high' or 'xhigh'")

    def model_input(self) -> dict[str, object]:
        """Return the entire provider-visible payload.

        Keeping this method small makes information-flow tests straightforward: no
        run-specific or controller-derived value is provider-visible.
        """

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.task_prompt},
            ],
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_completion_tokens,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "system_prompt": self.system_prompt,
            "task_prompt": self.task_prompt,
            "max_completion_tokens": self.max_completion_tokens,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
        }


@dataclass(frozen=True)
class IndependentOpportunity:
    """Metadata for one proposal, with adaptive-search fields absent by design."""

    study_id: str
    block_id: str
    run_id: str
    run_seed: int
    opportunity_index: int
    provider_attempt: int

    @classmethod
    def from_study_context(cls, context: ProposalContext) -> IndependentOpportunity:
        """Project a rich study context onto the non-adaptive information boundary."""

        return cls(
            study_id=context.study_id,
            block_id=context.block_id,
            run_id=context.run_id,
            run_seed=context.run_seed,
            opportunity_index=context.opportunity_index,
            provider_attempt=context.provider_attempt,
        )

    @property
    def request_id(self) -> str:
        return "no-search-" + content_hash(
            {
                "study_id": self.study_id,
                "block_id": self.block_id,
                "run_id": self.run_id,
                "run_seed": self.run_seed,
                "opportunity_index": self.opportunity_index,
                "provider_attempt": self.provider_attempt,
            }
        )[:24]


@dataclass(frozen=True)
class NoSearchRequest:
    """Provider request plus non-prompt idempotency metadata."""

    request_id: str
    model_input: dict[str, object]


@dataclass(frozen=True)
class BackendResponse:
    response_text: str
    candidate_source: str | None
    prompt_tokens: int | None
    completion_tokens: int | None


@runtime_checkable
class IndependentProposalBackend(Protocol):
    """Provider boundary; implementations must not receive controller state."""

    is_test_double: ClassVar[bool]

    def complete(self, request: NoSearchRequest) -> BackendResponse: ...


class NoSearchProposalGenerator:
    """Study-engine adapter for independent, feedback-free proposals."""

    def __init__(
        self,
        *,
        spec: NoSearchSpec,
        backend: IndependentProposalBackend,
        scientific: bool,
    ) -> None:
        if not isinstance(spec, NoSearchSpec):
            raise TypeError("spec must be NoSearchSpec, not a prompt-placebo spec")
        if scientific and getattr(backend, "is_test_double", False):
            raise ValueError("a deterministic fake backend cannot run scientifically")
        self._spec = spec
        self._backend = backend

    @property
    def provider_visible_input(self) -> dict[str, object]:
        """Return a fresh copy of the constant provider-visible payload."""

        payload = self._spec.model_input()
        return {
            **payload,
            "messages": [dict(message) for message in payload["messages"]],
        }

    def generate_independent(
        self, opportunity: IndependentOpportunity
    ) -> ProposalResult:
        request = NoSearchRequest(
            request_id=opportunity.request_id,
            model_input=self.provider_visible_input,
        )
        try:
            response = self._backend.complete(request)
        except RetryableProviderError:
            raise
        except Exception as error:
            raise RetryableProviderError(
                f"no-search provider attempt failed: {type(error).__name__}"
            ) from error
        return ProposalResult(
            response_text=response.response_text,
            candidate_source=response.candidate_source,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

    def generate(self, context: ProposalContext) -> ProposalResult:
        """Implement ``study.interfaces.ProposalGenerator`` without feedback use."""

        return self.generate_independent(IndependentOpportunity.from_study_context(context))


class DeterministicFakeBackend:
    """Offline-only backend used by unit and synthetic end-to-end tests."""

    is_test_double: ClassVar[bool] = True

    def __init__(
        self,
        *,
        prompt_tokens: int = 23,
        completion_tokens: int = 17,
        candidate_ir: str | None = None,
    ) -> None:
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("fake token counts cannot be negative")
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.requests: list[NoSearchRequest] = []
        if candidate_ir is None:
            source = (
                Path(__file__).resolve().parents[1]
                / "common"
                / "initial_candidate.ir.json"
            ).read_text(encoding="utf-8")
        else:
            source = candidate_ir
        try:
            payload = json.loads(source)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("fake no-search candidate must be JSON") from error
        if not isinstance(payload, dict) or (
            payload.get("schema_name") != "architecture_tensor_graph"
            or payload.get("schema_version") != "1.0"
        ):
            raise ValueError(
                "fake no-search candidate must be architecture_tensor_graph v1.0"
            )
        nodes = payload.get("nodes")
        edges = payload.get("edges")
        if not isinstance(nodes, list) or not all(
            isinstance(item, dict) and isinstance(item.get("node_id"), str)
            for item in nodes
        ):
            raise ValueError("fake no-search candidate has invalid nodes")
        if not isinstance(edges, list) or not all(
            isinstance(item, dict)
            and all(
                key in item
                for key in ("source", "target", "target_port", "kind")
            )
            for item in edges
        ):
            raise ValueError("fake no-search candidate has invalid edges")
        payload["nodes"] = sorted(nodes, key=lambda item: item["node_id"])
        payload["edges"] = sorted(
            edges,
            key=lambda item: (
                item["target"],
                item["target_port"],
                item["source"],
                item["kind"],
            ),
        )
        self._candidate_source = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def candidate_source(self) -> str:
        """Canonical provider-free seed document used by the smoke path."""

        return self._candidate_source

    def complete(self, request: NoSearchRequest) -> BackendResponse:
        self.requests.append(request)
        digest = hashlib.sha256(request.request_id.encode("utf-8")).hexdigest()[:16]
        payload = json.loads(self._candidate_source)
        payload["graph_id"] = f"offline_no_search_{digest}"
        metadata = dict(payload["metadata"])
        metadata["mechanism_hypothesis"] = (
            f"offline deterministic no-search fixture {digest}"
        )
        payload["metadata"] = metadata
        source = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return BackendResponse(
            response_text=source,
            candidate_source=source,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )


_MATCHED_GENERATOR_FIELDS = (
    "proposal_opportunities",
    "provider_attempts_per_opportunity",
    "repair_attempts_per_opportunity",
    "prompt_tokens",
    "completion_tokens",
    "repairs",
)


def assert_matched_generator_budget(
    baseline: BudgetSpec,
    comparison: BudgetSpec,
) -> None:
    """Fail unless both arms receive the same frozen generation opportunity.

    Candidate training and evaluation budgets are controlled elsewhere.  Repairs
    are included here because allowing one arm extra repair prompts would create
    additional generator feedback and token opportunity.
    """

    differences = {
        name: (getattr(baseline, name), getattr(comparison, name))
        for name in _MATCHED_GENERATOR_FIELDS
        if getattr(baseline, name) != getattr(comparison, name)
    }
    if differences:
        formatted = ", ".join(
            f"{name}={left!r} vs {right!r}"
            for name, (left, right) in differences.items()
        )
        raise GeneratorBudgetMismatch(
            f"no-search generator budget does not match comparison arm: {formatted}"
        )
