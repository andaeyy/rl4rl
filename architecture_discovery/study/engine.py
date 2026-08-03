"""One resumable offline-capable engine shared by causal conditions C0-C3."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager

from study.budget import BudgetExceeded, BudgetLedger, OpportunityOutcome
from study.contracts import (
    ParentPolicy,
    ProposalPolicy,
    RunSpec,
    RunState,
    StudySpec,
    utc_now,
)
from study.interfaces import (
    CandidateEvaluator,
    EvaluationResult,
    ProposalContext,
    ProposalGenerator,
    ProposalResult,
    RetryableEvaluationError,
    RetryableProviderError,
)
from study.scheduling import MPSLease
from study.serialization import (
    atomic_write_json,
    content_hash,
    create_json_exclusive,
    read_json,
    require_bool,
    require_int,
)


class RunStateError(RuntimeError):
    """Persisted run state does not agree with its frozen assignment."""


class CommonStudyEngine:
    """Execute either treatment policy without condition-specific control flow."""

    def __init__(
        self,
        *,
        study: StudySpec,
        run: RunSpec,
        generator: ProposalGenerator,
        evaluator: CandidateEvaluator,
        evaluation_lease_path: str | Path | None = None,
    ) -> None:
        if run.study_id != study.study_id:
            raise ValueError("run and study identifiers differ")
        canonical_condition = type(run.condition).for_id(run.condition.condition_id)
        if run.condition != canonical_condition:
            raise ValueError("run uses a noncanonical treatment definition")
        self.study = study
        self.run = run
        self.generator = generator
        self.evaluator = evaluator
        self.run_directory = Path(run.run_directory)
        self.state_path = self.run_directory / "run_state.json"
        self.evaluation_lease_path = (
            None if evaluation_lease_path is None else Path(evaluation_lease_path)
        )

    def execute(self) -> RunState:
        self._prepare_directory()
        state, ledger = self._load_or_initialize()
        if state.status == "completed":
            return state
        if state.seed_evaluation is None:
            seed_result = self._evaluate_seed()
            ledger.record_seed_evaluation(
                training_attempts=seed_result.training_attempts,
                training_steps=seed_result.training_steps,
                training_examples=seed_result.training_examples,
                mps_seconds=seed_result.mps_seconds,
                evaluation_cases=seed_result.evaluation_cases,
            )
            for _ in range(seed_result.infrastructure_retries):
                ledger.record_infrastructure_retry(require_active=False)
            state.seed_evaluation = seed_result.to_dict()
            self._persist(state, ledger)

        while ledger.terminal_opportunities < self.study.budget.proposal_opportunities:
            if state.active_opportunity is None:
                self._begin_opportunity(state, ledger)
                self._persist(state, ledger)
            self._advance_active_opportunity(state, ledger)
            self._persist(state, ledger)

        state.status = "completed"
        self._persist(state, ledger)
        return state

    def _prepare_directory(self) -> None:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        markers = {
            "run_spec.json": self.run.to_dict(),
            "study_spec.json": self.study.to_dict(),
        }
        for filename, expected in markers.items():
            path = self.run_directory / filename
            if path.exists():
                if read_json(path) != expected:
                    raise RunStateError(f"{filename} changed or directory collided")
            else:
                create_json_exclusive(path, expected)

    def _load_or_initialize(self) -> tuple[RunState, BudgetLedger]:
        if self.state_path.exists():
            state = RunState.from_dict(read_json(self.state_path))
            ledger = BudgetLedger.from_dict(state.ledger)
            self._validate_state(state, ledger)
            return state, ledger
        ledger = BudgetLedger(self.study.budget)
        state = RunState(
            study_id=self.study.study_id,
            block_id=self.run.block_id,
            run_id=self.run.run_id,
            condition_id=self.run.condition.condition_id.value,
            assignment_hash=self.run.assignment_hash,
            status="running",
            initial_candidate_id=self.study.initial_candidate_id,
            incumbent_id=self.study.initial_candidate_id,
            portfolio_ids=[self.study.initial_candidate_id],
            seed_evaluation=None,
            next_opportunity=1,
            active_opportunity=None,
            terminal_opportunities=[],
            ledger=ledger.to_dict(),
        )
        self._persist(state, ledger)
        return state, ledger

    def _validate_state(self, state: RunState, ledger: BudgetLedger) -> None:
        expected = (
            self.study.study_id,
            self.run.block_id,
            self.run.run_id,
            self.run.condition.condition_id.value,
            self.run.assignment_hash,
        )
        actual = (
            state.study_id,
            state.block_id,
            state.run_id,
            state.condition_id,
            state.assignment_hash,
        )
        if actual != expected:
            raise RunStateError("stored state belongs to a different frozen assignment")
        if ledger.spec.to_dict() != self.study.budget.to_dict():
            raise RunStateError("stored budget differs from the frozen study budget")
        if len(state.terminal_opportunities) != ledger.terminal_opportunities:
            raise RunStateError("terminal records and budget ledger disagree")
        if state.next_opportunity != ledger.terminal_opportunities + 1:
            raise RunStateError("next opportunity is not sequential")
        state_active = None
        if state.active_opportunity is not None:
            active = state.active_opportunity
            required_keys = {
                "opportunity_index",
                "parent_ids",
                "transition_active",
                "provider_attempts",
                "repairs",
                "previous_response",
                "proposal",
                "candidate_id",
                "evaluation",
                "started_at",
            }
            if set(active) != required_keys:
                raise RunStateError("active opportunity has invalid fields")
            try:
                state_active = require_int(
                    active["opportunity_index"], "opportunity_index"
                )
                transition_active = require_bool(
                    active["transition_active"], "transition_active"
                )
                provider_attempts = require_int(
                    active["provider_attempts"], "provider_attempts"
                )
                repairs = require_int(active["repairs"], "repairs")
            except ValueError as error:
                raise RunStateError(str(error)) from error
            if provider_attempts < 0 or repairs < 0:
                raise RunStateError("active opportunity counters cannot be negative")
            parent_ids = active["parent_ids"]
            if not isinstance(parent_ids, list) or any(
                not isinstance(value, str) or not value for value in parent_ids
            ):
                raise RunStateError("active parent_ids must be non-empty strings")
            expected_parents = (
                (state.incumbent_id,)
                if self.run.condition.parent_policy is ParentPolicy.SINGLE
                else tuple(
                    state.portfolio_ids[-self.study.portfolio_size :]
                )
            )
            if tuple(parent_ids) != expected_parents:
                raise RunStateError(
                    "active parents differ from the frozen parent policy"
                )
            expected_transition = (
                self.run.condition.proposal_policy
                is ProposalPolicy.SCHEDULED_TRANSITION
                and state_active in self.study.transition_opportunities
            )
            if transition_active is not expected_transition:
                raise RunStateError(
                    "active transition differs from the frozen proposal policy"
                )
            ledger_payload = ledger.to_dict()
            if provider_attempts != ledger_payload[
                "provider_attempts_by_opportunity"
            ].get(str(state_active), 0):
                raise RunStateError(
                    "active provider attempts and budget ledger disagree"
                )
            if repairs != ledger_payload["repairs_by_opportunity"].get(
                str(state_active), 0
            ):
                raise RunStateError("active repairs and budget ledger disagree")
        if state_active != ledger.active_opportunity:
            raise RunStateError("active opportunity and budget ledger disagree")
        if state.status not in {"running", "completed"}:
            raise RunStateError(f"unknown run status {state.status}")
        if state.status == "completed" and (
            ledger.terminal_opportunities != self.study.budget.proposal_opportunities
            or ledger.active_opportunity is not None
        ):
            raise RunStateError("completed state has unfinished opportunities")

    def _persist(self, state: RunState, ledger: BudgetLedger) -> None:
        self._validate_state(state, ledger)
        state.ledger = ledger.to_dict()
        state.state_revision += 1
        state.updated_at = utc_now()
        atomic_write_json(self.state_path, state.to_dict())

    def _evaluation_lease(self) -> ContextManager[Any]:
        if self.evaluation_lease_path is None:
            return nullcontext()
        return MPSLease(self.evaluation_lease_path, run_id=self.run.run_id)

    def _evaluate_seed(self) -> EvaluationResult:
        with self._evaluation_lease():
            return self.evaluator.evaluate_seed(
                self.study.initial_candidate_id,
                self.run.run_seed,
            )

    def _begin_opportunity(self, state: RunState, ledger: BudgetLedger) -> None:
        index = state.next_opportunity
        ledger.begin_opportunity(index)
        if self.run.condition.parent_policy is ParentPolicy.SINGLE:
            parents = (state.incumbent_id,)
        else:
            parents = tuple(state.portfolio_ids[-self.study.portfolio_size :])
        transition_active = (
            self.run.condition.proposal_policy
            is ProposalPolicy.SCHEDULED_TRANSITION
            and index in self.study.transition_opportunities
        )
        state.active_opportunity = {
            "opportunity_index": index,
            "parent_ids": list(parents),
            "transition_active": transition_active,
            "provider_attempts": 0,
            "repairs": 0,
            "previous_response": None,
            "proposal": None,
            "candidate_id": None,
            "evaluation": None,
            "started_at": utc_now(),
        }

    def _proposal_context(
        self, state: RunState, *, provider_attempt: int
    ) -> ProposalContext:
        active = self._active(state)
        return ProposalContext(
            study_id=self.study.study_id,
            block_id=self.run.block_id,
            run_id=self.run.run_id,
            run_seed=self.run.run_seed,
            condition=self.run.condition,
            opportunity_index=active["opportunity_index"],
            provider_attempt=provider_attempt,
            parent_ids=tuple(active["parent_ids"]),
            transition_active=active["transition_active"],
            repair=active["repairs"] > 0,
            previous_response=(
                None
                if active.get("previous_response") is None
                else str(active["previous_response"])
            ),
        )

    def _advance_active_opportunity(
        self, state: RunState, ledger: BudgetLedger
    ) -> None:
        active = self._active(state)
        if active["proposal"] is None:
            while active["proposal"] is None:
                if (
                    int(active["provider_attempts"])
                    >= self.study.budget.provider_attempts_per_opportunity
                ):
                    self._finish(
                        state,
                        ledger,
                        OpportunityOutcome.INFRASTRUCTURE_FAILURE,
                        failure_stage="provider_attempts_exhausted",
                    )
                    return
                attempt = ledger.start_provider_attempt()
                active["provider_attempts"] = attempt
                # Persist the consumed attempt before making the external request. A
                # process crash can retry the opportunity, but never replay this attempt.
                self._persist(state, ledger)
                try:
                    proposal = self.generator.generate(
                        self._proposal_context(state, provider_attempt=attempt)
                    )
                except RetryableProviderError:
                    if attempt < self.study.budget.provider_attempts_per_opportunity:
                        try:
                            ledger.record_infrastructure_retry()
                        except BudgetExceeded:
                            self._finish(
                                state,
                                ledger,
                                OpportunityOutcome.INFRASTRUCTURE_FAILURE,
                                failure_stage="infrastructure_retry_budget_exhausted",
                            )
                            return
                        self._persist(state, ledger)
                        continue
                    self._finish(
                        state,
                        ledger,
                        OpportunityOutcome.INFRASTRUCTURE_FAILURE,
                        failure_stage="provider_attempts_exhausted",
                    )
                    return
                ledger.record_provider_usage(
                    prompt_tokens=proposal.prompt_tokens,
                    completion_tokens=proposal.completion_tokens,
                )
                if proposal.candidate_source is None:
                    ledger.record_parse_failure()
                    can_request_again = (
                        attempt < self.study.budget.provider_attempts_per_opportunity
                    )
                    if can_request_again:
                        try:
                            ledger.record_repair()
                        except BudgetExceeded:
                            can_request_again = False
                    if can_request_again:
                        active["repairs"] = int(active.get("repairs", 0)) + 1
                        active["previous_response"] = proposal.response_text
                        self._persist(state, ledger)
                        continue
                active["proposal"] = proposal.to_dict()
                self._persist(state, ledger)

        active = self._active(state)
        proposal = ProposalResult.from_dict(active["proposal"])
        if proposal.candidate_source is None:
            self._finish(
                state,
                ledger,
                OpportunityOutcome.INVALID,
                failure_stage="proposal_parse",
            )
            return

        if active["candidate_id"] is None:
            candidate_id = content_hash(proposal.candidate_source)
            ledger.record_candidate_source(candidate_id)
            active["candidate_id"] = candidate_id
            self._persist(state, ledger)

        if active["evaluation"] is None:
            while active["evaluation"] is None:
                try:
                    with self._evaluation_lease():
                        evaluation = self.evaluator.evaluate_candidate(
                            proposal.candidate_source,
                            candidate_id=str(active["candidate_id"]),
                            opportunity_index=int(active["opportunity_index"]),
                            run_seed=self.run.run_seed,
                        )
                except RetryableEvaluationError:
                    try:
                        ledger.record_infrastructure_retry()
                    except BudgetExceeded:
                        self._finish(
                            state,
                            ledger,
                            OpportunityOutcome.INFRASTRUCTURE_FAILURE,
                            failure_stage="infrastructure_retry_budget_exhausted",
                        )
                        return
                    self._persist(state, ledger)
                    continue
                ledger.record_training(
                    attempts=evaluation.training_attempts,
                    steps=evaluation.training_steps,
                    examples=evaluation.training_examples,
                    mps_seconds=evaluation.mps_seconds,
                )
                ledger.record_evaluation(evaluation.evaluation_cases)
                for _ in range(evaluation.infrastructure_retries):
                    ledger.record_infrastructure_retry()
                active["evaluation"] = evaluation.to_dict()
                self._persist(state, ledger)

        evaluation = EvaluationResult.from_dict(self._active(state)["evaluation"])
        self._finish(
            state,
            ledger,
            evaluation.outcome,
            failure_stage=evaluation.failure_stage,
        )

    def _finish(
        self,
        state: RunState,
        ledger: BudgetLedger,
        outcome: OpportunityOutcome,
        *,
        failure_stage: str,
    ) -> None:
        active = self._active(state)
        ledger.finish_opportunity(outcome)
        if outcome is OpportunityOutcome.ACCEPTED:
            candidate_id = str(active["candidate_id"])
            state.incumbent_id = candidate_id
            if candidate_id not in state.portfolio_ids:
                state.portfolio_ids.append(candidate_id)
        terminal = {
            **active,
            "outcome": outcome.value,
            "failure_stage": failure_stage,
            "completed_at": utc_now(),
        }
        state.terminal_opportunities.append(terminal)
        state.active_opportunity = None
        state.next_opportunity += 1

    @staticmethod
    def _active(state: RunState) -> dict[str, Any]:
        if state.active_opportunity is None:
            raise RunStateError("run has no active opportunity")
        return state.active_opportunity
