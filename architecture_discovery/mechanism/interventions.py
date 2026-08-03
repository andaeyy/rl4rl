"""Trusted inference intervention hooks and preregistered rescue execution."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable

from mechanism.plans import InterventionSpec, RescueSpec
from mechanism.validation import require_finite


InterventionHook = Callable[
    [Any, InterventionSpec], AbstractContextManager[None]
]
MetricEvaluator = Callable[[Any], float]


class TrustedInterventionRegistry:
    """Evaluator-owned registry; plan data selects hooks but cannot define code."""

    def __init__(self) -> None:
        self._hooks: dict[str, InterventionHook] = {}

    def register(self, hook_id: str, hook: InterventionHook) -> None:
        if hook_id in self._hooks:
            raise ValueError(f"intervention hook {hook_id!r} is already registered")
        if not callable(hook):
            raise TypeError("intervention hook must be callable")
        self._hooks[hook_id] = hook

    def apply(
        self, model: Any, intervention: InterventionSpec
    ) -> AbstractContextManager[None]:
        try:
            hook = self._hooks[intervention.hook_id]
        except KeyError as error:
            raise ValueError(
                f"intervention uses unregistered hook {intervention.hook_id!r}"
            ) from error
        context = hook(model, intervention)
        if not isinstance(context, AbstractContextManager):
            raise TypeError("trusted intervention hook must return a context manager")
        return context


@dataclass(frozen=True)
class InterventionOutcome:
    intervention_id: str
    baseline_metric: float
    intervened_metric: float
    metric_delta: float


@dataclass(frozen=True)
class RescueOutcome:
    rescue_id: str
    baseline_metric: float
    disabled_metric: float
    rescued_metric: float
    disabled_delta: float
    recovered_delta: float


@dataclass(frozen=True)
class InterventionStudyResult:
    baseline_metric: float
    interventions: tuple[InterventionOutcome, ...]
    rescues: tuple[RescueOutcome, ...]


def run_intervention_study(
    model: Any,
    *,
    evaluator: MetricEvaluator,
    interventions: tuple[InterventionSpec, ...],
    rescues: tuple[RescueSpec, ...],
    registry: TrustedInterventionRegistry,
) -> InterventionStudyResult:
    """Run direct interventions and nested disable/restore rescue tests."""

    intervention_by_id = {item.intervention_id: item for item in interventions}
    if len(intervention_by_id) != len(interventions):
        raise ValueError("intervention IDs must be unique")
    baseline = require_finite(evaluator(model), "baseline metric")
    outcomes: list[InterventionOutcome] = []
    for intervention in interventions:
        with registry.apply(model, intervention):
            value = require_finite(
                evaluator(model), f"metric for {intervention.intervention_id}"
            )
        outcomes.append(
            InterventionOutcome(
                intervention_id=intervention.intervention_id,
                baseline_metric=baseline,
                intervened_metric=value,
                metric_delta=value - baseline,
            )
        )

    rescue_outcomes: list[RescueOutcome] = []
    for rescue in rescues:
        try:
            disabling = intervention_by_id[rescue.disabling_intervention_id]
            restoring = intervention_by_id[rescue.restoring_intervention_id]
        except KeyError as error:
            raise ValueError("rescue references an undeclared intervention") from error
        with registry.apply(model, disabling):
            disabled = require_finite(
                evaluator(model), f"disabled metric for {rescue.rescue_id}"
            )
            with registry.apply(model, restoring):
                rescued = require_finite(
                    evaluator(model), f"rescued metric for {rescue.rescue_id}"
                )
        rescue_outcomes.append(
            RescueOutcome(
                rescue_id=rescue.rescue_id,
                baseline_metric=baseline,
                disabled_metric=disabled,
                rescued_metric=rescued,
                disabled_delta=disabled - baseline,
                recovered_delta=rescued - disabled,
            )
        )

    final_metric = require_finite(evaluator(model), "post-intervention metric")
    if final_metric != baseline:
        raise RuntimeError("intervention hook did not restore model state")
    return InterventionStudyResult(
        baseline_metric=baseline,
        interventions=tuple(outcomes),
        rescues=tuple(rescue_outcomes),
    )
