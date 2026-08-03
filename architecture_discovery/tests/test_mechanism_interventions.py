from contextlib import contextmanager

from mechanism.fakes import toy_mechanism_plan
from mechanism.interventions import (
    TrustedInterventionRegistry,
    run_intervention_study,
)


class IntervenableToy:
    def __init__(self) -> None:
        self.route = 1.0


def test_disable_and_rescue_hooks_restore_state_and_record_recovery():
    plan = toy_mechanism_plan(seed_count=1)
    model = IntervenableToy()
    registry = TrustedInterventionRegistry()

    @contextmanager
    def route_hook(target, intervention):
        previous = target.route
        target.route = float(intervention.value)
        try:
            yield
        finally:
            target.route = previous

    registry.register("hook:route", route_hook)
    result = run_intervention_study(
        model,
        evaluator=lambda item: item.route,
        interventions=plan.interventions,
        rescues=plan.rescues,
        registry=registry,
    )
    direct = {item.intervention_id: item for item in result.interventions}
    assert result.baseline_metric == 1.0
    assert direct["intervention:disable"].metric_delta == -1.0
    rescue = result.rescues[0]
    assert rescue.disabled_metric == 0.0
    assert rescue.rescued_metric == 1.0
    assert rescue.recovered_delta == 1.0
    assert model.route == 1.0
