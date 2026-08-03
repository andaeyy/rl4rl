from dataclasses import replace

from analysis.power import (
    PowerSimulationSpec,
    select_sample_size,
    simulate_power,
    simulate_run_outcomes,
)


def _power_spec(**overrides: object) -> PowerSimulationSpec:
    values = {
        "blocks": 12,
        "baseline_rate": 0.35,
        "rate_ratio": 1.0,
        "dispersion_nb2": 1.0,
        "zero_inflation": 0.20,
        "block_log_standard_deviation": 0.35,
        "simulations": 80,
        "randomization_draws": 249,
        "alpha": 0.05,
        "target_power": 0.70,
        "seed": 90210,
    }
    values.update(overrides)
    return PowerSimulationSpec(**values)  # type: ignore[arg-type]


def test_sparse_overdispersed_simulation_is_run_level_and_reproducible() -> None:
    spec = _power_spec()
    first = simulate_run_outcomes(spec)
    second = simulate_run_outcomes(spec)
    assert first == second
    assert len(first.rows) == spec.blocks * 2
    assert len({row.run_id for row in first.rows}) == len(first.rows)
    assert any(row.itt_cluster_count == 0 for row in first.rows)


def test_power_simulation_reproduces_and_detects_a_known_large_effect() -> None:
    null = _power_spec()
    null_first = simulate_power(null)
    null_second = simulate_power(null)
    assert null_first == null_second
    assert null_first.estimated_power <= 0.10

    effect = simulate_power(replace(null, rate_ratio=12.0))
    assert effect.estimated_power > null_first.estimated_power
    assert effect.estimated_power >= 0.60


def test_sample_size_selection_returns_first_power_qualified_design() -> None:
    template = _power_spec(
        rate_ratio=15.0,
        simulations=60,
        randomization_draws=299,
        target_power=0.60,
    )
    selection = select_sample_size(template, (12, 4, 8))
    assert tuple(estimate.blocks for estimate in selection.estimates) == (4, 8, 12)
    qualifying = [
        estimate.blocks
        for estimate in selection.estimates
        if estimate.estimated_power >= selection.target_power
    ]
    assert selection.selected_blocks == (qualifying[0] if qualifying else None)
