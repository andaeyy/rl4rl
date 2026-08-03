"""Reproducible sparse and overdispersed run-count power simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from analysis.count_models import estimate_nb2_dispersion
from analysis.outcomes import (
    PilotDataset,
    RunOutcome,
    RunOutcomeTable,
    RunTerminalStatus,
)
from analysis.randomization_inference import Alternative, blocked_randomization_test


@dataclass(frozen=True)
class PilotCountParameters:
    condition_id: str
    assigned_runs: int
    mean_clusters: float
    dispersion_nb2: float
    zero_fraction: float


def estimate_pilot_count_parameters(
    pilot: PilotDataset,
    *,
    condition_id: str,
) -> PilotCountParameters:
    rows = pilot.outcomes.for_condition(condition_id)
    if len(rows) < 2:
        raise ValueError("pilot estimation requires at least two assigned runs")
    counts = tuple(row.itt_cluster_count for row in rows)
    return PilotCountParameters(
        condition_id=condition_id,
        assigned_runs=len(rows),
        mean_clusters=sum(counts) / len(counts),
        dispersion_nb2=estimate_nb2_dispersion(counts),
        zero_fraction=sum(count == 0 for count in counts) / len(counts),
    )


@dataclass(frozen=True)
class PowerSimulationSpec:
    """Paired-block count-generating process for one primary contrast."""

    blocks: int
    baseline_rate: float
    rate_ratio: float
    dispersion_nb2: float
    zero_inflation: float
    block_log_standard_deviation: float
    simulations: int
    randomization_draws: int
    alpha: float
    target_power: float
    seed: int
    target_condition: str = "target"
    reference_condition: str = "reference"
    proposal_exposure: int = 100
    token_exposure: int = 100_000

    def __post_init__(self) -> None:
        if self.blocks < 1 or self.simulations < 1 or self.randomization_draws < 1:
            raise ValueError("blocks, simulations, and randomization draws must be positive")
        if not math.isfinite(self.baseline_rate) or self.baseline_rate <= 0:
            raise ValueError("baseline_rate must be finite and positive")
        if not math.isfinite(self.rate_ratio) or self.rate_ratio <= 0:
            raise ValueError("rate_ratio must be finite and positive")
        if not math.isfinite(self.dispersion_nb2) or self.dispersion_nb2 < 0:
            raise ValueError("dispersion_nb2 must be finite and non-negative")
        if not 0 <= self.zero_inflation < 1:
            raise ValueError("zero_inflation must lie in [0, 1)")
        if self.block_log_standard_deviation < 0:
            raise ValueError("block_log_standard_deviation cannot be negative")
        if not 0 < self.alpha < 1 or not 0 < self.target_power < 1:
            raise ValueError("alpha and target_power must lie in (0, 1)")
        if self.target_condition == self.reference_condition:
            raise ValueError("target and reference conditions must differ")
        if self.proposal_exposure < 1 or self.token_exposure < 1:
            raise ValueError("simulated exposure must be positive")


@dataclass(frozen=True)
class PowerEstimate:
    blocks: int
    simulations: int
    rejections: int
    estimated_power: float
    monte_carlo_standard_error: float
    seed: int


@dataclass(frozen=True)
class SampleSizeSelection:
    target_power: float
    selected_blocks: int | None
    estimates: tuple[PowerEstimate, ...]


def _draw_count(
    generator: np.random.Generator,
    *,
    mean: float,
    dispersion: float,
    zero_inflation: float,
) -> int:
    if generator.random() < zero_inflation:
        return 0
    latent_mean = mean
    if dispersion > 0:
        latent_mean = float(
            generator.gamma(shape=1.0 / dispersion, scale=mean * dispersion)
        )
    return int(generator.poisson(latent_mean))


def simulate_run_outcomes(
    spec: PowerSimulationSpec,
    *,
    generator: np.random.Generator | None = None,
) -> RunOutcomeTable:
    """Draw one paired-block run-level dataset from Poisson or NB2 counts."""

    rng = generator if generator is not None else np.random.default_rng(spec.seed)
    rows: list[RunOutcome] = []
    for block_index in range(spec.blocks):
        block_effect = float(
            rng.lognormal(
                mean=-(spec.block_log_standard_deviation**2) / 2.0,
                sigma=spec.block_log_standard_deviation,
            )
        )
        reference_mean = spec.baseline_rate * block_effect
        target_mean = reference_mean * spec.rate_ratio
        for arm_index, (condition_id, mean) in enumerate(
            (
                (spec.reference_condition, reference_mean),
                (spec.target_condition, target_mean),
            )
        ):
            count = _draw_count(
                rng,
                mean=mean,
                dispersion=spec.dispersion_nb2,
                zero_inflation=spec.zero_inflation,
            )
            rows.append(
                RunOutcome(
                    study_id="synthetic-power-study",
                    block_id=f"block-{block_index:04d}",
                    run_id=f"block-{block_index:04d}-{condition_id}",
                    condition_id=condition_id,
                    run_seed=spec.seed + block_index * 2 + arm_index,
                    terminal_status=RunTerminalStatus.COMPLETED,
                    qualifying_cluster_count=count,
                    proposal_exposure=spec.proposal_exposure,
                    token_exposure=spec.token_exposure,
                )
            )
    return RunOutcomeTable(
        tuple(rows),
        tuple(row.run_id for row in rows),
    )


def simulate_power(spec: PowerSimulationSpec) -> PowerEstimate:
    """Estimate power of the preregisterable blocked randomization contrast."""

    rng = np.random.default_rng(spec.seed)
    rejections = 0
    for simulation_index in range(spec.simulations):
        outcomes = simulate_run_outcomes(spec, generator=rng)
        inference_seed = int(rng.integers(0, 2**63 - 1))
        result = blocked_randomization_test(
            outcomes,
            target_condition=spec.target_condition,
            reference_condition=spec.reference_condition,
            alternative=Alternative.GREATER,
            draws=spec.randomization_draws,
            seed=inference_seed,
            # Force bounded Monte Carlo work for simulation-based planning.
            exact_assignment_limit=1,
        )
        rejections += result.p_value <= spec.alpha
    power = rejections / spec.simulations
    return PowerEstimate(
        blocks=spec.blocks,
        simulations=spec.simulations,
        rejections=rejections,
        estimated_power=power,
        monte_carlo_standard_error=math.sqrt(
            power * (1.0 - power) / spec.simulations
        ),
        seed=spec.seed,
    )


def select_sample_size(
    template: PowerSimulationSpec,
    candidate_block_counts: Iterable[int],
) -> SampleSizeSelection:
    """Choose the first preregistered block count reaching target power."""

    candidates = tuple(sorted(set(candidate_block_counts)))
    if not candidates or any(value < 1 for value in candidates):
        raise ValueError("candidate block counts must be positive")
    estimates: list[PowerEstimate] = []
    selected: int | None = None
    for blocks in candidates:
        candidate = replace(
            template,
            blocks=blocks,
            seed=template.seed + blocks * 1_000_003,
        )
        estimate = simulate_power(candidate)
        estimates.append(estimate)
        if selected is None and estimate.estimated_power >= template.target_power:
            selected = blocks
    return SampleSizeSelection(
        target_power=template.target_power,
        selected_blocks=selected,
        estimates=tuple(estimates),
    )
