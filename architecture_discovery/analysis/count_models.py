"""Run-level Poisson and NB2 likelihoods with blocked contrasts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Mapping

from analysis.outcomes import RunOutcomeTable, RunTerminalStatus


def _validated_counts_and_means(
    counts: Iterable[int], means: Iterable[float]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    observed = tuple(counts)
    expected = tuple(means)
    if not observed or len(observed) != len(expected):
        raise ValueError("counts and means must have equal non-zero length")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in observed
    ):
        raise ValueError("counts must be non-negative integers")
    if any(not math.isfinite(value) or value <= 0 for value in expected):
        raise ValueError("means must be finite and positive")
    return observed, expected


def poisson_log_likelihood(
    counts: Iterable[int], means: Iterable[float]
) -> float:
    """Dependency-free Poisson log likelihood for independent run counts."""

    observed, expected = _validated_counts_and_means(counts, means)
    return sum(
        count * math.log(mean) - mean - math.lgamma(count + 1)
        for count, mean in zip(observed, expected, strict=True)
    )


def negative_binomial_log_likelihood(
    counts: Iterable[int],
    means: Iterable[float],
    *,
    dispersion: float,
) -> float:
    """Dependency-free NB2 log likelihood.

    ``Var(Y) = mean + dispersion * mean**2``.  Dispersion zero is exactly
    Poisson, which makes the model comparison boundary explicit.
    """

    observed, expected = _validated_counts_and_means(counts, means)
    if not math.isfinite(dispersion) or dispersion < 0:
        raise ValueError("dispersion must be finite and non-negative")
    if dispersion == 0:
        return poisson_log_likelihood(observed, expected)
    size = 1.0 / dispersion
    total = 0.0
    for count, mean in zip(observed, expected, strict=True):
        probability = size / (size + mean)
        total += (
            math.lgamma(count + size)
            - math.lgamma(size)
            - math.lgamma(count + 1)
            + size * math.log(probability)
            + count * math.log1p(-probability)
        )
    return total


def estimate_nb2_dispersion(counts: Iterable[int]) -> float:
    """Method-of-moments pilot estimate, truncated at the Poisson boundary."""

    values = tuple(counts)
    if len(values) < 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise ValueError("at least two non-negative integer run counts are required")
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return max(0.0, (variance - mean) / (mean * mean))


def blocked_count_log_likelihood(
    outcomes: RunOutcomeTable,
    *,
    condition_log_rates: Mapping[str, float],
    block_log_effects: Mapping[str, float],
    dispersion: float = 0.0,
    exclude_infrastructure_failures: bool = False,
) -> float:
    """Evaluate a preregistered blocked Poisson or NB2 run-level model.

    This function evaluates a fully specified model and adds no optional fitting
    dependency.  Final coefficient fitting may use a separately pinned package,
    but its likelihood must agree with these tested formulas.
    """

    counts: list[int] = []
    means: list[float] = []
    for row in outcomes.rows:
        if (
            exclude_infrastructure_failures
            and row.terminal_status is RunTerminalStatus.INFRASTRUCTURE_FAILURE
        ):
            continue
        if row.condition_id not in condition_log_rates:
            raise ValueError(f"missing rate for condition {row.condition_id}")
        if row.block_id not in block_log_effects:
            raise ValueError(f"missing effect for block {row.block_id}")
        linear_predictor = (
            condition_log_rates[row.condition_id] + block_log_effects[row.block_id]
        )
        if not math.isfinite(linear_predictor):
            raise ValueError("model linear predictors must be finite")
        counts.append(row.itt_cluster_count)
        means.append(math.exp(linear_predictor))
    if not counts:
        raise ValueError("model contains no run observations")
    if dispersion == 0:
        return poisson_log_likelihood(counts, means)
    return negative_binomial_log_likelihood(
        counts, means, dispersion=dispersion
    )


@dataclass(frozen=True)
class BlockedRateRatioEstimate:
    target_condition: str
    reference_condition: str
    blocks: int
    target_clusters: int
    reference_clusters: int
    rate_ratio: float
    log_rate_ratio: float
    standard_error: float
    confidence_interval: tuple[float, float]
    continuity_correction: float


def estimate_blocked_rate_ratio(
    outcomes: RunOutcomeTable,
    *,
    target_condition: str,
    reference_condition: str,
    alpha: float = 0.05,
) -> BlockedRateRatioEstimate:
    """Estimate a paired-block rate ratio for one preregistered contrast."""

    if target_condition == reference_condition:
        raise ValueError("target and reference conditions must differ")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    block_ids = sorted({row.block_id for row in outcomes.rows})
    target_total = 0
    reference_total = 0
    paired_blocks = 0
    for block_id in block_ids:
        block = outcomes.for_block(block_id)
        target = [row for row in block if row.condition_id == target_condition]
        reference = [row for row in block if row.condition_id == reference_condition]
        if not target and not reference:
            continue
        if len(target) != 1 or len(reference) != 1:
            raise ValueError(
                f"block {block_id} must contain exactly one run per contrast arm"
            )
        target_total += target[0].itt_cluster_count
        reference_total += reference[0].itt_cluster_count
        paired_blocks += 1
    if paired_blocks == 0:
        raise ValueError("no complete contrast blocks found")

    correction = 0.5 if target_total == 0 or reference_total == 0 else 0.0
    adjusted_target = target_total + correction
    adjusted_reference = reference_total + correction
    log_ratio = math.log(adjusted_target / adjusted_reference)
    standard_error = math.sqrt(1.0 / adjusted_target + 1.0 / adjusted_reference)
    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    interval = (
        math.exp(log_ratio - critical * standard_error),
        math.exp(log_ratio + critical * standard_error),
    )
    return BlockedRateRatioEstimate(
        target_condition=target_condition,
        reference_condition=reference_condition,
        blocks=paired_blocks,
        target_clusters=target_total,
        reference_clusters=reference_total,
        rate_ratio=math.exp(log_ratio),
        log_rate_ratio=log_ratio,
        standard_error=standard_error,
        confidence_interval=interval,
        continuity_correction=correction,
    )
