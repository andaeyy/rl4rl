"""Blocked randomization inference at the complete-run level."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from enum import StrEnum

from analysis.outcomes import RunOutcomeTable


class Alternative(StrEnum):
    GREATER = "greater"
    LESS = "less"
    TWO_SIDED = "two_sided"


@dataclass(frozen=True)
class RandomizationTestResult:
    target_condition: str
    reference_condition: str
    blocks: int
    observed_mean_difference: float
    p_value: float
    assignments_evaluated: int
    exact: bool
    alternative: Alternative
    seed: int


def _paired_differences(
    outcomes: RunOutcomeTable,
    target_condition: str,
    reference_condition: str,
) -> tuple[float, ...]:
    differences: list[float] = []
    for block_id in sorted({row.block_id for row in outcomes.rows}):
        rows = outcomes.for_block(block_id)
        targets = [row for row in rows if row.condition_id == target_condition]
        references = [row for row in rows if row.condition_id == reference_condition]
        if not targets and not references:
            continue
        if len(targets) != 1 or len(references) != 1:
            raise ValueError(
                f"block {block_id} must have one assigned run in each contrast arm"
            )
        differences.append(
            float(targets[0].itt_cluster_count - references[0].itt_cluster_count)
        )
    if not differences:
        raise ValueError("no paired blocks found for the requested contrast")
    return tuple(differences)


def _as_extreme(
    permuted: float,
    observed: float,
    alternative: Alternative,
) -> bool:
    tolerance = 1e-12
    if alternative is Alternative.GREATER:
        return permuted >= observed - tolerance
    if alternative is Alternative.LESS:
        return permuted <= observed + tolerance
    return abs(permuted) >= abs(observed) - tolerance


def blocked_randomization_test(
    outcomes: RunOutcomeTable,
    *,
    target_condition: str,
    reference_condition: str,
    alternative: Alternative | str = Alternative.TWO_SIDED,
    draws: int = 9_999,
    seed: int = 0,
    exact_assignment_limit: int = 65_536,
) -> RandomizationTestResult:
    """Paired within-block treatment-label randomization test.

    Under the sharp null, swapping target and reference labels negates a block's
    observed difference.  Scientific and infrastructure failures remain present
    through their intent-to-treat zero outcomes.
    """

    if target_condition == reference_condition:
        raise ValueError("target and reference conditions must differ")
    if draws < 1 or exact_assignment_limit < 1:
        raise ValueError("draws and exact_assignment_limit must be positive")
    resolved_alternative = Alternative(alternative)
    differences = _paired_differences(
        outcomes, target_condition, reference_condition
    )
    observed = sum(differences) / len(differences)
    assignment_count = 2 ** len(differences)

    if assignment_count <= exact_assignment_limit:
        statistics = (
            sum(sign * difference for sign, difference in zip(signs, differences))
            / len(differences)
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        )
        extreme = sum(
            _as_extreme(value, observed, resolved_alternative)
            for value in statistics
        )
        p_value = extreme / assignment_count
        evaluated = assignment_count
        exact = True
    else:
        generator = random.Random(seed)
        extreme = 0
        for _ in range(draws):
            value = sum(
                difference if generator.getrandbits(1) else -difference
                for difference in differences
            ) / len(differences)
            extreme += _as_extreme(value, observed, resolved_alternative)
        p_value = (extreme + 1) / (draws + 1)
        evaluated = draws
        exact = False

    return RandomizationTestResult(
        target_condition=target_condition,
        reference_condition=reference_condition,
        blocks=len(differences),
        observed_mean_difference=observed,
        p_value=p_value,
        assignments_evaluated=evaluated,
        exact=exact,
        alternative=resolved_alternative,
        seed=seed,
    )
