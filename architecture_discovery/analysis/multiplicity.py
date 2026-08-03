"""Deterministic family-wise and false-discovery p-value adjustments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from analysis.plan import MultiplicityMethod


@dataclass(frozen=True)
class AdjustedPValue:
    hypothesis_id: str
    raw_p_value: float
    adjusted_p_value: float
    method: MultiplicityMethod


def adjust_pvalues(
    p_values: Mapping[str, float],
    *,
    method: MultiplicityMethod | str,
) -> tuple[AdjustedPValue, ...]:
    if not p_values:
        raise ValueError("a multiplicity family cannot be empty")
    if any(not hypothesis.strip() for hypothesis in p_values):
        raise ValueError("hypothesis IDs cannot be empty")
    if any(
        not math.isfinite(value) or value < 0 or value > 1
        for value in p_values.values()
    ):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    resolved = MultiplicityMethod(method)
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)

    if resolved is MultiplicityMethod.BONFERRONI:
        adjusted = {
            hypothesis: min(1.0, value * count)
            for hypothesis, value in ordered
        }
    elif resolved is MultiplicityMethod.HOLM:
        adjusted = {}
        running = 0.0
        for rank, (hypothesis, value) in enumerate(ordered):
            running = max(running, (count - rank) * value)
            adjusted[hypothesis] = min(1.0, running)
    else:
        adjusted = {}
        running = 1.0
        for reverse_rank in range(count - 1, -1, -1):
            hypothesis, value = ordered[reverse_rank]
            rank = reverse_rank + 1
            running = min(running, value * count / rank)
            adjusted[hypothesis] = min(1.0, running)

    return tuple(
        AdjustedPValue(
            hypothesis_id=hypothesis,
            raw_p_value=p_values[hypothesis],
            adjusted_p_value=adjusted[hypothesis],
            method=resolved,
        )
        for hypothesis in sorted(p_values)
    )
