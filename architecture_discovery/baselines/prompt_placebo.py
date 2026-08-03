"""Prompt-placebo control contract, deliberately separate from no-search."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptPlaceboSpec:
    """Adaptive controller control with a placebo prompt manipulation.

    This is not a no-search baseline: it may retain controller state and therefore
    must be assigned, analyzed, and reported as a distinct condition.
    """

    placebo_id: str
    prompt_template: str
    retains_search_feedback: bool = field(default=True, init=False)
    schema_name: str = field(default="PromptPlaceboSpec", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        if not self.placebo_id or not self.prompt_template.strip():
            raise ValueError("placebo_id and prompt_template cannot be empty")
