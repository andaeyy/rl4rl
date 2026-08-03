"""Frozen novelty labels used only after the search run has completed."""

from __future__ import annotations

from enum import StrEnum


class NoveltyLabel(StrEnum):
    N0 = "N0"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"
    N4 = "N4"
    X = "X"


NOVELTY_DEFINITIONS: dict[NoveltyLabel, str] = {
    NoveltyLabel.N0: "Known equivalent mechanism.",
    NoveltyLabel.N1: "Scale, parameter, or implementation variant of a known mechanism.",
    NoveltyLabel.N2: "Recombination of known mechanisms without a new causal mechanism.",
    NoveltyLabel.N3: "Causally distinct mechanism assembled from known primitives.",
    NoveltyLabel.N4: "No matching mechanism in the frozen reference corpus.",
    NoveltyLabel.X: "Unresolved with the available evidence.",
}


def definition(label: NoveltyLabel | str) -> str:
    return NOVELTY_DEFINITIONS[NoveltyLabel(label)]
