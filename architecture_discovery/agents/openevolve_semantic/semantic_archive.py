"""Stable categorical cells and rare-family parent sampling for OpenEvolve."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from common.descriptor_schema import SEMANTIC_METRIC_NAMES


@dataclass
class SemanticArchive:
    per_axis: dict[str, dict[int, set[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set))
    )
    pairwise: dict[tuple[str, str], dict[tuple[int, int], set[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set))
    )
    signatures: dict[tuple[int, ...], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )

    def add(self, program_id: str, metrics: dict[str, float], axes: Iterable[str]) -> None:
        selected = list(axes)
        values = [int(round(metrics[axis])) for axis in selected]
        for axis, value in zip(selected, values):
            self.per_axis[axis][value].add(program_id)
        for left_index, left in enumerate(selected):
            for right_index in range(left_index + 1, len(selected)):
                right = selected[right_index]
                cell = (values[left_index], values[right_index])
                self.pairwise[(left, right)][cell].add(program_id)
        self.signatures[tuple(values)].add(program_id)


def install_semantic_archive() -> None:
    """Patch OpenEvolve to keep categorical codes stable across a run."""
    from openevolve.database import ProgramDatabase

    if getattr(ProgramDatabase, "_semantic_archive_installed", False):
        return

    original_coords = ProgramDatabase._calculate_feature_coords
    original_exploration = ProgramDatabase._sample_exploration_parent
    original_add = ProgramDatabase.add

    def fixed_coords(self, program):
        dimensions = self.config.feature_dimensions
        if not dimensions or not all(name.startswith("semantic_") for name in dimensions):
            return original_coords(self, program)
        coordinates: list[int] = []
        for name in dimensions:
            if name not in program.metrics:
                raise ValueError(f"candidate lacks semantic descriptor {name}")
            bins = self.feature_bins_per_dim.get(name, self.feature_bins)
            code = int(round(program.metrics[name]))
            coordinates.append(max(0, min(bins - 1, code)))
        return coordinates

    def rare_family_parent(self):
        dimensions = self.config.feature_dimensions
        if not dimensions or not all(name.startswith("semantic_") for name in dimensions):
            return original_exploration(self)
        identifiers = [
            identifier
            for identifier in self.islands[self.current_island]
            if identifier in self.programs
        ]
        if not identifiers:
            return original_exploration(self)
        counts = {
            name: Counter(
                int(round(self.programs[identifier].metrics.get(name, 0.0)))
                for identifier in identifiers
            )
            for name in dimensions
        }
        weights: list[float] = []
        for identifier in identifiers:
            program = self.programs[identifier]
            rarity = sum(
                1.0 / counts[name][int(round(program.metrics.get(name, 0.0)))]
                for name in dimensions
            )
            weights.append(rarity)
        return random.choices(
            [self.programs[identifier] for identifier in identifiers],
            weights=weights,
            k=1,
        )[0]

    def semantic_add(self, program, iteration=None, target_island=None):
        program_id = original_add(
            self,
            program,
            iteration=iteration,
            target_island=target_island,
        )
        dimensions = self.config.feature_dimensions
        if dimensions and all(name.startswith("semantic_") for name in dimensions):
            sidecar = getattr(self, "semantic_coverage", None)
            if sidecar is None:
                sidecar = SemanticArchive()
                self.semantic_coverage = sidecar
            all_axes = list(SEMANTIC_METRIC_NAMES.values())
            sidecar.add(program.id, program.metrics, all_axes)
            values = [int(round(program.metrics[name])) for name in all_axes]
            program.metadata["semantic_signature"] = values
        return program_id

    ProgramDatabase._calculate_feature_coords = fixed_coords
    ProgramDatabase._sample_exploration_parent = rare_family_parent
    ProgramDatabase.add = semantic_add
    ProgramDatabase._semantic_archive_installed = True


SELECTED_ONLINE_AXES = [
    SEMANTIC_METRIC_NAMES["token_representation"],
    SEMANTIC_METRIC_NAMES["positional_integration"],
    SEMANTIC_METRIC_NAMES["attention_organization"],
    SEMANTIC_METRIC_NAMES["depth_topology"],
]
