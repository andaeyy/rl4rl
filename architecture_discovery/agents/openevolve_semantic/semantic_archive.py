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
    ineligible_programs: set[str] = field(default_factory=set)

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
    # The semantic wrapper must sit outside the validity gate so failed
    # programs can be recorded as unknown without ever entering parent pools.
    # Installing the inner policy here makes patch order deterministic for
    # tests, offline tools, and native runners alike.
    from common.openevolve_policy import install_validity_first_policy
    from openevolve.database import ProgramDatabase

    install_validity_first_policy()
    if getattr(ProgramDatabase, "_semantic_archive_installed", False):
        return

    original_coords = ProgramDatabase._calculate_feature_coords
    original_exploration = ProgramDatabase._sample_exploration_parent
    original_island_random = ProgramDatabase._sample_from_island_random
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

    def rare_family_from_identifiers(self, identifiers):
        dimensions = self.config.feature_dimensions
        identifiers = [
            identifier
            for identifier in identifiers
            if identifier in self.programs
            and self.programs[identifier].metrics.get("eligible_for_parent", 0.0)
            >= 0.5
        ]
        if not identifiers:
            return None
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

    def rare_family_parent(self):
        dimensions = self.config.feature_dimensions
        if not dimensions or not all(name.startswith("semantic_") for name in dimensions):
            return original_exploration(self)
        selected = rare_family_from_identifiers(
            self,
            self.islands[self.current_island],
        )
        return selected if selected is not None else original_exploration(self)

    def rare_family_island_random(self, island_id):
        dimensions = self.config.feature_dimensions
        if not dimensions or not all(name.startswith("semantic_") for name in dimensions):
            return original_island_random(self, island_id)
        resolved_island = island_id % len(self.islands)
        selected = rare_family_from_identifiers(
            self,
            self.islands[resolved_island],
        )
        return selected if selected is not None else original_island_random(
            self,
            resolved_island,
        )

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
            coverage_eligible = (
                program.metrics.get("eligible_for_parent", 0.0) >= 0.5
            )
            if not coverage_eligible:
                # OpenEvolve itself may collapse an unexpected evaluator
                # exception or timeout to {"error": 0.0}, bypassing the normal
                # adapter. Keep that failure insertable and explicitly unknown.
                for name in all_axes:
                    program.metrics.setdefault(name, 0.0)
            values = [int(round(program.metrics[name])) for name in all_axes]
            program.metadata["semantic_signature"] = values
            program.metadata["semantic_coverage_eligible"] = coverage_eligible
            if coverage_eligible:
                sidecar.add(program.id, program.metrics, all_axes)
            else:
                # Preserve the failed record without pretending that the
                # all-unknown signature represents explored valid coverage.
                sidecar.ineligible_programs.add(program.id)
            if self.config.db_path:
                self._save_program(program)
        return program_id

    ProgramDatabase._calculate_feature_coords = fixed_coords
    ProgramDatabase._sample_exploration_parent = rare_family_parent
    ProgramDatabase._sample_from_island_random = rare_family_island_random
    ProgramDatabase.add = semantic_add
    ProgramDatabase._semantic_archive_installed = True


SELECTED_ONLINE_AXES = [
    SEMANTIC_METRIC_NAMES["token_representation"],
    SEMANTIC_METRIC_NAMES["positional_integration"],
    SEMANTIC_METRIC_NAMES["attention_organization"],
    SEMANTIC_METRIC_NAMES["depth_topology"],
]
