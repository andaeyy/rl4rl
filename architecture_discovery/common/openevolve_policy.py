"""Validity-first replacement policy shared by both OpenEvolve conditions."""

from __future__ import annotations

import math
import random
from functools import cmp_to_key
from typing import Any


def _quality(metrics: dict[str, Any]) -> tuple[float, float]:
    """Rank only by frozen validity/accuracy fields; metadata is intentionally absent."""
    try:
        eligible_value = float(metrics.get("eligible_for_parent", 0.0))
    except (TypeError, ValueError):
        eligible_value = 0.0
    try:
        score_value = float(metrics.get("search_score", 0.0))
    except (TypeError, ValueError):
        score_value = 0.0
    eligible = 1.0 if math.isfinite(eligible_value) and eligible_value >= 0.5 else 0.0
    score = score_value if math.isfinite(score_value) else 0.0
    return (
        eligible,
        max(0.0, min(1.0, score)),
    )


def canonical_combined_score(metrics: dict[str, Any]) -> float:
    """Encode validity-first quality for vendor paths that require one scalar.

    Eligible candidates occupy ``[2, 3]`` and ineligible candidates occupy
    ``[0, 1]``.  The score therefore preserves the same lexicographic ordering
    as :func:`_quality` without reading descriptors or model-size metadata.
    """

    eligible, score = _quality(metrics)
    return 2.0 * eligible + score


def _eligible(program: Any) -> bool:
    return _quality(program.metrics)[0] == 1.0


def install_validity_first_policy() -> None:
    """Patch comparison and parent pools without modifying the vendor checkout."""
    from openevolve.database import ProgramDatabase

    if getattr(ProgramDatabase, "_discovery_quality_policy_installed", False):
        return

    original_add = ProgramDatabase.add
    original_random_parent = ProgramDatabase._sample_random_parent
    original_sample = ProgramDatabase.sample
    original_sample_from_island = ProgramDatabase.sample_from_island

    def validity_first(self, program1, program2):
        quality1 = _quality(program1.metrics)
        quality2 = _quality(program2.metrics)
        if quality1 != quality2:
            return quality1 > quality2

        # Earlier completion wins an exact tie. The ID makes the rule total and
        # deterministic when imported traces have equal timestamps.
        order1 = (program1.iteration_found, program1.timestamp, program1.id)
        order2 = (program2.iteration_found, program2.timestamp, program2.id)
        return order1 < order2

    def best_eligible(self, identifiers=None):
        allowed = None if identifiers is None else set(identifiers)
        best = None
        for program in self.programs.values():
            if allowed is not None and program.id not in allowed:
                continue
            if not _eligible(program):
                continue
            if best is None or validity_first(self, program, best):
                best = program
        return best

    def eligible_top_programs(self, n=10, metric=None, island_idx=None):
        """Return only reportable programs while preserving vendor API semantics."""

        if island_idx is not None and (
            island_idx < 0 or island_idx >= len(self.islands)
        ):
            raise IndexError(
                f"Island index {island_idx} is out of range "
                f"(0-{len(self.islands) - 1})"
            )
        identifiers = (
            None if island_idx is None else self.islands[island_idx]
        )
        allowed = None if identifiers is None else set(identifiers)
        candidates = [
            program
            for program in self.programs.values()
            if _eligible(program)
            and (allowed is None or program.id in allowed)
        ]
        if metric is not None:
            candidates = [
                program for program in candidates if metric in program.metrics
            ]
            candidates.sort(
                key=lambda program: program.metrics[metric],
                reverse=True,
            )
            return candidates[:n]

        def compare(left, right):
            if validity_first(self, left, right):
                return -1
            if validity_first(self, right, left):
                return 1
            return 0

        candidates.sort(key=cmp_to_key(compare))
        return candidates[:n]

    def eligible_best_program(self, metric=None):
        """Return no best program when every stored record is ineligible."""

        programs = eligible_top_programs(self, n=1, metric=metric)
        best = programs[0] if programs else None
        self.best_program_id = best.id if best is not None else None
        return best

    def sanitize_parent_pools(self):
        """Remove legacy/injected ineligible records from every active pool."""

        eligible_ids = {
            identifier
            for identifier, program in self.programs.items()
            if _eligible(program)
        }
        for island in self.islands:
            island.intersection_update(eligible_ids)
        self.archive.intersection_update(eligible_ids)
        for feature_map in self.island_feature_maps:
            for key, identifier in list(feature_map.items()):
                if identifier not in eligible_ids:
                    del feature_map[key]

        if self.best_program_id not in eligible_ids:
            best = best_eligible(self)
            self.best_program_id = best.id if best is not None else None
        for index, island in enumerate(self.islands):
            current = self.island_best_programs[index]
            if current not in eligible_ids or current not in island:
                best = best_eligible(self, island)
                self.island_best_programs[index] = (
                    best.id if best is not None else None
                )
        return eligible_ids

    def validity_gated_add(self, program, iteration=None, target_island=None):
        if _eligible(program):
            return original_add(
                self,
                program,
                iteration=iteration,
                target_island=target_island,
            )

        # Preserve failed candidates as records, but never place them in a MAP
        # cell, island, archive, best slot, or inspiration pool.
        if iteration is not None:
            program.iteration_found = iteration
            self.last_iteration = max(self.last_iteration, iteration)
        program.metadata["eligible_for_parent"] = False
        self.programs[program.id] = program
        if self.config.db_path:
            self._save_program(program)
        self._enforce_population_limit()
        return program.id

    def guarded_sample(self, num_inspirations=None):
        if not sanitize_parent_pools(self):
            raise RuntimeError("OpenEvolve has no eligible parent candidates")
        parent, inspirations = original_sample(self, num_inspirations)
        if not _eligible(parent):  # defensive check against future vendor changes
            raise RuntimeError("OpenEvolve selected an ineligible parent")
        return parent, [program for program in inspirations if _eligible(program)]

    def eligible_random_parent(self):
        candidates = [
            program for program in self.programs.values() if _eligible(program)
        ]
        if not candidates:
            # Preserve the vendor exception path and message for an actually
            # empty database; otherwise fail with the policy-specific reason.
            if not self.programs:
                return original_random_parent(self)
            raise RuntimeError("OpenEvolve has no eligible parent candidates")
        return random.choice(candidates)

    def guarded_sample_from_island(self, island_id, num_inspirations=None):
        if not sanitize_parent_pools(self):
            raise RuntimeError("OpenEvolve has no eligible parent candidates")
        parent, inspirations = original_sample_from_island(
            self,
            island_id,
            num_inspirations,
        )
        if not _eligible(parent):  # defensive check against future vendor changes
            raise RuntimeError("OpenEvolve selected an ineligible parent")
        return parent, [program for program in inspirations if _eligible(program)]

    ProgramDatabase._is_better = validity_first
    ProgramDatabase.add = validity_gated_add
    ProgramDatabase.get_best_program = eligible_best_program
    ProgramDatabase.get_top_programs = eligible_top_programs
    ProgramDatabase._sample_random_parent = eligible_random_parent
    ProgramDatabase.sample = guarded_sample
    ProgramDatabase.sample_from_island = guarded_sample_from_island
    ProgramDatabase._discovery_quality_policy_installed = True
