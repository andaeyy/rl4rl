"""Rule-based taxonomy suggestions for representational boundary crossings."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl4rl.schema import AnnotationSource, BoundaryLabel, EditAnnotation


@dataclass(frozen=True, slots=True)
class Taxonomy:
    version: str
    components: dict[str, dict[str, str]]

    @classmethod
    def load(cls, path: str | Path) -> Taxonomy:
        with Path(path).open("rb") as handle:
            value: dict[str, Any] = tomllib.load(handle)
        components = {
            str(component): {
                _normalize(technique): str(family)
                for technique, family in techniques.items()
            }
            for component, techniques in value.get("components", {}).items()
        }
        return cls(version=str(value.get("version", "unknown")), components=components)

    def family(self, component: str, technique: str | None) -> str | None:
        if technique is None:
            return None
        component_map = self.components.get(_normalize(component), {})
        return component_map.get(_normalize(technique))

    def suggest(
        self,
        *,
        component: str,
        operation: str,
        before: str | None,
        after: str | None,
    ) -> EditAnnotation:
        before_family = self.family(component, before)
        after_family = self.family(component, after)
        if before is None or after is None:
            label = BoundaryLabel.AMBIGUOUS
            rationale = "A substitution boundary needs both before and after states."
        elif before_family is None or after_family is None:
            label = BoundaryLabel.AMBIGUOUS
            rationale = "At least one technique is absent from the taxonomy."
        elif before_family == after_family:
            label = BoundaryLabel.PRESERVING
            rationale = f"Both techniques map to {before_family!r}."
        else:
            label = BoundaryLabel.CHANGING
            rationale = (
                f"Technique family changes from {before_family!r} to {after_family!r}."
            )
        return EditAnnotation(
            component=_normalize(component),
            operation=_normalize(operation),
            before=before,
            after=after,
            ontology_family_before=before_family,
            ontology_family_after=after_family,
            boundary_label=label,
            rationale=rationale,
            confidence=0.75 if label != BoundaryLabel.AMBIGUOUS else 0.25,
            annotation_source=AnnotationSource.HEURISTIC,
            needs_review=True,
        )


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
