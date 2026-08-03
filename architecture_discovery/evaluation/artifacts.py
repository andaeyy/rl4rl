"""Layer-specific artifact roots and safe JSON record storage."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.evaluation_profiles import EvaluationLayer
from evaluation.records import ArtifactReference, canonical_json


@dataclass(frozen=True)
class EvaluationArtifactRoots:
    base: Path
    layer_a: Path
    layer_b: Path
    layer_c: Path

    @classmethod
    def under(cls, base: str | Path) -> "EvaluationArtifactRoots":
        root = Path(base).resolve()
        result = cls(
            base=root,
            layer_a=root / "online" / "layer_a",
            layer_b=root / "sealed" / "layer_b",
            layer_c=root / "sealed" / "layer_c",
        )
        result.validate()
        return result

    def validate(self) -> None:
        roots = tuple(path.resolve() for path in (self.layer_a, self.layer_b, self.layer_c))
        if len(set(roots)) != 3:
            raise ValueError("evaluation layers require distinct artifact roots")
        for index, first in enumerate(roots):
            for second in roots[index + 1 :]:
                if first in second.parents or second in first.parents:
                    raise ValueError("one evaluation layer root cannot contain another")
        if self.base.resolve() not in self.layer_a.resolve().parents:
            raise ValueError("Layer A root escaped the artifact base")
        if self.base.resolve() not in self.layer_b.resolve().parents:
            raise ValueError("Layer B root escaped the artifact base")
        if self.base.resolve() not in self.layer_c.resolve().parents:
            raise ValueError("Layer C root escaped the artifact base")

    def prepare(self) -> None:
        self.validate()
        for root in (self.layer_a, self.layer_b, self.layer_c):
            root.mkdir(parents=True, exist_ok=True)

    def for_layer(self, layer: EvaluationLayer) -> Path:
        if layer is EvaluationLayer.SEARCH:
            return self.layer_a
        if layer is EvaluationLayer.QUALIFICATION:
            return self.layer_b
        if layer is EvaluationLayer.CONFIRMATION:
            return self.layer_c
        raise ValueError(f"unknown evaluation layer {layer!r}")


class JsonEvaluationArtifactStore:
    """Write records only inside one configured layer root."""

    def __init__(
        self,
        roots: EvaluationArtifactRoots,
        layer: EvaluationLayer,
    ) -> None:
        roots.validate()
        self._roots = roots
        self._layer = layer
        self._root = roots.for_layer(layer)

    @property
    def layer(self) -> EvaluationLayer:
        return self._layer

    @property
    def root(self) -> Path:
        return self._root

    def write_json(
        self, record_id: str, payload: dict[str, Any]
    ) -> ArtifactReference:
        if not record_id or any(character in record_id for character in ("/", "\\", "\x00")):
            raise ValueError("record_id is not safe for an artifact filename")
        self._root.mkdir(parents=True, exist_ok=True)
        destination = (self._root / f"{record_id}.json").resolve()
        if self._root.resolve() not in destination.parents:
            raise ValueError("artifact destination escaped its layer root")
        if destination.exists():
            raise FileExistsError(
                f"evaluation artifact already exists: {destination.name}"
            )
        serialized = canonical_json(payload) + "\n"
        data = serialized.encode("utf-8")
        temporary = self._root / f".{record_id}.{uuid.uuid4().hex}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise FileExistsError(
                    f"evaluation artifact already exists: {destination.name}"
                ) from error
        finally:
            if temporary.exists():
                temporary.unlink()
        return ArtifactReference(
            layer=self._layer,
            relative_path=destination.relative_to(self._root).as_posix(),
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def read_json(self, reference: ArtifactReference) -> dict[str, Any]:
        reference.validate(expected_layer=self._layer)
        candidate = (self._root / reference.relative_path).resolve()
        if self._root.resolve() not in candidate.parents:
            raise ValueError("artifact reference escaped its layer root")
        data = candidate.read_bytes()
        if hashlib.sha256(data).hexdigest() != reference.sha256:
            raise ValueError("artifact content hash mismatch")
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError("evaluation artifacts must contain a JSON object")
        return result
