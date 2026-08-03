"""Treatment- and outcome-blinded novelty-review packet generation."""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from novelty.serialization import require_identifier, require_sha256, utc_now
from novelty.signatures import MechanismSignature


PACKET_SCHEMA_NAME = "BlindedNoveltyReviewPacket"
PACKET_SCHEMA_VERSION = "1.0"
_FORBIDDEN_KEYS = {
    "ancestry",
    "candidate_id",
    "condition",
    "condition_id",
    "controller",
    "outcome",
    "parameter_count",
    "parameterization_hash",
    "parent_id",
    "public_accuracy",
    "qualification_record_id",
    "run_id",
    "search_score",
    "signature_hash",
    "snapshot_sha256",
}
_FORBIDDEN_TEXT = (
    re.compile(r"\bC[0-3]\b", re.IGNORECASE),
    re.compile(r"\b(?:openevolve|autoresearch)\b", re.IGNORECASE),
    re.compile(r"\b(?:greedy|generic|semantic)\s+(?:agent|controller)\b", re.IGNORECASE),
    re.compile(r"\b(?:public|official|shadow|search)\s+(?:accuracy|score|result)\b", re.IGNORECASE),
    re.compile(r"\bparameter[_ -]?count\b", re.IGNORECASE),
    re.compile(r"\b(?:parent[_ -]?id|run[_ -]?id|condition[_ -]?id|ancestry)\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:k|m|b)?\s*(?:parameters|params)\b", re.IGNORECASE),
)


class ReviewLeakageError(ValueError):
    pass


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"review packets cannot contain {type(value).__name__}")


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def assert_blind_text(value: str, field_name: str) -> None:
    for pattern in _FORBIDDEN_TEXT:
        if pattern.search(value):
            raise ReviewLeakageError(
                f"{field_name} contains a treatment, controller, size, ancestry, or outcome cue"
            )


def _assert_blind_payload(value: Any, path: str = "packet") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_KEYS:
                raise ReviewLeakageError(f"{path} contains forbidden field {key!r}")
            _assert_blind_payload(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_blind_payload(item, f"{path}[{index}]")
    elif isinstance(value, str):
        assert_blind_text(value, path)


@dataclass(frozen=True)
class ReviewMaterial:
    """Post-search material accepted by the trusted blinding service.

    Its schema intentionally has no treatment, outcome, scale, ancestry, or
    controller field. The internal candidate ID is used only to produce an
    opaque ID and is stored in the separate blinding index.
    """

    candidate_id: str
    signature: MechanismSignature
    mechanism_summary: str
    causal_claim: str
    falsifiable_prediction: str
    nearest_reference_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.candidate_id, "candidate_id")
        for field_name in (
            "mechanism_summary",
            "causal_claim",
            "falsifiable_prediction",
        ):
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"{field_name} cannot be empty")
            assert_blind_text(value, field_name)
        references = tuple(sorted(set(self.nearest_reference_ids)))
        for reference_id in references:
            require_identifier(reference_id, "nearest_reference_id")
        object.__setattr__(self, "nearest_reference_ids", references)


@dataclass(frozen=True)
class BlindedReviewPacket:
    packet_id: str
    blinded_candidate_id: str
    corpus_sha256: str
    mechanism_evidence: Mapping[str, Any]
    mechanism_summary: str
    causal_claim: str
    falsifiable_prediction: str
    nearest_reference_ids: tuple[str, ...]
    packet_version: str = PACKET_SCHEMA_VERSION
    schema_name: str = PACKET_SCHEMA_NAME

    def __post_init__(self) -> None:
        require_identifier(self.packet_id, "packet_id")
        require_identifier(self.blinded_candidate_id, "blinded_candidate_id")
        require_sha256(self.corpus_sha256, "corpus_sha256")
        object.__setattr__(self, "mechanism_evidence", _deep_freeze(self.mechanism_evidence))
        object.__setattr__(self, "nearest_reference_ids", tuple(self.nearest_reference_ids))
        payload = self.to_dict(validate=False)
        _assert_blind_payload(payload)

    def to_dict(self, *, validate: bool = True) -> dict[str, Any]:
        payload = {
            "schema_name": self.schema_name,
            "packet_version": self.packet_version,
            "packet_id": self.packet_id,
            "blinded_candidate_id": self.blinded_candidate_id,
            "corpus_sha256": self.corpus_sha256,
            "mechanism_evidence": _deep_thaw(self.mechanism_evidence),
            "mechanism_summary": self.mechanism_summary,
            "causal_claim": self.causal_claim,
            "falsifiable_prediction": self.falsifiable_prediction,
            "nearest_reference_ids": list(self.nearest_reference_ids),
        }
        if validate:
            _assert_blind_payload(payload)
        return payload


@dataclass(frozen=True)
class BlindingIndexEntry:
    packet_id: str
    blinded_candidate_id: str
    candidate_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "packet_id": self.packet_id,
            "blinded_candidate_id": self.blinded_candidate_id,
            "candidate_id": self.candidate_id,
        }


@dataclass(frozen=True)
class BlindingIndex:
    corpus_sha256: str
    created_at_utc: str
    entries: tuple[BlindingIndexEntry, ...]
    schema_name: str = "SealedNoveltyBlindingIndex"
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "corpus_sha256": self.corpus_sha256,
            "created_at_utc": self.created_at_utc,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _secret_bytes(secret: str | bytes) -> bytes:
    value = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
    if len(value) < 16:
        raise ValueError("blinding secret must contain at least 16 bytes")
    return value


def _opaque(secret: bytes, context: str, value: str, length: int = 24) -> str:
    digest = hmac.new(secret, f"{context}:{value}".encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:length]


def generate_blinded_packets(
    materials: Iterable[ReviewMaterial],
    *,
    corpus_sha256: str,
    blinding_secret: str | bytes,
) -> tuple[tuple[BlindedReviewPacket, ...], BlindingIndex]:
    require_sha256(corpus_sha256, "corpus_sha256")
    secret = _secret_bytes(blinding_secret)
    material_items = tuple(materials)
    candidate_ids = [material.candidate_id for material in material_items]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique when packets are generated")
    packets: list[BlindedReviewPacket] = []
    index_entries: list[BlindingIndexEntry] = []
    for material in material_items:
        scope = f"{corpus_sha256}:{material.candidate_id}"
        blinded_candidate_id = f"blind-{_opaque(secret, 'candidate', scope)}"
        packet_id = f"packet-{_opaque(secret, 'packet', scope)}"
        packet = BlindedReviewPacket(
            packet_id=packet_id,
            blinded_candidate_id=blinded_candidate_id,
            corpus_sha256=corpus_sha256,
            mechanism_evidence=material.signature.review_payload(),
            mechanism_summary=material.mechanism_summary,
            causal_claim=material.causal_claim,
            falsifiable_prediction=material.falsifiable_prediction,
            nearest_reference_ids=material.nearest_reference_ids,
        )
        packets.append(packet)
        index_entries.append(
            BlindingIndexEntry(
                packet_id=packet_id,
                blinded_candidate_id=blinded_candidate_id,
                candidate_id=material.candidate_id,
            )
        )
    packets.sort(key=lambda item: item.packet_id)
    index_entries.sort(key=lambda item: item.packet_id)
    if len({packet.packet_id for packet in packets}) != len(packets):
        raise RuntimeError("blinding-token collision; use a different blinding secret")
    return (
        tuple(packets),
        BlindingIndex(
            corpus_sha256=corpus_sha256,
            created_at_utc=utc_now(),
            entries=tuple(index_entries),
        ),
    )
