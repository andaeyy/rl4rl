"""Frozen reference-corpus records and tamper-evident freeze verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from novelty.serialization import (
    atomic_write_json,
    content_sha256,
    require_bool,
    require_identifier,
    require_int,
    require_sha256,
    require_str,
    utc_now,
)
from novelty.signatures import MechanismSignature


CORPUS_POPULATION_REQUIRED = "CORPUS_POPULATION_REQUIRED"
CORPUS_SCHEMA_NAME = "FrozenReferenceCorpus"
CORPUS_SCHEMA_VERSION = "1.0"
SEAL_SCHEMA_NAME = "ReferenceCorpusFreezeSeal"


class CorpusPopulationRequired(RuntimeError):
    """Raised when a corpus is structurally valid but not scientifically ready."""

    code = CORPUS_POPULATION_REQUIRED

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__(f"{self.code}: " + "; ".join(reasons))


def _date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO 8601 calendar date") from error


def _utc_timestamp(value: str, field_name: str) -> None:
    if not value.endswith("Z"):
        raise ValueError(f"{field_name} must be an explicit UTC timestamp")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field_name} is not a valid timestamp") from error


@dataclass(frozen=True)
class ReferenceMechanism:
    reference_id: str
    source_id: str
    source_locator: str
    publication_date: str
    source_sha256: str
    mechanism_name: str
    mechanism_summary: str
    signature: MechanismSignature
    independently_reviewed: bool
    duplicate_of: str | None = None
    reviewer_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_bool(self.independently_reviewed, "independently_reviewed")
        require_identifier(self.reference_id, "reference_id")
        require_identifier(self.source_id, "source_id")
        _date(self.publication_date, "publication_date")
        require_sha256(self.source_sha256, "source_sha256")
        if not self.source_locator.strip():
            raise ValueError("source_locator cannot be empty")
        if not self.mechanism_name.strip() or not self.mechanism_summary.strip():
            raise ValueError("mechanism name and summary cannot be empty")
        if self.duplicate_of is not None:
            require_identifier(self.duplicate_of, "duplicate_of")
            if self.duplicate_of == self.reference_id:
                raise ValueError("a corpus entry cannot be its own duplicate")
        object.__setattr__(self, "reviewer_notes", tuple(self.reviewer_notes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "source_id": self.source_id,
            "source_locator": self.source_locator,
            "publication_date": self.publication_date,
            "source_sha256": self.source_sha256,
            "mechanism_name": self.mechanism_name,
            "mechanism_summary": self.mechanism_summary,
            "signature": self.signature.to_dict(),
            "independently_reviewed": self.independently_reviewed,
            "duplicate_of": self.duplicate_of,
            "reviewer_notes": list(self.reviewer_notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReferenceMechanism":
        return cls(
            reference_id=require_str(payload["reference_id"], "reference_id"),
            source_id=require_str(payload["source_id"], "source_id"),
            source_locator=require_str(payload["source_locator"], "source_locator"),
            publication_date=require_str(
                payload["publication_date"], "publication_date"
            ),
            source_sha256=require_str(payload["source_sha256"], "source_sha256"),
            mechanism_name=require_str(
                payload["mechanism_name"], "mechanism_name"
            ),
            mechanism_summary=require_str(
                payload["mechanism_summary"], "mechanism_summary"
            ),
            signature=MechanismSignature.from_dict(payload["signature"]),
            independently_reviewed=require_bool(
                payload["independently_reviewed"], "independently_reviewed"
            ),
            duplicate_of=(
                None
                if payload.get("duplicate_of") is None
                else require_str(payload["duplicate_of"], "duplicate_of")
            ),
            reviewer_notes=tuple(str(value) for value in payload.get("reviewer_notes", ())),
        )


@dataclass(frozen=True)
class ReferenceCorpusManifest:
    corpus_id: str
    cutoff_date: str
    retrieval_date: str
    inclusion_policy: str
    duplicate_policy: str
    population_complete: bool
    population_attested_by: str | None
    population_attested_at_utc: str | None
    synthetic_fixture: bool
    entries: tuple[ReferenceMechanism, ...]
    schema_name: str = CORPUS_SCHEMA_NAME
    schema_version: str = CORPUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_bool(self.population_complete, "population_complete")
        require_bool(self.synthetic_fixture, "synthetic_fixture")
        if self.schema_name != CORPUS_SCHEMA_NAME or self.schema_version != CORPUS_SCHEMA_VERSION:
            raise ValueError("unsupported reference-corpus schema")
        require_identifier(self.corpus_id, "corpus_id")
        cutoff = _date(self.cutoff_date, "cutoff_date")
        retrieval = _date(self.retrieval_date, "retrieval_date")
        if retrieval < cutoff:
            raise ValueError("retrieval_date cannot precede cutoff_date")
        if not self.inclusion_policy.strip() or not self.duplicate_policy.strip():
            raise ValueError("corpus inclusion and duplicate policies cannot be empty")
        object.__setattr__(self, "entries", tuple(self.entries))
        identifiers = [entry.reference_id for entry in self.entries]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("reference IDs must be unique")
        known = set(identifiers)
        for entry in self.entries:
            if _date(entry.publication_date, "publication_date") > cutoff:
                raise ValueError(
                    f"reference {entry.reference_id} was published after the corpus cutoff"
                )
            if entry.duplicate_of is not None and entry.duplicate_of not in known:
                raise ValueError(
                    f"reference {entry.reference_id} names an unknown duplicate target"
                )
        self._validate_duplicate_chains()
        if self.population_complete:
            if not self.population_attested_by or not self.population_attested_at_utc:
                raise ValueError("a complete corpus requires a named population attestation")
            require_identifier(self.population_attested_by, "population_attested_by")
            _utc_timestamp(self.population_attested_at_utc, "population_attested_at_utc")
        elif self.population_attested_by is not None or self.population_attested_at_utc is not None:
            raise ValueError("an incomplete corpus cannot carry a population attestation")

    def _validate_duplicate_chains(self) -> None:
        duplicate_of = {
            entry.reference_id: entry.duplicate_of
            for entry in self.entries
            if entry.duplicate_of is not None
        }
        for start in duplicate_of:
            seen: set[str] = set()
            current: str | None = start
            while current in duplicate_of:
                if current in seen:
                    raise ValueError("duplicate relationships contain a cycle")
                seen.add(current)
                current = duplicate_of[current]

    @property
    def corpus_sha256(self) -> str:
        return content_sha256(self.to_dict())

    @property
    def scientific_readiness_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.synthetic_fixture:
            issues.append("synthetic corpus fixtures cannot qualify a scientific run")
        if not self.population_complete:
            issues.append("corpus population has not been attested complete")
        if not self.entries:
            issues.append("corpus has no reference mechanisms")
        if any(not entry.independently_reviewed for entry in self.entries):
            issues.append("one or more corpus entries lack independent review")
        policy_text = f"{self.inclusion_policy} {self.duplicate_policy}".lower()
        if "decision_required" in policy_text or "pi_required" in policy_text:
            issues.append("corpus policies still contain unresolved PI placeholders")
        return tuple(issues)

    def assert_scientific_ready(self) -> None:
        issues = self.scientific_readiness_issues
        if issues:
            raise CorpusPopulationRequired(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "corpus_id": self.corpus_id,
            "cutoff_date": self.cutoff_date,
            "retrieval_date": self.retrieval_date,
            "inclusion_policy": self.inclusion_policy,
            "duplicate_policy": self.duplicate_policy,
            "population_complete": self.population_complete,
            "population_attested_by": self.population_attested_by,
            "population_attested_at_utc": self.population_attested_at_utc,
            "synthetic_fixture": self.synthetic_fixture,
            "entries": [entry.to_dict() for entry in sorted(self.entries, key=lambda item: item.reference_id)],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReferenceCorpusManifest":
        return cls(
            schema_name=require_str(payload.get("schema_name", ""), "schema_name"),
            schema_version=require_str(
                payload.get("schema_version", ""), "schema_version"
            ),
            corpus_id=require_str(payload["corpus_id"], "corpus_id"),
            cutoff_date=require_str(payload["cutoff_date"], "cutoff_date"),
            retrieval_date=require_str(payload["retrieval_date"], "retrieval_date"),
            inclusion_policy=require_str(
                payload["inclusion_policy"], "inclusion_policy"
            ),
            duplicate_policy=require_str(payload["duplicate_policy"], "duplicate_policy"),
            population_complete=require_bool(
                payload["population_complete"], "population_complete"
            ),
            population_attested_by=(
                None
                if payload.get("population_attested_by") is None
                else require_str(
                    payload["population_attested_by"], "population_attested_by"
                )
            ),
            population_attested_at_utc=(
                None
                if payload.get("population_attested_at_utc") is None
                else require_str(
                    payload["population_attested_at_utc"],
                    "population_attested_at_utc",
                )
            ),
            synthetic_fixture=require_bool(
                payload["synthetic_fixture"], "synthetic_fixture"
            ),
            entries=tuple(ReferenceMechanism.from_dict(item) for item in payload["entries"]),
        )


@dataclass(frozen=True)
class CorpusFreezeSeal:
    corpus_id: str
    corpus_schema_version: str
    corpus_sha256: str
    entry_count: int
    population_complete: bool
    synthetic_fixture: bool
    frozen_at_utc: str
    schema_name: str = SEAL_SCHEMA_NAME
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        require_bool(self.population_complete, "population_complete")
        require_bool(self.synthetic_fixture, "synthetic_fixture")
        if self.schema_name != SEAL_SCHEMA_NAME or self.schema_version != "1.0":
            raise ValueError("unsupported corpus-freeze seal schema")
        require_identifier(self.corpus_id, "corpus_id")
        require_sha256(self.corpus_sha256, "corpus_sha256")
        if self.entry_count < 0:
            raise ValueError("entry_count cannot be negative")
        _utc_timestamp(self.frozen_at_utc, "frozen_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "corpus_id": self.corpus_id,
            "corpus_schema_version": self.corpus_schema_version,
            "corpus_sha256": self.corpus_sha256,
            "entry_count": self.entry_count,
            "population_complete": self.population_complete,
            "synthetic_fixture": self.synthetic_fixture,
            "frozen_at_utc": self.frozen_at_utc,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CorpusFreezeSeal":
        return cls(
            schema_name=require_str(payload.get("schema_name", ""), "schema_name"),
            schema_version=require_str(
                payload.get("schema_version", ""), "schema_version"
            ),
            corpus_id=require_str(payload["corpus_id"], "corpus_id"),
            corpus_schema_version=require_str(
                payload["corpus_schema_version"], "corpus_schema_version"
            ),
            corpus_sha256=require_str(payload["corpus_sha256"], "corpus_sha256"),
            entry_count=require_int(payload["entry_count"], "entry_count"),
            population_complete=require_bool(
                payload["population_complete"], "population_complete"
            ),
            synthetic_fixture=require_bool(
                payload["synthetic_fixture"], "synthetic_fixture"
            ),
            frozen_at_utc=require_str(payload["frozen_at_utc"], "frozen_at_utc"),
        )


@dataclass(frozen=True)
class CorpusVerification:
    valid: bool
    corpus_sha256: str | None
    issues: tuple[str, ...]
    scientific_ready: bool


@dataclass(frozen=True)
class CorpusSignatureMatch:
    """Auditable exact-signature evidence, never an automatic novelty label."""

    reference_id: str
    mechanism_match: bool
    parameterization_match: bool
    behavior_match: bool
    intervention_match: bool


def query_corpus(
    manifest: ReferenceCorpusManifest,
    signature: MechanismSignature,
) -> tuple[CorpusSignatureMatch, ...]:
    """Return references sharing graph or probe evidence with ``signature``.

    A match narrows human review but cannot assign N0-N4. In particular, an
    unmatched hash is not evidence for N4 because the corpus can be incomplete
    and canonicalization can miss a semantic equivalence.
    """

    matches: list[CorpusSignatureMatch] = []
    for entry in manifest.entries:
        reference = entry.signature
        mechanism_match = reference.graph.mechanism_hash == signature.graph.mechanism_hash
        behavior_match = reference.behavior.signature_hash == signature.behavior.signature_hash
        intervention_match = (
            reference.intervention.signature_hash == signature.intervention.signature_hash
        )
        if mechanism_match or behavior_match or intervention_match:
            matches.append(
                CorpusSignatureMatch(
                    reference_id=entry.reference_id,
                    mechanism_match=mechanism_match,
                    parameterization_match=(
                        reference.graph.parameterization_hash
                        == signature.graph.parameterization_hash
                    ),
                    behavior_match=behavior_match,
                    intervention_match=intervention_match,
                )
            )
    return tuple(sorted(matches, key=lambda item: item.reference_id))


def freeze_corpus(
    manifest: ReferenceCorpusManifest,
    *,
    manifest_path: str | Path,
    seal_path: str | Path,
    require_scientific_ready: bool = False,
) -> CorpusFreezeSeal:
    """Write a manifest and seal once; never overwrite an existing freeze."""

    if require_scientific_ready:
        manifest.assert_scientific_ready()
    manifest_file = Path(manifest_path)
    seal_file = Path(seal_path)
    if manifest_file.exists() or seal_file.exists():
        raise FileExistsError("a corpus freeze already exists; create a new versioned corpus")
    seal = CorpusFreezeSeal(
        corpus_id=manifest.corpus_id,
        corpus_schema_version=manifest.schema_version,
        corpus_sha256=manifest.corpus_sha256,
        entry_count=len(manifest.entries),
        population_complete=manifest.population_complete,
        synthetic_fixture=manifest.synthetic_fixture,
        frozen_at_utc=utc_now(),
    )
    atomic_write_json(manifest_file, manifest.to_dict())
    try:
        atomic_write_json(seal_file, seal.to_dict())
    except Exception:
        manifest_file.unlink(missing_ok=True)
        raise
    return seal


def verify_frozen_corpus(
    *,
    manifest_path: str | Path,
    seal_path: str | Path,
    require_scientific_ready: bool = False,
) -> CorpusVerification:
    issues: list[str] = []
    manifest: ReferenceCorpusManifest | None = None
    seal: CorpusFreezeSeal | None = None
    try:
        with Path(manifest_path).open(encoding="utf-8") as handle:
            manifest = ReferenceCorpusManifest.from_dict(json.load(handle))
    except Exception as error:
        issues.append(f"manifest invalid: {type(error).__name__}: {error}")
    try:
        with Path(seal_path).open(encoding="utf-8") as handle:
            seal = CorpusFreezeSeal.from_dict(json.load(handle))
    except Exception as error:
        issues.append(f"freeze seal invalid: {type(error).__name__}: {error}")
    if manifest is not None and seal is not None:
        if manifest.corpus_id != seal.corpus_id:
            issues.append("manifest and seal corpus IDs differ")
        if manifest.schema_version != seal.corpus_schema_version:
            issues.append("manifest and seal schema versions differ")
        if manifest.corpus_sha256 != seal.corpus_sha256:
            issues.append("manifest content hash differs from its freeze seal")
        if len(manifest.entries) != seal.entry_count:
            issues.append("manifest entry count differs from its freeze seal")
        if manifest.population_complete != seal.population_complete:
            issues.append("population status differs from its freeze seal")
        if manifest.synthetic_fixture != seal.synthetic_fixture:
            issues.append("synthetic-fixture status differs from its freeze seal")
        if require_scientific_ready:
            issues.extend(manifest.scientific_readiness_issues)
    scientific_ready = bool(
        manifest is not None and not manifest.scientific_readiness_issues
    )
    return CorpusVerification(
        valid=not issues,
        corpus_sha256=None if manifest is None else manifest.corpus_sha256,
        issues=tuple(issues),
        scientific_ready=scientific_ready,
    )
