"""Deterministic mechanism clustering with one contribution per run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from novelty.serialization import content_sha256, require_identifier, require_sha256
from novelty.signatures import MechanismSignature


@dataclass(frozen=True)
class CandidateMechanism:
    study_id: str
    candidate_id: str
    run_id: str
    snapshot_sha256: str
    qualification_record_id: str
    signature: MechanismSignature

    def __post_init__(self) -> None:
        require_identifier(self.study_id, "study_id")
        require_identifier(self.candidate_id, "candidate_id")
        require_identifier(self.run_id, "run_id")
        require_identifier(self.qualification_record_id, "qualification_record_id")
        require_sha256(self.snapshot_sha256, "snapshot_sha256")


@dataclass(frozen=True)
class MechanismClusterRecord:
    study_id: str
    cluster_id: str
    mechanism_cluster_key: str
    candidate_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    representative_by_run: tuple[tuple[str, str], ...]
    member_signature_hashes: tuple[str, ...]
    record_id: str
    schema_name: str = "MechanismClusterRecord"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        require_identifier(self.study_id, "study_id")
        require_identifier(self.cluster_id, "cluster_id")
        require_identifier(self.record_id, "record_id")
        require_sha256(self.mechanism_cluster_key, "mechanism_cluster_key")
        if tuple(sorted(set(self.candidate_ids))) != self.candidate_ids:
            raise ValueError("cluster candidate IDs must be sorted and unique")
        if tuple(sorted(set(self.run_ids))) != self.run_ids:
            raise ValueError("cluster run IDs must be sorted and unique")
        if tuple(sorted(self.representative_by_run)) != self.representative_by_run:
            raise ValueError("per-run representatives must use stable sorted order")
        if {run_id for run_id, _ in self.representative_by_run} != set(self.run_ids):
            raise ValueError("every contributing run must have exactly one representative")
        if tuple(sorted(set(self.member_signature_hashes))) != self.member_signature_hashes:
            raise ValueError("member signature hashes must be sorted and unique")
        for signature_hash in self.member_signature_hashes:
            require_sha256(signature_hash, "member_signature_hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "study_id": self.study_id,
            "cluster_id": self.cluster_id,
            "mechanism_cluster_key": self.mechanism_cluster_key,
            "candidate_ids": list(self.candidate_ids),
            "run_ids": list(self.run_ids),
            "representative_by_run": [
                {"run_id": run_id, "candidate_id": candidate_id}
                for run_id, candidate_id in self.representative_by_run
            ],
            "member_signature_hashes": list(self.member_signature_hashes),
        }


def cluster_candidates(
    candidates: Iterable[CandidateMechanism],
) -> tuple[MechanismClusterRecord, ...]:
    ordered = tuple(sorted(candidates, key=lambda item: (item.candidate_id, item.run_id)))
    candidate_ids = [item.candidate_id for item in ordered]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be globally unique before clustering")
    study_ids = {item.study_id for item in ordered}
    if len(study_ids) > 1:
        raise ValueError("one clustering call cannot mix candidates from different studies")
    grouped: dict[str, list[CandidateMechanism]] = {}
    for candidate in ordered:
        grouped.setdefault(candidate.signature.cluster_key, []).append(candidate)
    clusters: list[MechanismClusterRecord] = []
    for cluster_key, members in sorted(grouped.items()):
        per_run: dict[str, list[str]] = {}
        for member in members:
            per_run.setdefault(member.run_id, []).append(member.candidate_id)
        representatives = tuple(
            (run_id, min(candidate_ids_for_run))
            for run_id, candidate_ids_for_run in sorted(per_run.items())
        )
        cluster_id = f"mechanism-{cluster_key[:20]}"
        record_id = f"cluster-{content_sha256({'cluster_key': cluster_key})[:24]}"
        clusters.append(
            MechanismClusterRecord(
                study_id=members[0].study_id,
                cluster_id=cluster_id,
                mechanism_cluster_key=cluster_key,
                candidate_ids=tuple(sorted(member.candidate_id for member in members)),
                run_ids=tuple(sorted(per_run)),
                representative_by_run=representatives,
                member_signature_hashes=tuple(
                    sorted({member.signature.signature_hash for member in members})
                ),
                record_id=record_id,
            )
        )
    return tuple(clusters)


def unique_cluster_counts_by_run(
    clusters: Iterable[MechanismClusterRecord],
) -> Mapping[str, int]:
    contributions: dict[str, set[str]] = {}
    for cluster in clusters:
        for run_id in cluster.run_ids:
            contributions.setdefault(run_id, set()).add(
                cluster.mechanism_cluster_key
            )
    return {
        run_id: len(cluster_ids)
        for run_id, cluster_ids in sorted(contributions.items())
    }
