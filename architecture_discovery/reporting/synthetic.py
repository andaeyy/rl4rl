"""Provider-free, one-command reconstruction of a complete toy study report.

This fixture exercises the real artifact ledger, rerun policy, reconstruction,
ITT table, research-ledger firewall, cards, report validation, and derived-file
provenance.  It performs no model-provider request and no candidate training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artifacts import (
    ArtifactContext,
    EventKind,
    FailureClass,
    FailureRecord,
    RerunPolicy,
    RunArtifactStore,
    authorize_rerun,
)
from novelty.taxonomy import NoveltyLabel
from reconstruction import build_analysis_table
from reporting.adapters import AdaptedRunReport, adapt_run_store
from reporting.cards import DataCard, ModelCard
from reporting.records import (
    SECTION_KINDS,
    ArithmeticClaim,
    ArithmeticClaimKind,
    DerivedArtifact,
    DerivedArtifactKind,
    ExternalValidityRecord,
    ExternalValidityStatus,
    MeasurementStatus,
    QuantityDisclosure,
    ReportArtifact,
    ReportArtifactKind,
    ReportSection,
    ResourceDisclosure,
    SectionName,
    SectionStatus,
    SourceArtifactReference,
    StudyProvenance,
)
from reporting.report import (
    ReproducibilityReport,
    build_reproducibility_report,
    write_report_exclusive,
)
from research_ledger import (
    ConfidenceLevel,
    DiscriminatingTestSpec,
    HypothesisSpec,
    HypothesisStatus,
    PredictionSpec,
    ResearchLedger,
    ResearchProtocol,
    freeze_protocol,
)
from review import AdjudicationStatus, NoveltyReviewRecord, ReviewConfidence
from review.records import ReviewerAssessment
from study.serialization import content_hash, create_json_exclusive


STUDY_ID = "synthetic-reconstruction-study"
BLOCK_ID = "synthetic-block-0"
GENERATED_AT_UTC = "2026-07-31T00:00:00Z"


@dataclass(frozen=True)
class SyntheticReconstructionResult:
    report: ReproducibilityReport
    report_path: Path
    assignment_path: Path
    protocol_path: Path


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _assignment(condition_id: str, order_index: int) -> dict[str, Any]:
    return {
        "schema_name": "SyntheticFrozenAssignment",
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "block_id": BLOCK_ID,
        "run_id": f"synthetic-run-{condition_id.lower()}",
        "condition_id": condition_id,
        "order_index": order_index,
        "run_seed": 810_021,
    }


def _store(
    root: Path,
    assignment: dict[str, Any],
    *,
    code_sha256: str,
    config_sha256: str,
    environment_sha256: str,
) -> RunArtifactStore:
    context = ArtifactContext(
        study_id=STUDY_ID,
        block_id=BLOCK_ID,
        run_id=str(assignment["run_id"]),
        condition_id=str(assignment["condition_id"]),
        writer_component="provider-free-synthetic-fixture",
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        environment_sha256=environment_sha256,
        run_seed=int(assignment["run_seed"]),
        assignment_sha256=content_hash(assignment),
    )
    return RunArtifactStore(root / "runs" / context.run_id, context)


def _append_completed_run(store: RunArtifactStore, *, qualifying: bool) -> None:
    suffix = store.context.condition_id.lower()
    proposal_id = f"proposal-{suffix}"
    candidate_id = f"candidate-{suffix}"
    store.append(EventKind.RUN_STATUS, {"status": "running"})
    store.append(
        EventKind.PROPOSAL,
        {"proposal_id": proposal_id, "opportunity_index": 1},
    )
    store.append(
        EventKind.PARENT_SELECTION,
        {"selected_candidate_ids": ["synthetic-seed-candidate"]},
    )
    store.append(
        EventKind.CANDIDATE,
        {
            "candidate_id": candidate_id,
            "proposal_id": proposal_id,
            "parent_candidate_ids": ["synthetic-seed-candidate"],
        },
    )
    store.append(
        EventKind.TRAINING,
        {"training_id": f"training-{suffix}", "candidate_id": candidate_id},
    )
    store.append(
        EventKind.SEARCH_EVALUATION,
        {
            "evaluation_id": f"search-evaluation-{suffix}",
            "candidate_id": candidate_id,
            "source_layer": "layer_a",
        },
    )
    store.append(
        EventKind.BUDGET,
        {
            "totals": {
                "proposal_opportunities": 1,
                "provider_attempts": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "training_steps": 0,
                "mps_seconds": 0,
            }
        },
    )
    if qualifying:
        store.append(
            EventKind.MECHANISM_CLUSTER,
            {
                "cluster_id": "synthetic-mechanism-cluster",
                "mechanism_cluster_key": content_hash(
                    {"mechanism": "synthetic-state-routing"}
                ),
                "run_ids": [store.context.run_id],
                "candidate_ids": [candidate_id],
                "qualifies_for_primary": True,
            },
        )
        store.append(
            EventKind.PROMOTION,
            {"promotion_id": "synthetic-promotion-c0", "candidate_id": candidate_id},
        )
    store.append(EventKind.RUN_STATUS, {"status": "completed"})


def _append_candidate_failure(store: RunArtifactStore) -> None:
    store.append(EventKind.RUN_STATUS, {"status": "running"})
    store.append(
        EventKind.PROPOSAL,
        {"proposal_id": "proposal-c2", "opportunity_index": 1},
    )
    store.append(
        EventKind.BUDGET,
        {
            "totals": {
                "proposal_opportunities": 1,
                "provider_attempts": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "training_steps": 0,
                "mps_seconds": 0,
            }
        },
    )
    failure = FailureRecord.create(
        attempt_id="synthetic-c2-attempt-0",
        failure_class=FailureClass.INVALID_TRANSFORMER,
        stage="runtime_validity",
    )
    store.append(EventKind.FAILURE, failure.to_event_payload())


def _append_infrastructure_rerun(store: RunArtifactStore) -> None:
    store.append(EventKind.RUN_STATUS, {"status": "running"})
    failure = FailureRecord.create(
        attempt_id="synthetic-c3-attempt-0",
        failure_class=FailureClass.WORKER_CRASH,
        stage="training_worker",
    )
    store.append(EventKind.FAILURE, failure.to_event_payload())
    authorization = authorize_rerun(
        assigned_run_id=store.context.run_id,
        previous_attempt_id=failure.attempt_id,
        attempt_number=1,
        failure=failure,
        policy=RerunPolicy(),
    )
    store.append(EventKind.RERUN_ATTEMPT, authorization.to_event_payload())
    store.append(EventKind.RUN_STATUS, {"status": "running"})
    store.append(
        EventKind.BUDGET,
        {
            "totals": {
                "proposal_opportunities": 0,
                "provider_attempts": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "training_steps": 0,
                "mps_seconds": 0,
                "infrastructure_retries": 1,
            }
        },
    )
    store.append(EventKind.RUN_STATUS, {"status": "completed"})


def _review_artifacts(corpus_sha256: str) -> tuple[ReportArtifact, ...]:
    raw_records = tuple(
        ReviewerAssessment(
            record_id=f"synthetic-raw-review-{index}",
            packet_id="synthetic-blinded-packet",
            reviewer_pseudonym=f"reviewer-{index}",
            corpus_sha256=corpus_sha256,
            label=NoveltyLabel.N3,
            confidence=ReviewConfidence.MEDIUM,
            rationale="The tested mechanism differs causally from the indexed alternatives.",
            nearest_reference_ids=("reference-1",),
            created_at_utc=GENERATED_AT_UTC,
        )
        for index in range(1, 4)
    )
    adjudication = NoveltyReviewRecord(
        record_id="synthetic-adjudicated-review",
        packet_id="synthetic-blinded-packet",
        corpus_sha256=corpus_sha256,
        final_label=NoveltyLabel.N3,
        status=AdjudicationStatus.RESOLVED,
        raw_review_record_ids=tuple(item.record_id for item in raw_records),
        raw_labels=tuple(
            (item.reviewer_pseudonym, item.label) for item in raw_records
        ),
        label_counts=((NoveltyLabel.N3, 3),),
        created_at_utc=GENERATED_AT_UTC,
    )
    return (
        *(
            ReportArtifact.create(
                artifact_id=record.record_id,
                kind=ReportArtifactKind.RAW_REVIEW,
                payload=record.to_dict(),
                record_schema_name=record.schema_name,
            )
            for record in raw_records
        ),
        ReportArtifact.create(
            artifact_id=adjudication.record_id,
            kind=ReportArtifactKind.ADJUDICATED_REVIEW,
            payload=adjudication.to_dict(),
            record_schema_name=adjudication.schema_name,
        ),
    )


def _research_ledger_artifact(
    output_root: Path,
    *,
    code_sha256: str,
    config_sha256: str,
    environment_sha256: str,
) -> tuple[ReportArtifact, str, Path]:
    hypothesis = HypothesisSpec(
        hypothesis_id="synthetic-routing-hypothesis",
        hypothesis="A state-routing change can alter arithmetic behavior under fixed compute.",
        causal_claim="The routing change, rather than parameter count, causes the observed behavior.",
        predictions=(
            PredictionSpec(
                prediction_id="synthetic-prediction-1",
                statement="The routing intervention changes the preregistered arithmetic outcome.",
                falsification_condition="The paired intervention leaves the outcome unchanged.",
            ),
        ),
        nearest_alternative="The observation is explained by stochastic training variation.",
        discriminating_tests=(
            DiscriminatingTestSpec(
                test_id="synthetic-discriminating-test-1",
                description="Compare paired intervention and sham-intervention retrains.",
                prediction_if_claim_true="The targeted intervention changes the outcome reproducibly.",
                prediction_if_alternative_true="Targeted and sham interventions have similar effects.",
            ),
        ),
        initial_confidence=ConfidenceLevel.LOW,
        initial_status=HypothesisStatus.PLANNED,
    )
    protocol = ResearchProtocol(
        protocol_id="synthetic-research-protocol",
        study_id=STUDY_ID,
        research_scope="Provider-free reconstruction fixture; not a scientific result.",
        hypotheses=(hypothesis,),
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        environment_sha256=environment_sha256,
        pi_decision_sha256=content_hash({"fixture": "non-scientific"}),
        scientific=False,
    )
    protocol_path = output_root / "frozen_research_protocol.json"
    frozen = freeze_protocol(protocol, protocol_path)
    ledger = ResearchLedger(frozen)
    ledger.close_search(
        closure_id="synthetic-search-close",
        reason="The provider-free fixture reached its frozen synthetic roster.",
    )
    ledger.seal(
        seal_id="synthetic-ledger-seal",
        reason="No sealed scientific evidence is generated by this fixture.",
    )
    payload = ledger.to_dict()
    return (
        ReportArtifact.create(
            artifact_id="synthetic-research-ledger",
            kind=ReportArtifactKind.RESEARCH_LEDGER,
            payload=payload,
            record_schema_name="ResearchLedger",
        ),
        frozen.protocol_sha256,
        protocol_path,
    )


def _sections(artifacts: tuple[ReportArtifact, ...]) -> tuple[ReportSection, ...]:
    return tuple(
        ReportSection(
            name=name,
            status=(SectionStatus.COMPLETE if artifact_ids else SectionStatus.NOT_RUN),
            artifact_ids=artifact_ids,
            limitation=(
                None
                if artifact_ids
                else "The provider-free reconstruction fixture did not generate this record type."
            ),
        )
        for name, kinds in SECTION_KINDS.items()
        for artifact_ids in (
            tuple(
                artifact.artifact_id
                for artifact in artifacts
                if artifact.kind in kinds
            ),
        )
    )


def build_synthetic_reconstruction(
    output_root: str | Path,
) -> SyntheticReconstructionResult:
    """Create and independently reconstruct a complete four-condition toy report."""

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    code_sha256 = content_hash(
        {"module": "reporting.synthetic", "schema_version": "1.0"}
    )
    config_payload = {
        "fixture": "provider-free",
        "candidate_training": False,
        "conditions": ["C0", "C1", "C2", "C3"],
    }
    config_sha256 = content_hash(config_payload)
    environment_sha256 = content_hash(
        {"runtime": "python", "provider_requests": 0, "candidate_training": False}
    )
    assignments = tuple(_assignment(condition, index) for index, condition in enumerate(("C0", "C1", "C2", "C3")))
    assignment_manifest = {
        "schema_name": "SyntheticFrozenAssignmentRoster",
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "assignments": list(assignments),
    }
    assignment_path = root / "frozen_assignments.json"
    create_json_exclusive(assignment_path, assignment_manifest)

    stores = tuple(
        _store(
            root,
            assignment,
            code_sha256=code_sha256,
            config_sha256=config_sha256,
            environment_sha256=environment_sha256,
        )
        for assignment in assignments
    )
    _append_completed_run(stores[0], qualifying=True)
    _append_completed_run(stores[1], qualifying=False)
    _append_candidate_failure(stores[2])
    _append_infrastructure_rerun(stores[3])

    adapted: tuple[AdaptedRunReport, ...] = tuple(
        adapt_run_store(store, assignment_payload=assignment)
        for store, assignment in zip(stores, assignments, strict=True)
    )
    frozen_run_ids = tuple(str(item["run_id"]) for item in assignments)
    analysis_table = build_analysis_table(
        (item.reconstructed for item in adapted),
        assigned_run_ids=frozen_run_ids,
    )
    analysis_payload = analysis_table.to_dict()
    analysis_artifact = ReportArtifact.create(
        artifact_id="synthetic-itt-analysis",
        kind=ReportArtifactKind.ANALYSIS,
        payload=analysis_payload,
        record_schema_name=analysis_table.schema_name,
    )
    corpus_sha256 = content_hash(
        {"fixture_corpus": "synthetic-reference-only", "cutoff": "2026-07-01"}
    )
    review_artifacts = _review_artifacts(corpus_sha256)
    mechanism_dossier = ReportArtifact.create(
        artifact_id="synthetic-mechanism-dossier",
        kind=ReportArtifactKind.MECHANISM_DOSSIER,
        payload={
            "schema_name": "SyntheticMechanismDossier",
            "schema_version": "1.0",
            "cluster_id": "synthetic-mechanism-cluster",
            "hypothesis_id": "synthetic-routing-hypothesis",
            "causal_status": "not_tested_in_fixture",
            "limitations": [
                "This provider-free reconstruction contains no mechanism experiment."
            ],
        },
    )
    model_card = ModelCard(
        model_card_id="synthetic-model-card",
        candidate_id="candidate-c0",
        architecture_signature_sha256=content_hash(
            {"fixture_architecture": "synthetic-state-routing"}
        ),
        training_configuration_sha256=config_sha256,
        checkpoint_sha256=content_hash({"fixture_checkpoint": "not-materialized"}),
        parameter_count_metadata=0,
        intended_use="Exercise reproducibility plumbing without training a candidate.",
        evaluation_scope="Synthetic autoregressive integer-addition records only.",
        limitations=("No trained weights are produced by this fixture.",),
    )
    data_card = DataCard(
        data_card_id="synthetic-data-card",
        dataset_id="synthetic-arithmetic-fixture",
        generator_code_sha256=content_hash({"generator": "synthetic-records"}),
        split_policy_sha256=content_hash({"split": "no-examples-generated"}),
        seed_manifest_sha256=content_hash(
            {"run_seed": 810_021, "roster": list(frozen_run_ids)}
        ),
        disjointness_evidence_sha256=content_hash(
            {"status": "not_applicable_no_examples"}
        ),
        data_role="reconstruction_fixture",
        intended_use="Validate artifact and report reconstruction without model data.",
        limitations=("No train, search, qualification, or confirmation cases exist.",),
    )
    ledger_artifact, protocol_sha256, protocol_path = _research_ledger_artifact(
        root,
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        environment_sha256=environment_sha256,
    )
    artifacts = tuple(
        artifact
        for item in adapted
        for artifact in item.artifacts
    ) + (
        *review_artifacts,
        mechanism_dossier,
        analysis_artifact,
        ReportArtifact.create(
            artifact_id=model_card.model_card_id,
            kind=ReportArtifactKind.MODEL_CARD,
            payload=model_card.to_dict(),
            record_schema_name=model_card.schema_name,
        ),
        ReportArtifact.create(
            artifact_id=data_card.data_card_id,
            kind=ReportArtifactKind.DATA_CARD,
            payload=data_card.to_dict(),
            record_schema_name=data_card.schema_name,
        ),
        ledger_artifact,
    )

    derived_root = root / "derived"
    table_payload = analysis_payload
    table_path = derived_root / "run_outcomes.json"
    create_json_exclusive(table_path, table_payload)
    figure_payload = {
        "schema_name": "SyntheticOutcomeFigureSpecification",
        "schema_version": "1.0",
        "figure_type": "condition_outcome_bars",
        "x": [row["condition_id"] for row in analysis_payload["rows"]],
        "y": [
            0
            if row["terminal_status"] != "completed"
            else row["qualifying_cluster_count"]
            for row in analysis_payload["rows"]
        ],
        "independent_unit": "assigned_run",
    }
    figure_path = derived_root / "condition_outcome_figure.json"
    create_json_exclusive(figure_path, figure_payload)
    analysis_source = SourceArtifactReference(
        artifact_id=analysis_artifact.artifact_id,
        content_sha256=analysis_artifact.content_sha256,
    )
    derived_artifacts = (
        DerivedArtifact.create(
            artifact_id="synthetic-run-outcome-table",
            kind=DerivedArtifactKind.TABLE,
            title="One row per frozen assigned run",
            relative_path=str(table_path.relative_to(root)),
            sources=(analysis_source,),
            transformation_id="serialize-run-outcome-table-v1",
            code_sha256=code_sha256,
            config_sha256=config_sha256,
            content_sha256=content_hash(table_payload),
            generated_at_utc=GENERATED_AT_UTC,
        ),
        DerivedArtifact.create(
            artifact_id="synthetic-condition-outcome-figure",
            kind=DerivedArtifactKind.FIGURE,
            title="Synthetic qualifying clusters by assigned condition",
            relative_path=str(figure_path.relative_to(root)),
            sources=(analysis_source,),
            transformation_id="synthetic-condition-bars-v1",
            code_sha256=code_sha256,
            config_sha256=config_sha256,
            content_sha256=content_hash(figure_payload),
            generated_at_utc=GENERATED_AT_UTC,
        ),
    )
    resources = ResourceDisclosure(
        quantities=(
            QuantityDisclosure(
                "mps_compute", "mps_seconds", MeasurementStatus.MEASURED, 0.0,
                "The fixture does not launch candidate training.",
            ),
            QuantityDisclosure(
                "cpu_compute", "cpu_seconds", MeasurementStatus.UNKNOWN, None,
                "Wall-clock CPU instrumentation is outside this synthetic fixture.",
            ),
            QuantityDisclosure(
                "monetary_cost", "usd", MeasurementStatus.MEASURED, 0.0,
                "The fixture makes zero provider requests.",
            ),
            QuantityDisclosure(
                "energy", "kilowatt_hours", MeasurementStatus.UNKNOWN, None,
                "No calibrated energy meter is attached to this fixture.",
            ),
        ),
        prompt_tokens=0,
        completion_tokens=0,
        provider_usage_complete=True,
        notes=(
            "This is a provider-free reconstruction test, not a scientific experiment.",
            "Parameter count is retained only as descriptive metadata.",
        ),
    )
    provenance = StudyProvenance(
        study_id=STUDY_ID,
        study_spec_sha256=content_hash(
            {"study_id": STUDY_ID, "fixture": True, "scientific": False}
        ),
        config_sha256=config_sha256,
        code_sha256=code_sha256,
        environment_sha256=environment_sha256,
        randomization_sha256=content_hash(assignment_manifest),
        research_protocol_sha256=protocol_sha256,
        analysis_plan_sha256=content_hash(
            {"estimand": "one row per assigned run", "mode": "synthetic"}
        ),
        artifact_index_sha256=content_hash(
            sorted(item.artifact_index_sha256 for item in adapted)
        ),
        reference_corpus_sha256=corpus_sha256,
        generator_configuration_sha256=content_hash(
            {"generator": "provider-free-synthetic-event-fixture"}
        ),
    )
    report = build_reproducibility_report(
        report_id="synthetic-reproducibility-report",
        provenance=provenance,
        frozen_assignment_run_ids=frozen_run_ids,
        runs=(item.run_record for item in adapted),
        artifacts=artifacts,
        sections=_sections(artifacts),
        derived_artifacts=derived_artifacts,
        resources=resources,
        external_validity=ExternalValidityRecord(
            status=ExternalValidityStatus.ARITHMETIC_ONLY,
            primary_task_id="autoregressive-integer-addition",
            tested_task_ids=("autoregressive-integer-addition",),
            second_task_evidence_ids=(),
            scaling_evidence_ids=(),
            limitation=(
                "No language-domain, second-task, or scaling result exists; the fixture "
                "validates reconstruction only."
            ),
        ),
        claims=(
            ArithmeticClaim(
                claim_id="synthetic-reconstruction-claim",
                kind=ArithmeticClaimKind.SEARCH_YIELD,
                result_summary=(
                    "the frozen four-run fixture reconstructs one retained ITT row per assignment."
                ),
                evidence_artifact_ids=(analysis_artifact.artifact_id,),
                limitations=(
                    "The inputs are synthetic records and contain no trained-model evidence.",
                ),
            ),
        ),
        limitations=(
            "This provider-free fixture validates infrastructure, not architecture quality.",
            "It contains no candidate training, sealed evaluation, causal test, or replication.",
            "Its arithmetic-only claim cannot be generalized to language modeling.",
        ),
        generated_at_utc=_utc_now(),
    )
    report_path = root / "reproducibility_report.json"
    write_report_exclusive(report, report_path)
    return SyntheticReconstructionResult(
        report=report,
        report_path=report_path,
        assignment_path=assignment_path,
        protocol_path=protocol_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a provider-free synthetic reproducibility reconstruction."
    )
    parser.add_argument("--output", required=True, help="New or empty output directory")
    args = parser.parse_args(argv)
    result = build_synthetic_reconstruction(args.output)
    print(result.report_path)
    print(f"report_sha256={result.report.report_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
