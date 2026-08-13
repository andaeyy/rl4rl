from dataclasses import replace

import pytest

from artifacts import (
    ArtifactContext,
    ArtifactEmittingStudyEngine,
    EventKind,
    ImmutableStudyEventSink,
    RunArtifactStore,
    StudyEventSinkError,
)
from artifacts.study_sink import ARCHITECTURE_IR_MEDIA_TYPE
from baselines.no_search import (
    DeterministicFakeBackend,
    NoSearchProposalGenerator,
    NoSearchSpec,
)
from study.contracts import ConditionId, StudySpec
from study.fakes import DeterministicFakeEvaluator
from study.randomization import generate_plan
from study.serialization import content_hash


def test_ir_seed_and_proposals_are_bound_as_ir_artifacts(tmp_path) -> None:
    backend = DeterministicFakeBackend()
    study = replace(
        StudySpec.toy(
            study_id="ir-artifact-sink",
            block_count=1,
            proposal_opportunities=1,
        ),
        initial_candidate_id=content_hash(backend.candidate_source),
    )
    plan = generate_plan(study, tmp_path)
    run = next(
        item
        for item in plan.runs
        if item.condition.condition_id is ConditionId.C0
    )
    store = RunArtifactStore(
        run.execution_directory / "artifacts",
        ArtifactContext(
            study_id=study.study_id,
            block_id=run.block_id,
            run_id=run.run_id,
            condition_id=run.condition.condition_id.value,
            writer_component="test-primary-ir-artifact-sink",
            code_sha256=study.code_hash,
            config_sha256=study.common_config_hash,
            environment_sha256=study.environment_hash,
            run_seed=run.run_seed,
            assignment_sha256=run.assignment_hash,
        ),
    )
    generator = NoSearchProposalGenerator(
        spec=NoSearchSpec(
            system_prompt="Return declarative architecture IR.",
            task_prompt="Return one complete independent IR document.",
            max_completion_tokens=128,
        ),
        backend=backend,
        scientific=False,
    )

    ArtifactEmittingStudyEngine(
        study=study,
        run=run,
        generator=generator,
        evaluator=DeterministicFakeEvaluator(),
        artifact_sink=ImmutableStudyEventSink(
            store,
            initial_candidate_source=backend.candidate_source,
        ),
    ).execute()

    candidates = [
        event
        for event in store.scan().events
        if event.event_kind is EventKind.CANDIDATE
    ]
    assert len(candidates) == 2
    assert all(
        event.payload["source_media_type"] == ARCHITECTURE_IR_MEDIA_TYPE
        for event in candidates
    )
    assert all("source_object_sha256" in event.payload for event in candidates)
    proposals = [
        event
        for event in store.scan().events
        if event.event_kind is EventKind.PROPOSAL
        and event.payload.get("phase") == "provider_response"
    ]
    assert proposals[0].payload["candidate_source_media_type"] == (
        ARCHITECTURE_IR_MEDIA_TYPE
    )


def test_seed_media_binding_rejects_partial_schema(tmp_path) -> None:
    study = StudySpec.toy(study_id="partial-ir-sink", proposal_opportunities=1)
    run = generate_plan(study, tmp_path).runs[0]
    store = RunArtifactStore(
        run.execution_directory / "artifacts",
        ArtifactContext(
            study_id=study.study_id,
            block_id=run.block_id,
            run_id=run.run_id,
            condition_id=run.condition.condition_id.value,
            writer_component="test-primary-ir-artifact-sink",
            code_sha256=study.code_hash,
            config_sha256=study.common_config_hash,
            environment_sha256=study.environment_hash,
            run_seed=run.run_seed,
            assignment_sha256=run.assignment_hash,
        ),
    )

    with pytest.raises(StudyEventSinkError, match="not Architecture IR"):
        ImmutableStudyEventSink(
            store,
            initial_candidate_source=(
                '{"schema_name":"architecture_tensor_graph",'
                '"schema_version":"1.0","nodes":[],"edges":[],"metadata":{}}'
            ),
        )
