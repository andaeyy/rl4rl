from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import pytest

from analysis.outcomes import RunOutcome, RunOutcomeTable, RunTerminalStatus
from architecture_ir import (
    ArchitectureGraph,
    IREdge,
    IRNode,
    PrimitiveKind,
    TensorShape,
)
from novelty.clustering import (
    CandidateMechanism,
    MechanismClusterRecord,
    cluster_candidates,
    unique_cluster_counts_by_run,
)
from novelty.signatures import MechanismSignature, ProbeSignature
from study.budget import BudgetSpec
from study.contracts import BlockSpec, ConditionId, StudySpec
from study.engine import CommonStudyEngine
from study.fakes import DeterministicFakeEvaluator, DeterministicFakeGenerator
from study.randomization import (
    RandomizationPlan,
    generate_plan,
    load_or_create_plan,
)
from study.scheduling import MPSLease, MPSLeaseBusy
from study.serialization import atomic_write_json, content_hash


def _run(plan: RandomizationPlan, condition: ConditionId):
    return next(
        run for run in plan.runs if run.condition.condition_id is condition
    )


def _signature(prefix: str, *, width: int, class_name: str) -> MechanismSignature:
    token_shape = TensorShape(("B", "T"))
    hidden_shape = TensorShape(("B", "T", width))
    output_shape = TensorShape(("B", "T", 10))
    nodes = (
        IRNode(f"{prefix}-input", PrimitiveKind.INPUT, (), token_shape),
        IRNode(
            f"{prefix}-embed",
            PrimitiveKind.TOKEN_EMBEDDING,
            (token_shape,),
            hidden_shape,
            {"vocab_size": 20, "embedding_width": width},
        ),
        IRNode(
            f"{prefix}-attention",
            PrimitiveKind.ATTENTION,
            (hidden_shape,),
            hidden_shape,
            {
                "causal": True,
                "heads": 2,
                "projection": "dense",
                "class_name": class_name,
                "description": "claims a revolutionary shadow-proof mechanism",
            },
        ),
        IRNode(
            f"{prefix}-output",
            PrimitiveKind.READOUT,
            (hidden_shape,),
            output_shape,
            {"vocab_size": 10},
        ),
    )
    graph = ArchitectureGraph(
        graph_id=f"graph-{prefix}",
        input_node_id=f"{prefix}-input",
        output_node_id=f"{prefix}-output",
        nodes=nodes,
        edges=(
            IREdge(f"{prefix}-input", f"{prefix}-embed"),
            IREdge(f"{prefix}-embed", f"{prefix}-attention"),
            IREdge(f"{prefix}-attention", f"{prefix}-output"),
        ),
    )
    return MechanismSignature.create(
        graph,
        behavior=ProbeSignature.behavior(
            "behavior-v1", {"prefix_dependence": "present"}
        ),
        intervention=ProbeSignature.intervention(
            "intervention-v1", {"attention_zeroing": "large_effect"}
        ),
    )


def test_retries_and_seed_evaluation_do_not_inflate_condition_opportunity() -> None:
    spec = StudySpec.toy(
        study_id="adversarial-budget-symmetry",
        proposal_opportunities=3,
    )
    # Each condition gets the same deliberately hostile provider behavior.
    states = []
    with tempfile.TemporaryDirectory() as directory:
        plan = generate_plan(spec, Path(directory))
        for condition in ConditionId:
            state = CommonStudyEngine(
                study=spec,
                run=_run(plan, condition),
                generator=DeterministicFakeGenerator(
                    fail_first_attempts={1: 1},
                    parse_failures={2},
                ),
                evaluator=DeterministicFakeEvaluator(),
            ).execute()
            states.append(state)

    controlled_fields = (
        "seed_evaluations",
        "proposal_opportunities",
        "provider_attempts",
        "infrastructure_retries",
        "parse_failures",
        "repairs",
        "terminal_opportunities",
        "candidate_training_attempts",
    )
    signatures = {
        tuple(state.ledger[field] for field in controlled_fields) for state in states
    }
    assert len(signatures) == 1
    assert signatures.pop() == (1, 3, 5, 1, 2, 1, 3, 3)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_nonfinite_compute_budget_is_rejected(nonfinite: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        replace(BudgetSpec.toy(1), mps_seconds=nonfinite)


def test_rehashed_randomization_cannot_change_order_on_resume(tmp_path) -> None:
    spec = StudySpec.toy(
        study_id="adversarial-rehashed-order",
        study_seed=73,
        proposal_opportunities=2,
    )
    path = tmp_path / "randomization.json"
    original = load_or_create_plan(spec, output_root=tmp_path, plan_path=path)
    runs = list(original.blocks[0].runs)
    runs[0], runs[1] = runs[1], runs[0]
    rebuilt_runs = []
    for order_index, run in enumerate(runs):
        changed = replace(run, order_index=order_index, assignment_hash="pending")
        rebuilt_runs.append(
            replace(
                changed,
                assignment_hash=content_hash(changed.assignment_payload()),
            )
        )
    changed_block = replace(original.blocks[0], runs=tuple(rebuilt_runs))
    changed_blocks: tuple[BlockSpec, ...] = (
        changed_block,
        *original.blocks[1:],
    )
    changed_plan = replace(
        original,
        blocks=changed_blocks,
        assignment_hash="pending",
    )
    changed_plan = replace(
        changed_plan,
        assignment_hash=content_hash(changed_plan.assignment_payload()),
    )
    changed_plan.validate()
    atomic_write_json(path, changed_plan.to_dict())

    with pytest.raises(ValueError, match="randomization|assignment|frozen"):
        load_or_create_plan(spec, output_root=tmp_path, plan_path=path)


def test_parallel_mps_lease_attempt_fails_closed(tmp_path) -> None:
    path = tmp_path / "mps.lock"
    with MPSLease(path, run_id="run-one"):
        with pytest.raises(MPSLeaseBusy):
            MPSLease(path, run_id="run-two").acquire()
    assert not path.exists()


def test_class_names_descriptions_and_scale_cannot_spoof_a_new_mechanism() -> None:
    first = CandidateMechanism(
        study_id="study",
        candidate_id="candidate-a",
        run_id="run-one",
        snapshot_sha256="1" * 64,
        qualification_record_id="qualification-a",
        signature=_signature(
            "honest", width=16, class_name="OrdinaryAttention"
        ),
    )
    spoofed = CandidateMechanism(
        study_id="study",
        candidate_id="candidate-b",
        run_id="run-one",
        snapshot_sha256="2" * 64,
        qualification_record_id="qualification-b",
        signature=_signature(
            "spoofed", width=64, class_name="NeverBeforeSeenNovelAttention"
        ),
    )
    clusters = cluster_candidates((first, spoofed))
    assert len(clusters) == 1
    assert clusters[0].representative_by_run == (("run-one", "candidate-a"),)
    assert unique_cluster_counts_by_run(clusters) == {"run-one": 1}
    rendered = repr(first.signature.to_dict()).lower()
    assert "class_name" not in rendered
    assert "revolutionary" not in rendered


def test_relabelled_duplicate_mechanism_cannot_inflate_run_outcome() -> None:
    key = "a" * 64
    original = MechanismClusterRecord(
        study_id="study",
        cluster_id=f"mechanism-{key[:20]}",
        mechanism_cluster_key=key,
        candidate_ids=("candidate-one",),
        run_ids=("run-one",),
        representative_by_run=(("run-one", "candidate-one"),),
        member_signature_hashes=("b" * 64,),
        record_id="cluster-original",
    )
    relabelled = replace(
        original,
        cluster_id="mechanism-attacker-relabel",
        record_id="cluster-relabelled",
    )
    assert unique_cluster_counts_by_run((original, relabelled)) == {"run-one": 1}


def test_failed_assignment_cannot_disappear_and_candidate_rows_are_not_replicates() -> None:
    completed = RunOutcome(
        study_id="study",
        block_id="block",
        run_id="completed-run",
        condition_id="C0",
        run_seed=1,
        terminal_status=RunTerminalStatus.COMPLETED,
        qualifying_cluster_count=1,
        proposal_exposure=10,
        token_exposure=1_000,
    )
    with pytest.raises(ValueError, match="missing"):
        RunOutcomeTable(
            (completed,),
            ("completed-run", "failed-run"),
        )
    with pytest.raises(ValueError, match="candidate rows are not replicates"):
        RunOutcomeTable.from_records(
            (
                {
                    "schema_name": "CandidateRecord",
                    "candidate_id": "candidate",
                    "run_id": "completed-run",
                },
            ),
            assigned_run_ids=("completed-run",),
        )
