"""Deterministic tiny fakes for offline mechanism-infrastructure tests."""

from __future__ import annotations

from dataclasses import dataclass

from mechanism.claims import (
    DiscriminatingTest,
    EvidenceKind,
    EvidenceRequirement,
    MechanismClaim,
)
from mechanism.execution import FreshBuild, TrainingOutcome
from mechanism.plans import (
    ArmKind,
    ArmSpec,
    CounterfactualGrid,
    InterventionOperation,
    InterventionSpec,
    MechanismExperimentPlan,
    RescueSpec,
    ScalingManifest,
    SeedBundle,
    TrainingBudget,
)
from study.serialization import content_hash


@dataclass
class TinyFakeModel:
    token: str
    parameter_count: int
    trained: bool = False


class DeterministicTinyBuilder:
    def __init__(self, *, parameter_counts: dict[str, int] | None = None) -> None:
        self.parameter_counts = parameter_counts or {}
        self.calls: list[tuple[str, int]] = []

    def build_untrained(self, arm: ArmSpec, *, initialization_seed: int) -> FreshBuild:
        call_index = len(self.calls)
        self.calls.append((arm.arm_id, initialization_seed))
        token = f"{arm.arm_id}:{initialization_seed}:{call_index}"
        count = self.parameter_counts.get(arm.arm_id, 100 + call_index)
        return FreshBuild(
            model=TinyFakeModel(token=token, parameter_count=count),
            build_id=f"build:{arm.arm_id}:{initialization_seed}:{call_index}",
            arm_id=arm.arm_id,
            initialization_seed=initialization_seed,
            initial_state_sha256=content_hash(
                {
                    "token": token,
                    "seed": initialization_seed,
                    "variant": arm.builder_variant_id,
                }
            ),
            parameter_count_metadata=count,
        )


class DeterministicTinyTrainer:
    def __init__(self, *, fail: set[tuple[str, str]] | None = None) -> None:
        self.fail = fail or set()
        self.calls: list[tuple[str, str, int, int]] = []

    def train_from_scratch(
        self,
        build: FreshBuild,
        *,
        arm: ArmSpec,
        seeds: SeedBundle,
        budget: TrainingBudget,
    ) -> TrainingOutcome:
        model = build.model
        if not isinstance(model, TinyFakeModel):
            raise TypeError("tiny trainer only accepts TinyFakeModel")
        if model.trained:
            raise RuntimeError("tiny model instance was already trained")
        self.calls.append(
            (arm.arm_id, seeds.bundle_id, seeds.training_seed, seeds.data_order_seed)
        )
        if (arm.arm_id, seeds.bundle_id) in self.fail:
            raise RuntimeError("deterministic synthetic training failure")
        model.trained = True
        score_payload = {
            "arm": arm.arm_id,
            "training_seed": seeds.training_seed,
            "data_order_seed": seeds.data_order_seed,
        }
        score = int(content_hash(score_payload)[:8], 16) / 0xFFFFFFFF
        steps = min(2, budget.max_steps)
        examples = min(8, budget.max_examples)
        return TrainingOutcome(
            build_id=build.build_id,
            final_state_sha256=content_hash(
                {"initial": build.initial_state_sha256, "score": score}
            ),
            steps_completed=steps,
            examples_seen=examples,
            metrics=(("synthetic_accuracy", score),),
        )


def toy_mechanism_claim() -> MechanismClaim:
    return MechanismClaim(
        claim_id="claim:toy",
        candidate_snapshot_id="snapshot:toy",
        candidate_snapshot_sha256="a" * 64,
        proposed_mechanism="A routed attention state carries addition information.",
        causal_claim="The routed attention state causes improved carry generalization.",
        falsifiable_prediction="Removing the route erases the carry advantage.",
        nearest_alternative="The gain comes from additional generic capacity.",
        discriminating_tests=(
            DiscriminatingTest(
                test_id="test:ablation",
                description="Remove the proposed route and retrain from scratch.",
                prediction_if_claim_true="Carry generalization falls selectively.",
                prediction_if_alternative_true="Capacity matching restores the result.",
            ),
            DiscriminatingTest(
                test_id="test:rescue",
                description="Disable then restore the route at inference.",
                prediction_if_claim_true="Restoration rescues the output.",
                prediction_if_alternative_true="Restoration has no selective effect.",
            ),
        ),
        required_evidence=(
            EvidenceRequirement(
                requirement_id="evidence:ablation",
                kind=EvidenceKind.ABLATION,
                description="Paired from-scratch ablation evidence.",
            ),
            EvidenceRequirement(
                requirement_id="evidence:rescue",
                kind=EvidenceKind.RESCUE,
                description="Inference disable-and-rescue evidence.",
            ),
        ),
    )


def toy_mechanism_plan(*, seed_count: int = 2) -> MechanismExperimentPlan:
    claim = toy_mechanism_claim()
    arms = (
        ArmSpec(
            "arm:original",
            ArmKind.ORIGINAL,
            "builder:original",
            "none",
            "route",
        ),
        ArmSpec(
            "arm:removed",
            ArmKind.COMPONENT_REMOVED,
            "builder:removed",
            "remove route",
            "route",
        ),
        ArmSpec(
            "arm:replaced",
            ArmKind.COMPONENT_REPLACED,
            "builder:replaced",
            "replace route with identity",
            "route",
        ),
        ArmSpec(
            "arm:capacity",
            ArmKind.CAPACITY_MATCHED,
            "builder:capacity",
            "replace route with matched generic capacity",
            "route",
            target_parameter_count=1_000,
        ),
        ArmSpec(
            "arm:compute",
            ArmKind.COMPUTE_MATCHED,
            "builder:compute",
            "match measured training compute",
            "route",
        ),
    )
    seeds = tuple(
        SeedBundle(
            bundle_id=f"bundle:{index}",
            initialization_seed=100 + index,
            training_seed=200 + index,
            data_order_seed=300 + index,
        )
        for index in range(seed_count)
    )
    interventions = (
        InterventionSpec(
            intervention_id="intervention:disable",
            hook_id="hook:route",
            target_component="route",
            operation=InterventionOperation.ZERO,
            value=0.0,
            predicted_effect="Carry accuracy decreases.",
            falsification_condition="Carry accuracy remains unchanged.",
        ),
        InterventionSpec(
            intervention_id="intervention:restore",
            hook_id="hook:route",
            target_component="route",
            operation=InterventionOperation.RESTORE,
            value=1.0,
            predicted_effect="Carry accuracy recovers.",
            falsification_condition="Carry accuracy does not recover.",
        ),
    )
    return MechanismExperimentPlan(
        plan_id="mechanism-plan:toy",
        study_id="study:toy",
        run_id="run:toy",
        claim=claim,
        frozen_snapshot_id=claim.candidate_snapshot_id,
        frozen_snapshot_sha256=claim.candidate_snapshot_sha256,
        arms=arms,
        seed_bundles=seeds,
        training_budget=TrainingBudget(
            dataset_id="dataset:toy",
            optimizer_id="optimizer:toy",
            schedule_id="schedule:toy",
            max_steps=2,
            max_examples=8,
            wall_time_seconds=1.0,
        ),
        interventions=interventions,
        rescues=(
            RescueSpec(
                rescue_id="rescue:route",
                disabling_intervention_id="intervention:disable",
                restoring_intervention_id="intervention:restore",
                predicted_recovery="Restoration reverses the disabling effect.",
            ),
        ),
        counterfactual_grid=CounterfactualGrid(
            carry_depths=(1, 2),
            sequence_lengths=(8, 16),
            numeric_bases=(2, 10),
            symbol_mappings=("identity", "permuted"),
            representation_shifts=("reversed", "forward"),
        ),
        scaling_manifest=ScalingManifest(
            training_compute_steps=(100, 200),
            widths=(32, 64),
            depths=(2, 4),
            data_examples=(1_000, 2_000),
        ),
        scientific=False,
    )
