"""One adapter bridge shared by both OpenEvolve conditions."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult

from common.evaluator import (
    SearchEvaluationContext,
    evaluate_candidate,
    file_hash,
)


ROOT = Path(__file__).resolve().parents[1]


def evaluate_for_openevolve(program_path: str) -> EvaluationResult:
    profile = os.environ.get("DISCOVERY_TRAINING_PROFILE", "full_train_v1")
    run_seed = int(os.environ.get("DISCOVERY_TRAINING_SEED", "1"))
    device = os.environ.get("DISCOVERY_TRAIN_DEVICE", "mps")
    allow_cpu = os.environ.get("DISCOVERY_ALLOW_CPU_TRAINING", "0") == "1"
    training_root = Path(
        os.environ.get(
            "DISCOVERY_TRAINING_OUTPUT_ROOT",
            str(ROOT / "outputs" / "candidate_training"),
        )
    ).resolve()
    identifier = f"{file_hash(program_path)[:12]}_{uuid.uuid4().hex[:8]}"
    result = evaluate_candidate(
        program_path,
        training_profile=profile,
        training_seed=run_seed,
        training_output_dir=training_root / identifier,
        device=device,
        allow_cpu_for_tests=allow_cpu,
        evaluation_profile=os.environ.get("DISCOVERY_LAYER_A_PROFILE"),
        evaluation_case_count=(
            int(os.environ["DISCOVERY_LAYER_A_CASES"])
            if os.environ.get("DISCOVERY_LAYER_A_CASES")
            else None
        ),
        pi_decision_record_id=os.environ.get(
            "DISCOVERY_SCIENTIFIC_DECISION_RECORD"
        ),
        context=SearchEvaluationContext(
            study_id=os.environ.get("DISCOVERY_STUDY_ID", "native-replication"),
            block_id=os.environ.get("DISCOVERY_BLOCK_ID", "native-block"),
            run_id=os.environ.get("DISCOVERY_RUN_ID", "native-openevolve"),
            condition_id=os.environ.get(
                "DISCOVERY_CONDITION_ID", "native-openevolve"
            ),
        ),
    )
    view = result.controller_view()
    metrics = {
        "execution_ok": float(view.execution_ok),
        "transformer_valid": float(view.transformer_valid),
        "public_accuracy": view.public_accuracy,
        "search_score": view.search_score,
        "eligible_for_parent": float(view.eligible_for_parent),
        **dict(view.online_descriptor_codes),
    }
    artifacts = {
        "layer_a_record_id": view.record_id,
        "failure_stage": view.failure_stage,
        "infrastructure_failure": view.infrastructure_failure,
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
