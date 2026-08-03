"""Explicitly non-scientific regression for the vendor checkpoint.

This module is intentionally inside ``private_eval`` so the dependency audit
can prove that online controllers do not reach it. It tests decoder and archive
plumbing only. It is not a Layer A, B, or C study evaluation.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import torch

from common.candidate_loader import load_candidate
from common.descriptor_extractor import extract_descriptors
from common.evaluation_result import DiscoveryEvaluation
from common.task_adapter import DEFAULT_TASK
from private_eval.shadow_evaluator import (
    carry_cases,
    edge_cases,
    random_cases,
    shadow_seed,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "vendor" / "starting_model" / "checkpoints" / "best.pt"


def _regression_score(result: DiscoveryEvaluation) -> None:
    result.robustness_score = (
        result.shadow_accuracy + result.edge_accuracy + result.carry_accuracy
    ) / 3.0
    result.qualifies = (
        result.execution_ok
        and result.transformer_valid
        and result.official_accuracy >= 0.99
        and result.shadow_accuracy >= 0.99
    )
    if result.transformer_valid:
        result.combined_score = (
            0.7 * min(result.official_accuracy, result.shadow_accuracy)
            + 0.15 * result.edge_accuracy
            + 0.15 * result.carry_accuracy
        )


def evaluate_pretrained_baseline_regression(
    *,
    official_count: int = 64,
    shadow_count: int = 64,
    device: str = "cpu",
) -> DiscoveryEvaluation:
    """Run a known-checkpoint decoder regression outside online search."""

    result = DiscoveryEvaluation()
    started = time.perf_counter()
    model: torch.nn.Module | None = None
    try:
        module = load_candidate(ROOT / "common" / "initial_candidate.py")
        model, _ = module.build_untrained_model(0)
        checkpoint = torch.load(
            DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True
        )
        model.load_state_dict(checkpoint["model_state"])
        target = torch.device(device)
        model = model.to(target)
        model.eval()
        result.execution_ok = True
        result.transformer_valid = True
        result.parameter_count_metadata = sum(
            parameter.numel() for parameter in model.parameters()
        )
        result.official_accuracy, _ = DEFAULT_TASK.exact_match(
            model,
            random_cases(official_count, 2025),
            device=target,
            batch_size=min(512, official_count),
            failure_limit=0,
        )
        result.shadow_accuracy, _ = DEFAULT_TASK.exact_match(
            model,
            random_cases(shadow_count, shadow_seed()),
            device=target,
            batch_size=min(512, shadow_count),
            failure_limit=0,
        )
        result.edge_accuracy, _ = DEFAULT_TASK.exact_match(
            model,
            edge_cases(),
            device=target,
            batch_size=len(edge_cases()),
            failure_limit=0,
        )
        result.carry_accuracy, _ = DEFAULT_TASK.exact_match(
            model,
            carry_cases(),
            device=target,
            batch_size=len(carry_cases()),
            failure_limit=0,
        )
        descriptors = extract_descriptors(module, model)
        result.descriptor_vector = descriptors.categories
        result.descriptor_confidence = descriptors.confidence
        result.semantic_metrics = descriptors.codes
        _regression_score(result)
        if not result.qualifies:
            result.failure_stage = "pretrained_regression_accuracy"
    except Exception as error:
        result.failure_stage = "pretrained_regression"
        result.artifacts["error"] = f"{type(error).__name__}: {error}"
    finally:
        result.verify_seconds = time.perf_counter() - started
        if model is not None:
            del model
        gc.collect()
    return result

