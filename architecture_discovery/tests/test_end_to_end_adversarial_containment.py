from __future__ import annotations

from pathlib import Path

import pytest

from baselines.no_search import (
    DeterministicFakeBackend,
    NoSearchProposalGenerator,
    NoSearchSpec,
)
from common.candidate_contract import inspect_candidate_path
from common.device import DeviceUnavailableError, resolve_training_device
from common.task_adapter import DEFAULT_TASK
from common.trainer import ResumeMismatchError, _validate_resume
from common.training_client import WorkerError, build_worker_environment
from common.training_config import FULL_TRAIN_V1, SMOKE_TRAIN_V1, TrainingSeedBundle
from containment.audit import audit_runtime
from containment.policy import (
    CandidateFormat,
    ScientificExecutionRequest,
    assess_scientific_execution,
)
from containment.source_scan import RiskCategory, scan_python_path
from study.contracts import ConditionSpec
from study.interfaces import ProposalContext


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "e2e_adversarial"
    / "indirect_capability_escape.py"
)


def test_source_risk_scan_marks_obfuscated_capability_recovery_as_risky() -> None:
    report = scan_python_path(FIXTURE)
    assert report.parsed
    assert report.risky
    assert RiskCategory.DYNAMIC_BUILTINS in report.categories


def test_executed_python_candidate_contract_rejects_indirect_capability_escape() -> None:
    contract = inspect_candidate_path(FIXTURE)
    assert not contract.valid
    assert any("indirect" in reason or "builtin" in reason for reason in contract.reasons)


def test_scientific_arbitrary_python_remains_blocked_without_real_os_attestation() -> None:
    audit = audit_runtime(environment={"PYTORCH_ENABLE_MPS_FALLBACK": "0"})
    decision = assess_scientific_execution(
        audit,
        ScientificExecutionRequest(
            candidate_format=CandidateFormat.ARBITRARY_PYTHON,
            requested_device="mps",
            scientific=True,
            candidate_artifact_hash="a" * 64,
        ),
    )
    assert not decision.allowed
    assert any("containment" in blocker.lower() for blocker in decision.blockers)


def test_scientific_profile_rejects_cpu_and_mps_fallback(monkeypatch) -> None:
    with pytest.raises(DeviceUnavailableError, match="CPU is not permitted"):
        resolve_training_device(
            FULL_TRAIN_V1,
            "cpu",
            allow_cpu_for_tests=True,
        )
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(WorkerError, match="refusing to launch strict MPS"):
        build_worker_environment(
            requested_device="mps",
            allow_cpu_for_tests=False,
            model_seed=1,
        )


def test_worker_environment_exposes_no_parent_credentials() -> None:
    secrets = {
        "DISCOVERY_API_KEY": "discovery-value",
        "OPENAI_API_KEY": "openai-value",
        "AWS_SECRET_ACCESS_KEY": "aws-value",
        "GITHUB_TOKEN": "github-value",
        "DISCOVERY_SHADOW_SEED": "sealed-value",
        "LANG": "en_US.UTF-8",
    }
    environment = build_worker_environment(
        requested_device="cpu",
        allow_cpu_for_tests=True,
        model_seed=5,
        parent_environment=secrets,
    )
    assert environment["LANG"] == "en_US.UTF-8"
    assert not (set(secrets) - {"LANG"}).intersection(environment)
    assert not (set(secrets.values()) - {"en_US.UTF-8"}).intersection(
        environment.values()
    )


def test_checkpoint_from_a_different_candidate_is_rejected() -> None:
    seeds = TrainingSeedBundle.from_run_seed(3)
    checkpoint = {
        "candidate_source_hash": "old-candidate",
        "profile_hash": SMOKE_TRAIN_V1.profile_hash,
        "task_adapter_version": DEFAULT_TASK.version,
        "task_adapter_hash": DEFAULT_TASK.config_hash,
        "seed_bundle_hash": seeds.bundle_hash,
    }
    with pytest.raises(ResumeMismatchError, match="candidate"):
        _validate_resume(
            checkpoint,
            candidate_hash="new-architecture",
            profile=SMOKE_TRAIN_V1,
            task=DEFAULT_TASK,
            seeds=seeds,
        )


def test_no_search_model_input_is_unchanged_by_parent_or_search_history() -> None:
    backend = DeterministicFakeBackend()
    generator = NoSearchProposalGenerator(
        spec=NoSearchSpec(
            system_prompt="Propose one independent architecture.",
            task_prompt="Return one candidate without search feedback.",
            max_completion_tokens=128,
        ),
        backend=backend,
        scientific=False,
    )
    contexts = (
        ProposalContext(
            study_id="study",
            block_id="block",
            run_id="run",
            run_seed=1,
            condition=ConditionSpec.for_id("C0"),
            opportunity_index=1,
            provider_attempt=1,
            parent_ids=("parent-with-score-0.99",),
            transition_active=False,
        ),
        ProposalContext(
            study_id="study",
            block_id="block",
            run_id="run",
            run_seed=1,
            condition=ConditionSpec.for_id("C3"),
            opportunity_index=1,
            provider_attempt=1,
            parent_ids=("different-parent", "secret-history"),
            transition_active=True,
            repair=True,
        ),
    )
    for context in contexts:
        generator.generate(context)
    assert backend.requests[0].model_input == backend.requests[1].model_input
    rendered = repr(backend.requests[0].model_input)
    assert "0.99" not in rendered
    assert "secret-history" not in rendered
    assert "different-parent" not in rendered
