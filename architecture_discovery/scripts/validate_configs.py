from __future__ import annotations

import ast
import copy
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from common.evaluation_profiles import EVALUATION_PROFILES
from common.gpt56_sol import API_MODE, TARGET_MODEL
from common.openevolve_policy import _quality
from common.task_adapter import DEFAULT_TASK
from common.trainer import checkpoint_is_better
from common.training_config import SEED_DERIVATION_METHOD, get_training_profile
from evaluation.dependency_audit import assert_controller_dependencies_clean
from evaluation.records import CONTROLLER_SEARCH_FIELDS
from scripts.audit_scientific_readiness import audit_readiness
from study.contracts import ConditionSpec


def _config(name: str) -> dict:
    path = ROOT / "agents" / name / "config.yaml"
    return yaml.safe_load(path.read_text())


def _check_fitness_source() -> None:
    for function in (_quality, checkpoint_is_better):
        source = inspect.getsource(function)
        assert "parameter_count_metadata" not in source
        assert "parameter_count" not in source
    assert "parameter_count_metadata" not in CONTROLLER_SEARCH_FIELDS
    assert "shadow_accuracy" not in CONTROLLER_SEARCH_FIELDS
    assert "sealed_metrics" not in CONTROLLER_SEARCH_FIELDS


def main() -> None:
    greedy = _config("greedy_autoresearch")
    generic = _config("openevolve_generic")
    semantic = _config("openevolve_semantic")
    assert len(ConditionSpec.primary()) == 4
    assert set(EVALUATION_PROFILES) == {
        "unit_eval_v1",
        "smoke_eval_v1",
        "development_eval_v1",
        "scientific_layer_a_v1",
        "scientific_layer_b_v1",
        "scientific_layer_c_v1",
    }
    assert greedy["acceptance"]["use_parameter_count"] is False
    assert generic["early_stopping_patience"] is None
    assert semantic["early_stopping_patience"] is None
    assert generic["evaluator"]["parallel_evaluations"] == 1
    assert semantic["evaluator"]["parallel_evaluations"] == 1
    assert generic["evaluator"]["timeout"] > 1800
    assert semantic["evaluator"]["timeout"] > 1800
    assert generic["evaluator"]["max_retries"] == 0
    assert semantic["evaluator"]["max_retries"] == 0
    assert generic["database"]["feature_dimensions"] == ["complexity", "diversity"]
    assert all(
        name.startswith("semantic_")
        for name in semantic["database"]["feature_dimensions"]
    )
    for config in (generic, semantic):
        trace = config["evolution_trace"]
        assert trace["enabled"] is True
        assert trace["include_code"] is True
        assert trace["include_prompts"] is True

    generic_control = copy.deepcopy(generic)
    semantic_control = copy.deepcopy(semantic)
    generic_control["database"].pop("feature_dimensions")
    generic_control["database"].pop("feature_bins")
    semantic_control["database"].pop("feature_dimensions")
    semantic_control["database"].pop("feature_bins")
    assert generic_control == semantic_control, (
        "OpenEvolve conditions may differ only in archive descriptors and prompts"
    )

    shared_generation = (
        greedy["iterations"],
        greedy["reasoning_effort"],
        greedy["temperature"],
        greedy["top_p"],
        greedy["max_tokens"],
        greedy["timeout_seconds"],
        greedy["retries"],
        greedy["retry_delay_seconds"],
    )
    for config in (generic, semantic):
        assert (
            config["max_iterations"],
            config["llm"]["reasoning_effort"],
            config["llm"]["temperature"],
            config["llm"]["top_p"],
            config["llm"]["max_tokens"],
            config["llm"]["timeout"],
            config["llm"]["retries"],
            config["llm"]["retry_delay"],
        ) == shared_generation
        assert config["llm"]["models"] == [
            {"name": TARGET_MODEL, "weight": 1.0}
        ]
        assert config["llm"]["api_base"] == "https://api.openai.com/v1"
        assert config["diff_based_evolution"] is True
        assert config["evaluator"]["use_llm_feedback"] is False
    assert greedy["temperature"] is None
    assert greedy["top_p"] is None

    training_references = [
        greedy["training"],
        generic["training"],
        semantic["training"],
    ]
    assert training_references[0] == training_references[1]
    assert training_references[1] == training_references[2]
    training_reference = training_references[0]
    profile = get_training_profile(training_reference["profile"])
    assert profile.version == str(training_reference["profile_version"])
    assert training_reference["task_adapter"] == DEFAULT_TASK.version
    assert training_reference["seed_derivation"] == SEED_DERIVATION_METHOD
    assert profile.optimizer == "AdamW"
    assert profile.peak_learning_rate == 0.001
    assert profile.adamw_betas == (0.9, 0.98)
    assert profile.weight_decay == 0.1
    assert profile.scheduler == "cosine_decay_to_zero"
    assert profile.warmup_steps == 300
    assert profile.global_batch_size == 512
    assert profile.microbatch_size is None
    assert profile.gradient_accumulation_steps == 1
    assert profile.max_steps == 30_000
    assert profile.max_steps * profile.global_batch_size == 15_360_000
    assert profile.validation_interval == 1_000
    assert profile.validation_examples == 2_000
    assert profile.checkpoint_interval == 1_000
    assert profile.maximum_wall_seconds == 1_800
    assert profile.device_requirement == "mps"
    assert profile.dtype == "float32"
    assert profile.deterministic_algorithms is True
    assert profile.checkpoint_selection_rule.startswith(
        "higher_development_exact_match"
    )

    manifest = yaml.safe_load((ROOT / "experiment_manifest.yaml").read_text())
    manifest_generation = manifest["shared_generation"]
    assert manifest_generation["target_model"] == TARGET_MODEL
    assert manifest_generation["api_mode"] == API_MODE
    assert manifest_generation["reasoning_effort"] == greedy["reasoning_effort"]
    assert (
        manifest_generation["max_completion_tokens"] == greedy["max_tokens"]
    )
    assert (
        manifest_generation["request_timeout_seconds"]
        == greedy["timeout_seconds"]
    )
    assert manifest_generation["retries"] == greedy["retries"]
    assert (
        manifest_generation["retry_delay_seconds"]
        == greedy["retry_delay_seconds"]
    )
    manifest_training = manifest["training"]
    assert manifest_training["profile"] == profile.name
    assert str(manifest_training["profile_version"]) == profile.version
    assert manifest_training["task_adapter"] == DEFAULT_TASK.version
    assert manifest_training["seed_derivation"] == SEED_DERIVATION_METHOD

    readiness = yaml.safe_load((ROOT / "readiness_evidence.yaml").read_text())
    levels = readiness["levels"]
    assert set(levels) == {
        "infrastructure_implemented",
        "unit_tested",
        "offline_smoke_tested",
        "mps_validated",
        "pilot_ready",
        "pilot_validated",
        "main_study_ready",
    }
    assert all(
        isinstance(levels[name]["passed"], bool) and levels[name]["evidence"]
        for name in levels
    )
    if levels["main_study_ready"]["passed"]:
        assert all(levels[name]["passed"] for name in levels)
    if levels["pilot_validated"]["passed"]:
        assert levels["pilot_ready"]["passed"]
    assert manifest["study"]["launch_status"] == readiness["status"]

    decisions = yaml.safe_load((ROOT / "scientific_decisions.yaml").read_text())
    if decisions["status"] == "unresolved":
        assert readiness["status"] == "blocked"
    readiness_report = audit_readiness()
    assert readiness_report["provider_calls"] == 0
    assert readiness_report["training_runs"] == 0
    if readiness["status"] == "blocked":
        assert not readiness_report["main_study_ready"]

    forbidden_incentives = (
        "smallest model",
        "minimize parameter",
        "fewer parameter",
        "low-parameter",
        "compress the model",
    )
    prompt_paths = list((ROOT / "common" / "prompts").glob("*.md"))
    prompt_paths += list((ROOT / "agents").glob("**/*.md"))
    for path in prompt_paths:
        text = path.read_text().lower()
        for phrase in forbidden_incentives:
            assert phrase not in text, f"{path} contains forbidden incentive {phrase}"

    generic_prompt = (
        ROOT / "agents" / "openevolve_generic" / "system_prompt.md"
    ).read_text().lower()
    semantic_axis_terms = (
        "token representation",
        "positional integration",
        "attention organization",
        "feedforward",
        "normalization",
        "topology",
        "readout",
        "tokenization",
    )
    assert not any(term in generic_prompt for term in semantic_axis_terms)

    runner_sources = [
        (ROOT / "agents" / "greedy_autoresearch" / "run.py").read_text(),
        (ROOT / "common" / "openevolve_runner.py").read_text(),
    ]
    for source in runner_sources:
        assert "common\" / \"initial_candidate.py" in source
        assert "common\" / \"evaluator.py" in source
    training_sources = [
        (ROOT / "common" / "trainer.py").read_text(),
        (ROOT / "common" / "training_data.py").read_text(),
    ]
    for source in training_sources:
        assert "private_eval" not in source
        assert "DISCOVERY_SHADOW_SEED" not in source
        assert "2025" not in source
    assert_controller_dependencies_clean(
        (
            ROOT / "agents" / "greedy_autoresearch" / "run.py",
            ROOT / "common" / "openevolve_runner.py",
        ),
        project_root=ROOT,
    )
    _check_fitness_source()
    print("configuration invariants: PASS")


if __name__ == "__main__":
    main()
