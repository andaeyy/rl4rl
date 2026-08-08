from __future__ import annotations

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
from architecture_ir.interpreter import validate_ir_candidate_path
from evaluation.dependency_audit import assert_controller_dependencies_clean
from evaluation.records import CONTROLLER_SEARCH_FIELDS
from scripts.audit_scientific_readiness import audit_readiness
from scripts.validate_engineering_canaries import validate_controller_surfaces
from study.contracts import ConditionSpec


def _config(name: str) -> dict:
    path = ROOT / "agents" / name / "config.yaml"
    return yaml.safe_load(path.read_text())


def _require(condition: bool, message: str) -> None:
    """Fail explicitly even when Python optimization disables assertions."""

    if not condition:
        raise RuntimeError(f"configuration invariant failed: {message}")


def _check_fitness_source() -> None:
    for function in (_quality, checkpoint_is_better):
        source = inspect.getsource(function)
        _require(
            "parameter_count_metadata" not in source,
            f"{function.__name__} uses parameter-count metadata",
        )
        _require(
            "parameter_count" not in source,
            f"{function.__name__} uses parameter count",
        )
    _require(
        "parameter_count_metadata" not in CONTROLLER_SEARCH_FIELDS,
        "controller view exposes parameter-count metadata",
    )
    _require(
        "shadow_accuracy" not in CONTROLLER_SEARCH_FIELDS,
        "controller view exposes shadow accuracy",
    )
    _require(
        "sealed_metrics" not in CONTROLLER_SEARCH_FIELDS,
        "controller view exposes sealed metrics",
    )


def main() -> None:
    greedy = _config("greedy_autoresearch")
    semantic_autoresearch = _config("semantic_autoresearch")
    generic = _config("openevolve_generic")
    semantic = _config("openevolve_semantic")
    _require(len(ConditionSpec.primary()) == 4, "primary condition count is not four")
    _require(
        set(EVALUATION_PROFILES)
        == {
            "unit_eval_v1",
            "smoke_eval_v1",
            "development_eval_v1",
            "scientific_layer_a_v1",
            "scientific_layer_b_v1",
            "scientific_layer_c_v1",
        },
        "evaluation profile roster changed",
    )
    _require(
        greedy["acceptance"]["use_parameter_count"] is False,
        "greedy acceptance uses parameter count",
    )
    _require(
        semantic_autoresearch["condition"] == "semantic_autoresearch",
        "semantic Autoresearch condition ID changed",
    )
    _require(
        semantic_autoresearch["acceptance"]["use_parameter_count"] is False,
        "semantic Autoresearch acceptance uses parameter count",
    )
    _require(
        all(
            name.startswith("semantic_")
            for name in semantic_autoresearch["archive"]["axes"]
        ),
        "semantic Autoresearch archive contains a non-semantic axis",
    )
    _require(generic["early_stopping_patience"] is None, "generic early stopping enabled")
    _require(semantic["early_stopping_patience"] is None, "semantic early stopping enabled")
    _require(
        generic["evaluator"]["parallel_evaluations"] == 1,
        "generic evaluator is not sequential",
    )
    _require(
        semantic["evaluator"]["parallel_evaluations"] == 1,
        "semantic evaluator is not sequential",
    )
    _require(generic["evaluator"]["timeout"] > 1800, "generic timeout is too short")
    _require(semantic["evaluator"]["timeout"] > 1800, "semantic timeout is too short")
    _require(generic["evaluator"]["max_retries"] == 0, "generic evaluator retries enabled")
    _require(semantic["evaluator"]["max_retries"] == 0, "semantic evaluator retries enabled")
    _require(
        generic["database"]["feature_dimensions"] == ["complexity", "diversity"],
        "generic archive dimensions changed",
    )
    _require(
        all(
            name.startswith("semantic_")
            for name in semantic["database"]["feature_dimensions"]
        ),
        "semantic OpenEvolve archive contains a non-semantic axis",
    )
    for config in (generic, semantic):
        trace = config["evolution_trace"]
        _require(trace["enabled"] is True, "evolution trace is disabled")
        _require(trace["include_code"] is True, "evolution trace omits code")
        _require(trace["include_prompts"] is True, "evolution trace omits prompts")

    generic_control = copy.deepcopy(generic)
    semantic_control = copy.deepcopy(semantic)
    generic_control["database"].pop("feature_dimensions")
    generic_control["database"].pop("feature_bins")
    semantic_control["database"].pop("feature_dimensions")
    semantic_control["database"].pop("feature_bins")
    _require(
        generic_control == semantic_control,
        "OpenEvolve conditions differ outside archive descriptors and prompts",
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
    _require(
        (
            semantic_autoresearch["iterations"],
            semantic_autoresearch["reasoning_effort"],
            semantic_autoresearch["temperature"],
            semantic_autoresearch["top_p"],
            semantic_autoresearch["max_tokens"],
            semantic_autoresearch["timeout_seconds"],
            semantic_autoresearch["retries"],
            semantic_autoresearch["retry_delay_seconds"],
        )
        == shared_generation,
        "semantic Autoresearch generation settings are not shared",
    )
    for config in (generic, semantic):
        _require(
            (
                config["max_iterations"],
                config["llm"]["reasoning_effort"],
                config["llm"]["temperature"],
                config["llm"]["top_p"],
                config["llm"]["max_tokens"],
                config["llm"]["timeout"],
                config["llm"]["retries"],
                config["llm"]["retry_delay"],
            )
            == shared_generation,
            "OpenEvolve generation settings are not shared",
        )
        _require(
            config["llm"]["models"]
            == [{"name": TARGET_MODEL, "weight": 1.0}],
            "OpenEvolve model roster changed",
        )
        _require(
            config["llm"]["api_base"] == "https://api.openai.com/v1",
            "OpenEvolve API base changed",
        )
        _require(
            config["diff_based_evolution"] is False,
            "OpenEvolve must request complete IR documents, not source diffs",
        )
        _require(config["language"] == "json", "OpenEvolve language is not JSON")
        _require(
            config["file_suffix"] == ".json",
            "OpenEvolve candidate suffix is not .json",
        )
        _require(
            config["evaluator"]["use_llm_feedback"] is False,
            "evaluator LLM feedback is enabled",
        )
    _require(greedy["temperature"] is None, "greedy temperature must be unset")
    _require(greedy["top_p"] is None, "greedy top_p must be unset")

    training_references = [
        greedy["training"],
        semantic_autoresearch["training"],
        generic["training"],
        semantic["training"],
    ]
    _require(
        all(
            reference == training_references[0]
            for reference in training_references[1:]
        ),
        "controller training references differ",
    )
    training_reference = training_references[0]
    profile = get_training_profile(training_reference["profile"])
    expected_profile_fields = {
        "version": str(training_reference["profile_version"]),
        "optimizer": "AdamW",
        "peak_learning_rate": 0.001,
        "adamw_betas": (0.9, 0.98),
        "weight_decay": 0.1,
        "scheduler": "cosine_decay_to_zero",
        "warmup_steps": 300,
        "global_batch_size": 512,
        "microbatch_size": None,
        "gradient_accumulation_steps": 1,
        "max_steps": 30_000,
        "validation_interval": 1_000,
        "validation_examples": 2_000,
        "checkpoint_interval": 1_000,
        "maximum_wall_seconds": 1_800,
        "device_requirement": "mps",
        "dtype": "float32",
        "deterministic_algorithms": True,
    }
    for field, expected in expected_profile_fields.items():
        _require(
            getattr(profile, field) == expected,
            f"training profile field {field} changed",
        )
    _require(
        profile.max_steps * profile.global_batch_size == 15_360_000,
        "full training example budget changed",
    )
    _require(
        training_reference["task_adapter"] == DEFAULT_TASK.version,
        "task adapter version changed",
    )
    _require(
        training_reference["seed_derivation"] == SEED_DERIVATION_METHOD,
        "seed derivation changed",
    )
    _require(
        profile.checkpoint_selection_rule.startswith(
            "higher_development_exact_match"
        ),
        "checkpoint selection rule changed",
    )

    manifest = yaml.safe_load((ROOT / "experiment_manifest.yaml").read_text())
    manifest_generation = manifest["shared_generation"]
    generation_expectations = {
        "target_model": TARGET_MODEL,
        "api_mode": API_MODE,
        "reasoning_effort": greedy["reasoning_effort"],
        "max_completion_tokens": greedy["max_tokens"],
        "request_timeout_seconds": greedy["timeout_seconds"],
        "retries": greedy["retries"],
        "retry_delay_seconds": greedy["retry_delay_seconds"],
    }
    for field, expected in generation_expectations.items():
        _require(
            manifest_generation.get(field) == expected,
            f"manifest generation field {field} changed",
        )
    manifest_training = manifest["training"]
    _require(manifest_training["profile"] == profile.name, "manifest profile changed")
    _require(
        str(manifest_training["profile_version"]) == profile.version,
        "manifest profile version changed",
    )
    _require(
        manifest_training["task_adapter"] == DEFAULT_TASK.version,
        "manifest task adapter changed",
    )
    _require(
        manifest_training["seed_derivation"] == SEED_DERIVATION_METHOD,
        "manifest seed derivation changed",
    )

    readiness = yaml.safe_load((ROOT / "readiness_evidence.yaml").read_text())
    levels = readiness["levels"]
    provenance = readiness.get("engineering_evidence_provenance")
    _require(
        isinstance(provenance, dict)
        and set(provenance)
        == {
            "authority",
            "source_revision_bound",
            "externally_attested",
            "scientific_launch_authority",
        },
        "engineering evidence provenance schema is incomplete",
    )
    _require(
        provenance["authority"] == "local_self_report",
        "engineering evidence authority is misstated",
    )
    for field in (
        "source_revision_bound",
        "externally_attested",
        "scientific_launch_authority",
    ):
        _require(
            type(provenance[field]) is bool,
            f"engineering evidence provenance field {field} is not boolean",
        )
    if readiness["status"] != "blocked":
        _require(
            provenance["source_revision_bound"]
            and provenance["externally_attested"]
            and provenance["scientific_launch_authority"],
            "an unblocked launch requires revision-bound external evidence",
        )
    _require(
        set(levels)
        == {
            "infrastructure_implemented",
            "unit_tested",
            "offline_smoke_tested",
            "mps_validated",
            "pilot_ready",
            "pilot_validated",
            "main_study_ready",
        },
        "readiness level roster changed",
    )
    _require(
        all(
            type(levels[name].get("passed")) is bool
            and isinstance(levels[name].get("evidence"), str)
            and levels[name]["evidence"].strip()
            for name in levels
        ),
        "readiness levels require exact booleans and nonempty evidence",
    )
    if levels["main_study_ready"]["passed"]:
        _require(
            all(levels[name]["passed"] for name in levels),
            "main study is marked ready while a lower level is false",
        )
    if levels["pilot_validated"]["passed"]:
        _require(
            levels["pilot_ready"]["passed"],
            "pilot is marked validated without pilot readiness",
        )
    _require(
        manifest["study"]["launch_status"] == readiness["status"],
        "manifest and readiness launch statuses differ",
    )

    decisions = yaml.safe_load((ROOT / "scientific_decisions.yaml").read_text())
    if decisions["status"] == "unresolved":
        _require(
            readiness["status"] == "blocked",
            "unresolved decisions do not block readiness",
        )
    readiness_report = audit_readiness()
    _require(readiness_report["provider_calls"] == 0, "readiness audit called provider")
    _require(readiness_report["training_runs"] == 0, "readiness audit started training")
    if readiness["status"] == "blocked":
        _require(
            not readiness_report["main_study_ready"],
            "blocked readiness reports main-study readiness",
        )

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
            _require(
                phrase not in text,
                f"{path} contains forbidden incentive {phrase}",
            )

    generic_prompt = (
        ROOT / "agents" / "openevolve_generic" / "system_prompt.md"
    ).read_text().lower()
    semantic_axis_terms = (
        "semantic_token_representation",
        "semantic_positional_integration",
        "semantic_attention_organization",
        "semantic_feedforward_mechanism",
        "semantic_normalization",
        "semantic_depth_topology",
        "semantic_output_readout",
        "semantic_tokenization",
        "token representation",
        "positional integration",
        "attention organization",
        "feedforward mechanism",
        "depth topology",
        "output readout",
        "tokenization",
    )
    _require(
        not any(term in generic_prompt for term in semantic_axis_terms),
        "generic prompt exposes semantic archive axes",
    )

    ir_seed = ROOT / "common" / "initial_candidate.ir.json"
    seed_validation = validate_ir_candidate_path(ir_seed)
    _require(
        seed_validation.valid,
        "initial architecture IR is invalid: "
        + "; ".join(issue.message for issue in seed_validation.issues),
    )
    _require(
        (ROOT / "architecture_ir" / "interpreter.py").is_file(),
        "trusted architecture-IR interpreter is missing",
    )

    controller_prompts = [
        ROOT / "agents" / "greedy_autoresearch" / "program.md",
        ROOT / "agents" / "semantic_autoresearch" / "program.md",
        ROOT / "agents" / "openevolve_generic" / "system_prompt.md",
        ROOT / "agents" / "openevolve_semantic" / "system_prompt.md",
    ]
    for path in controller_prompts:
        prompt = " ".join(path.read_text(encoding="utf-8").lower().split())
        _require(
            "complete replacement" in prompt and "json" in prompt,
            f"{path} does not require a complete replacement IR JSON document",
        )
        _require(
            "never return executable" in prompt or "do not return python" in prompt,
            f"{path} does not explicitly prohibit executable Python candidates",
        )

    runner_sources = [
        (ROOT / "agents" / "greedy_autoresearch" / "run.py").read_text(),
        (ROOT / "agents" / "semantic_autoresearch" / "run.py").read_text(),
        (ROOT / "common" / "openevolve_runner.py").read_text(),
    ]
    for source in runner_sources:
        _require(
            "initial_candidate.ir.json" in source,
            "controller runner does not reference the shared initial IR candidate",
        )
        _require(
            "common\" / \"evaluator.py" in source,
            "controller runner does not reference the shared evaluator",
        )
        _require(
            "--engineering-pilot" in source,
            "controller runner lacks the explicit engineering-pilot mode",
        )
    training_sources = [
        (ROOT / "common" / "trainer.py").read_text(),
        (ROOT / "common" / "training_data.py").read_text(),
    ]
    for source in training_sources:
        _require("private_eval" not in source, "training source imports private evaluation")
        _require(
            "DISCOVERY_SHADOW_SEED" not in source,
            "training source reads the shadow seed",
        )
        _require("2025" not in source, "training source embeds the legacy shadow seed")
    assert_controller_dependencies_clean(
        (
            ROOT / "agents" / "greedy_autoresearch" / "run.py",
            ROOT / "agents" / "semantic_autoresearch" / "run.py",
            ROOT / "common" / "openevolve_runner.py",
        ),
        project_root=ROOT,
    )
    _check_fitness_source()
    canary = validate_controller_surfaces(ROOT)
    _require(canary["passed"], f"static controller surfaces failed: {canary['errors']}")
    _require(canary["real_provider_calls"] == 0, "surface validator called provider")
    _require(canary["local_fixture_calls"] == 4, "surface fixture count is not four")
    _require(
        canary["entrypoint_execution_runs"] == 0,
        "surface validator executed an entrypoint",
    )
    _require(
        canary["candidate_execution_runs"] == 0,
        "surface validator executed a candidate",
    )
    _require(canary["training_runs"] == 0, "surface validator started training")
    print("configuration invariants: PASS")


if __name__ == "__main__":
    main()
