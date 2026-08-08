"""Provider-free static validation for four controller surface contracts.

This command deliberately separates three claims that are easy to conflate:

* controller-surface validation statically checks entrypoints, configuration,
  prompts, and a complete declarative Architecture IR response fixture;
* optional MPS smoke-artifact validation checks internal consistency of an
  already completed trusted ten-step run without proving where it executed;
* scientific pilot readiness is never inferred by this command.

The deterministic fake response changes only non-executable graph metadata.  It
is size-bounded and checked through the same trusted IR validator used at the
candidate boundary.  No candidate model is constructed by the surface check,
no controller entrypoint is imported or executed, no provider SDK is
constructed, and no network request is made by this validator.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_ir import load_and_build_ir_candidate, validate_ir_candidate_json
from architecture_ir.codec import MAX_IR_JSON_BYTES
from common.training_config import SMOKE_TRAIN_V1


TRUSTED_CANDIDATE_RELATIVE_PATH = Path("common/initial_candidate.ir.json")
# Deliberately stricter than the interpreter's own hard ceiling.  A fake
# provider response is only a small canary fixture, not a large search result.
MAX_FAKE_RESPONSE_BYTES = min(128 * 1024, MAX_IR_JSON_BYTES)
REQUIRED_CANARY_CLI_FLAGS = (
    "--engineering-pilot",
    "--iterations",
    "--seed",
    "--output-dir",
    "--training-profile",
    "--evaluation-profile",
    "--evaluation-cases",
    "--device",
)


@dataclass(frozen=True)
class HarnessSpec:
    harness_id: str
    display_name: str
    agent_directory: str
    parent_policy: str
    proposal_policy: str
    prompt_candidates: tuple[str, ...]
    config_condition: str | None = None
    delegated_controller_kind: str | None = None


HARNESSES = (
    HarnessSpec(
        harness_id="normal_autoresearch",
        display_name="Normal Autoresearch",
        agent_directory="greedy_autoresearch",
        parent_policy="single",
        proposal_policy="ordinary",
        prompt_candidates=("program.md", "system_prompt.md"),
        config_condition="greedy_autoresearch",
    ),
    HarnessSpec(
        harness_id="semantic_autoresearch",
        display_name="Semantic Autoresearch",
        agent_directory="semantic_autoresearch",
        parent_policy="single",
        proposal_policy="semantic_transition",
        prompt_candidates=("program.md", "system_prompt.md"),
        config_condition="semantic_autoresearch",
    ),
    HarnessSpec(
        harness_id="openevolve",
        display_name="OpenEvolve",
        agent_directory="openevolve_generic",
        parent_policy="portfolio",
        proposal_policy="ordinary",
        prompt_candidates=("system_prompt.md", "program.md"),
        delegated_controller_kind="generic",
    ),
    HarnessSpec(
        harness_id="semantic_openevolve",
        display_name="Semantic OpenEvolve",
        agent_directory="openevolve_semantic",
        parent_policy="portfolio",
        proposal_policy="semantic_archive",
        prompt_candidates=("system_prompt.md", "program.md"),
        delegated_controller_kind="semantic",
    ),
)


class DeterministicFakeProvider:
    """A local response fixture; it has no client, endpoint, or network method."""

    def __init__(self, response: str) -> None:
        if not isinstance(response, str):
            raise TypeError("fake-provider response must be text")
        if len(response.encode("utf-8")) > MAX_FAKE_RESPONSE_BYTES:
            raise ValueError("fake-provider response exceeds the canary byte limit")
        validation = validate_ir_candidate_json(response)
        if not validation.valid:
            raise ValueError(
                "fake-provider response is not valid Architecture IR: "
                + "; ".join(issue.message for issue in validation.issues)
            )
        self._response = response
        self.calls = 0

    def complete(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("fake-provider prompt cannot be empty")
        self.calls += 1
        return self._response


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_ir_response(trusted_ir: str, spec: HarnessSpec) -> str:
    """Return one complete valid IR document with a metadata-only mutation."""

    validation = validate_ir_candidate_json(trusted_ir)
    if not validation.valid or validation.graph is None:
        raise ValueError(
            "trusted seed is not valid Architecture IR: "
            + "; ".join(issue.message for issue in validation.issues)
        )
    payload = validation.graph.to_dict()
    payload["metadata"]["engineering_canary_fixture"] = spec.harness_id
    response = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if len(response.encode("utf-8")) > MAX_FAKE_RESPONSE_BYTES:
        raise ValueError("fixed Architecture IR response exceeds the canary byte limit")
    child_validation = validate_ir_candidate_json(response)
    if not child_validation.valid:
        raise ValueError(
            "fixed Architecture IR response failed trusted validation: "
            + "; ".join(issue.message for issue in child_validation.issues)
        )
    return response


def _graph_structure(graph: Any) -> dict[str, Any]:
    """Return executable graph fields, excluding non-executable metadata."""

    payload = graph.to_dict()
    payload.pop("metadata", None)
    return payload


def _fake_prompt(spec: HarnessSpec, source_hash: str) -> str:
    return "\n".join(
        (
            "Engineering static-surface fixture. Do not execute candidates.",
            f"harness={spec.harness_id}",
            f"parent_policy={spec.parent_policy}",
            f"proposal_policy={spec.proposal_policy}",
            f"trusted_parent_sha256={source_hash}",
            "Return exactly one complete Architecture IR JSON document.",
        )
    )


def _declared_cli_flags(tree: ast.AST) -> set[str]:
    """Collect literal long options without importing or executing a module."""

    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "add_argument":
            continue
        for argument in node.args:
            if (
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.startswith("--")
            ):
                flags.add(argument.value)
    return flags


def _delegates_to_controller(tree: ast.AST, expected_kind: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Name) or function.id != "run_controller":
            continue
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == expected_kind
        ):
            return True
    return False


def _validate_harness(
    project_root: Path,
    spec: HarnessSpec,
    *,
    trusted_ir: str,
    trusted_artifact_hash: str,
) -> dict[str, Any]:
    agent_root = project_root / "agents" / spec.agent_directory
    entrypoint = agent_root / "run.py"
    config_path = agent_root / "config.yaml"
    errors: list[str] = []
    static_cli_contract_ok = False

    if not entrypoint.is_file():
        errors.append(f"missing entrypoint: {entrypoint.relative_to(project_root)}")
    else:
        try:
            entrypoint_tree = ast.parse(entrypoint.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as error:
            errors.append(f"entrypoint is not valid Python: {type(error).__name__}: {error}")
            entrypoint_tree = None
        if entrypoint_tree is not None:
            cli_flags = _declared_cli_flags(entrypoint_tree)
            delegation_ok = True
            if spec.delegated_controller_kind is not None:
                delegation_ok = _delegates_to_controller(
                    entrypoint_tree,
                    spec.delegated_controller_kind,
                )
                if not delegation_ok:
                    errors.append(
                        "entrypoint does not statically delegate to "
                        f"run_controller({spec.delegated_controller_kind!r})"
                    )
                shared_runner = project_root / "common" / "openevolve_runner.py"
                if not shared_runner.is_file():
                    errors.append("missing delegated common/openevolve_runner.py")
                else:
                    try:
                        shared_tree = ast.parse(
                            shared_runner.read_text(encoding="utf-8")
                        )
                    except (OSError, SyntaxError) as error:
                        errors.append(
                            "delegated runner is not valid Python: "
                            f"{type(error).__name__}: {error}"
                        )
                    else:
                        cli_flags.update(_declared_cli_flags(shared_tree))
            missing_flags = [
                flag for flag in REQUIRED_CANARY_CLI_FLAGS if flag not in cli_flags
            ]
            if missing_flags:
                errors.append(
                    "static CLI contract lacks required flags: "
                    + ", ".join(missing_flags)
                )
            static_cli_contract_ok = delegation_ok and not missing_flags

    config: dict[str, Any] | None = None
    if not config_path.is_file():
        errors.append(f"missing configuration: {config_path.relative_to(project_root)}")
    else:
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise TypeError("top-level YAML value must be a mapping")
            config = loaded
        except (OSError, TypeError, yaml.YAMLError) as error:
            errors.append(f"configuration is invalid: {type(error).__name__}: {error}")
    if config is not None:
        training = config.get("training")
        if not isinstance(training, dict):
            errors.append("configuration lacks a training mapping")
        else:
            if not isinstance(training.get("profile"), str):
                errors.append("configuration lacks a named training profile")
            if training.get("device") != "mps":
                errors.append("configured default training device is not strict MPS")
            if training.get("allow_cpu_for_tests") is not False:
                errors.append("configured controller does not disable CPU training")
        if (
            spec.config_condition is not None
            and config.get("condition") != spec.config_condition
        ):
            errors.append(
                "configured condition differs from the named harness: "
                f"{config.get('condition')!r} != {spec.config_condition!r}"
            )

    prompt_path = next(
        (
            agent_root / name
            for name in spec.prompt_candidates
            if (agent_root / name).is_file()
        ),
        None,
    )
    if prompt_path is None:
        choices = ", ".join(spec.prompt_candidates)
        errors.append(f"missing controller prompt (expected one of: {choices})")
    elif not prompt_path.read_text(encoding="utf-8").strip():
        errors.append("controller prompt is empty")

    fixed_response = _fixed_ir_response(trusted_ir, spec)
    fake = DeterministicFakeProvider(fixed_response)
    trusted_validation = validate_ir_candidate_json(trusted_ir)
    if not trusted_validation.valid or trusted_validation.graph is None:
        raise ValueError("trusted Architecture IR became invalid during validation")
    surface_ready = (
        entrypoint.is_file()
        and static_cli_contract_ok
        and config is not None
        and prompt_path is not None
        and not errors
    )
    if surface_ready:
        prompt = _fake_prompt(spec, trusted_artifact_hash)
        response = fake.complete(prompt)
        response_bytes = len(response.encode("utf-8"))
        if response_bytes > MAX_FAKE_RESPONSE_BYTES:
            errors.append("fake Architecture IR response exceeded the canary byte limit")
        child_validation = validate_ir_candidate_json(response)
        if not child_validation.valid or child_validation.graph is None:
            errors.append(
                "fixed response failed trusted Architecture IR validation: "
                + "; ".join(issue.message for issue in child_validation.issues)
            )
            executable_structure_unchanged = False
            graph_hash_changed = False
        else:
            executable_structure_unchanged = _graph_structure(
                child_validation.graph
            ) == _graph_structure(trusted_validation.graph)
            graph_hash_changed = (
                child_validation.graph_hash != trusted_validation.graph_hash
            )
        if not executable_structure_unchanged:
            errors.append("pre-reviewed mutation changed executable graph structure")
        if not graph_hash_changed:
            errors.append("pre-reviewed mutation did not change the graph document")
    else:
        response_bytes = 0
        child_validation = trusted_validation
        executable_structure_unchanged = True
        graph_hash_changed = False

    return {
        "harness_id": spec.harness_id,
        "display_name": spec.display_name,
        "agent_directory": spec.agent_directory,
        "expected_parent_policy": spec.parent_policy,
        "expected_proposal_policy": spec.proposal_policy,
        "entrypoint_present": entrypoint.is_file(),
        "entrypoint_executed": False,
        "static_cli_contract_passed": static_cli_contract_ok,
        "configuration_present": config_path.is_file(),
        "configuration_parsed": config is not None,
        "prompt_present": prompt_path is not None,
        "local_fixture_calls": fake.calls,
        "real_provider_calls": 0,
        "fixed_response_format": "complete_architecture_ir_json",
        "fixed_response_bytes": response_bytes,
        "fixed_response_byte_limit": MAX_FAKE_RESPONSE_BYTES,
        "candidate_graph_hash_changed": graph_hash_changed,
        "candidate_executable_structure_unchanged": executable_structure_unchanged,
        "candidate_ir_valid": child_validation.valid,
        "candidate_graph_hash": child_validation.graph_hash,
        "candidate_executed": False,
        "training_started": False,
        "passed": not errors,
        "errors": errors,
    }


def validate_controller_surfaces(project_root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate = root / TRUSTED_CANDIDATE_RELATIVE_PATH
    if not candidate.is_file():
        return {
            "passed": False,
            "trusted_candidate": str(candidate),
            "real_provider_calls": 0,
            "local_fixture_calls": 0,
            "candidate_execution_runs": 0,
            "training_runs": 0,
            "entrypoint_execution_runs": 0,
            "harnesses": [],
            "errors": [f"trusted candidate is missing: {candidate}"],
        }
    trusted_ir = candidate.read_text(encoding="utf-8")
    trusted_validation = validate_ir_candidate_json(trusted_ir)
    if not trusted_validation.valid:
        return {
            "passed": False,
            "trusted_candidate": str(candidate),
            "real_provider_calls": 0,
            "local_fixture_calls": 0,
            "candidate_execution_runs": 0,
            "training_runs": 0,
            "entrypoint_execution_runs": 0,
            "harnesses": [],
            "errors": [
                "trusted candidate failed Architecture IR validation: "
                + "; ".join(issue.message for issue in trusted_validation.issues)
            ],
        }
    trusted_hash = _sha256_file(candidate)
    harness_reports = [
        _validate_harness(
            root,
            spec,
            trusted_ir=trusted_ir,
            trusted_artifact_hash=trusted_hash,
        )
        for spec in HARNESSES
    ]
    passed = len(harness_reports) == 4 and all(
        report["passed"] for report in harness_reports
    )
    return {
        "passed": passed,
        "trusted_candidate": str(candidate),
        "trusted_candidate_sha256": trusted_hash,
        "trusted_candidate_graph_hash": trusted_validation.graph_hash,
        "candidate_format": "architecture_ir",
        "real_provider_calls": 0,
        "local_fixture_calls": sum(
            int(report["local_fixture_calls"]) for report in harness_reports
        ),
        "candidate_execution_runs": 0,
        "training_runs": 0,
        "entrypoint_execution_runs": 0,
        "harnesses": harness_reports,
        "errors": [
            error
            for report in harness_reports
            for error in report["errors"]
        ],
    }


def _exact_bool(payload: dict[str, Any], field: str, expected: bool) -> None:
    value = payload.get(field)
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{field} must be exactly {expected}")


def _safe_json_object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 2_000_000:
        raise ValueError(f"{path.name} exceeds the 2 MB evidence limit")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def validate_existing_mps_smoke(
    training_output_dir: str | Path | None,
    *,
    project_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Check smoke artifacts for consistency without proving execution origin."""

    if training_output_dir is None:
        return {
            "provided": False,
            "valid": False,
            "artifact_self_consistent": False,
            "execution_origin_attested": False,
            "training_started_by_validator": False,
            "errors": ["no existing MPS smoke output was provided"],
        }

    output = Path(training_output_dir).resolve()
    root = Path(project_root).resolve()
    trusted_candidate = root / TRUSTED_CANDIDATE_RELATIVE_PATH
    summary_path = output / "training_summary.json"
    manifest_path = output / "training_manifest.json"
    graph_path = output / "candidate_graph.json"
    checkpoint_path = output / "best_checkpoint.pt"
    event_path = output / "training_events.jsonl"
    errors: list[str] = []
    required = (
        summary_path,
        manifest_path,
        graph_path,
        checkpoint_path,
        event_path,
    )
    for path in required:
        if not path.is_file():
            errors.append(f"missing smoke artifact: {path.name}")
        elif path.is_symlink():
            errors.append(f"smoke artifact may not be a symlink: {path.name}")
    if errors:
        return {
            "provided": True,
            "output_dir": str(output),
            "valid": False,
            "artifact_self_consistent": False,
            "execution_origin_attested": False,
            "training_started_by_validator": False,
            "errors": errors,
        }

    try:
        if checkpoint_path.stat().st_size > 100_000_000:
            raise ValueError("best checkpoint exceeds the 100 MB smoke-evidence limit")
        if event_path.stat().st_size > 10_000_000:
            raise ValueError("training event log exceeds the 10 MB evidence limit")
        summary = _safe_json_object(summary_path)
        manifest = _safe_json_object(manifest_path)
        for field, expected in {
            "success": True,
            "scientific": False,
            "hardware_matched": True,
            "unsupported_operation_fallback": False,
            "cleanup_completed": True,
        }.items():
            _exact_bool(summary, field, expected)
        trusted_ir_validation = validate_ir_candidate_json(
            trusted_candidate.read_text(encoding="utf-8")
        )
        if not trusted_ir_validation.valid or trusted_ir_validation.graph is None:
            raise ValueError("trusted initial candidate is not valid Architecture IR")
        expected_graph_hash = trusted_ir_validation.graph_hash
        stored_artifact_hash = _sha256_file(graph_path)
        stored_ir_validation = validate_ir_candidate_json(
            graph_path.read_text(encoding="utf-8")
        )
        if not stored_ir_validation.valid:
            raise ValueError("stored candidate graph is not valid Architecture IR")
        if stored_ir_validation.graph_hash != expected_graph_hash:
            raise ValueError(
                "stored candidate graph differs from the trusted initial candidate"
            )
        expected_summary = {
            "profile_name": SMOKE_TRAIN_V1.name,
            "profile_version": SMOKE_TRAIN_V1.version,
            "profile_hash": SMOKE_TRAIN_V1.profile_hash,
            # TrainingResult retains this legacy field name for schema
            # compatibility; the manifest below records the precise IR terms.
            # Native controllers canonicalize the trusted seed before training,
            # while the standalone trainer preserves its original formatting.
            # In both cases identity must bind to the immutable stored artifact;
            # semantic equality to the checked-in seed is verified by graph hash.
            "candidate_source_hash": stored_artifact_hash,
            "device": "mps",
            "dtype": "float32",
            "steps_completed": SMOKE_TRAIN_V1.max_steps,
            "examples_processed": (
                SMOKE_TRAIN_V1.max_steps * SMOKE_TRAIN_V1.global_batch_size
            ),
        }
        for field, expected in expected_summary.items():
            if summary.get(field) != expected:
                raise ValueError(f"training summary {field} does not match smoke_train_v1")
        for field in (
            "best_development_loss",
            "final_training_loss",
            "train_seconds",
        ):
            value = summary.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"training summary {field} must be finite numeric data")

        for field, expected in {
            "allow_cpu_for_tests": False,
            "hardware_matched_scientific_run": False,
        }.items():
            _exact_bool(manifest, field, expected)
        expected_manifest = {
            "candidate_source_hash": expected_summary["candidate_source_hash"],
            "candidate_artifact_hash": stored_artifact_hash,
            "candidate_format": "architecture_ir",
            "candidate_graph_hash": expected_graph_hash,
            "profile_hash": SMOKE_TRAIN_V1.profile_hash,
            "requested_device": "mps",
            "selected_device": "mps",
            "parameter_count_role": "descriptive_metadata_only",
            "isolation_level": "engineering_only_or_scientific_gate_blocked",
        }
        for field, expected in expected_manifest.items():
            if manifest.get(field) != expected:
                raise ValueError(f"training manifest {field} does not match the smoke run")
        runtime = manifest.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError("training manifest lacks runtime evidence")
        _exact_bool(runtime, "mps_built", True)
        _exact_bool(runtime, "mps_available", True)
        if runtime.get("pytorch_enable_mps_fallback") not in {"", "0"}:
            raise ValueError("training manifest requested MPS fallback")
        decision = manifest.get("containment_decision")
        if not isinstance(decision, dict):
            raise ValueError("training manifest lacks containment decision")
        _exact_bool(decision, "allowed", True)
        _exact_bool(decision, "scientific", False)
        audit = manifest.get("containment_audit")
        if not isinstance(audit, dict):
            raise ValueError("training manifest lacks containment audit")
        if audit.get("visible_credential_names") not in ([], ()):
            raise ValueError("candidate worker observed credential-like environment names")

        if _sha256_file(graph_path) != stored_artifact_hash:
            raise ValueError("stored candidate graph changed during validation")
        if _sha256_file(checkpoint_path) != summary.get("checkpoint_sha256"):
            raise ValueError("best checkpoint hash does not match the training summary")
        if Path(str(summary.get("checkpoint_path", ""))).name != checkpoint_path.name:
            raise ValueError("training summary names an unexpected checkpoint")
        if Path(str(summary.get("event_log_path", ""))).name != event_path.name:
            raise ValueError("training summary names an unexpected event log")

        events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(events) != SMOKE_TRAIN_V1.max_steps:
            raise ValueError("smoke event log does not contain exactly ten optimizer steps")
        if [event.get("optimizer_step") for event in events] != list(
            range(1, SMOKE_TRAIN_V1.max_steps + 1)
        ):
            raise ValueError("smoke optimizer-step sequence is not contiguous")
        if any(
            isinstance(event.get("loss"), bool)
            or not isinstance(event.get("loss"), (int, float))
            or not math.isfinite(event["loss"])
            for event in events
        ):
            raise ValueError("smoke event log contains a non-finite loss")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("model_state"), dict
        ):
            raise ValueError("best checkpoint lacks a model_state mapping")
        interpreted = load_and_build_ir_candidate(
            graph_path,
            int(summary["initialization_seed"]),
        )
        initial_model = interpreted.model
        initial_state = initial_model.state_dict()
        trained_state = checkpoint["model_state"]
        if set(initial_state) != set(trained_state):
            raise ValueError("checkpoint model state does not match the trusted architecture")
        for name, initial_value in initial_state.items():
            trained_value = trained_state[name]
            if not isinstance(trained_value, torch.Tensor):
                raise ValueError(f"checkpoint state {name} is not a tensor")
            if trained_value.shape != initial_value.shape:
                raise ValueError(f"checkpoint state {name} has the wrong shape")
            if trained_value.dtype != initial_value.dtype:
                raise ValueError(f"checkpoint state {name} has the wrong dtype")
        parameters_changed = any(
            not torch.equal(initial_state[name], trained_state[name])
            for name in initial_state
        )
        if not parameters_changed:
            raise ValueError("smoke checkpoint is identical to fresh initialization")
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        errors.append(f"{type(error).__name__}: {error}")
        parameters_changed = False

    valid = not errors
    return {
        "provided": True,
        "output_dir": str(output),
        "valid": valid,
        "artifact_self_consistent": valid,
        "execution_origin_attested": False,
        "training_started_by_validator": False,
        "profile": SMOKE_TRAIN_V1.name,
        "scientific": False,
        "claim_scope": "self_authored_artifact_consistency_only",
        "parameters_changed": parameters_changed,
        "errors": errors,
    }


def build_report(
    *,
    project_root: str | Path = ROOT,
    mps_smoke_output: str | Path | None = None,
) -> dict[str, Any]:
    surfaces = validate_controller_surfaces(project_root)
    mps = validate_existing_mps_smoke(
        mps_smoke_output,
        project_root=project_root,
    )
    return {
        "schema_name": "FourHarnessStaticSurfaceReport",
        "schema_version": "2.0",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "static_controller_surfaces_and_optional_smoke_artifact_consistency",
        "status": "static_controller_surfaces_passed" if surfaces["passed"] else "blocked",
        "static_controller_surfaces_passed": surfaces["passed"],
        "mps_smoke_artifacts_self_consistent": mps["artifact_self_consistent"],
        "mps_execution_origin_attested": False,
        "provider_calls": 0,
        "local_fixture_calls": surfaces["local_fixture_calls"],
        "entrypoint_execution_runs": 0,
        "candidate_execution_runs": 0,
        "training_runs": 0,
        "scientific": False,
        "scientific_pilot_ready": False,
        "autonomous_generated_candidate_execution_ready": False,
        "static_controller_surfaces": surfaces,
        "existing_mps_smoke_artifacts": mps,
        "limitations": [
            "The fake provider is local and deterministic; no provider connectivity was tested.",
            "Entrypoints are parsed statically and never imported or executed.",
            "The fixed response is not injected into a live controller.",
            "The complete child IR was validated but never constructed or executed.",
            "Smoke artifacts are self-authored and cannot attest that MPS execution occurred.",
            "Smoke artifact consistency, when supplied, covers only the trusted checked-in seed.",
            "This report cannot authorize live generated-candidate execution or a scientific pilot.",
            "Use scripts/audit_scientific_readiness.py for the separate fail-closed scientific audit.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Statically validate four controller surfaces without executing them."
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--mps-smoke-output",
        type=Path,
        help="optional smoke_train_v1 output to check for artifact self-consistency",
    )
    parser.add_argument("--require-mps-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = build_report(
        project_root=arguments.project_root,
        mps_smoke_output=arguments.mps_smoke_output,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    passed = report["static_controller_surfaces_passed"]
    if arguments.require_mps_smoke:
        passed = passed and report["mps_smoke_artifacts_self_consistent"]
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
