"""Exact evidence contracts for provider-free Modal network isolation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from common.runtime_context import ExecutionContextV1

NETWORK_DENIAL_PROBE_SCHEMA_NAME = "ProviderFreeNetworkDenialProbe"
NETWORK_DENIAL_PROBE_SCHEMA_VERSION = "1.0"
NETWORK_DENIAL_PROBE_IP = "1.1.1.1"
NETWORK_DENIAL_PROBE_PORT = 443
NETWORK_DENIAL_PROBE_TIMEOUT_SECONDS = 1.0
NETWORK_DENIAL_EXCEPTION_TYPE = "PermissionError"
NETWORK_UNREACHABLE_EXCEPTION_TYPE = "NetworkUnreachableError"
NETWORK_DENIAL_EXCEPTION_TYPES = frozenset(
    {NETWORK_DENIAL_EXCEPTION_TYPE, NETWORK_UNREACHABLE_EXCEPTION_TYPE}
)
PROVIDER_FREE_MODAL_FUNCTIONS = frozenset(
    {"candidate_smoke", "checkpoint_resume", "cuda_environment", "offline_smoke"}
)

_PROBE_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "attempted_endpoint",
        "timeout_seconds",
        "denied",
        "exception_type",
        "execution_context",
    }
)
_ENDPOINT_FIELDS = frozenset({"ip", "port"})

_ACTION_OUTER_ROSTERS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "cuda_environment": (
        frozenset(
            {
                "artifact_manifest.json",
                "cuda_environment.json",
                "execution_context.json",
                "image_source_manifest.json",
                "provider_free_network_denial_probe.json",
                "remote_action_result.json",
            }
        ),
        frozenset(),
    ),
    "offline_smoke": (
        frozenset(
            {
                "artifact_manifest.json",
                "execution_context.json",
                "image_source_manifest.json",
                "provider_free_network_denial_probe.json",
                "remote_action_result.json",
            }
        ),
        frozenset({"offline_study"}),
    ),
    "candidate_smoke": (
        frozenset(
            {
                "artifact_manifest.checkpoint.json",
                "execution_context.json",
                "image_source_manifest.json",
                "provider_free_network_denial_probe.json",
                "remote_action_result.json",
            }
        ),
        frozenset({"candidate_smoke"}),
    ),
    "checkpoint_resume": (
        frozenset(
            {
                "artifact_manifest.json",
                "execution_context.json",
                "image_source_manifest.json",
                "provider_free_network_denial_probe.json",
                "resume_action_result.json",
                "resume_contract_verification.json",
                "resume_execution_context.json",
                "resume_progression_verification.json",
                "resume_source_binding.json",
            }
        ),
        frozenset({"candidate_smoke"}),
    ),
}


def validate_provider_free_network_denial_probe(
    payload: Mapping[str, Any],
    *,
    expected_context: ExecutionContextV1,
) -> None:
    """Reject any denial probe that is extensible, coerced, or context-mixed."""

    if not isinstance(payload, Mapping) or set(payload) != _PROBE_FIELDS:
        raise ValueError("network denial probe has an invalid exact schema")
    if (
        payload["schema_name"] != NETWORK_DENIAL_PROBE_SCHEMA_NAME
        or payload["schema_version"] != NETWORK_DENIAL_PROBE_SCHEMA_VERSION
    ):
        raise ValueError("network denial probe has the wrong schema contract")
    endpoint = payload["attempted_endpoint"]
    if not isinstance(endpoint, Mapping) or set(endpoint) != _ENDPOINT_FIELDS:
        raise ValueError("network denial probe endpoint has an invalid exact schema")
    if endpoint["ip"] != NETWORK_DENIAL_PROBE_IP:
        raise ValueError("network denial probe did not use the frozen routable IP")
    if type(endpoint["port"]) is not int or endpoint["port"] != (
        NETWORK_DENIAL_PROBE_PORT
    ):
        raise ValueError("network denial probe did not use the frozen HTTPS port")
    if type(payload["timeout_seconds"]) is not float or payload[
        "timeout_seconds"
    ] != NETWORK_DENIAL_PROBE_TIMEOUT_SECONDS:
        raise ValueError("network denial probe did not use the exact bounded timeout")
    if payload["denied"] is not True or type(payload["denied"]) is not bool:
        raise ValueError("network denial probe did not record exact denial")
    exception_type = payload["exception_type"]
    if (
        type(exception_type) is not str
        or exception_type not in NETWORK_DENIAL_EXCEPTION_TYPES
    ):
        raise ValueError("network denial probe exception classification is unsafe")
    if not isinstance(expected_context, ExecutionContextV1):
        raise TypeError("expected_context must be an ExecutionContextV1")
    if (
        expected_context.execution_backend != "modal"
        or expected_context.function_name not in PROVIDER_FREE_MODAL_FUNCTIONS
    ):
        raise ValueError("network denial probe expected context is not provider-free")
    try:
        observed_context = ExecutionContextV1.from_dict(payload["execution_context"])
    except (TypeError, ValueError) as error:
        raise ValueError("network denial probe execution context is invalid") from error
    if observed_context != expected_context:
        raise ValueError("network denial probe is mixed with another execution context")


def validate_provider_free_action_outer_roster(
    root: str | Path,
    *,
    function_name: str,
) -> None:
    """Require the exact top-level files and directories emitted by one action."""

    policy = _ACTION_OUTER_ROSTERS.get(function_name)
    if policy is None:
        raise ValueError("provider-free action has no frozen outer artifact roster")
    raw_root = Path(root)
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError("provider-free action root must be a regular directory")
    expected_files, expected_directories = policy
    entries = tuple(raw_root.iterdir())
    observed = {entry.name for entry in entries}
    expected = expected_files | expected_directories
    if observed != expected:
        raise ValueError(
            "provider-free outer artifact roster differs from the exact action policy "
            f"(missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)})"
        )
    for entry in entries:
        if entry.is_symlink():
            raise ValueError(
                f"provider-free outer artifact may not be a symlink: {entry.name}"
            )
        if entry.name in expected_files and not entry.is_file():
            raise ValueError(
                f"provider-free outer artifact must be a regular file: {entry.name}"
            )
        if entry.name in expected_directories and not entry.is_dir():
            raise ValueError(
                f"provider-free outer artifact must be a regular directory: {entry.name}"
            )
