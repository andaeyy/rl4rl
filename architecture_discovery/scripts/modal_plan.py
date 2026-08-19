"""Print the bounded Modal plan without importing Modal or contacting a service."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.evolution_run import (  # noqa: E402
    EVOLUTION_COMPLETION_TOKENS_PER_REQUEST,
    EVOLUTION_HARNESSES,
    EVOLUTION_INPUT_BYTES_PER_REQUEST,
    EVOLUTION_MAX_ITERATIONS,
    EvolutionRunSpec,
)
from modal_boundary import (  # noqa: E402
    APP_NAME,
    FUNCTION_SPECS,
    IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS,
    IMAGE_BUILD_CPU_REQUEST_CORES,
    IMAGE_BUILD_MEMORY_REQUEST_MIB,
    IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT,
    IMAGE_RECIPE_VERSION,
    MODAL_VERSION,
    OPENEVOLVE_60_FUNCTION_NAME,
    OPENEVOLVE_60_INPUT_BYTES_PER_REQUEST,
    OPENEVOLVE_60_ITERATIONS,
    PYTHON_VERSION,
    UV_VERSION,
    VOLUME_MOUNT_PATH,
    VOLUME_NAME,
    build_image_source_manifest,
    function_spec,
)


def build_plan(project_root: str | Path = ROOT) -> dict[str, Any]:
    source = build_image_source_manifest(project_root)
    source_total_bytes = sum(item.size_bytes for item in source.files)
    return {
        "schema_name": "ModalExecutionPlan",
        "schema_version": "1.2",
        "app": APP_NAME,
        "volume": VOLUME_NAME,
        "mount": str(VOLUME_MOUNT_PATH),
        "python": PYTHON_VERSION,
        "uv": UV_VERSION,
        "modal_sdk": MODAL_VERSION,
        "image_recipe": IMAGE_RECIPE_VERSION,
        "image_build": {
            "cpu_request_cores": IMAGE_BUILD_CPU_REQUEST_CORES,
            "cpu_soft_limit_cores": None,
            "memory_request_mib": IMAGE_BUILD_MEMORY_REQUEST_MIB,
            "memory_limit_mib": None,
            "gpu": None,
            "region": None,
            "timeout_seconds": IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS,
            "subprocess_thread_limit": IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT,
            "resource_limits_exposed": False,
            "platform_compute_cost_ceiling_enforced": False,
            "network_required": True,
            "source_copy_layers": 2,
            "source_copy_backend_resource_limits_exposed": False,
        },
        "image_source_sha256": source.manifest_sha256,
        "image_source": {
            "file_count": len(source.files),
            "total_bytes": source_total_bytes,
            "copy_source_bytes_upper_bound": 2 * source_total_bytes,
        },
        "dependency_lock_sha256": source.dependency_lock_sha256,
        "runtime_functions_preemptible": True,
        "platform_preemption_restart_possible": True,
        "logical_call_count_is_not_container_attempt_ceiling": True,
        "modal_cost_gate": {
            "price_basis_schema": "ModalPriceBasis/1.0",
            "price_basis_max_age_hours": 48,
            "scope": (
                "local_pre_popen_request_rate_and_one_gib_month_storage_"
                "estimate_not_platform_billing_cap"
            ),
            "platform_billing_cap_enforced": False,
        },
        "functions": {
            name: {
                "cpu_request_cores": spec.cpu_request_cores,
                "cpu_soft_limit_cores": spec.cpu_soft_limit_cores,
                "cpu_limit_kind": "soft_throttle_threshold",
                "memory_request_mib": spec.memory_request_mib,
                "memory_limit_mib": spec.memory_limit_mib,
                "memory_limit_kind": "hard",
                "platform_compute_cost_ceiling_enforced": False,
                "gpu": spec.gpu,
                "region": spec.region,
                "timeout_seconds": spec.timeout_seconds,
                "max_containers": spec.max_containers,
                "min_containers": spec.min_containers,
                "retries": spec.retries,
                "provider_secret": spec.provider_secret,
                "runtime_network_blocked": not spec.provider_secret,
                "volume_mount_path": spec.volume_mount_path,
            }
            for name, spec in FUNCTION_SPECS.items()
        },
        "exploratory_function": {
            "name": function_spec("exploratory_c0c3_pilot").name,
            "gpu": function_spec("exploratory_c0c3_pilot").gpu,
            "provider_secret": function_spec("exploratory_c0c3_pilot").provider_secret,
            "runtime_network_blocked": False,
            "retries": function_spec("exploratory_c0c3_pilot").retries,
            "timeout_seconds": function_spec("exploratory_c0c3_pilot").timeout_seconds,
            "max_containers": function_spec("exploratory_c0c3_pilot").max_containers,
            "mode": "exploratory_non_scientific",
        },
        "openevolve_60_function": {
            "name": function_spec(OPENEVOLVE_60_FUNCTION_NAME).name,
            "gpu": function_spec(OPENEVOLVE_60_FUNCTION_NAME).gpu,
            "provider_secret": function_spec(
                OPENEVOLVE_60_FUNCTION_NAME
            ).provider_secret,
            "runtime_network_blocked": False,
            "retries": function_spec(OPENEVOLVE_60_FUNCTION_NAME).retries,
            "timeout_seconds": function_spec(
                OPENEVOLVE_60_FUNCTION_NAME
            ).timeout_seconds,
            "max_containers": function_spec(
                OPENEVOLVE_60_FUNCTION_NAME
            ).max_containers,
            "iterations": OPENEVOLVE_60_ITERATIONS,
            "input_bytes_per_request_ceiling": (
                OPENEVOLVE_60_INPUT_BYTES_PER_REQUEST
            ),
            "training_profile": "smoke_train_cuda_v2",
            "scientific": False,
            "mode": "bounded_non_scientific_openevolve_60",
        },
        "configurable_evolution_function": {
            "name": function_spec("evolution_run").name,
            "gpu": function_spec("evolution_run").gpu,
            "provider_secret": function_spec("evolution_run").provider_secret,
            "runtime_network_blocked": False,
            "retries": function_spec("evolution_run").retries,
            "max_containers": function_spec("evolution_run").max_containers,
            "harnesses": list(EVOLUTION_HARNESSES),
            "minimum_iterations": 1,
            "maximum_iterations": EVOLUTION_MAX_ITERATIONS,
            "maximum_function_timeout_seconds": EvolutionRunSpec(
                EVOLUTION_HARNESSES[0], EVOLUTION_MAX_ITERATIONS
            ).function_timeout_seconds,
            "input_bytes_per_request_ceiling": (
                EVOLUTION_INPUT_BYTES_PER_REQUEST
            ),
            "completion_tokens_per_request_ceiling": (
                EVOLUTION_COMPLETION_TOKENS_PER_REQUEST
            ),
            "training_profile": "smoke_train_cuda_v2",
            "scientific": False,
            "mode": "bounded_non_scientific_evolution",
        },
        "remote_calls_started": 0,
    }


def main() -> None:
    print(json.dumps(build_plan(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
