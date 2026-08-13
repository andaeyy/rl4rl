from __future__ import annotations

import ast
import builtins
import errno
import hashlib
import importlib.util
import inspect
import json
import os
import stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from common.runtime_context import ExecutionContextV1

ROOT = Path(__file__).resolve().parents[1]
_SOURCE_TREE_SHA256 = "9" * 64
_ATTEMPT_ID = "e" * 32


def _cohort_identity(image_source_sha256: str):
    from modal_boundary import ModalLiveCohortIdentity

    return ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256=image_source_sha256,
        cohort_id="test-cohort-1",
    )


def _imported_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", 1)[0])
    return names


def test_only_modal_app_imports_the_optional_sdk() -> None:
    assert "modal" not in _imported_module_names(ROOT / "modal_boundary.py")
    assert "modal" not in _imported_module_names(ROOT / "common" / "runtime_context.py")
    assert "modal" in _imported_module_names(ROOT / "modal_app.py")


def test_modal_app_imports_cleanly_when_sdk_is_absent(monkeypatch) -> None:
    original_import = builtins.__import__

    def without_modal(name, *args, **kwargs):
        if name == "modal":
            raise ModuleNotFoundError("No module named 'modal'", name="modal")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_modal)
    spec = importlib.util.spec_from_file_location(
        "modal_app_without_sdk", ROOT / "modal_app.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.modal is None
    assert module.app is None
    assert module.IMAGE is None
    assert module.ARTIFACT_VOLUME is None
    assert module.PROVIDER_SECRET is None
    assert module.IMAGE_SOURCE_MANIFEST.files


def test_pure_boundary_import_has_no_remote_side_effect_surface() -> None:
    import modal_boundary

    assert not hasattr(modal_boundary, "app")
    assert not hasattr(modal_boundary, "Secret")
    assert not hasattr(modal_boundary, "Volume")


def test_pinned_modal_sdk_exposes_runtime_but_not_image_build_limit_tuples() -> None:
    modal_sdk = pytest.importorskip("modal")

    assert modal_sdk.__version__ == "1.5.3"
    function_signature = inspect.signature(modal_sdk.App.function)
    image_build_signature = inspect.signature(modal_sdk.Image.run_function)
    assert "tuple[float, float]" in str(
        function_signature.parameters["cpu"].annotation
    )
    assert "tuple[int, int]" in str(
        function_signature.parameters["memory"].annotation
    )
    assert "tuple" not in str(image_build_signature.parameters["cpu"].annotation)
    assert "tuple" not in str(
        image_build_signature.parameters["memory"].annotation
    )


def test_provider_canary_aggregate_exit_code_is_nonzero_after_any_failure() -> None:
    from modal_app import _provider_canary_aggregate_exit_code
    from modal_boundary import CANARY_ORDER, canary_run_suffix

    outcomes = [
        {
            "harness": harness,
            "run_id": f"canary-{canary_run_suffix(harness)}",
            "status": "success",
            "result": {"ok": True},
            "error_type": None,
        }
        for harness in CANARY_ORDER
    ]
    aggregate = {
        "schema_name": "ModalProviderCanaryAggregateResult",
        "schema_version": "1.0",
        "run_id_prefix": "canary",
        "harness_order": list(CANARY_ORDER),
        "outcomes": outcomes,
        "all_succeeded": True,
    }
    assert _provider_canary_aggregate_exit_code(aggregate) == 0

    failed = json.loads(json.dumps(aggregate))
    failed["outcomes"][1].update(
        {"status": "failed", "result": None, "error_type": "RuntimeError"}
    )
    failed["all_succeeded"] = False
    assert _provider_canary_aggregate_exit_code(failed) == 2


def test_provider_canary_aggregate_outcome_is_create_only_and_reopened(
    tmp_path,
    monkeypatch,
) -> None:
    import modal_app
    from modal_boundary import (
        CANARY_ORDER,
        MODAL_ACTION_ATTEMPT_ID_ENV,
        canary_run_suffix,
        validate_provider_canary_aggregate_outcome_receipt,
    )

    attempt_id = "c" * 32
    identity = _cohort_identity(modal_app.IMAGE_SOURCE_SHA256)
    monkeypatch.setattr(modal_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv(MODAL_ACTION_ATTEMPT_ID_ENV, attempt_id)
    aggregate_logical = modal_app.provider_canary_aggregate_outcome_receipt_path(
        identity,
        attempt_id,
    )
    tmp_path.joinpath(*aggregate_logical.parts).parent.mkdir(parents=True)
    aggregate = {
        "schema_name": "ModalProviderCanaryAggregateResult",
        "schema_version": "1.0",
        "run_id_prefix": "canary",
        "harness_order": list(CANARY_ORDER),
        "outcomes": [
            {
                "harness": harness,
                "run_id": f"canary-{canary_run_suffix(harness)}",
                "status": "failed" if index == 2 else "success",
                "result": None if index == 2 else {"private": "discarded"},
                "error_type": "RuntimeError" if index == 2 else None,
            }
            for index, harness in enumerate(CANARY_ORDER)
        ],
        "all_succeeded": False,
    }

    path = modal_app._persist_provider_canary_aggregate_outcome(
        aggregate,
        identity=identity,
        attempt_id=attempt_id,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path.joinpath(*aggregate_logical.parts)
    assert "private" not in json.dumps(payload)
    assert path.stat().st_nlink == 1
    assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
    assert not list(path.parent.glob(".provider-canary-aggregate-*.tmp"))
    assert (
        validate_provider_canary_aggregate_outcome_receipt(
            payload,
            expected_attempt_id=attempt_id,
            expected_run_id_prefix="canary",
            expected_source_tree_sha256=identity.source_tree_sha256,
            expected_image_source_sha256=modal_app.IMAGE_SOURCE_SHA256,
            expected_cohort_id=identity.cohort_id,
        )
        == payload
    )
    with pytest.raises(FileExistsError):
        modal_app._persist_provider_canary_aggregate_outcome(
            aggregate,
            identity=identity,
            attempt_id=attempt_id,
        )


@pytest.mark.parametrize("link_kind", ("parent", "destination"))
def test_provider_canary_aggregate_outcome_rejects_symlinks(
    tmp_path,
    monkeypatch,
    link_kind: str,
) -> None:
    import modal_app
    from modal_boundary import (
        CANARY_ORDER,
        MODAL_ACTION_ATTEMPT_ID_ENV,
        canary_run_suffix,
    )

    attempt_id = "d" * 32
    identity = _cohort_identity(modal_app.IMAGE_SOURCE_SHA256)
    monkeypatch.setattr(modal_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv(MODAL_ACTION_ATTEMPT_ID_ENV, attempt_id)
    aggregate_logical = modal_app.provider_canary_aggregate_outcome_receipt_path(
        identity,
        attempt_id,
    )
    aggregate_path = tmp_path.joinpath(*aggregate_logical.parts)
    aggregate_path.parent.parent.mkdir(parents=True)
    target = tmp_path / "outside"
    if link_kind == "parent":
        target.mkdir()
        aggregate_path.parent.symlink_to(
            target,
            target_is_directory=True,
        )
        expected_error = ValueError
        expected_message = "symlink|unsafe"
    else:
        aggregate_path.parent.mkdir()
        target.write_text("must remain unchanged\n", encoding="utf-8")
        aggregate_path.symlink_to(target)
        expected_error = FileExistsError
        expected_message = "exists"
    aggregate = {
        "schema_name": "ModalProviderCanaryAggregateResult",
        "schema_version": "1.0",
        "run_id_prefix": "canary",
        "harness_order": list(CANARY_ORDER),
        "outcomes": [
            {
                "harness": harness,
                "run_id": f"canary-{canary_run_suffix(harness)}",
                "status": "success",
                "result": {"ok": True},
                "error_type": None,
            }
            for harness in CANARY_ORDER
        ],
        "all_succeeded": True,
    }

    with pytest.raises(expected_error, match=expected_message):
        modal_app._persist_provider_canary_aggregate_outcome(
            aggregate,
            identity=identity,
            attempt_id=attempt_id,
        )

    if link_kind == "parent":
        assert not list(target.iterdir())
    else:
        assert target.read_text(encoding="utf-8") == "must remain unchanged\n"


def test_modal_volume_lookup_fails_closed_when_resource_is_missing() -> None:
    tree = ast.parse(
        (ROOT / "modal_app.py").read_text(encoding="utf-8"),
        filename=str(ROOT / "modal_app.py"),
    )
    volume_lookups = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_name"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "Volume"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "modal"
    ]
    assert len(volume_lookups) == 1
    create_keywords = {
        keyword.arg: keyword.value for keyword in volume_lookups[0].keywords
    }
    create_if_missing = create_keywords["create_if_missing"]
    assert isinstance(create_if_missing, ast.Constant)
    assert create_if_missing.value is False


def test_finalize_run_rejects_artifacts_above_download_bounds_before_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import modal_app
    import modal_boundary

    run_id = "oversized-finalize-1"
    run_directory = tmp_path / run_id
    run_directory.mkdir()
    (run_directory / "one.bin").write_bytes(b"12345678")
    (run_directory / "two.bin").write_bytes(b"abcdefgh")

    class FakeVolume:
        commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = FakeVolume()
    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", volume)
    monkeypatch.setattr(modal_boundary, "MAX_ARTIFACT_DOWNLOAD_FILE_BYTES", 128)
    monkeypatch.setattr(modal_boundary, "MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES", 20)

    with pytest.raises(
        modal_boundary.ArtifactIntegrityError,
        match="aggregate download byte cap",
    ):
        modal_app._finalize_run(
            run_directory,
            run_id=run_id,
            result={"ok": True},
        )

    assert (run_directory / "remote_action_result.json").is_file()
    assert not (run_directory / "artifact_manifest.json").exists()
    assert volume.commits == 0


@pytest.mark.parametrize(
    "function_name",
    (
        "offline_smoke",
        "cuda_environment",
        "candidate_smoke",
        "checkpoint_resume",
    ),
)
def test_provider_free_network_probe_records_exact_denial_without_network(
    function_name: str,
) -> None:
    import modal_app

    context = ExecutionContextV1(
        execution_backend="modal",
        run_id=f"probe-{function_name.replace('_', '-')}",
        app_name=modal_app.APP_NAME,
        function_name=function_name,
        modal_app_id="ap-probe123",
        modal_function_id="fu-probe123",
        modal_call_id="fc-probe123",
        modal_image_id="im-probe123",
        image_source_sha256="a" * 64,
        artifact_uri=(
            f"volume://{modal_app.VOLUME_NAME}/runs/"
            f"probe-{function_name.replace('_', '-')}"
        ),
    )
    calls = []

    def denied_connector(endpoint, timeout):
        calls.append((endpoint, timeout))
        raise PermissionError("sensitive platform-specific denial detail")

    result = modal_app._provider_free_network_denial_probe(
        context,
        connector=denied_connector,
    )

    assert set(result) == {
        "schema_name",
        "schema_version",
        "attempted_endpoint",
        "timeout_seconds",
        "denied",
        "exception_type",
        "execution_context",
    }
    assert result["schema_name"] == "ProviderFreeNetworkDenialProbe"
    assert result["schema_version"] == "1.0"
    assert result["attempted_endpoint"] == {
        "ip": "1.1.1.1",
        "port": 443,
    }
    assert result["timeout_seconds"] == 1.0
    assert result["denied"] is True
    assert result["exception_type"] == "PermissionError"
    assert result["execution_context"] == context.to_dict()
    assert calls == [(('1.1.1.1', 443), 1.0)]
    assert "sensitive" not in json.dumps(result)


def test_provider_free_network_probe_quarantines_successful_connection() -> None:
    import modal_app

    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="probe-success-1",
        app_name=modal_app.APP_NAME,
        function_name="offline_smoke",
        modal_app_id="ap-probe123",
        modal_function_id="fu-probe123",
        modal_call_id="fc-probe123",
        modal_image_id="im-probe123",
        image_source_sha256="a" * 64,
        artifact_uri=(
            f"volume://{modal_app.VOLUME_NAME}/runs/probe-success-1"
        ),
    )

    class UnexpectedConnection:
        closed = False

        def close(self) -> None:
            self.closed = True

    connection = UnexpectedConnection()
    with pytest.raises(modal_app.ProviderFreeNetworkExposureError):
        modal_app._provider_free_network_denial_probe(
            context,
            connector=lambda _endpoint, _timeout: connection,
        )
    assert connection.closed is True


@pytest.mark.parametrize(
    "connection_error",
    (
        TimeoutError("reachable endpoint was slow"),
        ConnectionRefusedError(errno.ECONNREFUSED, "reachable endpoint refused"),
    ),
)
def test_provider_free_network_probe_rejects_uncertain_connection_failures(
    connection_error: OSError,
) -> None:
    import modal_app

    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="probe-uncertain-1",
        app_name=modal_app.APP_NAME,
        function_name="offline_smoke",
        modal_app_id="ap-probe123",
        modal_function_id="fu-probe123",
        modal_call_id="fc-probe123",
        modal_image_id="im-probe123",
        image_source_sha256="a" * 64,
        artifact_uri=f"volume://{modal_app.VOLUME_NAME}/runs/probe-uncertain-1",
    )

    with pytest.raises(
        modal_app.ProviderFreeNetworkExposureError,
        match="denial could not be proven",
    ):
        modal_app._provider_free_network_denial_probe(
            context,
            connector=lambda *_args: (_ for _ in ()).throw(connection_error),
        )


@pytest.mark.parametrize("policy_errno", (errno.EACCES, errno.EPERM))
def test_provider_free_network_probe_accepts_exact_permission_policy_errno(
    policy_errno: int,
) -> None:
    import modal_app

    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="probe-policy-denial-1",
        app_name=modal_app.APP_NAME,
        function_name="offline_smoke",
        modal_app_id="ap-probe123",
        modal_function_id="fu-probe123",
        modal_call_id="fc-probe123",
        modal_image_id="im-probe123",
        image_source_sha256="a" * 64,
        artifact_uri=(
            f"volume://{modal_app.VOLUME_NAME}/runs/probe-policy-denial-1"
        ),
    )

    result = modal_app._provider_free_network_denial_probe(
        context,
        connector=lambda *_args: (_ for _ in ()).throw(
            OSError(policy_errno, "platform policy denied")
        ),
    )
    assert result["denied"] is True
    assert result["exception_type"] == "PermissionError"


def test_provider_free_network_probe_accepts_modal_network_unreachable() -> None:
    import modal_app
    from common.network_denial import validate_provider_free_network_denial_probe

    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="probe-network-unreachable-1",
        app_name=modal_app.APP_NAME,
        function_name="cuda_environment",
        modal_app_id="ap-probe123",
        modal_function_id="fu-probe123",
        modal_call_id="fc-probe123",
        modal_image_id="im-probe123",
        image_source_sha256="a" * 64,
        artifact_uri=(
            f"volume://{modal_app.VOLUME_NAME}/runs/"
            "probe-network-unreachable-1"
        ),
    )

    result = modal_app._provider_free_network_denial_probe(
        context,
        connector=lambda *_args: (_ for _ in ()).throw(
            OSError(errno.ENETUNREACH, "network is unreachable")
        ),
    )

    assert result["denied"] is True
    assert result["exception_type"] == "NetworkUnreachableError"
    validate_provider_free_network_denial_probe(
        result,
        expected_context=context,
    )


@pytest.mark.parametrize(
    "function_name",
    ("artifact_verify", "canary_greedy_autoresearch"),
)
def test_network_denial_probe_excludes_verifier_and_provider_canaries(
    function_name: str,
) -> None:
    import modal_app

    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="excluded-probe-1",
        app_name=modal_app.APP_NAME,
        function_name=function_name,
        modal_app_id="ap-probe123",
        modal_function_id="fu-probe123",
        modal_call_id="fc-probe123",
        modal_image_id="im-probe123",
        image_source_sha256="a" * 64,
        artifact_uri=(
            f"volume://{modal_app.VOLUME_NAME}/runs/excluded-probe-1"
        ),
    )
    called = False

    def forbidden_connector(_endpoint, _timeout):
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="not provider-free"):
        modal_app._provider_free_network_denial_probe(
            context,
            connector=forbidden_connector,
        )
    assert called is False


def test_network_probe_roster_covers_every_provider_free_execution_action() -> None:
    import modal_app

    expected = {
        name
        for name, spec in modal_app.FUNCTION_SPECS.items()
        if not spec.provider_secret and name != "artifact_verify"
    }
    assert expected == modal_app.PROVIDER_FREE_NETWORK_PROBE_FUNCTIONS


def test_modal_subprocess_environment_is_narrow_and_provider_scoped(
    monkeypatch,
) -> None:
    import modal_app

    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="modal-env-1",
        app_name="rl4rl-architecture-discovery",
        function_name="candidate_smoke",
        modal_app_id=None,
        modal_function_id=None,
        modal_call_id="fc-test",
        modal_image_id=None,
        image_source_sha256="a" * 64,
        artifact_uri=("volume://rl4rl-architecture-artifacts/runs/modal-env-1"),
    )
    operational_collisions = {
        "CUDA_HOME": "/secret-controlled/cuda-home",
        "CUDA_PATH": "/secret-controlled/cuda-path",
        "CUDA_VISIBLE_DEVICES": "secret-controlled-device",
        "LANG": "secret-controlled-locale",
        "LC_ALL": "secret-controlled-locale",
        "LD_LIBRARY_PATH": "/secret-controlled/libraries",
        "NVIDIA_DRIVER_CAPABILITIES": "secret-controlled-capabilities",
        "NVIDIA_VISIBLE_DEVICES": "secret-controlled-device",
        "PATH": "/secret-controlled/bin",
        "PYTHONPATH": "/secret-controlled/python",
        "SSL_CERT_FILE": "/secret-controlled/ca.pem",
        "TMPDIR": "/secret-controlled/tmp",
    }
    secrets = {
        "DISCOVERY_API_KEY": "provider-secret",
        "DISCOVERY_API_BASE": "https://api.openai.com/v1",
        "DISCOVERY_MODEL": "gpt-5.6-sol",
        "MODAL_IDENTITY_TOKEN": "modal-runtime-identity",
        "MODAL_TOKEN_SECRET": "modal-secret",
        "TINKER_API_KEY": "tinker-secret",
        "HF_TOKEN": "hugging-face-secret",
        "GITHUB_TOKEN": "github-secret",
        "UNRELATED_SECRET": "unrelated-secret",
    }
    for name, value in {**operational_collisions, **secrets}.items():
        monkeypatch.setenv(name, value)

    provider_free = modal_app._remote_environment(
        context,
        provider=False,
        requested_device="cuda",
    )
    assert set(provider_free).isdisjoint(secrets)
    assert provider_free["PATH"] == modal_app._STATIC_SUBPROCESS_PATH
    assert provider_free["PYTHONPATH"] == str(modal_app.REMOTE_PROJECT_ROOT)
    assert provider_free["TMPDIR"] == "/tmp"
    assert provider_free["LANG"] == "C.UTF-8"
    assert provider_free["LC_ALL"] == "C.UTF-8"
    assert provider_free["DISCOVERY_TRAIN_DEVICE"] == "cuda"
    assert provider_free["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert provider_free["CUDA_VISIBLE_DEVICES"] == "0"
    assert "NVIDIA_VISIBLE_DEVICES" not in provider_free
    assert "LD_LIBRARY_PATH" not in provider_free
    assert "SSL_CERT_FILE" not in provider_free
    assert set(operational_collisions.values()).isdisjoint(provider_free.values())
    assert "provider-secret" not in "\n".join(provider_free.values())

    provider = modal_app._remote_environment(
        context,
        provider=True,
        requested_device="cuda",
    )
    assert {name: provider[name] for name in modal_app.PROVIDER_ENVIRONMENT_KEYS} == {
        name: secrets[name] for name in modal_app.PROVIDER_ENVIRONMENT_KEYS
    }
    assert set(provider).intersection(secrets) == set(
        modal_app.PROVIDER_ENVIRONMENT_KEYS
    )
    assert set(provider).isdisjoint(
        {
            "MODAL_IDENTITY_TOKEN",
            "MODAL_TOKEN_SECRET",
            "TINKER_API_KEY",
            "HF_TOKEN",
            "GITHUB_TOKEN",
        }
    )
    assert set(operational_collisions.values()).isdisjoint(provider.values())

    monkeypatch.delenv("DISCOVERY_API_KEY")
    with pytest.raises(modal_app.RemoteActionError, match="missing required keys"):
        modal_app._remote_environment(
            context,
            provider=True,
            requested_device="cuda",
        )

    cpu = modal_app._remote_environment(
        context,
        provider=False,
        requested_device="cpu",
    )
    assert cpu["DISCOVERY_TRAIN_DEVICE"] == "cpu"
    assert "CUBLAS_WORKSPACE_CONFIG" not in cpu
    assert "CUDA_DEVICE_ORDER" not in cpu
    assert "CUDA_VISIBLE_DEVICES" not in cpu


@pytest.mark.parametrize(
    ("api_base", "model"),
    [
        ("https://example.com/v1", "gpt-5.6-sol"),
        ("https://api.openai.com/v1/", "gpt-5.6-sol"),
        ("https://api.openai.com/v1?tenant=alternate", "gpt-5.6-sol"),
        ("https://user@api.openai.com/v1", "gpt-5.6-sol"),
        ("https://api.openai.com/v1", "gpt-5.6-terra"),
        ("https://api.openai.com/v1", " gpt-5.6-sol"),
        ("https://api.openai.com/v1", "gpt-5.6-sol "),
    ],
)
def test_modal_provider_environment_requires_exact_official_contract(
    monkeypatch,
    api_base: str,
    model: str,
) -> None:
    import modal_app

    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="modal-provider-contract-1",
        app_name="rl4rl-architecture-discovery",
        function_name="canary_greedy_autoresearch",
        modal_app_id=None,
        modal_function_id=None,
        modal_call_id="fc-provider-contract",
        modal_image_id=None,
        image_source_sha256="a" * 64,
        artifact_uri=(
            "volume://rl4rl-architecture-artifacts/runs/"
            "modal-provider-contract-1"
        ),
    )
    secret = "provider-secret-never-report"
    monkeypatch.setenv("DISCOVERY_API_KEY", secret)
    monkeypatch.setenv("DISCOVERY_API_BASE", api_base)
    monkeypatch.setenv("DISCOVERY_MODEL", model)

    with pytest.raises(modal_app.RemoteActionError) as error:
        modal_app._remote_environment(
            context,
            provider=True,
            requested_device="cuda",
        )

    message = str(error.value)
    assert message == (
        "provider Secret does not match the pinned Modal provider contract"
    )
    assert secret not in message
    assert api_base not in message
    assert model not in message

    provider_free = modal_app._remote_environment(
        context,
        provider=False,
        requested_device="cuda",
    )
    assert set(modal_app.PROVIDER_ENVIRONMENT_KEYS).isdisjoint(provider_free)


def test_provider_secret_is_attached_lazily_in_source() -> None:
    source = (ROOT / "modal_app.py").read_text(encoding="utf-8")
    assert 'options["secrets"]' not in source
    assert "function.with_options(secrets=[PROVIDER_SECRET])" in source


def test_image_dependency_build_has_one_bounded_cpu_only_function() -> None:
    tree = ast.parse((ROOT / "modal_app.py").read_text(encoding="utf-8"))
    build_image = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_build_image"
    )
    run_functions = [
        node
        for node in ast.walk(build_image)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_function"
    ]
    assert len(run_functions) == 1
    call = run_functions[0]
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "install_image_dependencies"
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert isinstance(keywords["cpu"], ast.Name)
    assert keywords["cpu"].id == "IMAGE_BUILD_CPU_REQUEST_CORES"
    assert isinstance(keywords["memory"], ast.Name)
    assert keywords["memory"].id == "IMAGE_BUILD_MEMORY_REQUEST_MIB"
    assert isinstance(keywords["timeout"], ast.Name)
    assert keywords["timeout"].id == "IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS"
    assert ast.literal_eval(keywords["include_source"]) is False
    assert isinstance(keywords["env"], ast.Dict)
    assert ast.literal_eval(keywords["env"].keys[0]) == "PYTHONPATH"
    assert isinstance(keywords["kwargs"], ast.Dict)
    keyword_argument_names = {
        ast.literal_eval(key) for key in keywords["kwargs"].keys
    }
    assert keyword_argument_names == {
        "project_root",
        "thread_limit",
        "uv_version",
        "timeout_seconds",
    }
    assert "gpu" not in keywords
    assert "region" not in keywords
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "run_commands"
        for node in ast.walk(build_image)
    )


def test_two_manifest_copy_layers_surround_build_function(monkeypatch) -> None:
    import modal_app

    operations: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class FakeImage:
        def add_local_file(self, *args, **kwargs):
            operations.append(("add_local_file", args, kwargs))
            return self

        def add_local_dir(self, *args, **kwargs):
            operations.append(("add_local_dir", args, kwargs))
            return self

        def run_function(self, *args, **kwargs):
            operations.append(("run_function", args, kwargs))
            return self

        def env(self, *args, **kwargs):
            operations.append(("env", args, kwargs))
            return self

    fake_image = FakeImage()
    fake_modal = SimpleNamespace(
        Image=SimpleNamespace(
            debian_slim=lambda **_kwargs: fake_image,
        )
    )
    monkeypatch.setattr(modal_app, "modal", fake_modal)
    monkeypatch.setattr(modal_app, "_IMAGE_SOURCE_STAGING", None)

    assert modal_app._build_image() is fake_image

    assert [operation[0] for operation in operations] == [
        "add_local_dir",
        "run_function",
        "add_local_dir",
        "env",
    ]
    assert not any(operation[0] == "add_local_file" for operation in operations)

    dependency_copy, build, runtime_copy, environment = operations
    staged_root = Path(dependency_copy[1][0])
    assert staged_root.name == "source"
    assert dependency_copy[1][1] == "/opt/architecture_discovery"
    assert dependency_copy[2]["copy"] is True
    dependency_ignore = dependency_copy[2]["ignore"]
    assert callable(dependency_ignore)

    dependency_paths = modal_app._dependency_image_source_paths()
    manifest_paths = {
        item.relative_path for item in modal_app.IMAGE_SOURCE_MANIFEST.files
    }
    assert dependency_paths < manifest_paths
    for relative in manifest_paths:
        assert dependency_ignore(staged_root / relative) is (
            relative not in dependency_paths
        )
    assert dependency_ignore(staged_root) is False
    assert dependency_ignore(staged_root / "vendor") is False
    assert dependency_ignore(staged_root / "vendor" / "openevolve") is False
    assert dependency_ignore(staged_root / "common") is True
    assert dependency_ignore(staged_root / ".env") is True

    assert build[1][0].__module__ == "modal_image_build"
    assert build[2]["include_source"] is False
    assert build[2]["cpu"] == modal_app.IMAGE_BUILD_CPU_REQUEST_CORES == 2.0
    assert build[2]["memory"] == modal_app.IMAGE_BUILD_MEMORY_REQUEST_MIB == 8192
    assert not isinstance(build[2]["cpu"], tuple)
    assert not isinstance(build[2]["memory"], tuple)
    assert build[2]["env"] == {"PYTHONPATH": "/opt/architecture_discovery"}
    assert build[2]["kwargs"]["thread_limit"] == 2
    assert runtime_copy[1] == dependency_copy[1]
    assert runtime_copy[2] == {"copy": True}
    staged_paths = {
        path.relative_to(staged_root).as_posix()
        for path in staged_root.rglob("*")
        if path.is_file()
    }
    assert staged_paths == manifest_paths

    final_environment = environment[1][0]
    assert final_environment[modal_app.IMAGE_SOURCE_IDENTITY_ENV] == (
        modal_app.IMAGE_SOURCE_SHA256
    )


def test_runtime_function_options_are_single_use_and_secret_free() -> None:
    import modal_app

    for name, spec in modal_app.FUNCTION_SPECS.items():
        options = modal_app._function_options(name)
        assert options["single_use_containers"] is True
        assert options["block_network"] is (not spec.provider_secret)
        assert options["cpu"] == (2.0, 2.0)
        assert options["memory"] == (8192, 8192)
        assert options["cpu"] == (
            spec.cpu_request_cores,
            spec.cpu_soft_limit_cores,
        )
        assert options["memory"] == (
            spec.memory_request_mib,
            spec.memory_limit_mib,
        )
        assert options["min_containers"] == 0
        assert options["max_containers"] == 1
        assert options["retries"] == 0
        assert options["timeout"] == 300
        assert "region" not in options
        assert options["include_source"] is False
        assert options["name"] == spec.name
        assert options["image"] is modal_app.IMAGE
        assert options["volumes"] == {
            str(modal_app.VOLUME_MOUNT_PATH): modal_app.ARTIFACT_VOLUME
        }
        if spec.gpu is None:
            assert "gpu" not in options
        else:
            assert options["gpu"] == spec.gpu
        assert "secrets" not in options
        assert "restrict_modal_access" not in options


def test_provider_variant_attaches_only_the_named_secret(monkeypatch) -> None:
    import modal_app

    secret = object()
    calls = []

    class FakeFunction:
        def with_options(self, **kwargs):
            calls.append(kwargs)
            return "bound-provider-variant"

    monkeypatch.setattr(modal_app, "PROVIDER_SECRET", secret)
    result = modal_app._provider_variant(FakeFunction())

    assert result == "bound-provider-variant"
    assert calls == [{"secrets": [secret]}]


def test_local_entrypoint_exposes_distinct_provider_consent_and_recovery() -> None:
    tree = ast.parse((ROOT / "modal_app.py").read_text(encoding="utf-8"))
    entrypoint = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )
    argument_names = {argument.arg for argument in entrypoint.args.args}
    assert {
        "approved",
        "provider_approved",
        "harness",
        "expected_image_source_sha256",
        "source_tree_sha256",
        "cohort_id",
    } <= argument_names
    called_names = {
        node.func.id
        for node in ast.walk(entrypoint)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_require_local_action_approvals" in called_names
    assert "_require_durable_local_action_intent" in called_names
    assert "_run_single_canary_synchronously" in called_names
    durable_call = next(
        node
        for node in ast.walk(entrypoint)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_require_durable_local_action_intent"
    )
    remote_calls = [
        node
        for node in ast.walk(entrypoint)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"invoke_synchronously", "run_canaries_synchronously"}
    ]
    assert remote_calls
    assert all(durable_call.lineno < node.lineno for node in remote_calls)


def test_local_entrypoint_intent_wrapper_fails_closed_on_corruption(
    monkeypatch,
) -> None:
    import modal_app

    monkeypatch.setattr(
        modal_app,
        "validate_local_action_intent_for_entrypoint",
        lambda **_kwargs: (_ for _ in ()).throw(
            ValueError("corrupt durable intent")
        ),
    )
    with pytest.raises(SystemExit, match="before any remote invocation"):
        modal_app._require_durable_local_action_intent(
            identity=modal_app.ModalLiveCohortIdentity(
                source_tree_sha256="a" * 64,
                image_source_sha256="b" * 64,
                cohort_id="cohort-1",
            ),
            attempt_id="c" * 32,
            action="offline-smoke",
            run_id="offline-run-1",
            source_run_id=None,
            verifier_run_id=None,
            harness=None,
            environment={},
        )


def test_execution_context_records_available_modal_object_ids(monkeypatch) -> None:
    import modal_app

    monkeypatch.setattr(
        modal_app,
        "modal",
        SimpleNamespace(current_function_call_id=lambda: "fc-call123"),
    )
    monkeypatch.setattr(modal_app, "app", SimpleNamespace(app_id="ap-app123"))
    monkeypatch.setattr(
        modal_app,
        "candidate_smoke",
        SimpleNamespace(is_hydrated=True, object_id="fu-function123"),
        raising=False,
    )
    monkeypatch.setattr(
        modal_app,
        "IMAGE",
        SimpleNamespace(is_hydrated=True, object_id="im-fallback123"),
    )
    monkeypatch.setenv("MODAL_IMAGE_ID", "im-environment123")

    context = modal_app._execution_context("candidate_smoke", "modal-ids-1")

    assert context.modal_app_id == "ap-app123"
    assert context.modal_function_id == "fu-function123"
    assert context.modal_call_id == "fc-call123"
    assert context.modal_image_id == "im-environment123"

    monkeypatch.delenv("MODAL_IMAGE_ID")
    fallback = modal_app._execution_context("candidate_smoke", "modal-ids-2")
    assert fallback.modal_image_id == "im-fallback123"


def test_execution_context_prefers_executing_bound_function_id(monkeypatch) -> None:
    import modal_app

    monkeypatch.setattr(
        modal_app,
        "modal",
        SimpleNamespace(current_function_call_id=lambda: "fc-call123"),
    )
    monkeypatch.setattr(modal_app, "app", SimpleNamespace(app_id="ap-app123"))
    monkeypatch.setattr(
        modal_app,
        "_current_modal_function_id",
        lambda _function_name: "fu-boundvariant123",
    )
    monkeypatch.setenv("MODAL_IMAGE_ID", "im-image123")

    context = modal_app._execution_context(
        "canary_greedy_autoresearch",
        "modal-bound-variant-1",
    )

    assert context.modal_function_id == "fu-boundvariant123"


def test_current_function_id_reads_the_pinned_runtime_context(monkeypatch) -> None:
    import modal_app
    from modal._runtime.task_lifecycle_manager import _TaskLifecycleManager

    monkeypatch.setattr(
        _TaskLifecycleManager,
        "_singleton",
        SimpleNamespace(function_id="fu-executing123"),
    )

    assert (
        modal_app._current_modal_function_id("canary_greedy_autoresearch")
        == "fu-executing123"
    )


def test_artifact_verifier_requires_exactly_one_manifest(tmp_path) -> None:
    import modal_app
    from modal_boundary import ModalBoundaryError

    run = tmp_path / "modal-run-1"
    run.mkdir()
    checkpoint = run / "artifact_manifest.checkpoint.json"
    checkpoint.write_text("{}\n", encoding="utf-8")
    assert modal_app._select_artifact_manifest(run) == checkpoint

    final = run / "artifact_manifest.json"
    final.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ModalBoundaryError, match="exactly one"):
        modal_app._select_artifact_manifest(run)


def test_artifact_verifier_run_id_is_reserved_once_under_race(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import modal_app
    from modal_boundary import (
        ArtifactVerificationV1,
        ModalBoundaryError,
        build_artifact_manifest,
        load_artifact_manifest,
        verify_artifact_manifest,
        volume_artifact_uri,
        write_artifact_manifest,
    )

    volume_root = tmp_path / "volume"
    source_run_id = "artifact-source-race-1"
    verifier_run_id = "artifact-verifier-race-1"
    source_directory = volume_root / "runs" / source_run_id
    source_directory.mkdir(parents=True)
    (source_directory / "result.json").write_text(
        '{"ok": true}\n',
        encoding="utf-8",
    )
    source_context = ExecutionContextV1(
        execution_backend="modal",
        run_id=source_run_id,
        app_name=modal_app.APP_NAME,
        function_name="candidate_smoke",
        modal_app_id="ap-source123",
        modal_function_id="fu-source123",
        modal_call_id="fc-source123",
        modal_image_id="im-verifier123",
        image_source_sha256=modal_app.IMAGE_SOURCE_SHA256,
        artifact_uri=volume_artifact_uri(source_run_id),
    )
    (source_directory / "execution_context.json").write_text(
        json.dumps(source_context.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_manifest = build_artifact_manifest(
        source_directory,
        run_id=source_run_id,
        image_source_sha256=modal_app.IMAGE_SOURCE_SHA256,
    )
    write_artifact_manifest(source_directory, source_manifest)

    class FakeVolume:
        reloads = 0
        commits = 0

        def reload(self) -> None:
            self.reloads += 1

        def commit(self) -> None:
            self.commits += 1

    volume = FakeVolume()
    original_create = modal_app.create_fresh_run_directory
    barrier = __import__("threading").Barrier(2)

    def create_after_barrier(mount_root, run_id, **kwargs):
        barrier.wait(timeout=5)
        return original_create(mount_root, run_id, **kwargs)

    def execution_context(
        function_name: str,
        run_id: str,
        *,
        artifact_run_id: str | None = None,
    ) -> ExecutionContextV1:
        assert function_name == "artifact_verify"
        return ExecutionContextV1(
            execution_backend="modal",
            run_id=run_id,
            app_name=modal_app.APP_NAME,
            function_name=function_name,
            modal_app_id="ap-verifier123",
            modal_function_id="fu-verifier123",
            modal_call_id="fc-verifier123",
            modal_image_id="im-verifier123",
            image_source_sha256=modal_app.IMAGE_SOURCE_SHA256,
            artifact_uri=volume_artifact_uri(artifact_run_id or run_id),
        )

    monkeypatch.setattr(modal_app, "VOLUME_MOUNT_PATH", volume_root)
    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", volume)
    monkeypatch.setenv(
        modal_app.IMAGE_SOURCE_IDENTITY_ENV,
        modal_app.IMAGE_SOURCE_SHA256,
    )
    monkeypatch.setattr(modal_app, "create_fresh_run_directory", create_after_barrier)

    original_resolve = modal_app.resolve_existing_volume_run_directory
    resolved_sources: list[tuple[Path, str, bool]] = []

    def volume_path_after_lease(
        mount_root: Path,
        run_id: str,
        *,
        allow_mount_root_symlink: bool = False,
    ) -> Path:
        assert volume.commits >= 1
        resolved_sources.append(
            (mount_root, run_id, allow_mount_root_symlink)
        )
        return original_resolve(
            mount_root,
            run_id,
            allow_mount_root_symlink=allow_mount_root_symlink,
        )

    monkeypatch.setattr(
        modal_app,
        "resolve_existing_volume_run_directory",
        volume_path_after_lease,
    )
    monkeypatch.setattr(modal_app, "_execution_context", execution_context)

    def verify_once():
        try:
            return modal_app._artifact_verify_action(
                source_run_id,
                verifier_run_id,
            )
        except BaseException as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _index: verify_once(), range(2)))

    results = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    assert len(results) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ModalBoundaryError)
    assert "already exists" in str(failures[0])
    assert volume.reloads == 2
    assert volume.commits == 2

    result = ArtifactVerificationV1.from_dict(results[0])
    verifier_directory = volume_root / "runs" / verifier_run_id
    assert json.loads(
        (verifier_directory / "artifact_verification_result.json").read_text(
            encoding="utf-8"
        )
    ) == result.to_dict()
    verifier_manifest = load_artifact_manifest(
        verifier_directory / "artifact_manifest.json"
    )
    assert verify_artifact_manifest(verifier_directory, verifier_manifest) == {
        "run_id": verifier_run_id,
        "file_count": 3,
        "manifest_sha256": verifier_manifest.manifest_sha256,
        "verified": True,
    }
    assert result.source_run_id == source_run_id
    assert result.verifier_run_id == verifier_run_id
    assert all(
        item == (volume_root, source_run_id, True)
        for item in resolved_sources
    )

    monkeypatch.setattr(
        modal_app,
        "create_fresh_run_directory",
        original_create,
    )
    with pytest.raises(ModalBoundaryError, match="already exists"):
        modal_app._artifact_verify_action(source_run_id, verifier_run_id)
    assert volume.commits == 2

    failed_source_run_id = "artifact-source-invalid-1"
    failed_verifier_run_id = "artifact-verifier-failed-1"
    failed_source = volume_root / "runs" / failed_source_run_id
    failed_source.mkdir()
    (failed_source / "artifact_manifest.json").write_text(
        '{}\n',
        encoding="utf-8",
    )
    with pytest.raises(ModalBoundaryError):
        modal_app._artifact_verify_action(
            failed_source_run_id,
            failed_verifier_run_id,
        )
    assert volume.commits == 4
    failed_verifier = volume_root / "runs" / failed_verifier_run_id
    assert not (failed_verifier / "artifact_verification_result.json").exists()
    failure = json.loads(
        (failed_verifier / "artifact_verification_failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["source_run_id"] == failed_source_run_id
    assert failure["verifier_run_id"] == failed_verifier_run_id
    assert failure["message"] == "artifact verification failed; details suppressed"
    failure_manifest = load_artifact_manifest(
        failed_verifier / "artifact_manifest.json"
    )
    assert verify_artifact_manifest(failed_verifier, failure_manifest)[
        "verified"
    ] is True

    with pytest.raises(ModalBoundaryError, match="already exists"):
        modal_app._artifact_verify_action(
            failed_source_run_id,
            failed_verifier_run_id,
        )
    assert volume.commits == 4

    context_failure_verifier_id = "artifact-verifier-context-failed-1"

    def fail_context(*_args, **_kwargs):
        raise RuntimeError("sensitive context failure")

    monkeypatch.setattr(modal_app, "_execution_context", fail_context)
    with pytest.raises(RuntimeError, match="sensitive context failure"):
        modal_app._artifact_verify_action(
            failed_source_run_id,
            context_failure_verifier_id,
        )
    assert volume.commits == 4
    context_failure_directory = (
        volume_root / "runs" / context_failure_verifier_id
    )
    assert not context_failure_directory.exists()

    with pytest.raises(RuntimeError, match="sensitive context failure"):
        modal_app._artifact_verify_action(
            failed_source_run_id,
            context_failure_verifier_id,
        )
    assert volume.commits == 4


def test_download_persists_its_single_remote_verification_capture(
    tmp_path: Path,
) -> None:
    import modal_app
    from modal_boundary import (
        APP_NAME,
        ArtifactVerificationV1,
        RawArtifactManifestV1,
        build_artifact_manifest,
        verify_artifact_manifest,
        volume_artifact_uri,
    )

    run_id = "download-capture-1"
    identity = _cohort_identity("a" * 64)
    attempt_id = _ATTEMPT_ID
    run = tmp_path / run_id
    run.mkdir()
    (run / "result.json").write_text('{"ok": true}\n', encoding="utf-8")
    manifest = build_artifact_manifest(
        run,
        run_id=run_id,
        image_source_sha256="a" * 64,
    )
    raw_manifest = RawArtifactManifestV1.from_bytes(
        filename="artifact_manifest.json",
        raw_bytes=(
            json.dumps(manifest.to_dict(), indent=3, sort_keys=False) + "\n"
        ).encode("utf-8"),
    )
    local_verification = verify_artifact_manifest(run, manifest)
    verifier_run_id = "download-verifier-1"
    verifier_context = ExecutionContextV1(
        execution_backend="modal",
        run_id=verifier_run_id,
        app_name=APP_NAME,
        function_name="artifact_verify",
        modal_app_id="ap-verifier123",
        modal_function_id="fu-verifier123",
        modal_call_id="fc-verifier123",
        modal_image_id="im-verifier123",
        image_source_sha256="a" * 64,
        artifact_uri=volume_artifact_uri(run_id),
    )
    remote = ArtifactVerificationV1(
        source_run_id=run_id,
        verifier_run_id=verifier_run_id,
        manifest_filename=raw_manifest.filename,
        raw_manifest_sha256=raw_manifest.raw_sha256,
        raw_manifest_size_bytes=raw_manifest.raw_size_bytes,
        canonical_manifest_sha256=manifest.manifest_sha256,
        file_count=local_verification["file_count"],
        verifier_execution_context=verifier_context,
    ).to_dict()

    mixed_source = dict(remote)
    mixed_source["verifier_execution_context"] = dict(
        remote["verifier_execution_context"]
    )
    mixed_source["verifier_execution_context"]["image_source_sha256"] = "b" * 64
    with pytest.raises(modal_app.ModalBoundaryError, match="image source differs"):
        modal_app._persist_download_verification(
            source_run_id=run_id,
            verifier_run_id=verifier_run_id,
            remote_verification=mixed_source,
            raw_manifest=raw_manifest,
            identity=identity,
            attempt_id=attempt_id,
            project_root=tmp_path,
        )

    source_context = ExecutionContextV1(
        execution_backend="modal",
        run_id=run_id,
        app_name=APP_NAME,
        function_name="candidate_smoke",
        modal_app_id="ap-source123",
        modal_function_id="fu-source123",
        modal_call_id="fc-source123",
        modal_image_id="im-other123",
        image_source_sha256="a" * 64,
        artifact_uri=volume_artifact_uri(run_id),
    )
    with pytest.raises(modal_app.ModalBoundaryError, match="image ID differs"):
        modal_app._validated_artifact_verification(
            source_run_id=run_id,
            verifier_run_id=verifier_run_id,
            remote_verification=remote,
            raw_manifest=raw_manifest,
            source_execution_context=source_context,
        )

    logical, persisted = modal_app._persist_download_verification(
        source_run_id=run_id,
        verifier_run_id=verifier_run_id,
        remote_verification=remote,
        raw_manifest=raw_manifest,
        identity=identity,
        attempt_id=attempt_id,
        project_root=tmp_path,
    )

    assert logical == modal_app.modal_remote_verification_receipt_path(
        identity,
        run_id,
        verifier_run_id,
        attempt_id,
    ).as_posix()
    assert json.loads(persisted.read_text()) == remote
    with pytest.raises(FileExistsError):
        modal_app._persist_download_verification(
            source_run_id=run_id,
            verifier_run_id=verifier_run_id,
            remote_verification=remote,
            raw_manifest=raw_manifest,
            identity=identity,
            attempt_id=attempt_id,
            project_root=tmp_path,
        )

    recovery_verifier_run_id = "download-verifier-2"
    recovery_remote = dict(remote)
    recovery_context = dict(remote["verifier_execution_context"])
    recovery_context["run_id"] = recovery_verifier_run_id
    recovery_context["modal_call_id"] = "fc-verifier456"
    recovery_remote["verifier_run_id"] = recovery_verifier_run_id
    recovery_remote["verifier_execution_context"] = recovery_context
    recovery_logical, recovery_persisted = (
        modal_app._persist_download_verification(
            source_run_id=run_id,
            verifier_run_id=recovery_verifier_run_id,
            remote_verification=recovery_remote,
            raw_manifest=raw_manifest,
            identity=identity,
            attempt_id=attempt_id,
            project_root=tmp_path,
        )
    )
    assert recovery_logical == modal_app.modal_remote_verification_receipt_path(
        identity,
        run_id,
        recovery_verifier_run_id,
        attempt_id,
    ).as_posix()
    assert persisted.read_bytes() != b""
    assert json.loads(recovery_persisted.read_text()) == recovery_remote

    source = (ROOT / "modal_app.py").read_text(encoding="utf-8")
    download_branch = source.split('elif action == "download":', 1)[1].split(
        "        else:", 1
    )[0]
    assert download_branch.count("invoke_synchronously(") == 1
    assert download_branch.count("_capture_artifact_verifier_directory(") == 2
    assert download_branch.rindex("_capture_artifact_verifier_directory(") < (
        download_branch.index("_download_after_verification_receipt(")
    )
    assert "with suppress(Exception):" in download_branch
    assert "except BaseException:" not in download_branch

    helper_source = inspect.getsource(modal_app._download_after_verification_receipt)
    assert helper_source.index("_validated_artifact_verification(") < (
        helper_source.index("_persist_download_verification(")
    )
    assert helper_source.index("_persist_download_verification(") < (
        helper_source.index("download_artifacts(")
    )


def test_download_receipt_failure_prevents_source_directory_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import modal_app
    identity = _cohort_identity(modal_app.IMAGE_SOURCE_SHA256)

    events: list[str] = []
    raw_manifest = object()

    monkeypatch.setattr(
        modal_app,
        "_load_volume_manifest",
        lambda _run_id: events.append("load-manifest") or raw_manifest,
    )
    monkeypatch.setattr(
        modal_app,
        "_validated_artifact_verification",
        lambda **_kwargs: events.append("validate-verifier"),
    )

    def fail_receipt(**_kwargs):
        events.append("persist-receipt")
        raise OSError("injected receipt persistence failure")

    def forbidden_download(*_args, **_kwargs):
        events.append("publish-source")
        raise AssertionError("source download must not start")

    monkeypatch.setattr(modal_app, "_persist_download_verification", fail_receipt)
    monkeypatch.setattr(modal_app, "download_artifacts", forbidden_download)
    local_root = tmp_path / "downloads"

    with pytest.raises(OSError, match="receipt persistence failure"):
        modal_app._download_after_verification_receipt(
            source_run_id="receipt-source-1",
            verifier_run_id="receipt-verifier-1",
            remote_verification={"validated": True},
            identity=identity,
            attempt_id=_ATTEMPT_ID,
            local_root=local_root,
            project_root=tmp_path,
        )

    assert events == ["load-manifest", "validate-verifier", "persist-receipt"]
    assert not (local_root / "receipt-source-1").exists()


@pytest.mark.parametrize(
    "attack",
    ("ancestor_swap", "leaf_swap", "preexisting_symlink"),
)
def test_remote_verification_receipt_rejects_path_swaps_and_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    import modal_app

    source_run_id = "swap-source-1"
    verifier_run_id = "swap-verifier-1"
    identity = _cohort_identity(modal_app.IMAGE_SOURCE_SHA256)
    payload = {
        "source_run_id": source_run_id,
        "verifier_run_id": verifier_run_id,
        "verified": True,
    }
    verification = SimpleNamespace(
        source_run_id=source_run_id,
        verifier_run_id=verifier_run_id,
        to_dict=lambda: payload,
    )
    monkeypatch.setattr(
        modal_app,
        "_validated_artifact_verification",
        lambda **_kwargs: verification,
    )
    logical = modal_app.modal_remote_verification_receipt_path(
        identity,
        source_run_id,
        verifier_run_id,
        _ATTEMPT_ID,
    )
    destination = tmp_path.joinpath(*logical.parts)
    outside = tmp_path / "outside"

    if attack == "preexisting_symlink":
        destination.parent.mkdir(parents=True)
        outside.write_text("must remain unchanged\n", encoding="utf-8")
        destination.symlink_to(outside)
        expected_error = FileExistsError
    else:
        real_create = modal_app.create_json_exclusive

        def swapping_create(path, value):
            real_create(path, value)
            selected = Path(path)
            if attack == "ancestor_swap":
                original = selected.parent.with_name(
                    selected.parent.name + "-original"
                )
                selected.parent.rename(original)
                outside.mkdir()
                selected.parent.symlink_to(outside, target_is_directory=True)
            else:
                original = selected.with_name(selected.name + ".original")
                selected.rename(original)
                outside.write_text("must remain unchanged\n", encoding="utf-8")
                selected.symlink_to(outside)

        monkeypatch.setattr(modal_app, "create_json_exclusive", swapping_create)
        expected_error = ValueError

    with pytest.raises(expected_error):
        modal_app._persist_download_verification(
            source_run_id=source_run_id,
            verifier_run_id=verifier_run_id,
            remote_verification=payload,
            raw_manifest=object(),
            identity=identity,
            attempt_id=_ATTEMPT_ID,
            project_root=tmp_path,
        )
    if outside.is_file():
        assert outside.read_text(encoding="utf-8") == "must remain unchanged\n"
    else:
        assert not list(outside.iterdir())


def test_download_receipt_exists_before_source_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import modal_app
    identity = _cohort_identity(modal_app.IMAGE_SOURCE_SHA256)

    events: list[str] = []
    raw_manifest = object()
    receipt = tmp_path / "readiness" / "remote_verification.json"
    destination = tmp_path / "downloads" / "receipt-source-2"
    source_context = ExecutionContextV1(
        execution_backend="modal",
        run_id="receipt-source-2",
        app_name=modal_app.APP_NAME,
        function_name="candidate_smoke",
        modal_app_id="ap-source123",
        modal_function_id="fu-source123",
        modal_call_id="fc-source123",
        modal_image_id="im-source123",
        image_source_sha256=modal_app.IMAGE_SOURCE_SHA256,
        artifact_uri=modal_app.volume_artifact_uri("receipt-source-2"),
    )

    monkeypatch.setattr(
        modal_app,
        "_load_volume_manifest",
        lambda _run_id: events.append("load-manifest") or raw_manifest,
    )

    def validate(**kwargs):
        events.append(
            "validate-source"
            if kwargs.get("source_execution_context") is not None
            else "validate-verifier"
        )

    def persist(**_kwargs):
        events.append("persist-receipt")
        receipt.parent.mkdir(parents=True)
        receipt.write_text("{}\n", encoding="utf-8")
        return "readiness/remote_verification.json", receipt

    def publish(*_args, **_kwargs):
        events.append("publish-source")
        assert receipt.is_file()
        destination.mkdir(parents=True)
        return destination

    monkeypatch.setattr(modal_app, "_validated_artifact_verification", validate)
    monkeypatch.setattr(modal_app, "_persist_download_verification", persist)
    monkeypatch.setattr(modal_app, "download_artifacts", publish)
    monkeypatch.setattr(
        modal_app,
        "ARTIFACT_VOLUME",
        SimpleNamespace(read_file=lambda _path: b""),
    )
    monkeypatch.setattr(
        modal_app,
        "_read_bounded_json_object",
        lambda *_args, **_kwargs: source_context.to_dict(),
    )

    observed_destination, logical = modal_app._download_after_verification_receipt(
        source_run_id="receipt-source-2",
        verifier_run_id="receipt-verifier-2",
        remote_verification={"validated": True},
        identity=identity,
        attempt_id=_ATTEMPT_ID,
        local_root=tmp_path / "downloads",
        project_root=tmp_path,
    )

    assert observed_destination == destination
    assert logical == "readiness/remote_verification.json"
    assert events == [
        "load-manifest",
        "validate-verifier",
        "persist-receipt",
        "publish-source",
        "validate-source",
    ]


def test_verifier_volume_capture_is_create_only_and_validates_both_outcomes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import modal_app
    from modal_boundary import (
        ArtifactVerificationV1,
        build_artifact_manifest,
        volume_artifact_uri,
        write_artifact_manifest,
    )

    volume_root = tmp_path / "volume"
    source_run_id = "capture-source-1"
    identity = _cohort_identity(modal_app.IMAGE_SOURCE_SHA256)
    attempt_id = _ATTEMPT_ID

    class FakeVolume:
        def read_file(self, logical: str) -> bytes:
            return (volume_root / logical.removeprefix("/")).read_bytes()

    def context(verifier_run_id: str) -> ExecutionContextV1:
        return ExecutionContextV1(
            execution_backend="modal",
            run_id=verifier_run_id,
            app_name=modal_app.APP_NAME,
            function_name="artifact_verify",
            modal_app_id="ap-capture123",
            modal_function_id="fu-capture123",
            modal_call_id=f"fc-{verifier_run_id}",
            modal_image_id="im-capture123",
            image_source_sha256=modal_app.IMAGE_SOURCE_SHA256,
            artifact_uri=volume_artifact_uri(source_run_id),
        )

    def write_verifier(
        verifier_run_id: str,
        *,
        successful: bool,
    ) -> dict[str, object]:
        directory = volume_root / "runs" / verifier_run_id
        directory.mkdir(parents=True)
        verifier_context = context(verifier_run_id)
        (directory / "execution_context.json").write_text(
            json.dumps(verifier_context.to_dict(), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (directory / "image_source_manifest.json").write_text(
            json.dumps(
                modal_app.IMAGE_SOURCE_MANIFEST.to_dict(),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if successful:
            payload = ArtifactVerificationV1(
                source_run_id=source_run_id,
                verifier_run_id=verifier_run_id,
                manifest_filename="artifact_manifest.json",
                raw_manifest_sha256="a" * 64,
                raw_manifest_size_bytes=128,
                canonical_manifest_sha256="b" * 64,
                file_count=3,
                verifier_execution_context=verifier_context,
            ).to_dict()
            filename = "artifact_verification_result.json"
        else:
            payload = {
                "schema_name": "ModalArtifactVerificationFailure",
                "schema_version": "1.0",
                "source_run_id": source_run_id,
                "verifier_run_id": verifier_run_id,
                "error_type": "RuntimeError",
                "message": "artifact verification failed; details suppressed",
                "verifier_execution_context": verifier_context.to_dict(),
            }
            filename = "artifact_verification_failure.json"
        (directory / filename).write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = build_artifact_manifest(
            directory,
            run_id=verifier_run_id,
            image_source_sha256=modal_app.IMAGE_SOURCE_SHA256,
        )
        write_artifact_manifest(directory, manifest)
        return payload

    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", FakeVolume())

    success_id = "capture-verifier-success-1"
    successful_payload = write_verifier(success_id, successful=True)
    success = modal_app._capture_artifact_verifier_directory(
        source_run_id=source_run_id,
        verifier_run_id=success_id,
        expected_remote_verification=successful_payload,
        identity=identity,
        attempt_id=attempt_id,
        project_root=tmp_path,
    )
    assert success["remote_verifier_outcome"] == "success"
    assert success["validated"] is True
    success_capture = tmp_path.joinpath(
        *modal_app.modal_artifact_verifier_capture_directory_path(
            identity,
            source_run_id,
            success_id,
            attempt_id,
        ).parts
    )
    assert {item.name for item in success_capture.iterdir()} == {
        "artifact_manifest.json",
        "artifact_verification_result.json",
        "execution_context.json",
        "image_source_manifest.json",
    }
    with pytest.raises(modal_app.ModalBoundaryError, match="already exists"):
        modal_app._capture_artifact_verifier_directory(
            source_run_id=source_run_id,
            verifier_run_id=success_id,
            expected_remote_verification=successful_payload,
            identity=identity,
            attempt_id=attempt_id,
            project_root=tmp_path,
        )

    failure_id = "capture-verifier-failure-1"
    write_verifier(failure_id, successful=False)
    failure = modal_app._capture_artifact_verifier_directory(
        source_run_id=source_run_id,
        verifier_run_id=failure_id,
        expected_remote_verification=None,
        identity=identity,
        attempt_id=attempt_id,
        project_root=tmp_path,
    )
    assert failure["remote_verifier_outcome"] == "failure"
    assert failure["validated"] is True
    failure_capture = tmp_path.joinpath(
        *modal_app.modal_artifact_verifier_capture_directory_path(
            identity,
            source_run_id,
            failure_id,
            attempt_id,
        ).parts
    )
    assert {item.name for item in failure_capture.iterdir()} == {
        "artifact_manifest.json",
        "artifact_verification_failure.json",
        "execution_context.json",
        "image_source_manifest.json",
    }


def test_modal_timeout_kills_grandchildren_before_returning(
    tmp_path, monkeypatch
) -> None:
    """A TERM-resistant grandchild cannot write after timeout finalization."""

    import modal_app

    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="modal-timeout-1",
        app_name="rl4rl-architecture-discovery",
        function_name="candidate_smoke",
        modal_app_id=None,
        modal_function_id=None,
        modal_call_id=None,
        modal_image_id=None,
        image_source_sha256="b" * 64,
        artifact_uri=(
            "volume://rl4rl-architecture-artifacts/runs/modal-timeout-1"
        ),
    )
    marker = tmp_path / "forbidden-late-write"
    ready = tmp_path / "grandchild-ready"
    grandchild = (
        "import signal,sys,time;"
        "from pathlib import Path;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "Path(sys.argv[1]).write_text('ready');"
        "time.sleep(0.75);"
        "Path(sys.argv[2]).write_text('late')"
    )
    parent_program = tmp_path / "parent.py"
    parent_program.write_text(
        "\n".join(
            (
                "from pathlib import Path",
                "import subprocess",
                "import sys",
                "import time",
                "subprocess.Popen([sys.executable, '-c', sys.argv[1], "
                "sys.argv[2], sys.argv[3]])",
                "deadline = time.monotonic() + 2",
                "while not Path(sys.argv[2]).exists():",
                "    assert time.monotonic() < deadline",
                "    time.sleep(0.01)",
                "time.sleep(5)",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(modal_app, "REMOTE_PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("PATH", os.environ.get("PATH", "/usr/bin:/bin"))

    started = time.monotonic()
    with pytest.raises(
        modal_app.ProcessGroupClosureError,
        match="nested-process closure unverified",
    ):
        modal_app._run_command(
            [
                sys.executable,
                str(parent_program),
                grandchild,
                str(ready),
                str(marker),
            ],
            context=context,
            provider=False,
            requested_device="cpu",
            timeout_seconds=0.2,
        )
    elapsed = time.monotonic() - started

    assert ready.is_file(), "the adversarial grandchild never started"
    assert elapsed < 2.5
    time.sleep(0.8)
    assert not marker.exists()


def test_modal_command_capture_failure_closes_expected_process_group(
    tmp_path,
    monkeypatch,
) -> None:
    import modal_app

    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="modal-capture-failure-1",
        app_name="rl4rl-architecture-discovery",
        function_name="candidate_smoke",
        modal_app_id=None,
        modal_function_id=None,
        modal_call_id=None,
        modal_image_id=None,
        image_source_sha256="b" * 64,
        artifact_uri=(
            "volume://rl4rl-architecture-artifacts/runs/"
            "modal-capture-failure-1"
        ),
    )
    process = SimpleNamespace(pid=271828)
    cleanup_calls = []
    monkeypatch.setattr(modal_app, "REMOTE_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        modal_app.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    def failed_capture(_process):
        raise OSError("synthetic remote PGID capture failure")

    monkeypatch.setattr(modal_app, "capture_isolated_process_group", failed_capture)
    monkeypatch.setattr(
        modal_app,
        "terminate_process_group",
        lambda child, **kwargs: cleanup_calls.append(
            (child, kwargs["process_group_id"])
        ),
    )

    with pytest.raises(OSError, match="synthetic remote PGID capture failure"):
        modal_app._run_command(
            ["synthetic-remote-command"],
            context=context,
            provider=False,
            requested_device="cpu",
            timeout_seconds=2,
        )

    assert cleanup_calls == [(process, process.pid)]


def test_nested_worker_deadline_closes_its_group_before_outer_finalization(
    tmp_path, monkeypatch
) -> None:
    """The actual training client reaps its nested PGID before the outer cap."""

    import modal_app

    outer_timeout_seconds = 10
    inner_timeout_seconds = 1.5
    delayed_marker_seconds = 3.0
    marker = tmp_path / "forbidden-nested-write"
    ready = tmp_path / "nested-ready"
    term_seen = tmp_path / "nested-sigterm-seen"
    outer_after_inner = tmp_path / "outer-after-inner-cleanup"
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="nested-deadline-1",
        app_name="rl4rl-architecture-discovery",
        function_name="candidate_smoke",
        modal_app_id=None,
        modal_function_id=None,
        modal_call_id=None,
        modal_image_id=None,
        image_source_sha256="f" * 64,
        artifact_uri=(
            "volume://rl4rl-architecture-artifacts/runs/nested-deadline-1"
        ),
    )
    nested_bootstrap = tmp_path / "nested_bootstrap.py"
    nested_bootstrap.write_text(
        "\n".join(
            (
                "import os",
                "import signal",
                "import time",
                "from pathlib import Path",
                "def ignore_sigterm(_signum, _frame):",
                f"    Path({str(term_seen)!r}).write_text('seen')",
                "signal.signal(signal.SIGTERM, ignore_sigterm)",
                (
                    f"Path({str(ready)!r}).write_text("
                    "f'{os.getpid()}\\n{time.monotonic():.9f}\\n')"
                ),
                f"deadline = time.monotonic() + {delayed_marker_seconds!r}",
                "while time.monotonic() < deadline:",
                "    time.sleep(min(0.05, deadline - time.monotonic()))",
                f"Path({str(marker)!r}).write_text('late')",
            )
        ),
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}\n", encoding="utf-8")
    controller_program = tmp_path / "deadline_controller.py"
    controller_program.write_text(
        "\n".join(
            (
                "import os",
                "import sys",
                "import time",
                "from pathlib import Path",
                "from common import training_client",
                "from common.training_config import (",
                "    SMOKE_TRAIN_CUDA_V2, TrainingSeedBundle,",
                ")",
                "training_client.BOOTSTRAP = Path(sys.argv[1])",
                (
                    "remaining = float(os.environ["
                    "training_client.OUTER_PROCESS_DEADLINE_ENV"
                    "]) - time.monotonic()"
                ),
                f"assert remaining > {inner_timeout_seconds!r}",
                (
                    "training_client._OUTER_CLEANUP_GUARD_SECONDS = "
                    f"remaining - {inner_timeout_seconds!r}"
                ),
                "try:",
                "    training_client.run_worker_job(",
                "        mode='train',",
                "        candidate_path=sys.argv[2],",
                "        output_dir=sys.argv[3],",
                "        profile=SMOKE_TRAIN_CUDA_V2,",
                "        seeds=TrainingSeedBundle.from_run_seed(9),",
                "        requested_device='cpu',",
                "        allow_cpu_for_tests=True,",
                "    )",
                "except training_client.WorkerError:",
                "    Path(sys.argv[4]).write_text('inner-closed')",
                "    raise",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(modal_app, "REMOTE_PROJECT_ROOT", modal_app.PROJECT_ROOT)
    monkeypatch.setenv("PYTHONPATH", str(modal_app.PROJECT_ROOT))

    started = time.monotonic()
    with pytest.raises(modal_app.RemoteActionError, match="remote action failed"):
        modal_app._run_command(
            [
                sys.executable,
                str(controller_program),
                str(nested_bootstrap),
                str(candidate),
                str(tmp_path / "training-output"),
                str(outer_after_inner),
            ],
            context=context,
            provider=False,
            requested_device="cpu",
            timeout_seconds=outer_timeout_seconds,
        )
    elapsed = time.monotonic() - started

    assert ready.is_file(), "deadline-aware nested worker never started"
    assert term_seen.is_file(), "inner timeout never sent SIGTERM to the worker"
    assert outer_after_inner.is_file(), "outer finalized before inner cleanup returned"
    nested_pid_text, ready_at_text = ready.read_text(encoding="utf-8").splitlines()
    with pytest.raises(ProcessLookupError):
        os.kill(int(nested_pid_text), 0)
    assert elapsed < outer_timeout_seconds
    delayed_write_deadline = (
        float(ready_at_text) + delayed_marker_seconds + 0.25
    )
    remaining = delayed_write_deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
    assert not marker.exists()


def test_modal_success_closes_background_process_group_before_returning(
    tmp_path, monkeypatch
) -> None:
    """A successful leader cannot leave a late-writing grandchild behind."""

    import modal_app

    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="modal-background-1",
        app_name="rl4rl-architecture-discovery",
        function_name="candidate_smoke",
        modal_app_id=None,
        modal_function_id=None,
        modal_call_id=None,
        modal_image_id=None,
        image_source_sha256="d" * 64,
        artifact_uri=(
            "volume://rl4rl-architecture-artifacts/runs/modal-background-1"
        ),
    )
    marker = tmp_path / "forbidden-background-write"
    ready = tmp_path / "background-ready"
    grandchild = (
        "import signal,sys,time;"
        "from pathlib import Path;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "Path(sys.argv[1]).write_text('ready');"
        "time.sleep(0.75);"
        "Path(sys.argv[2]).write_text('late')"
    )
    parent_program = tmp_path / "successful_parent.py"
    parent_program.write_text(
        "\n".join(
            (
                "from pathlib import Path",
                "import subprocess",
                "import sys",
                "import time",
                "subprocess.Popen([sys.executable, '-c', sys.argv[1], "
                "sys.argv[2], sys.argv[3]])",
                "deadline = time.monotonic() + 2",
                "while not Path(sys.argv[2]).exists():",
                "    assert time.monotonic() < deadline",
                "    time.sleep(0.01)",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(modal_app, "REMOTE_PROJECT_ROOT", tmp_path)

    result = modal_app._run_command(
        [
            sys.executable,
            str(parent_program),
            grandchild,
            str(ready),
            str(marker),
        ],
        context=context,
        provider=False,
        requested_device="cpu",
        timeout_seconds=2,
    )

    assert result["returncode"] == 0
    assert ready.is_file()
    time.sleep(0.8)
    assert not marker.exists()


def test_provider_canary_purges_credential_bytes_before_finalization(
    tmp_path, monkeypatch
) -> None:
    import modal_app

    secret = "fake-provider-key-never-artifact"
    run_directory = tmp_path / "provider-canary-1"
    run_directory.mkdir()
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="provider-canary-1",
        app_name="rl4rl-architecture-discovery",
        function_name="canary_openevolve_generic",
        modal_app_id=None,
        modal_function_id=None,
        modal_call_id=None,
        modal_image_id=None,
        image_source_sha256="c" * 64,
        artifact_uri=(
            "volume://rl4rl-architecture-artifacts/runs/provider-canary-1"
        ),
    )

    class FakeVolume:
        commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = FakeVolume()
    monkeypatch.setenv("DISCOVERY_API_KEY", secret)
    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", volume)
    monkeypatch.setattr(
        modal_app,
        "_prepare_run",
        lambda _function_name, _run_id: (run_directory, context),
    )

    def leaking_action(directory: Path, _context: ExecutionContextV1):
        assert directory != run_directory
        assert not directory.is_relative_to(run_directory)
        (directory / "safe.json").write_text('{"safe":true}\n', encoding="utf-8")
        (directory / "leaked.txt").write_bytes(b"prefix" + secret.encode() + b"suffix")
        nested = directory / "nested"
        nested.mkdir()
        (nested / "second.bin").write_bytes(secret.encode())
        return {"mode": "test-provider-canary"}

    with pytest.raises(modal_app.RemoteActionError, match="credential material"):
        modal_app._execute_new_run(
            "canary_openevolve_generic",
            "provider-canary-1",
            leaking_action,
        )

    assert not (run_directory / "leaked.txt").exists()
    assert not (run_directory / "nested" / "second.bin").exists()
    assert not (run_directory / "safe.json").exists()
    assert volume.commits == 1
    for path in run_directory.rglob("*"):
        if path.is_file() and not path.is_symlink():
            assert secret.encode() not in path.read_bytes()
    failure = json.loads(
        (run_directory / "remote_action_failure.json").read_text(encoding="utf-8")
    )
    assert failure["message"] == "remote action failed; details suppressed"
    manifest = modal_app.load_artifact_manifest(
        run_directory / "artifact_manifest.json"
    )
    assert modal_app.verify_artifact_manifest(run_directory, manifest)["verified"]


def test_failed_provider_canary_persists_only_its_sanitized_attempt_ledger(
    tmp_path,
    monkeypatch,
) -> None:
    import modal_app
    from common.gpt56_sol import OFFICIAL_OPENAI_API_BASE, TARGET_MODEL
    from common.provider_attempts import (
        ProviderAttemptLedger,
        load_provider_attempt_ledger,
    )

    run_id = "provider-failure-ledger-1"
    run_directory = tmp_path / run_id
    run_directory.mkdir()
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id=run_id,
        app_name="rl4rl-architecture-discovery",
        function_name="canary_openevolve_generic",
        modal_app_id="ap-test123",
        modal_function_id="fu-test123",
        modal_call_id="fc-test123",
        modal_image_id="im-test123",
        image_source_sha256="c" * 64,
        artifact_uri=(
            f"volume://rl4rl-architecture-artifacts/runs/{run_id}"
        ),
    )

    class FakeVolume:
        commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = FakeVolume()
    monkeypatch.setenv("DISCOVERY_API_KEY", "offline-provider-key")
    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", volume)
    monkeypatch.setattr(
        modal_app,
        "_prepare_run",
        lambda _function_name, _run_id: (run_directory, context),
    )

    class OfflineProviderFailure(RuntimeError):
        request_id = "req_failure123"

    def failing_action(directory: Path, _context: ExecutionContextV1):
        controller = directory / "controller"
        controller.mkdir()
        (controller / "partial-output.txt").write_text(
            "must remain quarantined",
            encoding="utf-8",
        )
        ledger = ProviderAttemptLedger.create(
            controller / "provider_attempts.jsonl",
            harness="openevolve_generic",
            action="one_opportunity_engineering_canary",
            controller_run_id="controller-run-1",
            api_endpoint=OFFICIAL_OPENAI_API_BASE,
            model=TARGET_MODEL,
            environ={
                "DISCOVERY_EXECUTION_CONTEXT_JSON": json.dumps(
                    context.to_dict()
                )
            },
        )

        def fail():
            raise OfflineProviderFailure(
                "raw provider body must never reach the Volume"
            )

        return ledger.record_call(
            {
                "model": TARGET_MODEL,
                "messages": [{"role": "user", "content": "private prompt"}],
                "reasoning_effort": "high",
                "max_completion_tokens": 16_384,
                "seed": 1,
            },
            fail,
        )

    with pytest.raises(OfflineProviderFailure):
        modal_app._execute_new_run(
            "canary_openevolve_generic",
            run_id,
            failing_action,
        )

    published = run_directory / "controller" / "provider_attempts.jsonl"
    record = load_provider_attempt_ledger(published)[0]
    assert record.status == "error"
    assert record.error_class == "OfflineProviderFailure"
    assert record.provider_request_id == "req_failure123"
    assert not (run_directory / "controller" / "partial-output.txt").exists()
    persisted = b"".join(
        path.read_bytes()
        for path in run_directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    assert b"raw provider body" not in persisted
    assert b"private prompt" not in persisted
    assert volume.commits == 1


def test_post_request_staging_failure_preserves_only_success_attempt_usage(
    tmp_path,
    monkeypatch,
) -> None:
    import modal_app
    from common.gpt56_sol import OFFICIAL_OPENAI_API_BASE, TARGET_MODEL
    from common.provider_attempts import (
        ProviderAttemptLedger,
        load_provider_attempt_ledger,
    )

    run_id = "provider-post-request-invalid-1"
    run_directory = tmp_path / run_id
    run_directory.mkdir()
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id=run_id,
        app_name="rl4rl-architecture-discovery",
        function_name="canary_greedy_autoresearch",
        modal_app_id="ap-test123",
        modal_function_id="fu-test123",
        modal_call_id="fc-success123",
        modal_image_id="im-test123",
        image_source_sha256="d" * 64,
        artifact_uri=f"volume://rl4rl-architecture-artifacts/runs/{run_id}",
    )

    class FakeVolume:
        commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = FakeVolume()
    monkeypatch.setenv("DISCOVERY_API_KEY", "offline-provider-key")
    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", volume)
    monkeypatch.setattr(
        modal_app,
        "_prepare_run",
        lambda _function_name, _run_id: (run_directory, context),
    )

    def invalid_after_success(directory: Path, _context: ExecutionContextV1):
        controller = directory / "controller"
        controller.mkdir()
        ledger = ProviderAttemptLedger.create(
            controller / "provider_attempts.jsonl",
            harness="greedy_autoresearch",
            action="one_opportunity_engineering_canary",
            controller_run_id="controller-success-1",
            api_endpoint=OFFICIAL_OPENAI_API_BASE,
            model=TARGET_MODEL,
            environ={
                "DISCOVERY_EXECUTION_CONTEXT_JSON": json.dumps(context.to_dict())
            },
        )
        response = SimpleNamespace(
            id="chatcmpl-success123",
            _request_id="req_success123",
            usage=SimpleNamespace(
                prompt_tokens=321,
                completion_tokens=45,
                total_tokens=366,
            ),
        )
        ledger.record_call(
            {
                "model": TARGET_MODEL,
                "messages": [
                    {"role": "user", "content": "private paid prompt text"}
                ],
                "reasoning_effort": "high",
                "max_completion_tokens": 16_384,
                "seed": 1,
            },
            lambda: response,
        )
        _write = controller / "provider-response.txt"
        _write.write_text("private provider response text", encoding="utf-8")
        (controller / "run_manifest.json").write_text("{}\n", encoding="utf-8")
        return {"mode": "one_opportunity_engineering_canary"}

    with pytest.raises(ValueError, match="top-level roster"):
        modal_app._execute_new_run(
            "canary_greedy_autoresearch",
            run_id,
            invalid_after_success,
        )

    published = run_directory / "controller" / "provider_attempts.jsonl"
    records = load_provider_attempt_ledger(published)
    assert len(records) == 1
    record = records[0]
    assert record.status == "success"
    assert record.provider_response_id == "chatcmpl-success123"
    assert record.provider_request_id == "req_success123"
    assert record.input_tokens == 321
    assert record.output_tokens == 45
    assert record.total_tokens == 366
    assert record.error_class is None
    assert not (run_directory / "controller" / "run_manifest.json").exists()
    assert not (run_directory / "controller" / "provider-response.txt").exists()
    persisted = b"".join(
        path.read_bytes()
        for path in run_directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    assert b"private paid prompt text" not in persisted
    assert b"private provider response text" not in persisted
    assert b"offline-provider-key" not in persisted
    assert volume.commits == 1


def test_provider_canary_publishes_only_after_ephemeral_scan(
    tmp_path, monkeypatch
) -> None:
    import modal_app
    import scripts.validate_engineering_canaries as canary_validator

    secret = "fake-provider-key-not-published"
    run_directory = tmp_path / "provider-safe-1"
    run_directory.mkdir()
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="provider-safe-1",
        app_name="rl4rl-architecture-discovery",
        function_name="canary_greedy_autoresearch",
        modal_app_id=None,
        modal_function_id=None,
        modal_call_id=None,
        modal_image_id=None,
        image_source_sha256="2" * 64,
        artifact_uri=(
            "volume://rl4rl-architecture-artifacts/runs/provider-safe-1"
        ),
    )

    class FakeVolume:
        commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = FakeVolume()
    monkeypatch.setenv("DISCOVERY_API_KEY", secret)
    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", volume)
    monkeypatch.setattr(
        modal_app,
        "_prepare_run",
        lambda _function_name, _run_id: (run_directory, context),
    )
    action_directories = []
    validation_calls = []

    def validate_staging(directory, *, harness, execution_context):
        validation_calls.append((Path(directory), harness, execution_context))
        assert (Path(directory) / "safe.json").is_file()
        assert not (run_directory / "controller").exists()
        return {"valid": True}

    monkeypatch.setattr(
        canary_validator,
        "validate_private_canary_staging",
        validate_staging,
    )

    def safe_action(directory: Path, _context: ExecutionContextV1):
        action_directories.append(directory)
        assert directory != run_directory
        nested = directory / "controller"
        nested.mkdir()
        (nested / "safe.json").write_text('{"safe":true}\n', encoding="utf-8")
        assert not (run_directory / "controller").exists()
        return {"mode": "safe-provider-test"}

    result = modal_app._execute_new_run(
        "canary_greedy_autoresearch",
        "provider-safe-1",
        safe_action,
    )

    assert result["success"] is True
    assert validation_calls == [
        (
            action_directories[0] / "controller",
            "greedy_autoresearch",
            context,
        )
    ]
    assert (run_directory / "controller" / "safe.json").is_file()
    assert not action_directories[0].exists()
    assert volume.commits == 1
    manifest = modal_app.load_artifact_manifest(
        run_directory / "artifact_manifest.json"
    )
    assert modal_app.verify_artifact_manifest(run_directory, manifest)["verified"]


def test_provider_same_size_replacement_after_scan_cannot_publish_key(
    tmp_path, monkeypatch
) -> None:
    import modal_app
    import scripts.validate_engineering_canaries as canary_validator

    secret = b"same-size-provider-secret"
    safe_payload = b"x" * len(secret)
    run_directory = tmp_path / "provider-snapshot-1"
    run_directory.mkdir()
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="provider-snapshot-1",
        app_name="rl4rl-architecture-discovery",
        function_name="canary_greedy_autoresearch",
        modal_app_id=None,
        modal_function_id=None,
        modal_call_id=None,
        modal_image_id=None,
        image_source_sha256="3" * 64,
        artifact_uri="volume://rl4rl-architecture-artifacts/runs/provider-snapshot-1",
    )

    class FakeVolume:
        commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = FakeVolume()
    monkeypatch.setenv("DISCOVERY_API_KEY", secret.decode())
    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", volume)
    monkeypatch.setattr(
        modal_app,
        "_prepare_run",
        lambda _function_name, _run_id: (run_directory, context),
    )
    monkeypatch.setattr(
        canary_validator,
        "validate_private_canary_staging",
        lambda *_args, **_kwargs: {"valid": True},
    )
    staged_artifact: list[Path] = []

    def safe_action(directory: Path, _context: ExecutionContextV1):
        artifact = directory / "controller" / "result.bin"
        artifact.parent.mkdir()
        artifact.write_bytes(safe_payload)
        staged_artifact.append(artifact)
        return {"mode": "provider-snapshot-test"}

    original_snapshot = modal_app._snapshot_provider_artifacts

    def snapshot_then_replace(
        staging: Path,
        run: Path,
        *,
        reserved_artifacts: tuple[tuple[str, bytes], ...] = (),
    ):
        snapshots = original_snapshot(
            staging,
            run,
            reserved_artifacts=reserved_artifacts,
        )
        assert len(staged_artifact) == 1
        staged_artifact[0].write_bytes(secret)
        assert staged_artifact[0].stat().st_size == len(safe_payload)
        return snapshots

    monkeypatch.setattr(
        modal_app,
        "_snapshot_provider_artifacts",
        snapshot_then_replace,
    )

    result = modal_app._execute_new_run(
        "canary_greedy_autoresearch",
        "provider-snapshot-1",
        safe_action,
    )

    published = run_directory / "controller" / "result.bin"
    assert result["success"] is True
    assert published.read_bytes() == safe_payload
    assert secret not in b"".join(
        path.read_bytes()
        for path in run_directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    assert volume.commits == 1
    manifest = modal_app.load_artifact_manifest(
        run_directory / "artifact_manifest.json"
    )
    assert modal_app.verify_artifact_manifest(run_directory, manifest)["verified"]


def test_provider_publication_rejects_symlinks_and_oversize(
    tmp_path, monkeypatch
) -> None:
    import modal_app

    staging = tmp_path / "staging"
    run = tmp_path / "run"
    staging.mkdir()
    run.mkdir()
    target = staging / "target.txt"
    target.write_text("safe", encoding="utf-8")
    (staging / "link.txt").symlink_to(target)
    with pytest.raises(modal_app.RemoteActionError, match="symlinks"):
        modal_app._publish_scanned_provider_artifacts(staging, run)
    assert not any(run.iterdir())

    (staging / "link.txt").unlink()
    monkeypatch.setattr(modal_app, "MAX_ARTIFACT_DOWNLOAD_FILE_BYTES", 3)
    with pytest.raises(modal_app.RemoteActionError, match="per-file"):
        modal_app._publish_scanned_provider_artifacts(staging, run)
    assert not any(run.iterdir())


def test_provider_free_publication_cleans_all_partial_destinations(
    tmp_path, monkeypatch
) -> None:
    import modal_app

    staging = tmp_path / "staging"
    run = tmp_path / "run"
    (staging / "a").mkdir(parents=True)
    (staging / "b").mkdir()
    run.mkdir()
    (staging / "a" / "first.bin").write_bytes(b"first-safe")
    (staging / "b" / "second.bin").write_bytes(b"second-safe")
    original_open = Path.open

    def fail_second_destination(path: Path, mode="r", *args, **kwargs):
        if path == run / "b" / "second.bin" and mode == "xb":
            raise OSError("forced publication failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_second_destination)

    with pytest.raises(OSError, match="forced publication failure"):
        modal_app._publish_staged_artifacts(
            staging,
            run,
            scan_provider_credential=False,
        )

    assert not any(run.iterdir())


def test_staged_publication_reserves_provenance_and_exact_final_result_bytes(
    tmp_path,
    monkeypatch,
) -> None:
    import modal_app

    staging = tmp_path / "staging"
    run = tmp_path / "run"
    staging.mkdir()
    run.mkdir()
    (run / "execution_context.json").write_bytes(b"ctx!")
    (staging / "result.bin").write_bytes(b"data")
    reserved = (("remote_action_result.json", b"done"),)
    monkeypatch.setattr(modal_app, "MAX_ARTIFACT_DOWNLOAD_FILE_BYTES", 32)
    monkeypatch.setattr(modal_app, "MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES", 11)

    with pytest.raises(modal_app.RemoteActionError, match="aggregate byte cap"):
        modal_app._publish_staged_artifacts(
            staging,
            run,
            scan_provider_credential=False,
            reserved_artifacts=reserved,
        )

    assert {path.name for path in run.iterdir()} == {"execution_context.json"}
    monkeypatch.setattr(modal_app, "MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES", 12)
    modal_app._publish_staged_artifacts(
        staging,
        run,
        scan_provider_credential=False,
        reserved_artifacts=reserved,
    )
    assert (run / "result.bin").read_bytes() == b"data"


def test_many_tiny_long_path_artifacts_fail_before_manifest_overflow_publication(
    tmp_path,
    monkeypatch,
) -> None:
    import modal_app

    staging = tmp_path / "staging"
    run = tmp_path / "manifest-roster-run"
    staging.mkdir()
    run.mkdir()
    (run / "execution_context.json").write_bytes(b"{}\n")
    for index in range(8):
        name = f"artifact-{index}-{'x' * 80}.json"
        (staging / name).write_bytes(b"x")
    monkeypatch.setattr(modal_app, "MAX_ARTIFACT_MANIFEST_BYTES", 900)

    with pytest.raises(
        modal_app.RemoteActionError,
        match="prospective artifact manifest exceeds its byte cap",
    ):
        modal_app._publish_staged_artifacts(
            staging,
            run,
            scan_provider_credential=False,
            reserved_artifacts=(("remote_action_result.json", b"{}\n"),),
        )

    assert {path.name for path in run.iterdir()} == {"execution_context.json"}


def test_provider_stream_leak_is_rejected_before_any_digest_or_manifest_hash(
    tmp_path, monkeypatch
) -> None:
    import modal_app

    secret = "fake-provider-key-never-hash"
    run_directory = tmp_path / "provider-stream-1"
    run_directory.mkdir()
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="provider-stream-1",
        app_name="rl4rl-architecture-discovery",
        function_name="canary_greedy_autoresearch",
        modal_app_id=None,
        modal_function_id=None,
        modal_call_id=None,
        modal_image_id=None,
        image_source_sha256="1" * 64,
        artifact_uri=(
            "volume://rl4rl-architecture-artifacts/runs/provider-stream-1"
        ),
    )

    class FakeVolume:
        commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = FakeVolume()
    monkeypatch.setenv("DISCOVERY_API_KEY", secret)
    monkeypatch.setenv("DISCOVERY_API_BASE", "https://api.openai.com/v1")
    monkeypatch.setenv("DISCOVERY_MODEL", "gpt-5.6-sol")
    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", volume)
    monkeypatch.setattr(modal_app, "REMOTE_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        modal_app,
        "_prepare_run",
        lambda _function_name, _run_id: (run_directory, context),
    )

    def leaking_action(_directory: Path, _context: ExecutionContextV1):
        return modal_app._run_command(
            [
                sys.executable,
                "-c",
                (
                    "import os,sys;"
                    "key=os.environ['DISCOVERY_API_KEY'];"
                    "print(key);"
                    "sys.stderr.write(key+'\\n')"
                ),
            ],
            context=context,
            provider=True,
            requested_device="cpu",
            timeout_seconds=2,
        )

    with pytest.raises(modal_app.RemoteActionError, match="credential material"):
        modal_app._execute_new_run(
            "canary_greedy_autoresearch",
            "provider-stream-1",
            leaking_action,
            scan_provider_credential=True,
        )

    persisted = b"".join(
        path.read_bytes()
        for path in run_directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    assert secret.encode() not in persisted
    leaked_stream_digest = hashlib.sha256((secret + "\n").encode()).hexdigest()
    assert leaked_stream_digest.encode() not in persisted
    failure_result = json.loads(
        (run_directory / "remote_action_result.json").read_text(encoding="utf-8")
    )
    assert "stdout_sha256" not in failure_result
    assert "stderr_sha256" not in failure_result
    assert volume.commits == 1


def test_provider_failure_publishes_request_start_uncertainty_for_empty_ledger(
    tmp_path,
    monkeypatch,
) -> None:
    import modal_app

    staging = tmp_path / "private-staging"
    controller = staging / "controller"
    controller.mkdir(parents=True)
    ledger = controller / modal_app.PROVIDER_ATTEMPT_LEDGER_FILENAME
    ledger.write_bytes(b"")
    run_directory = tmp_path / "volume-run"
    run_directory.mkdir()
    monkeypatch.setenv("DISCOVERY_API_KEY", "synthetic-provider-key")

    modal_app._publish_failed_provider_attempt_ledger(
        staging,
        run_directory,
        harness="greedy_autoresearch",
        context=ExecutionContextV1(
            execution_backend="modal",
            run_id="zero-provider-attempt",
            app_name=modal_app.APP_NAME,
            function_name="canary_greedy_autoresearch",
            modal_app_id="ap-zero",
            modal_function_id="fu-zero",
            modal_call_id="fc-zero",
            modal_image_id="im-zero",
            image_source_sha256="a" * 64,
            artifact_uri=(
                "volume://rl4rl-architecture-artifacts/runs/zero-provider-attempt"
            ),
        ),
    )

    published = run_directory / "controller" / ledger.name
    assert published.is_file()
    assert published.read_bytes() == b""
    assert modal_app.load_provider_attempt_ledger(published) == ()
    uncertain = json.loads(
        (
            run_directory / "controller/provider_request_start_uncertain.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert uncertain["schema_name"] == "ProviderRequestStartUncertainEvidence"
    assert uncertain["provider_attempt_count_lower_bound"] == 0
    assert uncertain["provider_attempt_count_upper_bound"] == 1
    assert uncertain["provider_request_started"] == "unknown"
    assert uncertain["provider_attempt_ledger_state"] == "present"
    assert uncertain["billing_treatment"] == "reserve_one_full_approved_request"
    assert uncertain["modal_call_id"] == "fc-zero"
    assert not (
        run_directory / "controller/provider_request_not_started.json"
    ).exists()


def test_provider_failure_without_a_ledger_is_still_upper_bound_one(
    tmp_path,
    monkeypatch,
) -> None:
    import modal_app

    staging = tmp_path / "private-staging"
    staging.mkdir()
    run_directory = tmp_path / "volume-run"
    run_directory.mkdir()
    monkeypatch.setenv("DISCOVERY_API_KEY", "synthetic-provider-key")
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="missing-provider-ledger",
        app_name=modal_app.APP_NAME,
        function_name="canary_greedy_autoresearch",
        modal_app_id="ap-missing",
        modal_function_id="fu-missing",
        modal_call_id="fc-missing",
        modal_image_id="im-missing",
        image_source_sha256="a" * 64,
        artifact_uri=(
            "volume://rl4rl-architecture-artifacts/runs/missing-provider-ledger"
        ),
    )

    modal_app._publish_failed_provider_attempt_ledger(
        staging,
        run_directory,
        harness="greedy_autoresearch",
        context=context,
    )

    published = run_directory / "controller/provider_attempts.jsonl"
    assert published.read_bytes() == b""
    uncertain = json.loads(
        (
            run_directory / "controller/provider_request_start_uncertain.json"
        ).read_text(encoding="utf-8")
    )
    assert uncertain["provider_attempt_count_lower_bound"] == 0
    assert uncertain["provider_attempt_count_upper_bound"] == 1
    assert uncertain["provider_attempt_ledger_state"] == "missing"
    assert uncertain["billing_treatment"] == "reserve_one_full_approved_request"
    assert not (
        run_directory / "controller/provider_request_not_started.json"
    ).exists()


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("multiple", "multiple attempts"),
        ("mismatched_harness", "differs from its canary context"),
    ),
)
def test_failed_provider_ledger_must_be_one_opportunity_and_context_bound(
    tmp_path,
    monkeypatch,
    case,
    message,
) -> None:
    import modal_app
    from common.gpt56_sol import OFFICIAL_OPENAI_API_BASE, TARGET_MODEL
    from common.provider_attempts import ProviderAttemptLedger

    run_id = "failed-ledger-bound-1"
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id=run_id,
        app_name=modal_app.APP_NAME,
        function_name="canary_greedy_autoresearch",
        modal_app_id="ap-bound",
        modal_function_id="fu-bound",
        modal_call_id="fc-bound",
        modal_image_id="im-bound",
        image_source_sha256="a" * 64,
        artifact_uri=f"volume://rl4rl-architecture-artifacts/runs/{run_id}",
    )
    staging = tmp_path / "staging"
    controller = staging / "controller"
    controller.mkdir(parents=True)
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    ledger = ProviderAttemptLedger.create(
        controller / modal_app.PROVIDER_ATTEMPT_LEDGER_FILENAME,
        harness=(
            "semantic_autoresearch"
            if case == "mismatched_harness"
            else "greedy_autoresearch"
        ),
        action="one_opportunity_engineering_canary",
        controller_run_id="controller-bound-1",
        api_endpoint=OFFICIAL_OPENAI_API_BASE,
        model=TARGET_MODEL,
        environ={"DISCOVERY_EXECUTION_CONTEXT_JSON": json.dumps(context.to_dict())},
    )
    request = {
        "model": TARGET_MODEL,
        "messages": [{"role": "user", "content": "private"}],
        "max_completion_tokens": 16_384,
        "seed": 1,
    }
    count = 2 if case == "multiple" else 1
    for index in range(count):
        response = SimpleNamespace(
            id=f"chatcmpl-bound{index}",
            _request_id=f"req_bound{index}",
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )
        ledger.record_call(request, lambda response=response: response)
    monkeypatch.setenv("DISCOVERY_API_KEY", "synthetic-provider-key")

    with pytest.raises(modal_app.RemoteActionError, match=message):
        modal_app._publish_failed_provider_attempt_ledger(
            staging,
            run_directory,
            harness="greedy_autoresearch",
            context=context,
        )

    assert not any(run_directory.iterdir())


def test_remote_failure_record_never_persists_absolute_error_paths(
    tmp_path,
) -> None:
    import modal_app

    modal_app._record_failure(
        tmp_path,
        RuntimeError(
            "failed at /mnt/discovery/runs/private and /opt/architecture_discovery"
        ),
    )
    payload = json.loads(
        (tmp_path / "remote_action_failure.json").read_text(encoding="utf-8")
    )
    encoded = json.dumps(payload)
    assert payload == {
        "error_type": "RuntimeError",
        "message": "remote action failed; details suppressed",
    }
    assert "/mnt/discovery" not in encoded
    assert "/opt/architecture_discovery" not in encoded


def test_bounded_run_json_is_create_once_under_competing_writers(
    tmp_path,
    monkeypatch,
) -> None:
    import modal_app

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    barrier = __import__("threading").Barrier(2)
    original_budget = modal_app._ensure_retained_artifact_budget

    def synchronized_budget(*args, **kwargs):
        result = original_budget(*args, **kwargs)
        barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(
        modal_app,
        "_ensure_retained_artifact_budget",
        synchronized_budget,
    )

    def write_once(value: int):
        try:
            return modal_app._write_bounded_run_json(
                run_directory,
                "result.json",
                {"value": value},
            )
        except BaseException as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(write_once, (1, 2)))

    successes = [item for item in outcomes if isinstance(item, bytes)]
    failures = [item for item in outcomes if isinstance(item, BaseException)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], modal_app.RemoteActionError)
    assert "already exists" in str(failures[0])
    assert (run_directory / "result.json").read_bytes() == successes[0]


def test_bounded_run_json_quarantines_a_short_write(tmp_path, monkeypatch) -> None:
    import modal_app

    run_directory = tmp_path / "run"
    run_directory.mkdir()
    monkeypatch.setattr(modal_app.os, "write", lambda _fd, _payload: 0)

    with pytest.raises(modal_app.RemoteActionError, match="made no progress"):
        modal_app._write_bounded_run_json(
            run_directory,
            "result.json",
            {"value": 1},
        )

    assert not (run_directory / "result.json").exists()


@pytest.mark.parametrize(
    ("action", "harness"),
    (
        ("canaries", ""),
        ("canary", "openevolve_generic"),
    ),
)
def test_provider_canary_actions_require_distinct_provider_approval(
    action, harness
) -> None:
    import modal_app

    with pytest.raises(SystemExit, match="--provider-approved"):
        modal_app._require_local_action_approvals(
            action=action,
            approved=True,
            provider_approved=False,
            harness=harness,
            expected_image_source_sha256=modal_app.IMAGE_SOURCE_SHA256,
        )


def test_single_canary_recovery_is_one_synchronous_opportunity() -> None:
    import modal_app

    calls = []

    class FakeRemote:
        def __init__(self, name: str) -> None:
            self.name = name

        def remote(self, **kwargs):
            calls.append((self.name, kwargs))
            return {"ok": True}

    functions = {name: FakeRemote(name) for name in modal_app.CANARY_ORDER}
    result = modal_app._run_single_canary_synchronously(
        functions,
        harness="openevolve_generic",
        run_id="canary-recovery-1",
    )

    assert result == {
        "harness": "openevolve_generic",
        "result": {"ok": True},
    }
    assert calls == [
        (
            "openevolve_generic",
            {"run_id": "canary-recovery-1", "opportunities": 1},
        )
    ]


def test_single_canary_rejects_nonfrozen_harness_before_invocation() -> None:
    import modal_app

    with pytest.raises(SystemExit, match="one frozen ID"):
        modal_app._require_local_action_approvals(
            action="canary",
            approved=True,
            provider_approved=True,
            harness="arbitrary-harness",
            expected_image_source_sha256=modal_app.IMAGE_SOURCE_SHA256,
        )


def test_remote_actions_require_the_exact_approved_image_source_hash() -> None:
    import modal_app

    with pytest.raises(SystemExit, match="expected-image-source-sha256"):
        modal_app._require_local_action_approvals(
            action="cuda-environment",
            approved=True,
            provider_approved=False,
            harness="",
            expected_image_source_sha256="0" * 64,
        )


def test_remote_runtime_rejects_baked_image_identity_drift(monkeypatch) -> None:
    import modal_app

    monkeypatch.setenv(modal_app.IMAGE_SOURCE_IDENTITY_ENV, "0" * 64)
    with pytest.raises(modal_app.RemoteActionError, match="baked image source"):
        modal_app._prepare_run("cuda_environment", "identity-drift-1")


def test_prepare_run_explicitly_trusts_only_the_frozen_mount_alias(
    tmp_path, monkeypatch
) -> None:
    import modal_app

    run_directory = tmp_path / "prepared-run"
    context = SimpleNamespace(to_dict=lambda: {"run_id": "mount-opt-in-1"})
    calls = []

    class FakeVolume:
        reloads = 0
        commits = 0

        def reload(self) -> None:
            self.reloads += 1

        def commit(self) -> None:
            self.commits += 1

    volume = FakeVolume()

    def create_run(mount_root, run_id, *, allow_mount_root_symlink=False):
        calls.append((mount_root, run_id, allow_mount_root_symlink))
        run_directory.mkdir()
        return run_directory

    monkeypatch.setenv(
        modal_app.IMAGE_SOURCE_IDENTITY_ENV,
        modal_app.IMAGE_SOURCE_SHA256,
    )
    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", volume)
    monkeypatch.setattr(modal_app, "create_fresh_run_directory", create_run)
    monkeypatch.setattr(modal_app, "_execution_context", lambda *_args: context)
    monkeypatch.setattr(modal_app, "_atomic_json", lambda *_args: None)

    actual_directory, actual_context = modal_app._prepare_run(
        "cuda_environment",
        "mount-opt-in-1",
    )

    assert actual_directory == run_directory
    assert actual_context is context
    assert calls == [
        (modal_app.VOLUME_MOUNT_PATH, "mount-opt-in-1", True),
    ]
    assert volume.reloads == 1
    assert volume.commits == 1


def test_durable_pre_action_lease_precedes_staging_and_refuses_restart(
    tmp_path,
    monkeypatch,
) -> None:
    import modal_app
    from modal_boundary import ModalBoundaryError, volume_artifact_uri

    volume_root = tmp_path / "volume"
    volume_root.mkdir()
    run_id = "durable-lease-1"
    run_directory = volume_root / "runs" / run_id
    action_directories: list[Path] = []

    class FakeVolume:
        reloads = 0
        commits = 0

        def reload(self) -> None:
            self.reloads += 1

        def commit(self) -> None:
            self.commits += 1
            if self.commits == 1:
                assert {path.name for path in run_directory.iterdir()} == {
                    "execution_context.json",
                    "image_source_manifest.json",
                }

    volume = FakeVolume()

    def context(function_name: str, selected_run_id: str) -> ExecutionContextV1:
        return ExecutionContextV1(
            execution_backend="modal",
            run_id=selected_run_id,
            app_name=modal_app.APP_NAME,
            function_name=function_name,
            modal_app_id="ap-lease",
            modal_function_id="fu-lease",
            modal_call_id="fc-lease",
            modal_image_id="im-lease",
            image_source_sha256=modal_app.IMAGE_SOURCE_SHA256,
            artifact_uri=volume_artifact_uri(selected_run_id),
        )

    monkeypatch.setenv(
        modal_app.IMAGE_SOURCE_IDENTITY_ENV,
        modal_app.IMAGE_SOURCE_SHA256,
    )
    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", volume)
    monkeypatch.setattr(modal_app, "VOLUME_MOUNT_PATH", volume_root)
    monkeypatch.setattr(modal_app, "_execution_context", context)
    monkeypatch.setattr(
        modal_app,
        "_provider_free_network_denial_probe",
        lambda _context: {"denied": True},
    )

    def action(directory: Path, _context: ExecutionContextV1) -> dict[str, str]:
        action_directories.append(directory)
        assert volume.commits == 1
        assert directory != run_directory
        assert not directory.resolve().is_relative_to(volume_root.resolve())
        assert not (run_directory / "cuda_environment.json").exists()
        (directory / "cuda_environment.json").write_text(
            '{"cuda":true}\n',
            encoding="utf-8",
        )
        return {"mode": "lease-order-test"}

    result = modal_app._execute_new_run("cuda_environment", run_id, action)

    assert result["success"] is True
    assert volume.commits == 2
    assert volume.reloads == 1
    assert (run_directory / "cuda_environment.json").is_file()
    assert not action_directories[0].exists()

    with pytest.raises(ModalBoundaryError, match="already exists"):
        modal_app._execute_new_run("cuda_environment", run_id, action)

    assert len(action_directories) == 1
    assert volume.commits == 2
    assert volume.reloads == 2


def test_uncertain_process_group_closure_never_finalizes_artifacts(
    tmp_path, monkeypatch
) -> None:
    import modal_app
    from modal_boundary import volume_artifact_uri

    volume_root = tmp_path / "volume"
    volume_root.mkdir()
    run_id = "uncertain-process-group-1"
    run_directory = volume_root / "runs" / run_id
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id=run_id,
        app_name="rl4rl-architecture-discovery",
        function_name="candidate_smoke",
        modal_app_id=None,
        modal_function_id=None,
        modal_call_id=None,
        modal_image_id=None,
        image_source_sha256=modal_app.IMAGE_SOURCE_SHA256,
        artifact_uri=volume_artifact_uri(run_id),
    )

    class FakeVolume:
        reloads = 0
        commits = 0

        def reload(self) -> None:
            self.reloads += 1

        def commit(self) -> None:
            self.commits += 1

    volume = FakeVolume()
    monkeypatch.setenv(
        modal_app.IMAGE_SOURCE_IDENTITY_ENV,
        modal_app.IMAGE_SOURCE_SHA256,
    )
    monkeypatch.setattr(modal_app, "ARTIFACT_VOLUME", volume)
    monkeypatch.setattr(modal_app, "VOLUME_MOUNT_PATH", volume_root)
    monkeypatch.setattr(modal_app, "_execution_context", lambda *_args: context)
    monkeypatch.setattr(
        modal_app,
        "_provider_free_network_denial_probe",
        lambda _context: {"denied": True},
    )
    staged_directories: list[Path] = []

    def uncertain_action(directory: Path, _context: ExecutionContextV1):
        staged_directories.append(directory)
        assert directory != run_directory
        (directory / "partial-training.bin").write_bytes(b"partial")
        raise modal_app.ProcessGroupClosureError("group closure unconfirmed")

    with pytest.raises(modal_app.ProcessGroupClosureError):
        modal_app._execute_new_run(
            "candidate_smoke",
            run_id,
            uncertain_action,
        )

    assert volume.reloads == 1
    assert volume.commits == 1
    assert not staged_directories[0].exists()
    assert {path.name for path in run_directory.iterdir()} == {
        "execution_context.json",
        "image_source_manifest.json",
    }
    assert not (run_directory / "artifact_manifest.json").exists()
    assert not (run_directory / "remote_action_failure.json").exists()
