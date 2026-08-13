from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import pytest
from openevolve.config import LLMModelConfig
from openevolve.database import ProgramDatabase

ROOT = Path(__file__).resolve().parents[1]
PROCESS_PARALLEL_SOURCE = (
    ROOT / "vendor" / "openevolve" / "openevolve" / "process_parallel.py"
)


def _load_vendored_process_parallel():
    module_name = "vendored_process_parallel_boundary_test"
    spec = importlib.util.spec_from_file_location(module_name, PROCESS_PARALLEL_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_process_pool_initargs_never_pickle_literal_provider_key(
    monkeypatch,
) -> None:
    process_parallel = _load_vendored_process_parallel()
    config = process_parallel.Config()
    literal_key = "fake-provider-key-never-pickle"
    model = LLMModelConfig(name="test-model", api_key=literal_key)
    config.llm.api_key = literal_key
    config.llm.models = [model]
    config.llm.evaluator_models = [model]
    database = ProgramDatabase(config.database)
    captured = {}

    class FakeExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("DISCOVERY_API_KEY", literal_key)
    monkeypatch.setattr(process_parallel, "ProcessPoolExecutor", FakeExecutor)
    controller = process_parallel.ProcessParallelController(
        config,
        str(ROOT / "agents" / "openevolve_generic" / "evaluator.py"),
        database,
    )
    controller.start()

    pickled = pickle.dumps(captured["initargs"])
    assert literal_key.encode() not in pickled
    assert b"${DISCOVERY_API_KEY}" in pickled
    assert len(captured["initargs"]) == 2


def test_worker_initializer_resolves_reference_before_constructing_client(
    monkeypatch,
) -> None:
    process_parallel = _load_vendored_process_parallel()
    literal_key = "inherited-fake-provider-key"
    config = process_parallel.Config()
    model = LLMModelConfig(name="test-model", api_key=literal_key)
    config.llm.api_key = literal_key
    config.llm.models = [model]
    config.llm.evaluator_models = [model]
    database = ProgramDatabase(config.database)
    controller = process_parallel.ProcessParallelController(
        config,
        str(ROOT / "agents" / "openevolve_generic" / "evaluator.py"),
        database,
    )
    monkeypatch.setenv("DISCOVERY_API_KEY", literal_key)
    serialized = controller._serialize_config(config)
    assert serialized["llm"]["api_key"] == "${DISCOVERY_API_KEY}"
    assert serialized["llm"]["models"][0]["api_key"] == "${DISCOVERY_API_KEY}"

    client_calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            client_calls.append(kwargs)

    from openevolve.llm import openai as openai_module
    from openevolve.llm.ensemble import LLMEnsemble

    monkeypatch.setattr(openai_module.openai, "OpenAI", FakeOpenAI)
    process_parallel._worker_init(
        serialized,
        str(ROOT / "agents" / "openevolve_generic" / "evaluator.py"),
    )
    ensemble = LLMEnsemble(process_parallel._worker_config.llm.models)

    assert ensemble.models[0].api_key == literal_key
    assert client_calls[0]["api_key"] == literal_key
    assert client_calls[0]["api_key"] != "${DISCOVERY_API_KEY}"


@pytest.mark.parametrize(
    "serialized_value",
    ("literal-provider-key", "${OPENAI_API_KEY}", "${DISCOVERY_API_KEY}suffix"),
)
def test_worker_rejects_unexpected_serialized_api_key(
    monkeypatch,
    serialized_value: str,
) -> None:
    process_parallel = _load_vendored_process_parallel()
    monkeypatch.setenv("DISCOVERY_API_KEY", "inherited-fake-provider-key")

    with pytest.raises(ValueError, match="approved DISCOVERY_API_KEY reference"):
        process_parallel._resolve_worker_api_key_reference(serialized_value)


@pytest.mark.parametrize("inherited_value", (None, ""))
def test_worker_rejects_missing_or_empty_inherited_api_key(
    monkeypatch,
    inherited_value: str | None,
) -> None:
    process_parallel = _load_vendored_process_parallel()
    if inherited_value is None:
        monkeypatch.delenv("DISCOVERY_API_KEY", raising=False)
    else:
        monkeypatch.setenv("DISCOVERY_API_KEY", inherited_value)

    with pytest.raises(ValueError, match="must be non-empty"):
        process_parallel._resolve_worker_api_key_reference(
            "${DISCOVERY_API_KEY}"
        )
