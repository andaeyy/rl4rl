from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from architecture_ir import validate_ir_candidate_json
from baselines.no_search import NoSearchRequest
from baselines.provider import OpenAIIndependentProposalBackend
from study.runtime_adapters import canonicalize_architecture_ir


_INITIAL_PATH = (
    Path(__file__).resolve().parents[1] / "common" / "initial_candidate.ir.json"
)


def _ir(label: str) -> str:
    payload = json.loads(_INITIAL_PATH.read_text(encoding="utf-8"))
    payload["graph_id"] = f"no_search_provider_{label}"
    payload["metadata"]["mechanism_hypothesis"] = f"provider mechanism {label}"
    return canonicalize_architecture_ir(
        json.dumps(payload),
        require_hypothesis=True,
    )


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self.requests = []
        self.content = content

    def create(self, **payload):
        self.requests.append(payload)
        return SimpleNamespace(
            choices=(
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.content
                    )
                ),
            ),
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=9),
        )


def test_real_no_search_adapter_receives_only_constant_model_input(tmp_path) -> None:
    base_ir = _ir("base")
    proposal_ir = _ir("proposal")
    completions = _FakeCompletions(proposal_ir)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = OpenAIIndependentProposalBackend(
        client=client,
        frozen_base_source=base_ir,
        request_log_root=tmp_path,
    )
    model_input = {
        "model": "gpt-5.6-sol",
        "messages": [{"role": "user", "content": "fixed task"}],
        "reasoning_effort": "high",
        "max_completion_tokens": 128,
    }
    response = backend.complete(
        NoSearchRequest(request_id="no-search-request-1", model_input=model_input)
    )

    assert completions.requests == [model_input]
    assert response.candidate_source == proposal_ir
    assert validate_ir_candidate_json(response.candidate_source).valid
    record = json.loads((tmp_path / "no-search-request-1.json").read_text())
    assert record["provider_visible_input"] == model_input
    rendered = json.dumps(record)
    for forbidden in ("parent_ids", "scores", "candidate_history", "transition"):
        assert forbidden not in rendered


def test_real_no_search_adapter_rejects_python_diff_output(tmp_path) -> None:
    completions = _FakeCompletions(
        "<<<<<<< SEARCH\nVALUE = 1\n=======\nVALUE = 2\n>>>>>>> REPLACE"
    )
    backend = OpenAIIndependentProposalBackend(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        frozen_base_source=_ir("invalid-response-base"),
        request_log_root=tmp_path,
    )

    response = backend.complete(
        NoSearchRequest(
            request_id="no-search-invalid-response",
            model_input={"messages": []},
        )
    )

    assert response.candidate_source is None
