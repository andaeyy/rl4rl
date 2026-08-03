from __future__ import annotations

import json
from types import SimpleNamespace

from baselines.no_search import NoSearchRequest
from baselines.provider import OpenAIIndependentProposalBackend


class _FakeCompletions:
    def __init__(self) -> None:
        self.requests = []

    def create(self, **payload):
        self.requests.append(payload)
        return SimpleNamespace(
            choices=(
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="<<<<<<< SEARCH\nVALUE = 1\n=======\nVALUE = 2\n>>>>>>> REPLACE"
                    )
                ),
            ),
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=9),
        )


def test_real_no_search_adapter_receives_only_constant_model_input(tmp_path) -> None:
    completions = _FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = OpenAIIndependentProposalBackend(
        client=client,
        frozen_base_source="VALUE = 1\n",
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
    assert response.candidate_source == "VALUE = 2\n"
    record = json.loads((tmp_path / "no-search-request-1.json").read_text())
    assert record["provider_visible_input"] == model_input
    rendered = json.dumps(record)
    for forbidden in ("parent_ids", "scores", "candidate_history", "transition"):
        assert forbidden not in rendered

