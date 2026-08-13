"""Real-provider boundary for independent no-search proposals.

The backend receives only ``NoSearchRequest``: a constant provider-visible
payload plus an idempotency/logging ID. It never receives parents, candidate
history, scores, transition state, or evaluation feedback. Importing this
module makes no request.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, ClassVar

from baselines.no_search import BackendResponse, NoSearchRequest
from study.runtime_adapters import (
    ArchitectureIRProposalError,
    canonicalize_architecture_ir,
)
from study.serialization import content_hash


class OpenAIIndependentProposalBackend:
    """Adapt an injected OpenAI-compatible client without adding retries."""

    is_test_double: ClassVar[bool] = False

    def __init__(
        self,
        *,
        client: Any,
        frozen_base_source: str,
        request_log_root: str | Path,
    ) -> None:
        self.client = client
        self.frozen_base_source = canonicalize_architecture_ir(
            frozen_base_source,
            require_hypothesis=False,
        )
        self.request_log_root = Path(request_log_root).resolve()
        self.request_log_root.mkdir(parents=True, exist_ok=True)

    def complete(self, request: NoSearchRequest) -> BackendResponse:
        response = self.client.chat.completions.create(**request.model_input)
        response_text = response.choices[0].message.content or ""
        candidate_source = None
        try:
            candidate_source = canonicalize_architecture_ir(
                response_text,
                require_hypothesis=True,
            )
        except ArchitectureIRProposalError:
            candidate_source = None
        usage = response.usage
        record = {
            "schema_name": "NoSearchProviderRecord",
            "schema_version": "1.0",
            "request_id": request.request_id,
            "provider_visible_input": request.model_input,
            "provider_visible_input_hash": content_hash(request.model_input),
            "response_text": response_text,
            "candidate_source_hash": (
                None if candidate_source is None else content_hash(candidate_source)
            ),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }
        destination = self.request_log_root / f"{request.request_id}.json"
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return BackendResponse(
            response_text=response_text,
            candidate_source=candidate_source,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
