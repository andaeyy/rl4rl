"""One reproducible GPT-5.6 Sol generation profile for every controller."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit


TARGET_MODEL = "gpt-5.6-sol"
API_MODE = "chat_completions"
OFFICIAL_OPENAI_API_BASE = "https://api.openai.com/v1"
SUPPORTED_REASONING_EFFORTS = frozenset(
    {"none", "low", "medium", "high", "xhigh", "max"}
)


def _optional_value(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


@dataclass(frozen=True)
class ProviderEndpoint:
    """Normalized, non-secret provider identity suitable for run manifests."""

    base_url: str
    provider_identity: str

    def manifest_fields(self) -> dict[str, str]:
        return {
            "provider_identity": self.provider_identity,
            "api_endpoint": self.base_url,
        }


def resolve_provider_endpoint(
    api_base: str,
    *,
    scientific: bool,
) -> ProviderEndpoint:
    value = api_base.strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("DISCOVERY_API_BASE must be an absolute HTTP(S) endpoint")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "DISCOVERY_API_BASE may not contain credentials, query parameters, or fragments"
        )
    netloc = parsed.hostname.lower()
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    path = parsed.path.rstrip("/")
    normalized = urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))
    if scientific and normalized != OFFICIAL_OPENAI_API_BASE:
        raise ValueError(
            "scientific controller runs are pinned to the official OpenAI API "
            f"endpoint {OFFICIAL_OPENAI_API_BASE!r}"
        )
    identity = (
        "openai_official"
        if normalized == OFFICIAL_OPENAI_API_BASE
        else "openai_compatible_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    )
    return ProviderEndpoint(base_url=normalized, provider_identity=identity)


@dataclass(frozen=True)
class GPT56SolProfile:
    """Resolved request settings shared by greedy and OpenEvolve runs."""

    model: str
    reasoning_effort: str
    max_completion_tokens: int
    timeout_seconds: int
    retries: int
    retry_delay_seconds: int
    seed: int
    environment_overrides_allowed: bool = True

    @classmethod
    def resolve(
        cls,
        *,
        model: str,
        seed: int,
        default_reasoning_effort: str,
        default_max_completion_tokens: int,
        default_timeout_seconds: int,
        default_retries: int,
        default_retry_delay_seconds: int,
        environ: Mapping[str, str] | None = None,
        allow_environment_overrides: bool = True,
    ) -> GPT56SolProfile:
        env = os.environ if environ is None else environ
        if model != TARGET_MODEL:
            raise ValueError(
                "This controlled study is pinned to "
                f"{TARGET_MODEL!r}, but DISCOVERY_MODEL is {model!r}. "
                f'Run: export DISCOVERY_MODEL="{TARGET_MODEL}"'
            )

        reasoning_effort = (
            _optional_value(env, "DISCOVERY_REASONING_EFFORT")
            or default_reasoning_effort
        ).lower()
        if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
            choices = ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
            raise ValueError(
                f"unsupported DISCOVERY_REASONING_EFFORT={reasoning_effort!r}; "
                f"choose one of: {choices}"
            )

        canonical_limit = _optional_value(
            env, "DISCOVERY_MAX_COMPLETION_TOKENS"
        )
        legacy_limit = _optional_value(env, "DISCOVERY_MAX_TOKENS")
        if legacy_limit:
            raise ValueError(
                "DISCOVERY_MAX_TOKENS is a legacy GPT-4-style override; unset it "
                "and use DISCOVERY_MAX_COMPLETION_TOKENS"
            )
        token_text = (
            canonical_limit
            or str(default_max_completion_tokens)
        )
        timeout_text = (
            _optional_value(env, "DISCOVERY_REQUEST_TIMEOUT_SECONDS")
            or str(default_timeout_seconds)
        )
        retries_text = (
            _optional_value(env, "DISCOVERY_REQUEST_RETRIES")
            or str(default_retries)
        )
        retry_delay_text = (
            _optional_value(env, "DISCOVERY_RETRY_DELAY_SECONDS")
            or str(default_retry_delay_seconds)
        )
        try:
            max_completion_tokens = int(token_text)
            timeout_seconds = int(timeout_text)
            retries = int(retries_text)
            retry_delay_seconds = int(retry_delay_text)
        except ValueError as exc:
            raise ValueError(
                "completion-token, timeout, and retry overrides must be integers"
            ) from exc
        if max_completion_tokens <= 0:
            raise ValueError("max completion tokens must be positive")
        if timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if retries < 0:
            raise ValueError("request retries cannot be negative")
        if retry_delay_seconds < 0:
            raise ValueError("retry delay cannot be negative")
        if not allow_environment_overrides:
            frozen_mismatches = []
            if reasoning_effort != default_reasoning_effort.lower():
                frozen_mismatches.append("DISCOVERY_REASONING_EFFORT")
            if max_completion_tokens != default_max_completion_tokens:
                frozen_mismatches.append("DISCOVERY_MAX_COMPLETION_TOKENS")
            if timeout_seconds != default_timeout_seconds:
                frozen_mismatches.append("DISCOVERY_REQUEST_TIMEOUT_SECONDS")
            if retries != default_retries:
                frozen_mismatches.append("DISCOVERY_REQUEST_RETRIES")
            if retry_delay_seconds != default_retry_delay_seconds:
                frozen_mismatches.append("DISCOVERY_RETRY_DELAY_SECONDS")
            if frozen_mismatches:
                raise ValueError(
                    "scientific generation settings are frozen; conflicting "
                    "environment values: " + ", ".join(frozen_mismatches)
                )

        return cls(
            model=model,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            seed=seed,
            environment_overrides_allowed=allow_environment_overrides,
        )

    def chat_completion_request(
        self, messages: Sequence[Mapping[str, str]]
    ) -> dict[str, Any]:
        """Build a GPT-5.6 Chat Completions request without sampling controls."""

        return {
            "model": self.model,
            "messages": list(messages),
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_completion_tokens,
            "seed": self.seed,
        }

    def manifest_fields(self) -> dict[str, Any]:
        """Return non-secret settings that make a run auditable."""

        return {
            "model": self.model,
            "api_mode": API_MODE,
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_completion_tokens,
            "request_timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "temperature": None,
            "top_p": None,
            "request_seed": self.seed,
            "generation_seed_support": "best_effort_api_seed",
            "request_settings_source": (
                "environment_overrides_permitted"
                if self.environment_overrides_allowed
                else "frozen_controller_configuration"
            ),
        }
