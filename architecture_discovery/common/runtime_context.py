"""Strict, credential-free execution provenance for local and remote runs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar
from urllib.parse import urlsplit

_SAFE_NAME = re.compile(r"\A[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")
_SAFE_RUN_ID = re.compile(r"\A[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_SAFE_MODAL_ID = re.compile(r"\A[a-z]{1,8}-[a-zA-Z0-9_-]{1,120}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty text or null")
    return value


def _safe_name(value: str | None, field_name: str) -> None:
    if value is not None and _SAFE_NAME.fullmatch(value) is None:
        raise ValueError(f"{field_name} is not a safe logical name")


def _safe_modal_id(value: str | None, field_name: str) -> None:
    if value is not None and _SAFE_MODAL_ID.fullmatch(value) is None:
        raise ValueError(f"{field_name} is not a valid Modal object identifier")


def _safe_artifact_uri(value: str | None) -> None:
    if value is None:
        return
    parsed = urlsplit(value)
    if parsed.scheme not in {"local", "volume"}:
        raise ValueError("artifact_uri must use the local or volume scheme")
    if not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("artifact_uri must name a store and contain no credentials")
    if _SAFE_NAME.fullmatch(parsed.netloc) is None:
        raise ValueError("artifact_uri store name is unsafe")
    if parsed.query or parsed.fragment or "\\" in parsed.path or "\x00" in parsed.path:
        raise ValueError("artifact_uri contains forbidden components")
    parts = tuple(part for part in parsed.path.split("/") if part)
    if any(part in {".", ".."} for part in parts):
        raise ValueError("artifact_uri may not traverse directories")
    if not all(_SAFE_NAME.fullmatch(part) for part in parts):
        raise ValueError("artifact_uri contains an unsafe path component")


@dataclass(frozen=True)
class ExecutionContextV1:
    """An exact, non-extensible record of where one trusted action ran.

    Values are deliberately limited to identifiers, a digest, and a credential-free
    artifact URI. Arbitrary environment metadata does not belong in this record.
    """

    execution_backend: str
    run_id: str
    app_name: str | None
    function_name: str | None
    modal_app_id: str | None
    modal_function_id: str | None
    modal_call_id: str | None
    modal_image_id: str | None
    image_source_sha256: str | None
    artifact_uri: str | None

    SCHEMA_NAME: ClassVar[str] = "ExecutionContext"
    SCHEMA_VERSION: ClassVar[str] = "1.0"
    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_name",
            "schema_version",
            "execution_backend",
            "run_id",
            "app_name",
            "function_name",
            "modal_app_id",
            "modal_function_id",
            "modal_call_id",
            "modal_image_id",
            "image_source_sha256",
            "artifact_uri",
        }
    )

    def __post_init__(self) -> None:
        if self.execution_backend not in {"local", "modal"}:
            raise ValueError("execution_backend must be 'local' or 'modal'")
        if _SAFE_RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id is not a safe run identifier")
        _safe_name(self.app_name, "app_name")
        _safe_name(self.function_name, "function_name")
        _safe_modal_id(self.modal_app_id, "modal_app_id")
        _safe_modal_id(self.modal_function_id, "modal_function_id")
        _safe_modal_id(self.modal_call_id, "modal_call_id")
        _safe_modal_id(self.modal_image_id, "modal_image_id")
        if (
            self.image_source_sha256 is not None
            and _SHA256.fullmatch(self.image_source_sha256) is None
        ):
            raise ValueError("image_source_sha256 must be a lowercase SHA-256")
        _safe_artifact_uri(self.artifact_uri)
        modal_fields = (
            self.modal_app_id,
            self.modal_function_id,
            self.modal_call_id,
            self.modal_image_id,
        )
        if self.execution_backend == "local" and any(modal_fields):
            raise ValueError("local execution contexts may not contain Modal IDs")
        if self.execution_backend == "modal":
            if self.app_name is None or self.function_name is None:
                raise ValueError(
                    "Modal execution requires logical app and function names"
                )
            if self.image_source_sha256 is None:
                raise ValueError("Modal execution requires the image source digest")
            if self.artifact_uri is None or not self.artifact_uri.startswith(
                "volume://"
            ):
                raise ValueError(
                    "Modal execution requires a credential-free volume URI"
                )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "schema_name": self.SCHEMA_NAME,
            "schema_version": self.SCHEMA_VERSION,
            "execution_backend": self.execution_backend,
            "run_id": self.run_id,
            "app_name": self.app_name,
            "function_name": self.function_name,
            "modal_app_id": self.modal_app_id,
            "modal_function_id": self.modal_function_id,
            "modal_call_id": self.modal_call_id,
            "modal_image_id": self.modal_image_id,
            "image_source_sha256": self.image_source_sha256,
            "artifact_uri": self.artifact_uri,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionContextV1:
        if not isinstance(payload, Mapping) or set(payload) != cls.FIELDS:
            raise ValueError("execution context has unexpected or missing fields")
        if payload["schema_name"] != cls.SCHEMA_NAME:
            raise ValueError("expected ExecutionContext schema")
        if payload["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unsupported ExecutionContext schema version")
        backend = payload["execution_backend"]
        run_id = payload["run_id"]
        if not isinstance(backend, str) or not isinstance(run_id, str):
            raise ValueError("execution_backend and run_id must be text")
        return cls(
            execution_backend=backend,
            run_id=run_id,
            app_name=_optional_text(payload["app_name"], "app_name"),
            function_name=_optional_text(payload["function_name"], "function_name"),
            modal_app_id=_optional_text(payload["modal_app_id"], "modal_app_id"),
            modal_function_id=_optional_text(
                payload["modal_function_id"], "modal_function_id"
            ),
            modal_call_id=_optional_text(payload["modal_call_id"], "modal_call_id"),
            modal_image_id=_optional_text(payload["modal_image_id"], "modal_image_id"),
            image_source_sha256=_optional_text(
                payload["image_source_sha256"], "image_source_sha256"
            ),
            artifact_uri=_optional_text(payload["artifact_uri"], "artifact_uri"),
        )

    @classmethod
    def local(
        cls, *, run_id: str, artifact_uri: str | None = None
    ) -> ExecutionContextV1:
        return cls(
            execution_backend="local",
            run_id=run_id,
            app_name=None,
            function_name=None,
            modal_app_id=None,
            modal_function_id=None,
            modal_call_id=None,
            modal_image_id=None,
            image_source_sha256=None,
            artifact_uri=artifact_uri,
        )
