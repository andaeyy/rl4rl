"""Append-only, credential-free accounting for actual provider API attempts.

The ledger deliberately stores no prompts, completions, request bodies, error
messages, headers, or credentials.  One terminal JSONL record is appended for
each invocation of the official client transport, including invocations that
raise.  The file itself is created once with ``O_EXCL`` and is only reopened
with ``O_APPEND``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from common.gpt56_sol import OFFICIAL_OPENAI_API_BASE, TARGET_MODEL
from common.runtime_context import ExecutionContextV1

PROVIDER_ATTEMPT_LEDGER_FILENAME = "provider_attempts.jsonl"
PROVIDER_ATTEMPT_LEDGER_ENV = "DISCOVERY_PROVIDER_ATTEMPT_LEDGER"
PROVIDER_ATTEMPT_HARNESS_ENV = "DISCOVERY_PROVIDER_HARNESS"
PROVIDER_ATTEMPT_ACTION_ENV = "DISCOVERY_PROVIDER_ACTION"
PROVIDER_ATTEMPT_SCHEMA = "ProviderAttemptRecord/1.0"

_EXECUTION_CONTEXT_ENV = "DISCOVERY_EXECUTION_CONTEXT_JSON"
_SAFE_LOGICAL_NAME = re.compile(r"\A[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")
_SAFE_RUN_ID = re.compile(r"\A[a-zA-Z0-9][a-zA-Z0-9_.-]{0,255}\Z")
_SAFE_PROVIDER_ID = re.compile(r"\A[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,255}\Z")
_SAFE_ERROR_CLASS = re.compile(r"\A[a-zA-Z_][a-zA-Z0-9_.]{0,127}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_MAX_LEDGER_BYTES = 4 * 1024 * 1024
_MAX_LEDGER_RECORDS = 10_000
_DARWIN_CANONICAL_DIRECTORY_ALIASES = (
    (Path("/var"), Path("/private/var")),
    (Path("/tmp"), Path("/private/tmp")),
)

_T = TypeVar("_T")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"provider attempt {field} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(
            f"provider attempt {field} must be a UTC timestamp"
        ) from error
    if parsed.tzinfo != UTC:
        raise ValueError(f"provider attempt {field} must be a UTC timestamp")
    return parsed


def _safe_optional_provider_id(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SAFE_PROVIDER_ID.fullmatch(value) is None:
        raise ValueError(f"provider attempt {field} is not a safe provider ID")
    return value


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generation_settings_sha256(request: Mapping[str, Any]) -> str:
    """Hash the exact non-message request settings sent to the SDK.

    Message content is intentionally excluded because it can contain source or
    provider output and is not needed to reconcile the generation contract.
    Credentials cannot be part of a Chat Completions request mapping, but they
    are rejected explicitly as a defense-in-depth invariant.
    """

    if not isinstance(request, Mapping):
        raise TypeError("provider request must be a mapping")
    forbidden = {
        "api_key",
        "authorization",
        "credential",
        "password",
        "secret",
        "token",
    }
    if any(str(key).lower() in forbidden for key in request):
        raise ValueError("provider request settings contain a credential field")
    settings = {
        str(key): value for key, value in request.items() if str(key) != "messages"
    }
    return _canonical_sha256(settings)


def _validated_ledger_path(
    path: str | Path,
    *,
    allow_missing_leaf: bool,
) -> Path:
    """Return an absolute path only when no existing component is a symlink."""

    raw = os.fspath(path)
    if not raw or "\x00" in raw:
        raise ValueError("provider attempt ledger path is invalid")
    selected = Path(os.path.abspath(raw))
    # macOS exposes two fixed root aliases used by tempfile. Canonicalize only
    # those OS-owned aliases, then operate on the canonical path; arbitrary
    # caller-controlled symlink components remain forbidden below.
    for alias, canonical in _DARWIN_CANONICAL_DIRECTORY_ALIASES:
        if selected != alias and alias not in selected.parents:
            continue
        try:
            alias_details = os.lstat(alias)
        except FileNotFoundError:
            continue
        if (
            stat.S_ISLNK(alias_details.st_mode)
            and alias.resolve(strict=True) == canonical
        ):
            selected = canonical / selected.relative_to(alias)
            break
    current = Path(selected.anchor)
    parts = selected.parts[1:]
    for index, component in enumerate(parts):
        current /= component
        is_leaf = index == len(parts) - 1
        try:
            details = os.lstat(current)
        except FileNotFoundError:
            if is_leaf and allow_missing_leaf:
                return selected
            raise
        if stat.S_ISLNK(details.st_mode):
            raise ValueError(
                "provider attempt ledger path components may not be symlinks"
            )
        if not is_leaf and not stat.S_ISDIR(details.st_mode):
            raise ValueError(
                "provider attempt ledger ancestor must be a directory"
            )
    return selected


def _descriptor_identity(
    descriptor: int,
    *,
    expected: tuple[int, int] | None = None,
) -> tuple[int, int]:
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise ValueError("provider attempt ledger must be one regular file")
    identity = (details.st_dev, details.st_ino)
    if expected is not None and identity != expected:
        raise ValueError("provider attempt ledger inode changed after binding")
    return identity


@dataclass(frozen=True)
class ProviderAttemptContext:
    """Stable run context repeated on every attempt for standalone auditing."""

    harness: str
    action: str
    controller_run_id: str
    execution_backend: str
    action_run_id: str
    modal_call_id: str | None
    api_endpoint: str
    model: str

    def __post_init__(self) -> None:
        for value, field in (
            (self.harness, "harness"),
            (self.action, "action"),
        ):
            if _SAFE_LOGICAL_NAME.fullmatch(value) is None:
                raise ValueError(f"provider attempt {field} is not a safe name")
        for value, field in (
            (self.controller_run_id, "controller_run_id"),
            (self.action_run_id, "action_run_id"),
        ):
            if _SAFE_RUN_ID.fullmatch(value) is None:
                raise ValueError(f"provider attempt {field} is not a safe run ID")
        if self.execution_backend not in {"local", "modal"}:
            raise ValueError("provider attempt execution backend is invalid")
        _safe_optional_provider_id(self.modal_call_id, "modal_call_id")
        if self.execution_backend == "modal" and self.modal_call_id is None:
            raise ValueError("Modal provider attempts require a call ID")
        if self.execution_backend == "local" and self.modal_call_id is not None:
            raise ValueError("local provider attempts may not contain a Modal call ID")
        if self.api_endpoint != OFFICIAL_OPENAI_API_BASE:
            raise ValueError("provider attempts require the official OpenAI endpoint")
        if self.model != TARGET_MODEL:
            raise ValueError("provider attempts require the frozen target model")

    @classmethod
    def build(
        cls,
        *,
        harness: str,
        action: str,
        controller_run_id: str,
        api_endpoint: str,
        model: str,
        environ: Mapping[str, str] | None = None,
    ) -> ProviderAttemptContext:
        environment = os.environ if environ is None else environ
        encoded_context = environment.get(_EXECUTION_CONTEXT_ENV)
        if encoded_context:
            try:
                raw_context = json.loads(encoded_context)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "provider execution context is not valid JSON"
                ) from error
            if not isinstance(raw_context, dict):
                raise ValueError("provider execution context must be an object")
            execution = ExecutionContextV1.from_dict(raw_context)
            backend = execution.execution_backend
            action_run_id = execution.run_id
            modal_call_id = execution.modal_call_id
        else:
            backend = "local"
            action_run_id = controller_run_id
            modal_call_id = None
        return cls(
            harness=harness,
            action=action,
            controller_run_id=controller_run_id,
            execution_backend=backend,
            action_run_id=action_run_id,
            modal_call_id=modal_call_id,
            api_endpoint=api_endpoint,
            model=model,
        )


@dataclass(frozen=True)
class ProviderAttemptRecord:
    schema_name: str
    schema_version: str
    harness: str
    action: str
    controller_run_id: str
    execution_backend: str
    action_run_id: str
    modal_call_id: str | None
    attempt_ordinal: int
    started_at_utc: str
    ended_at_utc: str
    status: str
    api_endpoint: str
    model: str
    generation_settings_sha256: str
    provider_response_id: str | None
    provider_request_id: str | None
    usage_known: bool
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    error_class: str | None

    SCHEMA_NAME: ClassVar[str] = "ProviderAttemptRecord"
    SCHEMA_VERSION: ClassVar[str] = "1.0"
    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_name",
            "schema_version",
            "harness",
            "action",
            "controller_run_id",
            "execution_backend",
            "action_run_id",
            "modal_call_id",
            "attempt_ordinal",
            "started_at_utc",
            "ended_at_utc",
            "status",
            "api_endpoint",
            "model",
            "generation_settings_sha256",
            "provider_response_id",
            "provider_request_id",
            "usage_known",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "error_class",
        }
    )

    def __post_init__(self) -> None:
        if self.schema_name != self.SCHEMA_NAME:
            raise ValueError("expected ProviderAttemptRecord schema")
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError("unsupported provider attempt schema version")
        ProviderAttemptContext(
            harness=self.harness,
            action=self.action,
            controller_run_id=self.controller_run_id,
            execution_backend=self.execution_backend,
            action_run_id=self.action_run_id,
            modal_call_id=self.modal_call_id,
            api_endpoint=self.api_endpoint,
            model=self.model,
        )
        if (
            isinstance(self.attempt_ordinal, bool)
            or not isinstance(self.attempt_ordinal, int)
            or self.attempt_ordinal <= 0
        ):
            raise ValueError("provider attempt ordinal must be a positive integer")
        started = _parse_utc(self.started_at_utc, "started_at_utc")
        ended = _parse_utc(self.ended_at_utc, "ended_at_utc")
        if ended < started:
            raise ValueError("provider attempt ended before it started")
        if self.status not in {"success", "error"}:
            raise ValueError("provider attempt status is invalid")
        if _SHA256.fullmatch(self.generation_settings_sha256) is None:
            raise ValueError("provider generation settings digest is invalid")
        _safe_optional_provider_id(
            self.provider_response_id, "provider_response_id"
        )
        _safe_optional_provider_id(self.provider_request_id, "provider_request_id")
        if type(self.usage_known) is not bool:
            raise ValueError("provider attempt usage_known must be boolean")
        tokens = (self.input_tokens, self.output_tokens, self.total_tokens)
        if self.usage_known:
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in tokens
            ):
                raise ValueError("known provider usage requires nonnegative integers")
            if self.total_tokens != self.input_tokens + self.output_tokens:
                raise ValueError("provider token totals do not reconcile")
        elif any(value is not None for value in tokens):
            raise ValueError("unknown provider usage requires null token counts")
        if self.status == "success":
            if self.error_class is not None:
                raise ValueError(
                    "successful provider attempts may not contain an error"
                )
        else:
            if (
                not isinstance(self.error_class, str)
                or _SAFE_ERROR_CLASS.fullmatch(self.error_class) is None
            ):
                raise ValueError("failed provider attempt has an unsafe error class")
            if self.provider_response_id is not None:
                raise ValueError(
                    "failed provider attempts may not contain a response ID"
                )
            if self.usage_known:
                raise ValueError("failed provider attempts may not contain token usage")

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in sorted(self.FIELDS)
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProviderAttemptRecord:
        if not isinstance(payload, Mapping) or set(payload) != cls.FIELDS:
            raise ValueError("provider attempt has unexpected or missing fields")
        for field in (
            "schema_name",
            "schema_version",
            "harness",
            "action",
            "controller_run_id",
            "execution_backend",
            "action_run_id",
            "started_at_utc",
            "ended_at_utc",
            "status",
            "api_endpoint",
            "model",
            "generation_settings_sha256",
        ):
            if not isinstance(payload[field], str):
                raise ValueError(f"provider attempt {field} must be text")
        return cls(**{field: payload[field] for field in cls.FIELDS})


def _usage(response: object) -> tuple[bool, int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    values = (input_tokens, output_tokens, total_tokens)
    known = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in values
    ) and total_tokens == input_tokens + output_tokens
    if not known:
        return False, None, None, None
    return True, input_tokens, output_tokens, total_tokens


def _request_id(value: object) -> str | None:
    for attribute in ("_request_id", "request_id"):
        candidate = getattr(value, attribute, None)
        if isinstance(candidate, str) and _SAFE_PROVIDER_ID.fullmatch(candidate):
            return candidate
    return None


def _response_id(response: object) -> str | None:
    value = getattr(response, "id", None)
    if isinstance(value, str) and _SAFE_PROVIDER_ID.fullmatch(value):
        return value
    return None


def _read_records_from_descriptor(descriptor: int) -> list[ProviderAttemptRecord]:
    _descriptor_identity(descriptor)
    details = os.fstat(descriptor)
    if details.st_size > _MAX_LEDGER_BYTES:
        raise ValueError("provider attempt ledger exceeds its size limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    payload = bytearray()
    while len(payload) <= _MAX_LEDGER_BYTES:
        chunk = os.read(
            descriptor,
            min(64 * 1024, _MAX_LEDGER_BYTES + 1 - len(payload)),
        )
        if not chunk:
            break
        payload.extend(chunk)
    if len(payload) > _MAX_LEDGER_BYTES:
        raise ValueError("provider attempt ledger exceeds its size limit")
    try:
        text = bytes(payload).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("provider attempt ledger is not UTF-8") from error
    if text and not text.endswith("\n"):
        raise ValueError("provider attempt ledger has a truncated final record")
    records: list[ProviderAttemptRecord] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError("provider attempt ledger contains a blank record")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"provider attempt ledger line {line_number} is invalid JSON"
            ) from error
        if not isinstance(raw, dict):
            raise ValueError("provider attempt ledger record must be an object")
        records.append(ProviderAttemptRecord.from_dict(raw))
        if len(records) > _MAX_LEDGER_RECORDS:
            raise ValueError("provider attempt ledger has too many records")
    for expected, record in enumerate(records, start=1):
        if record.attempt_ordinal != expected:
            raise ValueError("provider attempt ordinals are not contiguous")
        if records and (
            record.harness,
            record.action,
            record.controller_run_id,
            record.execution_backend,
            record.action_run_id,
            record.modal_call_id,
            record.api_endpoint,
            record.model,
        ) != (
            records[0].harness,
            records[0].action,
            records[0].controller_run_id,
            records[0].execution_backend,
            records[0].action_run_id,
            records[0].modal_call_id,
            records[0].api_endpoint,
            records[0].model,
        ):
            raise ValueError("provider attempt run context changes within the ledger")
    return records


class ProviderAttemptLedger:
    """Create-once, append-only writer around an actual provider operation."""

    def __init__(
        self,
        path: Path,
        context: ProviderAttemptContext,
        identity: tuple[int, int],
    ) -> None:
        self.path = path
        self.context = context
        self._identity = identity

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        harness: str,
        action: str,
        controller_run_id: str,
        api_endpoint: str,
        model: str,
        environ: Mapping[str, str] | None = None,
    ) -> ProviderAttemptLedger:
        selected = _validated_ledger_path(path, allow_missing_leaf=True)
        context = ProviderAttemptContext.build(
            harness=harness,
            action=action,
            controller_run_id=controller_run_id,
            api_endpoint=api_endpoint,
            model=model,
            environ=environ,
        )
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(selected, flags, 0o600)
        try:
            identity = _descriptor_identity(descriptor)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return cls(selected, context, identity)

    @classmethod
    def open_existing_from_environment(
        cls,
        *,
        api_endpoint: str,
        model: str,
        environ: Mapping[str, str] | None = None,
    ) -> ProviderAttemptLedger | None:
        environment = os.environ if environ is None else environ
        path = environment.get(PROVIDER_ATTEMPT_LEDGER_ENV)
        if not path:
            return None
        harness = environment.get(PROVIDER_ATTEMPT_HARNESS_ENV)
        action = environment.get(PROVIDER_ATTEMPT_ACTION_ENV)
        controller_run_id = environment.get("DISCOVERY_RUN_ID")
        if not harness or not action or not controller_run_id:
            raise ValueError("provider attempt ledger environment is incomplete")
        selected = _validated_ledger_path(path, allow_missing_leaf=False)
        context = ProviderAttemptContext.build(
            harness=harness,
            action=action,
            controller_run_id=controller_run_id,
            api_endpoint=api_endpoint,
            model=model,
            environ=environment,
        )
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(selected, flags)
        try:
            records = _read_records_from_descriptor(descriptor)
            identity = _descriptor_identity(descriptor)
        finally:
            os.close(descriptor)
        for record in records:
            if (
                record.harness,
                record.action,
                record.controller_run_id,
                record.execution_backend,
                record.action_run_id,
                record.modal_call_id,
                record.api_endpoint,
                record.model,
            ) != (
                context.harness,
                context.action,
                context.controller_run_id,
                context.execution_backend,
                context.action_run_id,
                context.modal_call_id,
                context.api_endpoint,
                context.model,
            ):
                raise ValueError("provider attempt ledger context does not match")
        return cls(selected, context, identity)

    def record_call(
        self,
        request: Mapping[str, Any],
        operation: Callable[[], _T],
    ) -> _T:
        settings_sha256 = generation_settings_sha256(request)
        selected = _validated_ledger_path(self.path, allow_missing_leaf=False)
        if selected != self.path:
            raise ValueError("provider attempt ledger path changed after binding")
        flags = os.O_RDWR | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            _descriptor_identity(descriptor, expected=self._identity)
            records = _read_records_from_descriptor(descriptor)
            ordinal = len(records) + 1
            started = _utc_now()
            try:
                response = operation()
            except BaseException as error:
                record = self._record(
                    ordinal=ordinal,
                    started=started,
                    status="error",
                    settings_sha256=settings_sha256,
                    response_id=None,
                    request_id=_request_id(error),
                    usage=(False, None, None, None),
                    error_class=type(error).__name__,
                )
                self._append(descriptor, record, expected_identity=self._identity)
                raise
            record = self._record(
                ordinal=ordinal,
                started=started,
                status="success",
                settings_sha256=settings_sha256,
                response_id=_response_id(response),
                request_id=_request_id(response),
                usage=_usage(response),
                error_class=None,
            )
            self._append(descriptor, record, expected_identity=self._identity)
            return response
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _record(
        self,
        *,
        ordinal: int,
        started: str,
        status: str,
        settings_sha256: str,
        response_id: str | None,
        request_id: str | None,
        usage: tuple[bool, int | None, int | None, int | None],
        error_class: str | None,
    ) -> ProviderAttemptRecord:
        usage_known, input_tokens, output_tokens, total_tokens = usage
        return ProviderAttemptRecord(
            schema_name=ProviderAttemptRecord.SCHEMA_NAME,
            schema_version=ProviderAttemptRecord.SCHEMA_VERSION,
            harness=self.context.harness,
            action=self.context.action,
            controller_run_id=self.context.controller_run_id,
            execution_backend=self.context.execution_backend,
            action_run_id=self.context.action_run_id,
            modal_call_id=self.context.modal_call_id,
            attempt_ordinal=ordinal,
            started_at_utc=started,
            ended_at_utc=_utc_now(),
            status=status,
            api_endpoint=self.context.api_endpoint,
            model=self.context.model,
            generation_settings_sha256=settings_sha256,
            provider_response_id=response_id,
            provider_request_id=request_id,
            usage_known=usage_known,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            error_class=error_class,
        )

    @staticmethod
    def _append(
        descriptor: int,
        record: ProviderAttemptRecord,
        *,
        expected_identity: tuple[int, int],
    ) -> None:
        _descriptor_identity(descriptor, expected=expected_identity)
        payload = (
            json.dumps(
                record.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("provider attempt ledger append was incomplete")
        os.fsync(descriptor)


def load_provider_attempt_ledger(
    path: str | Path,
) -> tuple[ProviderAttemptRecord, ...]:
    """Strictly load a bounded ledger without changing it."""

    selected = _validated_ledger_path(path, allow_missing_leaf=False)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(selected, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        return tuple(_read_records_from_descriptor(descriptor))
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def provider_attempt_totals(
    records: Sequence[ProviderAttemptRecord],
) -> dict[str, int]:
    """Return mechanically reconcilable counts and known token totals."""

    success_count = sum(record.status == "success" for record in records)
    error_count = sum(record.status == "error" for record in records)
    usage_known_count = sum(record.usage_known for record in records)
    input_tokens = sum(record.input_tokens or 0 for record in records)
    output_tokens = sum(record.output_tokens or 0 for record in records)
    total_tokens = sum(record.total_tokens or 0 for record in records)
    if total_tokens != input_tokens + output_tokens:
        raise ValueError("provider ledger aggregate token totals do not reconcile")
    return {
        "attempt_count": len(records),
        "success_count": success_count,
        "error_count": error_count,
        "usage_known_count": usage_known_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
