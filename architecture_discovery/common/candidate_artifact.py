"""Trusted dispatch for declarative IR and legacy Python candidate artifacts.

Provider-backed discovery uses architecture IR.  Legacy Python candidates remain
available for checked-in regression fixtures, but scientific Python execution is
still governed by the separate OS-containment policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch.nn as nn

from architecture_ir import ArchitectureGraph, RuntimeBindings
from architecture_ir.interpreter import validate_ir_candidate_path
from common.candidate_contract import ContractResult, inspect_candidate_path, validate_candidate
from common.candidate_loader import load_candidate
from containment.policy import CandidateFormat


IR_SUFFIXES = frozenset({".json", ".ir"})
PYTHON_SUFFIXES = frozenset({".py"})


@dataclass(frozen=True)
class CandidateInspection:
    valid: bool
    candidate_format: CandidateFormat
    reasons: tuple[str, ...]
    graph_hash: str | None = None
    architecture_hash: str | None = None


@dataclass(frozen=True)
class CandidateBuild:
    model: nn.Module
    metadata: dict[str, Any]
    candidate_format: CandidateFormat
    module: ModuleType | None = None
    graph: ArchitectureGraph | None = None
    runtime_bindings: RuntimeBindings | None = None


def candidate_format_for_path(path: str | Path) -> CandidateFormat:
    suffix = Path(path).suffix.lower()
    if suffix in IR_SUFFIXES:
        return CandidateFormat.ARCHITECTURE_IR
    if suffix in PYTHON_SUFFIXES:
        return CandidateFormat.ARBITRARY_PYTHON
    raise ValueError(
        "candidate artifact must use a declarative IR (.json/.ir) or legacy "
        "Python (.py) suffix"
    )


def inspect_candidate_artifact(path: str | Path) -> CandidateInspection:
    candidate = Path(path)
    if not candidate.is_file():
        return CandidateInspection(
            False,
            CandidateFormat.ARCHITECTURE_IR,
            (f"candidate artifact does not exist: {candidate}",),
        )
    try:
        candidate_format = candidate_format_for_path(candidate)
    except ValueError as error:
        return CandidateInspection(
            False,
            CandidateFormat.ARCHITECTURE_IR,
            (str(error),),
        )
    if candidate_format is CandidateFormat.ARBITRARY_PYTHON:
        contract = inspect_candidate_path(candidate)
        return CandidateInspection(
            contract.valid,
            candidate_format,
            contract.reasons,
        )
    validation = validate_ir_candidate_path(candidate)
    return CandidateInspection(
        validation.valid,
        candidate_format,
        tuple(
            f"architecture IR [{issue.code}]: {issue.message}"
            for issue in validation.issues
        ),
        validation.graph_hash,
        validation.architecture_hash,
    )


def build_candidate_artifact(path: str | Path, *, seed: int) -> CandidateBuild:
    candidate = Path(path).resolve()
    inspection = inspect_candidate_artifact(candidate)
    if not inspection.valid:
        raise ValueError("candidate contract failed: " + "; ".join(inspection.reasons))

    if inspection.candidate_format is CandidateFormat.ARCHITECTURE_IR:
        from architecture_ir.interpreter import load_and_build_ir_candidate

        interpreted = load_and_build_ir_candidate(candidate, seed)
        return CandidateBuild(
            model=interpreted.model,
            metadata=dict(interpreted.metadata),
            candidate_format=inspection.candidate_format,
            graph=interpreted.graph,
            runtime_bindings=interpreted.bindings,
        )

    module = load_candidate(candidate)
    built = module.build_untrained_model(seed)
    if not isinstance(built, tuple) or len(built) != 2:
        raise TypeError(
            "build_untrained_model(seed) must return (torch.nn.Module, metadata)"
        )
    model, metadata = built
    contract: ContractResult = validate_candidate(module, model)
    if not contract.valid:
        raise ValueError(
            "candidate runtime contract failed: " + "; ".join(contract.reasons)
        )
    if not isinstance(metadata, dict):
        raise TypeError("candidate metadata must be a dictionary")
    return CandidateBuild(
        model=model,
        metadata=dict(metadata),
        candidate_format=inspection.candidate_format,
        module=module,
    )
