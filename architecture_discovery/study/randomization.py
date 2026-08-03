"""Blocked C0-C3 randomization frozen before any study run executes."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from study.contracts import BlockSpec, ConditionId, ConditionSpec, RunSpec, StudySpec
from study.serialization import (
    content_hash,
    create_json_exclusive,
    read_json,
    require_int,
    require_str,
    stable_id,
)


def _derived_seed(study_seed: int, *parts: object) -> int:
    material = ":".join([str(study_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class RandomizationPlan:
    study_id: str
    study_spec_hash: str
    randomization_seed: int
    blocks: tuple[BlockSpec, ...]
    assignment_hash: str
    schema_name: str = field(default="RandomizationPlan", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        require_str(self.study_id, "study_id")
        require_str(self.study_spec_hash, "study_spec_hash")
        require_str(self.assignment_hash, "assignment_hash")
        require_int(self.randomization_seed, "randomization_seed")

    @property
    def runs(self) -> tuple[RunSpec, ...]:
        return tuple(run for block in self.blocks for run in block.runs)

    def assignment_payload(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "study_spec_hash": self.study_spec_hash,
            "randomization_seed": self.randomization_seed,
            "blocks": [block.to_dict() for block in self.blocks],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            **self.assignment_payload(),
            "assignment_hash": self.assignment_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RandomizationPlan:
        if payload.get("schema_name") != "RandomizationPlan":
            raise ValueError("expected RandomizationPlan schema")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported RandomizationPlan schema version")
        plan = cls(
            study_id=require_str(payload["study_id"], "study_id"),
            study_spec_hash=require_str(
                payload["study_spec_hash"], "study_spec_hash"
            ),
            randomization_seed=require_int(
                payload["randomization_seed"], "randomization_seed"
            ),
            blocks=tuple(BlockSpec.from_dict(block) for block in payload["blocks"]),
            assignment_hash=require_str(
                payload["assignment_hash"], "assignment_hash"
            ),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if len(self.blocks) < 1:
            raise ValueError("randomization plan contains no blocks")
        if [block.block_index for block in self.blocks] != list(
            range(len(self.blocks))
        ):
            raise ValueError("block indices are not contiguous")
        if any(block.study_id != self.study_id for block in self.blocks):
            raise ValueError("block belongs to a different study")
        run_ids: set[str] = set()
        directories: set[str] = set()
        for block in self.blocks:
            for run in block.runs:
                expected_run_hash = content_hash(run.assignment_payload())
                if run.assignment_hash != expected_run_hash:
                    raise ValueError(f"assignment hash mismatch for {run.run_id}")
                if run.run_id in run_ids:
                    raise ValueError(f"duplicate run ID: {run.run_id}")
                if run.run_directory in directories:
                    raise ValueError(f"duplicate run directory: {run.run_directory}")
                run_ids.add(run.run_id)
                directories.add(run.run_directory)
        expected_plan_hash = content_hash(self.assignment_payload())
        if self.assignment_hash != expected_plan_hash:
            raise ValueError("randomization-plan assignment hash mismatch")


def generate_plan(spec: StudySpec, output_root: str | Path) -> RandomizationPlan:
    """Generate a deterministic complete-block assignment without touching disk."""

    root = Path(output_root).resolve()
    blocks: list[BlockSpec] = []
    for block_index in range(spec.block_count):
        block_seed = _derived_seed(spec.study_seed, "block", block_index)
        conditions = list(ConditionSpec.primary())
        random.Random(block_seed).shuffle(conditions)
        block_id = stable_id(
            "block",
            {
                "study_id": spec.study_id,
                "block_index": block_index,
                "block_seed": block_seed,
            },
            length=12,
        )
        runs: list[RunSpec] = []
        for order_index, condition in enumerate(conditions):
            # All four conditions in a block share the same stochastic stream.
            # Treatment assignment, not initialization or data order, is the contrast.
            run_seed = _derived_seed(spec.study_seed, "paired-run", block_index)
            run_id = stable_id(
                "run",
                {
                    "study_id": spec.study_id,
                    "block_id": block_id,
                    "condition_id": condition.condition_id.value,
                    "run_seed": run_seed,
                },
                length=16,
            )
            run_directory = str(root / spec.study_id / "runs" / run_id)
            row = {
                "study_id": spec.study_id,
                "block_id": block_id,
                "run_id": run_id,
                "condition": condition.to_dict(),
                "order_index": order_index,
                "run_seed": run_seed,
                "run_directory": run_directory,
            }
            runs.append(
                RunSpec(
                    study_id=spec.study_id,
                    block_id=block_id,
                    run_id=run_id,
                    condition=condition,
                    order_index=order_index,
                    run_seed=run_seed,
                    run_directory=run_directory,
                    assignment_hash=content_hash(row),
                )
            )
        blocks.append(
            BlockSpec(
                study_id=spec.study_id,
                block_id=block_id,
                block_index=block_index,
                randomization_seed=block_seed,
                runs=tuple(runs),
            )
        )
    plan_payload = {
        "study_id": spec.study_id,
        "study_spec_hash": spec.spec_hash,
        "randomization_seed": spec.study_seed,
        "blocks": [block.to_dict() for block in blocks],
    }
    plan = RandomizationPlan(
        study_id=spec.study_id,
        study_spec_hash=spec.spec_hash,
        randomization_seed=spec.study_seed,
        blocks=tuple(blocks),
        assignment_hash=content_hash(plan_payload),
    )
    plan.validate()
    return plan


def load_or_create_plan(
    spec: StudySpec,
    *,
    output_root: str | Path,
    plan_path: str | Path,
) -> RandomizationPlan:
    """Load an existing frozen table, or create it exactly once before execution."""

    destination = Path(plan_path)
    expected = generate_plan(spec, output_root)
    if destination.exists():
        plan = RandomizationPlan.from_dict(read_json(destination))
    else:
        try:
            create_json_exclusive(destination, expected.to_dict())
            plan = expected
        except FileExistsError:
            # Another coordinator won the creation race. Its frozen bytes are authority.
            plan = RandomizationPlan.from_dict(read_json(destination))
    if plan.study_id != spec.study_id:
        raise ValueError("frozen plan belongs to a different study")
    if plan.study_spec_hash != spec.spec_hash:
        raise ValueError("study specification changed after randomization was frozen")
    if len(plan.blocks) != spec.block_count:
        raise ValueError("frozen plan block count differs from the study specification")
    if plan.to_dict() != expected.to_dict():
        raise ValueError(
            "frozen randomization differs from the deterministic StudySpec assignment"
        )
    expected_conditions = set(ConditionId)
    if any(
        {run.condition.condition_id for run in block.runs} != expected_conditions
        for block in plan.blocks
    ):
        raise ValueError("frozen plan is not a complete C0-C3 blocked design")
    return plan
