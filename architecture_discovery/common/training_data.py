"""Deterministic evaluator-owned public training and development data."""

from __future__ import annotations

import hashlib
import random

import torch

from common.task_adapter import FixedAdditionTask


def _example_seed(data_seed: int, namespace: str, *positions: int) -> int:
    parts = [str(data_seed), namespace, *(str(position) for position in positions)]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def public_development_cases(seed: int, count: int) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    position = 0
    upper = 10**10 - 1
    while len(cases) < count:
        rng = random.Random(_example_seed(seed, "development", position))
        case = (rng.randint(0, upper), rng.randint(0, upper))
        position += 1
        if case not in seen:
            seen.add(case)
            cases.append(case)
    return cases


def training_case(
    data_seed: int,
    optimizer_step: int,
    example_position: int,
    *,
    min_digits: int,
    max_digits: int,
    excluded_cases: set[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    retry = 0
    while True:
        rng = random.Random(
            _example_seed(
                data_seed, "training", optimizer_step, example_position, retry
            )
        )
        digits_a = rng.randint(min_digits, max_digits)
        digits_b = rng.randint(min_digits, max_digits)
        case = (
            rng.randint(0, 10**digits_a - 1),
            rng.randint(0, 10**digits_b - 1),
        )
        if not excluded_cases or case not in excluded_cases:
            return case
        retry += 1


def training_batch(
    *,
    task: FixedAdditionTask,
    data_seed: int,
    optimizer_step: int,
    batch_size: int,
    example_offset: int = 0,
    min_digits: int,
    max_digits: int,
    excluded_cases: set[tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    cases = [
        training_case(
            data_seed,
            optimizer_step,
            position,
            min_digits=min_digits,
            max_digits=max_digits,
            excluded_cases=excluded_cases,
        )
        for position in range(example_offset, example_offset + batch_size)
    ]
    input_ids, labels = task.collate(cases)
    return input_ids, labels, cases
