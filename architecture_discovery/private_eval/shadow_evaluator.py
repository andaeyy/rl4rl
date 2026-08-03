"""Private split generation and carry-focused cases."""

from __future__ import annotations

import os
import random
import secrets


_SESSION_SHADOW_SEED = secrets.randbits(63)


def shadow_seed() -> int:
    value = os.environ.get("DISCOVERY_SHADOW_SEED")
    return int(value) if value else _SESSION_SHADOW_SEED


def random_cases(count: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [
        (rng.randint(0, 9_999_999_999), rng.randint(0, 9_999_999_999))
        for _ in range(count)
    ]


def edge_cases() -> list[tuple[int, int]]:
    return [
        (0, 0),
        (0, 1),
        (9_999_999_999, 0),
        (9_999_999_999, 1),
        (9_999_999_999, 9_999_999_999),
        (5_000_000_000, 5_000_000_000),
        (1_111_111_111, 8_888_888_889),
        (1_234_567_890, 9_876_543_210),
        (1, 9_999_999_999),
    ]


def carry_cases() -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    for width in range(1, 11):
        run = int("9" * width)
        cases.append((run, 1))
        cases.append((run, run))
        cases.append((5 * (10 ** (width - 1)), 5 * (10 ** (width - 1))))
    return cases
