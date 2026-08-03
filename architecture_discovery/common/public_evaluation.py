"""Deterministic public Layer A case generation.

This generator is controller-visible by design. It contains no shadow, edge,
carry, Layer B, or Layer C cases. Scientific use still requires a frozen PI
decision and an explicit evaluation plan.
"""

from __future__ import annotations

from common.training_config import stable_hash
from common.training_data import public_development_cases


PUBLIC_LAYER_A_SOURCE_ID = "public-addition-search-v1"
PUBLIC_LAYER_A_SEED = 2025
PUBLIC_LAYER_A_SOURCE_SHA256 = stable_hash(
    {
        "source_id": PUBLIC_LAYER_A_SOURCE_ID,
        "generator": "public_development_cases",
        "namespace": "development",
        "seed": PUBLIC_LAYER_A_SEED,
        "operand_min": 0,
        "operand_max": 9_999_999_999,
        "deduplicate": True,
    }
)


def public_search_cases(count: int) -> list[tuple[int, int]]:
    if count <= 0:
        raise ValueError("Layer A case count must be positive")
    return public_development_cases(PUBLIC_LAYER_A_SEED, count)

