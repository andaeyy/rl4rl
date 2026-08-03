from __future__ import annotations

import hashlib
import json
import re

import pytest

from review.blinding import (
    ReviewLeakageError,
    ReviewMaterial,
    generate_blinded_packets,
)
from test_novelty_signatures import build_graph, signature


def test_packet_masks_candidate_treatment_outcome_size_and_ancestry() -> None:
    internal_candidate_id = "C3-run-seven-candidate-nine"
    corpus_hash = hashlib.sha256(b"frozen corpus").hexdigest()
    material = ReviewMaterial(
        candidate_id=internal_candidate_id,
        signature=signature(build_graph(prefix="blind", width=64, heads=8)),
        mechanism_summary="Routes contextual state through a causal attention path.",
        causal_claim="The routing operation changes how prefix state reaches the readout.",
        falsifiable_prediction="Zeroing the routed path removes the prefix-state effect.",
        nearest_reference_ids=("reference-one",),
    )

    packets, index = generate_blinded_packets(
        (material,),
        corpus_sha256=corpus_hash,
        blinding_secret="fixture-secret-with-enough-bytes",
    )
    packet_text = json.dumps(packets[0].to_dict(), sort_keys=True)
    lowered = packet_text.lower()

    assert internal_candidate_id not in packet_text
    assert re.search(r"\bC3\b", packet_text) is None
    assert "condition_id" not in lowered
    assert "openevolve" not in lowered
    assert "accuracy" not in lowered
    assert "parameter_count" not in lowered
    assert "parameterization_hash" not in lowered
    assert index.entries[0].candidate_id == internal_candidate_id
    assert index.entries[0].packet_id == packets[0].packet_id
    with pytest.raises(TypeError):
        packets[0].mechanism_evidence["mechanism_graph_hash"] = "tampered"


def test_packet_generation_rejects_textual_treatment_leakage() -> None:
    with pytest.raises(ReviewLeakageError, match="contains a treatment"):
        ReviewMaterial(
            candidate_id="candidate-safe",
            signature=signature(build_graph(prefix="leak")),
            mechanism_summary="C3 from OpenEvolve obtained a strong public accuracy result.",
            causal_claim="The mechanism changes routing.",
            falsifiable_prediction="Removing the route removes the effect.",
        )
