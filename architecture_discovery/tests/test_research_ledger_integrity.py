from __future__ import annotations

from pathlib import Path

import pytest

from research_ledger import ResearchLedger, freeze_protocol
from test_research_ledger_protocol import toy_protocol


def test_event_chain_and_ledger_hash_are_reproducibly_verified(tmp_path: Path) -> None:
    ledger = ResearchLedger(freeze_protocol(toy_protocol(), tmp_path / "protocol.json"))
    close = ledger.close_search(
        closure_id="close-event",
        reason="The synthetic search reached its frozen budget.",
    )
    sealed = ledger.seal(
        seal_id="ledger-seal",
        reason="No post-search evidence exists in this boundary fixture.",
    )

    ledger.verify_integrity()
    assert close.previous_event_sha256 is None
    assert sealed.previous_event_sha256 == close.event_sha256
    assert len(ledger.ledger_hash) == 64
    with pytest.raises(TypeError):
        close.payload["reason"] = "mutated"
