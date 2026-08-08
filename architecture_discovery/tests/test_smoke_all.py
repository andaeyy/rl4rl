from __future__ import annotations

from types import SimpleNamespace

from scripts.smoke_all import _pretrained_regression_fixture


def test_pretrained_regression_fixture_cannot_be_mistaken_for_controller_lineage(
    tmp_path,
):
    source = tmp_path / "candidate.py"
    source.write_text("source = 'fixture'\n", encoding="utf-8")
    result = SimpleNamespace(to_dict=lambda: {"qualifies": True})

    fixture = _pretrained_regression_fixture(source, result, "offline-run")

    assert fixture["schema_name"] == "OfflinePretrainedDecoderRegressionFixture"
    assert fixture["controller_lineage"] is False
    assert fixture["retention_decision"] == "not_applicable"
    assert fixture["parent_id"] is None
    assert "candidate_id" not in fixture
