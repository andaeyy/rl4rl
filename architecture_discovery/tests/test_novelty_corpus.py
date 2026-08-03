from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from novelty.corpus import (
    CORPUS_POPULATION_REQUIRED,
    CorpusPopulationRequired,
    ReferenceCorpusManifest,
    ReferenceMechanism,
    freeze_corpus,
    query_corpus,
    verify_frozen_corpus,
)
from test_novelty_signatures import build_graph, signature


def synthetic_manifest() -> ReferenceCorpusManifest:
    entry = ReferenceMechanism(
        reference_id="synthetic-reference-one",
        source_id="synthetic-source-one",
        source_locator="fixture://synthetic-source-one",
        publication_date="2024-02-01",
        source_sha256=hashlib.sha256(b"synthetic source").hexdigest(),
        mechanism_name="Synthetic routed attention",
        mechanism_summary="Fixture-only mechanism for corpus plumbing tests.",
        signature=signature(build_graph(prefix="corpus")),
        independently_reviewed=True,
        reviewer_notes=("Synthetic test fixture only.",),
    )
    return ReferenceCorpusManifest(
        corpus_id="synthetic-corpus-v1",
        cutoff_date="2025-01-01",
        retrieval_date="2026-07-31",
        inclusion_policy="Include every synthetic source in the fixture registry.",
        duplicate_policy="Link exact synthetic duplicates to the earliest fixture.",
        population_complete=True,
        population_attested_by="fixture-builder",
        population_attested_at_utc="2026-07-31T12:00:00Z",
        synthetic_fixture=True,
        entries=(entry,),
    )


def test_freeze_verifies_then_detects_manifest_mutation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "corpus.json"
    seal_path = tmp_path / "corpus.freeze.json"
    manifest = synthetic_manifest()
    freeze_corpus(manifest, manifest_path=manifest_path, seal_path=seal_path)

    verified = verify_frozen_corpus(
        manifest_path=manifest_path,
        seal_path=seal_path,
    )
    assert verified.valid
    assert not verified.scientific_ready

    payload = json.loads(manifest_path.read_text())
    payload["entries"][0]["mechanism_summary"] = "Mutated after freeze."
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    mutated = verify_frozen_corpus(
        manifest_path=manifest_path,
        seal_path=seal_path,
    )
    assert not mutated.valid
    assert any("content hash" in issue for issue in mutated.issues)


def test_existing_freeze_cannot_be_overwritten(tmp_path: Path) -> None:
    manifest_path = tmp_path / "corpus.json"
    seal_path = tmp_path / "corpus.freeze.json"
    manifest = synthetic_manifest()
    freeze_corpus(manifest, manifest_path=manifest_path, seal_path=seal_path)

    with pytest.raises(FileExistsError, match="already exists"):
        freeze_corpus(manifest, manifest_path=manifest_path, seal_path=seal_path)


def test_synthetic_and_unpopulated_corpora_fail_scientific_gate() -> None:
    with pytest.raises(CorpusPopulationRequired) as error:
        synthetic_manifest().assert_scientific_ready()
    assert CORPUS_POPULATION_REQUIRED in str(error.value)

    template_path = Path(__file__).parents[1] / "novelty" / "reference_corpus.template.json"
    template = ReferenceCorpusManifest.from_dict(json.loads(template_path.read_text()))
    with pytest.raises(CorpusPopulationRequired) as template_error:
        template.assert_scientific_ready()
    assert CORPUS_POPULATION_REQUIRED in str(template_error.value)


def test_independent_review_flag_rejects_string_truthiness() -> None:
    payload = synthetic_manifest().to_dict()
    payload["entries"][0]["independently_reviewed"] = "false"

    with pytest.raises(ValueError, match="independently_reviewed must be boolean"):
        ReferenceCorpusManifest.from_dict(payload)


def test_corpus_query_reports_matches_without_assigning_a_novelty_label() -> None:
    manifest = synthetic_manifest()
    query = signature(build_graph(prefix="query", width=64, heads=8))

    matches = query_corpus(manifest, query)

    assert len(matches) == 1
    assert matches[0].reference_id == "synthetic-reference-one"
    assert matches[0].mechanism_match
    assert not matches[0].parameterization_match
    assert not hasattr(matches[0], "novelty_label")
