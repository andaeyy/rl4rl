import json

from common.lineage_schema import CandidateRecord, append_record


def test_lineage_record_is_append_only_jsonl(tmp_path):
    path = tmp_path / "lineage.jsonl"
    record = CandidateRecord(
        run_id="test",
        condition="control",
        seed=1,
        candidate_id="child",
        parent_id="parent",
    )
    append_record(path, record)
    append_record(path, record)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["parent_id"] == "parent"


def test_evaluation_fields_are_flattened_into_raw_record(tmp_path):
    path = tmp_path / "lineage.jsonl"
    record = CandidateRecord(
        run_id="test",
        condition="control",
        seed=1,
        candidate_id="child",
        parent_id="parent",
        evaluation={
            "qualifies": True,
            "parameter_count_metadata": 6080,
            "verify_seconds": 0.1,
        },
    )
    append_record(path, record)
    payload = json.loads(path.read_text())
    assert payload["qualifies"] is True
    assert payload["parameter_count_metadata"] == 6080
    assert "evaluation" not in payload
