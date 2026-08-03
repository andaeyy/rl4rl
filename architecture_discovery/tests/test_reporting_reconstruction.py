from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from reporting.synthetic import build_synthetic_reconstruction
from study.serialization import content_hash


def test_provider_free_fixture_reconstructs_files_with_hash_provenance(tmp_path) -> None:
    output = tmp_path / "reconstruction"
    result = build_synthetic_reconstruction(output)

    artifacts = {
        artifact.artifact_id: artifact for artifact in result.report.artifacts
    }
    for derived in result.report.derived_artifacts:
        path = output / derived.relative_path
        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert content_hash(payload) == derived.content_sha256
        assert content_hash(derived.provenance_payload()) == derived.provenance_sha256
        for source in derived.sources:
            assert artifacts[source.artifact_id].content_sha256 == source.content_sha256

    assert result.report.resources.prompt_tokens == 0
    assert result.report.resources.completion_tokens == 0
    assert result.report.resources.provider_usage_complete is True
    assert result.report.external_validity.status.value == "arithmetic_only"


def test_one_command_module_entrypoint_is_offline_and_complete(tmp_path) -> None:
    output = tmp_path / "cli-reconstruction"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "reporting.synthetic",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output / "reproducibility_report.json").is_file()
    assert "report_sha256=" in completed.stdout
