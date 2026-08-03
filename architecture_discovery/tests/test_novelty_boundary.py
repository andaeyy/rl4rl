from __future__ import annotations

from pathlib import Path

from novelty.dependency_audit import audit_science_boundary


def test_project_controllers_cannot_import_postsearch_science() -> None:
    project_root = Path(__file__).parents[1]
    assert audit_science_boundary(project_root) == ()


def test_audit_detects_controller_import_and_descriptor_leak(tmp_path: Path) -> None:
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "controller.py").write_text(
        "from novelty.clustering import cluster_candidates\n"
        "import importlib\n"
        "importlib.import_module('review.records')\n",
        encoding="utf-8",
    )
    (tmp_path / "study").mkdir()
    (tmp_path / "novelty").mkdir()
    (tmp_path / "novelty" / "bad.py").write_text(
        "from common.descriptor_extractor import extract_descriptors\n",
        encoding="utf-8",
    )
    (tmp_path / "review").mkdir()

    issues = audit_science_boundary(tmp_path)
    assert {issue.rule for issue in issues} == {
        "controller_imports_postsearch_science",
        "scientific_novelty_imports_online_descriptor",
    }
    assert sum(
        issue.rule == "controller_imports_postsearch_science" for issue in issues
    ) == 2
