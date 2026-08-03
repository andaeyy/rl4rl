from pathlib import Path

import pytest

from evaluation.dependency_audit import (
    assert_controller_dependencies_clean,
    audit_controller_sources,
    audit_local_import_graph,
)


ROOT = Path(__file__).resolve().parents[1]


def test_current_controller_sources_do_not_directly_import_or_read_sealed_layers():
    controller_sources = tuple((ROOT / "agents").glob("*/run.py"))
    assert controller_sources
    assert audit_controller_sources(controller_sources) == ()


@pytest.mark.parametrize(
    "source, expected_code",
    [
        ("from sealed_eval.qualification import LayerBQualificationRunner\n", "forbidden_import"),
        ("from private_eval.shadow_evaluator import shadow_seed\n", "forbidden_import"),
        (
            "from evaluation.records import QualificationEvaluationRecord\n",
            "sealed_record_import",
        ),
        (
            "import evaluation.records as records\n"
            "record_type = records.ConfirmationEvaluationRecord\n",
            "sealed_record_access",
        ),
        (
            "from importlib import import_module\n"
            "value = import_module('sealed_eval.confirmation')\n",
            "forbidden_dynamic_import",
        ),
        (
            "from pathlib import Path\n"
            "value = Path('outputs/sealed/layer_b/result.json').read_text()\n",
            "sealed_path_access",
        ),
    ],
)
def test_source_audit_detects_sealed_imports_and_literal_path_access(
    tmp_path, source, expected_code
):
    candidate = tmp_path / "controller.py"
    candidate.write_text(source, encoding="utf-8")
    issues = audit_controller_sources((candidate,))
    assert expected_code in {issue.code for issue in issues}


def test_dependency_graph_detects_transitive_sealed_import(tmp_path):
    (tmp_path / "controller.py").write_text(
        "from bridge import public_score\n", encoding="utf-8"
    )
    (tmp_path / "bridge.py").write_text(
        "from sealed_eval.qualification import LayerBQualificationRunner\n"
        "public_score = 1.0\n",
        encoding="utf-8",
    )
    issues = audit_local_import_graph((tmp_path / "controller.py",), tmp_path)
    assert any(issue.detail.startswith("sealed_eval") for issue in issues)
    with pytest.raises(RuntimeError, match="evaluation-boundary audit failed"):
        assert_controller_dependencies_clean(
            (tmp_path / "controller.py",), project_root=tmp_path
        )


def test_dependency_graph_accepts_public_search_record_only(tmp_path):
    (tmp_path / "controller.py").write_text(
        "from public_bridge import SearchEvaluationRecord\n", encoding="utf-8"
    )
    (tmp_path / "public_bridge.py").write_text(
        "from evaluation import SearchEvaluationRecord\n", encoding="utf-8"
    )
    assert_controller_dependencies_clean(
        (tmp_path / "controller.py",), project_root=tmp_path
    )
