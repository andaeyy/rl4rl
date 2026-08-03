from pathlib import Path

from containment.source_scan import RiskCategory, scan_python_path, scan_python_source


FIXTURES = Path(__file__).parent / "fixtures" / "adversarial_candidates"


def test_adversarial_python_capabilities_are_reported_without_execution():
    expected = {
        "indirect_builtins.py": {RiskCategory.DYNAMIC_BUILTINS, RiskCategory.FILESYSTEM},
        "credential_reader.py": {RiskCategory.CREDENTIAL_ACCESS},
        "child_process.py": {RiskCategory.CHILD_PROCESS},
        "network_client.py": {RiskCategory.NETWORK},
        "hardcoded_solver.py": {RiskCategory.DIRECT_TASK_SOLVER},
        "stale_checkpoint.py": {RiskCategory.CHECKPOINT_OR_STATE},
    }
    for filename, categories in expected.items():
        report = scan_python_path(FIXTURES / filename)
        assert report.parsed, (filename, report.syntax_error)
        assert categories.issubset(report.categories), (filename, report.to_dict())


def test_dynamic_lookup_is_detected_even_without_a_forbidden_import():
    report = scan_python_source(
        """
def build_untrained_model(seed):
    table = globals()["__builtins__"]
    reader = getattr(table, "open")
    return reader("secret")
"""
    )
    assert RiskCategory.DYNAMIC_BUILTINS in report.categories
    assert RiskCategory.FILESYSTEM in report.categories


def test_source_scanner_does_not_claim_safety_for_parse_failures():
    report = scan_python_source("def broken(:")
    assert not report.parsed
    assert report.risky


def test_source_report_never_contains_file_contents_outside_examined_expression(tmp_path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def build_untrained_model(seed):\n    return seed\n", encoding="utf-8")
    report = scan_python_path(candidate)
    assert report.parsed
    assert not report.risky
