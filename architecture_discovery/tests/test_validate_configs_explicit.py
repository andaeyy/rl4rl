from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_configs.py"


def test_configuration_validator_contains_no_optimization_sensitive_asserts():
    tree = ast.parse(VALIDATOR.read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def test_explicit_invariant_failure_survives_optimized_python():
    program = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from scripts.validate_configs import _require; "
        "_require(False, 'optimized-sentinel')"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode != 0
    assert "optimized-sentinel" in result.stderr
