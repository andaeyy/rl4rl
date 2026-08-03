"""Static guards for the controller/post-search science boundary."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_POSTSEARCH_PACKAGES = ("novelty", "review")
_ONLINE_DESCRIPTOR_PACKAGES = (
    "common.descriptor_extractor",
    "common.descriptor_schema",
)


@dataclass(frozen=True)
class BoundaryIssue:
    path: str
    line: int
    imported_module: str
    rule: str


def _imports(path: Path) -> Iterable[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return ()
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.lineno, node.module))
        elif isinstance(node, ast.Call) and node.args:
            function_name = ""
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            if function_name in {"__import__", "import_module"}:
                module = node.args[0]
                if isinstance(module, ast.Constant) and isinstance(module.value, str):
                    found.append((node.lineno, module.value))
    return tuple(found)


def _python_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from path.rglob("*.py")


def audit_science_boundary(project_root: str | Path) -> tuple[BoundaryIssue, ...]:
    """Find controller imports of post-search science and descriptor leakage."""

    root = Path(project_root)
    issues: list[BoundaryIssue] = []
    controller_paths = [root / "agents", root / "study"]
    vendor_controller = root / "vendor" / "openevolve" / "openevolve" / "controller.py"
    if vendor_controller.exists():
        controller_paths.append(vendor_controller)
    for path in _python_files(controller_paths):
        for line, imported in _imports(path):
            if imported == _POSTSEARCH_PACKAGES[0] or imported.startswith("novelty."):
                issues.append(
                    BoundaryIssue(
                        str(path.relative_to(root)),
                        line,
                        imported,
                        "controller_imports_postsearch_science",
                    )
                )
            if imported == _POSTSEARCH_PACKAGES[1] or imported.startswith("review."):
                issues.append(
                    BoundaryIssue(
                        str(path.relative_to(root)),
                        line,
                        imported,
                        "controller_imports_postsearch_science",
                    )
                )
    for path in _python_files([root / "novelty", root / "review"]):
        for line, imported in _imports(path):
            if any(
                imported == package or imported.startswith(f"{package}.")
                for package in _ONLINE_DESCRIPTOR_PACKAGES
            ):
                issues.append(
                    BoundaryIssue(
                        str(path.relative_to(root)),
                        line,
                        imported,
                        "scientific_novelty_imports_online_descriptor",
                    )
                )
    return tuple(sorted(issues, key=lambda item: (item.path, item.line, item.rule)))
