"""Static dependency checks for the controller-to-sealed evaluation boundary.

This is a readiness guard, not an OS sandbox.  It catches direct imports,
transitive local imports, dynamic imports with literal module names, and
literal reads of known sealed paths.  Dynamic path construction remains part
of the separate containment threat model.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FORBIDDEN_MODULE_PREFIXES = ("sealed_eval", "private_eval")
FORBIDDEN_RECORD_NAMES = frozenset(
    {"QualificationEvaluationRecord", "ConfirmationEvaluationRecord"}
)
FORBIDDEN_PATH_TOKENS = (
    "sealed_eval",
    "private_eval",
    "sealed/layer_b",
    "sealed/layer_c",
)


@dataclass(frozen=True)
class DependencyIssue:
    path: str
    line: int
    code: str
    detail: str


def _forbidden_module(name: str | None) -> bool:
    if not name:
        return False
    return any(
        name == prefix or name.startswith(f"{prefix}.")
        for prefix in FORBIDDEN_MODULE_PREFIXES
    )


def _literal_strings(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    )


def audit_python_source(path: str | Path) -> tuple[DependencyIssue, ...]:
    source_path = Path(path)
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError) as error:
        return (
            DependencyIssue(
                path=str(source_path),
                line=getattr(error, "lineno", 0) or 0,
                code="source_unreadable",
                detail=str(error),
            ),
        )
    issues: list[DependencyIssue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _forbidden_module(alias.name):
                    issues.append(
                        DependencyIssue(
                            str(source_path),
                            node.lineno,
                            "forbidden_import",
                            alias.name,
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if _forbidden_module(node.module):
                issues.append(
                    DependencyIssue(
                        str(source_path),
                        node.lineno,
                        "forbidden_import",
                        node.module or "",
                    )
                )
            if node.module in {"evaluation.records", "evaluation"}:
                leaked = FORBIDDEN_RECORD_NAMES.intersection(
                    alias.name for alias in node.names
                )
                for name in sorted(leaked):
                    issues.append(
                        DependencyIssue(
                            str(source_path),
                            node.lineno,
                            "sealed_record_import",
                            name,
                        )
                    )
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_RECORD_NAMES:
                issues.append(
                    DependencyIssue(
                        str(source_path),
                        node.lineno,
                        "sealed_record_access",
                        node.attr,
                    )
                )
        elif isinstance(node, ast.Call):
            function_name = ""
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            literals = tuple(value.lower().replace("\\", "/") for value in _literal_strings(node))
            if function_name in {"__import__", "import_module"}:
                for literal in literals:
                    if _forbidden_module(literal):
                        issues.append(
                            DependencyIssue(
                                str(source_path),
                                node.lineno,
                                "forbidden_dynamic_import",
                                literal,
                            )
                        )
            if function_name in {
                "open",
                "Path",
                "read_text",
                "read_bytes",
                "glob",
                "rglob",
            }:
                for literal in literals:
                    if any(token in literal for token in FORBIDDEN_PATH_TOKENS):
                        issues.append(
                            DependencyIssue(
                                str(source_path),
                                node.lineno,
                                "sealed_path_access",
                                literal,
                            )
                        )
    return tuple(issues)


def audit_controller_sources(paths: Iterable[str | Path]) -> tuple[DependencyIssue, ...]:
    return tuple(
        issue
        for path in paths
        for issue in audit_python_source(path)
    )


def _local_module_path(project_root: Path, module: str) -> Path | None:
    parts = module.split(".")
    file_candidate = project_root.joinpath(*parts).with_suffix(".py")
    if file_candidate.is_file():
        return file_candidate.resolve()
    package_candidate = project_root.joinpath(*parts, "__init__.py")
    if package_candidate.is_file():
        return package_candidate.resolve()
    return None


def _imported_modules(path: Path) -> tuple[tuple[str, int], ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return ()
    results: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            results.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            results.append((node.module, node.lineno))
    return tuple(results)


def audit_local_import_graph(
    entry_paths: Iterable[str | Path], project_root: str | Path
) -> tuple[DependencyIssue, ...]:
    """Follow local imports and report any transitive route into sealed code."""

    root = Path(project_root).resolve()
    queue = [Path(path).resolve() for path in entry_paths]
    visited: set[Path] = set()
    issues: list[DependencyIssue] = []
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        visited.add(path)
        issues.extend(audit_python_source(path))
        for module, line in _imported_modules(path):
            if _forbidden_module(module):
                issues.append(
                    DependencyIssue(
                        str(path),
                        line,
                        "forbidden_transitive_import",
                        module,
                    )
                )
                continue
            local_path = _local_module_path(root, module)
            if local_path is not None and local_path not in visited:
                queue.append(local_path)
    unique = {
        (issue.path, issue.line, issue.code, issue.detail): issue for issue in issues
    }
    return tuple(unique[key] for key in sorted(unique))


def assert_controller_dependencies_clean(
    paths: Iterable[str | Path],
    *,
    project_root: str | Path | None = None,
) -> None:
    issues = (
        audit_local_import_graph(paths, project_root)
        if project_root is not None
        else audit_controller_sources(paths)
    )
    if issues:
        rendered = "; ".join(
            f"{issue.path}:{issue.line} [{issue.code}] {issue.detail}"
            for issue in issues
        )
        raise RuntimeError(f"controller evaluation-boundary audit failed: {rendered}")
