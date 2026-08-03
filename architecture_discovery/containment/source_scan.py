"""Defense-in-depth risk scanner for exploratory Python candidates.

The scanner reports suspicious capabilities, including common indirect access
patterns.  It must never be used as proof that arbitrary Python is safe.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class RiskCategory(StrEnum):
    DYNAMIC_BUILTINS = "dynamic_builtins"
    FILESYSTEM = "filesystem"
    CREDENTIAL_ACCESS = "credential_access"
    CHILD_PROCESS = "child_process"
    NETWORK = "network"
    CHECKPOINT_OR_STATE = "checkpoint_or_state"
    DIRECT_TASK_SOLVER = "direct_task_solver"
    DYNAMIC_CODE = "dynamic_code"


@dataclass(frozen=True)
class SourceFinding:
    category: RiskCategory
    message: str
    line: int
    column: int
    expression: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "expression": self.expression,
        }


@dataclass(frozen=True)
class SourceRiskReport:
    parsed: bool
    findings: tuple[SourceFinding, ...]
    syntax_error: str | None = None

    @property
    def risky(self) -> bool:
        return not self.parsed or bool(self.findings)

    @property
    def categories(self) -> frozenset[RiskCategory]:
        return frozenset(finding.category for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed,
            "risky": self.risky,
            "syntax_error": self.syntax_error,
            "findings": [finding.to_dict() for finding in self.findings],
        }


_IMPORT_CATEGORIES: dict[str, RiskCategory] = {
    "builtins": RiskCategory.DYNAMIC_BUILTINS,
    "ctypes": RiskCategory.DYNAMIC_CODE,
    "importlib": RiskCategory.DYNAMIC_CODE,
    "io": RiskCategory.FILESYSTEM,
    "multiprocessing": RiskCategory.CHILD_PROCESS,
    "os": RiskCategory.FILESYSTEM,
    "pathlib": RiskCategory.FILESYSTEM,
    "pickle": RiskCategory.CHECKPOINT_OR_STATE,
    "requests": RiskCategory.NETWORK,
    "httpx": RiskCategory.NETWORK,
    "shutil": RiskCategory.FILESYSTEM,
    "socket": RiskCategory.NETWORK,
    "subprocess": RiskCategory.CHILD_PROCESS,
    "sys": RiskCategory.DYNAMIC_CODE,
    "urllib": RiskCategory.NETWORK,
}

_CALL_CATEGORIES: dict[str, RiskCategory] = {
    "__import__": RiskCategory.DYNAMIC_CODE,
    "compile": RiskCategory.DYNAMIC_CODE,
    "eval": RiskCategory.DYNAMIC_CODE,
    "exec": RiskCategory.DYNAMIC_CODE,
    "fork": RiskCategory.CHILD_PROCESS,
    "getenv": RiskCategory.CREDENTIAL_ACCESS,
    "open": RiskCategory.FILESYSTEM,
    "popen": RiskCategory.CHILD_PROCESS,
    "Popen": RiskCategory.CHILD_PROCESS,
    "run": RiskCategory.CHILD_PROCESS,
    "spawn": RiskCategory.CHILD_PROCESS,
    "system": RiskCategory.CHILD_PROCESS,
    "urlopen": RiskCategory.NETWORK,
    "create_connection": RiskCategory.NETWORK,
    "load": RiskCategory.CHECKPOINT_OR_STATE,
    "load_state_dict": RiskCategory.CHECKPOINT_OR_STATE,
    "setstate": RiskCategory.CHECKPOINT_OR_STATE,
}

_DYNAMIC_BUILTIN_NAMES = {
    "__builtins__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
}

_CREDENTIAL_WORDS = {
    "environ",
    "environment",
    "api_key",
    "access_token",
    "auth_token",
    "password",
    "secret",
    "private_key",
}


def _expression(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        return type(node).__name__
    return " ".join(segment.strip().split())[:240]


def _dotted_name(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _constant_strings(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    )


class _RiskVisitor(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.source = source
        self.findings: list[SourceFinding] = []
        self.function_stack: list[str] = []

    def _add(self, node: ast.AST, category: RiskCategory, message: str) -> None:
        finding = SourceFinding(
            category=category,
            message=message,
            line=getattr(node, "lineno", 0),
            column=getattr(node, "col_offset", 0),
            expression=_expression(self.source, node),
        )
        key = (finding.category, finding.line, finding.column, finding.message)
        if not any(
            (item.category, item.line, item.column, item.message) == key
            for item in self.findings
        ):
            self.findings.append(finding)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            category = _IMPORT_CATEGORIES.get(root)
            if category:
                self._add(node, category, f"candidate imports capability-bearing module {root}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        root = (node.module or "").split(".", 1)[0]
        category = _IMPORT_CATEGORIES.get(root)
        if category:
            self._add(node, category, f"candidate imports from capability-bearing module {root}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id in _DYNAMIC_BUILTIN_NAMES:
            self._add(
                node,
                RiskCategory.DYNAMIC_BUILTINS,
                f"candidate accesses dynamic builtin namespace via {node.id}",
            )
        if node.id.lower() in _CREDENTIAL_WORDS:
            self._add(node, RiskCategory.CREDENTIAL_ACCESS, "candidate references credential-like state")

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        lowered = node.attr.lower()
        if lowered in _CREDENTIAL_WORDS:
            self._add(node, RiskCategory.CREDENTIAL_ACCESS, "candidate accesses credential-like attribute")
        if node.attr in {"read", "read_text", "read_bytes", "write", "write_text", "write_bytes"}:
            self._add(node, RiskCategory.FILESYSTEM, "candidate accesses filesystem-like operation")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = _dotted_name(node.func)
        leaf = name[-1] if name else ""
        category = _CALL_CATEGORIES.get(leaf)
        if category:
            self._add(node, category, f"candidate invokes restricted capability {'.'.join(name) or leaf}")

        strings = {value.lower() for value in _constant_strings(node)}
        if leaf in {"getattr", "vars", "globals", "locals"} or strings.intersection(
            {"open", "__import__", "eval", "exec", "environ", "system", "popen"}
        ):
            self._add(
                node,
                RiskCategory.DYNAMIC_BUILTINS,
                "candidate performs indirect builtin or capability lookup",
            )
        if "open" in strings:
            self._add(
                node,
                RiskCategory.FILESYSTEM,
                "candidate resolves a filesystem capability indirectly",
            )
        if strings.intersection(_CREDENTIAL_WORDS):
            self._add(node, RiskCategory.CREDENTIAL_ACCESS, "candidate looks up a credential-like name")

        if self.function_stack and self.function_stack[-1] in {"forward", "add", "solve", "decode"}:
            if leaf in {"int", "str", "join"}:
                self._add(
                    node,
                    RiskCategory.DIRECT_TASK_SOLVER,
                    "task-facing method performs symbolic/Python decoding",
                )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        strings = {value.lower() for value in _constant_strings(node)}
        if strings.intersection({"__builtins__", "open", "__import__", "eval", "exec"}):
            self._add(
                node,
                RiskCategory.DYNAMIC_BUILTINS,
                "candidate indexes an indirect builtin namespace",
            )
        if strings.intersection(_CREDENTIAL_WORDS):
            self._add(node, RiskCategory.CREDENTIAL_ACCESS, "candidate indexes credential-like state")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:  # noqa: N802
        if self.function_stack and self.function_stack[-1] in {"forward", "solve", "decode"}:
            if len(node.keys) >= 8 and all(
                isinstance(key, ast.Constant) for key in node.keys if key is not None
            ):
                self._add(
                    node,
                    RiskCategory.DIRECT_TASK_SOLVER,
                    "task-facing method contains a hard-coded lookup table",
                )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        if self.function_stack and self.function_stack[-1] in {"add", "solve", "decode"}:
            if isinstance(node.op, (ast.Add, ast.FloorDiv, ast.Mod, ast.Mult)):
                self._add(
                    node,
                    RiskCategory.DIRECT_TASK_SOLVER,
                    "candidate contains direct arithmetic in a task solver method",
                )
        self.generic_visit(node)


def scan_python_source(source: str) -> SourceRiskReport:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return SourceRiskReport(False, (), f"{error.msg} at line {error.lineno}")
    visitor = _RiskVisitor(source)
    visitor.visit(tree)
    ordered = tuple(sorted(visitor.findings, key=lambda item: (item.line, item.column, item.category)))
    return SourceRiskReport(True, ordered)


def scan_python_path(path: str | Path) -> SourceRiskReport:
    candidate_path = Path(path)
    try:
        source = candidate_path.read_text(encoding="utf-8")
    except OSError as error:
        return SourceRiskReport(False, (), f"source unreadable: {error}")
    return scan_python_source(source)
