"""Static and runtime checks for architecture-only candidate modules.

These checks are defense in depth.  They reduce accidental or obvious contract
violations but are not a filesystem or network sandbox.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn

from common.task_adapter import VOCAB_SIZE
from containment.source_scan import scan_python_source


@dataclass(frozen=True)
class ContractResult:
    valid: bool
    reasons: tuple[str, ...]


_FORBIDDEN_IMPORT_ROOTS = {
    "builtins",
    "common",
    "ctypes",
    "http",
    "importlib",
    "io",
    "os",
    "pathlib",
    "pickle",
    "private_eval",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
_FORBIDDEN_CALL_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "open",
}
_FORBIDDEN_ATTRIBUTES = {
    "backward",
    "load",
    "load_state_dict",
    "read_bytes",
    "read_text",
    "save",
    "system",
    "urlopen",
    "write_bytes",
    "write_text",
}
_OPTIMIZER_NAMES = {
    "Adadelta",
    "Adagrad",
    "Adam",
    "AdamW",
    "ASGD",
    "LBFGS",
    "NAdam",
    "Optimizer",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
    "SparseAdam",
}
_FORBIDDEN_RUNTIME_CONTROL_CALLS = {
    ("torch", "set_deterministic_debug_mode"),
    ("torch", "set_float32_matmul_precision"),
    ("torch", "use_deterministic_algorithms"),
}
_FORBIDDEN_RUNTIME_CONTROL_TARGETS = {
    ("torch", "backends", "cuda", "matmul", "allow_tf32"),
    ("torch", "backends", "cudnn", "allow_tf32"),
    ("torch", "backends", "cudnn", "benchmark"),
    ("torch", "backends", "cudnn", "deterministic"),
}


def _call_name(node: ast.Call) -> tuple[str, ...]:
    parts: list[str] = []
    current: ast.expr = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _attribute_name(node: ast.expr) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def inspect_candidate_source(source: str) -> ContractResult:
    reasons: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return ContractResult(False, (f"candidate syntax error: {error}",))

    risk_report = scan_python_source(source)
    for finding in risk_report.findings:
        reasons.append(
            "candidate source risk "
            f"[{finding.category.value}] line {finding.line}: {finding.message}"
        )

    builders = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build_untrained_model"
    ]
    if len(builders) != 1:
        reasons.append("candidate must define exactly one build_untrained_model(seed)")
    elif [argument.arg for argument in builders[0].args.args] != ["seed"]:
        reasons.append("build_untrained_model must accept exactly one positional seed")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in _FORBIDDEN_IMPORT_ROOTS:
                    reasons.append(f"forbidden candidate import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in _FORBIDDEN_IMPORT_ROOTS or root == "torch.optim":
                reasons.append(f"forbidden candidate import: {node.module}")
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if not name:
                continue
            if name[-1] in _FORBIDDEN_CALL_NAMES:
                reasons.append(f"forbidden candidate call: {'.'.join(name)}")
            if name[-1] in _FORBIDDEN_ATTRIBUTES:
                reasons.append(f"forbidden candidate operation: {'.'.join(name)}")
            if "optim" in name or name[-1] in _OPTIMIZER_NAMES:
                reasons.append(f"candidate-controlled optimizer: {'.'.join(name)}")
            if name in _FORBIDDEN_RUNTIME_CONTROL_CALLS:
                reasons.append(
                    "candidate-controlled deterministic runtime: " + ".".join(name)
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = [node.target]
            for target in targets:
                name = _attribute_name(target)
                if name in _FORBIDDEN_RUNTIME_CONTROL_TARGETS:
                    reasons.append(
                        "candidate-controlled deterministic runtime: "
                        + ".".join(name)
                    )

    for builder in builders:
        if any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(builder)):
            reasons.append("training/control loops are not allowed in the model builder")

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "add":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.BinOp) or not isinstance(child.op, ast.Add):
                continue
            names = {
                operand.id
                for operand in (child.left, child.right)
                if isinstance(operand, ast.Name)
            }
            if names == {"a", "b"}:
                reasons.append("candidate performs direct Python answer computation")

    return ContractResult(not reasons, tuple(dict.fromkeys(reasons)))


def inspect_candidate_path(path: str | Path) -> ContractResult:
    candidate_path = Path(path)
    try:
        source = candidate_path.read_text(encoding="utf-8")
    except OSError as error:
        return ContractResult(False, (f"candidate source unreadable: {error}",))
    return inspect_candidate_source(source)


def validate_candidate(module: ModuleType, model: nn.Module) -> ContractResult:
    reasons: list[str] = []
    try:
        source_result = inspect_candidate_source(inspect.getsource(module))
        reasons.extend(source_result.reasons)
    except (OSError, TypeError) as error:
        reasons.append(f"could not inspect candidate source: {type(error).__name__}")

    if not callable(getattr(module, "build_untrained_model", None)):
        reasons.append("missing build_untrained_model")
    if not isinstance(model, nn.Module):
        reasons.append("build_untrained_model did not return torch.nn.Module")
        return ContractResult(False, tuple(dict.fromkeys(reasons)))

    parameters = list(model.parameters())
    if not parameters:
        reasons.append("candidate model has no trainable parameters")
    elif any(parameter.device.type != "cpu" for parameter in parameters):
        reasons.append("untrained candidate must initially reside on CPU")
    if any(buffer.device.type != "cpu" for buffer in model.buffers()):
        reasons.append("untrained candidate buffers must initially reside on CPU")

    attention_modules = [
        child
        for child in model.modules()
        if isinstance(child, nn.MultiheadAttention)
        or "attention" in child.__class__.__name__.lower()
    ]
    if not attention_modules:
        reasons.append("no self-attention module found")

    try:
        probe = torch.zeros((1, 1), dtype=torch.long)
        output = model(probe)
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            reasons.append("forward is not tensor-in, logits-out")
        elif output.shape != (1, 1, VOCAB_SIZE):
            reasons.append(
                "forward output must have shape [batch, sequence, Phase-1 vocabulary]"
            )
    except Exception as error:
        reasons.append(f"forward probe failed: {type(error).__name__}: {error}")

    try:
        was_training = model.training
        model.eval()
        first = torch.tensor([[1, 2, 3]], dtype=torch.long)
        second = torch.tensor([[1, 2, 4]], dtype=torch.long)
        with torch.no_grad():
            first_logits = model(first)
            second_logits = model(second)
        if not torch.allclose(
            first_logits[:, :2],
            second_logits[:, :2],
            rtol=1e-5,
            atol=1e-6,
        ):
            reasons.append("forward computation is not causal")
        model.train(was_training)
    except Exception as error:
        reasons.append(f"causality probe failed: {type(error).__name__}: {error}")

    return ContractResult(not reasons, tuple(dict.fromkeys(reasons)))
