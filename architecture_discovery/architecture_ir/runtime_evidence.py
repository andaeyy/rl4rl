"""Runtime and metamorphic evidence for transformer validity.

Module names are not evidence.  Scientific callers provide evaluator-owned
bindings from validated IR attention node IDs to concrete module paths.  The
probes verify that the bound modules execute and that intervening on their
outputs changes model behavior.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class RuntimeBindings:
    graph_hash: str
    attention_modules: Mapping[str, str]
    provenance: str = "trusted_ir_interpreter"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "attention_modules", MappingProxyType(dict(self.attention_modules))
        )

    def validate(self) -> tuple[str, ...]:
        problems: list[str] = []
        if len(self.graph_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.graph_hash
        ):
            problems.append("binding graph_hash is not a lowercase SHA-256 digest")
        if self.provenance != "trusted_ir_interpreter":
            problems.append("runtime bindings were not produced by the trusted IR interpreter")
        if not self.attention_modules:
            problems.append("no attention module bindings were supplied")
        if len(set(self.attention_modules.values())) != len(self.attention_modules):
            problems.append("multiple IR attention nodes bind to the same module path")
        return tuple(problems)


@dataclass(frozen=True)
class RuntimeValidityEvidence:
    graph_hash: str
    observed_model_graph_hash: str | None
    binding_provenance: str
    expected_device: str
    observed_parameter_devices: tuple[str, ...]
    observed_buffer_devices: tuple[str, ...]
    output_device: str | None
    attention_calls: Mapping[str, int]
    attention_output_norms: Mapping[str, float]
    causal_mask_buffers_observed: tuple[str, ...]
    causal_prefix_max_delta: float | None
    sequence_dependence_max_delta: float | None
    attention_intervention_max_delta: float | None
    attention_intervention_max_deltas: Mapping[str, float]
    influenced_parameter_tensors: int
    trainable_parameter_tensors: int
    checks: Mapping[str, bool]
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "attention_calls", MappingProxyType(dict(self.attention_calls)))
        object.__setattr__(
            self,
            "attention_output_norms",
            MappingProxyType(dict(self.attention_output_norms)),
        )
        object.__setattr__(
            self,
            "attention_intervention_max_deltas",
            MappingProxyType(dict(self.attention_intervention_max_deltas)),
        )
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))
        object.__setattr__(self, "errors", tuple(self.errors))

    @property
    def passed(self) -> bool:
        required = {
            "binding_valid",
            "graph_identity",
            "device_placement",
            "attention_executed",
            "causal_prefix_invariance",
            "sequence_dependence",
            "attention_influences_output",
            "each_attention_influences_output",
            "parameters_influence_output",
        }
        return not self.errors and all(self.checks.get(name) is True for name in required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_hash": self.graph_hash,
            "observed_model_graph_hash": self.observed_model_graph_hash,
            "binding_provenance": self.binding_provenance,
            "expected_device": self.expected_device,
            "observed_parameter_devices": list(self.observed_parameter_devices),
            "observed_buffer_devices": list(self.observed_buffer_devices),
            "output_device": self.output_device,
            "attention_calls": dict(self.attention_calls),
            "attention_output_norms": dict(self.attention_output_norms),
            "causal_mask_buffers_observed": list(self.causal_mask_buffers_observed),
            "causal_prefix_max_delta": self.causal_prefix_max_delta,
            "sequence_dependence_max_delta": self.sequence_dependence_max_delta,
            "attention_intervention_max_delta": self.attention_intervention_max_delta,
            "attention_intervention_max_deltas": dict(
                self.attention_intervention_max_deltas
            ),
            "influenced_parameter_tensors": self.influenced_parameter_tensors,
            "trainable_parameter_tensors": self.trainable_parameter_tensors,
            "checks": dict(self.checks),
            "errors": list(self.errors),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class FreshBuildEvidence:
    same_seed_state_equal: bool
    different_seed_state_differs: bool
    distinct_instances: bool
    all_initial_parameters_on_cpu: bool
    first_state_hash: str | None
    repeated_state_hash: str | None
    different_seed_state_hash: str | None
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            not self.errors
            and self.same_seed_state_equal
            and self.different_seed_state_differs
            and self.distinct_instances
            and self.all_initial_parameters_on_cpu
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "same_seed_state_equal": self.same_seed_state_equal,
            "different_seed_state_differs": self.different_seed_state_differs,
            "distinct_instances": self.distinct_instances,
            "all_initial_parameters_on_cpu": self.all_initial_parameters_on_cpu,
            "first_state_hash": self.first_state_hash,
            "repeated_state_hash": self.repeated_state_hash,
            "different_seed_state_hash": self.different_seed_state_hash,
            "errors": list(self.errors),
            "passed": self.passed,
        }


def _unwrap_model(value: Any) -> nn.Module:
    if isinstance(value, nn.Module):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], nn.Module):
        return value[0]
    raise TypeError("builder must return nn.Module or (nn.Module, metadata)")


def _state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def probe_fresh_build(
    builder: Callable[[int], Any],
    *,
    seed: int,
    different_seed: int | None = None,
) -> FreshBuildEvidence:
    errors: list[str] = []
    hashes: list[str | None] = [None, None, None]
    models: list[nn.Module] = []
    different_seed = seed + 1 if different_seed is None else different_seed
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    mps_rng: torch.Tensor | None = None
    cuda_rng: list[torch.Tensor] | None = None
    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            mps_rng = torch.mps.get_rng_state()
        except RuntimeError:
            mps_rng = None
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        try:
            cuda_rng = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
        except RuntimeError:
            cuda_rng = None
    try:
        models = [
            _unwrap_model(builder(seed)),
            _unwrap_model(builder(seed)),
            _unwrap_model(builder(different_seed)),
        ]
        hashes = [_state_hash(model) for model in models]
    except Exception as error:
        errors.append(f"fresh-build probe failed: {type(error).__name__}: {error}")
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if mps_rng is not None and hasattr(torch.mps, "set_rng_state"):
            torch.mps.set_rng_state(mps_rng)
        if cuda_rng is not None and hasattr(torch, "cuda"):
            torch.cuda.set_rng_state_all(cuda_rng)

    devices = {
        parameter.device.type
        for model in models
        for parameter in model.parameters()
    }.union(
        buffer.device.type
        for model in models
        for buffer in model.buffers()
    )
    return FreshBuildEvidence(
        same_seed_state_equal=bool(hashes[0] and hashes[0] == hashes[1]),
        different_seed_state_differs=bool(hashes[0] and hashes[0] != hashes[2]),
        distinct_instances=len(models) == 3 and len({id(model) for model in models}) == 3,
        all_initial_parameters_on_cpu=bool(models) and devices.issubset({"cpu"}),
        first_state_hash=hashes[0],
        repeated_state_hash=hashes[1],
        different_seed_state_hash=hashes[2],
        errors=tuple(errors),
    )


def _tensor_from_output(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def _zero_output(output: Any) -> Any:
    if isinstance(output, torch.Tensor):
        return torch.zeros_like(output)
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (torch.zeros_like(output[0]), *output[1:])
    raise TypeError("bound attention module did not return a tensor or tensor-first tuple")


def _has_lower_triangular_mask(module: nn.Module) -> bool:
    for buffer in module.buffers(recurse=False):
        if buffer.ndim < 2 or buffer.shape[-1] != buffer.shape[-2] or buffer.shape[-1] < 2:
            continue
        matrix = buffer.detach().reshape(-1, buffer.shape[-2], buffer.shape[-1])[0]
        if matrix.dtype == torch.bool:
            allowed = matrix
        elif torch.is_floating_point(matrix):
            if torch.isinf(matrix).any() or (matrix < -1e4).any():
                allowed = ~(torch.isinf(matrix) | (matrix < -1e4))
            else:
                allowed = matrix != 0
        else:
            allowed = matrix != 0
        lower = torch.tril(torch.ones_like(allowed, dtype=torch.bool))
        if torch.equal(allowed.cpu(), lower.cpu()):
            return True
    return False


def _max_delta(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.numel() == 0 or second.numel() == 0:
        return 0.0
    return float((first.detach() - second.detach()).abs().max().cpu())


def probe_runtime_validity(
    model: nn.Module,
    *,
    bindings: RuntimeBindings,
    token_ids: torch.Tensor,
    expected_device: str,
    causal_tolerance: float = 1e-5,
    influence_tolerance: float = 1e-8,
) -> RuntimeValidityEvidence:
    """Collect intervention and metamorphic evidence from one built model.

    ``token_ids`` must contain one sequence with at least four tokens.  The
    final token is perturbed for the causal test; the first token is perturbed
    for the sequence-dependence test.
    """

    binding_errors = list(bindings.validate())
    errors = list(binding_errors)
    checks: dict[str, bool] = {"binding_valid": not binding_errors}
    model_graph_hash = getattr(model, "graph_hash", None)
    graph_identity = (
        isinstance(model_graph_hash, str)
        and model_graph_hash == bindings.graph_hash
    )
    checks["graph_identity"] = graph_identity
    if not isinstance(model_graph_hash, str):
        errors.append("model does not expose a trusted graph_hash identity")
    elif model_graph_hash != bindings.graph_hash:
        errors.append(
            "runtime binding graph_hash does not match the interpreted model "
            f"({bindings.graph_hash} != {model_graph_hash})"
        )
    calls = {node_id: 0 for node_id in bindings.attention_modules}
    output_norms = {node_id: 0.0 for node_id in bindings.attention_modules}
    bound_modules: dict[str, nn.Module] = {}
    mask_buffers: list[str] = []
    handles: list[Any] = []
    was_training = model.training
    output_device: str | None = None
    causal_delta: float | None = None
    sequence_delta: float | None = None
    intervention_delta: float | None = None
    intervention_deltas: dict[str, float] = {}
    influenced_parameters = 0
    trainable_parameters = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    input_device: str | None = None

    token_ids_valid = (
        token_ids.ndim == 2 and token_ids.shape[0] == 1 and token_ids.shape[1] >= 4
    )
    if not token_ids_valid:
        errors.append("token_ids must have shape [1, sequence] with sequence >= 4")

    for node_id, path in bindings.attention_modules.items():
        try:
            module = model.get_submodule(path)
            bound_modules[node_id] = module
            if _has_lower_triangular_mask(module):
                mask_buffers.append(node_id)
        except Exception as error:
            errors.append(f"attention binding {node_id}->{path} failed: {type(error).__name__}: {error}")

    def make_hook(node_id: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            calls[node_id] += 1
            tensor = _tensor_from_output(output)
            if tensor is None:
                errors.append(f"attention binding {node_id} returned no observable tensor")
            else:
                output_norms[node_id] = max(
                    output_norms[node_id], float(tensor.detach().float().norm().cpu())
                )

        return hook

    try:
        model.eval()
        for node_id, module in bound_modules.items():
            handles.append(module.register_forward_hook(make_hook(node_id)))

        # Identity mismatches invalidate the evidence but should not suppress
        # the remaining diagnostics.  Only malformed bindings, token inputs,
        # or unresolved module paths make executing the probe meaningless.
        bindings_resolved = len(bound_modules) == len(bindings.attention_modules)
        if not binding_errors and token_ids_valid and bindings_resolved:
            base = token_ids.to(expected_device)
            input_device = base.device.type
            future_changed = base.clone()
            future_replacement = base[:, -2]
            if torch.equal(future_replacement, base[:, -1]):
                future_replacement = torch.where(
                    base[:, -1] == 0,
                    torch.ones_like(base[:, -1]),
                    torch.zeros_like(base[:, -1]),
                )
            future_changed[:, -1] = future_replacement
            prefix_changed = base.clone()
            prefix_replacement = base[:, 1]
            if torch.equal(prefix_replacement, base[:, 0]):
                prefix_replacement = torch.where(
                    base[:, 0] == 0,
                    torch.ones_like(base[:, 0]),
                    torch.zeros_like(base[:, 0]),
                )
            prefix_changed[:, 0] = prefix_replacement
            with torch.no_grad():
                baseline_logits = model(base)
                future_logits = model(future_changed)
                prefix_logits = model(prefix_changed)
            if not all(isinstance(item, torch.Tensor) and item.ndim == 3 for item in (baseline_logits, future_logits, prefix_logits)):
                errors.append("model must return rank-3 tensor logits")
            else:
                output_device = baseline_logits.device.type
                causal_delta = _max_delta(baseline_logits[:, :-1], future_logits[:, :-1])
                sequence_delta = _max_delta(baseline_logits[:, -1:], prefix_logits[:, -1:])

                for handle in handles:
                    handle.remove()
                handles.clear()
                # Intervene on each bound attention module separately.  An
                # aggregate intervention can hide a dead/bypassed attention
                # node when another live node still changes the logits.
                for node_id in sorted(bound_modules):
                    intervention_handle = bound_modules[node_id].register_forward_hook(
                        lambda _module, _inputs, output: _zero_output(output)
                    )
                    try:
                        with torch.no_grad():
                            intervened_logits = model(base)
                    finally:
                        intervention_handle.remove()
                    intervention_deltas[node_id] = _max_delta(
                        baseline_logits, intervened_logits
                    )
                aggregate_handles = [
                    module.register_forward_hook(
                        lambda _module, _inputs, output: _zero_output(output)
                    )
                    for module in bound_modules.values()
                ]
                try:
                    with torch.no_grad():
                        aggregate_intervened_logits = model(base)
                finally:
                    for handle in aggregate_handles:
                        handle.remove()
                intervention_delta = _max_delta(
                    baseline_logits, aggregate_intervened_logits
                )

                model.zero_grad(set_to_none=True)
                gradient_logits = model(base)
                gradient_logits.float().square().mean().backward()
                influenced_parameters = sum(
                    1
                    for parameter in model.parameters()
                    if parameter.requires_grad
                    and parameter.grad is not None
                    and float(parameter.grad.detach().abs().max().cpu()) > influence_tolerance
                )
    except Exception as error:
        errors.append(f"runtime validity probe failed: {type(error).__name__}: {error}")
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    parameter_devices = tuple(sorted({parameter.device.type for parameter in model.parameters()}))
    buffer_devices = tuple(sorted({buffer.device.type for buffer in model.buffers()}))
    device_ok = (
        bool(parameter_devices)
        and set(parameter_devices) == {expected_device}
        and set(buffer_devices).issubset({expected_device})
        and output_device == expected_device
        and input_device == expected_device
    )
    each_attention_influences = (
        set(intervention_deltas) == set(bindings.attention_modules)
        and bool(intervention_deltas)
        and all(
            delta > influence_tolerance for delta in intervention_deltas.values()
        )
    )
    checks.update(
        {
            "device_placement": device_ok,
            "attention_executed": bool(calls) and all(count > 0 for count in calls.values()),
            "causal_prefix_invariance": causal_delta is not None and causal_delta <= causal_tolerance,
            "sequence_dependence": sequence_delta is not None and sequence_delta > influence_tolerance,
            "attention_influences_output": each_attention_influences,
            "each_attention_influences_output": each_attention_influences,
            "parameters_influence_output": influenced_parameters > 0,
            "causal_mask_buffer_observed": bool(mask_buffers),
        }
    )
    return RuntimeValidityEvidence(
        graph_hash=bindings.graph_hash,
        observed_model_graph_hash=(
            model_graph_hash if isinstance(model_graph_hash, str) else None
        ),
        binding_provenance=bindings.provenance,
        expected_device=expected_device,
        observed_parameter_devices=parameter_devices,
        observed_buffer_devices=buffer_devices,
        output_device=output_device,
        attention_calls=calls,
        attention_output_norms=output_norms,
        causal_mask_buffers_observed=tuple(sorted(mask_buffers)),
        causal_prefix_max_delta=causal_delta,
        sequence_dependence_max_delta=sequence_delta,
        attention_intervention_max_delta=intervention_delta,
        attention_intervention_max_deltas=intervention_deltas,
        influenced_parameter_tensors=influenced_parameters,
        trainable_parameter_tensors=trainable_parameters,
        checks=checks,
        errors=tuple(dict.fromkeys(errors)),
    )
