"""Starter paper plots; matplotlib is an optional dependency."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rl4rl.metrics import frontier_progression, rolling_diversity
from rl4rl.schema import BoundaryLabel, TrajectoryEvent


def write_all(
    events: list[TrajectoryEvent], output_dir: str | Path, *, window: int = 20
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_frontier(events, output / "frontier_progression.png"),
        plot_boundary_mix(events, output / "boundary_edit_mix.png"),
        plot_diversity(events, output / "rolling_architecture_diversity.png", window),
    ]
    return paths


def plot_frontier(events: list[TrajectoryEvent], path: str | Path) -> Path:
    plt = _pyplot()
    points = frontier_progression(events)
    if not points:
        raise ValueError("no valid, qualifying parameter observations to plot")
    steps, parameters = zip(*points, strict=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.step(steps, parameters, where="post", linewidth=2)
    axis.set_yscale("log")
    axis.set_xlabel("Search step")
    axis.set_ylabel("Best qualifying parameters (log scale)")
    axis.set_title("Frontier progression")
    axis.grid(alpha=0.25)
    return _save(figure, path)


def plot_boundary_mix(events: list[TrajectoryEvent], path: str | Path) -> Path:
    plt = _pyplot()
    counts = Counter(edit.boundary_label for event in events for edit in event.edits)
    labels = [
        BoundaryLabel.PRESERVING,
        BoundaryLabel.CHANGING,
        BoundaryLabel.AMBIGUOUS,
        BoundaryLabel.NOT_APPLICABLE,
    ]
    values = [counts[label] for label in labels]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar([label.value for label in labels], values)
    axis.set_xlabel("Edit classification")
    axis.set_ylabel("Edit count")
    axis.set_title("Representational-boundary edit mix")
    axis.tick_params(axis="x", rotation=20)
    return _save(figure, path)


def plot_diversity(
    events: list[TrajectoryEvent], path: str | Path, window: int = 20
) -> Path:
    plt = _pyplot()
    points = rolling_diversity(events, window=window)
    steps, diversity = zip(*points, strict=True) if points else ([], [])
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(steps, diversity, linewidth=2)
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Search step")
    axis.set_ylabel("Unique architecture regions / observations")
    axis.set_title(f"Rolling architecture diversity (window={window})")
    axis.grid(alpha=0.25)
    return _save(figure, path)


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plotting requires the analysis extra: uv sync --extra analysis"
        ) from exc
    return plt


def _save(figure, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    return destination
