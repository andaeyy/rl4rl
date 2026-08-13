# RL4RL

RL4RL is a research codebase for reconstructing and measuring how autonomous
discovery systems search the AdderBoard architecture space. The benchmark is
treated as an experimental apparatus: the primary objects of study are edits,
lineages, rollbacks, verifier outcomes, architectural regimes, and frontier
progression.

The repository starts with a deliberately small, auditable core:

- a canonical JSONL schema shared by OpenEvolve, autoresearch, and
  TTT-Discover trajectories;
- an explicit taxonomy for ontology-preserving versus representationally
  boundary-crossing edits;
- an autoresearch TSV adapter;
- lineage validation and descriptive search-dynamics metrics;
- optional plotting helpers for the first paper figures; and
- tests plus a synthetic example (not an experimental result).

Read [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md) before running new experiments.
It contains the verified literature links, corrections to the inherited notes,
paper-positioning advice, data requirements, and ordered next steps.

The separate `architecture_discovery/` execution system is configured to use
Modal with NVIDIA CUDA as the canonical backend for new remote engineering
runs; its bounded live validation remains pending. Version-1 MPS artifacts
remain historical compatibility evidence and are not treated as cross-device
equivalents. No paid run or scientific launch is authorized by the migration.
Start with the provider-free checks and approval gates in
[architecture_discovery/README.md](architecture_discovery/README.md).

## Quick start

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --extra dev --extra analysis
uv run rl4rl validate data/examples/synthetic_trajectory.jsonl
uv run rl4rl summarize data/examples/synthetic_trajectory.jsonl \
  --external-frontier 36
uv run pytest
```

The core package has no runtime dependencies. If you do not need plots:

```bash
PYTHONPATH=src python -m rl4rl.cli validate \
  data/examples/synthetic_trajectory.jsonl
PYTHONPATH=src python -m rl4rl.cli summarize \
  data/examples/synthetic_trajectory.jsonl --external-frontier 36
```

To normalize an autoresearch results log:

```bash
uv run rl4rl parse-autoresearch path/to/results.tsv \
  --run-id autoresearch-replication-01 \
  --output data/interim/autoresearch-replication-01.jsonl
```

The adapter does not invent ancestry. Supply a parent column in the TSV or join
the normalized events against the full Git graph in a later ingestion step.

To generate the starter figures (requires the `analysis` extra):

```bash
uv run rl4rl plot data/examples/synthetic_trajectory.jsonl \
  --output-dir outputs/figures
```

## Repository layout

```text
configs/                 Versioned taxonomy configuration
data/examples/           Synthetic, executable examples
data/raw/                Immutable source artifacts (ignored by Git)
data/interim/            Normalized but not adjudicated events
data/processed/          Analysis-ready, adjudicated trajectories
docs/                    Schema and measurement documentation
outputs/                 Generated tables and figures (ignored by Git)
schemas/                 Interchange JSON Schema
src/rl4rl/               Library, adapters, CLI, metrics, and plots
tests/                    Unit tests
references.bib           Seed bibliography
```

## Data contract

One JSON object represents one proposed or evaluated change. Important fields
include:

- stable event, run, and parent identifiers;
- discovery paradigm and chronological step;
- verifier status, validity, acceptance, and reward-hacking evidence;
- architecture snapshot, parameter count, accuracy, and a design fingerprint;
- one or more component-level edit annotations; and
- source-artifact references and annotation provenance.

Use `data/raw` only for immutable exports. Record source hashes before parsing.
Do not treat heuristic taxonomy suggestions as ground truth; all primary labels
should be independently annotated and adjudicated.

## Development

```bash
make check
```

`make check` runs Ruff and pytest. The project is licensed under MIT; see
[LICENSE](LICENSE).
