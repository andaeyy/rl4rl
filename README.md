# RL4RL Architecture Discovery

This repository contains research infrastructure for reproducible, autonomous
transformer-architecture discovery on AdderBoard. AdderBoard is used as a
correctness and accuracy environment; parameter count is metadata, not an
optimization objective.

The engineering infrastructure is tested offline. Scientific pilot and
main-study execution remain fail-closed until the evidence and governance gates
documented in `architecture_discovery/readiness_evidence.yaml` are satisfied.

## Clone and set up

```bash
git clone --recurse-submodules https://github.com/rl4rlresearch/rl4rl.git
cd rl4rl
git submodule update --init --recursive
cd architecture_discovery
uv sync --python 3.12
```

Start with
[`architecture_discovery/README.md`](architecture_discovery/README.md) for the
system design, offline validation commands, MPS policy, and scientific launch
gates.

## Repository layout

- `architecture_discovery/` — executable research infrastructure and tests.
- `RIGOROUS_EXPERIMENT_PLAN_V2.md` — primary causal and scientific design.
- `PROJECT_DIRECTION_AND_PAPER_ROADMAP.md` — project direction and publication
  roadmap.
- `CONFIGURE_DISCOVERY_AGENTS_PROMPT.md` — historical setup artifact retained
  for provenance; it is not the current experiment protocol.

Third-party sources are retained under `architecture_discovery/vendor/` as
pinned submodules. The OpenEvolve submodule uses a dedicated fork commit with a
small, documented retry-accounting patch; see
`architecture_discovery/vendor/openevolve/UPSTREAM.md`.
