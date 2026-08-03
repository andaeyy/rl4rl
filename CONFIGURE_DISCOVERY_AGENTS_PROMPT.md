# Prompt: Configure Three AdderBoard Architecture-Discovery Agents

> **Historical setup artifact, not an experiment protocol.** This prompt
> predates evaluator-owned from-scratch training, strict scientific MPS
> execution, the common C0-C3 causal engine, and the Layer A/B/C evaluation
> firewall. Do not run it or use its CPU-fallback, candidate-interface,
> shadow-evaluation, controller-comparison, or budget instructions. The
> controlling scientific protocol is `RIGOROUS_EXPERIMENT_PLAN_V2.md`; the
> current engineering contract is
> `architecture_discovery/IMPLEMENTATION_CONTRACT.md`.

Copy the text below into Codex from the project root.

---

You are configuring a reproducible architecture-discovery experiment on this machine.

## Machine

- macOS 26.5.1
- Apple M5 MacBook Air
- 24 GB unified memory
- 10 CPU cores
- Apple GPU through PyTorch MPS when supported
- Homebrew is installed
- Codex is installed
- System Python is 3.14.6
- `uv`, Claude CLI, Gemini CLI, and NVIDIA CUDA tools were not found during the initial audit

## Research objective

Configure three autonomous research agents that search for valid architectural solutions to AdderBoard:

1. Greedy incumbent-based Autoresearch
2. Generic OpenEvolve
3. Semantic OpenEvolve

AdderBoard supplies the validity and accuracy test. The agents seek architecture-family and mechanism novelty.

Do not optimize, reward, rank, retain, stop, or break ties by parameter count. Record parameter count as metadata.

Do not expose public leaderboard solutions, winning component names, a novelty reference corpus, or shadow-test seeds to the search agents.

Do not run the full experiment during setup. Finish with one baseline evaluation and one cheap smoke iteration for each controller.

## Safety and approval

- Inspect before changing files.
- Keep all project files under the current workspace.
- Request approval before installing Homebrew packages, downloading repositories, or making network calls.
- Never print API keys or secret values.
- Detect credentials by variable name and report only whether each required variable exists.
- Create `.env.example`; do not create a populated `.env`.
- Do not submit anything to AdderBoard.
- Do not start unattended agent loops during configuration.

## Environment setup

1. Read `PROJECT_DIRECTION_AND_PAPER_ROADMAP.md`.
2. Inspect the machine and current workspace.
3. Use Python 3.12 in a workspace-local virtual environment named `.venv`.
4. If Python 3.12 or `uv` is missing, request approval to install `python@3.12` and `uv` with Homebrew.
5. Use PyTorch MPS when the shared candidate supports it. Fall back to CPU with a logged reason.
6. Pin dependency versions and repository commits.
7. Freeze the AdderBoard repository commit in `experiment_manifest.yaml`.
8. Freeze the starting-model source and commit in the same manifest.
9. Use the same starting candidate, evaluator, generator model, mutation format, prompt budget, and evaluation budget for all three agents.

The original Karpathy Autoresearch training code requires an NVIDIA GPU. Do not try to run its nanochat workload. Implement its research protocol around the shared AdderBoard candidate.

## Required project layout

Create this structure:

```text
architecture_discovery/
  README.md
  experiment_manifest.yaml
  pyproject.toml
  .env.example
  common/
    initial_candidate.py
    candidate_contract.py
    evaluator.py
    evaluation_result.py
    descriptor_schema.py
    descriptor_extractor.py
    lineage_schema.py
    prompts/
      shared_system.md
      shared_task.md
  private_eval/
    README.md
    shadow_evaluator.py
  agents/
    greedy_autoresearch/
      program.md
      config.yaml
      run.py
    openevolve_generic/
      system_prompt.md
      config.yaml
      evaluator_adapter.py
    openevolve_semantic/
      system_prompt.md
      config.yaml
      evaluator_adapter.py
      semantic_archive.py
  scripts/
    check_environment.py
    run_baseline.py
    smoke_all.py
    validate_configs.py
  tests/
    test_candidate_contract.py
    test_evaluator.py
    test_descriptors.py
    test_no_size_objective.py
    test_lineage_records.py
  outputs/
    raw/
    checkpoints/
    logs/
```

## Shared candidate contract

All agents must modify copies of the same `initial_candidate.py`.

The candidate must expose:

```python
def build_model():
    ...

def add(model, a: int, b: int) -> int:
    ...
```

The evaluator must check:

- import and execution
- at least one self-attention layer
- tensor-in, logits-out forward computation
- autoregressive token generation
- no Python addition logic in the forward or decoding path
- official-style accuracy
- private shadow accuracy
- edge and carry-chain accuracy
- resource use

Use a short smoke suite during setup. Keep the full official and shadow suites available for experiment runs.

## Shared evaluation result

Return separate fields:

```yaml
execution_ok:
transformer_valid:
official_accuracy:
shadow_accuracy:
edge_accuracy:
carry_accuracy:
robustness_score:
qualifies:
combined_score:
parameter_count_metadata:
train_seconds:
verify_seconds:
failure_stage:
artifacts:
```

Set:

```text
qualifies =
  execution_ok
  AND transformer_valid
  AND official_accuracy >= 0.99
  AND shadow_accuracy >= 0.99
```

`combined_score` may use validity, accuracy, carry accuracy, and robustness. It must not use parameter count, code length, latency, memory, or a public leaderboard score.

The evaluator may return code length and resource values as descriptors or metadata. It must not reward lower values.

Add tests that fail if any configuration or scoring function uses parameter-count metadata in:

- fitness
- incumbent acceptance
- archive replacement
- parent selection
- early stopping
- prompt instructions

## Agent 1: Greedy Autoresearch

Implement a single-incumbent, sequential research loop.

### State

- one current candidate
- one append-only candidate ledger
- accepted lineage in Git
- immutable copies of rejected and crashed candidates

### Loop

1. Read the current candidate and prior ledger entries.
2. Write a proposal before editing:
   - hypothesis
   - architectural assumption under test
   - expected computation
   - planned code change
3. Copy and edit the candidate.
4. Commit or hash the proposed artifact.
5. Evaluate it.
6. Append the complete result to the ledger.
7. Accept the child when it passes the validity gate and frozen robustness floor.
8. Revert candidates that fail either threshold.
9. Continue from the accepted child.

Accept valid plateau moves. Do not require a scalar improvement over a 100 percent baseline. This produces a validity-filtered single-lineage walk.

The agent prompt must not contain architecture-family descriptor names. This agent serves as the greedy control.

Use a fixed candidate-evaluation limit. Disable indefinite `NEVER STOP` behavior.

Save each proposal, prompt, response, diff, code artifact, parent ID, decision, token count, and timestamp.

## Agent 2: Generic OpenEvolve

Install and pin OpenEvolve after approval.

Configure it with:

- the same LLM endpoint and model used by the other agents
- one worker for the first controlled runs
- fixed random seed supplied per run
- fixed iteration count
- no early stopping
- diff-based evolution
- prompt and response logging
- evolution trace enabled
- full candidate code included in traces
- artifacts enabled
- persistent database and checkpoints
- no LLM evaluator

Use generic quality-diversity descriptors:

- code complexity or code length
- structural edit diversity

These values define archive cells. They do not enter `combined_score`.

Use validity and robustness to replace candidates within a cell. Use completion order as the last deterministic tie-break. Do not use parameter count.

The generic prompt must describe the task and validity constraints. It must not name semantic architecture axes.

## Agent 3: Semantic OpenEvolve

Start from the generic OpenEvolve configuration. Change only the archive descriptors and semantic prompt additions required for architecture-family exploration.

Use stable categorical codes for:

- token representation
- positional integration
- attention projection mechanism
- attention organization
- feedforward mechanism
- normalization
- depth and topology
- output readout
- tokenization

Create a frozen mapping from category names to integer codes. Reserve one code for `unknown_or_other` on each axis.

Do not trust self-reported descriptors from generated code. Extract descriptors with independent static and runtime checks. Store extractor confidence and unresolved fields.

Avoid one large joint grid. Implement:

- per-axis coverage
- selected pairwise cells
- complete descriptor signatures
- parent sampling from occupied and underexplored cells

Use validity and robustness for within-cell replacement. The novelty reference corpus remains hidden and plays no role in online fitness.

The semantic prompt may name abstract architecture axes. It must not name known AdderBoard winners, their code, or their constants.

## Shared language-model configuration

Pin the primary study to GPT-5.6 Sol through environment variables:

```text
DISCOVERY_API_KEY
DISCOVERY_API_BASE
DISCOVERY_MODEL
DISCOVERY_REASONING_EFFORT
DISCOVERY_MAX_COMPLETION_TOKENS
DISCOVERY_REQUEST_TIMEOUT_SECONDS
DISCOVERY_REQUEST_RETRIES
DISCOVERY_RETRY_DELAY_SECONDS
```

Require `DISCOVERY_MODEL=gpt-5.6-sol` and fail fast if a stale model such as
GPT-4.1 is selected. Never hard-code the API key. Record the resolved model
identifier and all non-secret request settings in each run manifest.

Set the same:

- Chat Completions API mode
- high reasoning effort
- 16,384 maximum completion tokens
- 300-second request timeout
- two retries with a three-second delay
- omitted temperature and top-p
- prompt templates

across all three agents.

Pass the controller seed to every provider request and record that API seed
support is best-effort. Do not claim exact LLM reproducibility.

## Logging schema

Each evaluated candidate needs:

```text
run_id
condition
seed
candidate_id
parent_id
inspiration_ids
proposal_text
mechanism_hypothesis
prompt_hash
response_hash
code_hash
diff
proposal_timestamp
completion_timestamp
execution_ok
transformer_valid
official_accuracy
shadow_accuracy
edge_accuracy
carry_accuracy
robustness_score
qualifies
descriptor_vector
descriptor_confidence
retention_decision
archive_cells
rollback_target
future_parent_count
input_tokens
output_tokens
train_seconds
verify_seconds
parameter_count_metadata
failure_stage
```

Write raw JSONL records. Generate analysis tables from raw records. Do not hand-edit result tables.

## Configuration invariants

The configuration validator must confirm:

- all agents start from the same code hash
- all agents use the same evaluator hash
- all agents use the same generator model settings
- all agents receive the same candidate and token budgets
- no score reads parameter-count metadata
- no prompt asks for smaller, minimal, compressed, or low-parameter models
- generic OpenEvolve does not receive semantic descriptors
- semantic OpenEvolve does not receive the novelty reference corpus
- evolution traces preserve rejected candidates
- each candidate has one lineage record

## Smoke tests

Run:

1. environment check
2. candidate contract tests
3. baseline smoke evaluation
4. one no-op or harmless mutation through greedy Autoresearch
5. one generic OpenEvolve iteration
6. one semantic OpenEvolve iteration
7. configuration-invariant tests

Keep smoke evaluation cheap. Do not launch training sweeps or unattended loops.

## Final report

Report:

- files created
- pinned commits and dependency versions
- Python and device selected
- MPS or CPU status
- credential variable presence without values
- baseline smoke result
- one smoke result per controller
- test results
- commands for a five-iteration pilot
- unresolved setup risks

Stop after the smoke tests and wait for approval before starting any paid API calls beyond the approved smoke iterations or any experiment longer than five candidate evaluations.

---
