# RL4RL research handoff

Source audit and recommendations current to **August 2, 2026**.

## Executive assessment

The project is viable, but the paper should be positioned more narrowly than
the inherited roadmap. A June 2026 paper, **Heuresis**, already studies search
strategies for autonomous research agents across quality, diversity, novelty,
lineages, convergence, and reward hacking. A generic claim that this is the
first empirical study of autonomous discovery search behavior would therefore
be difficult to defend.

The strongest remaining contribution is more specific:

> A cross-paradigm, architecture-grounded study of whether autonomous discovery
> systems cross pre-declared representational boundaries, in a controlled task
> where a materially better external architecture frontier is known.

AdderBoard gives this study an unusual counterfactual: the search can stop at a
locally well-supported floor while public, verified architectures demonstrate
that much smaller solutions exist in different representational families. That
is more concrete than asking an LLM judge whether an idea merely sounds novel.

Prefer **representational-family preserving/crossing** in methods and figures.
“Ontology-changing” is memorable but philosophically loaded; it can remain the
informal motivation if the operational term is defined first.

## Verified sources for the named systems

### TTT-Discover

- Paper: [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175)
  (Yuksekgonul et al., arXiv:2601.16175, submitted January 22, 2026; v2 February
  5, 2026).
- Code: [test-time-training/discover](https://github.com/test-time-training/discover).
- What the paper actually establishes: TTT-Discover performs problem-specific
  reinforcement learning at test time with an entropic objective and
  PUCT-inspired state reuse. Its standard implementation uses gpt-oss-120b,
  rank-32 LoRA, 50 training steps, and 512 rollouts per step (eight groups of
  64). The total comparison budget is therefore 25,600 rollouts.
- OpenAI model source: [official gpt-oss developer resources](https://developers.openai.com/learn/gpt-oss)
  and the [official Transformers deployment guide](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers#pick-your-model).
- Important scope note: AdderBoard is not an environment evaluated in the paper.
  Your environment, reward, constraint handling, and all resulting claims are a
  new adaptation that must be documented independently.

### AlphaEvolve and OpenEvolve

- AlphaEvolve paper: [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)
  (Novikov et al., arXiv:2506.13131, 2025).
- OpenEvolve source: [algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve).
- I found no dedicated OpenEvolve arXiv paper. The repository asks users to cite
  it as software: Asankhaya Sharma, *OpenEvolve: an open-source evolutionary
  coding agent* (2025).
- Do not use “OpenEvolve/AlphaEvolve” interchangeably. OpenEvolve is an
  open-source implementation inspired by AlphaEvolve, with its own MAP-Elites,
  island, sampling, artifact, and provider choices. Cite both when the paper
  discusses the family and the concrete implementation.

### autoresearch

- Canonical source: [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
- I found no dedicated arXiv paper by Karpathy for this project. Cite the
  repository as software.
- The official repository describes a small autonomous loop for nanochat: an
  agent edits `train.py`, trains for a fixed five-minute wall-clock budget,
  evaluates validation bits per byte, and keeps or discards the change. Humans
  program the loop through `program.md`.
- The AdderBoard adaptation, Claude Code/Opus version, 30-day dates, Exp76–85
  sequence, and stopping statement are not claims established by Karpathy's
  repository. They are primary data from this project and need immutable logs,
  prompts, commit objects, and hashes in the artifact release.

### AdderBoard and the originating experiment

- Benchmark and live leaderboard: [anadim/AdderBoard](https://github.com/anadim/AdderBoard).
- Originating author article: [Addition Under Pressure](https://dimitrisp.substack.com/p/addition-under-pressure)
  (Dimitris Papailiopoulos, February 19, 2026).
- No AdderBoard arXiv paper was found. The repository provides a software-style
  BibTeX entry and should be the authoritative source for current rules.
- The original experiment produced 6,080 parameters for Claude Code and 1,644
  for Codex. The 1,694-parameter autoresearch Exp76 is a different result and
  should not be conflated with the original 1,644-parameter Codex submission.
- The live rules require at least 99% accuracy on 10,000 held-out random pairs,
  fixed seed 2025; the verifier also runs ten edge cases. Unique parameters are
  counted after weight tying/deduplication. Fixed/sinusoidal positional
  encodings are exempt; learned positional encodings count.

## Corrections required in the inherited notes

1. **The category leaders were reversed in one paragraph.** As of this audit,
   the trained leader is **36 parameters**, while the hand-coded listing is
   **6\*** parameters. Later parts of the notes state this correctly, but the
   “well characterized human ceiling” paragraph does not.
2. **The 6-parameter result needs an asterisk.** AdderBoard says four additional
   hard-coded weight values were counted in the parent 10-parameter submission;
   under strict counting the model has ten unique values. Use `6*` and report a
   robustness analysis against both 6 and 10.
3. **These are reference frontiers, not ceilings.** The leaderboard is live and
   can improve. Archive a dated commit of AdderBoard for every paper draft and
   write “best publicly known frontier as of DATE/COMMIT.”
4. **The proposed TTT reward has the wrong apparent direction.** TTT-Discover
   maximizes reward. “Parameter count if accuracy passes, zero otherwise” would
   reward larger qualifying models unless the adapter negates or inverts the
   raw score. Define and test a monotone smaller-is-better reward explicitly.
   Also avoid an entirely sparse pass/fail signal: preregister how sub-threshold
   accuracy contributes without allowing accuracy to dominate size after 99%.
5. **OpenEvolve is not AlphaEvolve.** Describe it as a particular open-source
   evolutionary coding system, not as a drop-in empirical proxy whose behavior
   automatically transfers to the proprietary AlphaEvolve system.
6. **“Under a minute on a single GPU” needs local evidence.** The public source
   specifies the verifier workload but not a universal runtime. Report hardware,
   software versions, median, dispersion, and warm/cold timing.
7. **Internal system/version claims need artifacts.** “GPT 5.5,” “Opus 4.6,”
   50-trial baselines, exact OpenEvolve iterations, Exp76–85, and the April–May
   autonomous run are not supported by the public papers. Preserve the exact
   provider model IDs and source logs rather than citing a related paper.

## Closest and supporting literature

### The critical comparison: Heuresis

[Heuresis: Search Strategies for Autonomous AI Research Agents Across Quality,
Diversity and Novelty](https://arxiv.org/abs/2606.25198) (Antoniades et al.,
arXiv:2606.25198v2, July 1, 2026) is now mandatory related work.

It compares Greedy, MAP-Elites, Go-Explore, Islands, Curiosity, and Omni on
nanoGPT, on-policy RL, and model unlearning. It reports 3,222 scored runs,
embedding-space idea diversity, search lineages, novelty judgments grounded by
web search, convergence, held-out performance, and a reward-hacking audit. Its
headline result is that novel ideas are rare and rarely coincide with the
quality frontier. It also finds independent islands converging on similar
building blocks and migration sometimes accelerating convergence.

The differentiation to state explicitly:

- Heuresis measures semantic idea diversity and literature-relative novelty;
  RL4RL measures transitions between a task-specific, preregistered set of
  architectural representation families.
- Heuresis does not have a known, dramatically better external architecture
  frontier for every task; AdderBoard makes missed regions observable.
- RL4RL compares three different discovery paradigms on the same verifier and
  objective, including test-time learning, evolutionary code search, and a
  long-horizon greedy commit loop.
- RL4RL can connect invalid/cheating ancestors, rollback behavior, stopping
  language, and architectural transitions in complete trajectories.

Do not copy Heuresis's embedding distance as the primary diversity measure. It
is a useful robustness check, and its authors explicitly show that population
diversity does not imply individual novelty.

### Other directly relevant work

- [Epistemic Uncertainty for Test-Time Discovery](https://arxiv.org/abs/2605.11328)
  (Riaz et al., 2026) argues that ordinary test-time RL can prioritize familiar,
  lower-variance mutations and proposes an ensemble-disagreement exploration
  bonus. It is closely aligned with the diversity-collapse hypothesis, though
  its claims should be treated as a separate preprint to replicate rather than
  as settled evidence.
- [Mathematical discoveries from program search with large language models](https://doi.org/10.1038/s41586-023-06924-6)
  (Romera-Paredes et al., Nature 2024) introduces FunSearch, the central earlier
  example of a frozen LLM used as a variation operator with an evaluator and
  island-based evolutionary search.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) is the source
  for the original sinusoidal positional encoding and Transformer architecture.
- [RoFormer](https://arxiv.org/abs/2104.09864) is the primary RoPE source and
  supports treating RoPE as integration of position through rotations in the
  attention computation rather than as another additive position vector.
- [Train Short, Test Long](https://arxiv.org/abs/2108.12409) is the primary ALiBi
  source; ALiBi biases attention scores rather than adding position embeddings.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) is the
  source for SwiGLU and related gated feed-forward variants.
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) is
  the RMSNorm source.
- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)
  is the appropriate source for the delayed-generalization phenomenon discussed
  in the originating addition experiment and leaderboard recipes.

The bibliography is seeded in [references.bib](references.bib).

## Operational hypothesis and preregistered outcomes

Use this as the primary hypothesis:

> Conditional on a fixed search budget, current autonomous discovery pipelines
> propose and accept representational-family-preserving edits more often than
> representational-family-crossing edits, and their explored architecture
> distribution becomes more concentrated over time even when the known external
> frontier occupies different families.

Recommended primary outcomes:

1. event-level boundary-crossing proposal rate, summarized at the run level;
2. accepted boundary-crossing rate;
3. time or compute to first crossing, with runs that never cross treated as
   right-censored;
4. architecture-family coverage and its change over fixed compute windows;
5. distance to the dated trained frontier (36 parameters at this audit);
6. fraction of frontier improvements attributable to preserving versus crossing
   edits; and
7. stopping after the last novel family visit, measured without interpreting
   private chain-of-thought.

Secondary outcomes:

- edit-category entropy and architecture-fingerprint revisit rate;
- lineage depth, branching, dominant-ancestor share, and island overlap;
- rollback and invalid/error rates;
- descendants of invalid/verifier-exploiting intermediates and their eventual
  validity;
- confidence/stopping statements extracted from ordinary agent messages;
- semantic-embedding diversity as a robustness comparison to Heuresis; and
- sensitivity to alternate boundary taxonomies.

Events within a run are dependent. Do not report tiny event-level p-values as if
hundreds of mutations were independent replications. Use run-level bootstrap
intervals, randomization/permutation tests across matched runs, or hierarchical
models only when the number of independent runs supports them.

## Annotation protocol

1. Freeze taxonomy v0.1 before outcome analysis. The initial suggestions live
   in `configs/taxonomy.toml`.
2. Segment compound commits into component-level edits. One event may contain
   several labels and counts as a crossing event if any adjudicated edit crosses.
3. Give two annotators the before/after code or architecture summary while
   hiding system identity, final score, and descendant success when feasible.
4. Annotate independently as preserving, changing, ambiguous, or not applicable.
5. Report raw agreement and Cohen's kappa (or Krippendorff's alpha for more than
   two annotators), then adjudicate.
6. Repeat headline analyses under a conservative rule where ambiguous edits are
   preserving and a liberal rule where they are changing.
7. Keep heuristic or LLM-generated labels separate; never silently merge them
   with human gold labels.

The inherited example classifications are sensible starting hypotheses:

- learned additive position embedding → sinusoidal additive encoding:
  preserving;
- sinusoidal/additive → RoPE or ALiBi: crossing;
- lookup embedding table → circular-arc parametric function: crossing;
- independent learned Q/K/V/O → tied or low-rank learned maps: preserving;
- learned maps → deterministic algebraic rotations/transposes: crossing;
- ReLU/GELU → another pointwise activation: preserving;
- pointwise FFN → multiplicative SwiGLU gate: provisionally crossing;
- fewer layers, smaller width, head-count changes, or sharing learned norm scales:
  preserving.

These labels should be adjudicated at the mechanism level, not assigned from a
keyword alone.

## Data to collect before analysis

### OpenEvolve

- exact OpenEvolve commit/config, island count, population settings, seed, model
  ID, prompts, and token/monetary budget;
- complete checkpoint/database exports, not only elites;
- program source and evaluator output for every candidate;
- parent and inspiration IDs, island membership, migration events, and timestamps;
- verifier stdout/stderr, parameter counts, accuracy, and exploit adjudication;
- acceptance/archive decisions and any manual restarts.

### autoresearch

- full Git repository including otherwise unreachable discarded commits;
- tracked or archived results TSV, `program.md`, original `train.py`, verifier
  wrapper, and environment lock;
- stdout/stderr and verifier output for every experiment;
- ordinary agent messages containing hypotheses, confidence, rollback, and stop
  decisions;
- exact start/end timestamps, context resets, model ID, permissions, and any
  watchdog behavior; and
- all independent replications using the identical prompt/setup.

### TTT-Discover

- exact scaffold commit and new AdderBoard environment code;
- reward definition and direction tests;
- every rollout's initial state, reused ancestor, generated action/code, reward,
  verifier record, and token count;
- per-step policy/LoRA checkpoints and training metrics;
- PUCT buffer state and selection probabilities;
- model/Tinker configuration, seed, failures, and total compute/cost; and
- matched frozen-policy Best-of-N and state-reuse baselines.

### External frontier

- dated AdderBoard repository commit and leaderboard snapshot;
- verified source for every architecture used in the comparison set;
- normalized component annotations under the same taxonomy as agent attempts;
- separate trained and hand-coded categories; and
- both 6 and 10 parameter interpretations for the disputed hand-coded leader.

## Ordered next steps

### Week 1: freeze evidence and make ingestion real

1. Copy every raw artifact into `data/raw/<paradigm>/<run-id>/` without editing it.
2. Generate SHA-256 manifests and record source repository commits, prompts,
   model IDs, dates, seeds, and budgets in a run manifest.
3. Export unreachable autoresearch commits before repository maintenance or GC.
4. Adapt the provided TSV parser, then write OpenEvolve and TTT adapters against
   actual artifacts. Never infer ancestry when the source graph is available.
5. Run `rl4rl validate` and manually inspect ten randomly selected events per
   source.

### Week 2: pilot the taxonomy

1. Sample at least 100 edits, stratified by paradigm, outcome, and early/late
   trajectory position.
2. Have two people label them independently using only architecture evidence.
3. Revise taxonomy definitions before seeing system-level headline rates.
4. Freeze taxonomy v1 and document all changes from v0.1.
5. Annotate the dated external-frontier architectures using the same protocol.

### Weeks 3–4: establish replication and controls

1. Complete multiple independent runs per paradigm. One 30-day run cannot
   distinguish a system effect from a single trajectory.
2. Match or normalize budgets in verifier calls, generated tokens, wall-clock,
   and estimated cost; report all four rather than asserting exact equivalence.
3. For TTT-Discover, include frozen-policy Best-of-25,600 and state-reuse-only
   controls so policy learning is separable from sampling and reuse.
4. For OpenEvolve, retain runs with and without exploit descendants only if the
   intervention is preregistered; otherwise analyze naturally occurring invalid
   ancestors observationally and avoid causal language.
5. For autoresearch, run the identical “never stop / radical changes” prompt
   enough times to estimate stopping variability.

### Week 5: produce the minimum viable results section

Generate these figures first:

1. best qualifying parameter count versus verifier calls, one panel per run;
2. stacked preserving/changing/ambiguous edit shares over fixed windows;
3. architecture-family coverage or entropy over time;
4. lineage trees with valid, invalid, rejected, and frontier-improving nodes;
5. time-to-first-boundary-crossing survival plot; and
6. explored component-family combinations versus the dated external frontier.

Then run robustness checks with alternate ambiguity treatment, excluding and
including invalid candidates, exact versus neighborhood revisit definitions,
and trained-only versus trained-plus-hand-coded reference frontiers.

### Week 6: write against the strongest alternative explanation

The paper should rule out, or clearly delimit:

- insufficient compute rather than representational bias;
- an initial-code/prompt prior rather than a general system tendency;
- taxonomy choices manufactured after observing winners;
- verifier artifacts or parameter-count loopholes;
- event dependence masquerading as replication;
- dynamic leaderboard drift; and
- search systems being incomparable because budgets and accessible actions differ.

The clean conclusion is evidence of a bias under the studied tasks, prompts,
scaffolds, and budgets—not a universal inability of agents to invent new
architectures.

## Commands available now

```bash
# Install and verify the scaffold
uv sync --extra dev --extra analysis
make check

# Exercise the canonical schema and metrics on synthetic data
uv run rl4rl validate data/examples/synthetic_trajectory.jsonl
uv run rl4rl summarize data/examples/synthetic_trajectory.jsonl \
  --external-frontier 36

# Normalize a real autoresearch TSV
uv run rl4rl parse-autoresearch /path/to/results.tsv \
  --run-id autoresearch-replication-01 \
  --output data/interim/autoresearch-replication-01.jsonl

# Generate starter plots
uv run rl4rl plot data/interim/autoresearch-replication-01.jsonl \
  --output-dir outputs/figures/autoresearch-replication-01
```

The next code contribution should be an adapter for the actual OpenEvolve
checkpoint/database format. Build it from a real raw export; do not design that
parser from guessed field names.
