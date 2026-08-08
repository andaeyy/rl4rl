# 10-by-4 MPS Pilot: Iteration Results and Implications

## Result summary

The pilot ran four architecture-search setups for 10 proposal iterations each.
The evaluator initialized and trained each seed and proposal from scratch on
MPS for 10 steps and 160 examples.

- All 40 proposal iterations completed.
- All 44 training jobs succeeded: four seed evaluations and 40 proposals.
- All runtime and Transformer-validity checks passed.
- Public exact-match accuracy, development exact-match accuracy, and search
  score were `0.0` for every candidate.
- The 40 proposals contained 31 unique normalized executable architectures.

The tables report best development loss as a diagnostic; lower is better. The
controllers did not use development loss as their search score, so it did not
control acceptance, archive replacement, or parent selection.

| Setup | Valid proposals | Unique normalized proposals | Parameter range | Lowest development loss | Search outcome |
|---|---:|---:|---:|---:|---|
| Greedy Autoresearch | 10/10 | 8/10 | 6,080-137,984 | 7.377, iteration 3 | Accepted all 10 |
| Semantic Autoresearch | 10/10 | 9/10 | 6,080-11,328 | 9.203, iterations 1 and 3 | Built 3 archive cells |
| Generic OpenEvolve | 10/10 | 9/10 | 6,080-9,155 | 7.188, iteration 8 | Seed retained after score tie |
| Semantic OpenEvolve | 10/10 | 8/10 | 6,000-9,154 | 5.836, iteration 6 | Seed retained; 2 configured MAP cells |

The common seed was a two-layer causal decoder with 6,080 parameters. Its best
development loss was 9.751 and its exact-match accuracy was `0.0`.

## Experiment 1: Greedy Autoresearch

Greedy Autoresearch used one incumbent. Its mechanics-only rule accepted each
valid proposal, so every iteration became the parent path for later changes.
All rows had public accuracy and search score `0.0`.

| Iteration | Candidate architecture | Parameters | Best dev loss | Controller result |
|---:|---|---:|---:|---|
| Seed | Conventional two-layer decoder | 6,080 | 9.751 | Initial parent |
| 1 | Four-stage causal decoder, depth 2 to 4 | 11,328 | 9.203 | Accepted |
| 2 | Four-stage gated-carry decoder, GELU to gated SiLU | 14,400 | 9.540 | Accepted |
| 3 | Twelve-stage gated-carry decoder, depth 4 to 12 | 41,536 | **7.377** | Accepted |
| 4 | Six-stage parallel alignment/carry attention | 137,600 | 14.883 | Accepted |
| 5 | Serial alignment-then-carry attention | 125,696 | 13.057 | Accepted |
| 6 | Parallel alignment/carry fusion | 137,984 | 14.870 | Accepted |
| 7 | Serial alignment-then-carry attention | 125,696 | 13.057 | Accepted; duplicate of iteration 5 |
| 8 | Serial alignment/carry with sinusoidal positions | 124,576 | 10.752 | Accepted |
| 9 | Serial alignment/carry with learned positions | 125,696 | 13.057 | Accepted; duplicate of iterations 5 and 7 |
| 10 | Serial alignment/carry with hybrid learned and sinusoidal positions | 125,696 | 11.389 | Accepted |

Iterations 5, 7, and 9 shared normalized architecture hash
`ebffa61a9e75`. The controller trained the same executable design three times
under different proposal documents.

Iteration 3 had the lowest development loss in this setup. The controller then
accepted iteration 4, whose loss increased from 7.377 to 14.883, because both
candidates had the same zero-valued search score. The final candidate ended at
11.389, above the iteration 3 diagnostic result.

## Experiment 2: Semantic Autoresearch

Semantic Autoresearch selected parents through a semantic archive. It preserved
one incumbent per descriptor cell and retained tied candidates only when they
opened a new cell. All rows had public accuracy and search score `0.0`.

| Iteration | Candidate architecture | Parameters | Best dev loss | Archive result |
|---:|---|---:|---:|---|
| Seed | Conventional two-layer decoder | 6,080 | 9.751 | Base cell |
| 1 | Four-stage causal refinement, depth 2 to 4 | 11,328 | **9.203** | Same cell; incumbent preserved |
| 2 | Three-stage causal decoder, depth 2 to 3 | 8,704 | 9.382 | Same cell; incumbent preserved |
| 3 | Four-stage carry refinement, depth 2 to 4 | 11,328 | **9.203** | Same cell; duplicate of iteration 1 |
| 4 | Two-layer gated-SiLU decoder | 7,616 | 10.021 | **New feed-forward cell** |
| 5 | Concat-projected positional skip | 8,128 | 10.058 | Same gated cell; incumbent preserved |
| 6 | Heterogeneous parallel attention | 9,664 | 10.000 | Same gated cell; incumbent preserved |
| 7 | Sinusoidal positions with gated SiLU | 7,056 | 10.772 | **New positional cell** |
| 8 | Learned input/output feature gate | 7,072 | 11.430 | Same positional/gated cell; incumbent preserved |
| 9 | Softmax-routed parallel attention | 8,082 | 10.432 | Same positional/gated cell; incumbent preserved |
| 10 | Fixed-mix dual attention routes | 8,080 | 10.409 | Same positional/gated cell; incumbent preserved |

Iterations 1 and 3 shared normalized architecture hash `0b8c1ec6cda9`
and produced the same parameter count and loss.

The archive ended with three cells:

1. Conventional feed-forward and learned positions.
2. Gated-SiLU feed-forward and learned positions.
3. Gated-SiLU feed-forward and sinusoidal positions.

Only iterations 4 and 7 increased coverage. Depth, skip-fusion, and routed
attention proposals collapsed into an occupied descriptor cell despite their
graph changes.

## Experiment 3: Generic OpenEvolve

Generic OpenEvolve distributed programs across four islands and sampled
several earlier descendants as parents. All proposals had public accuracy
`0.0`, search score `0.0`, and combined score `2.0`. The combined-score rule
awarded two points when a candidate passed the joint eligibility gate. Under
the pilot's zero eligibility threshold, that gate required successful execution
and Transformer validity.

| Iteration | Parent | Candidate architecture | Parameters | Best dev loss | Result |
|---:|---|---|---:|---:|---|
| Seed | None | Conventional two-layer decoder | 6,080 | 9.751 | Best retained after tie |
| 1 | Seed | Depth-routed shallow/deep state mix | 6,082 | 9.900 | Valid and eligible |
| 2 | Seed | Embedding/shallow/deep state mix | 6,083 | 10.113 | Valid and eligible |
| 3 | Seed | Embedding/shallow/deep state mix | 6,083 | 10.113 | Normalized duplicate of iteration 2 |
| 4 | Seed | Parallel routed global/partitioned attention | 7,106 | 9.640 | Valid and eligible |
| 5 | Seed | Parallel one-head/four-head attention | 7,616 | 9.818 | Valid and eligible |
| 6 | Iteration 2 | Parallel attention fusion in both blocks | 9,155 | 10.163 | Valid and eligible |
| 7 | Iteration 3 | Concat-projected head fusion | 7,619 | 10.163 | Valid and eligible |
| 8 | Iteration 4 | Hierarchical positional/shallow/deep fusion | 7,874 | **7.188** | Valid and eligible |
| 9 | Iteration 1 | Operand-anchored depth routing | 6,594 | 7.933 | Valid and eligible |
| 10 | Iteration 6 | Sequential one-head then four-head refinement | 8,195 | 9.996 | Valid and eligible |

Iterations 2 and 3 shared normalized architecture hash `25a834a4e235`.
All 10 proposals also shared the same nine-axis semantic signature, even though
their graphs used different routing and fusion mechanisms.

Iteration 8 produced the lowest development loss in this setup, followed by
iteration 9. The population could not favor either candidate because their
exact-match and search scores tied the seed at zero. The database kept 11
programs, while the seed remained the reported best.

## Experiment 4: Semantic OpenEvolve

Semantic OpenEvolve combined four islands with semantic MAP-Elites cells. All
proposals had public accuracy `0.0`, search score `0.0`, and combined score
`2.0`. The combined-score rule awarded two points when a candidate passed the
joint eligibility gate. Under the pilot's zero eligibility threshold, that
gate required successful execution and Transformer validity.

| Iteration | Parent | Candidate architecture | Parameters | Best dev loss | Semantic/archive result |
|---:|---|---|---:|---:|---|
| Seed | None | Conventional two-layer decoder | 6,080 | 9.751 | Base MAP cell; best retained after tie |
| 1 | Seed | Parallel routed one-head/four-head attention | 7,106 | 9.623 | Base semantic signature |
| 2 | Seed | Heterogeneous parallel attention routing | 7,106 | 9.640 | Base semantic signature |
| 3 | Seed | Parallel attention routing | 7,106 | 9.640 | Normalized duplicate of iteration 1 |
| 4 | Seed | Heterogeneous parallel attention routing | 7,106 | 9.640 | Exact duplicate of iteration 2 |
| 5 | Seed | Gated-SiLU carry decoder | 7,616 | 10.021 | New full signature; configured MAP cell unchanged |
| 6 | Iteration 2 | Concat-projected residual fusion | 9,154 | **5.836** | Base configured MAP cell |
| 7 | Iteration 3 | Learned plus sinusoidal positional fusion | 7,106 | 10.516 | **New configured positional cell** |
| 8 | Iteration 4 | Cross-depth shallow/deep fusion | 7,634 | 6.972 | Base configured MAP cell |
| 9 | Seed | RMS-normalized causal decoder | 6,000 | 10.522 | New full signature; configured MAP cell unchanged |
| 10 | Iteration 6 | Three-layer fixed depth routing | 8,704 | 9.651 | Base configured MAP cell |

Iterations 1 and 3 shared normalized architecture hash `454a57ed043c`.
Iterations 2 and 4 shared the same raw graph and normalized architecture hash
`dd29fda98585`, so iteration 4 repeated an identical training job.

The run produced four full nine-axis semantic signatures:

1. Base descriptor signature.
2. Gated-SiLU feed-forward signature from iteration 5.
3. Dual-basis positional signature from iteration 7.
4. RMS-normalization signature from iteration 9.

The configured online archive used only token representation, positional
integration, attention organization, and depth topology. That four-axis grid
counted two MAP cells. Feed-forward and normalization changes did not change
those four coordinates.

Iteration 6 produced the lowest development loss in the full 10-by-4 pilot at
5.836. Iteration 8 reached 6.972. The zero-valued search score prevented these
diagnostic differences from changing the reported best program.

## Cross-experiment implications

### 1. The training budget supplied no architecture-ranking signal

All 44 models ended with zero exact-match accuracy. Greedy accepted each valid
proposal, Semantic Autoresearch used cell novelty, and both OpenEvolve runs
retained the seed after score ties. None of those outcomes measures addition
quality.

The development losses contain a weak diagnostic signal:

- Greedy iteration 3: 7.377 versus seed 9.751.
- Semantic Autoresearch iterations 1 and 3: 9.203 versus seed 9.751.
- Generic OpenEvolve iteration 8: 7.188 versus seed 9.751.
- Semantic OpenEvolve iteration 6: 5.836 versus seed 9.751.

Ten training steps cannot establish that these candidates will reach higher
exact-match accuracy. They identify candidates for longer controlled retraining.
Development loss should remain diagnostic unless the scientific protocol
defines it as controller-visible feedback before launch.

### 2. Greedy search drifted after its strongest diagnostic result

Greedy iteration 3 had its lowest development loss. The controller then
accepted seven more zero-score ties, expanded to 137,984 parameters, and ended
with development loss 11.389. Its parameter peak was 22.7 times the seed size.

This path shows the consequence of a mechanics-only acceptance rule: later
valid changes replace earlier candidates without evidence of better task
behavior. It does not show that large models perform worse. Ten equal optimizer
steps also provide different amounts of compute for 6,080 and 137,984
parameters.

### 3. Semantic archives preserved mechanism coverage under score ties

Semantic Autoresearch retained three families instead of following one chain.
That behavior kept gated feed-forward and positional mechanisms available after
the accuracy score flattened.

The descriptors also compressed many changes. Semantic Autoresearch placed
depth, skips, and routed attention into existing cells. Semantic OpenEvolve
created four full signatures but only two cells on its configured axes. Archive
axis selection controls which forms of novelty receive protection.

### 4. Population structure could not replace a missing fitness signal

OpenEvolve sampled several parents and produced descendants through three
generations. Both runs still retained their seeds as best because all scores
tied. Islands preserved lineages, but they could not identify which lineage
learned addition better.

Generic OpenEvolve's single semantic signature also shows a descriptor gap:
routing, fusion, and attention-granularity changes all received the base code.
The semantic representation needs axes for mechanisms that agents change in
practice.

### 5. Duplicate proposals consumed six within-run opportunities

Each setup repeated at least one normalized architecture:

| Setup | Duplicate proposal slots | Duplicate relationship |
|---|---:|---|
| Greedy Autoresearch | 2 | Iterations 5, 7, and 9 were one executable architecture |
| Semantic Autoresearch | 1 | Iterations 1 and 3 matched |
| Generic OpenEvolve | 1 | Iterations 2 and 3 matched |
| Semantic OpenEvolve | 2 | Iterations 1/3 matched; iterations 2/4 matched |

Within-run deduplication would have prevented six redundant training jobs. The
four setups also overlapped with one another, leaving 31 unique normalized
architectures among 40 proposals. Cross-setup overlap can support controlled
comparison, so the study protocol should decide whether to cache those results
or run fresh training for each setup.

### 6. Parameter count described search behavior, not quality

Greedy reached 137,984 parameters, while both OpenEvolve variants stayed below
10,000. Semantic Autoresearch also stayed near the seed scale. The controllers
received no size reward or penalty.

The zero exact-match scores support no relationship between parameter count and
quality. Future runs should keep parameter count as metadata and control
training through frozen examples, optimizer steps, wall-time limits, and
reported realized compute.

## Recommended next experiment

1. Calibrate a training horizon where the conventional baseline reaches
   nonzero, stable exact-match accuracy across more than one seed.
2. Freeze that training and evaluation treatment before comparing controllers.
3. Reject a normalized architecture duplicate before training while charging
   the proposal opportunity under a predeclared rule.
4. Add semantic axes for feed-forward type, normalization, and routing/fusion,
   or state why the archive excludes them.
5. Repeat the same 10-iteration comparison before increasing the proposal
   budget.
6. Retrain promising candidates, including Greedy iteration 3, Generic
   OpenEvolve iteration 8, and Semantic OpenEvolve iterations 6 and 8, with
   independent seeds.

## Interpretation boundary

This engineering pilot establishes that all four setups can generate valid
Architecture IR, train fresh weights on MPS, and retain iteration records. The
ten-step treatment produced no exact-match ranking, so the run does not support
claims about architecture superiority or controller effectiveness.

## Local raw artifacts

Git excludes the raw run directories because they contain checkpoints and
provider transcripts. The machine that ran the pilot retains the artifacts
under `outputs/engineering_10x4/`, with one subdirectory for each setup. A
GitHub clone contains this report, not those local artifacts.
