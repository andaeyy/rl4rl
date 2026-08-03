# Confirmatory Protocol Amendment

## Valid Architectural Discovery in Autonomous Research Systems

**Status:** proposed preregistration amendment  
**Applies to:** `PROJECT_DIRECTION_AND_PAPER_ROADMAP.md`  
**Decision:** do not start the main study until every item in the readiness gate passes  
**Scientific unit:** one complete search run  
**Benchmark role:** AdderBoard is a controlled synthetic testbed, not evidence of broad scientific invention

---

# 1. Research claim

## Confirmatory question

Under a fixed generator, prompt-information budget, starting program, execution environment, and total search budget, how do:

1. portfolio memory versus one-incumbent memory, and
2. a scheduled architecture-transition operator versus an ordinary mutation operator

affect the number of distinct, valid, post hoc mechanism clusters that an autonomous research system discovers?

## Causal estimands

Let \(M\in\{0,1\}\) denote portfolio memory and \(T\in\{0,1\}\) denote the transition operator. Let \(Y_r(B)\) count distinct valid mechanism clusters in run \(r\) by budget \(B\).

The study estimates:

\[
\tau_M(B)=E[Y(B)\mid M=1]-E[Y(B)\mid M=0]
\]

\[
\tau_T(B)=E[Y(B)\mid T=1]-E[Y(B)\mid T=0]
\]

\[
\tau_{MT}(B)=
\{E[Y(B)\mid M=1,T=1]-E[Y(B)\mid M=1,T=0]\}
-
\{E[Y(B)\mid M=0,T=1]-E[Y(B)\mid M=0,T=0]\}
\]

The confirmatory estimand uses a fixed total generator-token budget. Candidate-count and accelerator-hour analyses serve as sensitivity analyses.

## Permitted claim

If the confidence interval supports a positive effect, the paper may claim:

> Under the frozen AdderBoard protocol, portfolio memory or the scheduled transition operator increased the yield of distinct valid mechanism clusters relative to the matched control.

The paper must not claim that the system invented a mechanism, demonstrated general scientific creativity, or found a design absent from all prior work.

---

# 2. Separate three questions

The current roadmap mixes three forms of evidence. The revision assigns each form a separate status.

| Question | Evidence | Status |
|---|---|---|
| Does the search intervention change behavior? | randomized C0-C3 experiment | confirmatory |
| Does the intervention increase descriptor coverage? | online archive and post hoc descriptor statistics | manipulation check |
| Did a candidate use a corpus-absent mechanism? | blinded review, causal tests, clean-room reproduction, corpus audit | exploratory scientific case study |

Descriptor coverage cannot serve as a primary outcome because the portfolio treatment uses the same descriptors to drive search. Such an outcome would reward the treatment for satisfying its own control signal.

Retention and reuse rates also cannot serve as evidence that portfolio memory works. The controller forces portfolio systems to retain branches that a one-incumbent system cannot retain. Report those rates as mechanism checks.

---

# 3. Clean 2 by 2 design

## Treatment factors

### Factor M: memory topology

- \(M=0\): keep one active parent.
- \(M=1\): keep \(K\) active parents.

Both levels use the same candidate record, quality gate, parent-quality score, mutation operator, prompt template, and feedback fields. The treatment changes the number of live parents.

### Factor T: proposal operator

- \(T=0\): request one architecture proposal under the task rules.
- \(T=1\): request one proposal that changes one stated architectural assumption.

Apply \(T=1\) on a fixed schedule, such as every fifth proposal opportunity. Do not trigger it after measured stagnation. An outcome-dependent trigger changes treatment exposure across runs and complicates the causal contrast.

## Four conditions

| Condition | Memory | Proposal operator |
|---|---|---|
| C0 | one parent | ordinary |
| C1 | one parent | scheduled transition |
| C2 | \(K\)-parent portfolio | ordinary |
| C3 | \(K\)-parent portfolio | scheduled transition |

## Remove these confounds

- Use the same prompt schema and slot count in all four conditions. In one-parent conditions, mark unused parent slots and match the token budget with frozen neutral text. Do not expose rejected history to those conditions.
- Use the same within-parent quality rule. Do not use “accept any passing child” for one condition and “replace only on higher robustness” for another.
- Match prompt tokens by padding with task-relevant neutral material or cap total generator tokens at the run level.
- Keep the full-rewrite operator out of the main factorial. Test it in a separate randomized experiment.
- Keep native Autoresearch and OpenEvolve comparisons out of the confirmatory estimate. Report them as system replications.

## Rename the memory factor if descriptors drive parent selection

If \(M=1\) samples underexplored descriptor cells, the treatment combines portfolio memory with descriptor-guided exploration. Call the factor **descriptor-guided portfolio search**. Do not interpret its effect as memory alone.

---

# 4. Randomization and run isolation

## Randomization

Create blocks before execution. Each block contains one run from C0, C1, C2, and C3.

For each block:

1. assign a shared starting artifact and data-generation seed;
2. randomize condition order;
3. run conditions across the same provider and hardware window;
4. rotate order across blocks with a balanced schedule.

The shared seed acts as a blocking variable. It does not make language-model calls deterministic.

## Isolation

Each run starts from:

- a new output directory;
- an empty cache;
- a new process tree;
- the same read-only starting artifact;
- no prior run ledger, prompt, response, archive, or checkpoint.

Record the exact model snapshot, API mode, retry behavior, request identifiers, timestamps, dependency lock hash, evaluator image hash, host telemetry, and controller commit.

If the provider does not expose an immutable model snapshot, the paper must identify provider drift as a validity threat. Blocked scheduling reduces that threat but does not remove it.

---

# 5. Evaluation firewall

## Three disjoint evaluation layers

### Layer A: adaptive search feedback

The controller may query Layer A for every candidate. Layer A contains:

- public training data;
- public development cases;
- contract failures;
- bounded error summaries;
- search-time robustness checks.

Layer A determines retention and parent selection.

### Layer B: sealed qualification

A trusted evaluation service runs Layer B at frozen checkpoints, such as budgets 50, 100, and 200. The controller receives no case-level failures, scores, or pass/fail signal from Layer B. Layer B determines which candidates enter novelty annotation after the run ends.

### Layer C: untouched confirmation

After data freeze and candidate selection, an independent evaluator runs Layer C once per selected mechanism cluster. Layer C contains:

- fresh random seeds;
- stratified carry patterns;
- length shifts;
- digit-symbol permutations;
- formatting perturbations;
- numerical-precision checks.

Do not use Layer C to tune, rerun, select, or repair a candidate.

## Statistical qualification

Preserve the AdderBoard benchmark rule of at least 99% on its 10,000 fixed-seed cases as a benchmark label.

For scientific claims, report:

- exact successes and trials;
- an exact or Wilson confidence interval;
- stratum-level results;
- the Layer C result.

An observed score of 99% on 10,000 random cases does not establish that population accuracy exceeds 99%. Define the scientific threshold through a lower confidence bound or require exact correctness on a finite, enumerated test domain.

## No adaptive use of hidden outcomes

The controller, generator, archive, and retention rule must never read Layer B or Layer C results. A hidden seed does not protect a test set if its aggregate score enters every search decision.

---

# 6. Candidate containment and validity

Generated code is untrusted.

Run each candidate in a disposable sandbox with:

- no network;
- no access to evaluator source, seeds, expected outputs, other candidates, or host files;
- a read-only candidate package;
- a fixed dependency allowlist;
- blocked subprocess creation unless the protocol requires it;
- CPU, memory, file-size, and wall-time limits;
- captured system calls and file accesses;
- a fresh process for each evaluation.

## Transformer validity

Static name checks cannot establish transformer validity. Use:

1. an allowlisted intermediate representation or module graph;
2. runtime hooks that verify causal self-attention receives the token prefix;
3. a generic decoder supplied by the evaluator;
4. source and bytecode scans for answer computation and test access;
5. metamorphic tests that change digit symbols, positions, lengths, and batch order;
6. adversarial red-team candidates written before the pilot.

Require the candidate to expose a model and tokenizer. The evaluator owns the `add` wrapper and autoregressive loop. This removes a large Python surface that can solve the task outside the model.

---

# 7. Outcome definitions

## Primary outcome

At budget \(B^\*\), count distinct valid mechanism clusters per run:

\[
Y_r(B^\*) =
\left|
\operatorname{Cluster}
\left(
\{v: v\in r,\ Q_B(v)=1,\ t(v)\le B^\*\}
\right)
\right|
\]

Here \(Q_B\) uses Layer B qualification. Freeze the clustering rule before the main study.

Count one mechanism once per run. Descendants, parameter variants, and code refactors do not add to the count.

## Key secondary outcome

The probability that a run produces at least one mechanism cluster that:

1. passes Layer B;
2. reproduces from a clean implementation;
3. passes Layer C;
4. receives an N3 or N4 label under blinded review.

## Manipulation checks

- proposal-level family transitions;
- valid descriptor coverage;
- portfolio occupancy;
- lineage entropy;
- retention and reuse of alternative families.

## Diagnostic outcomes

- proposal-to-code survival;
- execution and validity failure rates;
- generator tokens;
- accelerator-hours;
- wall time;
- infrastructure failures;
- verifier exploit attempts;
- training-seed reproduction rate.

## Novelty labels

Revise the labels:

- N0: equivalent to a reference instance.
- N1: parameter or scale variant.
- N2: corpus-absent recombination of known mechanisms.
- N3: distinct causal mechanism that uses known primitives.
- N4: no matching mechanism found in the frozen corpus or the post-study audit.
- X: unresolved.

Call N3 a **mechanism distinction**, not confirmed novelty. Reserve **corpus-absent mechanism candidate** for N4.

---

# 8. Annotation and mechanism review

## Annotation sample

Annotate every Layer B-qualified mechanism cluster. If cost prevents full review, sample clusters with a frozen probability and use inverse-probability weights. Do not select cases by condition, aesthetic appeal, or final score.

## Review process

Use at least three reviewers for N3 and N4 decisions.

Mask:

- condition;
- controller;
- parameter count;
- run outcome;
- candidate ancestry;
- author identity.

Reviewers receive canonicalized code, module graphs, behavior probes, and nearest-corpus packets in randomized order.

## Agreement

Report:

- the full confusion matrix;
- weighted Krippendorff alpha or weighted kappa;
- bootstrap confidence intervals;
- raw labels before adjudication;
- the fraction marked X.

Do not use a bare 0.67 cutoff as proof of construct validity. Calibrate the codebook on examples that do not enter the confirmatory study.

## Mechanism evidence ladder

An N3 or N4 dossier must contain:

1. a falsifiable causal account;
2. component ablations with matched retraining controls;
3. interventions that distinguish the account from its closest alternatives;
4. minimal counterexample sets;
5. a clean-room implementation by a person who did not inspect the original code;
6. reproduction across frozen training seeds;
7. Layer C results;
8. a corpus search log and post-study literature audit.

Attention plots and linear probes may support a dossier. They cannot establish the mechanism without interventions.

---

# 9. Power and sample size

## Pilot

Use the pilot to estimate:

- the run-level mean and variance of mechanism-cluster yield;
- the zero rate;
- the intrablock correlation;
- the cost distribution;
- the Layer B qualification rate;
- annotation load.

Exclude pilot runs from the confirmatory analysis after any codebook, evaluator, treatment, or outcome change.

## Simulation-based power

Before the main study:

1. fit candidate count models to masked pilot labels;
2. simulate the planned blocked factorial;
3. set a smallest effect of interest;
4. choose a fixed run count that attains the target power or interval precision;
5. publish the simulation code and decision rule.

“At least 10 to 12 seeds” is not a sample-size rule. Rare binary novelty outcomes can require far more runs. If feasible power exceeds the budget, make valid mechanism-cluster yield the confirmatory outcome and keep N4 evidence as case studies.

---

# 10. Statistical analysis plan

## Primary model

Analyze run-level mechanism-cluster counts with a blocked negative-binomial or Poisson model:

\[
\log E[Y_r] =
\beta_0+\beta_M M_r+\beta_T T_r+\beta_{MT}M_rT_r+\alpha_{\text{block}(r)}
\]

Use a negative-binomial model if pilot dispersion warrants it. Specify the decision before unmasking main-study outcomes.

Report:

- marginal mean differences;
- rate ratios;
- 95% confidence intervals;
- randomization-inference \(p\)-values as a robustness check.

Test the memory main effect and transition main effect as the two confirmatory contrasts. Treat the interaction as a key secondary contrast unless power calculations support it.

## Time-to-event outcomes

Use proposal opportunity or cumulative generator tokens as the time scale. Apply discrete-time survival models with right censoring. Wall-clock time belongs in a resource analysis because provider and hardware congestion affect it.

## Failed runs

Define two failure classes before execution:

- infrastructure failure: provider outage, host crash, corrupted storage;
- scientific failure: invalid code, timeout caused by the candidate, exhausted budget.

Rerun infrastructure failures under a frozen rule. Count scientific failures in the assigned condition. Report intention-to-treat results and a sensitivity analysis that treats unresolved infrastructure failures as worst-case outcomes.

## Multiple testing

Keep two confirmatory contrasts. Control the false discovery rate across the named secondary family. Label all other analyses exploratory.

## Mediation

Do not claim that coverage causes novelty from a correlation between coverage and N3 or N4 yield. A mediation claim requires stronger assumptions or a separate randomized intervention.

---

# 11. Reference corpus and contamination

Freeze:

- sources;
- search queries;
- databases;
- cutoff date;
- inclusion criteria;
- duplicate rules;
- descriptor codebook;
- corpus hashes.

After the main study, run a second literature search with the same protocol and a new cutoff date. Report whether any N4 label changes.

Hiding the corpus from the agent does not show that the base model lacked exposure during pretraining. Phrase the result as system-level generation relative to a corpus. Add a no-search baseline that samples independent solutions from the same model and token budget. This estimates how much the search process adds beyond model recall and one-shot generation.

---

# 12. External validity

AdderBoard offers a useful controlled environment, but one arithmetic task cannot support claims about architecture discovery as a general capability.

Choose one:

1. narrow the paper title and claims to autoregressive arithmetic architectures; or
2. preregister a second task with a different computational bottleneck and the same evaluation firewall.

A strong Hazy-style extension would test whether a discovered mechanism predicts a scaling law across task parameters. For addition, vary:

- number length;
- carry-chain length;
- base;
- vocabulary size;
- number of simultaneous arithmetic queries.

Fit accuracy or required width as a function of those variables. A mechanism that explains scaling behavior carries more scientific value than a code diff that passes one 10-digit benchmark. This follows Hazy work that uses synthetic tasks, controlled scaling, and theory to explain architecture behavior.

---

# 13. Reproducibility package

Release:

- preregistration and amendments;
- controller and evaluator containers;
- dependency locks and source hashes;
- run randomization table;
- all prompts, responses, code, failures, and lineage records;
- raw immutable event logs;
- analysis scripts;
- power simulations;
- annotation packets and raw labels;
- mechanism dossiers;
- a model and data card;
- a cost and energy report.

Keep Layer C secret until the paper decision if public release would compromise follow-up evaluation. Commit its generator and hash before the main study.

---

# 14. Readiness gate

Do not launch the paid pilot until all items pass.

- [x] Common C0-C3 controller exists in one orchestration layer.
- [x] Each treatment differs only in its assigned factor.
- [ ] Run order randomization and blocking table are frozen.
- [ ] Layer A, B, and C data generators are disjoint and hashed.
- [x] Layer B and C cannot enter online fitness, feedback, or retention.
- [ ] Candidate sandbox passes adversarial escape and cheating tests.
- [x] Evaluator owns fresh candidate initialization, fixed training data,
      optimization, public-development checkpoint selection, and the generic
      autoregressive decoder.
- [ ] Descriptor and mechanism codebooks pass calibration on held-out examples.
- [ ] Primary mechanism clustering rule is frozen.
- [ ] Pilot exclusion and amendment rules are written.
- [x] Simulation-based sample-size rule is implemented.
- [x] Analysis code runs on synthetic null and known-effect data.
- [x] Offline reconstruction from immutable synthetic artifacts is implemented
      and tested.
- [ ] An independent external reviewer has reproduced a frozen run package.

---

# 15. Remaining implementation and evidence blockers

The primary offline infrastructure now exists, but the confirmatory study is
not launch-ready.

1. Arbitrary Python is still not proven contained by an OS boundary on the real
   MPS host. Static scanning and credential scrubbing remain defense in depth.
2. The typed architecture IR and runtime probes exist, but a trusted
   evaluator-owned interpreter is not connected to scientific evaluation
   records.
3. `full_train_v1` has not completed in an MPS-available process, and CPU smoke
   results cannot satisfy this gate.
4. Layer B/C sources, counts, thresholds, custody, and release artifacts are
   not frozen. Smoke-scale profiles are rejected rather than silently reused.
5. The reference corpus, independent reviewer roster, mechanism plan,
   replication policy, final analysis plan, and external-validity plan require
   scientific decisions and real evidence.
6. Local hash chains and content-addressed indexes are implemented, but their
   heads are not yet retained by a signed, WORM, or independent external
   authority.
7. The no-search infrastructure is feedback-free offline and its injected-client
   real-provider adapter exists, but the frozen scientific assignment and paid
   pilot artifact path are not integrated.
8. The main study remains blocked until the paid pilot is reconstructed and its
   estimates are used to freeze the final power and analysis plan.
9. Scientific receipts now reject truthiness and numeric coercion, cross-link
   the study, protocol, mechanism, replication, analysis, MPS, and pilot
   artifacts, and require explicit PI launch authorization. The external-anchor
   signature verifier and actual custodian receipt remain unimplemented.

Online shadow scoring, smoke-size scientific evaluation, native-controller
confounding, descriptor-based novelty claims, rehashed randomization attacks,
type-confused booleans and counts, active-state treatment tampering,
canonical-cluster relabeling, indirect capability recovery, non-finite budget
bypasses, checkpoint step coercion, and syntactic MPS receipts now have hard
regression coverage.

---

# 16. Paper positioning

The strongest paper studies why autonomous architecture search loses or preserves valid mechanisms.

The contribution should combine:

- a controlled causal comparison of search organization;
- a stage-resolved trajectory dataset;
- an evaluation firewall for adaptive research agents;
- one or more mechanism dossiers that connect architecture to scaling behavior.

That package fits the Hazy pattern of joining systems instrumentation with an explanation of model behavior. A controller leaderboard would offer less scientific value.

---

# 17. Research precedents

- [Zoology: Measuring and Improving Recall in Efficient Language Models](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis) motivates controlled synthetic tasks, scaling studies, and theory-backed mechanism claims.
- [Efficient language models as arithmetic circuits](https://hazyresearch.stanford.edu/blog/2024-06-22-ac) frames architecture comparison through the computations that model families can implement.
- [A Paradigm Shift in ML Validation: Evaluating Workflows, Not Tasks](https://hazyresearch.stanford.edu/blog/2023-08-21-workflows) supports evaluation at the research-workflow level.
- [AI Agents That Matter](https://arxiv.org/abs/2407.01502) motivates sealed holdouts, cost reporting, and reproducible agent evaluation.
- [CORE-Bench](https://arxiv.org/abs/2409.11363) treats computational reproducibility as a prerequisite for research-agent claims.
