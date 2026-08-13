# Beyond Local Search

## Project Direction and Paper Roadmap: Valid Architectural Novelty in Autonomous Discovery

**Working title:** *Beyond Local Search: Measuring Valid Architectural Novelty in Autonomous Discovery*

**Document status:** revised research roadmap; no completed-findings claim  
**Confirmatory amendment:** `RIGOROUS_EXPERIMENT_PLAN_V2.md` supersedes the causal design, evaluation, outcome, power, and launch-readiness provisions below  
**Primary environment:** AdderBoard as an executable validity and accuracy test  
**Primary target:** valid architectural mechanisms that differ from known solution families  
**Non-objective:** model-size minimization

---

## July 2026 implementation status (superseded execution snapshot)

This section records the pre-Modal execution condition in the past tense. Its
MPS-specific execution and evidence statements are superseded by the August
2026 amendment immediately below; they are not current launch blockers.

Evaluator-owned candidate training had been implemented: architecture
candidates returned a seeded, untrained CPU model; trusted code owned the fixed
Phase-1 task adapter, deterministic public data, AdamW training,
public-development checkpoint selection, then-strict MPS selection, resume
hashes, and evaluator-owned decoding. The pretrained vendor checkpoint was
isolated to a regression test.

Generated candidates ran in a credential-scrubbed subprocess, but this was not
a complete filesystem or network sandbox. The common C0-C3 engine, typed Layer
A/B/C firewall, comparable budget ledger, blocked randomization, then-sequential
MPS scheduler, no-search control, novelty/mechanism/replication/statistics
infrastructure, adversarial suite, immutable event ledger, reconstruction, and
reporting had been implemented and offline-tested. At that time, scientific
execution remained blocked by the trusted IR/OS boundary, full-profile evidence
on the then-canonical MPS condition, external artifact anchoring, frozen
corpus/reviewer/policy artifacts, unresolved PI decisions, scientific no-search
and Layer B orchestration, explicit PI authorization, and a completed pilot.
Scientific receipts were exact-typed and cross-linked, but the external-anchor
signature verifier remained an open gate. This historical implementation status
does not authorize a paid run.

Any later reference in this historical roadmap to an online "shadow" score is
superseded: only public Layer A information may adapt search. Layer B is sealed
post-run qualification, and Layer C is one-shot confirmation.

## August 2026 execution-condition amendment

Modal with NVIDIA CUDA is canonical for new remote engineering runs and the
future scientific execution profile. `full_train_cuda_v2` and
`smoke_train_cuda_v2` are new, separately hashed conditions; they do not reuse
or imply equivalence with `full_train_v1` or `smoke_train_v1` on MPS. Active
resource accounting and scheduling are accelerator-neutral and remain strictly
sequential. Historical MPS records and receipts remain readable without hash
rewriting.

This amendment changes execution infrastructure only. It does not fill any PI
decision, authorize paid provider work, prove arbitrary-Python containment, or
relax the protocol, evaluation firewall, custody, external-anchor, pilot, or
launch gates below. Live Modal CUDA validation and post-run cleanup evidence
must be recorded before Modal infrastructure can be considered validated.

---

# 1. Scientific objective

## Central question

Can an autonomous research system propose, implement, and validate architectural mechanisms that fall outside its starting family and a frozen corpus of known designs?

The study examines the research process rather than a compression leaderboard. AdderBoard supplies three useful properties:

- executable correctness tests
- a clear autoregressive-transformer validity rule
- a task that requires digit alignment, per-digit computation, and carry propagation

Parameter count does not define success in this study. The experiment records parameter count as metadata because architecture scale can affect training and compute. The prompt, selection rule, archive, stopping rule, and paper claims must not reward lower parameter counts.

## Role of AdderBoard

AdderBoard acts as a staged feasibility environment. Online eligibility uses
public Layer A only:

\[
Q_A(v)=\mathbf{1}[
\operatorname{valid\_transformer}(v)
\land A_{\mathrm{public}}(v)\geq \tau_A
]
\]

A candidate that fails \(Q_A(v)\) cannot become a search parent. After the run
is frozen, sealed Layer B determines qualification for novelty and mechanism
review. Layer C confirms only the preregistered representative of a selected
mechanism cluster and never returns information to search.

The study does not use:

- leaderboard rank as a scientific outcome
- parameter thresholds as success criteria
- parameter reduction as a reward
- public frontier solutions as targets
- smaller size as a tie-breaker

## Intended contribution

The paper should contribute:

1. a stage-resolved account of where architectural ideas disappear between proposal and validation
2. a reproducible test of whether search structure changes the rate of valid architectural discovery
3. a validated method for distinguishing family changes, recombinations, and mechanism-level novelty

## Calibrated claim

Use the phrase **architectural novelty relative to a frozen reference corpus**. No experiment can establish novelty relative to all past and unpublished work.

The strongest warranted claim would be:

> Under matched models, prompts, starting code, and evaluation budgets, a descriptor-aware search process produced more valid architecture families and more expert-confirmed mechanism candidates than incumbent-only search.

Avoid claims about general creativity, intelligence, or invention across science.

---

# 2. Definitions

## Candidate architecture

Each evaluated candidate \(v\) has:

\[
\mathcal{R}(v)=(c_v,z_v,b_v,y_v,m_v)
\]

where:

- \(c_v\): complete code and configuration
- \(z_v\): architectural descriptor vector
- \(b_v\): observed mechanism and behavior profile
- \(y_v\): execution, validity, and accuracy outcomes
- \(m_v\): provenance, prompt, parentage, tokens, time, and compute

## Architectural descriptor

\[
z(v)=(
z_{\mathrm{pos}},
z_{\mathrm{embed}},
z_{\mathrm{attn\_proj}},
z_{\mathrm{attn\_org}},
z_{\mathrm{ffn}},
z_{\mathrm{norm}},
z_{\mathrm{topology}},
z_{\mathrm{readout}},
z_{\mathrm{token}}
)
\]

Training choices receive separate labels. An optimizer or curriculum change can enable a discovery, but it does not count as an architecture-family change.

## Representational-family transition

\[
\operatorname{RFT}(p\rightarrow v)=
\mathbf{1}[z_{\mathrm{arch}}(p)\neq z_{\mathrm{arch}}(v)
\text{ on at least one frozen family axis}]
\]

RFT is a movement measure. It does not establish novelty by itself. An agent can rediscover a known family transition.

## Novelty levels

Annotators assign one of the following levels after a candidate passes the validity gate:

- **N0 Known instance:** same mechanism and family combination as a reference design
- **N1 New parameterization:** different dimensions, constants, or ranks within a known mechanism
- **N2 New recombination:** known components combined in a reference-corpus-absent configuration
- **N3 New structural mechanism:** a different information-routing or computation mechanism within known primitives
- **N4 Mechanism candidate:** the frozen corpus contains no matching mechanism, and ablations support a distinct causal account
- **X Unresolved:** evidence does not support a stable label

Primary novelty analyses:

- valid N2+ candidates
- valid N3+ candidates
- expert-confirmed N4 candidates

N4 requires more than an unfamiliar code diff. Reviewers must identify the computation the architecture performs and show that the mechanism survives independent implementation and testing.

## Architectural distance

\[
d_{\mathrm{arch}}(u,v)=
\sum_j w_j\mathbf{1}[z_j(u)\neq z_j(v)]
\lambda d_{\mathrm{struct}}(u,v)
\gamma d_{\mathrm{behavior}}(u,v)
\]

Freeze descriptor axes, weights, structural features, and behavioral probes before the main study.

---

# 3. Reference corpus and novelty assessment

## Frozen reference corpus

Build the corpus before experimental runs. Include:

- AdderBoard trained and hand-coded submissions, with category labels
- published arithmetic-transformer mechanisms
- common transformer architecture families
- relevant neural architecture search primitives
- public agent-generated arithmetic architectures available before the freeze date

Record:

- source URL and commit or archive hash
- access date
- implementation hash where code exists
- architectural descriptors
- short mechanism account
- trained or constructed status

Keep the corpus hidden from search agents. Give agents the task rules and starting repository, not a list of known winning components.

## Novelty review

Use two review stages.

### Stage A: blinded architecture annotation

Two domain-competent annotators label:

- architecture-family coordinates
- RFT status
- novelty level
- closest reference-corpus entries
- confidence and ambiguity

Mask:

- agent and harness identity
- condition
- parameter count
- final lineage success
- leaderboard similarity

### Stage B: mechanism review

For every N3 or N4 candidate:

1. write a causal account of how it aligns digits, computes digit sums, and propagates carry
2. remove or replace claimed components
3. run counterfactual inputs that separate the claimed mechanism from known alternatives
4. reproduce the design from a clean implementation
5. search the frozen corpus for the closest mechanism
6. ask a third expert to confirm or reject the novelty claim

The main paper reports both proposed and confirmed novelty labels.

## Annotation thresholds

- RFT agreement below 0.67: revise the codebook before the main study
- N2+ agreement below 0.67: revise novelty definitions
- N4 disagreement: report the candidate as unresolved

Retain raw labels and adjudication notes.

---

# 4. Research systems

## Autoresearch condition

Autoresearch represents incumbent-based sequential research:

1. inspect the current accepted candidate and research ledger
2. state a proposal and mechanism hypothesis
3. edit the candidate
4. run validity and accuracy tests
5. record the full result
6. retain or revert under the assigned selection condition

The original keep-only-if-score-improves rule does not fit this study. Replace it with a preregistered discovery rule.

### Autoresearch retention rule

Retain a candidate as the new incumbent when it passes the validity gate and a frozen robustness floor. Revert candidates that fail either threshold. Among passing candidates, accept the completed candidate without consulting its novelty label or parameter count.

This rule creates a validity-filtered architectural walk. It avoids a hidden novelty reward in the control condition.

The single-incumbent structure still creates a research constraint. One candidate supplies the code base for the next experiment, so alternative valid families can disappear when the system cannot maintain branches.

## OpenEvolve condition

OpenEvolve represents population-based research:

1. sample one lineage parent
2. sample top and diverse inspiration candidates
3. generate a diff or full rewrite
4. evaluate the child
5. assign architectural descriptors
6. place valid candidates into an archive
7. preserve multiple families and migrate candidates between islands

### OpenEvolve archive rule

Use architecture descriptors instead of generic code length or raw edit distance.

Archive cells should use a sparse set of preregistered axes:

- positional integration
- token representation
- attention organization
- feedforward mechanism
- topology

Avoid one large Cartesian grid. Maintain:

- per-axis coverage
- selected pairwise maps
- a set of complete descriptor signatures

Within a cell, prefer candidates with:

1. higher robustness
2. higher training reproducibility
3. earlier completion as a deterministic final tie-break

Do not use parameter count.

## Harness interpretation

Autoresearch and OpenEvolve differ in more than interface:

- Autoresearch uses one active lineage.
- OpenEvolve preserves a portfolio of lineages.
- Autoresearch keeps research memory in code, logs, and agent context.
- OpenEvolve stores programs, parentage, prompts, metrics, artifacts, and archive position.

The main study tests search organization. It does not rank commercial coding products.

---

# 5. Experimental conditions

Use one common orchestration layer for the primary causal experiment. Hold the base language model, snapshot, tools, starting code, evaluator, mutation format, and budget fixed. Change two factors:

- **proposal policy:** ordinary proposal or transition operator
- **research memory:** single incumbent or descriptor-aware archive

Run native Autoresearch and OpenEvolve configurations as a secondary system-level replication. This separation prevents software-package differences from carrying the main causal claim.

For the initial controller study, pin every condition to `gpt-5.6-sol` with
high reasoning effort, a 16,384-token completion ceiling, a 300-second timeout,
two retries with a three-second delay, and no temperature or top-p fields. Use
Chat Completions until both controller implementations move together to
Responses; endpoint differences must not be confounded with research-memory
differences. Pass the run seed to the API and describe it as best-effort
reproducibility.

## C0: single incumbent with ordinary proposals

- one active lineage
- task and validity rules only
- no architectural descriptors in the prompt
- accept candidates that pass validity and robustness thresholds

Purpose: estimate valid-family discovery under ordinary sequential research.

## C1: single incumbent with transition proposals

- use the same retention rule as C0
- activate a transition operator on a preregistered, outcome-independent
  opportunity schedule shared with C3
- ask the model to reconsider one abstract architectural assumption
- do not name known components or public solutions

Purpose: estimate the proposal-policy effect without population memory.

## C2: descriptor archive with ordinary proposals

- preserve valid candidates across architecture descriptor cells
- sample parents from occupied and underexplored cells
- use the task-only proposal prompt from C0
- do not expose novelty labels or the reference corpus

Purpose: estimate the research-memory effect without an explicit transition prompt.

## C3: descriptor archive with transition proposals

- use the archive and parent sampling from C2
- use the transition operator from C1
- provide inspiration candidates from different occupied cells

Purpose: estimate the combined effect and the interaction between proposal policy and research memory.

## Prompt placebo

Add a pilot-only condition that tells the agent to seek novel designs without giving descriptor names or changing retention. Compare it with C0 before the main study. This check separates motivational wording from a structured transition operator.

## Optional C4: full-rewrite operator

Allow periodic full candidate rewrites instead of code patches.

Purpose: test whether diff-based mutation blocks mechanism changes.

Run C4 only after the pilot shows that implementation attrition, rather than proposal attrition, limits RFTs.

---

# 6. Starting architecture

Use a conventional, validated transformer as the primary starting point:

- standard learned token embeddings
- standard positional encoding
- ordinary attention projections
- conventional feedforward layer
- no architecture tricks copied from compact public winners

The existing 6,080-parameter AdderBoard baseline can serve this role because it supplies a familiar anchor. Its parameter count has no target meaning.

Add a second starting family as a robustness study after the main experiment. A second start tests whether results depend on the initial architecture. Do not mix starting architectures inside the primary causal comparison.

---

# 7. Evaluation stack

## Gate 1: execution

- code parses and imports
- training completes within the resource cap
- generation uses the allowed interface
- evaluation returns complete records

## Gate 2: transformer validity

- at least one self-attention layer
- tensor-in, logits-out forward pass
- generic autoregressive decoding outside the model
- no explicit Python addition logic
- no answer encoding in input format

## Gate 3: task accuracy

- official fixed-seed AdderBoard evaluation
- private random-seed shadow evaluation
- edge cases
- carry-chain strata

Require at least 99% on official and shadow tests.

## Gate 4: robustness

Report:

- accuracy across new seeds
- sensitivity to initialization and retraining
- longer carry chains
- modest length shifts
- input-symbol permutation
- numerical-precision sensitivity

Robustness supports the discovery claim. It does not change AdderBoard’s official qualification rule.

## Gate 5: mechanism

Use:

- component ablations
- activation and attention probes
- targeted counterexamples
- clean reimplementation
- independent retraining

The mechanism review separates a stable architectural discovery from verifier overfitting or accidental numerical behavior.

---

# 8. Trajectory instrumentation

## Independent unit

The independent experimental unit is a complete run. Candidates share history and cannot serve as independent samples.

## Candidate record

Store one immutable record per proposal:

- run_id
- condition
- harness
- base_model and snapshot
- seed
- candidate_id
- parent_id
- inspiration_ids
- proposal_text
- stated mechanism hypothesis
- prompt and response hashes
- complete code hash
- diff
- proposal and completion timestamps
- execution status
- transformer-validity status
- official accuracy
- shadow accuracy
- stress-test results
- qualifies
- descriptor vector
- RFT label
- novelty labels
- archive or incumbent decision
- rollback target
- future resampling count
- LM input and output tokens
- training and verification compute
- parameter count as descriptive metadata
- annotation version

## Funnel stages

Track each candidate across:

1. proposed
2. translated into code
3. executed
4. transformer-valid
5. accuracy-valid
6. retained in research memory
7. selected again as a parent or inspiration
8. confirmed as an architecture or mechanism discovery

This funnel identifies the loss stage:

- ideation
- implementation
- execution
- validity
- task performance
- memory
- reuse
- novelty confirmation

## Storage rules

- raw records remain immutable
- analysis scripts generate derived tables
- prompts and responses receive stable hashes
- rejected and crashed candidates remain available
- OpenEvolve evolution tracing must include code, prompts, responses, artifacts, parentage, and timestamps
- Autoresearch must save candidate code before rollback

---

# 9. Outcomes

## Confirmatory primary outcome

`RIGOROUS_EXPERIMENT_PLAN_V2.md` supersedes the older co-primary framing. The
confirmatory unit is the complete assigned run, and the primary outcome is the
number of unique Layer B-qualified mechanism clusters reached by the frozen
budget. Descendants, refactors, and parameter variants in one mechanism cluster
count once per run.

## Manipulation check: valid architecture-family coverage

### 1. Valid architecture-family coverage

At a fixed evaluation budget:

\[
C_r(B)=
\left|
\{z(v):v\in r,\ Q(v)=1,\ t(v)\leq B\}
\right|
\]

Report complete signatures, per-axis coverage, and selected pairwise coverage.

## Secondary outcome: confirmed novelty yield

\[
Y_r^{N3+}(B)=
\#\{v:Q(v)=1,\ N(v)\geq N3,\ t(v)\leq B\}
\]

Also report the probability that a run produces at least one confirmed N3+ candidate.

## Secondary outcomes

- valid N2+ yield
- number of valid RFTs
- time to first valid RFT
- time to first confirmed N3+ candidate
- proposal-to-code RFT survival
- code-to-validity RFT survival
- validity-to-retention survival
- retention-to-reuse survival
- architectural distance from the start
- distance from the frozen reference corpus
- mechanism diversity
- lineage concentration
- occupied-cell revisit rate
- abandoned-family count
- official-shadow accuracy gap
- retraining success rate
- verifier-exploit rate

## Diagnostic outcomes

Record parameter count, training time, token use, and accelerator time to explain resource effects. Do not rank discoveries by parameter count.

---

# 10. Hypotheses

| ID | Hypothesis | Falsifying pattern |
|---|---|---|
| H1 | Incumbent-only search proposes fewer RFTs than descriptor-aware search. | Similar run-level proposal rates after opportunity and budget matching. |
| H2 | Incumbent-only search loses more valid alternative families at retention and reuse. | Valid alternative families survive and receive follow-up work at the same rate. |
| H3 | Descriptor-aware archives increase valid family coverage. | Coverage remains unchanged or consists of invalid cosmetic variants. |
| H4 | Higher valid family coverage increases the chance of an N3+ discovery. | Coverage rises without any increase in confirmed mechanism novelty. |
| H5 | Diff-only mutation causes implementation attrition for coordinated family changes. | Full rewrites do not improve code realization or validity of proposed RFTs. |
| H6 | Diversity declines before research plateaus in incumbent-only runs. | Diversity remains stable, or decline follows termination rather than preceding it. |

The study should accept negative results. If the agent proposes many RFTs but they fail training, the paper should locate the bottleneck in validation rather than ideation.

---

# 11. Pilot and main experiment

## Pilot

- four factorial conditions: C0-C3
- one prompt-placebo condition
- six independent seeds per condition
- 100 evaluated candidates per run
- 30 runs
- 3,000 candidate evaluations

The pilot estimates:

- valid RFT base rate
- N2+ and N3+ base rate
- run-to-run variance
- annotation burden
- implementation and validity attrition
- archive occupancy
- intervention leakage
- compute distribution

Use the pilot to check feasibility, revise power estimates, and decide whether the prompt placebo warrants inclusion in the main study. Keep C0-C3 unless compute limits require a preregistered reduction. Do not select conditions from parameter-count results.

## Main experiment

- C0-C3 in the common orchestration layer
- at least 10 to 12 independent seeds per condition, subject to pilot power analysis
- 200 candidate evaluations per run
- fixed LM-token and accelerator caps
- outcome reporting at 50, 100, and 200 evaluations

Disable outcome-based early stopping.

Run a smaller native-harness replication after the primary experiment:

- Autoresearch with C0 and C1 policies
- OpenEvolve with C2 and C3 policies
- at least five seeds per native configuration

Treat these runs as portability evidence, not a clean harness ranking.

## Budget matching

Report three budgets:

- evaluated candidates
- LM tokens or rollouts
- accelerator-hours

Use candidate count as the primary x-axis. Report token and compute sensitivity because one architecture may cost more to train.

---

# 12. Statistical analysis

## Primary analysis

- resample and cluster at the run level
- compare coverage with run-level bootstrap intervals or hierarchical count models
- compare time to first valid RFT and N3+ candidate with survival analysis
- censor runs with no event
- compare probability of at least one N3+ discovery with a hierarchical logistic model

## Condition model

Include:

- proposal policy, research memory, and their interaction as fixed effects
- native harness as a separate replication stratum
- starting architecture as a separate robustness stratum
- run as the independent unit

## Reporting

Report:

- effect sizes
- uncertainty intervals
- all seeds
- failed runs
- annotation sensitivity
- reference-corpus sensitivity

Use one or two primary tests. Apply false-discovery-rate control to secondary hypotheses.

---

# 13. Controls

## Required controls

- identical starting code
- identical base language model and snapshot
- identical tool permissions
- identical prompt budget
- identical evaluator
- no internet during runs
- fixed reference-corpus cutoff
- private shadow tests
- blinded novelty annotation
- complete rejected-candidate logging

## Negative controls

- formatting and refactor edits should receive N0
- dimension changes within a known family should receive N1
- symbol renaming should not affect architectural distance
- evaluator-only changes must fail scope checks

## Leakage audit

Review prompts and model outputs for:

- names of public winning components
- copied code fragments
- leaderboard-specific constants
- references to hidden corpus entries

Discard and rerun contaminated conditions under preregistered rules.

---

# 14. Figures and tables

## Main figures

1. **Proposal-to-discovery funnel**
   - proposed
   - implemented
   - valid
   - retained
   - reused
   - novelty-confirmed

2. **Architecture-family coverage over budget**
   - all run traces
   - proposal-policy, research-memory, and interaction effects
   - valid candidates only

3. **Architecture-space map**
   - starting architecture
   - frozen reference families
   - valid discovered families
   - unresolved candidates

4. **Lineage and memory**
   - Autoresearch accepted line with rejected branches
   - OpenEvolve archive and island graph
   - parent and inspiration edges separated

5. **Novelty yield**
   - N0-N4 distribution
   - expert-confirmed N3+ candidates
   - all seeds

6. **Mechanism case studies**
   - architecture diagram
   - causal mechanism claim
   - ablation results
   - closest reference design

## Main tables

1. controlled system settings
2. architecture descriptor codebook
3. novelty rubric and agreement
4. run-level coverage and novelty effects
5. confirmed mechanism candidates and evidence
6. limitations and permitted claims

Exclude leaderboard rankings and parameter-count races from the main paper.

---

# 15. Execution roadmap

## Phase 0: scope freeze

- freeze the claim
- freeze the AdderBoard commit
- freeze the starting architecture
- freeze the reference-corpus cutoff
- state that parameter count is descriptive
- create a preregistration

**Deliverable:** scope document and frozen artifact manifest

## Phase 1: instrumentation

- create the candidate schema
- preserve prompts, proposals, code, parentage, results, and retention
- add private validity and shadow tests
- test lineage reconstruction

**Deliverable:** immutable trajectory store and integrity report

## Phase 2: taxonomy calibration

- draft architecture-family codebook
- label 50 to 100 calibration edits
- revise ambiguous axes
- conduct blinded dual annotation

**Deliverable:** frozen codebook and agreement report

## Phase 3: novelty corpus

- collect and hash known designs
- annotate mechanisms and descriptors
- create blinded comparison packets
- test nearest-reference procedures

**Deliverable:** frozen reference corpus and novelty-review protocol

## Phase 4: pilot

- run C0-C3
- run the prompt placebo
- audit leakage
- estimate variance and attrition
- inspect archive behavior
- lock the main-study sample size

**Deliverable:** pilot report and powered main-study plan

## Phase 5: main experiment

- run the C0-C3 factorial experiment
- monitor infrastructure
- rerun corrupt jobs under frozen rules
- keep outcome analysis blinded until data freeze

**Deliverable:** frozen trajectories and run-level result tables

## Phase 6: mechanism validation

- select N3 and N4 candidates under the rubric
- conduct ablations and counterfactual probes
- reproduce candidates from clean code
- complete third-expert review

**Deliverable:** mechanism dossiers

## Phase 7: robustness and writing

- second starting architecture
- budget sensitivity
- annotation sensitivity
- reference-corpus sensitivity
- paper and artifact release

---

# 16. Go and no-go rules

- **Log integrity below 95%:** restrict claims to systems with complete trajectories.
- **RFT agreement below 0.67:** revise the architecture codebook.
- **Novelty agreement below 0.67:** revise the novelty rubric.
- **No valid RFTs in the pilot:** inspect proposal and implementation attrition before scaling.
- **Coverage gain without N3+ gain:** claim broader exploration, not architectural discovery.
- **N3+ candidates without replication:** report mechanism candidates, not confirmed discoveries.
- **Prompt leakage:** discard the affected condition.
- **Harness imbalance in tokens or compute:** correct budgets or report a stratified analysis.
- **Frequent valid RFTs in C0:** reject the local-search hypothesis and study later-stage loss.

---

# 17. Paper structure

## 1. Introduction

- endpoint accuracy hides the research path
- validity does not establish novelty
- autonomous systems need stage-resolved architectural evaluation
- contributions and scope

## 2. Related work

- research agents as search policies
- program evolution
- neural architecture search
- novelty search and quality-diversity
- arithmetic-transformer mechanisms

## 3. Experimental environment

- AdderBoard validity role
- starting architecture
- frozen evaluator and reference corpus
- controlled harness settings

## 4. Measurement

- candidate graph
- RFT descriptors
- novelty levels
- annotation and mechanism review

## 5. Retrospective trajectories

- proposal and implementation patterns
- family coverage
- loss stages
- motivating cases

## 6. Controlled intervention

- conditions and budgets
- coverage outcome
- N3+ outcome
- robustness

## 7. Mechanism case studies

- confirmed candidates
- ablations
- reference comparisons
- failed novelty claims

## 8. Limitations

- one task
- reference-corpus incompleteness
- model contamination
- descriptor dependence
- annotation cost
- mechanism uncertainty

## 9. Conclusion

- measured constraints on valid architectural discovery
- implications for proposal operators, memory, and selection

---

# 18. Immediate actions

1. Remove parameter minimization from all prompts, rewards, archives, and stopping rules.
2. Freeze a conventional starting transformer and AdderBoard validity wrapper.
3. Build the candidate manifest and preserve rejected artifacts.
4. Draft the architecture-family and novelty codebooks.
5. Build the frozen reference corpus without exposing it to agents.
6. Run a small C0 versus C3 instrumentation test before the full pilot.

## Sequencing rule

Validate the novelty construct before scaling the search experiment. Validate candidate mechanisms before claiming architectural discovery.
