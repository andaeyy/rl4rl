# Data and measurement protocol

## Unit of analysis

The canonical unit is a proposed or evaluated change, not a final model. An
event may contain multiple component edits. Preserve proposal events, failures,
invalid verifier outputs, and rollbacks; dropping them would condition the
analysis on successful search.

Every event must have a stable ID and run ID. Parent IDs should come from the
source system's actual ancestry. Do not infer parents from row order unless that
inference is explicitly stored in `provenance` and excluded in lineage-sensitive
analyses.

## Three different notions that must not be conflated

1. **Syntactic edit distance**: how much source text changed.
2. **Architecture diversity**: how many distinct, pre-declared architectural
   fingerprints or feature combinations were visited.
3. **Representational boundary crossing**: whether an edit changes the family
   of primitives used by a component.

A large diff can preserve the same architecture, while a small diff can replace
additive position vectors with rotation inside attention. Embedding-space idea
distance is a useful sensitivity analysis, but it is not the primary ontology
measure.

## Boundary labels

- `preserving`: the edit changes parameterization or capacity inside the same
  declared representational family.
- `changing`: it moves a component between pre-declared representational
  families.
- `ambiguous`: the transition is unknown, compound, or disputed.
- `not_applicable`: the edit does not alter model representation (for example,
  logging or a training-time-only change).

The TOML taxonomy provides heuristic suggestions. For the paper, use at least
two independent annotators, keep them blind to agent identity and outcome where
possible, report agreement, and adjudicate disagreements before analysis.

## Initial metric definitions

- **Mutation acceptance rate**: accepted / (accepted + rejected). Invalid,
  error, and proposal-only events are reported separately rather than silently
  entering the denominator.
- **Boundary-crossing rate**: events containing at least one `changing` edit /
  events containing at least one annotated edit.
- **Accepted boundary-crossing rate**: same numerator restriction among
  accepted events.
- **Edit entropy**: Shannon entropy of
  `(component, operation, boundary_label)` categories. Normalized entropy divides
  by the maximum entropy for the observed number of categories.
- **Architecture diversity ratio**: unique architecture fingerprints / events
  with a fingerprint.
- **Revisit rate**: one minus the architecture diversity ratio. This starter
  definition measures exact revisits; later add neighborhood-based revisits.
- **Rollback rate**: rolled-back events / all events.
- **Frontier gap ratio**: best qualifying parameter count found by a run divided
  by the chosen external reference frontier. State whether the 36-parameter
  trained or disputed 6-parameter hand-coded reference is used.

These are descriptive statistics. Runs—not individual events—are the primary
independent sampling units for uncertainty intervals and system comparisons.

## Required provenance

For every run, archive:

- exact agent/model identifier, scaffold commit, prompt, seed, and budget;
- repository commit and environment lockfile;
- all proposals, responses, diffs, stdout/stderr, verifier reports, and timing;
- exact parent selection and island/migration metadata where applicable;
- acceptance, rollback, and stopping decisions;
- source-artifact hash and parser version; and
- any human intervention or post-hoc repair.

Treat data in `data/raw` as immutable. Normalize to `data/interim`, then write
only adjudicated events to `data/processed`.
