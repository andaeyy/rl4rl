Discover architectural mechanisms that solve ten-digit addition through autoregressive token generation.

The model must use self-attention and a tensor-in, logits-out forward pass.
Expose only `build_untrained_model(seed)`. The evaluator supplies the fixed
Phase-1 vocabulary, padded operands, reversed eleven-digit target plus EOS,
answer-only loss, optimizer, training data, checkpoints, and decoding. Every
architecture is freshly initialized and optimized under the same profile.
Online parent eligibility uses the frozen public Layer A rule. Any sealed
qualification occurs only after the complete run is frozen and is never shown
to the search agent.

Phase 1 freezes tokenization. Change the model architecture, not the evaluator's
task format or training treatment.

Use observed failures to reason about digit alignment, digit computation, carry propagation, or another mechanism that the code and tests can support.
