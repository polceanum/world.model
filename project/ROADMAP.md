# Roadmap

## Milestone 1

One runnable synthetic RGB vertical slice: deterministic simulator, typed
persistent belief, hybrid dynamics, debug oracle, global and residual RGB
measurement paths, causal correction, lifecycle, uncertainty, restitution/drag
identification, training/checkpointing, held-out baselines, and visual demo.

## Milestone 2

Stronger event timing, hypothesis branch/prune/merge, learned camera correction,
fixed-lag smoothing, richer geometry, and the window-spectral ablation.

The first Milestone 2 increment is now implemented: an explicit abstraction
registry, point-versus-rigid-sphere routing, and reversible LLM-style belief
tokens. Next, train residual-token processing and evidence-driven abstraction
selection while retaining per-abstraction accuracy and complexity gates.

The active capacity ladder is evidence-gated: first qualify the repaired
3.00M-parameter dense typed-attention rung on repeated fixed selectors; then
compare data-only, depth, width, and bounded timestamped-history changes one at
a time. A Mac result advances to a tens-of-millions single-CUDA-GPU rung only
after disjoint RGB-only validation/test/OOD evidence shows a stable capacity
ceiling and predicts a useful gain. The runtime belief/filter contracts do not
change between rungs.

Depth-only growth now has an exact inherited-function path: contiguous new
pre-norm residual blocks start with zero MHA/SwiGLU output projections, while
all smaller-model tensors load strictly. This removes relearning as a depth
comparison confound. It does not weaken the selector/plateau gate, and width
growth remains graph-initialized until it has its own proved handoff.

Every rung now uses the same count-pooled consecutive training-trend report for
early collapse diagnosis. Scale decisions still require fixed disjoint RGB-only
selectors/test/OOD; noisy or incomplete training windows cannot authorize a
larger model.

## Milestone 3

Add exactly one useful second modality behind the existing observation contract.

## Milestone 4

Connect real calibrated data or pretrained perception while preserving the
belief/dynamics/filter contracts. Foundation perception should compile video
into persistent executable abstractions rather than make pixel generation the
runtime state.
