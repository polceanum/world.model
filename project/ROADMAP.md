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

## Milestone 3

Add exactly one useful second modality behind the existing observation contract.

## Milestone 4

Connect real calibrated data or pretrained perception while preserving the
belief/dynamics/filter contracts. Foundation perception should compile video
into persistent executable abstractions rather than make pixel generation the
runtime state.
