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
tokens. The later pre-generalization programme explored residual-token
processing and evidence-driven abstraction selection, including a dense
typed-attention campaign. Those campaigns and their checkpoints are historical
diagnostics, not the active roadmap or launch recommendation.

The current evidence-led programme first qualified the minimal differentiable
RGB core, then observable RGB-D temporal state, its public one-object bridge,
exactly two fully visible objects, and one known calibrated orbital-camera
family. The separate partial-visibility/recovery, variable-radius, and
identifiable-drag families failed and are closed; their source is not merged
into the accepted runtime. No next rung has been selected or authorized. A
future rung must introduce one genuinely new capability, freeze disjoint
development/selector/confirmation/final manifests and hard gates before
access, and preserve all accepted accuracy, uncertainty-diagnostic,
identity/event, rollout, gradient, memory, and throughput contracts. Any new
posterior-calibration claim requires its own explicit qualification. A failed
family stops rather than receiving marginal retuning.

Historical attention work established an exact inherited-function path for
depth-only growth: contiguous new pre-norm residual blocks start with zero
MHA/SwiGLU output projections while all smaller-model tensors load strictly.
That reusable transform does not authorize a capacity rung. Learned residuals,
depth, width, history, or accelerator scale may grow only after a smaller
structured model has a localized, fixed-manifest plateau and the new capacity
has its own predeclared handoff and non-regression evidence.

Every learned rung uses count-pooled consecutive training-trend reporting for
early collapse diagnosis. Scale decisions require fixed disjoint
observable-input validation/test/OOD evidence appropriate to the declared
modality; noisy or incomplete training windows cannot authorize a larger model.

## Milestone 3

The first useful second modality, observable metric depth paired with RGB, is
qualified behind the existing observation contract for the narrow accepted
families. Additional modalities require their own independently measurable rung
and may not weaken the RGB/RGB-D lower-rung controls.

## Milestone 4

Connect real calibrated data or pretrained perception while preserving the
belief/dynamics/filter contracts. Foundation perception should compile video
into persistent executable abstractions rather than make pixel generation the
runtime state.
