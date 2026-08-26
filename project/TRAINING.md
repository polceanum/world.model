# Training and qualification

## Active policy

Orpheus scales through deterministic, fail-fast capability rungs. Each rung
changes one independently measurable source of complexity, binds disjoint
train/selector/confirmation/final manifests before access, and preserves every
accepted lower-rung gate. A failed family stops; thresholds and final sets are
not repeatedly tuned.

Runtime inputs must be observable modalities plus calibration, timestamps, and
declared priors. Simulator state may label losses and metrics but may not enter
a claimed RGB/RGB-D forward path. Continuous state, parameter, and rollout
learning uses ordinary PyTorch autograd. Analytic tensor equations provide
inductive bias; learned residual capacity is added only after an identifiable
structured baseline leaves measured error.

## Qualified base

The supported standalone developer command is:

```bash
conda run -n orpheus python scripts/run_minimal_toy_ladder.py \
  --config configs/minimal_differentiable_toy_cpu.yaml \
  --report runs/minimal_differentiable_toy_v2/report.json \
  --checkpoint runs/minimal_differentiable_toy_v2/model.pt
```

Do not rerun its published final manifest merely to regenerate provenance.
Future runs require fresh output paths and produce atomic, versioned,
weights-only project checkpoints whose SHA-256 is bound in the report.

## Closed monocular temporal rung

Do not launch the former temporal runner or qualification workflow. Clean
source commit `8889818619121351d342490786331e854364532c` completed the frozen
32-update development phase and failed its 16-seed audit. The immutable report
is retained at ignored local path
`.archive/20260826-pre-generalization/temporal-free-motion-attempt-2/development_report.json`
with SHA-256
`be488d045e259c0804a2a2b24215fa4eb3025d69f6113d8dbefe21d72f827554`.
It records `passed: false`, `review_ready: false`,
`stopped_after: development_audit`, and
`protected_data_materialized: false`.

The trained audit failed 10 gates. Current position/velocity RMSE was
`0.016128 m`/`0.070461 m/s`, and horizon position RMSE at
`0.1/0.25/0.5/1.0/2.0 s` was
`0.022907/0.033205/0.050360/0.084191/0.149501 m`. The fit assigned only
`0.000534` as much weight to the oldest frame as the anchor and concentrated
`77.63%`/`91.71%` of its mass on the last three/five frames through a learned
`10.0338 /s` taper. Thus the trainable weighting collapsed temporal extent
even though the exact free-motion oracle, simulator comparison, semigroup,
gradient, geometry, memory, and latency checks were correct.

A confidence-only development diagnostic produced current position/velocity
RMSE `0.00842 m`/`0.01660 m/s` and horizon position RMSE
`0.01000/0.01239/0.01638/0.02430/0.03963 m`. That does not qualify the family:
future velocity remained `0.01652 -> 0.01502 m/s` against the `0.01 m/s` gate,
and the early zero-velocity-baseline ratios still exceeded `0.5`. This was the
terminal second of two monocular architecture attempts. Do not change its
thresholds, run a third variant, or materialize selector, confirmation, or
final seeds. The config, runner, estimator, and dedicated tests were removed
from the active source tree after archiving the report.

## Next temporal rung

The seed-free single-frame foundation for the next rung now passes. Simulator
v7 provides observable `[T, 1, H, W]` metric surface depth from the same exact
nearest ray--sphere winner used by RGB, visibility, and instance output. A
parameter-free measurement uses the RGB-derived subpixel centre,
differentiable bilinear depth, known radius, canonical camera, and perspective
radius correction; it consumes no labels, simulator state, instance maps, or
object IDs. Its 18-case grid reaches `0.00336217 m` position RMSE and has
finite gradients to centre, RGB, and depth. Focused validation is `29 passed`
and independent review passes, without an episode seed namespace or protected
access.

This is not a training or convergence result. The next permitted training work
is a new observable-depth/RGB-D temporal state fit, not another monocular
reliability-taper variant. Before any episode generation,
freeze a new config and protocol that declares fresh development, selector,
confirmation, and final manifests; at most two architecture attempts; RGB-D
noise/calibration and missing-depth behavior; an RGB-only ablation; per-axis
current/velocity/horizon limits; trivial-baseline and semigroup gates;
uncertainty coverage; gradient ownership; parameters/state/RSS; and separated
RGB-D-observation versus state-only rollout latency. Runtime may consume only
RGB-D, calibration, timestamps, and declared priors—never simulator state.

The complete `1075 passed, 16 skipped` repository gate belongs to the clean
pre-failure source commit above. After deletion of the rejected experiment and
addition of the RGB-D core, the new complete source gate passes `1091` tests
with `16` expected inactive-device skips in `418.49 s`.

The broad `train.py`, `evaluate.py`, and `demo.py` workflow remains available
for `OnlineWorldModel` smoke/integration checks, but no older sustained profile
is an active accuracy campaign or deployment incumbent. Exact-resume,
checkpoint, validation-support, and promotion integrity remain tested reusable
contracts.

Historical campaign commands and evidence through specification 1.51 remain
in Git commit `c16acc99` and the ignored local pre-generalization archive.
