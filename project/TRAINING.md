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

## Frozen RGB-D temporal rung — do not materialize yet

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

This is not a training or convergence result. Specification 1.54 freezes the
new observable-depth temporal protocol at config SHA-256
`5667cdb3603682b8d80a3e42793d25e36989269df1afacfa9b1028f2451101e9`
and canonical protocol payload SHA-256, computed before inserting its
self-reporting digest field,
`4e334e9d7942ea3f2416c0a9f5ca8e327d1d0a1e9131074f20c051ebd3163ad7`.
The source/config contract can be inspected without generating data:

```bash
conda run -n orpheus python scripts/run_rgbd_temporal_free_motion_ladder.py \
  --phase protocol \
  --config configs/rgbd_temporal_free_motion_cpu.yaml
```

Do not run `--phase development` or `--phase qualification` at this boundary.
Development `41000000--41000023`, selector `42000000--42000023`, confirmation
`43000000--43000023`, and one-shot final `44000000--44000047` are all fresh
and unopened.

The estimator is parameter-free: its parameter/buffer/state-dict counts and
optimizer updates are zero. It measures sixteen RGB-D frames, gives every row
uniform weight in the differentiable exact free-motion WLS fit, and queries
state at `0.1/0.25/0.5/1.0/2.0 s`. Confidence and Boolean validity are
diagnostic/fail-closed only. The protocol gates current and per-axis state,
every horizon, stationary/zero-velocity/two-frame baselines, semigroup,
resources, fixed-output VJPs to RGB and depth, an explicitly worse RGB-only
control, and zero-validity missing depth. Its finite OLS covariance is only an
i.i.d. residual diagnostic; no calibrated posterior, coverage, or proper-score
claim is made.

The combined seed-free implementation gate is `103 passed`, independent review
passes, and the final repository gate is `1129 passed, 16 skipped in 428.18 s`.
That does not authorize a development run or establish accuracy. When
development is later authorized, it requires fresh
report/checkpoint paths and a clean exact source. A passing zero-optimizer
development artifact must then be independently reviewed by exact digest
before protected access.
The qualification runner reserves an exclusive durable ledger and records
materialization before it evaluates selector, confirmation, and final in that
order; the first failure stops later access.

Only after this standalone rung qualifies should it enter the public runtime.
The first bridge must use one batched composite `rgbd` packet carrying
`[B,3,H,W]` RGB and `[B,1,H,W]` depth plus calibration and explicit image
size. A modality-qualified sensor key avoids current cache/history/scheduler
collisions. Separate same-timestamp RGB and depth packets are outside the
qualified contract. Runtime may consume only RGB-D, calibration, timestamps,
and declared priors—never simulator state.

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
