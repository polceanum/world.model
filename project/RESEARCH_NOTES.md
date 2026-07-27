# Research notes

## Hypotheses

- Persistent predict/correct state should recover more cheaply than repeated
  clip encoding.
- Structured event jumps should improve collision horizons over smooth motion.
- Residual ROI perception should approach repeated-global correction quality at
  lower latency.
- Explicit uncertainty and observability gates should reduce identity and
  parameter hallucination under occlusion.

These are hypotheses, not results. Empirical entries must identify config,
checkpoint, split/seeds, device, commands, metrics, and failure cases.

## Evidence so far

The deterministic CPU vertical slice and reduced MPS compatibility paths have
run. Exact long-form commands and artifacts are recorded in `project/STATUS.md`.
The promoted checkpoint remains the validation-selected step 72 at
`runs/accuracy-closed-frozen-94/checkpoints/best_rollout.pt`.

On the explicit 16-episode fresh-validation manifest
`100004–100019` (disjoint from trainer validation and test), that checkpoint
produced:

- current position MAE/RMSE `0.186991 / 0.239613 m`;
- 0.1/0.25/0.5-second model RMSE
  `0.236517 / 0.189670 / 0.174269 m`;
- 0.5-second constant-velocity RMSE `0.511659 m`;
- collision F1 `0.042553`;
- injected-perturbation recovery `20.09%`;
- 90% forecast coverage `97.75%`;
- zero distance-gated ID switches and zero non-finite outputs.

Collision-conditioned model RMSE was
`0.149769 / 0.137729 / 0.174269 m`; this is respectively
`22.08% / 56.87% / 65.94%` below constant velocity on the exact same masks.

### Temporal RGB velocity experiment

The original one-frame RGB position-to-velocity coupling is effectively
inert at 20 Hz because its covariance is amplified by `1/dt²`. A causal
three-position least-squares history keyed by persistent object ID is now
implemented and measured explicitly.

On the fresh selection manifest, a deliberately calibrated
`1.0 (m/s)²` variance ceiling changed:

- velocity RMSE `1.369454 → 1.309964 m/s`;
- ordinary same-step velocity improvement `0.001594 → 0.025985 m/s`;
- collision F1 `0.042553 → 0.055172`;
- current position MAE `0.186991 → 0.190923 m` (worse);
- 0.25-second RMSE `0.189670 → 0.201318 m` (worse);
- perturbation recovery `20.09% → 19.26%` (worse).

History sizes three/four and variance ceilings one/two/four all showed the
same tradeoff. Therefore temporal velocity remains opt-in, and its default
uncertainty propagation has no empirical ceiling.

A 22-step frozen-global continuation completed in `183.15 s` and selected
step 94 by the tiny trainer-validation loss (`0.249018`). On the larger fresh
manifest with temporal evidence enabled it raised collision F1 to `0.121622`
and reduced velocity RMSE to `1.277519 m/s`, but position MAE rose to
`0.196397 m`, 0.5-second RMSE to `0.184454 m`, and perturbation recovery fell
to `11.84%`. The checkpoint is retained as a truthful negative result at
`runs/temporal-continuation-94`, not promoted.

### Interpretation

- Persistent RGB correction and structured rollout are functional and beat
  constant velocity at longer horizons.
- Event-window semantics and missing-edge pooling are now correct; event skill
  remains limited by state estimation and data, not metric alignment alone.
- Temporal position slopes contain useful velocity information, especially
  for high-error/collision frames, but the current diagonal confidence/update
  rule injects enough correlated error to harm the primary trajectory metrics.
- Drag/restitution updates execute under explicit observability gates but
  remain numerically negligible; useful online identification is unproven.
- No current result establishes the recommended collision F1, full occlusion
  recovery, parameter convergence, or the full 3,000-step MPS schedule.
