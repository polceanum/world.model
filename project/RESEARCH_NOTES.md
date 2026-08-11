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

### Typed-attention stability and scaling decision

The current Mac rung is a 3,004,656-parameter model, including a 1,103,626-
parameter width-128/four-block dense typed-attention residual over at most 22
scene/entity/relation tokens. It already uses pre-RMSNorm, scaled dot-product
multi-head attention, and SwiGLU. Exact replay of the first collision-isolated
campaign found that its recurrent step-280 failure is not lack of capacity: a
joint `17.6842` raw gradient is localized to the normal/tangent force output
rows, leaving about `0.8573` in the rest of the interaction module. A typed
force-row optimizer cap is therefore the next controlled repair.

The next scale decision is gated on a fresh repaired learning curve and broad
plateau, not training loss. Once qualified, compare data-only, width, depth,
and bounded-history rungs one at a time, increasing balanced continuously
varied episode draws with parameter count. Use fixed disjoint RGB-only
validation/test/OOD manifests and keep the accepted smaller model as a
non-regression control. Long-context efficiency techniques and MoE are
deferred because the current token set is short and neither addresses the
measured failure.

The fresh force-isolated run remains healthy through sampled update 32. The
step-24 raw collision-row norm is `4.45588`, its row coefficient is `0.224422`,
the post-row interaction norm/coefficient are `2.36835/0.422235`, and the true
raw whole-model/final coefficient are `4.94611/0.202179`. This is a contained
outlier rather than a severe shared-gradient collapse; the update is finite and
applied, supported trajectory count is 396, frozen perception gradient is zero,
and RSS is `2,891,427,840` bytes. Step 32 independently contains a raw
collision-row norm of `3.23987`; its subsequent interaction coefficient is
`0.929705`, the update is applied, all scenarios have four sampled blocks, and
sampled trusted identity switches are zero. There is still no trained fixed
selector.

Through sampled step 72, one severe force-row warning occurs at step 64. The
joint normal/tangent row norm is `21.4665` inside a `21.5377` raw interaction
norm. After the force cap, the interaction norm is `2.01547`, so unrelated
attention gradients retain a `0.496162` stage coefficient rather than the
raw-total `0.0464303` coefficient. The next sampled block returns to force
coefficient `0.976879` and stage coefficient `0.686862`, with positive future
correction, zero sampled identity switches, and unchanged memory. This
supports the isolation mechanism but does not prove the event is harmless to
accuracy; checkpoint 128, the former 152/280 boundaries, and fixed selector
512 remain required.

The durable step-128 checkpoint confirms that the optimization experiment is
actually isolated: 177 inherited tensors have zero bitwise changes, every one
of 48 attention tensors changes, and the only 48 Adam states belong to those
attention parameters at step 128. All serialized state and linked protected
artifacts pass finite/hash checks. The sampled step-128 identity switches
match the preceding collision-isolated control on the identical seed/window;
aggregate sampled rate is `0.8608%`. This rules out scope drift, dead attention
capacity, corrupted optimizer state, and protected-reference mutation through
the checkpoint, but it does not establish held-out accuracy.

The next exact schedule landmark, step 152 on frames 7--11, confirms a large
optimizer-health improvement. Raw interaction norm/retained stage coefficient
progresses from `28.1387/0.03554` in the normalized campaign to
`7.11114/0.14308` with collision isolation and `2.46615/0.48940` with force
isolation. In the current run the force group is only `0.25152` and unclipped;
collision is locally bounded from `1.70491`, all objectives have support,
identity switches are zero, future correction is positive, and the update is
finite/applied. This validates the repair at one historical boundary but does
not replace step 280 or broad fixed validation.

Step 280 then disproves the assumption that decoder parameter-row isolation is
sufficient. Its raw force/total parameter norms are `989.7965/995.5391`; by the
time the row cap runs, shared projections and attention blocks already carry
order-one-to-ten gradients and the effective total update retains only
`0.0010045`. The campaign is stopped at durable step 256 and cannot count
toward convergence.

Specification 1.28 moves semantic isolation to the causal location: each raw
node, collision, and joint-force output invocation receives an optional
backward-only norm cap before the decoder/shared stack, followed by the existing
parameter hierarchy. Exact diagnostic replay from the same step-256 optimizer,
RNG, and data state reduces the later step-280 parameter norm to `10.8330`,
bounds the maximum shared parameter norm to `0.0851`, and leaves a `0.6979`
post-row interaction-stage coefficient. The batch remains finite, supported,
applied, and physically comparable; localized severe coefficients remain
visible. This establishes causal optimizer repair, not accuracy or
generalization. A fresh weights-only campaign must still pass selector 512 and
the declared plateau before any scale rung advances.

Exact capacity census for later one-axis studies:

- current/data-only: `3,004,656` total, `1,103,626` attention parameters;
- depth six at width 128: `3,530,480` total, `1,629,450` attention parameters;
- width 192/four blocks/SwiGLU 768: `4,342,896` total, `2,441,866` attention
  parameters; and
- future single-GPU width 256/six blocks/SwiGLU 1024: `8,305,648` total,
  `6,404,618` attention parameters.

These are design points, not accepted checkpoints. Modern long-context and
sparse-inference mechanisms are deferred because 22 structured tokens do not
exercise their intended bottlenecks. Data coverage and held-out physical
generalization must scale with capacity.

The deterministic CPU vertical slice and reduced MPS compatibility paths have
run. Exact long-form commands and artifacts are recorded in `project/STATUS.md`.

### Accuracy-v4 closed-loop promotion

The promoted step-648 checkpoint is
`runs/accuracy-closed-structured-v4/checkpoints/best_rollout.pt` (SHA-256
`9b943f60128a2bd15298847d8c7de4dd3166646f3644720a3149155e57d85bcd`).
It continues the selected step-584 perception state for 64 causal closed-loop
RGB updates. Full validation selected rollout-position loss `0.0119829765`.

The fair ROI-local confirmation comparison on seeds `100064–100095` was:

| Metric | Step 584 | Step 648 |
| --- | ---: | ---: |
| current position MAE/RMSE (m) | 0.083808 / 0.109239 | 0.083282 / 0.109426 |
| velocity RMSE (m/s) | 0.730034 | 0.731623 |
| 0.1 s forecast RMSE (m) | 0.134093 | 0.132424 |
| 0.25 s forecast RMSE (m) | 0.174492 | 0.171900 |
| 0.5 s forecast RMSE (m) | 0.231253 | 0.226994 |
| perturbation recovery | 0.482786 | 0.478172 |
| collision F1 | 0.594203 | 0.608059 |
| 90% forecast coverage | 0.868147 | 0.867599 |

Promotion is based on forecast improvements at every horizon on both selection
and confirmation, plus the confirmation F1 gain. The tiny current RMSE,
velocity, recovery, and coverage regressions are real and are not averaged
away.

After freezing model choices, the final standard-test block
`200064–200095` measured:

- current position MAE/RMSE `0.089336 / 0.116908 m`;
- velocity RMSE `0.792257 m/s`;
- 0.1/0.25/0.5-second forecast RMSE
  `0.138279 / 0.177703 / 0.232862 m`;
- collision-conditioned improvement over constant velocity
  `30.97% / 54.38% / 50.66%`;
- perturbation recovery `45.30%`, positive on `97.92%` of horizons;
- collision precision/recall/F1 `0.765217 / 0.550000 / 0.640000`;
- 100% distance-gated detection, zero ID switches, zero dropped/non-finite
  forecasts, and nominal-90% coverage `86.95%`.

The report is
`runs/accuracy-closed-structured-v4/evaluation/final-test32/report.md`.
Step 648 improves the prior step-584 frozen test on position, velocity, every
forecast horizon, collision precision/recall/F1, false-positive rate, and
coverage, while perturbation recovery decreases slightly
`45.72% → 45.30%`.

An exhaustive validation threshold sweep found collision probabilities
saturated near `0.018` and `0.998`; no threshold improved F1. The `0.5`
threshold remains because the remaining mistakes are state/timing structural,
not ranking errors. Metric-scale probes were also negative: mean-radius
analytic depth had about `0.795 m` error versus `0.148 m` for learned depth,
and a photometric-radius estimate failed confirmation. A two-frame anisotropic
velocity slope remains only a future opportunity.

### Historical accuracy-v3 structured RGB candidate

The optional synthetic-disc centre extractor consumes RGB only. It subtracts a
row-median background estimate, labels foreground components, splits touching
discs at distance-transform peaks, computes weighted pixel centroids, and
Hungarian-aligns them to learned proposals. Structured centres are applied as a
straight-through forward refinement; the raw learned centre is now retained for
an explicit auxiliary smooth-L1 loss. Normal sphere profiles use foreground
threshold `0.04`; noise-heavy `toy_hard` and `cloud_single_gpu` use `0.08`.

The step-584 candidate at
`runs/accuracy-depth-finetune-v1/checkpoints/best_measurement.pt` is a controlled
512-update measurement continuation from the established step-72 weights. On a
one-time 32-episode confirmation manifest, seeds `100064–100095`, it compared
with the paired step-72 checkpoint as follows:

| Metric | Step 72 | Step 584 candidate |
| --- | ---: | ---: |
| current position MAE (m) | 0.098357 | 0.085103 |
| current position RMSE (m) | 0.131311 | 0.110556 |
| current velocity RMSE (m/s) | 0.765381 | 0.730581 |
| 0.1 s forecast RMSE (m) | 0.152992 | 0.134886 |
| 0.25 s forecast RMSE (m) | 0.189531 | 0.175246 |
| 0.5 s forecast RMSE (m) | 0.241308 | 0.231256 |
| perturbation recovery fraction | 0.422657 | 0.482774 |
| collision F1 | 0.568182 | 0.622222 |
| distance-gated detection recall | 0.998047 | 1.000000 |
| distance-gated ID switches | 0 | 0 |
| 90% forecast coverage | 0.894737 | 0.864857 |

The exact candidate and baseline reports are
`runs/accuracy-structured-peak-v2/depth-finetune-best-confirm32/report.md` and
`runs/accuracy-structured-peak-v2/baseline-step72-confirm32/report.md`.
The candidate improves every listed point/trajectory metric and collision F1,
but worsens nominal uncertainty coverage.

These reports predate the restriction of fast structured refinement to
projected ROIs: both global and fast updates called the full-frame extractor.
They remain useful paired evidence for the checkpoint, not final source-state
metrics.

The finalized ROI-local implementation was first screened on the reused
selection seeds `100032–100063`. Relative to the same checkpoint with
full-frame ordinary refinement, position RMSE improved
`0.128560 -> 0.127250 m`, velocity RMSE
`0.789148 -> 0.780543 m/s`, and the three forecast RMSE values became
`0.150932 / 0.190620 / 0.248704 m`; collision F1 declined by `0.0166` to
`0.588235`. This passed the declared no-regression gate and no later choice was
made from the test split.

The final frozen run used standard-test seeds `200064–200095`:

- current position MAE/RMSE `0.090847 / 0.118600 m`;
- current velocity RMSE `0.812524 m/s`;
- 0.1/0.25/0.5-second forecast RMSE
  `0.141520 / 0.181431 / 0.237585 m`;
- collision-conditioned improvement over constant velocity
  `29.47% / 53.79% / 50.05%`;
- perturbation recovery `45.72%`, positive on `97.40%` of horizons;
- collision precision/recall/F1
  `0.703390 / 0.518750 / 0.597122`;
- 100% distance-gated detection, zero ID switches, zero dropped/non-finite
  forecasts, and nominal-90% coverage `86.62%`.

The report is `runs/accuracy-roi-local-v3/final-test32/report.md`.

### Rejected accuracy experiments

A longer 1,120-step from-scratch run at
`runs/accuracy-structured-physical-v1` did not converge. On selection seeds
`100032–100063`, its best saved candidate measured current RMSE `0.422225 m`,
0.1/0.25/0.5-second RMSE `0.431482 / 0.453668 / 0.525709 m`, perturbation
recovery `18.80%`, collision F1 `0.316940`, and detection recall `66.60%`.
The report is
`runs/accuracy-structured-peak-v2/scratch-best-fresh32/report.md`.
That process had already loaded the earlier unweighted measurement objective
before the final metric loss weighting was implemented, so it is a truthful
rejected run, not a clean test of the completed training protocol.

An experimental `0.02 m` collision-hazard lookahead left physical trajectories
unchanged but reduced confirmation collision F1 from `0.622222` to `0.594406`.
It was rejected and its code/config surface removed. The negative report remains
at
`runs/accuracy-structured-peak-v2/depth-finetune-hazard-0p02-confirm32/report.md`.

A final 256-update continuation with the completed raw-centre objective also
failed promotion. No validation point beat the inherited `0.115593 m`
measurement MAE; validation degraded as high as `0.291207 m`. The run is kept
at `runs/accuracy-final-perception-v3`. The stable step-584 checkpoint remained
the accuracy-v3 selection and later initialized accuracy-v4.

### Earlier temporal RGB velocity experiment

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

- Direct RGB geometry was the dominant accuracy lever in this toy world; the
  step-584 candidate established accurate ROI-local state, and the step-648
  continuation then improved final-test position, velocity, forecasts, and
  collision skill.
- Event-window semantics and missing-edge pooling are correct, but collision F1
  `0.640000` remains below the recommended `0.75` gate. Saturated logits make
  threshold tuning ineffective.
- Final forecast coverage `0.869518` is below the nominal 90% target and needs
  calibration without sacrificing point accuracy.
- Temporal position slopes contain useful velocity information, especially
  for high-error/collision frames, but the current diagonal confidence/update
  rule injects enough correlated error to harm the primary trajectory metrics.
- Drag/restitution updates execute under explicit observability gates but
  remain numerically negligible; useful online identification is unproven.
- The ROI-local structured fast path and raw learned-centre auxiliary objective
  are implemented and tested, but no result establishes the recommended
  collision F1, full occlusion recovery, parameter convergence, or the full
  3,000-step MPS schedule.
