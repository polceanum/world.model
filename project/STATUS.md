# Project status

**Date:** 2026-07-27
**Specification:** `PROJECT_SPEC.md` 1.0  
**Current state:** runnable RGB-only Milestone 1 vertical slice with converged
tiny localization/forecasting, explicit fresh-seed/velocity evidence, and
corrected event semantics; collision/event accuracy acceptance remains open

## What works

- A deterministic 3-D RGB sphere simulator provides explicit timestamps,
  calibrated cameras, identities, physical state, depth-ordered
  rendering/occlusion, collisions/events, and disjoint seed splits.
- `WorldBelief` is the persistent, modality-independent runtime truth. Every
  ingest predicts a prior, projects expected measurements, associates,
  computes innovation, corrects the posterior, updates lifecycle/observable
  slow parameters, and revises arbitrary-time future rollouts.
- Dynamics combine analytic kinematics, bounded modal state, structured
  interactions, explicit event jumps, learned residuals, and uncertainty
  propagation.
- RGB includes intermittent global discovery and ordinary residual ROI
  updates. The fast path remains active (12 of 16 demo updates), but its
  inverse-depth residual is reliability-gated at the analytic prior until a
  trained checkpoint proves positive held-out metric-space improvement.
- Fixed-dataset RGB pretraining now sweeps every frame for every loader batch,
  rather than coupling episode batches to only even or odd frames.
- Measurement validation spans configured frames and selects
  `best_measurement.pt` by calibrated backprojected world-position MAE rather
  than the summed, possibly negative Gaussian NLL objective.
- Closed-loop training restores the best localized perception checkpoint and
  applies a configurable 10x learning-rate reduction. Resume reapplies the
  correct phase learning rate after loading optimizer state.
- Closed-loop global discovery/backbone weights remain trainable only for a
  configurable adaptation window, then freeze while the fast ROI path,
  filter, dynamics, and identifier continue training. This prevents the
  localization drift observed during unrestricted continuation.
- Fast ROI outputs are supervised at every usable prior frame in persistent
  belief-slot order; they are not incorrectly rematched by their own current
  output values.
- Collision occurrence is aggregated over every internal physics substep in a
  rollout segment while persistent motion-mode logits remain instantaneous.
  Training and evaluation insert exact `[h-dt_obs, h]` query boundaries so
  frame event labels are never compared with a cumulative or arbitrary
  horizon segment.
- The correction objective retains the small specification-required sparsity
  regularizer and now also penalizes current/future posterior updates that fail
  to improve over a detached prior. Rare collision positives receive a
  bounded, explicitly configured BCE weight.
- Evaluation can derive a deterministic fresh-validation manifest from
  checkpoint provenance, persist its exact seeds/non-overlap status, report
  current and ordinary-correction velocity metrics, and compare model and
  baselines on identical future-collision masks.
- RGB has a separate bounded persistent-ID temporal position history with
  explicit timestamps and uncertainty. It can provide a cheap causal
  velocity-only correction without new weights or history re-encoding, but is
  disabled in public profiles because its current accuracy tradeoff fails the
  overall validation gate.
- The debug oracle is registered only when explicitly enabled. Every result
  below uses RGB plus known calibration; simulator state is used only for
  supervision, evaluation alignment, and explicitly labelled baselines.

## Environment

- Conda environment: `orpheus`
- Python: 3.10.20
- Process architecture: x86_64
- PyTorch: 2.10.0, installed build preserved unchanged
- MPS: compiled, but unavailable to the current final-validation process;
  earlier recorded real forward/backward optimizer steps completed on `mps`
- CUDA: unavailable
- Precision: float32

The current launcher process reports MPS unavailable, so three
hardware-conditional tests skip. Earlier direct runs exercised the same
MPS-specific paths successfully; the preserved run evidence is described
below.

## Current converged CPU evidence

Training command:

```bash
conda run -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name convergence-tiny-cpu-v1
```

Observed result:

- 70 optimizer steps on CPU in 59.845 s;
- 64 global RGB pretraining steps at learning rate `0.002`, covering all 16
  frames of eight deterministic training episodes;
- six full RGB closed-loop steps at learning rate `0.0002`;
- best 16-frame/four-episode validation world-position MAE: 0.422427 m at
  step 64, with 0.734375 recall/precision at the 0.5 m gate;
- that best localized checkpoint was restored before the closed-loop stage;
- closed-loop validation rollout loss: 0.277626;
- final training-window future correction improvement: +0.008551 m;
- finite losses and gradients, with no oracle runtime input.

Artifacts:

- `runs/convergence-tiny-cpu-v1/checkpoints/best_rollout.pt`
- `runs/convergence-tiny-cpu-v1/checkpoints/best_measurement.pt`
- `runs/convergence-tiny-cpu-v1/checkpoints/last.pt`
- `runs/convergence-tiny-cpu-v1/metrics.jsonl`
- `runs/convergence-tiny-cpu-v1/train_summary.json`

The best and last checkpoint contain the same validated step-70 tensors for
this run. `best_measurement.pt` is the selected step-64 perception handoff.

### Primary two-episode test protocol

```bash
conda run -n orpheus python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/convergence-tiny-cpu-v1/checkpoints/best_rollout.pt \
  --split test \
  --device cpu \
  --output runs/convergence-tiny-cpu-v1/evaluation/best-test
```

Observed held-out RGB-only results:

- current position MAE/RMSE: 0.218436 / 0.289611 m;
- distance-gated recall and precision at 0.5 m: 0.59375 / 0.59375
  (38 of 64 object-frames), up from 0/64 in the earlier short run;
- model position RMSE at 0.10 / 0.25 / 0.50 s:
  0.296358 / 0.224920 / 0.180852 m;
- constant-velocity RMSE:
  0.286786 / 0.246651 / 0.535446 m;
- static RMSE:
  0.231903 / 0.189103 / 0.186144 m;
- 0.50 s model RMSE is 66.22% below constant velocity and 2.84% below static;
- injected-perturbation error reduction: 0.141265 m, or 27.71%, positive for
  all 12 evaluated object-horizons;
- 90% forecast coverage: 93.86%;
- distance-gated ID switch rate: 0 over 38 associations;
- no dropped forecasts and no non-finite output;
- collision F1: 0.

Reports:

- `runs/convergence-tiny-cpu-v1/evaluation/best-test/report.md`
- `runs/convergence-tiny-cpu-v1/evaluation/last-test/report.md`
- `runs/convergence-tiny-cpu-v1/evaluation/best-measurement-test/report.md`

The closed-loop stage improved over the step-64 perception checkpoint:
current MAE 0.249597 -> 0.218436 m, detection recall 0.484375 -> 0.59375,
0.50 s forecast RMSE 0.232091 -> 0.180852 m, and perturbation recovery
6.80% -> 27.71%.

### Wider eight-episode held-out check

```bash
conda run -n orpheus python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/convergence-tiny-cpu-v1/checkpoints/best_rollout.pt \
  --split test \
  --device cpu \
  --set evaluation.episodes=8 \
  --output runs/convergence-tiny-cpu-v1/evaluation/best-test-8episodes
```

Observed over 256 held-out object-frames:

- current position MAE/RMSE: 0.182494 / 0.239292 m;
- 0.5 m recall/precision: 0.75 / 0.75 (192/256);
- model RMSE at 0.10 / 0.25 / 0.50 s:
  0.239256 / 0.187325 / 0.162259 m;
- constant-velocity RMSE:
  0.234788 / 0.260275 / 0.491278 m;
- static RMSE:
  0.195517 / 0.166665 / 0.172445 m;
- 0.50 s model RMSE is 66.97% below constant velocity and 5.91% below static;
- injected-perturbation reduction: 0.069255 m or 19.59%, positive on 72.92%
  of 48 object-horizons;
- 90% forecast coverage: 96.05%;
- gated ID switch rate: 0 over 192 associations;
- collision F1: 0;
- no dropped forecasts and no non-finite output.

Report:
`runs/convergence-tiny-cpu-v1/evaluation/best-test-8episodes/report.md`.
The wider perturbation result is 0.41 percentage points below the recommended
20% gate, so that gate is not claimed as achieved on the larger sample.

### Accuracy and event follow-up

A controlled continuation protected global perception after the six-step
closed-loop adaptation window while training the ROI/filter/dynamics modules:

```bash
conda run --no-capture-output -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name accuracy-closed-frozen-94 \
  --resume runs/convergence-tiny-cpu-v1/checkpoints/best_rollout.pt \
  --set training.steps=94 \
  --set training.eval_every=8 \
  --set training.checkpoint_every=8 \
  --set training.log_every=2
```

The 24 new CPU steps completed in 176.439 s. Step 72 became the
validation-selected `best_rollout.pt` and remained best through step 94. The
step-94 `last.pt` preserved and slightly improved position accuracy on the
repeated diagnostic test seeds:

```bash
conda run --no-capture-output -n orpheus python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/accuracy-closed-frozen-94/checkpoints/last.pt \
  --split test \
  --device cpu \
  --set evaluation.episodes=8 \
  --output \
    runs/accuracy-closed-frozen-94/evaluation/last-test-8episodes-exact-events
```

With corrected exact-window event scoring, this exploratory comparison found:

- current position MAE: 0.178773 m versus 0.182494 m at step 70;
- 0.5 m recall/precision: 0.753906 / 0.753906;
- model RMSE at 0.10 / 0.25 / 0.50 s:
  0.237282 / 0.186030 / 0.161387 m;
- constant-velocity RMSE: 0.232863 / 0.258475 / 0.490275 m;
- 0.50 s model RMSE is 67.08% below constant velocity and 6.09% below static;
- perturbation reduction: 19.49%, positive on 72.92% of evaluated horizons;
- 90% coverage: 96.60%; ID switches: 0 over 193 gated associations;
- exact-window collision F1: 0.028169.
- informative online restitution updates: pre/post MAE
  0.10399209 / 0.10399210 over four samples, mean signed improvement
  -0.000000015 and update magnitude 0.000000045;
- informative drag updates: pre/post MAE 0.04388363 / 0.04388311 over
  53 samples, mean signed improvement 0.000000529 and update magnitude
  0.000001914;
- no complete reliably anchored visible-to-fully-occluded-to-visible
  transition occurred, so sequence-aware growth/recovery values are correctly
  null rather than inferred from pooled frames.

Artifacts:

- `runs/accuracy-closed-frozen-94/checkpoints/last.pt`
- `runs/accuracy-closed-frozen-94/train_summary.json`
- `runs/accuracy-closed-frozen-94/evaluation/last-test-8episodes-exact-events/report.md`

The corresponding RGB-only demo command was:

```bash
conda run --no-capture-output -n orpheus python demo.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/accuracy-closed-frozen-94/checkpoints/last.pt \
  --device cpu \
  --output demo_outputs/accuracy-closed-frozen-94
```

It uses four global and 12 fast ROI updates. Mean ordinary current/future
prior-to-posterior improvements are +0.008432 m / +0.011079 m:

- `demo_outputs/accuracy-closed-frozen-94/online_correction.gif`
- `demo_outputs/accuracy-closed-frozen-94/parameter_estimates.png`
- `demo_outputs/accuracy-closed-frozen-94/summary.json`
- `demo_outputs/accuracy-closed-frozen-94/frames/`

Applying only the corrected event semantics to the unchanged step-70
checkpoint raised collision F1 from 0 to 0.055556 without changing weights.
That report is at
`runs/accuracy-events-v2/evaluation/pretrain-checkpoint-test-8episodes/report.md`.

A further 32-step continuation with exact-window, positive-balanced event loss
was also run rather than assumed successful:

```bash
conda run --no-capture-output -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name accuracy-events-balanced-102 \
  --resume runs/convergence-tiny-cpu-v1/checkpoints/best_rollout.pt \
  --set training.steps=102 \
  --set training.eval_every=8 \
  --set training.checkpoint_every=8 \
  --set training.log_every=2
```

It completed in 277.627 s and improved its sampled validation rollout loss to
0.270487, but did not generalize: eight-episode current MAE was 0.183974 m,
0.50 s RMSE 0.168842 m, perturbation recovery 14.86%, and collision F1
0.027397. It is retained as a truthful negative result and is not promoted
over the safer position checkpoint.

The step-70, step-94, and balanced-event comparisons repeatedly inspect the
same fixed eight test episodes. They are exploratory diagnostics, not an
independent model-selection protocol. Step 72 remains the validation-selected
checkpoint; the small apparent test improvement of `last.pt` requires
confirmation on a fresh, larger held-out seed set.

## Fresh validation, velocity evidence, and temporal ablation

The evaluator now reserves an explicit checkpoint-selection block after the
trainer's validation episodes:

```bash
conda run --no-capture-output -n orpheus python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/accuracy-closed-frozen-94/checkpoints/best_rollout.pt \
  --split validation \
  --seed-protocol fresh_validation \
  --device cpu \
  --set model.rgb.temporal_velocity_enabled=false \
  --set evaluation.episodes=16 \
  --output runs/temporal-rgb-evidence/fresh-validation-final-baseline
```

This 56.85-second CPU run used exactly seeds `100004–100019`. The report
asserts no overlap with trainer validation (`100000–100003`) or the reserved
test range. It is model-selection validation evidence, not final test
acceptance. The unchanged step-72 checkpoint produced:

- current position MAE/RMSE: `0.186991 / 0.239613 m`;
- distance-gated current velocity MAE/RMSE:
  `0.647751 / 1.369454 m/s` over 377 object-frames;
- ordinary velocity prior/posterior norm error:
  `1.747554 / 1.745960 m/s`, only `0.001594 m/s` improvement;
- model RMSE at 0.10 / 0.25 / 0.50 seconds:
  `0.236517 / 0.189670 / 0.174269 m`;
- collision F1 `0.042553`;
- perturbation recovery `20.0935%`, positive on `78.125%` of 96 horizons;
- 90% forecast coverage `97.7522%`, zero ID switches, zero dropped/non-finite
  forecasts.

Future-collision-conditioned model RMSE at 0.10 / 0.25 / 0.50 seconds was
`0.149769 / 0.137729 / 0.174269 m`, respectively
`22.08% / 56.87% / 65.94%` below constant velocity on the exact same masks.

The implemented temporal path keeps a separate sensor-local three-position
history keyed by persistent object ID. It survives global/ROI feature-cache
changes, requires strictly increasing timestamps and nonambiguous
associations, and performs a velocity-only diagonal correction. With an
explicit experimental variance ceiling of `1.0 (m/s)²`:

```bash
conda run --no-capture-output -n orpheus python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/accuracy-closed-frozen-94/checkpoints/best_rollout.pt \
  --split validation \
  --seed-protocol fresh_validation \
  --device cpu \
  --set evaluation.episodes=16 \
  --set model.rgb.temporal_velocity_enabled=true \
  --set model.rgb.temporal_velocity_variance_ceiling=1.0 \
  --output runs/temporal-rgb-evidence/fresh-validation-final-temporal
```

Velocity RMSE improved to `1.309964 m/s`, ordinary correction improvement to
`0.025985 m/s`, short collision-conditioned RMSE to `0.140309 m`, and
collision F1 to `0.055172`. However, current position MAE worsened to
`0.190923 m`, 0.25-second RMSE to `0.201318 m`, perturbation recovery to
`19.2569%`, and calibration/detection also regressed. Ceilings two/four and
history size four showed the same tradeoff. No inference-only temporal setting
was promoted; the public profiles keep it disabled and use uncapped propagated
uncertainty when explicitly enabled without an override.

A controlled frozen-global continuation was run rather than inferred:

```bash
conda run --no-capture-output -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name temporal-continuation-94 \
  --resume runs/accuracy-closed-frozen-94/checkpoints/best_rollout.pt \
  --set model.rgb.temporal_velocity_enabled=true \
  --set model.rgb.temporal_velocity_variance_ceiling=1.0 \
  --set training.steps=94 \
  --set training.eval_every=8 \
  --set training.checkpoint_every=8 \
  --set training.log_every=2
```

Twenty-two new CPU steps completed in `183.147 s`, with finite gradients and
no oracle runtime input. Step 94 lowered the small trainer-validation rollout
loss to `0.249018`. On the larger fresh manifest with temporal correction it
improved velocity RMSE to `1.277519 m/s` and collision F1 to `0.121622`, but
position MAE worsened to `0.196397 m`, 0.10/0.25/0.50-second RMSE to
`0.243738 / 0.207295 / 0.184454 m`, and perturbation recovery to `11.843%`.
Disabling temporal inference on the same weights was also worse. This run is a
truthful negative result and step 72 remains the promoted checkpoint.

Artifacts:

- `runs/temporal-rgb-evidence/fresh-validation-final-baseline/report.md`
- `runs/temporal-rgb-evidence/fresh-validation-final-temporal/report.md`
- `runs/temporal-rgb-evidence/fresh-validation-temporal-ceiling2/report.md`
- `runs/temporal-rgb-evidence/fresh-validation-temporal-ceiling4/report.md`
- `runs/temporal-continuation-94/checkpoints/best_rollout.pt`
- `runs/temporal-continuation-94/evaluation/fresh-validation-enabled/report.md`
- `runs/temporal-continuation-94/evaluation/fresh-validation-disabled/report.md`

### Demo

```bash
conda run -n orpheus python demo.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/convergence-tiny-cpu-v1/checkpoints/best_rollout.pt \
  --device cpu \
  --output demo_outputs/convergence-tiny-cpu-v1
```

The held-out 16-frame RGB-only demo used four global and 12 fast ROI updates.
Mean ordinary prior-to-posterior improvement is now positive:
+0.007777 m for current state and +0.010584 m for future error. Artifacts:

- `demo_outputs/convergence-tiny-cpu-v1/online_correction.gif`
- `demo_outputs/convergence-tiny-cpu-v1/parameter_estimates.png`
- `demo_outputs/convergence-tiny-cpu-v1/summary.json`
- `demo_outputs/convergence-tiny-cpu-v1/frames/`

## MPS evidence

Earlier reduced explicit MPS smoke:

```bash
conda run -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name milestone1-mps-smoke-final \
  --device mps \
  --set training.steps=3 \
  --set training.rgb_pretrain_steps=2 \
  --set training.tbptt_steps=2 \
  --set training.batch_size=1 \
  --set training.train_episodes=1 \
  --set training.validation_episodes=1 \
  --set training.eval_every=3 \
  --set training.checkpoint_every=3 \
  --set training.log_every=1 \
  --set evaluation.episodes=1
```

Observed result: three finite optimizer steps in 90.996 s on `mps`, including
global and differentiable fast ROI backward paths. A separate reduced two-step
run of the full 96x96 `toy_mps` architecture completed in 132.157 s at
`runs/milestone1-toy-mps-scaled-smoke`. These are hardware compatibility
checks, not convergence claims. The full 3,000-step schedule remains unrun.

## Final validation

Full suite:

```bash
conda run --no-capture-output -n orpheus pytest
```

Passed: 148 tests with three MPS-unavailable skips in 34.46 s.

Lint and formatting:

```bash
conda run -n orpheus ruff check .
conda run -n orpheus ruff format --check .
```

Passed: all checks clean; 150 Python files formatted.

Bytecode compilation:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-final-pycache PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
```

Passed.

Final public smoke workflow:

```bash
conda run --no-capture-output -n orpheus python train.py \
  --config configs/toy_smoke.yaml \
  --run-name accuracy-final-smoke
```

Passed: 12 finite CPU optimizer steps in 186.092 s (four RGB pretraining,
eight full closed-loop), both best checkpoints and `last.pt` were written, the
fast path received supervision on seven frames in the final batch, global
perception was frozen after its configured adaptation window, and
`oracle_runtime_input_used` was false. This intentionally tiny smoke is a
wiring check, not an accuracy result. Artifacts are under
`runs/accuracy-final-smoke/`.

Default MPS plan:

```bash
conda run -n orpheus python train.py \
  --config configs/toy_mps.yaml --dry-run
```

Passed with PyTorch 2.10.0, 96x96 images, 72 frames, RGB-only runtime, and the
configured 3,000-step schedule. It selected `cpu` because MPS was unavailable
to this final-validation process.

## Known limitations and open acceptance gates

- Collision semantics are now correct and F1 is nonzero, but accuracy remains
  poor (fresh-validation F1: 0.042553 versus the recommended 0.75). The
  rejected temporal continuation reached 0.121622 while materially worsening
  state/forecast/recovery metrics, so it is not a promoted accuracy result.
  Focused exact-state tests and a non-persisted scratch oracle diagnostic point
  to RGB state accuracy and event uncertainty—not loss of collisions between
  internal substeps—as the limiting factors.
- Fresh-validation perturbation reduction is 20.09%, but it has now been used
  during model development and needs confirmation on a new untouched manifest.
  The older wider test result was 19.59%, narrowly below the recommended 20%
  gate.
- Global vertical-centre predictions remain less image-dependent than target
  motion. More varied measurement training and explicit component losses are
  still warranted.
- Fast inverse-depth residual learning is safely disabled, not solved. It must
  be trained with belief-slot-aligned cached sequences and enabled only after a
  held-out per-mode correction gate passes.
- Repeated correlated RGB measurements can still make the diagonal filter
  overconfident. Fresh-validation 90% coverage is 97.75%, but explicit
  correlation handling remains absent. The new temporal velocity measurement
  is disabled by default because treating overlapping three-frame estimates as
  independent improves velocity RMSE while degrading aggregate closed-loop
  accuracy.
- Directional before/after metrics confirm that physical parameter updates are
  currently numerically tiny. On fresh validation, restitution and drag signed
  error reduction are slightly negative (`-3.67e-8` over 13 informative
  updates and `-6.95e-8` over 104, respectively). Useful identification is not
  established.
- The evaluated short sequences contain no usable held-out distance-gated
  occlusion interval; rendered occlusion recovery remains supported mainly by
  focused geometry/lifecycle tests.
- At this tiny scale fast ROI latency (42.83 ms) is not lower than global
  latency (38.97 ms).
- Max-aggregating substep collision logits preserves thresholded
  "occurred anywhere" semantics, but is not a calibrated probability-of-union
  calculation.
- Belief-slot-aligned fast supervision avoids rematching conditioned outputs,
  but its per-frame belief-to-label assignment can still switch under close
  crossings. Sequence-level training identity alignment remains future work.
- The evaluator now reports collision-conditioned matched model/baseline
  forecasts, but still lacks the full physics-violation and saved failure-plot
  suite.
- Multiple-hypothesis association, multi-frame tentative births, estimated
  camera pose, continuous collision timing, real video, and a second modality
  remain future work.

## Next concrete tasks

1. Add correlation-aware temporal RGB velocity uncertainty, an innovation/
   observability gate, and a validation-selected probabilistic collision head;
   the current independent diagonal update and exact-window event-balanced
   loss did not generalize.
2. Train fast ROI updates on belief-slot-aligned cached sequences with
   teacher-forced/jittered valid ROIs, then validate and enable the depth
   residual through a per-mode improvement gate.
3. Add correlation-aware measurement uncertainty and evaluate the implemented
   expansion/recovery metrics on an occlusion-rich held-out split.
4. Add physics-violation metrics and saved collision/forecast failure plots.
5. Run the full `toy_mps` schedule and a larger event/occlusion-balanced split.

All `runs/` and `demo_outputs/` paths above are real local artifacts but are
gitignored by design; checkpoints and generated media are not published in the
source commit.
