# Project Orpheus

Orpheus is a local PyTorch research prototype for a persistent,
uncertainty-aware physical world belief. It predicts a prior, projects expected
sensor measurements, associates observations, computes innovation, corrects the
posterior, updates observable slow parameters, and immediately revises future
rollouts.

Its scaling principle is predictive abstraction: retain the smallest
executable representation that predicts well. A freely moving ball can be a
point with a trajectory; contact can refine the same persistent entity into a
rigid sphere. Learned foundation-model features and residual tokens may enrich
that state, but `WorldBelief` remains authoritative.

```text
RGB + calibration + timestamp
             │
             ▼
global discovery / ROI-local residual measurement
             │
             ▼
associate → innovate → correct → identify
             │
             ▼
       persistent WorldBelief
             │
             ▼
analytic + modal + interaction + event dynamics
             │
             ▼
    arbitrary-time future rollout
```

The first vertical slice uses a deterministic synthetic RGB sphere world with
collisions, occlusion, camera motion, and variable drag/restitution. Simulator
state is reserved for labels, evaluation, tests, and a clearly marked debug
oracle. RGB-only operation must not consume oracle state.

## Current status

The first complete RGB-only vertical slice is runnable and tested. It includes
the simulator, typed persistent belief, hybrid dynamics, oracle debug path,
global/ROI RGB measurements, association and innovation, fast correction,
occlusion-aware lifecycle, observability-gated parameter updates, training,
held-out evaluation, and demo export. Evaluation now supports explicit
fresh-validation seed manifests, current velocity/correction evidence, and
collision-conditioned model/baseline errors.

Accuracy-v3 added an optional, RGB-only structured centre measurement for the
synthetic disc world. It estimates the row-wise background from pixels, labels
foreground components, separates touching discs with distance-transform peaks,
and assigns centres to learned proposals with Hungarian matching. This is a
synthetic-world image prior, not simulator-state input; the modality-independent
belief and online filter contracts are unchanged. The sphere profiles enable
it, with a stricter `0.08` foreground threshold in the noisy `toy_hard` and
`cloud_single_gpu` profiles.

Accuracy-v4 continues the selected perception state for 64 causal closed-loop
RGB updates and promotes the validation-selected step-648 rollout checkpoint.
On paired confirmation seeds it improves every forecast horizon and collision
F1 over step 584, with tiny mixed current-state, velocity, and perturbation
tradeoffs.

This remains implementation evidence, not a completed research milestone. The
frozen step-648 checkpoint was evaluated on 32 reserved-test episodes, seeds
`200064–200095`. It measured `0.0893 / 0.1169 m` current position MAE/RMSE,
`0.7923 m/s` velocity RMSE, `0.1383 / 0.1777 / 0.2329 m` forecast RMSE at
0.1/0.25/0.5 seconds, `45.30%` perturbation recovery, collision F1 `0.6400`,
100% distance-gated detection, zero ID switches, and 90% forecast coverage of
`86.95%`. RGB-only runtime used four global and 12 ROI-local updates in the
demo and improved current/future error after observations. Collision F1
remains below the recommended `0.75` gate, useful online physical-parameter
identification is not established, and the full MPS schedule has not run. See
[`project/STATUS.md`](project/STATUS.md) for exact commands, current metrics,
and limitations.

A bounded three-sample, persistent-ID RGB motion history is implemented as an
opt-in experiment. On 16 fresh validation episodes it improved velocity RMSE
but regressed localization and aggregate forecasts, so it remains disabled in
the public profiles. A controlled continuation raised collision F1 to 0.1216
but worsened the primary physical metrics and was likewise not promoted.

The scaled profile now also keeps scarce global scale anchors in a separate
persistent-ID ring, so centre-only ROI updates cannot evict the multi-frame
depth evidence. A robust axis-local trajectory estimate corrects only the
calibrated camera-depth component of `WorldBelief`. With unchanged model
weights, current position and every recursive 0.1–1.0-second position horizon
improved on two disjoint paired MPS blocks. Velocity and collision F1 regressed,
so this is a position/depth improvement rather than a claim that event dynamics
are solved; exact metrics and reports are in `project/STATUS.md`.

## Quick start

Use the existing `orpheus` environment. PyTorch is an externally managed
prerequisite and must not be reinstalled by this package.

```bash
conda activate orpheus
pip install -e ".[dev]"
python train.py --config configs/toy_mps.yaml
python evaluate.py --config configs/toy_mps.yaml --checkpoint <path>
python demo.py --config configs/toy_mps.yaml --checkpoint <path>
pytest
```

For the multi-day shared-model accuracy campaign:

```bash
python train.py \
  --config configs/sustained_accuracy_mps_v3.yaml \
  --initialize-from \
    runs/20260730-192625-scaled-sustained-e2e-v1/checkpoints/best_measurement.pt \
  --run-name "$(date -u +%Y%m%d-%H%M%S)-scaled-sustained-v3" \
  --device mps
```

This profile runs 8,192 paired global/fast RGB updates followed by 8,192
supported causal updates over the balanced eight-scenario pool. It uses
40-frame mature forecast anchors, keeps the imported reference separate from
the mutable training state, skips unsupported causal draws instead of
consuming empty optimizer steps, and applies an interaction-local clip before
the whole-model clip. Promotion checks velocity, detection, lifecycle, events,
identity, every axis/horizon, calibration, coverage, and scenario support.
The v1/v2 profiles and runs remain audit controls; neither is a convergence
result. See `project/ACCURACY_AUDIT.md` and `project/TRAINING.md` for exact
evidence and the required medium qualification before a sustained launch.

The historical v3 profile runs the convolutional RGB backbone and ROI path on MPS, pins
the small global proposal transformer to CPU through differentiable copies,
and switches the full persistent model to CPU at the causal boundary. The
proposal fallback avoids a reproduced PyTorch 2.10 MPS NaN-gradient kernel on
finite full-resolution features; a matched benchmark found the branch-heavy
online filter/dynamics backward pass substantially faster on CPU on this Mac.
`--resume` preserves both configured phase devices and the exact absolute
sample/RNG/source protocol. Use `--initialize-from` for a changed device,
objective, curriculum, or source implementation.

For an immutable promotion decision for a legacy CPU-fallback candidate, run
the full trainer manifest once for the protected reference and candidate on an
active-Aqua MPS session:

```bash
python scripts/replay_promotion_mps.py \
  --config configs/attention_pilot_mps.yaml \
  --reference runs/<run>/checkpoints/best_rollout.pt \
  --candidate runs/<run>/checkpoints/validation_step_000128.pt \
  --output runs/$(date -u +%Y%m%d-%H%M%S)-mps-promotion-replay
```

The command is a gate, not a generic benchmark: it exits successfully only
when the candidate improves on MPS while passing the existing per-axis,
per-horizon, lifecycle, identity, collision/event, calibration, and support
guardrails. Its report records both checkpoint SHA-256s and the validation
protocol hash. New `attention_pilot_mps` runs already execute their selector
directly on MPS.

For checkpoint selection without reusing trainer-validation or test seeds:

```bash
python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint <path> \
  --split validation \
  --seed-protocol fresh_validation \
  --seed-offset 64 \
  --set evaluation.episodes=16
```

Use one disjoint validation offset for candidate selection and a later untouched
offset for confirmation; do not select checkpoints on the reserved test split.

To collect exact-timestamp causal RGB event windows and fit the experimental
uncertainty-aware gate:

```bash
python scripts/train_rgb_change_point_gate.py \
  --config configs/scaled_curriculum.yaml \
  --checkpoint <path> \
  --device mps \
  --train-episodes 8 \
  --validation-episodes 8 \
  --validation-seed-offset 256 \
  --gate-type mlp \
  --hidden-features 8 \
  --fit-outgoing-proposal \
  --proposal-hidden-features 8
```

The output contains cached feature tensors, a report, resolved config, and a
weights-identical checkpoint with explicit gate coefficients. The scaled
profile keeps both the gate and outgoing proposal disabled: current learned
candidates did not pass the paired downstream velocity gate. The proposal is
consumed on the exact causal frame selected by its gate; later post-event
samples return to the ordinary estimator. Cached tensors can be supplied with
`--train-cache` and `--validation-cache` to refit without rerunning RGB
perception.

For the deterministic convergence/debug run:

```bash
python train.py --config configs/tiny_overfit.yaml --run-name tiny-debug
```

Generated training, evaluation, and demo directory basenames begin with a UTC
`YYYYMMDD-HHMMSS-` timestamp, so ordinary filename sorting puts the newest
artifact last. Explicit `--run-name` and `--output` values are treated as
human-readable labels; the command's JSON result contains the actual path.
Superseded demos are retained under `demo_outputs/archive/`.

The current local RGB-only selection bundle is
`runs/20260727-193657-selected-contact-confidence-v1/`; the latest visual
result is
`demo_outputs/20260727-193538-contact-confidence-v1/online_correction.gif`.
These generated artifacts are gitignored. The seven-regime interaction
curriculum is configured in `configs/tiny_interactions.yaml`.

The user-provided locally built PyTorch in the existing environment is
MPS-enabled. The corrected
two-phase smoke at
`runs/20260801-231521-audit-v2-final-verified-smoke/` exercised one hybrid RGB
update, two persistent causal updates, selector validation, checkpoint
round-trip, and byte-preserving no-op resume. It is wiring evidence, not an
accuracy or convergence result. See `project/STATUS.md` for exact values.

## Documentation

- [Authoritative specification](PROJECT_SPEC.md)
- [Current status](project/STATUS.md)
- [Architecture](project/ARCHITECTURE.md)
- [Data contracts](project/DATA_CONTRACTS.md)
- [Predictive abstractions](project/PREDICTIVE_ABSTRACTIONS.md)
- [Getting started](docs/getting_started.md)
- [Known work](project/TASKS.md)

Licensed under Apache-2.0.
