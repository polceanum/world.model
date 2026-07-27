# Project Orpheus

Orpheus is a local PyTorch research prototype for a persistent,
uncertainty-aware physical world belief. It predicts a prior, projects expected
sensor measurements, associates observations, computes innovation, corrects the
posterior, updates observable slow parameters, and immediately revises future
rollouts.

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

PyTorch 2.10.0 in the existing environment is MPS-enabled. A reduced explicit
MPS training smoke has exercised both global RGB and differentiable fast ROI
backward paths. The full 3,000-step `toy_mps` protocol remains an overnight
experiment, not a result claimed here.

## Documentation

- [Authoritative specification](PROJECT_SPEC.md)
- [Current status](project/STATUS.md)
- [Architecture](project/ARCHITECTURE.md)
- [Data contracts](project/DATA_CONTRACTS.md)
- [Getting started](docs/getting_started.md)
- [Known work](project/TASKS.md)

Licensed under Apache-2.0.
