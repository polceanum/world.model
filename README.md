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
global discovery / residual ROI measurement
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

This is implementation evidence, not a completed research milestone. The
current deterministic RGB checkpoint reaches 75.39% distance-gated localization
over eight held-out episodes and reduces 0.5-second RMSE from 0.491 m for
constant velocity to 0.161 m. Ordinary global/fast corrections improve the
held-out demo. Collision window semantics are now correct, but measured event
skill remains weak (the promoted checkpoint scores 0.0426 F1 on fresh
validation and 0.0556 on the older exploratory test); wider untouched-test
perturbation recovery is narrowly below its recommended gate, and the full MPS
schedule has not run.
See [`project/STATUS.md`](project/STATUS.md) for exact commands, metrics, and
limitations.

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
  --set evaluation.episodes=16
```

For the deterministic convergence/debug run:

```bash
python train.py --config configs/tiny_overfit.yaml --run-name tiny-debug
```

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
