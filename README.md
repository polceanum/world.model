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

Broad heterogeneous campaigns are paused. The current specification first
requires a cheap, identifiable RGB-to-state-to-rollout unit to converge before
association, lifecycle, contact, camera motion, or learned dynamics are added
back. That unit now passes with `0.007644 m` final RGB state RMSE and
`0.007991 m` short-rollout RMSE through an ordinary differentiable
inverse-rendering, calibrated-backprojection, and analytic-kinematics graph.

The full `OnlineWorldModel` train/evaluate/demo path remains runnable, but no
older multi-day campaign is a current launch recommendation or promotion
incumbent. Historical experiments and their limitations remain in
[`project/STATUS.md`](project/STATUS.md); the minimal qualification workflow is
below.

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

### Minimal differentiable convergence qualification

The one-sphere convergence ladder is a deliberately isolated developer
qualification, not a second deployment model. Run it through its dedicated
command with fresh output paths:

```bash
conda run -n orpheus python scripts/run_minimal_toy_ladder.py \
  --config configs/minimal_differentiable_toy_cpu.yaml \
  --report runs/minimal_differentiable_toy_v2/report.json \
  --checkpoint runs/minimal_differentiable_toy_v2/model.pt
```

The successful JSON report is the complete oracle, train, selector,
confirmation, and one-shot final evaluation record. A failed report stops at
the first rejected rung. The checkpoint is an atomic, versioned
weights-only project checkpoint at step 72; it is suitable for initialization,
not exact optimizer resume. The report records its SHA-256 digest. Reload it
into the matching minimal estimator with:

```python
from world_model.training.checkpointing import load_model_weights
from world_model.training.minimal_toy import DifferentiableToyStateEstimator
from world_model.utils.config import load_config

config = load_config("configs/minimal_differentiable_toy_cpu.yaml")
model = DifferentiableToyStateEstimator(
    image_size=config.simulator.image_size,
    world_radius_m=config.simulator.radius_range[0],
)
load_model_weights("runs/minimal_differentiable_toy_v2/model.pt", model=model,
                   expected_config=config)
```

`train.py`, `evaluate.py`, and `demo.py` remain the supported
`OnlineWorldModel` workflow and must not be used with this deliberately smaller
estimator. The ladder exists to close identifiability and gradient correctness
before another complexity rung enters that production path.

### Temporal and long-horizon qualification

The first generalization rung fits one anchor state from sixteen RGB frames,
then evaluates cheap analytic rollouts through two seconds. Development may use
only the frozen development manifests and emits a review checkpoint without
opening selector, confirmation, or final data:

```bash
conda run -n orpheus python scripts/run_temporal_free_motion_ladder.py \
  --phase development \
  --config configs/temporal_free_motion_toy_cpu.yaml \
  --report runs/temporal_free_motion_toy_v1/development_report.json \
  --checkpoint runs/temporal_free_motion_toy_v1/development_model.pt
```

Run this only from a clean committed source tree. The development report binds
the checkpoint SHA-256; independently compute and review both that checkpoint
digest and the report's own digest. The same clean commit may then consume the
protected sets exactly once:

```bash
conda run -n orpheus python scripts/run_temporal_free_motion_ladder.py \
  --phase qualification \
  --config configs/temporal_free_motion_toy_cpu.yaml \
  --report runs/temporal_free_motion_toy_v1/qualification_report.json \
  --checkpoint runs/temporal_free_motion_toy_v1/development_model.pt \
  --development-report runs/temporal_free_motion_toy_v1/development_report.json \
  --reviewed-checkpoint-sha256 <reviewed-checkpoint-sha256> \
  --reviewed-report-sha256 <reviewed-report-sha256>
```

The qualification command creates a durable access ledger before generating
protected examples and stops at the first failed split. It never trains or
changes the reviewed weights. This rung is still a standalone identifiability
test; the next rung must reproduce it through the public online belief API
before any scene or network-capacity expansion.

The former grounded, attention, change-point, and multi-day accuracy commands
are historical evidence only and have been removed from the active workflow.
Their records remain in Git history and the ignored, local pre-generalization
archive. Broad training stays paused until the current isolated
temporal/long-horizon rung preserves the minimal gates.

The general evaluator still supports explicit disjoint manifests for a
compatible `OnlineWorldModel` checkpoint. Such a smoke or diagnostic must not
be presented as promotion of the standalone minimal estimator.

For the deterministic convergence/debug run:

```bash
python train.py --config configs/tiny_overfit.yaml --run-name tiny-debug
```

## Live monitoring

In a second terminal, one command follows the newest verified-active training
or evaluation run (including timestamped evaluations nested below a training
run):

```bash
conda activate orpheus
python monitor.py
```

The monitor is read-only. It does not import the model, load checkpoints, use
the network, or consume accelerator memory. It polls every 60 seconds, prints
only when an artifact or process state changes, and emits an unchanged
heartbeat every 10 polls. The dashboard includes phase, step/target and ETA,
robust recent-loss trend, fixed-validation decision and horizon RMSEs,
validation/evaluation episode progress, device placement, checkpoint age, and
hard failure/staleness/non-finite signals. `evaluate.py` now writes its atomic
`evaluation_progress.json` automatically from pre-load initialization through
completed, failed, or interrupted state; `--progress` additionally echoes the
same events to stdout.

Pin a particular run, relax the interval further, take a one-shot snapshot, or
emit JSON with:

```bash
python monitor.py --run runs/<timestamped-run> --interval 120
python monitor.py --once
python monitor.py --once --json
```

Generated training, evaluation, and demo directory basenames begin with a UTC
`YYYYMMDD-HHMMSS-` timestamp, so ordinary filename sorting puts the newest
artifact last. Explicit `--run-name` and `--output` values are human-readable
labels; the command's JSON result contains the actual path. Historical local
runs and demos were archived and removed before the generalization program.

## Documentation

- [Authoritative specification](PROJECT_SPEC.md)
- [Current status](project/STATUS.md)
- [Architecture](project/ARCHITECTURE.md)
- [Generalization ladder](project/GENERALIZATION_LADDER.md)
- [Data contracts](project/DATA_CONTRACTS.md)
- [Predictive abstractions](project/PREDICTIVE_ABSTRACTIONS.md)
- [Getting started](docs/getting_started.md)
- [Known work](project/TASKS.md)

Licensed under Apache-2.0.
