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

The former specification-1.46 grounded and earlier multi-day accuracy commands
are intentionally no longer presented as launch instructions. They are
historical evidence only; broad training stays paused until the next isolated
complexity rung preserves the minimal gates.

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
