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

The first temporal extension is also closed as a terminal development failure.
Its trainable reliability taper discarded most of the 16-frame history and
could not identify velocity accurately enough for long rollouts. Protected
selector, confirmation, and final manifests were never opened. The next
generalization rung now has a passing parameter-free RGB-D metric measurement
core plus a frozen, reviewed parameter-free temporal protocol. Every fresh
41m/42m/43m/44m manifest remains unopened; this is not a third monocular-taper
attempt, development evidence, or a convergence claim.

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

### Terminal temporal experiment and next rung

The frozen attempt-2 monocular temporal experiment stopped at development on
clean source commit `8889818619121351d342490786331e854364532c`. Its audit
missed 10 accuracy/baseline gates: current position/velocity RMSE was
`0.016128 m`/`0.070461 m/s`, and position RMSE at
`0.1/0.25/0.5/1.0/2.0 s` was
`0.022907/0.033205/0.050360/0.084191/0.149501 m`. Physics, oracle, gradients,
semigroup, resource, and geometry diagnostics passed, so the failure is in
temporal observation/weighting rather than the rollout equations.

The learned taper reached `10.0338 /s`: the oldest-to-anchor weight ratio was
only `0.000534`, with `77.63%` of mass on the last three frames and `91.71%`
on the last five. A development-only confidence-weight diagnostic improved
current position/velocity to `0.00842 m`/`0.01660 m/s` and the five horizon
errors to `0.01000/0.01239/0.01638/0.02430/0.03963 m`, but future velocity
remained `0.01652 -> 0.01502 m/s` against the `0.01 m/s` gate and the early
zero-velocity-baseline ratios still failed. Attempt 2 of 2 is therefore
exhausted. Do not rerun it or access its protected manifests.

The immutable failed development report is retained under the ignored local
archive with SHA-256
`be488d045e259c0804a2a2b24215fa4eb3025d69f6113d8dbefe21d72f827554`.
The removed config, runner, estimator, and tests are no longer supported
entry points.

### Seed-free RGB-D metric core

Simulator v7 now exposes observable metric surface depth as
`[T, 1, H, W]`, with zero meaning no return. Exact ray--sphere intersection
uses the same nearest surface winner for depth, instance, visibility, and RGB.
The parameter-free measurement consumes an RGB-derived differentiable
subpixel centre, bilinearly samples depth, applies perspective radius
correction using the known radius and canonical camera, and never reads labels,
state, instance maps, or object IDs.

A seed-free 18-case public-renderer grid passes with maximum/RMSE position
error `0.00613210 m`/`0.00336217 m` and maximum/RMSE centre error
`0.0272064 px`/`0.00802947 px`. Finite gradient norms to centre, RGB, and depth
are `0.673917`, `0.0718314`, and `6.92869`. Invalid finite/extreme rows fail
closed with zero finite gradients, and float16/bfloat16 are rejected. The
focused result is `29 passed`, with independent review passing.

This result used no episode seed namespace or protected data. It qualifies a
single-frame RGB-D metric observation primitive only. The post-deletion/core
full suite passes `1091` tests with `16` expected inactive-device skips.

### Frozen parameter-free RGB-D temporal protocol

Specification 1.54 freezes the next standalone qualification without opening
any episode manifest. The exact config SHA-256 is
`5667cdb3603682b8d80a3e42793d25e36989269df1afacfa9b1028f2451101e9`; the
canonical protocol payload SHA-256, computed before inserting its
self-reporting digest field, is
`4e334e9d7942ea3f2416c0a9f5ca8e327d1d0a1e9131074f20c051ebd3163ad7`.
Fresh manifests are development `41000000--41000023`, selector
`42000000--42000023`, confirmation `43000000--43000023`, and one-shot final
`44000000--44000047`.

The estimator measures all 16 RGB-D frames independently, applies uniform
weights in the differentiable exact free-motion WLS fit, and queries existing
analytic dynamics at `0.1/0.25/0.5/1.0/2.0 s`. It has zero parameters,
buffers, optimizer state, and optimizer updates. The frozen gates cover
current/per-axis/horizon state, trivial baselines, semigroup consistency,
fixed-output VJPs to both RGB and depth, RGB-only and missing-depth ablations,
diagnostic OLS covariance, memory, and separated observation/rollout latency.
Protected access is durably ordered selector -> confirmation -> final after a
separately reviewed passing development artifact.

The combined seed-free protocol gate is `103 passed`, independent review
passes, and the final repository gate is `1129 passed, 16 skipped in 428.18 s`.
Development and every protected split remain unopened, so no temporal accuracy
or long-horizon convergence is claimed. The next implementation rung after
qualification is a composite batched `rgbd` packet and one-slot public online
bridge; capacity and scene complexity remain frozen until then.

The former grounded, attention, change-point, and multi-day accuracy commands
are historical evidence only and have been removed from the active workflow.
Their records remain in Git history and the ignored, local pre-generalization
archive. Broad training stays paused while the observable-depth/RGB-D rung is
specified and qualified without weakening the accepted minimal gates.

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
