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
core plus a frozen, reviewed parameter-free temporal protocol. The first 41m
development run passed its frozen metrics, but a representation-sensitive
post-run comparator required a source repair and invalidated those artifacts
for qualification. A fresh v2 development rerun and exactly one ledgered
42m -> 43m -> 44m qualification now pass on source `df0235a9`; final is
consumed and must not be rerun. This is not a third monocular-taper attempt or
a claim beyond the standalone rung.

The composite one-slot RGB-D `OnlineWorldModel` bridge is now qualified under
specification 1.55 on canonical clean source `526b5123`. The first development
artifact from `ebda5a8` passed its numerical gates but was rejected because its
persisted JSON protocol was compared to tuple-valued in-memory data with raw
Python equality; no protected split was opened. A fresh canonical development
run and independent audit then authorized exactly one ledgered
46m -> 47m -> 48m qualification. All protected splits passed and final is
consumed. This accepts only the frozen one-slot free-motion bridge; it is ready
to merge, not evidence for broader scene, modality, or learned-capacity
convergence.

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

Specification 1.54 froze the standalone qualification before its first
episode access. The exact config SHA-256 is
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

The first development run used clean commit `8e68035` and seeds
`41000000--41000023`. All `82` reported scalar metrics were finite and all
frozen gates passed, with current
position/velocity RMSE `0.00279934 m`/`0.00207092 m/s`, horizon position RMSE
`0.00286018/0.00297530/0.00322108/0.00385564/0.00539692 m`, perception time
`0.235061 s`, and rollout time `0.00391524 s`. Its report, checkpoint, and
manifest SHA-256 digests are respectively
`9cbea9f25181769ee5b6a87b097e738a29cdb9b386c8018b3044f07d58aa03e2`,
`6acd88edd203cdebb2b0820bad388e06a4c610ea1c659ff9f8ea6d701ad28059`, and
`b92f1e3f12475986ebd2971ad2de70187432c0941caa0335ea53e82abc3c1d01`.

That is historical conditional development evidence only. Audit found a raw
Python tuple/list equality check after JSON roundtrip; the repaired verifier
uses a canonical JSON SHA comparison plus regression coverage. Because that
repair changes source, the old report/checkpoint pair must not qualify
protected access. That text records the v1 invalidation boundary; it is not the
current qualification evidence.

#### Qualified v2 development and one-shot protected result

The qualified runtime is commit
`df0235a92a81d3c1d2ba4e69e64d639562e3dfe8` with the unchanged config/protocol
hashes above. Fresh v2 development report SHA-256
`4cf1657ee95645c8c647433a8be660520e9cdc1a5e6ac106d85bd24547b4e740` and
checkpoint SHA-256
`fd663e5fa52dded8156a3178070966e3458d93a7b5a49dd5dcb2cc0d6278514e`
passed audit. Its scientific errors match v1; perception/rollout time is
`0.233866867 s`/`0.003934262 s`, with `545804288` bytes maximum RSS.

Protected selector, confirmation, and final then passed exactly once under the
durable access ledger. Qualification-report bytes, canonical summary, and
ledger SHA-256 are respectively
`7e4cface087620f058ade4cc83ac5fd197685ba26c8f0afb5089d8f7e646fe0d`,
`7e9954ae34ce55b6923765de0c084d5075f238bd012554eeb44049a0db161658`, and
`9fc139291dfb34b10125321d06fdf06ab68ed65df32f62c273a95e5ca7aa7b8b`.
The ledger records selector -> confirmation -> final before each access; final
is consumed and must not be rerun.

| split | current position / velocity | 0.1 / 0.25 / 0.5 / 1 / 2 s position RMSE |
| --- | --- | --- |
| development v2 | 2.799 mm / 2.071 mm/s | 2.860 / 2.975 / 3.221 / 3.856 / 5.397 mm |
| selector | 3.073 mm / 1.991 mm/s | 3.159 / 3.306 / 3.589 / 4.254 / 5.774 mm |
| confirmation | 3.078 mm / 1.644 mm/s | 3.078 / 3.094 / 3.162 / 3.431 / 4.328 mm |
| final | 2.905 mm / 2.226 mm/s | 2.954 / 3.056 / 3.290 / 3.934 / 5.560 mm |

Every split has `82` finite metrics; all `103` gate comparisons pass with zero
failures, optimizer updates, or learned state. Final two-second error is
`18.5%` of its gate and only `3%` above development. Rollout is
`3.80--3.95 ms`, perception is `232--237 ms`, maximum RSS is `554 MB`, both
modalities retain VJP norms `0.1156--3.6564`, semigroup error stays below
`2.384e-7 m`/`3.725e-9 m/s`, and every trivial/RGB-only ablation gate passes.

OLS covariance remains diagnostic, not calibrated uncertainty. Artifacts are
SHA-bound and tamper-evident but owner-writable. At that boundary the next rung
was the composite batched `rgbd` public `OnlineWorldModel` bridge; its later
qualification is recorded below. The standalone source gate was
`1130 passed, 16 skipped in 414.82 s`; the integrated bridge tree now passes
the newer complete repository gate recorded below. Its first integrated-tree
boundary was `1207 passed, 16 skipped in 431.10 s`.

The former grounded, attention, change-point, and multi-day accuracy commands
are historical evidence only and have been removed from the active workflow.
Their records remain in Git history and the ignored, local pre-generalization
archive. Broad training stays paused while the accepted bridge evidence is
merged and the next bounded rung is predeclared without weakening the
standalone or public-integration gates.

### Qualified public RGB-D online bridge — final consumed once

The bridge uses one composite batched `rgbd` packet with RGB `[B,3,H,W]`,
depth `[B,1,H,W]`, batched calibration, explicit image size, and a
modality-qualified stream key. Raw metric positions have one direct filter
owner; a persistent-ID-aligned uniform sixteen-frame WLS history emits
velocity-only evidence; and the parameter-free analytic dynamics answers
`0.1/0.25/0.5/1/2 s` queries. Missing depth never falls back to RGB.

The frozen config SHA-256 is
`c40b3438c7fd60646d356db3fe54050039912ace288d9db89620b626106993a3`, and
the seed-free canonical protocol SHA-256 is
`e536b0d0b721042bff55501faf3445456219fcc987334b6ec1e892688ea560b2`.
At the source-freeze boundary, the development and protected namespaces were
all unopened. The current focused source/config/protocol gate is
`421 passed in 62.72 s`; the complete repository gate is
`1209 passed, 16 skipped in 434.37 s`. The bridge owns zero
parameters, state-dict entries, or optimizer state; complete batch-four
persistent runtime tensor storage is `25,364` bytes against a `32,768`-byte
gate.

RGB-D evaluation and demos now use the real composite packet and truthful
`observation_modality: rgbd`/`rgb_only: false` metadata. The evaluator labels
its fifteen-frame warmup; demo aggregate errors remain pooled across warmup
and post-warmup frames. The legacy RGB workflow is preserved when the bridge
is disabled. Per-frame VJPs cover both RGB and depth for current velocity and
every horizon.

For an already-active persistent object, a well-formed frame with missing
depth, nonfinite or otherwise invalid depth in the sampled measurement support,
or no foreground appends an invalid causal row. In the frozen sixteenth-frame
full-window ablation, diagnostics are `sample_count: 16` and `valid_count: 15`.
The invalid measurement emits no valid/admissible temporal fit or direct
velocity evidence, correction, or birth. A finite diagnostic fit may exist
with `fit_valid: false` but is not admissible evidence. Before birth, the same
invalid frame advances runtime time but creates no object-history row and no
birth. Malformed packets, nonfinite RGB/calibration, unsupported low precision,
stale/duplicate/unknown streams, and invalid prepared propagation reject
atomically without consuming runtime or propagation state.
Ordinary checkpoints do not serialize live temporal histories and caches, so
exact mid-history stream resume is unsupported; replay the observations to
rebuild that state.

Development seeds are `45000000--45000023`; selector, confirmation, and
one-shot final are `46000000--46000015`, `47000000--47000015`, and
`48000000--48000031`.

The first development execution on source
`ebda5a8bfa7b1131b827202f575351d116c78d01` passed every numerical gate, but
its persisted protocol validator rejected JSON lists against the tuple-valued
in-memory protocol despite equal canonical content and protocol SHA. Its
report/checkpoint SHA-256 values are
`2104ee87bcabdbd5312b4026a33e44e1de7d197e50215ec7f0bf0e0bb56992e3` and
`38f4b2ef5addb98bb966360213d3bb36b43da606367fc60cd75d2ec487f1b866`.
Those artifacts are diagnostic only; selector, confirmation, and final were
not opened from them.

The accepted clean source is
`526b5123e6385c575a5777936272330d28972b93`, with runtime-source and worktree
fingerprints
`1eeaa176ad9be8886976910fe53028fb6de498adda73a2d20170f206b6134b40` and
`90d0624a119e118e76b58061f7e5582dffc906f47d85cc4dde997b2f765bb07a`.
The empty model-state SHA-256 is
`4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.
Canonical development report/checkpoint/manifest SHA-256 values are
`dce6f920da85fbf696b7ae8a7a91d9cbf7d9084176e51ad7c319f92a6efe4966`,
`48249f1a5a0467b1da8c7bdb5ad9e909f8c502631ec2fbad832cb490a00c3099`, and
`069eb3331543727c911a07cc9a1bb352f6185ac8ceac7fafca502c9d7fab6d80`.
The report and checkpoint are `22,346` and `46,596` bytes.
Independent audit passed before protected access.

| split | current position / velocity RMSE | 2 s position / velocity RMSE | growth slope | perception / rollout |
| --- | --- | --- | --- | --- |
| development | 3.068470 mm / 2.191966 mm/s | 5.609913 mm / 1.983371 mm/s | 1.270721 mm/s | 0.415134 s / 3.575380 ms |
| selector | 3.177543 mm / 2.313401 mm/s | 5.881384 mm / 2.093251 mm/s | 1.351921 mm/s | 0.422070 s / 3.569962 ms |
| confirmation | 5.681172 mm / 1.658775 mm/s | 6.188252 mm / 1.500921 mm/s | 0.253540 mm/s | 0.414407 s / 3.537710 ms |
| final | 2.996787 mm / 2.221047 mm/s | 5.433965 mm / 2.009688 mm/s | 1.218589 mm/s | 0.417436 s / 3.566628 ms |

Protected selector, confirmation, and final passed exactly once in order. Their
manifest/result SHA-256 pairs are respectively
`2159b044e089774b3b7df95509ac2cded19528de6ff133ae1b158a354ed7fbb9` /
`9ac6b7cc1b97da9961345fdcf5488ddec3ac6a0186215699a55a66acfbb983cb`,
`2cad3224740b4d73871ff1d1e60795d45dc149ad03d197513eddf514cb9946bf` /
`1a3996914d59f840b2645e4b886f1027b830fa6f81c5763eb1735f25149aa9bc`,
and
`3c5c904203ddd46ea790322e446466b2c58e603015456f239715aa07135011a3` /
`40d39accec8c2c6efa97f06a2f2748c580a5666b54c7dac4df36e3d7dc718bd1`.
Qualification-report and ledger SHA-256 values are
`7fd1829f663606910ac81990e4b633c63b1460dbc31dd24c71eedbd91b422908`
and `cf6a10dd672aafbdd91c92871ae349fef0c549d865cc6532e6c42f7d9be14e32`.
They are `47,353` and `1,626` bytes. Final is consumed and must not be rerun.

Every split has `175/175` finite scalars, zero failures, full `16/16`
per-target/per-modality history VJP support, one position owner, no identity
change, and zero learned or optimizer state. Final semigroup error is
`2.384186e-7 m`/`1.862645e-9 m/s`; maximum RSS is `708,853,760` bytes and
runtime tensor state remains `25,364` bytes. Audit recomputed gates from
finite persisted evidence but did not regenerate protected episodes or rerun
latency/RSS. The ledger is durable atomically replaced state, not an
append-only timestamped transition log; access-before-materialization is
supported by clean committed control flow plus the final ledger and result
receipts. Artifacts are SHA-bound and fresh-path protected but remain
owner-writable, so they are tamper-evident rather than OS write-once.

This qualifies the declared one-slot, one-sphere, fixed-camera, fixed-parameter
free-motion bridge only. It does not establish uncertainty calibration,
multi-object/contact behavior, camera-motion handling, learned capacity, or a
general world model. After merge, the next bounded predeclared rung is one
changed capability at a time. Before new data it must freeze moving-camera,
multi-object, association/occlusion, contact/event, task-success, capacity,
gradient, accuracy, memory, and rollout-throughput gates; no accepted bridge
threshold or consumed-final result may be retuned.

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
