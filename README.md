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
consumed. This accepts only the frozen one-slot free-motion bridge; it is
merged to `main` at `3eed0b7`, not evidence for broader scene, modality, or
learned-capacity convergence.

Specification 1.56 now also accepts the exactly-two-visible RGB-D rung on
clean commit `3b781e6`. Development and exactly-once selector -> confirmation
-> final all passed; final is consumed and must not be rerun. This remains a
parameter-free, differentiable, fixed-camera, non-contact result—not evidence
for partial visibility, contact, variable count, learned capacity, or general
world-model convergence.

Specification 1.57 now also accepts that same exactly-two-visible family under
one known calibrated orbital camera. Development and the exactly-once
selector -> confirmation -> final ladder all passed on committed source
`c15afd6`; final is consumed and must not be rerun. This remains a
parameter-free result for one certified orbit—not general moving-camera,
unknown-pose, learned-pose, occlusion, contact, variable-count, or learned-
capacity evidence.

Specification 1.58 now freezes the next genuinely new capability: distinct
per-object linear drag inferred from the same complete sixteen-frame RGB-D
history under the accepted known orbit. The source adds no learned parameter
or optimizer update. It uses a differentiable adaptive analytic fit, atomically
writes fit-owned position/velocity/log-drag evidence on the anchor frame, and
stores exactly three calibrated float32 uncertainty-scale buffers. This is a
pre-access source freeze, not a result: all four seedless 64-ordinal splits are
unopened and the canonical run directory does not exist.

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
`1130 passed, 16 skipped in 414.82 s`; the integrated bridge tree later passed
the accepted one-object complete repository gate recorded below. Its first
integrated-tree boundary was `1207 passed, 16 skipped in 431.10 s`.

The former grounded, attention, change-point, and multi-day accuracy commands
are historical evidence only and have been removed from the active workflow.
Their records remain in Git history and the ignored, local pre-generalization
archive. Broad training remains paused. The accepted one-object and fixed-
camera two-object bridges are merged. The reviewed specification-1.57 known-
calibrated orbital-camera net tree is also merged and published to GitHub
`main` through acceptance commit
`00a712d640cdb828f24a194817443daa57e6df65`; final remains consumed. Any next
rung still requires a genuinely new capability frozen under its own pre-access
protocol.

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
all unopened. That accepted one-object bridge boundary passed the focused
source/config/protocol gate at `421 passed in 62.72 s` and the complete
repository gate at
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
general world model. At that historical one-slot boundary, no multi-object,
association/occlusion, contact/event, task-success, camera-motion, or learned-
capacity result was claimed; the subsequently qualified exactly-two-visible
fixed-camera rung is recorded below. Each later rung must predeclare its one
changed capability and all applicable gradient, accuracy, identity/event,
memory, and rollout-throughput gates. No accepted bridge threshold or
consumed-final result may be retuned.

### Qualified two-visible-object RGB-D rung — final consumed once

Architecture attempt 2 preserves the accepted one-object behavior and adds
exactly two chromatically distinct spheres that remain fully visible,
image-separated, and non-contacting. Differentiable chromatic-plus-spatial
two-slot RGB-D geometry owns unordered metric measurement; hard Hungarian is
only the discrete stable-identity control. Direct metric position has one
owner, a uniform differentiable WLS fit uses all sixteen frames for velocity,
and analytic dynamics answer position and velocity at
`0.1/0.25/0.5/1/2 s`. The rung owns zero learned parameters and optimizer
updates.

Frozen config/protocol SHA-256 values are
`84e6f44b818bb9323a774bdba9492ef056e2a2747b93517fa38497ba83218bba` /
`42b9dca23fed303d5cee4641c8d8753977a872fc90d0b1086658d7f12b823ea0`.
The empty model-state SHA-256 is
`4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.
The accepted source is commit/origin
`3b781e653a0287b2aa926e7c0b969e9197d48e42`, with runtime/worktree fingerprints
`810b237082ae99735527985c544dc28834b806489c555b464191c3b3e62520e7` /
`fdbbe6fe3a85b491578d4cda2dc880f1dc21726f3469d3717976662796f12f23`.
Development report/checkpoint/ledger SHA-256 values are
`dfed30c29b7dc07adcfd01a233e3de3a42f32d8b333a1b8d696dae144af98f4b`,
`e59e0d4b0f8f747b38fb0699cbf9f1491f0ca81dbffdcde5bcb56b04002c6bed`, and
`9a49f574f6b7bdc0211d01d1ad4e5591d9155e7191f0a04b5e48e864ca56e579`;
qualification-report/ledger SHA-256 values are
`085f5206ac02f01fa5d7f5bc0cad055f75d401447cc090c800431fabf89ef1ef` and
`3cc22f65f809ad4afb08bf26a1984157beb1076acb5fc910d8edc3f5df0035af`.

Every split reports `396` finite gated metrics and no failures. Current
position/current velocity/two-second position RMSE is
`1.9029872e-5/3.1932373e-5/7.4638663e-5` on development and
`1.7838631e-5/3.1881889e-5/7.1961138e-5` on final, in metres and metres per
second as applicable. Final two-second velocity RMSE is `2.8845629e-5 m/s`.
Identity coverage is one with zero switches, mismatches, or ambiguities.
Visibility is one and event count is zero.

Current position reaches anchor frame 15 only (`1/16`, exact zero non-anchor
gradient); current velocity and every rollout position/velocity reach `16/16`
RGB and depth frames. Four distinct B4 scenes per split have exact zero
cross-scene coupling. Final perception/five-query rollout is
`0.352510 s`/`0.00359524 s`, runtime state is `28,512` bytes, and maximum RSS
is `579,817,472` bytes. Learned, buffer/model-state, optimizer/scheduler/RNG,
and update counts remain zero. Exact source gates are `43 passed` focused,
`281 passed in 15.61 s` combined, and
`1275 passed, 16 skipped in 447.29 s` complete; independent final audit passes.

The ledgers completed in development then selector -> confirmation -> final
order. Final is consumed: do not rerun qualification or tune against it. Audit
did not reopen raw protected episodes. Owner-writable evidence is SHA-bound and
tamper-evident, not OS-enforced WORM storage. Live history remains outside
ordinary checkpoints. The reviewed tree was fast-forwarded to GitHub `main`
through `1e951520e5a2bf06c1932f64b8334e552247de82`. At that acceptance boundary,
partial visibility and missed-observation recovery had been proposed as the
next rung. Specification 1.57 instead inserted and has now accepted one smaller
known-camera-motion rung first. The failed partial-visibility family remains
closed and must not be retried. Misses/recovery, occlusion, variable count,
contact, unknown camera pose, variable physics, uncertainty calibration,
tasks, added modalities, and learned capacity all remain unqualified.

### Qualified known-calibrated orbital-camera RGB-D rung — final consumed once

At the historical pre-access boundary, specification 1.57 froze exactly two
fixed-radius spheres fully visible, image-separated, and non-contacting, with
gravity `0`, drag `0.05`, complete RGB-D, and zero learned parameters, learned
state, pose estimation, or optimizer updates. Sixteen physical primitives
cross four orbital phases and two directions to form 128 joint scenes. The
public runtime receives exact time-aligned `world_from_camera`; camera pose is
not inferred.

Frozen config, harness, runner, test, protocol, and certificate SHA-256 values
are respectively
`a9c348ea54b168ec78780d59d3b3eb066344d3a7551464b9aad1e5b9ac6d6cbd`,
`02e75b325bdf7bad310f8973a786a396b8762104261702b299a9f8103748e569`,
`11bee2e4d05f83caaf9dbed6ca2a54d4c11b7c70e4bf8e1747b261b8518ef192`,
`d08c7bb4a1ba998a51dc2f0ddb1946596a5a299ed236cdf6a91b5711e2d0a1af`,
`7146befc869ea5f975177dd1c2da4691026439e1d36d84415aa23f696e61ef65`,
and `7832ddb49081292d0f50a5eb63edb38fefb49d136d7e1757c73d9c658e42a36f`.
Simulator semantics remain `sphere_world_v7`.

The exact per-split gate schema has 685 finite floats. It includes public
physics and resource gates, fixed-output VJPs to RGB, depth, and
`world_from_camera`, and a negative control that reuses the stale frame-0
camera transform on history frames `1--15`. The correct calibrated path must
beat that control for current position, current velocity, and two-second
position. Certificate-bound visibility, separation, non-contact, camera-motion,
and calibration extrema cover all 128 joint scenes before seed access.

| split | seeds | manifest SHA-256 |
| --- | --- | --- |
| development | `61000000--61000031` | `eb558805c2974302c33abef4531e142bb60e8f20045d8530330838223a6899a0` |
| selector | `62000000--62000023` | `c97fff97459ee9962b972cb7905887c2b2ed6eb5a1837d908f1512ce77e6d97f` |
| confirmation | `63000000--63000023` | `b47f03633732fc2986939e71007a0a79b12db2b42f0b5261b4ebd2d0a304f544` |
| final | `64000000--64000047` | `82927d192b53f2e4af11491f53039c145acfd8e0401a3e2b0b1e974591ee4174` |

At that historical boundary all four namespaces were unopened and no artifact
existed. The exact freeze was then committed as
`c15afd6d57963b24bb98c5171462ff927e7c72fd`, with local upstream `0/0`,
worktree fingerprint
`0a5acfc54a5af482643b0c1037cf566a700e6122d2e6b51f7f4ad713ff652d2e`,
and runtime fingerprint
`bec3ca667fa464a3bbe82a83c14ffa924920ca367f14b6d9036ce52af041b83b`.
Development passed and external review bound the exact checkpoint, report, and
development-ledger hashes before protected access.

Development report/ledger/checkpoint SHA-256 values are
`56d7e32c461d9b5e3fbca5e2e11e015662cd08c3d60dfb4807e75cbcb7f8e37b`,
`3f9d5c9cf88ae7e40517337799e270d02493e99ed58eaec24884e276dcec5ddf`,
and `c473bb6d5f453c786c681509350d66364e1f1c61a2656a7c35354ab806da1a25`;
qualification report/ledger SHA-256 values are
`6daf2dea453db7c3a32b7950c8f31201ccc3fc32b9da1b14d8cc97dbd46ee0ad`
and `2aeb1c0194332004350c98628210d42724e31ece16614a210e2a84d6640b2719`.
The five artifacts occupy `88,743`/`1,544`/`78,573`/`202,540`/`2,293` bytes
in that order and form the exact single-link, no-extra terminal inventory.

| split | result SHA-256 | current position / velocity RMSE |
| --- | --- | --- |
| development | `555871b24bfb764712d8dcae8473d5a9ad4c0ec6e9f02ffd42b2063af3cd7bc2` | `1.5474954e-5 m` / `1.9066727e-5 m/s` |
| selector | `fcfd1b39393a8e41d0b112244b7e5ca4fe3c0b2e4e63b4cd729659781198e9d6` | `1.5932185e-5 m` / `1.8599896e-5 m/s` |
| confirmation | `c3d644786d308a03d619eaf2a4d954bc216b1daf8655a9217f09e372ab27cd0b` | `1.6963295e-5 m` / `1.9596576e-5 m/s` |
| final | `b8ae823e961a981360717be273fe10d1ff5f9ce3bcbd6c396ba78fd5fdf0a4bf` | `1.7444936e-5 m` / `1.8386924e-5 m/s` |

Every split has `685/685` finite float metrics, all `686` constraints pass,
and `failures: []`. Final stale-calibration current-position/current-velocity/
two-second-position RMSE is
`0.053232131 m`/`0.069683790 m/s`/`0.185857911 m`. All eight camera strata,
identity/history, public physics, and the certificate pass. Final minimum total
and temporal-frame VJP L1 are `2.502e-5` and `5.819e-8`; cross-scene,
non-anchor, and homogeneous-row gradients are exactly zero.

Perception/rollout spans `0.3452--0.3944 s`/`0.003469--0.003704 s`, persistent
state is `28,512` bytes, RSS remains below `578 MB`, and initial/final state
has the same empty SHA-256
`4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.
The full source gate is
`1302 passed, 16 skipped in 594.59 s (0:09:54)` and two independent
qualification audits pass. The ledger completed selector -> confirmation ->
final and stopped after `final_test`; final is consumed.

The terminal snapshot is not an append-only signed history, and exactly-once
is inferred from final state plus committed control flow. Ignored artifacts
are unsigned/unlogged and filesystem-deletable, external hashes carry no
reviewer identity, upstream equality used a local tracking ref without a fresh
network fetch, and raw protected episodes were not rederived or audited.
Acceptance is therefore narrow and tamper-evident, not a general moving-camera
or cryptographic-transparency claim.

### Frozen identifiable per-object drag rung — no governed result yet

Specification 1.58 retains exactly two fully visible, separated,
non-contacting fixed-radius spheres, complete RGB-D, zero gravity, and the
known calibrated orbit. It changes only drag: each persistent object has a
different unknown constant in `0.045--0.325 /s`, inferred from all sixteen
public observations. Runtime inputs remain RGB, metric depth, calibration,
intrinsics, image metadata, and timestamps; no drag label, simulator state,
instance map, or object identity is supplied.

The seedless governed family has development, selector, confirmation, and
final splits of 64 exact ordinals each. Four rational physical primitives are
crossed with two drag-slot-swapped counterfactual roles and eight camera
strata. Odd-sixteenth camera phases and inward-shifted rational drag grids make
all 32 governed drag values and all protected joint traces disjoint. This is a
designed interpolation family, not iid or distribution-free evidence.

The estimator owns zero learned parameters and zero optimizer updates. A
differentiable 257-node log-drag profile plus same-sized local refinement emits
fit-owned anchor position, velocity, log drag, and raw diagonal variances.
Development alone may calibrate three uncertainty-scale buffers from cached
scale-one evidence using the one-indexed rank-59 scene-max statistic. The
smallest deployed float32 triple meeting the cached arithmetic is installed
atomically once, with no episode replay. The checkpoint therefore contains
exactly three CPU float32 scalars, or 12 tensor bytes.

The fixed profile, harness, runner, tests, protocol, and independent scene
certificate SHA-256 values are respectively
`a22f364601b8f87cdec3fd6bff7d757f134867bf66d9fa176c1f2d881a700c45`,
`1a48832fb898b552a6f19cd2cadaa77634d02585db873145cb466b1111f01f56`,
`6838a2128dc07d65439811b8a789bcc89935ba7bb0eb5ed997629ab9794548db`,
`30a01c8df07a82923b96eb92911881a7190f2ad875c0073ade40565a7ce87335`,
`d4abaf22e775afc6f807b268f08aa68ae44210a40192b1b05653957720f48c70`,
and
`588c8fe2e2baa38dcb097a012b5ec6517b3ce9733a7c8d068e71c98a1c5f5f9e`.
The 263-float gate-schema and artifact-schema bundle SHA-256 values are
`cb3a65efaa3cb06eb5eaa5bad0f556c41578d8250f805a4e3c314cfa0d22bb1d`
and
`d2022f17aa805a9ff6b8ae65ce981f3d0c4f1fdbfbb53e08f69939a80d62eecc`.

The independent formal certificate covers all 256 scenes and 14,336 frames
without invoking governed public physics, rendering, perception, or runtime.
Minimum full-mask support is 21 pixels, silhouette gap is `5.11001396 px`,
world surface gap is `1.37303865 m`, and fit excitation is `0.02090907 m`;
visibility is one and every contact/event count is zero. A separate consumed
cardinal family supplied threshold-design and API-equivalence checks only.

Source verification passes the `68`-test seed-free harness, the guarded scene
certificate, and the complete pre-documentation repository gate at
`1490 passed, 16 skipped in 781.56 s (0:13:01)`, plus Ruff, formatting,
compilation, exact critical-source rehashing, diff checks, and independent
audits. Development is forbidden until this exact specification-1.58 tree is
clean, committed, published, and equal to its configured upstream.

The canonical artifact root is
`runs/rgbd_two_visible_orbital_camera_identifiable_drag_v1`. It is absent.
Development must consume ordinals `0--63` once, cache only sufficient evidence,
derive and install the three scales once, and produce a reviewable report,
ledger, and restricted checkpoint. Only an external review of all three exact
SHA-256 values may authorize the ledgered selector -> confirmation -> final
sequence. Final cannot be rerun or tuned against.

Direct-anchor diagonal uncertainty supports the one public multi-horizon
request. It does not claim covariance-complete sequential re-anchoring,
universal calibration, contacts, occlusion, variable count, pose inference,
learned capacity, planning, or general world-model convergence. Until governed
evidence passes, specification 1.57 remains the highest accepted result.

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
