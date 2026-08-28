# Training and qualification

## Active policy

Orpheus scales through deterministic, fail-fast capability rungs. Each rung
changes one independently measurable source of complexity, binds disjoint
train/selector/confirmation/final manifests before access, and preserves every
accepted lower-rung gate. A failed family stops; thresholds and final sets are
not repeatedly tuned.

Runtime inputs must be observable modalities plus calibration, timestamps, and
declared priors. Simulator state may label losses and metrics but may not enter
a claimed RGB/RGB-D forward path. Continuous state, parameter, and rollout
learning uses ordinary PyTorch autograd. Analytic tensor equations provide
inductive bias; learned residual capacity is added only after an identifiable
structured baseline leaves measured error.

## Qualified base

The supported standalone developer command is:

```bash
conda run -n orpheus python scripts/run_minimal_toy_ladder.py \
  --config configs/minimal_differentiable_toy_cpu.yaml \
  --report runs/minimal_differentiable_toy_v2/report.json \
  --checkpoint runs/minimal_differentiable_toy_v2/model.pt
```

Do not rerun its published final manifest merely to regenerate provenance.
Future runs require fresh output paths and produce atomic, versioned,
weights-only project checkpoints whose SHA-256 is bound in the report.

## Closed monocular temporal rung

Do not launch the former temporal runner or qualification workflow. Clean
source commit `8889818619121351d342490786331e854364532c` completed the frozen
32-update development phase and failed its 16-seed audit. The immutable report
is retained at ignored local path
`.archive/20260826-pre-generalization/temporal-free-motion-attempt-2/development_report.json`
with SHA-256
`be488d045e259c0804a2a2b24215fa4eb3025d69f6113d8dbefe21d72f827554`.
It records `passed: false`, `review_ready: false`,
`stopped_after: development_audit`, and
`protected_data_materialized: false`.

The trained audit failed 10 gates. Current position/velocity RMSE was
`0.016128 m`/`0.070461 m/s`, and horizon position RMSE at
`0.1/0.25/0.5/1.0/2.0 s` was
`0.022907/0.033205/0.050360/0.084191/0.149501 m`. The fit assigned only
`0.000534` as much weight to the oldest frame as the anchor and concentrated
`77.63%`/`91.71%` of its mass on the last three/five frames through a learned
`10.0338 /s` taper. Thus the trainable weighting collapsed temporal extent
even though the exact free-motion oracle, simulator comparison, semigroup,
gradient, geometry, memory, and latency checks were correct.

A confidence-only development diagnostic produced current position/velocity
RMSE `0.00842 m`/`0.01660 m/s` and horizon position RMSE
`0.01000/0.01239/0.01638/0.02430/0.03963 m`. That does not qualify the family:
future velocity remained `0.01652 -> 0.01502 m/s` against the `0.01 m/s` gate,
and the early zero-velocity-baseline ratios still exceeded `0.5`. This was the
terminal second of two monocular architecture attempts. Do not change its
thresholds, run a third variant, or materialize selector, confirmation, or
final seeds. The config, runner, estimator, and dedicated tests were removed
from the active source tree after archiving the report.

## Qualified RGB-D temporal rung — one-shot final consumed

The seed-free single-frame foundation for the next rung now passes. Simulator
v7 provides observable `[T, 1, H, W]` metric surface depth from the same exact
nearest ray--sphere winner used by RGB, visibility, and instance output. A
parameter-free measurement uses the RGB-derived subpixel centre,
differentiable bilinear depth, known radius, canonical camera, and perspective
radius correction; it consumes no labels, simulator state, instance maps, or
object IDs. Its 18-case grid reaches `0.00336217 m` position RMSE and has
finite gradients to centre, RGB, and depth. Focused validation is `29 passed`
and independent review passes, without an episode seed namespace or protected
access.

This is not a training or convergence result. Specification 1.54 freezes the
new observable-depth temporal protocol at config SHA-256
`5667cdb3603682b8d80a3e42793d25e36989269df1afacfa9b1028f2451101e9`
and canonical protocol payload SHA-256, computed before inserting its
self-reporting digest field,
`4e334e9d7942ea3f2416c0a9f5ca8e327d1d0a1e9131074f20c051ebd3163ad7`.
The source/config contract can be inspected without generating data:

```bash
conda run -n orpheus python scripts/run_rgbd_temporal_free_motion_ladder.py \
  --phase protocol \
  --config configs/rgbd_temporal_free_motion_cpu.yaml
```

At the specification-freeze boundary, development `41000000--41000023`,
selector `42000000--42000023`, confirmation `43000000--43000023`, and
one-shot final `44000000--44000047` were all fresh and unopened. Development
v1 was later invalidated as recorded below. Fresh v2 development and the
ledgered protected sequence have now completed. Do not run
`--phase qualification` again: final is consumed.

The estimator is parameter-free: its parameter/buffer/state-dict counts and
optimizer updates are zero. It measures sixteen RGB-D frames, gives every row
uniform weight in the differentiable exact free-motion WLS fit, and queries
state at `0.1/0.25/0.5/1.0/2.0 s`. Confidence and Boolean validity are
diagnostic/fail-closed only. The protocol gates current and per-axis state,
every horizon, stationary/zero-velocity/two-frame baselines, semigroup,
resources, fixed-output VJPs to RGB and depth, an explicitly worse RGB-only
control, and zero-validity missing depth. Its finite OLS covariance is only an
i.i.d. residual diagnostic; no calibrated posterior, coverage, or proper-score
claim is made.

The first development run used clean commit `8e68035` and seeds
`41000000--41000023`. All `82` reported scalar metrics were finite and all
frozen gates passed. Current
position/velocity RMSE was `0.00279934 m`/`0.00207092 m/s`; horizon position
RMSE was `0.00286018/0.00297530/0.00322108/0.00385564/0.00539692 m`; measured
perception and rollout time was `0.235061 s`/`0.00391524 s`. Its exact
SHA-256 evidence is:

- report: `9cbea9f25181769ee5b6a87b097e738a29cdb9b386c8018b3044f07d58aa03e2`;
- checkpoint: `6acd88edd203cdebb2b0820bad388e06a4c610ea1c659ff9f8ea6d701ad28059`;
- development manifest:
  `b92f1e3f12475986ebd2971ad2de70187432c0941caa0335ea53e82abc3c1d01`.

Treat that result as a historical conditional pass, not qualifying evidence.
Audit found a raw Python tuple/list equality comparison after JSON roundtrip.
The repaired source compares the canonical JSON SHA and adds regression
coverage, which changes the source relative to commit `8e68035`. Therefore the
old report/checkpoint pair must not authorize protected access. Rerun
development from a fresh clean repaired source to fresh v2 report/checkpoint
paths, then obtain independent exact-digest review. That instruction is the
historical v1 invalidation record; it has now been satisfied.

The qualified source is
`df0235a92a81d3c1d2ba4e69e64d639562e3dfe8`. Fresh v2 report/checkpoint
SHA-256 are
`4cf1657ee95645c8c647433a8be660520e9cdc1a5e6ac106d85bd24547b4e740` and
`fd663e5fa52dded8156a3178070966e3458d93a7b5a49dd5dcb2cc0d6278514e`.
Its scientific metrics match v1; perception/rollout latency is
`0.233866867 s`/`0.003934262 s`, and maximum RSS is `545804288` bytes.

The exclusive ledger then recorded selector, confirmation, and final before
each access. All passed exactly once. Qualification-report, canonical-summary,
and ledger SHA-256 are
`7e4cface087620f058ade4cc83ac5fd197685ba26c8f0afb5089d8f7e646fe0d`,
`7e9954ae34ce55b6923765de0c084d5075f238bd012554eeb44049a0db161658`, and
`9fc139291dfb34b10125321d06fdf06ab68ed65df32f62c273a95e5ca7aa7b8b`.
Manifest SHA-256 values are development
`b92f1e3f12475986ebd2971ad2de70187432c0941caa0335ea53e82abc3c1d01`,
selector `56cb85d9a30129f6e7153075b7334fd4737986f1c9679650533d20e7e0763cf8`,
confirmation `eb61687ec0a8508a563b6e7c3dfb67b4393a6eaa9d35b2788eaa373d79e5df16`,
and final `42e0320bf6f62b78c881951ec78486714b4432a1592fe2a5047027e2bbd0339f`.

Every split reports `82` finite metrics and all `103` gate comparisons pass,
with zero failures, optimizer updates, or learned state. Development, selector,
confirmation, and final current position/velocity RMSE in mm and mm/s are
`2.799/2.071`, `3.073/1.991`, `3.078/1.644`, and `2.905/2.226`; their
two-second position RMSE is `5.397/5.774/4.328/5.560 mm`. Final two-second
error is `18.5%` of gate and `3%` above development. The remaining horizon,
runtime, VJP, semigroup, baseline, and RGB-only-ablation gates all pass as
bound in the qualification report.

The standalone source gate was `104 passed` focused and
`1130 passed, 16 skipped in 414.82 s` complete. The later integrated bridge
first boundary passed `1207 passed, 16 skipped in 431.10 s`; canonical
comparator source now passes `1209 passed, 16 skipped in 434.37 s` complete.
OLS covariance remains diagnostic, not calibrated.
Artifacts are SHA-bound and tamper-evident but owner-writable. Preserve them;
do not rerun final.

The standalone rung is qualified and may now enter the public runtime bridge.
The first bridge must use one batched composite `rgbd` packet carrying
`[B,3,H,W]` RGB and `[B,1,H,W]` depth plus calibration and explicit image
size. A modality-qualified sensor key avoids current cache/history/scheduler
collisions. Separate same-timestamp RGB and depth packets are outside the
qualified contract. Runtime may consume only RGB-D, calibration, timestamps,
and declared priors—never simulator state.

## Qualified public bridge — final consumed once

Specification 1.55 implements and freezes that public one-slot bridge. The
exact config SHA-256 is
`c40b3438c7fd60646d356db3fe54050039912ace288d9db89620b626106993a3`; the
seed-free canonical `bridge_protocol()` SHA-256, before its self-reporting
field is added, is
`e536b0d0b721042bff55501faf3445456219fcc987334b6ec1e892688ea560b2`.
Development `45000000--45000023`, selector `46000000--46000015`, confirmation
`47000000--47000015`, and final `48000000--48000031` were all unopened at
source freeze. The audited sequence below has now consumed them exactly as
declared; final must never be rerun.

The runtime accepts one composite batched `rgbd` packet. Raw observable metric
position has one direct correction owner. Sixteen raw positions, aligned by
persistent object ID, enter a uniform differentiable exact free-motion WLS fit
that emits velocity-only evidence. Parameter-free analytic dynamics answers
all five future horizons. The bridge owns zero parameters and state-dict
entries, and no optimizer may be constructed for its qualification.

The qualification gate requires fixed-output VJPs to RGB and depth separately
for current velocity and every horizon at every input frame. For an already-
active persistent object, a well-formed frame with missing depth, nonfinite or
otherwise invalid depth in the sampled measurement support, or no foreground
appends an invalid causal row. In the frozen sixteenth-frame full-window
ablation, diagnostics are `sample_count: 16` and `valid_count: 15`. The invalid
measurement emits no valid/admissible temporal fit or direct velocity evidence,
correction, or birth. A finite diagnostic fit may be computed with
`fit_valid: false` but is not admissible evidence. Before birth, the same
invalid frame advances runtime time but creates no object-history row and no
birth. Malformed packets, nonfinite RGB or calibration, float16/bfloat16,
stale or duplicate stream evidence, an unknown modality, or invalid prepared
propagation reject atomically without changing or consuming runtime state.

Full batch-four persistent runtime tensor storage is counted recursively by
unique storage and must remain within `32,768` bytes; current frozen source
measures `25,364` bytes. The current targeted source/config/protocol gate is
`421 passed in 62.72 s`, the complete repository gate is
`1209 passed, 16 skipped in 434.37 s`, and independent source review passes.

Public RGB-D evaluator and demo reports carry truthful
`observation_modality: rgbd` and `rgb_only: false` metadata. The evaluator
labels its fifteen-frame warmup and must not be compared as if it were the
standalone qualification. Demo aggregate errors remain pooled across warmup
and post-warmup frames; they are not warmup-separated accuracy evidence.
Legacy RGB execution remains unchanged. Model checkpoint round-trip covers the
parameter-free module/configuration only: live temporal histories and caches
are not serialized, so exact mid-history stream resume is unsupported and
must be rebuilt by replaying observations.

The first clean development on `ebda5a8` had all `175` reported scalar metrics
finite and passed all frozen gate checks, but
its audit rejected raw tuple/list protocol comparison after JSON roundtrip.
No protected data was accessed. Its report/checkpoint are archived under
ignored `runs/rgbd_online_bridge_v1/rejected_ebda5a8_json_protocol/` with
SHA-256
`2104ee87bcabdbd5312b4026a33e44e1de7d197e50215ec7f0bf0e0bb56992e3`
and
`38f4b2ef5addb98bb966360213d3bb36b43da606367fc60cd75d2ec487f1b866`.

Commit `526b5123e6385c575a5777936272330d28972b93` compares canonical
protocol JSON and rejects tampering. Audit runtime/worktree fingerprints are
`1eeaa176ad9be8886976910fe53028fb6de498adda73a2d20170f206b6134b40`
and
`90d0624a119e118e76b58061f7e5582dffc906f47d85cc4dde997b2f765bb07a`.
Fresh canonical development report/checkpoint SHA-256 are
`dce6f920da85fbf696b7ae8a7a91d9cbf7d9084176e51ad7c319f92a6efe4966`
(`22,346` bytes) and
`48249f1a5a0467b1da8c7bdb5ad9e909f8c502631ec2fbad832cb490a00c3099`
(`46,596` bytes); manifest SHA-256 is
`069eb3331543727c911a07cc9a1bb352f6185ac8ceac7fafca502c9d7fab6d80`.
All `175` scalars are finite and all gates pass. Current position/velocity is
`3.068470 mm`/`2.191966 mm/s`; two-second position/velocity is
`5.609913 mm`/`1.983371 mm/s`; slope is `1.270721 mm/s`; perception/rollout
is `0.415134 s`/`3.575380 ms`; and persistent runtime state is `25,364` bytes.

Only that independently approved pass created the exclusive protected ledger.
Selector, confirmation, and final were recorded before materialization and
passed exactly once. Qualification-report/ledger SHA-256 are
`7fd1829f663606910ac81990e4b633c63b1460dbc31dd24c71eedbd91b422908`
(`47,353` bytes) and
`cf6a10dd672aafbdd91c92871ae349fef0c549d865cc6532e6c42f7d9be14e32`
(`1,626` bytes). Initial/final empty state SHA-256 is
`4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.

| split | manifest SHA-256 | result SHA-256 |
| --- | --- | --- |
| selector | `2159b044e089774b3b7df95509ac2cded19528de6ff133ae1b158a354ed7fbb9` | `9ac6b7cc1b97da9961345fdcf5488ddec3ac6a0186215699a55a66acfbb983cb` |
| confirmation | `2cad3224740b4d73871ff1d1e60795d45dc149ad03d197513eddf514cb9946bf` | `1a3996914d59f840b2645e4b886f1027b830fa6f81c5763eb1735f25149aa9bc` |
| final | `3c5c904203ddd46ea790322e446466b2c58e603015456f239715aa07135011a3` | `40d39accec8c2c6efa97f06a2f2748c580a5666b54c7dac4df36e3d7dc718bd1` |

| split | current position / velocity | two-second position / velocity | slope | perception / rollout |
| --- | --- | --- | --- | --- |
| development | `3.068470 mm` / `2.191966 mm/s` | `5.609913 mm` / `1.983371 mm/s` | `1.270721 mm/s` | `0.415134 s` / `3.575380 ms` |
| selector | `3.177543 mm` / `2.313401 mm/s` | `5.881384 mm` / `2.093251 mm/s` | `1.351921 mm/s` | `0.422070 s` / `3.569962 ms` |
| confirmation | `5.681172 mm` / `1.658775 mm/s` | `6.188252 mm` / `1.500921 mm/s` | `0.253540 mm/s` | `0.414407 s` / `3.537710 ms` |
| final | `2.996787 mm` / `2.221047 mm/s` | `5.433965 mm` / `2.009688 mm/s` | `1.218589 mm/s` | `0.417436 s` / `3.566628 ms` |

Every split has `175/175` reported scalar metrics finite with no gate failures;
all required history VJPs reach
`16/16` frames, identity change is zero, and RGB-only, missing-depth,
no-foreground, semigroup, memory, checkpoint, and zero-state gates pass. Final
state is `25,364` bytes; maximum RSS is `708,853,760` bytes; semigroup error is
at most `2.384186e-7 m`/`1.862645e-9 m/s`.

The ledger is complete and stopped after final. Evidence is atomically replaced
and hash-bound but owner-writable, not OS-enforced append-only storage. Audit
did not reinspect or rematerialize raw protected episodes. Do not rerun final,
tune on it, change this accepted rung, or enlarge capacity inside it. The bridge
is merged to `main` at `3eed0b71e6f18c7036bf376c075493a89d5fdc9f`; the
next rung must be separately predeclared before any new multi-object,
association, contact, task, or capacity data.

The complete `1075 passed, 16 skipped` repository gate belongs to the clean
pre-failure source commit above. After deletion of the rejected experiment and
addition of the RGB-D core, the new complete source gate passes `1091` tests
with `16` expected inactive-device skips in `418.49 s`. The integrated bridge
tree's accepted one-object boundary later passed
`1209 passed, 16 skipped in 434.37 s`; the current specification-1.56
two-visible-object source result is recorded in the section below.

The broad `train.py`, `evaluate.py`, and `demo.py` workflow remains available
for `OnlineWorldModel` smoke/integration checks, but no older sustained profile
is an active accuracy campaign or deployment incumbent. Exact-resume,
checkpoint, validation-support, and promotion integrity remain tested reusable
contracts.

Historical campaign commands and evidence through specification 1.51 remain
in Git commit `c16acc99` and the ignored local pre-generalization archive.

## Historical two-visible-object source-freeze workflow

This section preserves the pre-access contract and exact commands that governed
the subsequently accepted qualification. Final is now consumed; do not rerun
either episode command below. Specification 1.56 froze architecture attempt 2
at exactly two fully
visible, image-separated, non-contact fixed-radius spheres. It preserves the
accepted one-object path. Parameter-free differentiable chromatic-plus-spatial
two-slot RGB-D geometry produces unordered metric measurements; hard Hungarian
is used only for discrete stable identity. Direct metric position has one
owner. Sixteen persistent-ID-aligned raw positions enter uniform
differentiable exact free-motion WLS for velocity, and analytic dynamics answer
position and velocity at `0.1/0.25/0.5/1/2 s`. No optimizer is constructed and
optimizer updates remain zero.

Frozen config/protocol SHA-256 values are
`84e6f44b818bb9323a774bdba9492ef056e2a2747b93517fa38497ba83218bba` and
`42b9dca23fed303d5cee4641c8d8753977a872fc90d0b1086658d7f12b823ea0`.
The exact harness, harness-test, and runner SHA-256 values are
`198cac1c4d683e3c983f70c0106827aaf883636d4bd6454e94011c3975c1b64a`,
`d5dd3c18515589b4589e0179a68e29112d45987a513308df022cece5bf75e896`, and
`a8e6d9f51380eede3b6a94f085e9741f67883e2740c6203c16aec4a5dcfa1bc1`.
The simulator remains `sphere_world_v7`.

The fixed VJP contract distinguishes direct current measurement from temporal
outputs. Current position is owned by anchor frame 15 only (`1/16` reach, with
every non-anchor gradient exactly zero). Current velocity and all five rollout
positions and velocities must reach all `16/16` RGB and depth frames. Each
split's B4 audit comprises four distinct scenes and requires exact zero
cross-scene coupling.

Source-only validation passed `43` focused tests,
`281 passed in 15.61 s` across the accepted one-object/configuration/two-object
harnesses, and `1275 passed, 16 skipped in 447.29 s` repository-wide. Ruff
lint, Ruff format-check, diff integrity, and two independent audits also passed.
These commands did not generate an episode and are not accuracy evidence.

The canonical seed-free protocol-inspection command was:

```bash
conda run -n orpheus python scripts/run_rgbd_two_visible_qualification.py \
  --phase protocol \
  --config configs/rgbd_two_visible_free_motion_cpu.yaml
```

The exact frozen tree then had to be clean and committed before the one fixed
development attempt could use fresh paths:

```bash
conda run -n orpheus python scripts/run_rgbd_two_visible_qualification.py \
  --phase development \
  --config configs/rgbd_two_visible_free_motion_cpu.yaml \
  --report runs/rgbd_two_visible_bridge_v1/development_report.json \
  --checkpoint runs/rgbd_two_visible_bridge_v1/development_model.pt
```

The qualification phase was authorized only after that exact development
report and checkpoint passed independent digest review:

```bash
conda run -n orpheus python scripts/run_rgbd_two_visible_qualification.py \
  --phase qualification \
  --config configs/rgbd_two_visible_free_motion_cpu.yaml \
  --development-report runs/rgbd_two_visible_bridge_v1/development_report.json \
  --checkpoint runs/rgbd_two_visible_bridge_v1/development_model.pt \
  --report runs/rgbd_two_visible_bridge_v1/qualification_report.json \
  --reviewed-checkpoint-sha256 <reviewed-checkpoint-sha256> \
  --reviewed-report-sha256 <reviewed-report-sha256>
```

The frozen namespace/hash bindings are:

| split | exact seed range | manifest SHA-256 |
| --- | --- | --- |
| development | `49000000--49000031` | `5a47a1a4a1405ba4c2fc3bce0087131d98fabfceb899beb26c6b4ba824a130f8` |
| selector | `50000000--50000023` | `415bc33407a46b79d0a3a746a8f5b192e31cfd4f6a68b9764e9b9943b7e6d7fe` |
| confirmation | `51000000--51000023` | `14f7dc3b762e4f987acbedcece815abd1c262bc9da60322f7f054e2c4eb4b3b1` |
| final | `52000000--52000047` | `b7e8913e938e2f7ae7f937979a60279916ff1a06f071427bcce9f08b0e354e75` |

At the freeze boundary all were unopened and no episode, report, checkpoint,
or ledger existed. The fixed development ledger path was
`runs/rgbd_two_visible_bridge_v1/development_attempt_2_access.json`; it was
created before development materialization and alone authorized the
constructor. After a reviewed pass, the protected ledger at
`runs/rgbd_two_visible_bridge_v1/qualification_attempt_2_access.json` did the
same for selector -> confirmation -> final exactly once. A failure would have
stopped immediately, left later splits unopened, and permitted no retuning,
renamed retry, or final reuse. Checkpoint review used an empty,
optimizer-/RNG-free payload loaded with
`weights_only=True`; terminal report bytes preceded the ledger's terminal
digest.

This rung excludes occlusion/reappearance, variable object count, contact,
camera motion, variable physical parameters, uncertainty calibration,
tasks/planning, extra modalities, and learned capacity. Ordinary checkpoints
do not serialize live histories, so exact mid-history resume is unsupported.

## Accepted two-visible-object qualification — final consumed once

The exact workflow above passed on commit/origin
`3b781e653a0287b2aa926e7c0b969e9197d48e42`, runtime fingerprint
`810b237082ae99735527985c544dc28834b806489c555b464191c3b3e62520e7`, and
worktree fingerprint
`fdbbe6fe3a85b491578d4cda2dc880f1dc21726f3469d3717976662796f12f23`.
Config/protocol/empty-state SHA-256 values are
`84e6f44b818bb9323a774bdba9492ef056e2a2747b93517fa38497ba83218bba`,
`42b9dca23fed303d5cee4641c8d8753977a872fc90d0b1086658d7f12b823ea0`, and
`4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.

Development report/checkpoint/ledger SHA-256 values are
`dfed30c29b7dc07adcfd01a233e3de3a42f32d8b333a1b8d696dae144af98f4b`,
`e59e0d4b0f8f747b38fb0699cbf9f1491f0ca81dbffdcde5bcb56b04002c6bed`, and
`9a49f574f6b7bdc0211d01d1ad4e5591d9155e7191f0a04b5e48e864ca56e579`.
Qualification-report/ledger SHA-256 values are
`085f5206ac02f01fa5d7f5bc0cad055f75d401447cc090c800431fabf89ef1ef` and
`3cc22f65f809ad4afb08bf26a1984157beb1076acb5fc910d8edc3f5df0035af`.

| split | manifest SHA-256 | result SHA-256 |
| --- | --- | --- |
| development | `5a47a1a4a1405ba4c2fc3bce0087131d98fabfceb899beb26c6b4ba824a130f8` | `2eaefcf40b459414492e849d24bbf50fc4638294dedfe4b5350fc011b599cfa2` |
| selector | `415bc33407a46b79d0a3a746a8f5b192e31cfd4f6a68b9764e9b9943b7e6d7fe` | `ede4e91e708645a761065ff43993e1df05800422673d5be1b1f77b2bd3c001ce` |
| confirmation | `14f7dc3b762e4f987acbedcece815abd1c262bc9da60322f7f054e2c4eb4b3b1` | `204e5f5a65c73b721e038cf50ef732068ba4a901a68c78c8cb8d7f79a60b4ad8` |
| final | `b7e8913e938e2f7ae7f937979a60279916ff1a06f071427bcce9f08b0e354e75` | `7b9ba4df3a2595c9a671322f6650ed170a0b3cfbd092d9bf46612abbe9db6dae` |

All four splits have `396` finite gated metrics and no failures. Their current
position/current velocity/two-second position RMSE values are respectively:

- development: `1.9029872e-5 m` / `3.1932373e-5 m/s` / `7.4638663e-5 m`;
- selector: `1.6885011e-5 m` / `3.1409136e-5 m/s` / `7.0948345e-5 m`;
- confirmation: `1.6776625e-5 m` / `3.2594633e-5 m/s` / `7.2466125e-5 m`;
  and
- final: `1.7838631e-5 m` / `3.1881889e-5 m/s` / `7.1961138e-5 m`.

Final two-second velocity RMSE is `2.8845629e-5 m/s`; worst final per-axis
position/velocity RMSE is `1.020557e-4 m`/`4.503076e-5 m/s`. Identity coverage
is one with zero switches, mismatches, or ambiguities. Minimum Hungarian margin
is approximately `199.96`, visibility is one, event count is zero, and minimum
silhouette/world gap is `9.534 px`/`0.75985 m`.

Every split has current-position reach on the one anchor frame with exact zero
non-anchor gradient, `16/16` reach for current velocity and every rollout
position/velocity, four distinct B4 scenes, and zero cross-scene coupling.
Final minimum total/minimum temporal-frame/maximum VJP L1 is
`2.0798e-5`/`5.7799e-8`/`4.2197`. Final perception/five-query rollout time is
`0.352510 s`/`0.00359524 s`; persistent tensor state is `28,512` bytes and
maximum RSS is `579,817,472` bytes. Learned parameters, module buffers/model
state, optimizer/scheduler/RNG state, and updates remain zero.

The exact source gates are `43 passed` focused,
`281 passed in 15.61 s` combined, and
`1275 passed, 16 skipped in 447.29 s` complete. Independent final audit passes.
Both ledgers are complete/passed in development then selector -> confirmation
-> final order. The audited directory has exactly five single-link regular
files with no temporary, alias, or second-attempt artifact. Audit did not
reopen raw protected episodes; owner-writable evidence remains tamper-evident,
not OS-enforced WORM storage.

Final is consumed. Do not run development or qualification again, and do not
tune any threshold, architecture, or capacity against it. The reviewed tree is
merged to GitHub `main` through
`1e951520e5a2bf06c1932f64b8334e552247de82`; the separately frozen bounded
partial-visibility and missed-observation-recovery rung follows below. It
retains exactly two objects, fixed camera/physics, non-contact motion, the
parameter-free analytic rollout, and every accepted lower-rung gate.

## Historical partial-visibility attempt-1 freeze and failure

Specification 1.57 froze architecture attempt 1 before any episode access.
At that boundary the public parameter-free RGB-D path kept exactly two
fixed-radius, fixed-camera, non-contact spheres with known gravity/drag. It
added bounded
partial visibility and exactly one target-local missing-depth observation at
frame 15 or 16 in the one-miss strata. Sixteen WLS rows permitted at most that
one invalid row, required frame 17 valid, and retained fifteen missed-target
supports. Exactly one filter-owned `0.08 +/- 1e-6` miss-variance increment had
to be followed by immediate next-frame same-ID recovery and an otherwise
unchanged all-`FREE` trace. Full occlusion/reappearance, contact, variable
count, and capacity generalization were outside the workflow.

Frozen config/harness/runner/test/protocol SHA-256 values were
`7d563382a8f4b6e301ac30510152f1b1409da32248aacf15dff460ea71d29e2c`,
`99084d9fb421faa8dbe7ef20f7a88ee5e196cce498586c0fae2b92eebddc36d4`,
`c97f20638c876045cb25adfe23d39db6daed749e42ab5eed1dea6aacac8dd90f`,
`e712f9b6ee1cd8775f8f8a1d07ee0844fe1ac1e8ac73a2a2233c9a231cce892e`,
and
`e178d572a238c17eaa4c23f1b0942e2c4e70103a73af3ab51736fffe36b0d8fd`.
The simulator was and remains `sphere_world_v7`.

| split | exact seeds | pure manifest SHA-256 |
| --- | --- | --- |
| development | `53000000--53000031` | `ca1fb17e87df5216c4429342f74dcccd2c31b11b8d48bb3c76eee27e139cf391` |
| selector | `54000000--54000023` | `1b1e6ef6938705bcc7e2a66ad5ee4622860c9ea9ec3e6c19c86e8a8534209b28` |
| confirmation | `55000000--55000023` | `72d7c922029d300e3d28409bcb55a843633caac10b482f680ae769a442739e9f` |
| final | `56000000--56000047` | `70b60f48769a26c5587febf778443fd38f5814a39e80ec7da1c98dea9c389ded` |

The protocol required cross-split unique pure scene signatures, 98
authoritative gate fields, exactly 2,167 finite scalar metrics per eventual
split, target-region RGB/depth VJPs, exact zero scheduled-miss and cross-scene
gradients, global all-`FREE`/no-spurious-miss tracing, exact ledgers, and the
canonical five artifacts. Seed-free combined validation was
`436 passed in 60.26s`; Ruff/format/diff and two independent audits passed. The
full exact current-byte repository result was:
`1398 passed, 16 skipped in 487.93s (0:08:07)`.

At the historical freeze all four namespaces were unopened and none of the
five canonical evidence files existed. The required next order was exact clean
commit, bound complete repository gate, sole development, independent digest
audit, then protected selector -> confirmation -> final only after a pass.

The exact clean source commit was
`7e67823667769e47bad3678207f2c01bd3edbfe4`. Its sole development
authorization terminated during private constructor/preflight: seed
`53000000` completed construction; seed `53000001` rendered `58` frames and
failed exact renderer visibility at frame `4`, where mild rear support/visible
was `20/15`, so `0.75 < 0.80` despite continuous visibility `0.826827`.
Failure occurred before model, collate, or runtime.

The immutable development report is `13,948` bytes with SHA-256
`7c08c794690a10d46100b8d17ee448e3a83960d265ec7859bb91cd6d2ac9ca9d`;
the `1,110`-byte ledger has SHA-256
`e4993abefefe07e0b0fb57a65769fa270012524d62c8ebab4b7db0251979aab4`.
Runtime/worktree fingerprints are
`2345bcf6d785cd864301dbcdcb23cc8f7287f1815615fd1e30e6f635084f12c3`
and `0d44cabadce831238fe1c8c1cda450677b62f20af3fcf9a411fa4ef621b1842f`.
In-memory cursor `2` is inferred rather than durable; seeds `>=53000002` were
untouched. No checkpoint exists. Protected 54m--56m never opened and are
permanently unused. Exactly two single-link live files—the report and
ledger—remain. Attempt 1 is consumed and must not be resumed or renamed.

## Frozen partial-visibility recovery workflow — attempt 2 not run

Specification 1.58 retains simulator `sphere_world_v7`, the accepted
specification-1.56 base, the failed specification-1.57 evidence, and the
public history, miss-isolation, and recovery semantics above. The terminal
architecture attempt uses one finite table of `16` rational
primitives crossed with the `8` exact `D4` transforms, yielding `128` unique
physical cells. Float32 world evolution is an exact `342`-substep recurrence
identical to the public solver. Exact raster support owns renderer visibility.

The world/renderer/table/absolute-table/ordered-state/state-set/unordered-
geometry SHA-256 values are
`32b34e716ec639cabdd5d36f1c0d30fa17b187546bb5653e4fa7d0a9d6af65d4`,
`4362f06929f8e8958c1f12e8d2077dded6f8dda3bfdb99eed425899bb289f412`,
`c3f17e805de234fecb1f1928b47e8fd2127d608447e7b1e87df9a2ec970ce3aa`,
`f86f218317d656c16f4c85e5b4a75b2e52094724316a3132b0a6e44715bec86e`,
`bc3e6349fc0d5effecbb53920a9c4224203067f05306330723f8c75dd9f35c57`,
`96a53595bf7d21b84fed772baef4b754b6e777b7560a8083d303814fa5f611b5`,
and `27a8dabb2d9936e635cde5b2155fffa5eddb89679b477175119917627772cafa`.
Certified discriminant, overlap-depth, projected-drift, and D4-conjugacy
margins are respectively `5.20199537e-5 >= 5e-5`,
`0.831737 m >= 0.8 m`, `1.144409e-5 px <= 2e-5 px`, and
`2.861023e-6 m <= 4e-6 m`. Minimum actual visibility margin is `0.05`.
One-pixel hypothetical clearance is exactly `0.0` under an inclusive gate and
must not be called positive slack.

Frozen config/harness/runner/qualification-test/config-test/protocol SHA-256
values are
`b18f787987394f77771dbf31dae1642bd042b81e64b02a3e93b8cd048dd3416b`,
`859dedf68031ee66cec1334d2fc094078bc2aacf0deac4388c53337033b63519`,
`a16c0712b611ebe64dd5052efde3f73e3c5aa18f1b1c5f825f571c2674e598c0`,
`f4d35320f484484429cdfadb9f3faed6ad5c1ad85492d6ffaa378a7076955714`,
`8f4e14c7ccff5c6af4d820c555956ce12b66854b1aef7ee3fb5ddbaad7abd40a`,
and `5f049f060f6e8a9682d9413e6bc2d8f9f228f6e2aee67cde16f98d234cac8a3b`.

| split | exact seeds | pure manifest SHA-256 |
| --- | --- | --- |
| development | `57000000--57000031` | `ded3d75a7d248e3f9746b03b0cf249f32739208713c4287c45deb5eefd11f8e2` |
| selector | `58000000--58000023` | `effa598aa07a44c100da115f71828e00754f181729063899353d22b551f7227a` |
| confirmation | `59000000--59000023` | `9240a1dd465574de8ac032e318f3cee618909ed6a5b3e91c5fd8c87bad146cec` |
| final | `60000000--60000047` | `17fdd50896729b981357960ea0db74ef19e059e21bc8d8e41a7048cf237200a6` |

Their signature-list SHA-256 values are development
`8426ea4d0a7e1d507c5d7fc825afa8864ee694a04df622cba955b92ffd4350c0`,
selector `d421862763a3e0bc0af042fd81704c836c2123ad0fa260130e791cb250c0b2c7`,
confirmation
`261f975fcd46795ff9f56c94857de69942ea047455f65cc0341bdc515cc76af5`,
and final
`1837d40a35ddba88e3a91f74c5b2c398aa01675ad8e84efa2fe660bbf49e34a2`.
All are unopened; no v2 artifact exists.

The v2 run root is `runs/rgbd_partial_visibility_recovery_v2/`. Raw
construction/evaluation are private. A direct guard protects the immutable
live-v1 report/ledger, tracked fixtures match those exact bytes, and only an
exact canonical ledger-minted capability may construct a scene. There is no
oracle. Exactly five canonical v2 single-link files are permitted, and the
restricted checkpoint must load with `weights_only=True`.

The canonical gate passes `315 passed in 345.27 s`; the public-solver proof
passes `1 passed in 244.52 s`; the full suite passes
`1407 passed, 16 skipped in 816.14 s`; and two independent final audits report
`PASS`. These are source/security proofs, not episode evidence.

Do not execute development yet. First commit and push the exact source/docs
freeze, then prove the tree clean with `HEAD == upstream`. The development
ledger may then mint the sole attempt-2 capability. Independently audit a
passing report/checkpoint/ledger before selector -> confirmation -> final is
authorized exactly once. Any development failure stops the rung permanently;
there is no architecture attempt 3.
