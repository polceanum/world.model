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
is ready to merge to `main`; the next rung must be separately predeclared
before any new multi-object, association, contact, task, or capacity data.

The complete `1075 passed, 16 skipped` repository gate belongs to the clean
pre-failure source commit above. After deletion of the rejected experiment and
addition of the RGB-D core, the new complete source gate passes `1091` tests
with `16` expected inactive-device skips in `418.49 s`. The integrated bridge
tree is the current complete repository result at
`1209 passed, 16 skipped in 434.37 s`.

The broad `train.py`, `evaluate.py`, and `demo.py` workflow remains available
for `OnlineWorldModel` smoke/integration checks, but no older sustained profile
is an active accuracy campaign or deployment incumbent. Exact-resume,
checkpoint, validation-support, and promotion integrity remain tested reusable
contracts.

Historical campaign commands and evidence through specification 1.51 remain
in Git commit `c16acc99` and the ignored local pre-generalization archive.
