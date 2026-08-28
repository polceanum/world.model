# Project status

## Active generalization program — 2026-08-28

The pre-generalization public base was commit
`c16acc99ef13757fc8f88528bfd0d66db4a2f4fd`; the cleaned generalization
foundation is GitHub `main` commit
`08ae63adc5ade2e5061f54539fc7a25564c8c8d6`, and the accepted public RGB-D
bridge is merged at `3eed0b71e6f18c7036bf376c075493a89d5fdc9f`. Broad heterogeneous training
remains paused. The active contract is specification 1.58. The accepted base
now comprises the specification-1.51 differentiable one-sphere unit, the
qualified standalone two-second RGB-D rung, and its qualified public one-slot
`OnlineWorldModel` bridge. The accepted branch additionally contains the
exactly-two-visible RGB-D qualification recorded below—not any older campaign
checkpoint. The specification-1.57 partial-visibility prefreeze is historical:
architecture attempt 1 failed immutably during development before model,
collate, or runtime. Specification 1.58 freezes the terminal architecture
attempt 2. That tree is ready for commit/push only; no v2 namespace is opened,
and no v2 development or accuracy result exists.

The qualified unit achieved, on its single previously untouched final set:

- RGB world-position RMSE `0.00764440699 m`;
- image-centre RMSE `0.00522461 px`;
- apparent-radius relative RMSE `0.00219904` (`0.2199%`);
- `0.1 s` rollout RMSE `0.00799061917 m`; and
- finite measurement validity `1.0`.

Its runtime path is ordinary autograd from RGB through soft foreground
evidence, four finite-difference Gauss--Newton inverse-rendering stages,
calibrated backprojection, temporal state estimation, and analytic tensor
kinematics. Simulator state remains supervision/evaluation only. The accepted
implementation is commit `7344e67d`; promotion bookkeeping is `c16acc99`.
The original qualified research source is now recoverable through GitHub tag
`archive/minimal-differentiable-toy-v2-f8d66da`, which peels to
`f8d66da17983aa0269649fff69cc13cec5ad1311`.

### Cleanup boundary

The 254-run historical workspace was archived and removed from active use.
All non-checkpoint metadata, the exact compatibility fixture, the qualified
report/legacy artifact, and a complete source bundle occupy approximately
`33 MiB` under ignored local path
`.archive/20260826-pre-generalization/`. The superseded run tree and 660
duplicate checkpoints occupied about `7.6 GiB`; they were intentionally
deleted after archive verification. `runs/` was empty at that cleanup boundary;
it now contains later ignored RGB-D temporal and public-bridge qualification
evidence described below. Repository caches, generated demos, and selected
stale temporary clones/caches were also removed.

Four unreferenced rejected campaign profiles and three one-off campaign tools
have been removed from the active tree. Reusable typed contracts, analytic
physics, checkpoint integrity, evaluation metrics, and smoke fixtures remain.
Historical status, task, training, and accuracy records remain in Git commit
`c16acc99` and the ignored local pre-generalization archive; active tracked
memory is intentionally concise.

### Terminal result: monocular temporal free motion

The closed isolated rung kept the already identifiable one-sphere, fixed
camera/radius/gravity/drag, and contact-free world. It changed only temporal
estimation and horizon:

1. fit anchor position and velocity from a bounded RGB-derived history with a
   differentiable closed-form weighted least-squares solution to the exact
   linear-drag equations;
2. roll forward only through `AnalyticKinematics`; and
3. evaluate `0.1/0.25/0.5/1.0/2.0 s` horizons, semigroup consistency,
   gradient reachability, calibration diagnostics, and separated perception/
   state-rollout throughput.

The exact differentiable free-motion basis itself passed its oracle and
semigroup gates. The terminal architecture attempt 2 of 2 then ran its
development-only protocol from clean commit
`8889818619121351d342490786331e854364532c` and failed 10 accuracy and
trivial-baseline gates. Audit current position/velocity RMSE was
`0.016128 m`/`0.070461 m/s`; horizon position RMSE at
`0.1/0.25/0.5/1.0/2.0 s` was
`0.022907/0.033205/0.050360/0.084191/0.149501 m`. Oracle state, direct
equations, simulator agreement, gradients, geometry, semigroup consistency,
latency, and memory remained correct, localizing the failure to temporal
measurement/weighting rather than rollout physics.

The learned reliability taper reached `10.0338 /s`, reducing the
oldest-to-anchor weight ratio to `0.000534`; `77.63%` of fit mass fell on the
last three frames and `91.71%` on the last five. A confidence-only
development diagnostic improved current position/velocity to
`0.00842 m`/`0.01660 m/s` and the five horizon position errors to
`0.01000/0.01239/0.01638/0.02430/0.03963 m`. It still failed: future velocity
was `0.01652 -> 0.01502 m/s` versus the `0.01 m/s` limit, and the early
zero-velocity-baseline ratios exceeded `0.5`.

The immutable failed report is archived locally at ignored path
`.archive/20260826-pre-generalization/temporal-free-motion-attempt-2/development_report.json`
with SHA-256
`be488d045e259c0804a2a2b24215fa4eb3025d69f6113d8dbefe21d72f827554`.
It states `protected_data_materialized: false`. Selector seeds
`32000000--32000015`, confirmation seeds `33000000--33000015`, and final
seeds `34000000--34000031` remain unopened. The failed config, runner,
estimator, and tests have been removed from the active tree. Do not retry this
family or access its protected data.

The next structural rung is observable-depth/RGB-D temporal state under the
new frozen protocol recorded below, not a third monocular reliability-taper
attempt. Its single-frame metric measurement core now passes as recorded
below. At that historical boundary, the planned order placed public
`OnlineWorldModel` integration before moving camera, identifiable drag,
variable metric scale, and then multiple objects. Specification 1.56
supersedes that ordering after accepting the public bridge: exactly two fully
visible non-contact objects are now the frozen rung before moving camera,
identifiable drag, and variable metric scale; variable set size,
identity/occlusion, analytic contact, observable material parameters, known
actions and counterfactual planning, and richer modalities/geometry remain
later. Model capacity grows only after a smaller structured rung demonstrably
plateaus.

### Passing seed-free RGB-D metric core

Simulator protocol `sphere_world_v7` exposes observable metric surface depth
as `[T, 1, H, W]`, with zero denoting no return. Exact ray--sphere intersection
selects one consistent nearest winner for depth, instance, visibility, and
RGB. The parameter-free metric measurement combines an RGB-derived
differentiable subpixel centre with differentiable bilinear depth, known sphere
radius, perspective radius correction, and the canonical camera. Labels,
simulator state, instance maps, and object IDs are not inputs.

The seed-free 18-case public-renderer grid passes with:

- maximum/RMSE position error `0.00613210 m`/`0.00336217 m`;
- maximum/RMSE centre error `0.0272064 px`/`0.00802947 px`; and
- finite centre/RGB/depth gradient norms
  `0.673917`/`0.0718314`/`6.92869`.

Invalid finite or extreme rows fail closed with finite zero outputs and zero
gradients; float16 and bfloat16 are rejected. Focused validation is
`29 passed`, and independent review passes. This proof consumed no episode seed
namespace and no protected data. It establishes only single-frame RGB-D metric
state: temporal execution, velocity and long-horizon qualification, and every
convergence claim remain pending. The source protocol itself is frozen below.

### Qualified standalone RGB-D temporal protocol — final consumed once

Specification 1.54 now freezes the standalone parameter-free temporal rung.
The exact config SHA-256 is
`5667cdb3603682b8d80a3e42793d25e36989269df1afacfa9b1028f2451101e9`, and
the canonical protocol payload SHA-256 before insertion of its self-reporting
digest field is
`4e334e9d7942ea3f2416c0a9f5ca8e327d1d0a1e9131074f20c051ebd3163ad7`.
The disjoint manifests are development `41000000--41000023`, selector
`42000000--42000023`, confirmation `43000000--43000023`, and final
`44000000--44000047`. All four were unopened at the specification-freeze
boundary. The v1 development and its invalidation are retained below; the
fresh v2 development and all three protected splits have since been consumed
under the qualified evidence contract.

The estimator has zero learned parameters, tensor buffers, optimizer state,
and optimizer updates. It independently measures all sixteen RGB-D frames,
fits anchor position/velocity with uniform differentiable exact free-motion
WLS, and uses `AnalyticKinematics` for
`0.1/0.25/0.5/1.0/2.0 s` queries. Confidence and validity cannot taper or
select temporal rows. Fixed-output VJPs must reach both RGB and depth for
anchor and every horizon state. The frozen RGB-only ablation must degrade,
while missing depth must invalidate the complete estimate without falling
back to RGB. OLS covariance is an i.i.d. residual diagnostic only, not a
calibrated posterior claim.

The first development run used clean commit `8e68035` and the declared
`41000000--41000023` seeds. All `82` reported scalar metrics were finite and
all frozen gates passed. Current
position/velocity RMSE was `0.00279934 m`/`0.00207092 m/s`; horizon position
RMSE was `0.00286018/0.00297530/0.00322108/0.00385564/0.00539692 m`;
perception and rollout times were `0.235061 s` and `0.00391524 s`. The report,
checkpoint, and manifest SHA-256 digests are
`9cbea9f25181769ee5b6a87b097e738a29cdb9b386c8018b3044f07d58aa03e2`,
`6acd88edd203cdebb2b0820bad388e06a4c610ea1c659ff9f8ea6d701ad28059`, and
`b92f1e3f12475986ebd2971ad2de70187432c0941caa0335ea53e82abc3c1d01`.

Audit found one source-integrity defect: qualification compared raw Python
tuple/list structures after JSON roundtrip. The repaired source compares their
canonical JSON SHA and includes a regression test. Although the numerical
development result is a historical conditional pass, the repair changes
source, so its report/checkpoint pair must not qualify protected access. A
fresh clean-source development rerun must use fresh v2 report/checkpoint paths
and be independently reviewed before qualification. That paragraph records the
historical post-repair boundary, not current status.

The qualified runtime source is
`df0235a92a81d3c1d2ba4e69e64d639562e3dfe8`. Fresh v2 development report and
checkpoint SHA-256 are
`4cf1657ee95645c8c647433a8be660520e9cdc1a5e6ac106d85bd24547b4e740` and
`fd663e5fa52dded8156a3178070966e3458d93a7b5a49dd5dcb2cc0d6278514e`.
The scientific metrics match v1; perception/rollout time is
`0.233866867 s`/`0.003934262 s`, and maximum RSS is `545804288` bytes.

Protected qualification passed exactly once in ledgered
selector -> confirmation -> final order. Qualification-report bytes,
canonical summary, and ledger SHA-256 are
`7e4cface087620f058ade4cc83ac5fd197685ba26c8f0afb5089d8f7e646fe0d`,
`7e9954ae34ce55b6923765de0c084d5075f238bd012554eeb44049a0db161658`, and
`9fc139291dfb34b10125321d06fdf06ab68ed65df32f62c273a95e5ca7aa7b8b`.
Development, selector, confirmation, and final manifest SHA-256 values are
`b92f1e3f12475986ebd2971ad2de70187432c0941caa0335ea53e82abc3c1d01`,
`56cb85d9a30129f6e7153075b7334fd4737986f1c9679650533d20e7e0763cf8`,
`eb61687ec0a8508a563b6e7c3dfb67b4393a6eaa9d35b2788eaa373d79e5df16`, and
`42e0320bf6f62b78c881951ec78486714b4432a1592fe2a5047027e2bbd0339f`.

| split | current position / velocity | horizon position RMSE at 0.1/0.25/0.5/1/2 s |
| --- | --- | --- |
| development v2 | 2.799 mm / 2.071 mm/s | 2.860 / 2.975 / 3.221 / 3.856 / 5.397 mm |
| selector | 3.073 mm / 1.991 mm/s | 3.159 / 3.306 / 3.589 / 4.254 / 5.774 mm |
| confirmation | 3.078 mm / 1.644 mm/s | 3.078 / 3.094 / 3.162 / 3.431 / 4.328 mm |
| final | 2.905 mm / 2.226 mm/s | 2.954 / 3.056 / 3.290 / 3.934 / 5.560 mm |

All splits have `82` finite metrics, all `103` frozen gate comparisons pass,
and failures, optimizer updates, and learned state are zero. Final two-second
error is `18.5%` of its gate and only `3%` above development. Across splits,
rollout is `3.80--3.95 ms`, perception is `232--237 ms`, maximum RSS is
`554 MB`, RGB/depth VJPs span `0.1156--3.6564`, and semigroup errors are at most
`2.384e-7 m`/`3.725e-9 m/s`. Baseline and RGB-only ablation gates all pass.
Final is consumed and must not be rerun.

OLS covariance remains diagnostic rather than calibrated uncertainty.
Artifacts are SHA-bound and tamper-evident but owner-writable, a nonblocking
operational caveat. The standalone rung is qualified; broader temporal/world-
model convergence is not claimed.

The standalone protocol does not itself claim the later public bridge. That
separate bridge has now passed its own development and exactly-once protected
qualification under the contract below.

### Qualified public `OnlineWorldModel` RGB-D bridge — final consumed

The one-slot causal bridge is implemented behind the ordinary observation,
`MeasurementSet`, `WorldBelief`, correction, checkpoint, evaluator, and demo
contracts. It accepts one composite batched `rgbd` packet per timestamp with
RGB `[B,3,H,W]`, depth `[B,1,H,W]`, batched intrinsics/extrinsics, explicit
image size, and a modality-qualified stream key. Independent source review is
complete. The accepted scope remains one sphere, fixed camera, known
free-motion physics, and no contact.

The exact configuration SHA-256 is
`c40b3438c7fd60646d356db3fe54050039912ace288d9db89620b626106993a3`.
The current seed-free canonical `bridge_protocol()` payload SHA-256, computed
before inserting the self-reporting digest, is
`e536b0d0b721042bff55501faf3445456219fcc987334b6ec1e892688ea560b2`.
The frozen disjoint manifests were:

- development `45000000--45000023`;
- selector `46000000--46000015`;
- confirmation `47000000--47000015`; and
- one-shot final `48000000--48000031`.

All were unopened at source freeze. They have since been consumed only in the
audited sequence recorded below.

The bridge has a single direct observable-position owner. Supported RGB-D
axes replace the filter mean with the raw metric measurement and use its
declared variance; no learned corrector or second position filter shares that
ownership. A persistent-object-ID-aligned sixteen-row raw metric history feeds
the uniform differentiable exact free-motion WLS fit and emits velocity-only
`DirectVelocityEvidence`. Parameter-free analytic dynamics alone owns
`0.10/0.25/0.50/1.00/2.00 s` rollout. Parameters, state-dict keys/bytes,
optimizer state, and optimizer updates are all zero.

Fixed-output per-frame VJPs reach both RGB and depth from current velocity and
every horizon. For an already-active persistent object, a well-formed frame
with missing depth, nonfinite or otherwise invalid depth in the sampled
measurement support, or no foreground appends an invalid causal row. In the
frozen sixteenth-frame full-window ablation, diagnostics are `sample_count: 16`
and `valid_count: 15`. The invalid measurement emits no valid/admissible
temporal fit or direct velocity evidence, correction, or birth; a finite
diagnostic fit with `fit_valid: false` is not admissible evidence. Before
birth, the same invalid frame advances runtime time but creates no object-
history row and no birth. Malformed packets, nonfinite RGB/calibration,
float16/bfloat16, unknown modalities, duplicate same-time stream keys, stale
temporal evidence, and invalid prepared propagation reject atomically across
belief, history, modality cache, runtime observation scheduler, diagnostics,
direct evidence, ingest count, and prepared-propagation consumption.

The evaluator and demo use the real composite packet and report truthful
`observation_modality: rgbd` and `rgb_only: false` metadata. The evaluator
identifies the fifteen-frame warmup and keeps its warmup-aware metrics distinct
from standalone qualification. Demo aggregate errors remain pooled across
warmup and post-warmup frames and are not a warmup-separated accuracy report.
Legacy RGB packet, cadence, metric, checkpoint-migration, and demo behavior
remains unchanged when RGB-D is disabled. Ordinary model checkpoints
intentionally omit live temporal histories/caches, so exact mid-history stream
resume is unsupported; causal observation replay rebuilds runtime state.

The first clean development on `ebda5a8` had all `175` reported scalar metrics
finite and passed all frozen gate checks, but
audit found a raw tuple/list protocol comparison that was representation-
sensitive after JSON roundtrip. It is rejected promotion evidence, no
protected data was accessed, and its report/checkpoint are archived under
ignored `runs/rgbd_online_bridge_v1/rejected_ebda5a8_json_protocol/` with
SHA-256
`2104ee87bcabdbd5312b4026a33e44e1de7d197e50215ec7f0bf0e0bb56992e3`
and
`38f4b2ef5addb98bb966360213d3bb36b43da606367fc60cd75d2ec487f1b866`.

Commit `526b5123e6385c575a5777936272330d28972b93` repairs comparison with
canonical protocol JSON and rejects tampering. Its audit binds runtime
fingerprint
`1eeaa176ad9be8886976910fe53028fb6de498adda73a2d20170f206b6134b40`
and worktree fingerprint
`90d0624a119e118e76b58061f7e5582dffc906f47d85cc4dde997b2f765bb07a`.
The focused gate is `421 passed in 62.72 s`; the complete repository gate is
`1209 passed, 16 skipped in 434.37 s`.

Fresh canonical development report/checkpoint SHA-256 are
`dce6f920da85fbf696b7ae8a7a91d9cbf7d9084176e51ad7c319f92a6efe4966`
(`22,346` bytes) and
`48249f1a5a0467b1da8c7bdb5ad9e909f8c502631ec2fbad832cb490a00c3099`
(`46,596` bytes); development-manifest SHA-256 is
`069eb3331543727c911a07cc9a1bb352f6185ac8ceac7fafca502c9d7fab6d80`.
All `175` reported scalars are finite and every gate passes. Development
current position/velocity is `3.068470 mm`/`2.191966 mm/s`; two-second
position/velocity is `5.609913 mm`/`1.983371 mm/s`; growth slope is
`1.270721 mm/s`; perception/rollout is `0.415134 s`/`3.575380 ms`; and full
batch-four persistent tensor state is `25,364` bytes.

Protected qualification report and ledger SHA-256 are
`7fd1829f663606910ac81990e4b633c63b1460dbc31dd24c71eedbd91b422908`
(`47,353` bytes) and
`cf6a10dd672aafbdd91c92871ae349fef0c549d865cc6532e6c42f7d9be14e32`
(`1,626` bytes). Initial and final empty state both hash to
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

Every split has `175/175` reported scalar metrics finite and no gate failures.
All required history VJPs reach
`16/16` frames, identity changes are zero, and RGB-only, missing-depth,
no-foreground, semigroup, memory, checkpoint, and zero-state gates pass. Final
persistent state is `25,364` bytes, maximum RSS is `708,853,760` bytes, and
semigroup error is at most `2.384186e-7 m`/`1.862645e-9 m/s`.

The ledger is complete and stopped after final. Final is consumed and must
never be rerun. Evidence is SHA-bound and atomically replaced, but remains
owner-writable rather than OS-enforced append-only storage. The audit did not
reinspect or rematerialize raw protected episodes.

This one-sphere, fixed-camera, contact-free public bridge is accepted and
merged to `main`. At that boundary no broader multi-object, association,
occlusion, contact, planning/task, learned-capacity, or additional-modality
convergence was claimed; the later exactly-two-visible acceptance is recorded
below. Each bounded rung must predeclare its changed capability and
per-object, association, event/contact, task-success, gradient, accuracy,
memory, and rollout gates. It may not alter this accepted rung or tune on the
consumed final.

## Historical freeze: exactly two visible non-contact objects

At its pre-access boundary, specification 1.56 froze architecture attempt 2.
It added exactly two fully visible, image-separated, non-contact, fixed-radius
spheres while retaining the accepted one-object behavior. The fixed-camera,
fixed-parameter RGB-D runtime uses parameter-free differentiable chromatic and
spatial two-slot geometry, a hard Hungarian branch only for discrete stable
identity, one direct metric-position owner, uniform differentiable WLS over
all sixteen frames for velocity, and analytic position/velocity rollout at
`0.1/0.25/0.5/1/2 s`. There are zero learned parameters and optimizer updates;
this is the cheap differentiable inductive-bias baseline, not a learned-
capacity result.

Current frozen config/protocol SHA-256 values are
`84e6f44b818bb9323a774bdba9492ef056e2a2747b93517fa38497ba83218bba` and
`42b9dca23fed303d5cee4641c8d8753977a872fc90d0b1086658d7f12b823ea0`.
Harness/test/runner SHA-256 values are
`198cac1c4d683e3c983f70c0106827aaf883636d4bd6454e94011c3975c1b64a`,
`d5dd3c18515589b4589e0179a68e29112d45987a513308df022cece5bf75e896`, and
`a8e6d9f51380eede3b6a94f085e9741f67883e2740c6203c16aec4a5dcfa1bc1`.
The eventual clean commit/runtime/worktree fingerprints were intentionally not
claimed at that boundary. Later documentation could change a worktree
fingerprint without changing those four frozen file hashes.
Seed-free validation passed the focused harness (`43 passed`), combined
accepted one-object/config/harness gate (`281 passed in 15.61 s`), complete
repository (`1275 passed, 16 skipped in 447.29 s`), Ruff lint and format, diff
integrity, and two independent audits.

The VJP contract is explicit: current position reaches frame 15 only (`1/16`)
with exactly zero non-anchor gradient; current velocity and every future
position and velocity reach `16/16` RGB and depth frames. B4 uses four distinct
audit scenes per split and requires exactly zero cross-scene coupling.

Development `49000000--49000031`, selector `50000000--50000023`, confirmation
`51000000--51000023`, and one-shot final `52000000--52000047` were all unopened.
Their predeclared manifest SHA-256 values are respectively
`5a47a1a4a1405ba4c2fc3bce0087131d98fabfceb899beb26c6b4ba824a130f8`,
`415bc33407a46b79d0a3a746a8f5b192e31cfd4f6a68b9764e9b9943b7e6d7fe`,
`14f7dc3b762e4f987acbedcece815abd1c262bc9da60322f7f054e2c4eb4b3b1`, and
`b7e8913e938e2f7ae7f937979a60279916ff1a06f071427bcce9f08b0e354e75`.
No episode, report, checkpoint, or ledger existed at that boundary.
Development had to wait for one clean committed source tree, run exactly once,
and receive independent audit. Only a pass could authorize exactly-once
selector -> confirmation -> final; any failure would stop without retuning or
opening a later split.

Evidence control was predeclared through fixed durable development/protected
ledgers, constructor-level single-use authorization, a restricted
`weights_only=True` empty-state checkpoint, and report creation before the
ledger was made terminal. The ledger/report files were to be hash-bound and
fresh-path protected but owner-writable rather than OS write-once.
Exact mid-history resume remains unsupported because live temporal histories
are not serialized. No occlusion, reappearance, variable object count,
contact, camera motion, variable physics, uncertainty calibration, task,
additional-modality, or learned-capacity claim is made.

## Accepted exactly-two-visible qualification — final consumed

The frozen tree was committed and pushed at
`3b781e653a0287b2aa926e7c0b969e9197d48e42`. Runtime/worktree fingerprints are
`810b237082ae99735527985c544dc28834b806489c555b464191c3b3e62520e7` and
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

| split | result SHA-256 | current position / velocity RMSE | 2 s position RMSE |
| --- | --- | --- | --- |
| development | `2eaefcf40b459414492e849d24bbf50fc4638294dedfe4b5350fc011b599cfa2` | `1.9029872e-5 m` / `3.1932373e-5 m/s` | `7.4638663e-5 m` |
| selector | `ede4e91e708645a761065ff43993e1df05800422673d5be1b1f77b2bd3c001ce` | `1.6885011e-5 m` / `3.1409136e-5 m/s` | `7.0948345e-5 m` |
| confirmation | `204e5f5a65c73b721e038cf50ef732068ba4a901a68c78c8cb8d7f79a60b4ad8` | `1.6776625e-5 m` / `3.2594633e-5 m/s` | `7.2466125e-5 m` |
| final | `7b9ba4df3a2595c9a671322f6650ed170a0b3cfbd092d9bf46612abbe9db6dae` | `1.7838631e-5 m` / `3.1881889e-5 m/s` | `7.1961138e-5 m` |

Every split has `396` finite gated metrics and no failures. Final two-second
velocity RMSE is `2.8845629e-5 m/s`; worst final per-axis position/velocity
RMSE is `1.020557e-4 m`/`4.503076e-5 m/s`. Identity coverage is one with zero
switches, mismatches, or ambiguities; minimum Hungarian margin is about
`199.96`. Visibility is one, event count is zero, and minimum silhouette/world
gap is `9.534 px`/`0.75985 m`.

Every split preserves anchor-only `1/16` current-position VJP reach with zero
non-anchor gradient, `16/16` temporal-output reach, four distinct B4 scenes,
and zero cross-scene coupling. Final minimum total/minimum temporal-frame/
maximum VJP L1 is `2.0798e-5`/`5.7799e-8`/`4.2197`.
Final perception/five-query rollout is `0.352510 s`/`0.00359524 s`; runtime
state is `28,512` bytes and maximum RSS is `579,817,472` bytes. Learned,
buffer/model-state, optimizer/scheduler/RNG, and update counts are zero.

The development ledger is complete and passed; the protected ledger is
complete in exact selector -> confirmation -> final order. Final is consumed
and must not be rerun. The
audited directory has exactly five single-link regular files with no temporary,
alias, or second-attempt artifact. Independent final audit passes without
reopening raw protected episodes. Evidence remains owner-writable and
tamper-evident, not OS-enforced WORM storage.

The accepted claim is exactly two fully visible, image-separated,
fixed-radius, non-contact spheres under fixed-camera free motion. It does not
cover partial visibility, missed-observation recovery, contact, variable count,
learned capacity, or general convergence. After merge, the next rung could add
only bounded partial visibility and missed-observation recovery under a new
freeze; no consumed-final tuning was allowed. That merge is complete:
GitHub `main` contains the reviewed net tree through
`1e951520e5a2bf06c1932f64b8334e552247de82`.

## Historical specification-1.57 pre-development freeze

Specification 1.57 preserved that accepted two-visible result and froze
architecture attempt 1 for the next rung. At that boundary the scope remained
exactly two fixed-radius `0.21 m`, fixed-camera, non-contact spheres with known zero
gravity and `0.05` drag, public composite RGB-D input, zero learned or
optimizer state, and the parameter-free analytic five-horizon rollout. The
only new conditions were bounded partial visibility that never became full
occlusion and exactly one isolated target-local missing-depth observation at
frame 15 or 16 in the declared one-miss strata.

Frames `0--17` were ingested and the sixteen-row history used frames `2--17`,
with frame 17 required valid. No-miss/co-object histories retained 16 valid
rows; the missed target had exactly 15. The miss changed no RGB, calibration,
co-object depth, or non-target depth pixel. It emitted no target measurement or
velocity evidence, triggered exactly one filter-owned `0.08 +/- 1e-6`
miss-variance increment, and had to recover the same persistent identity on
the next frame. The required trace was `0 -> 1 -> 0` missed steps, all-`FREE`,
with no switch, mismatch, ambiguity, birth, death, false miss, contact, or
event.

Frozen config, harness, runner, test, and pre-self-hash protocol SHA-256 values
were respectively
`7d563382a8f4b6e301ac30510152f1b1409da32248aacf15dff460ea71d29e2c`,
`99084d9fb421faa8dbe7ef20f7a88ee5e196cce498586c0fae2b92eebddc36d4`,
`c97f20638c876045cb25adfe23d39db6daed749e42ab5eed1dea6aacac8dd90f`,
`e712f9b6ee1cd8775f8f8a1d07ee0844fe1ac1e8ac73a2a2233c9a231cce892e`,
and
`e178d572a238c17eaa4c23f1b0942e2c4e70103a73af3ab51736fffe36b0d8fd`.
Simulator protocol was and remains `sphere_world_v7`.

| split | exact seed range | pure manifest SHA-256 |
| --- | --- | --- |
| development | `53000000--53000031` | `ca1fb17e87df5216c4429342f74dcccd2c31b11b8d48bb3c76eee27e139cf391` |
| selector | `54000000--54000023` | `1b1e6ef6938705bcc7e2a66ad5ee4622860c9ea9ec3e6c19c86e8a8534209b28` |
| confirmation | `55000000--55000023` | `72d7c922029d300e3d28409bcb55a843633caac10b482f680ae769a442739e9f` |
| final | `56000000--56000047` | `70b60f48769a26c5587febf778443fd38f5814a39e80ec7da1c98dea9c389ded` |

Pure scene-parameter signature-list SHA-256 values were development
`f22a2e26df99edda751d13c383733c447139afe4de840bae64b3e03758155baf`,
selector `85bf300a1af8547746663a9b10403fa8b3d726533d0f68955c2cad5ecf3a4d75`,
confirmation
`e6319849bb4b4974ceeb6752eadf7235f8e82e47440f0e2b7d1be75191600931`,
and final
`575e4af1694825e40c780c2a64c783232b013bf8432c8fbe76d095180f0c9d5f`;
they were unique across splits without episode construction.

The historical frozen protocol had 98 authoritative gate fields and required
exactly 2,167 finite scalar metrics per eventual split. It included pooled/
stratum/miss/co-object accuracy, exact target-region RGB/depth VJPs with zero
gradient at the scheduled miss and outside frames `2--17`, exact zero cross-scene
coupling, pure scene signatures, global all-`FREE`/no-spurious-miss tracing,
resources, zero-state ownership, exact ledgers, and a canonical five-artifact
inventory. Seed-free combined validation was `436 passed in 60.26s`; Ruff
lint, Ruff format-check, and diff integrity were clean, and two independent
audits passed. The full exact current-byte repository result was:
`1398 passed, 16 skipped in 487.93s (0:08:07)`.

At this historical freeze, every development/selector/confirmation/final
namespace was unopened and no episode, materialized manifest, report,
checkpoint, ledger, result, or other evidence artifact existed. The required
next action was to commit the exact clean source and documentation, bind the
complete repository gate, execute the sole development run, and obtain
independent digest review. That attempt later failed as recorded below, so its
selector -> confirmation -> final sequence never opened. This was not
full-occlusion/reappearance, contact,
variable-count, history-capacity, learned-capacity, extra-modality,
task/planning, or general-convergence evidence.

## Terminal architecture-attempt-1 development failure

Architecture attempt 1 was committed as the clean source tree
`7e67823667769e47bad3678207f2c01bd3edbfe4` and consumed its one development
authorization. The immutable `13,948`-byte report SHA-256 is
`7c08c794690a10d46100b8d17ee448e3a83960d265ec7859bb91cd6d2ac9ca9d`;
the `1,110`-byte development-ledger SHA-256 is
`e4993abefefe07e0b0fb57a65769fa270012524d62c8ebab4b7db0251979aab4`.
The evidence binds v1 config/protocol/development-manifest SHA-256
`7d563382a8f4b6e301ac30510152f1b1409da32248aacf15dff460ea71d29e2c`,
`e178d572a238c17eaa4c23f1b0942e2c4e70103a73af3ab51736fffe36b0d8fd`,
and
`ca1fb17e87df5216c4429342f74dcccd2c31b11b8d48bb3c76eee27e139cf391`,
plus runtime/worktree fingerprints
`2345bcf6d785cd864301dbcdcb23cc8f7287f1815615fd1e30e6f635084f12c3`
and `0d44cabadce831238fe1c8c1cda450677b62f20af3fcf9a411fa4ef621b1842f`.

Seed `53000000` completed the private constructor. Seed `53000001` rendered
`58` frames and then failed renderer visibility preflight at frame `4`: the
mild rear sphere had support `20` and visible support `15`, so exact raster
visibility was `0.75 < 0.80` despite continuous visibility `0.826827`. The
failure preceded model, collate, and runtime. Cursor `2` is inferable only from
the in-memory execution and was never durably persisted. Seeds `>=53000002`
were untouched. No checkpoint exists. The protected 54m--56m namespaces never
opened and are permanently unused. The live v1 inventory is exactly two
single-link regular files, the report and ledger, with no alias or temporary
artifact. Attempt 1 is terminally consumed.

## Specification-1.58 terminal attempt-2 source freeze

Architecture attempt 2 of the maximum two is frozen at
`runs/rgbd_partial_visibility_recovery_v2/`. It retains the public history,
miss-isolation, and recovery semantics above. The accepted 1.56 base and
failed 1.57 attempt remain immutable. Simulator metadata remains
`sphere_world_v7`.
The replacement constructor uses one finite table of `16` rational primitives
crossed with the `8` exact `D4` transforms, yielding `128` unique physical
cells. The raw constructor/evaluator are private; seed mapping is public and
deterministic. Float32 world evolution follows the exact public solver over a
`342`-substep recurrence, and exact renderer support—not continuous visibility
alone—governs preflight.

World/renderer trace SHA-256 values are
`32b34e716ec639cabdd5d36f1c0d30fa17b187546bb5653e4fa7d0a9d6af65d4`
and `4362f06929f8e8958c1f12e8d2077dded6f8dda3bfdb99eed425899bb289f412`.
Table/absolute-table/ordered-state/state-set/unordered-geometry SHA-256 values
are
`c3f17e805de234fecb1f1928b47e8fd2127d608447e7b1e87df9a2ec970ce3aa`,
`f86f218317d656c16f4c85e5b4a75b2e52094724316a3132b0a6e44715bec86e`,
`bc3e6349fc0d5effecbb53920a9c4224203067f05306330723f8c75dd9f35c57`,
`96a53595bf7d21b84fed772baef4b754b6e777b7560a8083d303814fa5f611b5`,
and `27a8dabb2d9936e635cde5b2155fffa5eddb89679b477175119917627772cafa`.

The proof has explicit margins: discriminant
`5.20199537e-5 >= 5e-5`, overlap depth `0.831737 m >= 0.8 m`, projected drift
`1.144409e-5 px <= 2e-5 px`, D4 conjugacy
`2.861023e-6 m <= 4e-6 m`, speed `0.0406846--0.0520633 m/s`, physical gap
`0.616238 m`, world-boundary clearance `0.211665 m`, image clearance
`19.4474 px`, separated excess `1.44462 px`, partial margin `0.421646 px`,
actual visibility margin `0.05`, and support counts `18/14/14`. The
one-pixel hypothetical clearance is exactly `0.0` under an inclusive gate and
is not positive slack.

Current config/harness/runner/qualification-test/config-test/protocol SHA-256
values are
`b18f787987394f77771dbf31dae1642bd042b81e64b02a3e93b8cd048dd3416b`,
`859dedf68031ee66cec1334d2fc094078bc2aacf0deac4388c53337033b63519`,
`a16c0712b611ebe64dd5052efde3f73e3c5aa18f1b1c5f825f571c2674e598c0`,
`f4d35320f484484429cdfadb9f3faed6ad5c1ad85492d6ffaa378a7076955714`,
`8f4e14c7ccff5c6af4d820c555956ce12b66854b1aef7ee3fb5ddbaad7abd40a`,
and `5f049f060f6e8a9682d9413e6bc2d8f9f228f6e2aee67cde16f98d234cac8a3b`.
The exact tracked v1 report/ledger fixtures match the two live files.

| split | exact seeds | manifest SHA-256 | signature-list SHA-256 |
| --- | --- | --- | --- |
| development | `57000000--57000031` | `ded3d75a7d248e3f9746b03b0cf249f32739208713c4287c45deb5eefd11f8e2` | `8426ea4d0a7e1d507c5d7fc825afa8864ee694a04df622cba955b92ffd4350c0` |
| selector | `58000000--58000023` | `effa598aa07a44c100da115f71828e00754f181729063899353d22b551f7227a` | `d421862763a3e0bc0af042fd81704c836c2123ad0fa260130e791cb250c0b2c7` |
| confirmation | `59000000--59000023` | `9240a1dd465574de8ac032e318f3cee618909ed6a5b3e91c5fd8c87bad146cec` | `261f975fcd46795ff9f56c94857de69942ea047455f65cc0341bdc515cc76af5` |
| final | `60000000--60000047` | `17fdd50896729b981357960ea0db74ef19e059e21bc8d8e41a7048cf237200a6` | `1837d40a35ddba88e3a91f74c5b2c398aa01675ad8e84efa2fe660bbf49e34a2` |

All v2 namespaces are unopened and no v2 artifact exists. Direct live-v1
guarding, exact tracked fixtures, a private raw constructor/evaluator, and an
exact canonical ledger-minted capability prevent unauthorized construction or
v1 reuse. The runtime has no oracle. Exactly five canonical v2 files are
allowed, and checkpoint loading is restricted to `weights_only=True`.

The canonical gate passes `315 passed in 345.27 s`; the public-solver proof
passes `1 passed in 244.52 s`; the full repository suite passes
`1407 passed, 16 skipped in 816.14 s`; and two independent final audits report
`PASS`. This is source-integrity and security evidence only. The tree is ready
for commit and push, not development or qualification. Development may begin
only after a clean tree proves `HEAD == upstream`. If it fails, the rung ends
with no attempt 3.

## Validation state

The pre-failure clean source at commit `8889818619121351d342490786331e854364532c`
passed the complete repository gate:
`1075 passed, 16` expected inactive-MPS skips in `425.70 s`. Ruff lint passes;
all `224` Python files were formatted; compileall over production,
tests, scripts, and entry points passed; the explicit specification-version
contract passed; and `git diff --check` was clean. This is preserved
pre-failure source-integrity evidence, not evidence for the post-deletion/core
tree. The complete post-deletion/core gate passes `1091` tests with `16`
expected inactive-device skips in `418.49 s`.

The qualified RGB-D temporal runtime remains bound to the repaired-source
`104 passed` focused gate and complete repository gate
`1130 passed, 16 skipped in 414.82 s`. The accepted public bridge has the
canonical-comparator focused gate `421 passed in 62.72 s` and complete
repository gate `1209 passed, 16 skipped in 434.37 s`. The older `1091` result
remains the historical single-frame-core boundary, and both raw-comparator v1
development artifacts remain non-qualifying evidence.

The accepted two-visible-object source retains `43 passed` focused,
`281 passed in 15.61 s` combined, and
`1275 passed, 16 skipped in 447.29 s` complete, with Ruff/format/diff clean.
The exact independent qualification audit also passes.

The historical specification-1.57 partial-visibility/isolated-miss freeze
passed the seed-free combined source gate at `436 passed in 60.26s`, with
Ruff/format/diff clean and two independent audits. Its full exact current-byte
repository result was
`1398 passed, 16 skipped in 487.93s (0:08:07)`. Architecture attempt 1 later
failed during renderer preflight as recorded above; no model, collate,
runtime, checkpoint, or protected access followed.

The specification-1.58 attempt-2 frozen tree passes the canonical gate at
`315 passed in 345.27 s`, the independent public-solver proof at
`1 passed in 244.52 s`, and the full repository gate at
`1407 passed, 16 skipped in 816.14 s`. Two independent final audits report
`PASS`. This exact source freeze remains uncommitted/unpushed and has no v2
episode or evidence result.

No accepted claim exists beyond the exactly-two-visible non-contact RGB-D
family. Attempt 1 is a constructor/preflight failure, not recovery evidence.
Attempt 2 is frozen but not committed, pushed, developed, qualified, or
accepted; full occlusion, reappearance, contact, variable count, additional
modality, planning, learned capacity, and general convergence remain
unqualified.
