# Project status

## Active generalization program — 2026-08-30

The pre-generalization public base was commit
`c16acc99ef13757fc8f88528bfd0d66db4a2f4fd`; the cleaned generalization
foundation is GitHub `main` commit
`08ae63adc5ade2e5061f54539fc7a25564c8c8d6`, and the accepted public RGB-D
bridge is merged at `3eed0b71e6f18c7036bf376c075493a89d5fdc9f`. Broad heterogeneous training
remains paused. The active contract is specification 1.57. The accepted base
now comprises the specification-1.51 differentiable one-sphere unit, the
qualified standalone two-second RGB-D rung, and its qualified public one-slot
`OnlineWorldModel` bridge. The accepted branch additionally contains the
exactly-two-visible RGB-D qualification recorded below—not any older campaign
checkpoint.

The accepted base now also includes the completed specification-1.57
known-calibrated orbital-camera qualification recorded below. All four splits
passed in order and final is consumed. This is a narrow parameter-free result
for one certified orbit, not a claim for general camera motion, camera-pose
learning, occlusion, contact, variable count, or learned capacity.
The reviewed net tree is merged and published to GitHub `main` through
acceptance commit `00a712d640cdb828f24a194817443daa57e6df65`. Final remains
consumed; any next rung still requires a genuinely new pre-access protocol.

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

### Terminal result: variable metric radius

The variable-radius core, scene, source implementation, and their audits are
green, but neither permitted development attempt produced a valid development
result:

- attempt 1, bound to source commit prefix `db669b`, terminated in the
  no-gradient harness path. Its independently audited, internally byte-bound
  development report is SHA-256
  `7f194a41bd5e64328f0a57d8142aad8a81f01d2b449386bb05939fb3ed49b142`
  (`66,758` bytes), and its ledger is
  `aec6c9500d3cd8ca6a152b8107578b2b441a544dca605fe7f6ae59a61f0d021e`
  (`10,248` bytes). It created no checkpoint or protected artifact. Its own
  frozen strict reread validator rejects the bundle because of tuple/list JSON
  normalization and mutable directory-link-count binding defects;
- attempt 2, bound to source commit prefix `dcb815`, terminated with a
  `ValueError` because the fixed-radius public bridge was paired with the
  preserved variable simulator-radius range. Its source-bound, fail-closed
  report is SHA-256
  `a6efb2873c7248cc5bc9a010fb26b30615f544469810895aac9943cd14770fb7`
  (`67,835` bytes), and its ledger is
  `3d05e68af3c6fe1b4c4abc09bd5998fe9c83734bb3658861bcda14aca5934cdf`
  (`10,255` bytes). The ledger record self-hash has prefix `f9f6bca1`;
  generation 3 records active batch `[0, 1, 2, 3]`, next ordinal `0`, and zero
  completed batches. It also created no checkpoint or protected artifact. The
  v2 frozen strict report, ledger, publication-surface, directory-binding, and
  prior-attempt validators all pass.

Source-order audit shows that attempt 2 computed the four nominal rows and the
alternate-prior control in memory, then failed in the legacy control before
resource aggregation, evidence finalization, or `complete_batch`; therefore no
durable scientific metrics exist. A separate PyTorch warning converted a
graph-live owner-error diagnostic to a Python scalar. It was not the terminal
exception but remains an unqualified diagnostic limitation, with no repair or
replay after closure.

The frozen v2 protocol permits at most two architecture attempts. The family
is therefore closed: there is no retry or v3, no protected access, and no
qualification. This is NOT a successful qualification or acceptance. Public
`threshold_design_only` evidence was threshold-design evidence only and never
formal acceptance. Radius variance remains an engineering-fixed,
uncalibrated value; no calibrated radius-uncertainty claim is made. The two
evidence bundles remain ignored and local. Specification 1.57 and the accepted
known-orbital-camera result remain the highest accepted contract.

### Terminal result: identifiable per-object drag

Identifiable drag is a separate failed rung, not a variable-radius attempt.
Its exact source freeze is
`0e283d841281fbf98842c9969f02a026a5489dce`; the terminal branch tip is
`8516034760d0aa6c98c0d0e065bda4838f902dcd`. Both remain on dedicated pushed branch
`agent/rgbd-identifiable-drag-rung-1`, unmerged to `main`. The source,
certificate, and security reviews were green; the pre-documentation full gate
was `1490 passed, 16 skipped`. Frozen protocol and certificate
SHA-256 values are
`d4abaf22e775afc6f807b268f08aa68ae44210a40192b1b05653957720f48c70`
and
`588c8fe2e2baa38dcb097a012b5ec6517b3ce9733a7c8d068e71c98a1c5f5f9e`.
Those are implementation and source-integrity results only.

The sole and maximum development attempt, attempt 1 of 1, materialized four
ordinals `[0, 1, 2, 3]`. Generic collation then failed closed with the exact
error `ValueError: metadata tuple differs across batch at metadata.albedo`.
The ledger records zero completed ordinals, zero completed batches, and zero
completed evidence. The failure occurred before model construction, reset,
runtime ingest, fitting, calibration, VJP evaluation, or checkpoint creation;
there is no scientific accuracy result.

The terminal report is SHA-256
`b64d0ce512e223d03831448ce4c54196abf4f5c660e03329ee029d62dd53e307`
at `50,872` bytes. The terminal ledger is SHA-256
`95d929fc15b365b01689037ec10b954e60d1166c7aaa7dc67c80eec01ff1b694`
at `2,804` bytes. Selector, confirmation, and final stayed unopened; no
protected result, qualification, or acceptance exists. Strict duplicate-free
schema and report-ledger cross-binding audit passes. The family is
permanently closed and must not be repaired, renamed, retried, or selected
again. Specification 1.57 remains the active and highest accepted contract.
The drag branch's specification identifiers 1.58 and 1.59 remain branch-local
historical provenance; a future active source freeze must not reuse them.

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
superseded that ordering by accepting exactly two fully visible non-contact
objects; specification 1.57 later accepted their known calibrated
camera-motion extension. The subsequent variable-radius family exhausted both
permitted development attempts without qualification and is closed as recorded
above. Variable set size, identity/occlusion, analytic contact, observable
material parameters, known actions and counterfactual planning, and richer
modalities/geometry remain later. Model capacity grows only after a smaller
structured rung demonstrably plateaus.

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

The then-accepted claim was exactly two fully visible, image-separated,
fixed-radius, non-contact spheres under fixed-camera free motion. That merge is
complete: GitHub `main` contains the reviewed net tree through
`1e951520e5a2bf06c1932f64b8334e552247de82`. Specification 1.57 subsequently
added and has now accepted one known calibrated orbital-camera family. The
failed partial-visibility family remains closed and must not be retried.

## Historical two-visible orbital-camera source freeze

At its pre-access boundary, specification 1.57 kept exactly two fully visible,
image-separated, non-contact fixed-radius spheres, complete RGB-D, gravity
`0`, and drag `0.05`. The only new capability was a known calibrated orbital
camera; the runtime consumed time-aligned `world_from_camera` and performed no
pose estimation. Sixteen physical primitives crossed eight camera strata—four
phases by two directions—for 128 joint scenes. Learned parameters,
learned/model state, optimizer state, and updates remained zero.

The frozen byte/payload bindings are:

- config `a9c348ea54b168ec78780d59d3b3eb066344d3a7551464b9aad1e5b9ac6d6cbd`;
- harness `02e75b325bdf7bad310f8973a786a396b8762104261702b299a9f8103748e569`;
- runner `11bee2e4d05f83caaf9dbed6ca2a54d4c11b7c70e4bf8e1747b261b8518ef192`;
- harness tests `d08c7bb4a1ba998a51dc2f0ddb1946596a5a299ed236cdf6a91b5711e2d0a1af`;
- canonical protocol `7146befc869ea5f975177dd1c2da4691026439e1d36d84415aa23f696e61ef65`;
  and
- seed-free certificate `7832ddb49081292d0f50a5eb63edb38fefb49d136d7e1757c73d9c658e42a36f`.

The exact per-split schema contains 685 finite floats. The certificate binds
all 16 physical trajectories and 128 camera appearances, full visibility,
silhouette/boundary separation, non-contact public physics, non-degenerate
orbital motion, and exact calibration. An explicit negative control replaces
history-frame `world_from_camera` values `1--15` with the stale frame-0
transform. Correct calibration must beat it for current position, current
velocity, and two-second position. Fixed-output VJPs cover RGB, depth, and
`world_from_camera`; current position remains anchor-only, every temporal
output reaches all sixteen frames, and cross-scene coupling is zero. Resource
gates include one CPU thread, `3.0 s` perception, `0.075 s` five-query rollout,
`65,536` persistent bytes, `2.5 GB` maximum RSS, and `1.0 GB` RSS growth.

| split | exact seeds | manifest SHA-256 |
| --- | --- | --- |
| development | `61000000--61000031` | `eb558805c2974302c33abef4531e142bb60e8f20045d8530330838223a6899a0` |
| selector | `62000000--62000023` | `c97fff97459ee9962b972cb7905887c2b2ed6eb5a1837d908f1512ce77e6d97f` |
| confirmation | `63000000--63000023` | `b47f03633732fc2986939e71007a0a79b12db2b42f0b5261b4ebd2d0a304f544` |
| final | `64000000--64000047` | `82927d192b53f2e4af11491f53039c145acfd8e0401a3e2b0b1e974591ee4174` |

At that boundary all four namespaces were unopened and the fixed
`runs/rgbd_two_visible_orbital_camera_v1/` directory was absent. No manifest
artifact, report, checkpoint, ledger, result, development evidence, or
accuracy evidence existed. Its eventual exact files were
`development_report.json`, `development_model.pt`,
`development_attempt_1_access.json`, `qualification_report.json`, and
`qualification_attempt_1_access.json`. Fresh-path, symlink/link-count,
inventory, atomic-write, and stable-read checks applied. Development could
execute once only after clean `HEAD` equalled its configured published
upstream with zero ahead/behind count. Protected access then required external
review of the exact checkpoint, development-report, and development-ledger
hashes before an exactly-once selector -> confirmation -> final ledger.

The source passes the moving-camera file (`26 passed in 128.43 s`) and
accepted/configuration regressions (`254 passed in 6.83 s`). The exact current
specification-1.57 tree passes the full repository gate
(`1302 passed, 16 skipped in 594.59 s (0:09:54)`). Two independent
science/security audits pass. These are source/integrity checks only; no
episode accuracy may be inferred from them.

## Accepted two-visible orbital-camera qualification — final consumed

The exact frozen source is commit
`c15afd6d57963b24bb98c5171462ff927e7c72fd`, with local upstream `0/0`,
worktree fingerprint
`0a5acfc54a5af482643b0c1037cf566a700e6122d2e6b51f7f4ad713ff652d2e`,
and runtime fingerprint
`bec3ca667fa464a3bbe82a83c14ffa924920ca367f14b6d9036ce52af041b83b`.
Development passed and the external three-hash review completed before the
protected ledger admitted selector -> confirmation -> final.

| artifact | SHA-256 | bytes |
| --- | --- | ---: |
| development report | `56d7e32c461d9b5e3fbca5e2e11e015662cd08c3d60dfb4807e75cbcb7f8e37b` | `88,743` |
| development ledger | `3f9d5c9cf88ae7e40517337799e270d02493e99ed58eaec24884e276dcec5ddf` | `1,544` |
| checkpoint | `c473bb6d5f453c786c681509350d66364e1f1c61a2656a7c35354ab806da1a25` | `78,573` |
| qualification report | `6daf2dea453db7c3a32b7950c8f31201ccc3fc32b9da1b14d8cc97dbd46ee0ad` | `202,540` |
| qualification ledger | `2aeb1c0194332004350c98628210d42724e31ece16614a210e2a84d6640b2719` | `2,293` |

The exact terminal inventory contains these five single-link regular files and
no temporary or extra entry. Split result SHA-256 values are development
`555871b24bfb764712d8dcae8473d5a9ad4c0ec6e9f02ffd42b2063af3cd7bc2`,
selector
`fcfd1b39393a8e41d0b112244b7e5ca4fe3c0b2e4e63b4cd729659781198e9d6`,
confirmation
`c3d644786d308a03d619eaf2a4d954bc216b1daf8655a9217f09e372ab27cd0b`,
and final
`b8ae823e961a981360717be273fe10d1ff5f9ce3bcbd6c396ba78fd5fdf0a4bf`.
Every split has exactly `685/685` finite float metrics, all `686` constraints
pass, and `failures: []`.

| split | current position / velocity RMSE | stale current position / velocity / 2 s position RMSE |
| --- | --- | --- |
| development | `1.5474954e-5 m` / `1.9066727e-5 m/s` | `0.05323550 m` / `0.069687995 m/s` / `0.185869285 m` |
| selector | `1.5932185e-5 m` / `1.8599896e-5 m/s` | `0.053338622 m` / `0.069820234 m/s` / `0.186224087 m` |
| confirmation | `1.6963295e-5 m` / `1.9596576e-5 m/s` | `0.053142687 m` / `0.069565501 m/s` / `0.185543335 m` |
| final | `1.7444936e-5 m` / `1.8386924e-5 m/s` | `0.053232131 m` / `0.069683790 m/s` / `0.185857911 m` |

All eight camera strata pass. Minimum total VJP L1 across the four splits is
`2.366e-5/2.259e-5/3.048e-5/2.502e-5`; minimum temporal-frame VJP L1 is
`6.546e-8/7.199e-8/3.345e-8/5.819e-8`. Cross-scene, non-anchor, and
homogeneous-row gradients are exactly zero. Identity/history, camera
calibration, public physics, and the frozen certificate remain exact.

Perception spans `0.3452--0.3944 s`, five-query rollout spans
`0.003469--0.003704 s`, persistent state is `28,512` bytes, and RSS remains
below `578 MB`. Parameter/model/optimizer state and updates are zero; initial
and final model state are the same empty digest
`4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.
The full source gate is `1302 passed, 16 skipped in 594.59 s (0:09:54)` and
two independent qualification audits pass.

The terminal evidence is complete/passed and stops after `final_test`; every
access was started and passed. Final is consumed. Integrity limitations remain
explicit: the terminal snapshot is not an append-only signed history;
exactly-once is inferred from final state plus committed control flow; ignored
artifacts are unsigned/unlogged and filesystem-deletable; external hashes have
no reviewer identity; upstream equality used a local tracking ref without a
fresh network fetch; and raw protected episodes were not rederived or audited.

Acceptance is limited to this certified known orbit with exactly two fully
visible separated non-contact spheres, complete RGB-D, gravity `0`, and drag
`0.05`. It is not general moving-camera, pose-learning, occlusion, recovery,
contact, variable-count, learned-capacity, or general-convergence evidence.
The failed partial-visibility, variable-radius, and identifiable-drag families
stay closed. The next active rung must be a genuinely new capability frozen
before access and must exclude all three families.

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

The specification-1.57 orbital-camera implementation's source gates are
`26 passed in 128.43 s`, `254 passed in 6.83 s`, and exact-current-tree
`1302 passed, 16 skipped in 594.59 s (0:09:54)`; two independent
qualification audits pass. Development and protected qualification are
complete, and final is consumed.

The highest accepted claim is now the certified exactly-two-visible,
non-contact, known-orbital-camera RGB-D family. General moving-camera behavior,
unknown or learned pose, contact, partial visibility, misses/recovery, variable
count, additional modality, planning, and general convergence remain
unqualified. The failed partial-visibility and variable-radius families must
not be retried. The identifiable-drag source remains on dedicated pushed branch
`agent/rgbd-identifiable-drag-rung-1`, unmerged to `main`; that family is also
terminal and must not be selected again.
