# Project status

## Active generalization program — 2026-08-27

The pre-generalization public base was commit
`c16acc99ef13757fc8f88528bfd0d66db4a2f4fd`; the cleaned generalization
foundation is GitHub `main` commit
`08ae63adc5ade2e5061f54539fc7a25564c8c8d6`. Broad heterogeneous training
remains paused. The active contract is specification 1.55; the accepted
convergence base remains the specification-1.51 differentiable one-sphere
RGB-to-state-to-rollout unit, not any older campaign checkpoint.

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
it now contains only the later ignored RGB-D temporal v1 historical development
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
below. After the temporal rung passes, scaling remains ordered: public `OnlineWorldModel`
integration; moving camera; identifiable drag; variable metric scale; two
non-contact objects; variable set size; identity/occlusion; analytic contact;
observable material parameters; known actions and counterfactual planning;
then richer modalities/geometry. Model capacity grows only after a smaller
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

The standalone protocol does not claim the later public bridge. That bridge is
now implemented and frozen under the separate pre-development contract below;
no model capacity or scene complexity should be added before it reproduces the
qualified accuracy, latency, and active-object invalid-row/pre-birth no-history
behavior without RGB fallback.

### Frozen public `OnlineWorldModel` RGB-D bridge — development unopened

The one-slot causal bridge is implemented behind the ordinary observation,
`MeasurementSet`, `WorldBelief`, correction, checkpoint, evaluator, and demo
contracts. It accepts one composite batched `rgbd` packet per timestamp with
RGB `[B,3,H,W]`, depth `[B,1,H,W]`, batched intrinsics/extrinsics, explicit
image size, and a modality-qualified stream key. Independent source review is
complete. This is frozen pre-development implementation evidence, not an
accuracy or convergence result.

The exact configuration SHA-256 is
`c40b3438c7fd60646d356db3fe54050039912ace288d9db89620b626106993a3`.
The current seed-free canonical `bridge_protocol()` payload SHA-256, computed
before inserting the self-reporting digest, is
`e536b0d0b721042bff55501faf3445456219fcc987334b6ec1e892688ea560b2`.
The frozen disjoint manifests are:

- development `45000000--45000023`;
- selector `46000000--46000015`;
- confirmation `47000000--47000015`; and
- one-shot final `48000000--48000031`.

All are unopened. No bridge episode, development report, protected ledger, or
protected result exists yet.

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

The final current-byte targeted gate is `419 passed in 61.33 s`, and the
complete repository gate is `1207 passed, 16 skipped in 431.10 s`. Full
batch-four persistent runtime tensor state, counted recursively by unique
underlying storage, is `25,364` bytes against the `32,768`-byte gate. The next
step is a clean commit/push, then exactly one development reproduction to fresh
artifacts and independent audit. Only a pass may open selector, then
confirmation, then final exactly once. Scaling remains closed until that
sequence passes.

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
`1130 passed, 16 skipped in 414.82 s`. The integrated public bridge has the
separate current-byte targeted gate `419 passed in 61.33 s` and complete
repository gate `1207 passed, 16 skipped in 431.10 s`. The older `1091` result
remains the historical single-frame-core boundary, and the v1 development
metrics remain non-qualifying evidence.

No accepted broader multi-object, contact, additional-modality, or planning
convergence claim exists beyond the standalone two-second RGB-D rung.
