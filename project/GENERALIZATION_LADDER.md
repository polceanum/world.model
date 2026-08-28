# Generalization ladder

## Purpose

Scale the qualified differentiable RGB-to-state-to-rollout core into the
general online world model described by `PROJECT_SPEC.md` without returning to
uninterpretable multi-day campaigns. Each rung adds one independently
measurable source of complexity, preserves every accepted lower-rung gate, and
stops after at most two architecture attempts if the new capability does not
converge.

The runtime consumes observable modalities, calibration, timestamps, known
actions, and explicitly declared priors. Simulator truth is limited to labels,
metrics, and debug-only assertions. Continuous state and parameter paths use
ordinary PyTorch autograd; analytic tensor equations carry physical inductive
bias. Learned residual capacity is added only after a smaller structured model
has a measured, repeatable error it cannot represent.

## Evidence contract for every rung

Before generating examples, freeze:

- disjoint development, selector, confirmation, and one-shot final manifests;
- the changed capability, source/config/protocol hashes, optimizer, update
  budget, and maximum two architecture attempts;
- per-axis, per-scenario, and per-horizon position and velocity limits;
- uncertainty coverage, proper-score, and sharpness limits when uncertainty is
  predicted;
- lifecycle, identity, association-coverage, and event confusion limits when
  those concepts exist;
- finite non-vanishing gradients to every intended learned owner, plus proof
  that frozen owners do not change;
- direct-versus-composed rollout consistency and a bounded long-horizon error
  envelope;
- separated observation, belief-update, and state-only rollout latency,
  persistent-state bytes, parameter count, and process-memory ceilings; and
- atomic project checkpoint/reload, clean source provenance, and supported
  evaluation/demo paths.

Development data may diagnose and repair implementation defects. Thresholds,
families, and final manifests do not change after selector access. A selector
failure stops before confirmation; a confirmation failure stops before final;
the final set is never recycled into development.

## Ordered capability rungs

### 1. Monocular temporal extent and long-horizon free motion — closed

Keep one visible sphere, fixed calibrated camera, known radius/gravity/drag,
and no contact. Replace two-frame velocity differencing with a differentiable
weighted exact linear-drag fit over a 16-frame RGB history. Query analytic
state at `0.1/0.25/0.5/1.0/2.0 s`, measure error growth and semigroup
consistency, and keep RGB-to-state timing separate from cheap state rollout.

Architecture attempt 2 of 2 terminated at its development audit without
opening protected data. Current position/velocity RMSE was
`0.016128 m`/`0.070461 m/s`; horizon position RMSE was
`0.022907/0.033205/0.050360/0.084191/0.149501 m`, producing 10 gate failures.
The immutable report SHA-256 is
`be488d045e259c0804a2a2b24215fa4eb3025d69f6113d8dbefe21d72f827554`.
The learned taper concentrated `77.63%` of temporal mass in the last three
frames and `91.71%` in the last five. Removing that taper in a development-only
diagnostic recovered most position gates, but future velocity remained
`0.01652 -> 0.01502 m/s` against `0.01 m/s` and the early zero-velocity-baseline
test still failed. The physics/oracle path was correct. This family is closed,
its protected manifests stay unopened, and its code is not an active workflow.

### 2. Observable-depth/RGB-D temporal state — standalone rung qualified

Retain the same one-sphere, fixed-camera, contact-free analytic world, but
measure metric depth as an observable input rather than asking a learned
monocular reliability taper to recover velocity from correlated
backprojection error. Simulator v7 now emits metric surface depth using the
same exact nearest ray--sphere winner as RGB, visibility, and instance output.
A parameter-free RGB-centre plus bilinear-depth measurement passed a seed-free
18-case public-renderer grid: maximum/RMSE position error
`0.00613210 m`/`0.00336217 m`, maximum/RMSE centre error
`0.0272064 px`/`0.00802947 px`, and finite centre/RGB/depth gradient norms
`0.673917`/`0.0718314`/`6.92869`. It consumes no simulator label, state, or
instance map. Focused validation is `29 passed` and independent review passes;
no episode seed namespace or protected split was accessed.

This single-frame metric core is not temporal convergence. Specification 1.54
freezes the next standalone protocol at config SHA-256
`5667cdb3603682b8d80a3e42793d25e36989269df1afacfa9b1028f2451101e9`
and canonical protocol payload SHA-256, computed before insertion of its
self-reporting digest field,
`4e334e9d7942ea3f2416c0a9f5ca8e327d1d0a1e9131074f20c051ebd3163ad7`.
It reserves development `41000000--41000023`, selector
`42000000--42000023`, confirmation `43000000--43000023`, and final
`44000000--44000047`. Every namespace was unopened at the protocol-freeze
boundary; that unopened state is historical. Fresh v2 development and the
protected sequence have now completed exactly once.

The estimator owns zero parameters, buffers, persistent tensor state, and
optimizer updates. It measures all sixteen frames independently, fits anchor
position/velocity with uniform differentiable exact free-motion WLS, and
queries analytic state at `0.1/0.25/0.5/1.0/2.0 s`. Neither confidence nor
validity may taper or select valid temporal rows. Gates cover current and
per-axis state, every horizon, trivial baselines, semigroup consistency,
fixed-output RGB/depth VJPs, an explicitly degraded RGB-only control,
fail-closed missing depth, diagnostic OLS covariance, memory, and separated
observation/rollout latency. An exclusive ledger records protected access in
selector -> confirmation -> final order only after an independently reviewed
passing clean-source development artifact.

On clean commit `8e68035`, the first development run over
`41000000--41000023` produced `82` finite reported scalar metrics and passed
all frozen gates. Current
position/velocity RMSE was `0.00279934 m`/`0.00207092 m/s`; horizon position
RMSE was `0.00286018/0.00297530/0.00322108/0.00385564/0.00539692 m`; perception
and rollout time was `0.235061 s`/`0.00391524 s`. Report, checkpoint, and
manifest SHA-256 digests are
`9cbea9f25181769ee5b6a87b097e738a29cdb9b386c8018b3044f07d58aa03e2`,
`6acd88edd203cdebb2b0820bad388e06a4c610ea1c659ff9f8ea6d701ad28059`, and
`b92f1e3f12475986ebd2971ad2de70187432c0941caa0335ea53e82abc3c1d01`.

Audit found a raw Python tuple/list equality check after JSON roundtrip. The
current source replaces it with canonical JSON SHA comparison and regression
coverage. That repair changes source, so the first run is a historical
conditional pass and its old artifacts cannot qualify protected access. A
fresh clean-source development rerun to fresh v2 paths was therefore required.
That paragraph remains the historical v1 invalidation record.

Qualified source `df0235a92a81d3c1d2ba4e69e64d639562e3dfe8` produced audited
v2 report/checkpoint SHA-256
`4cf1657ee95645c8c647433a8be660520e9cdc1a5e6ac106d85bd24547b4e740` and
`fd663e5fa52dded8156a3178070966e3458d93a7b5a49dd5dcb2cc0d6278514e`.
The one-shot qualification-report, canonical-summary, and ledger SHA-256 are
`7e4cface087620f058ade4cc83ac5fd197685ba26c8f0afb5089d8f7e646fe0d`,
`7e9954ae34ce55b6923765de0c084d5075f238bd012554eeb44049a0db161658`, and
`9fc139291dfb34b10125321d06fdf06ab68ed65df32f62c273a95e5ca7aa7b8b`.

Development, selector, confirmation, and final current position/velocity are
`2.799/2.071`, `3.073/1.991`, `3.078/1.644`, and `2.905/2.226` mm and mm/s;
their two-second position errors are `5.397/5.774/4.328/5.560 mm`. Every split
has `82` finite metrics and all `103` comparisons pass with zero failures,
optimizer updates, or learned state. Final is `18.5%` of its gate and `3%`
above development, with no collapse. Runtime, VJP, semigroup, baseline, and
RGB-only gates all pass. Final is consumed and must not be rerun.

The rung is qualified only at its declared standalone scope. OLS uncertainty
remains diagnostic; artifacts are SHA-bound but owner-writable. The former two
monocular attempts do not reset, and capacity remains fixed until the public
bridge reproduces this behavior.

### 3. Public one-slot online integration — qualified; final consumed

The qualified observation and temporal estimator are now implemented behind
the normal observation, `MeasurementSet`, `WorldBelief`, predict/observe/
correct, project-checkpoint, evaluator, and demo contracts. The frozen bridge
uses one composite batched `rgbd` packet containing `[B,3,H,W]` RGB,
`[B,1,H,W]` depth, batched calibration, and explicit image size. A modality-
qualified stream key prevents collisions between cache, temporal-history, and
scheduler ownership. The standalone two-second rung remains the qualified
diagnostic oracle; the public integration now has its own accepted evidence
below.

There is exactly one direct observable metric-position owner. The bridge
aligns sixteen raw positions by persistent object ID, fits velocity only with
uniform differentiable exact free-motion WLS, and uses parameter-free analytic
dynamics for every horizon. It owns zero parameters, state-dict entries, and
optimizer updates. Complete batch-four persistent runtime tensor state is
`25,364` recursively enumerated unique-storage bytes against a `32,768`-byte
gate. Fixed-output VJPs reach current velocity and every horizon through each
of the sixteen RGB and depth frames.

For an already-active persistent object, a well-formed frame with missing
depth, nonfinite or otherwise invalid depth in the sampled measurement support,
or no foreground appends an invalid causal row; the frozen sixteenth-frame
full-window ablation reports `sample_count: 16` and `valid_count: 15`. It emits
no valid/admissible temporal fit or direct velocity evidence, correction, or
birth; a finite diagnostic fit with `fit_valid: false` is not admissible
evidence. Before birth, the same frame advances runtime time but creates no
object-history row and no birth. Malformed packets, nonfinite RGB/calibration,
unsupported low precision, unknown/duplicate/stale streams, and invalid
prepared propagation reject atomically without mutating temporal state or
consuming one-use propagation.

Evaluator and demo paths carry truthful RGB-D modality metadata. The evaluator
labels its fifteen-frame warmup, while demo aggregate errors remain pooled
across warmup and post-warmup frames. The legacy RGB path remains supported.
Ordinary project checkpoints reproduce the parameter-free module and
configuration, but exact live-stream resume from a partially populated
temporal history is explicitly unsupported.

The frozen config SHA-256 is
`c40b3438c7fd60646d356db3fe54050039912ace288d9db89620b626106993a3`; the
current seed-free canonical protocol SHA-256 is
`e536b0d0b721042bff55501faf3445456219fcc987334b6ec1e892688ea560b2`.
Independent source/config/protocol audit passes, and the final current-byte
targeted gate is `421 passed in 62.72 s`; the complete repository gate is
`1209 passed, 16 skipped in 434.37 s`.

The first clean `ebda5a8` development had all `175` reported scalar metrics
finite and passed all frozen gate checks, but a
raw tuple/list protocol comparison after JSON roundtrip invalidated promotion.
No protected data was accessed. The rejected report/checkpoint remain under
ignored `runs/rgbd_online_bridge_v1/rejected_ebda5a8_json_protocol/` with
SHA-256
`2104ee87bcabdbd5312b4026a33e44e1de7d197e50215ec7f0bf0e0bb56992e3`
and
`38f4b2ef5addb98bb966360213d3bb36b43da606367fc60cd75d2ec487f1b866`.
Canonical JSON comparison/tamper rejection is source
`526b5123e6385c575a5777936272330d28972b93`.

Fresh canonical development report/checkpoint/manifest SHA-256 are
`dce6f920da85fbf696b7ae8a7a91d9cbf7d9084176e51ad7c319f92a6efe4966`,
`48249f1a5a0467b1da8c7bdb5ad9e909f8c502631ec2fbad832cb490a00c3099`, and
`069eb3331543727c911a07cc9a1bb352f6185ac8ceac7fafca502c9d7fab6d80`.
It has all `175` reported scalar metrics finite with no gate failures; current
position/velocity is
`3.068470 mm`/`2.191966 mm/s`, two-second position/velocity
`5.609913 mm`/`1.983371 mm/s`, slope `1.270721 mm/s`, and
perception/rollout `0.415134 s`/`3.575380 ms`.

Selector, confirmation, and one-shot final then passed exactly once. The
qualification-report/ledger SHA-256 values are
`7fd1829f663606910ac81990e4b633c63b1460dbc31dd24c71eedbd91b422908` and
`cf6a10dd672aafbdd91c92871ae349fef0c549d865cc6532e6c42f7d9be14e32`.
Protected manifest/result SHA-256 pairs are selector
`2159b044e089774b3b7df95509ac2cded19528de6ff133ae1b158a354ed7fbb9` /
`9ac6b7cc1b97da9961345fdcf5488ddec3ac6a0186215699a55a66acfbb983cb`,
confirmation
`2cad3224740b4d73871ff1d1e60795d45dc149ad03d197513eddf514cb9946bf` /
`1a3996914d59f840b2645e4b886f1027b830fa6f81c5763eb1735f25149aa9bc`,
and final
`3c5c904203ddd46ea790322e446466b2c58e603015456f239715aa07135011a3` /
`40d39accec8c2c6efa97f06a2f2748c580a5666b54c7dac4df36e3d7dc718bd1`.

| split | current position / velocity | two-second position / velocity | slope | perception / rollout |
| --- | --- | --- | --- | --- |
| development | `3.068470 mm` / `2.191966 mm/s` | `5.609913 mm` / `1.983371 mm/s` | `1.270721 mm/s` | `0.415134 s` / `3.575380 ms` |
| selector | `3.177543 mm` / `2.313401 mm/s` | `5.881384 mm` / `2.093251 mm/s` | `1.351921 mm/s` | `0.422070 s` / `3.569962 ms` |
| confirmation | `5.681172 mm` / `1.658775 mm/s` | `6.188252 mm` / `1.500921 mm/s` | `0.253540 mm/s` | `0.414407 s` / `3.537710 ms` |
| final | `2.996787 mm` / `2.221047 mm/s` | `5.433965 mm` / `2.009688 mm/s` | `1.218589 mm/s` | `0.417436 s` / `3.566628 ms` |

Every split has `175/175` reported scalar metrics finite with no gate failures
and `16/16` required history VJPs,
with zero identity change. Ablation, semigroup, memory, checkpoint, and
zero-state gates pass; final persistent state is `25,364` bytes and maximum RSS
is `708,853,760` bytes. The ledger is complete, final is consumed, and no raw
protected episode was reinspected. Evidence remains owner-writable despite
atomic replacement and hash binding.

The one-sphere contact-free public bridge is accepted and merged to `main` at
`3eed0b71e6f18c7036bf376c075493a89d5fdc9f`. At that boundary no multi-object,
association/occlusion, contact/event, task/planning, modality, or learned-
capacity convergence was claimed; rung 4 records the subsequent bounded
exactly-two-visible acceptance. Each next rung must
predeclare those changed capabilities and all accuracy, identity/association,
event, task-success, gradient, memory, and rollout-throughput gates. It may not
alter this accepted rung or tune on final.

### 4. Exactly two visible non-contact objects — qualified; final consumed

Specification 1.56 changes one capability: object count becomes exactly two.
Both fixed-radius spheres remain fully visible, image-separated, and
non-contacting under a fixed camera and fixed known dynamics. The accepted
one-object behavior is retained. Occlusion, reappearance, variable set size,
contact, moving camera, variable parameters, extra modalities, tasks, and
learned capacity remain closed.

Architecture attempt 2 uses parameter-free differentiable chromatic-plus-
spatial symmetric two-slot RGB-D geometry. Hard Hungarian assignment is a
discrete stable-identity controller only, never a claimed differentiable state
owner. Direct metric position has one owner; sixteen persistent-ID-aligned raw
measurements feed uniform differentiable exact free-motion WLS velocity; and
analytic dynamics answer position and velocity at
`0.1/0.25/0.5/1/2 s`. This is the smallest equation-led multi-object baseline,
not a learned-capacity result.

Config/protocol SHA-256 values are
`84e6f44b818bb9323a774bdba9492ef056e2a2747b93517fa38497ba83218bba` /
`42b9dca23fed303d5cee4641c8d8753977a872fc90d0b1086658d7f12b823ea0`.
Current position must reach anchor frame 15 only (`1/16`, exact zero
non-anchor gradient); current velocity and every horizon position/velocity
must reach all `16/16` RGB and depth frames. B4 uses four distinct scenes per
split and requires exact zero cross-scene coupling.

At the source-freeze boundary, seed-free validation passed `43` focused tests,
`281 passed in 15.61 s` combined, `1275 passed, 16 skipped in 447.29 s`
complete, Ruff/format/diff, and two independent audits. At that time no episode
result existed. Development `49000000--49000031`, selector
`50000000--50000023`, confirmation `51000000--51000023`, and final
`52000000--52000047` were unopened. Their manifest SHA-256 values are
`5a47a1a4a1405ba4c2fc3bce0087131d98fabfceb899beb26c6b4ba824a130f8`,
`415bc33407a46b79d0a3a746a8f5b192e31cfd4f6a68b9764e9b9943b7e6d7fe`,
`14f7dc3b762e4f987acbedcece815abd1c262bc9da60322f7f054e2c4eb4b3b1`, and
`b7e8913e938e2f7ae7f937979a60279916ff1a06f071427bcce9f08b0e354e75`.

That exact tree was committed at
`3b781e653a0287b2aa926e7c0b969e9197d48e42`. The one fixed development run and
independent audit passed, then the durable ledger admitted selector ->
confirmation -> final exactly once. All four splits have `396` finite gated
metrics and no failures. Development/selector/confirmation/final current
position RMSE is
`1.9029872e-5/1.6885011e-5/1.6776625e-5/1.7838631e-5 m`; current velocity RMSE
is `3.1932373e-5/3.1409136e-5/3.2594633e-5/3.1881889e-5 m/s`; and two-second
position RMSE is
`7.4638663e-5/7.0948345e-5/7.2466125e-5/7.1961138e-5 m`.

Identity coverage is one with zero switches, mismatches, or ambiguities;
visibility is one and event count is zero. Every split preserves anchor-only
current-position reach, `16/16` temporal-output reach, four unique B4 scenes,
and zero cross-scene coupling. Final perception/five-query rollout is
`0.352510 s`/`0.00359524 s`, runtime state is `28,512` bytes, and maximum RSS
is `579,817,472` bytes. Learned/module/optimizer/RNG state and updates are zero.
The exact final audit passes; it did not reopen raw protected episodes.
Owner-writable evidence is tamper-evident rather than OS-enforced WORM storage.
Final is consumed and cannot be rerun or tuned against.

### 5. Known calibrated orbital camera — qualified; final consumed

The reviewed two-visible tree was fast-forwarded to GitHub `main` through
`1e951520e5a2bf06c1932f64b8334e552247de82`.

Specification 1.57 inserted one smaller capability before the formerly
proposed partial-visibility rung. The historical source freeze retained exactly
two fixed-radius objects, full visibility, image separation, non-contact free
motion, complete RGB-D, gravity `0`, drag `0.05`, and the accepted
parameter-free analytic rollout. It added only a known calibrated orbital
camera supplied as time-aligned `world_from_camera` and did not infer pose.
Sixteen disjoint physical primitives cross four phases and two directions to
produce 128 joint scenes. Simulator semantics remain
`sphere_world_v7`.

Config/harness/runner/tests SHA-256 values are
`a9c348ea54b168ec78780d59d3b3eb066344d3a7551464b9aad1e5b9ac6d6cbd`,
`02e75b325bdf7bad310f8973a786a396b8762104261702b299a9f8103748e569`,
`11bee2e4d05f83caaf9dbed6ca2a54d4c11b7c70e4bf8e1747b261b8518ef192`,
and `d08c7bb4a1ba998a51dc2f0ddb1946596a5a299ed236cdf6a91b5711e2d0a1af`.
Canonical protocol/certificate SHA-256 values are
`7146befc869ea5f975177dd1c2da4691026439e1d36d84415aa23f696e61ef65`
and `7832ddb49081292d0f50a5eb63edb38fefb49d136d7e1757c73d9c658e42a36f`.
The exact 685-float schema includes public physics and resources, an explicit
stale frame-0-calibration negative control, and fixed-output VJPs to RGB,
depth, and `world_from_camera`. Learned parameters/state, pose estimation, and
optimizer updates remain zero.

Development `61000000--61000031`, selector `62000000--62000023`, confirmation
`63000000--63000023`, and final `64000000--64000047` have manifest SHA-256
`eb558805c2974302c33abef4531e142bb60e8f20045d8530330838223a6899a0`,
`c97fff97459ee9962b972cb7905887c2b2ed6eb5a1837d908f1512ce77e6d97f`,
`b47f03633732fc2986939e71007a0a79b12db2b42f0b5261b4ebd2d0a304f544`,
and `82927d192b53f2e4af11491f53039c145acfd8e0401a3e2b0b1e974591ee4174`.
At that boundary all were unopened and the fixed artifact root did not exist.
Development had to run once only after clean `HEAD` equalled published
upstream; external review had to bind the checkpoint, development report, and
development ledger before exactly-once selector -> confirmation -> final
access.

Source gates are `26 passed in 128.43 s`, `254 passed in 6.83 s`, and
exact-current-tree `1302 passed, 16 skipped in 594.59 s (0:09:54)`; two
independent science/security audits pass. They are source-freeze evidence only.
No development, accuracy, protected-split, or acceptance result existed at
that historical boundary.

The exact tree was then committed as
`c15afd6d57963b24bb98c5171462ff927e7c72fd`, with local upstream `0/0`,
worktree fingerprint
`0a5acfc54a5af482643b0c1037cf566a700e6122d2e6b51f7f4ad713ff652d2e`,
and runtime fingerprint
`bec3ca667fa464a3bbe82a83c14ffa924920ca367f14b6d9036ce52af041b83b`.
Development passed external three-hash review, then selector, confirmation,
and final passed exactly once. Development report/ledger/checkpoint hashes are
`56d7e32c461d9b5e3fbca5e2e11e015662cd08c3d60dfb4807e75cbcb7f8e37b`,
`3f9d5c9cf88ae7e40517337799e270d02493e99ed58eaec24884e276dcec5ddf`,
and `c473bb6d5f453c786c681509350d66364e1f1c61a2656a7c35354ab806da1a25`;
qualification report/ledger hashes are
`6daf2dea453db7c3a32b7950c8f31201ccc3fc32b9da1b14d8cc97dbd46ee0ad`
and `2aeb1c0194332004350c98628210d42724e31ece16614a210e2a84d6640b2719`.
Their respective byte sizes are
`88,743/1,544/78,573/202,540/2,293`.

Development/selector/confirmation/final result hashes are
`555871b24bfb764712d8dcae8473d5a9ad4c0ec6e9f02ffd42b2063af3cd7bc2`,
`fcfd1b39393a8e41d0b112244b7e5ca4fe3c0b2e4e63b4cd729659781198e9d6`,
`c3d644786d308a03d619eaf2a4d954bc216b1daf8655a9217f09e372ab27cd0b`,
and `b8ae823e961a981360717be273fe10d1ff5f9ce3bcbd6c396ba78fd5fdf0a4bf`.
Every split has `685/685` finite float metrics, all `686` constraints pass,
and `failures: []`. Current-position RMSE spans
`1.5474954e-5--1.7444936e-5 m`; current-velocity RMSE spans
`1.8386924e-5--1.9596576e-5 m/s`. Stale-camera position/velocity/two-second
position remains near `0.053/0.070/0.186`, separating known calibration from
the frame-0 control.

All eight strata, identity/history, certificate/public physics, resources, and
RGB/depth/calibration VJPs pass. Cross-scene, non-anchor, and homogeneous-row
gradients are zero. Perception/rollout is
`0.3452--0.3944 s`/`0.003469--0.003704 s`, state is `28,512` bytes, RSS is
below `578 MB`, and learned/optimizer state is zero. The five-file terminal
inventory is exact; the ledger stops after `final_test`; final is consumed;
and two independent qualification audits pass.

Evidence remains a terminal snapshot rather than signed append-only history;
exactly-once is inferred from final state plus committed control flow. Ignored
artifacts are unsigned/unlogged and filesystem-deletable, hashes carry no
reviewer identity, upstream equality used an unfetched local tracking ref, and
raw protected episodes were not rederived or audited. Acceptance therefore
covers only the declared known orbit, not arbitrary camera motion or pose
learning.

### 6. Partial visibility and missed-observation recovery — failed family closed

The separate partial-visibility and missed-observation-recovery family failed
and is terminal. It is not an accepted rung, must not be revived or retried,
and cannot be renamed as the next capability. Its historical material remains
only as failure provenance.

### 7. Next independent capability — not yet frozen

No next active rung is selected or authorized by this document. It must add one
genuinely new capability, freeze source/manifests/gates before access, preserve
every accepted lower-rung gate, and remain distinct from the closed partial
family. Candidates include identifiable drag, variable metric scale,
variable-size sets, contact/material identification, actions/planning, or a
new useful modality; listing them does not predeclare or authorize one.

### 8. Variable-size object sets

Variable-size sets require their own predeclared association/lifecycle and
recovery contract rather than inheriting the failed rung 6 family. Require
per-object state/horizon gates, association coverage, identity-switch limits,
permutation invariance, and recovery after missed observations before
increasing scene count or backbone capacity.

### 9. Hybrid contact and material identification

Introduce sparse two-sphere contact with an analytic hard forward resolver and
a differentiable local surrogate for learnable pre-contact state/parameter
paths. Gate collision timing and impulse/event F1 at every horizon, post-impact
state, energy/momentum diagnostics, and observable restitution/friction
identification. Do not use fabricated straight-through gradients or simulator
event labels at runtime.

### 10. Actions, counterfactuals, and planning tasks

Condition the shared dynamics on known forces/actions and score factual and
counterfactual trajectories. Tasks include interception, collision avoidance,
goal-reaching, and information gathering under occlusion. Planning must reuse
the same cheap state-only rollout, report task success alongside physical
accuracy, and remain consistent with the online posterior after new evidence.

### 11. Richer geometry and real observations

Only after the synthetic structured ladder closes, expand object geometry,
camera environments, RGB-D/video data, and useful asynchronous sensors. Keep a
simulator-to-real calibration split, explicit missing-modality behavior, and
the same state/uncertainty/identity/event/latency evidence rather than replacing
physical evaluation with perceptual similarity alone.

## Capacity policy

Capacity is a response to a demonstrated residual, not a rung by itself. For
each plateau, first localize whether error is observation, association, state
fit, parameter identification, event resolution, or rollout. Add the smallest
owner that can represent that residual; compare it with the equation-led
baseline under the same manifest; retain it only if selector and confirmation
improve without violating lower-rung or efficiency gates. Do not re-enable the
legacy attention stack or broad learned residual dynamics merely because more
parameters are available.

Long-horizon work extends the query envelope only after the shorter envelope
passes. Report absolute error at each horizon, error growth per second,
direct-versus-composed disagreement, and task-level degradation. A model is
not considered converged because a pooled short-horizon score plateaus while a
long-horizon axis, scenario, event, or identity slice continues to regress.
