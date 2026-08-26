# Project status

## Active generalization program — 2026-08-26

The pre-generalization public base is GitHub `main` commit
`c16acc99ef13757fc8f88528bfd0d66db4a2f4fd`. Broad heterogeneous training
remains paused. The active contract is specification 1.54; the accepted
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
deleted after archive verification. `runs/` is now empty. Repository caches,
generated demos, and selected stale temporary clones/caches were also removed.

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

### Frozen RGB-D temporal protocol — all manifests unopened

Specification 1.54 now freezes the standalone parameter-free temporal rung.
The exact config SHA-256 is
`5667cdb3603682b8d80a3e42793d25e36989269df1afacfa9b1028f2451101e9`, and
the canonical protocol payload SHA-256 before insertion of its self-reporting
digest field is
`4e334e9d7942ea3f2416c0a9f5ca8e327d1d0a1e9131074f20c051ebd3163ad7`.
The disjoint manifests are development `41000000--41000023`, selector
`42000000--42000023`, confirmation `43000000--43000023`, and final
`44000000--44000047`. All four remain unopened: no development or protected
episode has been generated.

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

Protected access can begin only from a clean, passing, independently reviewed
development report/checkpoint pair. An exclusive ledger must record access
before materialization in selector -> confirmation -> final order and stop on
the first failure. The combined seed-free implementation/protocol gate is
`103 passed`, independent review passes, and the final repository gate is
`1129 passed, 16 skipped in 428.18 s`. This is source-contract evidence only:
no development accuracy, protected result, temporal convergence, or
long-horizon convergence is claimed.

The later online rung must use one batched composite `rgbd` packet and a
modality-qualified sensor key before it touches `MeasurementSet`, causal
history, checkpoints, evaluator, or demo. The standalone protocol does not
claim that public bridge, and no model capacity or scene complexity should be
added before temporal qualification.

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

The newer four-file RGB-D temporal protocol surface has the proportional
`103 passed` focused gate, independent review, and a final complete repository
gate of `1129 passed, 16 skipped in 428.18 s`. The older `1091` result remains
the historical single-frame-core boundary.

No accepted long-horizon temporal, multi-object, contact, multimodal, or planning
convergence claim exists yet.
