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

### 2. Observable-depth/RGB-D temporal state — protocol frozen; all data unopened

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
`44000000--44000047`; every namespace remains unopened.

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

The combined seed-free source/protocol gate is `103 passed`, independent review
passes, and the final repository gate is `1129 passed, 16 skipped in 428.18 s`.
No development episode or protected split has been generated, so this is not
development evidence or a temporal/long-horizon convergence claim. The former
two monocular attempts do not reset; this is a new modality and structural
capability, not a third tuning attempt. Capacity remains fixed until this rung
qualifies.

### 3. Public one-slot online integration

Put the qualified observation and temporal fit behind the normal
observation, `MeasurementSet`, `WorldBelief`, predict/observe/correct, project
checkpoint, evaluator, and demo contracts. Reproduce the accepted temporal
metrics while ingesting frames causally and prove bounded update cost and
persistent-state memory. The standalone ladder remains a diagnostic oracle, not a second
production architecture.

The first bridge uses one composite batched `rgbd` packet containing
`[B,3,H,W]` RGB, `[B,1,H,W]` depth, batched calibration, and explicit image
size. It uses a modality-qualified sensor key because current cache, temporal-
history, and scheduler maps otherwise collide on `sensor_id`; separate
same-timestamp RGB/depth packets also make elapsed-time ownership order-
dependent. The bridge must fit only raw associated metric positions, keep
missing depth fail-closed, and synchronize checkpoint/evaluator/demo schemas.

### 4. Observable nuisance variables and additional useful modalities

Add one variable at a time: known moving-camera pose, then identifiable drag,
then variable metric scale. Camera pose or IMU is calibration input, not hidden
simulator state. RGB-only depth inference is a later ablation against the
accepted RGB-D state, not the route used to establish metric observability.
Audio is added only for an event or material task where it provides
independently scored information.

### 5. Multiple non-contact objects

Progress from two visible objects to variable set size, partial occlusion, and
reappearance. Require per-object state/horizon gates, association coverage,
identity-switch limits, permutation invariance, and recovery after missed
observations before increasing scene count or backbone capacity.

### 6. Hybrid contact and material identification

Introduce sparse two-sphere contact with an analytic hard forward resolver and
a differentiable local surrogate for learnable pre-contact state/parameter
paths. Gate collision timing and impulse/event F1 at every horizon, post-impact
state, energy/momentum diagnostics, and observable restitution/friction
identification. Do not use fabricated straight-through gradients or simulator
event labels at runtime.

### 7. Actions, counterfactuals, and planning tasks

Condition the shared dynamics on known forces/actions and score factual and
counterfactual trajectories. Tasks include interception, collision avoidance,
goal-reaching, and information gathering under occlusion. Planning must reuse
the same cheap state-only rollout, report task success alongside physical
accuracy, and remain consistent with the online posterior after new evidence.

### 8. Richer geometry and real observations

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
