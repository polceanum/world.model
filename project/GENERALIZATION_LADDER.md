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

### 1. Temporal extent and long-horizon free motion — active

Keep one visible sphere, fixed calibrated camera, known radius/gravity/drag,
and no contact. Replace two-frame velocity differencing with a differentiable
weighted exact linear-drag fit over a 16-frame RGB history. Query analytic
state at `0.1/0.25/0.5/1.0/2.0 s`, measure error growth and semigroup
consistency, and keep RGB-to-state timing separate from cheap state rollout.

### 2. Public one-slot online integration

Put the qualified inverse renderer and temporal fit behind the normal
observation, `MeasurementSet`, `WorldBelief`, predict/observe/correct, project
checkpoint, evaluator, and demo contracts. Reproduce rung-1 metrics while
ingesting frames causally and prove bounded update cost and persistent-state
memory. The standalone ladder remains a diagnostic oracle, not a second
production architecture.

### 3. Observable nuisance variables and useful modalities

Add one variable at a time: known moving-camera pose, then identifiable drag,
then variable metric scale. Prefer RGB-D for metric scale before asking
monocular RGB to infer an unobservable radius/depth combination. Camera pose or
IMU is calibration input, not hidden simulator state. Audio is added only for
an event or material task where it provides independently scored information.

### 4. Multiple non-contact objects

Progress from two visible objects to variable set size, partial occlusion, and
reappearance. Require per-object state/horizon gates, association coverage,
identity-switch limits, permutation invariance, and recovery after missed
observations before increasing scene count or backbone capacity.

### 5. Hybrid contact and material identification

Introduce sparse two-sphere contact with an analytic hard forward resolver and
a differentiable local surrogate for learnable pre-contact state/parameter
paths. Gate collision timing and impulse/event F1 at every horizon, post-impact
state, energy/momentum diagnostics, and observable restitution/friction
identification. Do not use fabricated straight-through gradients or simulator
event labels at runtime.

### 6. Actions, counterfactuals, and planning tasks

Condition the shared dynamics on known forces/actions and score factual and
counterfactual trajectories. Tasks include interception, collision avoidance,
goal-reaching, and information gathering under occlusion. Planning must reuse
the same cheap state-only rollout, report task success alongside physical
accuracy, and remain consistent with the online posterior after new evidence.

### 7. Richer geometry and real observations

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
