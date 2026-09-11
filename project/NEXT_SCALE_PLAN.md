# Impact-first world-model scale plan

## Objective

Expand the compact CPU-first model's useful behavioral envelope: make it reason
causally over several known interventions, remain accurate through repeated
contacts over 4--8 seconds, support more visible objects, and establish a clean
route from spheres to general rigid geometry. Preserve the public belief,
observation, action, rollout, and planning contracts and the managed 250 MiB
artifact budget.

This plan deliberately prioritizes capabilities a downstream controller can use.
Calibration polish and evaluator throughput remain conditional work: undertake
them only when they block a capability gate or make the next governed run
impractical.

## Guardrails checked before implementation

- Planning remains a required downstream acceptance test and is never a
  training loss.
- Every future intervention is a public, known action applied at its absolute
  timestamp. Hidden simulator actions remain censored.
- The original single-action and action-free paths are compatibility oracles.
- State-first tests establish dynamics evidence only; they cannot be presented
  as RGB-D perceptual qualification.
- New shape metadata must decode old one-component sphere checkpoints exactly.
- Scale object capacity through configuration and packed active interactions,
  not learned slot identity or a large attention model.
- Retain summaries and bounded vector animation data, never episode tensors or
  rendered frame sequences.

## Phase A — Multi-action causal dynamics and genuine long horizons

Status: **implemented and passing** in
`runs/20260910-impact-scale-v3`.

1. Add an immutable ordered `WorldImpulseSchedule` around the existing
   `WorldImpulseAction`. Validate the whole schedule before rollout and require
   strictly increasing timestamps per batch row.
2. Split each dynamics interval at every scheduled timestamp, apply each
   impulse exactly once to its persistent target, and continue through the same
   analytic contact resolver. Preserve the exact single-action branch.
3. Exercise 4- and 8-second recursive rollouts with two or three actions,
   object contact, floor contact, wall contact, and repeated bounded contact.
4. Measure horizon position/velocity error, collision F1 and timing, energy
   error, exact action count, latency, and source-belief immutability.
5. Keep the three illustrative trajectories as inline vector payloads capped at
   96 KiB each.

The corrected pilot passes every declared gate. Its worst 8-second position
RMSE is `5.994e-4 m`; the floor/wall compound collision F1 is `0.9167` with a
one-frame timing error. The first v1 run remains as a failed diagnostic: it
identified that checkpoint loading incorrectly replaced scene-configured plane
offsets. Environment buffers are now rebound after learned-state loading and
the regression is tested directly.

## Phase B — Action-sequence planning and replanning

Status: **implemented and passing** in the same governed pilot.

1. Accept single impulses, schedules, and the existing no-action candidate in
   the planner without changing the public result shape.
2. Flatten `B x K` candidates into one rollout, including heterogeneous
   schedule lengths, while retaining the serial implementation as the exact
   oracle.
3. Certify candidate sets privately only after public costs are produced.
   Require a unique oracle winner and normalized winner margin.
4. Test K=8 and K=32 choices for pair contact and repeated-wall behavior, then
   replan after executing the shared first action.

All four K/scenario slices select the oracle winner with zero normalized
regret, successful terminal goals, exact replanning consistency, and zero
serial/vectorized cost difference. Measured vectorized latencies range from
`0.018` to `0.037 s`.

## Phase C — General rigid geometry without a compatibility break

Status: **implemented and passing** in
`runs/20260910-rigid-capability-v2`.

1. Encode primitive geometry explicitly while retaining component zero as the
   legacy conservative radius. A one-component geometry tensor decodes exactly
   as a sphere.
2. Support sphere and box tags plus box half-extents in a centralized codec;
   keep `ObjectBelief.radius` behavior unchanged for legacy callers.
3. Add an analytic oriented-box reference simulator and observable RGB-D
   box fitting. Qualify static/moving boxes before mixed sphere/box contacts.
4. Add SAT-based box contact and sphere-box closest-point contact behind the
   existing dynamics interface. Dense N<=6 remains the numerical oracle; any
   learned correction must be bounded, antisymmetric, and justified by a
   truth-state ablation.

Do not call the codec itself box support. Promotion requires visible RGB-D
recovery, persistent identity, action targeting, contact timing, and planning
on held-out aspect ratios and orientations.

The completed implementation preserves the exact legacy all-sphere renderer
and contact path. It adds exact ray/OBB RGB-D rendering, public multi-view
surface fitting, oriented SAT box/box contact, closest-point sphere/box
contact, and a separate rigid reference integrator. The held-out box/box and
sphere/box qualification passes primitive recovery, metric geometry, two-second
prediction, collision timing, action invariants, and K=8/K=32 planning. The
passing run is `134,420` bytes. The v1 diagnostic remains visible because it
identified an overly narrow action-isolation tolerance and an insufficiently
separated planning task; neither protected threshold was weakened.

## Phase D — Increase visible-set capacity

Status: **complete N=7/8 qualification implemented and passing** in
`runs/20260910-perceptual-scale-v1`.

The independent N=8 profile derives ten proposals from
`max_objects=8 + birth_proposals=2`, loads the incumbent strictly, and ingests
three public calibrated RGB-D frames. It observes all eight objects with
`5.96e-8 m` position RMSE, `0.293 s` measured three-frame inference latency,
finite state,
and `87,436` learned-weight bytes. This is explicitly development evidence,
not full perceptual qualification.

The governed ladder qualifies N=7/8 across:

1. separated motion and broad known actions;
2. pair contact and repeated contact;
3. birth/removal and remove-then-birth slot reuse;
4. short partial occlusion and identity recovery;
5. the strongest sensor/camera/parameter compositions.

Raise image resolution only if pixel-support ablation identifies it as the
owner. Keep N=12/16 state-only until N=8 passes the complete perception,
lifecycle, contact, and downstream planning gates.

All twelve count/family cells pass proposal, exact-count, current-position,
identity, lifecycle/recovery, action, contact, and finite-state gates. The four
N=7/8 by K=8/32 planning slices select the oracle winner with zero regret and
exact serial/vectorized agreement. The run retains three small N=8 vector
animations and no generated frames; it occupies `279,700` bytes. N=12/16
remain state-only because no higher-count visual evidence was added.

## Phase E — Integrated capability benchmark

Status: **implemented and passing** in
`runs/20260911-integrated-capability-v2`.

After box and N=8 single-family gates pass, add a compact integrated benchmark
that combines changing membership, a moving calibrated camera, short
occlusion, mixed rigid primitives, repeated contact, and a two/three-action
schedule. Use deterministic manifests, stream RGB-D, reduce online, and retain
only summaries plus three small vector examples.

The checkpoint may advance only if it:

- passes the existing absolute physical and planning floors;
- passes the new 4/8-second action-sequence floors;
- improves the paired macro capability score by at least 3% with a positive
  95% bootstrap bound and improves the worst supported family by at least 1%;
- regresses no accepted-like slice by more than 2%;
- remains within 1 MiB learned weights, 2.5 GiB RSS, the N=16 state-only
  `0.10 s` gate, and no more than 10% incumbent online-latency regression.

The deterministic four-object bridge combines two spheres, two held-out
oriented boxes, lifecycle changes, a moving calibrated camera, one-frame full
occlusion and recovery, five repeated contact intervals, three absolute-time
known actions, and a four-second open-loop rollout. Its public RGB-D geometry
and native-rate post-event state produce `2.824e-6 m` four-second position RMSE
and `1.694e-6 m/s` velocity RMSE; all five model/reference contact frames match.
K=8/K=32 planning selects the oracle winner with zero regret, exact serial
parity, and `0.020/0.033 s` vectorized latency. The portable evidence occupies
`86,746` bytes and includes the true 4.00-second endpoint as an inline vector
animation. This qualifies the incumbent behavior; it is not a new learned
checkpoint promotion, so the paired 3% candidate-improvement rule was not
invoked and no residual was widened.

## Completion and next frontier

Phases A--E are implemented in order and remain separate regression tiers. No
learned module was widened because the only integrated development failure was
owned by public geometry/state estimation: public face-plane and algebraic
sphere fits removed it directly. The next scale phase should retain these gates
while extending only genuinely new behavior—rotational contact response,
less appearance-supervised discovery, or visual qualification beyond N=8—after
a fresh plan freezes the corresponding observability and efficiency limits.

The implementation gate passed Ruff, Ruff format, compileall, diff hygiene,
the focused and broad compatibility checks, and the complete repository suite:
`2457 passed, 16 skipped` in `1:11:47`. The final report-copy reconciliation
then passed all 12 focused dashboard tests plus Ruff, format, and diff hygiene.
