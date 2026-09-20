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

## Phase F — Open-world discovery, rotational contact, and self-calibration

Status: **implemented and passing** in
`runs/20260911-open-world-six-dof-v2`.

1. Replace predeclared appearance handles in the new path with discovery from
   public calibrated RGB-D support. Use appearance measured in the current
   observation only for cross-view grouping and temporal association; allocate
   persistent IDs inside the runtime.
2. Estimate metric pose, linear velocity, and local angular velocity, align
   equivalent box axes over time, and preserve identity across a short missing
   observation. Initialize physical parameters from explicit neutral priors.
3. Identify mass and drag from public isolated intervention/free-motion
   evidence and identify effective restitution/friction from observed contact.
   Reject inconsistent evidence and contract uncertainty only on accepted
   updates.
4. Add opt-in contact points, analytic inertia, angular impulse, frictional
   torque, and orientation propagation. Preserve exact all-sphere behavior and
   compare against an implementation-independent six-DoF simulator.
5. Add terminal world-pose planning with quaternion-geodesic error. Require
   K=8/K=32 correct winners, zero/low regret, exactly-once actions, isolation,
   source immutability, and exact serial/vectorized agreement.
6. Visualize orientation RMSE and parameter convergence with named axes; show
   pose spokes and contact rings in a bounded SVG animation without retaining
   RGB-D frames.

The passing run discovers two unfamiliar objects without a prototype or label
input, recovers the box ID after one missing frame, reduces mean physical-
parameter error from `0.9983` to `3.24e-15`, matches both reference contact
frames, and stays below `0.000511 m` position and `0.267 degrees` orientation
RMSE through 1.2 seconds. Pose-aware K=8/K=32 planning has the correct winner,
zero regret, exact cost parity, and `0.456/0.606 s` median vectorized latency.
The 77,966-byte run retains only JSON, HTML, and its manifest.

## Phase G — Appearance-independent touching separation and longer recovery

Status: **implemented and passing** in
`runs/20260911-open-world-touching-recovery-v1`.

1. Add an opt-in `depth_geometry` discovery mode that partitions connected
   public depth support from silhouette medial peaks rather than chromatic
   components. Merge nearby raster fragments before partitioning and use
   calibrated metric fits to group the unordered proposals across views.
2. Make temporal association cues explicitly configurable. Qualify with both
   appearance and primitive-label weights set to zero, so identity follows
   predicted metric position and public shape scale rather than colour or a
   private category.
3. Retain tracks through an eight-frame complete RGB-D gap, then reassociate
   recovered observations to the original runtime-owned persistent IDs.
4. Require post-recovery K=8/K=32 counterfactual action selection, exact
   serial/vectorized agreement, source-belief immutability, and bounded CPU
   latency. Planning remains an evaluation only.
5. Retain one compact vector animation and a named recovery-error curve. Encode
   object identity by colour and distinguish model/reference by filled-solid
   versus open-dashed marks, rather than assigning role colours that can imply
   false correspondence.

The deterministic N=3 qualification uses identical RGB albedo for two oriented
boxes and one sphere. Its front depth support has two connected components but
geometry discovery yields three instances. All three IDs survive eight missing
frames with no duplicates; maximum force-free gap error is `0.034052 m`,
post-recovery position/velocity RMSE is `0.003109 m`/`0.026675 m/s`, and
K=8/K=32 planning selects the private oracle winner with zero regret and exact
cost parity in `0.364/0.542 s`. The run occupies `62,806` bytes and retains no
RGB-D frame, raster image, or video.

## Phase H — Batched multi-contact six-DoF scale bridge

Status: **implemented and passing** in
`runs/20260912-multicontact-six-dof-v1`.

1. Preserve lexicographic pair resolution inside each scene while vectorizing
   independent batch and planning-candidate rows. Keep the dense/serial path as
   the exact numerical and action-selection oracle.
2. Exercise alternating spheres and oriented boxes at N=4, N=6, and N=8 over
   two seconds with three known impulses, all adjacent contact pairs, repeated
   contacts, and simultaneous contacts.
3. Require bounded position, velocity, orientation, contact attribution and
   contact-timing error at every cardinality. Keep this explicitly state-first;
   the initial belief comes from a controlled oracle and truth is withheld
   after initialization.
4. Require downstream N=8 K=8/K=32 planning, exact winner and numerical parity,
   at least 5x vectorization speedup, source-belief immutability, and fresh-
   process CPU latency gates.
5. Publish one compact N=4/N=6/N=8 vector gallery with object-identity colours,
   role fill/line styles, projected box-orientation spokes, action times, and
   contact markers. Retain no image frames or video.

Every count recovers all adjacent reference pairs with contact-pair F1 `1.0`
and exact first-contact timing. Maximum position error is `0.005234 m`, maximum
velocity error is `0.043886 m/s`, and maximum orientation error is `3.2282
degrees`. K=8/K=32 planning selects the private winner with zero regret and zero
cost difference. Candidate-row batching is `7.61x`/`28.12x` faster than the
serial oracle. The final run is `450,359` bytes and retains only JSON, HTML, and
manifest data.

## Phase I — Integrated public visual-dynamic scale

Status: **implemented and passing** in
`runs/20260912-visual-dynamic-scale-v1`.

1. Discover N=4, N=6, and N=8 alternating spheres and observably oriented boxes
   from six moving calibrated RGB-D views without runtime prototypes, instance
   maps, private IDs, or primitive labels.
2. Require two consecutive public detections before allocating an ID, retire
   after two misses, and carry runtime-owned identity through reduced multi-view
   visibility, visible removal, replacement, and a separate birth.
3. Estimate anchor motion from bounded public history, retaining legacy
   two-sample behavior by default. Use explicit measurement-resolution
   deadzones only in this declared profile.
4. Feed the public belief directly into the Phase H batched six-DoF path for
   three known impulses, every adjacent pair, repeated and simultaneous
   contacts, and a two-second forecast.
5. Require N=8 K=8/K=32 planning from the same belief with correct private
   winner, zero regret, exact serial cost parity, at least 5x speedup, and source
   immutability.
6. Retain three inline vector forecasts with runtime-ID colours, separate
   model/reference roles, pose spokes, action/contact events, and named world
   axes. Keep all generated RGB-D and frame media transient.

Every count achieves proposal F1, identity accuracy, lifecycle F1, primitive
accuracy, and contact-pair F1 of `1.0`. Worst two-second position, velocity, and
orientation errors are `0.013789 m`, `0.090706 m/s`, and `9.4057 degrees`.
K=8/K=32 planning selects the private winner with zero regret and exact cost
parity; candidate batching is `7.57x`/`27.86x` faster than serial. The passing
run is `484,220` bytes and contains no raster, video, or per-frame artifacts.

## Phase J — Cross-capability prediction hardening

Status: **implemented and passing** in
`runs/20260913-capability-hardening-v2`.

1. Freeze accepted metrics instead of inferring regression from unlike
   dashboard scores. Replay protected long-horizon, integrated, open-world,
   recovery, state-first multi-contact, and public visual protocols through one
   deterministic aggregate audit.
2. Use truth-state component ablations to identify the owner of any drift
   before changing capacity. Require every source protocol's original gates in
   addition to the paired envelope.
3. Correct public pose only from causal observable history. Preserve the
   legacy tracker output unless an explicit profile enables the bounded
   outlier gate.
4. Reduce interaction cost by skipping unused primitive geometry paths, while
   preserving pair order, heterogeneous batch dispatch, dense/serial parity,
   action invariants, and source immutability.
5. Show accepted, candidate, delta, threshold, and status directly in the
   dashboard. Plot only repeated comparable protocols as trends and keep the
   protected long-horizon curve separate.

All 32 paired checks pass. Public N=8 two-second position, velocity, and
orientation errors improve by `68.9%`, `84.3%`, and `44.7%`, while repeated-
contact F1 improves by `32.0%`. The protected 8-second position RMSE remains
`5.994e-4 m`; aggregate K=8/K=32 planning retains exact winners, zero regret,
goal success, serial parity, and zero cost difference. The lower-is-better
macro ratio improves by `5.74%`. Isolated N=8 mixed-rigid rollout time falls
from the `36.658 s` failed diagnostic to `18.528 s`. The final compact audit is
477,609 bytes with no retained source frames or video.

## Phase K — Single-model adaptive physical generalization

Status: **implemented and passing** in
`runs/20260914-adaptive-physics-scale-v2`.

1. Keep one shared analytic `DynamicsModel`; adapt only the mass, drag,
   restitution, and friction already stored per object in `WorldBelief`.
2. Derive evidence from public calibrated RGB-D position traces: free motion,
   a precisely timed known impulse, and an isolated stationary-boundary
   collision. Do not expose simulator velocities, parameters, IDs, or primitive
   labels to runtime inference.
3. Accept a parameter update only when the trace is finite, temporally ordered,
   physically identifiable, and below the declared fit residual. Contract only
   the corresponding existing uncertainty block on acceptance.
4. Repeat public N=4/N=6/N=8 discovery and lifecycle evidence, then forecast
   heterogeneous mixed rigid contacts and three known actions for four seconds
   against a finer independent simulator.
5. Require per-object and aggregate position, velocity, orientation, contact,
   lifecycle, identity, latency, invariance, and finite-state gates. Preserve
   raw frame-exact contact F1 and separately report the declared one-frame
   timing-tolerant score.
6. Require N=8 K=8/K=32 downstream action choice with exact serial winners and
   costs, zero regret, goal success, source immutability, and at least 5x
   batching speedup.
7. Publish only compact JSON/HTML/manifest evidence plus inline vector
   keyframes, including parameter convergence, truth-parameter/state
   attribution, and an all-object prediction ledger.

All `72/72` parameter blocks update. Mean relative physical-parameter error
falls from `0.4374` under neutral priors to `0.0023`, with a worst parameter
error of `0.01875`. N=4/N=6/N=8 four-second position RMSE is
`0.02940/0.01576/0.02596 m`; one-frame-aligned repeated-contact F1 is
`0.8000/0.7857/0.6479`. K=8/K=32 planning remains exact with zero regret and
`7.48x/28.12x` batching speedup. The passing run is 939,418 bytes and retains
no generated frames, raster animation, or video. The exact final source passes
Ruff, Ruff format, compileall, focused behavioral tests, browser inspection,
and the complete repository suite: `2506 passed, 16 skipped` in `1:37:19`.

## Phase L — Learned physical-evidence adaptation

Status: **development bridge implemented and passing** in
`runs/20260914-neural-adaptive-physics-v2`.

1. Keep calibrated RGB-D geometry, persistent object state, exact known-action
   application, contact resolution, and planning as the structured scaffold.
   Replace the hand-written physical-parameter fit with one shared learned
   adapter rather than learning an unconstrained trajectory shortcut.
2. Encode only public object-motion evidence: metric position differences,
   elapsed times, known impulse vectors, observed stationary-boundary normals,
   and causal invariant candidate statistics. Private parameters are optimizer
   targets only and never runtime inputs.
3. Use a compact permutation-invariant evidence transformer: width 48, four
   heads, two pre-RMSNorm self-attention blocks, SwiGLU width 96, one learned
   query, and bounded mean/variance heads for mass, drag, restitution, and
   friction. It has `48,920` trainable float32 parameters (`191.1 KiB`).
4. Train from scratch on a deterministic streamed distribution for `2,048`
   AdamW updates with batch size 128 (`262,144` examples). Retain only the
   weights-only checkpoint, reduced metrics, HTML, and manifest; planning is
   evaluated downstream and never enters the optimized loss.
5. Require learning-curve reduction, held development and compositional-edge
   parameter accuracy, evidence-order invariance, changed and non-zero weights,
   the existing N=4/N=6/N=8 public RGB-D four-second forecast, and K=8/K=32
   serial/vectorized planning agreement.

The passing v2 run changes all `48,920` parameters from their deterministic
initialization and reduces held training-distribution parameter error to
`3.26%` (`11.17%` p95); the harder edge-compositional mean is `12.87%`.
On the existing public RGB-D N=4/N=6/N=8 ladder, mean parameter error is
`2.25%/2.15%/2.00%` and four-second endpoint position RMSE is
`0.04756/0.04157/0.03708 m`. K=8 and K=32 both retain the correct winner, zero
regret, goal success, and exact serial/vectorized cost parity. The compact run
is about `1.2 MiB` and retains no training batches, optimizer state, RGB-D
frames, or raster/video media.

The failed v1 run remains visible because it exposed a real shortcut: the
synthetic event token encoded an absolute time and used a different pre-action
cadence than the public runtime. Removing absolute event time and matching the
causal measurement cadence fixed the transfer failure. This phase establishes
learned cross-episode inference plus observation-conditioned adaptation; it is
not an end-to-end pixels-to-futures model, a protected incumbent promotion, or
online gradient learning.

## Phase M — Learned twelve-second stability

Status: **implemented and passing** in
`runs/20260914-neural-long-horizon-v1`.

1. Measure the Phase-L checkpoint against the independent simulator through
   12 seconds before changing the architecture. Retain N=4/N=6/N=8 curves at
   0.5/1/2/4/6/8/10/12 seconds plus a truth-parameter solver floor.
2. Keep the exact 48,920-parameter, two-layer/four-head/width-48 evidence
   transformer. Anchor its prediction to causal sufficient statistics already
   carried by public evidence, and learn only a bounded normalized residual.
3. Add a differentiable multi-horizon physical-response objective for free
   decay, accumulated travel, known-action response, normal restitution, and
   tangential retention at 0.5/2/4/8/12 seconds. Planning remains excluded
   from training.
4. Train for 8,192 deterministic AdamW updates over 1,048,576 streamed
   examples with warmup and cosine decay. Retain a weights-only checkpoint and
   reduced evidence; discard batches and optimizer state.
5. Require every N=4/N=6/N=8 position slice at 2/4/8/12 seconds to improve by
   at least 2% against the prior neural checkpoint. Also require absolute
   12-second limits, unchanged contact F1, finite immutable rollouts, complete
   weight learning, and exact K=8/K=32 planning winners and costs.

The incumbent audit confirmed genuine accelerating drift: N=4 rose from
`0.02210 m` at two seconds to `0.04756/0.20500/0.38369 m` at 4/8/12 seconds.
The new model lowers those points to
`0.00599/0.02989/0.07460/0.12375 m`. At 12 seconds, N=4/N=6/N=8 improve from
`0.38369/0.17512/0.18273 m` to `0.12375/0.15255/0.07668 m`, reductions of
`67.75%/12.89%/58.04%`. Every declared per-horizon comparison passes; contact
F1 remains `1.0` at all counts.

Held mean parameter error falls from `3.26%` to `2.35%`, and compositional OOD
error falls from `12.87%` to `5.11%`. The multi-horizon objective falls
`53.78%`. All `48,920/48,920` parameters are non-zero and changed from
initialization; learned weights remain `191.1 KiB`. K=8/K=32 planning keeps the
same exact private-oracle winner, zero regret, successful goals, and zero
serial/vectorized cost difference. The compact run is `2,397,914` bytes and
contains three downsampled inline vector forecasts through the true 12-second
endpoint, with no frame directories, raster media, optimizer state, or
training batches.

The truth-parameter 12-second solver floor is still
`0.10054/0.13972/0.06982 m` at N=4/N=6/N=8. The remaining candidate gap is
therefore small for N=6/N=8 but material for contact-heavy N=4. This phase does
not claim indefinite prediction or twelve-second planning qualification.

## Phase N — Event-local rigid-contact correction

Status: **in progress**.

The first N=4 twelve-second screen rejects a context-free normal-impulse
correction: relation multiplier logits of `-0.5/0.0/+0.5` produce endpoint
position RMSE `0.30379/0.12375/0.21856 m`.  The zero-output analytic resolver
is therefore the local optimum for that scalar intervention.  Phase N will not
promote a global impulse bias; it proceeds only with evidence that a
geometry- and event-timing-conditioned residual owns a protected error slice.

A second deterministic public-state N=4, four-second screen found a finite
integration-timing seam around an imminent rigid contact: a disposable local
half-grid reduced endpoint position RMSE from `0.02989 m` to `0.00522 m` and
made first-contact timing exact.  It did not generalize: the same direction
raised the N=6 twelve-second endpoint from `0.15255 m` to `0.16662 m`.
Accordingly no timing-refinement runtime path is retained or promoted.  The
screen remains useful attribution evidence, but the next milestone must target
substantially longer-horizon stability rather than a narrow short-contact gain.

1. Keep Phase M's public RGB-D belief, learned physical-parameter adapter, and
   analytic six-DoF resolver fixed as the control.  Attribute the remaining
   long-horizon error with truth-parameter and truth-state rollouts before
   training any new capacity.
2. Do not pursue a global or local grid refinement as a Phase-N model change:
   the N=6 counterexample rejects that intervention. Only if a residual timing
   error remains material after the extended-horizon work below, train the
   existing zero-initialized relation impulse rows on causal, event-local pair features.
   Disable continuous pair forces and node acceleration so the learned output
   can only make a bounded symmetric correction to an already-resolved normal
   impulse.
3. Require exact analytic equality at zero output, antisymmetric momentum
   exchange, immutable source beliefs, finite rollouts, and no learned effect
   outside geometrically applicable active pairs.  Candidate selection and
   planning remain evaluation-only.
4. Compare a fixed N=4/N=6/N=8 mixed-rigid repeated-contact manifest against
   the Phase M checkpoint and its truth-parameter solver floor through twelve
   seconds.  Keep per-count, per-horizon position and orientation evidence
   separate; reject a pooled gain that regresses any protected slice.
5. Retain a compact checkpoint and reduced report only if the one-component
   screen demonstrates a real improvement.  Otherwise preserve the analytic
   fallback and record the contact correction as rejected rather than widening
   the model.

## Phase O — Forty-eight-second stability frontier

Status: **diagnostic implementation in progress**.

The next meaningful scale-up is not another contact micro-optimization. It is
to quadruple the open-loop horizon from twelve to forty-eight seconds and
optimize the complete 12--48-second tail. The model remains one shared
48,920-parameter bounded causal-residual transformer plus analytic rigid
rollout; no ensemble, trajectory-ranking loss, or hidden-state shortcut is
introduced.

1. Establish a deterministic no-write N=4/N=6/N=8 48-second audit from the
   Phase-M checkpoint. Report `2/4/8/12/16/20/24/32/40/48 s` position and
   velocity curves, contact quality, finiteness, source immutability, and the
   exact known-action schedule.
2. Attribute the tail separately with public-belief, truth-parameter, and
   truth-state solver controls. Do not train longer merely because error
   increases: change only the smallest owner supported by that comparison.
3. If physical-parameter error owns the tail, extend the existing causal
   physical-response objective to balanced `12/16/20/24/32/40/48 s` tail
   samples and train the same model from scratch. Preserve 2--12 second
   performance as a compatibility envelope; planning stays downstream-only.
4. If the truth-state solver owns the tail, improve the analytic integration
   only through a separately gated, all-cardinality intervention. If public
   belief owns it, improve observable-state history before widening the
   transformer. Do not combine these changes in one run.
5. Promote a 48-second candidate only if every N=4/N=6/N=8 slice preserves all
   2--12 second gates (no slice regresses by more than 2%), improves the
   12--48 second paired tail score, remains finite and source-immutable,
   preserves K=8/K=32 serial/vectorized planning parity, and stays within the
   existing model-size and artifact limits. Set absolute 48-second floors
   after the frozen baseline audit, not retrospectively.
6. Publish only the run summary, portable report, dashboard data, and at most
   three vector long-horizon animations. Each animation must visibly label
   prediction versus reference and show its horizon; no frame directory,
   video, or raw trajectory tensor is retained.

## Completion and next frontier

Phases A--M are implemented in order and remain separate regression tiers.
Before Phase L, no learned module was widened because every integrated failure
was owned by public geometry/state estimation or support association. Public
face-plane and algebraic sphere fits, bounded causal pose history, and
geometry-only support partitioning corrected those owning seams directly.
Phase J adds the permanent cross-tier regression envelope, and Phase K
establishes an exact analytic physical-adaptation oracle. Phase L adds learned
capacity only at that now-measurable evidence-fusion seam; it does not erase or
relabel the earlier failed experiments. Phase M extends that same compact model
to twelve seconds and makes its solver floor explicit.

Phase K closes analytic adaptive physical generalization at the same public
visual scale. Phase L then proves that one small transformer can learn the
same adaptation interface from varied episodes and generalize it to the real
public RGB-D ladder without an ensemble or trajectory-ranking loss. Phase M
then stabilizes its open-loop extrapolation through 12 seconds without adding
capacity or a planning loss. The next highest-impact frontier is to learn
residual state transitions only where the
existing truth-state ablations show systematic model error: begin with
event-local heterogeneous rigid contact, keep the learned correction bounded
and antisymmetric, and preserve the analytic solver as the zero-residual
fallback. Qualify that correction against the complete per-object and
cross-capability envelope before widening the transformer, moving RGB-D
qualification above N=8, or attempting end-to-end perception. Flush
featureless unions, unknown calibration, and identity recovery through
unobserved impulses or collisions remain explicit observability limits.

All earlier protected tiers remain separate compatibility gates, with isolated
fresh-process latency evidence for multi-contact and public visual execution.
Absolute timing remains separately adjudicated by those strict runners rather
than by a thermally contaminated aggregate process. Phase M is a governed
development promotion, not a protected historical qualification.
