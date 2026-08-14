# Changelog

## Unreleased — 2026-07-28

### 2026-08-13 paper-guided hypothesis selector

- Added `HypothesisRolloutEngine` and `HypothesisSelection` for parallel
  short-step candidate rollouts, masked uncertainty-aware error scoring, and
  deterministic per-batch model selection. The implementation is functional
  but has not yet been connected to a promoted training run.
- Added a structural adapter for any dynamics candidate exposing
  `predict_step`, with a regression test using two real `DynamicsModel`
  instances.
- Added `HypothesisDynamicsPool`, which carries normalized candidate weights
  across receding-horizon cycles and assimilates delayed masked observations;
  fixed-candidate selection and source-belief immutability are tested.
- Exposed the pool through `OnlineWorldModel.predict_hypotheses` and
  `assimilate_hypotheses`, preserving injected candidate ownership and the
  persistent-belief source-of-truth contract.
- Full suite after integration: `744 passed, 6 skipped`; skips are expected
  MPS-availability gates.
- Fixed candidate reporting to use the accumulated posterior argmax rather
  than the instantaneous error argmin; added a regression test for sequential
  evidence.
- Added a non-mutating `ConstantVelocityDynamics` baseline with optional
  exponential damping and uncertainty propagation for heterogeneous
  candidate-pool tests.
- Fixed its variance propagation broadcast so RGB rollouts retain the required
  `[B,T,N,D]` trajectory shape; the validator caught this in the first live
  RGB smoke before any metric was recorded.
- Corrected the regression test to exercise the structural `predict_step`
  adapter, then completed a real toy RGB smoke: learned candidate score
  `2.126778` versus damped constant-velocity `8.872571`, with candidate `0`
  selected. This remains a plumbing check, not broad validation.
- Added `scripts/evaluate_hypothesis_pool.py`, which evaluates learned versus
  constant-velocity hypotheses on RGB episodes with persistent-ID alignment.
  The two-episode toy report is retained under the timestamped `runs/` path;
  it is not a promotion or convergence result.
- The same harness completed a one-episode attention-scale RGB smoke (148
  queries) under `runs/20260813-220000-hypothesis-pool-attention-smoke/`; all
  queries selected the learned candidate. Fresh-random-weight RMSE is recorded
  as an execution check only.
- Added configurable `HypothesisDynamicsPool.evidence_decay` to support
  context adaptation without silently changing the persistent default; a
  regression test covers local model switching.
- Added `--evidence-decay` to the evaluation harness and retained a protected
  attention-scale decay-0.1 smoke. It switched to the baseline 4/148 times;
  this is adaptation evidence, not a broad accuracy qualification.
- Completed the four-episode protected decay matrix (592 queries). Decay 0.1
  increased local switching `5 -> 27` and slightly improved long-horizon mean
  RMSE without broad regression; it remains unpromoted pending full protocol
  evidence.
- Extended the hypothesis evaluation harness with lifecycle mismatch, identity
  coverage, collision precision/recall/F1, and selected uncertainty. Fixed a
  real RGB supervision bug by bootstrapping simulator-to-runtime IDs from
  nearest current world positions before carrying them forward.
- Completed an eight-episode protected decay-0.1 run (1,184 queries). The
  learned candidate remained dominant (`1,149/35` learned/baseline selections),
  selected RMSE improved slightly over learned-only at every horizon, and the
  baseline's collision F1 was zero. This supports guarded fallback, not model
  promotion or convergence.
- Started the required 32-episode protected decay-0.1 comparison at
  `runs/20260813-230000-hypothesis-pool-protected-32ep/` from the immutable
  reference checkpoint. MPS was tested but unavailable in the subprocess;
  the run is CPU-only and does not alter training artifacts.
- Batched multi-horizon evaluation to avoid recomputing the same learned
  rollout per horizon. A one-episode attention smoke completes in `103.78 s`
  CPU; the full protected run is being relaunched with the optimized harness.
- Completed the full 32-episode protected decay-0.1 comparison (4,736 queries).
  Selection improved every pooled position horizon slightly without broad
  event/lifecycle/uncertainty regression, but gains are too small for promotion
  or training restart. The immutable learned incumbent remains deployed.
- Added `BallisticContactDynamics`, an analytic gravity/drag candidate with
  conservative contact-event logits, and generalized the evaluation harness to
  three candidates. Nine focused tests pass; a fresh toy smoke selected the
  learned candidate on all 59 queries, so no gain is claimed.
- Completed the three-candidate protected eight-episode comparison (1,184
  queries). Ballistic was selected once and regressed mature event/position
  behavior; it remains diagnostic and unpromoted.
- Added composite position/lifecycle/collision evidence scoring. Event weight
  0.5 improved events but regressed mature y position and was rejected; a
  four-episode event-weight 0.1 screen is promising but remains unpromoted.
- Completed the lower-weight eight-episode event-aware validation. Collision F1
  improves across horizons, but mature position is not uniformly non-regressive;
  event weighting remains opt-in and the incumbent default is unchanged.
- Added three focused regression tests; all pass. The stopped relation-only
  campaign remains protected and unchanged.

### 2026-08-12 functional node-drift prior and selector-512 diagnosis

- Completed exact zero-node evaluations for both drift-regularized step-512
  donors. The cosine/constant relation paths score `0.342289`/`0.329317`
  against protected `0.321316`, fail 100/98 broad guardrails, and remain
  unsupported for promotion despite mixed y/velocity gains. Preserved the
  first stalled two-worker evaluator attempt and reran the identical protocol
  successfully with its default `num_workers=0` path.
- Qualified fresh constant-rate relation-only optimization with a real
  two-update exact-resume CPU smoke. The strict audit proves 46 changed
  permitted attention tensors and 46 complete Adam owners, two exact frozen
  node tensors with no optimizer state, 177 exact inherited tensors, exact
  protected state, and complete finiteness. This authorizes a sustained
  relation-first experiment from protected weights, not an accuracy claim or
  scale increase.
- Committed and pushed the relation-only decision as `c3fe110`, then launched
  the full 8,192-update/65,536-example campaign at
  `runs/20260813-073710-attention-relation-constant-stage-a/`. Replaced an
  initially provenance-incomplete supervisor source archive with a real
  detached Git worktree before any resume; exact commit, dirty state, runtime
  fingerprint, and worktree fingerprint now match the trainer. Both one-shot
  jobs are healthy with empty stderr while step-zero validation runs.
- Proved exact step-zero equality for the active relation-only campaign: all
  225 tensors and 2,584 metrics match the protected control. The first 16
  balanced updates apply with zero skips/failures, exact frozen-node isolation,
  and mixed near-control axis/horizon behavior; no accuracy claim is made.
- Fixed collapse-auditor attribution so an explicitly frozen relation-only
  node path cannot create a false severe-clip warning. Real relation-output
  clipping remains reported. Added a regression test; focused tests, Ruff,
  formatting, and diff checks pass.
- Continued the immutable relation-only campaign through step 64. All updates
  apply with exact eight-way repeats of the eight-scenario balanced batch,
  zero skips/failures, 2,462 cadence-recorded causal trajectories, exact zero
  node behavior, stable memory, and at least `0.812481` complete interaction-
  gradient retention. Exact matched current velocity, lifecycle/identity,
  0.10-second and 1.00-second pooled position improve, while current/middle-
  horizon position, x, and collision F1 remain adverse. No retuning or
  promotion precedes fixed validation.
  Focused supervisor/auditor tests pass `39 passed`; Ruff is clean.
- Preserved and strictly audited the active relation-only step-128 checkpoint.
  All 46 permitted tensors and Adam owners are live, both node tensors and all
  177 inherited tensors remain exact, protected checkpoints remain intact,
  and serialization/provenance hashes pass. The complete matched 72--128
  window is finite and balanced, improves current/short x and collision F1,
  but regresses pooled 0.25--1.00-second position; selector 512 remains the
  first promotion gate.
- Audited complete relation-only updates 136--192. All operational invariants
  pass with balanced support, no skips/uncontained clipping, at least
  `0.725199` complete-gradient retention, and flat memory. Exact matched
  current and every pooled position horizon regress mainly on x/z, with most
  secondary slices adverse. Recorded this as real accuracy warning without
  conflating it with collapse or changing the fixed-selector protocol.
- Preserved and strictly audited the relation-only step-256 checkpoint with
  exact 46-trainable/2-frozen attention scope, complete Adam state, unchanged
  inherited/protected state, and finite provenance. The matched 200--256
  window repairs current/velocity/lifecycle/event and short-horizon position,
  but x and pooled 0.50--1.00-second position remain adverse; fixed validation
  remains authoritative.
- Audited complete relation-only updates 264--320. The run remains balanced,
  finite, support-complete, and safely clipped. Matched x repairs at mature
  horizons and pooled 1.00-second position improves, while an increasing z
  regression becomes the remaining axis limitation. Identity/NLL improve and
  lifecycle/event/velocity remain mixed; continue to fixed validation.
- Preserved and strictly audited relation-only checkpoint 384 with exact
  trainable/frozen/Adam/inherited/protected scope and finite provenance. The
  matched 328--384 window regresses current and every pooled position horizon,
  mainly x, despite improved velocity, aggregate event, lifecycle, identity,
  and coverage. Recorded the renewed axis migration as wobble, not collapse.
- Audited complete relation-only updates 392--448. This is the strongest
  sampled window so far: current position improves across all axes, every
  pooled horizon and current velocity improve, and complete interaction
  retention stays `1.0`. Event F1/lifecycle remain adverse, so the result is
  not promoted ahead of the fixed step-512 selector.
- Rejected the relation-only step-512 candidate despite its better aggregate
  score (`0.305413` versus protected `0.321316`). Mature pooled horizons and
  all x horizons improve, but current/short-z accuracy, lifecycle coverage and
  precision, collision F1, identity, and 109 broad guardrails regress. The
  strict step-512 audit passes exact 46-trainable/2-frozen scope, 177 inherited
  tensors, 46 Adam owners, protected state, provenance, and finiteness. A
  premature copy of step-384 `last.pt` was detected by embedded step/hash and
  quarantined; it is not selector evidence. The mutable run continues while
  the protected deployment incumbent remains step zero.
- Added an opt-in expected-step assertion to the attention-checkpoint auditor.
  The true selector artifact passes `512/512`; the prematurely copied artifact
  now fails explicitly as embedded step 384 rather than allowing a misleading
  step-512 report. Focused tests pass (`4 passed`) and Ruff is clean.
- Published specification 1.43 and hardened attention audits against stale or
  mixed Adam boundaries. Non-empty optimizer steps must now exactly equal the
  embedded checkpoint step. A synthetic 128/127 artifact fails and the real
  relation-only selector passes payload/expected/Adam `512/512/[512]`.
- Audited complete relation-only updates 520--576 after the rejected selector.
  All operational gates pass with 2,650 trajectories and at least `0.673214`
  complete-gradient retention. Matched current state/velocity, short position,
  aggregate event, identity, coverage90, and uncertainty improve; mature x
  increasingly regresses, producing a `+0.013616 m` 1-second pooled deficit
  and adverse 1-second collision F1. Continue unchanged to selector 1024.
- Preserved and strictly audited relation-only checkpoint 640. Embedded,
  expected, and Adam steps agree; exact trainable/frozen/inherited/protected
  scope and finiteness pass. Complete matched updates 584--640 improve current
  state/velocity, every pooled position horizon, and every x horizon. A raw
  step-616 force spike is safely isolated before shared backprop; collision F1
  and mature velocity remain the accuracy watch items.
- Published specification 1.44 and fixed `--after-step` to be strictly
  exclusive for training, validation, and reference rows. This prevents
  adjacent windows from double-counting their boundary. The auditor plus
  specification-version suite passes (`18 passed`); corrected post-640
  monitoring begins at step 648.
- Recorded the complete relation-only 648--704 dynamics audit: 64 applied
  updates, exact eight-way balance, 2,130 causal trajectories, no skips or
  uncontained clipping, and flat memory. Different-draw trends improve mature
  position/event/lifecycle/calibration but regress current/short state,
  velocity, identity, x, and z, so selector 1024 remains authoritative.
- Preserved and strictly audited relation-only checkpoint 768 with exact
  payload/Adam/scope/inherited/protected/provenance integrity. Complete
  712--768 dynamics are operationally healthy and improve every position
  slice versus the preceding different-draw window; selected velocity horizons
  and mature target coverage remain diagnostic limitations before selector
  1024.
- Recorded the complete 776--832 relation-only dynamics window. Operational
  integrity passes with 2,578 causal trajectories and flat memory; forecast
  support improves, while most accuracy/calibration slices reverse on
  different draws. No promotion or protocol mutation is authorized before the
  fixed step-1024 selector.
- Preserved and strictly audited relation-only checkpoint 896 and complete
  840--896 dynamics. Structural and operational integrity pass; isolated
  hard-contact force sensitivities are contained. The final pre-selector
  different-draw trend improves velocity/identity/calibration but regresses x,
  every position horizon, events, and mature lifecycle support.
- Recorded complete relation-only updates 904--960: operational integrity
  passes with 2,943 trajectories and all causal terms. Different-draw evidence
  is horizon-dependent, improving mature forecasts/events/lifecycle while
  regressing current/short state, identity, coverage90, and calibration.
- Completed and rejected the relation-only step-1024 fixed selector. Candidate
  score is `0.3409900` versus protected `0.3213162`, with 116 guardrail
  failures and no support failures. The regression is concentrated in current
  and short-horizon x, especially `reference_pairs`; velocity and collision F1
  improve but do not offset broad state/lifecycle/identity failures. The
  protected step-zero incumbent remains exact and unmodified.
- Strictly audited `validation_step_001024.pt`: payload/expected/Adam step,
  46-trainable/2-frozen scope, 177 inherited tensors, protected checkpoints,
  provenance, protocol, architecture, and finiteness all pass. Complete
  updates 968--1024 also pass balance, support, clipping, optimizer, and memory
  gates. This rules out corruption while preserving behavioral
  non-convergence evidence; the declared mutable 8,192-step run continues.

- Rejected the warmup/cosine control at its complete 32-episode step-512
  selector: score `0.3475480` versus protected `0.3213162`, 116 broad guardrail
  failures plus failed improvement, zero support failures, and regression at
  every pooled position horizon. Familiar `reference_pairs` current x reaches
  `0.720231 m`. A strict audit proves all 48 attention tensors/Adam states live,
  all 177 inherited tensors and both protected checkpoints exact, and all
  serialized state finite.
- Added specification-1.42 relation-first qualification. The new
  `attention_relation` scope trains shared typed tokens/blocks and relation
  outputs while freezing the two node-decoder tensors. The attention auditor
  now supports explicit frozen prefixes and exact expected tensor/optimizer
  coverage. Focused tests pass (`270 passed in 20.18 s`), the complete suite
  passes (`736 passed, 6 skipped in 211.71 s`), and Ruff/format checks are
  clean.
- Kept every capacity rung closed. The managed environment denied launchd
  unload after exhausting its approval quota, so the externally running
  rejected job must be stopped before a zero-node ablation or relation-first
  campaign consumes compute.

- Completed the drift successor's authoritative 32-episode step-512 selector
  and rejected it at score `0.3332533` versus protected `0.3213162`, with 105
  broad guardrail failures.  `reference_pairs` current x worsens
  `0.242694 -> 0.732948 m` and every x horizon regresses, disproving the
  favorable interpretation of late heterogeneous training windows.
- Strictly audited and preserved the rejected step-512 artifact: all 48
  attention tensors changed, all 177 inherited tensors and both protected
  artifacts remain exact, complete attention-only Adam state is at step 512,
  and all serialized values are finite.  Stopped both one-shot launchd jobs at
  the durable selector boundary.
- Recorded ADR-112 and kept capacity scaling gated.  Authorized the already
  implemented same-capacity schedule control from the untouched graph
  checkpoint with 384 warmup updates, 8,192 fixed cosine-decay updates, and a
  0.1 floor; rejected attention weights may not seed it.
- Committed and pushed the rejection as `1926547`, then launched the clean
  same-capacity successor at
  `runs/20260812-155706-attention-node-drift-warmup-cosine-stage-a/` with an
  MPS trainer and exact-source supervisor.  Its 32-episode step-zero selector
  exactly reproduces score `0.3213162196` with zero guardrail/support failures;
  a cross-run audit proves all 225 tensors equal and both stderr files remain
  empty.  Step 8 then logs the exact warmup rate `1.0416667e-6`, a finite
  wholly unclipped applied update, all eight scenarios, 349 trajectory targets,
  and zero skips.  This qualifies initialization/schedule execution only, not
  learned accuracy.
- Audited the schedule successor's complete updates 8--64 window. All 64
  updates apply with balanced eight-way exposure, 2,462 trajectory targets,
  zero skips/failures, minimum complete interaction retention `0.497461`, and
  stable memory. An exact same-draw comparison with the rejected constant-rate
  run is mixed and nearly neutral: current position, lifecycle, and identity
  improve slightly while velocity and four longer position horizons worsen
  slightly. The run remains immutable and unpromoted through selector 512.
- Strictly audited the schedule successor at durable step 128. All 48 attention
  tensors changed, all 177 inherited tensors remain exact, only the 48
  attention parameters own complete finite Adam state, and protected artifacts
  remain intact. The complete matched 72--128 window is operationally healthy
  but worse on current/every position horizon, mainly x, with mixed velocity/y
  behavior and adverse lifecycle/event/identity slices. Continue unchanged to
  the authoritative selector; this is neither promotion nor a scale trigger.
- Audited complete scheduled updates 136--192. Support, balance, optimizer,
  uncertainty, semantic gradient containment and memory remain healthy, with
  minimum shared retention `0.804336`. Exact matched position remains adverse
  while velocity, identity and uncertainty improve; the run has only 25.13%
  of constant cumulative LR exposure at this warmup boundary and continues
  unchanged to fixed selector 512.
- Strictly qualified the scheduled step-256 checkpoint and complete matched
  200--256 window. Tensor/Adam/protected-state integrity passes; the current
  position gap narrows to `+0.005475 m`, but x and every position horizon
  remain adverse. Deterministic emitted y bias falls from the constant-rate
  step-256 `0.195037` to `0.128556 m/s²`, while drift still dominates context
  variation. Preserve the immutable run to selector 512.
- Audited complete scheduled updates 264--320. Exact current/short position is
  now nearly tied with constant rate while current/short velocity, identity,
  collision F1 and coverage90 improve; long position, selected axes, lifecycle
  and median NLL remain adverse. The prior hazardous update 280 is safely
  contained. Continue through warmup completion and selector 512.
- Completed linear warmup at durable step 384 and passed strict tensor/Adam/
  protected-state audit. Exact current and 0.10-second position plus lifecycle
  improve versus constant rate, while 0.25--1.00-second position and mixed
  velocity/event/calibration slices remain adverse. Continue unchanged to the
  fixed selector.
- Audited the first complete cosine-phase updates 392--448 window. All support,
  update, memory, and finiteness checks pass; a step-424 raw-gradient spike is
  contained at `0.126551` shared retention and does not recur. Exact-draw
  identity/lifecycle/event metrics improve slightly, but current position,
  velocity, coverage90, and nearly all position horizons remain adverse. Keep
  selector 512 and every capacity rung gated.

- Added an opt-in, state-free `warmup_cosine` closed-loop learning-rate
  protocol while preserving exact historical `constant` behavior. Warmup and
  cosine durations use absolute causal update indices and never depend on the
  extensible total-step budget; schedule changes are exact-resume
  incompatibilities. A real CPU smoke and exact resume logged the expected
  `0.0002` second warmup rate and `0.0001100000` first cosine rate at
  `runs/20260812-123215-lr-schedule-smoke/`. This is implementation evidence,
  not accuracy qualification, and it does not alter the active pinned run.
  The complete repository suite passes with `732 passed, 6 skipped`; the skips
  are the expected MPS-unavailable device cases in the restricted test process.
- Audited the active drift campaign's complete steps 136--192 window. It is
  operationally healthy and now improves current and 0.10/0.25-second position
  versus the predecessor, but still regresses velocity and 0.50--1.00-second
  position. Preserve the run to selector 512 without promotion or mutation.
- Repeated the operational and integrity audit at the durable step-256
  boundary. All 48 attention tensors and exactly their 48 complete Adam owners
  changed, all 177 inherited tensors and both protected incumbents remain
  exact, and every serialized value is finite. The complete matched 184--240
  window again improves current/short position while regressing every velocity
  horizon and 0.50--1.00-second position. Same-draw calibration measures a
  finite but still drift-dominated mean node acceleration of
  `[-0.052641, 0.195037, -0.021283] m/s²`; this is non-promotable limitation
  evidence pending selector 512, not collapse.
- Audited the first complete post-checkpoint window at updates 208--264. It
  remains operationally clean, and node drift decreases while context
  variation increases, but matched current/every-horizon position has returned
  to small regressions and four of five velocity horizons remain worse. This
  is explicit wobble evidence; the immutable run continues to its fixed
  selector rather than being promoted, stopped, or retuned from training rows.
- Added deterministic functional-prior gradient alignment for the raw physical
  task, unit drift prior, and configured total objective, including full-module
  and node-decoder norms/cosines with correct unused/zero-gradient semantics.
  Two exact step-256 draws alternate between restoring and conflicting
  directions, ruling out a fixed sign bug and localizing the remaining wobble
  to stochastic task/prior conflict under the constant rate. The focused tests
  pass (`4 passed`) and the complete suite passes (`734 passed, 6 skipped`),
  with Ruff, format, compileall, and diff checks clean.
- Audited complete updates 264--320. Operational health remains clean and the
  candidate improves materially against its own preceding window, but the
  matched predecessor comparison still splits at 0.25/0.50 seconds and every
  velocity horizon regresses. Selector 512 remains authoritative.
- Audited complete updates 296--352. The operational gates still pass and the
  candidate now improves predecessor-matched 0.50--1.00-second position,
  current/through-0.75-second velocity, collision F1, and identity. Small
  current/0.10-second position, 1.00-second velocity, and lifecycle regressions
  remain, so this is a promising horizon shift rather than promotion evidence.
- Strictly qualified the durable step-384 checkpoint and complete updates
  328--384. Integrity, optimizer ownership, support, and resources remain
  clean. The newest window improves current/mature position, current and most
  velocity horizons, every collision horizon, identity, and lifecycle
  precision; short position, 0.10-second velocity, y/z current axes, and median
  uncertainty NLL remain mixed. Adjacent gradient-alignment draws reproduce
  the alternating conflict while x residual bias shrinks and y bias persists.

- Reached and strictly audited the specification-1.36 step-1024 checkpoint.
  All 48 attention tensors changed, all 177 inherited tensors remain exact,
  exactly 48 finite Adam owners are at step 1024, both protected artifacts are
  unchanged, and all source/configuration/protocol/model hashes pass.
- Rejected the step-1024 selector by 111 broad guardrails.  The scalar score is
  effectively flat (`0.3213162 -> 0.3212919`), but current position, x,
  coverage, precision, and the two shortest horizons regress.  The familiar
  `reference_pairs` current x error more than doubles
  (`0.242694 -> 0.573947 m`) and all its x horizons worsen.
- Audited the complete 904--960 training window: all updates/objectives remain
  supported, minimum interaction retention is `0.514307`, RSS is flat, and no
  numerical or optimizer failure exists.  Classified the selector failure as
  behavioral overfit and stopped both one-shot jobs at the durable boundary.
- Recorded ADR-109: do not resume or scale from the rejected weights.  Launch
  the already smoke-qualified specification-1.39 drift successor from the
  untouched protected control with `complexity=1.0`, `drift=0.08`, unchanged
  broad selectors, and the full convergence budget.
- Refreshed the scaling review against the original Transformer,
  compute-optimal scaling, Qwen3, DeepSeek-V3, and V-JEPA 2.  The next useful
  experiment is the repaired small-rung learning curve; GQA/MLA/MoE/local
  attention solve absent long-context or cluster bottlenecks and do not
  justify bypassing the fixed physical-generalization gate.
- Committed and pushed the step-1024 rejection as `176796f`, then launched the
  clean specification-1.39 successor at
  `runs/20260812-102557-attention-node-drift-008-stage-a/`.  Its one-shot MPS
  trainer and exact-source convergence supervisor use Standard launchd
  scheduling, `KeepAlive=false`, and the unchanged 8,192/24,576 convergence
  budget.  Both stderr files are empty.
- Completed the successor's mandatory 32-episode step-zero selector in
  `991.16 s`.  It exactly reproduces the protected score `0.3213162196` and
  all recorded current/horizon, axis, coverage, precision, event, identity,
  and calibration evidence with zero guardrail/support failures.  This clears
  initialization integrity; it is not trained accuracy or convergence.
- Audited the first 16 balanced updates against the complexity-only
  predecessor on identical seeds.  All updates apply with 511 causal
  trajectories, no skip/failure, every scenario twice, nonzero drift loss, all
  13 mature terms by update 16, and minimum complete interaction retention
  `0.615309`.  Early physical differences are millimetric and mixed, so fixed
  selector 512 remains the first accuracy authority.
- Extended that matched audit through update 24: 907 causal trajectories,
  every scenario three times, no schedule/support mismatch, no terminal or
  uncontained gradient failure, and only millimetric mixed position changes.
  The rising heterogeneous per-block scalar is therefore not classified as
  collapse; fixed selector 512 remains the first generalization authority.
- Extended the matched audit through update 32: all eight scenarios contribute
  four times, 1,175 causal trajectories are present, current and short-horizon
  position improve slightly, and long horizons remain within `0.000167 m` of
  the predecessor.  Sparse collision F1 is lower and remains a watch item;
  neither that discrete training sample nor the raw scalar can supersede the
  fixed selector.
- Completed the first eight-block dynamics window through update 64.  All
  updates apply with 2,462 causal trajectories, balanced scenarios, healthy
  complete-gradient retention, and no failure.  Current/short-horizon error is
  slightly better while 0.50--1.00-second aggregate regressions remain below
  `0.001 m`; squared y drift nevertheless reaches `0.011169`.  Durable step
  128 is therefore a strict functional-drift checkpoint, not an automatic
  continuation claim, and selector 512 remains the accuracy authority.
- Strictly audited the successor's durable step-128 checkpoint. All 48
  attention tensors changed, all 177 inherited tensors remain bitwise exact,
  exactly 48 finite Adam owners are at step 128, both explicitly supplied
  protected checkpoints remain initial, and source/config/protocol/model/file
  hashes pass. The report is
  `attention_checkpoint_audit_step_000128.json` beside the run.
- Closed the exact schedule-matched updates 72--128 window with 2,586 causal
  trajectories, eight draws from every scenario, all 13 objectives, no skips
  or failed updates, minimum interaction retention `0.391475`, and flat RSS.
  The window regresses current position `12.76 mm` and every horizon
  `8.28--15.94 mm` versus the already rejected predecessor, while collision F1
  and identity improve. This is adverse training evidence, not fixed-selector
  rejection.
- Measured the step-128 residual on the same balanced draw used for the
  predecessor calibration. RMS emitted node acceleration falls to
  `0.106243 m/s^2` from `0.206605 m/s^2`, and mean y acceleration falls to
  `0.171365` from `0.356690 m/s^2`, proving the drift prior operates. More
  than 99.95% of the remaining activity is still context-invariant. Continue
  the immutable constant-rate campaign to selector 512; if it rejects, test a
  separately versioned warmup/decay successor before increasing capacity.
- Corrected the emitted-acceleration calibration population: its forward hook
  had included prepared no-gradient attention calls that the differentiable
  activity/drift records intentionally exclude. It now traces exactly 144
  gradient-enabled causal calls and 5,184 active-object evaluations on the
  step-128 draw, asserts agreement with the loss records, and records the trace
  scope explicitly. The activity, drift, complexity, and RMS results are
  unchanged; the corrected mean is
  `[-0.065046, 0.171365, -0.015823] m/s^2`. Focused diagnostics pass 42 tests;
  the full suite passes 721 tests with 6 expected MPS-unavailable skips in the
  command environment. Ruff, compileall, and diff checks pass.

- Strictly audited the live specification-1.36 step-512 checkpoint. All 48
  attention tensors changed, all 177 inherited tensors remain exact, exactly
  48 finite Adam states are at step 512, protected checkpoints remain initial,
  and hashes/protocol agree. The candidate score improves
  `0.3213162 -> 0.3177418`, but it is rejected by 109 guardrails: current
  position/velocity and short horizons regress, with a severe
  `reference_pairs` failure. This is learned behavior, not corruption or
  optimizer collapse.
- Ran fixed-manifest typed ablations. Halving both decoders and removing force
  rows are harmful. Zero-y improves all five horizons but still fails 97
  guardrails. Zeroing the complete node decoder is strongest at score
  `0.297330`, improves all five horizons, and reduces failures to 72 while
  remaining non-promotable. The relation branch is useful; functional node
  acceleration is the main localized error.
- Advanced the contract to specification 1.38 and added opt-in
  `attention_node_activity`: active-object-weighted mean squared bounded node
  acceleration accumulated over the causal rollout, with equal x/y/z
  diagnostics and no persistent/inference state. Omitted historical weights
  contribute exactly zero. Focused tests pass (`4 passed`); the full suite
  passes (`718 passed, 6 skipped in 212.36 s`). The immutable 1.36 run remains
  live and unchanged toward selector 1024; no capacity promotion is claimed.
- Audited the complete matched steps 520--640 window. All 16 balanced blocks
  apply with `4,678` causal trajectories, no terminal/resource failure, and
  minimum interaction retention `0.198063`. Relative to the unregularized
  predecessor, current and 0.10-second position improve `9.62/7.52 mm`, while
  later horizons are neutral/slightly adverse by at most `1.06 mm`; most
  velocity horizons, collision, identity, and lifecycle improve slightly, but
  current velocity regresses `0.0189 m/s`. Continue to selector 1024 without
  treating this heterogeneous training window as promotion evidence.
- Added `scripts/measure_attention_node_activity.py`, a deterministic,
  checkpoint-hashed one-balanced-draw calibration of emitted acceleration and
  functional/parameter-prior gradients. On the rejected step-512 candidate it
  reproduces activity `0.042669`, RMS acceleration `0.206565 m/s²`, and a
  `0.673351` functional gradient versus `0.052798` for unit complexity. The
  derived equal-gradient weight is `0.078411`; record `0.08` for a successor
  rather than guessing `1.0`. Ruff, format, compile, real execution, and the
  exact 8,192-update dry-run pass.
- Repeated calibration on four balanced draws spanning step indices
  127/255/383/511. Activity stays within `0.042484--0.042695` and the
  equal-gradient weight within `0.078292--0.078442`, ruling out single-window
  tuning. Extended tracing over `10,182` active-object invocations shows mean
  acceleration `[-0.024866, 0.356690, 0.006175] m/s²` with only
  `[0.000736, 0.001865, 0.000746]` standard deviation. More than 99.997% of
  total activity is squared mean drift.
- Advanced the contract to specification 1.39 and separated
  `attention_node_drift` from total activity and residual variation. The
  successor will use axis-neutral drift weight `0.08`, allowing balanced
  contextual/event variation while discouraging the observed scene-wide
  force. Focused tests pass (`8`), full regression passes
  (`719 passed, 6 skipped in 203.45 s`), Ruff passes, and the exact drift dry
  run resolves. The live immutable 1.36 run is unchanged.
- Completed a real two-update, exact-resume protected-control CPU smoke of the
  prospective `complexity=1.0`, `drift=0.08` objective. Update two applies all
  13 causal terms with 343 trajectories and records nonzero activity/drift/
  variation of `4.98557e-6/4.97644e-6/9.12632e-9`. Its strict checkpoint audit
  passes with all 48 attention tensors and only those tensors changed, 177
  inherited tensors exact, 48 complete Adam owners at step two, finite state,
  and exact protected reference. The eight-episode smoke remains explicitly
  `last_unvalidated` and makes no accuracy claim.
- Audited the live immutable campaign through logged update 728. Every update
  is applied, all eight scenarios have 12 equal draws, RSS is flat near
  `2.913 GB`, and there is no terminal failure, skipped draw, or uncontained
  interaction clip. The complete 640--696 balanced window passes; the later
  704--728 window is only half complete and is not treated as trend evidence.
  The durable step-640 checkpoint also passes exact tensor, optimizer,
  finiteness, protected-state, hash, and protocol audit.
- Completed the next full steps 704--760 balanced audit and durable step-768
  checkpoint audit. All eight blocks apply with equal eight-scenario draws,
  2,115 supported trajectories, all 13 causal terms, minimum complete-gradient
  retention `0.269683`, identity `3/300`, and no terminal/resource failure.
  Every horizon and axis is recorded in the audit; the pooled 0.10--1.00-second
  position curve is `0.201204/0.227514/0.272258/0.311915/0.337189 m`. Step 768
  has all 48 attention tensors changed, 177 inherited tensors exact, exactly
  48 finite Adam owners at step 768, and both protected artifacts exact.
- Corrected two project-status command examples from obsolete singular
  checkpoint-auditor flags to the executable's actual repeatable `--protected`
  and plural `--require-protected-checkpoints` interface, and included the
  strict changed-tensor and complete-optimizer-state gates used by the audit.

### 2026-08-11 accumulated node-gradient repair and scale gate

- Advanced the training contract to specification 1.36 after the latest
  persisted step-512 selector and exact typed ablations localized a broad
  generalization failure to the attention node-y decoder row. Added an opt-in,
  axis-neutral `attention_node_complexity` objective equal to mean squared L2
  decoder-row energy, including bias, plus per-axis diagnostics. Historical
  configs omit it exactly; no forward equation, tensor shape, runtime rule, or
  axis capability changes. Focused schedule/objective/config/checkpoint tests
  pass (`312 passed`).
- Preserved the live YAML unchanged for exact continuation. The repair will be
  enabled only by a separately recorded override at weight `1.0`. At the
  rejected checkpoint x/y/z row energies are
  `0.000154/0.012417/0.000146`, so the prior targets the disproportionate row
  without a y-specific rule. Capacity scaling remains blocked pending a fresh
  protected-control campaign and fixed-manifest qualification.
- Intentionally stopped the rejected specification-1.35 trajectory at its
  durable step-640 checkpoint after the next equal training window reversed
  the temporary post-selector improvement. Booted out its trainer and
  supervisor; the only shutdown stderr is the expected multiprocessing
  semaphore cleanup warning. The protected deployment incumbent remains step
  zero.
- Passed final implementation gates: `716 passed, 5 skipped, 1 deselected`
  non-device; five direct host-MPS tests; Ruff format/check; compileall; diff
  check; and an 8,192-update/65,536-draw dry run with the explicit complexity
  override.
- Committed and pushed specification 1.36 as `bbdb3ad`, then launched the
  fresh repaired campaign at
  `runs/20260811-234157-attention-node-parsimony-stage-a/`. Trainer and
  immutable-commit convergence supervisor are running once with empty stderr.
  Metadata confirms clean source, MPS RGB measurement, CPU closed loop, no
  oracle, and the exact `attention_node_complexity=1.0` override. Initial
  32-episode protected-control validation completed in `1,034.57 s`: all
  `225/225` tensors and `2,844/2,844` common non-resource metrics exactly match
  the protected control at score `0.3213162196`.
- Audited the first eight regularized updates. Exact predecessor seeds,
  scenarios, and 349-trajectory support match; all updates apply with no skip,
  zero trusted identity switches, complete interaction retention `1.0`, and
  applied gradient `0.254750` versus predecessor `0.283628`. The whole-run
  auditor passes. This qualifies launch/objective wiring only; no trained
  selector or convergence result exists yet.
- Audited the complete step-8--64 matched training window. All 64 updates
  apply with exact scenario balance, `2,461` causal trajectories, no skips or
  terminal failures, minimum complete-interaction retention `0.566722`, stable
  `2.891 GB` sampled RSS, and empty trainer/supervisor stderr. Relative to the
  unregularized predecessor, pooled position and velocity are slightly worse,
  every position horizon changes by less than `0.0008 m`, and trusted identity
  has one extra switch. This is recorded as an early trend warning; fixed
  selector 512, not the sampled window, remains the accuracy and scale gate.
- Persisted and passed the step-128 attention checkpoint audit. Every one of
  48 attention tensors changed, all 177 inherited tensors remain exact,
  complete finite Adam state belongs to exactly the 48 attention parameters at
  step 128, protected checkpoints remain exact, and all recorded hashes agree.
  The matched 128-update dynamics audit also passes with equal scenario draws,
  `5,047` causal trajectories, full horizon support, minimum interaction
  retention `0.373366`, and stable `2.911 GB` sampled RSS. Current and every
  pooled position horizon improve versus the predecessor, but identity rises
  from `4/699` to `9/703`, 0.25/0.50-second velocity regresses, and collision
  F1 is slightly lower. Continue to fixed selector 512 without promotion.
- Cleared the matched post-checkpoint step-152 stress boundary with all 13
  objectives, 343 causal trajectories, `0/50` trusted identity switches, and
  an entirely unclipped finite update. Across steps 128--152, every position
  axis at every horizon improves and lifecycle support is slightly better;
  current and 0.10-second velocity plus aggregate collision F1 regress while
  one-second velocity improves. The step-128 identity spike does not persist
  in later matched blocks. Keep the run unchanged and selector 512
  authoritative.
- Audited the live trajectory through sampled step 208. The apparent step-184
  position spike is a short-only batch with a changed ungated matched-frame
  set, not fixed-manifest evidence. The following matched 192/200 blocks
  improve current position and all five position horizons with equal identity
  switches and much healthier interaction retention. Adding step 208 leaves a
  small 0.50-second/collision/lifecycle tradeoff, while the auditor still
  passes with complete support, applied updates, stable memory, and empty
  stderr. No capacity promotion is claimed before selector 512.
- Refreshed the primary-source architecture review against the original
  Transformer, compute-optimal scaling, Qwen3, Gemma 3, DeepSeek-V3, and
  V-JEPA 2. The existing dense pre-RMSNorm/SwiGLU typed-set attention and
  evidence-gated data/depth/width/history ladder remain the appropriate path;
  long-context or cluster-efficiency mechanisms are deferred until a measured
  bottleneck requires them. No live protocol or model tensor was changed.
- Reached durable live checkpoint step 256. Its independent audit proves all
  48 attention tensors changed, all 177 inherited tensors remain exact,
  exactly 48 finite Adam states are at step 256, both protected checkpoints
  equal the initializer, and every provenance/hash check passes. Matched
  128--256 training evidence improves long horizons and several guardrails but
  regresses current x/z, short horizons, and identity, so selector 512 remains
  authoritative and no larger rung is launched.
- Advanced the offline evidence contract to specification 1.37 after an audit
  invocation with no protected arguments exposed a vacuous `true` protection
  result. The auditor now records protected count, emits `null` when protection
  was unchecked, and can fail an omitted protected set. The corrected step-256
  audit requires and verifies both protected artifacts; focused tests and Ruff
  pass. The immutable live trainer remains truthfully specification 1.36.
- Audited the first complete post-step-256 matched window through step 312.
  Every update applies with balanced support, finite gradients, stable memory,
  no terminal artifact, and contained interaction clipping. Current and long-
  horizon position plus x at every horizon improve, while short-horizon
  position, current velocity, most y horizons, aggregate collision F1, and
  median uncertainty regress. The result is recorded as healthy but mixed;
  fixed selector 512 remains the promotion and scaling gate.
- Reached and strictly audited durable residual-parsimony checkpoint step 384.
  Attention-only tensor and Adam scope, finiteness, provenance, and both
  protected initializer checkpoints all pass. The matched 328--384 window is
  operationally healthy but regresses current/every-horizon position through
  x while y/z, velocity, collision, and lifecycle improve. No axis-indexing or
  selector implementation defect was found; scaling remains blocked through
  fixed selector 512.

- Stopped the specification-1.31 campaign after the trainer safely rejected
  attempted optimizer step 988 before Adam. Complete interaction retention was
  `0.0971759`, below the declared `0.1` minimum; the supervisor detected the
  durable terminal artifact and did not relaunch. No step-1024 selector or
  convergence result exists.
- Localized the missing isolation level to recursive accumulation into shared
  attention. Across 144 calls, nominal `0.1` per-invocation caps allowed
  aggregate force/impulse output norms `0.219855/0.115811`; normal-force and
  maximum shared gradients reached `10.9076/5.01609` despite later decoder-row
  clipping.
- Advanced the contract/runtime to specification 1.35 and made semantic output
  caps aggregate per-optimizer-draw L2 budgets. Each of `K` registered calls
  receives `cap / sqrt(K)`; one-call behavior is exact. Added multi-invocation
  regression coverage; the focused attention/config/checkpoint/auditor suite
  passes (`314 passed`). Matched replay from durable step 896 remains pending.
- Completed the exact-state step-896-to-988 diagnostic replay on MPS RGB/CPU
  closed loop. It reproduced the failed draw's seeds, scenario order, 284
  causal trajectories, and all 13 objective terms. Raw/post-row interaction
  norm fell `15.1704/10.2906 -> 4.58029/1.54333`, complete retention rose
  `0.0971759 -> 0.647948`, and maximum shared gradient fell
  `5.01609 -> 0.225929`; every aggregate semantic norm stayed below `0.1`.
  The ordinary assertion passed and the harness stopped deliberately before
  Adam. Its report is `aggregate_gradient_replay_report.json` in the replay
  run; the generic auditor truthfully returns expected failure because a
  deliberate stop still uses the durable terminal-artifact channel.
- Passed the complete post-repair gates: `711 passed, 5 skipped, 1 deselected`
  non-device; five direct host-MPS regressions; Ruff format/check across 193
  files; compileall; diff check; and the attention-pilot dry run. The restricted
  device-marker worker has one expected MPS-unavailable skip, while the direct
  host tests and replay metadata prove actual MPS execution.
- Committed/pushed the replay evidence as `23ecf9d` and launched the fresh
  specification-1.35 campaign at
  `runs/20260811-170842-attention-aggregate-isolated-stage-a/` weights-only from
  the protected graph control. Trainer and immutable-source supervisor are
  active once with empty stderr under the declared 8,192/4,096/24,576
  convergence envelope. Initial fixed validation completed in `978.263 s`
  with all `225/225` tensors and `2,584/2,584` metrics exactly equal to the
  prior protected selector, including score `0.3213162195855908`; the first
  balanced optimizer block is active. No trained accuracy, convergence, or
  scale claim exists.
- Audited the first logged balanced block at step 8. All eight optimizer
  updates apply with exact eight-scenario balance, `349` causal trajectories,
  zero skipped draws, interaction retention `1.0`, applied gradient `0.283628`,
  and `0.489052` loss. The whole-run dynamics auditor passes without failure,
  duplicate, uncontained interaction clip, or terminal artifact. Its severe
  typed-output warning reflects the declared aggregate node/force budgets;
  the complete interaction update is not clipped. Peak sampled RSS is
  `2,936,651,776` bytes. This is health evidence, not convergence.
- Extended the live audit through step 32: all 32 updates apply, every scenario
  has four logged draws, minimum complete interaction retention is `0.716607`,
  no skip/duplicate/terminal/uncontained-interaction failure occurs, and RSS
  remains within `2.937--2.970 GB`. Matched step-24/32 losses stay close to the
  predecessor while repaired gradient norms fall from `4.8887/1.8296` to
  `0.8720/0.4513`. The incomplete training window is not a fixed-selector
  accuracy result.
- Completed the first eight-block step-64 dynamics audit. It passes with 64
  applied updates, exact eight-way scenario balance, `2,461` trajectories,
  zero skipped/duplicate/terminal/uncontained failures, minimum interaction
  retention `0.585590`, and bounded `2.992 GB` RSS. The complete training
  window records current position RMSE `0.267191 m`, 0.1--1.0-second RMSE
  `0.264288/0.303162/0.364353/0.420610/0.443199 m`, identity-switch rate
  `0.7772%`, coverage90 `96.60%`, collision F1 `0.176`, and median uncertainty
  NLL `-0.84906`. These are health diagnostics, not fixed validation.
- Audited the durable specification-1.35 step-128 checkpoint. All 128 updates
  apply with exact 16-per-scenario balance and no skip/duplicate/terminal/
  uncontained failure. All 48 attention tensors change, all 177 inherited
  tensors remain exact, exactly 48 attention parameters own finite step-128
  Adam state, and RSS stays at `2.992 GB`. The matched 72--128 training window
  slightly regresses every position horizon plus lifecycle/uncertainty while
  improving identity and short/mid velocity, so it is retained as a trend
  warning without promotion. Clarified the intentional cadence: checkpoints
  occur every 128 updates, while fixed 32-episode selectors occur every 512;
  no step-128 selector was expected.
- Completed the matched steps 128--184 training-window audit. Long position
  improves at 0.5/0.75/1.0 seconds and identity switches fall from 9/386 to
  2/380, but collision F1 falls `0.242775 -> 0.210526`, lifecycle weakens, and
  current/short position regresses slightly. Optimizer/support/memory health
  remains clean. A step-136 force-row spike is contained with `0.23550`
  complete retention and does not immediately recur. Retained the result as a
  tradeoff without promotion or protocol mutation pending fixed step 512.
- Audited the durable step-256 checkpoint: all 48 attention tensors changed,
  all 177 inherited tensors remain exact, exactly 48 attention parameters own
  finite step-256 Adam state, and all 256 updates retain balanced support with
  no skip/duplicate/failure or RSS growth. The matched 192--248 window keeps
  position nearly flat but regresses every velocity horizon, collision F1
  (`0.222222 -> 0.189873`), and uncertainty; identity improves slightly.
  Retained the checkpoint as integrity evidence without accuracy promotion.
- Extended the corrected campaign audit through step 264. All updates apply
  with exact eight-scenario balance, complete causal/objective support, stable
  `2,991,591,424`-byte RSS, and no terminal or uncontained failure. The latest
  raw interaction gradient `2.850016` is contained at `0.350875` complete
  retention; severe typed-output coefficients remain truthfully visible.
  Refreshed the Transformer/scaling review against the original Transformer,
  Chinchilla, Llama 3, Gemma 3, DeepSeek-V3, FlashAttention, V-JEPA 2, and
  ObjectForesight. The current short-set architecture already has the relevant
  dense mechanisms; efficiency machinery for long KV caches or sparse experts
  is deferred, and the existing physical regressions keep capacity promotion
  blocked until fixed-selector plateau and disjoint test/OOD non-regression.
- Added an optional deterministic matched-reference mode to the training-
  dynamics auditor. It canonicalizes both streams, requires exact alignment of
  optimizer step, seeds, scenarios, draw index, frame window, and rollout
  anchors, fails missing/mismatched schedules, and reports independently
  sufficient-statistic-pooled candidate/reference summaries plus signed nested
  deltas. Focused tests report `15 passed`; Ruff and diff gates pass.
- Audited the corrected and predecessor steps 192--272 with the new mode. All
  11 rows align exactly. Long-position and median uncertainty are nearly flat
  or slightly better, but current position is `0.004070 m` worse, every
  velocity horizon regresses, collision F1 falls `0.197044 -> 0.165049`, and
  lifecycle precision/coverage weaken. This schedule-controlled warning does
  not replace the fixed step-512 selector or authorize capacity growth.
- Passed the deterministic step-280 historical stress boundary in the live
  repaired campaign. All support is present, semantic applied norms remain
  below `0.1`, and the complete interaction stage retains `0.462814` with no
  failure. Same-seed current position and all five position horizons improve,
  while collision F1 and 0.25/0.5/1.0-second velocity regress. Preserved this
  as optimizer-repair evidence and an accuracy tradeoff, not promotion.
- Closed the complete schedule-exact steps 264--320 trend window. All eight
  scenarios, support, optimizer updates, gradient containment, observability,
  and resources remain healthy. Short position and y improve, but every x
  horizon, medium/long pooled position, four velocity horizons, coverage90,
  and identity regress; switches increase from three to six on nearly equal
  association support. Preserved the window as a broad warning and kept fixed
  selector 512 authoritative.
- Passed the exact step-384 attention checkpoint audit: all 48 attention
  tensors changed, all 177 inherited tensors remain exact, optimizer state is
  complete/finite/attention-only at Adam step 384, and every provenance hash
  agrees. The matched steps 328--384 window repairs current position/velocity,
  collision F1, and every x horizon relative to the predecessor, but pooled
  forecasts, every z horizon, three later velocity horizons, lifecycle, and
  identity regress. Continued the immutable run toward selector 512 without a
  learning-rate, cooldown, or capacity mutation.

- Audited the live immutable campaign through durable step 896. All updates
  apply with exact eight-scenario logged balance, no skip/terminal/uncontained
  failure, stable RSS, 177 inherited tensors exact, all 48 attention tensors
  live, and complete attention-only Adam state at step 896. The newest pooled
  training window keeps current error flat but regresses every forecast
  horizon versus the prior window; it is recorded as a trend warning while the
  unchanged fixed selector at step 1024 remains authoritative.
- Advanced the observability contract to specification 1.34. The dynamics
  auditor now emits configurable complete/incomplete training-trend windows
  with count-pooled position axes, position/velocity horizons, lifecycle,
  identity, uncertainty, event F1, slow-parameter observability, support,
  gradient retention, and memory. Focused tests reject the prior temptation to
  average unequal-support derived metrics.
- Closed the first newly reported live trend tail at step 832. It regresses
  from the preceding unusually strong sampled window but remains better than
  the prior window on current and four of five position horizons, with healthy
  lifecycle/event/support/gradient/resource evidence. It is recorded as
  optimization wobble; no weights, optimizer controls, or promotion change.
- Advanced the contract to specification 1.33 and implemented the narrow
  function-preserving depth handoff anticipated by ADR-102. A trained shallow
  attention stack may now seed only contiguous appended blocks; new MHA and
  SwiGLU output projections are exact zero, so shallow and grown token streams
  and decoded outputs match at zero tolerance. Malformed inherited blocks,
  width changes, changed shape-invisible head/runtime semantics, and other
  partial transfers still fail before mutation. Trainer metadata durably names
  the transform, source checkpoint, and appended indices.
- Audited the unaffected sustained run through durable step 768. Checkpoint
  scope, inherited/protected hashes, complete attention-only Adam ownership,
  finiteness, balanced exposure, support, memory, and dynamics all pass. The
  newest equal sampled window improves current and every forecast horizon, but
  selector 1024 remains mandatory before any accuracy or scale decision.
- Advanced the contract to specification 1.32 after a pre-scale handoff audit
  found that a trained four-block attention checkpoint could partially seed a
  six-block destination while leaving random new blocks on the learned decoder
  path. Weight-only loading now validates all keys and shapes before copying,
  rejects partially present allowed module prefixes, and leaves rejected
  destinations unchanged. Added a regression test and made the rung order
  consistently data-only, depth, width, bounded history, then single CUDA.
- Re-audited the immutable specification-1.31 campaign through update 704:
  all updates apply with exact logged scenario balance, no skipped/terminal/
  uncontained failure, stable 2.92 GB RSS, and a passing dynamics audit. The
  latest equal training window improves medium/long horizons but regresses
  current/short error and identity, so selector 1024 remains the next decision
  point and capacity scaling remains blocked.
- Stopped the fresh specification-1.30 attention campaign after the pre-Adam
  retention gate rejected deterministic attempted update 60 at `0.0850405`.
  The run had applied 59 supported updates but had no durable trained selector;
  none of its mutable state is promoted or resumed.
- Added structured terminal optimizer diagnostics and made the offline dynamics
  auditor fail durable numerical/retention failures even when the last sampled
  training row is healthy. Exact replay matched all 400--454 comparable
  model/data fields at updates 8--56 and captured update-60 seeds, scenarios,
  support, physical metrics, and the full pre-mutation gradient hierarchy.
- Localized the residual gradient to the accumulated node decoder: norm
  `11.6617`, world-y row `11.5014`, versus maximum shared non-decoder norm
  `0.124876`. A joint accumulated node x/y/z cap of `1.0` reconstructs
  `0.552059` complete-stage retention on the failed gradient, versus the
  required `0.1`.
- Advanced the contract/runtime to specification 1.31. Added the node decoder
  group cap before collision/force/impulse/interaction/global clipping,
  protocol/resume and legacy-null semantics, raw/applied/intermediate
  telemetry, config/checkpoint/auditor coverage, and focused regression tests.
- Ran a fresh non-promotable protected-control replay with MPS RGB and CPU
  closed loop. Its 225 tensors and 2,583 comparable initial-selector metrics
  are exact at score `0.3213162196`. On the same update-60 seeds it remains
  fully supported and retains `0.565343` at the complete interaction stage;
  the diagnostic deliberately stops before Adam and records
  `node_row_repair_report.json` beside the run.
- Refreshed the evidence-gated scale ladder against the original Transformer,
  Llama 3, Gemma 3, V-JEPA 2, and compute-optimal scaling evidence. Orpheus
  keeps its modern RMS-pre-norm/SwiGLU set Transformer; GQA, RoPE, local
  attention, sparse experts, and long-context kernels remain deferred until
  their bottlenecks exist. Minimum proportional exposure is now explicit:
  9,728/12,288/23,040 updates for the 3.53M/4.34M/8.31M rungs. Scaling remains
  blocked until the repaired 3.00M control reaches fixed-selector plateau.
- Passed the complete non-device suite (`706 passed, 5 skipped, 1 deselected`),
  host MPS marker (`1 passed, 711 deselected`), five direct MPS regressions,
  focused affected suites, Ruff format/check, compileall, diff check, and host
  MPS dry-run resolution (`measurement=mps`, `closed_loop=cpu`).
- Launched the clean committed specification-1.31 campaign at
  `runs/20260811-063308-attention-node-isolated-stage-a/` under one-shot
  Standard launchd with an immutable-source convergence supervisor. Its
  32-episode initial RGB-only selector is exact against the protected control:
  225/225 tensors and 2,578/2,578 comparable metrics equal, score
  `0.3213162196`, and only the expected protocol hash differs. Training remains
  active. Its live dynamics audit now passes through 128 applied updates with
  exact logged scenario balance, zero skipped draws or terminal failures, no
  uncontained interaction clip, finite uncertainty, and axis/horizon support.
  The former update-60 rejection is cleared. Independent step-128 audit proves
  177 inherited tensors exact, all 48 attention tensors changed, exactly 48
  attention-owned Adam states at step 128, finite serialized state, and intact
  architecture/source/protocol/model hashes. This is optimizer-health evidence,
  not a fixed-selector convergence or scale result. The live audit then passes
  the historical update-152 stress position with `0.344214` complete-stage
  retention, 343 causal targets, every horizon supported, cumulative trusted
  identity 11/820 (`1.34%`), and no skipped, terminal, or uncontained update.
- Passed the former catastrophic update-200 impulse boundary in the fresh run.
  Raw/applied gradient is `1.14436/1.0`; impulse multiplier/additive norms are
  ordinary at `0.18604/0.00781`; complete retention is `0.873850`; 339 causal
  targets and every horizon are supported. The live audit passes all 200
  updates with exact logged scenario balance, zero skips/failures/uncontained
  clips, cumulative trusted identity 14/1,127 (`1.24%`), finite uncertainty,
  and bounded memory. This remains training-health rather than selector
  accuracy evidence.
- Independently audited the fresh step-256 checkpoint: 177 inherited tensors
  exact, all 48 attention tensors changed, exactly those 48 own finite Adam
  state at step 256, and architecture/source/protocol/model hashes agree.
- Passed the historically recurrent update-280 boundary. Raw interaction fell
  to `2.86878`, versus `52.9646` before row isolation and `17.7050` after
  collision-only isolation; post-row norm is `1.29273` and complete retention
  is `0.348580`. The update applies with 145 causal targets, every horizon, and
  no skip. The cumulative 280-update audit passes with exact logged scenario
  balance, no terminal/uncontained failure, trusted identity 19/1,532
  (`1.24%`), finite uncertainty, and bounded memory. Fixed-selector accuracy
  remains pending.
- Preserved and independently audited the unchanged campaign at step 384.
  All 177 inherited tensors remain exact; all 48 attention tensors changed;
  exactly those 48 parameters own finite Adam state at step 384; serialized
  state and architecture/source/runtime/protocol/protected hashes pass. The
  live audit reports 384 applied updates, exact 48-block support for every
  scenario, zero skips/terminal failures/uncontained clips, trusted identity
  26/2,105 (`1.235%`), pooled coverage90 `90.34%`, every-horizon weighted
  support, and unchanged `2.923 GB` peak RSS. Weighted 0.1--1.0-second position
  RMSE is `0.2926/0.3206/0.3651/0.4123/0.4515 m`; this remains training-window
  evidence, not fixed-selector accuracy or convergence.
- Completed the first trained fixed selector at step 512 on all 32 RGB-only
  validation episodes. Its latest persisted evaluation is rejected: score
  `0.325191` versus protected step-zero `0.321316`; current position RMSE
  `0.267023` versus `0.251460 m`; target coverage `0.36575` versus `0.37625`;
  precision `0.347258` versus `0.357312`; collision F1 `0.144186` versus
  `0.195489`; and broad x/reference-pair regressions. All scenarios have four episodes and no
  support failure, so this is not a manifest artifact. The step-512 artifact
  audit passes with 177 inherited tensors exact, all 48 attention tensors live,
  attention-only Adam step 512, finite state, and intact hashes; the dynamics
  audit passes 512 applied updates with exact balance and no optimizer failure.
  The safe incumbent remains step zero and scaling stays blocked while the
  rejected mutable trajectory continues toward the next selector.
- Continued the rejected mutable attention trajectory through step 576 without
  touching the safe incumbent. The audit passes all 576 balanced updates with
  no skip, terminal failure, or uncontained clip. Local tangent-force/node
  outliers at steps 560/568 are contained before the shared stage, which
  retains `0.3522/0.7547` versus the `0.1` minimum. In equal eight-block
  training samples, every pooled horizon and every x/z horizon improves after
  selector rejection, but coverage90 slips `90.49% -> 89.17%` and y at 0.5
  seconds regresses. This is repair-trajectory evidence only; selector 1024
  remains required.
- Audited the following balanced steps 584--640 and retained the adverse result:
  every pooled horizon and every x/z horizon worsens relative to steps
  520--576, while identity improves and coverage90 falls to `88.10%`. Because
  deterministic windows/support differ, this demonstrates training-sample
  wobble rather than a matched held-out conclusion. The full auditor passes all
  640 updates; step 640's force outlier leaves `0.6927` post-isolation
  shared-stage retention. No promotion, scale, or protocol change is made
  before selector 1024.

### 2026-08-11 impulse-gradient isolation and fail-fast monitoring

- Stopped the fresh output-isolated campaign after update 200 exposed an
  uncovered recursive impulse path: raw gradient `857.1579`, impulse
  multiplier/additive rows `830.3828/210.3096`, maximum shared norm `6.2401`,
  and only `0.001167` complete-stage retention. Preserved the independently
  audited step-128 checkpoint as the last durable source; no trained selector
  or convergence claim was made.
- Advanced the contract/runtime to specification 1.30. Added joint impulse
  per-invocation output and accumulated decoder-row caps, configuration and
  resume/selector semantics, legacy-null normalization, and complete named
  telemetry without changing forward values, parameter count, or tensor
  shapes.
- Added an optional complete-interaction retention fail-fast gate; the active
  pilot rejects sub-10% post-isolation retention before Adam mutates. The
  offline auditor now treats the same pattern as a hard failure while retaining
  successfully contained local semantic clips as warnings.
- Replayed the durable step-128 model/Adam/RNG/sampler state through the same
  step-200 seeds/window. Raw norm fell to `7.4410`, maximum shared norm to
  `0.05334`, complete-stage retention rose to `0.64704`, and 1-second sampled
  RMSE improved `0.441224 -> 0.437779 m` with identical support/identity/
  coverage. The replay audit passes with no severe or uncontained blocks; the
  non-promotable report is
  `runs/20260811-033712-step200-impulse-gradient-replay-v1/impulse_gradient_replay_report.json`.
- Passed focused tests (`281`), complete non-device tests (`697 passed,
  5 skipped, 1 deselected`), the host device marker (`1 passed, 702
  deselected`), five direct MPS regressions, Ruff, compileall, dry run, and diff
  checks. Capacity scaling remains gated on a fresh small-rung fixed-selector
  learning curve and plateau.
- Committed and pushed the repair as `d38cc9b`, then launched the fresh
  specification-1.30 campaign at
  `runs/20260811-042704-attention-impulse-isolated-stage-a/` with MPS RGB,
  CPU closed loop, immutable source provenance, and the unchanged supervised
  convergence envelope. The complete initial selector exactly reproduces all
  225 tensors and 2,583 comparable metrics at score `0.3213162196`; its
  equality audit is retained beside the run.
- Audited sampled update 8: all eight scenario families and causal support are
  present, raw/applied interaction norm is `0.673975`, no local or global clip
  fires, the update is finite/applied, RSS is bounded, and trainer/supervisor
  stderr remain empty. This is launch health, not convergence or promotion.

### 2026-08-11 typed-output backpropagation isolation

- Stopped the force-row-isolated sustained campaign at durable step 256 after
  update 280 exposed a remaining causal optimizer defect: decoder parameter-row
  clipping ran after a raw `989.7965` force signal had already produced severe
  gradients throughout the shared attention stack. The finite step-280 update
  is diagnostic only and no trained selector or convergence claim was made.
- Added separately configurable per-invocation backward caps for typed node,
  collision, and joint normal/tangent-force outputs before their decoder and
  shared attention representation. Retained the later parameter-row,
  interaction, and global caps; bound the new fields into config validation,
  resume compatibility, and selector protocol hashes; and added raw/applied/
  minimum/effective output-gradient telemetry plus auditor warnings.
- Replayed exact updates 257--280 from the durable source state on MPS RGB plus
  CPU closed loop. At the same step-280 seeds/window, the later raw parameter
  norm fell `995.5391 -> 10.8330`, the largest shared projection/block norm was
  `0.0851`, the post-row interaction stage retained `0.6979`, and the supported
  finite update was applied. The audit passes with localized severe output/row
  warnings. The diagnostic artifact is
  `runs/20260811-004400-step280-output-gradient-replay-v1/typed_output_gradient_replay_report.json`.
- Advanced the architectural contract and runtime metadata to specification
  1.28. Added focused tests proving the hook bounds typed gradients before they
  reach upstream shared features and that the offline auditor reports the new
  hierarchy. At that repair boundary, a fresh immutable weights-only campaign
  and complete regression gates remained pending; capacity scaling stayed
  gated.
- Passed final gates (`297` affected/checkpoint tests; `678 passed, 5 skipped,
  1 deselected` non-device; host MPS marker and five direct MPS regressions;
  Ruff, compileall, dry run, and diff check), committed/pushed `9d0502b`, and
  launched the clean weights-only campaign at
  `runs/20260811-012103-attention-output-isolated-stage-a/`. Trainer metadata
  records clean immutable source, MPS RGB/CPU closed-loop placement, RGB-only
  input, no oracle, and protocol hash `21daf4a8...d7f`. An exact-source
  one-shot supervisor carries the 8,192 minimum, 4,096 extensions,
  four-selector/1% plateau rule, and 24,576 hard limit. Initial fixed
  validation is active; no trained accuracy or convergence claim exists.
- Completed the fresh run's 32-episode initialization selector in `969.521 s`.
  All 225 model tensors, the protected model hash, and all 2,583 comparable
  non-protocol metrics are exact. Preserved the equality audit beside the run.
- Passed the former step-64 force-gradient stress position on identical seeds
  and support. Typed-output isolation reduces raw total gradient
  `21.5377 -> 2.14592`, joint force-row norm `21.4665 -> 1.75123`, and maximum
  non-decoder shared norm `0.04242 -> 0.00540`; all 64 updates are applied and
  the dynamics auditor reports no hard failure. Training remains active toward
  durable step 128 and selector 512; no accuracy/convergence promotion is
  claimed from a sampled batch.
- Advanced the scaling contract to specification 1.29 after reviewing the
  original Transformer, compute-optimal scaling, maximal-update transfer, and
  current dense video-world-model evidence. Fixed a one-axis ladder from the
  3.00M control/data curve through 3.53M depth, 4.34M width, bounded timestamped
  history, and an 8.31M single-CUDA rung with parameter-scaled data and disjoint
  validation/test/OOD gates. Deferred long-context/sparse infrastructure that
  does not solve a measured 22-token bottleneck.
- Added `scripts/audit_attention_checkpoint.py` and focused tests so every
  architecture-growth checkpoint can independently prove full-file/model
  hashes, recursive finiteness, resolved tensor shape/dtype compatibility,
  inherited/protected tensor equality, named optimizer ownership, and Adam
  steps. Documented the reproducible command in `project/TRAINING.md`.
- Preserved and audited the output-isolated step-128 checkpoint. All 177
  inherited tensors remain exact, all 48 attention tensors changed, all 48 and
  only those parameters own Adam state at step 128, protected hashes are
  unchanged, and every serialized value is finite. The dynamics audit passes
  128 balanced applied updates with zero skips/hard failures and stable memory.
  Retained the sparse sampled identity increase (`1.153%` versus paired
  predecessor `0.861%`) as a mandatory step-512 selector warning rather than
  tuning to heterogeneous train batches. The complete non-device suite passes
  `681 passed, 5 skipped, 1 deselected`.

### 2026-08-10 typed-attention stage-A pilot

- Committed and pushed the collision-row isolation as `70c2e3b`, then launched
  the fresh weights-only MPS campaign at
  `runs/20260810-180502-attention-collision-isolated-stage-a/`. Initial
  metadata records clean immutable source, RGB-only/no-oracle execution, MPS
  measurement, CPU closed loop, float32, and the protected protocol-14 graph
  checkpoint as initializer. The mandatory step-zero 32-episode selector
  completes in `976.793 s` and exactly reproduces protected score
  `0.3213162196`, every pooled axis/horizon metric, and model hash
  `1354bdfc...df91` with no guardrail/support failure or stderr. Attention-only
  training is active. The first durable step-128 checkpoint passes exact
  scope/state/hash audit: 177 inherited tensors remain bitwise unchanged, all
  48 attention tensors change, all 48 optimizer states are attention-owned at
  Adam step 128, every tensor is finite, and protected artifacts remain exact.
  The dynamics audit reports 128 applied updates, balanced support, zero
  skips/failures/severe clips, and bounded `2.99 GB` RSS. Sampled row caps
  preserve materially more unrelated interaction gradient as intended. The
  former update-152 event-heavy failure improves from raw norm `28.1387` and
  retained coefficient `0.03554` to `7.1111/0.14308`, with complete support
  and no severe clip. Durable step 256 passes exact scope/state/hash audit with
  all inherited tensors unchanged, all 48 attention tensors live, attention-
  only Adam state at step 256, finite state, balanced support, stable RSS, and
  zero hard audit failure. An isolated step-248 trusted-identity rate of
  `6.35%` returns to zero at step 256 and is `0.975%` in aggregate; it remains
  a fixed-selector warning. Update 280 and the first trained selector remain
  pending; no trained accuracy or convergence claim exists yet.

- Stopped the collision-isolated campaign after step 280 reproduced a severe
  complete-interaction coefficient of `0.05648` despite an ordinary unclipped
  collision-row norm of `0.23553`. The failed update is not checkpointed and
  exact step 256 remains protected. Added specification 1.26 plus read-only,
  finite-checked raw-gradient diagnostics for every attention parameter and
  semantic node/relation decoder row so the exact replay can localize the real
  source without guessing another cap.

- Replayed the exact step-256 continuation from clean commit `70c2e3b` under
  external read-only instrumentation. Steps 264/272/280 match the original
  deterministic telemetry exactly. Step 280 localizes raw norms
  `17.3894/3.2159` to normal/tangent force rows (joint `17.6842`) inside the
  `17.7050` interaction norm; collision is ordinary at `0.2355`. Added an
  optional joint force-row cap, raw/applied diagnostics, protocol binding,
  auditor coverage, and focused tests. The stage-A config sets the cap to 1.0;
  this changes no forward dynamics. Also corrected checkpoint specification
  metadata from stale 1.25 to specification 1.27.

- Passed the repaired full gates (`664 passed, 5 skipped, 1 deselected` in the
  non-device suite; host device marker passed; all five direct host-MPS tests
  passed; Ruff, format, compileall, dry run, and diff check passed), committed
  and pushed `b3b69c1`, then launched the fresh weights-only stage-A campaign
  at `runs/20260810-213857-attention-force-isolated-stage-a/`. Its one-shot
  Standard launchd job uses MPS RGB measurement, CPU closed loop, float32,
  clean source provenance, no oracle input, and the protected graph control.
  The initial 32-episode selector is active; no trained result is claimed.

- Completed the force-isolated run's exact step-zero selector in `959.695 s`.
  The model hash and all tensors equal the protected control, and 2,583
  comparable broad metrics have zero differences. Attached an exact-source
  convergence supervisor with the declared 8,192 minimum, 4,096 extensions,
  four-selector/1% plateau rule, and 24,576 hard limit. The first eight
  balanced updates pass the live dynamics audit with all eight scenarios,
  support 349, zero skips, frozen perception, raw gradient `0.668898`, no
  clipping, zero trusted identity switches, bounded 2.874 GB RSS, and empty
  trainer/supervisor stderr. No trained selector is yet available.

- Audited sampled updates 16, 24, and 32 of the force-isolated campaign. All are
  finite, supported attention-only updates with zero perception gradient and
  bounded RSS. At step 24 a raw `4.45588` collision-row norm is locally capped
  before the remaining `2.36835` interaction norm is capped; the offline
  auditor reports no severe clip or hard failure; step 32 independently
  contains a `3.23987` collision-row norm, applies its update, keeps exact
  scenario balance and returns sampled trusted identity switches to zero.
  Recorded an exact capacity census for later data-only, depth-6, width-192, and
  width-256/depth-6 rungs, but did not launch or promote a larger model before
  fixed-selector and plateau evidence.

- Continued the live force-isolated audit through sampled step 72. Step 64 has
  one severe raw joint-force norm of `21.4665`, but the targeted row cap leaves
  a `2.01547` post-row interaction norm and `0.496162` coefficient for
  unrelated attention learning rather than suppressing it to `0.04643`. Step
  72 immediately returns to ordinary force/stage coefficients. All 72 updates
  remain applied and exactly scenario-balanced, with zero skips, frozen
  perception, finite state, bounded RSS, and no hard auditor failure. Retain
  the severe row warning and do not infer promotion before fixed validation.

- Completed the force-isolated step-128 durable checkpoint and independent
  exact-source audit. All 177 inherited tensors remain bitwise unchanged, all
  48 attention tensors change, all 48 optimizer states are attention-owned at
  Adam step 128, all serialized state is finite, and protected best/reference
  hashes remain exact. The dynamics audit records 128 applied balanced
  updates, zero skips or hard failures, one isolated severe force-row warning
  at step 64, frozen perception, and bounded RSS. This is scope/integrity
  evidence only; no trained fixed selector exists yet.

- Passed the exact historical step-152 event-window boundary. On identical
  seeds/frames, raw interaction norm and retained stage coefficient improve
  from `28.1387/0.03554` before row isolation and `7.11114/0.14308` after
  collision isolation to `2.46615/0.48940` in the force-isolated run. Force is
  ordinary and unclipped, support/identity/uncertainty/scope/resource checks
  pass, and the auditor retains only the isolated step-64 warning. Step 280 and
  the first trained fixed selector remain pending.

- Added an optional four-block, width-128, four-head typed attention residual
  over scene, entity, and candidate-relation tokens. RMS pre-normalization,
  scaled dot-product attention, and SwiGLU feed-forwards add `1,103,626`
  parameters (`1,901,030 -> 3,004,656` total) without making tokens persistent
  state.
- Kept current object/relation tokens permutation-equivariant: there are no
  slot-index position embeddings or RoPE. Output heads decode bounded node/
  antisymmetric-pair forces, event logits/jumps, and uncertainty residuals and
  initialize to exact graph identity.
- Added strict weights-only architecture growth: every inherited checkpoint
  tensor is required, and only `dynamics.attention_interactions.*` may be
  absent. Added an attention-only training scope and included the module in
  interaction-local gradient clipping/diagnostics.
- Added `configs/attention_pilot_mps.yaml`: 8,192 balanced updates, 65,536
  episode draws, eight scenarios, 128-step durable checkpoints, 512-step exact
  selectors, MPS RGB measurement, and CPU closed-loop attention.
- Hybrid host smoke `runs/20260810-111959-attention-pilot-smoke/` completes one
  supported update in `626.51 s`, with loss `3.729162`, raw/applied gradient
  norm `1.8602/1.8602`, zero skipped updates, `2,004,131,840` bytes maximum
  RSS, and no oracle input. All 177 inherited tensors are bitwise unchanged;
  only the four zero-initialized decoder tensors acquire nonzero moments.
- Focused architecture/config/checkpoint verification reports `251 passed`;
  final complete-suite and host-MPS outcomes are recorded in project status.
- Committed/pushed the stage-A implementation as `a84ef20` and launched the
  clean 8,192-update one-shot campaign at
  `runs/20260810-114053-attention-pilot-stage-a/` with Standard launch QoS,
  `KeepAlive=false`, clean source provenance, MPS measurement, CPU closed loop,
  and the protocol-14 step-64 runtime protected as its initialization control.
- The campaign's complete 32-episode initialization selector exactly
  reproduces the protected graph score `0.3213162196`. Its first eight
  attention-only updates are finite and supported, draw each scenario once,
  apply every optimizer update, keep perception gradients at zero, and bound
  the raw attention gradient `3.6997645 -> 0.9999997`; this is progress
  evidence, not an accuracy promotion.
- Corrected the offline training-dynamics auditor so `log_every > 1` no longer
  mislabels sampled metric-row count as completed optimizer-update count. It
  now reports the authoritative absolute trainer step, logged confirmations,
  metric gaps, and an explicit sparse-telemetry warning; five focused tests
  cover the new contract.
- Stopped and preserved the first sustained pilot at durable update 128 after
  exact checkpoint audit found all inherited tensors unchanged and all state
  finite, but only 47/48 attention tensors learned: `scene_projection.weight`
  was dead because runtime `global_code` is always zero. The run is diagnostic
  evidence and cannot count toward convergence.
- Replaced that dead scene input with a 55-value context derived from
  authoritative global uncertainty, gravity, camera transform/motion/
  intrinsics/uncertainty, calibration, and reserved global code. Exact graph
  identity and permutation consistency remain intact; a new regression proves
  live scene-projection gradient when global code is zero.
- Corrected verification reports `129 passed` for focused dynamics/scope/
  checkpoint coverage, `650 passed, 5 skipped, 1 deselected` for the complete
  non-device suite, and `1 passed, 655 deselected` on host MPS; Ruff, format,
  compileall, and diff checks pass.
- Corrected two-update host smoke
  `runs/20260810-133010-attention-live-scene-smoke/` completes in `649.00 s`
  with MPS measurement, CPU closed loop, finite supported gradients, zero
  skips, and no oracle input. Exact checkpoint audit leaves all 177 inherited
  tensors unchanged, changes all 48 attention tensors, and gives the scene
  projection a nonzero parameter delta and Adam moment by update two.
- Launched the corrected 8,192-update campaign at
  `runs/20260810-134330-attention-live-scene-stage-a/` from clean commit
  `25d82d8` as one Standard/default, `KeepAlive=false` LaunchAgent. Metadata
  verifies MPS RGB, CPU closed loop, no oracle, and clean repaired-runtime
  provenance; the first 32-episode initialization heartbeat is finite.
- The corrected initialization selector completes in `966.681 s` and exactly
  reproduces protected score `0.3213162196`. Update 8 is finite, fully
  supported and scope-clean with loss `0.4890857`, unclipped gradient
  `0.2631448`, zero skips/identity switches/stderr, and bounded memory; this is
  optimizer evidence, not promotion.
- Stopped the corrected live-scene campaign at sampled update 64 after its raw
  attention gradient reached `45.3456` and the interaction-local coefficient
  fell to `0.02205`, versus `1.3231` on the exact same episode batch in the
  dead-scene control. Ordinary, closely matched objective terms and identical
  seeds isolate this as conditioning rather than a uniquely difficult batch.
- Added fixed non-affine RMS normalization immediately before the 55-to-128
  scene projection. It bounds mixed latent/variance/world/pixel feature scale,
  adds no trainable parameter, and preserves the zero-output residual. An
  extreme `1000x` intrinsics regression verifies finite RMS-bounded input.
- Normalized-model verification reports `20 passed` focused, `651 passed,
  5 skipped, 1 deselected` complete non-device, and `1 passed, 656 deselected`
  on host MPS; Ruff check/format, compileall, and diff checks pass.
- Committed and pushed the conditioning repair as `de06fcb`, then launched
  `runs/20260810-144901-attention-conditioned-stage-a/` under one-shot
  Standard/default LaunchAgent
  `com.polceanum.orpheus.attention-conditioned-20260810-144901`. Clean runtime
  metadata records PyTorch 2.10, MPS RGB measurement, CPU closed loop,
  float32, RGB-only/no-oracle execution, and the protected graph source.
- The exact 32-episode initialization selector completes in `1001.259 s` and
  reproduces protected score `0.3213162196`. Matched updates 8/16 have raw
  gradients `0.2535/1.3194`, improved from `0.2631/1.9980` before scene
  normalization; both are finite, supported, and applied with zero stderr.
  This is conditioning/optimizer evidence only, not an accuracy promotion.
- The exact update-64 failure batch confirms the repair: matched seeds/events/
  support and effectively unchanged loss now produce raw gradient `2.2961`
  instead of `45.3456`, improving the local coefficient `0.02205 -> 0.43552`.
  All first 64 updates remain finite and applied, sampled telemetry has zero
  skips/stderr, and peak RSS stays bounded at `2,903,666,688` bytes. Sustained
  selector accuracy and convergence remain unproven.
- The conditioned run's durable step-128 checkpoint passes exact audit: all
  177 inherited tensors remain bitwise unchanged, all 48 attention tensors
  change, optimizer state belongs only to attention at step 128, protected
  step-zero hashes remain exact, and every serialized tensor is finite. The
  dynamics auditor reports 128 applied balanced updates, zero skips/failures/
  severe clips, raw sampled gradients `0.2535..6.3168`, and bounded
  `2,905,124,864`-byte peak RSS. Accuracy remains pending the step-512 selector.
- The durable step-256 audit again leaves all 177 inherited tensors exact,
  changes all 48 attention tensors, keeps 48 optimizer states attention-only
  at step 256, preserves protected hashes, and finds no nonfinite state or
  skips. One event-heavy update-152 batch is explicitly warned for severe
  clipping (`28.1387 -> 1.0`); twelve subsequent sampled blocks are normal,
  making it isolated rather than a continuing collapse through this boundary.
  Peak RSS grows only about 10.5 MB to `2,915,614,720` bytes.
- Stopped the normalized campaign after severe clipping recurred at update 280,
  exactly 128 updates after step 152, on another deterministic frames 7--11
  contact-heavy batch. Raw norm `52.9646` retained only `0.01888`; durable step
  256 remains scope-clean and finite, but the run has no selector and cannot
  count toward convergence.
- Localized the failure to the typed relation decoder collision-logit row via
  Adam moments. Added an optional norm-1 row-local cap before the interaction
  and global caps, while reconstructing/logging true raw row, interaction, and
  whole-model norms and extending severe-clip auditing. This changes optimizer
  protocol only; forward collision/event semantics remain unchanged.
- Collision-isolation verification reports `236 passed` focused, `216 passed`
  final affected, `657 passed, 5 skipped, 1 deselected` complete non-device,
  and `1 passed, 662 deselected` on host MPS. Ruff, formatting, compileall, and
  diff checks pass.

### 2026-08-10 fast-ROI ownership stability

- Traced protocol 20's apparent association wobble to an exact disconnected-
  component ownership tie in the structured RGB fast path: a `0.0000765`
  predicted-measurement change produced a `0.2807869` centre jump before any
  structural assignment difference.
- Rejected a blanket `0.20` distance cap after paired full validation showed
  that it removed useful long-range evidence and regressed overall accuracy.
- Added explicit fast-ROI ownership margin/ambiguity output and reject only
  scale-aware floating-point ties; source-bound measurements retain their
  predicted centre and global discovery owns ambiguous recovery.
- Advanced `PROJECT_SPEC.md` to 1.21 and rollout validation to protocol 14;
  synchronized checkpoint metadata to 1.21, and added focused disconnected-
  component/subpixel regression coverage.
- Paired 32-episode public evaluation improves joint posterior RMSE
  `0.8087382 -> 0.8079388 m`, x `0.7304025 -> 0.7169902 m`, velocity
  `1.1085611 -> 1.0949822 m/s`, identity switches `37 -> 35`, collision F1
  `0.166227 -> 0.171504`, and four of five horizons with zero nonfinite output.
- Exact physical validation improves protected step 64 score/current/velocity/
  identity to `0.3213162 / 0.2514599 m / 1.0931909 m/s / 0.0135922`.
  Corrected step 512 remains microscopically worse at `0.3213287`; retain step
  64 and stop extending the converged y-only direction.
- Final checks report `642 passed, 5 skipped, 1 deselected` off-device and
  `1 passed, 647 deselected` on host MPS; Ruff, format, diff, and compileall
  checks pass.

### 2026-08-10 axis-isolated correction recovery

- Stopped the healthy but regressive mean-head-only protocol-19 worker at its
  durable step-192 checkpoint after three exact fixed selectors plateaued at
  15, 17, and 24 guardrail failures; step zero remained protected.
- Added schema-checked, provenance-recorded leading-row checkpoint composition
  to the modular evaluator, with focused tests and CLI validation.
- Qualified x/y/z independently at step 64 and y at steps 128/192. X alone
  reproduced the downstream reference-pair y/z regression; z was rejected;
  step-192 y-only passed every guardrail and improved score to `0.3216427`,
  current position to `0.2537443 m`, velocity to `1.0949210 m/s`, identity,
  and every joint horizon.
- Added exact `updater_mean_y` training scope semantics: excluded rows have
  gradients and moments masked and are restored after AdamW, including
  decoupled weight decay. Focused tests report `208 passed`; the complete
  non-device suite reports `641 passed, 5 skipped, 1 deselected`, and host MPS
  device validation reports `1 passed, 646 deselected`. Ruff lint and format
  checks pass.
- Committed/pushed the y-only contract as `3ad5ee2` and launched clean one-shot
  run `runs/20260810-042627-protocol20-y-only-recovery/` from the accepted
  step-192 y-row candidate. Metadata records MPS measurement, CPU closed loop,
  RGB-only/no-oracle runtime, clean source, 512 balanced updates, effective
  learning rate `5e-6`, and exact validation cadence 64.
- Completed protocol 20's first 64 balanced updates and fixed 32-episode
  selector with no optimizer/support/numerical failure. Step 64 is
  guardrail-clean at score `0.3215594`, but its small gain and mixed
  velocity/late-horizon deltas are only interim evidence, not convergence.
- Proved from the numbered runtime checkpoint that every model tensor except
  the learned mean-head weight/bias is bitwise frozen, only y row 1 changes,
  and excluded Adam moment rows remain exactly zero through weight decay.
- Rejected protocol 20 step 128 after it regressed score to `0.3216703` and
  crossed baseline scenario coverage, identity, and 0.10-second x guardrails.
  The optimizer/support/resource audit still passes and all other scenarios
  are nearly flat, so the protected step-64 incumbent remains selected while
  the unchanged trajectory continues to the predeclared step-192 gate.
- Rejected step 192 at the effectively unchanged score `0.3216706` with the
  same three baseline failures. Exact checkpoint deltas show diminishing but
  nonzero y-row motion, finite row-1-only Adam moments, and bitwise preservation
  of every excluded tensor; this is behavioral saturation, not scope or
  optimizer collapse. Continue the declared 512-update run for sufficient
  plateau evidence while retaining step 64.
- Completed step 256 with zero guardrail failures and near-exact recovery of
  the step-64 behavior, but correctly rejected its `0.3215611` score because it
  is `0.00000167` worse than the protected incumbent. Smooth y-row checkpoint
  motion produces a nonmonotonic discrete association response; optimizer,
  support, uncertainty, resources, and scope isolation remain healthy.
- Rejected step 320 at `0.3216708` after it returned to the same baseline
  coverage/identity/0.10-second-x threshold as steps 128/192. Exact row deltas
  and optimizer state remain smooth, finite, and y-only. Closely spaced
  64-update rejections do not satisfy the formal 512-step plateau spacing, so
  continue the declared run with step 64 protected.
- Completed step 384 in the guardrail-clean regime at `0.3215634`, but rejected
  it because it remained `0.00000405` worse than protected step 64. The
  post-320 audit and exact y-row-only checkpoint isolation pass.
- Rejected step 448 at `0.3216787` after baseline coverage, identity, and
  x@100-ms crossed guardrails again. The 384--448 optimizer segment remains
  finite, balanced, unclipped, supported, and exactly confined to mean-head y
  row 1, proving the x failure is downstream association feedback rather than
  optimizer scope leakage.
- Completed all 512 protocol-20 updates and final validation in `19,798.94 s`.
  Step 512 was rejected at `0.3216317` by `reference_pairs` current-x and
  x@100-ms guardrails; protected step 64 remains best at `0.3215594`.
- Passed the full-run dynamics audit: 64/64 applied balanced blocks, exactly 64
  draws per scenario, zero skipped/clipped updates, finite gradients and
  uncertainty, causal support `123--519`, bounded `1.285--1.347 GB` RSS, empty
  stderr, and launch-agent exit code zero. Exact checkpoint comparison proves
  only mean-head y row 1 and its Adam moments changed. Record the campaign as a
  healthy association-sensitive plateau rather than a convergence promotion.

### 2026-08-09 scenario-balanced optimization

- Audited protocol-17 step 2,048 and exact module/scale ablations. Training is
  finite, but later updater/dynamics updates trade camera/glancing depth
  accuracy against heavy/light and other scenario gains; no candidate was
  promoted.
- Added strict deterministic scenario-balanced step-indexed batches so every
  shared-model optimizer update contains equal support from all declared
  regimes and resumes exactly from its absolute draw.
- Added configuration/loader/sampler regressions and the production
  `configs/sustained_accuracy_balanced_mps.yaml` profile.
- Completed a real eight-scenario causal smoke from the step-512 checkpoint:
  one finite supported update, zero perception leakage, terminal RGB-only
  validation, finite checkpoint, and 1.20 GB maximum RSS. The smoke is not an
  accuracy promotion.
- Committed/pushed specification 1.18 as `b646582`, stopped the superseded
  protocol-17 trainer/supervisor at durable step 2,304, and launched
  `runs/20260809-212649-protocol18-balanced-scenarios/` from the fixed-validated
  step-512 candidate. Its clean-source one-shot trainer and exact-commit
  supervisor are active with MPS measurement, CPU closed-loop execution,
  Standard QoS, advancing 32-episode initialization heartbeat, and empty
  stderr; no trained result exists yet.
- Completed initialization and the first 72 balanced updates without skipped,
  non-finite, support, scope, worker, or memory failure. Early balanced updates
  materially reduce typical gradient clipping and increase causal support
  relative to protocol 17.
- Isolated one severe step-64 gradient to baseline seed `16081` and the
  recursive rollout-velocity path through continuous pair-force outputs. Both
  hierarchical clips worked and step 72 returned to normal; recurrence and
  fixed validation remain required before changing or promoting the model.
- Made `audit_training_dynamics.py` warn on global or interaction clipping that
  retains less than 10% of the raw gradient and report the exact step and both
  coefficients. Added a focused regression test.

### 2026-08-09 rollout uncertainty-gradient repair

- Stopped protocol 16 and its convergence supervisor at causal update 552
  after its rejected step-512 validation and first corrected late-phase blocks
  supplied enough evidence for a deeper objective audit. No nonfinite,
  optimizer, lifecycle-support, or restart collapse was observed.
- Recorded the exact step-512 every-axis/horizon, tracking, identity, event,
  and support results; the candidate failed 122 unchanged guardrails and was
  not promoted.
- Fixed rollout Gaussian likelihood so realised forecast error is detached
  from the trajectory mean and calibrates variance only. Deterministic means
  now receive one declared point-loss gradient and remain censored across
  causally unseen external actuation.
- Added a direct gradient regression and advanced the specification/checkpoint
  contract to 1.17. Protocol 16 is not resumable; protocol 17 must start
  weights-only from the same accepted reference.
- Committed/pushed the repair as `6dba48e` and launched
  `runs/20260809-091718-v12-protocol17-rollout-variance-only/`. Its one-shot
  trainer records clean source, host MPS measurement, CPU closed-loop RGB-only
  execution, no oracle, Standard QoS, advancing initialization heartbeat, and
  empty stderr.
- Attached an exact-commit isolated convergence supervisor with the unchanged
  8,192 minimum, 4,096 extensions, four-validation/1% plateau decision, and
  24,576 hard limit; it is waiting durably for the first complete segment.
- Audited all 64 supported fast-ROI optimizer blocks through step 512. No
  skipped update, support collapse, nonfinite state, scope leak, retained-memory
  growth after warm-up, or persistent gradient escalation occurred.
- Rejected the fixed 32-episode step-512 candidate despite improved pooled
  score, every joint horizon, and every current axis. It failed 113 unchanged
  guardrails through velocity, coverage, identity, late axes, and scenario
  tradeoffs; no checkpoint was promoted.
- Confirmed the rollout-gradient repair improves the matched phase over
  protocol 16 on score, current position, velocity, coverage, precision, four
  horizons, and failure count, while preserving its remaining regressions.
- Verified the first `state_dynamics` block has no optimized measurement or
  fast support, exactly zero perception gradient, finite interaction gradient,
  real trajectory support, an unclipped finite total norm, and stable memory.
- Audited all 64 logged state/dynamics blocks through step 1,024: every update
  applied with finite state, six lifecycle-only windows remained a minority,
  local interaction spikes recovered, frozen-perception gradient stayed zero,
  and RSS remained flat.
- Rejected the complete step-1,024 candidate. It improved z rollouts, collision
  F1, identity churn, baseline behavior, and forecast coverage relative to step
  512, but regressed x/y and medium-to-long joint rollouts; score worsened to
  `0.3413697` and reference-guardrail failures increased `113 -> 134`.
- Confirmed mutable/training support failures remained zero and continued the
  exact campaign without promotion toward the step-1,536 fixed validation.
- Completed and rejected the fixed step-1,536 validation. It recovered score,
  current state, velocity, tracking, calibration, all current axes, x/y at
  every horizon, and four joint horizons relative to step 1,024, but remained
  worse than the reference at medium/long horizons and failed 122 broad
  guardrails across scenario balance, axes, coverage, identity, and events.
- Audited checkpoint update direction, optimizer moments, support frequency,
  clipping, frozen-perception isolation, and memory. The run is finite and
  directionally learning rather than numerically wobbling or collapsing.
- Measured event-versus-trajectory gradient conflict on the shared interaction
  trunk across four collision batches. Kept decoupling as an evidence-backed
  follow-up rather than changing architecture after one recovering validation.
- Preserved step 1,536, then exact-resumed protocol 17 under a new one-shot
  trainer and exact-source convergence supervisor. Step 1,544 consumed draw
  1,544 with all 13 objectives, no retry/skip, zero perception gradient,
  finite norm, stable memory, and empty stderr.
- Found that the one-shot helper's mandatory `--run-name` violated the
  trainer's exact in-place resume contract. Quarantined one accidental nested
  continuation and preserved one pre-training directory-collision failure;
  neither changed the authoritative step-1,536 checkpoint.
- Made `--run-name` optional only for exact resume, added a launch-payload
  regression, and relaunched the unchanged checkpoint under corrected
  one-shot trainer/supervisor jobs. The first authoritative resumed update is
  finite, supported, optimizer-applied, and perception-frozen.
- Documented the duplicate append-only step-1,544 telemetry row produced when
  an uncheckpointed logged tail is deterministically replayed after exact
  resume. Verified that convergence consumes numbered validation checkpoints,
  not raw training rows; an attempt-aware logger repair remains post-campaign.
- Added a read-only sustained-training dynamics auditor with replay-row
  equivalence checks, optimizer/support/scope/data-progress failure detection,
  loss/gradient/support/RSS summaries, scenario draw counts, and pooled,
  identity, event, uncertainty, per-axis, and every-horizon validation output.
- Ran it on live protocol 17 through unique logged step 1,592: all eight
  canonical post-1,536 blocks were finite, supported, optimizer-applied, and
  perception-frozen with no skipped draw or memory high-water growth.
- Extended the same audit with live lifecycle coverage/precision, identity,
  collision, uncertainty, correction, parameter-observability, axis-local,
  and horizon-resolved distributions. Ten canonical blocks through step 1,608
  pass; no health metric justifies interrupting the unchanged protocol.
- Traced the step-1,624 identity outlier to a deliberately perturbed,
  contact-heavy recovery batch with valid fully visible targets. Added pooled
  perturbed-versus-clean identity accounting; clean logged identity rate is
  lower and post-1,536 clean blocks contain no switch.

### 2026-08-09 perception-local auxiliary-gradient repair

- Stopped and preserved protocol 15 after validation candidates at steps
  1,024–2,560 met the four-candidate failed-plateau rule. Training was finite,
  supported, and correctly frozen by tensor group, but no candidate passed the
  fixed broad selector and long-horizon x accuracy regressed.
- Rejected exact step-1,536 dynamics-only and updater/identifier-only module
  qualifications. Neither subsystem independently retained the early
  fast-ROI localization gain across the complete forecast horizon.
- Fixed an objective-routing leak: frozen fast-ROI measurement supervision can
  no longer train dynamics/updater through its predicted-prior input. It is
  now a detached `frozen_fast_measurement` diagnostic unless a real fast RGB
  perception path is trainable, and cannot impersonate causal fast support.
- Added predicate, scope, full closed-loop objective, and support regressions;
  advanced the specification to 1.16. A fresh qualification is required.
- Synchronized checkpoint specification metadata from stale `1.12` to `1.16`
  and added a contract-header consistency regression.
- Committed/pushed the repair as `310d419` and launched
  `runs/20260809-065710-v11-protocol16-perception-local-objectives/` from the
  same accepted reference. The one-shot job records clean source, PyTorch
  2.10/MPS availability, MPS measurement plus CPU causal execution, and an
  advancing initialization heartbeat with empty stderr; it has no trained
  accuracy result yet.
- Verified protocol 16's first post-transition block has no optimized
  measurement term and exactly zero perception gradient. Corrected its
  follow-up support diagnostic so frozen observed ROI slots cannot appear as
  differentiable causal fast support.
- Attached a one-shot exact-source convergence supervisor to protocol 16. It
  verifies the initial 8,192-step segment and may launch only full 4,096-step
  exact-resume extensions until the four-validation plateau rule or 24,576
  hard limit; its launch event is durable and stderr is empty.

### 2026-08-08 frozen-loss objective-integrity repair

- Audited the frozen-backbone campaign through 4,744 supported causal updates
  and ten numbered validations (initialization plus steps 512–4,608). The
  process, optimizer, frozen backbone, and finite state were healthy, but
  candidates oscillated across axes/scenarios and none passed the unchanged
  broad selector. Step 512 improved pooled RMSE at every forecast horizon but
  regressed coverage, identity, and y-axis/scenario guardrails.
- Found and fixed a causal objective bug: the trainable ROI-only
  `fast_projection` caused a container-wide predicate to include a completely
  frozen global-discovery loss in the measurement objective. A representative
  step mixed global `5.287398` with fast ROI `0.050371` into `2.668884`,
  contributing no global gradient while making total-loss trends misleading.
- Global trainability now follows only detector/shared-stage/pyramid paths.
  Frozen global loss remains an explicit diagnostic and cannot enter or scale
  the fast-ROI objective. Added predicate and real closed-loop batch
  regressions, and advanced the specification to 1.14.
- Intentionally stopped and unloaded the flawed one-shot campaign at step
  4,744. Its artifacts are preserved; it is neither converged nor promoted.
- Committed and pushed the repair as `c13d5d9`, then launched
  `runs/20260808-161058-v9-protocol14-fast-roi-objective/` from the same
  accepted reference and unchanged balanced protocol. The clean-source
  one-shot job uses MPS measurement and CPU closed-loop execution and is
  advancing its initialization validation with empty stderr; it has no
  corrected training metric or accuracy result yet.
- Stopped that replacement at step 720 after its first trained validation
  regressed score to `0.3749701`, x RMSE to `0.4224541`, and every horizon
  despite finite state and intact coverage support.
- Fixed a second objective bug: losing/fixing one RGB branch no longer
  renormalizes the remaining branch over a smaller denominator. Fixed
  `1:fast_weight` coefficients now remain fixed under missing support.
- Exact modular qualification showed doubled-weight fast ROI reproduced most
  of the regression (`0.3602169`), while the earlier half-weight fast ROI alone
  improved score to `0.3110033`, current position to `0.2509520`, every axis,
  every horizon, precision, collision F1, and identity, with remaining
  velocity/coverage/scenario failures truthfully rejected.
- Added an explicit `fast_roi` scope and a configured completed-causal-update
  transition to a late scope. The next evidence-led curriculum uses 512
  fast-ROI-only updates followed by `state_dynamics`; specification is 1.15.
- Committed and pushed the staged repair as `2fea10a`, then launched
  `runs/20260808-193216-v10-protocol15-staged-fast-roi-state-dynamics/` from
  the same accepted reference. Its clean-source one-shot job is active in the
  fixed initialization validation with MPS available and stderr empty; no
  trained accuracy result exists yet.

### 2026-08-07 modular long-horizon qualification and fast-ROI isolation

- Audited the protocol-13 causal campaign through 6,096 updates and eleven
  complete post-initialization validations. Optimizer state accumulated and
  memory/numerics remained stable, but no candidate passed the fixed broad
  selector. Step 4,096 improved pooled RMSE at every horizon while regressing
  tracking coverage/identity and scenario guardrails, demonstrating a real
  accuracy-support tradeoff rather than convergence.
- Added a schema-checked modular checkpoint composer and an exact fixed-seed
  RGB-only qualification CLI. It records timestamp-first raw metrics, strict
  failure lists, provenance, and a weights-only candidate without importing
  optimizer/RNG state.
- Rejected dynamics-only, 25% full-model interpolation, and other modular
  candidates truthfully. The best diagnostic candidate combines the accepted
  global detector/shared backbone with the donor fast ROI, state update,
  identifier, and dynamics: its weighted 0.1–1.0 s score improves from
  `0.3296688` to `0.2909420`, with lower RMSE at every horizon, but it remains
  unaccepted because identity, z-axis, forecast-coverage, and scenario
  guardrails fail.
- Added `state_dynamics_fast_roi`, which freezes all shared backbone/global
  discovery tensors while adapting the ROI-only projection/updater and causal
  state modules. This closes the trainability leak identified by the modular
  audit without relaxing selection.
- Advanced the authoritative specification to 1.13. A new long run must start
  weights-only from the accepted reference with zero global-adaptation steps;
  no rejected modular checkpoint is a deployment baseline.
- Committed and pushed the repair as `ea67f8d`, then launched
  `runs/20260807-223146-v8-protocol13-frozen-fast-roi/` from the accepted
  step-zero reference with 8,192 causal updates, zero global-adaptation steps,
  the frozen-backbone fast-ROI scope, MPS measurement execution, and CPU
  causal execution. The clean-source one-shot job is alive in initialization
  validation with empty stderr. The 32-episode initialization completed with
  the exact tensor-linked accepted score `0.3296687588`; its first eight
  causal draws produced one logged finite step-8 update with zero skips,
  global perception frozen, and empty stderr. It has no trained-candidate
  accuracy result yet.

### 2026-08-06 protocol-13 mutable-optimisation and resource-integrity repair

- Audited the terminated protocol-12 campaign through logged step 11,776 and
  durable step 11,648. macOS unified logs identify an `OS_REASON_JETSAM` kill
  during system-wide memory pressure; no terminal summary or convergence
  decision exists.
- Reconstructed all six causal validations. Every 512-update block restored
  the step-zero rollout incumbent and reset Adam because a scenario or
  reference-relative deployment floor failed, even when pooled coverage was
  finite and the raw score improved past the fixed reference.
- Split deployment eligibility from mutable optimizer viability. Protocol 13
  still rejects incomplete/regressed scenario slices for promotion.
  Invalid/nonfinite candidates fail closed; well-formed mutable state rolls
  back only for pooled current/all-horizon coverage below the absolute floors.
  Checkpoints and convergence inspection persist/use the two decisions
  independently.
- Added a regression proving that scenario-only deployment rejection preserves
  updated tensors and Adam state, while the existing pooled-collapse
  regressions continue to require verified incumbent restoration.
- Bounded sustained macOS loading to two non-persistent workers and one
  prefetched batch per worker, release the prior MPS/CUDA allocator cache at
  phase transitions, and log process maximum RSS with every optimizer metric.
- Made CLI state transition explicitly from `starting` to `running`. A
  supervisor-proved process disappearance now writes `ExternalTrainerExit` to
  the primary training state, current failure artifact, and append-only failure
  history; killed extension children receive the same treatment.
- Closed the documented fast-ROI fail-closed gap: positive crop evidence now
  requires both a nonnegative mapping and `matched_slots=true`, with a stale
  nonnegative-index regression.
- Advanced the authoritative specification to 1.12 and updated sustained
  training documentation for the failed protocol-12 campaign and fresh
  protocol-13 qualification requirement.
- Passed `603` non-device tests, all `70` host MPS/device tests, Ruff,
  compileall, and diff checks. A real one-update causal smoke at
  `runs/20260806-213442-protocol13-one-update-smoke/` completed in `45.65 s`
  with finite state, no skipped draw, no oracle input, and `616,239,104` bytes
  maximum RSS; its unsupported random-weight validation is wiring evidence,
  not an accuracy result.
- Committed and pushed the repair as `1470b2e`, then launched the clean-source
  8,192-update causal continuation at
  `runs/20260806-213753-v7-protocol13-causal-convergence/` from the finite
  step-10,240 candidate. The one-shot Standard/default job has one live PID,
  explicit running state, empty stderr, host MPS visibility, CPU closed-loop
  execution, and an advancing atomic initialization validation; it has no new
  accuracy result yet.

### 2026-08-04 conservative repository cleanup

- Inventoried tracked source, ignored files, cache directories, run/demo
  footprint, empty directories, and tracked references while the protocol-12
  convergence campaign continued.
- Quarantined 3.0 MiB of regenerable Python bytecode, pytest/Ruff caches, and
  editable-install metadata under
  `/private/tmp/orpheus-cleanup-20260804-215308/`.
- Quarantined the genuinely empty
  `demo_outputs/20260728-151223-scaled-step257/` directory.
- Preserved all nonempty run/demo evidence, including the active campaign and
  its initialization source. `git clean -ndX` now lists only the intentionally
  ignored `runs/` and `demo_outputs/` trees.
- Made no executable-source change: the active exact-resume fingerprint is
  unchanged, and low-static-reference evaluation helpers remain because
  `PROJECT_SPEC.md` explicitly requires them.

### 2026-08-04 live step-6144 source-integrity audit

- Verified that `main` and `origin/main` both pointed to `fa9f7a9` and that the
  executable-source fingerprint still exactly matched the active checkpoint.
- Audited paired global/fast supervision, persistent ROI identity masks,
  phase trainability, causal support, finite optimizer/checkpoint state,
  selection, exact resume, prepared propagation, and supervisor extension
  logic without finding a defect exercised by the active campaign.
- Confirmed with a production-profile CPU probe that objective-connected ROI
  heads receive finite nonzero gradients when positive crop support exists;
  negative-only batches correctly omit unsupported state gradients.
- Passed Ruff and the complete non-device regression gate:
  `599 passed, 5 skipped, 1 deselected in 247.25s`. The live validation
  advanced during the suite and both production stderr logs stayed empty.
- Deferred two executable hardening items until the exact-resume campaign is
  terminal: require `matched_slots` as an explicit positive-crop gate even
  when a stale nonnegative target index is supplied, and expose a distinct
  live/running CLI state. Current production callers sanitize rejected indices,
  and `training_progress.json` already supplies truthful live heartbeats.

### 2026-08-03 launch-QoS and integration-grid collapse audit

- Audited and intentionally stopped
  `runs/20260803-101108-v5-protocol11-balanced-qualification/` after five of
  32 initialization-validation episodes and before any optimizer update,
  metric, or checkpoint. Heartbeats advanced, stderr stayed empty, and the
  original process remained authoritative; the durable interruption/failure
  artifacts record `KeyboardInterrupt`, so this is not numerical-collapse or
  convergence evidence.
- Identified a concrete launch-QoS regression. The one-shot plist classified
  explicitly requested training as launchd `Background`, reducing observed CPU
  use to roughly `100–198%` and increasing mean validation time to
  `117.380 s/episode`. A matched repaired foreground control used roughly
  `525%` CPU and took `25.305 s/episode`. The launcher now uses launchd's
  Standard/default classification while retaining `KeepAlive=false` and
  `caffeinate`.
- Fixed dtype-sensitive substep selection. Float32 20 Hz timestamp differences
  previously made 22 of 39 nominal six-tick intervals execute seven belief
  substeps even though the 120 Hz simulator labels used six. Only
  precision-indistinguishable integral ratios now snap; genuinely non-integral
  intervals still ceil and retain interval event accumulation.
- Added typed one-use prepared propagation so causal training can inspect the
  exact prior and then pass it through ordinary ingestion without recomputing
  dynamics, zeroing `dt`, losing interval collisions, or creating a second
  source of truth. Source/result tensors and dynamics parameters/buffers/mode
  are revision-bound; in-place value or graph-metadata mutation, replacement,
  reuse, and nonuniform batch targets fail closed. This version-tracked
  contract uses `no_grad`, not `inference_mode`.
- Allowed training/selector rollouts to skip stacking unused auxiliary
  trajectories while preserving the complete public rollout default.
- Added the missing burn-in regression proving that a physically
  distance-rejected association cannot seed slow-parameter frame/identity
  history; re-audited tentative births, fast-ROI identity, global discovery,
  and runtime-observation parameter gates without finding another structural
  defect.
- Bumped rollout validation protocol 11 to 12 and advanced the authoritative
  specification to version 1.11. Simulator v4, measurement protocol 5, and
  selection metric 6 are unchanged.
- Completed a fixed-seed foreground protocol-12 timing control at
  `runs/20260803-105244-v6-protocol12-foreground-timing/`: validation batch
  time was `29.578 s`, versus `123.660 s` for the same seed/scenario in the
  Background job (`4.18x` faster). It was intentionally interrupted before
  its first optimizer update and is timing evidence only.
- Completed a reduced production-model causal smoke at
  `runs/20260803-110550-v6-protocol12-one-update-smoke/`. It applied one
  supported finite update (`loss=4.273417`, pre-clip gradient norm `3.012750`,
  applied norm `1.246404`), skipped no batches, wrote and revalidated
  `checkpoints/last.pt`, then correctly rejected the step-one candidate's
  slight score regression (`0.2181897208` versus incumbent `0.2181881493`).
  This proves repaired optimizer/checkpoint/selector wiring, not convergence.
- Passed `100` affected tests, the complete sandbox suite
  (`599 passed, 6 MPS-only skipped`), Ruff format/lint, compileall, the
  production-profile CPU dry run, and `git diff --check`. A new host-MPS rerun
  remains pending because the execution approval service hit its external
  usage limit; no device result was fabricated.
- Launched the unshortened 16,384-step protocol-12 convergence campaign at
  `runs/20260803-112948-v6-protocol12-full-convergence/` from accepted
  `best_rollout.pt` and clean commit `e08c4d0`. Host metadata confirms MPS
  availability for measurement training, CPU for the configured closed-loop
  phase, one Standard/default launchd process, empty stderr, and initial
  validation throughput of `28.342`, `26.702`, and `27.385 s/episode`.
- Completed all 32 broad initialization episodes in `889.508 s`. The supported
  imported incumbent was accepted at score `0.3310606914`, position RMSE
  `0.308032 m`, velocity RMSE `1.135027 m/s`, target coverage `0.322500`,
  prediction precision `0.370903`, collision F1 `0.225519`, ID-switch rate
  `0.006834`, and nominal-90% coverage `0.890140`. The subsequent independent
  MPS measurement-incumbent validation is advancing; its first two episodes
  took `104.511` and `82.898 s`, with empty stderr and no optimizer update yet.
- Audited the same live run after more than 2,000 MPS updates. Fixed validation
  selection score improved monotonically from `11.901029` at initialization to
  `5.688880`, `5.625772`, and `5.305358` through step 1536, with no nonfinite,
  skipped-update, restart, or checkpoint failure. No trained checkpoint has
  been promoted because fast-ROI MAE remains worse than the imported
  `0.189315 m` guardrail; the safe incumbent remains intact.
- Attached `scripts/supervise_convergence.py` as the one-shot Standard/default
  LaunchAgent `com.polceanum.orpheus.convergence-20260803-112948`. It monitors
  exact trainer PID `31197`, requires the full 16,384-step segment, and applies
  the declared 4,096-step extension/1% plateau protocol up to the 24,576-step
  safety limit. Supervisor stderr is empty.
- Continued the full campaign through 3,584 finite MPS measurement updates.
  Step 2048 produced the best raw broad score so far (`4.868897`); steps
  2560/3072 scored `5.029407`/`5.081339`, still far better than the
  `11.901029` initialization. Latest complete-window train loss/MAE are
  `0.82949`/`0.23005 m`. Fast-ROI MAE remains near `0.32 m`, so selector
  guardrails correctly preserve the imported checkpoint while the declared
  long phase continues.

### 2026-08-03 cadence, progress, and finite-state collapse audit

- Audited and intentionally stopped
  `runs/20260803-084843-v4-balanced-qualification/` before its first optimizer
  update. It was the original one-shot process, actively computing finite CPU
  dynamics with stable workers and no stderr/relaunch, but emitted no partial
  metrics during roughly 44 minutes of atomic initial validation. Its durable
  terminal state truthfully records `KeyboardInterrupt`; it supplies no
  convergence evidence.
- Fixed an off-by-one scheduler bug: `global_every_steps=3` previously emitted
  `GLOBAL, FAST, FAST, FAST, GLOBAL` despite the specification requiring two
  intervening fast updates. Cadence three is now exactly
  `GLOBAL, FAST, FAST, GLOBAL`, and configuration requires a positive integer.
  Historical “cadence-three” reports are corrected to actual cadence four.
- Bumped rollout validation protocol 10 to 11 because identical YAML now
  produces different persistent observations and rollouts. Measurement
  protocol 5, simulator v4, and selection metric 6 remain unchanged.
- Added atomic per-episode `training_progress.json` and flushed validation
  heartbeats with phase/kind, completed/total counts, timings, PID, seed,
  scenario, and protocol hash. Interrupted validation records its exception
  type without turning partial results into selector evidence.
- Delayed training iterator/worker startup until the first real data draw so
  unused prefetch workers do not consume resources or obscure initial
  validation.
- Added grouped per-device finite-state checks immediately after every Adam
  step. Checkpoint save/load now validates model parameters/buffers,
  optimizer/scheduler tensors, and scalar finite nonnegative optimizer steps
  before replacement or destination mutation. Regression tests cover NaN
  weights/buffers, Inf moments, negative steps, non-mutating corrupt loads, and
  byte-preserving failed overwrites.
- Completed one finite corrected CPU causal update at
  `runs/20260803-095310-v5-cadence-progress-cpu-smoke/` and one finite host-MPS
  measurement update at
  `runs/20260803-095618-v5-poststep-mps-host-smoke/`. Both are wiring evidence
  from random initialization and neither is promoted.
- Advanced the authoritative specification to version 1.10 and added ADR-073.

### 2026-08-03 initialization-support and launch-failure audit

- Proved that
  `runs/20260803-000858-v3-collapse-repair-qualification/` never performed an
  optimizer update. It contains one step-zero initialization-validation row,
  no `last.pt` or best checkpoint, and a terminal
  `AssertionError: initialization validation must establish the first
  incumbent`. The launchd job then restarted more than 2,284 times and each
  retry failed against the occupied run directory. Removed the job and
  preserved the artifacts; this run supplies no convergence evidence.
- Reduced the `impulse_perturbation` event probability from `0.12` to `0.02`
  per observation interval. The old setting expected about 4.68 hidden events
  in every 40-frame episode and erased nearly all deterministic one-second
  support. The corrected fixed four-episode slice retains real impulses, has
  three independently supported episodes, and passes every configured
  predictable/matched horizon floor.
- Added shared scene-causal future masks to deterministic trainer and evaluator
  position, velocity, event, collision-conditioned, and correction metrics.
  Forecast calibration deliberately retains stochastic outcomes and now
  publishes its coordinate count; reports publish predictable and censored
  deterministic counts.
- Replaced binary nonzero validation support with per-scenario, per-horizon
  minimum predictable-target, matched-target, and supported-episode floors.
  Exact support evidence is persisted, while missing physical metric schema
  now raises instead of being misreported as ordinary zero support.
- Included fully resolved scenario generator configurations and all support
  floors in measurement/rollout protocol hashes; bumped simulator semantics to
  `sphere_world_v4`, rollout protocol to 10, measurement protocol to 5, and
  rollout selection metric to 6.
- Made unsupported imported initialization recoverable, removed full-manifest
  validation before every causal optimizer update, and required the first
  later incumbent to pass available reference/training guardrails.
- Added atomic `training_state.json` and `training_failure.json` CLI artifacts,
  append-only `training_failures.jsonl` history, exclusive fresh run-directory
  claiming, and resume-attempt state so a retry cannot overwrite or contradict
  an earlier terminal result. Added a one-shot macOS LaunchAgent helper with
  `KeepAlive=false`, and supervisor cleanup for both verified initial
  completion and initial failure.
- Added an exclusive per-run training lock, durable interrupted-attempt
  failure history, and supervisor consumption of authoritative terminal
  trainer state. Supervisor PID checks now verify command/run identity so PID
  reuse cannot be mistaken for a live trainer.
- Made every pooled, scenario, horizon, and axis selector field reproducible
  from its retained exact additive evidence. Current-protocol selector
  artifacts are rejected if axis fields, support markers, raw schema, derived
  values, tensor hashes, or fixed-reference state disagree.
- Persisted an incomplete-reference-comparison marker across in-place and
  branched exact resumes. The first supported candidate after an unsupported
  diagnostic state establishes a complete fixed reference but cannot compare
  with and promote itself.
- Promoted fast-ROI source belief slot/object ID from ad-hoc auxiliary keys to
  a typed, paired, validated measurement contract. Stale evidence after slot
  reuse is rejected; global discovery remains free Hungarian association.
- Passed Ruff, compileall, the complete sandbox suite
  (`577 passed, 6 MPS-only skipped`), and the host MPS families
  (`38 passed`). Completed one finite production-profile causal CPU update and
  one real host-MPS optimizer update with terminal checkpoints. Neither
  one-step smoke was promoted or described as an accuracy result.
- Advanced the authoritative specification to version 1.9.

### 2026-08-03 lifecycle, identity, and gradient-collapse audit

- Stopped and preserved
  `runs/20260802-123714-v3-medium-qualification/` after step-1536 validation
  improved conditional position/velocity metrics but grew predictions from
  `3950` to `5274`, identity switches from `10` to `146`, and collision false
  positives from `242` to `469`. The 38 guardrail failures correctly rejected
  it; this is not convergence evidence and its last checkpoint must not be
  resumed under the repaired protocol.
- Distance-gated first-time simulator target alignment and moved the gate
  ahead of Hungarian assignment. Existing persistent-ID target mappings remain
  locked so an identity swap is penalized instead of silently relabelled.
- Implemented real configurable multi-frame tentative births as detached,
  modality/sensor-local evidence outside `WorldBelief`, with increasing
  timestamps, world-distance consistency, cardinality-first matching, and no
  permanent ID until confirmation.
- Pre-gated the core maximum-cost association matrix before Hungarian solving,
  preventing invalid selected edges from reducing valid match cardinality.
  Configuration and direct construction now reject non-finite or non-positive
  maximum costs.
- Added source slot/object-ID metadata to prior-conditioned fast ROI
  measurements and prohibited cross-identity updates while preserving free
  gated global-discovery association.
- Removed births from slow-parameter observation gates and reset per-target
  frame history whenever the accepted runtime identity changes.
- Added a causal-only RGB observation-module gradient cap before the
  interaction-local and whole-model caps, retaining reconstructed raw-total
  diagnostics. Bounded global causal perception adaptation to one 512-update
  validation interval.
- Changed zero-support pooled validation from fatal RMSE conversion into an
  explicit unsupported selection result with raw counts, rejection reasons,
  numbered/reference artifacts, and no fabricated zero-error metric.
- Completed a four-update CPU end-to-end wiring run at
  `runs/20260802-233339-collapse-repair-cpu-smoke/`: two paired RGB and two
  supported causal updates, no skipped/non-finite batch, terminal checkpoint
  complete, and no oracle runtime input. It is not an accuracy result.
- Passed the complete sandbox suite (`536 passed, 6 MPS-only skipped`) and all
  corresponding host MPS test families (`36 passed`), plus compile, Ruff,
  formatting, and diff checks.
- Pushed repair commit `c869571` to `origin/main` and completed the clean
  hybrid host smoke at
  `runs/20260803-000212-collapse-repair-host-smoke/`: four finite updates, no
  skipped batch, real causal/ROI support, causal perception clipping active,
  clean checkpoint provenance, and complete terminal validation. Its slightly
  better tiny pooled score was correctly rejected for coverage and short-y
  guardrail regressions; it is not a promotion claim.
- Pushed the clean-smoke evidence as `baca6a8` and attempted the corrected
  3,072-update balanced qualification at
  `runs/20260803-000858-v3-collapse-repair-qualification/` under
  launchd/caffeinate. Later audit proved that it failed during step-zero
  initialization and took no optimizer steps; its apparent live PID was a
  restart loop, not launch health or convergence evidence.
- Advanced the specification to version 1.8.

### 2026-08-02 supported-causal and convergence-stability audit

- Stopped and preserved
  `runs/20260801-232229-scaled-sustained-v2/` after proving that 121 of
  173 logged causal rows had an exactly zero gradient and that its perception
  handoff collapsed current and future target coverage. It is not a
  convergence result and must not be resumed under the repaired protocol.
- Require explicit differentiable trajectory/state/parameter support or valid
  persistent fast-ROI slots before a causal optimizer update. Unsupported
  deterministic draws advance the sample counter, are logged separately, and
  retry only up to a configured finite bound.
- Added absolute and fixed-reference-relative handoff/causal coverage floors.
  The first unsupported candidate cannot become a synthetic best checkpoint;
  later support collapse restores the verified rollout incumbent and resets
  Adam while ordinary broad-score rejection retains a supported mutable
  candidate.
- Split fast-ROI supervision into identity, ROI, crop-evidence,
  exact-geometry, existence, and visibility masks. Valid empty crops now train
  only negative existence/visibility; unsupported attribute and likelihood
  terms are omitted.
- Count all eligible confident ROI outputs in selection precision, including
  mapped empty crops and valid unmapped persistent tracks, train the fast
  temporal cache on adjacent frames, and support-normalize global and fast
  measurement objectives independently with a fixed weight.
- Removed constant inactive factory-query existence supervision and
  unsupported rollout/event zero terms. Unsupported physical RMSE diagnostics
  now report missing support instead of a misleading `0.0`.
- Aligned analytic contact resolution with the simulator's boundary-before-
  pair order, two solver iterations, sequential corner handling, inverse-mass
  position correction, geometric friction, restitution combination, and event
  accumulation. A randomized three-sphere differential test now agrees within
  floating-point tolerance.
- Tightened online drag and restitution supervision to causally observed
  pre/post evidence, clean intervals, the correct boundary object, and the
  identifiable pair minimum. Runtime observation/birth evidence and
  distance-gated target mapping are explicit.
- Added hierarchical gradient clipping for the learned recursive interaction
  network before whole-model clipping, with raw subsystem/total,
  intermediate, coefficient, and final applied norms in every training log.
  The hard smoke batch retained a pre-global norm `4.8616` after locally
  reducing an interaction norm `85.7563` to `1.0`, then applied the global
  `2.0` limit.
- Made every declared scenario a versioned checkpoint-selection slice.
  Promotion now requires complete scenario support and per-scenario broad
  non-regression/coverage guardrails, so a better pooled score cannot conceal
  a missing dynamics family. Balanced scenario lists must be unique and
  validation budgets must cover every configured scenario.
- Reject negative/non-integral RGB phase boundaries instead of silently
  bypassing the paired-measurement phase and its validated handoff.
- Added `configs/sustained_accuracy_mps_v3.yaml`, keeping one shared
  eight-scenario model, hybrid MPS/CPU execution, 8,192 paired RGB updates,
  8,192 supported causal updates, and the existing broad selection
  guardrails. Short v3 runs are qualification evidence only, not convergence.
- Completed the final-tree hybrid smoke at
  `runs/20260802-121629-convergence-v3-final-audit-smoke/`: two MPS RGB
  updates and two supported CPU causal updates, no skipped draw, version-5
  scenario-aware selection, and a complete terminal checkpoint. Its
  single-scenario two-episode metrics are wiring evidence only.
- Committed and pushed the audited v3 protocol as `c0acf16`, then launched the
  clean 3,072-update balanced qualification at
  `runs/20260802-123714-v3-medium-qualification/`. It retains 32-episode,
  eight-anchor validation every 512 updates; launch health is not a
  convergence claim.
- Advanced the specification to version 1.7 and added the supported-causal
  optimization and hierarchical-gradient-stability contract.

### 2026-08-01 convergence-integrity audit

- Reproduced a data-dependent PyTorch 2.10 MPS failure where finite
  `2x96x64x64` RGB backbone features generated NaN attention/MLP weight
  gradients. Added a configurable hybrid workaround that retains CNN/ROI work
  on MPS and executes only the global proposal transformer on CPU through
  differentiable copies; selector protocol hashes and resume semantics include
  the execution flag.
- Changed training resume, evaluation, demo, and RGB gate fitting to
  deserialize checkpoints on CPU. This preserves CPU AdamW scalar steps and
  owner-device moments in the mixed optimizer and avoids placing an unused
  saved optimizer on MPS.
- Detached RGB covariance linearization centres/depths from mean heads while
  preserving variance-head gradients, closing a second route by which
  calibration/filter covariance could alter position predictions.
- Added recoverable terminal validation, a durable completion marker,
  last-checkpoint-only in-place resume, and byte-preserving completed-run
  inspection semantics.
- Restricted runtime-qualified Hungarian assignment to lifecycle-confident
  proposals and retained confident false positives on target-empty frames.
- Completed the final hybrid smoke at
  `runs/20260801-231521-audit-v2-final-verified-smoke/`: one finite RGB update,
  two finite closed-loop updates, terminal validation, finite model/optimizer
  checkpoint, and byte-identical no-op resume with truthful zero-update CLI
  reporting. It is not an accuracy claim.
- Committed and pushed the corrected source as `df98f63`, then launched the
  clean-source v2 campaign and its bounded convergence supervisor at
  `runs/20260801-232229-scaled-sustained-v2/`. Launch health is recorded, but
  no long-run accuracy or convergence result is claimed yet.
- Preserved and superseded the legacy sustained run after proving that its
  phase handoff discarded all 8,192 RGB updates from the mutable causal path;
  deployment rejection and optimisation continuation are now separate.
- Added `configs/sustained_accuracy_mps_v2.yaml` with 40-frame mature
  one-second support, batch two, 16,384 training episodes, corrected global
  horizon weighting, forecast NLL, and bounded hashed trend-validation anchors.
- Added deterministic absolute-step batch sampling compatible with the legacy
  DataLoader stream, strict semantic resume validation, MPS RNG round-trip,
  immutable launch-time source provenance, and exact additive metric totals.
- Added mature/cold forecast accounting, scene-wide deterministic censoring
  after unseen external actuation, distinct position/velocity correction
  objectives, per-axis/per-horizon/per-scenario/per-seed metrics, structured
  selection rejection reasons, and context-rich training logs.
- Prioritized scarce pair collisions in conditioned windows and removed
  frozen global-perception loss from the causal optimized total.
- Wired RGB appearance supervision and froze the learned corrector visibility
  head whose output is overwritten by explicit RGB visibility.
- Propagated world XYZ belief covariance through the pinhole Jacobian before
  RGB association instead of comparing metre-squared variance with normalized
  image and inverse-depth residuals.
- Removed an absolute fast-ROI centre clamp that corrupted valid partially
  offscreen priors.
- Made filter uncertainty the sole missed-track variance authority and changed
  measurement-quality gating from optimistic maximum confidence to a
  conservative cap, including per-axis/direct-velocity support.
- Fixed repeated low-speed ground bounce chatter, made the glancing scenario
  genuinely oblique, aligned analytic pair restitution with the simulator,
  preserved OOD ranges after named-scenario resolution, and isolated render
  RNG from physics/lifecycle/actuation RNG.
- Rejected the currently unimplemented `birth_confirmations != 1` setting
  instead of silently accepting a dead configuration value.
- Preserved collision evidence across dynamics substeps and separated endpoint
  contact from interval pair/boundary collision; floor support, walls, ceiling,
  and sleeping now have distinct tested semantics.
- Corrected association/innovation slot mappings, confidence gating, recycled
  belief-state resets, timestamp/finite contracts, projected ROI covariance,
  out-of-view lifecycle handling, and one-authority miss uncertainty.
- Replaced MAE-only perception selection with a versioned runtime-qualified
  MAE/recall/precision/F1 selector using exact pooled counts. Capacity-limited
  births now prefer the strongest qualified proposal.
- Removed unsupported zero objectives, detached already-supervised state and
  structured-RGB mean paths from their variance-calibration NLLs, and aligned
  collision-conditioned samples to a scored event endpoint.
- Added phase-specific MPS/CPU execution with exact device/handoff markers,
  executable-source fingerprints, tensor-linked measurement selector
  artifacts, and no-op resume preservation.
- Changed ID-switch evaluation to use an independent framewise geometric match
  instead of the training-only locked target mapping that could report zero
  switches after a real track swap.
- Advanced the specification to 1.6 and simulator metadata to
  `sphere_world_v3` for the corrected event/contact physics contract.
- Expanded the specification with identifiable-target, safe-incumbent,
  exact-resume, RNG-isolation, and convergence-evidence contracts.
- Verification results are recorded in `project/STATUS.md` after the final
  hybrid-device smoke; short smokes remain wiring evidence rather than
  accuracy claims.

### Added

- Per-axis per-horizon rollout loss records and fixed-denominator aggregation,
  matching the existing global multistep objective instead of renormalizing
  short-only x/y/z windows to full strength.
- Joint collision/maximum-horizon window sampling that satisfies both intents
  when possible and preserves a sampled long-horizon example when a collision
  is too late.
- Explicit `gradient_norm_pre_clip`, `gradient_clip_coefficient`, and
  `gradient_norm_applied` training metrics so clipped batch-one updates are not
  misread as exploding gradients.
- A finite-state/optimizer/loss audit of the active sustained MPS campaign,
  including the exact reason the first causal candidate was rejected.
- A provenance-verifying convergence supervisor for the sustained MPS
  campaign. It monitored the then-existing trainer, removed its completed KeepAlive
  job only after artifact verification, resumes `last.pt` sequentially in
  complete causal blocks, survives supervisor restarts without overlapping
  trainers, and records child failures without infinite retries. The
  historical supervisor ran as
  `com.polceanum.orpheus.convergence-20260730-192625`.
- A strict four-validation plateau decision over accepted/rejected numbered
  candidates, with 1% recent-safe improvement continuation, inconclusive-
  evidence continuation, and a 24,576-step hard limit reported separately as
  `limit_hit` unless plateau is actually demonstrated.
- The original 12,288-update sustained accuracy campaign was launched on Apple
  MPS under the historical persistent macOS job
  `com.polceanum.orpheus.sustained-20260730-192625`; its timestamp-first
  artifact root is `runs/20260730-192625-scaled-sustained-e2e-v1/`.
- A sustained eight-scenario MPS accuracy profile with two complete
  measurement passes, one complete causal-window pass, explicit minimum/plateau
  rules, and the selected fixed-scale point/scale runtime as weights-only
  initialization.
- Pooled physical rollout checkpoint selection: horizon-weighted position RMSE
  plus fixed-reference and moving-incumbent guardrails for current velocity,
  every horizon, 0.5 m distance-gated recall/precision and identity, forecast
  lifecycle coverage, collision F1, and nominal-90% calibration.
- Exact validation protocol and seed-manifest hashes, tensor-hash-verified
  incumbent/reference checkpoints, numbered validation snapshots, and a fresh
  causal optimizer state at phase handoff.
- Bounded training rollout anchors and shared posterior rollout tensors,
  preserving every online observation/state loss while reducing recursive
  causal cost; validation retains all posterior anchors and omits only its
  redundant prior future rollout.
- `project/ACCURACY_AUDIT.md`, separating comparable broad results from
  context-specific gains, rejected recursive interventions, and direct
  evidence of scaled-model undertraining.
- A disabled-by-default gravity-axis RGB velocity intervention with 21 causal
  gravity/lateral/contact/prior/candidate features, acceleration-aware slope
  context, soft uncertainty gain, and strictly unobserved non-gravity
  covariance.
- On-policy feature collection from an intervention-enabled checkpoint,
  enabling a dataset-aggregation refit against the belief distribution created
  by the correction itself.
- Gravity-intervention evaluator diagnostics for eligible features and soft
  gain activation.
- A disabled-by-default intervention-aware camera-lateral RGB velocity
  proposal using exact pre-correction belief state, 19 causal axis-local/joint
  features, a bounded delta, continuous soft abstention through measurement
  variance, folded feature normalization, and post-filter fitting.
- Lateral-intervention evaluator diagnostics for eligible feature count, mean
  soft gain, and gains above one half, plus checkpoint compatibility and
  focused filter/runtime tests.
- A quality-aware persistent-ID point/scale trajectory observer with separate
  bounded point and scale-anchor rings, axis-local robust IRLS fitting,
  timestamp extrapolation, typed direct position evidence, and conservative
  camera-depth belief correction.
- Explicit global scale quality that rejects image-boundary truncation and
  overlap-split components as depth-history anchors while retaining their
  useful RGB centres.
- A strict weights-only curriculum initializer (`train.py --initialize-from`)
  that records provenance and resets optimizer, scheduler, RNG, and steps
  instead of disguising changed simulator semantics as a resume.
- Persistent belief-ID-to-target alignment for closed-loop supervision, so
  nearby objects cannot silently exchange targets during contacts.
- Disabled-by-default, boundary-gated ROI component-scale measurements for
  investigating fast monocular depth without trusting truncated crops.
- Repository agent policy and project memory anchored to the user-provided
  authoritative Project Orpheus specification.
- Plain-PyTorch packaging, strict dataclass/YAML configuration, device
  reporting, deterministic seeding, and local atomic IO.
- Deterministic 3-D sphere physics with gravity, drag, restitution, walls,
  pair collisions, calibrated fixed/orbit cameras, depth-ordered RGB rendering,
  visibility/segmentation labels, events, and disjoint data splits.
- Typed persistent `WorldBelief`, padded object/camera state, uncertainty,
  lifecycle/IDs, hypotheses, packing, validation, and arbitrary-time
  `BeliefTrajectory`.
- Hybrid dynamics: analytic kinematics/quaternions, stable modal state,
  structured graph interactions, contact/event jumps, learned residuals, and
  uncertainty propagation.
- Debug-only oracle observation adapter and a real RGB adapter with global
  discovery, calibrated projection/back-projection, Jacobian uncertainty,
  residual ROI updates, and differentiable MPS fallback.
- Explicit Hungarian association, innovation, surprise scheduling, robust
  analytic/learned correction, same-timestamp asynchronous grouping, and
  identity-aware cache invalidation.
- Projected geometric occlusion, occlusion-aware lifecycle/existence,
  out-of-view separation, reappearance, and missed-state uncertainty growth.
- Observability-gated recurrent drag/restitution/mass/friction/geometry updates
  that modify persistent state without optimizer steps at runtime.
- Global and fast-path measurement supervision, causal closed-loop training,
  injected perturbations, horizon-weighted rollout/event/uncertainty/parameter
  losses, JSONL logs, resume, and separate best measurement/rollout
  checkpoints.
- Held-out RGB-only evaluation with static, constant-velocity, analytic, and
  labelled oracle-parameter analytic baselines; fair common masks; assignment,
  distance-gated detection/identity, calibration, events, actual identifier
  gates, latency, and JSON/Markdown reports.
- RGB-only demo frames/GIF with actual scheduled measurements, prior/posterior
  errors, future revision, predicted events, uncertainty, and slow-parameter
  plots.
- Focused unit/integration tests for simulator, contracts, dynamics, filtering,
  RGB global/ROI paths, MPS gradients, occlusion/lifecycle, checkpoint
  compatibility, CLI train/resume/evaluate, and oracle exclusion.
- Explicit fresh-validation seed protocols with persisted non-overlap
  provenance, collision-conditioned forecast/baseline metrics, and direct
  current/ordinary-correction velocity evidence.
- An opt-in bounded persistent-ID RGB temporal position history, causal
  least-squares velocity measurement, and post-association velocity-only
  correction with explicit uncertainty and availability diagnostics.
- Optional RGB-only structured sphere localization: robust row-background
  subtraction, foreground components, touching-disc distance-peak splitting,
  photometric centroids, global proposal alignment, and projected-ROI local
  refinement.
- Direct raw learned-centre supervision so a structured forward measurement
  does not hide detector/ROI localization error during training.
- Calibrated metric world-position RGB Huber/NLL losses with explicit
  per-term weights.
- Collision-conditioned TBPTT window sampling with causal RGB prefix burn-in.
- Explicit `evaluate.py --seed-offset` support for reproducible paired
  selection and confirmation manifests.
- A deterministic 32-frame multistep profile with recursive
  0.10/0.25/0.50/0.75/1.00-second horizons and explicit long-window sampling.
- Stable demo world bounds, manual legends, recency-faded absolute forecast
  history, matched future endpoints, per-frame absolute errors, and
  scoring-only lookahead that keeps the displayed horizon fixed.
- Sortable UTC timestamp prefixes for all newly generated training,
  evaluation, and demo directory names, including explicit CLI labels.
- A recoverable `demo_outputs/archive/` containing the eight superseded demo
  sets; the current RGB-only demo remains directly under `demo_outputs/`.
- Camera-parallax, glancing-impact, and unequal-mass deterministic scenario
  presets, expanding the mixed curriculum from four regimes to seven.
- Camera-Jacobian projection of posterior world covariance for correctly
  oriented image-space uncertainty ellipses with an explicit legend entry.
- Predictive-abstraction contracts and documentation: an explicit registry,
  point-trajectory versus rigid-sphere routing, and a parameter-free,
  reversible belief-token sequence containing scene, camera, entity
  kinematic, dynamical-programme, and lifecycle tokens.
- Runtime accessors for current abstraction assignments and transformer-ready
  belief tokens. Both are derived on demand from `WorldBelief` and add no
  checkpoint parameters.
- A balanced `tiny_all_scenarios.yaml` profile covering reference, baseline,
  elastic, damped, impulse, camera-parallax, glancing, and heavy/light
  interactions with one shared runtime checkpoint.
- Per-evaluation simulator version, ordered scenario mixture, and exact
  episode-scenario provenance.
- A closed-loop `dynamics`-only trainable scope for controlled shared-model
  adaptation experiments.

### Changed after integration audit

- Restricted closed-loop scopes now freeze the checkpoint-compatible ROI event
  and identifier variance heads while their outputs have no end-to-end
  objective. The active sustained campaign keeps its original semantics for
  protocol integrity; corrected behavior is the default for new configs.
- The scaled curriculum enables the confirmed multi-frame camera-depth
  observer. Across two disjoint paired two-episode MPS blocks, current
  position RMSE and every 0.1–1.0-second recursive position horizon improved;
  the velocity/collision-F1 regression remains an explicit follow-up rather
  than being hidden as an overall model promotion.
- The scaled accuracy curriculum now uses identifiable fixed sphere scale and
  re-anchors global RGB tracks every three frames while preserving two fast ROI
  updates per cycle.
- Closed-loop TBPTT sampling now guarantees at least one valid future forecast
  anchor; late event conditioning can no longer spend a full causal update
  with zero rollout loss.
- Scaled velocity terms receive equal weight to position terms, and the
  `state_dynamics` scope freezes perception while adapting dynamics, filter,
  and online identification.

- Resumed rollout selection now rejects inherited best scores when validation
  count, scenario mixture, sequence length, object-count range, project seed,
  horizons, or selection-metric semantics differ.
- RGB temporal-history collision resets are edge-triggered, allowing outgoing
  velocity evidence to accumulate while collision mode remains active.
- The authoritative specification is now version 1.4 and requires identical
  explicit episode manifests, fixed-reference broad checkpoint guardrails,
  verifiable incumbent weights, and declared minimum/plateau training coverage
  for checkpoint comparisons and per-scenario gates applied to one shared
  model.
- Delayed analytic contact jumps until the estimated geometric gap plus a
  `0.25σ` position-uncertainty margin reaches contact, reducing premature
  lateral damping on the selected multistep block.
- Consolidated the accepted checkpoint, evaluation, scenario audit, and
  rejected-training record into one timestamp-first run; permanently removed
  64 superseded run directories after explicit user authorization.

- Preserved positive observation `dt` through training/evaluation so RGB
  position differences can inform velocity correction.
- Compared scheduler uncertainty in metres of standard deviation rather than
  variance and calibrated the current cost/uncertainty thresholds.
- Made model and baseline forecast masks identical and report dropped tracks.
- Added semantic checkpoint validation before state loading.
- Prevented pretraining validation from being labeled best rollout.
- Replaced ungated “recall” labels with assignment coverage plus 0.5 m
  distance-gated detection.
- Replaced label-proxy parameter observability metrics with actual runtime
  observability, gate, and update diagnostics.
- Made demo overlays use the measurements actually scheduled by runtime and
  aligned every forecast with available ground-truth time.
- Fixed deterministic pretraining frame selection so every fixed episode sees
  every frame instead of being permanently coupled to even/odd frame indices.
- Added multi-frame calibrated RGB localization metrics and changed perception
  checkpoint selection from summed NLL to world-position MAE.
- Restored the best physically localized perception checkpoint at the
  closed-loop boundary and added a configurable 0.1x phase learning-rate
  handoff that also applies after resume.
- Added a reliability gate for the fast inverse-depth delta after diagnosis
  showed that component doubled signed depth error; fast centre and other ROI
  residuals remain active.
- Expanded the deterministic tiny profile to 64 measurement plus six
  closed-loop steps across eight fixed training and four validation episodes.
- Added regression tests for complete fixed-frame coverage, learning-rate
  bounds, and the fast-depth gate.
- Added a configurable closed-loop global-perception adaptation window; the
  backbone/global detector freeze afterward while ROI/filter/dynamics training
  continues.
- Changed fast ROI supervision from one freely rematched frame to every usable
  prior frame with targets aligned through persistent belief slots.
- Defined rollout collision logits as per-segment occurrence, aggregated
  impacts across all internal substeps, and aligned training/evaluation to
  exact preceding observation windows.
- Added bounded rare-positive collision BCE weighting and tests for exact
  event-window query geometry.
- Supplemented correction sparsity with current/future posterior-improvement
  hinges against a detached prior and separate diagnostics.
- Added visible-to-occluded-to-visible uncertainty/identity metrics that track
  the established persistent ID while hidden rather than requiring a current
  localization match.
- Added informative before/after restitution and drag metrics, including signed
  error reduction and physical update magnitude; these expose the current
  identifier's numerically negligible updates.
- Corrected direct-velocity filtering so a modality must provide explicit
  world velocity, log variance, and validity rather than accidentally treating
  unrelated RGB value dimensions as velocity covariance.
- Separated persistent temporal measurement history from disposable ROI
  feature caches; global discovery can invalidate feature crops without
  erasing safely associated same-ID motion evidence.
- Excluded invalid graph edges and diagonals from learned event max-pooling, so
  a valid negative residual can suppress rather than be clamped by stored
  zeros.
- Changed checkpoint validation to treat structured and temporal RGB controls
  as measurement/fusion semantics. Legacy checkpoints missing those fields
  normalize to historical defaults rather than silently adopting requested
  non-default behavior.
- Changed checkpoint validation from one rotating batch/short prefix to every
  configured validation episode and a complete causal unroll.
- Split current-state and rollout position/velocity objectives and changed
  best-rollout selection to physical position loss with a meaningful minimum
  delta.
- Changed rollout and future-correction aggregation to average all eligible
  anchors per physical horizon before applying configured weights, with a
  fixed denominator that does not inflate short-only tail windows.
- Versioned rollout-selection semantics, persisted per-horizon validation
  losses, and reset incompatible inherited best scores on resume.
- Added a maximum-horizon-capable window preference beside collision-priority
  sampling.
- Added independent position/velocity perturbation magnitudes and representative
  collision-window sampling.
- Enabled synthetic structured centers in sphere profiles, with a measured
  `0.08` foreground threshold and noise regression for `toy_hard`/cloud.
- Continued the selected step-584 perception state for 64 causal closed-loop
  RGB updates and promoted validation-selected step 648 after paired ROI-local
  selection/confirmation forecast and collision evidence.

### Evidence

- Deterministic CPU run:
  `runs/convergence-tiny-cpu-v1`.
- Held-out report:
  `runs/convergence-tiny-cpu-v1/evaluation/best-test`.
- Wider eight-episode held-out report:
  `runs/convergence-tiny-cpu-v1/evaluation/best-test-8episodes`.
- Frozen-perception continuation and corrected-event evaluation:
  `runs/accuracy-closed-frozen-94/evaluation/last-test-8episodes-exact-events`.
- Exact-window semantics-only report:
  `runs/accuracy-events-v2/evaluation/pretrain-checkpoint-test-8episodes`.
- Negative balanced-event continuation:
  `runs/accuracy-events-balanced-102/evaluation/best-test-8episodes`.
- Demo:
  `demo_outputs/archive/20260726-223129-convergence-tiny-cpu-v1`.
- Frozen-continuation demo:
  `demo_outputs/archive/20260726-232939-accuracy-closed-frozen-94`.
- Reduced real MPS training:
  `runs/milestone1-mps-smoke-final`.
- Reduced two-step run of the full-size `toy_mps` architecture:
  `runs/milestone1-toy-mps-scaled-smoke`.
- Final 12-step public CPU smoke:
  `runs/accuracy-final-smoke`.
- Fresh 16-episode validation baseline and temporal ablations:
  `runs/temporal-rgb-evidence`.
- Negative 22-step temporal continuation and paired fresh validation reports:
  `runs/temporal-continuation-94`.
- Structured-center step-72 selection report:
  `runs/accuracy-structured-peak-v2/baseline-step72-fresh32`.
- Successful controlled RGB depth fine-tune:
  `runs/accuracy-depth-finetune-v1`.
- Paired 32-episode candidate selection report:
  `runs/accuracy-structured-peak-v2/depth-finetune-best-fresh32`.
- Untouched 32-episode candidate confirmation and paired step-72 baseline:
  `runs/accuracy-structured-peak-v2/depth-finetune-best-confirm32` and
  `runs/accuracy-structured-peak-v2/baseline-step72-confirm32`.
- Rejected scratch training report:
  `runs/accuracy-structured-peak-v2/scratch-best-fresh32`.
- Promoted accuracy-v4 continuation, paired validation, and final test:
  `runs/accuracy-closed-structured-v4`.
- Promoted accuracy-v4 RGB-only demo:
  `demo_outputs/archive/20260727-102411-accuracy-closed-structured-v4`.
- Fresh 32-frame one-second baseline and rejected continuation reports:
  `runs/accuracy-multistep-v1`,
  `runs/accuracy-multistep-balanced-v4`, and
  `runs/accuracy-multistep-long-v5`.
- Stable full-horizon forecast-history demo:
  `demo_outputs/archive/20260727-125455-accuracy-v4-forecast-history`.

The earlier frozen continuation reached 75.39% distance-gated
recall/precision over eight held-out episodes, 0.161387 m 0.5-second forecast
RMSE versus 0.490275 m for constant velocity, zero gated ID switches, and
positive perturbation recovery. At that stage, collision F1 was only 0.042553
on fresh validation. This is historical evidence; the current step-648 result
is recorded below.

The continuation and event-loss comparisons repeatedly inspected the same
eight fixed test episodes and were therefore exploratory. Step 72 remained
selected at that stage; it was later superseded by the paired step-584 and
step-648 validation protocol. Run and demo artifact directories are local and
gitignored.

The new fresh-validation protocol evaluated the unchanged selected step-72
checkpoint on seeds `100004–100019`: current position MAE was `0.186991 m`,
0.5-second RMSE `0.174269 m`, collision F1 `0.042553`, and perturbation
recovery `20.09%`. Temporal inference improved velocity RMSE and short
collision/event metrics but regressed localization, aggregate forecasting, and
perturbation recovery under every tested history/variance setting, so it
remains disabled. The temporal continuation reached F1 `0.121622` but was
also rejected because position MAE rose to `0.196397 m`, 0.5-second RMSE to
`0.184454 m`, and perturbation recovery fell to `11.84%`.

The RGB-only structured localization and controlled step-584 measurement
fine-tune were evaluated on explicit disjoint manifests. On validation seeds
`100064–100095`, the step-584 checkpoint reached current position MAE/RMSE
`0.085103 / 0.110556 m`, recovery `48.28%`, and collision F1 `0.622222`;
the paired step-72 baseline had RMSE `0.131311 m`, recovery `42.27%`, and F1
`0.568182`. Those reports used the older full-frame ordinary refinement.

After ordinary structured measurement was restricted to projected ROIs,
accuracy-v3 model choices were frozen before a reserved standard-test run on
seeds `200064–200095`. That then-final source/checkpoint pair reached position
MAE/RMSE
`0.090847 / 0.118600 m`, velocity RMSE `0.812524 m/s`,
0.10/0.25/0.50-second forecast RMSE
`0.141520 / 0.181431 / 0.237585 m`, perturbation recovery `45.72%`,
collision F1 `0.597122`, 100% distance-gated detection, zero ID switches,
86.62% nominal-90% coverage, and no dropped/non-finite forecasts. Report:
`runs/accuracy-roi-local-v3/final-test32/report.md`.

The longer scratch run at
`runs/accuracy-structured-physical-v1` was explicitly rejected: on its
selection manifest it produced position MAE/RMSE
`0.273908 / 0.422225 m`, 0.50-second RMSE `0.525709 m`, recovery `18.80%`,
collision F1 `0.316940`, and detection recall/precision
`66.60% / 55.22%`. A 2 cm anticipatory collision-hazard experiment was also
removed after F1 fell from `0.622222` to `0.594406` on the untouched
confirmation block.

A later 256-update measurement continuation with the completed raw-centre
objective did not beat the inherited validation MAE and degraded as high as
`0.291207 m`; `runs/accuracy-final-perception-v3` is retained as a rejected
experiment rather than promoted.

Accuracy-v4 supersedes the step-584 promotion. The step-648 checkpoint
(`9b943f60128a2bd15298847d8c7de4dd3166646f3644720a3149155e57d85bcd`)
was selected at full-validation rollout-position loss `0.0119829765`. On paired
confirmation it improved all three forecast horizons and collision F1
`0.594203 → 0.608059`, with small mixed current/velocity/recovery tradeoffs.
The frozen final test reached position MAE/RMSE
`0.089336 / 0.116908 m`, velocity RMSE `0.792257 m/s`,
0.1/0.25/0.5-second RMSE `0.138279 / 0.177703 / 0.232862 m`, recovery
`45.30%`, collision precision/recall/F1
`0.765217 / 0.550000 / 0.640000`, 100% detection, zero ID switches, and no
dropped/non-finite forecasts. An exhaustive threshold probe was negative
because logits were saturated; mean-radius and photometric analytic-depth
probes were also rejected.

The new 32-frame multistep baseline on fresh-validation seeds
`100096–100111` reached recursive position RMSE
`0.162863 / 0.190546 / 0.218011 / 0.230611 / 0.228255 m` at
0.10/0.25/0.50/0.75/1.00 seconds. Aggressive, balanced, and conservative
one-second continuations failed to improve the mean 0.50–1.00-second result,
so step 648 remains promoted. Oracle-start and learned-dynamics ablations
showed that RGB state/velocity and slow-parameter estimation, rather than the
recursive dynamics residuals, dominate the remaining error.

On 2026-07-27 the misleading XY/GIF overlay was corrected: ground-truth past
and current horizon are no longer drawn on top of each other, identities use
stable colours and labels, and start/end/time-direction markers make the
two-sphere collision trace explicit. The final regenerated artifact is
`demo_outputs/20260727-162848-accuracy-v6-blended-velocity/online_correction.gif`.

The same investigation fixed the model's near-zero camera-lateral velocity
gain with bounded persistent-ID temporal evidence restricted to young tracks.
The selected `accuracy-lateral-velocity-v5` checkpoint improves current
position/velocity RMSE and every recursive 0.1–1.0 s forecast horizon on
fresh-validation seeds `100096–100111`; the exact report is
`runs/accuracy-lateral-velocity-v5/evaluation/select16/report.md`. Collision F1
and nominal coverage regress slightly and excessive early collision damping is
still visible, so event timing/calibration remains open. Stronger continuous,
raw-measurement, and short adapted-training variants are recorded as rejected
runs rather than promoted.

A deterministic four-regime interaction suite was added for baseline,
high-restitution elastic pairs, damped high-friction contacts, and labelled
external impulses. `configs/tiny_interactions.yaml` exercises three-object,
40-frame episodes while preserving the canonical RGB/episode/belief
contracts. Scenario names are recorded in episode metadata and singleton
scenario overrides support paired evaluation.

Two mixed training attempts were completed on CPU. The eight-step closed-loop
run improved impulse-driven forecasts by about 3% but was neutral elsewhere.
The subsequent eight-step RGB plus eight-step closed-loop adaptation improved
velocity while broadly regressing position and three-object detection recall.
Both are retained under `runs/accuracy-interactions-v1` and `v2` as rejected
experiments; the lateral-velocity step-648 checkpoint remains selected.

The reference-physics investigation advances the specification to 1.2 and the
sphere dataset metadata to v2. A new `reference_pairs` scenario separates the
ensured pair impact from the first floor impact, uses familiar low-friction
parameters, and has an integration regression for post-impact separation and
event timing. Ensured-pair height, surface gap, and speed are now configurable.
The specification and agent guide also define a future mature physics engine as
an optional independent RGB dataset backend, never privileged runtime input.

Evaluation now reports x/y/z current and every-horizon position errors plus
axis-resolved velocity/correction evidence. Training can weight rollout
position by axis while retaining joint interactions and events. Structured RGB
discs now provide both point centers and equivalent-area radii; the reference
profile uses the known sphere radius to form calibrated analytic depth evidence
without simulator-state input. Position confidence is separate from existence
confidence so a trustworthy matched point cannot create spurious tracks.
Temporal velocity history can reopen for a bounded number of post-event samples,
and global scheduling now correctly accumulates consecutive association
failures.

Two short CPU continuations completed on the clean reference curriculum. The
second selected step 672 by its internal two-episode rollout loss, but a
four-seed external test showed essentially no improvement over the inherited
weights; it is retained only as the weight source for the parameter-free
structured runtime. Denser global cadence, direct raw-point velocity history,
fast-path structured confidence, and zero learned-correction scale were tested
and rejected because they failed to improve 1-second x error. This is not
claimed as a solved accuracy result.

An additional held-out audit separated the RGB abstraction into point and
scale errors. Disc centres were already subpixel accurate, while single-frame
equivalent-area scale produced heavy-tailed depth errors under overlap and
truncation. RGB observations now support disabled-by-default, tested
depth-disagreement covariance inflation and optional simultaneous
temporal/position-innovation velocity evidence. Neither policy is enabled:
both failed the untouched 1-second test gate. A 128-step balanced RGB
continuation was also rejected after paired 1-second RMSE worsened. The
selected checkpoint and published metrics remain unchanged; ADR-046 records
the multi-frame point/scale trajectory design supported by the evidence.

Added `configs/scaled_curriculum.yaml`: one 1.90M-parameter shared RGB world
model trained from 4,096 deterministic, continuously varied episodes across
eight balanced scenario families, with separate 256-episode validation/test
splits and an OOD path. Training plans and summaries now report capacity,
episode draws, nominal dataset passes, and scenario families. The chosen
48,000-example schedule uses batch-one/eight-step causal graphs and four
renderer workers after a bounded MPS run exposed the memory and simulation
throughput limits of batch four. Cross-device checkpoint resume now restores
the default PyTorch RNG state on CPU even when model tensors map to MPS.

Continued the scaled MPS checkpoint from step 256 to a persisted step-896
measurement checkpoint. On two fresh-validation episodes it improved current
position and 0.1–0.75-second forecasts, detection, and collision F1, but
regressed velocity, calibration, and the 1.0-second forecast; it is recorded as
mixed evidence rather than promoted.

Hardened long MPS runs by filtering non-finite structured proposal rows,
raising on non-finite validation losses, checkpointing final weights before
expensive validation, and transferring evaluation diagnostics to CPU before
float64 accumulation. The last change fixes a directly reproduced MPS-only
evaluation crash.

Added an opt-in acceleration-aware causal temporal velocity estimator. It
removes known quadratic acceleration before fitting velocity at the current
timestamp and can expose only camera-lateral plus post-event gravity-axis
evidence, leaving monocular depth velocity unobserved. Focused tests cover the
endpoint-bias correction and subspace projection. Matched MPS ablations found
useful vertical-velocity signal but no net multistep/event promotion, so the
scaled default remains disabled and the reports record the rejected policies.

Added an opt-in causal RGB trajectory change-point path. It compares the latest
three point observations after removing known gravitational acceleration,
preserves the independently validated scale-anchor history when reopening the
kinematic window, and permits a two-sample gravity-only outgoing correction.
Evaluation now reports trigger counts and rates. Paired MPS ablations rejected
both a noisy permissive gate and an endpoint-contact gate that was too sparse;
the scaled profile therefore records the feature and thresholds explicitly but
keeps it disabled.

Added an offline-supervised, online-cheap RGB trajectory-gate workflow.
Exact history timestamps align simulator-only training labels with the causal
RGB windows that produced each feature vector. Nine uncertainty-aware features
feed either logistic regression or a tiny one-hidden-layer MLP; learned
coefficients live in explicit runtime config, and cached feature tensors permit
threshold/model refits without rerunning perception. The artifact writer
preserves source checkpoint training/seed provenance. Linear, loose-MLP, and
sparse-MLP policies were all rejected for promotion: the sparse policy was
safe but produced a small velocity regression on one paired seed.

Added a tiny optional bounded outgoing gravity-velocity proposal behind the
learned RGB change-point gate. Training caches now include the belief prior,
aligned simulator-only supervision target, and delta; the proposal consumes
nine causal RGB/contact features, prior gravity velocity, and gate
probability. A refractory interval prevents repeated feedback triggers.
Runtime consumes the proposal once on the exact frame selected by its causal
gate, fixing an initial train/application timing mismatch. Positive-only and
joint gate-focused fits improved offline RMSE, and the aligned runtime slightly
improved current and 0.1-second position on one seed, but velocity regressed.
The gate and proposal therefore remain disabled in the scaled profile. The
next target is an intervention-aware camera-lateral outgoing correction, not
further gravity-axis threshold tuning.

## 2026-08-09 — specification 1.19 correction integrity and scale path

- Stopped and rejected protocol 18 at durable step 128 after exact fixed
  validation worsened current position and all five forecast horizons.
- Preserved exact dynamics-only, updater-plus-identifier, and updater-only
  ablations that localize the regression to the learned fast updater.
- Added opt-in innovation-anchored learned correction with explicit per-axis
  world-state evidence, support/confidence masking, and zero-innovation mean
  invariance while preserving legacy checkpoint behavior by default.
- Added focused axis-local/support-mask/config/checkpoint tests (`142 passed`),
  including legacy-false checkpoint normalization; full non-device regression
  reports `636 passed, 5 skipped, 1 deselected` and host MPS reports `1 passed`.
- Advanced the specification to 1.19 with a staged abstraction-token attention
  ladder from a 1--4M parameter Mac pilot to later CUDA-scale latent video
  pretraining, gated by disjoint generalization and broad non-regression.
- Ran the exact 32-episode inherited-head protocol-19 qualification. Pooled
  score improved slightly, but velocity, short horizons, and multiple scenarios
  regressed; one finite supported balanced update then worsened every x metric.
  The candidate is rejected rather than promoted.
- Added deterministic fresh-initialization module donation to the modular
  qualifier so changed corrector output heads can be reset and evaluated with
  explicit seed/prefix provenance before sustained training.
- Rejected the full mean/variance/gate reset (`0.350730`) and the semantically
  precise mean-only reset (`0.324176`) against the inherited-gain candidate;
  preserved both exact 32-episode reports and selected mean-only solely as the
  clean mutable recovery state.
- Added an updater-only causal trainability scope so the new gain head can
  recover broad correction accuracy without simultaneous dynamics,
  identification, or perception drift.
- Launched clean one-shot protocol-19 updater recovery from the mean-reset
  candidate: 512 balanced updates, validation/checkpoint every 64, 2e-5
  updater LR, MPS measurement plus CPU closed loop, Standard QoS, no oracle.

## 2026-08-10 — protocol-19 recovery initialization and causal health

- Completed the recovery run's exact 32-episode RGB-only initialization in
  `700.677 s`; score `0.3241755` exactly reproduced the independent mean-reset
  qualification.
- Verified updater-only optimization at logged steps 8 and 16: finite
  unclipped gradients, frozen perception and interaction dynamics, all eight
  scenarios per update, nonzero causal trajectory support, and zero skipped
  draws.
- Confirmed that the cold-start step-8 row's eight objectives were intentional
  maturity gating rather than lost rollout supervision; the mature step-16
  row restored all 13 objectives including deterministic multistep position
  and velocity losses.
- Reached the durable step-64 validation boundary with eight clean logged
  blocks, equal scenario counts, no skipped or clipped updates, median gradient
  `0.3560`, median trajectory support `334`, and a warning-free dynamics
  audit; the exact trained selector is in progress and no promotion is claimed.
- Completed and rejected the step-64 selector: score worsened from `0.324176`
  to `0.338432`, current position and all five pooled horizons regressed, and
  elastic-pair x forecasts failed severely despite better velocity and
  collision F1.
- Found that the nominal updater recovery changed the already-compatible
  trunk, variance, gate, mode, and existence paths as well as the reset mean
  head. Added an `updater_mean` scope that exposes exactly the mean-head weight
  and bias, with focused schedule/config coverage.
- Launched a clean 512-update replacement from the untouched mean-reset
  candidate with mean-head-only trainability, effective LR `5e-6`, MPS RGB
  measurement, CPU closed loop, and exact validation every 64 updates.
- Completed corrected step-64 validation. Exact tensor comparison confirmed
  only the mean-head weight/bias changed. The candidate was rejected by 16
  guardrails with a near-flat score (`0.324176 -> 0.324672`), versus 110
  failures and score `0.338432` under updater-wide training; all pooled x
  metrics improved while small y/z and velocity regressions remained, mainly
  in `reference_pairs`. Continued the declared run toward step 128.
- 2026-08-13: Reviewed the original AAAI ORPHEUS and ToM simulation papers. Recorded ADR-118: preserve the persistent point/trajectory abstraction but add short-step, error-selected hypothesis rollouts as the next accuracy experiment; no active run was modified.
## 2026-08-14 — position-gated event-aware selector

- Added an opt-in position gate to hypothesis scoring so event/lifecycle
  evidence cannot override a materially better position forecast.
- Standardized collision-event indexing on `MotionMode.COLLISION` and added
  focused coverage for the gate and synthetic event selection.
- Completed an eight-episode protected RGB comparison. The gated selector was
  non-regressive in all pooled position axes/horizons and slightly improved
  collision F1, but remains opt-in pending a larger disjoint evaluation.
- Completed a disjoint two-episode pilot (`seed=200000`) as a guardrail check;
  it had mixed long-horizon y/event results and was rejected as promotion
  evidence. The gate remains opt-in.
- Switched evaluation-only rollout execution from `torch.no_grad()` to
  `torch.inference_mode()` and verified a one-episode RGB smoke report.
- Completed the first optimized eight-episode disjoint gated-selector matrix;
  x/z improved at some long horizons, but y and collision F1 regressed at the
  tail, so promotion was rejected and the learned default was preserved.
- Added validated opt-in `--axis-weights X Y Z` scoring through the selector,
  pool, runtime, and evaluator. A two-episode `1 2 1` pilot improved long-
  horizon y but regressed z/event metrics and was rejected.
- Added opt-in `--axis-gate-ratio` scoring, which blocks candidates with a
  materially worse error on any axis. The two-episode pilot was conservative
  but had a small 0.25 s y regression, so it remains unpromoted.
- Completed the larger eight-episode per-axis-gated matrix. It improved x/y at
  long horizons and preserved lifecycle/identity metrics, but small z and
  collision-F1 regressions remain; default promotion was rejected.
- Added opt-in posterior position blending for evaluator experiments. The
  fresh pilot improved x but regressed z across horizons and y mid-horizon, so
  it was rejected under the all-axis guardrails.
- Corrected blended uncertainty reporting to propagate within- and
  between-hypothesis mixture variance; verified a real one-episode smoke.
- Exposed `--temperature` for posterior calibration. A `0.25` blend pilot
  improved long-horizon position error but regressed 0.50 s collision F1 and
  was rejected.
- Ran a position-only sharp blend to isolate event coupling; it still had a
  0.10 s collision-F1 regression and mixed axes, so it was rejected.
- Exposed `--event-threshold`; a two-episode `0.8` pilot was non-regressive and
  slightly improved event F1, but remains opt-in pending larger evidence.
- Completed the larger threshold-0.8 matrix; long-horizon y/z and event F1
  regressed, so the default threshold remains 0.5.
- Corrected ballistic contact dynamics to clamp ground crossings and apply
  restitution velocity jumps; added regression coverage and an RGB smoke.
- Evaluated the corrected ballistic candidate; it improved long-horizon x/y but
  regressed z and had mixed event F1, so promotion was rejected.
- Added vectorized approaching-pair elastic impulses to the ballistic candidate;
  fresh evidence improved y/events but retained small z regressions, so no
  selector promotion was made.
- Full regression suite after contact changes: `753 passed, 6 skipped`; skips
  are expected MPS-conditional tests.
- Expanded the evaluator with a named undamped constant-velocity hypothesis;
  it improved x/y but regressed long-horizon z and was not promoted.
- Added per-horizon hypothesis selection counts to reports and verified the
  variable-candidate accounting with a real smoke evaluation.
- Used horizon-resolved counts to identify ballistic short/long wins and
  undamped-velocity sparsity; scheduled a data-driven conditioned prior rather
  than hard-coding candidate exclusions.
- Exposed opt-in uncertainty-aware hypothesis scoring; the pilot suppressed
  undamped velocity but regressed long-horizon z/event metrics and was rejected.
- Added opt-in horizon-conditioned evidence decay; scale `1.0` over-selected
  ballistic at long horizons and regressed y/event metrics, so it was rejected.
- Tested signed prior-preserving decay (`-0.5`); it improved long-horizon x/y
  but regressed z and short-horizon event F1, so it was rejected.
- Added opt-in independent per-horizon posteriors. They improved long-horizon
  x/y and confirmed cross-horizon contamination, but event F1 regressed at
  0.50/1.00 s and promotion was rejected.
- Added opt-in event-error gating; it improved short-horizon event F1 but
  collapsed tail event performance and worsened y/z, so it was rejected.
- Tested event threshold `0.95`; event precision/F1 improved, but long-horizon y
  regressed and the threshold was rejected.
- Added per-horizon event-probability histograms for offline threshold sweeps;
  a real smoke verified histogram totals.
- Made event histograms label-aware (positive/negative) and verified exact
  conservation, enabling truthful offline threshold calibration.
- Added `scripts/sweep_event_histograms.py` to aggregate label-aware bins and
  estimate precision/recall/F1 at conservative bin-aligned thresholds.
- Completed a bounded two-episode attention-checkpoint calibration; learned
  events lead through 0.75 s while ballistic contact leads at 1.00 s, so no
  horizon-specific threshold or candidate was promoted.
- Completed an eight-episode compatible calibration; learned event prediction
  wins every horizon and the earlier ballistic signal does not replicate.
- Added axis-independent delayed-evidence selection. The unrestricted form
  regressed z; the x/y-only form improved both axes at all horizons while
  preserving z, lifecycle, identity, and event metrics in an eight-episode
  guardrail, so it remains opt-in pending fresh-draw qualification.
- Qualified the x/y-only selector on fresh seeds 100--103: x/y improved at
  every horizon, z was exactly unchanged, and lifecycle/identity/event metrics
  were unchanged. Runtime default promotion remains pending an explicit
  per-axis posterior contract.
- Exposed validated per-axis posterior weights/indices through
  `HypothesisSelection`, `HypothesisDynamicsPool`, and `OnlineWorldModel`, with
  an oracle integration test proving `WorldBelief` remains unchanged.
- Added typed evaluation config controls for axis composition. The attention
  pilot config now enables qualified x/y-only composition; other configs keep
  the joint default, and a no-CLI-flag smoke verified the resolved setting.
- Re-ran the attention dry run and full suite after promotion: `755 passed,
  6 skipped`; skips are MPS-conditional because MPS is unavailable here.
- Tested injecting the joint persistent prior into coordinate posteriors; it
  erased the qualified x/y gains and was rejected. The accepted path keeps
  persistent joint evidence and coordinate-specific delayed evidence separate.
- Added an opt-in axis-prior-strength control; a `0.05` fresh-draw pilot also
  collapsed to baseline and was rejected, leaving the default at zero.
- Qualified axis-prior strength `0.001` across eight episodes and enabled it
  in the attention pilot config; x/y improved while z/lifecycle/identity/event
  metrics remained non-regressive.
- Revalidated the promoted prior on independent seeds 200--201; x/y gains
  persisted with unchanged z/event/lifecycle/identity metrics, and added a
  `--no-axis-independent` baseline override.
- Added `scripts/compare_hypothesis_reports.py` to mechanize per-axis,
  lifecycle, identity, event, and uncertainty regression checks.
- Completed a one-update, eight-scenario attention training smoke on CPU with
  finite loss and validated checkpoints; this remains an entry-point check,
  not a convergence result.
- Evaluated that one-step checkpoint on RGB and rejected promotion after x
  regressed on every horizon versus the protected reference.
- Rechecked the restarted `orpheus` environment: PyTorch 2.10.0 is MPS-built
  but MPS remains unavailable because this runtime is x86_64; added the exact
  hardware/device result and focused audit result (`191 passed`) to status.
- Installed the locally built PyTorch wheel from `~/Work/pytorch` into
  `orpheus` (`2.9.0a0+gitcbe1a35`) with dependencies untouched. MPS remains
  unavailable on this x86_64 runtime; the real MPS tensor smoke failed with
  the backend availability error, while the focused suite remained green.
- Added a permanent agent rule: never choose or install a different PyTorch
  build independently; preserve the user-provided build and change it only
  when the user explicitly supplies or names the replacement.
- Corrected the environment diagnosis: the restored local PyTorch build works
  on MPS in the active Aqua session (`mps:0` matrix smoke passed). The earlier
  unavailable probe ran in the agent sandbox, which cannot access Metal.
- Re-read both original mental-simulation papers and added robust nearby-world
  hypothesis evidence: expected delayed loss plus optional score dispersion.
  The new runtime path preserves `WorldBelief`; focused tests passed
  (`192 passed`).
- Completed the first full active-Aqua MPS project smoke with the supplied
  local torch build. It is finite but not promoted because the one-step final
  1.00 s validation rollout RMSE regressed versus the initial incumbent.
- Added deterministic, uncertainty-scaled nearby-belief rollout sampling and
  RGB evaluator switches for a matched robust-ensemble experiment. The default
  evaluator behaviour remains unchanged.
- Ran and rejected the first matched robust three-world MPS probe: it retained
  the learned candidate for every decision and regressed x/y RMSE at guarded
  horizons, despite unchanged lifecycle/identity/event/calibration metrics.
- Preserved an interrupted 128-step attention campaign's step-zero selector
  checkpoint and launched its required timestamped Aqua/MPS continuation from
  that durable checkpoint. The trainer's numbered-checkpoint resume contract
  was respected; no incomplete run was overwritten or promoted.
- Recorded the continuation's first normal and collision-heavy optimizer
  batches. The step-16 event spike remained finite and passed the pre-Adam
  interaction-retention gate; no numerical safeguard was relaxed.
- Recorded the step-32 post-stress training-health sample: finite state/event
  losses and unclipped gradient, with validation intentionally deferred to the
  configured step-64 full-mixture checkpoint.
- Completed and rejected the step-64 active-Aqua RGB full-mixture validation:
  selector score and aggregate/per-axis rollout RMSE regressed against the
  protected incumbent, alongside coverage, precision, and identity guardrail
  failures. The checkpoint is retained only as diagnosis evidence.
- Completed the 128-update active-Aqua MPS campaign and rejected its final
  checkpoint despite a better global selector score (`0.3097148` versus
  `0.3213162`): z/lifecycle/identity guardrails still regressed. Recorded the
  clean optimizer audit (finite, 128 applied updates, balanced scenarios, no
  collapse) and preserved the incumbent checkpoint.
- Added opt-in axis-selective typed-attention recovery scopes. They restrict
  AdamW updates to exactly one x/y/z node-acceleration row and preserve every
  excluded row and optimizer moment exactly; focused schedule/config tests
  pass (`270 passed`).
- Launched the timestamped active-Aqua MPS `attention_node_z` 512-update
  qualification from the protected incumbent. It is a controlled pending run,
  not a promotion or reported accuracy result.
- Completed that run's step-zero active-Aqua fixed-manifest selector in
  `1232.20 s`. The imported incumbent was reproduced exactly and passed all
  guardrails before any z-only update; recovery training remains pending.
- Attempted a full active-Aqua MPS RGB closed-loop forward/backward smoke to
  qualify replacing the CPU causal fallback. Metal failed before model metrics
  with an interrupted-XPC pipeline compilation error; no device configuration
  or model claim was changed.
- Recorded the z-only recovery's first durable step-16 training health record:
  finite loss/gradients, one applied update, and nonzero causal support under
  the protected z-only scope. It is not an accuracy result or promotion.
- Isolated the full-MPS qualification failure: basic active-Aqua reductions
  remain finite, while typed-attention causal propagation reaches non-finite
  uncertainty before loss construction and the complete backward reproduces
  the Metal pipeline/XPC error. Retained CPU as the safe causal backend.
- Fixed zero-tangent contact normalization in the plane and pair solvers. The
  prior subnormal denominator could flush to zero on MPS and produce a NaN
  before a false collision mask; CPU contact/parity tests and an active-Aqua
  MPS regression test pass. Full RGB MPS qualification remains pending.
- Replaced the training-only MPS integer-ID `amax` with an exact sequential
  elementwise maximum over belief slots. This removes the remaining MPS
  reduction pipeline failure in the bounded full RGB causal smoke; CPU/CUDA
  retain the original reduction and full-pilot qualification remains pending.
- Added and passed an active-Aqua MPS integration regression for the bounded
  complete `attention_pilot_mps` RGB causal graph (`1 passed in 53.64 s`). It
  verifies finite loss and z-attention gradients through persistent
  predict–observe–associate–correct–rollout; it is numerical qualification,
  not an accuracy or throughput promotion.
- Switched future `attention_pilot_mps` closed-loop selector validation to MPS
  after that qualification. Existing CPU-fallback run configurations remain
  immutable and require an explicit MPS guarded replay before promotion; the
  unchanged bounded source-config regression passed in active Aqua (`1 passed
  in 51.06 s`).
- Added `scripts/replay_promotion_mps.py`, an immutable active-Aqua MPS replay
  gate for legacy CPU-fallback candidates. It replays reference and candidate
  checkpoints over the exact trainer manifest and persists the existing
  broad-guardrail decision; no candidate result is claimed yet.
- Bound MPS replay reports to both checkpoint SHA-256 digests, the exact
  validation protocol hash, PyTorch version, precision, and backend.
- Documented the one-shot active-Aqua MPS promotion-replay command and its
  immutable guardrail/provenance contract in the public workflow.
- Fixed horizon-specific selected lifecycle/identity accounting in the
  heterogeneous-pool evaluator; later-horizon rows no longer reuse the first
  query-time active mask. Focused evaluator and hypothesis tests pass.
- Made heterogeneous-pool selected metrics causally valid: score the
  pre-observation posterior/axis/blend and report target-conditioned choices
  separately as delayed-evidence diagnostics. Historical hindsight-conditioned
  pool reports remain diagnostic only.
- Ran the corrected one-episode active-Aqua MPS hypothesis-pool smoke at
  `runs/20260814-130651-causal-hypothesis-pool-mps-smoke`: joint choice remains
  learned on all 148 decisions, while x-axis prior selection sometimes chooses
  analytic alternatives. It is a diagnostic, not a promotion. The evaluator
  now applies UTC timestamp prefixes to its output artifacts.
- Corrected the damped constant-velocity fallback's kinematics: its position
  now uses the exact linear-drag integral rather than undamped displacement
  followed by a velocity-only decay. The zero-damping candidate is unchanged;
  focused hypothesis and online-loop tests pass (`25 passed`).
- Completed and rejected the z-only recovery's first fixed-manifest candidate
  at step 128. Its selector score worsened to `0.3229766` from the protected
  incumbent's `0.3213162`, with 37 broad guardrail failures; the incumbent
  remains intact and no MPS promotion replay was run for this ineligible
  candidate.
- Completed and rejected the z-only recovery's second fixed-manifest candidate
  at step 256. It reduced the first candidate's aggregate RMSE and improved
  collision F1, but selector score `0.3224730` still missed the protected
  `0.3213162` baseline and 20 broad guardrails failed, principally y/z horizon
  and reference-pair dynamics. The incumbent remains intact and the
  ineligible candidate correctly bypassed MPS promotion replay.
- Completed and rejected the z-only recovery's third fixed-manifest candidate
  at step 384. Selector score improved only to `0.3223566`; 23 guardrails
  still failed, particularly y/z horizons, damped-contact events, and
  reference-pair dynamics. This records a plateau in the z-only recovery path,
  not a promotable improvement.
- Completed the bounded 512-step z-only recovery diagnostic. Its terminal
  selector was `0.3223764` against `0.3213162` for the incumbent, with 24
  strict guardrail failures. Finite, causally supported training throughout
  establishes a failed parameterization plateau; no further z-only run or MPS
  promotion replay is warranted.
