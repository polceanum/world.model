# Tasks

## Active convergence target — corrected control before attention scaling

- [x] Diagnose protocol-17 step-2,048 per-axis/per-scenario regression with
  exact learned-corrector scale and updater/dynamics checkpoint ablations.
- [x] Implement deterministic, exact-resumable batches with equal support from
  all eight scenario families and strict configuration validation.
- [x] Complete a real batch-eight checkpoint-initialized causal update plus
  terminal RGB-only validation; verify finite gradients, causal support,
  frozen perception, seed/scenario membership, checkpointing, throughput, and
  maximum RSS.
- [x] Commit and push specification 1.18, stop the superseded protocol-17
  trainer/supervisor at a durable checkpoint, and launch the clean immutable
  balanced campaign from protocol-17 step 512.
- [x] Diagnose the protocol-18 step-64 hard-window gradient by exact episode,
  objective, and interaction output row; verify both declared clips bounded
  the finite update and add explicit severe-clip reporting to the dynamics
  auditor.
- [x] Stop protocol 18 at durable step 128 after exact fixed validation proved
  all-horizon regression; preserve its artifacts and do not resume it.
- [x] Isolate the regression with exact dynamics-only,
  updater-plus-identifier, and updater-only fixed-manifest ablations.
- [x] Repair the learned corrector so explicit per-axis world innovation and
  declared field support mask learned mean/variance residuals, while legacy
  checkpoint semantics remain reproducible by default.
- [x] Run the complete non-device/device regression suites and exact 32-episode
  fixed RGB-only qualification of the innovation-anchored inherited-head
  control; reject it as mixed because old absolute-delta heads cannot be
  reinterpreted as innovation gains despite a slightly better pooled score.
- [x] Qualify deterministic three-head and mean-only resets with composition
  provenance. Reject both as deployment candidates; retain the cleaner
  mean-only candidate as mutable recovery state because only that output's
  mathematical meaning changed.
- [x] Add an updater-only trainable scope that freezes dynamics, identification,
  and perception while the clean innovation-gain head recovers.
- [ ] Complete the active 512-update balanced updater-only recovery from the
  mean-reset
  candidate until the exact 32-episode selector regains the legacy fixed
  reference without axis/horizon/scenario regressions.
- [ ] If the recovered control is non-regressing, train balanced protocol 19
  long enough for repeated fixed validation and a declared plateau; inspect
  pooled and per-scenario
  current/velocity, x/y/z, every horizon, identity, lifecycle, events,
  calibration, support, optimizer state, and memory.
- [ ] Continue through the predeclared balanced minimum and only declare
  convergence after four comparable fixed validations satisfy the existing
  plateau rule; extend rather than promote a broad regression.
- [ ] After the corrected control qualifies, implement the Mac attention pilot:
  2--4 pre-normalized entity/relation/event blocks, width 128, four heads,
  bounded belief history, typed residual decoders, and 1--4M added parameters.
- [ ] Compare the attention pilot against the accepted control and a
  parameter-matched graph/MLP control on disjoint validation/test and OOD
  manifests; scale data with capacity and reject any broad regression.

## Supported sustained scaled accuracy campaign

- [x] Stop and preserve the invalid v2 campaign after proving that zero-gradient
  causal rows consumed scheduled updates and coverage collapsed at the
  measurement handoff. Remove its persistent trainer/supervisor jobs without
  deleting evidence.
- [x] Require causal trajectory/state/parameter or persistent fast-ROI support
  for every optimizer update; count and bound deterministic unsupported draws.
- [x] Repair fast-ROI positive/negative supervision, reliability masks,
  temporal-cache pretraining, false-positive selection precision, and
  separately normalized global/fast losses.
- [x] Add absolute and relative deployment-support guardrails, truthful first
  candidate handling, and verified rollback plus optimizer reset only on
  catastrophic pooled support collapse.
- [x] Align analytic contacts with the labelled simulator and tighten
  observation-gated drag/restitution supervision.
- [x] Diagnose the remaining hard-window gradient spike by subsystem and add
  explicit local interaction clipping before the whole-model clip without
  changing the forward dynamics architecture.
- [x] Require complete persisted current/horizon support and broad
  non-regression guardrails for every declared scenario, so pooled improvement
  cannot hide a missing or collapsed dynamics family.
- [x] Reject duplicate balanced scenario entries and invalid negative or
  non-integral RGB phase boundaries before deterministic validation/training.
- [x] Add and smoke-test `configs/sustained_accuracy_mps_v3.yaml` with one
  shared eight-scenario model and hybrid MPS/CPU execution.
- [x] Re-run the final audited tree on host MPS/CPU at
  `runs/20260802-121629-convergence-v3-final-audit-smoke/`; confirm four finite
  updates, real causal/ROI support, no skipped draw, complete terminal
  checkpointing, and a truthful coverage-based rejection.
- [x] Commit/push clean source `c0acf16` and launch the bounded 3,072-update
  eight-scenario qualification at
  `runs/20260802-123714-v3-medium-qualification/`; verify host MPS/CPU devices,
  clean provenance, launchd persistence, and active computation.
- [x] Stop and preserve that qualification after its first causal validation
  exposed an eight-scenario identity/birth/collision collapse despite better
  conditional RMSE. Do not resume its step-1728 checkpoint under changed
  lifecycle, association, supervision, and optimizer semantics.
- [x] Distance-gate first-time privileged target mappings, pre-gate core and
  tentative Hungarian assignments, and add cardinality counterexamples.
- [x] Implement configurable multi-frame tentative birth confirmation as
  detached modality/sensor-local evidence outside `WorldBelief`.
- [x] Bind prior-conditioned fast ROI measurements to their source persistent
  identity and keep unrestricted gated Hungarian only for global discovery.
- [x] Restrict slow-parameter temporal evidence to accepted associations and
  reset it on runtime-ID replacement.
- [x] Add causal-only local RGB-perception clipping, preserve true raw gradient
  diagnostics, and bound causal global-perception adaptation to 512 updates.
- [x] Make zero-support pooled validation persist and reject an unsupported
  candidate instead of crashing or inventing zero horizon RMSE.
- [x] Pass compile, Ruff, formatting, diff, full sandbox tests
  (`536 passed, 6 MPS-only skipped`), and the corresponding host MPS device
  families (`36 passed`).
- [x] Complete a four-update CPU end-to-end wiring smoke with two real
  supported causal updates and no skipped/non-finite batch.
- [x] Commit and push the complete audited v3 repair set from a fully passing
  test/lint/type-quality gate.
- [x] Commit and push the lifecycle/identity/gradient repair as `c869571`.
- [x] Re-run the final committed repair tree on host MPS/CPU and verify both
  local gradient caps, source-bound ROI association, tentative births,
  checkpointing, and terminal validation at
  `runs/20260803-000212-collapse-repair-host-smoke/`.
- [x] Launch a new timestamped medium balanced qualification after that host
  smoke, then prove by artifact audit that it failed during initialization,
  took zero optimizer steps, and was repeatedly relaunched. Stop and preserve
  `runs/20260803-000858-v3-collapse-repair-qualification/`; do not count it as
  convergence evidence.
- [x] Repair per-interval stochastic-event density, shared trainer/evaluator
  causal censoring, per-scenario horizon/episode support floors, incomplete
  initialization recovery, metric-schema error handling, exact support
  persistence, and resolved-scenario protocol hashing.
- [x] Add atomic CLI failure/completion state and a one-shot macOS launch helper
  with `KeepAlive=false`; make the legacy supervisor remove failed initial
  jobs.
- [x] Make pooled/scenario/axis selector metadata exactly reproducible from
  retained additive evidence, require a durable incomplete-reference marker
  across branched resumes, and prevent the first recovered supported candidate
  from self-promotion.
- [x] Promote fast-ROI source slot/object ID from untyped auxiliary data to a
  validated measurement contract; reject stale identity after slot reuse while
  retaining unrestricted global Hungarian discovery.
- [x] Make the convergence supervisor consume authoritative terminal trainer
  state without requiring a PID and reject a live-but-reused unrelated PID.
- [x] Pass the complete quality gate (`577 passed, 6 MPS-only skipped`), the
  host MPS device families (`38 passed`), a real one-step host-MPS optimizer
  run, and a production-profile CPU causal update under simulator v4 /
  selector v6.
- [x] Commit and push the complete initialization/support/process-integrity
  repair as `97415b0`.
- [x] Launch a clean timestamped protocol-v10 balanced qualification with the
  one-shot helper, prove that it did not relaunch/deadlock/collapse, then stop
  and preserve it before any update after discovering the cadence-semantics
  bug.
- [x] Fix `global_every_steps` so cadence three is exactly
  `GLOBAL, FAST, FAST, GLOBAL`, validate it as a positive integer, and bump
  rollout validation protocol 10 to 11 without changing the measurement,
  simulator, or selector versions.
- [x] Add per-episode atomic validation progress/heartbeats, defer training
  worker startup until the first draw, and preserve full-manifest selector
  atomicity.
- [x] Reject nonfinite parameters/optimizer state immediately after Adam and
  validate model buffers, optimizer/scheduler state, and nonnegative step
  counters before checkpoint save/load. Prove corrupt overwrites leave the
  old checkpoint intact.
- [x] Complete one corrected CPU causal update and one host-MPS measurement
  update with finite post-step/checkpoint state and durable validation
  progress.
- [x] Commit and push the cadence/progress/finite-state repair as `2487b7e`.
- [x] Launch a clean timestamped protocol-v11 balanced qualification with the
  one-shot helper, prove that its progress was finite and single-process, then
  stop it at zero updates after diagnosing launchd Background QoS as the
  roughly fourfold validation-throughput regression.
- [x] Remove the Background launch classification, align float32 belief
  substep counts with the simulator's integral physics grid, reuse one typed
  causal propagation for supervision and ingestion, omit unused rollout
  auxiliary stacking, and bump rollout protocol 11 to 12.
- [x] Complete a matched Standard/default-priority protocol-v12 validation
  timing control and confirm that the foreground throughput regression is
  removed (`29.578 s` for fixed seed `100000`, versus `123.660 s` under
  Background QoS).
- [x] Complete one reduced production-model protocol-v12 causal update with
  nonzero supported gradients, finite post-step/checkpoint state, terminal
  validation, and correct rejection of its slightly regressed candidate.
- [x] Commit and push the launch-QoS/integration/prepared-propagation repair as
  `e08c4d0`.
- [x] Launch the full clean timestamped protocol-v12 convergence campaign from
  pushed commit `e08c4d0` and verify Standard/default launch QoS, one
  authoritative PID, host MPS availability, advancing per-episode progress,
  expected foreground throughput, and empty stderr.
- [x] Verify that the 32-episode broad initialization completes with supported,
  accepted score `0.3310606914` and writes its reference/best/step-zero
  checkpoints without stderr or nonfinite state.
- [x] Verify that the separate 32-episode measurement-incumbent initialization
  completes and that more than 2,000 real finite MPS optimizer updates occur
  with advancing checkpoints and empty stderr.
- [x] Attach the exact-PID convergence supervisor with the predeclared
  16,384-step minimum, 4,096-step extensions, four-validation/1% plateau rule,
  and 24,576-step hard limit.
- [x] Audit the terminated protocol-12 campaign through step 11,776: preserve
  its artifacts, prove the macOS JETSAM kill, and document that no terminal
  summary, convergence decision, or deployment promotion exists.
- [x] Prove that six causal validation intervals repeatedly reset finite
  candidates and Adam to step zero because deployment support was incorrectly
  used as mutable-state viability.
- [x] Split protocol-13 deployment support from catastrophic mutable viability,
  retain per-scenario/reference guardrails for promotion, and preserve a finite
  pooled candidate unless it falls below absolute coverage floors.
- [x] Bound sustained macOS workers/prefetch, release accelerator caches on
  phase transitions, and log process maximum RSS.
- [x] Persist explicit running state and convert supervisor-proved external
  trainer exits into the primary terminal failure contract.
- [x] Pass `603` non-device regressions, all `70` host MPS/device regressions,
  Ruff, compileall, diff checks, and a real one-update protocol-13 CPU causal
  smoke with finite checkpoint/state and process-RSS evidence.
- [x] Re-audit the live campaign at step 3584; confirm continued finite
  training/checkpoints, a step-2048 raw-score best of `4.868897`, later scores
  still far better than initialization, and no evidence justifying a
  mid-protocol reset.
- [x] Re-audit executable training/selection/continuation paths during the
  step-6144 validation; preserve the exact runtime fingerprint, confirm
  production ROI gradients, pass `599` non-device tests, and verify that the
  validation heartbeat continues under CPU contention.
- [x] Complete a conservative cleanup during live training: remove/quarantine
  regenerable bytecode, test/lint caches, editable-install metadata, and the
  empty step-257 demo while retaining the active run, initialization input,
  accepted/rejected scientific evidence, and all specification-required
  modules.
- [x] After the campaign terminates, perform a
  tracked-code simplification review against `PROJECT_SPEC.md`; do not remove
  dormant modality-independent contracts merely because the first RGB slice
  does not yet call them. The review found no safe tracked-module deletion.
- [x] After the campaign terminates, fail closed
  in `supervised_slot_measurement_losses` by requiring `matched_slots` for
  positive crop evidence even if a stale nonnegative target index is supplied;
  add the explicit stale-index regression.
- [x] After the campaign terminates, make
  `training_state.json` distinguish an active/running invocation from initial
  startup, while preserving atomic terminal failure/completion semantics and
  `training_progress.json` as the detailed heartbeat.
- [x] Launch
  `runs/20260806-213753-v7-protocol13-causal-convergence/` from clean pushed
  commit `1470b2e` and the best finite protocol-12 raw causal candidate as
  weights-only initialization; verify one Standard/default process, explicit
  running state, empty stderr, host MPS visibility, CPU causal policy, and
  advancing full initialization validation.
- [x] At consecutive 512-update protocol-13 validations, verify that optimizer
  history accumulates across scenario-only deployment rejections and process
  RSS remains bounded.
- [x] Audit the coupled protocol-13 campaign through 6,096 causal updates,
  preserve its eleven rejected post-initialization validations, and stop it
  after proving a repeated forecast-accuracy versus tracking-support tradeoff.
- [x] Add exact modular checkpoint qualification and prove that preserving the
  accepted global discovery stack while importing causal fast-ROI/state
  modules yields the strongest every-horizon diagnostic candidate without
  falsely promoting its remaining identity/z/coverage regressions.
- [x] Add `state_dynamics_fast_roi` so causal ROI learning cannot update shared
  backbone stages after global perception is frozen.
- [x] Commit/push `ea67f8d` and launch the clean 8,192-update frozen-backbone
  campaign from the accepted reference with zero global-adaptation steps.
- [x] Verify the complete 32-episode initialization retains the exact accepted
  score/hash linkage, then observe a finite supported step-8 update with zero
  global-perception trainability and no skipped draws or stderr.
- [x] Inspect the new frozen-backbone qualification at its declared validation steps
  for support, perception/interaction/global gradient balance, lifecycle
  precision/coverage, identity switches, collisions, calibration, every
  horizon, and every scenario slice. Stop on a repeated structural collapse.
  The audit reached step 4,744 and found a frozen-global-loss objective bug.
- [x] Make causal global-loss inclusion depend on a real trainable detector,
  shared-stage, or pyramid path rather than the ROI-only fast projection; keep
  frozen global loss diagnostic-only and cover the final objective in a real
  closed-loop regression.
- [x] Launch a fresh specification-1.14 frozen-fast-ROI campaign weights-only
  from the same accepted reference with clean source and unchanged protocol.
- [x] Inspect its first corrected optimizer metrics. Confirm they omit
  `measurement_global`, retain `frozen_global_measurement`, and show the
  raw `measurement_fast` diagnostic before interpreting accuracy trends. The
  step-512 regression exposed support-dependent branch reweighting.
- [x] Preserve fixed global/fast coefficients when one branch is absent or
  frozen; add direct branch-support regressions.
- [x] Qualify fast-ROI-only tensors from both half- and full-weight step-512
  candidates on the exact 32-episode manifest and record the opposing results.
- [x] Add an explicit fast-ROI-only scope plus a paired, exact causal-update
  transition to a late state/dynamics scope, with per-update phase metrics.
- [x] Launch the clean specification-1.15 campaign for 512 fast-ROI-only
  updates followed by state/dynamics from the accepted reference.
- [x] Inspect the step-512 localization checkpoint and prove exact scope/tensor
  isolation. Stop protocol 15 after four later candidates meet the failed-
  plateau rule without velocity/coverage or long-horizon repair.
- [x] Qualify step-1,536 dynamics-only and updater/identifier-only donors on
  the exact 32-episode manifest; preserve both rejected reports.
- [x] Prevent frozen fast-ROI measurement auxiliaries from training dynamics
  or the belief updater through their prior-conditioned input; keep detached
  diagnostics and add real closed-loop objective/support regressions.
- [x] Synchronize checkpoint `specification_version` metadata with the
  authoritative 1.16 contract and test that the two cannot drift silently.
- [x] Commit/push specification 1.16 as `310d419` and launch a clean
  weights-only replacement with verified Standard QoS, host MPS, clean source,
  one trainer, advancing initialization heartbeat, and empty stderr.
- [x] Inspect protocol 16 at step 512 and its first late-phase updates. Preserve
  its rejected 122-guardrail candidate, then stop at update 552 after objective
  tracing exposes duplicated rollout-likelihood mean gradients.
- [x] Verify the first late-phase block omits optimized fast measurement and
  has zero perception gradient; make `causal_fast_support_count` exclude
  observed-but-frozen ROI slots instead of overstating causal support.
- [x] Attach the exact-launch-source convergence supervisor with the 8,192
  minimum, 4,096 extensions, four-validation/1% plateau rule, and 24,576 hard
  limit; verify one supervisor PID, matching runtime fingerprint, durable wait
  event, and empty stderr.
- [x] Make rollout Gaussian likelihood variance-only by detaching forecast-mean
  error, including after hidden external actuation; prove absent mean gradient
  and finite variance-widening gradient in a direct regression.
- [x] Commit/push specification 1.17 and launch protocol 17 weights-only from
  the accepted reference with verified clean source, MPS availability,
  Standard QoS, advancing heartbeat, and empty trainer/supervisor stderr.
- [ ] Monitor protocol 17's optimizer/support/identity/uncertainty, every-axis,
  every-horizon, and every-scenario dynamics through the declared minimum and
  plateau/extension rule without promoting any broad regression.
- [x] Add a deterministic live dynamics audit that canonicalizes exact-resume
  replay rows, verifies their model/data equivalence, detects finite/update/
  support/scope/data-progress failures, and reports every-axis/horizon fixed
  validation plus live lifecycle/identity/event/uncertainty/observability
  evidence, including pooled clean-versus-recovery-perturbed identity rates,
  without treating training loss as convergence.
- [ ] After protocol 17 reaches a terminal convergence decision, make resumed
  metrics attempts explicit so an uncheckpointed pre-stop tail and its exact
  replay cannot be naïvely double-counted. Preserve append-only evidence and
  keep convergence based solely on verified validation checkpoints.
- [x] Audit protocol 17 through the complete step-512 fixed validation and
  first late-phase block. Preserve its rejected 113-guardrail checkpoint;
  verify state/dynamics routing, support, gradients, finite state, and memory.
- [x] Audit protocol 17 through the complete step-1,024 validation. Preserve
  its rejected 134-guardrail checkpoint; record the z/collision/coverage gains,
  x/y and medium-to-long joint regressions, zero support failures, finite
  optimizer behavior, frozen perception, and flat memory.
- [x] Audit protocol 17 through the complete step-1,536 validation. Preserve
  its rejected 122-guardrail checkpoint; record the broad recovery from step
  1,024, remaining medium/long and scenario regressions, finite optimizer and
  support state, and measured event-versus-trajectory shared-trunk gradient
  conflicts without prematurely changing the protocol.
- [x] Resume protocol 17 exactly from step 1,536 after the audit; verify sample
  draw 1,544, optimizer/support continuity, frozen perception, one trainer,
  one exact-source supervisor, Standard QoS, and empty stderr.
- [x] Repair the one-shot launch helper so exact in-place `--resume` omits
  `--run-name`; quarantine the accidental nested attempt, preserve the failed
  launch evidence, and relaunch the unchanged step-1,536 checkpoint under one
  monitored trainer and supervisor.
- [ ] If identity churn persists after the structural repair, add an explicit
  supervised pairwise association-margin objective and test duplicate
  suppression against missed live tracks; do not tune appearance weight alone.
- [ ] Decide whether confirmed candidates that exceed free capacity should
  retain bounded confirmation state for later allocation; current behavior
  requires reconfirmation after a slot opens.
- [ ] Launch the clean-source 16,384-update v3 campaign only after the medium
  qualification passes. Monitor at least four comparable corrected-protocol
  validations and do not call a hard-budget stop convergence.
- [ ] Compare the selected v3 checkpoint against its exact imported reference
  on at least 64 fresh balanced RGB-only episodes, then generate timestamp-first
  reports and demos only for a broadly accepted checkpoint.

- [x] Audit all comparable training/evaluation artifacts and document which
  improvements generalize, which are context-specific, and which intervention
  candidates regress the recursive online loop.
- [x] Replace position-loss-only rollout selection with pooled physical
  multihorizon selection and fixed-reference non-regression guardrails for
  velocity, detection recall/precision, lifecycle, events, identity, and
  calibration.
- [x] Bind resumed incumbent/reference metrics to exact validation-protocol,
  seed-manifest, and model-tensor hashes; retain every numbered validation
  checkpoint.
- [x] Bound training rollout anchors without skipping online frame ingestion or
  state supervision, reuse posterior rollouts, and remove redundant prior
  future rollouts from validation.
- [x] Add `configs/sustained_accuracy_mps.yaml`: 8,192 measurement updates plus
  4,096 causal windows across all eight scenario families, with rejected
  intervention heads disabled.
- [x] Implement a provenance-verifying, restart-aware convergence supervisor
  that waits for the full minimum, resumes only `last.pt` in non-overlapping
  4,096-update blocks, records failures, and distinguishes a demonstrated
  plateau from a 24,576-step budget limit.
- [x] Preserve and supersede the legacy 12,288-update campaign at
  `runs/20260730-192625-scaled-sustained-e2e-v1/` after proving its perception
  handoff restored step-zero weights and its deterministic long-horizon
  objective contained non-identifiable targets. It stopped at logged step
  `9400`; do not describe it as converged.
- [x] Launch the persistent convergence supervisor. Continue in complete 4,096
  blocks while the final 1,024 updates produce at least 1% safe improvement or
  while four-point plateau evidence is incomplete/contradictory; stop only on
  the declared plateau or the 24,576-step hard limit.
- [x] Audit the apparently unstable causal loss, checkpoint/optimizer state,
  phase handoff, clipping, and first broad validation. Confirm finite state and
  batch-one hard-window variance rather than numerical divergence.
- [x] Correct axis-specific rollout losses to use the fixed global configured
  horizon denominator, add joint collision/maximum-horizon sampling, and make
  pre-clip versus applied gradient norms explicit.
- [x] Freeze the checkpoint-compatible ROI event and identifier variance heads
  in restricted training scopes while they have no end-to-end objective.
- [x] Separate the guardrail-safe deployment incumbent from the mutable phase-
  handoff candidate so downstream causal losses retain and repair useful RGB
  perception gains.
- [x] Add mature/cold forecast support, scene-wide censoring after hidden
  external actuation, forecast NLL, velocity correction guards, per-axis and
  per-scenario selection evidence, and exact structured rejection reasons.
- [x] Make exact resume preserve the absolute next sample, CPU/MPS RNG,
  immutable process-start provenance, objective/data semantics, and exact
  additive counts.
- [x] Fix simulator/render confounds: independent RNG streams, stable resting
  contact, true glancing impacts, compositional OOD ranges, and consistent
  pair restitution in simulator and analytic dynamics.
- [x] Add `configs/sustained_accuracy_mps_v2.yaml` with 40-frame mature
  one-second support, batch two, 16,384 unique training episodes, corrected
  horizon semantics, and bounded deterministic trend-validation anchors.
- [x] Complete the bounded-anchor MPS handoff smoke, inspect every produced
  checkpoint/metric, and record its exact wall time. It is wiring evidence,
  not a promotion comparison. The final three-update run is
  `runs/20260801-231521-audit-v2-final-verified-smoke/`; it completed in
  `109.1528 s`, wrote a completed terminal-validation marker, and passed a
  byte-preserving exact no-op resume.
- [x] Reproduce and fix the data-dependent PyTorch 2.10 MPS detector-gradient
  failure. Keep CNN/ROI work on MPS, pin only the proposal transformer to CPU,
  and prove finite mixed-device gradients, clipping, two optimizer steps, and
  checkpoint restore.
- [x] Detach RGB covariance linearization coordinates from mean heads; keep
  variance-head calibration differentiable.
- [x] Recover interrupted final validation without optimizer updates, restrict
  in-place resume to `checkpoints/last.pt`, and deserialize hybrid checkpoints
  on CPU to preserve optimizer ownership and accelerator memory.
- [x] Launch the timestamped v2 MPS campaign from the preserved step-8192
  `best_measurement.pt` using clean source `df98f63`, then stop and supersede
  it at logged step `9576` after the 2 August support/coverage audit. Its
  trainer and supervisor jobs are no longer active; artifacts remain at
  `runs/20260801-232229-scaled-sustained-v2/`.
- [x] Cancel the v2 completion/comparison tasks because repairing support,
  selection, and optimization changes the protocol. Never resume v2 in place
  or describe its finite loss as convergence evidence.
- [ ] Confirm the selected checkpoint on at least 64 fresh balanced validation
  episodes, report every scenario/horizon, then use the reserved test split
  only if all broad gates pass.

## Milestone 1 vertical slice — implemented

- [x] Read `PROJECT_SPEC.md` in full and inspect the `orpheus` environment.
- [x] Add repository policy, packaging, strict configs, and project memory.
- [x] Validate editable install, CLI dry runs, and root entry points.
- [x] Implement deterministic labelled RGB sphere simulator and datasets.
- [x] Implement canonical belief dataclasses, packing, invariants, and lifecycle.
- [x] Implement analytic/modal/interaction/event dynamics and uncertainty.
- [x] Validate the oracle debug filter and perturbation recovery.
- [x] Implement RGB global discovery, calibrated projection/back-projection, and
  propagated measurement uncertainty.
- [x] Implement residual ROI measurements, association, surprise scheduling,
  identity-aware caches, and direct fast-path supervision.
- [x] Run the persistent RGB-only
  predict–observe–associate–innovate–correct loop.
- [x] Implement depth-ordered projected occlusion, occlusion-aware misses,
  identity retention, and miss uncertainty growth.
- [x] Implement and unit-test observability-gated drag/restitution updates
  without optimizer updates at inference time.
- [x] Train a deterministic tiny run and save measurement, rollout, and last
  checkpoints with truthful selection provenance.
- [x] Evaluate held-out RGB episodes against static, constant-velocity, default
  analytic, and labelled oracle-parameter analytic baselines.
- [x] Export a real prior/posterior RGB-only GIF, frames, parameter plot, and
  machine-readable summary.
- [x] Exercise global and differentiable fast ROI training on MPS.
- [x] Run Ruff, Pytest, CLI smoke, MPS-specific, checkpoint-compatibility, and
  checkpoint-round-trip checks.
- [x] Synchronise README, status, design decisions, and changelog with evidence.
- [x] Decouple deterministic measurement frames from loader-batch parity so
  every fixed episode receives every configured frame.
- [x] Select perception checkpoints by multi-frame calibrated world-position
  MAE and restore the best localized state before closed-loop handoff.
- [x] Apply a separately configured 10x closed-loop learning-rate reduction,
  including after optimizer-state resume.
- [x] Gate the demonstrably harmful fast inverse-depth residual while retaining
  fast centre/appearance updates and the ordinary ROI runtime path.
- [x] Reach positive ordinary current/future demo correction and nonzero
  distance-gated held-out RGB localization.
- [x] Freeze global RGB discovery after a configurable closed-loop adaptation
  window; verify continued ROI/filter/dynamics training without localization
  drift.
- [x] Supervise fast ROI measurements on every usable frame in persistent
  belief-slot order rather than rematching conditioned outputs.
- [x] Aggregate collision occurrence across internal physics substeps and
  align training/evaluation logits to exact preceding observation windows.
- [x] Add bounded rare-positive collision weighting and explicit
  prior-to-posterior current/future improvement guards.
- [x] Report sequence-aware occlusion identity/uncertainty transitions and
  directional before/after drag/restitution update metrics with explicit
  zero-sample/null behavior.
- [x] Add explicit fresh-validation manifests that are disjoint from trainer
  validation and the reserved test range, with exact seed provenance.
- [x] Report current velocity error, ordinary prior-to-posterior velocity
  correction, temporal-evidence availability/variance, and
  collision-conditioned model/baseline forecasts.
- [x] Implement a bounded causal RGB position history keyed by persistent ID
  and a cheap explicit velocity-only correction path. Keep it disabled by
  default because the first validation-selected ablations regress the primary
  localization/forecast metrics.
- [x] Exclude invalid graph edges and diagonals when pooling learned event
  residuals, so valid negative collision evidence is not clamped to zero.
- [x] Add RGB-only structured disc localization with touching-silhouette peak
  splitting for global discovery and a projected-ROI refinement for ordinary
  updates.
- [x] Keep raw learned RGB centres explicitly supervised when a structured
  forward measurement is active.
- [x] Add calibrated metric-space RGB position supervision and explicit
  measurement-term weights.
- [x] Validate every configured validation episode through a complete causal
  online unroll.
- [x] Add causal prefix burn-in and collision-conditioned sampling for
  mid-episode TBPTT windows.
- [x] Separate state/rollout position and velocity objectives and select
  rollout checkpoints by physical position loss.
- [x] Add explicit evaluation seed offsets for paired selection and
  confirmation manifests.
- [x] Harden structured RGB thresholds for noisy `toy_hard` and cloud
  profiles with a deterministic noise regression.
- [x] Treat structured and temporal RGB measurement controls as checkpoint
  semantics while normalizing absent legacy fields to their old defaults.
- [x] Continue the selected perception state through 64 causal closed-loop RGB
  updates and promote step 648 only after paired ROI-local
  selection/confirmation forecast evidence.
- [x] Probe collision thresholds exhaustively and retain `0.5` after saturated
  logits showed the residual failure is structural rather than a threshold
  choice.
- [x] Reject mean-radius and photometric analytic depth replacements after
  metric-space and confirmation checks failed.
- [x] Aggregate rollout/correction losses globally per horizon, keep the
  configured horizon denominator fixed, sample maximum-horizon-capable
  windows, and version checkpoint selection semantics.
- [x] Add a 32-frame deterministic multistep profile covering recursive
  0.10/0.25/0.50/0.75/1.00-second forecasts.
- [x] Stabilise demo axes, margins, and legends; retain historical posterior
  forecasts with fading alpha; show matched endpoint error; and reserve
  scoring-only lookahead so the displayed horizon remains fixed.
- [x] Prefix new train/evaluation/demo artifact folders with sortable UTC
  timestamps and recoverably archive superseded demos.
- [x] Project posterior world covariance through the camera Jacobian before
  drawing image-space uncertainty.
- [x] Gate analytic pair/plane contact jumps by position uncertainty and
  validate the `0.25σ` setting on the 16-episode multistep block.
- [x] Add camera-parallax, glancing-impact, and unequal-mass interaction
  regimes; train a 140-episode seven-regime continuation and reject it after
  the original-task confirmation gate regressed.
- [x] Consolidate the accepted checkpoint/evidence and remove 64 superseded
  run directories at the user's request.

## Predictive-abstraction modernization — in progress

- [x] Amend `PROJECT_SPEC.md` and agent instructions so compact executable
  predictive abstractions, rather than sensor reconstruction, are the scaling
  unit.
- [x] Add an explicit registry for implemented abstraction families.
- [x] Route ordinary free motion to a point-trajectory abstraction and refine
  contact-like modes to rigid-sphere execution without discarding belief
  geometry or parameters.
- [x] Add reversible scene, camera, entity-kinematic, dynamical-programme, and
  lifecycle belief tokens with persistent IDs, masks, and abstraction kinds.
- [x] Expose derived abstraction assignments and tokens from
  `OnlineWorldModel` without adding checkpoint parameters or a second state.
- [ ] Add learned token projections and a small object-interaction transformer
  that predicts bounded belief residual/event proposals.
- [ ] Train the transformer first as an auxiliary masked/future latent
  predictor; do not let it alter runtime state until it beats the structured
  baseline on held-out position, event, calibration, and complexity gates.
- [ ] Replace the mode-only router with evidence-driven model selection using
  predictive likelihood, calibration, correction cost, and abstraction
  complexity.
- [ ] Let abstraction assignments prune execution only after a
  proximity/uncertainty gate proves that free point trajectories refine before
  imminent contacts; the current hybrid dynamics remains authoritative.
- [ ] Add multi-hypothesis depth/size abstractions so monocular ambiguity is
  represented rather than collapsed.
- [ ] Connect a pretrained/foundation video feature provider behind the RGB
  measurement contract, retaining a local/offline and checkpoint-compatible
  path.

## Milestone 1 research acceptance — not yet achieved

- [x] Reach nonzero distance-gated RGB detection recall/precision on held-out
  episodes (100% recall and precision over the final 32-episode reserved-test
  block with ROI-local ordinary updates).
- [x] Sustain the recommended 20% injected-perturbation recovery improvement
  on a wider protocol (45.30% over the promoted final 32-episode reserved-test
  block).
- [ ] Learn useful collision prediction (the promoted checkpoint reaches
  0.6400 F1 on the final reserved-test block, materially above the old 0.0426
  result but still below the 0.75 target).
- [ ] Demonstrate calibrated uncertainty expansion/recovery through held-out
  rendered occlusions.
- [ ] Demonstrate distance-gated drag and restitution convergence from RGB,
  beyond merely executing the observability/update gates.
- [ ] Make the ROI path measurably cheaper than the global path at the target
  scale. Full-frame discovery has been removed from ordinary updates and the
  local centroid operation costs about 3.94 ms for eight 20x20 ROIs, but total
  tiny-profile fast/global latency remains 50.54 / 48.74 ms.
- [ ] Train and validate fast inverse-depth residuals on the now
  belief-slot-aligned cached sequences; enable them only after per-mode
  current/future improvement.
- [x] Beat the promoted step-648 checkpoint on the 16-episode
  0.50/0.75/1.00-second RGB-only fresh-validation block with the bounded
  lateral young-track initializer. A disjoint confirmation block is still
  required before a reserved-test promotion claim.
- [x] Add temporal RGB velocity evidence so post-association motion can be
  assimilated without re-encoding history.
- [x] Add and evaluate a tiny bounded outgoing gravity-velocity proposal on
  exact-timestamp RGB event windows. Its offline event-window fit improved,
  but paired online velocity regressed, so it remains disabled.
- [x] Validate and enable a camera-lateral young-track velocity initializer in
  `tiny_lateral_velocity.yaml`; it improves current state and every 0.1–1.0 s
  forecast horizon on the 16-episode fresh-validation gate.
- [x] Correct the demo ground-truth XY display so each identity has one
  colour-coded past/current-horizon trajectory with explicit time direction.
- [ ] Resolve the remaining monocular scale/height ambiguity; uncertainty-aware
  contact improves timing, collision F1, and multistep position, but velocity
  remains worse and the right-ball forecast is still too vertical.
- [x] Add deterministic baseline, elastic-pair, damped-contact, and
  impulse-perturbation scenario mixtures without changing episode/runtime
  tensor contracts.
- [x] Run paired per-scenario RGB-only evaluations before and after mixed
  closed-loop and RGB-adapted continuations; retain both as rejected evidence.
- [ ] Train multi-object discovery/association on a balanced three-object
  curriculum with per-query recall validation before another dynamics
  continuation. The short mixed RGB adaptation reduced detection recall.
- [ ] Run the full 3,000-step `configs/toy_mps.yaml` schedule and a materially
  larger held-out test split.
- [x] Export sequence-aware occlusion survival/uncertainty metrics with
  explicit zero-sample behavior.
- [x] Export collision-conditioned model/baseline forecast metrics.
- [ ] Export physics-violation diagnostics and representative failure plots.

## Deferred architecture

- [x] Multi-frame tentative birth confirmation with bounded sensor-local
  evidence, strictly increasing timestamps, distance-gated cardinality-first
  assignment, and permanent IDs only after confirmation.
- [ ] Multiple-hypothesis branch/prune/merge.
- [ ] Estimated camera pose and fixed-lag smoothing.
- [ ] Continuous collision timing and richer geometry.
- [ ] Spectral fixed-window ablation.
- [ ] A second modality and real calibrated video adapter.
- [ ] Correlation-aware temporal measurement covariance or learned confidence
  gating that improves velocity without degrading localization.
- [ ] Evaluate a two-frame anisotropic position-slope velocity measurement;
  it is a future opportunity, not an implemented/promoted path.
- [ ] Implement an observability-driven gravity-aligned motion gate only if it
  propagates position, velocity, covariance, and events coherently and passes
  wider/OOD validation; do not ship the promising fixed-axis diagnostic blend.
- [ ] Replace the synthetic disc prior with a learned or externally structured
  real-video adapter without changing the measurement/belief contracts.
- [x] Add axis-resolved current/forecast position and velocity metrics plus
  independently weighted rollout-position losses, while keeping event
  detection and interaction dynamics joint.
- [x] Add a familiar `reference_pairs` regime whose ensured sphere collision
  is temporally separated from its first floor impact and regression-test that
  separation.
- [ ] Add an optional mature physics-engine dataset backend behind the
  canonical episode and RGB `ObservationPacket` contracts. Keep it out of the
  smoke dependency set and report it as a separate dataset family.
- [ ] Improve RGB-derived x velocity and 0.5–1.0 s prediction on the clean
  reference regime. Direct measured-point history and denser global cadence
  improved some current-state metrics but failed the multistep selection gate.
- [x] Add one balanced eight-scenario profile with a single shared checkpoint,
  deterministic scenario ordering, fixed-radius RGB geometry, and per-scenario
  evaluation reports.
- [x] Make resumed best-checkpoint compatibility depend on the validation
  episode count, scenario mixture, sequence length, object-count range, seed,
  horizons, and metric version.
- [x] Make collision-triggered RGB temporal-history resets edge-triggered so a
  sustained collision mode can accumulate outgoing velocity evidence.
- [ ] Improve three-object global discovery in `damped_contacts` and
  `reference_pairs`; detection recall remains the main current-state bottleneck
  in the weakest held-out examples.
- [ ] Learn event-conditioned outgoing lateral velocity that improves the
  paired 0.5–1.0 second gate, especially for `reference_pairs`, without
  regressing the seven easier regimes.
- [x] Train an outgoing lateral correction against its actual
  post-filter intervention and recursive forecast effect, with an explicit
  learned abstention/gain objective; do not reuse the gravity-only gate as a
  proxy for camera-lateral collision evidence. Reject it after the paired
  0.5–1.0-second forecast regressed despite offline and short-horizon gains.
- [x] Implement and evaluate an acceleration-
  aware axis-local proposal, joint collision context, calibrated uncertainty,
  gravity-only covariance, and one on-policy dataset-aggregation pass. Reject
  promotion after the paired block regressed velocity, detection, and selected
  longer horizons despite strong current/short-horizon y gains.
- [ ] Train observation interventions through the persistent
  association/ROI feedback loop and recursive 0.1–1.0-second losses. Include
  detection coverage and identity stability in selection so a locally useful
  state correction cannot degrade future observability.
- [ ] Raise shared-model collision F1 and reduce identity switches on a larger
  balanced test manifest; the current 16-episode result is only
  `0.320388` F1 with three switches.
- [x] Audit structured RGB point/scale accuracy and separate centre error from
  monocular depth error. Centre localization is subpixel; heavy-tailed
  radius-derived depth under overlap is dominant.
- [x] Add disabled-by-default, tested gates for associated depth-disagreement
  covariance inflation and combined temporal/position-innovation velocity
  evidence. Reject both policies after final-test multistep regressions.
- [x] Add a persistent-ID multi-frame point/scale trajectory measurement with
  independent bounded point/scale rings, axis-local robust estimates,
  uncertainty, boundary/overlap quality gates, and camera-depth-only direct
  belief correction. It is parameter-free rather than learned; the scaled
  policy improves all 0.1–1.0-second position horizons on two disjoint paired
  blocks while exposing an event/velocity tradeoff.
- [ ] Supervise scale quality with visible fraction, boundary truncation,
  component overlap, temporal scale consistency, and prediction disagreement;
  validate calibration by quality bucket before enabling correction.
- [ ] Train event-conditioned outgoing velocity on balanced pre-contact and
  post-contact windows, with constant/damped motion represented as a learned
  low-complexity prior rather than a hardcoded runtime rule.
- [ ] Recover collision F1 and outgoing-velocity accuracy after enabling the
  confirmed point/scale observer; position, detection, and calibration improve
  strongly, but velocity and event F1 regress on the paired blocks.
- [x] Add a tested opt-in acceleration-aware temporal velocity fit that
  estimates velocity at the current timestamp and exposes only camera-lateral
  plus post-event gravity-axis evidence. Reject continuous/all-axis and
  endpoint-contact variants after paired MPS selection tradeoffs.
- [x] Add a causal, acceleration-compensated RGB trajectory change-point
  capability with observable-axis projection, independent point/scale reset
  semantics, reset provenance, diagnostics, and a two-sample post-event fit.
  Reject the permissive and endpoint-contact policies after paired MPS checks.
- [x] Train linear and eight-hidden-unit nonlinear uncertainty-aware
  change-point gates on balanced, exact-timestamp RGB history windows; cache
  features, preserve checkpoint provenance, and reject both learned policies
  after held-out classification and paired MPS runtime checks.
- [x] Learn a calibrated outgoing-velocity proposal jointly with the
  observation-side event gate. The sparse learned gate is safe, but the
  fitted proposal and its trigger-aligned correction were rejected because
  velocity still regressed when it acted.
- [x] Add a 1.90M-parameter scaled shared-model profile with 4,096 training,
  256 validation, and 256 test episodes across all eight scenario families.
- [x] Record model parameter count, episode draws, nominal manifest passes,
  split sizes, and scenario families in training plans/summaries.
- [x] Fix cross-device RNG restoration so a CPU checkpoint can resume on MPS.
- [x] Complete 1,024 mixed-scenario measurement draws and one full causal
  update with the scaled model on MPS; retain it as an unvalidated scale
  artifact, not an accuracy promotion.
- [x] Continue the shared scaled model to step 896 on MPS and run a disjoint
  paired two-episode check. Retain the checkpoint as mixed evidence: current
  and 0.1–0.75 s position improve, while velocity and 1.0 s prediction regress.
- [x] Persist final weights before expensive validation, reject non-finite
  validation aggregates, ignore non-finite structured proposal rows, and make
  parameter-update reporting safe for MPS float64 accumulation.
- [ ] Complete the 48,000-example `configs/scaled_curriculum.yaml` schedule on
  MPS or CUDA, then evaluate validation, test, and OOD splits using fixed
  manifests and per-scenario slices.
- [x] Make the primary scaled monocular accuracy curriculum physically
  identifiable with fixed known radius; keep variable radius as a separately
  labelled transfer/OOD identification task.
- [x] Add strict weights-only curriculum initialization with reset
  optimizer/RNG/step provenance.
- [x] Diagnose six-frame ROI/tracker drift and provisionally select a
  three-frame global anchor cadence on two disjoint paired validation blocks.
- [x] Prevent collision-conditioned TBPTT windows from producing zero-horizon
  causal updates.
- [x] Run a bounded sampler-corrected causal continuation through step 16 and
  reject its weights after 0.25–1.00-second paired regressions.
- [ ] Confirm cadence three and sampler-corrected causal training on at least
  16 fresh-validation episodes before test promotion.
- [x] Profile and reduce closed-loop validation/rollout cost before another
  large run; bound training anchors, reuse posterior rollouts, and skip the
  validation-only prior future rollout while retaining all posterior anchors.
- [ ] Add gradient checkpointing or optimizer accumulation if eight-step,
  batch-one causal updates remain the throughput bottleneck.
- [ ] Connect ROI `measured_event_features` to the fast corrector through a
  zero-initialized checkpoint-migrated adapter, then require nonzero finite
  event-head gradients and broad recursive improvement before enabling it.
- [ ] Add an observable latent-space calibration objective for slow
  restitution/drag uncertainty before unfreezing the identifier variance head.
