# Changelog

## Unreleased — 2026-07-28

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
