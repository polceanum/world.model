# Tasks

## Paper-guided next target — short-step hypothesis selection

- [x] Add a functional, non-mutating multi-hypothesis rollout wrapper and a
  structural `predict_step` adapter that reuse `WorldBelief` and the existing
  `RolloutStep` contract.
- [x] Add masked, uncertainty-aware per-batch trajectory scoring and
  deterministic selection with posterior weights; cover empty inputs, masks,
  occluded frames, and uncertainty behavior with focused tests.
- [x] Add a persistent candidate pool that carries evidence weights across
  cycles, supports late observation assimilation, and preserves the source
  belief; verify it with fixed synthetic candidates and real dynamics adapters.
- [x] Expose candidate prediction and delayed evidence assimilation through
  `OnlineWorldModel` without storing hypotheses as alternate world truth.
- [x] Ensure reported selection follows accumulated posterior weights rather
  than instantaneous error alone; add a prior/evidence regression test.
- [ ] Connect the selector to a small analytic/learned candidate pool and run
  the complete incumbent comparison protocol before changing model weights.
- [ ] Only if the selector passes broad guardrails, train or adapt candidate
  models and audit support, optimizer, lifecycle, identity, uncertainty,
  events, every axis, and every horizon.

## Active convergence target — typed attention scaling from corrected control

- [x] Implement a backward-compatible, exact-resumable constant or linear-
  warmup/cosine closed-loop learning-rate protocol. Derive it from absolute
  causal update index and an explicit decay duration so convergence extensions
  cannot reshape the schedule; prove legacy constant compatibility, semantic
  resume rejection, and a real `0.0002 -> 0.00011` checkpoint-resume smoke.
- [x] Evaluate the constant-rate drift candidate at fixed selector 512. Reject
  score `0.3332533` versus protected `0.3213162` with 105 broad guardrail
  failures, dominated by `reference_pairs` x at every horizon; strictly audit
  its intact finite scope and stop both one-shot jobs at the durable boundary.
- [x] Launch a separately versioned same-capacity warmup/cosine control from
  the protected graph checkpoint, never the rejected learned attention
  weights. Use 384 absolute warmup updates, 8,192 fixed cosine-decay updates,
  a 0.1 floor, unchanged objectives/data/selectors, and prove exact step-zero
  model/metric reproduction under clean one-shot trainer/supervisor jobs.
- [x] Qualify the warmup/cosine control through its first trained fixed
  selector. Reject score `0.3475480` versus protected `0.3213162`, with 116
  broad guardrail failures plus failed improvement and zero support failures.
  Every pooled position horizon regresses and `reference_pairs` current x is
  `0.720231 m`; strict 48-tensor/Adam/inherited/protected/finiteness audit
  passes, proving behavioral generalization failure rather than corruption.
- [x] Verify the rejected schedule trainer and supervisor are absent after the
  host pause. Their logs stop at step 592 and no checkpoint newer than the
  rejected durable step-512 selector exists; do not resume either trajectory.
- [x] Implement specification-1.42 `attention_relation` training scope and a
  generic frozen-attention-prefix checkpoint audit. Freeze exactly the two
  node-decoder tensors while training the other 46 attention tensors; prove
  configuration, requires-grad partition, tensor equality, and exact optimizer
  ownership (`736 passed, 6 skipped`; Ruff/format clean).
- [x] Run exact fixed-manifest zero-node modular ablations of both drift-
  regularized step-512 checkpoints. The cosine and constant donors score
  `0.342289`/`0.329317` versus protected `0.321316`, with 100/98 guardrail
  failures and zero support failures; neither donor is promotable. Combined
  with the earlier no-drift zero-node score `0.297330`, this localizes the
  remaining experiment to fresh relation-only gradients rather than reuse of
  any rejected learned state.
- [x] Launch the full constant-rate `attention_relation` campaign
  weights-only from the untouched graph control. The exact two-update CPU
  smoke passes 46-trainable/2-frozen tensor and optimizer ownership, complete
  finiteness, causal support, and resume. The active clean-commit trainer uses
  MPS measurement/CPU closed loop, and its corrected supervisor runs from a
  detached Git worktree whose provenance exactly matches the trainer.
- [ ] Monitor the relation-first campaign through its 8,192-update minimum,
  65,536 balanced examples, fixed selectors, and declared plateau. Audit all
  46 permitted tensors, both frozen node tensors, optimizer state, support,
  lifecycle, identity, uncertainty, events, every axis/horizon, test, and OOD
  evidence; keep every scale gate closed until broad convergence is proved.
  Step-zero is bitwise exact across 225 tensors and 2,584 metrics. The first
  64 balanced updates pass with exact node isolation, no skips/failures,
  stable memory, and mixed near-control axis/horizon movement. The complete
  sampled 8--64 window remains finite with at least `0.812481` complete
  interaction-gradient retention; continue unchanged to the step-128
  structural audit and step-512 fixed selector.
- [x] Strictly audit and preserve the relation-only step-128 checkpoint. Exact
  46-trainable/2-frozen attention scope, 177 inherited tensors, 46 Adam owners,
  protected checkpoints, source/protocol hashes, and finiteness all pass. The
  matched 72--128 window improves current/short x and collision behavior but
  regresses pooled 0.25--1.00-second position; continue to selector 512.
- [x] Audit complete relation-only updates 136--192. Operational integrity,
  balance, support, clipping containment, and memory pass, but exact matched
  current/every-horizon position regress mainly on x/z, with mostly adverse
  velocity/lifecycle/identity/event slices. Preserve the warning and continue
  unchanged to the predeclared fixed selector rather than promote or retune on
  heterogeneous training samples.
- [x] Preserve and strictly audit relation-only checkpoint 256 and complete
  updates 200--256. Scope/Adam/protected/finiteness evidence passes. Current,
  velocity, lifecycle, event, and short-horizon position recover versus the
  matched control, while x at every horizon, pooled 0.50--1.00 seconds,
  identity switching, and median NLL remain adverse. Continue to selector 512.
- [x] Audit complete updates 264--320. All operational gates pass. The earlier
  x/long-horizon regression largely repairs and pooled 1.00-second position
  improves, but z now regresses increasingly with horizon while current and
  0.10--0.75-second pooled position remain near-tie adverse. Preserve the
  migrating-axis evidence and keep selector 512 authoritative.
- [x] Preserve/audit checkpoint 384 and complete updates 328--384. Structural
  integrity passes exactly. Current and every pooled position horizon regress
  mainly on x, while velocity/event/lifecycle/identity improve and uncertainty
  is mixed. The earlier z trend does not persist, confirming axis migration
  without proving generalization; continue unchanged to selector 512.
- [x] Audit complete updates 392--448. Operational integrity is exact with no
  complete interaction clipping. Current position improves across every axis,
  current velocity and every pooled horizon improve, and identity/coverage/NLL
  improve; event F1 and lifecycle remain adverse. Treat this as encouraging
  sampled evidence only and continue directly to fixed selector 512.
- [x] Complete and reject the relation-only step-512 selector without losing
  the mutable trajectory. Aggregate score and mature pooled/x horizons improve,
  but current/short-z, coverage, precision, event, identity, and 109 pooled/
  per-scenario guardrails fail, especially on reference, impulse, elastic, and
  baseline regimes. Strict checkpoint scope/Adam/protected/provenance audit
  passes; the protected incumbent remains step zero and training continues to
  the required 8,192-update evidence horizon.
- [x] Prevent a fixed-boundary attention audit from silently labelling an old
  `last.pt` while validation is still publishing the requested checkpoint.
  `--expected-step` now records and enforces the embedded step; the true 512
  artifact passes and the quarantined step-384 artifact fails as intended.
- [x] Require non-empty serialized Adam steps to equal the checkpoint payload
  step. Record the agreement, reject stale/mixed optimizer boundaries, and
  prove both a synthetic 128/127 failure and the real 512/512 pass.
- [x] Audit the complete relation-only post-selector updates 520--576 window.
  Operational integrity, balance, support, clipping containment, identity,
  uncertainty, and memory pass. Current state/velocity and 0.10-second
  position improve, but mature x increasingly regresses and drives a
  `+0.013616 m` 1-second pooled deficit; preserve this limitation unchanged to
  fixed selector 1024 rather than retune on training draws.
- [x] Preserve/audit relation-only checkpoint 640 and complete updates
  584--640. Strengthened payload/expected/Adam, scope, inherited, protected,
  finiteness, support, balance, and resource gates pass. Every pooled position
  horizon and x horizon improve on matched draws; collision F1 and mature
  velocity remain adverse. Continue unchanged to selector 1024.
- [x] Make `audit_training_dynamics.py --after-step N` strictly exclude step N
  for candidate, validation, and matched-reference rows. Regression tests pass;
  corrected post-640 evidence begins at 648 without overlapping the prior
  window. Do not fabricate matched deltas after the control ends at 640.
- [x] Audit the complete non-overlapping updates 648--704 window. All 64
  updates, eight-way balance, 2,130 trajectories, support, clipping
  containment, finiteness, and flat memory pass. Later position horizons,
  event/lifecycle, coverage, and NLL improve versus the preceding
  different-draw window, while current/short position, velocity, identity,
  x, and especially z worsen. Preserve this mixed evidence unchanged to the
  fixed step-1024 selector.
- [x] Preserve and strictly audit the relation-only step-768 checkpoint and
  complete updates 712--768. Exact step/Adam/scope/inherited/protected/
  finiteness evidence passes; 64 updates, eight-way balance, 2,081
  trajectories, all causal terms, clipping containment, and flat memory pass.
  Every position slice improves versus the preceding different-draw window;
  0.25/0.50/1.00-second velocity and later target coverage remain watch items.
  Continue unchanged to selector 1024.
- [x] Audit complete relation-only updates 776--832. All operational gates,
  eight-way balance, 2,578 trajectories, support, clipping containment, and
  flat memory pass. Versus the preceding different-draw window, target
  coverage and selected velocity horizons improve, while current state, every
  axis/position horizon, identity, event, coverage90, and NLL are adverse.
  Preserve the reversal as a watch signal and keep selector 1024 authoritative.
- [x] Preserve/audit relation-only checkpoint 896 and complete updates
  840--896. Exact step/Adam/scope/inherited/protected/finiteness evidence and
  all operational gates pass. Hard-contact force sensitivities are contained
  and nonpersistent. Different-draw current velocity, y/z, identity, and NLL
  improve, while x, every position horizon, event F1, and mature lifecycle
  support are adverse. Keep the fixed step-1024 selector authoritative.
- [x] Audit complete updates 904--960. All operational gates, exact balance,
  all 13 causal terms, 2,943 trajectories, clipping containment, and flat
  memory pass. Different-draw current/short state, identity, coverage90, and
  NLL are adverse, while mature position/velocity, event F1, and lifecycle
  improve. Continue unchanged through the fixed step-1024 selector.
- [x] Complete and reject the relation-only step-1024 selector. Exact fixed
  validation scores `0.3409900` versus protected `0.3213162` with 116 broad
  guardrail failures and zero support failures. Current/short position,
  coverage, precision, and identity regress, dominated by `reference_pairs`
  x, while velocity and collision F1 improve. Strict step/Adam/scope/
  inherited/protected/finiteness audit passes, so this is behavioral
  non-convergence rather than checkpoint corruption; no candidate is promoted.
- [x] Audit complete non-overlapping updates 968--1024. All 64 updates,
  eight-way balance, all 13 objectives, 2,352 trajectories, clipping
  containment, and flat memory pass. Hard-contact force spikes remain bounded
  and nonpersistent. Preserve the fixed-selector regression and continue the
  immutable trajectory toward 8,192 updates without weakening selection.
- [x] Correct collapse-auditor severe-clip attribution for a deliberately
  frozen relation-only node path. Ignore node row/output coefficients only
  under explicit `closed_loop_scope_attention_relation_only=1`; preserve real
  relation warnings. Regression tests pass (`16 passed`).
- [x] Qualify the active warmup/cosine control through fixed selectors and the
  declared plateau. Keep all depth/width/history scaling gated on broad
  fixed-manifest convergence and disjoint RGB-only generalization. Its first
  complete updates 8--64 audit passes with all 64 updates, balanced eight-way
  exposure, 2,462 trajectory targets, zero skips/failures, minimum complete
  interaction retention `0.497461`, and stable peak RSS. The exact matched
  constant-rate comparison is a near-tie with mixed axes/horizons, so continue
  unchanged to the durable step-128 audit and fixed selector 512. The strict
  step-128 artifact audit now passes with `48/48` attention tensors live,
  all 177 inherited tensors exact, complete attention-only Adam state, finite
  serialization, and intact protected incumbents. The exact matched 72--128
  window is position/x/collision/lifecycle adverse despite mixed velocity/y
  gains; retain it as a watch item and keep selector 512 authoritative. The
  next balanced 136--192 window is still position-adverse against the exact
  constant-rate draws, but improves velocity, identity, uncertainty and
  gradient conditioning; with only 25.13% of constant cumulative LR exposure
  at this boundary, continue without mutation rather than infer collapse. At
  durable step 256 the strict audit passes again and the matched position gap
  narrows materially, though x/every horizon remain adverse. Deterministic
  emitted y bias is `0.128556 m/s²`, down from constant-rate `0.195037`, but
  remains drift-dominated; preserve selector 512 as the decision boundary.
  Complete 264--320 then nearly matches constant position while improving
  current/short velocity, identity, event F1 and coverage90; long position,
  y/long-axis, lifecycle and NLL deficits remain. Continue through warmup 384
  and selector 512 without promotion or mutation. Warmup now completes cleanly
  at durable step 384: strict integrity passes and current/0.10-second position
  plus lifecycle improve versus constant, but 0.25--1.00-second position and
  several velocity/event/calibration slices remain adverse. Preserve the
  selector-512 gate. The first complete cosine-phase 392--448 window is also
  operationally healthy and contains one step-424 spike above the 10% floor,
  while exact-draw identity/lifecycle/event improve but current position,
  velocity, coverage90, and nearly every position horizon remain slightly
  adverse. This is neither collapse nor broad convergence; keep the selector-
  512 and scale gates unchanged. The completed selector subsequently rejects
  the schedule more broadly than constant rate, closing this task as a failed
  same-capacity control rather than convergence.
- [x] Strictly audit the active candidate at durable step 256 and measure its
  emitted residual on a deterministic causal draw. Verify `48/48` attention
  tensors and complete Adam state changed, all 177 inherited tensors and both
  protected incumbents remain exact, and record the remaining drift-dominated
  y acceleration alongside the complete 184--240 axis/horizon trade-off.
- [x] Add read-only exact task/prior/configured-total gradient-alignment
  diagnostics over the full attention module and node decoder. On two balanced
  step-256 draws, prove alignment alternates from jointly restoring to directly
  conflicting; retain warmup/cosine as the gated same-capacity experiment and
  do not mutate the current run or add gradient surgery.
- [x] Qualify durable step 384 and the complete 328--384 matched window. Prove
  exact inherited/protected state and complete Adam ownership, reproduce
  alternating task/prior alignment, and record the first broad mature-horizon,
  velocity, collision, and identity gains without promoting before selector
  512.

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
- [x] Stop the first updater-wide recovery at durable step 64 after exact
  validation rejected it and checkpoint deltas proved that the compatible
  trunk, gate, variance, existence, and mode paths had drifted with the reset
  mean head.
- [x] Stop the corrected mean-head-only recovery at durable step 192 after
  three exact selectors proved a systematic rejected plateau despite 1,536
  balanced episode draws and healthy optimizer/support/resource dynamics.
- [x] Add provenance-recorded row-level checkpoint composition and qualify the
  step-64 x/y/z rows plus the step-128/192 y learning curve. Reject x and z;
  retain step-192 y-only as the guardrail-clean corrected incumbent
  (`0.3241755 -> 0.3216427`).
- [x] Add an exact `updater_mean_y` training scope that masks excluded rows,
  restores them across AdamW decay, and clears their optimizer moments.
- [x] Launch a clean y-only sustained recovery from the accepted step-192 row
  composition without reactivating x/z. At step 64, independently verify that
  only mean-head row 1 changed and every excluded Adam moment remains zero.
- [x] Train protocol 20 through repeated exact validation and a declared
  plateau. Step 64 is guardrail-clean and internally accepted at `0.3215594`,
  while steps 128/192 are rejected at `0.3216703/0.3216706` after the same
  baseline association threshold regressed coverage, identity, and
  0.10-second x. Step 256 returns to zero guardrail failures at `0.3215611` but
  is microscopically worse than step 64 and is not accepted; step 320 returns
  to the same rejected baseline threshold at `0.3216708`; step 384 returns to
  the clean regime at `0.3215634` but does not beat step 64; step 448 crosses
  back to a rejected baseline association regime at `0.3216787`. Continue the
  declared 512-update run with step 64 protected, and inspect pooled and per-scenario
  current/velocity, x/y/z, every horizon, identity, lifecycle, events,
  calibration, support, optimizer state, and memory. Step 512 is rejected at
  `0.3216317` by `reference_pairs` x/current and x@100-ms guardrails; the full
  optimizer/scope/resource audit passes and step 64 remains protected.
- [x] Continue through the predeclared 512-update balanced minimum and reject
  promotion after seven consecutive candidates fail to improve protected step
  64. Record this as a healthy bounded-recovery plateau, not deployment
  convergence.
- [x] Trace the apparent association wobble to a fast-ROI measurement bug and
  reject numerically tied disconnected-component ownership without reducing
  the ordinary `0.75` recovery range. Exact replay removes the subpixel-to-
  `0.28` centre jump and makes step 64/512 behavior stable; paired public and
  physical validation improve the small control under rollout protocol 14.
- [ ] Extend local ownership evidence to touching/merged same-component RGB
  objects using observation-derived appearance or basin evidence, without
  simulator identity or cross-assigning source-bound ROI rows.
- [x] Implement stage A of the Mac attention pilot as four RMS-pre-normalized
  scene/entity/relation blocks, width 128, four heads, SwiGLU width 512, typed
  zero-initialized residual decoders, and 1.104M added parameters. Preserve
  object-slot permutation equivariance and exact graph behavior at
  initialization.
- [x] Add strict architecture-growth loading and an attention-only optimization
  scope. A one-update hybrid MPS/CPU smoke leaves all 177 inherited tensors
  bitwise unchanged and gives optimizer moments only to the four output-head
  tensors, with finite supported loss/gradients and bounded memory.
- [x] Add a function-preserving depth-only handoff for a qualified attention
  incumbent: accept only contiguous appended blocks, initialize their MHA and
  SwiGLU output projections to exact zero, prove zero-tolerance shallow/grown
  output equality, require identical non-depth runtime semantics, persist the
  transform provenance, and keep malformed/width partial transfers fail-atomic.
- [x] Repair the offline dynamics auditor's sparse-cadence progress count:
  distinguish absolute completed trainer step from logged optimizer
  confirmations, expose step gaps, and warn that sampled loss/gradient
  distributions are not per-update curves.
- [x] Add reproducible non-overlapping training-trend windows to the dynamics
  auditor. Pool physical sufficient statistics for current axes, every
  position/velocity horizon, coverage, collision F1, identity, lifecycle,
  uncertainty, parameter observability, support, gradient retention, and
  memory; explicitly label incomplete tails and keep selectors authoritative.
  The first live tail closes at step 832 with mixed position/velocity/identity
  movement but healthy support/lifecycle/event/gradient/resource evidence; do
  not intervene before fixed selector 1024.
- [x] Stop and preserve the first attention pilot at its durable update-128
  checkpoint after exact audit found that 47/48 attention tensors changed but
  `scene_projection.weight` did not: its sole input, `global_code`, remains
  zero throughout the current runtime. Verify all inherited weights and linked
  control hashes remained exact and every serialized tensor stayed finite.
- [x] Repair the scene token to consume live authoritative belief context:
  global code/uncertainty, gravity, camera transform/motion/intrinsics/
  uncertainty, and calibration. Preserve zero-output graph identity and prove
  finite nonzero scene-projection gradient with zero global code.
- [x] Pass the corrected focused suite (`129 passed`), complete non-device
  suite (`650 passed, 5 skipped, 1 deselected`), host-MPS device marker
  (`1 passed, 655 deselected`), Ruff, format, compileall, and diff gates.
- [x] Complete a clean two-update host smoke from the protected graph control:
  verify MPS RGB/CPU closed-loop placement, finite supported updates, zero
  skipped draws and frozen-perception gradient, all 177 inherited tensors
  exact, all 48 attention tensors changed, and nonzero scene-projection weight
  plus Adam moment by update two.
- [x] Stop the first live-scene sustained campaign at sampled update 64 after
  its interaction gradient rose to `45.3456` and the local clip coefficient
  collapsed to `0.02205` on an ordinary matched control batch. Trace the
  defect to mixed-unit raw scene features entering the projection before any
  Transformer normalization.
- [x] Add fixed non-affine pre-projection RMS conditioning for the scene token
  and an extreme-intrinsics regression that proves bounded finite projection
  input without adding a learnable scale or changing zero-output graph
  identity.
- [x] Pass the normalized implementation gates: focused dynamics `20 passed`,
  complete non-device `651 passed, 5 skipped, 1 deselected`, host MPS
  `1 passed, 656 deselected`, Ruff check/format, compileall, and diff check.
- [ ] Run the declared 8,192-update, 65,536-draw balanced attention-only
  campaign through repeated complete 32-episode selectors and a real plateau;
  retain the protocol-14 step-64 graph runtime as the protected control. The
  first run at `runs/20260810-114053-attention-pilot-stage-a/` is stopped and
  cannot count because its scene input was dead. The live-scene run at
  `runs/20260810-134330-attention-live-scene-stage-a/` is also stopped and
  cannot count because mixed-unit scene inputs caused severe projection
  gradient conditioning by sampled update 64. The normalized campaign at
  `runs/20260810-144901-attention-conditioned-stage-a/` started from the same
  protected graph checkpoint. Its complete step-zero selector exactly
  reproduces score `0.3213162196`; updates 8/16 are finite and improve matched
  raw-gradient conditioning from `0.2631/1.9980` to `0.2535/1.3194`. The exact
  update-64 failure batch improves `45.3456 -> 2.2961` at effectively unchanged
  loss and complete matched support, confirming the conditioning repair.
  The durable step-128 checkpoint passes exact scope/optimizer/finite/hash/
  support/resource audit: 177 inherited tensors exact, all 48 attention
  tensors live, optimizer state attention-only at step 128, and no severe
  clips or skips. Step 256 repeats the exact checkpoint pass; one event-heavy
  step-152 severe clip is followed by twelve normal sampled blocks and no
  resource growth, but recurs exactly 128 updates later at step 280 with raw
  norm `52.9646` and coefficient `0.01888`. The run is stopped at durable step
  256 without a selector and cannot count toward convergence.
- [x] Localize the periodic severe gradient with per-parameter Adam moments:
  the relation decoder collision-logit row dominates by orders of magnitude;
  normalized scene, entity/relation projections, and other typed rows do not.
- [x] Add optional attention collision-row clipping before the complete
  interaction/global hierarchy, preserve reconstructed true raw norms and
  row/interaction/global coefficients, bind the cap into protocol semantics,
  and extend the offline auditor plus focused tests.
- [x] Pass repaired gates: focused `236 passed`, final affected `216 passed`,
  complete non-device `657 passed, 5 skipped, 1 deselected`, host MPS
  `1 passed, 662 deselected`, Ruff, format, compileall, and diff check.
- [x] Stop and preserve the collision-isolated attention campaign at
  `runs/20260810-180502-attention-collision-isolated-stage-a/`, launched
  weights-only from the same protected protocol-14 graph control and clean
  commit `70c2e3b`. Its 32-episode step-zero selector exactly reproduces score
  `0.3213162196` and the complete protected model hash. Prove the periodic
  frames 7--11 batches no longer suppress unrelated gradients before accepting
  its first trained complete selector. Continue through repeated selectors and
  the declared plateau. Through durable step 128 the auditor passes with 128
  applied updates, balanced support, zero skips/failures/severe clips, and
  bounded memory. Exact audit proves 177 inherited tensors unchanged, all 48
  attention tensors live, all 48 optimizer states attention-owned at Adam step
  128, finite state, and intact protected hashes. Partial training remains
  non-promotion evidence. The former step-152 event-heavy failure now improves
  raw norm/retained interaction coefficient from `28.1387/0.03554` to
  `7.1111/0.14308` with complete support and no severe clip. Step 280 recurrence
  and the step-512 selector are still pending. Durable step 256 independently
  passes exact tensor/optimizer/hash/finite/resource audit: all inherited
  tensors remain exact, all 48 attention tensors and only those optimizer
  owners are live at Adam step 256, and 256 balanced updates have zero hard
  audit failure. Track the isolated step-248 trusted-identity spike (`6.35%`;
  aggregate `0.975%`) through fixed validation without tuning to one batch.
  Stop the run after step 280 reproduces severe whole-interaction clipping
  (`17.7050`, coefficient `0.05648`) outside the collision row (`0.23553`).
  Preserve exact step 256 and do not count this campaign toward convergence.
- [x] Add read-only raw-gradient telemetry for every named attention parameter
  and all semantic node/relation decoder rows before any clipping; finite-check
  the complete diagnostic without changing gradients or forward behavior.
- [x] Replay exact steps 257--280 from the durable checkpoint with optimizer,
  RNG, and deterministic data continuity. Shared telemetry is bit-exact at
  steps 264/272/280; step 280 localizes `17.6842` joint raw gradient to the
  normal/tangent force rows inside `17.7050` total interaction gradient.
- [x] Add a configured joint force-row cap before the complete interaction
  hierarchy, preserve raw row/group/stage diagnostics, protocol-bind it, make
  the auditor inspect it, and correct stale checkpoint specification metadata.
- [x] Relaunch a fresh weights-only 3.00M-parameter stage-A campaign from the
  protected graph control at
  `runs/20260810-213857-attention-force-isolated-stage-a/`, clean commit
  `b3b69c1`, under one-shot Standard launchd with MPS measurement, CPU closed
  loop, and no oracle. Its mandatory step-zero selector is complete.
- [x] Complete and independently audit the step-zero selector: all model
  tensors and 2,583 comparable broad metrics exactly reproduce the protected
  graph control; specification metadata is 1.27 and only the optimization
  protocol hash changes.
- [x] Attach an exact-source one-shot convergence supervisor with the 8,192
  minimum, 4,096 extensions, four-selector/1% plateau rule, and 24,576 hard
  limit; verify the trainer/supervisor runtime fingerprint matches and stderr
  is empty.
- [x] Audit the first eight balanced updates: all scenarios supported, 349
  trajectory targets, no skipped draws, frozen perception, finite unclipped
  gradients, zero trusted identity switches, bounded RSS, and a passing live
  dynamics report.
- [x] Audit sampled updates 16 through 72. Every update is finite and supported; the
  step-24/32 collision-row outliers are contained by the row cap before the
  interaction/global hierarchy, no severe coefficient or hard auditor failure
  occurs, frozen perception remains exact, all scenarios remain balanced, and
  RSS remains bounded. Do not infer an accuracy direction before the fixed
  selector.
- [x] Localize the severe step-64 warning to the joint force decoder rows:
  `21.4665` of `21.5377` raw interaction norm. Verify row isolation leaves a
  `2.01547` post-row norm and `0.496162` stage coefficient for unrelated
  learning, then verify step 72 returns to ordinary force/stage coefficients.
  Keep the warning visible and require checkpoint/selector evidence before
  deciding whether it is isolated or systematic.
- [x] Independently audit the force-isolated step-128 checkpoint: all 177
  inherited tensors exact, all 48 attention tensors live, optimizer state
  attention-only at Adam step 128, all serialized state finite, protected
  control/reference hashes intact, 128 applied balanced updates, and bounded
  memory. Preserve the audit JSON beside the run.
- [x] Pass the exact historical step-152 failure boundary. On the identical
  seeds/window, improve raw interaction norm/retained stage coefficient from
  `28.1387/0.03554` before row isolation and `7.11114/0.14308` after collision
  isolation to `2.46615/0.48940`; keep force ordinary and preserve complete
  support, identity, uncertainty, finite-state, scope, and resource health.
- [x] Stop the force-row-isolated campaign at durable step 256 after exact
  step 280 shows that post-backward decoder-row clipping cannot prevent a raw
  `989.7965` force signal from first contaminating shared attention gradients;
  preserve the finite applied update only as diagnostic evidence.
- [x] Add separately configured per-invocation node, collision, and joint-force
  typed-output backward caps before the decoder/shared stack, retain the later
  parameter-row hierarchy, protocol-bind both layers, and expose them to the
  offline auditor.
- [x] Replay exact steps 257--280 from the durable source state. On the same
  step-280 seeds/window, reduce the later parameter norm `995.5391 -> 10.8330`,
  bound the maximum shared parameter gradient to `0.0851`, retain `0.6979` at
  the post-row interaction stage, apply a finite supported update, and keep
  localized severe output/row coefficients visible as warnings.
- [x] Stop the first fresh output-isolated campaign immediately after update
  200 exposes a previously uncapped impulse multiplier/additive gradient:
  raw total `857.1579`, impulse rows `830.3828/210.3096`, shared maximum
  `6.2401`, and complete-stage retention `0.001167`. Preserve step 128 as the
  last durable source and do not count step 200 or this run toward convergence.
- [x] Add separately configured per-invocation and accumulated decoder-row
  isolation for the joint impulse outputs, protocol-bind the controls, validate
  legacy `null` behavior, and expose raw/applied/minimum/effective telemetry.
- [x] Make sub-10% complete interaction retention after all local isolation a
  hard offline-audit failure, and add an active fail-fast gate that clears
  gradients and rejects the update before Adam state or weights change.
- [x] Replay updates 129--200 from the durable step-128 model/optimizer/RNG/
  sampler state. On the same update-200 seeds/window/support, reduce raw norm
  `857.1579 -> 7.4410`, maximum shared norm `6.2401 -> 0.05334`, and increase
  complete-stage retention `0.001167 -> 0.64704`; pass the offline replay audit
  with no severe or uncontained blocks. Keep this branch explicitly
  non-promotable because earlier repaired updates change the learned weights.
- [x] Stop the new immutable weights-only 3.00M stage-A campaign at
  `runs/20260811-042704-attention-impulse-isolated-stage-a/`, launched from the
  protected graph control under specification 1.30 and clean commit `d38cc9b`.
  Its 32-episode step-zero selector is tensor/metric exact with the protected
  control (`0.3213162196`), and sampled update 8 is finite, fully balanced,
  supported, and entirely unclipped (`0.6740` raw/applied interaction norm),
  but attempted update 60 is deterministically rejected before Adam at
  `0.0850405` complete-stage retention. Do not count its 59 applied updates or
  reuse its mutable state.
- [x] Replay the exact failed trajectory through update 60 with durable
  structured failure diagnostics. Match all 400--454 comparable model/data
  fields at logged updates 8--56, localize the remaining `11.6617` accumulated
  node-decoder norm to world-y `11.5014`, and verify the largest shared
  non-decoder attention tensor is only `0.124876`.
- [x] Add a joint accumulated node x/y/z decoder cap before the existing
  collision/force/impulse/interaction/global hierarchy; protocol-bind it,
  normalize legacy checkpoints to `null`, expose raw/applied/intermediate
  diagnostics, make terminal optimizer failures durable, and make the offline
  auditor fail them.
- [x] Complete the fresh protected-control node-row-repaired causal replay
  through update 60. Its initial selector is tensor/metric exact; on the same
  seeds it is fully supported and retains `0.565343` at the complete stage.
  Preserve its deliberate pre-Adam stop and report as diagnostic-only.
- [x] Pass full regression gates: `706 passed, 5 skipped, 1 deselected`
  non-device; host MPS marker and five direct regressions pass; Ruff,
  compileall, diff, and host dry-run placement pass.
- [x] Launch a new clean specification-1.31 campaign at
  `runs/20260811-063308-attention-node-isolated-stage-a/` from clean commit
  `5b2da41`. Its complete 32-episode step-zero selector is tensor/metric exact
  with the protected control at score `0.3213162196`; trainer and immutable-
  source convergence supervisor are active once with empty stderr.
- [ ] Monitor the active campaign. It must pass update 60, the later
  64/152/200/280
  stress positions, durable checkpoint integrity, fixed selector 512, repeated
  selectors, and plateau while retaining identity, per-axis, every-horizon,
  lifecycle, event, uncertainty, and scenario guardrails before scaling.
  Through update 128, the live audit passes with all updates applied, exact
  logged scenario balance, zero skips/failures/uncontained interaction clips,
  bounded RSS, cumulative trusted identity rate `1.006%`, calibrated finite
  uncertainty, and support at every axis/horizon. The durable checkpoint audit
  passes with 177 inherited tensors exact, all 48 attention tensors live, and
  attention-only Adam state at step 128. Update 152 subsequently passes with
  exact logged scenario balance, 343 causal targets, every horizon supported,
  `0.344214` complete-stage retention, and cumulative trusted identity rate
  `1.34%`. This clears the update-60/64/152 and first checkpoint-integrity gates
  only. Update 200 then passes with ordinary impulse multiplier/additive norms
  `0.18604/0.00781`, `0.873850` complete retention, 339 causal targets, exact
  scenario balance, no skip/failure, and cumulative trusted identity `1.24%`.
  The step-256 checkpoint independently passes with 177 inherited tensors
  exact, all 48 attention tensors live, and attention-only Adam step 256.
  Update 280 then passes with raw/post-row interaction `2.86878/1.29273`,
  `0.348580` complete retention, 145 causal targets, every horizon supported,
  no skip/failure, and cumulative trusted identity `1.24%`. All historical
  optimizer stress gates now pass. The independent step-384 checkpoint audit
  also passes: 177 inherited tensors remain exact, all 48 attention tensors
  are live, exactly those 48 own finite Adam state at step 384, and all hashes
  agree. Across 384 updates, every scenario has 48 logged blocks, no update is
  skipped or uncontained, cumulative identity is 26/2,105 (`1.235%`), pooled
  coverage90 is `90.34%`, all horizons have weighted support, and memory remains
  bounded. Fixed selector 512, repeated selectors, plateau, and held-out
  generalization remain open. The fixed step-512 selector subsequently
  completes all 32 episodes with full eight-scenario support but is rejected:
  score `0.330772` versus step-zero `0.321316`, pooled current position RMSE
  `0.295016` versus `0.251460 m`, target coverage `0.34775` versus `0.37625`,
  and broad x/z, reference-pair, and impulse-perturbation regressions. Its
  checkpoint audit is scope/optimizer/hash clean and the live audit passes all
  512 updates, so preserve the safe step-zero incumbent while continuing the
  mutable trajectory toward the next selector. Repeated selectors, repair,
  plateau, held-out generalization, and scaling remain open.
  Through step 576, the continued trajectory remains optimizer/support clean.
  Equal eight-block training samples improve every pooled horizon and all x/z
  horizons relative to the pre-selector window, while coverage90 slips
  `90.49% -> 89.17%` and y at 0.5 seconds worsens. Steps 560/568 contain local
  force/node outliers with post-isolation shared-stage retention
  `0.3522/0.7547`, safely above the `0.1` gate. Continue unchanged to fixed
  selector 1024; do not treat the sampled repair direction as promotion.
  The following equal scenario-balanced window, steps 584--640, reverses that
  direction: every pooled horizon and every x/z horizon worsens, coverage90
  falls to `88.10%`, while identity improves to `0.76%`. Support counts and
  deterministic windows differ, so classify this as training-sample wobble,
  not a matched regression or repair. The auditor still passes all 640 updates
  and step 640 retains `0.6927` after semantic isolation. Keep selector 1024 as
  the next decision point. Through step 704, the auditor still passes all
  updates with exact logged balance, zero skips/failures/uncontained clips,
  and stable memory. The next equal window is mixed: current/short horizons and
  identity worsen, 0.50--1.00-second horizons improve materially, coverage is
  flat, and shared-stage retention is healthy. The following 712--768 window
  improves current and every horizon, identity, and current-state coverage;
  the exact step-768 scope/optimizer/finite/protected audit passes. Continue
  unchanged; neither heterogeneous window is matched selector evidence. The
  step-896 checkpoint also passes with all 48 attention tensors live, all 177
  inherited tensors exact, attention-only complete Adam state at step 896,
  finite serialization, and intact protected hashes. Its complete 840--896
  training window keeps current error nearly flat but regresses every forecast
  horizon and lifecycle/event slices relative to 776--832 while retaining
  healthy coverage, uncertainty, support, gradients, and memory. Treat this as
  a trend warning and continue unchanged to the fixed selector at step 1024;
  do not promote, scale, or mutate the protocol from heterogeneous windows.
  The campaign subsequently terminated safely at attempted optimizer step 988:
  complete interaction retention `0.0971759` fell below the `0.1` floor, so
  Adam did not mutate and selector 1024 never ran. The supervisor verified the
  durable failure and stopped without relaunch. This is a failed,
  non-promotable campaign rather than convergence evidence.
- [x] Localize attempted step 988. Per-invocation output hooks allowed 144
  recursive force/impulse gradients to accumulate beyond their nominal budget,
  producing normal-force/shared norms `10.9076/5.01609` despite decoder-row
  isolation. Do not remove or lower the complete-retention gate.
- [x] Implement specification-1.35 aggregate per-draw semantic output budgets
  using `cap / sqrt(K)` for `K` registered calls; preserve exact one-call and
  forward behavior. Focused regression gates pass (`314 passed`).
- [x] Replay exact model/Adam/RNG/sampler state from durable step 896 through
  attempted step 988 under specification 1.35. Require aggregate semantic
  norms at/below configured caps, complete retention at/above `0.1`, and a
  deliberate pre-Adam diagnostic stop before any fresh sustained campaign.
  The matched draw passes at `0.647948` retention; raw/post-row norm falls
  `15.1704/10.2906 -> 4.58029/1.54333`, maximum shared norm falls
  `5.01609 -> 0.225929`, and every aggregate semantic applied norm is below
  `0.1`. Preserve the replay as non-promotable.
- [ ] Launch a fresh immutable specification-1.35 attention-only campaign
  weights-only from the protected graph control. Require exact step-zero
  selector equality, step-988 containment, repeated fixed selectors, broad
  non-regression, plateau, test/OOD generalization, and no lifecycle/identity/
  uncertainty/axis/horizon regression before scaling.
  The campaign is now active at
  `runs/20260811-170842-attention-aggregate-isolated-stage-a/` from clean commit
  `23ecf9d`; trainer/supervisor each run once with empty stderr. The initial
  32-episode selector completed exactly: `225/225` tensors and `2,584/2,584`
  metrics match the protected selector at score `0.3213162195855908`. The
  first balanced optimizer block is active. Keep this task open until trained
  selectors, plateau, test/OOD, and broad non-regression actually pass.
  Step 8 subsequently completed with all eight updates applied, exact
  eight-scenario balance, `349` causal trajectories, zero skips, interaction
  retention `1.0`, applied gradient `0.283628`, and a passing whole-run audit.
  Continue unchanged to durable checkpoint/selector boundaries; do not infer
  convergence from this single healthy block.
  The run is now healthy through step 32: all updates apply, exact four-draw
  balance per scenario, minimum complete retention `0.716607`, no failure, and
  nearly flat RSS. Continue to the first fixed trained selector without a
  protocol mutation.
  Step 64 also passes the complete eight-block audit: 64 applied updates,
  eight draws per scenario, `2,461` causal trajectories, zero skip/duplicate/
  terminal/uncontained failure, minimum retention `0.585590`, and bounded
  `2.992 GB` RSS. Axis/horizon/lifecycle/identity/uncertainty diagnostics are
  recorded in STATUS; keep this task open for fixed selector evidence.
  The durable step-128 checkpoint passes: all 48 attention tensors changed,
  all 177 inherited tensors remain exact, and finite Adam state belongs to
  exactly the 48 attention parameters at step 128. The matched 72--128 window
  slightly regresses all position horizons and lifecycle/uncertainty while
  improving identity and short/mid velocity; keep it as a trend warning.
  `checkpoint_every=128` and `eval_every=512` are intentional, so the first
  trained fixed selector is step 512, not step 128.
  The complete matched 128--184 window improves 0.5--1.0-second position and
  identity but regresses collision F1, lifecycle, and slightly current/short
  position; velocity is mixed. Preserve this as an event/identity tradeoff and
  require the step-512 selector rather than promoting training-window gains.
  The step-256 checkpoint also passes exact attention-only tensor/Adam/
  finiteness audit. Its matched 192--248 window regresses every velocity
  horizon, collision F1, and uncertainty while position is nearly flat and
  identity improves slightly. Continue without promotion to fixed step 512.
  Step 264 remains live and contained: all 264 updates apply, exact scenario
  balance and all 13 objectives are present, RSS remains `2.992 GB`, and the
  `2.850016` raw interaction gradient retains `0.350875` after the declared
  semantic/row hierarchy. No runtime failure exists, but the earlier physical
  regressions mean the no-regression prerequisite for scaling is still false.
  A refreshed primary-source review finds no missing long-context/MoE kernel
  that addresses this 22-token bottleneck; keep the one-axis capacity ladder
  gated on fixed selector, plateau, and disjoint test/OOD evidence.
- [x] Add deterministic matched-reference mode to the dynamics auditor. Require
  exact step/seed/scenario/draw/frame/anchor alignment, fail missing or
  mismatched schedules, and independently pool sufficient statistics before
  signed candidate-minus-reference deltas. Focused tests pass (`15 passed`).
  The exact steps 192--272 comparison is schedule-clean but shows all velocity
  horizons and collision/lifecycle/current-state slices regressing while long
  position and median uncertainty are nearly flat/slightly better. Preserve
  this warning and keep fixed selector 512 authoritative.
  The historical step-280 stress block also passes with `0.462814` complete
  retention and all aggregate semantic output norms below `0.1`. Exact matched
  physical evidence improves current/all position horizons but regresses
  collision F1 and three velocity horizons, especially one second. Keep the
  run unchanged; this is repair qualification plus another accuracy tradeoff.
  The complete matched steps 264--320 window remains optimizer/support clean
  but is non-promotable: short position and y improve, while every x horizon,
  medium/long pooled position, four velocity horizons, coverage90, and identity
  regress. Identity doubles from three to six switches on nearly equal support.
  Require the fixed step-512 selector; do not tune from this training window.
  The durable step-384 checkpoint also passes: all 48 attention tensors are
  live, all 177 inherited tensors remain exact, and complete finite Adam state
  belongs only to the 48 attention parameters at step 384. The matched
  328--384 window improves current position/velocity, collision F1, and every x
  horizon, but regresses every pooled forecast horizon, every z horizon, three
  later velocity horizons, lifecycle, and identity. Preserve the repaired x
  direction without promotion; fixed selector 512 remains authoritative and
  the no-regression scaling prerequisite remains false.
- [x] Complete the latest persisted step-512 audit and typed decoder
  ablations. Prove the checkpoint is finite/scope-clean, localize the broad
  regression to a node-y row whose L2 norm is about nine times x/z, and retain
  the zero-y composition only as a non-promotable diagnostic despite its
  pooled all-horizon improvement.
- [x] Implement specification-1.36 residual parsimony as an opt-in,
  axis-neutral attention-node decoder-row energy objective with exact legacy
  zero contribution and per-axis diagnostics. Focused tests pass (`312`).
- [ ] Qualify the residual-parsimony repair from the protected graph control
  with a recorded `training.loss_weights.attention_node_complexity=1.0`
  override. Require a finite scope-clean smoke, complete 32-episode fixed
  selectors, repeated plateau evidence, and strict per-scenario/test/OOD
  non-regression before scaling depth or width. The fresh run is active at
  `runs/20260811-234157-attention-node-parsimony-stage-a/` from clean commit
  `bbdb3ad`; its one-shot trainer and immutable-source supervisor each run
  once with empty stderr. Initial validation exactly matches all 225 control
  tensors and 2,844 common non-resource metrics. The first eight balanced
  updates apply with matched support, no skip, finite gradient, and a passing
  dynamics audit. Through step 64, the complete eight-block auditor still
  passes with all scenarios balanced, all updates applied, `2,461` causal
  trajectories, minimum complete retention `0.566722`, stable memory, and
  empty trainer/supervisor stderr. Exact matched comparison is slightly adverse
  on pooled current position (`+0.000237 m`), velocity (`+0.005579 m/s`), all
  five position horizons (`+0.000262` to `+0.000758 m`), and one trusted
  identity switch. Treat this as an early warning, not selector evidence.
  Continue to checkpoint 128 and fixed selector 512 without a promotion or
  capacity claim. The step-128 checkpoint now passes exact integrity audit:
  all 48 attention tensors are live, all 177 inherited tensors remain exact,
  exactly 48 complete finite Adam states are at step 128, and protected/hash
  provenance is intact. The full matched window improves current and every
  position horizon with complete balanced support and stable memory, but
  trusted identity is `9/703` versus `4/699` and 0.25/0.50-second velocity plus
  collision F1 regress slightly. Continue unchanged to selector 512 while
  treating identity as an explicit guardrail warning, not a promotion. The
  post-checkpoint trajectory clears the historical step-152 stress position:
  all updates through 152 apply with complete support, minimum stage retention
  `1.0`, stable memory, and no new identity excess after the step-128 block.
  Every axis/horizon position slice improves versus the matched predecessor,
  but current/0.10-second velocity and aggregate collision F1 regress. Preserve
  both sides of the tradeoff and continue unchanged to selector 512. Through
  sampled step 208, the isolated step-184 short-only association-set change is
  followed by healthy complete-support blocks. Matched steps 192/200 improve
  current state and every pooled position horizon; adding step 208 produces a
  small 0.50-second/collision/lifecycle tradeoff but no support, identity,
  optimizer, numerical, or resource collapse. Keep selector 512 authoritative
  and do not launch a larger rung from cadence-sampled evidence. Durable step
  256 now passes exact tensor/optimizer/protected-checkpoint audit. The matched
  128--256 window improves long position horizons, velocity, collision,
  lifecycle, uncertainty, and memory but regresses current x/z, short position,
  and identity. The complete matched 256--312 window remains optimizer-healthy
  and improves current/long-horizon position plus x at every horizon, but
  regresses short-horizon position, current velocity, most y horizons,
  aggregate collision F1, and median uncertainty error. Continue unchanged to
  fixed selector 512; do not promote or scale from this mixed training window.
  Durable step 384 passes strict tensor/optimizer/finiteness/provenance and
  required-protection audits. Its exact 328--384 matched window regresses
  current and every pooled position horizon, entirely through x while y/z
  improve; velocity, collision, and lifecycle improve slightly, but identity
  and uncertainty remain adverse. A source audit finds no axis-order or
  selector bug. Keep the scale gate closed through selector 512.
- [x] Make protected-checkpoint audits non-vacuous under specification 1.37:
  record protected count, return `null` when none were checked, and provide a
  required-protection gate that fails an empty set. Rerun step 256 with both
  protected artifacts and preserve their file/model hashes.
- [x] Audit the immutable residual-parsimony step-512 selector and run
  same-manifest functional ablations. Reject the full candidate despite its
  improved scalar score because 109 broad guardrails fail; preserve the
  relation/force branch as useful and localize the dominant regression to
  emitted node acceleration.
- [x] Implement specification-1.38 `attention_node_activity` as an opt-in,
  axis-neutral, active-object-normalized functional prior over bounded node
  acceleration across the causal rollout. Verify exact legacy opt-out,
  differentiability, padding support, reset semantics, and full regression
  (`718 passed, 6 skipped`). Retain it as diagnostic infrastructure after the
  more precise drift decomposition below.
- [x] Calibrate functional activity on four deterministic balanced causal
  draws at the rejected step-512 checkpoint. Persist the checkpoint hash,
  axis values, RMS acceleration, functional/complexity gradient norms, emitted
  mean/std/range, invocation count, and active support. Prove more than 99.997%
  is context-invariant drift rather than useful variation.
- [x] Implement specification-1.39 `attention_node_drift` as the squared mean
  emitted acceleration, separate from residual variation. Verify balanced
  positive/negative activity has zero drift cost, exact legacy opt-out, full
  regression (`719 passed, 6 skipped`), and a `drift=0.08` dry run.
- [x] Run the specification-1.39 objective for two real balanced CPU updates,
  including an exact checkpoint resume. Verify a nonzero drift term on update
  two, all 13 causal terms, 343 supported trajectories, attention-only tensor
  and Adam ownership, exact inherited/protected state, and finite serialization.
  Keep the deliberately tiny smoke `last_unvalidated`; it is not an accuracy
  or generalization promotion.
- [x] Continue the immutable specification-1.36 campaign unchanged through
  fixed selector 1024. The strict step-1024 checkpoint audit passes, but the
  fixed selector rejects it by 111 broad guardrails despite a microscopically
  better scalar score. The familiar `reference_pairs` current/x trajectory,
  short horizons, coverage, precision, and several camera/contact slices
  regress. Stop the one-shot jobs at the durable boundary; classify this as
  behavioral overfit rather than numerical or optimizer collapse.
- [ ] Launch and qualify a clean
  specification-1.39 successor from the protected graph control with both
  `attention_node_complexity=1.0` and `attention_node_drift=0.08` recorded;
  do not seed it from rejected specification-1.36 weights. Require exact
  step-zero reproduction, repeated fixed selectors, validation/test/OOD
  non-regression, and plateau before scaling capacity. Preserve the same
  8,192-update minimum, 4,096-update extensions, and 24,576 hard limit unless
  a separately versioned fixed-manifest result justifies a protocol change.
  The clean run is active at
  `runs/20260812-102557-attention-node-drift-008-stage-a/` from pushed commit
  `176796f`; its one-shot trainer and immutable-source supervisor have empty
  stderr. The 32-episode step-zero selector completes in `991.16 s` and exactly
  reproduces the protected control at score `0.3213162196`, with zero
  guardrail/support failures. Initial causal updates, trained selectors, and
  convergence remain pending. The complete first 64-update window passes the
  matched dynamics audit with every scenario represented eight times, 2,462
  trajectories, no skips/failures, and minimum complete interaction retention
  `0.615309`. Pooled current and short-horizon position improve slightly, while
  0.50--1.00-second aggregate differences are sub-millimetric regressions.
  The strict durable step-128 checkpoint audit passes with all 48 attention
  tensors changed, 177 inherited tensors exact, 48 complete Adam owners at
  step 128, finite state, and both protected checkpoints unchanged. The
  complete matched 72--128 window has 2,586 trajectories and healthy support,
  gradients, identity/lifecycle, and resources, but position regresses
  `8.28--15.94 mm` across horizons versus the rejected predecessor. Same-draw
  functional calibration proves the prior reduces RMS node activity to
  `0.1062 m/s^2` from the predecessor's `0.2066 m/s^2`, while more than 99.95%
  of the smaller residual remains drift. The emitted-value diagnostic now
  verifies its 144 gradient-enabled calls and 5,184 active-object evaluations
  exactly match the differentiable loss population. Preserve the immutable
  run to fixed selector 512; do not promote or scale from training-window
  evidence.
- [x] Add a reusable read-only attention-checkpoint auditor that records whole-
  file/model hashes, recursive finiteness, configured shape/dtype agreement,
  inherited/protected tensor equality, named optimizer ownership, and Adam
  steps; preserve the exact step-128 checkpoint and JSON report.
- [x] Harden the future scale handoff under specification 1.32: preflight all
  checkpoint keys and tensor shapes before copying; reject partial learned
  attention-prefix growth such as a trained four-block source into a random
  six-block destination; prove rejection leaves the destination bitwise
  unchanged. The active graph-control-to-attention run is unaffected.
- [ ] Run a one-axis-at-a-time scaling study after stage A qualifies: matched
  data-only, depth, width, and bounded-history rungs with increasing balanced
  episode draws, fixed disjoint RGB-only validation/test/OOD manifests, and
  the accepted smaller checkpoint as a non-regression reference. Preserve the
  current 8,192--24,576-update curve as the data-only evidence; compare the
  declared 3.53M depth-six and 4.34M width-192 candidates before the 8.31M
  single-CUDA rung. Initialize depth six through the exact identity-appended
  transform; width remains graph-initialized. Test maximal-update
  parameterization only as a separate matched control, never as an implicit
  checkpoint reinterpretation.
- [ ] Add stage-B bounded timestamped belief/innovation history only after the
  current-belief attention stage qualifies; use temporal-relative encoding,
  never arbitrary object-slot order.
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
- [ ] Implement and benchmark the paper-inspired receding-horizon hypothesis bank: short-step transition loss, nearby residual/contact hypotheses, and innovation/uncertainty-based selection with unchanged broad guardrails.
