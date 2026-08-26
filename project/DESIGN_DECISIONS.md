# Design decisions

## ADR-153 — Run grounded recursive optimization on the measured faster CPU

- **Date:** 2026-08-21
- **Status:** accepted for the grounded profile; disabled-path trainer smoke
  complete and retained-hinge sustained campaign pending
- **Context:** The first full active-Aqua MPS launch was numerically healthy,
  but step-zero validation took `5165.944729` seconds and eight causal updates
  took about `1699` seconds. A matched same-window benchmark after removing
  unowned graph work measured recursive forward/backward compute approximately
  `3.5--3.9x` faster on CPU than the user-provided custom MPS build. The task is
  branch-heavy recursive autograd over small typed states, so MPS availability
  did not translate to better throughput. A `0.05 s` learned-effect hold gave
  only about a `1.16x` late-scope CPU compute speedup and changed forward
  semantics.
- **Decision:** Keep `device.preference: mps` and the supported hybrid RGB/MPS
  execution policy, but set `device.closed_loop_preference: cpu` in
  `configs/grounded_convergence_mps.yaml`. Keep
  `learned_effect_interval_seconds: null`, so learned relation effects execute
  at exact microstep cadence. Treat both device placement and cadence as
  resolved protocol and exact-resume semantics. Do not alter or replace the
  user's custom PyTorch build.
- **Alternatives considered:** insist on MPS because it is available; reduce
  the validation manifest or horizon support; enable the semantically
  non-identical multi-rate hold from throughput evidence alone; change the
  model architecture before removing unowned graph work; reinstall PyTorch.
- **Consequences:** The repaired CPU fixed 32-seed diagnostic completes in
  `261.963382` seconds and remains finite with selection score `0.2654622904`,
  close to the prior MPS step-zero score `0.2654857751`. This is execution and
  same-manifest accuracy evidence, not bitwise device parity, a trained gain,
  or convergence. The subsequent eight-update CPU trainer smoke used the
  disabled future-correction ablation and qualifies execution only. The full
  repository, lint, format, compile, and diff gates pass. Committed source,
  retained-hinge campaign execution, and the complete sustained selection
  protocol remain mandatory.

## ADR-152 — Expose every long causal update through an atomic stage heartbeat

- **Date:** 2026-08-21
- **Status:** accepted and implemented; disabled-path trainer exercise complete
  and retained-hinge sustained-run exercise pending
- **Context:** Training metrics are intentionally emitted sparsely. During the
  first MPS campaign this made a healthy sequence of long data, forward, and
  backward stages look stalled, and the first cadence row represented eight
  completed updates rather than one extremely slow update. Existing
  validation progress could also outlive its PID and mislabel a later trainer.
- **Decision:** Atomically overwrite `training_progress.json` at the `data`,
  `forward`, `backward`, and `optimizer` boundaries. Record PID, completed and
  attempted updates, target, absolute data-draw index, retry count, phase and
  scope, elapsed/stage/last-update timings, and applied-update state. Keep loss
  and physical trends in sparse `metrics.jsonl`. The read-only monitor accepts
  running progress only when the trainer lock is held and the recorded PID is
  compatible with its owner; otherwise it warns and ignores the stale running
  heartbeat.
- **Alternatives considered:** log a complete metrics row at every stage;
  reduce the monitor interval; infer progress only from CPU utilization;
  deserialize checkpoints from the monitor; trust the newest progress file
  without checking its process identity.
- **Consequences:** Operators can distinguish data generation, forward,
  backward, optimizer, retry, and completed-update time without changing
  checkpoint-selection evidence or consuming accelerator memory. Focused
  monitor/schedule tests pass. The completed eight-update disabled-path CPU
  smoke exercised the artifact through real applied updates; the retained-
  hinge sustained campaign remains outstanding.

## ADR-151 — Elide exact-zero attention only when semantics and ownership prove identity

- **Date:** 2026-08-21
- **Status:** accepted, implemented, and repository-gated
- **Context:** The protected checkpoint's typed attention decoders are exact
  zero in the early `state_roi` phase, yet recursive dynamics executed the
  complete token/attention/SwiGLU stack 138 times per draw. The frozen stack
  could not change the forward result or learn, while configured typed-output
  hooks still intercepted and potentially clipped upstream gradients owned by
  the belief/perception path. Functional-node bookkeeping also retained graph
  tensors despite having no owner.
- **Decision:** Prove exact-zero decoder state from finite values and cache the
  proof against parameter identity/version, `requires_grad`, and grad mode.
  Return the original structured interaction directly only when the output is
  exact zero and no trainable semantic decoder owner can consume a gradient.
  Fail open for a trainable decoder under autograd, every nonzero decoder, and
  training dropout. Permit no-grad/inference bypass of an exact-zero decoder
  even if it is declared trainable. Install typed-output hooks only for a
  semantic output with a trainable attention owner, and retain node-activity
  records only for a configured objective or trainable node output.
- **Alternatives considered:** always run the frozen stack; manufacture zero
  residual tensors after attention; key the cache only by `requires_grad`;
  bypass a zero trainable decoder during its first backward; leave old hooks
  active because their parameter gradients are zero.
- **Consequences:** Structured forward values and their true upstream gradient
  paths remain exact, while irrelevant recursive work and graph retention are
  removed. Later trainable/nonzero attention stages execute normally. The
  repaired 32-seed step-zero diagnostic remains finite and close to the prior
  selector, and the full repository gate passes. This optimization does not
  qualify learned attention accuracy.

## ADR-150 — Route each causal objective only through its semantic owner

- **Date:** 2026-08-21
- **Status:** accepted, implemented, and repository-gated
- **Context:** Section 186 froze the fast perception branch during
  `state_dynamics`, but its auxiliary ROI forward consumed a live predicted
  prior and cache. Geometry, existence, colour, likelihood, and world-position
  losses could therefore train the updater, identifier, or dynamics through
  conditioning rather than their authoritative physical objectives. Separately,
  an exact-zero event coefficient still built pair-event auxiliaries/BCE. The
  extra prior future-correction rollout also lacked an explicit switch, so its
  throughput cost could not be isolated in a matched ablation even though its
  prior-versus-posterior improvement hinge is accuracy-relevant.
- **Decision:** Feed the auxiliary-only fast-ROI forward a detached cloned
  prior and detached modality cache. Preserve the ordinary ingest's live
  prepared propagation and live cache. Resolve effective stage event weight
  before rollout construction and omit event auxiliary/BCE graph work at exact
  zero while retaining detached physical event metrics. Add
  `training.closed_loop_prior_future_correction_enabled`: legacy/default
  `true`, with the grounded accuracy profile also retaining `true`. Disabled
  means a matched ablation that omits only the extra prior rollout and future-
  correction terms, never current correction or posterior rollout. Bind the
  flag to configuration, checkpoints, validation protocol, and exact resume.
- **Alternatives considered:** rely on frozen perception parameters alone;
  detach the complete prior and lose physical state gradients; keep a zero-
  multiplied event graph for diagnostics; remove all correction supervision;
  change legacy checkpoint behavior silently.
- **Consequences:** Perception-local auxiliaries no longer steer physical state
  through their inputs, while the real predict-observe-associate-innovate-
  correct path remains differentiable. Event metrics survive a zero event
  objective without its unowned graph. Historical checkpoints retain their
  prior rollout behavior. Matched exact-cadence CPU timing puts the hinge's
  overhead at about `10.6%` of recursive compute and `6.4%` including data,
  which is modest relative to its explicit accuracy role. The completed eight-
  update disabled-path CPU smoke is technical execution evidence only; it does
  not authorize removing the correction hinge. Focused ownership/config/
  checkpoint regressions and the full repository gate pass; a changed flag
  still requires a fresh weights-only campaign.

## ADR-133 — Score runtime interventions through the runtime prediction seam

- **Date:** 2026-08-15
- **Status:** accepted and implemented
- **Context:** `evaluate.py --runtime-hypothesis-pool` correctly attached the
  explicit post-load controller, but broad forecast metrics called
  `model.dynamics.rollout` directly. This made the report metadata claim an
  intervention while measurements scored the learned candidate alone.
- **Decision:** Every scored future anchor now calls
  `OnlineWorldModel.predict`, the public normal-runtime seam. The controller
  remains outside `WorldBelief`, uses only delayed associated RGB evidence,
  and only composes configured axes. It rolls out candidate zero plus actually
  selected axis candidates rather than every alternative when they cannot
  affect emitted outputs. The evaluator provides an optional batch-level
  progress callback exposed as `evaluate.py --progress`.
- **Consequences:** Existing learned-only evaluation behavior is unchanged;
  enabled runtime-pool reports are now semantically valid. The feature remains
  opt-in until the complete fixed 32-episode MPS guardrail comparison passes.
  Reports count each configured axis's learned/CV/damped-CV/ballistic choice at
  forecast anchors; pre-evidence learned fallback is counted separately only
  through that same learned candidate label. Progress is atomically persisted
  to the timestamped evaluation directory so launchd/terminal detachment
  cannot turn a still-running guardrail run into an apparently empty log.

## ADR-131 — Runtime selection consumes associated RGB evidence only

- **Date:** 2026-08-15
- **Status:** accepted and implemented; broad MPS qualification pending
- **Context:** The fixed 32-episode MPS evaluator established a narrow causal
  benefit from x-only candidate selection, but its simulator-aligned targets
  are evaluation-only and cannot enter the normal RGB runtime.
- **Decision:** Add an opt-in runtime controller outside `WorldBelief`. After
  every corrected posterior it records short candidate rollouts. At a later
  packet it scores only forecasts whose configured endpoint exactly matches
  the RGB timestamp, using `world_position` reconstructed by the RGB module
  and the normal association mapping. It rejects late, interpolated evidence
  and slot-reused identities. The learned candidate remains candidate zero;
  only configured x coordinates are spliced into future trajectories, while
  learned lifecycle, identity, event, uncertainty, and the authoritative
  belief remain intact.
- **Alternatives considered:** use simulator target alignment in the runtime;
  score posterior positions after correction; maintain a second candidate
  belief; interpolate delayed asynchronous measurements; replace entire
  trajectories including lifecycle/events.
- **Consequences:** The feature is causally executable with RGB only, but it
  is disabled by default. It must pass the complete fixed active-Aqua MPS
  protocol before normal runtime promotion. Exact endpoint matching means a
  future asynchronous/interpolation contract is required before non-aligned
  sensors can contribute delayed evidence.

## ADR-132 — Runtime-policy evaluation is post-load and checkpoint-strict

- **Date:** 2026-08-15
- **Status:** accepted and implemented; MPS report pending
- **Context:** Enabling the runtime pool in a resolved config must not make an
  existing checkpoint appear semantically compatible, while old checkpoints
  naturally lack the newly introduced disabled-policy fields.
- **Decision:** `evaluate.py --runtime-hypothesis-pool` first validates and
  loads the checkpoint against its unmodified disabled runtime config, then
  attaches the parameter-free policy as an explicit evaluation intervention.
  Missing historical pool fields normalize only to the disabled defaults.
  An enabled policy or changed enabled-policy values remain strict runtime
  mismatches.
- **Consequences:** Reports label the intervention and its candidate/horizon/
  axis policy. The checkpoint is neither rewritten nor treated as trained with
  the policy. A run that fails before producing a report is no evidence for or
  against the policy.

## ADR-117 — Make adjacent trend windows non-overlapping

- **Date:** 2026-08-13
- **Status:** accepted and implemented in specification 1.44
- **Context:** The dynamics auditor named its filter `after_step` and computed
  `after_step + 1` as the first expected step, but selected records using
  `step >= after_step`. Consecutive audits therefore shared one boundary row,
  biasing pooled physical sufficient statistics and confusing post-checkpoint
  windows. Candidate, validation, and reference filters all had the same bug.
- **Decision:** Define the boundary strictly as `step > after_step` in all
  three paths and regression-test exact-boundary exclusion plus matched-step
  alignment.
- **Alternatives considered:** rename the option to `at-or-after-step`; keep
  overlapping windows and document them; compensate by choosing boundary+1 at
  every call site; deduplicate only during report comparison.
- **Consequences:** `--after-step 640` now begins at the first persisted row
  above 640 (currently 648), without including 640. Historical JSONL remains
  unchanged. The old matched control ends at 640, so later matched comparison
  correctly fails rather than silently degrading to unmatched evidence.

## ADR-116 — Reject payload/Adam boundary mismatches

- **Date:** 2026-08-13
- **Status:** accepted and implemented in specification 1.43
- **Context:** ADR-115 binds an externally requested selector boundary to the
  payload step, but the auditor previously only displayed each parameter's
  Adam step. A checkpoint labelled internally as step 640 could therefore pass
  basic integrity while carrying stale step-512 or mixed optimizer state.
- **Decision:** Require the non-empty unique serialized optimizer-step set to
  equal the embedded checkpoint step, record the Boolean agreement, and fail
  independently of the optional external expected-step gate. Preserve empty
  optimizer state for legitimate step-zero or non-optimizer artifacts.
- **Alternatives considered:** rely on parameter ownership alone; inspect Adam
  steps manually; require only that all owners share some step; make the rule
  conditional on the external expected step.
- **Consequences:** A synthetic payload/Adam 128/127 mismatch fails, while the
  active selector passes payload/expected/Adam `512/512/[512]`. This changes
  qualification evidence only and leaves the immutable specification-1.42
  training process and serialized artifacts untouched.

## ADR-115 — Bind fixed-boundary audits to the embedded checkpoint step

- **Date:** 2026-08-13
- **Status:** accepted and implemented
- **Context:** The step-512 training heartbeat is emitted before the expensive
  fixed validation and its atomic checkpoint publication complete. An external
  command that immediately copied `last.pt` therefore received the valid but
  older step-384 artifact and could have named its report as step 512. Manual
  inspection caught the mismatch through the embedded step and model hash, but
  the auditor did not make the requested boundary part of pass/fail semantics.
- **Decision:** Add optional `--expected-step`/`expected_step` input, persist it
  in the report, and fail whenever it differs from the checkpoint's embedded
  step. Fixed-boundary operational audits must provide it. Preserve historical
  behavior when the option is absent so old reports and general ad hoc audits
  remain reproducible.
- **Alternatives considered:** publish a pre-validation checkpoint; change the
  pinned trainer's heartbeat ordering; infer the step from the filename;
  require operators to inspect hashes manually; treat the stale copy as model
  corruption.
- **Consequences:** The real step-512 artifact passes, while the quarantined
  copy fails with the exact 384-versus-512 mismatch. The change is read-only
  audit hardening: it does not modify training, serialization, selection,
  optimizer state, runtime behavior, or the active immutable campaign.

## ADR-114 — Train relations from protected weights, not contaminated donors

- **Date:** 2026-08-13
- **Status:** accepted; two-update qualification passed, sustained run active
- **Context:** Exact zero-node compositions of the warmup/cosine and constant-
  rate drift checkpoints remain harmful: scores `0.342289` and `0.329317`
  versus protected `0.321316`, with 100 and 98 broad guardrail failures and no
  support failures. Both improve y/velocity but strongly regress x. In
  contrast, the earlier complexity-only zero-node composition scored
  `0.297330` and improved every pooled horizon, although it still failed 72
  guardrails. The drift donors' shared relation stacks were trained through a
  nonzero node decoder, so deleting node output after training cannot remove
  node task/prior gradients already written into shared features.
- **Decision:** Reject both learned donors and initialize a constant-rate
  `attention_relation` campaign weights-only from the untouched graph control.
  Freeze the zero node decoder from update zero, leaving node activity, drift,
  complexity, gradients, and optimizer ownership exactly zero. Keep the
  declared 8,192-update/65,536-example minimum and every fixed selector,
  support, lifecycle, identity, uncertainty, event, axis, horizon, test, and
  OOD gate. Do not scale depth, width, history, or device budget.
- **Alternatives considered:** reuse either rejected donor; infer relation-only
  behavior from post-hoc deletion; restore the harmful node branch; add a new
  gate or regularizer before testing the already implemented isolation; scale
  the Transformer; relax fixed guardrails.
- **Consequences:** A two-update CPU smoke passes exact resume and strict
  46-trainable/2-frozen tensor and Adam ownership with complete finiteness and
  causal support. This validates execution only. A complete fixed selector is
  required for accuracy evidence, and only repeated accepted selectors plus
  disjoint test/OOD evidence can qualify convergence or capacity growth.
  The sustained run's step-zero control is bitwise exact across all 225 model
  tensors and 2,584 metrics. Its first 16 balanced updates remain finite,
  support-complete, node-exact, and behaviorally near the matched full-attention
  control with mixed axes/horizons. Continue unchanged; neither early sampled
  movement nor contained semantic-output clipping changes the selector gate.
  This remains true through step 64: all updates and balanced scenario draws
  complete without skips, node behavior remains exactly zero, complete
  interaction-gradient retention stays at or above `0.812481`, and the mixed
  axis/horizon plus adverse sparse collision samples provide no evidence for
  an early protocol mutation. The step-128 structural and step-512 selector
  gates remain authoritative.
  Step 128 subsequently passes the strict 46-trainable/2-frozen tensor and
  optimizer audit with exact inherited/protected state and complete finiteness.
  Its matched 72--128 window improves current state, short-horizon x, and
  collision F1 but regresses pooled 0.25--1.00-second position, especially z.
  Because optimization integrity is proved while accuracy remains mixed, this
  strengthens—not relaxes—the decision to wait for the fixed selector.
  The next complete 136--192 window is more broadly adverse: current and every
  pooled position horizon regress on exact draws, mainly x/z, while all
  operational invariants remain healthy. This rules out a simple monotonic
  early-win narrative but does not override the fixed-selector design: sampled
  windows are heterogeneous and have already changed sign. Continue unchanged
  to selector 512, where identical fixed episodes and scenario guardrails can
  distinguish generalization from training-window wobble.
  Updates 200--256 then reverse much of that sampled regression while passing
  a second strict checkpoint audit: current/velocity, lifecycle, event F1 and
  short position improve, but x remains adverse at every horizon and pooled
  0.50--1.00-second position still regresses. This observed sign reversal is
  direct evidence that neither adjacent window is a valid promotion/rejection
  substitute for the fixed selector; keep the protocol immutable through 512.
  Updates 264--320 reinforce that conclusion: mature x and pooled 1.00-second
  position repair, but the adverse residual migrates to z and grows with
  horizon. The model is learning nontrivial relation behavior rather than
  monotonically collapsing along one axis, yet sampled generalization remains
  unproved. Do not chase the moving axis with a mid-run rule or loss change.
  Checkpoint 384 then passes strict integrity while its matched window returns
  to an x-dominant all-horizon position regression; the prior z trend does not
  persist. Secondary velocity/event/lifecycle/identity gains coexist with
  calibration and long-event losses. This is precisely the unstable evidence
  pattern for which the fixed selector was declared, so retain the immutable
  run through 512 rather than fitting another adjacent window.
  Updates 392--448 subsequently improve current position across all axes,
  current velocity, and every pooled horizon with no complete interaction
  clipping, while collision F1 and lifecycle remain adverse. This favorable
  reversal validates continuing the experiment but does not erase prior
  wobble or authorize promotion; only the fixed selector can establish whether
  the learned relation behavior generalizes beyond the sampled draws.
  The fixed step-512 selector answers that question negatively for promotion
  but not for continued optimization. Aggregate score improves to `0.305413`
  and mature pooled/x forecasts improve, yet 109 pooled and scenario-specific
  guardrails fail, led by familiar reference, impulse, elastic, and baseline
  regimes; current and short-horizon z, lifecycle, event, and identity are
  adverse. Zero support failures and a strict tensor/Adam/protected/provenance
  audit prove this is a genuine mixed generalization result rather than
  collapse or corruption. Keep the step-zero incumbent protected, continue the
  separate mutable trajectory to the declared 8,192-update minimum, and do not
  scale capacity or weaken guardrails.
  The complete post-selector 520--576 window remains operationally exact and
  improves current state, velocity, 0.10-second position, aggregate collision,
  identity, coverage90, and uncertainty on matched draws. However, x error
  grows monotonically relative to the control across forecast horizons and
  drives a `+0.013616 m` pooled 1-second deficit; 1-second collision F1 is also
  adverse. This localizes the next watch item without changing the decision:
  preserve the same trajectory through fixed selector 1024, where identical
  held-out episodes can distinguish a persistent relation shortcut from
  heterogeneous sampled behavior.
  The subsequent complete 584--640 window reverses that mature-x deficit:
  current state/velocity, every pooled position horizon, and every x horizon
  improve on exact draws, with identity and uncertainty also favorable.
  Collision F1 at every supported horizon and 0.75/1.00-second velocity remain
  adverse. A step-616 raw force sensitivity of `363.837` is reduced to
  `0.071543` by the declared aggregate semantic cap before shared gradients;
  the finite complete update is wholly retained and the spike does not recur
  through 640. Strict step-640 integrity passes. Preserve the immutable run to
  selector 1024 rather than promote from this favorable sampled reversal.

## ADR-113 — Reject schedule-only repair and qualify relations before nodes

- **Date:** 2026-08-12
- **Status:** accepted and implemented; ablations complete, superseded by ADR-114
- **Context:** The protected-control warmup/cosine experiment is finite,
  balanced, scope-exact, and support-complete, but its first trained fixed
  selector worsens score `0.3213162 -> 0.3475480` with 116 broad guardrail
  failures plus failed improvement. Current position, coverage, precision,
  event F1, identity, and every position horizon regress. Familiar
  `reference_pairs` current x reaches `0.720231 m`. This is worse overall than
  the rejected constant-rate score `0.3332533`, so insufficient warmup is not
  the remaining explanation. Earlier same-manifest ablation of a related
  rejected checkpoint improved score to `0.297330` when the complete node
  decoder was zeroed, retaining learned relation/event outputs.
- **Decision:** Keep depth, width, history, and CUDA scale closed. Add an
  `attention_relation` scope that freezes the protected zero node decoder and
  trains the remaining typed projections, set-attention/SwiGLU blocks, output
  norm, and relation decoder. Extend the strict auditor with declared frozen
  attention prefixes so qualification requires 46 changed tensors with exactly
  46 Adam owners, two exact frozen node tensors, exact inherited/protected
  state, and complete finiteness. First run a fixed-manifest zero-node ablation
  of the latest rejected checkpoint; only matching evidence can authorize a
  fresh relation-first run from the untouched graph control.
- **Alternatives considered:** scale a model whose smaller control still
  regresses; continue the rejected schedule to 8,192; increase the drift prior;
  permanently delete node residuals; freeze one axis; relax guardrails; use
  GQA, MLA, MoE, or longer token history to hide a short-token shortcut.
- **Consequences:** Relation-first training is a diagnostic stage, not a claim
  that all motion is constant velocity. If it qualifies, a later same-capacity
  experiment may restore node acceleration only behind a causal, zero-default,
  observation-derived evidence gate. A real unmodelled force remains
  representable when evidence opens that gate. Focused implementation tests
  pass, and the complete suite passes (`736 passed, 6 skipped`); no new
  accuracy result exists yet.

## ADR-112 — Reject constant-rate drift before any capacity growth

- **Date:** 2026-08-12
- **Status:** accepted; rejected run stopped, same-capacity schedule control authorized
- **Context:** Late matched training windows for the drift-regularized
  width-128/four-block residual improved many pooled mature horizons, but the
  first authoritative trained selector disagrees.  At step 512, score worsens
  `0.3213162 -> 0.3332533` with 105 guardrail failures.  Familiar
  `reference_pairs` current x rises `0.242694 -> 0.732948 m` and every x
  horizon regresses.  Current/short pooled position, coverage, precision, and
  identity also worsen.  A strict artifact audit proves complete finite
  attention-only optimization and exact inherited/protected state, so this is
  generalization failure rather than corrupt state, missing support, or a dead
  module.
- **Decision:** Stop the constant-rate trajectory at its durable selector and
  never promote, resume, or grow from its learned attention weights.  Keep the
  scale gate closed.  Run the pre-authorized same-capacity schedule control
  weights-only from the untouched graph checkpoint with the same peak rate,
  objectives, data, and selectors; use 384 absolute warmup updates, 8,192
  fixed cosine-decay updates, and minimum scale 0.1.  Preserve the rejected
  run as evidence and require the same plateau/generalization gates before
  depth-six is eligible.
- **Alternatives considered:** scale depth or width immediately; continue the
  rejected constant-rate weights; relax guardrails because long horizons and
  collision F1 improved; add GQA/MLA/MoE/FlashAttention; initialize the
  schedule run from step 512; infer success from heterogeneous training
  windows.
- **Consequences:** The next comparison changes only optimization schedule and
  reduces early cumulative update magnitude without changing model capacity or
  physical contracts.  A 384-update warmup is about 4.7% of the 8,192-update
  minimum and reaches peak rate before selector 512.  If it also fails, the
  evidence points to objective/representation context rather than insufficient
  parameter count; scaling remains scientifically unjustified.  The successor's
  complete updates 8--64 audit is operationally clean and exactly balanced.
  On identical draws it is effectively tied with the constant-rate predecessor:
  current position/lifecycle/identity improve slightly while current velocity
  and four longer position horizons worsen slightly.  This confirms schedule
  isolation without supplying trained-selector evidence, so the decision gate
  remains unchanged.  At durable step 128, exact tensor/optimizer/protected-
  state audit passes, while matched updates 72--128 are worse on current and
  every pooled position horizon, mainly x, with mixed velocity/y behavior and
  adverse lifecycle/event/identity slices.  Because this early warmup boundary
  has deliberately accumulated much less update magnitude than the constant
  control, continue to the predeclared selector rather than rejecting, retuning,
  or scaling from the training window.  Updates 136--192 preserve that
  conclusion: position remains adverse on identical draws, while velocity,
  identity, uncertainty and gradient conditioning improve.  Linear warmup has
  accumulated only 25.13% of the constant schedule's scalar rate through step
  192, so neither matched lag nor heterogeneous adjacent-window movement is a
  substitute for selector 512.  By step 256 the matched current-position gap
  narrows to `+0.005475 m` and deterministic emitted y bias falls to
  `0.128556 m/s²` versus constant-rate `0.195037`, while x and every position
  horizon remain adverse and emitted variation remains small.  Structural
  audit passes completely.  This is evidence that the schedule changes the
  intended failure mode, but not that it has generalized; continue unchanged.
  Updates 264--320 subsequently narrow exact current/short position to a near
  tie and improve velocity, identity, event F1 and coverage90, while long
  position and several calibration/lifecycle slices remain adverse. This is
  the expected evidence pattern for continuing the predeclared schedule to its
  fixed selector, not for early promotion or capacity growth.
  Warmup completion at step 384 then improves matched current/0.10-second
  position and lifecycle with clean checkpoint scope, while mature position
  horizons remain adverse. This confirms optimization recovery is real but
  incomplete and leaves the selector decision unchanged.
  The first complete cosine-phase 392--448 window remains finite and balanced,
  contains one `0.126551`-retention hard-event spike without recurrence, and
  improves matched identity/lifecycle/event slices. Current position, velocity,
  coverage90, and nearly every position horizon are still slightly adverse,
  so stable optimization is not being conflated with broad convergence and the
  selector/capacity gates remain unchanged.

## ADR-111 — Diagnose functional-prior conflict before changing its weight

- **Date:** 2026-08-12
- **Status:** accepted and implemented as read-only evidence
- **Context:** The drift-prior successor remains finite and scope-clean, but y
  residual drift and long-horizon/velocity behavior wobble across complete
  matched windows. A growing residual under a positive penalty could mean a
  sign bug, insufficient weight, physical-task conflict, or optimizer momentum;
  scalar losses and total norms cannot distinguish them.
- **Decision:** On one deterministic causal graph, reconstruct the raw physical
  task by subtracting configured node priors, differentiate task, unit drift,
  and configured total objectives, and report norms plus task/total-versus-
  drift cosine over the full attention module and typed node decoder. Treat
  missing gradients as unused and zero-norm cosine as undefined. Sample more
  than one balanced draw and keep fixed selectors authoritative.
- **Alternatives considered:** assume the loss sign is wrong; increase drift
  weight mid-run; project conflicting task gradients; inspect decoder weights
  only; infer conflict from emitted acceleration; scale capacity.
- **Consequences:** Exact step-256 draws 254 and 255 show opposite directions.
  Node-decoder task/drift cosine is `-0.877315` then `+0.219746`; after all
  configured priors, total/drift cosine is `-0.666858` then `+0.413340`.
  There is no fixed sign defect: some physical draws reward residual drift and
  others restore it. This supports a decaying-rate same-capacity successor if
  selector 512 rejects, but does not authorize changing the immutable live run.
  Focused tests pass (`4 passed`) and the complete repository suite passes
  (`734 passed, 6 skipped`).
  The same test at step 384 reproduces configured-total/drift node-decoder
  cosine `-0.292264/+0.945411` on adjacent balanced draws 382/383. The
  diagnosis is persistent across checkpoint age rather than a step-256
  coincidence; the fixed selector still governs any schedule decision.

## ADR-110 — Make schedule repair exact and evidence-gated before scaling

- **Date:** 2026-08-12
- **Status:** implemented as opt-in infrastructure; successor not authorized
- **Context:** The active drift-prior campaign is finite and supported, but its
  early matched windows trade short-axis improvements against longer-horizon
  regressions while nearly constant y residual activity persists. Modern dense
  training practice commonly uses warmup and cosine decay, but changing the
  live constant rate would destroy the controlled trajectory. Tying cosine to
  mutable `training.steps` would also rewrite the future schedule whenever the
  convergence supervisor extends a run.
- **Decision:** Support only explicit `constant` and `warmup_cosine` causal
  schedules. Compute rate from zero-based causal optimizer update index,
  excluding measurement pretraining. Configure fixed warmup and cosine-decay
  durations plus a minimum scale; hold the floor after decay. Treat these as
  exact-resume semantics, normalize missing historical fields to `constant`,
  and require weights-only initialization for any change. Preserve the active
  specification-1.39 process unchanged until fixed selector 512.
- **Alternatives considered:** mutate the live rate; infer decay from total
  steps; use a stateful opaque scheduler; scale depth/width now; reject from
  sampled training rows; treat the schedule smoke as accuracy evidence.
- **Consequences:** A convergence extension cannot retroactively reshape the
  learning-rate curve, and no scheduler state can drift from the absolute
  optimizer step. Unit/config/checkpoint tests pass, and a real two-update CPU
  smoke plus exact one-update resume logs the analytically expected
  `0.0002 -> 0.00011` transition. The smoke is `last_unvalidated`; only a new
  complete fixed-manifest campaign can establish an accuracy benefit. The
  complete repository suite passes with `732 passed, 6 skipped`; the skips are
  restricted-process MPS availability checks, not training failures.

## ADR-109 — Reject complexity-only step 1024 before scaling

- **Date:** 2026-08-12
- **Status:** accepted; rejected campaign stopped, drift successor authorized
- **Context:** The specification-1.36 complexity-only attention campaign
  reached a durable, strictly audited step-1024 checkpoint with healthy
  optimizer, support, finite-state, provenance, and resource evidence.  Its
  selector score changes only `0.3213162196 -> 0.3212919367`, but 111 broad
  guardrails fail.  Selected current position, x, coverage, precision, and the
  two shortest horizons regress.  `reference_pairs` current position rises
  `0.212965 -> 0.383810 m`, current x `0.242694 -> 0.573947 m`, and every x
  forecast horizon worsens.  Complete training windows remain operationally
  healthy, so more of the same optimization would extend behavioral overfit
  rather than repair an infrastructure failure.
- **Decision:** Stop the immutable specification-1.36 trainer and supervisor
  at step 1024.  Do not promote, resume, or use its learned weights for growth.
  Launch the specification-1.39 successor weights-only from the untouched
  protected graph control with `attention_node_complexity=1.0` and
  `attention_node_drift=0.08`.  Preserve the declared convergence budget and
  broad selectors.  Keep the capacity gate closed until the successor passes
  repeated fixed validation plus disjoint test/OOD non-regression.
- **Alternatives considered:** continue complexity-only to 8,192 because it is
  numerically healthy; accept the microscopically improved scalar score;
  relax scenario guardrails; initialize the drift run from step 1024; add
  depth, width, history, MLA, GQA, or MoE immediately.
- **Consequences:** This separates healthy optimization from useful
  generalization.  The next run tests the narrow measured defect—scene-wide
  emitted acceleration—without discarding useful context-sensitive residual
  capacity.  Modern Transformer efficiency mechanisms remain future compute
  options, not accuracy repairs for the short typed-token set.  Any later
  capacity rung must preserve exact smaller-model function where supported,
  increase balanced data exposure with parameters, and beat the accepted
  smaller control on fixed RGB-only validation/test/OOD evidence.

## ADR-108 — Penalize context-invariant node drift, not useful variation

- **Date:** 2026-08-12
- **Status:** accepted and implemented; protected-control smoke passed,
  sustained qualification pending
- **Context:** ADR-107 correctly measured a large functional node residual, but
  total mean-squared activity does not distinguish a harmful scene-wide bias
  from useful object-, relation-, or event-conditioned variation. Four
  deterministic balanced draws at the rejected step-512 checkpoint give
  tightly stable activity `0.042484--0.042695 (m/s²)²` and equal-gradient
  weights `0.078292--0.078442`. A richer trace over `10,182` active-object
  invocations finds mean acceleration
  `[-0.024866, 0.356690, 0.006175] m/s²` but standard deviation only
  `[0.000736, 0.001865, 0.000746]`. Squared mean drift is `0.04266783` versus
  total activity `0.04266899`: more than 99.997% is context-invariant drift.
- **Decision:** Decompose emitted node activity per axis into squared mean
  drift and residual variation, pooled from sums, squared sums, and active
  support over the complete causal draw. Expose all three diagnostics. Add
  opt-in `attention_node_drift` as a distinct exact-weight objective and use
  `0.08` for the prospective successor, matching the unit decoder-complexity
  restoring-gradient scale. Retain total activity as a diagnostic/optional
  objective but do not configure it for this failure.
- **Alternatives considered:** penalize all activity at `0.08`; use unit
  activity weight; freeze/zero node outputs; penalize y specifically; treat
  the nearly constant output as useful learned physics; scale the model.
- **Consequences:** Balanced positive/negative residual variation can remain
  unpenalized while a scene-wide learned force pays a soft cost. All axes are
  treated identically, and a genuine unmodelled constant force can still be
  learned when held-out evidence outweighs the prior. Historical configs omit
  the exact drift key and remain unchanged. Focused drift/version tests pass
  (`8 passed`), the full suite passes (`719 passed, 6 skipped`), Ruff passes,
  and the 8,192-update `drift=0.08` dry-run resolves correctly. No accuracy or
  convergence promotion is claimed before a fresh protected-control campaign.
  A two-update CPU smoke plus exact resume subsequently exercised a genuinely
  nonzero drift objective, all 13 causal terms, strict attention-only optimizer
  ownership, inherited/protected equality, and finite checkpoint state. Its
  deliberately reduced eight-episode validation is `last_unvalidated`; this
  closes the implementation/wiring risk but not the sustained accuracy gate.
  The sustained successor's strict step-128 audit later passes with all 48
  attention tensors changed, all inherited/protected state exact, and complete
  finite Adam state.  On the same deterministic draw, RMS emitted node
  acceleration is `0.106243 m/s^2` and mean y acceleration is
  `0.171365 m/s^2`, versus `0.206605/0.356690 m/s^2` at the rejected
  predecessor's step 512.  The prior therefore controls its target, but the
  complete matched updates 72--128 position window is broadly worse and more
  than 99.95% of the smaller activity remains drift.  This mixed evidence
  requires fixed selector 512; it does not authorize a mid-run schedule
  change, capacity increase, acceptance, or rejection.  The emitted-value
  calibration is restricted to gradient-enabled causal attention calls and
  asserts exact call/object-count agreement with the differentiable activity
  records; prepared no-gradient rollouts are not mixed into this statistic.

## ADR-107 — Regularize emitted node activity before increasing capacity

- **Date:** 2026-08-12
- **Status:** accepted as diagnostic infrastructure; objective refined by ADR-108
- **Context:** The immutable specification-1.36 residual-parsimony campaign's
  complete step-512 checkpoint is finite, scope-clean, and causally trained,
  yet its fixed 32-episode selector is rejected. Score improves slightly
  (`0.3213162 -> 0.3177418`) and 0.5--1.0-second pooled horizons improve, but
  current position, velocity, target coverage, precision, and the two shortest
  horizons regress. `reference_pairs` current position doubles
  (`0.212965 -> 0.429954 m`). Exact same-manifest ablations show that halving
  both output decoders is harmful, removing relation force rows is decisively
  harmful, restoring only node-y to zero improves every forecast horizon but
  still fails 97 guardrails, and restoring the complete node decoder to zero
  is strongest at score `0.297330` but still fails 72 guardrails. Relation
  outputs therefore contain useful interaction learning; functional node
  acceleration is the localized broad error, and decoder parameter energy is
  an insufficient proxy for it.
- **Decision:** Add an opt-in `attention_node_activity` objective equal to the
  active-object-weighted mean squared bounded node acceleration emitted across
  every attention invocation in the current causal data draw. Pool numerator
  and support before deriving x/y/z diagnostics and their equal mean. Store the
  differentiable records only transiently; they are not buffers, checkpoint
  state, or runtime belief. Require the exact loss-weight key, so legacy and
  omitted configurations contribute zero and inference remains unchanged.
- **Alternatives considered:** hard-zero or freeze the node decoder; penalize
  only y; remove the useful relation/force branch; globally shrink all typed
  residuals; scale depth/width despite a known small-rung regression; relax
  selector guardrails.
- **Consequences:** The architecture retains general acceleration capacity on
  every axis and can pay the soft cost when held-out evidence supports
  non-inertial motion. Focused tests verify active-mask normalization, exact
  per-axis values, differentiability, reset behavior, and opt-in weighting;
  the complete suite passes (`718 passed, 6 skipped`). This implementation is
  not an accuracy promotion. The immutable 1.36 campaign continues unchanged
  toward selector 1024; a 1.38 successor is justified only if that selector
  does not repair broad guardrails, and it must start from the protected graph
  control rather than the rejected step-512 candidate. A deterministic
  balanced causal-draw diagnostic at the rejected step-512 checkpoint measures
  node activity `0.042669 (m/s²)²`, split x/y/z as
  `0.000618/0.127347/0.000042`, and RMS acceleration `0.206565 m/s²`. Unit
  activity and complexity restoring-gradient norms are `0.673351/0.052798`,
  giving the `0.078411` equal-gradient value. Unit weight is rejected as an
  unjustified 12.75-fold increase over the existing prior's gradient scale.
  ADR-108 subsequently proves that nearly all activity is squared mean drift
  and selects the more context-preserving drift objective at weight `0.08`.

## ADR-106 — Require non-vacuous protected-checkpoint evidence

- **Date:** 2026-08-12
- **Status:** accepted and implemented; live training semantics unchanged
- **Context:** The first step-256 attention audit omitted `--protected`
  arguments and reported `protected_checkpoints_exactly_initial: true` because
  every member of an empty set passed. The model/optimizer checks were valid,
  but the protection claim was unchecked and could be mistaken for evidence.
- **Decision:** Record protected-checkpoint count, represent an empty check as
  `null`, and expose a qualification gate that fails when no protected paths
  were supplied. Retain file and model-state hashes for every supplied path.
- **Alternatives considered:** rely on command discipline; auto-discover
  sibling filenames; keep vacuous truth and infer it from empty hash maps.
- **Consequences:** Qualification commands are explicit and fail closed,
  while exploratory audits may omit protection without making a false claim.
  This changes only the offline auditor and specification version; the active
  immutable 1.36 trainer and checkpoint format are unchanged.
- **Evidence:** Focused tests cover both a required nonempty protected set and
  required empty-set failure. The corrected step-256 audit checks two paths;
  both model states equal the initializer and their hashes match step 128.

## ADR-105 — Budget recursive typed-output gradients across the optimizer draw

- **Date:** 2026-08-11
- **Status:** accepted, implemented, and matched-replay qualified
- **Context:** The specification-1.31 campaign safely rejected attempted update
  988 at `0.0971759` complete interaction retention. The normal-force decoder
  row reached `10.9076` and shared block gradients reached `5.01609` even
  though every invocation obeyed its local output cap. Across 144 recursive
  calls, force/impulse applied output norms accumulated to `0.219855/0.115811`
  around nominal `0.1` caps. Decoder-row caps cannot repair gradients that have
  already entered the shared stack.
- **Decision:** Interpret each semantic output cap as an aggregate per-draw L2
  budget. With `K` registered invocations, apply `cap / sqrt(K)` locally. Reset
  registration counts per optimizer draw and retain the existing aggregate
  raw/applied/minimum-coefficient diagnostics and downstream row/global gates.
- **Alternatives considered:** remove or lower the 10% retention gate; accept
  the finite globally normalized update; lower all learning rates; enlarge the
  model; add only another decoder-row cap.
- **Consequences:** Single-invocation behavior and all forward outputs are
  unchanged, while repeated semantic gradients have a true total budget before
  reaching shared attention. The failed campaign remains non-promotable and
  scaling remains blocked pending a step-896 matched replay, fresh selectors,
  and plateau.
- **Evidence:** Exact model/Adam/RNG/sampler replay from durable step 896 used
  the same attempted-step-988 seeds, scenarios, 284 causal trajectories, and
  all 13 objective terms. Raw/post-row norm fell `15.1704/10.2906 ->
  4.58029/1.54333`; complete retention rose `0.0971759 -> 0.647948`; the
  largest shared gradient fell `5.01609 -> 0.225929`; and all four aggregate
  semantic applied norms stayed below `0.1`. The harness stopped before Adam.

## ADR-104 — Pool comparable training-trend windows from sufficient statistics

- **Date:** 2026-08-11
- **Status:** accepted and implemented; selectors remain authoritative
- **Context:** The whole-run dynamics audit exposed distributions, but
  consecutive per-axis/horizon comparisons were assembled ad hoc. Averaging
  per-batch RMSE or rates would bias windows with unequal physical support, and
  partial tails could look spuriously better or worse than complete windows.
- **Decision:** Emit non-overlapping configurable logged-block windows from the
  existing read-only auditor. Pool SSE/counts, event counts, association counts,
  and coverage counts before deriving physical metrics; expose position axes,
  position/velocity horizons, lifecycle, identity, uncertainty, collision F1,
  parameter observability, causal support, gradient retention, and memory.
  Mark every tail window complete or incomplete. For deterministic predecessor
  investigations, align candidate/reference rows by step and require exact
  seed, scenario, draw, frame-window, and rollout-anchor schedules before
  emitting independently pooled summaries and signed deltas.
- **Alternatives considered:** continue one-off `jq` calculations; average
  already-derived per-row metrics; run extra validation at every log line;
  treat noisy training windows as selector evidence.
- **Consequences:** Collapse and axis/horizon trade-offs can be monitored
  reproducibly without touching the live optimizer or paying validation cost.
  These heterogeneous windows remain diagnostic only and cannot promote,
  reject, scale, or declare convergence without fixed-manifest evidence.
- **Evidence:** The matched-reference mode has focused pass/fail coverage and
  rejects a seed mismatch. On the live specification-1.35 campaign, all 11
  cadence rows at steps 192--272 align exactly with the predecessor; the pooled
  report exposes simultaneous long-position neutrality and broad velocity,
  collision, lifecycle, and current-state regressions that one-row comparisons
  obscured.

## ADR-103 — Grow attention depth with appended exact-identity blocks

- **Date:** 2026-08-11
- **Status:** accepted and implemented; no capacity rung promoted
- **Context:** ADR-102 correctly rejected random partial depth handoffs, but
  restarting every deeper candidate from the graph-only control would discard
  a qualified smaller attention function and confound capacity with relearning.
  Orpheus uses pre-normalized residual blocks, so appended blocks have a narrow
  exact identity parameterization.
- **Decision:** Permit only contiguous appended attention blocks. Copy all
  inherited tensors strictly; zero each new attention output weight/bias and
  SwiGLU output weight; retain ordinary finite internal initialization; and
  record the grown block indices. Require identical resolved runtime model
  semantics except for increased depth, catching shape-invisible head/dropout
  changes. Reject width changes, block holes/reordering, or any missing
  inherited/non-block tensor before destination mutation.
- **Alternatives considered:** keep every depth rung graph-initialized; copy
  random appended blocks; reset learned typed decoders; interpolate tensors;
  allow arbitrary state-dict partial loading.
- **Consequences:** A future depth-six rung can start bitwise function-equal to
  a qualified depth-four incumbent and learn new capacity gradually. Width 192
  still has no function-preserving handoff. This implementation does not
  authorize scaling before the current rung passes its fixed selectors and
  plateau/generalization gates.

## ADR-102 — Reject partial learned-module growth before tensor loading

- **Date:** 2026-08-11
- **Status:** accepted and implemented; larger rungs remain evidence-gated
- **Context:** The generic weight-only loader allowed all missing keys below
  the new attention prefix. That is correct when a graph-only source contains
  no attention module, but it also allowed a trained four-block source to seed
  a six-block destination with two random new blocks. The random blocks alter
  the features consumed by learned typed decoders before optimization, so the
  handoff is neither zero-output nor function-preserving. PyTorch's ordinary
  load path could also copy compatible tensors before reporting a later
  incompatibility.
- **Decision:** Preflight source/destination keys and shapes before any copy.
  An allowed missing prefix is all-or-none unless it satisfies ADR-103's exact
  appended-depth transform. Reject unexpected, disallowed-missing, unsafe
  partial-prefix, and shape-incompatible handoffs without mutating the
  destination. Unsupported attention growth initializes from the structured
  graph control; the accepted smaller model is its fixed evaluation reference.
- **Alternatives considered:** accept random added blocks; silently reset the
  learned decoder; partially embed wider tensors; treat `strict=False` as
  sufficient; implement an untested identity-growth transform during the live
  campaign.
- **Consequences:** The active run is unchanged because its source had no
  attention prefix. Future depth/width experiments cannot accidentally report
  random architecture drift as training or generalization. The smallest next
  capacity step is depth six, followed by width 192, after the current rung
  qualifies.

## ADR-101 — Isolate impulse jumps and fail before shared-stage starvation

- **Date:** 2026-08-11
- **Status:** accepted and implemented; fresh qualification pending
- **Context:** The fresh node/collision/force-output-isolated campaign reaches
  update 200 with complete physical support but raw total/interaction norm
  `857.1579`. Impulse multiplier/additive decoder rows contribute
  `830.3828/210.3096`, shared projection/block gradients reach `6.2401`, and
  the later complete interaction clip retains only `0.001167`. The existing
  auditor calls this a severe warning but still returns `pass`; the finite
  global norm therefore hides near-total shared-update starvation.
- **Decision:** Treat multiplier/additive impulse proposals as one explicit
  semantic group with a `0.1` per-invocation output-gradient cap and `1.0`
  accumulated decoder-row cap in the active pilot. Add protocol-bound optional
  minimum complete-interaction retention, set it to `0.1`, clear gradients and
  fail before Adam on violation, and make the offline auditor fail the same
  post-isolation condition. Preserve legacy missing controls as `null`.
- **Alternatives considered:** scale width/depth; lower the global learning
  rate; accept finite global clipping as sufficient; cap the whole relation
  tensor and mix unrelated semantics; silently discard the bad batch; tune
  against sampled forecast metrics.
- **Consequences:** Forward dynamics, parameter count, tensor shapes, and
  inference are unchanged. A matched non-promotable step-128--200 replay
  reduces raw norm to `7.4410`, shared maximum to `0.05334`, and raises
  complete-stage retention to `0.64704`, but earlier bounded updates change
  weights so it is not a forward-exact one-update ablation. A fresh
  weights-only campaign and fixed selectors remain mandatory before any
  capacity scaling.

## ADR-100 — Scale predictive abstractions with matched compute/data evidence

- **Date:** 2026-08-11
- **Status:** accepted as the next gated ladder; current rung still qualifying
- **Context:** The typed-output repair passes the former step-64 optimizer
  failure on identical data: raw gradient is `2.14592` rather than `21.5377`,
  force-row norm is `1.75123` rather than `21.4665`, maximum non-decoder shared
  norm is `0.00540` rather than `0.04242`, and sampled 1-second error is not
  worse. The current model already implements the relevant modern dense-
  Transformer mechanisms over at most 22 structured tokens. It does not yet
  have a trained fixed selector or plateau, so more capacity is not yet a
  defensible remedy.
- **Decision:** Preserve the complete 3.00M control trajectory as the data-only
  curve. After it qualifies, vary one axis at a time: 3.53M depth six, 4.34M
  width 192, bounded timestamped belief/innovation history, then an 8.31M
  width-256/depth-six single-CUDA rung. Give every candidate at least the
  control's continuously varied balanced draws and increase draws with
  parameters to a complete selector interval. Require disjoint RGB-only
  validation/test/OOD non-regression. Reserve RoPE/relative positions for real
  timestamped history; defer FlashAttention/GQA/MLA/MoE/sharding until measured
  token, memory, or throughput evidence calls for them. Treat maximal-update
  parameterization as a separate matched experiment. Later dense JEPA-style
  RGB pretraining must feed typed proposals into authoritative `WorldBelief`.
- **Alternatives considered:** scale width/depth immediately; replace the
  explicit belief with an autoregressive video transformer; add every modern
  LLM optimization at once; reuse the same data budget for a larger model;
  infer generalization from training loss or video reconstruction quality.
- **Consequences:** The Mac run remains slow but scientifically useful: it
  establishes optimizer health, a real small-rung learning curve, and whether
  the next limitation is data, capacity, or missing temporal context. Cloud
  compute can later accelerate the same contracts without making the local
  experiment disposable. Scaling remains blocked by evidence gates, not by a
  speculative architectural preference.

## ADR-099 — Isolate the reproduced force-head spike before scaling

- **Date:** 2026-08-10
- **Status:** accepted and implemented; fresh qualification pending
- **Context:** Exact continuation from the collision-isolated step-256
  checkpoint reproduces steps 264/272/280 with zero difference across all
  shared deterministic telemetry. At step 280, normal/tangent force decoder
  rows have raw norms `17.3894/3.2159` and joint norm `17.6842` inside the
  complete `17.7050` interaction norm. The collision row is only `0.2355`.
  The relation-decoder weight norm is `17.6189`; remaining interaction
  gradient outside the two force rows is approximately `0.8573`.
- **Decision:** Add one optional joint normal/tangent-force row cap before the
  existing interaction and global caps, configured to `1.0` for stage A.
  Preserve read-only semantic row telemetry and reconstruct the true raw
  hierarchy. Bind the new cap into resume/selector protocol semantics and the
  offline auditor. Correct checkpoint specification metadata from stale 1.25
  to current specification 1.27. Relaunch weights-only from the protected
  graph control after gates; do not resume or count either flawed campaign.
- **Alternatives considered:** scale the Transformer immediately; lower the
  whole learning rate; cap all decoder rows; widen the interaction cap; treat
  the finite step as healthy noise; add GQA/MoE/FlashAttention for a 22-token
  set.
- **Consequences:** On the reproduced raw gradient, the force cap would reduce
  the pre-interaction-cap norm to approximately `1.3172`, so the existing 1.0
  interaction cap retains about `0.7592` rather than `0.05648`. Forward
  physics and capacity are unchanged. Only a fresh fixed-selector learning
  curve can qualify the repair or unlock width/depth/history scaling.
  The fresh force-isolated campaign subsequently passes the first historical
  step-152 boundary with raw interaction norm `2.46615` and retained
  interaction-stage coefficient `0.48940`, versus `28.1387/0.03554` before
  row isolation and `7.11114/0.14308` after collision-only isolation. This is
  boundary-specific optimizer evidence; step 280 and fixed selectors remain
  mandatory before qualifying the decision for accuracy or scaling.

## ADR-098 — Localize the complete attention gradient before another repair

- **Date:** 2026-08-10
- **Status:** accepted and instrumented; exact replay pending
- **Context:** Collision-row isolation improves the former step-152 failure,
  but repaired update 280 still has raw interaction norm `17.7050` and retains
  `0.05648`. Its collision row is only `0.23553` and is not clipped, proving
  that the remaining recurrence lies elsewhere in the shared attention model.
- **Decision:** Stop at exact durable step 256. Record pre-mutation raw norms
  for all named attention parameters and semantic decoder rows, finite-check
  them, and replay steps 257--280 with optimizer/RNG/data continuity. Repair
  only the reproduced dominant path, then restart the qualification from the
  protected graph control.
- **Alternatives considered:** continue because the update is finite; assume
  the collision row remains causal; cap every decoder row; lower the complete
  learning rate; widen the interaction cap; diagnose from checkpoint moments
  after the failed update was not durably saved.
- **Consequences:** The stopped run cannot count toward convergence, but its
  clean step-256 state supports exact localization. Telemetry grows by stable
  scalar fields without changing forward computation or gradients. A later
  targeted repair must still pass the former boundaries and fixed selectors.

## ADR-097 — Require repaired checkpoint and selector evidence before scaling

- **Date:** 2026-08-10
- **Status:** accepted; step-256 integrity passed, campaign continuing
- **Context:** The collision-isolated campaign reached durable update 128 with
  128 applied balanced updates, no numerical/support/resource failure, exact
  inherited-weight isolation, all 48 attention tensors live, and attention-
  only Adam state. This is substantially stronger than a smoke but still
  precedes the former update-152/280 anomaly and the first trained fixed
  selector at update 512. Sampled scenario loss is heterogeneous and cannot
  establish convergence.
- **Decision:** Continue the unchanged 3.00M-parameter stage-A campaign through
  the former anomaly boundaries, repeated complete selectors, and a declared
  plateau. Do not add stage-B history, increase width/depth, or introduce MoE/
  long-context optimizations until the current dense typed residual improves
  or safely matches all protected multi-horizon guardrails. Judge convergence
  on fixed selectors and later disjoint/OOD manifests, not sampled train loss.
- **Alternatives considered:** scale immediately because state is finite; stop
  at checkpoint 128; tune against individual sampled losses; add history and
  capacity together; replace dense attention with MoE on the 22-token set.
- **Consequences:** The current run remains long enough to reveal a real
  learning curve on Mac hardware and produces a defensible scale/no-scale
  decision. A successful stage A unlocks bounded timestamped history and then
  one-axis-at-a-time dense capacity growth with commensurate data; a regression
  preserves the exact smaller control and triggers diagnosis before scaling.
  The first former periodic failure at update 152 now retains `0.14308` rather
  than `0.03554` of the interaction-stage gradient and passes the auditor; the
  update-280 recurrence and fixed selectors remain required.
  Durable step 256 also passes exact optimizer/scope/hash/resource integrity.
  One sampled trusted-identity spike at step 248 is retained as a selector
  warning rather than a reason to tune against a single stochastic outcome;
  its aggregate rate through step 256 is `0.975%` and the next block is zero.

## ADR-096 — Isolate collision-logit gradients before the interaction group

- **Date:** 2026-08-10
- **Status:** accepted and implemented; repaired campaign pending
- **Context:** Scene pre-projection normalization removed the matched update-64
  scale failure, but severe finite gradients appeared at updates 152 and 280,
  exactly 128 updates apart on deterministic frames 7--11 contact-heavy
  batches. Their retained interaction coefficients were `0.03554/0.01888`.
  The complete interaction cap kept state finite but reduced every unrelated
  attention gradient by the same factor. Step-256 Adam moments localize the
  dominant variance to relation-decoder collision row 1 (`0.03050` RMS), not
  the normalized scene or entity/relation projections.
- **Decision:** Stop the run at its clean durable step-256 boundary. Add an
  optional collision-row norm cap before the existing interaction and global
  caps; set it to `1.0` for the pilot. Reconstruct and report the true raw
  hierarchy algebraically, expose row/stage/total coefficients, bind the cap
  into protocol compatibility, and make the auditor inspect it. Restart
  weights-only from the protected graph control.
- **Alternatives considered:** continue because the global cap is finite;
  normalize all entity/relation inputs; lower the complete learning rate;
  reduce collision loss globally; increase the interaction cap; hide the row
  spike in post-clip telemetry; resume the step-256 optimizer state.
- **Consequences:** Rare event supervision can no longer monopolize the shared
  interaction update, while force, impulse, uncertainty, labels, forward
  predictions, and collision physics remain unchanged. The new run must still
  prove periodic-batch conditioning, broad selector accuracy, and convergence;
  the repair itself earns no promotion.

## ADR-095 — Normalize mixed-unit scene features before projection

- **Date:** 2026-08-10
- **Status:** accepted and implemented; corrected relaunch pending
- **Context:** The repaired scene token became live, but the first sustained
  run's sampled interaction norm grew from `0.2631` at update 8 to `45.3456`
  at update 64. On the exact update-64 seeds/window, the old control was only
  `1.3231`; losses and physical event counts were ordinary and closely
  matched. The scene vector mixed pixel-space intrinsics with latent values,
  log variances, gravity, camera motion, and a homogeneous transform before
  its first linear layer. RMS pre-normalization inside each Transformer block
  occurs after that projection and cannot condition its weight gradient.
- **Decision:** Stop the campaign rather than normalize the failure away with
  a larger clip. Apply fixed, non-affine RMS normalization to the complete
  scene feature vector immediately before `scene_projection`. Keep analytic
  physical quantities unchanged in `WorldBelief` and structured dynamics;
  only condition the learned residual token. Restart weights-only from the
  protected graph control after gates and a short smoke.
- **Alternatives considered:** continue because global clipping kept updates
  finite; increase the clip; lower learning rate without repairing scale;
  divide intrinsics by a hard-coded image size; add a learnable affine input
  norm; normalize all entity/relation fields without evidence they are the
  failing path.
- **Consequences:** Projection input has fixed RMS independently of camera
  pixel scale, there is no new unbounded affine factor, and zero-output graph
  identity remains exact. The stopped live-scene run is diagnostic only and
  does not count toward the 8,192-update convergence budget.

## ADR-094 — Scale relational capacity as a zero-initialized typed residual

- **Date:** 2026-08-10
- **Status:** accepted for stage-A qualification; accuracy promotion pending
- **Context:** Protocol 20 is a healthy, de-noised y-only plateau and no longer
  has a known numerical or optimizer defect. The original Transformer shows
  that multi-head scaled dot-product attention shortens content-dependent
  interaction paths; later foundation models commonly use pre-normalization,
  RMSNorm, gated feed-forwards, and relative/rotary temporal encoding. Orpheus
  entities are an unordered persistent-ID set, not language positions, and the
  Mac must establish a causal scaling curve before larger compute is useful.
- **Decision:** Add four RMS-pre-normalized width-128/four-head blocks with
  SwiGLU width 512 over derived scene/entity/relation tokens. Do not use learned
  slot positions or RoPE on current object order. Decode a bounded residual
  into the existing graph/event/uncertainty contract, initialize its heads at
  zero, strictly transfer all inherited weights, and optimize the 1.099M new
  parameters alone first. Add timestamp-relative history only as a separately
  qualified next stage.
- **Alternatives considered:** replace `WorldBelief` and analytic dynamics with
  a video transformer; train the larger model from scratch; add slot-position
  embeddings; use a decoder-only causal language layout; unfreeze the whole
  runtime immediately; add temporal history in the same first experiment.
- **Consequences:** The pilot is exactly the accepted graph at step zero,
  remains permutation-equivariant, fits at 3.00M total parameters, and can be
  rejected without damaging inherited weights. Attention costs about 1.66x a
  representative CPU prediction step and roughly 16--19 seconds per full
  fixed-validation episode on this Mac. The declared run therefore validates
  every 512 updates while checkpointing every 128. Stage-A accuracy, bounded
  history, parameter-matched control, disjoint test/OOD, and CUDA scaling all
  remain pending.

## ADR-093 — Reject only numerically tied fast-ROI component ownership

- **Date:** 2026-08-10
- **Status:** accepted and implemented under rollout protocol 14
- **Context:** Protocol 20 changed only learned-corrector y row 1 smoothly, yet
  some checkpoints toggled reference-pair x/identity behavior. Exact replay of
  seed `100024` showed that before association changed, a `0.0000765` predicted
  RGB change caused a `0.2807869` structured-centre jump. Two disconnected
  components had equal nearest supported distance (`0.0` versus `2.98e-8`
  ownership margin). A blanket `0.20` residual cap removed the jump but
  regressed the full public posterior and horizons by discarding legitimate
  recovery evidence.
- **Decision:** After growing the selected local RGB component, compare its
  nearest supported distance with the nearest supported pixel outside that
  component. If their margin is within scale-aware `32 * eps` equality
  tolerance, mark ownership ambiguous, preserve the predicted centre, and let
  global discovery recover. Retain the configured ordinary fast-path distance
  range. Expose finite ownership-margin and ambiguity diagnostics and advance
  rollout validation from protocol 13 to 14.
- **Alternatives considered:** tighten every ROI to `0.20`; add association
  hysteresis after the bad measurement; accept whichever component tensor
  ordering returns first; use simulator identity; scale the updater before
  repairing perception; treat the discontinuity as optimizer noise.
- **Consequences:** The failing step-64/512 replay becomes structurally and
  numerically stable. Paired 32-episode public evaluation improves joint/x/
  velocity accuracy, identity, collision F1, NLL, and four horizons with only
  sub-percent y/z/detection tradeoffs. The exact physical selector improves
  step 64 to `0.3213162`; step 512 remains worse at `0.3213287`, confirming a
  genuine y-only plateau. Touching objects already merged into one RGB
  component remain a documented observation-ownership limitation for the
  attention-era perception work.

## ADR-092 — Treat the first y-only gain as interim, not convergence

- **Date:** 2026-08-10
- **Status:** accepted monitoring decision after the first fixed selector
- **Context:** Protocol 20 completed 64 balanced y-only updates without a
  numerical, support, identity, lifecycle, uncertainty, optimizer, or resource
  failure. Its exact selector passed every guardrail and improved pooled score
  `0.3216427 -> 0.3215594` and current RMSE
  `0.2537443 -> 0.2532523 m`. The gain is nevertheless small: velocity worsened
  by `0.0004332 m/s`, coverage/precision fell by about `0.00025/0.00015`, and
  0.50--1.00-second RMSE worsened by only micrometres. Exact tensor inspection
  proves this is real y-row behavior rather than scope leakage.
- **Decision:** Retain step 64 as the internal guardrail-safe incumbent while
  continuing the unchanged mutable y-only trajectory through repeated fixed
  selectors. Do not call this convergence, a legacy-reference replacement, or
  permission to scale attention. Judge the direction from the complete
  learning curve and the existing plateau/generalization contracts.
- **Alternatives considered:** stop on the first lower pooled score; reject the
  candidate for sub-tolerance late-horizon changes; unfreeze x/z to accelerate
  progress; skip directly to a larger transformer.
- **Consequences:** The selector still protects a recoverable best checkpoint,
  while longer training can show whether the small short-horizon gain compounds,
  plateaus, or reverses. Any later regression leaves this numbered checkpoint
  intact and cannot silently reactivate the known x/z failure mode.
- **Step-128 evidence:** The next candidate regressed pooled score to
  `0.3216703` and was rejected. Its failures are confined to baseline coverage,
  identity, and 0.10-second x after an association threshold crossing; the
  training audit and y-row tensor isolation still pass. Continue to step 192
  because the prior y-only curve recovered after the same intermediate gate,
  while retaining step 64 as the immutable incumbent.
- **Step-192 evidence:** The candidate remained rejected at `0.3216706` with
  the same baseline failures. The y-row update magnitude fell by more than
  sixfold from the first segment, but finite row-local gradients and Adam
  moments persist and excluded tensors remain bitwise fixed. Treat this as the
  first repeated saturation evidence, not enough by itself to declare a
  plateau; continue the already-declared 512-update run with step 64 protected.
- **Step-256 evidence:** The candidate crossed back into the guardrail-clean
  association regime at score `0.3215611`, only `0.00000167` worse than step
  64, and was not accepted. Consecutive y-row deltas are smooth and row-local,
  so the discrete baseline identity changes are an association-threshold
  response rather than optimizer oscillation. Continue the declared run and
  retain step 64; do not promote metric equivalence as a new improvement.
- **Step-320 evidence:** The candidate returned to the rejected regime at
  `0.3216708` with the same three baseline failures. The y row and its moments
  still move smoothly and exclusively. Four consecutive 64-step candidates
  without acceptance are useful bounded-recovery saturation evidence but do
  not meet the sustained policy's 512-step spacing; complete the declared
  512-update run rather than claiming convergence early.
- **Step-384 evidence:** The candidate returned to the guardrail-clean regime
  at `0.3215634`, but remained `0.00000405` worse than step 64. All optimizer,
  support, memory, uncertainty, and y-row-isolation checks pass. Treat the
  repeated clean/rejected band changes as evidence of a discontinuous online
  association response to a smooth parameter trajectory, not useful monotonic
  convergence.
- **Step-448 evidence:** The candidate crossed back into the rejected regime
  at `0.3216787`, including a baseline x@100-ms failure even though exact
  tensor comparison proves only y row 1 changed. This is stronger evidence
  that the remaining limitation sits in trajectory-level association feedback.
  Complete step 512 for the declared endpoint; do not promote any late
  candidate unless the unchanged full selector passes.
- **Step-512 decision:** The endpoint was rejected at `0.3216317` by
  `reference_pairs` current-x and x@100-ms guardrails. The complete 512-update
  audit passes with all 64 balanced blocks applied, no skips/clips/failures,
  finite support/uncertainty, bounded memory, clean process exit, and exact
  y-row-only model/Adam state. Seven consecutive candidates after step 64 did
  not improve it. Protocol 20 is therefore a completed bounded-recovery
  plateau. Stop extending the same y-only optimizer direction; retain step 64
  and target the discontinuous association/trajectory feedback before adding
  attention capacity.

## ADR-091 — Continue correction recovery on the accepted y row only

- **Date:** 2026-08-10
- **Status:** accepted after three fixed validations and five row ablations
- **Context:** Mean-head-only protocol 19 remained numerically healthy through
  192 balanced updates, but exact scores plateaued/regressed at `0.3246722`,
  `0.3246772`, and `0.3249595`, with 15, 17, and 24 guardrail failures. The
  x row alone reproduced almost the complete y/z reference-pair trajectory
  regression because its current correction changed later association. The
  z row was slightly regressive. Y-only was guardrail-clean at steps 64 and
  192; step 192 improved score `0.3241755 -> 0.3216427`, current position
  `0.2559540 -> 0.2537443 m`, velocity `1.0966767 -> 1.0949210 m/s`, identity,
  and all five horizons.
- **Decision:** Stop the joint mean-row campaign at durable step 192. Promote
  its y-only composition as the corrected recovery incumbent, not yet as the
  legacy deployment replacement. Add schema-checked row composition and an
  `updater_mean_y` scope that exactly preserves excluded values and AdamW
  moments. Continue sustained optimization only on y with the unchanged full
  selector; x and z remain at the neutral mean-reset values.
- **Alternatives considered:** run the rejected joint rows to 512; promote the
  attractive x slice despite its downstream failures; combine y with rejected
  z; assume row-local output implies trajectory-local effects; proceed to the
  attention pilot before the corrected control reaches the legacy reference.
- **Consequences:** The next campaign has a validated improving start and
  cannot recreate the known x/z correction cascade through optimizer leakage.
  It must still demonstrate repeated plateau/generalization evidence and beat
  the legacy fixed reference before attention scaling.

## ADR-089 — Surface severe clipping; do not redesign from one bounded hard window

- **Date:** 2026-08-09
- **Status:** accepted monitoring policy; recurrence qualification pending
- **Context:** The first 63 protocol-18 updates were substantially more stable
  than protocol 17, but update 64 produced a raw norm of `30.3853`. Exact replay
  localized it to one baseline trajectory's recursive velocity gradient
  through normal/tangential pair-force outputs. Local interaction and global
  clipping bounded the applied update, and update 72 recovered immediately.
  The existing auditor checked finiteness but did not call out clip severity.
- **Decision:** Warn whenever either clipping coefficient is below `0.1` and
  report exact steps/coefficients. Keep the existing forward dynamics and
  hierarchical clips while measuring recurrence and fixed-manifest effects.
  Do not infer collapse or change balanced optimization from one finite,
  successfully bounded hard example.
- **Alternatives considered:** stop protocol 18 immediately; remove recursive
  velocity supervision; reduce force capacity; lower the clip thresholds;
  compute separate per-scenario gradients on every update.
- **Consequences:** Monitoring now distinguishes ordinary clipping from an
  update that discards over 90% of a raw group gradient. A repeated severe-clip
  rate or broad fixed-validation regression will trigger a gradient-aggregation
  intervention; isolated hard examples remain trainable without silent drift.

## ADR-088 — Aggregate every scenario in each shared-model optimizer update

- **Date:** 2026-08-09
- **Status:** accepted and implemented; sustained qualification pending
- **Context:** Protocol 17 remained finite and supported, yet steps 512--2,048
  repeatedly exchanged accuracy between camera/depth, glancing contacts,
  heavy/light impacts, and other regimes. Exact module swaps localized the
  later regression to coupled updater/dynamics drift. The causal loader used
  batch size two over eight randomly shuffled scenarios, so individual updates
  optimized an incomplete and often conflicting view of the shared objective.
- **Decision:** Add a manifest-bound scenario-balanced step-indexed sampler.
  Each update contains equal support from every declared scenario, per-scenario
  pools shuffle independently, and an absolute data-draw index reconstructs
  exact continuation. Reject partial/unequal protocols. Initialize a new
  state/dynamics campaign weights-only from the best step-512 candidate; keep
  perception frozen and the full batch-one selector unchanged.
- **Alternatives considered:** continue protocol 17 unchanged; reduce the
  corrector gate globally; freeze only the updater; combine early updater with
  late dynamics; lower learning rate on random pair batches; relax guardrails.
- **Consequences:** One update is more expensive and uses about 1.20 GB in the
  measured batch-eight smoke, but represents the actual shared objective and
  remains practical on this host. This is a new optimization protocol under
  specification 1.18, not an exact resume or an accuracy claim.


## ADR-001 — Persistent belief is authoritative

- **Date:** 2026-07-26
- **Status:** accepted
- **Context:** Online observations must revise a continuously maintained physical
  model without re-encoding all history.
- **Decision:** `WorldBelief` is the runtime source of truth. Dynamics creates a
  prior and observation modules correct it incrementally.
- **Alternatives:** sliding clips; recurrent sensor tokens.
- **Consequences:** timestamps, masks, uncertainty, IDs, and lifecycle are
  explicit; modality caches cannot silently replace physical state.

## ADR-002 — Stable modality contract

- **Date:** 2026-07-26
- **Status:** accepted
- **Decision:** Sensors emit typed `MeasurementSet` values and project beliefs
  through the common observation-module contract. Core dynamics has no RGB
  branch.
- **Consequences:** future audio, skeleton, depth, and IMU adapters can reuse the
  scheduler/filter.

## ADR-003 — Measurement-space prediction

- **Date:** 2026-07-26
- **Status:** accepted
- **Decision:** The milestone predicts centres, apparent size, inverse depth,
  visibility, appearance, and covariance rather than future RGB pixels.
- **Consequences:** state errors remain interpretable and the renderer cannot
  hide incorrect physics.

## ADR-004 — Synthetic RGB first, oracle debug only

- **Date:** 2026-07-26
- **Status:** accepted
- **Decision:** A labelled 3-D sphere world is the first real modality. Simulator
  state can supervise/evaluate and drive a clearly named debug oracle, but an
  RGB claim must consume images plus known calibration only.

## ADR-005 — Stable modes instead of a fixed DCT window

- **Date:** 2026-07-26
- **Status:** accepted
- **Decision:** Keep bounded rotation-decay modal state beside explicit
  kinematics. A window DCT may be evaluated later as a baseline.
- **Consequences:** the dynamical programme supports causal correction and
  arbitrary query times.

## ADR-006 — Hybrid structured physics

- **Date:** 2026-07-26
- **Status:** accepted
- **Decision:** Use analytic timestamp-aware kinematics and collision impulses,
  stable modes, small learned residuals/interactions, structured event jumps,
  and explicit process noise.
- **Alternatives:** one opaque transition MLP.

## ADR-007 — Diagonal uncertainty for Milestone 1

- **Date:** 2026-07-26
- **Status:** accepted
- **Decision:** Store clamped diagonal log variance plus categorical logits.
- **Consequences:** filtering and calibration are tractable; quaternion
  uncertainty is an acknowledged tangent-space approximation.

## ADR-008 — Known calibrated moving camera

- **Date:** 2026-07-26
- **Status:** accepted
- **Decision:** Milestone 1 receives calibrated intrinsics/extrinsics while still
  transforming measurements explicitly between camera and world frames.
- **Consequences:** camera estimation is deferred without learning image-space
  physics.

## ADR-009 — Plain PyTorch and YAML

- **Date:** 2026-07-26
- **Status:** accepted
- **Decision:** Use dataclasses, `PyYAML`, standard PyTorch, local JSONL, and
  simple root CLIs. Do not use Hydra, Lightning, hosted tracking, or compiled
  ROI/graph dependencies.

## ADR-010 — Apache-2.0

- **Date:** 2026-07-26
- **Status:** accepted
- **Decision:** License the initial repository under Apache-2.0 for permissive
  use with explicit patent terms.

## ADR-011 — Device evidence must follow the actual subprocess

- **Date:** 2026-07-26
- **Status:** amended after direct device probes
- **Context:** MPS availability is process-sensitive on this host: some pytest
  launcher subprocesses report it unavailable, while direct conda Python
  allocates MPS tensors and completes training.
- **Decision:** Record the device reported by each artifact/command, never infer
  it from another subprocess, and never replace the installed PyTorch.
- **Evidence:** The primary deterministic run is CPU-labelled. Separate reduced
  runs completed real optimizer steps on `mps`.

## ADR-012 — Checkpoint compatibility precedes state loading

- **Date:** 2026-07-26
- **Status:** accepted
- **Context:** Same-shaped weights can still be semantically incompatible with
  camera geometry or physical boundaries; loading a checkpoint can overwrite
  derived dynamics buffers.
- **Decision:** Validate the complete model/runtime configuration plus
  architecture-sensitive simulator fields (image size, frame/physics rate,
  bounds, radius range, gravity, and camera-pose contract) before
  `load_state_dict`.
- **Consequences:** Evaluation/demo budget and split settings may change, but
  incompatible physical/runtime semantics fail loudly.

## ADR-013 — Selection provenance is part of checkpoint truth

- **Date:** 2026-07-26
- **Status:** accepted
- **Context:** Measurement pretraining loss and closed-loop rollout loss are not
  comparable.
- **Decision:** Save separate `best_measurement.pt` and `best_rollout.pt`
  artifacts. A rollout checkpoint can only be selected by an explicit
  closed-loop validation rollout term.

## ADR-014 — Distance-gated detection is separate from assignment coverage

- **Date:** 2026-07-26
- **Status:** accepted
- **Context:** Hungarian alignment always returns assignments when counts match,
  even for physically poor estimates.
- **Decision:** Retain ungated assignments only for fair model/baseline error
  masks. Report them as assignment coverage, and use a documented 0.5 m gate
  for detection, identity, and parameter-alignment quality.

## ADR-015 — Geometric occlusion guides lifecycle

- **Date:** 2026-07-26
- **Status:** accepted
- **Decision:** Use calibrated projected circle overlap and camera depth to
  predict partial/full occlusion. Fully occluded objects are excluded from ROI
  association but retain identity, receive slower existence decay, and grow
  uncertainty until reappearance.

## ADR-016 — Narrow association ambiguity for the toy cost scale

- **Date:** 2026-07-26
- **Status:** accepted for current profiles
- **Context:** The initial margin of 0.5 marked every real RGB pair ambiguous
  when selected costs were about 0.2 and alternatives differed by roughly
  0.03. This disabled all slow-parameter updates.
- **Decision:** Use an absolute margin of 0.02 for the current normalized cost
  scale, retain hard slow-update suppression for genuinely ambiguous pairs,
  and report actual runtime gates/update counts.
- **Consequence:** Identifier heads train and update, but useful parameter
  convergence remains an empirical requirement rather than a claimed result.

## ADR-017 — Physical localization gates the closed-loop curriculum

- **Date:** 2026-07-26
- **Status:** accepted after convergence diagnosis
- **Context:** Summed RGB measurement loss was dominated by a negative Gaussian
  NLL. The selected 12-step detector had nearly image-independent vertical
  centres and about 0.97 m held-out proposal error despite a favourable scalar
  objective. Fixed loader position also coupled each episode to only even or
  odd frame indices.
- **Decision:** Fixed datasets sweep a frame only after every loader batch has
  seen the preceding frame. Measurement validation aggregates configured
  frames and ranks checkpoints by calibrated backprojected world-position MAE.
  Closed-loop training restores that checkpoint and uses a separately
  configured lower learning rate.
- **Consequences:** The tiny profile now trains 64 measurement steps across all
  frames of eight episodes, then six RGB-only closed-loop steps at 0.1x
  learning rate. Checkpoint metadata records the physical localization score;
  negative NLL remains an optimization/calibration diagnostic, not a claim of
  accurate perception.

## ADR-018 — Fast depth residuals require reliability evidence

- **Date:** 2026-07-26
- **Status:** accepted as a safety gate
- **Context:** The earlier ROI inverse-depth delta doubled the signed depth
  error on both train and held-out episodes. Zeroing only that component
  changed ordinary fast corrections from harmful to beneficial; zeroing centre
  deltas made results worse.
- **Decision:** Keep the global/ROI architecture and fast centre, size, colour,
  existence, appearance, and uncertainty outputs, but default
  `fast_depth_residual_enabled` to false. The ROI measurement retains the
  analytic predicted inverse depth until a trained residual passes an explicit
  held-out per-mode current/future improvement gate.
- **Consequences:** The runtime still exercises fast ROI measurements on
  ordinary frames and no history is re-encoded. Learning the fast depth delta
  is deferred to belief-slot-aligned cached-sequence supervision; the gate is
  not presented as a solved depth estimator.

## ADR-019 — Protect calibrated discovery during downstream convergence

- **Date:** 2026-07-26
- **Status:** accepted after controlled continuation
- **Context:** Extending unrestricted closed-loop training from the step-70
  checkpoint produced a lower sampled validation loss but degraded eight-test
  current MAE from 0.182494 m to 0.236864 m and 0.5-second RMSE from 0.162259 m
  to 0.269230 m.
- **Decision:** Keep the RGB backbone/global detector trainable only for an
  explicit `closed_loop_global_trainable_steps` adaptation window, then freeze
  them while the ROI updater, filter, dynamics, and identifier continue
  training. Fast ROI supervision is applied on every usable frame and follows
  the belief-slot-to-target assignment rather than rematching conditioned
  outputs.
- **Evidence:** A 24-step frozen continuation preserved 75.39% distance-gated
  detection, improved current MAE to 0.178773 m, and improved 0.5-second RMSE
  to 0.161387 m over eight held-out episodes.
- **Consequences:** Longer profiles can configure a larger joint-adaptation
  window. Freezing is a convergence safeguard, not a claim that global
  perception is complete.

## ADR-020 — Event logits have explicit interval semantics

- **Date:** 2026-07-26
- **Status:** accepted
- **Context:** Simulator frame labels mean that a collision occurred anywhere
  in the preceding observation interval. Dynamics previously returned only
  the final internal-substep event mode, and multi-horizon rollout segments did
  not match those label windows.
- **Decision:** Persistent `motion_mode_logits` remain instantaneous endpoint
  state. `RolloutStep.event_logits[..., COLLISION]` instead means occurrence
  anywhere in that rollout segment and max-aggregates internal substeps.
  Training/evaluation insert exact `{h-dt_obs, h}` boundaries and select only
  the endpoint logit for each frame label. Zero-duration segments explicitly
  contain no collision occurrence.
- **Evidence:** Focused dynamics tests force a collision before the final
  internal substep and verify that the segment retains it. On the unchanged
  RGB checkpoint, the exact-window implementation changed held-out F1 from 0
  to 0.0556 without changing weights. A separate scratch oracle-state
  diagnostic was consistent with the corrected semantics but is not retained
  as acceptance evidence.
- **Consequences:** Event accuracy is now measured against the right contract,
  but remains an open learning problem rather than being repaired by metric
  relabeling.

## ADR-021 — Correction regularization cannot reward harmful posteriors

- **Date:** 2026-07-26
- **Status:** accepted
- **Context:** A correction-magnitude penalty alone encourages zero updates,
  even when evidence should repair state. The specification also requires
  sparsity so the filter cannot reconstruct state gratuitously each frame.
- **Decision:** Retain the small correction-sparsity term and add supervised
  hinge losses on current and future posterior error relative to a detached
  prior. The prior is detached so dynamics cannot satisfy the relative
  objective by making its incoming prediction worse.
- **Consequences:** Logs separately report correction magnitude, current
  improvement, and future improvement. Absolute state/rollout losses and the
  sparsity term remain active.

## ADR-022 — Acceptance metrics must measure temporal direction

- **Date:** 2026-07-26
- **Status:** accepted
- **Context:** Pooled visible/occluded uncertainty and parameter MAE on frames
  where an update gate fired do not prove uncertainty growth/recovery or that
  an online parameter update moved toward truth.
- **Decision:** Occlusion evaluation anchors a target to a persistent
  prediction ID on a reliable visible frame, follows that ID without a
  localization gate while fully hidden, and reports paired peak growth,
  identity survival, and reobservation contraction only after a complete
  visible-hidden-visible transition. Parameter evaluation snapshots physical
  values before ingest and reports signed before/after error change only when
  persistent identity, distance gating, runtime update gating, and
  evaluation-only informative event/motion evidence all agree.
- **Consequences:** Missing transitions produce explicit zero counts and null
  rates. Update counts can no longer be presented as identification progress;
  the current tiny checkpoint is truthfully measured as making
  numerically negligible drag/restitution changes.

## ADR-023 — Checkpoint selection uses explicit fresh validation seeds

- **Date:** 2026-07-27
- **Status:** accepted
- **Context:** Repeatedly inspecting the same eight test episodes made earlier
  continuation comparisons exploratory rather than independent evidence. The
  trainer also uses the first configured validation episodes internally.
- **Decision:** `evaluate.py --seed-protocol fresh_validation --split
  validation` begins immediately after the checkpoint's recorded
  `training.validation_episodes`. Reports persist every seed and assert that
  this manifest overlaps neither trainer validation nor the reserved test
  range. `--seed-offset` can select a later explicit block, but fresh
  validation rejects offsets that overlap trainer validation. Standard
  validation/test evaluation also accepts an explicit offset so paired
  comparisons never depend on implicit checkpoint provenance.
  Collision-conditioned forecasts use simulator collision labels only as an
  evaluation mask over `(anchor, target]`, with identical masks for the model
  and every baseline.
- **Evidence:** The step-72 comparison used seeds `100004–100019`; metadata
  records no trainer-validation or test overlap.
- **Consequences:** These episodes are suitable for checkpoint selection, not
  final test acceptance. Reusing them for several temporal hyperparameter
  ablations makes those ablations selection evidence and requires a later
  confirmation block before promotion.

## ADR-024 — Temporal RGB history is separate, causal, and gated by evidence

- **Date:** 2026-07-27
- **Status:** accepted experimentally; disabled by default
- **Context:** A one-frame position-to-velocity update amplifies RGB covariance
  by `1/dt²` and changed held-out velocity error by only about `0.0013 m/s`.
  ROI feature caches are also intentionally invalidated by global discovery,
  so they cannot safely own persistent kinematic history.
- **Decision:** Add a bounded sensor-local history of corrected RGB positions,
  diagonal variance, timestamps, validity, and persistent object IDs. A
  modality hook updates it after association and position correction; three
  strictly increasing same-ID samples produce a causal least-squares velocity
  observation followed by a velocity-only diagonal correction. Global/ROI
  feature cache changes do not erase this history. Births, deaths, ambiguity,
  ID changes, stale timestamps, reset, and detach are explicit. The default
  propagates position uncertainty without a variance ceiling; any empirical
  ceiling must be an explicit operational override.
- **Evidence:** On a development validation block, a three-position slope was
  more accurate than the prior. On fresh selection seeds, a calibrated
  `1.0 (m/s)²` ceiling improved current velocity RMSE from `1.369454` to
  `1.309964 m/s` and collision F1 from `0.042553` to `0.055172`, but worsened
  current position MAE from `0.186991` to `0.190923 m`, 0.25-second RMSE from
  `0.189670` to `0.201318 m`, and perturbation recovery from `20.09%` to
  `19.26%`. A 22-step continuation raised F1 to `0.121622` but further
  regressed the primary physical metrics.
- **Consequences:** The architecture and diagnostics are retained, but
  `temporal_velocity_enabled` remains false in public profiles and the
  continuation checkpoint is a negative experiment, not the promoted model.

## ADR-025 — Missing interaction edges are not zero-valued event evidence

- **Date:** 2026-07-27
- **Status:** accepted
- **Context:** Dense graph tensors store diagonal and absent edges as zero.
  Max-pooling those tensors prevented a valid negative learned edge logit from
  suppressing an analytic false positive.
- **Decision:** Mask invalid edges and diagonals before neighbor max-pooling.
  Preserve signed values on valid edges and use a finite neutral residual only
  for nodes with no valid neighbor.
- **Consequences:** Positive and negative learned interaction-event evidence
  now has the intended semantics. The unchanged step-72 checkpoint's fresh
  collision F1 remained `0.042553`, so this correctness fix is not presented
  as an accuracy gain by itself.

## ADR-026 — Synthetic structured RGB localization is an optional measurement prior

- **Date:** 2026-07-27
- **Status:** accepted for synthetic sphere profiles
- **Context:** The learned proposal head localized horizontal image position
  reasonably but collapsed much of the vertical motion. The specification
  explicitly permits structured sphere geometry in the RGB adapter, while
  forbidding simulator state at runtime.
- **Decision:** Global discovery may refine learned proposal centres using
  only the current RGB frame: estimate the row-wise background by a robust
  median, threshold foreground evidence, split connected silhouettes at
  distance-transform peaks, compute photometric centroids, and align those
  centroids to learned proposal slots with Hungarian assignment. Ordinary
  updates use a separate projected-ROI RGB centroid refinement; they do not
  run global connected-component discovery. Both paths retain the original
  learned centre in `auxiliary.raw_centre`, and training applies a direct raw
  centre loss so the structured forward measurement cannot starve the learned
  heads of localization gradients.
- **Scope:** The dataclass default remains disabled for future real-video
  adapters. Current synthetic sphere YAML profiles enable it. Noise-free,
  default, and `toy_mps` profiles use threshold `0.04`; `toy_hard` and cloud
  profiles use the measured noise-robust threshold `0.08`.
- **Evidence:** On RGB frames from seeds `100004–100019`, peak splitting
  increased matched global centroids from `455/512` to `507/512`, with mean
  normalized centre error `0.0014439`. A separate `toy_mps` diagnostic matched
  all `57/57` visible objects on sampled frames. At threshold `0.04`,
  `toy_hard` noise produced hundreds of false basins; threshold `0.08`
  restored the expected ten components in the sampled hard/cloud scenes.
- **Consequences:** This is a transparent RGB-only toy prior, not evidence of
  general real-video perception. Structured controls alter measurement means
  and variances and are therefore checkpoint semantics. Missing legacy fields
  normalize to their historical defaults; a checkpoint cannot silently
  enable or retune structured/temporal measurement behavior.

## ADR-027 — Optimize and select measurement and rollout quality in physical units

- **Date:** 2026-07-27
- **Status:** accepted
- **Context:** The old RGB loss could be dominated by a negative
  heteroscedastic NLL, and the old aggregate rollout loss was dominated by
  velocity because metres and metres/second were averaged without an explicit
  tradeoff. Tiny validation improvements as small as a few millionths then
  selected checkpoints that regressed held-out position and recovery.
- **Decision:** Supervise calibrated RGB world position with Huber and
  Gaussian-NLL terms, apply explicit per-measurement weights, and expose raw
  learned-centre supervision separately from the structured forward
  measurement. Closed-loop objectives keep position and velocity terms
  separate for current state and rollout. `best_rollout.pt` is selected by
  rollout-position loss with a `1e-5` minimum improvement; aggregate aliases
  remain only for backward-compatible logging/configuration.
- **Consequences:** Loss magnitude is no longer presented as localization
  accuracy. `best_measurement.pt` is selected by metric world-position MAE,
  while rollout selection prioritizes the physical forecast quantity used for
  promotion. Uncertainty NLL remains active at a deliberately smaller weight.

## ADR-028 — Closed-loop windows require causal context and representative validation

- **Date:** 2026-07-27
- **Status:** accepted
- **Context:** Previous validation consumed one rotating batch and only the
  first TBPTT frames. Mid-episode train windows started from a cold reset, and
  uniform short windows provided weak collision coverage.
- **Decision:** Every checkpoint validation evaluates every configured
  validation episode and the complete causal episode. A sampled mid-episode
  training window first ingests its RGB prefix under `no_grad`, detaches the
  resulting persistent belief/caches/history, and then backpropagates only
  through the selected TBPTT span. Half of windows preferentially contain a
  collision when labels are available; labels choose a training span only and
  never enter runtime. Position and velocity perturbation magnitudes are
  explicit independent configuration values.
- **Consequences:** Validation values are comparable across checkpoints and
  measure the same persistent online loop used by evaluation. The extra
  validation and causal prefix work costs more compute but removes the
  alternating-seed and cold-start biases that invalidated earlier tiny
  selection.

## ADR-029 — Promotion requires paired physical evidence, not scalar tuning

- **Date:** 2026-07-27
- **Status:** accepted after accuracy-v4 continuation
- **Context:** A checkpoint may improve forecast dynamics while showing tiny
  mixed changes in current position, velocity, recovery, or coverage. Event
  threshold sweeps can also appear attractive without fixing mis-timed or
  structurally wrong state.
- **Decision:** Promote a closed-loop checkpoint only when its physical gains
  repeat on paired selection and confirmation manifests. Do not retune the
  collision threshold when positive/negative logits are saturated, and do not
  replace learned depth with an analytic radius rule unless it passes the same
  metric-space confirmation protocol.
- **Evidence:** Step 648 improved all forecast horizons on both manifests and
  confirmation collision F1 `0.594203 → 0.608059`; its final-test F1 was
  `0.640000`. No threshold improved the saturated event logits. Mean-radius
  depth produced about `0.795 m` error versus `0.148 m` learned, and a
  photometric-radius variant failed confirmation.
- **Consequences:** Step 648 supersedes step 584 despite tiny mixed secondary
  tradeoffs, which remain reported. Collision threshold stays `0.5`, learned
  depth stays active, and a two-frame anisotropic velocity slope remains an
  unimplemented research option rather than a promoted inference heuristic.

## ADR-030 — Multistep objectives aggregate globally by physical horizon

- **Date:** 2026-07-27
- **Status:** accepted
- **Context:** Averaging a weighted horizon loss separately at every anchor
  lets short tail windows renormalise their few available horizons to full
  weight. This overrepresents short predictions and makes a scalar validation
  score depend on window geometry rather than the configured physical
  objective.
- **Decision:** Accumulate each configured horizon across every eligible
  anchor, average within that horizon, and only then apply the configured
  horizon weights with their fixed total denominator. Apply the same rule to
  future-correction loss. Prefer collision-bearing windows first and
  maximum-horizon-capable windows otherwise with an explicit probability.
  Persist per-horizon validation losses and a selection-semantics version;
  inherited scores from older semantics cannot silently suppress a new best
  checkpoint.
- **Consequences:** A 0.5/0.75/1.0-second objective means the same thing across
  train windows, and a tiny tail cannot inflate short-horizon gradients.
  Sequence length still limits eligibility, so the dedicated deterministic
  multistep profile uses 32 frames. Three controlled continuations failed the
  external physical gate, so this correctness change does not promote a new
  checkpoint by itself.

## ADR-031 — Forecast visualisation preserves history and fixed geometry

- **Date:** 2026-07-27
- **Status:** accepted
- **Context:** Automatically derived axes/legends and showing only the newest
  line made GIF motion difficult to distinguish from plot-layout motion and
  hid whether successive forecasts were converging toward the realised path.
- **Decision:** Fix world bounds, margins, legend entries, and legend
  placement. Retain each posterior forecast in absolute world coordinates and
  fade it monotonically by age, while drawing the newest prior/posterior more
  strongly. Match the latest forecast endpoint to evaluation-only ground truth
  with Hungarian assignment and display absolute prior/posterior error plus
  correction gain. Generate enough scoring-only lookahead that every displayed
  frame keeps the requested horizon.
- **Consequences:** The GIF directly exposes forecast drift and revision
  without changing runtime inference. Simulator labels remain overlay/scoring
  data read after RGB ingest and never become model input.

## ADR-032 — Do not hide state error with horizon-specific forecast blending

- **Date:** 2026-07-27
- **Status:** accepted as a research guard
- **Context:** Oracle-start diagnostics show that one-second error falls from
  about `0.221 m` to `0.091 m` when current position and velocity are supplied,
  and to `0.0473 m` when slow parameters are also supplied. Learned dynamics
  ablations have millimetre-or-smaller leverage. A split diagnostic found a
  `7.48%` long-horizon gain from suppressing gravity-orthogonal displacement,
  but a scalar or horizon-specific interpolation regressed held-out error.
- **Decision:** Prioritise RGB depth, anisotropic velocity, and slow-state
  observability. Do not add a fitted per-horizon or fixed-axis output blend. A
  future gravity-aligned motion gate is acceptable only when driven by
  uncertainty/observability, propagated coherently through position, velocity,
  covariance, and event state, and confirmed on wider and OOD motion.
- **Consequences:** Reported trajectories remain physically self-consistent
  rather than cosmetically calibrated to one split. Step 648 remains promoted
  until an RGB-only candidate improves paired long-horizon physical metrics.

## ADR-033 — Use bounded camera-lateral velocity evidence only for track initialization

- **Date:** 2026-07-27
- **Status:** accepted for the tiny lateral-velocity profile
- **Context:** RGB centroids contained the missing horizontal displacement, but
  the isotropic position-to-velocity fallback divided full 3-D backprojection
  variance by `dt²`. At 20 Hz this produced roughly `700 (m/s)²` uncertainty
  and negligible horizontal gain. Continuously applying a strong temporal
  slope improved velocity but regressed localization and forecasts.
- **Decision:** Maintain bounded causal positions by persistent ID, estimate a
  least-squares slope, project only onto the known camera-lateral world
  direction, and give unobserved axes high variance so analytic gravity and
  depth dynamics remain authoritative. Clear history at collision events and
  restrict evidence to young tracks. Blend associated RGB world position into
  corrected posterior history by `0.125`; do not use simulator state or a
  horizon-specific output correction.
- **Evidence:** On fresh-validation seeds `100096–100111`, the selected
  candidate lowers current position/velocity RMSE by `6.84% / 3.60%` and
  recursive 0.1–1.0 s position RMSE by `7.49–11.18%`. Collision F1 changes
  `0.404092 → 0.398922` and nominal-90% coverage
  `0.862745 → 0.846814`. Strong continuous, raw two-frame, raw three-frame,
  and eight-step adaptation variants were rejected.
- **Consequences:** The new behavior is explicit checkpoint semantics and
  opt-in through `configs/tiny_lateral_velocity.yaml`; legacy checkpoints
  normalize to the old disabled defaults. Event timing and uncertainty
  calibration remain separate acceptance gates.

## ADR-034 — Ground-truth trajectory overlays separate history, horizon, identity, and time

- **Date:** 2026-07-27
- **Status:** accepted
- **Context:** Ground truth was drawn twice with one colour: once from episode
  start through the future endpoint and again from the current frame. In a
  two-sphere collision that produced overlapping curves and apparent vertical
  lines with no clear identity or time direction.
- **Decision:** Draw each object once per temporal segment: faint dotted past
  through now, then a solid current horizon. Use persistent per-object colours,
  identity labels, sampled time markers, start/end glyphs, and a final-segment
  arrow. Keep the legend and world geometry fixed across GIF frames.
- **Consequences:** Historical forecasts and ground truth can be compared
  without duplicated traces or layout motion. Simulator positions remain
  post-ingest scoring/overlay data only and never enter RGB runtime inference.

## ADR-035 — Interaction regimes are deterministic physical range presets

- **Date:** 2026-07-27
- **Status:** accepted
- **Context:** A single broad random range made it difficult to tell whether a
  checkpoint handled elastic pair collisions, damped contacts, and external
  disturbances or merely averaged over them.
- **Decision:** Add named physical presets selected deterministically from an
  ordered `scenario_mixture` by episode seed. Presets alter only simulator
  parameter/impulse sampling and write their name into episode metadata; they
  do not change timestamps, object padding, RGB packets, belief contracts, or
  runtime access to labels.
- **Consequences:** Training and evaluation can use the same data contract
  while reporting each interaction regime separately. A singleton mixture
  gives a reproducible per-scenario evaluation; the four-element mixture gives
  balanced deterministic training coverage.

## ADR-036 — Reject short mixed-interaction adaptations that trade position for velocity

- **Date:** 2026-07-27
- **Status:** accepted rejection
- **Context:** Eight closed-loop mixed steps helped impulse-driven scenes but
  were neutral elsewhere. Adding eight three-object RGB pretraining steps
  improved velocity and some collision scores but reduced discovery recall and
  regressed position forecasts across the paired scenario suite.
- **Decision:** Do not promote either `accuracy-interactions-v1` or
  `accuracy-interactions-v2`. Keep `accuracy-lateral-velocity-v5` selected.
  Before further dynamics training, require a balanced multi-object
  perception curriculum and explicit per-query/distance-gated recall gate.
- **Consequences:** Scenario support is shipped as validated infrastructure,
  while checkpoint limitations remain truthful. Finite execution across a
  regime is not presented as accurate generalization.

## ADR-037 — Generated artifact directories sort newest by timestamp

- **Date:** 2026-07-27
- **Status:** accepted
- **Context:** Training runs, repeated evaluations, and RGB demos used
  inconsistent suffix timestamps or unversioned names. This made the newest
  GIF difficult to identify and allowed repeated evaluation output to reuse a
  directory.
- **Decision:** Prefix every newly created training, evaluation, and demo
  directory basename with a UTC `YYYYMMDD-HHMMSS-` timestamp, including
  caller-supplied labels. Treat an existing prefix as idempotent. A resume
  without a new run name continues in the checkpoint's original run
  directory. Keep research runs in place because reports and checkpoint
  provenance cite them; move superseded demos into a recoverable timestamped
  `demo_outputs/archive/`.
- **Consequences:** Lexicographic folder order is chronological and every CLI
  reports its actual generated path. Existing research evidence remains valid.
  Timestamp resolution is one second, so callers launching identical labels
  within the same second must provide distinct labels.

## ADR-038 — Delay uncertain contact jumps and reject the broader adapted checkpoint

- **Date:** 2026-07-27
- **Status:** accepted
- **Context:** The contact resolver already localized analytic impacts at
  120 Hz substeps, but monocular RGB height uncertainty placed some posterior
  means too close to a surface and caused premature jumps. A structured
  apparent-radius depth replacement was tested and rejected because physical
  radius and depth are not separately observable from one fixed-camera
  silhouette.
- **Decision:** Require the geometric gap plus `0.25` projected standard
  deviations to reach contact before applying a pair/plane jump. Keep the
  original step-648 weights with this explicit runtime semantic. Add camera
  parallax, glancing-impact, and unequal-mass regimes to the deterministic
  curriculum, but reject the 24-RGB/16-closed-loop step-680 continuation
  because it regressed the decisive original 16-episode multistep gate.
- **Evidence:** On seeds `100096–100111`, the new contact semantics change
  0.10/0.25/0.50/0.75/1.00-second RMSE from
  `0.150671/0.173691/0.196885/0.204839/0.209191` to
  `0.147986/0.168943/0.189775/0.191977/0.200973 m`; collision F1 changes
  `0.398922 → 0.409836`. Velocity RMSE regresses
  `0.762795 → 0.830722 m/s`. The adapted step-680 checkpoint reaches
  `0.241969 m` at one second and is rejected.
- **Consequences:** Contact timing now responds explicitly to uncertainty,
  rather than treating an uncertain mean as exact. The remaining fixed-camera
  scale/height ambiguity is not claimed solved. Per the user's explicit
  cleanup request, superseded run directories were deleted after the accepted
  checkpoint and compact evidence were consolidated into one timestamped run.

## ADR-039 — Predictive abstractions mediate foundation-model scaling

- **Date:** 2026-07-27
- **Status:** accepted; first contract increment implemented
- **Context:** Scaling to realistic video with a monolithic video generator or
  opaque transformer state would discard the useful compression already
  demonstrated by representing a ball as a persistent point and trajectory.
  Conversely, fixing every entity permanently to the sphere schema cannot
  express articulated, field-like, or unknown processes.
- **Decision:** Make the smallest executable predictive abstraction the unit
  of scaling. `WorldBelief` remains authoritative and retains explicit state,
  uncertainty, identity, geometry, slow parameters, and residual codes.
  Abstraction assignments select an execution family as a derived view. The
  first router uses `POINT_TRAJECTORY` for free motion and refines to
  `RIGID_SPHERE` for contact-like modes. Add a reversible typed belief-token
  interface for future foundation encoders and transformers.
- **Alternatives considered:** replace the runtime with autoregressive video
  generation; store a transformer KV cache as the only state; create a fixed
  universal rigid-body ontology; add future abstraction names without
  executors.
- **Consequences:** existing checkpoints and runtime behavior are unchanged;
  the new router/tokenizer contain no learned parameters and never cache a
  second physical state. The router initially reports a recommended execution
  family but does not prune the existing hybrid contact-candidate path, because
  mode-only routing could miss an imminent collision. Future learned residuals,
  routing, generative hypotheses, actions, and language must produce typed
  proposals and pass structured prediction/calibration/complexity gates before
  assimilation.

## ADR-040 — Separate familiar reference physics from harder learnable regimes

- **Date:** 2026-07-27
- **Status:** accepted
- **Context:** The prior ensured-pair generator allowed the pair impact and
  first floor impact to occur in the same 50 ms RGB interval. Floor friction
  could then cancel lateral velocity immediately after an otherwise valid pair
  impulse. The resulting trace was physically explainable but visually
  surprising and unsuitable as the primary correctness demonstration.
- **Decision:** Add a named `reference_pairs` scenario with fixed visible
  radius, low drag/friction, familiar restitution, stronger approach speed,
  and enough initial height/surface gap to separate pair and floor events.
  Parameterize ensured-pair height, surface gap, and speed. Regression-test
  that the reference pair is separating after collision and does not receive a
  simultaneous ground event. Treat compound and unusual dynamics as separately
  named later curricula, not as evidence that an ambiguous reference is
  acceptable.
- **Consequences:** Simulator data/version metadata advances to sphere-world
  v2. The model remains able to learn arbitrary dynamics from observations;
  only the benchmark's meaning is made explicit. Existing v1 metrics and
  checkpoints remain historical evidence and are not directly comparable
  without identifying their simulator family.

## ADR-041 — Use a physics engine as an independent dataset backend, not a predictor

- **Date:** 2026-07-27
- **Status:** accepted; implementation deferred
- **Context:** A mature rigid-body engine would provide independently
  implemented contacts, friction, rolling, spin, stacking, and compound
  interactions that are useful for realistic validation. Neither PyBullet nor
  MuJoCo is currently installed in `orpheus`; Gymnasium is available. Adding a
  heavyweight dependency during this accuracy investigation would obscure the
  current causal comparison and make the default smoke path less reliable.
- **Decision:** Future engine integration must sit behind the canonical
  episode and timestamped RGB `ObservationPacket` contracts. Privileged engine
  state is restricted to supervision, evaluation, tests, and labelled oracle
  debugging. Record engine/version, solver, timestep, units, seed, scenario,
  and split manifest, and report engine-backed metrics separately. Keep the
  analytic sphere world as the fast invariant oracle and do not feed engine
  state/equations to the runtime predictor.
- **Consequences:** The same `WorldBelief`, measurement, association,
  innovation, correction, and rollout code must work across both backends.
  The engine tests whether dynamics were learned rather than memorized from
  one simulator, without narrowing the architecture to rigid-body physics.

## ADR-042 — Axis-local losses and structured point/scale evidence remain subordinate to held-out multistep gates

- **Date:** 2026-07-27
- **Status:** accepted with limitations
- **Context:** Aggregate 3-D error hid a weak camera-lateral x estimate. Raw
  structured RGB centers were much more accurate than the posterior on that
  axis, while continuous learned-residual ablations had negligible effect.
- **Decision:** Export x/y/z current and forecast position/velocity metrics and
  allow separately weighted rollout-position losses. Use an RGB disc's center
  as point evidence and its connected-component area as explicit scale/depth
  evidence in the reference scenario. Keep object existence/lifecycle
  confidence distinct from position confidence. Preserve constant/damped
  per-axis motion as the low-complexity prior, while joint geometric/event
  context remains able to change any component.
- **Evidence:** On the four-seed held-out diagnostic, the selected cadence-4
  candidate reached current x RMSE `0.473544 m` and 1-second model x RMSE
  `0.885634 m`, versus `1.063635 m` for constant velocity. Cadence 2 worsened
  the model's 1-second x RMSE to `0.962442 m`. Replacing conservative temporal
  blending with raw measured points improved current x velocity but worsened
  cadence-4 1-second x RMSE to `0.940926 m`. Applying structured confidence to
  fast ROI centers also worsened it to `0.889826 m`; those variants were
  rejected.
- **Consequences:** The implementation now makes the axis failure visible and
  prevents a jointly averaged metric from hiding it. The accepted changes
  improve interpretability and preserve multistep behavior, but do not solve
  absolute x accuracy; a larger clean reference curriculum and learned
  uncertainty/velocity gating remain required.

## ADR-043 — Select one shared checkpoint on an explicit balanced scenario manifest

- **Date:** 2026-07-28
- **Status:** accepted
- **Context:** Scenario-specific tuning can make each small synthetic regime
  look better while avoiding the intended question: whether the same
  abstraction, belief, observer, and dynamics can predict across distinct
  interactions. The evaluator's prior default fresh-validation offset also
  depended on the checkpoint's embedded validation count, which made an
  adapted checkpoint appear better on different episodes.
- **Decision:** Add one deterministic eight-scenario profile and use one shared
  checkpoint. Persist the ordered scenario mixture and episode-scenario list.
  Paired selection must specify the same seed offset or seed manifest, object
  counts, sequence length, and horizons. Resumed best-score compatibility now
  includes every selection-defining field available in configuration.
- **Consequences:** The first apparent training improvement was invalidated,
  and no candidate weights were promoted. This is stricter but prevents seed
  composition from masquerading as learning. Scenario-specific checkpoints
  remain diagnostic ablations only.

## ADR-044 — Permit dynamics-only adaptation but promote only paired gains

- **Date:** 2026-07-28
- **Status:** accepted as an experiment control; candidates rejected
- **Context:** Short full-model mixed-regime adaptation risks degrading RGB
  discovery while testing whether shared dynamics can improve. A restricted
  scope is useful for separating these effects.
- **Decision:** Add `training.closed_loop_trainable_scope`, with `all` as the
  compatibility default and `dynamics` as the controlled alternative. Run
  conservative and higher-rate dynamics-only continuations, but require the
  same paired held-out manifest for promotion.
- **Evidence:** At explicit fresh-validation offset 8, the higher-rate
  candidate changed one-second aggregate RMSE from `0.285499` to `0.285456 m`
  while worsening current position, x position, and earlier horizons. The
  conservative candidate likewise produced no robust shared gain.
- **Consequences:** The option remains available for longer experiments, but
  the selected artifact retains the prior step-672 learned weights. Training
  activity alone is not reported as accuracy improvement.

## ADR-045 — Reset temporal velocity history on collision edges, not collision state

- **Date:** 2026-07-28
- **Status:** accepted
- **Context:** Collision mode can remain active for several observations.
  Clearing history on every active frame prevents the observer from collecting
  the outgoing samples needed to estimate post-contact velocity.
- **Decision:** Store an aligned `reset_active` flag per track, clear samples
  only on the false-to-true edge, and continue accumulating while the mode is
  sustained.
- **Consequences:** The causal observer can learn outgoing velocity without
  re-encoding history. The correction is invariant-tested, though the current
  selected episodes showed no material aggregate metric change, so it is not
  presented as an accuracy gain.

## ADR-046 — Treat monocular scale quality and velocity evidence as explicit, separately gated signals

- **Date:** 2026-07-28
- **Status:** accepted infrastructure; experimental policies rejected
- **Context:** A raw RGB audit found subpixel structured-disc centres
  (`0.1388 px` RMSE), but much larger 3-D error (`0.3865 m` RMSE), dominated
  by heavy-tailed radius-derived depth (`0.3837 m` camera-depth RMSE) during
  overlap and truncation. Temporal least-squares velocity and instantaneous
  position-innovation velocity are correlated evidence and should be testable
  separately rather than silently replacing one another.
- **Decision:** Add optional, configuration-explicit depth-disagreement
  covariance inflation after association, and optional position-innovation
  velocity coupling alongside temporal velocity. Both default to disabled.
  Promotion requires identical paired seed manifests and every declared
  multistep horizon.
- **Evidence:** Adaptive depth improved four of five aggregate test horizons
  but regressed the 1-second endpoint (`0.364040 → 0.364672 m`) and x endpoint
  (`0.462605 → 0.463177 m`). Hybrid velocity improved every horizon on two
  validation blocks, then regressed final-test 1-second RMSE
  (`0.364040 → 0.365094 m`). A 128-step balanced RGB continuation raised
  paired-validation 1-second RMSE from `0.285499` to `0.329262 m`.
- **Consequences:** The selected configuration and checkpoint are unchanged.
  The next accuracy implementation is a learned multi-frame point/scale
  trajectory measurement: estimate axes from recent evidence, expose
  scale/occlusion quality, and allow joint event context to gate changes. It
  remains observation evidence correcting `WorldBelief`, not a parallel
  source of truth.

## ADR-047 — Scale examples and capacity with bounded causal graphs

- **Date:** 2026-07-28
- **Status:** accepted; full training pending
- **Context:** The selected 156,490-parameter model and 128 fixed training
  episodes are debugging scale. A first larger run also showed that batch-four,
  16-step closed-loop graphs at 1.90 million parameters create excessive
  memory/latency even on MPS, while on-the-fly rendering is CPU-bound without
  loader workers.
- **Decision:** Add one 1,901,030-parameter shared profile over 4,096 training
  and 256 validation episodes spanning eight balanced scenario families. Use
  batch one, eight-step TBPTT, four renderer workers, 48,000 episode draws, and
  no rendered-episode memory cache. Keep the same belief/runtime contracts and
  require disjoint RGB-only multistep evaluation.
- **Evidence:** MPS was built and available outside the execution sandbox.
  Four data workers reduced the 64–128 pretraining segment to roughly
  `174 s`, versus about `511 s` for the first CPU-bound segment including
  validation. A bounded run completed 256 measurement updates (1,024 episode
  draws), selected validation world-position MAE `0.645048 m`, and checkpointed
  one full causal MPS update at step 257 with finite gradient norm `3.950501`.
  This is pipeline evidence, not an accuracy promotion.
- **Consequences:** The full 48,000-step run is now mechanically defined but
  remains expensive. Its first candidate must be evaluated against the
  selected small checkpoint and simple baselines on identical manifests.
  Gradient checkpointing or explicit optimizer accumulation may be added later
  without changing runtime semantics.

## ADR-048 — Preserve final weights before validation and reject mixed long-horizon gains

- **Date:** 2026-07-28
- **Status:** accepted
- **Context:** A 1.90M-parameter, 24-frame closed-loop validation remained
  compute-active for about 84 minutes and prevented the old trainer from
  checkpointing its final eight optimizer updates. A separate higher-rate
  continuation also produced one non-finite proposal row, and MPS evaluation
  exposed a direct float64-cast failure in scoring-only parameter metrics.
- **Decision:** Write final weights before entering final validation, then
  overwrite them with validated selection metadata only after success. Ignore
  non-finite proposal rows during structured RGB assignment, fail explicitly
  on non-finite validation aggregates, and transfer MPS diagnostics to CPU
  before float64 accumulation. Do not promote a checkpoint unless the same
  disjoint manifest improves current state and every declared forecast gate.
- **Evidence:** Step 896 improved paired current-position RMSE by `18.60%` and
  0.10/0.25/0.50/0.75-second RMSE by
  `12.89/5.68/5.04/1.16%`, but worsened velocity RMSE by `94.11%` and
  one-second RMSE by `5.04%`. Direct MPS regression tests pass after the
  reporting fix.
- **Consequences:** Step 896 is retained as diagnostic perception evidence,
  not as the selected model. Large-model throughput must be profiled and the
  velocity/long-horizon objective must improve before scaling the schedule
  further.

## ADR-049 — Separate weights-only curriculum transfer from checkpoint resume

- **Date:** 2026-07-29
- **Status:** accepted
- **Context:** The scaled monocular curriculum varied unknown sphere radius
  from `0.16–0.25 m` while RGB back-projection used the range mean. A single
  apparent radius cannot identify physical radius and depth independently.
  Reusing the learned detector on an identifiable fixed-radius accuracy
  curriculum is useful, but changing those simulator semantics is not a valid
  optimizer/RNG resume.
- **Decision:** `train.py --initialize-from` loads a trusted checkpoint's model
  tensors strictly while resetting step, optimizer, scheduler, and RNG into a
  new timestamped run. `--resume` remains the only exact-continuation path and
  the two options are mutually exclusive. The primary scaled accuracy
  curriculum uses a known `0.21 m` radius; variable-radius data remains an
  OOD/parameter-identification problem rather than a hidden ambiguity in the
  localization gate.
- **Evidence:** A 1,024-draw, eight-scenario MPS transfer reduced the
  eight-episode measurement world-position MAE from `0.614574 m` to
  `0.380453 m`. The stricter online result did not improve automatically,
  proving that tracker/fusion drift remains distinct from measurement
  identifiability.
- **Consequences:** Transfer provenance is explicit in run metadata and
  summaries. Fixed-scale results do not establish variable-size
  generalisation, and must not be compared as same-dataset checkpoint deltas.

## ADR-050 — Re-anchor scaled tracks every three frames and require forecastable windows

- **Date:** 2026-07-29
- **Status:** accepted provisionally
- **Context:** The fixed-scale detector reached `0.380453 m` standalone MAE,
  while six-frame online anchoring produced roughly `0.9–1.2 m` current RMSE.
  A conservative ROI scale estimate improved some position metrics but
  degraded velocity, detection, event F1, and calibration. Causal training
  also sampled late collision windows with zero valid future horizons.
- **Decision:** Keep ROI scale extraction behind the disabled
  `structured_disc_fast_depth_enabled` gate. In the scaled profile, run global
  discovery every three frames and retain fast ROI updates on the intervening
  two. Closed-loop window selection must leave at least one anchor with the
  shortest configured future horizon, even when a late collision label would
  otherwise take priority.
- **Evidence:** On seeds `100016–100017`, the configuration then labelled
  cadence three improved current
  RMSE/MAE by `5.9%/21.3%`, velocity slightly, 0.10–0.75-second RMSE by
  `7.6–12.8%`, collision F1 from `0.138` to `0.357`, and detection
  recall/precision from `0.500/0.377` to `0.694/0.568`. On disjoint seeds
  `100018–100019`, it improved current RMSE/MAE by `6.3%/12.5%`, velocity by
  `10.7%`, 0.10–0.75-second RMSE by `3.4–9.0%`, collision F1, and target
  coverage. One switch occurred on the first pair and nominal 90% coverage
  worsened on the second. The corrected sampler produced nonzero position and
  velocity rollout losses at step 6. The 3 August audit later proved that the
  implementation emitted `GLOBAL, FAST, FAST, FAST, GLOBAL`; this evidence
  therefore supports a denser-than-six cadence but is not evidence for the
  intended exact three-frame sequence.
- **Consequences:** Corrected cadence three remains the scaled design default,
  but the old evidence does not validate its exact sequence; it requires a
  fresh rollout-protocol-11 validation/test manifest. The old cadence-four
  one-second horizon remained essentially unchanged. Longer causal training is
  allowed only from forecast-supervised windows and remains subject to paired
  promotion gates. A step-16 sampler-corrected continuation under the old
  cadence was subsequently rejected: current state improved marginally, but
  0.25–1.00-second forecasts regressed.

## ADR-051 — Preserve scarce scale anchors separately from per-frame point history

- **Date:** 2026-07-29
- **Status:** accepted for scaled position prediction; event/velocity follow-up
  required
- **Context:** Global fixed-radius RGB localization is substantially more
  accurate than the online posterior, but apparent scale is heavy-tailed under
  overlap and truncation. A single mixed five-frame history at global cadence
  three can never contain three global scale observations because intervening
  centre-only ROI samples evict them.
- **Decision:** Keep two bounded sensor-local rings per persistent object ID:
  one per-frame point ring for axis-local velocity and one scale-anchor ring
  that advances only for nonambiguous associated global silhouettes. Reject
  boundary-truncated and multi-peak/overlap-split silhouettes as scale anchors
  without discarding their useful centres. Fit an inverse-variance,
  three-iteration Huber/IRLS trajectory independently by axis and evaluate it
  at the current timestamp. Emit the result as optional typed direct position
  evidence, projected onto the calibrated camera-depth direction by default,
  and correct `WorldBelief` through the normal robust diagonal filter.
- **Alternatives considered:** average raw depths across moving objects; enlarge
  the mixed velocity window until it happens to retain scale frames; enable the
  rejected single-frame ROI scale correction; store the fitted trajectory as a
  second authoritative tracker.
- **Evidence:** On fresh-validation seeds `100016–100017`, the initial
  parameter-free weights-identical ablation reduced current position RMSE from
  `0.906217` to `0.684258 m` and recursive
  0.10/0.25/0.50/0.75/1.00-second RMSE from
  `0.780533/0.626639/0.650438/0.773491/1.007125` to
  `0.698125/0.526197/0.517001/0.562448/0.618672 m`. Velocity RMSE regressed
  `1.082334 → 1.148529 m/s`, collision F1 regressed
  `0.357143 → 0.271186`, and detection precision regressed
  `0.568182 → 0.480583`. On disjoint seeds `100018–100019`, current RMSE
  improved `1.165912 → 0.804367 m` and every recursive horizon improved from
  `1.010213/0.877051/0.922522/0.988420/1.267293` to
  `0.802223/0.630088/0.644967/0.693634/0.760509 m`. Detection
  recall/precision improved `0.391667/0.345588 → 0.675000/0.455056`, nominal
  90% coverage improved `0.554667 → 0.722522`, and identity switches remained
  zero. Confirmation velocity RMSE regressed
  `0.889775 → 0.986646 m/s` and collision F1 regressed
  `0.190476 → 0.078431`.
- **Consequences:** The observer removes the cadence/history impossibility and
  attacks the measured monocular depth ceiling without new weights or history
  re-encoding. Position and velocity evidence retain independent validity and
  uncertainty. The scaled profile enables the confirmed position policy
  because current state and all declared position horizons improve strongly on
  both paired blocks; it does not claim an overall event-model promotion.
  Event-conditioned outgoing velocity, per-scenario gates,
  correlation-aware calibration, variable-radius depth, and real video remain
  open.

## ADR-052 — Keep acceleration-aware gravity velocity behind an observed-event gate

- **Date:** 2026-07-29
- **Status:** capability accepted; scaled policy rejected
- **Context:** After point/scale position correction, aggregate velocity error
  is dominated by the gravity axis. A straight line through an accelerated
  history estimates velocity near the window midpoint rather than at the
  current timestamp. Over a five-frame 20 Hz window, gravity alone creates
  roughly a `0.98 m/s` endpoint bias. The existing post-event history reset
  also depends on the transient endpoint `COLLISION` mode.
- **Decision:** Support an optional causal known-acceleration fit that subtracts
  quadratic displacement about the query timestamp, then projects evidence
  only into the calibrated camera-lateral and gravity subspace. Never expose
  monocular camera-depth slope through this gate. Apply the gravity component
  only after an edge-triggered event reset. Keep the scaled default disabled
  until RGB trajectory residuals provide a validated contact/change-point
  trigger.
- **Alternatives considered:** enable noisy all-axis slope; use the
  uncompensated line slope as current velocity; treat every endpoint contact
  mode as an event reset; promote a small velocity gain despite multistep/event
  regressions.
- **Evidence:** On validation seed `100016`, unrestricted all-axis correction
  raised velocity RMSE to `2.228049 m/s`. Camera-lateral plus gravity reduced
  velocity RMSE from `1.288819` to `1.150215 m/s` and vertical RMSE from
  `1.965171` to `1.654356 m/s`, but worsened current position and every
  recursive horizon and reduced collision F1. Raising variance reduced the
  tradeoff but still worsened all position horizons. Post-event-only correction
  was exactly identical to baseline because no usable endpoint collision reset
  occurred. Broadening reset to endpoint contact yielded only
  `0.02–0.09%` gains at three horizons, essentially flat/slightly worse results
  at two horizons, and collision F1 `0.285714 → 0.266667`.
- **Consequences:** The estimator math and configuration contract are tested
  and available for the next observed-change-point experiment, while the
  validated scaled runtime remains unchanged. This avoids encoding constant
  velocity as a hard rule: analytic acceleration is removable known context,
  and observation evidence still controls whether velocity is corrected.

## ADR-053 — Retain causal RGB change points as an opt-in capability

- **Date:** 2026-07-29
- **Status:** capability accepted; scaled policy rejected
- **Context:** Endpoint motion modes miss collisions that begin and end between
  observations, so ADR-052 needs an observation-side event gate. Consecutive
  RGB point segments can expose a velocity discontinuity, but monocular depth
  noise and ordinary accelerated motion can resemble one.
- **Decision:** Add a causal three-point kinematic change-point detector. It
  subtracts the velocity change explained by known gravity, operates only in
  explicitly observable subspaces, resets the point ring without destroying
  independent scale anchors, and records reset provenance. A change-point
  reset may expose a short gravity-only outgoing-velocity correction after two
  post-event samples; it never exposes camera-depth slope. Require the
  acceleration-aware post-event correction whenever this detector is enabled.
  Keep it explicitly disabled in the scaled profile.
- **Alternatives considered:** reset on every raw residual; treat all endpoint
  contact modes as event onsets; discard scale anchors at a kinematic reset;
  silently promote a policy because one current-position metric improved.
- **Evidence:** On seed `100016`, a permissive detector fired `45/111`
  inspected object frames and changed current RMSE `0.648034 → 0.637888 m`, but
  regressed velocity RMSE `1.288819 → 1.357281 m/s` and collision F1
  `0.285714 → 0.250000`. Conservative, gravity-only, and provenance-decoupled
  variants fired `23`, `10`, and `15` times and all regressed current and/or
  velocity accuracy. Requiring an endpoint contact reduced activation to
  `1/110` on seed `100016` and `0/177` on seed `100017`; the first fast check
  was identical to its baseline on comparable current, velocity, detection,
  and 0.1-second metrics. This is safe but too sparse to solve outgoing
  velocity.
- **Consequences:** The runtime/data contract can now express and diagnose
  observed motion discontinuities without simulator input or history
  re-encoding. Promotion still requires a learned, uncertainty-aware RGB gate
  trained on balanced contact/no-contact windows and paired multistep evidence.

## ADR-054 — Learn event gates offline, but do not promote without state benefit

- **Date:** 2026-07-29
- **Status:** workflow accepted; learned policies rejected
- **Context:** The heuristic change-point detector in ADR-053 could be noisy or
  inert. A learned gate needs balanced labels, exact alignment to asynchronous
  RGB history, uncertainty inputs, and cheap inference. Adding a large
  recurrent network or updating weights online would violate the intended
  fast correction path.
- **Decision:** Export nine causal features per persistent object: signed and
  absolute acceleration-compensated residual, standardized residual, adjacent
  velocities, reversal, minimum speed, propagated log variance, and contact
  probability. Preserve the exact three observation timestamps. Train either
  logistic regression or an eight-hidden-unit MLP offline using simulator
  collision/velocity only as supervision. Store coefficients in typed runtime
  config and cache feature tensors. At inference, apply at most a tiny MLP and
  no history re-encoding or weight update. Keep every learned gate disabled in
  the scaled default until paired physical metrics improve.
- **Alternatives considered:** label every RGB feature with the immediately
  preceding simulator frame; train a large sequence model; accept classifier
  precision without measuring downstream state; enable the sparse gate because
  it is nearly neutral.
- **Evidence:** Misaligned logistic labels produced validation precision
  `0.2727` and recall `0.0822`. Exact timestamp alignment yielded 543
  train/398 validation windows with 197/146 observable event positives. The
  linear gate could not meet useful precision/recall. A loose MLP reached
  precision `0.6000`, recall `0.1644`, but fired 13 times on seed `100016` and
  collapsed detection recall to `0.2083`. A sparse threshold reached held-out
  precision `0.7500`, recall `0.0411`. It was exactly baseline on seed
  `100016`; on seed `100017`, current RMSE improved
  `0.702313 → 0.702296 m` and 0.1-second RMSE
  `0.715284 → 0.715256 m`, but velocity RMSE regressed
  `1.148770 → 1.154865 m/s`.
- **Consequences:** Event-gate data generation, caching, fitting, configuration,
  and provenance are now reproducible. The experiment falsifies “better event
  classification alone fixes velocity.” The next accuracy target is a
  calibrated learned outgoing-velocity proposal trained jointly with the gate,
  while the validated runtime remains unchanged.

## ADR-055 — Retain the event gate and outgoing proposal as disabled capabilities

- **Date:** 2026-07-30
- **Status:** workflow accepted; runtime policies rejected
- **Context:** The sparse learned gate in ADR-054 has useful precision but poor
  recall and does not itself estimate an outgoing state. Deleting it would
  discard exact-timestamp event data and a cheap diagnostic, while enabling it
  would regress velocity. A proposal trained only on true events can also fail
  when exposed to runtime false positives or a different post-reset window.
- **Decision:** Keep the learned gate disabled by default and add a tiny
  one-hidden-layer proposal for a bounded gravity-axis velocity delta. Its
  inputs are the nine causal gate features, current prior gravity velocity, and
  gate probability. Cache aligned prior/target velocities during
  simulator-supervised collection, calibrate proposal variance on held-out
  windows, impose a refractory interval, and consume the proposal only on the
  exact frame whose features triggered it. Simulator values remain labels and
  metrics only. Do not activate either learned component unless paired online
  state and forecast metrics improve.
- **Alternatives considered:** remove the gate; apply a positive-only regressor
  to every selected runtime window; reuse a newly recomputed proposal after
  two post-reset samples; promote a small position gain despite worse velocity.
- **Evidence:** The aligned collection contained 543 train and 398 validation
  windows. A positive-only proposal reduced held-out positive-window RMSE from
  `2.795367` to `1.194373 m/s`, but did not model false selections. A
  gate-focused, `1.5 m/s`-bounded joint fit reduced all-window validation RMSE
  `1.693066 → 1.548441 m/s` and gate-selected RMSE
  `1.638817 → 1.537791 m/s`. Delayed runtime application on seed `100017`
  worsened current/velocity RMSE to `0.702754 m` / `1.170360 m/s`. Consuming
  the aligned proposal immediately improved current position
  `0.702313 → 0.701963 m` and 0.1-second forecast
  `0.715284 → 0.714966 m`, but velocity still worsened
  `1.148770 → 1.173099 m/s`.
- **Consequences:** The scaled runtime and accepted checkpoint remain
  unchanged. The code can now reproduce the failed experiment without
  conflating classification, regression, and timing. The next accuracy target
  is an intervention-aware camera-lateral outgoing correction trained on its
  actual post-filter and recursive forecast effect, with learned
  abstention/gain; the gravity-only gate is not evidence for lateral events.

## ADR-056 — Fit the actual lateral filter intervention, but reject runtime promotion

- **Date:** 2026-07-30
- **Status:** workflow accepted; runtime policies rejected
- **Context:** The gravity-axis event gate and outgoing proposal did not
  estimate the dominant camera-lateral collision response. The prior collector
  also read the belief after ordinary direct velocity correction, so its
  outgoing value was not the actual prior on which a new intervention would
  act.
- **Decision:** Export pre-direct-correction velocity, log variance,
  measurement confidence, camera-lateral basis, eight lateral and eight
  gravity kinematic features, and contact probability. Fit a tiny MLP to a
  bounded lateral measurement delta and continuous soft gain while simulating
  the same diagonal robust Kalman update used at runtime. Fold training feature
  normalization into its first layer. Map low gain to large measurement
  variance, optionally with a power of at least one, rather than introducing a
  hard event gate. Keep the capability disabled in the shared profile unless
  paired recursive metrics improve.
- **Alternatives considered:** reuse the gravity-only classifier as a lateral
  gate; supervise the already-corrected posterior; promote on offline
  post-filter RMSE; promote on one short-horizon seed; repeatedly apply a
  high-confidence fixed correction.
- **Evidence:** The aligned dataset contained 543 train and 398 disjoint
  validation windows. A 12-unit head overfit, worsening held-out RMSE
  `0.648080 → 0.702765 m/s`. A standardized, strongly regularized one-unit head
  improved it to `0.497431 m/s`. It improved seed `100017` x-velocity
  `0.421218 → 0.352271 m/s`, but on the protocol-matched two-episode block
  x-velocity regressed `0.568277 → 0.576981 m/s` and x-position forecast RMSE
  regressed by `3.12%`, `4.52%`, and `5.47%` at 0.5, 0.75, and 1.0 seconds.
  A squared-gain variance variant failed the fast current-position gate.
- **Consequences:** The typed observation/filter workflow can now train and
  measure the intervention it actually performs, and failed fits remain
  reproducible artifacts. No runtime default or accepted checkpoint changes.
  The next accuracy target is the much larger gravity-axis velocity error,
  using an axis-local acceleration-aware correction with joint collision
  context and recursive multihorizon acceptance.

## ADR-057 — Aggregate on-policy gravity corrections, but require observability preservation

- **Date:** 2026-07-30
- **Status:** workflow accepted; runtime policy rejected
- **Context:** Ordinary accepted RGB velocity evidence is camera-lateral only,
  leaving the dominant gravity-axis velocity mostly under analytic dynamics.
  A raw acceleration-aware RGB slope is noisier than the prior, while a learned
  residual fitted on baseline priors changes the distribution it sees when
  repeatedly applied.
- **Decision:** Add a separate gravity-only intervention head over 21 causal
  features: gravity and lateral three-point kinematics, contact probability,
  exact pre-filter gravity velocity/variance, and acceleration-aware candidate
  residual/variance. Preserve non-gravity means and mark their measurement
  variance unobserved. Fit the actual robust diagonal filter intervention.
  Permit collection from an intervention-enabled checkpoint for one-step
  dataset aggregation. Keep the feature disabled in the shared configuration
  unless current velocity, detection/identity, and every selected recursive
  horizon pass paired evaluation.
- **Alternatives considered:** expose the raw y slope continuously; update all
  velocity axes together; accept current-position gains despite worse
  velocity; weaken a failing head only through fixed variance; train forever
  on baseline priors.
- **Evidence:** The baseline-prior fit reduced held-out gravity residual RMSE
  `2.222436 → 1.771491 m/s`. One on-policy pass collected 498/373 train/
  validation windows and reduced held-out RMSE `2.113796 → 1.854939 m/s`.
  Seed `100017` improved current position, y/total velocity, and 0.1-second
  forecast. On the paired `100016–100017` block, current y position improved
  `13.03%` and collision-conditioned 0.1-second forecast improved `13.49%`,
  but y velocity regressed `4.03%`, detection recall fell `14.65%`, collision
  F1 fell `19.34%`, and overall 0.25–1.0-second forecasts regressed.
- **Consequences:** The repository can reproduce baseline-prior and on-policy
  axis-local intervention experiments without oracle runtime state or
  cross-axis covariance leakage. The failed paired result shows that local
  post-filter objectives and one DAgger-style pass do not capture future
  observability. The next target must train through association, ROI
  scheduling, identity, and recursive horizon losses.

## ADR-058 — Train one sustained shared model behind fixed broad guardrails

- **Date:** 2026-07-30
- **Status:** policy accepted; v1/v2 instances superseded, v3 pending qualification
- **Context:** The fixed-scale 1.90M-parameter model received only 1,024
  measurement episode draws and 16 rejected causal updates. Later intervention
  heads improved their local fit targets but regressed the recursive RGB loop.
  The nominal 40,000-update causal schedule would take months at the measured
  161–242 seconds per old update, while another short screen would not answer
  convergence. The old position-loss selector could also promote a checkpoint
  that lost tracks or regressed velocity, events, identity, or uncertainty.
- **Decision:** Initialize one shared eight-scenario model from the selected
  fixed-scale point/scale runtime, with experimental intervention heads
  disabled. Run 8,192 measurement updates and 4,096 causal windows. Score one
  wide-horizon anchor per training TBPTT window while ingesting and supervising
  every frame; score every horizon supported by that anchor, with the sampler
  balancing collision and long-horizon windows, and score every eligible
  posterior anchor in validation. Select by
  pooled horizon-weighted physical position RMSE only when current
  position/velocity, every horizon, distance-gated recall/precision and
  identity, forecast lifecycle coverage, collision F1, and nominal-90%
  calibration remain within declared tolerances against both the moving best
  and the fixed initialization reference. Retain numbered candidates. Bind
  selection metadata to exact protocol/seed and tensor hashes, and start fresh
  AdamW moments at the measurement-to-causal handoff.
- **Alternatives considered:** continue fitting isolated event/velocity heads;
  launch the infeasible 48,000-step profile unchanged; stop after 1,024 causal
  windows; select only the lowest position loss; compare each candidate only
  with the moving incumbent; reuse incumbent metrics without linked weights.
- **Evidence:** The broadly tested small model beats constant velocity at every
  horizon but remains inaccurate. Fixed scale and point/scale history improve
  scaled position on two disjoint blocks while velocity/event metrics regress.
  Old scaled causal runs cost 161–242 seconds per update and an eight-episode
  validation remained active after about 84 minutes. One bounded anchor across
  4,096 independent windows provides about 512 windows per scenario and 4,096
  multihorizon anchors, twice the causal-anchor count of the rejected
  1,024-window/two-anchor draft.
- **Consequences:** The first credible campaign is expected to take roughly
  three to seven days on MPS. No accuracy gain may be claimed until it
  completes and a winner passes at least 64 fresh balanced validation episodes
  with per-scenario reporting. If the best safe checkpoint occurs in the final
  1,024 causal updates with at least 1% improvement, extend rather than
  declaring convergence.

## ADR-059 — Continue sustained training with verified plateau evidence

- **Date:** 2026-07-30
- **Status:** policy accepted; no convergence supervisor currently active
- **Context:** A fixed 12,288-update process cannot determine in advance
  whether the broad validation objective has plateaued. Manual ad-hoc
  extensions would invite short-run decisions, training-loss selection, or
  accidental resumption from the selected checkpoint instead of the mutable
  optimizer/RNG iterate. The initial macOS job may also restart after normal
  completion, while an unattended supervisor must not overlap trainers or
  retry deterministic failures forever.
- **Decision:** Verify the completed summary, `last.pt`, linked
  best/reference selectors, all numbered validation candidates, exact
  validation protocol, and actual model-tensor hashes. Resume in place only
  from `last.pt` and change only `training.steps`, in complete 4,096-update
  causal blocks. Extend immediately when the best guardrail-safe checkpoint
  lands in the final 1,024 updates with at least 1% relative primary-score
  improvement. Declare plateau only when the exact four latest 512-step
  validations accept no candidate and the best raw primary-score gain over the
  safe pre-window incumbent is below 1%; missing or contradictory evidence
  triggers another block. Cap the campaign at 24,576 total updates. A valid
  plateau at the cap remains `plateau`; reaching the cap without that evidence
  is `limit_hit`, not convergence. Persist supervisor state/events, monitor the
  initial PID, reattach only to an exact matching extension after restart, and
  stop automatic retries after a recorded child failure.
- **Alternatives considered:** stop unconditionally at the original budget;
  make extension decisions from training loss; resume from
  `best_rollout.pt`; accept two rejections as a plateau; train indefinitely;
  launch independent extension processes; let launchd retry failed children
  without a recorded terminal state.
- **Evidence:** The trainer already saves the optimizer/RNG-bearing `last.pt`,
  tensor-linked selector checkpoints, numbered accepted/rejected validation
  candidates, and a validation protocol hash that intentionally excludes the
  training budget. Focused convergence/provenance tests and existing
  checkpoint-integrity tests report `17 passed`. The persistent LaunchAgent
  started successfully beside the existing trainer with one supervisor and
  one trainer process, and wrote its initial waiting event without errors.
- **Consequences:** The campaign can run unattended without weakening its
  scientific stopping rule, and interrupted supervision can recover without
  overlapping the active trainer. `limit_hit` is explicitly a safety-budget
  outcome. Neither `plateau` nor `limit_hit` promotes a model by itself: at
  least 64 fresh balanced validation episodes and all broad guardrails remain
  required before reserved-test evaluation.

## ADR-060 — Fix global axis-horizon weighting without mutating a live protocol

- **Date:** 2026-07-31
- **Status:** implementation accepted; v1/v2 campaigns superseded, v3 pending qualification
- **Context:** Batch-one losses in the sustained run varied sharply. A complete
  numerical audit found finite weights, gradients, and optimizer moments, but
  also found that the actively optimized x/y/z rollout-position losses did not
  implement ADR-030. Aggregate position used the fixed total configured
  horizon denominator; each axis was renormalized over only the horizons
  available in one sampled window. Collision conditioning also returned before
  long-horizon intent was sampled, leaving only 26/73 logged windows with a
  one-second target. The first causal validation improved every broad metric
  and the first four horizons but missed the one-second guardrail by
  `0.004311 m`.
- **Decision:** Emit per-axis per-horizon losses and aggregate x/y/z over the
  same fixed configured denominator as aggregate position. Sample collision
  and maximum-horizon intent independently; satisfy both when possible, and
  retain long-horizon intent when a late collision makes the conjunction
  impossible. Log both the pre-clip gradient norm and the bounded applied norm.
  Keep the two behaviors configurable only to preserve the already-running
  campaign: its base profile explicitly records both legacy values as false,
  while new configurations default true. Do not interrupt, reinterpret, or
  silently change that live run. Freeze the unused ROI event head and
  objective-disconnected identifier variance head in restricted adaptation
  scopes without deleting their checkpoint tensors.
- **Alternatives considered:** diagnose divergence from individual raw losses;
  raise or lower the learning rate mid-run; weaken the broad one-second
  guardrail; mix corrected objective semantics into an in-place continuation;
  discard collision sampling; claim a code-level fix as measured accuracy.
- **Evidence:** Through step 8776, all 73 logged causal rows and all checkpoint
  tensors/moments were finite. Median/maximum total loss were `9.325/29.91`;
  71/73 pre-clip norms exceeded one but every update was clipped. Step 8704
  improved weighted score `0.543%`, current position `1.445%`, velocity
  `2.816%`, gated coverage `10.865%`, precision `11.521%`, collision F1
  `5.042%`, and 0.10–0.75-second forecasts, while one-second RMSE regressed
  `2.445%`. The final repository suite reports `318 passed, 4 skipped`, with
  all four hardware-conditional files passing directly on MPS.
- **Consequences:** Raw console loss remains expected to vary across object
  counts, scenarios, matches, events, and horizon support, but future logs
  expose the actual clipped update magnitude. The affected v1 campaign is a
  preserved legacy-objective experiment; the later v2 campaign was also
  superseded by ADR-070. A separate timestamped supported campaign must
  complete balanced coverage and broad validation before this fix can be
  credited with improved physical accuracy. ROI event features and slow
  uncertainty calibration remain explicit follow-up model tasks rather than
  nominally trainable dead heads.

## ADR-061 — Separate deployment safety from the mutable optimisation trajectory

- **Date:** 2026-08-01
- **Status:** accepted and implemented
- **Context:** The sustained step-8192 perception candidate improved weighted
  selection score `15.69%`, one-second RMSE `33.13%`, target coverage `23.28%`,
  and collision F1 `31.45%`, but velocity regressed `4.44%` and correctly
  failed the 2% deployment guard. The trainer then reloaded the safe step-zero
  checkpoint before causal training. Exact tensor comparison showed 79/84
  global RGB tensors changed at handoff and 0/84 differed from step zero in
  every later causal checkpoint.
- **Decision:** A rejected candidate does not replace the safe deployment
  incumbent, but a finite candidate remains the mutable optimisation state so
  downstream objectives can repair its failed guardrail. Persist the fixed
  reference, moving incumbent, candidate tensor hash, acceptance, and every
  structured rejection reason separately. Do not conflate checkpoint
  selection with phase-state rollback.
- **Alternatives considered:** weaken the velocity guard; always deploy the
  primary-score winner; always roll training back to the safe incumbent;
  discard the completed perception phase and restart.
- **Consequences:** Safety gates retain their meaning without erasing useful
  representation learning. A later candidate still needs all broad
  guardrails; continuing from it is not promotion.

## ADR-062 — Optimise deterministic futures only while causally identifiable

- **Date:** 2026-08-01
- **Status:** accepted and implemented
- **Context:** At 20 Hz and 24 frames, only anchors 0–3 support a one-second
  target, while temporal velocity becomes observable around frame three.
  Random restitution/mass/drag are not visually identifiable before contact.
  The impulse scenario has a `92.24%` chance of an unseen intervention within
  one second, and coupled contacts can transfer that intervention to any
  object. Exact point/event targets after it are mutually incompatible.
- **Decision:** Separate cold and mature tracks. Train deterministic position,
  velocity, event, and posterior-improvement losses only on mature support
  before any unseen scene actuation. Continue Gaussian forecast likelihood on
  realised stochastic outcomes so uncertainty can widen. Use 40-frame
  episodes, fixed configured horizon denominators, and explicit support counts.
- **Alternatives considered:** hardcode constant velocity; remove stochastic
  scenarios; train all realised point targets; ignore cold-start error; weaken
  one-second selection.
- **Consequences:** The model remains free to learn arbitrary dynamics, but is
  no longer pushed toward an average of impossible futures. Cold-start,
  pre-identification, stochastic, and mature deterministic performance must be
  reported separately.

## ADR-063 — Make continuation, simulation, and trend validation reproducible

- **Date:** 2026-08-01
- **Status:** accepted and implemented
- **Context:** Resume replayed an early shuffled permutation, MPS RNG was not
  saved, and long-running checkpoints sampled whatever Git commit was current
  at save time. Rendering and physics shared a generator, so changing only
  render noise changed later impulses and trajectories by up to `0.733 m` in
  the regression fixture. Full all-anchor 40-frame trend validation was also
  operationally excessive.
- **Decision:** Address training samples by absolute optimiser step while
  exactly reproducing the legacy DataLoader draw stream. Save/restore CPU,
  CUDA, and MPS RNG where applicable. Validate exact resume semantics before
  overwriting run metadata and use weights-only initialization for changed
  protocols. Capture source provenance once at launch. Give rendering an
  independent deterministic generator. Validate batch-one episodes with exact
  seed/scenario attribution and a hashed deterministic bounded spread of
  forecast anchors; use a separate larger balanced promotion manifest.
- **Alternatives considered:** serialize DataLoader prefetch internals; allow
  arbitrary resume overrides; query live Git state at every save; seed render
  noise from the physics stream; validate every anchor at every checkpoint.
- **Consequences:** Interrupted training has a well-defined next sample and
  source identity, renderer ablations are physically paired, counts are exact,
  and frequent validation is affordable. Changing the anchor count or any
  objective/data field invalidates selector compatibility.

## ADR-064 — Preserve interval events and distinguish floor support from boundaries

- **Date:** 2026-08-01
- **Status:** accepted and implemented
- **Context:** The dynamics rollout previously exposed only the final
  substep's collision result. A fast impact that separated before the
  observation endpoint therefore disappeared before temporal-history and
  parameter-observability gates consumed it. Separately, every analytic plane
  contact was collapsed into `ground_contact`, so a slow side-wall or ceiling
  contact could cancel tangential motion or preserve sleeping state.
- **Decision:** OR pair and boundary collision evidence across every numerical
  substep while retaining endpoint contact separately. Propagate both through
  zero-step, rollout, runtime, temporal-history, and observability contracts.
  Tag environment planes explicitly; only the lower vertical support plane is
  ground. Cancel sub-threshold inward normal velocity as a constraint but do
  not invent sleep from one substep or erase tangential sliding.
- **Alternatives considered:** lower event thresholds; infer collisions from
  the final motion mode; call every static plane ground; keep simulator and
  belief sleep semantics intentionally different.
- **Consequences:** A collision remains causally available at the next RGB
  update, while wall/ceiling interactions stay real boundary events. Slow
  floor settling no longer produces repeated bounce labels, and side-wall
  contact cannot freeze an otherwise free trajectory.

## ADR-065 — Select perception on lifecycle-qualified pooled evidence

- **Date:** 2026-08-01
- **Status:** accepted and implemented
- **Context:** Measurement validation used the detector's lower confidence
  threshold and selected the smallest localization loss. A proposal could look
  accurate in isolation while never crossing the lifecycle birth threshold,
  so improved MAE could hide collapsed runtime recall. Frame/episode averages
  also gave ratios unequal denominators.
- **Decision:** Evaluate proposals at `lifecycle.birth_confidence`; pool
  additive true positives, targets, proposals, matched counts, and absolute
  errors before deriving runtime MAE, recall, precision, and F1. Restrict the
  assignment itself to lifecycle-qualified proposals so a low-confidence
  localization cannot steal a target; count confident proposals on empty
  target frames as false positives. Select with a versioned broad score plus
  MAE/recall/precision non-regression gates. A candidate with no qualified
  localization support is unusable. Allocate capacity-constrained births by
  stable descending confidence and reset every identity-specific field when
  recycling a slot.
- **Alternatives considered:** keep MAE-only selection; lower the runtime birth
  threshold to match training; average per-frame F1; allocate by query index.
- **Consequences:** `best_measurement.pt` now means usable discovery evidence,
  not merely a well-localized sub-threshold query. Recall and precision cannot
  silently collapse during perception pretraining.

## ADR-066 — Normalize only supported objectives and separate mean from calibration

- **Date:** 2026-08-01
- **Status:** accepted and implemented
- **Context:** Unsupported state/parameter/horizon terms were represented by
  differentiable zeros and then averaged across TBPTT frames, diluting the
  rare event and parameter gradients. State and structured-RGB calibration
  NLLs also duplicated their explicit robust mean gradients with an
  inverse-variance multiplier; a one-batch audit measured this as the dominant
  state gradient, while the first corrected smoke still exposed RGB NLL as a
  hard-frame gradient confound. Collision-conditioned windows often contained
  an event without placing it at any scored endpoint.
- **Decision:** Omit unsupported terms and average each objective only across
  its real support. Keep fixed configured horizon denominators where selection
  is fixed-weight. Detach the already-supervised mean error inside state and
  structured-RGB calibration NLLs while retaining their variance gradients.
  Also linearize RGB world covariance at detached centre/depth coordinates so
  covariance objectives cannot reach mean heads through the Jacobian; leave
  forecast NLL as a proper distributional score. Align a feasible sampled
  collision with the shortest scored event endpoint.
- **Alternatives considered:** tune learning rate/clip norm around the biased
  objective; increase parameter weights without fixing support; remove NLL;
  count any collision anywhere in a window as a positive endpoint.
- **Consequences:** Batch composition still causes truthful stochastic loss
  variation, but it no longer changes an objective merely by adding
  unsupported zero examples. Calibration and mean fitting no longer fight
  through duplicate gradients.

## ADR-067 — Make phase-device continuation and selector linkage explicit

- **Date:** 2026-08-01
- **Status:** accepted and implemented
- **Context:** On this Mac the CNN-heavy perception phase benefits from MPS,
  while a matched branch-heavy causal update was about nine times faster on
  CPU. A single device policy wasted days. The first hybrid implementation
  then exposed continuation hazards: both sides of the phase boundary can
  write the same completed step, documentation-only commits invalidated the
  whole-worktree hash, external config paths could disable Git provenance,
  `best_measurement.pt` was not tensor-linked, and a zero-update resume could
  rewrite an MPS checkpoint as CPU.
- **Decision:** Resolve and record measurement and closed-loop devices
  independently in the immutable protocol. At the boundary, reset optimizer
  moments, move once, and clear runtime caches. Use an explicit handoff marker
  to interpret equal-step checkpoints. Hash executable training source content
  independently of commit/docs while retaining full Git provenance. Verify
  measurement and rollout selector files by protocol, tensor hash, step, and
  actual device, copying and re-verifying linked artifacts for a new run
  directory. Never reserialize an already-complete exact resume.
- **Alternatives considered:** run all phases on MPS; run all phases on CPU;
  permit arbitrary resume device changes; rely on filenames/metric flags;
  reject every commit after launch; rewrite no-op checkpoints for convenience.
- **Consequences:** The default remains one simple training command, but each
  phase uses the measured faster backend. A pause/resume cannot silently
  change numerical semantics or lose the selected perception state, while
  documentation may still advance during a multi-day campaign.

## ADR-068 — Pin the proposal transformer to CPU on PyTorch 2.10 MPS

- **Date:** 2026-08-01
- **Status:** accepted and implemented
- **Context:** The final hybrid smoke failed before its first optimizer update
  despite a finite scalar loss. On the exact seeds `1,2`, a finite contiguous
  `2x96x64x64` MPS backbone feature map generated NaN gradients in eight
  attention/feed-forward matrix weights. The same weights/features were fully
  finite on CPU. Random or constant MPS features did not reproduce it, and a
  replacement token-linear implementation produced unacceptable MPS/CPU
  forward disagreement.
- **Decision:** Keep the convolutional backbone and fast ROI path on MPS, but
  pin the small global proposal transformer to CPU when
  `device.global_detector_cpu_on_mps=true`. Copy its feature input to CPU and
  every output back to the image device without detaching autograd. Include
  the flag in measurement/rollout protocol hashes and exact-resume config
  comparison, with missing legacy fields normalized to the historical
  `false` behavior.
- **Alternatives considered:** run the entire measurement phase on CPU;
  replace attention with an unverified custom projection; ignore the finite
  loss and clip non-finite gradients; disable global detector training.
- **Consequences:** The main convolutional workload still uses MPS and the
  architecture/tensor contracts are unchanged, but the measurement phase is
  explicitly hybrid rather than whole-model MPS. The exact failing batch now
  has zero non-finite gradients and completes AdamW. A host regression covers
  both device gradients, clipping, checkpoint restore, and a second update.

## ADR-069 — Recover terminal validation and deserialize checkpoints on CPU

- **Date:** 2026-08-01
- **Status:** accepted and implemented
- **Context:** An interruption after the final optimizer checkpoint but before
  validation left no way to distinguish complete from pending validation. An
  in-place resume from selector/numbered artifacts could overwrite historical
  source-run state. Mapping a hybrid checkpoint directly to MPS also moved
  non-capturable Adam scalar steps there, while evaluation/demo mapped a full
  unused optimizer onto accelerator memory.
- **Decision:** Save `final_validation_completed=0` before terminal validation
  and `1` only after it succeeds. Exact resume recovers a pending validation
  with zero optimizer updates. Permit in-place resume only from the exact
  `checkpoints/last.pt`; require a new run or weights-only initialization for
  other artifacts. Deserialize trainer, evaluator, demo, and gate-fitting
  checkpoints on CPU, then let state loading copy weights/moments to their
  owning parameter devices. Preserve legacy/completed no-op checkpoint and
  summary bytes.
- **Alternatives considered:** treat the final prevalidation checkpoint as
  complete; always rerun terminal validation; permit any checkpoint to define
  its parent run; map the whole payload to the active accelerator.
- **Consequences:** Completion is durable and recoverable without accidental
  extra training. Historical selector files cannot be silently overwritten.
  Hybrid Adam steps stay on CPU, moments follow parameter owners, and
  evaluation does not duplicate optimizer-sized accelerator storage.

## ADR-070 — Require supported causal updates and clip recursive interactions hierarchically

- **Date:** 2026-08-02
- **Status:** accepted and implemented
- **Context:** The corrected v2 campaign remained finite but did not constitute
  meaningful convergence. Of 173 logged causal training rows, 121 (`69.94%`)
  had an exactly zero pre-clip gradient while still consuming scheduled
  updates. The measurement handoff also collapsed distance-gated current
  target coverage from `0.28754` to `0.04480` and one-second forecast target
  coverage from `0.76146` to `0.05273`. Direct inspection found positive-only
  fast-ROI confidence supervision, invalid attribute targets on empty or
  unreliable crops, false positives missing from selector precision, an
  always-present inactive-query existence loss, and causal windows that could
  optimize global auxiliary perception without any trajectory support. After
  those repairs, an exact hard-window gradient audit found that
  `dynamics.interactions` contributed `85.76` of a raw total norm `85.89`;
  whole-model clipping alone reduced every other useful gradient by the same
  roughly `0.0233` factor.
- **Decision:** A causal optimizer update requires real differentiable
  trajectory/state/parameter support or supported persistent fast-ROI slots.
  Unsupported deterministic draws are counted and retried without advancing
  optimizer state. Fast-ROI masks and objectives follow their actual evidence,
  empty crops train only negative existence/visibility, selector precision
  includes every eligible confident output, and global/fast losses are
  support-normalized separately. Handoff and later candidates must satisfy
  absolute and reference-relative coverage floors; support collapse restores
  the verified incumbent and resets Adam. Clip the learned recursive
  interaction block to `interaction_grad_clip_norm` before applying the
  declared whole-model clip, and persist every raw/intermediate/applied norm
  and coefficient. Persist and guard every declared scenario as a separate
  selector slice; missing scenario support or a slice-level regression rejects
  promotion even when the pooled score improves. Require unique entries in the
  balanced scenario list and a nonnegative integral RGB phase boundary, so
  deterministic validation coverage and handoff semantics cannot be bypassed
  by a malformed resolved configuration.
- **Alternatives considered:** lower the global learning rate around zero
  support; regard zero-gradient rows as benign dataset variance; globally clip
  the `85+` interaction spike; remove the interaction residual; promote
  conditional low-RMSE candidates despite collapsed coverage; turn empty ROI
  crops into zero-valued geometry targets.
- **Consequences:** Completed updates now mean the model received causal
  learning signal, rollout selection cannot reward disappearing predictions,
  and hard contact windows cannot erase unrelated measurement/filter
  gradients merely by dominating the global clip. The forward interaction
  architecture remains unchanged. Raw batch loss and raw gradient spikes
  remain diagnostics rather than convergence claims, and the v2 campaign is a
  preserved invalid-optimization control rather than evidence to extend. The
  final-tree smoke confirmed the stricter behavior: its step-four pooled score
  improved `0.558737 → 0.548741`, but coverage fell and the candidate was
  correctly rejected by both pooled and `reference_pairs` slice guardrails.

## ADR-071 — Confirm births, pre-gate assignment, and isolate causal perception gradients

- **Date:** 2026-08-03
- **Status:** accepted and implemented
- **Context:** The first v3 causal validation improved conditional physical
  RMSE but grew predicted object frames by 33.5%, distance-gated identity
  switches from 10 to 146, collision false positives from 242 to 469, and
  calibration error. Investigation found that unmatched confident global
  proposals were born immediately, first-time simulator target mappings had no
  physical gate, fast ROI rows could cross-update identities, and three
  Hungarian call sites applied gates only after solving. Births also opened
  slow-parameter labels without an accepted innovation. Separately, late raw
  gradients were dominated by the RGB detector/backbone after interactions
  had already been locally stabilized, so the whole-model cap starved the
  filter/dynamics update. Removing false target mappings also exposed a
  zero-support validation path that raised during RMSE derivation instead of
  persisting a rejected candidate.
- **Decision:** Tentative discoveries remain detached `(modality, sensor)`-
  local observation history outside `WorldBelief`; require configured
  consecutive, strictly-later detections within a finite world-space gate
  before allocating a monotonic ID. Gate inadmissible core association,
  tentative confirmation, and new privileged target-alignment edges before
  Hungarian assignment using a penalty that makes valid cardinality primary
  and distance/cost secondary. Lock fast ROI evidence to its explicit source
  slot and object ID, while global discovery remains free to associate.
  Parameter observability requires an accepted runtime association and its
  frame baseline resets on runtime-ID replacement. During causal training
  only, clip the complete RGB observation module locally before the separate
  interaction and global caps, reconstruct/log the true raw norm, and restrict
  global perception adaptation to the first 512 causal updates. A pooled
  validation with no physical support retains raw counts, writes an explicitly
  unsupported rejected artifact, and establishes no deployable incumbent.
- **Alternatives considered:** tighten the birth confidence alone; shorten
  track lifetime; accept improved conditional RMSE despite duplicate tracks;
  post-filter Hungarian results; hardcode slot-order association globally;
  freeze all perception at the phase boundary; lower the shared learning rate.
- **Consequences:** No tentative proposal becomes predictive physical state,
  invalid edges cannot suppress valid assignments, and ROI evidence cannot
  corrupt another identity. Simulator visibility or a newborn cannot fabricate
  parameter evidence. Perception retains a bounded causal adaptation window
  without erasing smaller dynamics/filter gradients. These changes define a
  new validation/optimization protocol, so the stopped qualification cannot be
  resumed or compared as a continuation. A fresh broad qualification is still
  required; the repair is not itself an accuracy or convergence claim. The
  clean committed host smoke confirmed finite hybrid MPS/CPU execution and
  correct guardrail rejection, but its two episodes remain wiring evidence.

## ADR-072 — Require causal validation opportunity and one-shot training launches

- **Date:** 2026-08-03
- **Status:** accepted and implemented
- **Context:** The supposedly active repaired qualification never trained. Its
  only metrics row was step-zero initialization validation; all four fixed
  `impulse_perturbation` episodes lacked one-second deterministic support under
  an external-impulse probability of `0.12` sampled every observation
  interval. The resulting assertion terminated the trainer. `launchctl
  submit` had inferred KeepAlive behavior and restarted the same command more
  than 2,284 times; later attempts hit the occupied run directory. The
  evaluator also scored deterministic errors across hidden interventions while
  training correctly censored them, and pooled binary support could rest on
  one matched object.
- **Decision:** Interpret and document external-impulse probability at its
  actual per-observation cadence. Use `0.02` for the stochastic scenario,
  retaining real surprises while the fixed manifest preserves clean windows.
  Censor deterministic trainer and evaluator point/event/correction metrics
  scene-wide after unseen actuation, while leaving forecast likelihood and
  calibration stochastic. Require positive per-scenario, per-horizon floors
  for label-only predictable targets and matched targets, plus at least two
  independently supported episodes in the v3 profile. Persist exact support
  counts and include the resolved scenario generator parameters, simulator
  version, metric version, and floors in protocol hashes. Treat only the
  explicit insufficient-support exception as a rejected candidate; metric
  schema errors remain fatal. Recompute every retained derived selector field
  from its exact additive evidence before accepting checkpoint provenance.
  An unsupported imported initialization is persisted and training continues,
  with broad validation retried at the configured cadence rather than before
  every optimizer update. Persist an incomplete-reference-comparison marker
  across exact-resume branches; the first later supported candidate establishes
  the fixed reference without promoting itself. Fresh CLI runs write
  starting/failed/completed state artifacts and take an exclusive per-run
  lock. New macOS training launches use an explicit one-shot LaunchAgent plist
  with `KeepAlive=false`; the legacy convergence supervisor consumes terminal
  trainer state, verifies monitored PID command identity, and removes its
  initial job after verified completion or failure.
- **Alternatives considered:** label the restart storm as ordinary launch
  health; remove the stochastic scenario; score unknowable post-impulse point
  targets; accept any nonzero pooled denominator; keep retrying full
  validation before each update; retain inferred launchd KeepAlive.
- **Consequences:** A process listing or occupied directory can no longer be
  mistaken for learning progress. The exact fixed validation slice now has
  three supported impulse episodes out of four, predictable target counts
  `116/98/80/62/47`, and matched target counts `40/30/23/16/12` across
  `0.1/0.25/0.5/0.75/1.0 s`, passing the configured `4`, `2`, and `2`
  floors. One seed remains truthfully unsupported. This changes simulator and
  validation semantics (`sphere_world_v4`, rollout protocol 10, selection
  metric 6); older runs remain evidence but cannot be exact-resumed into this
  protocol. The production-profile CPU smoke performed one finite supported
  causal update and then rejected its apparent multihorizon position gain
  because coverage, events, axes, and scenario slices regressed. The host MPS
  smoke performed one finite clipped optimizer update and completed terminal
  validation. Both are wiring/integrity evidence, not convergence or
  deployment promotion.

## ADR-073 — Count complete cadence frames and make long validation observable

- **Date:** 2026-08-03
- **Status:** accepted and implemented
- **Context:** The first protocol-v10 qualification appeared alive but emitted
  no stdout, metrics, checkpoint, or structured progress for roughly 44
  minutes. Process inspection proved active CPU dynamics rather than a
  deadlock. The independent numerical audit then found that
  `global_every_steps=3` actually emitted
  `GLOBAL, FAST, FAST, FAST, GLOBAL`, contradicting ADR-050's intended two
  intervening ROI updates. Every historical report labelled cadence three had
  therefore measured cadence four. The same audit found that configuration
  accepted zero cadence, unused training workers prefetched during initial
  validation, and post-Adam or stored checkpoint tensors were not explicitly
  checked for nonfinite corruption.
- **Decision:** Define `global_every_steps` as the complete distance between
  global frames and require a positive integer. Cadence three is exactly
  `GLOBAL, FAST, FAST, GLOBAL`; test the whole sequence. Bump only rollout
  validation protocol 10 to 11 because measurement semantics, simulator
  semantics, and selector formula are unchanged. Preserve full-manifest
  selector atomicity while writing atomic per-episode
  `training_progress.json` heartbeats with split/kind, counts, timings, PID,
  seed/scenario, and protocol hash. Create the deterministic training loader in
  advance but do not start its iterator/workers until the first actual draw.
  After every Adam step, reject nonfinite parameters, moments, or invalid step
  counters. Validate model buffers/weights, optimizer/scheduler state, and
  step counters before atomic save and before load mutates a destination.
- **Alternatives considered:** reinterpret the configuration value as the
  number of allowed fast frames; relabel all profiles as cadence four; retain
  silent atomic validation and infer health from CPU usage; stream partial
  selector scores; start both loader pools at process launch; rely on the next
  forward pass to discover optimizer corruption.
- **Consequences:** The interrupted protocol-v10 run is preserved as a
  zero-update diagnostic and cannot supply convergence evidence. Historical
  cadence-labelled results remain useful only under their actual cadence-four
  behavior. A fresh protocol-v11 qualification is required. Validation remains
  scientifically atomic but operationally inspectable, and corrupt post-step
  state cannot become a resumable checkpoint. The new CPU causal smoke
  completed one finite update with per-episode progress; the host MPS
  measurement smoke completed one finite clipped update and checkpoint
  round-trip. Both are wiring evidence, not accuracy promotion.

## ADR-074 — Align integration ticks, reuse causal propagation, and restore launch QoS

- **Date:** 2026-08-03
- **Status:** accepted and implemented
- **Context:** The first protocol-v11 qualification advanced cleanly with no
  stderr or nonfinite state, but its first five closed-loop validation episodes
  averaged `117.380 s` versus `25.305 s` for a matched repaired foreground
  control. The generated LaunchAgent classified explicitly requested training
  as `Background`; macOS consequently reduced observed CPU use from roughly
  `525%` in the control to `100–198%`. Separately, a dynamics call audit found
  two independent numerical-path defects. Float32 20 Hz timestamp differences
  put 22 of 39 nominal six-tick intervals just above an integer ratio, so a
  literal ceiling executed seven 120 Hz belief substeps while simulator labels
  used six. The causal training loop also computed the same deterministic
  prior once for supervision and again inside `ingest`, including interval
  dynamics and event work both times.
- **Decision:** Do not emit launchd `ProcessType=Background`; use its portable
  Standard/default classification while retaining `KeepAlive=false` and
  `caffeinate`. Select belief substep counts with a dtype-aware near-integer
  snap only when the ratio is indistinguishable at timestamp precision, and
  otherwise retain the ceiling. Bump rollout validation protocol 11 to 12.
  Expose a typed one-use prepared propagation from `OnlineWorldModel` so the
  exact prior inspected by training is consumed by ordinary ingestion after
  strict source/timestamp/device and dynamics parameter/buffer/mode revision
  checks, with original `dt` and interval collision evidence intact. The
  zero-copy mutation guard deliberately supports autograd/`no_grad` rather
  than `inference_mode`. Permit training rollouts to omit unused trajectory-
  auxiliary stacking while preserving the public complete default.
- **Alternatives considered:** interpret the fourfold slowdown as model
  collapse; reduce validation anchors or scenario coverage; leave the
  float-dependent 6/7 grid because elapsed time remained conserved; set
  runtime state to the prepared prior and ingest at zero `dt`; classify the
  job as launchd `Interactive`; batch validation anchors before proving
  selector parity.
- **Consequences:** The stopped protocol-v11 run remains truthful zero-update
  throughput evidence, not an accuracy result. Protocol-v12 candidates require
  fresh selector/reference validation. The simulator and belief model now
  share the intended tick grid, interval events remain causal, and training
  avoids one redundant noninitial propagation without bypassing the persistent
  predict–observe–associate–innovate–correct loop. A matched timing check and
  reduced production-model optimizer smoke passed; the smoke candidate was
  correctly rejected for a slight validation regression. A full-manifest
  sustained qualification is still required before any convergence claim.

## ADR-075 — Freeze executable source through the supervised exact-resume campaign

- **Date:** 2026-08-04
- **Status:** accepted for the active campaign; post-campaign hardening pending
- **Context:** A live step-6144 audit found no defect exercised by the current
  trainer. It did identify a redundant fail-closed check that should eventually
  require `matched_slots=true` before a fast-ROI row can receive positive crop
  evidence even if a stale caller supplies a nonnegative target index. Current
  production callers already replace rejected indices with `-1`. The audit
  also found that `training_state.json` remains `starting` throughout a live
  invocation even though detailed `training_progress.json` heartbeats are
  correct. Either edit would change executable source while the convergence
  supervisor may still need to exact-resume the step-16,384 checkpoint.
- **Decision:** Do not change `train.py` or `world_model/*.py` until the active
  supervisor records `plateau`, `limit_hit`, or a terminal failure. Preserve
  the exact runtime-source fingerprint
  `43eaaea369ac13a430b2efff224b7f88db973f0a133593966326c095cb16c330`.
  Permit documentation and test-result updates because they are excluded from
  the numerical runtime fingerprint. Queue the fail-closed ROI helper check
  and the explicit running-state artifact as the first post-campaign
  hardening, each with focused tests.
- **Alternatives considered:** patch the live worktree and let the extension
  fail source verification; weaken exact-resume provenance; stop a healthy
  trainer to relaunch under changed semantics; create a second hidden source
  checkout for extensions.
- **Consequences:** The active initial segment and any supervisor extension
  remain numerically comparable and exactly resumable. The two low-risk
  hardening items remain explicit rather than being silently lost, but neither
  is misrepresented as the cause of the current fast-ROI regression. New
  runtime improvements must begin from a new timestamped initialization after
  the campaign decision, not mutate this campaign in place.

## ADR-076 — Preserve scientific evidence during conservative artifact cleanup

- **Date:** 2026-08-04
- **Status:** accepted
- **Context:** The working tree contained 3.0 MiB of regenerable Python,
  pytest, Ruff, and editable-install caches, one empty demo directory, 2.0 GiB
  of run artifacts, and 22 MiB of nonempty demos. The run tree includes the
  live convergence campaign, the checkpoint it was initialized from,
  accepted baselines, rejected controls, and reports cited by the accuracy
  audit. A static reference scan also found two presently uncalled evaluation
  helpers, but both are named in the authoritative specification's required
  repository structure. The supervisor may still exact-resume the active
  campaign and verifies every `train.py` and `world_model/*.py` byte.
- **Decision:** Remove only reproducible caches, generated editable-install
  metadata, and genuinely empty artifact directories during the campaign.
  Quarantine them under a timestamped `/private/tmp` path so the cleanup is
  immediately recoverable. Retain every nonempty run/demo artifact unless a
  later audit proves that it is neither an active dependency nor scientific
  evidence. Do not use broad `git clean -fdX`, because `runs/` and
  `demo_outputs/` are intentionally ignored. Do not delete tracked code based
  on static call count alone; first reconcile it with `PROJECT_SPEC.md`, then
  make any executable simplification only after the supervisor is terminal.
- **Alternatives considered:** delete all ignored files; prune checkpoint
  tensors from rejected runs while keeping reports; delete every module with
  no current static caller; postpone even cache cleanup.
- **Consequences:** The repository loses disposable clutter without risking
  the running process, exact continuation, accepted checkpoints, or truthful
  historical evidence. The active runtime fingerprint remains unchanged.
  Large experiment artifacts remain visible by design rather than being
  silently discarded for cosmetic disk savings. A later destructive retention
  policy requires an explicit evidence manifest and a terminal campaign.

## ADR-077 — Separate deployment support from mutable optimizer viability

- **Date:** 2026-08-06
- **Status:** accepted; rollout validation protocol 13
- **Context:** In the protocol-12 full campaign, all six causal validations at
  steps 8,704–11,264 restored the accepted step-zero incumbent and reset Adam.
  Each candidate had finite pooled current/horizon evidence, and the raw
  step-10,240 score (`0.329669`) improved on the fixed reference
  (`0.3310606914`), but a scenario or reference-relative coverage guardrail
  failed. The old Section 173 wording treated every later deployment-support
  failure as catastrophic mutable support collapse. Therefore causal learning
  could never accumulate for more than the 512-update validation interval,
  contradicting Section 164's separation of safe deployment from repairable
  mutable optimization.
- **Decision:** Keep absolute, reference-relative, per-scenario, and broad
  guardrails for deployment selection. Define catastrophic mutable viability
  more narrowly: the candidate must be structurally valid/finite and retain
  pooled current and every configured forecast-horizon coverage above the
  absolute handoff floors. A scenario-only, reference-relative, or broad-score
  failure rejects promotion but preserves candidate tensors and optimizer
  history. Invalid/nonfinite state terminates fail-closed; only absolute pooled
  support collapse in a well-formed candidate restores the verified incumbent
  and resets Adam. Persist both failure sets and use mutable viability, not
  deployment eligibility, when interpreting raw convergence candidates.
- **Alternatives considered:** remove scenario guardrails; lower the
  `elastic_pairs` floor until every candidate passes; retain blanket rollback
  but lengthen validation intervals; always train from rejected tensors but
  hide the distinction in checkpoint metadata.
- **Consequences:** Deployment safety is unchanged, while causal updates can
  repair the scenario that caused rejection. Protocol-12 selector artifacts
  are not comparable or exact-resumable under protocol 13. A new broad
  qualification must prove both sustained optimizer-history accumulation and
  eventual multi-scenario promotion; this decision itself is not an accuracy
  result.

## ADR-078 — Treat long-run resource and external-process state as training integrity

- **Date:** 2026-08-06
- **Status:** accepted
- **Context:** The protocol-12 trainer and supervisor disappeared during a
  system-wide macOS memory-pressure storm. Unified logs report
  `OS_REASON_JETSAM` at `2026-08-06 01:01:39.691`. The run used four loader
  workers with default two-batch prefetch and crossed from MPS to CPU without
  explicitly releasing allocator cache. Because the kernel killed the process
  rather than raising Python, `training_state.json` remained stale and no
  terminal trainer failure existed.
- **Decision:** Sustained macOS profiles default to a low explicit worker count,
  one prefetched batch per worker, and non-persistent workers. Phase transitions
  move/reset the model, collect Python garbage, and empty the allocator cache
  for the previous MPS/CUDA device. Metrics include process maximum RSS.
  `train.py` writes a live `running` state before trainer entry. The convergence
  supervisor writes an `ExternalTrainerExit` terminal artifact to the ordinary
  training-state/failure/history contract whenever it proves that a monitored
  trainer vanished, including extension subprocesses.
- **Alternatives considered:** assume unrelated applications alone caused the
  kill; reduce model or validation coverage without measuring resident memory;
  use persistent workers; leave the supervisor-only failure as sufficient
  terminal evidence.
- **Consequences:** These controls reduce avoidable resident memory and make an
  OS kill auditable without changing the model, fixed manifest, selector, or
  full-validation atomicity. Maximum RSS is a high-water diagnostic rather
  than instantaneous attribution, so a future campaign still needs host
  monitoring before claiming that memory growth is eliminated.

## ADR-079 — Qualify modular candidates and isolate fast-ROI adaptation

- **Date:** 2026-08-07
- **Status:** accepted implementation policy; accuracy promotion pending
- **Context:** Protocol 13 accumulated more than 6,000 causal updates without
  numerical or resource collapse, but no checkpoint passed the fixed broad
  selector. The full step-4,096 candidate improved pooled RMSE at every
  horizon while reducing coverage. A dynamics-only transplant was worse at
  every horizon; a coherent state+dynamics transplant improved most pooled
  metrics but regressed x and identity. Adding donor fast-ROI tensors while
  retaining the accepted global detector/shared backbone produced the best
  diagnostic score (`0.2909420` versus `0.3296688`) and improved every
  horizon, but still failed identity, z, coverage, and scenario guardrails.
  Importing or interpolating shared/global perception caused severe coverage
  loss. The existing `state_dynamics_roi` training scope continued to update
  shared backbone stages after global-only heads were frozen.
- **Decision:** Make module boundaries executable in offline qualification via
  schema-checked, provenance-recorded checkpoint composition. Add a narrower
  `state_dynamics_fast_roi` training scope: dynamics, belief updater,
  identifier, ROI updater, and ROI-only fast projection may adapt, while all
  shared backbone stages, global pyramid projections, and the detector remain
  frozen. Start the new campaign weights-only from the accepted reference and
  set the global-adaptation duration to zero. Keep all existing deployment
  guardrails unchanged.
- **Alternatives considered:** relax per-scenario guardrails; promote the best
  pooled modular candidate; continue the coupled run indefinitely; use
  dynamics-only transfer; interpolate the complete checkpoint; keep training
  the first two shared backbone stages after global heads freeze.
- **Consequences:** Future causal learning cannot silently move global RGB
  discovery through a supposedly fast-ROI scope. Rejected modular artifacts
  remain reproducible scientific evidence, not accepted baselines. Acceptance
  still requires a complete long run and the exact balanced selector, so this
  decision is a targeted training correction rather than a claim of success.

## ADR-080 — Exclude frozen global discovery from the fast-ROI objective

- **Date:** 2026-08-08
- **Status:** accepted; corrected qualification pending
- **Context:** The frozen-backbone campaign correctly reported
  `global_perception_trainable=0`, and tensor comparison proved every shared
  and global RGB parameter remained unchanged. Its loss assembly nevertheless
  tested all parameters below the backbone container. The ROI-exclusive
  `fast_projection` remained trainable, so the code misclassified the global
  discovery loss as trainable. A representative step logged global loss
  `5.287398` and fast-ROI loss `0.050371`, then optimised their average
  `2.668884`; the global term had no gradient path while dominating the
  reported scalar loss. ADR-081 separately corrects the later accidental
  reallocation of its fixed coefficient.
- **Decision:** Determine global-loss trainability only from the global
  detector, shared backbone stages, and global pyramid projections. Retain a
  frozen global measurement as a diagnostic, but exclude it from the
  measurement objective whenever none of those components is trainable. Test
  the trainability predicate and a complete causal batch under
  `state_dynamics_fast_roi`.
- **Alternatives considered:** ignore the constant because it has zero
  derivative; compensate by doubling the learning rate; remove all global
  diagnostics; continue the existing run and reinterpret its scalar loss.
- **Consequences:** The fast-ROI objective now has its declared scale and the
  reported total reflects trainable behavior. The interrupted 4,744-update
  campaign remains useful diagnostic evidence but cannot prove convergence.
  Because this is an objective change, training restarts weights-only from the
  same accepted reference under specification 1.14.

## ADR-081 — Preserve branch coefficients and stage fast ROI before state dynamics

- **Date:** 2026-08-08
- **Status:** accepted implementation policy; qualification pending
- **Context:** The specification-1.14 run was finite and supported but its
  first 512-update candidate regressed every horizon (`0.3749701` versus
  `0.3296688`), led by x RMSE (`0.4224541` versus `0.3300525`). Correctly
  removing frozen global discovery exposed a second bug: when only fast ROI
  remained, loss assembly renormalized it from coefficient `0.5` to `1.0`.
  An exact fast-ROI-only transplant from that candidate reproduced most of the
  damage at score `0.3602169`. Conversely, a fast-ROI-only transplant from the
  earlier fixed-half-weight step 512 improved score to `0.3110033`, current
  position RMSE to `0.2509520`, and every axis/horizon, while slightly
  regressing velocity and coverage and still failing scenario guardrails.
- **Decision:** Combine global and fast measurement objectives over the fixed
  denominator `1 + fast_roi_pretrain_weight`, treating an absent/frozen branch
  as zero rather than reallocating its coefficient. Add `fast_roi` as an
  explicit scope and an optional paired late scope plus exact causal-update
  transition. The next campaign uses 512 fast-ROI-only updates followed by
  `state_dynamics`, preserving the accepted deployment incumbent throughout.
- **Alternatives considered:** keep the doubled fast coefficient; lower the
  learning rate globally; promote the rejected modular candidate; manually
  splice checkpoints; continue training every module indefinitely; freeze all
  learned state after the early ROI gain.
- **Consequences:** Auxiliary gradient scale is invariant to support and
  trainability. The staged campaign tests the observed localization gain while
  giving velocity/coverage repair a disjoint later phase. Scope and boundary
  are exact configuration semantics, and the intermediate ROI state remains
  unaccepted until the ordinary broad selector passes.

## ADR-082 — Keep measurement auxiliaries perception-local

- **Date:** 2026-08-09
- **Status:** accepted; corrected qualification pending
- **Context:** Protocol 15 was finite and obeyed its parameter freeze boundary,
  but four late candidates formed a failed accuracy plateau. Exact module
  ablations showed both late dynamics and late updater/identifier tensors
  damaged long-horizon forecasts. Although every RGB parameter was frozen in
  `state_dynamics`, the fast ROI is prior-conditioned, so its measurement loss
  still required gradients through the propagated prior and silently trained
  the physical stack. The scope froze weights but not the unintended
  cross-module objective path.
- **Decision:** Classify fast-measurement trainability only from shared fast
  encoder stages, the ROI-exclusive projection, and the ROI updater. If none
  is trainable, detach and log `frozen_fast_measurement` plus its component
  diagnostics, exclude the branch from the optimized measurement total, and
  omit it from causal fast-support terms. State/dynamics remain supervised by
  their explicit physical objectives.
- **Alternatives considered:** continue because the path is mathematically
  differentiable; reduce the global learning rate; promote step 512 despite
  guardrail failures; train dynamics and perception jointly again; remove fast
  diagnostics entirely.
- **Consequences:** Trainable scopes now isolate both tensors and objective
  ownership. Protocol-15 weights cannot validate the corrected objective; the
  next campaign starts weights-only from the same accepted reference under
  specification 1.16 and unchanged broad selection. Checkpoint metadata is
  synchronized to the same specification version and tested against the
  authoritative document header.

## ADR-083 — Isolate rollout uncertainty from forecast-mean learning

- **Date:** 2026-08-09
- **Status:** accepted; corrected qualification pending
- **Context:** Protocol 16 verified perception-local loss routing and remained
  finite through update 552, but a full objective trace found rollout Gaussian
  NLL consumed the live mean error while the deterministic per-axis/horizon
  point loss already supervised the same trajectory. Low predicted variance
  could therefore duplicate and amplify a point-mean gradient. The NLL also
  retained this path when deterministic targets were correctly censored after
  an unseen external actuation, teaching a mean for a causally unidentifiable
  outcome. State uncertainty had already detached its mean error, so rollout
  behavior was inconsistent with the declared calibration contract.
- **Decision:** Detach rollout mean error inside Gaussian NLL. Realised future
  outcomes train forecast variance through likelihood; only identifiable
  deterministic rollout point losses train the trajectory mean. Preserve NLL
  after hidden actuation solely as a variance-calibration signal and directly
  regress both absent mean gradient and finite widening variance gradient.
- **Alternatives considered:** retain both gradients and tune the NLL weight;
  clamp inverse variance more aggressively; drop rollout NLL entirely; allow
  hidden interventions to supervise the most likely deterministic response;
  continue protocol 16 and reinterpret its loss.
- **Consequences:** Forecast mean and uncertainty objectives have explicit,
  non-overlapping ownership. Protocol-16 weights cannot prove convergence
  under the corrected objective, so its trainer and supervisor remain stopped
  and protocol 17 starts weights-only from the same accepted reference under
  specification 1.17 and the unchanged broad selector.

## ADR-084 — Require repeated broad evidence before decoupling interaction objectives

- **Date:** 2026-08-09
- **Status:** accepted; protocol-17 continuation active
- **Context:** Protocol 17's step-1,024 candidate improved z, collision, and
  tracking behavior while regressing x/y and medium/long forecasts. A
  four-batch gradient attribution at the later step-1,408 checkpoint found
  scenario-dependent multi-task conflict in the shared interaction edge trunk:
  event-versus-z trajectory cosine was negative on all four audited collision
  batches, and x/y/velocity conflicts changed with the physical regime. A
  checkpoint-compatible detached event-trunk gradient is technically possible.
  However, the complete step-1,536 validation then recovered current state,
  velocity, tracking, calibration, all current axes, x/y horizons, and four
  joint horizons relative to step 1,024. It remained rejected, but did not
  establish a repeated worsening trend or the declared convergence plateau.
- **Decision:** Preserve the gradient-conflict measurements as a concrete
  follow-up, but do not change objective ownership or forward architecture
  during a recovering sustained campaign. Continue protocol 17 exactly from
  its durable step-1,536 state to the predeclared 8,192 minimum and apply the
  unchanged validation/extension rule. Introduce event/shared-trunk
  decoupling only in a new timestamped protocol if later comparable fixed
  validations again regress or establish a failed plateau. Any such protocol
  must retain a trainable event head, preserve checkpoint-compatible forward
  values where possible, and pass the same broad selector rather than claiming
  success from gradient cosine alone.
- **Alternatives considered:** continue blindly without recording the
  conflict; stop at 1,536 and immediately start a new architecture; remove the
  collision objective; relax scenario guardrails; promote the better pooled
  current-state metrics; treat heterogeneous batch loss as convergence.
- **Consequences:** The user-requested sustained training receives adequate
  duration, while a measured negative-transfer mechanism remains available if
  later evidence warrants intervention. Step 1,536 remains a rejected
  numbered checkpoint; the exact resumed trainer and supervisor preserve
  optimizer, RNG, data draw, source fingerprint, and convergence semantics.

## ADR-085 — Exact in-place resume must not inject a run name

- **Date:** 2026-08-09
- **Status:** accepted and implemented
- **Context:** `train.py` deliberately treats `--resume` without `--run-name`
  as an in-place optimizer/RNG continuation from the source run's exact
  `checkpoints/last.pt`. The macOS one-shot helper nevertheless required and
  always emitted `--run-name`. With a run-local resolved config this could
  create a nested sibling run; with the original config it correctly failed
  on the occupied target directory. Neither attempt updated the authoritative
  checkpoint, but the mismatch made the supported persistent-launch path
  unable to express the trainer's exact-resume contract.
- **Decision:** Make the launch helper's run name optional. Require it for a
  new run or weights-only initialization, but omit the argument entirely for
  exact resume and derive only default log filenames from the resume run
  directory. Keep `train.py` as the final validator that exact resume points
  to `checkpoints/last.pt`.
- **Alternatives considered:** weaken `train.py` to accept an existing named
  directory; reuse the run-local config and reinterpret nested artifacts;
  launch an unmonitored background shell; discard the failed-attempt evidence.
- **Consequences:** Persistent macOS launch now faithfully represents both new
  and exact-resume workflows. Failed-attempt state remains auditable, the
  accidental nested artifacts are quarantined rather than counted, and the
  active protocol continues from the unchanged step-1,536 model, optimizer,
  RNG, and data-draw state.

## ADR-086 — Do not infer optimizer progress from raw append-only row counts

- **Date:** 2026-08-09
- **Status:** accepted monitoring policy; attempt-aware logging pending
- **Context:** The stopped protocol-17 process wrote a step-1,544 training row
  after its durable step-1,536 validation checkpoint. Exact resume correctly
  restored checkpoint state and replayed draw 1,544, producing a second row.
  Apart from elapsed time, finite-check duration, and process RSS, both rows
  are identical. Rewriting the active append-only artifact would discard
  useful crash/restart evidence, while changing numerical runtime source
  during an exact campaign would invalidate later continuations.
- **Decision:** Preserve both rows and canonicalize repeated `(split, step)`
  records in manual training-dynamics audits. Convergence and promotion must
  continue to consume only tensor/protocol-hash-verified numbered validation
  checkpoints and terminal summaries. Add explicit attempt/resume generation
  metadata after the active campaign reaches a terminal decision.
- **Alternatives considered:** count every JSONL line as an update; delete the
  first row; truncate the metrics file at every resume; change the running
  protocol's runtime source; treat the deterministic replay as model wobble.
- **Consequences:** The optimizer/RNG trajectory stays exact and all restart
  evidence remains auditable. Raw line counts are not authoritative progress;
  the checkpoint step, data-draw invariant, unique logged steps, process state,
  and fixed validation checkpoints are.

## ADR-087 — Separate live optimizer health from fixed-validation convergence

- **Date:** 2026-08-09
- **Status:** accepted and implemented
- **Context:** Per-batch loss and gradients vary substantially across the
  balanced collision/camera/damping curriculum, and an exact resume may
  preserve then replay an uncheckpointed logged tail. Manual line counts or a
  smooth-loss expectation can therefore misdiagnose both healthy hard batches
  and duplicated telemetry as collapse. Conversely, waiting only for the next
  32-episode validation can leave genuine finite/support/scope failures
  undetected for hours.
- **Decision:** Use a read-only deterministic audit for live health. Count the
  latest row for each `(split, step)`, require duplicate replay rows to agree
  in every model/data metric except process timing and RSS, and fail on
  numerical, optimizer, causal-support, objective-support, gradient,
  frozen-scope, or data-draw invariant violations. Report training
  lifecycle/identity/event/uncertainty, correction, observability, axis, and
  horizon distributions together with recovery-perturbed versus clean pooled
  identity rates and fixed-validation axes/horizons, while reserving
  convergence and promotion exclusively for tensor-verified complete
  validation checkpoints and the declared plateau selector.
- **Alternatives considered:** infer convergence from smoothed training loss;
  ignore training until validation; delete duplicate rows; treat every JSONL
  line as a distinct optimizer update; weaken the fixed broad selector.
- **Consequences:** Numerical collapse can be caught promptly without
  mistaking curriculum variance for failure, and restart telemetry remains
  auditable. A passing health report proves only that optimization is
  functioning; it cannot promote weights or establish accuracy convergence.

## ADR-088 — Reject protocol 18 and anchor correction to typed innovation

- **Date:** 2026-08-09
- **Status:** accepted and implemented; broad qualification pending
- **Context:** The balanced protocol-18 optimizer remained finite and fully
  supported through step 128, but exact fixed validation worsened current state
  and every forecast horizon. Dynamics-only was nearly neutral, whereas
  updater-only reproduced the full regression. Inspection showed that the
  learned corrector reduced camera-space innovation to pooled statistics and
  then applied arbitrary deltas to all packed fast-state and variance fields.
  It therefore lacked the axis/sign evidence needed for its outputs and
  ignored the modality's declared state support.
- **Decision:** Stop and reject protocol 18. Add explicit opt-in
  innovation-anchored semantics: form whitened world-position and supported
  velocity evidence per associated pair, treat learned outputs as bounded
  gains on that evidence, apply per-axis measurement/surprise confidence, and
  mask learned mean and variance changes to supported components. Zero
  innovation produces zero learned mean movement. Preserve unanchored legacy
  semantics as the default for historical checkpoint reproduction; the next
  config opts in and starts weights-only.
- **Alternatives considered:** train protocol 18 longer; lower its learning
  rate; remove the state gate; train dynamics only; relax the elastic-pair
  guardrail; scale the same updater with a larger network.
- **Consequences:** The correction path again satisfies the measurement support
  contract and retains learnable context-dependent positive or negative gains.
  Focused tests pass. The first exact qualification proved that directly
  loading old absolute-delta heads under the gain interpretation is a mixed,
  invalid transfer: the pooled score improved slightly, but velocity, short
  horizons, and several scenarios regressed, and one finite balanced update
  worsened every x metric. Exact reset ablations then showed that only the mean
  head should reset: resetting mean/variance/gate worsened score to `0.350730`,
  while mean-only reached `0.324176`. Both are rejected for deployment, but the
  mean-only candidate is the typed mutable start for updater-only recovery
  before dynamics are unfrozen.

## ADR-089 — Scale transformers over explicit predictive abstractions

- **Date:** 2026-08-09
- **Status:** accepted architecture direction; implementation gated on ADR-088
- **Context:** Modern Transformers provide content-dependent token interaction
  and scalable parallel training; Perceiver-style bottlenecks handle dense
  multimodal inputs, and JEPA-style feature prediction scales self-supervision.
  Contemporary video world models also show strong distributed physical
  representations. None of that removes Orpheus's need for fast explicit
  trajectories, uncertainty, identity, lifecycle, online correction, and
  measurable long-horizon state accuracy. The current laptop also cannot
  establish a credible path by jumping directly to foundation-model scale.
- **Decision:** Scale through derived entity, relation, event, scene/camera,
  and bounded-history tokens while `WorldBelief` remains authoritative. Begin
  only after the corrected small control qualifies. The Mac pilot adds 2--4
  pre-normalized width-128/four-head blocks and typed residual decoders, then
  compares them with the accepted model and a parameter-matched graph control.
  Wider single-GPU and foundation-video rungs follow only with commensurate
  data and fixed disjoint generalization evidence.
- **Alternatives considered:** replace the runtime with a video generator;
  tokenize all RGB history autoregressively; immediately increase every hidden
  width; use training loss as the scaling gate; defer all attention work until
  cloud compute is available.
- **Consequences:** Local work tests the architecture and scaling law rather
  than pretending the Mac can train a foundation model. Every rung records
  parameters, data exposure, throughput, memory, plateau evidence, and broad
  RGB-only/OOD metrics. Attention proposes typed updates and cannot bypass the
  analytic/filtering contracts.

## ADR-090 — Recover a semantically reset correction head in isolation

- **Date:** 2026-08-10
- **Status:** accepted after exact step-64 rejection
- **Context:** The innovation-anchored migration changed only the learned mean
  output from an absolute delta to an innovation gain. The first recovery
  scope froze perception, dynamics, and identification but still trained the
  complete updater. Its optimizer was numerically healthy, yet exact step-64
  validation worsened score `0.324176 -> 0.338432`, regressed current position
  and all horizons, and severely damaged elastic-pair x prediction. Tensor
  deltas showed that the compatible trunk and sibling heads changed alongside
  the reset mean head.
- **Decision:** Add an `updater_mean` scope that trains exactly the learned
  corrector's mean-head weight and bias. Restart from the clean mean-reset
  checkpoint with effective learning rate `5e-6`; do not continue from the
  rejected updater-wide state. Broadening the scope requires prior fixed-
  manifest evidence that mean-only recovery is useful but capacity-limited.
- **Alternatives considered:** continue the updater-wide run to 512 updates;
  resume from its rejected step-64 iterate; lower learning rate without
  narrowing scope; unfreeze dynamics to compensate; proceed to attention
  scaling despite the regression.
- **Consequences:** The next result isolates whether the new gain head can
  recover useful bias correction while preserving all compatible updater
  behavior. It may learn more slowly, but any fixed-validation change is
  attributable to the intended semantic repair rather than shared-trunk or
  sibling-head forgetting.

## ADR-091 — Reject a scene token whose declared input is dead

- **Date:** 2026-08-10
- **Status:** accepted and implemented; corrected campaign pending
- **Context:** The first typed-attention campaign reached a finite durable
  update-128 checkpoint with all inherited parameters frozen exactly and 48
  attention optimizer states. Exact deltas showed 47 attention tensors had
  changed, while `scene_projection.weight` and its Adam first moment remained
  exactly zero. Its only input was `WorldBelief.global_code`; repository-wide
  use audit showed this reserved field is initialized to zero and never
  corrected by the current RGB runtime. A learned bias/type embedding still
  let the token aggregate objects, but it did not carry the global fields and
  calibrated camera context required by the attention contract.
- **Decision:** Stop and preserve the pilot at its durable step-128 boundary;
  do not resume it or count its updates toward the corrected rung. Derive the
  stage-A scene input from authoritative `WorldBelief`: global code, summaries
  of global uncertainty, gravity, camera transform, linear/angular motion,
  intrinsics, summaries of camera uncertainty, and calibration. Summarize
  covariance vectors so dynamics does not depend on modality-specific packing
  widths. Retain zero-output residual initialization, unordered object-token
  semantics, and attention-only optimization.
- **Alternatives considered:** continue 8,192 updates because the scene bias
  can still aggregate objects; delete the unused scene token; fabricate a
  learned global code outside the filter; resume the trained entity/relation
  stack under a changed scene projection.
- **Consequences:** The corrected Mac rung adds 1,103,626 parameters and has a
  55-value scene input in the current configuration. A focused regression
  proves the scene projection receives finite nonzero gradient even when
  global code is zero. The corrected run restarts weights-only from the same
  protected graph control, so no incompatible optimizer/history state is
  smuggled across the semantic change.

## ADR-092 — Isolate typed-output backpropagation before shared attention

- **Date:** 2026-08-11
- **Status:** accepted and implemented; fresh sustained qualification pending
- **Context:** Collision- and force-decoder parameter-row caps kept optimizer
  updates finite, but they execute only after autograd has populated every
  shared attention gradient. At the deterministic step-280 recurrence, the
  force-row-isolated campaign produced raw total/force norms
  `995.5391/989.7965`, post-row interaction norm `106.7798`, and effective
  total coefficient `0.0010045`; shared projections and blocks had already
  received order-one-to-ten gradients. Finite weights therefore did not mean
  useful shared learning was preserved.
- **Decision:** Add optional per-invocation backward hooks on the raw typed
  node, collision, and joint normal/tangent-force outputs. Cap these semantic
  groups before gradients enter their decoder or the shared stack, then retain
  the existing decoder-row, complete-interaction, and global caps for repeated
  invocation accumulation. Bind all caps into resume/selector protocol
  semantics and report raw/applied output norms, invocation counts, minimum
  and aggregate coefficients separately from later parameter gradients.
- **Evidence:** An explicitly non-promotable branch replayed updates 257--280
  from the durable step-256 optimizer/RNG/sampler state. On the same step-280
  seeds and frames, the later parameter norm fell to `10.8330`, the largest
  shared projection/block norm was `0.0851`, the post-row interaction stage
  retained `0.6979`, and the supported finite update was applied. The offline
  audit passes with a truthful warning for severe localized typed-output and
  force-row coefficients. Step-264 forward metrics remained exactly equal to
  the source; step-272 1-second RMSE remained effectively neutral
  (`0.28163 -> 0.28137`).
- **Alternatives considered:** accept finite but globally suppressed updates;
  keep adding decoder parameter-row caps; lower the whole learning rate;
  remove collision/force objectives; increase model capacity before repair.
- **Consequences:** The repair changes backward conditioning, not forward
  dynamics or checkpoint tensors. Parameter-gradient telemetry is raw only
  relative to the later row/module/global hierarchy because it necessarily
  observes gradients after output conditioning. The replay qualifies a fresh
  weights-only campaign, not its weights, accuracy, generalization, plateau,
  or convergence. Capacity scaling remains gated on that campaign.

## ADR-093 — Isolate accumulated node rows and make optimizer rejection durable

- **Date:** 2026-08-11
- **Status:** accepted, implemented, and causally replayed; sustained selector pending
- **Context:** The fresh impulse-isolated campaign exactly reproduced the
  protected step-zero selector and remained finite/support-complete through
  update 59, but its update-60 complete interaction stage retained only
  `0.0850405`. The pre-Adam fail-fast correctly rejected the update. The
  original failure artifact lacked the gradient hierarchy, and the offline
  auditor looked only at sampled metric rows, so a terminal failed run could
  appear healthy. An instrumented exact replay matched all 400--454 comparable
  model/data fields at every logged update 8--56 and captured update 60. The
  node decoder was `11.6617`, dominated by world-y `11.5014`, while the largest
  shared non-decoder tensor was `0.124876`. Existing force isolation worked;
  per-invocation node clipping had not bounded accumulation across 144 calls.
- **Decision:** Add a joint accumulated x/y/z node-decoder cap before the
  collision/force/impulse and complete-interaction caps, with raw/applied and
  intermediate interaction telemetry and full resume/protocol semantics.
  Persist structured diagnostics on every interaction-retention rejection and
  make the offline auditor fail durable terminal numerical/optimizer
  failures. Keep the rejected trajectory and all replays non-promotable.
- **Alternatives considered:** lower the whole learning rate; raise or remove
  the 10% gate; accept finite normalized updates; enlarge the attention stack;
  cap only the y row; infer the cause from sampled rows without exact replay.
- **Consequences:** Forward values, parameter count, tensors, inference, and
  the typed dynamics contract are unchanged. A one-update reconstruction
  predicts post-row norm `1.81140` and complete-stage retention `0.552059` with
  the configured cap `1.0`. A fresh protected-control replay exactly reproduced
  the initial selector and reached the same update-60 seeds with complete
  support, raw/post-row norm `1.96175/1.76884`, and healthy `0.565343`
  retention; it deliberately stopped before Adam. No capacity rung is
  authorized until the repaired 3.00M control reaches complete selectors and
  plateau.

## ADR-094 — Penalize unsupported attention-node complexity before scaling

- **Date:** 2026-08-11
- **Status:** accepted and implemented; sustained qualification active
- **Context:** The corrected aggregate-gradient campaign's complete step-512
  RGB-only selector is a genuine model rejection rather than corruption or
  optimizer collapse. The latest persisted candidate scores `0.3251911`
  versus the protected `0.3213162`. Exact decoder ablations show that removing
  only the trained node-y row improves current position, x/y/z, velocity,
  collision F1, and all five position horizons relative to the protected
  control, although strict scenario and short-horizon coverage guardrails
  still reject it. At step 512 the x/y/z decoder-row L2 norms are
  `0.01242/0.11143/0.01207`; y therefore holds about `97.6%` of node-row
  squared energy. The vertical residual changes contact timing and can redirect
  structured pair impulses into x, explaining the apparently cross-axis
  failure. Zero initialization and ordinary AdamW decay did not preserve the
  intended inertial bias.
- **Decision:** Add an opt-in `attention_node_complexity` loss equal to the
  mean squared L2 energy of the three identically treated node-decoder rows,
  including bias. Log the aggregate and each axis. Historical configs without
  the exact weight contribute zero. Use a recorded weight of `1.0` for the
  first repaired campaign; at the rejected checkpoint this is a small
  `0.004239` loss with a finite `0.07518` restoring-gradient norm. Preserve all
  forward equations, tensor shapes, residual bounds, attention capacity, and
  the ability for multistep/event evidence to learn acceleration on any axis.
- **Alternatives considered:** permanently zero or freeze world-y; reduce only
  the y learning rate; weaken vertical rollout supervision; enlarge the
  Transformer; relax scenario guardrails; rely on global weight decay; accept
  the no-y diagnostic as a deployment checkpoint.
- **Consequences:** This is a soft complexity prior, not a hardcoded constant-
  velocity law or axis exception. It directly implements the specification's
  requirement that learned residuals pay evidence to rewrite predictable
  inertial motion. Focused schedule/objective/config/checkpoint tests pass
  (`312 passed`). A fresh protected-control smoke and complete fixed-manifest
  campaign remain required; no capacity increase is authorized by the
  diagnostic ablations. The fresh campaign exactly reproduces the protected
  initial selector and passes its first complete 64-update dynamics audit, but
  matched physical deltas are neutral/slightly adverse and no trained fixed
  selector exists yet. The durable step-128 checkpoint subsequently passes
  exact scope/optimizer/finiteness/provenance audit and its matched training
  window improves every pooled position horizon, but trusted identity switches
  rise from `4/699` to `9/703` and selected velocity/event slices regress. The
  scale gate therefore remains closed through fixed selector 512.
## ADR-118 — Use short-step hypothesis rollouts and error-based selection for difficult dynamics

- **Context:** The current attention-only campaign is numerically finite and causally supported, but its latest update (step 1272) is heavily clipped and the last accepted validation candidate still fails broad long-horizon non-regression. Two original Orpheus papers describe a more reliable operating regime: maintain a compact mental state, evolve it in small ordered time steps, and compare multiple minimally different simulations against subsequent reality rather than trusting one long rollout.
- **Decision:** Preserve `WorldBelief` and the typed point/trajectory abstraction, but make the next accuracy rung a receding-horizon hypothesis bank. Each hypothesis shares the analytic state transition and differs only in bounded residual/contact parameters; produce short rollouts, score them with innovation/error and uncertainty, and blend or select the calibrated winner before the next observation. Train short-step transition/innovation losses first, then add longer horizons as consistency checks. Do not replace the persistent filter with an autoregressive history encoder.
- **Rationale:** Small ordered steps reduce compounding event-order errors; a bank prevents one miscalibrated collision hypothesis from dominating; post-observation error provides the same online model-selection signal used by the papers. This directly targets the current x-axis/interaction regressions while retaining modality-independent state and cheap online updates.
- **Guardrails:** Keep the current fixed-manifest selector, per-axis/per-horizon support checks, and protected incumbent. A hypothesis or blend is eligible only with finite uncertainty, complete support, and no regression against the accepted reference. The running attention campaign must finish under its unchanged executable fingerprint; this ADR defines the subsequent weights-only/new-source experiment.
- **Evidence:** AAAI ORPHEUS reports interleaved small-step physics/behavior effects and model assignment by simulation error ([paper](https://cdn.aaai.org/ocs/10371/10371-46146-1-PB.pdf)); the ToM framework recommends multiple minimally different simulations and probabilistic pruning when the model is imperfect ([paper](https://arxiv.org/pdf/1405.5048)).

## ADR-119 — Score nearby mental-simulation ensembles by expected error and fragility

- **Context:** ADR-118's candidate bank compared one rollout per dynamics hypothesis. The two source papers additionally warn that a seemingly successful single simulation may be an isolated outcome of an imperfect world model; they recommend small ordered steps, nearby alternative simulations, and selection over a range of outcomes.
- **Decision:** Add an opt-in asynchronous robust-evidence path to `HypothesisRolloutEngine` and `HypothesisDynamicsPool`. It accepts a same-candidate-order set of nearby short rollout samples, computes the usual real delayed-target score for every sample, and selects from `mean(score) + risk_penalty * std(score)`. It preserves the current one-sample behaviour exactly when `risk_penalty=0`, retains per-axis score dispersion, and updates only the injected pool's evidence weights. `WorldBelief` is never sampled, replaced, or made subordinate to the ensemble.
- **Rationale:** Expected error favours candidates that work across plausible imagined worlds; the optional dispersion term rejects a candidate whose apparent win depends on a brittle perturbation. The current persistent posterior remains the slow memory of model evidence, while real RGB-derived delayed targets remain the sole source of correction.
- **Guardrails:** This capability is opt-in and is not yet promoted in an RGB report. Candidate order must match across samples; all samples must have finite scores; lifecycle, event, uncertainty, identity, and all-axis/per-horizon regression checks remain mandatory. The runtime wrapper explicitly leaves belief state unchanged.
- **Evidence:** The AAAI paper assigns models from simulation-vs-reality error and interleaves effects in small steps ([paper](https://cdn.aaai.org/ocs/10371/10371-46146-1-PB.pdf)); the ToM paper recommends evaluating a range of minimally different simulations to prune isolated false success ([paper](https://arxiv.org/pdf/1405.5048)).

## ADR-120 — Preserve interrupted selector evidence and restart numbered checkpoints as new runs

- **Date:** 2026-08-14
- **Status:** accepted and active
- **Context:** The first 128-step attention campaign completed the expensive,
  deterministic 32-episode step-zero validation before a machine/session
  restart interrupted the process. It left a valid
  `validation_step_000000.pt`, but no exact `last.pt`. The trainer deliberately
  rejects in-place resumption from a selector or numbered checkpoint because
  that would blur provenance and optimizer-history semantics.
- **Decision:** Preserve the interrupted directory and all its selector
  artifacts unchanged. Resume its numbered checkpoint only into a new,
  timestamped run using `--run-name`, while retaining the exact source path in
  run state and durable Aqua launch-agent logs. Treat the source validation as
  a frozen control and the continuation as a fresh candidate; neither may be
  promoted without complete fixed-manifest guardrails.
- **Consequences:** Restart recovery is auditable and does not overwrite an
  incomplete experiment. A session interruption cannot be misreported as
  numerical failure, convergence, or a missing validation. The extra initial
  validation cost is accepted in exchange for explicit provenance.

## ADR-121 — Repair typed residual axes without cross-axis optimizer drift

- **Date:** 2026-08-14
- **Status:** accepted for controlled qualification
- **Context:** The completed 128-update all-mixture attention campaign improved
  x and y as well as its aggregate selector score, but regressed z rollout
  accuracy. Its unrestricted `attention` scope updates all three rows of the
  typed node-acceleration decoder together, so a z recovery could overwrite
  the coordinates that already improved.
- **Decision:** Add opt-in `attention_node_x`, `attention_node_y`, and
  `attention_node_z` closed-loop scopes. Each exposes only the selected typed
  output row; before each AdamW update, excluded rows and their optimizer
  moments are cleared, snapshotted, and restored exactly afterwards. The
  shared transformer and all non-node heads remain frozen.
- **Rationale:** Coordinate-wise training is a controlled diagnostic, not an
  assumption that the world is factorized. Analytic integration, structured
  interactions, and the runtime all remain joint; this simply prevents an
  evidence-backed repair of one residual component from silently perturbing
  other qualified outputs through decoupled weight decay or stale moments.
- **Guardrails:** The scope is training-only and opt-in. Promotion still uses
  the identical all-axis, all-horizon, lifecycle, identity, event, and
  calibration comparator; improving z alone is insufficient. Unit tests prove
  x/y rows remain bitwise unchanged under AdamW while z updates.

## ADR-122 — Treat zero-tangent contact as a finite physical state on MPS

- **Date:** 2026-08-14
- **Status:** accepted
- **Context:** An active-Aqua MPS typed-attention propagation exposed a NaN in
  the contact resolver before uncertainty propagation. At a non-collision
  plane contact, the solver computed a friction direction by dividing an exact
  zero tangential vector by `clamp_min(1e-7)`. The later collision multiplier
  was false, but the affected MPS kernel flushed the subnormal denominator and
  formed `0/0` first. CPU arithmetic had hidden this defect.
- **Decision:** Use a shared safe tangent-direction helper in plane and pair
  solvers. It uses the true speed for physically meaningful directions and a
  unit denominator for zero/near-zero speed, yielding an exact zero direction
  at rest. No contact decision, restitution law, friction coefficient, or
  learned residual is changed.
- **Consequences:** The CPU solver remains parity-tested and the active-Aqua
  MPS zero-tangent regression is finite. This removes one backend-specific
  numerical defect but does not qualify the complete RGB/MPS backward path or
  alter the CPU fallback / accuracy promotion guardrails.

## ADR-123 — Keep training-only identity pooling exact while avoiding MPS reduction failure

- **Date:** 2026-08-14
- **Status:** accepted for MPS compatibility
- **Context:** After the contact normalization repair, the ordinary MPS
  predict–observe correction was finite, but the full training loop still
  failed while converting accepted observations into persistent target IDs.
  The helper used an integer `amax` across the small belief-slot dimension;
  the affected MPS build attempted to compile its NaN-propagating reduction
  companion despite this bookkeeping path being non-differentiable.
- **Decision:** On MPS only, compute the same candidate-ID maximum as a
  sequential series of elementwise `maximum` operations. CPU and CUDA retain
  the direct `amax`. This preserves the sentinel `-1` and duplicate-candidate
  semantics exactly, while keeping `WorldBelief`, association, identity, and
  parameter-observability contracts unchanged.
- **Consequences:** The active-Aqua MPS identity test and a bounded complete
  RGB closed-loop forward/backward smoke are finite. This is a numerical
  backend compatibility route, not a model or accuracy change; the full
  attention-pilot MPS run and all promotion guardrails remain required.

## ADR-124 — Qualify MPS causality with the actual bounded attention-pilot graph

- **Date:** 2026-08-14
- **Status:** accepted
- **Context:** Individual MPS primitives, contact propagation, and
  training-time identity pooling were finite after ADR-122 and ADR-123, but
  those checks could not prove that their composition through the real RGB
  closed-loop graph was safe.
- **Decision:** Retain an active-Aqua MPS integration regression that loads
  `attention_pilot_mps`, forces its closed-loop device to MPS, executes a
  generated RGB episode through predict–observe–associate–correct–rollout,
  and requires finite loss plus a finite typed z-decoder gradient. The test
  uses a one-episode four-step override solely to bound numerical
  qualification cost.
- **Consequences:** The measured graph is MPS-finite on the supplied machine
  (`1 passed in 53.64 s`). This does not change the authoritative
  `WorldBelief` contract, the CPU fallback, training policy, or the full
  fixed-manifest accuracy gate. Larger MPS throughput and accuracy campaigns
  require their own evidence.

## ADR-125 — Require MPS fixed-manifest validation for new attention-pilot candidates

- **Date:** 2026-08-14
- **Status:** accepted
- **Context:** The historical attention-pilot configuration retained a CPU
  closed-loop fallback while MPS causal defects were being isolated. After
  ADR-124, leaving that default in place would allow a future candidate to be
  selected without the requested active-Aqua MPS validation.
- **Decision:** Set `attention_pilot_mps` closed-loop preference to `mps` for
  newly created runs. Do not mutate a live run's resolved configuration;
  historical CPU-fallback candidates require an explicit MPS guarded replay
  before any promotion.
- **Consequences:** New selector comparisons run on the qualified MPS graph.
  This changes execution placement only, not model weights, loss, simulator,
  `WorldBelief`, or comparator thresholds. CPU remains available through an
  explicit configuration override for diagnosis, never as an implicit
  promotion substitute.

## ADR-126 — Replay legacy candidates on MPS before promotion

- **Date:** 2026-08-14
- **Status:** accepted
- **Context:** The active z-only recovery was correctly launched with an
  immutable CPU closed-loop resolved configuration before MPS qualification.
  Rewriting its live configuration would corrupt experiment provenance, while
  accepting its CPU selector alone would violate the MPS promotion contract.
- **Decision:** Add `scripts/replay_promotion_mps.py`. It loads reference and
  candidate weights into independent MPS runtimes, replays the trainer's
  deterministic validation manifest for both, persists raw metric evidence,
  and uses the identical selector and support guardrails. It exits successfully
  only for an MPS-eligible improvement.
- **Consequences:** Historical CPU-fallback trials remain useful diagnosis
  evidence but cannot silently become an incumbent. The additional replay cost
  is deliberately paid only at a fixed candidate milestone, not during normal
  training polling.

## ADR-127 — Score hypothesis selection before assimilating its delayed target

- **Date:** 2026-08-14
- **Status:** accepted
- **Context:** The heterogeneous-pool evaluator produced candidate rollouts,
  used the matching future target to update its posterior, then reported that
  target-conditioned selection against the very same target. This was useful
  as delayed-evidence fitting diagnosis but not a causal forecast metric.
- **Decision:** Evaluate selected positions, lifecycle, identity, events, and
  blend weights from the pool posterior that existed before current delayed
  evidence is assimilated. Persist post-assimilation selection counts under
  explicit `posterior_*` diagnostic names for studying recovery and candidate
  diversity.
- **Consequences:** Evaluation now follows predict–observe–correct–then
  revise-future order. Reported online selection can be worse than historical
  hindsight-conditioned diagnostic figures, which is expected and must not be
  hidden by reinterpreting old reports as new accuracy evidence.

## ADR-128 — Integrate selectable damped-velocity fallbacks exactly

- **Date:** 2026-08-14
- **Status:** accepted
- **Context:** The heterogeneous pool exposes a damped constant-velocity
  alternative for short-lived drag-like motion. Its former implementation
  decayed velocity but used an undamped position increment, producing an
  internally inconsistent candidate at longer query times.
- **Decision:** Integrate the candidate's `dv/dt = -d v` branch in closed
  form: position advances by `v * (1 - exp(-d * dt)) / d` and velocity by
  `exp(-d * dt)`. Preserve the exact existing `d = 0` constant-velocity
  branch.
- **Consequences:** This strengthens an explicit analytic hypothesis without
  hard-coding simulator parameters or altering the persistent belief, learned
  dynamics, filter, selector, or live experiment. Any measured gain still
  requires the normal causal full-mixture and MPS promotion gates.

## ADR-129 — Diagnose joint pool lock-in separately from causal axis diversity

- **Date:** 2026-08-14
- **Status:** accepted for diagnosis; no runtime-default change
- **Context:** The protected RGB-only eight-episode active-Aqua MPS pool
  comparison used four short-step candidates and causal pre-observation
  selection. The joint selector chose learned 1,179 of 1,184 decisions, but
  its causal axis diagnostic chose analytic alternatives for x, especially at
  longer horizons. Thus a lack of useful joint switching is not evidence that
  all candidate dynamics are indistinguishable.
- **Decision:** Keep the persistent `WorldBelief`, learned joint default, and
  existing calibration/event/lifecycle guardrails unchanged. Treat joint pool
  lock-in as a scored transition-diversity problem: the next pool change must
  establish, on a fixed multi-episode causal protocol, whether joint evidence
  aggregation obscures localized axis/event advantages. It may not promote an
  axis splice or hardcode a constant-velocity override from this diagnostic.
- **Consequences:** Candidate evidence remains delayed RGB-derived
  measurement evidence, not simulator state; all future selection variants
  require the same full fixed-manifest guardrails and MPS promotion evidence.

## ADR-130 — Isolate demonstrated x-axis selection benefit before changing defaults

- **Date:** 2026-08-15
- **Status:** accepted for evaluator/pool configuration; runtime integration remains pending
- **Context:** Exact additive evidence from eight RGB-only MPS episodes shows
  causal axis-aware selection improves x at 0.50/0.75/1.00 s, but the small
  number of y switches worsens y over the same horizons. Z stays effectively
  learned-only. The full joint selector is still locked to learned dynamics.
- **Decision:** Evaluate one immutable x-only (`axes=[0]`) selection protocol
  from the protected checkpoint, retaining learned y/z. Do not alter the
  runtime default, learned weights, belief, candidate set, evidence source, or
  joint event/lifecycle path first. Promotion remains contingent on the full
  all-axis/per-horizon/lifecycle/identity/event/calibration report.
- **Consequences:** This is an evidence-localized selection ablation, not an
  axis-factorized physics claim. It can fail because even an x-only position
  splice may affect tracked object availability, event interpretation, or
  calibrated uncertainty; all such effects remain measured.

  The completed eight-episode result retains the x improvement, restores y to
  the learned candidate, and leaves lifecycle/identity/collision totals exact.
  The fixed 32-episode MPS comparison passes this boundary: x improves at all
  horizons, y/z/lifecycle/identity match learned, and event false positives
  improve slightly. Set the evaluator pool's independently selected axes to
  `[0]`. The pool is not yet the normal runtime default, so this acceptance
  neither changes `WorldBelief` nor silently alters predict–correct behavior;
  runtime integration has a separate explicit opt-in and full-guardrail gate.

## ADR-134 — Bind heterogeneous model choice to local causal evidence

- **Date:** 2026-08-15
- **Status:** accepted architecture contract; implementation repair active and
  not yet qualified
- **Context:** A reread of the AAAI ORPHEUS paper and its Theory-of-Mind
  framework makes their core mechanism precise: perception constructs an
  entity-centred imaginary world; domain-expert and learned models coexist in
  a replaceable pool; model assignment is revised from simulation-versus-
  reality error; physical and behavioral effects are interleaved in stable
  small steps; and several nearby future simulations expose uncertainty and
  brittle outcomes. The first normal-runtime integration violated the local
  scope of that evidence. It learned candidate preference only from a
  0.05-second delayed RGB target, then used that choice for x forecasts through
  1.00 second. A matched four-episode MPS comparison against the learned-only
  runtime shows why this is invalid: 1.00-second x RMSE regresses
  `0.771005 -> 0.895082` (`+16.1%`), aggregate position RMSE regresses
  `0.618738 -> 0.672120` (`+8.63%`), NLL worsens `1.985%`, and global/fast
  update latency rises `2.216x`/`2.392x`.
- **Decision:** `WorldBelief` remains the sole persistent physical truth and
  the papers' perceptual mental image. Candidate applicability/evidence lives
  beside it and is keyed by persistent entity, state component/axis,
  interaction/event regime, supported horizon or short-step interval, and the
  exact source-belief/dynamics revision. A choice is never transferred beyond
  those keys. Missing support uses an explicit accepted-learned fallback and is
  reported separately from positive learned-model selection. Continuous
  evidence uses the innovation likelihood under predictive plus measurement
  variance, robust masks/influence, and explicit support/freshness/
  observability. Any external belief replacement, reset, lifecycle slot reuse,
  incompatible correction, runtime-mode change, or dynamics mutation
  invalidates pending evidence. Selected effects compose through bounded
  state-transition steps; position, velocity, and variance remain coherent,
  and joint event/cross-axis state cannot be replaced by an unrelated
  coordinate splice. Candidate branches remain transient futures and never
  become alternate world truth.
- **Alternatives considered:** retain the current 0.05-second scene/axis choice
  at every horizon; tune evidence decay until the four examples improve;
  select with simulator truth; splice position while retaining learned
  velocity and uncertainty; discard analytic candidates; or increase learned
  model capacity before repairing runtime semantics.
- **Consequences:** The current runtime pool remains opt-in and is rejected for
  promotion. The immediate repair is horizon-bound fallback, followed by
  per-entity/regime applicability, combined-uncertainty scoring, causal
  invalidation, coherent short-step composition, and removal of redundant
  propagation. Every change requires focused tests and an exact paired MPS
  comparison including current and every x/y/z position/velocity horizon,
  lifecycle, identity, events, calibration, nonfinite integrity, and global/
  fast/forecast latency. The four-episode result is diagnostic because it
  covers only four of eight scenarios and overlaps validation; it cannot
  establish convergence even though it is sufficient to reject the policy.
- **Primary sources:** Polceanu, Parenthoën, and Buche, “ORPHEUS: Mental
  Simulation as Support for Decision-Making in a Virtual Agent”
  (https://cdn.aaai.org/ocs/10371/10371-46146-1-PB.pdf); Polceanu and Buche,
  “Towards A Theory-Of-Mind-Inspired Generic Decision-Making Framework”
  (https://arxiv.org/abs/1405.5048).

## ADR-135 — Monitor durable artifacts without touching model execution

- **Date:** 2026-08-15
- **Status:** accepted
- **Context:** Long RGB/MPS training and evaluation phases can spend minutes in
  one real causal batch or forecast anchor. Frequent ad hoc shell/process
  checks create noise, while raw `metrics.jsonl` rows are too wide to make
  convergence, guardrail rejection, or a dead process obvious. Optional
  evaluator progress also meant a normal invocation could not be monitored
  after detaching.
- **Decision:** Provide one root `monitor.py` that reads only durable atomic
  run artifacts, a bounded recent JSONL tail, checkpoint file metadata, and
  advisory process state. It recursively discovers timestamp-first nested
  runs, prefers verified-active work, uses a 60-second default interval,
  suppresses unchanged snapshots, and emits a ten-poll heartbeat. Raw training
  loss is summarized by rolling medians and previous-window delta; immutable
  validation decisions and per-horizon RMSE remain the accuracy signal. The
  monitor never imports/executes the model, deserializes checkpoints, consumes
  accelerator memory, mutates the run, or contacts a service. Standard
  evaluation progress is durable by default; `--progress` adds stdout only.
- **Alternatives considered:** high-frequency external polling; loading the
  latest checkpoint to probe it; requiring TensorBoard or an experiment-
  tracking service; treating a stale PID file or old summary as proof of live
  or completed work; and diagnosing collapse from one mixed-scenario loss.
- **Consequences:** A local terminal can follow training/evaluation cheaply and
  truthfully, including resumed older run directories and nested evaluations.
  Hard failure/nonfinite/staleness signals are visible without changing model
  timing. Evaluation output is planned and `initializing` is persisted before
  model/checkpoint setup; caught exceptions and keyboard interruption record
  terminal state with their last progress. An uncatchable process kill still
  leaves no terminal event, so the monitor reports the dead process as stale
  rather than fabricating completion.

## ADR-136 — Isolate familiar pair impacts in the complete simulator scene

- **Date:** 2026-08-15
- **Status:** accepted; simulator protocol `sphere_world_v6`
- **Context:** Several supposedly simple pair-collision episodes put the same
  spheres into floor or third-body contact in the same observation interval.
  The resulting friction impulse could cancel x velocity, making both ground
  truth and learned prediction look physically unfamiliar. Free-flight data
  also used a timestep-dependent integrator different from the analytic prior.
- **Decision:** Preflight the complete deterministic scene with the production
  solver. Keep both ensured-pair objects free from every other contact for two
  observation frames after impact, reserve their corridor, delay births and
  impulses, and resample extra objects when needed. Use the same closed-form
  gravity/linear-drag free-flight transition as analytic dynamics. Label
  genuinely compound interactions separately.
- **Consequences:** The familiar benchmark now tests the interaction it names;
  v4/v5 reports remain historical diagnostics and cannot serve as matched v6
  promotion controls. Contact substep-rate changes still need separate parity
  qualification.

## ADR-137 — Make independent raw RGB temporal history an explicit protocol

- **Date:** 2026-08-15
- **Status:** accepted; fast-depth clause superseded by ADR-142
- **Context:** Corrected posterior positions and prior-conditioned ROI copies
  are correlated with the dynamics being estimated. Treating them as fresh
  temporal observations reinforces model bias and understates uncertainty.
  Replacing them unconditionally, however, silently changed historical
  checkpoints that encoded a nonzero posterior/measurement blend.
- **Decision:** Add an exact legacy-false semantic switch. Historical configs
  keep their configured blend; new grounded configs opt into independent raw
  RGB history. Persist per-sample, per-axis provenance; source-bound ROI values
  count only where structured centre/scale directly observes them. The 1.46
  candidate enabled structured fast depth; ADR-142 subsequently disables it
  because component completeness is not observable. Keep no age cutoff and
  optionally compensate known `WorldBelief.gravity` to estimate current-time
  velocity.
- **Consequences:** Old checkpoint behavior is preserved while new training can
  remove observer self-reinforcement. A real combined-camera online test must
  show nonzero fast-path support before launch. The strict mode uses no
  simulator state, and current fast-path depth support remains intentionally
  absent until scale completeness is independently qualified.

## ADR-138 — Separate clean accuracy from recovery and bind immutable evidence

- **Date:** 2026-08-15
- **Status:** accepted
- **Context:** Public evaluation formerly injected a belief perturbation into
  the same runtime used for ordinary accuracy, unlike clean trainer
  validation. A mutable `last.pt` could also be replaced between primary load,
  recovery reload, and final hashing.
- **Decision:** Keep the primary pass intervention-free at the evaluator level.
  Run recovery in an independent replay from the same immutable checkpoint
  payload, and publish its metrics in a disjoint namespace. Bind exact bytes,
  checkpoint/evaluation versions, source provenance, resolved protocol, seed
  manifest, scenarios, horizons, batching, and metric schema. Reject any
  nonfinite state/event/metric before completed JSON and exclude lifecycle slot
  reuse from recovery matches.
- **Consequences:** Primary metrics are comparable with fixed validation and
  bitwise invariant to recovery settings. Simulator scenarios may still
  contain declared physical impulses; reports distinguish those from evaluator
  perturbations. Per-scenario slices are additive diagnostic views unless they
  contain every promotion guardrail.

## ADR-139 — Version relation binding and keep learned effects local

- **Date:** 2026-08-15
- **Status:** accepted for opt-in qualification
- **Context:** Symmetric endpoint binding makes a relation token structurally
  meaningful, but enabling it implicitly would change historical attention
  checkpoints. Learned pair residuals also acted far outside contact, while
  repeated attention evaluation dominated rollout cost.
- **Decision:** Gate endpoint binding behind a legacy-false checkpoint semantic
  and enable it only in the new campaign. Provide an opt-in, smooth gap/
  closing/uncertainty pair-applicability multiplier recomputed every physical
  microstep. Provide an opt-in within-call multi-rate learned proposal cache
  invalidated by topology changes or collisions. Keep analytic contact and all
  physical/uncertainty microsteps exact and keep both optimizations off by
  default.
- **Consequences:** Historical inference stays stable. New relation capacity is
  entity-incidence aware and local to plausible interactions. The measured
  MPS multi-rate gain is useful but modest, so cadence remains disabled until
  matched accuracy and latency pass.

## ADR-140 — Stage perceptual anchoring before relation learning

- **Date:** 2026-08-15
- **Status:** accepted for the next long campaign
- **Context:** The protected checkpoint's typed attention decoders are zero;
  generic attention stages produced aggregate long-horizon gains but damaged
  current coverage and familiar scenarios. Current RGB tracking/velocity
  support is the larger end-to-end limitation.
- **Decision:** Train `state_roi` first: filter/updater, identifier, fast ROI,
  projection, and early visual features only. At a declared causal boundary,
  transition to `state_relation_roi`, adding graph-edge and relation/shared
  attention parameters while preserving zero/untrained node acceleration,
  analytic kinematics, global discovery, and unrelated heads. Use balanced
  batches, a complete warmup/useful-rate/decay schedule, repeated immutable
  validation, and a multi-thousand-update budget.
- **Consequences:** Capacity is added where interaction evidence exists without
  sacrificing the constant/damped free-flight prior. No stage is promoted from
  a local win; all scenario/axis/horizon/lifecycle/event/calibration guardrails
  remain binding.

## ADR-141 — Slow parameters require independent, uncertainty-aware evidence

- **Date:** 2026-08-15
- **Status:** accepted
- **Context:** The analytic identifier used corrected-position residuals and
  could interpret ambiguous or copied ROI coordinates as strong physical
  evidence. Position displacement was also being treated as if it directly
  measured post-impact restitution.
- **Decision:** Use causal measured-minus-prior error, actual elapsed time,
  combined predictive and measurement variance, and explicit confidence.
  Respect per-axis independence provenance and fail closed when source-bound
  ROI provenance is absent. Position supports drag only; restitution requires
  supported direct velocity or the labelled oracle debug path.
- **Consequences:** Asynchronous RGB updates remain cheap, but slow parameter
  gates now reflect observability rather than residual magnitude alone. Cold
  temporal support deliberately delays the analytic update instead of
  inventing evidence.

## ADR-142 — Retain fast centres but reject unqualified fast component depth

- **Date:** 2026-08-16
- **Status:** accepted
- **Context:** A prior-conditioned ROI can localize a foreground component, but
  its measured radius does not reveal whether overlap, truncation, or merging
  hid or added part of the physical disc. On seed 100000 the 28 accepted
  radius ratios averaged `1.1587`; over eight seeds, fast depth on/off pooled
  current position RMSE was `0.27719/0.13479 m` and precision/recall was
  `0.70265/0.68704` versus `0.96628/0.92870`.
- **Decision:** Keep structured fast centres as independent lateral evidence
  and disable structured fast depth in the grounded profile. Require an
  explicit completeness/visibility model and matched broad evidence before
  re-enabling component radius as independent depth.
- **Consequences:** The cheap residual path still corrects image-plane motion,
  while ambiguous scale cannot drag posterior z or break association. This
  rejects one evidence claim, not the point-and-trajectory abstraction.

## ADR-143 — Calibrate temporal velocity variance to empirical residuals

- **Date:** 2026-08-16
- **Status:** accepted
- **Context:** Eight-seed gravity-aware direct velocity evidence had y MSE
  `3.82435` against variance `0.23707`, about `16.13x` overconfidence. The
  `0.25` ceiling prevented the filter from representing the observed error.
- **Decision:** Raise only the grounded temporal velocity variance ceiling to
  `4.0`; retain independent raw history and continuous gravity fitting. Reject
  the tested contact-free change-point reset because its early noisy resets
  regressed the calibrated observer.
- **Consequences:** With identical weights and fast depth off, position all-axis
  RMSE improves `0.134785 -> 0.130092 m`, distance-gated velocity improves
  `0.774363 -> 0.759842 m/s`, and F1 improves `0.947120 -> 0.953730` over
  seeds `100000--100007`. This is observer calibration evidence, not trained
  model convergence.

## ADR-144 — Add finite smooth event hazards behind a legacy-false semantic

- **Date:** 2026-08-16
- **Status:** accepted for training qualification
- **Context:** Hard analytic event logits correctly protected physical jumps
  but gave relation parameters little ownership of event timing. Dense
  self-pair projected variance also reached `sqrt(0)` before masking, producing
  an infinite derivative and recursive `0 * inf` NaNs.
- **Decision:** Add opt-in smooth pair/boundary contact and collision hazards
  from gap, incoming normal motion, uncertainty, and learned relation residuals.
  Keep the hard resolver as the physical fail-safe and use a straight-through
  positive floor for resolved events. Clamp projected variance to a
  dtype-aware positive floor before square root. Supervise unique matched
  pairs directly in addition to node events.
- **Consequences:** CPU and active-Aqua MPS recursive gradients are finite and
  event decoders have a causal owner. Historical checkpoints remain exact with
  the flag false. A same-weight eight-seed preflight changed horizon position
  RMSE by at most `1.63e-7 m` and left collision F1/coverage exact, clearing the
  inherited physical baseline without claiming event learning. The feature
  remains unpromoted until trained fixed RGB validation demonstrates accuracy
  and calibration.

## ADR-145 — Weight each event horizon against the configured schedule

- **Date:** 2026-08-16
- **Status:** accepted
- **Context:** Re-normalizing over only currently eligible event anchors let a
  late `0.1`-weight horizon inherit full unit weight when earlier anchors lacked
  causal support. That silently changed the objective across windows.
- **Decision:** Emit explicit per-horizon event terms and apply the fixed
  configured horizon-weight denominator, just as for the physical rollout
  terms. Missing early support contributes no numerator and does not reshape
  the declared schedule.
- **Consequences:** Event pressure is comparable across batches, causal support
  patterns, and exact resume. Direct pair and node ownership share the existing
  event scale rather than doubling it.

## ADR-146 — Require causal support and a trainable stage owner

- **Date:** 2026-08-16
- **Status:** accepted
- **Context:** Discovery births receive hard-zero runtime velocity and have no
  trainable incoming prior. Scoring their velocity/correction in the objective
  diluted denominators, while event loss in `state_roi` could pressure shared
  perception even though its event/relation heads were frozen.
- **Decision:** Support velocity only for matched active slots with
  `age_steps > 0`; support correction only when both prior and posterior are
  active under the same causal age contract. Structurally omit unsupported
  loss terms. Set grounded event weight to `0.0` in `state_roi` and `0.05` in
  `state_relation_roi`, structurally omitting zero-weight terms. Preserve
  historical event weights when no scope override exists.
- **Consequences:** Every optimized term has a causal trainable path. Explicit
  counters expose excluded/supported coordinates and objects. Public physical
  metrics remain unfiltered and continue to reveal poor newborn estimates.

## ADR-147 — Reject pair applicability and increase formal validation cadence

- **Date:** 2026-08-16
- **Status:** accepted for the next campaign
- **Context:** The matched seed-100000 pair gate regressed current position/
  velocity `0.56%/0.38%` and 0.25/0.50/0.75/1.00-second position
  `0.75%/0.88%/1.02%/0.77%` without collision-F1 gain. The prior 1,024-update
  evaluation cadence also provided too few formal observations for a stable
  plateau decision over 9,216 updates.
- **Decision:** Keep pair applicability disabled. Evaluate the fixed manifest
  every 512 updates, yielding 18 post-update validations through step 9,216,
  plus an immutable step-zero baseline. Preserve the 3,072 scope transition
  and declared warmup/cosine/tail schedule.
- **Consequences:** A plausible local gate cannot enter the campaign after a
  matched regression. The longer experiment now has enough predeclared
  observations to distinguish convergence from noisy batch wobble, but no
  convergence is claimed before those evaluations exist.

## ADR-148 — Use an algebraically stable smooth conjunction on production MPS

- **Date:** 2026-08-20
- **Status:** accepted numerical repair; full campaign qualification pending
- **Context:** The first active-Aqua specification-1.47 campaign at
  `runs/20260820-213418-grounded-convergence-spec147-mps` failed before any
  optimizer update in its first incumbent validation (`0/32`) because
  trajectory `pair_event_logits` contained NaN or Inf. Exact seed-100000 replay
  showed that all model inputs and learned outputs were finite until the
  custom PyTorch `2.9.0a0+gitcbe1a35` MPS `torch.logaddexp` primitive
  overflowed for finite inputs with magnitude around `90`. The smooth
  collision conjunction `-logaddexp(-a, -b)` consequently became `-Inf` for a
  distant valid pair. This was a backend numerical defect, not training
  collapse, corrupt weights, or an invalid uncertainty state.
- **Decision:** Evaluate the same soft minimum as
  `minimum(a, b) - softplus(-abs(a - b))`. The form is algebraically identical
  over real inputs and retains the existing hazard, gradients,
  straight-through resolved-event floor, hard analytic jump fail-safe,
  devices, tensor shapes, and checkpoint contents. Treat forward and backward
  finiteness at extreme logits as a production device contract. Retain the
  failed run as terminal evidence and start the repaired 9,216-update campaign
  in a fresh timestamped directory rather than resume or reuse it.
- **Alternatives considered:** clamp the hazard logits; disable the new smooth
  event semantic; move event evaluation to CPU; ignore or sanitize the
  nonfinite auxiliary after rollout; or attribute the failure to model
  instability and change weights or capacity.
- **Consequences:** CPU hybrid tests pass `33` cases with `3` Aqua-MPS skips,
  and the two focused active-Aqua MPS regressions pass. The exact first
  production episode (seed `100000`, 40 frames, 8 rollout anchors) completes
  on MPS with finite loss `2.279386520385742` and 307 finite metrics in about
  137.4 seconds. The frozen repository suite passes `960` tests with `14`
  expected non-Aqua MPS-context skips; lint, format, compile, and diff checks
  pass. This qualifies only the localized repair and repository integrity.
  Complete `32/32` initialization validation, a campaign relaunch, selector
  behavior, prediction accuracy, and convergence remain unproved.

## ADR-149 — Batch validation anchors only after exact MPS parity

- **Date:** 2026-08-21
- **Status:** accepted and qualified for the grounded profile
- **Context:** Specification 1.48 repaired the MPS event hazard, but serial
  initialization validation remained too expensive: a manually interrupted
  attempt at `runs/20260820-221902-grounded-convergence-spec148-mps` reached
  only `2/32` episodes in `1559.189234` seconds. Repeated child-level elapsed-
  time guards and eight independent posterior rollouts per episode caused
  avoidable accelerator synchronization and repeated learned-dynamics work.
- **Decision:** Let `DynamicsModel` validate elapsed time once per segment and
  retain one complete finite-output boundary; private analytic/modal/
  uncertainty paths may consume that normalized tensor. Use an all-positive
  path only when every row advances, retaining exact masks for mixed or zero
  rows. Add validation-only anchor-major batching behind typed config. Keep
  episode ingestion and persistent runtime state batch one, pad only repeated
  terminal query time, slice exact prefixes before scoring, and subdivide or
  fall back serially when modality or metadata differs, including lifecycle
  flags carried in metadata.
  Preserve serial `1` as the generic and legacy default and make the field plus
  batching semantic exact-resume and validation-protocol state.
- **Alternatives considered:** reduce the fixed manifest or anchor count;
  weaken finite checks; batch complete episodes and lose exact attribution;
  assume metadata is static; enable batching from a CPU smoke; resume an old
  campaign after changing execution protocol.
- **Consequences:** On the frozen 32-seed active-Aqua MPS protocol, serial and
  batched execution took `3760.393956` and `2012.605486` seconds
  (`1.8684208x`). All 3141 comparable values passed tolerance with no missing
  or nonnumeric differences; maximum finite absolute/relative deltas were
  `7.62939453125e-06` and `6.334555944e-07`, and final runtime SHA-256 matched.
  All 256 anchors batched in 32 calls with zero fallback. The grounded profile
  may therefore use size `8`; other profiles remain serial unless separately
  qualified. This is execution qualification only. No optimizer update,
  prediction-accuracy improvement, checkpoint promotion, or convergence has
  been demonstrated.

## ADR-154 — Close an identifiable differentiable toy before scaling

- **Date:** 2026-08-26
- **Status:** accepted; minimal ladder qualified
- **Context:** Repeated heterogeneous campaigns remained difficult to
  interpret because perception, association, lifecycle, filtering, contact,
  learned residuals, and long recursive graphs changed together. A direct
  audit also found hard RGB component values carrying fabricated
  straight-through derivatives. Simple dynamics should not require a large
  model or multi-day optimization.
- **Decision:** Stop broad scaling and require a deterministic three-rung unit:
  exact oracle-state analytic motion, RGB-only metric state, then a short
  RGB-derived analytic rollout. Fix one visible sphere and every identifiable
  physical parameter; exclude contact, lifecycle ambiguity, camera motion,
  and noise. Use disjoint train/selector/confirmation/final manifests and
  materialize the final set once. A failed family stops without gate changes.
- **Consequences:** Frozen source `f8d66da` passed with final state and rollout
  RMSE `0.007644 m` and `0.007991 m`, versus `0.05 m` limits. This is a
  convergence and gradient proof for the toy only. It does not promote the
  accumulated research branch or establish heterogeneous-scene accuracy.

## ADR-155 — Derive RGB geometry through a differentiable physical surrogate

- **Date:** 2026-08-26
- **Status:** accepted for the minimal path
- **Context:** The rejected v1 toy used a learned colour-conditioned radius
  calibrator. It overfit its eight training seeds and amplified a roughly
  `0.1 px` soft-centre bias into large monocular depth error. Supplying the
  exact centre to the same inverse renderer reduced radius error to `0.25%`,
  localizing the issue to sequential centre/radius inference rather than the
  dynamics equations.
- **Decision:** Jointly infer centre and log-radius with four ordinary-autograd
  finite-difference Gauss--Newton stages. Use a smooth form of the public
  renderer, candidate-independent full-frame RGB residuals, analytic nuisance
  albedo, a dtype-scaled positive damping term, and bounded trust updates.
  Keep hard connected-component values detached and expose raw learned
  geometry separately. Prevent source-bound copied world axes from entering
  analytic position or position-derived-velocity fusion.
- **Consequences:** A seed-free 100-profile renderer grid passes centre/radius
  limits with finite-difference gradient agreement. The deployed toy uses 29
  profile renders per frame, has no arbitrary Gibbs temperature or simulator
  truth in its forward path, and preserves finite nonzero mask-head gradients.
  More complex learned residuals remain gated on an independently measurable
  failure of this equation-led baseline.
