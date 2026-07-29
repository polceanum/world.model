# Tasks

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

- [ ] Multi-frame tentative birth confirmation (configuration currently
  supports the Milestone 1 value of one confirmation).
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
- [ ] Raise shared-model collision F1 and reduce identity switches on a larger
  balanced test manifest; the current 16-episode result is only
  `0.320388` F1 with three switches.
- [x] Audit structured RGB point/scale accuracy and separate centre error from
  monocular depth error. Centre localization is subpixel; heavy-tailed
  radius-derived depth under overlap is dominant.
- [x] Add disabled-by-default, tested gates for associated depth-disagreement
  covariance inflation and combined temporal/position-innovation velocity
  evidence. Reject both policies after final-test multistep regressions.
- [ ] Replace single-frame radius depth in ordinary correction with a learned
  persistent-ID multi-frame point/scale trajectory measurement. Produce
  per-axis estimates and uncertainty from axis-local history, then use joint
  interaction/event context to gate departures and cross-axis coupling.
- [ ] Supervise scale quality with visible fraction, boundary truncation,
  component overlap, temporal scale consistency, and prediction disagreement;
  validate calibration by quality bucket before enabling correction.
- [ ] Train event-conditioned outgoing velocity on balanced pre-contact and
  post-contact windows, with constant/damped motion represented as a learned
  low-complexity prior rather than a hardcoded runtime rule.
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
- [ ] Profile and reduce closed-loop validation/rollout cost before another
  large run; the 24-frame eight-episode validator exceeded 84 minutes and
  evaluator rollout calls averaged about 9.8 seconds.
- [ ] Add gradient checkpointing or optimizer accumulation if eight-step,
  batch-one causal updates remain the throughput bottleneck.
