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
- [ ] Beat the promoted step-648 checkpoint on paired 0.50/0.75/1.00-second
  RGB-only validation. Three horizon-weight/window-only continuations were
  neutral or negative and remain rejected.
- [x] Add temporal RGB velocity evidence so post-association motion can be
  assimilated without re-encoding history. Validation has not yet justified
  enabling it in public profiles.
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
