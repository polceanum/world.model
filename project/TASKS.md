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

## Milestone 1 research acceptance — not yet achieved

- [x] Reach nonzero distance-gated RGB detection recall/precision on held-out
  episodes (59.375% on the two-episode protocol; 75% over eight episodes).
- [ ] Sustain the recommended 20% injected-perturbation recovery improvement
  on the wider test protocol (27.71% on two episodes; 19.59% on eight).
- [ ] Learn useful collision prediction (the promoted checkpoint reaches
  0.0426 fresh-validation F1 and 0.0556 on the older exploratory test;
  semantics are fixed but skill remains far below the 0.75 target).
- [ ] Demonstrate calibrated uncertainty expansion/recovery through held-out
  rendered occlusions.
- [ ] Demonstrate distance-gated drag and restitution convergence from RGB,
  beyond merely executing the observability/update gates.
- [ ] Make the ROI path measurably cheaper than the global path at the target
  scale or explain the intended compute tradeoff with profiling.
- [ ] Train and validate fast inverse-depth residuals on the now
  belief-slot-aligned cached sequences; enable them only after per-mode
  current/future improvement.
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
