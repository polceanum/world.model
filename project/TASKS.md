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

## Milestone 1 research acceptance — not yet achieved

- [x] Reach nonzero distance-gated RGB detection recall/precision on held-out
  episodes (59.375% on the two-episode protocol; 75% over eight episodes).
- [ ] Sustain the recommended 20% injected-perturbation recovery improvement
  on the wider test protocol (27.71% on two episodes; 19.59% on eight).
- [ ] Learn useful collision prediction (current collision F1: 0).
- [ ] Demonstrate calibrated uncertainty expansion/recovery through held-out
  rendered occlusions.
- [ ] Demonstrate distance-gated drag and restitution convergence from RGB,
  beyond merely executing the observability/update gates.
- [ ] Make the ROI path measurably cheaper than the global path at the target
  scale or explain the intended compute tradeoff with profiling.
- [ ] Train fast inverse-depth residuals on belief-slot-aligned cached
  sequences and enable them only after per-mode current/future improvement.
- [ ] Add temporal RGB velocity/event evidence so collision impulses are
  assimilated instead of only propagated by oracle-correct dynamics.
- [ ] Run the full 3,000-step `configs/toy_mps.yaml` schedule and a materially
  larger held-out test split.
- [ ] Export physics-violation diagnostics, collision-conditioned metrics,
  occlusion survival metrics, and representative failure plots.

## Deferred architecture

- [ ] Multi-frame tentative birth confirmation (configuration currently
  supports the Milestone 1 value of one confirmation).
- [ ] Multiple-hypothesis branch/prune/merge.
- [ ] Estimated camera pose and fixed-lag smoothing.
- [ ] Continuous collision timing and richer geometry.
- [ ] Spectral fixed-window ablation.
- [ ] A second modality and real calibrated video adapter.
