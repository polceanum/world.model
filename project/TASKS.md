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

## Milestone 1 research acceptance — not yet achieved

- [ ] Reach nonzero distance-gated RGB detection recall/precision on held-out
  episodes.
- [ ] Exceed the recommended 20% injected-perturbation recovery improvement
  (current short run: 6.52%).
- [ ] Learn useful collision prediction (current collision F1: 0).
- [ ] Demonstrate calibrated uncertainty expansion/recovery through held-out
  rendered occlusions.
- [ ] Demonstrate distance-gated drag and restitution convergence from RGB,
  beyond merely executing the observability/update gates.
- [ ] Make the ROI path measurably cheaper than the global path at the target
  scale or explain the intended compute tradeoff with profiling.
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
