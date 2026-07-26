# Changelog

## Unreleased — 2026-07-26

### Added

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

### Changed after integration audit

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
  `demo_outputs/convergence-tiny-cpu-v1`.
- Frozen-continuation demo:
  `demo_outputs/accuracy-closed-frozen-94`.
- Reduced real MPS training:
  `runs/milestone1-mps-smoke-final`.
- Reduced two-step run of the full-size `toy_mps` architecture:
  `runs/milestone1-toy-mps-scaled-smoke`.
- Final 12-step public CPU smoke:
  `runs/accuracy-final-smoke`.

The frozen continuation reaches 75.39% distance-gated recall/precision over
eight held-out episodes, 0.161387 m 0.5-second forecast RMSE versus 0.490275 m
for constant velocity, zero gated ID switches, and positive perturbation
recovery. Milestone 1 acceptance is still not claimed: best exact-window
collision F1 is only 0.0556, wider perturbation recovery is about 19.5% versus
the recommended 20%, parameter convergence and held-out occlusion recovery
remain incomplete, and the full MPS schedule has not run.

The continuation and event-loss comparisons repeatedly inspect the same eight
fixed test episodes and are therefore exploratory. Step 72 remains selected by
validation loss; any preference for step-94 `last.pt` requires confirmation on
a fresh, larger held-out seed set. Run and demo artifact directories are local
and gitignored.
