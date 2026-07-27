# Changelog

## Unreleased — 2026-07-27

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
  `demo_outputs/convergence-tiny-cpu-v1`.
- Frozen-continuation demo:
  `demo_outputs/accuracy-closed-frozen-94`.
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
  `demo_outputs/accuracy-closed-structured-v4`.
- Fresh 32-frame one-second baseline and rejected continuation reports:
  `runs/accuracy-multistep-v1`,
  `runs/accuracy-multistep-balanced-v4`, and
  `runs/accuracy-multistep-long-v5`.
- Stable full-horizon forecast-history demo:
  `demo_outputs/accuracy-v4-forecast-history`.

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
`demo_outputs/accuracy-v6-blended-velocity/online_correction.gif`.

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
