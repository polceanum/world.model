# Changelog

## Unreleased — 2026-07-28

### Added

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
- The authoritative specification is now version 1.3 and requires identical
  explicit episode manifests for checkpoint comparisons and per-scenario gates
  for one shared model.
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
