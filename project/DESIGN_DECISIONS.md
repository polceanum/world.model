# Design decisions

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
- **Evidence:** On seeds `100016–100017`, cadence three improved current
  RMSE/MAE by `5.9%/21.3%`, velocity slightly, 0.10–0.75-second RMSE by
  `7.6–12.8%`, collision F1 from `0.138` to `0.357`, and detection
  recall/precision from `0.500/0.377` to `0.694/0.568`. On disjoint seeds
  `100018–100019`, it improved current RMSE/MAE by `6.3%/12.5%`, velocity by
  `10.7%`, 0.10–0.75-second RMSE by `3.4–9.0%`, collision F1, and target
  coverage. One switch occurred on the first pair and nominal 90% coverage
  worsened on the second. The corrected sampler produced nonzero position and
  velocity rollout losses at step 6.
- **Consequences:** Cadence three is the scaled default but still requires a
  wider validation/test manifest. The one-second horizon remains essentially
  unchanged. Longer causal training is allowed only from forecast-supervised
  windows and remains subject to paired promotion gates. A step-16
  sampler-corrected continuation was subsequently rejected: current state
  improved marginally, but 0.25–1.00-second forecasts regressed.

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
