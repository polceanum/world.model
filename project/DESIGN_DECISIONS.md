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
