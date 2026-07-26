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
