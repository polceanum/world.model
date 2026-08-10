# Project Orpheus agent guide

Before changing code, read these files in order:

1. `PROJECT_SPEC.md` in full;
2. `project/STATUS.md`;
3. `project/TASKS.md`;
4. `project/DESIGN_DECISIONS.md`;
5. `project/CHANGELOG.md`.

`PROJECT_SPEC.md` is the authoritative architectural contract. Preserve the
public tensor contracts, the persistent `WorldBelief`, timestamped asynchronous
observations, explicit uncertainty, persistent identity/lifecycle, hybrid
dynamics, and the separation between fast state correction and slow parameter
identification.

Working rules:

- Use the existing `orpheus` conda environment for every Python command.
- Do not reinstall or replace PyTorch. Model execution remains local.
- Keep `train.py`, `evaluate.py`, and `demo.py` as the simple public workflow.
- Add or update focused tests with every behavioural change.
- Keep project memory and relevant architecture documents synchronized.
- Record exact commands, observed results, limitations, and unfinished work.
- Do not replace the architecture with a clip predictor or opaque dynamics MLP.
- The oracle/state observation module is debug-only. Never use simulator state
  as an input to a claimed RGB result.
- Do not add hosted services, API keys, databases, Hydra, Lightning, external
  tracking, or heavy infrastructure without a documented architectural need.
- Preserve user work. If it conflicts with the specification, record the
  conflict in `project/DESIGN_DECISIONS.md` and make the smallest correction.
- Preserve predictive abstractions as first-class executable state. Prefer the
  simplest representation that predicts within calibrated uncertainty (for
  example, a point trajectory for a freely moving ball) and refine it only
  when interactions require richer geometry or dynamics.
- Foundation models, transformers, and generative decoders may extract,
  complete, or propose abstractions. They must not replace `WorldBelief` with
  an opaque sensor latent or make generated pixels the primary evidence of
  physical correctness.
- Treat entity/relation tokens as unordered typed sets. Do not add slot-index
  positional embeddings or RoPE to arbitrary padded object order; temporal
  encodings belong only to explicitly timestamped bounded-history tokens.
- New attention capacity must begin as a zero-output residual around the
  qualified structured model, load every inherited weight strictly, and train
  in an isolated scope before any shared module is unfrozen.
- Preserve at least one familiar, invariant-tested reference-physics regime.
  Unusual and compound dynamics are valid learnable scenarios, but label and
  evaluate them separately so simulator quirks cannot masquerade as model
  accuracy or failure.
- Treat a mature physics engine as an optional independent RGB dataset backend,
  never as runtime privileged input or a replacement for learned dynamics.
  It must emit the canonical episode/observation contracts and record its
  engine version, solver settings, units, timestep, scenario, and seed.
- Prefer one shared checkpoint across the declared scenario mixture. Treat
  scenario-specific checkpoints as diagnostic ablations unless an explicit
  observation-derived regime router is part of the runtime.
- Compare checkpoints on the same explicit seed manifest, ordered scenario
  mixture, object counts, sequence length, horizons, and metric semantics.
  Never infer an accuracy gain from evaluator defaults that select different
  episodes.
- Treat short runs as wiring or throughput checks. Do not discard a shared
  model or claim convergence until its declared balanced minimum training
  coverage has completed and broad validation has reached a predeclared
  plateau.
- When heterogeneous scenario updates demonstrably trade accuracy between
  regimes, use deterministic manifest-bound scenario-balanced optimizer
  batches; never describe a randomly shuffled dataset as per-update balance.
- For the sustained campaign, plateau requires four exact consecutive
  512-step validation candidates with no acceptance and less than 1% raw
  primary-score improvement over the safe pre-window incumbent. Missing or
  contradictory evidence means continue a complete block; reaching a hard
  budget without this evidence is `limit_hit`, not convergence.
- Select long-run checkpoints by pooled physical metrics with distance-gated
  detection/identity, lifecycle, event, velocity, every-horizon, and
  calibration guardrails. Apply non-regression checks against both the moving
  incumbent and a fixed pre-campaign reference, and retain numbered validation
  snapshots with verifiable weight and protocol provenance.
- Require every declared scenario to have explicit current/horizon validation
  support and its own persisted broad guardrails. A better pooled score cannot
  promote a candidate that loses support or materially regresses one scenario.
  Keep balanced scenario lists unique so their fixed seed residues really
  visit every declared family.
- Keep the guardrail-safe deployment incumbent separate from the mutable
  optimisation trajectory. A candidate rejected for deployment may still be
  the correct state from which causal training repairs the failed guardrail.
- Apply deterministic point/event losses only to futures identifiable from the
  causal anchor. After an unseen external actuation, train calibrated forecast
  likelihood but censor deterministic targets for the coupled scene.
- Isolate renderer RNG from physics/lifecycle/actuation RNG. A render-only
  configuration change must leave the physical trajectory and event labels
  exactly unchanged for the same seed.
- Treat `--resume` as exact continuation: preserve the absolute next sample,
  optimiser and CPU/MPS RNG, objective/data protocol, and launch-time source
  provenance. Use `--initialize-from` for a changed curriculum.
- Preserve collision evidence across every dynamics substep in an observation
  interval. Keep endpoint contact separate from interval collision, and never
  classify side-wall or ceiling support as ground/sleep.
- Select perception checkpoints with the same confidence/lifecycle semantics
  used by runtime births. Pool additive measurement counts/errors before
  deriving MAE, recall, precision, or F1.
- Omit unsupported objectives rather than averaging fabricated zero examples.
  Use uncertainty calibration to train variance without duplicating an
  explicitly supervised state-mean gradient.
- Apply the same isolation to rollout likelihood: detach forecast-mean error,
  train variance from realised outcomes, and never learn a deterministic mean
  across an unseen external actuation.
- Keep RGB measurement auxiliaries perception-local. A frozen fast ROI can
  remain differentiable through its prior input, but that must not let its
  measurement loss train dynamics or the belief updater.
- Do not consume a causal optimizer step from global auxiliary perception
  alone. Require explicit differentiable trajectory/state/parameter support or
  a valid persistent fast-ROI slot; count deterministic skipped draws
  separately from completed updates and bound retries.
- Keep fast-ROI identity, crop, exact-geometry, existence, and visibility
  support distinct. Valid empty crops train negative existence/visibility,
  unsupported attributes are omitted, and every eligible confident false
  positive remains in selector precision.
- Treat tentative births as detached `(modality, sensor)`-local observation
  history, never as physical state. Require configured consecutive,
  strictly-later, distance-gated detections before allocating a permanent ID.
- Gate inadmissible association, tentative-confirmation, and new privileged
  target-mapping edges before Hungarian assignment so invalid low-cost
  combinations cannot reduce the number of valid matches.
- Bind prior-conditioned fast ROI rows to their explicit source belief slot and
  object ID. They may be rejected by normal gates but may never cross-update a
  different persistent identity; global discovery remains freely associated.
- Reject numerically tied ownership between disconnected foreground components
  on the source-conditioned fast ROI path. Preserve the predicted centre and
  leave ambiguous or large recovery to global discovery rather than allowing a
  subpixel crop change to create a discontinuous identity-bearing correction.
- Derive slow drag/restitution supervision only from accepted runtime
  observations across clean causally observable intervals. Simulator track
  existence or a newborn may map supervision/evaluation identities, but it
  must not open a runtime parameter gate by itself. Reset the temporal
  parameter baseline whenever the associated runtime ID changes.
- Preserve hierarchical gradient evidence when the recursive interaction
  network is locally clipped before the whole model: log raw subsystem and
  total norms, both coefficients, the pre-global norm, total coefficient, and
  final applied norm. Treat both clip limits as resume/protocol semantics.
- When one typed attention proposal row repeatedly dominates the complete
  interaction gradient, isolate that row before the interaction/global caps
  and retain raw/applied diagnostics at every hierarchy. Do not let rare event
  supervision suppress unrelated force, uncertainty, or token gradients.
- Apply the configured RGB perception-local gradient cap only during causal
  training, before the whole-model cap, and retain the true reconstructed raw
  total. Paired RGB pretraining keeps its original whole-model clipping
  semantics.
- A configured phase-specific device switch is part of the resolved protocol,
  not a resume override. Exact resume must verify and preserve every linked
  selector artefact; a no-op inspection must not rewrite a durable checkpoint.
- Preserve the tested PyTorch 2.10 MPS workaround: backbone and ROI tensors
  stay on MPS, but the small global proposal transformer is pinned to CPU
  through differentiable copies when
  `device.global_detector_cpu_on_mps=true`. Do not describe that phase as
  whole-model MPS execution or remove the fallback without matched finite-
  gradient and accuracy evidence.
- Deserialize training, evaluation, and demo checkpoints on CPU, then let
  state loading place tensors on their owners. This preserves CPU Adam step
  scalars for the hybrid optimizer and avoids copying unused optimizer moments
  to accelerator memory.
- Interpret `model.rgb.global_every_steps` as the complete frame-to-frame
  distance between global observations. Cadence three is exactly
  `GLOBAL, FAST_ROI, FAST_ROI, GLOBAL`; test the sequence, not only a counter
  threshold, and bump the rollout protocol when cadence semantics change.
- Keep selector validation full-manifest and atomic, but emit per-episode
  stdout and durable `training_progress.json` heartbeats. Do not start
  training-loader workers before initialization/handoff validation finishes.
- Reject nonfinite model parameters or optimizer state immediately after an
  optimizer step. Validate model buffers, weights, optimizer/scheduler tensors,
  and nonnegative step counters before checkpoint replacement and before
  mutating a destination during load.
- Keep belief-dynamics substep counts aligned with the simulator's nominal
  physics grid: float timestamp representation noise may snap to an
  indistinguishable integer ratio, but genuinely longer intervals must still
  ceil and preserve interval event accumulation.
- Reuse one typed, validated propagation when training needs the same causal
  prior for supervision and ordinary ingestion. Never emulate reuse by
  ingesting the prior at zero `dt`, and reject stale, reused, wrong-source, or
  wrong-time prepared values. Preparation/consumption is atomic with respect
  to belief and dynamics tensor/mode revisions and uses `torch.no_grad()`, not
  `torch.inference_mode()`, when gradients are disabled.
- Launch explicitly requested sustained training at launchd's Standard/default
  process classification with `KeepAlive=false`; do not mark it as
  `Background` maintenance, and verify matched validation throughput before a
  multi-day campaign.
- Learned fast-state mean corrections must be anchored to explicit supported
  world-state innovation. Zero innovation means zero learned mean change, and
  per-axis confidence/support masks both mean and variance residuals.
- Reset and record any checkpoint head whose mathematical output meaning
  changes across protocols; never reinterpret inherited residual-head numbers
  as gains merely because their tensor shapes still load.
- Recover a neutrally reset correction head in an updater-only trainable scope
  before joint dynamics adaptation, and require broad fixed validation rather
  than treating recovery from the deliberately weaker start as promotion.
- Scale attention over entity, relation, event, scene/camera, and bounded
  history tokens derived from `WorldBelief`. Decode outputs into typed
  proposals; attention never becomes an opaque replacement for persistent
  belief, analytic dynamics, association, filtering, or uncertainty.
- Increase model capacity only after the smaller rung has no known correctness
  regression and has completed a fixed-manifest plateau check. Scale data with
  parameters and require disjoint RGB-only generalization plus broad
  non-regression against the accepted smaller control.
