# Training

Training uses deterministic on-the-fly sphere episodes and one architecture
through measurement pretraining, closed-loop RGB correction, perturbation
recovery, horizon-weighted rollout loss, event supervision, and observable
parameter losses. Ground truth is supervision rather than an ordinary runtime
input. Each optimizer batch resets belief and causally advances to one
configured truncated-backpropagation window; it does not re-encode a history
clip. A mid-episode window first runs its complete RGB prefix through the real
online filter under `no_grad`, detaches the resulting persistent state, and
then optimizes the selected window.

Closed-loop windows are sampled across the complete episode. With the default
`collision_window_probability: 0.50`, an eligible window is conditioned to
contain a labelled collision with probability 0.5; the label chooses the loss
window only and is never passed to the RGB runtime. Collision and
maximum-horizon intents are sampled independently by default. A compatible
window satisfies both; if a collision is too late, a sampled long-horizon
window is retained. Position and velocity losses are separate for both current
state and future rollout. Aggregate and x/y/z rollout losses all use the fixed
total configured horizon denominator, so a short-only window cannot be
renormalized to full multistep weight. Current default physical weights are
`state_position: 2.0`, `state_velocity: 0.25`,
`rollout_position: 4.0`, and `rollout_velocity: 0.1`, so velocity units cannot
silently dominate physical-position checkpoint quality.

Checkpoints contain model/optimizer/step, resolved config, RNG state, metrics,
specification version, git metadata, simulator version, device, and precision.
Metrics are written locally as JSONL. Measurement-only and closed-loop
validation select separate `best_measurement.pt` and `best_rollout.pt`
checkpoints, preventing a pretraining loss from being mislabeled as rollout
quality. Fixed-dataset measurement training sweeps all frames independently of
loader-batch position. Validation iterates the complete validation loader;
closed-loop validation causally unrolls every full episode, while measurement
validation uses the configured evenly spaced frames from every episode.

An exact in-place resume also requires the executable-source fingerprint stored
in the checkpoint. Documentation and tests are intentionally excluded from
that numerical fingerprint, but `train.py` and `world_model/*.py` are not.
Do not patch executable source while a sustained supervisor may still extend a
run; finish the campaign under identical source, then start changed semantics
through a new timestamped weights-only initialization.

Measurement checkpoints use the versioned runtime-qualified global-discovery
and fast-ROI selector. It pools world-position error, target/proposal/match
counts, recall, precision, F1, and fast bootstrap coverage using the same birth
confidence and eligibility semantics as the persistent runtime. A confident
false positive remains in the precision denominator.
Rollout checkpoints use pooled physical metrics over the complete validation
manifest. Their primary score is horizon-weighted position RMSE; a candidate
must also remain within declared guardrails for current position and velocity,
every horizon, 0.5 m distance-gated recall and precision, forecast lifecycle
coverage, collision F1, distance-gated identity switches, and nominal-90%
position calibration. Guardrails are checked against both the moving incumbent
and the fixed initialization reference. Every validation candidate is retained
as a numbered checkpoint.

Rollout checkpoint metadata contains a canonical validation-protocol hash, the
explicit validation seed-manifest hash, and tensor hashes linking incumbent and
reference metrics to real checkpoint weights. It also retains exact additive
pooled/scenario evidence and every per-seed support marker. Verification
recomputes all pooled, scenario, horizon, and axis selector fields from those
raw sums; internally consistent derived-score tampering is still invalid. A
resumed run reuses metrics only when the linked files, hashes, raw evidence,
support schema, and fixed-reference state verify. An unsupported diagnostic
state persists an incomplete-reference-comparison marker across in-place and
branched resumes. The first later supported candidate establishes a complete
fixed reference but cannot promote itself. This prevents a rejected `last.pt`
or copied numbered snapshot from carrying better incumbent metrics without the
corresponding model state or from self-comparing into deployment.

RGB supervision includes metric-space position after calibrated
backprojection. Supported targets use a smooth-L1 (Huber) term plus a diagonal
Gaussian calibration NLL, in addition to evidence-backed existence, image
geometry, colour, measurement NLL, visibility, and appearance losses. These
terms are not assumed to share one validity mask: a valid empty fast ROI trains
negative existence and visibility only, while unreliable geometry or absent
crop evidence omits centre, depth, colour, appearance, world-position, and NLL
targets. When structured disc centres replace learned centres in the forward
pass, the supported unrefined learned centre is retained as `raw_centre` and
receives its own smooth-L1 auxiliary loss. Global and fast objectives are
support-normalized independently before their fixed configured combination.
Paired RGB pretraining creates a detached one-sequence provisional belief from
the anchor proposals solely to supply prior-conditioned ROI crops on adjacent
frames. It is not online runtime state and does not claim lifecycle
confirmation; causal training, validation, evaluation, and demos use the real
tentative-birth policy.
Fast ROI measurements carry typed, paired source belief-slot and persistent-ID
fields. Association may reject the evidence through its normal gates but may
not cross-assign it, including after slot reuse under a different ID. Global
discovery omits these fields and retains unrestricted gated Hungarian
association.
The default measurement weights are:

```yaml
rgb_existence: 1.0
rgb_geometry: 1.0
rgb_colour: 0.25
rgb_nll: 0.05
rgb_visibility: 0.25
rgb_appearance: 0.25
rgb_raw_centre: 2.0
rgb_world_position: 8.0
rgb_world_position_nll: 0.05
```

The small NLL weights prevent variance fitting from overwhelming metric
localization while retaining an uncertainty-training signal.

Training logs retain legacy `gradient_norm` and
`gradient_norm_pre_clip` as the raw whole-model norm. The sustained v3
protocol locally clips the complete RGB observation module to
`closed_loop_perception_grad_clip_norm` during causal training and independently
clips `dynamics.interactions` to `interaction_grad_clip_norm`, then computes
`gradient_norm_pre_global_clip` and applies `grad_clip_norm` to the complete
model. The groups are disjoint and the trainer reconstructs the true original
whole-model norm before either local cap. Logs expose both raw local norms,
coefficients/applied norms, the raw total, pre-global norm, global and total
coefficients, and final `gradient_norm_applied`. The perception-local cap is
disabled during paired RGB pretraining, preserving that phase's original
whole-model clipping behavior. Raw per-batch total loss remains heterogeneous
at batch one because scenario, object count, association support, events, and
available horizons vary; checkpoint decisions use the fixed broad validation
manifest instead.

At the phase boundary the trainer carries the best supported measurement
candidate into causal optimization, starts fresh causal AdamW moments, and applies
`closed_loop_learning_rate_scale` (0.1 in current profiles) to protect
perception while downstream filter/dynamics objectives begin. Ordinary broad
rollout rejection retains the fixed safe deployment incumbent but does not
discard a finite supported mutable measurement candidate. Explicit absolute or
fixed-reference-relative support collapse instead restores the verified
rollout incumbent and resets Adam. A causal optimizer step requires
differentiable trajectory/state/parameter support or supported persistent
fast-ROI slots; global auxiliary discovery alone cannot consume it.
Unsupported draws advance the deterministic sample counter and retry up to the
configured cap without advancing optimizer state. Global
discovery/backbone parameters remain trainable for
`closed_loop_global_trainable_steps` (512 in the repaired v3 profile), then
freeze while the ROI updater,
filter, dynamics, and identifier continue learning. Fast ROI losses follow the
persistent belief-slot assignment on every usable frame.

Privileged simulator targets map losses and metrics but cannot create runtime
evidence. A first target-to-runtime-ID mapping is admitted only within the
same `0.5 m` physical selection gate used for reported accuracy, with the gate
applied before Hungarian assignment; an established live-ID mapping remains
locked to expose identity swaps. Births do not open parameter supervision.
Drag/restitution history advances only from accepted distance-gated runtime
associations and resets whenever a target is next observed under a different
runtime ID.

Validation support is equally explicit. If the pooled manifest has no valid
current or configured-horizon physical mapping, the trainer retains additive
zero counts, marks the selection metric unsupported, writes a rejected
numbered/reference diagnostic artifact, and continues without a rollout
incumbent. It does not emit a synthetic zero RMSE or abort the run. Promotion
also requires declared per-scenario minima for label-predictable targets,
matched targets, and independently supported episodes at every horizon.
Training and evaluation censor deterministic point/event/correction metrics
scene-wide after an unseen external actuation while retaining those realised
outcomes for forecast likelihood/calibration and publishing both support
counts.

Collision logits describe occurrence over exact observation windows. Training
expands each requested horizon into `[h-dt_obs, h]` query boundaries, aggregates
internal-substep impacts, and applies bounded positive weighting through
`collision_positive_weight_max`. Correction training retains a small sparsity
term and adds current/future improvement hinges against a detached prior.

## Current promoted continuation

Accuracy-v4 resumed the selected step-584 perception checkpoint for 64
closed-loop RGB updates on CPU:

```bash
conda run --no-capture-output -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name accuracy-closed-structured-v4 \
  --resume runs/accuracy-depth-finetune-v1/checkpoints/best_measurement.pt \
  --device cpu \
  --seed 17 \
  --set training.steps=648 \
  --set training.rgb_pretrain_steps=584 \
  --set training.train_episodes=64 \
  --set training.validation_episodes=32 \
  --set training.batch_size=2 \
  --set training.fixed_dataset=true \
  --set training.learning_rate=0.0002 \
  --set training.weight_decay=0.0001 \
  --set training.eval_every=16 \
  --set training.checkpoint_every=16 \
  --set training.log_every=4 \
  --set training.measurement_validation_frames=16
```

The first segment was intentionally stopped after step 608 had been selected,
avoiding redundant full validation. Training then resumed from that selected
step with `eval_every=1000`, `checkpoint_every=20`, and `log_every=4`, while
keeping the other settings above, and completed step 648. The resumed segment
reported `699.7628 s`; it is not added to the earlier segment as though this
were one uninterrupted wall-clock measurement. Full validation selected step
648 at rollout-position loss `0.0119829765`.

The promoted artifact is
`runs/accuracy-closed-structured-v4/checkpoints/best_rollout.pt`. Promotion was
confirmed on paired ROI-local selection and confirmation manifests; it was not
based on training loss alone.

Metric-scale probes did not justify replacing learned depth. Mean-radius
analytic depth produced approximately `0.795 m` error versus `0.148 m` for the
learned estimate, and a photometric-radius variant failed confirmation.
Structured image centres therefore do not imply analytic depth. A two-frame
anisotropic position-slope velocity estimate is a future experiment, not an
implemented or promoted training path.

The tiny convergence profile uses 64 measurement steps and six jointly
trainable closed-loop steps. Fast inverse-depth deltas remain gated at the
analytic prior until a separately trained ROI checkpoint passes held-out
per-mode correction tests.

## Scaled generalization curriculum

`configs/scaled_curriculum.yaml` keeps the same online architecture and expands
the shared model to about 1.90 million parameters, versus about 0.16 million in
the selected tiny all-scenario profile. It declares 4,096 training episodes,
256 validation episodes, and 256 test episodes across the same eight balanced
scenario families. Seeds vary continuous initial state, physical parameters,
camera motion, object count, appearance, event timing, and observation noise
within those families.

The 48,000-step schedule draws 48,000 episode examples at batch size one,
approximately 11.7 passes through the deterministic seed manifest. The small
microbatch and eight-step TBPTT window bound the retained causal graph for the
larger model; four loader workers overlap deterministic rendering with MPS/GPU
training. With `fixed_dataset: false`, episodes are generated on demand and
frame/window locations are resampled rather than retaining the rendered
dataset in memory.
Use:

```bash
python train.py --config configs/scaled_curriculum.yaml
python evaluate.py \
  --config configs/scaled_curriculum.yaml \
  --checkpoint <path> \
  --split validation
python evaluate.py \
  --config configs/scaled_curriculum.yaml \
  --checkpoint <path> \
  --split test
python evaluate.py \
  --config configs/scaled_curriculum.yaml \
  --checkpoint <path> \
  --split ood
```

The configuration prefers CUDA, then MPS, then CPU through `device: auto`.
CPU is suitable for smoke tests but not an efficient way to complete this
schedule. Run summaries record model parameter count, episode draws, nominal
dataset passes, split sizes, and scenario families so a short run cannot be
mistaken for the full protocol.

## Sustained shared-model accuracy campaign

### Supported v3 campaign

The 2 August supported-gradient audit stopped and preserved both earlier
sustained protocols. Use the v3 profile for new training:

```bash
python train.py \
  --config configs/sustained_accuracy_mps_v3.yaml \
  --initialize-from \
    runs/20260730-192625-scaled-sustained-e2e-v1/checkpoints/best_measurement.pt \
  --run-name "$(date -u +%Y%m%d-%H%M%S)-scaled-sustained-v3" \
  --device mps
```

For a multi-hour macOS run, launch the same command as a one-shot LaunchAgent
instead of using `launchctl submit`:

```bash
python scripts/launch_training_once.py \
  --label com.polceanum.orpheus.v3-$(date -u +%Y%m%d-%H%M%S) \
  --config configs/sustained_accuracy_mps_v3.yaml \
  --initialize-from \
    runs/20260730-192625-scaled-sustained-e2e-v1/checkpoints/best_measurement.pt \
  --run-name "$(date -u +%Y%m%d-%H%M%S)-scaled-sustained-v3" \
  --device mps
```

Use one captured timestamp for both label and run name in an actual launch.
The helper invokes the active `orpheus` Python through `caffeinate`, writes an
explicit plist, and sets `KeepAlive=false`; a failed process remains terminal.

This is a new weights-only curriculum, not `--resume`. It has 40 frames,
batch two, 16,384 unique training episodes, 8,192 measurement updates, 8,192
supported causal updates, explicit mature/cold and
stochastic/deterministic support, fixed global horizon denominators, forecast
NLL, and deterministic bounded trend-validation anchors. It adds finite
unsupported-draw retries, absolute/reference-relative coverage floors,
separate global/fast measurement normalization, adjacent cached ROI training,
and independent `1.0` RGB-perception and interaction-local causal clips before
the `2.0` whole-model clip. It also requires two consistent global/recovery
detections within `0.5 m` before permanent birth and limits global causal
perception adaptation to 512 updates. The safe deployment incumbent remains
separate from the mutable phase-handoff trajectory. Promotion still requires
at least 64 fresh balanced episodes and every broad guardrail.

`device.preference=mps` applies to the convolution-heavy measurement phase.
`device.closed_loop_preference=cpu` switches the same persistent model at the
phase boundary after resetting causal optimizer moments and runtime caches. A
matched batch-two benchmark measured approximately `9.16 s` data generation,
`7.15 s` forward, and `2.70 s` backward on CPU; the equivalent branch-heavy
causal update was about nine times slower in device compute on MPS. This is a
backend choice, not an architectural fork. Both devices, the handoff state,
and selector artifacts are part of the exact-resume protocol.

On PyTorch 2.10, `device.global_detector_cpu_on_mps=true` keeps the CNN and
fast ROI computation on MPS but pins only the small global proposal transformer
to CPU. The exact finite batch with seeds `1,2` and a `2x96x64x64` backbone
feature map produced NaN MPS weight gradients in all attention/MLP matrix
weights; the CPU block gives byte-identical detector outputs, finite gradients
back through the MPS feature copy, and finite AdamW updates. This execution
flag is part of measurement and rollout selector protocol hashes and exact
resume semantics.

Checkpoints are deserialized on CPU. Model loading copies weights to the
existing phase placement, while optimizer loading puts Adam moments on each
parameter owner and keeps non-capturable scalar steps on CPU. Evaluation and
demos use the same CPU-deserialization rule so a saved optimizer is not
duplicated in accelerator memory. In-place exact resume accepts only the
source run's `checkpoints/last.pt`; selector/numbered checkpoints require a new
run or `--initialize-from`. A pending terminal-validation marker is recoverable
without an optimizer update.

The v3 four-update qualification at
`runs/20260802-110951-convergence-v3-hierarchical-clip-smoke/` proves finite
hybrid execution, real causal support, hierarchical clipping, checkpointing,
and selection wiring only. A medium qualification must pass before launching
the full clean-source profile. Consult `project/STATUS.md` for current evidence
and do not call a short run convergence.

The final audited-tree smoke is
`runs/20260802-121629-convergence-v3-final-audit-smoke/`. It reran two paired
RGB and two causal updates after the contact, unmapped-ROI, support,
checkpoint, and per-scenario selector repairs. The RGB phase used MPS for the
backbone/ROI path and CPU for the proposal transformer; the causal phase used
CPU. Both causal draws had real trajectory/fast-slot support (`122/32` and
`161/38`), none was skipped, and the terminal checkpoint is complete.

That smoke intentionally overrides the mixture to `reference_pairs` and uses
two validation episodes, so it is a host execution/guardrail check rather than
the required balanced qualification. Its terminal pooled score improved, but
coverage regressed and selector version `5.0` rejected the candidate in both
the pooled and scenario slice. The full/medium commands must retain all eight
scenarios and at least one validation episode per scenario; the v3 profile
uses 32.

The first medium qualification was stopped and preserved at
`runs/20260802-123714-v3-medium-qualification/` from clean pushed commit
`c0acf16`. It declared 3,072 updates: 1,024 paired RGB plus the repository's
minimum meaningful 2,048-update causal interval. With batch two and
`train_episodes=6144`, it makes exactly 6,144 deterministic episode draws over
one balanced manifest; the shuffled phase-specific scenario counts are
expected to be close to, but are not assumed to equal, 256/512 examples per
scenario. Keep the profile's 32 validation episodes, eight anchors, and
512-update evaluation cadence unchanged. Its first causal validation improved
conditional RMSE but exposed broad birth/identity/collision collapse and was
rejected by 38 guardrails. The trainer was stopped after logged step 1776 and
the job removed. Do not exact-resume its step-1728 `last.pt`: the repaired
lifecycle, association, target-supervision, and perception-gradient semantics
require a new timestamped run and protocol. See `project/STATUS.md` for the
exact audit.

The repaired committed tree passed a clean four-update host smoke at
`runs/20260803-000212-collapse-repair-host-smoke/`. Its paired RGB phase used
MPS with the CPU proposal-transformer workaround, and its causal phase used
CPU. Both causal updates had real trajectory and fast-ROI support, the new
perception-local cap reduced raw perception norms above `3.1` to `1.0` without
scaling interaction norms below `1.0`, no update was skipped, and terminal
validation/checkpointing completed. The two-episode `reference_pairs` candidate
was rejected for coverage/short-y guardrails, so this is device/protocol
qualification only. A new eight-scenario medium run, not this smoke, must
provide trend evidence.

The attempted replacement at
`runs/20260803-000858-v3-collapse-repair-qualification/` never completed its
step-zero imported-reference validation and took zero optimizer steps. A
per-observation external-impulse rate of `0.12` removed deterministic
one-second support from every fixed impulse episode; the resulting assertion
terminated the trainer. Its `launchctl submit` job then restarted more than
2,284 times, with later attempts failing against the occupied directory. The
job is removed and the run is a preserved failure artifact, not an active
campaign or convergence evidence.

The first protocol-10 replacement at
`runs/20260803-084843-v4-balanced-qualification/` was then stopped before any
optimizer update when an audit proved that `global_every_steps=3` actually
inserted three rather than two fast ROI frames. It was actively computing, not
deadlocked or relaunched, but its silent full-manifest validation made progress
impossible to inspect. Preserve it as a zero-update diagnostic.

The next protocol-11 replacement at
`runs/20260803-101108-v5-protocol11-balanced-qualification/` was also stopped
before any optimizer update after five visible, finite validation episodes.
Its mean completed-batch time was `117.380 s`, versus
`25.305 s/episode` for a matched repaired foreground control. The generated
LaunchAgent incorrectly declared explicitly requested training as
`ProcessType=Background`, and observed CPU use fell from roughly `525%` to
`100–198%`. The run is a preserved launch-throughput diagnostic, not an
accuracy result.

Historical sustained runs through 3 August used simulator `sphere_world_v4`,
rollout protocol 12, and
selection metric 6. Cadence three is exactly
`GLOBAL, FAST, FAST, GLOBAL`; historical cadence-three reports used actual
cadence four and are not selector-comparable. The impulse rate is `0.02` per
observation interval.
Promotion requires each scenario to meet its per-horizon label-opportunity and
matched-target floors and its supported-episode floor. Deterministic evaluator
metrics use the same unseen-actuation censor as training, while calibration
retains stochastic futures. Launch macOS training with
`scripts/launch_training_once.py`; its generated LaunchAgent has
`KeepAlive=false` and Standard/default process classification, and `train.py`
writes explicit starting, failure, and completed state artifacts. Never add
`ProcessType=Background` to a user-requested sustained campaign. During
validation, tail stdout or inspect
`training_progress.json` for exact per-episode counts, timings, seed/scenario,
PID, and protocol hash; partial progress never becomes selector evidence.
Training workers start only on the first actual draw. Post-step and checkpoint
finite-state checks terminate before corrupt parameters/moments can become a
resumable artifact. See `project/STATUS.md` for the exact failures, smokes, and
fixed-manifest support evidence.

Protocol 12 also aligns float32 observation intervals with the simulator's
nominal physics tick count and reuses one validated prepared propagation for
causal supervision plus ingestion. Do not resume a protocol-11 selector as if
these recursive numerical semantics were unchanged. Run a matched
Standard/default-priority timing check before the next full qualification.
A reduced CPU smoke at
`runs/20260803-110550-v6-protocol12-one-update-smoke/` completed one supported,
finite production-model causal update and terminal validation. Its step-one
candidate slightly regressed the fixed reference and was rejected, so the
incumbent remained authoritative. Treat this as optimizer/checkpoint/selector
wiring evidence only, not as an accuracy trend.

The full unshortened continuation at
`runs/20260803-112948-v6-protocol12-full-convergence/`, initialized from that
smoke's accepted `best_rollout.pt`. It uses the profile's 16,384 steps and
episodes, 8,192-step MPS measurement phase, 8,192-step CPU closed-loop phase,
32-episode fixed validation, 128-step checkpoints, and 512-step selector
interval. The trainer and supervisor were killed by macOS memory pressure at
step 11,776, after the durable step-11,648 checkpoint. It has no completed
summary and is a failed/incomplete campaign, not convergence evidence.

New sustained runs use rollout protocol 13. Deployment support and mutable
optimizer viability are now separate: pooled coverage below the absolute
floors still restores the verified incumbent and resets optimizer state, while
a finite pooled candidate that fails only a scenario, reference-relative, or
broad deployment guardrail remains the mutable iterate so later causal updates
can repair it. Validation checkpoints record both decisions. The macOS profile
uses two non-persistent workers with one prefetched batch each, releases the
previous accelerator allocator cache at phase transitions, and logs process
maximum RSS. `train.py` writes `starting`, `running`, and terminal states; the
supervisor converts a proved external trainer exit into an ordinary durable
training failure.

The first production protocol-13 continuation is
`runs/20260806-213753-v7-protocol13-causal-convergence/`. It starts
weights-only from the rejected-but-finite protocol-12 step-10,240 candidate and
sets `training.rgb_pretrain_steps=0`, `training.steps=8192`. The configured
measurement device remains MPS-capable, while this causal-only invocation uses
the explicit CPU closed-loop device. Treat its initial full-manifest
validation, launch state, and empty stderr as health evidence only until real
optimizer/selector metrics are persisted.

Specification 1.15 preserves the configured global/fast measurement
coefficients when either branch is frozen or unsupported. It also permits one
explicit causal scope transition. The evidence-led frozen-perception campaign
uses:

```bash
python train.py \
  --config configs/sustained_accuracy_mps_v3.yaml \
  --initialize-from <accepted-reference-checkpoint> \
  --run-name <timestamp>-staged-fast-roi-state-dynamics \
  --device mps \
  --set training.rgb_pretrain_steps=0 \
  --set training.steps=8192 \
  --set training.closed_loop_global_trainable_steps=0 \
  --set training.closed_loop_trainable_scope=fast_roi \
  --set training.closed_loop_late_trainable_scope=state_dynamics \
  --set training.closed_loop_scope_transition_steps=512
```

The first 512 completed causal updates train only the ROI-exclusive projection
and updater. From update 513 onward those tensors freeze and only belief
update, identifier, and dynamics adapt. The resolved scopes and boundary are
exact continuation semantics; changing them requires weights-only
initialization. This schedule does not promote the step-512 candidate or relax
any fixed-manifest guardrail.

The convergence supervisor is also active under launchd label
`com.polceanum.orpheus.convergence-20260803-112948`, monitoring initial trainer
PID `31197`. It waits for the complete 16,384-step summary, verifies every
selector/checkpoint link, and then applies the repository's fixed plateau rule:
4,096-step extensions, a four-validation 1% tail decision, and a 24,576-step
hard limit. Its durable log is
`runs/20260803-112948-v6-protocol12-full-convergence/convergence_supervisor.jsonl`;
stdout/stderr are
`/private/tmp/20260803-112948-convergence-supervisor.{stdout,stderr}.log`.
Do not manually start a second trainer or supervisor for this run.

As of 2026-08-04 the measurement phase has reached step 3584. The raw fixed
score reached `4.868897` at step 2048, then `5.029407` and `5.081339` at
2560/3072; all are materially better than the `11.901029` imported score.
Fast-ROI MAE remains the non-negotiable blocker at approximately `0.32 m`
versus the `0.189315 m` incumbent, so no trained measurement candidate has
been promoted. Continue the declared phase and do not weaken the selector or
start a competing trainer while its fixed validation is in flight.

### Superseded v2 campaign

`configs/sustained_accuracy_mps_v2.yaml` and
`runs/20260801-232229-scaled-sustained-v2/` are retained audit controls. The
trainer and supervisor were stopped at logged step `9576` after 121 of 173
logged causal rows were found to have exactly zero gradient and the measurement
handoff collapsed current/future coverage. Repairing optimizer support,
selection semantics, and gradient scaling defines a new protocol, so v2 must
not be resumed in place or compared as if it were a completed convergence run.

### Preserved legacy campaign

`configs/sustained_accuracy_mps.yaml` is the tractable successor to launching
the nominal 48,000-step profile unchanged on one Mac. It retains the same
1.90M-parameter architecture and all eight scenario families, initializes from
the selected fixed-scale point/scale runtime, and keeps the rejected
change-point/outgoing/intervention heads disabled.

The declared minimum is 8,192 measurement updates (two complete 4,096-episode
passes) plus 4,096 independently sampled causal windows (one complete nominal
causal pass, about 512 windows per scenario). Every frame still advances and
supervises the persistent belief. Training limits the expensive recursive
forecast to one earliest eligible anchor per four-frame TBPTT window; that
anchor evaluates every configured horizon supported from its timestamp. The
window sampler mixes collision-conditioned and long-horizon windows so all
declared 0.1–1.0-second horizons receive campaign-wide support. Validation
remains unbounded and scores all eligible posterior anchors.

The campaign launched on 2026-07-30 predates the corrected per-axis global
horizon normalization and joint sampler. Its config explicitly keeps both
legacy controls false so an automatic in-place continuation cannot change the
training protocol halfway through. It was manually superseded at logged step
9400 after the audit proved that the old handoff discarded all perception
updates from its mutable causal path. Its artifacts remain valid historical
evidence, but its supervisor and trainer are stopped and it must not be
extended or described as converged.

```bash
python train.py \
  --config configs/sustained_accuracy_mps.yaml \
  --initialize-from \
    runs/20260729-084712-scaled-point-scale-trajectory-v1/checkpoints/runtime_ablation.pt \
  --run-name scaled-sustained-e2e-v1 \
  --device mps
```

Do not judge causal convergence before 2,048 causal updates, and always
complete the 4,096-window minimum. After that minimum,
`scripts/supervise_convergence.py` verifies `train_summary.json`, `last.pt`,
the linked best/reference selectors, every numbered validation candidate,
their protocol hashes, and their actual model-tensor hashes before deciding.
It resumes the mutable training iterate from `last.pt`, preserving optimizer
and RNG state; `best_rollout.pt` remains a separate immutable selection
artifact.

The predeclared decision rule is:

- continue for another complete 4,096-update causal block when the best
  guardrail-safe checkpoint is in the final 1,024 updates and improves the
  primary score by at least 1%;
- declare a plateau only when the exact four most recent 512-step validation
  points accepted no candidate and even their best raw primary score improved
  less than 1% over the safe pre-window incumbent;
- treat missing or contradictory four-point evidence as inconclusive and
  continue for another complete block;
- stop at 24,576 total updates. If the four-point plateau rule is satisfied at
  that boundary, report `plateau`; otherwise report `limit_hit`, which is a
  budget stop and not a convergence claim.

The supervisor is restart-aware and never intentionally overlaps two
extension trainers. An already-running initial trainer can be monitored with
`--initial-trainer-pid`; disappearance before a verified summary records an
explicit failure and exits nonzero. A failed extension is also recorded and
is not retried forever by a persistent job. Each completed segment is recorded
under `runs/<run>/convergence/`; machine-readable events, state, and the final
decision are written to `convergence_supervisor.jsonl`,
`convergence_supervisor_state.json`, and `convergence_report.json`.

For the preserved legacy campaign, the supervisor was launched persistently
with these
arguments:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python \
  scripts/supervise_convergence.py \
  --config configs/sustained_accuracy_mps.yaml \
  --run runs/20260730-192625-scaled-sustained-e2e-v1 \
  --device mps \
  --initial-trainer-pid 37360 \
  --initial-launchctl-label \
    com.polceanum.orpheus.sustained-20260730-192625 \
  --maximum-total-steps 24576
```

That historical job has been booted out. Do not relaunch it.
The selected checkpoint still requires balanced fresh-validation confirmation
of at least 64 episodes before the reserved test split is used.
