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
Measurement checkpoints select by calibrated backprojected world-position MAE.
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
reference metrics to real checkpoint weights. A resumed run reuses those
metrics only when the linked files and hashes verify. This prevents a rejected
`last.pt` or copied numbered snapshot from carrying better incumbent metrics
without the corresponding model state.

RGB supervision now includes metric-space position after calibrated
backprojection. It uses a smooth-L1 (Huber) term plus a diagonal Gaussian NLL,
in addition to existence, image geometry, colour, measurement NLL, visibility,
and appearance losses. When structured disc centres replace learned centres in
the forward pass, the unrefined learned centre is retained as `raw_centre` and
receives its own smooth-L1 auxiliary loss. The default measurement weights are:

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

Training logs retain legacy `gradient_norm` as the norm returned before
clipping and additionally report `gradient_norm_pre_clip` and
`gradient_norm_applied`, plus the exact `gradient_clip_coefficient`. The
applied norm is bounded by `grad_clip_norm` and is the relevant
update-magnitude diagnostic. Raw per-batch total loss remains heterogeneous at
batch one because scenario, object count, association support, events, and
available horizons vary; checkpoint decisions use the fixed broad validation
manifest instead.

At the phase boundary the trainer restores the best localized measurement
weights, starts fresh causal AdamW moments, and applies
`closed_loop_learning_rate_scale` (0.1 in current profiles) to protect
perception while downstream filter/dynamics objectives begin. The restored
measurement candidate must pass the same broad rollout guardrails or the fixed
pre-campaign runtime is restored. Global discovery/backbone parameters remain trainable for
`closed_loop_global_trainable_steps`, then freeze while the ROI updater,
filter, dynamics, and identifier continue learning. Fast ROI losses follow the
persistent belief-slot assignment on every usable frame.

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

### Corrected v2 campaign

The 1 August convergence-integrity audit superseded the active legacy campaign
without deleting it. Use the corrected profile for new training:

```bash
python train.py \
  --config configs/sustained_accuracy_mps_v2.yaml \
  --initialize-from \
    runs/20260730-192625-scaled-sustained-e2e-v1/checkpoints/best_measurement.pt \
  --run-name "$(date -u +%Y%m%d-%H%M%S)-scaled-sustained-v2" \
  --device mps
```

This is a new weights-only curriculum, not `--resume`. It has 40 frames,
batch two, 16,384 unique training episodes, 8,192 measurement updates, 8,192
causal updates, explicit mature/cold and stochastic/deterministic support,
fixed global horizon denominators, forecast NLL, and deterministic bounded
trend-validation anchors. The safe deployment incumbent remains separate from
the mutable phase-handoff trajectory. Promotion still requires at least 64
fresh balanced episodes and every broad guardrail.

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

The clean-source v2 campaign is active at
`runs/20260801-232229-scaled-sustained-v2/` from commit `df98f63`. Its submitted
trainer job is
`com.polceanum.orpheus.sustained-v2-20260801-232229`; the paired supervisor is
`com.polceanum.orpheus.convergence-v2-20260801-232229`. The supervisor waits
for the complete 16,384-step segment, then applies only verified 4,096-step
extensions up to 24,576 total steps. Treat the initial 32-episode validation
latency as expected and consult `project/STATUS.md` for current PID/log paths.

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
