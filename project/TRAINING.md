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
window only and is never passed to the RGB runtime. Position and velocity losses
are separate for both current state and future rollout. Current default physical
weights are
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

```bash
python train.py \
  --config configs/sustained_accuracy_mps.yaml \
  --initialize-from \
    runs/20260729-084712-scaled-point-scale-trajectory-v1/checkpoints/runtime_ablation.pt \
  --run-name scaled-sustained-e2e-v1 \
  --device mps
```

Do not judge causal convergence before 2,048 causal updates. Complete all
4,096, then call a plateau only after four consecutive 512-step validations
have no accepted candidate and under 1% primary-score improvement. Extend by a
further 4,096 causal updates only if the best checkpoint occurs within the
final 1,024 updates and improves at least 1% without a guardrail regression.
The selected checkpoint still requires a balanced fresh-validation
confirmation of at least 64 episodes before the reserved test split is used.
