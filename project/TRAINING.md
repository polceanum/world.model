# Training

Training uses deterministic on-the-fly sphere episodes and one architecture
through measurement pretraining, closed-loop RGB correction, perturbation
recovery, horizon-weighted rollout loss, event supervision, and observable
parameter losses. Ground truth is supervision rather than an ordinary runtime
input. Each optimizer batch resets belief and causally unrolls one configured
window; it does not re-encode a history clip.

Checkpoints contain model/optimizer/step, resolved config, RNG state, metrics,
specification version, git metadata, simulator version, device, and precision.
Metrics are written locally as JSONL. Measurement-only and closed-loop
validation select separate `best_measurement.pt` and `best_rollout.pt`
checkpoints, preventing a pretraining loss from being mislabeled as rollout
quality. Fixed-dataset measurement training sweeps all frames independently of
loader-batch position. Measurement validation averages configured frames and
selects by calibrated backprojected world-position MAE, because summed
heteroscedastic NLL can be negative without implying usable localization.

At the phase boundary the trainer restores the best localized measurement
checkpoint and applies `closed_loop_learning_rate_scale` (0.1 in current
profiles) to protect perception while downstream filter/dynamics objectives
begin. The tiny convergence profile uses 64 measurement steps and six
closed-loop steps. Fast inverse-depth deltas remain gated at the analytic prior
until a separately trained ROI checkpoint passes held-out per-mode correction
tests.
