# Project status

**Date:** 2026-07-26  
**Specification:** `PROJECT_SPEC.md` 1.0  
**Current state:** runnable Milestone 1 vertical slice; quantitative acceptance
is not achieved

## What works

- A deterministic 3-D RGB sphere simulator provides timestamps, calibrated
  fixed/moving cameras, padded identities and physical state, depth-ordered
  rendering/occlusion, collisions, events, and disjoint seed splits.
- `WorldBelief` is the persistent modality-independent source of truth. The
  runtime accepts timestamped packets, advances a prior, projects measurements,
  associates, computes innovation, corrects the posterior, updates lifecycle
  and observable slow parameters, and revises arbitrary-time rollouts.
- Dynamics combine analytic gravity/drag/quaternions, bounded modal state,
  learned structured interactions/residuals, explicit contact/event jumps, and
  diagonal uncertainty propagation.
- RGB has a global discovery detector and a shallow residual ROI path. ROI
  caches are identity-checked and invalidated on global recovery or lifecycle
  changes. Projected overlap distinguishes occlusion from out-of-view misses.
- The debug oracle is registered only when explicitly enabled. All reported
  training, evaluation, and demo runs below consumed RGB plus known calibration;
  simulator state was used only for labels, metrics, and labelled baselines.
- Training supports global and fast-path measurement supervision, causal
  closed-loop unrolls, injected perturbations, horizon-weighted state/event/
  uncertainty/parameter losses, atomic checkpoints, resume, and local JSONL.
- Evaluation uses common forecast masks for the model and baselines, reports
  dropped tracks, separates assignment coverage from 0.5 m distance-gated
  detections, and reports the identifier's actual observability/gates/updates.

## Environment

- Python: 3.10.20 in conda environment `orpheus`
- Process architecture: x86_64
- PyTorch: 2.10.0; left installed and unchanged
- MPS: compiled and available in direct conda Python probes; real forward and
  backward training completed on `mps`
- CUDA: unavailable
- Precision used: float32
- Some `conda run ... pytest` launcher subprocesses report MPS unavailable and
  skip three conditional tests. Running those tests through
  `conda run -n orpheus python -m pytest` exercises MPS successfully.

## Current CPU evidence

Training command:

```bash
conda run -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name milestone1-tiny-overfit-cpu-v4-final
```

Observed result: 24 steps on CPU (12 RGB pretraining, 12 closed-loop), 84.888 s,
finite losses/gradients, no oracle input. Global measurement loss moved from
2.25014 at step 1 to -0.390325 at step 12. Closed-loop total loss was 1.802052
at step 13 and 0.670239 at step 24; the final training perturbation correction
improved current error by 0.018978 m. The genuine closed-loop validation rollout
loss was 0.463825.

Artifacts:

- `runs/milestone1-tiny-overfit-cpu-v4-final/checkpoints/best_rollout.pt`
- `runs/milestone1-tiny-overfit-cpu-v4-final/checkpoints/best_measurement.pt`
- `runs/milestone1-tiny-overfit-cpu-v4-final/checkpoints/last.pt`
- `runs/milestone1-tiny-overfit-cpu-v4-final/metrics.jsonl`
- `runs/milestone1-tiny-overfit-cpu-v4-final/train_summary.json`

Held-out command:

```bash
conda run -n orpheus python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/milestone1-tiny-overfit-cpu-v4-final/checkpoints/best_rollout.pt \
  --split test \
  --output runs/milestone1-tiny-overfit-cpu-v4-final/evaluation/best-test-final
```

On two test episodes, model position RMSE was 0.735898 / 0.717700 /
0.685969 m at 0.10 / 0.25 / 0.50 s. Constant-velocity RMSE was 0.738739 /
0.730434 / 0.853260 m; static RMSE was 0.719683 / 0.714149 / 0.694280 m.
Injected-perturbation improvement was 0.083751 m (6.524%, positive on all 12
evaluated object-horizons). Forecast 90% coverage was 94.298%.

Assignment coverage was 64/64, but no assignment was within the truthful 0.5 m
detection gate. Collision F1 was 0. Identifier diagnostics recorded 46 drag and
9 restitution updates above the gate threshold, but distance-gated parameter
MAE is unavailable because localization failed. These are implementation
signals, not successful physical identification.

Report:
`runs/milestone1-tiny-overfit-cpu-v4-final/evaluation/best-test-final/report.md`.
The same protocol was also run on `last.pt`; best and last contain the same
step-24 model tensors and produced identical accuracy metrics. Its report is
`runs/milestone1-tiny-overfit-cpu-v4-final/evaluation/last-test-final/report.md`.

Demo command:

```bash
conda run -n orpheus python demo.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/milestone1-tiny-overfit-cpu-v4-final/checkpoints/best_rollout.pt \
  --output demo_outputs/milestone1-tiny-overfit-cpu-v4-final
```

The 16-frame held-out RGB-only demo used 4 global and 12 fast ROI updates.
Mean ordinary prior-to-posterior improvement was **-0.020306 m** for current
state and **-0.016870 m** for future error: correction worsened this demo on
average. The artifact is retained because it truthfully exposes that failure.

## Current MPS evidence

Reduced explicit MPS smoke:

```bash
conda run -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name milestone1-mps-smoke-final \
  --device mps \
  --set training.steps=3 \
  --set training.rgb_pretrain_steps=2 \
  --set training.tbptt_steps=2 \
  --set training.batch_size=1 \
  --set training.train_episodes=1 \
  --set training.validation_episodes=1 \
  --set training.eval_every=3 \
  --set training.checkpoint_every=3 \
  --set training.log_every=1 \
  --set evaluation.episodes=1
```

Observed result: 3 finite optimizer steps in 90.996 s on `mps`, including a
two-frame closed-loop global-to-fast ROI backward pass with 4 matched
object-frames and direct fast-path supervision. This validates the implementation
path only, not convergence.

After that smoke succeeded, the full-size 96×96 `toy_mps` architecture was
exercised with a deliberately reduced two-step budget:

```bash
conda run -n orpheus python train.py \
  --config configs/toy_mps.yaml \
  --run-name milestone1-toy-mps-scaled-smoke \
  --device mps \
  --set training.steps=2 \
  --set training.rgb_pretrain_steps=1 \
  --set training.tbptt_steps=2 \
  --set training.batch_size=1 \
  --set training.train_episodes=1 \
  --set training.validation_episodes=1 \
  --set training.eval_every=2 \
  --set training.checkpoint_every=2 \
  --set training.log_every=1 \
  --set evaluation.episodes=1
```

Observed result: one global pretraining step and one full RGB closed-loop step
completed on MPS in 132.157 s with finite losses/gradients, 6 matched
object-frames, direct fast-path supervision, and a real validated rollout
checkpoint. Its correction improvement was -0.005626 m, so this is hardware
compatibility evidence rather than learning success.

## Final validation commands

```bash
conda run -n orpheus python -m pip install -e . --no-deps
```

Passed earlier in this session; `importlib.metadata` and `world_model` both
report version 0.1.0. `--no-deps` deliberately leaves the installed PyTorch
untouched.

```bash
conda run -n orpheus python -m ruff check .
conda run -n orpheus python -m ruff format --check .
```

Passed: all checks clean; 133 Python files already formatted.

```bash
conda run -n orpheus python -m pytest
```

Passed: 88 tests in 34.49 s, including the three MPS-specific tests.

```bash
conda run -n orpheus pytest
```

Passed: 85 tests, 3 MPS-conditional skips in 31.34 s. This launcher observes a
different MPS availability result, as noted above.

```bash
PYTHONPYCACHEPREFIX=/tmp/orpheus_compile_cache \
  conda run -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
```

Passed with no output.

```bash
conda run -n orpheus python train.py \
  --config configs/toy_mps.yaml --dry-run
```

Passed and selected `mps`; the resolved default remains 96×96, 72 frames, and
3,000 training steps.

Local untrained-path benchmarks were also saved:

- CPU: `runs/milestone1-tiny-overfit-cpu-v4-final/benchmark_cpu.json`
  (global RGB 5.123 ms, first ingest 12.555 ms, 0.1/0.5/1.0 s rollout
  470.052 ms; three repeats).
- MPS: `runs/milestone1-mps-smoke-final/benchmark_mps.json`
  (global RGB 23.227 ms, first ingest 130.664 ms, rollout 4,400.476 ms; two
  repeats). Small workloads and CPU association/substep overhead make this a
  compatibility profile, not a favourable accelerator benchmark.

## Known limitations and failed acceptance gates

- The tiny held-out localization gate has 0 true detections; Milestone 1 is not
  complete despite assignment coverage and finite forecast metrics.
- Perturbation recovery is 6.52%, below the recommended 20% improvement.
- Collision F1 is 0; event learning needs event-balanced episodes and longer
  training.
- Slow-parameter gates execute and learned identifier heads are nonzero, but
  restitution stays visually flat and drag moves only slightly on the demo.
  No distance-gated RGB parameter accuracy is established.
- The two short test episodes contained no usable distance-gated occlusion
  interval, so learned uncertainty/identity recovery through rendered
  occlusion is not empirically established beyond focused geometry/lifecycle
  tests.
- On this tiny CPU report the fast ROI mean latency (43.864 ms) is slightly
  slower than the global path (42.756 ms). The path is incremental in
  architecture but has no measured speed win at this scale.
- The demo worsens current and future error on average.
- Evaluation does not yet export the full physics-violation, collision-
  conditioned, occlusion-survival, and failure-plot suite from the specification.
- Multi-frame tentative birth confirmation, multiple-hypothesis association,
  estimated camera pose, continuous collision timing, real video, and a second
  modality remain future work.
- `identification.restitution_event_threshold` and
  `identification.ambiguity_gate` remain reserved configuration fields; the
  current updater gates on structured mode/event probability and the
  associator's ambiguity flag.
- The full 3,000-step `configs/toy_mps.yaml` protocol has not been run.

## Next concrete tasks

1. Train the full `toy_mps` schedule with event-balanced and occlusion-balanced
   sampling; evaluate more than two held-out episodes.
2. Improve calibrated RGB world localization until distance-gated recall is
   nonzero, then reassess correction, uncertainty, identity, and parameter MAE.
3. Add collision-conditioned and occlusion-survival metrics plus saved failure
   plots and physics-violation diagnostics.
4. Validate drag/restitution convergence on controlled RGB sequences with
   observable excitation, not only that the update gates execute.
