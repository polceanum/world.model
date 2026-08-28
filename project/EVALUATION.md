# Evaluation

Held-out evaluation is causal and observable-input-only. A report must identify
its modality explicitly, including whether it is RGB-only, RGB-D, or a clearly
marked oracle ablation. Evaluation measures state/forecast error by horizon,
correction improvement, assignment coverage, distance-gated detection/identity,
collision events, runtime parameter observability/update gates, uncertainty
coverage/NLL/sharpness, finite outputs, and component latency.

Transparent static, constant-velocity, default analytic, and explicitly
labelled oracle-parameter analytic baselines use the same episode contracts and
forecast masks. Simulator labels align metrics but are never fed back to the
runtime. Full results include JSON plus Markdown and never use future
observations to score an earlier belief.

For the supported generic workflow and options, start with:

```bash
conda run -n orpheus python evaluate.py --help
```

Checkpoint comparison for a newly generated generic run uses the reserved
validation range, never the test range.
`fresh_validation` begins after the number of validation episodes stored in the
checkpoint by default. `--seed-offset` selects an explicit later range and is
rejected if it overlaps trainer validation:

```bash
# Candidate selection after trainer-validation seeds 100000–100031.
conda run -n orpheus python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint <path> \
  --split validation \
  --seed-protocol fresh_validation \
  --seed-offset 32 \
  --set evaluation.episodes=32

# One-time untouched confirmation; do not tune on this report.
conda run -n orpheus python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint <selected-path> \
  --split validation \
  --seed-protocol fresh_validation \
  --seed-offset 64 \
  --set evaluation.episodes=32
```

The report records the protocol, intended role, offset, exact episode seeds,
and overlap checks. A selection manifest may be revisited while comparing
candidates; a confirmation manifest should remain untouched until the choice is
fixed. The reserved test split is for final assessment, not model selection.
These generic commands do not authorize access to any consumed qualification
manifest. A new capability rung requires its own frozen pre-access protocol and
fresh output paths.

## Historical pre-cleanup accuracy-v4 evidence

Before the 2026-08-26 repository cleanup, accuracy-v4 was compared with its
step-584 initialization on the same ROI-local selection and confirmation
manifests. On confirmation seeds
`100064–100095`, step 648 changed 0.1/0.25/0.5-second RMSE from
`0.134093 / 0.174492 / 0.231253 m` to
`0.132424 / 0.171900 / 0.226994 m`, while collision F1 increased
`0.594203 → 0.608059`. Current position, velocity, and perturbation recovery
had small mixed changes, so promotion rests on the repeated forecast and event
gains, not a claim that every metric improved. The historical report paths
were:

- `runs/accuracy-closed-structured-v4/evaluation/select32/report.md`
- `runs/accuracy-closed-structured-v4/evaluation/confirm32/report.md`

The step-584 selection and accuracy-v3 final reports were under the historical
path `runs/accuracy-roi-local-v3/`. The promoted step-648 checkpoint was at
`runs/accuracy-closed-structured-v4/checkpoints/best_rollout.pt` and was
evaluated once on the reserved standard-test block. These paths were removed
from the active workspace during the verified 7.6-GiB cleanup. They remain
recoverable as compact metadata from the ignored pre-generalization archive and
as project history in commit `c16acc99ef13757fc8f88528bfd0d66db4a2f4fd`.
The historical invocation is intentionally not reproduced as an executable
command: it is provenance, not a supported launch or permission to regenerate
consumed evidence.

That report covers exactly seeds `200064–200095`; it is the final evidence for
the promoted checkpoint and was not used for additional tuning. The final
collision precision/recall/F1 is `0.765217 / 0.550000 / 0.640000`; nominal-90%
forecast coverage is `86.95%`.

Historically, an exhaustive validation threshold probe did not improve
collision skill: predicted probabilities were already concentrated near
`0.018` and `0.998`, including structural false positives/negatives. The
historical public threshold remained `0.5`; moving it would have disguised
state/timing errors rather than calibrated them.

Those historical broad-model limitations are recorded rather than hidden.
Physics-violation and failure-plot suites were not exported, and parameter MAE
was withheld when localization failed the configured 0.5 m metric gate.
Collision F1 and nominal-90% uncertainty coverage remained below their then
acceptance targets. Current rung status, exact consumed-evidence boundaries,
and limitations are authoritative in `project/STATUS.md` and
`project/GENERALIZATION_LADDER.md`.
