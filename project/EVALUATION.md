# Evaluation

Held-out evaluation is causal and RGB-only unless a report is explicitly marked
oracle ablation. It measures state/forecast error by horizon, correction
improvement, assignment coverage, distance-gated detection/identity, collision
events, runtime parameter observability/update gates, uncertainty
coverage/NLL/sharpness, finite outputs, and component latency.

Transparent static, constant-velocity, default analytic, and explicitly
labelled oracle-parameter analytic baselines use the same episode contracts and
forecast masks. Simulator labels align metrics but are never fed back to the
runtime. Full results include JSON plus Markdown and never use future
observations to score an earlier belief.

Checkpoint comparison uses the reserved validation range, never the test range.
`fresh_validation` begins after the number of validation episodes stored in the
checkpoint by default. `--seed-offset` selects an explicit later range and is
rejected if it overlaps trainer validation:

```bash
# Candidate selection after trainer-validation seeds 100000–100031.
python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint <path> \
  --split validation \
  --seed-protocol fresh_validation \
  --seed-offset 32 \
  --set evaluation.episodes=32

# One-time untouched confirmation; do not tune on this report.
python evaluate.py \
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

Accuracy-v4 was compared with its step-584 initialization on the same
ROI-local selection and confirmation manifests. On confirmation seeds
`100064–100095`, step 648 changed 0.1/0.25/0.5-second RMSE from
`0.134093 / 0.174492 / 0.231253 m` to
`0.132424 / 0.171900 / 0.226994 m`, while collision F1 increased
`0.594203 → 0.608059`. Current position, velocity, and perturbation recovery
had small mixed changes, so promotion rests on the repeated forecast and event
gains, not a claim that every metric improved. Reports:

- `runs/accuracy-closed-structured-v4/evaluation/select32/report.md`
- `runs/accuracy-closed-structured-v4/evaluation/confirm32/report.md`

The step-584 selection and accuracy-v3 final reports remain historical at
`runs/accuracy-roi-local-v3/`. Once model choices were frozen, the promoted
step-648 checkpoint was evaluated on the reserved standard-test block:

```bash
conda run --no-capture-output -n orpheus python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/accuracy-closed-structured-v4/checkpoints/best_rollout.pt \
  --split test \
  --seed-protocol standard \
  --seed-offset 64 \
  --device cpu \
  --set evaluation.episodes=32 \
  --output runs/accuracy-closed-structured-v4/evaluation/final-test32
```

That report covers exactly seeds `200064–200095`; it is the final evidence for
the promoted checkpoint and was not used for additional tuning. The final
collision precision/recall/F1 is `0.765217 / 0.550000 / 0.640000`; nominal-90%
forecast coverage is `86.95%`.

An exhaustive validation threshold probe did not improve collision skill:
predicted probabilities were already concentrated near `0.018` and `0.998`,
including structural false positives/negatives. The public threshold remains
`0.5`; moving it would disguise state/timing errors rather than calibrate them.

Current limitations are recorded rather than hidden: physics-violation and
failure-plot suites are not yet exported, and parameter MAE is withheld when
localization fails the configured 0.5 m metric gate. Collision F1 and nominal
90% uncertainty coverage remain below their acceptance targets.
