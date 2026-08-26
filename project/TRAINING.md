# Training and qualification

## Active policy

Orpheus scales through deterministic, fail-fast capability rungs. Each rung
changes one independently measurable source of complexity, binds disjoint
train/selector/confirmation/final manifests before access, and preserves every
accepted lower-rung gate. A failed family stops; thresholds and final sets are
not repeatedly tuned.

Runtime inputs must be observable modalities plus calibration, timestamps, and
declared priors. Simulator state may label losses and metrics but may not enter
a claimed RGB/RGB-D forward path. Continuous state, parameter, and rollout
learning uses ordinary PyTorch autograd. Analytic tensor equations provide
inductive bias; learned residual capacity is added only after an identifiable
structured baseline leaves measured error.

## Qualified base

The supported standalone developer command is:

```bash
conda run -n orpheus python scripts/run_minimal_toy_ladder.py \
  --config configs/minimal_differentiable_toy_cpu.yaml \
  --report runs/minimal_differentiable_toy_v2/report.json \
  --checkpoint runs/minimal_differentiable_toy_v2/model.pt
```

Do not rerun its published final manifest merely to regenerate provenance.
Future runs require fresh output paths and produce atomic, versioned,
weights-only project checkpoints whose SHA-256 is bound in the report.

## Current temporal rung

The runner and config are frozen, and focused implementation/static gates pass.
No development artifact has yet been created from clean committed source, and
all protected namespaces remain unopened. Commit the reviewed tree and run all
repository gates first; the runner rejects a dirty checkout and requires the
same clean source provenance throughout both phases.

Development is the only permitted first launch. Use fresh, non-aliasing output
paths that do not already exist:

```bash
conda run -n orpheus python scripts/run_temporal_free_motion_ladder.py \
  --phase development \
  --config configs/temporal_free_motion_toy_cpu.yaml \
  --report runs/temporal_free_motion_toy_v1/development_report.json \
  --checkpoint runs/temporal_free_motion_toy_v1/development_model.pt
```

This phase alone materializes development-train seeds
`31000000--31000031` and development-audit seeds
`31100000--31100015`. It performs exactly 32 batch-four AdamW updates and
writes an atomic project-compatible weights-only checkpoint plus a report that
must state `protected_data_materialized: false`. Inspect both artifacts and
independently record their full SHA-256 values before proceeding.

Only if the development report passes and is review-ready may the same clean
source commit run the one protected qualification command:

```bash
conda run -n orpheus python scripts/run_temporal_free_motion_ladder.py \
  --phase qualification \
  --config configs/temporal_free_motion_toy_cpu.yaml \
  --report runs/temporal_free_motion_toy_v1/qualification_report.json \
  --checkpoint runs/temporal_free_motion_toy_v1/development_model.pt \
  --development-report \
    runs/temporal_free_motion_toy_v1/development_report.json \
  --reviewed-checkpoint-sha256 <independently-reviewed-checkpoint-sha256> \
  --reviewed-report-sha256 <independently-reviewed-report-sha256>
```

Qualification reads each reviewed artifact once, verifies its supplied
64-hex SHA-256 plus exact config/source/protocol/model/report agreement, and
loads no optimizer state. Before protected access it exclusively creates
`runs/temporal_free_motion_toy_v1/qualification_attempt_2_access.json`.
That ledger makes architecture attempt 2 of 2 one-shot: do not delete or reuse
it to retry a failed family. Access is fail-fast and ordered as 16 selector
seeds, 16 confirmation seeds, then 32 final seeds; a failed earlier gate keeps
later data unopened. Do not edit code or documentation between development and
qualification.

The frozen rung uses 16 RGB observations over 0.75 seconds, a differentiable
exact linear-drag least-squares anchor-state fit, and analytic rollouts at
`0.1/0.25/0.5/1.0/2.0 s`. Every protected split must meet current
position/velocity, per-horizon position, future-velocity, centre/radius,
validity, trivial-baseline, semigroup, two-second per-mask-scalar gradient,
memory, and separated perception/state-only latency gates. Passing unit or
development checks is not convergence evidence.

The broad `train.py`, `evaluate.py`, and `demo.py` workflow remains available
for `OnlineWorldModel` smoke/integration checks, but no older sustained profile
is an active accuracy campaign or deployment incumbent. Exact-resume,
checkpoint, validation-support, and promotion integrity remain tested reusable
contracts.

Historical campaign commands and evidence through specification 1.51 remain
in Git commit `c16acc99` and the ignored local pre-generalization archive.
