# Getting started

Use the existing environment and do not reinstall PyTorch:

```bash
conda activate orpheus
pip install -e ".[dev]"
python train.py --config configs/toy_smoke.yaml
python evaluate.py --config configs/toy_smoke.yaml --checkpoint <path>
python demo.py --config configs/toy_smoke.yaml --checkpoint <path>
pytest
```

Use a validation-only selection manifest that does not reuse trainer
validation or test seeds with:

```bash
python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint <path> \
  --split validation \
  --seed-protocol fresh_validation \
  --set evaluation.episodes=16
```

The experimental causal RGB velocity path is intentionally off by default.
It can be ablated with
`--set model.rgb.temporal_velocity_enabled=true`; current project evidence
does not justify enabling it for primary accuracy claims.

`configs/toy_smoke.yaml` is the minimum executable smoke.
`configs/tiny_overfit.yaml` is the deterministic CPU convergence/debug run
(about one minute on the validated host).
`configs/toy_mps.yaml` selects MPS automatically when the installed PyTorch and
host expose it, otherwise `auto` falls back to CPU.
