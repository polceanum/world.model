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

`configs/toy_smoke.yaml` is the minimum executable smoke.
`configs/tiny_overfit.yaml` is the deterministic CPU convergence/debug run
(about one minute on the validated host).
`configs/toy_mps.yaml` selects MPS automatically when the installed PyTorch and
host expose it, otherwise `auto` falls back to CPU.
