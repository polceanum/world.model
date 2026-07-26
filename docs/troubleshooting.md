# Troubleshooting

- If `mps` is requested but unavailable, run `auto` or `cpu`; do not replace the
  custom PyTorch installation.
- Unknown YAML keys are errors by design.
- RGB packets require known camera calibration in Milestone 1.
- A checkpoint is trusted local pickle data; do not load untrusted files.
- A report marked RGB-only must have debug oracle disabled.

