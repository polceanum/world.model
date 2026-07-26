# Coding guidelines

- Run Python only through `conda run -n orpheus`.
- Use float32 physical tensors, int64 indices, bool masks, and explicit devices.
- Type and document public interfaces, shapes, units, and frames.
- Keep modality-specific code behind observation modules.
- Preserve input beliefs during rollout; test mutation and finite values.
- Prefer small composable PyTorch modules over framework infrastructure.
- Use `grid_sample` rather than compiled ROI operators and SciPy Hungarian on
  CPU for the small association problem.
- Reject unknown configuration and nonmonotonic timestamps clearly.
- Add focused tests, then synchronize status/tasks/design docs.

