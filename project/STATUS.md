# Project status

## Active generalization program — 2026-08-26

The canonical checkout is synchronized with GitHub `main` at
`c16acc99ef13757fc8f88528bfd0d66db4a2f4fd`. Broad heterogeneous training
remains paused. The accepted base is the specification-1.51 differentiable
one-sphere RGB-to-state-to-rollout unit, not any older campaign checkpoint.

The qualified unit achieved, on its single previously untouched final set:

- RGB world-position RMSE `0.00764440699 m`;
- image-centre RMSE `0.00522461 px`;
- apparent-radius relative RMSE `0.00219904` (`0.2199%`);
- `0.1 s` rollout RMSE `0.00799061917 m`; and
- finite measurement validity `1.0`.

Its runtime path is ordinary autograd from RGB through soft foreground
evidence, four finite-difference Gauss--Newton inverse-rendering stages,
calibrated backprojection, temporal state estimation, and analytic tensor
kinematics. Simulator state remains supervision/evaluation only. The accepted
implementation is commit `7344e67d`; promotion bookkeeping is `c16acc99`.
The original qualified research source is now recoverable through GitHub tag
`archive/minimal-differentiable-toy-v2-f8d66da`, which peels to
`f8d66da17983aa0269649fff69cc13cec5ad1311`.

### Cleanup boundary

The 254-run historical workspace was archived and removed from active use.
All non-checkpoint metadata, the exact compatibility fixture, the qualified
report/legacy artifact, and a complete source bundle occupy approximately
`33 MiB` under ignored local path
`.archive/20260826-pre-generalization/`. The superseded run tree and 660
duplicate checkpoints occupied about `7.6 GiB`; they were intentionally
deleted after archive verification. `runs/` is now empty. Repository caches,
generated demos, and selected stale temporary clones/caches were also removed.

Four unreferenced rejected campaign profiles and three one-off campaign tools
have been removed from the active tree. Reusable typed contracts, analytic
physics, checkpoint integrity, evaluation metrics, and smoke fixtures remain.
Historical status, task, training, and accuracy records remain in Git commit
`c16acc99` and the ignored local pre-generalization archive; active tracked
memory is intentionally concise.

### Current rung: temporal free-motion and long-horizon consistency

The next isolated rung keeps the already identifiable one-sphere, fixed
camera/radius/gravity/drag, and contact-free world. It changes only temporal
estimation and horizon:

1. fit anchor position and velocity from a bounded RGB-derived history with a
   differentiable closed-form weighted least-squares solution to the exact
   linear-drag equations;
2. roll forward only through `AnalyticKinematics`; and
3. evaluate `0.1/0.25/0.5/1.0/2.0 s` horizons, semigroup consistency,
   gradient reachability, calibration diagnostics, and separated perception/
   state-rollout throughput.

The solver and fail-fast ladder are implemented and frozen on
`agent/general-world-model-rung-1`. The exact differentiable free-motion basis
uses a cancellation-stable float32 path, per-row normalized weights, and
finite zero outputs for invalid rows. The temporal estimator has only four
trainable mask-head scalars; there is no learned transition. Architecture
attempt 2 of the declared maximum 2 and resolved-config SHA-256
`cb40cf08178453f1b0045afd293e82237b31e19b3f38b3136cce95830bd25cd8`
are immutable.

Focused temporal implementation validation passes `61` tests, and the broader
config/checkpoint/temporal compatibility selection passes `245` tests. Static
independent review passes the frozen access, provenance, baseline, gradient,
and artifact contracts. No development artifact has yet been generated from a
clean committed source. Selector seeds `32000000--32000015`, confirmation
seeds `33000000--33000015`, and final seeds `34000000--34000031` remain
unopened. The published specification-1.51 final set will not be reused.

The next action is to freeze the current bytes in a clean commit, run the
development-only 32-update/16-audit workflow, and independently inspect and
hash its report and checkpoint. Protected qualification may run once only if
that development evidence passes without any source change.

After this rung passes, scaling remains ordered: public `OnlineWorldModel`
integration; moving camera; identifiable drag; RGB-D metric scale; two
non-contact objects; variable set size; identity/occlusion; analytic contact;
observable material parameters; known actions and counterfactual planning;
then richer modalities/geometry. Model capacity grows only after a smaller
structured rung demonstrably plateaus.

## Validation state

The current cleanup/temporal source passes the complete repository gate:
`1075 passed, 16` expected inactive-MPS skips in `425.70 s`. Ruff lint passes;
all `224` Python files are already formatted; compileall over production,
tests, scripts, and entry points passes; the explicit specification-version
contract passes; and `git diff --check` is clean. This qualifies source
integrity only. A clean committed development artifact, protected temporal
qualification, and any general-world-model convergence claim remain pending.

No general multi-object, contact, long-horizon, multimodal, or planning
convergence claim exists yet.
