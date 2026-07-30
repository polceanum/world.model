# Project Orpheus agent guide

Before changing code, read these files in order:

1. `PROJECT_SPEC.md` in full;
2. `project/STATUS.md`;
3. `project/TASKS.md`;
4. `project/DESIGN_DECISIONS.md`;
5. `project/CHANGELOG.md`.

`PROJECT_SPEC.md` is the authoritative architectural contract. Preserve the
public tensor contracts, the persistent `WorldBelief`, timestamped asynchronous
observations, explicit uncertainty, persistent identity/lifecycle, hybrid
dynamics, and the separation between fast state correction and slow parameter
identification.

Working rules:

- Use the existing `orpheus` conda environment for every Python command.
- Do not reinstall or replace PyTorch. Model execution remains local.
- Keep `train.py`, `evaluate.py`, and `demo.py` as the simple public workflow.
- Add or update focused tests with every behavioural change.
- Keep project memory and relevant architecture documents synchronized.
- Record exact commands, observed results, limitations, and unfinished work.
- Do not replace the architecture with a clip predictor or opaque dynamics MLP.
- The oracle/state observation module is debug-only. Never use simulator state
  as an input to a claimed RGB result.
- Do not add hosted services, API keys, databases, Hydra, Lightning, external
  tracking, or heavy infrastructure without a documented architectural need.
- Preserve user work. If it conflicts with the specification, record the
  conflict in `project/DESIGN_DECISIONS.md` and make the smallest correction.
- Preserve predictive abstractions as first-class executable state. Prefer the
  simplest representation that predicts within calibrated uncertainty (for
  example, a point trajectory for a freely moving ball) and refine it only
  when interactions require richer geometry or dynamics.
- Foundation models, transformers, and generative decoders may extract,
  complete, or propose abstractions. They must not replace `WorldBelief` with
  an opaque sensor latent or make generated pixels the primary evidence of
  physical correctness.
- Preserve at least one familiar, invariant-tested reference-physics regime.
  Unusual and compound dynamics are valid learnable scenarios, but label and
  evaluate them separately so simulator quirks cannot masquerade as model
  accuracy or failure.
- Treat a mature physics engine as an optional independent RGB dataset backend,
  never as runtime privileged input or a replacement for learned dynamics.
  It must emit the canonical episode/observation contracts and record its
  engine version, solver settings, units, timestep, scenario, and seed.
- Prefer one shared checkpoint across the declared scenario mixture. Treat
  scenario-specific checkpoints as diagnostic ablations unless an explicit
  observation-derived regime router is part of the runtime.
- Compare checkpoints on the same explicit seed manifest, ordered scenario
  mixture, object counts, sequence length, horizons, and metric semantics.
  Never infer an accuracy gain from evaluator defaults that select different
  episodes.
- Treat short runs as wiring or throughput checks. Do not discard a shared
  model or claim convergence until its declared balanced minimum training
  coverage has completed and broad validation has reached a predeclared
  plateau.
- For the sustained campaign, plateau requires four exact consecutive
  512-step validation candidates with no acceptance and less than 1% raw
  primary-score improvement over the safe pre-window incumbent. Missing or
  contradictory evidence means continue a complete block; reaching a hard
  budget without this evidence is `limit_hit`, not convergence.
- Select long-run checkpoints by pooled physical metrics with distance-gated
  detection/identity, lifecycle, event, velocity, every-horizon, and
  calibration guardrails. Apply non-regression checks against both the moving
  incumbent and a fixed pre-campaign reference, and retain numbered validation
  snapshots with verifiable weight and protocol provenance.
