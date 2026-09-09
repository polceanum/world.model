# General-capability scale-up plan

## Outcome

Scale the compact CPU-first world model along three useful dimensions—factor
composition, open-loop horizon, and visible object count—without changing its
public belief/action/rollout interfaces, training planning outcomes directly,
or growing artifacts beyond the managed 250 MiB budget.

The calibrated structured checkpoint remains the incumbent. Scaling means a
larger verified behavioral envelope, not merely more parameters, episodes, or
optimizer updates.

## Current evidence and first bottleneck

- N=1--6 physical behavior passes calibrated camera motion, known actions,
  partial visibility, and the compositional physical holdout.
- Sensor noise and variable physical parameters miss only opposite edges of
  the declared uncertainty-coverage interval.
- Factor-conditioned planning is exact for four families. Sensor-noise K=32
  has one immature target history; compositional K=32 fails mainly through
  unstable target histories and fine impulse-magnitude ordering.
- State-only N=8/12/16 execution is already finite, batch-independent, and
  comfortably inside the N=16 latency gate. Full RGB-D evidence stops at N=6.
- Evaluation throughput, rather than learned-model size, is the current
  engineering bottleneck.

The next run is therefore diagnostic, not a larger training campaign.

## Stage 1 — Locate and repair the existing failure owner

1. Run the remaining same-task ablations on the sensor and compositional
   planning failures: clean observation, truth association, truth physical
   parameters, truth anchor state/history, and truth-state rollout. Open the
   private oracle only after each public candidate decision, as today.
2. Record which intervention changes target resolution, winner identity,
   regret, and goal success. Require a repeatable owner on at least two seed
   blocks before changing model capacity.
3. Calibrate covariance independently of state means using only observable
   quality signals such as valid-depth fraction, innovation magnitude, track
   age, and post-event sample count. Fit on public development calibration
   rows and validate on disjoint rows. Prefer a monotone, bounded calibrator
   over a neural head.
4. Repair immature planning anchors by making task eligibility and history
   maturity explicit. Do not fabricate velocity evidence or silently remove a
   failing cardinality.

Exit when both boundary calibration misses have margin inside the accepted
coverage band, the compositional K=32 error has an evidence-backed owner, and
all existing physical/planning regressions remain within 2%.

## Stage 2 — Make evaluation cheaper before making it larger

1. Reuse public observation encodings and paired incumbent/candidate episode
   materialization within one process. Never cache private truth in runtime
   inputs or retain RGB-D after reduction.
2. Batch only semantically identical public inference work. Keep B1 and dense
   dynamics as numerical oracles and require exact identities/events plus
   numerical agreement within the existing `1e-6` parity tolerance before
   enabling a faster path by default.
3. Benchmark three repetitions of the complete 22-cell physical pass and the
   12-slice planning pass. Target at least a 40% wall-time reduction with no
   more than 10% latency regression in any public online path.

This stage is infrastructure-only: it cannot promote a checkpoint.

## Stage 3 — Add a real long-horizon ladder

1. Add deterministic state-first episodes long enough for 4- and 8-second
   open-loop evaluation from a mature public anchor. Extend the existing
   horizon vector to `0.05/0.10/0.25/0.50/1/2/4/8 s`.
2. Start with fixed membership and no unobserved future action. Then add known
   future actions passed causally into rollout, single contact, and repeated
   bounded contacts as separate strata.
3. Report position/velocity error, uncertainty coverage, collision timing,
   energy drift, and planning winner/regret at every horizon. Measure recursive
   compounding rather than interpolating a two-second result.
4. Add at most three 4/8-second vector forecast examples to the dashboard,
   including uncertainty envelopes and horizon labels. Keep each example below
   96 KiB and all qualitative evidence for a run below 1 MiB.

Run an incumbent-only pilot first. Freeze absolute gates from task geometry
and safety tolerance before any training sees these manifests; never derive a
passing threshold from the incumbent's observed error.

## Stage 4 — Promote perceptual capacity from N=6 to N=8

1. Keep proposal capacity configurable as `max_objects + birth_proposals` and
   instantiate N=8 without changing legacy N=1/2/6 checkpoint loading.
2. Qualify packed active-pair interaction against the dense oracle at N<=6,
   then use it for N=8. Retain N=12/16 as state-only pressure tests until N=8
   perception, association, lifecycle, and planning pass.
3. Create controlled N=7/8 RGB-D strata for separated objects, contact, dynamic
   membership, short occlusion/recovery, and the strongest existing two-factor
   compositions. Increase image resolution only if a pixel-support ablation
   proves 64x64 is the owner.
4. Preserve the current CPU ceilings where meaningful: learned weights <=1
   MiB, RSS <=2.5 GiB, state-only N=16 six-horizon rollout <=0.10 s, and no more
   than 10% incumbent latency regression. Report N=8 perception latency
   separately rather than hiding it in aggregate wall time.

N=8 is a new qualification boundary. N=12/16 perceptual support must not be
claimed from state-only probes.

## Stage 5 — Broaden composition, then add the smallest owned residual

Introduce pairwise compositions before a single broad mixture:

1. sensor noise + camera motion;
2. physical variation + known actions;
3. contact + partial visibility/recovery;
4. the three-way combinations that dominate downstream planning regret.

Use the Stage-1 ablations to choose at most one component change per attempt:

- clean-observation ownership: improve observable proposal geometry;
- association ownership: add bounded appearance/recovery memory;
- parameter ownership: improve observable parameter estimation and its
  uncertainty;
- truth-state contact ownership: add or widen only the antisymmetric relation
  residual;
- no state/dynamics ownership: repair planning task resolution or cost
  sensitivity, not the world model.

Every learned change must pass the 64-example/256-update screen before a
2,048-update run. Extend by 1,024 updates only after a complete validation
improves; stop at 8,192 updates, 24 hours, or the existing four-validation
plateau rule.

## Promotion and visualization

Retain the existing absolute physical/planning gates. Promotion additionally
requires a >=3% paired macro-score improvement with positive 95% bootstrap
bound, >=1% improvement in the worst supported family, no accepted-like slice
regression above 2%, and online latency no more than 10% worse than the
incumbent.

Add two compact dashboard views as the envelope expands:

- a capability-frontier matrix with object count on X, forecast horizon on Y,
  and cells marked measured/pass/fail/unmeasured;
- an ablation attribution panel showing which intervention recovers each
  failed metric and whether a model change is justified.

Keep summaries and portable HTML, the incumbent and two recent promoted
checkpoints, and only the newest bounded failed-run diagnostics. Retain no
generated episodes or frame media. Ordinary completed runs remain below 5 MiB
where practical, and optional evidence stops being retained before protected
artifacts could exceed the 250 MiB rolling cap.

## Immediate execution order

1. Implement the paired sensor/compositional ablation runner and dashboard
   attribution panel.
2. Measure and optimize evaluation throughput with exact parity tests.
3. Apply the smallest observable covariance/history repair supported by the
   ablations; rerun all six factor families and planning gates.
4. Run the incumbent-only 4/8-second pilot and freeze the long-horizon gates.
5. Open N=8 perceptual development only after the N<=6 envelope is clean.
