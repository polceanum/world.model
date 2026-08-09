# Project status

**Date:** 2026-08-09
**Specification:** `PROJECT_SPEC.md` 1.18
**Current state:** runnable RGB-only Milestone 1 vertical slice with accurate
synthetic-disc localization, ROI-local online correction, explicit
selection/confirmation/test manifests, horizon-balanced recursive training,
stable forecast-history visualisation, axis-resolved diagnostics, an
invariant-tested familiar reference-pair regime, and one balanced eight-regime
shared-model profile, plus a quality-aware persistent-ID multi-frame
point/scale depth observer; the first sustained campaign is preserved as a
superseded legacy-objective control after a convergence-integrity audit; the
corrected v2 campaign is also preserved and stopped after a second audit proved
that unsupported rows consumed causal updates and coverage collapsed; the
first v3 qualification is preserved but stopped after its first causal
validation exposed lifecycle/identity collapse and perception-gradient
starvation; the repaired runtime passes complete regression, host MPS device,
and clean hybrid MPS/CPU end-to-end wiring checks, but no repaired
qualification result, convergence result, or broad promotion exists yet; a
second repaired eight-scenario qualification was proven never to have trained
and its launchd restart storm has been stopped; a third protocol-v10
qualification was intentionally interrupted before its first metric after an
audit found that configured cadence three actually executed cadence four;
cadence semantics, progress observability, delayed worker startup, and
post-step/checkpoint finite-state integrity are now repaired under rollout
protocol 11; clean CPU causal and host-MPS optimizer-progress smokes completed
without nonfinite/zero-gradient/restart failure, but neither one-step smoke
earned a deployment promotion; the first protocol-v11 qualification was
truthfully alive and finite but intentionally stopped at zero updates after a
launch-QoS audit found a roughly fourfold `Background` throttling regression;
float timestamp integration-grid drift and duplicate causal propagation are
repaired under rollout protocol 12; a reduced production-model smoke completed
one finite, supported causal optimizer update and terminal validation; the
full protocol-v12 campaign reached 11,776 logged updates but was killed by
macOS memory pressure before its 16,384-step target, and its deployment support
gate repeatedly reset otherwise finite causal candidates to step zero, so it
is incomplete rather than converged; protocol 13 separates deployment
selection from catastrophic mutable-state viability and bounds long-run
worker/allocator memory; its repaired campaign accumulated 6,096 supported
causal updates with stable optimizer/memory/numerics but no strict promotion;
exact modular qualification isolated shared-backbone drift and found a strong
every-horizon fast-ROI/state/dynamics diagnostic candidate that still fails
identity, z, coverage, and scenario guardrails; a new
`state_dynamics_fast_roi` scope now freezes shared/global RGB perception for
the next clean long run; strict convergence and acceptance remain unproven,
and collision, identity, z-axis, coverage, and scenario-wide non-regression
remain open; the protocol-15 staged campaign is preserved as a finite failed
plateau after exact ablations exposed fast-measurement gradient leakage into
the frozen-perception state/dynamics phase; protocol 16 verified that repair
but was intentionally stopped at update 552 after a further audit found that
rollout likelihood duplicated the deterministic forecast-mean gradient; both
objective defects are repaired, but no corrected convergence result exists
yet; protocol 17 has now completed fixed validation through step 1,536 without
optimizer/support collapse; its latest candidate recovered current state,
velocity, tracking, calibration, x/y, and four joint horizons relative to step
1,024, but still failed the fixed reference on medium/long horizons and broad
scenario balance, so it remains rejected; an exact step-1,536 resume is active
under a corrected one-shot launcher toward the unchanged 8,192-step minimum
without promotion; fixed step 2,048 improved selected slices but regressed
camera/depth and glancing-contact forecasts; exact module ablations isolated
late updater/dynamics drift under random two-scenario updates, deterministic
eight-scenario optimizer batches are now implemented, and a real batch-eight
smoke completed one finite supported update and terminal validation at 1.20 GB
maximum RSS; the replacement campaign is pending immutable launch

## 2026-08-09 — scenario-balanced optimization repair

Protocol 17 remains numerically healthy, but its complete fixed step-2,048
candidate exposed continued cross-regime wobble rather than broad convergence.
The weighted score was `0.3594687`, versus `0.3385591` at step 1,536 and the
`0.3296688` fixed reference. Current z RMSE regressed to `0.288938 m`; the
camera-parallax slice reached about `0.2528 m` current error and glancing
impacts `0.5116 m`, both with large depth regressions, while heavy/light
impacts improved strongly. The candidate was rejected with 120 fixed-reference
guardrail failures.

Exact fixed-manifest scale and module ablations showed that reducing the
learned-corrector scale only traded metrics. Restoring the step-512 updater
into the step-2,048 model repaired camera/glancing depth but regressed other
scenarios, while the inverse composition failed badly. This localizes the
failure to coupled updater/dynamics drift rather than perception, finite state,
or one bad gate. The batch-two loader gave each update only a random pair from
the eight scenario families, so no update represented the declared shared
objective.

Specification 1.18 adds deterministic manifest-bound scenario-balanced
optimizer sampling. `training.scenario_balanced_batches=true` requires complete
equal support; each scenario pool shuffles independently and exact continuation
is reconstructed from the absolute draw. The new
`configs/sustained_accuracy_balanced_mps.yaml` uses one example from every
scenario per update, freezes the qualified RGB stack, and trains updater,
identifier, and dynamics from protocol-17 step 512 for 4,096 updates / 32,768
episode draws.

A real CPU smoke initialized from `validation_step_000512.pt`, ingested all
eight scenario names in one update, and reported loss `3.636137`, raw/applied
gradient norms `1.395285 / 1.045683`, all 13 causal objective terms, 188
trajectory-support rows, zero perception gradient, no skipped update, and
maximum RSS `1,201,246,208` bytes. Its terminal eight-scenario RGB-only
validation and finite checkpoint completed in `335.49 s`. This is integrity
and throughput evidence only, not an accuracy promotion. Artifact:
`/private/tmp/orpheus-balanced-smoke/20260809-211033-20260809-balanced-batch8-smoke/`.

## 2026-08-09 — exact-resume launcher repair and monitored continuation

The first two post-audit launch attempts did not alter the authoritative
step-1,536 checkpoint. The first used the run-local resolved configuration
with a relative `project.output_dir`; together with the old launch helper's
mandatory `--run-name`, that created a new nested run instead of an exact
continuation. Its finite steps 1,544--1,568 were quarantined intact at
`/private/tmp/20260809-171508-protocol17-accidental-nested-resume` and are not
protocol-17 evidence. The second correctly resolved the original config but
still passed `--run-name`; `train.py` rejected the already-existing target
directory before loading or updating the model. The supervisor truthfully
recorded that external trainer exit, and its state was archived at
`convergence/convergence_supervisor_state_failed_resume1536b.json` before the
explicitly acknowledged retry. `checkpoints/last.pt` remained the durable
step-1,536 source throughout.

`scripts/launch_training_once.py` now makes `--run-name` optional and omits it
from the child command for an exact `--resume`; new runs still require an
explicit name. A regression asserts that the resulting launch payload uses
the source run's `checkpoints/last.pt` without injecting `--run-name`.
Verification was:

```bash
PYTHONPATH=. conda run -n orpheus pytest -q \
  tests/unit/test_launch_training_once.py
# 3 passed in 0.67s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus \
  ruff check scripts/launch_training_once.py \
  tests/unit/test_launch_training_once.py
# All checks passed!

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus \
  ruff format --check scripts/launch_training_once.py \
  tests/unit/test_launch_training_once.py
# 2 files already formatted

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus \
  pytest -q -p no:cacheprovider \
  tests/unit/test_convergence_supervisor.py \
  tests/unit/test_launch_training_once.py \
  tests/unit/test_train_entrypoint.py
# 32 passed in 6.07s
```

The corrected one-shot trainer
`com.polceanum.orpheus.protocol17-resume1536c-20260809-172500` is active with
trainer PID `27142`; `training_state.json` records an in-place resume from the
authoritative `checkpoints/last.pt`. Its first authoritative resumed metric is
step 1,544/data draw 1,544: all 13 causal objective terms had support, the
optimizer update applied, frozen-perception gradient was exactly zero, and
raw/applied gradient norm was finite at `1.033924`. The corrected exact-source
supervisor
`com.polceanum.orpheus.protocol17-convergence-resume1536c-20260809-172500`
is active with PID `27404`, has acquired the run lock, and is waiting for the
unchanged step-8,192 segment. Both corrected stderr logs are empty. No
convergence or promotion is claimed.

The append-only `metrics.jsonl` contains two step-1,544 training rows. The
original process had logged that update after its step-1,536 checkpoint but
was stopped before a newer checkpoint; exact resume therefore restored step
1,536 and deterministically replayed draw 1,544. The two records are identical
in every model/data/loss/gradient field and differ only in elapsed time,
finite-check timing, and process RSS. This is a telemetry-generation issue,
not two optimizer updates in the resumed state. Convergence is unaffected:
`world_model.training.convergence` reads tensor-verified numbered validation
checkpoints and the terminal summary, never raw training-row counts. Until a
post-campaign logger repair is committed, audits must canonicalize repeated
`(split, step)` rows and retain both originals as restart evidence.

## 2026-08-09 — deterministic live training-dynamics audit

`scripts/audit_training_dynamics.py` now turns the sustained-run health checks
into a repeatable read-only command. It canonicalizes append-only replay rows
by `(split, step)` while retaining and reporting the originals, requires
replayed model/data metrics to agree apart from process timing/RSS fields, and
fails on nonfinite metrics, missing optimizer updates, absent causal/objective
support, nonpositive gradients, state/dynamics perception-gradient leakage,
or a broken `training_data_draw_step = step + skipped_draws` invariant. It
also reports loss/gradient/support distributions, clipping, RSS, scenario
draw counts, and fixed-validation pooled, identity, event, uncertainty,
per-axis, and every-horizon evidence. It does not use training loss as a
convergence decision.

The live protocol-17 audit through unique logged step 1,592 passed with eight
canonical blocks from step 1,536, eight applied optimizer updates, no skipped
draw, trajectory support of 27--73 rows, raw gradient norm
`0.274486--1.327438`, loss `0.734846--4.035967`, three clipped blocks, bounded
RSS at or below the earlier `983797760`-byte high-water mark, and no failure.
The two step-1,544 rows are identical in all model/data metrics and are
reported as one canonical block plus one preserved replay record. No new
fixed validation exists beyond the rejected step-1,536 candidate, so this is
health evidence only, not accuracy or convergence evidence.

The audit now also summarizes live lifecycle coverage/precision, identity
churn, collision signal, position calibration, correction benefit, drag and
restitution observability, axis-local rollout objectives, and horizon-resolved
RMSE/coverage. Through unique step 1,608, all ten canonical post-1,536 blocks
still pass. Distance-gated identity-switch rate was zero in every logged
block, median position coverage90 was `0.972222`, median current correction
improvement was `+0.020439 m`, drag/restitution observability remained
represented, and every axis and horizon had finite supported diagnostics.
Future correction improvement was approximately neutral at median
`-0.000102 m`; this is monitored but is not a fixed-validation regression.

Step 1,624 produced the largest late-phase single-batch identity value: three
distance-gated switches over 16 associations. Exact seed/window inspection
showed this was a deliberately hard recovery batch, not malformed truth: it
used two injected belief perturbations, one pair collision, five ground
collisions, fully visible valid targets, and no hidden external actuation.
Replay-aware pooled accounting through step 1,640 gives 11 switches over 521
associations (`0.021113`) in perturbed blocks versus 2 over 183 (`0.010929`)
in clean blocks; the latter two occurred earlier, while all post-1,536 clean
blocks have zero switches. Overall logged rate is 13/704 (`0.018466`), close
to fixed step-1,536 validation's `0.016119`. The monitor now reports these
strata explicitly. This is a recovery/contact difficulty signal to recheck at
step 2,048, not evidence of an unperturbed lifecycle collapse or a reason to
interrupt the protocol.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus \
  pytest -q -p no:cacheprovider \
  tests/unit/test_audit_training_dynamics.py
# 4 passed in 0.06s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus \
  ruff check scripts/audit_training_dynamics.py \
  tests/unit/test_audit_training_dynamics.py
# All checks passed!

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus \
  ruff format --check scripts/audit_training_dynamics.py \
  tests/unit/test_audit_training_dynamics.py
# 2 files already formatted

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python \
  scripts/audit_training_dynamics.py \
  --run runs/20260809-091718-v12-protocol17-rollout-variance-only \
  --after-step 1536
# status: pass; last_training_step: 1592; failures: []
```

## 2026-08-09 — protocol-17 step-1,536 recovery and exact continuation

The complete fixed 32-episode step-1,536 validation at
`runs/20260809-091718-v12-protocol17-rollout-variance-only/` finished in
`908.8502 s` under unchanged protocol hash
`e31bf1cde4e4adf8603190b3258e086d6f749ad1d5689427d60e367f9fbb53a0`.
The candidate was not promoted. Its weighted score was `0.3385591`, versus
`0.3296688` for the fixed reference and `0.3413697` at step 1,024. Current
position RMSE improved to `0.2482949 m`, velocity to `1.0326004 m/s`, target
coverage to `0.3890`, precision to `0.3718929`, identity-switch rate to
`0.0161186`, and collision F1 remained above reference at `0.2311321`.
Position coverage moved closer to nominal at `0.9134915`.

Current x/y/z RMSE was `0.299827 / 0.205018 / 0.230265 m`. Joint horizon RMSE
was `[0.263368, 0.286204, 0.325554, 0.360225, 0.380415] m`. Relative to step
1,024, current position, velocity, coverage, precision, identity, calibration,
all current axes, x at every horizon, y at every horizon, and the first four
joint horizons improved. The 1.0-second joint horizon worsened slightly
`0.375831 -> 0.380415 m`; z improved through 0.25 seconds but regressed at
0.5--1.0 seconds. Relative to the fixed reference, the 0.1/0.25-second joint
horizons improved, while 0.5/0.75/1.0 seconds remained worse.

The candidate failed 122 reference guardrails, down from 134 at step 1,024 but
still broad: 32 x, 16 y, 15 z, 28 joint-position, 18 coverage, 4 velocity, 3
precision, 3 identity, 2 collision, and 1 calibration failures. Scenario
failures were baseline 2, camera parallax 20, damped contacts 11, elastic pairs
16, glancing impacts 21, heavy/light impacts 25, impulse perturbation 11, and
reference pairs 9. Mutable and training-support failures stayed zero.
Reference pairs and baseline improved strongly; the remaining deployment
failures are concentrated in glancing impacts, camera parallax, elastic
long-horizon motion, heavy/light x motion, and some scenario-local coverage
and identity tradeoffs.

All 64 logged state/dynamics updates from 1,032 through 1,536 applied an
optimizer update. Frozen-perception gradient stayed exactly zero, RSS
high-water stayed `983797760` bytes, and one isolated `1.8559e-05` lifecycle
gradient recovered on the following update. Checkpoint deltas were directional
rather than random: update cosine similarity across the 512--1,024 and
1,024--1,280 intervals was `0.968` for interactions, `0.928` for modal
dynamics, `0.939` for uncertainty, and `0.949` for the learned corrector.
Optimizer moments in the durable step-1,408 checkpoint were finite and had the
expected 512-step frozen-fast and 896-step state/dynamics counters.

A read-only four-batch gradient attribution exposed a real multi-task
diagnostic: on the shared edge trunk, event-versus-z trajectory cosine was
negative on all four audited collision batches (`-0.631`, `-0.511`, `-0.653`,
`-0.416`), while x/y/velocity conflicts varied by scenario. This is a
plausible source of event/axis tradeoffs, but not yet evidence for an
architecture change: step 1,536 broadly recovered from step 1,024, and the
declared sustained plan says not to judge causal convergence before adequate
duration. Event/trunk decoupling therefore remains a measured follow-up only
if later comparable validations regress again or establish a failed plateau.

The original trainer and supervisor were unloaded after preserving the
step-1,536 checkpoint during the audit. The operational incident and corrected
exact continuation are recorded in the section above. The active supervisor
retains the 8,192 minimum, 4,096 extensions, final-1,024/1% four-validation
plateau rule, and 24,576 hard limit. No convergence or promotion is claimed.

## 2026-08-09 — rollout uncertainty-gradient repair

Protocol 16 at
`runs/20260809-065710-v11-protocol16-perception-local-objectives/` was
intentionally stopped at logged causal update 552, after its step-512 broad
validation and the first late-phase blocks had provided enough evidence to
audit the corrected objective. The trainer and convergence-supervisor
LaunchAgents were unloaded and their PIDs are absent. The only stderr message
is Python's expected resource-tracker warning from externally terminating the
worker pool; no nonfinite loss, optimizer collapse, support collapse, or
automatic restart occurred.

The fixed 32-episode step-512 validation exactly reproduced protocol 15's
early fast-ROI phase, as expected: score `0.3214190` versus reference
`0.3296688`, current position RMSE `0.2626464` versus `0.2841220 m`, velocity
RMSE `1.087324` versus `1.055478 m/s`, coverage `0.36825` versus `0.382`,
precision `0.350464` versus `0.361229`, collision F1 `0.204651` versus
`0.181818`, and identity-switch rate `0.017184` versus `0.013548`. Axis RMSE
was `0.278836 / 0.215060 / 0.288009 m`; horizon RMSE was
`[0.274357, 0.278067, 0.308430, 0.335595, 0.355628] m`. The candidate failed
122 unchanged reference guardrails and was not promoted. Forecast coverage
declined from `0.49625` at 0.1 seconds to `0.405` at 1.0 second. Mutable
support, gradients, and finite-state gates remained valid.

Static objective tracing then found that rollout Gaussian NLL used the same
live forecast error as the deterministic per-axis/horizon point loss. It
therefore added a second inverse-variance-weighted gradient to the trajectory
mean and could teach a deterministic mean even after a causally unseen
external actuation. This contradicted the already-enforced state-uncertainty
contract and made low-variance examples capable of overwhelming the declared
point objective. Rollout NLL now detaches the mean error: realised outcomes
calibrate forecast variance, while deterministic rollout means receive their
sole supervised gradient from the identifiable point objective. A regression
proves hidden-actuation NLL leaves the belief/mean gradient absent while still
widening an under-dispersed variance. This objective change advances the
specification to 1.17 and requires a fresh weights-only protocol-17 run from
the same accepted reference; protocol 16 must not be resumed.

Verification on the specification-1.17 repair:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider \
  tests/unit/test_training_objective_regressions.py \
  tests/unit/test_training_schedule.py \
  tests/integration/test_checkpoint_roundtrip.py
# 129 passed in 16.17s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider -m "not device"
# 620 passed, 5 skipped, 1 deselected in 167.44s

# Host-MPS device marker and the five otherwise sandbox-skipped MPS tests:
# 1 passed, 625 deselected in 5.01s
# 5 passed in 8.23s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff check .
# All checks passed
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus \
  ruff format --check .
# 188 files already formatted
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-protocol17-pycache PYTHONPATH=. \
  conda run -n orpheus python -m compileall -q \
  train.py evaluate.py demo.py scripts world_model tests
# exit 0; git diff --check also passed
```

The fresh corrected campaign is active at
`runs/20260809-091718-v12-protocol17-rollout-variance-only/` under one-shot
Standard/default LaunchAgent
`com.polceanum.orpheus.protocol17-rollout-variance-20260809-091718`. It starts
weights-only from the unchanged accepted step-zero reference, uses the fixed
32-episode selector, and retains the 512-update `fast_roi` then
`state_dynamics` schedule. Immutable metadata records clean pushed commit
`6dba48eaa39a4df926dcdca085864ceddb95cb50`, PyTorch `2.10.0`, MPS
built/available, MPS measurement plus CPU closed-loop execution, RGB-only
runtime, and no oracle. Trainer PID `9466` is active with Standard QoS; the
trainer and supervisor stderr files are empty.

Exact-launch-source supervisor PID `9591` runs from detached clean clone
`/private/tmp/orpheus-protocol17-runtime-6dba48e/` under LaunchAgent
`com.polceanum.orpheus.protocol17-convergence-20260809-091718`. Its durable
events confirm the 8,192-step minimum, 4,096-step extensions, final-1,024
window, 1% four-validation plateau rule, and 24,576 hard limit. It is waiting
for the initial segment and cannot overlap another trainer. Initialization
completed all 32 episodes in `708.55 s` and exactly reproduced the accepted
score `0.3296688`. Causal metric blocks through update 512 all had real support,
applied an optimizer update, skipped no gradients, kept interaction gradient
zero in the `fast_roi` phase, and passed the post-step finite check. Across the
first four consecutive 64-update quarters, mean heterogeneous-window losses
were `3.3888 / 3.5659 / 3.3182 / 2.5461` and mean raw gradients were
`4.2628 / 3.6357 / 3.9217 / 3.0525`. These non-identically supported losses
are conditioning evidence, not accuracy/convergence metrics; nevertheless the
absence of a rising loss or gradient envelope argues against early optimizer
collapse. Two later fast-perception outliers at updates 368 and 448 reached
raw norms `10.5515` and `14.0315`; both were isolated measurement/variance
events, correctly capped to approximately `1.0`, followed by ordinary norms,
and left model/optimizer state finite. RSS high-water moved from `921 MB` on
the first block to `984 MB` at update 128, then stayed exactly flat through
update 512. Stderr remains empty.

The complete fixed 32-episode step-512 candidate improved raw score
`0.3296688 → 0.3189699` and current position RMSE
`0.2841220 → 0.2504031 m`, with current x/y/z RMSE
`0.268917 / 0.208281 / 0.269087 m`. Joint forecast RMSE improved at every
horizon to `[0.262201, 0.270166, 0.305863, 0.334885, 0.357770] m`.
Nevertheless it was correctly rejected by 113 unchanged guardrails: velocity
worsened `1.055478 → 1.079254 m/s`, target coverage
`0.3820 → 0.3755`, precision `0.361229 → 0.358216`, and identity-switch rate
`0.013548 → 0.018421`. Collision F1 improved slightly
`0.181818 → 0.189376`, and 90% position coverage moved closer to nominal
`0.941595 → 0.931655`. Forecast coverage remained below reference at all five
horizons. Long-horizon x regressed at 0.75 and 1.0 seconds and y regressed at
0.5 and 0.75 seconds despite the lower joint RMSE.

The failure breakdown is broad rather than a pooled-metric artifact: 29
guardrails failed in `heavy_light_impacts`, 22 in `glancing_impacts`, 21 in
`baseline`, 14 in `impulse_perturbation`, 8 in `camera_parallax`, 5 each in
`reference_pairs` and `elastic_pairs`, 2 in `damped_contacts`, and 7 pooled.
By metric family the 113 failures comprise 26 x, 21 y, 12 z, and 19 joint
position failures; 20 coverage, 4 precision, 4 velocity, 3 identity, 3
collision, and 1 calibration failure. The same phase under protocol 16 scored
`0.3214190`, so variance-only rollout likelihood materially improved score,
current position, velocity, coverage, precision, four of five horizons, and
failure count relative to the duplicated-gradient objective; it slightly
worsened 1.0-second RMSE, identity, collision F1, and calibration relative to
that rejected candidate. No candidate is promoted.

The first post-transition block at update 520 proves corrected late-phase
routing: `state_dynamics` is active, optimized measurement is absent,
`frozen_fast_measurement=1.678343` remains diagnostic, causal fast support and
perception gradient are exactly zero, trajectory support is 138, interaction
gradient is finite at `0.407258`, and total/applied norm is an unclipped
`0.536599`. The run remains active; no convergence claim exists yet.

All 64 logged `state_dynamics` blocks from update 520 through 1,024 applied an
optimizer update and passed finite-state checks. Fifty-eight had state support,
51 had rollout support, and 48 exposed all 13 causal objective families; six
were legitimate negative-lifecycle-only windows rather than empty draws.
Median trajectory support was 65 and median raw gradient norm was `0.756674`.
Seven event-conditioned interaction gradients used the isolated local cap and
only one block reached the global cap; every following spike recovered, the
perception gradient stayed exactly zero, and RSS high-water stayed exactly
`983797760` bytes. Rollout NLL had median `-0.501280`; one finite damped-contact
outlier reached `17.204458` and did not recur.

The complete step-1,024 validation finished all 32 episodes in `921.14 s` with
the unchanged protocol hash. It was not promoted. Score regressed from the
reference/step-512 values `0.3296688 / 0.3189699` to `0.3413697`. Current
position remained better than reference but worse than step 512 at
`0.2693728 m`; velocity was `1.067599 m/s`. Current x/y/z RMSE was
`0.307633 / 0.225661 / 0.268559 m`. Relative to step 512, all five z horizons
improved to `[0.273445, 0.261571, 0.270725, 0.291681, 0.309352] m`, collision
F1 improved `0.189376 -> 0.240363`, identity churn improved
`0.018421 -> 0.016469`, and every forecast-coverage horizon improved slightly.
However, x horizons regressed to
`[0.337209, 0.375243, 0.454931, 0.507940, 0.529834] m`, y regressed at four of
five horizons, and joint horizons regressed to
`[0.282900, 0.294747, 0.329719, 0.360698, 0.375831] m`.

Reference-guardrail failures increased `113 -> 134`. Baseline failures fell
`21 -> 2`, coverage `20 -> 18`, and identity `3 -> 2`, but x-position failures
rose `26 -> 35`, joint position `19 -> 25`, and failures broadened in reference,
elastic, damped, and camera scenarios. Mutable/training support failures stayed
zero. The trainer resumed normally at update 1,032 with all 13 objective
families and 70 trajectory-support items; exact-source training continues to
the step-1,536 validation to determine whether the axis tradeoff is transient
or worsening. No convergence claim exists.

## 2026-08-09 — staged-campaign plateau and auxiliary-gradient repair

The staged campaign at
`runs/20260808-193216-v10-protocol15-staged-fast-roi-state-dynamics/`
completed 2,976 logged causal updates and broad validation through step 2,560.
It remained finite, supported, single-process, and active until intentionally
stopped; every RGB tensor changed only during the first 512-update `fast_roi`
phase, and only dynamics/updater/identifier tensors changed afterward. There
was no numerical, lifecycle-support, optimizer, or scope-freeze collapse.

It nevertheless reached a failed accuracy plateau. Step 512 improved the raw
score from `0.3296688` to `0.3214190`, current position RMSE from `0.2841220`
to `0.2626464 m`, and four of five forecast horizons, but failed 122 reference
guardrails including velocity, coverage, precision, y, identity, and scenario
slices. The four subsequent exact candidates at steps 1,024–2,560 were all
rejected and none improved the fixed incumbent score by 1%; scores were
`0.3450999`, `0.3328185`, `0.3500511`, and `0.3614501`. This satisfies the
declared failed-plateau rule, so the one-shot LaunchAgent was unloaded rather
than spending the remaining 8,192-step budget unchanged.

Exact 32-episode module ablations at the least-bad late checkpoint, step 1,536,
showed that dynamics alone regressed all five horizons to
`[0.307012, 0.311106, 0.337824, 0.364638, 0.374338] m` and failed 139
guardrails. Updater plus identifier improved 0.1-second RMSE and coverage but
regressed the longer horizons to
`[0.275294, 0.291523, 0.332584, 0.368270, 0.389032] m`, failing 114
guardrails. Reports are in
`runs/20260809-062728-protocol15-step1536-dynamics-only/` and
`runs/20260809-063510-protocol15-step1536-updater-identifier/`.

The audit found an objective-routing bug. During `state_dynamics`, every RGB
parameter was correctly frozen, but the fast ROI output remained
differentiable through its propagated-prior input. Fast measurement loss was
therefore still included and trained dynamics/updater to improve an auxiliary
measurement residual, contrary to the staged isolation contract. The fast
branch now counts as trainable only when a shared fast encoder stage, the
ROI-only projection, or the ROI updater is trainable. Otherwise its scalar and
components are detached `frozen_fast_measurement` diagnostics, excluded from
the optimized total and fast causal-support map. A real closed-loop regression
proves the state/dynamics scope has no measurement term or fast support.
The same pass found checkpoint metadata still declared specification `1.12`;
the package constant now matches authoritative specification `1.16`, with a
regression that reads and compares the contract header.

Focused verification on the repair:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider \
  tests/unit/test_training_schedule.py \
  tests/unit/test_training_objective_regressions.py
# 103 passed in 7.27s
```

Complete verification after the provenance repair:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider -m "not device"
# 620 passed, 5 skipped, 1 deselected in 153.74s

/bin/zsh -lc 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  conda run --no-capture-output -n orpheus pytest -q \
  -p no:cacheprovider -m device'
# 1 passed, 625 deselected in 4.57s (host MPS)

# The five otherwise sandbox-skipped MPS tests were also selected explicitly.
# 5 passed in 7.20s (host MPS)

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus \
  ruff format --check .
# 188 files already formatted

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff check .
# All checks passed

PYTHONPYCACHEPREFIX=/private/tmp/orpheus-protocol16-pycache PYTHONPATH=. \
  conda run -n orpheus python -m compileall -q \
  train.py evaluate.py demo.py scripts world_model tests
# exit 0; git diff --check also passed
```

The raw interaction gradient had one extreme but finite step-1,736 outlier
(`28053.64`) on a glancing-impact batch. The configured interaction-local cap
reduced it before the whole-model cap, optimizer/model state stayed finite,
and it did not recur at that scale; this remains a conditioning diagnostic,
not evidence of numerical collapse. The corrected objective requires a fresh
weights-only qualification. No protocol-15 candidate is promoted.

The corrected replacement is active at
`runs/20260809-065710-v11-protocol16-perception-local-objectives/` under
one-shot Standard/default LaunchAgent
`com.polceanum.orpheus.protocol16-perception-local-20260809-065710`. It
launched from clean pushed commit `310d41922b4489126ca9710b76093c5cf4a2ee04`,
the same accepted step-zero reference, the unchanged 32-episode selector, and
the same 512-update `fast_roi` then `state_dynamics` schedule. Metadata records
PyTorch `2.10.0`, MPS built/available, MPS measurement, CPU closed-loop,
RGB-only runtime, and no oracle. Trainer PID `5760` is active in initialization
validation; its first episode completed in `10.560 s` and stderr is empty. No
protocol-16 trained checkpoint or accuracy result exists yet.

The first protocol-16 late-phase block at step 520 proves the repaired
objective route is active: `loss_measurement` and `measurement_fast` are
absent, `frozen_fast_measurement=1.684341` remains diagnostic,
`perception_gradient_norm_pre_clip=0`, trajectory support is nonzero, and the
total raw gradient is finite at `0.486013`. The same row exposed a metrics-only
follow-up: `causal_fast_support_count` still reported 48 observed frozen slots
even though they could not support an update. Causal support now reports that
count only when the fast measurement has a real nonzero derivative into the
optimized total; raw observed slot counts remain available separately.

A one-shot convergence supervisor is attached under LaunchAgent
`com.polceanum.orpheus.protocol16-convergence-20260809-065710`, with supervisor
PID `7587`. It executes from an isolated clean local clone of exact launch
commit `310d419`; its runtime/source fingerprints match the trainer checkpoint
provenance exactly. It waits for the complete 8,192-step segment, verifies
every selector/tensor link, applies the declared four-consecutive-512-step and
1% plateau rule, and may launch only whole 4,096-step exact-resume extensions
up to the 24,576 hard limit. Durable events are in
`runs/20260809-065710-v11-protocol16-perception-local-objectives/convergence_supervisor.jsonl`;
supervisor stderr is empty. The later support-count reporting fix is not
hot-loaded into this numerical trajectory; its raw observed count must be
interpreted alongside the zero perception gradient and absent differentiable
fast support. This preserves exact training semantics.

## 2026-08-08 — frozen-loss audit and objective-integrity repair

The frozen-backbone campaign at
`runs/20260807-223146-v8-protocol13-frozen-fast-roi/` was intentionally stopped
and its one-shot LaunchAgent unloaded after 4,744 supported causal updates.
It completed initialization and nine post-initialization validations through
step 4,608 with finite optimizer state, no shared/global RGB tensor drift, and
no promotion. It did not collapse numerically, but it oscillated across axes
and scenarios. The step-512 candidate had a better score (`0.3141055` versus
`0.3296688`) and lower pooled RMSE at all five horizons
(`[0.275420, 0.279301, 0.292794, 0.320858, 0.352984]` versus
`[0.293176, 0.294921, 0.315401, 0.339072, 0.360882]`), but it regressed target
coverage (`0.3735` versus `0.3820`), ID switches (`0.020151` versus
`0.013548`), y accuracy, and 128 scenario guardrails. Later checkpoints traded
these properties rather than converging broadly.

The audit found a concrete objective bug. Although metrics truthfully reported
`global_perception_trainable=0`, loss assembly considered every parameter under
the RGB backbone. The ROI-only `fast_projection` therefore made completely
frozen global discovery appear trainable. At the last logged step, global loss
`5.2873979` and fast-ROI loss `0.0503706` were averaged into measurement loss
`2.6688843`. The constant global term contributed no gradient, dominated the
scalar loss, and invalidated any loss-convergence interpretation. The later
audit below distinguishes that diagnostic bug from the fixed fast-branch
coefficient, which still had to remain one half.

Specification 1.14 and ADR-080 now require a real trainable path. Global loss
inclusion checks only the detector, shared stages, and pyramid projections.
Under `state_dynamics_fast_roi`, global loss is retained as
`frozen_global_measurement` diagnostics but excluded from the optimized
measurement term. Predicate and complete causal-batch regressions pass.

Verification on the repaired tree:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider \
  tests/unit/test_training_schedule.py \
  tests/unit/test_training_objective_regressions.py
# 99 passed in 4.90s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider -m "not device"
# 611 passed, 5 skipped, 1 deselected in 147.57s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff check .
# All checks passed

PYTHONPYCACHEPREFIX=/private/tmp/orpheus-pycache PYTHONPATH=. \
  conda run -n orpheus python -m compileall -q \
  train.py evaluate.py demo.py scripts world_model tests
# exit 0
```

The corrected staged campaign is active at
`runs/20260808-193216-v10-protocol15-staged-fast-roi-state-dynamics/` under
one-shot Standard/default LaunchAgent
`com.polceanum.orpheus.protocol15-staged-20260808-193216`. It launched from
clean pushed commit `2fea10aab0b96442ee0ae63a29a88d155e5dc646`, uses the
same accepted step-zero reference and unchanged 32-episode selector, targets
8,192 causal updates, trains `fast_roi` through update 512, then transitions
to `state_dynamics`. Metadata records PyTorch 2.10.0, MPS built/available,
MPS measurement, CPU closed-loop execution, RGB-only runtime, and no oracle.
PID `98671` is active in initialization validation with stderr empty. No
trained candidate or new accuracy result exists yet.

The stopped run's `training_state.json` remains stale at `running` because a
direct SIGTERM cannot execute Python's terminal-state writer; the absent PID,
unloaded LaunchAgent, final step 4,744, and this audit are authoritative. A
fresh weights-only run from the same accepted reference is required; exact
resume would retain the flawed objective.

That replacement ran at
`runs/20260808-161058-v9-protocol14-fast-roi-objective/` under the one-shot
Standard/default LaunchAgent
`com.polceanum.orpheus.protocol14-fast-roi-20260808-161058`. It launched from
clean pushed commit `c13d5d9402d1f6932492ddaffa144f1cdbde80a6`, uses the same
accepted step-zero initialization, 8,192 causal-update target, zero global
adaptation steps, and `state_dynamics_fast_roi` scope. Metadata records
PyTorch 2.10.0, `mps_built=true`, `mps_available=true`, MPS measurement,
CPU closed-loop execution, RGB-only runtime, and no oracle. It was later
stopped at step 720 after the first trained validation exposed the separate
branch-weight bug documented above; it is not active, promoted, or converged.

## 2026-08-08 — fixed branch weights and staged-scope repair

The specification-1.14 campaign remained finite, supported, and resource-
stable, with zero skipped updates, unchanged global/shared RGB tensors, and
empty stderr. Its first trained candidate at step 512 nevertheless regressed
score from `0.3296688` to `0.3749701`, current x RMSE from `0.3300525` to
`0.4224541`, and every forecast horizon. It retained pooled current/horizon
coverage and therefore was not a numerical or lifecycle collapse, but the
accuracy regression was large enough to stop and preserve the run at logged
step 720 rather than spend the remaining budget unchanged.

Gradient evidence identified support-dependent branch reweighting. With both
global and fast measurement objectives, `fast_roi_pretrain_weight=1.0` assigns
each coefficient `0.5`. After frozen global discovery became diagnostic-only,
the single-branch fallback silently assigned fast ROI coefficient `1.0`.
Typical ROI gradients were 6–30 times the local cap, so this changed their
direction relative to state/rollout gradients rather than merely changing the
displayed scalar.

Two exact 32-episode modular qualifications isolate the effect:

- specification-1.14 step-512 fast ROI alone scores `0.3602169`, regressing
  every horizon and reproducing most of the full candidate's damage;
- the earlier fixed-half-weight step-512 fast ROI alone scores `0.3110033`
  versus base `0.3296688`, lowers current position RMSE from `0.2841220` to
  `0.2509520`, improves x/y/z and all five horizons, slightly improves
  precision, collision F1, and ID switches, but slightly regresses velocity
  and coverage and still fails 120 strict scenario/reference guardrails.

Specification 1.15 and ADR-081 now keep the denominator `1 + fast_weight`
fixed when either branch is unavailable. An unavailable branch contributes
zero without donating its coefficient. The trainer also supports an explicit
`fast_roi` scope and one exact causal-update scope transition. The next run is
configured for 512 fast-ROI-only updates followed by `state_dynamics`, so the
observed localization gain can be frozen while the later phase repairs
velocity, identity, coverage, and rollout behavior without further perception
drift. The intermediate state remains a candidate; promotion is unchanged.

Verification on the staged-scope tree:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider \
  tests/unit/test_config.py tests/unit/test_training_schedule.py \
  tests/unit/test_training_objective_regressions.py \
  tests/unit/test_fast_roi_supervision.py
# 241 passed in 11.57s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider -m "not device"
# 618 passed, 5 skipped, 1 deselected in 149.72s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff check .
# All checks passed

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus \
  ruff format --check .
# 188 files already formatted

PYTHONPYCACHEPREFIX=/private/tmp/orpheus-pycache PYTHONPATH=. \
  conda run -n orpheus python -m compileall -q \
  train.py evaluate.py demo.py scripts world_model tests
# exit 0
```

## 2026-08-07 — long-horizon audit and frozen-backbone correction

The protocol-13 campaign at
`runs/20260806-213753-v7-protocol13-causal-convergence/` was gracefully stopped
at logged step 6,096 after eleven complete post-initialization 512-step
validations produced no promotion. Its optimizer counters accumulated rather
than resetting, maximum RSS stayed at `1,354,514,432` bytes, validation support
remained finite, and stderr stayed empty. This is a healthy but non-converged
campaign, not a collapse. Step 4,096 was the strongest raw full candidate:
weighted horizon score `0.2854974` versus accepted `0.3296688`, with lower RMSE
at every horizon, but lower target/forecast coverage and worse identity and
per-scenario guardrails. Later candidates recovered coverage but gave back the
RMSE gain.

Exact fixed-manifest modular qualifications then separated the coupling:

- dynamics-only step-4,096 transfer was rejected at score `0.3481675` and
  worsened every horizon;
- dynamics/updater/identifier transfer improved score to `0.3265871` and most
  pooled metrics, but regressed x, identity, and 1.0-second RMSE;
- 25% interpolation of the complete model collapsed horizon coverage to
  `0.2250–0.2925` and was rejected at score `0.3387919`;
- the accepted global detector/shared backbone plus donor fast ROI, updater,
  identifier, and dynamics was strongest at score `0.2909420`, improving
  horizon RMSE from `[0.2931762, 0.2949214, 0.3154012, 0.3390725, 0.3608819]`
  to `[0.2249350, 0.2408390, 0.2732086, 0.3060187, 0.3372541]` seconds-aligned
  metres. It remains rejected on identity, small z regressions, forecast
  coverage, and per-scenario guardrails and is not a deployment baseline.
- retaining the accepted ROI lifecycle/appearance heads reduced the gain
  (score `0.3221206`) without restoring acceptance.

The evidence identifies an actual trainability leak: `state_dynamics_roi`
kept the first two shared backbone stages trainable after global-exclusive
heads froze. The new `state_dynamics_fast_roi` scope trains dynamics, belief
update, identifier, ROI updater, and ROI-only fast projection while freezing
every shared backbone/global-discovery tensor. Specification 1.13 and ADR-079
make this distinction explicit. The next production run must initialize
weights-only from the accepted step-zero reference, use zero global-adaptation
steps, and retain the unchanged protocol-13 selector.

New verification on the amended tree:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider -m "not device"
# 609 passed, 5 skipped, 1 deselected in 146.89s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider \
  tests/unit/test_training_schedule.py tests/unit/test_config.py \
  tests/unit/test_checkpoint_composition.py
# 191 passed in 4.02s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff check .
# All checks passed
```

The exact qualification reports/checkpoints are under:

- `runs/20260807-214551-modular-dynamics-step4096/`;
- `runs/20260807-215329-modular-state-dynamics-step4096/`;
- `runs/20260807-220123-interpolated-full-step4096-w025/`;
- `runs/20260807-220945-modular-state-dynamics-fast-roi-step4096/`;
- `runs/20260807-221743-modular-spatial-roi-step4096/`.

The corrected production campaign is active at
`runs/20260807-223146-v8-protocol13-frozen-fast-roi/`. It was launched from
clean pushed commit `ea67f8d5d78826072908c32dc9fd3ddf00576192` through the
one-shot Standard/default LaunchAgent
`com.polceanum.orpheus.protocol13-frozen-roi-20260807-223146`. The resolved
configuration uses `rgb_pretrain_steps=0`, `steps=8192`,
`closed_loop_global_trainable_steps=0`, and
`closed_loop_trainable_scope=state_dynamics_fast_roi`. Host metadata records
Python 3.10, PyTorch 2.10.0, `mps_built=true`, `mps_available=true`, RGB
measurement device MPS, causal device CPU, RGB-only runtime, and no debug
oracle. PID `86599` is alive at the status cut, initialization validation is
complete on all 32 fixed episodes under unchanged protocol hash
`e31bf1cde4e4adf8603190b3258e086d6f749ad1d5689427d60e367f9fbb53a0`,
and stderr is empty. The tensor-linked step-zero checkpoint is accepted with
the exact source score `0.3296687588`; its best/reference/current hashes and
steps all agree. The first causal metric is step 8 after eight draws with zero
skips, one finite optimizer update, `global_perception_trainable=0`, local ROI
perception clipping active, and maximum RSS `1,018,089,472` bytes. No trained
candidate validation, new promotion, or convergence claim exists yet.

## 2026-08-06 — protocol-12 terminal audit and protocol-13 convergence repair

The trainer (`PID 31197`) and supervisor (`PID 35788`) for
`runs/20260803-112948-v6-protocol12-full-convergence/` are no longer running.
The latest metric is step 11,776 and the last durable resumable checkpoint is
step 11,648. There is no `train_summary.json` or convergence decision. Unified
macOS logs record an `OS_REASON_JETSAM` termination at
`2026-08-06 01:01:39.691` during a system-wide memory-pressure event. This is
an externally killed incomplete run, not a plateau or successful convergence.

The causal history also exposed an independent optimization deadlock. Each of
the six broad causal validations at steps 8,704 through 11,264 restored the
accepted step-zero rollout checkpoint and reset Adam. The raw pooled selection
score improved from the fixed `0.3310606914` reference to `0.329669` at step
10,240, but that candidate failed the stricter `elastic_pairs`
reference-relative coverage floor. Treating this deployment failure as a
catastrophic training-support collapse erased every 512-update learning block,
so no causal optimizer history accumulated beyond one validation interval.

Protocol 13 repairs the distinction required by specification Section 164:

- complete per-scenario, reference-relative, and broad guardrails still reject
  deployment promotion;
- nonfinite/invalid state fails closed; a well-formed candidate restores the
  incumbent and resets optimizer state only when pooled current/all-horizon
  coverage falls below absolute floors;
- every validation checkpoint records deployment failures and mutable
  viability failures separately, and convergence inspection uses the latter
  only to determine whether a raw candidate is safe enough to continue
  optimizing.

Long-run integrity is also hardened. Sustained macOS loading is bounded to two
non-persistent workers and one prefetched batch per worker. Phase transitions
collect Python garbage and release the previous MPS/CUDA allocator cache.
Every training metric includes the process maximum-RSS high-water mark.
`training_state.json` becomes `running` before trainer entry, and the
supervisor records a disappeared or externally killed trainer as a terminal
`ExternalTrainerExit` in the primary failure contract. Fast-ROI positive crop
evidence now also requires the explicit association match bit, closing the
previously documented stale-index defensive gap.

Protocol-12 checkpoints are historical evidence and are not exact-resumable
under protocol 13. The best raw step-10,240 candidate is a legitimate
weights-only initialization source for a new timestamped protocol-13
qualification after the quality gate; it must not be called promoted or
converged.

Quality and device verification on the repaired tree:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider -m "not device"
# 603 passed, 5 skipped, 1 deselected in 150.48s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider \
  tests/unit/test_modal_dynamics.py \
  tests/integration/test_rgb_measurements.py \
  tests/unit/test_association.py tests/unit/test_device.py \
  tests/unit/test_evaluation_parameter_update_metrics.py \
  tests/unit/test_rgb_temporal_history.py
# host execution: 70 passed in 8.45s

PYTHONPATH=. conda run -n orpheus ruff check .
# All checks passed!

PYTHONPATH=. conda run -n orpheus ruff format --check .
# 185 files already formatted

PYTHONPYCACHEPREFIX=/private/tmp/orpheus-pycache PYTHONPATH=. \
  conda run -n orpheus python -m compileall -q \
  train.py evaluate.py demo.py scripts world_model tests
# exit 0

git diff --check
# no output
```

The host environment is Python `3.10.20`, PyTorch `2.10.0`,
`mps_built=true`, and `mps_available=true`. The sandbox correctly cannot expose
MPS, so accelerator families were rerun against the host.

A real deterministic CPU causal smoke completed at
`runs/20260806-213442-protocol13-one-update-smoke/`. It applied one supported
optimizer update with no skipped draw, completed both one-episode validations,
wrote a finite terminal checkpoint, persisted `state=completed`, used no oracle
runtime input, and logged process maximum RSS `616,239,104` bytes. Its
from-scratch one-episode candidate had zero physical tracking support and no
deployable incumbent, so the artifact is strictly protocol/checkpoint wiring
evidence, not an accuracy or convergence result. The production qualification
must initialize from the trained protocol-12 candidate rather than this random
smoke.

The repaired production continuation is now active at
`runs/20260806-213753-v7-protocol13-causal-convergence/`. It is a fresh
weights-only protocol-13 run from the finite step-10,240 protocol-12 candidate,
with `rgb_pretrain_steps=0`, 8,192 supported causal updates, 32 fixed balanced
validation episodes, 512-update selector cadence, two bounded loader workers,
and the profile's explicit CPU closed-loop device. It loaded from clean pushed
source commit `1470b2e7186aebe77646e44e3097650abdb57f9d`; run metadata records
`mps_built=true`, `mps_available=true`, `measurement_device=mps`,
`closed_loop_device=cpu`, and matching immutable source fingerprints.

The one-shot Standard/default LaunchAgent
`com.polceanum.orpheus.protocol13-20260806-213753` has `KeepAlive=false`, one
launch, trainer PID `74486`, empty stderr, and explicit
`training_state.json: state=running`. It is currently executing the complete
32-episode initialization validation under protocol hash
`e31bf1cde4e4adf8603190b3258e086d6f749ad1d5689427d60e367f9fbb53a0`.
No optimizer update or new accuracy metric existed at this status cut; launch
health must not be reported as convergence or improvement.

## 2026-08-04 — conservative repository cleanup during live training

The repository was inventoried without touching the active numerical runtime.
At `2026-08-04T21:53:08Z`, the trainer (`PID 31197`) and convergence supervisor
(`PID 35788`) were both still alive. The latest metrics row was finite
measurement update 6,192, the complete step-6,144 measurement validation had
been persisted, and the supervisor continued to write
`waiting_for_segment` heartbeats for the declared 16,384-step minimum.

The cleanup removed 3.0 MiB of regenerable Python bytecode, pytest/Ruff caches,
and editable-install package metadata, plus the empty
`demo_outputs/20260728-151223-scaled-step257/` directory. These paths were moved
to the recoverable quarantine
`/private/tmp/orpheus-cleanup-20260804-215308/`. A follow-up
`git clean -ndX` listed only `runs/` and `demo_outputs/`; no ignored cache,
package-build, temporary checkpoint, coverage, or editor artifact remains in
the repository.

The 2.0 GiB `runs/` tree and 22 MiB of nonempty demos were deliberately
retained. They contain the active campaign, its initialization checkpoint,
selector/reference checkpoints, reproducibility metadata, documented accepted
baselines, rejected controls, and visual audit evidence. Deleting checkpoints
merely because they are large would make the accuracy audit less reproducible.
Likewise, no tracked Python module was deleted: the two evaluation helpers with
no current static caller, `event_metrics.py` and `tracking_metrics.py`, are
explicitly part of the authoritative `PROJECT_SPEC.md` repository contract.
More generally, any `train.py` or `world_model/*.py` deletion would change the
live exact-resume fingerprint. Tracked-code simplification remains a
post-campaign task and must preserve the specification rather than equating
low current call frequency with obsolescence.

Cleanup verification:

```bash
git clean -ndX
# Would remove demo_outputs/
# Would remove runs/

find . -type d \( -name __pycache__ -o -name .pytest_cache \
  -o -name .ruff_cache -o -name .mypy_cache -o -name .hypothesis \) \
  -prune -print
# no output

ps -o pid,ppid,etime,state,%cpu,%mem,command -p 31197,35788
# both original processes alive; trainer on MPS and supervisor waiting

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus \
  ruff check --no-cache .
# All checks passed.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider \
  tests/unit/test_artifact_naming.py \
  tests/unit/test_convergence_supervisor.py
# 25 passed in 5.75s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python -c \
  '<checkpoint/current runtime-source fingerprint comparison>'
# current=stored=43eaaea369ac13a430b2efff224b7f88db973f0a133593966326c095cb16c330
# runtime_match=True

git diff --check
# no output
```

The first read-only fingerprint probe looked for provenance under the obsolete
`source_provenance` checkpoint key and failed with `KeyError`; the corrected
probe used the current `git` payload and produced the matching result above.
No full suite was repeated because tracked executable/test code did not change;
the immediately preceding complete non-device result remains
`599 passed, 5 skipped, 1 deselected`.

## 2026-08-04 — live step-6144 code and continuation audit

The protocol-12 trainer and convergence supervisor remain active while the
repository is audited. At `2026-08-04T21:26:11Z`, the trainer had completed
6,144 finite MPS measurement updates and 19 of 32 episodes in the atomic
step-6144 validation. Both trainer and supervisor stderr files remain empty,
and validation heartbeats continued while the regression suite shared CPU
resources.

The committed repository is synchronized with GitHub:

```text
branch: main
HEAD: fa9f7a9cb7a20287a4e8535a9552b717b5a90f8e
origin/main: fa9f7a9cb7a20287a4e8535a9552b717b5a90f8e
worktree before documentation update: clean
```

A convergence-critical static and dynamic audit covered paired global/fast
measurement supervision, persistent-slot target mapping, positive/negative
ROI masks, phase trainability, causal support, optimizer/checkpoint finite
state, selector promotion, exact resume, prepared propagation, and supervisor
extension logic. A production-profile CPU probe using two real
`elastic_pairs` episodes confirmed finite, nonzero gradients in every
objective-connected ROI head; only the deliberately disconnected ROI event
head was absent. A negative-only tiny probe correctly omitted state/geometry
head gradients rather than fabricating positive support.

No defect exercised by the active campaign was found. One defensive helper
hardening remains valid: `supervised_slot_measurement_losses` should require
both a nonnegative target index and `matched_slots=true` before treating a crop
as positive evidence. Current production callers already replace rejected
indices with `-1`, so the condition is redundant on this run and does not
explain its fast-ROI regression. The long-lived `training_state.json` also
retains `state=starting` until terminal completion/failure; operational
progress is truthful in `training_progress.json`, but a distinct running state
would be clearer.

Both executable edits are deferred until this exact-resume campaign reaches a
terminal supervisor decision. The current executable-source fingerprint is
`43eaaea369ac13a430b2efff224b7f88db973f0a133593966326c095cb16c330`,
exactly matching `checkpoints/last.pt`. Documentation/test-only commits are
safe, but changing `train.py` or `world_model/*.py` now would correctly make a
later supervisor extension reject exact continuation.

Audit commands and observed results:

```bash
PYTHONPATH=. conda run -n orpheus ruff check .
# All checks passed.

PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q \
  -m 'not device'
# 599 passed, 5 skipped, 1 deselected in 247.25s

PYTHONPATH=. conda run -n orpheus python -c \
  '<checkpoint/current runtime-source fingerprint comparison>'
# runtime_match True
```

The five skips are MPS-only tests because sandboxed verification processes
cannot see MPS. The separately launched production process records PyTorch
`2.10.0`, `mps_built=true`, `mps_available=true`, and `device=mps`; its
advancing MPS optimizer/checkpoint evidence remains authoritative.

## 2026-08-04 — continued measurement convergence through step 3584

The same single trainer (`PID 31197`, launch count one) and convergence
supervisor (`PID 35788`, launch count one) remain active after approximately
20 hours. Both stderr logs are empty, checkpoints continue on cadence, and no
metric contains a nonfinite value. The trainer has completed 3,584 MPS
measurement updates and is nine episodes into the atomic step-3584 validation.

New complete fixed-manifest results:

| step | selection score ↓ | global MAE | runtime recall | runtime precision | fast-ROI MAE | accepted |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 2048 | `4.868897` | `0.251612 m` | `0.381250` | `0.523156` | `0.325113 m` | no |
| 2560 | `5.029407` | `0.231081 m` | `0.328750` | `0.579295` | `0.322065 m` | no |
| 3072 | `5.081339` | `0.235489 m` | `0.277500` | `0.623596` | `0.321362 m` | no |

Step 2048 is the best raw broad score so far. The two subsequent results are a
modest regression from that point, but remain substantially better than the
`11.901029` imported measurement score and do not indicate collapse.
Independent 512-update training-window means also remain healthy: mean total
loss moved from `1.63568` in the first window to `0.79936`, `0.88187`, and
`0.82949` in the latest three complete windows, while matched-proposal world
MAE moved from `0.67002 m` to `0.24136`, `0.27832`, and `0.23005 m`.

Fast-ROI localization remains the explicit blocker. Its fixed MAE has settled
near `0.32 m`, worse than the imported `0.189315 m` incumbent, even while
global localization and precision improved. The selector has therefore
preserved `best_measurement.pt` and rejected every trained candidate. At this
point the declared measurement phase is not yet halfway complete and the
broad/training trends are still materially better than initialization.
Changing weights, clipping, or the guardrail during an in-flight validation
would create an incomparable protocol and discard the safe handoff. The
evidence-backed action is to continue the full phase, retain the incumbent,
and let the step-3584 and later fixed validations decide whether fast ROI
recovers. The convergence supervisor remains in `waiting_for_segment` and
will apply the complete-block plateau rule after step 16,384.

## 2026-08-03 — step-2008 convergence audit and plateau supervisor

The active campaign at
`runs/20260803-112948-v6-protocol12-full-convergence/` remains one live
launchd process (`PID 31197`, launch count one) after more than 11 hours. It is
in MPS measurement pretraining at approximately step `2008 / 16384`; stderr is
empty, every logged optimizer update is finite and applied, no JSON metric or
log contains `NaN`/`Infinity`, and `checkpoints/last.pt` continues to advance
on the declared 128-step cadence.

Raw minibatch loss is intentionally not used as the convergence decision
because batches mix eight scenarios and different object counts. Its
256-update means nevertheless improved from `2.25981` in steps 1–256 to
`1.01154`, `1.21964`, `0.721624`, `0.957202`, `0.711013`, `1.00612`, and
`0.761188` in subsequent windows through step 2008. Mean matched-proposal world
MAE fell from `1.02777 m` in the first window to approximately
`0.216–0.339 m` in recent complete windows. Pre-clip gradient norms are
variable and frequently exceed the global bound, but finite gradients are
consistently clipped to the configured `2.0`; there is no skipped or rejected
optimizer update.

The fixed 32-episode measurement selector provides the trustworthy trend:

| step | selection score ↓ | global MAE | runtime recall | runtime precision | fast-ROI MAE | accepted |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | `11.901029` | `1.560788 m` | `0.228750` | `0.298046` | `0.189315 m` | yes |
| 512 | `5.688880` | `0.239608 m` | `0.256250` | `0.571031` | `0.344317 m` | no |
| 1024 | `5.625772` | `0.368076 m` | `0.248750` | `0.496259` | `0.234358 m` | no |
| 1536 | `5.305358` | `0.247284 m` | `0.288750` | `0.589286` | `0.307533 m` | no |

The lower-is-better broad score has improved at every validation and the
trained global path is much better than its imported baseline. This is real
progress, not collapse. No candidate has been promoted because the independent
fast-ROI guard correctly rejects its localization regression relative to the
`0.189315 m` incumbent. The imported measurement checkpoint therefore remains
safe and will be restored at the phase handoff unless a later candidate
improves without that regression. The current evidence supports continuing the
long run; changing learning rate, clipping, or guardrails at one quarter of
the measurement phase would discard a monotone broad trend without proving a
better replacement.

The repository convergence supervisor is now attached persistently:

```text
label: com.polceanum.orpheus.convergence-20260803-112948
PID: 35788
minimum segment: 16384 updates
extension size: 4096 updates
plateau tail: 1024 updates
minimum relative gain: 1%
hard limit: 24576 updates
state: waiting_for_segment
stderr bytes: 0
```

It monitors the exact initial trainer PID, verifies the completed summary and
all selector links, and only resumes from the finite in-place `last.pt` after a
complete 16,384-step segment. It declares plateau only from the predeclared
four-validation rule; otherwise it extends by complete 4,096-step blocks up to
24,576 and reports `limit_hit` rather than fabricating convergence. Durable
events are in `convergence_supervisor.jsonl`; the final decision will be in
`convergence_report.json`.

## 2026-08-03 — full protocol-12 convergence campaign launched

The full default sustained profile is active at
`runs/20260803-112948-v6-protocol12-full-convergence/`, initialized from the
accepted protocol-12 selector checkpoint
`runs/20260803-110550-v6-protocol12-one-update-smoke/checkpoints/best_rollout.pt`.
It was launched from clean pushed commit `e08c4d0` with:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python \
  scripts/launch_training_once.py \
  --label com.polceanum.orpheus.v6-20260803-112948 \
  --config configs/sustained_accuracy_mps_v3.yaml \
  --run-name 20260803-112948-v6-protocol12-full-convergence \
  --device mps \
  --initialize-from \
    runs/20260803-110550-v6-protocol12-one-update-smoke/checkpoints/best_rollout.pt
```

The resolved schedule is the unshortened profile: `16,384` optimizer steps,
`16,384` training episodes, `8,192` measurement-pretraining steps, `8,192`
closed-loop steps, 32 fixed validation episodes, validation every 512 steps,
and checkpoints every 128 steps. Measurement training uses MPS float32; the
profile intentionally moves causal closed-loop optimization to CPU.

Initial launch evidence:

```text
launchd label: com.polceanum.orpheus.v6-20260803-112948
launchd state / runs / PID: running / 1 / 31197
KeepAlive / ProcessType: false / omitted (Standard default)
source commit / dirty: e08c4d0 / false
MPS built / host available: true / true
runtime modality / oracle: rgb / false
closed-loop initialization validation: 32 / 32 episodes complete
closed-loop validation elapsed / mean: 889.508 s / 27.797 s per episode
closed-loop batch-time range: 24.559–28.342 s
initial selector score / support / accepted: 0.3310606914 / true / true
position / velocity RMSE: 0.308032 m / 1.135027 m/s
target coverage / prediction precision: 0.322500 / 0.370903
collision F1 / ID-switch rate: 0.225519 / 0.006834
nominal-90% position coverage: 0.890140
measurement-incumbent validation: 2 / 32 episodes complete
first two MPS measurement batch seconds: 104.511, 82.898
CPU utilization during closed-loop validation: approximately 530–541%
stderr bytes: 0
optimizer updates so far: 0
```

This restores the expected foreground throughput and shows no restart,
deadlock, nonfinite failure, or launch-QoS regression. The complete broad
initialization selector accepted the imported checkpoint and wrote
`best_rollout.pt`, `reference_rollout.pt`, and
`validation_step_000000.pt`. The trainer is now evaluating the separate
measurement incumbent on host MPS before the first optimizer update. The first
measurement episode paid substantial first-use MPS/hybrid warmup; the second
fell from `104.511` to `82.898 s`, with live progress and empty stderr. No
training trend or convergence claim exists yet. The authoritative heartbeat is
`training_progress.json`; measurement selector evidence remains atomic until
all 32 episodes complete.

## 2026-08-03 — launch-QoS and dynamics-call collapse audit

The clean commit-`2487b7e` run at
`runs/20260803-101108-v5-protocol11-balanced-qualification/` was alive,
single-launch, finite, and advancing. It completed five distinct fixed
validation seeds with no stderr:

```text
completed episodes: 5 / 32
batch seconds: 123.660, 96.602, 118.193, 127.201, 121.245
mean completed-batch time: 117.380 s
optimizer updates: 0
metrics/checkpoints: none
terminal progress: validation_interrupted / KeyboardInterrupt
terminal trainer state: failed / KeyboardInterrupt
```

This was not numerical collapse. The heartbeat advanced, the original PID
remained authoritative, and the interruption stack landed inside ordinary
finite interaction rollout work. It was stopped deliberately because a matched
repaired foreground control with the same 40 frames, eight rollout anchors,
birth-confirmation/lifecycle settings, model, and first 16 validation seeds
took `404.879 s`, or `25.305 s/episode`. The older 32-episode v3 initialization
took `30.861 s/episode`. The current one-shot plist set
`ProcessType=Background`; observed utilization fell from roughly `525%` in the
foreground control to about `100–198%`. The `4.64x`/`3.80x` wall-clock ratios
track that lost parallel CPU use. Correcting global cadence adds at most four
global observations over 40 frames (`10 -> 14`) and cannot explain the
regression; validation-loader inter-batch gaps were below `0.16 s`.

The launch helper no longer emits a Background classification. Its
`KeepAlive=false` and `caffeinate` one-shot semantics remain unchanged.

The same audit found two independent dynamics-path problems:

- float32 20 Hz timestamps caused 22 of 39 intended `6 x 1/120 s` frame
  intervals to execute seven belief-dynamics substeps because a literal
  ceiling saw a ratio a few representation units above six; simulator labels
  always used six; and
- every noninitial causal frame propagated the persistent belief once for
  supervision/current-correction and again inside `OnlineWorldModel.ingest`.

Belief dynamics now snap only precision-indistinguishable integral ratios and
otherwise retain the ceiling. A typed, single-use prepared propagation lets
training inspect and then consume the same prior through the ordinary ingest
path without replacing `WorldBelief`, zeroing elapsed time, or losing interval
collision evidence. Belief/result tensors and dynamics parameters/buffers/mode
are revision-bound; in-place value/graph mutation, tensor replacement, reuse,
and nonuniform batch targets fail closed. The guard uses ordinary autograd or
`no_grad`, not `inference_mode`. Training rollouts may also skip stacking
unused auxiliary trajectories. These numerical semantics bump rollout
validation protocol 11 to 12; simulator `sphere_world_v4`, measurement
protocol 5, and selection metric 6 remain unchanged. Old selector artifacts
are not protocol-v12 incumbents.

The identity/lifecycle re-audit found no remaining structural defect in
tentative-birth cardinality gating, fast-ROI source identity, global discovery,
or accepted-association-only parameter history. A missing regression now
proves that a physically distance-rejected burn-in association cannot seed
slow-parameter frame/ID history.

Final commands, test counts, timing controls, commit, and the next launch are
recorded below when they have actually completed. A direct foreground timing
control at
`runs/20260803-105244-v6-protocol12-foreground-timing/` completed fixed
reference seed `100000` in `29.578 s` (`33.493 s` including loader/startup),
versus `123.660 s` for the same seed/scenario in the Background job: a `4.18x`
improvement and within the historical foreground range. It established a
single-scenario diagnostic reference, then was intentionally interrupted while
drawing its first training batch; it performed zero optimizer updates and is
not accuracy/convergence evidence.

Current-tree quality gate:

```bash
PYTHONPATH=. conda run -n orpheus pytest -q \
  tests/integration/test_prepared_propagation.py \
  tests/unit/test_prepared_closed_loop.py \
  tests/unit/test_analytic_dynamics.py \
  tests/unit/test_event_window_scoring.py \
  tests/unit/test_training_objective_regressions.py \
  tests/unit/test_parameter_supervision.py \
  tests/unit/test_launch_training_once.py \
  tests/unit/test_scheduler.py \
  tests/integration/test_trainer_checkpoint_integrity.py
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q
PYTHONPATH=. conda run -n orpheus ruff format .
PYTHONPATH=. conda run -n orpheus ruff check .
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-v6-pycache PYTHONPATH=. \
  conda run -n orpheus python -m compileall -q \
  train.py evaluate.py demo.py scripts world_model tests
PYTHONPATH=. conda run -n orpheus python train.py \
  --config configs/sustained_accuracy_mps_v3.yaml --dry-run --device cpu
git diff --check
```

```text
affected tests: 100 passed in 97.20 s on the final documented tree
full sandbox suite: 599 passed, 6 MPS-only skipped in 315.68 s
Ruff format: 185 files unchanged
Ruff lint: all checks passed
compileall: passed
production-profile CPU dry run: passed
git diff --check: passed
Python / PyTorch: 3.10.20 / 2.10.0
MPS compiled / sandbox-visible: true / false
```

The attempted host-MPS family rerun could not start because the execution
approval service reported its external-usage limit; this is an infrastructure
block, not a passing device result. The previous committed tree's host MPS
families remain valid only for that earlier source.

A reduced production-model causal smoke then exercised the repaired path
through a real optimizer step and terminal checkpoint/validation:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/sustained_accuracy_mps_v3.yaml \
  --run-name v6-protocol12-one-update-smoke \
  --initialize-from \
    runs/20260730-192625-scaled-sustained-e2e-v1/checkpoints/best_measurement.pt \
  --device cpu \
  --set 'simulator.scenario_mixture=[reference_pairs]' \
  --set simulator.sequence_frames=16 \
  --set training.steps=1 \
  --set training.rgb_pretrain_steps=0 \
  --set training.batch_size=1 \
  --set training.tbptt_steps=4 \
  --set training.rollout_anchors_per_window=1 \
  --set training.validation_rollout_anchors_per_episode=1 \
  --set training.minimum_rollout_age_steps=1 \
  --set training.validation_minimum_predictable_target_count_per_scenario_horizon=1 \
  --set training.validation_minimum_matched_target_count_per_scenario_horizon=1 \
  --set training.validation_minimum_supported_episodes_per_scenario=1 \
  --set training.train_episodes=2 \
  --set training.validation_episodes=1 \
  --set training.num_workers=0 \
  --set training.checkpoint_every=1 \
  --set training.eval_every=1 \
  --set training.log_every=1 \
  --set training.maximum_no_gradient_batches_per_update=16 \
  --set 'training.horizon_weights=[1.0,1.5,2.0]' \
  --set 'evaluation.horizons_seconds=[0.1,0.25,0.5]' \
  --set evaluation.episodes=1
```

```text
run: runs/20260803-110550-v6-protocol12-one-update-smoke/
terminal state: completed
optimizer updates: 1
skipped no-gradient batches: 0
loss: 4.273417
gradient norm before local/global clipping: 3.012750
gradient norm applied: 1.246404
post-step finite-state check: passed
terminal checkpoint: checkpoints/last.pt
initial incumbent score: 0.21818814932904948
step-1 candidate score: 0.2181897207709593
selection: rejected; imported incumbent preserved
elapsed: 16.915 s
oracle runtime input: false
```

The one-step candidate's tiny broad-score regression was correctly rejected,
which is selector-integrity evidence rather than an accuracy result. The
repair was committed and pushed as `e08c4d0`; the full production launch is
recorded above. No full-manifest protocol-v12 accuracy, promotion, or
convergence result exists yet.

## 2026-08-03 — cadence, progress, and finite-state collapse audit

The clean one-shot qualification at
`runs/20260803-084843-v4-balanced-qualification/` did not relaunch, deadlock,
or numerically collapse. The original PID accumulated roughly 83 CPU minutes
in about 44 wall-clock minutes inside finite dynamics/contact rollout work.
Its stable eight worker processes showed no churn, stderr stayed empty, and
the lock/source/command identities matched commit `97415b0`. It nevertheless
produced no metric or checkpoint because validation was full-manifest atomic
and had no per-episode heartbeat.

The run was intentionally interrupted before training after an independent
audit found a real runtime-semantics bug:

```text
configured global_every_steps: 3
old actual modes: GLOBAL, FAST, FAST, FAST, GLOBAL
specified modes:  GLOBAL, FAST, FAST, GLOBAL
optimizer updates: 0
metrics/checkpoints: none
terminal state: failed / KeyboardInterrupt
launchd restarts: 0
```

The bug inserted a third stale ROI-only frame, increasing tracker drift and
identity/lifecycle risk. Historical results labelled “cadence three” therefore
measured actual cadence four. They remain evidence for that old behavior but
are not comparable selector/reference evidence for the corrected runtime.
Rollout validation protocol is now 11; simulator v4, measurement protocol 5,
and selection metric 6 are unchanged.

The repaired runtime now:

- treats `global_every_steps` as the complete distance between global frames
  and rejects zero, non-integral, or boolean values;
- regression-tests the exact `GLOBAL, FAST, FAST, GLOBAL` sequence;
- atomically writes `training_progress.json` and flushes one stdout heartbeat
  per validation episode with phase, counts, timings, PID, seed/scenario, and
  protocol hash while keeping partial results out of checkpoint selection;
- starts the training iterator/workers only when the first training draw is
  required, not during initial/handoff validation;
- checks all floating/complex parameters and optimizer tensors immediately
  after every successful Adam step, including finite nonnegative scalar step
  counters; and
- rejects nonfinite model buffers/weights, optimizer/scheduler state, or
  invalid step counters before atomic checkpoint replacement and before load
  mutates a destination. Tests prove an existing checkpoint stays byte
  identical after a corrupt overwrite attempt.

Two current-tree wiring runs completed:

```text
runs/20260803-095310-v5-cadence-progress-cpu-smoke/
  device: CPU
  actual updates: 1 causal
  loss: 888.003357
  raw/applied gradient norm: 15085.8096 / 1.0000
  trajectory/fast/objective support: 32 / 32 / 3
  skipped draws: 0
  elapsed: 123.94 s
  terminal state: completed
  selection: unsupported random initialization; no promotion

runs/20260803-095618-v5-poststep-mps-host-smoke/
  device: host MPS, PyTorch 2.10.0
  actual updates: 1 RGB measurement
  loss: 911.012146
  raw/applied gradient norm: 22329.9609 / 1.0000
  skipped draws: 0
  elapsed: 43.16 s
  terminal state: completed
  selection: unusable random initialization; no promotion
```

Both runs are finite-state/progress wiring evidence only. Their random-model
losses and unsupported selectors are not accuracy results. The CPU run's
`training_progress.json` ended `validation_complete` with exact seed/scenario
and protocol hash; the host MPS run persisted a finite `last.pt`.

Final current-tree quality gate:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus ruff format .
PYTHONPATH=. conda run --no-capture-output -n orpheus ruff check .
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q \
  tests/unit/test_config.py tests/unit/test_scheduler.py \
  tests/unit/test_training_schedule.py \
  tests/integration/test_rgb_online_loop.py \
  tests/integration/test_checkpoint_roundtrip.py \
  tests/integration/test_trainer_checkpoint_integrity.py
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  conda run --no-capture-output -n orpheus pytest -q \
  tests/integration/test_rgb_measurements.py \
  tests/unit/test_modal_dynamics.py \
  tests/unit/test_evaluation_parameter_update_metrics.py \
  tests/unit/test_association.py
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-v5-audit-pycache PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python -m compileall -q \
  train.py evaluate.py demo.py scripts world_model tests
git diff --check
```

```text
Ruff format: 182 files formatted/already formatted
Ruff lint: all checks passed
affected scheduler/config/trainer/checkpoint tests: 252 passed in 66.88 s
full sandbox suite: 583 passed, 6 MPS-only skipped in 275.96 s
host MPS families: 38 passed in 13.16 s
compileall: passed
```

The first direct MPS smoke inside the restricted sandbox failed before model
construction because that process reported MPS compiled but unavailable. The
same command rerun on the host completed successfully as recorded above; no
dependency or PyTorch installation was changed.

## 2026-08-03 — initialization-support and launch-failure audit

The apparent qualification at
`runs/20260803-000858-v3-collapse-repair-qualification/` was not training
slowly or noisily: it never took an optimizer step. Its complete durable
evidence is:

```text
metrics rows: 1
row: step 0, validation_initialization_incumbent
checkpoints: reference_rollout.pt, validation_step_000000.pt
last.pt / best_rollout.pt / best_measurement.pt: absent
terminal error:
  AssertionError: initialization validation must establish the first incumbent
```

The step-zero metrics and checkpoint tensors are finite, and the stored tensor
hashes match. This rules out numeric NaN/Inf collapse at that point. Instead,
all four fixed `impulse_perturbation` validation episodes were unsupported at
one second. The old probability `0.12` was applied once per 20 Hz observation
interval, not once per episode, so a 40-frame episode expected `4.68` unseen
impulses. The exact affected seeds produced event frames:

```text
100004: 19, 24, 29
100012: 5, 17, 35, 39
100020: 8, 17
100028: 5, 10, 11, 26, 33, 37
```

That left essentially no causally identifiable one-second point targets at the
fixed validation anchors. The assertion killed the first process.
Unfortunately the `launchctl submit` job behaved as KeepAlive and restarted
it more than 2,284 times. Each later process failed with `FileExistsError`
against the first attempt's occupied run directory. The stderr log grew to
about 1.17 MB / 18,000 lines without learning. The launchd label
`com.polceanum.orpheus.v3-repair-20260803-000858` has been booted out. The run
is retained as a launch/protocol failure and contains no convergence trend.

The current repair:

- changes the stochastic impulse rate to `0.02` per observation interval,
  which still produces real surprises but preserves deterministic windows;
- uses one shared scene-wide causal mask in training and evaluation for point,
  event, collision-conditioned, and correction metrics after unseen
  actuation, while forecast calibration still scores stochastic outcomes;
- requires per-scenario, per-horizon minima of four label-predictable targets,
  two matched targets, and two independently supported episodes for v3
  promotion;
- persists exact predictable/censored/coordinate/episode support evidence and
  lets only an explicit insufficient-support condition become a rejected
  candidate; missing metric schema is fatal;
- records fully resolved scenario configurations in protocol hashes and bumps
  simulator semantics to `sphere_world_v4`;
- continues after an unsupported imported initialization, removes the
  accidental full 32-episode validation before every causal optimizer update,
  and applies available fixed-reference/training guardrails before the first
  later promotion;
- writes atomic starting, failed, and completed CLI state artifacts; and
- launches future macOS jobs from an explicit one-shot plist with
  `KeepAlive=false`. The legacy convergence supervisor also boots out a failed
  initial job.

An exact current-tree CPU replay of the imported checkpoint on the four fixed
impulse seeds produced:

```text
per-seed supported: 100004 yes, 100012 yes, 100020 no, 100028 yes
supported episodes: 3 / 4 (minimum 2)
predictable targets @ 0.1/0.25/0.5/0.75/1.0 s:
  116 / 98 / 80 / 62 / 47 (minimum 4 each)
matched targets @ 0.1/0.25/0.5/0.75/1.0 s:
  40 / 30 / 23 / 16 / 12 (minimum 2 each)
scenario selection support: pass
pooled selection support: pass
```

One seed is deliberately reported unsupported rather than hidden. This is
support/protocol evidence, not an accuracy result.

The completed quality gate on the final repaired tree was:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus ruff format --check .
PYTHONPATH=. conda run --no-capture-output -n orpheus ruff check .
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-v4-audit-pycache PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python -m compileall -q \
  train.py evaluate.py demo.py scripts world_model tests
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q
```

```text
Ruff format: 182 files already formatted
Ruff lint: all checks passed
compileall: passed
pytest: 577 passed, 6 MPS-only skipped in 180.59 s
```

Host hardware and focused accelerator validation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python -c \
  "import platform, torch; print(platform.python_version()); \
   print(torch.__version__); print(torch.backends.mps.is_built()); \
   print(torch.backends.mps.is_available())"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  conda run --no-capture-output -n orpheus pytest -q \
  tests/integration/test_rgb_measurements.py \
  tests/unit/test_modal_dynamics.py \
  tests/unit/test_evaluation_parameter_update_metrics.py \
  tests/unit/test_association.py
```

```text
Python 3.10.20
PyTorch 2.10.0
MPS built: true
MPS available: true
38 passed in 8.36 s
```

The clean production-profile causal smoke used the imported paired-RGB
checkpoint but no oracle runtime input:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/sustained_accuracy_mps_v3.yaml \
  --initialize-from \
    runs/20260730-192625-scaled-sustained-e2e-v1/checkpoints/best_measurement.pt \
  --run-name 20260803-081824-v4-collapse-audit-cpu-smoke \
  --device cpu \
  --set training.steps=1 \
  --set training.rgb_pretrain_steps=0 \
  --set training.train_episodes=16 \
  --set training.validation_episodes=16 \
  --set training.batch_size=2 \
  --set training.num_workers=0 \
  --set training.eval_every=1 \
  --set training.checkpoint_every=1 \
  --set training.log_every=1
```

It completed one real update in `839.80 s`, with loss `3.006042`, raw/applied
gradient `5.617366 / 1.268207`, trajectory support `27`, fast-ROI support `12`,
thirteen differentiable causal objective terms, no skipped batch, and a
terminally validated `last.pt`. The fixed reference passes tensor, protocol,
support-schema, and raw-additive recomputation checks.

Pooled validation changed as follows:

| metric | imported reference | step 1 |
|---|---:|---:|
| position RMSE | `0.314256 m` | `0.307586 m` |
| velocity RMSE | `1.178649 m/s` | `1.180215 m/s` |
| target coverage | `0.293627` | `0.292157` |
| prediction precision | `0.359760` | `0.364972` |
| collision F1 | `0.269939` | `0.254545` |
| ID-switch rate | `0.013267` | `0.013378` |
| nominal-90% coverage | `0.880747` | `0.884518` |
| horizon RMSE 0.1/0.25/0.5/0.75/1.0 s | `0.325038 / 0.329932 / 0.338065 / 0.349248 / 0.357262 m` | `0.319815 / 0.320804 / 0.325284 / 0.333406 / 0.334316 m` |

The aggregate/horizon position gains were not promoted: coverage, collision,
some axes, and scenario support regressed, producing 61 explicit guardrail
reasons. The run has no `best_rollout.pt`; that is correct for a one-update
smoke. Artifacts:

- `runs/20260803-081824-v4-collapse-audit-cpu-smoke/checkpoints/last.pt`
- `runs/20260803-081824-v4-collapse-audit-cpu-smoke/checkpoints/reference_rollout.pt`
- `runs/20260803-081824-v4-collapse-audit-cpu-smoke/metrics.jsonl`
- `runs/20260803-081824-v4-collapse-audit-cpu-smoke/train_summary.json`
- `runs/20260803-081824-v4-collapse-audit-cpu-smoke/training_state.json`

The actual host-MPS optimizer-progress smoke was:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name 20260803-083723-v4-mps-optimizer-smoke \
  --device mps \
  --set device.global_detector_cpu_on_mps=true \
  --set training.steps=1 \
  --set training.rgb_pretrain_steps=1 \
  --set training.train_episodes=2 \
  --set training.checkpoint_every=1 \
  --set training.eval_every=1 \
  --set training.log_every=1 \
  --set training.num_workers=0
```

It completed on MPS in `429.06 s`; training/validation losses were finite
(`26.659035 / 5.401521`), the first-step raw gradient `2946.836426` was clipped
to `1.0`, the optimizer update was applied, terminal validation completed, and
no failure artifact exists. A random one-update measurement model was correctly
reported unusable and not promoted. This proves real MPS optimizer/checkpoint
wiring, not accuracy. Artifacts are under
`runs/20260803-083723-v4-mps-optimizer-smoke/`.

## 2026-08-03 — v3 collapse audit and lifecycle/identity repair

The 3,072-update qualification at
`runs/20260802-123714-v3-medium-qualification/` was not converging safely.
Its first causal validation at step `1536` reduced the lower-is-better pooled
score from `0.7462555` to `0.5878015`, position RMSE from `0.8274599 m` to
`0.5960934 m`, and velocity RMSE from `1.4834236 m/s` to `1.0865632 m/s`.
Those conditional gains hid structural collapse:

- predicted object frames rose from `3950` to `5274` for `4000` targets;
- distance-gated identity switches rose from `10/1333` (`0.007502`) to
  `146/2217` (`0.065855`);
- collision false positives rose from `242` to `469`, with collision F1
  falling from `0.222222` to `0.191781`;
- nominal-90% coverage moved from `0.877841` to `0.978778`, increasing
  calibration error from `0.022159` to `0.078778`;
- all eight scenario slices contributed to the `38` persisted rejection
  reasons.

The selector correctly rejected the candidate. Training was deliberately
stopped after logged update `1776` / data draw `1779`; the last durable
checkpoint is step `1728`. The launchd job was removed and the run is
preserved as a failed-protocol diagnostic. Do not resume it: the source defects
below change lifecycle, association, supervision, and optimizer semantics.

The audit found and repaired seven causal defects:

1. New persistent simulator-target mappings were accepted at arbitrary
   distance. In the failed validation, only `1777/3830` newly assigned
   belief-target frames were within the declared `0.5 m` physical gate.
   New mappings are now distance-gated before Hungarian assignment; a live
   persistent ID mapping remains locked while its target exists.
2. Runtime configuration exposed `birth_confirmations`, but values other than
   one were rejected and every unmatched confident global proposal received a
   permanent ID immediately. Detached `(modality, sensor)`-local tentative
   evidence now requires consecutive strictly-later detections within a
   configurable world-space gate before monotonic ID allocation.
3. Core association and tentative confirmation solved ungated costs and
   discarded invalid pairs afterward. Both now pre-gate with
   valid-cardinality-first assignment. The regression matrix
   `[[0,20],[20,31]]` at maximum cost `30` retains both valid cross-pairs.
4. Prior-conditioned fast ROI rows could be Hungarian-assigned to another
   persistent object even though their features already mixed in the source
   prior. Fast rows now carry source slot/ID and may update only that identity;
   global discovery remains freely associated.
5. Births and stale target history could open drag/restitution supervision
   without a same-ID accepted innovation. Parameter gates now use accepted,
   distance-gated runtime associations only and reset their temporal baseline
   whenever the associated runtime ID changes.
6. Late causal raw gradients were dominated by RGB perception rather than the
   already locally clipped interaction block. An exact snapshot decomposed raw
   norm `11.784` into global-detector `10.036`, backbone `5.959`, ROI `1.585`,
   interaction-edge `0.306`, and interaction-node `0.195`. A causal-only
   `1.0` local cap now covers the complete RGB observation module before the
   independent interaction and whole-model clips. RGB pretraining retains its
   original whole-model clipping semantics, and global causal perception
   adaptation is bounded to `512` updates.
7. Once false privileged mappings were removed, a deliberately tiny
   zero-support validation crashed while deriving horizon RMSE. Pooled
   validation now retains the raw zero counts, marks selection unsupported,
   persists numbered/reference diagnostic artifacts, and rejects the candidate
   without fabricating zero RMSE or aborting training.

A dirty-tree CPU wiring run completed while these changes were under test:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/sustained_accuracy_mps_v3.yaml \
  --initialize-from \
    runs/20260730-192625-scaled-sustained-e2e-v1/checkpoints/best_measurement.pt \
  --run-name 20260802-233339-collapse-repair-cpu-smoke \
  --device cpu \
  --set 'simulator.scenario_mixture=[reference_pairs]' \
  --set training.steps=4 \
  --set training.rgb_pretrain_steps=2 \
  --set training.train_episodes=8 \
  --set training.validation_episodes=2 \
  --set training.batch_size=2 \
  --set training.eval_every=2 \
  --set training.checkpoint_every=1 \
  --set training.log_every=1 \
  --set training.num_workers=0 \
  --set training.validation_rollout_anchors_per_episode=2 \
  --set training.measurement_validation_frames=2
```

```text
run: runs/20260802-233339-collapse-repair-cpu-smoke/
updates: 4/4 (2 paired RGB, 2 supported causal)
episode draws: 8
skipped/no-gradient batches: 0
device: CPU
elapsed: 176.5751 s
oracle runtime input: false
```

At its final causal update, the true raw norm was `3.1444`; perception was
locally reduced from `3.10125` to `1.0`, interaction remained unscaled at
`0.18152`, and the pre-global/final norm was `1.12674`. This proves gradient,
runtime, checkpoint, and terminal-validation wiring only. Its two
`reference_pairs` validation episodes and inherited safe incumbent are far too
small for an accuracy or convergence claim. A clean host MPS/CPU smoke and
then a new timestamped balanced qualification remained required.

The clean host requirement was subsequently satisfied from pushed commit
`c8695716b11f64462741971e7179cccf3f54b15a`:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/sustained_accuracy_mps_v3.yaml \
  --initialize-from \
    runs/20260730-192625-scaled-sustained-e2e-v1/checkpoints/best_measurement.pt \
  --run-name 20260803-000212-collapse-repair-host-smoke \
  --device mps \
  --set 'simulator.scenario_mixture=[reference_pairs]' \
  --set training.steps=4 \
  --set training.rgb_pretrain_steps=2 \
  --set training.train_episodes=8 \
  --set training.validation_episodes=2 \
  --set training.batch_size=2 \
  --set training.eval_every=2 \
  --set training.checkpoint_every=1 \
  --set training.log_every=1 \
  --set training.num_workers=0 \
  --set training.validation_rollout_anchors_per_episode=2 \
  --set training.measurement_validation_frames=2
```

```text
run: runs/20260803-000212-collapse-repair-host-smoke/
source: c8695716b11f64462741971e7179cccf3f54b15a, clean
updates: 4/4 (2 paired RGB, 2 supported causal)
episode draws: 8
devices: paired RGB MPS/CPU hybrid; causal CPU
elapsed: 216.6836 s
skipped/no-gradient batches: 0
oracle runtime input: false
terminal checkpoint: step 4, final_validation_completed=1
```

Both paired RGB updates kept the causal perception-local cap disabled and were
finite under the ordinary whole-model clip: raw norms `5.85800` and `4.35465`
were applied at `2.0`. The causal updates had trajectory/fast-slot support
`69/23` and `111/36`. Their raw perception norms `3.22456` and `3.10125`
were locally capped at `1.0`; interaction norms `0.77003` and `0.18152`
remained unscaled, leaving final norms `1.32850` and `1.12674`.

Terminal pooled score moved `0.4017391 → 0.3968408` (lower is better), but
target coverage fell `0.23 → 0.20`, 0.1-second forecast coverage fell
`0.50 → 0.40`, and 0.1-second y RMSE crossed its tolerance. The selector
persisted all six pooled/scenario reasons, retained the safe incumbent, and
reported zero distance-gated identity switches. This is clean execution and
guardrail evidence, not a promoted accuracy result. A new balanced medium
qualification remains the next convergence test.

The next medium qualification was launched after the clean smoke and evidence
commit were pushed:

```bash
launchctl submit \
  -l com.polceanum.orpheus.v3-repair-20260803-000858 \
  -o /private/tmp/20260803-000858-v3-collapse-repair-qualification.stdout.log \
  -e /private/tmp/20260803-000858-v3-collapse-repair-qualification.stderr.log \
  -- /usr/bin/caffeinate -dimsu \
  /usr/bin/env PYTHONPATH=/Users/mike/Work/world.model \
  /usr/local/Caskroom/miniforge/base/envs/orpheus/bin/python \
  /Users/mike/Work/world.model/train.py \
  --config /Users/mike/Work/world.model/configs/sustained_accuracy_mps_v3.yaml \
  --initialize-from \
  /Users/mike/Work/world.model/runs/20260730-192625-scaled-sustained-e2e-v1/checkpoints/best_measurement.pt \
  --run-name 20260803-000858-v3-collapse-repair-qualification \
  --device mps \
  --set training.steps=3072 \
  --set training.rgb_pretrain_steps=1024 \
  --set training.train_episodes=6144 \
  --set training.checkpoint_every=64
```

```text
run: runs/20260803-000858-v3-collapse-repair-qualification/
source: baca6a8cc418a9f1a8e6321124a46026cfcc0004, clean and pushed
declared updates: 3,072 = 1,024 paired RGB + 2,048 causal
actual optimizer updates: 0
actual episode draws: 0
scenarios: eight unique balanced families
validation: 32 fixed episodes, eight anchors, every 512 updates
devices: MPS paired RGB / CPU causal
lifecycle: two confirmations within 0.5 m
gradient caps: perception 1.0 / interaction 1.0 / whole model 2.0
global causal perception window: 512 updates
```

Immediate host inspection saw PID `70085` spending CPU on the initial
validation, but later forensic inspection proved that process failed before
training and launchd repeatedly replaced it. This is exactly why PID/CPU and
empty-at-launch logs are not sufficient launch-health evidence. The complete
failure evidence and corrected protocol are recorded in the newer audit
section above.

Known lifecycle limitations remain explicit: confirmation is spatial-only, so
repeated association failure near a missed live track can still confirm a
duplicate under the current single-hypothesis tracker. If more confirmed
proposals arrive than there are free slots, confidence-ordered allocation
keeps the strongest and an unallocated real candidate must reconfirm after
capacity opens. These affect recovery latency/duplicate risk but do not
corrupt an existing belief slot.

Verification on the repaired tree:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-collapse-pycache PYTHONPATH=. \
  conda run --no-capture-output -n orpheus \
  python -m compileall -q world_model tests train.py evaluate.py demo.py
conda run --no-capture-output -n orpheus \
  ruff check world_model tests train.py evaluate.py demo.py
conda run --no-capture-output -n orpheus \
  ruff format --check world_model tests train.py evaluate.py demo.py
git diff --check
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q
```

Compile, Ruff, formatting, and diff checks passed. The complete sandbox suite
reported `536 passed, 6 skipped in 143.58s`; every skip was an MPS-only test.
The environment is Python `3.10.20`, PyTorch `2.10.0`, with MPS compiled in.
The same final source then ran the skipped device families outside the sandbox:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q \
  tests/integration/test_rgb_measurements.py \
  tests/unit/test_association.py \
  tests/unit/test_evaluation_parameter_update_metrics.py \
  tests/unit/test_modal_dynamics.py
```

Host MPS was available and all `36` tests passed in `8.36s`.

## 2026-08-02 — supported-causal convergence repair and v3 qualification

The supposedly corrected v2 campaign is finite but not a valid convergence
trajectory. It was stopped at logged step `9576`, and both persistent jobs were
removed without deleting artifacts:

```text
run: runs/20260801-232229-scaled-sustained-v2/
stopped trainer: com.polceanum.orpheus.sustained-v2-20260801-232229
stopped supervisor: com.polceanum.orpheus.convergence-v2-20260801-232229
```

Across its 173 logged causal training rows, 121 (`69.94%`) had an exactly zero
pre-clip gradient. Those rows still consumed scheduled updates. The imported
step-zero validation had current distance-gated target coverage `0.287539` and
one-second forecast target coverage `0.761458`; at the step-8192 perception
handoff those values had collapsed to `0.044805` and `0.052734`. Conditional
RMSE among the few surviving tracks therefore understated the failure.

The full audit found and repaired the following independent defects:

- causal updates could be consumed by global auxiliary measurement loss with
  no differentiable state, rollout, correction, parameter, or fast-slot
  support;
- inactive factory queries contributed a constant existence objective;
- fast ROI confidence was positive-only, empty/unreliable crops supervised
  unsupported attributes, false positives were absent from selector
  precision, and pretraining never exercised the temporal cache;
- global and fast measurement losses were diluted by support-dependent list
  averaging, while unsupported event/physical terms appeared as zero-valued
  examples or misleading zero RMSE;
- the first unsupported rollout candidate could be labelled best, later
  support collapse did not reliably restore a verified incumbent/reset Adam,
  and handoff state could be committed before validation completed;
- analytic contact order, iteration, corner handling, friction, positional
  correction, restitution, and event reduction diverged from the simulator;
- drag/restitution labels could use unobserved baselines, the wrong collision
  object, or an unidentifiable member-specific pair coefficient.

The repaired trainer now advances an optimizer update only with explicit
causal trajectory/state/parameter support or supported persistent fast-ROI
slots. Unsupported deterministic draws are counted, retry-bounded, and do not
advance optimizer state. Fast ROI masks follow actual evidence; valid empty
crops train only negative existence/visibility, all eligible confident
outputs enter precision, adjacent frames exercise the cache, and global/fast
losses are normalized independently. Absolute and reference-relative coverage
floors reject collapsed candidates; later support collapse restores the
verified incumbent and resets optimizer state.

Randomized three-sphere simulator/model contact differential testing now has
maximum position disagreement `5.96e-08 m` and velocity disagreement
`1.79e-07 m/s`. Sleep onset is intentionally not mirrored by that one-step
test because the simulator requires a sustained-contact counter.

An exact hard-window gradient decomposition then exposed a separate stability
problem: `dynamics.interactions` contributed norm `85.7563` of the total
`85.8882`, dominated by the final edge-network layer. Whole-model clipping
alone would scale every other gradient by about `0.0233`. The v3 protocol
therefore clips the learned interaction subsystem to `1.0` before applying the
whole-model `2.0` clip and logs raw subsystem/total, intermediate, and final
norms and coefficients. This retains the forward interaction capacity and
does not hardcode a physics law.

The first hierarchical-clipping smoke was:

```text
runs/20260802-110951-convergence-v3-hierarchical-clip-smoke/
completed updates: 4 (2 paired RGB, 2 persistent causal)
episode draws: 8
elapsed: 213.0543 s
devices: paired RGB MPS/CPU hybrid; causal CPU
oracle runtime input: false
causal trajectory support: 115 then 92
causal fast-slot support: 24 then 32
unsupported retries: 0
```

The first causal update had raw/pre-global/final norms
`4.599 / 4.599 / 2.0`. The hard second update had raw interaction/locally
applied/pre-global/final norms
`85.7563 / 1.0 / 4.8616 / 2.0`, with local coefficient `0.011661` and global
coefficient `0.411388`. All four optimizer updates were finite and genuinely
supported.

The smoke retained the imported safe rollout incumbent. Its deliberately tiny
two-episode validation measured selection score `0.567704`, current position
RMSE `0.839461 m`, velocity RMSE `1.709358 m/s`, target coverage `0.38`,
prediction precision `0.429379`, and horizon RMSE
`0.764942/0.580586/0.459451/0.503248/0.621400 m`. The two-update measurement
candidate had runtime birth recall/precision/F1
`0.20/0.6667/0.3077` and fast-ROI target coverage/precision/F1
`0.2667/1.0/0.4211`. These numbers only prove wiring and guardrails; they are
too small for an accuracy comparison.

The later final-tree audit also closed a broader selector hole: pooled metrics
could improve while an entire configured scenario had no complete physical
support. Rollout selector/checkpoint version `5.0` and validation protocol
version `8` now persist every declared scenario slice, require current and all
horizon support, and apply broad non-regression plus causal coverage floors
inside each slice. The configuration rejects a validation budget smaller than
the unique balanced scenario list, duplicate scenario entries, and invalid
negative/non-integral RGB phase boundaries. Valid unmapped persistent ROI queries also
now train negative existence/visibility and enter confidence precision; they
cannot disappear from supervision merely because simulator matching found no
target identity.

The exact final-tree host smoke is:

```text
runs/20260802-121629-convergence-v3-final-audit-smoke/
completed updates: 4 (2 paired RGB, 2 persistent causal)
episode draws: 8
elapsed: 244.8993 s
devices: RGB backbone/ROI MPS, proposal transformer CPU, causal runtime CPU
oracle runtime input: false
causal trajectory support: 122 then 161
causal fast-slot support: 32 then 38
unsupported retries: 0
selector/protocol: 5.0 / 8
```

All four updates were finite. Their raw/applied gradient norms were
`5.8580/2.0`, `4.3546/2.0`, `3.6477/2.0`, and `3.9386/2.0`; the final causal
interaction norm was `0.9617`, below its local `1.0` cap. The fixed
`reference_pairs` slice had complete support at initialization, handoff, and
terminal validation. After only two causal updates, the pooled score improved
`0.558737 → 0.548741`, but current coverage fell `0.350 → 0.315` and
0.1-second forecast coverage fell `0.800 → 0.700`. The selector correctly kept
the imported incumbent and recorded both pooled and scenario-specific
rejections. This is evidence that the new guard works, not an accuracy gain or
convergence result.

The new full profile is `configs/sustained_accuracy_mps_v3.yaml`: one shared
eight-scenario model, 8,192 paired RGB updates, 8,192 supported causal updates,
40-frame episodes, batch two, MPS measurement/CPU causal placement, global
clip `2.0`, interaction clip `1.0`, and the existing broad rollout guardrails.

A clean-source medium qualification was launched after commit
`c0acf1673819b1c3a892722d0a40c7bd085c5ea2` was pushed to `origin/main`:

```text
run: runs/20260802-123714-v3-medium-qualification/
LaunchAgent: com.polceanum.orpheus.v3-medium-20260802-123714
stdout: /private/tmp/20260802-123714-v3-medium-qualification.stdout.log
stderr: /private/tmp/20260802-123714-v3-medium-qualification.stderr.log
updates: 3,072 total = 1,024 paired RGB + 2,048 supported causal
episode draws: 6,144 across the unique balanced eight-scenario manifest
validation: 32 fixed episodes, eight anchors, every 512 updates
checkpoint cadence: 64 updates
devices: MPS paired RGB / CPU causal
```

At launch verification, launchd reported the job running as PID `55851`; host
process inspection showed active computation at approximately `525%` CPU,
MPS was built/available, run metadata recorded clean source and RGB-only
runtime, and both error/output logs were empty while the initial fixed
reference validation was in progress. This is launch-health evidence only.
The qualification outcome and four causal validation points remain pending;
no v3 convergence or promotion is claimed.

## 2026-08-01 — complete convergence-integrity audit and corrected v2 path

The legacy campaign
`runs/20260730-192625-scaled-sustained-e2e-v1/` was manually superseded at
logged step `9400`; its trainer, supervisor, and launchd KeepAlive job were
stopped without deleting any artifacts. It is not a converged result. Fixed
16-episode validation improved loss `9.9637 → 8.5636`, velocity
`1.3672 → 1.2348 m/s`, and collision F1 `0.1664 → 0.2092`, while one-second
RMSE regressed `0.9686 → 0.9980 m` and forecast-calibration coverage regressed.
Only two causal validations existed.

The audit proved that the step-8192 perception candidate was discarded from
the mutable training path after correctly failing only the velocity deployment
guard. It changed `79/84` global RGB tensors and scored `0.725038` versus the
step-zero `0.860012`, including one-second RMSE
`0.968568 → 0.647654 m`; later causal checkpoints restored all 84 global RGB
tensors exactly to step zero. The trainer now keeps the old safe incumbent for
deployment but continues causal optimisation from the finite stronger
candidate.

Additional convergence-affecting fixes now implemented are:

- absolute-step deterministic sampling and exact resume compatibility, with
  MPS RNG and immutable process-start Git provenance;
- a strict resume semantic diff before any run metadata is overwritten;
- fixed-denominator axis/horizon loss, pair-collision-prioritized windows,
  mature/cold forecast separation, scene-wide deterministic censoring after
  unseen actuation, and explicit forecast NLL;
- distinct position/velocity correction objectives and frozen disconnected
  heads;
- exact validation counts, per-axis/scenario/seed attribution, structured
  rejection reasons, and bounded deterministic trend-validation anchors;
- unit-correct pinhole projection of world covariance into association
  measurement coordinates;
- filter-only miss uncertainty growth and conservative position-quality gates;
- independent render and physical RNG streams, stable low-speed ground
  contact, true glancing offsets, compositional OOD ranges, and identical
  simulator/model pair-restitution combination;
- appearance supervision for RGB association embeddings and removal of frozen
  global losses from the optimized causal objective;
- lifecycle-qualified assignment before Hungarian matching, including
  confident false-positive accounting on target-empty frames;
- recoverable terminal validation, strict last-checkpoint-only in-place resume,
  and CPU checkpoint deserialization so hybrid optimizer ownership and
  accelerator memory remain correct;
- detached covariance-linearization coordinates, preventing calibration or
  filter-covariance objectives from leaking into RGB position/depth heads;
- a narrowly scoped PyTorch 2.10 workaround: the RGB backbone and ROI path run
  on MPS, while the small global proposal transformer runs on CPU through
  differentiable copies because the exact finite 64x64 MPS feature batch
  reproducibly generated NaN matrix-weight gradients.

The replacement profile is
`configs/sustained_accuracy_mps_v2.yaml`: 40 frames, batch two, 16,384 unique
training episodes, 8,192 RGB plus 8,192 causal updates, 32 balanced trend
episodes, and a deterministic bounded rollout-anchor manifest. The profile
retains the eight shared scenarios and one shared checkpoint.

Environment and verification:

```text
conda environment: orpheus
Python: 3.10.20
PyTorch: 2.10.0
MPS built/available outside sandbox: true/true
CPU suite: 500 passed, 6 MPS-only skips in 139.38 s
host accelerator regression set: 17 passed in 9.26 s
focused final selector/checkpoint suites: 126 passed
Ruff check: passed
Ruff format check: passed
compileall: passed for entry points, scripts, world_model, and tests
git diff --check: passed
```

The final bounded-anchor smoke is
`runs/20260801-231521-audit-v2-final-verified-smoke/`. It completed one real
RGB-pretrain update and two persistent closed-loop updates in `109.1528 s`.
The RGB update had finite loss `1.201544` and pre-clip gradient norm `8.488681`;
the two causal updates had finite losses `8.742573` and `2.106336` with
pre-clip norms `3.482935` and `2.472007`. MPS executed the backbone/ROI path,
CPU executed the proposal transformer, and the causal phase ran on CPU.

The imported broad incumbent on the deliberately tiny two-episode validation
manifest remains selected: score `0.768465`, position `1.031542 m`, velocity
`1.644270 m/s`, and one-second RMSE `0.767462 m`. The mutable handoff candidate
scored `0.383917` with one-second RMSE `0.456663 m`, but was correctly rejected
for calibration, x-axis, y-at-0.75-second, and forecast-coverage guardrails.
This is strong wiring/selection evidence, not an accuracy comparison: two
validation episodes and three optimizer updates cannot establish convergence.

`last.pt` is step three, records `final_validation_completed=1`, and every
model and AdamW tensor is finite. An exact completed-run resume performed zero
updates and preserved both durable files byte-for-byte:

```text
last.pt SHA-256:
  5e7196a5a0dbe6ff9d40b7ff6b031c1cd4636f1ec815473f0a568abc85d013d6
train_summary.json SHA-256:
  157be114a86327102bfb5aaceda0af2a13016994796c8ffb5751d5b6aa814f71
```

The no-op CLI result truthfully reports
`no_op_exact_resume=true` and `optimizer_updates_this_invocation=0`; those
ephemeral inspection fields are not written over the original campaign
summary.

Exact final verification commands:

```bash
PYTHONPATH=. conda run -n orpheus pytest -q

PYTHONPATH=. conda run -n orpheus pytest -q \
  tests/integration/test_rgb_measurements.py \
  tests/unit/test_association.py::test_association_transfers_cost_to_cpu_without_mps_float64 \
  tests/unit/test_evaluation_parameter_update_metrics.py::test_directional_parameter_metrics_transfer_before_float64_accumulation \
  tests/unit/test_modal_dynamics.py::test_modal_device_when_available

PYTHONPATH=. conda run -n orpheus ruff check .
PYTHONPATH=. conda run -n orpheus ruff format --check .
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-pycache-final PYTHONPATH=. \
  conda run -n orpheus python -m compileall \
  train.py evaluate.py demo.py scripts world_model tests
git diff --check

PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/sustained_accuracy_mps_v3.yaml \
  --initialize-from \
    runs/20260730-192625-scaled-sustained-e2e-v1/checkpoints/best_measurement.pt \
  --run-name convergence-v3-final-audit-smoke \
  --device mps \
  --set 'simulator.scenario_mixture=[reference_pairs]' \
  --set training.steps=4 \
  --set training.rgb_pretrain_steps=2 \
  --set training.train_episodes=8 \
  --set training.validation_episodes=2 \
  --set training.batch_size=2 \
  --set training.eval_every=2 \
  --set training.checkpoint_every=1 \
  --set training.log_every=1 \
  --set training.num_workers=0 \
  --set training.validation_rollout_anchors_per_episode=2 \
  --set training.measurement_validation_frames=2
```

The accelerator tests and training command above ran outside the restricted
process sandbox, where host MPS is available. The older v2 no-op continuation
evidence below remains valid for that historical checkpoint, but no v2
artifact is compatible with the stricter v3 selector/contact protocol.

Earlier bounded smokes remain failure/audit artifacts, not current evidence.
In particular, `runs/20260801-223113-audit-v2-final-smoke/` stopped before its
first update after exposing the data-dependent MPS gradient fault.

### Corrected sustained v2 campaign was stopped and superseded

The corrected campaign launched from clean committed source
`df98f637b39607db5ede78dfeafab9ca61ef7d50` at
`2026-08-01T23:23:46Z`:

```text
run: runs/20260801-232229-scaled-sustained-v2/
trainer LaunchAgent:
  com.polceanum.orpheus.sustained-v2-20260801-232229
trainer PID at verification: 9889
trainer stdout:
  /private/tmp/20260801-232229-scaled-sustained-v2.stdout.log
trainer stderr:
  /private/tmp/20260801-232229-scaled-sustained-v2.stderr.log
supervisor LaunchAgent:
  com.polceanum.orpheus.convergence-v2-20260801-232229
supervisor PID at verification: 9980
supervisor stdout:
  /private/tmp/20260801-232229-convergence-v2.stdout.log
supervisor stderr:
  /private/tmp/20260801-232229-convergence-v2.stderr.log
```

`run_metadata.json` records dirty `false`, runtime-source fingerprint
`d6039706f3fd97296cd4f2ff1bf84b4cfd4ec5d9124fdfd597d665e23b11c132`,
MPS built/available `true/true`, measurement device `mps`, closed-loop device
`cpu`, RGB runtime, and oracle disabled. Both submitted jobs were verified
`running` at launch. The supervisor persisted `supervisor_started` and
`waiting_for_segment` events for the 16,384-step minimum, with 4,096-step
extensions and a 24,576-step hard limit.

This block is historical launch evidence. On 2 August both LaunchAgents were
booted out after the supported-gradient audit; the trainer stopped at logged
step `9576` and must not be resumed under the changed v3 protocol. Its retained
`metrics.jsonl` is an audit control, not convergence evidence.

## 2026-07-31 — sustained-loss stability and horizon-objective audit

The active MPS trainer and convergence supervisor remain healthy as PIDs
`37360` and `41396`. At the audit cutoff, training had reached step `8776`
(`584/4096` causal updates); it subsequently logged at least step `8792`. The
latest durable checkpoint at the cutoff was
`runs/20260730-192625-scaled-sustained-e2e-v1/checkpoints/last.pt` at step
`8768`. All logged values, all 177 model tensors, and all AdamW moments were
finite. Every one of the 87 causal optimizer states reported step `576`,
exactly matching `8768 - 8192`; learning rate remained `5e-6`.

The apparent console instability is primarily heterogeneous batch-one noise,
not numerical divergence. Across 73 logged causal rows, total loss had median
`9.325`, mean `9.472`, 95th percentile `19.00`, and maximum `29.91`.
Measurement supervision correlated `0.969` with total loss and dominated hard
low-match ROI windows. Logged `gradient_norm` was the value returned before
clipping: 71/73 rows exceeded `1.0`, but every applied update was clipped to
the configured norm. Block loss means declined overall rather than exploding.

The first causal validation at step `8704` improved the fixed reference's
weighted score by `0.543%`, current position RMSE by `1.445%`, velocity RMSE
by `2.816%`, gated target coverage by `10.865%`, precision by `11.521%`, and
collision F1 by `5.042%`. Position RMSE improved at 0.10/0.25/0.50/0.75
seconds by `1.93% / 3.40% / 2.91% / 0.77%`. It was correctly rejected because
1.00-second RMSE regressed `2.445%`, only `0.004311 m` beyond the declared 2%
guardrail. The safe imported incumbent remains selected. One validation after
only 512 causal updates is neither convergence nor evidence to interrupt the
declared 4,096-window minimum.

The audit did find one real objective bug: the configured x/y/z rollout losses
were normalized over only the horizons available in each sampled window,
whereas the aggregate position loss used the fixed total configured horizon
weight. Short-only windows therefore received full axis-loss scale and
underweighted the rare 1.00-second target; only 26/73 logged windows exposed a
one-second target. The corrected implementation now emits per-axis
per-horizon terms, uses the fixed configured denominator, and can sample
collision and maximum-horizon intents jointly. When a late collision cannot
fit in a maximum-horizon-capable window, the sampled long-horizon example is
retained instead of being silently lost.

The code also distinguishes `gradient_norm_pre_clip` from
`gradient_norm_applied` and records the exact `gradient_clip_coefficient`.
Four objectively disconnected tensors—the ROI event head and identifier
variance head weights/biases—remain checkpoint-compatible but are frozen in
restricted closed-loop scopes until their outputs receive explicit
corrector/calibration objectives.

The already-running campaign deliberately retains both legacy controls as
`false` in `configs/sustained_accuracy_mps.yaml`. Its Python process already
loaded those semantics, and any automatic in-place extension must remain
comparable. New profiles default to the corrected behavior. After the active
minimum completes, the next timestamped causal campaign should initialize from
a validation-proven candidate, set `rgb_pretrain_steps=0`, enable both controls,
and complete a new 4,096-window balanced pass before any promotion claim.

Focused verification completed while the MPS trainer continued:

```bash
conda run -n orpheus ruff check \
  world_model/training/loop.py world_model/training/trainer.py \
  world_model/utils/config.py tests/unit/test_training_schedule.py \
  tests/unit/test_config.py
conda run -n orpheus pytest -q \
  tests/unit/test_training_schedule.py \
  tests/unit/test_fast_roi_supervision.py \
  tests/unit/test_config.py \
  tests/integration/test_cli_smoke.py
git diff --check
```

Ruff and `git diff --check` passed. The final combined
config/schedule/fast-ROI/CLI subset reported `108 passed in 53.90 s`.
Compileall passed, and the corrected causal-only dry run resolved 4,096
batch-one draws over all eight scenarios with RGB-only runtime.

```bash
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-stability-pycache PYTHONPATH=. \
  conda run -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
PYTHONPATH=. conda run -n orpheus python train.py \
  --config configs/sustained_accuracy_mps.yaml --dry-run --device cpu \
  --set training.rgb_pretrain_steps=0 --set training.steps=4096 \
  --set training.normalize_rollout_axes_over_configured_horizons=true \
  --set training.joint_collision_long_horizon_sampling=true
PYTHONPATH=. conda run -n orpheus pytest
PYTHONPATH=. conda run -n orpheus pytest -q \
  tests/integration/test_rgb_measurements.py \
  tests/unit/test_association.py \
  tests/unit/test_evaluation_parameter_update_metrics.py \
  tests/unit/test_modal_dynamics.py
```

The final sandboxed full suite reported
`318 passed, 4 skipped in 118.34 s`; all four skips were the expected
MPS-availability conditionals. The same four files then passed with direct MPS
access (`21 passed in 3.09 s`). The active trainer and supervisor remained the
only training/supervision processes and were still healthy after verification.
No corrected-objective training metric exists yet, so this change is not
claimed as an accuracy promotion.

## 2026-07-30 — sustained shared-model campaign preflight

The complete artifact audit is
[`project/ACCURACY_AUDIT.md`](ACCURACY_AUDIT.md). It separates three contexts:

- the 156k-parameter checkpoint is the only model with a completed balanced
  16-episode test over all eight scenario families; it beats constant velocity
  at every horizon but remains at `0.200430 m` current position RMSE,
  `0.968753 m/s` velocity RMSE, `0.364040 m` one-second RMSE, and `0.320388`
  collision F1;
- fixed physical scale, the historical cadence-three configuration (actual
  cadence four), and the point/scale observer improve scaled position, but the
  current 1.90M-parameter weights
  received only 1,024 measurement episode draws and no accepted causal
  training;
- the later change-point, outgoing, lateral, and gravity heads improve their
  local cached objectives but regress velocity, detection, events, identity,
  or longer recursive horizons online. They remain disabled.

The next experiment is therefore one shared scaled model rather than another
isolated head. `configs/sustained_accuracy_mps.yaml` declares 8,192 measurement
updates followed by 4,096 independent causal windows across the eight balanced
scenario families: two complete measurement passes, one nominal causal pass,
and about 512 causal windows per scenario. The imported runtime is
`runs/20260729-084712-scaled-point-scale-trajectory-v1/checkpoints/runtime_ablation.pt`.

Training now limits the expensive recursive forecast to one earliest eligible
anchor per four-frame TBPTT window while still ingesting and supervising every
frame. The sampled collision/long-horizon windows cover the complete declared
horizon set. Posterior rollouts are shared by forecast and correction losses.
Full validation retains every eligible posterior anchor but skips the
redundant prior future rollout.

Checkpoint selection is physical and pooled over the complete validation
manifest. The primary score is horizon-weighted position RMSE. A candidate
must also remain within declared tolerances for current position and velocity,
every horizon, 0.5 m distance-gated recall/precision and identity, forecast
lifecycle coverage, collision F1, and nominal-90% calibration. These guards
apply against both the moving incumbent and the fixed imported reference.
Every validation candidate is numbered. Exact simulator/model/runtime/metric/
batch/seed semantics and model tensor hashes bind metrics to real incumbent
and reference weights on resume. Causal AdamW moments start fresh at the phase
handoff.

### Active sustained run

The complete campaign started at `2026-07-30T19:26:55Z` from committed
revision `da558d5`:

```bash
launchctl submit \
  -l com.polceanum.orpheus.sustained-20260730-192625 \
  -o /private/tmp/20260730-192625-scaled-sustained-e2e-v1.stdout.log \
  -e /private/tmp/20260730-192625-scaled-sustained-e2e-v1.stderr.log \
  -- /usr/bin/caffeinate -dimsu \
  /usr/bin/env PYTHONPATH=/Users/mike/Work/world.model \
  /usr/local/Caskroom/miniforge/base/envs/orpheus/bin/python \
  /Users/mike/Work/world.model/train.py \
  --config /Users/mike/Work/world.model/configs/sustained_accuracy_mps.yaml \
  --initialize-from \
  /Users/mike/Work/world.model/runs/20260729-084712-scaled-point-scale-trajectory-v1/checkpoints/runtime_ablation.pt \
  --run-name 20260730-192625-scaled-sustained-e2e-v1 \
  --device mps
```

`launchctl print` reported the job in the running state with trainer PID
`37360`; `run_metadata.json` independently records PyTorch `2.10.0`, device
`mps`, MPS built/available `true`, `float32`, RGB runtime, and oracle disabled.
The active artifact root is
`runs/20260730-192625-scaled-sustained-e2e-v1/`. The trainer first evaluates
the fixed 16-episode reference manifest, which is expected to take roughly
60–90 minutes before emitting the first validation metrics. Absence of early
metrics is therefore not interpreted as a completed or failed run.

The autonomous convergence supervisor was launched at
`2026-07-30T20:44:38Z` from committed revision `3c03e5a`. LaunchAgent
`com.polceanum.orpheus.convergence-20260730-192625` reported state `running`
with supervisor PID `41396`, while the one and only trainer remained PID
`37360`. It monitors that trainer without modifying or interrupting it, waits
for a tensor/protocol-verified 12,288-step completion, then removes the
initial KeepAlive job before sequentially resuming `last.pt` in complete
4,096-update blocks:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python \
  scripts/supervise_convergence.py \
  --config configs/sustained_accuracy_mps.yaml \
  --run runs/20260730-192625-scaled-sustained-e2e-v1 \
  --device mps \
  --initial-trainer-pid 37360 \
  --initial-launchctl-label \
    com.polceanum.orpheus.sustained-20260730-192625 \
  --maximum-total-steps 24576
```

The supervisor calls a plateau only after four exact consecutive 512-step
validations accept no candidate and raw primary-score improvement remains
below 1%. A recent guardrail-safe gain of at least 1%, missing evidence, or
contradictory evidence requests another complete block. At the hard limit, a
demonstrated plateau remains a plateau; otherwise the result is truthfully
`limit_hit`. The script persists events/state/report files, reattaches to an
exact in-place extension after restart, prevents overlapping trainers, and
records an initial-PID or child failure without an infinite automatic retry.
Its persistent event log is
`runs/20260730-192625-scaled-sustained-e2e-v1/convergence_supervisor.jsonl`;
the first two verified events are `supervisor_started` and
`waiting_for_segment`. Standard output/error are
`/private/tmp/20260730-192625-convergence-supervisor.stdout.log` and
`/private/tmp/20260730-192625-convergence-supervisor.stderr.log`. The latter
was empty after launch.

Focused verification after this implementation:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff format \
  world_model/training/convergence.py scripts/supervise_convergence.py \
  tests/unit/test_convergence_supervisor.py
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff check \
  world_model/training/convergence.py scripts/supervise_convergence.py \
  tests/unit/test_convergence_supervisor.py
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest \
  tests/unit/test_convergence_supervisor.py \
  tests/integration/test_trainer_checkpoint_integrity.py -q
```

Result: Ruff passed and `17 passed in 3.54 s`. The complete repository suite
then reported `304 passed, 4 skipped in 108.06 s`; all skips were the expected
sandbox-visible MPS conditionals, while the real trainer continued on direct
MPS.

### Environment and validation

Direct hardware inspection on 2026-07-30 reported Python `3.10.20`, PyTorch
`2.10.0`, MPS built `true`, MPS available `true`, and a successful tensor
allocation on `mps:0`.

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest \
  tests/unit/test_config.py \
  tests/unit/test_training_schedule.py \
  tests/unit/test_fast_roi_supervision.py \
  tests/unit/test_event_window_scoring.py \
  tests/integration/test_trainer_checkpoint_integrity.py \
  tests/integration/test_cli_smoke.py -q
```

Result: `101 passed in 36.50 s`.

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff format .
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff check .
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-sustained-pycache PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest
git diff --check
```

Ruff left all 168 files unchanged and passed; compileall and `git diff --check`
passed. The sandboxed full suite reported `291 passed, 4 skipped in 81.07 s`.
The four MPS-conditional files then ran with direct device access and reported
`21 passed in 2.12 s`.

The real campaign command also passed `train.py --dry-run --device mps`,
resolving 12,288 steps, 4,096 training episodes, 16 validation episodes,
batch one, all eight scenario families, and RGB-only runtime.

```bash
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-convergence-pycache PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python -m compileall -q \
  world_model/training/convergence.py scripts/supervise_convergence.py \
  tests/unit/test_convergence_supervisor.py
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest \
  tests/integration/test_cli_smoke.py -q
git diff --check
```

The new supervisor files passed `compileall`; the existing CLI smoke test
reported `1 passed in 30.98 s`, and `git diff --check` passed.

### Representative scaled MPS smoke

The following wiring/throughput check ran the actual imported incumbent,
full-anchor validation, and eight one-anchor causal backward updates:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/sustained_accuracy_mps.yaml \
  --initialize-from \
    runs/20260729-084712-scaled-point-scale-trajectory-v1/checkpoints/runtime_ablation.pt \
  --run-name sustained-mps-smoke8-v1 \
  --device mps \
  --set training.steps=8 \
  --set training.rgb_pretrain_steps=0 \
  --set training.train_episodes=16 \
  --set training.validation_episodes=1 \
  --set training.num_workers=0 \
  --set training.eval_every=8 \
  --set training.checkpoint_every=4 \
  --set training.log_every=1 \
  --set evaluation.episodes=1
```

Artifact:
`runs/20260730-185438-sustained-mps-smoke8-v1/`. It contains `last.pt`,
verified `best_rollout.pt` and `reference_rollout.pt`, and numbered step-zero
and step-eight validation checkpoints. Total wall time was `1510.29 s`.
Initial and final one-episode validations took `335.93 s` and `399.22 s`.
The eight causal updates averaged `96.64 s` (`78.09–112.56 s`), versus about
`242 s/update` in the previous fixed-cadence causal run.

On this intentionally non-generalizable one-episode smoke manifest, step eight
passed the predeclared guards:

| metric | imported step 0 | step 8 |
| --- | ---: | ---: |
| weighted horizon score | `0.689518` | `0.689004` |
| current position RMSE | `0.743342 m` | `0.742693 m` |
| current velocity RMSE | `1.365972 m/s` | `1.365994 m/s` |
| gated recall / precision | `0.402778 / 0.381579` | unchanged |
| collision F1 / ID-switch rate | `0.071429 / 0` | unchanged |
| nominal-90% position coverage | `0.547101` | `0.565217` |

This is a mechanical and timing result, not an accuracy promotion. Eight
updates and one validation episode are far below the declared convergence and
scenario-coverage minimum. The measured throughput predicts roughly five days
for the complete campaign, within the predeclared three-to-seven-day range.
Do not judge causal convergence before 2,048 causal updates; complete all
4,096, and extend only if the best safe checkpoint is still materially
improving in the final 1,024 updates. Final promotion requires at least 64
fresh balanced validation episodes with per-scenario results before any
reserved test evaluation.

## 2026-07-30 — on-policy gravity-axis correction rejected

The dominant y/gravity velocity error is now addressed by a separate,
disabled-by-default intervention path. The RGB temporal observer exports a
21-value causal feature vector: acceleration-compensated gravity-axis
kinematics, camera-lateral context, contact probability, the exact pre-filter
gravity prior and variance, and an acceleration-aware RGB slope residual and
variance. A tiny axis-local MLP proposes only a gravity-axis velocity delta and
soft measurement gain. Non-gravity means are preserved and their measurement
variance is explicitly unobserved, preventing an early implementation from
silently contracting x covariance and changing association.

The first balanced MPS collection produced 543 training and 398 disjoint
validation windows. A one-unit regularized head reduced held-out post-filter
gravity residual RMSE `2.222436 → 1.771491 m/s`, but repeated runtime
application shifted its own input distribution. After correcting the
gravity-only covariance contract, it improved seed `100017` current position
and 0.1-second forecast but regressed y velocity.

One dataset-aggregation pass then collected 498 training and 373 validation
windows while rolling out the first head. The on-policy refit reduced its
held-out prior/post-filter RMSE `2.113796 → 1.854939 m/s`. On seed `100017`,
it improved current position `0.702313 → 0.686879 m`, total velocity
`1.148770 → 1.108480 m/s`, y velocity `1.941731 → 1.882283 m/s`, and
0.1-second forecast `0.715284 → 0.712301 m`.

The protocol-matched two-episode block rejected promotion. Current y position
improved `0.335656 → 0.291913 m` (`13.03%`), current position improved
`0.684258 → 0.669713 m`, 0.1-second forecast improved
`0.698125 → 0.690724 m`, and collision-conditioned 0.1-second forecast
improved `0.216634 → 0.187418 m`. However, y velocity regressed
`1.902265 → 1.979012 m/s`, detection recall fell
`0.687500 → 0.586806`, collision F1 fell `0.271186 → 0.218750`, and overall
forecast RMSE regressed at 0.25/0.50/0.75/1.00 seconds by
`1.03%/3.55%/1.95%/1.24%`. The scaled default and accepted checkpoint remain
unchanged.

The next accuracy target is end-to-end intervention training through the
persistent association/ROI loop and recursive horizons, with detection
coverage and identity stability in the selection objective. Per-update
post-filter supervision, even with one on-policy aggregation pass, is
insufficient.

Primary artifacts:

- baseline-prior aligned caches and initial fit:
  `runs/20260730-101936-rgb-gravity-intervention-aligned-8x8-v1/`;
- stable regularized baseline-prior fit:
  `runs/20260730-103853-rgb-gravity-intervention-regularized-v2/`;
- on-policy collection/refit:
  `runs/20260730-105512-rgb-gravity-intervention-on-policy-8x8-v4/`;
- fast on-policy report:
  `runs/20260730-105512-rgb-gravity-intervention-on-policy-8x8-v4/evaluation/20260730-112140-gravity-fast-offset17/report.md`;
- protocol-matched multihorizon report:
  `runs/20260730-105512-rgb-gravity-intervention-on-policy-8x8-v4/evaluation/20260730-113746-gravity-select2-offset16-frames48/report.md`.

Validation used:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff format .
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff check .
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest
```

Ruff formatted one file and passed all 167 files. The sandboxed suite reported
`267 passed, 4 skipped in 99.18 s`; all four hardware-conditional tests passed
directly on Apple MPS (`4 passed in 3.04 s`). The collection, on-policy
aggregation, and runtime evaluations used Apple MPS; cached tiny-head sweeps
used CPU.

## 2026-07-30 — intervention-aware lateral correction rejected

The RGB trajectory path now exports the exact pre-direct-correction velocity,
variance, confidence, camera-lateral basis, and a 19-value causal feature
vector. This fixes a supervision leak in the earlier collector, which read the
belief after the ordinary temporal correction. A disabled-by-default,
one-hidden-layer intervention head proposes a bounded lateral measurement and
a continuous soft gain. The gain maps to measurement variance and the ordinary
analytic filter performs the actual correction; there is no hard event gate,
history re-encoding, online optimizer step, or simulator runtime input.
Feature standardization is folded into the first-layer coefficients. A
configurable gain power can make low-confidence proposals abstain more
strongly.

Eight MPS collection episodes produced 543 aligned training windows; eight
disjoint episodes produced 398 validation windows. The initial 12-unit fit
overfit (`0.648080 → 0.702765 m/s` held-out RMSE). A regularized one-unit fit
improved held-out post-filter lateral RMSE `0.648080 → 0.497431 m/s` and MAE
`0.341558 → 0.287335 m/s`.

That offline gain did not survive the primary paired recursive test. On seed
`100017`, it improved x-velocity `0.421218 → 0.352271 m/s`, total velocity
`1.148770 → 1.115892 m/s`, and collision-conditioned 0.1-second position
`0.123723 → 0.119199 m`. On the protocol-matched two-episode
`100016–100017` block, current x-position improved
`0.544812 → 0.538721 m` and collision-conditioned 0.1-second RMSE improved
`0.216634 → 0.189408 m`, but x-velocity regressed
`0.568277 → 0.576981 m/s`; x forecast regression grew from `0.41%` at
0.1 seconds to `5.47%` at 1.0 second. Squaring the soft gain in the variance
mapping reduced offline aggressiveness but failed the fast runtime gate
(current position `0.702313 → 0.704354 m`). Both candidates are rejected and
the scaled default remains disabled.

The next concrete accuracy target is the gravity-axis state/dynamics path:
held-out y-velocity RMSE is about `1.9 m/s`, far larger than lateral or
camera-depth velocity error. It should receive an axis-local, acceleration-
aware learned correction with joint collision context and recursive
multihorizon supervision, without perturbing the better axes.

Primary artifacts:

- aligned MPS caches:
  `runs/20260730-090236-rgb-lateral-intervention-aligned-8x8-v1/`;
- selected regularized offline fit:
  `runs/20260730-092214-rgb-lateral-intervention-regularized-v2/`;
- protocol-matched paired report:
  `runs/20260730-092214-rgb-lateral-intervention-regularized-v2/evaluation/20260730-094915-lateral-select2-offset16-frames48/report.md`;
- uncertainty-tightened rejected fit and fast report:
  `runs/20260730-095159-rgb-lateral-intervention-soft-square-v3/` and
  `evaluation/20260730-095531-lateral-fast-offset17/report.md`.

Validation:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff format .
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff check .
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest
```

Ruff formatted one file and passed all 167 files. The sandboxed full suite
reported `266 passed, 4 skipped in 79.80 s`; the four hardware-conditional
tests then passed directly on Apple MPS (`4 passed in 1.67 s`). The focused
filter/RGB/config/checkpoint suite reported `79 passed in 9.31 s`. The MPS
collection and three runtime evaluations used Apple MPS; the small cached MLP
fits ran on CPU.

## 2026-07-30 — bounded outgoing-velocity proposal rejected

The learned RGB event gate remains available but disabled; it was not removed
because it has no effect on the default runtime and preserves a reproducible
exact-timestamp diagnostic/data path. The event-fitting workflow now also
caches the belief's outgoing gravity-velocity prior, aligned simulator-only
supervision target, and target delta. An optional eight-hidden-unit proposal
uses the nine causal gate features, scaled prior velocity, and gate probability
to emit a bounded scalar delta and calibrated variance. Runtime inference is a
single tiny MLP with no history re-encoding or online weight update. A
configurable six-sample refractory interval prevents immediate feedback
retriggering.

An initial runtime implementation waited for a later post-reset history and
therefore recomputed the proposal from a window different from the supervised
trigger. This is fixed and unit-tested: a proposal is consumed once on its
exact causal trigger frame, while later post-event samples use the ordinary
acceleration-aware estimator.

The eight-scenario MPS collection produced 543 training windows (197 positive)
and 398 disjoint validation windows (146 positive). A positive-only fit reduced
positive-window validation RMSE from `2.795367` to `1.194373 m/s`, but it did
not model false runtime selections. The selected joint gate-focused,
`1.5 m/s`-bounded fit reduced all-window RMSE
`1.693066 → 1.548441 m/s` and gate-selected RMSE
`1.638817 → 1.537791 m/s`; MAE did not improve.

On fresh-validation seed `100017`, delayed application was rejected at current
position/velocity RMSE `0.702754 m` / `1.170360 m/s`. Correct immediate
application slightly improved current position `0.702313 → 0.701963 m` and
0.1-second forecast `0.715284 → 0.714966 m`, but velocity regressed
`1.148770 → 1.173099 m/s`. The scaled gate and proposal remain disabled, and
the accepted checkpoint is unchanged. The next concrete accuracy task is an
intervention-aware camera-lateral outgoing correction with learned
abstention/gain, trained against post-filter velocity and recursive forecast
effects rather than gravity-only event labels.

Primary artifacts:

- aligned feature/target caches and positive-only fit:
  `runs/20260730-074928-rgb-outgoing-proposal-aligned-8x8-v1/`;
- bounded joint fit:
  `runs/20260730-081923-rgb-outgoing-proposal-gate-focused-bounded-v4/`;
- delayed runtime report:
  `runs/20260730-081923-rgb-outgoing-proposal-gate-focused-bounded-v4/evaluation/20260730-082303-proposal-fast-offset17/report.md`;
- trigger-aligned runtime report:
  `runs/20260730-081923-rgb-outgoing-proposal-gate-focused-bounded-v4/evaluation/20260730-083325-proposal-immediate-fast-offset17/report.md`.

Collection/fitting and paired MPS evaluation used:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python \
  scripts/train_rgb_change_point_gate.py \
  --config configs/scaled_curriculum.yaml \
  --checkpoint runs/20260729-084712-scaled-point-scale-trajectory-v1/checkpoints/runtime_ablation.pt \
  --device mps --train-episodes 8 --validation-episodes 8 \
  --validation-seed-offset 256 --minimum-precision 0.7 \
  --gate-type mlp --hidden-features 8 --fit-steps 1500 \
  --fit-outgoing-proposal --proposal-hidden-features 8 \
  --proposal-fit-steps 2000 --set simulator.sequence_frames=32 \
  --output runs/rgb-outgoing-proposal-aligned-8x8-v1

PYTHONPATH=. conda run --no-capture-output -n orpheus python evaluate.py \
  --config runs/20260730-081923-rgb-outgoing-proposal-gate-focused-bounded-v4/config.resolved.yaml \
  --checkpoint runs/20260730-081923-rgb-outgoing-proposal-gate-focused-bounded-v4/checkpoints/change_point_gate.pt \
  --split validation --seed-protocol fresh_validation --seed-offset 17 \
  --device mps --set simulator.sequence_frames=48 \
  --set evaluation.episodes=1 --set 'evaluation.horizons_seconds=[0.1]' \
  --set 'training.horizon_weights=[1.0]' \
  --output runs/20260730-081923-rgb-outgoing-proposal-gate-focused-bounded-v4/evaluation/proposal-immediate-fast-offset17
```

Final validation:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff format .
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff check .
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest
```

Ruff left all 167 files unchanged and passed. The sandboxed full suite reported
`264 passed, 4 skipped in 75.23 s`; the four hardware-conditional tests then
passed directly on MPS (`4 passed in 1.63 s`). The focused temporal/gate/config/
checkpoint suite reported `77 passed in 5.15 s`.

```bash
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-outgoing-proposal-pycache \
  PYTHONPATH=. conda run --no-capture-output -n orpheus python \
  -m compileall -q world_model train.py evaluate.py demo.py scripts tests
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/scaled_curriculum.yaml --dry-run --device cpu
git diff --check
```

Compileall, the unchanged 48,000-draw/eight-scenario scaled dry run, and
`git diff --check` passed with Python `3.10.20` and PyTorch `2.10.0`. The
sandboxed dry run reported MPS compiled but unavailable; the paired evaluation
and direct device tests ran against the host MPS device.

## What works

- A deterministic 3-D RGB sphere simulator provides explicit timestamps,
  calibrated cameras, identities, physical state, depth-ordered
  rendering/occlusion, collisions/events, and disjoint seed splits.
- `WorldBelief` is the persistent, modality-independent runtime truth. Every
  ingest predicts a prior, projects expected measurements, associates,
  computes innovation, corrects the posterior, updates lifecycle/observable
  slow parameters, and revises arbitrary-time future rollouts.
- Dynamics combine analytic kinematics, bounded modal state, structured
  interactions, explicit event jumps, learned residuals, and uncertainty
  propagation.
- RGB includes intermittent learned global discovery, an optional transparent
  RGB-only structured disc-centre proposal, and ordinary residual ROI updates.
  Global structured discovery uses row-background subtraction, connected
  components, touching-disc peak splitting, and proposal alignment. Ordinary
  structured refinement samples only projected ROIs and never invokes the
  global SciPy/Hungarian path. The fast path remains active (12 of 16 demo
  updates), while its inverse-depth residual is reliability-gated at the
  analytic prior until a trained checkpoint proves positive held-out
  metric-space improvement.
- Fixed-dataset RGB pretraining now sweeps every frame for every loader batch,
  rather than coupling episode batches to only even or odd frames.
- Measurement validation spans configured frames across every validation
  episode and selects
  `best_measurement.pt` by calibrated backprojected world-position MAE rather
  than the summed, possibly negative Gaussian NLL objective.
- Structured forward centres preserve the raw detector/ROI centre as explicit
  gradient-bearing auxiliary evidence. A separate raw-centre target prevents
  the accurate image heuristic from hiding detector error during training.
- RGB supervision includes calibrated metric-space world-position Huber/NLL
  terms with explicit weights; closed-loop position and velocity terms are
  separately weighted.
- Closed-loop training restores the best localized perception checkpoint and
  applies a configurable 10x learning-rate reduction. Resume reapplies the
  correct phase learning rate after loading optimizer state.
- Closed-loop global discovery/backbone weights remain trainable only for a
  configurable adaptation window, then freeze while the fast ROI path,
  filter, dynamics, and identifier continue training. This prevents the
  localization drift observed during unrestricted continuation.
- Fast ROI outputs are supervised at every usable prior frame in persistent
  belief-slot order; they are not incorrectly rematched by their own current
  output values.
- Mid-episode TBPTT windows ingest their complete RGB prefix causally under
  `no_grad`, detach the resulting persistent state, and preferentially sample
  collision-bearing or maximum-horizon-capable spans. Closed-loop validation
  evaluates all configured episodes through their complete causal sequence.
- Per-horizon rollout and future-correction losses are averaged over all
  eligible anchors before configured horizon weights are applied. A fixed
  configured denominator prevents short tail windows from silently
  renormalising themselves into larger objectives. Checkpoint selection records
  the physical per-horizon validation losses and rejects legacy scores with
  incompatible aggregation semantics.
- Collision occurrence is aggregated over every internal physics substep in a
  rollout segment while persistent motion-mode logits remain instantaneous.
  Training and evaluation insert exact `[h-dt_obs, h]` query boundaries so
  frame event labels are never compared with a cumulative or arbitrary
  horizon segment.
- The correction objective retains the small specification-required sparsity
  regularizer and now also penalizes current/future posterior updates that fail
  to improve over a detached prior. Rare collision positives receive a
  bounded, explicitly configured BCE weight.
- Evaluation can derive a deterministic fresh-validation manifest from
  checkpoint provenance or accept an explicit `--seed-offset`, persist exact
  seeds/non-overlap status, report current and ordinary-correction velocity
  metrics, and compare model and baselines on identical future-collision
  masks.
- RGB has a separate bounded persistent-ID temporal position history with
  explicit timestamps and uncertainty. It can provide a cheap causal
  velocity-only correction without new weights or history re-encoding, but is
  disabled in public profiles because its current accuracy tradeoff fails the
  overall validation gate.
- The debug oracle is registered only when explicitly enabled. Every result
  below uses RGB plus known calibration; simulator state is used only for
  supervision, evaluation alignment, and explicitly labelled baselines.
- Demo world axes, panel margins, and manual legends are fixed across GIF
  frames. Every posterior forecast is retained at its original world-space
  anchor with recency-faded alpha, while the latest prior/posterior paths,
  Hungarian-matched endpoint connectors, and absolute errors remain explicit.
  Extra simulator frames supply scoring-only lookahead so every displayed
  frame uses the same requested forecast horizon.

## Environment

- Conda environment: `orpheus`
- Python: 3.10.20
- Process architecture: x86_64
- PyTorch: 2.10.0, installed build preserved unchanged
- MPS: compiled and available to the direct `conda run` processes used for the
  current training/evaluation; sandboxed subprocesses may still report it
  unavailable
- CUDA: unavailable
- Precision: float32

Direct MPS tests and evaluation are now part of the current evidence. A
sandboxed launcher can still skip hardware-conditional tests, so each command
below states whether it ran inside or outside that boundary.

## Accuracy-v4 promoted CPU evidence

This is the current result. Accuracy-v3 step 584 is the initialization and
paired baseline, not the promoted checkpoint. Older step-70/72/94 experiments
later in this document remain historical evidence.

### RGB-only structured-centre diagnostics

The optional synthetic-disc prior consumes RGB pixels only. Global discovery
uses row-median background subtraction, connected components,
distance-transform peaks for touching discs, and Hungarian alignment to
learned proposals. On fresh-validation seeds `100004–100019`, labels used only
for scoring found:

- `507 / 512` target centres matched;
- mean normalized centre error `0.0014439`;
- maximum normalized centre error `0.0304`.

A sample from the default/MPS-sized profile matched `57 / 57` visible targets
with mean/max normalized error `0.000855 / 0.004479`. The noisier `toy_hard`
and `cloud_single_gpu` profiles use the regression-tested foreground threshold
`0.08`, rather than the ordinary `0.04`.

The ordinary fast path does not repeat global discovery. It samples only the
projected object ROIs, estimates local foreground, and grows the component
nearest each prior centre. Its isolated RGB diagnostic matched `256 / 256`
visible targets with mean normalized centre error `0.0184`. The isolated CPU
microbenchmark was approximately `3.94 ms` for eight 20x20 ROIs. This is a
primitive benchmark, not the complete online-update latency reported below.

### Promoted checkpoint and closed-loop continuation

The promoted checkpoint is:

- path:
  `runs/accuracy-closed-structured-v4/checkpoints/best_rollout.pt`;
- step: `648`;
- SHA-256:
  `9b943f60128a2bd15298847d8c7de4dd3166646f3644720a3149155e57d85bcd`.

It is a controlled 64-update causal closed-loop RGB continuation from the
selected step-584 perception state. The first command was:

```bash
conda run --no-capture-output -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name accuracy-closed-structured-v4 \
  --resume runs/accuracy-depth-finetune-v1/checkpoints/best_measurement.pt \
  --device cpu \
  --seed 17 \
  --set training.steps=648 \
  --set training.rgb_pretrain_steps=584 \
  --set training.train_episodes=64 \
  --set training.validation_episodes=32 \
  --set training.batch_size=2 \
  --set training.fixed_dataset=true \
  --set training.learning_rate=0.0002 \
  --set training.weight_decay=0.0001 \
  --set training.eval_every=16 \
  --set training.checkpoint_every=16 \
  --set training.log_every=4 \
  --set training.measurement_validation_frames=16
```

The first segment was intentionally stopped after the selected step-608
validation to avoid redundant full-validation work. The exact resume command
was:

```bash
conda run --no-capture-output -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name accuracy-closed-structured-v4 \
  --resume runs/accuracy-closed-structured-v4/checkpoints/best_rollout.pt \
  --device cpu \
  --seed 17 \
  --set training.steps=648 \
  --set training.rgb_pretrain_steps=584 \
  --set training.train_episodes=64 \
  --set training.validation_episodes=32 \
  --set training.batch_size=2 \
  --set training.fixed_dataset=true \
  --set training.learning_rate=0.0002 \
  --set training.weight_decay=0.0001 \
  --set training.eval_every=1000 \
  --set training.checkpoint_every=20 \
  --set training.log_every=4 \
  --set training.measurement_validation_frames=16
```

It completed step 648 in `699.7628 s`; no inaccurate sum with the earlier
segment is claimed. Full validation selected rollout-position loss
`0.0119829765`.

### Selection and confirmation

Step 648 and its step-584 initialization used the same ROI-local manifests:

| Evidence | Seeds | Step | Current MAE/RMSE (m) | Velocity RMSE (m/s) | 0.1/0.25/0.5 s RMSE (m) | Recovery | F1 | Coverage 90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selection baseline | `100032–100063` | 584 | `0.098112 / 0.127250` | `0.780543` | `0.150932 / 0.190620 / 0.248704` | `43.99%` | `0.588235` | `85.75%` |
| selection candidate | `100032–100063` | 648 | `0.093674 / 0.122592` | `0.793015` | `0.145440 / 0.185183 / 0.242517` | `44.29%` | `0.585938` | `85.88%` |
| confirmation baseline | `100064–100095` | 584 | `0.083808 / 0.109239` | `0.730034` | `0.134093 / 0.174492 / 0.231253` | `48.28%` | `0.594203` | `86.81%` |
| confirmation candidate | `100064–100095` | 648 | `0.083282 / 0.109426` | `0.731623` | `0.132424 / 0.171900 / 0.226994` | `47.82%` | `0.608059` | `86.76%` |

Both candidates retained 100% distance-gated detection and zero ID switches.
Step 648 was promoted because the forecast improvement repeated at every
horizon on selection and confirmation, while confirmation collision F1 also
increased. The tiny mixed current-position, velocity, recovery, and coverage
changes are reported rather than hidden.

Reports:

- `runs/accuracy-closed-structured-v4/evaluation/select32/report.md`
- `runs/accuracy-closed-structured-v4/evaluation/confirm32/report.md`
- step-584 selection baseline:
  `runs/accuracy-roi-local-v3/depth-finetune-select32/report.md`

### Final reserved RGB-only test

The final source/checkpoint pair was evaluated once on standard test seeds
`200064–200095`:

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

Observed over 32 RGB-only episodes:

- current position MAE/RMSE:
  `0.0893355713 / 0.1169080874 m`;
- current velocity RMSE: `0.7922569746 m/s`;
- model position RMSE at 0.10/0.25/0.50 seconds:
  `0.1382787450 / 0.1777031324 / 0.2328615442 m`;
- constant-velocity RMSE at the same horizons:
  `0.1557861172 / 0.3349483002 / 0.4719492865 m`;
- collision-conditioned model RMSE reduction versus constant velocity:
  `0.309652802 / 0.543772525 / 0.506596258`;
- injected-perturbation recovery: `0.452961913` (`45.30%`), positive on
  `97.92%` of evaluated horizons;
- collision precision/recall/F1:
  `0.765217 / 0.550000 / 0.640000`, false-positive rate `0.0255682`;
- distance-gated detection recall/precision: `1.0 / 1.0`;
- distance-gated ID switches: `0`;
- nominal-90% forecast coverage: `86.9518%`;
- zero dropped forecasts and zero non-finite outputs;
- mean CPU global/fast/rollout latency:
  `48.742 / 50.544 / 225.309 ms`.

The runtime used no oracle input. Evaluator-only simulator labels aligned
metrics and transparent baselines. There was no qualifying
visible-to-fully-occluded-to-visible sequence in these short episodes, so
sequence occlusion metrics are correctly null. Online identification executed,
but updates remained tiny: mean absolute informative drag/restitution updates
were `7.67e-7 / 1.74e-7`, with signed error changes
`+7.48e-8 / -1.28e-7`.

Report and machine-readable metrics:

- `runs/accuracy-closed-structured-v4/evaluation/final-test32/report.md`
- `runs/accuracy-closed-structured-v4/evaluation/final-test32/evaluation.json`

### Recursive multistep status

These are true recursive rollouts from one persistent RGB posterior, not
independent next-step predictions. At the tiny profile's 20 Hz observation
rate, the 0.10/0.25/0.50/0.75/1.00-second queries span 2/5/10/15/20 future
observation intervals. Dynamics internally advances roughly
12/30/60/90/120 physics substeps at 120 Hz. No future RGB or simulator state is
fed into the rollout.

The new `configs/tiny_multistep.yaml` extends synthetic sequences to 32 frames,
evaluates through one second, and makes full-horizon windows explicit. The
promoted step-648 baseline was evaluated on fresh-validation seeds
`100096–100111`:

```bash
conda run --no-capture-output -n orpheus python train.py \
  --config configs/tiny_multistep.yaml --dry-run --device cpu
```

Result: **passed**. It resolved the RGB-only 48x48 architecture, 32-frame
sequences, 656 total steps, PyTorch 2.10.0, and CPU because this launcher
reports MPS unavailable.

The exact baseline evaluation command was:

```bash
conda run --no-capture-output -n orpheus python evaluate.py \
  --config configs/tiny_multistep.yaml \
  --checkpoint runs/accuracy-closed-structured-v4/checkpoints/best_rollout.pt \
  --split validation \
  --seed-protocol fresh_validation \
  --seed-offset 96 \
  --device cpu \
  --set evaluation.episodes=16 \
  --output runs/accuracy-multistep-v1/evaluation/baseline-select16
```

| Checkpoint | Current RMSE (m) | 0.10 s | 0.25 s | 0.50 s | 0.75 s | 1.00 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| promoted step 648 | `0.149951` | `0.162863` | `0.190546` | `0.218011` | `0.230611` | `0.228255` |
| aggressive 16-update continuation | `0.154447` | `0.168657` | `0.199543` | `0.230619` | `0.248444` | `0.243224` |
| conservative 8-update short-window continuation | `0.150226` | `0.163706` | `0.191533` | `0.218606` | `0.230512` | `0.228311` |
| conservative 8-update one-second continuation | `0.149833` | `0.163161` | `0.190705` | `0.218114` | `0.230712` | `0.228364` |

The promoted model's mean 0.50/0.75/1.00-second RMSE is `0.225626 m`; the
best conservative continuation reached `0.225730 m`, a `0.046%` regression.
The aggressive run regressed every horizon. None was promoted, and the
step-648 SHA remains unchanged. Reports:

- `runs/accuracy-multistep-v1/evaluation/baseline-select16/report.md`
- `runs/accuracy-multistep-v1/evaluation/candidate-select16/report.md`
- `runs/accuracy-multistep-balanced-v4/evaluation/select16-long/report.md`
- `runs/accuracy-multistep-long-v5/evaluation/select16/report.md`

The baseline still substantially outperforms constant velocity as the horizon
grows: constant-velocity RMSE is
`0.181946 / 0.347282 / 0.683607 / 0.944198 / 1.423512 m`. Baseline collision
F1 is `0.404092`, detection recall/precision are both `0.970703`, and
nominal-90% coverage is `0.862745` on this longer protocol.

Evaluation-only oracle-start and dynamics ablations locate the remaining
error. At one second, replacing only the current position with labels reduced
coordinate RMSE from about `0.221` to `0.168 m`; replacing position and
velocity reduced it to `0.091 m`; additionally replacing slow physical
parameters reduced it to `0.0473 m`. Conversely, disabling the learned graph
acceleration changed a separate eight-seed one-second result by only
`+0.000277 m`; modal and learned-impulse effects were below `0.000005 m`.
The next accuracy work therefore belongs in RGB depth and anisotropic velocity
state estimation, not larger residual dynamics. Specifically, global RGB
measurement camera-depth RMSE/bias is `0.223 / -0.144 m`; filtering improves
that to `0.184 / -0.089 m`, while camera-x error grows from `0.053` to
`0.103 m` over four ROI steps, consistent with velocity drift.

A fit/held-out diagnostic found that suppressing gravity-orthogonal displacement
while retaining vertical dynamics could reduce held-out 0.50–1.00-second RMSE
by `7.48%`. It is deliberately not implemented as a fixed output blend:
doing so would make position inconsistent with velocity, covariance, events,
and the online prior, and the fitted gate could encode this split's motion
distribution. A production version must be observability/uncertainty-driven,
propagate a coherent belief trajectory, and pass wider/OOD confirmation.

### Current RGB-only forecast-history demo

```bash
MPLCONFIGDIR=/private/tmp/orpheus-mpl-cache \
conda run --no-capture-output -n orpheus python demo.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/accuracy-closed-structured-v4/checkpoints/best_rollout.pt \
  --device cpu \
  --output demo_outputs/accuracy-v4-forecast-history
```

The 16-frame demo used four global and 12 ROI-local updates with no oracle
input. Ten scoring-only lookahead frames keep the recursive horizon at
`0.50 s` in every displayed frame. The fixed-geometry GIF retains all 16
posterior forecasts with fading alpha and highlights the latest prior and
posterior. It labels endpoint matches and errors directly. Across 15 paired
comparisons, mean current prior/posterior error was
`0.212537 / 0.192126 m`; mean 0.50-second prior/posterior error was
`0.359118 / 0.341841 m`. Mean improvements were
`+0.020411 / +0.017277 m`, and maximum predicted collision probability was
`0.981901`. Ground truth is used only for the overlay and scores recorded in
the summary. Real artifacts:

- `demo_outputs/archive/20260727-125455-accuracy-v4-forecast-history/online_correction.gif`
- `demo_outputs/archive/20260727-125455-accuracy-v4-forecast-history/parameter_estimates.png`
- `demo_outputs/archive/20260727-125455-accuracy-v4-forecast-history/summary.json`
- `demo_outputs/archive/20260727-125455-accuracy-v4-forecast-history/frames/`

### Superseded and rejected experiments

- Step 584 remains the accuracy-v3 initialization and paired baseline. Its
  prior frozen test reached position RMSE `0.118600 m`, velocity RMSE
  `0.812524 m/s`, 0.5-second RMSE `0.237585 m`, recovery `45.72%`, and
  collision F1 `0.597122`; step 648 supersedes it.

- The 1,120-step from-scratch run
  `runs/accuracy-structured-physical-v1` was rejected. On selection seeds its
  current MAE/RMSE was `0.273908 / 0.422225 m`, 0.5-second RMSE
  `0.525709 m`, perturbation recovery `18.80%`, collision F1 `0.316940`,
  and detection recall/precision `66.60% / 55.22%`.
- A 2 cm collision-hazard lookahead was removed after confirmation F1 fell
  from `0.622222` to `0.594406` without changing physical trajectories.
- A later 256-update measurement continuation with the completed raw-centre
  objective did not beat inherited validation MAE and degraded as high as
  `0.291207 m`. `runs/accuracy-final-perception-v3` is retained as a rejected
  run, not a promoted checkpoint.
- Exhaustive collision-threshold validation found probabilities saturated near
  `0.018 / 0.998`; no threshold improved F1, so `0.5` remains unchanged and
  state/timing structure remains the target.
- Mean-radius analytic depth was rejected (`0.795 m` error versus `0.148 m`
  learned), and a photometric-radius alternative failed confirmation. Neither
  diagnostic changed runtime.

### Public smoke workflow

```bash
conda run --no-capture-output -n orpheus python train.py \
  --config configs/toy_smoke.yaml \
  --run-name accuracy-v3-smoke \
  --device cpu
```

This completed 12 finite CPU optimizer steps in `194.0503 s` (four perception,
eight closed-loop), wrote both selected checkpoints plus `last.pt`, exercised
ROI-local supervision, and used no oracle runtime input. Its
`1006.558 m` validation measurement MAE is intentionally highly inaccurate:
four perception updates cannot train this model. This run is wiring evidence
only. Artifacts are under `runs/accuracy-v3-smoke/`.

## Historical CPU convergence evidence (superseded)

Training command:

```bash
conda run -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name convergence-tiny-cpu-v1
```

Observed result:

- 70 optimizer steps on CPU in 59.845 s;
- 64 global RGB pretraining steps at learning rate `0.002`, covering all 16
  frames of eight deterministic training episodes;
- six full RGB closed-loop steps at learning rate `0.0002`;
- best 16-frame/four-episode validation world-position MAE: 0.422427 m at
  step 64, with 0.734375 recall/precision at the 0.5 m gate;
- that best localized checkpoint was restored before the closed-loop stage;
- closed-loop validation rollout loss: 0.277626;
- final training-window future correction improvement: +0.008551 m;
- finite losses and gradients, with no oracle runtime input.

Artifacts:

- `runs/convergence-tiny-cpu-v1/checkpoints/best_rollout.pt`
- `runs/convergence-tiny-cpu-v1/checkpoints/best_measurement.pt`
- `runs/convergence-tiny-cpu-v1/checkpoints/last.pt`
- `runs/convergence-tiny-cpu-v1/metrics.jsonl`
- `runs/convergence-tiny-cpu-v1/train_summary.json`

The best and last checkpoint contain the same validated step-70 tensors for
this run. `best_measurement.pt` is the selected step-64 perception handoff.

### Primary two-episode test protocol

```bash
conda run -n orpheus python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/convergence-tiny-cpu-v1/checkpoints/best_rollout.pt \
  --split test \
  --device cpu \
  --output runs/convergence-tiny-cpu-v1/evaluation/best-test
```

Observed held-out RGB-only results:

- current position MAE/RMSE: 0.218436 / 0.289611 m;
- distance-gated recall and precision at 0.5 m: 0.59375 / 0.59375
  (38 of 64 object-frames), up from 0/64 in the earlier short run;
- model position RMSE at 0.10 / 0.25 / 0.50 s:
  0.296358 / 0.224920 / 0.180852 m;
- constant-velocity RMSE:
  0.286786 / 0.246651 / 0.535446 m;
- static RMSE:
  0.231903 / 0.189103 / 0.186144 m;
- 0.50 s model RMSE is 66.22% below constant velocity and 2.84% below static;
- injected-perturbation error reduction: 0.141265 m, or 27.71%, positive for
  all 12 evaluated object-horizons;
- 90% forecast coverage: 93.86%;
- distance-gated ID switch rate: 0 over 38 associations;
- no dropped forecasts and no non-finite output;
- collision F1: 0.

Reports:

- `runs/convergence-tiny-cpu-v1/evaluation/best-test/report.md`
- `runs/convergence-tiny-cpu-v1/evaluation/last-test/report.md`
- `runs/convergence-tiny-cpu-v1/evaluation/best-measurement-test/report.md`

The closed-loop stage improved over the step-64 perception checkpoint:
current MAE 0.249597 -> 0.218436 m, detection recall 0.484375 -> 0.59375,
0.50 s forecast RMSE 0.232091 -> 0.180852 m, and perturbation recovery
6.80% -> 27.71%.

### Wider eight-episode held-out check

```bash
conda run -n orpheus python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/convergence-tiny-cpu-v1/checkpoints/best_rollout.pt \
  --split test \
  --device cpu \
  --set evaluation.episodes=8 \
  --output runs/convergence-tiny-cpu-v1/evaluation/best-test-8episodes
```

Observed over 256 held-out object-frames:

- current position MAE/RMSE: 0.182494 / 0.239292 m;
- 0.5 m recall/precision: 0.75 / 0.75 (192/256);
- model RMSE at 0.10 / 0.25 / 0.50 s:
  0.239256 / 0.187325 / 0.162259 m;
- constant-velocity RMSE:
  0.234788 / 0.260275 / 0.491278 m;
- static RMSE:
  0.195517 / 0.166665 / 0.172445 m;
- 0.50 s model RMSE is 66.97% below constant velocity and 5.91% below static;
- injected-perturbation reduction: 0.069255 m or 19.59%, positive on 72.92%
  of 48 object-horizons;
- 90% forecast coverage: 96.05%;
- gated ID switch rate: 0 over 192 associations;
- collision F1: 0;
- no dropped forecasts and no non-finite output.

Report:
`runs/convergence-tiny-cpu-v1/evaluation/best-test-8episodes/report.md`.
The wider perturbation result is 0.41 percentage points below the recommended
20% gate, so that gate is not claimed as achieved on the larger sample.

### Accuracy and event follow-up

A controlled continuation protected global perception after the six-step
closed-loop adaptation window while training the ROI/filter/dynamics modules:

```bash
conda run --no-capture-output -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name accuracy-closed-frozen-94 \
  --resume runs/convergence-tiny-cpu-v1/checkpoints/best_rollout.pt \
  --set training.steps=94 \
  --set training.eval_every=8 \
  --set training.checkpoint_every=8 \
  --set training.log_every=2
```

The 24 new CPU steps completed in 176.439 s. Step 72 became the
validation-selected `best_rollout.pt` and remained best through step 94. The
step-94 `last.pt` preserved and slightly improved position accuracy on the
repeated diagnostic test seeds:

```bash
conda run --no-capture-output -n orpheus python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/accuracy-closed-frozen-94/checkpoints/last.pt \
  --split test \
  --device cpu \
  --set evaluation.episodes=8 \
  --output \
    runs/accuracy-closed-frozen-94/evaluation/last-test-8episodes-exact-events
```

With corrected exact-window event scoring, this exploratory comparison found:

- current position MAE: 0.178773 m versus 0.182494 m at step 70;
- 0.5 m recall/precision: 0.753906 / 0.753906;
- model RMSE at 0.10 / 0.25 / 0.50 s:
  0.237282 / 0.186030 / 0.161387 m;
- constant-velocity RMSE: 0.232863 / 0.258475 / 0.490275 m;
- 0.50 s model RMSE is 67.08% below constant velocity and 6.09% below static;
- perturbation reduction: 19.49%, positive on 72.92% of evaluated horizons;
- 90% coverage: 96.60%; ID switches: 0 over 193 gated associations;
- exact-window collision F1: 0.028169.
- informative online restitution updates: pre/post MAE
  0.10399209 / 0.10399210 over four samples, mean signed improvement
  -0.000000015 and update magnitude 0.000000045;
- informative drag updates: pre/post MAE 0.04388363 / 0.04388311 over
  53 samples, mean signed improvement 0.000000529 and update magnitude
  0.000001914;
- no complete reliably anchored visible-to-fully-occluded-to-visible
  transition occurred, so sequence-aware growth/recovery values are correctly
  null rather than inferred from pooled frames.

Artifacts:

- `runs/accuracy-closed-frozen-94/checkpoints/last.pt`
- `runs/accuracy-closed-frozen-94/train_summary.json`
- `runs/accuracy-closed-frozen-94/evaluation/last-test-8episodes-exact-events/report.md`

The corresponding RGB-only demo command was:

```bash
conda run --no-capture-output -n orpheus python demo.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/accuracy-closed-frozen-94/checkpoints/last.pt \
  --device cpu \
  --output demo_outputs/accuracy-closed-frozen-94
```

It uses four global and 12 fast ROI updates. Mean ordinary current/future
prior-to-posterior improvements are +0.008432 m / +0.011079 m:

- `demo_outputs/archive/20260726-232939-accuracy-closed-frozen-94/online_correction.gif`
- `demo_outputs/archive/20260726-232939-accuracy-closed-frozen-94/parameter_estimates.png`
- `demo_outputs/archive/20260726-232939-accuracy-closed-frozen-94/summary.json`
- `demo_outputs/archive/20260726-232939-accuracy-closed-frozen-94/frames/`

Applying only the corrected event semantics to the unchanged step-70
checkpoint raised collision F1 from 0 to 0.055556 without changing weights.
That report is at
`runs/accuracy-events-v2/evaluation/pretrain-checkpoint-test-8episodes/report.md`.

A further 32-step continuation with exact-window, positive-balanced event loss
was also run rather than assumed successful:

```bash
conda run --no-capture-output -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name accuracy-events-balanced-102 \
  --resume runs/convergence-tiny-cpu-v1/checkpoints/best_rollout.pt \
  --set training.steps=102 \
  --set training.eval_every=8 \
  --set training.checkpoint_every=8 \
  --set training.log_every=2
```

It completed in 277.627 s and improved its sampled validation rollout loss to
0.270487, but did not generalize: eight-episode current MAE was 0.183974 m,
0.50 s RMSE 0.168842 m, perturbation recovery 14.86%, and collision F1
0.027397. It is retained as a truthful negative result and is not promoted
over the safer position checkpoint.

The step-70, step-94, and balanced-event comparisons repeatedly inspect the
same fixed eight test episodes. They are exploratory diagnostics, not an
independent model-selection protocol. Step 72 was the validation-selected
checkpoint at that stage; accuracy-v3 has since superseded it.

## Historical step-72 validation and temporal ablation

The evaluator now reserves an explicit checkpoint-selection block after the
trainer's validation episodes:

```bash
conda run --no-capture-output -n orpheus python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/accuracy-closed-frozen-94/checkpoints/best_rollout.pt \
  --split validation \
  --seed-protocol fresh_validation \
  --device cpu \
  --set model.rgb.temporal_velocity_enabled=false \
  --set evaluation.episodes=16 \
  --output runs/temporal-rgb-evidence/fresh-validation-final-baseline
```

This 56.85-second CPU run used exactly seeds `100004–100019`. The report
asserts no overlap with trainer validation (`100000–100003`) or the reserved
test range. It is model-selection validation evidence, not final test
acceptance. The unchanged step-72 checkpoint produced:

- current position MAE/RMSE: `0.186991 / 0.239613 m`;
- distance-gated current velocity MAE/RMSE:
  `0.647751 / 1.369454 m/s` over 377 object-frames;
- ordinary velocity prior/posterior norm error:
  `1.747554 / 1.745960 m/s`, only `0.001594 m/s` improvement;
- model RMSE at 0.10 / 0.25 / 0.50 seconds:
  `0.236517 / 0.189670 / 0.174269 m`;
- collision F1 `0.042553`;
- perturbation recovery `20.0935%`, positive on `78.125%` of 96 horizons;
- 90% forecast coverage `97.7522%`, zero ID switches, zero dropped/non-finite
  forecasts.

Future-collision-conditioned model RMSE at 0.10 / 0.25 / 0.50 seconds was
`0.149769 / 0.137729 / 0.174269 m`, respectively
`22.08% / 56.87% / 65.94%` below constant velocity on the exact same masks.

The implemented temporal path keeps a separate sensor-local three-position
history keyed by persistent object ID. It survives global/ROI feature-cache
changes, requires strictly increasing timestamps and nonambiguous
associations, and performs a velocity-only diagonal correction. With an
explicit experimental variance ceiling of `1.0 (m/s)²`:

```bash
conda run --no-capture-output -n orpheus python evaluate.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/accuracy-closed-frozen-94/checkpoints/best_rollout.pt \
  --split validation \
  --seed-protocol fresh_validation \
  --device cpu \
  --set evaluation.episodes=16 \
  --set model.rgb.temporal_velocity_enabled=true \
  --set model.rgb.temporal_velocity_variance_ceiling=1.0 \
  --output runs/temporal-rgb-evidence/fresh-validation-final-temporal
```

Velocity RMSE improved to `1.309964 m/s`, ordinary correction improvement to
`0.025985 m/s`, short collision-conditioned RMSE to `0.140309 m`, and
collision F1 to `0.055172`. However, current position MAE worsened to
`0.190923 m`, 0.25-second RMSE to `0.201318 m`, perturbation recovery to
`19.2569%`, and calibration/detection also regressed. Ceilings two/four and
history size four showed the same tradeoff. No inference-only temporal setting
was promoted; the public profiles keep it disabled and use uncapped propagated
uncertainty when explicitly enabled without an override.

A controlled frozen-global continuation was run rather than inferred:

```bash
conda run --no-capture-output -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name temporal-continuation-94 \
  --resume runs/accuracy-closed-frozen-94/checkpoints/best_rollout.pt \
  --set model.rgb.temporal_velocity_enabled=true \
  --set model.rgb.temporal_velocity_variance_ceiling=1.0 \
  --set training.steps=94 \
  --set training.eval_every=8 \
  --set training.checkpoint_every=8 \
  --set training.log_every=2
```

Twenty-two new CPU steps completed in `183.147 s`, with finite gradients and
no oracle runtime input. Step 94 lowered the small trainer-validation rollout
loss to `0.249018`. On the larger fresh manifest with temporal correction it
improved velocity RMSE to `1.277519 m/s` and collision F1 to `0.121622`, but
position MAE worsened to `0.196397 m`, 0.10/0.25/0.50-second RMSE to
`0.243738 / 0.207295 / 0.184454 m`, and perturbation recovery to `11.843%`.
Disabling temporal inference on the same weights was also worse. This run is a
truthful negative result and was not promoted. Step 72 was retained at that
stage and has since been superseded by the step-584 accuracy-v3 checkpoint.

Artifacts:

- `runs/temporal-rgb-evidence/fresh-validation-final-baseline/report.md`
- `runs/temporal-rgb-evidence/fresh-validation-final-temporal/report.md`
- `runs/temporal-rgb-evidence/fresh-validation-temporal-ceiling2/report.md`
- `runs/temporal-rgb-evidence/fresh-validation-temporal-ceiling4/report.md`
- `runs/temporal-continuation-94/checkpoints/best_rollout.pt`
- `runs/temporal-continuation-94/evaluation/fresh-validation-enabled/report.md`
- `runs/temporal-continuation-94/evaluation/fresh-validation-disabled/report.md`

### Demo

```bash
conda run -n orpheus python demo.py \
  --config configs/tiny_overfit.yaml \
  --checkpoint runs/convergence-tiny-cpu-v1/checkpoints/best_rollout.pt \
  --device cpu \
  --output demo_outputs/convergence-tiny-cpu-v1
```

The held-out 16-frame RGB-only demo used four global and 12 fast ROI updates.
Mean ordinary prior-to-posterior improvement is now positive:
+0.007777 m for current state and +0.010584 m for future error. Artifacts:

- `demo_outputs/archive/20260726-223129-convergence-tiny-cpu-v1/online_correction.gif`
- `demo_outputs/archive/20260726-223129-convergence-tiny-cpu-v1/parameter_estimates.png`
- `demo_outputs/archive/20260726-223129-convergence-tiny-cpu-v1/summary.json`
- `demo_outputs/archive/20260726-223129-convergence-tiny-cpu-v1/frames/`

## MPS evidence

Earlier reduced explicit MPS smoke:

```bash
conda run -n orpheus python train.py \
  --config configs/tiny_overfit.yaml \
  --run-name milestone1-mps-smoke-final \
  --device mps \
  --set training.steps=3 \
  --set training.rgb_pretrain_steps=2 \
  --set training.tbptt_steps=2 \
  --set training.batch_size=1 \
  --set training.train_episodes=1 \
  --set training.validation_episodes=1 \
  --set training.eval_every=3 \
  --set training.checkpoint_every=3 \
  --set training.log_every=1 \
  --set evaluation.episodes=1
```

Observed result: three finite optimizer steps in 90.996 s on `mps`, including
global and differentiable fast ROI backward paths. A separate reduced two-step
run of the full 96x96 `toy_mps` architecture completed in 132.157 s at
`runs/milestone1-toy-mps-scaled-smoke`. These are hardware compatibility
checks, not convergence claims. The full 3,000-step schedule remains unrun.

## Final validation

Full suite:

```bash
conda run --no-capture-output -n orpheus pytest
```

Result: **191 passed, 3 skipped in 67.65 s**. The three skips are the
hardware-conditional MPS tests; this launcher process reports MPS unavailable.

Lint and formatting:

```bash
conda run --no-capture-output -n orpheus ruff check .
conda run --no-capture-output -n orpheus ruff format --check .
```

Results: **passed** after mechanically formatting four changed Python files.
Ruff reported all checks clean and all 154 Python files formatted.

Bytecode compilation:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-accuracy-v4-pycache PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
```

Result: **passed**.

Default MPS plan:

```bash
conda run --no-capture-output -n orpheus python train.py \
  --config configs/toy_mps.yaml --dry-run
```

Result: **passed**. It resolved PyTorch 2.10.0, RGB-only 96x96 input, 72
frames, and the configured 3,000-step schedule. It selected CPU because MPS
was unavailable to this process.

Whitespace/error check:

```bash
git diff --check
```

Result: **passed**.

## Known limitations and open acceptance gates

- Collision semantics are correct, but final-test precision/recall/F1
  `0.765217 / 0.550000 / 0.640000` remains below the recommended `0.75` F1
  gate. Recall is the larger failure.
- Current velocity RMSE remains weak at `0.792257 m/s`. The experimental
  overlapping-frame temporal update is disabled because it improved velocity
  while regressing aggregate physical accuracy.
- On the fresh 32-frame multistep protocol, model RMSE plateaus at about
  `0.218–0.230 m` from 0.5 to 1.0 seconds. Three weight/window-only
  continuations failed to beat step 648. Oracle-start evidence identifies RGB
  depth/velocity and slow-parameter state as the dominant ceiling; no oracle
  state is used in normal operation.
- Nominal-90% forecast coverage is `86.9518%`, so the diagonal filter is
  overconfident despite strong point accuracy. Explicit temporal/cross-modal
  correlation handling is absent.
- Online drag/restitution identification executes behind observability gates,
  but mean update magnitudes remain around `1e-7` and useful parameter
  convergence is not established.
- The final 32 short episodes contain no qualifying
  visible-to-fully-occluded-to-visible sequence. Occlusion uncertainty
  expansion and reobservation contraction remain supported by focused tests,
  not end-to-end final-test evidence.
- At this tiny CPU scale, full fast ROI updates (`50.544 ms`) are not faster
  than global updates (`48.742 ms`), despite the isolated structured ROI
  primitive taking only about `3.94 ms` for eight crops.
- The full 3,000-step `toy_mps` schedule remains unrun. Existing MPS results
  establish compatibility only, not convergence or throughput.
- Fast inverse-depth residual learning is safely disabled, not solved. It must
  be trained with belief-slot-aligned cached sequences and enabled only after a
  held-out per-mode correction gate passes.
- Max-aggregating substep collision logits preserves thresholded
  "occurred anywhere" semantics, but is not a calibrated probability-of-union
  calculation.
- Belief-slot-aligned fast supervision avoids rematching conditioned outputs,
  but its per-frame belief-to-label assignment can still switch under close
  crossings. Sequence-level training identity alignment remains future work.
- The evaluator now reports collision-conditioned matched model/baseline
  forecasts, but still lacks the full physics-violation and saved failure-plot
  suite.
- Multiple-hypothesis association, multi-frame tentative births, estimated
  camera pose, continuous collision timing, real video, and a second modality
  remain future work.

## Next concrete tasks

1. Improve RGB depth in the persistent posterior, then add gravity-aligned,
   anisotropic velocity evidence with covariance/observability gating. Any
   horizontal forecast gate must update position, velocity, covariance, and
   event rollouts coherently; do not add a horizon-specific output blend.
   Train a validation-selected probabilistic event head without sacrificing
   the promoted point/forecast accuracy; threshold tuning cannot fix the
   saturated structural errors.
2. Calibrate repeated RGB uncertainty to restore nominal coverage, then test
   expansion/recovery on an occlusion-rich held-out split.
3. Profile and optimize the complete ROI update, then train inverse-depth
   residuals on belief-slot-aligned jittered/cached sequences and enable them
   only after a held-out per-mode correction gate passes.
4. Make drag/restitution identification numerically effective under explicit
   observability, and report parameter convergence on an event-rich split.
5. Add physics-violation metrics and saved collision/forecast failure plots.
6. Run the full `toy_mps` schedule when MPS is available and compare CPU/MPS
   throughput without changing the runtime/data contracts.

The paths above record the original experiment locations. Per the explicit
2026-07-27 cleanup request, superseded `runs/` directories were later deleted
after the selected checkpoint and compact evidence were consolidated. Generated
artifacts remain gitignored and are not published in the source commit.

## RGB lateral-motion and trajectory-display correction (2026-07-27)

The trajectory display now draws ground truth exactly once per object and
separates the faint dotted past from the solid current forecast horizon.
Persistent object colours, `GT <id>` labels, start squares, endpoint crosses,
time-direction arrows, fixed axes, and a fixed legend remove the prior
double-drawn curve/vertical-line ambiguity. Historical posterior forecasts
remain in absolute coordinates and fade by age.

The model-side defect was an unobservable isotropic finite-difference update:
at 20 Hz it divided full backprojection covariance by `dt²`, yielding roughly
`700 (m/s)²` velocity variance and almost zero horizontal Kalman gain. The new
opt-in `configs/tiny_lateral_velocity.yaml` path maintains a bounded,
persistent-ID temporal history, extracts only the camera-lateral component,
preserves analytic vertical/depth velocity, resets across collision events,
and only initializes young tracks. A conservative `0.125` blend of associated
RGB world positions into corrected posterior history gives the horizontal
signal without continuously overriding physical dynamics.

Selected checkpoint:
`runs/20260727-193657-selected-contact-confidence-v1/checkpoints/best_rollout.pt`
(step 648, CPU runtime semantics). On the same fresh-validation seeds
`100096–100111`, the previous step-648 baseline versus the selected candidate
was:

- current position RMSE: `0.149951 → 0.139696 m` (6.84% lower);
- current velocity RMSE: `0.791269 → 0.762795 m/s` (3.60% lower);
- recursive 0.10/0.25/0.50/0.75/1.00-second RMSE:
  `0.162863/0.190546/0.218011/0.230611/0.228255 →
  0.150671/0.173691/0.196885/0.204839/0.209191 m`;
- collision F1: `0.404092 → 0.398922`;
- nominal-90% coverage: `0.862745 → 0.846814`;
- no dropped or nonfinite forecasts occurred.

The original exact report was consolidated with the newly accepted contact
semantics under
`runs/20260727-193657-selected-contact-confidence-v1/evaluation/`.
Rejected ablation directories were removed after their outcomes were recorded.
Continuous strong temporal updates, an eight-step adapted continuation, raw
two-frame RGB slopes, and raw three-frame slopes all failed the wider physical
gate.

The regenerated RGB-only demo is
`demo_outputs/20260727-162848-accuracy-v6-blended-velocity/online_correction.gif`,
with
`summary.json`, 32 PNG frames, and `parameter_estimates.png` in the same
directory. It uses seed `200000`, a fixed 1.0-second displayed horizon and 20
undisplayed lookahead frames. Mean current posterior error was `0.194245 m`;
mean 1.0-second posterior endpoint error was `0.272091 m`. Ground truth was
used only for scoring and overlay.

Known remaining limitation: this collision-heavy demo still predicts excessive
early damping of lateral motion. The RGB velocity information path is now
causal and nonzero, but the event model assigns maximum collision probability
about `0.9819` over the horizon and does not localize collision time sharply.
Continuous collision timing/event calibration is the next model-side accuracy
task. The selected point-accuracy gain also trades away 1.28% relative
collision F1 and 1.85% relative nominal-90% coverage on this selection block.

Commands and outcomes for this change:

```bash
PYTHONPATH=. conda run -n orpheus python evaluate.py \
  --config configs/tiny_lateral_velocity.yaml \
  --checkpoint runs/accuracy-lateral-velocity-v5/checkpoints/blended_birth_window.pt \
  --split validation --seed-protocol fresh_validation --seed-offset 96 \
  --output runs/accuracy-lateral-velocity-v5/evaluation/select16
```

Result: passed on CPU, 16 RGB-only episodes, seeds `100096–100111`,
`oracle_runtime_input_used=false`; metrics are recorded above.

```bash
MPLCONFIGDIR=/private/tmp/orpheus-mpl PYTHONPATH=. \
  conda run -n orpheus python demo.py \
  --config configs/tiny_lateral_velocity.yaml \
  --checkpoint runs/accuracy-lateral-velocity-v5/checkpoints/blended_birth_window.pt \
  --output demo_outputs/accuracy-v6-blended-velocity
```

Result: passed on CPU; generated 32 frames, GIF, summary, and parameter plot.

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest
conda run --no-capture-output -n orpheus ruff check .
conda run --no-capture-output -n orpheus ruff format --check .
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-lateral-pycache PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
git diff --check
```

Results: `200 passed, 3 skipped in 59.90 s`; all Ruff checks passed and all
154 Python files were formatted; compileall and diff checks passed. The three
skips are the existing MPS-conditional tests, because this process reports MPS
unavailable. Checkpoint SHA-256:
`d5628d9df10ebd9fea30223cc2b38b41248e7e361b90988306a36a34fabad2b2`.
GIF SHA-256:
`74f6cf96fd0cd12723f7c1a255ab44ab9f15e8206909b8a51fc21c6f16cfe690`.

## Mixed interaction scenario audit (2026-07-27)

`configs/tiny_interactions.yaml` adds a deterministic four-regime mixture with
three simultaneous objects and 40-frame episodes:

- `baseline`: the existing in-distribution ranges;
- `elastic_pairs`: restitution `0.78–0.95`, drag `0.005–0.04`, friction
  `0.03–0.12`, and faster initial motion;
- `damped_contacts`: restitution `0.18–0.42`, drag `0.18–0.32`, and friction
  `0.28–0.55`;
- `impulse_perturbation`: moderate contact parameters plus labelled random
  impulses with probability `0.12` per observation interval.

Scenario selection is `seed % len(scenario_mixture)`, is recorded in episode
metadata, and changes physical sampling only; tensor/runtime contracts are
unchanged. Focused simulator/config coverage passed `42` tests.

Two CPU continuations from the selected lateral-velocity checkpoint were
completed and rejected:

1. `runs/accuracy-interactions-v1` ran eight closed-loop steps over 96 mixed
   training and 12 mixed validation episodes in `677.31 s`. It was nearly
   neutral on baseline/elastic/damped scenes and improved the impulse regime
   by `3.07%` current-position RMSE, `2.83%` 0.5-second RMSE, and `4.55%`
   relative collision F1.
2. `runs/accuracy-interactions-v2` ran eight RGB pretraining plus eight
   closed-loop steps in `847.90 s`. Although velocity RMSE improved
   `5.46–33.31%`, position forecasts regressed broadly and three-object
   detection recall fell, so its step-664 checkpoint is not promoted.

Paired four-episode scenario results on fresh-validation seeds
`100012–100015` for the untouched step-648 checkpoint versus the rejected
step-664 RGB-adapted checkpoint were:

| Scenario | current RMSE m | 0.5 s RMSE m | 1.0 s RMSE m | collision F1 | detection recall |
| --- | --- | --- | --- | --- | --- |
| baseline old/new | `0.2570/0.3004` | `0.3014/0.3241` | `0.3375/0.3550` | `0.3836/0.5795` | `0.5146/0.4417` |
| elastic old/new | `0.3472/0.3866` | `0.4036/0.4242` | `0.4498/0.4380` | `0.3333/0.2778` | `0.3854/0.2479` |
| damped old/new | `0.1998/0.2665` | `0.2211/0.3002` | `0.2124/0.3250` | `0.2152/0.2632` | `0.5542/0.4771` |
| impulse old/new | `0.1948/0.2710` | `0.2242/0.3069` | `0.2305/0.3414` | `0.4348/0.5000` | `0.5625/0.4458` |

All evaluations were RGB-only, used no oracle runtime input, and produced no
nonfinite or dropped forecasts. The original model therefore executes across
all regimes, but is not accurate enough for three-object elastic interactions.
The main blocker is global multi-object discovery/association rather than
rollout-only dynamics adaptation. The selected
`accuracy-lateral-velocity-v5` checkpoint remains promoted.

Exact training commands:

```bash
PYTHONPATH=. conda run -n orpheus python train.py \
  --config configs/tiny_interactions.yaml \
  --run-name accuracy-interactions-v1 \
  --resume runs/accuracy-lateral-velocity-v5/checkpoints/blended_birth_window.pt

PYTHONPATH=. conda run -n orpheus python train.py \
  --config configs/tiny_interactions.yaml \
  --run-name accuracy-interactions-v2 \
  --resume runs/accuracy-lateral-velocity-v5/checkpoints/blended_birth_window.pt
```

Per-regime reports are under
`runs/accuracy-interactions-v1/evaluation/*-{baseline4,trained4}` and
`runs/accuracy-interactions-v2/evaluation/*-trained4`.

Final source validation:

```bash
conda run --no-capture-output -n orpheus ruff check .
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest
```

Results: Ruff passed; `203 passed, 3 skipped in 62.17 s`. The skips are the
existing MPS-conditional tests because MPS was unavailable to this process.

## Sortable artifact cleanup (2026-07-27)

New training, evaluation, and demo artifact directory basenames are prefixed
with a UTC `YYYYMMDD-HHMMSS-` timestamp. This also applies to explicit
`--run-name` and `--output` labels; already-prefixed names are unchanged, and
resuming without a new run name continues in the checkpoint's existing
directory. The CLI JSON result is the source of truth for the generated path.

The latest RGB-only demo is now directly visible at:

- `demo_outputs/20260727-162848-accuracy-v6-blended-velocity/online_correction.gif`
- `demo_outputs/20260727-162848-accuracy-v6-blended-velocity/summary.json`
- `demo_outputs/20260727-162848-accuracy-v6-blended-velocity/parameter_estimates.png`
- `demo_outputs/20260727-162848-accuracy-v6-blended-velocity/frames/`

Eight superseded demo sets were moved, not deleted, under
`demo_outputs/archive/`, retaining timestamp-first names derived from their
existing filesystem times. Historical `runs/` directories were deliberately
left in place because checkpoint and report paths are cited throughout the
research record.

Commands and outcomes:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff format .
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff check .
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest
```

Results: all 156 Python files were already formatted; Ruff passed; `206 passed,
3 skipped in 61.17 s`. The skips are the existing MPS-conditional tests.

```bash
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-artifact-pycache PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
git diff --check
```

Results: compileall and diff checks passed. The process used macOS,
Python `3.10.20`, and PyTorch `2.10.0`; MPS is built but unavailable to this
process, so validation ran on CPU. The latest GIF still has SHA-256
`74f6cf96fd0cd12723f7c1a255ab44ab9f15e8206909b8a51fc21c6f16cfe690`,
confirming that cleanup changed its location, not its contents.

## Uncertainty-aware contact and seven-scenario audit (2026-07-27)

The large lime rings in the RGB panel were misleading: diagonal world
variance was multiplied by a fixed pixel scale. The demo now projects the full
diagonal world-position covariance through the calibrated camera Jacobian,
draws the resulting oriented 90% ellipse, and labels it separately from the
posterior point.

The dynamics contact resolver already applied impulses at 120 Hz geometric
contact substeps. The early lateral damping came from treating an uncertain
posterior mean as exact. Pair and plane contact now require
`gap + 0.25 * sigma_gap <= contact_margin`. A single-frame structured
apparent-radius depth replacement was tested first and rejected: current RMSE
rose to `0.519538 m` because independently varying physical radius and depth
are not identifiable from one fixed-camera silhouette.

On the unchanged selected step-648 weights and fresh-validation seeds
`100096–100111`, old versus accepted `0.25σ` semantics were:

| Metric | old | accepted |
| --- | ---: | ---: |
| current position RMSE | `0.139696 m` | `0.137969 m` |
| current velocity RMSE | `0.762795 m/s` | `0.830722 m/s` |
| 0.10 s RMSE | `0.150671 m` | `0.147986 m` |
| 0.25 s RMSE | `0.173691 m` | `0.168943 m` |
| 0.50 s RMSE | `0.196885 m` | `0.189775 m` |
| 0.75 s RMSE | `0.204839 m` | `0.191977 m` |
| 1.00 s RMSE | `0.209191 m` | `0.200973 m` |
| collision F1 | `0.398922` | `0.409836` |
| nominal 90% coverage | `0.846814` | `0.847733` |

The accepted checkpoint and report are:

- `runs/20260727-193657-selected-contact-confidence-v1/checkpoints/best_rollout.pt`
- `runs/20260727-193657-selected-contact-confidence-v1/evaluation/20260727-185824-select16/report.md`
- `runs/20260727-193657-selected-contact-confidence-v1/scenario-audit/`

Checkpoint SHA-256 remains
`d5628d9df10ebd9fea30223cc2b38b41248e7e361b90988306a36a34fabad2b2`.

Three regimes were added to the existing baseline/elastic/damped/impulse
mixture: moving-camera parallax, fast low-friction glancing impacts, and broad
unequal-mass impacts. A deterministic CPU continuation used 140 training and
14 validation episodes, 24 RGB updates and 16 closed-loop updates:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/tiny_interactions.yaml \
  --run-name interaction-parallax-v3 \
  --resume runs/accuracy-lateral-velocity-v5/checkpoints/blended_birth_window.pt \
  --set training.steps=688 \
  --set training.rgb_pretrain_steps=672 \
  --set training.checkpoint_every=8 \
  --set training.eval_every=8
```

Result: completed on CPU in `1718.97 s`; step 680 was internally selected at
validation rollout-position loss `0.067989`. On paired two-episode breadth
checks it improved current position and 0.5/1.0-second position in all seven
regimes, and velocity in six. It was nevertheless rejected on the decisive
original 16-episode gate: versus the accepted inherited weights, current RMSE
regressed `0.137969 → 0.159862 m`, 1-second RMSE
`0.200973 → 0.241969 m`, detection recall
`0.992188 → 0.947266`, and collision F1
`0.409836 → 0.384615`. Its compact metrics and configuration remain under
`runs/20260727-193657-selected-contact-confidence-v1/rejected-interaction-training/`;
its checkpoint was not retained.

The regenerated RGB-only demo command was:

```bash
MPLCONFIGDIR=/private/tmp/orpheus-mpl PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python demo.py \
  --config configs/tiny_lateral_velocity.yaml \
  --checkpoint runs/accuracy-lateral-velocity-v5/checkpoints/blended_birth_window.pt \
  --output demo_outputs/contact-confidence-v1
```

Actual artifact:
`demo_outputs/20260727-193538-contact-confidence-v1/online_correction.gif`.
Mean current posterior error improved from `0.194245` to `0.183953 m`; mean
1-second endpoint error improved from `0.272091` to `0.259955 m`. The
uncertainty ellipse is now correctly projected. The fixed-camera right-ball
forecast remains too vertical because physical radius/depth/height are still
weakly observable; this is a known limitation, not a solved claim.

After verifying the consolidated checkpoint and evidence, 64 superseded run
directories (`~156 MB`) were permanently removed at the user's request.
`runs/` now contains only the timestamped `1.1 MB` selected bundle above.

Final source validation:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff format .
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff check .
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-contact-pycache PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
git diff --check
```

Results: one Python file was reformatted and 155 were unchanged, Ruff passed,
`209 passed, 3 skipped in 61.12 s`, and compileall/diff checks passed. The three skips are
the existing MPS-conditional tests because MPS was unavailable to this
process. The new GIF SHA-256 is
`a164b44e0d581d37373f34346d447450ceb8a568ad13a438f1841773677628d4`.

## Predictive-abstraction foundation (2026-07-27)

`PROJECT_SPEC.md` is now version 1.1 and makes the smallest useful executable
predictive abstraction the scaling unit. Foundation perception, transformers,
and generative objectives may extract entities, residual information, model
families, or future hypotheses, but `WorldBelief` remains authoritative and
generated pixels are not accepted as evidence of correct physical state.

The first source increment adds `world_model/abstractions/`:

- an explicit registry containing implemented `POINT_TRAJECTORY` and
  `RIGID_SPHERE` families;
- a deterministic router that labels free-motion entities as cheap point
  trajectories and contact-like modes as rigid spheres;
- an `AbstractionAssignment` with kind, routing confidence, complexity cost,
  reason, and active mask;
- a parameter-free `WorldBeliefTokenizer` producing typed scene, camera,
  entity-kinematic, dynamical-programme, and lifecycle tokens; and
- exact decoding back into the matching belief schema, including identity,
  masks, fast/slow state, uncertainty, modal state, camera, and lifecycle.

`OnlineWorldModel.predictive_abstractions()` and
`OnlineWorldModel.predictive_tokens()` expose these derived views. They do not
cache a second physical state and introduce no model parameters or checkpoint
keys. The current router is inspectable infrastructure, not a trained result:
it does not yet prune the existing hybrid dynamics path because mode-only
routing could miss a not-yet-labelled imminent collision. A validated
proximity/uncertainty refinement gate and evidence-driven selection must
precede assignment-controlled execution.

Focused validation:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus \
  pytest tests/unit/test_predictive_abstractions.py \
  tests/unit/test_belief_invariants.py tests/unit/test_packing.py
```

Result: `14 passed in 1.29 s`.

Full source validation:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff format .
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff check .
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest
```

Results: all 162 Python files were already formatted, Ruff passed, and
`213 passed, 3 skipped in 62.23 s`. The skips remain the MPS-conditional tests
because this subprocess reports MPS unavailable. No training or held-out
evaluation was run for this parameter-free contract increment, so the selected
checkpoint and recorded accuracy metrics are unchanged.

## Reference physics and axis-resolved prediction (2026-07-27)

The specification is now 1.2. The clean `reference_pairs` scenario separates
the ensured sphere-pair impact from the first floor impact and records
sphere-world simulator version 2. The model/data contracts are unchanged.
PyBullet and MuJoCo were not installed in `orpheus`; Gymnasium was present.
No engine was installed. A mature engine is specified as a future independent
RGB dataset backend, not a predictor or default smoke dependency.

The RGB path now extracts an explicit point and equivalent-area scale from each
structured disc. In the fixed-radius reference profile, scale produces
calibrated analytic depth evidence. Per-axis position and velocity metrics and
axis-weighted rollout losses make the weak x component visible. Fast kinematic
state, joint interaction/event context, slow parameter gates, identity, and
uncertainty remain explicit.

Two short CPU continuations were run on the clean curriculum. The selected
step-672 weights came from the second continuation's lower internal two-episode
rollout loss (`0.066595`), but an external four-seed test was essentially
unchanged from the inherited weights. This is not evidence that the
continuation generalized. The promoted artifact is therefore described as the
step-672 weight source plus a parameter-free structured point/scale runtime,
not as a successful accuracy-training claim.

Full RGB-only standard-test result on seeds `200000–200015`:

- current position x/y/z RMSE:
  `0.416612 / 0.111049 / 0.174254 m`;
- current aggregate position RMSE: `0.268491 m`;
- current x/aggregate velocity RMSE: `1.120426 / 0.985151 m/s`;
- model x RMSE at 0.10/0.25/0.50/0.75/1.00 seconds:
  `0.434326 / 0.474210 / 0.582723 / 0.729778 / 0.816342 m`;
- aggregate model RMSE at those horizons:
  `0.280165 / 0.305264 / 0.370350 / 0.452147 / 0.503794 m`;
- one-second constant-velocity x/aggregate RMSE:
  `0.970399 / 1.628896 m`;
- collision F1: `0.461538`;
- distance-gated detection recall/precision: `0.720703 / 0.720703`;
- identity switches: `0`; predicted/target object frames:
  `1024 / 1024`; non-finite outputs: `0`.

This is a coherent runnable result and the model beats constant velocity at the
one-second x endpoint, but absolute x accuracy is not yet good. The final demo
visibly retains near-vertical forecasts for some frames; the corrected ground
truth now shows familiar ballistic and pair-contact motion, making that model
failure diagnosable.

Accepted artifacts:

- checkpoint:
  `runs/20260727-233802-reference-physics-v1/checkpoints/best_rollout.pt`
  (SHA-256
  `075245ae5edc426c98c2df0d74e1bff53c8f6a762fd05e24bf4677c06673e2b8`);
- evaluation:
  `runs/20260727-233802-reference-physics-v1/evaluation/20260727-234344-final-test16/`;
- demo:
  `demo_outputs/20260727-234542-reference-physics-final/online_correction.gif`
  (SHA-256
  `8c7405b82e86e75849cba7c7b0fa4117d15d9065c687536c32b9cd4e067a5f35`).

Selection diagnostics rejected denser global cadence, direct raw-point
temporal blending (with and without post-event reopening), fast-path structured
confidence, and a zero learned-correction scale. They improved isolated
current-state quantities or were neutral but failed the four-seed one-second x
selection gate.

Four superseded timestamped run directories and three superseded demo
directories were moved, without deletion, to
`/private/tmp/orpheus-superseded-20260727/`. The workspace `runs/` directory
now contains only the selected reference-physics bundle; the newest demo is the
only non-archive directory under `demo_outputs/`.

Environment and final validation commands:

```bash
conda run -n orpheus python -c \
  "import platform,torch; print(platform.python_version(), torch.__version__, \
  torch.backends.mps.is_built(), torch.backends.mps.is_available())"
PYTHONPATH=. conda run -n orpheus python evaluate.py \
  --config configs/tiny_lateral_velocity.yaml \
  --checkpoint runs/20260727-233802-reference-physics-v1/checkpoints/best_rollout.pt \
  --split test \
  --output runs/20260727-233802-reference-physics-v1/evaluation/final-test16 \
  --device cpu
PYTHONPATH=. conda run -n orpheus python demo.py \
  --config configs/tiny_lateral_velocity.yaml \
  --checkpoint runs/20260727-233802-reference-physics-v1/checkpoints/best_rollout.pt \
  --output demo_outputs/reference-physics-final \
  --device cpu
PYTHONPATH=. conda run -n orpheus python -m ruff format .
PYTHONPATH=. conda run -n orpheus python -m ruff check .
PYTHONPATH=. conda run -n orpheus pytest
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-reference-pycache PYTHONPATH=. \
  conda run -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
git diff --check
```

Environment: Python `3.10.20`, PyTorch `2.10.0`, MPS built `True`, MPS
available to the process `False`; evaluation/training therefore ran on CPU in
float32. Formatting changed seven files, Ruff passed, and pytest reported
`220 passed, 3 skipped in 175.22 s`; all skips were MPS-conditional.
Compileall passed. The first combined compile/diff invocation exposed one
Markdown trailing-space error; it was corrected and the final diff check
passed.

## 2026-07-28 — one shared model across eight scenarios

`configs/tiny_all_scenarios.yaml` now exercises, in deterministic order,
`reference_pairs`, `baseline`, `elastic_pairs`, `damped_contacts`,
`impulse_perturbation`, `camera_parallax`, `glancing_impacts`, and
`heavy_light_impacts`. It keeps one RGB runtime, one `WorldBelief` contract,
and one checkpoint; no scenario state or simulator state enters the model.

The selected checkpoint retains the learned step-672 weights and adds the
continuous causal RGB point-history observer across all regimes. Three
all-scenario adaptations were screened: a short full-model continuation, a
conservative dynamics-only continuation, and an eight-step dynamics-only
continuation at `1e-4`. The apparent first gain was traced to an evaluator seed
offset confound: the source checkpoint embedded two validation episodes while
the candidates embedded eight, so the default fresh-validation blocks
differed. After forcing the identical offset-8 manifest, neither dynamics
candidate improved the declared multistep objective. No adapted weights were
promoted.

The trainer now refuses to inherit a best rollout score when the validation
episode count, scenario mixture, sequence length, object-count range, seed,
horizons, or metric version changes. Evaluation outputs also persist the
simulator version, ordered scenario mixture, and scenario selected for every
episode. Collision-triggered temporal-history clearing is edge-triggered so a
sustained event mode does not erase every outgoing sample.

Final shared RGB-only test on seeds `200000–200015`, exactly two episodes from
each scenario:

- current position aggregate/x RMSE: `0.200430 / 0.156614 m`;
- current velocity aggregate/x RMSE: `0.968753 / 0.497078 m/s`;
- aggregate recursive position RMSE at
  0.10/0.25/0.50/0.75/1.00 seconds:
  `0.214112 / 0.245655 / 0.290137 / 0.320995 / 0.364040 m`;
- x recursive position RMSE at those horizons:
  `0.164730 / 0.202167 / 0.279684 / 0.365591 / 0.462605 m`;
- one-second constant-velocity aggregate/x RMSE:
  `1.527034 / 0.804134 m`;
- collision F1: `0.320388`;
- distance-gated RGB detection recall/precision:
  `0.762957 / 0.873473`;
- identity switches: `3`; non-finite outputs: `0`.

Per-scenario two-episode one-second aggregate RMSE was `0.2715` baseline,
`0.3144` elastic, `0.3251` damped, `0.2682` impulse, `0.3519` camera parallax,
`0.2905` glancing, `0.2764` heavy/light, and `0.5147 m` reference pairs.
Reference pairs remains the weakest multistep regime; damped contacts remains
the weakest current localization regime. Fine-grained reference diagnostics
separate two bottlenecks: missed three-object RGB discovery and inaccurate
post-contact lateral velocity. Lower temporal-velocity variance, longer
history, repeated-reset behavior, a change-point heuristic, and the trained
dynamics candidates did not pass paired selection.

Accepted artifacts:

- shared checkpoint:
  `runs/20260728-091315-selected-all-scenarios-v1/checkpoints/best_rollout.pt`;
- combined test report:
  `runs/20260728-091315-selected-all-scenarios-v1/evaluation/20260728-093649-final-test16-v13/report.md`;
- eight per-scenario reports under the same run's `evaluation/` directory;
- representative reference failure demo:
  `demo_outputs/20260728-092305-all-scenarios-reference/online_correction.gif`;
- representative damped failure demo:
  `demo_outputs/20260728-092305-all-scenarios-damped/online_correction.gif`.

The shared result substantially beats constant velocity at one second and is
stable and finite across the matrix, but it is not yet high absolute accuracy.
The next accuracy work is balanced three-object discovery supervision followed
by event-conditioned outgoing-velocity learning, with the same explicit
paired manifest and per-scenario regression gates.

Artifact SHA-256:

- checkpoint:
  `0aba6b222e10446aa892c810bd632c393e3b8d2195858f48e68022f66e847af2`;
- reference GIF:
  `c732b21096b304bf058db22f55369432a27569a5b5045a82e2c80614f6a2fa8e`;
- damped GIF:
  `76d52d34d11550caf4f002989221fe667d19a7b7328ae74ceafa2de8d315ca15`.

Commands run for the final shared selection included:

```bash
PYTHONPATH=. conda run -n orpheus python train.py \
  --config configs/tiny_all_scenarios.yaml \
  --resume runs/20260727-233802-reference-physics-v1/checkpoints/best_rollout.pt \
  --run-label shared-dynamics-v1
PYTHONPATH=. conda run -n orpheus python evaluate.py \
  --config configs/tiny_all_scenarios.yaml \
  --checkpoint <candidate> \
  --split validation --seed-protocol fresh_validation --seed-offset 8 \
  --device cpu --set evaluation.episodes=8
PYTHONPATH=. conda run -n orpheus python evaluate.py \
  --config configs/tiny_all_scenarios.yaml \
  --checkpoint runs/20260728-091315-selected-all-scenarios-v1/checkpoints/best_rollout.pt \
  --split test --device cpu \
  --output runs/20260728-091315-selected-all-scenarios-v1/evaluation/20260728-093649-final-test16-v13
PYTHONPATH=. conda run -n orpheus python demo.py \
  --config configs/tiny_all_scenarios.yaml \
  --checkpoint runs/20260728-091315-selected-all-scenarios-v1/checkpoints/best_rollout.pt \
  --device cpu --episode-index 0 \
  --output demo_outputs/20260728-092305-all-scenarios-reference
PYTHONPATH=. conda run -n orpheus python -m ruff format .
PYTHONPATH=. conda run -n orpheus python -m ruff check .
PYTHONPATH=. conda run -n orpheus pytest
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-all-scenarios-pycache \
  PYTHONPATH=. conda run -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
git diff --check
```

Final environment: Python `3.10.20`, PyTorch `2.10.0`, MPS built `True`,
MPS available to this process `False`; the shared sweep therefore ran on CPU.
Ruff passed, pytest reported `231 passed, 3 skipped in 173.23 s` with only
MPS-conditional skips, compileall passed, and `git diff --check` passed.
Five superseded run directories and the previous reference demo were moved
without deletion to `/private/tmp/orpheus-superseded-20260728/`; the workspace
now contains only the selected run and the two newest timestamped demos
(besides the existing demo archive).

## 2026-07-28 — RGB evidence audit and rejected accuracy candidates

The selected shared checkpoint remains
`runs/20260728-091315-selected-all-scenarios-v1/checkpoints/best_rollout.pt`.
No new weights or runtime policy passed the final multistep gate, so the
published test metrics above remain authoritative.

A direct held-out RGB diagnostic measured structured-centre RMSE at
`0.1388 px`. Global RGB 3-D measurement RMSE was `0.3865 m`, with axis RMSE
`0.0663 / 0.0992 / 0.3677 m`; camera-depth RMSE was `0.3837 m`. The dominant
failure is heavy-tailed single-frame scale/depth when components overlap,
truncate, or partially occlude one another.

Three bounded candidates were evaluated:

- simultaneous temporal least-squares and position-innovation velocity
  evidence improved every rollout horizon on paired validation offsets 8 and
  16, but the untouched 16-episode test regressed current position
  `0.200430 → 0.202235 m` and 1-second rollout
  `0.364040 → 0.365094 m`;
- adaptive covariance inflation for associated scale/depth disagreement
  improved final-test aggregate RMSE at 0.10/0.25/0.50/0.75 seconds to
  `0.210540 / 0.242193 / 0.284037 / 0.320669 m`, but regressed the 1-second
  endpoint to `0.364672 m` and x endpoint to `0.463177 m`;
- 128 additional balanced RGB measurement updates completed in `166.74 s`,
  but paired offset-8 1-second RMSE worsened
  `0.285499 → 0.329262 m` and detection recall fell
  `0.850694 → 0.817708`.

All were rejected. The generic evidence mechanisms remain tested and disabled
by default (`false`/`null`) so a future learned quality policy can use them
without changing belief contracts. The next accuracy milestone is a learned
multi-frame point/scale trajectory measurement aligned by persistent identity:
axis-local estimates and uncertainty, explicit scale/occlusion quality, and
joint event context gating when axes may change. It must correct
`WorldBelief`; it must not become a second history state or consume simulator
state.

Focused validation completed before documentation:

```bash
PYTHONPATH=. conda run -n orpheus pytest \
  tests/unit/test_config.py \
  tests/unit/test_structured_rgb_centres.py \
  tests/integration/test_rgb_online_loop.py \
  tests/integration/test_checkpoint_roundtrip.py
PYTHONPATH=. conda run -n orpheus python train.py \
  --config configs/tiny_all_scenarios.yaml \
  --resume runs/20260728-091315-selected-all-scenarios-v1/checkpoints/best_rollout.pt \
  --run-name balanced-rgb-discovery-v1 \
  --set training.steps=800 --set training.rgb_pretrain_steps=800 \
  --set training.learning_rate=0.0001 \
  --set training.eval_every=32 --set training.checkpoint_every=32
PYTHONPATH=. conda run -n orpheus python evaluate.py \
  --config configs/tiny_all_scenarios.yaml \
  --checkpoint <candidate> \
  --split validation --seed-protocol fresh_validation --seed-offset 8 \
  --device cpu --set evaluation.episodes=8
```

The focused suite passed `66` tests. Final validation ran:

```bash
PYTHONPATH=. conda run -n orpheus python -m ruff format .
PYTHONPATH=. conda run -n orpheus python -m ruff check .
PYTHONPATH=. conda run -n orpheus pytest
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-accuracy-pycache \
  PYTHONPATH=. conda run -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
git diff --check
```

Ruff passed; pytest reported `235 passed, 3 skipped in 67.85 s`, with
all skips MPS-conditional; compileall and `git diff --check` passed. Nine
rejected timestamped runs were moved without deletion to
`/private/tmp/orpheus-superseded-20260728-accuracy/runs/`. The workspace
`runs/` directory again contains only the selected timestamped artifact.

## 2026-07-28 — scaled curriculum and MPS proof

`configs/scaled_curriculum.yaml` defines the next generalization experiment:

- one shared `1,901,030`-parameter model, versus `156,490` parameters in the
  selected tiny all-scenario model;
- 4,096 training, 256 validation, and 256 test episodes;
- eight balanced scenario families, with continuous seed-driven variation in
  initial state, physical parameters, camera, object count, appearance, event
  timing, and noise;
- 48,000 episode draws (`11.71875` nominal manifest passes), batch one,
  eight-step TBPTT, and four on-the-fly renderer workers;
- the same RGB packets, `WorldBelief`, association, innovation, correction,
  identification, event, and rollout contracts.

The sandboxed process reported MPS built but unavailable. Running the same
diagnostic outside the execution sandbox confirmed Python `3.10.20`, PyTorch
`2.10.0`, MPS built `True`, and MPS available `True`. Explicit MPS resume then
found and fixed a checkpoint bug: `map_location=mps` had moved the saved CPU
RNG state to MPS before calling `torch.set_rng_state`. Restoration now
explicitly transfers CPU/CUDA generator states to CPU first.

Observed bounded scale run:

- artifact:
  `runs/20260728-131727-scaled-curriculum-1k-v1/`;
- 256 measurement optimizer updates, representing 1,024 episode draws from
  the 4,096-episode pool;
- best 16-episode measurement validation world-position MAE:
  `0.645048 m`;
- one full 48-frame causal MPS update checkpointed at step 257;
- step-257 rollout-position training loss `0.019657`, total loss `6.796081`,
  gradient norm `3.950501`, and no non-finite failure;
- device recorded in the checkpoint: `mps`.

The first causal step used a batch-one, eight-update graph and remained
expensive; a second was interrupted as memory and wall time continued to grow.
This artifact proves the large model/data/MPS/checkpoint path, but has no
closed-loop validation and is not promoted as more accurate than the selected
checkpoint. The full schedule has not been run.

Validation for this change:

```bash
PYTHONPATH=. conda run -n orpheus python train.py \
  --config configs/scaled_curriculum.yaml --dry-run
PYTHONPATH=. conda run -n orpheus python -m ruff format .
PYTHONPATH=. conda run -n orpheus python -m ruff check .
PYTHONPATH=. conda run -n orpheus pytest
PYTHONPATH=. conda run -n orpheus pytest \
  tests/integration/test_rgb_measurements.py::test_roi_sampling_mps_training_cpu_fallback_is_differentiable \
  tests/unit/test_association.py::test_association_transfers_cost_to_cpu_without_mps_float64 \
  tests/unit/test_modal_dynamics.py::test_modal_device_when_available
```

The full sandboxed suite passed `238` tests and skipped the three
MPS-conditional tests in `63.65 s`. Running those three tests outside the
sandbox, where MPS is available, passed all three in `2.40 s`. Ruff passed.

## 2026-07-28 — longer scaled MPS continuation and paired result

No prior trainer was active when this work began. The scaled checkpoint was
continued on direct MPS from step 256 to the persisted step-896 measurement
checkpoint, representing 640 additional optimizer updates over the same
4,096-episode, eight-scenario curriculum. The accepted stable segment used
24-frame episodes, batch one, four-step TBPTT, four renderer workers, and
`2.5e-5` measurement learning rate. Its eight causal updates used the
configured 10x phase reduction.

The first continuation used `1e-4` and encountered a non-finite learned
proposal during step-768 validation. Global structured assignment now ignores
non-finite proposal rows while retaining finite candidates, and validation
fails explicitly if its aggregate loss is non-finite. The stable rerun
reported measurement-validation world-position MAE:

- step 640: `0.614591 m`;
- step 768: `0.614583 m`;
- step 896: `0.614574 m`;
- original step 256: `0.645048 m`.

That internal measurement metric improved by `4.72%`. The final eight-episode
closed-loop validation was stopped after the complete process reached the
two-hour cap: it had spent about 84 minutes in validation, remained
compute-active, and had produced no result. Step-896 weights were already
safe; the eight later causal updates existed only in the interrupted process.
Training now writes the final `last.pt` before entering expensive final
validation, so future interruptions cannot discard completed optimizer work.

The unbiased paired RGB-only confirmation used fresh-validation seeds
`100016–100017`, disjoint from both checkpoints' trainer-validation manifests:

| Metric | step 256 | step 896 | Change |
| --- | ---: | ---: | ---: |
| current position RMSE | `0.945633 m` | `0.769763 m` | `-18.60%` |
| current position MAE | `0.546452 m` | `0.447509 m` | `-18.11%` |
| current velocity RMSE | `0.608411 m/s` | `1.180957 m/s` | `+94.11%` |
| 0.10 s position RMSE | `0.828191 m` | `0.721454 m` | `-12.89%` |
| 0.25 s position RMSE | `0.711230 m` | `0.670801 m` | `-5.68%` |
| 0.50 s position RMSE | `0.756100 m` | `0.718019 m` | `-5.04%` |
| 0.75 s position RMSE | `0.798965 m` | `0.789715 m` | `-1.16%` |
| 1.00 s position RMSE | `0.832044 m` | `0.873989 m` | `+5.04%` |
| 0.5 m detection recall | `0.243056` | `0.340278` | `+0.097222` |
| 0.5 m detection precision | `0.165094` | `0.270718` | `+0.105624` |
| collision F1 | `0.153846` | `0.400000` | `+0.246154` |
| forecast 90% coverage | `0.704981` | `0.668582` | `-0.036399` |

Both checkpoints had zero distance-gated ID switches and no non-finite
outputs. The longer run is therefore useful perception/short-horizon evidence,
but it fails the velocity, calibration, and one-second promotion gates. It
does not replace the selected tiny checkpoint. The sample contains only two
episodes and is confirmation evidence, not broad validation/test acceptance.

On the candidate, mean global/fast RGB update latency was approximately
`1.625 / 1.504 s`, and future rollout latency was `9.823 s` per evaluator
call. The current large closed-loop implementation is too slow for routine
full-manifest iteration and must be profiled before the 48,000-draw schedule.

Artifacts:

- checkpoint:
  `runs/20260728-152237-scaled-longer-stable-v2/checkpoints/best_measurement.pt`
  (step 896, SHA-256
  `125c4c45e2780a98d5321392d9ebea2cd72b98fab66d773a29fcbf4d2dc9cd4f`);
- candidate paired report:
  `runs/20260728-152237-scaled-longer-stable-v2/evaluation/20260728-174523-scaled-step896-paired-confirm2-offset16/report.md`;
- baseline paired report:
  `runs/20260728-131727-scaled-curriculum-1k-v1/evaluation/20260728-173848-scaled-step256-paired-confirm2-offset16/report.md`;
- supplementary candidate report on seeds `100008–100009`:
  `runs/20260728-152237-scaled-longer-stable-v2/evaluation/20260728-173118-scaled-longer-stable-v2-confirm2/report.md`.

Exact main training command:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/scaled_curriculum.yaml \
  --resume runs/20260728-151508-scaled-longer-v1/checkpoints/best_measurement.pt \
  --run-name scaled-longer-stable-v2 --device mps --seed 44 \
  --set simulator.sequence_frames=24 --set training.steps=904 \
  --set training.rgb_pretrain_steps=896 --set training.batch_size=1 \
  --set training.tbptt_steps=4 --set training.train_episodes=4096 \
  --set training.validation_episodes=8 --set training.num_workers=4 \
  --set training.learning_rate=0.000025 --set training.eval_every=128 \
  --set training.checkpoint_every=64 --set training.log_every=10 \
  --set evaluation.episodes=8
```

Exact paired evaluation command shape (checkpoint/output changed per side):

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python evaluate.py \
  --config runs/20260728-152237-scaled-longer-stable-v2/config.resolved.yaml \
  --checkpoint <step-256-or-step-896-checkpoint> \
  --split validation --seed-protocol fresh_validation --seed-offset 16 \
  --device mps --set evaluation.episodes=2 --output <paired-label>
```

Validation commands for the stability changes:

```bash
PYTHONPATH=. conda run -n orpheus ruff check \
  world_model/observations/rgb/structured_centres.py \
  world_model/training/trainer.py \
  tests/unit/test_structured_rgb_centres.py
PYTHONPATH=. conda run -n orpheus pytest \
  tests/unit/test_structured_rgb_centres.py \
  tests/integration/test_checkpoint_roundtrip.py
PYTHONPATH=. conda run -n orpheus pytest \
  tests/unit/test_training_schedule.py tests/integration/test_cli_smoke.py
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest \
  tests/unit/test_evaluation_parameter_update_metrics.py \
  tests/unit/test_association.py tests/integration/test_rgb_measurements.py -q
```

Results before the final full-suite run were `17 passed`, `22 passed`, and
`16 passed` respectively; the last command executed directly with MPS
available and included the new MPS float64-transfer regression.

Final repository validation:

```bash
PYTHONPATH=. conda run -n orpheus ruff format --check .
PYTHONPATH=. conda run -n orpheus ruff check .
PYTHONPATH=. conda run -n orpheus pytest
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-scaled-pycache \
  PYTHONPATH=. conda run -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
git diff --check
```

Ruff and `git diff --check` passed. Pytest reported
`239 passed, 4 skipped in 67.13 s`; all skips were hardware-conditional in the
sandboxed process, and the relevant direct-MPS subset passed as described
above.

## 2026-07-29 — identifiable scale, observer cadence, and useful causal windows

Longer training alone was not the initial remedy. The step-896 artifact had
completed 896 measurement updates but no persisted causal adaptation. Two
matched 16-update causal continuations from those weights were rejected:

- all trainable modules worsened current position by `2.76%`, velocity by
  `37.15%`, and 0.10–0.50-second forecasts by `1.36–3.20%`;
- freezing RGB and adapting only dynamics/filter/identifier still worsened
  current position by `2.65%`, velocity by `53.62%`, and 0.10–0.50-second
  forecasts by `1.24–3.10%`;
- increasing state/rollout velocity weights from `0.5` to `2.0` produced
  essentially the same rejected result.

Training supervision now keeps a persistent model-object-ID to simulator-target
mapping rather than recomputing nearest position at each contact. A matched
short run did not encounter a target swap, so this is a correctness fix rather
than an explanation for those regressions.

The scaled curriculum also contained an identifiability error: physical sphere
radius varied from `0.16–0.25 m`, while monocular back-projection used the
range mean. RGB apparent radius alone cannot separate unknown physical radius
from depth. `train.py --initialize-from` now supports strict weights-only
curriculum transfer with a reset step/optimizer/scheduler/RNG and recorded
provenance; it is mutually exclusive with exact `--resume`. The primary scaled
accuracy curriculum now uses fixed known `0.21 m` radius.

A 1,024-draw MPS transfer from step-896 weights completed in `659.55 s` across
all eight scenario families. Its best eight-episode measurement
world-position MAE was `0.380453 m`, versus `0.614574 m` before transfer.
The initial two-episode full online result did not improve, localizing the next
bottleneck to online ROI/tracker drift rather than global RGB localization.

Matched runtime gates on fixed-scale seeds `100016–100017`:

| Policy | current RMSE | velocity RMSE | 0.10 / 0.25 / 0.50 / 0.75 / 1.00 s RMSE |
| --- | ---: | ---: | --- |
| six-frame global anchor | `0.963351 m` | `1.086428 m/s` | `0.844761 / 0.704927 / 0.745774 / 0.853151 / 1.007125 m` |
| ROI component-scale ablation | `0.925932 m` | `1.308184 m/s` | `0.836798 / 0.710704 / 0.744329 / 0.849377 / 1.003551 m` |
| global every frame | `0.936374 m` | `1.296193 m/s` | `0.841321 / 0.674488 / 0.678569 / 0.785979 / 1.001967 m` |
| historical config value 3 (actual global every four frames) | `0.906217 m` | `1.082334 m/s` | `0.780533 / 0.626639 / 0.650438 / 0.773491 / 1.007125 m` |

The ROI scale policy was rejected because velocity, detection, collision F1,
and calibration regressed. It remains implemented behind a disabled,
crop-boundary-gated configuration flag. The policy then labelled cadence three
actually retained three fast ROI frames per cycle and improved current
position, 0.10–0.75-second
forecasts, detection, collision F1, and coverage. A disjoint confirmation on
seeds `100018–100019` improved current RMSE from `1.244437` to `1.165912 m`,
velocity from `0.996642` to `0.889775 m/s`, and 0.10–0.75-second forecasts by
`3.4–9.0%`; one-second error was unchanged. Nominal 90% coverage worsened on
that harder pair. This selected a denser historical policy, not the corrected
exact three-frame cadence; the 3 August protocol-11 audit supersedes the
counter semantics.

Closed-loop window sampling previously allowed late collision-conditioned
windows with no valid future horizon. One such step consumed `172 s` with
`loss_rollout=0`. The sampler now guarantees that at least one anchor supports
the shortest configured forecast. The next sampler-corrected steps had
nonzero position and velocity rollout losses. At step 8, a paired evaluation
showed small improvements at current state and every horizon, unchanged
detection/event/identity counts, and 90% coverage improving from `0.737805` to
`0.739837`; this justified the ongoing extension to step 32 but is not yet a
meaningful convergence claim.

Primary artifacts:

- transferred checkpoint:
  `runs/20260728-223558-scaled-fixed-scale-transfer-1k-v1/checkpoints/best_measurement.pt`;
- fixed-scale base report:
  `runs/20260728-223558-scaled-fixed-scale-transfer-1k-v1/evaluation/20260728-225136-fixed-scale-select2-offset16/report.md`;
- historical cadence-three reports (actual cadence four):
  `runs/20260728-231250-scaled-global-cadence3-ablation-v1/evaluation/20260728-232212-global-cadence3-select2-offset16/report.md`
  and
  `runs/20260728-231250-scaled-global-cadence3-ablation-v1/evaluation/20260728-233559-global-cadence3-confirm2-offset18/report.md`;
- sampler-corrected step-8 report:
  `runs/20260728-235003-scaled-fixed-cadence3-causal-valid-v2/evaluation/20260729-001101-causal-step8-select2-offset16/report.md`.

The main fixed-scale transfer command was:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/scaled_curriculum.yaml \
  --initialize-from runs/20260728-152237-scaled-longer-stable-v2/checkpoints/best_measurement.pt \
  --run-name scaled-fixed-scale-transfer-1k-v1 --device mps --seed 47 \
  --set simulator.sequence_frames=24 --set training.steps=256 \
  --set training.rgb_pretrain_steps=256 --set training.batch_size=4 \
  --set training.train_episodes=4096 --set training.validation_episodes=8 \
  --set training.num_workers=4 --set training.learning_rate=0.00001 \
  --set training.eval_every=64 --set training.checkpoint_every=64 \
  --set training.log_every=8 --set evaluation.episodes=2
```

All full online reports above were RGB-only, used MPS, and recorded
`oracle_runtime_input_used=false`. The selection/confirmation sample is four
episodes, not the required wider validation/test manifest.

The sampler-corrected continuation was extended to a safe step-16 checkpoint.
Steps 10–16 took about 35 minutes on MPS; individual two-step graphs ranged
from roughly 6.5 to 10 minutes depending on object/interaction density. Step 16
slightly improved current position and velocity versus the unchanged
historical cadence-three weights (actual cadence four), but regressed
0.25/0.50/0.75/1.00-second RMSE from
`0.626639/0.650438/0.773491/1.007125 m` to
`0.627064/0.652029/0.776092/1.008927 m`. Detection, collision F1, and identity
counts were unchanged. It is rejected, and the interrupted target of step 32
is not reported as completed. The exact report is
`runs/20260729-001136-scaled-fixed-cadence3-causal-valid32-v3/evaluation/20260729-005424-causal-step16-select2-offset16/report.md`.

Final repository validation for this change:

```bash
PYTHONPATH=. conda run -n orpheus ruff format --check .
PYTHONPATH=. conda run -n orpheus ruff check .
PYTHONPATH=. conda run -n orpheus pytest
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest \
  tests/integration/test_rgb_measurements.py::test_roi_sampling_mps_training_cpu_fallback_is_differentiable \
  tests/unit/test_association.py::test_association_transfers_cost_to_cpu_without_mps_float64 \
  tests/unit/test_evaluation_parameter_update_metrics.py::test_directional_parameter_metrics_transfer_before_float64_accumulation \
  tests/unit/test_modal_dynamics.py::test_modal_device_when_available -q
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-accuracy-pycache \
  PYTHONPATH=. conda run -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
git diff --check
```

Ruff passed. The sandboxed full suite reported
`243 passed, 4 skipped in 101.85 s`; the four hardware-conditional tests then
passed directly on MPS (`4 passed in 2.49 s`). Compileall and
`git diff --check` passed. A first direct-MPS selector command named a stale
test and collected nothing; the corrected command above is the executed
hardware result.

## 2026-07-29 — persistent point/scale trajectory depth correction

The historical cadence-three (actual cadence-four) scaled observer exposed a
structural history limitation:
the existing five-frame point/velocity ring could never retain three global
scale measurements because the two intervening centre-only ROI frames evicted
the previous scale anchor. The RGB observer now keeps two bounded rings per
persistent ID:

- every reliable associated frame contributes a point sample for the existing
  axis-local velocity estimate;
- only nonambiguous global silhouettes with trustworthy scale contribute to a
  separate scale-anchor ring;
- image-boundary-truncated and overlap-split components retain their accurate
  RGB centres but cannot become scale anchors;
- a three-iteration Huber/IRLS inverse-variance line is fit independently per
  world axis and evaluated at the current timestamp;
- the resulting position evidence is projected onto calibrated camera depth
  by default, carries explicit variance/validity, and corrects `WorldBelief`
  through the ordinary robust diagonal filter.

No model weights, simulator state, future RGB, or history re-encoding are used.
The runtime-ablation checkpoint contains the unchanged historical
cadence-three weights (actual cadence four) and explicit new configuration
semantics:

`runs/20260729-084712-scaled-point-scale-trajectory-v1/checkpoints/runtime_ablation.pt`.

Paired MPS results:

| Evidence | Current RMSE m | Velocity RMSE m/s | 0.10 / 0.25 / 0.50 / 0.75 / 1.00 s RMSE m | F1 | Detection R/P | Coverage 90 |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| offset-16 historical cadence-three baseline (actual four) | `0.906217` | `1.082334` | `0.780533 / 0.626639 / 0.650438 / 0.773491 / 1.007125` | `0.357143` | `0.694444 / 0.568182` | `0.737805` |
| offset-16 trajectory candidate | `0.684258` | `1.148529` | `0.698125 / 0.526197 / 0.517001 / 0.562448 / 0.618672` | `0.271186` | `0.687500 / 0.480583` | `0.780204` |
| offset-18 historical cadence-three baseline (actual four) | `1.165912` | `0.889775` | `1.010213 / 0.877051 / 0.922522 / 0.988420 / 1.267293` | `0.190476` | `0.391667 / 0.345588` | `0.554667` |
| offset-18 quality-gated confirmation | `0.804367` | `0.986646` | `0.802223 / 0.630088 / 0.644967 / 0.693634 / 0.760509` | `0.078431` | `0.675000 / 0.455056` | `0.722522` |

The position result repeats strongly: current RMSE improves by `24.5%` and
`31.0%`; one-second RMSE improves by `38.6%` and `40.0%`. Every declared
position horizon improves on both disjoint blocks. Confirmation also improves
detection and calibration with zero identity switches. This removes much of
the measured persistent monocular depth/tracker ceiling and is enabled in
`configs/scaled_curriculum.yaml`.

This is deliberately a position-accuracy promotion, not an overall event-model
claim. Velocity RMSE regresses by `6.1%` and `10.9%`, and collision F1
regresses on both two-episode blocks. The next concrete limitation is
event-conditioned outgoing velocity/contact classification under the improved
depth state, followed by wider per-scenario confirmation. The sample remains
four validation episodes and is not a reserved-test acceptance result.

Exact completed MPS command shape:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python evaluate.py \
  --config configs/scaled_curriculum.yaml \
  --checkpoint \
    runs/20260729-084712-scaled-point-scale-trajectory-v1/checkpoints/runtime_ablation.pt \
  --split validation --seed-protocol fresh_validation \
  --seed-offset <16-or-18> --device mps \
  --set evaluation.episodes=2 \
  --set model.rgb.temporal_position_enabled=true \
  --set model.rgb.temporal_position_min_samples=3 \
  --set model.rgb.temporal_position_robust_threshold=2.0 \
  --set model.rgb.temporal_position_variance_scale=8.0 \
  --set model.rgb.temporal_position_variance_floor=0.04 \
  --set model.rgb.temporal_position_variance_ceiling=0.5 \
  --set model.rgb.temporal_position_depth_only=true \
  --output <timestamped-output-label>
```

Reports:

- final-source selection:
  `runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-100628-select2-offset16-final-v5/report.md`;
- initial selection diagnostic:
  `runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-090821-select2-offset16-v2/report.md`;
- exact quality-gate diagnostic on the same selection block:
  `runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-092718-select2-offset16-quality-v3/report.md`;
- disjoint confirmation:
  `runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-094349-confirm2-offset18-quality-v4/report.md`.

An initial evaluation was stopped after discovering the mixed-ring cadence
impossibility; it produced no report and is not evidence. A later diagnostic
that changed ordinary single-frame depth semantics was also rejected as
confounded; ordinary checkpoint measurement behavior was restored, while the
stricter quality mask remains limited to the new scale-anchor ring.

Final validation for the committed source:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff format .
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff check .
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest
```

One Python file was mechanically formatted, Ruff passed, and Pytest reported
`250 passed, 4 skipped in 116.45 s`. The skips were the four
hardware-conditional MPS tests. Running those exact tests directly where MPS
was available reported `4 passed in 3.28 s`.

```bash
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-point-scale-pycache PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/scaled_curriculum.yaml --dry-run --device cpu
git diff --check
```

Compileall and `git diff --check` passed. The dry run resolved the
1.90M-parameter-contract scaled RGB-only 48,000-draw/eight-scenario plan with
the new observer configuration, Python `3.10.20`, PyTorch `2.10.0`, and MPS
built. The sandboxed dry-run process reported MPS unavailable and therefore
used explicit CPU; the paired evaluations and direct device tests above ran on
MPS.

## 2026-07-29 — acceleration-aware outgoing-velocity investigation

The remaining paired velocity regression is concentrated on the gravity axis.
The temporal observer now has an opt-in causal fit that removes the known
quadratic acceleration about the packet timestamp before estimating current
velocity. Its correction subspace contains calibrated camera-lateral motion
and, only after an event reset, the gravity axis. Camera-depth velocity remains
unobserved. The scaled default keeps this option off.

Focused validation passed `21` tests before the final contact diagnostic
(`tests/unit/test_rgb_temporal_history.py` plus checkpoint roundtrip). Matched
one-episode MPS selection on seed `100016` gave:

| Policy | Current RMSE m | Velocity RMSE m/s | Vertical RMSE m/s | 0.10 / 0.25 / 0.50 / 0.75 / 1.00 s RMSE m | Collision F1 |
| --- | ---: | ---: | ---: | --- | ---: |
| validated lateral-only baseline | `0.648034` | `1.288819` | `1.965171` | `0.661424 / 0.524227 / 0.568464 / 0.662076 / 0.751615` | `0.285714` |
| lateral + gravity | `0.662815` | `1.150215` | `1.654356` | `0.672865 / 0.537488 / 0.579988 / 0.671576 / 0.759725` | `0.235294` |
| conservative variance | `0.656559` | `1.215172` | `1.814179` | `0.666575 / 0.530451 / 0.572762 / 0.665180 / 0.753400` | `0.250000` |
| post-event endpoint collision only | `0.648034` | `1.288819` | `1.965171` | identical to baseline | `0.285714` |
| endpoint contact onset diagnostic | `0.647704` | `1.288726` | `1.964988` | `0.660803 / 0.524367 / 0.568098 / 0.662317 / 0.751496` | `0.266667` |

The continuous policies prove that acceleration-aware RGB slope contains useful
vertical signal, but both fail the position/event promotion gate. The
post-event policy is inert because `COLLISION` is an interval event and rarely
survives as the endpoint mode at an observation. Treating all endpoint contacts
as event onsets produces only negligible mixed changes and one extra event
false positive. It is rejected. The next concrete task is a causal,
RGB-trajectory-residual change-point detector trained on balanced
contact/no-contact windows.

Primary reports:

- baseline:
  `runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-134615-gravity-axis-offset16-baseline/report.md`;
- unrestricted gravity:
  `runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-132941-gravity-axis-offset16-selection/report.md`;
- conservative gravity:
  `runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-140223-gravity-axis-conservative-offset16-selection/report.md`;
- endpoint-collision-only:
  `runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-142009-post-event-gravity-offset16-selection/report.md`;
- endpoint-contact diagnostic:
  `runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-143722-contact-onset-gravity-offset16-selection/report.md`.

Final validation for this change:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff format .
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff check .
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest
```

Ruff formatted two files and passed. The sandboxed full suite reported
`252 passed, 4 skipped in 190.54 s`; the four hardware-conditional tests then
passed directly on MPS (`4 passed in 6.38 s`).

```bash
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-acceleration-aware-pycache \
  PYTHONPATH=. conda run --no-capture-output -n orpheus python \
  -m compileall -q world_model train.py evaluate.py demo.py scripts tests
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/scaled_curriculum.yaml --dry-run --device cpu
git diff --check
```

Compileall and `git diff --check` passed. The dry run resolved the unchanged
1.90M-parameter-contract, 48,000-draw, eight-scenario RGB-only plan with Python
`3.10.20`, PyTorch `2.10.0`, and MPS built. The sandboxed dry run used explicit
CPU; the evaluations and direct device tests above ran on MPS.

## 2026-07-29 — causal RGB trajectory change-point investigation

The RGB temporal observer now has an opt-in causal three-point discontinuity
detector. It removes the segment-velocity change explained by known gravity,
projects only onto declared observable axes, and can reopen the point ring
without discarding the slower scale-anchor ring. Reset provenance prevents a
gravity change point from accidentally exposing lateral or monocular-depth
evidence. A separate two-sample acceleration-aware fit permits earlier
outgoing gravity velocity after a validated event. Runtime measurement
auxiliary data and evaluation reports expose trigger masks, scores, counts,
and rates. No simulator state enters this path.

The scaled default remains disabled. On MPS seed `100016`, the permissive gate
improved current RMSE `0.648034 → 0.637888 m` but regressed velocity
`1.288819 → 1.357281 m/s`, vertical velocity
`1.965171 → 2.085003 m/s`, 0.1-second prediction
`0.661424 → 0.654775 m`, and collision F1
`0.285714 → 0.250000`, while firing `45/111` times. Conservative,
gravity-only, and provenance-decoupled variants also failed the joint gate.
Requiring a contact endpoint reduced triggers to `1/110` on seed `100016` and
`0/177` on seed `100017`; the seed-`100016` fast run was identical to baseline
on comparable state, detection, and 0.1-second metrics. That policy is safe but
does not address the missed between-frame event. The next concrete accuracy
task is a learned, uncertainty-aware gate trained on balanced contact and
no-contact RGB windows.

Primary reports:

- permissive:
  `runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-160516-rgb-change-point-offset16-selection/report.md`;
- conservative:
  `runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-161717-rgb-change-point-conservative-offset16-selection/report.md`;
- decoupled gravity-only correction:
  `runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-164508-rgb-change-point-decoupled-offset16-selection/report.md`;
- contact-gated seed `100016`:
  `runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-171503-rgb-change-point-two-sample-fast-offset16/report.md`;
- contact-gated seed `100017`:
  `runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-172205-rgb-change-point-two-sample-fast-offset17/report.md`.

Validation for the committed source:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff format .
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff check .
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest
```

Ruff left all 164 files unchanged and passed. The sandboxed full suite reported
`258 passed, 4 skipped in 118.68 s`; the four hardware-conditional tests then
passed directly on MPS (`4 passed in 3.62 s`). A focused temporal/config/
checkpoint run reported `71 passed in 3.90 s`.

```bash
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-change-point-pycache PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/scaled_curriculum.yaml --dry-run --device cpu
git diff --check
```

Compileall, the scaled dry run, and `git diff --check` passed. The dry run
resolved the unchanged 1.90M-parameter-contract, 48,000-draw, eight-scenario
RGB-only plan with Python `3.10.20`, PyTorch `2.10.0`, and MPS built. The paired
evaluations and direct device tests ran on MPS.

## 2026-07-29 — learned uncertainty-aware RGB event gate

A reproducible offline-supervised/online-cheap gate workflow now consumes the
exact three timestamps behind each causal RGB history feature. Simulator
collision and velocity state are used only to label those cached training
windows. Runtime input remains RGB, calibration, persistent belief, and
sensor-local history. The nine features include signed/absolute
acceleration-compensated residual, propagated-uncertainty standardized
residual, adjacent velocities, reversal, minimum speed, log variance, and
belief contact probability. Logistic regression or a one-hidden-layer MLP can
be refit from cached tensors in seconds, and coefficients are explicit
checkpoint-compatible runtime semantics. Artifact creation preserves the
source checkpoint's training and seed provenance.

The aligned eight-scenario MPS collection produced 543 training windows
(197 observable event positives) and 398 disjoint validation windows
(146 positives). The linear gate failed the useful precision/recall gate. An
eight-hidden-unit MLP at the loose threshold reached validation precision
`0.600000`, recall `0.164384`, and F1 `0.258065`; on seed `100016` it fired
13 times, collapsed detection recall `0.458333 → 0.208333`, and regressed
current/velocity RMSE to `0.699426 m` / `1.652333 m/s`. It was rejected.

The sparse MLP threshold reached precision `0.750000`, recall `0.041096`, and
F1 `0.077922`. It fired twice on each paired seed. Seed `100016` was exactly
baseline on current, velocity, detection, and 0.1-second metrics. On seed
`100017`, current RMSE changed `0.702313 → 0.702296 m` and 0.1-second RMSE
`0.715284 → 0.715256 m`, but velocity RMSE regressed
`1.148770 → 1.154865 m/s`. It is also rejected. The scaled runtime remains
unchanged and the learned gate stays disabled. The next concrete accuracy task
is a jointly calibrated outgoing-velocity proposal rather than further gate
threshold tuning.

Primary artifacts:

- aligned cached dataset and linear fit:
  `runs/20260729-214956-rgb-change-point-linear-aligned-8x8-v2/`;
- loose MLP fit:
  `runs/20260729-221614-rgb-change-point-mlp-aligned-8x8-v4/`;
- loose paired seed `100016`:
  `runs/20260729-221614-rgb-change-point-mlp-aligned-8x8-v4/evaluation/20260729-222116-mlp-fast-offset16/report.md`;
- sparse MLP fit:
  `runs/20260729-222143-rgb-change-point-mlp-aligned-precision70-v5/`;
- sparse paired seeds `100016` and `100017`:
  `runs/20260729-222143-rgb-change-point-mlp-aligned-precision70-v5/evaluation/20260729-222606-mlp-fast-offset16/report.md`
  and
  `runs/20260729-222143-rgb-change-point-mlp-aligned-precision70-v5/evaluation/20260729-223027-mlp-fast-offset17/report.md`.

Final validation:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff format .
PYTHONPATH=. conda run --no-capture-output -n orpheus python -m ruff check .
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest
```

Ruff left all 167 files unchanged and passed. The sandboxed full suite reported
`262 passed, 4 skipped in 94.45 s`; the four hardware-conditional tests passed
directly on MPS (`4 passed in 2.74 s`). The focused gate/temporal/config/
checkpoint suite reported `75 passed in 3.95 s`.

```bash
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-learned-gate-pycache PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/scaled_curriculum.yaml --dry-run --device cpu
git diff --check
```

Compileall, the unchanged 48,000-draw/eight-scenario scaled dry run, and
`git diff --check` passed. Feature collection and paired evaluations ran on
MPS; cached logistic/MLP fitting ran on CPU.
