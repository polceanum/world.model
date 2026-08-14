# Project status

**Date:** 2026-08-13
**Specification:** `PROJECT_SPEC.md` 1.44; the active immutable relation-only
campaign uses specification 1.42 and the rejected schedule control uses 1.41

## Latest verified state — 2026-08-13

### New paper-guided iteration — selector seam implemented

The prior sustained relation-only campaign was stopped on user request; its
checkpoints and evidence remain immutable. A new implementation slice is now
available in `world_model/dynamics/hypothesis_rollout.py`. It runs any set of
existing `RolloutStep` predictors from the same cloned `WorldBelief`, scores
their short-step trajectories against asynchronous/occluded target frames with
masked position NLL, and returns per-batch posterior weights plus a deterministic
selected hypothesis. This is an executable model-pool/selection contract, not
yet a promoted accuracy result or a replacement for the protected incumbent.

Focused verification:

```bash
PYTHONPATH=. conda run -n orpheus pytest -q tests/unit/test_hypothesis_rollout.py
# 3 passed
conda run -n orpheus python -m compileall -q world_model/dynamics/hypothesis_rollout.py tests/unit/test_hypothesis_rollout.py
git diff --check
```

The next experiment must connect this selector to a fixed small candidate pool
and compare it against the incumbent on the full 32-episode, every-axis,
every-horizon protocol before any training campaign is resumed.

The selector now also exposes a structural `rollout_dynamics` adapter for any
candidate object with `predict_step`, verified with two independent
`DynamicsModel` instances. No long run has been relaunched yet.

The persistent `HypothesisDynamicsPool` is now implemented on top of that
adapter. It retains normalized evidence weights across cycles, supports late
assimilation of masked observations, and reports the selected candidate while
leaving the source belief untouched. A fixed-candidate synthetic test selects
the correct model after delayed evidence; this is selector functionality, not
an accuracy claim for RGB scenarios.

`OnlineWorldModel.predict_hypotheses` and `assimilate_hypotheses` now expose the
pool through the live runtime while keeping the pool injected and the belief
authoritative. This is ready for a protected evaluation experiment; no
candidate has been promoted and no training has restarted.

Full regression verification after the runtime integration:

```bash
PYTHONPATH=. conda run -n orpheus pytest -q
# 744 passed, 6 skipped in 483.36s
```

The six skips are expected tests gated on unavailable MPS hardware in the test
process; no failures occurred.

The selector audit also fixed posterior/instantaneous-choice divergence:
`HypothesisDynamicsPool` now reports the posterior argmax after accumulated
evidence, with a regression test proving that a single later observation cannot
erase a stronger prior without sufficient evidence.

The pool now includes a transparent `ConstantVelocityDynamics` candidate for
heterogeneous model comparisons. It advances active positions from velocity,
supports optional exponential damping, propagates uncertainty, and leaves the
source belief unchanged. It is a baseline hypothesis, not a hard-coded rule
for promotion.

The first real RGB pool smoke caught and fixed an uncertainty-shape broadcast
bug in that baseline (`[B,N,D]` variance was accidentally expanded to
`[B,1,N,D]`). The trajectory validator caught it before any result was
recorded; the corrected implementation is covered by the focused tests below.

After the fix, a one-frame `configs/toy_smoke.yaml` RGB smoke (seed `100000`,
CPU, two 0.05/0.10-second queries) completed end to end. Against simulator
future state used only for evaluation, the learned dynamics candidate scored
`2.126778` and the damped constant-velocity candidate `8.872571`; posterior
weights were `0.998826/0.001174` and candidate `0` was selected. This is a
plumbing sanity check, not a multi-episode accuracy result.

The reusable evaluation harness is `scripts/evaluate_hypothesis_pool.py`. Its
two-episode aligned toy report is
`runs/20260813-215846-hypothesis-pool-toy-2/report.json`. It uses RGB for
runtime state and aligns simulator supervision by persistent object ID. The
learned candidate was selected on all 118 scored frame/horizon queries in this
small run. Selected x/y/z RMSE was `0.3393/0.3236/1.1076 m` and
`0.4225/0.3143/1.1265 m` at 0.10 s for the two episodes; at 1.00 s it was
`0.4141/0.2943/0.9345 m` and `0.4991/0.0523/0.8509 m`. These numbers are
random-initialization toy evidence only, not an incumbent qualification.

The same harness completed a one-episode attention-scale RGB smoke at
`runs/20260813-220000-hypothesis-pool-attention-smoke/report.json` on CPU
(148 scored queries). The learned candidate was selected throughout; selected
position RMSE x/y/z was `0.7964/0.5686/1.1904 m` at 0.10 s and
`1.2472/0.3351/1.1508 m` at 1.00 s. This used fresh random weights and is only
an architectural execution check, not evidence of convergence.

The protected-checkpoint comparison also identified a real selection limitation:
with the default evidence decay of `1.0`, the learned candidate stayed selected
for all 148 queries even though the constant-velocity candidate had slightly
lower x error at some horizons. `HypothesisDynamicsPool` now exposes explicit
`evidence_decay` in `(0,1]`; a focused test proves that decay permits local model
switching while `1.0` preserves persistent accumulation. The default remains
`1.0` until a fixed-decay comparison is run.

An attention-scale protected-checkpoint smoke with `evidence_decay=0.1` is
retained at
`runs/20260813-220845-hypothesis-pool-protected-decay01/report.json`. It
selected the learned candidate 144/148 times and the constant-velocity
candidate 4/148 times. Selected x/y/z RMSE was
`0.8146/0.5221/0.6365 m` at 0.10 s and `1.0867/0.3322/0.6683 m` at 1.00 s,
versus 148/148 learned selections with decay 1.0. This demonstrates adaptive
selection behavior, but it is a one-episode smoke and not a promotion gate.

The immutable relation-only campaign has completed and rejected its fixed
step-1024 selector. The 32 RGB-only validation episodes completed in
`1,236.713 s` with four balanced repeats of every scenario and zero mutable or
protected training-support failures. Candidate score is `0.3409900` versus
the protected step-zero incumbent's `0.3213162`; 116 incumbent/reference
guardrails fail, so `selection_accepted=0` and both protected checkpoints
remain bitwise at step zero. Current position worsens
`0.251460 -> 0.313353 m`; x is the dominant regression
(`0.281775 -> 0.402440 m`), z also worsens
(`0.263691 -> 0.303771 m`), while y is essentially flat/slightly better
(`0.201906 -> 0.200839 m`) and velocity improves
`1.093191 -> 1.048411 m/s`. Pooled 0.10/0.25/0.50/0.75/1.00-second position
worsens by `0.057141/0.042869/0.019589/0.013925/0.000434 m`; the nearly flat
one-second result is insufficient because current and short-horizon behavior
regress broadly. Target coverage and prediction precision fall by
`0.017000/0.017033`, identity switching rises
`0.013592 -> 0.021456`, while collision F1 and coverage90 improve slightly.
The worst slice is `reference_pairs`: current x reaches `0.853208 m` versus
`0.242694 m`, and every x horizon fails. This is clear non-convergence at the
selector boundary, not a promotable long-horizon tradeoff.

The candidate itself passes the strict structural audit at embedded,
expected, and Adam step `1024`. All 46 permitted relation-path tensors changed
and own complete optimizer state; both frozen node tensors, all 177 inherited
tensors, and both protected checkpoints remain exact; provenance, protocol,
architecture, and serialized finiteness pass. Candidate SHA-256 is
`10384a797922ec71a19cf1fa12c44718e235cb2c1660ceed2af91c7ea5b618b6`
and model-state hash is
`a0a4f2cf867696ef471a1a794c269380d8f4a17ed5c487501d86f3967bdc4136`.
The durable artifact is `checkpoints/validation_step_001024.pt`; its audit is
`attention_checkpoint_audit_step_001024.json` in the active run.

The complete non-overlapping updates 968--1024 window also passes all
operational gates: 64 applied updates, exact eight-way scenario balance, all
13 causal terms in every logged block, 2,352 trajectories, zero skipped draws,
no uncontained interaction clipping, and flat RSS at `2,924,761,088` bytes.
Minimum complete interaction-gradient retention is `0.150987` at the
force-dominated hard-contact step 992. Similar force sensitivities at
984/1000 are locally and globally bounded, then recover to a fully retained
`0.325660` norm at step 1024; there is no numerical collapse or resource leak.
The fixed selector nevertheless proves that the learned trajectory has not
generalized. Keep the protected incumbent deployed and continue the declared
mutable 8,192-step evidence campaign unchanged so later fixed selectors can
distinguish delayed convergence from persistent relation-objective drift.

Commands run for this boundary were:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 960 --trend-window-blocks 8
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_attention_checkpoint.py --checkpoint runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/validation_step_001024.pt --initial-checkpoint runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/validation_step_000000.pt --config runs/20260813-073710-attention-relation-constant-stage-a/config.resolved.yaml --protected runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/best_rollout.pt --protected runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/reference_rollout.pt --output runs/20260813-073710-attention-relation-constant-stage-a/attention_checkpoint_audit_step_001024.json --expected-step 1024 --frozen-attention-prefix dynamics.attention_interactions.node_decoder. --require-all-attention-changed --require-complete-attention-optimizer-state --require-protected-checkpoints
```

The complete relation-only updates 904--960 window passes operationally. All
64 updates apply, every scenario appears exactly eight times, all 13 causal
terms contribute in every cadence block, 2,943 causal trajectories are
sampled, no draw is skipped, no clipping is uncontained, and RSS remains flat
at `2,924,761,088` bytes. Minimum complete interaction retention is
`0.370484` at the high-support step 960, whose raw `2.699174` norm is
force-dominated and safely bounded. No failure, rollback, or persistent
resource/gradient anomaly follows.

Absolute current position/velocity RMSE is
`0.292962 m / 1.640873 m/s`; x/y/z is
`0.322173/0.246964/0.304457 m`. Position RMSE at
0.10/0.25/0.50/0.75/1.00 seconds is
`0.294236/0.334889/0.395318/0.432672/0.440859 m`, with target coverage
`100.00/100.00/97.66/97.66/95.33%`. Against the preceding different-draw
840--896 window, current position/velocity and all current axes worsen, and
0.10/0.25-second position worsens `0.036690/0.011233 m`. Conversely,
0.50/0.75/1.00-second position improves
`0.020446/0.025633/0.044971 m`, velocity at 0.25--1.00 seconds improves
`0.015321--0.263036 m/s`, collision F1 improves `0.028803`, and lifecycle
precision/coverage improve `0.055613/0.087440`. Identity, coverage90, and
median NLL remain adverse. This horizon-dependent, draw-confounded reversal is
not convergence or collapse; continue unchanged through the final predeclared
step-1024 selector.

Commands run for this window were:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 896 --trend-window-blocks 8
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 832 --trend-window-blocks 8
```

The relation-only step-896 structural boundary is preserved and passes the
strict audit. `last.pt` first passed at embedded/expected/Adam steps
`896/896/[896]`; those verified bytes were copied to
`checkpoints/checkpoint_step_000896.pt` and independently re-audited. Both
files have SHA-256
`2aaf6dd75fb733b833e230ace224d9834816a2dcfcad2166afdf757cd2e2050c`;
the model-state hash is
`1667b8a5bb35c72a56e88119f6c8a78b085d24e86a820feac7010a83ce62a54e`.
All 46 permitted attention tensors and exactly 46 Adam owners are live, both
frozen node tensors and all 177 inherited tensors remain exact, both protected
incumbents remain exact, and finiteness/source/protocol checks pass. The
durable audit is `attention_checkpoint_audit_step_000896.json` in the active
run.

The complete 840--896 window passes operationally: 64 applied updates, exact
eight-way balance, 2,163 causal trajectories, no skipped draws, no uncontained
clipping, and flat RSS at `2,924,761,088` bytes. Minimum complete interaction
retention is `0.208007` at the hard-contact step 848. Its raw norm `4.807532`
is force-dominated (`4.517561`), locally capped before the complete bounded
update, and does not recur at 856/864. Step 872 has a raw force-output
sensitivity of `2.884494` reduced to `0.044487` before parameter backprop;
the resulting complete norm is only `0.517666` and wholly retained. Both
draws contain substantial collision support and remain finite/applied rather
than indicating optimizer collapse.

Absolute current position/velocity RMSE is
`0.244778 m / 1.245206 m/s`; current x/y/z is
`0.278787/0.197755/0.250839 m`. Position RMSE at
0.10/0.25/0.50/0.75/1.00 seconds is
`0.257545/0.323656/0.415764/0.458305/0.485830 m`. Against the preceding
different-draw 776--832 window, current velocity improves `0.164184 m/s`, y/z
improve `0.023453/0.033980 m`, trusted switch rate improves `0.009617`, and
median NLL improves `0.023452`. The pre-selector limitation is increasingly
mature position: x worsens `0.058509 m`, and pooled position worsens
`0.014035/0.043259/0.074097/0.080967/0.090517 m` across 0.10--1.00 seconds.
Collision F1 falls `0.076792`, lifecycle precision/coverage fall
`0.017474/0.036342`, and one-second target coverage falls `0.041234`.
Different draws prevent rejection, but this is the explicit long-horizon/event
watch signal the fixed step-1024 manifest must adjudicate. Continue unchanged.

Commands run for this boundary were:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_attention_checkpoint.py --checkpoint runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/checkpoint_step_000896.pt --initial-checkpoint runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/validation_step_000000.pt --config runs/20260813-073710-attention-relation-constant-stage-a/config.resolved.yaml --protected runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/best_rollout.pt --protected runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/reference_rollout.pt --output runs/20260813-073710-attention-relation-constant-stage-a/attention_checkpoint_audit_step_000896.json --expected-step 896 --frozen-attention-prefix dynamics.attention_interactions.node_decoder. --require-all-attention-changed --require-complete-attention-optimizer-state --require-protected-checkpoints
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 832 --trend-window-blocks 8
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 768 --trend-window-blocks 8
```

The complete non-overlapping relation-only updates 776--832 window passes all
operational gates. All 64 optimizer updates apply, every scenario appears
exactly eight times, 2,578 causal trajectories are sampled, no draw is
skipped, no interaction clipping is uncontained, and RSS remains flat at
`2,924,761,088` bytes. Minimum complete interaction-gradient retention is
`0.801737` at step 816. Strong local collision-output isolation at steps
792/808/816/832 remains contained before shared backpropagation. Step 800 has
eight contributing relation-scope causal terms with 233 trajectory supports,
a finite applied update, full gradient retention, and targets rather than a
causal dropout.

Absolute current position/velocity RMSE is
`0.243979 m / 1.409390 m/s`; current x/y/z is
`0.220279/0.221208/0.284819 m`. Position RMSE at
0.10/0.25/0.50/0.75/1.00 seconds is
`0.243510/0.280397/0.341667/0.377338/0.395313 m`, with target coverage
`100.00/98.31/98.31/98.31/98.08%`. Collision F1 is `0.217143`, trusted
identity switching is `7/355`, lifecycle precision/coverage is
`0.366899/0.389831`, current coverage90 is `0.979842`, and median uncertainty
NLL is `-0.761216`.

Against the preceding different-draw 712--768 window, current position and
velocity worsen `0.056932 m / 0.239553 m/s`; x/y/z and every position horizon
worsen, by `0.058629/0.068326/0.072813/0.051028/0.037679 m` pooled over
0.10--1.00 seconds. Trusted switch rate, collision F1, coverage90, lifecycle
precision, and median NLL are also adverse. Conversely, 0.25/1.00-second
velocity improve `0.746478/0.119993 m/s`, lifecycle target coverage improves
`0.027651`, and forecast target coverage improves by `2.14--4.91` percentage
points at every horizon. This heterogeneous draw-confounded reversal is a
watch signal, not fixed-manifest regression evidence. Keep the immutable run
unchanged through checkpoint 896 and selector 1024.

Commands run for this window were:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 768 --trend-window-blocks 8
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 704 --trend-window-blocks 8
```

The relation-only step-768 structural boundary is preserved and passes the
strict checkpoint audit. `last.pt` first passed at embedded/expected/Adam
steps `768/768/[768]`; its verified bytes were then copied to
`checkpoints/checkpoint_step_000768.pt` and independently re-audited. Both
files have SHA-256
`d2e1d85553949e9d3c32fed10383ad10e82818105740cceec1262d3cbde1e3e0`;
the model-state hash is
`43b6b1a69cd004184d00f4b32390f98097e24d9881ec64641a85a1a0299969d3`.
All 46 permitted attention tensors and exactly 46 Adam owners are live, both
frozen node tensors and all 177 inherited tensors remain exact, both protected
incumbents remain exact, and serialized finiteness/source/protocol checks pass.
The durable audit is `attention_checkpoint_audit_step_000768.json` in the
active run.

The complete non-overlapping 712--768 window also passes operationally: all
64 updates apply, all eight scenarios appear exactly eight times, 2,081 causal
trajectories are sampled, every cadence block has all 13 contributing causal
terms, no draw is skipped, no clipping is uncontained, and RSS remains flat at
`2,924,761,088` bytes. Minimum complete interaction-gradient retention is
`0.395006` at step 744. Repeated strong local collision-output isolation is
contained before shared backpropagation and does not cause rollback, missing
support, nonfinite state, or resource growth.

Absolute current position/velocity RMSE is
`0.187047 m / 1.169836 m/s`, current x/y/z is
`0.188838/0.172034/0.199259 m`, and 0.10/0.25/0.50/0.75/1.00-second position
is `0.184881/0.212071/0.268854/0.326310/0.357634 m`. Against the preceding
different-draw 648--704 window, current position/velocity improve
`0.184067 m / 0.317872 m/s`, all current axes improve, and every position
horizon improves by `0.183748/0.174085/0.081119/0.017418/0.030332 m`.
Collision F1, trusted switch rate, lifecycle precision/coverage, coverage90,
and median NLL also improve by `0.059748`, `0.015499`,
`0.034621/0.005361`, `0.029725`, and `0.146744`. The remaining diagnostic
watch items are velocity at 0.25/0.50/1.00 seconds, adverse by
`0.644878/0.121007/0.026917 m/s`, and target coverage at 0.25--1.00 seconds,
lower by `1.57--3.75` percentage points. These draws differ, so this is
encouraging trend evidence rather than promotion; continue unchanged to fixed
selector 1024.

Commands run for this boundary were:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_attention_checkpoint.py --checkpoint runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/checkpoint_step_000768.pt --initial-checkpoint runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/validation_step_000000.pt --config runs/20260813-073710-attention-relation-constant-stage-a/config.resolved.yaml --protected runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/best_rollout.pt --protected runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/reference_rollout.pt --output runs/20260813-073710-attention-relation-constant-stage-a/attention_checkpoint_audit_step_000768.json --expected-step 768 --frozen-attention-prefix dynamics.attention_interactions.node_decoder. --require-all-attention-changed --require-complete-attention-optimizer-state --require-protected-checkpoints
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 704 --trend-window-blocks 8
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 640 --trend-window-blocks 8
```

The complete non-overlapping relation-only updates 648--704 window passes the
operational audit. All 64 optimizer updates apply, every one of the eight
scenario families appears exactly eight times, 2,130 causal trajectories are
sampled, no draw is skipped, no interaction clip is uncontained, and process
RSS remains flat at `2,924,761,088` bytes. The minimum complete interaction
gradient retention is `0.368333` at step 648; subsequent locally severe
collision-output isolation at steps 664/680/688/696/704 remains contained
before shared backpropagation, with complete-update retention
`1.0/0.704155/1.0/1.0/1.0`. The step-672 minimum of eight contributing causal
objective terms is supported rather than dropout: it has 269 causal
trajectories, finite applied gradients, and forecast targets at every horizon,
and the preceding healthy window has the same minimum under relation-only
scope.

The window's absolute pooled current position/velocity RMSE is
`0.371113 m / 1.487709 m/s`; x/y/z current position is
`0.369088/0.199405/0.487019 m`. Position RMSE at
0.10/0.25/0.50/0.75/1.00 seconds is
`0.368629/0.386156/0.349974/0.343729/0.387966 m`, target coverage is
`97.73/97.73/97.33/96.91/96.91%`, collision F1 is `0.166667`, trusted
identity switching is `6/322`, lifecycle precision/coverage is
`0.338727/0.356818`, current coverage90 is `0.963145`, and median uncertainty
NLL is `-0.773368`.

A different-draw within-candidate comparison against 584--640 is diagnostic
only. It improves 0.50/0.75/1.00-second position by
`0.044099/0.108650/0.114204 m`, collision F1 by `0.064626`, lifecycle
precision/coverage by `0.052528/0.068005`, coverage90 by `0.004050`, and median
NLL by `0.132954`. It worsens current position/velocity by
`0.063320 m / 0.416573 m/s`, 0.10/0.25-second position by
`0.065659/0.038789 m`, trusted switch rate by `0.010912`, and current x/z by
`0.048677/0.151647 m` while y improves `0.063414 m`. This heterogeneous,
draw-confounded movement is neither collapse nor convergence and does not
authorize promotion or retuning. Continue the immutable trajectory to the
fixed step-1024 selector.

Commands run for this boundary were:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 640 --trend-window-blocks 8
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 577 --trend-window-blocks 8
```

The rejected warmup/cosine trainer and supervisor are no longer loaded after
the host pause.  The trainer log ends at step 592, and step 512 remains its
latest durable selector checkpoint; there is no competing training process.

Two new exact 32-episode RGB-only modular evaluations close the post-hoc
zero-node question.  Importing the 46 learned shared/relation tensors from the
warmup/cosine step-512 donor while preserving the protected zero node decoder
scores `0.3422885013` versus protected `0.3213162196`, fails 100 broad
guardrails, and has zero support failures. Current x worsens
`0.281775 -> 0.361844 m`, z `0.263691 -> 0.290008 m`, and every pooled
position horizon regresses, although y and velocity improve. The exact
constant-rate drift donor behaves similarly: score `0.3293166386`, 98 broad
guardrail failures, zero support failures, current x
`0.281775 -> 0.371597 m`, improved y/velocity, and only the 1.00-second pooled
horizon improves. Reports are
`runs/20260813-070400-attention-cosine-step512-zero-node-ablation/report.json`
and
`runs/20260813-071355-attention-constant-step512-zero-node-ablation/report.json`.
The first multiprocessing evaluator attempt made no progress with two sleeping
workers and was preserved with a `-stalled` suffix; the identical default
`num_workers=0` path completed normally in 538.91 seconds.

These results do not qualify either learned donor. They also show why a
post-hoc zero-node composition is not the same experiment as relation-first
training: both donors' shared stack was optimized through a nonzero node
decoder and node-drift/task gradients. The earlier no-drift complexity
checkpoint remains the only learned relation path with beneficial post-hoc
evidence (score `0.2973304139` and all pooled horizons improved, though 72
guardrails still failed). The next controlled run therefore uses constant-rate
`attention_relation` directly from the untouched graph control, so the node
decoder and its entire gradient path remain exactly zero from update zero.

That exact protocol now passes a real two-update CPU qualification at
`runs/20260813-072457-attention-relation-constant-cpu-smoke/`. Both balanced
updates have 343 causal trajectories, all 13 causal objective groups, no
skipped/no-gradient draw, finite applied gradients, and exact resume. Update 1
correctly changes only the zero-initialized relation decoder; update 2 sends
nonzero gradients through every shared relation component while node gradient,
activity, drift, and complexity remain exactly zero. The strict step-2 audit
passes with 46 changed permitted tensors, exactly 46 complete finite Adam
owners at step 2, two exact frozen node tensors with no optimizer state, all
177 inherited tensors exact, the protected reference exact, and all serialized
state finite. Checkpoint SHA-256 is
`c701ebd5efac6a143a8a7b8fa278674bfff5c571ca2cb9c6cac6f00bc306c18e`;
the audit is
`runs/20260813-072457-attention-relation-constant-cpu-smoke/attention_checkpoint_audit_step_000002.json`.
The eight-episode smoke selector is intentionally support-incomplete and is not
accuracy evidence. A full immutable 8,192-update/65,536-example campaign is
the next action; depth, width, history, and CUDA scaling remain closed.

That sustained campaign is now active at
`runs/20260813-073710-attention-relation-constant-stage-a/` from clean pushed
commit `c3fe110`. Trainer launchd label
`com.polceanum.orpheus.attention-relation-20260813-073710` runs once under
Standard/default priority and `caffeinate`; PID at launch audit was `47723`.
Metadata records Python 3.10.20, PyTorch 2.10.0, MPS built/available, RGB
measurement on MPS, closed-loop execution on CPU, float32, RGB-only runtime,
no oracle, and runtime fingerprint
`1ab6aeb933767dcbfe51aabdfa23075a91066a3d1753f5974823416e31904317`.
The resolved configuration records `attention_relation`, constant rate, 8,192
updates, 16,384 training episodes, batch eight, 65,536 nominal balanced draws,
and 32 fixed validation episodes.

The first supervisor launch exposed and prevented a future exact-resume bug in
the launch procedure: a plain source archive has no Git metadata, so
`capture_git_metadata()` would return null provenance and a later extension
would correctly refuse the checkpoint. The waiting supervisor was stopped
before it could resume anything, the archive was preserved with
`-archive-incomplete`, and the runtime path was replaced by a real detached Git
worktree at `c3fe110`. Its independently recomputed commit, dirty flag,
runtime-source fingerprint, and worktree fingerprint now exactly match the
trainer (`c3fe110`, false, `1ab6...4317`, `55a5...6d82`). The supervisor was
restarted once from that exact worktree under label
`com.polceanum.orpheus.attention-relation-convergence-20260813-073710`; PID at
the corrected audit was `48111`. It records the unchanged 8,192 minimum,
4,096-update extensions, four-selector/1% plateau rule, and 24,576 hard limit.
Both corrected stderr files are empty. The mandatory step-zero selector is in
progress; no optimizer update, learned accuracy result, or convergence claim
exists yet.

The step-zero selector subsequently completed in `834.99 s` and exactly
reproduced the protected control: all 225 model tensors and all 2,584
checkpoint metrics are bitwise equal, including score `0.3213162196`, every
axis/horizon, lifecycle, identity, event, support, and uncertainty metric. The
first 16 causal updates are now complete. They contain two draws from each of
the eight scenarios, 511 trajectory targets, all updates applied, zero skips
or failures, minimum complete interaction retention `0.917732`, stable peak
RSS `2,827,161,600` bytes, and exact zero node parameter gradient, activity,
drift, and complexity. At step 8 the aggregate loss/gradient is
`0.489130/0.216930`; at step 16 it is `3.902318/1.089642`, with final applied
gradient `1.0`. The dynamics audit passes.

Against the earlier full-attention control on identical steps 8 and 16, the
relation-only path is deliberately a near-control rather than an early win:
current position improves by `0.000045 m`, velocity by `0.005314 m/s`, and x
by `0.001724 m`; z worsens `0.000639 m`. Pooled 0.10/0.25/0.50-second position
worsens `0.000092/0.001763/0.004523 m`, while 0.75/1.00-second improves
`0.000986/0.000797 m`. Lifecycle coverage/precision and identity improve
slightly, collision F1 falls `0.012821`, and uncertainty NLL changes only
`-0.000126` at the median. These heterogeneous training draws are not fixed-
selector evidence and do not justify promotion or mutation; continue to the
durable checkpoint and selector gates.

The first live audit also exposed a diagnostics-only bug. The collapse auditor
reported the node output hook's clipping coefficient under `attention_relation`
even though the frozen zero decoder blocks that backward path and its parameter
gradient is exactly zero. The auditor now excludes node row/output coefficients
only when explicit relation-only telemetry is true, while retaining all real
force/collision/impulse warnings. A new regression test proves this behavior.
The corrected step-8 audit still reports the real contained force-output
coefficient `0.098598` (the matched full-attention control was more strongly
clipped at `0.052808`); decoder-row and complete-interaction coefficients are
`1.0`, so it remains a watch item rather than collapse. Focused tests pass
`16 passed`; Ruff format/check and `git diff --check` pass. The complete suite
is deferred until a natural training validation/checkpoint pause to avoid
competing with the CPU-heavy sustained trainer.

The unchanged relation-only campaign has now reached optimizer step 64. All
64 updates applied with zero skipped draws, exactly eight draws from each of
the eight scenarios, 2,462 cadence-recorded causal trajectories, all 13
objective groups whenever supported, exact zero node gradient/activity/drift/
complexity, and no stderr or terminal failure. The dynamics audit passes;
minimum complete-interaction gradient retention is `0.812481`, while rare
local typed-output coefficients below 10% remain visible warnings rather than
uncontained failures. Peak recorded RSS is `2,916,241,408` bytes and the live
trainer was independently observed using about 480% CPU, so it is progressing
rather than stalled. The complete sampled steps 8--64 window has pooled
position RMSE `0.263631/0.304886/0.368387/0.423629/0.442650 m` at
0.10/0.25/0.50/0.75/1.00 seconds. Against the exact matched full-attention
control, current velocity improves `0.028277 m/s`, current y improves
`0.002108 m`, and pooled 0.10/1.00-second position improves
`0.000658/0.000549 m`; current position is `0.001390 m` worse and the
0.25/0.50/0.75-second horizons are `0.001724/0.004035/0.003019 m` worse.
The candidate x axis remains adverse while z improves at every horizon;
collision F1 remains the clearest early adverse signal. These heterogeneous
cadence samples are not selector evidence. Continue the
immutable run to the step-128 structural audit and the authoritative step-512
fixed selector without retuning.

The preserved step-128 checkpoint now passes its strict structural audit. Its
SHA-256 is `2ad3d36a481879a9bdef72681a84888f521a30b2e68ca2bb95c605247ccb9607`;
all 46 permitted shared/relation tensors changed, both frozen node tensors and
all 177 inherited tensors remain exact, exactly 46 finite Adam owners are at
step 128, both protected checkpoints remain exact, and source, seed-manifest,
rollout-protocol, stored-state, and recomputed-state hashes agree. The durable
artifact is `checkpoints/checkpoint_step_000128.pt` and the self-contained
report is `attention_checkpoint_audit_step_000128.json` in the active run.

The complete matched updates 72--128 dynamics window also passes with eight
draws from every scenario, 2,582 cadence-recorded causal trajectories, all 13
objective groups, no skips or uncontained clipping, minimum complete-gradient
retention `0.461247`, and stable peak RSS `2,924,761,088` bytes. Versus the
full-attention control on the exact same draws, current position and velocity
improve `0.001327 m` and `0.013272 m/s`, x improves through 0.50 seconds, and
collision F1 improves at every horizon. Pooled position improves `0.004786 m`
at 0.10 seconds but regresses `0.002252/0.013123/0.012234/0.019868 m` at
0.25/0.50/0.75/1.00 seconds, mainly from z and mid-horizon y. This is clean,
mixed training evidence rather than convergence or collapse; continue without
promotion or protocol mutation to the fixed step-512 selector.

The subsequent complete updates 136--192 window remains operationally clean
but is accuracy-adverse on the exact matched draws. All 64 updates apply with
eight draws per scenario, 2,355 cadence-recorded causal trajectories, all 13
objective groups, zero skips/uncontained clipping, minimum complete-gradient
retention `0.725199`, and flat peak RSS `2,924,761,088` bytes. Versus the
full-attention control, current position worsens `0.024467 m`; pooled position
worsens `0.027327/0.021932/0.010096/0.008613/0.010786 m` across
0.10/0.25/0.50/0.75/1.00 seconds, mainly x and z. Current velocity worsens
`0.007516 m/s`; lifecycle, trusted identity, coverage90, and collision F1 are
also mostly adverse, although y improves slightly at several horizons and
1.00-second collision F1 improves. This is genuine sampled regression, not
optimizer collapse. Adjacent matched windows have changed sign, so the
predeclared fixed selector remains the first valid behavioral decision point;
continue without promotion or mid-protocol retuning.

The preserved step-256 checkpoint also passes the strict structural audit.
Its SHA-256 is
`13ebe30362ea03fdc9dd998c98c681e730a422eb95f24868eaba757cd3c840fa`;
all 46 permitted tensors and exactly 46 Adam owners are live at step 256,
both frozen node tensors and all 177 inherited tensors remain exact, both
protected checkpoints remain exact, and finiteness/provenance/protocol hashes
pass. The durable artifact is `checkpoints/checkpoint_step_000256.pt`; the
report is `attention_checkpoint_audit_step_000256.json` in the active run.

Complete updates 200--256 partially repair the preceding matched regression
while remaining operationally exact: eight draws per scenario, 2,859 sampled
causal trajectories, all required objective support, zero skips/uncontained
clipping, minimum complete-gradient retention `0.594210`, and flat peak RSS.
Versus the control, current position/velocity improve `0.000545 m` and
`0.077125 m/s`; lifecycle precision/coverage and collision F1 at every horizon
improve, as do pooled 0.10/0.25-second position by `0.000477/0.002903 m`.
The remaining error is localized but meaningful: x regresses at every horizon,
driving pooled 0.50/0.75/1.00-second regressions of
`0.001042/0.011037/0.012596 m`; trusted identity switching and median NLL are
also adverse. Continue to the fixed selector without promotion or retuning.

Complete updates 264--320 again pass every operational invariant with eight
draws per scenario, 2,198 sampled causal trajectories, no skips/uncontained
clipping, minimum complete-gradient retention `0.591258`, and flat peak RSS.
The prior x/long-horizon failure largely repairs on exact matched draws: x
improves by `0.001650/0.004794/0.011681 m` at 0.50/0.75/1.00 seconds and
pooled 1.00-second position improves `0.001286 m`. Current and pooled
0.10--0.75-second position are near-ties but remain adverse by
`0.000175--0.002002 m`; the remaining axis limitation has shifted to z, whose
regression grows from `0.003267 m` at 0.10 seconds to `0.012620 m` at 1.00
second. Trusted identity switching and median NLL improve, while lifecycle,
velocity, and collision slices remain mixed. This migrating axis behavior is
further evidence that heterogeneous training windows cannot replace fixed
validation; continue unchanged to selector 512.

The preserved step-384 checkpoint passes the same strict audit. SHA-256 is
`8f69158746b78cfbca3a719758875136828a32d95d38dc6d764f4bce47f27045`;
all 46 permitted tensors and Adam owners are live at step 384, both node
tensors and all 177 inherited tensors remain exact, protected artifacts remain
exact, and all serialized/provenance state is finite. The durable artifact and
report are `checkpoints/checkpoint_step_000384.pt` and
`attention_checkpoint_audit_step_000384.json` in the active run.

Complete updates 328--384 remain balanced and operationally healthy with
2,642 sampled causal trajectories, all 13 objective groups, no skips or
uncontained clipping, minimum complete-gradient retention `0.585676`, and
flat peak RSS. Accuracy is adverse again on exact matched draws: current
position worsens `0.007449 m` and pooled position worsens
`0.002423/0.005036/0.010147/0.011660/0.007834 m` at every horizon, mainly x.
Current velocity, aggregate collision F1, lifecycle, trusted identity, and
coverage90 improve; long-horizon collision F1 and upper-tail uncertainty NLL
worsen. The prior increasing-z pattern does not persist and the dominant error
migrates back to x. This is wobble without collapse, not fixed-selector
generalization; continue unchanged through step 512.

Complete updates 392--448 are the strongest matched training window so far
while remaining operationally exact: eight draws per scenario, 1,883 sampled
causal trajectories, zero skips/uncontained clipping, complete interaction
retention `1.0` throughout, and flat peak RSS. Current position improves
`0.017153 m` across x/y/z and current velocity improves `0.030032 m/s` versus
the control. Every pooled position horizon improves by
`0.011712/0.011493/0.001840/0.001445/0.003534 m`; trusted identity, coverage90,
and uncertainty NLL also improve. Collision F1 worsens `0.084184` and lifecycle
precision/coverage remain slightly adverse; 1.00-second x is effectively tied
but `0.000145 m` worse. This is encouraging training evidence, not fixed-
manifest promotion evidence. Continue unchanged into the step-512 selector.

The authoritative 32-episode step-512 selector has now completed in
`1148.202 s` and correctly rejects the learned candidate. Its aggregate
selection score improves from protected `0.3213162196` to `0.3054133022`, and
the 0.25/0.50/0.75/1.00-second pooled position horizons improve by
`0.002185/0.024047/0.019757/0.024179 m`. All five x horizons improve and y is
mostly better after 0.10 seconds. This apparent aggregate win is not broad:
current position worsens `0.251460 -> 0.271726 m`, current z worsens
`0.263691 -> 0.315527 m`, 0.10-second pooled position worsens
`0.265184 -> 0.279456 m`, and 0.10/0.25-second z worsen to
`0.318801/0.293951 m`. Target coverage, prediction precision, collision F1,
and trusted identity also regress. The selector reports 109 guardrail
failures—13 pooled and 96 scenario-specific—concentrated in
`reference_pairs` (23), `impulse_perturbation` (23), `elastic_pairs` (18),
and `baseline` (14), with zero protected or mutable training-support
failures. The protected deployment incumbent and reference therefore remain
unchanged at step zero while the explicitly separate mutable trajectory
continues toward the predeclared 8,192-update minimum.

The true durable selector checkpoint is
`checkpoints/validation_step_000512.pt`, SHA-256
`ad645233d833c869f0493cb091231fc3fcb4353d5253fdd44133d3eb015ae979`.
Its strict audit passes: 46 permitted tensors changed, two node tensors and all
177 inherited tensors remain exact, exactly 46 complete finite Adam owners are
at step 512, both protected checkpoints remain exact, and source commit
`c3fe110`, runtime fingerprint, protocol hashes, and serialized finiteness all
match. The report is `attention_checkpoint_audit_step_000512.json` in the run.
An audit command started immediately after the training heartbeat but before
validation atomically saved its checkpoint initially copied the still-current
step-384 `last.pt`. Embedded step and model hash exposed the timing error; both
misnamed artifacts were quarantined with
`premature-stale-step384` suffixes and are explicitly not step-512 evidence.
Training resumed normally and reached step 520 with both first-launch jobs
still running and both stderr logs empty.

The checkpoint auditor now closes that diagnostics race with an optional
`--expected-step` contract. It records the requested boundary and fails when
the checkpoint's embedded step differs. The real selector passes with
embedded/expected step `512/512`; replaying the quarantined artifact fails
exactly with `checkpoint step 384 does not match expected step 512`. This does
not change the pinned trainer, checkpoint serialization, selector, model, or
optimizer. Focused tests pass `4 passed in 5.93 s`; Ruff format/check pass.

Specification 1.43 further requires the unique non-empty serialized Adam-step
set to equal the checkpoint payload step. The auditor now fails stale or mixed
optimizer state even when no external expected step is supplied, and records
`optimizer_steps_match_checkpoint`. A synthetic payload at step 128 with all
Adam owners at step 127 fails exactly; the real active selector passes with
payload/expected/Adam steps `512/512/[512]`. Focused tests pass
`5 passed in 5.73 s`; Ruff format/check and `git diff --check` pass. This is read-only qualification
hardening and does not change the pinned specification-1.42 trainer.

The first complete post-selector updates 520--576 window also passes the
operational audit: all 64 updates apply, eight cadence blocks each contain all
eight scenarios, 2,650 causal trajectories are sampled, no draw is skipped,
minimum complete-interaction gradient retention is `0.673214`, and recorded
peak RSS remains flat at `2,924,761,088` bytes. A rare step-568 force-output
gradient is locally contained at coefficient `0.005344` before shared
backpropagation; the resulting complete update retains `0.673214`, remains
finite, and applies. There is no uncontained clip or stderr.

Against the exact same-draw full-attention control, relation-only current
position improves `0.002802 m`, current velocity improves `0.045913 m/s`, y
improves `0.014045 m`, aggregate collision F1 improves `0.010526`, trusted
identity switch rate improves `0.003038`, coverage90 improves `0.003243`, and
median uncertainty NLL improves `0.003909`. The 0.10-second position horizon
improves `0.003834 m`. The remaining limitation is mature x drift: x worsens
`0.002358/0.007322/0.012089/0.018035/0.024884 m` over
0.10/0.25/0.50/0.75/1.00 seconds, while z is only mildly adverse and y is
strongly better at short horizons. Pooled 0.25/0.50/0.75/1.00-second position
therefore worsens `0.000939/0.006302/0.010493/0.013616 m`; 1.00-second
collision F1 also worsens `0.153846` despite the aggregate event gain.
Lifecycle precision is `0.001952` adverse and target coverage `0.001092`
better. This is a clean long-horizon accuracy warning, not collapse or
promotion evidence. Continue the unchanged trajectory to selector 1024 so the
fixed manifest can determine whether the x/event trade-off persists.

The next complete updates 584--640 window reverses the mature-x warning while
remaining operationally exact: all 64 updates apply, every scenario appears
eight times, 2,060 causal trajectories are sampled, no draw is skipped,
minimum complete-interaction retention is `0.852413`, and peak RSS remains
flat. Step 616 has a real learned force-sensitivity spike: raw aggregate force
output gradient `363.837` is isolated to `0.071543` before decoder/shared
backpropagation (minimum local coefficient `0.00003493`). The downstream force
decoder norm is only `0.286615`, every shared block/projection norm is below
`0.023`, the complete gradient is wholly retained at `0.310713`, and the
finite supported update applies. It does not recur at steps 624, 632, or 640;
there is no uncontained clipping or stderr.

Matched current position and velocity improve `0.009648 m` and
`0.015793 m/s`; x/y/z current position all improve. Every pooled position
horizon improves by `0.012180/0.003781/0.008340/0.012343/0.013490 m`, and x
improves by `0.027351/0.017329/0.023485/0.029444/0.028516 m` at
0.10/0.25/0.50/0.75/1.00 seconds. Trusted identity, coverage90, and
uncertainty improve; lifecycle precision is `0.001144` adverse and target
coverage `0.001142` better. The remaining adverse slices are aggregate
collision F1 (`-0.057959`), collision F1 at every supported horizon, and
0.75/1.00-second velocity (`+0.039511/+0.093056 m/s`). This is the first
complete post-selector window with every pooled position horizon improved,
but it remains sampled evidence rather than promotion.

The preserved step-640 checkpoint passes the specification-1.43-strengthened
audit with embedded/expected/Adam steps `640/640/[640]`. SHA-256 is
`b3b8f41cfb0bea14943068097be141b91ec37a6dc0808ba8d46ffd0e5521f439`;
model-state hash is
`7b55f4696d9a5f821801bf0a40059ddd528c97cde349faf72f073f14add1b12d`.
All 46 permitted tensors and exactly 46 Adam owners are live, both frozen node
tensors and all 177 inherited tensors remain exact, both protected artifacts
remain exact, and finiteness/source/protocol hashes pass. The artifacts are
`checkpoints/checkpoint_step_000640.pt` and
`attention_checkpoint_audit_step_000640.json` in the active run. Continue
unchanged to fixed selector 1024; do not infer convergence from one favorable
training window.

Specification 1.44 fixes an audit-window boundary bug discovered immediately
after step 640. `--after-step N` filtered training, validation, and reference
rows with `step >= N` even though its cadence expectation was already `N+1`.
Adjacent summaries could therefore double-count their boundary row. All three
filters now use strict `step > N`; a regression test covers candidate,
validation, and matched-reference exclusion. The corrected live command
`--after-step 640` reports first/last steps `648/656` and exactly two blocks,
where the old implementation reported `640/656` and three. Focused tests pass
`17 passed in 0.26 s`; Ruff format/check pass. The historical matched control
ends at 640, so exact-draw comparisons beyond it correctly fail for missing
reference steps and will not be fabricated; absolute cadence health continues
while fixed selectors remain authoritative for behavior. The authoritative
auditor plus specification-version suite passes `18 passed in 1.58 s`.

Commands run for this decision were:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/evaluate_modular_candidate.py --config runs/20260812-155706-attention-node-drift-warmup-cosine-stage-a/config.resolved.yaml --base runs/20260812-155706-attention-node-drift-warmup-cosine-stage-a/checkpoints/validation_step_000000.pt --donor runs/20260812-155706-attention-node-drift-warmup-cosine-stage-a/checkpoints/validation_step_000512.pt --module dynamics.attention_interactions.scene_projection --module dynamics.attention_interactions.entity_projection --module dynamics.attention_interactions.relation_projection --module dynamics.attention_interactions.type_embedding --module dynamics.attention_interactions.blocks --module dynamics.attention_interactions.output_norm --module dynamics.attention_interactions.relation_decoder --output runs/20260813-070400-attention-cosine-step512-zero-node-ablation --device cpu --num-workers 0
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/evaluate_modular_candidate.py --config runs/20260812-102557-attention-node-drift-008-stage-a/config.resolved.yaml --base runs/20260812-102557-attention-node-drift-008-stage-a/checkpoints/validation_step_000000.pt --donor runs/20260812-102557-attention-node-drift-008-stage-a/checkpoints/validation_step_000512.pt --module dynamics.attention_interactions.scene_projection --module dynamics.attention_interactions.entity_projection --module dynamics.attention_interactions.relation_projection --module dynamics.attention_interactions.type_embedding --module dynamics.attention_interactions.blocks --module dynamics.attention_interactions.output_norm --module dynamics.attention_interactions.relation_decoder --output runs/20260813-071355-attention-constant-step512-zero-node-ablation --device cpu --num-workers 0
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python train.py --config configs/attention_pilot_mps.yaml --run-name attention-relation-constant-dry-run --device cpu --initialize-from runs/20260810-042627-protocol20-y-only-recovery/checkpoints/validation_step_000064.pt --set training.closed_loop_trainable_scope=attention_relation --dry-run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python train.py --config configs/attention_pilot_mps.yaml --run-name attention-relation-constant-cpu-smoke --device cpu --initialize-from runs/20260810-042627-protocol20-y-only-recovery/checkpoints/validation_step_000064.pt --set training.closed_loop_trainable_scope=attention_relation --set training.steps=1 --set training.train_episodes=8 --set training.validation_episodes=8 --set training.num_workers=0 --set training.eval_every=1 --set training.checkpoint_every=1 --set training.validation_minimum_predictable_target_count_per_scenario_horizon=1 --set training.validation_minimum_matched_target_count_per_scenario_horizon=1 --set training.validation_minimum_supported_episodes_per_scenario=1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python train.py --config configs/attention_pilot_mps.yaml --device cpu --resume runs/20260813-072457-attention-relation-constant-cpu-smoke/checkpoints/last.pt --set training.closed_loop_trainable_scope=attention_relation --set training.steps=2 --set training.train_episodes=8 --set training.validation_episodes=8 --set training.num_workers=0 --set training.eval_every=1 --set training.checkpoint_every=1 --set training.validation_minimum_predictable_target_count_per_scenario_horizon=1 --set training.validation_minimum_matched_target_count_per_scenario_horizon=1 --set training.validation_minimum_supported_episodes_per_scenario=1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_attention_checkpoint.py --checkpoint runs/20260813-072457-attention-relation-constant-cpu-smoke/checkpoints/validation_step_000002.pt --initial-checkpoint runs/20260813-072457-attention-relation-constant-cpu-smoke/checkpoints/validation_step_000000.pt --config runs/20260813-072457-attention-relation-constant-cpu-smoke/config.resolved.yaml --protected runs/20260813-072457-attention-relation-constant-cpu-smoke/checkpoints/reference_rollout.pt --output runs/20260813-072457-attention-relation-constant-cpu-smoke/attention_checkpoint_audit_step_000002.json --frozen-attention-prefix dynamics.attention_interactions.node_decoder. --require-all-attention-changed --require-complete-attention-optimizer-state --require-protected-checkpoints
git diff --check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus pytest -q tests/unit/test_audit_training_dynamics.py tests/unit/test_convergence_supervisor.py
conda run -n orpheus python -m ruff check scripts/supervise_convergence.py scripts/audit_training_dynamics.py tests/unit/test_audit_training_dynamics.py tests/unit/test_convergence_supervisor.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 0 --trend-window-blocks 6 --reference-run runs/20260811-170842-attention-aggregate-isolated-stage-a
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_attention_checkpoint.py --checkpoint runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/validation_step_000512.pt --initial-checkpoint runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/validation_step_000000.pt --config runs/20260813-073710-attention-relation-constant-stage-a/config.resolved.yaml --protected runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/best_rollout.pt --protected runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/reference_rollout.pt --output /tmp/orpheus-attention-relation-audit-step512-expected.json --expected-step 512 --frozen-attention-prefix dynamics.attention_interactions.node_decoder. --require-all-attention-changed --require-complete-attention-optimizer-state --require-protected-checkpoints
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_attention_checkpoint.py --checkpoint runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/checkpoint_step_000512-premature-stale-step384.pt --initial-checkpoint runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/validation_step_000000.pt --config runs/20260813-073710-attention-relation-constant-stage-a/config.resolved.yaml --output /tmp/orpheus-attention-relation-audit-stale-expected512.json --expected-step 512 --frozen-attention-prefix dynamics.attention_interactions.node_decoder. --require-all-attention-changed --require-complete-attention-optimizer-state
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus pytest -q tests/unit/test_audit_attention_checkpoint.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff format --check scripts/audit_attention_checkpoint.py tests/unit/test_audit_attention_checkpoint.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff check scripts/audit_attention_checkpoint.py tests/unit/test_audit_attention_checkpoint.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 513 --trend-window-blocks 8 --reference-run runs/20260811-170842-attention-aggregate-isolated-stage-a
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 577 --trend-window-blocks 8 --reference-run runs/20260811-170842-attention-aggregate-isolated-stage-a
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_attention_checkpoint.py --checkpoint runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/checkpoint_step_000640.pt --initial-checkpoint runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/validation_step_000000.pt --config runs/20260813-073710-attention-relation-constant-stage-a/config.resolved.yaml --protected runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/best_rollout.pt --protected runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/reference_rollout.pt --output runs/20260813-073710-attention-relation-constant-stage-a/attention_checkpoint_audit_step_000640.json --expected-step 640 --frozen-attention-prefix dynamics.attention_interactions.node_decoder. --require-all-attention-changed --require-complete-attention-optimizer-state --require-protected-checkpoints
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus pytest -q tests/unit/test_audit_training_dynamics.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff format --check scripts/audit_training_dynamics.py tests/unit/test_audit_training_dynamics.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff check scripts/audit_training_dynamics.py tests/unit/test_audit_training_dynamics.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260813-073710-attention-relation-constant-stage-a --after-step 640 --trend-window-blocks 8
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_attention_checkpoint.py --checkpoint runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/validation_step_000512.pt --initial-checkpoint runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/validation_step_000000.pt --config runs/20260813-073710-attention-relation-constant-stage-a/config.resolved.yaml --protected runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/best_rollout.pt --protected runs/20260813-073710-attention-relation-constant-stage-a/checkpoints/reference_rollout.pt --output /tmp/orpheus-attention-relation-audit-step512-optimizer-boundary.json --expected-step 512 --frozen-attention-prefix dynamics.attention_interactions.node_decoder. --require-all-attention-changed --require-complete-attention-optimizer-state --require-protected-checkpoints
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus pytest -q tests/unit/test_audit_attention_checkpoint.py tests/integration/test_checkpoint_roundtrip.py::test_checkpoint_specification_version_matches_authoritative_contract
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff format --check scripts/audit_attention_checkpoint.py tests/unit/test_audit_attention_checkpoint.py world_model/utils/version.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff check scripts/audit_attention_checkpoint.py tests/unit/test_audit_attention_checkpoint.py world_model/utils/version.py
git diff --check
```

The specification-1.41 warmup/cosine control is now rejected at its first
trained fixed selector.  Its 32-episode RGB-only score worsens from protected
`0.3213162196` to `0.3475479692`, versus `0.3332532750` for the already
rejected constant-rate control.  There are 116 incumbent/fixed-reference
guardrail failures plus the failed improvement rule (`117` rejection reasons)
and zero training-support failures.  Selection current position worsens
`0.251460 -> 0.304462 m`, target coverage `0.376250 -> 0.358500`, prediction
precision `0.357312 -> 0.339972`, collision F1
`0.195489 -> 0.179310`, and trusted identity-switch rate
`1.3592% -> 2.0984%`.  Every pooled position horizon regresses:
`0.265184/0.277452/0.309911/0.335387/0.357837 ->`
`0.316393/0.317028/0.333434/0.357789/0.374069 m`.  The dominant familiar-
physics failure remains `reference_pairs`: current x
`0.242694 -> 0.720231 m`, with x at 0.10/0.25/0.50/0.75/1.00 seconds
`0.776552/0.760688/0.835178/0.919359/1.000081 m`.  Warmup slightly reduces
that current-x failure relative to constant rate (`0.732948 m`) but broadens
regression elsewhere; schedule repair has therefore failed and no capacity
growth is authorized.

The durable step-512 artifact is structurally exact.  All 48 attention tensors
changed, all 177 inherited tensors are bitwise unchanged, exactly 48 complete
finite Adam owners are at step 512, both protected checkpoints remain exact,
and all serialized state is finite.  Checkpoint SHA-256 is
`bd9bd0eb658a75059b97e1397726d726c29869468ce6ba65996640bcf707b9da`,
model-state hash is
`953a1cac1dc2edeebacbea07e266f6da5d234f4477b6a4e70f8aafa338f8e288`,
and the report is
`runs/20260812-155706-attention-node-drift-warmup-cosine-stage-a/attention_checkpoint_audit_step_000512.json`.
An earlier launchd unload attempt was denied after the managed approval quota
was exhausted. The subsequent host pause removed both one-shot services; the
trainer log later ends at step 592 with empty stderr and no durable validation
checkpoint beyond step 512.

Specification 1.42 records the narrower evidence-backed repair.  A new
`attention_relation` scope trains the 46 shared typed-token/relation tensors
while keeping both node-decoder tensors frozen exactly at the protected zero
initializer.  The checkpoint auditor now accepts explicit frozen-attention
prefixes and can require exact 46/2 tensor and optimizer ownership.  Focused
config/scope/auditor tests pass, and the complete suite passes
`736 passed, 6 skipped in 211.71 s`; the skips are restricted-process MPS
availability cases. Ruff and format checks pass. This
is qualification infrastructure, not accuracy evidence.  The next experiment
must begin weights-only from the untouched graph control, validate a zero-node
relation-only ablation first, and retain every existing selector/test/OOD
guardrail.  Only a qualified relation-first model may add an observation-
derived, zero-default evidence gate for node acceleration; width/depth/history
remain downstream.

Verification commands for this change were:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus pytest -q tests/integration/test_checkpoint_roundtrip.py::test_checkpoint_specification_version_matches_authoritative_contract tests/unit/test_config.py tests/unit/test_training_schedule.py tests/unit/test_audit_attention_checkpoint.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus pytest -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff format --check world_model/utils/config.py world_model/utils/version.py world_model/training/trainer.py scripts/audit_attention_checkpoint.py tests/unit/test_config.py tests/unit/test_training_schedule.py tests/unit/test_audit_attention_checkpoint.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff check world_model/utils/config.py world_model/utils/version.py world_model/training/trainer.py scripts/audit_attention_checkpoint.py tests/unit/test_config.py tests/unit/test_training_schedule.py tests/unit/test_audit_attention_checkpoint.py
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-pycache PYTHONPATH=. conda run -n orpheus python -m compileall -q world_model scripts tests
git diff --check
```

The specification-1.39 constant-rate drift candidate has now been stopped at
its durable step-512 selector boundary.  The full 32-episode RGB-only selector
rejects it: primary score worsens from the protected `0.3213162196` to
`0.3332532750` with `105` broad guardrail failures.  Pooled current position
worsens `0.235574 -> 0.253207 m`, current velocity
`1.051687 -> 1.053425 m/s`, target coverage `0.442852 -> 0.430156`, precision
`0.334068 -> 0.323761`, trusted identity switches `21 -> 27`, and the
0.10/0.25-second position horizons worsen
`0.249561/0.262173 -> 0.265588/0.271238 m`.  The 0.50/0.75/1.00-second
horizons improve slightly, collision F1 improves `0.118077 -> 0.131335`, and
coverage90 is effectively flat, but those gains cannot promote a broadly
regressed candidate.  The dominant familiar-physics failure is
`reference_pairs` current x `0.242694 -> 0.732948 m`, with every x horizon
worse.  The apparently favorable late training windows therefore did not
generalize to the fixed manifest.

The rejection is behavioral rather than structural.  The strict step-512
audit passes with all `48/48` attention tensors changed, all 177 inherited
tensors bitwise exact, exactly 48 complete attention-only Adam owners at step
512, all serialized state finite, and both protected step-zero artifacts
unchanged.  The rejected checkpoint SHA-256 is
`e2101a839076f2bd230545aaedf0a0cf408fcc482d32765463f2a024144087fe`, its
model-state hash is
`79e384408b7f13e4bfb3b4f23b8e7146a5141ea0e51c8849c6af8e821d9115b5`, and
the report is
`runs/20260812-102557-attention-node-drift-008-stage-a/attention_checkpoint_audit_step_000512.json`.
Both one-shot launchd jobs are unloaded; the stale `training_state.json`
remains `running` only because external termination does not rewrite immutable
run evidence.

This result keeps the capacity gate closed.  A refreshed primary-source review
of the original Transformer, compute-optimal scaling, Llama 3, maximal-update
parameterization, SlotFormer, and V-JEPA 2 confirms that Orpheus already uses
the relevant short-token dense core: scaled multi-head attention, residual
branches, pre-RMSNorm, SwiGLU, balanced data, and typed object/relation tokens.
GQA, MLA, FlashAttention, and MoE target long-context memory or activated
compute rather than this at-most-22-token generalization failure.  The next
controlled experiment is therefore the already implemented same-capacity
`warmup_cosine` successor, initialized weights-only from the untouched graph
checkpoint—not from rejected attention weights—with 384 warmup updates,
8,192 fixed cosine-decay updates, and a 0.1 floor.  The 384-step warmup is
about 4.7% of the declared minimum, consistent with the measured long-run
warmup lesson while still reaching peak rate before selector 512.  Only a
guardrail-clean learning curve and declared plateau can reopen the fixed
3.53M depth-six, 4.34M width-192, bounded-history, and 8.31M CUDA ladder.

That schedule control is now active at
`runs/20260812-155706-attention-node-drift-warmup-cosine-stage-a/` from clean
pushed commit `1926547bce2c6c91d3031605b8b5e43b0b2886ab`.  Trainer launchd label
`com.polceanum.orpheus.attention-drift-cosine-20260812-155341` runs as PID
`32510`; exact detached-source supervisor label
`com.polceanum.orpheus.attention-drift-cosine-convergence-20260812-155341`
runs as PID `32750`.  Both are one-shot Standard/default jobs under
`caffeinate`, use `KeepAlive=false`, and have zero-byte stderr.  Metadata
records PyTorch `2.10.0`, MPS built/available, RGB measurement on MPS,
closed-loop execution on CPU, float32, RGB-only runtime, no oracle, and clean
source fingerprint
`c4775f783c0cd6e87c1dcf1b23e4984ee54cbee6576fec9fdede619502b7c8f9`.
The supervisor retains the unchanged 8,192 minimum, 4,096 extensions,
four-selector/1% plateau rule, and 24,576 hard limit.

The successor's mandatory step-zero selector completed in `981.48 s` and
exactly reproduces protected score `0.3213162196`, with zero guardrail or
support failures and exact current, every-horizon, coverage, precision, event,
identity, and calibration metrics.  A cross-run tensor audit passes with all
225 model tensors equal, including all 48 attention and 177 inherited tensors;
all state is finite and both local protected artifacts have the same model
hash `1354bdfca1cef965c0cd907ea8c157c0fd82169e64f24da656eb42dd1a96df91`.
The report is
`runs/20260812-155706-attention-node-drift-warmup-cosine-stage-a/attention_checkpoint_audit_step_000000.json`.
This proves a valid isolated handoff only; it is not trained accuracy or
convergence evidence.  The first logged causal block at step 8 independently
confirms the absolute schedule: learning rate is
`1.0416667e-6 = 8/384 * 5e-5`; loss/gradient are
`0.489044/0.244324`, the update is wholly unclipped and applied, all eight
scenarios contribute 349 trajectory targets, skips are zero, peak RSS is
`2,954,977,280` bytes, and both stderr files remain empty.

The first complete scheduled updates 8--64 window also passes the dynamics
auditor.  All 64 optimizer updates apply, all eight scenarios contribute
exactly eight draws, all mature objectives are supported, and 2,462 causal
trajectory targets are present with zero skips, terminal failures, or
uncontained interaction clips.  Minimum complete interaction retention is
`0.497461`; trusted identity switches are `3/387`; lifecycle precision/target
coverage are `0.384694/0.397679`; current coverage90 is `0.968880`; and peak
RSS stabilizes at `3,019,993,088` bytes.  Pooled current x/y/z RMSE is
`0.281739/0.219617/0.292220 m`; 0.10/0.25/0.50/0.75/1.00-second position RMSE
is `0.262096/0.303683/0.369725/0.427292/0.446976 m`.  Against the rejected
constant-rate run on the exact same eight steps, seeds, scenarios, frames, and
support, current position improves `0.000547 m`, lifecycle and identity improve
slightly, but current velocity worsens `0.011510 m/s` and the 0.25--1.00-second
position horizons worsen `0.000856/0.004015/0.005783/0.002280 m`.  This near-tie
is expected during low-rate warmup and is neither promotion nor regression
evidence; the fixed step-512 selector remains authoritative.

The durable scheduled step-128 checkpoint independently passes the strict
scope/integrity audit.  All `48/48` attention tensors changed, all 177 inherited
tensors remain bitwise exact, exactly the 48 attention parameters own complete
finite Adam state at step 128, both protected incumbents remain exact, and all
architecture/source/protocol/model hashes agree.  The checkpoint SHA-256 is
`eda6aa68b016ce2c7bb0ba2f5b5e78656582a168bc776b9f2078b05d16408225`,
the model-state hash is
`c78b16769fedc6f41cdac61022cfdcc528bd90ed61598e93f06a43928c0b7d4b`,
and the report is
`runs/20260812-155706-attention-node-drift-warmup-cosine-stage-a/attention_checkpoint_audit_step_000128.json`.

Complete updates 72--128 also pass the dynamics auditor with exact eight-way
scenario balance, all 13 mature objectives, 2,594 trajectories, zero skips or
failed updates, minimum complete interaction retention `0.362529`, and flat
`3,019,993,088`-byte peak RSS.  Local collision/event hooks sometimes retain
less than 10%, but contain those rare signals before the shared stage; no
complete update is starved.  Against the constant-rate run on exactly the same
steps/data, current position is `0.014107 m` worse, dominated by x
`+0.037286 m`; every pooled position horizon is
`+0.013738/+0.013842/+0.017214/+0.007199/+0.012912 m` worse.  Current velocity
improves `0.008492 m/s`; velocity horizons and y are mixed, while lifecycle,
identity, collision F1, and median uncertainty NLL are slightly adverse.  The
warmup trajectory has accumulated substantially less update magnitude at this
early boundary, so this is a real watch item but not fixed-manifest rejection.
Continue unchanged to selector 512; do not scale or promote from this window.

The next complete scheduled updates 136--192 window remains operationally
healthy as warmup reaches half of peak rate.  All eight scenarios contribute
exactly eight draws, all 13 mature objectives are supported, 2,368 trajectory
targets contribute, no update is skipped or failed, minimum complete
interaction retention rises to `0.804336`, and peak RSS remains exactly
`3,019,993,088` bytes.  Against the constant-rate run on identical draws, the
scheduled candidate improves current velocity `0.013391 m/s`, trusted switches
`4 -> 3`, median uncertainty NLL `0.015114`, and selected event/y slices, but
current position is `0.032704 m` worse and every pooled position horizon is
`0.032574/0.029765/0.009940/0.004508/0.003085 m` worse.  Current x/y/z differ
by `+0.045208/+0.010432/+0.038140 m`; lifecycle precision/coverage and pooled
collision F1 are also slightly adverse.  Relative to its own preceding
72--128 window on different draws, velocity, x, lifecycle, identity, collision
F1, uncertainty, and gradient retention improve while pooled current position,
y/z, most horizons, and coverage worsen.  At step 192 linear warmup has
accumulated only `25.13%` of the constant run's scalar learning-rate exposure,
so this is mixed slow-learning evidence rather than fixed-manifest rejection.
The immutable run continues to selector 512.

The durable scheduled step-256 checkpoint also passes strict independent
audit: all 48 attention tensors changed, all 177 inherited tensors remain
bitwise exact, exactly 48 complete finite Adam owners are at step 256, and both
protected incumbents remain exact.  Checkpoint SHA-256 is
`79908412c80271451a32541829004a03eb1353621217c18f8bb37d7e5dfd1d1b`,
model-state hash is
`b924aa47abe2b55a4348653d7169057af952d9313d37c16e557ac65ec4427a80`,
and the audit is
`runs/20260812-155706-attention-node-drift-warmup-cosine-stage-a/attention_checkpoint_audit_step_000256.json`.

Complete matched updates 200--256 narrow the earlier physical gap while
remaining non-promotable.  Exact eight-way balance, 2,829 trajectories, all
updates, minimum objective support eight, zero skips/failures, `0.559496`
minimum complete interaction retention, and flat memory pass.  Versus constant
rate on identical draws, current position is now only `+0.005475 m` worse;
y/z current are effectively equal/slightly better and lifecycle/collision F1
improve, while x remains `+0.014387 m` worse and every pooled position horizon
is `+0.007494/+0.011221/+0.015484/+0.015140/+0.014774 m` worse.  Velocity and
identity remain mixed/adverse.  A deterministic functional measurement at the
checkpoint reports emitted RMS acceleration `0.078721 m/s²`, mean
`[-0.043612, 0.128556, -0.012016] m/s²`, and standard deviation
`[0.000884, 0.004134, 0.000481]` over 4,224 active-object evaluations.  This is
materially smaller y bias than the failed constant-rate step-256 mean
`[-0.052641, 0.195037, -0.021283]`, but drift still dominates variation.  The
same draw has configured-total/drift cosine `-0.014628` over all attention and
`-0.167769` in the node decoder, reproducing weak stochastic task/prior
conflict rather than a fixed sign defect.  Continue unchanged to selector 512.

Complete scheduled updates 264--320 provide the first near-neutral exact
position comparison as warmup approaches peak.  The audit passes with exact
eight-way balance, 2,198 trajectories, minimum objective support eight, zero
skips/failures, `0.281649` minimum complete interaction retention, and flat
memory.  Versus constant rate on identical draws, current position differs by
only `+0.001458 m` and x by `+0.001346 m`; 0.10/0.25/0.50-second position is
within `+0.001174/+0.002190/+0.001426 m`.  Current and 0.10/0.25/0.50-second
velocity improve, trusted switches improve `4 -> 3`, collision F1 improves
`0.167939 -> 0.244275`, and coverage90 improves slightly.  Remaining deficits
are concentrated at 0.75/1.00 seconds `+0.004711/+0.008812 m`, y current/short,
long x/z, lifecycle, and median uncertainty NLL.  The historically dangerous
update 280 is contained with complete shared retention `0.908539` despite
local event caps.  Drift reaches about `0.0099` while contextual variation is
nonzero and intermittently rising.  This is promising matched recovery, not
promotion; warmup completion at 384 and the fixed selector at 512 remain the
next evidence gates.

Linear warmup completes exactly at durable step 384 with peak LR `5e-5` and a
clean strict checkpoint audit.  All 48 attention tensors changed, all 177
inherited tensors remain exact, exactly 48 complete finite Adam owners are at
step 384, and protected incumbents remain exact.  Checkpoint SHA-256 is
`4157b8123b203db86c5b2140c988f1d6d7039e3b6702c5dee093d18fd6023181`,
model-state hash is
`1c5ba4be5443f4351c526c68c24717d743a577986940c1ec58395842d2e80769`,
and the audit is
`runs/20260812-155706-attention-node-drift-warmup-cosine-stage-a/attention_checkpoint_audit_step_000384.json`.

The exact matched 328--384 window improves current position `0.002751 m`,
current y/z, 0.10-second position `0.003219 m`, lifecycle precision/coverage,
and retains neutral identity versus constant rate.  Remaining position deficits
move to 0.25/0.50/0.75/1.00 seconds at
`+0.002567/+0.007591/+0.011478/+0.006106 m`, mainly long x/z.  Current and most
velocity horizons, selected collision horizons, coverage90, and uncertainty
are mixed/adverse.  Exact eight-way balance, all 13 objectives, 2,666
trajectories, zero failures/skips, `0.763731` minimum complete interaction
retention versus constant `0.565708`, and flat memory pass.  Drift remains
bounded near `0.01` with nonzero variation instead of the rejected trajectory's
earlier rapid bias growth.  This is current/short recovery with unresolved
mature-horizon generalization; selector 512 remains authoritative.

The first complete cosine-phase window, updates 392--448, remains structurally
healthy but is not a broad accuracy win.  All eight scenarios contribute
exactly eight draws, all 64 updates apply, 1,874 trajectory targets and every
required objective are present, memory remains flat, and there are no skipped,
terminal, or uncontained updates.  One difficult event draw at step 424 raises
the raw gradient to `7.901921`; the typed-output, decoder-row, and shared caps
contain it with `0.126551` complete-interaction retention, above the declared
`0.1` rejection floor.  The following logged updates return to fully retained
gradients, so this is an isolated contained spike rather than collapse.
Against the failed constant-rate run on the exact same draws, trusted identity
improves `2 -> 1` switches and lifecycle precision/coverage plus pooled event
F1 improve slightly.  Current position is nevertheless `+0.003879 m` worse,
current velocity `+0.070337 m/s` worse, coverage90 `-0.003031`, and the five
position horizons differ by
`+0.000926/+0.002356/+0.000443/+0.000000/+0.001569 m`; axes and velocity
horizons remain mixed.  This confirms stable schedule execution, not fixed-
manifest convergence or a capacity limit.  Continue unchanged to selector
512 and keep every scale rung closed.

Historical live-run record follows.  The clean specification-1.39 successor was active at
`runs/20260812-102557-attention-node-drift-008-stage-a/` from pushed clean
commit `176796ff94d89eb79304c58b46e88f9a1ecb9cad`.  Its resolved config differs
from the rejected specification-1.36 campaign by exactly the added
`attention_node_drift: 0.08`; `attention_node_complexity: 1.0`, data, model,
optimizer, selector, and convergence settings are unchanged.  Metadata records
PyTorch `2.10.0`, MPS built and available in the host launch context, RGB
measurement on MPS, closed-loop belief/dynamics on CPU, float32, no oracle,
and a clean source fingerprint.  The trainer label is
`com.polceanum.orpheus.attention-drift-20260812-102513`; the supervisor label
is `com.polceanum.orpheus.attention-drift-convergence-20260812-102513`.  Both
are one-shot Standard/default launchd jobs with `KeepAlive=false` under
`caffeinate`; the supervisor runs from an exact detached source worktree at
the same commit.  Both stderr files are empty.

The mandatory 32-episode step-zero selector completed in `991.16 s` and
exactly reproduces the protected graph control: score `0.3213162196`, current
position `0.2514599 m`, velocity `1.0931909 m/s`, selected horizons
`0.265184/0.277452/0.309911/0.335387/0.357837 m`, target coverage `0.37625`,
precision `0.357312`, collision F1 `0.195489`, trusted identity-switch rate
`1.3592%`, and coverage90 `93.3861%`, with zero guardrail or support failures.
The imported runtime is durably preserved as `validation_step_000000.pt`,
`best_rollout.pt`, and `reference_rollout.pt`.  The active convergence policy
requires 8,192 updates, uses 4,096-update extensions, four exact consecutive
512-step candidates plus less than 1% raw improvement for plateau, and has a
24,576 hard limit.  No trained accuracy or convergence result exists yet.

The first complete eight-block window through update 64 passes the dynamics
audit.  All 64 updates apply, every scenario contributes eight times, there are
no skipped draws or terminal/uncontained interaction failures, and the window
provides `2,462` causal trajectories.  Update eight records genuinely
nonzero activity/drift/variation
`8.10495e-5/8.09151e-5/1.34405e-7`; update 16 restores all 13 mature causal
terms; update 64 reaches
`4.11472e-3/4.11290e-3/1.82318e-6`.  Its squared x/y/z mean drift is
`0.001170/0.011169/0.00000048`, so y drift is an explicit early-risk signal and
is about `8.8%` of the rejected step-512 y activity.  The event-heavy update
16 global norm is safely clipped `1.62520 -> 1.00000`, with minimum complete
interaction retention `0.615309`, above the `0.1` rejection floor.  The
schedule-matched comparison against the complexity-only predecessor uses the
same seeds and has one additional causal trajectory.  Pooled current position
improves by
`0.000422 m`; horizons differ by
`-0.001568/-0.000612/+0.000954/+0.000168/+0.000738 m`, while current velocity
improves by `0.034592 m/s`.  Collision F1 is `0.0470` lower on this small
discrete training pool; identity retains the same four switches on 384 versus
385 associations, lifecycle precision/coverage improve slightly, and current
coverage90 differs by `-0.001313`.  These are watch items, not selector-level
evidence.

The durable step-128 boundary is now independently qualified.  Its strict
checkpoint audit passes: all 48 attention tensors changed, all 177 inherited
tensors remain bitwise exact, exactly 48 complete finite Adam owners are at
step 128, both explicitly supplied protected checkpoints remain equal to the
initializer, and architecture, source, protocol, model-state, and whole-file
hashes agree.  The checkpoint SHA-256 is
`759e1c5fc72cbf43f9bed80a06878bd363e9ea9ac36e0779dad517e47e8f6f54`,
the model-state hash is
`c46f8701623c3c49d7310844d946239e247b354d341d286cb63ed42eb878a636`,
and the report is
`runs/20260812-102557-attention-node-drift-008-stage-a/attention_checkpoint_audit_step_000128.json`.

The complete schedule-matched updates 72--128 window also passes operational
health with all eight scenarios represented eight times, `2,586` causal
trajectories, all 13 objectives, zero skips or failed updates, flat
`2,978,533,376`-byte peak RSS, and minimum complete-interaction retention
`0.391475`.  It does not show an accuracy gain over the already rejected
complexity-only predecessor: current position is `0.012756 m` worse, x/y/z
are `0.020090/0.012332/0.004464 m` worse, velocity is `0.017016 m/s` worse,
and position at 0.10/0.25/0.50/0.75/1.00 seconds is
`0.012087/0.011816/0.010595/0.008278/0.015943 m` worse.  Collision F1 improves
`0.020695` and identity improves from five to two switches, while lifecycle
rates are equal, coverage90 is `0.002799` lower, and median uncertainty NLL is
`0.025968` worse.  These are heterogeneous training-window diagnostics, not
fixed-manifest rejection or promotion evidence.

The same-draw functional report at step 128 measures activity/drift/variation
`0.0112875/0.0112825/0.00000506 (m/s^2)^2`, RMS emitted acceleration
`0.106243 m/s^2`, and mean emitted acceleration
`[-0.065046, 0.171365, -0.015823] m/s^2` over `5,184` active-object
evaluations in 144 gradient-enabled causal attention calls.  The calibration
utility now explicitly excludes no-gradient prepared-rollout calls so its
emitted-value population exactly matches the differentiable activity/drift
population.  Compared with the rejected predecessor at step 512, RMS activity
is about half (`0.1062` versus `0.2066 m/s^2`) and mean y acceleration is less
than half (`0.1715` versus `0.3567 m/s^2`), proving the configured prior acts
on its intended target.  More than `99.95%` of remaining activity is still
context-invariant drift, however, and the matched position window is broadly
adverse.  Continue the immutable run to the authoritative fixed selector 512;
do not scale capacity or alter its constant learning rate from this diagnostic
window alone.  If that selector rejects the candidate, test warmup plus decay
as a separately versioned protected-control successor before adding depth or
width.

The live run has subsequently reached update 352 with empty stderr; trainer
and supervisor remain alive and the MPS trainer was using `520%` CPU at the
2026-08-12 health check. The complete schedule-matched 136--192 window passes
with every scenario eight
times, all 13 objectives, no failed/uncontained update, and flat RSS. Relative
to the rejected complexity-only predecessor it improves current position by
`0.017231 m`, x/z by `0.025865/0.023192 m`, and the 0.10/0.25-second horizons
by `0.014988/0.005339 m`. It remains worse at 0.50/0.75/1.00 seconds by
`0.013869/0.015243/0.012813 m`, velocity by `0.007944 m/s`, y by
`0.000982 m`, and coverage90 by `0.004990`; identity and lifecycle are also
mixed. This is the first training window to recover current and short-horizon
position, but it remains non-promotable evidence before selector 512.

Specification 1.40 adds a backward-compatible, opt-in `warmup_cosine`
closed-loop learning-rate protocol without changing the live pinned process.
Its rate is a pure function of absolute causal update index, explicit warmup
and decay durations, and a minimum scale, so extending `training.steps` cannot
reshape the schedule. Historical configs normalize to exact `constant`
behavior and a schedule change is rejected by exact resume. The real CPU smoke
at `runs/20260812-123215-lr-schedule-smoke/` completed two updates and an exact
third-update resume; it logged the expected `0.0002` second warmup rate and
`0.0001100000` first cosine rate. Its two-episode validation is explicitly
`last_unvalidated` and is implementation evidence only. No scheduled successor
is authorized unless the active fixed selector rejects. The complete repository
suite passes with `732 passed, 6 skipped in 244.57 s`; all six skips are the
expected MPS-unavailable cases in the restricted test process, while the host
launch context independently proves MPS available and active.

The durable step-256 checkpoint independently passes the strict attention
audit. All `48/48` attention tensors changed, all 177 inherited tensors remain
bitwise exact, exactly 48 complete finite Adam owners are at step 256, both
protected incumbent checkpoints remain model-state-equal to step zero, and
architecture, source, protocol, stored-state, and recomputed model hashes
agree. The checkpoint SHA-256 is
`e9c94bb2e5facd6a2aa833d0c7d8f4f5cb8b8bbd76dc031ee2fbcc6bea70d788`;
its model-state hash is
`e086c2528af79370477013911427c2e5343b871f6e611f39f9c6a248eda40dce`.
The qualified report is
`runs/20260812-102557-attention-node-drift-008-stage-a/attention_checkpoint_audit_step_000256.json`.
An earlier invocation that incorrectly compared the learned attention module
against the graph-only source checkpoint is transparently retained with an
`_invalid_graph_initializer` suffix and is not qualification evidence.

The complete schedule-matched updates 184--240 window also passes operational
health: all eight scenarios occur eight times, all 13 objectives are present,
`2,549` causal trajectories contribute, no update fails, minimum complete
interaction retention is `0.359573`, and RSS is flat. Versus the rejected
complexity-only predecessor, current position improves `0.011870 m`, x/z
improve `0.019308/0.018524 m`, and 0.10/0.25-second position improves
`0.009625/0.004403 m`. The candidate remains worse at 0.50/0.75/1.00 seconds
by `0.002679/0.005120/0.007016 m`, at current velocity by `0.031758 m/s`, and
at every velocity horizon by `0.005850--0.106037 m/s`; y, collision F1,
lifecycle precision/coverage, coverage90, and median uncertainty NLL are also
worse. The step-256 same-draw functional calibration is finite but still
drift-dominated: RMS emitted node acceleration is `0.117747 m/s²`, mean
acceleration is `[-0.052641, 0.195037, -0.021283] m/s²`, and activity/drift/
variation are `0.0138644/0.0137544/0.000109986 (m/s²)²`. This is a coherent
short-position versus velocity/long-horizon limitation, not corruption or
collapse; selector 512 remains authoritative.

The first complete post-checkpoint window, updates 208--264, also passes all
operational gates with `2,803` trajectories, equal eight-scenario support, all
13 objectives, no failed update, flat RSS, and the same `0.359573` minimum
complete-interaction retention. Its matched behavior wobbles rather than
extending the earlier position gain: current position is now `0.003874 m`
worse than the rejected predecessor and aggregate position at
0.10/0.25/0.50/0.75/1.00 seconds is worse by
`0.005148/0.006213/0.000402/0.003941/0.003922 m`. Current velocity improves
`0.005984 m/s`, but four of five velocity horizons remain worse, including
`0.081391 m/s` at 0.75 seconds. Node drift falls from `0.016990` at update 208
to `0.014272 (m/s²)²` at 264 while variation rises from `0.00001149` to
`0.00009277 (m/s²)²`, proving the learned residual is becoming less constant
but not yet behaviorally convergent. Do not infer a plateau or intervene from
this heterogeneous training window; retain selector 512 as the fixed gate.

The next exact balanced window, updates 264--320, passes again with `2,198`
trajectories, all objectives, equal scenario support, no skips/failures, flat
RSS, and minimum complete-interaction retention `0.279453`. Against the
predecessor it improves current and 0.10/0.25-second position by
`0.003082/0.001778/0.001540 m`, but regresses 0.50/0.75/1.00-second position
by `0.003408/0.003267/0.003395 m` and every velocity horizon by
`0.002305--0.017385 m/s`. Against the preceding complete candidate window,
however, current position improves `0.026530 m` and four position horizons
improve `0.016933--0.039513 m`; 0.75 seconds is effectively flat
(`+0.001378 m`). Uncertainty NLL and identity improve, while 0.25-second
velocity worsens. This is genuine local learning plus an unresolved
position/velocity and short/long-horizon trade-off, not a monotone broad
convergence curve.

Specification 1.41 adds read-only task/prior gradient-alignment evidence. Two
deterministic balanced draws at the exact step-256 checkpoint prove there is no
fixed sign bug and expose stochastic conflict instead. Draw 255 has task/
drift node-decoder cosine `+0.219746` and configured-total/drift cosine
`+0.413340`, so both local descent directions reduce drift. Draw 254 has
`-0.877315/-0.666858`, so the physical task and even the configured total
objective locally favor increased drift. On draw 254 the configured drift
gradient norm is `0.053814` versus task node-decoder norm `0.178992`; on draw
255 they are `0.051983/0.339891`. This alternating conflict explains why the
soft prior can reduce average bias yet still wobble under a constant rate. It
supports the already-gated warmup/cosine same-capacity successor if selector
512 rejects; it does not authorize a live weight change or gradient surgery.
The gradient-alignment change passes its focused `4 passed` suite and the
complete repository suite (`734 passed, 6 skipped in 231.52 s`), plus Ruff,
format, compileall, and diff checks. The six skips are the expected restricted-
process MPS availability cases; the independent host trainer remains on MPS.

The newest complete matched updates 296--352 window remains operationally
clean with `2,449` causal trajectories, all 13 objectives, eight draws from
every scenario, no skips/failures, flat RSS, and `0.279453` minimum complete-
interaction retention. It is the first recent window to shift the mature
horizon favorably against the predecessor: 0.50/0.75/1.00-second position
improves `0.003456/0.008174/0.014185 m`, current velocity improves
`0.024917 m/s`, velocity through 0.75 seconds improves
`0.005691--0.095032 m/s`, collision F1 improves `0.037921`, and identity has
one fewer switch. It still regresses current and 0.10-second position by
`0.001715/0.002855 m`, 1.00-second velocity by `0.039951 m/s`, and lifecycle
precision/coverage by `0.001861/0.003261`; 0.25-second position is effectively
equal (`+0.000117 m`). This is encouraging long-horizon learning but remains
mixed cadence evidence before the fixed selector.

The durable step-384 checkpoint passes the strict audit with all `48/48`
attention tensors changed, all 177 inherited tensors bitwise exact, exactly 48
complete finite Adam owners at step 384, both protected incumbents unchanged,
and matching stored/recomputed model hashes. Its SHA-256 is
`04adbc454761e802f5ff2ff6654b86ddd3b07f2f95c4fd6d9ec64130b72ccbde` and
model-state hash is
`a97fcea36d746e371cc8b49f109956fc3c90268daccee3ff84be1a6e332fd948`;
the report is
`runs/20260812-102557-attention-node-drift-008-stage-a/attention_checkpoint_audit_step_000384.json`.
Same-checkpoint draws 382/383 again alternate configured-total/drift node-
decoder cosine `-0.292264/+0.945411`, so the conflict diagnosis reproduces.
Mean y acceleration remains high at `0.217825/0.217058 m/s²`, while mean x
has contracted to about `-0.017 m/s²` and contextual variation is larger. The
model is learning useful context/relation behavior without yet eliminating
the global y component.

The complete matched updates 328--384 window is the broadest favorable
training window so far. It passes with `2,682` trajectories, all objectives,
equal scenario support, no skips/failures, flat RSS, and `0.565708` minimum
complete-interaction retention. Versus the predecessor it improves current
position `0.001088 m`, 0.50/0.75/1.00-second position
`0.003460/0.010106/0.014922 m`, current velocity `0.020690 m/s`, four velocity
horizons `0.009813--0.073908 m/s`, collision F1 at every horizon, identity by
four switches, and lifecycle precision slightly with equal coverage. It still
regresses 0.10/0.25-second position `0.002340/0.002509 m`, 0.10-second
velocity `0.008552 m/s`, current y/z, and median uncertainty NLL. This is
promising but remains non-promotable before selector 512.

The immutable specification-1.36 residual-parsimony campaign has been stopped
at its durable step-1024 selector boundary.  The checkpoint is structurally
valid, not collapsed: its strict audit passes with all 48 attention tensors
changed, all 177 inherited tensors bitwise exact, exactly 48 finite Adam owners
at step 1024, both protected artifacts unchanged, and matching source,
configuration, protocol, and model hashes.  The checkpoint SHA-256 is
`36a196165e95675275efd8949b657853264600bb8032d9dfaffa8383a42b8081` and
its model-state hash is
`a62808c8de646f54d31e9a827e791bbe78ad0e11750c222dae8b98564bba9b6c`.
The audit is persisted at
`runs/20260811-234157-attention-node-parsimony-stage-a/attention_checkpoint_audit_step_001024.json`.

The authoritative fixed selector rejects the learned candidate by 111 broad
guardrails.  Its scalar score is microscopically better
(`0.3213162196 -> 0.3212919367`), but selected current position worsens
`0.251460 -> 0.274762 m`, x worsens `0.281775 -> 0.342313 m`, target coverage
falls `0.37625 -> 0.36775`, precision falls `0.35731 -> 0.34808`, and the two
shortest mature horizons worsen from `0.265184/0.277452` to
`0.282590/0.290155 m`.  The familiar `reference_pairs` regime is the clearest
failure: current position worsens `0.212965 -> 0.383810 m`, current x
`0.242694 -> 0.573947 m`, and x at 0.10--1.00 seconds worsens from
`0.296812/0.383256/0.562842/0.687318/0.791271` to
`0.603153/0.625942/0.708155/0.804530/0.891246 m`.  Camera parallax and
glancing impacts add further x/z, coverage, calibration, and identity
failures.  This is behavioral overfit, not corruption, support loss, resource
growth, or numerical failure.

The complete steps 904--960 training window independently remains healthy:
2,939 trajectories, all 13 objectives, minimum complete-interaction retention
`0.514307`, identity `5/419`, lifecycle precision/coverage
`0.40661/0.44093`, coverage90 `0.96953`, and position horizons
`0.28076/0.32357/0.38991/0.42468/0.42918 m`, with flat RSS and no rejected
update.  Healthy optimizer dynamics therefore did not repair fixed-manifest
generalization.  Continuing the same objective beyond step 1024 is not
justified.  Both one-shot jobs were booted out after validation completed; no
post-1024 update was written.  The run's stale `training_state.json` still says
`running` because external launchd termination does not rewrite immutable run
artifacts; external process state is stopped.

The next campaign is the already implemented specification-1.39 successor,
initialized weights-only from the untouched protected graph control, with
`attention_node_complexity=1.0` and axis-neutral
`attention_node_drift=0.08`.  It must reproduce the step-zero selector and
then pass repeated fixed validation, disjoint test/OOD, scenario, axis,
identity, event, support, and calibration guardrails before any depth or width
increase.  The literature review does not justify an LLM-shaped rewrite:
scaled multi-head attention, residual paths, pre-normalization, and SwiGLU are
already present, while GQA/MLA/MoE/local attention primarily address long
contexts or cluster economics absent from the at-most-22-token Mac rung.
Compute and data will scale together only after this smaller rung demonstrates
a real generalization curve and plateau.

Before its deliberate step-1024 stop, the immutable residual-parsimony campaign
at `runs/20260811-234157-attention-node-parsimony-stage-a/` remained
operationally healthy. Its trainer and convergence supervisor each launched
once, both stderr files were empty, sampled RSS remained near `2.92 GB`, and
the protected incumbent remained step zero. The following step-640--768
evidence records earlier checks on that now-stopped trajectory.

The durable specification-1.36 step-640 `last.pt` also passes the strict
checkpoint audit: all 48 attention tensors changed, all 177 inherited tensors
remain bitwise exact, exactly 48 complete finite Adam states are at step 640,
both protected checkpoints remain exact initial state, and serialized values,
hashes, shape/dtype contract, protocol, and provenance all pass. Its checkpoint
SHA-256 is
`4c3858b3100d18478f54c5193ab3f6a5915ac069aab3b152941cd469203ff14b`
and model-state hash is
`0aea533c772db158add9125d7412b0c69ef41dc509cb6737f103d8e40861eefc`.
An audit over logged steps 640--728 passes with 728 applied optimizer updates,
zero skipped draws, equal 12-draw representation of all eight scenarios, no
terminal failure, no uncontained interaction clip, and flat `2.913 GB` RSS.
The complete 640--696 window retains at least `0.216657` of the post-typed
interaction gradient and supports 2,131 trajectories. The 704--728 window is
only four of eight blocks complete, so its lower sampled position errors are
not comparable evidence of improvement and are not used for promotion.

The subsequent complete steps 704--760 balanced window also passes: all eight
scenarios are drawn eight times, all eight updates apply with all 13 causal
terms, 2,115 trajectories contribute, minimum complete-interaction retention
is `0.269683`, and there is no terminal failure, skipped draw, uncontained
interaction clip, or material memory growth. Pooled position RMSE is
`0.201204/0.227514/0.272258/0.311915/0.337189 m` from 0.10 to 1.00 seconds;
current position/velocity are `0.215572 m` / `1.252608 m/s`; identity is
`3/300 = 1.0%`; lifecycle precision/coverage are `0.366213/0.360491`; and
current position coverage90 is `0.985348`. These are healthy heterogeneous
training diagnostics, not fixed-manifest validation. The next durable
step-768 checkpoint passes the strict audit with file SHA-256
`9a2867ce6d31724311f12be33ae6ff1c76b02f47b44465eba202469fee7c83fc`,
model hash
`ddb76172be37fb3196506c2134989db622a74e9d66fd8b0bf1dd0042e07d0764`,
all 48 attention tensors changed, 177 inherited tensors exact, exactly 48
complete Adam owners at step 768, finite serialization, and both protected
checkpoints exact initial.

Step 512 is a valid but rejected learned candidate. Its exact audit passes all
tensor, inherited-state, Adam ownership/step, finiteness, protected-checkpoint,
hash, and protocol gates. Pooled selector score improves
`0.3213162196 -> 0.3177418187`, but current position worsens
`0.251460 -> 0.283202 m`, target coverage falls `0.37625 -> 0.35575`, precision
falls `0.35731 -> 0.33768`, and 109 strict guardrails fail. Its pooled forecast
horizons are `0.289063/0.288099/0.301042/0.328230/0.344516 m` versus protected
`0.265184/0.277452/0.309911/0.335387/0.357837 m`: the two shortest regress and
0.5--1.0 seconds improve. `reference_pairs` current position is the largest
failure (`0.212965 -> 0.429954 m`).

Same-manifest ablations prove that global residual shrinkage and deleting
relation force rows are harmful. Zero-y improves every horizon but remains
rejected with 97 failures. Zeroing the complete node decoder is strongest at
score `0.297330` and improves every horizon to
`0.263736/0.262437/0.280491/0.307728/0.328537 m`, but still fails 72 broad
guards. Specification 1.38 therefore added the opt-in, axis-neutral
`attention_node_activity` functional prior while retaining all node capacity.
It measures bounded acceleration actually emitted for active objects across
the causal rollout and changes neither inference nor historical configs.
Focused tests pass (`4 passed`); the complete CPU-visible suite passes
`718 passed, 6 skipped in 212.36 s`. An `orpheus` dry-run with both exact
regularizers succeeds. Specification 1.39 further separates squared mean drift
from useful residual variation. Capacity scaling remains blocked pending
selector 1024 and, if needed, a clean protected-control 1.39 successor.

A schedule-matched 16-block audit over steps 520--640 passes with all scenarios
drawn 16 times, `4,678` causal trajectories, every update applied, no terminal
failure, minimum complete-interaction retention `0.198063`, and flat RSS.
Against the unregularized predecessor on the same draws, current position
improves `9.62 mm` and 0.10-second position improves `7.52 mm`; every current
axis and four of five velocity horizons improve. The later position horizons
remain nearly neutral/slightly adverse (`+0.22` to `+1.06 mm`), current
velocity regresses `0.0189 m/s`, collision F1 improves `0.0032`, identity rate
improves `0.017` percentage points, and lifecycle precision/coverage improve
about `0.3` percentage points. This is encouraging causal-window evidence, not
a fixed-selector promotion or proof of convergence.

The reproducible activity calibration report is
`runs/20260811-234157-attention-node-parsimony-stage-a/attention_node_activity_calibration_step_000512.json`.
It hashes the candidate as
`9dec4da06af3a991374e1df0e87668b9557bc5b65fd7446c9c97395d891ad17f`
and measures activity `0.042669 (m/s²)²`, x/y/z
`0.000618/0.127347/0.000042`, and RMS emitted acceleration
`0.206565 m/s²` on one balanced eight-scenario causal draw. Unit activity has
gradient norm `0.673351` versus `0.052798` for unit decoder complexity; their
equal-gradient weight ratio is `0.078411`. Four draws bound this ratio within
`0.078292--0.078442`. Across `10,182` active-object invocations, emitted mean
is `[-0.024866, 0.356690, 0.006175] m/s²` and standard deviation is only
`[0.000736, 0.001865, 0.000746]`; squared drift `0.04266783` accounts for more
than 99.997% of activity `0.04266899`. A prospective specification-1.39
successor therefore records `attention_node_drift=0.08`, preserving balanced
context-sensitive variation rather than penalizing all activity.

The prospective specification-1.39 objective has now passed a real
protected-control CPU smoke and exact resume at
`runs/20260812-065434-attention-node-drift-008-cpu-smoke/`. Two balanced
eight-scenario updates both applied with all 13 causal objective terms. At the
second update, support is 343 trajectories, complexity is `3.22494e-7`, total
activity is `4.98557e-6`, squared drift is `4.97644e-6`, and residual variation
is `9.12632e-9`; the nonzero drift term therefore participates in the actual
optimizer graph. The strict step-2 audit passes with all 48 attention tensors
changed, 177 inherited tensors exact, exactly 48 Adam owners at step two, and
the protected graph exact. Its checkpoint SHA-256 is
`cb8a361e0659d8d597c80c7ba342408c554637e7b6f769ba6db9a610dc3deeb0`.
This smoke deliberately used only eight validation episodes and reduced
support gates; its checkpoint kind is truthfully `last_unvalidated`, so it
qualifies wiring, scope, exact resume, and numerical behavior only—not accuracy
or generalization.

Commands verified in the `orpheus` environment for this change:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus pytest -q tests/unit/test_hybrid_dynamics.py tests/unit/test_training_schedule.py -k 'attention_node_activity or attention_node_complexity'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus pytest -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus pytest -q tests/integration/test_checkpoint_roundtrip.py tests/unit/test_hybrid_dynamics.py tests/unit/test_training_schedule.py -k 'specification_version or attention_node_activity or attention_node_complexity'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff check world_model/dynamics/attention.py world_model/training/loop.py tests/unit/test_hybrid_dynamics.py tests/unit/test_training_schedule.py world_model/utils/version.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff format --check world_model/dynamics/attention.py world_model/training/loop.py tests/unit/test_hybrid_dynamics.py tests/unit/test_training_schedule.py world_model/utils/version.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260811-234157-attention-node-parsimony-stage-a --after-step 520 --trend-window-blocks 16 --reference-run runs/20260811-170842-attention-aggregate-isolated-stage-a
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/measure_attention_node_activity.py --config runs/20260811-234157-attention-node-parsimony-stage-a/config.resolved.yaml --checkpoint runs/20260811-234157-attention-node-parsimony-stage-a/checkpoints/validation_step_000512.pt --step-index 511 --device cpu --output runs/20260811-234157-attention-node-parsimony-stage-a/attention_node_activity_calibration_step_000512.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python train.py --config configs/attention_pilot_mps.yaml --run-name attention-node-drift-008-dry-run --device mps --initialize-from runs/20260810-042627-protocol20-y-only-recovery/checkpoints/validation_step_000064.pt --set training.loss_weights.attention_node_complexity=1.0 --set training.loss_weights.attention_node_drift=0.08 --dry-run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python train.py --config configs/attention_pilot_mps.yaml --run-name attention-node-drift-008-cpu-smoke --device cpu --initialize-from runs/20260810-042627-protocol20-y-only-recovery/checkpoints/validation_step_000064.pt --set training.steps=1 --set training.train_episodes=8 --set training.validation_episodes=8 --set training.num_workers=0 --set training.eval_every=1 --set training.checkpoint_every=1 --set training.validation_minimum_predictable_target_count_per_scenario_horizon=1 --set training.validation_minimum_matched_target_count_per_scenario_horizon=1 --set training.validation_minimum_supported_episodes_per_scenario=1 --set training.loss_weights.attention_node_complexity=1.0 --set training.loss_weights.attention_node_drift=0.08
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python train.py --config configs/attention_pilot_mps.yaml --device cpu --resume runs/20260812-065434-attention-node-drift-008-cpu-smoke/checkpoints/last.pt --set training.steps=2 --set training.train_episodes=8 --set training.validation_episodes=8 --set training.num_workers=0 --set training.eval_every=1 --set training.checkpoint_every=1 --set training.validation_minimum_predictable_target_count_per_scenario_horizon=1 --set training.validation_minimum_matched_target_count_per_scenario_horizon=1 --set training.validation_minimum_supported_episodes_per_scenario=1 --set training.loss_weights.attention_node_complexity=1.0 --set training.loss_weights.attention_node_drift=0.08
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_attention_checkpoint.py --config runs/20260812-065434-attention-node-drift-008-cpu-smoke/config.resolved.yaml --checkpoint runs/20260812-065434-attention-node-drift-008-cpu-smoke/checkpoints/last.pt --initial-checkpoint runs/20260812-065434-attention-node-drift-008-cpu-smoke/checkpoints/validation_step_000000.pt --protected runs/20260812-065434-attention-node-drift-008-cpu-smoke/checkpoints/reference_rollout.pt --require-protected-checkpoints --require-all-attention-changed --require-complete-attention-optimizer-state --output runs/20260812-065434-attention-node-drift-008-cpu-smoke/attention_checkpoint_audit_step_000002.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_attention_checkpoint.py --config runs/20260811-234157-attention-node-parsimony-stage-a/config.resolved.yaml --checkpoint runs/20260811-234157-attention-node-parsimony-stage-a/checkpoints/last.pt --initial-checkpoint runs/20260811-234157-attention-node-parsimony-stage-a/checkpoints/validation_step_000000.pt --protected runs/20260811-234157-attention-node-parsimony-stage-a/checkpoints/best_rollout.pt --protected runs/20260811-234157-attention-node-parsimony-stage-a/checkpoints/reference_rollout.pt --require-protected-checkpoints --require-all-attention-changed --require-complete-attention-optimizer-state --output runs/20260811-234157-attention-node-parsimony-stage-a/attention_checkpoint_audit_step_000768.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260811-234157-attention-node-parsimony-stage-a --after-step 640 --trend-window-blocks 8
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py --run runs/20260811-234157-attention-node-parsimony-stage-a --after-step 704 --trend-window-blocks 8
```

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
maximum RSS; the superseded protocol-17 jobs are stopped and the clean
protocol-18 balanced campaign was intentionally stopped at its durable
step-128 checkpoint after exact fixed validation rejected every forecast
horizon and localized the dominant regression to the learned updater rather
than dynamics or identification; the updater was found to discard axis/sign
innovation and to apply unconstrained learned mean/variance residuals to
unsupported packed fields; specification 1.20 now provides innovation-
anchored, support-masked correction plus exact row-isolated recovery; the
step-192 y-only composition is the first guardrail-clean corrected incumbent;
the subsequent 512-update y-only campaign is complete and plateaued; its
apparent late association wobble was traced to an exact fast-ROI component-
ownership tie, not optimizer collapse; specification 1.21 and rollout protocol
14 now reject only numerically tied disconnected components and leave recovery
to global discovery; paired public and exact physical validation improve the
small control without a broad regression, step 64 remains better than step
512; stage A of the Mac-scale typed attention pilot is now implemented as an
exact zero-output graph residual, and its one-update hybrid MPS/CPU smoke is
finite, supported, scope-clean, and memory-bounded, but it has no accuracy or
generalization promotion yet; the declared 8,192-update attention-only
campaign reached a durable, finite step-128 checkpoint but was stopped after
exact tensor audit proved its scene projection consumed an always-zero
`global_code`; specification 1.23 repairs the scene token to consume live
global, gravity, camera-pose/motion/intrinsics/uncertainty, and calibration
belief context while preserving exact zero-output graph identity; its corrected
smoke passes, but the first sustained live-scene run was stopped at sampled
update 64 after mixed-unit raw scene features caused a `45.3456` interaction
gradient and `0.02205` local clip coefficient; specification 1.24 adds fixed
non-affine pre-projection RMS conditioning; that repair removes the matched
scene spike, but a later campaign is stopped at clean step 256 after collision-
logit row spikes recur exactly 128 updates apart; specification 1.25 isolates
that typed proposal row before the complete interaction cap; exact replay now
localizes one recurrence to the typed normal/tangent force rows, but the
force-row-isolated campaign is also stopped at durable step 256 after the same
step-280 batch contaminates the shared stack before its decoder-row cap;
specification 1.28 adds typed-output backpropagation isolation, exact replay
contains the known failure without shared collapse, and a fresh sustained
qualification now passes the identical step-64 stress position without shared
collapse, but that campaign is stopped after update 200 exposes an uncapped
impulse-multiplier/additive path that again starves the shared stage;
specification 1.30 isolates impulse outputs and decoder rows, rejects any
post-isolation complete interaction retention below 10% before Adam mutates,
and makes the offline auditor fail the same condition; matched diagnostic
replay contains update 200, but the fresh campaign is stopped after attempted
update 60 exposes unbounded accumulation in the node decoder; specification
1.31 isolates that x/y/z group, persists complete terminal optimizer evidence,
and passes a fresh protected-control update-60 causal replay at `0.565343`
retention; full regression gates and an exact clean sustained step-zero
relaunch now pass, while fixed trained selectors, plateau, convergence, and
every capacity promotion remain pending; specification 1.32 additionally
preflights weight-only handoffs and rejects unsafe partial learned attention
growth before any destination tensor is copied; specification 1.33 adds an
exact identity-initialized exception for contiguous appended depth only

## 2026-08-11 — step-512 failure localized; residual-parsimony repair implemented

The specification-1.35 campaign is preserved and intentionally stopped at its
durable step-640 checkpoint after the causal defect was localized; its safe
deployment incumbent stays at step zero. The latest persisted complete
step-512 selector is rejected at score `0.3251911400` versus protected
`0.3213162196`. Candidate/current position is `0.267023 m` versus
`0.251460 m`; velocity improves `1.093191 -> 1.040257 m/s`, but x regresses
`0.281775 -> 0.326179 m`, target coverage falls `0.37625 -> 0.36575`,
precision falls `0.357312 -> 0.347258`, collision F1 falls
`0.195489 -> 0.144186`, and identity-switch rate rises
`1.359% -> 1.723%`. The fixed position horizons change by
`+0.009407/+0.015413/+0.003433/+0.002079/-0.001947 m`; only one second
improves. The selector has complete support and the exact checkpoint audit
passes: all 48 attention tensors changed, all 177 inherited tensors remain
bitwise exact, exactly 48 complete finite Adam states are at step 512, and all
source/config/protocol/model hashes agree. This is a behavioral regression,
not corruption, scope leakage, or collapse.

Typed ablations localize the broad error. Node-only and relation-only
candidates score `0.374304` and `0.332082`; a half-strength complete attention
residual scores `0.328838`. Restoring only the node-y decoder row to exact zero
scores `0.308092` and improves current position (`0.251460 -> 0.247369 m`),
velocity (`1.093191 -> 1.038949 m/s`), collision F1
(`0.195489 -> 0.207048`), all three current axes, and every position horizon:
`0.265184/0.277452/0.309911/0.335387/0.357837 ->`
`0.256247/0.268004/0.294245/0.320580/0.344243 m`. It still fails two pooled
short-horizon coverage guards and 77 scenario-specific guards, so it is a
diagnostic rather than a promotable checkpoint. The dominant full-candidate
failure is `reference_pairs`: current position `0.212965 -> 0.340263 m`, driven
by x `0.242694 -> 0.523379 m`, even while its velocity improves.

The node decoder's x/y/z row L2 norms are
`0.012420/0.111429/0.012073`; corresponding energies are
`0.000154/0.012417/0.000146`. Specification 1.36 adds an opt-in, axis-neutral
`attention_node_complexity` objective: mean squared L2 row energy including
bias, with per-axis diagnostics and exact-zero contribution for historical
configs. Weight `1.0` gives `0.004239` loss and `0.07518` restoring-gradient
norm at the rejected checkpoint. It preserves forward behavior, shapes, all
axes, and evidence-supported acceleration; it is a soft inertial-complexity
prior, not a frozen y rule. Focused schedule/objective/config/checkpoint tests
report `312 passed in 32.87 s`; the three new focused tests report
`3 passed`. The existing live YAML was restored unchanged so its exact-resume
protocol was not silently mutated. The final `last.pt` is step 640 under
specification 1.35. Both one-shot launch services were cleanly booted out; the
only shutdown stderr is the expected multiprocessing cleanup warning for 14
semaphores. A separately recorded override and fresh protected-control
campaign are required before any scale rung.

Primary diagnostic artifacts:

- full checkpoint audit:
  `runs/20260811-170842-attention-aggregate-isolated-stage-a/attention_checkpoint_audit_step_000512.json`;
- node-only:
  `runs/20260811-223628-attention-node-only-step512/`;
- relation-only:
  `runs/20260811-224740-attention-relation-only-step512/`;
- half residual:
  `runs/20260811-225939-attention-half-step512/`;
- node-y restored to zero:
  `runs/20260811-231155-attention-without-node-y-step512/report.json`.

The repair is committed and pushed on `main` as `bbdb3ad`. A fresh one-shot
campaign is active at
`runs/20260811-234157-attention-node-parsimony-stage-a/`, weights-only from the
protected protocol-14 graph checkpoint. Trainer label
`com.polceanum.orpheus.attention-parsimony-20260811-234134` and immutable-source
supervisor label
`com.polceanum.orpheus.attention-parsimony-convergence-20260811-234134` have
each launched exactly once with empty stderr. Metadata records clean commit
`bbdb3ad2e75498708c4bdd36741df973bd45f66a`, PyTorch `2.10.0`, MPS built and
available, measurement on MPS, closed-loop state/dynamics on CPU, RGB-only
input, and oracle disabled. The resolved config records
`attention_node_complexity: 1.0`, 8,192 minimum updates, 65,536 balanced
episode draws, 32-episode selectors every 512 updates, and checkpoints every
128. The immutable supervisor enforces 4,096-update extensions, four-selector
1% plateau evidence, and a 24,576 hard limit. Initial protected-control
validation completed all 32 episodes in `1,034.57 s`. The resulting score and
public metrics exactly reproduce the protected control, including score
`0.3213162196`, current position `0.251460 m`, velocity `1.093191 m/s`, target
coverage `0.37625`, precision `0.357312`, collision F1 `0.195489`, trusted
identity-switch rate `1.3592%`, and coverage90 `93.3861%`. Independent exact
comparison proves all `225/225` model tensors and all `2,844/2,844` common
non-resource metrics are bitwise/equally identical; only the expected
specification metadata changes from 1.35 to 1.36.

The first balanced eight-update block is also complete. It uses the exact same
seeds/scenario order and `349` causal trajectories as the unregularized
predecessor, applies all eight updates, skips none, retains the complete
interaction gradient at `1.0`, records zero trusted identity switches over 61
associations, and lowers applied gradient norm `0.283628 -> 0.254750` at
essentially unchanged total loss (`0.489052 -> 0.489055`). The new x/y/z node
complexity energies are
`1.816e-6/1.376e-6/9.331e-6` (`4.174e-6` mean). The complete dynamics auditor
returns `pass`; its severe warning is the declared localized typed-output
budget, while the complete interaction/global update is not clipped. Peak
sampled RSS is `2,891,116,544` bytes versus predecessor
`2,936,651,776`. This is launch/objective integrity evidence, not accuracy,
plateau, or convergence.

The first complete eight-block trend window now reaches step 64. The dynamics
auditor returns `pass` with all 64 updates applied, exact eight-draw support
from each scenario, `2,461` causal trajectories, zero skips or terminal
failures, minimum complete-interaction retention `0.566722`, and stable sampled
RSS `2,891,116,544` bytes. Both one-shot launch services remain live once and
both stderr files remain empty. Exact schedule-matched comparison with the
unregularized predecessor is still effectively neutral and slightly adverse:
pooled current position is `+0.000237 m`, current velocity is `+0.005579 m/s`,
and position horizons at 0.10/0.25/0.50/0.75/1.00 seconds are
`+0.000262/+0.000278/+0.000404/+0.000731/+0.000758 m`; y improves slightly at
current and one-second horizons, while x worsens throughout. Trusted identity
switches are `4/385` versus `3/386`. Coverage support is unchanged. This is an
early training-window warning, not a fixed-manifest rejection: checkpoint 128
and the first trained selector at step 512 remain the next integrity and
accuracy authorities. No depth or width increase is authorized yet.

The durable step-128 checkpoint independently passes the complete attention
integrity audit. All `48/48` attention tensors changed; all `177/177`
inherited tensors remain bitwise exact; exactly 48 complete finite Adam states
belong to the attention parameters at optimizer step 128; protected rollout
checkpoints remain exact; and architecture, source, configuration, protocol,
manifest, and stored model hashes agree. The audit is persisted at
`runs/20260811-234157-attention-node-parsimony-stage-a/attention_checkpoint_audit_step_000128.json`.
Across the exact matched 16-block schedule, all 128 updates apply with 16 draws
from every scenario, `5,047` causal trajectories, support at every horizon,
minimum complete-interaction retention `0.373366`, and stable sampled RSS
`2,911,186,944` bytes. Relative to the unregularized predecessor, current
position improves `0.004581 m`; x/y improve `0.012395/0.001112 m` while z is
`0.001082 m` worse. All pooled position horizons improve by
`0.004981/0.004607/0.003291/0.001164/0.003709 m`; current and one-second
velocity improve, while 0.25 and 0.50 seconds regress by
`0.003955/0.015837 m/s`. Collision F1 is effectively flat but slightly worse
by `0.000519`; lifecycle precision/coverage improve slightly and uncertainty
median NLL improves. The open warning is trusted identity: `9/703` switches
versus `4/699` on the matched predecessor, with four of the five extra switches
concentrated in the step-128 block. Fixed selector 512 remains authoritative.
The checkpoint node x/y/z energies are
`0.000159/0.001337/0.00000994`; the regularizer is active but y remains
dominant, so its efficacy is not yet established.

The first post-checkpoint stress segment through step 152 also passes. The
historical event-heavy step-152 draw has the exact matched seeds, `343` causal
trajectories, all 13 objective terms, `0/50` trusted identity switches, finite
uncertainty, and a completely unclipped `0.702822` interaction update. Across
matched steps 128--152, all four balanced blocks apply with 1,482 causal
trajectories, full short/mid-horizon support, minimum complete retention `1.0`,
and unchanged sampled peak RSS. Relative to the predecessor, current x/y/z
improve `0.003282/0.004502/0.002316 m`, and every axis at every position
horizon improves; pooled improvements grow from `0.004031 m` at 0.10 seconds
to `0.014209 m` at one second. Lifecycle precision/coverage and coverage90
improve slightly. The remaining tradeoff is explicit: current velocity is
`0.030249 m/s` worse, 0.10-second velocity is `0.054660 m/s` worse, and pooled
collision F1 is `0.011124` lower, while one-second velocity improves
`0.030357 m/s`. The four extra identity switches in this window are entirely
the already-recorded step-128 spike; steps 136 and 152 exactly match the
predecessor switch counts. Continue unchanged to the fixed selector rather
than tuning from this heterogeneous event window.

The subsequent causal trajectory remains healthy through sampled step 208,
but it also demonstrates why cadence samples cannot authorize scaling. Step
184 is a short-only 0.10/0.25-second batch whose ungated association set
changes from 84 matched frames in the predecessor to 95 in the candidate; its
`0.457138 m` versus `0.358063 m` current RMSE therefore pools a different and
harder match set rather than proving a fixed-manifest collapse. The next
complete matched blocks at steps 192 and 200 recover: current position improves
`0.005257 m`, every pooled position horizon improves by
`0.007728/0.003066/0.001863/0.005322/0.004511 m`, current velocity improves
`0.003713 m/s`, identity switches are equal, and complete-interaction
retention rises from `0.460052` to `0.749123`. Including step 208 leaves a
small mixed diagnostic: current improves `0.003060 m`, four of five position
horizons improve by `0.001024--0.007022 m`, while 0.50 seconds regresses
`0.003930 m`, collision F1 falls `0.02439`, and lifecycle precision/coverage
fall `0.01160/0.00602`. The auditor still returns `pass`, all updates apply,
all scenarios and horizons retain support, no new identity switch appears,
RSS remains lower than the predecessor, and both launch stderr files remain
empty. This is neither collapse nor scale qualification; fixed selector 512
remains the first accuracy authority. The complete read-only report is
`runs/20260811-234157-attention-node-parsimony-stage-a/training_dynamics_audit_after_step_000185_through_000208.json`.

A refreshed primary-source review of the original Transformer, compute-optimal
scaling, Qwen3, Gemma 3, DeepSeek-V3, and V-JEPA 2 does not justify changing
the active small-token architecture. Orpheus already uses the applicable
modern dense ingredients: scaled multi-head attention, pre-RMSNorm, SwiGLU,
typed permutation-equivariant set tokens, zero-output residual growth, and
explicit bounded typed decoders. GQA, local/global attention, MLA, MoE, and
Flash-style kernels primarily address long-context KV memory, large-batch
throughput, or cluster economics and are not expected accuracy fixes for at
most 22 tokens. The useful scaling lesson is procedural: qualify the 3.00M
control; preserve its full data-only curve; then compare the exact-identity
3.53M depth-six rung, the 4.34M width-192 rung, and only afterward bounded
timestamped history, with parameter-proportional balanced data and disjoint
RGB-only validation/test/OOD gates. The planned 8.31M width-256/depth-six rung
remains the first single-CUDA candidate. No larger run has been launched.

The live specification-1.36 trajectory has now reached durable checkpoint
step 256 without interrupting the original one-shot process. The independent
checkpoint audit passes: all 48 attention tensors changed, all 177 inherited
tensors remain bitwise exact, complete finite Adam state belongs to exactly
the 48 attention parameters at step 256, all serialized state is finite, and
source/config/protocol/model hashes agree. Both protected `best_rollout.pt`
and `reference_rollout.pt` model states exactly equal the initializer. The
checkpoint hash is `ee34540134882277142bf0397cfbf364426570f8b3c85c1a8fb22b81ab104cc4`;
its model-state hash is
`f152b1e0014ce8a1bd8fba3be5ade59a3645e3d44263e55b196e3a8dcf2a72e9`.

Matched training evidence from steps 128--256 is mixed and remains
non-promotable. Current position is `0.008560 m` worse, driven by x/z
regressions of `0.015784/0.009884 m` while y improves `0.002188 m`; current
velocity improves `0.013364 m/s`. Position at 0.10/0.25 seconds regresses
`0.009078/0.005336 m`, while 0.50/0.75/1.00 seconds improves
`0.003742/0.005007/0.006838 m`. Four excess identity switches remain, but
collision F1, lifecycle precision/coverage, most velocity horizons,
uncertainty NLL, and memory improve. A contained step-240 force spike retains
`0.215624` at the complete interaction stage and is followed by a fully
unclipped step-248 block that improves every sampled position horizon. The
dynamics auditor returns `pass`, with all 256 updates applied, zero skips, no
terminal artifact, full scenario/horizon support, and stable `2.911 GB` peak
RSS. Fixed selector 512 remains authoritative.

The first step-256 audit invocation accidentally omitted protected paths and
returned a vacuous `protected_checkpoints_exactly_initial: true`. Rerunning
with both paths proved the artifacts intact. Specification 1.37 repairs the
offline tool: reports now include `protected_checkpoint_count`, return `null`
for an unchecked empty set, and can require a nonempty protected set. Focused
tests report `3 passed`; Ruff check passes after formatting. This does not
change the live 1.36 trainer, its tensors, or its protocol.

The first complete post-checkpoint window, exact matched steps 256--312, also
passes the training-dynamics auditor and remains mixed rather than converged.
All 312 optimizer updates apply, every eight-step block has complete balanced
scenario and horizon support, minimum complete-interaction retention is
`0.335188`, there is no terminal artifact, and sampled RSS remains flat at
`2,911,186,944` bytes. Relative to the unregularized predecessor, current
position improves `0.001599 m`; x improves at all five rollout horizons and
the pooled 0.50/0.75/1.00-second position errors improve
`0.001106/0.000509/0.003974 m`. The counter-evidence is equally important:
current velocity regresses `0.037677 m/s`, pooled 0.10/0.25-second position
regresses `0.001859/0.002192 m`, y regresses at four horizons, aggregate
collision F1 falls `0.011657`, and uncertainty median error rises `0.001063`.
Trusted identity improves by one switch. The historical step-280 force stress
is contained with complete-stage retention `0.335188`; steps 288--312 remain
finite and applied, and step 312 needs no local attention-node clipping. The
durable report is
`runs/20260811-234157-attention-node-parsimony-stage-a/training_dynamics_audit_after_step_000256_through_000312.json`.
This is healthy optimization evidence, not an accuracy promotion; fixed
selector 512 remains the first authoritative trained comparison.

The live trajectory has now reached durable checkpoint step 384. The strict
specification-1.37 offline audit, applied read-only to the immutable 1.36
checkpoint, passes with both protected artifacts required: all 48 attention
tensors changed; all 177 inherited tensors remain bitwise exact; exactly 48
complete finite Adam states belong to the attention module at step 384; every
serialized tensor is finite; and model/source/config/protocol hashes agree.
Both protected selector checkpoints still exactly equal the initializer. The
checkpoint SHA-256 is
`90b7997b61f2bcf91f232cefc167f7f69e4727b5cb35a34f0dc85a8af8233885`
and its model-state hash is
`d1544b10782df96f6f9c36f1de6c6aaf013020ebfb8719ca192769e3c5dece21`.

The exact matched 328--384 dynamics window passes operational gates but is an
accuracy regression. All updates apply, all eight scenarios and every horizon
remain supported, minimum complete-interaction retention is `0.429332`, no
draw is skipped, no terminal artifact exists, and RSS stays exactly
`2,911,186,944` bytes. Current position is `0.005024 m` worse and every pooled
position horizon is `0.001668--0.006286 m` worse. The regression is localized
to x, which is `0.008052--0.021339 m` worse at every horizon; y and z improve
at every horizon. Current velocity improves `0.003291 m/s`, collision F1 and
lifecycle support improve slightly, but one excess identity switch and worse
uncertainty NLL remain. The node/y decoder energy declined over steps 328--368
before a small rebound by step 384, so the penalty is active but has not yet
produced broad accuracy. A source audit found no axis-indexing, aggregation,
or selector-contract defect: the penalty treats all decoder rows identically,
and y acceleration can change contact timing and x impulses through the
structured dynamics. Scaling remains blocked pending fixed selector 512.
Durable evidence is in `attention_checkpoint_audit_step_000384.json` and
`training_dynamics_audit_after_step_000328_through_000384.json` in the run
directory.

Implementation verification on Python `3.10.20` / PyTorch `2.10.0`:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q \
  tests/unit/test_training_schedule.py tests/unit/test_training_objective_regressions.py \
  tests/unit/test_config.py tests/integration/test_checkpoint_roundtrip.py
# 312 passed in 32.87 s

PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -m 'not device'
# 716 passed, 5 skipped, 1 deselected in 213.71 s

PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q \
  tests/integration/test_rgb_measurements.py::test_roi_sampling_mps_training_mode_no_grad_uses_inference_path \
  tests/integration/test_rgb_measurements.py::test_roi_sampling_mps_training_native_bilinear_path_is_differentiable \
  tests/integration/test_rgb_measurements.py::test_global_rgb_cpu_detector_trains_and_roundtrips_with_mps_backbone \
  tests/unit/test_association.py::test_association_transfers_cost_to_cpu_without_mps_float64 \
  tests/unit/test_evaluation_parameter_update_metrics.py::test_directional_parameter_metrics_transfer_before_float64_accumulation
# 5 passed in 8.85 s on host MPS
```

Ruff format/check, compileall, `git diff --check`, and the full 8,192-update
dry-run with `training.loss_weights.attention_node_complexity=1.0` pass. The
dry run resolves 65,536 balanced episode draws, eight scenarios, and 32
RGB-only validation episodes.

## 2026-08-11 — specification-1.35 campaign passes durable step 384

The fresh aggregate-gradient attention campaign remains active at
`runs/20260811-170842-attention-aggregate-isolated-stage-a/` from clean source
commit `23ecf9d`. The trainer and convergence supervisor have each launched
once, both stderr files are empty, no `training_failure.json` exists, and
measurement remains on host MPS while closed-loop belief/dynamics execute on
CPU. Through step 384, all optimizer updates apply, every logged block contains
one draw from all eight scenarios, all 13 causal objectives remain supported,
no draw is skipped, no interaction update violates the `0.1` retention floor,
and maximum RSS remains exactly `2,991,591,424` bytes.

The independent step-384 checkpoint audit passes. All 48 attention tensors
changed, all 177 inherited tensors remain bitwise exact, exactly the 48
attention parameters own complete finite Adam state at step 384, every
serialized tensor is finite, and source/config/protocol hashes agree. The
checkpoint model-state hash is
`5278374307390e808816c3ecee491e6d66b4e5be7fc3d693cd349ae134b3b2d7`; the
durable report is
`runs/20260811-170842-attention-aggregate-isolated-stage-a/attention_checkpoint_audit_step_000384.json`.

The exact schedule-matched steps 328--384 window contains 2,655 candidate
causal trajectories. Relative to the predecessor on the same seeds,
scenarios, draws, frames, and rollout anchors, current position improves
`0.263840 -> 0.257064 m`, current velocity improves
`1.604568 -> 1.589278 m/s`, collision F1 improves
`0.178571 -> 0.194805`, and x improves at every horizon by
`0.004142--0.021893 m`. The result is still non-promotable: pooled horizon
RMSE worsens by `0.000955/0.001289/0.002553/0.002968/0.007237 m`, z worsens by
`0.005104--0.052711 m`, three later velocity horizons regress, lifecycle
precision/coverage slip by about `0.002`, and identity records six rather than
five switches. This is evidence that the repaired trajectory can correct the
earlier x weakness, not broad convergence.

The primary-source Transformer review does not expose a missing efficiency
mechanism for this at-most-22-token model. Dense scaled attention, multiple
heads, pre-normalization, RMSNorm, and SwiGLU are already present; GQA, MLA,
MoE, and Flash-style kernels chiefly target long-context memory or very large
compute. Compute-optimal and current video-world-model results instead support
the existing policy: scale data with parameters, use a plateau-triggered
cooldown only as a separately versioned experiment, and require disjoint
generalization. Capacity scaling therefore remains blocked until fixed
selector 512, repeated plateau evidence, and test/OOD non-regression pass.

Exact read-only commands run against the live campaign were:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python \
  scripts/audit_attention_checkpoint.py \
  --checkpoint runs/20260811-170842-attention-aggregate-isolated-stage-a/checkpoints/last.pt \
  --initial-checkpoint runs/20260811-170842-attention-aggregate-isolated-stage-a/checkpoints/validation_step_000000.pt \
  --config runs/20260811-170842-attention-aggregate-isolated-stage-a/config.resolved.yaml \
  --require-all-attention-changed \
  --require-complete-attention-optimizer-state \
  --output runs/20260811-170842-attention-aggregate-isolated-stage-a/attention_checkpoint_audit_step_000384.json
PYTHONPATH=. conda run --no-capture-output -n orpheus python \
  scripts/audit_training_dynamics.py \
  --run runs/20260811-170842-attention-aggregate-isolated-stage-a \
  --reference-run runs/20260811-063308-attention-node-isolated-stage-a \
  --after-step 328 --trend-window-blocks 8
```

Next: continue the immutable trajectory to the first trained 32-episode fixed
selector at step 512. Do not alter learning rate, add a cooldown, or launch a
larger rung from sampled training-window evidence.

## 2026-08-11 — attention campaign passes durable step 896

The immutable specification-1.31 attention-only campaign remains active under
the original trainer and convergence supervisor. Both launchd jobs have run
exactly once, both stderr files remain empty, no terminal optimizer artifact
exists, and the dynamics auditor passes through all 896 applied updates. Every
logged block contains one example from each of the eight declared scenarios,
there are no skipped draws or uncontained interaction clips, and maximum RSS
remains flat at `2,922,790,912` bytes.

The independent step-896 checkpoint audit passes. All 48 attention tensors
have changed, all 177 inherited tensors remain bitwise exact, exactly the 48
attention parameters own complete finite Adam state at step 896, every
serialized tensor is finite, source/config/protocol hashes agree, and the
protected step-zero incumbent/reference checkpoints remain unchanged. The
artifact is
`runs/20260811-063308-attention-node-isolated-stage-a/attention_checkpoint_step_000896_audit.json`.

The complete pooled training window at steps 840--896 has 2,169 causal
trajectory targets, current position RMSE `0.246462 m`
(`x/y/z = 0.270753/0.218706/0.247167 m`), current velocity RMSE
`1.228381 m/s`, and position-horizon RMSE
`0.263285/0.329719/0.414078/0.460056/0.487879 m`. Current coverage90 is
`97.90%`, identity is `5/298` (`1.678%`), collision F1 is `0.1607`,
drag/restitution observable counts are `153/28`, median uncertainty NLL is
`-0.7862`, median current/future correction improvements are positive, and
minimum complete-interaction retention is `0.6763`.

Against steps 776--832, current error is essentially flat
(`0.248962 -> 0.246462 m`) but every forecast horizon is worse, lifecycle
target coverage falls `40.15% -> 34.88%`, and collision F1 falls
`0.2373 -> 0.1607`; coverage, uncertainty, optimizer support, gradients, and
resources remain healthy. This is a real heterogeneous training-trend warning,
not matched checkpoint evidence. The fixed step-1024 RGB-only selector remains
the next accuracy/model-change decision; no scale promotion is authorized.

## 2026-08-11 — attempted step 988 exposes aggregate recursive-gradient gap

The specification-1.31 attention campaign is stopped and is not converged.
The trainer applied 987 finite supported updates, then correctly rejected
attempted optimizer step 988 before Adam because complete interaction-gradient
retention was `0.0971759`, below the configured `0.1` floor. The convergence
supervisor consumed `training_failure.json`, reported `CampaignIncompleteError`,
booted out the failed launch agent, and did not relaunch. The safe deployment
incumbent remains protected step zero; selector 1024 never ran.

The failure is finite and localized. All 13 objectives have support with 284
causal trajectories. Raw interaction norm is `15.1704`; the normal-force
decoder row is `10.9076`, node z is `2.1911`, and the largest shared block
gradient is `5.01609`. Per-invocation hooks individually obeyed their caps, but
144 recursive calls accumulated applied force/impulse output norms of
`0.219855/0.115811` around nominal `0.1` caps. Later decoder-row clipping cannot
remove the gradient already accumulated in shared attention.

Specification 1.35 repairs that gap by treating each typed node/collision/
force/impulse output cap as one aggregate per-draw L2 budget. With `K`
registered invocations, each hook receives `cap / sqrt(K)`, which bounds the
sum of applied squared norms while preserving single-call behavior exactly.
Forward inference, parameter/tensor shapes, and `WorldBelief` are unchanged.
Focused attention/config/checkpoint/auditor tests pass (`314 passed`); Ruff and
diff checks pass. A diagnostic exact-state replay from durable step 896 through
the same step-988 draw is the next gate. No capacity increase is authorized.

That replay is now complete at
`runs/20260811-162919-step988-aggregate-gradient-replay-v1/`. It reproduces the
same step-988 seeds, eight scenarios, 284 causal trajectories, and all 13
objective terms. Raw interaction norm falls `15.1704 -> 4.58029`, post-row norm
falls `10.2906 -> 1.54333`, complete retention rises `0.0971759 -> 0.647948`,
and the largest shared gradient falls `5.01609 -> 0.225929`. Aggregate applied
node/collision/force/impulse output norms are
`0.04939/0.00442/0.06029/0.00839`, all below `0.1`. The ordinary retention
assertion passed and the harness then stopped before Adam. The generic offline
auditor returns expected `fail` because the deliberate diagnostic stop is
persisted through the terminal-failure channel; the separate concise artifact
is `aggregate_gradient_replay_report.json`. This qualifies the repair only,
not weights, accuracy, convergence, or scaling. A fresh protected-control
campaign remains required.

Broad repair gates pass from clean commit candidate source on macOS with
Python `3.10.20` and PyTorch `2.10.0`: the complete non-device suite reports
`711 passed, 5 skipped, 1 deselected` in `176.51 s`; the restricted device
marker reports one expected MPS-unavailable skip; the five direct host-MPS
regressions pass in `7.76 s`; all 193 Python files pass Ruff format/check;
compileall succeeds; `git diff --check` is clean; and the attention-pilot CPU
dry run confirms 8,192 updates, batch eight, eight-way scenario balance,
65,536 nominal episode draws, RGB-only validation, and no fixed dataset. Host
training/replay metadata independently confirms MPS is built and available,
with measurement on MPS and closed-loop state on CPU.

The fresh specification-1.35 protected-control campaign is active at
`runs/20260811-170842-attention-aggregate-isolated-stage-a/` from clean commit
`23ecf9d`. Trainer launch label
`com.polceanum.orpheus.attention-aggregate-isolated-20260811-170751` and the
immutable-source convergence supervisor label
`com.polceanum.orpheus.attention-aggregate-convergence-20260811-170751` have
each run exactly once with empty stderr. Metadata records PyTorch `2.10.0`, MPS
built/available, measurement on MPS, closed loop on CPU, RGB-only input, no
oracle, and weights-only initialization from the protected step-64 graph
checkpoint. The supervisor enforces 8,192 minimum updates, 4,096-update
extensions, a four-selector/1% plateau rule, and a truthful 24,576 hard limit.
The mandatory 32-episode step-zero selector completed in `978.263 s` and is
exactly reproducible against the prior protected-control selector: all
`225/225` model tensors are bitwise equal and all `2,584/2,584` checkpoint
metrics are exactly equal, including selection score `0.3213162195855908`.
The trainer then entered its first balanced optimizer block and remains
genuinely active (`R` state, approximately `541%` CPU and `1.95 GB` RSS at the
18:28 host check); the launch count remains one and stderr is empty. No trained
metric, accuracy gain, plateau, generalization result, or capacity
authorization exists yet.

The first logged balanced block reached step 8 in `1,276.365 s` total. It uses
the same seeds and scenario order as the predecessor, applies all eight
optimizer updates, has zero skipped-gradient draws, `349` causal trajectories,
and the expected eight cold-start objective terms. Loss is `0.4890517` versus
the predecessor's `0.4890301`. Aggregate node/force output budgets reduce their
raw norms `0.69604/0.15816` to `0.01015/0.01350`; collision/impulse are
negligible. The resulting complete interaction coefficient is `1.0`, the
finite applied attention/global norm is `0.283628`, and no global clip occurs.
The dynamics auditor passes with no failures, duplicates, skipped draws,
uncontained interaction clips, or terminal artifact; its severe local-output
clip warning is expected observability of the new aggregate budget rather than
evidence that the complete interaction update was starved. Peak sampled RSS is
`2,936,651,776` bytes. This proves early optimizer health only, not an accuracy
trend or convergence.

The same run is healthy through logged step 32. All `32/32` updates apply;
each scenario appears exactly four times, with no skipped draws, duplicate
rows, terminal failure, or uncontained interaction clip. The four sampled
losses span `0.489052--5.764816` because cold-start support and heterogeneous
event/rollout batches differ; on every matched seed block they remain close to
the predecessor. Median pre-clip gradient is `0.661644`; only step 16 uses the
ordinary global clip, and minimum complete interaction retention is `0.716607`.
At steps 24 and 32 the repaired norms are `0.8720/0.4513`, versus predecessor
norms `4.8887/1.8296`, while their losses remain close or slightly lower. RSS
is nearly flat at `2.937--2.970 GB`. The whole-run auditor still passes; its
four-block physical window is deliberately incomplete and cannot substitute
for the first fixed trained selector.

The first complete eight-block training window now passes through step 64.
All `64/64` updates apply with exactly eight draws per scenario, `2,461`
causal trajectories, zero skipped draws or duplicate rows, no terminal or
uncontained interaction clip, and minimum complete retention `0.585590`.
Median/max raw gradient is `0.986608/1.707679`; four blocks use the ordinary
global clip. On the identical step-64 batch, the repair reduces raw gradient
`3.24257 -> 0.96806` while keeping loss close (`2.05477 -> 2.08984`), and all
13 objectives remain supported. RSS stays bounded at `2.937--2.992 GB`.

The complete heterogeneous training-window diagnostics are recorded without
promotion: current position RMSE is `0.267191 m` (`x/y/z =
0.282210/0.217194/0.295563 m`), current velocity RMSE is `1.460520 m/s`, and
0.1/0.25/0.5/0.75/1.0-second position RMSE is
`0.264288/0.303162/0.364353/0.420610/0.443199 m`. At 1 second, axis RMSE is
`x/y/z = 0.614303/0.231116/0.398114 m`; x is the hardest axis. Horizon target
coverage is `98.73/98.73/94.58/94.58/94.58%`; current coverage90 is `96.60%`.
There are three trusted switches in 386 associations (`0.7772%`), collision F1
is `0.176`, lifecycle target coverage/precision is `0.39135/0.37896`, median
uncertainty position NLL is `-0.84906`, and drag/restitution observability has
`167/39` objects. These are on-policy training-window health diagnostics, not
the fixed step-512 selector or generalization evidence.

The first durable trained checkpoint at step 128 now passes both artifact and
whole-run audits. All `128/128` updates apply with exactly 16 draws per
scenario, no skips, duplicate rows, terminal failure, or uncontained
interaction clip; sampled RSS remains `2.992 GB`. The checkpoint contains 225
model tensors: all 48 attention tensors changed, all 177 inherited tensors are
bitwise unchanged, and exactly the 48 attention parameters own finite Adam
state at step 128. Its model hash is
`417baf91fadd6797052df2eefd56f7812bcc28e8fe71d4c6a72920d1540bc7ac`;
the reusable report is `attention_checkpoint_audit_step_000128.json` beside
the run.

The matched steps 72--128 training window is mixed and therefore remains a
warning rather than a promotion. Relative to the failed predecessor on the
same seeds, current position is nearly flat (`0.322644 -> 0.323073 m`) but all
five pooled position horizons regress slightly:
`0.326410/0.347188/0.375946/0.408533/0.427788 ->
0.327745/0.350080/0.378844/0.409725/0.432133 m`. Current y/z improve while x
regresses; short/mid velocity improves, 0.75/1.0-second velocity regresses
slightly. Identity improves from four switches/310 associations to one/313,
collision F1 is nearly flat (`0.12346 -> 0.12579`), lifecycle coverage and
precision regress by about half a percentage point, and median uncertainty
NLL regresses slightly (`-0.70091 -> -0.69714`). Minimum complete interaction
retention improves `0.22271 -> 0.50595`. This is a genuine heterogeneous trend
warning; the protocol intentionally checkpoints every 128 updates but runs its
expensive fixed 32-episode selector every 512 updates. No step-128 validation
was expected or missed. Continue unchanged to the fixed step-512 selector.

The next complete matched training window, steps 128--184, confirms a real
event/identity tradeoff without optimizer collapse. Relative to the failed
predecessor on identical seeds, 0.5/0.75/1.0-second position RMSE improves
`0.346603/0.407169/0.444389 -> 0.340015/0.399048/0.436764 m`, driven mainly by
the z axis at one second (`0.461908 -> 0.439423 m`); current/0.1/0.25-second
position regress slightly. Identity improves from nine switches/386
associations to two/380, while collision F1 regresses `0.242775 -> 0.210526`
and lifecycle precision/coverage regress `0.37173/0.40336 ->
0.36735/0.39706`. Velocity is mixed across horizons; median uncertainty NLL
is essentially flat/slightly better. All eight scenarios have eight draws,
all 13 objectives remain supported, no update skips or fails, and RSS remains
`2.992 GB`. One step-136 force-decoder spike is contained by the row/global
hierarchy with complete retention `0.23550` and maximum shared gradient only
`0.06889`; it does not recur in the next six logged blocks. Continue without
promotion or protocol mutation to fixed validation at step 512.

The second durable trained checkpoint at step 256 also passes exact artifact
and whole-run audits. All 48 attention tensors changed, all 177 inherited
tensors remain bitwise unchanged, exactly the 48 attention parameters own
finite Adam state at step 256, and the model hash is
`3635dae3b120a41236cbcceaf4e450814c36774ffb4709f02d2bb8f767ca6a5b`.
Across all 256 updates the 32 logged blocks contain exactly 32 draws per
scenario, zero skipped draws, duplicate rows, uncontained interaction clips,
or terminal failures; RSS remains bounded at `2.992 GB`. The reusable artifact
report is `attention_checkpoint_audit_step_000256.json` beside the run.

The complete matched steps 192--248 window is not promotable. Current position
is nearly flat (`0.298443 -> 0.299372 m`); 0.25/0.75/1.0-second position
improves only slightly while 0.1/0.5 seconds regress slightly. Every velocity
horizon regresses, collision F1 falls `0.222222 -> 0.189873` (including
one-second F1 `0.214286 -> 0.066667`), and median uncertainty NLL weakens
slightly. Identity improves from five switches/376 associations to four/374;
lifecycle coverage/precision are unchanged. The late eight-objective step-232
row exactly matches the predecessor's sparse batch support and is not support
collapse. This is a second heterogeneous accuracy warning with clean optimizer
health; continue to the declared step-512 fixed selector without promotion.

The next logged block at step 264 also passes the read-only dynamics audit.
All `264/264` optimizer updates have applied with exact eight-scenario balance,
no skipped draw, duplicate row, terminal failure, or uncontained interaction
clip. The block has `313` causal trajectories and all 13 objective terms; RSS
remains exactly at the sampled `2,991,591,424`-byte plateau. Its raw interaction
gradient reaches `2.850016`, but the semantic/row hierarchy contains it at
`0.350875` complete retention, above the declared `0.1` rejection floor. The
very small force/node per-invocation coefficients remain visible warnings, not
hidden evidence of collapse. Trainer and supervisor each remain active once,
both stderr files are empty, and no `training_failure.json` exists.

The primary-source Transformer refresh does not change the promotion decision.
The current 3.00M model already implements the applicable short-set backbone:
dense scaled dot-product attention, pre-RMSNorm, SwiGLU, residual paths, typed
permutation-equivariant tokens, and bounded typed decoders. Llama 3 and the
original Transformer reinforce long, data-rich training of a stable dense
backbone; Chinchilla reinforces scaling examples with parameters. Gemma 3,
DeepSeek-V3, and FlashAttention target long-context KV memory, sparse capacity,
or accelerator bandwidth that are not bottlenecks for at most 22 tokens.
V-JEPA 2 and ObjectForesight support the later complementary path: large-scale
latent video pretraining plus explicit object-level trajectories. They do not
justify replacing authoritative `WorldBelief` or enlarging a rung that still
regresses velocity/event slices. The fixed step-512 selector, repeated
selectors, plateau, disjoint test/OOD, and broad non-regression therefore remain
mandatory before the prepared depth-six handoff is used.

The dynamics auditor now has an optional matched-reference mode so repeated
manual `jq` comparisons cannot accidentally mix different training samples.
It canonicalizes both append-only streams, aligns by optimizer step, and hard-
fails missing reference steps or differences in seeds, scenario order, draw
index, frame-window bounds, or rollout-anchor selection. Each aligned set is
then independently count-pooled before reporting nested signed deltas across
the complete physical trend summary. Focused tests report `15 passed`; Ruff
check/format and diff checks pass.

Applied to the corrected run versus its deterministic predecessor at steps
192--272, all 11 logged blocks align with zero schedule mismatch. The repaired
candidate has current position `0.293305` versus `0.289235 m` (`+0.004070`),
with x improving `0.001046 m` but y/z worsening `0.003712/0.010458 m`. Position
horizons change by `+0.002671/+0.001173/+0.001576/-0.001407/-0.000196 m`, while
all velocity horizons regress by `0.005839--0.046315 m/s`. Collision F1 falls
`0.197044 -> 0.165049`, mostly from the one-second slice; identity remains five
switches with four more associations, lifecycle precision/coverage weaken by
`0.002964/0.003870`, and coverage90 weakens by `0.000843`. Median uncertainty
NLL improves by `-0.008281` while its worst sampled value weakens. This is
schedule-controlled diagnostic evidence of a broad tradeoff, not a selector;
continue unchanged to fixed step 512 without promotion or scaling.

The historically important step-280 schedule position subsequently completes
without optimizer failure. It has the exact predecessor seeds/scenarios,
`145` causal trajectories, all 13 objective terms, raw interaction norm
`2.160692`, and complete retention `0.462814`; no uncontained clip occurs.
Aggregate applied node/collision/force/impulse output norms are
`0.05592/0.01675/0.06437/0.00857`, all below the configured `0.1` budgets.
Against the matched predecessor, current position improves `0.242093 ->
0.238464 m`, all axes improve, and the five position horizons improve by
`0.00687/0.01266/0.02120/0.03090/0.02051 m`. Current velocity improves
`0.01068 m/s`, but velocity at 0.25/0.5/1.0 seconds regresses by
`0.11140/0.00977/0.22061 m/s`; collision F1 falls `0.5 -> 0.3333` while
identity, lifecycle, support, and coverage are exact. The corrected gradient
hierarchy is functioning, but physical accuracy remains mixed.

The complete schedule-matched steps 264--320 window now closes with eight
logged blocks, exact eight-draw exposure per scenario, `2,198` causal
trajectories, no skip/failure/uncontained clip, minimum complete retention
`0.350875`, median/max raw gradient `1.060409/2.850016`, and unchanged sampled
RSS `2,991,591,424` bytes. Candidate current position is effectively flat
against the predecessor (`0.237349 -> 0.237564 m`): x regresses `0.007085 m`,
y improves `0.007561 m`, and z regresses only `0.000529 m`. Current velocity
improves `0.008360 m/s`.

The forecast tradeoff remains broad. Position at 0.1/0.25 seconds improves by
`0.006267/0.002683 m`, while 0.5/0.75/1.0 seconds regress by
`0.003060/0.003405/0.005307 m`. Every x horizon regresses by
`0.005056--0.011497 m`; every y horizon and four of five z horizons improve.
Velocity at 0.1 seconds improves `0.001307 m/s`, but the other four horizons
regress by `0.001952--0.045895 m/s`. Collision F1 is almost flat
(`0.184615 -> 0.180451`) with large offsetting horizon changes. Lifecycle
precision/coverage improve about `0.0012`, coverage90 falls `0.002660`, median
uncertainty NLL weakens `0.001046` while its worst sampled value improves, and
drag/restitution observability each gain one object. Identity is the clearest
regression: six switches/297 associations versus three/294. This is a complete
deterministic training-trend warning with healthy optimization, not fixed
validation; continue unchanged to selector 512 without promotion or scaling.

## 2026-08-11 — pooled convergence-trend observability implemented

The whole-run auditor previously proved optimizer/support/resource integrity
and exposed physical distributions, but consecutive axis/horizon comparisons
were one-off calculations. It now emits configurable non-overlapping windows
(`--trend-window-blocks`, default eight logged cadence blocks), marks incomplete
tails, and pools physical sufficient statistics before deriving metrics. RMSE
uses summed SSE/coordinate counts; coverage and precision use pooled counts;
identity uses switches/associations; collision F1 uses pooled TP/FP/FN. Each
window includes current position axes and velocity, all configured position and
velocity horizons, forecast coverage, lifecycle support, uncertainty,
correction improvement, event F1, drag/restitution observability, causal
support, scenario balance, gradient retention, and memory. Focused auditor
tests pass (`12 passed`).

The live run remains unaffected and its updated auditor returns `pass` through
step 800 with no failures. Its latest complete window, steps 712--768, has
2,096 causal trajectory targets, all 13 objective terms in every logged block,
balanced eight-way scenario exposure, current position RMSE `0.184647 m`
(`x/y/z = 0.182561/0.170300/0.199884 m`), current velocity RMSE
`1.151526 m/s`, and position horizons
`0.184250/0.209727/0.265618/0.320229/0.347584 m`. Identity is `2/314`,
current coverage90 is `99.37%`, pooled collision F1 is `0.1333`, drag/
restitution observable counts are `191/30`, uncertainty NLL median is
`-0.9156`, and minimum complete-interaction retention is `0.6885` with stable
`2,922,790,912`-byte RSS.

The step-776--800 tail was correctly labelled incomplete; it subsequently
closed at step 832 with eight blocks and 2,628 causal targets. Against the
exceptionally strong 712--768 sample it regresses current position
`0.184647 -> 0.248962 m`, all position horizons
`0.184250/0.209727/0.265618/0.320229/0.347584 ->
0.242408/0.272900/0.333325/0.364523/0.370667 m`, identity `2/314 -> 6/371`,
coverage90 `99.37% -> 97.68%`, and median uncertainty NLL
`-0.9156 -> -0.7432`. Against 648--704 it still improves current, 0.10, 0.25,
0.50, and 1.00-second position; only 0.75-second position is modestly worse.
Lifecycle target coverage/precision improve to `40.15%/37.60%`, forecast
coverage is `98.08--100%`, collision F1 improves to `0.2373`, drag/restitution
observability is `164/41`, median current/future corrections remain positive,
minimum complete-interaction retention is `0.7384`, RSS stays exactly flat,
and the audit has no failure. Velocity is horizon-mixed. This is real sampled
optimization wobble, not broad collapse or matched evidence; fixed selector
1024 remains the next promotion/model-change decision.

## 2026-08-11 — function-preserving depth growth implemented

The next depth rung no longer needs to discard a future qualified four-block
attention function. The weight-only loader now recognizes exactly one safe
partial transfer: contiguous zero-based attention blocks appended after a
complete inherited stack. It copies every inherited tensor strictly, retains
ordinary finite internal initialization in the new blocks, and zeros each new
MHA output weight/bias and SwiGLU output weight. Because both pre-norm residual
branches then emit exact zero, focused tests prove the shallow and grown token
streams plus learned decoded outputs are equal at `rtol=0, atol=0` before any
optimizer update.

The transform remains fail-atomic. A deliberately malformed source missing an
output tensor from inherited block two is rejected and every destination
tensor remains bitwise unchanged. Width changes, holes, reordered blocks, and
missing projections/decoders are still unsupported. The loader records grown
block indices `(4, 5)` in returned initialization provenance. Focused
checkpoint tests pass (`31 passed`). The trainer durably records the transform,
source checkpoint, and appended indices in `run_metadata.json`; exact resume
preserves that record. A shape-compatible four-to-eight-head change is also
rejected because the complete model/runtime/simulator semantics must match
except for increased `attention_layers`. This is scaling-path infrastructure,
not evidence that the current model has converged or that depth six should
launch.

The immutable specification-1.31 attention campaign is unaffected by these
repository changes and remains live. Its audit passes through update 720 with
all 720 updates applied, exactly 90 sampled blocks for each of eight scenarios,
zero skipped draws, zero terminal/uncontained failure, empty trainer/supervisor
stderr, and bounded maximum RSS `2,922,790,912` bytes. Selector 512 remains
rejected; selector 1024 remains the next matched accuracy decision.

The next durable boundary at step 768 also passes both audits. All 48 attention
tensors have changed, all 177 inherited tensors remain exact, optimizer state
belongs only to all 48 attention tensors at Adam step 768, protected checkpoint
hashes are unchanged, and every serialized tensor is finite. The dynamics
audit records all 768 updates applied, 96 logged blocks per scenario, zero
skips/failures/uncontained interaction clips, and unchanged `2,922,790,912`-
byte peak RSS. The audit artifact is
`runs/20260811-063308-attention-node-isolated-stage-a/attention_checkpoint_step_000768_audit.json`.

The equal sampled training window 712--768 improves 648--704 on current and
every horizon: current RMSE `0.379947 -> 0.184647 m`; 0.10/0.25/0.50/0.75/1.00
second RMSE `0.382316/0.405186/0.357122/0.341593/0.385963 ->
0.184250/0.209727/0.265618/0.320229/0.347584 m`. Current x/y/z RMSE becomes
`0.182561/0.170300/0.199884 m`, trusted identity improves from `6/326` to
`2/314`, and current-state coverage90 rises `96.08% -> 99.37%` with comparable
causal support (`2,122 -> 2,096`). Minimum complete-interaction retention
remains healthy (`0.6885`) and maximum raw gradient is `1.4524`. These are
heterogeneous sampled batches, not a fixed-manifest selector; they justify
continuing unchanged to selector 1024, not scaling early.

Final repository gates after the depth-handoff semantic repair pass: focused
checkpoint tests `31 passed`; complete non-device suite `709 passed, 5 skipped,
1 deselected in 214.35 s`; host MPS marker `1 passed, 714 deselected`; Ruff
format/check, compileall, and diff check pass.

## 2026-08-11 — scaling handoff integrity repaired

The modern-Transformer review still supports gradual evidence-gated scaling,
but the current 3.00M rung has not qualified: fixed selector 512 was rejected
at score `0.330772` versus the protected `0.321316`, with pooled current
position RMSE `0.295016` versus `0.251460 m` and broad x/z and scenario
regressions. The mutable trajectory remains healthy through update 720: every
update is applied, all eight scenarios have 90 logged blocks, no skipped draw,
terminal failure, or uncontained interaction clip exists, memory is bounded at
`2,922,790,912` bytes, and the offline dynamics audit returns `pass`. This is
continued optimization evidence, not a scale authorization; selector 1024 is
the next comparable decision point.

The equal eight-block training window at steps 648--704 is mixed against
584--640. It improves pooled 0.50/0.75/1.00-second RMSE from
`0.4035/0.4669/0.5161` to `0.3571/0.3416/0.3860 m`, but worsens current,
0.10, and 0.25-second RMSE from `0.2961/0.2984/0.3475` to
`0.3799/0.3823/0.4052 m`; x/z current error and trusted identity also worsen,
while y and long-horizon z improve. Coverage90 is essentially flat
(`96.11% -> 96.08%`), support is comparable (`2,054 -> 2,122` targets), and
minimum shared-stage retention improves (`0.6927 -> 0.9072`). These are
heterogeneous training samples, not a fixed-manifest gain or regression, so no
architecture or optimizer change is justified before selector 1024.

A pre-scale code audit found that the weight-only loader could accept a trained
four-block attention checkpoint for a six-block destination because the added
block keys were under the allowed attention prefix. Those random blocks change
the representations seen by learned typed decoders before training, violating
the zero-output/function-preserving handoff contract. The loader now preflights
all source/destination keys and tensor shapes before copying, rejects any
partially present allowed module prefix, and leaves a rejected destination
bitwise unchanged. The active run is unaffected: it loaded the entire new
attention prefix from the graph-only control under immutable commit `5b2da41`.

The gradual order is now consistently data-only, depth six, width 192, bounded
timestamped history, then width-256/depth-six single CUDA. Architecture rungs
start from the structured graph control until an exact identity-preserving
growth transform exists; the accepted smaller attention model remains the
fixed non-regression reference. Focused checkpoint tests pass (`29 passed`);
the complete non-device suite passes (`707 passed, 5 skipped, 1 deselected in
236.26 s`); Ruff format/check, compileall, diff check, and the unchanged
8,192-update/65,536-draw CPU dry-run pass.

## 2026-08-11 — accumulated node-gradient repair; small-rung campaign active

The trainer and supervisor for
`runs/20260811-042704-attention-impulse-isolated-stage-a/` are stopped. The
campaign applied 59 finite supported updates but never reached a durable
trained checkpoint or selector. Attempted update 60 was rejected before Adam
because complete interaction retention fell to `0.0850405`, below the declared
`0.1`. Its exact balanced seeds are
`13760,11713,9514,12171,15788,5701,3510,15031`; all eight scenarios and 332
causal trajectory targets were present, so this is not support collapse.

The exact diagnostic run
`runs/20260811-053408-step60-retention-replay-v3/` reproduces every comparable
model/data field at logged updates 8--56 (400 fields at step 8, 454 at most
later rows, zero mismatches after excluding timing/RSS) and atomically records
the rejected update. Its raw/post-existing-row interaction norm is
`28.2744/11.7591`. The node decoder is `11.6617`, dominated by world-y
`11.5014`; the largest shared non-decoder attention gradient is only
`0.124876`. Force isolation reduces its raw `25.7085` group to `1.0`; collision
and impulse are not the remaining cause. The offline dynamics audit now
correctly returns `fail` because the run has a terminal optimizer artifact.
The concise report is `step60_retention_failure_report.json` beside the run.

Specification 1.31 adds `training.attention_node_grad_clip_norm`; the active
profile uses `1.0`. It jointly clips the accumulated x/y/z node decoder before
the existing collision, force, impulse, complete-interaction, and global
hierarchy. Forward values, tensor shapes, inference, the 3,004,656 parameter
count, and `WorldBelief` contracts are unchanged. A direct reconstruction of
the failed gradient gives post-row norm `1.81140` and retention `0.552059`.
Structured retention failures now persist attempted step, zero applied-update
marker, seeds/scenarios, support/physical metrics, and all gradient levels;
the offline auditor treats a terminal numerical/retention artifact as a hard
failure even if sampled JSONL rows were healthy.

The fresh causal repair replay at
`runs/20260811-055605-step60-node-row-repair-replay-v1/` starts weights-only
from the same protected graph control. Its 32-episode initial selector is
exact: 225/225 tensors bitwise equal, 2,583/2,583 comparable metrics equal,
model hash `1354bdfc...f91`, and score `0.3213162196`. It uses host MPS for RGB,
CPU for closed loop, float32, RGB-only runtime, and no oracle. It reaches the
same attempted update-60 seeds with all eight scenarios and 332 causal targets;
raw/post-row interaction norm is `1.96175/1.76884`, complete-stage retention is
`0.565343`, and maximum shared non-decoder norm is `0.004064`. The wrapper
deliberately stops before Adam after the retention check so the diagnostic is
non-promotable. Its report is `node_row_repair_report.json` beside the run.

The Transformer scaling review confirms the current block already uses the
relevant modern core: RMS pre-normalization, scaled multi-head attention,
SwiGLU, residual paths, typed set tokens, and zero-output growth. GQA/local
attention/FlashAttention address long-token inference or kernel bottlenecks
that do not exist at at most 22 structured tokens; RoPE remains invalid for
unordered slots. The evidence-gated ladder remains 3.00M control, 3.53M
depth-six, 4.34M width-192, then 8.31M width-256/depth-six on CUDA, with minimum
parameter-proportional balanced exposures of 8,192/9,728/12,288/23,040 updates.
No larger rung is authorized until a clean specification-1.31 3.00M campaign
completes fixed selectors and the declared plateau without broad regression.

Verification at this boundary passes: affected config/trainer/checkpoint/
entrypoint/auditor tests reported `294 passed` before the final hierarchy
assertion; the complete non-device suite then reported `706 passed, 5 skipped,
1 deselected in 173.60 s`; the host device marker reported `1 passed, 711
deselected in 3.05 s`; five direct MPS regressions reported `5 passed in
7.76 s`. Ruff format (`193 files already formatted`), Ruff check, compileall,
and `git diff --check` pass. The host dry run resolves PyTorch `2.10.0`, MPS
built/available, MPS RGB measurement, CPU closed loop, float32, 8,192 updates,
65,536 balanced episode draws, eight scenarios, RGB-only evaluation, and no
oracle.

A clean sustained relaunch is active at
`runs/20260811-063308-attention-node-isolated-stage-a/` from committed source
`5b2da41dbc7467d86fb1b2fe3b3be2ca349df612`, with dirty source false and
runtime fingerprint
`703fe9c9b5775f32d8f04fb85d30cf9d92a5715e56dd31cdf5ca7a22e05ef42a`.
The one-shot Standard trainer
`com.polceanum.orpheus.attention-node-isolated-20260811-063238` and immutable-
source supervisor
`com.polceanum.orpheus.attention-node-convergence-20260811-063238` are both
running once with empty stderr. Host placement is MPS RGB measurement plus CPU
closed loop, float32, RGB-only, and no oracle. The complete 32-episode initial
selector persisted score `0.3213162196`; exact audit finds all 225 tensors
bitwise equal and all 2,578 comparable metric fields equal to the preceding
protected control, with only the expected protocol hash different. The
supervisor enforces 8,192 minimum updates, 4,096-update extensions, a 1,024-
update tail, 1% plateau threshold, and a 24,576 hard limit. The live auditor
passes through 128 applied updates, proving that the former unsampled update-60
pre-Adam failure no longer recurs. Across 16 logged blocks every scenario
appears exactly 16 times, causal support spans 154--471 targets, skipped draws
remain zero, no uncontained interaction clip or terminal failure exists, and
maximum RSS is `2,896,859,136` bytes. The independent durable-checkpoint audit
passes: all 177 inherited tensors remain bitwise exact, all 48 attention
tensors changed, exactly those 48 parameters own finite Adam state at step 128,
and architecture/source/protocol/model hashes agree. Cumulative trusted
identity switches are 7/696 (`1.006%`), position coverage90 is `91.01%`, and
weighted position RMSE has support at every horizon (`0.2958/0.3253/0.3714/
0.4145/0.4347 m`). X remains the hardest axis (`0.3386 -> 0.5963 m` from 0.1
to 1.0 seconds), while y remains comparatively flat (`0.2494 -> 0.2537 m`).
The checkpoint audit is
`attention_checkpoint_step_000128_audit.json` beside the run. Training remains
active and the live audit also passes the historical update-152 stress
position: all 152 updates apply, each scenario appears 19 times across logged
blocks, cumulative trusted identity is 11/820 (`1.34%`), and update 152 retains
`0.344214` at the complete interaction stage with 343 causal targets and every
horizon supported. It subsequently passes the former catastrophic update-200
impulse boundary with raw/applied gradient `1.14436/1.0`, impulse multiplier/
additive norms `0.18604/0.00781`, `0.873850` complete-stage retention, 339
causal targets, and every horizon supported. The cumulative audit passes all
200 updates with each scenario represented 25 times, zero skips/failures/
uncontained clips, trusted identity 14/1,127 (`1.24%`), pooled coverage90
`90.25%`, finite uncertainty, and bounded `2,922,790,912`-byte RSS.

The independent step-256 checkpoint audit also passes: all 177 inherited
tensors remain exact, all 48 attention tensors changed, exactly those 48 own
finite Adam state at step 256, and every architecture/source/protocol/model
hash agrees. The report is `attention_checkpoint_step_000256_audit.json`
beside the run. The fresh trajectory then passes the historically recurrent
update-280 boundary. Raw interaction is `2.86878`, versus `52.9646` in the
normalized campaign and `17.7050` after collision-only isolation; accumulated
node/force norms are `0.76515/2.74932`, post-row norm is `1.29273`, and complete
retention is `0.348580`. The update applies with 145 causal targets, all 13
objective terms, every horizon, finite uncertainty, and no skip. Cumulatively,
all 280 updates pass with each scenario represented 35 times, no terminal or
uncontained failure, trusted identity 19/1,532 (`1.24%`), coverage90 `90.27%`,
and bounded memory. These are optimizer/training-window health results, not a
trained fixed selector, accuracy promotion, generalization, plateau,
convergence, or capacity result.

The unchanged campaign now has a second independently audited boundary at
step 384. The checkpoint audit passes with all 177 inherited tensors bitwise
exact, all 48 attention tensors changed, exactly those 48 parameters owning
finite Adam state at step 384, and matching architecture, source, runtime,
protocol, model-state, and protected-control hashes. The checkpoint file
SHA-256 is `bcfa31c34e8d7084bd2f256e4d11fbdd850083e5eaf9933845607f077f21e179`;
the report is `attention_checkpoint_step_000384_audit.json` beside the run.
The live dynamics audit passes all 384 applied updates with no skipped draw,
terminal failure, or uncontained interaction clip. Each of the eight scenarios
appears exactly 48 times in the cadence rows; cumulative causal trajectory
support is 15,083 targets, trusted identity is 26/2,105 (`1.235%`), pooled
position coverage90 is 8,548/9,462 (`90.34%`), and maximum RSS remains
`2,922,790,912` bytes. Weighted position RMSE has support at every horizon and
is `0.2926/0.3206/0.3651/0.4123/0.4515 m` at 0.1/0.25/0.5/0.75/1.0 seconds.
X remains the hardest axis and reaches `0.6174 m` at 1 second, versus
`0.2523 m` for y and `0.4083 m` for z. These are balanced training-window and
state-integrity diagnostics only. The first trained fixed selector at step 512
is still pending, so no accuracy, generalization, convergence, plateau, or
capacity promotion is claimed.

The first trained fixed selector at step 512 is complete and rejected. All 32
RGB-only validation episodes (`100000`--`100031`) completed, with four episodes
for each of the eight scenarios and no support failure. The latest persisted
candidate score is `0.3251911400`, worse than protected step-zero
`0.3213162196`; `selection_accepted=0` and the safe incumbent remains step
zero. Pooled current position RMSE regresses `0.251460 -> 0.267023 m`, target
coverage `0.37625 -> 0.36575`, prediction precision
`0.357312 -> 0.347258`, and x/z current RMSE
`0.281775/0.263691 -> 0.326179/0.268807 m`; y and velocity improve. The
largest scenario failure is reference pairs, especially its x trajectory. The
comparison records 113 incumbent guardrail failures plus the worse selection
score. Position coverage90 remains stable at `93.43%`, so the principal
failure is point/state, events, identity, lifecycle, and broad scenario
non-regression rather than nonfinite uncertainty.

The candidate checkpoint itself is valid: all 177 inherited tensors remain
bitwise exact, all 48 attention tensors changed, exactly those 48 own finite
Adam state at step 512, and every architecture/source/runtime/protocol/model
hash passes. The report is `attention_checkpoint_audit_step_000512.json`
beside the run; checkpoint SHA-256 is
`23e0f5dcd1b429cca8e8686a77a38fcbc45a11fa44fd81b3251403f15ab382fb`.
The dynamics audit passes all 512 applied updates with exact 64-block scenario
balance, zero skips/terminal failures/uncontained clips, and bounded memory.
This localizes the failure to the learned attention residual rather than
corruption, scope drift, missing support, optimizer collapse, or a scheduler
jump. The mutable trajectory remains active, as required for causal repair of
rejected candidates; this first rejection is not plateau evidence and does not
authorize capacity scaling.

The rejected mutable trajectory has continued through step 576 without
replacing the step-zero incumbent. The full dynamics audit passes all 576
updates with exact 72-block support for each scenario, zero skipped or terminal
updates, no uncontained interaction clip, and unchanged peak RSS. Two local
typed-gradient outliers occur at steps 560 and 568. Step 560 has raw node/force
norms `3.822/19.941` (z `3.442`, tangent force `19.364`); semantic rows reduce
the interaction to `2.839`, leaving `0.3522` complete shared-stage retention.
Step 568 has raw force `6.634`; rows reduce it to `1.325`, leaving `0.7547`
shared-stage retention. Both exceed the `0.1` fail-fast threshold after local
isolation and apply with full causal/objective support. Their lower raw-to-final
coefficients remain visible warnings, not uncontained failures.

An equal eight-block training-window comparison shows a possible repair
direction, not held-out promotion. Step 456--512 versus 520--576 weighted
position RMSE changes across 0.1/0.25/0.5/0.75/1.0 seconds from
`0.2508/0.3025/0.3707/0.4452/0.4713` to
`0.2229/0.2694/0.3572/0.4052/0.4273 m`. X and z improve at every horizon;
identity changes `4/324 -> 4/346`; coverage90 slips `90.49% -> 89.17%`; y at
0.5 seconds worsens `0.2597 -> 0.3098 m`. The fixed selector at step 1024 must
confirm whether this is real generalization repair. No model, optimizer,
curriculum, incumbent, or scale decision changes at this boundary.

The next equal balanced training window, steps 584--640, reverses the sampled
repair trend. Relative to 520--576, weighted position RMSE at
0.1/0.25/0.5/0.75/1.0 seconds changes from
`0.2229/0.2694/0.3572/0.4052/0.4273` to
`0.2984/0.3475/0.4035/0.4669/0.5161 m`; x and z worsen at every horizon.
Identity improves from 4/346 (`1.16%`) to 2/262 (`0.76%`), but coverage90 falls
again from `89.17%` to `88.10%`. The windows have exact scenario balance but
different deterministic seeds, window difficulty, and support counts
(`2,627` versus `2,054` causal targets), so this is evidence of an unstable
training-sample trajectory rather than a matched validation regression.

The full dynamics audit still passes all 640 updates with exact 80-block
scenario balance, zero skipped/terminal updates, no uncontained clip, and
bounded memory. Step 640 has raw force norm `4.697`; semantic rows reduce the
interaction to `1.444`, leaving `0.6927` complete shared-stage retention. The
fixed step-512 rejection is therefore unresolved: neither the favourable
520--576 window nor the adverse 584--640 window establishes generalization.
Continue the unchanged trajectory to fixed selector 1024 and do not promote,
scale, or retune from these heterogeneous training samples.

Exact verification commands:

```bash
PYTHONPATH=. conda run -n orpheus pytest -q \
  tests/unit/test_training_schedule.py tests/unit/test_config.py \
  tests/unit/test_audit_training_dynamics.py tests/unit/test_train_entrypoint.py \
  tests/integration/test_checkpoint_roundtrip.py
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -m "not device"
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q -m device
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q \
  tests/integration/test_rgb_measurements.py::test_roi_sampling_mps_training_mode_no_grad_uses_inference_path \
  tests/integration/test_rgb_measurements.py::test_roi_sampling_mps_training_native_bilinear_path_is_differentiable \
  tests/integration/test_rgb_measurements.py::test_global_rgb_cpu_detector_trains_and_roundtrips_with_mps_backbone \
  tests/unit/test_association.py::test_association_transfers_cost_to_cpu_without_mps_float64 \
  tests/unit/test_evaluation_parameter_update_metrics.py::test_directional_parameter_metrics_transfer_before_float64_accumulation
PYTHONPATH=. conda run -n orpheus ruff format --check .
PYTHONPATH=. conda run -n orpheus ruff check .
PYTHONPYCACHEPREFIX=/tmp/orpheus-pycache PYTHONPATH=. \
  conda run -n orpheus python -m compileall train.py world_model scripts tests
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/attention_pilot_mps.yaml --device mps --dry-run
git diff --check
```

## 2026-08-11 — impulse-gradient path repaired; scaling remains gated

The trainer and convergence supervisor for
`runs/20260811-012103-attention-output-isolated-stage-a/` were intentionally
stopped immediately after logged update 200. The last durable source remains
the independently audited step-128 checkpoint (file SHA-256
`954ee4990e2f7b6e575bfae24057fca0d0f17ae0cdaf1cc4d3467c87806c1700`);
the step-200 update is not reusable and the campaign cannot count toward
convergence. Its launch services are unloaded. The two stderr lines are the
expected multiprocessing semaphore warning from forced shutdown, not a
training exception.

On seeds `15928,9665,8986,13355,2028,5437,12662,4399`, frames 4--8,
the update was finite and supported across all 13 objective terms but had raw
total/interaction norm `857.1579`. Impulse multiplier/additive decoder rows
were `830.3828/210.3096`; the largest shared projection/block norm was
`6.24006`; post-row interaction norm was still `856.8679`, so the complete
interaction stage retained only `0.001167`. The old offline auditor emitted a
severe warning but returned `pass`, which was also inadequate monitoring.

Specification 1.30 adds separately configured joint impulse output and
decoder-row caps without changing forward values, tensor shapes, parameter
count, or inference. It also adds an optional protocol-bound
`minimum_interaction_gradient_retention`; the active config uses `0.1` and
rejects a starved causal update before `optimizer.step()`. The offline auditor
now reports any sub-10% complete-stage retention after local isolation as a
hard failure while retaining successfully contained local typed clips as
warnings. Legacy checkpoints normalize all new controls to `null`.

The non-promotable replay at
`runs/20260811-033712-step200-impulse-gradient-replay-v1/` resumes the exact
step-128 model/Adam/RNG/sampler state and reaches the same update-200 seeds,
frames, support, identity (`1/59`), and coverage90 (`0.86222`) on host MPS RGB
plus CPU closed loop. Earlier bounded impulse updates change the learned
trajectory, so this is not a forward-exact one-update ablation. At step 200,
raw norm is `7.44100`, post-row interaction norm is `1.54550`, complete-stage
retention is `0.64704`, impulse-row norm is `0.14373`, and maximum shared norm
is `0.05334`. One-second RMSE is `0.437779 m` versus `0.441224 m` in the failed
trajectory. The offline replay audit passes nine logged blocks with no severe
or uncontained clipping and maximum RSS `2,929,733,632` bytes. The report is
`impulse_gradient_replay_report.json` inside that run. The replay was
intentionally interrupted after step 200 and is not a checkpoint, selector,
accuracy promotion, or convergence result.

Verification after the repair: focused config/trainer/auditor/checkpoint tests
`281 passed in 20.27 s`; complete non-device suite `697 passed, 5 skipped,
1 deselected in 174.47 s`; host device marker `1 passed, 702 deselected in
2.98 s`; five direct MPS regressions `5 passed in 7.38 s`; Ruff format/check,
compileall, dry run, and `git diff --check` pass. The dry run still resolves
the unchanged 3.00M rung, 8,192 updates, 65,536 balanced episode draws, eight
scenarios, and 32 RGB-only validation episodes. No larger model is justified
until a new immutable small-rung campaign completes fixed selectors and the
declared plateau.

The repair was committed and pushed to `main` as `d38cc9b`. A fresh
weights-only campaign was launched at
`runs/20260811-042704-attention-impulse-isolated-stage-a/` under one-shot
Standard/default LaunchAgent
`com.polceanum.orpheus.attention-impulse-isolated-20260811-042704`, trainer PID
`26459` at launch audit. Metadata records clean commit
`d38cc9bf049e84d868c098217684dbd698897733`, runtime-source fingerprint
`9b1ed8e51c5e0c5c4b877356011c1303a4f864e4b970565aac53ad70f6786eda`,
PyTorch 2.10.0, host MPS RGB measurement, CPU closed loop, float32, RGB-only
runtime, and no oracle. An immutable detached-copy supervisor is active as
`com.polceanum.orpheus.attention-impulse-convergence-20260811-042704`, PID
`26586` at attachment, with the unchanged 8,192 minimum, 4,096 extensions,
four-selector/1% plateau rule, and 24,576 hard limit. Both stderr files are
empty and neither service has restarted.

The mandatory 32-episode initial selector completed in `964.832 s` and exactly
reproduces the prior graph control. All 225 tensors are bitwise equal with
model hash `1354bdfca1cef965c0cd907ea8c157c0fd82169e64f24da656eb42dd1a96df91`;
all 2,583 comparable non-protocol selector fields are exact. Only the expected
protocol hash changes from `21daf4a8...d7f` to
`b98691adcf0d242568a7f46710f9b4a6f3f93dd1a9583a3a8e971f40f3ca3701`.
Score remains `0.3213162196`; current position/velocity RMSE are
`0.251460 m / 1.093191 m/s`; 0.10/0.25/0.50/0.75/1.00-second position RMSE is
`0.265184/0.277452/0.309911/0.335387/0.357837 m`; collision F1 is `0.195489`,
trusted identity-switch rate is `1.3592%`, and position coverage90 is
`93.3861%`. Checkpoint metadata records specification 1.30 and float32. The
equality report is
`runs/20260811-042704-attention-impulse-isolated-stage-a/checkpoint_step_000000_equality_audit.json`.
Training then became active; there was still no trained selector, accuracy gain,
generalization result, plateau, convergence, or scale promotion.

The first sampled training block at update 8 is healthy. Its batch contains
all eight scenario families, eight supported objective terms, 349 causal
trajectory targets, no skipped/no-gradient draw, and a finite applied update.
Raw and applied whole-interaction gradient norms are both `0.673975`; all
typed-output, decoder-row, complete-interaction, and global clip coefficients
are `1.0`. The largest semantic row is ordinary normal force at `0.652459`;
impulse multiplier/additive norms are `3.22e-10/1.82e-11`. RSS is
`2,935,676,928` bytes. Both launchd services were then on their first invocation
with empty stderr. This was only early optimizer-health evidence; the later
attempted update-60 failure and specification-1.31 repair above supersede its
active-state description.

Exact commands used for this repair boundary:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python \
  /private/tmp/replay_orpheus_step200_impulse_clips.py
PYTHONPATH=. conda run -n orpheus python scripts/audit_training_dynamics.py \
  --run runs/20260811-033712-step200-impulse-gradient-replay-v1 \
  --after-step 128
PYTHONPATH=. conda run -n orpheus pytest -q \
  tests/unit/test_training_schedule.py tests/unit/test_config.py \
  tests/unit/test_audit_training_dynamics.py \
  tests/integration/test_checkpoint_roundtrip.py
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -m "not device"
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q -m device
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q \
  tests/integration/test_rgb_measurements.py::test_roi_sampling_mps_training_mode_no_grad_uses_inference_path \
  tests/integration/test_rgb_measurements.py::test_roi_sampling_mps_training_native_bilinear_path_is_differentiable \
  tests/integration/test_rgb_measurements.py::test_global_rgb_cpu_detector_trains_and_roundtrips_with_mps_backbone \
  tests/unit/test_association.py::test_association_transfers_cost_to_cpu_without_mps_float64 \
  tests/unit/test_evaluation_parameter_update_metrics.py::test_directional_parameter_metrics_transfer_before_float64_accumulation
PYTHONPATH=. conda run -n orpheus python train.py \
  --config configs/attention_pilot_mps.yaml --device cpu --dry-run
PYTHONPATH=. conda run -n orpheus ruff format --check .
PYTHONPATH=. conda run -n orpheus ruff check .
PYTHONPYCACHEPREFIX=/tmp/orpheus-pycache PYTHONPATH=. \
  conda run -n orpheus python -m compileall \
  world_model scripts train.py evaluate.py demo.py
git diff --check
```

## 2026-08-11 — shared-gradient failure repaired before capacity scaling

The force-row-isolated trainer and convergence supervisor were intentionally
stopped after update 280 exposed a real remaining optimizer defect. On seeds
`15200,2273,6754,8851,11284,4181,6726,3095`, frames 7--11, the update was
finite and supported but produced raw total/interaction norm `995.5391`, raw
force-decoder norm `989.7965`, and post-row interaction norm `106.7798`.
The effective total coefficient was `0.0010045`; shared projections and blocks
already carried order-one-to-ten gradients before parameter-row clipping ran.
The last durable source remains step 256 with model hash
`79b1819d97f1f30ecdf18ce12977bf64b627eedaca6487f98397a6824c02c922`.
Neither the step-280 update nor the stopped campaign counts toward convergence.

Specification 1.28 and the implementation now add separately configured
per-invocation backward caps on raw typed node, collision, and joint
normal/tangent-force outputs. These hooks run before gradients enter the
decoder or shared attention stack and change no forward values or checkpoint
tensors. The existing parameter-row caps remain as a second layer for
accumulation across recursive invocations. Config validation, legacy checkpoint
normalization, selector/resume protocol hashing, named telemetry, offline audit
warnings, and upstream-gradient regression tests cover the new hierarchy.
`configs/attention_pilot_mps.yaml` sets all three output caps to `0.1`.

An explicitly non-promotable diagnostic branch resumed the durable step-256
model/optimizer/RNG/sampler state and replayed updates 257--280 with the new
backward conditioning. It used PyTorch 2.10.0, host MPS for RGB measurement,
CPU for closed-loop state/dynamics, float32, RGB-only input, and no oracle.
Step 264 preserved every non-gradient forward metric exactly. Step 272 remained
healthy and unclipped at the parameter hierarchy (`0.3177` total gradient,
maximum shared-block norm `0.00315`, zero trusted identity switches, 1-second
RMSE `0.28137` versus source `0.28163`).

The decisive step 280 used the exact same seeds and frames. The later raw
parameter norm fell `995.5391 -> 10.8330`, force-decoder norm fell
`989.7965 -> 10.7843`, post-row interaction norm fell
`106.7798 -> 1.43288`, and the largest shared projection/block parameter norm
was `0.08506` (combined shared L2 `0.25750`). The post-row interaction stage
retained `0.69790` and the global stage retained `1.0`; the supported update was
finite and applied. Trusted identity switches (`1`) and position coverage90
(`0.88`) match the known batch. The offline dynamics audit reports `pass`, no
hard failures, and a truthful severe warning for localized node/force output
and force-row coefficients. This is repair qualification, not fixed-selector,
accuracy, generalization, plateau, or convergence evidence.

The diagnostic service was booted out immediately after step 280. Its artifacts
are under `runs/20260811-004400-step280-output-gradient-replay-v1/`, including
`typed_output_gradient_replay_report.json`; stdout/stderr are
`/private/tmp/step280-output-gradient-replay-v1.stdout.log` and `.stderr.log`
(stderr remained empty). The affected unit/checkpoint suite reports
`297 passed in 52.10 s`; the complete non-device suite reports
`678 passed, 5 skipped, 1 deselected in 171.77 s`; the host device marker
reports `1 passed in 2.17 s`; and the five hardware-conditional MPS tests
report `5 passed in 8.01 s`. Ruff format/check, compileall, dry run, and
`git diff --check` pass. The CPU dry run resolves 8,192 updates, 65,536 balanced
episode draws, four nominal passes, eight scenarios, and 32 validation
episodes; sandbox MPS availability is false while direct host tests prove it is
available. Immutable commit/push and fresh weights-only sustained launch remain
pending at this status boundary.

The primary-source scaling review does not justify a capacity jump while this
fresh small-rung qualification is absent. The original Transformer supports
scaled multi-head attention, residual normalization, and controlled
width/depth/head ablations—not unqualified parameter growth. Compute-optimal
work requires data to scale with parameters; maximal-update parameterization is
a candidate for transferring tuned hyperparameters across future width rungs;
and current V-JEPA evidence supports staged latent video pretraining followed by
an action-conditioned predictor. Conversely, physical-law benchmarks show that
visually strong video generation can still fail OOD physics. The next order is
therefore: fresh 3.00M control through step 280/selector/plateau; matched
data-only scaling; width; depth; bounded timestamped history; then single-CUDA
GPU. Dense JEPA-style RGB pretraining may later be distilled into explicit
`WorldBelief`, not replace it.

The repair and synchronized specification/project memory were committed and
pushed to `main` as `9d0502b`. A fresh weights-only campaign is active at
`runs/20260811-012103-attention-output-isolated-stage-a/` under one-shot
Standard/default LaunchAgent
`com.polceanum.orpheus.attention-output-isolated-20260811-012103`,
`KeepAlive=false`, with trainer PID `14294` at launch audit. Metadata records
clean immutable commit `9d0502b4153a80e5f37d93a6142f9ffd3a0b3359`, runtime
source fingerprint
`82647f40748936058b3ec33201d63cc5026ea23935be7e7d7f3d644fa15f4232`,
PyTorch 2.10.0, MPS available/built, MPS RGB measurement, CPU closed loop,
float32, RGB-only runtime, no oracle, and the protected protocol-14 graph
checkpoint initializer. The resolved selector protocol hash is
`21daf4a8b1349429f3f631282bbc69fce202ffbba028a3a9cee06ca06b311d7f`.

An exact-source convergence supervisor is attached from a clean detached copy
of the same commit under one-shot LaunchAgent
`com.polceanum.orpheus.attention-output-convergence-20260811-012103`,
supervisor PID `14684` at attachment. Its independently computed commit,
runtime-source, and worktree fingerprints exactly match the trainer. It records
the unchanged 8,192-update minimum, complete 4,096-update extensions,
four-selector/1% plateau rule, and truthful 24,576 hard limit. Trainer and
supervisor stderr are empty. The mandatory initial 32-episode selector is
complete and training is active. No trained selector, accuracy gain,
generalization result, plateau, convergence, or scale promotion exists yet.

The initial selector completed all 32 episodes in `969.521 s`. All 225 model
tensors are bitwise identical to the preceding protected step-zero checkpoint,
both hashes are
`1354bdfca1cef965c0cd907ea8c157c0fd82169e64f24da656eb42dd1a96df91`,
and all 2,583 comparable non-protocol metrics are exact. Score is
`0.3213162196`; current position/velocity RMSE are `0.251460 m / 1.093191
m/s`; 0.10/0.25/0.50/0.75/1.00-second position RMSE is
`0.265184/0.277452/0.309911/0.335387/0.357837 m`; collision F1 is `0.195489`,
trusted identity-switch rate is `1.3592%`, and position coverage90 is
`93.3861%`. The equality artifact is
`runs/20260811-012103-attention-output-isolated-stage-a/checkpoint_step_000000_equality_audit.json`.

The fresh campaign passes the identical historical step-64 schedule position.
Relative to the force-row-only predecessor on the same seeds, frames, and 154
trajectory targets, raw whole/interaction gradient falls `21.5377 -> 2.14592`,
the joint force parameter-row norm falls `21.4665 -> 1.75123`, relation-decoder
weight norm falls `21.4054 -> 2.01100`, and the maximum non-decoder shared-stack
parameter norm falls `0.04242 -> 0.00540`. The post-row interaction stage
retains `0.62863`, versus `0.49616`, while 1-second sampled RMSE is fractionally
better (`0.377141` versus `0.377330 m`). All 64 updates are applied with exact
scenario balance, zero skipped draws or hard audit failures, frozen perception,
zero trusted switches on the stress batch, stable `2.883 GB` maximum RSS, and
empty stderr. One severe step-8 typed-output coefficient remains truthfully
reported; later sampled blocks do not show systematic starvation. Step 64 is
not a durable checkpoint and is optimizer evidence only. The first durable
trained checkpoint is step 128 and the first trained complete selector is step
512.

Durable step 128 passes a new reusable checkpoint-isolation audit. The exact
numbered checkpoint is preserved at
`runs/20260811-012103-attention-output-isolated-stage-a/checkpoints/step_000128.pt`
(file SHA-256 `954ee4990e2f7b6e575bfae24057fca0d0f17ae0cdaf1cc4d3467c87806c1700`,
model hash `07bfffec5d0c51c9b7c753d340fcbc5852c4efb8d4513e493efc6b0b373a77bd`).
All 177 inherited tensors remain bitwise exact; all 48 attention tensors have
changed; all 48 and only those attention parameters own Adam state at step 128;
every serialized numeric value is finite; configured names/shapes/dtypes
match; and both protected selector checkpoints retain model hash
`1354bdfca1cef965c0cd907ea8c157c0fd82169e64f24da656eb42dd1a96df91`.
The report is `checkpoint_step_000128_audit.json` beside the run.

The complete dynamics audit through step 128 passes 128 applied updates, exact
16-way exposure for every scenario, zero skipped draws or hard failures,
frozen perception, and stable `2.886 GB` maximum RSS. Only the already known
step-8 typed-output warning is severe. Median effective node/force coefficients
are `1.0` and collision is `0.815`; the repair is not a universal low-gradient
gate. Sampled identity remains a real warning: the output-isolated run has
`8/694 = 1.153%` trusted switches versus `6/697 = 0.861%` in the paired
force-row predecessor, with the difference concentrated at step 128. Aggregate
sampled 0.25/0.50/0.75/1.00-second RMSE improves, 0.10-second RMSE is
`0.00017 m` worse, and coverage is slightly higher. These sparse heterogeneous
batches neither promote nor reject the model; the complete step-512 selector
must resolve identity and every axis/horizon/scenario guardrail.

`scripts/audit_attention_checkpoint.py` now makes the previously ad hoc
architecture-growth audit repeatable. It recomputes model and whole-file
hashes, checks recursive finiteness, verifies configured tensor shapes/dtypes,
compares inherited and protected tensors exactly, maps serialized optimizer
IDs back to named parameters, and validates Adam steps. Its unit/integration
tests pass (`3 passed`; focused checkpoint/auditor group `39 passed`). The
complete non-device suite passes `681 passed, 5 skipped, 1 deselected in
215.22 s`; hardware-only tests were not rerun at this boundary. Trainer and
supervisor remain live with empty stderr.

Review of the original Transformer and current primary scaling/video-model
work is now encoded in specification 1.29. The current implementation already
has the applicable small-token ingredients: dense scaled dot-product attention,
pre-RMSNorm, SwiGLU, typed set tokens, and bounded typed outputs. Flash-style
kernels, GQA/MLA, MoE, sharding, and language-style positional order do not
address a measured bottleneck at 22 tokens. The fixed next ladder is the
completed control/data learning curve, then one-axis depth and width controls
with parameter-proportional balanced draws, then timestamped bounded history,
then an 8.31M single-CUDA candidate. A later JEPA-style dense RGB pretraining
stage must distill or cross-attend into explicit belief proposals and pass the
same state/horizon/identity/event/calibration/OOD gates. No larger candidate is
launched before selector and plateau evidence qualify the current rung.

## 2026-08-10 — exact force-head localization and scale/no-scale decision

The collision-isolated step-256 checkpoint was replayed from clean detached
commit `70c2e3b3dfd590a470077ceca7c224977152945a` with its exact optimizer,
CPU/MPS RNG, and deterministic sampler state. External instrumentation wrapper
SHA-256 `187fa606488e6aa9f4fab4c05678c69c9540affa1ac7452294d81263cbea1683`
collected raw named-parameter and semantic decoder-row norms before calling the
unaltered committed clip function. It did not change forward values, losses,
gradients, optimizer state, RNG, data order, or checkpoint semantics. All 362
shared deterministic fields at step 264 and all 308 shared fields at step 272
match the original with zero differences; step 280 exactly reproduces loss
`2.7366230488`, raw interaction norm `17.7049923`, and retained coefficient
`0.05648124`.

The failure is now localized. At step 280 the relation-decoder weight norm is
`17.6189251`; semantic normal/tangent force rows are `17.3893547/3.2159121`
(joint `17.6842231`). The collision row is ordinary at `0.2355342` and the
remaining interaction gradient is approximately `0.8573238`. A joint unit
force-row cap would leave approximately `1.3171956` before the existing unit
interaction cap, retaining about `0.7591881` rather than `0.0564812`. This is
specific evidence for force-row optimizer isolation, not evidence for lower
global learning rate or more model capacity. The replay service is stopped;
its stale `training_state.json` says running only because launchd bootout was
an external stop. No update-280 checkpoint exists.

Specification 1.27 and the implementation now add an optional joint
normal/tangent force-row cap before the interaction/global hierarchy. The cap
changes no forward physics, targets, or parameters. Telemetry retains raw
semantic rows plus raw/applied group/stage/global norms; the offline auditor
includes the new coefficient; the resolved validation protocol binds it.
`configs/attention_pilot_mps.yaml` sets both collision-row and force-row caps to
`1.0`. Checkpoint metadata now correctly reports specification 1.27 rather than
the stale 1.25 constant discovered during this audit.

Focused verification completed:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q \
  -p no:cacheprovider tests/unit/test_training_schedule.py \
  tests/unit/test_config.py tests/unit/test_audit_training_dynamics.py
```

Result: `223 passed in 14.90s`. Final verification also completed:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q \
  -p no:cacheprovider -m 'not device'
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q \
  -p no:cacheprovider -m device
PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q \
  -p no:cacheprovider <five exact hardware-conditional MPS node IDs>
PYTHONPATH=. conda run --no-capture-output -n orpheus ruff format --check .
PYTHONPATH=. conda run --no-capture-output -n orpheus ruff check .
PYTHONPYCACHEPREFIX=/private/tmp/orpheus-force-clip-pycache PYTHONPATH=. \
  conda run --no-capture-output -n orpheus python -m compileall -q \
  world_model train.py evaluate.py demo.py scripts tests
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/attention_pilot_mps.yaml --dry-run --device cpu
git diff --check
```

Results: non-device suite `664 passed, 5 skipped, 1 deselected in 170.03s`;
host device marker `1 passed, 669 deselected in 3.07s`; five direct host-MPS
tests `5 passed in 7.66s`; Ruff format/check, compileall, dry run, and diff
check pass. The dry run resolves 8,192 balanced updates, 65,536 episode draws,
eight scenarios, 32 validation episodes, Python 3.10 environment, PyTorch
2.10.0, and MPS built. The sandboxed process reports MPS unavailable; direct
hardware tests prove host MPS availability. A first direct-MPS command used
three stale historical node names and collected no tests; the corrected exact
command above is the reported result. A fresh training smoke and sustained
launch remain pending at this status update. The evidence artifact is
`runs/20260810-180502-attention-collision-isolated-stage-a/gradient_localization_report.json`;
the diagnostic replay is under
`/private/tmp/orpheus-replay-70c2e3b-20260810/runs/20260810-211022-gradient-localization-replay/`.

Review of the original Transformer, compute-optimal scaling, current efficient
attention/dense-MoE practice, and recent physical-law evaluation supports the
existing staged decision. The 3,004,656-parameter pilot already has four
pre-RMSNorm width-128/four-head blocks and SwiGLU over at most 22 typed tokens.
RoPE remains reserved for true timestamped history rather than arbitrary set
order. FlashAttention/GQA/MoE solve unmeasured long-context or routing costs at
this rung and are deferred. First qualify the repaired dense model through the
former failure boundaries, repeated complete selectors, and the declared
plateau. Then compare data-only, depth, width, and bounded-history rungs one at
a time, increasing continuously varied balanced data with parameters and
requiring fixed disjoint RGB-only validation/test/OOD non-regression before a
single-CUDA-GPU scale-up.

The verified repair was committed and pushed to `main` as `b3b69c1`. A fresh
weights-only campaign is active at
`runs/20260810-213857-attention-force-isolated-stage-a/` under one-shot
Standard/default LaunchAgent
`com.polceanum.orpheus.attention-force-isolated-20260810-213857`,
`KeepAlive=false`, with trainer PID `2209` at launch audit. Metadata records the
clean immutable commit and source fingerprint, PyTorch 2.10.0, MPS available,
MPS RGB measurement, CPU closed loop, float32, RGB-only runtime, no debug
oracle, and the protected protocol-14 graph checkpoint initializer. The
resolved protocol hash is
`6612f9107c4817436ddb71f6dac53f3a1754dcea25533629468c819d52adbc6f`.
The mandatory initial 32-episode selector is in progress with durable
per-episode heartbeats and zero stderr; no optimizer update, trained selector,
accuracy improvement, or convergence claim exists yet.

The initial selector completed all 32 episodes in `959.695 s`. Exact tensor
comparison against the preceding collision-isolated step-zero checkpoint finds
zero changed tensors and the same model hash
`1354bdfca1cef965c0cd907ea8c157c0fd82169e64f24da656eb42dd1a96df91`.
Excluding the intentionally new optimization-protocol hash and timing fields,
all 2,583 comparable selector fields match exactly, including every pooled/
per-scenario axis and horizon, lifecycle/identity, event, support, and
calibration metric. Score is `0.3213162196`; current position/velocity RMSE are
`0.251460 m / 1.093191 m/s`; 0.10/0.25/0.50/0.75/1.00-second RMSE is
`0.265184/0.277452/0.309911/0.335387/0.357837 m`; collision F1 is `0.195489`,
distance-gated ID-switch rate is `1.3592%`, and position coverage90 is
`93.3861%`. This is the exact protected equality control, not a trained gain.

A clean exact-source convergence supervisor is attached from detached commit
`b3b69c1` under one-shot LaunchAgent
`com.polceanum.orpheus.attention-force-convergence-20260810-213857`, supervisor
PID `2662` at attachment. Its runtime fingerprint exactly matches the trainer:
`199b33bbb6c43dd82a02dcd7c5299d4bd410f6abc999a1f3cc0c8ebb9bc73257`.
It durably records the 8,192-step minimum, complete 4,096-step extensions,
four-selector/1% plateau rule, and truthful 24,576 hard limit; reaching the
limit without plateau is `limit_hit`, not convergence. Trainer and supervisor
stderr remain empty.

The first eight balanced optimizer updates also pass the live dynamics audit.
One draw from every scenario is applied, trajectory support is `349`, skipped
draws are zero, perception gradient is exactly zero, raw interaction/whole
gradient is `0.668898` with coefficient `1.0`, joint force-row norm is
`0.646357` with coefficient `1.0`, collision-row norm is effectively zero,
distance-gated identity switches are zero, and maximum RSS is
`2,874,376,192` bytes. This is early optimizer health only; steps 152/280 and
the first trained selector at 512 remain required.

Sampled updates 16 through 72 remain finite, supported, and scope-clean. At step
24 the collision row produces a raw norm of `4.45588`; its configured local
cap reduces it to `1.0`, after which the complete interaction cap reduces the
remaining `2.36835` norm to `1.0`. The true raw whole-model norm is `4.94611`,
the final total coefficient is `0.202179`, the optimizer update is applied,
perception gradient remains exactly zero, RSS is `2,891,427,840` bytes, and
the offline dynamics auditor reports no severe clip or hard failure. Step 32
similarly reduces a raw collision norm of `3.23987` to `1.0`; the subsequent
interaction coefficient is a mild `0.929705`, all 32 balanced updates are
applied, every scenario has four sampled blocks, trusted identity switches are
zero in the block, and RSS stays flat. This is evidence that the hierarchy
contains collision-row outliers, not selector evidence of an accuracy gain.

Step 64 exposes one severe but correctly isolated force-row event. Its raw
interaction norm is `21.5377`, of which `21.4665` is confined to the joint
normal/tangent decoder rows (`7.5142/20.1083`). The configured force-row cap
reduces that group to `1.0`; the remaining interaction norm is `2.01547`, so
unrelated attention gradients retain a `0.496162` stage coefficient instead
of the raw-total `0.0464303` coefficient. Collision, node, support, identity,
finite-state, frozen-perception, and memory diagnostics remain ordinary. Step
72 immediately returns to force coefficient `0.976879`, interaction-stage
coefficient `0.686862`, zero sampled identity switches, positive future
correction, and unchanged RSS. The auditor truthfully retains step 64 as a
severe row-clip warning while reporting no hard failure. This is one contained
typed-row outlier, not yet systematic recurrence or selector evidence; the
run continues unchanged toward checkpoint 128 and boundaries 152/280.

The first durable trained checkpoint at step 128 passes an independent
exact-source audit. All 177 inherited tensors remain bitwise equal to step
zero; all 48 attention tensors change; all 48 optimizer states belong only to
attention parameters and report Adam step 128; model, optimizer, and scheduler
state are finite; and checkpoint hash
`19cc53de2ac9cbbf88be38aeb94b689e373b6aa989fbd63c6c6674e2899c8010`
matches its metadata. Best/reference model states remain exactly equal to the
protected control hash `1354bdfc...df91`; protocol and validation-manifest
hashes remain `6612f910...bc6f` and `e27bdf2d...46be`. Step 128 itself has
force coefficient `1.0`, collision coefficient `0.373758`, interaction-stage
coefficient `0.851525`, support 486, positive future correction, frozen
perception, and stable `2,892,918,784`-byte maximum RSS. Its two sampled
identity switches reproduce exactly in the matched preceding control; sampled
aggregate identity rate through step 128 is `6/697 = 0.8608%`. The durable
audit artifact is
`runs/20260810-213857-attention-force-isolated-stage-a/checkpoint_step_000128_audit.json`.
This proves scope and continuation integrity, not accuracy promotion; former
boundaries 152/280 and fixed selector 512 remain required.

The repaired run passes the exact historical step-152 boundary. On the same
seed manifest (`9664,8881,1058,2123,7628,3477,2326,12831`) and frames 7--11,
the normalized pre-isolation campaign had raw interaction norm `28.1387` and
retained `0.03554`; collision-row isolation improved this to `7.11114` and
`0.14308`. The force-isolated campaign now has raw norm `2.46615`: force rows
are ordinary at `0.25152` and unclipped, collision is `1.70491` with row
coefficient `0.58654`, and the complete interaction-stage coefficient is
`0.48940`. Support is 343 across all 13 objective terms, the update is applied,
sampled identity switches are zero, future correction is positive `0.02992 m`,
coverage90 is `92.16%`, perception gradient is zero, and maximum RSS remains
bounded at `2,897,362,944` bytes. The complete dynamics auditor passes with
152 exactly balanced applied updates, zero skips/hard failures, and only the
already recorded isolated step-64 force-row warning. This boundary repair is
optimizer-health evidence, not fixed-selector accuracy evidence; step 280 and
selector 512 remain required.

An architecture/scaling review against the original Transformer, modern dense
LLM practice, compute-optimal scaling, set attention, Perceiver-style latent
bottlenecks, and recent video-world-model evidence does not justify changing
the active pilot. It already uses pre-RMSNorm, scaled dense multi-head
attention, SwiGLU, typed permutation-equivariant set tokens, and bounded typed
decoders. FlashAttention, GQA/MLA, and MoE address unmeasured long-context,
cache, or large-model bottlenecks at the current maximum of 22 tokens. Exact
local capacity census for proposed later rungs is: current/data-only
`3,004,656` total (`1,103,626` attention), depth-6 `3,530,480`
(`1,629,450` attention), width-192/four-block `4,342,896` (`2,441,866`
attention), and future single-GPU width-256/depth-6 `8,305,648`
(`6,404,618` attention). None is launched. Promotion order remains data-only,
depth, width, then bounded timestamped history, with parameter-scaled data and
fixed disjoint RGB-only validation/test/OOD gates.

## 2026-08-10 — typed-attention scene context and input conditioning repaired

Primary-source review retained the useful Transformer mechanism—parallel
content-dependent interaction—while rejecting language-token assumptions that
conflict with Orpheus. The new optional dynamics module uses four
RMS-pre-normalized scaled-dot-product attention blocks, width 128, four heads,
and SwiGLU width 512 over derived scene, entity, and candidate-relation tokens.
There are no object-slot positional embeddings and no RoPE on unordered padded
slots. Typed decoders propose bounded node acceleration, antisymmetric pair
force, event-logit/jump, and process-noise residuals through the existing
analytic/event/uncertainty contracts. `WorldBelief` remains authoritative.

The corrected module adds `1,103,626` parameters to the
`1,901,030`-parameter accepted runtime for `3,004,656` total. Its 55-wide
scene input is derived from authoritative belief fields rather than only the
reserved global code. All output heads initialize at exact zero. Unit
tests prove exact graph equality at initialization and permutation equivariance
after nonzero decoding. Weight-only growth rejects any unexpected key and
allows missing keys only below `dynamics.attention_interactions.*`. The first
training stage exposes only those new parameters; attention also shares the
interaction-local gradient clip and diagnostics.

The original host smoke is preserved at
`runs/20260810-111959-attention-pilot-smoke/`. Exact command:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/attention_pilot_mps.yaml \
  --initialize-from runs/20260810-042627-protocol20-y-only-recovery/checkpoints/validation_step_000064.pt \
  --run-name attention-pilot-smoke \
  --set training.steps=1 \
  --set training.checkpoint_every=1 \
  --set training.eval_every=1 \
  --set training.validation_episodes=16 \
  --set evaluation.episodes=16
```

It completed one optimizer update in `626.51 s` with RGB measurement on MPS,
closed loop on CPU, float32, no oracle input, loss `3.7291617`, raw/applied
gradient norm `1.8602252/1.8602252`, no skipped update, and maximum RSS
`2,004,131,840` bytes. It drew one example from each of the eight scenarios.
The reduced smoke manifest intentionally cannot satisfy the full selector's
per-scenario support contract, so `last.pt` is correctly labelled
`last_unvalidated`; it is wiring evidence, not an accuracy candidate.

Exact checkpoint audit finds 177 inherited keys and zero changed inherited
tensors. All 48 new attention tensors receive optimizer state, but only
`node_decoder.weight/bias` and `relation_decoder.weight/bias` have nonzero
first moments after update one, the expected gradient flow from zero output
heads. The smoke exposed one observability omission: the local interaction
gradient group originally named only the legacy graph. The attention module is
now included and a focused regression test verifies its local clip/metric.

`configs/attention_pilot_mps.yaml` declares 8,192 balanced attention-only
updates, 65,536 episode draws (four nominal passes over 16,384 continuously
varied episodes), checkpoints every 128 updates, and a full 32-episode selector
every 512 updates. Measured representative CPU prediction cost is
`0.0363 -> 0.0603 s` per 50 ms step; fixed validation costs roughly 16--19
seconds per episode on this Mac. Remaining work is the full convergence run,
stage-B timestamped history, a parameter-matched graph/MLP control, and
disjoint test/OOD qualification. No accuracy improvement is claimed from the
smoke.

The first sustained campaign is preserved at
`runs/20260810-114053-attention-pilot-stage-a/`, launched from commit
`a84ef20` with label
`com.polceanum.orpheus.attention-20260810-114053`. It ran as one
Standard/default, `KeepAlive=false` job under `caffeinate`; metadata records a
clean source fingerprint, PyTorch `2.10.0`, MPS measurement, CPU closed loop,
float32, and RGB-only/no-oracle runtime. Initial fixed validation completed all
32 episodes in `977.689 s` under protocol hash
`6064c5b1a055e943a3f3900ed63596b6402c7d7ad5a4d45f7b2d77351bc8c648`.
The exact zero-output initialization reproduces and protects the protocol-14
graph incumbent at score `0.3213162196`; this is an equality control, not an
attention accuracy claim.

The first logged balanced block reached update 8 with one draw from each
scenario, loss `0.5049295`, 349 supported trajectory targets, all eight
objective rows supported, and all optimizer updates applied. Its attention/
global raw norm `3.6997645` is correctly bounded to `0.9999997` by the local
interaction cap; the frozen perception gradient is exactly zero, skipped
draws and stderr bytes are zero, and peak recorded RSS is `2,837,905,408`
bytes. This is healthy early optimization evidence only. No trained selector
or generalization promotion exists yet. Logs are
`/private/tmp/20260810-114053-attention-pilot-stage-a.stdout.log` and
`/private/tmp/20260810-114053-attention-pilot-stage-a.stderr.log`.

At update 32 the run remains finite and supported, with zero stderr and a
nearly flat `2,867,974,144`-byte peak RSS. Sparse update-8/16/24/32 losses
`0.5049/3.4646/5.4748/3.0221` are not a monotonic failure signal: event loss
is absent in the first sampled window and contributes `2.2928/3.5294/3.5814`
in the next three, whose ground/pair-contact loads differ substantially. The
offline dynamics auditor previously mislabeled four cadence samples as four
applied updates. It now reports the authoritative completed trainer step
(`32`), four logged confirmations, explicit `[8, 8, 8]` sampling gaps, and a
warning that loss/gradient distributions are sparse. Hard finite-state,
support, and clipping checks still execute on every update.

The pilot reached a durable `checkpoints/last.pt` at update/data draw
`128/128`, with zero skipped draws and no stderr. Exact checkpoint audit finds
all 177 inherited tensors bitwise equal to step zero, 48 optimizer states all
owned by attention, finite model/optimizer tensors, and intact best/reference
hashes at protected score `0.3213162196`. Forty-seven attention tensors changed;
only `scene_projection.weight` remained exactly unchanged with a zero Adam
first moment. Repository-wide use audit proved why: `WorldBelief.global_code`
is initialized to zero and no runtime path updates it. The learned scene-token
bias could still aggregate entity/relation tokens, but the declared global and
camera scene input was dead.

The job and PID were stopped after step 128 and are confirmed absent. This run
is finite architecture-diagnostic evidence, not a trained selector, plateau,
or accuracy result, and it must not be resumed. Specification 1.23 replaces
the dead input with a fixed context derived from global code, summarized global
uncertainty, gravity, camera transform/motion/intrinsics, summarized camera
uncertainty, and calibration state. The corrected model has a `128 x 55` scene
projection; a regression proves its weight receives finite nonzero gradient
even while global code is zero. The focused dynamics/scope/checkpoint suite
reports `129 passed` in `18.54 s`. The complete non-device suite reports
`650 passed, 5 skipped, 1 deselected` in `171.69 s`; the host-MPS device marker
reports `1 passed, 655 deselected` in `3.04 s`. `ruff check .`,
`ruff format --check .`, compileall, and `git diff --check` pass. A new clean
host smoke is complete at
`runs/20260810-133010-attention-live-scene-smoke/`.

Exact smoke command:

```bash
PYTHONPATH=. conda run --no-capture-output -n orpheus python train.py \
  --config configs/attention_pilot_mps.yaml \
  --initialize-from runs/20260810-042627-protocol20-y-only-recovery/checkpoints/validation_step_000064.pt \
  --run-name 20260810-133010-attention-live-scene-smoke \
  --device mps \
  --set training.steps=2 \
  --set training.checkpoint_every=2 \
  --set training.eval_every=2 \
  --set training.validation_episodes=16 \
  --set evaluation.episodes=16
```

It completed in `649.0025 s` with MPS measurement, CPU closed loop, float32,
RGB only, no oracle input, two optimizer updates/eight-scenario balanced
batches, 16 episode draws, zero skips, loss `0.9372296`, raw/applied gradient
`0.3623803/0.3623803`, and peak RSS `2,700,517,376` bytes. The reduced
manifest correctly retains `last.pt` as `last_unvalidated`; no accuracy claim
is made.

Exact step-two checkpoint audit finds all 177 inherited tensors bitwise
unchanged and all 48 attention tensors changed. `scene_projection.weight` has
2,432 nonzero entries relative to step zero (`max=3.7078e-5`,
`L2=0.0011339`) and its Adam first moment is nonzero (`L2=2.3550e-6`). Every
optimizer state belongs to attention, model/optimizer tensors are finite,
perception gradient is zero, data draw is exactly two, and source provenance
is clean commit `c9f9dc6`. This closes the dead-scene wiring defect; sustained
accuracy and convergence are still unproven.

The first corrected sustained campaign is preserved at
`runs/20260810-134330-attention-live-scene-stage-a/` under one-shot
Standard/default LaunchAgent
`com.polceanum.orpheus.attention-live-20260810-134330`, `KeepAlive=false` and
`caffeinate`. Its former PID `81275` and job are now stopped; metadata records clean
commit `25d82d8`, runtime fingerprint
`b80851654c0c85ea1c16fb9b80a388221568ffeeb6b9ccfb1cece1c09716bc79`,
PyTorch `2.10.0`, MPS built/available and used for RGB measurement, CPU closed
loop, float32, RGB-only/no-oracle runtime, and the protected protocol-14 graph
checkpoint as its source. The first initialization episode completed in
`24.999 s` under unchanged selector hash
`6064c5b1a055e943a3f3900ed63596b6402c7d7ad5a4d45f7b2d77351bc8c648`.
The complete 32-episode initialization finished in
`966.681 s` and exactly reproduces score/current/velocity
`0.3213162196 / 0.2514599 m / 1.0931909 m/s`, axes
`0.281775/0.201906/0.263691 m`, and horizons
`0.265184/0.277452/0.309911/0.335387/0.357837 m`. All 177 inherited tensors
match the earlier zero-output control exactly and both typed decoders remain
zero at step zero.

The first repaired training block reached update 8 with loss `0.4890857` and
raw/applied interaction gradient `0.2631448/0.2631448`, so neither clip fired;
the superseded dead-scene pilot had `3.6998 -> 1.0` on the identical sampled
draw. It has 349 trajectory targets, all eight objective terms, one example
from every scenario, zero skipped draws and sampled distance-gated identity
switches, zero perception gradient, every horizon supported, peak RSS
`2,810,531,840` bytes, and empty stderr. This is live optimizer-health evidence
only, not a trained selector or accuracy promotion.

The run was intentionally stopped after its sampled update-64 metric exposed
a conditioning failure. The exact update-64 seeds, window, two pair-collision
intervals, four ground-collision objects, one wall collision, and one external
actuation match the old dead-scene control. Most objective terms are close,
including rollout NLL `-0.6872` versus `-0.6829`, but raw interaction gradient
is `45.3456` versus `1.3231`; the local coefficient collapses to `0.02205`.
Earlier sampled live norms were
`0.2631/1.9980/4.7883/5.5396/6.2062/2.2633/6.4135`, so update 64 is the
endpoint of a scale-sensitive path rather than nonfinite loss or one uniquely
difficult physical batch. Global clipping kept the applied update finite, but
continuing would spend most updates at a resolution-dependent effective rate.

Root cause is the 55-value scene vector mixing pixel-scale intrinsics with
latent values, log variances, gravity, camera pose/motion, and calibration
before `scene_projection`; block RMSNorm occurs only after that projection.
Specification 1.24 and the implementation now apply fixed non-affine RMS
normalization immediately before the scene projection. It adds no parameter,
retains exact zero-output graph behavior, and keeps absolute physical fields
unchanged for analytic dynamics. A focused regression scales intrinsics by
`1000x` and verifies finite scene input with vector norm `sqrt(55)` before
projection. The focused dynamics file reports `20 passed`; complete gates and
a normalized host conditioning smoke are the next required steps. Complete
non-device verification reports `651 passed, 5 skipped, 1 deselected` in
`171.34 s`; the host-MPS marker reports `1 passed, 656 deselected` in `3.01 s`.
Ruff check, Ruff format check, compileall, and `git diff --check` pass. Shutdown
emitted only Python's resource-tracker warning for 14 worker semaphores; the
process is confirmed absent. The stopped run has no trained selector or
promotion and cannot count toward convergence.

The normalized sustained campaign is preserved at
`runs/20260810-144901-attention-conditioned-stage-a/` under one-shot
Standard/default LaunchAgent
`com.polceanum.orpheus.attention-conditioned-20260810-144901`,
`KeepAlive=false`; its former trainer PID `84633` is stopped. It starts from clean pushed commit
`de06fcb`; metadata verifies PyTorch `2.10.0`, MPS available and used for RGB
measurement, CPU closed loop, float32, RGB-only/no-oracle execution, and the
unchanged protected graph checkpoint. The fixed protocol hash remains
`6064c5b1a055e943a3f3900ed63596b6402c7d7ad5a4d45f7b2d77351bc8c648`.

The complete 32-episode step-zero selector finished in `1001.259 s`, visited
every scenario four times, emitted no stderr, and exactly reproduced protected
score `0.3213162196`. Update 8 uses the same seeds/window as both prior pilots:
loss is `0.4891017`, raw/applied gradient `0.2534595/0.2534595`, all 349
trajectory targets and eight objective terms are supported, and there are no
skips or sampled distance-gated identity switches. Update 16 remains finite on
the matched harder contact block; its raw norm is `1.3194281` versus `1.9979866`
before normalization and the local clip coefficient improves from `0.50050`
to `0.75790`. Coverage/precision/identity/calibration improve slightly, while
event loss and x/y rollout terms are worse on this single sample. These are
early conditioning results only; update 64, durable checkpoint integrity,
full selectors, broad non-regression, and plateau remain pending.

The matched update-64 conditioning check now passes. Seeds, frame window, two
pair-collision intervals, four ground-collision objects, one wall collision,
one external actuation, support `154/13`, coverage/precision/identity, and
coverage90 are identical to the unnormalized run. Loss is effectively
unchanged (`2.0483899 -> 2.0493994`), but raw gradient falls
`45.3455582 -> 2.2960890` (19.75x) and the retained local coefficient rises
`0.0220529 -> 0.4355230`. Rollout NLL worsens slightly
`-0.6871663 -> -0.6790932`; no accuracy claim is made from this sampled batch.
All eight logged blocks are finite,
supported, applied, and stderr-free; peak RSS is `2,903,666,688` bytes. The
specific projection-scale collapse is resolved. The run continues toward its
first durable step-128 integrity audit and step-512 complete selector.

The durable `checkpoints/last.pt` at update 128 passes exact integrity audit.
All 177 inherited tensors remain bitwise equal to step zero; all 48 attention
tensors changed, including `scene_projection.weight`. All 48 optimizer states
belong only to attention and have Adam step exactly 128. Model and optimizer
contain no nonfinite tensors. Protected `best_rollout.pt`,
`reference_rollout.pt`, and `validation_step_000000.pt` retain the identical
model hash
`1354bdfca1cef965c0cd907ea8c157c0fd82169e64f24da656eb42dd1a96df91`
at step zero; the mutable step-128 hash is
`2edab07779258fb9ed39f116bb77f129b74a25f58674883d8b9980493095b383`.

The offline dynamics audit reports 128 authoritative applied updates, 16
sparse telemetry blocks at exact eight-update cadence, 16 draws from each of
the eight scenarios in those balanced blocks, zero skipped draws, failures,
or severe clips, and trajectory support `154..471` (median `329`). Sampled
loss is `0.4891..5.6757` (median `3.6210`); raw gradient is
`0.2535..6.3168` (median `3.2646`, p95 `6.0123`). Fourteen of 16 sampled blocks
use the ordinary 1.0 local cap, but none retain less than 10% of the raw norm.
Peak recorded RSS remains `2,905,124,864` bytes and stderr remains empty.
Scope, optimizer, support, finite-state, and resource health are proven through
the first checkpoint; trained accuracy remains unproven until update 512.

The durable step-256 checkpoint also passes exact integrity audit. All 177
inherited tensors remain bitwise unchanged, all 48 attention tensors changed,
all 48 optimizer states remain attention-only at Adam step 256, protected
step-zero hashes remain exact, and model/optimizer state is finite. Its mutable
model hash is
`f9a95062e94b2a426a84d711d9cd5c095dbe04988cd5a8ac088b8c5db6b9514d`.
Peak recorded RSS is `2,915,614,720` bytes, only about 10.5 MB above step 128;
there are zero skipped draws, stderr bytes, or audit failures.

One explicit severe-clip warning remains at sampled update 152: an unusually
event-heavy batch with 22 ground-collision objects and seven pair-collision
intervals produced finite loss `3.4391`, raw gradient `28.1387`, and local
coefficient `0.03554`. It retained complete trajectory/objective support and
was followed by twelve consecutive sampled blocks with gradients
`0.1672..3.8626`; no second severe event occurred through step 256. The
post-128 audit therefore passes with this isolated warning rather than showing
a systematic scale collapse through that boundary.

The warning later recurred at sampled update 280, exactly 128 updates after
step 152. Both failures use deterministic frames 7--11, heavy ground/pair
contact, no external actuation, ordinary finite total loss, and complete
objective support. Step 280 has loss `2.9351`, raw interaction gradient
`52.9646`, and retained coefficient `0.01888`; it followed a normal step 264
and therefore is not monotonic parameter divergence. This exact periodic
recurrence makes the failure systematic enough to stop. The job was unloaded
immediately, preserving clean durable step 256; shutdown emitted only Python's
14-semaphore resource-tracker warning. The run has no trained selector and
cannot resume or count toward convergence.

Checkpoint Adam-moment audit localizes the dominant variance to the typed
relation decoder's collision-logit row: its weight second-moment RMS is
`0.03050`, versus `0.01542/0.01380` for normal/tangent force,
`3.13e-5` for relation projection, `7.41e-6` for entity projection, and
`1.43e-6` for the normalized scene projection. This rules out another scene
input-scale failure. Specification 1.25 adds an optional collision-row norm cap
before the complete interaction/global caps, with true raw hierarchy
reconstruction and auditor visibility. The attention pilot config sets this
row cap to `1.0`; forward physics and event semantics are unchanged.

Repair verification reports `236 passed` for the first focused set and
`216 passed` for the final affected config/training/auditor set. Complete
non-device verification reports `657 passed, 5 skipped, 1 deselected` in
`164.77 s`; host MPS reports `1 passed, 662 deselected` in `3.02 s`.
Ruff check/format, compileall, and diff checks pass. A fresh weights-only
campaign is now active at
`runs/20260810-180502-attention-collision-isolated-stage-a/` under one-shot
Standard/default LaunchAgent
`com.polceanum.orpheus.attention-collision-isolated-20260810-180502`,
`KeepAlive=false`, with trainer PID `88970`. It starts from pushed clean commit
`70c2e3b`; metadata verifies PyTorch `2.10.0`, MPS built/available and used for
RGB measurement, CPU closed loop, float32, RGB-only/no-oracle execution, and
the protected protocol-14 graph checkpoint as its weights-only initializer.
The resolved protocol hash is
`9cff424179133097847955f041cf35c73efb5947b66cd877b395b9c57f516fcb`.
The mandatory step-zero 32-episode selector completed all episodes in
`976.793 s` (`987.004 s` including persistence) and exactly reproduces the
protected score `0.3213162196`, current position `0.2514599 m`, velocity
`1.0931909 m/s`, axes `0.281775/0.201906/0.263691 m`, and horizons
`0.265184/0.277452/0.309911/0.335387/0.357837 m`. Coverage, precision,
identity, collision, and calibration metrics also match. `best_rollout.pt`,
`reference_rollout.pt`, and `validation_step_000000.pt` all record exact model
hash `1354bdfca1cef965c0cd907ea8c157c0fd82169e64f24da656eb42dd1a96df91`.
There are no selector guardrail/support failures and stderr remains empty.
Attention-only training is active; no trained accuracy or convergence
improvement is claimed from the equality control.

The repaired campaign has now reached its first durable checkpoint at update
128. The offline dynamics audit reports 128 applied updates, 16 cadence-eight
confirmations, exactly 16 sampled draws from every scenario, zero skipped
draws/failures/severe clips, trajectory support `154..486` (median `329`),
complete objectives, empty stderr, and stable peak RSS `2,990,858,240` bytes.
Sampled raw gradients span `0.9164..7.2588` with median `4.5879`; the varying
contact/event load produces heterogeneous batch loss rather than a monotonic
explosion. At update 112, for example, 24 ground-contact objects produce a raw
collision-row norm of `4.2285`; row isolation caps it before the remaining
interaction block and preserves a `0.3512` interaction-stage coefficient.
Update 120 is fully unclipped at `0.9164`, and update 128 retains `0.8287` at
the post-row interaction stage despite a raw collision-row norm of `2.9903`.

Exact step-128 checkpoint audit finds all 177 inherited tensors bitwise equal
to step zero, all 48 attention tensors changed, and all 48 serialized optimizer
states mapped only to named attention parameters at Adam step 128. Model and
optimizer tensors are finite. The mutable hash is
`bbadb23698e3712199c892e485e27e8480d8063a55901cd81112ef5d2cae9122` and
matches checkpoint metadata; `best_rollout.pt`, `reference_rollout.pt`, and
`validation_step_000000.pt` remain finite at step zero with exact protected
hash `1354bdfca1cef965c0cd907ea8c157c0fd82169e64f24da656eb42dd1a96df91`.
This proves optimizer/scope/resource integrity only. The former periodic
batches, first trained selector at update 512, and convergence still require
continued evidence; the campaign continues under the same one-shot job.

The first former failure boundary at update 152 now passes. Its deterministic
window again contains 22 ground-contact objects, seven pair-collision
intervals, and one wall collision. The stopped conditioned run produced raw
interaction norm `28.1387` and retained coefficient `0.03554`; the repaired
run produces `7.1111` and retains `0.14308` at the interaction stage. Its raw
collision row is `1.6490`, is locally reduced to `1.0`, and no longer forces a
severe complete-block cap. All 13 objectives and 343 trajectory targets remain
supported, the optimizer update applies, peak RSS remains unchanged, and
stderr is empty. The post-128 auditor passes with zero failures/severe clips.
This directly qualifies the update-152 repair but does not substitute for the
second recurrence at 280 or fixed-selector accuracy.

The durable step-256 checkpoint also passes exact scope and integrity audit.
All 177 inherited tensors remain bitwise equal to step zero, all 48 attention
tensors changed, and all 48 serialized optimizer states map only to named
attention parameters at Adam step 256. Model and optimizer state are finite;
the mutable hash
`a53faf0a5e32aace06cffcd1a3a595eb3d6f4740d110c8c6f578f00059712690`
matches metadata, while all three protected step-zero artifacts retain exact
hash `1354bdfca1cef965c0cd907ea8c157c0fd82169e64f24da656eb42dd1a96df91`.
The aggregate auditor reports 256 applied updates, 32 draws from each scenario,
zero skips/failures/severe clips, trajectory support `154..504` (median `339`),
and unchanged `2,990,858,240`-byte peak RSS. One update-248 batch reaches four
trusted identity switches over 63 associations (`6.35%`), versus `3.08%` on
the stopped conditioned control; update 256 returns to zero and aggregate
trusted switching remains 14/1,436 (`0.975%`). This warning remains reserved
for fixed-selector judgment rather than being hidden or prematurely repaired.
At update 280 the periodic severe whole-interaction failure recurs in a new
form. Raw norm is reduced from the conditioned run's `52.9646` to `17.7050`,
but the interaction-stage coefficient is still severe at `0.05648`. Crucially,
the collision-row norm is only `0.23553` and receives no row-local clipping;
the post-row interaction norm remains the full `17.7050`. The batch has all 13
objectives, 145 trajectory targets, 19 ground-contact objects, four pair
intervals, one wall collision, finite loss/state, and stable RSS. The auditor
correctly retains the severe warning. The LaunchAgent is stopped and absent;
shutdown emitted only the known 14-semaphore resource-tracker warning. The
verified step-256 checkpoint remains exact and the failed update is not
checkpointed.

Specification 1.26 now requires complete read-only gradient localization
before another repair. Training telemetry records the raw norm of every named
attention parameter and every semantically labelled node/relation output row
before clipping. This is observability only and does not mutate gradients or
forward dynamics. The next experiment is an exact optimizer/RNG/data replay
from step 256 through 280 to identify the actual dominant path. No first
trained selector or convergence result exists.

The present capacity is `1,103,626` typed-attention parameters and `3,004,656`
parameters for the complete model, with at most 22 scene/entity/relation
tokens under the six-slot contract. Review of the original Transformer and
current dense/MoE systems supports retaining dense scaled-dot-product
attention, pre-RMSNorm, and SwiGLU for this rung. GQA/latent attention/local
attention primarily address long-context KV cost, while MoE introduces a
routing/load-balancing axis; neither addresses the current short structured
set or earns inclusion before dense stage A converges. Capacity will therefore
increase one dense axis at a time only after broad promotion, with data draws
and disjoint/OOD qualification increased alongside it as required by
specification section 191.

Historical verification for specification 1.22 before the live-scene repair:

- `conda run --no-capture-output -n orpheus pytest -q -p no:cacheprovider -m 'not device'`
  reports `649 passed, 5 skipped, 1 deselected` in `168.52 s`.
- Host-MPS `PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q -p no:cacheprovider -m device`
  reports `1 passed, 654 deselected` in `3.11 s`.
- The final focused attention/config/scope/checkpoint set reports `251 passed`
  in `20.54 s`.
- `ruff check .`, `ruff format --check .`, `git diff --check`, and compileall
  pass.

## 2026-08-10 — fast-ROI ownership tie repaired; small control de-noised

The protocol-20 endpoint regression was localized to validation seed `100024`
(`reference_pairs`). Before any assignment changed, a `0.0000765` maximum
change in the predicted RGB measurement at `1.60 s` changed one structured ROI
centre by `0.2807869` normalized image units. The two disconnected foreground
components had ownership margins `0.0` and `2.98e-8`; the crop sampler was
therefore choosing an identity-bearing component at a numerical tie. The later
ROI miss and global ambiguity were downstream effects. Collision masks,
boundary events, modes, and association structure were unchanged before the
measurement jump.

A blanket fast-path distance cap was explicitly rejected: it improved the
failing episode and identity count but worsened the 32-episode public posterior
and every model horizon by rejecting legitimate long-range evidence. The
implemented repair instead grows the selected RGB-connected component, finds
the nearest supported pixel outside it, records the ownership margin, and
rejects only margins within a scale-aware `32 * eps` equality tolerance. The
source-bound ROI retains its predicted centre; global discovery remains the
only large/ambiguous recovery path. Rollout validation advances from protocol
13 to 14.

On the exact failing episode, corrected step 64 and step 512 have no structural
association difference, no measurement delta above `0.01`, and no posterior
state delta above `0.001`. Their x SSE/count are respectively
`6.3904306/66` and `6.3918110/66`, versus the old `6.7835381/72` and
`8.2411388/72` discontinuous pair.

Paired 32-episode RGB-only public evaluation of the same step-64 weights gives:

| metric | protocol-13 behavior | ownership-tie repair |
|---|---:|---:|
| posterior current position RMSE | 0.8087382 m | 0.8079388 m |
| posterior current x RMSE | 0.7304025 m | 0.7169902 m |
| posterior current velocity RMSE | 1.1085611 m/s | 1.0949822 m/s |
| model 0.10/0.25/0.50/0.75/1.00 s RMSE | 0.692976/0.689654/0.695147/0.701112/0.693358 m | 0.692282/0.689676/0.694432/0.697434/0.690292 m |
| distance-gated identity switches | 37 | 35 |
| collision F1 | 0.166227 | 0.171504 |
| forecast Gaussian NLL | 1.197517 | 1.190278 |

Detection recall/precision change only `0.35875 -> 0.35800` and
`0.334421 -> 0.333955`; y/z posterior RMSE change by `+0.64%/+0.52%` while
the joint posterior, x, velocity, identity, collision, calibration likelihood,
and four of five horizons improve. There are zero nonfinite outputs.

The stricter 32-episode physical selector improves protected step 64 from the
stored protocol-13 score/current/velocity/identity values
`0.3215594 / 0.2532523 m / 1.0953541 m/s / 0.0142487` to protocol-14 values
`0.3213162 / 0.2514599 m / 1.0931909 m/s / 0.0135922`. Target coverage remains
`0.37625`; horizons are
`0.265184/0.277452/0.309911/0.335387/0.357837 m`. Re-evaluated step 512 is
effectively identical but still worse at score `0.3213287`, so longer y-only
training did not improve the corrected plateau and step 64 remains the small
control.

Durable audit artifacts are under
`runs/20260810-105616-fast-roi-ownership-audit/`; public baseline/candidate
reports and the complete step-512 physical log are preserved there. Known
limitations remain real-video/OOD qualification, same-connected-component
overlap ownership, legacy-reference replacement, and the unimplemented typed
attention pilot.

Final verification on this tree:

- `conda run -n orpheus pytest -q -p no:cacheprovider tests/integration/test_checkpoint_roundtrip.py`
  reports `27 passed`.
- `conda run --no-capture-output -n orpheus pytest -q -p no:cacheprovider -m 'not device'`
  reports `642 passed, 5 skipped, 1 deselected` in `167.63 s`.
- Host-MPS `PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q -p no:cacheprovider -m device`
  reports `1 passed, 647 deselected` in `3.27 s`.
- `conda run -n orpheus ruff check .`, `ruff format --check .`,
  `git diff --check`, and compileall all pass. The first full-suite attempt
  correctly exposed a stale checkpoint specification constant (`1.20`); it is
  updated to `1.21`, the focused checkpoint suite passes, and the complete
  suite was rerun successfully.

## 2026-08-10 — joint recovery stopped; y-only corrected incumbent accepted

The isolated mean-head campaign at
`runs/20260810-012116-protocol19-anchored-mean-recovery/` was stopped cleanly
at durable step 192. The launchd service and PID are gone; no stale state file
is being treated as a live worker. Across 24 logged eight-scenario blocks it
applied every optimizer update, skipped and clipped none, kept RSS bounded at
`1,327,321,088` bytes, and passed support/identity/uncertainty auditing. Its
three exact trained selectors nevertheless confirmed a rejected plateau:

| step | score | current RMSE | velocity RMSE | failures |
|---:|---:|---:|---:|---:|
| 64 | 0.3246722 | 0.2562508 m | 1.1053538 m/s | 15 |
| 128 | 0.3246772 | 0.2566319 m | 1.1048962 m/s | 17 |
| 192 | 0.3249595 | 0.2568297 m | 1.1040418 m/s | 24 |

The step-zero corrected reference remained protected at score `0.3241755`.
The worker was not allowed to consume the remaining declared updates after
1,536 balanced episode draws and a third comparable rejection proved that the
same joint-row direction was worsening rather than recovering.

Row-level composition now supplies exact source/donor/tensor provenance and
runs the unchanged 32-episode RGB-only selector. Results are preserved at:

- `runs/20260810-033923-protocol19-step64-x-row/`: rejected, score
  `0.3246780`, 15 failures. X alone reproduced almost all downstream
  reference-pair y/z regression through later association and rollout.
- `runs/20260810-034635-protocol19-step64-y-row/`: accepted, score
  `0.3234974`, zero failures.
- `runs/20260810-035350-protocol19-step64-z-row/`: rejected, score
  `0.3242361`, two failures.
- `runs/20260810-040115-protocol19-step128-y-row/`: rejected despite score
  `0.3235735`, two scenario guardrail failures.
- `runs/20260810-040840-protocol19-step192-y-row/`: accepted, score
  `0.3216427`, zero failures. This is the new corrected recovery incumbent.

The accepted step-192 y-only candidate improves current position
`0.2559540 -> 0.2537443 m`, velocity `1.0966767 -> 1.0949210 m/s`, identity
switch rate `0.0155844 -> 0.0142579`, and every joint horizon from
`0.270262/0.279750/0.311927/0.337838/0.361140 m` to
`0.267754/0.277482/0.309300/0.335433/0.358422 m`. Current x/z and one-second
x/z also improve through the full online trajectory, even though only row 1
of the mean-head weight/bias comes from the donor. This confirms that output-
row ablations are not trajectory-axis-independent.

Specification 1.20 and `updater_mean_y` now make the next optimizer scope
exact: non-y rows have gradients and per-element Adam moments cleared, and
their values are restored after decoupled weight decay. Verification commands
and outcomes on this Mac were:

```text
conda run -n orpheus pytest tests/unit/test_training_schedule.py tests/unit/test_config.py tests/unit/test_checkpoint_composition.py
# 208 passed in 13.59s
conda run -n orpheus pytest -q -p no:cacheprovider -m 'not device'
# 641 passed, 5 skipped, 1 deselected in 164.33s
conda run -n orpheus pytest -q -p no:cacheprovider -m device
# 1 passed, 646 deselected in 3.02s on host MPS
conda run -n orpheus ruff check ...
conda run -n orpheus ruff format --check ...
# pass
```

The next concrete action is a clean weights-only y-scope campaign initialized
from
`runs/20260810-040840-protocol19-step192-y-row/candidate.pt`, with MPS RGB
measurement, CPU closed-loop execution, balanced batches, reduced effective
learning rate, and exact validation every 64 updates. The corrected incumbent
still scores above the approximately `0.318` legacy reference, so it is not a
deployment replacement and does not yet unlock the attention pilot.

That campaign is active at
`runs/20260810-042627-protocol20-y-only-recovery/` under one-shot launchd label
`com.polceanum.orpheus.protocol20-y-recovery-20260810-042609` as PID `58922`.
Metadata records clean commit `3ad5ee2`, PyTorch `2.10.0`, MPS built/available,
MPS measurement, CPU closed loop, RGB-only/no-oracle runtime, the accepted
step-192 y-row initialization, `updater_mean_y`, 512 balanced updates, effective
learning rate `5e-6`, and checkpoint/evaluation cadence 64. Initialization and
the first 64 optimizer updates completed with no skip, clip, nonfinite value,
scope failure, or restart. The eight logged balanced blocks contain equal
support from every scenario, 156--442 causal trajectory rows, finite gradients
from `0.000500` to `0.029813`, 382 gated identity associations with five
switches, and bounded RSS through `1,346,781,184` bytes.

Exact step-64 RGB-only validation is preserved at
`runs/20260810-042627-protocol20-y-only-recovery/checkpoints/validation_step_000064.pt`.
It passed every moving-incumbent
guardrail and was internally accepted: score `0.3216427 -> 0.3215594`, current
position `0.2537443 -> 0.2532523 m`, and identity `0.0142579 -> 0.0142487`.
Current x/y/z each improved. The 0.10-second horizon improved
`0.2677536 -> 0.2669477 m`, but 0.50--1.00 seconds changed by only
`0.0000008--0.0000061 m` in the worse direction; velocity also changed
`1.0949210 -> 1.0953541 m/s`. These are guardrail-safe, not evidence of broad
convergence, and the corrected control remains behind the approximately
`0.318` legacy reference.

Independent checkpoint comparison proves the optimizer scope held at runtime:
only `updater.learned_corrector.mean_head.{weight,bias}` changed, only leading
row 1 changed within those tensors, all other model tensors are bitwise equal
to step zero, and Adam `exp_avg`/`exp_avg_sq` are nonzero only in row 1. The
checkpoint retains specification 1.20 and clean source/runtime provenance.

The unchanged mutable trajectory then completed step 128 without an optimizer,
support, numerical, clipping, memory, or scope failure, but its fixed candidate
was correctly rejected. Score regressed from the protected step-64 incumbent
to `0.3216703`, current position to `0.2536474 m`, velocity to
`1.0969573 m/s`, and identity to `0.0155440`. The selector reported three
fixed-reference failures and two moving-incumbent failures. They localize to
the baseline scenario: target coverage fell `0.32885 -> 0.32308`, identity
switch rate rose `0.00488 -> 0.01456`, and 0.10-second x RMSE rose
`0.41439 -> 0.42383 m`. Other scenarios are essentially flat or slightly
improved, and tensor comparison still finds changes only in mean-head y row 1.

This is a scenario/association threshold regression, not evidence of numerical
collapse. Step 192 then completed with the same healthy optimizer/support/
resource profile but did not recover: score `0.3216706`, current position
`0.2536476 m`, velocity `1.0969583 m/s`, identity `0.0155440`, and all three
baseline fixed-reference failures are effectively identical to step 128.

The y row is approaching a stationary point rather than becoming frozen by a
bug. Its weight change shrank from L2 `0.001135` over steps 0--64 to
`0.0000756` over steps 128--192, while finite Adam moments remain nonzero only
in row 1. Every other model tensor remains bitwise equal to step zero. Across
192 updates, the audit reports 24/24 logged balanced blocks applied, no skip or
clip, gradients `0.000489--0.029813`, 156--486 trajectory-support rows, equal
scenario counts, finite uncertainty, and bounded RSS through
`1,346,781,184` bytes.

Step 256 returned across the discontinuous baseline association threshold and
passed every fixed-reference and moving-incumbent guardrail. Its score
`0.3215611`, current position `0.2532510 m`, velocity `1.0953627 m/s`, identity
`0.0142487`, coverage, precision, collision F1, calibration, axes, and horizons
are all effectively the protected step-64 regime. It was correctly not
accepted because its score is still `0.00000167` worse than step 64.

Checkpoint geometry shows continued small y-row movement rather than an Adam
oscillation or scope defect: weight-row L2 changes were
`0.001135/0.000489/0.0000756/0.0001115` across the four 64-update segments,
finite row-1 moments persist, and every excluded row/tensor remains bitwise
fixed. The online association response is nonmonotonic across this smooth
parameter path, explaining why steps 128/192 shared one discrete identity and
steps 64/256 another.

Step 320 crossed back into the same rejected discrete baseline regime as steps
128/192: score `0.3216708`, current position `0.2536469 m`, velocity
`1.0969604 m/s`, identity `0.0155440`, and the same baseline coverage,
identity, and 0.10-second x failures. The 256--320 y-weight-row segment moved
smoothly by L2 `0.0000939`; only row 1 changed, its Adam state is finite and
nonzero, and all excluded model/optimizer rows remain exact. This strengthens
the association-threshold diagnosis rather than indicating optimizer
oscillation or scope leakage.

Step 384 returned to the guardrail-clean regime. Its score `0.3215634`,
current position `0.2532533 m`, velocity `1.0953645 m/s`, identity `0.0142487`,
coverage, precision, collision F1, calibration, axes, and all five horizons
are again effectively the protected step-64 behavior. It was not accepted:
the score is `0.00000405` worse than step 64 and fails only the required
minimum-improvement condition. The post-320 audit reports eight applied
balanced blocks, zero skips/clips/failures, gradients
`0.000215--0.006305`, support `232--468`, and unchanged bounded RSS.

Step 448 crossed into the rejected association regime. Score is `0.3216787`,
current position `0.2536490 m`, velocity `1.0969621 m/s`, and identity
`0.0155440`. The baseline scenario fails coverage and identity against the
fixed reference, and also fails 0.10-second x RMSE against both the protected
incumbent and fixed reference. This x regression is a downstream trajectory
effect rather than x-head drift: exact checkpoint comparison from step 384
to 448 finds changes only in `mean_head.{weight,bias}` row 1, with weight-row
L2 `0.0001126` and bias delta `0.0000080`; every excluded tensor remains
bitwise fixed. The segment audit again passes all optimizer, support,
uncertainty, identity, sampling, memory, and numerical checks.

The campaign completed all 512 declared updates and its final 32-episode
selector in `19,798.94 s`. The one-shot launch agent exited with code zero and
stderr remained empty. Step 512 was rejected at score `0.3216317`, current
position `0.2537522 m`, velocity `1.0953680 m/s`, identity `0.0142764`, and
pooled horizons `0.267573/0.277447/0.309314/0.335455/0.358436 m`. It failed
current x and 0.10-second x guardrails only in `reference_pairs`, despite y
RMSE reaching the run minimum `0.2075070 m`. Step 64 remains the protected
best at score `0.3215594`.

The completed learning curve alternates between near-identical clean and
rejected association regimes while its sole trainable row moves smoothly.
Across all 64 logged optimizer blocks, every update applied, every scenario
appeared exactly 64 times, no draw was skipped, no gradient was clipped,
gradient norms were finite (`0.0000239--0.062438`, median `0.001527`), causal
support was `123--519`, and RSS remained `1.285--1.347 GB`. Full step-zero to
step-512 comparison finds only `mean_head.{weight,bias}` row 1 changed (weight
L2 `0.002242`, bias L2 `0.000166`); Adam moments are nonzero only in row 1.
Protocol 20 therefore establishes a healthy but non-improving plateau. More
updates in the same y-only direction are not justified; the next accuracy
work must make association/trajectory feedback robust before capacity scaling.

Step 64 remains the immutable selected incumbent. The run completed its
predeclared 512-update budget without another acceptance and established the
applicable bounded-recovery plateau evidence. It is not a deployment
replacement or an attention-scaling authorization.

## 2026-08-09 — protocol-18 rejection and innovation-anchored repair

Protocol 18 reached a durable step-128 checkpoint with 128 balanced optimizer
updates, zero skipped updates, all 13 causal objectives, 486 trajectory-support
rows, finite model/Adam tensors, complete RNG state, and synchronized data draw.
The rare step-64 recursive-velocity spike remained bounded and did not recur in
the next logged blocks. The trainer and supervisor were deliberately booted out
after exact early validation proved broad accuracy regression; the stale
`training_state.json` is not evidence of a live process.

Exact fixed 32-episode RGB-only validation is preserved at
`runs/20260809-224159-protocol18-step128-early-diagnostic/report.json`. The
weighted score worsened `0.3189518 -> 0.3340991`, current position worsened
`0.2502196 -> 0.2683071 m`, velocity was flat/slightly worse
`1.0792038 -> 1.0794120 m/s`, and all five horizons worsened from
`0.261999/0.270180/0.305863/0.334885/0.357770 m` to
`0.277947/0.290649/0.324950/0.350739/0.366774 m`. Identity and collision F1
improved, but elastic-pair x prediction regressed catastrophically. The
candidate is rejected.

Exact module ablations established causality. Dynamics-only at
`runs/20260809-225002-protocol18-step128-dynamics-only/report.json` was nearly
neutral (`0.3189518 -> 0.319478`), with small mixed scenario changes.
Updater-plus-identifier at
`runs/20260809-225754-protocol18-step128-updater-identifier/report.json`
worsened the score to `0.335865`; updater-only at
`runs/20260809-230632-protocol18-step128-updater-only/report.json` reproduced
it at `0.335849`. The identifier is negligible and late dynamics partly
compensated for a defective correction path.

Code inspection found two contract violations in that path. The corrector
received only pooled camera-space innovation statistics, losing explicit axis
and sign evidence, then applied learned fast-state and variance deltas to every
packed component regardless of `supported_state_fields`. The specification
1.19 repair is opt-in to preserve historical checkpoints: learned mean
residuals become bounded gains on explicit whitened world-state innovation;
per-axis confidence and support mask mean and variance updates; zero supported
innovation produces exactly zero learned mean update. RGB position evidence
can affect position and its documented temporal velocity coupling, but not
orientation, angular velocity, or modal state. Focused
filter/config/checkpoint tests report `142 passed`, including a finite nonzero
learned-gradient regression. Full non-device regression reports `636 passed, 5
skipped, 1 deselected` in `173.22 s`; the sandbox cannot
expose MPS, while the same device-marked command on the host reports `1 passed,
640 deselected` in `4.72 s`. Ruff check and repository-wide format check pass.
The first fixed 32-episode qualification is complete and exposes an inherited-
head migration defect described below; a reset-head qualification remains due
before sustained training.

The clean source is committed and pushed as `b53c116`. Qualification run
`runs/20260809-233714-protocol19-anchored-correction-qualification/` loaded the
protocol-17 step-512 weights under the new semantics and evaluated the exact
32-episode selector before training. The step-zero pooled score improved
slightly from the legacy-semantics `0.3189518` to `0.3180155`, and 0.50/0.75/
1.00-second RMSE improved to `0.303777/0.331702/0.355324 m`. However, current
position worsened `0.250220 -> 0.251928 m`, velocity worsened
`1.079204 -> 1.103289 m/s`, and 0.10/0.25-second RMSE worsened to
`0.265631/0.274496 m`. Scenarios were mixed: baseline, impulse, and heavy/light
current RMSE improved, while reference, elastic, damped, camera, and especially
glancing current RMSE regressed. This is not broad acceptance.

The cause is deterministic: the old learned mean head represented an absolute
state delta, while specification 1.19 interprets it as an innovation gain.
Shape-compatible loading silently reused incompatible numbers. One balanced
training update was finite and fully supported (`loss=3.532505`, raw/applied
gradient `0.548843`, all 13 objectives, 257 trajectory rows, zero skips, 1.19
GB maximum RSS), proving the new path trains; nevertheless it immediately
worsened the selector score to `0.322579` and every x metric, so its step-one
candidate was correctly rejected with 19 guardrail failures. The step-zero
checkpoint remains the run's internal best but is not promoted over the legacy
reference.

`scripts/evaluate_modular_candidate.py` now supports a deterministic
`--fresh-initialization` donor with explicit seed/module provenance. Resetting
mean, variance, and gate heads together was safely rejected at
`runs/20260809-235830-protocol19-anchored-reset-heads/`: score worsened to
`0.350730`, although every horizon gained forecast coverage. This reset was too
broad because variance remains a log-variance residual and the gate remains a
gate; their mathematical meanings did not change.

The exact mean-only reset is preserved at
`runs/20260810-000755-protocol19-anchored-reset-mean/`. It retained support and
the compatible variance/gate/trunk/lifecycle outputs, but score still worsened
from the inherited-gain `0.318015` to `0.324176`; horizons became
`0.270262/0.279750/0.311927/0.337838/0.361140 m`, and the selector reported 71
guardrail failures. This is expected evidence that the old absolute residual
had learned useful measurement-bias correction; it is not accepted, deleted,
or mislabelled as a gain model.

The mean-only `candidate.pt` is now the clean mutable recovery start. A new
`updater` trainable scope freezes dynamics, identifier, and all RGB perception
while training the anchored filter (excluding its already disconnected
visibility head). The next bounded campaign must recover the fixed legacy
reference with updater-only balanced optimization before any joint dynamics
phase or attention-capacity pilot.

The recovery scope and evidence were committed/pushed as clean `d97b613` after
`200` focused schedule/config/spec tests plus Ruff and format checks passed.
One-shot Standard launchd job
`com.polceanum.orpheus.protocol19-updater-recovery-20260810-0020` is active as
PID `46547` at
`runs/20260810-001838-protocol19-anchored-updater-recovery/`. Its immutable
metadata records clean commit `d97b613`, PyTorch `2.10.0`, MPS built/available,
MPS RGB measurement, CPU closed-loop dynamics, RGB-only runtime, no oracle,
and the exact mean-reset `candidate.pt` source. Stderr is empty and the fixed
initialization completed all 32 episodes in `700.677 s` through the expected
ordered scenarios. Its score was exactly `0.3241755`, matching the independent
mean-reset qualification and confirming reproducible checkpoint loading and
selector execution.

The bounded recovery declares 512 balanced updater-only updates, 4,096 episode
draws, learning rate `2e-5`, exact 32-episode validation/checkpoint cadence 64,
and logging cadence 8. It is not a convergence or promotion claim. The first
causal rows at steps 8 and 16 verify updater-only scope, finite nonzero
unclipped gradients (`0.2613` and `0.3101`), zero perception/interaction
gradient, balanced eight-scenario membership, and zero skipped draws. Step 8
had eight contributing objectives because its frame-3 anchor contained only
cold-start tracks; deterministic rollout losses were correctly withheld while
rollout NLL remained supported. Step 16 used a mature frame-9 anchor and
restored all 13 causal objectives, including rollout position and velocity.
The first trained fixed selector remains due at step 64; each selector must
compare against both the clean reset start and the legacy fixed reference.

At the step-64 boundary, all eight logged optimizer blocks had applied with
every scenario represented exactly eight times, zero skipped draws, no clipped
or severe-clipped block, median raw gradient `0.3560`, median causal support
`334`, and peak RSS `1,332,187,136` bytes. The dynamics auditor reports
`status=pass` with no failures or warnings. Exact validation then rejected the
step-64 candidate: score worsened `0.3241755 -> 0.3384320`, current position
worsened `0.2559540 -> 0.2747649 m`, coverage fell `0.3765 -> 0.36075`, and all
five horizons worsened from `0.270262/0.279750/0.311927/0.337838/0.361140 m`
to `0.286194/0.293699/0.328519/0.354228/0.371656 m`. Velocity improved
`1.096677 -> 1.068461 m/s` and collision F1 improved
`0.187817 -> 0.213270`, but these do not offset broad position regression;
elastic-pair x forecasts failed particularly severely. The selector retained
step zero and the one-shot job was stopped at the durable step-64 checkpoint.

Checkpoint deltas exposed a narrower protocol defect: `updater` trained the
compatible corrector trunk, modality embedding, variance/gate, mode, and
existence paths alongside the reset mean head. Shared trunk matrices moved by
`0.10--0.22` L2 while the new mean-head weight moved only `0.0068`, confounding
semantic recovery with broad updater forgetting. A new `updater_mean` scope
now makes exactly `mean_head.weight` and `mean_head.bias` trainable. Focused
schedule/config tests report `200 passed`; the complete non-device suite reports
`638 passed, 5 skipped, 1 deselected` in `190.97 s`, and the host device suite
reports `1 passed, 643 deselected` in `4.64 s`. Ruff lint and repository-wide
format checks pass.
The corrected campaign must restart from the clean mean-reset candidate at
effective learning rate `5e-6`, not continue from the rejected step-64 state.
That replacement is active at
`runs/20260810-012116-protocol19-anchored-mean-recovery/` under one-shot launchd
job `com.polceanum.orpheus.protocol19-mean-recovery-20260810-012053` as PID
`49490`. Metadata records clean commit `c3f982c`, PyTorch `2.10.0`, MPS
built/available, MPS measurement, CPU closed loop, RGB-only runtime, no oracle,
the clean mean-reset source, scope `updater_mean`, and effective LR `5e-6`.
The process is actively computing the exact step-zero selector at roughly
`542%` CPU with empty stderr. Initialization completed all 32 episodes in
`741.313 s` and reproduced score `0.3241755` exactly.

The corrected campaign then reached durable step 64 with eight balanced logged
blocks, zero skips or clips, median raw gradient `0.01269`, median causal
support `334`, peak RSS `1,327,321,088` bytes, and a warning-free dynamics
audit. Exact checkpoint comparison proves only
`updater.learned_corrector.mean_head.weight` and `.bias` changed; every other
model tensor is bitwise unchanged. This validates the new scope.

Fixed step-64 validation was near-neutral but rejected: score changed
`0.3241755 -> 0.3246722`, current position `0.2559540 -> 0.2562508 m`, and
velocity `1.0966767 -> 1.1053538 m/s`. Coverage, precision, identity, and
collision F1 were flat or better. Every pooled x metric improved, including
1-second x RMSE `0.4856162 -> 0.4811387 m`, while small y/z regressions were
concentrated in `reference_pairs`. Guardrail failures fell from 110 in the
rejected updater-wide step 64 to 16. The selector retained step zero; the
mean-only mutable iterate remains numerically viable and is continuing toward
step 128 under the declared long run. This is recovery evidence, not promotion
or convergence.

Specification 1.19 also records the compute path reviewed against transformer,
Perceiver IO, compute-optimal scaling, JEPA/video-world-model, and recent
object-centric physical-prediction literature. The first scale rung is not a
large pixel transformer: after the correction repair passes, a 1--4M parameter
pre-normalized entity/relation/event attention pilot will consume derived
`WorldBelief` tokens and decode typed residual proposals. It must beat the
accepted smaller and parameter-matched graph controls on disjoint RGB-only
generalization and every existing guardrail before a wider CUDA rung.

## 2026-08-09 — protocol-18 early dynamics and severe-clip observability

Protocol 18 established the imported step-512 protocol-17 candidate as its
immutable step-zero reference on the unchanged 32-episode selector. The
reference score is `0.3189518`; this is initialization evidence, not a new
accuracy result. Through update 72, all nine logged optimizer blocks applied,
every scenario appeared exactly once per update, no causal-support, finite-
state, frozen-perception, worker, or memory failure occurred, and maximum RSS
stabilized at `1,423,511,552` bytes.

The first seven balanced blocks reduced median raw gradient from `4.5935` in
the comparable protocol-17 prefix to `0.9645`, reduced clipping from `7/7` to
`2/7`, and raised median causal trajectory support from `69` to `349`. Update
64 then exposed a real rare-window spike: raw interaction/full norms were
`28.5453 / 30.3853`; local interaction clipping retained `0.0350`, and global
clipping retained an overall `0.0658`, producing a finite applied norm of
`2.0`. The next logged update returned to an unclipped `0.9823` norm with all
13 objectives and 385 trajectory rows.

An exact CPU replay from `validation_step_000000.pt` localized the spike to
the `baseline` episode seed `16081`: its interaction norm was `37.1272`, while
the other seven scenarios were at or below `1.3271`. Weighted objective
autograd attributed `36.7589` to recursive rollout velocity; every other term
was at or below `0.4291`. The interaction output-row gradients were concentrated
in continuous normal/tangential force, not impulse residuals, event BCE,
uncertainty, or parameter identification. This is currently one bounded hard
trajectory rather than evidence of numerical collapse; recurrence frequency
remains under audit before any optimization change.

The dynamics auditor previously treated any finite clipped update as an
unqualified pass. It now warns and reports exact steps whenever global or
interaction clipping retains less than 10% of the raw gradient. Focused tests
report `5 passed`; Ruff check and format-check pass. Running it on protocol 18
now reports step 64 with total/interaction coefficients `0.0658212 / 0.0350320`.
No checkpoint is promoted; the first trained fixed validation remains due at
step 512.

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

The repair was committed and pushed to `main` as clean commit `b646582`. The
protocol-17 supervisor and trainer were then booted out; their last logged row
is step 2,360 and their last durable checkpoint is step 2,304. Because direct
SIGTERM cannot execute Python's terminal-state writer, its generated
`training_state.json` remains stale at `running`; absent launchd jobs and this
record are authoritative. No protocol-17 step after 2,048 was broad-validated
or promoted.

The new run is
`runs/20260809-212649-protocol18-balanced-scenarios/`, initialized weights-only
from protocol-17 `validation_step_000512.pt` (SHA-256
`67f197f136de8be98e9cfdc3c070cfe69ec0499e51b1f77d2955dcbf8978472d`).
One-shot Standard launchd trainer
`com.polceanum.orpheus.protocol18-balanced-20260809-212627` is active as PID
`38073`; metadata records clean commit `b646582`, PyTorch `2.10.0`, MPS
built/available, MPS measurement, CPU closed loop, RGB-only runtime, and no
oracle. Initialization heartbeat has completed three of 32 fixed episodes
under unchanged selector hash
`e31bf1cde4e4adf8603190b3258e086d6f749ad1d5689427d60e367f9fbb53a0`;
trainer stderr is empty.

Exact-source supervisor
`com.polceanum.orpheus.protocol18-convergence-20260809-212627` is active as PID
`38232` from detached clean clone
`/private/tmp/orpheus-protocol18-runtime-b646582`. It requires 4,096 balanced
updates, may extend by complete 2,048-update dataset passes, retains the
four-validation/1% plateau rule, and has a 12,288-update hard limit. Supervisor
source hash matches the repository and stderr is empty. No protocol-18 trained
candidate, promotion, plateau, or convergence result exists yet.

Verification for specification 1.18:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider -m 'not device'
# 630 passed, 5 skipped, 1 deselected in 201.67s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output \
  -n orpheus pytest -q -p no:cacheprovider -m device
# 1 passed, 635 deselected in 4.53s on host MPS

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus ruff check .
# All checks passed

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n orpheus \
  ruff format --check .
# 190 files already formatted

PYTHONPYCACHEPREFIX=/private/tmp/orpheus-protocol18-pycache PYTHONPATH=. \
  conda run -n orpheus python -m compileall -q \
  train.py evaluate.py demo.py scripts world_model tests
# exit 0; git diff --check also passed
```

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

The predeclared four-episode protected matrix is now complete under
`runs/20260813-221000-hypothesis-pool-protected-decay-matrix/`. Across 592
queries, decay `1.0` selected learned/baseline `587/5` times; decay `0.1`
selected `565/27`. Mean selected x/y/z RMSE at horizons 0.10/0.25/0.50/0.75/1.00
seconds was respectively:

```text
decay 1.0: 0.4940/0.3384/0.6390, 0.5171/0.3135/0.6477,
          0.5577/0.2239/0.6622, 0.5959/0.1650/0.6747,
          0.6190/0.2084/0.6628 m
decay 0.1: 0.4937/0.3385/0.6391, 0.5170/0.3099/0.6480,
          0.5575/0.2195/0.6622, 0.5959/0.1647/0.6747,
          0.6168/0.2052/0.6628 m
```

This is a small protected comparison, not a full incumbent selector. Decay
0.1 improves adaptation and has no broad regression here, but the effect is too
small to justify changing the deployed default or restarting training alone.

The evaluation harness now also reports persistent-ID-aligned lifecycle
mismatch, identity coverage, collision precision/recall/F1, and selected
position uncertainty. A toy smoke exposed an ID mismatch in the first version;
the corrected nearest-current-position bootstrap now yields non-empty aligned
metrics. This matters because RGB runtime IDs are not required to equal
simulator slot IDs.

The eight-episode protected run with decay `0.1` is retained at
`runs/20260813-223000-hypothesis-pool-protected-8ep/report.json` (1,184 scored
queries). Learned/baseline selection counts are `1,149/35`. Mean selected
x/y/z RMSE at 0.10/0.25/0.50/0.75/1.00 seconds is
`0.6112/0.3227/0.6614`, `0.6214/0.2894/0.6673`,
`0.6482/0.2189/0.6827`, `0.6645/0.2097/0.6972`, and
`0.6959/0.2219/0.7127 m`. Candidate learned-only RMSE is slightly worse at
every horizon, so selection provides a small improvement without changing the
incumbent. Mean selected collision F1 is `0.2044/0.1891/0.1861/0.1950/0.1834`;
the constant-velocity candidate has zero collision F1 and is materially worse
at mature horizons. Lifecycle mismatch totals are `274/244/194/150/115` and
mean selected position standard deviation is `0.636/0.644/0.660/0.679/0.704 m`.
This is stronger evidence for guarded fallback, but it is still an 8-episode
qualification rather than the required 32-episode promotion protocol.

The candidate pool now also supports `BallisticContactDynamics`, an analytic
gravity/drag hypothesis with conservative ground and sphere-contact event
logits. The evaluation harness accepts three candidates (learned,
constant-velocity, ballistic) and all nine focused selector tests pass. A fresh
toy RGB smoke selected the learned candidate on all 59 queries; the ballistic
candidate was not selected, so no accuracy gain is claimed yet. This negative
result is retained as evidence that event-aware candidate construction still
needs protected multi-episode qualification.

That protected qualification is now complete at
`runs/20260814-003000-hypothesis-pool-3cand-8ep/report.json` (1,184 queries).
Selection counts are learned/constant-velocity/ballistic `1148/35/1`. Mean
selected x/y/z RMSE is `0.6112/0.3227/0.6614 m` at 0.10 s,
`0.6214/0.2891/0.6673 m` at 0.25 s, `0.6482/0.2194/0.6826 m` at 0.50 s,
`0.6645/0.2096/0.6972 m` at 0.75 s, and `0.6959/0.2220/0.7127 m` at 1.00 s.
Ballistic candidate mean collision F1 is `0.2097/0.1479/0.1142/0.0931/0.0250`
across those horizons; it is worse than learned at mature horizons and was
selected only once. Keep it diagnostic and unpromoted.

Candidate selection now accepts explicit composite evidence weights for
position, lifecycle, and collision events. A protected eight-episode run with
`event_weight=0.5`, `lifecycle_weight=0.1` selected learned/constant/ballistic
`643/517/24`; collision F1 improved strongly, but y position regressed at
mature horizons, so that setting is rejected. A four-episode lower-weight
screen at `event_weight=0.1`, `lifecycle_weight=0.05` selected
`459/128/5` and improved mean collision F1 at 0.10/0.25/0.50 seconds to
`0.3755/0.2978/0.2846` versus learned-only `0.2168/0.1233/0.1644`, with
selected x/y/z RMSE `0.5174/0.3173/0.5331`, `0.5433/0.3290/0.5398`, and
`0.6009/0.2504/0.5544 m` at those horizons. This is a promising screen, not
yet a full promotion result.

The lower-weight eight-episode validation is complete at
`runs/20260814-030000-hypothesis-pool-eventweight01-8ep/report.json`.
Selections are learned/constant/ballistic `928/241/15`. Mean selected collision
F1 improves over learned-only at every horizon (`0.2985/0.2895/0.2418/0.2029/0.1274`
versus `0.2002/0.1737/0.1763/0.1807/0.1259`), while selected x/y/z RMSE is
`0.6115/0.3328/0.6632`, `0.6213/0.3399/0.6700`,
`0.6511/0.2603/0.6844`, `0.6685/0.2390/0.6962`, and
`0.6973/0.2369/0.7116 m`. The event gain is real, but mature position is not
uniformly non-regressive; event weighting remains opt-in and the position-only
incumbent default is unchanged.

The required 32-episode decay-0.1 protected comparison completed from the
immutable reference checkpoint at
`runs/20260813-230000-hypothesis-pool-protected-32ep/report.json`. It contains
4,736 scored frame/horizon queries, with learned/baseline selections
`4,599/137`. Relative to learned-only, posterior selection slightly improves
every pooled position horizon: selected mean x/y/z RMSE is
`0.7114/0.4940/0.8543`, `0.7206/0.4555/0.8644`,
`0.7484/0.3588/0.8831`, `0.7748/0.3318/0.9059`, and
`0.8129/0.3190/0.9286 m` at 0.10/0.25/0.50/0.75/1.00 seconds. Learned-only
values are respectively `0.7115/0.4940/0.8543`,
`0.7211/0.4564/0.8644`, `0.7506/0.3611/0.8829`,
`0.7812/0.3408/0.9056`, and `0.8193/0.3205/0.9283 m`.
Selected collision F1 is `0.1469/0.1722/0.1682/0.1598/0.1491`; learned-only is
`0.1483/0.1697/0.1669/0.1618/0.1515`, while the constant-velocity candidate
has zero event F1. Lifecycle mismatch totals are `720/636/503/383/280`,
identity coverage totals are `3249/2967/2499/2037/1583`, and selected mean
position standard deviation is `0.749/0.764/0.795/0.836/0.882 m`.
This passes the guarded selector comparison without broad regression, but the
gain is small and does not justify promoting a new checkpoint or restarting
training. An attempted MPS smoke was rejected by the runtime availability
check (`compiled=True, available=False`), so this protocol is truthfully
CPU-only.

The first process ended at the session boundary before producing output. The
harness was then optimized to compute all requested horizons in one rollout per
frame (then assimilate evidence per horizon), preserving the selection protocol
while reducing redundant dynamics work. A batched attention smoke completes
one episode in `103.78 s` CPU.

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
### 2026-08-13 paper-guided accuracy review

The active `attention-relation-constant-stage-a` campaign is still running with finite metrics, nonzero causal support, and no skipped batches; it is not yet an accepted convergence result. Its latest observed row (step 1272) has loss `4.36610`, gradient norm `4.01200`, clip coefficient `0.24925`, trajectory support `297`, objective-term support `13`, and zero skipped batches. This is a heavy-clipping warning, not evidence of collapse by itself.

Review of the [AAAI ORPHEUS paper](https://cdn.aaai.org/ocs/10371/10371-46146-1-PB.pdf) and [ToM-inspired simulation framework](https://arxiv.org/pdf/1405.5048) led to ADR-118. The next accuracy experiment should use a compact hypothesis bank with ordered short-step rollouts and innovation/error-based selection or calibrated blending. This is deliberately scheduled after the unchanged campaign so its executable fingerprint and selector evidence remain valid.

At 22:11 BST the same process had advanced to optimizer step 1280. The newest train row is finite (`loss_total=4.05008`, gradient norm `0.47420`, total clip coefficient `1.0`, trajectory support `321`, objective support `13`, skipped batches `0`). The post-1152 dynamics audit remains structurally passing; it reports stable RSS (`2.92 GB`) and complete scenario balance, but warns about sparse cadence and earlier severe typed-output clipping. No validation checkpoint beyond step 1024 exists yet.
## 2026-08-14 — gated event-aware hypothesis selection

The short-step hypothesis bank now supports an opt-in position gate: event and
lifecycle evidence may select a nearby candidate only when its position score
is within a configured ratio of the best position candidate. This prevents the
event-aware selector from trading a large long-horizon position regression for
collision F1. Collision indexing uses `MotionMode.COLLISION` throughout.

Protected RGB-only evaluation on eight episodes with `event_weight=0.1`,
`lifecycle_weight=0.05`, and `position_gate_ratio=0.05` produced choices
`[1152, 30, 2]` (learned, constant-velocity, ballistic). Compared with the
learned candidate, selected mean RMSE was non-regressive at every horizon and
collision F1 improved from `0.2002` to `0.2049` at 0.10 s, `0.1737` to
`0.1813` at 0.25 s, `0.1763` to `0.1837` at 0.50 s, and `0.1807` to `0.1828`
at 0.75 s (unchanged at 1.00 s). This remains evaluation-only and opt-in;
the default runtime and protected checkpoint are unchanged.

Artifact: `runs/20260814-050000-hypothesis-pool-gated-eventweight01-8ep/report.json`.

A disjoint follow-up pilot on seeds `200000–200001` completed in
`runs/20260814-080000-hypothesis-pool-gated-eventweight01-disjoint2/report.json`.
It selected `[285, 11, 0]` hypotheses. Mean selected versus learned RMSE was
non-regressive at 0.10/0.25/0.75/1.00 s for x and z, but y regressed at 1.00 s
(`0.3949` vs `0.3659` m) and collision F1 fell at 1.00 s (`0.0000` vs
`0.0556`). Because this is only two episodes and has a mixed tail result, it
is rejected as promotion evidence. A larger disjoint matrix remains required.

The evaluator now uses `torch.inference_mode()` for its RGB-only loop. This is
semantics-preserving (no gradients or parameter updates are possible) and
removes autograd version-counter overhead from repeated learned rollouts. A
one-episode CPU smoke completed successfully after the change:
`runs/20260814-083000-hypothesis-pool-inference-smoke/report.json`.

The first full disjoint eight-episode matrix after that optimization completed
at `runs/20260814-090000-hypothesis-pool-gated-eventweight01-disjoint8/report.json`.
Choices were `[1157, 27, 0]` (learned, constant-velocity, ballistic). Selected
versus learned mean RMSE (x/y/z, metres) was:

| horizon | learned | gated selected | collision F1 |
|---|---|---|---|
| 0.10 s | 0.7740 / 0.4704 / 0.8181 | 0.7740 / 0.4712 / 0.8181 | 0.1399 → 0.1411 |
| 0.25 s | 0.8043 / 0.4151 / 0.8275 | 0.8043 / 0.4149 / 0.8275 | 0.1434 → 0.1434 |
| 0.50 s | 0.8417 / 0.3010 / 0.8304 | 0.8409 / 0.3031 / 0.8295 | 0.1550 → 0.1578 |
| 0.75 s | 0.8742 / 0.2923 / 0.8140 | 0.8711 / 0.2951 / 0.8104 | 0.1718 → 0.1692 |
| 1.00 s | 0.9011 / 0.2759 / 0.8379 | 0.8880 / 0.2897 / 0.8397 | 0.1279 → 0.1065 |

Lifecycle mismatch and identity coverage were unchanged because the selected
alternatives shared the learned active masks. The 1.00-second y and event
regressions reject promotion despite x/z gains. The next selector experiment
should use axis-balanced or calibrated blending evidence rather than increasing
event weight; no default or checkpoint was changed.

An opt-in per-axis position weighting seam is now implemented across the
selector, pool, runtime, and evaluator (`--axis-weights X Y Z`), with default
`1 1 1` preserving prior behavior. A fresh two-episode y-emphasized pilot
(`1 2 1`) is recorded at
`runs/20260814-100000-hypothesis-pool-axis-y2-disjoint2/report.json`.
It improved 0.50–1.00 s y RMSE (`0.6291 → 0.6206` and `0.4682 → 0.4460` m)
and x, but z regressed slightly and collision F1 fell at 0.50 s
(`0.4763 → 0.3896`). This underpowered mixed result is rejected; no weighting
is enabled by default.

An opt-in per-axis guard (`--axis-gate-ratio`) now requires every candidate axis
error to remain within tolerance of the best candidate on that axis, preventing
scalar error trade-offs. Focused coverage passes. The fresh two-episode pilot
at `runs/20260814-110000-hypothesis-pool-axisgate-disjoint2/report.json`
selected `[293, 1, 2]` hypotheses. Mean selected versus learned RMSE was nearly
identical, with small x/z improvements at 0.10 s and collision F1 `0.2614 →
0.2649`, but y regressed at 0.25 s (`0.3352 → 0.3370` m). This underpowered
result is not promoted; a larger disjoint matrix is required.

The larger eight-episode per-axis-gated matrix completed at
`runs/20260814-120000-hypothesis-pool-axisgate-disjoint8/report.json` with
selection counts `[1141, 40, 3]`. Selected versus learned mean RMSE (x/y/z m)
and collision F1 were:

| horizon | learned | axis-gated selected | collision F1 |
|---|---|---|---|
| 0.10 s | 1.2151 / 0.5794 / 0.9925 | 1.2151 / 0.5783 / 0.9926 | 0.1185 → 0.1151 |
| 0.25 s | 1.2239 / 0.5219 / 0.9917 | 1.2235 / 0.5209 / 0.9918 | 0.1421 → 0.1442 |
| 0.50 s | 1.2271 / 0.3906 / 0.9917 | 1.2269 / 0.3895 / 0.9914 | 0.1434 → 0.1419 |
| 0.75 s | 1.2179 / 0.3821 / 1.0006 | 1.2138 / 0.3658 / 1.0011 | 0.1447 → 0.1450 |
| 1.00 s | 1.2208 / 0.3548 / 1.0237 | 1.2131 / 0.3473 / 1.0250 | 0.2238 → 0.2253 |

Lifecycle mismatch and identity coverage were identical to learned-only. The
small z regressions at 0.50–1.00 s and F1 regression at 0.10/0.50 s fail the
strict promotion gate, so the axis-gated selector remains opt-in.

The evaluator also supports opt-in posterior position blending via
`--blend-positions`: candidate positions are averaged with selector posterior
weights while lifecycle/event metrics remain tied to the selected hypothesis.
The two-episode pilot at
`runs/20260814-130000-hypothesis-pool-blend-disjoint2/report.json` selected
`[281, 9, 6]`. Blending improved x and some event F1, but z worsened at every
horizon and y worsened at 0.50/0.75 s. It is rejected for promotion and remains
an evaluation-only seam for future calibrated blending.

The blend uncertainty report is now truthful: mixture variance is propagated as
the posterior-weighted sum of within-hypothesis variance and between-hypothesis
mean spread, rather than reusing the selected candidate's variance. A real
one-episode CPU smoke completed at
`runs/20260814-140000-hypothesis-pool-blend-uncertainty-smoke/report.json` with
finite uncertainty values at every horizon.

Hypothesis posterior temperature is now configurable via `--temperature` (the
default remains `1.0`). A sharp-temperature blend pilot at `temperature=0.25`
completed at
`runs/20260814-150000-hypothesis-pool-blend-temp025-disjoint2/report.json`.
It selected `[285, 10, 1]`; x/y RMSE improved from 0.50 s onward and z was
stable, but collision F1 regressed at 0.50 s (`0.1494 → 0.1172`). It is
therefore rejected under the event guardrail and remains opt-in.

Event decoding now has an explicit `--event-threshold` (default `0.5`). A
stricter threshold pilot at `0.8` completed at
`runs/20260814-170000-hypothesis-pool-eventthreshold08-disjoint2/report.json`.
It selected `[285, 7, 4]`; all pooled position axes were non-regressive and
collision F1 improved slightly at 0.10/0.50/1.00 s. This is encouraging but
only two episodes, so the threshold remains opt-in pending a larger matrix.

The required eight-episode threshold matrix completed at
`runs/20260814-180000-hypothesis-pool-eventthreshold08-disjoint8/report.json`.
It selected `[1114, 53, 17]`. Long-horizon x improved, but y/z regressed at
0.50–1.00 s and collision F1 fell at 0.10, 0.50, and 1.00 s. The two-episode
threshold gain was therefore sampling noise; `0.8` is rejected and the default
event threshold remains `0.5`.

Corrected `BallisticContactDynamics`: detected ground crossings now apply an
explicit restitution velocity jump and clamp the contact point to the ground
surface. Previously this hypothesis emitted collision events while leaving the
post-contact velocity underground/downward, making its event and state
predictions physically inconsistent. Focused tests now require positive
post-bounce velocity and a real RGB smoke completed at
`runs/20260814-190000-hypothesis-pool-ballistic-contact-smoke/report.json`.

A fresh two-episode comparison with the corrected ballistic candidate completed
at `runs/20260814-200000-hypothesis-pool-ballistic-contact-disjoint2/report.json`.
Selection counts were `[219, 26, 51]`; long-horizon x/y improved, but z
regressed at 0.50–1.00 s and event F1 was mixed. The candidate-quality fix is
kept, but this selector result is rejected for promotion.

To isolate event coupling, a position-only sharp blend (`event_weight=0.0`,
`temperature=0.25`) was evaluated on fresh seeds at
`runs/20260814-160000-hypothesis-pool-blend-temp025-positiononly-disjoint2/report.json`.
It selected `[269, 21, 6]`; long-horizon x improved, but 0.10 s collision F1
fell (`0.1331 → 0.0997`) and other axes were mixed. It is rejected as well.
Further gains require better calibrated event hypotheses, not simply removing
event evidence from the selector.

Added vectorized approaching-pair elastic impulses to the ballistic hypothesis.
The two-episode comparison at
`runs/20260814-210000-hypothesis-pool-ballistic-pair-disjoint2/report.json`
selected `[251, 7, 38]`; y and several event-F1 horizons improved, but small z
regressions remained at 0.25/0.50/1.00 s. The contact-response implementation
is retained and tested, while selector promotion remains rejected.

Full regression verification after the contact-response changes:
`PYTHONPATH=. conda run --no-capture-output -n orpheus pytest -q` →
`753 passed, 6 skipped in 177.21 s`. All skips are expected MPS-conditional
tests because this environment reported MPS unavailable for the run.

Expanded the evaluator bank with an undamped constant-velocity candidate and
named all four candidates in reports. The two-episode comparison at
`runs/20260814-220000-hypothesis-pool-velocity4-disjoint2/report.json` selected
`[208, 6, 5, 77]` (learned, undamped velocity, damped velocity, ballistic).
Undamped velocity improved x/y through 0.75 s, but z regressed substantially at
0.75/1.00 s. The expanded bank is retained for research, not promoted.

Reports now include `selection_counts_by_horizon`, keyed by forecast horizon,
alongside global counts. A one-episode smoke verified variable-candidate shape
and horizon accounting at
`runs/20260814-230000-hypothesis-pool-horizon-count-smoke/report.json`.

The first horizon-resolved four-candidate comparison completed at
`runs/20260814-240000-hypothesis-pool-velocity4-horizon2/report.json`.
Ballistic selections were concentrated at 0.10 s and reappeared at 0.75–1.00
s, while undamped velocity was selected only five times across both episodes.
This supports a horizon-conditioned prior or learned interaction trigger as
the next principled intervention; no horizon exclusion is hard-coded.

Added opt-in `--uncertainty-aware` scoring to use predictive variance in
hypothesis NLL. The fresh four-candidate pilot at
`runs/20260815-000000-hypothesis-pool-uncertainty4-disjoint2/report.json`
selected `[259, 3, 0, 34]`; it suppressed undamped velocity, but increased
ballistic long-horizon wins and regressed z and collision F1 at 0.75–1.00 s.
It is rejected for this checkpoint and remains opt-in.

Added opt-in `--horizon-decay-scale` to discount accumulated hypothesis
evidence more aggressively for longer forecasts. The pilot at scale `1.0`
completed at
`runs/20260815-010000-hypothesis-pool-horizondecay1-velocity4-disjoint2/report.json`.
It selected `[212, 14, 8, 62]`, increased ballistic long-horizon wins, and
regressed long-horizon y and collision F1 despite x gains. It is rejected; the
default persistent evidence update is unchanged.

The opposite, prior-preserving direction is also available via signed
`--horizon-decay-scale`. Scale `-0.5` completed at
`runs/20260815-020000-hypothesis-pool-horizondecay-neg05-velocity4-disjoint2/report.json`.
It selected `[230, 13, 9, 44]`; long-horizon x/y and some event F1 improved,
but z regressed across horizons and 0.10 s event F1 fell. Both decay directions
are rejected for this checkpoint; default evidence persistence remains intact.

Added opt-in `--independent-horizons`, which keeps separate persistent
hypothesis posteriors per horizon while reusing the same rollout computation.
The fresh comparison at
`runs/20260815-030000-hypothesis-pool-independent-horizons-velocity4-disjoint2/report.json`
selected `[216, 15, 3, 62]`. Long-horizon x/y improved strongly and z stayed
near learned-only, confirming cross-horizon posterior contamination, but event
F1 regressed at 0.50/1.00 s. It remains opt-in pending event-calibrated
posteriors.

Added opt-in `--event-gate-ratio` to prevent candidates with materially worse
collision loss from winning. Combined with independent horizon posteriors, the
fresh pilot at
`runs/20260815-040000-hypothesis-pool-independent-eventgate-velocity4-disjoint2/report.json`
selected `[209, 36, 5, 46]`. It improved short-horizon event F1 but collapsed
event performance at 0.75–1.00 s and worsened y/z, so it is rejected. Event
logit calibration/training is required rather than a hard event gate.

An additional precision-oriented threshold pilot at `--event-threshold 0.95`
completed at
`runs/20260815-050000-hypothesis-pool-eventthreshold095-disjoint2/report.json`.
It raised event precision/F1 (0.50 s F1 `0.262`) and improved z, but y
regressed at 0.75/1.00 s. It is rejected under all-axis guardrails; event
calibration cannot be solved safely by threshold alone.

Evaluator reports now include per-horizon, per-candidate
`event_probability_histograms` (ten bins over [0,1]). This permits offline
precision/recall threshold sweeps from one expensive rollout. A real smoke
verified finite histograms whose bin totals match observed object counts at
`runs/20260815-060000-hypothesis-pool-event-histogram-smoke/report.json`.

Histograms are now label-aware: reports include positive-target and
negative-target probability bins in addition to all-event bins. The one-episode
smoke at
`runs/20260815-070000-hypothesis-pool-labeled-event-histogram-smoke/report.json`
validated `all = positive + negative` for every horizon and candidate, enabling
truthful offline threshold sweeps.

Added `scripts/sweep_event_histograms.py`, which aggregates these bins across
episodes and reports conservative bin-aligned precision/recall/F1 estimates
without rerunning rollouts. On the labeled smoke it reproduced the recorded
ballistic-contact event counts (0.50 s F1 `0.154` at the best available
threshold). This remains a calibration diagnostic; no threshold or candidate
has been promoted.

A bounded two-episode calibration using the attention checkpoint completed at
`runs/20260815-081500-hypothesis-pool-labeled-calibration-2ep`. Offline sweeps
favoured the learned event model at 0.10--0.75 s (best F1 `0.316/0.290/0.267/0.235`)
and ballistic contact at 1.00 s (F1 `0.381`). The best bin-aligned thresholds
were horizon-dependent (`0.10--0.20`). This is directional evidence only: the
sample is too small for promotion, and event reporting does not change
position selection. An earlier eight-episode attempt was stopped after it
exceeded the practical CPU runtime without producing a partial report.

The compatible `reference_rollout.pt` calibration then completed across eight
episodes at
`runs/20260815-083000-hypothesis-pool-labeled-calibration-8ep`. The learned
event candidate had the best offline F1 at every horizon (`0.149/0.147/0.138/
0.153/0.222` for 0.10--1.00 s); ballistic contact did not generalize beyond
the earlier two-episode draw. The best supported threshold is approximately
`0.10` (0.25 s is degenerate at this histogram resolution). Since this only
changes event reporting and does not improve position selection, the runtime
default remains unchanged.

Added an opt-in axis-independent delayed-evidence selector. The unrestricted
three-axis pilot at
`runs/20260815-091000-hypothesis-pool-axis-independent-8ep` improved x/y but
introduced small z regressions, so it was rejected. Restricting independent
selection to x/y with `--axis-independent --axis-independent-axes 0 1` passed
the full eight-episode guardrail at
`runs/20260815-093000-hypothesis-pool-axis-independent-xy-8ep`: mean x/y RMSE
improved at every horizon (`1.0132/0.4696` to `1.0111/0.4677` at 0.10 s and
`1.2850/0.2646` to `1.1804/0.2425` at 1.00 s), z was exactly unchanged, and
lifecycle, identity, and event metrics were unchanged. This is accepted as
an opt-in positional improvement; the default selector remains joint until a
config-level rollout is explicitly qualified.

Fresh-draw qualification used seeds 100--103 with identical reference
checkpoint/config pairs. The x/y-only selector artifact is
`runs/20260815-100000-hypothesis-pool-axis-independent-xy-fresh4ep`, compared
with joint baseline
`runs/20260815-100500-hypothesis-pool-baseline-fresh4ep`. Mean RMSE improved
from `0.36497/0.38407` to `0.36189/0.38193` (x/y) at 0.10 s and from
`0.69364/0.26446` to `0.45663/0.24965` at 1.00 s; z remained exactly equal at
all horizons. Lifecycle mismatch, identity coverage, and event F1 were equal
for every horizon. The improvement is therefore qualified as robust opt-in
evidence, but not yet the default runtime path.

The per-axis posterior contract is now explicit in `HypothesisSelection`:
`axis_scores`, `axis_posterior_weights(temperature=...)`, and
`axis_selected_index` are validated tensors derived from delayed position
evidence. `HypothesisDynamicsPool.selected_axis_index(...)` and
`OnlineWorldModel.selected_hypothesis_axes(...)` expose the persistent-pool
choice without replacing `WorldBelief`; joint selection remains the default.
The runtime contract is covered by the oracle integration test.

The setting is now config-controlled: `configs/attention_pilot_mps.yaml`
enables x/y composition with `hypothesis_axis_independent: true` and axes
`[0, 1]`; all other configs retain the joint default. A no-CLI-flag smoke at
`runs/20260815-101500-hypothesis-pool-axis-independent-config-smoke` recorded
the resolved setting and completed successfully. This promotes the qualified
behavior only for the attention pilot config, not globally.

A follow-up pilot attempted to inject the full joint persistent prior into
per-axis posterior weights at
`runs/20260815-103000-hypothesis-pool-axis-prior-fresh2ep`. On the same fresh
seeds, x/y returned to joint-baseline RMSE (the prior overpowered useful
coordinate-specific evidence), so the change was rejected and removed. The
accepted axis path intentionally uses current delayed per-axis position
evidence while the joint pool retains the persistent prior; a separately
calibrated per-axis prior is still required before changing that semantics.

An opt-in `--axis-prior-strength` sweep was added to measure that calibration
explicitly. Strength `0.05` at
`runs/20260815-104000-hypothesis-pool-axis-prior005-fresh2ep` still returned
the fresh baseline x/y values at every horizon, so it is rejected for now.
The default remains strength `0.0`, preserving the qualified gains.

The weak prior probe at strength `0.001` passed the full eight-episode
guardrail at `runs/20260815-110000-hypothesis-pool-axis-prior001-8ep`. Relative
to the joint eight-episode baseline, mean x/y RMSE improved at every horizon
(at 1.00 s, `1.2850/0.2646` to `1.1825/0.2456`), while z, lifecycle, identity,
event F1, and uncertainty remained non-regressive. The attention pilot config
now sets `hypothesis_axis_prior_strength: 0.001`; a no-CLI-flag config smoke at
`runs/20260815-111500-hypothesis-pool-axis-prior001-config-smoke` verified the
resolved value.

Broader verification passed the attention profile dry run and the full
repository suite: `755 passed, 6 skipped in 225.32 s`; all skips were
MPS-conditional because this environment reports MPS built but unavailable.
The dry run resolved eight scenario families, 8,192 training steps, and the
axis-composition config without starting training. A source audit shows the
axis pool is consumed through explicit evaluator/runtime accessors; there is no
hidden simulator-state path or replacement of `WorldBelief`.

A bounded attention training smoke also completed on CPU with one optimizer
update across all eight scenario families. Command:
`conda run -n orpheus python train.py --config configs/attention_pilot_mps.yaml
--device cpu --set training.steps=1 --set training.validation_episodes=8
--set training.train_episodes=8 --set training.num_workers=0
--set training.validation_minimum_supported_episodes_per_scenario=1
--set training.validation_minimum_predictable_target_count_per_scenario_horizon=1
--set training.validation_minimum_matched_target_count_per_scenario_horizon=1`.
The run completed one update with finite loss `4.048636`, causal trajectory
support `453`, zero skipped batches, and validated best rollout loss `0.386337`
(position RMSE `0.400537`). Checkpoints exist at
`runs/20260814-050546-orpheus-attention-pilot-mps/checkpoints/last.pt` and
`best_rollout.pt`. This is an entry-point smoke, not a convergence claim.

The one-step `last.pt` was checked through the RGB hypothesis evaluator at
`runs/20260815-113000-hypothesis-pool-one-step-checkpoint-smoke`. It remained
finite, but on seed 100 its x RMSE worsened at every horizon versus the
protected reference while y/z improved; it is therefore rejected as a model
promotion and retained only as a smoke artifact.

Generalization on independent seeds 200--201 also passed. Promoted config
artifact: `runs/20260815-112000-hypothesis-pool-axis-prior001-newdraw2ep`;
matched joint baseline: `runs/20260815-112500-hypothesis-pool-joint-newdraw2ep`.
At 1.00 s, mean x/y RMSE improved `1.0357/0.4030` to `0.8787/0.3736`; z
stayed exactly `0.9204`, event F1 stayed `0.1811`, and lifecycle/identity
counts matched. The evaluator now accepts `--no-axis-independent` for explicit
joint-baseline reproduction.

Added `scripts/compare_hypothesis_reports.py` as a reusable guardrail checker.
It aggregates per-horizon x/y/z RMSE, lifecycle mismatch, identity coverage,
event F1, and uncertainty, and returns a nonzero exit status on regressions.
The independent-draw comparison passed with the predeclared `1e-4` uncertainty
tolerance; exact deltas are saved in `/tmp/newdraw-guardrail.json`.

After the machine restart, the required environment was rechecked directly:
`conda run --no-capture-output -n orpheus python -c "import torch; print(torch.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())"`
reported `2.10.0 True False`. The interpreter and kernel are both `x86_64`
(`/usr/local/Caskroom/miniforge/base/envs/orpheus/bin/python`, macOS 26.5.2),
so this host still cannot execute Apple MPS kernels despite the PyTorch build
including MPS support. No long CPU training run was launched. A focused audit
suite passed: `191 passed in 5.33s` covering hypothesis rollout, comparator,
oracle online-loop, and config contracts. The next convergence run remains
blocked on an Apple-Silicon/MPS-capable host (or an explicitly approved
long-running CPU campaign); no checkpoint was promoted from this audit.

The `orpheus` environment was then explicitly overlaid with the local wheel
`/Users/mike/Work/pytorch/dist/torch-2.9.0a0+gitcbe1a35-cp310-cp310-macosx_26_0_x86_64.whl`.
Import verification reports `torch 2.9.0a0+gitcbe1a35` from the environment,
with MPS built but unavailable. A real tensor smoke on `device='mps'` raises
the backend's macOS-version availability error; the focused contracts suite
still passes `191 passed in 4.51s`. The stale 2.10 metadata was moved to a
recoverable `/private/tmp/orpheus-torch-metadata.*` directory, and no project
source or checkpoint was changed by the environment repair.

Correction: the preceding unavailable result was an artifact of the isolated
agent execution context, which has no connection to the Aqua/WindowServer
Metal service. Running the exact same `orpheus` environment in the active GUI
session with `launchctl asuser 501` reports `torch 2.9.0a0+gitcbe1a35`,
`mps_built=True`, and `mps_available=True`; an actual `mps:0` 4-by-4 matrix
multiplication returned `64.0`. The Mac has both Intel UHD 630 and AMD Radeon
Pro 5500M Metal devices. Use the active Aqua session for MPS training and
evaluation; do not infer accelerator availability from the agent sandbox.

The new paper-informed goal is active: improve robust RGB-only long-horizon
prediction through persistent, calibrated mental simulation rather than a
single long autoregressive forecast. Both source papers were re-read. Their
actionable common points are small ordered simulation steps, heterogeneous
model selection by prediction-vs-reality error, and outcome-range selection
to reject isolated false successes. The first implementation is an opt-in
robust ensemble scorer: it aggregates real delayed-target candidate loss over
nearby imagined rollout samples as `mean + risk_penalty * std`, reports
per-axis spread, updates only pool evidence, and preserves `WorldBelief`.
Focused unit/runtime tests passed: `192 passed in 4.20s`. It has not yet been
promoted for RGB accuracy; evaluator integration and fixed-manifest comparison
remain the next task.

The evaluator now exposes deterministic uncertainty-scaled nearby-belief
rollouts through `--ensemble-samples`, `--ensemble-position-std-scale`,
`--ensemble-velocity-std-scale`, and `--ensemble-risk-penalty`. Sample zero is
the exact current belief; additional samples are active-slot-only perturbations
derived from explicit position/velocity uncertainty and a fixed CPU generator.
The central rollout remains the reported forecast while all samples contribute
only to delayed model evidence. `tests/unit/test_hypothesis_rollout.py` and
the oracle runtime integration pass `25 passed in 1.55s`. A matched one-episode
active-Aqua MPS RGB baseline is currently running at
`runs/20260814-ensemble-baseline-1ep.json`; no ensemble result is claimed or
promoted until the paired run and report comparison complete.

The matched active-Aqua MPS robust-ensemble probe completed and is rejected.
Control: `runs/20260814-ensemble-baseline-1ep.json`; candidate:
`runs/20260814-ensemble-robust025-1ep.json`. Both use the protected reference,
seed 100000, one RGB episode, and MPS measurement path. The candidate used
three belief samples, position/velocity scales `0.1`, and risk penalty `0.25`.
It selected the learned candidate on all 148 decisions, exactly as the
control, so the robust evidence did not create a useful regime switch. The
guardrail comparison at tolerance `1e-4` failed: x/y RMSE regressed at 0.25 s
by `0.002572/0.004424`, x at 0.50 s by `0.004549`, and x at 1.00 s by
`0.000237`; lifecycle, identity, event F1, and uncertainty were unchanged or
non-regressive. Exact deltas are in `/private/tmp/ensemble-robust025-guardrail.json`.
Do not promote this setting or spend a full manifest on it. The next accuracy
target is candidate diversity/learned-transition accuracy, since a robust
selector cannot help while one candidate dominates every observed regime.

The active-Aqua MPS smoke at
`runs/20260814-074043-orpheus-attention-pilot-mps` completed one optimizer
step and its eight-scenario final validation. It used torch
`2.9.0a0+gitcbe1a35`, `measurement_device: mps`, and produced valid
`last.pt`/`best_rollout.pt` checkpoints. Observed train loss was `4.048636`;
the final validation rollout RMSE at 1.00 s was `0.338335` versus the
initial-incumbent `0.327330`, so this finite execution smoke is not a model
promotion.

The first intended 128-step attention continuation was interrupted after its
step-zero, 32-episode incumbent validation (the machine/session restart left
`runs/20260814-082711-attention-robust-transition-128` with no optimizer
update). Its durable validation selector score is `0.3213161872`, position
RMSE `0.2514598 m` (x/y/z `0.2817742/0.2019070/0.2636911`), and velocity RMSE
`1.0931900 m/s`. This is a valid frozen control, not evidence of training
progress. The trainer correctly disallows an in-place resume from numbered
checkpoints, so the selector checkpoint is preserved and a timestamped
continuation is now running at
`runs/20260814-083918-attention-robust-transition-128-continuation` through
the active Aqua launch agent `com.orpheus.attention.robust.transition128.continuation`.
It uses the supplied torch `2.9.0a0+gitcbe1a35`, MPS measurement path and CPU
closed-loop dynamics, with 128 updates, validation every 64, checkpoint every
32, and durable stdout/stderr logs in `/private/tmp`. No trained checkpoint is
yet available or promoted.

While the continuation initializes, the directly relevant regression suite was
rerun in the required environment:
`conda run --no-capture-output -n orpheus pytest tests/unit/test_hypothesis_rollout.py tests/integration/test_oracle_online_loop.py tests/unit/test_train_entrypoint.py`.
It passed `32 passed in 2.09s` on Python 3.10.20. This verifies the ensemble,
`WorldBelief` preservation, and resume-state contracts; it does not substitute
for the pending active-Aqua RGB validation.

The Aqua continuation reached optimizer step 8/128 without a failure or
support exhaustion. Its first durable training record has finite total loss
`0.4890455`, pre-clip/applied gradient norm `0.2124018/0.2124018`, and an
effective causal trajectory-support count of `349`; no global, node, force,
collision, or impulse parameter gradient clip engaged. Typed output-local
clips were active as designed (node and force backpropagation), not a global
collapse. This is a health observation only: it is one stochastic training
batch, not a validation result or promotion.
