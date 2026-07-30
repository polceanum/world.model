# Accuracy audit

**Evidence cut-off:** 2026-07-30  
**Scope:** existing RGB-only checkpoints, reports, resolved configurations, and
training logs in this repository

## Executive conclusion

The evidence supports three separate conclusions that should not be collapsed
into one accuracy claim:

1. The small shared model is the only checkpoint with a completed test covering
   all eight declared scenario families. It predicts materially better than
   constant velocity at every reported horizon, but its absolute accuracy,
   collision F1, detection coverage, and identity stability are not yet good
   enough to be a final model.
2. The 1.90M-parameter scaled model has useful perception weights and a strong
   parameter-free point/scale observer, but it has not received sustained
   fixed-scale end-to-end training. Its best position result was established on
   four validation episodes, and it regresses velocity or event metrics.
3. The later lateral, gravity, and event interventions are useful diagnostics,
   not candidate shared models. They were fitted on hundreds of local windows
   and evaluated after little or no shared-model adaptation. Their offline or
   one-seed gains did not survive the full online feedback loop.

The justified next decision is therefore one sustained, end-to-end scaled
campaign initialized from the current scaled point/scale checkpoint, with all
experimental intervention heads disabled initially. Before committing to the
nominal 48,000-update run, closed-loop throughput and checkpoint selection must
be corrected so the campaign is computationally credible and cannot select a
position-only win that regresses velocity, detection, identity, events, or
calibration.

## Rules for comparing evidence

Results are directly comparable only when they share the model/runtime
semantics, simulator regime, sequence length, object-count range, horizons,
split, seed manifest, and metric implementation. In particular:

- the 48x48 small model and 64x64 scaled model are separate experimental
  families;
- a cached intervention fit is not comparable with a recursive online
  evaluation;
- a one- or two-episode screen is not broad validation;
- selection and confirmation blocks must use the same seeds on both sides;
- simulator state is acceptable for scoring and supervision, but all accepted
  runtime results below report `rgb_only=true` and
  `oracle_runtime_input_used=false`;
- results from the earlier variable-radius scaled curriculum do not measure the
  current identifiable fixed-radius curriculum.

These rules follow the protocol metadata persisted in the reports, rather than
inferring comparability from similar run names.

## Evidence ledger

| Artifact | Actual scope | Status | What it establishes |
| --- | --- | --- | --- |
| [Eight-scenario test report](../runs/20260728-091315-selected-all-scenarios-v1/evaluation/20260728-093649-final-test16-v13/report.md) | 16 reserved test episodes, exactly two from each of eight scenarios; CPU; 48x48; 2–3 objects | Accepted broad small-model evidence | The structured online architecture predicts substantially better than constant velocity across the complete small scenario mixture. |
| [Eight-scenario checkpoint](../runs/20260728-091315-selected-all-scenarios-v1/checkpoints/best_rollout.pt) | Step 672; approximately 156k parameters | Best broadly tested checkpoint | A reliable comparison baseline for the small profile, but architecturally incompatible with the 1.90M-parameter scaled profile. |
| [Fixed-scale measurement summary](../runs/20260728-223558-scaled-fixed-scale-transfer-1k-v1/train_summary.json) | 256 measurement updates, batch four, 1,024 episode draws, eight scenarios, MPS | Accepted initialization evidence | Fixed physical radius makes monocular RGB scale identifiable and improves global measurement validation. It contains no closed-loop training. |
| [Cadence-three selection](../runs/20260728-231250-scaled-global-cadence3-ablation-v1/evaluation/20260728-232212-global-cadence3-select2-offset16/report.md) and [confirmation](../runs/20260728-231250-scaled-global-cadence3-ablation-v1/evaluation/20260728-233559-global-cadence3-confirm2-offset18/report.md) | Two episodes per block, four scenarios in total, MPS | Provisionally accepted runtime policy | Re-anchoring globally every three frames is better than every six on the matched blocks, but it is not an eight-scenario acceptance result. |
| [Point/scale selection](../runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-100628-select2-offset16-final-v5/report.md) and [confirmation](../runs/20260729-084712-scaled-point-scale-trajectory-v1/evaluation/20260729-094349-confirm2-offset18-quality-v4/report.md) | Same two paired blocks as cadence three; MPS | Accepted for scaled position only | Multi-frame scale anchors greatly improve current and multistep position, while exposing velocity and event regressions. |
| [Current scaled runtime checkpoint](../runs/20260729-084712-scaled-point-scale-trajectory-v1/checkpoints/runtime_ablation.pt) | Step-zero runtime wrapper; 1.90M-parameter state; cadence three and point/scale enabled | Recommended weights-only initialization | Its model tensors are byte-identical to the selected fixed-scale step-256 measurement weights. It adds runtime semantics, not additional trained weights. |
| [Step-16 causal report](../runs/20260729-001136-scaled-fixed-cadence3-causal-valid32-v3/evaluation/20260729-005424-causal-step16-select2-offset16/report.md) | 16 closed-loop updates; two selection episodes | Rejected | The short continuation slightly improves current state but regresses the 0.25–1.0-second forecasts. |
| [Lateral intervention report](../runs/20260730-092214-rgb-lateral-intervention-regularized-v2/evaluation/20260730-094915-lateral-select2-offset16-frames48/report.md) | Local head trained on 543 windows and evaluated on two episodes | Rejected | Offline lateral correction does not improve the complete recursive online objective. |
| [On-policy gravity report](../runs/20260730-105512-rgb-gravity-intervention-on-policy-8x8-v4/evaluation/20260730-113746-gravity-select2-offset16-frames48/report.md) | One on-policy aggregation pass and two online episodes | Rejected | A locally useful gravity correction changes later association and observability enough to regress broad metrics. |

## What works, and where

### Shared structured prediction works on the complete small scenario mixture

The selected small checkpoint was evaluated on seeds `200000–200015`, with the
ordered eight-scenario mixture repeated twice. Its aggregate test result was:

| Metric | Model | Constant velocity |
| --- | ---: | ---: |
| Current position RMSE | `0.200430 m` | not applicable |
| Current velocity RMSE | `0.968753 m/s` | not applicable |
| 0.10 s position RMSE | `0.214112 m` | `0.229335 m` |
| 0.25 s position RMSE | `0.245655 m` | `0.388984 m` |
| 0.50 s position RMSE | `0.290137 m` | `0.736210 m` |
| 0.75 s position RMSE | `0.320995 m` | `1.040425 m` |
| 1.00 s position RMSE | `0.364040 m` | `1.527034 m` |

The same report records collision F1 `0.320388`, detection recall/precision
`0.762957/0.873473`, three distance-gated identity switches, and nominal-90%
forecast coverage `0.717519`. The large advantage over constant velocity,
especially at 0.5–1.0 seconds, confirms that the persistent belief and
structured dynamics are useful. The remaining metrics prevent calling this
high accuracy.

Per-scenario two-episode reports identify distinct failure contexts:

| Scenario | Current position RMSE | Current velocity RMSE | 1.00 s RMSE | Detection recall | Collision F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference pairs | `0.225914` | `1.022508` | `0.514741` | `0.775000` | `0.600000` |
| baseline | `0.179120` | `1.057606` | `0.271469` | `0.850000` | `0.327869` |
| elastic pairs | `0.162413` | `0.733903` | `0.314357` | `0.737500` | `0.518519` |
| damped contacts | `0.398835` | `0.851636` | `0.325109` | `0.512500` | `0.285714` |
| impulse perturbation | `0.198366` | `1.037854` | `0.268241` | `0.831250` | `0.256410` |
| camera parallax | `0.227364` | `1.079831` | `0.351915` | `0.781250` | `0.253968` |
| glancing impacts | `0.197196` | `1.076803` | `0.290528` | `0.850000` | `0.382979` |
| heavy/light impacts | `0.189405` | `0.979926` | `0.276434` | `0.837500` | `0.298851` |

The source reports are the eight `scenario-*-test2` reports under
[the selected run's evaluation directory](../runs/20260728-091315-selected-all-scenarios-v1/evaluation/).
Reference pairs is the weakest one-second prediction regime; damped contacts
is the weakest current-localization/detection regime. This argues against
optimizing only an aggregate mean.

The per-scenario reports are independent two-episode evaluations. Their
identity-switch counts must not be summed or substituted for the three switches
observed in the combined 16-episode sequence.

### Fixed scale and denser global anchoring work in the scaled profile

The fixed-radius transfer completed 256 measurement updates with batch four,
for 1,024 episode draws or `0.25` nominal passes through the 4,096-episode
training pool. It selected validation world-position MAE `0.380453 m`. The
supporting checkpoint is
[best_measurement.pt](../runs/20260728-223558-scaled-fixed-scale-transfer-1k-v1/checkpoints/best_measurement.pt).

Changing global discovery cadence from six frames to three improved current
state, short/mid-horizon position, detection, and collision F1 on selection
seeds `100016–100017`, and repeated most state/forecast improvements on
confirmation seeds `100018–100019`. This is strong evidence for the cadence
policy, but the two blocks cover only `reference_pairs`, `baseline`,
`elastic_pairs`, and `damped_contacts`.

### Multi-frame point/scale history works for position, not yet across the board

The following comparisons use identical checkpoints, episode manifests, and
horizons within each seed block. The only intended change is the point/scale
trajectory observer.

| Block and policy | Current position | Velocity | 0.10 / 0.25 / 0.50 / 0.75 / 1.00 s RMSE | F1 | Detection R/P | Coverage 90 |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| offset 16 cadence-three baseline | `0.906217` | `1.082334` | `0.780533 / 0.626639 / 0.650438 / 0.773491 / 1.007125` | `0.357143` | `0.694444 / 0.568182` | `0.737805` |
| offset 16 point/scale | `0.684258` | `1.148529` | `0.698125 / 0.526197 / 0.517001 / 0.562448 / 0.618672` | `0.271186` | `0.687500 / 0.480583` | `0.780204` |
| offset 18 cadence-three baseline | `1.165912` | `0.889775` | `1.010213 / 0.877051 / 0.922522 / 0.988420 / 1.267293` | `0.190476` | `0.391667 / 0.345588` | `0.554667` |
| offset 18 point/scale | `0.804367` | `0.986646` | `0.802223 / 0.630088 / 0.644967 / 0.693634 / 0.760509` | `0.078431` | `0.675000 / 0.455056` | `0.722522` |

Every position horizon improves substantially on both blocks, as do
confirmation detection and calibration. Velocity regresses `6.1%` and `10.9%`,
and collision F1 regresses on both blocks. The correct interpretation is a
confirmed position observer with an unresolved state/event coupling, not an
overall accuracy promotion.

## What failed, and why the failures do not justify rapid discard

### More pretraining on the earlier ambiguous curriculum was mixed

The earlier variable-radius scaled continuation reached step 896 and improved
its internal measurement validation MAE from `0.645048` to `0.614574 m`.
Against step 256 on identical seeds `100016–100017`, current position improved
`18.6%` and 0.10–0.75-second forecasts improved, but velocity RMSE worsened
`94.1%` and the one-second endpoint worsened `5.0%`. See the
[step-256 report](../runs/20260728-131727-scaled-curriculum-1k-v1/evaluation/20260728-173848-scaled-step256-paired-confirm2-offset16/report.md)
and
[step-896 report](../runs/20260728-152237-scaled-longer-stable-v2/evaluation/20260728-174523-scaled-step896-paired-confirm2-offset16/report.md).

Those weights are not a rejected proof that long training cannot work. The
curriculum varied physical radius while monocular back-projection assumed a
mean radius, so radius and depth were not identifiable. The current scaled
curriculum fixes radius at `0.21 m`.

### The fixed-scale shared model has barely received causal training

On the offset-16 selection block:

| Weights | Causal updates | Current position | Velocity | 0.10 / 0.25 / 0.50 / 0.75 / 1.00 s RMSE |
| --- | ---: | ---: | ---: | --- |
| cadence-three baseline | `0` | `0.906217` | `1.082334` | `0.780533 / 0.626639 / 0.650438 / 0.773491 / 1.007125` |
| sampler-corrected step 8 | `8` | `0.906162` | `1.081974` | `0.780467 / 0.626616 / 0.650370 / 0.773388 / 1.007019` |
| sampler-corrected step 16 | `16` | `0.906124` | `1.081620` | `0.780446 / 0.627064 / 0.652029 / 0.776092 / 1.008927` |

The step-8 report is
[here](../runs/20260728-235003-scaled-fixed-cadence3-causal-valid-v2/evaluation/20260729-001101-causal-step8-select2-offset16/report.md).
Eight updates are indistinguishable from the baseline at practical precision.
The step-16 longer-horizon regression justifies retaining the baseline, but it
does not establish convergence or the value of sustained training.

### Local intervention heads improve their fit target but change future inputs

The regularized lateral head reduced held-out post-filter lateral RMSE
`0.648080 → 0.497431 m/s`. In the actual two-episode recursive loop, however,
the point/scale baseline versus lateral candidate was:

| Metric | Baseline | Lateral candidate |
| --- | ---: | ---: |
| Current position RMSE | `0.684258` | `0.682624` |
| Total velocity RMSE | `1.148529` | `1.181901` |
| x velocity RMSE | `0.568277` | `0.576981` |
| 0.10 s RMSE | `0.698125` | `0.697626` |
| 0.50 s RMSE | `0.517001` | `0.524101` |
| 1.00 s RMSE | `0.618672` | `0.635806` |
| Detection recall | `0.687500` | `0.673611` |
| Coverage 90 | `0.780204` | `0.749636` |

The on-policy gravity head likewise reduced its cached held-out residual RMSE
`2.113796 → 1.854939 m/s`, then changed the online loop as follows:

| Metric | Baseline | Gravity candidate |
| --- | ---: | ---: |
| Current position RMSE | `0.684258` | `0.669713` |
| Total velocity RMSE | `1.148529` | `1.189127` |
| y velocity RMSE | `1.902265` | `1.979012` |
| 0.10 s RMSE | `0.698125` | `0.690724` |
| 0.50 s RMSE | `0.517001` | `0.535345` |
| 1.00 s RMSE | `0.618672` | `0.626332` |
| Detection recall | `0.687500` | `0.586806` |
| Collision F1 | `0.271186` | `0.218750` |
| Identity switches | `1` | `5` |

Both rejections are correct for runtime promotion. They also demonstrate that
per-update residual fitting, even with one dataset-aggregation pass, is not a
substitute for training through future association, ROI scheduling, identity,
and recursive rollout losses.

Earlier change-point and outgoing-proposal artifacts show the same distinction:
the offline reports live under
[the aligned gate run](../runs/20260729-222143-rgb-change-point-mlp-aligned-precision70-v5/)
and
[the bounded outgoing-proposal run](../runs/20260730-081923-rgb-outgoing-proposal-gate-focused-bounded-v4/).
Neither produced an across-metric online promotion.

## Direct evidence of undertraining

The configured scaled campaign in
[configs/scaled_curriculum.yaml](../configs/scaled_curriculum.yaml) declares:

- `4,096` training episodes and eight balanced scenario families;
- `8,000` RGB measurement updates;
- `48,000` total updates, leaving `40,000` causal updates;
- batch one and eight-step TBPTT;
- 256 validation and 256 test episodes;
- recursive horizons through one second.

What has actually been completed on the current fixed-radius curriculum is:

- 256 measurement optimizer updates with batch four, or 1,024 episode draws
  and `0.25` nominal manifest passes;
- no trained point/scale parameters—the point/scale improvement is a runtime
  estimator over the same selected measurement tensors;
- at most 16 sampler-corrected shared causal updates, whose weights were
  rejected;
- intervention fits on hundreds of extracted windows, not thousands of
  end-to-end episodes;
- paired online screening mostly on one or two episodes at a time.

The current point/scale runtime checkpoint and the selected fixed-scale
measurement checkpoint contain identical `model_state` tensors. The relevant
artifacts are:

- [current runtime checkpoint](../runs/20260729-084712-scaled-point-scale-trajectory-v1/checkpoints/runtime_ablation.pt);
- [selected fixed-scale measurement checkpoint](../runs/20260728-223558-scaled-fixed-scale-transfer-1k-v1/checkpoints/best_measurement.pt);
- [fixed-scale training summary](../runs/20260728-223558-scaled-fixed-scale-transfer-1k-v1/train_summary.json).

It is therefore accurate to say that the current scaled model is undertrained,
especially in the end-to-end causal phase.

## Why simply launching 48,000 updates is not yet credible

Existing logs show that steps 10–16 of the scaled causal continuation took
about 35 minutes on MPS, with individual two-step graphs taking roughly
6.5–10 minutes depending on scene complexity. A separate eight-episode
closed-loop validation remained compute-active for approximately 84 minutes.
The logs are persisted in
[the step-16 metrics stream](../runs/20260729-001136-scaled-fixed-cadence3-causal-valid32-v3/metrics.jsonl)
and the interrupted longer run in
[the step-896 metrics stream](../runs/20260728-152237-scaled-longer-stable-v2/metrics.jsonl).

At that measured rate, 40,000 causal updates and repeated 256-episode
validation are not a practical single-machine campaign. Reducing the requested
training duration would repeat the current problem; launching the nominal
schedule unchanged would produce an effectively non-terminating experiment.
Profiling, memory/graph reduction, and a tiered validation cadence are
prerequisites to sustained training, not reasons to return to tiny local
ablations.

## Checkpoint-selection limitation found and corrected before the campaign

Before this audit, the trainer saved `best_measurement.pt` using validation
world-position MAE and saved `best_rollout.pt` using only aggregate
`rollout_position` loss. Velocity, detection, identity, collision F1, and
calibration were evaluated only after selection. That explains the repeated
pattern of a candidate winning its fitted metric and then failing promotion;
longer training alone would not have fixed the mismatch.

The sustained campaign now pools physical SSE/counts across its exact
validation manifest. It minimizes horizon-weighted position RMSE while
guarding current position/velocity, every horizon, 0.5 m distance-gated
detection recall/precision and identity, forecast lifecycle coverage,
collision F1, and nominal-90% calibration against both the moving incumbent
and a fixed initialization reference. It retains every numbered validation
candidate and verifies protocol and tensor hashes before reusing incumbent
metadata on resume. Per-scenario fresh confirmation remains a post-training
promotion requirement.

## Justified sustained-campaign decision

The next accuracy campaign should use the following evidence-backed policy:

1. **Start from the current scaled state.** Use
   [runtime_ablation.pt](../runs/20260729-084712-scaled-point-scale-trajectory-v1/checkpoints/runtime_ablation.pt)
   through `train.py --initialize-from`. This preserves the selected 1.90M
   fixed-scale measurement tensors while resetting optimizer, scheduler, RNG,
   and step provenance for a new campaign.
2. **Keep diagnostic interventions off initially.** The lateral, gravity,
   change-point, and outgoing-proposal gates remain disabled, as in the
   [scaled configuration](../configs/scaled_curriculum.yaml). The confirmed
   cadence-three and point/scale position observers remain active.
3. **Use the bounded but complete causal path.** Every frame advances and
   supervises belief, while one earliest eligible anchor per four-frame TBPTT
   window scores every horizon supported from that timestamp. The sampler
   deliberately mixes collision-conditioned and long-horizon windows so the
   campaign covers both events and the complete declared horizon set.
   Posterior rollout tensors are reused for correction, and validation omits
   only the redundant prior future rollout.
4. **Train in sustained phases.** Run 8,192 measurement updates (two complete
   manifest passes), followed by 4,096 independent causal windows (one
   complete nominal causal pass, about 512 windows per scenario). Do not judge
   causal convergence before 2,048 updates. Extend only under the predeclared
   final-window improvement rule in `project/TRAINING.md`.
5. **Retain every validation candidate.** Persist numbered validation
   checkpoints, the moving best, and the fixed reference with exact
   protocol/seed and tensor-hash provenance. A position-only scalar winner is
   insufficient.
6. **Evaluate all eight scenarios at meaningful milestones.** Cheap smoke
   checks may catch failures, but they must not decide promotion. Selection and
   confirmation should each include balanced per-scenario slices on fixed,
   disjoint manifests; the reserved test split remains untouched until a
   candidate passes both.
7. **Require broad non-regression.** Promotion must consider aggregate and
   per-axis current state, every 0.1–1.0-second horizon, collision-conditioned
   forecasts, collision F1, detection recall/precision, identity switches, and
   uncertainty coverage. Any permitted tolerance should be declared before
   looking at candidate results.

This decision does not claim that duration alone guarantees convergence. It
does establish that the shared scaled model has not yet been trained long
enough to answer that question, and that the next experiment must optimize the
whole online system rather than another isolated residual target.
