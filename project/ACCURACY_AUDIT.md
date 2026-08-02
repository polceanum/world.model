# Accuracy audit

**Evidence cut-off:** 2026-08-02
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

The justified next decision is a medium supported-gradient qualification of
the audited v3 protocol, followed by one sustained, end-to-end shared campaign
only if support, coverage, gradient, and broad validation trends remain
healthy. The experimental intervention heads remain disabled. No existing v1,
v2, or v3-smoke artifact establishes broad scaled convergence.

## 2026-08-02 supported-causal convergence audit

The v2 campaign
`runs/20260801-232229-scaled-sustained-v2/` was stopped and preserved at logged
step `9576`. It remained finite, but it was not optimizing the declared causal
problem reliably:

- 121 of 173 logged causal rows (`69.94%`) had an exactly zero pre-clip
  gradient while consuming scheduled updates;
- distance-gated current target coverage fell from `0.287539` at the imported
  reference to `0.044805` at the measurement handoff;
- one-second forecast target coverage fell from `0.761458` to `0.052734`;
- low conditional position RMSE was therefore measured on a small surviving
  subset rather than demonstrating accurate persistent prediction.

The implementation audit found multiple root causes rather than one learning-
rate problem. Causal batches could optimize global discovery with no
trajectory support; inactive factory queries supplied a constant existence
loss; fast ROI confidence had no empty-crop negatives; unsupported crops
trained attributes and exact geometry; fast false positives were absent from
selector precision; global/fast direct losses had support-dependent relative
weight; and unsupported event/physical terms appeared as zero examples.
Checkpoint viability and rollback also did not make support collapse distinct
from an ordinary broad-score rejection.

Contact and identification audits found further target inconsistency. Analytic
contacts differed from the simulator in resolution order, iteration count,
corner sequencing, friction, inverse-mass position correction, restitution,
and event accumulation. Drag and restitution supervision could be derived
without two accepted observations, across an unclean interval, from the wrong
boundary object, or from an unidentifiable member-specific pair coefficient.
The repaired randomized three-sphere differential has maximum disagreement
`5.96e-08 m` in position and `1.79e-07 m/s` in velocity.

After these repairs, an exact hard-window decomposition exposed the largest
remaining optimization instability. The learned interaction block contributed
norm `85.7563` of raw whole-model norm `85.8882`. A single global clip to `2.0`
would scale all unrelated gradients by about `0.0233`. The v3 protocol first
clips that recursive learned subsystem to `1.0`, leaving pre-global norm
`4.8616`, then applies the whole-model `2.0` clip. Both stages and the original
raw norms are logged.

The first hierarchical-clipping four-update hybrid smoke,
`runs/20260802-110951-convergence-v3-hierarchical-clip-smoke/`, completed two
paired RGB and two causal updates in `213.0543 s`. The causal updates had real
trajectory/fast-slot support `115/24` and `92/32`, with no unsupported retry
and no oracle runtime input. It retained the imported safe incumbent. Its
two-episode validation and two-update measurement metrics are wiring evidence,
not an accuracy comparison.

The subsequent final-tree smoke,
`runs/20260802-121629-convergence-v3-final-audit-smoke/`, added explicit
scenario-aware selection and valid-unmapped ROI negatives. It completed in
`244.8993 s` with causal trajectory/fast-slot support `122/32` and `161/38`.
After two causal updates the fixed `reference_pairs` pooled score moved
`0.558737 → 0.548741`, but current target coverage moved
`0.350 → 0.315` and 0.1-second forecast coverage
`0.800 → 0.700`. The candidate was therefore rejected by both pooled and
scenario-slice guardrails. This is the desired truthful outcome: a small
conditional score gain is not promoted as broad improvement.

`configs/sustained_accuracy_mps_v3.yaml` now declares one shared
eight-scenario model with 8,192 paired RGB and 8,192 supported causal updates.
The next admissible evidence is a medium qualification followed by at least
four comparable corrected-protocol validations and a 64-or-more-episode fresh
balanced confirmation. A finite loss or exhausted budget alone is not
convergence.

## 2026-08-01 convergence-integrity audit

The first sustained scaled campaign is now preserved as a legacy-protocol
control and was manually superseded at logged step `9400` (latest durable
checkpoint observed during the audit: step `9344`). It was neither numerically
divergent nor broadly converged. The evidence below comes from the retained
[metrics stream](../runs/20260730-192625-scaled-sustained-e2e-v1/metrics.jsonl),
[supervisor log](../runs/20260730-192625-scaled-sustained-e2e-v1/convergence_supervisor.jsonl),
and linked validation checkpoints in that run.

On the exact 16-episode validation manifest:

| Step/candidate | Validation loss | Selection score | Position | Velocity | 1.00 s | Target coverage | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| imported step 0 | `9.9637` | `0.860012` | `1.0352` | `1.3672` | `0.9686` | `0.3685` | `0.1664` |
| measurement handoff, step 8192 | `9.8397` | `0.725038` | `0.9770` | `1.4278` | `0.6477` | `0.4542` | `0.2188` |
| causal step 8704 | `9.0768` | `0.855345` | `1.0202` | `1.3287` | `0.9923` | `0.4085` | `0.1748` |
| causal step 9216 | `8.5636` | `0.849660` | `1.0290` | `1.2348` | `0.9980` | `0.3791` | `0.2092` |

The step-8192 value uses the same validation objective, but it is a
measurement-phase handoff rather than a point on the causal-optimisation
trend. The full 0.10/0.25/0.50/0.75/1.00-second RMSE vectors were
`0.8851/0.7672/0.7780/0.8410/0.9686` at step 0,
`0.8679/0.7777/0.7456/0.7127/0.6477` at the handoff, and
`0.8676/0.7281/0.7416/0.8239/0.9980` at causal step 9216.

The causal path therefore learned real velocity/event structure: step 0 to
9216 improved validation loss `14.1%`, velocity `9.68%`, and collision F1
`25.7%`. It simultaneously regressed one-second position `3.03%` and
0.10/0.25/0.50/0.75-second forecast-coverage calibration from
`0.8734/0.8566/0.8123/0.7386` to `0.8485/0.8287/0.7843/0.7168`. With only two
causal validations, the declared four-point plateau test could not be
evaluated. This is real but traded-off learning, not broad convergence.

The critical defect was the phase handoff. Step 8192 changed `79/84` global RGB
tensors and improved the selection score `15.69%`, one-second RMSE `33.13%`,
target coverage `23.28%`, and collision F1 `31.45%`; velocity regressed `4.44%`
and correctly failed the 2% deployment guard. The old trainer then reloaded
the step-zero safe checkpoint. Global RGB tensors at steps 8704, 9216, and
`last.pt` were bit-identical to step zero, proving that all 8192 perception
updates were discarded from the mutable causal trajectory.

Raw loss wobble was dominated by heterogeneous perception failures, not NaNs:
across 150 causal log rows, total loss had median/mean/max
`9.134/9.595/29.908`, correlation with measurement loss was `0.9666`, and
`146/150` pre-clip gradients exceeded the unit clip threshold. The median
effective clip coefficient was `0.2043`. All 177 model tensors, Adam moments,
and optimiser steps were finite and internally consistent. Batch-one variation
therefore explains much of the visible noise, while near-universal clipping
makes each update unusually sensitive to the current batch composition.

The old profile also mismatched training support to selection. Across those
150 rows, the 0.10/0.25/0.50/0.75/1.00-second objectives had support in
`150/129/114/89/57` rows. Legacy per-window normalization turned the configured
selection weights of `10/15/20/25/30%` into mean effective shares of
`28.55/21.83/21.10/17.12/11.40%`. The 24-frame, 20 Hz episodes also supplied
only four possible one-second anchors; temporal velocity becomes observable at
about frame three, leaving three of those four anchors cold.

Some requested futures were not point-identifiable at all. In the impulse
scenario, the probability of at least one unseen intervention within one
second is `1 - 0.88^20 = 92.24%`, so an exact deterministic post-impulse target
cannot be inferred from the anchor. Mass, restitution, and drag were sampled
independently of appearance, so exact pre-contact post-collision outcomes also
cannot be inferred before an interaction makes those parameters observable.

The 1 August corrected implementation and
`configs/sustained_accuracy_mps_v2.yaml` address these confounds:

- retain a safe deployment incumbent while continuing optimisation from a
  stronger finite handoff candidate;
- use 40 frames, mature anchors, fixed global horizon denominators, and joint
  long-horizon/pair-collision sampling;
- censor deterministic coupled-scene targets after hidden actuation while
  training forecast NLL;
- use unit-correct projected association covariance, stable resting contact,
  genuinely glancing impacts, simulator/model-consistent pair restitution,
  compositional OOD ranges, and independent render/physics RNG;
- preserve pair and boundary collision evidence over complete observation
  intervals, distinguish floor support from walls/ceiling, and prevent a
  single slow contact from inventing sleep;
- select perception with lifecycle-qualified pooled MAE/recall/precision/F1,
  allocate scarce birth slots by confidence, and reset all identity-specific
  state when a slot is recycled;
- omit unsupported loss terms, calibrate state variance without duplicating
  its supervised mean gradient, and align collision-conditioned windows with
  an actually scored event endpoint;
- fix filter/lifecycle uncertainty ownership, structured-depth validity and
  edge process noise; supervise appearance and velocity correction explicitly
  while excluding frozen or structurally dead objectives;
- preserve exact absolute-step sampling with `StepIndexedBatchSampler`, CPU/MPS
  RNG, immutable process-start provenance, linked selector tensor hashes,
  exact additive counts, and structured rejection reasons;
- report axis/scenario/seed and cold/mature attribution under a hashed bounded
  trend-validation anchor protocol, with ID switches measured from an
  independent framewise association rather than the training-only locked
  target map.
- restrict runtime-qualified assignment to lifecycle-confident proposals,
  retain false positives on target-empty frames, and version the changed
  selector semantics;
- detach RGB covariance linearization coordinates from mean heads while
  keeping variance calibration trainable;
- use a tested hybrid measurement placement on this PyTorch 2.10 host
  (CNN/ROI MPS, proposal transformer CPU) after reproducing data-dependent NaN
  MPS matrix gradients, and deserialize checkpoints on CPU to preserve hybrid
  optimizer ownership;
- recover interrupted terminal validation without training, restrict
  in-place resume to the exact `last.pt`, and preserve completed no-op artifacts
  byte-for-byte.

These were correctness and experimental-integrity fixes, not evidence that v2
achieved higher held-out accuracy. The later 2 August audit above found
additional optimizer-support, coverage, fast-ROI, contact, and gradient-scaling
failures, stopped v2, and replaced it with the v3 protocol. The
three-update smoke in
`runs/20260801-231521-audit-v2-final-verified-smoke/` proves finite two-phase
execution, selection, terminal checkpointing, and exact no-op resume only.

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
