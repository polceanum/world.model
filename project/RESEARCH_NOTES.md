# Research notes

## Hypotheses

- Persistent predict/correct state should recover more cheaply than repeated
  clip encoding.
- Structured event jumps should improve collision horizons over smooth motion.
- Residual ROI perception should approach repeated-global correction quality at
  lower latency.
- Explicit uncertainty and observability gates should reduce identity and
  parameter hallucination under occlusion.

These are hypotheses, not results. Empirical entries must identify config,
checkpoint, split/seeds, device, commands, metrics, and failure cases.

## Evidence so far

### Typed-attention stability and scaling decision

The drift-regularized constant-rate successor has now failed its first trained
fixed selector despite encouraging late matched training windows.  At step 512
its score is `0.3332533` versus protected `0.3213162`, with 105 guardrail
failures; `reference_pairs` current x is `0.732948 m` versus `0.242694 m`, and
all x horizons regress.  Complete support and a strict finite scope audit rule
out missing data, dead capacity, inherited drift, optimizer corruption, and
protected-checkpoint mutation.  This is direct evidence that heterogeneous
training-window improvement was not fixed-manifest generalization.

The 12 August source review does not rescue this candidate by suggesting a
larger Transformer.  The original architecture coupled attention capacity with
residual normalization, scheduled long training, and large balanced batches;
compute-optimal work scales examples with parameters; Llama 3 used proxy scale
experiments, curated data, warmup/cosine training, and a deliberately stable
dense architecture; V-JEPA 2 separately measured gains from curation, model
size, duration, resolution, and cooldown; and SlotFormer supports attention
over object abstractions rather than opaque pixels.  Orpheus already has the
applicable short-token dense mechanisms.  Test the isolated 384-step
warmup/8,192-step cosine schedule from the graph control next.  Depth, width,
history, and CUDA scale remain downstream hypotheses, not a response to this
rejection.

The isolated schedule experiment is now live at
`runs/20260812-155706-attention-node-drift-warmup-cosine-stage-a/`.  Its
step-zero selector and all 225 model tensors exactly reproduce the protected
control, while run provenance binds the intended schedule and clean commit.
This removes initialization and protocol drift as alternative explanations;
only trained selectors can determine whether reduced early cumulative update
magnitude repairs generalization.

The specification-1.36 residual-parsimony trajectory reached its authoritative
step-1024 selector without numerical, support, optimizer, or resource collapse,
but failed 111 deployment guardrails.  Its nearly flat scalar score conceals a
large familiar-physics regression: `reference_pairs` current x rises from
`0.242694` to `0.573947 m`, and every 0.10--1.00-second x horizon worsens.  The
fixed selector therefore resolves the prior ambiguity: the small model is not
yet capacity-limited in a way that authorizes scaling; it learned a broadly
misgeneralizing, nearly context-invariant node acceleration.  Continue with the
already smoke-qualified context-drift objective from the untouched control,
not with more steps or more parameters on the rejected trajectory.

The schedule successor's first complete updates 8--64 window supplies the
intended controlled early-learning comparison. All 64 updates apply with exact
eight-way scenario balance, 2,462 causal trajectory targets, zero skips or
uncontained failures, and stable memory. Against the constant-rate run on the
same steps and data, current position improves only `0.000547 m`, while current
velocity worsens `0.011510 m/s` and 0.25--1.00-second position differs by
`+0.000856/+0.004015/+0.005783/+0.002280 m`; lifecycle and identity improve
slightly. Treat this as a healthy low-rate warmup near-tie, not evidence of
convergence, collapse, promotion, or capacity limitation.

At the next complete boundary, structural and behavioral evidence separate
cleanly. The step-128 checkpoint audit proves all 48 attention tensors and only
their complete Adam states moved, all 177 inherited tensors and protected
incumbents stayed exact, and serialization is finite. Yet the exact matched
72--128 window is `0.014107 m` worse on current position and
`0.013738/0.013842/0.017214/0.007199/0.012912 m` worse across the five position
horizons, led by current x `+0.037286 m`; current velocity improves `0.008492
m/s`, with mixed velocity/y and adverse collision/lifecycle/identity evidence.
This rules out corruption but does not establish generalization. The scheduled
run remains immutable to selector 512, where a fixed manifest—not cumulative
training-window movement—will decide rejection or continuation.

The 12 August primary-source refresh reinforces this choice.  The original
Transformer's relevant contribution here is content-dependent multi-head
interaction, short dependency paths, residual layers, normalization, balanced
batches, and a deliberately scheduled long optimization run.  Compute-optimal
scaling evidence says data exposure must rise with parameter count rather than
enlarging an undertrained model.  Qwen3 and DeepSeek-V3 show that modern dense
or sparse LLMs still build on the same residual Transformer core; their
GQA/MLA/MoE mechanisms chiefly reduce long-context KV or activated-compute
costs.  V-JEPA 2 supports a later self-supervised video encoder and latent
prediction stage, but not replacing explicit object state or using a larger
pixel latent to hide a failed physical selector.  For Orpheus's at-most-22
typed tokens, the evidence-backed order remains: qualify drift-regularized
width-128/depth-four, retain its full data curve, then compare exact-identity
depth growth, width growth, and bounded timestamped history one axis at a time.

Specification 1.34 replaces ad hoc training-window calculations with pooled
auditor output. The live step-712--768 complete window has current x/y/z RMSE
`0.182561/0.170300/0.199884 m`, position-horizon RMSE
`0.184250/0.209727/0.265618/0.320229/0.347584 m`, current velocity RMSE
`1.151526 m/s`, identity `2/314`, current coverage90 `99.37%`, collision F1
`0.1333`, and minimum shared retention `0.6885`. All 13 causal objective terms
are present, with 2,096 trajectory targets and balanced eight-way scenario
exposure. The former step-776--800 partial tail closes at 832 with eight blocks.
It regresses against 712--768 on current/every position horizon, identity, and
uncertainty, but remains better than 648--704 on current and four of five
position horizons. Lifecycle coverage/precision, forecast coverage, event F1,
correction medians, gradient retention, and memory remain healthy; velocity is
mixed. Classify this as sampled optimizer wobble and continue unchanged to the
fixed selector, not as convergence, collapse, or model-change evidence. The
whole-run audit remains `pass`.

The current Mac rung is a 3,004,656-parameter model, including a 1,103,626-
parameter width-128/four-block dense typed-attention residual over at most 22
scene/entity/relation tokens. It already uses pre-RMSNorm, scaled dot-product
multi-head attention, and SwiGLU. Exact replay of the first collision-isolated
campaign found that its recurrent step-280 failure is not lack of capacity: a
joint `17.6842` raw gradient is localized to the normal/tangent force output
rows, leaving about `0.8573` in the rest of the interaction module. A later
force-row-only campaign showed that parameter clipping ran too late to protect
the shared stack; specification 1.28 therefore isolates typed output gradients
at every recursive invocation before they enter decoder/shared parameters.

The next scale decision is gated on a fresh repaired learning curve and broad
plateau, not training loss. Once qualified, compare data-only, depth, width,
and bounded-history rungs one at a time, increasing balanced continuously
varied episode draws with parameter count. Use fixed disjoint RGB-only
validation/test/OOD manifests and keep the accepted smaller model as a
non-regression control. Long-context efficiency techniques and MoE are
deferred because the current token set is short and neither addresses the
measured failure.

The depth comparison can now inherit the qualified four-block function
exactly. Appended pre-norm blocks zero only their attention output weight/bias
and SwiGLU output weight, making both residual branches exact identities while
allowing the output projections to learn immediately. Focused tests prove
zero-tolerance token-stream and decoded-output equality; a checkpoint missing
an inherited block tensor or changing shape-invisible head semantics is
rejected without destination mutation. The trainer persists the transform,
source, and appended indices in run metadata. This is a handoff-integrity
result, not accuracy or scale-promotion evidence.

The unaffected specification-1.31 training trajectory passes its durable
step-768 checkpoint and dynamics audits. The equal 712--768 sampled window
improves the preceding 648--704 window at current state and all five horizons,
with trusted switches `6/326 -> 2/314` and current-state coverage90
`96.08% -> 99.37%`. Because the episode batches differ, this is evidence
against immediate collapse, not generalization evidence; the fixed selector at
1024 remains the next promotion/no-scale decision.

The fresh force-isolated run remains healthy through sampled update 32. The
step-24 raw collision-row norm is `4.45588`, its row coefficient is `0.224422`,
the post-row interaction norm/coefficient are `2.36835/0.422235`, and the true
raw whole-model/final coefficient are `4.94611/0.202179`. This is a contained
outlier rather than a severe shared-gradient collapse; the update is finite and
applied, supported trajectory count is 396, frozen perception gradient is zero,
and RSS is `2,891,427,840` bytes. Step 32 independently contains a raw
collision-row norm of `3.23987`; its subsequent interaction coefficient is
`0.929705`, the update is applied, all scenarios have four sampled blocks, and
sampled trusted identity switches are zero. There is still no trained fixed
selector.

Through sampled step 72, one severe force-row warning occurs at step 64. The
joint normal/tangent row norm is `21.4665` inside a `21.5377` raw interaction
norm. After the force cap, the interaction norm is `2.01547`, so unrelated
attention gradients retain a `0.496162` stage coefficient rather than the
raw-total `0.0464303` coefficient. The next sampled block returns to force
coefficient `0.976879` and stage coefficient `0.686862`, with positive future
correction, zero sampled identity switches, and unchanged memory. This
supports the isolation mechanism but does not prove the event is harmless to
accuracy; checkpoint 128, the former 152/280 boundaries, and fixed selector
512 remain required.

The durable step-128 checkpoint confirms that the optimization experiment is
actually isolated: 177 inherited tensors have zero bitwise changes, every one
of 48 attention tensors changes, and the only 48 Adam states belong to those
attention parameters at step 128. All serialized state and linked protected
artifacts pass finite/hash checks. The sampled step-128 identity switches
match the preceding collision-isolated control on the identical seed/window;
aggregate sampled rate is `0.8608%`. This rules out scope drift, dead attention
capacity, corrupted optimizer state, and protected-reference mutation through
the checkpoint, but it does not establish held-out accuracy.

The next exact schedule landmark, step 152 on frames 7--11, confirms a large
optimizer-health improvement. Raw interaction norm/retained stage coefficient
progresses from `28.1387/0.03554` in the normalized campaign to
`7.11114/0.14308` with collision isolation and `2.46615/0.48940` with force
isolation. In the current run the force group is only `0.25152` and unclipped;
collision is locally bounded from `1.70491`, all objectives have support,
identity switches are zero, future correction is positive, and the update is
finite/applied. This validates the repair at one historical boundary but does
not replace step 280 or broad fixed validation.

Step 280 then disproves the assumption that decoder parameter-row isolation is
sufficient. Its raw force/total parameter norms are `989.7965/995.5391`; by the
time the row cap runs, shared projections and attention blocks already carry
order-one-to-ten gradients and the effective total update retains only
`0.0010045`. The campaign is stopped at durable step 256 and cannot count
toward convergence.

Specification 1.28 moves semantic isolation to the causal location: each raw
node, collision, and joint-force output invocation receives an optional
backward-only norm cap before the decoder/shared stack, followed by the existing
parameter hierarchy. Exact diagnostic replay from the same step-256 optimizer,
RNG, and data state reduces the later step-280 parameter norm to `10.8330`,
bounds the maximum shared parameter norm to `0.0851`, and leaves a `0.6979`
post-row interaction-stage coefficient. The batch remains finite, supported,
applied, and physically comparable; localized severe coefficients remain
visible. This establishes causal optimizer repair, not accuracy or
generalization. A fresh weights-only campaign must still pass selector 512 and
the declared plateau before any scale rung advances.

That fresh campaign exactly reproduces the protected step-zero selector and
passes the former step-64 force failure on identical seeds, frames, and support.
Raw whole/interaction gradient falls from `21.5377` to `2.14592`, force-row
norm from `21.4665` to `1.75123`, relation-decoder weight norm from `21.4054`
to `2.01100`, and maximum non-decoder shared norm from `0.04242` to `0.00540`.
The post-row interaction stage retains `0.62863`, all 64 updates are applied,
scenario balance/support/resource checks pass, and sampled 1-second RMSE is
fractionally better (`0.377141` versus `0.377330 m`). This is strong causal
optimizer evidence, not durable-checkpoint, selector, plateau, or
generalization evidence.

Durable step 128 proves the repaired experiment remains isolated and live: 177
inherited tensors are exact, every one of 48 attention tensors changed, only
those 48 parameters own Adam state at step 128, all state is finite, and the
protected model hash is unchanged. Across the 16 sparsely logged training
blocks, pooled 0.25--1.00-second position RMSE improves relative to the paired
force-row run, while 0.10-second RMSE is `0.00017 m` worse and trusted identity
switching is `8/694 = 1.153%` versus `6/697 = 0.861%`. The extra switches are
concentrated in the step-128 batch. This is a real fixed-selector warning but
not a causal failure diagnosis: sampled batches are heterogeneous and do not
support per-scenario guardrails. Preserve the unchanged model through the
historical step-152/280 boundaries and complete selector 512.

Update 200 invalidates that campaign before its first trained selector. On an
otherwise supported batch, uncapped impulse multiplier/additive rows reach
`830.3828/210.3096`, raw interaction norm reaches `857.1579`, shared block/
projection gradients reach `6.2401`, and the complete stage retains only
`0.001167`. This is a distinct recursive jump path, not evidence that the
3.00M rung lacks capacity. The trainer and supervisor are stopped; durable
step 128 remains the last reusable artifact.

Specification 1.30 gives the joint impulse outputs the same causal two-level
isolation as force: per-invocation output backpropagation plus accumulated
decoder-row clipping. It also rejects any active causal update whose complete
interaction stage retains less than `0.1` after local isolation, before Adam
mutates state, and makes the offline auditor fail the same condition. A
non-promotable step-128--200 replay reaches the same stress seeds/window with
raw norm `7.4410`, shared maximum `0.05334`, and complete-stage retention
`0.64704`; the nine logged replay blocks have no severe/uncontained clip. The
replay trajectory is not forward-exact because earlier newly bounded updates
change the weights. It qualifies a fresh start, not accuracy, generalization,
or convergence.

The fresh specification-1.30 campaign at
`runs/20260811-042704-attention-impulse-isolated-stage-a/` from clean commit
`d38cc9b`. Its 32-episode RGB-only step-zero selector is a strict control:
all 225 tensors are bitwise equal, all 2,583 comparable non-protocol metrics
are exact, and score remains `0.3213162196`; only the expected protocol hash
changes. Sampled update 8 contains all eight scenario families, eight supported
objective terms, 349 causal trajectory targets, and a finite `0.673975`
raw/applied interaction gradient with every local/stage/global coefficient at
`1.0`. Impulse rows are ordinary on this batch and RSS is `2,935,676,928`
bytes. This establishes clean launch and early optimizer health only. The
historical stress positions, fixed selector 512, repeated selectors, and the
declared plateau still gate any accuracy or capacity decision. It was later
stopped before a trained selector: attempted update 60 deterministically fell
below the new gate at `0.0850405` complete-stage retention.

The structured exact replay shows why. Every comparable logged field at
updates 8--56 is exact. Update 60 has complete support but an accumulated node
decoder norm `11.6617`, dominated by world-y `11.5014`; the largest shared
non-decoder tensor is only `0.124876`. Per-invocation node hooks protected the
shared stack but not the decoder sum across 144 calls. Specification 1.31 adds
a joint accumulated node cap and durable terminal optimizer diagnostics. A
fresh protected-control replay exactly reproduces all 225 initial tensors and
2,583 selector metrics, then reaches the same update-60 seeds at `0.565343`
complete-stage retention with full support. It deliberately stops before Adam
and is optimizer-health evidence, not accuracy or convergence.

Exact capacity census for later one-axis studies:

- current/data-only: `3,004,656` total, `1,103,626` attention parameters;
- depth six at width 128: `3,530,480` total, `1,629,450` attention parameters;
- width 192/four blocks/SwiGLU 768: `4,342,896` total, `2,441,866` attention
  parameters; and
- future single-GPU width 256/six blocks/SwiGLU 1024: `8,305,648` total,
  `6,404,618` attention parameters.

These are design points, not accepted checkpoints. Modern long-context and
sparse-inference mechanisms are deferred because 22 structured tokens do not
exercise their intended bottlenecks. Data coverage and held-out physical
generalization must scale with capacity.

Primary-source review adds two useful modern qualifications. Compute-optimal
training means the current full plateau trajectory should be retained as the
data-only curve and larger candidates should receive parameter-proportional
balanced draws; larger-but-undertrained is not a valid scale result. Maximal-
update parameterization may make later width hyperparameters transferable, but
must first win as a separate matched physical-prediction control. V-JEPA 2.1's
dense masked/deep self-supervision supports a later scalable RGB representation
stage, but its distributed visual features should be distilled or cross-
attended into typed proposals while `WorldBelief` remains authoritative.

The 11 August primary-source refresh leads to the same operational conclusion.
The original Transformer couples multi-head attention with residual paths,
normalization, Adam warmup/decay, large balanced batches, and long training;
parameter count alone was never the recipe. Llama 3 keeps a stable dense
architecture and attributes most gains to curated/diverse data and scale, with
linear warmup, cosine decay, staged context increases, recovery checks, and
final annealing. Gemma 3's local/global alternation primarily controls
long-context KV memory. V-JEPA 2 improves with curated video, capacity,
duration, resolution, and cooldown in measured stages. Orpheus already has
RMS-pre-norm and SwiGLU; at no more than 22 unordered structured tokens, GQA,
RoPE, local attention, sparse experts, and flash kernels target absent
bottlenecks. The next capacity result must therefore be the declared one-axis
depth/data ladder after small-rung convergence, not an LLM-shaped rewrite.

DeepSeek-V3 and ObjectForesight sharpen that conclusion rather than changing
it. DeepSeek-V3's MLA and MoE choices make a 671B-parameter language model with
long autoregressive KV state economical; neither mechanism supplies free
physical accuracy to a dense set of at most 22 tokens. ObjectForesight instead
shows the directly relevant scaling pattern: retain explicit 3D object
trajectories and build millions of automatically curated, geometrically gated
training clips around them. For Orpheus, future compute should therefore scale
two complementary paths: a larger self-supervised RGB/video encoder that emits
typed evidence, and the explicit object/relation/history predictor that updates
`WorldBelief`. The active 3.00M rung must first establish a fixed-selector
learning curve, plateau, and held-out generalization; otherwise increasing
width or depth cannot distinguish capacity limitation from optimization or
objective limitation.

A handoff audit found an orthogonal pre-scale defect: the allowed attention
missing-key prefix also permitted a trained four-block source to seed a
six-block destination with two random blocks. Because its typed decoders are
already learned, that changes predictions at initialization. Specification
1.32 now makes allowed new modules all-or-none and preflights every key and
shape before copying. Rejected partial growth leaves the destination bitwise
unchanged. Specification 1.33 now qualifies one narrow exception: contiguous
appended depth with zero attention/SwiGLU output projections preserves the
complete learned shallow function exactly. Width and every other unsupported
growth still start neutrally from the graph control; the smaller attention
checkpoint remains the non-regression reference.

The clean specification-1.31 run is now active at
`runs/20260811-063308-attention-node-isolated-stage-a/` from commit `5b2da41`.
Its fixed 32-episode initial selector exactly preserves the preceding control:
all 225 model tensors and all 2,578 comparable non-protocol fields match at
score `0.3213162196`. Both trainer and immutable-source convergence supervisor
are running once with empty stderr. This rules out initialization and launch
drift, but it is deliberately not counted as trained accuracy, generalization,
or convergence; fixed trained selectors and the declared plateau remain the
scale gate.

The same fresh run passes the former update-60 failure boundary and persists a
fully audited step-128 checkpoint. The live auditor reports all 128 updates
applied, each scenario exactly 16 times across logged blocks, 154--471 causal
targets per block, zero skipped draws, no terminal failure or uncontained
interaction clip, and maximum RSS `2,896,859,136` bytes. Artifact audit proves
all 177 inherited tensors exact, all 48 attention tensors live, exactly those
48 Adam owners at step 128, finite state, and intact hashes. Cumulative trusted
identity rate is `1.006%`, position coverage90 is `91.01%`, every horizon has
weighted support, and position RMSE grows from `0.2958 m` at 0.1 seconds to
`0.4347 m` at 1 second. X is hardest (`0.3386 -> 0.5963 m`); y is flat
(`0.2494 -> 0.2537 m`). These are heterogeneous training-window health
diagnostics and causal repair evidence, not proof that the fixed-selector
accuracy curve has plateaued or generalized.

The repaired run also passes the historical update-152 stress position. Raw
interaction norm is `2.90517`; after local force/collision/impulse rows the
complete stage retains `0.344214`, and Adam applies with 343 causal targets and
all 13 objective terms. Every horizon is supported, uncertainty is finite,
skips remain zero, and cumulative trusted identity is 11/820 (`1.34%`). The
live audit passes all 152 applied updates with each scenario represented 19
times across logged blocks and no terminal or uncontained failure. Update 200,
280, and fixed selectors remain required.

The same fresh run clears the former catastrophic update-200 impulse boundary.
Its raw/applied gradient is `1.14436/1.0`; impulse multiplier/additive norms are
`0.18604/0.00781` rather than the old failure's `830.383/210.310`; all impulse
row coefficients are `1.0`, and the complete interaction retains `0.873850`.
The update has 339 causal targets, all 13 objective terms, every horizon,
finite uncertainty, and no skip. Cumulatively, the auditor passes all 200
updates with each scenario represented 25 times, no terminal or uncontained
failure, trusted identity 14/1,127 (`1.24%`), coverage90 `90.25%`, and bounded
RSS. This causally validates the impulse repair on the fresh trajectory; it is
not a fixed-selector accuracy result.

The fresh step-256 checkpoint independently preserves the experiment: all 177
inherited tensors are exact, all 48 attention tensors changed, exactly those
48 own finite Adam state at step 256, and every recorded hash agrees. The
fresh trajectory then clears the historically recurrent update-280 boundary.
Raw interaction is `2.86878`, versus `52.9646` in the normalized campaign and
`17.7050` after collision-only isolation. Accumulated node/force norms are
`0.76515/2.74932`; semantic rows reduce the interaction to `1.29273`; complete
retention is `0.348580`; and Adam applies. The window contains 145 causal
targets, all 13 objectives, every horizon, finite uncertainty, and no skip.
Across 280 updates, the live audit passes with each scenario represented 35
times, no terminal/uncontained failure, trusted identity 19/1,532 (`1.24%`),
coverage90 `90.27%`, and bounded memory. This closes the known optimizer
stress boundaries but does not substitute for the fixed selector at 512.

The unchanged run remains intact through an independently audited step-384
checkpoint. All inherited tensors are exact, every attention tensor is live,
and attention-only Adam ownership and step counts are complete. Across the 48
logged balanced blocks, all 384 optimizer updates apply, cumulative causal
support is 15,083 targets, trusted identity is 26/2,105 (`1.235%`), and pooled
coverage90 is `90.34%`. Weighted position RMSE is `0.2926/0.3206/0.3651/
0.4123/0.4515 m` from 0.1 through 1.0 seconds. The x axis remains the principal
training-window error (`0.6174 m` at 1 second), while y is `0.2523 m` and z is
`0.4083 m`. This is evidence that the repaired optimizer remains stable and
causally supported, not evidence that accuracy has improved: fixed-manifest
step-512 validation, repeated selectors, plateau, and held-out tests remain
the gates for convergence and any scale decision.

The first held-out learning result is negative. At step 512 the full fixed
32-episode RGB-only selector has complete eight-scenario support but rejects
the candidate at score `0.330772` versus the exact step-zero control's
`0.321316`. Pooled current position RMSE worsens `0.251460 -> 0.295016 m`,
target coverage `0.37625 -> 0.34775`, prediction precision
`0.357312 -> 0.329465`, and current x/z RMSE
`0.281775/0.263691 -> 0.362714/0.304134 m`. Reference pairs and impulse
perturbations dominate the 131 fixed-reference guardrail failures, with
additional camera, baseline, and glancing regressions. The model remains
calibrated in aggregate (`93.28%` position coverage90), so increasing variance
would not repair the principal point/state error.

This is not optimizer or checkpoint collapse. The independent artifact audit
proves exact inherited tensors, all attention tensors live, complete
attention-only Adam state at step 512, finite serialization, and intact hashes;
the dynamics audit proves all 512 balanced updates apply without support or
retention failure. Since the inherited graph model is unchanged and the
attention branch began at exact zero output, the learned typed residual is
causally responsible for the regression. Its node decoder is strongly
anisotropic (row norms x/y/z `0.0116/0.1154/0.0150`) while the relation decoder
is largest on collision (`0.1745`), indicating coupled state/event damage rather
than direct growth of the failed x/z heads. Preserve the safe step-zero
incumbent and continue the rejected mutable trajectory to test repair at later
fixed selectors; one rejected candidate is neither convergence nor plateau and
does not justify scaling.

The first balanced post-rejection segment, steps 520--576, is moving in a
plausible repair direction on training samples. Against the equal step
456--512 window, weighted position RMSE improves at every horizon from
`0.2508/0.3025/0.3707/0.4452/0.4713` to
`0.2229/0.2694/0.3572/0.4052/0.4273 m`; x and z improve at every horizon and
identity changes from 4/324 to 4/346. This does not reproduce the fixed
selector protocol, and coverage90 falls `90.49% -> 89.17%` while y at 0.5
seconds worsens. Therefore it supports continued training, not acceptance.

Steps 560 and 568 also clarify the gradient contract. Raw-to-final retention
can fall below 10% when a severe typed output is deliberately isolated, but
the fail-fast gate measures the complete shared-interaction coefficient after
all semantic output/decoder-row caps. Tangent-force/z outliers are reduced to
post-row norms `2.839/1.325`, leaving shared-stage coefficients
`0.3522/0.7547`; both updates remain supported and valid. The offline auditor
agrees and reports no uncontained interaction event. Repeated occurrence may
be relevant to the learned-residual accuracy failure, but it is not itself an
optimizer correctness defect.

The immediately following steps 584--640 show why cadence samples cannot drive
selection. Against the equal scenario-balanced 520--576 window, pooled RMSE
worsens at every horizon from `0.2229/0.2694/0.3572/0.4052/0.4273` to
`0.2984/0.3475/0.4035/0.4669/0.5161 m`, and x/z worsen at every horizon.
Identity improves `1.16% -> 0.76%`, while coverage90 falls
`89.17% -> 88.10%`. The two windows have different seeds, target counts, and
difficulty despite exact scenario balance. This reverses the apparent repair
direction but does not prove continued held-out regression. It strengthens the
need for selector 1024 and rules out claiming convergence from a favourable
training prefix.

The next equal window, steps 648--704, is mixed rather than a stable reversal.
Pooled 0.50/0.75/1.00-second RMSE improves to
`0.3571/0.3416/0.3860 m`, while current/0.10/0.25-second RMSE worsens to
`0.3799/0.3823/0.4052 m`; current x/z and trusted identity worsen, but y and
long-horizon z improve. Coverage90 is flat near `96.1%`, causal support is
comparable (`2,122` versus `2,054`), and minimum complete shared-stage
retention improves to `0.9072`. The full auditor passes 704 applied updates,
88 logged blocks per scenario, zero skipped draws, no terminal failure, no
uncontained interaction clip, and bounded `2,922,790,912`-byte RSS. This is
continued healthy but heterogeneous optimization, not selector evidence or a
capacity authorization.

The deterministic CPU vertical slice and reduced MPS compatibility paths have
run. Exact long-form commands and artifacts are recorded in `project/STATUS.md`.

### Residual-parsimony attention qualification

The specification-1.36 attention-node parsimony campaign at
`runs/20260811-234157-attention-node-parsimony-stage-a/` exactly reproduces
the protected graph control at step zero and remains scope-clean through its
durable step-384 checkpoint. All 48 attention tensors are live, all 177
inherited tensors remain exact, exactly the attention tensors own complete
finite Adam state at step 384, and both protected selector artifacts still
equal the initializer. The immutable run therefore remains a clean test of
the opt-in decoder-row-energy prior rather than a resume or freezing accident.

Its sampled learning evidence is heterogeneous. The exact matched 256--312
window improves current and longer-horizon position plus x at every horizon,
but regresses short horizons, current velocity, most y horizons, collision,
and median uncertainty. The later exact 328--384 window reverses the position
direction: current and every pooled horizon regress, entirely through x
(`+0.008052` to `+0.021339 m`), while y and z improve at every horizon.
Velocity, collision F1, and lifecycle improve slightly; identity and
uncertainty remain adverse. All 384 updates nevertheless apply with complete
balanced support, bounded memory, no skip or terminal failure, and minimum
complete-interaction retention above the declared floor.

The implementation is axis-neutral: it averages the squared L2 energy of the
three world-axis node-decoder rows and maps those rows directly to bounded
world-axis acceleration. No axis-order, aggregation, or selector-contract bug
was found. The cross-axis behavior is expected from structured contacts: a y
residual changes contact timing and can therefore redirect pair impulses in x.
Decoder energy is a proxy for functional residual complexity, not a guarantee
of held-out accuracy. The first trained fixed selector at step 512 remains the
decision boundary; neither the favourable nor adverse sampled window supports
promotion, hyperparameter mutation, or model scaling.

### Accuracy-v4 closed-loop promotion

The promoted step-648 checkpoint is
`runs/accuracy-closed-structured-v4/checkpoints/best_rollout.pt` (SHA-256
`9b943f60128a2bd15298847d8c7de4dd3166646f3644720a3149155e57d85bcd`).
It continues the selected step-584 perception state for 64 causal closed-loop
RGB updates. Full validation selected rollout-position loss `0.0119829765`.

The fair ROI-local confirmation comparison on seeds `100064–100095` was:

| Metric | Step 584 | Step 648 |
| --- | ---: | ---: |
| current position MAE/RMSE (m) | 0.083808 / 0.109239 | 0.083282 / 0.109426 |
| velocity RMSE (m/s) | 0.730034 | 0.731623 |
| 0.1 s forecast RMSE (m) | 0.134093 | 0.132424 |
| 0.25 s forecast RMSE (m) | 0.174492 | 0.171900 |
| 0.5 s forecast RMSE (m) | 0.231253 | 0.226994 |
| perturbation recovery | 0.482786 | 0.478172 |
| collision F1 | 0.594203 | 0.608059 |
| 90% forecast coverage | 0.868147 | 0.867599 |

Promotion is based on forecast improvements at every horizon on both selection
and confirmation, plus the confirmation F1 gain. The tiny current RMSE,
velocity, recovery, and coverage regressions are real and are not averaged
away.

After freezing model choices, the final standard-test block
`200064–200095` measured:

- current position MAE/RMSE `0.089336 / 0.116908 m`;
- velocity RMSE `0.792257 m/s`;
- 0.1/0.25/0.5-second forecast RMSE
  `0.138279 / 0.177703 / 0.232862 m`;
- collision-conditioned improvement over constant velocity
  `30.97% / 54.38% / 50.66%`;
- perturbation recovery `45.30%`, positive on `97.92%` of horizons;
- collision precision/recall/F1 `0.765217 / 0.550000 / 0.640000`;
- 100% distance-gated detection, zero ID switches, zero dropped/non-finite
  forecasts, and nominal-90% coverage `86.95%`.

The report is
`runs/accuracy-closed-structured-v4/evaluation/final-test32/report.md`.
Step 648 improves the prior step-584 frozen test on position, velocity, every
forecast horizon, collision precision/recall/F1, false-positive rate, and
coverage, while perturbation recovery decreases slightly
`45.72% → 45.30%`.

An exhaustive validation threshold sweep found collision probabilities
saturated near `0.018` and `0.998`; no threshold improved F1. The `0.5`
threshold remains because the remaining mistakes are state/timing structural,
not ranking errors. Metric-scale probes were also negative: mean-radius
analytic depth had about `0.795 m` error versus `0.148 m` for learned depth,
and a photometric-radius estimate failed confirmation. A two-frame anisotropic
velocity slope remains only a future opportunity.

### Historical accuracy-v3 structured RGB candidate

The optional synthetic-disc centre extractor consumes RGB only. It subtracts a
row-median background estimate, labels foreground components, splits touching
discs at distance-transform peaks, computes weighted pixel centroids, and
Hungarian-aligns them to learned proposals. Structured centres are applied as a
straight-through forward refinement; the raw learned centre is now retained for
an explicit auxiliary smooth-L1 loss. Normal sphere profiles use foreground
threshold `0.04`; noise-heavy `toy_hard` and `cloud_single_gpu` use `0.08`.

The step-584 candidate at
`runs/accuracy-depth-finetune-v1/checkpoints/best_measurement.pt` is a controlled
512-update measurement continuation from the established step-72 weights. On a
one-time 32-episode confirmation manifest, seeds `100064–100095`, it compared
with the paired step-72 checkpoint as follows:

| Metric | Step 72 | Step 584 candidate |
| --- | ---: | ---: |
| current position MAE (m) | 0.098357 | 0.085103 |
| current position RMSE (m) | 0.131311 | 0.110556 |
| current velocity RMSE (m/s) | 0.765381 | 0.730581 |
| 0.1 s forecast RMSE (m) | 0.152992 | 0.134886 |
| 0.25 s forecast RMSE (m) | 0.189531 | 0.175246 |
| 0.5 s forecast RMSE (m) | 0.241308 | 0.231256 |
| perturbation recovery fraction | 0.422657 | 0.482774 |
| collision F1 | 0.568182 | 0.622222 |
| distance-gated detection recall | 0.998047 | 1.000000 |
| distance-gated ID switches | 0 | 0 |
| 90% forecast coverage | 0.894737 | 0.864857 |

The exact candidate and baseline reports are
`runs/accuracy-structured-peak-v2/depth-finetune-best-confirm32/report.md` and
`runs/accuracy-structured-peak-v2/baseline-step72-confirm32/report.md`.
The candidate improves every listed point/trajectory metric and collision F1,
but worsens nominal uncertainty coverage.

These reports predate the restriction of fast structured refinement to
projected ROIs: both global and fast updates called the full-frame extractor.
They remain useful paired evidence for the checkpoint, not final source-state
metrics.

The finalized ROI-local implementation was first screened on the reused
selection seeds `100032–100063`. Relative to the same checkpoint with
full-frame ordinary refinement, position RMSE improved
`0.128560 -> 0.127250 m`, velocity RMSE
`0.789148 -> 0.780543 m/s`, and the three forecast RMSE values became
`0.150932 / 0.190620 / 0.248704 m`; collision F1 declined by `0.0166` to
`0.588235`. This passed the declared no-regression gate and no later choice was
made from the test split.

The final frozen run used standard-test seeds `200064–200095`:

- current position MAE/RMSE `0.090847 / 0.118600 m`;
- current velocity RMSE `0.812524 m/s`;
- 0.1/0.25/0.5-second forecast RMSE
  `0.141520 / 0.181431 / 0.237585 m`;
- collision-conditioned improvement over constant velocity
  `29.47% / 53.79% / 50.05%`;
- perturbation recovery `45.72%`, positive on `97.40%` of horizons;
- collision precision/recall/F1
  `0.703390 / 0.518750 / 0.597122`;
- 100% distance-gated detection, zero ID switches, zero dropped/non-finite
  forecasts, and nominal-90% coverage `86.62%`.

The report is `runs/accuracy-roi-local-v3/final-test32/report.md`.

### Rejected accuracy experiments

A longer 1,120-step from-scratch run at
`runs/accuracy-structured-physical-v1` did not converge. On selection seeds
`100032–100063`, its best saved candidate measured current RMSE `0.422225 m`,
0.1/0.25/0.5-second RMSE `0.431482 / 0.453668 / 0.525709 m`, perturbation
recovery `18.80%`, collision F1 `0.316940`, and detection recall `66.60%`.
The report is
`runs/accuracy-structured-peak-v2/scratch-best-fresh32/report.md`.
That process had already loaded the earlier unweighted measurement objective
before the final metric loss weighting was implemented, so it is a truthful
rejected run, not a clean test of the completed training protocol.

An experimental `0.02 m` collision-hazard lookahead left physical trajectories
unchanged but reduced confirmation collision F1 from `0.622222` to `0.594406`.
It was rejected and its code/config surface removed. The negative report remains
at
`runs/accuracy-structured-peak-v2/depth-finetune-hazard-0p02-confirm32/report.md`.

A final 256-update continuation with the completed raw-centre objective also
failed promotion. No validation point beat the inherited `0.115593 m`
measurement MAE; validation degraded as high as `0.291207 m`. The run is kept
at `runs/accuracy-final-perception-v3`. The stable step-584 checkpoint remained
the accuracy-v3 selection and later initialized accuracy-v4.

### Earlier temporal RGB velocity experiment

The original one-frame RGB position-to-velocity coupling is effectively
inert at 20 Hz because its covariance is amplified by `1/dt²`. A causal
three-position least-squares history keyed by persistent object ID is now
implemented and measured explicitly.

On the fresh selection manifest, a deliberately calibrated
`1.0 (m/s)²` variance ceiling changed:

- velocity RMSE `1.369454 → 1.309964 m/s`;
- ordinary same-step velocity improvement `0.001594 → 0.025985 m/s`;
- collision F1 `0.042553 → 0.055172`;
- current position MAE `0.186991 → 0.190923 m` (worse);
- 0.25-second RMSE `0.189670 → 0.201318 m` (worse);
- perturbation recovery `20.09% → 19.26%` (worse).

History sizes three/four and variance ceilings one/two/four all showed the
same tradeoff. Therefore temporal velocity remains opt-in, and its default
uncertainty propagation has no empirical ceiling.

A 22-step frozen-global continuation completed in `183.15 s` and selected
step 94 by the tiny trainer-validation loss (`0.249018`). On the larger fresh
manifest with temporal evidence enabled it raised collision F1 to `0.121622`
and reduced velocity RMSE to `1.277519 m/s`, but position MAE rose to
`0.196397 m`, 0.5-second RMSE to `0.184454 m`, and perturbation recovery fell
to `11.84%`. The checkpoint is retained as a truthful negative result at
`runs/temporal-continuation-94`, not promoted.

### Interpretation

- Direct RGB geometry was the dominant accuracy lever in this toy world; the
  step-584 candidate established accurate ROI-local state, and the step-648
  continuation then improved final-test position, velocity, forecasts, and
  collision skill.
- Event-window semantics and missing-edge pooling are correct, but collision F1
  `0.640000` remains below the recommended `0.75` gate. Saturated logits make
  threshold tuning ineffective.
- Final forecast coverage `0.869518` is below the nominal 90% target and needs
  calibration without sacrificing point accuracy.
- Temporal position slopes contain useful velocity information, especially
  for high-error/collision frames, but the current diagonal confidence/update
  rule injects enough correlated error to harm the primary trajectory metrics.
- Drag/restitution updates execute under explicit observability gates but
  remain numerically negligible; useful online identification is unproven.
- The ROI-local structured fast path and raw learned-centre auxiliary objective
  are implemented and tested, but no result establishes the recommended
  collision F1, full occlusion recovery, parameter convergence, or the full
  3,000-step MPS schedule.
