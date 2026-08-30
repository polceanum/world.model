# Architecture

```text
timestamped ObservationPacket
          │
          ▼
registered RGB / RGB-D / debug-oracle observation module
          │  MeasurementSet + covariance
          ▼
projection ─ association ─ innovation
          │
          ▼
analytic + gated learned correction
          │
          ├── bounded per-ID sensor history
          │       └── optional velocity-only correction
          ├── lifecycle / identity
          │       └── detached tentative evidence before permanent birth
          └── observability-gated slow parameter update
          │
          ▼
persistent WorldBelief
          │
          ├── derived predictive-abstraction router
          │       ├── point trajectory for free motion
          │       └── rigid sphere for contact execution
          │
          ├── reversible typed belief tokens
          │       └── scene / camera / kinematic / programme / lifecycle
          ├── optional typed attention residual
          │       ├── dense pre-RMSNorm + SwiGLU set processing
          │       └── bounded force / event / uncertainty proposals
          │
          ▼
analytic kinematics + stable modes + interactions + event jumps + uncertainty
          │
          ▼
prior WorldBelief / arbitrary-time BeliefTrajectory
```

Dependency direction is `utils → typed belief/observation contracts →
dynamics/filter/fusion/identification → runtime → training/evaluation/demo`.
Simulator labels are never imported as ordinary RGB runtime measurements.

RGB discovery has an optional structured image prior for the synthetic disc
world. It uses RGB pixels only:

1. estimate the static floor/background with a per-row RGB median;
2. threshold foreground colour residuals and label connected components;
3. split touching silhouettes into nearest distance-transform peak basins;
4. compute photometrically weighted component centres; and
5. use Hungarian assignment to align those centres with learned proposal slots.

Global discovery applies the resulting centre as a detached straight-through
residual, while preserving the detector's unrefined `raw_centre` for explicit
auxiliary supervision. The structured centre carries a pixel-calibrated
measurement variance and is backprojected through the same camera calibration
as every other RGB measurement. It neither changes `WorldBelief` fields nor
introduces simulator data into runtime. The dataclass default is off for future
worlds/modalities; the current sphere profiles opt in. Ordinary profiles use a
`0.04` foreground threshold, while noisy `toy_hard` and
`cloud_single_gpu` use `0.08`.

The fast path remains prior-conditioned: projected object ROIs, persistent
object IDs, and cached object features feed the residual ROI updater. Structured
centre refinement samples those ROIs in one batched `grid_sample`, estimates
local background from each crop perimeter, seeds the nearest supported
foreground component to the prior centre, and grows only that component with
tensor operations. It never invokes the full-frame SciPy discovery/assignment
routine. A missing, invalid, too-small, or out-of-gate local component falls
back to the learned/predicted measurement, so this operation cannot globally
reacquire an object. Depth residuals remain disabled in the current profiles
until a trained ROI checkpoint passes a held-out correction gate.

Because each fast ROI is already conditioned on a source prior, its
`MeasurementSet` records the source belief slot and persistent object ID. Core
association may accept that row only for the same source if the normal
uncertainty/confidence gates pass; it may not cross-update another object.
Global discovery retains free gated Hungarian matching. Every maximum-cost or
distance gate is applied before assignment with valid cardinality taking
precedence over residual cost, avoiding post-hoc invalid pairs that create
unnecessary misses and rebirths.

Unmatched global/recovery discoveries use bounded tentative evidence keyed by
`(modality, sensor_id)`. This state is detached observation history, not a
second world model: it has no permanent ID and does not enter correction,
dynamics, rollouts, or parameter identification. Strictly later detections
must remain within the configured world-distance gate for the configured
number of consecutive confirmations; only then does lifecycle allocate a
monotonic ID in `WorldBelief`.

Persistent object `motion_mode_logits` describe the instantaneous endpoint
state. A rollout step's collision logit instead describes occurrence anywhere
inside that prediction segment; internal physics substeps are aggregated.
Frame-labelled training/evaluation brackets each target with
`[h-dt_obs, h]`, so event probabilities have one explicit temporal meaning.
Substep selection treats a timestamp ratio that is indistinguishable from an
integer at the belief dtype as that integer, keeping nominal 20 Hz float32
observations aligned with the simulator's six 120 Hz ticks. Genuine fractional
intervals still ceil.

When closed-loop training needs the same next prior for direct supervision and
runtime correction, `OnlineWorldModel` returns a typed one-use prepared
propagation. Ordinary ingestion validates and consumes it with its original
elapsed time and interval-event evidence, then performs the unchanged
observation, association, innovation, correction, lifecycle, cache, scheduler,
and diagnostic stages. It is neither persistent state nor a second prediction
path; stale, reused, wrong-source, or wrong-time values are rejected. Dynamics
parameters/buffers and train/eval mode are part of the same atomic revision.
This zero-copy guard uses PyTorch tensor mutation versions, so prepared
propagation runs under ordinary autograd or `torch.no_grad()`, not
`torch.inference_mode()`.

Closed-loop optimization first permits a profile-specific period of joint
global-perception adaptation, then freezes the RGB backbone/global detector.
The ordinary ROI path remains trainable and is supervised in persistent
belief-slot order on every usable prior frame. Correction supervision combines
separate position/velocity state and rollout accuracy, a small sparsity term,
and guards against posteriors that worsen current or future error.

RGB has two deliberately separate caches. `RGBModalityCache` contains
disposable feature maps, crops, and ROI support and is invalidated by global
discovery or an identity mismatch. `RGBTemporalPositionHistory` is a bounded
sensor-local measurement history keyed by persistent object ID. It survives
global/ROI mode changes, accepts only observed nonambiguous identities at
strictly increasing timestamps, and never replaces `WorldBelief`.

The temporal observer has two independently bounded rings per persistent ID.
The per-frame point ring emits the existing axis-local least-squares velocity
measurement. A separate scale-anchor ring retains only nonambiguous global
disc measurements whose silhouettes are neither boundary-truncated nor
overlap-split; intervening centre-only ROI frames cannot evict these scarce
depth anchors. A robust inverse-variance trajectory fit extrapolates the
point/scale abstraction to the current timestamp and can emit conservative
camera-depth position evidence. Both signals pass through typed analytic
correction into `WorldBelief`; neither ring is a second physical state.

The original continuous velocity-only policy remains bounded to young or
post-event tracks in public profiles. The new depth-position policy is
configuration-gated and must pass paired multistep, tracking, event, and
calibration evaluation before promotion.

The accepted one-object RGB-D path is deliberately smaller than the learned
RGB stack. A composite packet carries batched RGB, metric surface depth,
calibration, timestamp, and explicit image size. Parameter-free differentiable
geometry owns the metric measurement, direct position has exactly one filter
owner, and `RGBDTemporalPositionHistory` stores sixteen raw positions in
persistent-ID order. Uniform differentiable exact free-motion WLS emits
velocity only; analytic dynamics own the five future position/velocity
queries. Live history is causal runtime state rather than canonical physical
state and is not serialized in ordinary checkpoints.

Specification 1.56 froze and subsequently qualified the exactly-two-visible
extension of that same path. A symmetric chromatic-plus-spatial two-slot RGB-D
module produces
unordered measurements for two fully visible, image-separated, non-contact
spheres. Hard Hungarian is isolated to discrete stable identity; it is not a
differentiable measurement or state owner. Current-position gradients reach
anchor frame 15 only, while velocity and every analytic rollout reach all
sixteen RGB/depth frames. Four-scene batch VJPs require exact zero cross-scene
coupling. The extension owns zero parameters and optimizer updates and leaves
the accepted one-object behavior intact. Its one fixed development run and
ordered selector -> confirmation -> final qualification passed on clean commit
`3b781e653a0287b2aa926e7c0b969e9197d48e42`; final is consumed and must not be
rerun. The accepted scope remains exactly two fully visible, image-separated,
non-contact objects. Specification 1.57 subsequently qualified one known,
time-aligned calibrated orbital-camera family without adding pose estimation,
learned state, or optimizer updates. That reviewed result is published through
acceptance commit `00a712d640cdb828f24a194817443daa57e6df65`; it establishes
neither general camera motion nor unknown- or learned-pose inference.

The attempted partial-visibility and missed-observation-recovery family failed
its bounded development protocols and is closed. It must not be revived,
renamed, or treated as the next rung. Specification 1.58 instead froze the
genuinely new capability of distinct per-object linear-drag identification
inside the already accepted exactly-two-visible, known-orbit family. Variable
count, contact/material identification, occlusion/recovery, unknown camera
pose, actions/planning, richer modalities, and learned capacity remain
unqualified candidates rather than part of this rung.

The new RGB-D temporal fit is still parameter-free. Once a persistent ID has
sixteen complete metric positions, a differentiable bounded variable-profile
fit jointly emits fit-owned anchor position, anchor velocity, log drag, and
raw diagonal variance. That complete tuple directly and atomically replaces
the corresponding belief fields. Earlier frames keep the accepted direct
metric-position behavior, and partial tuple/axis evidence fails closed.
`ObjectBelief.log_drag` remains the sole physical-parameter owner.

Raw fitted covariance is not called calibrated. The frozen one-pass
development procedure would have derived position, velocity, and drag scale
factors from cached scale-one sufficient evidence and installed all three CPU
float32 scalar buffers atomically. Protected splits could only have loaded
that reviewed exact three-buffer state. Analytic rollout propagates diagonal
uncertainty directly
from the fitted anchor for each absolute query time, preserving one-call query
partition invariance. Arbitrary repeated external re-anchoring remains outside
the claim because `WorldBelief` does not carry the induced full covariance.

The formal scene family is seedless: each conceptual split owns ordinals
`0--63`, formed by four rational physical primitives, two exact drag-slot
counterfactuals, and eight orbital-camera strata. Independent recurrence and
ray arithmetic certify all 256 governed scenes without calling public physics,
renderer, perception, or runtime. The frozen protocol made runtime
materialization available only through ledger-owned four-ordinal capabilities
after the exact source was clean, committed, and published. No governed runtime
capability had been issued or consumed at the specification-1.58 source-freeze
boundary.

That protocol is now terminal after its sole development attempt. The private
constructor materialized the first four development episodes, but the generic
batch collator rejected their deliberately heterogeneous `metadata.albedo`
tuples before model construction, runtime ingestion, fitting, calibration, or
prediction. No protected capability was issued. The analytic fit, atomic
evidence, three-buffer checkpoint, direct-anchor uncertainty, and certificate
remain reusable source components, but their governed family is not an
accepted runtime result. No successor capability is currently selected.

Evaluation seed manifests are explicit. `fresh_validation` starts after the
checkpoint's trainer-validation episodes by default; `--seed-offset` can select
a later disjoint range for a one-time confirmation. Both are asserted disjoint
from the test range. Simulator collision state may condition evaluation metrics
over a future window, but is never passed into the runtime.

The abstraction and token layers are derived from `WorldBelief` on demand.
They do not cache physical state and add no parameters to existing
checkpoints. The initial router selects the lowest-complexity executable
operator supported by current evidence: point-trajectory execution in free
motion and rigid-sphere execution for contact-like modes. Full geometry and
slow parameters remain in the belief, making refinement lossless.
This first assignment is inspectable but does not yet prune the hybrid
dynamics path: free objects still run cheap contact-candidate detection so
they can refine before an imminent impact.

`WorldBeliefTokenizer` is a reversible bridge to future attention-based
models. It gives entity tokens stable IDs, types, masks, and abstraction kinds
instead of asking a transformer to infer the schema from one opaque vector.
Learned token projections, residual updates, generative hypotheses, and
evidence-driven routing remain subsequent work.
