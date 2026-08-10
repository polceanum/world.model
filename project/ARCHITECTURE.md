# Architecture

```text
timestamped ObservationPacket
          │
          ▼
registered RGB / debug-oracle observation module
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
