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
          └── observability-gated slow parameter update
          │
          ▼
persistent WorldBelief
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

Persistent object `motion_mode_logits` describe the instantaneous endpoint
state. A rollout step's collision logit instead describes occurrence anywhere
inside that prediction segment; internal physics substeps are aggregated.
Frame-labelled training/evaluation brackets each target with
`[h-dt_obs, h]`, so event probabilities have one explicit temporal meaning.

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

The temporal history can emit a causal least-squares world-velocity
measurement after three positions. A second analytic correction updates only
velocity and its diagonal uncertainty. This path has no new model weights and
is currently disabled in public profiles because its first fresh-validation
ablations improved velocity but regressed primary localization/forecast
metrics.

Evaluation seed manifests are explicit. `fresh_validation` starts after the
checkpoint's trainer-validation episodes by default; `--seed-offset` can select
a later disjoint range for a one-time confirmation. Both are asserted disjoint
from the test range. Simulator collision state may condition evaluation metrics
over a future window, but is never passed into the runtime.
