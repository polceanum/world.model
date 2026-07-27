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

Persistent object `motion_mode_logits` describe the instantaneous endpoint
state. A rollout step's collision logit instead describes occurrence anywhere
inside that prediction segment; internal physics substeps are aggregated.
Frame-labelled training/evaluation brackets each target with
`[h-dt_obs, h]`, so event probabilities have one explicit temporal meaning.

Closed-loop optimization first permits a profile-specific period of joint
global-perception adaptation, then freezes the RGB backbone/global detector.
The ordinary ROI path remains trainable and is supervised in persistent
belief-slot order on every usable prior frame. Correction supervision combines
state/rollout accuracy, a small sparsity term, and guards against posteriors
that worsen current or future error.

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
checkpoint's trainer-validation episodes and is asserted disjoint from the
test range. Simulator collision state may condition evaluation metrics over a
future window, but is never passed into the runtime.
