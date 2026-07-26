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
