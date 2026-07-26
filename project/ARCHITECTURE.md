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

