# Data contracts

Public tensors are batch-major. `B` is batch, `T` time, `N` padded object slots,
`M` unordered measurements, and `K` stable modes.

- `ObservationPacket`: immutable raw timestamped sensor event.
- `MeasurementSet`: `[B,M,Dm]` values/log variance, existence, mask, optional
  appearance/class evidence, frame, supported state fields, auxiliary tensors.
- `ObjectBeliefTensor`: `[B,N,...]` persistent object state/parameters/masks.
- `WorldBelief`: `[B]` timestamp plus objects, camera, gravity, global state,
  next ID, and metadata.
- `AssociationResult`: unique belief/measurement pairs and unmatched masks.
- `BeliefTrajectory`: `[B,T,N,...]` sampled future state and uncertainty.

Conventions: float32 physical state, int64 IDs, bool masks, seconds, metres,
scalar-last quaternions `[x,y,z,w]`, normalized image coordinates `[-1,1]`, and
clamped log variance.

