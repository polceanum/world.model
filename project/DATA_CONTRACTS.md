# Data contracts

Public tensors are batch-major. `B` is batch, `T` time, `N` padded object slots,
`M` unordered measurements, `K` stable modes, `L` token length, `R` bounded
history length, and `H_img,W_img` image height/width.

- `ObservationPacket`: immutable raw timestamped sensor event. The qualified
  RGB-D form is one composite modality-qualified packet containing
  `[B,3,H_img,W_img]` RGB, `[B,1,H_img,W_img]` metric depth, batched calibration,
  explicit image size, and one timestamp; separate same-time RGB/depth packets
  are not equivalent.
- `MeasurementSet`: `[B,M,Dm]` values/log variance, existence, mask, optional
  appearance/class evidence, frame, supported state fields, auxiliary tensors.
- `ObjectBeliefTensor`: `[B,N,...]` persistent object state/parameters/masks.
- `WorldBelief`: `[B]` timestamp plus objects, camera, gravity, global state,
  next ID, and metadata.
- `AssociationResult`: unique belief/measurement pairs and unmatched masks.
- `BeliefTrajectory`: `[B,T,N,...]` sampled future state and uncertainty.
- `DirectVelocityEvidence`: required `[B,N,3]` world velocity/log variance plus
  `[B,N]` validity/confidence, with an optional jointly validated
  `[B,N,3]` position/log-variance and position-validity mask, and an optional
  jointly validated `[B,N,1]` log-drag/log-variance plus `[B,N]` drag-validity
  mask. Drag evidence requires complete position/velocity/drag validity for
  the same slots; partial groups or axes fail closed. The object is already
  aligned to persistent belief slots and the updater applies the complete
  supported tuple atomically.
- `RGBTemporalPositionHistory`: persistent IDs plus two bounded
  `[B,N,R,...]` rings: per-frame point timestamps/positions/variance/validity
  and trustworthy scale-anchor timestamps/positions/variance/validity.
  It is sensor-local causal evidence, not physical state.
- `RGBDTemporalPositionHistory`: persistent IDs plus bounded `[B,N,R]`
  timestamps/sample/valid masks and `[B,N,R,3]` raw metric positions. Invalid
  sampled-depth rows retain their causal sample position but fail the complete
  uniform fit closed. The current qualified/frozen RGB-D paths use `R=16`.
  This live sensor-local state is not serialized by an ordinary checkpoint.
- Identifiable-drag RGB-D state adds three conditional persistent CPU float32
  scalar buffers: position, velocity, and drag uncertainty scale. They are
  installed as one atomic development-calibration group and serialize as
  exactly three state-dict entries / 12 tensor bytes. Live histories remain
  outside the checkpoint.
- `AbstractionAssignment`: `[B,N]` abstraction kind, routing confidence,
  complexity cost, refinement reason, and active mask. It is derived from
  `WorldBelief`, not stored as independent physical state.
- `PredictiveTokenBatch`: reversible `[B,L,Dtoken]` values plus token type,
  persistent object ID, object slot, abstraction kind, validity, timestamp,
  next-ID state, and camera calibration mask. Initial token types are scene,
  camera, entity kinematic, entity programme, and entity lifecycle.

Conventions: float32 physical state, int64 IDs, bool masks, seconds, metres,
scalar-last quaternions `[x,y,z,w]`, normalized image coordinates `[-1,1]`, and
clamped log variance.

When present, RGB `MeasurementSet.auxiliary` exposes
`world_velocity`, `world_velocity_log_variance`, and
`world_velocity_valid_mask` for diagnostics in measurement-slot order.
Invalid slots contain finite placeholders and must be ignored by the mask.
The multi-frame point/scale observer similarly exposes
`world_trajectory_position`, `world_trajectory_position_log_variance`, and
`world_trajectory_position_valid_mask`.

Token adapters must reproduce every canonical belief tensor exactly when
decoded against a matching schema. Learned transformer embeddings are not
themselves canonical belief state.
