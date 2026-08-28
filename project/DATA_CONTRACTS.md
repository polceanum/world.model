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
  `[B,N,3]` position/log-variance and independent position-validity mask. It is
  already aligned to persistent belief slots.
- `RGBTemporalPositionHistory`: persistent IDs plus two bounded
  `[B,N,R,...]` rings: per-frame point timestamps/positions/variance/validity
  and trustworthy scale-anchor timestamps/positions/variance/validity.
  It is sensor-local causal evidence, not physical state.
- `RGBDTemporalPositionHistory`: persistent IDs plus bounded `[B,N,R]`
  timestamps/sample/valid masks and `[B,N,R,3]` raw metric positions. Invalid
  sampled-depth rows retain their causal sample position. The accepted
  specification-1.55/1.56 paths require all sixteen valid rows and fail their
  complete uniform fit closed. The historical specification-1.57
  partial-visibility attempt also used `R=16`, permitted at most one scheduled
  object-local invalid target row, required the newest row valid, and required
  exactly fifteen valid target supports; the co-object remained independently
  valid with sixteen. Its first development construction failed exact renderer
  visibility preflight before any model, collate, runtime, or checkpoint state.
  Specification 1.58 preserves the same public history semantics in the
  terminal attempt-2 source freeze. The invalid row emits no direct
  state/velocity evidence, and the filter alone owns its single `0.08`
  missed-state variance increment. This live sensor-local state is not
  serialized by an ordinary checkpoint.
- Attempt-2 finite constructor contract: private, immutable constructor data
  selected from one finite `16`-rational-primitive by `8`-element exact-`D4`
  table.
  The resulting `128` physical cells are unique. They are preflight/scoring
  inputs, never a public `ObservationPacket`, `MeasurementSet`, or runtime
  oracle. Exact float32 world evolution is the public-solver-identical
  `342`-substep recurrence, while exact raster support owns visibility.
- Attempt-2 authorization contract: exact canonical single-use capability
  minted by the durable attempt-2 ledger. A manifest or seed record is
  descriptive data, not authority. The raw constructor/evaluator reject
  direct calls, protect the immutable two-file live-v1 failure, and admit only
  this capability. The v2 evidence contract permits exactly five canonical
  single-link files and loads the restricted checkpoint with
  `weights_only=True`; it contains no oracle input.
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
