# World belief

The canonical belief contains batch-major padded object tensors, monotonic
integer IDs, active/existence/visibility state, position and velocity,
quaternion orientation, angular velocity, geometry/appearance/residual codes,
stable modal state, bounded physical parameter beliefs, motion modes, lifecycle
counters, diagonal log variance, and recurrent parameter memory.

It also contains timestamp, calibrated camera belief, gravity, global code,
next monotonic ID, active modalities, and diagnostics metadata. Object slots are
storage, not identity. Padded slots always have ID `-1` and masks are mandatory.

Predictive abstractions are selected views over these fields. Routing a freely
moving sphere as a point trajectory does not discard its radius or slow
physical parameters; the full belief remains available for lossless
refinement at contact. Typed belief tokens are likewise reversible views for
future transformer processing, never a second persistent source of truth.
