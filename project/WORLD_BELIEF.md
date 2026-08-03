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

Tentative birth evidence is explicitly outside the canonical belief. It is
bounded, detached, modality/sensor-local observation history with no object ID
and no participation in filtering, dynamics, rollouts, or slow-parameter
updates. Confirmation allocates a fresh monotonic ID and initializes a normal
belief slot; reset discards all tentative evidence.

A prepared propagation is also transient rather than a competing belief. It
is a typed one-use result derived from the current `WorldBelief` for one exact
future observation timestamp. The runtime may expose it to training
supervision and then consume it through ordinary ingestion, but validates its
source and preserves its elapsed-time/event evidence. It is never cached as
authoritative state and cannot be reused after the persistent belief or
dynamics parameter/buffer revision advances. Prepared propagation uses
version-tracked tensors and is therefore not an `inference_mode` API.
