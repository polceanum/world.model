# Multimodal design

Every sensor arrives as an immutable `ObservationPacket` with modality,
sensor ID, timestamp, payload, calibration, coordinate frame, confidence, and
metadata. Registered modules validate payloads, initialise or encode unordered
measurements, project expected measurements from a belief, calculate typed
innovation, and expose modality-specific losses.

The initial implementation registers `rgb` and clearly labelled
`debug_oracle`. Future modalities add an adapter and optional projector; they do
not change the persistent state scheduler or physical dynamics.

