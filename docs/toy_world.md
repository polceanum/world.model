# Synthetic sphere world

The toy environment evolves 3-D spheres in a bounded world with gravity, drag,
ground/wall and pair collisions, restitution, tangential damping, occlusion, and
a calibrated fixed or moving camera. A perspective disc renderer emits RGB plus
segmentation and visibility labels.

Simulator state is available for supervision, evaluation, and debug-oracle
tests. Ordinary runtime receives only image pixels, timestamp, and calibration.

