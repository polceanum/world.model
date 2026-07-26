# Datasets

The first dataset is a deterministic 3-D sphere simulator rendered to RGB.
Episode records include images, timestamps, moving-camera calibration, padded
object states and physical parameters, visibility/projection labels, and contact
events. Train, validation, test, and OOD seeds occupy disjoint configured ranges.

On-the-fly generation is authoritative. Optional trusted local tensor caches may
be added without changing the episode contract.

