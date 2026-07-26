# System overview

Normal operation is event-driven. For each packet timestamp, the runtime advances
the current posterior to a prior, projects the expected sensor measurement,
selects global or residual perception, associates unordered proposals, computes
innovation, corrects supported fast state, updates object lifecycle, and admits
slow parameter evidence only when observable.

`WorldBelief` remains persistent between calls. Future trajectories are
re-generated from the corrected posterior and do not require history replay or
network-weight updates.

