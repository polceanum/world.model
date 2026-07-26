# Dynamics

The dynamics stack is intentionally decomposed:

1. timestamp-aware analytic kinematics with gravity, drag, and quaternion
   integration;
2. stable vectorised rotation-decay modal state;
3. relative-geometry interaction messages;
4. structured ground/pair event and impulse jumps;
5. diagonal uncertainty propagation and nonnegative process noise;
6. an arbitrary-query-time rollout engine that never mutates its input.

The simulator and model share physical units and broad structure, but simulator
labels and true parameters are not supplied to the RGB runtime.

