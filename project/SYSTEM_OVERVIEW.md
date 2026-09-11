# System overview

Normal operation is event-driven. For each packet timestamp, the runtime advances
the current posterior to a prior, projects the expected sensor measurement,
selects global or residual perception, associates unordered proposals, computes
innovation, corrects supported fast state, updates object lifecycle, and admits
slow parameter evidence only when observable.

`WorldBelief` remains persistent between calls. Future trajectories are
re-generated from the corrected posterior and do not require history replay or
network-weight updates.

The same belief can be viewed through a predictive-abstraction router. Free
motion currently selects a cheap point trajectory; contact-like modes refine
the entity to analytic sphere or oriented-box execution when observable shape
evidence is available; legacy geometry remains exactly spherical. A reversible typed token adapter exposes
scene, camera, kinematic, programme, and lifecycle information for future
attention-based models. These are derived interfaces, not alternate runtime
state.

The newest bounded capability path removes predeclared appearance prototypes
for unfamiliar rigid objects. Public calibrated RGB-D observations discover
and associate sphere/box support, recover identity after a short dropout, and
derive metric pose and angular velocity. Known isolated interventions,
contact-free motion, and observed collisions update bounded mass, drag,
restitution, and friction estimates from neutral priors.

An opt-in six-DoF executor adds off-centre contact, angular impulse, frictional
torque, and orientation propagation while preserving exact legacy sphere
behavior. The resulting belief supports downstream terminal world-pose
planning with exact serial/vectorized winner parity. This is currently an N=2,
calibrated-camera, short-recovery sphere/box claim; category-independent
touching-instance separation, longer recovery, and multi-contact visual scale
remain explicit next steps.
