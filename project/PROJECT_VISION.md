# Project vision

Orpheus maintains a persistent, uncertainty-aware belief about a physical world.
Sensor modules translate timestamped asynchronous observations into structured
measurements; a modality-independent dynamics and filtering core predicts,
associates, measures innovation, corrects, and immediately revises future
rollouts.

The first accepted proofs use small synthetic RGB and RGB-D sphere scenes to
exercise persistent state, causal measurement/correction, identity in fully
visible fixed-cardinality scenes, analytic free-motion rollout, and explicit
uncertainty diagnostics. They do not establish collision handling from pixels,
occlusion/recovery, calibrated posterior uncertainty, variable cardinality,
online physical-parameter identification, actions/planning, or a general
visual world model. Those remain staged architectural goals with independent
pre-access protocols. See `PROJECT_SPEC.md` for the complete accepted contract.

The scaling unit is a predictive abstraction: the smallest persistent,
executable representation that explains observations within calibrated
uncertainty. A ball should remain a point with a trajectory when that is
sufficient and refine to a sphere/contact model only when interaction demands
it. Foundation models and generative objectives extract and complete these
abstractions; they do not replace the world belief with generated pixels.
