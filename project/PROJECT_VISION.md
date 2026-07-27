# Project vision

Orpheus maintains a persistent, uncertainty-aware belief about a physical world.
Sensor modules translate timestamped asynchronous observations into structured
measurements; a modality-independent dynamics and filtering core predicts,
associates, measures innovation, corrects, and immediately revises future
rollouts.

The first proof uses small synthetic RGB sphere scenes. Its purpose is to test
the full architecture—identity, collisions, occlusion, uncertainty, and online
physical-parameter identification—not to claim a general visual world model.
See `PROJECT_SPEC.md` for the complete contract.

The scaling unit is a predictive abstraction: the smallest persistent,
executable representation that explains observations within calibrated
uncertainty. A ball should remain a point with a trajectory when that is
sufficient and refine to a sphere/contact model only when interaction demands
it. Foundation models and generative objectives extract and complete these
abstractions; they do not replace the world belief with generated pixels.
