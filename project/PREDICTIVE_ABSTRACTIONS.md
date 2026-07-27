# Predictive abstractions

## Purpose

Orpheus scales by extracting compact executable models from complex
observations. It does not attempt to preserve or regenerate every sensor detail
before it can make a prediction.

For a freely moving ball, identity, position, velocity, uncertainty, and a
trajectory operator are usually sufficient. Near contact, the same persistent
entity can refine to a sphere with radius, mass, restitution, friction, and a
contact operator. The richer model is used only while its additional structure
is predictively useful.

## Contract

A predictive abstraction has:

- persistent identity;
- a named abstraction family;
- explicit state and uncertainty;
- an executable evolution operator;
- measurement projectors for correction;
- slow physical parameters with observability gates;
- learned residual tokens for unexplained appearance, semantics, or dynamics;
- a complexity cost used during future model selection.

`WorldBelief` remains the source of truth. Abstraction assignments and token
sequences are derived views. They cannot silently maintain independent
positions, identities, or physical parameters.

## Initial executable families

`POINT_TRAJECTORY` uses analytic kinematics, stable modes, position, velocity,
and uncertainty. The current deterministic router selects it for ordinary free
motion.

`RIGID_SPHERE` additionally uses radius, mass, restitution, friction, and the
sphere contact resolver. The router refines to it for ground contact, pair
contact, collision, rolling, and sliding modes.

All object properties remain present in `WorldBelief` in either mode. Routing
changes the selected predictive operator, not the canonical stored state.
In the first implementation the assignment is an inspectable recommendation;
the existing hybrid dynamics still runs cheap contact-candidate detection for
free objects so an imminent collision cannot be missed. Assignment-driven
execution pruning waits for a validated proximity/uncertainty refinement gate.

## Belief-token interface

`WorldBeliefTokenizer` emits a reversible typed sequence:

1. scene token;
2. camera token;
3. kinematic token per entity;
4. dynamical-programme token per entity;
5. lifecycle token per entity.

Each entity token carries persistent object identity and the selected
abstraction kind. Padding is explicit. This provides an LLM-style sequence
interface without reducing the belief to an undifferentiated token stream.

A future transformer may:

- infer entities and relations from foundation visual features;
- update residual tokens;
- propose abstraction refinements;
- predict events and multiple future programmes;
- condition predictions on actions or language.

Its outputs must be typed proposals that are checked against measurements,
uncertainty, physical constraints, and simpler baselines before being
assimilated.

## Selection objective

The intended evidence-driven router minimizes:

```text
future state and event error
+ uncertainty miscalibration
+ correction magnitude
+ abstraction complexity
```

A more complex abstraction is promoted only if it improves held-out prediction
or calibration beyond a configured margin. Per-abstraction and worst-slice
metrics prevent improvements on rare complex interactions from degrading
ordinary motion.

## Scaling direction

Likely future families include rigid SE(3) bodies, articulated graphs,
connected curves, spatial fields, and generic learned object programmes. They
should be added only with an implemented executor, measurement projector,
uncertainty semantics, and validation data. Empty ontology labels are not
useful abstractions.
