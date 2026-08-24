# Project Orpheus

## Authoritative Technical Specification and Codex Build Directive

**Status:** Living authoritative specification
**Version:** 1.69
**Date:** 26 July 2026; predictive-abstraction and interpretable-physics amendments 27 July 2026; shared-regime selection amendment 28 July 2026; sustained-training and broad-checkpoint-selection amendment 30 July 2026; convergence-integrity, identifiable-forecast, runtime-invariant, and continuation-integrity amendments 1 August 2026; supported-causal-optimization and hierarchical-gradient-stability amendment 2 August 2026; lifecycle, identity, supervision, perception-gradient-integrity, validation-support, launch-failure-integrity, cadence-semantics, progress-observability, finite-state, integration-grid, prepared-propagation, and launch-QoS amendments 3 August 2026; mutable-optimisation and long-run resource-integrity amendments 6 August 2026; modular-qualification and fast-ROI isolation amendment 7 August 2026; trainable-path objective-integrity and staged-scope amendments 8 August 2026; perception-local auxiliary-gradient routing, rollout uncertainty-gradient isolation, scenario-balanced optimization, innovation-anchored correction, and staged abstraction-attention scaling amendments 9 August 2026; axis-isolated correction recovery, fast-ROI ownership stability, zero-initialized typed-attention pilot, live scene-context, mixed-unit scene-conditioning, collision-head gradient isolation, complete typed-attention gradient localization, force-head isolation, and evidence-gated capacity scaling amendments 10 August 2026; typed-output, impulse-jump, accumulated node-gradient isolation, measured compute/data scaling, function-preserving architecture-handoff, identity-initialized appended-depth, pooled training-trend observability, aggregate recursive semantic-gradient budgeting, and residual-parsimony amendments 11 August 2026; non-vacuous protected-checkpoint audit, functional residual-activity, context-sensitive drift, exact absolute-index learning-rate schedule, residual-prior gradient-alignment, and relation-first typed-attention qualification amendments 12 August 2026; fixed-boundary checkpoint, optimizer-step, and exclusive trend-window audit-integrity amendments 13 August 2026; evidence-bounded heterogeneous mental-simulation, low-noise live-monitoring, familiar-simulator, independent-RGB-evidence, clean-evaluation, semantic-versioning, and staged-convergence amendments 15 August 2026
**Amendment:** observation-completeness, calibrated temporal uncertainty, finite differentiable-event, causal-objective-support, and campaign-cadence amendments 16 August 2026; production-MPS event-hazard numerical-integrity amendment 20 August 2026; dynamics elapsed-time synchronization, validation-anchor batching, auxiliary-gradient ownership, zero-output residual elision, live-update observability, measured phase-device policy, comprehensive promotion evidence, immutable paired replay, fail-closed convergence semantics, axis-gated learned correction, batch-macro physical objectives, axiswise correction hinges, provenance-bound updater composition, exact-resume snapshot/publication ownership hardening, immutable-initializer/paired-wiring qualification, and common rich fixed-32 step-zero equivalence amendments 21 August 2026; regime-local hypothesis applicability and bounded recursive composition amendment 22 August 2026; forecast-only hypothesis isolation, learned-uncertainty ownership, exact abstention, RGB temporal-velocity veto, output-only causal residual diagnostics, exact lateral updater-head ownership, and scenario-axis-horizon tail-risk objective amendments 23 August 2026; runtime-local observation-fitted transition candidate, bounded diminishing-returns gate, event-frame-targeted training data, detector-only multi-instance discovery repair, raw learned-existence supervision-boundary, opt-in dense multi-instance global-discovery, causal observation-model selection, terminal dense typed-attribute evidence, frozen-foundation-feature feasibility, adaptive Gaussian local-model evidence, event-epoch local-model evidence, and differentiable hard-runtime assimilation surrogate amendments 24 August 2026
**Intended location in repository:** `/PROJECT_SPEC.md`  
**Primary local environment:** conda environment `orpheus`, PyTorch with Apple MPS support  
**Initial runtime modality:** synthetic RGB, with privileged simulator state used only for supervision, evaluation, and debugging  
**Long-term modality policy:** arbitrary asynchronous modalities through stable observation-module contracts

---

## 0. How to use this document

This file is both:

1. the permanent technical specification for the repository; and
2. the detailed build directive for Codex or another coding agent working from an empty repository.

Place this file in the repository root as `PROJECT_SPEC.md`. Create an `AGENTS.md` that instructs every coding agent to read `PROJECT_SPEC.md`, `project/STATUS.md`, `project/TASKS.md`, and `project/DESIGN_DECISIONS.md` before changing code.

The specification defines the intended system, public interfaces, implementation sequence, training strategy, evaluation criteria, and engineering constraints. It should prevent the project from drifting into a sequence of disconnected prototypes.

### Specification precedence

When implementation choices conflict, use this order:

1. correctness and explicit acceptance criteria in this specification;
2. stable public interfaces and tensor/data contracts in this specification;
3. documented design decisions in `project/DESIGN_DECISIONS.md`;
4. simplicity and maintainability;
5. runtime performance;
6. implementation convenience.

A coding agent may refine internal implementations, but it must not silently remove the following architectural properties:

- persistent world belief;
- causal online predict–observe–correct operation;
- modality-independent core state;
- explicit timestamps and asynchronous observations;
- uncertainty-aware updates;
- separation of fast state from slow physical parameters;
- object persistence and data association;
- hybrid structured and learned dynamics;
- long-horizon rollout training;
- recovery from imperfect beliefs rather than pure teacher forcing;
- simple local training and evaluation commands;
- a path from Apple MPS to CUDA without architectural redesign.

### Exact short instruction to give Codex

After placing this file at the repository root, give Codex this instruction:

> Read `PROJECT_SPEC.md` in full before writing code. Treat it as the authoritative specification. Start from the empty repository and implement the complete first vertical slice described under “Implementation programme” and “Milestone 1 definition of done,” not merely a scaffold. Create and maintain all repository memory files required by the specification. Use the existing conda environment `orpheus`; do not reinstall or replace PyTorch. Keep the user-facing workflow to `python train.py`, `python evaluate.py`, and `python demo.py` with YAML configuration. Run tests and the toy end-to-end validation locally. Record decisions, current status, commands run, known limitations, and next tasks in the repository. Do not stop at placeholder classes, pseudocode, or an oracle-only demonstration.

---

# Part I — Purpose, context, and commitments

## 1. Mission

Build a modular online multimodal physical world model that continuously estimates a persistent, uncertainty-aware state of the physical world, predicts its future evolution, and cheaply revises those predictions as new ground-truth observations arrive.

The system is not fundamentally a video generator. It is not fundamentally an audio model, skeleton model, or robot-state predictor. It models a latent world; sensor-specific modules translate between observations and the shared world belief.

The intended online loop is:

\[
\text{predict world}
\rightarrow
\text{predict measurements}
\rightarrow
\text{receive observations}
\rightarrow
\text{associate}
\rightarrow
\text{compute innovation}
\rightarrow
\text{correct belief}
\rightarrow
\text{predict again}.
\]

The key product is a continuously updated `WorldBelief`, not a generated frame sequence.

## 2. Original research motivation

The architecture was motivated partly by a spectral trajectory-generation idea: rather than autoregressively predicting each future state, infer a compact description of a complete temporal process and evaluate that process at future times.

The paper **“Spectral Diffusion for Protein Dynamics”** (Phipps et al., arXiv:2607.04134, 2026) generates temporal DCT spectral volumes and transforms them back into whole protein trajectories. Its central useful inductive bias is that slow and fast temporal behaviour can be exposed in an ordered frequency representation, allowing a model to generate temporally coherent windows without feeding every generated frame back into the model.

For Project Orpheus, the important adaptation is not to copy a fixed-window protein diffusion model. It is to transform the idea into online amortised system identification:

\[
\text{observed prefix}
\xrightarrow{\text{inference}}
\text{compact dynamical programme}
\xrightarrow{\text{deterministic evolution}}
\text{future world trajectory}.
\]

A fixed DCT vector is a **trajectory code**: it reconstructs one finite window. The project instead needs a **dynamical state**: a sufficient statistic that can be updated causally and evolved beyond a fixed window.

Therefore the online core will use stable continuous/modal state—closely related to spectral and Koopman representations—combined with explicit kinematics, learned interactions, discrete events, and online filtering. A direct DCT/window predictor may be implemented later as a baseline, but it is not the persistent online state.

Reference:

- Hew Phipps, Matteo Cagiada, Santiago D. Villalba, Charlotte M. Deane. “Spectral Diffusion for Protein Dynamics.” arXiv:2607.04134, 2026. https://arxiv.org/abs/2607.04134

## 3. Non-negotiable design principles

### 3.1 Predict the world, not a sensor

The shared state describes objects, agents, geometry, motion, interactions, physical parameters, uncertainty, and coordinate frames. RGB, audio, skeletons, depth, IMU, and future sensors are observation sources.

Sensor embeddings may be cached inside modality modules. They must not become the only representation of world state.

### 3.2 Persistent belief, not repeated clip encoding

The system maintains state continuously. It does not re-encode the entire observation history on every step.

A more expensive initialisation or recovery pass is permitted. Normal operation must use incremental prediction and correction.

### 3.3 Cheap updates on incoming ground truth

Most adaptation is a state update, not a network-weight update.

Every incoming observation should update:

- current object state;
- uncertainty;
- event/motion mode;
- visibility and existence;
- gradually, identifiable slow physical parameters.

Large network weights remain fixed during normal online inference. Optional parameter-efficient test-time adaptation belongs to future work and must not be required for the initial loop.

### 3.4 Hybrid dynamics

Known mathematical structure should be explicit:

- timestamps;
- coordinate transforms;
- position/velocity integration;
- quaternion or Lie-group orientation integration;
- gravity;
- simple damping;
- equal-and-opposite pairwise impulses where applicable;
- bounded/stable modal evolution.

Learned components model:

- residual forces;
- contact and event probabilities;
- impulse corrections;
- unmodelled interactions;
- uncertainty growth;
- belief corrections.

### 3.5 Object-centric persistent state

Physical entities have persistent identity. The world belief is a typed object/graph state, not an undifferentiated temporal token sequence.

The architecture must support object creation, occlusion, reappearance, and removal. It must preserve data association as an explicit subsystem.

### 3.6 Multimodal by construction

Every observation carries its own timestamp. Modalities may arrive at different rates and be absent on arbitrary steps.

Adding a modality must not require changes to the belief, dynamics, or runtime scheduler beyond registering a conforming observation module and, where necessary, adding a modality-specific projector.

### 3.7 Uncertainty is part of state

The model must know when it is uncertain. Uncertainty affects association, correction gain, object lifecycle decisions, parameter updates, and prediction intervals.

The initial implementation may use diagonal Gaussian uncertainty for tractability. The public contracts must permit low-rank, full, ensemble, or mixture representations later.

### 3.8 The toy validation is a scale reduction, not a different architecture

The first runnable system may use:

- low-resolution synthetic RGB;
- a small maximum object count;
- a small CNN;
- diagonal uncertainty;
- a small graph network;
- one belief hypothesis;
- simple rigid bodies.

It must still execute the actual target loop: discovery, association, persistent belief, dynamics, prediction, innovation, correction, uncertainty, event handling, and online slow-parameter updates.

### 3.9 Stable interfaces over endless redesign

The core contracts in this document should be frozen early. Individual neural architectures can improve behind them.

A change to a backbone should not require changing the world belief. A change from RGB to skeleton observations should not require changing dynamics. A change from diagonal to low-rank covariance should not require changing the public runtime API.

### 3.10 Straightforward operations

The normal workflow is:

```bash
conda activate orpheus
pip install -e ".[dev]"
python train.py --config configs/toy_mps.yaml
python evaluate.py --config configs/toy_mps.yaml --checkpoint runs/<run>/checkpoints/best.pt
python demo.py --config configs/toy_mps.yaml --checkpoint runs/<run>/checkpoints/best.pt
pytest
```

No web service, authentication system, API tokens, job server, experiment database, Docker requirement, Kubernetes layer, or distributed orchestration is needed for the first implementation.

### 3.11 Predictive abstractions are the scaling unit

The system should extract the smallest persistent representation that is
sufficient for useful prediction, rather than preserving sensor detail by
default.  A freely moving ball may be represented by identity, a point,
velocity, uncertainty, and a trajectory model.  When contact becomes relevant,
the same entity may refine to a sphere with geometry, mass, restitution, and a
contact operator.  The richer representation must not erase the cheaper one or
create a second source of truth.

Predictive abstractions must be:

- **executable:** they name the state and operator needed to roll forward;
- **persistent:** identity and inferred properties survive between observations;
- **minimal:** use the least-complex implemented model that explains observations
  within calibrated uncertainty;
- **refinable:** entities may move between point, rigid-body, articulated,
  field, or learned-residual representations as evidence and tasks require;
- **compositional:** relations and events connect entities without fusing them
  into an opaque scene vector;
- **correctable:** every abstraction projects expected measurements so new
  observations can produce innovation and revise the belief;
- **hybrid:** explicit state is accompanied by bounded learned residual tokens
  for information that the current schema does not yet explain.

Modern foundation models, transformers, and generative objectives are
perception and inference tools behind this contract.  They may propose
entities, abstraction families, relations, residual tokens, or multiple future
hypotheses.  They do not replace `WorldBelief` with an opaque video latent, and
photorealistic generation is never sufficient evidence of correct physical
prediction.

The first implemented abstraction families are:

1. `POINT_TRAJECTORY` for free motion using position, velocity, uncertainty,
   and analytic/modal evolution; and
2. `RIGID_SPHERE` when contact modes require radius and structured impulse
   dynamics.

All sphere properties remain stored in `WorldBelief` during cheap point
execution so refinement is lossless.  Abstraction assignments and LLM-style
tokens are derived views, never independently cached physical truth.
The first mode-based assignment is inspectable but must not prune cheap
contact-candidate detection; execution pruning requires a validated
proximity/uncertainty gate that refines before an imminent interaction.

## 4. Scope

### 4.1 Initial scope

The first end-to-end implementation targets synthetic multi-object rigid-body scenes observed through RGB:

- 3D sphere dynamics in a bounded scene, projected into a small RGB image;
- gravity;
- sphere–sphere and sphere–plane collisions;
- drag/friction-like effects;
- variable restitution and mass;
- partial and complete occlusion;
- a moving calibrated camera;
- object entrance/existence events where practical;
- online future rollout and correction.

The simulator provides exact states and events for supervision and evaluation. Runtime inference consumes RGB and calibration, not simulator state. An optional state/oracle observation module is allowed only for unit tests, debugging, and dynamics ablation.

### 4.2 Deliberate initial exclusions

The first milestone does not need:

- photorealistic rendering;
- unrestricted real-world object discovery;
- humans or deformable bodies;
- fluids, fracture, or topology change;
- arbitrary natural audio;
- large-language interfaces;
- planning or control;
- multi-GPU training;
- full differentiable rendering;
- online backpropagation through large networks;
- a production service API.

The architecture must leave clean extension points for these areas.

### 4.3 Long-term scope

Without changing the core loop, the repository should support:

- depth, stereo, optical flow, LiDAR, event cameras;
- microphones and acoustic event observations;
- human or robot skeletons;
- motion capture;
- IMU and proprioception;
- multiple cameras;
- articulated and deformable object state;
- real video and robotics datasets;
- counterfactual interventions;
- planning over belief trajectories;
- large-scale CUDA training.

## 5. Definitions

**Observation:** raw or preprocessed sensor data with a timestamp, calibration, coordinate frame, confidence, and modality identifier.

**Measurement:** structured evidence extracted from an observation, such as an object centroid, depth, joint coordinate, angular rate, impact event, or learned feature with uncertainty.

**World state:** a hypothetical exact state of the scene at a time.

**World belief:** a probability-bearing estimate of world state maintained by the model.

**Prior:** belief after dynamics prediction and before assimilating the latest observation.

**Posterior:** corrected belief after assimilating observations.

**Innovation:** discrepancy between an observed measurement and the measurement predicted from the prior.

**Dynamical programme:** compact state and parameters that determine a coherent future evolution when passed through the dynamics model.

**Fast state:** variables that can change every timestep, such as pose, velocity, contact mode, and visibility.

**Slow parameters:** properties inferred from accumulated evidence, such as drag, restitution, mass ratio, friction, geometry, or material code.

**Measurement projector:** modality-specific function mapping a world belief to expected sensor-space measurements.

**Observation encoder:** modality-specific function mapping raw observations and prior predictions to structured measurements.

**Association:** matching measurements to persistent object identities or declaring births, misses, and ambiguous assignments.

**Event:** a discrete dynamics transition such as impact, contact, attachment, release, sleep, external actuation, creation, or removal.

**Predictive abstraction:** the smallest typed state and executable evolution
operator that explains an entity or process within calibrated uncertainty.

**Residual token:** a bounded learned feature carried beside explicit state for
appearance, semantics, or dynamics not yet captured by the selected
abstraction.

**Belief token:** a typed, reversible view of scene, camera, entity state,
dynamical programme, or lifecycle information derived from `WorldBelief` for
attention-based processing.

---

# Part II — Formal system model

## 6. Probabilistic formulation

Let the latent world state at timestamp \(t\) be \(S_t\), slow parameters be \(\Theta_t\), discrete event/mode variables be \(M_t\), and all observations received up to \(t\) be \(O_{\le t}\).

The desired belief is:

\[
B_t =
p(S_t,\Theta_t,M_t \mid O_{\le t}).
\]

The initial implementation approximates this distribution using:

- a mean state;
- diagonal log variances;
- categorical mode probabilities;
- existence probabilities;
- optionally one or more weighted hypotheses.

For a time interval \(\Delta t\), prediction is:

\[
B^-_{t+\Delta t}
=
\mathcal{F}_{\psi}
\left(B_t,\Delta t,u_{t:t+\Delta t}\right),
\]

where \(\mathcal{F}_{\psi}\) combines explicit integration and learned residual/event models.

For modality \(m\), the expected measurement is:

\[
\hat Y^{(m)}_{t}
=
\mathcal{H}^{(m)}_{\phi}
\left(B^-_t,C^{(m)}_t\right),
\]

where \(C^{(m)}_t\) includes calibration and frame information.

The observation encoder produces:

\[
Y^{(m)}_t, R^{(m)}_t =
\mathcal{E}^{(m)}_{\eta}
\left(O^{(m)}_t,\hat Y^{(m)}_t,\text{cache}^{(m)}\right),
\]

where \(R^{(m)}_t\) is measurement uncertainty.

After association \(A_t^{(m)}\), innovation is:

\[
r_t^{(m)}
=
Y_t^{(m)}
-
\hat Y_t^{(m)}.
\]

Correction is:

\[
B_t
=
\mathcal{U}_{\omega}
\left(B^-_t,
      Y_t^{(m)},
      \hat Y_t^{(m)},
      r_t^{(m)},
      R_t^{(m)},
      A_t^{(m)}\right).
\]

If multiple modalities arrive at the same timestamp, they may be assimilated sequentially in a deterministic configured order or jointly after association. The first implementation should use sequential updates because it is simpler and naturally supports asynchronous events.

## 7. Dynamical programme formulation

For object \(i\), define fast state:

\[
x_{i,t} =
\left[
p_{i,t},
v_{i,t},
q_{i,t},
\omega_{i,t},
h_{i,t}^{\text{mode}}
\right],
\]

where:

- \(p\): position;
- \(v\): linear velocity;
- \(q\): orientation quaternion;
- \(\omega\): angular velocity;
- \(h^{\text{mode}}\): compact stable modal state.

Define slow parameters:

\[
\theta_{i,t} =
\left[
\log m_i,
\operatorname{logit} e_i,
\log c^{\text{drag}}_i,
\operatorname{logit}\mu_i,
g_i,
a_i,
d_i
\right],
\]

where:

- \(m\): mass or mass ratio;
- \(e\): restitution;
- \(c^{\text{drag}}\): drag/damping;
- \(\mu\): friction-like coefficient;
- \(g_i\): geometry code or known geometry parameters;
- \(a_i\): appearance code;
- \(d_i\): learned residual dynamics code.

The initial toy implementation may mask orientation and angular velocity losses for spherical objects while keeping the fields in contracts.

### 7.1 Analytic free-motion component

For substep \(\delta t\):

\[
a_i^{\text{base}}
=
g
-
c_i^{\text{drag}} v_i
+
a_i^{\text{external}},
\]

\[
v_i'
=
v_i+\delta t\,a_i^{\text{base}},
\]

\[
p_i'
=
p_i+\delta t\,v_i'.
\]

Orientation integration uses a unit quaternion:

\[
q_i'
=
\operatorname{normalize}
\left(
q_i \otimes
\exp_q\left(\tfrac12\delta t\,\omega_i\right)
\right).
\]

The code must renormalise quaternions and test norm stability.

### 7.2 Stable modal component

A fixed finite-window DCT is awkward for online updates. Instead maintain \(K\) stable rotation–decay modes. For each mode \(k\):

\[
z_{i,k,t}\in\mathbb{R}^{2d_m}.
\]

For each paired mode dimension:

\[
z_{i,k,t+\delta t}
=
\rho_{i,k}^{\delta t}
\begin{bmatrix}
\cos(\omega_{i,k}\delta t) & -\sin(\omega_{i,k}\delta t)\\
\sin(\omega_{i,k}\delta t) & \cos(\omega_{i,k}\delta t)
\end{bmatrix}
z_{i,k,t}.
\]

Parameterise:

\[
\rho_{i,k}=\exp(-\operatorname{softplus}(\alpha_{i,k}))
\]

for non-growing modes. Permit explicitly configured integrator/constant modes separately so sustained translation is not forced to decay.

A learned readout maps modes to bounded residual acceleration or latent dynamics features:

\[
a_i^{\text{modal}}
=
W_{\text{modal}} z_{i,t}.
\]

This preserves the spectral idea—compact coherent modes with explicit phase and frequency—while permitting causal correction and arbitrary-horizon evaluation.

### 7.3 Interaction graph

At each dynamics substep, construct candidate edges \(i\rightarrow j\) using geometry, distance, uncertainty-expanded bounds, and optionally learned interaction likelihood.

An edge feature includes:

\[
e_{ij} =
[
p_j-p_i,\;
v_j-v_i,\;
\|p_j-p_i\|,\;
r_i,\;
r_j,\;
\theta_i,\;
\theta_j,\;
P_i,\;
P_j,\;
m_i^{\text{mode}},\;
m_j^{\text{mode}}
].
\]

A small message network predicts:

- contact probability;
- event logits;
- residual force/acceleration;
- impulse magnitude or correction;
- uncertainty contribution.

Messages are aggregated symmetrically. Where an impulse interpretation applies, enforce equal-and-opposite pairwise action in the update rather than allowing two unrelated force outputs.

### 7.4 Event dynamics

The continuous state is conditioned on an explicit categorical mode. Initial event types:

- `FREE`;
- `GROUND_CONTACT`;
- `PAIR_CONTACT`;
- `COLLISION`;
- `ROLLING`;
- `SLIDING`;
- `SLEEPING`;
- `OCCLUDED` (observation mode, not physical force mode);
- `EXTERNALLY_ACTUATED`;
- `CREATED`;
- `REMOVED`.

The toy milestone needs at least `FREE`, `GROUND_CONTACT`, `PAIR_CONTACT/COLLISION`, `SLEEPING`, and `OCCLUDED`.

A collision transition can apply a structured jump:

\[
x_{t^+}
=
J_\psi(x_{t^-},e_{ij},\theta_i,\theta_j).
\]

For spheres, use analytic collision normal \(n\) and a predicted or parameter-derived scalar impulse \(j\):

\[
v_i^+ = v_i^- - \frac{j}{m_i}n,
\qquad
v_j^+ = v_j^- + \frac{j}{m_j}n.
\]

The learned model may predict a bounded correction to the analytic impulse, not unconstrained post-collision velocities.

## 8. Online filtering formulation

The filter maintains mean and uncertainty. The initial version uses diagonal variance in a canonical state vector.

### 8.1 Prediction

For state mean:

\[
\mu_t^- = F_\psi(\mu_{t-1},\Delta t).
\]

For diagonal variance:

\[
\sigma_t^{-2}
=
\operatorname{diagApprox}
\left(A_t P_{t-1}A_t^\top\right)
+
Q_\psi(B_{t-1},\Delta t),
\]

where \(A_t\) may be an analytic or automatic-differentiation local Jacobian for simple kinematics. To keep the first implementation straightforward, closed-form variance propagation may be used for position/velocity and a learned non-negative process-noise increment for the remaining dimensions.

### 8.2 Measurement update

For directly comparable dimensions, use an analytic diagonal Kalman proposal:

\[
K =
\frac{\sigma^{-2}}
{\sigma^{-2}+R}.
\]

\[
\mu^{\text{base}}
=
\mu^- + K\odot r.
\]

\[
\sigma_{\text{base}}^2
=
(1-K)\odot\sigma^{-2}.
\]

A small learned corrector then outputs a bounded residual update:

\[
\Delta\mu,\Delta\log\sigma^2,g
=
U_\omega[
\mu^-,
\log\sigma^{-2},
Y,
\hat Y,
r,
\log R,
\text{visibility},
\text{mode},
\text{modality embedding}
].
\]

\[
\mu^+
=
\mu^{\text{base}}
+
g\odot\Delta\mu.
\]

The gate \(g=\operatorname{sigmoid}(\cdot)\) prevents arbitrary resets. Updates must be masked so a modality changes only supported state components unless the learned coupling is explicitly documented.

### 8.3 Slow-parameter update

Slow parameters are updated using persistent, structured residual evidence rather than one frame:

\[
h^{\theta}_{i,t}
=
\operatorname{GRU}_\theta
(h^{\theta}_{i,t-1},
[r_t,\Delta r_t,\text{event},\text{observability features}]).
\]

\[
\theta_{i,t}
=
\operatorname{projectBounds}
\left(
\theta_{i,t-1}
+
g^\theta_{i,t}\odot\Delta\theta_{i,t}
\right).
\]

The gate must be small by default and event-dependent. Examples:

- restitution updates only after identifiable impacts;
- mass ratio updates only after interactions with observable momentum exchange;
- drag updates during sufficiently long free-flight segments;
- friction updates during observable surface contact/sliding;
- geometry updates only when visible.

### 8.4 Recovery and reinitialisation

A large innovation should not always be absorbed as continuous noise. The filter must classify likely causes:

- normal measurement noise;
- missed or incorrect association;
- discrete physical event;
- camera/calibration discontinuity;
- new object;
- model failure.

When confidence falls below configured thresholds, schedule a slow/global observation pass. Reinitialisation should be local to affected objects whenever possible.

## 9. Multiple hypotheses

The public state must allow:

\[
\mathcal B_t =
\{(w_t^h,B_t^h)\}_{h=1}^H.
\]

The first default configuration may use \(H=1\). The data structures and runtime should not assume this forever.

Future branches may represent:

- collision versus near miss;
- continued occlusion versus object removal;
- alternative depth interpretations;
- ambiguous identity assignments;
- alternate external actions.

Hypothesis likelihood is based on predicted measurement likelihood. Pruning, merging, and branching should be isolated in `belief/hypotheses.py`, not scattered across modalities.

---

# Part III — Canonical data and tensor contracts

## 10. General conventions

### 10.1 Batch, time, object order

Use explicit named comments and validation for dimensions:

- `B`: batch;
- `T`: time;
- `N`: maximum objects;
- `D`: feature/state dimension;
- `M`: measurements;
- `K`: modal components;
- `H`: hypotheses.

Public dataclasses store tensors in batch-major form. Sequence training tensors generally use `[B, T, ...]`. Online runtime beliefs generally use `[B, N, ...]`.

Do not silently infer whether a dimension is time or objects.

### 10.2 Numeric conventions

- floating state: `torch.float32`;
- categorical indices: `torch.int64`;
- masks: `torch.bool`;
- timestamps: float seconds in a monotonic timebase;
- quaternions: scalar-last `[x,y,z,w]` or scalar-first, chosen once and documented; use scalar-last for this project;
- angles: radians;
- positions: metres in world frame;
- linear velocity: metres/second;
- angular velocity: radians/second;
- mass: kilograms or dimensionless mass ratio, documented per dataset;
- image coordinates: normalised `[-1,1]` in projectors; raw pixels only at IO boundaries;
- covariance: variance, not standard deviation, internally represented as clamped log variance.

### 10.3 Coordinate frames

Every geometric observation and belief must identify a frame:

- `world`;
- `camera:<sensor_id>`;
- `object:<object_id>`;
- `body:<agent_id>`;
- arbitrary calibrated frames.

Transforms use homogeneous \(4\times4\) matrices or a typed `RigidTransform` with translation and quaternion. The transform convention must be tested. Prefer `T_target_from_source`.

### 10.4 Padding and masks

The model supports variable object counts using fixed `N_max` tensors plus masks in the first implementation.

Never treat padded objects as real objects. All losses, graph edges, attention, association, and metrics must mask them.

## 11. Required dataclasses

The following examples define intent. Exact import organisation may vary, but fields and semantics should remain.

### 11.1 `ObservationPacket`

```python
@dataclass(frozen=True)
class ObservationPacket:
    modality: str
    sensor_id: str
    timestamp: float
    payload: Any
    calibration: Mapping[str, Tensor | float | int | str]
    frame_id: str
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Requirements:

- immutable at runtime;
- payload validation delegated to modality;
- timestamp must be finite;
- no assumption of synchronous modalities;
- calibration version may be included in metadata.

### 11.2 `MeasurementSet`

```python
@dataclass
class MeasurementSet:
    modality: str
    sensor_id: str
    timestamp: Tensor                 # [B]
    values: Tensor                    # [B, M, Dm]
    log_variance: Tensor              # [B, M, Dm] or broadcastable
    existence_logits: Tensor          # [B, M]
    measurement_mask: Tensor          # [B, M]
    appearance: Tensor | None         # [B, M, Da]
    class_logits: Tensor | None
    frame_id: str
    supported_state_fields: tuple[str, ...]
    auxiliary: dict[str, Tensor]
```

Measurements are unordered proposals until association.

### 11.3 `ObjectBeliefTensor`

```python
@dataclass
class ObjectBeliefTensor:
    object_id: Tensor                 # [B, N], int64; -1 for padding
    active: Tensor                    # [B, N], bool
    existence_logit: Tensor           # [B, N]

    position: Tensor                  # [B, N, 3]
    velocity: Tensor                  # [B, N, 3]
    orientation: Tensor               # [B, N, 4]
    angular_velocity: Tensor          # [B, N, 3]

    geometry: Tensor                  # [B, N, Dg]
    appearance: Tensor                # [B, N, Da]
    residual_dynamics: Tensor         # [B, N, Dd]

    modal_state: Tensor               # [B, N, K, 2, Dm]
    modal_frequency: Tensor           # [B, N, K, Dm]
    modal_decay_raw: Tensor           # [B, N, K, Dm]

    log_mass: Tensor                  # [B, N, 1]
    restitution_logit: Tensor         # [B, N, 1]
    log_drag: Tensor                  # [B, N, 1]
    friction_logit: Tensor            # [B, N, 1]

    motion_mode_logits: Tensor        # [B, N, Cmode]
    visibility_logit: Tensor          # [B, N]
    age_steps: Tensor                 # [B, N]
    missed_steps: Tensor              # [B, N]

    fast_log_variance: Tensor         # [B, N, Dfast]
    slow_log_variance: Tensor         # [B, N, Dslow]

    parameter_memory: Tensor          # [B, N, Dh]
```

The code must provide packing/unpacking maps for `Dfast` and `Dslow`, with unit tests.

### 11.4 `CameraBelief`

```python
@dataclass
class CameraBelief:
    world_from_camera: Tensor         # [B, 4, 4]
    linear_velocity: Tensor           # [B, 3]
    angular_velocity: Tensor          # [B, 3]
    intrinsics: Tensor                # [B, 3, 3]
    log_variance: Tensor              # [B, Dcamera]
    calibrated: Tensor                # [B], bool
```

For the first synthetic milestone, camera calibration may be treated as known while camera pose is provided or estimated in a configurable mode. The contracts should support camera belief updates later.

### 11.5 `WorldBelief`

```python
@dataclass
class WorldBelief:
    timestamp: Tensor                 # [B]
    objects: ObjectBeliefTensor
    camera: CameraBelief
    gravity: Tensor                   # [B, 3]
    global_code: Tensor               # [B, Dglobal]
    global_log_variance: Tensor       # [B, Dglobal_var]
    next_object_id: Tensor            # [B]
    active_modalities: tuple[str, ...]
    metadata: dict[str, Any]
```

### 11.6 `AssociationResult`

```python
@dataclass
class AssociationResult:
    belief_indices: Tensor            # [B, P]
    measurement_indices: Tensor       # [B, P]
    pair_mask: Tensor                 # [B, P]
    pair_cost: Tensor                 # [B, P]
    unmatched_beliefs: Tensor         # [B, N]
    unmatched_measurements: Tensor    # [B, M]
    ambiguous: Tensor                 # [B, P]
```

### 11.7 `BeliefTrajectory`

```python
@dataclass
class BeliefTrajectory:
    timestamps: Tensor                # [B, T]
    positions: Tensor                 # [B, T, N, 3]
    velocities: Tensor                # [B, T, N, 3]
    orientations: Tensor              # [B, T, N, 4]
    motion_mode_logits: Tensor        # [B, T, N, Cmode]
    fast_log_variance: Tensor         # [B, T, N, Dfast]
    active_mask: Tensor               # [B, T, N]
    event_logits: Tensor | None
    auxiliary: dict[str, Tensor]
```

### 11.8 Predictive abstraction contracts

`AbstractionAssignment` stores a model-family kind, selection confidence,
complexity cost, refinement reason, and active mask for each `[B,N]` entity
slot.  Selection confidence is distinct from physical-state uncertainty.

`PredictiveTokenBatch` is a reversible derived view:

```python
@dataclass
class PredictiveTokenBatch:
    values: Tensor                    # [B,L,Dtoken]
    valid_mask: Tensor                # [B,L]
    token_type: Tensor                # [L]
    object_slot: Tensor               # [L], -1 for scene/camera
    object_id: Tensor                 # [B,L]
    abstraction_kind: Tensor          # [B,L]
    timestamp: Tensor                 # [B]
    next_object_id: Tensor            # [B]
    camera_calibrated: Tensor         # [B]
```

The initial token vocabulary is `SCENE`, `CAMERA`, `ENTITY_KINEMATIC`,
`ENTITY_PROGRAM`, and `ENTITY_LIFECYCLE`.  Tokenization must round-trip the
explicit belief exactly for a matching schema.  A future learned projection or
causal transformer may consume these tokens, but transformer outputs must be
decoded into typed proposals and assimilated through the existing
predict–observe–associate–innovate–correct loop.

## 12. Invariants

Implement runtime validators enabled in tests and optionally debug mode:

- active objects have nonnegative IDs;
- padded objects have ID `-1`;
- object IDs are unique within each batch element;
- timestamps never move backward in the online runtime;
- quaternion norms are close to one;
- log variances are finite and clamped;
- positions and velocities contain no NaN/Inf;
- probabilities/logits use masks correctly;
- association does not assign a measurement or belief twice;
- dynamics rollout does not mutate the input belief;
- observation modules do not mutate raw packets;
- the world belief is modality-independent;
- device and dtype remain consistent.


# Part IV — Multimodal observation architecture

## 13. Observation-module contract

The core runtime must not contain modality-specific branches such as `if rgb`, `if audio`, or `if skeleton` outside registration/configuration. Each modality implements a shared protocol.

A practical abstract interface:

```python
class ObservationModule(nn.Module, Protocol):
    modality_name: str

    def validate_packet(
        self,
        packet: ObservationPacket,
    ) -> None:
        """Raise a useful error for invalid payload/calibration."""

    def initialise_measurements(
        self,
        packets: Sequence[ObservationPacket],
        context: "ObservationContext",
    ) -> MeasurementSet:
        """Global/slow path used when no reliable prior exists."""

    def encode_measurements(
        self,
        packets: Sequence[ObservationPacket],
        prior: WorldBelief,
        predicted: "PredictedMeasurements",
        cache: "ModalityCache | None",
    ) -> tuple[MeasurementSet, "ModalityCache"]:
        """Fast residual-driven path used during normal online operation."""

    def project(
        self,
        belief: WorldBelief,
        sensor_context: "SensorContext",
    ) -> "PredictedMeasurements":
        """Map prior belief into expected modality-space measurements."""

    def innovation(
        self,
        measured: MeasurementSet,
        predicted: "PredictedMeasurements",
        association: AssociationResult,
    ) -> "InnovationSet":
        """Compute typed residuals, including wrapped/angular residuals."""

    def measurement_likelihood(
        self,
        innovation: "InnovationSet",
    ) -> Tensor:
        """Return log likelihood or compatible association score."""

    def training_losses(
        self,
        outputs: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
        masks: Mapping[str, Tensor],
    ) -> Mapping[str, Tensor]:
        """Return modality-specific supervised/self-supervised losses."""
```

The exact Python typing can use an abstract base class instead of a `Protocol`, but the responsibilities must remain separated.

### 13.1 Context objects

`ObservationContext` should include:

- timestamp;
- known camera/sensor calibration;
- maximum object count;
- optional predicted regions;
- device/dtype;
- training/evaluation flag.

`PredictedMeasurements` should include:

- predicted values;
- predicted measurement covariance;
- projected object IDs;
- visibility probability;
- projected spatial support/ROI;
- modality-specific auxiliary predictions;
- valid mask.

`InnovationSet` should include:

- residual values;
- whitened residual;
- innovation norm;
- pair mapping;
- masks;
- modality embedding/index;
- event/surprise features.

### 13.2 Registry

Create an explicit registry:

```python
OBSERVATION_MODULES: dict[str, type[ObservationModule]]
```

Registration may use a simple decorator or configuration map. Avoid a complex plugin packaging system initially. Loading a configured modality should require no edits to runtime logic.

### 13.3 Modality availability

The world model may run with:

- one modality;
- any subset of registered modalities;
- modalities that start or stop mid-episode;
- unequal rates;
- delayed observations if configured.

When no observation arrives, dynamics prediction still advances the belief and uncertainty.

### 13.4 Sensor-specific state

Persistent sensor caches belong to the modality runtime, not the `WorldBelief`, unless the cached quantity is itself a physical world estimate.

Examples that belong in modality cache:

- previous image feature pyramid;
- audio STFT overlap buffer;
- event-camera voxel accumulation;
- ROI feature cache.

Examples that belong in world belief:

- camera pose;
- acoustic source position;
- human joint pose;
- object appearance code used for association;
- calibrated sensor transform uncertainty.

## 14. Asynchronous scheduling

The runtime is event-driven by observation timestamps.

### 14.1 Event loop

For each incoming `ObservationPacket` or same-timestamp packet group:

1. validate timestamp;
2. propagate belief from current timestamp to packet timestamp;
3. project expected measurement for that sensor;
4. choose fast or slow observation path;
5. encode measurements;
6. associate;
7. compute innovation;
8. classify surprise/event;
9. correct fast state;
10. update lifecycle;
11. update slow parameters if observable;
12. cache diagnostics;
13. optionally produce a future rollout.

Pseudocode:

```python
def ingest(packet_group: Sequence[ObservationPacket]) -> WorldBelief:
    t = common_or_min_timestamp(packet_group)
    prior = dynamics.predict(current_belief, t - current_belief.timestamp)

    posterior = prior
    for packet in deterministic_modality_order(packet_group):
        module = registry[packet.modality]
        predicted = module.project(posterior, context_for(packet))
        measurements, cache = module.encode_measurements(
            [packet], posterior, predicted, cache_for(packet.sensor_id)
        )
        association = associator.match(posterior, measurements, predicted)
        innovation = module.innovation(measurements, predicted, association)
        posterior = updater.correct(
            posterior, measurements, predicted, association, innovation
        )
        posterior = lifecycle.update(posterior, measurements, association)
        posterior = identifier.update(
            posterior, innovation, association, observability(packet, posterior)
        )

    current_belief = posterior.with_timestamp(t)
    return current_belief
```

### 14.2 Same-timestamp fusion

Default deterministic order for the eventual multimodal system:

1. high-rate ego-motion/proprioception such as IMU;
2. direct structured pose/depth measurements;
3. RGB/object measurements;
4. audio/event evidence;
5. derived or semantic observations.

This order must be configurable. Tests should verify that same-timestamp order is deterministic.

### 14.3 Delayed observations

The first milestone may reject observations older than the current belief timestamp with a clear error. Preserve an interface for fixed-lag smoothing later. Do not implement an elaborate out-of-sequence filter initially.

## 15. RGB observation module

RGB is the first non-oracle modality. It should be deliberately small but architecturally real.

### 15.1 Two-path design

#### Slow/global path

Run:

- at initialisation;
- when no prior exists;
- on a configured cadence;
- when uncertainty or unexplained residual exceeds a threshold;
- when object birth is suspected;
- after association failure.

Responsibilities:

- extract a full-frame feature pyramid;
- propose object measurements;
- estimate existence;
- estimate image-space centre and apparent size;
- estimate depth or camera-space position;
- produce appearance embeddings;
- optionally produce coarse masks;
- detect unexplained objects.

#### Fast/residual path

Run on every ordinary frame:

- project each active object's expected image support;
- enlarge ROI according to uncertainty;
- crop or ROI-sample current feature maps;
- combine current crop with predicted soft mask/support, expected centre/size, previous cached feature, and optional frame-difference/flow features;
- predict a measurement correction and measurement uncertainty.

The fast path answers: “How does the observation differ from the prior prediction?” It should not rediscover the whole scene.

### 15.2 Initial RGB network

Use a lightweight pure-PyTorch CNN so the repository does not depend on a large pretrained model for the toy milestone.

Suggested global backbone:

- input: `[B, 3, H, W]`;
- four convolutional stages;
- channels approximately `[32, 64, 96, 128]`;
- stride 2 in stages 2–4;
- GroupNorm or LayerNorm rather than BatchNorm, because local batch sizes may be small;
- SiLU/GELU activations;
- optional feature pyramid projection to a common 64–96 channels.

Suggested global proposal head:

- `N_query = N_max + N_birth_extra` learned object queries;
- 2–3 small cross-attention layers over flattened low-resolution features, or a simpler spatial soft-argmax/heatmap detector;
- outputs per query:
  - existence logit;
  - normalised centre `(u,v)`;
  - log apparent radius/size;
  - depth or inverse depth;
  - measurement log variance;
  - appearance embedding;
  - optional mask coefficients.

A tiny DETR-like head is acceptable, but do not add a large transformer library. Implement a small `nn.MultiheadAttention` block if used.

Suggested fast ROI updater:

- `roi_align` equivalent implemented using `torch.nn.functional.grid_sample`, avoiding a hard dependency on compiled torchvision operators;
- ROI output around `16x16` or `24x24`;
- input channels include:
  - RGB crop features;
  - predicted soft support mask;
  - normalised coordinate channels;
  - optional previous-frame crop feature;
  - optional simple temporal difference;
- 3–4 small conv layers plus MLP;
- outputs:
  - delta centre;
  - delta log apparent size;
  - delta inverse depth;
  - existence/visibility correction;
  - appearance update gate;
  - log measurement variance;
  - innovation/event features.

### 15.3 Toy scene geometry inference

For spherical objects with known or estimated physical radius \(r_w\), calibrated focal length \(f\), and observed pixel radius \(r_p\), an analytic depth proposal is:

\[
z \approx \frac{f r_w}{r_p}.
\]

This single-frame proposal is evidence, not truth. Connected-component area
has heavy-tailed error under overlap, boundary truncation, and partial
occlusion even when its centre is subpixel accurate. The RGB module must expose
scale quality and anisotropic uncertainty using observable inputs such as
visible fraction, boundary contact, component ambiguity, temporal scale
consistency, and disagreement with the predicted measurement. Low-quality
scale may update image-plane point position without receiving the same depth
correction weight.

The scalable RGB representation is a persistent-ID multi-frame point/scale
trajectory measurement. Axis-local estimators should preserve predictable
motion and quantify their own evidence, while joint object/event context gates
departures caused by interactions. This is not a second world state: bounded
modality history produces a timestamped measurement and uncertainty, which
then follows normal association, innovation, and `WorldBelief` correction.
Constant or damped motion is a learnable low-complexity bias, not an
unconditional hardcoded rule.

Use this as a structured initial proposal. A learned bounded residual may correct rendering/perspective errors.

Back-project centre to camera coordinates:

\[
x = \frac{(u-c_x)z}{f_x},
\qquad
y = \frac{(v-c_y)z}{f_y}.
\]

Transform into world frame using camera calibration/belief.

If geometry radius is uncertain, propagate that uncertainty into depth. Do not allow the network to hide every ambiguity in an unconstrained world-state decoder.

### 15.4 Measurement vector for initial RGB module

Recommended canonical measurement per proposal:

\[
y^{rgb}
=
[u,\;v,\;\log r_p,\;\operatorname{invdepth},\;c_r,\;c_g,\;c_b],
\]

plus:

- existence logit;
- visibility logit;
- appearance embedding;
- optional coarse mask;
- diagonal log variance for geometric dimensions.

Colour values are only association cues in the toy data; they must not be treated as object IDs.

### 15.5 Predicted RGB measurements

The initial projector does not need to render photorealistic RGB. It predicts:

- projected centre;
- expected apparent radius;
- expected inverse depth;
- expected visibility/depth order;
- expected soft silhouette/support;
- expected appearance embedding;
- covariance in measurement coordinates.

This is sufficient for innovation-based filtering.

### 15.6 RGB losses

At proposal stage, use Hungarian matching to simulator objects during training. Losses:

- focal or BCE existence loss;
- L1/Huber centre loss;
- Huber log-size and inverse-depth loss;
- Gaussian NLL using predicted measurement variance;
- contrastive or cosine appearance consistency;
- optional Dice/BCE mask loss;
- visibility BCE;
- calibration regulariser.

At fast-update stage, train both measurement accuracy and downstream posterior/future improvement.

### 15.7 Initialisation

Use several initial frames when available, but do not require a fixed clip forever.

A practical initialisation procedure:

1. run global detector on first frame;
2. create provisional object beliefs;
3. on subsequent 2–4 frames, run prediction–association–correction;
4. estimate velocity from filtered position differences;
5. initialise slow parameters to priors with high uncertainty;
6. initialise modal states near zero;
7. mark the belief as `initialised` once enough state dimensions are observable.

The public API may expose `initialize(packets)` for convenience, but internally it should call the same `ingest` loop.

## 16. State/oracle observation module

Implement a small `StateObservationModule` for:

- dynamics/filter unit tests;
- debugging;
- isolating perception from dynamics;
- establishing upper bounds;
- training curriculum diagnostics.

It consumes simulator-derived noisy state measurements and projects directly from belief state.

Rules:

- it is not the main demo;
- acceptance of the project requires RGB mode;
- no core runtime path may depend on it;
- it must be clearly labelled `debug_oracle`;
- configs must make accidental use visible in logs and evaluation reports.

## 17. Future modality specifications

Do not implement all of these initially. Define documentation and stubs only when they are useful; avoid empty production classes. The core contracts must support them.

### 17.1 Depth

Measurements:

- per-object depth;
- point clusters;
- depth silhouettes;
- surface normals;
- uncertainty.

Projector:

- expected depth support and order;
- object surfaces or coarse geometry.

Depth should strongly update position/geometry, weakly update appearance, and not directly alter unrelated dynamics parameters.

### 17.2 Skeleton/body pose

Measurements:

- named joints;
- confidence per joint;
- joint covariance;
- bone lengths or kinematic constraints;
- root pose;
- optional contact labels.

World extension:

- `ArticulatedBelief` composed of persistent parts/joints;
- kinematic graph;
- joint angles/velocities;
- actuator/action latent.

A skeleton stream can be the sole modality. The core dynamics should treat it as structured object/part measurements.

### 17.3 Audio

Raw audio rates are much higher than world update rates. The audio module should maintain its own overlap buffer and emit timestamped acoustic measurements/events, not one world update per PCM sample.

Measurements may include:

- impact/event probability and time;
- source-direction estimate;
- source embedding;
- fundamental/modal frequency estimates;
- energy envelope;
- event uncertainty.

Projector may predict:

- event time distributions;
- source identity likelihood;
- coarse acoustic embedding conditioned on contact/material/event state.

Audio can reveal hidden collisions and material properties. It should update event state, source association, and slow material/restitution parameters more strongly than object position unless localisation is available.

### 17.4 IMU

Measurements:

- angular velocity;
- specific force;
- optionally magnetometer;
- sensor covariance.

Projector uses camera/body motion belief. IMU should be assimilated at high rate and is especially useful for ego-motion.

### 17.5 Optical flow

Measurements:

- sparse or dense flow;
- confidence;
- object-aligned flow summaries.

Projector derives expected image motion from object/camera state. Flow provides velocity evidence even when appearance detection is weak.

### 17.6 LiDAR/point clouds

Measurements:

- point sets or object clusters;
- pose/shape proposals;
- covariance.

Use set/point encoders inside the modality module. Do not move raw point tokens into the persistent world state.

### 17.7 Event cameras

The module accumulates events over short windows and emits motion/edge measurements with exact timestamps or micro-batches. The world scheduler remains event-driven.

## 18. Cross-modal consistency

When multiple modalities are available, train and evaluate consistency through the shared belief rather than forcing direct feature alignment everywhere.

Examples:

- a visible impact should cause a high predicted audio-event probability;
- an audio impact during visual occlusion should increase collision/event posterior;
- skeleton hand contact and RGB object motion should agree on interaction timing;
- IMU camera motion and RGB optical flow should agree;
- depth and monocular projected size should agree within uncertainty.

Potential loss:

\[
\mathcal L_{\text{cross}}
=
\sum_{m\ne n}
D\left(
Y^{(m)}_{\text{observed}},
H^{(m)}(U_n(B,Y^{(n)}))
\right),
\]

meaning that assimilating modality \(n\) should improve predictions in modality \(m\) when they describe the same event.

Do not implement a large all-to-all cross-modal transformer as the first fusion mechanism. The shared belief is the primary fusion bottleneck.

---

# Part V — Perception, association, and lifecycle

## 19. Residual-driven perception in detail

The perception design should minimise repeated work and focus compute where the current belief is uncertain.

### 19.1 Predicted regions

For each object, project its mean and uncertainty to image space. Expand the ROI by a configurable number of standard deviations:

\[
R_i =
\operatorname{bbox}
\left(
H(\mu_i)
\pm
\kappa\sqrt{\operatorname{diag}(J_i P_i J_i^\top)}
\right).
\]

Clamp ROIs to valid image bounds. Objects with high uncertainty receive larger ROIs.

### 19.2 Cached features

The RGB module may cache:

- previous feature pyramid;
- per-object pooled appearance feature;
- previous ROI coordinates;
- previous projected support;
- previous observation timestamp.

Invalidate caches on:

- device/config changes;
- global reinitialisation;
- large camera discontinuity;
- object death;
- sequence boundary.

Cache use must be optional and testable against a no-cache path.

### 19.3 Surprise map

Compute a coarse full-frame surprise signal even on the fast path:

- predicted soft silhouettes or occupancy;
- current low-resolution objectness/feature map;
- discrepancy map;
- unexplained high-confidence regions.

Use it to trigger global discovery and to avoid tunnel vision when objects enter outside predicted ROIs.

### 19.4 Observation scheduling

A simple scheduler chooses:

- `FAST_ROI`;
- `GLOBAL_DISCOVERY`;
- `RECOVERY`;
- `SKIP` for a modality.

Inputs:

- time since last global pass;
- maximum object covariance;
- unexplained surprise;
- association failures;
- birth probability;
- user-configured compute budget.

Initially implement deterministic thresholds. A learned active-perception policy is future work.

## 20. Data association

Association is explicit and independent of the RGB network.

### 20.1 Cost matrix

For prior object \(i\) and measurement \(j\):

\[
C_{ij}
=
w_g C_{ij}^{\text{geometry}}
+
w_a C_{ij}^{\text{appearance}}
+
w_m C_{ij}^{\text{motion}}
+
w_c C_{ij}^{\text{class}}
+
w_e C_{ij}^{\text{existence}}.
\]

Suggested components:

- geometry: Mahalanobis distance in measurement space;
- appearance: \(1-\) cosine similarity;
- motion: discrepancy between expected and measured velocity/flow;
- class: negative compatible class probability;
- existence: penalty for low measurement confidence.

Gate impossible pairs before Hungarian assignment:

\[
C_{ij}=\infty
\quad\text{if}\quad
r_{ij}^\top S_{ij}^{-1}r_{ij}>\tau.
\]

Use `scipy.optimize.linear_sum_assignment` for the initial CPU association because object counts are small. Keep tensors transferred minimally. A batched GPU matcher is unnecessary initially.

### 20.2 Ambiguity

Record ambiguity when the best and second-best costs are close. The first implementation can retain the best assignment but:

- reduce correction confidence;
- avoid aggressive appearance/slow-parameter updates;
- log ambiguity;
- optionally trigger a global pass.

This creates a clean path to multiple-hypothesis association later.

### 20.3 Birth

An unmatched high-confidence measurement becomes a tentative object.

Lifecycle:

1. `TENTATIVE`;
2. confirmed after `birth_confirmations` consistent detections;
3. assigned a permanent monotonic ID;
4. initial covariance reflects measurement uncertainty;
5. initial velocity covariance is high unless multiple measurements establish velocity.

For the toy simulator, objects may all exist from frame one initially. Still implement the lifecycle rather than hard-coding fixed identities.

### 20.4 Miss and occlusion

An unmatched belief is not immediately removed.

- increment `missed_steps`;
- lower visibility;
- propagate dynamics and uncertainty;
- retain identity through configured occlusion duration;
- distinguish projected out-of-view from unexplained disappearance;
- do not update appearance or slow parameters from a miss.

### 20.5 Death/removal

Deactivate an object when:

- existence probability falls below threshold for enough time;
- it exits the known scene bounds with consistent motion;
- a configured removal event is observed;
- maximum miss duration is exceeded.

Do not recycle IDs within an episode.

### 20.6 Merge/split

The initial sphere scene does not require physical merge/split. The interfaces should permit a measurement to be marked `compound` or ambiguous. Do not invent a complex merge/split system before it is needed.

## 21. Object identity and appearance

Appearance exists to aid association and sensor prediction, not to become the physical state.

Use an appearance embedding with:

- L2 normalisation;
- uncertainty or update gate;
- exponential moving average after confident associations;
- no update during occlusion/ambiguity;
- optional supervised colour/texture reconstruction in toy scenes.

Object identity must not be encoded solely as a one-hot colour. Randomise colours and repeat similar colours in harder validation subsets.

## 22. Camera motion

The initial simulator includes camera motion to avoid learning image-space dynamics as world physics.

Two supported configurations:

1. **known camera pose**: calibration and world-from-camera transform are provided to the RGB module;
2. **estimated camera pose**: future extension or optional milestone using visual/IMU measurements.

Milestone 1 may use known camera pose, but all measurements must still pass through explicit coordinate transforms. Never silently assume a static camera.

The training data should include:

- translations;
- gentle rotations/orbits;
- focal length/zoom variation only if calibration supplies it;
- held-out trajectories.

## 23. Measurement-space prediction versus pixel generation

The core model predicts structured measurements, not future RGB pixels.

Benefits:

- dynamics errors remain interpretable;
- the renderer cannot conceal incorrect state;
- training is cheaper;
- modality modules remain replaceable;
- online correction uses well-defined residuals.

An optional visualisation renderer can draw predicted objects into frames. A learned photorealistic decoder is not required for Milestone 1.


# Part VI — Dynamics architecture

## 24. Required dynamics-module separation

Use the following layers:

1. `AnalyticKinematics`: deterministic timestamp-aware integration;
2. `ModalDynamics`: stable continuous latent modes;
3. `InteractionGraph`: pairwise and global learned residuals;
4. `EventModel`: contact/event probabilities and structured jumps;
5. `UncertaintyDynamics`: process noise and variance propagation;
6. `RolloutEngine`: substeps, event ordering, output sampling.

Do not implement a single opaque `DynamicsMLP` that maps packed state to next packed state.

## 25. Analytic kinematics

The low-complexity prior for each translational axis is constant or
parameter-damped velocity plus known external acceleration. This is an
inductive bias, not a hard forecast rule. Learned modal/residual dynamics and
explicit events may revise an axis when observations, interactions, or scene
context provide evidence.

Training and evaluation must expose position and velocity errors separately
for x, y, and z as well as jointly. The model should be able to learn
axis-local regularities while retaining cross-axis and pairwise context for
event gates. For example, a contact detected from full 3-D geometry may open a
transition that changes one or more velocity components; absent such evidence,
the learned residual should not needlessly rewrite predictable inertial
motion.

Zero-output initialization alone does not preserve that bias after many
optimizer updates. Learned acceleration residuals must therefore support an
explicit, protocol-bound parsimony objective or evidence gate. A first
implementation may penalize the mean squared L2 energy of identically treated
x/y/z typed decoder rows, with per-axis diagnostics. This is a soft training
prior, not an axis-specific runtime rule: all axes retain the same capacity,
and multistep/event evidence may pay the complexity cost when non-inertial
dynamics improve held-out predictions. Historical configurations that omit
the exact objective weight must retain zero contribution.

Parameter energy is only a proxy for the acceleration used by a rollout. If a
fixed-manifest ablation shows that suppressing a complete typed node branch
repairs prediction while decoder-row parsimony does not, training must expose
an optional functional activity objective over the bounded node acceleration
actually emitted for active objects across every causal rollout invocation.
Pool squared acceleration sums and active-object counts before taking the
per-axis mean, then average the three identically treated axes. Padding and
variable object counts must not change the objective. The statistic may
regularize training and provide per-axis diagnostics, but must not become
persistent belief state, an inference-time gate, or a hardcoded constant-
velocity rule. Omission of its exact loss-weight key contributes exactly zero.

Functional diagnostics must decompose mean squared activity into squared mean
drift and residual variation for each axis. If broad evidence localizes a
failure to nearly context-invariant acceleration, prefer the squared-mean
drift objective over total activity: it penalizes a learned scene-wide force
while allowing balanced object-, relation-, and event-conditioned residuals
to vary. This remains a soft, identically axis-treated prior. A genuinely
unmodelled constant force may still be learned when its held-out forecast gain
outweighs the cost. Drift and activity are separate exact opt-in loss terms;
historical configs contribute neither by default.

### 25.1 Time handling

Every call accepts a real \(\Delta t\), not an integer frame count. Support irregular observation intervals.

For stability, split long intervals:

```python
num_substeps = ceil(dt / max_substep)
sub_dt = dt / num_substeps
```

The default toy simulator and model may use 30 Hz observations with dynamics substeps around 1/120 s.

### 25.2 Gravity

Store gravity in `WorldBelief`. It may be known initially and later estimated globally.

Apply gravity only to objects whose mode allows it. Sleeping/contact modes may use constrained handling.

### 25.3 Drag and damping

Use a stable exponential form where practical:

\[
v(t+\Delta t)
=
v(t)\exp(-c\Delta t)
+
\frac{g}{c}(1-\exp(-c\Delta t))
\]

for positive \(c\), with a numerically safe small-\(c\) branch. A simpler semi-implicit Euler implementation is acceptable initially if tested over the configured range.

### 25.4 Orientation

Even if toy spheres do not expose orientation, implement:

- quaternion multiplication;
- exponential-map update;
- normalisation;
- geodesic orientation loss;
- tests for zero and small angular velocity.

This avoids redesign when non-spherical bodies are added.

### 25.5 Boundaries

The model knows coarse scene boundary geometry through global/environment state or configured planes. It should not hard-code image boundaries as physical walls.

For initial sphere–plane contacts:

- compute signed distance;
- derive normal;
- predict contact/event;
- apply restitution and tangential damping/friction;
- correct penetration conservatively.

## 26. Modal dynamics design

### 26.1 Purpose

The modal bank captures coherent residual temporal structure that is awkward to express as simple position/velocity integration:

- oscillation;
- damping;
- periodic actuation;
- slowly varying latent forces;
- deformable modes in future extensions;
- model residuals that should remain temporally coherent.

It should not replace explicit position and velocity.

### 26.2 State

For each object:

- `K_modal`: default 4 in toy;
- per mode paired state `[2, D_modal]`;
- positive frequency, bounded to a configured range;
- nonnegative decay;
- readout to residual acceleration and optional event/context features.

Initialisation:

- state near zero;
- frequencies drawn or predicted from object code;
- decay biased toward stable;
- readout weights small.

### 26.3 Update

Implement the rotation–decay transform analytically and vectorised. Avoid constructing large dense matrices.

For state components \(x,y\):

```python
angle = frequency * dt
decay = exp(-softplus(decay_raw) * dt)
x_new = decay * (cos(angle) * x - sin(angle) * y)
y_new = decay * (sin(angle) * x + cos(angle) * y)
```

### 26.4 Correction

The belief updater may correct modal phase/amplitude when persistent innovation indicates a coherent mismatch. Use strong regularisation so a single noisy frame does not arbitrarily rewrite all modes.

### 26.5 Stability tests

Test:

- no growth for positive decay;
- exact identity at `dt=0`;
- approximate composition:
  \(F(F(z,dt_1),dt_2)\approx F(z,dt_1+dt_2)\);
- gradients finite on CPU, MPS, and CUDA where available;
- long rollout remains finite.

## 27. Interaction graph

### 27.1 Graph construction

For small `N_max`, use a dense pair mask and vectorised pairwise differences. Avoid an external graph library.

Candidate edge mask:

- both objects active;
- `i != j`;
- distance below configured interaction radius plus uncertainty margin;
- optionally top-k nearest neighbours.

For `N<=10`, dense \(N^2\) is simple and cheap.

### 27.2 Edge network

Suggested toy architecture:

- input edge feature 32–96 dimensions;
- 2–3 layer MLP, hidden 64–128;
- LayerNorm;
- SiLU;
- outputs:
  - contact logit;
  - collision logit;
  - scalar normal impulse residual;
  - tangential damping/friction residual;
  - residual force vector in an invariant local basis;
  - edge process-noise contribution.

Use relative vectors and scalar invariants. Avoid feeding absolute world position to pairwise networks except through clearly justified environment features.

### 27.3 Symmetry and conservation bias

For pairwise interaction:

- compute one unordered pair output where practical;
- apply equal and opposite force/impulse;
- divide by learned/estimated mass;
- use normal/tangent basis from relative geometry;
- ensure swapping object order produces consistent transformed output.

Add tests for permutation equivariance and action–reaction symmetry.

### 27.4 Node/global network

Aggregate edge messages with masked sum or mean. A small node MLP combines:

- self state;
- modal readout;
- aggregated messages;
- global/environment code;
- current mode;
- uncertainty.

It outputs bounded residual acceleration and mode transition features.

### 27.5 Ground/environment node

Represent the ground and known static boundaries as explicit environment contacts or special fixed nodes, rather than arbitrary hidden bias. For the initial implementation, a dedicated sphere–plane contact path is simpler than adding all planes as graph nodes, but keep the interface compatible.

## 28. Collision/event model

### 28.1 Event probability

Predict event logits before applying a jump. Supervise against simulator contact/event labels.

Features:

- signed gap;
- relative normal velocity;
- tangential velocity;
- radius/geometry;
- restitution/friction beliefs;
- uncertainty;
- previous mode;
- residual/surprise history.

### 28.2 Analytic impulse proposal

For a collision normal \(n\), relative normal velocity:

\[
v_{rel,n}
=
(v_j-v_i)\cdot n.
\]

For approaching bodies, analytic impulse magnitude:

\[
j_{\text{analytic}}
=
-\frac{(1+e)v_{rel,n}}
{1/m_i+1/m_j}.
\]

Use estimated restitution and mass. Clamp to nonnegative and apply only when event/contact conditions hold.

The network predicts:

- a bounded multiplicative factor;
- a bounded additive residual;
- event confidence.

For example:

\[
j =
\operatorname{softplus}
\left(
j_{\text{analytic}}(1+0.25\tanh a)
+
0.1\,\operatorname{softplus}(b)
\right).
\]

Exact scales belong in config and should be normalised by dataset units.

### 28.3 Penetration handling

Prediction errors can create overlap. Use a small positional projection along the normal, split by inverse mass, with a maximum correction per substep. Do not train a network to tolerate arbitrarily deep penetration.

Log penetration statistics as a physics diagnostic.

### 28.4 Event timing

A fixed substep model is sufficient initially. Future continuous collision detection may refine event time. Use small enough substeps that the toy model does not tunnel through objects.

### 28.5 Sleeping

An object can enter sleeping mode if:

- speed below threshold;
- near ground/contact;
- no significant force/event for a configured duration.

Wake it on collision, external force, or high innovation.

## 29. Uncertainty dynamics

### 29.1 Process noise

Process-noise increment should be nonnegative and depend on:

- elapsed time;
- speed;
- event/contact likelihood;
- occlusion duration;
- interaction density;
- residual dynamics magnitude;
- model confidence.

Parameterise with `softplus` and clamp.

### 29.2 Variance propagation

At minimum, propagate position/velocity uncertainty using:

\[
\operatorname{Var}(p')
\approx
\operatorname{Var}(p)
+
\Delta t^2\operatorname{Var}(v)
+
Q_p,
\]

\[
\operatorname{Var}(v')
\approx
\operatorname{Var}(v)
+
Q_v.
\]

Include covariance cross terms only when the representation is upgraded beyond diagonal.

### 29.3 Event uncertainty

Around ambiguous collision timing, uncertainty should expand rather than outputting an overconfident average. Event entropy can add process noise.

### 29.4 Calibration constraints

Clamp log variance to a configured finite range. Penalise both overconfidence and pathological inflation through Gaussian NLL and calibration metrics.

## 30. Rollout engine

### 30.1 Interface

```python
class DynamicsModel(nn.Module):
    def predict(
        self,
        belief: WorldBelief,
        dt: Tensor | float,
    ) -> WorldBelief:
        ...

    def rollout(
        self,
        belief: WorldBelief,
        query_times: Tensor,
        *,
        return_events: bool = True,
    ) -> BeliefTrajectory:
        ...
```

`rollout` must not mutate `belief`.

### 30.2 Arbitrary query times

The model should return predictions at arbitrary sorted future times. Internally substep as needed. This allows:

- irregular sensor updates;
- random-access future queries;
- coarse-to-fine planning;
- fair evaluation across frame rates.

### 30.3 Receding horizon

The runtime does not store one fixed DCT trajectory as truth. It stores the current belief and regenerates a cheap future trajectory when requested.

After a new observation, future rollout starts from the corrected posterior. No full history re-encoding is required.

### 30.4 Rollout cache

A future optimisation may cache deterministic rollouts and invalidate only affected objects after correction. Do not implement complex partial invalidation in Milestone 1. Keep a clear cache boundary.

## 31. Dynamics baselines

Implement simple baselines for evaluation, not as competing repository architectures:

- static position;
- constant velocity;
- analytic gravity/drag without learned interactions;
- oracle-parameter analytic dynamics;
- optionally fixed-window DCT predictor later.

All baselines should consume the same dataset/evaluation contracts.

---

# Part VII — Online system identification

## 32. Purpose

System identification estimates slowly changing or constant physical properties from prediction residuals. It is distinct from correcting instantaneous pose/velocity.

A one-frame position error usually provides strong evidence about state, weak evidence about drag, and almost no evidence about mass. The updater must encode this distinction.

## 33. Parameter beliefs

Initial toy parameters:

- object radius/geometry;
- log mass or mass ratio;
- restitution;
- drag;
- ground friction/tangential damping.

For each parameter maintain:

- constrained mean;
- log variance;
- prior;
- observability score;
- last-update timestamp;
- update count.

Parameter transformations:

- mass, drag, radius: `exp` or `softplus`;
- restitution, friction: sigmoid into configured bounds;
- unconstrained latent codes: direct with norm regularisation.

## 34. Identifiability and observability

Do not report a parameter as learned when the episode does not identify it.

Examples:

- absolute mass is not observable from isolated ballistic motion under mass-independent gravity;
- restitution is primarily observable from impacts;
- friction is observable during sliding/contact;
- drag needs sustained motion relative to medium;
- geometry/depth are confounded in monocular images without size priors or motion cues.

Implement per-parameter observability gates:

```python
@dataclass
class Observability:
    mass_ratio: Tensor
    restitution: Tensor
    drag: Tensor
    friction: Tensor
    geometry: Tensor
```

Gates derive from events, visibility, motion magnitude, association confidence, and measurement geometry.

Evaluation should condition parameter metrics on adequate observability.

## 35. Parameter updater

Use a small recurrent accumulator per object:

- input: whitened innovations, derivative/EMA of innovations, event logits/labels during training, current state, current parameter belief, observability;
- recurrent state: `parameter_memory`;
- output: bounded parameter delta, variance delta, evidence gate.

A GRU cell with hidden dimension 32–64 is enough.

Update rule:

```python
raw_delta, raw_var_delta, evidence = updater(features, memory)
gate = sigmoid(evidence) * observability
theta_new = project_bounds(theta + slow_lr * gate * tanh(raw_delta))
logvar_new = clamp(logvar + gate * tanh(raw_var_delta), min_lv, max_lv)
```

The online update is part of inference and uses no gradient step.

## 36. Optional local optimisation

After the learned updater works, an optional small differentiable optimiser may refine a few parameters over a recent fixed window:

\[
\theta^\star
=
\arg\min_\theta
\sum_{\tau=t-W}^{t}
\left\|
Y_\tau-H(F_\theta(B_{t-W},\tau))
\right\|_{R^{-1}}^2
+
\lambda\|\theta-\theta_{\text{prior}}\|^2.
\]

Constraints:

- only small parameter vectors;
- fixed small iteration count;
- disabled by default on the MPS toy path;
- no optimisation of full network weights;
- benchmark the cost and improvement.

Do not block Milestone 1 on Gauss–Newton or second-order methods.

## 37. Parameter-supervision strategy

Synthetic data provides exact parameters. Use:

- direct parameter regression/NLL;
- downstream rollout loss;
- consistency across prefixes of the same episode;
- intervention/generalisation tests.

Prefix consistency:

\[
\mathcal L_{\theta\text{-consistency}}
=
\|\hat\theta_{0:t_1}-\hat\theta_{0:t_2}\|^2
\]

for parameters expected to remain constant, weighted by posterior confidence and observability.

## 38. Counterfactual validation

A true system-identification representation should support changed conditions.

Evaluate by:

1. infer object state/parameters from a prefix;
2. replace gravity, initial velocity, restitution, or scene boundary;
3. roll out;
4. compare with simulator counterfactual.

This is stronger evidence than reconstructing the original trajectory.

The first implementation may include a small counterfactual evaluation script after the main online loop works.

---

# Part VIII — Belief correction and surprise handling

## 39. Fast-state corrector

### 39.1 Inputs

Per associated object/measurement pair:

- prior packed fast state;
- prior log variance;
- measured vector;
- predicted measurement;
- raw innovation;
- whitened innovation;
- measurement log variance;
- visibility/existence;
- motion-mode probabilities;
- time since last observation;
- modality embedding;
- association cost/ambiguity;
- event features.

### 39.2 Outputs

- corrected state residual;
- posterior log-variance residual;
- state-component gate;
- event-mode logit residual;
- visibility/existence update;
- optional modal-state correction.

### 39.3 Structure

Use an MLP or small gated residual block. Do not use a large sequence transformer for one update.

Suggested hidden size 128, 3 layers, LayerNorm/SiLU.

### 39.4 State constraints

After correction:

- normalise quaternion;
- clamp reasonable parameter/state ranges from config;
- resolve severe physical penetration;
- ensure finite variances;
- apply masks.

### 39.5 Correction sparsity

Penalise unnecessary large corrections:

\[
\mathcal L_{\text{correction}}
=
\sum_d
\frac{|\Delta\mu_d|}
{\sqrt{\sigma_d^{-2}+\epsilon}}.
\]

This encourages the dynamics to explain predictable motion while allowing correction when evidence demands it.

## 40. Innovation classification

A small classifier labels innovation cause probabilities:

- `NOISE`;
- `STATE_DRIFT`;
- `PHYSICAL_EVENT`;
- `ASSOCIATION_ERROR`;
- `NEW_OBJECT`;
- `CAMERA_OR_SENSOR_SHIFT`;
- `UNKNOWN_MODEL_ERROR`.

Toy supervision can derive some labels from simulator events and injected corruptions. The classifier controls:

- whether to update state;
- whether to update slow parameters;
- whether to trigger discovery;
- whether to increase uncertainty;
- whether to reject association.

Avoid making this classifier a hard single point of failure; use probabilities/gates.

## 41. Robust updates

Use robust innovation clipping or Huber-like influence. A single bad detection must not catastrophically move the belief.

Possible robust factor:

\[
w(r)=\min\left(1,\frac{c}{\|r_{\text{white}}\|+\epsilon}\right).
\]

The model may learn a confidence gate, but keep a deterministic cap as a safety measure.

## 42. Unmatched measurements and beliefs

### 42.1 Unmatched measurements

Send to lifecycle birth logic and surprise map. Do not force association.

### 42.2 Unmatched beliefs

Apply missed-observation update:

- lower visibility;
- slightly lower existence depending on projected visibility;
- increase uncertainty;
- keep dynamics prediction;
- schedule global search if expected visible.

### 42.3 Occluded objects

If depth ordering/projected overlap predicts occlusion, a miss should not strongly reduce existence. This is a core reason to use a world-space model.

## 43. Smoothing

Normal runtime is filtering, not retrospective smoothing. A fixed-lag smoother can be a future module for training labels or offline evaluation. Do not mix future observations into online metrics.

---

# Part IX — Uncertainty and hypothesis quality

## 44. Initial uncertainty representation

Use diagonal Gaussian variance for:

- position;
- velocity;
- orientation tangent approximation;
- angular velocity;
- selected modal components;
- slow parameters;
- camera state if estimated.

Use categorical distributions for:

- motion mode;
- existence;
- visibility;
- event.

Document that quaternion uncertainty is an approximation in the first version.

## 45. Likelihoods

Use Gaussian NLL for continuous targets:

\[
\mathcal L_{\text{NLL}}
=
\frac12
\left[
\frac{(y-\mu)^2}{\sigma^2}
+
\log\sigma^2
\right].
\]

Clamp variance and average only over valid masks.

Use BCE/focal loss for existence/visibility/events and cross entropy for mutually exclusive modes.

## 46. Calibration metrics

Report:

- empirical coverage at 50%, 80%, 95%;
- expected calibration error for categorical events;
- NLL;
- sharpness/average predicted standard deviation;
- error versus predicted uncertainty correlation;
- calibration by horizon;
- calibration under occlusion and collision separately.

A model with lower RMSE but severe overconfidence is not considered strictly better.

## 47. Hypothesis interface

`HypothesisSet` should support:

```python
@dataclass
class HypothesisSet:
    beliefs: list[WorldBelief]
    log_weights: Tensor
```

Default `H=1`. Later:

- branch on ambiguous associations/events;
- reweight using measurement likelihood;
- prune low weights;
- merge close beliefs.

Do not contaminate basic tensor contracts with an extra hypothesis dimension prematurely; wrap beliefs at the runtime level.


# Part X — Training methodology

## 48. Training objective

Train the complete causal loop, not isolated one-step modules only.

A training episode alternates:

\[
\text{observe}
\rightarrow
\text{predict}
\rightarrow
\text{observe}
\rightarrow
\text{correct}
\rightarrow
\text{roll out}
\rightarrow
\cdots
\]

The model must encounter its own imperfect beliefs during training.

## 49. Episode structure

For each sampled synthetic episode:

1. generate simulator trajectory, RGB frames, calibration, states, parameters, identities, visibility, and events;
2. select an initial observation prefix;
3. initialise belief using RGB global path;
4. run online updates over a sequence;
5. at random update points, request future rollouts of random horizons;
6. compute posterior, rollout, event, association, parameter, and calibration losses;
7. backpropagate through a bounded unroll window;
8. detach belief/history between truncated windows while preserving numerical state.

Suggested first config:

- simulator frames: 64–96;
- image size: 96×96;
- frame rate: 30 Hz;
- objects: 3–6 initially, random up to 8 in validation;
- initialisation observations: 3–5;
- training unroll: 16–32 updates;
- rollout query horizons: 0.1, 0.25, 0.5, 1.0, and optionally 2.0 seconds;
- batch size on MPS: start 4–8 and make configurable.

## 50. Training stages within one architecture

A curriculum may alter data difficulty and which losses are enabled, but must not replace the model with disposable architectures.

### Stage A — Component sanity and oracle-assisted debugging

Purpose:

- verify dynamics, filter, tensor contracts, and losses;
- use state/oracle measurements;
- no claim of final success.

Duration should be short. The code remains part of tests/baselines.

### Stage B — RGB measurement pretraining

Train global and fast RGB measurement heads using simulator labels.

- random single frames and short pairs;
- object proposal matching;
- measurement uncertainty;
- appearance consistency;
- visibility/masks.

This may run independently to stabilise perception, but the same module is used end-to-end.

### Stage C — Oracle-to-RGB mixed online training

Run complete loop while randomly selecting oracle or RGB measurements with an annealed oracle probability. This is a training technique, not a separate runtime architecture.

- start with enough oracle observations to stabilise dynamics/filter learning;
- reduce oracle probability toward zero;
- final validation is RGB-only.

### Stage D — Full RGB closed-loop training

All online corrections use RGB. Continue to supervise latent states, events, and parameters from simulator ground truth.

### Stage E — Harder distribution

Increase within the same simulator/config family:

- more objects;
- more similar appearances;
- longer occlusions;
- stronger camera motion;
- parameter ranges;
- irregular frame drops;
- longer horizons;
- mild sensor noise.

Introduce compound or unfamiliar physical regimes only after the familiar
reference regime passes simulator invariants and qualitative trajectory
inspection. Report per-scenario results so gains on an unusual regime cannot
hide regression on the interpretable baseline.

The preferred result across a scenario curriculum is one shared checkpoint and
one persistent runtime architecture. Scenario-specific checkpoints are
diagnostic ablations only unless the deployed system includes an explicit,
observation-derived regime router. A balanced shared run must record the
ordered scenario mixture and report the aggregate result together with every
scenario slice.

The project must not stop at Stage A or B.

### 50.1 Dataset-size and capacity scaling

The tiny profiles are debugging instruments, not evidence of generalization.
A generalization run should use thousands of deterministic seeded episodes,
hundreds of disjoint validation/test episodes, and balanced coverage of every
declared scenario family. Continuous initial conditions, physical parameters,
camera paths, object counts, appearances, event timing, and observation noise
must vary within each family. Report episode draws and approximate dataset
passes as well as optimizer steps.

Increase model capacity behind the frozen contracts: perception width,
appearance/residual state, interaction/event networks, filter capacity, and
parameter memory may grow, while `WorldBelief`, timestamped observations,
association, innovation, correction, and rollout APIs remain unchanged. Use
one shared checkpoint. Capacity comparisons must keep seed manifests and
evaluation semantics fixed, and a larger model is promoted only by disjoint
RGB-only multistep validation and test—not training loss.

On-the-fly datasets should avoid caching thousands of rendered episodes in
memory. Shuffle deterministic seed manifests and resample frame/window
positions between passes. Bound retained closed-loop graphs with microbatches
and short TBPTT windows; overlap deterministic rendering using a tested,
configurable worker count. A full-scale schedule may be handed off to MPS or a
single CUDA GPU when local CPU throughput makes completion impractical, but a
bounded two-phase smoke run is still required on the current machine.

## 51. Perturbation/recovery training

At random times, corrupt the belief before an observation:

- position offset;
- velocity offset;
- depth error;
- covariance miscalibration;
- wrong event mode;
- missed collision;
- parameter bias;
- appearance drift;
- simulated association ambiguity;
- camera pose perturbation where supported.

Then reveal a correct RGB observation and train the system to recover.

Perturbation magnitudes should be sampled relative to state scales and curriculum difficulty.

Key downstream loss:

\[
\mathcal L_{\text{recovery-future}}
=
\sum_{\tau\in\mathcal H}
w_\tau
d\left(
\hat S_{t+\tau}^{\text{post-correction}},
S_{t+\tau}^{*}
\right).
\]

This prevents the updater from merely matching the current frame while damaging velocity or parameters.

## 52. Teacher forcing policy

Ground-truth states are supervision, not ordinary inputs.

During full closed-loop training:

- dynamics starts from model posterior;
- next correction consumes RGB-derived measurements;
- ground truth is used for losses and optional scheduled debug/oracle measurement;
- do not reset belief to exact state every frame.

Use teacher forcing only for targeted component pretraining or an explicitly logged curriculum mode.

## 53. Losses

Define a `LossTerms` mapping and log every component separately.

### 53.1 Current posterior state loss

\[
\mathcal L_{\text{state}}
=
\lambda_p \operatorname{Huber}(\hat p,p^*)
+
\lambda_v \operatorname{Huber}(\hat v,v^*)
+
\lambda_q d_{SO(3)}(\hat q,q^*)^2
+
\lambda_\omega \operatorname{Huber}(\hat\omega,\omega^*).
\]

Mask unsupported sphere orientation terms in toy data.

### 53.2 Future rollout loss

For query horizons:

\[
\mathcal L_{\text{rollout}}
=
\sum_{\tau}
w_\tau
\left[
\lambda_p d_p(\hat p_{t+\tau},p^*_{t+\tau})
+
\lambda_v d_v(\hat v_{t+\tau},v^*_{t+\tau})
\right].
\]

Use increasing or balanced horizon weights rather than letting numerous short steps dominate.

### 53.3 Measurement loss

Global/fast RGB measurement losses described earlier, including Gaussian NLL.

### 53.4 Event loss

- focal/BCE collision/contact event loss;
- mode cross entropy;
- event-time tolerance metric/loss if events are frame-aligned;
- higher weight for rare collisions.

### 53.5 Parameter loss

\[
\mathcal L_{\text{param}}
=
\sum_k
o_k
\left[
\frac{(\hat\theta_k-\theta_k^*)^2}{\sigma_k^2}
+
\log\sigma_k^2
\right],
\]

where \(o_k\) is observability.

### 53.6 Existence/visibility loss

BCE or focal loss with lifecycle masks.

### 53.7 Association/appearance loss

- Hungarian proposal supervision;
- contrastive appearance embedding;
- optional pairwise association classification;
- no direct loss on arbitrary persistent numeric IDs.

### 53.8 Uncertainty loss

- Gaussian NLL;
- categorical calibration regularisation if needed;
- variance floor/ceiling penalties;
- optional coverage penalty.

### 53.9 Physics regularisers

Use only where assumptions apply:

- penetration penalty;
- pairwise action–reaction consistency;
- quaternion norm;
- modal stability;
- energy/momentum consistency around isolated elastic collisions;
- bounded acceleration/impulse;
- sleep/contact consistency.

Do not enforce global energy conservation in scenes with drag, inelastic contact, or external actuation.

### 53.10 Correction penalty

Penalise unjustified large state and parameter corrections, scaled by uncertainty and observability.

### 53.11 Programme consistency

Different prefixes of the same episode should infer compatible slow parameters and dynamics codes:

\[
\mathcal L_{\text{programme}}
=
\sum_{t_1<t_2}
\|\theta_{t_1}-\operatorname{stopgrad}(\theta_{t_2})\|^2
\]

where the later estimate is adequately observable/confident.

### 53.12 Total loss

Example:

\[
\mathcal L =
w_s\mathcal L_{\text{state}}
+w_r\mathcal L_{\text{rollout}}
+w_m\mathcal L_{\text{measurement}}
+w_e\mathcal L_{\text{event}}
+w_\theta\mathcal L_{\text{param}}
+w_x\mathcal L_{\text{exist}}
+w_a\mathcal L_{\text{association}}
+w_u\mathcal L_{\text{uncertainty}}
+w_p\mathcal L_{\text{physics}}
+w_c\mathcal L_{\text{correction}}
+w_g\mathcal L_{\text{programme}}.
\]

All weights live in YAML and are printed in run metadata.

## 54. Matching ground truth to beliefs

Because persistent belief order is not ground-truth order, metrics/losses should match active beliefs to simulator objects by known association during training or by Hungarian state distance during evaluation.

Never compute object-wise loss by assuming slot index equality after the first frame.

## 55. Optimisation

Initial default:

- `AdamW`;
- learning rate around `3e-4` for small modules, configurable;
- weight decay `1e-4` or lower;
- gradient clipping, default global norm 1.0;
- warmup plus cosine decay optional but simple;
- no exotic optimiser dependency.

Parameter groups may use a lower learning rate for perception if pretrained, but avoid many groups initially.

At a measurement-to-causal phase boundary, restore weights without silently
restoring stale perception-stage optimiser moments. Start the causal optimiser
state deliberately and record the phase learning rate. A short causal screen is
an implementation check, not convergence evidence. Before judging a shared
model, train for a declared minimum amount of balanced scenario coverage and
continue until that minimum completes plus a predeclared broad-validation
plateau criterion is met.

## 56. Mixed precision and compilation

- MPS: float32 by default; do not assume AMP support/benefit;
- CUDA: optional autocast and `GradScaler` from config;
- CPU: float32;
- `torch.compile`: optional and off by default until correctness is established;
- log device and precision.

## 57. Truncated backpropagation

Long online sequences can exceed memory. Use configurable truncated BPTT:

1. carry numerical `WorldBelief`;
2. after `tbptt_steps`, detach all tensors through a recursive utility;
3. retain object IDs, lifecycle counters, and values;
4. continue online operation.

Test that detaching does not alter values or masks.

## 58. Gradient safety

- check finite total loss;
- optionally log per-module gradient norms;
- skip/update safely on nonfinite gradients with a clear warning and counter;
- do not silently continue indefinitely;
- clamp high-risk quantities such as inverse depth, variance, mass, and modal frequency.

## 59. Checkpointing

Each checkpoint contains:

- model state;
- optimiser state;
- scheduler state;
- global step/epoch;
- full resolved config;
- random number generator states where practical;
- metric summary;
- specification version;
- git commit hash and dirty status;
- data/simulator version;
- device/precision metadata.

Save:

- `last.pt`;
- `best_rollout.pt`;
- a fixed-reference rollout checkpoint;
- periodic numbered validation checkpoints.

Use atomic temp-file rename to avoid corrupt checkpoints.

`best_rollout.pt` must be chosen with physical, pooled validation metrics rather
than a scale-dependent training loss. The primary score is a declared
horizon-weighted position RMSE, and a candidate may replace the incumbent only
when it improves that score without material regression in:

- current position and velocity;
- every declared forecast horizon;
- distance-gated object recall and precision;
- forecast lifecycle coverage;
- collision F1;
- distance-gated identity switches;
- uncertainty calibration relative to the nominal coverage target.

Apply these guardrails against both the moving incumbent and the fixed
pre-campaign reference so a sequence of individually tolerated changes cannot
ratchet into a large regression. The validation checkpoint must persist the
exact seed manifest and a canonical protocol hash covering simulator, model,
runtime, metric, horizon, and validation-batch semantics. Never trust incumbent
metrics on resume unless the corresponding incumbent weights are available and
their recorded tensor hash is verified. Keep every numbered validation
snapshot so later per-scenario confirmation can choose honestly without
discarding a long campaign.

## 60. Reproducibility

Seed:

- Python `random`;
- NumPy;
- PyTorch CPU;
- CUDA when available;
- simulator episode generator.

MPS/CUDA exact bitwise determinism may not always be possible. Record reproducibility mode and provide a CPU deterministic smoke test.

## 61. Logging

Use:

- console progress;
- JSONL metrics in run directory;
- optional TensorBoard through PyTorch's writer;
- saved resolved YAML;
- saved evaluation JSON;
- generated plots/videos.

Do not introduce an external tracking service or require API keys.

Log at minimum:

- total and component losses;
- current/posterior state errors;
- rollout errors by horizon;
- correction improvement;
- event metrics;
- parameter error;
- uncertainty/NLL;
- object count/association statistics;
- penetration/physics diagnostics;
- timing and memory;
- learning rate;
- gradient norm.

---

# Part XI — Synthetic validation environment

## 62. Why this environment

The first validation must be:

- cheap enough for a MacBook Pro with MPS;
- rich enough to exercise the final architecture;
- deterministic and fully labelled;
- genuinely online and visual;
- scalable to cloud training without data-contract changes.

The chosen environment is a vectorised synthetic 3D sphere world rendered as small RGB frames.

This avoids building a general-purpose rigid-body engine while still including:

- 3D state;
- perspective projection;
- depth ambiguity and occlusion;
- collisions and discontinuities;
- camera movement;
- unknown dynamics parameters;
- data association;
- belief correction.

The sphere world is the first simulator backend, not a permanent definition of
the dynamics the model may learn. Dataset generation must preserve a backend
boundary so a mature rigid-body engine, another simulator, or recorded real
data can emit the same episode and observation contracts.

## 63. Simulator state

For each object:

- persistent simulator ID;
- position `[3]`;
- velocity `[3]`;
- radius;
- mass;
- restitution;
- drag;
- surface friction/tangential damping;
- RGB albedo/texture code;
- active/existence;
- sleep state.

Global:

- gravity;
- bounded box or floor/side planes;
- camera trajectory and intrinsics;
- lighting parameters;
- simulator timestep;
- episode seed.

## 64. Physics

The architecture is intended to learn more than one physical system. The
benchmark, however, must make the selected system explicit and auditable.
Maintain at least one familiar reference regime whose trajectories can be
judged visually, then add unusual restitution, friction, mass-ratio,
actuation, or camera regimes under distinct scenario names. Do not silently
mix a surprising or confounded regime into the reference demo.

Every deterministic reference regime must be backed by invariant tests where
applicable:

- sphere-pair impulses conserve linear momentum;
- relative normal speed obeys configured restitution within numerical
  tolerance;
- tangential changes are attributable to the configured friction model;
- separating overlaps do not receive a second impulse;
- event labels identify the interval in which the state jump occurs;
- curriculum construction avoids accidental simultaneous events unless the
  scenario is explicitly named and evaluated as a compound event.

In particular, the ensured pair-collision setup must provide a clean temporal
window between the pair impact and the first floor impact. Otherwise floor
friction can cancel lateral motion in the same observation interval, making a
correct but compound trajectory look like an incorrect or sticky pair
collision. Compound pair/floor events remain useful later, but they belong in
a separately labelled harder scenario.

### 64.1 Integration

Use a stable semi-implicit Euler at a high substep rate, e.g. 120–240 Hz.

### 64.2 Sphere–sphere collision

Use analytic contact detection and impulse resolution. Include:

- restitution;
- inverse-mass position correction;
- optional tangential impulse/friction;
- event labels and exact/approximate contact time.

### 64.3 Sphere–plane collision

Ground and optional walls. Include restitution and tangential damping.

### 64.4 Drag

Apply linear or exponential drag.

### 64.5 External events

Optionally inject rare impulses at labelled times in harder configs. This tests surprise/event handling.

### 64.6 Simulator/model mismatch

The model should not simply share all simulator equations and parameters. Introduce mild mismatch:

- simulator may use slightly different drag integration;
- random tangential friction;
- small external perturbations;
- rendering noise.

The explicit model still provides structure, while learned residuals have work to do.

Simulator/model mismatch must not mean simulator ambiguity. Record the
integrator, contact law, scenario name, parameter ranges, and simulator/data
version in generated artifacts. A model may learn unfamiliar dynamics, but
evaluation claims require either familiar interpretable physics or sufficient
invariant/counterfactual diagnostics to establish what “correct” means.

### 64.7 Independent physics-engine reference backend

Add a mature physics-engine backend when the dependency and training budget
permit. Its purpose is independent realistic ground truth for contacts,
rolling, spin, friction, stacked/compound interactions, and counterfactual
validation; it must not leak engine state or equations into runtime inference.

The backend must:

- emit the canonical episode tensors, timestamps, calibration, identities,
  visibility, parameters, and event labels;
- render or export RGB through the same `ObservationPacket` path;
- expose privileged state only to supervision, evaluation, unit tests, and
  clearly labelled oracle debugging;
- identify engine name/version, solver settings, timestep, units, and scenario
  in metadata;
- provide deterministic seeds where the engine supports them;
- use explicit train/validation/test manifests;
- report metrics separately from the lightweight sphere backend.

Do not make a heavyweight engine mandatory for the default local smoke test.
The analytic sphere backend remains the fast deterministic invariant oracle.
An engine-backed reference is an additional dataset family used to verify that
the model learns dynamics rather than reproducing quirks of one handwritten
simulator. The same model and persistent belief contracts must run on both
without architectural replacement.

## 65. Rendering

Implement a lightweight renderer with NumPy/PyTorch/Pillow or pure PyTorch operations.

Required:

- perspective projection;
- depth sorting;
- circles/discs with apparent size;
- simple Lambertian-like shading or radial gradient;
- background/floor cue;
- soft or anti-aliased edges where practical;
- occlusion;
- camera movement;
- RGB output `[3,H,W]` in `[0,1]`.

Optional:

- cast blob shadows to improve depth cues;
- simple textures/markers;
- low-level noise and colour jitter.

The renderer does not need to be differentiable because ground-truth states supervise measurement and dynamics modules. The model's measurement projector is differentiable.

## 66. Camera

Sample calibrated camera trajectories:

- fixed;
- linear translation;
- orbit around scene;
- gentle rotation;
- combinations.

Provide intrinsics and extrinsics to the runtime in Milestone 1.

Ensure objects remain visible enough for learnability while still creating partial/full occlusions.

## 67. Data generation

Support two modes:

1. on-the-fly generation from deterministic episode seeds;
2. optional cached `.pt`/`.npz` shards for speed and reproducibility.

Start with on-the-fly or a small pre-generated validation set. Avoid a complex dataset service.

Each episode record:

```python
{
    "rgb": Tensor[T, 3, H, W],
    "timestamps": Tensor[T],
    "camera": {...},
    "objects": {
        "id": Tensor[T, N],
        "active": Tensor[T, N],
        "position": Tensor[T, N, 3],
        "velocity": Tensor[T, N, 3],
        "radius": Tensor[T, N, 1],
        "mass": Tensor[T, N, 1],
        "restitution": Tensor[T, N, 1],
        "drag": Tensor[T, N, 1],
        "friction": Tensor[T, N, 1],
        "visible_fraction": Tensor[T, N],
    },
    "events": {...},
    "seed": int,
}
```

Pad to `N_max` with masks.

## 68. Dataset splits

Use seed ranges or explicit manifests, not random split at runtime.

Ensure test distributions include:

- held-out initial states;
- held-out parameter combinations;
- held-out camera trajectories;
- longer horizons;
- optionally out-of-range but reasonable restitution/drag values.

Keep one in-distribution validation split and one compositional/OOD split.

## 69. Difficulty configurations

### `toy_smoke.yaml`

- 64×64;
- 2–3 objects;
- fixed camera;
- short sequences;
- fast CPU/MPS smoke test.

### `toy_mps.yaml`

- 96×96;
- 3–6 objects;
- camera movement;
- occlusion;
- variable mass/restitution/drag;
- 64–96 frames;
- small model.

### `toy_hard.yaml`

- 128×128;
- 4–10 objects;
- similar colours;
- stronger occlusion;
- irregular dropped frames;
- longer horizons;
- external impulses;
- CUDA recommended.

All use the same model classes and data contracts.

## 70. Toy perception labels

Provide:

- projected centre;
- apparent radius;
- inverse depth;
- visible fraction;
- coarse segmentation mask;
- object colour/appearance;
- existence;
- ground-truth association.

These labels pretrain and evaluate measurement extraction. RGB-only runtime must not receive ground-truth IDs or states.

## 71. Required demonstrations

The demo must visualise, for a held-out RGB episode:

- current RGB frame;
- measured object proposals;
- prior projected positions/support;
- posterior projected positions;
- ground-truth object positions;
- future rollout before correction;
- future rollout after correction;
- uncertainty ellipses/bands where practical;
- detected/predicted collision events;
- parameter estimates over time;
- per-step error and correction improvement.

Export a GIF or MP4 if dependencies permit; PNG sequences are an acceptable fallback.

---

# Part XII — Evaluation

## 72. Core evaluation questions

1. Does the model maintain correct identities?
2. Does it predict future state better than simple baselines?
3. Does correction improve the future forecast?
4. Does uncertainty reflect actual error?
5. Does it handle occlusion and collisions?
6. Does it infer observable physical parameters?
7. Is the online update cheap?
8. Does irregular timing work?
9. Does the same architecture run on MPS and CUDA?

## 73. Metrics

### 73.1 State

- position RMSE/MAE by horizon;
- velocity RMSE;
- orientation geodesic error when applicable;
- depth error;
- error during visible versus occluded intervals.

Normalise and also report physical units.

### 73.2 Forecast improvement

For each correction time:

\[
\Delta E_h =
E_h(\text{prior rollout})-
E_h(\text{posterior rollout}).
\]

Report:

- mean \(\Delta E_h\);
- percentage of updates with positive improvement;
- improvement by horizon;
- improvement after injected perturbation;
- improvement after collision/occlusion.

This is a primary project metric.

### 73.3 Baselines

Compare against:

- static;
- constant velocity;
- analytic gravity/drag with default parameters;
- analytic dynamics with oracle parameters;
- oracle-measurement version of the model.

### 73.4 Events

- contact/collision precision, recall, F1;
- event timing error;
- false event rate;
- mode accuracy/confusion matrix.

### 73.5 Tracking

- identity switches;
- IDF1-like score;
- object recall/precision;
- birth confirmation latency;
- survival through occlusion;
- false births/deaths.

For toy ground truth, implement simplified transparent metrics rather than importing a large MOT package.

### 73.6 Parameters

- MAE/NLL for restitution, drag, radius, friction, and mass ratio;
- convergence versus number of informative events;
- metrics conditioned on observability;
- calibration.

### 73.7 Physics diagnostics

- maximum/mean penetration;
- action–reaction residual;
- quaternion norm;
- rollout divergence/NaN rate;
- energy/momentum error on specially configured conservative scenes.

### 73.8 Uncertainty

Calibration metrics from Part IX, by horizon and condition.

### 73.9 Performance

At batch size 1:

- RGB global pass latency;
- RGB fast path latency;
- association latency;
- predict/correct latency;
- 1 s and 2 s rollout latency;
- peak allocated memory where available.

Report on current hardware rather than fabricating a target. Aspirational toy goal: a normal fast update and short rollout suitable for interactive use on Apple Silicon, with global discovery allowed to be slower.

## 74. Milestone 1 quantitative acceptance

The exact achievable numbers depend on training budget, but the implementation is not complete until it demonstrates all of the following on a held-out synthetic RGB set:

- end-to-end RGB operation with no oracle measurement input;
- persistent IDs and lifecycle functioning;
- no NaNs in a long evaluation run;
- corrected rollouts improve mean future position error relative to prior rollouts after injected perturbations;
- learned model beats constant-velocity baseline at collision-containing horizons;
- uncertainty grows during occlusion and contracts after reliable observation;
- event metrics are materially above chance;
- parameter estimates move toward ground truth when informative events occur;
- commands run on MPS in the `orpheus` environment;
- all unit/integration tests pass;
- demo output clearly shows prior versus posterior improvement.

Recommended target gates, adjustable only with documented evidence:

- at least 20% reduction in 1-second rollout RMSE after correction on perturbation episodes;
- at least 15% lower 1-second position RMSE than constant velocity on collision episodes;
- collision F1 at least 0.75 in the base toy distribution;
- ID switch rate below 2% of object-frame associations in the base distribution;
- 90% uncertainty interval empirical coverage between 80% and 98%.

Do not tune only to these numbers; report full curves and failure cases.

## 75. Evaluation protocol

- fixed test seeds;
- no training on test episodes;
- RGB-only and oracle-ablation reports clearly separated;
- evaluate best checkpoint and last checkpoint;
- save config/checkpoint hash;
- output JSON and human-readable Markdown summary;
- generate plots by horizon;
- include at least a few failure visualisations.

Checkpoint comparisons must use the same explicit episode-seed manifest,
scenario order, object-count distribution, sequence length, horizon set, and
metric semantics. A resumed training run must not inherit a best score when any
of those selection-defining fields changes. Changing a checkpoint's embedded
validation count must never silently change the episodes used for a paired
comparison; use an explicit seed offset or persisted seed manifest.

Promote a shared-model adaptation only when paired held-out evidence improves
the declared selection objective without hiding material per-scenario
regressions. Negative adaptations are valid evidence and must remain labelled
as rejected rather than being reported as accuracy gains.

---

# Part XIII — Public runtime API

## 76. Minimal user-facing API

```python
model = OnlineWorldModel.from_config(config)
model.reset(batch_size=1)

belief = model.ingest(observation_packet)
future = model.predict(query_times=[0.1, 0.5, 1.0, 2.0])
```

Convenience:

```python
belief = model.initialize(initial_packets)
belief, future = model.step(packet, prediction_horizon=2.0)
```

The lower-level `ingest` API is authoritative.

## 77. `OnlineWorldModel`

Suggested structure:

```python
class OnlineWorldModel(nn.Module):
    def __init__(
        self,
        observation_modules: Mapping[str, ObservationModule],
        dynamics: DynamicsModel,
        associator: Associator,
        updater: BeliefUpdater,
        lifecycle: ObjectLifecycle,
        identifier: ParameterIdentifier,
        scheduler: ObservationScheduler,
        belief_factory: BeliefFactory,
    ) -> None:
        ...

    def reset(self, batch_size: int = 1) -> None:
        ...

    def initialize(
        self,
        packets: Sequence[ObservationPacket],
    ) -> WorldBelief:
        ...

    def ingest(
        self,
        packets: ObservationPacket | Sequence[ObservationPacket],
    ) -> WorldBelief:
        ...

    def predict(
        self,
        query_times: Sequence[float] | Tensor,
    ) -> BeliefTrajectory:
        ...

    @property
    def belief(self) -> WorldBelief | None:
        ...
```

Normal runtime state is explicit and resettable. Avoid global singletons.

## 78. Training API

The trainer should operate on a functional sequence runner rather than depending on the stateful singleton used by demos. Reuse the same module logic.

Suggested:

```python
outputs = sequence_model.run_episode(
    batch,
    mode="closed_loop_rgb",
    rollout_queries=queries,
    perturbation_policy=policy,
)
losses = loss_computer(outputs, batch)
```

## 79. CLI

### `train.py`

```bash
python train.py \
  --config configs/toy_mps.yaml \
  --run-name baseline
```

Optional flags:

- `--resume PATH`;
- `--device auto|mps|cuda|cpu`;
- `--seed`;
- `--set key=value` for a small number of dotted overrides, if implemented simply.

### `evaluate.py`

```bash
python evaluate.py \
  --config configs/toy_mps.yaml \
  --checkpoint runs/baseline/checkpoints/best_rollout.pt \
  --split test \
  --output runs/baseline/evaluation
```

### `demo.py`

```bash
python demo.py \
  --config configs/toy_mps.yaml \
  --checkpoint ... \
  --seed 123 \
  --output demo_outputs/seed_123
```

### Additional developer scripts

Place under `scripts/`, not root:

- generate/cache dataset;
- inspect episode;
- benchmark latency;
- validate config;
- export demo animation.

## 80. Configuration

Use plain YAML loaded into typed dataclasses. Avoid Hydra/OmegaConf.

Requirements:

- defaults and validation;
- resolved config saved;
- unknown keys raise errors;
- paths resolved relative to repo/config location clearly;
- no arbitrary code execution in YAML.

Example top-level:

```yaml
project:
  name: orpheus
  seed: 42
  output_dir: runs

device:
  preference: auto
  cuda_amp: true
  mps_float32: true

simulator:
  type: sphere_world
  image_size: [96, 96]
  frame_rate: 30
  physics_rate: 120
  sequence_frames: 72
  min_objects: 3
  max_objects: 6

model:
  max_objects: 8
  state:
    geometry_dim: 8
    appearance_dim: 32
    residual_dynamics_dim: 16
    modal_count: 4
    modal_dim: 3
  rgb:
    backbone_channels: [32, 64, 96, 128]
    global_every_steps: 15
    roi_size: 20
  dynamics:
    max_substep: 0.008333333
    hidden_dim: 96
  filter:
    hidden_dim: 128

training:
  batch_size: 6
  steps: 30000
  learning_rate: 0.0003
  weight_decay: 0.0001
  tbptt_steps: 24
  grad_clip_norm: 1.0
  checkpoint_every: 1000
  eval_every: 1000

evaluation:
  horizons_seconds: [0.1, 0.25, 0.5, 1.0, 2.0]
```

Provide complete configs in the repository; this excerpt is illustrative.


# Part XIV — Repository architecture

## 81. Required repository tree

Create the following from the empty repository. Small deviations are acceptable only when documented and clearly improve cohesion.

```text
.
├── AGENTS.md
├── PROJECT_SPEC.md
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── train.py
├── evaluate.py
├── demo.py
│
├── configs/
│   ├── default.yaml
│   ├── toy_smoke.yaml
│   ├── toy_mps.yaml
│   ├── toy_hard.yaml
│   └── cloud_single_gpu.yaml
│
├── world_model/
│   ├── __init__.py
│   ├── py.typed
│   │
│   ├── abstractions/
│   │   ├── __init__.py
│   │   ├── contracts.py
│   │   ├── registry.py
│   │   ├── router.py
│   │   └── tokenizer.py
│   │
│   ├── belief/
│   │   ├── __init__.py
│   │   ├── object_belief.py
│   │   ├── world_belief.py
│   │   ├── camera_belief.py
│   │   ├── packing.py
│   │   ├── hypotheses.py
│   │   ├── lifecycle.py
│   │   └── validation.py
│   │
│   ├── observations/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── packets.py
│   │   ├── measurements.py
│   │   ├── context.py
│   │   ├── state/
│   │   │   ├── __init__.py
│   │   │   └── module.py
│   │   └── rgb/
│   │       ├── __init__.py
│   │       ├── backbone.py
│   │       ├── global_detector.py
│   │       ├── roi_updater.py
│   │       ├── projector.py
│   │       ├── module.py
│   │       ├── cache.py
│   │       └── losses.py
│   │
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── association.py
│   │   ├── costs.py
│   │   ├── innovation.py
│   │   ├── scheduler.py
│   │   └── surprise.py
│   │
│   ├── filtering/
│   │   ├── __init__.py
│   │   ├── prediction.py
│   │   ├── correction.py
│   │   ├── analytic_update.py
│   │   ├── learned_update.py
│   │   └── uncertainty.py
│   │
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── analytic.py
│   │   ├── quaternion.py
│   │   ├── modal.py
│   │   ├── graph.py
│   │   ├── contacts.py
│   │   ├── events.py
│   │   ├── uncertainty.py
│   │   └── rollout.py
│   │
│   ├── identification/
│   │   ├── __init__.py
│   │   ├── parameters.py
│   │   ├── observability.py
│   │   ├── recurrent_updater.py
│   │   └── local_optimiser.py
│   │
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── online_world_model.py
│   │   ├── sequence_runner.py
│   │   ├── state.py
│   │   └── diagnostics.py
│   │
│   ├── simulator/
│   │   ├── __init__.py
│   │   ├── sphere_world.py
│   │   ├── physics.py
│   │   ├── collisions.py
│   │   ├── camera.py
│   │   ├── renderer.py
│   │   ├── episode.py
│   │   └── labels.py
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── synthetic.py
│   │   ├── collate.py
│   │   ├── splits.py
│   │   └── caching.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── loop.py
│   │   ├── losses.py
│   │   ├── matching.py
│   │   ├── perturbations.py
│   │   ├── curriculum.py
│   │   ├── checkpointing.py
│   │   └── logging.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   ├── baselines.py
│   │   ├── state_metrics.py
│   │   ├── tracking_metrics.py
│   │   ├── event_metrics.py
│   │   ├── calibration.py
│   │   ├── latency.py
│   │   └── reports.py
│   │
│   ├── visualisation/
│   │   ├── __init__.py
│   │   ├── frames.py
│   │   ├── trajectories.py
│   │   ├── uncertainty.py
│   │   ├── animation.py
│   │   └── plots.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── device.py
│       ├── seeds.py
│       ├── tensors.py
│       ├── transforms.py
│       ├── io.py
│       ├── profiling.py
│       └── version.py
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_belief_invariants.py
│   │   ├── test_packing.py
│   │   ├── test_quaternion.py
│   │   ├── test_modal_dynamics.py
│   │   ├── test_analytic_dynamics.py
│   │   ├── test_collisions.py
│   │   ├── test_association.py
│   │   ├── test_filter_update.py
│   │   ├── test_observability.py
│   │   ├── test_config.py
│   │   └── test_device.py
│   ├── integration/
│   │   ├── test_simulator_episode.py
│   │   ├── test_oracle_online_loop.py
│   │   ├── test_rgb_measurements.py
│   │   ├── test_rgb_online_loop.py
│   │   ├── test_checkpoint_roundtrip.py
│   │   └── test_cli_smoke.py
│   └── regression/
│       └── test_fixed_seed_metrics.py
│
├── scripts/
│   ├── generate_dataset.py
│   ├── inspect_episode.py
│   ├── benchmark.py
│   ├── validate_config.py
│   └── render_demo.py
│
├── project/
│   ├── PROJECT_VISION.md
│   ├── ARCHITECTURE.md
│   ├── SYSTEM_OVERVIEW.md
│   ├── WORLD_BELIEF.md
│   ├── MULTIMODAL_DESIGN.md
│   ├── FILTERING.md
│   ├── DYNAMICS.md
│   ├── DATA_CONTRACTS.md
│   ├── TRAINING.md
│   ├── DATASETS.md
│   ├── EVALUATION.md
│   ├── DESIGN_DECISIONS.md
│   ├── ROADMAP.md
│   ├── TASKS.md
│   ├── STATUS.md
│   ├── RESEARCH_NOTES.md
│   ├── CHANGELOG.md
│   └── CODING_GUIDELINES.md
│
└── docs/
    ├── getting_started.md
    ├── extending_modalities.md
    ├── toy_world.md
    └── troubleshooting.md
```

Do not create empty files solely to satisfy the tree. Populate required documentation and implement the modules needed for the current milestone. Future-only modality packages should be documented rather than filled with meaningless stubs.

## 82. Root files

### `AGENTS.md`

Must tell coding agents:

- read `PROJECT_SPEC.md`;
- inspect `project/STATUS.md`, `TASKS.md`, `DESIGN_DECISIONS.md`, and `CHANGELOG.md`;
- preserve public contracts;
- update tests and docs;
- run relevant commands;
- record exact unfinished work;
- do not replace the architecture with a simpler clip predictor;
- do not use oracle state in the claimed RGB result;
- do not add heavy infrastructure without documented need.

### `README.md`

Concise entry point:

- what Orpheus is;
- current implemented status, not aspirational claims;
- quick start;
- commands;
- diagram;
- links to spec and docs;
- current toy results once available;
- known limitations.

### `LICENSE`

Choose a permissive license unless the user specifies otherwise. Apache-2.0 is a reasonable default because it includes explicit patent terms. Record the choice in design decisions.

### `requirements.txt`

Keep runtime dependencies straightforward. Do not list/replace PyTorch if the environment already has a custom compiled installation; in `pyproject.toml`, make PyTorch an optional/external documented prerequisite or use a broad dependency only if it will not force reinstall.

Likely dependencies:

- `numpy`;
- `scipy`;
- `PyYAML`;
- `Pillow`;
- `matplotlib`;
- `tqdm`;
- optional `tensorboard`.

Development:

- `pytest`;
- `pytest-cov`;
- `ruff`;
- `mypy` or `pyright` if chosen.

Avoid mandatory OpenCV, torchvision compiled ops, graph libraries, Hydra, Lightning, Ray, W&B, or MLflow initially.

### `pyproject.toml`

Configure:

- package metadata;
- Python version, e.g. 3.11+ if environment supports it;
- setuptools;
- ruff;
- pytest;
- type checking;
- optional `dev` extras;
- console scripts only if useful, while retaining root scripts.

## 83. Project memory files

These files are not ceremonial. They prevent agent drift.

### `project/STATUS.md`

Always state:

- what works now;
- last validated commands;
- latest checkpoint/result paths if committed only as references;
- current blockers;
- known failures;
- next concrete task;
- hardware/environment used;
- date.

### `project/TASKS.md`

Use checkboxes grouped by current milestone. Each task should be verifiable. Keep deferred ideas separate.

### `project/DESIGN_DECISIONS.md`

Use ADR-like entries:

- ID/date;
- context;
- decision;
- alternatives considered;
- consequences;
- status.

Initial decisions should include:

- persistent belief;
- multimodal observation contracts;
- measurement-space prediction;
- synthetic RGB first;
- stable modal rather than fixed DCT state;
- hybrid physics;
- diagonal uncertainty initially;
- no Lightning/Hydra;
- known camera pose in Milestone 1;
- Apache-2.0 or selected licence.

### `project/CHANGELOG.md`

Record user-visible/research-significant changes. Do not duplicate every commit.

### `project/RESEARCH_NOTES.md`

Capture hypotheses, experiments, findings, and failure analysis. Clearly label speculation versus evidence.

### `project/ROADMAP.md`

Milestones and acceptance criteria, not a wish list.

### `project/DATA_CONTRACTS.md`

Mirror canonical dataclasses/tensor shapes and update when contracts change.

## 84. Module dependency direction

Preferred dependency direction:

```text
utils / typed data
        ↓
belief + observation contracts
        ↓
dynamics / filtering / fusion / identification
        ↓
runtime sequence orchestration
        ↓
training / evaluation / demos
```

Simulator/datasets may depend on common typed data and utilities, but core runtime must not import training code.

Avoid circular imports by:

- placing shared dataclasses in low-level packages;
- using `TYPE_CHECKING`;
- passing interfaces rather than importing concrete modules.

## 85. Public versus internal API

Public:

- configuration loader;
- `ObservationPacket`;
- `WorldBelief`;
- `BeliefTrajectory`;
- `OnlineWorldModel`;
- observation-module registration;
- training/evaluation CLI.

Internal details may change:

- exact CNN layers;
- graph MLP shape;
- matching cost implementation;
- recurrent updater hidden structure.

Mark exports deliberately in `__init__.py`.

---

# Part XV — Implementation programme

## 86. General execution rule

Build one integrated vertical system. Do not spend months on isolated toy architectures, but do use small, testable steps inside the same architecture.

Codex should continue until the first vertical slice is runnable, tested, and documented. A repository containing only folder structure, dataclass placeholders, or pseudocode is not a valid result.

## 87. Phase 0 — Repository and executable skeleton

Deliver:

- root files;
- package installation;
- configs;
- device selection;
- typed config loader;
- project memory docs;
- test harness;
- `train.py`, `evaluate.py`, `demo.py` that parse config and fail only with meaningful unimplemented status during the first commit.

Then proceed immediately; do not stop here.

Acceptance:

- `pip install -e ".[dev]"` succeeds without replacing custom PyTorch;
- `pytest` runs;
- config validation works;
- `python train.py --config configs/toy_smoke.yaml --dry-run` prints resolved plan.

## 88. Phase 1 — Simulator, labels, and baselines

Implement:

- vectorised sphere world;
- deterministic seeds;
- camera and renderer;
- exact labels/events;
- dataset/collate;
- static/constant velocity/analytic baselines;
- inspection visualisation.

Acceptance:

- fixed seed regression test;
- collision momentum/restitution tests;
- rendered episode inspection;
- shapes/masks valid;
- evaluation baselines run.

## 89. Phase 2 — Belief, analytic/modal dynamics, and oracle filter

Implement:

- canonical belief dataclasses;
- packing/validation;
- analytic dynamics;
- modal bank;
- event/contact model initial structured implementation;
- uncertainty propagation;
- state/oracle observation module;
- association/lifecycle;
- learned/analytic correction;
- online sequence runner.

Acceptance:

- oracle noisy measurements initialise and update belief;
- rollout works at arbitrary query times;
- injected state perturbation is corrected;
- posterior rollout improves over prior;
- collision handling stable;
- all unit/integration tests pass.

This is a debugging checkpoint, not the claimed final milestone.

## 90. Phase 3 — RGB global measurements

Implement:

- lightweight backbone;
- proposal head;
- Hungarian supervised training;
- measurement projector;
- measurement uncertainty;
- RGB pretraining mode;
- visual diagnostics.

Acceptance:

- held-out measurement accuracy substantially better than naive centre guessing;
- proposal recall/precision and depth error reported;
- uncertainty finite/calibrated at a basic level;
- no simulator state used as runtime input.

## 91. Phase 4 — RGB fast residual updater and association

Implement:

- projected ROIs;
- `grid_sample` ROI extraction;
- cached features;
- fast measurement corrections;
- appearance association;
- global scheduler/surprise;
- occlusion handling;
- lifecycle.

Acceptance:

- IDs persist through synthetic occlusion;
- fast path produces valid measurements;
- global path recovers from loss;
- runtime diagnostics distinguish fast/global passes.

## 92. Phase 5 — Full closed-loop training

Implement:

- episode unroll;
- truncated BPTT;
- perturbation/recovery;
- rollout queries;
- complete losses;
- checkpoint/resume;
- JSONL/TensorBoard logging;
- RGB-only curriculum endpoint.

Acceptance:

- train command runs on MPS;
- loss decreases on smoke overfit;
- closed-loop RGB evaluation runs;
- correction improves future prediction;
- baseline comparisons saved;
- checkpoint roundtrip exact enough.

## 93. Phase 6 — Online parameter identification

Implement:

- parameter beliefs;
- observability gates;
- recurrent updater;
- supervised and rollout parameter losses;
- plots.

Acceptance:

- restitution estimates improve after collisions;
- drag estimates improve during free motion;
- unobservable parameters retain high uncertainty rather than false certainty;
- future rollout improves versus fixed default parameters.

If Phase 6 is too large for the first coding pass, the interfaces and a working bounded simple updater must still be present, with the full recurrent refinement next in `TASKS.md`. The user specifically wants online adaptation, so do not omit parameter update entirely.

## 94. Phase 7 — Evaluation and demo completion

Implement:

- complete evaluator;
- calibration;
- tracking/events;
- latency benchmark;
- prior/posterior demo;
- Markdown report;
- failure examples.

Acceptance: Milestone 1 definition below.

## 95. Milestone 1 definition of done

Milestone 1 is complete only when:

1. the repository installs in the `orpheus` environment without reinstalling PyTorch;
2. unit and integration tests pass;
3. `toy_smoke` can overfit a tiny fixed dataset;
4. `toy_mps` trains with MPS;
5. evaluation uses RGB-only observations;
6. the model maintains a persistent belief and object IDs;
7. it produces prior and posterior future rollouts;
8. a new frame cheaply corrects state without re-encoding history or updating model weights;
9. posterior future prediction improves on held-out perturbation episodes;
10. collisions, occlusion, uncertainty, and camera motion are exercised;
11. at least restitution and drag have online update paths;
12. metrics compare against baselines;
13. demo artefacts visibly show operation;
14. `README.md`, project status, decisions, tasks, and changelog match reality;
15. no core module is a placeholder.

## 96. Milestone 2 — Stronger dynamics and hypotheses

After Milestone 1, improve without redesign:

- typed predictive-abstraction registry and reversible belief-token layer;
- evidence-driven abstraction selection and learned residual-token processing;
- branch/merge hypotheses;
- continuous collision timing;
- richer object geometry;
- stronger graph equivariance;
- learned camera update;
- fixed-lag smoothing;
- DCT/window spectral baseline;
- longer horizons.

## 97. Milestone 3 — Second modality

Choose one based on product direction:

- audio for hidden impacts/material inference;
- skeleton for human/object interaction;
- IMU for ego-motion;
- depth for geometry.

Adding it should validate the observation contract and shared-belief fusion. Do not build all modalities at once.

## 98. Milestone 4 — Real data

Integrate strong pretrained perception or calibrated datasets behind the RGB module. Preserve core state/dynamics/filter interfaces.

Potential approach:

- use externally produced detections/masks/depth as structured observation modules first;
- then fine-tune end-to-end where feasible;
- explicitly model domain mismatch and uncertainty.

---

# Part XVI — Testing strategy

## 99. Unit testing

Every mathematical component needs focused tests.

### 99.1 Geometry

- transform inversion/composition;
- camera projection/back-projection;
- quaternion identity/composition;
- finite small-angle gradients.

### 99.2 Modal dynamics

Tests listed in Part VI.

### 99.3 Physics

- isolated gravity trajectory;
- drag decay;
- elastic equal-mass collision;
- inelastic collision;
- sphere–plane bounce;
- no impulse for separating bodies;
- bounded penetration correction;
- permutation symmetry.

### 99.4 Belief

- shape/mask invariants;
- pack/unpack roundtrip;
- clone/detach/device transfer;
- no mutation during rollout;
- ID uniqueness.

### 99.5 Association

- obvious matches;
- gated impossible match;
- unmatched births;
- ambiguous pair;
- no duplicate assignment.

### 99.6 Filter

- zero innovation leaves mean approximately unchanged;
- low measurement noise causes stronger correction;
- high measurement noise causes weaker correction;
- posterior variance contracts;
- missed observation expands uncertainty;
- robust clipping rejects extreme outlier.

### 99.7 Parameter observability

- restitution gate near zero without collision;
- drag gate active with sufficient free motion;
- mass ratio gate active only with interaction;
- no update during ambiguous association.

## 100. Integration tests

Use tiny deterministic models/data:

- simulator → RGB → measurements;
- oracle online loop;
- RGB global initialisation;
- RGB fast update;
- prior/posterior rollout;
- checkpoint save/load;
- one training step;
- one evaluation episode;
- CLI subprocess smoke.

Keep CI tests under a reasonable CPU runtime. Mark longer MPS/CUDA tests separately.

## 101. Regression tests

Store a small set of expected metric ranges rather than exact neural outputs. For deterministic simulator physics, exact arrays are acceptable.

Do not commit large checkpoints. A tiny random/fixed checkpoint may be used only if truly necessary.

## 102. Overfit test

Provide a command/config that overfits 8–32 deterministic episodes. This is a required debugging tool.

Success:

- proposal loss falls;
- state/rollout loss falls;
- no identity collapse;
- demo visibly matches.

## 103. Device tests

- CPU always;
- MPS when available;
- CUDA when available.

Use `pytest.mark` and clear skips. Avoid unsupported MPS operations or provide explicit CPU fallback for small association only.

---

# Part XVII — Performance and scaling

## 104. Apple MPS considerations

The `orpheus` conda environment already contains compiled PyTorch with MPS. Do not run a command that uninstalls/reinstalls it.

Recommendations:

- pure PyTorch operations;
- avoid compiled third-party ops;
- float32;
- modest batch/image sizes;
- `num_workers=0` or low initially if macOS multiprocessing causes friction;
- profile memory;
- avoid Python loops over pixels;
- vectorise pairwise object interactions;
- association may run on CPU because `N` is small;
- use `grid_sample` for ROIs;
- make checkpoint writing infrequent enough not to dominate.

Implement `select_device("auto")`:

1. CUDA if available and requested/preferred;
2. MPS if available;
3. CPU.

Log actual selected device.

## 105. Straightforward cloud CUDA path

A cloud user should be able to:

```bash
git clone ...
cd ...
python -m venv .venv
source .venv/bin/activate
# install the appropriate CUDA PyTorch separately
pip install -e ".[dev]"
python train.py --config configs/cloud_single_gpu.yaml
```

The first cloud path is one CUDA GPU. No API tokens are required by this repository.

## 106. Multi-GPU future path

Do not implement distributed training until a single GPU is a bottleneck. Prepare by:

- avoiding global mutable training state;
- keeping batch dimension explicit;
- checkpointing rank-independently;
- using deterministic sampler interfaces;
- separating model and trainer.

When needed, use standard `torchrun` + `DistributedDataParallel`, not a new orchestration framework.

## 107. Dataset scaling

On-the-fly simulation may become CPU-bound. Future options:

- pre-generated shards;
- parallel episode workers;
- memory-mapped tensors;
- GPU simulator;
- mixed real/synthetic manifests.

Preserve the episode data contract.

## 108. Profiling

Provide `scripts/benchmark.py` with:

- warmup;
- synchronisation for MPS/CUDA where supported;
- per-component timings;
- batch size and object count sweep;
- JSON output.

Optimise after measuring. Fast path should avoid global processing every frame.

## 109. Complexity expectations

For small object count:

- graph dynamics: \(O(N^2)\), acceptable for \(N\le 10\);
- ROI perception: approximately \(O(N)\);
- global RGB: depends on image pixels, run intermittently;
- association: \(O(N^3)\) Hungarian on CPU, negligible at small N.

For larger scenes, later use spatial indexing, sparse edges, and hierarchical belief. Do not prematurely add them.

---

# Part XVIII — Engineering standards

## 110. Code style

- type hints on public functions and complex internals;
- docstrings explain semantics, frames, shapes, and units;
- descriptive names;
- functions small enough to test;
- composition over inheritance;
- dataclasses for structured data;
- no wildcard imports;
- no hidden global device;
- explicit masks;
- assertions/validators in debug/tests.

Use Ruff formatting/linting or equivalent, configured in `pyproject.toml`.

## 111. Error handling

Raise actionable errors:

- unknown config key;
- nonmonotonic timestamp;
- missing calibration;
- unsupported modality;
- shape mismatch;
- checkpoint/config incompatibility;
- unavailable requested device.

Do not silently change behaviour.

## 112. Comments and documentation

Comments should explain why, invariants, or numerical considerations—not restate obvious code.

Every public class should link conceptually to the relevant project document.

## 113. Dependency policy

A dependency needs:

- clear benefit;
- maintenance/portability check;
- entry in design decisions if substantial;
- no hidden service/API requirement.

Prefer standard library, NumPy/SciPy, and PyTorch.

## 114. Security and data policy

The initial project is local research software. Still:

- do not execute config content as code;
- avoid unsafe pickle loading from untrusted sources; checkpoint loading is trusted/local and documented;
- do not download models/data automatically without explicit command;
- do not embed secrets;
- no telemetry;
- sanitise output paths.

## 115. Git discipline

Codex should:

- make coherent commits if instructed/allowed;
- avoid committing generated runs/checkpoints/datasets;
- create `.gitignore`;
- keep docs synchronised;
- not rewrite user changes;
- report exact tests run.

Suggested `.gitignore`:

- Python caches;
- environments;
- `runs/`;
- `data/cache/`;
- checkpoints;
- generated demos;
- editor/OS files.

## 116. Compatibility

Record supported Python/PyTorch versions after inspecting `orpheus`. Do not assume exact versions before checking.

Use feature detection for optional device functionality.

## 117. No fake completeness

Do not:

- return random tensors from “implemented” modules;
- hide oracle inputs in RGB configs;
- claim online system identification if parameters are directly copied from labels;
- report training metrics from the training set as held-out results;
- leave `pass`/`NotImplementedError` in Milestone 1 paths;
- create dozens of empty modality classes;
- generate only a README and scaffold.

---

# Part XIX — Failure modes and mitigations

## 118. Perception dominates the project

Risk: object discovery becomes an open-ended computer-vision effort.

Mitigation:

- synthetic supervised RGB measurements;
- simple shapes;
- explicit measurement contract;
- optional oracle debugging;
- later replace RGB module with stronger pretrained perception without changing core.

## 119. Decoder hides bad physics

Risk: a powerful renderer creates plausible frames despite wrong state.

Mitigation:

- primary state/measurement losses;
- no photorealistic decoder in Milestone 1;
- intervention/counterfactual evaluation.

## 120. Smooth dynamics smear collisions

Risk: modal/ODE models average impulses.

Mitigation:

- explicit event logits;
- structured jump map;
- small substeps;
- event loss;
- piecewise rollout.

## 121. Autoregressive drift

Risk: recursive state errors accumulate.

Mitigation:

- structured analytic dynamics;
- stable modal state;
- long rollout losses;
- recurrent filtering with ground-truth observations;
- calibrated uncertainty;
- no frame-by-frame pixel generation.

## 122. Deterministic future averages ambiguity

Risk: one belief averages collision/no-collision futures.

Mitigation:

- uncertainty;
- event probabilities;
- `HypothesisSet` interface;
- later branching;
- evaluate ambiguous subsets.

## 123. Parameter hallucination

Risk: model confidently infers unobservable mass/friction.

Mitigation:

- observability gates;
- parameter priors/variance;
- conditional metrics;
- no update during ambiguity;
- counterfactual tests.

## 124. Filter learns to reset state

Risk: correction network ignores dynamics and reconstructs state each frame.

Mitigation:

- gated residual update;
- correction penalty;
- ROI residual input;
- missing-observation training;
- long-horizon posterior loss;
- ablation of correction magnitude.

## 125. Dynamics learns dataset shortcuts

Risk: colour or camera pixels predict physical parameters.

Mitigation:

- randomise appearance independently;
- repeated/similar colours;
- held-out combinations;
- explicit geometry/physics factors;
- counterfactual changes.

## 125.1 Benchmark physics is visually surprising or accidentally confounded

Risk: the model is judged against ground truth whose pair, floor, wall, or
external events overlap unintentionally, or whose unusual law was not
identified. Correct learning then looks wrong, while actual simulator defects
can be mistaken for model error.

Mitigation:

- a named familiar reference regime;
- analytic invariants and per-event diagnostics;
- separation of elementary events in the reference curriculum;
- separately named compound-event scenarios;
- plots of pre/post velocity and event time as well as position;
- simulator/data versioning and per-scenario metrics.

## 126. MPS unsupported operations

Risk: development stalls on platform issues.

Mitigation:

- pure PyTorch;
- no compiled ROI/graph ops;
- CPU fallback for Hungarian;
- device smoke tests early;
- document any fallback.

## 127. Endless architectural iteration

Risk: repeated redesign without integrated result.

Mitigation:

- freeze contracts;
- implement phases in this specification;
- use design decisions;
- define acceptance metrics;
- improve internals behind interfaces;
- maintain status/tasks;
- do not chase all modalities before RGB loop works.

## 128. Toy overfitting

Risk: system only works on coloured spheres.

Mitigation:

- use toy as architecture validation, not scientific conclusion;
- OOD parameter/camera splits;
- similar colours;
- camera movement;
- occlusion;
- plugin boundary;
- next milestone must test a second modality or real perception source.

---

# Part XX — Research hypotheses and required ablations

## 129. Primary hypotheses

H1. A persistent predict–correct belief model updates more cheaply and maintains better long-horizon state than re-encoding a full sliding clip.

H2. Stable modal state plus analytic kinematics reduces long-horizon drift relative to an unconstrained recurrent transition.

H3. Explicit event jumps improve collision prediction relative to smooth residual dynamics alone.

H4. Residual ROI perception achieves comparable correction quality to repeated global encoding at lower online cost.

H5. Separating fast-state correction and slow-parameter identification improves stability and parameter interpretability.

H6. Uncertainty-aware association/correction improves occlusion recovery and avoids catastrophic updates.

## 130. Required ablations after base model works

- no modal state;
- no learned residual dynamics;
- no explicit event jump;
- full/global RGB every frame versus ROI fast path;
- learned corrector versus analytic-only update;
- no uncertainty features;
- no slow-parameter updater;
- re-encode short clip baseline if feasible;
- constant velocity and analytic baselines.

Do not block initial implementation on every ablation, but design evaluation to add them straightforwardly.

## 131. Spectral baseline

A later `WindowSpectralPredictor` may:

- encode an observation prefix;
- predict DCT or continuous Fourier coefficients for a fixed future state window;
- reconstruct whole trajectory non-autoregressively;
- compare fixed-window accuracy, compute, and extrapolation against online modal belief.

This baseline tests the original paper-inspired idea. It must not replace the online filter because it is awkward to update with each new observation and tied to a horizon.

## 132. Novelty framing

The broad ingredients—state-space filtering, object-centric models, Koopman/modal dynamics, graph interactions, and system identification—have precedents. The potential contribution lies in their specific integration:

- online multimodal residual assimilation;
- compact deterministic dynamical programmes;
- explicit object/parameter uncertainty;
- cheap receding-horizon revision;
- event-aware hybrid modal physics;
- sensor-independent shared belief;
- counterfactual/online identification evaluation.

Do not make novelty claims in README before empirical evidence and literature review.

---

# Part XXI — Future extensions

## 133. Richer geometry

- cuboids/meshes;
- SE(3)-equivariant geometry;
- signed distance fields;
- part-based objects;
- learned shape uncertainty.

## 134. Articulated systems

- skeleton/kinematic tree state;
- joint limits;
- actuator/action latent;
- contacts between parts and objects;
- human manipulation.

## 135. Deformable modal state

The modal bank naturally extends to deformation coordinates. Separate rigid pose from object-local modes. Add geometry projector and material parameters.

## 136. Audio

Implement impact event/source measurements first, not waveform synthesis. Later attach a differentiable/learned acoustic renderer if useful.

## 137. Reservoir/ESN dynamics

A reservoir or echo-state module may be evaluated as a bounded residual temporal memory behind the `DynamicsModel` interface. It should not replace explicit state/physics without evidence.

## 138. 3D spatial attention

For many objects/parts, use geometry-aware attention or equivariant graph networks. The belief already supplies coordinates and frames.

## 139. Planning and control

A planner can roll out interventions/actions through the belief dynamics. Keep action input reserved in dynamics signatures, even if unused initially.

## 140. Semantic hierarchy

Future beliefs may contain:

- scenes/rooms;
- objects;
- parts;
- agents;
- relations;
- latent goals.

Do not conflate this with initial metric physical state.

## 141. Real-world adaptation

- pretrained segmentation/tracking/depth modules as observation providers;
- self-supervised measurement consistency;
- domain randomisation;
- calibration;
- small labelled captured datasets;
- uncertainty-aware fallback.

---

# Part XXII — Exact Codex operating directive

## 142. Initial behaviour

When given this file in an empty repository, Codex must:

1. read it completely;
2. inspect the actual Python/PyTorch/MPS environment without replacing PyTorch;
3. create `AGENTS.md` and project memory files;
4. write a concise implementation plan into `project/TASKS.md`;
5. record initial decisions;
6. build through the integrated phases;
7. run tests frequently;
8. keep docs accurate;
9. implement a working vertical slice, not merely explain what should be done.

Do not ask the user to choose low-level options already resolved here. Make reasonable implementation choices consistent with the specification and document them.

## 143. Prioritisation

When time or context is constrained, prioritise in this order:

1. simulator/data correctness;
2. belief contracts/invariants;
3. analytic dynamics and online loop;
4. oracle debug validation;
5. RGB measurement path;
6. RGB closed-loop correction;
7. training/evaluation/checkpointing;
8. parameter identification;
9. performance polish;
10. future extensions.

Return a truthful status if not every phase is complete, but leave the repository runnable and record exact continuation tasks.

## 144. Required evidence before declaring success

Codex should include in its final report:

- files created/changed;
- architectural summary;
- exact commands run;
- test results;
- training/evaluation run summary;
- demo output paths;
- observed metrics, clearly labelled;
- known limitations;
- next tasks from `project/TASKS.md`.

No fabricated results.

## 145. Documentation update rule

For each significant feature:

- update relevant architecture/design document;
- update tasks/status;
- add/change tests;
- update changelog if user-visible;
- then report completion.

Documentation need not literally precede every line of code, but it must be synchronised in the same work unit.

## 146. Simplicity constraints

Do not introduce:

- REST/GraphQL;
- authentication;
- tokens;
- cloud SDK;
- database;
- Docker/Kubernetes requirement;
- Lightning/Hydra;
- external experiment tracking;
- automatic paid-resource provisioning.

A simple local PyTorch repository is the goal.

## 147. Anti-drift checks

Before completing a work session, verify:

- Is `WorldBelief` still the persistent source of truth?
- Are learned/generative latents subordinate to explicit executable
  abstractions, with a reversible path back to typed belief proposals?
- Does the system use the simplest abstraction that passes predictive and
  uncertainty gates, rather than increasing model complexity by default?
- Can a new modality be added without editing dynamics?
- Does the runtime use timestamps?
- Does a new frame correct the belief without weight training?
- Are slow parameters updated more conservatively than fast state?
- Are uncertainty and masks propagated?
- Is RGB-only evaluation truly RGB-only?
- Does the toy use the same core runtime as future systems?
- Do train/evaluate/demo remain simple?
- Are docs truthful?

---

# Part XXIII — Appendix A: Core pseudocode

## 148. Online step

```python
def ingest_packets(
    state: RuntimeState,
    packets: Sequence[ObservationPacket],
) -> tuple[RuntimeState, WorldBelief]:
    packets = validate_and_sort_packets(packets)

    for timestamp, group in group_by_timestamp(packets):
        if state.belief is None:
            state.belief = initialise_from_group(group, state)
            continue

        dt = timestamp - float(state.belief.timestamp.item())
        if dt < 0:
            raise OutOfSequenceObservationError(...)

        prior = state.dynamics.predict(state.belief, dt)
        posterior = prior

        for packet in order_group(group, state.config.fusion.order):
            module = state.observation_modules[packet.modality]
            sensor_context = build_sensor_context(packet, posterior)

            predicted = module.project(posterior, sensor_context)
            mode = state.scheduler.choose(
                packet=packet,
                belief=posterior,
                predicted=predicted,
                diagnostics=state.diagnostics,
            )

            measurements, new_cache = module.observe(
                packet=packet,
                prior=posterior,
                predicted=predicted,
                cache=state.caches.get(packet.sensor_id),
                mode=mode,
            )
            state.caches[packet.sensor_id] = new_cache

            association = state.associator.match(
                posterior, measurements, predicted
            )
            innovation = module.innovation(
                measurements, predicted, association
            )

            cause = state.surprise_classifier(
                posterior, innovation, association
            )

            posterior = state.updater.correct(
                prior=posterior,
                measured=measurements,
                predicted=predicted,
                association=association,
                innovation=innovation,
                cause=cause,
            )

            posterior = state.lifecycle.apply(
                posterior, measurements, association, predicted
            )

            observable = state.observability(
                posterior, innovation, association, cause
            )
            posterior = state.identifier.update(
                posterior, innovation, association, observable
            )

            state.diagnostics.record(...)

        state.belief = posterior.with_timestamp(timestamp)

    return state, state.belief
```

## 149. Training unroll

```python
def run_training_episode(batch, model, config):
    belief = None
    outputs = []

    for t in range(batch.num_steps):
        packets = make_rgb_packets(batch, t)

        if belief is not None:
            belief = maybe_perturb(belief, batch, t, config)

        belief, diagnostics = model.functional_ingest(
            belief=belief,
            packets=packets,
            caches=...,
            training=True,
        )

        query_times = sample_rollout_queries(t, batch, config)
        trajectory = model.dynamics.rollout(belief, query_times)

        outputs.append({
            "belief": belief,
            "trajectory": trajectory,
            "diagnostics": diagnostics,
        })

        if (t + 1) % config.training.tbptt_steps == 0:
            belief = detach_world_belief(belief)
            detach_caches(...)

    losses = loss_computer(outputs, batch)
    return losses, outputs
```

## 150. Pair interaction

```python
def pairwise_interactions(objects, active_mask, dt):
    rel_pos = objects.position[:, None, :, :] - objects.position[:, :, None, :]
    rel_vel = objects.velocity[:, None, :, :] - objects.velocity[:, :, None, :]

    distance = safe_norm(rel_pos, dim=-1)
    normal = rel_pos / distance.clamp_min(eps).unsqueeze(-1)

    pair_mask = (
        active_mask[:, :, None]
        & active_mask[:, None, :]
        & ~identity_mask
        & candidate_distance_gate(...)
    )

    features = build_edge_features(...)
    edge_outputs = edge_network(features)

    analytic_impulse = compute_analytic_impulse(...)
    impulse = bounded_residual_impulse(analytic_impulse, edge_outputs)

    # Apply each unordered pair once to ensure equal/opposite action.
    delta_v = scatter_pair_impulses(impulse, normal, inverse_mass, pair_mask)

    return delta_v, edge_outputs.event_logits, edge_outputs.process_noise
```

---

# Part XXIV — Appendix B: Configuration validation

## 151. Required validation examples

Reject:

- `max_objects < simulator.max_objects`;
- nonpositive frame/physics rate;
- `max_substep` larger than observation timestep without explicit allowance;
- invalid restitution/friction bounds;
- image dimensions not positive;
- `modal_count < 0`;
- unsupported device;
- RGB modality enabled without camera calibration mode;
- evaluation horizons outside generated episode length;
- oracle input enabled in an RGB-only evaluation config;
- checkpoint config incompatibility.

Warn:

- batch likely too large for MPS based on simple heuristic;
- global discovery cadence too long relative to object entry;
- no collision events expected for parameter identification;
- parameter loss enabled for unobservable parameter in current simulator config.

---

# Part XXV — Appendix C: Suggested initial resolved model sizes

## 152. Smoke

- image 64×64;
- 3 object max;
- backbone 16/32/48/64;
- appearance 16;
- graph hidden 48;
- filter hidden 64;
- modes 2×2;
- batch 2;
- sequence 24;
- rollout 0.5 s.

## 153. MPS

- image 96×96;
- 8 belief slots;
- 3–6 objects;
- backbone 32/64/96/128;
- appearance 32;
- geometry 8;
- residual dynamics 16;
- graph hidden 96;
- filter hidden 128;
- 4 modes, modal dimension 3;
- batch 4–8;
- sequence 64–96;
- TBPTT 24;
- rollout up to 2 s.

## 154. Cloud single GPU

- image 128–192;
- 12–20 slots;
- backbone 48/96/160/256 or pretrained adapter;
- graph/filter hidden 192–256;
- batch scaled to memory;
- longer sequences;
- AMP;
- more difficult simulator or real-data measurements.

Changing these sizes must not change dataclass semantics or runtime flow.

---

# Part XXVI — Appendix D: Completion checklist

## 155. Repository

- [ ] `PROJECT_SPEC.md` present.
- [ ] `AGENTS.md` points to it.
- [ ] package installs.
- [ ] configs validated.
- [ ] project memory populated.
- [ ] `.gitignore` and license present.

## 156. Simulator

- [ ] deterministic sphere physics.
- [ ] collisions tested.
- [ ] camera/rendering.
- [ ] exact labels/events.
- [ ] train/val/test manifests.

## 157. Belief/runtime

- [ ] persistent IDs.
- [ ] timestamps.
- [ ] uncertainty.
- [ ] lifecycle.
- [ ] arbitrary-time rollout.
- [ ] no mutation bugs.

## 158. Dynamics

- [ ] analytic kinematics.
- [ ] stable modal bank.
- [ ] interaction graph.
- [ ] event jumps.
- [ ] process noise.
- [ ] physics diagnostics.

## 159. Observations

- [ ] oracle debug module.
- [ ] RGB global module.
- [ ] RGB fast residual module.
- [ ] projector.
- [ ] association.
- [ ] surprise/global scheduling.

## 160. Training

- [ ] closed-loop unroll.
- [ ] perturbation recovery.
- [ ] future losses.
- [ ] checkpoints/resume.
- [ ] MPS run.
- [ ] overfit test.

## 161. Identification

- [ ] bounded parameter beliefs.
- [ ] observability gates.
- [ ] online restitution/drag update.
- [ ] uncertainty/plots.

## 162. Evaluation/demo

- [ ] baselines.
- [ ] prior/posterior improvement.
- [ ] tracking/events.
- [ ] calibration.
- [ ] latency.
- [ ] held-out RGB-only report.
- [ ] visual demo.

---

# Part XXVII — Convergence-integrity amendment

## 163. Identifiable deterministic targets

Point forecasts and discrete event targets are valid optimisation targets only
while the requested future is identifiable from the causal belief available at
the anchor.

- A newly discovered track without enough timestamped observations for
  velocity is a cold-start forecast. Report it separately and train calibrated
  distributional uncertainty; do not let it dominate mature deterministic
  trajectory fitting.
- A future random intervention that is not present in the anchor observation
  makes the coupled scene stochastic until a later observation exposes it.
  Censor deterministic position, velocity, event, and correction-improvement
  losses across that hidden intervention. Continue to train forecast
  likelihood and calibration on its realised outcome.
- A hidden physical parameter may justify a broad predictive distribution
  before it becomes observable. Exact pre-event point accuracy and
  post-event system-identification accuracy must be reported separately.

This rule does not hardcode constant velocity or one physics law. It prevents
the model from being trained toward the mean of mutually incompatible futures
while preserving a learned probabilistic prior over those futures.

## 164. Safe deployment versus mutable optimisation state

A guardrail-safe deployment incumbent and the weights from which optimisation
continues are distinct roles.

- A candidate that improves the primary objective but fails a safety guardrail
  must not replace the deployment incumbent.
- The same rejection must not automatically discard the candidate from the
  mutable training trajectory. Downstream causal objectives must be allowed to
  repair its failed guardrail unless the candidate is nonfinite or otherwise
  invalid.
- Checkpoints must record the exact incumbent, fixed reference, mutable
  candidate, rejection reasons, validation protocol, and tensor provenance.

Phase handoffs must therefore preserve useful perception learning even when
the first closed-loop validation says it is not yet safe to deploy.

## 165. Exact continuation and simulator isolation

An exact resume must preserve architecture, data protocol, objective,
simulator, seed, validation semantics, optimiser state, CPU/accelerator RNG,
and the next absolute-step sample. A changed curriculum or objective is a
weights-only initialization into a new timestamped run, not a resume.

Source provenance is captured once when the process starts and reused by every
checkpoint. A long-running process must not claim that later worktree commits
were the source it loaded.

Independent simulator subsystems use independent deterministic RNG streams.
Changing render noise, image settings, or renderer implementation must not
change initial physics, lifecycle, external actuation, or event trajectories
for the same physical seed.

## 166. Convergence evidence

Raw batch loss is not a convergence criterion when batches vary in object
count, visibility, lifecycle, scenario, event support, or forecast horizons.
Training must record the sampled seed/scenario/context, pre-clip and applied
gradient norms, exact additive support, and immutable source/data provenance.

Trend validation must:

- use an explicit fixed seed manifest and batch-one per-episode attribution;
- report pooled and per-axis, per-horizon, per-scenario, cold/mature, and
  deterministic/stochastic metrics;
- use deterministic forecast-anchor support recorded in the protocol hash;
- retain exact count totals rather than averages of per-batch counts.

Frequent trend validation may use a deterministic spread of bounded forecast
anchors while ingesting and scoring every current observation. Promotion
requires a separately declared larger balanced manifest and all broad
guardrails. Four or more comparable corrected-protocol validations are needed
for a plateau claim; a finite hard budget without that evidence is not
convergence.

---

# Part XXVIII — Runtime and continuation integrity amendment

## 167. Interval evidence is part of the causal state transition

Events that occur between two observation timestamps must survive until the
observation at the end of that interval is assimilated. A collision/contact
flag is an interval reduction, not merely the state of the final numerical
substep.

- Dynamics must OR discrete collision evidence across every substep in the
  interval while retaining endpoint contact separately.
- Rollout event logits scored at a requested observation endpoint must
  describe the matching preceding simulator interval.
- Zero-duration propagation contains no new event even when the source belief
  is currently in an event mode.
- Temporal measurement histories and slow-parameter observability gates may
  reset/open from interval evidence; they must not depend on the last substep
  still overlapping.

## 168. Floor support, boundary collision, and sleep are distinct

Only the lower vertical support plane is ground. Side walls and the ceiling
are boundary contacts/collisions and must never be converted into ground
support or sleep.

A sub-threshold inward normal velocity at a supporting plane is cancelled as
a resting constraint while tangential velocity remains active. A single slow
substep must not invent sleeping state. The learned belief dynamics may
preserve an already inferred sleeping posterior only while floor-supported;
the simulator requires sustained support for its configured duration.

## 169. Runtime-usable perception defines measurement selection

Perception checkpoints are selected on proposals that can actually enter the
persistent runtime, using the same lifecycle birth-confidence threshold.
Selection must pool additive true-positive, target, proposal, matched, and
absolute-error totals before deriving:

- runtime-qualified world-position MAE;
- recall;
- precision;
- F1.

Localization without lifecycle-qualified recall is not a deployable
perception result. A candidate with no qualified match cannot be promoted.
The assignment used to derive runtime-qualified evidence must itself exclude
proposals below the lifecycle confidence threshold; a low-confidence accurate
proposal cannot consume a target before an actually usable proposal is
matched. Conversely, a confident proposal on a frame with no visible target
is a false positive and must remain in the pooled precision denominator.
When qualified unmatched proposals exceed free belief slots, allocation is
confidence-ordered with deterministic tie handling. Every recycled slot must
reset all identity-specific fast, slow, modal, residual, uncertainty, and
memory fields before receiving a new monotonic object ID.

## 170. Supported objectives and uncertainty calibration

Loss aggregation must not treat an unsupported frame/horizon/parameter as a
zero-valued training example. Average each objective only over observations
that supply its causal support, then apply the declared fixed horizon
denominator where selection uses fixed horizon weights.

When state position or a structured RGB measurement already has an explicit
robust mean objective, its calibration NLL uses the observed squared error as
a detached target for variance. This prevents the same mean error from being
optimized twice, once with an unbounded inverse-variance multiplier. Forecast
likelihood remains a proper distributional objective and may update its
predictive mean and variance.

World-space covariance propagated from RGB is linearized at detached image
centre/depth coordinates while remaining differentiable with respect to the
predicted measurement variance. A calibration or filter-covariance objective
must not update the centre, radius, or depth mean heads through the covariance
Jacobian.

Collision-conditioned sampling must place a labelled collision at a scored
forecast endpoint whenever the requested event and horizon constraints are
jointly feasible; merely placing an event somewhere inside an unscored window
does not provide a positive event target.

## 171. Phase devices and exact continuation artefacts

One training invocation may use different configured devices by phase when
that is an explicit resolved-config choice. The current profile uses MPS for
the convolution-heavy RGB phase and CPU for the branch-heavy persistent
closed-loop phase. The phase boundary resets optimizer moments, moves the
model once, clears runtime caches, and records both devices in the validation
protocol. This is not permission to change devices during an exact resume.

Exact continuation additionally requires:

- a runtime-source content fingerprint independent of documentation-only Git
  changes, plus complete worktree/commit provenance for audit;
- an explicit phase/handoff marker when two devices can legitimately write
  checkpoints at the same completed step;
- tensor hash, step, selector-version, device, and file linkage for retained
  measurement and rollout checkpoints;
- copying and re-verifying linked selector artefacts when continuation writes
  to another run directory;
- no rewrite of a completed checkpoint during a zero-update inspection.

Only the exact `checkpoints/last.pt` may be resumed in place. A selector or
numbered checkpoint requires a new run name or weights-only initialization so
historical source artifacts cannot be overwritten. A checkpoint saved before
terminal validation records that validation as pending; exact resume must
recover the missing validation without an optimizer update and then persist a
completed marker. Legacy completed checkpoints without this marker retain
byte-preserving no-op behavior.

If a required linked artefact is missing or fails verification, exact resume
must fail loudly. A changed executable source, objective, data protocol, or
device policy uses weights-only initialization into a new timestamped run.

## 172. Narrow MPS execution workarounds remain explicit

Backend workarounds may move a small operation to CPU without changing the
model/data contracts, but the resolved execution policy and validation
protocol must record the move. For PyTorch 2.10 on the current Apple host, the
convolutional RGB backbone and ROI updater remain on MPS while the small global
proposal transformer executes on CPU through differentiable feature/output
copies. This avoids a reproduced data-dependent NaN weight-gradient kernel on
finite full-resolution features.

The workaround is configurable as
`device.global_detector_cpu_on_mps`, defaults on for MPS, and may be disabled
only as an explicit changed execution protocol. Tests must cover finite
nonzero gradients on both devices, global clipping, optimizer updates,
checkpoint round-trip, and a second update after restore. Training resume,
evaluation, and demos deserialize checkpoints on CPU and let model/optimizer
state loading place only required tensors on their owners; they must not map a
saved full optimizer onto MPS merely to evaluate or visualize weights.

---

# Part XXIX — Supported causal optimization and hierarchical stability amendment

## 173. An optimizer step requires causal support

An attempted training draw and a completed optimizer update are distinct. A
persistent closed-loop update may advance the optimizer only when its loss has
explicit differentiable support from at least one of:

- current state, rollout, correction, event, existence, uncertainty, or
  observable physical-parameter supervision; or
- a valid fast-ROI slot tied to the persistent runtime state.

A global-discovery auxiliary loss alone cannot consume a causal update. When a
sampled window has no supported causal gradient, training must skip the
optimizer step, advance the deterministic draw counter, record the skip and
support counts, and retry only up to a declared finite limit. Checkpoints must
distinguish attempted draws from completed updates so exact continuation
cannot silently change the sample stream.

Fast-ROI supervision must keep identity mapping, ROI validity, crop evidence,
reliable exact geometry, existence validity, and visibility validity as
separate masks. A valid empty or missed crop supplies negative existence and
visibility evidence, but it must not fabricate centre, depth, colour,
appearance, likelihood, or world-position targets. Selection precision counts
every eligible confident ROI output, including false positives, and temporal
ROI pretraining uses adjacent frames with the same cache contract used online.
Global and fast measurement objectives are support-normalized independently
and combined with fixed declared weights.

Training viability has separate deployment and mutable-optimisation contracts.
A deployable handoff or causal candidate must retain declared minimum current
and future coverage, a declared fraction of its fixed reference, and complete
scenario support. The first unsupported candidate is a rejected/reference
artifact, never a synthetic best checkpoint. Zero physical support retains
additive counts and an explicit unsupported marker; it must neither fabricate
zero RMSE nor abort validation.

Deployment rejection alone must not reset the mutable optimizer trajectory.
A finite candidate whose pooled current and all-horizon forecast coverage
remain above the absolute configured floors continues training even when a
scenario slice is unsupported, a reference-relative coverage floor fails, or
a broad selection guardrail rejects promotion. Those are precisely the
deficits subsequent causal updates must be allowed to repair. Restore the
verified rollout incumbent and reset optimizer state only when a well-formed
finite candidate's pooled current/all-horizon coverage falls below those
absolute floors. A nonfinite or structurally invalid candidate instead fails
closed under the numerical/schema integrity rules. Validation checkpoints must
record both contracts and their independent failure reasons. This distinction
implements the mutable/deployment separation required by Section 164.

Pooled accuracy must not hide a missing or regressed scenario. Every declared
scenario needs at least one episode in the fixed validation manifest and
complete current/horizon physical support before a candidate can be promoted.
The ordered balanced scenario list contains unique family names, so a
validation budget at least as large as that list deterministically visits
every family rather than silently spending residues on duplicate entries.
The selector persists each scenario slice and applies the same broad
non-regression checks within each slice, plus the absolute and
reference-relative current/forecast coverage floors used to protect causal
training. An aggregate score improvement with one unsupported or collapsed
scenario is a rejection, not convergence evidence.

Recursive structured-interaction gradients may be locally clipped before the
whole-model clip when a declared subsystem repeatedly dominates the update.
This changes optimization scale, not forward dynamics capacity. The resolved
protocol must include both clip limits, and every logged update must expose the
raw total norm, raw subsystem norm, subsystem coefficient and applied norm,
pre-global norm, global coefficient, total coefficient, and final applied
norm. A local clip must not hide the original raw norm or be described as
convergence evidence.

---

# Part XXX — Lifecycle, identity, supervision, and gradient integrity amendment

## 174. Tentative evidence is not persistent physical state

An unmatched global or recovery measurement may require multiple consistent
detections before birth. Until confirmation, its bounded evidence is detached,
sensor-local observation history rather than `WorldBelief`: it has no permanent
object ID and cannot participate in dynamics, filtering, rollouts, or
slow-parameter identification. Candidate history is keyed by modality and
sensor, accepts only strictly later timestamps, and matches detections within a
declared finite world-space gate. Gate impossible pairs before Hungarian
assignment with valid-cardinality taking precedence over distance. Allocate a
new monotonic ID only after the configured consecutive confirmation count.

All gated assignment layers follow the same ordering rule. Core association,
tentative confirmation, and first-time privileged target mapping must exclude
inadmissible edges before solving, rather than solve first and discard invalid
pairs afterward. Existing persistent target-to-ID mappings remain locked while
the target is active so training penalizes swaps; a new simulator target
mapping is permitted only within the physical evaluation gate.

Fast ROI measurements are prior-conditioned evidence for a particular
persistent ID. They carry their source belief slot and object ID and may update
only that source if the ordinary uncertainty/confidence gates pass. They may be
rejected, but never cross-assigned to another persistent identity. Global
discovery measurements remain free to use gated Hungarian association.

Local RGB component ownership is part of the same identity contract. When two
disconnected foreground components have indistinguishable nearest support for
one source-conditioned ROI, the fast path must mark the measurement ambiguous
and retain the predicted centre instead of allowing sampling or floating-point
ties to choose an identity-bearing correction. Large or ambiguous recovery is
the responsibility of global discovery. The fast path must expose its local
ownership ambiguity/margin for diagnostics, and a change to this rule advances
the rollout-validation protocol because it changes the online trajectory.

Slow drag/restitution supervision opens only for accepted runtime
associations, not births or simulator-visible targets. Per-target temporal
parameter history records the associated runtime ID and resets before forming
a label whenever that ID changes. Burn-in evidence uses the same physical
distance gate as the optimized window.

## 175. Perception gradients must not starve state learning

When causal RGB gradients repeatedly dominate the whole-model clip, the
declared RGB observation module may be locally clipped before the independent
interaction-local and final whole-model clips. Local groups must be disjoint.
The trainer reconstructs and logs the true raw total norm together with each
local raw norm, coefficient, applied norm, the pre-global norm, global
coefficient, total coefficient, and final norm. The perception-local cap is a
causal-stage stability policy and is disabled during paired RGB pretraining,
where the ordinary whole-model clip retains its original semantics. Clip
limits and the bounded causal global-perception adaptation window are resolved
configuration and checkpoint-protocol semantics.

---

# Part XXXI — Validation support and launch-failure integrity amendment

## 176. Deterministic accuracy requires auditable causal opportunity

An observation-interval intervention probability is not an episode-level
probability. Scenario parameters must state the sampling cadence, and fixed
validation manifests must be checked against the resulting event sequence.
A stochastic-intervention scenario must contain real interventions while also
retaining enough clean windows at every declared horizon to evaluate the
deterministic predictor.

Because object dynamics interact, an unseen external actuation on any object
can invalidate a point target for the complete coupled scene until a later
observation exposes it. Training and standalone evaluation must use the same
causal mask for deterministic position, velocity, event, collision-conditioned,
and correction-improvement metrics. Distributional likelihood and calibration
continue to score the realised stochastic outcome. Reports must publish both
the censored deterministic support and the uncensored calibration coordinate
count so an apparent RMSE improvement cannot be manufactured by dropping hard
futures.

Promotion support is stronger than a nonzero pooled denominator. Every
scenario must satisfy configured positive minima for:

- label-only causally predictable targets at every horizon;
- matched predicted targets at every horizon; and
- independently supported validation episodes.

Exact additive counts, per-seed support, per-scenario support, and the configured
floors are checkpoint evidence. Missing metric schema is an implementation
error and must fail loudly; only a well-formed zero/insufficient-support result
may become a truthful rejected candidate. Fully resolved per-scenario simulator
parameters, simulator version, metric version, support floors, seed manifest,
and execution policy are part of the validation-protocol hash.
Every retained derived selector value, including pooled/scenario and
axis-resolved metrics, must be reproducible from the stored exact additive
evidence. A selector artifact whose derived score contradicts its raw sums is
invalid even when its tensor hash and internally recomputed weighted score
match.

## 177. Initialization failure must be recoverable and terminal process state explicit

A weights-only initialization is a candidate, not an assumed incumbent. Its
first broad validation may establish a fixed diagnostic reference while being
unsupported for deployment. Persist the candidate and reasons, continue the
declared causal optimization path, and retry broad validation only at the
configured evaluation cadence and terminal boundary. Do not run the complete
validation manifest before every optimizer update. The first later supported
incumbent must pass every available fixed-reference and training-viability
guardrail; incomplete initialization evidence never becomes a synthetic best
checkpoint.
An unsupported diagnostic artifact does not constitute a numerical fixed
reference. The checkpoint carries a durable incomplete-reference-comparison
marker across both in-place and branched exact resumes. The first later
supported candidate establishes the complete fixed reference but cannot
promote itself; only a subsequent supported candidate can be compared with and
promoted against that reference.

Every fresh CLI training invocation writes a timestamped starting state before
expensive initialization, changes it atomically to running immediately before
entering the trainer, writes a terminal failure artifact with exception type,
message, and traceback on an in-process failure, and writes a completed state
only after terminal validation succeeds. A convergence supervisor that proves
the trainer process disappeared must also change this primary state to failed
and retain the prior live state in append-only history; an operating-system
kill must not leave a stale starting/running marker. Persistent macOS launches
are one-shot launchd jobs
wrapped by `caffeinate`, with `KeepAlive=false`. A failed trainer must remain
failed rather than being silently relaunched against an occupied run directory.
A per-run exclusive training lock prevents concurrent exact-resume writers,
and every failed attempt is retained in append-only history even when a later
attempt succeeds. Interrupts and cleanup failures are terminal evidence rather
than unrecorded or masking exceptions.
A convergence supervisor monitoring an older KeepAlive job must boot it out on
either verified completion or verified initial-process failure.

---

# Part XXXII — Cadence semantics, progress observability, and finite-state amendment

## 178. Global cadence counts complete observation frames

`model.rgb.global_every_steps` is the positive integral distance between
global-discovery frames, including the global frame itself. A value of three
means:

```text
GLOBAL, FAST_ROI, FAST_ROI, GLOBAL, FAST_ROI, FAST_ROI, GLOBAL
```

It does not mean three fast frames followed by a fourth global frame. Global
discovery and recovery both reset the sensor-local fast-frame counter. Tests
must assert complete mode sequences rather than only forcing the counter to a
threshold.

Changing this counter interpretation changes persistent observations,
association, lifecycle, and future rollouts for identical YAML. It therefore
requires a new rollout-validation protocol version and a fresh weights-only
qualification; historical selector/reference artifacts from the old cadence
semantics are not comparable. The measurement-only protocol, simulator
version, and selection formula need not change when their behavior and
evidence schema are unchanged.

## 179. Long validation remains atomic but visibly alive and finite

Checkpoint selection remains full-manifest and atomic: no partial validation
may become a score, reference, incumbent, or convergence point. Separately,
each validation episode must update a durable, atomic progress snapshot and
flush a human-readable heartbeat. The snapshot records at least phase/split,
validation kind, completed and total batches/episodes, elapsed and last-batch
time, process ID, last seed/scenario when attributable, and the exact protocol
hash. An interruption records the exception type and last completed progress
without fabricating partial metrics.

Every standard evaluation must likewise write a durable atomic progress file
by default; its timestamped output is planned and an `initializing` event is
written before model construction or checkpoint loading. A caught exception or
interruption atomically replaces that event with terminal type/message and the
last completed progress. An uncatchable process death remains explicitly stale,
never implicitly complete. Console JSON is optional presentation, not the
persistence gate.
The repository provides one read-only local monitor that recursively discovers
timestamped training and evaluation artifacts, prefers a verified-live run,
and reports phase, step/target, robust rolling training trends, fixed-
validation decisions, per-horizon accuracy, device placement, checkpoint age,
and hard nonfinite/failure/staleness signals. Its default polling is relaxed,
unchanged snapshots are suppressed apart from a coarse heartbeat, and it must
not import/execute the model, load checkpoints, consume accelerator memory,
contact a service, or mutate run state.

Training workers must not be started or allowed to prefetch while
initialization or handoff validation is still running. Construct the training
loader deterministically, but create its iterator only when the first training
draw is actually required. Sustained macOS profiles use a low explicit worker
count, one prefetched batch per worker, and non-persistent workers unless a
measured profile justifies a larger resident pool.

Every successful optimizer call is followed immediately by a grouped
finite-state check over floating/complex model parameters and all optimizer
state tensors. Optimizer step counters must be scalar, finite, and
nonnegative. Before an atomic checkpoint replacement, validate model
parameters and persistent buffers, optimizer state, step counters, and any
scheduler tensors. Loading validates the payload before mutating the
destination model or optimizer. A corrupt candidate must leave an existing
checkpoint byte-for-byte intact and terminate the invocation rather than
becoming resumable state.

---

# Part XXXIII — Integration-grid, prepared-propagation, and launch-QoS amendment

## 180. Belief dynamics use the intended physical integration grid

The simulator and belief dynamics must interpret an observation interval that
is nominally an integral number of physics ticks as the same tick count.
Timestamp storage in float32 may put a mathematically integral
`elapsed / max_substep` ratio a few representational units above the integer;
a literal ceiling must not invent an additional substep. Substep-count
selection may snap only to a nearest integer that is indistinguishable at the
elapsed tensor's declared floating-point precision. A genuinely longer,
non-integral interval continues to use the ceiling, elapsed time is still
divided across every chosen substep, and interval collision/contact evidence
is accumulated across them all.

Changing this rule changes recursive dynamics, learned-residual execution, and
event timing for identical timestamps. It therefore requires a new rollout
validation protocol and fresh qualification evidence. Tests cover ordinary
20 Hz float32 timestamp differences against the 120 Hz simulator grid, full
one-second event query plans, non-integral intervals outside tolerance,
batched maximum elapsed time, and interval event accumulation.

## 181. One causal propagation may serve supervision and assimilation

The predict stage for one observation timestamp is computed once. Training may
inspect that exact prior for measurement supervision, correction diagnostics,
and future comparison, then pass a typed one-use prepared propagation into
the ordinary `OnlineWorldModel.ingest` path. The prepared value must retain
the original elapsed time and all interval event auxiliaries needed by
filtering and parameter observability.

Prepared propagation is not a second belief authority and may not bypass the
runtime loop. Consumption verifies that it was made from the current
persistent source state for the requested timestamp, batch, device, and dtype;
stale, reused, wrong-source, or wrong-time values fail loudly. Assimilation,
association, innovation, lifecycle, scheduler, cache, diagnostics, and
timestamp update semantics remain identical to an ordinary ingest. Setting
the runtime state to the prior and ingesting with zero elapsed time is
forbidden because that would erase interval evidence and temporal velocity
semantics. Forward-state, metric, event, and gradient parity tests guard the
optimization.

Preparation and consumption are one atomic model-revision operation: dynamics
parameter/buffer identity and mutation versions plus training/evaluation mode
must still match. The zero-copy mutation guard depends on PyTorch tensor
version counters, so prepared propagation is intentionally unsupported inside
`torch.inference_mode()`; use `torch.no_grad()` for no-gradient validation and
deployment paths that consume this contract.

Rollout callers may explicitly decline trajectory auxiliary stacking when they
consume only state, uncertainty, activity, and event tensors. The public
rollout default retains complete auxiliary output. Validation-anchor batching
is a separate optimization and must not be introduced without fixed-manifest
selector parity and a measured throughput benefit.

## 182. User-requested sustained training is not background maintenance

A one-shot macOS training job remains `KeepAlive=false`, but it must not be
classified as launchd `Background` work. Sustained training is explicitly
requested compute; the background classification can throttle multi-core CPU
dynamics enough to make a healthy validation resemble a stalled run.
Launchers use the portable Standard/default process classification and retain
`caffeinate` only for sleep prevention. A matched validation timing control
must be checked before committing to another multi-day campaign, and process
priority, CPU utilization, progress heartbeats, and numerical health must be
distinguished in status reports.

At phase-device transitions, move the complete model, reset transient runtime
state, collect unreachable Python objects, and release the previous MPS/CUDA
allocator cache. Training metrics record a process maximum-resident-set
high-water mark so resource growth can be distinguished from loss variance.
The selector remains atomic and full-manifest; memory mitigation must not
shorten validation or alter its fixed evidence.

---

# Part XXXIV — Modular qualification and fast-ROI isolation amendment

## 183. Module boundaries are checkpoint-selection and optimisation boundaries

The architectural separation between RGB discovery, fast residual
measurement, belief correction, identification, and dynamics must be usable
experimentally rather than existing only as class structure. An offline
candidate may retain a verified base checkpoint and import declared module
prefixes from a compatible donor. Such composition validates the complete
state-dict schema, shapes, dtypes, finite tensors, configuration semantics,
source paths, module prefixes, and blend weight. It never imports optimizer or
RNG state. A composed checkpoint is only a candidate until the unchanged full
balanced RGB-only validation protocol accepts it.

Shared RGB backbone stages are not fast-path-exclusive merely because the ROI
updater consumes their features. After a bounded global adaptation phase,
continued gradients through shared stages can move the global discovery
distribution and reduce persistent target coverage. Training therefore
supports an explicit `state_dynamics_fast_roi` scope that adapts dynamics,
belief update, slow identification, the ROI updater, and its ROI-only fast
projection while keeping all shared backbone stages, global feature-pyramid
projections, and the global detector frozen. The broader
`state_dynamics_roi` scope remains an explicit ablation that also adapts the
first two shared backbone stages.

Changing the trainable scope or global-adaptation duration starts a new
weights-only run and is recorded in resolved configuration and provenance.
It does not relax selection. Promotion still requires primary-score
improvement plus pooled, per-axis, every-horizon, lifecycle/identity, event,
calibration, coverage, and per-scenario non-regression against the same fixed
reference. Module composition is diagnostic evidence when rejected, not a
means to relabel a Pareto tradeoff as an accepted baseline.

---

# Part XXXV — Trainable-path objective-integrity amendment

## 184. Frozen auxiliary losses are diagnostics, not optimisation terms

Every causal auxiliary loss must have a differentiable path to at least one
currently trainable parameter before it enters the optimised total. A module
container is not sufficient evidence: path-exclusive parameters under the
same container do not make an unrelated output trainable. In particular, the
ROI-only RGB `fast_projection` must not classify global discovery as
trainable when the shared backbone stages, pyramid projections, and global
detector are frozen.

A frozen global-discovery loss may still be evaluated and logged under an
explicit frozen diagnostic name. It must not be averaged with the fast-ROI
measurement loss, change that loss's scale, consume gradient weight, or make
the total loss appear to converge or oscillate. Scope regressions must test
both `requires_grad` declarations and the final closed-loop objective terms.
Correcting a trainable-path classification changes the training objective and
therefore requires a fresh weights-only run rather than exact resume.

## 185. Branch coefficients and causal scope transitions are fixed protocol

Relative global/fast RGB measurement weights have one fixed denominator even
when a branch is frozen or lacks support on a particular draw. An unavailable
branch contributes zero; the available branch must not inherit its coefficient.
This prevents a trainability or support decision from silently doubling an
auxiliary gradient relative to state and rollout objectives.

A sustained run may declare one causal trainability-scope transition at an
exact completed-causal-update boundary. Both scopes and the boundary are
resolved configuration and continuation semantics. The intended evidence-led
use is a short `fast_roi` phase followed by `state_dynamics`: first improve
residual visual localization while the accepted belief/dynamics stack is
fixed, then freeze perception while velocity, identity, coverage, and rollout
dynamics adapt. Every training metric records which side of the boundary
produced the update. The transition never promotes the intermediate tensors;
the unchanged full RGB-only selector remains authoritative.

## 186. Measurement auxiliaries may train only their perception branch

A measurement loss being differentiable through a predicted prior is not
evidence that its perception branch is trainable. Global-discovery
supervision may update only global detector/shared-pyramid perception paths;
fast-ROI supervision may update only the shared fast encoder, ROI-exclusive
projection, and ROI updater. When those perception paths are frozen, retain
their losses as detached `frozen_*_measurement` diagnostics and exclude them
from the optimized measurement term and causal fast-support accounting.

In particular, a `state_dynamics` phase must not let fast-ROI geometry,
existence, colour, or likelihood auxiliaries steer dynamics or the posterior
corrector through the ROI's prior-conditioned input. Current-state, rollout,
event, uncertainty, correction, lifecycle, and observable-parameter losses
are the authoritative gradients for the physical stack. Any change to this
routing is objective protocol and requires a fresh weights-only run.

## 187. Rollout likelihood calibrates uncertainty without duplicating the mean objective

The deterministic rollout point loss is the sole supervised gradient for a
predictable forecast mean. Rollout Gaussian likelihood retains the realised
squared error as a detached calibration target and updates forecast variance
only; it must not send an additional inverse-variance-weighted gradient into
the mean. This matches the state-uncertainty rule and prevents low predicted
variance from overwhelming the declared per-axis/horizon point objective.

After an unseen external actuation, deterministic point/event means remain
censored for the coupled scene. The realised future may still train the
predictive variance through proper likelihood, but it must not train a mean
for a future that was not identifiable at the causal anchor. Regressions must
prove both properties directly: no mean gradient and a finite variance
gradient that widens under an under-dispersed hidden outcome.

## 188. Shared-regime optimization requires balanced gradient evidence

Balanced validation cannot repair an optimizer that sees one or two randomly
selected scenario families per update. When later shared-state updates improve
one family while degrading another, a sustained profile may require each
optimizer update to aggregate equal support from every declared scenario.

Scenario-balanced sampling must:

- bind scenario membership to the explicit dataset seed manifest;
- include the same positive number of examples from every unique scenario in
  each optimizer update;
- shuffle within each scenario pool independently rather than cycling one
  fixed tuple forever;
- remain deterministic and exactly reconstructable from the absolute data-draw
  index after interruption;
- reject unequal pools, partial batches, and batch sizes that cannot represent
  every scenario equally;
- record every contributing seed and scenario in training metrics; and
- leave the fixed, batch-one, full-manifest RGB-only selector unchanged.

This is an optimization protocol change and therefore starts a new
weights-only run. It does not average validation slices, relax per-scenario
guardrails, or make a smooth training loss a convergence criterion. Batch size,
scenario balancing, learning rate, trainable scope, and seed manifest remain
checkpoint and exact-continuation semantics.

---

# Part XXXVI — Innovation integrity and staged attention scaling amendment

## 189. Learned correction must be anchored to supported innovation

The fast corrector may not infer an unconstrained state displacement from only
pooled innovation statistics. Pooled mean, norm, and maximum discard the axis,
sign pattern, and state-field support needed to distinguish a useful residual
from a scenario prior. Every learned fast-state mean residual must therefore be
anchored to explicit innovation in the corresponding world-state component.

For each associated pair, construct a typed state-space innovation and
component confidence from the measurement's declared
`supported_state_fields`. A position-only observation may correct position;
documented temporal position coupling may also correct velocity. It may not
rewrite orientation, angular velocity, or modal state merely because those
fields share a packed tensor. Per-axis localization confidence and robust
surprise influence must mask learned mean and variance corrections as well as
the analytic proposal. With zero supported innovation, the learned mean update
is exactly zero. The learned network remains free to predict a positive or
negative bounded gain, so this is evidence conditioning rather than a
hardcoded physical law.

Historical checkpoints retain their original unanchored semantics unless a
new resolved protocol opts into the corrected path. An opt-in changes forward
semantics and requires a new weights-only campaign and complete fixed-manifest
qualification; it is never an exact resume. Any output head whose mathematical
meaning changes from an absolute residual to an innovation gain must be reset
to its declared neutral initialization with explicit checkpoint provenance.
Loading its old numerical weights under the new interpretation is not a valid
weights-only transfer.

When a neutral reset temporarily removes a previously useful correction, the
recovery curriculum may train the updater alone while freezing dynamics,
identifier, and perception. It must regain broad fixed-manifest accuracy before
joint state/dynamics adaptation; joint co-adaptation from a cold correction
head is not evidence that the new correction learned the intended function.
If only one output head changed semantics, the first recovery stage must freeze
the compatible updater trunk and sibling heads and train that reset head alone.
Only broaden updater trainability after an exact fixed validation shows that
the isolated head is useful but capacity-limited; do not let a broad recovery
scope erase compatible behavior merely because it belongs to the same module.

An output head with axis-labelled rows may be composed and qualified one row
at a time when joint recovery creates a cross-axis runtime regression. This is
an ablation and optimization control, not an assumption that full trajectories
are axis-independent: one row may change later association, identity, and all
future axes. Every row candidate therefore runs the complete fixed RGB-only
selector. A row-restricted training scope must preserve excluded parameter
rows exactly across gradients, optimizer moments, and decoupled weight decay;
zeroing gradients alone is not sufficient. Continue only from a guardrail-
clean row composition, and retain the prior incumbent until the corrected
model also passes the broader deployment reference.

## 190. Attention operates on predictive abstractions

The Transformer's durable contribution is content-dependent interaction among
tokens, parallel training, and a scalable residual backbone. Project Orpheus
uses those properties without turning RGB patches or an opaque sequence cache
into the source of truth. The scalable token set is derived from typed state:

- entity tokens carry persistent identity, kinematics, uncertainty, lifecycle,
  slow parameters, appearance, abstraction kind, timestamp, and freshness;
- relation tokens carry candidate pair geometry, relative motion, contact and
  observability evidence;
- event tokens carry discrete jumps, interventions, and interval evidence;
- scene/camera tokens carry global fields and calibrated sensor context; and
- bounded history tokens summarize sparse belief/innovation changes rather
  than replaying the full RGB history.

A pre-normalized attention backbone may propose residual forces, event logits,
uncertainty growth, abstraction refinements, and supported correction gains.
Its outputs must decode into typed proposals and pass through the existing
analytic dynamics, event, uncertainty, association, and filtering contracts.
It must preserve masking, permutation consistency, explicit timestamps, cheap
online updates, and the authoritative `WorldBelief`.

The first attention rung is an optional residual around the accepted
interaction graph. It uses RMS pre-normalization, scaled dot-product
multi-head attention, and SwiGLU feed-forward residuals over one scene token,
active entity tokens, and candidate-relation tokens. It has no learned object-
slot position embedding: current-belief entities and relations are sets, and
inventing a sequence order would violate permutation consistency. Rotary or
relative time encoding is reserved for later bounded history tokens carrying
explicit timestamp offsets; it must not be applied to arbitrary padded slot
indices.

Attention output heads start at exact zero and decode only into bounded node
acceleration, antisymmetric pair force, contact/collision logits, event-jump
residuals, and process-noise residuals. Enabling the module must therefore be
numerically identical to the inherited graph before optimization. A weights-
only architecture-growth transfer may accept missing keys only under the new
attention module prefix; every inherited tensor remains required and is
audited bitwise. The first optimization phase trains only the new module, and
its gradients share the existing interaction-local clip and finite diagnostics
before any inherited dynamics/filter/perception parameters may be unfrozen.

The four-block width-128 current-belief pilot is stage A of the Mac rung, not
the complete temporal model. Stage B adds a short timestamped history of
sparse belief/innovation changes only after stage A improves or safely matches
the fixed selector. This ordering separates relational-capacity evidence from
history-capacity evidence and avoids committing scarce Mac compute to two
untested axes at once.

Dense RGB/audio/depth features may later cross-attend into a fixed-size latent
or object-token bottleneck in the style of Perceiver IO. Self-supervised masked
latent prediction in the style of JEPA may pretrain perception and predictive
context. Pixel/video generation may be an auxiliary decoder or uncertainty
visualizer, but is not the primary physical accuracy objective.

Relevant foundations and current evidence include:

- Vaswani et al., “Attention Is All You Need,” arXiv:1706.03762;
- Jaegle et al., “Perceiver IO,” arXiv:2107.14795;
- Hoffmann et al., “Training Compute-Optimal Large Language Models,”
  arXiv:2203.15556;
- Assran et al., “V-JEPA 2,” 2025;
- Joseph et al., “Interpreting Physics in Video World Models,”
  arXiv:2602.07050; and
- Soraki et al., “ObjectForesight,” arXiv:2601.05237.

## 191. Capacity increases require a measured generalization ladder

Do not increase parameter count while a smaller model has a known semantic,
support, numerical, or optimizer defect. First repair and qualify the same
fixed RGB-only selector. Then scale one axis at a time while increasing data
coverage commensurately; a larger undertrained model is not progress.

The initial ladder is:

1. **Corrected control:** existing graph/filter capacity with
   innovation-anchored correction; qualify current, every horizon, every axis,
   every scenario, identity, event, support, and calibration.
2. **Mac attention pilot:** two to four pre-normalized object/relation blocks,
   width 128, four heads, short bounded token history, and roughly 1–4 million
   new parameters. Train on thousands of balanced continuously varied episodes
   and compare against a parameter-matched MLP/graph control.
3. **Single-GPU model:** width 256–384 and six to eight blocks with wider
   perception, masked latent pretraining, more objects/modalities, longer
   horizons, and tens of millions of parameters after the pilot passes.
4. **Foundation-scale encoder/predictor:** larger video pretraining and
   generative auxiliary decoders only when substantial CUDA compute and
   real-video data are available. Distill or adapt its proposals into the same
   explicit belief runtime.

Every rung declares parameter count, episode/token draws, approximate data
passes, peak memory, throughput, optimizer/gradient health, and a predeclared
training budget large enough to expose a plateau. Promotion requires disjoint
validation and test manifests, held-out initial conditions and parameter
combinations, OOD object counts/camera paths, recovery perturbations, and no
material regression against both the smaller accepted model and the analytic
low-complexity prior. Training loss alone, a shorter run, or one improved slice
cannot justify scaling.

## 192. Scene tokens must consume live belief context

A declared scene/camera token may not be implemented as a projection of a
reserved tensor that the online runtime never updates. Before sustained
attention training, audit gradient and checkpoint deltas for every new input
projection. A wholly unchanged projection after more than one supported
update is a defect unless the specification explicitly declares that input
inactive.

The stage-A scene token is derived directly from authoritative
`WorldBelief`: global code, summaries of global uncertainty, gravity, camera
pose, camera linear/angular motion, intrinsics, summaries of camera
uncertainty, and calibration state. Variance summaries keep the dynamics
interface independent of modality-specific covariance packing widths. The
scene token may aggregate entity/relation tokens through attention, but a
learned bias or type embedding alone does not satisfy the scene-context
contract.

This input repair preserves exact zero-output graph identity and the isolated
attention-only optimization scope. A pilot trained with the dead scene input
is retained as diagnostic evidence but cannot be resumed or counted toward
the corrected capacity rung; corrected training restarts weights-only from
the same protected graph control.

## 193. Mixed-unit token inputs require bounded pre-projection conditioning

Typed token features may combine latent coordinates, log variances, physical
SI-like quantities, homogeneous transforms, and sensor coordinates. Transformer
block pre-normalization occurs after token projection and therefore cannot
protect the input projection's weight gradient from raw feature scale. A
finite loss plus a repeatedly clipped projection gradient is a conditioning
defect, not evidence that a larger clip or longer training is required.

The stage-A scene vector receives a fixed non-affine RMS normalization
immediately before its learned projection. This bounds projection input norm
independently of pixel-space camera scale, adds no trainable scale parameter,
and preserves exact zero-output graph identity. Absolute analytic quantities
remain available to the structured dynamics; this normalization conditions
only the learned residual token. Any later typed input with heterogeneous
units must either use an explicit documented nondimensionalization or prove
equivalent bounded projection-gradient behavior on the declared data range.

A sustained capacity campaign must stop when sampled local gradient
coefficients show a new systematic order-of-magnitude collapse attributable
to an architectural input path. Preserve the diagnostic metrics, repair the
conditioning, pass a focused extreme-scale regression and complete gates, and
restart weights-only from the protected smaller control. Do not resume or
count the flawed campaign toward convergence.

## 194. Typed proposal heads require hierarchical gradient isolation

A typed decoder may jointly emit event logits, forces, impulses, and process
noise. These outputs share representation capacity but do not necessarily
share loss frequency or gradient scale. Rare class-balanced collision
supervision may put a large direct gradient on the collision-logit row even
when the complete loss, support, state, and all other proposal rows are
ordinary. Clipping the whole interaction module alone keeps the update finite
but suppresses unrelated force, uncertainty, and token learning by the same
small coefficient.

When repeated evidence localizes such spikes to one declared proposal row,
apply an optional row-local norm cap before the complete interaction-local and
whole-model caps. This is an optimizer isolation mechanism, not a hard-coded
collision rule: forward predictions, labels, event semantics, and the shared
attention representation are unchanged. The cap is part of the resolved
training and validation protocol and must be absent or explicitly configured.

Diagnostics must retain the true raw row norm, row coefficient and applied
norm, the interaction norm before and after row isolation, the interaction
stage coefficient, the effective total interaction coefficient, the true raw
whole-model norm, and the final applied norm. Reconstruct raw hierarchical
norms algebraically so local clipping cannot make training appear better
conditioned than it was. The dynamics auditor must treat any configured row
coefficient below its severe threshold as visible warning evidence.

Decoder parameter-row clipping happens only after autograd has already sent
that row's signal through the shared output normalization, attention blocks,
and input projections. It therefore cannot protect the shared representation
when recursive rollout amplification occurs upstream of the decoder. When an
exact replay localizes this failure, additionally cap the backward gradient on
the typed decoder output tensor at every attention invocation, before it enters
the decoder or shared stack. Keep node x/y/z, collision, and joint
normal/tangent force as separately configured semantic groups. These caps alter
backward conditioning only; forward values, tensor contracts, and checkpoint
weights remain unchanged. Retain the decoder parameter-row caps as a second
layer because repeated individually bounded invocations can still accumulate a
large decoder update.

Log each typed-output group's invocation count, root-sum-square raw and applied
norms, minimum per-invocation coefficient, and effective aggregate
coefficient. Parameter-gradient diagnostics remain raw with respect to the
later row/module/global hierarchy but necessarily occur after typed-output
backpropagation conditioning; reports must state that boundary rather than
misrepresenting them as the counterfactual no-hook parameter gradient. The
offline auditor treats a severe typed-output coefficient as visible evidence
alongside every later hierarchy.

The stage-A Mac pilot config caps the collision-logit row of the typed relation
decoder at norm 1.0 before the existing interaction cap of 1.0. This choice is
supported by two severe spikes exactly 128 updates apart at steps 152 and 280,
both on deterministic frames 7--11 contact-heavy batches, and by checkpoint
Adam moments localizing the dominant variance to the collision-logit row.
The pre-repair campaign is diagnostic only and cannot resume or count toward
convergence; repaired training restarts weights-only from the protected graph
control.

## 195. Severe shared-gradient recurrence requires complete localization

A row-local repair is qualified only for the failure it actually isolates. If
the same deterministic schedule position later produces severe complete-block
clipping while the repaired row is ordinary, stop at the last durable clean
checkpoint. Do not infer that the prior row remains causal, add another cap by
guess, lower the whole learning rate, or continue merely because gradients and
parameters remain finite.

Before any further optimizer repair, record the true raw norm of every trainable
attention parameter and each typed node/relation decoder row before any local or
global mutation. The diagnostics must be read-only, finite-checked, use stable
semantic names, and distinguish contact/collision logits, normal/tangent force,
impulse multiplier/additive, process noise, and node x/y/z outputs. Reproduce
the failure by exact optimizer/RNG/data continuation from the last clean
checkpoint and compare seeds, physical event counts, objective support, loss,
and association/lifecycle evidence. Only the reproduced dominant path may
receive a targeted conditioning or isolation change.

In the collision-isolated stage-A campaign, update 280 still produces a raw
interaction norm of `17.7050` and retains only `0.05648`, while the collision
row norm is an ordinary unclipped `0.23553`. The repaired run therefore stops
at its exact durable step-256 checkpoint. It cannot count toward convergence;
its step-280 update is diagnostic evidence for the complete-localization replay.

The subsequent force-row-isolated campaign exposes the deeper limitation at
the same deterministic update: decoder parameter-row clipping sees a raw
`989.7965` force norm only after that signal has produced order-one-to-ten
gradients throughout shared projections and blocks. An exact diagnostic branch
from its durable step 256 applies per-invocation typed-output caps of `0.1`.
On the identical update-280 seeds and frames, the later parameter-gradient norm
falls from `995.5391` to `10.8330`, the maximum shared projection/block norm is
`0.0851`, and the post-row interaction stage retains `0.6979` for shared
learning rather than an effective `0.0010`. The update is finite, supported,
and applied. Severe typed-output and decoder-row coefficients remain truthful
localized warnings; the diagnostic branch is not selector or convergence
evidence. A corrected campaign must restart weights-only from the protected
graph control and pass the complete fixed selector and plateau rules.

## 196. Scale the data and model only from a qualified stable rung

The original Transformer established that scaled dot-product multi-head
attention gives short content-dependent paths and parallel computation, but it
did not establish that parameter count alone creates physical abstraction or
out-of-distribution generalization. Modern dense transformers commonly add
pre-normalization, RMSNorm, gated feed-forwards such as SwiGLU, and efficient
attention kernels. Grouped-query attention, sparse experts, and IO-aware
attention primarily reduce long-context or large-batch cost. They are not
automatic accuracy improvements for the current set of at most 22 structured
tokens.

A primary-source review of later dense systems reinforces that restraint.
Llama 3 attributes most gains to data quality/diversity and training scale,
keeps a stable dense Transformer, and uses grouped-query attention primarily
for decode efficiency. Its recipe combines linear warmup, cosine decay,
staged context growth, explicit short-context recovery checks, and final
annealing/checkpoint averaging. Gemma 3 mixes local and global attention to
control long-context KV-cache memory. V-JEPA 2 scales curated video data,
model size, duration, and resolution progressively and applies a cooldown only
after measured plateauing. These are scaling-process lessons; GQA, local
attention, or a billion-parameter video encoder do not solve Orpheus's current
short-token recursive-gradient defect.

Project Orpheus therefore retains dense RMS-pre-normalized attention and
SwiGLU for the Mac rung. It does not add object-slot positional encoding or
RoPE to unordered entity/relation sets. Relative/rotary time features become
eligible only for explicitly timestamped bounded-history tokens. Flash-style
CUDA kernels, grouped-query attention, sparse experts, and distributed
sharding become eligible only when measured token length, memory, or compute
profiles identify the bottleneck they solve.

Scaling experiments use a fixed ladder rather than one expensive jump:

1. qualify the 3.00M-parameter width-128/four-block model through repeated
   fixed selectors and a declared plateau after all known semantic and
   optimizer defects are repaired;
2. run matched small-rung controls that vary one axis at a time (data draws,
   depth, width, then bounded temporal context), preserving the accepted model
   as a fixed reference;
3. fit empirical validation-loss/error versus parameters, examples, and
   compute, and advance only when the smaller model is capacity-limited rather
   than data-limited or optimization-limited;
4. increase continuously varied balanced episodes with parameters, following
   the compute-optimal lesson that larger undertrained models are wasteful;
5. require disjoint RGB-only validation/test, held-out physical-parameter
   combinations, novel object counts/camera paths, and per-scenario/every-
   horizon non-regression before promotion; and
6. move the unchanged contracts to a single CUDA GPU only after the Mac rung
   predicts a useful gain and records a portable resolved config, manifest,
   checkpoint, and throughput/memory budget.

The first concrete capacity census is fixed before running the ladder:

- the control is `3,004,656` total parameters with `1,103,626` in the
  width-128/four-block attention residual;
- a depth-only six-block candidate is `3,530,480` total with `1,629,450` in
  attention;
- a width-only 192/four-block candidate is `4,342,896` total with `2,441,866`
  in attention; and
- the first single-CUDA candidate is width 256/six blocks, `8,305,648` total
  with `6,404,618` in attention.

Using the control's 8,192-update/65,536-episode minimum as the data floor and
rounding parameter-proportional budgets upward to complete 512-update selector
intervals gives concrete lower bounds of 9,728 updates/77,824 episodes for the
depth-six rung, 12,288/98,304 for width 192, and 23,040/184,320 for the first
width-256/depth-six CUDA rung. These are minimum exposure budgets, not stopping
rules: every rung still continues to the same fixed-manifest plateau condition.

These counts are design points, not evidence that the larger candidates should
win. The active control's full 8,192--24,576-update plateau trajectory is also
the data-only learning curve; do not discard it and rerun an arbitrarily short
"data control." A capacity candidate receives at least the same continuously
varied balanced episode draws and increases them in proportion to parameter
growth, rounded upward to a complete 512-update selector interval. Report
wall-clock time and RGB/closed-loop device placement because small structured
tokens may remain CPU- or simulator-bound even when dense video work benefits
from MPS/CUDA.

Maximal-update parameterization may be tested as a separate matched control
before transferring learning-rate choices across materially different widths.
It is not retrofitted silently into an existing checkpoint and is not assumed
to transfer merely because it does so for language models. Conventional
parameterization remains the reference until the same fixed physical selector
shows a stable advantage.

The trainer supports exactly two declared closed-loop learning-rate protocols:
the historical `constant` rate and opt-in `warmup_cosine`. Warmup and cosine
durations are expressed in absolute causal optimizer updates, excluding any
measurement-pretraining phase. The cosine duration is explicit and never
inferred from mutable `training.steps`; extending a convergence campaign must
therefore preserve the complete past and future schedule. After the declared
decay duration, the rate remains at its configured minimum scale. Historical
checkpoints normalize missing schedule fields to `constant`; changing schedule
semantics requires weights-only initialization into a newly versioned run, not
an exact resume. Schedule implementation or smoke evidence does not authorize
starting a successor before the current fixed selector rejects its constant-
rate candidate.

After structured stage A and bounded timestamped history qualify, the next
perception scale axis is masked latent video pretraining with dense spatial and
temporal supervision, followed by distillation or cross-attention into typed
entity/relation proposals. The dense encoder may retain distributed physical
features that explicit probes miss, but `WorldBelief` remains the online source
of truth and current/horizon state accuracy, identity, events, calibration, and
OOD physics remain the promotion criteria. Video reconstruction quality or
video-language benchmark accuracy alone is insufficient.

Training loss is a diagnostic. Convergence requires the existing broad fixed-
manifest plateau rule, and generalization requires improvement on manifests
that were not used for architecture or checkpoint selection. Scaling video
generation alone is insufficient evidence: recent physical-law evaluations
show strong in-distribution performance can coexist with failure to extrapolate
the underlying dynamics. The explicit object/relation/event belief and
invariant-tested physics regimes remain the generalization probes.

Primary references for this decision:

- Vaswani et al., “Attention Is All You Need,” arXiv:1706.03762;
- Hoffmann et al., “Training Compute-Optimal Large Language Models,”
  arXiv:2203.15556;
- Grattafiori et al., “The Llama 3 Herd of Models,” arXiv:2407.21783;
- Gemma Team, “Gemma 3 Technical Report,” arXiv:2503.19786;
- Assran et al., “V-JEPA 2: Self-Supervised Video Models Enable Understanding,
  Prediction and Planning,” arXiv:2506.09985;
- Yang et al., “Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
  Hyperparameter Transfer,” arXiv:2203.03466;
- Dao, “FlashAttention-2,” arXiv:2307.08691;
- Yang et al., “Qwen3 Technical Report,” arXiv:2505.09388;
- Mur-Labadia et al., “V-JEPA 2.1: Unlocking Dense Features in Video
  Self-Supervised Learning,” arXiv:2603.14482; and
- Kang et al., “How Far Is Video Generation from World Model: A Physical Law
  Perspective,” ICML 2025.

## 197. Isolate every observed recursive impulse-gradient path before scaling

The first fresh typed-output-isolated campaign exposed a distinct failure at
optimizer update 200. The update was finite and fully supported, but the raw
interaction norm reached `857.1579`. Typed impulse-multiplier and impulse-
additive decoder rows contributed `830.3828` and `210.3096`; the configured
node, collision, and force output caps did not cover those two jump outputs.
Shared attention projections and blocks consequently received order-one to
order-six gradients before the later complete interaction cap retained only
`0.001167` of the update. This is optimizer starvation, not useful hard-
example learning, and the campaign must not count toward convergence.

Impulse multiplier/additive outputs are one semantic group and require the
same two-level isolation as recursive force outputs:

- a separately configured per-invocation output-gradient cap runs at the typed
  decoder boundary before the impulse gradient enters decoder weights or the
  shared token stack; and
- a separately configured joint decoder-row cap bounds the accumulated
  multiplier/additive parameter gradient before the complete interaction and
  whole-model caps.

Both levels are training-only and leave forward values, inference behavior,
parameter count, tensor shapes, and zero-output initialization unchanged.
Their configuration belongs to resume/selector protocol semantics, and their
raw norm, applied norm, coefficient, invocation count, and minimum per-
invocation coefficient must remain visible in live and offline diagnostics.
Legacy checkpoints without these fields reproduce their historical uncapped
behavior by normalizing both values to `null`.

The active attention qualification also configures a minimum complete-
interaction retention of `0.1` after all semantic output and decoder-row
caps. Falling below that value means isolation did not contain the outlier:
the trainer must clear gradients and reject the update before Adam state or
weights change. The offline dynamics auditor treats the same condition as a
hard failure, while a severe coefficient confined to a semantic local cap may
remain a warning when the complete interaction stage retains at least `0.1`.
The fail-fast threshold is optional protocol state so historical checkpoints
without it preserve their original behavior as `null`.

The failed step-200 update may be replayed from the preceding durable step-128
model/optimizer/RNG/sampler state only as a clearly non-promotable diagnostic.
A fresh qualification must still initialize weights-only from the protected
graph control, pass the matched step-200 stress batch, complete fixed RGB-only
selectors, and reach the declared plateau before width, depth, history, or
video-representation capacity is increased.

## 198. Bound accumulated node-decoder gradients after recursive output isolation

Per-invocation typed-output isolation does not bound the sum accumulated by a
decoder parameter that is reused throughout a recursive rollout. The first
fresh specification-1.30 campaign deterministically reached a fully supported
update 60 whose complete interaction stage retained only `0.0850405`; the
trainer correctly rejected it before Adam mutation. Exact replay matched every
comparable logged model/data field through update 56 and captured the rejected
batch's complete hierarchy.

The raw interaction gradient was `28.2744`. Existing collision, force, and
impulse row caps reduced their groups as configured, but the remaining
interaction norm was `11.7591`. The accumulated node decoder alone was
`11.6617`, dominated by its world-y row at `11.5014`; the largest shared
non-decoder attention parameter was only `0.124876`. This is a missing semantic
row cap, not evidence that shared attention, MPS, support, or model capacity
collapsed. A one-update norm reconstruction shows a joint node cap of `1.0`
would leave a `1.81140` interaction norm and `0.552059` complete-stage
retention, comfortably above the required `0.1`.

Training therefore supports a separately configured joint accumulated node
x/y/z decoder-gradient cap. It runs after per-invocation node-output hooks and
before collision, force, impulse, complete-interaction, and global caps. It
must report raw/applied node norm and coefficient plus the interaction norm
after node isolation, participate in exact resume/selector protocol hashing,
normalize to `null` for historical checkpoints, and remain visible to the
offline severe-clipping auditor. The repair changes backward conditioning
only: forward values, inference, tensor shapes, parameter count, typed outputs,
and `WorldBelief` contracts remain unchanged.

A durable numerical failure artifact is part of convergence evidence. The
trainer must persist the rejected optimizer step, batch seeds/scenarios,
support and physical diagnostics, full gradient hierarchy, configured minimum,
and an explicit zero applied-update marker. The offline auditor must fail a run
with such a terminal optimizer artifact even when its last sampled JSONL row
was healthy. Scaling remains prohibited until a fresh repaired small-rung run
passes fixed selectors and the declared plateau.

## 199. Architecture-growth handoffs must reject partial learned modules

An allowed missing-key prefix ordinarily means that the source checkpoint
contains none of a newly introduced module and the destination creates the
complete module at its declared neutral initialization. It must never mean
that an existing learned module may be partially copied while missing layers
are randomly initialized. In particular, loading a trained four-block
attention residual into a six-block destination with ordinary random new
blocks changes the hidden representation seen by the learned typed decoders
before any optimizer update; it is not a zero-output or function-preserving
growth operation. The exact appended-block transform in section 200 is the
only supported exception.

Weight-only loading must therefore preflight source/destination keys and tensor
shapes before copying any value. It rejects unexpected keys, disallowed
missing keys, incompatible shapes, and any allowed prefix that is present in
the source but only partially covers the destination. A rejected handoff leaves
the destination model bitwise unchanged.

Any attention growth that does not satisfy the explicit identity-initialized
depth transform in section 200 starts weights-only from the qualified
structured graph control, where the complete attention prefix is absent and
the typed decoders start at exact zero. The accepted smaller attention
checkpoint remains the fixed non-regression reference. This preserves a clean
capacity comparison and prevents random new blocks from masquerading as
training or generalization evidence.

## 200. Appended attention depth may inherit through exact identity blocks

The supported depth-growth transform is deliberately narrower than generic
partial state-dict loading. The source and destination must have contiguous
zero-based attention block indices, every inherited tensor must have the same
name and shape, and the only missing tensors may belong to blocks appended
after the complete source stack. Width changes, holes, reordered blocks,
missing tensors in an inherited block, and missing attention projections or
typed decoders remain hard preflight failures that leave the destination
unchanged.

The destination's resolved configuration is mandatory for this transform.
Checkpoint and destination model/runtime/simulator semantics must be identical
under the ordinary checkpoint compatibility contract after substituting only
the larger `attention_layers` value. This catches shape-invisible changes such
as attention head count, dropout, bounded-output scales, filter behavior, or
world geometry. Training/evaluation budgets and data volume may change because
the handoff starts a new optimizer/RNG schedule rather than an exact resume.

Each appended pre-normalized residual block is initialized as the exact
identity by zeroing its multi-head-attention output weight and bias and its
SwiGLU output weight. Internal query/key/value and feed-forward input weights
retain their ordinary finite initialization. The zero output projections make
both residual branches emit exact zero at handoff, while receiving a usable
gradient on the first optimizer update; deeper internal paths become live as
those projections move away from zero. The loader must record the appended
block indices in initialization provenance and tests must prove zero-tolerance
equality of the shallow and grown token streams and decoded outputs.

This transform changes the depth rung's initialization policy, not its
promotion gate. A qualified smaller attention checkpoint may now initialize a
depth-only candidate without changing its predictions at step zero. The
smaller checkpoint remains the fixed reference, the candidate starts a new
optimizer/RNG/data schedule, and it must still pass repeated fixed RGB-only
validation, test, OOD, scenario, uncertainty, identity, event, and horizon
non-regression gates. Width growth remains unsupported and therefore starts
from the neutral structured graph control until a separately proved
function-preserving transform exists.

## 201. Training trends must pool physical sufficient statistics

Sparse training-cadence records are health and diagnostic samples, not fixed-
manifest validation. The dynamics auditor must nevertheless expose consecutive
non-overlapping trend windows so support collapse, identity/lifecycle drift,
uncertainty failure, axis imbalance, horizon trade-offs, event failure,
parameter observability loss, gradient starvation, and resource growth are
visible before a selector boundary. Every window records its first/last step,
logged-block count, scenario exposure, and whether it is complete. An
incomplete tail window must never be compared as if it had the declared
support of a complete window.

Physical errors must be pooled from persisted sums of squared errors and
coordinate counts before taking a square root. Coverage and precision must be
pooled from their count numerators/denominators; collision F1 must be derived
from pooled true-positive, false-positive, and false-negative counts; identity
must pool switches and associations. Averaging already-derived per-batch RMSE,
coverage, precision, F1, or switch-rate values is forbidden because unequal
support would bias the trend. The same report includes current and every
configured forecast horizon for position and velocity, current position axes,
coverage, uncertainty NLL distribution, corrections, lifecycle support, slow-
parameter observability, causal support, minimum complete-interaction gradient
retention, and memory.

Trend windows can diagnose collapse or motivate a matched investigation, but
they cannot promote weights, declare convergence, authorize scaling, or
override fixed RGB-only selector/test/OOD guardrails. Heterogeneous training
samples may differ materially even under scenario-balanced batches. Capacity
and optimizer decisions therefore remain bound to repeated fixed selectors and
the declared plateau protocol.

When a prior trajectory is used as a deterministic diagnostic reference, the
auditor must offer an explicit matched-reference mode rather than relying on
ad hoc visual or scalar comparisons. Candidate cadence rows align by optimizer
step and must exactly match episode seeds, scenario order, data-draw index,
frame-window bounds, and rollout-anchor selection. A missing reference step or
schedule mismatch is a failed comparison, not permission to compare different
samples. Candidate and reference metrics are independently pooled from their
physical sufficient statistics over the aligned rows before reporting signed
candidate-minus-reference deltas for state, axes, every position/velocity
horizon, lifecycle, identity, uncertainty, collisions, observability, support,
gradient health, and resources. Such matched training evidence is stronger
diagnosis, but it remains subordinate to fixed selector/test/OOD promotion.

## 202. Recursive typed-output caps are aggregate per-draw budgets

Per-invocation output clipping and accumulated decoder-row clipping do not by
themselves bound the sum that repeated recursive invocations send through the
shared attention projections and blocks. The specification-1.31 small-rung
campaign proved this at attempted optimizer step 988. All 13 causal objectives
were supported and every tensor remained finite, but 144 attention invocations
produced aggregate force and impulse output-gradient norms of `0.219855` and
`0.227758` around nominal `0.1` caps. Token activations amplified the normal-
force decoder-weight gradient to `10.9076`; shared block gradients reached
`5.01609`; and the complete interaction stage retained only `0.0971759`, below
the declared `0.1` minimum. The trainer correctly rejected the update before
Adam and the supervisor correctly stopped the campaign before selector 1024.

Configured node, collision, force, and impulse output-gradient caps therefore
mean aggregate L2 budgets over one optimizer draw. If a semantic group has
`K` registered recursive invocations, each invocation receives a local limit
of `cap / sqrt(K)`. The sum of squared applied invocation norms is then bounded
by `cap^2` while a single invocation retains the historical behavior exactly.
Registration counts reset once per optimizer draw and must be complete before
backward. Diagnostics continue to report raw/applied aggregate norms,
invocation count, and minimum/effective coefficients; the cap value remains
resume/selector protocol state.

This repair changes backward conditioning only. Forward dynamics, inference,
parameter count, checkpoint tensor shapes, zero-output initialization,
`WorldBelief`, and every modality/runtime contract remain unchanged. The failed
trajectory is non-promotable. A repaired replay must begin from the preceding
durable step-896 optimizer/RNG/sampler state, contain the same step-988 draw,
and remain diagnostic; a clean small-rung campaign still requires repeated
fixed selectors and plateau before any capacity increase.

## 203. Protected-checkpoint audits must be non-vacuous

An integrity audit cannot prove that an incumbent or reference checkpoint was
preserved when no protected path was supplied. The empty universal statement
"all supplied protected checkpoints equal the initializer" is mathematically
true but operationally misleading and must not be rendered as a successful
protection result.

Attention checkpoint reports therefore record the protected-checkpoint count.
When the set is empty, `protected_checkpoints_exactly_initial` is `null`, not
`true`. Qualification commands must opt into a required-protection gate and
fail when no protected checkpoints were supplied. When paths are present, the
auditor retains their whole-file hashes and model-state hashes and fails if any
protected model state differs from the declared initializer. Model finiteness,
architecture agreement, inherited-tensor equality, optimizer ownership, and
Adam-step checks remain independent requirements.

This is an offline evidence-contract change only. It does not alter model
weights, optimizer behavior, runtime inference, selection metrics, or the
protocol of an already-running immutable campaign. A historical report created
without explicit protected paths is unchecked on that dimension and must be
rerun; it is not evidence of corruption by itself.

## 204. Functional priors require objective-gradient alignment evidence

A functional residual may grow despite an explicit restoring prior because
the physical task objective sometimes rewards that residual on the sampled
causal draw. Scalar prior loss, parameter energy, total gradient norm, and
emitted acceleration cannot distinguish an inverted implementation from an
underweighted prior, a genuinely conflicting task gradient, or optimizer
momentum across alternating draws.

The deterministic residual-calibration utility must therefore be able to
separate the unregularized task objective from every configured node prior and
measure gradients of the task, the unit drift prior, and the actual configured
total objective on the same graph. It reports norms and task-versus-drift and
total-versus-drift cosine alignment over both the complete attention module and
the typed node decoder. Missing parameter gradients contribute no fabricated
value; a zero gradient yields an undefined cosine rather than an arbitrary
zero. A positive cosine means gradient descent locally reduces the task and
drift objectives together, while a negative cosine exposes direct conflict.

Alignment is a read-only diagnostic, not a training-time gradient surgery
rule. It must use deterministic balanced RGB-only causal draws and preserve the
exact differentiable-call population used by the functional prior. One draw
cannot justify a weight or schedule change: inspect multiple draws and retain
fixed-selector behavior as authoritative. If task/prior alignment alternates
while the configured total objective still increases drift on some draws, a
lower or decaying learning rate is a cleaner same-capacity experiment than
silently increasing the prior, hard-projecting gradients, or scaling model
capacity. Such a successor remains a new weights-only protocol and may start
only after the immutable fixed selector rejects the current candidate.

## 205. Qualify relation/event residuals before evidence-gated node acceleration

Warmup and cosine decay do not repair an unconditional typed node residual that
learns a broadly misgeneralizing acceleration shortcut. The specification-1.41
width-128/four-block schedule control is structurally healthy at update 512,
but its fixed RGB-only selection score worsens from `0.3213162` to `0.3475480`
with 116 broad incumbent guardrail failures and one failed improvement rule.
The familiar `reference_pairs` current x error rises from `0.242694` to
`0.720231 m`, and every x horizon regresses. This is worse overall than the
already rejected constant-rate control and closes the schedule-only repair.
It does not authorize depth, width, history, or compute growth.

The next small-rung experiment separates typed interaction learning from
single-object acceleration. An `attention_relation` training scope keeps the
node decoder bitwise equal to the protected zero-output initializer while
training the scene/entity/relation projections, type embeddings, dense
attention/SwiGLU blocks, output normalization, and relation decoder. This is a
qualification stage, not a permanent claim that unmodelled node forces cannot
exist. It tests the already observed ablation signal that relation/event
residuals can improve pooled prediction while the unconditional node output is
the dominant source of free-flight regression.

The scope must initialize weights-only from the untouched graph control, never
from rejected attention weights. Its checkpoint audit must prove all 46
permitted attention tensors and exactly their 46 Adam states changed, both node
decoder tensors remained bitwise exact and own no optimizer state, all 177
inherited tensors remained exact, protected checkpoints remained exact, and
all serialized state is finite. Fixed validation, test, OOD, scenario, axis,
horizon, lifecycle, identity, event, and uncertainty rules remain unchanged.

Only after relation-first training qualifies may node acceleration return
behind an explicit observation-derived evidence gate. The gate must default to
zero residual at initialization, remain axis-neutral, consume causal belief,
innovation, uncertainty, contact/event, and timestamped context rather than
oracle labels, and pay a sparsity/calibration cost. It may open for genuinely
unmodelled forces, but cannot become a constant scene-wide correction merely
because the rest of the model is frozen. Qualify that gate as a separate
same-capacity experiment before reopening the capacity ladder.

## 206. Fixed-boundary audits must bind payload and optimizer steps

A checkpoint filename, external report name, or training heartbeat is not
authoritative evidence of the optimizer boundary contained in a serialized
artifact. Fixed validation may begin after an update heartbeat and publish its
checkpoint only after a long evaluation. During that interval, `last.pt` can
truthfully remain the preceding durable artifact. Copying it under the pending
selector's name must not create a passing but mislabelled audit.

Every fixed-boundary qualification audit must therefore receive the expected
optimizer step explicitly and fail unless it equals the checkpoint payload's
embedded step. It must also parse every serialized per-parameter optimizer
step and fail unless the non-empty unique set is exactly the embedded
checkpoint step. Mixed Adam steps, stale Adam state under a newer payload
step, and a stale payload under a newer requested boundary are distinct hard
failures. Reports record the expected step, embedded step, unique optimizer
steps, and their agreement.

The expected-step requirement is an operational qualification rule, not a
filename inference rule. General historical/ad-hoc audits may omit it to
preserve reproducibility, but they still fail an internal payload-versus-Adam
mismatch. A step-zero checkpoint may legitimately have no optimizer owners.
This contract changes audit evidence only; it does not alter training,
checkpoint serialization, selector ordering, inference, or model weights.

## 207. `after_step` audit boundaries are strictly exclusive

An audit requested after optimizer step `N` must select only records with
`step > N`. Including step `N` overlaps adjacent windows, double-counts their
shared boundary, and can bias pooled physical sufficient statistics toward a
single draw. The same exclusive predicate applies to candidate training rows,
validation rows, and any matched-reference rows. A matched comparison must
still fail if the reference lacks any selected candidate step.

The report retains `after_step=N`, and its first selected step is the first
actually persisted cadence or validation record greater than `N`; sparse
cadence need not produce `N+1`. Tests must cover candidate, reference, and
validation exclusion at the exact boundary. This is a read-only evidence
correction and does not reinterpret already serialized metrics or alter the
active trainer.

---

# Part XXXVII — Evidence-bounded heterogeneous mental-simulation amendment

## 208. The perceptual mental image is the persistent world belief

The two original Orpheus papers sharpen rather than replace the architecture
above. Their *mental image* or internal model is an entity-centred imaginary
world constructed from perception, including physical properties that may not
be directly observable. In this repository that role belongs exclusively to
`WorldBelief`: timestamped RGB measurements discover and correct entities,
while inferred state, uncertainty, identity, lifecycle, parameters, relations,
and events persist between observations. Simulator state remains supervision
and evaluation truth only.

Domain-expert models and learned models may coexist behind the dynamics
contract. Analytic kinematics, contact solvers, stable modes, learned residuals,
and behavioral/event operators are complementary candidate effects, not rival
sources of world state. Normal online improvement updates the belief and the
small applicability/evidence state associated with these models; it does not
require a large network-weight update or re-encoding observation history.

Primary sources for this interpretation are:

- Mihai Polceanu, Marc Parenthoën, and Cedric Buche, “ORPHEUS: Mental
  Simulation as Support for Decision-Making in a Virtual Agent,” AAAI 2015,
  https://cdn.aaai.org/ocs/10371/10371-46146-1-PB.pdf; and
- Mihai Polceanu and Cedric Buche, “Towards A Theory-Of-Mind-Inspired Generic
  Decision-Making Framework,” arXiv:1405.5048,
  https://arxiv.org/abs/1405.5048.

## 209. Local model applicability is evidence-bounded

A heterogeneous candidate pool must represent explicit applicability, not one
scene-wide winner. Evidence and assignment are local to at least:

- persistent entity identity;
- state component or axis;
- interaction/event regime;
- prediction horizon or composed short-step interval; and
- the belief/dynamics revision from which the prediction was made.

Candidate confidence is updated only by causal comparison between a prediction
made before an observation and the later associated measurement. A model
choice must never be transferred beyond the entity, component, regime, or
horizon for which that comparison supplies evidence. In particular, a choice
supported at 0.05 seconds cannot govern a 0.10--1.00-second query merely
because it belongs to the same axis. Unmatched horizons use the explicit
no-evidence fallback until they receive their own evidence or are reached by a
qualified composition of supported short steps.

The default no-evidence fallback is the accepted learned/structured runtime
incumbent. Reports must distinguish fallback caused by no applicable evidence
from a learned candidate selected by positive evidence. Applicability state
must expose support count, age/freshness, observability, regime identity,
uncertainty, and selection confidence. Missing or stale evidence cannot be
encoded as a zero error or an arbitrarily confident prior.

## 210. Physical and behavioral effects compose in stable short steps

Mental simulation advances through ordered bounded substeps. Within each
substep, analytic physical effects, structured interactions, learned
residuals, behavior/action effects, and discrete event jumps are interleaved
and accumulated under the same causal state-transition contract. An effect
that is selected for one local interval modifies that interval; a longer
forecast is produced by repeatedly evolving the resulting coherent state,
re-evaluating applicability at supported boundaries, and propagating
uncertainty. It is not produced by substituting an independently computed
long-horizon coordinate into an otherwise unrelated learned trajectory.

Axis-local evidence is useful, but emitted state must remain mathematically
coherent. When a candidate controls an axis it must supply, or trigger a
documented consistent recomputation of, the associated position mean,
velocity mean, and predictive variance. Event state and cross-axis coupling
remain joint. A position splice that retains incompatible velocity or
variance is invalid even when its position RMSE improves.

The runtime may maintain several simultaneous action/event/model branches and
roll them forward from cloned beliefs. Branches are possible futures, never
independent persistent truth. They are weighted by later measurement
likelihood, may be pruned or merged, and must leave the authoritative source
belief unmodified until ordinary association, innovation, and correction
assimilate real evidence.

## 211. Evidence combines both uncertainties and is invalidated causally

For a continuous associated measurement, model evidence must combine the
candidate's predictive uncertainty with measurement uncertainty. A Gaussian
candidate score uses the innovation under their sum, for example:

\[
S_k = \frac12\sum_d m_d
\left[
\frac{(y_d-\mu_{k,d})^2}{P_{k,d}+R_d}
+\log(P_{k,d}+R_d)
\right],
\]

with finite clamps and explicit masks. Candidate variance alone, measurement
variance alone, or unnormalised squared error is not the complete likelihood.
Nearby/range simulations may estimate expected score and fragility, but their
dispersion augments rather than replaces calibrated predictive uncertainty.

Pending evidence is valid only for the exact source belief identity and
revision, timestamp, persistent object IDs, candidate/dynamics revision,
runtime train/eval mode, and configured applicability cell that created it.
Any external belief replacement, reset, lifecycle slot reuse, incompatible
correction, parameter/buffer mutation, or mode change invalidates the pending
item. A late or invalidated target is discarded and reported; it must not
train confidence for a different state.

Continuous model improvement means updating this bounded evidence and
applicability state after every valid prediction-versus-reality comparison,
with robust influence and explicit forgetting where configured. It does not
authorize online full-model backpropagation. Approximate outcomes within a
calibrated useful range can outrank brittle point accuracy, but every reported
range must be tested against realised coverage and likelihood.

## 212. Runtime-pool promotion includes accuracy and cost

A new selection/composition policy is a runtime semantic change. Promotion
requires a paired learned-only control using the same checkpoint bytes, device,
precision, RGB-only observation path, seed/scenario manifest, anchors,
horizons, and metric implementation. Compare current state and x/y/z position
and velocity at every declared horizon, lifecycle/coverage, identity,
collisions/events, uncertainty likelihood and coverage, nonfinite state, and
latency for global updates, fast updates, and future rollouts.

The 15 August four-episode MPS diagnostic does not pass this gate. Against its
matched learned-only control, the runtime pool's x RMSE at 1.00 seconds is
`0.895082` versus `0.771005` (`+16.1%`), aggregate position RMSE is `0.672120`
versus `0.618738` (`+8.63%`), forecast Gaussian NLL is `0.850832` versus
`0.834268` (`+1.985%`), and global/fast update latency is `2.216x`/`2.392x`.
Lifecycle, identity, events, y, and z are unchanged, but those equal slices do
not offset the accuracy, calibration, and cost regressions. The policy is
therefore rejected and remains opt-in. This result covers four validation
episodes and is diagnostic rather than a broad convergence claim.

The immediate repair must eliminate unsupported horizon transfer, add the
explicit evidence/applicability semantics above, and remove duplicated
candidate/runtime propagation before another paired MPS comparison. No repair
is considered complete until focused tests and the matched runtime protocol
verify it.

---

# Part XXXVIII — Grounded RGB convergence and protocol-integrity amendment

## 213. Familiar validation physics must isolate the interaction under test

The familiar sphere-world protocol must not make a requested pair collision
coincide with a floor, wall, third-body, birth, or stochastic-impulse event
unless the scenario is explicitly labelled as compound. An ensured pair is
constructed and preflighted with the same complete scene, solver, observation
clock, and deterministic random streams used by the final episode. Its two
objects must remain free of every other contact through a declared clean
window of at least two observation frames after pair impact. Extra objects,
lifecycle births, and external impulses are resampled or delayed around that
window. Compound interactions remain useful, but must be separately named and
reported rather than silently contaminating a basic collision benchmark.

Free flight in simulator and analytic dynamics uses one closed-form
constant-gravity/linear-drag transition. This removes a systematic integrator
mismatch that formerly forced the learned residual to cancel the data
generator's timestep error. Contact resolution still runs on bounded physical
substeps; a faster contact rate is a new simulator protocol and needs its own
penetration, impulse, and endpoint qualification. The resulting simulator
protocol is `sphere_world_v6`. Old v4/v5 metric reports remain historical
diagnostics and cannot be compared as if generated by v6.

## 214. RGB temporal evidence distinguishes observation from copied prior

A position is eligible for temporal differentiation only to the extent that
its coordinates are independently supported by the RGB observation. Global
image discovery is observation-derived. A residual ROI is prior-conditioned:
only a direct structured image centre supplies camera-lateral coordinates and
only a valid structured scale supplies depth. A zero residual, disabled head,
or coordinate copied from a predicted ROI remains a useful ordinary filter
measurement but is not an independent temporal sample. Histories persist a
per-object, per-sample, per-world-axis support mask and fit each supported axis
without allowing an unsupported coordinate to complete or influence it.

Independent raw history is an explicit semantic opt-in. Historical
checkpoints retain their configured posterior/measurement blend; a missing
new flag resolves to that legacy behavior. New grounded protocols enable raw
history, an unbounded causal sample age unless a tested cutoff is supplied,
and a cadence that actually produces the configured minimum number of samples.
A combined-camera execution test must prove that the fast path emits nonzero
supported velocity evidence before launch. Structured fast depth was present
in the initial 1.46 candidate but is disabled by section 221 because component
completeness is not observable; only independently qualified scale may restore
that support.

When gravity in `WorldBelief` is known, a causal observer may subtract its
quadratic displacement before least-squares fitting and estimate velocity at
the current observation time. It combines that gravity-aligned estimate with
the ordinary orthogonal component, preserves strict raw-axis dependencies,
and propagates the shared measurement uncertainty rather than pretending the
two fits are independent. It consumes no simulator state and is opt-in; zero
gravity and legacy-disabled behavior remain exact fallbacks.

## 215. Structured RGB discovery is direct evidence, not permission for ghosts

For the synthetic RGB contract, connected foreground components supply direct
centres, observed radii, analytic inverse depth, and uncertainty. Ambiguous or
truncated scale remains usable only with explicitly inflated covariance.
Packet confidence multiplies structured confidence; a low-confidence packet
cannot be promoted to a fixed near-one birth probability. This evidence is
carried into the core existence logit with a straight-through learned
gradient.

Queries without a component or an explicitly missing component assignment
remain trainable but fail closed for runtime birth. A touching component that
was successfully split and assigned is not permission for unrelated learned
queries to create duplicate objects. A true no-component image and a bounded
partial-assignment gap remain explicit learned-discovery escape hatches for
future real video. This preserves a modern learned detector path without
allowing fresh random logits to dominate the currently reliable abstraction.

## 216. Primary evaluation is intervention-free and recovery is independent

Headline state, horizon, lifecycle, identity, event, baseline, uncertainty,
and per-scenario accuracy metrics come from one clean RGB-only online pass.
The evaluator must never perturb that authoritative runtime. An optional
recovery probe constructs an independent runtime from the exact same immutable
checkpoint payload, replays the causal RGB prefix, applies the declared belief
perturbation there, measures correction, and discards the branch. Changing or
disabling the probe must leave every non-latency primary physical metric and
the primary posterior trace bit-for-bit unchanged.

Evaluation binds the exact checkpoint bytes loaded, checkpoint/evaluation
simulator and specification versions, source provenance, resolved config,
split, seed manifest, scenario ordering, horizons, batching, metric schema,
and runtime intervention to canonical hashes. Mutable `last.pt` replacement
must not let a primary pass, recovery probe, step, and reported SHA refer to
different weights. Nonfinite beliefs, trajectories, event logits, or final
numeric metrics fail before a completed report; JSON output forbids NaN/Inf.
Per-scenario metrics are additive views of the same clean tensors and masks,
not second rollouts. Any incomplete slice is labelled diagnostic-only and
cannot satisfy a promotion guardrail by itself.

## 217. Learned interactions are local residual effects with explicit semantics

Constant/damped analytic motion remains the no-evidence prior. Learned pair
force, event, noise, and impulse residuals may be multiplied by a smooth
parameter-free applicability function of signed gap, closing motion,
uncertainty, and the current bounded-step geometry. The gate is recomputed on
every analytic microstep, including while a learned proposal is held, and may
not gate the analytic contact resolver. It defaults to exact identity-off and
is a checkpoint semantic.

Relation attention binds an unordered pair token to the symmetric mean of its
two endpoint entity tokens only behind an explicit semantic flag. Historical
attention checkpoints resolve the flag false and keep their old forward
function. A new grounded campaign may enable it, with permutation, inactive-
endpoint, checkpoint, and zero-decoder identity tests. A configurable
multi-rate cadence may hold a complete graph/attention proposal within one
predict call, but topology changes and collisions invalidate it. Analytic,
modal, contact, event, and uncertainty steps continue at every microstep. The
historical exact cadence remains the default until a paired accuracy/latency
gate passes.

## 218. Slow physical parameters use causal, observable, uncertain evidence

Drag/restitution identification consumes the causal prior error, never an
already-corrected posterior error. Position displacement is divided by actual
elapsed time, combined predictive-plus-measurement variance controls its
reliability, and direct confidence scales the signal. A source-bound ROI can
contribute an analytic position signal only on independently observed axes;
missing provenance fails closed. Position displacement may support drag but
cannot fabricate a post-impact restitution measurement. Restitution requires
direct supported pre/post velocity or the labelled debug oracle. Fast state
and slow parameters retain separate update gates and memories.

## 219. Convergence campaigns stage perception before interaction capacity

The next accepted campaign starts from the protected finite weights through a
weights-only, protocol-versioned transfer. The `state_roi` stage trains the
filter/updater, identifier, fast ROI projection, and early visual features
while leaving dynamics and global discovery frozen. Only after a declared
number of causal optimizer updates may `state_relation_roi` add relation/shared
attention and graph edge parameters while keeping unconditional node
acceleration, analytic dynamics, modal/event/uncertainty modules, and the
global detector frozen. This tests perceptual anchoring before interaction
capacity and prevents a generic node residual from becoming a free-flight
shortcut.

Training must be long enough for fixed validation to repeat across the
learning-rate schedule: balanced updates draw every declared scenario,
checkpoint frequently, validate on immutable disjoint manifests, and run
through warmup, useful-rate, and decay phases. A noisy mixed-scenario batch
loss is not convergence evidence. Promotion requires sustained improvement or
a truthful plateau on current state and every axis/horizon, plus coverage,
identity, event, calibration, finite-state, and latency guardrails. A local
slice win that regresses another critical scenario is retained as an ablation,
not promoted.

## 220. Remaining event-learning limits stay explicit

The current analytic contact path is intentionally strong and the new
familiar simulator makes its correctness interpretable. Learned event
calibration is not yet complete: hard analytic event logits, the unused
contact-logit scale, and analytic-only pair event logits limit how trajectory
loss can move a missed event boundary. This is a separately qualified target,
not a reason to hide perception, lifecycle, or simulator defects behind more
capacity. No documentation or report may claim full event convergence until
pair-specific differentiable event calibration passes the normal fixed RGB
promotion protocol.

---

# Part XXXIX — Causal RGB and event-objective convergence amendment

## 221. A component centre does not prove complete scale

A structured RGB centre remains useful direct lateral evidence, but the radius
of a prior-conditioned residual-ROI component is not automatically independent
depth evidence. Occlusion, overlap, truncation, and component merging make the
fraction of the physical disc represented by that component unobservable from
the component alone. A syntactically valid radius therefore cannot certify
component completeness.

The grounded campaign keeps structured fast centres but disables structured
fast depth. Re-enabling it requires an explicit completeness or visibility
model and matched multi-scenario evidence. In the qualifying diagnostic, 28
accepted seed-100000 fast-depth samples had a measured/true projected-radius
ratio with mean `1.1587` and range `1.026--1.2124`; across seeds
`100000--100007`, disabling fast depth reduced pooled current position RMSE
from `0.27719` to `0.13479 m`, with x/y/z
`0.14815/0.15601/0.42920` to `0.12922/0.13017/0.14443 m`, and raised
distance-gated precision/recall from `0.70265/0.68704` to
`0.96628/0.92870`. This rejects the fast-depth setting, not the direct-centre
abstraction or the general RGB measurement contract.

## 222. Temporal evidence uncertainty must cover observed error

Temporal velocity fitting remains causal, raw-RGB-only, per-axis, and
gravity-aware, but its covariance must describe empirical residual error rather
than saturate at a convenient small ceiling. The eight-seed diagnostic measured
gravity-aware direct-evidence MSE x/y/z of `0.18443/3.82435/0.14449` against
reported variance `0.22444/0.23707/0.25000`; the y evidence was about `16.13x`
overconfident. The grounded protocol therefore raises only the temporal
velocity variance ceiling from `0.25` to `4.0`.

With identical weights, simulator, seeds `100000--100007`, independent raw
history, gravity fitting, and fast depth disabled, that change improved current
position x/y/z/all RMSE from
`0.129215/0.130169/0.144431/0.134785` to
`0.123896/0.121938/0.143362/0.130092 m`; distance-gated velocity x/y/z/all
RMSE from `0.456640/1.241236/0.223000/0.774363` to
`0.454941/1.216228/0.214243/0.759842 m/s`; and precision/recall/F1 from
`0.966281/0.928704/0.947120` to `0.973025/0.935185/0.953730`. A proposed
contact-free change-point reset is rejected because it made noisy early resets
and regressed the calibrated setting.

## 223. Differentiable hazards complement the hard event resolver

Smooth pair and boundary contact/collision hazards are an explicit
legacy-false checkpoint semantic; the grounded candidate explicitly opts in so
the event path can be trained and evaluated. They use signed gap, incoming
normal motion, uncertainty, and learned relation residuals, while the analytic
contact resolver remains the fail-safe owner of physical jumps. A resolved hard
event may impose a positive forward logit floor through a straight-through
expression; it may not remove the hazard gradient. Pair collision supervision
is gathered in belief order from unique matched object pairs and combined with
node ownership.

Dense pair geometry must remain differentiable before diagonal masks are
applied. Projected directional variance is clamped to a dtype-aware positive
floor before square root so a zero self-pair cannot create the critical
`0 * sqrt'(0)` NaN. Recursive CPU and active-Aqua MPS gradient tests are
required. Event losses emitted per forecast horizon use the fixed configured
horizon-weight denominator, including when early anchors are causally
ineligible; an eligible late anchor may not silently inherit full unit weight.

This implementation makes event learning technically possible. It is not an
empirical accuracy promotion: no smooth-hazard checkpoint is accepted until
the full fixed RGB validation protocol passes.

The protected-weight, same-seed step-zero preflight may establish that the new
semantic does not damage the inherited physical baseline, but it is not a
substitute for training. Over validation seeds `100000--100007`, enabling the
smooth hazard changed current position/velocity RMSE by only
`-1.95e-7 m`/`+1.94e-8 m/s`; every 0.10--1.00-second position-RMSE delta had
magnitude at most `1.63e-7 m`, and collision F1 and target coverage were exact.
Mode logits changed substantially as intended, so this is a non-regressive
semantic preflight rather than a claim of bit-identical execution.

## 224. Objectives require causal support and a trainable owner

An RGB discovery birth initializes velocity by a hard runtime rule and has no
incoming trainable prior. Current velocity supervision is therefore supported
only on matched, active belief slots with `age_steps > 0`, expanded explicitly
over the three axes. Current correction supervision additionally requires both
the prior and posterior slot to be active; future correction uses the same
causal age/support contract. Unsupported velocity or correction terms are
structurally omitted rather than divided into a loss as zeros. Public physical
error metrics continue to score newborn estimates and may not be cosmetically
filtered by the optimization support mask.

Event supervision is likewise stage-owned. In the grounded curriculum its
effective weight is `0.0` in `state_roi`, where no event/relation owner is
trainable, and `0.05` in `state_relation_roi`, where relation event decoders
become trainable. A zero-effective-weight term is omitted from the objective
graph, not merely multiplied by zero after building an unrelated gradient
path. Historical configurations without per-scope weights retain their legacy
event weight.

## 225. Plausible gates remain rejected until they improve the whole protocol

The smooth pair-applicability multiplier remains a tested capability but is
disabled in the grounded campaign. On the matched seed-100000 diagnostic it
regressed current position/velocity by `0.56%/0.38%`, regressed position at
`0.25/0.50/0.75/1.00 s` by `0.75%/0.88%/1.02%/0.77%`, and produced no
collision-F1 gain. Local physical plausibility is insufficient evidence for a
runtime or training default; the analytic prior and relation model remain
ungated unless a broad paired protocol passes.

## 226. The long campaign has repeated formal convergence observations

The grounded campaign remains 9,216 balanced optimizer updates with the
declared 512-update warmup, 8,192-update cosine decay, 512-update minimum-rate
tail, and transition from `state_roi` to `state_relation_roi` at update 3,072.
Fixed-manifest evaluation occurs every 512 updates, yielding 18 post-update
validations through update 9,216 in addition to the immutable step-zero
baseline. This cadence is part of the campaign protocol and exact-resume
semantics.

Training loss movement alone is never convergence. Promotion requires a stable
sequence of fixed validations and non-regression across current state, every
x/y/z forecast horizon, velocity, lifecycle, identity, events, uncertainty,
finite-state, latency, and scenario slices, followed by disjoint validation,
test, and OOD evaluation. The repairs in this amendment authorize that
experiment; they do not claim that it has run or converged.

---

# Part XL — Production MPS event-hazard numerical-integrity amendment

## 227. Smooth event conjunction uses an algebraically stable device form

The smooth collision hazard combines two logit-space conditions with the
soft minimum

\[
-\operatorname{logaddexp}(-a,-b)
=
\min(a,b)-\operatorname{softplus}(-|a-b|).
\]

These expressions are algebraically identical over real inputs, but backend
primitive behavior is part of production numerical correctness. On the
user-provided custom PyTorch `2.9.0a0+gitcbe1a35` build in an active Aqua MPS
session, `torch.logaddexp` can overflow for finite inputs whose magnitudes are
only around `90`. A distant valid object pair then turns a finite negative
collision hazard into `-Inf`, which contaminates interval
`pair_event_logits` and correctly fails trajectory validation.

The production event path therefore evaluates the equivalent
`minimum-softplus` form. This is not a clamp, learned gate, CPU fallback, or
change to the event model: it preserves the mathematical hazard, hard analytic
contact resolver, straight-through resolved-event floor, gradients, and all
checkpoint tensors. Extreme-logit forward/backward tests must cover CPU and
active-Aqua MPS, and the complete production validation episode must remain
finite through every RGB frame and rollout anchor before another sustained
campaign is launched.

The first specification-1.47 campaign at
`runs/20260820-213418-grounded-convergence-spec147-mps` is retained as failure
evidence. It stopped before any optimizer update during the first incumbent
validation, at `0/32` completed episodes, with
`trajectory auxiliary pair_event_logits contains NaN or Inf`. After the
algebraic repair, the exact first validation episode (seed `100000`, 40 frames,
8 rollout anchors) completes on MPS with finite loss
`2.279386520385742` and 307 finite metrics in approximately 137.4 seconds.
This one-episode reproduction proves the localized repair only. It is not the
complete 32-episode initialization validation, a fixed-selector result, a
campaign relaunch, or convergence evidence. The frozen specification-1.48
repository gate passes 960 tests with 14 expected non-Aqua MPS-context skips;
lint, format, compile, and diff checks pass. The 9,216-update campaign must
start in a fresh timestamped run from that committed source.

---

# Part XLI — Dynamics synchronization and validation-throughput amendment

## 228. Composite dynamics validates elapsed time once per segment

`DynamicsModel` owns normalization and validation of the elapsed-time tensor
for a complete prediction segment. It accepts only a scalar or one value per
belief row and rejects every nonfinite or negative value before child dynamics
execute. Analytic kinematics, stable modal evolution, and uncertainty
propagation may expose private entry points for this already-normalized
`[batch]` tensor so the composite model does not repeat a tensor-to-host truth
reduction on every physical microstep. Their public APIs retain their complete
independent shape, finiteness, and nonnegativity guards.

Skipping redundant child guards must not weaken numerical integrity. The
composite boundary validates every floating output in the predicted belief,
event logits, and requested auxiliary tensors with one segment-level host
decision. Zero-duration prediction retains the explicit no-event contract and
is subject to the same complete finite-output check.

When every row advances by a positive duration, the composite model may use an
all-positive execution path that omits per-field `where` blending and
auxiliary masking. Mixed positive/zero batches must retain the row mask and
the historical zero-duration behavior exactly. Output and gradient parity,
invalid-time rejection, nonfinite child-output rejection, and active-Aqua MPS
execution are permanent regression contracts for both paths.

## 229. Validation may batch posterior anchors without batching episodes

`training.validation_rollout_anchor_batch_size` controls a validation-only
execution optimization. The repository-wide and historical default is `1`,
which preserves serial anchor rollout exactly. A checkpoint whose resolved
configuration predates the field is interpreted as `1`; changing the value is
an execution-protocol change, changes the rollout-validation protocol hash,
and is forbidden for exact resume. Use a fresh run or weights-only
`--initialize-from` when opting into another value.

The validation loader remains batch one so each seed and scenario keeps exact
attribution. Normal online RGB ingestion runs through the complete episode in
timestamp order and remains the sole owner of persistent runtime belief,
modality caches, temporal histories, lifecycle, and identity. At each selected
anchor, validation may clone the post-ingest `WorldBelief`, association
indices, and match support, then evaluate those posterior forecasts in
anchor-major chunks after the episode has been ingested. It must not
re-encode history, alter the posterior, or expose future observations.

Anchor query plans must be exact prefixes of the longest plan in their chunk.
Shorter rows are padded only by repeating their terminal query time; the
padded suffix is sliced away before the unchanged per-anchor loss, event, and
physical-metric scorer executes. Beliefs may be concatenated only when active
modalities and heterogeneous metadata are equal, including tensor-valued
metadata. Incompatible contiguous metadata groups, including lifecycle flags
carried in metadata, are subdivided, and a singleton group falls back to the
exact serial rollout rather than failing validation or weakening the belief
contract.

Batching is admitted only for `model.eval()` under `torch.no_grad()`, an
episode-loader batch size of one, unperturbed posterior-only validation, and
with the training-only future-correction rollout disabled. The current
implementation rejects multi-rate learned-effect holding because its
batch-global invalidation path has not been proven anchor-independent.
Execution metrics report requested anchors, actually batched anchors, serial
fallback anchors, and posterior rollout calls so metadata fragmentation or a
silent loss of throughput remains observable.

## 230. The grounded eight-anchor profile passed the fixed MPS gate

The generic default remains serial at `1`. The grounded convergence profile is
promoted to `validation_rollout_anchor_batch_size: 8` only because the exact
32-episode validation manifest passed the specification's parity and
throughput gate on the user-provided custom PyTorch
`2.9.0a0+gitcbe1a35` build with MPS active.

Using identical checkpoint bytes, seed/scenario order, RGB-only observations,
eight posterior anchors per episode, horizons, scorer, precision, and frozen
source, serial validation took `3760.393956` seconds and batched validation
took `2012.605486` seconds, a `1.8684208x` speedup. The comparison considered
`3141` values and found zero missing values, zero nonnumeric differences, and
zero numeric differences outside tolerance. The largest finite absolute
difference was `7.62939453125e-06` in additive
`physical_rollout_velocity@0.250s_sse`; the largest finite relative difference
was `6.334555944e-07`. Final runtime-state SHA-256 digests were identical. All
`256` requested anchors were batched in `32` posterior rollout calls with zero
serial fallback.

The durable evidence is:

- `runs/20260820-234059-validation-anchor-qualification-mps-32/serial/qualification.json`;
- `runs/20260820-234059-validation-anchor-qualification-mps-32/batched/qualification.json`;
- `runs/20260820-234059-validation-anchor-qualification-mps-32/comparison.json`.

The earlier serial campaign attempt at
`runs/20260820-221902-grounded-convergence-spec148-mps` was manually
interrupted after `2/32` initialization-validation episodes and is retained as
a throughput diagnostic only. Neither it nor the paired qualification ran an
optimizer update. This amendment makes no training, prediction-accuracy,
checkpoint-promotion, or convergence claim; the long grounded campaign and
its repeated fixed validation remain outstanding.

---

# Part XLII — Objective ownership and measured execution amendment

## 231. Prior-conditioned measurement auxiliaries stop at perception inputs

Section 186 applies to the complete autograd graph, not only parameter
`requires_grad` declarations. A fast-ROI measurement auxiliary may consume a
predicted prior and modality cache as conditioning, but both inputs are
detached for that auxiliary-only forward. The auxiliary prior is also cloned
so its diagnostic execution cannot alias persistent belief storage. This
prevents RGB geometry, existence, colour, likelihood, or world-position
supervision from reaching the updater, identifier, dynamics, or an earlier
runtime cache through the prior-conditioning path.

The ordinary online ingest remains unchanged: it consumes the one prepared
live propagation and the live runtime cache, then association, innovation,
correction, lifecycle, identification, and posterior-rollout objectives retain
their causal gradient paths. Detaching the auxiliary is therefore gradient
ownership, not detaching `WorldBelief`, replacing the runtime loop, or making
the ROI independent of its physical prior.

Objective graph construction follows the same ownership rule. Resolve the
active trainable scope and its effective event weight before requesting pair-
event rollout auxiliaries or building collision BCE. An exact zero effective
weight omits that objective graph entirely while retaining physical event
prediction and detached confusion/count metrics. Multiplying an already-built
unowned graph by zero is not equivalent.

`training.closed_loop_prior_future_correction_enabled` separately controls the
extra prior rollout used only by future correction-improvement supervision.
`true` is the historical and legacy-checkpoint default and remains enabled in
the grounded accuracy campaign: explicit prior-versus-posterior future
improvement is a core correction hinge, not expendable diagnostic work. Setting
the flag to `false` is a matched throughput/ablation protocol that omits that
prior rollout and every future-correction loss term while preserving current
correction, ordinary posterior ingestion, posterior future rollouts, state,
uncertainty, lifecycle, parameter, and event objectives. A disabled-path smoke
qualifies only that ablation's execution and cannot authorize removing the
accuracy objective. The flag is resolved configuration, objective protocol,
validation-protocol evidence, and exact-resume semantics; changing it requires
a new weights-only campaign.

On the current host, a matched exact-cadence `state_roi` one-update diagnostic
measured `data/forward/backward/total` as
`15.421/10.224/6.716/32.360` seconds with the hinge enabled, versus
`15.094/8.728/6.589/30.411` seconds disabled. Retaining the objective costs
about `10.6%` in recursive compute and `6.4%` including data, which is modest
relative to its direct accuracy role. This ephemeral timing is protocol-choice
evidence only; it is not a prediction improvement or convergence result.

## 232. Exact-zero attention is an executable identity, not recursive work

A typed attention residual whose complete node and relation decoders are
finite exact zero has an exact structured identity when no trainable semantic
output owner can consume a gradient. In that state the dynamics path returns
the original typed structured interaction directly and does not tokenize the
belief, execute attention/SwiGLU blocks, manufacture zero tensors, register
typed-output gradient hooks, or accumulate unused functional-node records.
Returning the original typed object preserves every structured value, alias,
auxiliary, and upstream gradient path exactly.

The bypass fails open whenever executing the attention stack could change
learning or forward semantics:

- with autograd enabled, any trainable decoder executes even at exact-zero
  initialization so its first gradient can make the residual learnable;
- a nonzero decoder always executes, including when frozen, because it may
  contribute forward state or transmit gradients to a trainable shared stack
  or input;
- configured training dropout executes so RNG continuation remains exact; and
- eligibility is invalidated by decoder parameter identity, tensor version,
  `requires_grad`, or grad-mode changes.

Under `torch.no_grad()` or inference, an exact-zero decoder may bypass even
when its parameters are declared trainable because no update can be consumed.
Typed-output clipping hooks are registered only when that semantic output has
a trainable attention owner. A frozen attention output may remain
differentiable with respect to the live belief; it must never clip an updater,
perception, or dynamics gradient merely because an old attention cap remains
configured. Functional node-activity bookkeeping is likewise enabled only
when a configured objective or trainable node output owns it.

## 233. Every long optimizer update has an atomic live stage heartbeat

Sparse metrics remain the source for losses and physical trends, but they are
not sufficient to distinguish a long healthy update from a stalled process.
During causal training, `training_progress.json` is atomically overwritten at
the `data`, `forward`, `backward`, and `optimizer` boundaries. It records the
trainer PID, completed and attempted update, target, absolute data-draw index,
retry count, phase/scope, elapsed time, accumulated stage timings, the last
completed update timings, and whether the optimizer update was applied.

This heartbeat is operational state, not checkpoint-selection evidence and
not an additional metrics stream. The read-only monitor may treat it as live
training progress only while the exclusive trainer lock is held and its PID is
compatible with the lock owner. A running heartbeat from another PID is
ignored and surfaced as stale evidence rather than overriding the active run.
Validation and terminal progress retain their existing atomic state contracts.

## 234. Phase devices follow measured end-to-end throughput

MPS availability does not imply that every workload is faster on MPS. Device
selection remains explicit resolved protocol and may differ between RGB
measurement/evaluation and recursive closed-loop optimization as permitted by
section 171. On the current Intel i9 host and user-provided custom PyTorch
`2.9.0a0+gitcbe1a35` build, matched production-window diagnostics found the
recursive forward/backward portion approximately `3.5--3.9x` faster on CPU
than active-Aqua MPS. The grounded profile therefore retains
`device.preference: mps` for supported measurement/evaluation execution and
sets `device.closed_loop_preference: cpu` for causal optimization. This does
not change model weights, tensor contracts, RGB-only semantics, or the
user-provided PyTorch installation.

The same diagnostic found that holding learned relation proposals for `0.05`
seconds reduced late-scope recursive compute by only about `16%` and was not
forward-identical. The grounded campaign therefore keeps
`learned_effect_interval_seconds: null`, preserving exact per-microstep learned
execution until a complete paired accuracy/latency gate justifies a semantic
change. Throughput diagnostics authorize device/cadence protocol choices only;
they do not constitute accuracy promotion or convergence evidence.

---

# Part XLIII — Comprehensive promotion evidence and immutable replay amendment

## 235. Physical selection requires complete current, horizon, axis, event, uncertainty, and identity evidence

The fixed RGB-only selector remains a physical-behaviour gate rather than a
training-loss selector. Its evidence contract advances to rollout-selection
metric version `7` and rollout-validation protocol version `16`. Standalone
held-out evaluation advances to metric schema `held_out_rgb_metrics_v3` and
per-scenario schema `clean_primary_additive_support_diagnostic_v3`.

Every supported pooled slice and every declared scenario slice must now retain
enough exact additive evidence to reconstruct and guard:

- current and every configured-horizon position and velocity error, pooled and
  separately for x, y, and z;
- target coverage and prediction precision, including current and each
  forecast horizon;
- collision true-positive, false-positive, false-negative, true-negative, and
  evaluated counts plus F1 at each horizon, rather than only one pooled event
  number;
- current and per-horizon position coverage, nominal-coverage error, Gaussian
  NLL, and predictive sharpness, pooled and per axis, derived from retained
  likelihood/sharpness sums and calibration coordinate counts; and
- forecast identity eligibility, distance-gated association count and
  coverage, mismatch count, and mismatch rate at every horizon.

Gaussian evidence has one canonical producer. Per-axis negative-log-
likelihood and sharpness sufficient statistics are emitted once and pooled
with `math.fsum` from those exact x/y/z sums and counts. An independently
reduced float32 pooled mean is not authoritative: signed axis NLL terms can
nearly cancel and make a second reduction disagree beyond the narrow
validation tolerance. The real heavy-light regression has axis NLL
`1.0278450847`, `-28.0431194305`, and `25.4184275866`; the prior independent
pooled reduction was `-1.5968445539`, while the canonical axis-sufficient-
statistics result is `-1.5968467593`. Canonicalization fixes the producer;
validator tolerance remains strict.

Forecast identity is evaluated by carrying the persistent target association
available at the causal anchor and comparing it with an independently
distance-gated Hungarian assignment of the predicted positions at each future
target frame. An unassociated future remains visible as missing association
coverage; it is not silently removed from identity eligibility. These metrics
are validation-only evidence and add no optimizer-time model forward or
backward path.

Pooled and scenario evidence are two additive views of the same clean online
pass. Every additive pooled field must have the same schema as every declared
scenario partition. Integer/count fields must sum exactly across scenarios;
floating sums must agree within a narrow declared numerical tolerance. A
missing field, missing scenario, unsupported horizon, absent positive or
negative event class, absent identity association, contradictory derived
metric, or nonfinite value fails selection closed. A favorable pooled score
cannot compensate for a failed scenario, axis, horizon, event, uncertainty, or
identity guardrail.

Core per-episode causal support and rich selection support are deliberately
separate. A core episode must have the complete raw schema, current position
and velocity support, and every configured position/velocity horizon-axis
floor. It need not contain both collision classes or an independently
associated forecast identity in that one episode. Those rare-event,
calibration, and identity requirements remain mandatory after exact pooling
by scenario and manifest. This avoids discarding valid causal episodes while
preserving the fail-closed rich selector. The real fixed-32 replay retains
`32/32` core-supported episodes and `8/8` rich-supported scenario families,
with four episodes in each scenario and complete pooled support.

Version-6 selector artifacts and version-15 rollout protocols are not exact-
resume compatible with this evidence contract. They remain valid historical
diagnostics under their original semantics. A model transferred to the new
contract uses weights-only initialization and fresh selector evidence; an
already running source-frozen version-1.50 campaign may continue under its
original protocol but its artifacts must not be relabelled as version 7/16.

## 236. Physical eligibility and comprehensive promotion are separate claims

An in-training fixed selector can decide whether a finite candidate is a safe
physical incumbent under the complete version-7 guardrails. It cannot measure
a matched wall-clock cost control while holding all execution variables fixed.
Trainer artifacts therefore mark latency support and comprehensive promotion
false and identify their scope as
`fixed_physical_incumbent_not_comprehensive_promotion`.

The external promotion replay reports two independent decisions:

1. `physical_promotion_eligible` requires candidate improvement, candidate and
   reference support, and every incumbent/fixed-reference pooled, scenario,
   axis, horizon, velocity, lifecycle, event, calibration, and identity
   guardrail; and
2. `comprehensive_promotion_eligible` additionally requires complete paired
   latency evidence on the same device and precision, with candidate/reference
   ratios no greater than `1.10` for RGB global update, RGB fast update, and
   future rollout latency.

Each latency component retains its finite mean, additive elapsed-time sum, and
positive integral sample count; the mean must agree with sum/count. Missing,
zero-reference, nonfinite, negative, mismatched, or over-limit timing evidence
fails the latency gate closed. `promotion_eligible` is only a compatibility
alias for comprehensive eligibility, never for physical eligibility alone. A
report may truthfully state `physical_promotion_eligible=true` while
`comprehensive_promotion_eligible=false` when paired latency is absent; it may
not call that result promoted.

Wall-clock measurements are deliberately excluded from the deterministic
primary-physical metric digest. The evaluator publishes the exact hashed key
list, metric scope, and canonical exclusion declaration, and the replay
recomputes the digest rather than trusting a stored hash. Recovery-probe fields
and known recovery summaries are also excluded because the primary pass must
remain intervention-free and bitwise independent of that isolated probe. The
exclusion is explicit evidence, not permission to omit a difficult physical
metric.

## 237. Paired promotion replay binds immutable bytes and the complete execution contract

Promotion replay must be resistant to mutable-checkpoint and mutable-report
time-of-check/time-of-use races. Each candidate and reference checkpoint is
opened once into an immutable byte snapshot; the SHA-256 digest, byte count,
checkpoint deserialization, embedded model-state hash, and replay all refer to
those same captured bytes. Each external evaluation report is likewise read,
hashed, and parsed from one byte capture.

Before a latency report can participate in promotion, replay verifies its
canonical protocol hash and exact binding to:

- the current executable runtime-source fingerprint and complete Git/worktree
  provenance;
- resolved configuration and the version-16 validation protocol;
- simulator version and fully resolved scenario parameters;
- validation split, the exact standard validation seed manifest, episode
  count, seed-to-scenario ordering, and batch counts;
- requested horizons and their observation-grid representation;
- checkpoint SHA-256 and byte count;
- device, precision, RGB-only execution, and finite-output evidence;
- the primary posterior trace and clean-primary physical metric digest; and
- disabled evaluator perturbation, recovery probe, oracle runtime input, and
  runtime hypothesis-pool intervention.

The grounded replay contract is the standard validation range beginning at
seed `100000`, with one online trainer-validation episode per loader batch and
the same ordered scenario mixture used by the selector. Standalone evaluator
batching may differ only as explicitly bound in its own protocol; it may not
change episodes, horizons, posterior semantics, or metric implementation.
Python, NumPy, CPU Torch, and MPS RNG state are reset for each paired arm so
evaluation order cannot leak random state from reference to candidate.

The replay independently recomputes selector metrics from additive evidence,
checks that scenario partitions sum to pooled values, validates every report
identity field, applies physical guardrails, then applies the paired latency
gate. Any missing or contradictory binding is an error or truthful rejection,
not a best-effort promotion. A non-comprehensively-eligible replay exits
nonzero even when its physical evidence is useful.

## 238. Optimization plateau is not comprehensive convergence

The four-selector plateau rule remains an optimization stopping criterion. It
does not by itself prove deployable accuracy and cost. Convergence inspection
must parse `latency_guardrail_supported`, `latency_guardrail_passed`, and
`comprehensive_promotion_eligible` as explicit fail-closed binary markers. A
claimed latency pass without support or a comprehensive promotion without a
latency pass is contradictory evidence and invalid.

The supervisor may stop a source-frozen campaign after its declared physical
optimization plateau, but it records that state separately from comprehensive
convergence. The read-only monitor renders `OPTIMIZATION PLATEAU` when the
training plateau is real but comprehensive eligibility is absent, and renders
`CONVERGED` only when both the plateau and comprehensive promotion gate pass.
Missing external latency evidence must never be displayed as success merely
because the optimizer stopped changing the incumbent.

Disjoint validation, test, and OOD RGB-only qualification remains mandatory
after an eligible fixed-manifest candidate, including held-out scenario,
object-count, camera, parameter, event, recovery, uncertainty, identity, and
long-horizon evidence. The paired validation latency replay does not replace
those generalization gates.

## 239. Qualification boundary and the first grounded selector result

The reviewed implementation snapshot before version/memory synchronization
had worktree fingerprint
`cb0dd4b4939ed1fba454fe33f7cb5722feb29c63073831bc8368d06b08b9002f`
and runtime-source fingerprint
`dd3de0d68489e62311fdf53f2d3b2f3720b0303c9c08c38443549127788779d3`.
Its pre-version parity artifacts remain under
`/private/tmp/orpheus-spec151-parity-final-20260821T084546Z` for provenance.

An earlier post-version parity compared clone runtime
`957a5277a266ea009e8500946370a5f2adc7fc26ce8c3a9e83fb6f287c74b05b`
with source-frozen reference runtime
`0d8499cf5a9f5ba87fe88f432659d519231942fd8ab923c615cbbfad2fd846da`.
Those historical artifacts are under
`/private/tmp/orpheus-spec151-parity-postversion-Rmaj0N`:

- `qualification.json` SHA-256
  `2ce77b69470a54ae65b74130dbda021c32cf96b022c335dbfe33d954ee85cd9`;
- supplemental evidence SHA-256
  `77281d1e8e69cfe40a76744805c7962cf3d538cd7ec5466a46c9eaef64548dfc`;
  and
- canonical protocol SHA-256
  `714436f02442ad3ad82d9b17d5772993a4f67b09e52a5cdb776c21427380991c`.

The trainer comparison preserves all `309` common metrics and all `9` loss
terms exactly while adding `182` evidence fields. The held-out evaluator
preserves all `1025` common non-latency metrics exactly while adding `577`
evidence fields. Posterior traces, final runtime state, model state, and the
`740491`-byte checkpoint with SHA-256
`61ad6691148bf4c070a9a63adf6f7be243ed1e1f9b612b8cdbb80ce342855475`
remain exact. The supplemental audit proves the canonical primary-key/hash/
exclusion contract and
the fail-closed result `physical=true`, `comprehensive=false` when latency is
missing. The only expected old/new metadata transitions are specification
`1.50 -> 1.51` and evaluator metric/per-scenario schema `v2 -> v3`. The
changed-suite gate is `304 passed in 96.19 s`; the focused hash-
repair gate is `34 passed in 6.94 s`; Ruff and diff checks pass. This is
implementation/parity evidence, not a real paired 32-episode MPS latency gate
or a model-accuracy promotion.

The source-frozen specification-1.50 grounded campaign at
`runs/20260821-052948-grounded-convergence-spec150-hybrid` reached its first
trained fixed selector at update `512`. Against the protected step-zero
incumbent, the candidate improved horizon-weighted score
`0.2654622904 -> 0.240757` (`9.31%`), current position
`0.1580637 -> 0.132546 m`, current velocity
`0.8310919 -> 0.749308 m/s`, all five pooled position horizons, every pooled
axis/horizon position slice, and all pooled velocity horizons. Coverage,
precision, collision F1, and nominal-coverage behavior also improved in the
pooled aggregate.

The same candidate failed `65` persisted scenario/axis guardrails and was
correctly rejected. The dominant failures were `35` z-position and `12`
x-position regressions plus scenario-specific damped-contact z/event/coverage,
heavy-light current-x/identity, camera x/z, and late collision-F1 failures.
Global identity switching also worsened while remaining within the older
global tolerance. Candidate checkpoint
`validation_step_000512.pt` is retained with whole-file SHA-256
`f8f1704c2552ea51a7626729140608203918c909cfc258d63158c347bef4eb86`;
the protected incumbent was not replaced. This is strong aggregate learning
and useful architecture-diagnosis evidence, but it is neither promotion nor
convergence. At this intermediate boundary the immutable version-1.50
optimization trajectory was allowed to reach its next declared selector at
update `1024`; section 240 records the resulting rejection and pause.
Version-1.51 evidence must be collected through a fresh weights-only
qualification rather than exact resume.

No real paired 32-seed MPS latency report, disjoint validation/test/OOD gate,
or comprehensive convergence claim exists at this amendment boundary.

## 240. Specification-1.51 closure, rich shadow result, and next repair

The final frozen repository gate ran in the unchanged `orpheus` environment
with custom Torch `2.9.0a0+gitcbe1a35` and available MPS. Candidate runtime-
source fingerprint
`6247e913c41dc150e0f2fb66fa86b42a1a504083dd16110ba0a682e07af579a5`
and worktree fingerprint
`1836ecc3e2c0e32aeac2deb7afedc85237e480850000a39e77a1d9088b3d15f8`
were unchanged across the gates. The exact full suite is `1080 passed, 16
skipped in 424.71s`; Ruff check passes, Ruff format reports all `219` files
already formatted, compileall passes with an isolated temporary bytecode
cache, and `git diff --check` passes.

The final fixed tiny-checkpoint parity replay is under
`/private/tmp/orpheus-spec151-final-parity.x8PY0e`. It compares candidate
runtime `6247e913...` with main reference runtime `0d8499cf...`. Strict
qualification SHA-256 is
`2ce77b69470a54ae65b74130dbda021c32cf96b022c335dbfe33d954ee85cd9`;
the fresh supplemental SHA-256 is
`3793c659fdb1de15d0f2792e8e58da66a3f9b82ad268a1e00c902eb9c87b86d0`.
The replay preserves all `309` common trainer metrics and `9` losses, all
`1025` common evaluator non-latency metrics, the `16`-frame posterior traces,
final runtime state, model state, and the exact `740491`-byte checkpoint with
SHA-256
`61ad6691148bf4c070a9a63adf6f7be243ed1e1f9b612b8cdbb80ce342855475`.
The exact shared-output digests are:

- common trainer metrics:
  `2a64b0162510396af241cfdc19c8c7327da62bc6cbf52bc06bf51e0392048b67`;
- total plus nine losses:
  `54eef069f1397f7af4b3fa7592a1b6557c5c26a26061ad92bf6dd8b64803d4d0`;
- trainer posterior trace:
  `409fa9aad104804134041e6dccb06d1dda36ab8f9e3b83765155bae16e1ca5f8`;
- final runtime state:
  `b53db3c4f51c7747f8f0666b4deb9fa13ce44c10341c787f2d65cd352074ed1d`;
- model state:
  `6ae10cbdbe37f13f3ec08202a343b8710d6034e472a96781c19a99b2015c055d`;
- common evaluator non-latency metrics:
  `04ca4ab9901996d29c4d6c6dd0acfdad0c15ee66e0576aebd0ad4581a6e914ba`;
- evaluator posterior trace:
  `06c416d8af3f4599bb6b46e872c55d793adb9929627f978865624f6b94d69be0`;
  and
- new primary-physical metrics:
  `49bd23fd780960f3d56af3e69b2c14050fe31e21609e3abd83be7ea0f7e58e8e`.

Three absent latency components correctly leave `physical=true`,
`comprehensive=false`.

The version-1.51 rich fixed-32 shadow replay of the step-512 candidate is
support-complete (`32/32` episodes and `8/8` scenarios) and improves the
selector `0.2654622895 -> 0.2407574475` (`9.31%`). It nevertheless has `237`
rich guardrail failures and is rejected. Overlapping diagnostic-family counts
are `61` velocity-axis, `13` pooled velocity, `67` Gaussian NLL, `8`
calibration, `15` event-horizon, and `11` identity failures; they must not be
summed as disjoint classes. Report SHA-256 is
`345792e657246f86f9e74013837cdbc18b37298c84d9f907b860521834f2f362`.
Neither physical nor comprehensive promotion passes, and deployment remains
the protected step-zero incumbent.

The unchanged source-frozen version-1.50 campaign was paused after its
numbered step-1024 checkpoint. Its optimizer is healthy: all and only the
intended `79` updater tensors changed, Adam steps are `1024`, frozen tensors
and buffers are exact, attention decoders remain exact zero, the event head is
unchanged, and parameters and moments are finite. That integrity does not make
the result accurate. Fixed-selector failures worsen `65 -> 118` from step 512
to 1024; score worsens `0.2407574 -> 0.2618322`, current position
`0.1325464 -> 0.1838629`, x `0.1031069 -> 0.2239196`, z
`0.1783838 -> 0.2075144`, identity/lifecycle
`0.0013598 -> 0.0110931`, and coverage/precision also fall. Velocity, y, and
collision F1 improve, but cannot compensate for the axis, position, and
lifecycle regressions. The campaign is paused, deployment stays at step zero,
and there is no convergence claim.

The next authorized learning work is a narrow axis-gated updater-state repair
initialized weights-only from the protected base, followed by a long paired
qualification. The repair must keep RGB-only `WorldBelief` truth, the causal
short-step analytic-plus-learned-plus-event loop, heterogeneous/local
applicability and model-selection semantics, and protected-incumbent rules.
Screen repaired and unchanged control arms through at least the fixed
step-512 and step-1024 selectors; extend only a surviving arm through a
bounded multi-selector qualification before fixed-32 version-7/version-16,
paired latency, and disjoint RGB-only test/OOD gates. Do not infer convergence
from healthy optimization or a favorable pooled score.

## 241. Specification-1.52 axis-gated updater-head repair and pre-launch contract

Specification 1.52 implements the narrow repair authorized by section 240
without changing the analytic physical update, the RGB-only truth boundary,
or the protected-incumbent selector. This is a source and protocol amendment
before retained training evidence. It is not a successful smoke, fixed-
manifest result, promotion, plateau, or convergence claim.

The opt-in
`model.filter.learned_correction_independent_axis_support` contract makes the
original typed `MeasurementSet` provenance authoritative. When enabled,
`world_position_independent_axis_mask` gates the learned corrector's mean and
log-variance contributions on each position coordinate and on velocity
coordinates derived from that position evidence. A copied or custom
`InnovationSet` cannot widen the mask. A source-bound ROI row with absent
independent-axis provenance fails closed to no learned position-derived
coordinate support; a historical global/unbound observation without source
identity retains all-axis compatibility. Direct analytic Kalman position and
velocity fusion is unchanged. The flag defaults to `false`, and historical
configuration/checkpoint normalization therefore preserves exact legacy
behavior unless the repaired protocol selects it explicitly.

The new `updater_state_heads` training scope exposes exactly these six tensors:

- `updater.learned_corrector.mean_head.weight` and `.bias`;
- `updater.learned_corrector.variance_head.weight` and `.bias`; and
- `updater.learned_corrector.gate_head.weight` and `.bias`.

The modality embedding, shared corrector representation, and the mode,
existence, and visibility sibling heads remain frozen. Freezing only sibling
parameters is insufficient because adapting the shared trunk would change
their outputs; the shared representation is therefore outside the functional
ownership boundary. An AdamW regression must prove that a state-head update
cannot change frozen parameters, optimizer state for frozen tensors, or the
three sibling outputs. Event, existence, parameter, and measurement losses
have explicit zero ownership in this repair phase.

`training.closed_loop_batch_macro_physical_losses_enabled` first normalizes
supported elements within each batch row and then averages only supported
rows. In the scenario-balanced eight-row campaign, this prevents an episode
or scenario with more matched objects or coordinates from dominating current
position, supported current velocity, detached-mean current Gaussian NLL,
rollout position/velocity/NLL, or correction hinges. Unsupported rows remain
omitted rather than represented by differentiable zeros.
`training.closed_loop_axiswise_correction_hinge_enabled` replaces vector-norm
non-regression with absolute per-coordinate posterior-versus-prior hinges, so
an improvement on y cannot cancel an x or z regression. The configured full-
horizon denominator remains fixed. Both switches default to `false` for exact
historical semantics.

A real deterministic balanced eight-scenario CPU batch was attributed against
the exact protected-step-zero plus complete step-512-updater composition,
whose model-state SHA-256 is
`88f2df4d8a2621e8907497298a6d264015714a961102a83d1f65cd9f4474318b`.
The provisional aggregate correction weight `0.1` divided equally over the
position hinge, velocity hinge, and correction-magnitude term supplied only
`0.1704%` of the total physical updater gradient norm. The magnitude term is
the analytic Kalman position-correction norm, not a learned-residual norm; its
gradient was concentrated in the variance head and opposed the other physical
gradient with cosine `-0.80133`. The pre-smoke profile therefore uses explicit
weights `correction_position=7.0`, `correction_velocity=2.0`, and
`correction_regularization=0.0`. On the measured vectors this predicts a
`12.4455%` correction-to-total norm ratio, total norm `0.0713825`, cosine
`-0.00433` against the other physical aggregate, and no clipping. These are
single-batch attribution predictions, not retained optimization or accuracy
evidence. Section 243 records their subsequent deterministic two-update wiring
exercise, which does not make them accuracy-qualified weights.

The weights-only initializer is governed by
`checkpoint_initializer_composition_v1`. Its protected base is the exact
specification-1.44 step-zero artifact (checkpoint SHA-256 `0ba00e72...`, model
SHA-256 `1354bdfc...`); the specification-1.50 step-zero compatibility witness
has checkpoint SHA-256 `b84e5299...` and the same model state; and the complete
step-512 updater donor has checkpoint SHA-256 `f8f1704c...` and model SHA-256
`1942c2c9...`. Composition selects `23` updater tensors/`126164` elements,
changes `21` tensors/`125971` elements, and must produce model SHA-256
`88f2df4d...`. The materializer captures each source once into immutable
bytes, verifies path confinement, SHA-256, byte count, step, model hash,
simulator/specification identity, source config, run ancestry, and allowed
target-config differences, then strict-loads the composed state. Its output is
non-overwriting and read-only, carries deep-copied composition metadata, has
no optimizer or scheduler state, and is categorically invalid for exact
resume. `donor_weight` is the exact numeric value `1.0`; a YAML boolean is not
a numeric donor weight even though Python's type hierarchy treats booleans as
integers. The production artifact must be materialized only after the source
is frozen. No durable initializer existed at this initial pre-launch boundary;
section 243 records the later immutable materialization.

The paired experiment has one exact causal contrast. Treatment and control
must share the same materialized initializer, seed, scenario-balanced draw
order, devices, `updater_state_heads` scope, runtime independent-axis gate,
loss weights, optimizer, schedule, cadence, and validation manifests.
Treatment enables batch-macro physical losses and axiswise correction hinges;
control disables only those two switches. A separate zero-update gate-on/
gate-off forward ablation measures the runtime semantic delta and is not the
training control. Before either long arm, run a deterministic two-update pair
and require identical draw identities, exactly the six allowed trainable
tensors and their Adam moments, exact frozen state and sibling outputs, finite
losses/gradients, and coherent cadence variation; section 243 records this
wiring gate as complete. Section 244 records the retained common fixed-32
step-zero evidence as complete without a rerun. Each retained arm is
configured for `3072` updates with fixed selectors every `512` updates; both
must reach at least selectors `512` and `1024` unless an integrity or nonfinite
failure requires a truthful stop. Do not interpret a short-run difference as
plateau evidence.

At the initial amendment boundary, every experimental gate was pending.
Section 243 subsequently closes only immutable initializer materialization and
the paired two-update wiring smoke; section 244 closes the common rich
fixed-32 step-zero comparison. The gate-on/gate-off zero-update ablation, long
paired training, real paired latency, and disjoint RGB-only
validation/test/OOD qualification remain pending. The protected step-zero
incumbent remains deployed, and no specification-1.52 checkpoint is promoted.

## 242. Exact-resume snapshot, publication, and run-ownership contract

Exact resume is one fail-closed transaction over immutable input evidence and
one exclusively owned destination. The CLI lifecycle state required by
section 177 is intentionally published before trainer entry. Before the
trainer publishes its own resolved config, run metadata, selector/history
artifact, checkpoint, or continuation evidence, it must:

- capture the primary checkpoint exactly once into private read-only bytes and
  derive its SHA-256, byte count, deserialization, compatibility checks, and
  any branch publication from that capture. A later source-path open may verify
  that the in-place identity is unchanged, but must never replace the captured
  bytes as the load or copy input;
- capture every required measurement/rollout selector and accepted numbered
  validation-history artifact once, validate the complete linked set, and
  stage only those captured bytes for later atomic publication;
- validate the complete finite model, optimizer, and scheduler structure;
  resolved configuration, specification/simulator/validation protocol,
  launch-time executable source, phase device, step and dynamic learning rate,
  handoff/final-validation markers, and optimizer/data-draw/skipped-draw
  counters; and
- validate Python, NumPy, CPU Torch, and applicable CUDA/MPS RNG state on
  private generator objects without changing process-global RNG.

Adam/AdamW continuation requires globally unique serialized parameter IDs,
exact destination group count/schema/cardinality, valid and destination-equal
static group options, a dynamic learning rate bound to the saved completed-
update schedule, state entries owned by declared destination parameters,
scalar step tensors with the required dtype/device, and every required finite
moment with the destination parameter's shape, dtype, and device. A deep-
copied optimizer must then complete one finite disposable next step. This
proof must not mutate the live model or optimizer.

After model/optimizer loading and linked-artifact preflight, the trainer
rechecks branch ownership and, for an in-place resume, the captured source
checkpoint identity. Only then may it restore process-global RNG, exactly
once, and atomically publish the already captured artifacts. A continuing
resume releases the deserialized checkpoint payload before its first update;
weights-only initialization likewise releases its deserialized payload after
model loading and provenance extraction. Captured primary bytes may remain
only as long as required to publish an unchanged branched no-op checkpoint.

Every exact-resume invocation that can mutate a run holds one non-blocking
per-run `flock` for the complete invocation. The CLI claims the run directory,
opens the lock inode, and passes that live inode-bound handle into the trainer;
the public direct API self-acquires the same lifetime ownership when no handle
is supplied. At acquisition, the trainer verifies that the open descriptor and
lock path name the same device/inode and that the descriptor holds the
exclusive lock, then keeps that verified handle open through the invocation.
This excludes cooperating writers for the lifetime of publication; cleanup
rechecks inode identity before it may unlink or restore a lock path.

Supplying any explicit `run_name`, including the source run's own name, is a
branched resume. A branch destination must be absent or empty when it is
atomically claimed. An unprefixed requested label resolves to one timestamped
directory exactly once; lock acquisition, preflight, and publication reuse
that resolved path. Immediately before the publication boundary, the trainer
requires the exact entry set created by this invocation and rejects any
concurrent or pre-existing evidence without overwriting it. In-place resume
must similarly recheck that `checkpoints/last.pt` still has the captured hash
and byte count. Failed lock acquisition or cleanup may remove/restore a lock
file only when the caller still owns that exact inode; it must never unlink a
winner's inode after losing the lock race.

Focused artifact-integrity qualification passed the complete checkpoint and
trainer gate (`128 passed`, `1` expected MPS-only skip), a post-format
ownership/entrypoint gate (`15 passed`), the complete entrypoint file
(`7 passed`), and the initializer materializer file (`3 passed`). On
active-Aqua custom Torch `2.9.0a0+gitcbe1a35` with MPS built and available,
invalid-MPS-RNG rejection before publication and exact MPS RNG restoration
both passed (`2 passed`, `0 skipped in 2.86s`).

The final post-provenance frozen-source command
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run --no-capture-output -n
orpheus pytest -q` passed `1163` tests with `17` skipped in `461.37s`
(`0:07:41`). Whole-tree Ruff check passed; Ruff format check reported `221`
files already formatted; isolated compileall over `world_model`, `tests`,
`train.py`, `evaluate.py`, `monitor.py`, and `scripts` passed; the focused final
source/version selection passed `6` tests; the authoritative version contract
passed `1` test in `0.78s`; and `git diff --check` was clean. A final
independent review returned PASS with no blockers. These results close source
and artifact integrity only. They are not a model/trainer smoke, fixed-
manifest metric, promotion, plateau, or convergence claim.

## 243. Immutable initializer and paired two-update wiring qualification

The frozen specification-1.52 source was committed and pushed at
`f08200f44646db6fa84f32de4b5bf538e647f546`. From that source, the production
weights-only initializer was materialized at
`/Users/mike/Work/world.model/runs/diagnostic_initializers/20260821-151100-spec152-axis-gated-updater-initializer`.
Its `initializer.pt` is `12,143,891` bytes with file SHA-256
`298b660bba574216321f68517ad1aee7403cc5812289279cb9099223c2eea4a5` and
model-state SHA-256
`88f2df4d8a2621e8907497298a6d264015714a961102a83d1f65cd9f4474318b`.
The initializer, resolved config, and manifest are all mode `0444`. The
manifest proves `23` selected tensors are exact donor values, `21` actually
changed, and every nonselected tensor is exact protected-base state. This is a
weights-only artifact and remains invalid for exact resume.

The paired two-update treatment and control completed successfully in
`20260821-151249-spec152-axis-gated-two-update-treatment` and
`20260821-151249-spec152-axis-gated-two-update-control`. Their hardened audit
is `/private/tmp/20260821-151249-orpheus-spec152-two-update-audit.json`, with
SHA-256 `5ff81f672b55b2915180f5cadefebb331602383d6cb14b5c6c980dc241acb18f`.
Audit schema `orpheus_spec152_two_update_audit_v2` reports `passed=true` and
zero failures; the audit script SHA-256 is
`07f357edddd785bc4b0bfe5f1e6ade77b5f89fd99a1ed1e542c00e331a43acaf`.
The paired resolved configs differ only in
`training.closed_loop_batch_macro_physical_losses_enabled` and
`training.closed_loop_axiswise_correction_hinge_enabled`. Both arms consumed
the exact same two ordered balanced batches, updated exactly the six declared
mean/variance/gate head tensors with exactly six Adam owners, preserved all
`219` frozen tensors/buffers, and recorded no retry or skipped draw.

For treatment, update-one/update-two `loss_total/gradient_norm` was
`0.8310171366/0.0713825151` and `1.5899894238/0.0575892627`. For control it was
`0.7718598247/0.0641159415` and `1.8167935610/0.0712272376`. Both used learning
rates `3.90625e-8` then `7.8125e-8`. Each run completed its terminal
`32/32` fixed validation. The resulting selector changes were only on the
order of `1e-8`, below the declared `1e-5` minimum-improvement threshold, so
both candidates were rejected.

This qualification proves initializer provenance, paired data/config
identity, six-head optimizer ownership, finite two-update execution, and
terminal-validation wiring. It is not evidence of accuracy improvement or
promotion. Section 244 closes the common rich fixed-32 step-zero gate from the
already retained pre-update artifacts. The separate gate-on/gate-off
zero-update forward ablation, `3072`-update paired arms, paired latency,
disjoint RGB-only validation/test/OOD, promotion, plateau, and convergence
gates remain pending.

## 244. Common rich fixed-32 step-zero equivalence

The retained `validation_step_000000.pt` artifacts from both `151249` arms
already constitute the required common rich fixed-32 step-zero baseline; no
evaluation rerun is required. They bind to initializer file SHA-256
`298b660bba574216321f68517ad1aee7403cc5812289279cb9099223c2eea4a5`,
initializer model-state SHA-256
`88f2df4d8a2621e8907497298a6d264015714a961102a83d1f65cd9f4474318b`, clean
source commit `f08200f44646db6fa84f32de4b5bf538e647f546`, and runtime-source
fingerprint `75ee1ae6d07124d738ce6a400517f27ab42d23ba0fb09fd4d05c4fc400d6c0e7`.

Both use rollout protocol version `16` with protocol hash
`dffa53ce82a9a7dee9e7a7b069f665754002b5f40f7ea0488e8f0f5ce7ad6708`,
selector version `7`, and seed-manifest SHA-256
`e27bdf2dffb5f36545bc7cbae5d88514fb9537cd5fa07cd26276ccefd41b46be`.
The manifest is exactly seeds `100000..100031`, four episodes for each of the
eight declared scenarios. All `32/32` episodes pass core support, every
scenario has `4/4` rich-support episodes, and the support-schema marker is
`1.0`.

Across the two arms, `281` pooled-additive, `2248` scenario-additive, `3296`
scenario, and `2064` per-seed evidence fields are exact. All `129` physical
validation fields, model state, empty Adam state, scheduler state, and RNG
state are exact. Outside the two declared objective flags, only objective-loss,
timing, and RSS diagnostics differ; those fields do not define the physical
baseline.

The common step-zero selector score is `0.2395286358786779`. Current position
RMSE is `0.15056456382003996`, velocity RMSE is `0.8118821097143433`,
association coverage is `0.90425`, precision is `0.9463631606488749`,
collision F1 is `0.2837465564738292`, identity-switch rate is
`0.0008092797410304828`, and position coverage90 is
`0.8435185185185186`. Paired latency support and comprehensive eligibility are
both false, so this common physical baseline cannot promote either arm. It
closes only the shared rich fixed-32 step-zero gate.

## 245. Axis-gate zero-update semantic ablation

The deterministic gate-on/gate-off zero-update ablation is complete. The
gate-off arm is the preserved fixed-32 forward from
`/private/tmp/20260821-spec152-axis-gate-zero-update-v2`; its exact raw,
progress, failure, and contract bytes were copied into the immutable finalized
artifact at `/private/tmp/20260821-spec152-axis-gate-zero-update-final`.
`report.json` is `422,423` bytes with SHA-256
`1fdb9ea9f3ba5ada2f38b9442bfd60653a5f54f179abd9458bc216ba209e1f1b`;
the retained finalizer source is `21,502` bytes with SHA-256
`acd521d8610ad11833c2aa8cc039aaf7b777ccb7adedc2b5d66d4748d8ac46ee`.
The report is schema `spec152_axis_gate_zero_update_v3`, `passed=true`, with
zero failures. Every file is mode `0444` and the directory is mode `0555`.

The finalizer did not construct a model, loader, optimizer, or dynamics
forward. Runtime sentinels prove no forbidden entry point was called. It
recomputed the production rich-support validator and selector from the exact
preserved forward, injected only the trainer-derived
`validation_rollout_selection_score`, and required exact source, config,
initializer, model-state, seed/scenario, and gate-on evidence identities.
Independent read-only recomputation reproduced rich support `1.0`, selector
`0.22629706900790234`, all six metric-scope schemas and hashes, and unchanged
initializer model-state SHA-256 `88f2df4d...`.

Gate-off improves the pooled selector from `0.2395286358786779` to
`0.22629706900790234`, current position RMSE from `0.15056456382003996` to
`0.1353678963014635`, velocity RMSE from `0.8118821097143433` to
`0.7488860388189023`, and collision F1 from `0.2837465564738292` to
`0.31891891891891894`. It changes `204` physical metrics, `152` pooled
additive metrics, `102` causal-additive metrics, and `123` derived validation
metrics. This proves the runtime gate is semantically non-vacuous. It does not
establish that gate-off is safer or promotable: gate-off versus gate-on has
`211` strict guardrail failures, the reverse comparison has `393`, and the
camera-parallax scenario score regresses from `0.14423294011785356` to
`0.14953841960363534`. The long treatment and control therefore retain the
declared gate-on runtime and compare only batch-macro reduction plus axiswise
hinges. Paired latency, long training, disjoint RGB-only test/OOD, promotion,
plateau, and convergence remain pending.

The exact version-2 producer bytes were overwritten before they could be
copied. The preserved contract and failure record consistently bind their
reported SHA-256/size, and an independent reviewer inspected those bytes
before replacement, but the finalized report truthfully records
`exact_producer_bytes_retained=false`. The completed forward outputs
themselves are retained byte-for-byte and are the sole numerical input to this
finalization.

## 246. Long treatment launch and repeated ownership gates

The frozen treatment arm launched from clean source
`f08200f44646db6fa84f32de4b5bf538e647f546` as
`runs/20260822-192031-spec152-axis-gated-3072-treatment`. Its immutable
initializer is the section-243 artifact, and its step-zero fixed-32 selector,
current position/velocity, lifecycle, event, calibration, and every declared
horizon metric match the section-244 common baseline exactly. The active
closed-loop graph is CPU float32, RGB is the only runtime modality, debug
oracle is disabled, and custom Torch is `2.9.0a0+gitcbe1a35` with MPS built
and available for the configured measurement preference.

The durable step-128 `last.pt` has SHA-256
`c33cdeaf812f40a44d7149747a17e42bd022182ab26a62e0facda70922f8e285`
and is `12,593,239` bytes. Against the initializer, exactly the six declared
mean/variance/gate head tensors changed and exactly those six own Adam moments;
every Adam step is `128`. The other `219` model entries and all four buffers
remain bit-exact, and all model/optimizer tensors are finite. Persisted
data-progress is `134` draws for `128` applied updates plus six no-gradient
retries, satisfying the exact-resume invariant.

Logged applied blocks are exactly steps `8..128` at cadence eight. Every block
contains all eight scenarios, no nonfinite value, no global clipping, and zero
perception/interaction gradient. Loss ranges `0.2861376..1.8364884` with mean
`1.2304651`; applied gradient norm ranges `0.0244227..0.1393826` with mean
`0.0779922`. Causal trajectory support remains `714..1088`; objective-family
support is `6..10`; maximum RSS is `1,045,860,352` bytes. Retry draw identities
are `6,23,39,50,71,117`. They are isolated structurally zero-gradient draws,
within the declared maximum of 16 retries per update; the later control must
reproduce their identities or fail paired-protocol qualification.

This is a technical ownership, finiteness, throughput, and exact-resume gate,
not an accuracy or convergence result. The step-512 and step-1024 fixed-
manifest selectors remain the first scientific continuation decisions. The
control arm remains unstarted until the treatment satisfies those declared
boundaries.

The same immutable comparison was repeated at durable step `256` before the
first accuracy gate. That `last.pt` is `12,593,239` bytes with SHA-256
`200f9aa2313f1d9939420284bbfe43e53c8d0f21616a99edcc004b67429ecaee`.
Exactly the six declared heads changed and own Adam moments, every Adam step is
`256`, the other `219` model entries/buffers are bit-exact, and all model and
optimizer tensors are finite. The independent report at
`/private/tmp/20260822-spec152-treatment-step256-audit.json` has SHA-256
`5de67a367d58a52ef31327c4107af5afa40253e8a02912f8352b2b9e7e6a398e`.
This repeated gate proves ownership continuity only; step `512` remains the
first trained fixed-manifest scientific decision.

---

## 247. Regime-local hypothesis applicability and bounded composition

The heterogeneous runtime hypothesis pool remains an explicit opt-in
evaluation intervention. Specification 1.53 repairs its causal applicability
boundary without changing the learned model, `WorldBelief`, normal runtime
default, simulator, or training campaign. A candidate choice is now stored and
queried by persistent entity, independent world axis, interaction regime, and
exact evidence horizon. Lifecycle identity changes clear every affected cell.

Interaction regime is derived only from the accepted learned/structured
prediction: free motion, ground contact, pair contact, collision, occlusion,
or external actuation. Simulator truth and the alternative candidate being
scored may not classify the regime. Constant-velocity candidates are eligible
only in free motion; ballistic contact is eligible in free, ground-contact,
and pair-contact regimes; collision and every unsupported regime fail back to
the learned candidate. The learned candidate supports every regime.

Applicability is fail-closed. A local cell must meet configured support-count,
evidence-age, RGB observability, and posterior confidence thresholds. Evidence
comes only from an accepted delayed candidate prediction followed by a
persistent-ID-associated RGB world-position measurement. Candidate scoring
uses predictive plus RGB measurement variance; optional robust influence
clamps one observation's relative log-evidence. Missing, stale, weak,
unobservable, lifecycle-invalid, or capability-incompatible evidence selects
the learned fallback. The evaluator must expose support, freshness,
observability, selected predictive variance, confidence margin, regime,
candidate/fallback counts, and exact additive partitions.

When `runtime.hypothesis_composition_step_seconds` is enabled, it must match
one configured evidence horizon exactly. Forecasts are recursively composed
on that bounded shared grid: every step first advances the learned joint model,
then replaces only supported configured position/velocity axes and their
chosen transparent candidate state. Learned uncertainty, lifecycle, identity,
motion mode, event output, cross-axis state, and all unsupported axes remain
authoritative. Collision evidence is max-pooled over each query interval.
Mixed-row or nonaligned query grids fall back to one ordinary learned rollout
and are reported separately; they may not silently couple rows or interpolate
evidence.

The following fields are strict, exact-resume-bound runtime semantics and
retain legacy-disabled defaults:

- `hypothesis_local_applicability_enabled=false`;
- `hypothesis_minimum_support_count=1`;
- `hypothesis_maximum_evidence_age_seconds=1.0`;
- `hypothesis_minimum_observability=0.0`;
- `hypothesis_minimum_confidence_margin=0.0`;
- `hypothesis_robust_influence_delta=0.0`; and
- `hypothesis_composition_step_seconds=null`.

The implementation gate includes a real one-episode RGB-only CPU evaluator
run with local applicability and `0.05`-second composition. It completed with
no oracle input or nonfinite output and emitted exact regime and composition
partitions. The post-version report is
`/private/tmp/20260822-213000-spec153-runtime-pool-clean-e2e/evaluation.json`
with SHA-256
`2ded1df22696e664e4936fd4eda6559dcbe2ef118906c70ac2f2ec1e12c85214`,
clean commit `41d2c092520aeb2c2c6c302466317e5d9c01f8ac`, and runtime-source fingerprint
`f3705bf77a924c7754146d21a8ca46508a497f64a84fbe041e31edda4ea1aec5`.
The focused compatibility gate passed `381` tests with `3`
expected inactive-Aqua MPS skips; after final edge-case additions, the direct
hypothesis/evaluator selection passed `56` tests. Ruff, compileall, and diff
checks pass. The final whole-repository gate passed `1196` tests with `17`
expected backend skips in `482.97s` (`0:08:02`); whole-tree Ruff check,
format check (`221 files already formatted`), isolated compileall, the version
contract (`1 passed in 0.79s`), and diff check also pass. These are
implementation and instrumentation results only. The actual local/composed
path also passed a focused active-Aqua MPS RGB regression under custom Torch
`2.9.0a0+gitcbe1a35` (`1 passed in 16.87s`), including finite output and exact
candidate/total/regime count partitions.
Promotion still requires a source-frozen matched learned-only versus
regime-local fixed-32 comparison, full physical/uncertainty/event/identity
guardrails, paired latency, and disjoint RGB-only validation/test/OOD evidence.

The matched fixed-32 boundary is now executable through
`scripts/compare_runtime_hypothesis_evaluations.py`. The comparator captures
each evaluator report once, binds both arms to the same immutable checkpoint,
clean source, tracked resolved configuration, exact standard-validation or
disjoint fresh-validation/test/OOD seeds and scenario mixture, five horizons,
MPS/float32 runtime environment, posterior
trace, evaluator schemas, and primary-physical digest. The reference must run
learned-only and the candidate must run the complete configured local/
composed policy. The candidate must exhibit positive non-learned selection and
composition while preserving exact support and current-posterior evidence.
Pooled and every scenario/horizon then fail closed on position and velocity
RMSE (pooled and x/y/z), Gaussian NLL, calibration error, sharpness, forecast
coverage, identity association/mismatch, collision F1, and dropped forecasts.
Promotion additionally requires at least `1e-5` metres summed pooled position
improvement and paired global/fast/rollout latency no worse than `1.10x`.
Physical, latency, and comprehensive eligibility are recorded separately.
`scripts/aggregate_runtime_hypothesis_promotion.py` then requires the standard-
validation, disjoint fresh-validation, test, and OOD comparisons together. It
reopens every declared raw evaluator report, verifies its captured identity,
reproduces each pair decision under the current clean source, and accepts only
four comprehensive decisions from the same checkpoint and policy.

Evaluator reports now persist the exact host, Python, Torch, requested/resolved
device, precision, and accelerator-capability environment needed for a paired
latency claim; the existing MPS replay verifier binds the same metadata. The
final affected comparator/evaluator/replay gate passes `62` tests in `46.39s`;
Ruff format/check, isolated compileall, and `git diff --check` pass.

The first production standard-validation attempt is terminal rejection
evidence, not promotion. After moving only detached integer diagnostic
reductions to the host to avoid a reproducible custom-Aqua-MPS compiler abort,
both clean commit-`33ff3e0` arms completed the exact 32-episode MPS manifest
with zero nonfinite outputs. The candidate composed 2450 x-axis steps, 1681
from non-learned candidates, but its persistent posterior trace differed from
the learned-only reference and violated the required forecast-only boundary.
A diagnostic comparison also failed 598 physical, uncertainty, event,
identity, and scenario guardrails, with every pooled position horizon slightly
worse. The comparator remains fail-closed; no disjoint run or runtime-default
promotion is permitted from this candidate.

---

## 248. Forecast-only hypothesis isolation and conservative velocity evidence

Specification 1.54 closes the causal failure exposed by the first 1.53
production pair. A scheduled hypothesis rollout is evidence for future
selection only. It may never be reused as the physical prior for ordinary or
prepared observation ingestion, even when its source belief, model revision,
and nominal endpoint match. Persistent `WorldBelief` propagation always calls
the canonical learned dynamics path. This makes the online posterior
independent of whether the optional pool is attached.

Composed forecasts may replace only configured supported position and velocity
axes. Predictive uncertainty remains owned by the learned trajectory; an
analytic candidate may not substitute its own state variance. If a composed
query contains no non-learned intervention, every physical trajectory tensor
and timestamp must be copied from one ordinary learned rollout exactly rather
than reconstructed through the short-step grid.

Delayed evidence may optionally include causal RGB temporal velocity, its
per-axis validity mask, and its reported variance. The evidence is collected
only after the observation module updates temporal history and before direct
velocity correction changes persistent belief. Two new strict, exact-resume-
bound fields retain disabled defaults:

- `hypothesis_velocity_evidence_weight=0.0`; and
- `hypothesis_velocity_nonregression_gate_enabled=false`.

When the non-regression gate is enabled, a non-learned entity-axis candidate
is eligible only if its uncertainty-aware temporal-velocity loss is no worse
than the learned candidate for the same valid RGB evidence. Missing or invalid
velocity evidence fails closed. Position evidence remains the ranking target;
the velocity condition is a veto, not simulator supervision and not a hidden
replacement objective.

The isolated learned-only reference and both CPU fixed-32 diagnostics retain
the exact posterior trace SHA-256
`5ecb850d11039c2a424b50e4aa499891dd396c7fec40007cfb98e5e9afb7936d`
and have zero changed `posterior_current*` metrics. The non-vacuous margin-zero
velocity-veto arm used 535 non-learned x substeps but regressed pooled position
RMSE at all five horizons by `2.66e-6` to `8.93e-6` m and regressed four of
five velocity horizons. The confidence-`0.001`/observability-`0.5` arm used
only three non-learned substeps and remained microscopically worse at 0.10 and
0.25 seconds before becoming exact learned fallback. These dirty-source CPU
reports are diagnostic rejection evidence, not a formal matched promotion:

- learned reference: `/private/tmp/20260823-033146-20260823-spec154-isolated-reference-cpu-r2/evaluation.json`, SHA-256 `e6cfd548553dc038b6ba3e1c7e3b3d95cbc8a4661f1c6832397d472bfe988fa1`;
- margin-zero veto: `/private/tmp/20260823-041910-spec154-velocity-veto-margin0-cpu/evaluation.json`, SHA-256 `d90ae8ba2c63c7e4cb558b62b018f56cc05ecd4e3bf78ea4bfc3e406f5cb6526`; and
- conservative veto: `/private/tmp/20260823-042330-spec154-velocity-veto-margin0001-observe05-cpu/evaluation.json`, SHA-256 `c4dee1919726d20a59f1f4a9c816417762a61f5afcff22a74fb7189ce21ed889`.

The focused CPU gate passes `322` tests. The final whole repository passes
`1221` tests with `19` expected inactive-backend skips in `448.06s`; the focused
active-Aqua MPS velocity-veto regression passes in `4.89s`. The learned
deployment default remains protected. No MPS fixed-32 or disjoint ladder is
authorized because the standard CPU diagnostics show no candidate accuracy
gain. The clean source-frozen confirmation on commit `19221baa4cdc85685f894938016d0b938e98ebb0`
uses the same fixed manifest, checkpoint, resolved config, CPU/float32 runtime,
and active-Aqua host contract in both arms. The reference report SHA-256 is
`060ba1ca81fe3bf7664a2222524c8a07f6cd5e6870935d3d22657208a33f01a8`,
the candidate is
`3d44a46e5f4ffd4014127cb86e754ef3acff2e0b6d8ed22ab652418ca2a4eca7`,
and the fail-closed comparison is
`42946ac61ebdb9f89f3a78d7ef7888dc7e3b3d086a98420c16a52697d5dfac33`.
The comparator confirms 535 alternative substeps and exact posterior trace,
but rejects 255 numerical guardrails, one structural event-support cell,
summed pooled position improvement `-3.5132182526070865e-05` m, and latency
ratios `1.2417/1.2181/1.8921` for global/fast/rollout versus the `1.10` limit.
This formal rejection cannot weaken the existing promotion thresholds.

---

## 249. Output-only causal residual diagnostics and rejection boundary

Specification 1.55 permits an opt-in, per-world-axis causal position residual
on forecast outputs only. Delayed RGB evidence records the learned candidate's
position error in the exact persistent entity/axis/regime cell. A correction
is eligible only under the ordinary local applicability contract and after two
successive nonzero residuals have the same sign. Lifecycle identity changes
clear both the value and its support. The configured gain is a strict finite
three-vector in `[0,1]`; every nonzero axis must be independently enabled.

The correction never mutates persistent `WorldBelief`, velocity, uncertainty,
events, lifecycle, or candidate evidence. Residual-only forecasts begin from
one exact canonical learned trajectory and add the bounded residual only to
the configured emitted position axis. This prevents short-step reconstruction
from perturbing unrelated axes. The all-zero default is exact legacy behavior,
and the field is exact-resume-bound. Evaluation reports applied count, signed
sum, and absolute sum per axis; the paired comparator accepts those diagnostics
as non-vacuous use but retains every physical, uncertainty, event, identity,
support, and latency guardrail.

The residual probe on 3,564 samples per axis found one-step lag cosine
`0.5125/0.6313/0.0440` for x/y/z. A 0.1 residual gain predicted MSE improvement
of `9.27%/11.66%/-0.12%`, so z was excluded. Real matched fixed-eight CPU
diagnostics nevertheless reject the family:

- x gain 0.1, canonical fallback: 428 applied cells, `42.69865` m absolute
  residual evidence, `0.00677117` m summed pooled position-RMSE improvement,
  but 72 failures (68 scenario-slice);
- x gain 0.1 with minimum observability 0.5: 197 applied cells and
  `0.00211706` m pooled improvement, but 56 failures; and
- x/y gains 0.1: 952 applied cells, `101.21379` m absolute residual evidence,
  and `0.01502498` m pooled improvement, but 60 failures.

Failures are coherent calibration, Gaussian-NLL, axis-RMSE, event-support, and
scenario regressions, especially camera parallax, heavy/light impacts,
glancing impacts, and impulse perturbation. No fixed-32, MPS fixed-manifest,
or disjoint promotion run is authorized. Deployment remains canonical learned
dynamics with zero residual gains. The mechanism is retained only as explicit
default-off research infrastructure; future work must jointly calibrate mean
and uncertainty from broader evidence rather than tune the fixed-eight set.

The final source gate passes `1229` tests with `20` expected inactive-backend
skips in `447.14s`; the focused active-Aqua residual regression passes in
`4.82s`. The probe SHA-256 is `82777b9482538de5701f4689b8cf2a42314873a54d27756102b7f3bde4ea7715`.
The matched learned reference is `46fa51bf...`; canonical x, observability-x,
and canonical x/y comparisons are `5ba37914...`, `8edf167c...`, and
`acc9dd37...`. These are dirty-source diagnostic artifacts and establish no
promotion or convergence claim.

A final read-only adaptive diagnostic does not reopen this mechanism.  It
estimated a bounded local least-squares gain only from causally prior residual
pairs.  With four prior pairs, x/y next-residual MSE improved
`19.81%/41.72%` across `3102/3498` samples per axis, but z worsened `1.10%`.
One reset-bounded evidence subset worsened x by `5.75%`, and the temporary
instrumentation did not establish complete scenario identity for every reset
group.  The required every-scenario non-regression proof is therefore absent.
Reports `6f656a50...` and `a6a00869...` are diagnostic only.  No adaptive
runtime state, configuration, fixed-eight/MPS ladder, or deployment change is
authorized.  Further work on this checkpoint may not tune residual gain,
duration, interpolation, or the same analytic candidates; it requires a
genuinely new independent-data or model-capacity hypothesis.

The previously unrecorded specification-1.52 treatment completed fixed
validation at steps 512 and 1024 before manual interruption at update 1044.
Both candidates were rejected: selector score moved `0.23952864` to
`0.24117313` and `0.24155691`, with 122 and 149 rejection reasons. At step
1024 x and y position RMSE improved `1.17%` and `2.53%`, and pooled velocity
improved `2.30%`, while z position RMSE worsened `7.84%`. This localizes the
next training repair to learned axis/uncertainty ownership; it does not justify
post-hoc inference composition or promotion of either checkpoint.

---

## 250. Exact lateral updater-head ownership

Specification 1.56 adds the `updater_state_heads_xy` training scope. It exposes
the same six typed mean, log-variance, and gate head tensors as
`updater_state_heads`, but only canonical fast-state rows `0,1,3,4` may change:
x/y position and x/y velocity. Rows `2` and `5` preserve z position/velocity,
and every row from `6` onward preserves orientation, angular velocity, and
modal state. The shared corrector representation and mode, existence, and
visibility siblings remain frozen.

Gradient masking alone is insufficient because AdamW weight decay and retained
moments can move a zero-gradient row. Before every optimizer step, the trainer
zeros excluded gradients and matching optimizer-state rows and snapshots their
parameter values. Immediately after the step it restores excluded parameter
rows exactly and keeps their moments zero. The configured trainable scope is
already exact-resume-bound; an exact resume cannot cross between unrestricted
six-head and lateral-row ownership.

This scope is motivated by the rejected specification-1.52 treatment, whose
step-1024 checkpoint changed only fast-state rows 0 through 5. Its largest head
drift was z position/velocity variance, while fixed validation improved x/y
position and pooled velocity but regressed z position by `7.84%`. The new
profile is otherwise exactly the 1.52 axis-gated CPU protocol: only project
name, scope, and the corresponding zero event-owner key differ.

A bounded dirty-source CPU wiring pair from the immutable initializer completed
two identical balanced eight-scenario updates per arm. Losses were bit-equal at
`0.8310171366` and `1.5899894238`; the unrestricted gradient norms were
`0.0713825151/0.0575892627`, while lateral ownership retained
`0.0654180571/0.0444234833`. The lateral checkpoint changed only rows
`0,1,3,4` of all six head tensors, preserved every excluded row and all other
model tensors exactly, and completed with no skipped or retried update. Its
checkpoint and metrics SHA-256 values are `51fe0089...` and `193ed79c...`; the
matched unrestricted values are `16490c2d...` and `23780a54...`.

This is wiring and optimizer-ownership evidence only. The eight-episode
validation manifest deliberately has only one episode per scenario and is not
a promotion gate. The final repository gate passes `1233` tests with `20`
expected inactive-backend skips in `439.94s`; Ruff, format, version, compile,
and diff checks pass. A clean source-frozen run must re-enter at the common
rich fixed-32 step-zero reference and stop at step 512 unless pooled and every
scenario/axis/horizon uncertainty, event, identity, coverage, and physical
guardrail justify continuation. Deployment remains the protected learned
incumbent.

The clean source-frozen CPU run at
`runs/20260823-065000-spec156-lateral-updater-3072-workers2` therefore stopped
after its step-512 selector. Exact row ownership held at steps 256, 384, and
512: only rows `0,1,3,4` of the six state heads changed, every excluded row and
all other model tensors remained bit-identical, and excluded Adam moments were
zero. Pooled selector score improved `0.23952864 -> 0.23875991`, current
position RMSE improved `0.15056456 -> 0.14956369` m, current velocity RMSE
improved `0.81188211 -> 0.80074560` m/s, and every pooled position horizon
improved. Selection nevertheless failed 34 guardrails: 12 collision, 9
Gaussian-NLL, 5 identity, 4 velocity, and 4 position failures, including
coherent baseline-x uncertainty, glancing-y accuracy/calibration, and
long-horizon event regressions. No support gate failed.

Component rollback on fixed eight shows that mean- or gate-head rollback keeps
the same 17 failures, while variance-head rollback creates 29 failures and
loses the useful state correction. A half-variance interpolation looked better
on eight episodes but failed 44 guardrails on the full fixed-32 manifest,
especially elastic-pairs velocity/identity/uncertainty. This rejects both
freezing and post-hoc scaling of variance. The next repair must add explicit
protected-base scenario/axis/horizon non-regression or equivalent objective
control; it may not weaken guardrails or promote this checkpoint. The numbered
checkpoint SHA-256 is `682d223f...`, its model-state hash is `9d629cd3...`, and
the clean runtime fingerprint is `29e663cd...`. Deployment remains step zero.

A deterministic early-checkpoint rerun from the same initializer reproduces
the original cadence losses exactly through update 128 and closes the
early-stopping hypothesis. Its full fixed-32 score worsens
`0.23952864 -> 0.23972992` and fails 21 guardrails, already including pooled
and sliced collision F1, elastic-pairs y/NLL, heavy-light events/identity, and
reference-pairs event/identity/NLL. Ownership remains exact and finite. The
step-128 checkpoint SHA-256 is `612d1f05...` and model-state hash is
`7c5ccc1c...`. The tradeoff therefore begins during warmup and later rotates
across scenarios; selecting an earlier checkpoint is not a repair.

---

## 251. Scenario-axis-horizon tail-risk objective

Specification 1.57 adds one opt-in training field,
`training.closed_loop_scenario_tail_fraction`. `null` preserves the exact
legacy supported-mean reductions. A finite non-boolean fraction in `(0, 1]`
requires scenario-balanced batches, one row for every declared scenario,
batch-macro physical losses, and axiswise correction hinges. Differentiable
training verifies the canonical declared scenario order before using the
field; validation remains unchanged.

When enabled, current and rollout position, velocity, and Gaussian-NLL losses
are reduced independently by world axis. Each axis first computes one mean per
supported scenario row, then averages the worst `ceil(fraction * supported)`
rows. Correction hinges use the same axis-separated tail reduction. Node and
pair event BCE use a scenario-row tail after deriving the globally bounded
class weight. Unsupported rows are omitted, and the existing fixed configured
horizon denominator remains unchanged. The mechanism changes optimization
only: runtime inference, physical metrics, selector guardrails, support floors,
and deployment semantics are untouched.

The dedicated CPU diagnostic profile uses a tail fraction of `0.25`, two
rollout anchors per window, collision-event weight `0.05`, and uncertainty
weight `0.025`, while retaining the exact specification-1.56 x/y head
ownership. The second anchor is required because a real balanced one-anchor
draw produced exactly zero recursive rollout and event gradient at the owned
heads. On the same eight-scenario draw, the calibrated two-anchor objective
produced finite owned-head gradient norm `0.11139997`, `1.6526x` the legacy
one-anchor norm, with cosine `0.72210806`. Weighted rollout-x, rollout-y,
velocity, NLL, and event norms were `0.06375`, `0.00638`, `0.02232`, `0.04554`,
and `0.04380`; total norm remained far below the global clip. The report is
`/private/tmp/orpheus-spec157-tail-gradient-probe-calibrated-20260823/report.json`
with SHA-256 `959869ad...`.

The clean frozen two-update tail-on/tail-off pair passes its wiring boundary:
the configurations differ only in tail fraction, all `5700` common step-zero
physical/seed/scenario values are exact, both seed batches match, and exactly
the permitted six x/y head tensors own finite Adam state. Treatment gradients
are materially larger but contained. Both two-update candidates are rejected
only because their microscopic score changes do not reach the `1e-5` minimum.
The paired audit is
`/private/tmp/20260823-111142-spec157-tail-two-update-audit.json`, SHA-256
`c57f3f56...`.

The clean step-32 rung clears the pooled threshold: score improves
`0.23952864 -> 0.23950921`, current position and velocity improve slightly,
and the specification-1.56 step-128 failure burden contracts from `21` to `5`.
Exact six-head/x-y-row ownership and full `32/32` seed plus `8/8` scenario
support hold. The remaining failures are pooled and reference-pair 0.75-second
collision F1, reference-pair 0.25-second identity mismatch, and heavy-light
1-second association coverage. One of 32 updates clips and two unsupported
draws retry successfully. The candidate remains rejected.

Continuation is unsafe. Update 48 produces a finite raw gradient norm of
`5230.0088`, globally clipped to `2.0`. Read-only fixed-32 evaluation of that
durable checkpoint remains forward-finite and improves score to `0.23947856`,
but failures expand to `11`, dominated by collision regressions. A 75% in-memory
interpolation of the step-32 heads still clears pooled improvement at
`0.23951368` but preserves all five failures exactly. Neither interpolation,
resumption, nor promotion is allowed. The step-32 and step-48 audit SHAs are
`1757f99f...` and `403e431d...`; deployment remains the protected step-zero
incumbent.

The complete repository gate passes `1249` tests with `20` expected
inactive-backend skips in `454.79s`; Ruff, format, compile, version, and diff
checks pass. Retain the default-null mechanism for research. A successor must
bound per-objective influence and give event/identity semantics direct typed
ownership or an equivalent protected-base constraint; increasing hard-tail
pressure on these state heads is rejected.

---

## 252. Forward-exact robust uncertainty influence

Specification 1.58 adds the optional exact-resume-bound field
`training.closed_loop_uncertainty_standardized_error_gradient_cap`. `null`
preserves the exact legacy Gaussian-NLL operations and gradients. A configured
value must be a finite positive non-boolean number. The NLL forward value,
reported proper score, validation metrics, selector, runtime inference, and
deployment behavior remain unchanged.

Let `x = squared_error * exp(-log_variance)` be the standardized-error term.
At `x <= cap`, backward is unchanged. Above the cap, backward follows the
logarithmic surrogate `cap + cap * log(x / cap)`, while a straight-through
construction retains the exact original `x` in forward. The standardized-
error contribution to the log-variance gradient is therefore bounded by the
cap instead of scaling without limit. This is an optimization-only influence
bound, not clipping of state, variance, loss, metrics, or evidence.

The dedicated diagnostic profile retains the specification-1.57 scenario-tail
fraction `0.25`, two rollout anchors, uncertainty weight `0.025`, and exact x/y
head ownership. It sets the standardized-error gradient cap to `25.0` and
removes event BCE from this state-head scope because the predecessor's event
gradient was material yet collision guardrails worsened. Physical event
dynamics and validation metrics remain active; only event objective ownership
is zero.

An extreme CPU regression proves exact forward equality and bounds the
log-variance gradient at `-12` where the legacy magnitude exceeds one million.
On the immutable initializer's first real balanced eight-scenario batch, the
cap is correctly inactive: current and rollout NLL losses and their owned-head
gradients are exact. Removing event ownership subtracts its weighted gradient,
with finite total owned-head norm `0.11976653`, cosine `0.93073324` to the
predecessor, and no clipping. The report is
`/private/tmp/20260823-spec158-robust-gradient-probe/report.json`, SHA-256
`645dadd1...`. This is implementation and routing evidence only. The final
repository gate passes `1260` tests with `20` expected
inactive-backend skips in `455.53s`; Ruff, format, compile, version, and diff
checks pass. Deployment stays at the protected step-zero incumbent.

The clean frozen two-update capped/uncapped pair passes its wiring boundary.
The resolved configurations differ only at the cap (`25.0` versus `null`), all
`5700` common step-zero physical/seed/scenario values are exact, and both
forward loss sequences are bit-identical. The capped arm retains `99.9158%`
and `93.8261%` of the control gradient norm on updates one and two; all norms
are finite and below clipping. Both arms use the same two ordered balanced seed
batches, change exactly the permitted six x/y head tensors, and create exactly
six optimizer-state entries. Both candidates are rejected only because their
approximately `1.95e-7` score improvements do not reach the `1e-5` minimum.
The paired audit is
`/private/tmp/20260823-spec158-robust-tail-two-update-audit-v2.json`, SHA-256
`227cb77c...`. This proves forward-exact backward influence control and state
ownership, not accuracy.

The bounded accuracy rungs reject every candidate while accepting the
numerical mechanism. A fresh step-32 run applies all 32 updates, retries two
unsupported draws, never clips, and has maximum raw gradient norm `1.630602`;
the specification-1.57 update-48 magnitude `5230.0088` does not recur. Exact
x/y ownership and complete fixed-32 support hold. Selector score improves
`0.23952864 -> 0.23951607`, clearing minimum improvement, but baseline
collision F1 regresses at current and `0.1s`, so the candidate is rejected.
The audit and checkpoint SHA-256 values begin `916af4ab...` and `a207ca97...`.

The fresh step-16 candidate is rejected for insufficient score improvement
and heavy-light `1s` identity association coverage. Interpolation cannot
simultaneously satisfy the selector: alpha `0.5` has zero guardrail failures
but misses minimum improvement, while alpha `0.8` clears minimum improvement
and reproduces both baseline collision failures. These are terminal diagnostic
results for specification 1.58. Do not resume or promote its checkpoints, and
do not continue gate-manifest interpolation. Preserve the protected step-zero
deployment. The next protocol must give collision-event semantics a direct
typed trainable owner or impose an equivalent protected-base constraint while
retaining the forward-exact uncertainty influence cap.

---

## 253. Direct typed collision-event ownership

Specification 1.59 adds the exact-resume-bound closed-loop scope
`updater_state_heads_xy_collision`. Its physical ownership is exactly the
specification-1.58 lateral boundary: rows `0,1,3,4` of the learned corrector's
mean, variance, and gate weight/bias tensors. It additionally exposes the
existing typed attention relation decoder, but AdamW may change only its
collision-logit row `1`. The shared attention projections and transformer,
node decoder, contact, force, impulse, and process-noise relation rows, every
other updater row, perception, identity, and all physical dynamics parameters
remain bit-exact. A positive exact scope-specific event-loss override is
mandatory, and typed attention must be enabled.

The logged forward objective and every runtime/event prediction stay
canonical. Backward ownership is separated explicitly. The ordinary weighted
backward structurally omits event BCE and updates the lateral state heads from
physical objectives only. A separate `autograd.grad` evaluates the exact
weighted event term with respect to the relation decoder alone; only collision
row `1` is accumulated. Recursive event derivatives reaching contact, force,
impulse, or process-noise rows are measured and discarded before parameter
clipping. Excluded decoder gradients and Adam moments are zeroed, excluded
values are snapshotted before AdamW and restored afterward. This is parameter-
specific objective routing, not a detached or altered event forward pass.

The dedicated profile
`configs/direct_collision_owner_updater_xy_repair_cpu.yaml` differs from the
specification-1.58 profile only in project name, the combined typed scope, and
event weight `0.01` in both the scope override and loss map. It retains the
scenario-tail fraction `0.25`, two anchors, uncertainty cap `25`, exact 120-Hz
learned cadence, validation anchor batching, all physical weights, seed/data
protocol, and selector unchanged.

On the immutable initializer's first balanced eight-scenario batch, a weight
of `0.05` would give collision row norm `0.13103` versus physical lateral-head
norm `0.11977`, making the new owner dominant. The calibrated weight `0.01`
gives collision-row norm `0.0262057`, about 21% of the disjoint combined norm,
while the state heads remain exactly at the physical-only norm `0.1197665`.
Unrestricted event BCE would have added `0.0087597` to those state heads; the
new route eliminates it. Recursive noncollision relation gradient norm
`0.0768893` is recorded and discarded. Every value is finite and collision
output clipping is inactive. The report is
`/private/tmp/20260823-spec159-direct-collision-owner-probe-v3/report.json`,
SHA-256 `f6904bd5...`; its script SHA-256 is `2d1ab284...`.

Focused schedule/config/exact-resume/objective checks pass `510` tests with one
expected inactive-MPS skip. The final whole-repository gate passes `1267`
tests with `20` expected unavailable-backend skips in `458.26s`; whole-tree
Ruff, the `224`-file format check, isolated compileall, version, and diff checks
pass. The frozen source is committed and pushed as `2154b68`.

The clean two-update direct-owner/event-zero pair passes on those exact source
bytes. Resolved configurations differ only by project name, typed scope, and
the matching scope/event weights. Both arms have identical ordered draws,
`5700` exact common step-zero physical/seed/scenario values, exact common
non-event losses, and bit-identical final state-head tensors. The control
checkpoint owns exactly the six lateral state-head Adam entries. Treatment
adds exactly relation-decoder weight and bias, and only collision row `1`
changes; every other relation row and model tensor stays exact. Both candidates
improve the selector by only about `1.9445e-7`, retain collision F1 exactly,
and fail only the `1e-5` minimum-improvement requirement. The paired audit is
`/private/tmp/20260823-spec159-direct-collision-two-update-audit.json`, SHA-256
`52589137c99386e126a1dff0b60ec5d0248ba625857d944a7dde51984b204228`;
its script SHA-256 is
`a597885d6ef5934a136e9d0a65db925d1552877a1b61eba33d10626704f51186`.

This is typed ownership, routing, and deterministic state-isolation evidence
only. The sequential pair is not a latency qualification because the control
initialization encountered a transient host slowdown. Fresh bounded fixed-
manifest accuracy, paired latency, promotion, and convergence remain
mandatory. Deployment stays at the protected step-zero incumbent.

Event-objective support is draw-local. A balanced causal draw may have no
supported rollout anchor and therefore no event term. Under the combined
scope, that absence must not fabricate event supervision or abort a valid
physical update: backpropagate the unchanged non-event objective, record
`direct_collision_event_objective_supported=0` with zero routed/discarded
event norms, and leave relation-decoder gradients and moments untouched. If an
event term is present but does not require gradients, fail closed. The first
fresh step-16 attempt exposed this distinction at attempted update 6 and
terminated after five finite unclipped updates; it produced no candidate and
must not be resumed. The repaired focused gate is `322 passed, 1 skipped`; the
final repository gate is `1269 passed, 20 skipped in 444.93s`, with Ruff,
224-file format, compileall, version, and diff checks clean. Accuracy must be
rerun fresh from the immutable initializer.

The fresh bounded accuracy rungs reject the current objective while retaining
the mechanism. Step 16 completes with exact eight-tensor ownership, one sparse
event draw, no clip/retry, and complete fixed-32 support. Its score improvement
is only `5.7653e-6`, and heavy-light 1-second identity association coverage
fails. The audit is
`/private/tmp/20260823-spec159-direct-collision-step16-audit.json`, SHA-256
`03a4b5f014f9696b44966e6714e7325174cdf6118e2a0c6ea9b567b2b0b63cdf`.

Step 32 also preserves exact ownership and complete support, handles sparse
event draws only at updates `6` and `23`, never clips or retries, and reaches
maximum raw norm `1.632861`. Selector score improves
`0.23952864 -> 0.23951438`, clearing minimum improvement. Nevertheless, the
same baseline current and 0.1-second collision-F1 guardrails that rejected
specification 1.58 remain. Collision row 1 changes non-vacuously (weight L2
`1.82117e-4`, bias magnitude `1.75639e-5`), but the resulting decisions trade
one false positive for one true positive becoming false negative; pooled F1
falls `0.266563 -> 0.264132`. The audit is
`/private/tmp/20260823-spec159-direct-collision-step32-audit.json`, SHA-256
`f19fe665174342c8a1594e647b065e1758012c6cc4a7edbba5144aa7a4173def`;
checkpoint SHA-256 is
`4f4a51f80c8123185da1cbdc543d77f27ebac59324685a149d5ab0c490af47dc`.

These are terminal accuracy results for the current direct-owner objective.
Do not resume, interpolate, or promote either checkpoint. Retain the typed
row-routing mechanism and protected step-zero deployment. Before another
training rung, separate node-level collision supervision (the selector-owned
decision) from pair-level event evidence and measure their gradient agreement;
increasing weight or duration without that attribution is not authorized.

---

## 254. Node-only typed collision ownership

Specification 1.60 separates the collision signal that owns the selector's
node-level decision from auxiliary pair-event evidence. The canonical forward
event objective remains unchanged: where pair support exists it is still the
historical half-sum of node and pair balanced BCE, and all logged losses,
runtime predictions, analytic events, physical metrics, and selector semantics
remain canonical. Training now also retains the independently aggregated node
and pair sufficient loss tensors as non-double-counted routing evidence.

The exact-resume-bound scope `updater_state_heads_xy_collision_node` retains
the specification-1.59 tensor boundary: rows `0,1,3,4` of the corrector mean,
variance, and gate weight/bias tensors plus only collision row `1` of the
typed relation decoder. Physical objectives update only the six state-head
tensors. The separate collision-owner `autograd.grad` consumes only the
node-event tensor; pair-event gradients and every noncollision relation row
are discarded. Missing node-event support remains a valid sparse physical
update with zero event-owner gradient, while a present detached node-event
term fails closed. Typed attention and a positive exact scope override remain
mandatory.

The first immutable balanced eight-scenario draw proves that the previous
combined route was directionally conflicted rather than simply too weak.
At event weight `0.01`, collision-row node and pair gradient norms are
`0.05904064` and `0.00681224`, with cosine `-0.97618997`; their canonical
combined norm is `0.02620572`. Node BCE is `15.51175976`, pair BCE is
`0.05387612`, and the canonical combined BCE is `7.78281879`. The `9.54e-7`
difference from an independently aggregated half-sum is ordinary float32
reduction order; no forward operation is replaced. The probe report is
`/private/tmp/20260823-spec160-node-pair-gradient-probe-v2/report.json`,
SHA-256 `ec560d9b330a839c11b4f7c539a65380239d8c7a78aa21aecfe41cb53600b9d8`;
its script SHA-256 is
`c2075057a4463f34d97cbe6a2ebe60c0adc9604b0e368580a8591d5effedeec2`.

The dedicated profile
`configs/node_collision_owner_updater_xy_repair_cpu.yaml` changes the
specification-1.59 profile only in project name, typed scope, and event weight.
Its node-only weight is `0.0045`, yielding a predicted first-draw collision-row
norm `0.02656829`, within about `1.4%` of the rejected combined route's
`0.02620572` while removing the opposing pair direction. This is calibrated
routing, not an accuracy claim. Focused objective/schedule/config/exact-resume
coverage passes `518 tests` with one expected inactive-MPS skip. The complete
repository passes `1275 tests` with `20` expected unavailable-backend skips in
`459.68s`; whole-tree Ruff, the `224`-file format check, isolated compileall,
version, and diff checks pass. A clean paired two-update ownership gate and
fresh bounded fixed-manifest accuracy remain mandatory before any continuation
or promotion. Deployment remains the protected step-zero incumbent.

The clean node-only/combined two-update pair passes on commit `2e0fabc`.
Resolved configs differ only in project name, typed scope, and their matched
scope/event weights. All `5700` common step-zero physical/seed/scenario values
and the first-draw node, pair, combined, and logged event losses are exact.
Both arms change exactly the same six lateral state heads plus collision row
`1`, own exactly eight Adam entries, and leave every other model tensor exact.
The state-head tensors are bit-exact across arms. Collision-row delta norms are
`1.26343e-6` node-only and `1.26077e-6` combined, with cosine `0.99992348`.
Both candidates have the same `1.94448e-7` score improvement and unchanged
collision F1, so both are rejected only by minimum improvement. The audit is
`/private/tmp/20260823-spec160-node-combined-two-update-audit.json`, SHA-256
`e5db2d9bcf2d5d4802e3097ef0c0c8f45fd9bf46c84ba5a76df0e649de37c463`;
script SHA-256 is
`cd3a74f94fd754428ae07d88883dd561a137b999932cf0ad21f3811ed3cf77d5`.
This is ownership evidence, not latency or accuracy qualification.

The fresh bounded accuracy rungs reject the node-only objective. Step 16
applies all updates with one sparse event draw, zero retry/clip, maximum raw
norm `0.419702`, exact eight-tensor ownership, and complete fixed-32 support.
Its score improvement is only `5.76530e-6`, and heavy-light 1-second identity
association coverage fails. The audit SHA-256 is
`51878e7815d767a0be0605098f44151c477695d2a8d84917e449c9cc34a59300`.

Step 32 also preserves exact ownership and support, handles sparse draws only
at updates `6` and `23`, never clips/retries, and has maximum raw norm
`1.631959`. Selector improvement `1.42553e-5` clears minimum, but baseline
collision F1 fails at aggregate/current and `0.1s`; pooled F1 falls
`0.266563 -> 0.264132`. Only collision row `1` changes beyond the six state
heads (weight L2 `1.84176e-4`, bias `1.78469e-5`). Its audit is
`/private/tmp/20260823-spec160-node-collision-step32-audit.json`, SHA-256
`632f39e6b7737273f532b79fbaa5498c510c05e994f4904201310e82b2a4f9ff`;
checkpoint SHA-256 is
`3dd8b914f7f3abfd2ce5c0046a1c6feac3240e413e45aa9f672ab5fe51bd6cdd`.

All `5829` common final physical, seed, scenario, and validation fields are
bit-exact between the specification-1.59 combined step-32 candidate and this
node-only candidate. Removing pair BCE at matched row magnitude therefore
does not alter any selector-visible decision in this bounded regime. Do not
continue, interpolate, or promote either specification-1.60 checkpoint.
Retain the routing instrumentation and protected step-zero deployment. Before
another training rung, isolate the node-event collision-row gradients by
horizon and determine whether aggregate/long-horizon supervision opposes the
failed `0.1s` baseline slice; do not increase duration or weight blindly.

---

## 255. Protected node-event routing into lateral state heads

Specification 1.61 addresses the actual owner of the specification-1.60
baseline collision regression. A fixed four-episode component screen proves
that the learned relation collision row is selector-inert: relation-only is
bit-exact to the protected initializer, while the six state heads alone
reproduce the complete step-32 baseline F1 failure. Mean-only, variance-only,
and gate-only retain the initializer F1. The failure appears only when learned
mean and learned variance are combined; gate and relation changes are not
required. The report is
`/private/tmp/20260823-spec161-baseline-component-screen-v3/report.json`,
SHA-256 `dcd10024572776e11a385d2aa8750470358ecfbbd2ac589c586fea317f4e1a9d`.

The new exact-resume-bound mapping
`training.closed_loop_state_event_loss_weights` is empty by default and may
contain only `updater_state_heads_xy_collision_node`. A positive value requires
that scope and its existing positive relation-owner event override. It does
not change the canonical forward event objective, logged total loss, runtime
predictions, analytic events, metrics, selector, or relation-row weight.
Instead, the same independently retained node-event tensor is differentiated
again with the configured state weight and accumulated only into rows
`0,1,3,4` of the mean, variance, and gate weight/bias tensors. Every excluded
state row is measured and discarded; the existing AdamW row restoration keeps
those values and moments exact. Missing node-event support remains a valid
physical-only update with explicit zero routing evidence.

The first balanced attribution shows why the protected route is needed. The
physical lateral-head gradient norm is `0.11976653` and its cosine to baseline
node-event descent is `-0.38539332`. Tail node-event gradient aligns with the
baseline at cosine `0.98805958`; its first-order break-even weight is
`0.03368828`. Equal-scenario and unit-normalized scenario routes require about
`0.0832` at their smaller norm and do not improve directional coverage. The
dedicated profile
`configs/protected_state_event_updater_xy_repair_cpu.yaml` therefore retains
the relation weight `0.0045` and adds a bounded state route `0.04`. Its real
applied proof gives physical, protected-event, combined applied-head, and
relation-row norms `0.11976653`, `0.05546733`, `0.11233944`, and `0.02656829`.
Excluded state-event norm is `0.00595449`; all values are finite and below
clipping. Attribution/proof report SHAs are `85eadc11...` and `a7472420...`.

Focused config, exact-resume, schedule, objective, and checkpoint coverage
passes `528` tests with one expected unavailable-MPS skip; whole-tree Ruff and
the `224`-file format check pass. The final repository gate passes `1285`
tests with `20` expected unavailable-backend skips in `496.20s`; Ruff, format,
isolated compile, version, and diff checks pass. The implementation is frozen,
committed, and pushed at `72e31a7`.

The clean route-on/zero-weight pair passes on that commit. Its configs differ
only at state-event weight `0.04` versus `0.0`; all `5700` common step-zero
physical/seed/scenario fields and `38` first-draw forward loss/event fields are
bit-exact. Both arms change exactly the six permitted lateral head tensors plus
collision relation weight/bias, own exactly those eight Adam states, and leave
all other model state exact. Treatment state-event gradient norms are
`0.0378016/0.0109499`; control norms are exactly zero. Audit SHA-256 is
`27dcc456572233f8046ec618b420435ec6343d2c7a3380bbf902a9db6b19d217`.
An earlier sandbox interruption and an empty-map override that merged instead
of clearing the route are preserved and explicitly excluded from the pair.

Fresh bounded accuracy is terminal and rejected. Step 16 improves selector by
only `6.32059e-6`, below the required `1e-5`, and also fails heavy-light
one-second identity association coverage (`1eb9225d...`). Step 32 clears the
minimum with `1.51940e-5` improvement and slightly improves pooled position and
velocity, but fails five guardrails: pooled `0.75s` collision F1, heavy-light
`1.0s` association coverage, reference-pairs current and `0.75s` collision F1,
and reference-pairs `0.25s` identity mismatch (`38d0de6e...`). Step 32 changes
only the same eight tensors/rows, has eight finite Adam owners at step 32, two
sparse-event draws (`6/23`), no retry or clip, and maximum norm `1.564329`.
Variable wall time is not paired latency evidence. Do not run step 64, tune the
weight further, resume, compose, or promote this objective. Deployment remains
the protected step-zero initializer; comprehensive promotion stays false.

---

## 256. Frozen-reference scenario-axis-horizon non-regression

Specification 1.62 adds an opt-in causal training guard after repeated
specification-1.56 through 1.61 candidates improved pooled accuracy while
rotating regressions across scenarios, axes, horizons, events, and identity.
`training.closed_loop_protected_reference_nonregression_weight` is a finite
nonnegative scalar and defaults to zero, which preserves the exact legacy
forward/backward path. A positive value is restricted to causal-only,
one-row-per-scenario balanced batches with batch-macro physical losses,
axiswise correction hinges, and zero attention dropout.

The trainer loads a separate frozen model from the tensor-verified step-zero
`reference_rollout.pt`. Construction, loading, freezing, and evaluation setup
of that optional model must preserve and restore the candidate's global RNG
state; enabling the guard may not alter the first candidate forward before a
nonzero hinge exists. For every candidate update it executes the candidate
once, restores the pre-forward Python/Torch/backend RNG state, replays the
reference under `no_grad` on the exact same batch, window, and perturbation
stream, then restores the candidate's post-forward RNG state. The runtime
continues to consume only RGB; simulator state appears only in supervised
error cells. The reference is not an alternate belief or inference branch.

Current position/velocity and every configured forecast horizon produce
additive Smooth-L1 cells separately for each world axis and scenario row.
Node-event logits additionally produce per-row/horizon BCE cells. Candidate
and reference schemas and support counts must match exactly; a candidate may
not satisfy the hinge by dropping tracks or horizons. The optimized term is
the mean positive candidate-minus-reference error across supported cells.
Direct event routing preserves this explicit weight when reconstructing its
non-event backward pass. Exact resume binds the field with legacy default
zero.

Focused config, objective, schedule, and checkpoint coverage passes `562`
tests with one expected unavailable-MPS skip. A production-shaped CPU probe
from the immutable initializer preserves all `328` supported cells. Exact
equality gives zero loss and zero regressed cells; a `1 cm` allowed mean-head
perturbation activates `136` cells with protected/base gradient-norm ratio
`0.111032`. The dedicated profile therefore uses the single provisional
weight `1.0`. The repaired two-update pair proves exact candidate-forward
parity, RNG continuation, and eight-tensor ownership (`53873f5d...`). The
only authorized step-32 rung applies all updates with zero skips and activates
the hinge on 29 updates, but it retains all five known spec-1.61 event/identity
guardrail failures at bit-identical values (`3aff0403...`). The candidate is
rejected, deployment remains step zero, and this objective is terminal: do not
retune its weight, extend its duration, or compose its checkpoint. The final
repository passes `1295` tests with `20` expected unavailable-backend skips in
`447.03s`; Ruff, the `225`-file format check, isolated compile, version, and
diff gates pass.

---

## 257. Runtime-local observation-fitted transition candidate

Specification 1.63 permits one bounded runtime hypothesis that is genuinely
different from the rejected fixed-coefficient and output-residual ladders. An
optional online constant-acceleration candidate fits a weighted running
acceleration independently for each persistent entity and world axis from the
existing causal RGB temporal-velocity measurement, its validity mask, and its
reported uncertainty. It owns no trainable parameter, persistent
`WorldBelief` field, checkpoint tensor, simulator input, or oracle state.

The candidate is disabled by default. Its three strict runtime semantics are
`hypothesis_online_acceleration_enabled=false`,
`hypothesis_online_acceleration_minimum_support_count=4`, and
`hypothesis_online_acceleration_maximum_mps2=20.0`. Enabling it requires RGB
temporal velocity and entity/axis/regime-local applicability. These fields are
exact-resume bound when the runtime pool is enabled and normalize only under
the complete disabled-policy legacy boundary. The evaluator policy version is
unchanged for the disabled four-candidate pool and advances only for the
enabled five-candidate intervention.

Evidence is associated by the normal persistent-ID RGB measurement mapping.
The fit is updated only after due forecasts have been scored, so a candidate
may not receive credit for a trajectory issued before it had sufficient
support. Identity reuse, inactivity, occlusion, contact, and every non-free
motion mode erase the local fit. Each acceleration observation is uncertainty
weighted and clipped to the configured physical bound. Before the configured
support count is reached, and for every unsupported entity/axis/regime, the
emitted forecast remains the exact learned candidate. Local fit state is
cleared at episode reset and is never serialized.

No threshold or duration ladder is authorized. Qualification consists of the
implementation/fallback gates followed by one source-frozen, sequential
learned-only versus enabled fixed-32 CPU comparison on the protected checkpoint
and standard RGB-only manifest. The candidate must be selected non-vacuously,
improve summed pooled position RMSE by at least `1e-5` metres, and pass every
existing pooled and scenario/horizon position, velocity, uncertainty, event,
identity, coverage, and support guardrail. Failure stops this hypothesis before
MPS. CPU success authorizes exactly one matched active-Aqua MPS confirmation;
it does not itself promote a default. This stopping boundary prevents marginal
accuracy from restarting the exhausted coefficient, interpolation, or residual
families.

The frozen implementation gate passes `437` affected tests with `3` expected
backend skips. The first candidate launch failed before episode one because
evaluator accounting still allocated four candidate slots; the failed `0/32`
artifact is not accuracy evidence. Candidate names and count partitions now
derive from the resolved policy, a one-episode five-candidate CPU report
completes finite, and the final repository passes `1309` tests with `20`
expected backend skips in `456.84s`. Ruff, the `225`-file format check,
isolated compile, specification version, and diff checks pass.

The sole repaired fixed-32 CPU pair on clean commit `f60aac7` is the terminal
accuracy decision. The enabled fifth candidate has zero selections at current
state and every configured horizon, failing the required non-vacuity gate.
Summed pooled position RMSE worsens `0.000107280209 m`; each of the five
horizons regresses. The paired comparison reports `246` physical guardrail
failures and latency ratios `1.24317/1.23565/1.73110` for global, fast, and
rollout execution. A damped-contact one-second event slice also has no positive
reference or candidate class support. These are independent failures; none is
treated as promotion evidence. The comparison report is
`/private/tmp/20260824-spec163-online-acceleration-fixed32-cpu-comparison.json`
(SHA-256 `ebeb6b63c4fcc990abd6ee56406c8053188d89409faef6d6e54d2a617111ad39`),
with source reports `ca20f6bf...` and `9fe1e4d6...`.

Therefore no MPS confirmation is run and no nearby configuration is tested.
The candidate remains default off, the protected step-zero deployment remains
unchanged, and this hypothesis is closed under the declared diminishing-
returns rule.

---

## 258. Intervention-specific runtime-candidate evidence

Every runtime-pool promotion comparison must derive its candidate names from
the exact configured policy rather than a fixed historical tuple. Selection,
composition, and per-horizon partitions must cover every configured candidate.
The comparison protocol records the complete ordered candidate list and the
subset newly introduced beyond the canonical learned, constant-velocity,
damped-velocity, and ballistic-contact pool.

Aggregate non-learned usage is insufficient evidence for a newly introduced
candidate. Each configured extension must have positive direct selection or
positive composed-step use on at least one configured intervention axis. Its
aggregate and per-horizon counts are retained separately. Missing metrics,
invalid partitions, duplicate/unknown required names, or zero direct and
composed use fail the physical promotion decision. Causal residual use remains
an independent valid intervention for profiles that explicitly configure no
new candidate.

This rule closes a concrete specification-1.63 evidence defect. The original
comparison accepted the runtime-usage sub-gate because older ballistic,
constant-velocity, and damped-velocity candidates were selected `1685` times,
although `online_local_acceleration` was selected for `0` final queries and
`0` composed steps at every horizon. Re-evaluating the immutable reports with
the candidate-specific rule adds the explicit usage failure, increasing the
physical failure count from `246` to `247`; the terminal rejection is
unchanged. The paired comparison schema advances from v2 to v3, and the
four-split aggregate verifier accepts only v3 comparisons.

Focused promotion, evaluator, rollout, config, and recovery coverage passes
`352` tests with `3` expected unavailable-MPS skips. The final repository
passes `1312` tests with `20` expected unavailable-backend skips in `451.96s`;
Ruff, the `225`-file format check, isolated compile, version, and diff checks
pass. This is evidence-path hardening only: runtime predictions,
`WorldBelief`, checkpoint tensors, the default-off candidate, and the protected
deployment are unchanged. It does not authorize another acceleration-candidate
run.

---

## 259. Event-frame-targeted ensured-pair training data

Specification 1.65 introduces default-off simulator controls for genuinely
new event-rich training data after existing collision objectives proved sparse
and unsupported at the longest horizons. The optional inclusive
`simulator.ensured_pair_event_frame_range` constrains the isolated ensured-pair
collision frame. The optional
`simulator.ensured_pair_vertical_speed_range` supplies a physically observable
upward launch so a delayed pair can remain above the floor under normal
gravity. Both fields are null by default.

Null must preserve the historical generator bit-for-bit: the pair keeps its
fixed `0.15 m/s` upward velocity, no additional random number is consumed, and
the pair is sampled exactly once. When an event-frame range is enabled, the
simulator may rejection-sample pair height, lateral offset, surface gap,
horizontal speed, and the explicitly configured vertical speed up to the
existing bounded scene-attempt count. The real high-rate solver determines the
accepted event frame and floor-clearance requirement. Exhaustion fails closed;
it may not silently emit an out-of-range or floor-confounded episode.

Both fields are strict resolved configuration and exact-resume semantics.
Missing historical checkpoint fields migrate only to null. Event-frame bounds
must be positive integers, ordered, and leave the configured clearance frames
inside the episode. Vertical-speed bounds must be finite, nonnegative, and
ordered. Simulator truth may select training examples and labels only; runtime
inference remains RGB-only and receives no target frame, velocity, or event
schedule.

Implementation alone does not authorize training. The next gate is one
balanced, immutable gradient probe on event-rich data. Every configured event
horizon must have material support and its collision-owner gradient must be
non-opposed to the aggregate. Failure closes this data intervention without a
training campaign. Success authorizes one bounded training/validation rung,
still subject to the complete pooled and scenario/axis/horizon physical,
event, uncertainty, identity, support, and latency gates.

The training result retains the already-computed differentiable node-event
tensor for each configured horizon as audit support. This does not add a loss,
change weighting, or alter predictions; it prevents a gradient gate from being
reconstructed from detached scalar logs. Pair and non-event horizon tensors
remain excluded from this audit channel.

Legacy cross-source generation at seed `17800` is bit-identical to commit
`c58f881` across every RGB, state, event, camera, label, and metadata field;
the canonical hash map is `5c630359...`. Enabled late-event generation is
deterministic and lands inside the requested frame range. Config, simulator,
and checkpoint compatibility coverage passes `379` tests with one expected
unavailable-MPS skip. The final repository passes `1325` tests with `20`
expected unavailable-backend skips in `465.25s`; Ruff, the `225`-file format
check, isolated compile, version, and diff gates pass. These are mechanism
gates, not accuracy evidence; the protected deployment remains unchanged.

The required training gate is terminally negative. One final balanced
feasibility profile used exactly two objects, surface gap `[1.8,3.4] m`,
vertical launch `[3.8,4.2] m/s`, target frames 20--22, and at most 256 real
solver attempts. Its eight ordered scenario seeds produced ensured-pair
impacts at frames `21/22/22/20/20/21/22/20`, proving that the generator can
place late physical events without floor confounding (`308d3cbf...`). The
subsequent immutable five-horizon gradient probe nevertheless failed closed at
the 1.00-second horizon because no supported
`event_collision_node@1.000s` tensor existed. The terminal report is
`/private/tmp/20260824-spec165-event-rich-horizon-gradient-terminal.json`
(SHA-256 `c4195944...`). Under the predeclared rule this rejects training before
any optimizer or MPS rung. Do not tune data ranges, sampling, event weights,
ownership, or duration on this evidence; specification 1.65 remains useful
default-off data capacity only.

---

## 260. Detector-only RGB multi-instance discovery repair

Specification 1.66 adds an exact-resume-bound RGB measurement-pretraining
ownership control after the bounded structured-split diagnostic localized a
real overlapping-disc discovery limitation. The historical value `all`
retains every measurement-pretraining owner exactly. The new
`global_detector` value freezes the shared RGB backbone, fast projection, ROI
updater, filter, identifier, dynamics, and every non-global module while
leaving the existing global detector trainable. This changes no runtime
forward, proposal representation, structured fallback, lifecycle threshold,
or deployed checkpoint.

The field is strict: only `all` and `global_detector` are valid. Historical
checkpoints missing it migrate only to `all`; every present value is bound by
exact resume. A positive aggregate measurement loss weight is mandatory when
the detector-only measurement-pretraining phase is nonempty, so a zero-gradient
protocol fails during configuration rather than after validation. A real RGB
measurement-loss backward and AdamW step must show a
finite, nonzero gradient owner set wholly inside
`observation_modules.rgb.global_detector`, with the changed tensors and Adam
state exactly equal to that owner set. Synthetic gradient assignment alone is
not sufficient ownership evidence.

Accuracy qualification is deliberately bounded. Use the immutable protected
step-zero initializer, RGB-only runtime, exactly three objects, and one ordered
episode from each of the eight declared scenarios per optimizer update. Run at
most one 128-update detector-only measurement-pretraining rung. Evaluate raw
learned query centres on a frozen eight-seed balanced manifest using a
one-to-one assignment against visible projected targets. Continue beyond this
rung only if top-target-count recall within `0.1` normalized image coordinates
improves by at least ten percentage points, confidence-threshold proposal
precision at the same distance does not fall by more than five percentage
points, and structured/runtime physical evidence has no broad regression. Otherwise
close this family without changing fallback admission, lifecycle confidence,
split thresholds, backbone ownership, duration, or loss weights. A passing
raw-query gate authorizes one fixed-32 physical comparison and then, only if
all existing guardrails pass, one active-Aqua confirmation. It is not itself
promotion evidence.

The implementation gate passes the focused config, schedule, RGB-supervision,
and checkpoint suites (`531 passed, 1` expected unavailable-MPS skip). The
real backward/optimizer regression passes independently. The final repository
passes `1331` tests with `20` expected unavailable-backend skips in `462.32s`;
Ruff, the `225`-file format check, version, and diff gates also pass. The
bounded gate is terminally negative. On clean commit `02ccf8b`, the sole CPU
rung applied all `128` balanced updates with no skipped batch, maximum finite
unclipped gradient norm `0.772205`, zero interaction gradient, and exactly `41`
changed/Adam-owned tensors, all inside the global detector. The fixed-32
measurement selector rejected it: score `1.2175323167` versus protected
`1.2169759717`.

The independent fixed-eight raw-query gate fails more strongly. Top-target-
count recall within `0.1` normalized image coordinates falls from `56/192`
(`29.17%`) to `38/192` (`19.79%`), missing the required improvement by a wide
margin. Confidence-threshold precision falls from `44/143` (`30.77%`) to
`53/512` (`10.35%`): training makes every one of the eight queries confident
without localizing enough of them. Therefore no fixed-32 physical comparison,
MPS confirmation, duration extension, learning-rate/loss-weight adjustment,
or admission change is authorized. The mechanism remains available, the
candidate is retained only as negative evidence, and deployment remains the
protected step-zero model. The terminal report is
`/private/tmp/20260824-spec166-detector-only-terminal.json` (SHA-256
`acd9a53b...`).

---

## 261. Raw learned-existence supervision boundary

Specification 1.67 separates learned global-detector classification from
structured runtime confidence without changing physical inference. The global
RGB module must retain the detector head's pre-confidence, pre-structured raw
existence logits as ephemeral auxiliary evidence. Structured component
confidence, unsupported-query suppression, packet confidence, association,
lifecycle, and the public `MeasurementSet.existence_logits` remain the runtime
authority exactly as before.

Global measurement supervision must use the raw learned logits for binary
positive/negative query classification whenever that auxiliary field is
present. It must reject non-tensor or shape-incompatible raw evidence. Target
assignment continues to use runtime measurement values and runtime existence
confidence; this repair is not permission to change Hungarian assignment or
runtime admission. Fast ROI semantics remain unchanged because their
existence logits are not replaced by the global structured-component path.

The repair is motivated by a concrete straight-through loss defect. Evaluating
BCE after substituting approximately `0.995` confidence for supported
components and `0.0001` for unsupported queries preserves a derivative with
respect to the learned logit, but evaluates that derivative at the substituted
forward value. On the frozen eight-seed, eight-anchor three-object manifest,
the protected initializer's mean absolute learned-logit residual is `0.418034`
while the substituted path exposes only `0.029132`. After the rejected
specification-1.66 rung, raw negative confidence is `0.935359`; corrected BCE
therefore supplies mean negative residual `0.935359` rather than `0.006299`.
The immutable diagnostic is
`/private/tmp/20260824-spec167-existence-gradient-probe.json` (SHA-256
`1cc459c9771f26a40af89051dae0eb95adeabde9c53578f3f074678a843d2f4a`).

This correctness repair authorizes one and only one fresh 128-update
detector-only rerun from the same protected initializer. It must preserve the
specification-1.66 three-object balanced data, optimizer, ownership boundary,
and fixed-eight gate: at least ten percentage points of top-target-count recall
improvement within `0.1` normalized image coordinates, no more than five
percentage points confidence-threshold precision regression, and no broad
structured/runtime regression. It does not authorize duration, learning-rate,
loss-weight, threshold, confidence, split, or admission tuning. A second miss
closes this detector family permanently before fixed-32 physical or MPS work.
The implementation gate passes the complete repository (`1334 passed, 20`
expected unavailable-backend skips in `468.79s`) together with Ruff, the
225-file format check, the version contract, and diff validation.

The authorized rerun is complete and terminally negative. Clean pushed commit
`6eeaa85` applied all `128` balanced detector-only updates with zero retry,
maximum finite unclipped gradient `0.804476`, zero interaction gradient, and
exactly `41` changed plus `41` Adam-owned tensors confined to the detector. The
fixed-32 measurement selector rejected candidate score `1.2175323715` against
reference `1.2169759717`. Corrected BCE materially repairs confidence: fixed-
eight confident proposals fall from `143` to `81` and precision rises from
`30.77%` to `37.04%`. It does not repair localization: top-target-count recall
falls from `56/192` (`29.17%`) to `54/192` (`28.13%`), far below the required
ten-point improvement. Therefore the detector-only family is permanently
closed. No physical fixed-32, MPS, duration, LR, loss-weight, threshold,
confidence, split, or admission follow-up is permitted. A future attempt must
use a genuinely different multi-instance discovery architecture. Terminal
evidence is `/private/tmp/20260824-spec167-raw-existence-detector-terminal.json`
(SHA-256 `7c8cf4985becf2fef1891d31a989f70c994d27cdde6c9d8b6f3e987bf0f8a7a7`).

---

## 262. Opt-in dense multi-instance global discovery

Specification 1.68 introduces one opt-in dense local-maximum global detector
after the query-detector family reached its terminal accuracy boundary. The
mode is disabled by default. Disabled models must contain no dense-detector
state and must preserve historical query-detector construction, checkpoint,
runtime, and training behavior exactly.

The dense center branch is fixed by the accepted feasibility probe: one
`3x3` convolution from the configured RGB feature width to 64 channels,
eight-group normalization, SiLU, and one `1x1` center-logit convolution. This
branch has exactly `55,553` parameters for the grounded 96-channel backbone.
It uses deterministic local-maximum top-query decoding and the unchanged
CenterNet-style focal heatmap objective with two-pixel Gaussian labels. The
qualified center branch must not acquire a second top-query BCE gradient or
attribute-loss gradient. A separate `1x1` typed attribute head consumes
detached center-trunk features and emits radius, inverse-depth residual,
colour, visibility, seven measurement log variances, and appearance. Runtime
proposals continue through the existing `GlobalDetectorOutput`, projection,
structured RGB override, association, lifecycle, and `MeasurementSet`
contracts. Generated projected centers are training labels only; runtime
inference consumes RGB features and calibration, never simulator truth.

`model.rgb.dense_global_detector_enabled` is a strict boolean, defaults false,
and is an exact-resume semantic. The matching
`training.rgb_pretrain_trainable_scope=dense_global_detector` freezes every
other parameter, including the historical query detector and shared RGB
backbone. Weight-only initialization may grow the complete
`observation_modules.rgb.dense_global_detector.` prefix only when the stored
model has the explicit legacy-false semantic, the target enables it, and all
other model semantics match. Partial prefix growth, exact resume across the
mode boundary, and undeclared model changes fail closed. Run metadata records
the deterministic module-growth prefix.

The off-repo feasibility evidence is fixed and non-promotional: `181/192`
(`94.27%`) top-count recall and `181/181` confident precision on the eight-seed,
eight-anchor three-object manifest, versus protected query-detector evidence
`56/192` (`29.17%`) and `44/143` (`30.77%`). The exact report is
`/private/tmp/20260824-spec168-dense-center-feasibility-exact-gate.json`
(SHA-256 `f7587471fa4810fbf315aa8c47bffdad8b064bd9952899ba69f6bf91bd4bcb06`).
This result authorizes implementation, not deployment.

Production qualification remains deliberately sequential. First pass focused
typed-output, focal-gradient, exact-owner, legacy-default, and checkpoint-
growth tests plus the complete repository gate. That implementation gate is
green at `1341 passed`, `20` expected unavailable-backend skips in `469.30s`,
together with Ruff, format, compile, version, and diff validation. Then perform exactly one clean
fixed-eight production-path repetition with the frozen dense center weights.
It must retain at least a ten-percentage-point recall improvement over the
protected query baseline, lose no more than five precision points, preserve
all eight scenario support, and keep the non-dense production state bit-exact.
Failure closes this fixed architecture without LR, duration, width, loss,
threshold, NMS, or admission tuning. Success alone authorizes a fixed-32
physical gate and then active-Aqua MPS evidence; it is not itself promotion.

The clean production-path fixed-eight repetition passes exactly: `181/192`
top-count and confidence-threshold true positives, `94.27%` recall, and
`181/181` precision, with all eight scenarios supported and every non-dense
model tensor bit-exact. The report is
`/private/tmp/20260824-spec168-dense-production-fixed8.json` (SHA-256
`2da16c96f5294e3bbc69e63220b663e5854bee90ca19351bdc86c8ac1a457b7c`)
and its temporary weight-only candidate is `3d1a3e09...`.

The authorized fixed-32 CPU/RGB-only physical comparison is terminally mixed
and therefore fails. Against the same seeds `100000..100031`, the dense
candidate improves every pooled forecast-position horizon by `0.28%..0.98%`
and current velocity by `2.41%`, but worsens current position by `1.74%`, driven
by a `6.04%` current-z regression. Current Gaussian NLL and calibration error
worsen, current detection precision falls `0.50%`, distance-gated identity-
switch rate doubles from `0.000563` to `0.001130`, and 14 scenario-horizon
position cells regress. Only 55 of 84 declared core comparison cells improve
or tie. The physical comparison is
`/private/tmp/20260824-spec168-dense-physical-comparison.json` (SHA-256
`b85385a6693945013aee35649375e0335120126b173fc9cd2f5ea93b9f45a320`).
The opt-in/default-off implementation remains as validated research capacity,
but this frozen architecture is not promoted or tuned. Active-Aqua MPS,
deployment, and additional training are not authorized from this result.

---

## 263. Causal observation-model pooling requires independent evidence

An observation-model pool must not admit a broadly rejected detector merely
because it emits more confident proposals. Candidate choice must be causal,
derived from prediction-versus-measurement evidence, and preserve protected
fallback when no candidate is strictly better. Simulator identity, target
count, future labels, and evaluation truth are forbidden runtime selectors.
`WorldBelief` remains the sole physical state; detached candidate measurements
may be compared, but an unselected candidate must not update filtering,
lifecycle, identity, dynamics, or persistent caches.

The first structured/dense pool diagnostic is terminally rejected and leaves
no production semantic. A protected fixed-eight audit contains only `14/944`
ambiguous pairs. In a complete temporary two-candidate path, dense has greater
associated-belief support on zero of 112 decision rows and lower equal-support
association cost on zero; the protected detector is selected everywhere.
All 6,839 common non-latency numeric evaluator metrics remain exact. Evidence
is `/private/tmp/20260824-spec168-association-ambiguity-fixed8-final.json`
(`e3fd7b6e...`) and
`/private/tmp/20260824-spec169-measurement-pool-fixed8-v3.json`
(`33f0e12e...`). The temporary implementation is fully reverted. Do not tune
association margins, score weights, or dense admission on this manifest. The
single separately bounded typed-attribute completion in section 264 is the
final diagnostic for this architecture; after its rejection, no further dense
attribute work is authorized. Reopen only when a materially different
observation candidate first proves strictly better causal evidence under a
predeclared independent gate.

---

## 264. Dense typed-attribute completion is a terminal diagnostic

The dense center branch's raw localization success did not establish that its
untrained typed attribute head could support physical filtering. Exactly one
bounded diagnostic may train only
`dense_global_detector.attribute_head.{weight,bias}` for 128 balanced CPU
updates from the frozen dense production candidate. The center trunk/head and
every other model tensor must remain exact. No learning-rate, duration,
objective, threshold, NMS, association, or admission sweep is permitted.

That diagnostic is complete and rejected. Training is finite and exactly
owned, but the protected fixed-eight CPU/RGB comparison passes only `62/84`
core non-regression cells. Twenty-two core metrics regress, including five of
six current/forecast position-NLL cells and current calibration error
(`0.0261084 -> 0.0290640`), and 21 scenario-horizon position cells regress.
The small gains in current position (`0.1208446 -> 0.1205859 m`), current z
(`0.1355318 -> 0.1352241 m`), velocity
(`0.7582763 -> 0.7582159 m/s`), and current NLL
(`-0.5992216 -> -0.6094756`) are insufficient.

The training report is
`/private/tmp/20260824-spec168-dense-attribute-feasibility.json` (SHA-256
`70c65cb3859989be1885fb33c1e75512f9823fb6b745d8245ef3a5c599a9f765`),
the valid candidate checkpoint is
`/private/tmp/20260824-spec168-dense-attribute-candidate-v2.pt` (SHA-256
`f0c03690318cedbe45e1b4c8a11beed9e7bdb660c06d985a354f53d1f85dab9a`),
and the comparison is
`/private/tmp/20260824-spec168-dense-attribute-fixed8-comparison.json`
(SHA-256
`7a2623374406ffbc8a88447c8974cf5387bdc5f7a80186830b1d2b068f99e215`).
No production semantic is retained. Do not run fixed-32 or MPS and do not
iterate this family further. Deployment remains the protected step-zero model;
future observation work requires materially different architecture and
independently better causal evidence.

---

## 265. Foundation RGB features require instance-support parity

A locally cached pretrained vision model may be evaluated as an independent
RGB feature provider, but pretrained attribute quality is not permission to
reduce object-discovery support. Foundation-model weights must remain local and
offline; simulator labels may train a temporary decoder but may not enter
runtime inference or candidate choice. Production integration requires raw
instance support, typed evidence, scenario breadth, and physical behavior to
clear predeclared gates before checkpoint/config semantics are added.

The first frozen-foundation probe uses the exact cached
`facebook/dinov2-small` weights (SHA-256
`ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`).
A temp-only decoder receives DINO patch features and emits center heatmap, log
radius, inverse depth, colour, and visibility after exactly 128 balanced CPU
updates. Against the trained dense typed candidate, held-out log-radius MAE
improves `1.99744 -> 0.03972`, inverse-depth relative MAE
`5.05671 -> 0.12472`, visibility MAE `0.03640 -> 0.02963`, and the four-field
composite `1.80009 -> 0.10063`.

The probe is nevertheless rejected: top-count center recall falls
`181/191 -> 128/191` (`94.76% -> 67.02%`), matched attribute support falls
`181 -> 128`, colour MAE worsens `0.10981 -> 0.20843`, and minimum scenario
recall is `37.5%` on heavy/light. Only three of six declared gates pass. The
report is
`/private/tmp/20260824-spec169-dinov2-typed-fixed8-feasibility.json`
(SHA-256
`35a57c65f71717941cdd07a1ac970b132aed932c76b798a5ba26538fdb6d27a8`).
No production code or semantic is retained. Do not tune resolution, decoder,
loss weights, duration, thresholds, or compose these attributes with rejected
dense centers on the gate manifest. Reopen only with a materially different
local instance-aware pretrained model or independently stronger evidence.

---

## 266. Adaptive local-model evidence must be event segmented

The heterogeneous runtime pool already supports per-entity, per-axis,
per-regime, and per-horizon applicability, causal delayed RGB evidence,
explicit learned fallback, forecast-only isolation, and calibrated candidate
scoring. A new local-model family must therefore demonstrate useful causal
adaptation across the protected broad gate rather than merely reuse the pool
interfaces or improve selected pooled horizons.

The first materially distinct adaptive candidate was a temp-only Gaussian
residual model. For each persistent entity, world axis, learned interaction
regime, and exact configured horizon, it accumulated delayed associated RGB
forecast errors and required four causal pairs before use. It fitted a clipped
online least-squares gain in `[0, 1]`, bounded residual corrections to
`0.25 m`, corrected only x/y forecast outputs, and expanded learned variance
only to the empirical corrected-error second moment when larger. It never
mutated `WorldBelief`, model weights, velocity, events, lifecycle, or identity;
unsupported and stale cells remained exactly on learned dynamics.

The fixed-eight CPU/RGB diagnostic is active and non-vacuous: `9,777` pair
updates, `7,875` supported corrections, `264` variance expansions, maximum
gain `1.0`, and maximum empirical second moment `0.0860407 m^2`. Pooled
position RMSE improves at 0.10 seconds (`0.142104 -> 0.129438 m`, `-8.91%`)
and 0.25 seconds (`0.167849 -> 0.160124 m`, `-4.60%`), with strong NLL gains.

The candidate is nevertheless rejected. Only `73/84` protected core cells
improve or tie; pooled 0.50/0.75-second position, 0.10/0.50-second
calibration, and 0.25-second identity evidence regress, while 12
scenario-horizon position cells fail across baseline, damped-contact,
elastic, glancing, heavy/light, and impulse slices. The compact report is
`/private/tmp/20260824-spec169-adaptive-gaussian-residual-fixed8.json`
(SHA-256
`ee0dcf2cd6289b6170c3df4417e8d80250c353e86c88bf4d2499d7ed90a5c087`),
diagnostics are SHA-256
`a150e8ab25349e6bdc315ebd93b35a0816abf46a4ce67e9653741e02222dde56`,
and the final evaluator report is SHA-256
`d2a4883dbaa789ab34ed43a7a4dc23e9506bd6420397f958235bb06672ea4326`.

No production semantic is retained. Do not tune pair support, residual bound,
gain, uncertainty expansion, horizon admission, or regime thresholds on this
manifest, and do not advance to fixed-32 or MPS. In particular, the successful
0.10/0.25-second cells may not be selected post hoc. The next local-model
candidate must use causal event/change-point segmentation or be a genuinely
different model with independent evidence.

---

## 267. Event segmentation does not rescue residual-history models

After the adaptive Gaussian residual failed the broad fixed-eight gate, its
only authorized continuation was a causal event/change-point boundary. The
candidate retained the exact four-pair support, `0.25 m` clipping, bounded
least-squares gain, x/y ownership, empirical second-moment uncertainty
expansion, and learned fallback. It added one rule only: when the accepted
learned trajectory changed interaction regime, all adaptive residual
statistics for that persistent entity and exact horizon began a new epoch.

The boundary is runtime-causal and candidate-independent. It uses only the
learned structured motion mode plus learned interval contact/collision outputs
already accepted by the ordinary regime classifier. Simulator truth,
evaluation targets, posterior target state, and alternative-candidate outputs
may not classify the transition. The reset does not mutate persistent
`WorldBelief`, model weights, velocity, events, lifecycle, or identity.

The protected fixed-eight CPU/RGB diagnostic exercises `435` regime
transitions across `248` entity-transition observations: `199` free-to-
collision, `125` collision-to-free, `56` collision-to-ground, `33`
ground-to-free, and smaller remaining transitions. It records `8,811` causal
residual pairs, `5,370` supported corrections, and `85` variance expansions.
The 0.10-second pooled position RMSE still improves
`0.142104 -> 0.138094 m`, but the prior 0.25-second improvement becomes a
regression (`0.167849 -> 0.169963 m`) and the candidate introduces
0.10-second identity mismatch rate `0.005587` from an exact-zero reference.

Only `68/84` core cells improve or tie and 13 scenario-horizon position cells
regress, versus the already-rejected unsegmented candidate's `73/84` and 12.
The final report is
`/private/tmp/20260824-spec170-event-epoch-adaptive-residual-fixed8.json`
(SHA-256
`370aaeb84f8dffb31ae6722a9e64b35e0eb95fe8a7137991d88d0185b3e57319`),
the evaluator report is SHA-256
`6a7bcd318e5eece05a70bb619af1948bebd7b10bfe7075071c2a0fbc36efd83a`,
event diagnostics are SHA-256
`b7cfbe1200980b396276251644778b29948d36cf67b0c83482b78c5bbccb7566`,
and the temp script is SHA-256
`d1c821a6305eeb57d764ee8b1bb410d2d9365bf9d9a95c28fe17ea90768a0199`.

This closes the adaptive/output-residual-history family. Do not tune regime
classification, epoch reset scope, support, residual bound, gain, uncertainty,
horizon admission, or adjacent thresholds, and do not advance this candidate
to fixed-32 or MPS. A future runtime candidate must carry genuinely different
causal model state and evidence rather than another partition of residual
history.

---

## 268. Hard runtime decisions require differentiable training surrogates

Hungarian identity assignment, lifecycle state changes, and resolved contact
jumps remain deterministic runtime inductive biases. They must not be the only
route by which the values feeding those decisions learn. Causal RGB training
therefore exposes an ephemeral `DifferentiableIngestTrace` containing the live
predicted belief, projected measurement, RGB measurement, full association
cost matrix, hard association, innovation, and final hard posterior for exactly
one packet. This object is training-local: it is never written to
`RuntimeState`, measurement caches, diagnostics, or checkpoints.

When `training.closed_loop_soft_association_temperature` is finite and positive,
the trainer computes a gated row-softmax over the same cost matrix consumed by
Hungarian matching. Invalid, low-confidence, source-incompatible, nonfinite,
and above-maximum-cost pairs remain excluded. The expected RGB world position
and supported temporal velocity receive physical target losses; column mass
above one receives a small exclusivity penalty. This relaxes assignment values,
not integer identities. Ordinary `ingest`, validation, evaluation, and
deployment continue to use the exact hard runtime output.

The `differentiable_state_estimator` scope trains the RGB observation module,
the recurrent belief updater, and the causal physical identifier together.
Equation-based gravity, drag, contact, and uncertainty propagation remain
frozen parameter modules but preserve derivatives with respect to estimated
state and identified physical parameters. Existing observable drag and
restitution losses plus analytic rollout losses derive those parameters without
asking a learned residual network to reproduce elementary mechanics. The
learned dynamics residual remains frozen in this phase and may be reopened only
for bounded residual model mismatch after state estimation qualifies.

All new semantics are legacy-off and exact-resume-bound: null temperature and
zero soft-association weights execute the historical path without requesting a
trace. Positive soft-association weights require the differentiable estimator
scope. The profile `configs/differentiable_physics_assimilation_cpu.yaml` uses
CPU causal execution, the analytic rollout prior, temperature `0.5`, and
explicit state/velocity/exclusivity weights. Training may use simulator state
only after ingestion to form losses; no privileged state enters runtime
association, lifecycle, correction, parameter updates, or prediction.

The bounded production smoke at
`/private/tmp/20260824-111854-20260824-spec169-differentiable-assimilation-one-update-smoke`
completed one balanced eight-scenario update with no skipped draw. It recorded
72 supported soft position coordinates, three supported temporal-velocity
coordinates, 70 observable drag objects, and seven observable restitution
objects. Loss and gradients were finite (`2.708743`, raw norm `1.875017`,
applied norm `1.017866`); the learned attention/dynamics residual gradient was
exactly zero and peak RSS was about `1.88 GiB`. This is graph, ownership, and
throughput evidence only. The warmup-scale single update is not accuracy,
promotion, or convergence evidence. The next experiment is one predeclared
short paired qualification against the hard-runtime baseline; stop if broad
fixed validation does not improve materially rather than iterating objective
weights on the gate manifest.

---

# Closing directive

Project Orpheus should emerge from the first serious implementation as a small but real online world-model system:

- it observes a physical scene;
- maintains persistent object beliefs;
- predicts dynamics using structured and learned components;
- expresses uncertainty;
- detects discontinuous events;
- receives another observation;
- updates state and gradually updates physical parameters;
- immediately revises the future;
- supports future modalities through a stable contract.

The first synthetic scene is not the destination. It is the minimum fully instrumented environment in which the architecture can be proven before spending money on large GPU training. The project should therefore be simple to operate, strict about interfaces and evidence, and ambitious about the problem structure from the first commit.
