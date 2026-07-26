# Evaluation

Held-out evaluation is causal and RGB-only unless a report is explicitly marked
oracle ablation. It measures state/forecast error by horizon, correction
improvement, assignment coverage, distance-gated detection/identity, collision
events, runtime parameter observability/update gates, uncertainty
coverage/NLL/sharpness, finite outputs, and component latency.

Transparent static, constant-velocity, default analytic, and explicitly
labelled oracle-parameter analytic baselines use the same episode contracts and
forecast masks. Simulator labels align metrics but are never fed back to the
runtime. Full results include JSON plus Markdown and never use future
observations to score an earlier belief.

Current limitations are recorded rather than hidden: physics-violation and
failure-plot suites are not yet exported, and parameter MAE is withheld when
localization fails the configured 0.5 m metric gate.
