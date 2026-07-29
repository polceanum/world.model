# Filtering

The online filter first produces a prior through dynamics. For supported RGB
geometry dimensions it uses a diagonal Kalman-style proposal, robustly clips
whitened residuals, and permits only a bounded learned residual/gate. Posterior
variance contracts for reliable associated measurements and expands on misses,
elapsed time, and process noise. Missed objects receive explicit additional
log-variance growth; predicted occlusion reduces existence decay while retaining
identity.

Fast pose/velocity corrections and slow parameter updates are separate. Drag,
restitution, friction, mass ratio, and geometry receive evidence only through
explicit observability gates.

After ordinary associated RGB correction, a modality may attach bounded
direct kinematic evidence in persistent belief-slot order. Velocity and
position validity are independent. Position evidence from the RGB
point/scale trajectory is projected onto the current calibrated camera-depth
axis by default; unobserved axes receive deliberately large variance. The
same diagonal robust Kalman update then corrects position/variance in
`WorldBelief`, followed by any valid velocity update. Scale anchors are reset
on collision edges, reject ambiguous associations, and use IRLS/Huber
reweighting so a single overlap-induced apparent-radius error does not become
authoritative depth.
