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
