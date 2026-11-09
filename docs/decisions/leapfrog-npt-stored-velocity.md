# D[leapfrog-npt-stored-velocity]

Recorded 2026-11-09.

Amends [D76](D76.md). With leapfrog and a barostat, store
$\mathbf v_\mathrm{stored}=\mathcal P_{\mathbf x_n}
(\mathbf v_\mathrm{coupled}-hM^{-1}\mathbf F_n)$, where
$h=\Delta t/2$ and $\mathcal P$ is the mass-weighted constraint projector
at the current positions. Apply it after coupling, at each native entry
boundary that stores the backward half-kick. SETTLE and SHAKE use their
existing velocity projectors; unconstrained particles retain the same
backward half-kick.

The former unprojected storage retains a constraint-normal force component.
Equivalent force representations can therefore change written velocities
despite identical constrained dynamics. The projector removes that
dependence. The next step reconstructs and projects its full velocity, so
the coupled velocity and positions must remain unchanged to rounding.
The thermostat, barostat and their kinetic quantities still consume the
projected full-step velocity before this storage operation.

No control key, file format, overwrite rule or velocity timestamp changes.
H5MD and checkpoints continue to carry the stored velocity, with leapfrog
offset $-0.5$; with a barostat its real constrained particles now have
constraint-consistent stored velocities. This does not convert the stored
array to full-step velocities. Existing checkpoints remain readable.

## Validation contract, declared before measurements

Against main and an independent step reference, bound position differences
by $10^{-11}$ nm in double and $10^{-7}$ nm in mixed, and reconstructed
projected coupled-velocity differences by $10^{-9}$ nm/ps in double and
$10^{-4}$ nm/ps in mixed. Check normalized bond velocity residuals with
bounds $10^{-10}$ nm/ps in double and $10^{-5}$ nm/ps in mixed. These cover
f64/f32 rounding; any trajectory difference beyond them stops validation.

Use independent NumPy mass-weighted projection and constrained step
arithmetic, CPU and GPU, both precision modes, water and hydrogen-bond
constraints, and coupling/reporting schedules. Run the existing leapfrog
constraints, Langevin-reference and energy-step determinism tests, then the
full suite at the final clean head. Results are pending. The separate
rigid-water-template PR rebases after this fix merges and reruns its failed
case with raw stored arrays and its unchanged numerical bounds.
