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

## First development check: stopped in the independent oracle

The first valid CPU double/Trotter fixture compared the changed leapfrog
route with velocity Verlet from the same source tree. At steps 10/11,
position differences were $2.4424906541753444\times10^{-14}$ and
$2.5757174171303632\times10^{-14}$ nm; cell differences were zero.
Projected full-step velocity differences were
$2.1662810434364133\times10^{-12}$ and
$2.0056456495609609\times10^{-12}$ nm/ps. The step-10 stored velocity
differed from the independent projected backward-half-kick reference by
$2.0832849334517789\times10^{-12}$ nm/ps, with normalized bond velocity
residual $1.1002474477654185\times10^{-15}$ nm/ps.

The independent SHAKE reference for step 11 did not converge within its
1000 sweeps at its declared $10^{-14}$ relative squared-bond criterion.
The test stopped there (0 passed, 1 failed); no oracle or trajectory
comparison was retried. These measurements are against velocity Verlet,
not archived main, and are not final clean-head gate results. Main,
GPU/mixed, existing leapfrog regressions and the full suite remain pending.

The CPU-only diagnosis found an f64 in-sweep cycle: the stale
pre-correction maximum was $1.014641954615416\times10^{-14}$ while the
completed-state maximum was $7.951959625309013\times10^{-15}$. The
propagated coordinate-rounding level was $1.42\times10^{-14}$. The
maintainer approved monitoring the completed sweep at $10^{-13}$ (about
seven times that level), without changing any comparison bound. The one
controlled checker rerun on the preserved acquisitions passed: maximum
position difference $7.2608585810485238\times10^{-14}$ nm, projected
full-step velocity difference $6.1612936974597687\times10^{-12}$ nm/ps,
stored-velocity oracle difference $4.724082236506888\times10^{-12}$ nm/ps,
and normalized bond velocity residual
$1.1002474477654185\times10^{-15}$ nm/ps. The independent SHAKE step
position difference was $7.1054273576010019\times10^{-15}$ nm.
