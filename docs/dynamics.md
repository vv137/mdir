# Dynamics, Minimization, and Restraints

Status: kept current; last checked against the code on 2026-10-21. The
decisions are D50, D60, D67, D72 to D74, D76, D77, and those that amended them
([decisions.md](decisions.md)); the keys are in
[the control file](paper/A-control-file.md).

This document describes the removal of the motion of the center of mass,
the thermostats and the barostat, a cell that changes, minimization,
restraints, and leapfrog. MDIR does not repartition the mass of hydrogen;
it runs the masses that a topology gives, repartitioned ones included, as
the Amber suite's systems at 4 fs.

## 1. Removal of the motion of the center of mass

The total momentum is removed where the velocities are drawn (M0). In a
run it drifts: rounding in the forces, most in mixed precision, and the
thermostat, which scales velocities, does not remove it.

$$\mathbf v_i \leftarrow \mathbf v_i - \frac{\sum_j m_j \mathbf v_j}{\sum_j m_j}$$

every `center_of_mass_interval` steps.

| Item | Proposal |
|---|---|
| What is removed | The linear momentum. Under periodic boundaries the angular momentum is not conserved and is left alone. |
| How | A `particle_for` with a sum of `vector<3xf64>`, as the virial is summed, and a `particle_for` that subtracts and scales for the thermostat |
| How often | `center_of_mass_interval` in `[dynamics]`: with a thermostat, when it acts; without one, only if given (D67) |
| Degrees of freedom | Three fewer, as in M0 |

## 2. Thermostat and barostat

### 2.1 Kinds

| | Proposal | Reason |
|---|---|---|
| Thermostat | Stochastic velocity rescaling (Bussi, Donadio, and Parrinello 2007 [[Bussi2007]](references.md#bussi2007)) | It samples the canonical distribution, and it takes the kinetic energy and a few random numbers for each step, not one for each particle. |
| Thermostat, the other | Langevin dynamics by the middle scheme ([[Zhang2019]](references.md#zhang2019); BAOAB of [[Leimkuhler2013]](references.md#leimkuhler2013)), with constraints g-BAOAB ([[Leimkuhler2016]](references.md#leimkuhler2016), D265), `method = "LANGEVIN"` (D135, Section 2.6) | The thermostat of most runs of Amber and OpenMM; it acts on each particle in every step, with three random numbers for each. |
| Barostat | Stochastic cell rescaling (Bernetti and Bussi 2020 [[Bernetti2020]](references.md#bernetti2020)), isotropic; semi-isotropic, with the pressure in a plane apart from that along its normal, as an option | It samples the distribution at constant pressure and is of first order: it has no momentum of the cell to store. |

Both carry the statement that they preserve the target distribution
(P10). Since M1: a Nosé–Hoover chain, `method = "NOSE-HOOVER"` (D163a);
Brownian dynamics as an integrator (D163b); and anisotropic coupling of the
barostat, each axis on its own (D163c). The keys are in
[the control file](paper/A-control-file.md).

Two barostats go by the name of Bussi:

| | Bussi, Zykova-Timan, and Parrinello 2009 [[Bussi2009]](references.md#bussi2009) | Bernetti and Bussi 2020 [[Bernetti2020]](references.md#bernetti2020) |
|---|---|---|
| Kind | Of second order: the cell has a momentum, and the thermostat acts on it as well | Of first order: the barostat of Berendsen [[Berendsen1984]](references.md#berendsen1984) with a term of noise |
| State beside the cell | The momentum of the cell | None |
| Parameters | A time | A time and a compressibility. A wrong compressibility changes how fast the volume relaxes, not what is sampled. |
| The volume | May oscillate | Relaxes |

MDIR takes the one of 2020 (D50). A checkpoint needs nothing for it.

### 2.2 In `dyn`

*Proposed, not implemented as written.* The couplings are not ops of
`dyn`: the driver writes them into the schedule of `@mdir_run` at the end
of each period of coupling (Section 2.4). The factor of the thermostat
and the strain of the barostat are calls to the host (`mdrtBussiFactor`,
`mdrtBarostatStrain` of `libmdrt`, which draw from Philox 4×32-10 by the
key of A13), and loops over particles (`md.map_particles`) remove the
motion of the center of mass and scale the velocities and the positions.
The proposal was:

```mlir
dyn.program @step(...) attributes {
    requires = ["temperature", "pressure"],
    provides = ["thermostatting", "barostatting"]} {
  ...
  %alpha = dyn.velocity_rescaling %kinetic, %freedom, %step
             temperature(300.0) tau(1.0) dt(%span) stream(0)
  %v2    = dyn.scale %v1, %alpha : !vec
  ...
}
```

| Op | Definition |
|---|---|
| `dyn.velocity_rescaling` | The factor of the thermostat, from the kinetic energy, the degrees of freedom, and random numbers |
| `dyn.cell_rescaling` | The factors of the barostat along the three edges, from the pressure along each |
| `dyn.scale` | A field times a number, or times a number for each direction |
| `dyn.scale_cell` | A cell with edges times a number for each direction |
| `dyn.random` | A random number (P5, A5, A13) |

| Item | Proposal |
|---|---|
| The kinetic energy of the thermostat | That of the velocities it scales (D67): with velocity Verlet, those at the end of a step, which are of one time. `K_T` of D45 estimates the temperature of the log. |
| The kinetic energy of the barostat | That of the velocities of the step without that of the center of mass, before the thermostat (Section 2.4), not `K_P` of D45 |
| The pressure of the barostat | With the correction for the dispersion ([nonbonded.md](nonbonded.md), Section 1.2) and the virial of the constraints ([constraints.md](constraints.md), Section 1) |
| How often | Every `n` steps, with `n` times the time step. A step between computes no global sum. On a device a step with global sums takes up to twice the time of one without (Section 10.8 of ops.md). |
| Groups with a thermostat each | One group in M1 |

Two rows of this table no longer hold: the thermostats control the optimal
temperature, measured from the kinetic energies of the half steps at the
steps that close a period of coupling (D263); and the log and the barostat
take the pressure of the groups that the barostat scales, with the kinetic
energy of their centers at the half steps (D264).

### 2.3 Random numbers

| Item | Proposal |
|---|---|
| The generator | Philox 4×32 with 10 rounds [[Salmon2011]](references.md#salmon2011), from the key of A13 |
| Where it runs | On the host, in `libmdrt`: the thermostat and the barostat take numbers for the system, not for a particle |
| Streams | 0 for the thermostat, 1 for the barostat. The draw index counts the numbers of one step. |
| Entity | A fixed key, as for a global move (A5) |
| Random numbers in kernels | For Langevin dynamics (D135): Philox emitted by the driver as `arith` ops in the kernel of the step, one block for each particle and step, stream 2, the entity the number of the particle in the input. A block gives four words, four uniform numbers $(w + \tfrac12)\,2^{-32}$, and by Box and Muller two pairs of normal numbers, of which three are taken. It agrees with the generator of the host to the last bit (`langevin.test`). |

### 2.4 The barostat as it is

Stochastic cell rescaling, isotropic or semi-isotropic, with velocity
Verlet and leapfrog (D72, D76, D77, D119):

| Item | Rule |
|---|---|
| When | At the end of the step that completes a period of coupling, after the removal of the motion of the center of mass and the thermostat; `interval` equals `interval`. The last step of a period computes the virial for it. |
| The pressure | That of the groups that the scaling moves by their centers (D264, [[Jung2020]](references.md#jung2020)): $P_a = (2K_{\text{half},g,a} + W_{g,aa} + C/(3V)) / V$ on axis $a$, with $\mathsf W_g = \sum_g \mathbf R_g\otimes\mathbf F_g$ the virial of the groups (the rigid groups by their centers, every other particle alone), $K_{\text{half},g} = K_g + \tfrac{\Delta t^2}{8}\sum_g F_{g,a}^2/M_g$ the kinetic energy of the centers at the half steps without that of the center of mass, exact from the forces since those of the constraints cancel within a group, and $C/V$ the virials of the correction for the dispersion and of the background of a net charge at the volume $V$ of the cell before the scaling. No term $k_B T / V$. |
| The change of the volume | One step of Euler and Maruyama over the period $\Delta t_p$ of the equation for $\lambda = \sqrt V$, eq. (7) and eq. (S7) of [[Bernetti2020]](references.md#bernetti2020): $\lambda' = \lambda - \tfrac12 f\lambda\, \big(P_0 - P - k_B T c/(2V)\big) + \sqrt{k_B T f c/2}\, R$, $\Delta\varepsilon = 2 \ln(\lambda'/\lambda)$, $f = \beta_T \Delta t_p / \tau_p$. Its noise does not depend on the volume, which the paper's reversible integrators need; to first order it is the step of $d\varepsilon = -(\beta_T/\tau_p)(P_0 - P)\, dt + \sqrt{2 k_B T \beta_T / (V \tau_p)}\, dW$, eq. (5), which GROMACS takes. Pressures in bar, $c = 16.6053906717$ bar nm³ mol/kJ, $T$ that of the bath. $R$ is the normal number of stream 1 of the step (A13), drawn on the host (`mdrtBarostatStrain`). |
| Scaling | The positions of every particle and the edges of the cell by $\mu = \exp(\Delta\varepsilon/3)$, the velocities by $1/\mu$; a group that the constraints keep rigid (a water of SETTLE, a group of SHAKE) moves with its center of mass and keeps its shape, since stretched bonds would be taken back by the constraints of the next step with a change of the velocities that heats the system. Virtual sites are placed again in the next step |
| The cell | Kept in memory on the host, where each iteration of a loop takes it, and the steps that follow a loop of periods in the same iteration take it again; the neighbor structures, whose test of validity compares the cell, are built again; the influence function of PME follows (pme.md); the log and the trajectory take the new edges (`mdrtSetBox`) and, for a triclinic cell, the tilts (`mdrtSetTilt`); a checkpoint keeps them, and a restart takes them. A triclinic cell scales as $H\,\mathrm{diag}(\boldsymbol\mu)$: each tilt with the column it is in, $b_x$ and $c_x$ by $\mu_x$, $c_y$ by $\mu_y$ (D127). A run stops if an edge becomes shorter than twice the cutoff, below which the minimum image misses pairs; a run that begins so is rejected |
| The scaled positions | With `work = "EXACT"`, the virtual sites are placed on the scaled positions, and those are evaluated in the new cell: their energy gives the work of the scaling, and their forces are those that the next step begins with (D77). With `"FIRST_ORDER"`, the next step begins with the forces of the positions before the scaling, and places the sites after its drift |
| The conserved energy, exact | Takes away what the scaling gives: $U(\mathbf x') - U(\mathbf x)$, the potential energy of the scaled positions less that of the positions before; $C\,(1/V' - 1/V)$, the change of the terms that are constants of the volume (the correction for the dispersion and the background of a net charge, $C$ their energy times the volume of the start), which `md.evaluate` leaves out (amended 2026-10-02: before, the conserved energy carried $E_c(V) - E_c(V_0)$); and $(1/\mu^2 - 1) K$, the change of the kinetic energy of the velocities it scales. Between scalings the dynamics is that of constant energy, so the conserved energy changes as there. On the mixture of `barostat.test`, 7.4 × 10⁻⁶ of its value over 8 ps instead of 2.2 × 10⁻³; on tri-alanine in 1218 OPC waters at 1 bar, 300 K, `time_constant = 2`, 2 fs, on a GPU in mixed precision (`examples/ala3`), 0.011 kcal/mol per ps instead of 2.1. The evaluation costs 0.13 ms per step at a period of coupling of 10 steps there, 20% of the rate (316 to 253 ns/day): the cost of the reversible integrator of [[Bernetti2020]](references.md#bernetti2020) (Table I; SI Sec. V.B), which this is but for the thermostat, applied once at the end of the period rather than in halves around the step. This conserved energy is not the paper's effective energy, which adds $P_0 \Delta V$, the terms of the noise, and the drift of eq. (7) to measure the violation of detailed balance (Sec. II.C); it tests the dynamics between scalings and the counting of the work, and the distribution of the volume tests the barostat |
| The scaling of Trotter type, the default | The last two steps of a period follow its loop: the step of energy whose pressure gives the strain and the new cell, and `step_trotter`, which kicks half, drifts half, scales the positions by $\mu$ (the rigid groups with their centers) and the velocities by $1/\mu$, drifts the other half, and evaluates in the new cell with the virial ([[Bernetti2020]](references.md#bernetti2020), SI Sec. V.C, eqs. S12a–d; eqs. S13a and S15 have two misprints, D92). The conserved energy takes away $(1/\mu^2 - 1) K$ of the velocities scaled and $-\ln\mu\, (\mathsf W_\text{before} + \mathsf W_\text{after}) / 2$, the virials of the groups and the constant terms before and after the step: $\sum_g \mathbf X_g\cdot\mathbf F_g$, from the forces of the evaluations, $\mathrm{tr}\,\mathsf W_\text{eval} - \sum_g\sum_{j\in g}(\mathbf x_j - \mathbf X_g)\cdot\mathbf F_j$, which the forces of the constraints do not enter (D116). No evaluation is added; the count drifts with the period of coupling (D92) |
| Semi-isotropic coupling | x and y scale together by the strain of $\ln A$ from $(P_{xx} + P_{yy})/2$ less $\gamma/L_z$, z by that of $\ln L_z$ from $P_{zz}$, each with its own normal number (the first two of stream 1 of the step), one step of Euler and Maruyama of eqs. (9a) and (9b) of [[Bernetti2020]](references.md#bernetti2020) (`mdrtBarostatStrainArea`, `mdrtBarostatStrainHeight`). The pressure of each axis takes its kinetic energy without the center of mass and the diagonal of the virial; the scale, the kinetic energy, and the virials are vectors of three, and the work sums over the axes. `work = "FIRST_ORDER"` is for isotropic coupling only (D119) |
| The period of coupling | The paper finds the fluctuations of the volume too large when $N_P \Delta t$ is not small against $\tau_p$ (a TIP3P box at $\tau_p = 0.5$ ps: $\sigma^2_V$ from 0.23 at $N_P \le 10$ to 0.45 nm⁶ at 100, Fig. 3b), with $N_P = 10$ as a compromise and little gain in rate beyond 20 to 40 in GROMACS (Fig. 4). A longer period that lowers the cost of the evaluation must keep $N_P \Delta t / \tau_p$ small |
| The conserved energy, first order | Takes away $-(\mu - 1) \mathrm{tr}\,\mathsf W_g$, the change of the potential energy to first order, as GROMACS does, and $(1/\mu^2 - 1) K$, that of the kinetic energy, exactly. $\mathsf W_g$, the virial of the rigid groups that move as wholes, is the $\mathsf W$ of the pressure above, which has the virial of the constraints, with twice the kinetic energy of the motion within the groups, $\sum \tfrac12 m \lVert\mathbf v - \mathbf V\rVert^2$ over each (the virial of the forces within a rigid group is minus that). What is left is the second order, $\tfrac12 (\mu - 1)^2\, d^2U/d\mu^2$, whose mean over the noise of $\Delta\varepsilon$ is proportional to its variance, and so to $f$: a drift that neither the time step nor the period of coupling reduces, only `time_constant`. On the mixture of `barostat.test` at 2 fs, 1.8 × 10⁻³ of the energy over 8 ps with `time_constant = 2`, 4.4 times less with `time_constant = 8`; on 1394 OPC waters with PME at 300 K and 1 bar, 2.1 kcal/mol per ps with `time_constant = 2` and 0.52 with `time_constant = 8`. The same runs at constant volume keep the conserved energy to 10⁻⁶ and 10⁻⁵ |
| Parameters | In `[ensemble]`: `ensemble = "NPT"`, `temperature`, and `pressure` in atm. In `[barostat]`: `method = "C-RESCALE"`, `time_constant` in ps (5 by default), `compressibility` in 1/atm (4.5 × 10⁻⁵ /bar by default), `coupling` (`"ISOTROPIC"`, `"SEMI_ISOTROPIC"`, or `"ANISOTROPIC"`, each axis on its own, D163c, with `compressibility` one number or three; with the second, `compressibility_z` in 1/atm, that of x and y by default, 0 to keep the height, `surface_tension` in dyn/cm of each surface normal to z, and `surfaces`, 2 by default), `work` (`"TROTTER"` by default, `"TROTTER_FIRST_ORDER"`, `"EXACT"`, `"FIRST_ORDER"`), and `interval`, which must be that of the thermostat. NPT needs a `[thermostat]` |

The drift of the conserved energy on tri-alanine in 1218 OPC waters
(`examples/ala3`, 4,905 particles, from its checkpoint after 100 ps at
constant pressure; 100 ps of each, 2 fs, SETTLE and SHAKE, PME, an RTX
3090, velocity Verlet, the barostat with the exact work and the step in
$\lambda$, thermostat and barostat every 10 steps, `time_constant` 0.5 ps, `time_constant` 2 ps),
from the slope of a line through the rows of the log:

| Ensemble | Mixed: kcal/mol/ps | kJ/mol/ns per atom | ns/day | Double: kcal/mol/ps | kJ/mol/ns per atom | ns/day |
|---|---|---|---|---|---|---|
| NVE | −0.0111 | −0.0094 | 426.5 | −0.0082 | −0.0070 | 83.8 |
| NVT | +0.0047 | +0.0040 | 423.0 | −0.0015 | −0.0013 | 83.7 |
| NPT | −0.0014 | −0.0012 | 253.9 | −0.0054 | −0.0046 | 57.8 |

The barostat costs 40% of the rate at this period: the evaluation after
each scaling, and the neighbor structures that each change of the cell
rebuilds (with the first-order work, 316 ns/day).


### 2.5 The effective energy of the barostat (planned)

The conserved energy of Section 2.4 tests the dynamics between scalings
and the counting of their work. Whether the barostat itself samples its
distribution is what the effective energy of [[Bernetti2020]](references.md#bernetti2020)
measures (Sec. II.C and SI Sec. IV): the work that the steps of the volume
do against detailed balance.

At fixed scaled positions and momenta the variable $\lambda = \sqrt{V}$ has
the stationary distribution $P(\lambda) \propto e^{-E(\lambda)/k_BT}$ with

$$E(\lambda) = K + U + P_0 \lambda^2 - k_BT \ln\lambda,$$

the last term from $dV = 2\lambda\, d\lambda$. Since
$-\partial(K + U)/\partial V = P_\text{int}$ at fixed scaled coordinates,

$$f(\lambda) \equiv -\frac{dE}{d\lambda}
  = -2\lambda\left(P_0 - P_\text{int} - \frac{k_BT}{2\lambda^2}\right),$$

and the step of eq. (S7), with $D = k_BT\beta_T/(4\tau_p)$ and the period
$\Delta t$ of the barostat, is

$$\lambda' = \lambda + \frac{D}{k_BT} f(\lambda)\, \Delta t + \sqrt{2D\Delta t}\, R.$$

The probability of a step and of its reverse are Gaussians of the same
variance $2D\Delta t$ about $\lambda + a f$ and $\lambda' + a f'$, with
$a = D\Delta t/k_BT$, $f = f(\lambda)$, $f' = f(\lambda')$ at the scaled
configuration. With $d = \lambda' - \lambda$,

$$k_BT \ln\frac{T(\lambda\to\lambda')}{T(\lambda'\to\lambda)}
  = k_BT\,\frac{(d + a f')^2 - (d - a f)^2}{4D\Delta t}
  = d\,\frac{f + f'}{2} + \frac{D\Delta t}{4k_BT}\left(f'^2 - f^2\right),$$

and the work of a step, which detailed balance would make zero, is

$$w = \Delta K + \Delta U + P_0 \Delta(\lambda^2) - k_BT\, \Delta\ln\lambda
  + d\,\frac{f + f'}{2} + \frac{\beta_T \Delta t}{16\,\tau_p}\left(f'^2 - f^2\right).$$

Its exponential has mean one from equilibrium,
$\langle e^{-w/k_BT}\rangle = 1$. The coefficient of the last term in the
image of the paper's eq. (S11), $\beta_T\Delta t/(4\tau_p k_BT)$, is not an
energy; `scripts/validation/barostat/effective_energy.py` checks on a model
in one variable that the closed form above equals the definition to
rounding and satisfies the identity, and that the other does not. The
effective energy is the conserved energy of Section 2.4 plus the sum of
$w - \Delta K - \Delta U$ over the steps of the volume.

To compute it, $P_\text{int}$ in $f$ and $f'$ must have one definition
before and after a scaling. With rigid groups scaled about their centers
of mass it is the molecular pressure; at the scaled positions MDIR has the
atomic virial without the constraints, so the molecular one needs
$\sum_\text{groups}\sum_i (\mathbf r_i - \mathbf R)\cdot\mathbf F_i$
as well. Without constraints the atomic pressure serves. Since D116 and
D264 the counting of the work and the pressure take the virial of the
groups from the forces of the evaluations, which defines $P_\text{int}$
the same before and after a scaling; the effective energy itself is not
computed yet ([roadmap](roadmap.md), barostat integrators).

### 2.6 Langevin dynamics

`method = "LANGEVIN"` of `[thermostat]`, with `friction` $\gamma$ in 1/ps
(D135). Without constraints each step is that of velocity Verlet with the
friction and the noise between the two halves of its drift, the middle
scheme [[Zhang2019]](references.md#zhang2019), BAOAB in the terms of
[[Leimkuhler2013]](references.md#leimkuhler2013):

$$
\begin{aligned}
\mathbf v &\leftarrow \mathbf v + \tfrac{\Delta t}{2}\,\mathbf F/m, &
\mathbf x &\leftarrow \mathbf x + \tfrac{\Delta t}{2}\,\mathbf v, \\
\mathbf v &\leftarrow c\,\mathbf v + \sqrt{(1 - c^2)\,k_B T/m}\;\mathbf R, &
\mathbf x &\leftarrow \mathbf x + \tfrac{\Delta t}{2}\,\mathbf v,
\end{aligned}
$$

with $c = e^{-\gamma\Delta t}$ and $\mathbf R$ three normal numbers of the
particle (Section 2.3), then the evaluation and the second half kick.
With constraints the step is g-BAOAB, the step of the Appendix of
[[Leimkuhler2016]](references.md#leimkuhler2016) with one drift in each A
(D265): B, the half kick and the projection of the
velocities on the constraints (RATTLE, SETTLE); A, a drift of
$\Delta t/2$, the constraints of the positions with the positions before
that drift as the reference, the velocities of the displacement, and
their projection at the new positions; O, the friction and the noise and
the projection; A again; the evaluation; B. The middle scheme that it
replaces let the noise and the friction act on velocities with a part
along the constraints, removed only by one constraint of the positions
over the whole step: on rigid water at 2 fs its state was 1.4 K above the
target and at 4 fs 4.8 K, against −0.2 and −0.4 K for g-BAOAB (#268).
Leapfrog takes the same step, whose stored velocities are those of the
second drift: with constraints every program of its step starts, as
velocity Verlet does, from the projected velocities of the time of the
positions (Section 6). A particle of mass 0 keeps its velocity.

| Item | Rule |
|---|---|
| Where | In every program of a step: `step`, `step_energy`, `step_virial`, and `step_trotter`, which scales the cell after the friction and the noise (with constraints after their projection, between the two A), so that the kinetic energy of the count of its work is that of the velocities it scales |
| The key | The seed; the number of the step, from a counter on the host that every step advances from the first step of the run (D129), so that a continued run takes the numbers of one that does not stop; the number of the particle in the input, so that the order of the positions does not change them |
| The momentum | Not kept: the degrees of freedom are $3N - N_c$, less 3 only if `center_of_mass_interval` removes the motion of the center of mass |
| `interval` | Paces the removal of the motion of the center of mass and the barostat, which act at the end of a period; 10 by default with a barostat, otherwise none |
| The log | No conserved energy: the run does not count the energy that the friction and the noise exchange with the bath (the roadmap). The temperature is that of $K_\text{half}$, not the optimal estimate: the friction and the noise act on the velocities of the middle of the step, which BAOAB samples at the temperature of the bath; with constraints the steps of the rows measure $K_\text{half}$ from the chord velocities of the second A of the step and of the first A of the next without its noise (D265) |
| The virial of the constraints | That of the forces that keep the bonds at their lengths at the end of the step, solved from its positions, velocities, and forces, $\mathbf e\cdot(\mathbf a_q - \mathbf a_p) = -\lvert\mathbf v_q - \mathbf v_p\rvert^2/d$, with the matrix of RATTLE; the changes of SHAKE and RATTLE within the step hold the friction and would give a pressure higher by $\gamma\Delta t/2$ of the virial of the constraints, 39 bar per 1/ps for water at 2 fs |

**Measured.** The noise of one step from velocities of 0 against Philox
of the host: within $5\times10^{-7}$ of $\sqrt{k_B T/m}$, the part of the
forces, on the CPU and on a GPU in double and mixed precision
(`langevin.test`, `langevin-gpu.test`). An ideal gas of 256 particles at
10 K coupled to 300 K with $\gamma$ = 5/ps relaxes at 9.6 ± 0.3/ps, against
$2\gamma$ = 10/ps; its mean kinetic energy over 50 ps is 228.0 ± 0.8
kcal/mol against 228.9. A run stopped at its checkpoints and continued
ends in the state of one that does not stop, bit for bit, on the CPU and
on a GPU. The tests of the ensembles of 1039 OPC waters at 2 fs with
$\gamma$ = 1/ps (`scripts/validation/ensembles/run.py --thermostat
langevin`, 5 ns a run): the density at 1 atm with cell rescaling
0.99611 ± 0.00030 g/cm³ against 0.99674 ± 0.00031 with velocity rescaling,
the slope of two pressures within 0.9 standard errors, that of two
temperatures within 2.0; the kinetic energy of the end of the step is
0.13% below $N_fk_BT/2$ (4 to 6 standard errors), so that its distribution
fails the test of Kolmogorov and Smirnov: at 2 fs the velocities of the end
of a step of the middle scheme are not those that it samples exactly,
which are those of the middle.

## 3. A cell that changes

A neighbor structure is valid in the cell that it was built in (Section
8.2 of ops.md). The barostat changes the cell.

| | A. The structure is built after every change of the cell | B. The structure follows the cell |
|---|---|---|
| How | Nothing new: the refresh finds another cell and builds | The barostat scales the positions that the structure was built at, and the reach of the structure shrinks with the smallest factor. The structure is valid while no particle has moved more than half of what is left of the skin. |
| Builds | At least one for each step of the barostat | As without a barostat |
| With a barostat every 10 steps and a build every 10 to 20 steps | Up to twice the builds | |

Proposal: A for M1, and B when the builds are measured to cost. *Done:
B (D80).* The test of validity takes the scale of each axis since the
build, $\mathbf m = \mathbf L \oslash \mathbf L^\text{ref}$, and holds while
$2\max_i \lVert \mathbf x_i - \mathbf m \odot \mathbf x^\text{ref}_i \rVert \le
\min(\mathbf m) R - r_c$, so a scaling of the cell rebuilds a structure only
when it would leave a pair within the cutoff out; the proof is in Section
4.1 of the white paper (`docs/paper/04-neighbors.md`). The inner list of a
dual list takes the same test against its last pruning (D114).

The cell is a value of the state that the loops carry (S1). The order of
the particles (D44) and the incidence structures do not depend on it. The
grid of particle mesh Ewald keeps its number of points and the correction
for the dispersion follows the volume.

## 4. Minimization

`[minimize]` in place of `[dynamics]` lowers the potential energy by
steepest descent (D73):

| Item | Rule |
|---|---|
| Start | The positions of the file taken onto the surface of the constraints, SETTLE (M-SHAKE below double precision) and SHAKE with the positions themselves as the reference, and the sites placed again, before the first evaluation (D120); the energy at the start in the log and the terms are of these positions |
| Direction | $\mathbf g = P(\mathbf F / m)$: the force over the mass, $0$ for virtual sites, with $P$ the projection that RATTLE applies to velocities, which takes off the parts along the bonds of SETTLE and SHAKE at the current positions |
| Step | $\mathbf x' = \mathbf x + h\, \mathbf g / \lVert\mathbf g\rVert_{16}$, with $\lVert\mathbf g\rVert_{16} = r \big(\sum_i (\lVert\mathbf g_i\rVert / r)^{16}\big)^{1/16}$ and $r$ the root mean square of $\lVert\mathbf g_i\rVert$; since $\lVert\mathbf g\rVert_{16} \ge \max_i \lVert\mathbf g_i\rVert$, no particle moves farther than $h$. Then SETTLE and SHAKE take the groups back to their shapes from $\mathbf x$, and the sites are placed |
| Acceptance | The step is taken if the energy at $\mathbf x'$ is lower: $h$ grows by 1.2, to at most 0.1 Å (D260; 1 Å before); otherwise $\mathbf x$ stays and $h$ shrinks by 0.2. The choice is made particle by particle (a map that selects), so that the fields keep storage of their own. A trial whose constrained lengths are not at their targets, $\lvert r^2/d^2 - 1\rvert > 10^{-4}$ below double precision and $10^{-8}$ in double, or not numbers, does not lower the energy: SETTLE or M-SHAKE did not take a group back from a step that turned it too far. Its energy is evaluated at $\mathbf x$, so that no position that is not a number reaches the neighbor structure, and is taken as $+\infty$ (D259) |
| Loops | Over the intervals between frames (one if there are none), over the intervals between energies in each, and over the steps; one `dyn.step @descend` per step, which evaluates the energy and the forces once |
| Log | The potential energy with the constant terms, the root mean square and the largest of the forces $m\mathbf g$ without their parts along the constraints, in kcal/mol/Å, over the particles with mass, the particle of the largest, and $h$ in Å |
| Checkpoint | At the end, with `checkpoint` of `[output]`: the positions, velocities of 0, and the integrator `MINIMIZATION`. A run that reads it, a minimization or a run of dynamics, takes the positions and the cell and begins anew at step 0 with drawn velocities; a minimization takes the positions and the cell of any checkpoint |
| Keywords | `[minimize]` takes `method = "STEEPEST_DESCENT"`, `steps` (positive, 100 by default; 0 is refused, #180), `initial_step` in Å (0.1), and `force_tolerance` in kcal/mol/Å (none). Of `[output]`, `energy_interval` must not be 0 and must divide `steps`, and `trajectory_interval`, where given, must be a multiple of `energy_interval` and divide `steps`. No thermostat or barostat. In mixed precision the forces are rounded to about 10⁻⁵ of their size, which bounds how far a minimization can go |
| Convergence | With `force_tolerance` (D219), the host compares the largest force of each row of the log, step 0 included, with it, where it reads the forces for the row anyway; at the first row below it the steps, rows, and frames after it are loops of no iteration, and the checkpoint is written at that step. The log says at the end whether it converged |
| Floor | At the same rows the host compares the length of the next step with a floor, $10^{-7}$ Å below double precision and $10^{-11}$ Å in double ($10^{-6}$ and $10^{-10}$ of the cap; D274): at the first row whose step has shrunk below it, and below the first step, the run ends as at convergence, and the log says at the end at which step, the step, the floor, and the largest force. Where trials no longer lower the energy by more than its rounding or noise, they are taken at random and the step falls to 0 (#369) |

On the target of D65 on the CPU in double precision, 500 steps take the
energy from −5348 to −7137 kcal/mol (from the positions of the file before
D120) and the root mean square of the
constrained forces from 8.7 to 2.1 kcal/mol/Å (`test/Driver/minimize.test`
takes 100). Without the masses in the direction, the steps that SHAKE and
SETTLE correct went uphill after 50 steps, and h fell to 0.

## 5. Restraints

`[[restraints]]`, any number of them, hold particles to their positions in
the file of coordinates (D74):

| Item | Rule |
|---|---|
| Energy | $k \lVert\mathbf x - \mathbf x^\text{ref}\rVert^2$ for each selected particle with mass, $k$ in kcal/mol/Å² (`force_constant`), as Amber's `restraint_wt`; the constants of restraints that select the same particle add |
| Reference | The positions $\mathbf r$ of the file of coordinates of `[input]`, also when the run begins from a checkpoint. Under a barostat (D124), $\mathbf x^\text{ref} = \mathbf s \odot \mathbf c + \mathbf o$, with $\mathbf s$ the edges of the cell over those of the file, axis by axis, $\mathbf c$ the center, which follows the cell, and $\mathbf o = \mathbf r - \mathbf c$ the offset, which stays. With `reference_scaling = "CENTER"`, the default, $\mathbf c$ is the mean of the references of every particle so restrained, so that they keep their shape, as a restrained solute should; with `"ALL"`, $\mathbf c = \mathbf r$ and $\mathbf o = 0$, each reference following the cell as the positions do, as restraints spread through the cell need. A particle that two restraints select takes one scaling. The fields hold $\mathbf o$ and $\mathbf c$ (`rest_x`, …, `rest_cx`, …); without a barostat they hold $\mathbf r$ |
| Forces, energy, virial | After the evaluation of the potential and the spreading of the forces of virtual sites, in every step, at the start, and in a minimization: $\mathbf F \mathrel{-}= 2k\mathbf d$ with $\mathbf d = \mathbf x - \mathbf x^\text{ref}$, $U \mathrel{+}= \sum k \lVert\mathbf d\rVert^2$, and $\mathsf W \mathrel{+}= \mathrm{diag}\,\big(\sum -2k\, \mathbf d \odot (\mathbf d + \mathbf o)\big)$, the derivative of the energy when the positions and the centers scale with the cell and the offsets do not; the off-diagonal elements are left out. Without a barostat $\mathbf o$ is taken as 0, and the trace, $-2U$, is exact. The log lists the energy of the restraints among the terms |
| Selection | `selection`, a mask of Amber in part: `:` residues by numbers (from 1) or names, `@` atoms by numbers or names, `:res@atoms`, `*`, with `!`, `&`, `|` and parentheses, in that order of precedence, and `*` and `?` in names. `!:WAT & !@H*` is every heavy atom but those of the waters. A mask that selects no particle with mass is an error; a run from `coordinates` has no names to select by |

On the target of D65, a run at constant energy with the heavy atoms of the
peptide restrained at 10 kcal/mol/Å² keeps the total energy to 4 × 10⁻⁴
over 2 ps at 1 fs, as the same run without them does; at constant
pressure the conserved energy drifts as it does without them, by the term
of Section 2.4.

Scaled about the origin with the cell, as before D124, the references of a
restrained solute shrink with it. Ubiquitin in OPC from the end of stage 3
of its tutorial, whose cell had become 0.926 of the file's in each edge,
had its restrained heavy atoms at a radius of gyration of 11.33 Å, against
11.73 Å for the file: the restraints pushed outward, and 100 ps more at
1 bar held the cell at 200,900 Å³, where a run without restraints went to
196,300 Å³ (the cell is at about −550 bar at 201,264 Å³). With the centers,
the restrained run averages 196,262 ± 580 Å³ over its second half, the free
one 196,293 ± 658: the same within the noise of runs still relaxing. On the
peptide of D65 in double precision, compressed by 16% in 2 ps at
$\tau_P$ = 0.5 ps with its heavy atoms at 100 kcal/mol/Å², the conserved
energy of the Trotter count changes by −10.4 kcal/mol, −8.8 without the
restraints, and −20.2 with the virial of the centers but without
$\mathbf o$; counted exactly, −3.1 and −2.4. On the POPC bilayer of the
tutorials, whose phosphorus atoms are restrained, `"CENTER"` holds the area
of the file over stage 3: 66.4 Å² a lipid over its second half, against
61.8 with `"ALL"`, and the barostat squeezes the height instead, the
pressure of x and y less that of z at 74 ± 11 bar against −1 ± 5.
Restraints spread through the cell take `"ALL"`.

## 6. Leapfrog

Leapfrog does what velocity Verlet does: virtual sites, SETTLE, SHAKE and
RATTLE, restraints, the thermostat, and the barostat (D76). The two store
different velocities and take the same steps. With `h = dt/2`, positions
`x_n`, forces `f_n = F(x_n)`, and `P_x` the projection that takes off the
velocities along the constraints at `x` (the second half of RATTLE):

| | Velocity Verlet | Leapfrog |
|---|---|---|
| Stored velocities | `v_n`, of the time of `x_n` | `v_{n−½}`, half a step behind (`velocity_offset = −0.5`) |
| Carried state | `x_n, v_n, f_n` | `x_n, v_{n−½}, f_n` |
| A step | `u = v_n + h f_n/m`; `x' = x_n + dt u`; constrain `x'` to `x_{n+1}`; `v_{n+½} = u + (x_{n+1} − x')/dt`; `f_{n+1}`; `v_{n+1} = P(v_{n+½} + h f_{n+1}/m)` | `u = v_{n−½} + dt f_n/m`; `x' = x_n + dt u`; constrain `x'` to `x_{n+1}`; `v_{n+½} = u + (x_{n+1} − x')/dt`; `f_{n+1}` |
| A step with energies | The same, with the energy and the virial of `x_{n+1}` | The plain step, to the bit, returning `v_{n+½}` to store and `v_{n+1} = P(v_{n+½} + h f_{n+1}/m)` for the energies; on the side, for the virial of the constraints alone, the drift of velocity Verlet `x'' = x_n + dt (P(v_{n−½} + h f_n/m) + h f_n/m)` |
| A step under Langevin dynamics, or one that scales the cell, with constraints | The step above, with the friction and the noise, or the scaling, in the middle of the drift | `u = P(v_{n−½} + h f_n/m) + h f_n/m`, then as velocity Verlet, in every program of a step |
| Start | `v_0` drawn or read | `v_{−½} = v_0 − h f_0/m` |

Why the steps are the same: without constraints, `v_{n−½} + dt f_n/m` is
velocity Verlet's `v_n + h f_n/m` when `v_n = v_{n−½} + h f_n/m`, which is
what leapfrog's `v_{n−½}` and velocity Verlet's `v_n` are to each other.
With constraints, velocity Verlet's `v_n` is `P(v_{n−½} + h f_n/m)`; the
two drifts differ by `dt (1 − P)(v_{n−½} + h f_n/m)`, which lies along the
directions, weighted by the inverse masses, in which SHAKE and SETTLE move
the particles back. Both are brought to the same point on the surface of
the constraints, to the tolerance of the solvers; the velocities that
follow, `(x_{n+1} − x_n)/dt`, are then the same too. The logs of the two
agree row by row (`test/Driver/leapfrog.test`, and
`test/Driver/leapfrog-constraints.test` with the peptide in OPC water).

This holds while the velocities do not change within the drift. Where
they do, the two forms are different steps. With
`δ = (1 − P)(v_{n−½} + h f_n/m)`, the part along the constraints that the
kick of a whole step keeps, Langevin dynamics (Section 2.6) drifts half
with `u`, takes `c u + …`, and drifts half with that: the drifts of the
two forms differ by `h (1 + c) δ`, still along the constraints, so the
positions agree, but the stored velocities, those after the friction plus
`(x_{n+1} − x')/dt`, differ by `−(1 − c) δ/2`, about `γ dt/2` of `δ`. The
step that scales the cell within its drift (Section 3, D92) scales the
velocities by `1/μ` there, and the same argument gives `(1/μ − 1) δ/2`.
In both, every program of a leapfrog step therefore takes the form of
velocity Verlet, with the projection. A plain step that kicked a whole
step there (#285) left the stored velocities of the dipeptide in water
1.3 × 10⁻³ nm/ps from those of this step after one step at 1/ps and
2 fs, linear in the friction (6.3 × 10⁻⁴ at 0.5/ps, 2.6 × 10⁻³ at 2/ps),
with `δ` up to 1.28 nm/ps, and the positions 2.9 × 10⁻⁵ nm from those of
velocity Verlet after 20 steps; they are now within 2.9 × 10⁻¹⁴ nm. A
NumPy reference of the step of velocity Verlet with a SHAKE and a RATTLE
of its own agrees with the plain step and with the step of energy within
4 × 10⁻¹⁵ nm and 2 × 10⁻¹² nm/ps (`test/Driver/leapfrog-langevin-reference.test`).

Every program of a step leaves the same state, to the bit in the
deterministic mode: the plain step, the step with energies, the steps
with the virial of the barostats, and the scaling step with and without
the energies. Two runs that differ only in the rows of their energies end
in the same state (`test/Driver/leapfrog-energy-step.test`, for each
thermostat, each count of the work of the barostat, and each coupling of
the cell on the CPU in double precision; on the CPU in mixed precision
and on a GPU in double and mixed precision for one case of each of the
three forms of the step by default, and for all of them in the suite of a
release, `lit -Drelease=1`). Outside
Langevin dynamics and the scaling step, the step with the virial would
differ from the plain step by the rounding of the solvers if it drifted
from the projected velocities; it drifts as the plain step does and takes
the other drift on the side.

What each quantity is taken from:

| Quantity | Velocity Verlet | Leapfrog |
|---|---|---|
| Energy, virial, pressure in the log | $\mathbf x_{n+1}$ and $\mathbf v_{n+1}$ of the step with energies | The same $\mathbf x_{n+1}$ and $\mathbf v_{n+1}$ |
| Virial of the constraints | $\tfrac12$ of the impulses of the positions ($\mathbf G = 2m\boldsymbol\Delta/\Delta t^2$ for $\mathbf x' = \mathbf x_n + \Delta t\, \mathbf v_n + \Delta t^2 \mathbf f_n/2m$) and $\tfrac12$ of those of the velocities ([constraints.md](constraints.md), Section 1) | The same: $\boldsymbol\Delta = \mathbf x_{n+1} - \mathbf x''$ from the drift $\mathbf x''$ of velocity Verlet, from $P(\mathbf v_{n-1/2} + h \mathbf f_n/m)$, which the step with the virial takes on the side: what the constraints added to its own drift, plus what that drift is ahead of $\mathbf x''$. The plain step computes no virial |
| Thermostat alone | Scales $\mathbf v_n$, with its kinetic energy | Scales $\mathbf v_{n-1/2}$, with its kinetic energy, as GROMACS does with leapfrog |
| Barostat (and the thermostat with it) | Pressure from $\mathbf v_{n+1}$; scales $\mathbf x_{n+1}$ by $\mu$ and $\mathbf v_{n+1}$ by $\alpha/\mu$; carries $\mathbf f' = \mathbf F(\mathbf x'_{n+1})$ (D77) | Pressure from $\mathbf v_{n+1}$; scales $\mathbf x_{n+1}$ by $\mu$ and $\mathbf v_{n+1}$ by $\alpha/\mu$; carries $\mathbf f'$ and stores $\mathbf v'_{n+1/2} = \mathbf v'_{n+1} - h \mathbf f'/m$, so that the next step is that of velocity Verlet |

The barostat couples the velocities of the time of the positions: scaling
the stored $\mathbf v_{n+1/2}$ by $1/\mu$ would leave the half kick $h \mathbf f_{n+1}/m$ in
$\mathbf v_{n+1}$ unscaled, and the kinetic energy of the log would change by
$(1/\mu - 1)\, h \sum \mathbf v_{n+1/2} \cdot \mathbf f_{n+1}$ more than the work that the barostat
counts. That term follows $dU/dt$; while the barostat compresses the target
of D65 by a fifth over 10 ps, the conserved energy drifted by 4.3 × 10⁻³
of its value with it, and by 1.3 × 10⁻³ without it, as with velocity
Verlet (1.4 × 10⁻³), with the work counted to first order. With the exact
work (D77) both integrators carry the forces of the scaled positions, and
without constraints the trajectory of leapfrog at constant pressure is that
of velocity Verlet to the bit (`test/Driver/barostat.test`).

The stored velocities are half a kick behind the ones the coupling leaves
without the motion of the center of mass. The forces of PME do not sum to
zero (their interpolation does not conserve momentum), so the momentum of
the stored velocities is $-h \sum \mathbf f$, 10⁻² amu nm/ps on the target of D65,
where that of velocity Verlet is at the rounding.

A checkpoint of leapfrog holds the forces, as one of velocity Verlet does,
and a run continues it exactly. A checkpoint of leapfrog written before
this holds none and is refused with a message.
