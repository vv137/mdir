# Dynamics, Minimization, and Restraints

Status: kept current; last checked against the code on 2026-10-22. The
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

*A design, partly built (D326, #541).* Step 2a has built the ops of the
thermostats and of the removal of the motion of the center of mass in
`include/mdir/Dialect/Dyn/DynOps.td` (`dyn.momentum`, `dyn.kinetic`,
`dyn.half_step_excess`, `dyn.velocity_rescaling`, `dyn.nose_hoover`,
`dyn.shift`, `dyn.scale`, `dyn.noise`, `dyn.to_bath`), the effects of
`dyn.step`, the rules of Section 2.2.4 that need no op of the barostat,
and the pass `dyn-expand-couplings`; the driver does not write the ops
yet. The ops formalize what the driver writes today. They add no
thermostat or barostat, change no key of the control file and no default,
and change no number. The rest of steps 2 to 4 of #541 builds them
(Section 2.2.9). Until step 3 Sections 1, 2.3, 2.4, 2.6, and 6 describe
what runs.

#### 2.2.1 What runs today

`lib/Driver/Builder.cpp` writes the couplings as text into the schedule of
`@mdir_run`, at the end of each period of coupling (`emitCoupling`,
`emitStrain`, and the closing steps in `emitLevel`):

| What | How it is written |
|---|---|
| Sums over the particles | `md.sum_particles`: the momentum $\sum_i m_i\mathbf v_i$, the kinetic energy of each axis, and $\tfrac{\Delta t^2}{8}\sum_i F_{i,a}^2/m_i$; with `md.sum_tuples` over the rigid groups, the kinetic energy of the centers of the groups (`emitGroupKinetic`) |
| The factor of the thermostat | `func.call @mdrtBussiFactor` (`runtime/mdrt.c`, `libmdrt`) or `@mdrtNoseHooverFactor` (`lib/Driver/Output.cpp`), on the host |
| The strain of the barostat | Arithmetic on the host for the pressure, then `func.call @mdrtBarostatStrain`, `@mdrtBarostatStrainArea` and `@mdrtBarostatStrainHeight`, or `@mdrtBarostatStrainAxis` once for each axis (`libmdrt`) |
| The cell | `memref.store` into `%box_memory`, a buffer of the host that each iteration of a loop loads (D72), and `func.call @mdrtSetBox` and `@mdrtSetTilt` for the log, the trajectory, and the checkpoint |
| The scaling | One `md.map_particles` that subtracts the velocity of the center of mass, multiplies by $\alpha$, and multiplies by $1/\mu_a$, in that order; the positions by `emitGroupScaling`, rigid groups by their centers through `md.gather_tuples` |
| The energy of the bath | Arithmetic on the host, then `func.call @mdrtAddBath` |
| Langevin and Brownian dynamics | Inside every program of a step: Philox as `arith` ops in an `md.map_particles` (`emitNoise`) |
| The barostat of Trotter type | Two closing steps, `dyn.step @step_virial` and `dyn.step @step_trotter(…, %mu)`, the second of which scales in the middle of its drift; with a period of one step, the state of the pressure in `%trotter_memory` and `func.call @mdrtSetBarostatState` |

A pass therefore sees sums and calls, and not what they are for: which
sums a coupling needs, which state of the host it reads and writes, and
whether the steps of a period compute a sum. `dyn.step` is `Pure`, though
the programs it applies hold host calls (`mdrtDeferStops`,
`mdrtStopConstraintFailure`). `requires` and `provides` of `dyn.program`
are checked against a list of names and read by no pass, and every
program of a step but those of Brownian dynamics claims `symplectic` and `time_reversible`,
those with the friction and the noise of Langevin dynamics and
`step_trotter` included.

#### 2.2.2 Principles

1. **Each op lowers to the operations that the driver writes today, on the
   same operands, in the same order of dependence.** An op returns the
   intermediate values that today's arithmetic takes, not a recombination
   of them: $E = \tfrac{\Delta t^2}{8}\sum F^2/m$ rather than
   $K_\text{half} = K + E$, since $K + \tfrac23\big((K + E) - K\big)$ need
   not round as $K + \tfrac23E$ does. The chain of the velocities (the
   center of mass subtracted, then $\alpha$, then $1/\mu_a$) stays one loop
   with the same order of products, since $(\mathbf v\alpha)\mu^{-1}$ and
   $\mathbf v(\alpha\mu^{-1})$ need not round alike. This keeps a run in
   the deterministic mode bitwise identical (Section 2.2.8).
2. **The random numbers are drawn inside the op that takes them.** The old
   proposal's `dyn.random` is dropped: the factor of stochastic velocity
   rescaling draws a gamma number by rejection
   [[Bussi2007]](references.md#bussi2007), so the count of its draws is
   not known before, and the key of A13 is fixed by the op, its step, and
   the stream of its kind (Section 2.2.6).
3. **The state of the host stays on the host**: the cell (D72), the
   Nosé–Hoover chain (D163a), the energy of the bath (D129), and the state
   of the pressure of a barostat of Trotter type that scales every step
   (D92). An op that reads or writes one declares it as an effect on a
   resource of the dialect, so that passes keep the order of those ops and
   do not remove them.
4. **The schedule does not change.** The loops of `setSchedule` stay as
   they are (Section 2.2.5); the ops take the places of the text that
   `emitCoupling` and the closing steps write.

#### 2.2.3 The ops

`!vec` is a field of 3 × f64 on a particle set, `!real` a scalar field on
it, `!ids` the numbers of the particles in the input, and `!md.relation`
the tuples of a set of rigid groups (a water of SETTLE, a cluster of
SHAKE). Every field operand of an op is on one particle set, as the
verifiers of `dyn.kick` and `dyn.drift` require. A *global sum* is a sum
over the particles, tuples, or pairs whose result the host reads; on a
device it is a reduction and a wait for its result (ops.md, Section 10.8).

**Measurements.** Pure; each needs global sums and lowers to
`md.sum_particles`, with `md.sum_tuples` over the groups, which run as
loops on the device and reductions.

| Op | Operands → results | Value |
|---|---|---|
| `dyn.kinetic` | `%v: !vec, %m: !real` → `vector<3xf64>` or f64 | $K_a = \tfrac12\sum_i m_iv_{i,a}^2$, or $K = \sum_i \tfrac12\big(m_i(\mathbf v_i\cdot\mathbf v_i)\big)$; one sum. The two round differently, and the driver takes the axes with a barostat and the total without one (`emitKineticVector`, `emitKineticEnergy`), so the op has both (step 2a) |
| `dyn.momentum` | `%v, %m` → `vector<3xf64>` | $\mathbf P = \sum_i m_i\mathbf v_i$; one sum |
| `dyn.half_step_excess` | `%f: !vec, %m, %dt: f64` → `vector<3xf64>` | $E_a = \tfrac{\Delta t^2}{8}\sum_i F_{i,a}^2/m_i$, a particle of mass 0 left out: $K_{\text{half},a} - K_a$, exact without constraints [[Jung2018]](references.md#jung2018); one sum |
| `dyn.group_kinetic` | `%x, %v, %f, %m, %cell: !md.cell, %dt`, `groups(%r: !md.relation, …)`, optional `center(%M: vector<3xf64>)` → three `vector<3xf64>` | $K_{g,a}$, $E_{g,a} = \tfrac{\Delta t^2}{8}\sum_g F_{g,a}^2/M_g$, and $K_{\text{half},g,a} = K_{g,a} + E_{g,a}$ of the centers of the groups (D264, [[Jung2020]](references.md#jung2020)). With `center`, the total mass, $K_g$ is without the motion of the center of mass, as when the run removes it. Two sums over the particles, two over the tuples of each set of groups, and one more with `center` |

The verifier of `dyn.group_kinetic` requires each relation to be a set of
groups of the particle set of the fields; a particle in no relation is a
group of one.

**Arithmetic of a coupling.** Pure, on scalars and `vector<3xf64>` of the
host; no sum. They lower to `arith` and `math` ops on the host.

| Op | Operands → results | Value |
|---|---|---|
| `dyn.pressure` | `%k: vector<3xf64>, %w: vector<3xf64>`, the edges `%L0, %L1, %L2: f64`, `%c: f64` → `vector<3xf64>` | $P_{aa} = c_\text{bar}\,(2K_{\text{half},g,a} + W_{g,aa} + C/(3V))/V$ in bar, with $V = (L_0L_1)L_2$, $C$ the virials of the constant terms times the volume, and $c_\text{bar} = 16.6053906717$ bar nm³ mol/kJ (D264) |
| `dyn.scale_cell` | the edges, and with a triclinic cell the tilts $b_x, c_x, c_y$ (3 or 6 × f64), `%mu: vector<3xf64>` → as many f64 | $H\,\mathrm{diag}(\boldsymbol\mu)$: each edge by its $\mu_a$, $b_x$ and $c_x$ by $\mu_x$, $c_y$ by $\mu_y$ (D127) |
| `dyn.scaling_work` | `kinetic(%k)`, `strain(%mu)` or `log_strain(%e)`, `virial_before(%wb)`, optional `virial_after(%wa)`, optional `energies(%u, %u1: f64)`, `constant(%c)`, `volumes(%V, %V1: f64)`; attribute `count` → f64 | $\Delta E_\text{sys}$, the energy that a scaling gives the system (Section 2.2.7) |

The verifier of `dyn.scaling_work` requires, for each `count`, exactly the
operands that its formula takes: `trotter` the log strain and both
virials; `trotter_before` the log strain and the virial before; `exact`
the strain and the energies; `first_order` the strain and the virial
before. `count` is how the work is counted, not the key `work`:
`TROTTER_FIRST_ORDER` with a period of one step counts as `trotter`, since
the next scaling takes the virial after the scaling anyway
(`countsAfterScaling`, D92).

**Draws.** Each takes the seed as an `i64` attribute and the number of the
step as an `i64` operand, and draws from its own stream (Section 2.2.6).
They lower to calls into `libmdrt` on the host.

| Op | Operands → results | Value | Effects |
|---|---|---|---|
| `dyn.velocity_rescaling` | `%k: f64, %step: i64`; attributes `seed`, `target_kinetic` $\bar K = \tfrac12N_fk_BT_0$, `freedom` $N_f$, `decay` $e^{-N_T\Delta t/\tau_T}$ → f64 | The factor $\alpha$ of [[Bussi2007]](references.md#bussi2007) (its appendix) for the kinetic energy $K$; 1 if $K \le 0$; `mdrtBussiFactor`, stream 0 | Pure: a function of its operands and attributes, since Philox is counter-based. The verifier requires $N_f > 0$, $\bar K > 0$, and $0 < $ `decay` $< 1$ |
| `dyn.cell_rescaling` | `%p: vector<3xf64>` (bar), the edges `%L0, %L1, %L2`, `%step: i64`; attributes `coupling` (`isotropic`, `semi_isotropic`, `anisotropic`), `seed`, `target` $P_0$ (bar), `compressibility` (one number, two for x and y and for z, or three; 1/bar), `rate` $N_P\Delta t/\tau_P$, `kT` (kJ/mol), `tension` $\gamma$ (bar nm, semi-isotropic only) → `vector<3xf64>` | The change $\Delta\varepsilon_a$ of the strain of each axis over the period [[Bernetti2020]](references.md#bernetti2020): isotropic, a third of $\Delta\varepsilon$ of the step of $\lambda = \sqrt V$, eq. (S7), from the mean of the three pressures (`mdrtBarostatStrain`); semi-isotropic, half of $\Delta\varepsilon_{xy}$ to x and y and $\Delta\varepsilon_z$ to z, eqs. (9a, 9b) (`mdrtBarostatStrainArea`, `mdrtBarostatStrainHeight`, D119); anisotropic, $\Delta\varepsilon_a$ of each axis (`mdrtBarostatStrainAxis`, D163c); stream 1 | Isotropic: pure. Semi-isotropic and anisotropic: writes the statistics of the pressures of the axes that the log reports at the end (`semiPressures`, `axisPressures` of `runtime/mdrt.c`). The verifier requires the count of `compressibility` of the coupling, not every compressibility 0 when anisotropic, `tension` only when semi-isotropic, `rate > 0`, and `kT > 0` |

**Thermostat with state.**

| Op | Operands → results | Value | Effects |
|---|---|---|---|
| `dyn.nose_hoover` | `%k: f64, %step: i64`; attributes `temperature` $T_0$, `time_constant` $\tau_T$, `chain_length` $M$, `period` $N_T\Delta t$ (ps), `freedom` $N_f$ → f64 | The factor of the velocities from one action of the chain over the period [[Martyna1992]](references.md#martyna1992), [[Martyna1996]](references.md#martyna1996) (D163a); `mdrtNoseHooverFactor` of `lib/Driver/Output.cpp`, on the host | Reads and writes the chain, $2M$ numbers of the host that the checkpoint keeps (`/parameters/mdir/thermostat_state`); stops the run when the factorization no longer follows the chain (D206). The verifier requires $M \ge 1$, $\tau_T > 0$, $T_0 > 0$, $N_f > 0$ |

The step operand of `dyn.nose_hoover` only names the step in the message
of a stop. Its attributes are those that `Output::setChain` takes from the
control file today; step 3 checks that the two agree.

**Updates of the particles.** Pure; each lowers to an `md.map_particles`
(and `md.gather_tuples` for groups), a loop on the device with no sum.

| Op | Operands → results | Value |
|---|---|---|
| `dyn.shift` | `%v: !vec, %u: vector<3xf64>` → `!vec` | $\mathbf v_i - \mathbf u$; with $\mathbf u = \mathbf P/M$ the removal of the motion of the center of mass |
| `dyn.scale` | `%v: !vec, %s: f64` or `vector<3xf64>` → `!vec` | $s\,\mathbf v_i$, or $\mathbf s\odot\mathbf v_i$ |
| `dyn.scale_positions` | `%x: !vec, %mu: vector<3xf64>, %m: !real, %cell`, `groups(%r: !md.relation, …)` → `!vec` | $\boldsymbol\mu\odot\mathbf x_i$ for a particle alone; each rigid group moves with its center of mass $\mathbf X_g$ to $\boldsymbol\mu\odot\mathbf X_g$ and keeps its shape (D72) |
| `dyn.noise` | `%v: !vec, %m: !real, %step: i64, %ids: !ids` (a field of i32); attributes `seed`, `decay` $c$, `spread` $\sigma$ → `!vec` | $c\,\mathbf v_i + (\sigma/\sqrt{m_i})\,\mathbf R_i$, with $\mathbf R_i$ three normal numbers of stream 2 of particle $i$ and the step; a particle of mass 0 keeps $\mathbf v_i$. Philox as `arith` ops in the loop, as `emitNoise` writes it |

A chain of `dyn.shift` and `dyn.scale` on one field whose intermediate
values have no other use lowers to one `md.map_particles` that applies
them in order, as the driver writes it today; no canonicalization folds two
scales into one. `dyn.noise` is the O step of Langevin dynamics with
$c = e^{-\gamma\Delta t}$ and $\sigma = \sqrt{(1 - c^2)k_BT_0}$
[[Leimkuhler2013]](references.md#leimkuhler2013), and the noise of
Brownian dynamics with $c = 1$ and $\sigma = \sqrt{2k_BT_0/(\gamma\Delta t)}$
on the velocity of the drift $\mathbf F_i/(m_i\gamma)$
[[Ermak1978]](references.md#ermak1978) (D163b). The attributes hold the
numbers that the driver computes for `emitNoise` today, so that the
constants of the kernel keep their bits.

**State of the host.** Each lowers to `memref` ops on a buffer of the host
and, where the log or the checkpoint keeps the value, a call into the
driver's runtime (`lib/Driver/Output.cpp`).

| Op | Operands → results | Effects |
|---|---|---|
| `dyn.load_cell` | → 3 or 6 × f64, the edges and the tilts | Reads the cell (`%box_memory`) |
| `dyn.store_cell` | 3 or 6 × f64 | Writes the cell, and the box of the log, the trajectory, and the checkpoint (`mdrtSetBox`, `mdrtSetTilt`); stops a run whose cell is not finite or changed by more than a factor of 2 along an axis at one coupling (#168, D225) |
| `dyn.load_pressure_state` | → three `vector<3xf64>`: the diagonals of $\mathsf W$ and of $\mathsf W_g$, and $K_{\text{half},g}$ after the thermostat | Reads `%trotter_memory` |
| `dyn.store_pressure_state` | the same three | Writes `%trotter_memory` and the checkpoint's `/parameters/mdir/barostat_state` (`mdrtSetBarostatState`) |
| `dyn.to_bath` | `%e: f64` | Adds $-\Delta E_\text{sys}$ to $E_\text{bath}$ (`mdrtAddBath`), which the conserved energy of the log and the checkpoint's `bath` take (D129) |

The pair `dyn.load_pressure_state` and `dyn.store_pressure_state` appears
only under the barostat of Trotter type with a period of one step (D92).
The verifier of `dyn.load_cell` and `dyn.store_cell` requires 3 or 6
values; every op of a module takes the same count, that of the cell of the
run.

`dyn.step` loses `Pure`: it takes the effects of the ops of the program it
applies, so that a step whose program holds `dyn.store_pressure_state` or a
stop is not moved across another. As built (step 2a), `dyn.step` collects
them itself from the body of its program: an effect on a value of the body
becomes one on its resource, and an op of unknown effects, such as the
`func.call` of `mdrtDeferStops`, reads and writes every resource. The
resources of step 2a are `dyn.bath`, `dyn.chain`, and `dyn.stop`; the
lowered programs of the corpus of #606 are those of before (Section
2.2.9).

#### 2.2.4 The contract of `dyn.program`

`requires` names what the bath gives the ops of a program, and `provides`
what the program claims. Once the ops exist, the verifier checks the
claims against the body, in both directions where a name says what the
body holds:

| Name | Rule |
|---|---|
| `requires = ["temperature"]` | If and only if the body has a `dyn.noise`; built: if |
| `requires = ["pressure"]` | If and only if the body has a `dyn.scale_positions`, whose strain the barostat computes from the pressure (`step_trotter`) |
| `provides = ["thermostatting"]` | If and only if the body has a `dyn.noise`; built: if |
| `provides = ["barostatting"]` | If and only if the body has a `dyn.scale_positions` |
| `provides = ["symplectic", "time_reversible"]` | Only if the body has neither: friction, noise, and a scaling within the drift are neither. Neither integrator conserves energy exactly [[Hairer2003]](references.md#hairer2003), and neither claims to (ops.md, Section 6.3); built for `dyn.noise` |
| `provides = ["no_global_sums"]` (new) | Only if the body computes no global sum: no op of the measurements above, no `md.sum_particles`, `md.sum_tuples`, or `md.sum_relation` with a floating-point result, and no `md.evaluate` that requests the energy, the virial, or a derivative; built |

The rules of the barostat wait for `dyn.scale_positions` (step 2b). Of the
others, step 2a checks the direction from the body to the claim. The
converse would refuse the driver's programs until step 3a writes
`dyn.noise`: the programs of Brownian dynamics claim `thermostatting` and
hold their noise as `arith` ops. A program also keeps its claims when
`dyn-expand-couplings` expands the ops that they rest on, which a check
from the claim to the body would refuse after the expansion.

`no_global_sums` allows what a plain step reduces today and does not wait
for: the flags of the constraints, an `i1` that `md.gather_tuples` yields
and that is exact in any order (D278), and the test of validity of the
neighbor structures within `md.evaluate`. The driver gives it to `@step`,
the program of the steps between couplings, so that D67's rule (a step
between couplings computes no global sum) is checked rather than kept by
the schedule alone.

Today's programs would change their claims: those of Langevin dynamics
take `thermostatting` instead of `symplectic` and `time_reversible`, and
`step_trotter` takes `barostatting` and `pressure`. Nothing reads these
attributes, so the change alters the text of the IR and no number.

#### 2.2.5 The schedule

The loops stay those of `setSchedule`. A period of coupling of $N_T$
steps, `interval` of `[thermostat]` (`getCouplingPeriod`, the interval of
`center_of_mass_interval` without a thermostat), is a loop of
$N_T - k$ plain steps, `dyn.step @step`, then $k$ closing steps and the
coupling, with $k = 2$ under the barostat of Trotter type unless the
period is one step, and $k = 1$ otherwise (`getClosingSteps`). The loop
over the periods of an interval between energies leaves the last period
to the interval, whose step of energy closes it. `interval` is resolved as
`resolveControlCoupling` does it today: 10 by default with a thermostat
that rescales, `center_of_mass_interval` equal to it or 0; under Langevin
dynamics it paces the removal of the motion of the center of mass and the
barostat, 10 by default with a barostat and none otherwise; the barostat's
`interval` equals the thermostat's; `steps` and every interval of output
are multiples of it.

The step operand of the draws is the step that ends the period,
`nestStart` plus the steps the loops have done (`emitStep`), counted from
the first step of the run (D129), in a Python simulation as well, whose
periods are counted from its first step (D196). The kernels of Langevin and
Brownian dynamics take the number of the step from the counter on the host
that every step advances (`%noise_memory`). Both are absolute, so a
trajectory depends neither on how the steps are grouped into loops nor on
a restart (D67).

The period maps onto the ops as follows (velocity Verlet; with leapfrog
the coupling takes the velocities of the time of the positions where the
barostat acts, and stores them half a kick behind by a `dyn.kick` of
$-\Delta t/2$; the thermostat alone scales the stored velocities, with
$K_T$ from those of the time of the positions, Section 6):

1. **Without a barostat.** The closing step is `@step`, or with
   constraints and a thermostat that rescales `@step_thermostat`, which
   also returns the measured $K_\text{half}$ (D263). Then
   `dyn.momentum` (with a removal), `dyn.kinetic`, $K_T$ from
   `dyn.half_step_excess` or the measured $K_\text{half}$,
   `dyn.velocity_rescaling` or `dyn.nose_hoover`, `dyn.shift` and
   `dyn.scale`, and `dyn.to_bath`.
2. **The barostat with `work = "EXACT"` or `"FIRST_ORDER"`.** The closing
   step is `@step_energy`, or `@step_coupling` with constraints, which
   returns the energy, the virial, and the diagonal of $\mathsf W_g$. Then
   the measurements, the factor of the thermostat, `dyn.group_kinetic`,
   `dyn.load_cell`, `dyn.pressure`, `dyn.cell_rescaling`, $\mu_a =
   e^{\Delta\varepsilon_a}$, `dyn.scale_cell`, `dyn.store_cell`,
   `dyn.scale_positions`; with `"EXACT"` the sites placed on the scaled
   positions and `md.evaluate` in the new cell, with the restraints and
   the learned potentials, as today (D77); `dyn.scaling_work`,
   `dyn.to_bath`, and the velocities by `dyn.shift`, `dyn.scale` by
   $\alpha$, and `dyn.scale` by $1/\mu_a$.
3. **The barostat of Trotter type.** The first closing step is
   `@step_virial`. Then `dyn.group_kinetic`, `dyn.load_cell`,
   `dyn.pressure`, `dyn.cell_rescaling`, `dyn.scale_cell`, and
   `dyn.store_cell`; the second closing step `dyn.step @step_trotter(…,
   %cell1, %dt, %mu)` in the new cell, whose program holds
   `dyn.drift` of $\Delta t/2$, `dyn.noise` under Langevin dynamics,
   `dyn.scale_positions`, `dyn.kinetic` of the velocities it scales
   ($K_{1/2,a}$), `dyn.scale` by $1/\mu_a$, and the second `dyn.drift`;
   then `dyn.scaling_work` with `count = "trotter"` or `"trotter_before"`,
   the thermostat as in item 1, and `dyn.to_bath`. With a period of one
   step `@step_virial` is not taken: `dyn.load_pressure_state` gives the
   pressure of the state that the step before left, and
   `dyn.store_pressure_state` keeps that of this one. At the end of an
   interval between energies `@step_trotter_energy` takes the place of
   `@step_trotter`.

The closing steps stay two `dyn.step`, with the ops of the strain between
them in `@mdir_run`: the strain is already an operand of `step_trotter`,
so a pass sees what depends on it (open question 1 of the PR asks whether
a region should hold them).

The cell stays a buffer of the host: every loop loads it at the start of
each iteration and after a loop of periods (`dyn.load_cell`), the neighbor
structures test their validity against it (D80), and restraints take their
references from it (D124). A sketch of a period of velocity Verlet with
stochastic velocity rescaling and the removal of the motion of the center
of mass, without constraints:

```mlir
%xk, %vk, %fk = dyn.step @step(%x, %v, %f, %m, %cell, %dt)
    : (!vec, !vec, !vec, !real, !md.cell, f64) -> (!vec, !vec, !vec)
%p    = dyn.momentum %vk, %m : !vec -> vector<3xf64>
%vcm  = arith.divf %p, %total_mass : vector<3xf64>
%k    = dyn.kinetic %vk, %m : !vec -> vector<3xf64>
%e    = dyn.half_step_excess %fk, %m, %dt : !vec -> vector<3xf64>
// K_T = K + (2/3) E without the kinetic energy of the center of mass, as
// emitOptimal writes it; K_t, the kinetic energy without it.
%alpha = dyn.velocity_rescaling %kopt, %step
    {seed = 314159 : i64, target_kinetic = 1.1e+04, freedom = 8.9e+03,
     decay = 0.99} : f64
%v1 = dyn.shift %vk, %vcm : !vec
%v2 = dyn.scale %v1, %alpha : !vec, f64
// The bath takes K_cm + (1 - alpha^2) K_t, which the arithmetic of
// emitCoupling gives as %heat.
dyn.to_bath %heat : f64
```

#### 2.2.6 Random numbers

The key is that of A13 and Section 2.3, unchanged: Philox 4×32-10
[[Salmon2011]](references.md#salmon2011) under the seed, with the counter
$(\text{step}, \text{entity}, 2^{24}\,\text{stream} + \text{block})$, the
block counting the calls of Philox of one entity in one stream and step
(the draw index of A13). The op names the stream by its kind:

| Op | Stream | Entity | Draws, in order of the blocks | Where |
|---|---|---|---|---|
| `dyn.velocity_rescaling` | 0 | 0 | A normal number, then the gamma number of $\chi^2_{N_f-1}$ by rejection [[Marsaglia2000]](references.md#marsaglia2000) | Host (`libmdrt`) |
| `dyn.cell_rescaling`, isotropic | 1 | 0 | Normal number 0 | Host |
| `dyn.cell_rescaling`, semi-isotropic | 1 | 0 | Normal numbers 0 (the area) and 1 (the height); each of the two calls draws both | Host |
| `dyn.cell_rescaling`, anisotropic | 1 | 0 | Normal number $a$ for axis $a$ | Host |
| `dyn.noise` | 2 | The number of the particle in the input | One block: four words, two pairs of Box–Muller, three numbers taken | Device or CPU loop, in the kernel |

`dyn.nose_hoover` draws nothing. A normal number on the host takes two
uniform numbers of 53 bits, a block of four words, so that the normal
number $n$ is that of block $n$; the cosine branch of Box–Muller is taken.
Since the ops lower to the same calls with the same arguments, they draw
what the driver draws today.

#### 2.2.7 What each coupling covers, and its conserved energy

The conserved energy of the log is $\tilde H = U + K + E_\text{bath}$
(D129; white paper, Section 6.5); each coupling gives the system
$\Delta E_\text{sys}$, and `dyn.to_bath` adds $-\Delta E_\text{sys}$ to
$E_\text{bath}$. $K_\text{cm} = \tfrac12\mathbf P\cdot\mathbf V_\text{cm}$
is the kinetic energy of the center of mass, $K_t = K - K_\text{cm}$, and
$K_T = K + \tfrac23(K_\text{half} - K)$ the kinetic energy of the optimal
temperature [[Jung2019]](references.md#jung2019).

| Coupling | The ops | $-\Delta E_\text{sys}$ to the bath |
|---|---|---|
| Removal of the motion of the center of mass (D60, D67) | `dyn.momentum`, `dyn.shift` | $K_\text{cm}$ |
| `V-RESCALE` (D50, D67, D263) | `dyn.kinetic`, `dyn.half_step_excess` or the measured $K_\text{half}$, `dyn.velocity_rescaling` of $K_T - K_\text{cm}$, `dyn.scale` by $\alpha$ | $K_\text{cm} + (1 - \alpha^2)K_t$, exactly, what the scaling takes |
| `NOSE-HOOVER` (D163a, D206, D263) | As `V-RESCALE`, with `dyn.nose_hoover` | The same: since D263 the bath counts the change of $K$ that the scaling makes, not the energy of the chain |
| `LANGEVIN` (D135, D265) | `dyn.noise` in every program of a step, by g-BAOAB with constraints; `interval` paces the removal and the barostat | $K_\text{cm}$ and the work of the barostat; the log shows no conserved energy, since the friction and the noise are not counted |
| `BROWNIAN` (D163b) | `dyn.noise` with $c = 1$ on the velocity of the drift, in every step; no coupling at the end of a period, no `[thermostat]`, `[barostat]`, or `center_of_mass_interval` | None; no conserved energy |
| `C-RESCALE`, `work = "TROTTER"` (D92, D116) | `@step_virial`, `dyn.cell_rescaling`, `@step_trotter`, `dyn.scaling_work` with `count = "trotter"` | $-\big[\sum_a(\mu_a^{-2} - 1)K_{1/2,a} - \sum_a\ln\mu_a\,\tfrac12\big(W_{b,aa} + \tfrac{C}{3V} + W_{a,aa} + \tfrac{C}{3V'}\big)\big]$, with $K_{1/2}$ that of the velocities scaled in the middle of the step and $\mathsf W_b$, $\mathsf W_a$ the virials of the groups from the evaluations before and after the scaling; then the thermostat's |
| `C-RESCALE`, `"TROTTER_FIRST_ORDER"` (D92) | The same, `count = "trotter_before"`; `@step_trotter` computes no virial | $-\big[\sum_a(\mu_a^{-2} - 1)K_{1/2,a} - \sum_a\ln\mu_a\,\big(W_{b,aa} + \tfrac{C}{3V}\big)\big]$; with a period of one step, as `"TROTTER"` |
| `C-RESCALE`, `"EXACT"` (D77) | `dyn.scale_positions`, the evaluation in the new cell, `dyn.scaling_work` with `count = "exact"` | $-\big[U(\mathbf x') - U(\mathbf x) + C(1/V' - 1/V) + \sum_a(\mu_a^{-2} - 1)\,\alpha^2K_{t,a}\big]$, added to the thermostat's |
| `C-RESCALE`, `"FIRST_ORDER"` (D72), isotropic only | `dyn.scale_positions`, `dyn.scaling_work` with `count = "first_order"` | $-\big[-\sum_a(\mu_a - 1)\tfrac13(\mathrm{tr}\,\mathsf W_g + C/V) + \sum_a(\mu_a^{-2} - 1)\,\alpha^2K_{t,a}\big]$, added to the thermostat's |

Isotropic, semi-isotropic, and anisotropic coupling (D72, D119, D163c)
differ only in `coupling` of `dyn.cell_rescaling`: every other op is axis
by axis already, with $\mu_a$ equal on the three axes, on x and y, or on
none, and a semi-isotropic `tension` enters the strain of the area. The
barostat takes the pressure of the groups that it scales, from the
kinetic energy of their centers at the half steps, without the motion of
the center of mass where the run removes it, $K_{\text{half},g}$ of
`dyn.group_kinetic`, and the
virial of the groups that the closing step returns (D264). The thermostats
take $K_T$ at the step that closes the period: from the forces without
constraints, and with constraints from the $K_\text{half}$ that the closing
step measures (`@step_thermostat`, `@step_coupling`, `@step_trotter`;
D203, D263). They scale every velocity, so the constraints of the
velocities hold.

The two rows of the old proposal that no longer held are now, by the
code:

| Item | Rule |
|---|---|
| The kinetic energy of the thermostat | $K_T - K_\text{cm}$, of the optimal temperature (D263), measured from the half steps around the step that closes the period: from the forces without constraints ($K + \tfrac23E$), measured with them. The factor scales $K_t$, and the bath counts the change of $K_t$ exactly. `K_T` of D45 is this estimate |
| The pressure of the barostat | That of the groups that it scales, $P_{aa}$ of `dyn.pressure`, with $K_{\text{half},g}$ of their centers at the half steps, without the motion of the center of mass where the run removes it, and $\mathsf W_g$ (D264); the log takes the same pressure |
| The pressure, its corrections | The virials of the correction for the dispersion and of the background of a net charge, $C/(3V)$ on each axis ([nonbonded.md](nonbonded.md), Section 1.2); the forces of the constraints sum to zero within a group and do not enter |
| How often | Every $N_T$ steps, with the period $N_T\Delta t$; a step between computes no global sum (`no_global_sums`). On a device a step with global sums takes up to twice the time of one without (ops.md, Section 10.8) |
| Groups with a thermostat each | One group in M1 |

#### 2.2.8 Existing inputs and checkpoints

**Bitwise in the deterministic mode.** Every op lowers to the operations
that the driver writes today, on the same operands and in the same order
of dependence (Section 2.2.2); the host calls keep their order among
themselves, which the effects of Section 2.2.3 enforce. Every value is
then computed by the same floating-point operations on the same inputs,
and the trajectories, logs, and checkpoints are those of `main`. The
lowering is a pass, `dyn-expand-couplings`, placed after `md-inline` and
before `md-bypass-updates`, so that every later pass sees what it sees
today; the code of the expansion is moved from `Builder.cpp`, not
rewritten. Data movement may differ (a `vector.extract` where the driver
had a scalar), never an arithmetic operation. `mdir emit
--stage=pipeline` and `--stage=lowered` print the module at a named
stage, so the module after the expansion can be compared with that of
`main`. The risks are the
chain of the velocities, which must stay one loop in today's order, and a
fusion of loops that sees other neighbors (#356); both are checked by the
bitwise runs of step 3, not argued.

**Checkpoints of `main`.** Nothing that a checkpoint holds changes: the
positions, velocities, and forces, the cell, `bath`, the chain
(`thermostat_state`), the state of the pressure (`barostat_state`), and the
step, from which the counter of the noise continues. A continuation
decides by the fingerprint, whose groups `physics`, `coupling`, and
`execution` hold the keys of the control file and do not change, so
`mdir run --continue` and `mdir.Simulation(checkpoint=…)` continue a
checkpoint of `main` as one of their own. `plan_sha256` of a checkpoint of
a Python simulation hashes the text of the module and will change; it is
provenance, which no continuation compares
([python-checkpoints.md](python-checkpoints.md)).

#### 2.2.9 The plan (#541)

| Step | What | Done when |
|---|---|---|
| 1 | This design | Reviewed |
| 2 | The ops in `DynOps.td`, with their verifiers and effects, `no_global_sums` in the verifier of `dyn.program`, round-trip and invalid tests in `test/Dialect/Dyn`, and `dyn-expand-couplings`, with a test for each op that its expansion is the text the driver writes today; ops.md, Sections 6 and 9.1. In three parts, as step 3: 2a the ops of 3a (built), 2b those of 3b, 2c those of 3c | The driver unchanged; the suite passes |
| 3a | The driver writes the thermostats: the removal of the motion of the center of mass, `V-RESCALE`, `NOSE-HOOVER`, `dyn.to_bath`, and `dyn.noise` for `LANGEVIN` and `BROWNIAN` | In the deterministic mode the NVT and Langevin runs of `examples/ala3`, `thermostat.test`, `nose-hoover.test`, `langevin.test`, and `brownian.test` are bitwise those of `main`, on the CPU and a GPU, in `mixed` and `double`; otherwise within the tolerances of the tests; a checkpoint of `main` continues bitwise |
| 3b | The barostat with `"EXACT"` and `"FIRST_ORDER"`, isotropic, semi-isotropic, and anisotropic: `dyn.group_kinetic`, `dyn.pressure`, `dyn.cell_rescaling`, the cell, `dyn.scale_positions`, `dyn.scaling_work` | The same for `barostat.test` and NPT of `examples/ala3` with these counts |
| 3c | The Trotter steps: `@step_trotter` with `dyn.scale_positions` in its body, `"TROTTER"` and `"TROTTER_FIRST_ORDER"`, the period of one step with `dyn.load_pressure_state` and `dyn.store_pressure_state`; the claims of Section 2.2.4 | The same with the default `work`, `barostat-pme-restart.test` bitwise; the Amber suite and NPT of `examples/ala3` at the rate of `main` on one device |
| 4 | In the PRs of step 3: this section describes the ops as built, the white paper's Section 6 and decisions.md say so | Each PR of step 3 updates them for what it moves |

**Step 2a, as built.** `dyn-expand-couplings` (in
`lib/Dialect/MD/Transforms/ExpandCouplings.cpp`) runs after `md-inline` and
before `md-bypass-updates`; it declares the functions of the host it calls
as the driver declares them. Two checks show that each op expands to the
text that the driver writes. `test/Dialect/MD/Transforms/expand-couplings-noise.mlir`
holds the program of a Langevin step of `mdir emit` with its noise as
`dyn.noise`, and the expansion must print that program verbatim.
`scripts/builder-equivalence/lift_couplings.py` replaces, in a module that
`mdir emit` prints, each sum, loop, and call that an op expands to with the
op; the module of the ops and that of the text, lowered by `mdir-opt` with
the pipeline of the run, must be the same text (`test/Lit/lift-couplings.test`,
and the corpus of #606 in #672). The ops change no
program the driver builds, and the lowered programs of the corpus with
`dyn.step` no longer `Pure` and the pass in the pipeline are those of
`main`.

Each part of step 3 checks the drift and distribution tests of Section 2.4
as they stand. Derivatives through the couplings (M2b), groups with a
thermostat each, and the fusion of the steps around a coupling (#356)
build on the ops and are not part of #541.

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
| Keywords | `[minimize]` takes `method = "STEEPEST_DESCENT"`, `steps` (positive, 100 by default; 0 is refused with "a minimization takes steps: 'steps' may not be 0", #180), `initial_step` in Å (0.1), and `force_tolerance` in kcal/mol/Å (none). Of `[output]`, `energy_interval` must not be 0 and must divide `steps`, and `trajectory_interval`, where given, must be a multiple of `energy_interval` and divide `steps`. No thermostat or barostat. In mixed precision the forces are rounded to about 10⁻⁵ of their size, which bounds how far a minimization can go |
| Convergence | With `force_tolerance` (D219), the host compares the largest force of each row of the log, step 0 included, with it, where it reads the forces for the row anyway; at the first row below it the steps, rows, and frames after it are loops of no iteration, and the checkpoint is written at that step. The log says at the end whether it converged |
| Floor | At the same rows the host compares the length of the next step with a floor, $10^{-7}$ Å below double precision and $10^{-11}$ Å in double ($10^{-6}$ and $10^{-10}$ of the cap; D274): at the first row whose step has shrunk below it, and below the first step, the run ends as at convergence, with exit status 0, and the log says at the end at which step, the step, the floor, and the largest force (`MDIR: stopped at step N: the step has shrunk to ...`); a Python minimization ends there too, with `stalled` in `state().minimization`. Where trials no longer lower the energy by more than its rounding or noise, they are taken at random and the step falls to 0 (#369) |

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
4 × 10⁻¹⁵ nm and 2 × 10⁻¹² nm/ps (`test/Driver/leapfrog-langevin-reference.test`). The plain step has one projection of the velocities more: on JAC of the Amber suite (RTX 4090, mixed precision) 0.149 ms a step instead of 0.135, 9% slower, and 0.160 instead of 0.148 with stochastic cell rescaling. With `work = "TROTTER_FIRST_ORDER"`, the scaling step without energies was off by 8.3 × 10⁻⁷ nm after 20 steps at 2000 atm; in the other ensembles the state after a step with the virial differed from the plain step's by rounding (2 × 10⁻¹⁴ nm after 20 steps), and the rows of the energy file are the same at their six decimals (#285).

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

## Multiple time steps

A design only, from the roadmap (#603). multiple time steps (r-RESPA, [[Tuckerman1992]](references.md#tuckerman1992)), the energy split into a fast part (bonded terms, the direct sum within a short cutoff) and a slow part (the rest, the reciprocal sum), the slow forces kicking half an outer step at its ends and the fast ones integrated by velocity Verlet within it; in MDIR two `md.evaluate` of two `@energy` functions that the terms are partitioned into, a program of an outer step that loops over inner ones, and a check of the resonance of the outer step with the fastest motions, which limits it to about 4 fs with constraints; a flexible cell, with shear, needs the off-diagonal virial and the pressure tensor (F4), a decision of its own.
