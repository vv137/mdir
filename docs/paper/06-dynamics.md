# 6. Dynamics

A step is a program of the `dyn` dialect that the driver writes for the
run (Section 3.1): kicks, drifts, the constraints and their projections,
the placement of virtual sites, and one evaluation of the forces at its
end. Couplings to a bath act between steps, at the end of a *period* of
$N_T$ steps. A minimization replaces the step by one of steepest descent
(Section 6.7). Every derivation below is of what the generated code
computes; $h = \Delta t/2$.

## 6.1 Integrators

**Velocity Verlet** [[Swope1982]](references.md#swope1982) with constraints, as written by
`emitPrograms` in `lib/Driver/Builder.cpp`:

$$
\begin{aligned}
\mathbf u &= \mathbf v_n + h\,\mathbf a_n, &
\mathbf x' &= \mathbf x_n + \Delta t\,\mathbf u, &
\mathbf x_{n+1} &= \mathcal C(\mathbf x'; \mathbf x_n),\\
\mathbf v_{n+\frac12} &= \mathbf u + (\mathbf x_{n+1} - \mathbf x')/\Delta t, &
\mathbf a_{n+1} &= \mathbf F(\mathbf x_{n+1})/m, &
\mathbf v_{n+1} &= \mathcal P_{\mathbf x_{n+1}}\big(\mathbf v_{n+\frac12} + h\,\mathbf a_{n+1}\big),
\end{aligned}
$$

where $\mathcal C$ solves the constraints on the positions from the
unconstrained $\mathbf x'$, with the old configuration giving the
directions of the corrections (Section 6.2), and $\mathcal P_\mathbf x$
projects the velocities onto the tangent space of the constraints at
$\mathbf x$ (RATTLE, [[Andersen1983]](references.md#andersen1983)). The term
$(\mathbf x_{n+1} - \mathbf x')/\Delta t$ gives the velocity the change
of the positions that the constraints made, so the velocity of the half
step is consistent with the drift. Inside the step the order is: drift;
the rigid waters; each set of SHAKE; the correction of the velocities;
the placement of the virtual sites; the evaluation, with the spreading of
the forces of the sites and the restraints; the half kick; and the
projection, waters first. A particle without mass is a virtual site:
`dyn.kick` leaves it alone.

**Leapfrog** [[HockneyEastwood1988]](references.md#hockneyeastwood1988) stores $\mathbf v_{n-\frac12}$; its
plain step is $\mathbf u = \mathbf v_{n-\frac12} + \Delta t\,\mathbf a_n$
followed by the same drift, constraint, and correction, with no second
kick. A step that needs the velocity at time $n$ (for the virial of the
constraints, the log, or the barostat) computes
$\mathbf u = \mathcal P_{\mathbf x_n}(\mathbf v_{n-\frac12} + h\mathbf
a_n) + h\mathbf a_n$, the projected velocity of time $n$ kicked again,
and continues as velocity Verlet; the two then give the same trajectory
to rounding (`test/Driver/leapfrog-constraints.test` compares their logs
to $10^{-7}$, D76). A run that starts leapfrog from velocities of time 0
takes $\mathbf v_{-\frac12} = \mathbf v_0 - h\mathbf a_0$.

**Precision.** In the mixed mode the positions, the velocities, and the
loops over particles (the kicks, the drift, the correction) are f64; the
kernels of the constraints compute in f32 on displacements taken in f64
(Section 7.1). The kick, the drift, the constraints, and the correction
are one kernel on a device (Section 8.2).

## 6.2 Constraints

The constraints are of two kinds: the bonds of hydrogen atoms in groups
of SHAKE, a heavy atom 0 with one to three hydrogen atoms $k$, at lengths
$d_k$; and rigid waters of three sites. Both sets are disjoint, and their
union is an `md.disjoint_union` (Section 3.2).

**Bonds by Newton's method** (M-SHAKE, [[Krautler2001]](references.md#krautler2001)). For a bond
$l = (a, b)$ the correction moves $b$ by $+\lambda_l\mathbf s_l/m_b$ and $a$
by $-\lambda_l\mathbf s_l/m_a$ along the old bond
$\mathbf s_l = \mathbf x^n_b - \mathbf x^n_a$, which conserves the
momentum of the group [[Ryckaert1977]](references.md#ryckaert1977). The new bond is

$$
\mathbf r_k(\boldsymbol\lambda) = \mathbf r'_k + \sum_l w_{kl}\,\lambda_l\,\mathbf s_l,
$$

with $\mathbf r'_k$ the bond of the unconstrained positions and $w_{kl}$
what bond $l$ moves bond $k$ by per unit of $\lambda_l\mathbf s_l$: the
inverse mass of each member that the two bonds share, with its sign. For
a group of SHAKE, whose bonds all start at atom 0,
$w_{kl} = \delta_{kl}/m_k + 1/m_0$. The constraints
$g_k(\boldsymbol\lambda) = \lVert\mathbf r_k\rVert^2 - d_k^2 = 0$ are
solved by Newton's method, each iteration solving the linearization at
the current bonds exactly:

$$
\sum_l 2\,(\mathbf r_k\cdot\mathbf s_l)\,w_{kl}\;\delta\lambda_l = d_k^2 - \lVert\mathbf r_k\rVert^2 ,
$$

by Gaussian elimination of at most 3 × 3. The error squares with each
iteration whatever the ratio of the masses; MDIR takes a fixed six
iterations, with no test of convergence, so that the kernel has no
divergent loop. The iteration of SHAKE by sweeps over the bonds converges
at a rate set by $m_\text{H}/m_\text{X}$, which repartitioned masses of
hydrogen make about $1/2$; with 12 sweeps steps of 4 fs were unstable
(Section 12). The old bond is taken in the periodic image of the new
one: with $\mathbf r^\text{mi}_l$ the new bond in the minimum image and
$\mathbf n$ the stored positions,
$\mathbf s_l = (\mathbf x^n_b - \mathbf x^n_a) + [\mathbf r^\text{mi}_l -
(\mathbf n_b - \mathbf n_a)]$, so a group that straddles the boundary of
the cell is solved in one image. The kernel carries the displacement of
each member, starting from zero, rather than its position; Section 7.3
explains why that matters in f32.

For an isolated one-bond group, `analytic_bonds = true` enables a
quadratic specialization (D128). With $\mathbf r$ the predicted bond,
$\mathbf s$ the old bond, $e=d^2-\mathbf r\cdot\mathbf r$, and
$b=\mathbf r\cdot\mathbf s$, the near root is evaluated as

$$
t = \frac{e}{b+\sqrt{b^2+(\mathbf s\cdot\mathbf s)e}},\qquad
\lambda = \frac{t}{1/m_a+1/m_b}.
$$

The changes are $-\lambda\mathbf s/m_a$ and
$+\lambda\mathbf s/m_b$, passed through the same impulse and virial
machinery. Rationalizing the root avoids cancellation for small
corrections. The candidate is accepted only when $b>0$ and its original
squared-distance residual is at most $10^{-12}d^2$ in double precision,
or $10^{-6}d^2$ in mixed or single precision. Ordered comparisons reject
NaNs; other candidates use the existing six-step Newton solver. This
fallback retains the default solver's fixed-iteration limitations and
does not add a general convergence guarantee. The option defaults to
false. On a synthetic gas of 1,728 noninteracting diatomics, the CPU
complete-stage median changed from 0.413 to 0.252 ms/step (1.64 times);
the small GPU case and the 23,558-atom JAC protein/water case showed no
resolved improvement. These measurements do not establish a speedup for solvated proteins; the benchmark inputs and
results, tagged with the base commit of each comparison, are in
`scripts/benchmarks/constraints/`. The old-bond image construction also
applies to triclinic cells; the tilted-face regression in Section 9.5
compares the analytic and Newton projections on CPU and GPU.

**Rigid water.** In double precision, a water is solved by SETTLE
[[Miyamoto1992]](references.md#miyamoto1992), in closed form: with $r_c = d_\text{HH}/2$, the height
$h = (d_\text{OH}^2 - r_c^2)^{1/2}$, and the distances of O and of the
hydrogens from the center of mass along it, $r_a = 2m_\text{H}h/(m_\text{O} +
2m_\text{H})$ and $r_b = h - r_a$, the new positions follow from three
rotations of the canonical triangle in a frame built from the old
molecule and the new center of mass. Below double precision the same
waters are solved by the Newton iteration above on their three
distances, O–H₁, O–H₂, and H₁–H₂, with the full signed $3\times3$
$w_{kl}$ (D112). Section 7.3 derives why: SETTLE computes absolute
positions, whose rounding in f32 becomes, through
$(\mathbf x_{n+1} - \mathbf x')/\Delta t$, a random error of the
velocities that heats the system; M-SHAKE carries the change itself, whose
rounding is relative to the change [[Jung2026]](references.md#jung2026).

**Velocities.** After the second kick, the velocities of each group are
projected so that every constrained distance has zero rate of change. For
a group of SHAKE with unit bond vectors $\mathbf e_j$ at $\mathbf
x_{n+1}$, the impulses $\tau_j$ along the bonds solve

$$
\sum_j A_{ij}\tau_j = -\mathbf e_i\cdot(\mathbf v_i - \mathbf v_0),
\qquad
A_{ij} = \frac{\delta_{ij}}{m_j} + \frac{\mathbf e_i\cdot\mathbf e_j}{m_0},
$$

and $\Delta\mathbf v_j = \tau_j\mathbf e_j/m_j$,
$\Delta\mathbf v_0 = -\sum_j\tau_j\mathbf e_j/m_0$. For a water the three
bonds O→H₁, O→H₂, H₁→H₂ give the same system with diagonal
$1/m_\text{O} + 1/m_\text{H}$, $1/m_\text{O} + 1/m_\text{H}$, $2/m_\text{H}$
and off-diagonal $\mathbf e_0\cdot\mathbf e_1/m_\text{O}$,
$-\mathbf e_0\cdot\mathbf e_2/m_\text{H}$, $\mathbf e_1\cdot\mathbf
e_2/m_\text{H}$, solved by Cramer's rule, in every precision.

**The virial of the constraints.** In velocity Verlet the positions
advance by $\tfrac{\Delta t^2}{2m}(\mathbf F + \mathbf G^\text{pos})$, so
the move $\boldsymbol\Delta_j$ that the solver made is the effect of a
force $\mathbf G^\text{pos}_j = 2m_j\boldsymbol\Delta_j/\Delta t^2$ over
the first half of the step; likewise the projection is the effect of
$\mathbf G^\text{vel}_j = 2m_j\Delta\mathbf v_j/\Delta t$ over the second
half. The virial of the step is the mean of the two halves, each at the
configuration where its force acts:

$$
\mathsf W_c = \tfrac12\Big[\sum_{j\ne 0}(\mathbf x^n_j - \mathbf x^n_0)\otimes\frac{2m_j\boldsymbol\Delta_j}{\Delta t^2}
+ \sum_{j\ne 0}(\mathbf x^{n+1}_j - \mathbf x^{n+1}_0)\otimes\frac{2m_j\Delta\mathbf v_j}{\Delta t}\Big].
$$

The forces of a group add to zero ($\sum_j m_j\boldsymbol\Delta_j = 0$),
so the displacements may be taken from member 0, whose term vanishes. At
the start, where no step has given the forces of the constraints, the log
takes $\sum_k(\mathbf x_k - \mathbf x_0)\cdot\mathbf G^0_k - 2K_\text{int}$
with $\mathbf G^0 = m\,\mathcal P(\mathbf F/m) - \mathbf F$, which keeps
the accelerations on the constraints, and $K_\text{int}$ the kinetic
energy of the motion within the group.

**Degrees of freedom.** $N_f = 3N_{m>0} - N_c - 3$, with
$N_c = 3N_\text{water} + \sum_\text{groups} n_\text{H}$; the three of the
center of mass are always subtracted.

## 6.3 The thermostat

Velocity rescaling with a stochastic term [[Bussi2007]](references.md#bussi2007) acts at the end
of each period of $N_T$ steps (10 by default). First the motion of the
center of mass is removed, $\mathbf v_i \leftarrow \mathbf v_i - \mathbf
V_\text{cm}$, and its kinetic energy $K_\text{cm}$ goes to the bath; then
the kinetic energy $K_t$ of the remaining motion is rescaled. With
$\bar K = \tfrac12 N_f k_BT_0$, $c = e^{-N_T\Delta t/\tau_T}$, a normal
number $R_1$, and $S$ distributed as $\chi^2_{N_f-1}$,

$$
K' = K_t + (1 - c)\Big(\frac{\bar K(R_1^2 + S)}{N_f} - K_t\Big)
+ 2R_1\sqrt{\frac{c(1-c)\,K_t\bar K}{N_f}},
\qquad
\alpha = \sqrt{K'/K_t},
$$

which is the exact solution over the period of the stochastic
differential equation of [[Bussi2007]](references.md#bussi2007) for $K$. $S$ is drawn as
$2\,\Gamma(\tfrac{N_f-1}{2}, 1)$ by the method of [[Marsaglia2000]](references.md#marsaglia2000), so
$N_f$ need not be an integer. One loop over particles applies
$\mathbf v \leftarrow \alpha(\mathbf v - \mathbf V_\text{cm})$, and the
bath takes $K_\text{cm} + (1 - \alpha^2)K_t$, which keeps the conserved
energy, the total plus the bath, flat when the dynamics conserve the
total. The velocity rescaled is that of time $n+1$ for velocity Verlet and
for leapfrog with a barostat, and the stored $\mathbf v_{n+\frac12}$ for
leapfrog alone. Without a thermostat the center of mass is removed only
if `center_of_mass_interval` asks.

**Langevin dynamics** (`method = "LANGEVIN"`, D135) acts in every step
instead, between the two halves of the drift: the middle scheme of
[[Zhang2019]](references.md#zhang2019), BAOAB in the terms of
[[Leimkuhler2013]](references.md#leimkuhler2013),

$$
\mathbf v \leftarrow \mathbf v + \tfrac{\Delta t}{2}\frac{\mathbf F}{m},\quad
\mathbf x \leftarrow \mathbf x + \tfrac{\Delta t}{2}\mathbf v,\quad
\mathbf v \leftarrow c\,\mathbf v + \sqrt{(1-c^2)\frac{k_BT}{m}}\,\mathbf R,\quad
\mathbf x \leftarrow \mathbf x + \tfrac{\Delta t}{2}\mathbf v,
$$

with $c = e^{-\gamma\Delta t}$ for the friction $\gamma$ and three normal
numbers $\mathbf R$ of the particle, then the constraints, the evaluation,
and the second half kick of Section 6.1. The constraints of the positions
and the velocities of the change remove the part of the noise along the
constraints, and the step that scales the cell (Section 6.4) scales after
the noise. The momentum of the center of mass is not kept, so
$N_f = 3N - N_c$ unless its motion is removed, and the log has no conserved
energy: the run does not count the energy that the friction and the noise
exchange with the bath. With constraints, the virial of their forces is
that of the forces that keep the bonds at their lengths at the end of the
step, $\mathbf e\cdot(\mathbf a_q - \mathbf a_p) = -\lvert\mathbf v_q -
\mathbf v_p\rvert^2/d$, solved with the matrix of RATTLE, rather than the
mean of the changes of SHAKE and RATTLE within the step, which the
friction makes smaller by $\gamma\Delta t/2$: those raised the pressure of
water at one volume by 39 bar per 1/ps of friction at 2 fs. An ideal gas of 256 particles warmed from 10 K by a
bath of 300 K with $\gamma$ = 5/ps relaxes its kinetic energy at 9.6 ± 0.3/ps,
against $2\gamma$ = 10/ps.

## 6.4 The barostat

Stochastic cell rescaling [[Bernetti2020]](references.md#bernetti2020) couples the cell to a
pressure $P_0$ every $N_P$ steps; $N_P = N_T$, and a barostat requires the
thermostat. The internal pressure for the strain is

$$
P = c\,\frac{2K_t + \operatorname{tr}\mathsf W + C/V}{3V},
$$

with $c = 16.6053906717$ bar nm³ mol/kJ, $K_t$ the kinetic energy of the
step without the center of mass before the thermostat, $\mathsf W$ the
virial with that of the constraints, and $C/V$ the virials of the
dispersion correction and of the neutralizing background at the current
volume (Section 5). With $\lambda = \sqrt V$, which makes the noise
additive, and $f = \beta_T N_P\Delta t/\tau_P$ (compressibility $\beta_T$,
time constant $\tau_P$), one step of Euler and Maruyama of eq. (S7) of
[[Bernetti2020]](references.md#bernetti2020) is

$$
\lambda' = \lambda - \frac{f\lambda}{2}\Big(P_0 - P - \frac{k_BT\,c}{2V}\Big)
+ \sqrt{\frac{k_BT\,c\,f}{2}}\,R,
\qquad
\Delta\varepsilon = 2\ln\frac{\lambda'}{\lambda},\qquad \mu = e^{\Delta\varepsilon/3}.
$$

The edges of the cell are multiplied by $\mu$; a run stops if an edge
falls below $2r_c$, where the minimum image would miss pairs.

**Semi-isotropic coupling** (D119). A bilayer normal to $z$ needs its
area and its height coupled apart: x and y scale together by the strain
of the area $\varepsilon_{xy} = \ln A$, from the mean of their pressures,
and z by that of the height $\varepsilon_z = \ln L_z$, from its own, by
eqs. (9a) and (9b) of [[Bernetti2020]](references.md#bernetti2020). The pressure of axis $a$ is

$$
P_{aa} = c\,\frac{2K_a + W_{aa} + C/(3V)}{V},
$$

with $K_a$ the kinetic energy along $a$ without the center of mass, and
one step of Euler and Maruyama over the period is

$$
\Delta\varepsilon_{xy} = -\frac{2f}{3}\Big(P_0 - \frac{\gamma}{L_z} - \frac{P_{xx} + P_{yy}}{2}\Big) + \sqrt{\frac{4k_BT\,c\,f}{3V}}\,R_0,
\qquad
\Delta\varepsilon_z = -\frac{f_z}{3}\big(P_0 - P_{zz}\big) + \sqrt{\frac{2k_BT\,c\,f_z}{3V}}\,R_1,
$$

$\mu = (e^{\Delta\varepsilon_{xy}/2}, e^{\Delta\varepsilon_{xy}/2}, e^{\Delta\varepsilon_z})$,
with $f_z$ from the compressibility of z, $\gamma$ the surface tension
times the number of surfaces (1 dyn/cm = 10 bar nm), and $R_0$, $R_1$ the
first two normal numbers of the step. The stationary density of the pair
follows from the equation of Fokker and Planck: with
$D_{xy} = 2k_BT\beta/(3V\tau_P)$ and the drift $A_{xy}$ of the first,
zero flux, $A_{xy}\rho = \partial_{\varepsilon_{xy}}(D_{xy}\rho)$, holds for

$$
\rho(\varepsilon_{xy}, \varepsilon_z) \propto A L_z\,e^{-(P_0V - \gamma A + F)/k_BT},
$$

since $\partial_{\varepsilon_{xy}}(P_0V - \gamma A + F) = V(P_0 - \gamma/L_z -
P_\parallel)$ with $P_\parallel = -L_z^{-1}\partial F/\partial A$, the mean of
$(P_{xx} + P_{yy})/2$ over the particles, and in the same way for the
height; $A L_z$ is the Jacobian of $(\ln A, \ln L_z)$. That is the
ensemble at constant normal pressure and surface tension. Without
tension the sum of the two steps is the step in $\ln V$ of eq. (5), whose
noise depends on the volume; the step in $\lambda$ above does not, and
the two agree to first order in the period. With the compressibility of z
zero, the height is kept to the bit. Positions scale by $\mu_a$ along
axis $a$ (rigid groups with their centers), velocities by $1/\mu_a$, and
a triclinic cell by $H\,\mathrm{diag}(\boldsymbol\mu)$, each tilt with
its column, so that the particles keep their lattice coordinates and the
cell stays lower triangular and reduced (D127); and
the work of a scaling below is a sum over the axes, with $K_a$ and the
diagonal of the virial of the groups in place of $K$ and the trace.

**Scaling.** A free particle moves to $\mu\mathbf x$; a water or a group
of SHAKE moves with its center of mass, $\mathbf x_j \to \mu\mathbf X +
(\mathbf x_j - \mathbf X)$, with $\mathbf X$ taken in the minimum image,
since stretched bonds would be taken back by the next step's constraints
with a change of the velocities that heats the system. Velocities are
multiplied by $1/\mu$. The neighbor structures stay valid under the
scaling if the test of Section 4.1 holds, so a change of the cell does not
force a build (D80).

**The work of a scaling.** The conserved energy must take away the energy
a scaling gives the system. The default (`work = "TROTTER"`, D92) scales
within the drift of the last step of a period, after eqs. (S12a–d) of
[[Bernetti2020]](references.md#bernetti2020): the step kicks half, drifts half, scales the positions
by $\mu$ and the velocities by $1/\mu$, drifts the other half,
constrains, and evaluates in the new cell. The scaling changes the kinetic
energy by $(\mu^{-2} - 1)K_{1/2}$, with $K_{1/2}$ that of the half-drifted
velocities, and the potential energy by
$\Delta U = -\int\mathcal W\,d\ln\mu$, which the trapezoid over the virials
before and after gives to second order:

$$
\Delta E_\text{sys} = (\mu^{-2} - 1)K_{1/2} - \ln\mu\cdot\tfrac12(\mathcal W_b + \mathcal W_a),
\qquad
\mathcal W = \sum_g\mathbf X_g\cdot\mathbf F_g + C/V .
$$

$\mathcal W$ is the virial of the groups that move as wholes, with
$\mathbf X_g$ the center of mass of group $g$ (a free particle is a group
of one) and $\mathbf F_g$ the total force on it; it is $-dU/d\ln\mu$
when the groups scale with their centers. In a periodic cell it is taken
from the virial of an evaluation and the places within each group,

$$
\sum_g\mathbf X_g\cdot\mathbf F_g
= \operatorname{tr}\mathsf W_\text{eval} - \sum_g\sum_{j\in g}(\mathbf x_j - \mathbf X_g)\cdot\mathbf F_j ,
$$

with the forces after the virtual sites have given theirs to their atoms
(a linear site leaves the virial as it is). The forces of the
constraints do not enter: they are internal to the groups and drop out
of both sums. The virial of the step, which has those of the
constraints, gives the same with twice the internal kinetic energy:
since $\sum_j m_j(\mathbf x_j - \mathbf X) = 0$,

$$
\frac{d}{dt}\sum_j m_j(\mathbf x_j - \mathbf X)\cdot(\mathbf v_j - \mathbf V)
= \sum_j m_j\lVert\mathbf v_j - \mathbf V\rVert^2 + \sum_j(\mathbf x_j - \mathbf X)\cdot\mathbf F_j ,
$$

and the left side is half the second derivative of the moment of inertia
about the center, zero for a rigid group, so $-\sum_j(\mathbf x_j -
\mathbf X)\cdot\mathbf F_j = 2K_\text{int}$ with $\mathbf F_j$ the forces
with those of the constraints, and $\mathcal W = \operatorname{tr}\mathsf
W + 2K_\text{int} + C/V$. MDIR counted so first, and the count was biased
with rigid groups: the identity needs the forces of the constraints of
one configuration, and in the step that scales, the constraints of the
positions take the forces of its start, before the scaling, and those of
the velocities the forces after it. The error of $\mathcal W_a$ is then
of first order in $\ln\mu$, and the trapezoid errs by a term of second
order of one sign: on 1039 rigid OPC waters the conserved energy drifted
by $-234$ kcal/mol/ns, at every period, and from the evaluations it
drifts by $+12.7 \pm 13$ (D116). The virial of the groups is also exact
for a group of SHAKE with two or three hydrogens, whose angles bend.
The virials before and after are those of the
steps around the scaling, half a step from it, so the count is exact to
second order in the strain but not in that offset, and the conserved
energy drifts slowly, more with longer periods, as the effective energy
of [[Bernetti2020]](references.md#bernetti2020) does (its Fig. 2c). Eqs. (S13a) and (S15) of that
paper, which write the step as one formula, leave out the scaling of the
positions and have $\Delta t$ for $\Delta t/2$; MDIR follows eqs.
(S12a–d) (D92). Three other counts are available:
`TROTTER_FIRST_ORDER` takes $\mathcal W_b$ alone and saves a virial;
`EXACT` scales at the end of the period, evaluates $U$ at the scaled
positions, and takes $U(\mathbf x') - U(\mathbf x) + C\,(1/V' - 1/V) +
(\mu^{-2} - 1)\alpha^2K_t$, starting the next step from the forces it
computed, where $C/V$ is the energy of the terms that are constants of
the volume, the correction for the dispersion and the background of a
net charge (Section 5), which `md.evaluate` leaves out. Before that term
was counted, the conserved energy carried $E_c(V) - E_c(V_0)$: on the
ff19SB peptide in OPC water compressed by the barostat, $-1.47$ kcal/mol
after 2 ps, exactly the dispersion correction of $-30.24$ kcal/mol times
$V_0/V - 1 = 0.0487$;
`FIRST_ORDER`, as GROMACS does, takes $-(\mu - 1)\mathcal W + (\mu^{-2} -
1)\alpha^2K_t$, from the trace of the virial of the step with twice the
internal kinetic energy, an identity of the trace that holds for each
axis only in the mean, and is for isotropic coupling only.

**What the counts give.** On a Lennard–Jones mixture over 80 ps and five
seeds at $\tau_P = 1$ ps, the conserved energy drifts per step by
$(0.7\pm0.6)\times10^{-6}\,k_BT$ under the Trotter count at $N_P = 2$,
$(3.6\pm2.4)\times10^{-6}$ at 10, and $(2.2\pm1.2)\times10^{-5}$ at 100;
counted exactly, within $3\times10^{-7}$ of zero; counted to first order,
about $4\times10^{-3}$ (D92). The first-order count leaves the second
order of the change of the potential energy, whose mean over the noise is
proportional to its variance and so to $f$: a drift that neither the step
nor the period reduces, only $\tau_P$ (tri-alanine in 1218 OPC waters:
2.13 kcal/mol/ps at $\tau_P$ = 2 ps, 0.52 at 8 ps, and 0.011 counted
exactly, D77).

**Schedule.** A period is $N_P - 2$ plain steps, a step that computes the
virial (whose pressure gives $\mu$), and the Trotter step, followed by
the removal of the motion of the center of mass and the thermostat. With
$N_P = 1$ the pressure is that of the previous step, kept in memory and
in checkpoints. Leapfrog with a barostat couples $\mathbf v_{n+1}$ and
stores $\mathbf v' - h\mathbf a'$.

**Restraints.** A positional restraint $k\lVert\mathbf x - \mathbf
x^\text{ref}\rVert^2$ depends on where its reference is when the cell
changes. MDIR writes the reference as a center that scales with the cell
and an offset that does not, $\mathbf x^\text{ref} = \mathbf s \odot
\mathbf c + \mathbf o$, $\mathbf s$ the edges of the cell over those of the
file of coordinates (D124). By default $\mathbf c$ is the mean of the
references of the restrained particles, so that a restrained solute keeps
its shape; for restraints spread through the cell, such as the phosphorus
atoms of a bilayer, each reference is its own center with no offset and
follows the cell as the positions do (the options `com` and `all` of
`refcoord-scaling` in GROMACS
[[GromacsManual2025]](references.md#gromacsmanual2025)). The virial is the
derivative of the energy when the positions and the centers scale,
$\operatorname{diag}\sum -2k\,\mathbf d \odot (\mathbf d + \mathbf o)$
with $\mathbf d = \mathbf x - \mathbf x^\text{ref}$. Scaling every
reference about the origin shrank those of ubiquitin with a cell that had
become 0.926 of the file's, and the restraints, pushing the protein
outward, held the cell 2.3% larger than a run without them; scaled about
their center, the two agree within the noise (196,262 ± 580 and 196,293 ±
658 Å³).

## 6.5 What the log reports

$K$ is the kinetic energy of $\mathbf v_{n+1}$. Without constraints the
log corrects the estimates of the temperature and the pressure for the
velocity of the integer step [[Jung2018]](references.md#jung2018) [[Jung2019]](references.md#jung2019): with
$\epsilon = \tfrac{\Delta t^2}{8}\sum_i\lVert\mathbf F_i\rVert^2/m_i$,
$K_T = K + \tfrac23\epsilon$ (the mean of $K_{n\pm\frac12}$ and $K_n$) and
$K_P = K + \epsilon$. With constraints both are $K$. Then

$$
T = \frac{2K_T}{N_fk_B},
\qquad
P = \frac{2K_P + \operatorname{tr}\mathsf W}{3V},
\qquad
\mathsf W = \sum\mathbf d\otimes\mathbf K + \mathsf W_\text{tuples} + \mathsf W_\text{rec} + \mathsf W_c + \dots,
$$

with the sign of $\mathsf W$ that of $\sum_i\mathbf x_i\otimes\mathbf F_i$:
positive for repulsion. $\mathsf W$ includes the virials of the
constraints, of the virtual sites, of the restraints, of the dispersion
correction ($6E_\text{disp}$), and of the neutralizing background
($3E_Q$). The total energy is $U + K$; the conserved energy is the total
plus what the bath has taken. At the start the log gives the diagonal of
$\mathsf W$ without the virials of the constraints, which other programs
give for the same positions; at the end of a run with semi-isotropic
coupling, the means of the pressures of x and y and of z that the
barostat took and of their difference, with errors from blocks of 100
periods (D119).

## 6.6 Random numbers

The thermostat and the barostat draw from Philox 4×32-10
[[Salmon2011]](references.md#salmon2011), a counter-based generator, on the host: the key is the
seed, the counter is $(\text{step}, 0, 2^{24}\,\text{stream} + \text{block})$, with stream 0 for the
thermostat and 1 for the barostat, and the step is that of the end of the
period. A uniform number is $(\lfloor w/2^{11}\rfloor + \tfrac12)\,2^{-53}$
from 64 bits $w$, and a normal number is the cosine branch of Box–Muller
[[BoxMuller1958]](references.md#boxmuller1958). A draw depends only on the seed, the stream, and the
step, so a run continued from a checkpoint draws what the uninterrupted
run would have drawn, which makes restarts bitwise. Langevin dynamics
draws in the kernel of the step, with Philox emitted as operations of the
IR: one block of four words for each particle and step, the counter
$(\text{step}, \text{particle}, 2^{25})$ (stream 2) with the number of the
particle in the input and the step from a counter that each step advances,
the words $w$ made uniform as $(w + \tfrac12)\,2^{-32}$, and both branches
of Box–Muller of two pairs, of which three numbers are taken. It agrees
with the generator of the host to the last bit. Initial velocities
are drawn from a Maxwell–Boltzmann distribution with xoshiro256**
[[Blackman2021]](references.md#blackman2021), their components along the constraints and the motion
of the center of mass removed, and scaled to $\tfrac12N_fk_BT$.

## 6.7 Minimization

`[minimize]` in place of `[dynamics]` lowers the potential energy by
steepest descent in the metric of the masses, on the surface of the
constraints (D73). The direction is the acceleration with its parts along
the constraints taken off, $\mathbf g = P(\mathbf F/m)$, where $P$ is the
projection that RATTLE applies to velocities at the current positions and
$\mathbf g_i = 0$ for a virtual site. Weighting by the masses is what keeps
the step downhill once SETTLE and SHAKE, which weigh the particles by
their masses, take the groups back to their shapes: without it the steps
went uphill after 50 steps on the target of D65, and the step size fell
to 0. A trial of step length $\ell$ is

$$
\mathbf x' = C\!\left(\mathbf x + \ell\,\frac{\mathbf g}{\lVert\mathbf g\rVert_{16}}\right),
\qquad
\lVert\mathbf g\rVert_{16} = r\Big(\sum_i \big(\lVert\mathbf g_i\rVert/r\big)^{16}\Big)^{1/16},
$$

with $r$ the root mean square of the $\lVert\mathbf g_i\rVert$, $C$ the
constraints applied from $\mathbf x$, and the sites placed again. Since
$\lVert\mathbf g\rVert_{16} \ge \max_i\lVert\mathbf g_i\rVert$, no particle
moves farther than $\ell$, and no reduction to a maximum is needed. A trial
that lowers the energy is taken and $\ell$ grows by a factor 1.2, to at most
1 Å; otherwise $\mathbf x$ stays and $\ell$ shrinks by a factor 0.2. The
choice is a select of each particle, so each field keeps its own storage.
A step is one `dyn.step @descend`, which evaluates the energy and the
forces once.

**The start** (D120). The positions of the file are first taken onto the
surface of the constraints, by SETTLE (M-SHAKE below double precision) and
SHAKE with themselves as the reference, and the sites are placed again,
before the first evaluation. From positions off the surface, every trial
carries besides the step the change that takes the groups to their shapes,
which does not shrink with $\ell$. A bilayer of 126 POPC of Lipid21
[[Dickson2022]](references.md#dickson2022) in TIP3P built by PACKMOL
[[Martinez2009]](references.md#martinez2009) had bonds of hydrogen up to 0.021 Å off their lengths and
hydrogens of different lipids 0.13 Å apart; that change brought one such
pair from 0.128 to 0.097 Å, raising the Lennard-Jones energy by
$5.6\times10^{15}$ kcal/mol, so no trial was ever taken and $\ell$ shrank to
0. Taken onto the surface first, the bilayer went from $1.05\times10^{16}$
to $-85{,}425$ kcal/mol in 5000 steps (`minimize-clash.test` runs four of
its lipids, from $2.9\times10^{15}$ to $-551$ kcal/mol in 200 steps). On
the target of D65 (`minimize.test`), whose bonds of hydrogen from tleap
are also off their lengths, the energy at the start is $-5348.8659$
kcal/mol on the surface rather than $-5348.4328$ off it.

The log gives the potential energy, the root mean square and the largest
of the forces $m\mathbf g$ over the particles with mass, in kcal/mol/Å, and
$\ell$. A minimization ends with a checkpoint of the positions and zero
velocities; a run that reads it begins anew at step 0, with drawn
velocities. In mixed precision the forces are rounded to about $10^{-5}$ of
their size, which bounds how far a minimization can go; a tolerance on the
force is planned.
