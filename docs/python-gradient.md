# The derivative of the energy in the tunable parameters (D[tunable-gradient])

Issue #203, the first implementation step of M2b (#138,
[roadmap](roadmap.md), Section 6.1). Status: implemented. The charges with
particle mesh Ewald are a decision of their own, in a pull request of
their own on #203 (maintainer's decision on PR #205); see [Stages](#stages).

A fit of the parameters of a potential by reweighting
[[ThalerZavadlav2021]](references.md#thalerzavadlav2021) needs
$\partial U/\partial\boldsymbol\theta$ at stored frames. This item gives it
at the state of a Python simulation, for the tunables of D213 and D226
([python-tunable.md](python-tunable.md)); the evaluation at $K$ frames and
the PyTorch adapter build on it.

## Interface

```python
system.tunables = [mdir.Tunable("sigma", "sigma"),
                   mdir.Tunable("epsilon", "epsilon", map=keep),
                   mdir.Tunable("wall_k", "k", term="wall")]
system.tunable_gradient = True          # a compile input: the program carries the derivative
program = mdir.compile(system, state, integrator, ensemble, execution, schedule)
sim = mdir.Simulation(program)
sim.run(10_000)                         # the steps are those of a program without the derivative
g = sim.tunables.gradient()             # no step: U and dU/dtheta at the state
g["sigma"]                              # (M,) float64, kJ/mol/nm, read-only
g.energy                                # the energy it is the derivative of, kJ/mol
g.version, g.step                       # the value version of theta, and the step of the state
g.zero                                  # the names whose derivative is zero by proof
```

- `System.tunable_gradient` (bool, default `False`) asks `mdir.compile` for
  a program that can evaluate the derivative. It is a compile input, as the
  declarations are: setting it advances the version of the system. Without
  it the program is the program of D213, text for text (maintainer's
  decision on PR #205, Q2).
- `Simulation.tunables.gradient()` evaluates the state without a step, as
  `run(0, energy=True)` does
  ([python-segments.md](python-segments.md#evaluations-without-a-step)),
  and returns a `TunableGradient`, a read-only mapping from the names of the
  tunables to arrays of the shape of their values, $(M,)$ float64: entry
  $m$ of the tunable `name` is $\partial U/\partial\theta_{\text{name},m}$.
  After it `state().forces` and, with velocity Verlet, `state().energies`
  are those of the state, as after `run(0, energy=True)`. Before the first
  run it is the start of the run (Q1).
- Units: kJ/mol per unit of the tunable (D213): per e for a charge, per nm
  for $\sigma$, dimensionless for $\epsilon$, and per unit of the constant
  or parameter for one of an expression.
- The positions, the cell, and the values of the tunables are those of the
  simulation. The gradient in the positions is `-state().forces`, and that
  in the strain the virial; they are not repeated here.
- Nothing is written to a file, and nothing enters the fingerprint or a
  checkpoint.

### Which energy

$U$ is the potential that the forces sample (maintainer's decision on
PR #205, Q5), as the columns of `observe` and the energies of the states of
$\boldsymbol\lambda$ take it (D210): where the run cuts a pair term at the
cutoff without a shift, each pair within the cutoff less its energy at
$r_c$,

$$
U = U_\text{run} - \sum_{r_{ij}<r_c} u_{ij}(r_c;\boldsymbol\theta)
    + E_\text{sh}(\boldsymbol\theta),
$$

over the Lennard-Jones under `Truncation.None_`, the pair terms, a Coulomb
cutoff, and the direct sum of PME without `CoulombModifier.PotentialShift`,
with $E_\text{sh}$ the estimate of the shift that the correction for the
dispersion adds when it is on. The weights of a reweighting are
differences of the potential that the trajectory sampled; the cut energy
differs from it by a term that fluctuates with the number of pairs within
the cutoff. With a switch or a shift of the Lennard-Jones and a shifted
Coulomb, $U = U_\text{run}$, the potential energy of
`state().energies`.

`g.energy` is this $U$, with every term that the host adds (the self term
and the background of PME, the correction for the dispersion, the tails of
pair terms, the estimates of the shift). Central differences of `g.energy`
in an entry of a tunable converge to the entry of the derivative, which is
how the tests check it. On the dipeptide with PME at
$\operatorname{erfc}(\beta r_c) = 10^{-5}$ and a switch of the
Lennard-Jones, `g.energy` is 0.128 kJ/mol above the reported energy, the
shift of the direct sum.

## Zero, and failure

Each tunable has one of three outcomes, fixed when the program is compiled
and listed in `Program.plan["tunables"][k]["gradient"]`:

| Outcome | What a script sees |
|---|---|
| `"rule"` | the derivative, computed |
| `"zero"` | an array of zeros, and the name in `g.zero`: the builder proved that the energy does not read the tunable (a constant or a parameter that the expression of its term does not name) |
| a failure | `mdir.compile` raises, naming the tunable and the op without a rule or the reason; no program is made |

`g.zero` lists what is proved from the expressions, not every zero. An
entry can be 0 without its tunable being listed: a pair of types that no
pair of particles within the cutoff has, a site that the state gives no
contribution, a charge or a table that no term of the model reads. Such a
zero is the value of the derivative at that state, computed; a name in
`g.zero` says that it is zero at every state.

Within the program, `md-differentiate` keeps the three outcomes of D161
for every value: independent of the parameter only by a proof, then
exactly zero; dependent with a rule; or an error that names the op. A
value that is not finite at the evaluation is an error of `gradient()`
(`SimulationError`), naming the tunable; it is never returned.

## Refused

`InputError` in every case:

- `gradient()` of a simulation whose program was compiled without
  `tunable_gradient`, or that declares no tunables, or that minimizes.
- `gradient()` while a view or a writable borrow of the simulation is
  alive (D220, D229): `SimulationError`, as for `run(0, energy=True)`.
- `System.tunable_gradient` without tunables, at compile.
- The charges with particle mesh Ewald or an implicit solvent, at compile,
  naming the tunable: not implemented yet (see [Stages](#stages)).
- The charges when a pair term that reads `q1` or `q2` has its tail in the
  correction for the dispersion; $\sigma$ or $\epsilon$ when a pair term
  reads `sigma` or `epsilon` (the pair terms of the Python model do not);
  a parameter of a term over centers of groups: not implemented yet.
- A per-type $\sigma$ (geometric) or $\epsilon$ that is 0 for a type whose
  map takes it, while another type's is not: $\sqrt{xy}$ has no derivative
  in $x$ at 0. The message says to give the type $-1$ in the map. It is an
  error of `mdir.compile`, and of an update to such values.
- The combinations that tunables refuse (D213): `[free_energy]`, `observe`,
  LJPME, pulls.

## How it is computed

The values of a program with tunables are functions of
$\boldsymbol\theta$ that the host computes (D213): the field of the
charges, the tables of $\sigma$ and $\epsilon$ by pairs of types, the table
of the tunable constants, the fields of the tuples, and the constants that
the host adds to the energy. With $\mathbf b(\boldsymbol\theta)$ the
buffers of the program and $c(\boldsymbol\theta)$ those constants,

$$
\frac{\partial U}{\partial\theta_m}
  = \sum_s \frac{\partial U}{\partial b_s}\frac{\partial b_s}{\partial\theta_m}
  + \frac{\partial c}{\partial\theta_m}.
$$

The program gives the first factor, by `md-differentiate`; the host
applies the chain rule and adds the last term (maintainer's decision on
PR #205, Q3: the chain rule is on the host, beside `applyTunables`, not in
the IR of a second potential).

### The potential `@tunable`

A program compiled with `tunable_gradient` carries one more potential,
`@tunable`: the terms of `@energy`, shifted at the cutoff as above, with
the tunables as arguments that `md-differentiate` can differentiate in.
The entry evaluates `md.evaluate @tunable(...) request [energy,
derivative(n), ...]` in a loop at its start that runs once when the host
asks (an argument of the entry, `%tunable_gradient`) and not at all
otherwise, and hands the results to the host.

`md-differentiate` knows two kinds of argument:

- **A number** (D161, D189): the tunable constants of pair terms are
  arguments of `@tunable` in place of the lookups of their table.
- **A field of the particles**, new here: `derivative(n)` of an argument
  that is a field of f64 with one component is a field,
  $g_i = \partial U/\partial a_i$. A sum over a relation that gathers the
  field gives a gather over it whose kernel is the derivative of the
  pair's energy in the value of the central particle, with no exchange
  contract, so that each particle of a pair takes its own; a sum over
  tuples gives a gather over them that yields the derivative in the value
  of each member; a sum over particles gives a map. A field that no sum of
  the energy takes has a field of zeros; any other use (a reciprocal sum,
  a map that computes another field) is an error that names the op.

The loops over pairs and tuples accumulate per particle already, as they
do the forces; the derivative in a table or in a field of tuples would
need a reduction into bins that no lowering has. So every site derivative
leaves the program as a field of the particles, through a *seed*: a field
of zeros $d$ that `@tunable` takes as an argument and adds, times a weight,
to the parameter where the kernel reads it. The value of the energy is
unchanged ($d = 0$), and $\partial U/\partial d_i$ is the weighted sum of
the site derivatives that the particle $i$ collects.

| Tunable | In `@tunable` | Field that leaves | Host |
|---|---|---|---|
| charge | the field of the charges itself; the 1-4 pairs form $fs_Cq_iq_j$ from the charges of their members (a tuple field holds $fs_C$) instead of reading the product | $\partial U/\partial q_i$ | adds over the particles of each entry |
| per-type $\sigma$, $\epsilon$ | $\sigma_{ij} = \sigma_{ab} + d_iw_{ab} + d_jw_{ba}$, with $a$, $b$ the types and $w_{ab} = \partial\sigma_{ab}/\partial\sigma_a$ a table that the host computes | $\sum_j \partial u_{ij}/\partial\sigma_{ab}\,w_{ab}$ | adds over the particles of each type: $\partial U/\partial\sigma_a$ |
| $\sigma$, $\epsilon$ by pairs | the same with $w_{ab} = 1$ for one pair of types in each row $a$, as many seeds as pairs share a type | $\sum_{j\in b}\partial u_{ij}/\partial\sigma_{ab}$ for the pair of the row | adds over the particles of the type of the row, times 1/2 for a pair of one type |
| a constant of a pair term | a number, argument of `@tunable` | a number | adds the tail |
| a parameter of a tuple term | $p_t + d_{t[s]}$ at one member $s$ of the tuple; tuples that share that member take different seeds | $\partial u_t/\partial p_t$ at the member | reads the tuple's value at its member |

The weights of a per-type tunable are the Jacobian of the combining rule
that `applyTunables` applies, written beside it: $1/2$ for Lorentz,
$\tfrac12\sqrt{\sigma_b/\sigma_a}$ for the geometric mean and
$\tfrac12\sqrt{\epsilon_b/\epsilon_a}$ for Berthelot, and 0 for a pair of
types that the rule does not give: one set apart (NBFIX), or one that a
tunable of the table by pairs takes. A pair of two particles of one type
adds to both, which is $\partial\sigma_{aa}/\partial\sigma_a = 1$. For a
tunable of the table by pairs, each pair $(a,b)$ that the map takes is put
in the row of $a$ or of $b$ of the first seed where that row is free, so
that a fit of a few pairs needs one or two seeds and the whole table of
$T$ types about $T/2 + 1$; this is the compression of a sparse Jacobian by
groups of columns that share no row
[[CurtisPowellReid1974]](references.md#curtispowellreid1974). For a tuple
term the member is the place whose particles are shared by the fewest
tuples, and the number of seeds is the largest number of tuples that share
one particle there.

The fields are accumulated in f64 in both precision modes: the host
function that takes them declares buffers of f64, which states the type
they are stored in. In mixed precision the kernels compute each
contribution in f32, as they compute the energy.

### What the host adds

| Constant of the energy | Its derivative |
|---|---|
| The tail of a pair term and the estimate of its shift (D209, D210), in a constant of the term | the Richardson-extrapolated central differences of the quadrature that computes them (`getPairTailDerivative`), exactly 0 for a constant that the expression does not read; proportional to $1/V$ |
| The correction for the dispersion, $E_\text{disp} = K\sum_{ab}n_{ab}\,4\epsilon_{ab}\sigma_{ab}^6$ over the ordered pairs of types, and the estimate of its shift, $E_\text{disp}(1 - V/(N\tfrac{4\pi}{3}r_c^3))$ | closed form: $G_{ab} = K'n_{ab}\,24\epsilon_{ab}\sigma_{ab}^5$ in $\sigma_{ab}$ and $K'n_{ab}\,4\sigma_{ab}^6$ in $\epsilon_{ab}$, and the row $a$ of a seed takes $\sum_b(G_{ab}+G_{ba})w_{ab}$, the same weights as in the kernel |

Under a Coulomb cutoff nothing that the host adds follows the charges.

**At $K$ frames (next step of #138).** The fields are linear in a seed
$g_k$ of the frame: the evaluator will call the same entry at each stored
frame (positions and cell given), accumulate $\sum_k g_k\,\partial
U_k/\partial d$ in f64, and apply the chain rule once.

## Stages

| Stage | State |
|---|---|
| Constants of pair terms; parameters of tuple terms | done |
| Per-type $\sigma$ and $\epsilon$, and the table by pairs of types | done |
| Charges with a Coulomb cutoff | done; the Python model has no reaction field |
| Charges with PME | not in this decision: refused at compile. It follows as a decision of its own on #203 (PR #205, Q4): a fourth result of `md.reciprocal`, the potential of the grid at the particles, in the three templates of PME, only in a program compiled with the derivative; the self term and the background by closed forms on the host |
| Tails of pair terms and the correction for the dispersion (host) | done for constants of pair terms and for $\sigma$, $\epsilon$; the tail of a pair term that reads the charges is refused |

## `observe` and tunables

The builder refuses tunables together with `observe` (D189), which the
Python model does not take. That stays as it is here (Q6): `observe` in
the Python model, with its file, is #188. A tunable constant of a pair
term has its $\partial U/\partial c$ at a state from `gradient()`, of the
same shifted potential as the columns of `observe`.

## Validation

Tests `python-tunable-gradient.test` (CPU) and
`python-tunable-gradient-gpu.test`, `Inputs/python_tunable_gradient.py`;
`differentiate-fields.mlir` and `differentiate-fields-invalid.mlir` for
the pass. "FD" is the central difference of `g.energy` after an update of
one entry, extrapolated from two steps (Richardson), relative to the
largest entry of the tunable; the steps are $10^{-3}$ of the value in
double precision and $2\times10^{-2}$ to $5\times10^{-2}$ in mixed, where
the energy is a sum of f32 terms.

| Quantity | Reference | CPU double | GPU double | CPU mixed | GPU mixed | Tolerance (double, mixed) |
|---|---|---|---|---|---|---|
| Springs over 6 pairs that share particles, $\partial U/\partial k$ (3 entries tied, one site kept) and $\partial U/\partial r_0$ (6), dipeptide in water, PME | NumPy, $\tfrac12(r-r_0)^2$ and $-k(r-r_0)$ at the positions of the state | 4.1e-16, 3.7e-16 | 4.1e-16, 3.7e-16 | 2.2e-7, 2.9e-7 | 2.7e-7, 3.4e-7 | 1e-11, 2e-6 |
| The same with the constants `a`, `l` of the pair term $a\,e^{-r/l}$ (switched) | FD | 1.8e-9 | 1.8e-9 | 2.9e-6 | 2.7e-6 | 1e-8, 2e-4 |
| $\partial U/\partial c$ of the pair term $-c/r^8$ on propane and water (224 atoms), plain cutoff, correction for the dispersion: $-8.772105\times10^{7}$ kJ/mol per unit | NumPy: $-\sum(r^{-8}-r_c^{-8})$ over the pairs not excluded within $r_c$ ($-8.765758\times10^{7}$), the tail $\nu(4\pi/V)P(-1/5r_c^5)$ ($-2.405\times10^{4}$), the shift estimate $\nu(4\pi r_c^3/3V - 1/N)P(-r_c^{-8})$ ($-3.942\times10^{4}$) | 1.7e-16 | 1.7e-16 | 1.3e-5 | 1.9e-5 | 1e-10, 1e-4 |
| The same | FD | 7.3e-15 | 7.5e-15 | 6.4e-8 | 7.4e-8 | 1e-9, 1e-4 |
| Per-type $\sigma$ and $\epsilon$ (8 of 9 types) with $\sigma$ and $\epsilon$ of three pairs of the table, one of a type with itself; dipeptide, PME, switched Lennard-Jones | FD | 5.7e-10 | 7.0e-10 | 2.6e-5 | 8.1e-6 | 1e-7, 5e-4 |
| Per-type $\sigma$ (geometric) and $\epsilon$ with pair tunables taking CT-OW (set apart in the model), OW-OW and CT-HC; propane and water, plain cutoff, correction for the dispersion | NumPy: the shifted Lennard-Jones from sums of $r^{-12}-r_c^{-12}$ and $r^{-6}-r_c^{-6}$ by pairs of types, the table built from $\boldsymbol\theta$ in NumPy, $E_\text{disp}$ and its shift estimate; differentiated in NumPy by central differences | 2.4e-11 | 2.5e-11 | 1.3e-6 | 1.3e-6 | 1e-8, 2e-5 |
| The same | FD | 2.4e-10 | 2.4e-10 | 1.0e-4 | 4.4e-5 | 1e-7, 5e-4 |
| Charges, 25 entries for 1,168 particles (one per atom name of the waters), Coulomb cutoff, dipeptide | NumPy: $\sum_j fq_j(1/r - 1/r_c)$ over the pairs not excluded within $r_c$ and $\sum fq_j/(1.2r)$ over the 1-4 pairs | 5.1e-16 | 7.7e-16 | 5.1e-8 | 1.2e-7 | 1e-9, 5e-6 |
| The same | FD | 1.3e-11 | 8.7e-12 | 1.2e-4 | 3.4e-5 | 1e-7, 5e-4 |
| $\sigma$ and $\epsilon$ by pairs of types after 200 steps of leapfrog under a barostat (the volume 0.9977 of the first), dipeptide, PME, plain cutoff, correction for the dispersion; the four largest entries of each and the smallest that is not 0 | FD | 3.5e-11 | 3.6e-11 | 7.4e-6 | 7.7e-6 | 1e-7, 5e-4 |

In mixed precision the difference of $\partial U/\partial c$ from NumPy,
$1.3\times10^{-5}$, is that of the kernel's table of $r^{-8}$ in f32
(D94), steep at the closest pairs; the energy has it as well, so the FD of
the mixed energy agrees to $6\times10^{-8}$.

Also checked: twelve steps of velocity Verlet through a `gradient()` equal,
to the bit, those of a program compiled without the derivative through
`run(0, energy=True)` (CPU and GPU, double and mixed); the plan, the
shapes, the read-only arrays, a constant that its expression does not read
(`"zero"`, zeros, in `g.zero`), the version after an update, and four
refusals, among them a live view and a live writable borrow. A
simulation continued from a checkpoint (D223), and a new stage from it,
give the derivative and its energy of the simulation that wrote the
checkpoint to the bit, in the four configurations
(`python-tunable-gradient-checkpoint*.test`).

The charges with PME, against torch-pme, belong to that decision.

## Performance

One RTX 3090 (GPU 0, 300 W cap), mixed precision.

**A run without tunables.** `mdir run` on JAC, main (26ad874) against this
branch, alternated three times, the rate of the second half in ns/day:
NVE 850.9, 848.6, 851.6 against 853.8, 851.5, 852.1; NPT 785.3, 784.5,
785.3 against 784.2, 783.7, 784.6. The module and the lowered IR (but for
the compile times that the GPU binaries record) of five programs are equal
between the two builds: JAC without tunables and with the table tunable by
pairs (GPU, mixed), the dipeptide without tunables and with charges,
$\sigma$, $\epsilon$ and a constant tunable (CPU double; GPU mixed).

**A Python simulation of JAC with tunable per-type $\sigma$ and
$\epsilon$** (16 types, NVT, 2 fs, constraints, 5,000 steps after 200, two
runs each): 0.2575 and 0.2572 ms per step without `tunable_gradient`,
0.2568 and 0.2581 with it. Compiling and the first 200 steps take 12.6 and
12.2 s without it, 12.8 and 13.0 s with it. `gradient()` takes 9.8 and
10.4 ms, `run(0, energy=True)` 9.0 and 9.7 ms: both begin an activation of
the entry, which is most of the time.
