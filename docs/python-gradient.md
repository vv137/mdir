# The derivative of the energy in the tunable parameters (D[tunable-gradient])

Issue #203, the first implementation step of M2b (#138,
[roadmap](roadmap.md), Section 6.1). Status: in progress; the stages that
are implemented are listed [below](#stages).

A fit of the parameters of a potential by reweighting
[[ThalerZavadlav2021]](references.md#thalerzavadlav2021) needs
$\partial U/\partial\boldsymbol\theta$ at stored frames. This item gives it
at the state of a Python simulation, for the tunables of D213 and D226
([python-tunable.md](python-tunable.md)); the evaluation at $K$ frames and
the PyTorch adapter build on it.

## Interface

```python
system.tunables = [mdir.Tunable("q", "charge", map=charge_map),
                   mdir.Tunable("sigma", "sigma"),
                   mdir.Tunable("wall_k", "k", term="wall")]
system.tunable_gradient = True          # a compile input: the program carries the derivative
program = mdir.compile(system, state, integrator, ensemble, execution, schedule)
sim = mdir.Simulation(program)
sim.run(10_000)                         # the steps are those of a program without the derivative
g = sim.tunables.gradient()             # no step: U and dU/dtheta at the state
g["q"]                                  # (M,) float64, kJ/mol/e
g.energy                                # the potential energy it is the derivative of, kJ/mol
g.version, g.step                       # the value version of theta, and the step of the state
g.zero                                  # the names whose derivative is zero by proof
```

- `System.tunable_gradient` (bool, default `False`) asks `mdir.compile` for
  a program that can evaluate the derivative. It is a compile input, as the
  declarations are: setting it advances the version of the system. Without
  it the program is the program of D213, text for text.
- `Simulation.tunables.gradient()` evaluates the state without a step, as
  `run(0, energy=True)` does
  ([python-segments.md](python-segments.md#evaluations-without-a-step)),
  and returns a read-only mapping from the names of the tunables to arrays
  of the shape of their values, $(M,)$ float64: entry $m$ of the tunable
  `name` is $\partial U/\partial\theta_{\text{name},m}$. After it
  `state().forces` and, with velocity Verlet, `state().energies` are those
  of the state, as after `run(0, energy=True)`.
- $U$ is the potential energy that `run(0, energy=True)` reports
  (`g.energy`), with every term that the host adds: the self term and the
  background of PME or of the reaction field, the correction for the
  dispersion, the tails of pair terms and the estimates of the shift
  (D209, D210). Central differences of that energy in an entry of a tunable
  converge to the entry of the derivative.
- Units: kJ/mol per unit of the tunable (D213): per e for a charge, per nm
  for $\sigma$, dimensionless for $\epsilon$, and per unit of the constant
  or parameter for one of an expression.
- The positions, the cell, and the values of the tunables are those of the
  simulation. The gradient in the positions is `-state().forces`, and that
  in the strain the virial; they are not repeated here.

## Zero, and failure

Each tunable has one of three outcomes, fixed when the program is compiled
and listed in `Program.plan["tunables"][k]["gradient"]`:

| Outcome | What a script sees |
|---|---|
| `"rule"` | the derivative, computed |
| `"zero"` | an array of zeros, and the name in `g.zero`: the differentiation proved that no value of the program that the tunable reaches enters the energy (D161) |
| a failure | `mdir.compile` raises, naming the tunable and the op without a rule or the reason that independence could not be proved; no program is made |

A value that is not finite at the evaluation is an error of
`gradient()` (`RuntimeError`), naming the tunable; it is never returned.

## Refused

- `gradient()` of a simulation whose program was compiled without
  `tunable_gradient`, or without tunables: `InputError`.
- `System.tunable_gradient` without tunables: `InputError` at compile.
- A kind of tunable or a term whose derivative is not implemented yet (see
  [Stages](#stages)): `InputError` at compile naming the tunable and what is
  missing. Nothing is left out silently.
- The combinations that tunables refuse (D213): `[free_energy]`, `observe`,
  LJPME, pulls.

## How it is computed

The values of a program with tunables are functions of
$\boldsymbol\theta$ that the host computes (D213): the field of the
charges, the tables of $\sigma$ and $\epsilon$ by pairs of types, the table
of the tunable constants, the fields of the tuples, and the constants that
the host adds to the energy. Write $\mathbf b(\boldsymbol\theta)$ for the
buffers of the program and $c(\boldsymbol\theta)$ for those constants.
Then

$$
\frac{\partial U}{\partial\theta_m}
  = \sum_s \frac{\partial U}{\partial b_s}\frac{\partial b_s}{\partial\theta_m}
  + \frac{\partial c}{\partial\theta_m}.
$$

1. **Sites, in the program.** `md-differentiate` gains the derivative of a
   potential in an argument that is a field of the particles, a table, or a
   field of a tuple set: `derivative(n)` of such an argument has the
   argument's type and holds $\partial U/\partial b_s$ for each site $s$ (a
   particle, a pair of types, a tuple). It has the three outcomes of the
   scalar derivative (D161): a proof of independence gives a zero, a rule
   gives the derivative, anything else is an error that names the op. The
   sums are accumulated in f64 in both precision modes.
2. **Chain rule, on the host.** The host pulls the site derivatives back
   through the code that built $\mathbf b$ from $\boldsymbol\theta$
   (`model::applyTunables` and the builder): the maps; the combining rule,
   with the pairs of types set apart from it and those that a pair tunable
   takes; the 1-4 products $fq_iq_js_C$; and it adds
   $\partial c/\partial\theta$: closed forms for the self term and the
   background of PME, the reaction field, and $\langle C_6\rangle$, and for
   the tails of pair terms the Richardson-extrapolated central differences
   of the quadratures that compute them (D209).

The evaluation is a branch of the entry at its start, taken only when the
host asks: a run takes no part of it, and a program without
`tunable_gradient` does not have it.

**At $K$ frames (next step of #138).** The site derivatives are linear in
the seed: the evaluator will call the same entry at each stored frame
(positions and cell given), accumulate $\sum_k g_k\,\partial U_k/\partial
\mathbf b$ in f64, and apply the chain rule once.

## Stages

| Stage | State |
|---|---|
| Constants of pair terms; parameters of tuple terms | in progress |
| Per-type $\sigma$ and $\epsilon$, and the table by pairs of types | planned |
| Charges with a cutoff or the reaction field | planned |
| Charges with PME | planned |
| Tails of pair terms and the correction for the dispersion (host) | planned |

## `observe` and tunables

The builder refuses tunables together with `observe` (D189), which the
Python model does not take. That stays as it is here: `observe` in the
Python model, with its file, is #188. A tunable constant of a pair term
has its $\partial U/\partial c$ at a state from `gradient()`.
