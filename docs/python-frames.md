# The frame evaluator (D240)

Issue #249, a step of M2b (#138, [roadmap](roadmap.md), Section 6.1).
Status: the first slice is implemented, as the maintainer ruled on PR #250
([Maintainer rulings](#maintainer-rulings)); what later slices add is in
[Slices](#slices).

A fit of the parameters of a potential by reweighting
[[ThalerZavadlav2021]](references.md#thalerzavadlav2021) takes frames $S_n$
sampled at $\hat{\boldsymbol\theta}$ and forms

$$
\langle O\rangle_{\boldsymbol\theta} = \sum_n w_n O(S_n),\qquad
w_n = \frac{e^{-(U_{\boldsymbol\theta}(S_n) - U_{\hat{\boldsymbol\theta}}(S_n))/k_BT}}
           {\sum_m e^{-(U_{\boldsymbol\theta}(S_m) - U_{\hat{\boldsymbol\theta}}(S_m))/k_BT}},
$$

whose derivative in $\boldsymbol\theta$ a framework takes by automatic
differentiation, given $U_{\boldsymbol\theta}(S_n)$ at every frame and a
rule for its derivative. D230 ([python-gradient.md](python-gradient.md))
gives $U$ and $\partial U/\partial\boldsymbol\theta$ at the state of a
simulation. The frame evaluator gives them at $K$ stored frames, with the
vector-Jacobian product

$$
\mathbf g \mapsto \sum_{n=0}^{K-1} g_n\,
\frac{\partial U_{\boldsymbol\theta}(S_n)}{\partial\boldsymbol\theta},
$$

and `mdir.torch.evaluate` makes it a differentiable operation of PyTorch.
The weights, the effective sample size, the loss, and the decision to
sample again stay in the framework (#138).

## Interface

### The evaluator

```python
evaluator = mdir.FrameEvaluator(program)     # a simulation of its own (D236)
evaluator.tunables.update({"sigma": s1})     # the TunableValues of D213
out = evaluator.evaluate(positions, cells=None, tilts=None, gradient=True)
out.energy                  # (K,) float64, kJ/mol: the potential that the forces sample (D210)
out.virial, out.volume      # (K,) float64, kJ/mol and nm^3
out.observables["lower.d_z0"]   # (K,) float64, the columns of `observe` (D232)
out.version                 # the value version of theta that the frames were evaluated at
out.depends["virial"]       # the tunables that an output may read; empty: none, by proof
out.jacobian["sigma"]       # (K, M) float64, or `out.jacobian is None` above the bound
g = out.vjp(cotangent)      # cotangent (K,) float64 on `energy`
g["sigma"]                  # (M,) float64, kJ/mol per unit of the tunable
g.zero                      # the names whose derivative is zero by proof (D230)
```

`mdir.FrameEvaluator(program, jacobian_bytes=2**28)` takes a program of
dynamics compiled with tunables (D213) and `System.tunable_gradient`
(D230): the energy of a frame is that of the potential `@tunable`, which
only such a program carries, so a program without it is refused whether or
not the derivative is asked for. The evaluator creates and keeps one
`mdir.Simulation` of the program for itself, which no script sees: the
simulation that samples keeps its state, its activation, its order of the
particles, and its leases, and the evaluator may be at other values of the
tunables than the sampler. The program need not be the sampler's (ruling
Q11): any program over the same particles. The values of its tunables begin
as those of the program and change only through `evaluator.tunables`.

`evaluate(positions, cells=None, tilts=None, *, gradient=True)`:

| Argument | Shape, type | Meaning |
|---|---|---|
| `positions` | `(K, N, 3)` or `(N, 3)`, float64 or float32; a NumPy array (a `numpy.memmap` too), or any object with `__dlpack__` | nm, **in the order of the input**, as `InitialState.positions` and `state().positions` are |
| `positions` | an iterable, one frame per item | each item an object with `positions` and `cell` (an `mdir.State`; a `Frame` of `mdir.read_h5md`), a pair `(positions, cell)`, or `(N, 3)` positions alone; `cells` and `tilts` must be `None` |
| `cells` | `None`, `(3,)`, or `(K, 3)` float64 | the diagonal $a_x, b_y, c_z$ of the cell of every frame, or of each, in nm (the edges of an orthorhombic cell); `None`: the cell of the program's initial state, with its tilts, whatever was evaluated before |
| `tilts` | `None`, `(3,)`, or `(K, 3)` float64 | the tilts $b_x, c_x, c_y$ of the cell of every frame, or of each, the arrays of `mdir.Cell` and of `Borrow.tilt` (D238); `None` with `cells`: an orthorhombic cell |
| `gradient` | bool | whether the derivative in the tunables is kept or can be asked for with `vjp` |

The cell of a frame of an iterable is an `mdir.Cell` (or any object with
`diagonal` and `tilt`), the array of its three edges, or the `(3, 3)`
matrix of its vectors in rows, lower triangular. One function of
`python/mdir/_frames.py`, `_cell`, knows these shapes.

**Frames of a file.** `evaluator.evaluate(mdir.read_h5md(path))` evaluates
the frames of an H5MD trajectory (D239,
[python-h5md.md](python-h5md.md)), from `mdir.H5MDReporter` or from
`[output] trajectory` of `mdir run`: the reader gives one `Frame` at a
time, with its `positions` and its `cell`, tilts included, and can be read
again for a second pass. A `Frame` unpacked by a script, `(positions,
edges)` or, for a triclinic cell, `(positions, cell_vectors)`, is taken as
a pair as well. Frames of a file in f64 give the numbers of the states that
were written, to the bit; a file in f32 changes the energy of the dipeptide
in water by at most $2.1\times10^{-3}$ kJ/mol and the derivative by
$5\times10^{-7}$ relative (`python-frames-h5md.test`). Any other reader
plugs in the same way: an object that can be iterated more than once and
gives frames with `positions` `(N, 3)` and `cell`; `len()` is used if it
has one.

It returns a `FrameEnergies`:

| Attribute | Shape and dtype | Meaning |
|---|---|---|
| `energy` | `(K,)` float64, read-only, on the host | $U_{\boldsymbol\theta}(S_n)$ in kJ/mol: `gradient().energy` of D230 at the frame, the potential that the forces sample, with everything that the host adds at the volume of the frame |
| `virial` | `(K,)` float64 | the virial of `state().energies` at an evaluation of the frame, kJ/mol |
| `volume` | `(K,)` float64 | nm³ |
| `observables` | dict of names to `(K,)` float64 | the columns of `Program.plan["observables"]` (D232), kJ/mol and kJ/mol per unit of the constant |
| `depends` | dict of the names of the outputs (`"energy"`, `"virial"`, `"volume"`, each column) to frozensets of names of tunables | the tunables that the output may read at fixed positions; an empty set is a proof that it reads none |
| `version`, `count`, `zero` | int, int, frozenset | the value version of $\boldsymbol\theta$; $K$; the tunables whose derivative is zero by proof |
| `jacobian` | dict of names to `(K, M)` float64, or `None` | $\partial U_n/\partial\theta_m$, when it is kept ([below](#the-product-and-its-memory)) |
| `vjp(cotangent)` | `(K,)` float64 in; a `FrameGradient` out | $\sum_n g_n\,\partial U_n/\partial\boldsymbol\theta$: a read-only mapping of names to `(M,)` float64, with `zero` and `version`, as `TunableGradient` is at one state |

There is no kinetic energy and no pressure: a frame has no velocities. A
script forms the pressure from the virial, the volume, and the kinetic term
of its ensemble. `mdir.KB` is Boltzmann's constant of the programs,
0.0083144626181532 kJ/(mol K) (ruling Q8).

**Memory.** The frames stay where the caller keeps them (ruling Q12): the
evaluator reads them in batches of at most 64 MiB of float64 positions and
gives each frame to its simulation in turn, so an array on disk, float32
frames, and an iterable that makes its frames as they are asked for are
taken without a copy of the whole. What the evaluator allocates is $8K$
bytes per output and the Jacobian up to its bound. The frames are the large
part, and the caller's: #140's 38,000 frames of 21,403 atoms are 19.5 GB in
float64 and 9.8 GB in float32, and a `State` kept whole holds the velocities
and the forces as well, three times that.

**Refused.** With `InputError`, naming the frame (`frame 17: ...`): a shape
that is not `(K, N, 3)` with the program's $N$; a type that is not float64
or float32; a position that is not finite; an edge that is not finite or is
below twice the cutoff (the check of a commit, D229); no frames; `cells`
with an iterable; and what a commit refuses of a cell (D238): tilts for a
program compiled for an orthorhombic cell, or none for a triclinic one
(which of the two a program takes is fixed when it is compiled), a cell
that is not reduced, a pairlist distance above half of the least of the
diagonal. With `UnsupportedError`: positions or cells that require a gradient (a PyTorch tensor with
`requires_grad`), whose gradient is not returned yet and is not zero. A
frame whose evaluation fails (two particles at one place, a neighbor
structure that overflows) raises `SimulationError` naming the frame. After
any of these the evaluator is usable and gives what it gave before: the
state of its simulation is that of the last frame that succeeded. The
checks of a frame are made when its turn comes, so the frames before a
refused one have been evaluated; nothing of the call is returned.
`FrameEvaluator` itself refuses, with `InputError`, a program without
tunables or without `tunable_gradient`; `evaluate` refuses a program that
minimizes.

### The product and its memory

The cotangent of a reweighting, $\partial L/\partial U_n$, is known only
after every $U_n$ is: the product cannot be accumulated in the pass that
computes the energies. Two ways, chosen by size (ruling Q3):

- **The Jacobian is kept.** The host applies the chain rule of D230 at
  every frame and keeps the row $\partial U_n/\partial\boldsymbol\theta$,
  $K\times M_\theta$ numbers of f64 for $M_\theta$ entries in all; `vjp` is
  a product of matrices on the host (0.05 ms for 2 entries at 200 frames,
  0.37 ms for 23,558 at 100).
- **A second pass.** `vjp(cotangent)` evaluates the frames again and adds
  $g_n$ times the row of each, at the cost of the first pass. The evaluator
  keeps a reference to the frames it was given, not a copy; the values of
  the tunables must be those of the first pass, or `vjp` raises
  `SimulationError` naming the two versions (a second pass would
  differentiate another potential).

The Jacobian is kept when $8KM_\theta$ is at most `jacobian_bytes` (256 MiB
by default: the two entries of a K-Cl NBFIX at any $K$; 3,355 entries at
$10^4$ frames), and `out.jacobian` is `None` otherwise. Every charge of
23,558 particles at $10^4$ frames would take 1.9 GB, and takes the second
pass. For an iterable without a length the rows are kept until they pass
the bound and dropped then. An iterator that can be read only once (a
generator) is taken when the Jacobian is kept or with `gradient=False`; if
its rows pass the bound, `evaluate` raises `InputError` naming the first
frame beyond it, since the second pass would need the frames again.

`gradient=False` keeps no rows and refuses `vjp`. It does not make a pass
faster: the energy of D210 is that of `@tunable`, which is evaluated with
its derivative at every frame (measured: 1.047 ms against 1.050 ms per
frame on the dipeptide).

### The PyTorch adapter

```python
import mdir.torch                      # imports torch; `import mdir` does not
out = mdir.torch.evaluate(evaluator, theta, positions, cells=None, tilts=None)
out.energy                             # (K,) float64 tensor with a grad_fn in theta
out.virial, out.volume, out.observables[name]     # (K,) float64 tensors
out.depends, out.version, out.count, out.zero     # as FrameEnergies
```

- `theta` is a `dict[str, Tensor]` of float64 tensors of shape `(M,)`, on
  one device, for some or all of the tunables; a tunable that is not named
  keeps the value of the evaluator and takes no derivative. The forward
  updates the tunables whose bits changed (`tunables.update`, D213: no
  compilation), evaluates the frames, and returns tensors on the device of
  `theta` (ruling Q9).
- The outputs are $K$ numbers each, which the host computes in f64 in every
  precision mode (the energy of a frame is a sum over the device that the
  host completes with the self term, the tails, and the estimates of the
  shift): they are copied to the device of `theta`, 8 bytes per frame.
  Nothing that has $N$ rows crosses the adapter in this slice, so no device
  buffer is leased (D220, D229) and no stream is handed over; positions
  given as a tensor of a device are copied to the host.
- The backward of `energy` is the product above: with the Jacobian kept, a
  product of tensors on the device of `theta`; otherwise the second pass,
  which needs that the evaluator has evaluated nothing else since
  (`SimulationError` if it has). A name in `zero` takes zeros.
- `virial` and each observable follow `out.depends` (ruling Q5): an output
  that reads no tunable named in `theta` is a tensor without a graph,
  correctly a constant in $\boldsymbol\theta$. An output that may read one
  has a `grad_fn` whose backward raises `UnsupportedError`, naming the
  output and the tunables, when it is given a cotangent that is not zero:
  its explicit derivative $\partial O/\partial\boldsymbol\theta$ at fixed
  positions is not implemented in this slice, and leaving it out silently
  would give a wrong gradient (#140, T1b: the direct term and the
  covariance cancel). A script that means to drop the term says so with
  `.detach()`.
- `positions`, `cells`, and `tilts` that require a gradient are refused
  (`UnsupportedError`).
- **No double backward.** The product of the backward is an operation for
  which no derivative is registered, so a second differentiation raises
  PyTorch's `RuntimeError` for a tensor that does not require a gradient; it
  is not a silent zero. A Hessian in $\boldsymbol\theta$ is out of scope.
- One operation at a time on an evaluator, from any thread, with the GIL
  released during the frames; a second call while one runs raises
  `SimulationError` (D196).

**The operations** (ruling Q4). `mdir::frame_energies` and `mdir::frame_vjp`
are `torch.library.custom_op`s; the first has `register_autograd`, whose
backward calls the second, and both have shape functions
(`register_fake`). An operation takes tensors and numbers only, so the
evaluator and the frames of a call are found by a number, that of the
evaluator, which is the same at every call so that a traced function is not
traced again; the serial of the call travels as a tensor and tells a
backward by a second pass whether the record is still that of its forward.
The two functions of `mdir.torch.evaluate` that check the arguments and
build the outputs are marked `torch.compiler.disable`: a compiled function
breaks its graph around them and takes the operation itself into its
graphs.

**`torch.compile`.** With `backend="aot_eager"` a compiled loss gives the
average and its gradient of the eager one to the bit
(`python-frames-torch*.test`). The default backend (Inductor) did not run
in the test environment for a reason that is the environment's, not the
operation's: it compiles C++ with `-std=c++20`, which the `g++` 9.5 found
first there does not take, and a function without any MDIR operation fails
in the same way. With `CXX` set to `g++` 11 the default backend runs, and
the compiled loss of the test gives the average of the eager one to
$7\times10^{-15}$ and its gradient to $1.4\times10^{-12}$ relative (the
compiled softmax rounds differently).

### Use: a K-Cl NBFIX fitted to the osmotic pressure (#140)

```python
system.tunables = [mdir.Tunable("sig", "sig", term="nbfix"), mdir.Tunable("eps", "eps", term="nbfix")]
system.tunable_gradient = True
program = mdir.compile(system, state, integrator, ensemble, execution, schedule)
sim, evaluator = mdir.Simulation(program), mdir.FrameEvaluator(program)
theta = {n: torch.tensor(sim.tunables[n], requires_grad=True) for n in ("sig", "eps")}
optimizer, kT = torch.optim.SGD(theta.values(), lr=1e-6), mdir.KB * 300.0
for update in range(4):
    sim.tunables.update({n: t.detach().numpy() for n, t in theta.items()})
    frames = []                                 # (positions, cell): 10,000 frames, 2.6 GB in float32
    keep = lambda s, st: frames.append((st.positions.astype(np.float32), st.cell))
    sim.reporters.append(mdir.CallbackReporter(keep, period=500))
    sim.run(5_000_000); sim.close_reporters()
    with torch.no_grad():                       # the reference: U at theta-hat, same frames
        u_hat = mdir.torch.evaluate(evaluator, theta, frames).energy
    for inner in range(3):                      # several updates on one trajectory
        out = mdir.torch.evaluate(evaluator, theta, frames)
        w = torch.softmax(-(out.energy - u_hat) / kT, dim=0)
        n_eff = torch.exp(-(w * torch.log(w)).sum())
        if n_eff < 0.6 * len(frames): break     # sample again
        pressure = (out.observables["lower.d_z0"] - out.observables["upper.d_z0"]) / (2 * area)
        loss = ((w * (pressure - slope * concentration(frames))).sum() - target) ** 2
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```

The walls read neither `sig` nor `eps`, so their columns are constants in
$\boldsymbol\theta$ by proof and the loss differentiates through the
weights alone; `concentration` is the script's own function of the
positions. The frames are kept as float32 positions and the cell, not as
whole states: 21,403 atoms take 0.26 MB per frame so, against 1.5 MB for a
state in float64 with its velocities and forces.

## Order and identity

- **The order of the input, at every boundary.** Frames come in the order
  of the input, which is the order of trajectory files, of `state()`, and
  of other programs. The order of the program (D44, D215) is chosen anew at
  each evaluation, since a frame is not near the one before it; `ids` stays
  inside the evaluator. The outputs of this slice are numbers per frame and
  per entry of a tunable, which have no order of particles. When the
  gradient in the positions is added, the evaluator returns it in the order
  of the input, gathered by the `ids` of the same evaluation, so that the
  stale gather of #137 (item 1) cannot happen in a script.
- **Cells.** Each frame has its edges (constant pressure). What follows the
  volume follows it per frame, as `gradient()` does after steps under a
  barostat: the tails of pair terms and the correction for the dispersion
  ($\propto 1/V$), the part of the estimate of the shift that does not
  follow the volume (#224), and with PME the background of a net charge
  ($\propto 1/V$) and the self term (constant). The grid of PME and $\beta$
  are those of the program, as in a run under a barostat.
- **Triclinic cells** (ruling Q10; D238). A program compiled for a
  triclinic cell takes the cell of each frame as its diagonal and its
  tilts, which a barostat scales together; the evaluation of a frame sets
  the six numbers (`Simulation::setCell`) after the checks of a commit:
  the reduced form ($\lvert b_x\rvert\le a_x/2$, $\lvert c_x\rvert\le
  a_x/2$, $\lvert c_y\rvert\le b_y/2$), a diagonal of at least twice the
  cutoff, a pairlist distance of at most half of the least of the
  diagonal, and tilts that are not all zero. A frame is not reduced for
  the caller: a cell that is not reduced is refused with the index of its
  frame. The grid of PME and the neighbor capacity stay the program's.
- **Neighbor structures** are built anew at every frame; nothing is carried
  from one frame to the next. The tests hold that the frames give the same
  numbers to the bit in another order and one at a time.
- **Positions are taken as given**: not projected onto the constraints, not
  wrapped, not scaled with the cell, as a commit takes them (D229). Virtual
  sites are placed from their atoms at the start of the evaluation,
  whatever the frame says, as at the start of a run.
- **Velocities** are not read. The simulation of the evaluator keeps those
  of the program's initial state; the virial of a frame equals that of the
  sampler at the same positions to the bit, whatever its velocities
  (`python-frames.test`, with velocity Verlet). For a program of leapfrog
  the virial is returned from the same evaluation but is not checked
  against a reference: the sampler's `state().energies` is unset there.

## What is differentiated

| Quantity | In this slice | Later |
|---|---|---|
| $U_n$ in $\boldsymbol\theta$ | the product, with the three outcomes of D230 for each tunable: `"rule"`; `"zero"`, a proof, listed in `zero`; or a refusal at `mdir.compile` | |
| An output $O_n$ that reads no tunable at fixed positions | a constant, by a proof that `depends` states | |
| An output that may read a tunable (the virial; a column of a term that a tunable enters) | the value; its derivative is refused where it is asked for, never returned as zero | M2b-3: $\partial\mathsf W/\partial\boldsymbol\theta$, a mixed derivative |
| $U_n$ in the positions and the strain | not returned; a request is refused | the forces and the virial of the forward pass, as #138 says |
| Second derivatives | refused | out of scope |

`depends[name]` is built from the plan of the program. A column of
`observe` belongs to one term: it may read the tunable constants and
parameters of that term, and the charges, if they are tunable (an
expression can read `q1` and `q2`; the plan does not say whether this one
does, so a tunable of the charges is listed for every column, which is
safe and not sharp). The Lennard-Jones parameters of the types enter no
observed term (D230 refuses a pair term that reads them). The energy and
the virial may read every tunable that is not in `zero`; the volume reads
none.

**Which energy.** $U_n$ is the potential that the forces sample (D210), the
energy of `gradient()`: under a plain cutoff it is the shifted potential,
not the energy that the run reports. #140 measured what the other choice
costs: 18% in the gradient of the osmotic pressure.

**Charges with a constant of a pair term.** A tunable of the charges and a
tunable constant of a pair term in one program did not lower when the
evaluator was written (#256); since the fix of #260 they do, and the
evaluator takes such a program in both modes (`python-frames.test`,
scenario `dependent`).

## Reference energies and what is stored

The weights need $U_{\hat{\boldsymbol\theta}}(S_n)$, the same function at
the values that sampled the frame. **The reference is an evaluation by the
same evaluator at $\hat{\boldsymbol\theta}$ of the same stored frames**
(ruling Q6), not a number that the sampler wrote:

- the energy file holds the reported energy, which under a plain cutoff is
  not the sampled potential (D210), to six decimals of a kcal/mol;
- with both energies from one evaluator, the difference is 0 to the bit at
  $\boldsymbol\theta = \hat{\boldsymbol\theta}$ in the deterministic mode,
  so the weights are uniform and the effective sample size is $K$
  (`python-frames.test`), and the rounding of a file enters only as a small
  change of the sample.

It costs one more pass per trajectory, the one that the first forward
makes anyway.

### Which stored precision serves which reweighting

Stored frames are not the states that the dynamics saw: a DCD holds f32
and an XTC rounds to $10^{-3}$ nm. Rounding a frame raises its energy
through the stiff terms, by $+50 \pm 12$ kJ/mol at $10^{-3}$ nm for the
dipeptide in flexible water (the bonds; the paper, Section 3.6, derives
$kq^2/12$ per bond for a grid of spacing $q$). **That rise is of the terms
that the fit leaves fixed, and it cancels in the difference
$U_{\boldsymbol\theta}(\tilde S_n) - U_{\hat{\boldsymbol\theta}}(\tilde
S_n)$ when both energies are evaluated at the same rounded frame**, which
is what the evaluator does. It does not cancel for a tunable of a stiff
term itself, and not against a reference energy that the sampler recorded
at the frame it visited. (An earlier version of this document said that
frames rounded to the precision of XTC cannot be used for reweighting;
that holds for those two cases only, #264.)

Measured (`scripts/validation/frames/precision.py`, GPU, double precision,
#264): 400 frames of one run, kept in f64, rounded to f32, and rounded to
$10^{-3}$ nm; the same reweighting from each, to $\boldsymbol\theta =
\hat{\boldsymbol\theta} + \tfrac12k_BT/\operatorname{std}(\partial
U/\partial\theta)$, with both energies evaluated at the stored frame.
"Error" is against the frames in f64; the derivative is that of a
reweighted average, $-\mathrm{Cov}_w(O, \partial U/\partial\theta)/k_BT$,
with its error also in units of its statistical error from 10 blocks;
$\langle\partial U/\partial\theta\rangle_w$ is the reweighted mean of
the derivative, which needs no observable.

*The alanine dipeptide in 382 flexible waters (1,168 atoms, PME, 300 K),
with a soft pair term $a\,e^{-r/l}$ and an added harmonic term over the 764
O-H bonds of the water ($k$ = 50,000 kJ/mol/nm², $r_0$ = 0.09572 nm); $O$
is the distance between the two methyl carbons.*

| Tunable | Frames | Rise of $U$, kJ/mol | Largest error of $\Delta U$, kJ/mol | $N_\text{eff}$ of 400 | Largest change of a weight, in $1/K$ | Error of $\langle\partial U/\partial\theta\rangle_w$ | Error of the derivative, relative; in its statistical error |
|---|---|---|---|---|---|---|---|
| (a) $l$ of the soft pair term | f32 | $-7\times10^{-5} \pm 0.0017$ | 3.9e-6 | 312.33 | 6.1e-6 | 1.1e-10 | 1.4e-6; 3.1e-6 |
| | $10^{-3}$ nm | $+49.8 \pm 12$ | 0.032 | 312.26 (f64: 312.33) | 0.045 | 3.0e-6 | 1.3e-3; 0.003 |
| (b) the charge of the oxygens of the water, PME | f32 | the same | 1.9e-5 | 352.89 | 1.2e-5 | 7.8e-9 | 2.7e-6; 6.9e-6 |
| | $10^{-3}$ nm | the same | 0.11 | 352.67 (352.89) | 0.098 | 1.4e-4 | 6.4e-3; 0.017 |
| (c) $k$ of the O-H term, a stiff term | f32 | the same | 1.1e-4 | 356.33 | 9.5e-5 | 2.3e-9 | 2.6e-4; 2.5e-5 |
| | $10^{-3}$ nm | the same | 1.6 | 353.68 (356.33) | 0.56 | 2.4e-2 | 2.7; 0.25 |
| (c) $r_0$ of the O-H term | f32 | the same | 9.1e-5 | 353.69 | 6.1e-5 | 6.2e-8 | 1.6e-5; 2.1e-5 |
| | $10^{-3}$ nm | the same | 0.69 | 352.18 (353.69) | 0.43 | 2.9e-3 | 0.18; 0.24 |

*60 A + 60 B Lennard-Jones particles at 100 K, no stiff term, $\sigma'$ of
the A-B pair term tunable; $O$ is the number of A below a plane.*

| Frames | Rise of $U$, kJ/mol | Largest error of $\Delta U$, kJ/mol | $N_\text{eff}$ of 400 | Error of $\langle\partial U/\partial\theta\rangle_w$ | Error of the derivative, relative; in its statistical error |
|---|---|---|---|---|---|
| f32 | $-1\times10^{-6} \pm 4\times10^{-5}$ | 5.9e-6 | 355.10 | 8.5e-8 | 1.1e-6; 1.5e-6 |
| $10^{-3}$ nm | $+0.03 \pm 0.22$ | 0.032 | 355.09 (355.10) | 7.1e-4 | 0.035; 0.047 |

What the numbers say:

- **f32** (a DCD, or H5MD with `positions="f32"`) serves every tunable
  measured: the error of $\Delta U$ is at most $10^{-4}$ kJ/mol and that of
  a derivative $3\times10^{-5}$ of its statistical error.
- **$10^{-3}$ nm** (an XTC at its default precision) serves a tunable of a
  soft nonbonded term and of the charges: the error of $\Delta U$ is 0.03
  and 0.11 kJ/mol ($0.01$ and $0.04\,k_BT$), the number of effective
  frames does not change, and a derivative moves by under 2% of its
  statistical error at 400 frames. The error does not average out with more
  frames, so a long fit meets it at about $10^{-3}$ to $10^{-2}$ relative.
- **A tunable of a stiff term** is not served by $10^{-3}$ nm: the error of
  $\Delta U$ reaches 1.6 kJ/mol ($0.64\,k_BT$) for the force constant, the
  weights change by half of $1/K$, and $\langle\partial U/\partial
  k\rangle_w$, a sum of $(r - r_0)^2$ that the rounding adds to directly,
  is 2.4% off. The Python model has such tunables (a parameter of a tuple
  term), so this is measured, not estimated.

**The reference from the run.** The same reweighting with
$U_{\hat{\boldsymbol\theta}}$ taken from the energy that the sampler
reported at the frame it visited (less a constant), in place of an
evaluation at the stored frame:

| System, tunable | Frames | Spread of the recorded energy less the evaluator's, kJ/mol | $N_\text{eff}$ of 400 (evaluated reference) | Error of the derivative, in its statistical error |
|---|---|---|---|---|
| Dipeptide, (a) | f64 | 0.038 | 312.14 (312.33) | 1e-4 |
| Dipeptide, (b) | f64 | 0.038 | 353.01 (352.89) | 0.010 |
| Dipeptide, (c) $k$ | f64 | 0.038 | 356.18 (356.33) | 0.016 |
| Dipeptide, (a), (b), (c) | $10^{-3}$ nm | the rise, $50 \pm 12$ | 9.6, 9.0, 8.1 | 7.0, 6.8, 3.4 |
| Mixture, plain cutoff | f64 | 0.24 ($0.28\,k_BT$) | 335.70 (355.10) | 0.83 |
| Mixture, plain cutoff | $10^{-3}$ nm | 0.24 | 324.16 (355.09) | 0.94 |

With rounded frames the rise no longer cancels and the weights collapse
to 9 effective frames of 400. Even with frames in f64 the recorded energy
is not the reference: it is the energy that the run reports, which under a
plain cutoff is not the sampled potential (D210) and differs from it by a
term that fluctuates with the number of pairs within the cutoff, 0.24
kJ/mol in the mixture, enough to move a derivative by 0.8 of its
statistical error; in the dipeptide, where only the direct sum of PME is
cut without a shift, the spread is 0.038 kJ/mol. Hence ruling Q6: the
reference is evaluated.

**f64 stays the default of stored frames** (`H5MDReporter`, D239;
`CallbackReporter` gives float64 copies, D207), for the reason that the
table gives and not because rounded frames fail in general: a trajectory in
f64 can be reused for any term and any tunable chosen later, the stiff ones
included, and it costs twice the bytes of f32. Frames in f32 are a sound
economy for fits of nonbonded parameters; an XTC serves such fits to about
$10^{-2}$ of a gradient and should not be used for bonded parameters or
with recorded reference energies. A larger system was not measured.

**The version of $\boldsymbol\theta$.** `out.version` is that of the
evaluator's values. The sampler already records its own: the column
`tunables_version` of the energy and observables files, and
`sim.tunables.history` with the step from which each version holds (D213).
Nothing is added to a trajectory file.

## Implementation

- `compiler::Simulation::evaluateFrame(positions, cell, gradient)` is the
  single evaluation of a frame. It checks the frame (the cell with
  `checkCommittedCell` of D238), makes the state of the host the frame (the
  positions given, the six numbers of the cell with `setCell`; the
  velocities and forces that the host has), ends the activation of the
  entry, and begins one from that state with the branch of the derivative
  (`evaluatePart` with `%tunable_gradient = 1`; the start of a run for the
  first frame of an evaluator, `%first_call = 2` after it): one start of
  an activation per frame, where a borrow, a commit, and `gradient()` take
  two and copy the state of the device to the host in between. The state of
  the last frame is not copied back from the device: a frame is not a
  continuation of the one before. If the evaluation fails, the positions
  and the cell of before are restored and the simulation has not failed.
  `collectTunableGradient` is the chain rule of D230, which `gradient()`
  shares.
- `Simulation._evaluate_frames` (Python, private) runs a batch of frames
  with the GIL released and returns the columns and the rows; an error
  carries the index of its frame.
- `python/mdir/_frames.py` holds `FrameEvaluator`, `FrameEnergies`,
  `FrameGradient`, the reading of the frames in batches, and `depends`;
  `python/mdir/torch.py` the operations. Both are files of the package in
  the build tree and in the wheel.
- The second pass applies the chain rule at each frame and adds the rows on
  the host; accumulating the fields on the device and applying it once
  ([python-gradient.md](python-gradient.md), "At $K$ frames") belongs to
  the slice that keeps the frames inside one activation.
- No kernel, runtime function, pass, or program text changes, and nothing
  on the path of a step.

## Validation

`test/Driver/Inputs/python_frames.py`; `python-frames.test` (CPU),
`python-frames-gpu.test`, `python-frames-torch.test`,
`python-frames-torch-gpu.test` (`REQUIRES: torch`, PyTorch 2.14.1), in
double and mixed precision, in the deterministic mode unless stated.
"FD" is a central difference extrapolated from two steps (Richardson).

| Quantity | Reference | CPU double | GPU double | CPU mixed | GPU mixed | Tolerance (double, mixed) |
|---|---|---|---|---|---|---|
| $U_n$, the row $\partial U_n/\partial\boldsymbol\theta$, the virial, and the four observed columns of 5 frames of the dipeptide in water (1,168 atoms, PME; 25 tied charges and $\sigma$, $\epsilon$ of the pair OW-OW tunable; an observed pair term and an observed wall) | the sampler's own `gradient()` and `state()` at the step of the frame | 0 | 0 | 0 | 0 | 0 |
| The same frames as an array, in another order, as pairs from a generator, and one alone | the first evaluation | 0 | 0 | 0 | 0 | 0 |
| The same in mixed precision without the deterministic mode, 3 frames: the energy (kJ/mol) and the row | the sampler's `gradient()` | | | 0, 0 | 0, 4.8e-9 | 5e-3 kJ/mol, 1e-5 |
| `vjp` with the Jacobian kept, and by a second pass | the cotangent times the rows; the first | 1.3e-16, 0 | 0, 0 | 0, 0 | 1.3e-16, 0 | 1e-14, 1e-13 |
| `vjp`, the three largest entries of the charges and $\sigma$, $\epsilon$, relative to the largest entry of each tunable | FD in $\boldsymbol\theta$ of $\sum_n g_nU_n$ | 2.1e-10 | 2.1e-10 | 1.6e-4 | 1.6e-4 | 1e-7, 5e-4 |
| $U_n$ and the row of 4 frames of leapfrog under a barostat, each with its cell (20.4444 to 20.4848 nm³): plain cutoff, correction for the dispersion, PME; $\sigma$ and $\epsilon$ of the 45 pairs of types and 25 tied charges | the sampler's `gradient()` at the frame | 0 | 0 | 0 | 0 | 0 |
| The first and the last of those frames: the energy (kJ/mol) and the row | `gradient()` of a simulation compiled from the frame, whose cell is then the cell of the program (what follows the volume is computed at it, not scaled; the same grid of PME) | 0, 1.5e-18 | 0, 1.5e-18 | 0, 1.5e-18 | 0, 1.5e-18 | 1e-9, 1e-12; 5e-3, 1e-5 |
| `vjp` over those frames at other values than sampled them (a net charge of 0.5 e, $\sigma$ scaled by 1.01), the three largest entries of each tunable | FD of $\sum_n g_nU_n$, each energy at the volume of its frame | 7.2e-11 | 6.3e-11 | 1.5e-4 | 1.5e-4 | 1e-7, 5e-4 |
| 60 A + 60 B with the A-B pair term's $\sigma'$ tunable, 150 frames at 100 K: at $\hat\sigma'$ the energies against the reference pass, the weights against $1/K$, $N_\text{eff}$ against $K$ | equal | 0 | 0 | 0 | 0 | 0 |
| $d\langle O\rangle_{\sigma'}/d\sigma'$ from the energies and the rows, $-(\langle OG\rangle_w - \langle O\rangle_w\langle G\rangle_w)/k_BT$ with $O$ the number of A below $z = 1.6$ nm (an observed step of strength 0): at $\hat\sigma'$ ($-70.52$ /nm) and at $\hat\sigma' + 0.002$ nm ($N_\text{eff} = 118$; $+38.80$ /nm) | FD in $\sigma'$ of the reweighted average on the same frames | 7.0e-7, 4.6e-7 | 7.0e-7, 4.6e-7 | 9.9e-3, 4.3e-3 | 9.9e-3, 4.3e-3 | 1e-5, 3e-2 |
| The same derivative from `loss.backward()` through `mdir.torch.evaluate` | the covariance formed from the arrays of the evaluator | 2.3e-13 | 2.0e-13 | 3.4e-13 | 1.2e-13 | 1e-11 |
| The backward by a second pass (`jacobian_bytes=0`) | the backward with the Jacobian kept | 2.2e-15 | 3.7e-16 | 1.3e-15 | 1.9e-15 | 1e-12 |
| `torch.autograd.gradcheck` of the reweighted average of 12 frames (`eps` 1e-5, `atol` 1e-6, `rtol` 1e-4) | finite differences of PyTorch | passes | passes | | | |
| A fit of $\sigma'$ from $\hat\sigma' = 0.3700$ nm to the reweighted average at $\sigma^\ast = 0.3705$ nm, 12 Newton steps with the derivative of autograd: the distance from $\sigma^\ast$ | $\sigma^\ast$ | 3.9e-16 nm | 2.8e-16 nm | 3.8e-8 nm | 9.8e-8 nm | 1e-9, 1e-5 nm |
| The reweighted average and its gradient under `torch.compile(backend="aot_eager")` | eager | 0 | 0 | 0 | 0 | 0 |

The fit is to a synthetic target, the reweighted average of the same
frames at $\sigma^\ast$, which is reachable exactly; the average of these
150 frames is monotonic in $\sigma'$ only up to about 0.371 nm, so the
target is 0.0005 nm away. A comparison with an independent run at
$\sigma^\ast$, within its statistical error, is #140's Step 2 and is not
a test of the suite.

In mixed precision the central differences of a reweighted average are
those of energies that are sums of f32 terms inside an exponential, over a
step of $\sigma'$ that the curvature of the average bounds ($N_\text{eff}$
falls from 150 to 118 over 0.002 nm): they resolve the derivative to about
1%. The derivative itself is the covariance of the arrays, which the
autograd of PyTorch reproduces to $2\times10^{-13}$ in both modes.

Also checked (`refusals`, and the scenario `torch`): the refusals of
[Interface](#interface), each with its error and the index of its frame,
and that the evaluator gives the energies of before after each; the
Jacobian at its bound (kept for 2 frames, dropped for 3; a generator
refused at frame 2); `vjp` after the values changed, with the Jacobian kept
(the product of the values evaluated) and without it (`SimulationError`);
`depends` of each output; a constant that its expression does not read
(`zero`, a column of zeros); a backward through the virial and through a
column that the tunable enters (`UnsupportedError`), and the same loss with
those outputs detached; a second backward (`RuntimeError`); positions that
require a gradient; tensors of another type or name; the records of the
adapter empty after every call.

### Triclinic frames and frames of a file

`python-frames-h5md.test`, `python-frames-h5md-gpu.test` (`REQUIRES:
hdf5`). 20 frames of a run of `mdir run` at constant pressure in a rhombic
dodecahedron of 403 waters (C-rescale, rigid water; $c_x$ from 1.3139 to
1.3813 nm, volumes 12.83 to 14.91 nm³), written in H5MD in f64 and read
with `mdir.read_h5md`; the evaluator's program has flexible water, PME on
the grid of the first cell, 11 tied charges and $\sigma$ of OW-OW
tunable. (A Python simulation refuses a barostat in a triclinic cell,
#255, so the frames are those of `mdir run`.)

| Quantity | Reference | CPU double | GPU double | CPU mixed | GPU mixed | Tolerance (double, mixed) |
|---|---|---|---|---|---|---|
| $U_n$, the row, the virial, and the volume of frames 0, 6, 13, and 19 of the triclinic run | `gradient()` and `state()` of a simulation compiled from the frame, whose cell is then the cell of the program | 0 | 0 | 0 | 0 | 0 |
| The 20 frames as arrays `(K, N, 3)` with `cells` and `tilts`, as pairs with an `mdir.Cell`, and as pairs with the `(3, 3)` cell vectors that a `Frame` unpacks to | the frames of the reader | 0 | 0 | 0 | 0 | 0 |
| `vjp` over the 20 frames, the three largest entries of the charges and $\sigma$ | FD in $\boldsymbol\theta$ of $\sum_n g_nU_n$ | 1.1e-10 | 7.1e-11 | 1.6e-4 | 1.6e-4 | 1e-7, 5e-4 |
| 5 frames that an `H5MDReporter` wrote in f64 from a run of the dipeptide, read one at a time: the energy, the virial, the rows | the evaluation of the states of the same steps (`CallbackReporter`) | 0 | 0 | 0 | 0 | 0 |
| The same from a file in f32: the energy (kJ/mol), the rows (relative) | the same | 2.1e-3, 5.2e-7 | 2.1e-3, 5.2e-7 | 5.9e-4, 2.5e-7 | 3.9e-4, 1.9e-7 | 2e-2 kJ/mol, 1e-4 |

Also checked there: without cells, a triclinic program takes its own cell
with its tilts; a cell that is not reduced, a triclinic program given
cells without tilts, and a diagonal below twice the cutoff are refused
with the index of their frame, in the words of a frame, and the evaluator
gives the energies of before after each; in `python-frames.test`, tilts
given to a program compiled for an orthorhombic cell, and a matrix of cell
vectors that is not lower triangular.

No device code changed, so the compute-sanitizer suite was not run.

## Performance

One RTX 3090 (GPU 0 under its lock, 300 W cap), PME, a cutoff of 0.8 nm,
1 fs; tunables: $\sigma$ and $\epsilon$ of the pair of types OW-OW, or
every charge; frames kept from a run in float64, 20 steps apart. One run
each.

### The loop that the interface allowed before

For each frame: `borrow()`, the positions written through DLPack in the
order of `ids` (PyTorch 2.14.1), `commit()`, `tunables.gradient()`, on a
second simulation of the program. Times in ms.

| System | Mode | Step | `run(0, energy=True)` | `gradient()` | Write | Commit | `gradient()` in the loop | Per frame | Frames/s | In steps |
|---|---|---|---|---|---|---|---|---|---|---|
| Dipeptide in water, 1,168 atoms | GPU, mixed | 0.068 | 0.98 | 1.09 | 0.33 | 1.08 | 1.11 | 2.55 | 392 | 37 |
| The same | GPU, double | 0.334 | 1.40 | 1.80 | 0.54 | 1.60 | 1.91 | 4.08 | 245 | 12 |
| The same, 20 frames | CPU, double | 13.2 | 31.4 | 43.5 | 0.46 | 33.3 | 44.7 | 78.5 | 12.7 | 6 |
| JAC, 23,558 atoms | GPU, mixed | 0.222 | 7.39 | 7.86 | 0.50 | 8.53 | 7.71 | 16.8 | 59.6 | 76 |
| JAC, every charge tunable (23,558 entries) | GPU, mixed | 0.224 | 6.99 | 7.78 | 0.64 | 7.72 | 7.83 | 16.3 | 61.4 | 73 |
| JAC, 10 frames | CPU, double | 399 | 945 | 1,282 | 1.6 | 919 | 1,264 | 2,185 | 0.46 | 5.5 |

The CPU rows ran with the default `Execution.threads` on a shared host: they show the proportions, not a rate of the CPU.

Two evaluations per frame, each the start of an activation (D215). Under
`MDRT_PROFILE`, one `gradient()` on JAC (8.5 ms) makes 58 copies between
host and device that take 4.0 ms, 86 launches (0.41 ms), and 62 waits
(0.44 ms); the remaining 3.6 ms are host work outside the counters of the
device runtime, not broken down further. On the dipeptide (1.26 ms): 58
copies, 0.41 ms; 74 launches, 0.27 ms; 62 waits, 0.20 ms. What each copy
carries was not measured.

### `FrameEvaluator.evaluate`

| System | Mode | Frames | Step, ms | `gradient()`, ms | Per frame, ms | Frames/s | In steps | Target of the design | The loop before |
|---|---|---|---|---|---|---|---|---|---|
| Dipeptide, 1,168 atoms | GPU, mixed | 400 | 0.071 | 1.14 | 1.05 | 952 | 14.9 | 1.1 ms | 2.55 ms |
| The same | GPU, double | 200 | 0.328 | 1.87 | 1.82 | 550 | 5.5 | | 4.08 ms |
| JAC, 23,558 atoms | GPU, mixed | 200 | 0.222 | 7.51 | 6.85 | 146 | 30.9 | 7.8 ms | 16.8 ms |
| JAC, every charge tunable | GPU, mixed | 100 | 0.223 | 7.72 | 7.22 | 139 | 32.3 | | 16.3 ms |
| JAC | GPU, double | 100 | 5.11 | 22.6 | 21.9 | 46 | 4.3 | | |

A frame costs a little less than `gradient()` (the state of the last frame
is not copied back to the host) and 2.3 to 2.5 times less than the loop
before. The second pass of `vjp` costs a pass (0.98 ms and 6.82 ms per
frame); with the Jacobian kept `vjp` takes 0.05 ms for the whole set (0.37
ms for 23,558 entries at 100 frames). The evaluator and its first frame
take 0.08 s on the dipeptide and 0.87 s on JAC (D236). A frame is still
the start of an activation: 15 and 31 steps of the same program. For
#140's system (21,403 atoms; 38,000 frames at one per ps of 4 × 9.5 ns), a
pass is about 4.5 minutes.

The steps of a run are not touched: no program text and no code on the
path of a step changes, so the rate of `mdir run` and of a Python
simulation is that of main.

### Target of the next slice

Not measured. The frames inside one activation: the entry takes the
positions and the cell of the next frame at a boundary, builds its neighbor
structures, and evaluates `@tunable`, without the copies and the host's
part of a start. At most 5 steps per frame: 0.35 ms on the dipeptide
(2,900 frames/s) and 1.1 ms on JAC (900 frames/s), and under a minute for
a pass of #140.

## The dependent terms alone (D243)

Issue #264. Status: implemented, as the maintainer ruled on PR #266
([Maintainer rulings (dependent terms)](#maintainer-rulings-dependent-terms)).
**The mode saves time, 2 to 2.5 times a frame for a tunable of a pair term
and 1.45 to 1.9 times for the charges; it does not change the accuracy of a
difference in mixed precision** ([measured](#what-the-mode-saves)).

A reweighting takes differences at one frame, $U_{\boldsymbol\theta}(S_n)
- U_{\hat{\boldsymbol\theta}}(S_n)$, and the derivative in
$\boldsymbol\theta$. With

$$
U = U_\text{fixed} + U_\text{dep}(\boldsymbol\theta),
$$

$U_\text{dep}$ the terms that a tunable enters, the fixed part cancels in
both. An evaluator of $U_\text{dep}$ alone gives the same weights and the
same product and evaluates less.

### Interface

```python
evaluator = mdir.FrameEvaluator(program, terms="dependent")   # "all" is the default
evaluator.terms                 # "dependent"
evaluator.program.plan["terms"] # what is kept: ["pair:nbfix"], ["lennard_jones", "coulomb"], ...
out = evaluator.evaluate(frames)
out.dependent_energy            # (K,) float64: U_dep at each frame, with its tails at the volume of the frame
out.energy                      # raises UnsupportedError: this evaluator has no potential energy
out.jacobian, out.vjp(g)        # those of U_dep, which are those of U
out.volume                      # as before
out.observables                 # the columns of the terms that are kept; the others are absent
out.unavailable                 # ("energy", "virial", "wall.energy", ...): what this mode does not give
```

- `energy` is not reused for $U_\text{dep}$: a number named `energy` would
  be taken for the potential. With `terms="all"`, `dependent_energy` raises
  in the same way (`out.unavailable` is then `("dependent_energy",)`), so a
  script written for one mode fails loudly in the other, and energies of
  the two modes cannot be subtracted by accident. `mdir.torch.evaluate`
  returns the same names, and its backward is the same product.
- The virial of $U_\text{dep}$ is not the virial, so `virial` raises too.
  The columns of `observe` of the terms that are left out are not in
  `observables`; `out.unavailable` lists them. `depends` holds what it
  held for the outputs that remain, under `"dependent_energy"` for the
  energy.
- The reference energies of a reweighting are `dependent_energy` at
  $\hat{\boldsymbol\theta}$ from the same evaluator.
- `evaluator.program` is the program that is evaluated: with
  `terms="dependent"` a second program, which the evaluator compiles from
  the model of the program it is given (`Program.plan["terms"]` lists its
  terms; that of any other program is `["all"]`). It is not a program to
  sample with: its forces are those of the dependent terms alone.

### Which terms are kept

The proofs are those behind `zero` and `depends`: a term is kept if a
tunable that is not `"zero"` enters it.

| Tunable | Kept (`plan["terms"]`) | Why not less |
|---|---|---|
| A constant of a pair term | that pair term, over its groups, with its tail and the estimate of its shift at the volume of the frame (`"pair:<name>"`) | |
| A parameter of a tuple term | that tuple term (`"tuple:<name>"`) | |
| $\sigma$, $\epsilon$ per type, or by pairs of types | the Lennard-Jones of all pairs within the cutoff, the correction for the dispersion and the estimate of its shift (`"lennard_jones"`) | the kernel runs over the neighbor structure of all particles; the pairs of other types are computed and cancel. A restriction to the particles of the types concerned would be a relation of its own, a later step |
| Charges | the direct sum, the excluded pairs, the 1-4 pairs, the reciprocal sum, the self term and the background; a pair term that reads `q1` or `q2` (`"coulomb"`) | the reciprocal sum is one quadratic form of all the charges, $\tfrac12\mathbf q^{\mathsf T}\mathsf A\mathbf q$: the part of the water with itself cannot be left out of a mesh, even where only the charges of a solute are tunable |

Left out: the bonds, angles, dihedrals, impropers, Urey-Bradley terms, and
CMAPs of the topology, the Lennard-Jones of the 1-4 pairs (which do not
follow the tunable table, D213), the terms of the positions (no tunable
enters one), every pair or tuple term without a tunable, the Lennard-Jones
without a tunable $\sigma$ or $\epsilon$, and the electrostatics, PME and
its mesh included, without tunable charges. What the host adds to the
energy follows its term: a tail that depends on $\boldsymbol\theta$ and on
the volume of the frame stays with its term, and one of a term that is
left out goes with it.

### A second program, not a branch

A frame is the start of an activation, which builds the buffers of the
whole model and evaluates the forces of the whole potential before
`@tunable` is reached; `@tunable` itself is 6% to 11% of an evaluation. A
branch in one program that evaluated a smaller `@tunable` would save a
part of that tenth. The evaluator therefore compiles a program whose every
potential has the dependent terms only (`Control::dependentTerms`, set by
`Program._dependent()`; `getDependentTerms` of `lib/Driver/Builder.cpp`
names them, and `Builder::emitTopologyPotential`, the correction for the
dispersion, and the tails of the pair terms ask it): no forces of the other
terms, no mesh without tunable charges, no observed potentials of terms
left out. The steps of the sampler's program are untouched, and D236
applies to the second program as to any. Making the evaluator costs one
more compilation, of a smaller program: 1.4 s to 2.6 s on the dipeptide
and 2.7 s to 4.1 s on JAC with its first frame, on a GPU.

The tuples of the bonded terms are still built and copied at the start of
an activation, though no term reads them; leaving them out is a further
saving that was not taken here.

### What the mode saves

One RTX 3090 (GPU 0 under its lock, 300 W cap), PME, a cutoff of 0.8 nm,
the deterministic mode; ms per frame of `evaluate`, one run each.

| System | Mode | Tunable | Kept | $U_\text{dep}$ of $U$ | The whole potential | The dependent terms | Ratio |
|---|---|---|---|---|---|---|---|
| Dipeptide, 1,168 atoms, 300 frames | GPU, mixed | $l$ of a pair term $a\,e^{-r/l}$ | `pair:soft` | 85 of 14,715 kJ/mol | 1.044 | 0.514 | 2.0 |
| | | the charges of the water | `coulomb` | 19,484 of 14,838 | 1.129 | 0.780 | 1.45 |
| | | $\sigma$ of the pair OW-OW | `lennard_jones` | 3,081 of 14,838 | 1.074 | 0.583 | 1.8 |
| JAC, 23,558 atoms, 150 frames | GPU, mixed | $l$ of the pair term | `pair:soft` | 1,855 of 307,486 | 7.436 | 2.971 | 2.5 |
| | | the charges of the water | `coulomb` | 410,959 of 309,283 | 7.677 | 4.121 | 1.9 |
| | | $\sigma$ of the pair OW-OW | `lennard_jones` | 56,008 of 309,283 | 7.272 | 3.230 | 2.25 |
| Dipeptide, 200 frames | GPU, double | $l$ of the pair term | `pair:soft` | | 2.140 | 0.918 | 2.3 |
| JAC, 60 frames | GPU, double | $l$ of the pair term | `pair:soft` | | 27.72 | 11.28 | 2.5 |

A frame of the dependent terms is still the start of an activation: 7 to
11 steps on the dipeptide and 11 to 16 on JAC.

**The next slice, and its bound.** What a frame costs beyond its kernels
is fixed by the start of an activation, in either mode. Under
`MDRT_PROFILE` on JAC in mixed precision, with the constant of a pair term
tunable, per frame over 300 frames:

| | The whole potential | The dependent terms |
|---|---|---|
| A frame | 6.99 ms | 2.72 ms |
| Copies between host and device | 50, 3.37 ms | 18, 1.13 ms |
| Launches of kernels (the build of the neighbor structure and the evaluation) | 85, 0.38 ms | 46, 0.13 ms |
| Waits | 51, 0.41 ms | 19, 0.21 ms |
| The rest: host work of the start outside the counters of the device runtime | 2.8 ms | 1.25 ms |

The kernels of a frame, the neighbor build included, are 0.79 ms of 6.99
and 0.34 ms of 2.72; the other 6.2 ms and 2.4 ms are the copies and the
host's part of a start, which are paid again at every frame and carry the
same data each time but the positions and the cell. The frames inside one
activation (the entry takes the positions and the cell of the next frame
at a boundary, builds its neighbor structures, and evaluates) would leave
the launches, the waits, and one copy of the positions: about 0.9 ms a
frame for the whole potential and 0.4 ms for the dependent terms on JAC, 8
and 7 times less than now, as a bound from these counters and not a
measurement. Two smaller items inside that: the tuples of the bonded terms
are among what a program of the dependent terms still builds and copies at
a start, though no term reads them (what the 18 copies carry was not
separated); and with Lennard-Jones tunables the kernel still runs over all
pairs (T6). None of the three is implemented here.

**Accuracy in mixed precision: unchanged.** The difference of two sums of
the size of $U_\text{dep}$ is not more accurate than that of two sums of
the size of $U$ in MDIR's mixed mode: the energy of each term is
accumulated in f64 from
contributions computed in f32, so the fixed terms give the same f64 number
at both values of $\boldsymbol\theta$ and cancel exactly, and the error of
a difference is that of the contributions of the dependent terms in either
mode. Measured against the difference of the whole potential in double
precision, at the same frames:

| System, tunable | Size of $\Delta U$, kJ/mol | Error of $\Delta U$, the whole potential (max; rms) | Error of $\Delta U$, the dependent terms (max; rms) | The two modes against each other |
|---|---|---|---|---|
| Dipeptide, $l$ | 8.9 | 6.9e-6; 5.6e-6 | 6.9e-6; 5.6e-6 | 2.3e-11 |
| Dipeptide, charges | 31 | 5.1e-3; 4.5e-3 | 5.2e-3; 4.5e-3 | 3.7e-4 |
| Dipeptide, $\sigma$ | 210 | 3.3e-3; 2.7e-3 | 3.2e-3; 2.7e-3 | 4.1e-4 |
| JAC, $l$ | 198 | 1.4e-4; 1.3e-4 | 1.4e-4; 1.4e-4 | 3.8e-10 |
| JAC, charges | 709 | 8.6e-2; 8.3e-2 | 8.6e-2; 8.3e-2 | 1.8e-3 |
| JAC, $\sigma$ | 4,155 | 5.8e-2; 5.5e-2 | 5.7e-2; 5.5e-2 | 1.3e-3 |

The Jacobian of the two modes is equal to the bit in every case, and
within $7\times10^{-7}$ of that of double precision. The error of a
difference in mixed precision, up to 0.09 kJ/mol for the charges of 23,558
particles, is a property of the f32 contributions; a fit that needs less
evaluates in double precision.

### Validation of the dependent terms

`python-frames.test`, `python-frames-gpu.test` (scenario `dependent`), and
the scenario `torch`; the deterministic mode. Programs of the
dipeptide in water with PME, 4 frames each: a constant of a pair term with
a parameter of a tuple term and an observed wall (`pair:soft`,
`tuple:spring`; $U_\text{dep}$ is 1.3% of $U$); 25 tied charges
(`coulomb`); $\sigma$ and $\epsilon$ of the 45 pairs of types with the
charges, under a plain cutoff with the correction for the dispersion, over
frames of a run under a barostat (`lennard_jones`, `coulomb`). $\Delta U$
is the difference of the energies at $\boldsymbol\theta = 1.01\,
\hat{\boldsymbol\theta}$ and at $\hat{\boldsymbol\theta}$.

| Quantity | Reference | CPU double | GPU double | CPU mixed | GPU mixed | Tolerance (double, mixed) |
|---|---|---|---|---|---|---|
| $\Delta U$ of the dependent terms, kJ/mol: `pair:soft`, `tuple:spring` ($\Delta U$ = 4.3) | $\Delta U$ of the whole potential | 1.5e-11 | 3.7e-12 | | | 1e-9 |
| The same: `coulomb` ($\Delta U$ = 309) | the same | 1.5e-11 | 0 | | | 1e-9 |
| The same: `lennard_jones`, `coulomb`, frames under a barostat ($\Delta U$ = 220) | the same | 1.5e-11 | 1.5e-11 | | | 1e-9 |
| In mixed precision, $\Delta U$ of the whole potential; of the dependent terms: `pair:soft`, `tuple:spring` | $\Delta U$ of the whole potential in double precision | | | 1.0e-5; 1.0e-5 | 2.4e-5; 2.4e-5 | the dependent terms at most 1.5 times the whole |
| The same: `coulomb` | the same | | | 2.2e-3; 2.0e-3 | 2.3e-3; 2.3e-3 | the same |
| The same: `lennard_jones`, `coulomb` | the same | | | 3.2e-3; 3.2e-3 | 3.2e-3; 3.2e-3 | the same |
| Tunable charges with a tunable constant of a pair term in one program (#256, #260; `coulomb`, `pair:soft`; $\Delta U$ = 305): $\Delta U$ of the dependent terms in double precision; in mixed precision, of the whole potential; of the dependent terms | $\Delta U$ of the whole potential, in double precision | 4.4e-11 | within the tolerance | 2.9e-3; 2.7e-3 | within the tolerance | 1e-9; the dependent terms at most 1.5 times the whole |
| The Jacobian of the dependent terms, and `vjp`, in the four programs | those of the whole potential | 0 | 0 | 0 | 0 | 1e-12, 1e-6 relative |
| 60 A + 60 B, 150 frames, at $\hat\sigma' + 0.002$ nm: the reweighted average; its derivative | those from the whole potential | 1.2e-16; 2.3e-13 | 0; 1.1e-13 | 0; 0 | 0; 0 | 1e-10, 1e-3 relative |
| The gradient of that average from `loss.backward()` through `mdir.torch.evaluate` with `terms="dependent"` | that with the whole potential | | 1.0e-13 | 0 | | 1e-10, 1e-3 relative |

Also checked: `plan["terms"]` of each program; `energy` and `virial` of the
dependent mode and `dependent_energy` of the whole raise
`UnsupportedError` naming the other; `unavailable` and `depends`; the
columns of the pair term that stays equal those of the whole potential,
and those of the wall are absent; the volumes are equal; `terms` other
than the two is refused.

## Without frames: a potential linear in its tunables

Where $U_\text{dep}$ is linear in the tunables,
$U_\text{dep} = \sum_k\theta_kA_k(\mathbf x)$, the sums $A_k =
\partial U/\partial\theta_k$ of a frame are all that a reweighting takes of
it:

$$
U_{\boldsymbol\theta}(S_n) - U_{\hat{\boldsymbol\theta}}(S_n)
= \sum_k(\theta_k - \hat\theta_k)\,A_k(S_n),
$$

and `observe` records them at sampling (D232), with the tail and the
estimate of the shift of a pair term at the volume of the frame, which
are linear in its constants as well. No coordinates are read again.

```python
nbfix = mdir.PairTerm()                      # the A-B pair in the coefficients of r^-12 and r^-6
nbfix.name, nbfix.groups, nbfix.expression = "nbfix", [":K", ":CL"], "c12/r^12 - c6/r^6"
nbfix.constants = [("c12", c12), ("c6", c6)]
nbfix.observe = ["c12", "c6"]                # two numbers a frame: sum r^-12 and -sum r^-6
system.pair_terms = [nbfix]
...
sim.reporters.append(mdir.ObservablesReporter("prod.obs", period=500))   # or State.observables
sim.run(5_000_000)
A = np.loadtxt("prod.obs")[:, [3, 4]] * 4.184       # nbfix.d_c12, nbfix.d_c6, kJ/mol per unit
delta = A @ (theta - theta_hat)                      # U(theta) - U(theta-hat) of each frame
w = np.exp(-(delta - delta.min()) / kT); w /= w.sum()
gradient = -(w @ (O[:, None] * A) - (w @ O) * (w @ A)) / kT   # d<O>/d(c12, c6)
```

(The file holds six decimals of a kcal/mol per unit of the constant, so
that the sum of $r^{-12}$ in nm, about $10^{9}$, is given to 16 digits and
smaller sums to fewer; `State.observables`, in a `CallbackReporter`, has
the values in full.)

Which potentials are linear: a Lennard-Jones pair term written in $C_{12}$
and $C_6$ (in $\sigma$ and $\epsilon$ it is linear in $\epsilon$ only); the
prefactor of any pair or tuple term; a harmonic term in its force
constant. A tabulated pair term is linear in the values of its table, but
a table is not a tunable of the Python model today, so that case is not
available. The charges are not linear ($U$ is quadratic in them).

**This does not replace stored frames.** A run kept only as such sums can
be reweighted in those tunables and no others: another term, another
functional form (the same pair term in $\sigma$, which is not linear), a
parameter that was not observed, or an observable that was not recorded
needs the coordinates. It is a shortcut for the fit that is planned when
the run is made, beside the trajectory, not in place of it.

Checked (`python-frames.test`, scenario `linear`): 60 A + 60 B with the
A-B correction as `c12/r^12 - c6/r^6` under a plain cutoff with the
correction for the dispersion, 150 frames, $\sigma'$ moved by 0.002 nm
($N_\text{eff}$ = 118):

| From the two sums of `observe` | Reference | CPU double | CPU mixed | GPU mixed | Tolerance (double, mixed) |
|---|---|---|---|---|---|
| $U_{\boldsymbol\theta} - U_{\hat{\boldsymbol\theta}}$ of each frame, kJ/mol (1.78 in the mean) | the evaluator of the dependent terms on the stored frames | 1.8e-13 | 2.1e-5 | 2.0e-5 | 1e-9, 2e-3 |
| The weights, in $1/K$ | the same | 4.5e-13 | 1.9e-5 | 2.0e-5 | |
| The reweighted average, relative | the same | 8.3e-16 | 3.8e-7 | 4.1e-7 | 1e-9, 1e-3 |
| Its derivative in $c_{12}$ and $c_6$, relative | the same, from the rows of the Jacobian | 1.6e-14 | 3.4e-6 | 3.7e-6 | 1e-9, 5e-3 |

### Maintainer rulings (dependent terms)

The six questions put on PR #266 were decided as recommended, and the mode
is taken.

| | Question | Ruling |
|---|---|---|
| T1 | Where the mode is chosen | `FrameEvaluator(program, terms="dependent")`; the evaluator compiles the second program |
| T2 | The name of the energy | `dependent_energy`; `energy` raises in that mode, and the reverse |
| T3 | The virial and the columns of terms left out | not given, and listed in `unavailable` |
| T4 | A branch or a second program | a second program |
| T5 | The record | a decision of its own, `D243` |
| T6 | Lennard-Jones tunables | all Lennard-Jones pairs within the cutoff, for now |

## Slices

| Slice | Contents |
|---|---|
| 1 (done) | `FrameEvaluator`, `evaluate` over arrays and iterables with one evaluation per frame, `energy`, `virial`, `volume`, `observables`, `depends`, the Jacobian kept or the second pass, `mdir.torch.evaluate`, `mdir.KB` |
| The dependent terms alone, and the example without frames (done, D243) | `terms="dependent"`, `dependent_energy`, `plan["terms"]`; the sums of `observe` for a potential linear in its tunables |
| 1, after D238 and D239 (done) | the tilts of each frame of a triclinic cell; the frames of `mdir.read_h5md` as input |
| 2 | the frames inside one activation; frames read from a device tensor without a copy to the host |
| 3 | the gradients `positions` and `strain` as outputs, with the comparison of the forces and the virial of PME with torch-pme (moved here from #203 by #138); the metatomic shape of #138 |
| Later | the derivative of the virial and of observed columns in $\boldsymbol\theta$ (M2b-3); the JAX adapter (M2b-2); learned terms (M3), whose parameters stay in the framework; frames over several nodes (M4) |

## Maintainer rulings

The twelve questions put on PR #250 were decided as follows
([comment](https://github.com/vv137/mdir/pull/250#issuecomment-6072592727)
and the two answers below it).

| | Question | Ruling |
|---|---|---|
| Q1 | Who owns the state that the frames are put in | `mdir.FrameEvaluator(program)` owns a second simulation of the program |
| Q2 | Inputs of the first slice | arrays and DLPack tensors `(K, N, 3)` in the order of the input, and iterables; no reader of DCD or XTC here |
| Q3 | The product | the Jacobian kept below `jacobian_bytes` (256 MiB), a second pass above |
| Q4 | The op of the adapter | `torch.library.custom_op` with `register_autograd`, behind `mdir.torch.evaluate`; the roadmap's line is corrected |
| Q5 | Outputs whose derivative in $\boldsymbol\theta$ is not implemented | returned, with a backward that raises unless detached; `depends` states the proofs |
| Q6 | The reference energies | an evaluation by the evaluator at $\hat{\boldsymbol\theta}$ of the stored frames |
| Q7 | A store of frames without loss | an H5MD reporter as its own item (#251); its reader gives the evaluator frames one at a time; this document warned about the precision of XTC; the warning is corrected by the measurements of #264 ([Which stored precision](#which-stored-precision-serves-which-reweighting)) |
| Q8 | Utilities of reweighting | none in MDIR; the constant `mdir.KB` |
| Q9 | Device and type of the adapter's outputs | float64 on the device of `theta`, copied from the host |
| Q10 | Triclinic cells | superseded: #206 was done first (D238); the evaluator takes the tilts of each frame |
| Q11 | Whether the program of the evaluator must be the sampler's | any program with tunables and `tunable_gradient` over the same particles |
| Q12 | The host memory of the frames | the caller keeps them; the evaluator reads them in turn |
