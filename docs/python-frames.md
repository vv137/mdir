# The frame evaluator (D240)

Issue #249, a step of M2b (#138, [roadmap](roadmap.md), Section 6.1).
Status: the first slice is implemented, as the maintainer ruled on PR #250
([Maintainer rulings](#maintainer-rulings)), and the frames of every call
are evaluated inside one activation of the evaluator's program
([D251](#the-frames-inside-one-activation-d251),
#291); what later slices add is in [Slices](#slices).

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
out = evaluator.evaluate(positions, cells=None, tilts=None, gradients=("tunables",))
out.energy                  # (K,) float64, kJ/mol: the potential that the forces sample (D210)
out.virial, out.volume      # (K,) float64, kJ/mol and nm^3
out.observables["lower.d_z0"]   # (K,) float64, the columns of `observe` (D232)
out.version                 # the value version of theta that the frames were evaluated at
out.depends["virial"]       # the tunables that an output may read; empty: none, by proof
out.jacobian["sigma"]       # (K, M) float64, or `out.jacobian is None` above the bound
g = out.vjp(cotangent)      # cotangent (K,) float64 on `energy`
g["sigma"]                  # (M,) float64, kJ/mol per unit of the tunable
g.zero                      # the names whose derivative is zero by proof (D230)
out.gradients["positions"]  # with gradients=(..., "positions"): (K, N, 3), dU/dx, kJ/mol/nm
out.gradients["strain"]     # with gradients=(..., "strain"): (K, 3, 3) float64, kJ/mol
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
The simulation runs a program of its own, built from the model of the one
given: a loop over frames
([below](#the-frames-inside-one-activation-d251)). The first
evaluator of a program compiles it (with its first frame 1.9 s for the
dipeptide in water and 2.7 s for JAC on a GPU); a later evaluator of the
same program takes the code that the first left with the `Program` (D236;
0.07 s).

`evaluate(positions, cells=None, tilts=None, *, gradients=("tunables",))`:

| Argument | Shape, type | Meaning |
|---|---|---|
| `positions` | `(K, N, 3)` or `(N, 3)`, float64 or float32; a NumPy array (a `numpy.memmap` too), or any object with `__dlpack__` | nm, **in the order of the input**, as `InitialState.positions` and `state().positions` are |
| `positions` | an iterable, one frame per item | each item an object with `positions` and `cell` (an `mdir.State`; a `Frame` of `mdir.read_h5md`), a pair `(positions, cell)`, or `(N, 3)` positions alone; `cells` and `tilts` must be `None` |
| `cells` | `None`, `(3,)`, or `(K, 3)` float64 | the diagonal $a_x, b_y, c_z$ of the cell of every frame, or of each, in nm (the edges of an orthorhombic cell); `None`: the cell of the program's initial state, with its tilts, whatever was evaluated before |
| `tilts` | `None`, `(3,)`, or `(K, 3)` float64 | the tilts $b_x, c_x, c_y$ of the cell of every frame, or of each, the arrays of `mdir.Cell` and of `Borrow.tilt` (D238); `None` with `cells`: an orthorhombic cell |
| `gradients` | a tuple of names among `"tunables"`, `"positions"`, `"strain"`; `("tunables",)` by default | the gradients of the energy that are asked (D252): `"tunables"`, the derivative in the tunables, kept as `jacobian` or asked for with `vjp`; `"positions"` and `"strain"`, the arrays `gradients["positions"]` and `gradients["strain"]` ([below](#the-gradients-in-the-positions-and-the-strain-d252)). `()` asks for none. Until D252 the derivative in the tunables was the argument `gradient=True`, which is gone |

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
| `gradients` | a read-only mapping of the names `"positions"` and `"strain"` that were asked to arrays `(K, N, 3)` and `(K, 3, 3)`; empty by default | $\partial U_n/\partial\mathbf x_i$ and $\partial U_n/\partial\boldsymbol\varepsilon$ ([below](#the-gradients-in-the-positions-and-the-strain-d252)) |

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
diagonal; a name of `gradients` that is not one of the three, or
`gradients` that is not a tuple of names. With `UnsupportedError`: positions or cells that require a gradient (a PyTorch tensor with
`requires_grad`), which the arrays of `evaluate` do not carry and which is
not zero (`mdir.torch.evaluate` takes them); `"positions"` or `"strain"`
from an evaluator of the dependent terms. A
frame whose evaluation fails (two particles at one place, a neighbor
structure that overflows) raises `SimulationError` naming the frame. After
any of these the evaluator is usable and gives what it gave before, to the
bit: a frame that is refused changes nothing, and a frame whose evaluation
fails ends the activation, so that nothing of it reaches another frame;
the next frame begins a new one, at the cost of one start. The
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
generator) is taken when the Jacobian is kept or without `"tunables"`; if
its rows pass the bound, `evaluate` raises `InputError` naming the first
frame beyond it, since the second pass would need the frames again.

`gradients` without `"tunables"` keeps no rows and refuses `vjp`. It does not make a pass
faster: the energy of D210 is that of `@tunable`, which is evaluated with
its derivative at every frame (measured: 1.047 ms against 1.050 ms per
frame on the dipeptide).

### The PyTorch adapter

```python
import mdir.torch                      # imports torch; `import mdir` does not
out = mdir.torch.evaluate(evaluator, theta, positions, cells=None, tilts=None, gradients=())
out.energy                             # (K,) float64 tensor with a grad_fn in theta
out.virial, out.volume, out.observables[name]     # (K,) float64 tensors
out.depends, out.version, out.count, out.zero     # as FrameEnergies
out.gradients                          # "positions" and "strain" where asked, as tensors
```

**What can be trained.** `energy` is differentiable once, in the tunables
and, where they are tensors that require a gradient, in the positions and
the cell ([below](#through-mdirtorchevaluate)). A loss on the forces or on
the stress can be evaluated, since both are outputs, but it cannot yet
train the tunables: its derivative in $\boldsymbol\theta$ is a second
derivative of $U$ (force matching, stage 4 of #138), and asking for it is
an error, `UnsupportedError`, never a zero.

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
  The gradient in the positions, where it is asked or a backward needs
  it, is the one array of $N$ rows a frame that crosses the adapter: it is
  copied from the host to the device of `theta` as the others are. No
  device buffer is leased (D220, D229) and no stream is handed over;
  positions given as a tensor of a device are copied to the host.
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
  `.detach()`. In a call whose positions or cell take a gradient every
  such output is a function of them as well, and has that `grad_fn`
  whatever `depends` says.
- `positions`, `cells`, and `tilts` that require a gradient are taken
  where each is one tensor, and `energy` is differentiable in them
  (D252, [below](#through-mdirtorchevaluate)); frames of an
  iterable carry no graph, and such frames that require a gradient are
  refused (`UnsupportedError`).
- **No double backward.** Every product of the backward is an operation
  whose own backward raises `UnsupportedError`, naming the derivative that
  was asked: a second differentiation is an error, not a silent zero
  (until D252 it was PyTorch's `RuntimeError` for an
  operation without a registered derivative). A Hessian in
  $\boldsymbol\theta$ is out of scope.
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

- **The order of the input, at every boundary and inside.** Frames come
  in the order of the input, which is the order of trajectory files, of
  `state()`, and of other programs, and the evaluator's program keeps the
  particles in that order, whatever `Execution.reorder` of the program
  given says (D251; until then the spatial order of D44
  was chosen anew at each frame). An order chosen at a frame would renumber
  every tuple set on the host and upload it again with its incidence
  structure at that frame, which is most of what the start of an
  activation costs, and it buys nothing: the neighbor structures bin the
  particles themselves when they are built, and the kernels of a frame
  take the same time in either order
  ([measured](#the-order-of-the-particles)). The outputs are numbers per
  frame and per entry of a tunable, which have no order of particles, and
  the gradient in the positions (D252), which is in the
  order of the input with nothing to gather, so that the stale gather of
  #137 (item 1) cannot happen in a script.
- **The energy of a frame is a function of the frame alone**, of the
  program, and of the values of the tunables: the same bits in any call,
  after any frames, after a frame that failed, and whatever other
  simulations do in between, in the deterministic mode
  (`python-frames-activation.test`). Against a sampler the sums are taken
  in the same order when the sampler keeps the order of the input
  (`Execution.reorder = False`), and the two agree to the bit; against a
  sampler in the spatial order, the default, they agree within the
  rounding of sums in another order: $1.7\times10^{-10}$ kJ/mol
  ($1.5\times10^{-14}$ relative) on systems of 1,168 to 1,209 atoms in
  double precision, and in mixed precision $1.8\times10^{-4}$ to
  $3.2\times10^{-4}$ kJ/mol on them ($3\times10^{-8}$ relative; on the
  CPU the orthorhombic sets agree to the bit and only the triclinic one
  differs) and $1.2\times10^{-4}$ kJ/mol of 307,567 on JAC
  ([Validation](#validation-of-the-frames-inside-one-activation)). A
  reweighting takes both of its energies from the evaluator
  ([below](#reference-energies-and-what-is-stored)) and is not affected; a
  comparison with energies that a sampler in the spatial order recorded
  is, at that size.
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
- **Neighbor structures** are built anew at every frame, in storage that
  is allocated once; no list is carried from one frame to the next, though
  the frames of a call share an activation. Keeping a structure would need
  frames within half a skin of each other, which stored frames are not,
  and would make the bits of a frame depend on the frames before it; the
  build is 0.17 ms of the 0.69 ms of a frame of JAC. The tests hold that
  the frames give the same numbers to the bit in another order, one at a
  time, and after repeats of a frame and jumps between distant ones.
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
| $U_n$ in the positions and the strain | since D252: `gradients["positions"]` and `gradients["strain"]`, the forces and the virial tensor of the forward pass with the opposite sign, and through `mdir.torch.evaluate` the backward in the positions and the cell | |
| Second derivatives, the derivative of the forces or of the stress in $\boldsymbol\theta$ among them | refused, with `UnsupportedError` | force matching: stage 4 of #138 |

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
\hat{\boldsymbol\theta} + \tfrac12k_BT/\mathrm{std}(\partial
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
  `checkCommittedCell` of D238), writes the positions into the buffer of
  the host that the entry takes at its boundary and the six numbers of
  the cell (`setCell` for what the host adds at the volume of the frame),
  and resumes the activation, which evaluates the frame and waits at the
  boundary again ([below](#the-frames-inside-one-activation-d251));
  the first frame, and the first after a failure or an update of the
  tunables, begin an activation. Until D251 every frame
  began one: a borrow, a commit, and `gradient()` take two and copy the
  state of the device to the host in between. If the evaluation fails, the
  activation is ended, the cell of before is restored, and the simulation
  has not failed. `collectTunableGradient` is the chain rule of D230,
  which `gradient()` shares.
- `Simulation._evaluate_frames` (Python, private) runs a batch of frames
  with the GIL released and returns the columns and the rows; an error
  carries the index of its frame.
- `python/mdir/_frames.py` holds `FrameEvaluator`, `FrameEnergies`,
  `FrameGradient`, the reading of the frames in batches, and `depends`;
  `python/mdir/torch.py` the operations. Both are files of the package in
  the build tree and in the wheel.
- The second pass applies the chain rule at each frame and adds the rows on
  the host; accumulating the fields on the device and applying it once
  ([python-gradient.md](python-gradient.md), "At $K$ frames") was not
  taken with the frames inside one activation: a field reaches the host in
  one small copy a frame, and the chain rule is not what a frame costs.
- No kernel and no runtime function of a device changes, and nothing on the
  path of a step: the text of every other program is what it was, to the
  byte.

### The frames inside one activation (D251)

Issue #291. Until this slice every frame was the start of an activation
(D215): the buffers of the whole model copied to the device, the tuples
renumbered and their incidence structures built on the host, then the
kernels. Of a frame of JAC on one RTX 4090 in mixed precision, 5.0 ms,
the kernels with the neighbor build are 0.8 ms.

- **A program of frames.** `Control::frames`, which the simulation of an
  evaluator sets as `dependentTerms` is set (no key of a control file),
  makes the entry of the program of segments a loop over frames in place
  of the start and the loop over parts. Everything but the positions and
  the cell is taken from the buffers of the host once, before the loop:
  the masses, the parameters, the tables, the tuples with their incidence
  structures, the identities.
- **The boundary.** Each iteration begins with `@mdrtFrameBoundary`, where
  the entry waits on its own stack as the entry of a run waits at
  `@mdrtPartBoundary`. The call returns the buffer of the host that holds
  the positions of the next frame, in the type of the state, and writes
  the six numbers of its cell. The buffer is a value of the iteration
  (`mdrt.from_buffer` of the result of a call), so that no pass can take
  the positions of one frame for those of another; in a loop,
  `md-exec-assign-storage` gives such a field a buffer of the loop and a
  copy, since a buffer that a loop owns may hold what it carries to its
  next iteration.
- **The iteration** builds the cell (triclinic included, D238, and the
  scale of the reference of restraints), places the virtual sites, and
  runs the text of the evaluation at the start of a run, unchanged, in a
  loop of one iteration with `mdrt.segment`: the neighbor structures are
  built anew (D218), the energy, the forces, and the virial are evaluated,
  and `@tunable` with the derivative. The results reach the host through
  the calls that a start makes (`mdrtWriteEnergies`, `mdrtWriteVirial`,
  `mdrtWriteObservables`, `mdrtWriteTunableGradient`,
  `mdrtWriteTunableGradientField`), and the host applies the chain rule
  at each frame as before.
- **Memory in $K$.** Nothing on the device or on the stack of the entry
  grows with the frames: the buffers of the loop are loop-carried, the
  storage of the neighbor structures is the function's, and the rows that
  the evaluation writes on the stack, which stand in the loop where the
  start of a run takes them, are kept at one depth by the lowerings (below,
  #309).
  Measured: 300,000 frames in one activation on the CPU with the resident
  memory constant to 0.1 MB, and 6,000 on a GPU with the memory of the
  device constant. There is no limit on $K$; the host keeps $8K$ bytes per
  output and the Jacobian up to its bound, as before.
- **What the assembly of the loop assumes is checked** (#302, #309). The
  builder makes the loop from the text that it emitted for the start of a
  run, in which a call becomes `func.call` and nothing else changes. Text
  that this assembly cannot take would still compile and give the right
  numbers, so the builder refuses, with an error that names the line, a
  program of frames whose loop holds one of the arguments of the entry
  that a frame replaces (`%positions`, `%lx`, `%ly`, `%lz`, the tilts, and
  `%box_memory`, which is filled from them: they are of the program, not
  of the frame), or whose start evaluates nothing. A call in the generic
  form, a `return`, and a name of the loop taken twice are refused by the
  parser.
- **The stack of the entry is kept by the lowerings** (#305, #309). The
  entry never returns, so memory of the stack that a frame took and did
  not give back would be taken again at every frame. The rows of the
  evaluation (`%terms`, the observed columns, the derivative in the
  tunables) are allocations in the loop, as they are in the loops over
  steps of a run. On the CPU `hoist-static-allocas` takes every allocation
  of a constant size once, where its function begins, when the lowering
  ends (D117); on a GPU the lowering gives back the stack of each
  iteration of a loop of the host, whatever its size. Until #309 the
  builder moved the lines `%name = memref.alloca() : type` before the loop
  itself and refused any other spelling of an allocation in it; that
  matching of text is gone, since the lowerings do the same for every
  spelling and for what a pass adds. What the pass cannot take once is an
  allocation whose size is not a constant. No program has one: the
  builder, the templates, and the passes allocate static shapes alone. So
  that this is a check and not a reading, the lowering of a program of
  frames for the CPU runs the pass with `refuse-rest`: a module in which
  an allocation would be left outside the first block of a function (and
  outside OpenMP, where each thread has its own) does not compile, and a
  program of frames that compiles has none
  (`test/Conversion/MDExecToLoops/hoist-allocas.mlir`). A user would see
  it where the evaluator is made: `mdir.FrameEvaluator(program)` raises
  `mdir.CompileError`, "cannot lower program: ... 'llvm.alloca' op takes
  memory of the stack of a size that is not a constant outside the first
  block of its function ... this is a defect of mdir, not of the input",
  with the line of the op in the emitted module. It is a defect to report,
  not something a model or a frame can cause or mend.
  `python-frames-loop.test` and `python-frames-loop-gpu.test` read the
  emitted module of four evaluators (the rows in the loop, each of a
  constant size; no call in the generic form; no argument of the entry
  that a frame replaces) and measure the memory of a frame: the entry
  runs on a stack of its own, so the address of the frame of its call of
  `@mdrtFrameBoundary` is the depth of that stack where it waits, and the
  depth after frame 201 is that after frame 1 to the byte, whatever put an
  allocation in the loop, the builder or a pass; the bytes in use of the
  allocator of the host, on the CPU and on a GPU, grow by 64 at most from
  frame 101 to frame 201 (measured: +0; see the next item). The depth says
  nothing of memory that a frame takes and gives back, and the test covers
  the programs it compiles: the dipeptide in water with the whole
  potential and with the dependent terms.
  The two handles of these tests,
  `Simulation._ir` (the emitted module) and `Simulation._stack_in_use`
  (the depth), are private: they are not part of the interface and may
  change or go.
- **The memory of the host does not grow with the frames or the calls**
  (#306). What a call returns is the caller's: $8K$ bytes per output and
  the rows of the Jacobian, freed with the result. Besides it the host
  keeps nothing for a frame or for a call. Measured with
  `scripts/validation/frames/heap.py`: one evaluator of the dipeptide, the
  bytes in use of the allocator (`mallinfo2`: `uordblks + hblkhd`, the
  thread cache off) read after 1, 2, 4, ... calls, where a vector that
  doubles would read the same number of bytes a frame at every reading.
  They read +0 between every two readings on the CPU for calls of 1, 25,
  50, and 256 frames, with the Jacobian kept, not returned
  (`jacobian_bytes=0`), and not asked (`gradients=()`), for the whole
  potential and the dependent terms, in double and mixed precision (48
  cases to 4,096 frames or more, and 8 to 102,400 or 131,072), and on a
  GPU, where they are the bytes of the host all the same, to 131,072
  frames. The reading sees everything that the host takes for a frame,
  since all of it comes from the allocator of the C library: the C++ of
  the driver and of the binding, the runtime (`malloc`, `calloc`,
  `realloc`), and the buffers of the compiled code, whose `malloc` and
  `free` are functions of the host. What is mapped otherwise is taken
  once: the stack of an activation and the code of an engine, and the
  address space mapped outside the heap (`/proc/self/maps`) is the same
  number at each of 18 readings to 131,072 frames on the CPU, for each of
  the three states of the Jacobian. Outside both are the memory of the
  device and the page-locked memory of the host that the driver gives
  (`python-memory-gpu.test` reads the device). `mallinfo2` does not see
  the small objects of Python either: the blocks
  of the interpreter (`sys.getallocatedblocks`) rise by 70 to 180 over
  the first 512 calls, while its free lists fill, and by 4 at most from
  there to call 4,096, and by 1 at most from there to call 131,072. The movement that #306 reports, some 600 bytes in
  one step and 16 up and down, was the reader of the test: it made the
  `ctypes` class of
  the structure and the handle of the C library anew at every reading,
  and those are blocks in use that differ from one reading to the next
  (+576 once, at the fourth handle of a process that evaluates nothing).
  The tests now read with a structure and a handle made once, and their
  bound is 64 bytes over 100 frames given as 3 calls of 25 frames and 25
  calls of one frame, which lets 0.64 bytes a frame and 2.3 a call pass;
  a vector of the host that takes 8 bytes a frame reads +1024 there,
  and one that takes 8 bytes a call +256.
- **What is not checked in the text.** A row that is taken once, where
  the function begins, holds at a frame what the frame before wrote (on
  the CPU; on a GPU it holds anything): the evaluation
  must write every slot before the call that takes the row, as it does
  today. No check of the text sees an evaluation that reads a slot first;
  this rests on the comparison, to the bit, of a frame with the same frame
  after other frames (`python-frames-activation*.test`), so the check of
  the builder is not the whole of what the loop assumes. Text that should
  stand before the loop and is in it is evaluated at every frame, with the
  right numbers: only a timing sees that.
- **The activation lives** across the batches of a call and across calls.
  It ends at an update of the tunables (their values are arguments of the
  entry), at a frame whose evaluation fails, and with the evaluator. The
  state of the host stays that of the program; only its cell is the
  frame's. The simulation of an evaluator takes no steps, no evaluation of
  its state, no view, and no borrow.
- **The program of the dependent terms** (D243) is kept with the program
  it is made from, so that a second evaluator with `terms="dependent"`
  compiles nothing: 1.62 s for the first on the dipeptide and 0.05 s for
  the second, where each took 1.7 s.

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
tunable. The frames of a Python simulation at constant pressure in that
cell, which D244 allows
([python-segments.md](python-segments.md#npt-in-a-triclinic-cell-d244)),
give the same zeros in the first two rows of the table
(`python-triclinic-npt-frames*.test`).

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

### Validation of the frames inside one activation

**Against the evaluator of before** (`main` at 7be2e19, one start of an
activation per frame), on the same stored frames:
`scripts/validation/frames/activation.py`. Four sets, 26 frames: the
dipeptide in water with the whole potential (25 tied charges, $\sigma$
and $\epsilon$ of OW-OW, the constant of a soft pair term; an observed
pair term and an observed wall) and with the dependent terms; 6 frames
under a barostat with leapfrog, each with its cell (20.442 to 20.506
nm³); 6 frames at constant pressure in a rhombic dodecahedron, each with
its diagonal and its tilts (12.478 to 12.729 nm³, the tilts moved by
$8.7\times10^{-3}$ nm). Each set is evaluated in one call, one frame per
call ($K = 1$), in the reverse order, in the order 0, 0, 0, last, 0, last,
last, 1, and again after a call in which a frame fails. Quantities: the
energy (up to $1.2\times10^4$ kJ/mol), the virial, the volume, the
observed columns, the rows of the Jacobian, and a product with a random
cotangent; the largest difference over all sets and ways, absolute for
the energy in kJ/mol and relative for the rows.

| Reference | CPU double | GPU double | CPU mixed | GPU mixed | Tolerance (double, mixed) |
|---|---|---|---|---|---|
| The evaluator of before on programs compiled in the order of the input (`Execution.reorder = False`), the deterministic mode: every quantity | 0 | 0 | 0 | 0 | 0 |
| The same, without the deterministic mode, mixed precision | | | 0 | energy 1.5e-11; virial 1.3e-9 and rows 4.7e-9 relative | the spread of that mode: the evaluator of before against itself, virial 1.1e-9, rows 2.9e-9 |
| The evaluator of before on programs in the spatial order, the deterministic mode: the energy; the rows | 1.7e-10; 1.6e-15 | 2.9e-11; 1.4e-15 | 1.8e-4; 2.2e-8 | 2.9e-4; 3.4e-8 | 1e-9, 1e-12; 5e-3, 1e-5 |
| The same, without the deterministic mode, mixed precision | | | 1.8e-4; 2.2e-8 | 3.2e-4; 4.2e-8 | 5e-3, 1e-5 |
| Every way of evaluating a set against its evaluation in one call, the deterministic mode | 0 | 0 | 0 | 0 | 0 |

The volume is equal to the bit in every row. Without the deterministic
mode on a GPU the sums of the forces are not taken in one order, on `main`
either, and the two evaluators differ by what one differs from itself.

**In the suite**, `python-frames-activation.test` and
`python-frames-activation-gpu.test` (the dipeptide in water, 6 frames; the
deterministic mode):

| Quantity | Reference | CPU double | GPU double | CPU mixed | GPU mixed | Tolerance (double, mixed) |
|---|---|---|---|---|---|---|
| The energy, the virial, and the rows of each frame | `gradient()` and `state()` of a sampler in the order of the input at the step of the frame | 0 | 0 | 0 | 0 | 0 |
| The energy (kJ/mol) and the rows of the first and the last frame | `gradient()` of a simulation in the spatial order compiled from the frame | 2.9e-11, 0 | 1.5e-11, 5.3e-16 | 0, 0 | 1.4e-4, 1.1e-8 | 1e-9, 1e-12; 5e-3, 1e-5 |
| One frame per call; the order 0, 0, 0, last, 0, last, last, 1; batches of two frames; 120 frames in one call: every output | the first evaluation | 0 | 0 | 0 | 0 | 0 |
| With the calls of a second evaluator (the dependent terms) and the steps of a sampler between the calls | the first evaluation | 0 | 0 | 0 | 0 | 0 |
| After a frame that fails in the middle of a call, one that fails as the first of an activation, and frames that are refused | the first evaluation | 0 | 0 | 0 | 0 | 0 |
| At other values of a tunable and at the first values again | the first evaluation | 0 | 0 | 0 | 0 | 0 |

The tests of the tables above (`python-frames*.test`,
`python-frames-h5md*.test`, `python-frames-torch*.test`,
`python-triclinic-npt-frames*.test`) keep every zero: their samplers are
compiled in the order of the input (`execution_` of
`test/Driver/Inputs/python_frames.py`), as the program of an evaluator is.
With the spatial order chosen at every frame, before the order of the
input was taken, the same tests passed with their samplers in the spatial
order, and the CPU in double precision gave every bit of the evaluator of
before.

No kernel and no function of the device runtime changed. The scenario
`activation` on a GPU in mixed precision ran under `compute-sanitizer
--tool memcheck` with no error (it held the lock of its GPU for 32 s).

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

### The frames inside one activation

One RTX 4090 (GPU 0 under its lock, nothing else on it), mixed precision,
not the deterministic mode, PME, a cutoff of 0.8 nm, flexible, 0.5 fs; the
constant $l$ of a soft pair term tunable; frames 40 steps apart kept in
float64, the same stored frames for both builds;
`scripts/benchmarks/frames/bench.py`. ms per frame of `evaluate` as a
function of the frames $K$ of a call, the least of 5 repeats after three
passes that are not timed. "Before" is `main` at 7be2e19, one start of an
activation per frame. These are not the rates of the RTX 3090 of the
tables above.

| System | Terms | | $K$ = 1 | 4 | 16 | 64 | 256 | Frames/s | In steps |
|---|---|---|---|---|---|---|---|---|---|
| Dipeptide, 1,168 atoms (a step 0.052 ms) | whole potential | before | 0.885 | 0.853 | 0.838 | 0.834 | 0.843 | 1,186 | 16.3 |
| | | now | 0.322 | 0.297 | 0.285 | 0.283 | 0.284 | 3,527 | 5.5 |
| | dependent | before | 0.482 | 0.447 | 0.437 | 0.434 | 0.420 | 2,380 | 8.1 |
| | | now | 0.194 | 0.168 | 0.160 | 0.160 | 0.158 | 6,316 | 3.1 |
| JAC, 23,558 atoms (a step 0.122 ms) | whole potential | before | 5.06 | 5.04 | 5.00 | 4.99 | 5.00 | 200 | 40.9 |
| | | now | 0.721 | 0.692 | 0.684 | 0.686 | 0.686 | 1,457 | 5.6 |
| | dependent | before | 2.11 | 2.09 | 2.07 | 2.07 | 2.08 | 482 | 17.0 |
| | | now | 0.476 | 0.459 | 0.444 | 0.450 | 0.447 | 2,239 | 3.7 |

3.0 and 2.7 times on the dipeptide, 7.3 and 4.6 times on JAC. A call of one
frame pays about 0.03 ms more than a frame of a long call, the work of a
call in Python. For #140's system (21,403 atoms, 38,000 frames) a pass is
about half a minute where it was three. With other tunables on JAC, at $K$
= 256: $\sigma$ and $\epsilon$ of OW-OW, 0.918 ms (whole) and 0.725 ms
(dependent); the charges of the water, 0.970 ms and 0.866 ms.

**What a frame is made of now.** Under `MDRT_PROFILE`, per frame, from the
difference of runs of 100 and 400 frames; a copy to the host waits for the
kernels before it, so the kernels' time is in the copies and the waits:

| | JAC, whole | JAC, dependent | Dipeptide, whole | Dipeptide, dependent |
|---|---|---|---|---|
| Before: copies between host and device | 50 | 18 | | |
| Launches | 65, 0.155 ms | 26, 0.061 ms | 53, 0.092 ms | 26, 0.051 ms |
| Copies (the positions to the device, 565 KB on JAC; the sums and the fields to the host) | 13, 0.419 ms | 7, 0.392 ms | 13, 0.118 ms | 7, 0.066 ms |
| Waits | 13, 0.040 ms | 7, 0.059 ms | 13, 0.012 ms | 7, 0.014 ms |

With `MDRT_WAIT`, which waits for every launch, the kernels of a frame of
JAC with the whole potential sum to 0.77 ms (the build of the neighbor
matrix 0.17 ms, the three pair kernels 0.15 ms), more than the 0.69 ms of
a frame without it: a frame costs what its kernels cost. What is outside
them is the copy of the positions (0.03 ms on JAC), the checks of a frame
and the chain rule on the host, and the call in Python, about 0.07 ms a
frame on JAC.

#### The order of the particles

The loop was first written with the spatial order chosen at every frame,
as a start chooses it. One RTX 4090, mixed precision, the whole potential,
ms per frame; (A) the spatial order at every frame, (B) the order of the
input, which was taken, (D) the spatial order of the first frame of an
activation kept for its frames:

| System | Atoms | Neighbor structure | Before | (A) | (B) | (D) |
|---|---|---|---|---|---|---|
| JAC | 23,558 | matrix | 5.00 | 4.41 | 0.688 | 0.693 |
| JAC | 23,558 | groups of 16 | | 3.97 | 0.704 | 0.758 |
| Factor IX | 90,906 | matrix | | 55.6 | 1.947 | 1.959 |
| Cellulose | 408,609 | matrix | | | 8.08 to 9.42 | 8.24 to 8.93 |

In (A) every tuple set is renumbered on the host and uploaded with its
incidence structure at every frame: per frame of JAC, 5 reads of the
identities and uploads of 1.6, 7.2, 21.6, 4.2, and 8.0 MB, and a cost that
grows faster than the system (#294). The kernels do not see the order:
with `MDRT_WAIT` those of a frame of Factor IX sum to 1.849 ms in (B) and
1.868 ms in (D), the build of the neighbor matrix 0.599 and 0.597 ms, the
pair kernels the same to 0.002 ms; on JAC 0.769 ms in (B) and 0.824 ms in
(A), whose two ordering kernels cost more than its pair kernels gain. And
the order of the input of these systems is a bad one already: consecutive
water oxygens are 2.97 nm apart in the mean in JAC and 4.99 nm in Factor
IX, as far as after a random shuffle of the waters (2.97 and 5.01 nm);
Cellulose is local (1.4 nm against 8.5 nm). Renumbering the tuples on the
device, which would let a frame be ordered at no such cost, is #295. (The
cost that grew faster than the system was found in #294 to be the memory
of the structures, mapped and faulted in anew at every build once a
structure is over 32 MiB; since D253 a build
takes the memory of the last, and a start in the order of the last start
builds nothing.)

Two further items that the roadmap named for this slice. *The tuples of
the bonded terms that a program of the dependent terms still builds*: they
are built once for an activation now and are in no frame. *A relation over
the particles of the types of a tunable Lennard-Jones pair*: not taken. Of
a frame of JAC with $\sigma$ and $\epsilon$ of OW-OW tunable and the
dependent terms, 0.725 ms, the kernels are 0.48 ms, of which the build of
the neighbor matrix is 0.17 ms and the three pair kernels 0.15 ms; a
structure over the 7,023 oxygens alone could take at most about 0.25 ms of
the 0.725 ms, for one kind of tunable, against a new relation in the
compiler.

**A run is not slower.** The text of the program of a run is the same to
the byte (`mdir emit` of JAC of the Amber suite, both builds). JAC of the
suite with the settings of `scripts/benchmarks/amber/bench.py`, two runs
each: NVE 1302.9 and 1305.7 ns/day before, 1305.8 and 1312.7 now (0.1325
and 0.1320 ms a step); NPT 1183.4 and 1173.5 before, 1186.7 and 1176.5
now.

## The dependent terms alone (D243)

Issue #264. Status: implemented, as the maintainer ruled on PR #266
([Maintainer rulings (dependent terms)](#maintainer-rulings-dependent-terms)).
**The mode saves time, 2 to 2.5 times a frame for a tunable of a pair term
and 1.45 to 1.9 times for the charges; it does not change the accuracy of a
difference in mixed precision** ([measured](#what-the-mode-saves)).
Those ratios are of frames that each began an activation; with the frames
inside one activation (D251) both modes cost less and the
mode saves 1.5 to 1.8 times for a tunable of a pair term and 1.1 to 1.3
times for the charges and for $\sigma$ and $\epsilon$
([measured](#the-frames-inside-one-activation)).

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

A frame of the dependent terms was still the start of an activation: 7 to
11 steps on the dipeptide and 11 to 16 on JAC.

**The next slice, and its bound** (done since:
[the frames inside one activation](#the-frames-inside-one-activation), 3.1
steps on the dipeptide and 3.7 on JAC). What a frame cost beyond its
kernels was fixed by the start of an activation, in either mode. Under
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
pairs (T6). None of the three was implemented in D243; D251
took the first, with which the second went, and measured the third.

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

## The gradients in the positions and the strain (D252)

Issue #313, slice 3. Status: implemented, as the maintainer ruled on PR
#314 ([Maintainer rulings (gradients)](#maintainer-rulings-gradients)).

The program of an evaluator evaluates the forces and the virial tensor of
every frame on the way to its energy (`md.evaluate @energy ... request
[energy, forces, virial]`), and until this slice gave the host the energy
and the trace of the virial alone. A caller who asks now takes the two
first derivatives of the energy of a frame that those hold, named as #138
names them after metatomic's standard outputs: the gradient in the
positions and the gradient in the strain.

### What is returned

For frame $n$ with positions $\mathbf x_i$ and cell vectors $\mathbf a,
\mathbf b, \mathbf c$ (the rows of $\mathsf h$), take the homogeneous
deformation

$$
\mathbf x_i \to (\mathsf 1 + \boldsymbol\varepsilon)\,\mathbf x_i,\qquad
\mathsf h \to \mathsf h\,(\mathsf 1 + \boldsymbol\varepsilon)^{\mathsf T}
$$

with any $3\times3$ matrix $\boldsymbol\varepsilon$, not only a symmetric
one. The two outputs are

$$
\mathbf G_i = \frac{\partial U}{\partial\mathbf x_i}\bigg\rvert_{\mathsf h}
= -\mathbf F_i,\qquad
\mathsf S_{\alpha\beta} =
\frac{\partial U}{\partial\varepsilon_{\alpha\beta}}\bigg\rvert_{\boldsymbol\varepsilon = 0}
= -\mathsf W_{\beta\alpha},
$$

of the energy $U$ that `energy` returns (D210), with everything that the
host adds at the volume of the frame. The paper derives both, and why the
second is the quantity of a pressure (Section 3.6).

```python
out = evaluator.evaluate(positions, cells, tilts, gradients=("tunables", "positions", "strain"))
out.jacobian, out.vjp(g)    # "tunables": as before
out.gradients["positions"]  # (K, N, 3), kJ/mol/nm, in the order of the input: dU/dx = -F
out.gradients["strain"]     # (K, 3, 3) float64, kJ/mol: dU/d(epsilon), row alpha, column beta
```

| | |
|---|---|
| The argument | `gradients=("tunables",)`, keyword only: a tuple of names among `"tunables"`, `"positions"`, and `"strain"`, each once, in any order. Anything else is an `InputError`: another name, a string, a `bool`. It replaces the argument `gradient` of slice 1, which is removed without an alias: `gradients=("tunables",)`, the default, is what `gradient=True` was, and `gradients=()` what `gradient=False` was |
| **The default** | **an evaluator that is not given `gradients` returns what it returned before this slice: the derivative in the tunables (`jacobian`, or `vjp` by a second pass), and no gradient in the positions or the strain; `out.gradients` is empty** |
| `"tunables"` | `out.jacobian` and `out.vjp`, unchanged ([above](#the-product-and-its-memory)); it is not an entry of `out.gradients` |
| `out.gradients` | a read-only mapping of those of `"positions"` and `"strain"` that were asked, in the order asked, to read-only arrays on the host |
| `"positions"` | `(K, N, 3)`, kJ/mol/nm, the rows in the order of the input, as the positions of the frame are (the program of an evaluator keeps that order, D251: nothing is gathered). In the type in which the program keeps its forces, with no conversion: float64 in `double`, float32 in `mixed`, and float64 in `mixed` with the deterministic mode, whose forces are kept in f64 (#102). There is no `dtype` argument: a caller who wants another type converts, and an f32 force widened to f64 carries no more |
| `"strain"` | `(K, 3, 3)` float64 in every mode, kJ/mol ($\boldsymbol\varepsilon$ has no unit), all nine components, not symmetrized, not divided by the volume: a script forms the stress $\mathsf S/V$ with `volume` |
| Memory | $24KN$ bytes (12 in f32) for `"positions"`, which the evaluator allocates and the caller owns: 565 KB a frame of JAC in f64. It is the one output that grows with $N$; a caller bounds it by the frames of a call. $72K$ bytes for `"strain"` |
| When not asked | nothing is evaluated, copied, or allocated for them: the request reaches the program with the frame, at its boundary, and the forces and the virial that are handed over are evaluated in a branch that a frame without the request does not take ([measured](#the-cost-of-the-gradients)). Calls that ask and calls that do not share an activation, and the other outputs of a frame are the same to the bit whatever is asked |
| When asked | one more evaluation of the forces of the frame, a third of a frame on the CPU ([measured](#the-cost-of-the-gradients)); `"positions"` and `"strain"` together cost what `"strain"` costs |
| `terms="dependent"` | `"positions"` and `"strain"` are refused (`UnsupportedError`), and `"gradients"` is listed in `unavailable`: the gradient of $U_\text{dep}$ is not that of $U$, as its virial is not the virial (T3). `"tunables"` is taken as before, and `out.gradients` is there and empty: it is the request that raises, not the attribute |

**Sign, and the relation to `virial`.** $\mathsf W$ is the virial tensor of
MDIR, $\sum \mathbf d\otimes\mathbf K$ over the interactions, in the sign
in which the pressure is $(2K + \mathrm{tr}\,\mathsf W)/3V$; what the
host adds at the volume of a frame (the correction for the dispersion with
the estimate of the shift, the background of a net charge), all $\propto
1/V$, enters as a third of its virial on each axis.

**With constraints, `virial` and $-\mathrm{tr}\,\mathsf S$ differ, and
a pressure takes `virial`.** `virial` is the trace that the pressure of a
run takes, which in a program with constraints has the term of the
constraint forces besides:

$$
\texttt{virial} = -\mathrm{tr}\,\mathsf S + W_c ,
$$

$W_c = 0$ in a program without constraints, where the two are equal to
the rounding of a sum (measured: 0 in double precision and at most
$4\times10^{-12}$ kJ/mol in mixed). $\mathsf S$ is a derivative of $U$
and has no such term: with rigid water the dipeptide at its first frame
has `virial` $= -7{,}075$ kJ/mol and $-\mathrm{tr}\,\mathsf S =
30{,}949$, $W_c = -38{,}024$. A script that forms the pressure of a
constrained model takes `virial`, with the kinetic term of its ensemble;
one that differentiates $U$ in the cell takes $\mathsf S$.

**Symmetry.** $\mathsf S$ is symmetric where the potential does not change
under a rotation of the positions and the cell together: every term of
pairs, of tuples, and of the reciprocal sum (measured on the dipeptide:
$\mathsf S - \mathsf S^{\mathsf T}$ is $2\times10^{-13}$ of 14,001 kJ/mol
in double precision and $1.3\times10^{-4}$ in mixed). A term of the
absolute positions (`mdir.ExternalTerm`) is not, and neither is $\mathsf
S$ then; it is returned as it is.

**Rectangular and triclinic cells.** The definition is the same for both,
and an orthorhombic program returns the nine components as a triclinic
one does. A program takes its cells reduced and lower triangular, and
$(\mathsf 1 + \boldsymbol\varepsilon)$ of such a cell is one only for an
$\boldsymbol\varepsilon$ on and above the diagonal ($\mathbf a$ stays
along $x$ and $\mathbf b$ in the plane $xy$). The components of $\mathsf
S$ below the diagonal are derivatives of the potential as a function of
the positions and of three free cell vectors, which the pair sums and the
reciprocal sum are.

**Restraints to reference positions: `"strain"` is refused.** The virial
of a restraint is taken about its reference (D124), $\sum\mathbf
F\otimes(\mathbf x - \mathbf x^\text{ref})$: the derivative of its energy
when the references deform with the cell, which is what a pressure
wants and not what a frame does, whose references stay. The virial
tensor of such a program is therefore not the derivative of the energy
of a frame in the strain (measured on the dipeptide with its heavy atoms
restrained at 500 kJ/mol/nm²: the diagonal differs from central
differences of the energy by up to 51 of 51,680 kJ/mol), and
`gradients=("strain",)` raises `UnsupportedError` for a program with
restraints, as do cells that require a gradient in the adapter;
`"positions"` is given and checked. `mdir.ExternalTerm` is not concerned:
its virial is the derivative.

**Constraints and virtual sites.** The positions are taken as given
(above), and $U$ is a function of all of them: $\mathbf G$ is the plain
gradient, not projected onto the constraints. A script that wants the
gradient along the constrained manifold projects it. A virtual site is
placed from its atoms whatever the frame says, so $U$ does not depend on
the row of the frame that holds it: the row of a site in $\mathbf G$ is
zero, and the rows of its atoms hold what the site passes to them (the
spread of its force), with the virial of the spread in $\mathsf S$. In
mixed precision the row of a site holds what the spread leaves of an f32
force, an ulp of it ($1.2\times10^{-4}$ kJ/mol/nm where the largest entry
is 3,876): the numbers are those of the program, and are not set to zero
on the host.

### Through `mdir.torch.evaluate`

```python
x = torch.tensor(frames, requires_grad=True)              # (K, N, 3)
out = mdir.torch.evaluate(evaluator, theta, x, cells, tilts, gradients=("strain",))
(g,) = torch.autograd.grad(out.energy.sum(), x)           # (K, N, 3), dU_n/dx
out.gradients["strain"]                                   # (K, 3, 3) tensor, as below
```

- **`gradients=()`** takes the names of the evaluator. `"positions"` and
  `"strain"` give `out.gradients[name]` as tensors on the device of
  `theta`. `"tunables"` is taken and changes nothing: the derivative in
  `theta` is autograd's, kept where a tensor of `theta` requires a
  gradient, as before.
- **`positions`, `cells`, and `tilts` that require a gradient** are taken
  where each is one tensor (`(K, N, 3)` or `(N, 3)`; `(K, 3)` or `(3,)`),
  and `energy` is differentiable in them, as it is in `theta`. The
  function that is differentiated is $U(\mathbf x, \mathsf h)$ at the
  Cartesian positions given, and its backward is a product with what the
  forward kept, without a second pass:
  $\bar{\mathbf x}_{n,i} = g_n\mathbf G_{n,i}$, and for the cell

  $$
  \frac{\partial U}{\partial\mathsf h}\bigg\rvert_{\mathbf x}
  = \mathsf h^{-\mathsf T}
    \Bigl(\mathsf S - \sum_i\mathbf G_i\otimes\mathbf x_i\Bigr)^{\mathsf T}
  $$

  (rows as the rows of $\mathsf h$), of which `cells` takes the diagonal
  and `tilts` the entries $(b, x)$, $(c, x)$, $(c, y)$; it follows from
  the chain rule on the deformation above, and uses $\mathsf S$ only on
  and above the diagonal. The two terms nearly cancel (636 kJ/mol/nm of
  14,000 kJ/mol on the dipeptide), so the sum over the particles is
  formed on the host in f64 in every mode. A gradient has the shape, the
  type, and the device of the tensor it is of; one cell given for every
  frame takes the sum over the frames. The gradient in the cell is not
  $\mathsf S$: the strain moves the positions with the cell, and is the
  output `gradients["strain"]`.
- **The gradients as outputs** depend on `theta` and on the positions,
  and their derivative, a second derivative of $U$, is not implemented:
  each has a `grad_fn` whose backward raises `UnsupportedError` when it is
  given a cotangent that is not zero, as `virial` does (ruling Q5), unless
  the output is detached; an output that reads no tunable of `theta`, in
  a call whose positions and cell take no gradient, is a constant without
  a graph. **A loss on the forces or on the stress can therefore be
  evaluated, and it cannot yet train the tunables: its derivative in
  $\boldsymbol\theta$ is force matching, which #138 defers (stage 4, the
  vector-Jacobian product of the forces in $\boldsymbol\theta$), and it is
  an error, not a zero.**
- **Second derivatives are refused with `UnsupportedError`**, for every
  input: the products of the backward (in `theta`, in the positions, in
  the cell) are operations whose own backward raises, naming the
  derivative that is missing. For `theta` this replaces the error of
  slice 1, PyTorch's `RuntimeError` "Trying to backward through
  mdir.frame_vjp.default but no autograd formula was registered", which
  was a refusal too but said nothing a user of the evaluator could act
  on. The products in the positions and the cell are operations of their
  own for this reason: written as a multiplication of tensors, the
  gradient $g_n\mathbf G_{n,i}$ would be differentiable in $g_n$ alone,
  and a second derivative in the positions would come out as zero.
- In a call whose positions or cell take a gradient, `virial` and every
  observed column are functions of them, whose derivative is not
  implemented either: their backward raises whatever `depends` says.
- With `terms="dependent"`, positions or cells that require a gradient
  are refused as `"positions"` and `"strain"` are.
- Frames of an iterable carry no graph: positions that require a gradient
  inside one are refused (`UnsupportedError`), in `FrameEvaluator.evaluate`
  as well, whose arrays carry none.

**The operations.** `mdir::frame_energies` takes the positions, the
cells, and the tilts as tensors where they require a gradient (empty
tensors otherwise: the frames are read from the record of the call, as
before) and returns, besides its four outputs, the gradient in the
positions, in the strain, and in the six numbers of the cell, each empty
unless it is asked or a backward needs it. `mdir::frame_input_vjp` is the
product of the cotangent of the energies with one of them. It and
`mdir::frame_vjp` have a registered backward that raises.

### Implementation of the gradients

- **The request travels with the frame.** `@mdrtFrameBoundary` takes a
  second buffer, two numbers that the host sets for each frame: whether
  its forces are asked, and its virial. They are not values of the
  activation, as `tunable_gradient` is, so that a call that asks and one
  that does not share an activation and no start is paid.
- **An evaluation of their own, in a branch.** After the evaluation of
  the frame the loop holds two loops of one iteration or none (the form
  that the frame of a reporter has, D207): one for a frame that asks for
  its forces alone, with `md.evaluate @energy ... request [forces]` and
  `mdrt.host_call @mdrtWriteFrameForces`; one for a frame that asks for
  its virial, with `request [forces, virial]`, `call
  @mdrtWriteFrameVirial` with the nine components, and the call of the
  forces if they are asked as well. The forces are those of the
  potential, with the spread of the virtual sites and the restraints,
  without the constraints. So a frame that asks pays one more evaluation
  of the forces ([measured](#the-cost-of-the-gradients)), and a frame
  that does not pays nothing. One loop for either request, with the two
  calls in loops inside it, was measured as well: a frame that asked for
  nothing took 45.35 to 45.65 ms in it, 1% more than on `main`, so the
  forces alone keep a loop of their own. The forces of the two loops
  come from two evaluations, of the same potential at the same frame;
  in the deterministic mode they are the same numbers to the bit, on the
  CPU and on a GPU, in double and mixed precision
  (`python-frame-gradients*.test`).
- **Why not the forces and the virial of the evaluation that is there.**
  That was written first: the two calls took `%f0` and the nine
  components of `%w0` of the evaluation of the frame, and nothing more
  was evaluated. But a frame reads of that evaluation only the energy,
  the trace and the diagonal of the virial, and the square of the forces,
  and a consumer of the whole of either changes what the kernels compute
  and how the loops are joined in every frame, asked or not: on the CPU
  in double precision a frame of the dipeptide that asked for nothing
  took 47.2 ms with both calls and 46.6 ms with the call of the virial
  alone, against 45.1 ms on `main`. With the evaluation in the branch it
  takes 44.75 to 45.04 ms, the text of a frame without the request is that of
  before but for a branch that is not taken, and its numbers are the same
  to the bit whatever a call asks. The price is that of the frames that
  ask.
- **On the host** the forces are written with the opposite sign straight
  into the array that the caller will hold, in their type
  (`Simulation::evaluateFrame` takes the memory); the virial is
  transposed, negated, and completed with a third of the virials of the
  correction for the dispersion and of the background on each axis
  (`Output::getFrameStrain`). A frame that asked and was not handed its
  forces or its virial fails, as a defect of mdir: it is not returned as
  zeros.
- No kernel and no function of a device runtime changes. The text of a
  program of frames changes by the buffer, its two loads, and the two
  branches with their evaluation, so the kept code of every evaluator is compiled anew once
  (`python-frames-loop*.test` read that text and pass unchanged). The
  programs of `mdir run` and of a `Simulation` are the same to the byte.

### Validation of the gradients

`test/Driver/Inputs/python_frame_gradients.py`;
`python-frame-gradients.test`, `-torch.test`, `-torchpme.test` (`REQUIRES:
torch, torchpme`), `-openmm.test` (`REQUIRES: openmm`), and their GPU
twins; the deterministic mode unless stated. Differences are relative to
the largest entry of the reference. "FD" is a central difference of the
evaluator's own `energy` in double precision, extrapolated from two steps
(for a program in mixed precision, of the same model compiled in double
precision on the same target).

**The steps of the differences.** The energy is only continuous where a
pair crosses the cutoff: the direct sum of PME is shifted there (D210),
and its force jumps by about $10^{-3}$ kJ/mol/nm a pair. A difference
over a long step is rough at the particles that have such a pair, and one
over a short step meets the rounding of the energy. Measured on the
dipeptide in double precision, the largest error of 102 components of
$\mathbf G$ is $7\times10^{-3}$ kJ/mol/nm at a step of $10^{-3}$ nm,
$3\times10^{-3}$ at $2\times10^{-5}$, $1\times10^{-4}$ at
$4\times10^{-6}$, and $3\times10^{-4}$ at $10^{-6}$; the tests take
$4\times10^{-6}$ nm for the positions, $10^{-6}$ for the strain, and
$2\times10^{-5}$ nm for the edges of the cell at fixed positions, where
the difference is of a quantity twenty times smaller. The tolerances in
double precision are those of the differences, not of the outputs.

*Systems:* the alanine dipeptide in 382 flexible waters (1,168 atoms,
PME, a switched Lennard-Jones, a soft pair term), orthorhombic; 403
flexible waters in a rhombic dodecahedron sheared by 1% so that its cell
is inside the reduced form (1,209 atoms, PME), triclinic. The CPU is
that of a login node; the GPU is one RTX 4090 in every table of this
section.

| Quantity | Reference | CPU double | GPU double | CPU mixed | GPU mixed | Tolerance (double, mixed) |
|---|---|---|---|---|---|---|
| $\mathbf G$ of the dipeptide: every component of the solute and of 12 atoms of the water, 102 | FD | 3.1e-8 | 3.3e-8 | 2.5e-5 | 3.0e-5 | 5e-7, 1e-4 |
| The diagonal of $\mathsf S$ of the dipeptide | FD under $\varepsilon_{\alpha\alpha}$ | 7.9e-8 | 8.3e-8 | 1.6e-5 | 1.1e-5 | 5e-7, 1e-4 |
| $\mathsf S$ of the water in the triclinic cell, on and above the diagonal | FD under $\varepsilon_{\alpha\beta}$, $\alpha\le\beta$ | 7.4e-9 | 6.1e-9 | 7.4e-6 | 7.9e-6 | 5e-7, 1e-4 |
| The same below the diagonal | FD with the deformed positions and cell rotated back to the lower-triangular form (a QR factorization), which a potential of pairs and of the mesh allows | 1.8e-8 | 2.2e-8 | 3.3e-6 | 3.4e-6 | 5e-7, 1e-4 |
| $\mathbf G$ of the water, 36 components | FD | 9.0e-10 | 9.0e-10 | 5.3e-7 | 5.3e-7 | 5e-7, 1e-4 |
| `virial` $+ \mathrm{tr}\,\mathsf S$, kJ/mol: the dipeptide; the water | 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 1e-9 relative |
| $\mathsf S - \mathsf S^{\mathsf T}$ of the dipeptide, kJ/mol of 14,001 | 0 | 2.3e-13 | 5.1e-13 | 1.3e-4 | 2.3e-4 | 1e-9, 1e-4 relative |
| $\mathbf G$ of a frame of the dipeptide | the forces of the sampler's state at that step, with the opposite sign (sums in a step and at a start: another order) | 2.5e-16 | 3.9e-16 | 1.5e-7 | 1.9e-7 | 1e-13, 1e-5 |
| The same without the deterministic mode: $\mathbf G$ in float32 in mixed | the forces of a sampler without it | 2.3e-16 | 3.4e-16 | 9.2e-8 | 1.7e-7 | 1e-10, 1e-5 |
| $\mathbf G$ of a program without the deterministic mode | $\mathbf G$ of the program with it: two programs, whose kernels differ on a device (no fused multiply-add and one order of the sums with it, D201) | 0 | 6.1e-12 | 3.5e-8 | 4.8e-7 | 1e-10, 1e-5 |
| With the correction for the dispersion under a plain cutoff and a net charge of 0.44 e (what the host adds at the volume): the diagonal of $\mathsf S$; `virial` $+ \mathrm{tr}\,\mathsf S$, kJ/mol | FD; 0 | 2.4e-8; 0 | 7.0e-9; 3.6e-12 | 1.1e-5; 3.6e-12 | 1.4e-5; 3.6e-12 | 5e-7, 1e-4; 1e-9 relative |
| With a term of the absolute positions on the water ($\mathsf S - \mathsf S^{\mathsf T}$ is 93,180 kJ/mol): $\mathsf S$ on and above the diagonal; $\mathbf G$, 24 components | FD | 2.0e-9; 1.9e-9 | 1.0e-9; 1.8e-9 | 1.5e-6; 2.1e-7 | 1.6e-6; 2.1e-7 | 5e-7, 1e-4 |
| With rigid water: $\mathbf G$, 36 components, in every position; the diagonal of $\mathsf S$ | FD | 1.1e-7; 1.8e-8 | 1.1e-7; 5.6e-9 | 2.7e-6; 4.1e-7 | 2.4e-6; 4.8e-7 | 5e-7, 1e-4 |
| With 387 virtual sites (OPC): $\mathbf G$ of 36 components of their atoms; the diagonal of $\mathsf S$; the rows of the sites | FD; FD; 0 | 1.0e-7; 9.9e-8; 0 | 1.0e-7; 1.2e-7; 0 | 1.5e-5; 5.0e-6; 3.1e-8 | 1.5e-5; 3.8e-6; 3.1e-8 | 5e-7, 1e-4; 1e-6 for the rows in mixed |
| With restraints to reference positions on the heavy atoms of the solute: $\mathbf G$ of its first 10 atoms; `"strain"` | FD; refused | 5.7e-9 | 5.3e-9 | 1.9e-6 | 1.9e-6 | 5e-7, 1e-4 |
| Calls that ask for nothing, for one gradient, and for all three, as states, as an array, and one frame alone, in one activation: every output, the forces asked alone and with the strain among them | the first evaluation | 0 | 0 | 0 | 0 | 0 |

The tolerance of the two rows without the deterministic mode was
$10^{-12}$ in double precision when the tests were written on the CPU,
where both differences are 0 and $2\times10^{-16}$; on the device the
program without the mode differs from the one with it by
$6.1\times10^{-12}$ of the largest force ($2.4\times10^{-8}$ kJ/mol/nm of
3,991), the rounding of sums of forces that cancel taken with and without
fused multiply-adds, so the tolerance is $10^{-10}$.


**Against torch-pme** [[Loche2025]](references.md#loche2025), the
reciprocal part (moved here from #203 by #138). The evaluator returns
whole gradients, so the Coulomb part is taken as the difference of the
gradients at the charges of the model and at charges of zero (every
charge a tunable), at the same frame. The reference is torch-pme 0.5.0's
sum of the mesh at the same positions, cell, charges, $\beta$ = 3.5/nm,
grid, and order 4, with the pair sums written in PyTorch as
`python_tunable_gradient.py` has them (the direct sum over the pairs
within the cutoff, the excluded pairs, the 1-4 pairs), differentiated by
autograd in the positions and in a strain of zero that deforms the
positions and the cell: all nine components of $\mathsf S$ are compared.
The tolerance is three times what the two programs change by between the
grid and one of twice the points. Each entry is CPU, GPU.

| System | Grid | $\mathbf G$, double | $\mathbf G$, mixed | $\mathsf S$, double | $\mathsf S$, mixed | MDIR and torch-pme change by ($\mathbf G$; $\mathsf S$) |
|---|---|---|---|---|---|---|
| Dipeptide, 1,168 charges, 79,024 pairs | 64 | 5.9e-5, 5.9e-5 | 5.9e-5, 5.9e-5 | 2.1e-5, 2.1e-5 | 2.0e-5, 2.0e-5 | 3.4e-5 and 5.8e-5; 2.3e-5 and 1.8e-5 |
| | 128 | 4.8e-6, 4.8e-6 | 5.3e-6, 5.1e-6 | 7.7e-7, 7.7e-7 | 5.8e-7, 6.8e-7 | |
| Water in the rhombic dodecahedron, 1,209 charges, 124,961 pairs | 32 | 8.5e-4, 8.5e-4 | 8.5e-4, 8.5e-4 | 3.0e-4, 3.0e-4 | 3.0e-4, 3.0e-4 | 8.1e-4 and 9.6e-4; 3.2e-4 and 2.5e-4 |
| | 64 | 5.2e-5, 5.2e-5 | 5.3e-5, 5.3e-5 | 8.8e-6, 8.8e-6 | 8.5e-6, 8.7e-6 | |
| JAC, 23,558 charges, 2,430,398 pairs (by hand: the system is not in the repository; no CPU run in mixed) | 128 | 7.5e-5, 7.5e-5 | GPU 7.6e-5 | 7.6e-6, 7.6e-6 | GPU 7.7e-6 | 3.1e-5 and 8.3e-5; 1.5e-5 and 8.8e-6 |
| | 256 | 4.9e-6, 4.9e-6 | GPU 4.3e-5 | 8.0e-7, 8.0e-7 | GPU 1.4e-6 | |

In mixed precision on JAC at 256 points the difference of $\mathbf G$,
$4.3\times10^{-5}$, is that of the f32 kernels, which a finer mesh does
not lower. On a device the two systems run in a process each: the
runtime keeps at most 8 plans of FFT a process, and a program in mixed
precision with tunable charges takes four a grid (#323).


**Against OpenMM 8.6.1**, the whole potential: the dipeptide with PME at
$\beta$ = 3.5/nm and the same grid, a cutoff of 0.8 nm, and the
Lennard-Jones shifted to 0 there, whose forces are those of OpenMM's
plain cutoff. $\mathbf G$ is compared with OpenMM's forces (its Reference
platform, double precision). OpenMM reports no virial, and it cuts the
Lennard-Jones and the direct sum without a shift, so that its energy
jumps where a pair crosses the cutoff: $\mathsf S$ is compared with
central differences, under each of the nine components of the strain, of
OpenMM's energy less $u_{ij}(r_c)$ of every pair within the cutoff that is
not an exception (summed in the test from OpenMM's own parameters), the
energy whose derivative its forces are; the deformed cell is rotated
back to the reduced form, which OpenMM takes as well. OpenMM's B-splines
are of order 5 and MDIR's of order 4.

| Quantity | Grid | CPU double | GPU double | CPU mixed | GPU mixed | MDIR and OpenMM change by |
|---|---|---|---|---|---|---|
| $\mathbf G$, 1,168 particles | 64 | 3.4e-5 | 3.4e-5 | 8.8e-5 | 8.9e-5 | 3.3e-5 and 2.4e-6 |
| | 128 | 3.7e-6 | 3.7e-6 | 8.7e-5 | 8.8e-5 | |
| $\mathsf S$, nine components | 64 | 8.4e-6 | 8.4e-6 | 2.0e-5 | 2.2e-5 | 1.1e-5 and 6.0e-7 |
| | 128 | 3.5e-6 | 3.5e-6 | 7.8e-6 | 1.0e-5 | |
| JAC, 23,558 particles (by hand): $\mathbf G$ | 128 | 2.1e-5 | | | 1.0e-4 | 2.0e-5 and 1.9e-6 |
| | 256 | 2.4e-6 | | | 1.0e-4 | |
| JAC: $\mathsf S$, nine components | 128 | 7.7e-6 | | | 1.2e-5 | 6.8e-6 and 4.3e-7 |
| | 256 | 4.6e-7 | | | 4.8e-6 | |

In mixed precision the difference of $\mathbf G$ at the finer grid is
that of the f32 kernels of the pairs, which a finer mesh does not lower:
$9\times10^{-5}$ on the dipeptide and $1.0\times10^{-4}$ on JAC, where
the tolerance of the test, three times the changes with the grid and
$5\times10^{-5}$ for f32, is $1.2\times10^{-4}$. JAC was not run on a
GPU in double precision or on the CPU in mixed.

**The adapter** (`python-frame-gradients-torch*.test`); on a GPU the
tunables are tensors of the device:

| Quantity | Reference | CPU double | GPU double | CPU mixed | GPU mixed | Tolerance (double, mixed) |
|---|---|---|---|---|---|---|
| The gradient of autograd in positions `(K, N, 3)` that require one, of $\sum_n c_nU_n$, on the host and on the device of `theta`; in float32 positions; for one frame `(N, 3)` | $c_n\mathbf G_n$ of the evaluator | 0 | 0 | 0 | 0 | 0 |
| The gradient in the edges `(K, 3)` of three frames of the dipeptide at fixed positions (largest 636 kJ/mol/nm) | FD in each edge | 2.0e-6 | 2.0e-6 | 1.6e-5 | 7.9e-6 | 1e-5, 1e-3 |
| One cell `(3,)` for every frame | the sum over the frames of the gradients of each | within the tolerance | within | within | within | 1e-13 |
| In the triclinic cell of the water: the diagonal; the tilts | FD in each | 2.8e-8; 3.2e-8 | 2.8e-8; 3.2e-8 | 2.8e-6; 6.8e-6 | 2.8e-6; 6.9e-6 | 1e-5, 1e-3 |
| `torch.autograd.gradcheck` of the energy in the diagonal and the tilts (`eps` 2e-5, `atol` 2e-2, `rtol` 1e-5) | the differences of PyTorch | passes | passes | | | |
| `gradients["positions"]`, `gradients["strain"]` as tensors | the arrays of the evaluator | 0 | 0 | 0 | 0 | 0 |
| The gradients in the positions, the cells, and the tunable under `torch.compile(backend="aot_eager")` | eager | 0 | 0 | 0 | 0 | 0 |


Also checked: the default (`jacobian`, an empty `gradients`); `gradients`
in any order and with a name twice; the types of `"positions"` in the
three modes; read-only arrays and mapping; the refusals of
[What is returned](#what-is-returned), `gradient=` among them
(`TypeError`), and that the evaluator gives the numbers of before after
each; in the adapter, a backward through a gradient that is an output
and through the virial of positions that require a gradient
(`UnsupportedError`), the same loss with the output detached, every
second derivative in the positions, the cells, and the tunables, alone
and in a loss that has another path to the same inputs
(`UnsupportedError`, in `python-frames-torch*.test` for the tunables),
the dependent terms, frames of an iterable that require a gradient, and
the records of the adapter empty after every call, a failed one included.

### The cost of the gradients

`scripts/benchmarks/frames/bench.py --gradients
none/positions/strain/positions,strain`, the same stored frames for both
builds, ms per frame of `evaluate`, the least of 5 repeats after three
passes that are not timed; "before" is `main` at f378ac4, whose
`evaluate` has no such argument. One run each.

*The CPU of a shared login node (32 cores, the default threads), double
precision, the dipeptide in water, the constant $l$ of a soft pair term
tunable; a step takes 15.0 ms.* `main` gave 44.84 to 45.30 ms at $K = 1$
and 44.93 to 45.27 ms at $K = 16$ over five runs, and the default call
of this branch 44.75 and 45.04 ms at $K = 1$ and 44.84 and 44.95 ms at
$K = 16$ over two.

| Asked besides `"tunables"` | $K$ = 1 | $K$ = 16 | Against `main` |
|---|---|---|---|
| before (`main`) | 44.90 | 45.19 | |
| nothing (the default) | 45.04 | 44.95 | within the spread of `main` |
| `"positions"` | 57.98 | 58.09 | +29% |
| `"strain"` | 60.30 | 60.28 | +34% |
| `"positions"`, `"strain"` | 60.47 | 60.59 | +34% |

The evaluator and its first frame take 1.08 s where they took 0.71 s:
the program of frames holds the two evaluations of the branches besides
its own.

*One RTX 4090 (a node of the cluster, the device idle before each run),
mixed precision, not the deterministic mode; the same stored frames for
both builds; `main` at f378ac4.* ms per frame.

| System | Asked besides `"tunables"` | $K$ = 1 | 4 | 16 | 64 | 256 | Against `main` at $K$ = 64 |
|---|---|---|---|---|---|---|---|
| Dipeptide, 1,168 atoms (a step 0.0525 ms) | before (`main`) | 0.3266 | 0.3117 | 0.3017 | 0.2998 | 0.2984 | |
| | nothing (the default) | 0.3206 | 0.3037 | 0.2987 | 0.2978 | 0.2959 | −0.7% |
| | `"positions"` | 0.3694 | 0.3492 | 0.3430 | 0.3401 | 0.3375 | +13% |
| | `"strain"` | 0.3994 | 0.3806 | 0.3742 | 0.3718 | 0.3715 | +24% |
| | `"positions"`, `"strain"` | 0.4158 | 0.3961 | 0.3896 | 0.3878 | 0.3861 | +29% |
| JAC, 23,558 atoms (a step 0.1270 ms) | before (`main`) | 0.7258 | 0.7076 | 0.7020 | 0.7010 | 0.7009 | |
| | nothing (the default) | 0.7267 | 0.7092 | 0.7050 | 0.7036 | 0.7039 | +0.4% |
| | `"positions"` | 0.8675 | 0.8488 | 0.8425 | 0.8412 | 0.9142 | +20% |
| | `"strain"` | 0.9164 | 0.8971 | 0.8918 | 0.8911 | 0.8905 | +27% |
| | `"positions"`, `"strain"` | 0.9895 | 0.9670 | 0.9613 | 0.9596 | 1.0266 | +37% |

A frame that asks for nothing costs what it costs on `main`, on both
systems (one run each: the differences of 0.7% and 0.4% have opposite
signs and are not resolved). On the device `"positions"` with `"strain"`
costs more than either, unlike on the CPU: the copy of the forces from
the device, $12N$ bytes in mixed precision, waits for the kernels of the
evaluation in the branch. With `"positions"` on JAC a call of 256 frames
takes 0.07 ms a frame more than one of 64: the array that it returns is
72 MB, whose pages the host touches for the first time. The evaluator
and its first frame take 2.32 s on the dipeptide and 2.98 s on JAC where
they took 1.50 s and 2.16 s.

### Maintainer rulings (gradients)

The seven questions put on PR #314 were decided as proposed, but the
first.

| | Question | Ruling |
|---|---|---|
| G1 | Names and shapes | one argument, `gradients=`, a tuple among `"tunables"`, `"positions"`, `"strain"`; the argument `gradient` is removed with no alias and no hint; `jacobian` stays the output of `"tunables"`, and `gradients` holds `"positions"` `(K, N, 3)` and `"strain"` `(K, 3, 3)`; the same for `mdir.torch.evaluate`. (Proposed: a second argument beside `gradient`) |
| G2 | The gradient in the strain | $\partial U/\partial\boldsymbol\varepsilon$ of the energy that `energy` returns, kJ/mol, float64, nine components for every cell, not symmetrized, not divided by the volume |
| G3 | Constraints and virtual sites | the plain gradient; $\mathsf S$ without the term of the constraint forces, and the document says which of `virial` and $-\mathrm{tr}\,\mathsf S$ a pressure takes; the row of a site is zero |
| G4 | The type of `"positions"` in `mixed` | that of the forces of the program, no `dtype` argument; `"strain"` float64 |
| G5 | The autograd contract | positions, cells, and tilts that require a gradient are taken, $U(\mathbf x, \mathsf h)$ at fixed positions; the gradients as outputs raise from a backward unless detached; every second derivative raises `UnsupportedError`, for `theta` as well. Force matching stays deferred: only the refusal |
| G6 | `terms="dependent"` | refused, `"gradients"` in `unavailable` |
| G7 | The cost when not asked | a branch that is not taken, with the measurement; the kept code of every evaluator is compiled anew once |

## Slices

| Slice | Contents |
|---|---|
| 1 (done) | `FrameEvaluator`, `evaluate` over arrays and iterables with one evaluation per frame, `energy`, `virial`, `volume`, `observables`, `depends`, the Jacobian kept or the second pass, `mdir.torch.evaluate`, `mdir.KB` |
| The dependent terms alone, and the example without frames (done, D243) | `terms="dependent"`, `dependent_energy`, `plan["terms"]`; the sums of `observe` for a potential linear in its tunables |
| 1, after D238 and D239 (done) | the tilts of each frame of a triclinic cell; the frames of `mdir.read_h5md` as input |
| 2 (done, D251, #291) | the frames inside one activation, in the order of the input; the program of the dependent terms kept with its program |
| 2b | frames read from a device tensor without a copy to the host (the positions of a frame are one copy of 0.03 ms on JAC now) |
| 3 (done, D252, #313) | the gradients `positions` and `strain` as outputs, with the comparison of the forces and the virial of PME with torch-pme (moved here from #203 by #138) and of the whole potential with OpenMM; autograd in the positions and the cell through `mdir.torch.evaluate`; one argument `gradients=` |
| 3b | the metatomic shape of #138 (a list of systems in, outputs out) |
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
