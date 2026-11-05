"""The JAX adapter of the frame evaluator, `mdir.jax` (D371,
docs/python-frames.md, The JAX adapter), on the CPU and on a CUDA device
(D373).

Usage: python_frames_jax.py ROOT SCENARIO PRECISION TARGET [FILE]

ROOT is Inputs/dipeptide. PRECISION is Double or Mixed (deterministic).
TARGET is CPU or GPU, the target of the programs; with GPU the arrays of
JAX are on its CUDA device, and the calls go to the handler of "CUDA"
(JAX_PLATFORMS=cpu keeps them on the CPU for CPU).

Scenarios:
  gate       against FrameEvaluator.evaluate of the same program, to the bit:
             the energies, the gradient in the positions, and the rows of the
             Jacobian in the tunables; vmap against a loop of single frames
             and jit against eager, to the bit, nested vmap and an axis of
             the frames other than the first too; in double precision,
             central differences of the energy in the tunables, the
             positions, and the cell (the edges of an orthorhombic cell and
             the six numbers of a triclinic one), with the step chosen from
             the second derivative, and jax.test_util.check_grads of the
             first order in both modes
  refusals   every row of the refusal matrix raises, never returns a value;
             data errors name the frame and leave the function usable; the
             dtype table; the identity of the program in a compiled call
  torch-dump  (an interpreter with torch) mdir.torch.evaluate on frames of
             the dipeptide and of water in a triclinic cell, written to FILE
  torch-compare  (an interpreter with jax) the same through mdir.jax, against
             FILE, to the bit
  device     (GPU) the handler of CUDA against that of the CPU on the same
             program, to the bit, for a program of the GPU and one of the
             CPU; the order of the calls on XLA's stream: JAX ops that
             produce the positions and consume the energies and the
             gradients with no wait in between, against synchronized calls;
             a call on another device than the program's is refused
  precisions (GPU) mixed against double on the same frames, deterministic
             and not, through mdir.jax on the device, at the tolerances of
             the evaluator in mixed precision
  sanitize   (GPU) a few calls of each kind, for compute-sanitizer
"""
import gc
import sys

import numpy as np

root, scenario, precision, target = sys.argv[1:5]
mode = f"{target} {precision}"
double = precision == "Double"
torch_side = scenario == "torch-dump"

if not torch_side:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.test_util

import mdir

if not torch_side:
    import mdir.jax
    # The arrays of a GPU run are on the CUDA device, those of a CPU run on
    # the CPU.
    assert jax.default_backend() == ("gpu" if target == "GPU" else "cpu"), jax.default_backend()

SOLUTE = 22


def compile_(system, state, precision_=None, deterministic=True, target_=None):
    execution = mdir.Execution()
    execution.target = getattr(mdir.Target, target_ or target)
    execution.precision = getattr(mdir.Precision, precision_ or precision)
    execution.deterministic = deterministic
    execution.reorder = False
    integrator = mdir.Integrator()
    integrator.timestep = 0.0005
    return mdir.compile(system, state, integrator, mdir.Ensemble(), execution, mdir.Schedule())


def one_pair(system, name):
    top = system.topology
    names, pairs = list(top.type_names), np.array(top.type_pairs)
    a = names.index(name)
    return np.where((pairs[:, 0] == a) & (pairs[:, 1] == a), 0, -1)


def tunables(system, charges=True):
    """sigma of the pair OW-OW, and the charge of each atom of the solute
    (one entry each), so that params has two names and 1 + 22 entries."""
    given = [mdir.Tunable("sigma", "sigma_pair", map=one_pair(system, "OW"))]
    if charges:
        count = len(system.topology.atom_names)
        given.append(mdir.Tunable("q", "charge",
                                  map=np.where(np.arange(count) < SOLUTE, np.arange(count), -1)))
    system.tunables = given
    system.tunable_gradient = True


def dipeptide(restrained=False, smooth=False):
    """The dipeptide in 382 waters with PME and a switched Lennard-Jones.
    `smooth`: the direct sum of PME screened to 1e-10 at the cutoff
    (`pme_tolerance`), so that its force, shifted to zero energy there,
    jumps by 1e-10 of its size when a pair crosses the cutoff (by 1e-5 at
    the default): the energy is then continuously differentiable to the
    accuracy of a central difference, which a pair that crosses the cutoff
    within a step would otherwise spoil."""
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.electrostatics = mdir.Electrostatics.PME
    system.truncation, system.switch_distance = mdir.Truncation.Switch, 0.7
    system.dispersion = mdir.DispersionCorrection.None_
    if smooth:
        system.pme_tolerance = 1e-10
    if restrained:
        system.restraints = [mdir.Restraint("!:WAT & !@H*", 500.0)]
    tunables(system)
    return system, state


def vectors(diagonal, tilt):
    return np.array([[diagonal[0], 0.0, 0.0], [tilt[0], diagonal[1], 0.0],
                     [tilt[1], tilt[2], diagonal[2]]])


def water(smooth=False):
    """403 flexible waters with PME in a rhombic dodecahedron, sheared by 1%
    with its positions, so that the cell is inside the reduced form and
    stays so under the steps of a difference (the dodecahedron has
    c_x = a_x / 2 and c_y = b_y / 2, the bound)."""
    triclinic = root + "/../triclinic"
    loaded = mdir.load_gromacs(triclinic + "/water.top", triclinic + "/dodecahedron.gro",
                               defines=["FLEXIBLE"])
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.8
    system.truncation = mdir.Truncation.None_
    system.dispersion = mdir.DispersionCorrection.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.pme_grid = [28] * 3
    if smooth:
        system.pme_tolerance = 1e-10
    tunables(system, charges=False)
    x = np.array(state.positions)
    h = vectors(state.cell.diagonal, state.cell.tilt)
    f = np.eye(3)
    f[0, 2] = f[1, 2] = f[0, 1] = -0.01
    x1, h1 = x @ f.T, h @ f.T
    q, r = np.linalg.qr(h1.T)
    q = q * np.sign(np.diag(r))
    return system, state, x1 @ q, np.tril(h1 @ q)


def frames_of(state, count, seed=5):
    """`count` frames near the state: its positions moved by 2e-3 nm at
    random, each with its own orthorhombic cell."""
    rng = np.random.default_rng(seed)
    x = np.array(state.positions)
    frames = x[None] + rng.normal(0.0, 2e-3, (count,) + x.shape)
    cells = np.array(state.cell.diagonal)[None] * (1.0 + 1e-3 * rng.standard_normal((count, 3)))
    return frames, cells


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), (text, str(exc))
        return str(exc)
    raise AssertionError(f"expected {error.__name__} ({text}); a value was returned")


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def leaves(tree):
    return [np.asarray(v) for v in jax.tree.leaves(tree)]


# --- gate -------------------------------------------------------------------

def directional(value, gradient, v):
    """The derivative of value(t) = U(y + t v) at t = 0 by central
    differences, and the exact one, gradient(0) . v, where gradient(t) is
    the exact gradient of U at y + t v and v a unit vector.

    The differences are extrapolated to a zero step by Ridders' method
    (Ridders, Adv. Eng. Software 4, 75 (1982); the tableau of Press et al.,
    Numerical Recipes, Section 5.7): steps h0 / 1.4^k, k = 0 to 15, and the
    estimate whose two neighbours in the tableau agree best, over the whole
    tableau (without the early stop of Numerical Recipes, which a step that
    carries pairs across the cutoff can trigger). The first step
    is chosen from the second derivative: the gradient changes by itself
    over l = |grad U| / |H v|, where H v, the product of the Hessian with
    v, is the central difference of the exact gradient over 1e-6, and
    h0 = l / 20. Returns the exact derivative, the difference, and the step
    of the estimate taken."""
    v = np.asarray(v).ravel()
    exact = np.asarray(gradient(0.0)).ravel()
    change = (np.asarray(gradient(1e-6)).ravel() - np.asarray(gradient(-1e-6)).ravel()) / 2e-6
    h = np.linalg.norm(exact) / max(np.linalg.norm(change), 1e-300) / 20.0
    con = 1.4
    table = [[0.0] * 16 for _ in range(16)]
    best, error, taken = None, np.inf, h
    for i in range(16):
        if i:
            h /= con
        table[0][i] = (float(value(h)) - float(value(-h))) / (2.0 * h)
        factor = con * con
        for j in range(1, i + 1):
            table[j][i] = (table[j - 1][i] * factor - table[j - 1][i - 1]) / (factor - 1.0)
            factor *= con * con
            estimate = max(abs(table[j][i] - table[j - 1][i]),
                           abs(table[j][i] - table[j - 1][i - 1]))
            if estimate <= error:
                error, best, taken = estimate, table[j][i], h
    return float(exact @ v), best, taken


def run_gate():
    system, state = dipeptide()
    program = compile_(system, state)
    f = mdir.jax.energy_fn(program)
    params = mdir.jax.params(program)
    assert list(params) == ["sigma", "q"] and params["q"].shape == (SOLUTE,)
    frames, cells = frames_of(state, 3)
    evaluator = mdir.FrameEvaluator(program)
    out = evaluator.evaluate(frames, cells, gradients=("tunables", "positions"))
    jx, jc = jnp.asarray(frames), jnp.asarray(cells)

    # Against the evaluator, to the bit.
    batched = jax.vmap(f, in_axes=(None, 0, 0))
    energies = batched(params, jx, jc)
    by_x = jax.vmap(jax.grad(f, argnums=1), in_axes=(None, 0, 0))(params, jx, jc)
    rows = jax.jacrev(batched)(params, jx, jc)
    exact = (same(energies, out.energy) and same(by_x, out.gradients["positions"]) and
             all(same(rows[n], out.jacobian[n]) for n in params))
    print(f"{mode}: the energies, the gradient in the positions, and the Jacobian in the "
          f"tunables against FrameEvaluator.evaluate, to the bit: {exact}")
    assert exact
    exact = same(f(params, jx[0], jc[0]), out.energy[0]) and same(
        jax.jit(f)(params, jx[1], jc[1]), out.energy[1])
    print(f"{mode}: one frame, eager and under jit, to the bit: {exact}")
    assert exact

    # vmap against a loop of single frames, jit against eager.
    grad = jax.grad(f, argnums=(0, 1, 2))
    loop = [grad(params, jx[k], jc[k]) for k in range(3)]
    compiled = [jax.jit(grad)(params, jx[k], jc[k]) for k in range(3)]
    mapped = jax.vmap(grad, in_axes=(None, 0, 0))(params, jx, jc)
    jit_mapped = jax.jit(jax.vmap(grad, in_axes=(None, 0, 0)))(params, jx, jc)
    looped = [np.stack(parts) for parts in zip(*[leaves(g) for g in loop])]
    exact = (all(same(a, b) for a, b in zip(looped, leaves(mapped))) and
             all(same(a, b) for a, b in zip(looped, leaves(jit_mapped))) and
             all(same(a, b) for g, c in zip(loop, compiled)
                 for a, b in zip(leaves(g), leaves(c))))
    print(f"{mode}: the energy's gradients in params, the positions, and the cell of each frame, "
          f"vmap and jit(vmap) against a loop, jit against eager, to the bit: {exact}")
    assert exact
    four, four_cells = frames_of(state, 4, seed=6)
    nested = jax.vmap(jax.vmap(f, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(
        params, jnp.asarray(four).reshape((2, 2) + four.shape[1:]),
        jnp.asarray(four_cells).reshape(2, 2, 3))
    moved = jax.vmap(f, in_axes=(None, 1, 0))(params, jnp.moveaxis(jnp.asarray(four), 0, 1),
                                              jnp.asarray(four_cells))
    shared = jax.vmap(f, in_axes=(None, 0, None))(params, jnp.asarray(four), jc[0])
    single = np.array([f(params, four[k], four_cells[k]) for k in range(4)])
    exact = (same(np.asarray(nested).reshape(4), single) and same(moved, single) and
             same(shared, np.array([f(params, four[k], jc[0]) for k in range(4)])))
    print(f"{mode}: nested vmap, frames on the second axis, and one cell for every frame against "
          f"a loop, to the bit: {exact}")
    assert exact

    if not double:
        return
    # Central differences in double precision, of the smooth form of the
    # program (`dipeptide`).
    system, state = dipeptide(smooth=True)
    program = compile_(system, state)
    f = mdir.jax.energy_fn(program)
    rng = np.random.default_rng(7)
    x0, c0 = jx[0], jc[0]
    g_theta, g_x, g_c = jax.grad(f, argnums=(0, 1, 2))(params, x0, c0)
    worst = {}

    def record(kind, found):
        exact_, difference, step = found
        error = abs(difference - exact_) / abs(exact_)
        before = worst.get(kind, (0.0, np.inf, 0.0))
        worst[kind] = (max(before[0], error), min(before[1], step), max(before[2], step))

    def flat(tree):
        return np.concatenate([np.asarray(leaf).ravel() for leaf in jax.tree.leaves(tree)])

    for name in params:
        direction = rng.standard_normal(params[name].shape)
        v = {n: jnp.zeros_like(params[n]) for n in params}
        v[name] = jnp.asarray(direction / np.linalg.norm(direction))

        def along(t, v=v):
            return {n: params[n] + t * v[n] for n in params}
        record(f"params['{name}']", directional(
            lambda t, along=along: f(along(t), x0, c0),
            lambda t, along=along: flat(jax.grad(f)(along(t), x0, c0)), flat(v)))
    directions = []
    for _ in range(3):
        direction = rng.standard_normal(x0.shape)
        directions.append(jnp.asarray(direction / np.linalg.norm(direction)))
    for i, axis in ((0, 0), (5, 2), (100, 1)):
        directions.append(jnp.zeros(x0.shape).at[i, axis].set(1.0))
    for v in directions:
        record("the positions", directional(
            lambda t, v=v: f(params, x0 + t * v, c0),
            lambda t, v=v: jax.grad(f, argnums=1)(params, x0 + t * v, c0), v))
    for a in range(3):
        e = jnp.zeros(3).at[a].set(1.0)
        record("the edges of the cell", directional(
            lambda t, e=e: f(params, x0, c0 + t * e),
            lambda t, e=e: jax.grad(f, argnums=2)(params, x0, c0 + t * e), e))
    wsystem, wstate, wx, wh = water(smooth=True)
    wprogram = compile_(wsystem, wstate)
    wf, wparams = mdir.jax.energy_fn(wprogram), mdir.jax.params(wprogram)
    wx, wh = jnp.asarray(wx), jnp.asarray(wh)
    lower = np.tril(np.ones((3, 3)))
    for a, b in ((0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1)):
        e = jnp.zeros((3, 3)).at[a, b].set(1.0)
        # The lower triangle alone: the primal takes lower-triangular cells.
        record("the six numbers of a triclinic cell (3, 3)", directional(
            lambda t, e=e: wf(wparams, wx, wh + t * e),
            lambda t, e=e: np.asarray(jax.grad(wf, argnums=2)(wparams, wx, wh + t * e)) * lower,
            e))
    for kind, (error, low, high) in worst.items():
        print(f"{mode}: central differences in {kind}: largest relative error {error:.1e} "
              f"(tolerance 1e-06; steps {low:.1e} to {high:.1e})")
    for kind, (error, low, high) in worst.items():
        assert error <= 1e-6, (kind, error)

    # jax.test_util.check_grads, first order, forward and reverse, in one
    # random direction of its own. Its difference is a plain central one,
    # without extrapolation, so it takes a tolerance on the scale of the
    # gradient as well: atol = 1e-6 |grad| |t|, the size of the derivative in
    # a direction t along the gradient, with |t| = n^(1/2) for the n normal
    # numbers of its direction, and rtol = 1e-6. The steps: 1e-7 in params
    # and the positions; 3e-5 nm in the cell, where the energy at fixed
    # positions is rougher (its rounding, measured: 1e-5 and 3e-5 pass,
    # 1e-4 errs by 6e-5 relative in a direction where the derivative
    # nearly cancels).
    def check(fun, argument, eps, label):
        scale = float(np.linalg.norm(flat(jax.grad(fun)(argument))))
        count = flat(argument).size
        jax.test_util.check_grads(fun, (argument,), order=1, modes=("fwd", "rev"), eps=eps,
                                  rtol=1e-6, atol=1e-6 * scale * np.sqrt(count))
        print(f"{mode}: jax.test_util.check_grads(order=1, modes=('fwd', 'rev')) in {label}, "
              f"step {eps:.0e}: passed")

    def lower_cell(six):
        return jnp.array([[six[0], 0.0, 0.0], [six[3], six[1], 0.0], [six[4], six[5], six[2]]])

    six = jnp.array([wh[0, 0], wh[1, 1], wh[2, 2], wh[1, 0], wh[2, 0], wh[2, 1]])
    check(lambda p: f(p, x0, c0), params, 1e-7, "params")
    check(lambda y: f(params, y, c0), x0, 1e-7, "the positions")
    check(lambda c: f(params, x0, c), c0, 3e-5, "the edges")
    check(lambda s: wf(wparams, wx, lower_cell(s)), six, 3e-5,
          "the six numbers of a triclinic cell (3, 3)")


# --- refusals -----------------------------------------------------------------

def run_refusals():
    system, state = dipeptide()
    program = compile_(system, state)
    f = mdir.jax.energy_fn(program)
    params = mdir.jax.params(program)
    x0 = jnp.asarray(np.array(state.positions))
    c0 = jnp.asarray(np.array(state.cell.diagonal))
    h0 = jnp.diag(c0)
    u0 = np.asarray(f(params, x0, c0))
    second = {
        "jax.hessian in params": lambda: jax.hessian(f)(params, x0, c0),
        "jax.hessian in the positions under jit":
            lambda: jax.jit(jax.hessian(f, argnums=1))(params, x0, c0),
        "jax.jacfwd(jax.grad) in the positions":
            lambda: jax.jacfwd(jax.grad(f, argnums=1), argnums=1)(params, x0, c0),
        "jax.jvp(jax.grad) in the positions":
            lambda: jax.jvp(lambda y: jax.grad(f, argnums=1)(params, y, c0), (x0,), (x0,)),
        "jax.grad of a function of jax.grad":
            lambda: jax.grad(lambda y: jax.grad(f, argnums=1)(params, y, c0).sum())(x0),
        "jax.jacrev(jax.jacrev) in params": lambda: jax.jacrev(jax.jacrev(f))(params, x0, c0),
        "jax.jacfwd(jax.jacfwd) in the cell":
            lambda: jax.jacfwd(jax.jacfwd(f, argnums=2), argnums=2)(params, x0, c0),
        "the mixed derivative of the forces in params (force matching)":
            lambda: jax.jacrev(lambda p: jax.grad(f, argnums=1)(p, x0, c0))(params),
        "jax.hessian in a (3, 3) cell": lambda: jax.hessian(f, argnums=2)(params, x0, h0),
    }
    for label, call in second.items():
        expect(mdir.UnsupportedError, call, "a second derivative of the energy")
    expect(NotImplementedError,
           lambda: jax.vmap(f, in_axes=(0, None, None))(
               {n: jnp.stack([v, v]) for n, v in params.items()}, x0, c0), "vmap over params")
    jax.config.update("jax_enable_x64", False)
    try:
        expect(mdir.InputError, lambda: f(params, x0, c0), "jax_enable_x64")
        expect(mdir.InputError, lambda: mdir.jax.energy_fn(program), "jax_enable_x64")
    finally:
        jax.config.update("jax_enable_x64", True)
    expect(mdir.InputError, lambda: f({"sigma": params["sigma"]}, x0, c0), "missing ['q']")
    expect(mdir.InputError, lambda: f(dict(params, eps=params["sigma"]), x0, c0),
           "not tunables of the program ['eps']")
    expect(mdir.InputError, lambda: f(dict(params, q=params["q"][:3]), x0, c0), "shape")
    expect(mdir.InputError, lambda: f(dict(params, q=params["q"].astype(jnp.float32)), x0, c0),
           "float64")
    expect(mdir.InputError, lambda: f([params["sigma"], params["q"]], x0, c0), "dict")
    expect(mdir.InputError, lambda: f(params, x0[:10], c0), "positions of shape")
    expect(mdir.InputError, lambda: f(params, x0.astype(jnp.int32), c0), "float64 or float32")
    expect(mdir.InputError, lambda: f(params, x0, c0.astype(jnp.float32)), "cell is float64")
    expect(mdir.InputError, lambda: f(params, x0, jnp.ones(6)), "shape (3,)")
    expect(mdir.InputError, lambda: mdir.jax.energy_fn(program, jacobian_bytes=-1),
           "jacobian_bytes")
    print(f"{mode}: refused at trace time: every second derivative "
          f"({len(second)} forms, under jit too), vmap over params, jax_enable_x64 off, a missing "
          "or an extra name of params, a wrong shape or type of params, the positions, or the "
          "cell")

    # The Jacobian above jacobian_bytes: 8 K M bytes for the frames of a call.
    entries = 1 + SOLUTE
    small = mdir.jax.energy_fn(program, jacobian_bytes=8 * entries * 2)
    frames, cells = frames_of(state, 3)
    jf, jc = jnp.asarray(frames), jnp.asarray(cells)
    assert same(jax.grad(small)(params, x0, c0)["q"], jax.grad(f)(params, x0, c0)["q"])
    two = jax.vmap(jax.grad(small), in_axes=(None, 0, 0))(params, jf[:2], jc[:2])
    expect(mdir.UnsupportedError,
           lambda: jax.vmap(jax.grad(small), in_axes=(None, 0, 0))(params, jf, jc),
           "above jacobian_bytes")
    by_x = jax.vmap(jax.grad(small, argnums=1), in_axes=(None, 0, 0))(params, jf, jc)
    none = mdir.jax.energy_fn(program, jacobian_bytes=0)
    expect(mdir.UnsupportedError, lambda: jax.grad(none)(params, x0, c0), "above jacobian_bytes")
    assert same(jax.grad(none, argnums=1)(params, x0, c0), jax.grad(f, argnums=1)(params, x0, c0))
    assert two["q"].shape == (2, SOLUTE) and by_x.shape == jf.shape
    print(f"{mode}: a derivative in params whose Jacobian passes jacobian_bytes is refused, for "
          "one frame and for the frames of a vmap; the gradient in the positions is given")

    # Restraints to reference positions: no gradient in the cell.
    rsystem, rstate = dipeptide(restrained=True)
    rf = mdir.jax.energy_fn(compile_(rsystem, rstate))
    expect(mdir.UnsupportedError, lambda: jax.grad(rf, argnums=2)(params, x0, c0),
           "restraints to reference positions")
    assert jax.grad(rf, argnums=1)(params, x0, c0).shape == x0.shape
    print(f"{mode}: the gradient in the cell of a program with restraints to reference positions "
          "is refused; the gradient in the positions is given")

    # Errors that depend on data, naming the frame of the call: JAX raises
    # them as JaxRuntimeError where the result is awaited and as ValueError
    # where the call runs as it is made. The function gives the same bits
    # after them.
    data_error = (jax.errors.JaxRuntimeError, ValueError)
    bad = jf.at[2, 7, 1].set(jnp.nan)
    message = expect(data_error,
                     lambda: jax.vmap(f, in_axes=(None, 0, 0))(params, bad, jc).block_until_ready(),
                     "frame 2 of the call")
    assert "not finite" in message, message
    message = expect(data_error, lambda: f(params, x0, jnp.full(3, 1.0)).block_until_ready(),
                     "frame 0 of the call")
    assert "twice the cutoff" in message, message
    message = expect(data_error,
                     lambda: jax.jit(f)(params, x0, jnp.full(3, 1.0)).block_until_ready(),
                     "frame 0 of the call")
    assert same(f(params, x0, c0), u0)
    print(f"{mode}: a position that is not finite and a cell below twice the cutoff raise an "
          "error of JAX naming the frame of the call, eager and under jit; the function then "
          "gives the same energy, to the bit")

    # What else works: eval_shape, scan, checkpoint, a (3, 3) cell, the
    # program's cell.
    shape = jax.eval_shape(f, params, x0, c0)
    assert shape.shape == () and shape.dtype == jnp.float64
    scanned = jax.lax.scan(lambda carry, y: (carry, f(params, y[0], y[1])), 0.0, (jf, jc))[1]
    exact = same(scanned, jax.vmap(f, in_axes=(None, 0, 0))(params, jf, jc))
    checkpointed = jax.grad(jax.checkpoint(f), argnums=1)(params, x0, c0)
    exact = exact and same(checkpointed, jax.grad(f, argnums=1)(params, x0, c0))
    full = jax.grad(f, argnums=2)(params, x0, h0)
    exact = exact and same(f(params, x0, h0), u0) and same(jnp.diagonal(full),
                                                           jax.grad(f, argnums=2)(params, x0, c0))
    exact = exact and same(f(params, x0), u0) and same(f(params, x0, None), u0)
    print(f"{mode}: eval_shape, lax.scan, jax.checkpoint, a cell (3, 3) whose gradient has the "
          f"edges on its diagonal, and the program's cell: {exact}")
    assert exact

    # The identity of the program in a compiled call, and a released key.
    lowered = jax.jit(f).lower(params, x0, c0)
    text = lowered.as_text()
    p = f._mdir_program
    assert "mdir_frames" in text and f"program = {p.key}" in text and p.fingerprint in text, text
    other = mdir.jax.energy_fn(program)
    assert other._mdir_program.key != p.key
    assert other._mdir_program.fingerprint == p.fingerprint
    stale = jax.jit(other).lower(params, x0, c0)
    mdir._core._xla_ffi_release(other._mdir_program.key)
    expect(data_error + (jax.errors.JaxRuntimeError,),
           lambda: stale.compile()(params, x0, c0).block_until_ready(), "was released")
    assert same(lowered.compile()(params, x0, c0), u0)
    print(f"{mode}: a compiled call names the program by its key and its fingerprint; a key that "
          "was released is refused when the executable is instantiated")

    # The dtype table.
    x32 = x0.astype(jnp.float32)
    if double:
        expect(mdir.InputError, lambda: f(params, x32, c0), "give the positions in float64")
        print(f"{mode}: float32 positions are refused; float64 give a float64 gradient: "
              f"{jax.grad(f, argnums=1)(params, x0, c0).dtype == jnp.float64}")
        return
    expect(mdir.InputError, lambda: f(params, x32, c0), "give the positions in float64")
    loose = compile_(system, state, deterministic=False)
    lf = mdir.jax.energy_fn(loose)
    narrow = mdir.FrameEvaluator(loose).evaluate(np.asarray(x32)[None], np.asarray(c0)[None],
                                                 gradients=("positions",))
    g32 = jax.grad(lf, argnums=1)(params, x32, c0)
    g64 = jax.grad(lf, argnums=1)(params, x0, c0)
    wide = mdir.FrameEvaluator(loose).evaluate(np.asarray(x0)[None], np.asarray(c0)[None],
                                               gradients=("positions",))
    if target == "CPU":
        exact = (g32.dtype == jnp.float32 and same(g32, narrow.gradients["positions"][0]) and
                 g64.dtype == jnp.float64 and
                 same(g64, wide.gradients["positions"][0].astype(np.float64)) and
                 same(lf(params, x32, c0), narrow.energy[0]))
        print(f"{mode}: deterministic: float32 positions refused; not deterministic (forces in "
              f"f32): float32 positions give the f32 forces of the program and float64 positions "
              f"those forces widened, to the bit: {exact}")
        assert exact
        return
    # Without the deterministic mode, the forces of a program of the GPU
    # are summed in an order that changes from one evaluation to the next
    # (bitwise determinism is the opt-in mode, D201): measured on the RTX
    # 5090, 1.4e-7 of the largest entry between two evaluations of one
    # evaluator. The types are compared, and the values at the tolerance
    # of the evaluator's own runs without the mode against each other
    # (docs/python-frames.md, Validation of the gradients: 1e-5).
    def close(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    gap = max(close(g32, narrow.gradients["positions"][0]),
              close(g64, wide.gradients["positions"][0]))
    shift = abs(float(lf(params, x32, c0)) - float(narrow.energy[0]))
    exact = (g32.dtype == jnp.float32 and g64.dtype == jnp.float64 and gap <= 1e-5 and
             shift <= 5e-3)
    print(f"{mode}: deterministic: float32 positions refused; not deterministic (forces in f32): "
          f"float32 positions give f32 forces and float64 positions f64 ones, against the "
          f"evaluator {gap:.1e} of the largest entry (tolerance 1e-5) and {shift:.1e} kJ/mol "
          f"(tolerance 5e-3): {exact}")
    assert exact
    del lf, f, other
    gc.collect()


# --- torch ----------------------------------------------------------------------

def torch_frames():
    system, state = dipeptide()
    frames, cells = frames_of(state, 3, seed=8)
    wsystem, wstate, wx, wh = water()
    return (system, state, frames, cells), (wsystem, wstate, wx, wh)


def run_torch_dump(path):
    import torch
    import mdir.torch
    (system, state, frames, cells), (wsystem, wstate, wx, wh) = torch_frames()
    weights = np.array([0.7, -1.3, 2.1])
    evaluator = mdir.FrameEvaluator(compile_(system, state))
    theta = {n: torch.tensor(evaluator.tunables[n], requires_grad=True) for n in ("sigma", "q")}
    tx, tc = torch.tensor(frames, requires_grad=True), torch.tensor(cells, requires_grad=True)
    out = mdir.torch.evaluate(evaluator, theta, tx, tc)
    gx, gc_, gs, gq = torch.autograd.grad((torch.tensor(weights) * out.energy).sum(),
                                          (tx, tc, theta["sigma"], theta["q"]))
    out = mdir.torch.evaluate(evaluator, theta, frames, cells)
    gs1, gq1 = torch.autograd.grad(out.energy.sum(), (theta["sigma"], theta["q"]))
    wevaluator = mdir.FrameEvaluator(compile_(wsystem, wstate))
    wtheta = {"sigma": torch.tensor(wevaluator.tunables["sigma"], requires_grad=True)}
    td = torch.tensor(np.diag(wh).copy(), requires_grad=True)
    tt = torch.tensor(np.array([wh[1, 0], wh[2, 0], wh[2, 1]]), requires_grad=True)
    wout = mdir.torch.evaluate(wevaluator, wtheta, torch.tensor(wx), td, tt)
    wd, wt, ws = torch.autograd.grad(wout.energy.sum(), (td, tt, wtheta["sigma"]))
    np.savez(path, energy=out.energy.detach().numpy(), gx=gx.numpy(), gc=gc_.numpy(),
             gs=gs.numpy(), gq=gq.numpy(), gs1=gs1.numpy(), gq1=gq1.numpy(), wenergy=wout.energy.detach().numpy(), wd=wd.numpy(),
             wt=wt.numpy(), ws=ws.numpy())
    print(f"{mode}: mdir.torch.evaluate written")


def run_torch_compare(path):
    reference = np.load(path)
    (system, state, frames, cells), (wsystem, wstate, wx, wh) = torch_frames()
    weights = jnp.asarray([0.7, -1.3, 2.1])
    program = compile_(system, state)
    f, params = mdir.jax.energy_fn(program), mdir.jax.params(program)
    batched = jax.vmap(f, in_axes=(None, 0, 0))
    energies = batched(params, jnp.asarray(frames), jnp.asarray(cells))
    gp, gx, gc_ = jax.grad(lambda p, x, c: jnp.sum(weights * batched(p, x, c)), argnums=(0, 1, 2))(
        params, jnp.asarray(frames), jnp.asarray(cells))
    ones = jax.grad(lambda p: jnp.sum(batched(p, jnp.asarray(frames), jnp.asarray(cells))))(params)
    wprogram = compile_(wsystem, wstate)
    wf, wparams = mdir.jax.energy_fn(wprogram), mdir.jax.params(wprogram)
    wu, (wp, wg) = jax.value_and_grad(wf, argnums=(0, 2))(wparams, jnp.asarray(wx),
                                                          jnp.asarray(wh))
    checks = {
        "the energies": same(energies, reference["energy"]),
        "the gradient in the positions": same(gx, reference["gx"]),
        "the gradient in the edges of the cells": same(gc_, reference["gc"]),
        "the gradient in params under a cotangent of ones (the VJP)":
        same(ones["sigma"], reference["gs1"]) and same(ones["q"], reference["gq1"]),
        "the energy in a triclinic cell": same(wu, reference["wenergy"][0]),
        "the gradient in its diagonal and its tilts": same(jnp.diagonal(wg), reference["wd"]) and
        same(jnp.stack([wg[1, 0], wg[2, 0], wg[2, 1]]), reference["wt"]),
        "its gradient in params": same(wp["sigma"], reference["ws"]),
    }
    for label, exact in checks.items():
        print(f"{mode}: against mdir.torch.evaluate, {label}, to the bit: {exact}")
    # Under a cotangent of weights the VJP is a sum over the frames of the
    # weights times the rows, which PyTorch forms as a product of a vector
    # with a matrix and XLA as the transpose of the contraction of each
    # frame: the order of the sum and its fused multiplications differ.
    weighted = max(float(np.max(np.abs(np.asarray(gp[n]) - reference[k]) / np.abs(reference[k])))
                   for n, k in (("sigma", "gs"), ("q", "gq")))
    print(f"{mode}: against mdir.torch.evaluate, the gradient in params under a cotangent of "
          f"weights: largest relative difference {weighted:.1e} (tolerance 1e-14)")
    assert weighted <= 1e-14
    assert all(checks.values())

# --- the device -----------------------------------------------------------------

def on_cpu(tree):
    return jax.device_put(tree, jax.devices("cpu")[0])


def run_device():
    assert target == "GPU"
    system, state = dipeptide()
    frames, cells = frames_of(state, 3)
    gpu, cpu = jax.devices("gpu")[0], jax.devices("cpu")[0]

    # The handler of CUDA against that of the CPU, on one program of each
    # target: the same loop of frames, so the same bits.
    for kind in ("GPU", "CPU"):
        program = compile_(system, state, target_=kind)
        f, params = mdir.jax.energy_fn(program), mdir.jax.params(program)
        batched = jax.vmap(f, in_axes=(None, 0, 0))
        everything = jax.jit(jax.vmap(jax.value_and_grad(f, argnums=(0, 1, 2)),
                                      in_axes=(None, 0, 0)))
        rows = jax.jit(jax.jacrev(batched))
        jx, jc = jnp.asarray(frames), jnp.asarray(cells)
        on_device = leaves(everything(params, jx, jc)) + leaves(rows(params, jx, jc))
        on_host = leaves(everything(on_cpu(params), on_cpu(jx), on_cpu(jc))) + leaves(
            rows(on_cpu(params), on_cpu(jx), on_cpu(jc)))
        placed = (everything(params, jx, jc)[0].devices() == {gpu} and
                  everything(on_cpu(params), on_cpu(jx), on_cpu(jc))[0].devices() == {cpu})
        exact = placed and all(same(a, b) for a, b in zip(on_device, on_host))
        print(f"{mode}: a program of the {kind}: the energies, the gradients in params, the "
              f"positions, and the cell, and the rows, the handler of CUDA against that of the "
              f"CPU, to the bit: {exact}")
        assert exact

    # The order on XLA's stream. Each call is preceded by a JAX op on the
    # device that writes its positions and followed by ops that read its
    # energies and gradients, all dispatched without a wait, eager and in
    # one compiled function; against the same ops on the results of calls
    # that were waited for.
    program = compile_(system, state)
    f, params = mdir.jax.energy_fn(program), mdir.jax.params(program)
    batched = jax.vmap(f, in_axes=(None, 0, 0))
    gradient = jax.vmap(jax.grad(f, argnums=1), in_axes=(None, 0, 0))
    jx, jc = jnp.asarray(frames), jnp.asarray(cells)

    @jax.jit
    def produce(x, t):
        return x + t * jnp.sin(jnp.arange(x.size, dtype=x.dtype)).reshape(x.shape) * 1e-3

    @jax.jit
    def consume(e, g):
        return jnp.cumsum(e) * 2.0 - e[::-1] + jnp.sum(g * g, axis=(1, 2))

    @jax.jit
    def chain(p, x, c, t):
        y = produce(x, t)
        return consume(batched(p, y, c), gradient(p, y, c))

    steps = [float(t) for t in np.linspace(0.0, 1.0, 8)]
    eager = [consume(batched(params, produce(jx, t), jc), gradient(params, produce(jx, t), jc))
             for t in steps]
    fused = [chain(params, jx, jc, t) for t in steps]
    reference = []
    for t in steps:
        y = produce(jx, t).block_until_ready()
        e = batched(params, y, jc).block_until_ready()
        g = gradient(params, y, jc).block_until_ready()
        e, g = jax.device_put(np.asarray(e), gpu), jax.device_put(np.asarray(g), gpu)
        reference.append(np.asarray(consume(e, g).block_until_ready()))
    exact = (all(same(a, b) for a, b in zip(eager, reference)) and
             all(same(a, b) for a, b in zip(fused, reference)))
    print(f"{mode}: on XLA's stream, {len(steps)} calls between JAX ops that write their positions "
          f"and read their energies and gradients, with no wait, eager and under one jit, against "
          f"synchronized calls, to the bit: {exact}")
    assert exact

    # A call on another device than the program's. With one device the
    # runtime's device is named by MDRT_DEVICE as the handler reads it at
    # each call; the runtime keeps the device it took when it began.
    import os
    before = os.environ.get("MDRT_DEVICE")
    os.environ["MDRT_DEVICE"] = "1"
    try:
        message = expect((jax.errors.JaxRuntimeError, ValueError),
                         lambda: jax.jit(f)(params, jx[0], jc[0]).block_until_ready(),
                         "the call is on device 0, and the program runs on device 1")
    finally:
        if before is None:
            del os.environ["MDRT_DEVICE"]
        else:
            os.environ["MDRT_DEVICE"] = before
    exact = same(jax.jit(f)(params, jx[0], jc[0]), batched(params, jx, jc)[0])
    print(f"{mode}: a call of a program of the GPU on another device than its runtime's is "
          f"refused, naming both; the next call gives the same energy, to the bit: {exact}")
    assert exact


def run_precisions():
    assert target == "GPU"
    system, state = dipeptide()
    frames, cells = frames_of(state, 5)
    jx, jc = jnp.asarray(frames), jnp.asarray(cells)
    results = {}
    for label, precision_, deterministic in (("double", "Double", True),
                                              ("mixed, deterministic", "Mixed", True),
                                              ("mixed", "Mixed", False)):
        program = compile_(system, state, precision_=precision_, deterministic=deterministic)
        f, params = mdir.jax.energy_fn(program), mdir.jax.params(program)
        e, (gp, gx, gc_) = jax.jit(jax.vmap(jax.value_and_grad(f, argnums=(0, 1, 2)),
                                            in_axes=(None, 0, 0)))(params, jx, jc)
        results[label] = (np.asarray(e), np.asarray(gx).astype(np.float64), np.asarray(gc_),
                          np.concatenate([np.asarray(gp[n]) for n in params], axis=1))
    energy, by_x, by_cell, by_theta = results["double"]

    def of_largest(a, b):
        axes = tuple(range(1, a.ndim))
        return float(np.max(np.max(np.abs(a - b), axis=axes) / np.max(np.abs(b), axis=axes)))

    for label in ("mixed, deterministic", "mixed"):
        e, gx, gc_, gp = results[label]
        found = (float(np.max(np.abs(e - energy))), of_largest(gx, by_x), of_largest(gc_, by_cell),
                 of_largest(gp, by_theta))
        print(f"{mode}: {label} against double, 5 frames: the energy {found[0]:.1e} kJ/mol "
              f"(tolerance 5e-3); the gradient in the positions {found[1]:.1e} and in the edges "
              f"{found[2]:.1e} of the largest entry of the frame (tolerance 1e-4); in params "
              f"{found[3]:.1e} of the largest entry (tolerance 1e-6)")
        assert found[0] <= 5e-3 and found[1] <= 1e-4 and found[2] <= 1e-4 and found[3] <= 1e-6, found


def run_sanitize():
    assert target == "GPU"
    system, state = dipeptide()
    frames, cells = frames_of(state, 2)
    program = compile_(system, state)
    f, params = mdir.jax.energy_fn(program), mdir.jax.params(program)
    jx, jc = jnp.asarray(frames), jnp.asarray(cells)
    batched = jax.vmap(f, in_axes=(None, 0, 0))
    energies = batched(params, jx, jc)
    grads = jax.jit(jax.vmap(jax.grad(f, argnums=(0, 1, 2)), in_axes=(None, 0, 0)))(params, jx, jc)
    single = f(params, jx[0])
    moved = jax.grad(f)(dict(params, sigma=params["sigma"] * 1.001), jx[0], jc[0])
    expect((jax.errors.JaxRuntimeError, ValueError),
           lambda: batched(params, jx.at[1, 3, 0].set(jnp.nan), jc).block_until_ready(),
           "frame 1 of the call")
    exact = same(batched(params, jx, jc), energies)
    jax.block_until_ready((grads, single, moved))
    print(f"{mode}: sanitize: energies, gradients, the program's cell, new values of params, a "
          f"frame that fails, and the same energies after it: {exact}")
    assert exact


if scenario == "gate":
    run_gate()
elif scenario == "refusals":
    run_refusals()
elif scenario == "torch-dump":
    run_torch_dump(sys.argv[5])
elif scenario == "torch-compare":
    run_torch_compare(sys.argv[5])
elif scenario == "device":
    run_device()
elif scenario == "precisions":
    run_precisions()
elif scenario == "sanitize":
    run_sanitize()
else:
    raise SystemExit(f"unknown scenario {scenario}")
print("jax passed" if not torch_side else "torch passed")
