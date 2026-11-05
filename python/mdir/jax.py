"""The frame evaluator as a pure function of JAX (docs/python-frames.md,
D371).

`energy_fn(program)` returns `energy_fn(params, positions, cell)`, the
potential energy that the forces sample (D210) of one frame in kJ/mol, a
pure function of the values of the tunables `params` (a dict of every
tunable of the program, as `params(program)` gives it), the positions,
and the cell. It works under `jax.jit`, `jax.grad` (in all three
arguments), `jax.jvp`, and `jax.vmap` over frames, which evaluates the
frames of a batch in one call. A second derivative raises
`UnsupportedError` at trace time; it is not zero.

The function is a `jax.custom_jvp` whose tangent is built from the outputs
of one primitive, `mdir_frames`, lowered through `jax.ffi` to the handler of
the extension (`python/XlaFfi.cpp`), which evaluates the frames on the
evaluator's existing loop (D251). Importing this module imports jax and
registers the handler; `import mdir` does not.
"""
import functools
import threading
import weakref

import numpy as np
import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

from . import _core, _frames

__all__ = ["energy_fn", "params"]

#: The lowest JAX that the adapter is tested with (Ruling 20, Q9).
_LOWEST = (0, 11, 2)


def _version(text):
    parts = []
    for part in text.split(".")[:3]:
        digits = "".join(c for c in part if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


if _version(jax.__version__) < _LOWEST:
    raise ImportError(f"mdir.jax needs JAX {'.'.join(map(str, _LOWEST))} or later; found "
                      f"{jax.__version__}")


_REGISTERED = []
_REGISTERING = threading.Lock()


def _register():
    """Registers the type of the state of an executable and the handlers of
    the targets with XLA, once, when the first function is made: for the
    CPU, and for CUDA where MDIR was built with CUDA and JAX has a CUDA
    device. XLA checks the version of the FFI of the handlers when they are
    registered, and the handlers check it again at each instantiation
    (`python/XlaFfi.cpp`). The client of a platform is made first: XLA takes
    a type only once it is, and refuses a handler whose state has a type
    that it does not know."""
    with _REGISTERING:
        if _REGISTERED:
            return
        platforms = ["cpu"]
        if "CUDA" in _core._xla_ffi_platforms():
            try:
                jax.devices("cuda")
                platforms.append("CUDA")
            except RuntimeError:
                pass  # no CUDA backend in this process (JAX_PLATFORMS=cpu)
        for platform in platforms:
            jax.devices(platform.lower())
            for name, registration in _core._xla_ffi_types(platform).items():
                jax.ffi.register_ffi_type(name, registration, platform=platform)
            for name, handlers in _core._xla_ffi_targets(platform).items():
                jax.ffi.register_ffi_target(name, handlers, platform=platform, api_version=1)
        _REGISTERED.append(True)

_SECOND = ("a second derivative of the energy of the frames is not implemented: the "
           "derivative of its gradient was asked for (jax.hessian, jax.jacfwd or jax.jvp of "
           "jax.grad, or jax.grad of a function of jax.grad). It is not zero")

# The primitive: the frames (K, N, 3) and their cells (K, 6) or (K, 0) at
# the values theta (M,), giving the energies (K,), the gradient in the
# positions (K, N, 3) in the type of the forces of the program, the
# gradient in the cell vectors (K, 3, 3), and the rows of the Jacobian in
# theta (K, M); an output that is not asked has no rows.
_frames_p = Primitive("mdir_frames")
_frames_p.multiple_results = True


def _shapes(theta, positions, cells, *, forces, cell, rows, force_type, **_):
    count, particles = positions.shape[0], positions.shape[1]
    return (jax.core.ShapedArray((count,), jnp.float64),
            jax.core.ShapedArray((count if forces else 0, particles, 3), force_type),
            jax.core.ShapedArray((count if cell else 0, 3, 3), jnp.float64),
            jax.core.ShapedArray((count if rows else 0, theta.shape[0]), jnp.float64))


_frames_p.def_abstract_eval(_shapes)


@functools.lru_cache(maxsize=None)
def _compiled(params):
    return jax.jit(functools.partial(_frames_p.bind, **dict(params)))


def _impl(*args, **params):
    # Outside a trace the primitive runs as a compiled call of its own.
    return _compiled(tuple(sorted(params.items())))(*args)


_frames_p.def_impl(_impl)

_lower = jax.ffi.ffi_lowering("mdir_frames")


def _lowering(ctx, theta, positions, cells, *, program, fingerprint, **_):
    # The program's identity is static: its key in the process and its
    # fingerprint, which the instantiation of an executable checks.
    return _lower(ctx, theta, positions, cells, program=np.int64(program),
                  fingerprint=fingerprint)


mlir.register_lowering(_frames_p, _lowering, platform="cpu")


def _lowering_cuda(ctx, *args, **params):
    if "CUDA" not in _core._xla_ffi_platforms():
        raise _core.UnsupportedError(
            "mdir.jax: this build of MDIR has no handler for CUDA (it was built without CUDA); "
            "place the arrays of the call on the CPU, or run JAX with JAX_PLATFORMS=cpu")
    return _lowering(ctx, *args, **params)


mlir.register_lowering(_frames_p, _lowering_cuda, platform="cuda")


def _refuse_jvp(primals, tangents, **params):
    # Every factor of the tangent of the energy is an output of this
    # primitive, so any second derivative reaches this rule.
    raise _core.UnsupportedError(_SECOND)


ad.primitive_jvps[_frames_p] = _refuse_jvp


def _batch(args, dims, **params):
    """Folds the mapped axis of the positions and the cells into the axis
    of the frames and binds once: one call for the frames of the batch."""
    theta, positions, cells = args
    theta_dim, positions_dim, cells_dim = dims
    if theta_dim is not None:
        raise NotImplementedError(
            "mdir.jax: jax.vmap over params is not supported: a call evaluates its frames at "
            "one value of the tunables; map the positions and the cells, or loop over the "
            "values of params")
    size = (positions.shape[positions_dim] if positions_dim is not None
            else cells.shape[cells_dim])

    def folded(value, dim):
        if dim is None:
            value = jnp.broadcast_to(value, (size,) + value.shape)
        else:
            value = jnp.moveaxis(value, dim, 0)
        return value.reshape((size * value.shape[1],) + value.shape[2:])

    positions, cells = folded(positions, positions_dim), folded(cells, cells_dim)
    if params["rows"]:
        _check_rows(positions.shape[0], theta.shape[0], params["jacobian_bytes"])
    outputs = _frames_p.bind(theta, positions, cells, **params)
    unfolded = [o.reshape((size, o.shape[0] // size if o.shape[0] else 0) + o.shape[1:])
                for o in outputs]
    return unfolded, [0] * len(unfolded)


batching.primitive_batchers[_frames_p] = _batch


def _check_rows(count, entries, jacobian_bytes):
    if 8 * count * entries > jacobian_bytes:
        raise _core.UnsupportedError(
            f"mdir.jax: the Jacobian of {count} frames in {entries} entries of the tunables is "
            f"{8 * count * entries} bytes, above jacobian_bytes ({jacobian_bytes}); the second "
            "pass that would evaluate the frames again is not implemented yet. Give a larger "
            "jacobian_bytes to energy_fn, or fewer frames per call")


def _zero(tangent):
    return isinstance(tangent, SymbolicZero)


def params(program):
    """The values of the tunables of `program` as the pytree that the
    function of `energy_fn` takes: a dict of every name of
    `program.plan["tunables"]` to a float64 array (M,), in MD units."""
    _check_x64()
    values = program._tunable_values()
    return {d["name"]: jnp.asarray(values[d["name"]], dtype=jnp.float64)
            for d in program.plan.get("tunables", [])}


def _check_x64():
    if not jax.config.jax_enable_x64:
        raise _core.InputError(
            "mdir.jax needs jax_enable_x64: the tunables, the cell, and the energy are float64 "
            "(jax.config.update('jax_enable_x64', True) before any array is made)")


def _float64(value, name):
    if value.dtype != jnp.float64:
        raise _core.InputError(f"mdir.jax: {name} is float64; found {value.dtype}")


class _Program:
    """What a function of `energy_fn` knows of its program: the private
    evaluator, which owns the simulation, and the static attributes of the
    calls."""

    def __init__(self, program, jacobian_bytes):
        self.evaluator = _frames.FrameEvaluator(program, jacobian_bytes=jacobian_bytes)
        registered = self.evaluator._simulation._xla_ffi_register()
        self.key, self.fingerprint = registered["key"], registered["fingerprint"]
        self.periodic, self.restrained = registered["periodic"], registered["restrained"]
        weakref.finalize(self, _core._xla_ffi_release, self.key)
        self.names = list(self.evaluator._slices)
        self.sizes = [end - begin for begin, end in self.evaluator._slices.values()]
        self.particles = self.evaluator._particles
        self.force_type = jnp.dtype(self.evaluator._force_type)
        self.jacobian_bytes = int(jacobian_bytes)
        edges, tilt = self.evaluator._edges, self.evaluator._tilt
        self.cell = None if edges is None else np.concatenate([edges, tilt])

    def bind(self, theta, positions, cells, forces=False, cell=False, rows=False):
        return _frames_p.bind(theta, positions, cells, program=self.key,
                              fingerprint=self.fingerprint, forces=bool(forces),
                              cell=bool(cell), rows=bool(rows), force_type=self.force_type,
                              jacobian_bytes=self.jacobian_bytes)


def _cell_rows(cell):
    """The six numbers a_x, b_y, c_z, b_x, c_x, c_y of the cell of a frame
    given as its three edges (3,) or its vectors in rows (3, 3), as the
    handler takes them (D238)."""
    if cell.shape == (3,):
        return jnp.concatenate([cell, jnp.zeros(3, cell.dtype)])
    return jnp.stack([cell[0, 0], cell[1, 1], cell[2, 2], cell[1, 0], cell[2, 0], cell[2, 1]])


def energy_fn(program, *, jacobian_bytes=1 << 28):
    """A pure function `energy_fn(params, positions, cell=None)` of one
    frame: the potential energy that the forces sample (D210), a float64
    scalar in kJ/mol.

    `program` is compiled with tunables and `System.tunable_gradient`, as
    for `mdir.FrameEvaluator`; the function evaluates it in a simulation of
    its own, which nothing else sees. `params` is a dict of the names of
    every tunable of the program to float64 arrays (M,) (`params(program)`
    gives the program's values); `positions` (N, 3) in nm, in the order of
    the input; `cell` the three edges (3,) of an orthorhombic cell or the
    cell vectors in rows (3, 3), lower triangular, float64 in nm, or None
    for the cell of the program's initial state (and for a program without
    a periodic cell). The forces are `-jax.grad(f, argnums=1)`. The
    Jacobian in the tunables, 8 K M bytes for K frames of a call, is kept
    up to `jacobian_bytes`; above it a derivative in `params` raises. See
    docs/python-frames.md."""
    _check_x64()
    _register()
    if int(jacobian_bytes) < 0:
        raise _core.InputError("mdir.jax.energy_fn: jacobian_bytes is not negative")
    p = _Program(program, jacobian_bytes)
    if p.evaluator.terms != "all":
        raise _core.InputError("mdir.jax.energy_fn: the program's energy is that of all terms")

    def cells_of(cell):
        if cell is None:
            if p.cell is None:
                return jnp.zeros((1, 0), jnp.float64)
            return jnp.asarray(p.cell, dtype=jnp.float64)[None]
        return _cell_rows(cell)[None]

    @jax.custom_jvp
    def energy(theta, positions, cell):
        energies = p.bind(jnp.concatenate([theta[n] for n in p.names]), positions[None],
                          cells_of(cell))[0]
        return energies[0]

    def energy_jvp(primals, tangents):
        theta, positions, cell = primals
        dtheta, dpositions, dcell = tangents
        wants_positions = not _zero(dpositions)
        wants_cell = cell is not None and not _zero(dcell)
        wants_theta = any(not _zero(dtheta[n]) for n in p.names)
        if wants_cell and p.restrained:
            raise _core.UnsupportedError(
                "mdir.jax: the gradient in the cell is not given for a program with restraints "
                "to reference positions: their virial is taken about the references, which is "
                "not the derivative of the energy of a frame in the strain (D252); the gradient "
                "in the positions is given")
        if wants_theta:
            _check_rows(1, sum(p.sizes), p.jacobian_bytes)
        flat = jnp.concatenate([theta[n] for n in p.names])
        energies, gradient, by_cell, rows = p.bind(
            flat, positions[None], cells_of(cell), forces=wants_positions, cell=wants_cell,
            rows=wants_theta)
        tangent = jnp.zeros((), jnp.float64)
        # Each term is a contraction of an output of the primitive with a
        # tangent, so that its transpose is a product with that output.
        if wants_positions:
            tangent = tangent + jnp.sum(gradient[0].astype(jnp.float64) * dpositions)
        if wants_cell:
            factor = jnp.diagonal(by_cell[0]) if cell.shape == (3,) else by_cell[0]
            tangent = tangent + jnp.sum(factor * dcell)
        if wants_theta:
            given = jnp.concatenate([
                jnp.zeros(theta[n].shape, jnp.float64) if _zero(dtheta[n]) else dtheta[n]
                for n in p.names])
            tangent = tangent + rows[0] @ given
        return energies[0], tangent

    energy.defjvp(energy_jvp, symbolic_zeros=True)

    def f(params, positions, cell=None):
        _check_x64()
        if not isinstance(params, dict):
            raise _core.InputError("mdir.jax: params is a dict of the names of the tunables to "
                                   f"float64 arrays; found {type(params).__name__}")
        missing = [n for n in p.names if n not in params]
        extra = [n for n in params if n not in p.names]
        if missing or extra:
            raise _core.InputError(
                "mdir.jax: params names every tunable of the program, "
                f"{p.names}" + (f"; missing {missing}" if missing else "") +
                (f"; not tunables of the program {extra}" if extra else ""))
        theta = {}
        for name, size in zip(p.names, p.sizes):
            value = jnp.asarray(params[name])
            _float64(value, f"params['{name}']")
            if value.shape != (size,):
                raise _core.InputError(f"mdir.jax: params['{name}'] has shape ({size},); found "
                                       f"{value.shape}")
            theta[name] = value
        x = jnp.asarray(positions)
        if x.shape != (p.particles, 3):
            raise _core.InputError(f"mdir.jax: expected positions of shape ({p.particles}, 3) in "
                                   f"nm, in the order of the input; found {x.shape}")
        if x.dtype not in (jnp.float64, jnp.float32):
            raise _core.InputError(f"mdir.jax: positions are float64 or float32; found {x.dtype}")
        if x.dtype == jnp.float32 and p.force_type == jnp.float64:
            raise _core.InputError(
                "mdir.jax: the program gives its forces in float64 (double precision, or mixed "
                "and deterministic), so that the gradient in float32 positions would be "
                "rounded: give the positions in float64")
        if cell is not None:
            if not p.periodic:
                raise _core.InputError("mdir.jax: the program has no periodic cell, and the "
                                       "frame gives one")
            cell = jnp.asarray(cell)
            _float64(cell, "cell")
            if cell.shape not in ((3,), (3, 3)):
                raise _core.InputError("mdir.jax: the cell of a frame is its three edges, "
                                       f"shape (3,), or its vectors in rows, shape (3, 3); "
                                       f"found {cell.shape}")
        # Float32 positions are widened exactly; their cotangent is
        # rounded back to float32 by the transpose of the conversion.
        return energy(theta, x.astype(jnp.float64), cell)

    f._mdir_program = p
    return f
