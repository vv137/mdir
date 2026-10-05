"""Validate host interchange independently of compiler lowering."""
import array
import gc
import pathlib
import sys
import numpy as np
import mdir

root = pathlib.Path(sys.argv[1])
loaded = mdir.load_amber(str(root / "dipeptide.prmtop"), str(root / "dipeptide.inpcrd"))
state, system = loaded.make_state(), loaded.make_system()
n = system.particle_count


def rejects(call, property_name, reason):
    try:
        call()
    except mdir.InputError as exc:
        text = str(exc)
        assert property_name in text and reason in text, text
    else:
        raise AssertionError("expected InputError")


def readonly(a, shape, dtype):
    assert type(a) is np.ndarray
    assert a.shape == shape and a.dtype == dtype
    assert a.flags.c_contiguous and not a.flags.writeable
    if a.size:
        try:
            a.flat[0] = 123
        except ValueError:
            pass
        else:
            raise AssertionError("returned array is writable")


positions = state.positions
readonly(positions, (n, 3), np.float64)
readonly(state.velocities, (0, 3), np.float64)
readonly(state.cell.vectors, (3, 3), np.float64)
readonly(state.cell.diagonal, (3,), np.float64)
readonly(state.cell.tilt, (3,), np.float64)
assert not np.shares_memory(positions, state.positions)
positions.setflags(write=True)
positions[0, 0] += 1
assert positions[0, 0] != state.positions[0, 0]
valid = state.positions.copy()
state.positions = memoryview(valid)
assert state.positions.tobytes() == valid.tobytes()
valid[0, 0] += 3
assert state.positions[0, 0] != valid[0, 0]
# Contiguous buffers may be unaligned; copying must not dereference them.
unaligned = np.ndarray((n, 3), dtype=np.float64, buffer=bytearray(n * 24 + 1), offset=1)
unaligned[:] = state.positions
state.positions = unaligned
assert state.positions.tobytes() == unaligned.tobytes()
bytes_before = state.positions.tobytes()
for value, reason in [
    (np.zeros(n * 3), "shape"), (np.zeros((n, 2)), "shape"),
    (np.zeros((n + 1, 3)), "shape"),
    (np.zeros((n, 3), dtype=np.float32), "float64"),
    (np.zeros((n, 3), dtype=np.int64), "float64"),
    (np.zeros((n, 3), dtype=object), "float64"),
    (np.zeros((n, 3), dtype=np.complex128), "float64"),
    (np.zeros((n, 3), dtype=">f8"), "float64"),
    (np.zeros((n, 6))[:, ::2], "C-contiguous"),
    (np.asfortranarray(np.zeros((n, 3))), "C-contiguous"),
    (np.full((n, 3), np.nan), "nonfinite"),
    (np.full((n, 3), np.inf), "nonfinite"),
    ([[0.0, 0.0, 0.0]] * n, "lists"),
]:
    rejects(lambda: setattr(state, "positions", value), "positions", reason)
    assert state.positions.tobytes() == bytes_before
rejects(lambda: setattr(state, "velocities", np.zeros((n - 1, 3))), "velocities", "shape")
state.velocities = np.zeros((n, 3))
state.velocities = np.empty((0, 3))


class Tensor:
    """CPU DLPack protocol with no buffer export or framework dependency."""
    def __init__(self, a):
        self.a = a
    def __dlpack_device__(self):
        return self.a.__dlpack_device__()
    def __dlpack__(self, stream=None):
        return self.a.__dlpack__(stream=stream)


values = state.positions.copy()
state.positions = Tensor(values)
assert state.positions.tobytes() == values.tobytes()
values[0, 0] += 2
assert state.positions[0, 0] != values[0, 0]
rejects(lambda: setattr(state, "positions", Tensor(np.zeros((n, 3), dtype=np.float32))), "positions", "float64")
rejects(lambda: setattr(state, "positions", Tensor(np.zeros((n, 6))[:, ::2])), "positions", "C-contiguous")


class DeviceTensor:
    def __dlpack_device__(self):
        return (2, 0)  # CUDA device type metadata; never contacts a device.
    def __dlpack__(self, stream=None):
        raise AssertionError("must reject a non-CPU device before export")


rejects(lambda: setattr(state, "positions", DeviceTensor()), "positions", "CPU DLPack")
cell = state.cell
rejects(lambda: setattr(cell, "diagonal", np.zeros(2)), "Cell.diagonal", "shape")
rejects(lambda: setattr(cell, "tilt", np.full(3, np.nan)), "Cell.tilt", "nonfinite")
cell.diagonal = array.array("d", [3.2] * 3)
assert cell.diagonal.tobytes() == np.full(3, 3.2).tobytes()

term = mdir.TupleTerm()
term.particles = np.array([[0, 1], [2, 3]], dtype=np.int32)
readonly(term.particles, (2, 2), np.int64)
term.parameters = [("k", array.array("d", [100, 200]))]
readonly(term.parameters[0][1], (2,), np.float64)
rejects(lambda: setattr(term, "particles", np.array([0, 1], dtype=np.int64)), "particles", "shape")
rejects(lambda: setattr(term, "particles", np.array([[-1, 1]], dtype=np.int64)), "particles", "negative")
rejects(lambda: setattr(term, "particles", np.array([[2**32, 1]], dtype=np.int64)), "particles", "range")
rejects(lambda: setattr(term, "particles", np.ones((2, 2))), "particles", "int32 or int64")
rejects(lambda: setattr(term, "particles", np.ones((2, 4), dtype=np.int64)[:, ::2]), "particles", "C-contiguous")
rejects(lambda: setattr(term, "parameters", [("k", np.zeros(1))]), "parameters", "shape")
rejects(lambda: setattr(term, "parameters", [("k", np.zeros((2, 1)))]), "parameters", "shape")
rejects(lambda: setattr(term, "parameters", [("k", np.full(2, np.inf))]), "parameters", "nonfinite")
rejects(lambda: setattr(term, "parameters", [("k", np.zeros(2, dtype=np.float32))]), "parameters", "float64")
assert term.parameters[0][1].tobytes() == np.array([100., 200.]).tobytes()
# Snapshots survive owners and empty arrays keep their schema.
snapshot, empty, vectors, indices = state.positions, state.velocities, state.cell.vectors, term.particles
fresh = mdir.InitialState()
fresh.positions = np.zeros((4, 3))
rejects(lambda: setattr(fresh, "positions", np.zeros((3, 3))), "positions", "shape")
del state, term, loaded, fresh
gc.collect()
readonly(snapshot, (n, 3), np.float64)
readonly(empty, (0, 3), np.float64)
readonly(vectors, (3, 3), np.float64)
readonly(indices, (2, 2), np.int64)
print("host arrays: shapes, dtype, strides, buffer/DLPack, ownership and atomic versions passed")
