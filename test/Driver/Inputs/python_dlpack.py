"""Read-only DLPack views of the buffers of a simulation (D220,
docs/python-dlpack.md), checked by consumers that do not use MDIR's code.

Usage: python_dlpack.py ROOT TARGET SCENARIO

Scenarios:
  numpy     (CPU) NumPy's from_dlpack: pointer sharing, read-only arrays,
            dtypes, IDs and values against state(), the values of
            tunables, leases that block runs, evaluations, and updates,
            capsules released exactly once (taken or not, versioned or
            legacy), refusals, and a tensor that outlives its simulation
  ctypes    (GPU) a reader of the capsule in ctypes and the CUDA driver API:
            the managed tensor (version, read-only flag, shape, strides,
            dtype, device), the device of the pointer, the values copied by
            the driver on a stream of its own that the producer hands off
            to, the deleter called without the GIL exactly once, the
            streams that are refused, and a tensor that outlives its
            simulation while another runs
  torch     (CPU and GPU) PyTorch: pointer sharing, dtypes, IDs and values,
            a consumer stream other than the default, consumer work still
            running when the aliases are deleted and the simulation runs,
            retained aliases after the simulation is deleted, and the
            allocator's reuse of the blocks once they are released
  sanitize  (GPU) a short life of views for compute-sanitizer
"""
import ctypes
import gc
import sys

import numpy as np
import mdir

root, target_name, scenario = sys.argv[1], sys.argv[2], sys.argv[3]
target = getattr(mdir.Target, target_name)
QUANTITIES = ("positions", "velocities", "forces")
# The (precision, deterministic) cases and the dtypes of the state and the
# forces that they store (#102: the deterministic mixed mode stores forces
# as the state is).
CASES = [("Double", True, "float64", "float64"),
         ("Mixed", True, "float64", "float64"),
         ("Mixed", False, "float64", "float32")]


def expect(error, call, text=""):
    try:
        call()
    except error as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected {error.__name__} ({text})")


def make(precision="Double", deterministic=True, method="VelocityVerlet", kind="NVT",
         minimize=False, pressure=1.0, cutoff=0.8):
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = cutoff, cutoff + (0.1 if cutoff < 1 else 0.005)
    system.switch_distance = cutoff - (0.1 if cutoff < 1 else 0.04)
    system.electrostatics = mdir.Electrostatics.PME
    system.tunables = [mdir.Tunable("q", "charge")]
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.method = getattr(mdir.IntegratorMethod, method)
    integrator.timestep = 0.001
    integrator.minimize = minimize
    ensemble.kind = getattr(mdir.EnsembleKind, "NVE" if minimize else kind)
    ensemble.temperature, ensemble.coupling_period = 300, 10
    ensemble.pressure, ensemble.tau_p = pressure, 0.1
    execution.target, execution.precision = target, getattr(mdir.Precision, precision)
    execution.deterministic = deterministic
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


def in_input_order(values, ids):
    out = np.empty_like(values)
    out[ids] = values
    return out


def check_values(state, ids, arrays, label):
    """The rows of the buffers are those of state() at the rows of ids."""
    assert np.array_equal(np.sort(ids), np.arange(len(ids))), (label, "ids")
    for q, a in zip(QUANTITIES, arrays):
        # state() widens f32 forces to f64 exactly.
        assert np.array_equal(in_input_order(a.astype(np.float64), ids), getattr(state, q)), (label, q)


def check_blocked(sim):
    expect(mdir.SimulationError, lambda: sim.run(1), "lease")
    expect(mdir.SimulationError, lambda: sim.run(0, energy=True), "lease")
    expect(mdir.SimulationError, lambda: sim.tunables.update({"q": sim.tunables["q"]}), "lease")
    sim.state()  # reads are allowed


# --- NumPy on the CPU --------------------------------------------------------

def numpy_scenario():
    for precision, deterministic, state_type, force_type in CASES:
        label = f"{precision}{'' if deterministic else ', not deterministic'}"
        sim = mdir.Simulation(make(precision, deterministic))
        expect(mdir.SimulationError, sim.view, "before its first run")
        sim.run(7)
        state = sim.state()
        view = sim.view()
        assert sim.leases == 1 and view.valid and view.step == 7 and view.device == (1, 0)
        assert view.positions.dtype == np.dtype(state_type)
        assert view.forces.dtype == np.dtype(force_type) and view.ids.dtype == np.dtype("int32")
        arrays = [np.from_dlpack(getattr(view, q)) for q in QUANTITIES]
        ids = np.from_dlpack(view.ids)
        for q, a in zip(QUANTITIES, arrays):
            assert a.ctypes.data == getattr(view, q).data_ptr, (label, q, "a copy")
            # NumPy 2 takes the versioned capsule and its read-only flag;
            # 1.x takes the legacy one, which has no flag.
            if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
                assert not a.flags.writeable, (label, q, "writeable")
            assert a.shape == (len(ids), 3)
        check_values(state, ids, arrays, label)
        q = np.from_dlpack(view.tunables["q"])
        assert np.array_equal(q, sim.tunables["q"]) and q.ctypes.data == view.tunables["q"].data_ptr
        assert sim.leases == 6, sim.leases
        check_blocked(sim)
        # The view's own lease ends; the arrays keep theirs.
        view.release()
        assert sim.leases == 5 and view.released
        expect(BufferError, lambda: view.positions.__dlpack__(), "released")
        expect(mdir.SimulationError, lambda: sim.run(1), "5 leases")
        del arrays, ids, q, a
        gc.collect()
        assert sim.leases == 0, sim.leases
        sim.run(1)
        assert not view.valid
        print(f"{label}: positions {state_type}, forces {force_type}, the program's buffers, "
              f"rows of state() at ids, runs blocked by 6 leases and allowed after them")

    # Capsules: each releases its lease once, taken by a consumer or not,
    # versioned or legacy.
    sim = mdir.Simulation(make())
    sim.run(3)
    with sim.view() as view:
        for version in (None, (1, 0), (1, 3)):
            capsule = view.positions.__dlpack__(max_version=version)
            assert sim.leases == 2
            del capsule
            assert sim.leases == 1
            a = np.from_dlpack(view.positions)
            assert sim.leases == 2
            b = a[1:]  # an alias of the alias
            del a
            assert sim.leases == 2
            del b
            assert sim.leases == 1
        expect(BufferError, lambda: view.positions.__dlpack__(copy=True), "copy")
        expect(BufferError, lambda: view.positions.__dlpack__(dl_device=(2, 0)), "device")
        expect(BufferError, lambda: view.positions.__dlpack__(stream=1), "no stream")
        view.positions.__dlpack__(stream=-1, dl_device=(1, 0), copy=False)
        kept = np.from_dlpack(view.forces)
        copy = kept.copy()
    assert sim.leases == 1
    # The array outlives the Python simulation, which another replaces.
    del sim
    gc.collect()
    other = mdir.Simulation(make())
    other.run(20)
    assert np.array_equal(kept, copy)
    del kept
    gc.collect()
    print("capsules released once, taken or not, legacy or versioned; refusals; "
          "an array outlives its simulation")

    # A minimization is refused under a lease.
    sim = mdir.Simulation(make(minimize=True))
    sim.minimize(3)
    with sim.view() as view:
        a = np.from_dlpack(view.positions)
        expect(mdir.SimulationError, lambda: sim.minimize(1), "lease")
        del a
    sim.minimize(1)
    # After a failed part the state is on the host only: a barostat at 1000
    # bar takes the cell below twice the cutoff of 1.24 nm (as in
    # python_resident.py).
    sim = mdir.Simulation(make(kind="NPT", pressure=1e3, cutoff=1.24))
    expect(mdir.SimulationError, lambda: [sim.run(10) for _ in range(1000)])
    assert sim.failed
    expect(mdir.SimulationError, sim.view, "failed")
    print("a minimization refused under a lease; no view after a failure")


# --- ctypes and the CUDA driver on a GPU -------------------------------------

class DLDevice(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int32), ("id", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", DLDevice), ("ndim", ctypes.c_int32),
                ("dtype", DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32),
                ("context", ctypes.c_void_p), ("deleter", ctypes.c_void_p),
                ("flags", ctypes.c_uint64), ("tensor", DLTensor)]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [("tensor", DLTensor), ("context", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]


api = ctypes.pythonapi
api.PyCapsule_GetPointer.restype = ctypes.c_void_p
api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
api.PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]
api.PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]
USED = {b"dltensor": ctypes.c_char_p(b"used_dltensor"),
        b"dltensor_versioned": ctypes.c_char_p(b"used_dltensor_versioned")}
DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def take(capsule, name):
    """What a consumer does: takes the managed tensor and renames the capsule."""
    assert api.PyCapsule_IsValid(capsule, name)
    pointer = api.PyCapsule_GetPointer(capsule, name)
    api.PyCapsule_SetName(capsule, USED[name])
    kind = DLManagedTensorVersioned if name == b"dltensor_versioned" else DLManagedTensor
    return pointer, kind.from_address(pointer)


def release(pointer, managed):
    # A foreign call through ctypes releases the GIL: the deleter runs
    # without it.
    DELETER(managed.deleter)(pointer)


class Driver:
    def __init__(self):
        self.cuda = ctypes.CDLL("libcuda.so.1")
        self.check(self.cuda.cuInit(0))

    def check(self, result):
        assert result == 0, f"CUDA error {result}"

    def enter(self, ordinal):
        device, context = ctypes.c_int(), ctypes.c_void_p()
        self.check(self.cuda.cuDeviceGet(ctypes.byref(device), ordinal))
        self.check(self.cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device))
        self.check(self.cuda.cuCtxSetCurrent(context))

    def ordinal_of(self, pointer):
        value = ctypes.c_int()
        # CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
        self.check(self.cuda.cuPointerGetAttribute(ctypes.byref(value), 9, ctypes.c_uint64(pointer)))
        return value.value

    def stream(self):
        s = ctypes.c_void_p()
        self.check(self.cuda.cuStreamCreate(ctypes.byref(s), 1))  # non-blocking
        return s

    def read(self, tensor, stream=None):
        dtype = {(2, 64): np.float64, (2, 32): np.float32, (0, 32): np.int32}[
            (tensor.dtype.code, tensor.dtype.bits)]
        shape = tuple(tensor.shape[k] for k in range(tensor.ndim))
        out = np.empty(shape, dtype)
        if stream is None:
            self.check(self.cuda.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p),
                                                 ctypes.c_uint64(tensor.data), ctypes.c_size_t(out.nbytes)))
        else:
            self.check(self.cuda.cuMemcpyDtoHAsync_v2(out.ctypes.data_as(ctypes.c_void_p),
                                                      ctypes.c_uint64(tensor.data),
                                                      ctypes.c_size_t(out.nbytes), stream))
            self.check(self.cuda.cuStreamSynchronize(stream))
        return out


def ctypes_scenario():
    driver = Driver()
    for precision, deterministic, state_type, force_type in CASES:
        label = f"{precision}{'' if deterministic else ', not deterministic'}"
        sim = mdir.Simulation(make(precision, deterministic))
        expect(mdir.SimulationError, sim.view, "before its first run")
        sim.run(7)
        state = sim.state()
        view = sim.view()
        assert view.device[0] == 2, view.device
        driver.enter(view.device[1])
        stream = driver.stream()
        arrays, taken = [], []
        for q in QUANTITIES + ("ids",):
            buffer = getattr(view, q)
            capsule = buffer.__dlpack__(stream=stream.value, max_version=(1, 0))
            pointer, managed = take(capsule, b"dltensor_versioned")
            t = managed.tensor
            assert (managed.major, managed.minor) == (1, 1) and managed.flags & 1, (label, q)
            assert t.data == buffer.data_ptr and t.byte_offset == 0, (label, q)
            assert (t.device.type, t.device.id) == view.device, (label, q)
            assert driver.ordinal_of(t.data) == view.device[1], (label, q, "device ordinal")
            ndim = 2 if q != "ids" else 1
            assert t.ndim == ndim and t.shape[0] == len(state.positions)
            assert [t.strides[k] for k in range(ndim)] == ([3, 1] if ndim == 2 else [1])
            arrays.append(driver.read(t, stream))
            taken.append((capsule, pointer, managed))
        assert arrays[0].dtype == np.dtype(state_type) and arrays[2].dtype == np.dtype(force_type)
        assert sim.leases == 5
        check_values(state, arrays[3], arrays[:3], label)
        check_blocked(sim)
        # The consumer releases each tensor once, without the GIL; the
        # capsule it renamed releases nothing more.
        for capsule, pointer, managed in taken:
            release(pointer, managed)
        assert sim.leases == 1, sim.leases
        del taken, capsule
        gc.collect()
        assert sim.leases == 1
        # The legacy capsule, and the default streams.
        for stream_value in (None, 1, 2, -1):
            capsule = view.forces.__dlpack__(stream=stream_value)
            pointer, managed = take(capsule, b"dltensor")
            assert np.array_equal(driver.read(managed.tensor), arrays[2])
            release(pointer, managed)
            del capsule
        assert sim.leases == 1
        expect(ValueError, lambda: view.forces.__dlpack__(stream=0), "ambiguous")
        expect(BufferError, lambda: view.forces.__dlpack__(dl_device=(1, 0)), "device")
        view.release()
        assert sim.leases == 0
        sim.run(1)
        driver.check(driver.cuda.cuStreamDestroy_v2(stream))
        print(f"{label}: positions {state_type}, forces {force_type} on device {view.device[1]}, "
              f"read by the driver on a stream of its own, released once without the GIL")

    # A capsule that no consumer takes releases its lease when destroyed; a
    # tensor taken and kept outlives its simulation while another runs.
    sim = mdir.Simulation(make("Mixed", False))
    sim.run(3)
    with sim.view() as view:
        capsule = view.positions.__dlpack__(max_version=(1, 0))
        assert sim.leases == 2
        del capsule
        assert sim.leases == 1
        kept = view.positions.__dlpack__(max_version=(1, 0))
        pointer, managed = take(kept, b"dltensor_versioned")
        before = driver.read(managed.tensor)
    del sim, view
    gc.collect()
    other = mdir.Simulation(make("Mixed", False))
    other.run(20)
    with other.view() as view:
        assert view.positions.data_ptr != managed.tensor.data
    assert np.array_equal(driver.read(managed.tensor), before)
    release(pointer, managed)
    del kept
    gc.collect()
    other.run(5)
    print("a capsule not taken releases once; a tensor outlives its simulation while another runs")


# --- PyTorch ----------------------------------------------------------------

def torch_scenario():
    import torch
    for precision, deterministic, state_type, force_type in CASES:
        label = f"{precision}{'' if deterministic else ', not deterministic'}"
        sim = mdir.Simulation(make(precision, deterministic))
        sim.run(7)
        state = sim.state()
        with sim.view() as view:
            on_device = view.device[0] == 2
            stream = torch.cuda.Stream() if on_device else None
            context = torch.cuda.stream(stream) if on_device else torch.no_grad()
            with context:
                tensors = [torch.from_dlpack(getattr(view, q)) for q in QUANTITIES]
                ids = torch.from_dlpack(view.ids)
                q = torch.from_dlpack(view.tunables["q"])
            for name, t in zip(QUANTITIES, tensors):
                assert t.data_ptr() == getattr(view, name).data_ptr, (label, name, "a copy")
                assert t.is_contiguous() and tuple(t.shape) == (len(state.positions), 3)
                if on_device:
                    assert t.device == torch.device("cuda", view.device[1]), t.device
            assert str(tensors[0].dtype) == f"torch.{state_type}"
            assert str(tensors[2].dtype) == f"torch.{force_type}" and ids.dtype == torch.int32
            if on_device:
                stream.synchronize()
            check_values(state, ids.cpu().numpy(), [t.cpu().numpy() for t in tensors], label)
            assert np.array_equal(q.numpy(), sim.tunables["q"]) and q.device.type == "cpu"
            assert sim.leases == 6
            check_blocked(sim)
            # Retained aliases: views and slices of the tensors.
            alias = tensors[0][2:5]
        del tensors, ids, q, t
        gc.collect()
        assert sim.leases == 1, sim.leases
        expect(mdir.SimulationError, lambda: sim.run(1), "1 lease")
        del alias
        gc.collect()
        assert sim.leases == 0
        sim.run(1)
        print(f"{label}: torch shares the buffers ({state_type}, {force_type}) "
              f"{'on a stream of its own' if on_device else 'on the CPU'}; "
              f"aliases hold their leases")

    sim = mdir.Simulation(make("Mixed", False))
    sim.run(5)
    sim.run(10)  # a part after the first, as the one below
    state = sim.state()
    view = sim.view()
    if view.device[0] == 2:
        order = torch.from_dlpack(view.ids).cpu().numpy()
    else:
        order = np.from_dlpack(view.ids).copy()
    expected = torch.from_numpy(state.positions[order])
    if view.device[0] == 2:
        # Consumer work still running when its aliases are deleted: the
        # next part waits for it before it writes the buffers.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.from_dlpack(view.positions)
            torch.cuda._sleep(1_000_000_000)  # about 0.5 s
            y = x.clone()
        del x
        view.release()
        assert sim.leases == 0
        # The consumer's work is still under way when the run begins.
        assert not stream.query(), "the consumer's work ended before the run"
        sim.run(10)
        stream.synchronize()
        assert torch.equal(y.cpu(), expected), "the run wrote the buffers under consumer work"
        print("consumer work outstanding at the release is waited for by the next part")
        view = sim.view()
    # A tensor that outlives its simulation, and the reuse of its blocks
    # once it is released.
    kept = torch.from_dlpack(view.positions)
    copy = kept.clone()
    old = {view.positions.data_ptr, view.velocities.data_ptr, view.forces.data_ptr}
    view.release()
    del sim, view
    gc.collect()
    other = mdir.Simulation(make("Mixed", False))
    other.run(20)
    with other.view() as view:
        new = {view.positions.data_ptr, view.velocities.data_ptr, view.forces.data_ptr}
    assert not (old & new), "blocks under a live tensor were given to another simulation"
    assert torch.equal(kept, copy)
    del kept
    gc.collect()
    third = mdir.Simulation(make("Mixed", False))
    third.run(20)
    with third.view() as view:
        reused = {view.positions.data_ptr, view.velocities.data_ptr, view.forces.data_ptr} & old
    print("a tensor outlives its simulation; its blocks go to no other simulation while it lives")
    print(f"blocks of the released simulation taken by the next: {len(reused)} of 3", file=sys.stderr)


# --- compute-sanitizer --------------------------------------------------------

def sanitize():
    driver = Driver()
    sim = mdir.Simulation(make("Mixed", False))
    sim.run(3)
    with sim.view() as view:
        driver.enter(view.device[1])
        stream = driver.stream()
        capsule = view.forces.__dlpack__(stream=stream.value, max_version=(1, 0))
        pointer, managed = take(capsule, b"dltensor_versioned")
        driver.read(managed.tensor, stream)
        release(pointer, managed)
        view.positions.__dlpack__()
    sim.run(4)
    with sim.view() as view:
        kept = view.positions.__dlpack__(max_version=(1, 0))
        pointer, managed = take(kept, b"dltensor_versioned")
    del sim
    gc.collect()
    driver.read(managed.tensor)
    release(pointer, managed)
    print("views of two parts, a tensor that outlives its simulation")


SCENARIOS = {"numpy": numpy_scenario, "ctypes": ctypes_scenario, "torch": torch_scenario,
             "sanitize": sanitize}
if scenario not in SCENARIOS:
    raise SystemExit(f"unknown scenario '{scenario}'")
SCENARIOS[scenario]()
print(f"dlpack {scenario} passed")
