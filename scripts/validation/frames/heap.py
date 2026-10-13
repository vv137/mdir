"""The bytes in use of the host across the frames of one frame evaluator
(#306, docs/python-frames.md).

  heap.py ROOT TARGET PRECISION TERMS BATCH KEEP CALLS [READER] [INFO]

ROOT is test/Driver/Inputs. The program is the alanine dipeptide in water
of the tests (1,168 atoms, PME, 25 tied charges and sigma and epsilon of
the pair OW-OW tunable, an observed pair term and an observed wall), in
the deterministic mode. TERMS is `all` or `dependent`. One evaluator takes
calls of BATCH frames each, the same array at every call; KEEP is

  kept     gradients=("tunables",), the rows of the Jacobian returned and kept by the
           result of the call (which is dropped before the next call);
  dropped  the same with jacobian_bytes=0: the rows are not returned;
  none     gradients=();
  nothing  no evaluation: an empty function in place of the call, which
           reads what the script and the interpreter move by themselves.

The bytes in use of the allocator (`mallinfo2`: `uordblks + hblkhd`, the
blocks allocated and not freed in every arena and the blocks mapped one by
one, with the thread cache off) are read after 1, 2, 4, ... calls, up to
CALLS: at powers of two a vector that doubles is read at the same place of
its growth each time, so that what grows with the frames reads the same
number of bytes a frame at every reading and what does not reads 0. CALLS
written `every:N` reads after each of N calls instead, as the observation
of #306 did.

READER is how they are read:

  once   (default) the structure of `mallinfo2` and the handle of the C
         library are made once, before the first reading, so that a reading
         allocates nothing that the next one counts;
  each   as `in_use()` of test/Driver/Inputs/python_frames.py did before
         #306: the class of the structure and the handle are made anew at
         every reading.

With INFO, a directory, `malloc_info` is written there at every reading
(`info-<calls>.xml`), whose counts by size of the free blocks tell which
sizes move. Each line printed: the calls, the frames, the bytes in use,
their change since the reading before, that change per frame, the address
space mapped outside the heap of the main arena (`/proc/self/maps`), which
holds what the allocator does not give, and the blocks that the
interpreter has allocated (`sys.getallocatedblocks`), which the allocator
does not see either.
"""
import ctypes
import gc
import os
import sys

root, target, precision, terms, batch, keep, calls = sys.argv[1:8]
reader = sys.argv[8] if len(sys.argv) > 8 else "once"
info = sys.argv[9] if len(sys.argv) > 9 else None
every = calls.startswith("every:")
batch, calls = int(batch), int(calls.split(":")[-1])
assert terms in ("all", "dependent") and keep in ("kept", "dropped", "none", "nothing")
assert reader in ("once", "each")
TCACHE_OFF = "glibc.malloc.tcache_count=0"
if TCACHE_OFF not in os.environ.get("GLIBC_TUNABLES", ""):
    # The allocator reads its tunables when the process begins.
    os.environ["GLIBC_TUNABLES"] = ":".join(
        filter(None, [os.environ.get("GLIBC_TUNABLES"), TCACHE_OFF]))
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import mdir

FIELDS = ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks",
          "uordblks", "fordblks", "keepcost")


class Mallinfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in FIELDS]


libc = ctypes.CDLL(None)
libc.mallinfo2.restype = Mallinfo2
libc.fopen.restype = ctypes.c_void_p
libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
libc.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
libc.fclose.argtypes = [ctypes.c_void_p]


def in_use_once():
    gc.collect()
    data = libc.mallinfo2()
    return data.uordblks + data.hblkhd


def in_use_each():
    class Each(ctypes.Structure):
        _fields_ = [(name, ctypes.c_size_t) for name in FIELDS]

    library = ctypes.CDLL(None)
    library.mallinfo2.restype = Each
    gc.collect()
    data = library.mallinfo2()
    return data.uordblks + data.hblkhd


def mapped():
    """The address space mapped outside the heap of the main arena, in
    bytes (as Inputs/python_memory.py reads it): what does not come from
    the allocator, such as the stack of an activation, the code of an
    engine, the arenas of the interpreter and of other threads, and on a
    GPU what the driver maps."""
    total = 0
    with open("/proc/self/maps") as f:
        for line in f:
            fields = line.split()
            if fields[-1] != "[heap]":
                low, high = (int(address, 16) for address in fields[0].split("-"))
                total += high - low
    return total


def tied_charges(system):
    top = system.topology
    residues = [top.residue_names[r] for r in top.residue_indices]
    entries, charge_map = {}, []
    for i, (atom, residue) in enumerate(zip(top.atom_names, residues)):
        key = (residue, atom) if residue == "WAT" else (residue, i)
        charge_map.append(entries.setdefault(key, len(entries)))
    return np.array(charge_map, dtype=np.int64)


def one_pair(system, a, b):
    top = system.topology
    names, pairs = list(top.type_names), np.array(top.type_pairs)
    a, b = sorted((names.index(a), names.index(b)))
    return np.where((pairs[:, 0] == a) & (pairs[:, 1] == b), 0, -1)


def program_():
    directory = root + "/dipeptide"
    loaded = mdir.load_amber(directory + "/dipeptide.prmtop", directory + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.truncation = mdir.Truncation.Switch
    system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
    system.electrostatics = mdir.Electrostatics.PME
    system.dispersion = mdir.DispersionCorrection.None_
    soft = mdir.PairTerm()
    soft.name, soft.expression = "soft", "a*exp(-r/l)"
    soft.constants = [("a", 2.0), ("l", 0.05)]
    soft.observe = ["l"]
    wall = mdir.ExternalTerm()
    wall.name, wall.expression, wall.selection = "wall", "0.5*k*max(0, z - z0)^2", ":WAT"
    wall.constants = [("k", 100.0), ("z0", 1.5)]
    wall.observe = ["z0"]
    system.pair_terms, system.external_terms = [soft], [wall]
    system.tunables = [mdir.Tunable("q", "charge", map=tied_charges(system)),
                       mdir.Tunable("sigma", "sigma_pair", map=one_pair(system, "OW", "OW")),
                       mdir.Tunable("epsilon", "epsilon_pair", map=one_pair(system, "OW", "OW"))]
    system.tunable_gradient = True
    integrator = mdir.Integrator()
    integrator.timestep = 0.0005
    execution = mdir.Execution()
    execution.target = getattr(mdir.Target, target)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic, execution.reorder = True, False
    return mdir.compile(system, state, integrator, mdir.Ensemble(), execution, mdir.Schedule())


program = program_()
sampler = mdir.Simulation(program)
states = []
for _ in range(4):
    sampler.run(4)
    states.append(sampler.state())
del sampler
positions = np.stack([s.positions for s in states])[np.arange(batch) % len(states)]
edges = np.tile(states[0].cell.diagonal, (batch, 1))
evaluator = mdir.FrameEvaluator(program, terms=terms,
                                **({"jacobian_bytes": 0} if keep == "dropped" else {}))
read = in_use_once if reader == "once" else in_use_each
print(f"{target} {precision}, terms={terms}, calls of {batch} frames, the Jacobian {keep}, "
      f"read {reader}", flush=True)


def measure():
    # In a function: its names are slots. At the level of the module each
    # `del result` and assignment would leave a dead entry in the dictionary
    # of the globals, which is rebuilt, larger once, when its entries are
    # used up (+1744 bytes at some call between the 17th and the 64th, with
    # an empty function in place of the evaluator as well).
    def call():
        if keep == "nothing":
            return None
        return evaluator.evaluate(positions, edges, gradients=() if keep == "none" else ("tunables",))

    done, last, before, at = 0, None, 0, 1
    while at <= calls:
        while done != at:
            result = call()
            del result
            done += 1
        now, outside = read(), mapped()
        if info:
            stream = libc.fopen(f"{info}/info-{done}.xml".encode(), b"w")
            libc.malloc_info(0, stream)
            libc.fclose(stream)
        change = "" if last is None else (
            f" {now - last:+d} bytes, {(now - last) / (batch * (at - before)):+.3f} a frame")
        print(f"calls {done:7d} frames {done * batch:8d} in use {now}{change}; "
              f"mapped outside the heap {outside}; "
              f"blocks of the interpreter {sys.getallocatedblocks()}", flush=True)
        last, before, at = now, at, (at + 1 if every else 2 * at)


measure()
