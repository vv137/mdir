"""Run the emitted rebuild entries and compare stable IDs with a periodic oracle."""
import ctypes as C
import itertools
import math
import random
import re
import subprocess
import sys
from pathlib import Path

MDIR_OPT = "mdir-opt"
I, F, D = C.c_int32, C.c_float, C.c_double
P = C.c_void_p


def compile_entries(directory, dims, reach, n, capacity, compact, gpu, audit=True):
    options = dict(zip(("nx", "ny", "nz"), dims))
    options.update(reach=reach, atoms=n, capacity=capacity, compact=int(compact),
                   host=int(not gpu), audit=int(audit))
    flags = " ".join(f"{k}={v}" for k, v in options.items())
    ir = subprocess.check_output([MDIR_OPT, f"--add-towers-build={flags}"],
                                 input=b"module {}")
    opts = ["mlir-opt", "--canonicalize", "--cse"]
    if gpu:
        ir = subprocess.check_output(opts + [
            "--gpu-lower-to-nvvm-pipeline=cubin-format=isa cubin-chip=sm_80 "
            "kernel-bare-ptr-calling-convention=true"], input=ir)
        match = re.search(rb'"(//[^"]*?\.entry[^"]*)"', ir)
        if not match:
            raise ValueError("missing PTX object")
        return re.sub(rb'\\([0-9A-Fa-f]{2})',
                      lambda m: bytes([int(m[1], 16)]), match[1])
    ir = subprocess.check_output(opts + [
        "--convert-scf-to-cf", "--convert-math-to-llvm", "--finalize-memref-to-llvm",
        "--convert-func-to-llvm=use-bare-ptr-memref-call-conv",
        "--convert-arith-to-llvm", "--convert-cf-to-llvm",
        "--reconcile-unrealized-casts"], input=ir)
    llvm = subprocess.check_output(["mlir-translate", "--mlir-to-llvmir"], input=ir)
    obj = directory / "host.o"
    subprocess.run(["llc", "-O2", "-filetype=obj", "-relocation-model=pic",
                    "-o", obj], input=llvm, check=True)
    lib = directory / ("host-" + "-".join(str(v) for v in options.values()) + ".so")
    subprocess.run(["cc", "-shared", obj, "-o", lib, "-lm"], check=True)
    return lib


class Entries:
    """Both targets use the static-shape bare-pointer memref calling convention."""
    def __init__(self, image, gpu):
        self.gpu = gpu
        self.allocations = []
        if not gpu:
            self.lib = C.CDLL(str(image))
            return
        self.driver = C.CDLL("libcuda.so.1")
        self.cuda("cuInit", 0)
        self.context = P()
        self.cuda("cuDevicePrimaryCtxRetain", C.byref(self.context), 0)
        self.cuda("cuCtxSetCurrent", self.context)
        self.module = P()
        self.cuda("cuModuleLoadData", C.byref(self.module), image)
        self.functions = {}

    def cuda(self, name, *args):
        status = getattr(self.driver, name)(*args)
        if status:
            raise RuntimeError(f"{name} returned {status}")

    def array(self, kind, n):
        if not self.gpu:
            return (kind * n)()
        ptr = P()
        self.cuda("cuMemAllocManaged", C.byref(ptr), C.c_size_t(n*C.sizeof(kind)), 1)
        self.allocations.append(ptr)
        return (kind * n).from_address(ptr.value)

    def run(self, phase, items, *args):
        name = "towers_" + phase
        if not self.gpu:
            getattr(self.lib, name)(*args)
            return
        if name not in self.functions:
            fn = P()
            self.cuda("cuModuleGetFunction", C.byref(fn), self.module, name.encode())
            self.functions[name] = fn
        values = [P(C.addressof(x)) if isinstance(x, C.Array) else x for x in args]
        argv = (P * len(args))(*(C.addressof(x) for x in values))
        self.cuda("cuLaunchKernel", self.functions[name], (items+127)//128, 1, 1,
                  128, 1, 1, 0, None, argv, None)

    def sync(self):
        if self.gpu:
            self.cuda("cuCtxSynchronize")

    def close(self):
        if self.gpu:
            for ptr in self.allocations:
                self.cuda("cuMemFree_v2", ptr)
            self.cuda("cuModuleUnload", self.module)
            self.cuda("cuDevicePrimaryCtxRelease", 0)


class Rebuild:
    def __init__(self, entries, dims, reach, n, capacity, compact):
        self.entries, self.n, self.compact = entries, n, compact
        self.cells = math.prod(dims)
        self.fields = 5 + 2*reach
        slots = capacity*self.cells
        alloc = entries.array
        self.map = alloc(I, self.cells*self.fields)
        self.count = alloc(I, self.cells)
        self.status = alloc(I, 1)
        self.raw, self.box = alloc(D, 3*n), alloc(D, 3)
        self.charge, self.types = alloc(F, n), alloc(I, n)
        self.keys, self.local = alloc(I, slots), alloc(F, 4*n)
        self.packed, self.packed_types = alloc(F, 4*slots), alloc(I, slots)
        self.ids, self.inverse = alloc(I, slots), alloc(I, n)
        self.spans = alloc(I, 2*self.cells*self.fields)
        entries.run("setup", self.cells, self.map, self.count, self.status)
        entries.sync()

    def step(self, reverse=False):
        run = self.entries.run
        if not self.compact:
            run("clear", self.cells, self.count)
        run("insert", self.n, self.raw, self.charge, self.box, self.count,
            self.keys, self.local, I(reverse))
        run("pack", self.cells*32, self.count, self.keys, self.local,
            self.types, self.map, self.packed, self.packed_types, self.ids,
            self.inverse, self.spans, self.status)


def check(directory, gpu, reach, compact):
    dims = (3, 4, 2*reach+1)
    lengths = (6, 8, dims[2]*2/reach)
    # Many tied subcell keys exercise both warp sort and the dense-cell path.
    coords = [(0.25, 0.25, 0.25)]*68
    coords += [(x, 2., 2.) for x in (0., 6., -12., 2., math.nextafter(2., 0.))]
    rng = random.Random(194)
    coords += [tuple(rng.uniform(-l, 2*l) for l in lengths) for _ in range(48)]
    n = len(coords)
    tested = 0
    for capacity in (32, n):
        entries = Entries(compile_entries(directory, dims, reach, n, capacity,
                                          compact, gpu), gpu)
        r = Rebuild(entries, dims, reach, n, capacity, compact)
        r.box[:] = lengths
        r.types[:] = [i % 17 for i in range(n)]
        r.charge[:] = [i/8 for i in range(n)]
        pairs = entries.array(I, n*n)
        for frame, reverse in itertools.product(range(2), (False, True)):
            xyz = [tuple(x + frame*(i % 7-3)*(axis+1)*.7
                         for axis, x in enumerate(p)) for i, p in enumerate(coords)]
            r.raw[:] = [p[axis] for axis in range(3) for p in xyz]
            r.step(reverse)
            entries.sync()
            bins = [[] for _ in range(r.cells)]
            for i, p in enumerate(xyz):
                quarters = [min(4*b-1, int(4*((x % l)/(l/b))))
                            for x, l, b in zip(p, lengths, dims)]
                cell = (quarters[0]//4*dims[1] + quarters[1]//4)*dims[2] + quarters[2]//4
                bins[cell].append((tuple(q % 4 for q in quarters), i))
            required = max(map(len, bins))
            if capacity < required:
                assert r.status[0] == required
                break
            assert r.status[0] == 0
            for cell, members in enumerate(bins):
                offset = 2*cell*r.fields
                start, count = r.spans[offset:offset+2]
                assert r.ids[start:start+count] == [i for _, i in sorted(members)]
                for slot in range(start, start+count):
                    atom = r.ids[slot]
                    assert r.inverse[atom] == slot
                    assert r.packed_types[slot] == atom % 17
                    assert r.packed[4*slot+3] == atom/8
                    for axis, (x, l, b) in enumerate(zip(xyz[atom], lengths, dims)):
                        v = x % l
                        bin_id = min(b-1, int(v/(l/b)))
                        assert r.packed[4*slot+axis] == F(v-(bin_id+.5)*(l/b)).value
            C.memset(C.addressof(pairs), 0, C.sizeof(pairs))
            entries.run("ownership", r.cells, r.spans, r.ids, pairs)
            entries.sync()
            assert max(pairs) <= 1
            assert all(pairs[i*n+j] == 0 for i in range(n) for j in range(i+1))
            for i, j in itertools.combinations(range(n), 2):
                delta = [math.remainder(a-b,l) for a,b,l in zip(xyz[i],xyz[j],lengths)]
                if sum(d*d for d in delta) < 4:
                    assert pairs[i*n+j] == 1, (frame, reverse, i, j)
                    tested += 1
        entries.close()
    return tested


if __name__ == "__main__":
    MDIR_OPT = sys.argv[3]
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    total = sum(check(directory, sys.argv[2] == "gpu", reach, compact)
                for reach in (1, 2) for compact in (False, True))
    print(f"{total} in-cutoff pair visits matched")
