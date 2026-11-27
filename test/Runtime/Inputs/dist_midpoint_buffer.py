"""Check active pairs as particles cross the buffered and physical cutoffs."""
import ctypes as C
import sys

I, D, P = C.c_int64, C.c_double, C.c_void_p
runtime = C.CDLL(sys.argv[1])


def descriptor(rank):
    class Memref(C.Structure):
        _fields_ = [("allocated", P), ("aligned", P), ("offset", I),
                    ("sizes", I * rank), ("strides", I * rank)]
    return Memref


def bind(name, result, *arguments):
    fn = getattr(runtime, name)
    fn.argtypes, fn.restype = list(arguments), result
    return fn


V, M, T = (descriptor(i) for i in (1, 2, 3))
abi = "_mlir_ciface_mdrtDist"
partition = bind(abi + "Partition", I, P, I, I, I, D, D, D)
make_halo = bind(abi + "MidpointHalo", I, I, P, D)
make_matrix = bind("mdrtDistMatrix", I, I, I, I, I, D, D, I, I)
rebuild = bind(abi + "MatrixRebuild", I, I, P)
build = bind(abi + "BuildMidpoint", None, I, I, P)
select = bind(abi + "FilterMatrix", I, I, I, I, I, P, P, P, I)
counts = bind(abi + "MatrixCounts", None, P, I, I)
handle = bind("mdrtDistMatrixHandle", I, I, I)
entries = bind("_mlir_ciface_mdrtHostMatrixEntries", None, P, I)

xyz = (D * 9)(5, 5, 5, 6.08, 5, 5, 6.135, 5, 5)
addr = C.addressof(xyz)
packed = M(addr, addr, 0, (3, 3), (3, 1))
owned = T(addr, addr, 0, (1, 3, 3), (9, 3, 1))
layout = partition(C.byref(packed), 1, 1, 1, 12, 12, 12)
halo = make_halo(layout, C.byref(owned), 1.4)
matrix = make_matrix(halo, 0, 3, 3, 1, 0.4, 4, 1)
assert rebuild(matrix, C.byref(packed))
build(halo, matrix, C.byref(packed))
for xs, prune in [((5, 6.08, 6.135), 1), ((5.045, 6.035, 6.092), 1),
                  ((5.079, 6.035, 6.06), 1), ((5.079, 6.035, 6.06), 0),
                  ((5, 6.08, 6.135), 1)]:
    for i, x in enumerate(xs):
        xyz[3 * i] = x
    assert not rebuild(matrix, C.byref(packed))
    assert select(halo, matrix, 0, 2, C.byref(owned), C.byref(owned), C.byref(packed), prune)
    ns, neighbors = V(), M()
    counts(C.byref(ns), matrix, 1)
    entries(C.byref(neighbors), handle(matrix, 1))
    degree = C.cast(ns.aligned, C.POINTER(C.c_int32))
    rows = C.cast(neighbors.aligned, C.POINTER(C.c_int32))
    for i in range(3):
        base = neighbors.offset + i * neighbors.strides[0]
        row = [rows[base + k] for k in range(degree[i])]
        assert len(set(row)) == len(row) and all(i < j < 3 for j in row)
        assert {j for j in row if abs(xs[i] - xs[j]) < 1} == {
            j for j in range(i + 1, 3) if abs(xs[i] - xs[j]) < 1}
bind("mdrtDistReleaseHalo", None, I)(halo)
bind("mdrtDistReleaseLayout", None, I)(layout)
