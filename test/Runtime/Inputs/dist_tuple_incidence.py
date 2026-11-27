"""Compare the topology ordering and force scatter with serial grouping."""
import ctypes as C
import sys

Index = C.c_int64
Pointer = C.POINTER(Index)


def descriptor(rank):
    class Memref(C.Structure):
        _fields_ = [("allocated", Pointer), ("aligned", Pointer),
                    ("offset", Index), ("sizes", Index * rank),
                    ("strides", Index * rank)]
    return Memref


Vector = descriptor(1)
Matrix = descriptor(2)
runtime = C.CDLL(sys.argv[1])
chunks = runtime._mlir_ciface_mdrtDistTupleChunks
chunks.argtypes = [C.POINTER(Matrix), Index, Index, Index, C.POINTER(Vector),
                   C.POINTER(Vector)]
chunks.restype = Index
incidence = runtime._mlir_ciface_mdrtDistTupleIncidence
incidence.argtypes = [C.POINTER(Matrix), C.POINTER(Vector), Index, Index, Index,
                      Index, C.POINTER(Vector), C.POINTER(Vector)]
incidence.restype = Index

arity, stride = 3, 47
owners = [7 if i % 101 == 0 else (3 if i % 13 else 5) for i in range(10241)]
local = [i for i, owner in enumerate(owners) if 3 <= owner < 6]
order = sorted(local, key=owners.__getitem__)
expected = {}
records = []
for i, owner in enumerate(owners):
    row = (i * 17) % stride
    records.extend((owner, row, (row + 7) % stride, row))
for p, t in enumerate(order):
    for a in range(arity):
        row = records[t * (arity + 1) + a + 1]
        expected.setdefault((owners[t], row), []).append(p * arity + a)

def vector(data):
    return Vector(data, data, 0, (len(data),), (1,))

storage = (Index * len(records))(*records)
rows = Matrix(storage, storage, 0, (len(owners), arity + 1), (arity + 1, 1))
indices = vector((Index * len(owners))())
boundaries = vector((Index * (len(owners) + 1))())
chunk_count = chunks(rows, 3, 6, 128, indices, boundaries)
assert indices.aligned[:len(order)] == order
expected_bounds = []
offset = 0
for domain in range(3, 6):
    size = owners.count(domain)
    expected_bounds.extend(range(offset, offset + size, 128))
    offset += size
assert boundaries.aligned[:chunk_count + 1] == expected_bounds + [len(order)]
entries = vector((Index * (len(order) * arity))())
starts = vector((Index * (len(order) * arity + 1))())
active = incidence(rows, indices, len(order), 3, 6, stride, entries, starts)
assert active == len(expected)
for i, key in enumerate(sorted(expected)):
    assert entries.aligned[starts.aligned[i]:starts.aligned[i + 1]] == expected[key]
assert starts.aligned[active] == len(order) * arity
