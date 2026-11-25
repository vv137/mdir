"""Exercise the checked add and an incomplete deterministic image payload."""
import ctypes as c
import sys

lib = c.CDLL(sys.argv[1])
word = c.c_int64
if sys.argv[2] == "overflow":
    add = lib.mdrtDistAddWords
    add.argtypes = [word, word]
    add.restype = word
    assert add(2**63 - 2, 1) == 2**63 - 1
    add(2**63 - 1, 1)
    raise AssertionError("overflow accepted")


def descriptor(shape, data):
    rank = len(shape)
    indices = word * rank

    class Memref(c.Structure):
        _fields_ = [("allocated", c.POINTER(c.c_double)),
                    ("aligned", c.POINTER(c.c_double)), ("offset", word),
                    ("sizes", indices), ("strides", indices)]

    strides = [1] * rank
    for i in reversed(range(rank - 1)):
        strides[i] = shape[i + 1] * strides[i + 1]
    return Memref(data, data, 0, indices(*shape), indices(*strides))


# Two particles on opposite sides of the x domain boundary are within reach.
data = (c.c_double * 6)(1.9, 2, 2, 2.1, 2, 2)
positions = descriptor([2, 3], data)
partition = lib._mlir_ciface_mdrtDistPartition
partition.restype = word
layout = partition(c.byref(positions), word(2), word(1), word(1),
                   c.c_double(4), c.c_double(4), c.c_double(4))
owned = descriptor([2, 1, 3], data)
build = lib._mlir_ciface_mdrtDistHalo
build.restype = word
halo = build(word(layout), c.byref(owned), c.c_double(0.5))
empty = descriptor([2, 0, 3], data)
lib._mlir_ciface_mdrtDistImageShifts(word(halo), c.byref(empty))
raise AssertionError("incomplete halo accepted")
