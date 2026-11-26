"""Check every rank-local Fourier coefficient against a sparse direct DFT."""
import cmath
import ctypes as c
import itertools
import math
import sys


class Box(c.Structure):
    _fields_ = [("lo", c.c_int * 3), ("hi", c.c_int * 3)]

    def points(self):
        return itertools.product(*(range(a, b) for a, b in zip(self.lo, self.hi)))


lib = c.CDLL(sys.argv[1])
lib.mdrtPmePencils.argtypes = [c.c_int] * 8
lib.mdrtPmePencils.restype = c.c_void_p
for func in (lib.mdrtPmeRealBox, lib.mdrtPmeSpectrumBox):
    func.argtypes = [c.c_void_p]
    func.restype = Box
for func in (lib.mdrtPmeForward, lib.mdrtPmeBackward):
    func.argtypes = [c.c_void_p] + [c.POINTER(c.c_double)] * 2
    func.restype = None
lib.mdrtPmeFreePencils.argtypes = [c.c_void_p]
lib.mdrtPmeFreePencils.restype = None
class FloatBuffer(c.Structure):
    _fields_ = [("allocated", c.POINTER(c.c_float)), ("aligned", c.POINTER(c.c_float)),
                ("offset", c.c_int64), ("size", c.c_int64), ("stride", c.c_int64)]

    @classmethod
    def wrap(cls, array):
        return cls(array, array, 0, len(array), 1)


def mixed_inverse(shape, px, py, mpi):
    # Compare at binary32 storage precision after each inverse axis.
    nx, ny, nz = shape
    half = nz // 2 + 1
    packed = (c.c_float * (2 * nx * ny * half))(
        *(math.sin(i * 0.31) for i in range(2 * nx * ny * half)))
    ref = (c.c_float * (nx * ny * nz))()
    mesh = lib.mdrtPmePencils(*shape, px, py, 4, mpi, 1)
    modes = lib.mdrtPmeSpectrumBox(mesh).points()
    pairs = [packed[2*((x*ny + y)*half + z) + k]
             for x, y, z in modes for k in range(2)]
    local = (c.c_double * len(pairs))(*pairs)
    rows = list(lib.mdrtPmeRealBox(mesh).points())
    result = (c.c_double * len(rows))()
    ordinary = lib._mlir_ciface_mdrtFFT32Backward3D
    ordinary.argtypes = [c.POINTER(FloatBuffer)] * 2 + [c.c_int64] * 3
    ordinary.restype = None
    ordinary(c.byref(FloatBuffer.wrap(packed)), c.byref(FloatBuffer.wrap(ref)), *shape)
    lib.mdrtPmeBackward(mesh, local, result)
    assert list(result) == [ref[(x*ny + y)*nz + z] for x, y, z in rows]
    lib.mdrtPmeFreePencils(mesh)


px, py, mpi = map(int, sys.argv[2:5])
nx, ny = 9, 13
for nz in (11, 12):
    shape = (nx, ny, nz)
    mesh = lib.mdrtPmePencils(*shape, px, py, 4, mpi, 0)
    rows = list(lib.mdrtPmeRealBox(mesh).points())
    modes = list(lib.mdrtPmeSpectrumBox(mesh).points())
    charges = {(8, 12, 7): -0.7, (0, 1, 0): 1.1, (4, 3, 1): 0.25}
    real = (c.c_double * len(rows))(*(charges.get(row, 0) for row in rows))
    fft = (c.c_double * (2 * len(modes)))()
    lib.mdrtPmeForward(mesh, real, fft)
    error = 0
    for i, mode in enumerate(modes):
        expected = sum(q * cmath.exp(-2j * math.pi * sum(
            m * r / n for m, r, n in zip(mode, point, shape)))
            for point, q in charges.items())
        error = max(error, abs(complex(fft[2*i], fft[2*i+1]) - expected))
    assert error < 6e-14, error
    lib.mdrtPmeBackward(mesh, fft, real)
    inverse_error = max((abs(x / math.prod(shape) - charges.get(row, 0))
                         for x, row in zip(real, rows)), default=0)
    assert inverse_error < 8e-16, inverse_error
    print(f"{shape} {px}x{py} modes={len(modes)} DFT={error:.8g} inverse={inverse_error:.8g}")
    lib.mdrtPmeFreePencils(mesh)
    mixed_inverse(shape, px, py, mpi)
