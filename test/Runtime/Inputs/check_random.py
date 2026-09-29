"""Checks the random numbers of the runtime: Philox 4x32-10 against the
known answers of Random123, and the factor of the thermostat against the
distribution that it samples."""

import ctypes
import math
import sys

library = ctypes.CDLL(sys.argv[1])

# The known answers of Random123 (kat_vectors): counter, key, result.
Words4 = ctypes.c_uint32 * 4
Words2 = ctypes.c_uint32 * 2
answers = [
    ([0, 0, 0, 0], [0, 0], [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]),
    (
        [0xFFFFFFFF] * 4,
        [0xFFFFFFFF] * 2,
        [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD],
    ),
    (
        [0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344],
        [0xA4093822, 0x299F31D0],
        [0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1],
    ),
]
for counter, key, expected in answers:
    result = Words4()
    library.mdrtPhilox4x32(Words4(*counter), Words2(*key), result)
    got = list(result)
    status = "ok" if got == expected else "FAILED"
    print(f"philox {status}: " + " ".join(f"{w:08x}" for w in got))

# With no memory of the kinetic energy, the thermostat draws it from the
# canonical distribution: a gamma of shape N/2, with mean T and variance
# 2 T^2 / N. With memory, the mean and the variance of a long chain are the
# same.
factor = library.mdrtBussiFactor
factor.restype = ctypes.c_double
factor.argtypes = [ctypes.c_int64, ctypes.c_int64] + [ctypes.c_double] * 4
target = 100.0
for freedom, decay in [(30.0, 0.0), (30.0, 0.9), (3.0, 0.5), (1.0, 0.0)]:
    kinetic = target
    values = []
    for step in range(200000):
        kinetic *= factor(2718, step, kinetic, target, freedom, decay) ** 2
        values.append(kinetic)
    mean = sum(values) / len(values)
    variance = sum((k - mean) ** 2 for k in values) / len(values)
    expected = 2.0 * target**2 / freedom
    ok = abs(mean / target - 1) < 0.02 and abs(variance / expected - 1) < 0.05
    print(
        f"bussi {'ok' if ok else 'FAILED'}: freedom {freedom} decay {decay} "
        f"mean {mean / target:.3f} variance {variance / expected:.3f}"
    )

# The same key gives the same numbers.
same = factor(1, 5, 50.0, 60.0, 30.0, 0.5) == factor(1, 5, 50.0, 60.0, 30.0, 0.5)
other = factor(1, 6, 50.0, 60.0, 30.0, 0.5) != factor(1, 5, 50.0, 60.0, 30.0, 0.5)
print(f"key {'ok' if same and other else 'FAILED'}")

# The strain of the barostat: with no temperature it is the drift alone, as
# in the worked example of the specification (P0 = 1 bar, P = -150 bar,
# beta_T = 4.5e-5 / bar, dt_p / tau_p = 0.02 / 5): -1.8e-7 * 151.
strain = library.mdrtBarostatStrain
strain.restype = ctypes.c_double
strain.argtypes = [ctypes.c_int64, ctypes.c_int64] + [ctypes.c_double] * 6
drift = strain(1, 0, -150.0, 1.0, 27.0, 0.0, 4.5e-5, 0.02 / 5.0)
print(f"strain drift {'ok' if abs(drift + 2.718e-5) < 1e-15 else 'FAILED'}: {drift:.6e}")
# At the target pressure its mean is 0 and its deviation sqrt(2 kT f c / V):
# 7.431417e-4 for V = 27 nm^3 at 300 K.
kT = 0.0083144626181532 * 300.0
values = [strain(7, n, 1.0, 1.0, 27.0, kT, 4.5e-5, 0.004) for n in range(40000)]
mean = sum(values) / len(values)
deviation = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
ok = abs(mean) < 4 * 7.431417e-4 / math.sqrt(len(values)) and abs(deviation / 7.431417e-4 - 1) < 0.02
print(f"strain noise {'ok' if ok else 'FAILED'}: mean {mean:.2e} deviation {deviation:.6e}")
