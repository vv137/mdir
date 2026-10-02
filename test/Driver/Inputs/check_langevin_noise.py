"""Checks the noise of Langevin dynamics against Philox 4x32-10 of the
runtime (which test/Runtime/random.test checks against Random123): after
one step from velocities of 0, with c = exp(-gamma dt) of e^-1 and a step
short enough that the forces move the velocities by less than 1e-4 of the
noise, each velocity is sqrt((1 - c^2) k_B T / m) times three normal
numbers from the block of Philox under the key of A13: the seed, and the
counter (step, number of the particle, stream 2 << 24). The velocities come
from `mdir checkpoint --print=velocities`.

    mdir checkpoint --print=velocities CHECKPOINT \\
        | check_langevin_noise.py LIBMDRT SEED STEP TEMPERATURE"""

import ctypes
import math
import sys

library = ctypes.CDLL(sys.argv[1])
seed, step, temperature = (int(sys.argv[2]), int(sys.argv[3]),
                           float(sys.argv[4]))
Words4 = ctypes.c_uint32 * 4
Words2 = ctypes.c_uint32 * 2
kT = 0.0083144626181532 * temperature
c = math.exp(-1.0)
worst, count = 0.0, 0
for line in sys.stdin:
    fields = line.split()
    index, mass = int(fields[0]), float(fields[1])
    velocity = [float(v) for v in fields[2:5]]
    count += 1
    if mass <= 0.0:
        continue
    words = Words4()
    library.mdrtPhilox4x32(
        Words4(step & 0xFFFFFFFF, step >> 32, index, 2 << 24),
        Words2(seed & 0xFFFFFFFF, seed >> 32), words)
    u = [(w + 0.5) * 2.0**-32 for w in words]
    r0, r1 = math.sqrt(-2 * math.log(u[0])), math.sqrt(-2 * math.log(u[2]))
    normal = [r0 * math.cos(2 * math.pi * u[1]),
              r0 * math.sin(2 * math.pi * u[1]),
              r1 * math.cos(2 * math.pi * u[3])]
    spread = math.sqrt((1 - c * c) * kT / mass)
    for v, n in zip(velocity, normal):
        worst = max(worst, abs(v - spread * n) / math.sqrt(kT / mass))
print(f"{count} particles, largest difference {worst:.1e} of sqrt(kT/m): "
      f"{'ok' if worst < 1e-4 else 'FAILED'}")
