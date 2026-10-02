"""Fits the rate at which the kinetic energy of an ideal gas relaxes to that
of the bath under Langevin dynamics, K(t) = K_eq + (K_0 - K_eq) e^(-r t),
whose rate is r = 2 gamma. Reads the rows of the logs of the cold start and
of the warm run, takes the rows of the first TIME ps, and says whether the
rate is within TOLERANCE of RATE.

    fit_relaxation.py COLD_LOG WARM_LOG PARTICLES TEMPERATURE TIME RATE
        TOLERANCE"""

import math
import sys


def rows(path):
    out = []
    for line in open(path):
        f = line.split()
        if len(f) > 5 and f[0] == "INFO:" and f[1].isdigit():
            out.append((float(f[2]), float(f[5])))
    return out


cold, warm = rows(sys.argv[1]), rows(sys.argv[2])
particles, temperature, span = (int(sys.argv[3]), float(sys.argv[4]),
                                float(sys.argv[5]))
expected, tolerance = float(sys.argv[6]), float(sys.argv[7])
equilibrium = 1.5 * particles * 0.0083144626181532 / 4.184 * temperature
start, k0 = cold[-1]
points = [(t - start, math.log((equilibrium - k) / (equilibrium - k0)))
          for t, k in warm if t - start <= span and k < equilibrium]
n = len(points)
mt = sum(t for t, _ in points) / n
my = sum(y for _, y in points) / n
slope = (sum((t - mt) * (y - my) for t, y in points) /
         sum((t - mt) ** 2 for t, _ in points))
ok = abs(-slope - expected) <= tolerance
print(f"rate {-slope:.2f} /ps from {n} rows: {'ok' if ok else 'FAILED'}")
