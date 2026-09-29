"""Reference values for charged.toml.

Evaluates, over all pairs of the particles of mixture.pdb and in the units
of the control file:

  - Lennard-Jones with Lorentz-Berthelot mixing, except for the pair of
    types AR and KR, whose sigma and epsilon the control file sets (NBFIX),
    cut at the cutoff with no shift;
  - Coulomb with the product of the charges and the constant of CODATA
    2018, cut at the cutoff with no shift;
  - the correction for the dispersion beyond the cutoff of the
    Lennard-Jones term: its r^-6 part only, for a uniform density, over the
    N (N - 1) ordered pairs of particles (Allen and Tildesley, 2017; the
    GROMACS manual, "Long range Van der Waals interactions"). The virial of
    the correction is 6 times its energy.

It prints the potential energy and the trace of the virial W = sum of
d (x) F over the pairs (B8), as the log has them at the start.

Usage: python3 charged_reference.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mixture_reference import EDGE, read  # noqa: E402

CUTOFF = 9.0
TYPES = {"AR": (0.238465, 3.4, 0.25), "KR": (0.3253, 3.65, -0.25)}
NBFIX = {("AR", "KR"): (0.2, 3.3)}
# kcal Å mol^-1 e^-2: 138.935457644 kJ nm mol^-1 e^-2 of CODATA 2018.
COULOMB = 138.935457644 * 10.0 / 4.184


def pair_parameters(a, b):
    """epsilon and sigma of the pair of types a and b."""
    key = (a, b) if (a, b) in NBFIX else (b, a)
    if key in NBFIX:
        return NBFIX[key]
    ea, sa, _ = TYPES[a]
    eb, sb, _ = TYPES[b]
    return math.sqrt(ea * eb), 0.5 * (sa + sb)


def main():
    particles = read()
    energy = 0.0
    virial = 0.0
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            a, xi = particles[i]
            b, xj = particles[j]
            d = [xi[k] - xj[k] for k in range(3)]
            d = [c - EDGE * round(c / EDGE) for c in d]
            r = math.sqrt(sum(c * c for c in d))
            if r >= CUTOFF:
                continue
            epsilon, sigma = pair_parameters(a, b)
            s6 = (sigma / r) ** 6
            lj = 4.0 * epsilon * (s6 * s6 - s6)
            lj_slope = -24.0 * epsilon * (2.0 * s6 * s6 - s6) / r
            coulomb = COULOMB * TYPES[a][2] * TYPES[b][2] / r
            energy += lj + coulomb
            # d . F with F = -u'(r) d / r.
            virial += -(lj_slope - coulomb / r) * r

    # The correction for the dispersion.
    counts = {}
    for name, _ in particles:
        counts[name] = counts.get(name, 0) + 1
    volume = EDGE ** 3
    tail_energy = 0.0
    for a in counts:
        for b in counts:
            epsilon, sigma = pair_parameters(a, b)
            c6 = 4.0 * epsilon * sigma ** 6
            pairs = counts[a] * (counts[b] - (1 if a == b else 0))
            tail_energy += -2.0 * math.pi / (3.0 * volume) * pairs * c6 / CUTOFF ** 3
    tail_virial = 6.0 * tail_energy

    print("potential energy without the correction %.6f" % energy)
    print("correction of the energy                %.6f" % tail_energy)
    print("potential energy                        %.6f" % (energy + tail_energy))
    print("trace of the virial                     %.6f" % (virial + tail_virial))


if __name__ == "__main__":
    main()
