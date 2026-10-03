"""Brownian dynamics in harmonic wells (D163b).

  brownian_wells.py write DIR N     writes DIR/wells.top, DIR/wells.gro,
                                    and DIR/wells.toml, the term of the
                                    wells: N atoms of argon, each in a well
                                    of its own on a lattice 1.2 nm apart,
                                    beyond the cutoff of one another
  brownian_wells.py check LOG K T DT GAMMA MASS N
                                    the mean energy of the wells in LOG
                                    against that of the discrete step, and
                                    the correlation of energies 10 steps
                                    apart against that of the step

A particle in the well U = k |x - c|^2 / 2 moves by the step of Ermak and
McCammon, x' = (1 - a) x + sqrt(2 kT dt / (m gamma)) R with a = k dt / (m
gamma), whose stationary variance along each axis is (kT / k) / (1 - a/2)
exactly, against kT / k of the canonical distribution. The mean energy of
N particles is then (3/2) N kT / (1 - a/2).
"""
import math
import sys

if sys.argv[1] == "write":
    out, n = sys.argv[2], int(sys.argv[3])
    with open(out + "/wells.top", "w") as f:
        f.write("""[ defaults ]
; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ
  1       2          no         1.0      1.0

[ atomtypes ]
; name  at.num  mass    charge  ptype  sigma  epsilon
  AR    18      39.948  0.0     A      0.34   0.0

[ moleculetype ]
  AR  1

[ atoms ]
  1  AR  1  AR  AR  1  0.0  39.948

[ system ]
Argon in wells

[ molecules ]
AR  %d
""" % n)
    side = round(n ** (1 / 3))
    assert side ** 3 == n
    edge = 1.2 * side
    centers = []
    with open(out + "/wells.gro", "w") as f:
        f.write("Argon in wells\n%5d\n" % n)
        for i in range(n):
            q = (i % side, (i // side) % side, i // side // side)
            center = [1.2 * (c + 0.5) for c in q]
            centers.append(center)
            f.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"
                    % (i + 1, "AR", "AR", (i + 1) % 100000, *center))
        f.write("%10.5f%10.5f%10.5f\n" % (edge, edge, edge))
    # The centers in Å, a parameter of each particle.
    with open(out + "/wells.toml", "w") as f:
        f.write("[[energy.external]]\n"
                "name       = \"well\"\n"
                "expression = \"0.5*k*((x-cx)^2 + (y-cy)^2 + (z-cz)^2)\"\n"
                "k          = 2.0\n"
                "selection  = \"@*\"\n")
        for axis, name in enumerate(("cx", "cy", "cz")):
            f.write("%s = [%s]\n" % (name, ", ".join(
                "%.1f" % (10 * c[axis]) for c in centers)))
    sys.exit(0)

log, k, temp, dt, gamma, mass, n = sys.argv[2:9]
k, temp, dt, gamma, mass, n = (float(k), float(temp), float(dt), float(gamma),
                               float(mass), int(n))
energies = []
for line in open(log):
    words = line.split()
    if len(words) > 4 and words[0] == "INFO:" and words[1].isdigit():
        if int(words[1]) >= 2000:
            energies.append(float(words[4]))
kcal = 4.184
kT = 0.0083144626181532 * temp
# k in kcal/mol/A^2 to kJ/mol/nm^2.
a = k * kcal * 100 * dt / (mass * gamma)
exact = 1.5 * n * kT / (1 - a / 2) / kcal
canonical = 1.5 * n * kT / kcal
mean = sum(energies) / len(energies)
count = len(energies)
var = sum((e - mean) ** 2 for e in energies) / (count - 1)
# The correlation of energies 10 steps apart. The energy is a sum of squares
# of positions that are correlated by rho = (1 - a)^10 over 10 steps, so the
# energies are by rho^2: the mobility 1 / (m gamma) sets it.
c = sum((u - mean) * (w - mean)
        for u, w in zip(energies[:-1], energies[1:])) / (count - 1) / var
expected = (1 - a) ** 20
# The errors of the mean and of the correlation of a process of the first
# order with that correlation (Bartlett).
error = math.sqrt(var / count * (1 + c) / (1 - c))
spread = math.sqrt((1 - c * c) / count)
print("a = %.4f, %d samples" % (a, count))
print("mean energy %.3f +- %.3f kcal/mol, discrete step %.3f, canonical %.3f"
      % (mean, error, exact, canonical))
print("within 4 errors of the discrete step: %s"
      % ("yes" if abs(mean - exact) < 4 * error else "no"))
print("correlation 10 steps apart %.4f +- %.4f, expected %.4f: %s"
      % (c, spread, expected,
         "agrees" if abs(c - expected) < 4 * spread else "differs"))
