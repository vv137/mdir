"""Reference values for mixture.toml.

Evaluates the energy, the forces, and the virial of the particles of
mixture.pdb over all pairs, in the units of the control file, and the
temperature and the pressure that the log has at the start.

The initial velocities have the temperature of the control file in the
kinetic energy of the velocities, K. The log estimates the temperature
and the pressure from the kinetic energies half a step before and after
as well, whose mean exceeds K by (dt^2 / 8) sum F^2 / m:

    temperature   from K + 2/3 of the excess
    pressure      from K + the excess

Usage: python3 mixture_reference.py
"""

import math
import os

EDGE = 23.2
CUTOFF = 9.0
SWITCH_FROM = 7.5
TEMPERATURE = 120.0
TIMESTEP = 0.004
# amu, as the types of the control file have them.
MASSES = {"AR": 39.95, "KR": 83.80}
# epsilon in kcal/mol, sigma in Å.
TYPES = {"AR": (0.238465, 3.4), "KR": (0.3253, 3.65)}

# kcal/(mol K), and atm in one kcal/(mol Å^3).
BOLTZMANN = 0.0083144626181532 / 4.184
ATM = 4.184 * 1000.0 * 16.6053906717 / 1.01325
# kcal/mol in one amu Å^2 / ps^2.
KINETIC = 0.01 / 4.184


def read():
    """The names and the positions of the particles."""
    path = os.path.join(os.path.dirname(__file__), "mixture.pdb")
    particles = []
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")):
            name = line[12:16].strip()
            position = [float(line[30 + 8 * k:38 + 8 * k]) for k in range(3)]
            particles.append((name, position))
    return particles


def switch(r):
    """The switching function and its derivative."""
    if r <= SWITCH_FROM:
        return 1.0, 0.0
    width = CUTOFF - SWITCH_FROM
    t = (r - SWITCH_FROM) / width
    value = 1.0 - 10.0 * t**3 + 15.0 * t**4 - 6.0 * t**5
    slope = (-30.0 * t**2 + 60.0 * t**3 - 30.0 * t**4) / width
    return value, slope


def main():
    particles = read()
    count = len(particles)
    energy = 0.0
    trace = 0.0
    forces = [[0.0, 0.0, 0.0] for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            d = []
            for k in range(3):
                component = particles[i][1][k] - particles[j][1][k]
                component -= EDGE * round(component / EDGE)
                d.append(component)
            r = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            if r >= CUTOFF:
                continue
            # Lorentz-Berthelot: the mean of the two sigma, and the
            # geometric mean of the two epsilon.
            first, second = TYPES[particles[i][0]], TYPES[particles[j][0]]
            epsilon = math.sqrt(first[0] * second[0])
            sigma = 0.5 * (first[1] + second[1])
            s6 = (sigma / r) ** 6
            u = 4.0 * epsilon * (s6 * s6 - s6)
            du = -(24.0 * epsilon / r) * (2.0 * s6 * s6 - s6)
            s, ds = switch(r)
            energy += u * s
            # d . K, with the force K on i due to j.
            trace += -(du * s + u * ds) * r
            for k in range(3):
                force = -(du * s + u * ds) * d[k] / r
                forces[i][k] += force
                forces[j][k] -= force

    # sum F^2 / m in (kcal/mol)^2 / (Å^2 amu), and with dt^2 in kcal/mol.
    square = sum(sum(c * c for c in force) / MASSES[particle[0]]
                 for force, particle in zip(forces, particles))
    excess = 0.125 * TIMESTEP**2 * square / KINETIC

    freedom = 3 * count - 3
    kinetic = 0.5 * freedom * BOLTZMANN * TEMPERATURE
    half = kinetic + excess
    optimal = kinetic + 2.0 * excess / 3.0
    temperature = 2.0 * optimal / (freedom * BOLTZMANN)
    pressure = (2.0 * half + trace) / (3.0 * EDGE**3) * ATM
    print("particles        ", count)
    print("potential energy ", repr(energy), "kcal/mol")
    print("kinetic energy   ", repr(kinetic), "kcal/mol")
    print("  at half steps  ", repr(half), "kcal/mol")
    print("temperature      ", repr(temperature), "K")
    print("virial, trace    ", repr(trace), "kcal/mol")
    print("pressure         ", repr(pressure), "atm")


if __name__ == "__main__":
    main()
