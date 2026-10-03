"""Independent VSWITCH energy and analytic derivative, in kcal/mol and Å.

Generate isolated ordinary and 1-4 pairs around both boundaries, then check
the actual run's energies, checkpoint forces, and diagonal virial. The
reference uses the expanded polynomial in r², independently of the
normalized cubic emitted by the driver. Central differences also check
the analytic slope. All particles start at rest and take one 1e-9 ps step,
so the final forces equal the initial ones to the printed precision.
"""

import argparse
from pathlib import Path
import re
import subprocess


DISTANCES = [6.0, 9.999, 10.0, 10.001, 10.5, 11.0, 11.999,
             12.0, 12.001, 13.0]
COULOMB = 138.935457644 / 4.184 * 10


def potential(r, sigma, epsilon):
    s6 = (sigma / r)**6
    u = 4 * epsilon * (s6*s6 - s6)
    du = 24 * epsilon * (s6 - 2*s6*s6) / r
    if r <= 10:
        return u, du
    if r >= 12:
        return 0.0, 0.0
    a, b = 144 - r*r, 144 + 2*r*r - 300
    switch = a*a*b / 44**3
    ds = (-4*r*a*b + 4*r*a*a) / 44**3
    return u*switch, du*switch + u*ds


def prepare(directory, target, precision, neighbors):
    directory.mkdir(parents=True, exist_ok=True)
    topology = """[ defaults ]
1 2 yes 1.0 1.0
[ atomtypes ]
A 18 40.0 0.0 A 0.34 4.184
Z 18 40.0 0.0 A 0.0 0.0
[ moleculetype ]
PAIR 1
[ atoms ]
1 A 1 PAIR A1 1 0.1 40.0
2 A 1 PAIR A2 1 -0.1 40.0
[ moleculetype ]
CHAIN 3
[ atoms ]
1 A 1 CHAIN A1 1 0.1 40.0
2 Z 1 CHAIN Z2 1 0.0 40.0
3 Z 1 CHAIN Z3 1 0.0 40.0
4 A 1 CHAIN A4 1 -0.1 40.0
[ bonds ]
1 2 1 0.15 0.0
2 3 1 0.15 0.0
3 4 1 0.15 0.0
[ pairs ]
1 4 1 0.36 2.092
[ system ]
squared-distance switch
[ molecules ]
PAIR 10
CHAIN 10
"""
    (directory / "system.top").write_text(topology)
    rows = []
    for chain in (False, True):
        for j, r in enumerate(DISTANCES):
            xs = [10, 11.5, 13, 10+r] if chain else [10, 10+r]
            for x in xs:
                i = len(rows) + 1
                rows.append(f"{j+1:5d}{'TST':<5}{'A':>5}{i:5d}"
                            f"{x/10:12.8f}{(30*j+10)/10:12.8f}"
                            f"{(40 if chain else 10)/10:12.8f}")
    (directory / "system.gro").write_text(
        "isolated pairs\n60\n" + "\n".join(rows) + "\n8.0 32.0 8.0\n")
    (directory / "run.toml").write_text(f"""[input]
topology = "system.top"
coordinates = "system.gro"
[output]
energy_interval = 1
checkpoint = "state.h5"
checkpoint_interval = 1
[energy]
cutoff = 12.0
switch_distance = 10.0
pairlist_distance = 13.5
lennard_jones_modifier = "SQUARED_DISTANCE_SWITCH"
dispersion_correction = "NONE"
electrostatics = "CUTOFF"
[constraints]
hydrogen_bonds = false
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 1e-9
steps = 1
[ensemble]
ensemble = "NVE"
temperature = 0.0
[boundary]
type = "PERIODIC"
[execution]
target = "{target}"
precision = "{precision}"
neighbor_structure = "{neighbors}"
fast_math = false
""")


def check(mdir, directory, precision):
    log = subprocess.check_output([mdir, "run", str(directory / "run.toml")],
                                  text=True)
    (directory / "run.log").write_text(log)
    forces = subprocess.check_output(
        [mdir, "checkpoint", "--print=forces", str(directory / "state.h5")],
        text=True)
    observed = [[float(v) / 41.84 for v in line.split()[2:5]]
                for line in forces.splitlines()]
    expected = []
    energies = [0.0] * 4
    virial = 0.0
    finite_difference = 0.0
    for chain in (False, True):
        for r in DISTANCES:
            u, du = potential(r, 3.6 if chain else 3.4,
                              0.5 if chain else 1.0)
            h = 1e-5
            fd = (potential(r+h, 3.6 if chain else 3.4,
                            0.5 if chain else 1.0)[0] -
                  potential(r-h, 3.6 if chain else 3.4,
                            0.5 if chain else 1.0)[0]) / (2*h)
            finite_difference = max(finite_difference, abs(fd-du))
            q = -0.01 * COULOMB if chain or r < 12 else 0.0
            energies[2 if chain else 0] += u
            energies[3 if chain else 1] += q/r
            slope = du - q/r**2
            virial -= r*slope
            expected.append([slope, 0, 0])
            if chain:
                expected.extend([[0, 0, 0], [0, 0, 0]])
            expected.append([-slope, 0, 0])
    assert len(observed) == len(expected) == 60
    error_force = max(abs(a-b) for o, e in zip(observed, expected)
                      for a, b in zip(o, e))
    names = ["Lennard-Jones", "Coulomb", "Lennard-Jones 1-4", "Coulomb 1-4"]
    error_energy = 0.0
    for name, reference in zip(names, energies):
        actual = float(re.search(r"MDIR:\s+" + re.escape(name) +
                                 r"\s+([-+0-9.eE]+)", log)[1])
        error_energy = max(error_energy, abs(actual-reference))
        print(f"{name}: reference {reference:.12g}, observed {actual:.12g}")
    actual_virial = [float(v) for v in re.search(
        r"virial at the start.*\n.*?MDIR:\s+(\S+)\s+(\S+)\s+(\S+)",
        log).groups()]
    error_virial = max(abs(actual_virial[0]-virial),
                       abs(actual_virial[1]), abs(actual_virial[2]))
    # Absolute tolerances include six-decimal energies/virials and printed
    # checkpoint forces; mixed allows f32 kernel and distance rounding.
    tolerance = 1e-6 if precision == "double" else 2e-5
    print(f"virial: reference {virial:.12g}, observed {actual_virial[0]:.12g}")
    print(f"differences: energy {error_energy:.3g}, force {error_force:.3g}, "
          f"virial {error_virial:.3g}; tolerance {tolerance:.3g}")
    assert finite_difference < 1e-7, finite_difference
    assert max(error_energy, error_force, error_virial) < tolerance
    print("squared-distance switch: energy, forces, virial, boundaries, "
          "1-4 pairs, and finite differences passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mdir")
    parser.add_argument("directory", type=Path)
    parser.add_argument("target", choices=["cpu", "gpu"])
    parser.add_argument("precision", choices=["mixed", "double"])
    parser.add_argument("neighbors", choices=["MATRIX", "GROUPS"])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    prepare(args.directory, args.target, args.precision, args.neighbors)
    if not args.prepare_only:
        check(args.mdir, args.directory, args.precision)
