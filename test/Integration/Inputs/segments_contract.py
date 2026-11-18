"""The contract of md_alg.traverse_pairs over a list of units
(D421; D417, Section 6): its SIMT schedule on a device
against its serial definition, which convert-md-exec-to-gpu runs on the host
over copies of the same lists (MDIR_TRAVERSE_ORACLE), in D415's five modes,
on the test route of the pair term over the segments of rigid kinds
(MDIR_RIGID_UNITS_TEST).

Usage: segments_contract.py NAME TOPOLOGY COORDINATES KINDS WHAT [CUTOFF PAIRLIST PRUNED]

  TOPOLOGY     a topology of GROMACS (.top) or of Amber (.prmtop), and its
               coordinates (.gro or .inpcrd)
  KINDS        the rigid kinds taken, as `mdir check` numbers them
  WHAT         outer or dual: the records of the list, or the inner records
               of a dual list pruned in its refresh
  CUTOFF ...   the cutoff, the pair list distance, and the pruned distance
               in nm, 0.8, 1.0, and 0.86 by default; a small cell takes
               less, since the covering reach of the list must be within
               half its least face width (D419)

As traverse_contract.py (D415, Section 10): each run evaluates the energy,
the virial, and the forces at the start in mixed precision, with Coulomb by
a plain cutoff and the loops over tuples a row a particle, so that what
differs between two runs is the traversal's alone. Five runs: simt, serial,
serial-reversed, serial-magnitudes, omitted. For each component X, the
magnitude is M = X(magnitudes) - X(omitted). A member's term is a product
of a weight of W and a slice's value, rounded once, so two evaluations of n
terms differ by at most 2 gamma_n M (D417, Section 6), gamma_n = n u /
(1 - n u), u = 2^-24, plus D415's floor of 4 u max M. n bounds the terms of
a component: the places within the reach of a unit, at most 0.11 per cubic
Angstrom over a sphere of 11 Angstrom, 613, times the places of a unit, at
most 4: n = 4096.

It prints the largest ratio of a difference to its bound, and fails if one
exceeds 1, if the traversal contributes nothing, or if the magnitudes do not
bound the sums. The potential and the total virial (its constraint part
included) of the general groups run are held to the serial definition's
within the same bound: they do not depend on the route, while the forces of
the atoms and the virial without its constraint part do where a site is not
a linear function of its frame (D421, Section 5).
"""
import os
import sys

import numpy as np

os.environ["MDIR_COMPILE_CACHE"] = "off"
import mdir  # noqa: E402

name, topology, coordinates, kinds, what = sys.argv[1:6]
cutoff, pairlist, pruned = (map(float, sys.argv[6:9]) if len(sys.argv) > 8
                            else (0.8, 1.0, 0.86))
TERMS = 4 * 1024
os.environ["MDIR_RIGID_UNITS_TEST"] = kinds


def build():
    if topology.endswith(".prmtop"):
        loaded = mdir.load_amber(topology, coordinates)
    else:
        loaded = mdir.load_gromacs(topology, coordinates)
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff = cutoff
    system.pairlist_distance = pairlist
    if what == "dual":
        system.pruned_distance = pruned
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.periodic = True
    system.rigid_water = True
    system.rigid_hydrogen_bonds = True
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep = 0.001
    ensemble.kind = mdir.EnsembleKind.NVE
    execution.target = mdir.Target.GPU
    execution.precision = mdir.Precision.Mixed
    execution.neighbor_structure = mdir.NeighborStructure.Groups
    return system, state, integrator, ensemble, execution


def evaluate(oracle):
    os.environ["MDIR_TRAVERSE_ORACLE"] = oracle
    program = mdir.compile(*build(), mdir.Schedule())
    simulation = mdir.Simulation(program)
    simulation.run(0, energy=True)
    at = simulation.state()
    energies = at.energies
    return np.concatenate([np.asarray(at.forces, dtype=np.float64).ravel(),
                           [energies["potential"], energies["virial"]]])


runs = {oracle: evaluate(oracle)
        for oracle in ("simt", "serial", "serial-reversed", "serial-magnitudes", "omitted")}
# The general groups run, for the quantities that do not depend on how the
# forces of the sites reach the atoms: the potential and the total virial,
# its constraint part included (the parts are split otherwise where a site
# is not a linear function of its frame, as Amber's extra points are).
os.environ.pop("MDIR_RIGID_UNITS_TEST")
general = evaluate("simt")
os.environ["MDIR_RIGID_UNITS_TEST"] = kinds
terms = runs["serial"] - runs["omitted"]
magnitude = runs["serial-magnitudes"] - runs["omitted"]
u = 2.0 ** -24
gamma = TERMS * u / (1 - TERMS * u)
floor = 4 * u * float(np.abs(magnitude).max())
assert float(np.abs(terms).max()) > 0, "the traversal contributes nothing"
slack = 2 * gamma * np.abs(magnitude) + floor
def where_of(k, count):
    return ("potential" if k == count - 2 else
            "virial" if k == count - 1 else f"force {k // 3}.{k % 3}")


excess = np.abs(terms) - (np.abs(magnitude) + slack)
assert np.all(excess <= 0), \
    f"the magnitudes do not bound the sums at {where_of(int(np.argmax(excess)), len(excess))}: " \
    f"{terms[int(np.argmax(excess))]:.6g} against {magnitude[int(np.argmax(excess))]:.6g}"
failed = False
for other in ("simt", "serial-reversed"):
    difference = np.abs(runs[other] - runs["serial"])
    ratio = difference / slack
    worst = int(np.argmax(ratio))
    where = ("potential" if worst == len(ratio) - 2 else
             "virial" if worst == len(ratio) - 1 else f"force {worst // 3}.{worst % 3}")
    print(f"{name} {what} {other}: largest difference "
          f"{float(ratio[worst]):.3g} of its bound ({where})", file=sys.stderr)
    if float(ratio.max()) > 1.0:
        failed = True
        print(f"{name} {what} {other}: beyond the bound at {where}: "
              f"{float(difference[worst]):.3g} > {float(slack[worst]):.3g}")
# The general run differs in the order of its sums and in how it forms the
# displacements of the sites, within the same model of a reordered sum.
difference = np.abs(general[-2:] - runs["serial"][-2:])
for k, what_ in ((0, "potential"), (1, "total virial")):
    ratio = float(difference[k] / slack[-2 + k])
    print(f"{name} {what} general: {what_} differs by {ratio:.3g} of its bound",
          file=sys.stderr)
    if ratio > 1.0:
        failed = True
        print(f"{name} {what} general: the {what_} is beyond the bound: "
              f"{float(difference[k]):.6g} > {float(slack[-2 + k]):.6g}")
if failed:
    sys.exit(1)
print(f"{name} {what}: the schedule and the reversed visit keep the serial "
      f"order's sums within the bound")
