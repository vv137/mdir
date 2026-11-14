"""The contract of md_alg.traverse_pairs (D[md-alg-traverse]): its SIMT
schedule on a device against its serial definition, which
convert-md-exec-to-gpu runs on the host over copies of the same lists
(MDIR_TRAVERSE_ORACLE).

Usage: traverse_contract.py SYSTEM WHAT PRECISION

  SYSTEM     dodecahedron TOPOLOGY GRO, or jac PRMTOP INPCRD
  WHAT       outer, dual, or fused: the list of groups, the inner list of a
             dual list pruned in the refresh, or pruned in the loop
             (D284, MDIR_FUSED_PRUNE=always)
  PRECISION  Mixed or Double

Each run evaluates the energy, the virial, and the forces at the start
(`run(0, energy=True)`), with Coulomb by a plain cutoff, so that no
reciprocal sum spreads with atomics, and with the loops over tuples taking
a row a particle (`tuples-once=false`, which the oracle sets): what differs
between two runs is then the traversal's alone. Five runs:

  simt               the schedule;
  serial             the serial definition, in the order of the entries;
  serial-reversed    the same, every loop of the visit reversed;
  serial-magnitudes  with |t| in place of each term t;
  omitted            with no term: the rest of the program.

For each component X (each force component, the potential, the virial),
the traversal's terms t_k have the magnitude M = X(magnitudes) -
X(omitted) = sum |t_k|. Two orders of a sum of n terms in a type of unit
roundoff u differ by at most 2 gamma_{n-1} M, gamma_m = m u / (1 - m u)
(the bound of D406 for a reordered sum). The test holds the schedule and
the reversed visit to the serial order within that bound, with n = 1024:
the particles within the reach of one, at most 0.11 per cubic Angstrom
over a sphere of 11 Angstrom, 613, bound the terms of a force, and a
lane's sums over its units bound those of an energy. Mixed precision sums
in f32 (u = 2^-24), double in f64 (u = 2^-53). A floor of 4 u max M covers
components whose terms cancel to nothing.

It prints the largest ratio of a difference to its bound, and fails if one
exceeds 1. It also fails if the traversal contributes nothing, or if its
magnitudes do not bound its sums, so that a test of nothing cannot pass.
"""
import os
import sys

import numpy as np

os.environ["MDIR_COMPILE_CACHE"] = "off"
import mdir  # noqa: E402

system_name, first, second, what, precision = sys.argv[1:6]
TERMS = 1024


def build():
    if system_name == "dodecahedron":
        loaded = mdir.load_gromacs(first, second)
        cutoff, pairlist, pruned = 0.8, 1.0, 0.86
    else:
        loaded = mdir.load_amber(first, second)
        cutoff, pairlist, pruned = 0.8, 1.1, 0.86
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff = cutoff
    system.pairlist_distance = pairlist
    if what != "outer":
        system.pruned_distance = pruned
    system.electrostatics = mdir.Electrostatics.Cutoff
    system.periodic = True
    system.rigid_water = True
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep = 0.001
    ensemble.kind = mdir.EnsembleKind.NVE
    execution.target = mdir.Target.GPU
    execution.precision = getattr(mdir.Precision, precision)
    execution.neighbor_structure = mdir.NeighborStructure.Groups
    return system, state, integrator, ensemble, execution


def evaluate(oracle):
    os.environ["MDIR_TRAVERSE_ORACLE"] = oracle
    # The pruning in the loop from any number of groups, or none at all.
    os.environ["MDIR_FUSED_PRUNE"] = "always" if what == "fused" else "off"
    program = mdir.compile(*build(), mdir.Schedule())
    fused = "fuse-pruning-groups=0" in program.pipeline
    assert fused == (what == "fused" and oracle == "simt"), program.pipeline
    simulation = mdir.Simulation(program)
    simulation.run(0, energy=True)
    at = simulation.state()
    energies = at.energies
    return np.concatenate([np.asarray(at.forces, dtype=np.float64).ravel(),
                           [energies["potential"], energies["virial"]]])


runs = {oracle: evaluate(oracle)
        for oracle in ("simt", "serial", "serial-reversed", "serial-magnitudes", "omitted")}
terms = runs["serial"] - runs["omitted"]
magnitude = runs["serial-magnitudes"] - runs["omitted"]
u = 2.0 ** -24 if precision == "Mixed" else 2.0 ** -53
gamma = (TERMS - 1) * u / (1 - (TERMS - 1) * u)
floor = 4 * u * float(np.abs(magnitude).max())
assert float(np.abs(terms).max()) > 0, "the traversal contributes nothing"
slack = 2 * gamma * np.abs(magnitude) + floor
assert np.all(np.abs(terms) <= np.abs(magnitude) + slack), \
    "the magnitudes do not bound the sums"
bound = slack
failed = False
for other in ("simt", "serial-reversed"):
    difference = np.abs(runs[other] - runs["serial"])
    ratio = difference / bound
    worst = int(np.argmax(ratio))
    where = ("potential" if worst == len(ratio) - 2 else
             "virial" if worst == len(ratio) - 1 else f"force {worst // 3}.{worst % 3}")
    print(f"{system_name} {what} {precision} {other}: largest difference "
          f"{float(ratio[worst]):.3g} of its bound ({where})", file=sys.stderr)
    if float(ratio.max()) > 1.0:
        failed = True
        print(f"{system_name} {what} {precision} {other}: beyond the bound at {where}: "
              f"{float(difference[worst]):.3g} > {float(bound[worst]):.3g}")
if failed:
    sys.exit(1)
print(f"{system_name} {what} {precision}: the schedule and the reversed visit "
      f"keep the serial order's sums within the bound")
