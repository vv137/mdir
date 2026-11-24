"""Two opposite unit charges beyond the direct cutoff: all force comes from PME."""
from pathlib import Path
import sys

import mdir
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3] / "scripts"))
from mdir_physical_constants import COULOMB_NATIVE

target, work = sys.argv[1], Path(sys.argv[2])
work.mkdir(parents=True, exist_ok=True)
(work / "pair.top").write_text("""[ defaults ]
1 2 no 1.0 1.0
[ atomtypes ]
I 1 10.0 0.0 A 0.3 0.0
[ moleculetype ]
PAIR 0
[ atoms ]
1 I 1 I P 1  1.0 10.0
2 I 1 I N 2 -1.0 10.0
[ system ]
Charged pair
[ molecules ]
PAIR 1
""")
box = np.array([3.0, 3.4, 3.8])
(work / "pair.gro").write_text("""Charged pair
2
    1I        P    1   0.321   0.432   0.543
    1I        N    2   1.533   1.641   1.765
3.0 3.4 3.8
""")
loaded = mdir.load_gromacs(str(work / "pair.top"), str(work / "pair.gro"))
system, state = loaded.make_system(), loaded.make_state()
system.periodic = True
system.cutoff, system.pairlist_distance = 0.8, 0.9
system.electrostatics = mdir.Electrostatics.PME
system.pme_alpha, system.pme_order, system.pme_grid = 2.0, 8, [40, 48, 50]
execution = mdir.Execution()
execution.target = getattr(mdir.Target, target)
execution.deterministic = True

for precision, rounding in [("Mixed", 2e-5), ("Double", 2e-12)]:
    execution.precision = getattr(mdir.Precision, precision)
    program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                           execution, mdir.Schedule())
    ordinary, observed = mdir.Simulation(program), mdir.Simulation(program)
    ordinary.run(1)
    observed.run(1, energy=True)
    a, b = ordinary.state(), observed.state()
    assert np.array_equal(a.positions, b.positions)
    forces = np.asarray(b.forces, dtype=float)
    relative = np.linalg.norm(np.asarray(a.forces) - forces) / np.linalg.norm(forces)
    assert relative < rounding, (relative, rounding)

    # Sum the reciprocal Ewald series directly at the observed positions.
    axis = np.arange(-14, 15)
    m = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    m = m[np.any(m, axis=1)] / box
    m2 = (m * m).sum(axis=1)
    phase = 2 * np.pi * (m @ (b.positions[0] - b.positions[1]))
    beta = system.pme_alpha
    factor = COULOMB_NATIVE / (np.pi * box.prod()) * np.exp(-(np.pi / beta)**2 * m2) / m2
    terms = factor * (1 - np.cos(phase))
    energy = terms.sum() - 2 * COULOMB_NATIVE * beta / np.sqrt(np.pi)
    virial = terms @ (1 - 2 * (np.pi / beta)**2 * m2)
    force = -2 * np.pi * (factor * np.sin(phase)) @ m
    tol = 2e-5 if precision == "Mixed" else 5e-8
    # The mesh force differentiates the interpolant. Its cancellation error
    # on this pair is larger than the difference between the two schedules.
    np.testing.assert_allclose(forces, [force, -force], rtol=0,
                               atol=2e-3 if precision == "Mixed" else 2e-6)
    np.testing.assert_allclose([b.energies["potential"], b.energies["virial"]],
                               [energy, virial], atol=tol, rtol=tol)
    print(f"{target} {precision}: relative force difference {relative:.3g}, "
          f"energy {energy:.9g} (delta {b.energies['potential'] - energy:.3g}), "
          f"virial {virial:.9g} (delta {b.energies['virial'] - virial:.3g})")
