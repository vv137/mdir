"""One cluster of 45 particles (44 neighbors per row), plus isolated atoms."""
import pathlib
import sys

import mdir
import numpy as np

work = pathlib.Path(sys.argv[1])
gro = work / "system.gro"
lines = gro.read_text().splitlines()
for i in range(1000):
    if i < 45:
        xyz = (1 + 0.18 * (i % 4), 1 + 0.18 * (i // 4 % 4), 1 + 0.18 * (i // 16))
    else:
        j = i - 45
        xyz = (5 + 2 * (j % 10), 5 + 2 * (j // 10 % 10), 5 + 2 * (j // 100))
    lines[2 + i] = lines[2 + i][:20] + "".join(f"{x:8.3f}" for x in xyz)
gro.write_text("\n".join(lines) + "\n")
loaded = mdir.load_gromacs(str(work / "plain.top"), str(gro))


def evaluate(width, deterministic, precision):
    system = loaded.make_system()
    system.cutoff, system.pairlist_distance = 1.0, 1.1
    execution = mdir.Execution()
    execution.target = mdir.Target.GPU
    execution.precision = getattr(mdir.Precision, precision)
    execution.neighbor_structure = mdir.NeighborStructure.Matrix
    execution.neighbor_capacity = width
    execution.deterministic = deterministic
    ensemble = mdir.Ensemble()
    ensemble.temperature = 0.0
    program = mdir.compile(system, loaded.make_state(), mdir.Integrator(),
                           ensemble, execution, mdir.Schedule())
    simulation = mdir.Simulation(program)
    simulation.run(0, energy=True)
    state = simulation.state()
    return np.array(state.forces), np.array(state.energies["potential"])


for precision, tolerance in (("Mixed", 2e-5), ("Double", 1e-12)):
    for deterministic in (True, False):
        reference = evaluate(64, deterministic, precision)
        narrow = evaluate(8, deterministic, precision)
        assert np.any(reference[0]) and np.isfinite(reference[0]).all()
        for actual, expected in zip(narrow, reference):
            if deterministic:
                assert actual.tobytes() == expected.tobytes()
            else:
                assert np.max(np.abs(actual - expected)) <= tolerance * np.max(np.abs(expected))
        if deterministic:
            repeat = evaluate(8, True, precision)
            assert narrow[0].tobytes() == repeat[0].tobytes()

        if deterministic:
            # The first build needs 1,935 records at width 1, but holds 1,000.
            recovered = evaluate(1, True, precision)
            for actual, expected in zip(recovered, reference):
                assert actual.tobytes() == expected.tobytes()
