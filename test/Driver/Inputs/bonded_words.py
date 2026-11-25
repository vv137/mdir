"""Dyadic spring forces at a half word and at the signed i64 limits."""
import os
from pathlib import Path
import subprocess
import sys

binary, directory = sys.argv[1:3]
work = Path(directory)
work.mkdir(parents=True, exist_ok=True)
(work / "atoms.gro").write_text(
    "Three collinear atoms\n3\n" + "".join(
        f"{1:5}AR   {'AR':>5}{i:5}{x:8.3f}{1.:8.3f}{1.:8.3f}\n"
        for i, x in enumerate((1.0, 1.25, 1.5), 1)
    ) + "3.0 3.0 3.0\n")
(work / "run.toml").write_text('''[input]
format = "GROMACS"
topology = "atoms.top"
coordinates = "atoms.gro"
[output]
energy_interval = 1
[energy]
cutoff = 8.0
pairlist_distance = 10.0
[dynamics]
integrator = "VELOCITY_VERLET"
steps = 1
time_step = 0.000001
seed = 314159
[ensemble]
ensemble = "NVE"
temperature = 0.0
[boundary]
type = "PERIODIC"
[constraints]
hydrogen_bonds = false
rigid_water = false
[execution]
precision = "MIXED"
target = "GPU"
''')
topology = '''[ defaults ]
1 2 no 1.0 1.0
[ atomtypes ]
AR 18 39.948 0.0 A 0.34 0.0
[ moleculetype ]
chain 1
[ atoms ]
1 AR 1 AR AR 1 0.0 39.948
2 AR 1 AR AR 2 0.0 39.948
3 AR 1 AR AR 3 0.0 39.948
[ bonds ]
1 2 1 {r1} {k}
1 3 1 {r2} {k}
[ system ]
Springs
[ molecules ]
chain 1
'''
def springs(k, stretch):
    (work / "atoms.top").write_text(topology.format(
        k=k, r1=0.25 - stretch, r2=0.5 - stretch))


if sys.argv[3:] == ["energy"]:
    import mdir
    import numpy as np

    # Each spring carries 1.5 energy words. Rounding each term gives 4
    # words in total, whereas rounding after summation would give 3.
    springs(3 * 2**-26, 0.125)
    loaded = mdir.load_gromacs(str(work / "atoms.top"), str(work / "atoms.gro"))
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 1.0
    run = mdir.Execution()
    run.target = mdir.Target.GPU
    forces = []
    for precision, mode, words in (
        (mdir.Precision.Mixed, "workunits-check", 4),
        (mdir.Precision.Mixed, "", 3),
        (mdir.Precision.Double, "workunits-check", 3),
    ):
        os.environ["MDIR_TUPLE_SUMS"] = mode
        run.precision = precision
        program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                               run, mdir.Schedule())
        assert ("bonded-units=" in program.pipeline) == (words == 4)
        sim = mdir.Simulation(program)
        sim.run(0, energy=True)
        first = sim.state()
        forces.append(np.array(first.forces))
        if words == 4:
            for _ in range(2):
                repeated = mdir.Simulation(program)
                repeated.run(0, energy=True)
                at = repeated.state()
                assert at.forces.tobytes() == first.forces.tobytes()
                for observable in ("potential", "virial"):
                    assert at.energies[observable].hex() == first.energies[observable].hex()
        actual = first.energies["potential"] * 2**32
        assert abs(actual - words) < 1e-5, (actual, words)
        print("energy words", words)
    # The two outer atoms have one slot and the middle of the two bonds
    # (atom zero in file order) has two. The continuous spring forces are
    # nonzero even though each rounds to zero on the Q18 grid.
    delta = np.abs(forces[0] - forces[1])
    assert delta.max() > 0
    assert np.all(delta <= np.array([[2], [1], [1]]) * 2**-19)
    print("mixed bonded force, energy and virial identical across three runs")
    sys.exit(0)

# A short stretch isolates force overflow: even the largest spring here
# has energy 2^28 kJ/mol, within the Q32 range.
for name, k, stretch, mode in (
        ("ties", 12 * 2**-18, 0.125, "workunits-check"),
        ("wide component", 2**17, 0.125, "workunits-check"),
        ("component overflow", 2**61, 2**-16, "workunits"),
        ("shared sum overflow", 2**60, 2**-16, "workunits"),
        ("global sum overflow", 2**60, 2**-16, "integer-direct"),
        ("infinite contribution", 2**130, 0.125, "workunits"),
        ("NaN contribution", 2**130, 0.0, "workunits"),
        ("energy term overflow", 2**38, 0.125, "workunits"),
        ("energy sum overflow", 2**37, 0.125, "workunits")):
    springs(k, stretch)
    env = dict(os.environ, MDIR_TUPLE_SUMS=mode, MDRT_WAIT="1")
    run = subprocess.run([binary, "run", "run.toml"], cwd=work, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if mode != "workunits-check":
        assert run.returncode and any(s in run.stdout for s in (
            "CUDA_ERROR_ILLEGAL_INSTRUCTION", "CUDA_ERROR_LAUNCH_FAILED")), run.stdout
    else:
        assert run.returncode == 0 and "0 word mismatches" in run.stdout, run.stdout
    print(name, "passed")
