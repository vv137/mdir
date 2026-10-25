"""The seed in the fingerprint of a Python checkpoint is exact (#620), and is
the text of the control file for the same seed.

Usage: python_fingerprint_seed.py ROOT CLI_CHECKPOINT

ROOT holds dipeptide/; CLI_CHECKPOINT is the checkpoint that `mdir run`
wrote with the seed 2^53. For each seed, a model with that seed is
compiled and its checkpoint written at step 0; the entry of the seed is
printed, and that of 2^53 is compared with the checkpoint of `mdir run`.
"""
import pathlib
import sys
import tempfile

import mdir

root = pathlib.Path(sys.argv[1])
cli = mdir.read_checkpoint(sys.argv[2])


def seed_entry(fingerprint):
    entries = [value for group, name, value in fingerprint
               if group == "coupling" and name == "[dynamics] seed"]
    assert len(entries) == 1, fingerprint
    return entries[0]


texts = {}
with tempfile.TemporaryDirectory(dir=root) as work:
    for seed in (2**53 - 1, 2**53, 2**53 + 1, 2**63 - 1):
        data = mdir.load_amber(str(root / "dipeptide/dipeptide.prmtop"),
                               str(root / "dipeptide/dipeptide.inpcrd"))
        system, state = data.make_system(), data.make_state()
        system.periodic = True
        ensemble = mdir.Ensemble()
        ensemble.seed = seed
        execution = mdir.Execution()
        execution.target = mdir.Target.CPU
        execution.precision = mdir.Precision.Double
        simulation = mdir.Simulation(mdir.compile(system, state, mdir.Integrator(), ensemble,
                                                  execution, mdir.Schedule()))
        path = pathlib.Path(work) / f"{seed}.h5"
        simulation.save_checkpoint(str(path))
        texts[seed] = seed_entry(mdir.read_checkpoint(str(path)).fingerprint)
        print(f"python seed {seed}: {texts[seed]}")
assert len(set(texts.values())) == len(texts), texts
assert texts[2**53] == seed_entry(cli.fingerprint), (texts[2**53], seed_entry(cli.fingerprint))
print("the seed of 2^53 is that of mdir run:", seed_entry(cli.fingerprint))
