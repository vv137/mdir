"""The build part of the key of the cache of the skins (D[skin-budget-record]):
the lines between the format (`mdir skins 2`) and the version, as a hash,
and whether they name the image.

Usage: skin_salt.py python ROOT   (the key of `Program.plan`, the dipeptide
                                   on the CPU)
       skin_salt.py check FILE    (the key of `mdir check --json`)
       skin_salt.py compare FILE  (whether each line's hash is that of the
                                   line before)
"""
import hashlib
import json
import sys


def build_part(key):
    lines = key.splitlines()
    assert lines[0] == "mdir skins 2", lines[0]
    end = next(i for i, line in enumerate(lines) if line.startswith("version "))
    part = "\n".join(lines[1:end])
    image = any(line.startswith("image ") for line in lines[1:end])
    return hashlib.sha256(part.encode()).hexdigest()[:16], image


mode, path = sys.argv[1:3]
if mode == "compare":
    rows = [line.split() for line in open(path)]
    for before, row in zip(rows, rows[1:]):
        print(before[0], "->", row[0], "same" if before[1] == row[1] else "different")
    sys.exit(0)
if mode == "check":
    key = json.load(open(path))["run"]["skins"]["cache_key"]
else:
    import mdir

    loaded = mdir.load_amber(path + "/dipeptide.prmtop", path + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff = 0.8
    system.pairlist_distance = mdir.Auto
    execution = mdir.Execution()
    execution.target = mdir.Target.CPU
    execution.precision = mdir.Precision.Double
    program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(),
                           execution, mdir.Schedule())
    key = program.plan["skin_cache_key"]
digest, image = build_part(key)
print("mdir" if mode == "check" else "python", digest, f"image={image}")
