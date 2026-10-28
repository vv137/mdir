"""Device-free emission and device evaluation of identical unit-width exports."""
import argparse
import json
import io
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["emit", "evaluate"])
parser.add_argument("root", type=Path)
parser.add_argument("scratch", type=Path)
parser.add_argument("--mdir", default="mdir")
parser.add_argument("--printer", default="mdir-opt")
args = parser.parse_args()
root, place = args.root.resolve(), args.scratch.resolve()
place.mkdir(parents=True, exist_ok=True)
fixtures = root / "test/Driver/Inputs/model-stages"
models = {1: fixtures / "painn-unit-rows1.mlir",
          64: fixtures / "painn-scalar-export.mlir"}
subprocess.run([sys.executable, str(root / "scripts/mlip/water-box.py"),
                str(place), "--n", "2", "--spacing", "5.0"], check=True)

if args.mode == "emit":
    for precision in ["DOUBLE", "MIXED"]:
        control = place / (precision + ".toml")
        control.write_text(f'''[input]
topology = "system.top"
coordinates = "system.gro"
[energy]
cutoff = 0.2
pairlist_distance = 0.25
electrostatics = "CUTOFF"
[[energy.model]]
name = "unit"
backend = "STAGES"
path = "{models[1]}"
[constraints]
hydrogen_bonds = false
rigid_water = false
[dynamics]
steps = 0
[ensemble]
ensemble = "NVE"
[boundary]
type = "PERIODIC"
[execution]
target = "GPU"
precision = "{precision}"
''')
        output = subprocess.check_output(
            [args.mdir, "emit", "--stage=lowered", str(control)])
        subprocess.run([args.printer, "-", "-o", "/dev/null"],
                       input=output, check=True)
    print("one-row exports: double and mixed lowered and verified")
else:
    import numpy as np

    def evaluate(rows, precision):
        # Ordinary CLI evaluation also covers the public run path; the
        # construction/segmented-program fix is reviewed separately in #716.
        folder = place / f"{precision}-rows{rows}"
        folder.mkdir(exist_ok=True)
        control = folder / "run.toml"
        control.write_text(f'''[input]
 topology = "{place / 'system.top'}"
 coordinates = "{place / 'system.gro'}"
 [output]
 energy = "energy.dat"
 energy_interval = 1
 checkpoint = "state.h5"
 checkpoint_interval = 1
 [energy]
 cutoff = 0.2
 pairlist_distance = 0.25
 electrostatics = "CUTOFF"
 [[energy.model]]
 name = "unit"
 backend = "STAGES"
 path = "{models[rows]}"
 [constraints]
 hydrogen_bonds = false
 rigid_water = false
 [dynamics]
 steps = 1
 time_step = 0.0005
 [ensemble]
 ensemble = "NVE"
 temperature = 0.0
 [boundary]
 type = "PERIODIC"
 [execution]
 target = "GPU"
 precision = "{precision.upper()}"
''')
        result = subprocess.run([args.mdir, "run", str(control)], cwd=folder,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=True, text=True)
        (folder / "run.log").write_text(result.stdout + result.stderr)
        force_text = subprocess.check_output(
            [args.mdir, "checkpoint", "--print=forces", str(folder / "state.h5")],
            text=True)
        forces = np.loadtxt(io.StringIO(force_text))[:, 2:]
        lines = (folder / "energy.dat").read_text().splitlines()
        names = lines[0].lstrip("# ").split()
        data = [line.split() for line in lines if line and not line.startswith("#")]
        # The potential column is in kcal/mol; comparisons use kJ/mol.
        potential = next(i for i, name in enumerate(names) if name.lower() == "potential")
        energy = float(data[-1][potential]) * 4.184
        return energy, forces

    reports = []
    for precision in ["Double", "Mixed"]:
        reference, actual = evaluate(64, precision), evaluate(1, precision)
        de = abs(actual[0] - reference[0])
        df = float(np.max(np.abs(actual[1] - reference[1])))
        # Same f32 body, with a different row grouping in the energy sum.
        energy_tolerance, force_tolerance = 1e-4, 1e-5
        report = dict(precision=precision, reference_energy=reference[0],
                      energy=actual[0], energy_difference=de,
                      force_difference=df,
                      energy_tolerance=energy_tolerance,
                      force_tolerance=force_tolerance)
        print(json.dumps(report), flush=True)
        assert np.isfinite(de) and de <= energy_tolerance
        assert np.isfinite(df) and df <= force_tolerance
        reports.append(report)
    (place / "comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
    print("one-row versus 64-row energy and forces passed")
