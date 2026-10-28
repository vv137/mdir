#!/usr/bin/env python3
"""Compare the term-order matrix against an archived builder base.

This emits programs only. The metadata-only MLIP fixture supplies a cutoff
without an optional model framework and refuses any attempt to evaluate a
model. CPU-only triplets and GPU-only models retain their expected errors.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def quoted(path):
    return json.dumps(str(path))


def controls(root, place):
    inputs = root / "test/Driver/Inputs"
    out = place / "controls"
    out.mkdir(parents=True, exist_ok=True)
    ordinary, hosted = [], []
    combo = f'''[input]
topology = {quoted(inputs / 'gb/peptide.prmtop')}
coordinates = {quoted(inputs / 'gb/peptide.inpcrd')}
[output]
energy_interval = 1
checkpoint = "order.h5"
checkpoint_interval = 1
pull = "pull.dat"
[energy]
cutoff = 40.0
pairlist_distance = 41.0
electrostatics = "CUTOFF"
implicit_solvent = "SOLVENT"
surface_area_energy = 0.0054
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 1e-9
steps = 1
[ensemble]
ensemble = "NVE"
temperature = 0.0
[constraints]
hydrogen_bonds = false
rigid_water = false
[boundary]
type = "NONE"
[[energy.external]]
name = "rank_trap"
expression = "k*(x-t)^2"
particles = [1]
k = 0.01
scaling = "CELL"
[[restraints]]
selection = "@1"
force_constant = 1.0
[execution]
precision = "PRECISION"
[[energy.bond]]
name = "rank_tuple"
expression = "k*(r-r0)^2"
particles = [[1, 2]]
k = 0.1
r0 = 1.0
[[energy.angle]]
name = "rank_centroid"
expression = "k*(theta-theta0)^2"
groups = ["@1", "@2", "@3"]
k = 0.1
theta0 = 1.0
[[energy.pair]]
name = "rank_pair"
expression = "a/r^6"
a = 0.01
'''

    def write(name, text, collection=ordinary):
        path = out / (name + ".toml")
        path.write_text(text)
        collection.append(path)
        return path

    for solvent in ("HCT", "OBC1", "OBC2"):
        for precision in ("DOUBLE", "MIXED"):
            write(f"combined-{solvent}-{precision}",
                  combo.replace("SOLVENT", solvent).replace("PRECISION", precision))
    bare = (root / "test/Driver/charged.toml").read_text().replace(
        '"Inputs/', '"' + str(inputs) + "/")
    for precision in ("DOUBLE", "MIXED"):
        write(f"bare-overrides-{precision}", bare +
              f'\n[execution]\nprecision = "{precision}"\n')

    pdb = inputs / "mw64.pdb"
    changed = place / "mw64u.pdb"
    lines = []
    for line in pdb.read_text().splitlines(True):
        if line.startswith("HETATM") and int(line[6:11]) % 3 == 0:
            line = line[:12] + " WU " + line[16:]
        lines.append(line)
    changed.write_text("".join(lines))
    test = (root / "test/Driver/triplet-terms.test").read_text()
    triplet = test.split("#--- run.toml\n", 1)[1].split("#--- end", 1)[0]
    triplet = triplet.replace("#--- terms\n", "").replace('"mw64.pdb"', quoted(pdb))
    typed = triplet.replace(quoted(pdb), quoted(changed)).replace(
        'expression = "lambda*',
        'expression = "lambda1*sqrt(lambda2*lambda3)/23.15*').replace(
        "lambda  = 23.15\n", "").replace(
        "mass = 18.015\n", 'mass = 18.015\nlambda = 23.15\n'
        '[[energy.type]]\nname = "WU"\nmass = 18.015\nlambda = 30.0\n')
    shared = typed.replace('expression = "A*epsilon',
                           'expression = "(lambda/23.15)*A*epsilon').replace(
        'name       = "two-body"',
        'name       = "two-body"\nmixing = { lambda = "arithmetic" }')
    for name, text in (("uniform", triplet), ("places", typed), ("shared", shared)):
        for precision in ("DOUBLE", "MIXED"):
            write(f"triplet-{name}-{precision}", text.replace(
                'precision = "DOUBLE"', f'precision = "{precision}"'))

    model_ir = '''md.potential @order_model(%positions: !md.field<@atoms, 3 x f64>, %box: !md.cell, %numbers: !md.field<@atoms, i32>) -> f64 {
  %total = md.sum_particles gather(%positions : !md.field<@atoms, 3 x f64>) {
  ^bb0(%position: vector<3xf64>):
    %x = vector.extract %position[0] : f64 from vector<3xf64>
    %square = arith.mulf %x, %x : f64
    %k = arith.constant COEFFICIENT : f64
    %energy = arith.mulf %square, %k : f64
    md.yield %energy : f64
  } : f64
  md.return %total : f64
}
'''
    models = []
    for index, coefficient in enumerate(("1.000000e-03", "2.000000e-03")):
        path = place / f"model-{index}.mlir"
        path.write_text(model_ir.replace("COEFFICIENT", coefficient))
        models.append(path)
    placeholder = place / "emission-only.pt"
    placeholder.write_text("Metadata-only emission fixture; never evaluate.\n")
    periodic = place / "peptide-periodic.inpcrd"
    periodic.write_text((inputs / "gb/peptide.inpcrd").read_text().rstrip() + "\n" +
                        "".join(f"{value:12.7f}" for value in
                                (100.0, 100.0, 100.0, 90.0, 90.0, 90.0)) + "\n")
    for precision in ("DOUBLE", "MIXED"):
        text = combo.replace("SOLVENT", "HCT").replace("PRECISION", precision)
        text = text.replace(quoted(inputs / "gb/peptide.inpcrd"), quoted(periodic))
        text = text.replace('type = "NONE"', 'type = "PERIODIC"')
        text = text.replace("temperature = 0.0", "temperature = 300.0")
        single = text + ('\n[[energy.model]]\nname = "stage_a"\nbackend = "STAGES"\n'
                         f'path = {quoted(models[0])}\n')
        write(f"model-stage-{precision}", single)
        write(f"model-two-stages-{precision}", single +
              '\n[[energy.model]]\nname = "stage_b"\nbackend = "STAGES"\n' +
              f'path = {quoted(models[1])}\n')
        opaque = text + ('\n[[energy.model]]\nname = "hosted"\nbackend = "METATOMIC"\n'
                         f'path = {quoted(placeholder)}\n')
        def host_case(kind, body):
            # #706 is a pre-existing stored-energy SSA error. Keep it in
            # the matrix, alongside valid host paths without restraints.
            write(f"model-host-restraints-{kind}-{precision}", body)
            body = body.replace('[[restraints]]\nselection = "@1"\n'
                                'force_constant = 1.0\n', '')
            hosted.append(write(f"model-host-{kind}-{precision}", body))

        host_case("nve", opaque)
        npt = opaque.replace('ensemble = "NVE"', 'ensemble = "NPT"\npressure = 1.0')
        npt = npt.replace('steps = 1\n', 'steps = 10\n')
        npt = npt.replace('energy_interval = 1\n', 'energy_interval = 10\n')
        npt = npt.replace('checkpoint_interval = 1\n', 'checkpoint_interval = 10\n')
        npt += ('\n[thermostat]\nmethod = "V-RESCALE"\ntime_constant = 0.1\ninterval = 10\n'
                '\n[barostat]\nmethod = "C-RESCALE"\ntime_constant = 1.0\ninterval = 10\n')
        host_case("npt", npt)
        minimized = re.sub(r'\[dynamics\]\n.*?(?=\[ensemble\])',
                           '[minimize]\nmethod = "STEEPEST_DESCENT"\nsteps = 1\n',
                           opaque, flags=re.S)
        minimized = minimized.replace('pull = "pull.dat"\n', '').replace(
            'checkpoint_interval = 1\n', '')
        host_case("min", minimized)
    return ordinary, hosted


def metadata_plugin(root, place, compiler):
    source = place / "metadata.c"
    source.write_text('''#include "mdir/MLIP/Plugin.h"
int32_t mdir_mlip_abi_version(void) { return MDIR_MLIP_ABI_VERSION; }
const char *mdir_mlip_build_versions(void) { return "order-matrix metadata only"; }
const char *mdir_mlip_torch_version(void) { return "none"; }
const char *mdir_mlip_last_error(void) { return "metadata-only fixture: never evaluate"; }
int mdir_mlip_cutoff(const mdir_mlip_load_options *o, double *r) { (void)o; *r = 0.3; return MDIR_MLIP_OK; }
int mdir_mlip_load(const mdir_mlip_load_options *o, mdir_mlip_model **m) { (void)o; (void)m; return MDIR_MLIP_ERROR; }
int mdir_mlip_describe(const mdir_mlip_model *m, mdir_mlip_description *d) { (void)m; (void)d; return MDIR_MLIP_ERROR; }
int mdir_mlip_evaluate(mdir_mlip_model *m, const mdir_mlip_inputs *i, mdir_mlip_outputs *o) { (void)m; (void)i; (void)o; return MDIR_MLIP_ERROR; }
int mdir_mlip_memory_stats(mdir_mlip_model *m, int32_t r, mdir_mlip_memory *s) { (void)m; (void)r; (void)s; return MDIR_MLIP_ERROR; }
void mdir_mlip_free(mdir_mlip_model *m) { (void)m; }
''')
    library = place / "liborder-metadata.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-I", str(root / "include"),
                    str(source), "-o", str(library)], check=True)
    return library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--printer", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--host-raw", action="store_true",
                        help="also compare hosted-module text before known baseline verifier errors")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be between 1 and 8")
    root, place = args.root.resolve(), args.scratch.resolve()
    place.mkdir(parents=True, exist_ok=True)
    ordinary, hosted = controls(root, place)
    library = metadata_plugin(root, place, args.compiler)
    for mode, files, expected_errors in (("exact", ordinary, 68),
                                         ("padded", hosted, 24),
                                         ("search", hosted, 24)):
        scratch = place / ("compare-" + mode)
        scratch.mkdir(exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).with_name("check.py")), "compare",
               str(args.old.resolve()), str(args.new.resolve()), "--only", "extra",
               "--forms", "printed", "--printer", str(args.printer.resolve()),
               "--stages", "module,lowered", "--jobs", str(args.jobs),
               "--scratch", str(scratch), "--show-errors"]
        for control in files:
            cmd += ["--control", str(control)]
        env = dict(os.environ, MDIR_MLIP_PAIRS=mode, MDIR_MLIP_PLUGIN=str(library))
        result = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        print(f"pair-list mode: {mode}", flush=True)
        print(result.stdout, end="", flush=True)
        if result.returncode:
            return result.returncode
        match = re.search(r"identical: (\d+) \(of which stop with the same error in both: (\d+)\)",
                          result.stdout)
        unexpected = []
        for line in result.stdout.splitlines():
            if not line.startswith("ERROR: "):
                continue
            support = ("triplet-" in line and "[gpu" in line and
                       "run on the CPU only" in line) or (
                       "extra/model-" in line and "[cpu" in line and
                       "runs on a GPU only" in line)
            baseline = ("extra/model-host-restraints-" in line and "[gpu" in line and
                        "use of undeclared SSA value name" in line)
            baseline |= ("extra/model-host-" in line and "[gpu" in line and
                         "does not state the effect of its body" in line and
                         "'<Default>'" in line)  # #707, independently checked on the base.
            if not support and not baseline:
                unexpected.append(line)
        if not match or int(match[2]) != expected_errors or unexpected:
            print(f"order-matrix fixture error: expected {expected_errors} support/baseline error cases",
                  file=sys.stderr)
            for line in unexpected:
                print(line, file=sys.stderr)
            return 2
        if args.host_raw:
            raw_place = place / ("raw-" + mode)
            raw_place.mkdir(exist_ok=True)
            raw = [sys.executable, str(Path(__file__).with_name("check.py")),
                   "compare", str(args.old.resolve()), str(args.new.resolve()),
                   "--only", "extra", "--forms", "raw", "--stages", "module",
                   "--targets", "gpu", "--jobs", str(args.jobs),
                   "--scratch", str(raw_place)]
            opaque = [path for path in ordinary if path.name.startswith("model-host-")]
            for control in opaque:
                raw += ["--control", str(control)]
            checked = subprocess.run(raw, env=env, text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
            print(f"supplementary hosted module text: {mode}", flush=True)
            print(checked.stdout, end="", flush=True)
            if checked.returncode:
                return checked.returncode
            match = re.search(r"identical: (\d+) \(of which stop with the same error in both: (\d+)\)",
                              checked.stdout)
            if not match or int(match[2]) != 0:
                print("order-matrix fixture error: hosted text was not emitted", file=sys.stderr)
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
