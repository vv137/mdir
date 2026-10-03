"""Independent file/hash and lifecycle oracle for the optional manifest (D168)."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime

mode, source, work, executable = sys.argv[1:]
source, work = Path(source).resolve(), Path(work).resolve()
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
base = (source / "mixture.toml").read_text().replace("Inputs/", str(source / "Inputs") + "/")
base = re.sub(r"^steps\s*= 200", "steps = 400", base, flags=re.M)


# Validate the small JSON Schema vocabulary used by the public schema without
# requiring a Python package in the build environment.
schema = json.loads((source.parent.parent / "docs/run-manifest.schema.json").read_text())

def validate(value, rule):
    for keyword in ("oneOf", "anyOf"):
        if keyword in rule:
            matches = 0
            for choice in rule[keyword]:
                try:
                    validate(value, choice)
                    matches += 1
                except AssertionError:
                    pass
            assert matches == 1 if keyword == "oneOf" else matches >= 1, (value, rule)
    if "type" in rule:
        types = rule["type"] if isinstance(rule["type"], list) else [rule["type"]]
        kind = {dict: "object", list: "array", str: "string", bool: "boolean",
                int: "integer", float: "number", type(None): "null"}[type(value)]
        assert kind in types or (kind == "integer" and "number" in types), (value, rule)
    if "const" in rule:
        assert value == rule["const"] and type(value) is type(rule["const"])
    if "enum" in rule:
        assert value in rule["enum"]
    if "minimum" in rule:
        assert value >= rule["minimum"]
    if "pattern" in rule:
        assert re.search(rule["pattern"], value)
    if isinstance(value, dict):
        assert set(rule.get("required", ())) <= value.keys()
        properties = rule.get("properties", {})
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key])
            elif rule.get("additionalProperties") is False:
                assert False, key
            elif isinstance(rule.get("additionalProperties"), dict):
                validate(item, rule["additionalProperties"])
    if isinstance(value, list):
        assert rule.get("minItems", 0) <= len(value) <= rule.get("maxItems", len(value))
        for item in value:
            if "items" in rule:
                validate(item, rule["items"])


def command(*args, status=0):
    p = subprocess.run([executable, *map(str, args)], text=True, capture_output=True)
    assert p.returncode == status, (args, p.returncode, p.stdout, p.stderr)
    return p.stdout + p.stderr


def control(name, manifest=True, target=mode):
    path = work / (name + ".toml")
    text = base + f'\ncheckpoint = "{name}.h5"\ncheckpoint_interval = 100\n'
    text += f'trajectory = "{name}.xtc"\ntrajectory_interval = 100\n'
    if manifest:
        text += f'manifest = "{name}.jsonl"\n'
    text += f'\n[execution]\ntarget = "{target.upper()}"\nprecision = "mixed"\n'
    path.write_text(text)
    return path


def records(path):
    lines = path.read_text().splitlines()
    data = [json.loads(line) for line in lines]
    for row in data:
        validate(row, schema)
        assert row["schema_version"] == 1
        datetime.strptime(row["timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ")
        if row["event"] == "end":
            assert row["elapsed_seconds"] >= 0
    return data


def hashes(record, snapshots=None):
    paths = set()
    for item in record["inputs"]:
        p = Path(item["path"])
        assert p.is_absolute() and p not in paths
        paths.add(p)
        data = snapshots[p] if snapshots and p in snapshots else p.read_bytes()
        assert item["sha256"] == hashlib.sha256(data).hexdigest()
        assert item["bytes"] == len(data)
    return paths


run = control("whole")
manifest = work / "whole.jsonl"
preflight = json.loads(command("check", run, "--json"))
entry = next(x for x in preflight["outputs"] if x["kind"] == "manifest")
assert entry["path"] == str(manifest) and entry["enabled"] and not entry["exists"]
command("emit", run)
assert not manifest.exists()
command("run", run)
a, b = records(manifest)
assert a["event"] == "start" and b["event"] == "end"
assert a["invocation"] == b["invocation"] == 1
assert a["part"] == 1 and a["first_step"] == 0 and a["requested_end_step"] == 400
assert b["status"] == "completed" and b["step"] == 400
assert a["precision"] == "mixed" and a["target"] == mode
assert re.fullmatch(r"[0-9a-f]{40}", a["build"]["git_commit"])
assert a["build"]["llvm_version"] == a["build"]["mlir_version"]
assert a["compile_seconds"] >= 0
assert a["effective"]["seed"] == "314159"
assert a["effective"]["trajectory_format"] == "XTC"
assert a["effective"]["neighbor_structures"] == ["matrix"]
assert a["effective"]["buffer_precision"]["state"] == "f64"
assert a["effective"]["buffer_precision"]["force"] == "f32"
assert a["effective"]["pme"] is None
# The public schema rejects additions without an intentional schema update.
try:
    validate(dict(a, undeclared=True), schema)
    assert False, "schema accepted an undeclared field"
except AssertionError as error:
    assert str(error) != "schema accepted an undeclared field"

# Both precision modes preserve the exact state when provenance is enabled.
for precision in ("mixed", "double"):
    with_manifest = control("precision-" + precision)
    without_manifest = control("baseline-" + precision, manifest=False)
    for path in (with_manifest, without_manifest):
        path.write_text(path.read_text().replace('precision = "mixed"', f'precision = "{precision}"'))
        command("run", path)
    assert "the states are identical" in command("checkpoint", with_manifest.with_suffix(".h5"), without_manifest.with_suffix(".h5"))
    meta, end = records(with_manifest.with_suffix(".jsonl"))
    assert meta["precision"] == precision and end["status"] == "completed"
    assert meta["effective"]["buffer_precision"]["force"] == ("f32" if precision == "mixed" else "f64")

assert run in hashes(a) and len(a["inputs"]) == 2
if mode == "gpu":
    assert a["device"]["visible_index"] == 0
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True)
    assert a["device"]["driver_api_version"] > 0
    assert a["device"]["compute_capability"][0] >= 1
    assert a["device"]["name"] in devices
    assert a["device"]["uuid_hex"] in devices.lower().replace("-", "")
    print("GPU manifest identity and hashes: ok")
    sys.exit(0)
assert a["device"] is None

# Turning provenance on changes neither dynamics nor the continuation state.
plain = control("plain", manifest=False)
command("run", plain)
assert "the states are identical" in command("checkpoint", work / "whole.h5", work / "plain.h5")
original = manifest.read_bytes()
command("run", run, "--continue")
assert manifest.read_bytes() == original
command("run", run)
assert (work / "#whole.jsonl.1#").read_bytes() == original

part = control("part")
command("run", part, "--continue", "--max-walltime", "0.000001", status=75)
p = records(work / "part.jsonl")
assert p[-1]["status"] == "stopped" and p[-1]["step"] == 100
assert p[-1]["reason"] == "the wall time"
checkpoint = work / "part.h5"
snapshot = {checkpoint: checkpoint.read_bytes()}
command("run", part, "--continue")
p = records(work / "part.jsonl")
assert [x["invocation"] for x in p] == [1, 1, 2, 2]
assert p[2]["first_step"] == 100 and p[2]["part"] == 2
assert p[-1]["status"] == "completed" and p[-1]["step"] == 400
assert checkpoint in hashes(p[2], snapshot)
assert "the states are identical" in command("checkpoint", work / "whole.h5", checkpoint)

# --no-append selects the part name, then later continuations use that name.
new = control("new")
command("run", new, "--continue", "--max-walltime", "0.000001", status=75)
kept = (work / "new.jsonl").read_bytes()
command("run", new, "--continue", "--no-append", "--max-walltime", "0.000001", status=75)
assert (work / "new.jsonl").read_bytes() == kept
command("run", new, "--continue")
assert [x["invocation"] for x in records(work / "new.part0002.jsonl")] == [1, 1, 2, 2]

# A retry with no checkpoint preserves even an unmatched start event.
retry = control("retry")
unmatched = dict(a)
(work / "retry.jsonl").write_text(json.dumps(unmatched) + "\n")
command("run", retry, "--continue", "--max-walltime", "0.000001", status=75)
assert [x["invocation"] for x in records(work / "retry.jsonl")] == [1, 2, 2]

# Reject damaged or future history without modifying it or running any steps.
for content in ('{"schema_version": 2}\n', '{"partial":', json.dumps(a)):
    (work / "retry.jsonl").write_text(content)
    before = (work / "retry.h5").read_bytes()
    text = command("run", retry, "--continue", status=1)
    assert "malformed or unsupported history" in text
    assert (work / "retry.jsonl").read_text() == content
    assert (work / "retry.h5").read_bytes() == before

# All 99 backups are protected; preflight describes the limit.
for i in range(1, 100):
    (work / f"#whole.jsonl.{i}#").touch()
assert "backup_limit" in command("check", run, "--json")
assert "99 backups" in command("run", run, status=1)

# Protect explicit inputs, the control file, output aliases, and .prev.
for name in ('collision.toml', 'whole.h5', 'whole.h5.prev', str(source / 'Inputs/mixture.pdb')):
    bad = work / "collision.toml"
    bad.write_text(run.read_text().replace('"whole.jsonl"', json.dumps(name)))
    assert "manifest" in command("check", bad, status=1)
    assert "manifest" in command("run", bad, status=1)

# Different names through a directory symlink can name one future output.
# Refuse the collision before either writer creates it.
(work / "alias").symlink_to(work, target_is_directory=True)
bad = work / "alias.toml"
bad.write_text(run.read_text().replace('"whole.jsonl"', '"alias/future.xtc"')
               .replace('"whole.xtc"', '"future.xtc"'))
assert "another output" in command("check", bad, status=1)
assert "another output" in command("run", bad, status=1)
assert not (work / "future.xtc").exists()
# Existing aliases of inputs and directories are also refused.
(work / "input-alias").symlink_to(source / "Inputs/mixture.pdb")
for name in ("input-alias", "alias"):
    bad.write_text(run.read_text().replace('"whole.jsonl"', json.dumps(name)))
    assert "manifest" in command("check", bad, status=1)
    assert "manifest" in command("run", bad, status=1)

# Active GROMACS includes are hashed; inactive includes are not opened.
gmx = work / "gromacs"
shutil.copytree(source / "Inputs/gromacs", gmx)
text = (source / "gromacs.test").read_text().split('#--- run.toml\n')[1].split('#--- end')[0]
text = text.replace('time_step     = 0.0005', 'time_step     = 0.002')
text = text.replace("cutoff       = 8.0", 'cutoff       = 8.0\nelectrostatics = "PME"')
text = text.replace('[output]', '[output]\nmanifest = "run.jsonl"')
(gmx / "run.toml").write_text(text)
with (gmx / "system.top").open("a") as f:
    f.write('\n#ifdef NEVER_ENABLED\n#include "absent.itp"\n#endif\n')
command("run", gmx / "run.toml")
gmxmeta = records(gmx / "run.jsonl")[0]
paths = hashes(gmxmeta)
assert gmxmeta["effective"]["pme"]["beta_inverse_angstrom"] > 0
assert all(x >= 8 for x in gmxmeta["effective"]["pme"]["grid"])
assert gmxmeta["warnings"] == json.loads(command("check", gmx / "run.toml", "--json"))["warnings"]
assert any(w["code"] == "long_time_step" for w in gmxmeta["warnings"])
module = command("emit", gmx / "run.toml")
grid = re.search(r"grid\(\[([0-9, ]+)\]\)", module)
beta = re.search(r"beta\(([0-9.eE+-]+)\)", module)
assert gmxmeta["effective"]["pme"]["grid"] == list(map(int, grid[1].split(',')))
assert abs(gmxmeta["effective"]["pme"]["beta_inverse_angstrom"] - float(beta[1]) * 0.1) < 1e-14

assert gmx / "forcefield.itp" in paths and gmx / "water.itp" in paths
bad = gmx / "bad.toml"
bad.write_text(text.replace('"run.jsonl"', '"water.itp"'))
before = (gmx / "water.itp").read_bytes()
assert "input of the run" in command("check", bad, status=1)
assert "input of the run" in command("run", bad, status=1)
assert (gmx / "water.itp").read_bytes() == before

# Completion events also cover paths that return early from the run summary.
minimum = control("minimum")
text = re.sub(r"\[dynamics\].*?\[ensemble\]", "[minimize]\nsteps = 400\n\n[ensemble]", minimum.read_text(), flags=re.S)
text = re.sub(r"^checkpoint_interval = 100\n", "", text, flags=re.M)
minimum.write_text(text)
command("run", minimum)
assert records(work / "minimum.jsonl")[-1]["status"] == "completed"
langevin = control("langevin")
langevin.write_text(langevin.read_text().replace('ensemble    = "NVE"', 'ensemble    = "NVT"') +
    '\n[thermostat]\nmethod = "LANGEVIN"\nfriction = 1.0\n')
command("run", langevin)
assert records(work / "langevin.jsonl")[-1]["status"] == "completed"

print("CPU manifest hashes, history, exact states, and output protection: ok")
