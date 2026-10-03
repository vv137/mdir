"""Check the preflight's JSON contract, warnings, units, and lack of writes."""

import json
import pathlib
import re
import subprocess
import sys

mdir, source, work = sys.argv[1:]
source = pathlib.Path(source).resolve()
work = pathlib.Path(work).resolve()
work.mkdir(parents=True, exist_ok=True)


def template(kind):
    text = subprocess.check_output([mdir, "template", kind], text=True)
    paths = {
        "system.pdb": source / "examples/argon/argon.pdb",
        "system.prmtop": source / "examples/ala3/ala3.prmtop",
        "system.inpcrd": source / "examples/ala3/ala3.inpcrd",
    }
    for name, path in paths.items():
        text = text.replace('"' + name + '"', json.dumps(str(path)))
    return text


def check(name, text, success=True):
    path = work / (name + ".toml")
    path.write_text(text)
    result = subprocess.run(
        [mdir, "check", "--json", str(path)], capture_output=True, text=True
    )
    assert result.returncode == (0 if success else 1), result.stderr + result.stdout
    assert not result.stderr, result.stderr
    report = json.loads(result.stdout)
    assert report["schema_version"] == 1
    assert report["ok"] is success
    assert bool(report["errors"]) is not success
    return report


def codes(report):
    return {warning["code"] for warning in report["warnings"]}


def notes(report):
    return [note["code"] for note in report["notes"]]


def output(report, kind):
    return next(item for item in report["outputs"] if item["kind"] == kind)


# An inactive named trajectory is listed but must not be backed up. Its
# quoted filename also checks proper JSON string escaping.
sentinel = work / 'frames with "quotes".xtc'
sentinel.write_bytes(b"keep this trajectory\n")
text = template("md").replace('"run.dcd"', json.dumps(sentinel.name))
text = re.sub(r"(?m)^steps\s*=\s*\d+", "steps = 10000", text)
text = re.sub(r"(?m)^time_step\s*=\s*[\d.]+", "time_step = 0.002", text)
report = check("pdb", text)
assert report["system"]["particles"] == 864
assert report["system"]["degrees_of_freedom"] == 2589
assert report["system"]["periodic"] is True
assert report["system"]["cell_angstrom"]["diagonal"] == [40.0] * 3
assert report["run"]["ensemble"] == "NVE"
assert report["run"]["time_step_ps"] == 0.002
assert report["run"]["duration_ns"] == 0.02
assert report["run"]["pme"] is None
assert report["run"]["target"] == "cpu"
assert report["run"]["precision"] == "double"
assert output(report, "trajectory")["path"] == str(sentinel)
assert output(report, "trajectory")["format"] == "XTC"
assert output(report, "trajectory")["exists"] is True
assert output(report, "trajectory")["enabled"] is False
assert output(report, "trajectory")["backup"] is None
assert codes(report) == {"no_checkpoint"}
assert notes(report) == []

# Active files that exist are noted with the name that a run keeps them
# under (D149), without being touched. A future checkpoint
# remains absent, and a missing input checkpoint does not prevent inspecting
# a stage before the preceding stage runs.
active = re.sub(r"(?m)^trajectory_interval\s*=\s*0", "trajectory_interval = 10", text)
active = active.replace("[output]", '[output]\ncheckpoint = "new.h5"\ncheckpoint_interval = 100')
active = active.replace("[input]", '[input]\ncheckpoint = "earlier.h5"')
report = check("active", active)
assert codes(report) == {"missing_input_checkpoint"}
assert notes(report) == ["output_backup"]
assert output(report, "trajectory")["backup"] == str(
    work / '#frames with "quotes".xtc.1#')
assert "no_checkpoint" not in codes(report)
assert output(report, "trajectory")["interval_steps"] == 10
assert output(report, "checkpoint")["interval_steps"] == 100
assert output(report, "checkpoint")["exists"] is False
assert not (work / "new.h5").exists()
assert sentinel.read_bytes() == b"keep this trajectory\n"

# Existing checkpoints are reported without loading or replacing their
# contents. A malformed checkpoint is outside this input-only preflight.
prior = work / "earlier.h5"
state = work / "new.h5"
prior.write_bytes(b"input checkpoint is not loaded\n")
state.write_bytes(b"output checkpoint is not replaced\n")
report = check("existing-checkpoints", active)
assert "missing_input_checkpoint" not in codes(report)
assert notes(report) == ["output_backup", "output_backup"]
assert output(report, "checkpoint")["backup"] == str(work / "#new.h5.1#")
assert prior.read_bytes() == b"input checkpoint is not loaded\n"
assert state.read_bytes() == b"output checkpoint is not replaced\n"
prior.unlink()
state.unlink()

risky = text.replace("[energy]", "[energy]\nrebuild_interval = 20")
risky = re.sub(r"(?m)^energy_interval\s*=\s*10", "energy_interval = 0", risky)
report = check("risky", risky)
assert {"fixed_rebuild_interval", "no_checkpoint", "no_energies"} <= codes(report)
# The log goes to the standard output in any case, without rows.
assert output(report, "log")["enabled"] is True
assert output(report, "log")["path"] is None
assert output(report, "log")["interval_steps"] == 0
assert output(report, "log")["count"] is None

# Both parse errors and missing input files remain machine-readable failures.
report = check("invalid", text.replace("cutoff ", "cutof "), success=False)
assert "unknown keyword 'cutof'" in report["errors"][0]
missing = re.sub(r"(?m)^coordinates\s*=.*", 'coordinates = "absent.pdb"', text)
report = check("missing", missing, success=False)
assert "cannot read" in report["errors"][0]

# A topology used to return before reporting any run settings.
report = check("amber", template("amber"))
assert report["system"]["particles"] == 4905
assert report["system"]["degrees_of_freedom"] == 7387
assert report["run"]["ensemble"] == "NPT"
assert report["run"]["duration_ns"] == 1.0
assert report["run"]["pressure_atm"] == 1.0
assert report["run"]["electrostatics"] == "PME"
assert report["run"]["pme"]["grid_points"] == [None] * 3
assert report["run"]["pme"]["beta_inverse_angstrom"] is None
assert report["run"]["pme"]["order"] == 4
assert report["run"]["constraints"]["distances"] > 0
assert report["run"]["constraints"]["hydrogen_bonds"] is True
assert report["run"]["constraints"]["rigid_water"] is True
assert report["run"]["target"] == "gpu"
assert report["run"]["precision"] == "mixed"
# The files of D149 and how many rows, frames, and checkpoints they get.
assert output(report, "log")["path"].endswith("run.log")
assert output(report, "log")["count"] == 101
assert output(report, "log")["count_of"] == "rows"
assert output(report, "energy")["format"] == "columns"
assert output(report, "energy")["enabled"] is True
assert output(report, "pull")["enabled"] is False
assert output(report, "trajectory")["count"] == 100
assert output(report, "trajectory")["count_of"] == "frames"
assert output(report, "checkpoint")["count"] == 10
assert [item["kind"] for item in report["outputs"]] == [
    "log", "energy", "pull", "trajectory", "checkpoint", "manifest"]

explicit = template("amber") + '\n[pme]\ngrid = [32, 40, 48]\nbeta = 0.35\n'
explicit = re.sub(r"(?m)^\[barostat\]$", '[barostat]\ncoupling = "SEMI_ISOTROPIC"', explicit)
report = check("explicit", explicit)
assert report["run"]["pme"]["grid_points"] == [32, 40, 48]
assert report["run"]["pme"]["beta_inverse_angstrom"] == 0.35
assert report["run"]["barostat_coupling"] == "SEMI_ISOTROPIC"

# Minimization iterations have no physical duration or integration step.
report = check("minimize", template("minimize"))
assert report["run"]["kind"] == "minimization"
assert report["run"]["steps"] == 2000
assert report["run"]["time_step_ps"] is None
assert report["run"]["duration_ns"] is None
assert output(report, "checkpoint")["at_end"] is True
assert output(report, "checkpoint")["enabled"] is True
assert output(report, "checkpoint")["count"] == 1
assert output(report, "log")["count"] is None

nvt = template("nvt").replace('"V-RESCALE"', '"LANGEVIN"')
nvt = re.sub(r"(?m)^time_constant\s*=.*", "friction = 1.0", nvt)
report = check("langevin", nvt)
assert report["run"]["ensemble"] == "NVT"
assert report["run"]["thermostat"] == "LANGEVIN"
assert report["run"]["duration_ns"] == 0.05

# An implementation cell is not reported as a physical periodic cell.
nonperiodic = text.replace('"PERIODIC"', '"NONE"')
nonperiodic = re.sub(r"(?m)^box\s*=.*\n", "", nonperiodic)
report = check("none", nonperiodic)
assert report["system"]["periodic"] is False
assert report["system"]["cell_angstrom"] is None

# The newer pulling output must be included alongside energies, trajectory,
# and checkpoint. Reuse the two-atom fixture, without running its dynamics.
fixture = (source / "test/Driver/pull-coordinates.test").read_text()
for name, next_name in (("two.top", "two.gro"), ("two.gro", "run.toml")):
    part = fixture.split("#--- " + name + "\n", 1)[1]
    (work / name).write_text(part.split("#--- " + next_name, 1)[0])
pulling = fixture.split("#--- run.toml\n", 1)[1].split("#--- tuple", 1)[0]
pull = work / "pull.dat"
pull.write_bytes(b"keep pulling coordinates\n")
report = check("pull", pulling)
assert report["system"]["periodic"] is False
assert report["system"]["cell_angstrom"] is None
assert output(report, "pull")["interval_steps"] == 20
assert output(report, "pull")["enabled"] is True
assert output(report, "pull")["exists"] is True
assert output(report, "pull")["count"] == 6
assert output(report, "pull")["backup"] == str(work / "#pull.dat.1#")
assert "output_backup" in notes(report)
assert pull.read_bytes() == b"keep pulling coordinates\n"
assert not (work / "two.h5").exists()
print("preflight JSON, warnings, units, and read-only checks passed")
