"""Where the coupling step runs (D[device-coupling]): the record of
`[execution] coupling_step` from `mdir check --json`, for each case that
keeps the host path, and the refusal of a value that is not a choice."""
import json
from pathlib import Path
import subprocess
import sys

inputs, work, mdir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
work.mkdir(parents=True, exist_ok=True)

DECK = """[input]
format      = "AMBER"
topology    = "{inputs}/dipeptide.prmtop"
coordinates = "{inputs}/dipeptide.inpcrd"

[energy]
cutoff         = 8.0
electrostatics = "PME"

[dynamics]
integrator = "{integrator}"
time_step  = 0.002
steps      = 100
{friction}
[ensemble]
ensemble    = "{ensemble}"
temperature = 300.0
{pressure}
{thermostat}
{barostat}
[constraints]
hydrogen_bonds = true
rigid_water    = true

[boundary]
type = "PERIODIC"

[execution]
target    = "{target}"
precision = "MIXED"
{coupling}
"""
RESCALE = '[thermostat]\nmethod = "V-RESCALE"\ntime_constant = 0.5\ninterval = 10\n'
CHAIN = '[thermostat]\nmethod = "NOSE-HOOVER"\ntime_constant = 0.5\ninterval = 10\n'
LANGEVIN = '[thermostat]\nmethod = "LANGEVIN"\nfriction = 1.0\n'
CELL = '[barostat]\nmethod = "C-RESCALE"\ntime_constant = 2.0\ninterval = 10\n'


def check(name, target="GPU", thermostat=RESCALE, barostat="", coupling=None,
          integrator="VELOCITY_VERLET", fail=False):
    deck = DECK.format(inputs=inputs, target=target, thermostat=thermostat,
                       barostat=barostat, integrator=integrator, friction="",
                       ensemble="NPT" if barostat else ("NVT" if thermostat else "NVE"),
                       pressure="pressure = 1.0" if barostat else "",
                       coupling=f'coupling_step = "{coupling}"' if coupling else "")
    path = work / f"{name}.toml"
    path.write_text(deck)
    done = subprocess.run([mdir, "check", "--json", str(path)], text=True,
                          capture_output=True)
    if fail:
        assert done.returncode != 0, name
        return done.stdout + done.stderr
    assert done.returncode == 0, (name, done.stdout, done.stderr)
    record = json.loads(done.stdout)["run"]["coupling_step"]
    assert record["category"] == "rule" and record["changes_rounding"] is False, record
    return record


def expect(record, value, reason, requested="DEVICE"):
    assert record["requested"] == requested, record
    assert record["value"] == value, record
    assert record["reason"] == reason, record


expect(check("nvt"), "DEVICE", "as requested")
expect(check("npt", barostat=CELL), "DEVICE", "as requested")
expect(check("host", coupling="HOST"), "HOST", "as requested", requested="HOST")
expect(check("cpu", target="CPU"), "HOST", "the CPU target has no device")
expect(check("chain", thermostat=CHAIN), "HOST", "the Nose-Hoover chain runs on the host")
expect(check("langevin", thermostat=LANGEVIN), "HOST",
       "Langevin dynamics has no coupling factor")
expect(check("nve", thermostat=""), "HOST", "the run has no coupling step")
message = check("invalid", coupling="EVERYWHERE", fail=True)
assert "coupling_step" in message, message
print("coupling_step: DEVICE by default; HOST on request, on the CPU, under a "
      "Nose-Hoover chain, under Langevin dynamics, without coupling; invalid refused")
