"""Exercise graph eligibility before any device-dependent setup."""
import json
from pathlib import Path
import subprocess
import sys

inputs = Path(__file__).parent
template = (inputs / "step-graph.toml.in").read_text()
control = Path(sys.argv[1])
base = template.format(inputs=inputs / "dipeptide", steps=20, period=4, mode="GRAPH")
pme = base[base.index("[pme]"):base.index("[dynamics]")]
# Observe every step of the prolate run so TOWERS needs no force-only
# kernel. The cutoff case exercises the same refusal with ordinary steps.
prolate = base.replace(pme, '[pme]\nkernel = "PROLATE"\ngrid = [32,32,32]\n')
cases = [
    (base.replace('"TOWERS"', '"MATRIX"'), "requires TOWERS"),
    (base.replace('"TOWERS"', '"GROUPS"'), "requires TOWERS"),
    (base.replace('"TOWERS"', '"MATRIX"').replace('[execution]', '[execution]\ndeterministic = true'), "deterministic execution"),
    (prolate.replace("interval = 4", "interval = 1"), "Gaussian PME"),
    (base.replace(pme, "").replace('"PME"', '"CUTOFF"'), "Gaussian PME"),
    (base.replace('"VELOCITY_VERLET"', '"LEAPFROG"'), "velocity Verlet"),
    (base.replace('"NVT"', '"NPT"') + '\n[barostat]\nmethod = "C-RESCALE"\n', "fixed cell"),
    (base.replace('"V-RESCALE"', '"NOSE-HOOVER"'), "V-rescale"),
    (base.replace("interval = 4", "interval = 1"), "no ordinary steps"),
]
for text, reason in cases:
    for request in ("GRAPH", "AUTO"):
        control.write_text(text.replace('step_graph = "GRAPH"', f'step_graph = "{request}"'))
        proc = subprocess.run([sys.argv[2], "check", "--json", str(control)], capture_output=True, text=True)
        if request == "GRAPH":
            assert proc.returncode and reason in proc.stdout, (reason, proc.stdout, proc.stderr)
        else:
            assert proc.returncode == 0, (reason, proc.stdout, proc.stderr)
            choice = json.loads(proc.stdout)["run"]["step_graph"]
            assert choice["value"] == "LAUNCH" and reason in choice["reason"], choice
