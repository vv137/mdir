"""Controls for the approved common short-trajectory matrix."""
from pathlib import Path
import subprocess
import sys

fixtures, work = map(Path, sys.argv[1:3])
model, integrator, ensemble = sys.argv[3:]
assert model in ('OPC', 'TIP3P')
assert integrator in ('VELOCITY_VERLET', 'LEAPFROG')
assert ensemble in ('NVE', 'NPT')
script = 'water_template_report.py' if model == 'OPC' else 'water_template_tip3p.py'
subprocess.run([sys.executable, str(Path(__file__).with_name(script)), str(fixtures), str(work)], check=True)
p = work / 'control.toml'
s = p.read_text().replace('integrator = "VELOCITY_VERLET"', f'integrator = "{integrator}"')
s = s.replace('ensemble = "NVE"', f'ensemble = "{ensemble}"')
s = s.replace('steps = 2', 'steps = 100').replace('energy_interval = 1', 'energy_interval = 10')
if ensemble == 'NPT':
    s += '''
[thermostat]
method = "V-RESCALE"
time_constant = 1.0
interval = 10
[barostat]
method = "C-RESCALE"
time_constant = 5.0
interval = 10
'''
p.write_text(s)
