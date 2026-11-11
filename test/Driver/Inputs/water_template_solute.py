"""Controls for alanine dipeptide in OPC or TIP3P with the template ON.

The protein's X-H bonds are constrained (hydrogen_bonds), so its heavy
atoms anchor the solute units of D[rigid-water-template]; --loose leaves
them free, which the template refuses. PME, 2 fs, and an NPT option with
C-RESCALE exercise cell changes between builds.

    water_template_solute.py INPUTS WORK MODEL ENSEMBLE STEPS [--loose]
"""
from pathlib import Path
import sys

inputs, work = (Path(a).resolve() for a in sys.argv[1:3])
model, ensemble, steps = sys.argv[3], sys.argv[4], int(sys.argv[5])
loose = sys.argv[6:] == ['--loose']
assert model in ('OPC', 'TIP3P') and ensemble in ('NVE', 'NVT', 'NPT')
assert len(sys.argv) in (6, 7) and (len(sys.argv) == 6 or loose)
files = (inputs / 'opc' / 'opc', ) if model == 'OPC' else (inputs / 'dipeptide' / 'dipeptide', )
coupling = ''
if ensemble != 'NVE':
    coupling += '''
[thermostat]
method = "V-RESCALE"
time_constant = 1.0
interval = 10
'''
if ensemble == 'NPT':
    coupling += '''
[barostat]
method = "C-RESCALE"
time_constant = 2.0
interval = 10
'''
(work / 'control.toml').write_text(f'''
[input]
topology = "{files[0]}.prmtop"
coordinates = "{files[0]}.inpcrd"
[output]
energy_interval = {max(1, steps // 10)}
[energy]
cutoff = 8.0
pairlist_distance = 10.0
pruned_distance = 8.6
electrostatics = "PME"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = {0.001 if loose else 0.002}
steps = {steps}
[ensemble]
ensemble = "{ensemble}"
temperature = 300.0
{'pressure = 1.0' if ensemble == 'NPT' else ''}
[constraints]
rigid_water = true
hydrogen_bonds = {'false' if loose else 'true'}
[boundary]
type = "PERIODIC"
[execution]
target = "GPU"
precision = "MIXED"
[performance]
"water.template" = "ON"
{coupling}''')
