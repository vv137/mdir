"""Control for the template reporter's carried-state isolation test."""
from pathlib import Path
import sys

fixtures, work = map(Path, sys.argv[1:])
(work / 'control.toml').write_text(f'''
[input]
topology = "{fixtures / 'three.prmtop'}"
coordinates = "{fixtures / 'three.inpcrd'}"
[output]
energy_interval = 1
[energy]
cutoff = 9.0
pairlist_distance = 10.0
electrostatics = "CUTOFF"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.001
steps = 2
[ensemble]
ensemble = "NVE"
temperature = 300.0
[constraints]
rigid_water = true
[boundary]
type = "PERIODIC"
[execution]
target = "GPU"
precision = "MIXED"
[performance]
"water.template" = "ON"
''')
