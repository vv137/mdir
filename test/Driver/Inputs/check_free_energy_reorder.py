"""Checks the free-energy file of D161 on a device where the particles are
put in order, with checkpoints, and from the checkpoint of another state,
against the same runs on the CPU in double precision.

    check_free_energy_reorder.py <directory> <mdir> [MIXED|DOUBLE]

The energies of the states come from `@alchemical`, whose charges are a map
over the particles; such a field once gave positions that were not numbers
on a device in runs that put the particles in order (issue #26). Prints a
line per check: every value finite, the rows of the device against those
of the CPU, and dH/dλ against the central difference of the energies of
the states beside it.
"""

import math
import os
import shutil
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
directory, mdir = sys.argv[1], sys.argv[2]
precision = sys.argv[3] if len(sys.argv) > 3 else "MIXED"
for name in ('eth_wat.prmtop', 'eth_wat.inpcrd'):
    shutil.copy(os.path.join(here, 'fep', name), directory)

h = 1e-2
CONTROL = '''[input]
topology = "eth_wat.prmtop"
coordinates = "eth_wat.inpcrd"
{start}
[output]
energy_interval = 10
free_energy = "{name}.dhdl"
checkpoint = "{name}.h5"
checkpoint_interval = 10

[free_energy]
couple = ":LIG"
state = {state}

[free_energy.lambdas]
coulomb = [0.5, {low}, {high}, 1.0]
vdw = [0.0, 0.0, 0.0, 0.5]

[energy]
cutoff = 9.0
pairlist_distance = 10.0
electrostatics = "PME"

[pme]
beta = 0.32
grid = [32, 32, 32]

[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.0005
steps = {steps}

[ensemble]
ensemble = "NVE"
temperature = 300.0

[boundary]
type = "PERIODIC"

[execution]
target = "{target}"
precision = "{precision}"
'''


def run(name, target, precision, state=0, start='', steps=20):
    text = CONTROL.format(name=name, target=target, precision=precision,
                          state=state, start=start, steps=steps,
                          low=0.5 - h, high=0.5 + h)
    with open(os.path.join(directory, name + '.toml'), 'w') as file:
        file.write(text)
    result = subprocess.run([mdir, 'run', name + '.toml'], cwd=directory,
                            capture_output=True, text=True)
    if result.returncode:
        sys.exit(f'mdir run {name}.toml failed:\n{result.stdout}\n'
                 f'{result.stderr}')
    with open(os.path.join(directory, name + '.dhdl')) as file:
        header = file.readline().split()[1:]
        file.readline()
        return [dict(zip(header, map(float, line.split()))) for line in file]


def report(label, worst, tolerance):
    ok = math.isfinite(worst) and worst <= tolerance
    print(f'{label}: {"ok" if ok else "FAILED"} (worst {worst:.2e}, '
          f'tolerance {tolerance:.0e})')


def compare(label, device, host, tolerance, central=True):
    finite = all(math.isfinite(v) for row in device for v in row.values())
    print(f'{label}, finite: {"ok" if finite and device else "FAILED"}')
    steps = {row['step']: row for row in host}
    worst = 0.0
    for row in device:
        other = steps.get(row['step'])
        if other is None:
            worst = math.inf
            continue
        for key, value in row.items():
            worst = max(worst, abs(value - other[key]))
    report(f'{label}, device against CPU', worst, tolerance)
    if not central:
        return
    worst = max(abs(row["dHdl.coulomb"] - (row["dU.2"] - row["dU.1"]) / (2 * h))
                for row in device)
    report(f'{label}, dH/dl against central differences', worst, 2e-3)


# From the coordinates, with a checkpoint (and a new order) every 10 steps.
device = run('ordered', 'GPU', precision)
host = run('ordered_cpu', 'CPU', 'DOUBLE')
compare('with checkpoints', device, host, 2e-3)

# Continue the same state, then restart at another state with new forces.
start = 'checkpoint = "{}"'
device = run('continued', 'GPU', precision,
             start=start.format('ordered.h5'), steps=20)
host = run('continued_cpu', 'CPU', 'DOUBLE',
           start=start.format('ordered_cpu.h5'), steps=20)
compare('same-state continuation', device, host, 2e-3)

# From the checkpoint of state 0 at state 3, whose forces are computed anew.
device = run('other', 'GPU', precision, state=3,
             start=start.format('ordered.h5'), steps=20)
host = run('other_cpu', 'CPU', 'DOUBLE', state=3,
           start=start.format('ordered_cpu.h5'), steps=20)
compare("from another state", device, host, 2e-3, central=False)
