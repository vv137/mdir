"""Checks [free_energy] (D161) on ethanol in TIP3P (Inputs/fep), decoupled
from the water, at the coordinates of tleap.

    check_free_energy.py <directory> <CPU|GPU> <DOUBLE|MIXED> <mdir>

Runs mdir in `directory` and prints one line per check:

- reaction field: dH/dλ of the Coulomb and of the Lennard-Jones and
  U(state 0) − U(state k) at three states against OpenMM 8.6.1 (Reference
  platform; NonbondedForce with charge offsets, the pairs within the ethanol
  as exceptions at full strength, the ethanol-water Lennard-Jones in a
  CustomNonbondedForce with the soft-core of Beutler et al. and an
  interaction group);
- particle mesh Ewald: the same against OpenMM with the same β and grid,
  within what the B-splines of order 4 (MDIR) and 5 (OpenMM) leave,
  2.8e-3 kcal/mol, which falls to 5e-5 on a grid of 80;
- dH/dλ against the central differences of the energies of the states
  beside it, in the Coulomb, the Lennard-Jones, and a component that only
  expressions take (a bond, a term over centers, a term of the positions);
- the energies of the other states against runs at those states.
"""

import os
import shutil
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
directory, target, precision, mdir = sys.argv[1:5]
for name in ('eth_wat.prmtop', 'eth_wat.inpcrd'):
    shutil.copy(os.path.join(here, 'fep', name), directory)

STATES = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.3), (1.0, 0.7),
          (1.0, 1.0)]
# OpenMM: (dH/dλ_coulomb, dH/dλ_vdw, U(state 0) − U(state k)), kcal/mol.
OPENMM = {
    'RF': [(-0.349729, 3.926729, 0.0), (-0.349729, 3.926729, 0.174864),
           (-0.349729, 3.926729, 0.349729), (-0.349729, 3.850620, -0.816828),
           (-0.349729, 3.752466, -2.337305), (-0.349729, 3.681771, -3.452375)],
    'PME': [(-0.269445, 3.926729, 0.0), (-0.273035, 3.926729, 0.135620),
            (-0.276625, 3.926729, 0.273035), (-0.276625, 3.850620, -0.893522),
            (-0.276625, 3.752466, -2.413999), (-0.276625, 3.681771, -3.529069)],
    # The reaction field with the Lennard-Jones shifted to 0 at the cutoff,
    # the shift of the soft-core taken at the cutoff, in σ.
    'RF_SHIFT': [(-0.349729, 3.524760, 0.0), (-0.349729, 3.524760, 0.174864),
                 (-0.349729, 3.524760, 0.349729),
                 (-0.349729, 3.448651, -0.696237),
                 (-0.349729, 3.350496, -2.055926),
                 (-0.349729, 3.279802, -3.050406)],
}
ELECTROSTATICS = {
    'RF': 'electrostatics = "REACTION_FIELD"\nreaction_field_dielectric = 78.3\n',
    'PME': 'electrostatics = "PME"\n[pme]\nbeta = 0.32\ngrid = [32, 32, 32]\n'
           'influence = "SPME"\n',
    'RF_SHIFT': 'electrostatics = "REACTION_FIELD"\n'
                'reaction_field_dielectric = 78.3\n'
                'lennard_jones_modifier = "POTENTIAL_SHIFT"\n',
}


def control(name, state, coulomb, vdw, electrostatics, extra='',
            components=''):
    text = f'''[input]
topology = "eth_wat.prmtop"
coordinates = "eth_wat.inpcrd"
[output]
energy_interval = 1
free_energy = "{name}.dhdl"
[free_energy]
couple = ":LIG"
state = {state}
[free_energy.lambdas]
coulomb = {coulomb}
vdw = {vdw}
{components}
[energy]
cutoff = 9.0
switch_distance = 9.0
pairlist_distance = 10.0
dispersion_correction = "NONE"
{ELECTROSTATICS[electrostatics]}
{extra}
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.001
steps = 1
[ensemble]
ensemble = "NVE"
temperature = 300.0
[boundary]
type = "PERIODIC"
[execution]
target = "{target}"
precision = "{precision}"
'''
    path = os.path.join(directory, name + '.toml')
    with open(path, 'w') as file:
        file.write(text)
    result = subprocess.run([mdir, 'run', name + '.toml'], cwd=directory,
                            capture_output=True, text=True)
    if result.returncode:
        sys.exit(f'mdir run {name}.toml failed:\n{result.stdout}\n'
                 f'{result.stderr}')
    with open(os.path.join(directory, name + '.dhdl')) as file:
        header = file.readline().split()[1:]
        file.readline()
        row = [float(v) for v in file.readline().split()]
    return dict(zip(header, row))


def report(label, worst, tolerance):
    verdict = 'ok' if worst <= tolerance else 'FAILED'
    print(f'{label}: {verdict} (worst {worst:.2e}, tolerance {tolerance:.0e})')


# Against OpenMM. In mixed precision the kernels of the pairs are in f32.
mixed = precision == 'MIXED'
for electrostatics, tolerance in (('RF', 5e-5 if mixed else 3e-6),
                                  ('RF_SHIFT', 5e-5 if mixed else 3e-6),
                                  ('PME', 4e-3)):
    coulomb = [s[0] for s in STATES]
    vdw = [s[1] for s in STATES]
    worst = 0.0
    worst_vdw = 0.0
    worst_vdw_energy = 0.0
    for k in (1, 3, 5):
        row = control(f'{electrostatics.lower()}{k}', k, coulomb, vdw,
                      electrostatics)
        reference = OPENMM[electrostatics]
        worst_vdw = max(worst_vdw, abs(row['dHdl.vdw'] - reference[k][1]))
        worst = max(worst, abs(row['dHdl.coulomb'] - reference[k][0]),
                    abs(row['dHdl.vdw'] - reference[k][1]))
        for j in range(len(STATES)):
            expected = reference[k][2] - reference[j][2]
            worst = max(worst, abs(row[f'dU.{j}'] - expected))
            if STATES[k][0] == STATES[j][0]:
                worst_vdw_energy = max(worst_vdw_energy,
                                       abs(row[f'dU.{j}'] - expected))
    report(f'{electrostatics} against OpenMM', worst, tolerance)
    if electrostatics == 'RF_SHIFT':
        # The Coulomb and total-energy differences include their f32 error;
        # pin the shifted Lennard-Jones derivative separately to catch #48.
        report('RF_SHIFT dH/dl.vdw against OpenMM', worst_vdw,
               1e-5 if mixed else 3e-6)
        report('RF_SHIFT vdW dU against OpenMM', worst_vdw_energy,
               1e-5 if mixed else 3e-6)

# Central differences: the Coulomb is quadratic in λ, the rest smooth.
h = 1e-2
coulomb = [0.5, 0.5 - h, 0.5 + h, 1.0, 1.0, 1.0, 0.5, 0.5]
vdw = [0.0, 0.0, 0.0, 0.5, 0.5 - h, 0.5 + h, 0.0, 0.0]
restraint = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3 - h, 0.3 + h]
extra = '''[[energy.bond]]
name = "hold"
particles = [[3, 100]]
expression = "lambda_restraint^2 * k * (r - r0)^2"
k = 2.0
r0 = 5.0
[[energy.bond]]
name = "pull"
groups = [":LIG", ":10"]
expression = "lambda_restraint * k * (r - r0)^2"
k = 1.5
r0 = 6.0
[[energy.external]]
name = "wall"
selection = ":LIG"
expression = "lambda_restraint * k * (z - 10.0)^2"
k = 0.5
'''
components = f'restraint = {restraint}'
runs = {k: control(f'central{k}', k, coulomb, vdw, 'PME', extra, components)
        for k in (0, 3, 7)}
first, second = runs[0], runs[3]
differences = {
    'coulomb': (first['dHdl.coulomb'],
                (first['dU.2'] - first['dU.1']) / (2 * h)),
    'vdw': (second['dHdl.vdw'], (second['dU.5'] - second['dU.4']) / (2 * h)),
    'restraint': (first['dHdl.restraint'],
                  (first['dU.7'] - first['dU.6']) / (2 * h)),
}
worst = max(abs(a - b) for a, b in differences.values())
report('dH/dl against central differences', worst,
       2e-3 if mixed else 1e-4)

# The energies of the other states against runs at those states:
# U_j - U_k from the run at k against -(U_k - U_j) from the run at j.
worst = max(abs(runs[k][f'dU.{j}'] + runs[j][f'dU.{k}'])
            for k in runs for j in runs)
report('the energies of the other states', worst, 1e-4 if mixed else 3e-6)

# At λ = 0 the run is the run without [free_energy]: its potential energy
# at every row of the log, with each electrostatics and each modifier of
# the Lennard-Jones that [free_energy] takes.
IDENTITY = '''[input]
topology = "eth_wat.prmtop"
coordinates = "eth_wat.inpcrd"
[output]
energy_interval = 2
energy = "{name}.energy"
{free}
[energy]
cutoff = 9.0
switch_distance = 9.0
pairlist_distance = 10.0
{electrostatics}
{modifier}
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.001
steps = 6
[ensemble]
ensemble = "NVE"
temperature = 300.0
[boundary]
type = "PERIODIC"
[execution]
target = "{target}"
precision = "{precision}"
'''
FREE = """[free_energy]
couple = ":LIG"
state = 0
[free_energy.lambdas]
coulomb = [0.0, 1.0]
vdw = [0.0, 1.0]
"""


def potentials(name, free, electrostatics, modifier):
    text = IDENTITY.format(name=name, free=free, target=target,
                           precision=precision, modifier=modifier,
                           electrostatics=electrostatics)
    with open(os.path.join(directory, name + '.toml'), 'w') as file:
        file.write(text)
    result = subprocess.run([mdir, 'run', name + '.toml'], cwd=directory,
                            capture_output=True, text=True)
    if result.returncode:
        sys.exit(f'mdir run {name}.toml failed:\n{result.stdout}\n'
                 f'{result.stderr}')
    with open(os.path.join(directory, name + '.energy')) as file:
        header = file.readline().split()[1:]
        file.readline()
        column = header.index('potential')
        return [float(line.split()[column]) for line in file]


worst = 0.0
for label, electrostatics in (
        ('cutoff', 'electrostatics = "CUTOFF"'),
        ('pme', 'electrostatics = "PME"'),
        ('rf', ELECTROSTATICS['RF'])):
    for modifier in ('', 'lennard_jones_modifier = "POTENTIAL_SHIFT"\n'
                         'dispersion_correction = "NONE"'):
        name = f'identity_{label}_{"shift" if modifier else "none"}'
        plain = potentials(name + '_plain', '', electrostatics, modifier)
        free = potentials(name + '_free', FREE, electrostatics, modifier)
        if len(plain) != len(free):
            worst = float('inf')
            continue
        # On a device the sums of the default mode are not reproducible
        # from run to run (D84), so two runs part after their first steps
        # whatever their Hamiltonians; there only the first row, at the
        # same configuration, tests the identity.
        if target == 'GPU':
            plain, free = plain[:1], free[:1]
        worst = max([worst] + [abs(a - b) for a, b in zip(plain, free)])
report('lambda = 0 against no [free_energy]', worst,
       1e-4 if mixed else 1e-6)
