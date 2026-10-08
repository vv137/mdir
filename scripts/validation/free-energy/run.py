#!/usr/bin/env python3
"""The hydration free energy of ethanol of Section 6.8 of the paper: the
control files of its stages and of its 14 states, and their runs.

    scripts/validation/free-energy/run.py WORK --mdir MDIR [--ps 500]
        [--energy-interval 250] [--seed 20261008] [--deterministic]
        [--equilibrated DIR] [--write-only]
    python3 scripts/free-energy.py WORK/s*.toml

WORK receives the inputs of test/Driver/Inputs/fep (ethanol with GAFF2 and
AM1-BCC charges in 467 TIP3P waters) and

  1-min.toml     minimization, 2000 steps, the heavy atoms of the ethanol
                 restrained (10 kcal/mol/Å²)
  2-nvt.toml     50 ps at 300 K and constant volume, the same restraints
  3-npt.toml     200 ps at 300 K and 1 atm, the restraints at 1 kcal/mol/Å²
  s00.toml ...   the 14 states of Appendix C.9, each --ps at 300 K and 1 atm
  s13.toml       from npt.h5, with the free-energy file every 250 steps

Every stage is at state 0 of [free_energy] but the runs of the states:
Langevin dynamics with a friction of 1/ps, SETTLE and SHAKE, 2 fs, particle
mesh Ewald with β = 0.32 Å⁻¹ on a grid of 32, a cutoff of 9 Å, no
correction for the dispersion, stochastic cell rescaling with a time
constant of 2 ps every 10 steps (the scaling of Trotter type, its default),
mixed precision on a GPU. The state k takes the seed --seed + k, the
stages before it --seed.

--equilibrated DIR copies npt.h5 from DIR instead of running the three
stages, so that two sets of runs begin from one state; with --deterministic
a build repeats its runs bit for bit, and compare.py takes the differences
of two sets. --write-only writes the control files and runs nothing. The
runs take the GPU that CUDA_VISIBLE_DEVICES names.
"""
import argparse
import os
import shutil
import subprocess
import sys

COULOMB = [0.0, 0.25, 0.5, 0.75] + [1.0] * 10
VDW = [0.0] * 5 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]

HEAD = """[input]
topology    = "eth_wat.prmtop"
coordinates = "eth_wat.inpcrd"
{checkpoint_in}
[output]
checkpoint      = "{name}.h5"
energy_interval = {energy_interval}
{output}
[free_energy]
couple = ":LIG"
state  = {state}
soft_core_alpha = 0.5

[free_energy.lambdas]
coulomb = {coulomb}
vdw     = {vdw}

[energy]
cutoff            = 9.0
switch_distance   = 9.0
pairlist_distance = 10.0
dispersion_correction = "NONE"
electrostatics    = "PME"

[pme]
beta = 0.32
grid = [32, 32, 32]
influence = "SPME"
"""

MINIMIZE = """
[minimize]
method = "STEEPEST_DESCENT"
steps  = 2000
"""

DYNAMICS = """
[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.002
steps      = {steps}
seed       = {seed}

[ensemble]
ensemble    = "{ensemble}"
temperature = 300.0
{pressure}
[thermostat]
method   = "LANGEVIN"
friction = 1.0
"""

BAROSTAT = """
[barostat]
method        = "C-RESCALE"
time_constant = 2.0
"""

RESTRAINTS = """
[[restraints]]
selection      = ":LIG & !@H*"
force_constant = {force_constant}
"""

TAIL = """
[constraints]
hydrogen_bonds = true
rigid_water    = true

[boundary]
type = "PERIODIC"

[execution]
target    = "GPU"
precision = "MIXED"
{deterministic}"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('work')
    parser.add_argument('--mdir', default='mdir')
    parser.add_argument('--ps', type=float, default=500.0,
                        help='the length of the run of each state')
    parser.add_argument('--energy-interval', type=int, default=250,
                        help='steps between rows of the free-energy file')
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--equilibrated', metavar='DIR')
    parser.add_argument('--write-only', action='store_true')
    arguments = parser.parse_args()

    work = arguments.work
    os.makedirs(work, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    inputs = os.path.join(here, '..', '..', '..', 'test', 'Driver', 'Inputs',
                          'fep')
    for name in ('eth_wat.prmtop', 'eth_wat.inpcrd'):
        shutil.copy(os.path.join(inputs, name), work)

    tail = TAIL.format(deterministic='deterministic = true\n'
                       if arguments.deterministic else '')

    def head(name, state, checkpoint, energy_interval, steps=0, output=''):
        if steps:
            output += f'checkpoint_interval = {steps}\n'
        return HEAD.format(
            name=name, state=state, energy_interval=energy_interval,
            output=output, coulomb=COULOMB, vdw=VDW,
            checkpoint_in=f'checkpoint  = "{checkpoint}.h5"\n'
            if checkpoint else '')

    def dynamics(steps, seed, pressure):
        text = DYNAMICS.format(
            steps=steps, seed=seed, ensemble='NPT' if pressure else 'NVT',
            pressure='pressure    = 1.0\n' if pressure else '')
        return text + (BAROSTAT if pressure else '')

    files = {
        '1-min': head('min', 0, None, 500) + MINIMIZE +
        RESTRAINTS.format(force_constant=10.0) + tail,
        '2-nvt': head('nvt', 0, 'min', 2500, 25000) +
        dynamics(25000, arguments.seed, False) +
        RESTRAINTS.format(force_constant=10.0) + tail,
        '3-npt': head('npt', 0, 'nvt', 5000, 100000) +
        dynamics(100000, arguments.seed, True) +
        RESTRAINTS.format(force_constant=1.0) + tail,
    }
    stages = [] if arguments.equilibrated else list(files)
    steps = int(round(arguments.ps / 0.002))
    for k in range(len(COULOMB)):
        name = f's{k:02d}'
        files[name] = (
            head(name, k, 'npt', arguments.energy_interval, steps,
                 f'free_energy     = "{name}.dhdl"\n') +
            dynamics(steps, arguments.seed + k, True) + tail)
        stages.append(name)
    for name, text in files.items():
        with open(os.path.join(work, name + '.toml'), 'w') as file:
            file.write(text)
    if arguments.equilibrated:
        shutil.copy(os.path.join(arguments.equilibrated, 'npt.h5'), work)
    if arguments.write_only:
        return
    for name in stages:
        with open(os.path.join(work, name + '.log'), 'w') as log:
            result = subprocess.run(
                [arguments.mdir, 'run', '--continue', name + '.toml'],
                cwd=work, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            sys.exit(f'mdir run {name}.toml failed; see {name}.log')


if __name__ == '__main__':
    main()
