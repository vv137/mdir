#!/usr/bin/env python3
"""The state against the time step under each thermostat (#268,
D263, D264, D265).

    scripts/validation/kinetic-step/run.py WORK --mdir MDIR
        [--system ethanol|water] [--ns 5] [--steps 0.5,1,2,4]
        [--thermostats vr,nh,lg] [--ensembles nvt,npt]
        [--precision MIXED|DOUBLE] [--target GPU|CPU] [--seed 20261016]
        [--deterministic]
        [--tleap TLEAP] [--write-only] [--parallel 4] [--only NAME,...]

WORK receives the system and the control files:

  ethanol   test/Driver/Inputs/fep: ethanol with GAFF2 in 467 TIP3P
            waters, SETTLE and SHAKE on the bonds of hydrogen (1410 atoms)
  water     a box of TIP3P water that tleap makes (solvateBox with a
            buffer of 11 Å), SETTLE

  1-min.toml, 2-nvt.toml, 3-npt.toml
            minimization, 50 ps at 300 K and constant volume, and 500 ps at
            300 K and 1 atm, Langevin dynamics with a friction of 1/ps at
            1 fs: the state eq.h5 that every run below begins from
  <t>-<dt>fs-<e>.toml
            --ns at 300 K, with the thermostat t (vr: stochastic velocity
            rescaling, nh: a Nosé–Hoover chain, both with a time constant
            of 0.5 ps every 10 steps; lg: Langevin dynamics with a friction
            of 1/ps), the step dt, and the ensemble e (nvt: the volume of
            eq.h5; npt: 1 atm, stochastic cell rescaling with a time
            constant of 2 ps every 10 steps)

All runs: PME with beta = 0.32 per Å on a grid of 32, a cutoff of 9 Å with
the potential shifted (so that the log's energy is that of the potential
the forces sample), no correction for the dispersion, a row every 20 fs.
Unless --write-only, the stages run first if eq.h5 is missing, then the
runs, --parallel of them at once on the device of the job.
analyze.py reads the logs.
"""
import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

LEAP = """source leaprc.water.tip3p
m = copy WAT
solvateBox m TIP3PBOX 11.0 iso
saveAmberParm m water.prmtop water.inpcrd
quit
"""

HEAD = """[input]
topology    = "{prmtop}"
coordinates = "{inpcrd}"
{checkpoint_in}
[output]
checkpoint      = "{name}.h5"
checkpoint_interval = {steps}
energy_interval = {energy_interval}

[energy]
cutoff            = 9.0
pairlist_distance = 10.5
dispersion_correction = "NONE"
electrostatics    = "PME"
lennard_jones_modifier = "POTENTIAL_SHIFT"
coulomb_modifier       = "POTENTIAL_SHIFT"

[pme]
beta = 0.32
grid = [32, 32, 32]
"""

MINIMIZE = """
[minimize]
method = "STEEPEST_DESCENT"
steps  = 2000
"""

DYNAMICS = """
[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = {dt}
steps      = {steps}
seed       = {seed}

[ensemble]
ensemble    = "{ensemble}"
temperature = 300.0
{pressure}
"""

THERMOSTATS = {
    'vr': '[thermostat]\nmethod = "V-RESCALE"\ntime_constant = 0.5\n'
          'interval = 10\n',
    'nh': '[thermostat]\nmethod = "NOSE-HOOVER"\ntime_constant = 0.5\n'
          'interval = 10\n',
    'lg': '[thermostat]\nmethod = "LANGEVIN"\nfriction = 1.0\n',
}

BAROSTAT = """
[barostat]
method        = "C-RESCALE"
time_constant = 2.0
interval      = 10
"""

TAIL = """
[constraints]
hydrogen_bonds = true
rigid_water    = true

[boundary]
type = "PERIODIC"

[execution]
target    = "{target}"
precision = "{precision}"
{deterministic}"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('work')
    parser.add_argument('--mdir', default='mdir')
    parser.add_argument('--system', default='ethanol',
                        choices=['ethanol', 'water'])
    parser.add_argument('--ns', type=float, default=5.0)
    parser.add_argument('--steps', default='0.5,1,2,4',
                        help='the time steps, fs')
    parser.add_argument('--thermostats', default='vr,nh,lg')
    parser.add_argument('--ensembles', default='nvt,npt')
    parser.add_argument('--precision', default='MIXED')
    parser.add_argument('--target', default='GPU')
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--tleap', default='tleap')
    parser.add_argument('--write-only', action='store_true')
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--only', default='',
                        help='the runs to run, by name; all if empty')
    args = parser.parse_args()

    work = args.work
    os.makedirs(work, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    if args.system == 'ethanol':
        prmtop, inpcrd = 'eth_wat.prmtop', 'eth_wat.inpcrd'
        inputs = os.path.join(here, '..', '..', '..', 'test', 'Driver',
                              'Inputs', 'fep')
        for name in (prmtop, inpcrd):
            shutil.copy(os.path.join(inputs, name), work)
    else:
        prmtop, inpcrd = 'water.prmtop', 'water.inpcrd'
        if not os.path.exists(os.path.join(work, prmtop)):
            with open(os.path.join(work, 'box.leap'), 'w') as file:
                file.write(LEAP)
            run([args.tleap, '-f', 'box.leap'], work, 'tleap.log')

    tail = TAIL.format(target=args.target, precision=args.precision,
                       deterministic='deterministic = true\n'
                       if args.deterministic else '')

    def head(name, checkpoint, energy_interval, steps):
        return HEAD.format(
            prmtop=prmtop, inpcrd=inpcrd, name=name, steps=steps,
            energy_interval=energy_interval,
            checkpoint_in=f'checkpoint  = "{checkpoint}.h5"\n'
            if checkpoint else '')

    def dynamics(dt, steps, seed, npt):
        return DYNAMICS.format(
            dt=dt, steps=steps, seed=seed,
            ensemble='NPT' if npt else 'NVT',
            pressure='pressure    = 1.0\n' if npt else '')

    files = {
        '1-min': head('min', None, 500, 2000) + MINIMIZE + tail,
        '2-nvt': head('nvt', 'min', 5000, 50000) +
        dynamics(0.001, 50000, args.seed, False) + THERMOSTATS['lg'] + tail,
        '3-npt': head('eq', 'nvt', 5000, 500000) +
        dynamics(0.001, 500000, args.seed, True) + THERMOSTATS['lg'] +
        BAROSTAT + tail,
    }
    stages = list(files)
    runs = []
    for k, thermostat in enumerate(args.thermostats.split(',')):
        for step in args.steps.split(','):
            dt = float(step) * 1e-3
            for ensemble in args.ensembles.split(','):
                name = f'{thermostat}-{step}fs-{ensemble}'
                npt = ensemble == 'npt'
                steps = int(round(args.ns * 1000.0 / dt))
                # A row every 20 fs, a multiple of the interval of coupling.
                interval = 10 * max(1, int(round(0.02 / dt / 10)))
                files[name] = (
                    head(name, 'eq', interval, steps) +
                    dynamics(dt, steps, args.seed + 1 + k, npt) +
                    THERMOSTATS[thermostat] + (BAROSTAT if npt else '') +
                    tail)
                runs.append(name)
    for name, text in files.items():
        with open(os.path.join(work, name + '.toml'), 'w') as file:
            file.write(text)
    if args.write_only:
        return
    if not os.path.exists(os.path.join(work, 'eq.h5')):
        for name in stages:
            run([args.mdir, 'run', name + '.toml'], work, name + '.log')
    if args.only:
        runs = [name for name in runs if name in args.only.split(',')]
    with ThreadPoolExecutor(args.parallel) as pool:
        for failure in pool.map(
                lambda name: run([args.mdir, 'run', name + '.toml'], work,
                                 name + '.log', stop=False), runs):
            if failure:
                print(failure, file=sys.stderr)


def run(command, cwd, log, stop=True):
    with open(os.path.join(cwd, log), 'w') as out:
        result = subprocess.run(command, cwd=cwd, stdout=out,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        message = f"{' '.join(command)} failed; see {log}"
        if stop:
            sys.exit(message)
        return message
    return None


if __name__ == '__main__':
    main()
