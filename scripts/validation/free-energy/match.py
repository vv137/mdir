#!/usr/bin/env python3
"""The rows of the free-energy file of MDIR against OpenMM with the same
Hamiltonian (openmm_ethanol.py) on the same positions and cell, for the
ethanol of test/Driver/Inputs/fep and the 14 states of Appendix C.9.

    match.py WORK --mdir MDIR [--label NAME] [--grid 32] [--target CPU]
        [--precision DOUBLE] [--platform Reference] [--jobs 8]
        COORDINATES...

COORDINATES are Amber coordinates with a cell, or checkpoints of MDIR
(`*.h5`, which need the Python package of MDIR on PYTHONPATH and are written
out as Amber coordinates with seven decimals, so that both programs read
the same numbers). For each, MDIR runs every state for one step and the
row of its start is compared with OpenMM's energies:

- `cut`: the energy of each pair cut at the cutoff, as OpenMM reports it;
- `shift`: the Lennard-Jones of the pairs of the ethanol shifted to 0 at
  the cutoff (the parameter `shift` of openmm_ethanol.py) and the direct
  sum of particle mesh Ewald shifted by S, summed from the coordinates:
  dH/dlambda_C + S, dU.k + (lambda_C(k) - lambda_C(state)) S (D210).

It prints, for each state, the largest difference of dH/dlambda and of the
energies to the other states (those below --finite kcal/mol in size: at a
state far from that of the sample the energy can be of any size), in
kcal/mol, against both.
"""
import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import openmm_ethanol  # noqa: E402

COULOMB, VDW = openmm_ethanol.COULOMB, openmm_ethanol.VDW

CONTROL = '''[input]
topology = "eth_wat.prmtop"
coordinates = "{coordinates}"
[output]
energy_interval = 1
free_energy = "{name}.dhdl"
[free_energy]
couple = ":LIG"
state = {state}
soft_core_alpha = 0.5
[free_energy.lambdas]
coulomb = {coulomb}
vdw = {vdw}
[energy]
cutoff = 9.0
switch_distance = 9.0
pairlist_distance = 10.0
dispersion_correction = "NONE"
electrostatics = "PME"
[pme]
beta = 0.32
grid = [{grid}, {grid}, {grid}]
influence = "SPME"
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


def write_inpcrd(checkpoint, path):
    import mdir
    c = mdir.read_checkpoint(checkpoint)
    x = (c.positions * 10.0).ravel()
    lines = ['the state of a checkpoint', f'{len(x) // 3:6d}']
    for i in range(0, len(x), 6):
        lines.append(''.join(f'{v:12.7f}' for v in x[i:i + 6]))
    cell = c.cell.vectors
    lines.append(''.join(f'{v:12.7f}' for v in (
        cell[0][0] * 10, cell[1][1] * 10, cell[2][2] * 10, 90.0, 90.0,
        90.0)))
    with open(path, 'w') as file:
        file.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('work')
    parser.add_argument('coordinates', nargs='+')
    parser.add_argument('--mdir', default='mdir')
    parser.add_argument('--label', default='mdir')
    parser.add_argument('--grid', type=int, default=32)
    parser.add_argument('--target', default='CPU')
    parser.add_argument('--precision', default='DOUBLE')
    parser.add_argument('--platform', default='Reference')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--finite', type=float, default=100.0)
    arguments = parser.parse_args()

    work = arguments.work
    os.makedirs(work, exist_ok=True)
    inputs = os.path.join(here, '..', '..', '..', 'test', 'Driver', 'Inputs',
                          'fep')
    prmtop = os.path.join(work, 'eth_wat.prmtop')
    shutil.copy(os.path.join(inputs, 'eth_wat.prmtop'), prmtop)
    states = len(COULOMB)

    for index, source in enumerate(arguments.coordinates):
        tag = f'c{index}'
        coordinates = os.path.join(work, tag + '.inpcrd')
        if source.endswith('.h5'):
            write_inpcrd(source, coordinates)
        else:
            shutil.copy(source, coordinates)

        # OpenMM, once for each grid, kept beside the coordinates.
        reference = os.path.join(work, f'{tag}.openmm{arguments.grid}')
        if not os.path.exists(reference):
            result = subprocess.run(
                [sys.executable, os.path.join(here, 'openmm_ethanol.py'),
                 'point', prmtop, coordinates, '--grid', str(arguments.grid),
                 '--platform', arguments.platform],
                capture_output=True, text=True)
            if result.returncode:
                sys.exit(result.stderr)
            with open(reference, 'w') as file:
                file.write(result.stdout)
        openmm = {'cut': {}, 'shift': {}}
        for line in open(reference):
            fields = line.split()
            if fields[0] == '#':
                s = float(fields[2])
            else:
                openmm[fields[0]][int(fields[1])] = [float(v)
                                                     for v in fields[2:]]

        def run(k):
            name = f'{tag}-{arguments.label}-g{arguments.grid}-s{k:02d}'
            with open(os.path.join(work, name + '.toml'), 'w') as file:
                file.write(CONTROL.format(
                    coordinates=tag + '.inpcrd', name=name, state=k,
                    coulomb=COULOMB, vdw=VDW, grid=arguments.grid,
                    target=arguments.target, precision=arguments.precision))
            result = subprocess.run(
                [arguments.mdir, 'run', name + '.toml'], cwd=work,
                capture_output=True, text=True)
            if result.returncode:
                sys.exit(f'mdir run {name}.toml failed:\n{result.stderr}')
            with open(os.path.join(work, name + '.dhdl')) as file:
                file.readline()
                file.readline()
                return [float(v) for v in file.readline().split()[2:]]

        with ThreadPoolExecutor(arguments.jobs) as pool:
            rows = list(pool.map(run, range(states)))

        print(f'# {source}: {arguments.label}, grid {arguments.grid}, '
              f'{arguments.target} {arguments.precision}; S = {s:.6f} '
              f'kcal/mol; MDIR - OpenMM, kcal/mol')
        print('# state  dHdl.c cut  dHdl.v cut   max dU cut  dHdl.c shift  '
              'dHdl.v shift  max dU shift  energies compared')
        worst = {'cut': [0.0, 0.0, 0.0], 'shift': [0.0, 0.0, 0.0]}
        for k, row in enumerate(rows):
            out = []
            count = 0
            for name in ('cut', 'shift'):
                ref = list(openmm[name][k])
                if name == 'shift':
                    ref[0] += s
                    for j in range(states):
                        ref[2 + j] += (COULOMB[j] - COULOMB[k]) * s
                kept = [j for j in range(states)
                        if abs(ref[2 + j]) < arguments.finite]
                count = len(kept)
                d = [row[0] - ref[0], row[1] - ref[1],
                     max((abs(row[2 + j] - ref[2 + j]) for j in kept),
                         default=0.0)]
                worst[name] = [max(w, abs(v)) for w, v in zip(worst[name], d)]
                out.append(f'{d[0]:11.2e} {d[1]:11.2e} {d[2]:11.2e}')
            print(f'{k:7d} ' + '  '.join(out) + f'  {count:3d}')
        for name in ('cut', 'shift'):
            print(f'# worst against {name}: dHdl.c {worst[name][0]:.2e}, '
                  f'dHdl.v {worst[name][1]:.2e}, dU {worst[name][2]:.2e}')


if __name__ == '__main__':
    main()
