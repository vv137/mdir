#!/usr/bin/env python3
"""GROMACS with the protocol of run.py for the ethanol of
test/Driver/Inputs/fep, to compare with MDIR: the rows of single
configurations, and the runs of the 14 states of Appendix C.9 with files
that scripts/free-energy.py reads.

    gromacs_ethanol.py match WORK --gmx GMX --mdir MDIR [--grid 32]
        [--target CPU] [--precision DOUBLE] [--jobs 8] COORDINATES...
    gromacs_ethanol.py run WORK --gmx GMX [--ps 500] [--seed 1]
        [--stage all|equilibration|<state>] [--mdrun "-ntmpi 1 -ntomp 8"]
    gromacs_ethanol.py convert DHDL.xvg OUT.dhdl

WORK must hold eth_wat.top and eth_wat.gro of gromacs_inputs.py, the one
topology that both programs read.

The setup of GROMACS (`mdp` below): the Verlet scheme with a cutoff of
0.9 nm for both interactions and the modifiers `Potential-shift`; PME of
order 4 on a grid of 32 with `ewald-rtol` = erfc(3.2 x 0.9), which gives
beta = 3.2 /nm; no correction for the dispersion; constraints on the bonds
of hydrogen (LINCS) and SETTLE; `sd` with tau-t = 1 ps (a friction of
1/ps), 2 fs, 300 K; `C-rescale` with tau-p = 2 ps, every 10 steps, a
compressibility of 4.5e-5 /bar, 1 atm. The free energy: `couple-moltype`
the ethanol, from `vdw-q` to `none`, `couple-intramol = no`, the 14 states
as `coul-lambdas` and `vdw-lambdas`, `sc-alpha = 0.5`, `sc-power = 1`,
`sc-r-power = 6`, `sc-sigma = 0.3`, `sc-coul = no`, and the energy at
every state for each row (`calc-lambda-neighbors = -1`).

`match` runs, for each file of coordinates (Amber or GROMACS coordinates,
or a checkpoint of MDIR, `*.h5`, which needs the Python package of MDIR and
is written as GROMACS coordinates), every state for one evaluation in both
programs, MDIR with a control file as that of run.py with the potential
shift, GROMACS by `mdrun -rerun`, and prints the largest differences of
dH/dlambda and of the energies to the other states (those below --finite
kcal/mol in size), in kcal/mol.

`run` makes one set: minimization, 50 ps at constant volume, 200 ps at
1 atm at state 0 with the heavy atoms of the ethanol restrained (the
constants of run.py: 10 and 1 kcal/mol/A^2 of k d^2), then each state for
--ps from the end of the equilibration with a seed of its own. It writes
s<k>.dhdl and s<k>.toml in WORK, the rows in the format of MDIR's
free-energy file, in kcal/mol. --stage runs one stage, so that a GPU can be
released between them.

`convert` writes the rows of a dhdl.xvg of GROMACS in that format.
"""
import argparse
import math
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

KCAL = 4.184
COULOMB = [0.0, 0.25, 0.5, 0.75] + [1.0] * 10
VDW = [0.0] * 5 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
EWALD_RTOL = math.erfc(3.2 * 0.9)

COMMON = '''cutoff-scheme       = Verlet
nstlist             = 10
rvdw                = 0.9
rcoulomb            = 0.9
vdw-type            = Cut-off
vdw-modifier        = Potential-shift
coulombtype         = PME
coulomb-modifier    = Potential-shift
ewald-rtol          = {rtol:.9e}
fourier-nx          = {grid}
fourier-ny          = {grid}
fourier-nz          = {grid}
pme-order           = 4
DispCorr            = no
free-energy         = yes
init-lambda-state   = {state}
coul-lambdas        = {coulomb}
vdw-lambdas         = {vdw}
couple-moltype      = LIG
couple-lambda0      = vdw-q
couple-lambda1      = none
couple-intramol     = no
sc-alpha            = 0.5
sc-power            = 1
sc-r-power          = 6
sc-sigma            = 0.3
sc-coul             = no
calc-lambda-neighbors = -1
dhdl-derivatives    = yes
separate-dhdl-file  = yes
'''

DYNAMICS = '''integrator          = sd
dt                  = 0.002
nsteps              = {steps}
ld-seed             = {seed}
tc-grps             = System
tau-t               = 1.0
ref-t               = 300.0
constraints         = h-bonds
constraint-algorithm = lincs
comm-mode           = Linear
nstcomm             = 10
nstcalcenergy       = 10
nstenergy           = {interval}
nstdhdl             = {interval}
nstlog              = 5000
nstxout-compressed  = 0
'''

BAROSTAT = '''pcoupl              = C-rescale
pcoupltype          = isotropic
tau-p               = 2.0
nstpcouple          = 10
ref-p               = 1.01325
compressibility     = 4.5e-5
refcoord-scaling    = com
'''

# The restraints of the equilibration, inserted into the molecule of the
# ethanol: GROMACS takes (K/2) d^2, run.py k d^2.
RESTRAINTS = '''
#ifdef POSRES
[ position_restraints ]
1 1 POSRES_K POSRES_K POSRES_K
2 1 POSRES_K POSRES_K POSRES_K
3 1 POSRES_K POSRES_K POSRES_K
#endif

'''

MDIR = '''[input]
topology = "eth_wat.top"
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
lennard_jones_modifier = "POTENTIAL_SHIFT"
coulomb_modifier = "POTENTIAL_SHIFT"
electrostatics = "PME"
[pme]
beta = 0.32
grid = [{grid}, {grid}, {grid}]
influence = "SPME"
[constraints]
hydrogen_bonds = true
rigid_water = true
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

CONTROL = '''# Read by scripts/free-energy.py only: the runs are those of GROMACS.
[output]
free_energy = "s{k:02d}.dhdl"

[free_energy]
state = {k}

[free_energy.lambdas]
coulomb = {coulomb}
vdw     = {vdw}

[ensemble]
temperature = 300.0
'''


def lambdas(values):
    return ' '.join(f'{v:g}' for v in values)


def common(state, grid=32):
    return COMMON.format(rtol=EWALD_RTOL, grid=grid, state=state,
                         coulomb=lambdas(COULOMB), vdw=lambdas(VDW))


def call(command, cwd, log):
    with open(os.path.join(cwd, log), 'a') as file:
        result = subprocess.run(command, cwd=cwd, stdout=file,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        sys.exit(f'{" ".join(command)} failed; see {os.path.join(cwd, log)}')


def read_xvg(path):
    """The rows of a dhdl.xvg: (time, dH/dl coul, dH/dl vdw, energies to
    every state), in kcal/mol."""
    legends, rows = [], []
    with open(path) as file:
        for line in file:
            if line.startswith('@ s') and 'legend' in line:
                legends.append(line.split('"')[1])
            elif line and line[0] not in '#@':
                rows.append([float(v) for v in line.split()])
    coul = next(i for i, l in enumerate(legends)
                if l.startswith('dH') and 'coul' in l)
    vdw = next(i for i, l in enumerate(legends)
               if l.startswith('dH') and 'vdw' in l)
    foreign = [i for i, l in enumerate(legends) if ' to ' in l]
    result = []
    for row in rows:
        values = row[1:]
        result.append((row[0], values[coul] / KCAL, values[vdw] / KCAL,
                       [values[i] / KCAL for i in foreign]))
    return result


def write_dhdl(rows, path, states):
    with open(path, 'w') as file:
        file.write('# step time dHdl.coulomb dHdl.vdw ' +
                   ' '.join(f'dU.{k}' for k in range(states)) + '\n# - ps ' +
                   ' '.join(['kcal/mol'] * (2 + states)) + '\n')
        for time, coul, vdw, du in rows:
            file.write(f'{int(round(time / 0.002))} {time:.6f} ' + ' '.join(
                f'{v:.6f}' for v in [coul, vdw] + du) + '\n')


def convert(arguments):
    rows = read_xvg(arguments.xvg)
    write_dhdl(rows, arguments.out, len(rows[0][3]))


def write_gro(checkpoint, path, template):
    """The positions and the cell of a checkpoint of MDIR as GROMACS
    coordinates with seven decimals, the names from `template`."""
    import mdir
    c = mdir.read_checkpoint(checkpoint)
    lines = open(template).read().split('\n')
    count = int(lines[1])
    out = [lines[0], lines[1]]
    for i in range(count):
        x = c.positions[i]
        out.append(lines[2 + i][:20] +
                   ''.join(f'{v:12.7f}' for v in x))
    cell = c.cell.vectors
    out.append(''.join(f'{cell[a][a]:12.7f}' for a in range(3)))
    with open(path, 'w') as file:
        file.write('\n'.join(out) + '\n')


def match(arguments):
    work = arguments.work
    states = len(COULOMB)
    for index, source in enumerate(arguments.coordinates):
        tag = f'c{index}'
        coordinates = tag + '.gro'
        if source.endswith('.h5'):
            write_gro(source, os.path.join(work, coordinates),
                      os.path.join(work, 'eth_wat.gro'))
        elif os.path.abspath(source) != os.path.abspath(
                os.path.join(work, coordinates)):
            shutil.copy(source, os.path.join(work, coordinates))

        def run(k):
            name = f'{tag}-g{arguments.grid}-s{k:02d}'
            # GROMACS: one evaluation of the configuration.
            with open(os.path.join(work, name + '.mdp'), 'w') as file:
                file.write(common(k, arguments.grid) +
                           'integrator = md\nnsteps = 0\nnstcalcenergy = 1\n'
                           'nstenergy = 1\nnstdhdl = 1\n')
            log = name + '.gmx.log'
            call([arguments.gmx, 'grompp', '-f', name + '.mdp', '-c',
                  coordinates, '-p', 'eth_wat.top', '-o', name + '.tpr',
                  '-po', name + '.out.mdp', '-maxwarn',
                  str(arguments.maxwarn)], work, log)
            call([arguments.gmx, 'mdrun', '-deffnm', name, '-rerun',
                  coordinates, '-ntmpi', '1', '-ntomp', '1', '-nb', 'cpu',
                  '-pin', 'off'], work, log)
            _, coul, vdw, du = read_xvg(os.path.join(work, name + '.xvg'))[0]
            # MDIR: the row of the start.
            label = f'{name}-{arguments.target.lower()}'
            with open(os.path.join(work, label + '.toml'), 'w') as file:
                file.write(MDIR.format(
                    coordinates=coordinates, name=label, state=k,
                    coulomb=COULOMB, vdw=VDW, grid=arguments.grid,
                    target=arguments.target, precision=arguments.precision))
            call([arguments.mdir, 'run', label + '.toml'], work,
                 label + '.log')
            with open(os.path.join(work, label + '.dhdl')) as file:
                file.readline()
                file.readline()
                row = [float(v) for v in file.readline().split()[2:]]
            return row, [coul, vdw] + du

        with ThreadPoolExecutor(arguments.jobs) as pool:
            rows = list(pool.map(run, range(states)))
        print(f'# {source}: grid {arguments.grid}, MDIR {arguments.target} '
              f'{arguments.precision}; MDIR - GROMACS, kcal/mol')
        print('# state      dHdl.c      dHdl.v      max dU  energies '
              'compared')
        worst = [0.0, 0.0, 0.0]
        for k, (row, ref) in enumerate(rows):
            kept = [j for j in range(states)
                    if abs(ref[2 + j]) < arguments.finite]
            d = [row[0] - ref[0], row[1] - ref[1],
                 max((abs(row[2 + j] - ref[2 + j]) for j in kept),
                     default=0.0)]
            if abs(ref[1]) < arguments.finite * 100:
                worst = [max(w, abs(v)) for w, v in zip(worst, d)]
            print(f'{k:7d} {d[0]:11.2e} {d[1]:11.2e} {d[2]:11.2e}  '
                  f'{len(kept):3d}')
        print(f'# worst: dHdl.c {worst[0]:.2e}, dHdl.v {worst[1]:.2e}, '
              f'dU {worst[2]:.2e}')


def run(arguments):
    work = arguments.work
    mdrun = shlex.split(arguments.mdrun)
    states = list(zip(COULOMB, VDW))
    steps = int(round(arguments.ps / 0.002))

    # The topology with the restraints of the equilibration in the
    # molecule of the ethanol, the first of the file.
    text = open(os.path.join(work, 'eth_wat.top')).read()
    first = text.index('[ moleculetype ]')
    second = text.index('[ moleculetype ]', first + 1)
    with open(os.path.join(work, 'restrained.top'), 'w') as file:
        file.write(text[:second] + RESTRAINTS + text[second:])

    def stage(name, mdp, start, velocities=None, restraint=0.0, seed=0):
        with open(os.path.join(work, name + '.mdp'), 'w') as file:
            file.write(mdp)
            if restraint:
                # (K/2) d^2 in kJ/mol/nm^2 for k d^2 in kcal/mol/A^2.
                file.write(f'define = -DPOSRES -DPOSRES_K='
                           f'{2.0 * restraint * KCAL * 100.0:g}\n')
        command = [arguments.gmx, 'grompp', '-f', name + '.mdp', '-c', start,
                   '-r', 'eth_wat.gro', '-p', 'restrained.top', '-o',
                   name + '.tpr', '-po', name + '.out.mdp', '-maxwarn',
                   str(arguments.maxwarn)]
        if velocities:
            command += ['-t', velocities]
        call(command, work, name + '.gmx.log')
        call([arguments.gmx, 'mdrun', '-deffnm', name, '-notunepme', '-pin',
              'off'] + mdrun, work, name + '.gmx.log')

    if arguments.stage in ('all', 'equilibration'):
        stage('min', common(0) + 'integrator = steep\nnsteps = 2000\n'
              'emtol = 0\nconstraints = h-bonds\n', 'eth_wat.gro',
              restraint=10.0)
        stage('nvt', common(0) + DYNAMICS.format(
            steps=25000, seed=arguments.seed, interval=2500) +
            'gen-vel = yes\ngen-temp = 300.0\n'
            f'gen-seed = {arguments.seed}\n', 'min.gro', restraint=10.0)
        stage('npt', common(0) + DYNAMICS.format(
            steps=100000, seed=arguments.seed + 1, interval=5000) + BAROSTAT +
            'continuation = yes\n', 'nvt.gro', 'nvt.cpt', restraint=1.0)
        if arguments.stage == 'equilibration':
            return
    for k in range(len(states)):
        if arguments.stage not in ('all', str(k)):
            continue
        name = f's{k:02d}'
        stage(name, common(k) + DYNAMICS.format(
            steps=steps, seed=arguments.seed + 10 * (k + 1),
            interval=arguments.energy_interval) + BAROSTAT +
            'continuation = yes\n', 'npt.gro', 'npt.cpt')
        rows = read_xvg(os.path.join(work, name + '.xvg'))
        # The row of the start is not one of the run's samples, as in
        # MDIR's file.
        write_dhdl([r for r in rows if r[0] > 0.0],
                   os.path.join(work, name + '.dhdl'), len(states))
        with open(os.path.join(work, name + '.toml'), 'w') as file:
            file.write(CONTROL.format(k=k, coulomb=COULOMB, vdw=VDW))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    m = commands.add_parser('match')
    m.add_argument('work')
    m.add_argument('coordinates', nargs='+')
    m.add_argument('--gmx', default='gmx')
    m.add_argument('--mdir', default='mdir')
    m.add_argument('--grid', type=int, default=32)
    m.add_argument('--target', default='CPU')
    m.add_argument('--precision', default='DOUBLE')
    m.add_argument('--jobs', type=int, default=8)
    m.add_argument('--finite', type=float, default=100.0)
    m.add_argument('--maxwarn', type=int, default=0)
    m.set_defaults(function=match)
    r = commands.add_parser('run')
    r.add_argument('work')
    r.add_argument('--gmx', default='gmx')
    r.add_argument('--ps', type=float, default=500.0)
    r.add_argument('--seed', type=int, default=1)
    r.add_argument('--energy-interval', type=int, default=250)
    r.add_argument('--stage', default='all')
    r.add_argument('--mdrun', default='-ntmpi 1 -ntomp 8')
    r.add_argument('--maxwarn', type=int, default=0)
    r.set_defaults(function=run)
    c = commands.add_parser('convert')
    c.add_argument('xvg')
    c.add_argument('out')
    c.set_defaults(function=convert)
    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == '__main__':
    main()
