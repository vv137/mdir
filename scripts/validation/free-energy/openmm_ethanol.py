#!/usr/bin/env python3
"""OpenMM with the Hamiltonian of [free_energy] (D161) for the ethanol of
test/Driver/Inputs/fep, to compare with MDIR: single points, and the runs of
the 14 states of Appendix C.9 with files that scripts/free-energy.py reads.

    openmm_ethanol.py point PRMTOP INPCRD [--electrostatics PME|RF]
        [--grid 32] [--platform Reference] [--states c0,v0 c1,v1 ...]
    openmm_ethanol.py run WORK [--ps 500] [--seed 1] [--precision mixed]
        [--energy-interval 250] [--stage all|equilibration|<state>]

The Hamiltonian, as the test of D161 describes it (test/Driver/Inputs/
check_free_energy.py):

- the charges of the ethanol are scaled by 1 - lambda_coulomb through
  offsets of the particles of the NonbondedForce (the direct and the
  reciprocal sums, the self term, and the excluded pairs follow them);
- every pair within the ethanol that is not excluded is an exception at
  full strength, with its charges and its Lennard-Jones;
- the Lennard-Jones of the ethanol with the water is taken out of the
  NonbondedForce (epsilon = 0 for the ethanol) and is a
  CustomNonbondedForce over the group of the ethanol and that of the rest,
  (1 - lambda_vdw) 4 eps (1/x^2 - 1/x), x = alpha lambda_vdw + (r/sigma)^6
  [Beutler1994], with the rule of Lorentz and Berthelot;
- no switch and no correction for the dispersion.

OpenMM cuts the energy of a pair at the cutoff, while its forces, as those
of MDIR, are those of the potential shifted to 0 there. The global
parameter `shift` (0 or 1) of the CustomNonbondedForce subtracts
(1 - lambda_vdw) 4 eps ((sigma/r_c)^12 - (sigma/r_c)^6) from each pair of
the ethanol within the cutoff, the shift of `POTENTIAL_SHIFT` of MDIR for
a decoupled pair (D210); it changes no force. `point` and `run` give the
energies of both conventions from the same positions. The direct sum of
particle mesh Ewald is left cut: its shift, f q_i q_j erfc(beta r_c)/r_c
for each pair of the ethanol within the cutoff, is summed by `point` from
the coordinates (S, as the test does) and is below 1e-4 kcal/mol.

`point` prints, for each state, dH/dlambda of both components and the
energy of every state less that of the state, in kcal/mol, as a row of the
free-energy file of MDIR, cut and shifted.

`run` makes the stages of run.py with OpenMM: minimization, 50 ps at
constant volume, 200 ps at 1 atm at state 0, then each state for --ps from
the end of the equilibration; LangevinMiddleIntegrator (1/ps, 2 fs, 300 K),
MonteCarloBarostat every 25 steps, SHAKE and SETTLE, particle mesh Ewald
with beta = 0.32 1/Angstrom on a grid of 32, a cutoff of 9 Angstrom. For
each state it writes s<k>.dhdl, the rows of both conventions of the same
trajectory, in WORK/shift and WORK/cut, each with a control file that only
scripts/free-energy.py reads. --stage runs one stage, so that a GPU can be
released between them.
"""
import argparse
import math
import os
import sys

import openmm as mm
import openmm.app as app
import openmm.unit as unit

KCAL = 4.184
COULOMB = [0.0, 0.25, 0.5, 0.75] + [1.0] * 10
VDW = [0.0] * 5 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
ALPHA = 0.5
CUTOFF = 0.9  # nm
BETA = 3.2    # 1/nm
NB_GROUP, LJ_GROUP = 1, 2


def build(prmtop, electrostatics='PME', grid=32, constraints=True):
    """The system and the indices of the particles of the ethanol."""
    if electrostatics == 'PME':
        method = app.PME
    else:
        method = app.CutoffPeriodic
    system = prmtop.createSystem(
        nonbondedMethod=method, nonbondedCutoff=CUTOFF * unit.nanometer,
        constraints=app.HBonds if constraints else None,
        rigidWater=constraints, removeCMMotion=True)
    nb = next(f for f in system.getForces()
              if isinstance(f, mm.NonbondedForce))
    nb.setUseDispersionCorrection(False)
    nb.setUseSwitchingFunction(False)
    if electrostatics == 'PME':
        nb.setPMEParameters(BETA, grid, grid, grid)
    else:
        nb.setReactionFieldDielectric(78.3)
    nb.setForceGroup(NB_GROUP)
    residue = next(prmtop.topology.residues())
    ligand = [atom.index for atom in residue.atoms()]
    inside = set(ligand)

    lj = mm.CustomNonbondedForce(
        '(1-lambda_vdw)*4*eps*(1/(x*x)-1/x-shift*(s6*s6-s6));'
        'x=alpha*lambda_vdw+(r/sig)^6; s6=(sig/rc)^6;'
        'sig=0.5*(sigma1+sigma2); eps=sqrt(epsilon1*epsilon2)')
    lj.addGlobalParameter('lambda_vdw', 0.0)
    lj.addGlobalParameter('shift', 0.0)
    lj.addGlobalParameter('alpha', ALPHA)
    lj.addGlobalParameter('rc', CUTOFF)
    lj.addPerParticleParameter('sigma')
    lj.addPerParticleParameter('epsilon')
    lj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    lj.setCutoffDistance(CUTOFF)
    lj.setUseLongRangeCorrection(False)
    lj.setUseSwitchingFunction(False)
    lj.addEnergyParameterDerivative('lambda_vdw')
    lj.setForceGroup(LJ_GROUP)

    nb.addGlobalParameter('lambda_coulomb', 0.0)
    parameters = []
    for i in range(system.getNumParticles()):
        charge, sigma, epsilon = nb.getParticleParameters(i)
        parameters.append((charge, sigma, epsilon))
        lj.addParticle([sigma, epsilon])
        if i in inside:
            nb.setParticleParameters(i, charge, sigma, 0.0)
            nb.addParticleParameterOffset('lambda_coulomb', i, -charge, 0.0,
                                          0.0)
    # The pairs within the ethanol at full strength: those of the topology
    # keep their exceptions (excluded or 1-4), the others become exceptions
    # with their charges and Lennard-Jones.
    listed = set()
    for k in range(nb.getNumExceptions()):
        i, j, _, _, _ = nb.getExceptionParameters(k)
        listed.add((min(i, j), max(i, j)))
    for a in ligand:
        for b in ligand:
            if a < b and (a, b) not in listed:
                qa, sa, ea = parameters[a]
                qb, sb, eb = parameters[b]
                nb.addException(a, b, qa * qb, 0.5 * (sa + sb),
                                unit.sqrt(ea * eb))
    for k in range(nb.getNumExceptions()):
        i, j, _, _, _ = nb.getExceptionParameters(k)
        lj.addExclusion(i, j)
    rest = [i for i in range(system.getNumParticles()) if i not in inside]
    lj.addInteractionGroup(ligand, rest)
    system.addForce(lj)
    return system, ligand


def energies(context, states, state):
    """dH/dlambda of the Coulomb and of the Lennard-Jones at `state` and
    the energy of every state less that of `state`, in kcal/mol, with the
    cut and with the shifted Lennard-Jones: (dhdl, du) of each. The energy
    is a quadratic of 1 - lambda_coulomb in the NonbondedForce and a
    function of lambda_vdw in the CustomNonbondedForce, each in a group of
    its own."""
    def group(index, derivative=False):
        result = context.getState(getEnergy=True,
                                  getParameterDerivatives=derivative,
                                  groups={index})
        energy = result.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole) / KCAL
        if derivative:
            return energy, result.getEnergyParameterDerivatives()[
                'lambda_vdw'] / KCAL
        return energy

    coulomb = {}
    for value in sorted({c for c, _ in states}):
        context.setParameter('lambda_coulomb', value)
        coulomb[value] = group(NB_GROUP)
    # The derivative of the quadratic, exact from two points beside the
    # state.
    here = states[state][0]
    h = 0.125
    side = []
    for value in (here - h, here + h):
        context.setParameter('lambda_coulomb', value)
        side.append(group(NB_GROUP))
    dcoulomb = (side[1] - side[0]) / (2.0 * h)
    result = []
    for shift in (0.0, 1.0):
        context.setParameter('shift', shift)
        vdw = {}
        for value in sorted({v for _, v in states}):
            context.setParameter('lambda_vdw', value)
            vdw[value] = group(LJ_GROUP, True)
        dvdw = vdw[states[state][1]][1]
        total = [coulomb[c] + vdw[v][0] for c, v in states]
        result.append(((dcoulomb, dvdw),
                       [energy - total[state] for energy in total]))
    context.setParameter('shift', 0.0)
    context.setParameter('lambda_coulomb', states[state][0])
    context.setParameter('lambda_vdw', states[state][1])
    return result


def direct_shift(system, ligand, positions, box):
    """S = sum of f q_i q_j erfc(beta r_c)/r_c over the pairs of a particle
    of the ethanol and one of the rest within the cutoff, kcal/mol: what
    shifting the direct sum of particle mesh Ewald removes at lambda = 0."""
    nb = next(f for f in system.getForces()
              if isinstance(f, mm.NonbondedForce))
    charges = [nb.getParticleParameters(i)[0].value_in_unit(
        unit.elementary_charge) for i in range(system.getNumParticles())]
    inside = set(ligand)
    total = 0.0
    for i in ligand:
        for j in range(len(charges)):
            if j in inside:
                continue
            d = [positions[i][a] - positions[j][a] for a in range(3)]
            d = [d[a] - box[a] * round(d[a] / box[a]) for a in range(3)]
            if math.sqrt(sum(c * c for c in d)) < CUTOFF:
                total += charges[i] * charges[j]
    return 138.935457644 / KCAL * total * math.erfc(BETA * CUTOFF) / CUTOFF


def point(arguments):
    prmtop = app.AmberPrmtopFile(arguments.prmtop)
    inpcrd = app.AmberInpcrdFile(arguments.inpcrd)
    system, ligand = build(prmtop, arguments.electrostatics, arguments.grid,
                           constraints=False)
    system.setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)
    platform = mm.Platform.getPlatformByName(arguments.platform)
    properties = {}
    if arguments.platform in ('CUDA', 'OpenCL'):
        properties['Precision'] = 'double'
    context = mm.Context(system, mm.VerletIntegrator(0.001), platform,
                         properties)
    context.setPositions(inpcrd.positions)
    if arguments.states:
        states = [tuple(float(v) for v in s.split(',')) for s in
                  arguments.states]
    else:
        states = list(zip(COULOMB, VDW))
    positions = inpcrd.positions.value_in_unit(unit.nanometer)
    box = [inpcrd.boxVectors[a][a].value_in_unit(unit.nanometer)
           for a in range(3)]
    if arguments.electrostatics == 'PME':
        print(f'# S {direct_shift(system, ligand, positions, box):.9f}')
    for k in range(len(states)):
        for name, (dhdl, du) in zip(('cut', 'shift'),
                                    energies(context, states, k)):
            print(name, k, ' '.join(f'{v:.6f}' for v in list(dhdl) + du))


CONTROL = '''# Read by scripts/free-energy.py only: the runs are those of OpenMM.
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


def run(arguments):
    here = os.path.dirname(os.path.abspath(__file__))
    inputs = os.path.join(here, '..', '..', '..', 'test', 'Driver', 'Inputs',
                          'fep')
    work = arguments.work
    for name in ('cut', 'shift'):
        os.makedirs(os.path.join(work, name), exist_ok=True)
    prmtop = app.AmberPrmtopFile(os.path.join(inputs, 'eth_wat.prmtop'))
    inpcrd = app.AmberInpcrdFile(os.path.join(inputs, 'eth_wat.inpcrd'))
    states = list(zip(COULOMB, VDW))
    temperature = 300.0 * unit.kelvin
    platform = mm.Platform.getPlatformByName(arguments.platform)
    properties = {}
    if arguments.platform in ('CUDA', 'OpenCL'):
        properties['Precision'] = arguments.precision

    def simulation(barostat, seed, restraint):
        system, ligand = build(prmtop)
        if restraint:
            # The heavy atoms of the ethanol, as run.py restrains them, in
            # kcal/mol/Angstrom^2 of k (x - x0)^2.
            force = mm.CustomExternalForce(
                'k*periodicdistance(x, y, z, x0, y0, z0)^2')
            force.addGlobalParameter('k', restraint * KCAL * 100.0)
            for name in ('x0', 'y0', 'z0'):
                force.addPerParticleParameter(name)
            atoms = list(prmtop.topology.atoms())
            for i in ligand:
                if atoms[i].element is not None and \
                        atoms[i].element.symbol != 'H':
                    force.addParticle(i, inpcrd.positions[i])
            system.addForce(force)
        if barostat:
            pressure = mm.MonteCarloBarostat(1.0 * unit.atmosphere,
                                             temperature, 25)
            pressure.setRandomNumberSeed(seed + 1)
            system.addForce(pressure)
        integrator = mm.LangevinMiddleIntegrator(
            temperature, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
        integrator.setRandomNumberSeed(seed)
        result = app.Simulation(prmtop.topology, system, integrator,
                                platform, properties)
        return result

    # Minimization and 50 ps at constant volume, then 200 ps at 1 atm; the
    # state at its end is kept in WORK/equilibrated.xml for the states.
    kept = os.path.join(work, 'equilibrated.xml')
    if arguments.stage in ('all', 'equilibration'):
        stage = simulation(False, arguments.seed, 10.0)
        stage.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
        stage.context.setPositions(inpcrd.positions)
        stage.minimizeEnergy(maxIterations=2000)
        stage.context.setVelocitiesToTemperature(temperature, arguments.seed)
        stage.step(25000)
        state = stage.context.getState(getPositions=True, getVelocities=True)
        stage = simulation(True, arguments.seed + 2, 1.0)
        stage.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        stage.context.setPositions(state.getPositions())
        stage.context.setVelocities(state.getVelocities())
        stage.step(100000)
        start = stage.context.getState(getPositions=True, getVelocities=True)
        with open(kept, 'w') as file:
            file.write(mm.XmlSerializer.serialize(start))
        volume = start.getPeriodicBoxVolume().value_in_unit(
            unit.angstrom ** 3)
        print(f'equilibrated: {volume:.1f} A^3', flush=True)
        if arguments.stage == 'equilibration':
            return
    with open(kept) as file:
        start = mm.XmlSerializer.deserialize(file.read())

    steps = int(round(arguments.ps / 0.002))
    header = ('# step time dHdl.coulomb dHdl.vdw ' +
              ' '.join(f'dU.{k}' for k in range(len(states))) + '\n# - ps ' +
              ' '.join(['kcal/mol'] * (2 + len(states))) + '\n')
    for k, (coulomb, vdw) in enumerate(states):
        if arguments.stage not in ('all', str(k)):
            continue
        run = simulation(True, arguments.seed + 10 * (k + 1), 0.0)
        context = run.context
        context.setPeriodicBoxVectors(*start.getPeriodicBoxVectors())
        context.setPositions(start.getPositions())
        context.setVelocities(start.getVelocities())
        context.setParameter('lambda_coulomb', coulomb)
        context.setParameter('lambda_vdw', vdw)
        files = []
        for name in ('cut', 'shift'):
            with open(os.path.join(work, name, f's{k:02d}.toml'), 'w') as f:
                f.write(CONTROL.format(k=k, coulomb=COULOMB, vdw=VDW))
            files.append(open(os.path.join(work, name, f's{k:02d}.dhdl'),
                              'w'))
            files[-1].write(header)
        volumes = []
        for step in range(arguments.energy_interval, steps + 1,
                          arguments.energy_interval):
            run.step(arguments.energy_interval)
            for file, (dhdl, du) in zip(files, energies(context, states, k)):
                file.write(f'{step} {step * 0.002:.6f} ' + ' '.join(
                    f'{v:.6f}' for v in list(dhdl) + du) + '\n')
            volumes.append(context.getState().getPeriodicBoxVolume()
                           .value_in_unit(unit.angstrom ** 3))
        for file in files:
            file.close()
        print(f'state {k}: mean volume {sum(volumes) / len(volumes):.1f} '
              f'A^3', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('point')
    p.add_argument('prmtop')
    p.add_argument('inpcrd')
    p.add_argument('--electrostatics', default='PME', choices=('PME', 'RF'))
    p.add_argument('--grid', type=int, default=32)
    p.add_argument('--platform', default='Reference')
    p.add_argument('--states', nargs='*')
    p.set_defaults(function=point)
    r = commands.add_parser('run')
    r.add_argument('work')
    r.add_argument('--ps', type=float, default=500.0)
    r.add_argument('--seed', type=int, default=1)
    r.add_argument('--platform', default='CUDA')
    r.add_argument('--precision', default='mixed')
    r.add_argument('--energy-interval', type=int, default=250)
    r.add_argument('--stage', default='all',
                   help="'equilibration', the number of a state, or 'all'")
    r.set_defaults(function=run)
    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == '__main__':
    main()
