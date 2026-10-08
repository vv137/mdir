#!/usr/bin/env python3
"""The free energies of the runs of run.py, with the legs of the Coulomb
and of the Lennard-Jones, and the differences of two sets of runs of the
same trajectories (two builds, --deterministic, the same seeds).

    scripts/validation/free-energy/compare.py WORK [OTHER] [--skip 0.1]
        [--mdir MDIR]

For WORK alone it prints ΔG by thermodynamic integration and by MBAR as
scripts/free-energy.py does (its functions are used), from state 0 to each
state, and over the legs: the states where the Coulomb moves and those
where the Lennard-Jones does, each with its uncertainty (that of MBAR from
the covariance of the two ends).

With OTHER it also prints, for each state, whether the rows of the logs of
the two runs are the same text, the largest difference of a row of the
free-energy files in dH/dλ and in the energy differences, and the
difference of ΔG, OTHER − WORK. The samples of OTHER are spaced as those of
WORK, so that a statistical inefficiency that rounds to another integer
does not change which rows enter. With --mdir, `mdir checkpoint` compares
the final checkpoints of each state bit for bit.
"""
import argparse
import glob
import importlib.util
import math
import os
import subprocess
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    'free_energy', os.path.join(here, '..', '..', 'free-energy.py'))
free_energy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(free_energy)


def read(work, skip):
    runs = [free_energy.read_run(path, skip)
            for path in sorted(glob.glob(os.path.join(work, 's*.toml')))]
    runs.sort(key=lambda run: run['state'])
    return runs


def strides(runs):
    """The spacing of the samples of each run, as scripts/free-energy.py."""
    lambdas = np.array(runs[0]['lambdas'])
    states = len(lambdas)
    result = []
    for run in runs:
        k = run['state']
        step = lambdas[min(k + 1, states - 1)] - lambdas[max(k - 1, 0)]
        series = run['dhdl'] @ step if np.any(step) else run['du'][:, k]
        result.append(free_energy.inefficiency(series))
    return result


def analyze(runs, inefficiencies, legs):
    """ΔG over each leg (a, b) by TI and by MBAR, with the uncertainties."""
    kt = free_energy.BOLTZMANN * runs[0]['temperature']
    lambdas = np.array(runs[0]['lambdas'])
    mean = np.array([run['dhdl'].mean(axis=0) for run in runs])
    error = np.array([run['dhdl'].std(axis=0, ddof=1) *
                      math.sqrt(g / len(run['dhdl']))
                      for run, g in zip(runs, inefficiencies)])
    kept = [run['du'][::max(1, int(math.ceil(g)))]
            for run, g in zip(runs, inefficiencies)]
    u = np.concatenate(kept, axis=0).T / kt
    counts = np.array([len(sample) for sample in kept], dtype=float)
    f, theta, _ = free_energy.mbar(u, counts)
    result = []
    for a, b in legs:
        weights = np.zeros_like(mean)
        for k in range(a + 1, b + 1):
            delta = lambdas[k] - lambdas[k - 1]
            weights[k - 1] += 0.5 * delta
            weights[k] += 0.5 * delta
        moving = weights != 0.0
        ti = float(np.sum(weights[moving] * mean[moving]))
        ti_error = math.sqrt(float(np.sum((weights[moving] *
                                           error[moving]) ** 2)))
        variance = theta[a, a] + theta[b, b] - 2.0 * theta[a, b]
        result.append((ti, ti_error, (f[b] - f[a]) * kt,
                       math.sqrt(max(variance, 0.0)) * kt))
    return result


def log_rows(path):
    with open(path) as file:
        return [line for line in file if line.startswith('INFO:')]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('work')
    parser.add_argument('other', nargs='?')
    parser.add_argument('--skip', type=float, default=0.1)
    parser.add_argument('--mdir')
    arguments = parser.parse_args()

    runs = read(arguments.work, arguments.skip)
    states = len(runs)
    lambdas = np.array(runs[0]['lambdas'])
    components = runs[0]['components']
    # The legs: the runs of states where one component alone moves.
    legs = [(0, k) for k in range(1, states)]
    names = [f'0 -> {k}' for k in range(1, states)]
    for c, component in enumerate(components):
        moving = [k for k in range(1, states)
                  if lambdas[k][c] != lambdas[k - 1][c]]
        if moving and (moving[0] - 1, moving[-1]) != (0, states - 1):
            legs.append((moving[0] - 1, moving[-1]))
            names.append(f'{component}, {moving[0] - 1} -> {moving[-1]}')
    inefficiencies = strides(runs)
    first = analyze(runs, inefficiencies, legs)

    if not arguments.other:
        print('# ΔG in kcal/mol')
        print(f'# {"states":>16} {"TI":>9} {"+-":>7} {"MBAR":>9} {"+-":>7}')
        for name, row in zip(names, first):
            print(f'{name:>18} {row[0]:9.4f} {row[1]:7.4f} {row[2]:9.4f} '
                  f'{row[3]:7.4f}')
        return

    others = read(arguments.other, arguments.skip)
    print('# rows of the two sets, all of each file (kcal/mol)')
    print('# state  rows  log rows  logs equal  rows that differ  '
          'max |d dHdl|  max |d dU|  rms d dU  checkpoints')
    largest = [0.0, 0.0]
    for k in range(states):
        a = free_energy.read_run(
            os.path.join(arguments.work, f's{k:02d}.toml'), 0.0)
        b = free_energy.read_run(
            os.path.join(arguments.other, f's{k:02d}.toml'), 0.0)
        if a['du'].shape != b['du'].shape:
            sys.exit(f'state {k}: the files have different numbers of rows')
        dhdl = np.abs(b['dhdl'] - a['dhdl'])
        du = b['du'] - a['du']
        differ = int(np.sum(np.any(du != 0.0, axis=1) |
                            np.any(dhdl != 0.0, axis=1)))
        la = log_rows(os.path.join(arguments.work, f's{k:02d}.log'))
        lb = log_rows(os.path.join(arguments.other, f's{k:02d}.log'))
        checkpoints = '-'
        if arguments.mdir:
            result = subprocess.run(
                [arguments.mdir, 'checkpoint',
                 os.path.join(arguments.work, f's{k:02d}.h5'),
                 os.path.join(arguments.other, f's{k:02d}.h5')],
                capture_output=True, text=True)
            checkpoints = (result.stdout + result.stderr).strip()
        largest = [max(largest[0], dhdl.max()), max(largest[1],
                                                    np.abs(du).max())]
        print(f'{k:7d} {len(du):5d} {len(la):9d} '
              f'{"yes" if la == lb else "NO":>11} {differ:17d} '
              f'{dhdl.max():13.6f} {np.abs(du).max():11.6f} '
              f'{math.sqrt(np.mean(du ** 2)):9.2e}  {checkpoints}')
    print(f'# largest: dHdl {largest[0]:.6f}, dU {largest[1]:.6f}')

    second = analyze(others, inefficiencies, legs)
    print('# ΔG in kcal/mol: WORK, OTHER, OTHER − WORK')
    print(f'# {"states":>16} {"TI":>9} {"+-":>7} {"TI other":>9} '
          f'{"d TI":>10} {"MBAR":>9} {"+-":>7} {"MBAR other":>10} '
          f'{"d MBAR":>10}')
    for name, x, y in zip(names, first, second):
        print(f'{name:>18} {x[0]:9.4f} {x[1]:7.4f} {y[0]:9.4f} '
              f'{y[0] - x[0]:10.2e} {x[2]:9.4f} {x[3]:7.4f} {y[2]:10.4f} '
              f'{y[2] - x[2]:10.2e}')


if __name__ == '__main__':
    main()
