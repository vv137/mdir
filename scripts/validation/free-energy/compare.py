#!/usr/bin/env python3
"""The free energies of the runs of run.py, with the legs of the Coulomb
and of the Lennard-Jones, and the differences of two sets of rows of the
same trajectories.

    scripts/validation/free-energy/compare.py WORK [OTHER] [--paired]
        [--skip 0.1] [--mdir MDIR]

For WORK alone it prints ΔG by thermodynamic integration and by MBAR as
scripts/free-energy.py does (its functions are used), from state 0 to each
state, and over the legs: the states where the Coulomb moves and those
where the Lennard-Jones does, each with its uncertainty (that of MBAR from
the covariance of the two ends).

With OTHER, a second set of runs of the same control files, it also
prints, for each state, how many rows of the two logs are the same text
before the first that differs, the differences of the rows of the
free-energy files, and the difference of ΔG, OTHER − WORK. With --mdir,
`mdir checkpoint` compares the final checkpoints of each state bit for
bit.

With --paired, the free-energy files of WORK hold two rows for each step,
two evaluations of one state (a build that writes the row twice, as the
one that measured the effect of issue #218: with the cell of before the
scaling of the barostat, then with the cell after it); the first rows are
one set and the second rows the other.

The samples of the second set are spaced as those of the first, so that a
statistical inefficiency that rounds to another integer does not change
which rows enter. The differences of the energies are those to the states
beside the run's, and those of dH/dλ of the components that move
there, which the estimates rest on: at a state far from the run's the
energy of a sample can be of any size, as can the derivative in the charges
of a particle whose Lennard-Jones is off.
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


def read(work, skip, part=None):
    """The runs of `work` without the first `skip` of each; with `part`,
    0 or 1, the first or the second row of each step."""
    runs = []
    for path in sorted(glob.glob(os.path.join(work, 's*.toml'))):
        run = free_energy.read_run(path, 0.0)
        for key in ('dhdl', 'du'):
            rows = run[key]
            if part is not None:
                if len(rows) % 2:
                    sys.exit(f'{path}: an odd number of rows with --paired')
                rows = rows[part::2]
            run[key] = rows[int(skip * len(rows)):]
        runs.append(run)
    runs.sort(key=lambda run: run['state'])
    return runs


def inefficiencies(runs):
    """The statistical inefficiency of each run, of the series that
    scripts/free-energy.py takes."""
    lambdas = np.array(runs[0]['lambdas'])
    states = len(lambdas)
    result = []
    for run in runs:
        k = run['state']
        step = lambdas[min(k + 1, states - 1)] - lambdas[max(k - 1, 0)]
        series = run['dhdl'] @ step if np.any(step) else run['du'][:, k]
        result.append(free_energy.inefficiency(series))
    return result


def analyze(runs, g, legs):
    """ΔG over each leg (a, b) by TI and by MBAR, with the uncertainties."""
    kt = free_energy.BOLTZMANN * runs[0]['temperature']
    lambdas = np.array(runs[0]['lambdas'])
    mean = np.array([run['dhdl'].mean(axis=0) for run in runs])
    error = np.array([run['dhdl'].std(axis=0, ddof=1) *
                      math.sqrt(gk / len(run['dhdl']))
                      for run, gk in zip(runs, g)])
    kept = [run['du'][::max(1, int(math.ceil(gk)))]
            for run, gk in zip(runs, g)]
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
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--skip', type=float, default=0.1)
    parser.add_argument('--mdir')
    arguments = parser.parse_args()
    if arguments.paired and arguments.other:
        sys.exit('--paired takes one set of runs')

    runs = read(arguments.work, arguments.skip,
                0 if arguments.paired else None)
    states = len(runs)
    lambdas = np.array(runs[0]['lambdas'])
    components = runs[0]['components']
    # From state 0 to each state, then the legs: the states over which one
    # component moves.
    legs = [(0, k) for k in range(1, states)]
    names = [f'0 -> {k}' for k in range(1, states)]
    for c, component in enumerate(components):
        moving = [k for k in range(1, states)
                  if lambdas[k][c] != lambdas[k - 1][c]]
        if moving and (moving[0] - 1, moving[-1]) != (0, states - 1):
            legs.append((moving[0] - 1, moving[-1]))
            names.append(f'{component}, {moving[0] - 1} -> {moving[-1]}')
    g = inefficiencies(runs)
    first = analyze(runs, g, legs)

    if not arguments.other and not arguments.paired:
        print('# ΔG in kcal/mol')
        print(f'# {"states":>16} {"TI":>9} {"+-":>7} {"MBAR":>9} {"+-":>7}')
        for name, row in zip(names, first):
            print(f'{name:>18} {row[0]:9.4f} {row[1]:7.4f} {row[2]:9.4f} '
                  f'{row[3]:7.4f}')
        return

    if arguments.paired:
        others = read(arguments.work, arguments.skip, 1)
        whole = (read(arguments.work, 0.0, 0), read(arguments.work, 0.0, 1))
    else:
        others = read(arguments.other, arguments.skip)
        whole = (read(arguments.work, 0.0), read(arguments.other, 0.0))
    print('# the rows of the two sets, all of each file, second - first, in '
          'kcal/mol; dU to the states beside the run\'s, dHdl of the components that move there')
    print('# state  rows  differ  equal log rows  max |d dHdl|  rms d dHdl  '
          'max |d dU|  rms d dU  mean d dU -  mean d dU +'
          + ('  checkpoints' if arguments.mdir else ''))
    for a, b in zip(*whole):
        k = a['state']
        if a['du'].shape != b['du'].shape:
            sys.exit(f'state {k}: the files have different numbers of rows')
        differ = int(np.sum(np.any(b['du'] != a['du'], axis=1) |
                            np.any(b['dhdl'] != a['dhdl'], axis=1)))
        beside = [j for j in (k - 1, k + 1) if 0 <= j < states]
        du = b['du'][:, beside] - a['du'][:, beside]
        path = lambdas[min(k + 1, states - 1)] != lambdas[max(k - 1, 0)]
        dhdl = b['dhdl'][:, path] - a['dhdl'][:, path]
        equal = '-'
        if arguments.other:
            la = log_rows(os.path.join(arguments.work, f's{k:02d}.log'))
            lb = log_rows(os.path.join(arguments.other, f's{k:02d}.log'))
            same = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y),
                        min(len(la), len(lb)))
            # The first row is the header.
            equal = f'{same - 1} of {len(la) - 1}'
        means = ['        -', '        -']
        for j, column in zip(beside, du.T):
            means[j > k] = f'{column.mean():9.2e}'
        checkpoints = ''
        if arguments.mdir:
            result = subprocess.run(
                [arguments.mdir, 'checkpoint',
                 os.path.join(arguments.work, f's{k:02d}.h5'),
                 os.path.join(arguments.other, f's{k:02d}.h5')],
                capture_output=True, text=True)
            checkpoints = '  ' + (result.stdout + result.stderr).strip()
        print(f'{k:7d} {len(du):5d} {differ:7d} {equal:>15} '
              f'{np.abs(dhdl).max():13.6f} '
              f'{math.sqrt(np.mean(dhdl ** 2)):11.6f} '
              f'{np.abs(du).max():11.6f} {math.sqrt(np.mean(du ** 2)):9.6f} '
              f'{means[0]:>12} {means[1]:>12}{checkpoints}')

    second = analyze(others, g, legs)
    print('# ΔG in kcal/mol: the first set, the second, second - first')
    print(f'# {"states":>16} {"TI":>9} {"+-":>7} {"TI 2":>9} '
          f'{"d TI":>10} {"MBAR":>9} {"+-":>7} {"MBAR 2":>9} '
          f'{"d MBAR":>10}')
    for name, x, y in zip(names, first, second):
        print(f'{name:>18} {x[0]:9.4f} {x[1]:7.4f} {y[0]:9.4f} '
              f'{y[0] - x[0]:10.2e} {x[2]:9.4f} {x[3]:7.4f} {y[2]:9.4f} '
              f'{y[2] - x[2]:10.2e}')


if __name__ == '__main__':
    main()
