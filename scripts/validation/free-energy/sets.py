#!/usr/bin/env python3
"""Independent sets of the runs of run.py or of openmm_ethanol.py: the free
energy of each set, and the mean over the sets with the standard error
from their scatter.

    scripts/validation/free-energy/sets.py [--skip 0.1] NAME=DIR,DIR,...
        [NAME=DIR,DIR,...]

Each DIR is one set (the 14 states). For each group it prints ΔG of each
set by thermodynamic integration and by MBAR, with the uncertainty that
scripts/free-energy.py assigns to the set, and the legs of the Coulomb
and of the Lennard-Jones; then the mean over the sets, their standard
deviation, the standard error of the mean (the standard deviation over the
square root of the number of sets), and the root mean square of the
uncertainties of the sets, to compare with the standard deviation. With
two groups it prints the difference of their means, second - first, with
the standard error from the two scatters.
"""
import argparse
import math
import sys

import numpy as np

import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('groups', nargs='+')
    parser.add_argument('--skip', type=float, default=0.1)
    arguments = parser.parse_args()

    names = ('TI', 'MBAR', 'TI Coulomb', 'MBAR Coulomb', 'TI LJ', 'MBAR LJ')
    summary = {}
    for group in arguments.groups:
        name, _, directories = group.partition('=')
        values, errors = [], []
        print(f'# {name}: ΔG in kcal/mol, with the uncertainty of each set')
        print('# set' + ''.join(f' {n:>13} {"+-":>6}' for n in names))
        for directory in directories.split(','):
            runs = compare.read(directory, arguments.skip)
            states = len(runs)
            lambdas = np.array(runs[0]['lambdas'])
            # The last state up to which the first component moves.
            split = max(k for k in range(1, states)
                        if lambdas[k][0] != lambdas[k - 1][0])
            legs = [(0, states - 1), (0, split), (split, states - 1)]
            result = compare.analyze(runs, compare.inefficiencies(runs), legs)
            row, error = [], []
            for leg in result:
                row += [leg[0], leg[2]]
                error += [leg[1], leg[3]]
            values.append(row)
            errors.append(error)
            print(f'{directory.rstrip("/").split("/")[-1][:5]:>5}' + ''.join(
                f' {v:13.4f} {e:6.4f}' for v, e in zip(row, error)))
        values, errors = np.array(values), np.array(errors)
        n = len(values)
        mean = values.mean(axis=0)
        deviation = values.std(axis=0, ddof=1) if n > 1 else \
            np.full(len(names), math.nan)
        summary[name] = (mean, deviation / math.sqrt(n), n)
        for label, row in (
                ('mean', mean), ('std', deviation),
                ('SE', deviation / math.sqrt(n)),
                ('rms +-', np.sqrt((errors ** 2).mean(axis=0)))):
            print(f'# {label:>7}' + ''.join(f' {v:13.4f} {"":6}'
                                            for v in row))
    if len(summary) == 2:
        (a, (ma, sa, _)), (b, (mb, sb, _)) = summary.items()
        print(f'# {b} - {a}, with the standard error from the scatter of '
              'the sets')
        for label, d, e in zip(names, mb - ma, np.hypot(sa, sb)):
            print(f'# {label:>14} {d:8.4f} +- {e:.4f}')


if __name__ == '__main__':
    sys.exit(main())
