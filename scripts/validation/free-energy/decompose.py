#!/usr/bin/env python3
"""Where the scatter of independent sets of free-energy runs comes from:
per state, per interval, and between the runs of one set (issue #259).

    scripts/validation/free-energy/decompose.py [--skip 0.1]
        [--permutations 20000] NAME=DIR,DIR,... [NAME=DIR,DIR,...]

Each DIR is one set (the runs of run.py, openmm_ethanol.py, or
gromacs_ethanol.py). For each group it prints, in kcal/mol:

- per state, the standard deviation over the sets of the contribution of
  the state to the thermodynamic integration (its trapezoid weight times
  the mean of dH/dlambda over its run), beside the uncertainty that the
  statistical inefficiency within a run gives (the rms over the sets);
- the sum over the states of those variances, the variance of the total
  over the sets, and their ratio: 1 for runs that are independent of one
  another, above 1 when the runs of one set deviate together. The
  fraction of permutations of the sets within each state whose ratio is at
  or above the actual one is the p of that ratio;
- per interval between neighboring states, the standard deviation over the
  sets of the difference of the free energies by MBAR, beside the
  uncertainty that MBAR assigns;
- the correlation over the sets of the two legs (the states over which the
  first component moves, and the rest), by both estimators;
- the same totals with 30% and 50% of each run left out, and the
  correlation over the sets of the totals of the two halves of the runs.

The statistical inefficiency and MBAR are those of scripts/free-energy.py.
Needs NumPy.
"""
import argparse
import math
import sys

import numpy as np

import compare

fe = compare.free_energy


def weights(lambdas):
    """The trapezoid weights of every state and component."""
    w = np.zeros_like(lambdas)
    for k in range(1, len(lambdas)):
        d = lambdas[k] - lambdas[k - 1]
        w[k - 1] += 0.5 * d
        w[k] += 0.5 * d
    return w


def contributions(runs, part=slice(None)):
    """Per state, the contribution to the integral and its uncertainty
    within the run."""
    w = weights(np.array(runs[0]['lambdas']))
    values, errors = [], []
    for run in runs:
        x = (run['dhdl'][part] * w[run['state']]).sum(axis=1)
        g = fe.inefficiency(x)
        values.append(x.mean())
        errors.append(x.std(ddof=1) * math.sqrt(g / len(x)))
    return np.array(values), np.array(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('groups', nargs='+')
    parser.add_argument('--skip', type=float, default=0.1)
    parser.add_argument('--permutations', type=int, default=20000)
    arguments = parser.parse_args()
    rng = np.random.default_rng(1)

    for group in arguments.groups:
        name, _, directories = group.partition('=')
        directories = directories.split(',')
        sets = [compare.read(d, arguments.skip) for d in directories]
        n, states = len(sets), len(sets[0])
        lambdas = np.array(sets[0][0]['lambdas'])
        split = max(k for k in range(1, states)
                    if lambdas[k][0] != lambdas[k - 1][0])
        legs = ([(k, k + 1) for k in range(states - 1)] +
                [(0, states - 1), (0, split), (split, states - 1)])
        c, e, m, me = [], [], [], []
        for runs in sets:
            values, errors = contributions(runs)
            c.append(values)
            e.append(errors)
            result = compare.analyze(runs, compare.inefficiencies(runs), legs)
            m.append([r[2] for r in result])
            me.append([r[3] for r in result])
        c, e, m, me = map(np.array, (c, e, m, me))

        print(f'# {name}: {n} sets, kcal/mol')
        print('# state  TI contribution: sd over sets  within a run')
        over = c.var(axis=0, ddof=1)
        within = (e ** 2).mean(axis=0)
        for k in range(states):
            print(f'{k:7d} {math.sqrt(over[k]):28.4f} '
                  f'{math.sqrt(within[k]):13.4f}')
        total = c.sum(axis=1)
        ratio = total.var(ddof=1) / over.sum()
        count = 0
        for _ in range(arguments.permutations):
            permuted = np.column_stack([rng.permutation(c[:, k])
                                        for k in range(states)])
            count += permuted.sum(axis=1).var(ddof=1) >= total.var(ddof=1)
        print(f'# TI: sum of the variances of the states {over.sum():.5f}, '
              f'within the runs {within.sum():.5f} (ratio '
              f'{over.sum() / within.sum():.2f}); variance of the total '
              f'{total.var(ddof=1):.5f} (sd {total.std(ddof=1):.4f}); '
              f'total / sum {ratio:.2f}, p {count / arguments.permutations:.3f}')
        print('# interval  MBAR: sd over sets  estimate of MBAR')
        for k in range(states - 1):
            print(f'{k:4d}->{k + 1:2d} {m[:, k].std(ddof=1):18.4f} '
                  f'{math.sqrt((me[:, k] ** 2).mean()):17.4f}')
        last = states - 1
        print(f'# MBAR: total sd {m[:, last].std(ddof=1):.4f}, estimate '
              f'{math.sqrt((me[:, last] ** 2).mean()):.4f}; sum of the '
              f'variances of the intervals {m[:, :last].var(axis=0, ddof=1).sum():.5f}, '
              f'variance of the total {m[:, last].var(ddof=1):.5f}')
        ti_first = c[:, :split + 1].sum(axis=1)   # the shared state is in both
        ti_second = total - ti_first
        print(f'# correlation of the legs over the sets: TI '
              f'{np.corrcoef(ti_first, ti_second)[0, 1]:+.2f}, MBAR '
              f'{np.corrcoef(m[:, last + 1], m[:, last + 2])[0, 1]:+.2f}')
        for skip in (0.3, 0.5):
            ti, mbar = [], []
            for d in directories:
                runs = compare.read(d, skip)
                r = compare.analyze(runs, compare.inefficiencies(runs),
                                    [(0, last)])[0]
                ti.append(r[0])
                mbar.append(r[2])
            print(f'# {skip:.0%} of each run left out: TI {np.mean(ti):.3f} '
                  f'sd {np.std(ti, ddof=1):.3f}, MBAR {np.mean(mbar):.3f} '
                  f'sd {np.std(mbar, ddof=1):.3f}')
        rows = min(len(run['dhdl']) for runs in sets for run in runs)
        half = rows // 2
        first = np.array([contributions(runs, slice(0, half))[0].sum()
                          for runs in sets])
        second = np.array([contributions(runs, slice(half, rows))[0].sum()
                           for runs in sets])
        print(f'# TI of the halves of the runs: sd {first.std(ddof=1):.3f} '
              f'and {second.std(ddof=1):.3f}, correlation over the sets '
              f'{np.corrcoef(first, second)[0, 1]:+.2f}')


if __name__ == '__main__':
    sys.exit(main())
