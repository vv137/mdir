#!/usr/bin/env python3
"""The scatter that sets of free-energy runs would have if their runs were
independent of one another: sets made anew from the runs (issue #259).

    scripts/validation/free-energy/permute.py [--trials 200] [--skip 0.1]
        DIR DIR ...

Each DIR is one set of runs, one run for each state. A synthetic set takes
the run of each state from a set drawn at random (a permutation of the
sets within each state): every run is kept, and whatever links the runs of
one set is removed. The script prints the standard deviation over the
sets of the free energy by thermodynamic integration and by MBAR for the
actual sets, its mean and standard deviation over the trials for the
synthetic ones, and the fraction of the trials at or above and at or below
the actual value, in kcal/mol. One trial is as many MBAR solutions as
there are sets. Needs NumPy.
"""
import argparse
import sys

import numpy as np

import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('sets', nargs='+')
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--skip', type=float, default=0.1)
    arguments = parser.parse_args()
    rng = np.random.default_rng(7)
    sets = [compare.read(d, arguments.skip) for d in arguments.sets]
    inefficiencies = [compare.inefficiencies(s) for s in sets]
    n, states = len(sets), len(sets[0])

    def scatter(source):
        """source[k][s]: the set whose run of state k enters set s."""
        ti, mbar = [], []
        for s in range(n):
            runs = [sets[source[k][s]][k] for k in range(states)]
            g = [inefficiencies[source[k][s]][k] for k in range(states)]
            result = compare.analyze(runs, g, [(0, states - 1)])[0]
            ti.append(result[0])
            mbar.append(result[2])
        return np.std(ti, ddof=1), np.std(mbar, ddof=1)

    actual = scatter([list(range(n))] * states)
    print(f'# {n} sets; sd over the sets, kcal/mol: TI {actual[0]:.4f}, '
          f'MBAR {actual[1]:.4f}', flush=True)
    trials = []
    for trial in range(arguments.trials):
        trials.append(scatter([list(rng.permutation(n))
                               for _ in range(states)]))
        if (trial + 1) % 10 == 0 or trial + 1 == arguments.trials:
            a = np.array(trials)
            print(f'# after {trial + 1} trials: TI {a[:, 0].mean():.4f} '
                  f'(sd {a[:, 0].std(ddof=1):.4f}), MBAR '
                  f'{a[:, 1].mean():.4f} (sd {a[:, 1].std(ddof=1):.4f}); at '
                  f'or above the actual: TI {np.mean(a[:, 0] >= actual[0]):.3f}'
                  f', MBAR {np.mean(a[:, 1] >= actual[1]):.3f}; at or below: '
                  f'TI {np.mean(a[:, 0] <= actual[0]):.3f}, MBAR '
                  f'{np.mean(a[:, 1] <= actual[1]):.3f}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
