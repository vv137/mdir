#!/usr/bin/env python3
"""Statistics of a long run at constant temperature and pressure: the
autocorrelation times, the volume and its fluctuation, and the kinetic
energy against the canonical distribution (issue #259).

    scripts/validation/free-energy/long_runs.py [--temperature 300]
        [--degrees-of-freedom N] [--skip-ps 100] [--path dC,dV]
        (--mdir LOG [DHDL] | --gromacs ENERGY.xvg [DHDL.xvg] |
         --table FILE --dt PS)

The series, rows at equal intervals:

- `--mdir`: the log of `mdir run` (its rows INFO: with the potential and
  kinetic energies and the volume) and, optionally, its free-energy file;
- `--gromacs`: the output of `gmx energy` with Potential, Kinetic-En., and
  Volume chosen in this order, and, optionally, the dhdl.xvg of the run;
- `--table`: a text file with the columns step, potential and kinetic
  energy (kcal/mol), volume (Angstrom^3), and optionally dH/dlambda of the
  two components, `--dt` ps apart (rows without the last two are allowed).

It prints the integrated autocorrelation times of the volume, of the
potential energy, and of dH/dlambda (the sum of the autocorrelation with
the window of Sokal, five times the time, in ps); the mean of the volume
with its standard error and its standard deviation; the isothermal
compressibility from the fluctuation, var(V) / (k_B T <V>), in 1/atm; the
kinetic temperature 2 <K> / (N_f k_B); the variance of K over the canonical
N_f (k_B T)^2 / 2; and the p of Kolmogorov and Smirnov for K, samples four
autocorrelation times apart, against the Gamma distribution of N_f / 2 and
k_B T, and for its shape alone (the sample standardized, against a normal
distribution). The degrees of freedom are the program's own count: three
for each particle, less the constraints, less three where the motion of the
center of mass is removed. `--path` gives the weights of dH/dlambda of the
Coulomb and of the Lennard-Jones (1,0 by default).

Needs NumPy and SciPy.
"""
import argparse
import math
import sys

import numpy as np
from scipy import stats

KB = 0.0019872042586408316  # kcal/(mol K)
KCAL = 4.184
ATM = 1.4583972e-5          # kcal/mol in one atm Angstrom^3


def tau(series, dt):
    """The integrated autocorrelation time, in the unit of dt."""
    x = np.asarray(series, float) - np.mean(series)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / np.arange(n, 0, -1)
    acf /= acf[0]
    t = 0.5
    for m in range(1, n):
        t += acf[m]
        if m >= 5 * t:
            break
    return t * dt


def xvg(path):
    return np.array([line.split() for line in open(path)
                     if line.strip() and line[0] not in '#@'], float)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--mdir', nargs='+')
    parser.add_argument('--gromacs', nargs='+')
    parser.add_argument('--table')
    parser.add_argument('--dt', type=float)
    parser.add_argument('--temperature', type=float, default=300.0)
    parser.add_argument('--degrees-of-freedom', type=int, required=True)
    parser.add_argument('--skip-ps', type=float, default=100.0)
    parser.add_argument('--path', default='1,0')
    arguments = parser.parse_args()
    wc, wv = (float(v) for v in arguments.path.split(','))
    dh, ddt = None, None
    if arguments.mdir:
        rows = np.array([line.split()[1:] for line in open(arguments.mdir[0])
                         if line.startswith('INFO:') and 'STEP' not in line],
                        float)
        dt = rows[1, 1] - rows[0, 1]
        u, k, v = rows[:, 3], rows[:, 4], rows[:, 8]
        if len(arguments.mdir) > 1:
            d = np.loadtxt(arguments.mdir[1], comments='#')
            ddt = d[1, 1] - d[0, 1]
            dh = wc * d[:, 2] + wv * d[:, 3]
    elif arguments.gromacs:
        a = xvg(arguments.gromacs[0])
        dt = a[1, 0] - a[0, 0]
        u, k, v = a[:, 1] / KCAL, a[:, 2] / KCAL, a[:, 3] * 1000.0
        if len(arguments.gromacs) > 1:
            import gromacs_ethanol
            rows = gromacs_ethanol.read_xvg(arguments.gromacs[1])
            ddt = rows[1][0] - rows[0][0]
            dh = np.array([wc * r[1] + wv * r[2] for r in rows])
    else:
        rows = [line.split() for line in open(arguments.table)
                if line.strip() and line[0] != '#']
        a = np.array([r[:4] for r in rows], float)
        dt = arguments.dt
        u, k, v = a[:, 1], a[:, 2], a[:, 3]
        steps = [float(r[0]) for r in rows if len(r) > 5]
        if len(steps) > 1:
            dh = np.array([wc * float(r[4]) + wv * float(r[5])
                           for r in rows if len(r) > 5])
            ddt = dt * (steps[1] - steps[0]) / (a[1, 0] - a[0, 0])

    first = int(arguments.skip_ps / dt)
    u, k, v = u[first:], k[first:], v[first:]
    kt = KB * arguments.temperature
    nf = arguments.degrees_of_freedom
    tv, tu, tk = tau(v, dt), tau(u, dt), tau(k, dt)
    independent = len(v) * dt / (2.0 * tv)
    kappa = v.var(ddof=1) / (kt * v.mean()) * ATM
    temperature = 2.0 * k / (nf * KB)
    nk = len(k) * dt / (2.0 * tk)
    sample = k[::max(1, int(math.ceil(4.0 * tk / dt)))]
    canonical = stats.kstest(sample, 'gamma', args=(nf / 2.0, 0.0, kt))
    shape = stats.kstest((sample - sample.mean()) / sample.std(ddof=1),
                         'norm')
    print(f'rows: {len(v)}, {dt:g} ps apart')
    line = (f'autocorrelation times (ps): volume {tv:.2f}, potential '
            f'{tu:.2f}, kinetic {tk:.2f}')
    if dh is not None:
        dh = dh[int(arguments.skip_ps / ddt):]
        line += f', dH/dlambda {tau(dh, ddt):.2f} (mean {dh.mean():.4f})'
    print(line)
    print(f'volume (Angstrom^3): {v.mean():.1f} +- '
          f'{v.std(ddof=1) / math.sqrt(independent):.1f}, sd '
          f'{v.std(ddof=1):.1f}')
    print(f'compressibility from the fluctuation (1e-5 /atm): '
          f'{kappa * 1e5:.2f} +- '
          f'{kappa * 1e5 * math.sqrt(2.0 / independent):.2f}')
    print(f'kinetic temperature (K): {temperature.mean():.3f} +- '
          f'{temperature.std(ddof=1) / math.sqrt(nk):.3f}')
    print(f'variance of K over the canonical: '
          f'{k.var(ddof=1) / (nf * kt ** 2 / 2.0):.3f} +- '
          f'{math.sqrt(2.0 / nk):.3f}')
    print(f'Kolmogorov-Smirnov of K, {len(sample)} samples: p '
          f'{canonical.pvalue:.3f} against the canonical distribution, '
          f'{shape.pvalue:.3f} for the shape')


if __name__ == '__main__':
    sys.exit(main())
