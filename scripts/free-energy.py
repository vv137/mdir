#!/usr/bin/env python3
"""The free energy between the states of [free_energy] (D161), from the
files that `[output] free_energy` writes, one run per state.

    python3 scripts/free-energy.py run0.toml run1.toml ... [--skip 0.1]

Each control file gives its state, the components of λ, and the
temperature; its free-energy file holds, at every energy, dH/dλ of each
component and U(λ_k) − U(λ of the run) for every state k, in kcal/mol. The
script prints ΔG from the first state to each other one by thermodynamic
integration [Kirkwood1935], the trapezoidal rule over the states in order,
and by MBAR [ShirtsChodera2008], with uncertainties from samples spaced by
their statistical inefficiency [Chodera2007]. These uncertainties are
those within the runs given; scripts/validation/free-energy/sets.py takes
the scatter of independent sets of runs. The keys are those of
docs/references.md. Needs NumPy, and Python 3.11 (tomllib) or tomli.
"""

import argparse
import math
import os
import sys

import numpy as np

try:
    import tomllib
except ImportError:  # Python before 3.11
    import tomli as tomllib

BOLTZMANN = 0.0019872042586408316  # kcal/(mol K), 0.0083144626181532 / 4.184


def read_run(path, skip):
    with open(path, 'rb') as file:
        control = tomllib.load(file)
    energy = control['free_energy']
    output = control.get('output', {})
    if 'free_energy' not in output:
        sys.exit(f'{path}: [output] names no free_energy file')
    name = output['free_energy']
    if not os.path.isabs(name):
        name = os.path.join(os.path.dirname(os.path.abspath(path)), name)
    with open(name) as file:
        columns = file.readline().split()[1:]
        file.readline()
        rows = [[float(v) for v in line.split()] for line in file
                if line.strip() and not line.startswith('#')]
    data = np.array(rows)
    data = data[int(skip * len(data)):]
    components = [c[len('dHdl.'):] for c in columns if c.startswith('dHdl.')]
    if 'dU.0' not in columns:
        sys.exit(f'{name}: a run of one state has no dU columns; the free '
                 f'energy needs [free_energy.lambdas] of two or more states')
    first = columns.index('dU.0')
    lambdas = energy['lambdas']
    return {
        'state': energy.get('state', 0),
        'temperature': control['ensemble']['temperature'],
        'components': components,
        'lambdas': [[lambdas[c][k] if c in lambdas else 0.0
                     for c in components]
                    for k in range(len(columns) - first)],
        'dhdl': data[:, [columns.index('dHdl.' + c) for c in components]],
        'du': data[:, first:],
    }


def inefficiency(series):
    """The statistical inefficiency g = 1 + 2 Σ (1 − t/N) C(t) of a series,
    summed until the autocorrelation C first falls to 0 [Chodera2007]."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 3:
        return 1.0
    dx = x - x.mean()
    variance = np.dot(dx, dx) / n
    if variance == 0.0:
        return 1.0
    g = 1.0
    for t in range(1, n - 1):
        c = np.dot(dx[:n - t], dx[t:]) / ((n - t) * variance)
        if c <= 0.0:
            break
        g += 2.0 * c * (1.0 - t / n)
    return max(g, 1.0)


def mbar(u, counts, tolerance=1e-10, iterations=100000):
    """The reduced free energies f_k and their covariance from the reduced
    energies u[k, n] of all samples n at every state k [ShirtsChodera2008],
    by the self-consistent equations, f_0 = 0."""
    k_count = u.shape[0]
    log_n = np.log(counts)
    f = np.zeros(k_count)
    for _ in range(iterations):
        # ln of the denominator of every sample, by log-sum-exp.
        a = f[:, None] - u + log_n[:, None]
        top = a.max(axis=0)
        log_denominator = top + np.log(np.exp(a - top).sum(axis=0))
        b = -u - log_denominator[None, :]
        top = b.max(axis=1)
        new = -(top + np.log(np.exp(b - top[:, None]).sum(axis=1)))
        new -= new[0]
        if np.max(np.abs(new - f)) < tolerance:
            f = new
            break
        f = new
    a = f[:, None] - u + log_n[:, None]
    top = a.max(axis=0)
    log_denominator = top + np.log(np.exp(a - top).sum(axis=0))
    w = np.exp(f[:, None] - u - log_denominator[None, :]).T  # N x K
    # The asymptotic covariance, Θ = V Σ (I − Σ Vᵀ N V Σ)⁺ Σ Vᵀ with
    # W = U Σ Vᵀ (Eq. D8 of Shirts and Chodera, from its SVD).
    _, sigma, vt = np.linalg.svd(w, full_matrices=False)
    v = vt.T
    s = np.diag(sigma)
    inner = np.eye(len(sigma)) - s @ v.T @ np.diag(counts) @ v @ s
    theta = v @ s @ np.linalg.pinv(inner) @ s @ v.T
    # The overlap matrix, O = Wᵀ W diag(N) (Section V of Shirts and
    # Chodera): O[k, l], the probability that a sample of state l would be
    # taken as one of state k; small values between neighbors mean that the
    # states sample too little of each other.
    overlap = w.T @ w @ np.diag(counts)
    return f, theta, overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('controls', nargs='+',
                        help='the control files of the runs, one per state')
    parser.add_argument('--skip', type=float, default=0.1,
                        help='the fraction of each run to leave out at its '
                             'start (default 0.1)')
    args = parser.parse_args()
    runs = sorted((read_run(path, args.skip) for path in args.controls),
                  key=lambda run: run['state'])
    states = runs[0]['du'].shape[1]
    temperature = runs[0]['temperature']
    components = runs[0]['components']
    if sorted(run['state'] for run in runs) != list(range(states)):
        sys.exit(f'expected one run for each of the {states} states, got the '
                 f'states {[run["state"] for run in runs]}')
    if any(run['temperature'] != temperature for run in runs):
        sys.exit('the runs are at different temperatures')
    kt = BOLTZMANN * temperature
    lambdas = np.array(runs[0]['lambdas'])  # states x components

    # Samples spaced by the statistical inefficiency of dH/dλ along the
    # path, or of the energy to the neighbors where no λ of the run moves.
    ti_mean, ti_error, kept = [], [], []
    for run in runs:
        k = run['state']
        step = lambdas[min(k + 1, states - 1)] - lambdas[max(k - 1, 0)]
        series = run['dhdl'] @ step if np.any(step) else run['du'][:, k]
        g = inefficiency(series)
        sample = run['dhdl'][::max(1, int(math.ceil(g)))]
        kept.append(run['du'][::max(1, int(math.ceil(g)))])
        ti_mean.append(run['dhdl'].mean(axis=0))
        ti_error.append(run['dhdl'].std(axis=0, ddof=1) *
                        math.sqrt(g / len(run['dhdl'])))
        print(f'state {k}: {len(run["dhdl"])} samples, statistical '
              f'inefficiency {g:.1f}, <dH/dl> ' +
              ', '.join(f'{c} {m:.4f} +- {e:.4f}' for c, m, e in
                        zip(components, ti_mean[-1], ti_error[-1])) +
              ' kcal/mol', file=sys.stderr)
    ti_mean, ti_error = np.array(ti_mean), np.array(ti_error)

    # Thermodynamic integration: the trapezoidal rule in each component.
    ti = [0.0]
    ti_variance = [0.0]
    weights = np.zeros_like(ti_mean)
    for k in range(1, states):
        delta = lambdas[k] - lambdas[k - 1]
        weights[k - 1] += 0.5 * delta
        weights[k] += 0.5 * delta
        # A component that does not move adds nothing, even where its
        # derivative is not finite (a decoupled particle on a charge).
        moving = weights != 0.0
        ti.append(float(np.sum(weights[moving] * ti_mean[moving])))
        ti_variance.append(float(np.sum((weights[moving] *
                                         ti_error[moving]) ** 2)))

    # MBAR over every sample of every state.
    u = np.concatenate(kept, axis=0).T / kt  # states x samples
    counts = np.array([len(sample) for sample in kept], dtype=float)
    f, theta, overlap = mbar(u, counts)

    print(f'# T = {temperature} K, kT = {kt:.6f} kcal/mol; ΔG from state 0, '
          'in kcal/mol')
    print('# +-: the uncertainty within the runs given, from their '
          'statistical inefficiency; the standard deviation of n '
          'independent sets is itself known to 1/sqrt(2(n-1)) of its value')
    print('# state ' + ' '.join(f'{c:>8}' for c in components) +
          '    TI      +-     MBAR     +-  overlap')
    for k in range(states):
        variance = theta[0, 0] + theta[k, k] - 2.0 * theta[0, k]
        print(f'{k:7d} ' + ' '.join(f'{v:8.4f}' for v in lambdas[k]) +
              f' {ti[k]:8.4f} {math.sqrt(ti_variance[k]):6.4f}'
              f' {f[k] * kt:8.4f} {math.sqrt(max(variance, 0.0)) * kt:6.4f}'
              + (f' {overlap[k - 1, k]:7.3f}' if k else '        '))


if __name__ == '__main__':
    main()
