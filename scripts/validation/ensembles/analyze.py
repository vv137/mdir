#!/usr/bin/env python3
"""Tests the ensembles that MDIR samples, from the logs that run.py
writes in WORK:

  - the kinetic energy at constant volume against the canonical
    distribution, a gamma distribution of N_f / 2 and k_B T: its mean
    N_f k_B T / 2, its variance N_f (k_B T)^2 / 2, and the whole
    distribution by a one-sample Kolmogorov-Smirnov test [Merz2018];
  - two temperatures: the potential energies of runs at T1 and T2 have
    ln P2(U) / P1(U) = (beta1 - beta2) U + c [Shirts2013];
  - two pressures: the volumes of runs at P1 and P2 and one temperature
    have ln P2(V) / P1(V) = -beta (P2 - P1) V + c [Shirts2013];
  - the density and the compressibility at 1 atm, against GROMACS with its
    own OPC (gmx-npt.xvg), if run.py ran it.

The slopes are estimated by maximum likelihood (a logistic regression of
which run a sample came from on the quantity), from samples spaced by
their statistical inefficiency, with the standard error of the estimate.

    scripts/validation/ensembles/analyze.py WORK

Needs numpy.
"""
import math
import os
import sys

import numpy as np

KB = 0.0083144626181532 / 4.184          # kcal/(mol K)
# 1 atm in kcal/(mol Å^3): 101325 J/m^3, 1e-30 m^3/Å^3, N_A, 4184 J/kcal.
ATM = 101325.0 * 1e-30 * 6.02214076e23 / 4184.0
WATERS = 1039
MASS = WATERS * (16.00000 + 2 * 1.00800)   # g/mol, the masses of the prmtop
FREEDOM = 3 * 3 * WATERS - 3 * WATERS - 3  # rigid waters, the center of mass


def read_mdir(path, skip=0.1):
    """The columns of the rows of an MDIR log, past `skip` of the run."""
    rows = []
    header = None
    for line in open(path):
        if line.startswith("INFO:") and "STEP" in line:
            header = line.split()[1:]
        elif line.startswith("INFO:") and header:
            fields = line.split()[1:]
            if len(fields) == len(header):
                rows.append([float(v) for v in fields])
    data = np.array(rows)
    start = int(len(data) * skip)
    return {name: data[start:, k] for k, name in enumerate(header)}


def read_xvg(path, skip=0.1):
    names, rows = [], []
    for line in open(path):
        if line.startswith("@ s") and "legend" in line:
            names.append(line.split('"')[1])
        elif not line.startswith(("#", "@")):
            rows.append([float(v) for v in line.split()])
    data = np.array(rows)
    start = int(len(data) * skip)
    return {name: data[start:, k + 1] for k, name in enumerate(names)}


def inefficiency(x):
    """The statistical inefficiency g = 1 + 2 sum of the autocorrelation,
    summed until it first falls below zero."""
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    var = np.dot(x, x) / n
    g = 1.0
    for t in range(1, n // 2):
        c = np.dot(x[:-t], x[t:]) / (n - t) / var
        if c <= 0:
            break
        g += 2.0 * c * (1.0 - t / n)
    return max(g, 1.0)


def subsample(x):
    g = inefficiency(x)
    step = int(math.ceil(g))
    return np.asarray(x)[::step], g


def mean_error(x):
    g = inefficiency(x)
    return np.mean(x), math.sqrt(np.var(x, ddof=1) * g / len(x))


def logistic(x1, x2):
    """The maximum-likelihood slope b of ln P2(x) / P1(x) = a + b x, and its
    standard error, from independent samples of the two runs."""
    x = np.concatenate([x1, x2])
    y = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])
    mu, sd = x.mean(), x.std()
    z = (x - mu) / sd
    w = np.zeros(2)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(w[0] + w[1] * z)))
        grad = np.array([np.sum(y - p), np.sum((y - p) * z)])
        h = p * (1 - p)
        hess = -np.array([[np.sum(h), np.sum(h * z)],
                          [np.sum(h * z), np.sum(h * z * z)]])
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-12:
            break
    cov = np.linalg.inv(-hess)
    return w[1] / sd, math.sqrt(cov[1, 1]) / sd


def gamma_cdf(x, shape, scale):
    """The regularized lower incomplete gamma function P(shape, x/scale),
    by its series below shape + 1 and its continued fraction above
    (Press et al., Numerical Recipes, Sec. 6.2)."""
    z = x / scale
    if z <= 0.0:
        return 0.0
    log_front = shape * math.log(z) - z - math.lgamma(shape)
    if z < shape + 1.0:
        term = total = 1.0 / shape
        a = shape
        for _ in range(100000):
            a += 1.0
            term *= z / a
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(log_front)
    b = z + 1.0 - shape
    c, d = 1.0 / 1e-300, 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - shape)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - math.exp(log_front) * h


def kolmogorov_smirnov(samples, cdf):
    """The statistic D of a one-sample Kolmogorov-Smirnov test and its p
    value, from the asymptotic Kolmogorov distribution."""
    x = np.sort(samples)
    n = len(x)
    f = np.array([cdf(v) for v in x])
    d = max(np.max(np.arange(1, n + 1) / n - f), np.max(f - np.arange(n) / n))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
                  for k in range(1, 101))
    return d, min(max(p, 0.0), 1.0)


def report(name, estimate, error, theory):
    print(f"{name}: {estimate:.6g} +- {error:.2g}, expected {theory:.6g}, "
          f"{(estimate - theory) / error:+.2f} standard errors")


def main():
    work = sys.argv[1]
    t1, t2 = 300.0, 306.0
    a = read_mdir(os.path.join(work, "nvt-300.log"))
    b = read_mdir(os.path.join(work, "nvt-306.log"))
    for run, t in ((a, t1), (b, t2)):
        kinetic = run["KINETIC_ENE"]
        mean, err = mean_error(kinetic)
        expected = 0.5 * FREEDOM * KB * t
        print(f"NVT {t:.0f} K, {len(kinetic)} rows:")
        report("  mean kinetic energy (kcal/mol)", mean, err, expected)
        # The variance, with its error from blocks of samples.
        k_sub, g = subsample(kinetic)
        var = np.var(k_sub, ddof=1)
        var_err = var * math.sqrt(2.0 / (len(k_sub) - 1))
        report("  variance of the kinetic energy", var, var_err,
               0.5 * FREEDOM * (KB * t) ** 2)
        # The whole distribution against Gamma(N_f / 2, k_B T), from the
        # samples spaced by their statistical inefficiency [Merz2018]; the
        # parameters are those of the bath, not fitted.
        d, p = kolmogorov_smirnov(
            k_sub, lambda v: gamma_cdf(v, 0.5 * FREEDOM, KB * t))
        print(f"  Kolmogorov-Smirnov against Gamma(N_f/2, k_B T), "
              f"{len(k_sub)} samples: D = {d:.4f}, p = {p:.3f}")
    u1, _ = subsample(a["POTENTIAL_ENE"])
    u2, _ = subsample(b["POTENTIAL_ENE"])
    slope, err = logistic(u1, u2)
    print(f"Two temperatures, {len(u1)} and {len(u2)} independent samples:")
    report("  slope of ln P2(U)/P1(U) (mol/kcal)", slope, err,
           1.0 / (KB * t1) - 1.0 / (KB * t2))

    c = read_mdir(os.path.join(work, "npt-1.log"))
    d = read_mdir(os.path.join(work, "npt-300.log"))
    v1, _ = subsample(c["VOLUME"])
    v2, _ = subsample(d["VOLUME"])
    slope, err = logistic(v1, v2)
    print(f"Two pressures, {len(v1)} and {len(v2)} independent samples:")
    report("  slope of ln P2(V)/P1(V) (1/Å^3)", slope, err,
           -(300.0 - 1.0) * ATM / (KB * t1))

    # The same test with semi-isotropic coupling, and with the height
    # held, if run.py --semi-only ran them (D119).
    for tag, what in (("semi", "semi-isotropic coupling"),
                      ("held", "semi-isotropic coupling, the height held")):
        paths = [os.path.join(work, f"npt-{p}-{tag}.log") for p in (1, 300)]
        if not all(os.path.exists(path) for path in paths):
            continue
        w1, _ = subsample(read_mdir(paths[0])["VOLUME"])
        w2, _ = subsample(read_mdir(paths[1])["VOLUME"])
        slope, err = logistic(w1, w2)
        print(f"Two pressures with {what}, {len(w1)} and {len(w2)} "
              f"independent samples:")
        report("  slope of ln P2(V)/P1(V) (1/Å^3)", slope, err,
               -(300.0 - 1.0) * ATM / (KB * t1))

    def density(v):
        return MASS / (6.02214076e23 * v * 1e-24)

    for name, run in (("MDIR", c),):
        v = run["VOLUME"]
        mean, err = mean_error(v)
        rho, rho_err = density(mean), density(mean) * err / mean
        v_sub, _ = subsample(v)
        kappa = np.var(v_sub, ddof=1) / (KB * t1 * np.mean(v_sub)) * ATM
        kappa_err = kappa * math.sqrt(2.0 / (len(v_sub) - 1))
        print(f"{name} at 300 K and 1 atm: density {rho:.5f} +- {rho_err:.5f} "
              f"g/cm^3, compressibility {kappa * 1e5 / 1.01325:.2f} +- "
              f"{kappa_err * 1e5 / 1.01325:.2f} 1e-5/bar, "
              f"temperature {np.mean(run['TEMPERATURE']):.2f} K")
    xvg = os.path.join(work, "gmx-npt.xvg")
    if os.path.exists(xvg):
        g = read_xvg(xvg)
        v = g["Volume"] * 1000.0   # nm^3 to Å^3
        mean, err = mean_error(v)
        rho, rho_err = density(mean), density(mean) * err / mean
        v_sub, _ = subsample(v)
        kappa = np.var(v_sub, ddof=1) / (KB * t1 * np.mean(v_sub)) * ATM
        kappa_err = kappa * math.sqrt(2.0 / (len(v_sub) - 1))
        print(f"GROMACS at 300 K and 1 atm: density {rho:.5f} +- {rho_err:.5f} "
              f"g/cm^3, compressibility {kappa * 1e5 / 1.01325:.2f} +- "
              f"{kappa_err * 1e5 / 1.01325:.2f} 1e-5/bar, "
              f"temperature {np.mean(g['Temperature']):.2f} K")


if __name__ == "__main__":
    main()
