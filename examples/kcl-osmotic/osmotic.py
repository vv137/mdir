#!/usr/bin/env python3
"""The osmotic pressure of KCl(aq) between walls on the ions, its
experimental value, and the reweighting in the K-Cl sigma' (#140).

    osmotic.py SIGMA_HAT RUN [RUN ...] [--drop PS] [--grid LO:HI:N] [--neff 0.5]

Each RUN is <replica directory>/<name>: <name>.obs, the columns of
`observe` of the production run, and cell.txt, the edges of the replica's
cell in A. SIGMA_HAT is the sigma' (A) that the runs sampled.

The correction of the K-Cl pair is linear in two numbers,
u = x12 c12/r^12 - x6 c6/r^6, with c12 = 4 eps0 sig0^12 and c6 = 4 eps0 sig0^6
of the Lorentz-Berthelot pair. The fitted sigma' moves the pair along
x12 = (sigma'/sig0)^12 - 1, x6 = (sigma'/sig0)^6 - 1 at fixed eps0, and
U(sigma') - U(sigma_hat) = dx12 A12 + dx6 A6 for the recorded columns
A12 = nbfix.d_x12 and A6 = nbfix.d_x6: the reweighting to any sigma' is
exact from two numbers a row, without frames.

The experimental osmotic pressure is that against pure water across a
membrane that passes only water, Pi = -RT ln a_w / V_w
= phi * 2 m M_w R T / V_w, with the osmotic coefficient phi(m) of
Hamer and Wu, J. Phys. Chem. Ref. Data 1, 1047 (1972), Table 28 (25 C),
the molality m of the molarity c from the densities of the CRC Handbook's
table of the concentrative properties of aqueous solutions (20 C, after
Sohnel and Novotny 1985), and V_w the molar volume of pure water at 25 C.
"""
import argparse
import math
import pathlib

import numpy as np

# The Lorentz-Berthelot K-Cl pair of the Joung-Cheatham ions for TIP3P
# (Joung and Cheatham, J. Phys. Chem. B 112, 9020 (2008)): R_min/2 of
# 1.705 and 2.513 A, epsilon 0.1936829 and 0.0355910 kcal/mol.
SIG0 = (1.705 + 2.513) * 2 ** (5 / 6) / 2      # 3.757811 A
EPS0 = math.sqrt(0.1936829 * 0.0355910)        # 0.0830263 kcal/mol
C12, C6 = 4 * EPS0 * SIG0 ** 12, 4 * EPS0 * SIG0 ** 6

KB = 0.0019872043          # kcal/mol/K
BAR = 69476.95             # bar per kcal/mol/A^3
NA_L = 6.02214076e-4       # (mol/L) per A^-3: c = n / (V NA_L)
R = 8.314462618            # J/mol/K
M_W, V_W = 0.0180153, 18.0686e-6   # kg/mol; m^3/mol (0.99705 g/cm^3 at 25 C)

# Hamer and Wu (1972), Table 28: phi of KCl(aq) at 25 C against molality.
HW_M = np.array([0.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])
HW_PHI = np.array([.927, .913, .906, .902, .900, .899, .898, .898, .898, .898, .900, .902, .905, .908, .912, .923, .936])
# CRC Handbook, concentrative properties of aqueous solutions, KCl at 20 C:
# molality (mol/kg) and molarity (mol/L).
CRC_M = np.array([0.0, .067, .135, .274, .415, .559, .706, .856, 1.010, 1.166, 1.327, 1.490, 1.829, 2.184, 2.555, 2.944])
CRC_C = np.array([0.0, .067, .135, .271, .409, .549, .691, .835, .980, 1.127, 1.276, 1.426, 1.733, 2.048, 2.370, 2.701])


def _pchip(x, y, t):
    """Monotone cubic (Fritsch-Carlson) interpolation, NumPy only."""
    x, y, t = np.asarray(x, float), np.asarray(y, float), np.asarray(t, float)
    h, d = np.diff(x), np.diff(y) / np.diff(x)
    m = np.zeros_like(y)
    for k in range(1, len(x) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    m[0], m[-1] = d[0], d[-1]
    k = np.clip(np.searchsorted(x, t) - 1, 0, len(x) - 2)
    s = (t - x[k]) / h[k]
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * y[k] + h10 * h[k] * m[k] + h01 * y[k + 1] + h11 * h[k] * m[k + 1]


def molality(c):
    return _pchip(CRC_C, CRC_M, c)


def phi(m):
    return _pchip(HW_M, HW_PHI, m)


def pi_exp(c, T=300.0):
    """Pi = -RT ln a_w / V_w (bar) at the molarity c (mol/L)."""
    m = molality(c)
    return phi(m) * 2 * m * M_W * R * T / V_W / 1e5


def pi_exp_slope(c, h=1e-3):
    return (pi_exp(c + h) - pi_exp(c - h)) / (2 * h)


def x_of(sigma):
    return (sigma / SIG0) ** 12 - 1.0, (sigma / SIG0) ** 6 - 1.0


def dx_dsigma(sigma):
    return 12 * sigma ** 11 / SIG0 ** 12, 6 * sigma ** 5 / SIG0 ** 6


def log_weights(a12, a6, sigma_hat, sigma, T=300.0):
    """log w of each row for the move sigma_hat -> sigma, unnormalized."""
    (x12h, x6h), (x12, x6) = x_of(sigma_hat), x_of(sigma)
    return -((x12 - x12h) * np.asarray(a12) + (x6 - x6h) * np.asarray(a6)) / (KB * T)


def normalized(lw):
    w = np.exp(np.asarray(lw) - np.max(lw))
    return w / w.sum()


def kish(w):
    """N_eff / N of normalized weights."""
    return 1.0 / (len(w) * np.sum(w ** 2))


def load(run, drop_ps=500.0):
    """The rows of one replica after dropping its first drop_ps."""
    run = pathlib.Path(run)
    edges = np.loadtxt(run.parent / "cell.txt")
    area = edges[0] * edges[1]
    with open(f"{run}.obs") as f:
        names = f.readline().lstrip("#").split()
    a = np.loadtxt(f"{run}.obs", comments="#", ndmin=2)
    col = {n: a[:, i] for i, n in enumerate(names)}
    t = col["time"] - col["time"][0]
    keep = t >= drop_ps - 1e-9
    return dict(
        pi=((col["lower.d_z0"] - col["upper.d_z0"]) / (2 * area) * BAR)[keep],
        c=(col["slab.d_s"] / 2 / (area * 2 * SLAB_HALF_WIDTH) / NA_L)[keep],
        a12=col["nbfix.d_x12"][keep], a6=col["nbfix.d_x6"][keep],
        ns=(t[keep][-1] - t[keep][0]) / 1000.0 if keep.any() else 0.0, area=area)


SLAB_HALF_WIDTH = 10.0     # A: the slab |z - L_z/2| < 10 A of the template's 'slab' term


def se(v):
    v = np.asarray(v, float)
    return v.std(ddof=1) / math.sqrt(len(v))


def at(reps, sigma_hat, sigma):
    """Per-replica reweighted <Pi>, <c>, f, and the pooled N_eff/N."""
    out = dict(pi=[], c=[], f=[])
    for r in reps:
        w = normalized(log_weights(r["a12"], r["a6"], sigma_hat, sigma))
        p, c = float(w @ r["pi"]), float(w @ r["c"])
        out["pi"].append(p); out["c"].append(c); out["f"].append(p - float(pi_exp(c)))
    lw = np.concatenate([log_weights(r["a12"], r["a6"], sigma_hat, sigma) for r in reps])
    out["neff"] = kish(normalized(lw))
    return out


def gradients(reps, sigma_hat):
    """-beta Cov(O, G) per replica: G = dU/dsigma', O_eff = Pi - Pi_exp'(c) c."""
    d12, d6 = dx_dsigma(sigma_hat)
    slope = float(pi_exp_slope(np.mean([r["c"].mean() for r in reps])))
    beta = 1 / (KB * 300.0)
    cov = lambda a, b: float(np.mean((a - a.mean()) * (b - b.mean())))
    g = dict(pi=[], c=[], oeff=[], oeff_x6=[])
    for r in reps:
        G = r["a12"] * d12 + r["a6"] * d6
        o = r["pi"] - slope * r["c"]
        g["pi"].append(-beta * cov(r["pi"], G)); g["c"].append(-beta * cov(r["c"], G))
        g["oeff"].append(-beta * cov(o, G)); g["oeff_x6"].append(-beta * cov(o, r["a6"]))
    return g


def root(d, f):
    """The first sign change of f(d) by linear interpolation, or None."""
    for i in range(len(d) - 1):
        if f[i] == 0 or f[i] * f[i + 1] < 0:
            return d[i] - f[i] * (d[i + 1] - d[i]) / (f[i + 1] - f[i])
    return None


def step_of(reps, sigma_hat, ds, neff_min):
    """The root of the reweighted f in the trust region (N_eff/N >= neff_min)
    if there is one, else the Newton step -f/(df/dsigma') of the covariance
    gradient, which the caller limits to the trust region."""
    f = np.array([np.mean(at(reps, sigma_hat, sigma_hat + d)["f"]) for d in ds])
    r = root(ds, f)
    if r is not None:
        return r, True
    fhat = np.mean([r_["pi"].mean() - float(pi_exp(r_["c"].mean())) for r_ in reps])
    return -fhat / np.mean(gradients(reps, sigma_hat)["oeff"]), False


def inefficiency(xs, dt_ps=0.2):
    """g dt = (1 + 2 sum rho) dt by blocking, at the longest block with 8 blocks a replica."""
    b, best = 1, None
    while all(len(x) // b >= 8 for x in xs):
        ratios = []
        for x in xs:
            m = len(x) // b
            xb = x[: m * b].reshape(m, b).mean(1)
            ratios.append(b * xb.var(ddof=1) / x.var(ddof=1))
        best, b = float(np.mean(ratios)) * dt_ps, b * 2
    return best


def analyze(runs, sigma_hat, drop_ps=500.0, grid=(-0.02, 0.02, 41), neff_min=0.5, out=print):
    reps = [load(r, drop_ps) for r in runs]
    n = len(reps)
    pi = np.array([r["pi"].mean() for r in reps]); c = np.array([r["c"].mean() for r in reps])
    f = pi - pi_exp(c)
    x12, x6 = x_of(sigma_hat)
    out(f"sigma' = {sigma_hat:.5f} A (x12 {x12:+.5f}, x6 {x6:+.5f}); {n} replicas, "
        f"{sum(r['ns'] for r in reps):.1f} ns after dropping {drop_ps:g} ps")
    out(f"  <Pi>  = {pi.mean():8.3f} +- {se(pi):.3f} bar")
    out(f"  <c>   = {c.mean():8.4f} +- {se(c):.4f} M  (about {float(molality(c.mean())):.2f} m)")
    out(f"  Pi_exp(<c>) = {float(pi_exp(c.mean())):.2f} bar")
    out(f"  f = <Pi> - Pi_exp = {f.mean():+.3f} +- {se(f):.3f} bar")
    g = gradients(reps, sigma_hat)
    for k, lab, u in (("pi", "d<Pi>/dsigma'", "bar/A"), ("c", "d<c>/dsigma'", "M/A"), ("oeff", "d<O_eff>/dsigma'", "bar/A"),
                      ("oeff_x6", "d<O_eff>/dx6 at fixed x12", "bar")):
        out(f"  {lab} = {np.mean(g[k]):+9.3f} +- {se(g[k]):.3f} {u}")
    d12, d6 = dx_dsigma(sigma_hat)
    for name, fn in (("Pi", lambda r: r["pi"]), ("c", lambda r: r["c"]), ("G", lambda r: r["a12"] * d12 + r["a6"] * d6)):
        out(f"  correlation time g*dt of {name}: {inefficiency([fn(r) for r in reps]):.0f} ps (blocking)")
    ds = np.linspace(*grid[:2], int(grid[2]))
    rows = []
    for d in ds:
        o = at(reps, sigma_hat, sigma_hat + d)
        rows.append((d, o["neff"], np.mean(o["f"]), se(o["f"])))
    out("  reweighted: d sigma' (A)  N_eff/N    f (bar)")
    for d, ne, fm, fs in rows:
        if abs(round(d / 0.005) * 0.005 - d) < 1e-9:
            out(f"             {d:+.3f}     {ne:.3f}    {fm:+.2f} +- {fs:.2f}")
    trust = np.array([d for d, ne, _, _ in rows if ne >= neff_min])
    lo, hi = float(trust.min()), float(trust.max())
    full, inside = step_of(reps, sigma_hat, trust, neff_min)
    jk = np.array([step_of([q for j, q in enumerate(reps) if j != k], sigma_hat, trust, neff_min)[0] for k in range(n)])
    jse = math.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
    step = min(max(full, lo), hi)
    how = "reweighted root of f" if inside else "Newton step from the covariance gradient (no root in the trust region)"
    out(f"  trust region (N_eff/N >= {neff_min}): d sigma' in [{lo:+.4f}, {hi:+.4f}] A")
    out(f"  {how}: sigma' = {sigma_hat + full:.5f} A (d {full:+.4f}, jackknife SE {jse:.4f} A)")
    out(f"  next sigma' (limited to the trust region): {sigma_hat + step:.5f} A")
    return dict(f=float(f.mean()), f_se=float(se(f)), root=float(sigma_hat + full), root_se=float(jse),
                next=float(sigma_hat + step))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sigma_hat", type=float)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--drop", type=float, default=500.0, help="ps dropped at the start of each run")
    ap.add_argument("--grid", default="-0.02:0.02:41", help="the range of d sigma' (A) and its points")
    ap.add_argument("--neff", type=float, default=0.5)
    a = ap.parse_args()
    lo, hi, n = a.grid.split(":")
    analyze(a.runs, a.sigma_hat, a.drop, (float(lo), float(hi), int(n)), a.neff)


if __name__ == "__main__":
    main()
