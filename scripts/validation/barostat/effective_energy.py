"""Checks the increment of the effective energy of a step of the barostat
(Bernetti and Bussi 2020, eq. S10; design-m1.md, Section 11.5) on a model
in one variable. From lambda drawn from P(lambda) ~ exp(-E/kT), one step

    lambda' = lambda + (D/kT) f dt + sqrt(2 D dt) R,   f = -E'(lambda),

has the shadow work w = dE + kT ln[T(l -> l') / T(l' -> l)], for which
<exp(-w/kT)> = 1 exactly. The closed form

    w = dE + d lambda (f + f')/2 + (beta dt / (16 tau)) (f'^2 - f^2)

equals it to rounding; the coefficient beta dt / (4 tau kT) of the image of
the paper's eq. (S11) does not, and misses <exp(-w/kT)> = 1. kT is not 1 so
that a coefficient of other units shows.

    python effective_energy.py
"""
import numpy as np
rng = np.random.default_rng(1)
kT, P0, beta, tau = 2.5, 1.3, 0.7, 1.9
# E(lambda) = U(V) + P0 V - kT ln(lambda), V = lambda^2; U a stiff well in V.
U  = lambda V: 4.0 * (V - 3.0) ** 2
dU = lambda V: 8.0 * (V - 3.0)
E  = lambda l: U(l * l) + P0 * l * l - kT * np.log(l)
f  = lambda l: -(dU(l * l) * 2 * l + 2 * P0 * l - kT / l)
D  = kT * beta / (4 * tau)
# Exact samples from P(lambda) on a fine grid.
grid = np.linspace(1e-3, 4.0, 400001)
p = np.exp(-(E(grid) - E(grid).min()) / kT); c = np.cumsum(p); c /= c[-1]
lam = np.interp(rng.random(4_000_000), c, grid)
for dt in (0.02, 0.05, 0.1):
    lp = lam + (D / kT) * f(lam) * dt + np.sqrt(2 * D * dt) * rng.standard_normal(lam.size)
    ok = lp > 0; l, m = lam[ok], lp[ok]
    base = E(m) - E(l) + (m - l) * 0.5 * (f(l) + f(m))
    df2 = f(m) ** 2 - f(l) ** 2
    exact = kT * (((l - m - (D/kT)*f(m) * dt) ** 2 - (m - l - (D/kT)*f(l) * dt) ** 2) / (4 * D * dt)) + E(m) - E(l)
    mine = base + beta * dt / (16 * tau) * df2
    paper = base + beta * dt / (4 * tau * kT) * df2
    for name, w in (("direct", exact), ("beta dt/(16 tau)", mine), ("paper beta dt/(4 tau kT)", paper)):
        print(f"dt {dt}: {name:26s} <exp(-w/kT)> = {np.mean(np.exp(-w / kT)):.5f}  <w> = {np.mean(w):+.2e}")
    print(f"        max |mine - direct| = {np.max(np.abs(mine - exact)):.1e}")
