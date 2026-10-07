"""Numerical checks for docs/octane.md (D[octane]): the tables of the analysis
of [Toutouni2026] (the key is that of docs/references.md), the formulas that
the document proves in their place, by Monte Carlo and in closed form, and
the overflow of a leaf of fixed capacity.

    python3 octane_analysis.py

Needs NumPy; takes about a minute."""
import numpy as np
from math import pi, floor, log2
rng = np.random.default_rng(20261008)

print("== Tables 1 and 2 of the paper (no periodic boundary, r = 1)")
def p_safe(l, D, r=1.0): return max(1 - 2*r*(2**l - 1)/D, 0.0)**3
for D in (520, 1040, 2080, 4160, 9320, 18640):
    lm = floor(log2(D/2 + 1))
    El = 1 + sum(p_safe(k, D) for k in range(1, lm + 1))
    print(f"D={D:6d} l_m={lm:2d} E[l*]={El:7.3f} E[l_m-l*]={lm-El:6.3f}  P_c(l_m)={1-p_safe(lm,D):.5f}")

print("== The level of the smallest containing node, periodic cube (Monte Carlo against the formula)")
def mc_levels(D, r, lmax, n=2_000_000):
    x = rng.random((n, 3))*D
    lam = np.zeros(n, dtype=int)   # deepest level whose cell holds [x-r, x+r] on every axis, -1: crosses the cell
    ok_prev = np.ones(n, bool)
    res = np.full(n, -1)
    for l in range(0, lmax + 1):
        s = D/2**l
        inside = np.all((np.floor((x - r)/s) == np.floor((x + r)/s)), axis=1)
        ok_prev &= inside
        res[ok_prev] = l
    return res
D, r, lmax = 64.0, 1.0, 5      # leaf side 2 = 2r: u = 1
res = mc_levels(D, r, lmax)
for l in range(0, lmax + 1):
    f = max(1 - 2**(l+1)*r/D, 0)**3
    print(f"  P(level >= {l}) MC {np.mean(res >= l):.5f}  formula (1-2^(l+1) r/D)^3 = {f:.5f}")
climb = lmax - res
u = 2*r/(D/2**lmax)
print(f"  mean climb from the leaf MC {climb.mean():.4f}; sum formula {sum(1-max(1-u*2.0**-j,0)**3 for j in range(lmax+1)):.4f}; limit 6u-4u^2+8u^3/7 = {6*u-4*u*u+8*u**3/7:.4f}")
for u in (0.25, 0.5, 1.0):
    print(f"  u={u}: limit of the mean climb {6*u-4*u*u+8*u**3/7:.4f}")

print("== Leaves that a query box meets: E = (1+u)^3")
for u in (0.5, 1.0, 2.0):
    s = 1.0; rr = u*s/2
    x = rng.random((1_000_000, 3))*64
    cells = np.prod(np.floor((x + rr)/s) - np.floor((x - rr)/s) + 1, axis=1)
    print(f"  u={u}: MC {cells.mean():.4f}  (1+u)^3 = {(1+u)**3:.4f}")

print("== The update: displacement uniform in a ball (disk) of radius p, leaf side d = 2p")
def update_mc(dim, n=4_000_000, kmax=12):
    d, p = 2.0, 1.0
    x = rng.random((n, dim))*d*2**kmax
    v = rng.normal(size=(n, dim)); v /= np.linalg.norm(v, axis=1)[:, None]
    v *= p*rng.random(n)[:, None]**(1/dim)
    y = x + v
    L = np.zeros(n, int)
    for k in range(kmax):
        s = d*2**k
        crossed = np.any(np.floor(x/s) != np.floor(y/s), axis=1)
        L[crossed] = k + 1
    return L
for dim in (2, 3):
    L = update_mc(dim)
    pe, p1, p2, EL = np.mean(L > 0), np.mean(L == 1), np.mean(L > 1), L.mean()
    se = np.sqrt(pe*(1-pe)/L.size)
    print(f"  {dim}D MC: P_exit {pe:.5f} (+-{se:.5f})  P(L=1) {p1:.5f}  P(L>1) {p2:.5f}  P(L=1)/P_exit {p1/pe:.4f}  E[L] {EL:.5f}")
m1, m2, m3 = 3/8, 2/(5*pi), 1/(8*pi)      # E|x|, E|xy|, E|xyz| in the unit ball
def P_le(k, s=2.0):
    a = 2.0**-k/s
    return 1 - 3*m1*a + 3*m2*a*a - m3*a**3
print(f"  3D exact: P_exit = 9/16 - 3/(10 pi) + 1/(64 pi) = {9/16-3/(10*pi)+1/(64*pi):.5f}; P(L=1) = {P_le(1)-P_le(0):.5f}; P(L>1) = {1-P_le(1):.5f}; E[L] = {6*m1/2-4*m2/4+(8/7)*m3/8:.5f}")
n1, n2 = 4/(3*pi), 1/(2*pi)                # E|x|, E|xy| in the unit disk
print(f"  2D exact: P_exit = 4/(3 pi) - 1/(8 pi) = 29/(24 pi) = {29/(24*pi):.5f}; E[L] = 4 n1/2 - (4/3) n2/4 = {4*n1/2-(4/3)*n2/4:.5f}")
print(f"  paper: 2D P_exit 0.3846; 3D P_exit 0.4237, P(L=1) 0.1937, P(L>1) 0.2300, E[L] -> 4*0.1937 = {4*0.1937:.4f}; its geometric tail sums to P(L>=1) = 2*0.1937 = {2*0.1937:.4f}")

print("== The update in a real step: E|dx| = dt sqrt(2 k T / (pi m))")
kB = 0.0083144626  # kJ/mol/K
for name, T, m, dt, s in (("argon 120 K, 1 fs, leaf 10 A", 120, 39.948, 0.001, 1.0), ("water oxygen 300 K, 2 fs, leaf 8 A", 300, 15.999, 0.002, 0.8), ("hydrogen 300 K, 2 fs, leaf 8 A", 300, 1.008, 0.002, 0.8)):
    m1x = dt*np.sqrt(2*kB*T/(pi*m))   # nm
    print(f"  {name}: E|dx| = {m1x*10:.2e} A; P_exit ~ 3 E|dx|/s = {3*m1x/s:.2e}; E[L] ~ 6 E|dx|/s = {6*m1x/s:.2e}")

print("== Overflow of a leaf of capacity ceil(b mu) with Poisson occupancy of mean mu")
from math import exp, lgamma, ceil
def tail(mu, c):  # P(X > c)
    return 1 - sum(exp(-mu + k*np.log(mu) - lgamma(k+1)) for k in range(0, c+1))
for mu in (4, 8, 32, 128):
    for b in (1.1, 1.5):
        c = ceil(b*mu); t = tail(mu, c)
        print(f"  mu={mu:4d} b={b}: capacity {c:4d}  P(overflow of one leaf) = {t:.3e}; leaves for 1e6 atoms {int(1e6/mu):7d}, expected overflowing {1e6/mu*t:.3g}")

print("== The mean occupancy at which the Chernoff bound excludes an overflow among 1e6 particles (probability 0.999)")
from math import log
for b in (1.5, 1.1):
    c = b*log(b) - b + 1
    mu = 1.0
    while (1e6/mu)*exp(-c*mu) > 1e-3:
        mu += 1
    print(f"  b={b}: exponent {c:.5f} mu, mu >= {mu:.0f}")
print("== The share of searches from the root in the systems of the paper (D/r from its Table 3, r = 3 sigma)")
for D in (480, 1033, 2230):
    q = 2*3*3.374/D
    print(f"  D={D} A: D/r = {D/(3*3.374):.1f}, 1-(1-2r/D)^3 = {1-(1-q)**3:.4f}, l_m = {floor(log2(D/(2*3*3.374)+1))}")
