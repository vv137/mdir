"""The tables of r² that the prolate kernel's direct sum takes
(D[prolate-pair-table], issue #380), rebuilt as md-exec-expand-radial builds
them (D94), with their error against the analytic functions.

    pair_table.py MODULE.mlir CUTOFF_NM

MODULE is `mdir emit` of a program with `[pme] kernel = "PROLATE"` in
mixed precision: its md.chebyshev holds P(t) = (1 - Phi(x))/(1 - x),
t = 2x - 1, x = r/r_c. The two functions of s = r² that the kernel
tabulates are the energy's g = (1 - x) P(t) / r and the force's
h = -(dg/dr)/r (times f q_i q_j; the force on a particle is h d).
For each, as the pass does: the fewest bits b from 4 to 12 whose cubics
(Newton form through four Chebyshev nodes of each interval, in powers of
u = s - start) are within 3e-8 in f64 of the function, relative to its
largest value on each interval; then the error of the cubics evaluated in
f32 by fused multiply-adds at 17 points of each interval (the test of the
pass, at most 3e-7), and the largest error relative to the function's
value at the cutoff's tenth (r = r_c/10) for scale.
"""
import re
import sys

import numpy as np


def chebyshev(c, y):
    """Clenshaw's recurrence."""
    b1 = np.zeros_like(y)
    b2 = np.zeros_like(y)
    for ck in c[:0:-1]:
        b1, b2 = ck + 2 * y * b1 - b2, b1
    return c[0] + y * b1 - b2


def derivative(c):
    n = len(c)
    d = np.zeros(n + 1)
    for k in range(n - 1, 0, -1):
        d[k - 1] = d[k + 1] + 2 * k * c[k]
    d[0] *= 0.5
    return d[:n - 1]


def fit(f, low, high, bits):
    shift = 23 - bits
    base = np.float32(low).view(np.uint32) >> shift
    last = np.float32(high).view(np.uint32) >> shift
    keys = np.arange(base, last + 1, dtype=np.uint32)
    start = (keys << shift).view(np.float32).astype(np.float64)
    end = ((keys + 1) << shift).view(np.float32).astype(np.float64)
    width = end - start
    t = np.cos(np.pi * (2 * np.arange(4) + 1) / 8)
    u = 0.5 * width[:, None] * (1 + t[None, :])
    g = f(start[:, None] + u)
    d = g.copy()
    for j in range(1, 4):
        for i in range(3, j - 1, -1):
            d[:, i] = (d[:, i] - d[:, i - 1]) / (u[:, i] - u[:, i - j])
    # Horner on the Newton form, to powers of u (poly[:, k] of u^k).
    p = np.zeros((len(keys), 4))
    p[:, 0] = d[:, 3]
    for i in range(2, -1, -1):
        q = np.zeros_like(p)
        q[:, 1:] += p[:, :3]
        q -= p * u[:, i:i + 1]
        q[:, 0] += d[:, i]
        p = q
    x = width[:, None] * np.arange(17)[None, :] / 16
    exact = f(start[:, None] + x)
    largest = np.abs(exact).max(1)
    fit64 = ((p[:, 3:4] * x + p[:, 2:3]) * x + p[:, 1:2]) * x + p[:, 0:1]
    pf = p.astype(np.float32)
    xf = x.astype(np.float32)
    v = pf[:, 3:4]
    for k in (2, 1, 0):
        v = (v.astype(np.float64) * xf + pf[:, k:k + 1]).astype(np.float32)  # fma
    err64 = (np.abs(fit64 - exact).max(1) / largest).max()
    err32 = (np.abs(v.astype(np.float64) - exact).max(1) / largest).max()
    return len(keys), err64, err32, np.abs(v.astype(np.float64) - exact).max()


def main():
    text = open(sys.argv[1]).read()
    rc = float(sys.argv[2])
    m = re.search(r'md\.chebyshev %\S+ \[([^\]]*)\]', text)
    c = np.array([float(v) for v in m.group(1).split(',')])
    dc = derivative(c)
    x = lambda s: np.sqrt(s) / rc
    t = lambda s: 2 * x(s) - 1
    g = lambda s: (1 - x(s)) * chebyshev(c, t(s)) / np.sqrt(s)

    def h(s):
        r, p, dp = np.sqrt(s), chebyshev(c, t(s)), 2 * chebyshev(dc, t(s)) / rc
        dgdr = (-p / rc + (1 - x(s)) * dp) / r - (1 - x(s)) * p / s
        return -dgdr / r
    high = rc * rc
    low = high * 2.0 ** -10
    for name, f in (('energy', g), ('force', h)):
        for bits in range(4, 13):
            n, e64, e32, absolute = fit(f, low, high, bits)
            if e64 <= 3e-8:
                break
        scale = abs(f(np.array([(rc / 10) ** 2]))[0])
        print(f'{name}: {len(c)} terms, {bits} bits, {n} intervals, {16 * n / 1024:.0f} KiB, '
              f'error in f64 {e64:.1e}, in f32 {e32:.1e} of the largest value of an interval, '
              f'largest absolute {absolute:.1e} = {absolute / scale:.1e} of its value at r_c/10')


if __name__ == '__main__':
    main()
