"""Write a case for the GPU prototype (esp_bench.cu).
gen_case.py NAME gauss ORDER BETA K1 K2 K3 OUT
gen_case.py NAME esp P C C1 K1 K2 K3 OUT
Layout (little endian): i32 N K1 K2 K3 P D kind ncheb ; f64 L[3] rc off xscale ;
f64 x[3N] q[N] ; f64 wcoef[P*D] dcoef[P*D] (weight of point j as a polynomial
in t = 2s - 1, ascending, and d/du of it) ; f64 B1[K1] B2[K2] B3[K3/2+1] ;
f64 cheb[ncheb] chebd[ncheb] (psi(x) and x psi'(x) times lambda/(2 C0) in T_k(2x^2-1))."""
import sys
import numpy as np
from numpy.polynomial import chebyshev as Ch
from numpy.polynomial import polynomial as Po
import pswf
from accuracy import load, bmod, F

name, kind = sys.argv[1], sys.argv[2]
x, q, L = load(name)
rc = 0.8
N = len(x)
if kind == 'gauss':
    n = int(sys.argv[3]); beta = float(sys.argv[4]); K = [int(v) for v in sys.argv[5:8]]; out = sys.argv[8]
    P = n
    off = n - 1
    from accuracy import bspline
    ts = np.cos(np.pi * (np.arange(64) + 0.5) / 64)
    s = 0.5 * (ts + 1)
    M, dM = bspline(n, s)  # column t: M_n(s + t); point j = first + j has t = n - 1 - j
    wf = lambda j, t: M[:, n - 1 - j]
    deg = n - 1
    W = []; Dd = []
    for j in range(P):
        c = Po.polyfit(ts, M[:, n - 1 - j], deg)
        W.append(c)
        Dd.append(Po.polyder(c) * 2.0)  # d/du = d/ds = 2 d/dt
    Bs = [bmod(n, K[a]) * np.exp(-np.pi ** 2 * (np.fft.fftfreq(K[a], 1.0 / K[a]) / L[a]) ** 2 / beta ** 2) for a in range(3)]
    cheb = np.zeros(1); chebd = np.zeros(1); xscale = np.pi ** 2 / beta ** 2  # the kernel takes pi^2/beta^2
    kindi = 0
else:
    P = int(sys.argv[3]); c = float(sys.argv[4]); c1 = float(sys.argv[5]); K = [int(v) for v in sys.argv[6:9]]; out = sys.argv[9]
    off = P / 2.0 - 1
    win = pswf.Window(c1, P)
    split = pswf.Splitting(c)
    ts = np.cos(np.pi * (np.arange(200) + 0.5) / 200)
    s = 0.5 * (ts + 1)
    W = []; Dd = []
    deg = None
    for dg in range(4, 20):
        ok = True
        for j in range(P):
            y = win.w(s + off - j)
            cf = Po.polyfit(ts, y, dg)
            if np.max(np.abs(Po.polyval(ts, cf) - y)) > 3e-8 * win.p.psi0:
                ok = False
                break
        if ok:
            deg = dg
            break
    for j in range(P):
        y = win.w(s + off - j)
        cf = Po.polyfit(ts, y, deg)
        W.append(cf)
        Dd.append(Po.polyder(cf) * 2.0)
    Bs = []
    for a in range(3):
        ka = np.fft.fftfreq(K[a], 1.0 / K[a])
        Bs.append(1.0 / win.hat(2 * np.pi * ka / K[a]) ** 2)
    # chi_hat(k) = lambda psi(k/c) / (2 C0), k = 2 pi rc |m|, zero beyond c
    xs = np.cos(np.pi * (np.arange(400) + 0.5) / 400)  # nodes in y = 2x^2 - 1
    xx = np.sqrt((xs + 1) / 2)
    p = split.p
    f0 = p.lam * p(xx) / (2 * p.C0)
    f1 = p.lam * xx * p.d(xx) / (2 * p.C0)
    nch = 1
    for nch in range(4, 60):
        a0 = Ch.chebfit(xs, f0, nch - 1)
        if np.max(np.abs(Ch.chebval(xs, a0) - f0)) < 1e-9:
            break
    cheb = Ch.chebfit(xs, f0, nch - 1)
    chebd = Ch.chebfit(xs, f1, nch - 1)
    xscale = 2 * np.pi * rc / c  # x = xscale |m|
    kindi = 1
    print('poly degree', deg, 'cheb terms', nch)
V = np.prod(L)
Bs[0] = Bs[0] * F / (np.pi * V)
D = len(W[0])
with open(out, 'wb') as f:
    np.array([N, K[0], K[1], K[2], P, D, kindi, len(cheb)], dtype=np.int32).tofile(f)
    np.array([L[0], L[1], L[2], rc, off, xscale], dtype=np.float64).tofile(f)
    x.astype(np.float64).tofile(f)
    q.astype(np.float64).tofile(f)
    np.array(W, dtype=np.float64).tofile(f)
    Dd = [np.concatenate([d_, np.zeros(D - len(d_))]) for d_ in Dd]
    np.array(Dd, dtype=np.float64).tofile(f)
    Bs[0].astype(np.float64).tofile(f)
    Bs[1].astype(np.float64).tofile(f)
    Bs[2][:K[2] // 2 + 1].astype(np.float64).tofile(f)
    np.asarray(cheb, dtype=np.float64).tofile(f)
    np.asarray(chebd, dtype=np.float64).tofile(f)
print('wrote', out, 'N', N, 'K', K, 'P', P, 'D', D)
