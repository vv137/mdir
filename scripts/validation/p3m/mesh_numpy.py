#!/usr/bin/env python3
"""An oracle of the three influence functions of PME in NumPy
(D304; docs/pme-m1.md, Section 1.1), written from Section 1 of
docs/pme-m1.md and not from MDIR's code: spreading with B-splines, the FFT,
the product with G, and the gathering of analytic differentiation, with G
of "SPME", "OPTIMAL", and "P3M" (eq. 2.28 of Ballenegger et al. 2012 by
the direct sum over the aliases J in Z^3, |J_a| <= 2, not the form of the
edges that MDIR computes).

    mesh_numpy.py random
        the reciprocal Delta of each G against the Ewald sum over the wave
        vectors, for 300 charges +-1 at random in a cube of 20 A and for
        150 dipoles of +-0.8 e at 1 A (charges uncorrelated between
        molecules), 4 configurations each, beta = 0.4 / A, order 4
    mesh_numpy.py energy PRMTOP INPCRD BETA K LOGS
        the reciprocal energy of a cubic system on K^3 points of order 4
        with each G, against "Coulomb reciprocal" of the log
        LOGS/<G>-o4-KxKxK/run.log of MDIR (sweep.py's cases), in kcal/mol
"""
import math
import sys

import numpy as np

f = 1.0


def bspline(n, x):
    x = np.asarray(x, float)
    if n == 2:
        return np.where((x >= 0) & (x <= 2), 1 - np.abs(x - 1), 0.0)
    return (x * bspline(n - 1, x) + (n - x) * bspline(n - 1, x - 1)) / (n - 1)


def moduli(K, n):
    m = [0.0] + [float(bspline(n, j + 1)) for j in range(n)]
    out = np.empty(K)
    for k in range(K):
        re = sum(m[j + 1] * math.cos(2 * math.pi * k * j / K) for j in range(n - 1))
        im = sum(m[j + 1] * math.sin(2 * math.pi * k * j / K) for j in range(n - 1))
        out[k] = 1.0 / (re * re + im * im)
    return out


def signed(K):
    k = np.arange(K)
    return np.where(k <= K // 2, k, k - K)


def lam2(K, n):
    out = np.ones(K)
    for k, m in enumerate(signed(K)):
        if m == 0:
            continue
        x = math.pi * m / K
        j = np.concatenate([np.arange(-50, 0), np.arange(1, 51)])
        s1 = 1 + np.sum((x / (x + math.pi * j)) ** n)
        s2 = 1 + np.sum((x / (x + math.pi * j)) ** (2 * n))
        out[k] = (s1 / s2) ** 2
    return out


def influences(L, K, n, beta):
    V = L ** 3
    ms = signed(K) / L
    m1, m2, m3 = np.meshgrid(ms, ms, ms, indexing="ij")
    msq = m1 ** 2 + m2 ** 2 + m3 ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(msq > 0, f / (math.pi * V) * np.exp(-math.pi ** 2 * msq / beta ** 2) / msq, 0.0)
    b = moduli(K, n)
    B = b[:, None, None] * b[None, :, None] * b[None, None, :]
    l2 = lam2(K, n)
    Lam = l2[:, None, None] * l2[None, :, None] * l2[None, None, :]
    # eq. 2.28 by the direct sum over J in Z^3, |J_a| <= 2
    num = np.zeros_like(msq)
    s0 = np.zeros_like(msq)
    s2 = np.zeros_like(msq)
    n_idx = signed(K)
    for j1 in range(-2, 3):
        for j2 in range(-2, 3):
            for j3 in range(-2, 3):
                nu = [n_idx + j * K for j in (j1, j2, j3)]
                U2 = 1.0
                comps = []
                for a, v in enumerate(nu):
                    t = v / K
                    u = np.where(v == 0, 1.0, np.sinc(t) ** n)  # np.sinc(t) = sin(pi t)/(pi t)
                    u = np.where((v != 0) & (v % K == 0), 0.0, u)
                    shape = [1, 1, 1]
                    shape[a] = K
                    U2 = U2 * (u ** 2).reshape(shape)
                    comps.append((v / L).reshape(shape))
                mm = comps[0] ** 2 + comps[1] ** 2 + comps[2] ** 2
                num += U2 * np.exp(-math.pi ** 2 * mm / beta ** 2)
                s0 += U2
                s2 += U2 * mm
    with np.errstate(divide="ignore", invalid="ignore"):
        P3M = np.where(msq > 0, f / (math.pi * V) * num / (s0 * s2), 0.0)
    return {"SPME": B * C, "OPTIMAL": B * Lam * C, "P3M": P3M}


def mesh_forces(x, q, L, K, n, G):
    u = x * K / L
    base = np.floor(u).astype(int)
    Q = np.zeros((K, K, K))
    pts, w, dw = [], [], []
    for a in range(3):
        ks = base[:, a:a + 1] - np.arange(n)[None, :]          # floor(u) - j
        arg = u[:, a:a + 1] - ks                                # in [0, n)
        pts.append(ks % K)
        w.append(bspline(n, arg))
        dw.append(bspline(n - 1, arg) - bspline(n - 1, arg - 1))
    for i in range(len(q)):
        Q[np.ix_(pts[0][i], pts[1][i], pts[2][i])] += q[i] * np.einsum(
            "a,b,c->abc", w[0][i], w[1][i], w[2][i])
    phi = K ** 3 * np.fft.ifftn(G * np.fft.fftn(Q)).real
    F = np.zeros_like(x)
    for i in range(len(q)):
        block = phi[np.ix_(pts[0][i], pts[1][i], pts[2][i])]
        F[i, 0] = -q[i] * K / L * np.einsum("abc,a,b,c->", block, dw[0][i], w[1][i], w[2][i])
        F[i, 1] = -q[i] * K / L * np.einsum("abc,a,b,c->", block, w[0][i], dw[1][i], w[2][i])
        F[i, 2] = -q[i] * K / L * np.einsum("abc,a,b,c->", block, w[0][i], w[1][i], dw[2][i])
    return F


def ewald_forces(x, q, L, beta, kmax):
    V = L ** 3
    k = np.arange(-kmax, kmax + 1)
    m = np.stack(np.meshgrid(k, k, k, indexing="ij"), -1).reshape(-1, 3) / L
    msq = np.sum(m ** 2, 1)
    keep = msq > 0
    m, msq = m[keep], msq[keep]
    C = np.exp(-math.pi ** 2 * msq / beta ** 2) / msq
    phase = np.exp(2j * math.pi * x @ m.T)                      # (N, M)
    S = q @ phase                                               # (M,)
    # F_i = -f/(2 pi V) sum_m C 2 Re(conj(S) q_i 2 pi i m e^{2 pi i m x_i})
    t = (np.conj(S)[None, :] * phase * (2j * math.pi)).real
    return -(f / (2 * math.pi * V)) * 2 * q[:, None] * ((t * C[None, :]) @ m)


def system(kind, N, L, rng):
    if kind == "ions":
        x = rng.random((N, 3)) * L
        q = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    else:
        c = rng.random((N // 2, 3)) * L
        d = rng.normal(size=(N // 2, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        x = np.concatenate([c + 0.5 * d, c - 0.5 * d]) % L
        q = np.concatenate([np.full(N // 2, 0.8), np.full(N // 2, -0.8)])
    return x, q


def random_systems():
    L, beta, n = 20.0, 0.4, 4
    for kind in ("ions", "dipoles"):
        for K in (16, 12, 10):
            G = influences(L, K, n, beta)
            sums = {name: 0.0 for name in G}
            ref2 = 0.0
            for seed in range(4):
                rng = np.random.default_rng(seed)
                x, q = system(kind, 300, L, rng)
                Fr = ewald_forces(x, q, L, beta, 16)
                ref2 += np.sum(Fr ** 2)
                for name, g in G.items():
                    sums[name] += np.sum((mesh_forces(x, q, L, K, n, g) - Fr) ** 2)
            row = " ".join(f"{name} {math.sqrt(s / ref2):.4e}" for name, s in sums.items())
            print(f"{kind} K={K} h={L / K:.2f} A order {n} beta {beta}: Delta_rec {row}", flush=True)


def energy(prmtop, inpcrd, beta, K, logs):
    import re
    import parmed
    coulomb = 332.0637133  # kcal/mol A / e^2, Amber's 18.2223^2
    s = parmed.load_file(prmtop, inpcrd)
    q = np.array([a.charge for a in s.atoms])
    L = float(s.box[0])
    x = np.array(s.coordinates) % L
    G = influences(L, K, 4, beta)
    u = x * K / L
    base = np.floor(u).astype(int)
    Q = np.zeros((K, K, K))
    idx, w = [], []
    for a in range(3):
        ks = base[:, a:a + 1] - np.arange(4)[None, :]
        idx.append(ks % K)
        w.append(bspline(4, u[:, a:a + 1] - ks))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                np.add.at(Q, (idx[0][:, i], idx[1][:, j], idx[2][:, k]),
                          q * w[0][:, i] * w[1][:, j] * w[2][:, k])
    Qh = np.fft.fftn(Q)
    for name, g in G.items():
        e = 0.5 * coulomb * np.sum(g * np.abs(Qh) ** 2)
        log = open(f"{logs}/{name}-o4-{K}x{K}x{K}/run.log").read()
        mdir = float(re.search(r"Coulomb reciprocal +(\S+)", log)[1])
        print(f"K {K} {name}: NumPy {e:.6f} MDIR {mdir:.6f} kcal/mol, "
              f"difference {abs(e - mdir) / abs(mdir):.1e}")


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "random":
        random_systems()
    elif len(sys.argv) == 7 and sys.argv[1] == "energy":
        energy(sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
