"""Relative force error Delta (paper eq. 16) of Gaussian SPME and of ESP on a
system of the Amber suite, against a reference Ewald sum of 9+ digits.

Coulomb of all pairs (no exclusions): Delta is a property of the
approximation of the lattice sum, and the excluded pairs add the same
analytic correction to every method.  Units: nm, e, kJ/mol.
Written for issue #334.
"""
import os, sys, json, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import erfc
from numpy.polynomial import legendre as Lg
import pswf

F = 138.935457644
D = os.path.join(os.environ.get('ESP_WORK', '.'), '')  # the work directory
SUITE = os.path.join(os.environ.get('MDIR_BENCH_DIR', '.'), 'Amber24_Benchmark_Suite', 'PME', '')
DIRS = {'jac': 'JAC_production_NVE', 'cel': 'Cellulose_production_NVE', 'stmv': 'STMV_production_NPT_4fs'}


def load(name):
    cache = f'{D}sys-{name}.npz'
    try:
        d = np.load(cache)
        return d['x'], d['q'], d['L']
    except FileNotFoundError:
        pass
    import parmed
    b = SUITE + DIRS[name] + '/'
    s = parmed.load_file(b + 'prmtop', b + 'inpcrd')
    x = np.array(s.coordinates, dtype=float) / 10.0
    q = np.array([a.charge for a in s.atoms], dtype=float)
    L = np.array(s.box[:3], dtype=float) / 10.0
    x = x - np.floor(x / L) * L
    np.savez(cache, x=x, q=q, L=L)
    return x, q, L


def good_sizes(maxn=1200):
    out = []
    for n in range(4, maxn, 2):
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            out.append(n)
    return out


SIZES = good_sizes()


def next_size(n):
    for s in SIZES:
        if s >= n:
            return s
    raise ValueError(n)


class Pairs:
    """Pairs within rc (minimum image), kept for repeated real-space sums."""

    def __init__(self, x, L, rc, sample=None):
        t = cKDTree(x, boxsize=L)
        if sample is None:
            p = t.query_pairs(rc, output_type='ndarray')
            self.i, self.j = p[:, 0], p[:, 1]
        else:
            lists = t.query_ball_point(x[sample], rc)
            ii = np.concatenate([np.full(len(l), s) for s, l in zip(sample, lists)])
            jj = np.concatenate([np.array(l, dtype=np.int64) for l in lists])
            m = ii != jj
            self.i, self.j = ii[m], jj[m]
        d = x[self.i] - x[self.j]
        d -= np.round(d / L) * L
        self.d = d
        self.r = np.sqrt((d * d).sum(1))
        self.sample = sample
        self.N = len(x)

    def forces(self, q, fr):
        """fr(r) = -u'(r)/r per unit f q q."""
        s = F * q[self.i] * q[self.j] * fr(self.r)
        fv = self.d * s[:, None]
        out = np.zeros((self.N, 3))
        for a in range(3):
            out[:, a] += np.bincount(self.i, fv[:, a], self.N)
            if self.sample is None:
                out[:, a] -= np.bincount(self.j, fv[:, a], self.N)
        return out


def gauss_fr(beta):
    return lambda r: erfc(beta * r) / r ** 3 + 2 * beta / np.sqrt(np.pi) * np.exp(-(beta * r) ** 2) / r ** 2


def esp_fr(split, rc):
    def fr(r):
        y = r / rc
        return (1 - split.Phi(y)) / r ** 3 + split.chi(y) / (rc * r ** 2)
    return fr


def bspline(n, w):
    """Weights M_n(w + n - 1 - j), j = 0..n-1, and their derivatives (B-splines, recursion)."""
    N = len(w)
    A = np.zeros((N, n + 1))  # A[:, t] = M_k(w + t), A[:, n] stays 0
    A[:, 0] = w
    A[:, 1] = 1 - w
    prev = A.copy()
    for k in range(3, n + 1):
        prev = A.copy()
        B = np.zeros_like(A)
        for t in range(k):
            x = w + t
            a = A[:, t]
            b = A[:, t - 1] if t >= 1 else 0.0
            B[:, t] = (x * a + (k - x) * b) / (k - 1)
        A = B
    dA = np.zeros((N, n))
    for t in range(n):
        dA[:, t] = prev[:, t] - (prev[:, t - 1] if t >= 1 else 0.0)
    # columns t: M_n(w + t) and M_n'(w + t); the point of column t is floor(u) - t
    return A[:, :n], dA


def bmod(n, K):
    """|b(k)|^2 of SPME (Essmann 1995)."""
    # M_n at integers 1..n-1
    w = np.zeros(1)
    M, _ = bspline(n, w)  # M_n(0 + j), j = 0..n-1
    Mi = M[0]
    k = np.arange(K)
    s = np.zeros(K, dtype=complex)
    for j in range(n - 1):
        s += Mi[j + 1] * np.exp(2j * np.pi * k * j / K)
    return 1.0 / np.abs(s) ** 2


class Rec:
    """Reciprocal sum on a grid with a tensor-product window; forces by AD."""

    def __init__(self, x, q, L, K, kind, **kw):
        self.x, self.q, self.L, self.K = x, q, L, np.array(K)
        self.kind = kind
        self.kw = kw

    def stencil(self, u):
        kind = self.kind
        if kind == 'gauss':
            n = self.kw['order']
            fl = np.floor(u)
            w = u - fl
            M, dM = bspline(n, w)
            first = fl.astype(np.int64) - (n - 1)
            # point first + t, t = 0..n-1, has column j = n-1-t
            return first, M[:, ::-1], dM[:, ::-1]
        else:
            win = self.kw['window']
            P = win.P
            first = np.ceil(u - P / 2.0).astype(np.int64)
            t = np.arange(P)
            du = u[:, None] - (first[:, None] + t[None, :])
            return first, win.w(du), win.dw(du)

    def run(self, sample=None):
        x, q, L, K = self.x, self.q, self.L, self.K
        N = len(x)
        u = x / L * K
        st = [self.stencil(u[:, a]) for a in range(3)]
        P = st[0][1].shape[1]
        Q = np.zeros(int(np.prod(K)))
        t = np.arange(P)
        chunk = max(1, 4000000 // P ** 3)
        idx_cache = []
        for s0 in range(0, N, chunk):
            sl = slice(s0, min(N, s0 + chunk))
            ix = (st[0][0][sl, None] + t) % K[0]
            iy = (st[1][0][sl, None] + t) % K[1]
            iz = (st[2][0][sl, None] + t) % K[2]
            flat = (ix[:, :, None, None] * K[1] + iy[:, None, :, None]) * K[2] + iz[:, None, None, :]
            wgt = q[sl, None, None, None] * st[0][1][sl, :, None, None] * st[1][1][sl, None, :, None] * st[2][1][sl, None, None, :]
            Q += np.bincount(flat.ravel(), wgt.ravel(), len(Q))
        Q = Q.reshape(K)
        Qh = np.fft.rfftn(Q)
        k0 = np.fft.fftfreq(K[0], 1.0 / K[0])
        k1 = np.fft.fftfreq(K[1], 1.0 / K[1])
        k2 = np.arange(K[2] // 2 + 1)
        m0, m1, m2 = k0 / L[0], k1 / L[1], k2 / L[2]
        msq = m0[:, None, None] ** 2 + m1[None, :, None] ** 2 + m2[None, None, :] ** 2
        V = np.prod(L)
        if self.kind == 'gauss':
            n = self.kw['order']
            beta = self.kw['beta']
            B = [bmod(n, K[a]) for a in range(3)]
            if self.kw.get('optimal'):
                # the aliasing factor of Ballenegger 2012, as MDIR's influence = "OPTIMAL"
                for a in range(3):
                    k = np.arange(K[a])
                    kk = np.where(k > K[a] // 2, k - K[a], k)
                    xx = np.pi * kk / K[a]
                    j = np.arange(-50, 51)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        r = xx[:, None] / (xx[:, None] + np.pi * j[None, :])
                    r[kk == 0] = 0
                    r[kk == 0, 50] = 1
                    lam = (r ** n).sum(1) / (r ** (2 * n)).sum(1)
                    B[a] = B[a] * lam ** 2
            spec = np.exp(-np.pi ** 2 * msq / beta ** 2)
        else:
            win = self.kw['window']
            B = []
            for a in range(3):
                ka = np.fft.fftfreq(K[a], 1.0 / K[a])
                B.append(1.0 / win.hat(2 * np.pi * ka / K[a]) ** 2)
            split, rc = self.kw['split'], self.kw['rc']
            xi = 2 * np.pi * np.sqrt(msq)
            # chi_hat is a function of |m|: tabulate it on the distinct values
            flatxi = xi.ravel()
            spec = np.interp(flatxi, *self._chitab(split, rc, flatxi.max())).reshape(xi.shape)
        Bk = B[0][:, None, None] * B[1][None, :, None] * B[2][None, None, :K[2] // 2 + 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            C = F / (np.pi * V) * spec / msq
        C[0, 0, 0] = 0
        G = Bk * C
        h = np.ones(K[2] // 2 + 1)
        h[0] = 0.5
        if K[2] % 2 == 0:
            h[-1] = 0.5
        E = np.sum(h[None, None, :] * G * np.abs(Qh) ** 2)
        phi = np.fft.irfftn(G * Qh, s=tuple(K)) * np.prod(K)
        phif = phi.ravel()
        idx = np.arange(N) if sample is None else np.asarray(sample)
        Fo = np.zeros((len(idx), 3))
        chunk = max(1, 2000000 // P ** 3)
        for s0 in range(0, len(idx), chunk):
            ii = idx[s0:s0 + chunk]
            ix = (st[0][0][ii, None] + t) % K[0]
            iy = (st[1][0][ii, None] + t) % K[1]
            iz = (st[2][0][ii, None] + t) % K[2]
            flat = (ix[:, :, None, None] * K[1] + iy[:, None, :, None]) * K[2] + iz[:, None, None, :]
            ph = phif[flat]
            W = [st[a][1][ii] for a in range(3)]
            S = [st[a][2][ii] for a in range(3)]
            gx = np.einsum('nabc,na,nb,nc->n', ph, S[0], W[1], W[2])
            gy = np.einsum('nabc,na,nb,nc->n', ph, W[0], S[1], W[2])
            gz = np.einsum('nabc,na,nb,nc->n', ph, W[0], W[1], S[2])
            Fo[s0:s0 + len(ii)] = -q[ii, None] * np.stack([gx, gy, gz], 1) * (K / L)[None, :]
        return E, Fo

    _tabs = {}

    def _chitab(self, split, rc, ximax):
        key = (id(split), rc, round(ximax, 6))
        if key not in Rec._tabs:
            xs = np.linspace(0, ximax * 1.0001 + 1e-9, 20001)
            Rec._tabs[key] = (xs, split.chihat(xs * rc))
        return Rec._tabs[key]


def delta(Fm, Fr):
    return np.sqrt(np.mean(np.sum((Fm - Fr) ** 2, 1))) / np.sqrt(np.mean(np.sum(Fr ** 2, 1)))


def beta_of_tol(tol, rc):
    lo, hi = 0.0, 1.0
    while erfc(hi * rc) > tol:
        hi *= 2
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if erfc(mid * rc) > tol:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
