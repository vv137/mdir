"""The prolate spheroidal wave function of order zero, on the host, in f64.

psi_0^c on [-1, 1] as an expansion in normalized Legendre polynomials whose
coefficients are the eigenvector of the smallest eigenvalue of the symmetric
tridiagonal matrix of the prolate differential operator in that basis
(Xiao, Rokhlin, Yarvin 2001; Osipov, Rokhlin, Xiao 2013).  Written for the
measurement of issue #334; an independent implementation, not a copy.
"""
import numpy as np
from numpy.polynomial import legendre as L


class PSWF:
    def __init__(self, c, n=None):
        self.c = float(c)
        if n is None:
            n = int(2 * c + 40)
        ks = np.arange(0, 2 * n, 2, dtype=float)  # even degrees only
        d = ks * (ks + 1) + c * c * (2 * ks * (ks + 1) - 1) / ((2 * ks + 3) * (2 * ks - 1))
        e = c * c * (ks[:-1] + 2) * (ks[:-1] + 1) / ((2 * ks[:-1] + 3) * np.sqrt((2 * ks[:-1] + 1) * (2 * ks[:-1] + 5)))
        A = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        w, v = np.linalg.eigh(A)
        beta = v[:, 0]
        # normalized Legendre -> plain Legendre coefficients
        coef = np.zeros(2 * n)
        coef[0::2] = beta * np.sqrt(ks + 0.5)
        if L.legval(0.0, coef) < 0:
            coef = -coef
        self.coef = coef                  # psi = sum coef_k P_k, with int psi^2 = 1
        self.dcoef = L.legder(coef)
        self.icoef = L.legint(coef, lbnd=0.0)  # int_0^x psi
        self.psi0 = L.legval(0.0, coef)
        # eigenvalue of int_{-1}^{1} e^{icxt} psi(t) dt = lambda psi(x), at x = 0
        self.lam = 2 * coef[0] / self.psi0
        self.C0 = L.legval(1.0, self.icoef)  # int_0^1 psi
        self.xg, self.wg = L.leggauss(max(200, int(4 * c) + 100))
        self.psig = L.legval(self.xg, coef)

    def __call__(self, x):
        return L.legval(np.asarray(x, dtype=float), self.coef)

    def d(self, x):
        return L.legval(np.asarray(x, dtype=float), self.dcoef)

    def integral(self, x):
        """int_0^x psi, for |x| <= 1."""
        return L.legval(np.asarray(x, dtype=float), self.icoef)

    def ft(self, k):
        """int_{-1}^{1} psi(t) cos(k t) dt, by Gauss-Legendre quadrature."""
        k = np.asarray(k, dtype=float)
        return np.cos(np.multiply.outer(k, self.xg)) @ (self.wg * self.psig)

    def ft_d(self, k):
        """d/dk of ft."""
        k = np.asarray(k, dtype=float)
        return -np.sin(np.multiply.outer(k, self.xg)) @ (self.wg * self.psig * self.xg)

    def moment2(self):
        """<x^2> = int x^2 psi / int psi over [-1, 1]."""
        return np.sum(self.wg * self.psig * self.xg ** 2) / np.sum(self.wg * self.psig)


def c_of_eps(eps):
    """Bandwidth c such that psi_0^c(1) = eps with int psi^2 = 1 (the paper's
    Table 2: 9.5392 at 1e-3), by bisection."""
    lo, hi = 0.5, 80.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p = PSWF(mid)
        if p(1.0) > eps:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class Splitting:
    """chi(x) = psi_0^c(x)/C0 on [-1, 1]; Phi(y) = int_0^y chi; chi_hat(0) = 1."""

    def __init__(self, c):
        self.p = PSWF(c)
        self.c = c

    def Phi(self, y):
        y = np.minimum(np.abs(np.asarray(y, dtype=float)), 1.0)
        return self.p.integral(y) / self.p.C0

    def chi(self, y):
        y = np.asarray(y, dtype=float)
        return np.where(np.abs(y) < 1, self.p(np.minimum(np.abs(y), 1.0)) / self.p.C0, 0.0)

    def chihat(self, k):
        """Fourier transform of chi as an even function on R, normalized to 1 at 0."""
        return self.p.ft(k) / (2 * self.p.C0)

    def chihat_d(self, k):
        return self.p.ft_d(k) / (2 * self.p.C0)

    def self_coeff(self):
        """lim_{r->0} Phi(r/rc)/r = chi(0)/rc."""
        return self.p.psi0 / self.p.C0

    def x2(self):
        return self.p.moment2()


class Window:
    """phi_1(u) = psi_0^{c1}(2u/P), u in grid units, |u| <= P/2."""

    def __init__(self, c1, P):
        self.p = PSWF(c1)
        self.P = P
        self.c1 = c1

    def w(self, u):
        u = np.asarray(u, dtype=float)
        a = 2 * u / self.P
        return np.where(np.abs(a) <= 1, self.p(np.clip(a, -1, 1)), 0.0)

    def dw(self, u):
        u = np.asarray(u, dtype=float)
        a = 2 * u / self.P
        return np.where(np.abs(a) <= 1, self.p.d(np.clip(a, -1, 1)) * 2 / self.P, 0.0)

    def hat(self, kappa):
        """int phi_1(u) e^{-i kappa u} du = (P/2) int psi(t) cos(kappa P t / 2) dt."""
        return 0.5 * self.P * self.p.ft(0.5 * self.P * np.asarray(kappa, dtype=float))


if __name__ == "__main__":
    for eps in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
        c = c_of_eps(eps)
        s = Splitting(c)
        print(f"eps {eps:g}: c {c:.5f}  lambda0 {s.p.lam:.6e}  C0 {s.p.C0:.6e}  chihat(0) {s.chihat(0.0):.15f}  chihat(c) {s.chihat(c):.3e}  <x2> {s.x2():.6e}")
