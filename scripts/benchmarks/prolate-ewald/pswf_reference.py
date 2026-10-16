"""A reference of psi_0^c, independent of the Legendre method of MDIR: the
eigenfunction of the integral operator itself, int_{-1}^{1} e^{icxt} psi(t)
dt = lambda psi(x), discretized by Gauss-Legendre (Nystrom) in mpmath at 40
digits, its even part cos(cxt), and psi elsewhere by the Nystrom
interpolation psi(x) = (1/lambda) sum_j w_j cos(c x x_j) psi(x_j).
Normalized to int psi^2 = 1 and psi(0) > 0.  Prints, for each c, lambda_0,
psi at a few points, C0 = int_0^1 psi, and chi_hat(k) = lambda psi(k/c) /
(2 C0) at k = c/2, as the lit test of the host's PSWF reads them.
pswf_reference.py C..."""
import sys
import mpmath as mp

mp.mp.dps = 40
N = 96



def nodes():
    # Gauss-Legendre nodes and weights by Newton on P_N
    xs, ws = [], []
    for i in range(1, N + 1):
        x = mp.cos(mp.pi * (i - mp.mpf(1) / 4) / (N + mp.mpf(1) / 2))
        for _ in range(100):
            p0, p1 = mp.mpf(1), x
            for k in range(2, N + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = N * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < mp.mpf(10) ** (-38):
                break
        xs.append(x)
        ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws


xs, ws = nodes()
for c in [mp.mpf(a) for a in sys.argv[1:]]:
    A = mp.matrix(N, N)
    for i in range(N):
        for j in range(N):
            A[i, j] = ws[j] * mp.cos(c * xs[i] * xs[j])
    E, V = mp.eig(A)
    k = max(range(N), key=lambda i: mp.re(E[i]))
    lam = mp.re(E[k])
    v = [mp.re(V[i, k]) for i in range(N)]
    norm = mp.sqrt(mp.fsum(ws[i] * v[i] ** 2 for i in range(N)))
    v = [a / norm for a in v]
    psi = lambda x: mp.fsum(ws[j] * mp.cos(c * x * xs[j]) * v[j] for j in range(N)) / lam
    if psi(0) < 0:
        v = [-a for a in v]
    C0 = mp.quad(psi, [0, 1])
    print(f'c {mp.nstr(c, 8)} lambda {mp.nstr(lam, 17)} C0 {mp.nstr(C0, 17)}')
    for x in ['0', '0.25', '0.5', '0.75', '1']:
        print(f'  psi({x}) = {mp.nstr(psi(mp.mpf(x)), 17)}')
    print(f'  chihat(c/2) = {mp.nstr(lam * psi(mp.mpf(1) / 2) / (2 * C0), 17)}')
