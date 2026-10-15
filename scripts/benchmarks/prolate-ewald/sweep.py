"""Delta against grid for Gaussian SPME and ESP.  Usage:
sweep.py NAME gauss ORDER   |  sweep.py NAME esp P   |  sweep.py NAME points
Writes JSON lines to sweep-NAME-<method>-<n>.jsonl."""
import sys, json, time
import numpy as np
from accuracy import *

name, method = sys.argv[1], sys.argv[2]
arg = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rc = 0.8
x, q, L = load(name)
N = len(x)
ref = np.load(f'{D}ref-{name}-0.npz')
Fref = ref['F']
pr = Pairs(x, L, rc)
out = open(f'{D}sweep-{name}-{method}-{arg}.jsonl', 'w')


def emit(d):
    out.write(json.dumps(d) + '\n')
    out.flush()
    print(d, flush=True)


def gauss_eval(beta, K, order, optimal=False):
    Fd = pr.forces(q, gauss_fr(beta))
    E, Fr = Rec(x, q, L, K, 'gauss', order=order, beta=beta, optimal=optimal).run()
    return delta(Fd + Fr, Fref), Fd, E


def esp_eval(c, c1, P, K):
    split = pswf.Splitting(c)
    win = pswf.Window(c1, P)
    Fd = pr.forces(q, esp_fr(split, rc))
    E, Fr = Rec(x, q, L, K, 'esp', window=win, split=split, rc=rc).run()
    return delta(Fd + Fr, Fref), E


def grid_of(n):
    # n points along the longest axis; others in proportion to L
    s = n / L.max()
    return [next_size(int(np.ceil(s * L[a] - 1e-9))) for a in range(3)]


if method == 'points':
    # MDIR's default: erfc tolerance 1e-5, h_max 1.2 A, order 4, Essmann
    b = beta_of_tol(1e-5, rc)
    K = [next_size(int(np.ceil(L[a] / 0.12))) for a in range(3)]
    d, _, _ = gauss_eval(b, K, 4)
    emit(dict(what='mdir-default', beta=b, K=K, order=4, delta=d))
    # the suite's settings (bench.py): pmemd's beta and grid, OPTIMAL
    for tag, b in [('suite-nve', 3.9467022824439113), ('suite-npt', 3.486390059027421)]:
        Kg = {'jac': [64, 64, 64], 'cel': [256, 128, 128], 'stmv': [224, 224, 240]}[name]
        d, _, _ = gauss_eval(b, Kg, 4, optimal=True)
        emit(dict(what=tag, beta=b, K=Kg, order=4, optimal=True, delta=d))
        d, _, _ = gauss_eval(b, Kg, 4, optimal=False)
        emit(dict(what=tag, beta=b, K=Kg, order=4, optimal=False, delta=d))
    # the real-space part alone: grid and order large
    for tol in [1e-4, 1e-5, 1e-6]:
        b = beta_of_tol(tol, rc)
        d, _, _ = gauss_eval(b, grid_of(160), 10)
        emit(dict(what='gauss-realspace-limit', tol=tol, beta=b, delta=d))
elif method == 'gauss':
    order = arg
    for tol in [3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]:
        b = beta_of_tol(tol, rc)
        Fd = pr.forces(q, gauss_fr(b))
        for n in range(16, 140, 2):
            K = grid_of(n)
            if min(K) < 2 * order:
                continue
            E, Fr = Rec(x, q, L, K, 'gauss', order=order, beta=b).run()
            d = delta(Fd + Fr, Fref)
            emit(dict(method='gauss', order=order, tol=tol, beta=b, K=K, delta=d))
            if d < 2e-6 or n > 100 and d > 1e-2:
                break
elif method == 'esp':
    P = arg
    for eps in [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6]:
        c = pswf.c_of_eps(eps)
        for c1fac in [0.5, 1.0]:
            c1 = pswf.c_of_eps(eps * c1fac)
            split = pswf.Splitting(c)
            win = pswf.Window(c1, P)
            Fd = pr.forces(q, esp_fr(split, rc))
            nf0 = c * L.max() / (np.pi * rc)
            for n in range(max(8, int(0.6 * nf0)), int(2.5 * nf0) + 4, 2):
                K = grid_of(n)
                if min(K) < 2 * P:
                    continue
                E, Fr = Rec(x, q, L, K, 'esp', window=win, split=split, rc=rc).run()
                d = delta(Fd + Fr, Fref)
                emit(dict(method='esp', P=P, eps=eps, c=c, c1=c1, c1fac=c1fac, nf_eq15=nf0, K=K, delta=d))
                if d < 0.7 * eps:
                    break
