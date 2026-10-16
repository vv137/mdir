"""Reference forces of a system: Gaussian Ewald at 9+ digits, two settings
compared.  Usage: run_ref.py NAME [NSAMPLE]"""
import sys, time
import numpy as np
from accuracy import *

name = sys.argv[1]
nsample = int(sys.argv[2]) if len(sys.argv) > 2 else 0
x, q, L = load(name)
N = len(x)
rng = np.random.default_rng(1)
sample = None if nsample == 0 else np.sort(rng.choice(N, nsample, replace=False))
res = {}
for tag, rc, tol, h, order in [('A', 1.4, 1e-12, 0.040, 10), ('B', 1.6, 1e-14, 0.034, 12)]:
    t0 = time.time()
    beta = beta_of_tol(tol, rc)
    K = [next_size(int(np.ceil(L[a] / h))) for a in range(3)]
    pr = Pairs(x, L, rc, sample)
    Fd = pr.forces(q, gauss_fr(beta))
    if sample is not None:
        Fd = Fd[sample]
    E, Fr = Rec(x, q, L, K, 'gauss', order=order, beta=beta).run(sample)
    res[tag] = Fd + Fr
    print(tag, 'rc', rc, 'beta', beta, 'K', K, 'order', order, 'Erec', E, f'{time.time()-t0:.1f}s', flush=True)
d = delta(res['A'], res['B'])
print('Delta(A, B) =', d)
np.savez(f'{D}ref-{name}-{nsample}.npz', F=res['B'], sample=np.array([]) if sample is None else sample, dAB=d)
