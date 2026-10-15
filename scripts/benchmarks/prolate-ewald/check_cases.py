"""Delta of the cases of a system on a sample of its particles, against the
sampled reference (run_ref.py NAME NSAMPLE).  check_cases.py NAME NSAMPLE"""
import sys, json
import numpy as np
from accuracy import *

name, ns = sys.argv[1], int(sys.argv[2])
D = os.path.join(os.environ.get('ESP_WORK', '.'), '')
ref = np.load(f'{D}ref-{name}-{ns}.npz')
sample = ref['sample'].astype(np.int64)
Fref = ref['F']
x, q, L = load(name)
pr = Pairs(x, L, 0.8, sample)
out = open(f'{D}check-{name}.jsonl', 'w')
for c in json.load(open(f'{D}cases/cases-{name}.json')):
    if c['key'].startswith('gauss'):
        Fd = pr.forces(q, gauss_fr(c['beta']))[sample]
        E, Fr = Rec(x, q, L, c['K'], 'gauss', order=c['order'], beta=c['beta']).run(sample)
    else:
        split = pswf.Splitting(c['c'])
        Fd = pr.forces(q, esp_fr(split, 0.8))[sample]
        E, Fr = Rec(x, q, L, c['K'], 'esp', window=pswf.Window(c['c1'], c['P']), split=split, rc=0.8).run(sample)
    d = delta(Fd + Fr, Fref)
    r = dict(name=name, target=c['target'], key=c['key'], K=c['K'], delta=d, delta_jac=c.get('delta'))
    out.write(json.dumps(r) + '\n'); out.flush()
    print(r, flush=True)
