"""The cases of the GPU measurement: for each target Delta, the cheapest
grids found on JAC (best-jac.json), scaled to each system by the spacing."""
import json, os, subprocess, sys
import numpy as np
from accuracy import load, next_size

D = os.path.join(os.environ.get('ESP_WORK', '.'), '')
best = json.load(open(D + 'best-jac.json'))
Ljac = 6.223
picks = {
    '0.001': ['gauss4', 'gauss6', 'esp5', 'esp6'],
    '0.0003': ['gauss4', 'gauss6', 'esp5', 'esp6'],
    '0.0001': ['gauss4', 'gauss6', 'gauss8', 'esp5', 'esp6', 'esp7'],
    '1e-05': ['gauss4', 'gauss6', 'gauss8', 'esp6', 'esp7', 'esp8'],
}
cases = []
for name in sys.argv[1:]:
    x, q, L = load(name)
    for t, keys in picks.items():
        for k in keys:
            b = best[t][k]
            h = Ljac / b['K'][0]
            K = [next_size(int(np.ceil(L[a] / h - 1e-9))) for a in range(3)]
            out = f'{D}cases/{name}-{t}-{k}.bin'
            if k.startswith('gauss'):
                args = [name, 'gauss', str(b['order']), repr(b['beta'])] + [str(v) for v in K] + [out]
            else:
                args = [name, 'esp', str(b['P']), repr(b['c']), repr(b['c1'])] + [str(v) for v in K] + [out]
            r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gen_case.py')] + args, capture_output=True, text=True, check=True)
            print(r.stdout.strip().replace('\n', ' | '))
            cases.append(dict(name=name, target=float(t), key=k, K=K, file=out, **{kk: b[kk] for kk in b if kk in ('beta', 'order', 'P', 'c', 'c1', 'delta')}))
    # the suite's own setting and MDIR's default
    for tag, beta, K in [('suite', 3.9467022824439113, {'jac': [64, 64, 64], 'cel': [256, 128, 128], 'stmv': [224, 224, 240]}[name]),
                         ('default', 3.904266592926094, [next_size(int(np.ceil(L[a] / 0.12))) for a in range(3)])]:
        out = f'{D}cases/{name}-{tag}-gauss4.bin'
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gen_case.py'), name, 'gauss', '4', repr(beta)] + [str(v) for v in K] + [out], check=True, capture_output=True)
        cases.append(dict(name=name, target=tag, key='gauss4', K=K, file=out, beta=beta, order=4))
json.dump(cases, open(D + 'cases/cases-' + '-'.join(sys.argv[1:]) + '.json', 'w'), indent=1)
