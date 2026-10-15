"""For each method and target Delta, the smallest grid that reaches it."""
import json, glob, os, sys
D = os.path.join(os.environ.get('ESP_WORK', '.'), '')
import numpy as np

name = sys.argv[1] if len(sys.argv) > 1 else 'jac'
rows = []
for f in glob.glob(f'{D}sweep-{name}-*.jsonl'):
    for line in open(f):
        d = json.loads(line)
        if 'method' in d:
            rows.append(d)
targets = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
out = {}
for t in targets:
    print(f'--- Delta <= {t:g}')
    for key in [('gauss', 4), ('gauss', 6), ('gauss', 8), ('esp', 4), ('esp', 5), ('esp', 6), ('esp', 7), ('esp', 8)]:
        m, n = key
        cand = [r for r in rows if r['method'] == m and r.get('order', r.get('P')) == n and r['delta'] <= t]
        if not cand:
            print(f'{m} {n}: none')
            continue
        b = min(cand, key=lambda r: (np.prod(r['K']), r['delta']))
        extra = f"tol {b['tol']:g} beta {b['beta']:.4f}" if m == 'gauss' else f"eps {b['eps']:g} c {b['c']:.4f} c1 {b['c1']:.4f} nf(eq15) {b['nf_eq15']:.1f}"
        print(f"{m} {n}: K {b['K']} points {np.prod(b['K'])} Delta {b['delta']:.3e}  {extra}")
        out.setdefault(str(t), {})[f'{m}{n}'] = b
json.dump(out, open(f'{D}best-{name}.json', 'w'), indent=1)
