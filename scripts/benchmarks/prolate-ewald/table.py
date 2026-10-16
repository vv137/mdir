"""The best variant of each case, by the time of the whole sequence.
table.py PROTO_JSONL [check]  (check: Delta of the GPU forces, JAC only)"""
import json, os, sys
D = os.path.join(os.environ.get('ESP_WORK', '.'), '')
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1])]
check = len(sys.argv) > 2
best = {}
for r in rows:
    key = (r['case']['file'], r['prec'])
    if key not in best or r['sequence_us'] < best[key]['sequence_us']:
        best[key] = r
if check:
    from accuracy import load, Pairs, gauss_fr, esp_fr, delta
    import pswf
    x, q, L = load('jac')
    ref = np.load(D + 'ref-jac-0.npz')['F']
    pr = Pairs(x, L, 0.8)
print(f"{'case':28s} {'prec':4s} {'K':>16s} {'layout':6s} {'lanes':>5s} {'order':6s} {'seq':>8s} {'seqE':>8s} {'spread':>7s} {'copy':>6s} {'fft2':>6s} {'prod':>6s} {'gather':>7s} {'places':>6s}" + ('  Delta_gpu' if check else ''))
for key in sorted(best, key=lambda k: (k[1], k[0])):
    r = best[key]
    c = r['case']
    extra = ''
    if check and r['forces'] != '-' and c['name'] == 'jac':
        fr = np.fromfile(r['forces'], dtype=np.float64)[:3 * len(x)].reshape(-1, 3)
        if c['key'].startswith('gauss'):
            fd = pr.forces(q, gauss_fr(c['beta']))
        else:
            fd = pr.forces(q, esp_fr(pswf.Splitting(c['c']), 0.8))
        extra = f"  {delta(fd + fr, ref):.3e} (host {c.get('delta', float('nan')):.3e})"
    elif check:
        # the forces of the best variant may not have been written: find the written one
        w = [x_ for x_ in rows if x_['case']['file'] == c['file'] and x_['prec'] == r['prec'] and x_['forces'] != '-']
        if w and c['name'] == 'jac':
            fr = np.fromfile(w[0]['forces'], dtype=np.float64)[:3 * len(x)].reshape(-1, 3)
            fd = pr.forces(q, gauss_fr(c['beta'])) if c['key'].startswith('gauss') else pr.forces(q, esp_fr(pswf.Splitting(c['c']), 0.8))
            extra = f"  {delta(fd + fr, ref):.3e} (host {c.get('delta', float('nan')):.3e})"
    name = f"{c['name']}-{c['target']}-{c['key']}"
    print(f"{name:28s} {r['prec']:4s} {str(r['K']):>16s} {r['layout']:6s} {r['glanes']:5d} {r['order']:6s} {r['sequence_us']:8.1f} {r['sequence_energy_us']:8.1f} {r['spread_us']:7.1f} {r['copy_us']:6.1f} {r['fft_fwd_us'] + r['fft_bwd_us']:6.1f} {r['product_us']:6.1f} {r['gather_us']:7.1f} {r['places_us']:6.1f}" + extra)
