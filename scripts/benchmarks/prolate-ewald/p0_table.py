"""The table of P0 (#337) from the log of a job of p0_job.py: means and
standard deviations over the rounds after the first, in µs, of the
gathering with the weights evaluated in the kernel and precomputed, from
the grid in global memory and from tiles; the B-spline of odd order has a
modulus of 0 at K/2 and so an infinite energy, which does not change the
time of a kernel.  p0_table.py LOG"""
import ast, json, re, statistics, sys

rows = {}
for line in open(sys.argv[1]):
    m = re.match(r"(\d) (bspline|pswf) (\d) (\d+) (\{.*\}) D (\d+)", line)
    if m:
        r, window, P, lanes, values, D = m.groups()
        values = ast.literal_eval(values)
    else:
        m = re.match(r"UNPARSED \S+/\w+-(bspline|pswf)-(\d)\.bin '(.*)'$", line.strip())
        if not m:
            continue
        window, P, text = m.groups()
        text = ast.literal_eval("'" + text + "'").strip()
        text = re.sub(r'-?\binf\b|-?\bnan\b', '0', text)
        d = json.loads(text)
        values = {k: v for k, v in d.items() if k.startswith('p0_') or k == 'gather_us'}
        lanes, D = str(d['glanes']), str(d['D'])
        r = None
    rows.setdefault((window, int(P), int(lanes), int(D)), []).append(values)
# A row of an UNPARSED line has no round: drop the first of each key.
keys = ['p0_gather_us', 'p0_gather_tiled_us', 'p0_gather_tiled_padded_us', 'p0_precompute_us', 'p0_gather_pre_us',
        'p0_gather_pre_tiled_us', 'p0_gather_pre_tiled_padded_us']
print('| Window | P | D | Lanes | Gather, weights in the kernel: global | tiles | tiles, padded | Precompute | Gather, weights precomputed: global | tiles | tiles, padded |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for key in sorted(rows, key=lambda k: (k[1], k[0], k[2])):
    v = rows[key][1:]
    cell = lambda k: f"{statistics.mean(x[k] for x in v):.0f} ± {statistics.stdev(x[k] for x in v):.0f}"
    window, P, lanes, D = key
    print(f"| {'B-spline' if window == 'bspline' else 'PSWF'} | {P} | {D} | {lanes} | " + ' | '.join(cell(k) for k in keys) + ' |')
