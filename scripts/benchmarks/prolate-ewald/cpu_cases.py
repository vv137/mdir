"""The cases of the prolate kernel against the Gaussian on the CPU target
(issue #371), matched by the measured force error at a cutoff of 9 Å.

    cpu_cases.py accuracy NAME DIR [GAUSSIAN-SPACINGS.json]
    cpu_cases.py mixed DIR CASE...
    cpu_cases.py pick DIR OUT.json

`accuracy` writes the control files of the sweep of Delta for the system
NAME, a directory under $ESP_CPU_DIR/sys with sys.prmtop and sys.inpcrd, and
npt.h5 for a state equilibrated at constant pressure, which is then the
input (its positions and cell): grid_sweep.py `run` and `report` run them
and measure Delta (eq. 16 of Liang et al. 2026 over the Coulomb forces of
the topology, one step on the CPU in double precision, against the Gaussian
at erfc 1e-13, 0.33 Å, order 8, as D266). The Gaussian of orders 4, 6, and
8 at erfc(beta r_c) from 1e-4 to 1e-7 over the spacings below, the prolate
kernel at its automatic grid for epsilon 1e-3, 1e-4, 1e-5 and P = 5, 6, 7,
MDIR's default, and the uncharged topology. With GAUSSIAN-SPACINGS (the
`pick` output of another system), the Gaussian takes only the spacings
within a factor 1.25 of the picks there, since Delta at one cutoff follows
the spacing.

`mixed` copies the cases to <case>-mixed in mixed precision, the precision of
the timings, so that `report` gives their Delta there too.

`pick` reads DIR/report.jsonl (grid_sweep.py report) and writes the cases of each tier, epsilon = 1e-3, 1e-4, 1e-5: the
prolate kernel at its automatic grid for P = 5, 6, 7, and for each Gaussian
order the grid of fewest points, over the tolerances, whose Delta is at most
1.1 times that of the prolate default (P = 6) of the tier; at 1e-3 also
MDIR's Gaussian default. The fastest Gaussian at a tier is then the fastest
of its orders in the timing.
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_sweep import REFERENCES, control, smooth  # noqa: E402

CUTOFF = 9.0
EPSILONS = [1e-3, 1e-4, 1e-5]
POINTS = [5, 6, 7]
GAUSSIAN = {4: [1e-4, 1e-5, 1e-6], 6: [1e-4, 1e-5, 1e-6, 1e-7], 8: [1e-4, 1e-5, 1e-6, 1e-7]}
SPACINGS = [2.6, 2.4, 2.2, 2.05, 1.9, 1.78, 1.66, 1.55, 1.45, 1.35, 1.26, 1.18, 1.1, 1.03,
            0.96, 0.9, 0.84, 0.78, 0.73, 0.68, 0.64, 0.6, 0.56, 0.52, 0.49, 0.46, 0.43, 0.4,
            0.37, 0.345, 0.32, 0.3, 0.28]
FINEST = {4: 0.28, 6: 0.46, 8: 0.56}   # the finest spacing tried for each order
COARSEST = {4: 1.9, 6: 2.6, 8: 2.6}
MATCH = 1.1


def system(name):
    d = os.path.join(os.environ['ESP_CPU_DIR'], 'sys', name)
    top, crd, npt = f'{d}/sys.prmtop', f'{d}/sys.inpcrd', f'{d}/npt.h5'
    if os.path.exists(npt):
        out = subprocess.run([os.environ.get('MDIR', 'mdir'), 'checkpoint', npt],
                             capture_output=True, text=True, check=True).stdout
        edges = [10 * float(v) for v in re.search(r'box:\s+(\S+) (\S+) (\S+) nm', out).groups()]
        return top, crd, npt, edges
    last = open(crd).read().splitlines()[-1].split()
    return top, crd, None, [float(v) for v in last[:3]]


def accuracy(name, d, near=None):
    d = os.path.abspath(d)
    top, crd, npt, edges = system(name)
    os.makedirs(d, exist_ok=True)
    import parmed
    neutral = parmed.load_file(top)
    for atom in neutral.atoms:
        atom.charge = 0.0
    neutral.save(f'{d}/uncharged.prmtop', overwrite=True)
    cases = {'reference': REFERENCES['reference'], 'uncharged': ['tolerance = 1e-5'],
             'gaussian-default': []}
    keep = None
    if near:
        # the spacings of the picks of another system, by order
        keep = {}
        for tier in json.load(open(near)).values():
            for c in tier:
                if c['kernel'] == 'gaussian':
                    keep.setdefault(c['order'], []).append(c['spacing'])
    for order, tolerances in GAUSSIAN.items():
        for tol in tolerances:
            seen = set()
            for h in SPACINGS:
                if not FINEST[order] <= h <= COARSEST[order]:
                    continue
                if keep is not None and not any(s / 1.25 <= h <= s * 1.25 for s in keep.get(order, [])):
                    continue
                grid = tuple(smooth(math.ceil(e / h - 1e-9)) for e in edges)
                if grid in seen:
                    continue
                seen.add(grid)
                cases[f'g{order}-{tol:g}-{grid[0]}x{grid[1]}x{grid[2]}'] = [
                    f'tolerance = {tol:g}', f'order = {order}', f'grid = [{grid[0]}, {grid[1]}, {grid[2]}]']
    for eps in EPSILONS:
        for points in POINTS:
            if (points, eps) == (5, 1e-5):
                continue  # refused: the window of 5 points is too short (D266)
            cases[f'p{points}-{eps:g}-auto'] = ['kernel = "PROLATE"', f'tolerance = {eps:g}',
                                                f'order = {points}']
    for case, pme in cases.items():
        os.makedirs(f'{d}/{case}', exist_ok=True)
        t = f'{d}/uncharged.prmtop' if case == 'uncharged' else top
        text = control(t, crd, CUTOFF, pme)
        if npt:
            text = text.replace('[output]', f'checkpoint  = "{npt}"\n\n[output]', 1)
        open(f'{d}/{case}/run.toml', 'w').write(text)
    json.dump(edges, open(f'{d}/edges.json', 'w'))
    print(len(cases), 'cases; edges', edges)


def mixed(d, names):
    for case in names:
        src, dst = f'{d}/{case}', f'{d}/{case}-mixed'
        os.makedirs(dst, exist_ok=True)
        text = open(f'{src}/run.toml').read().replace('precision = "DOUBLE"', 'precision = "MIXED"')
        g = re.search(r'grid of (\d+) x (\d+) x (\d+)', open(f'{src}/run.log').read()) \
            if os.path.exists(f'{src}/run.log') else None
        if g and 'grid =' not in text:
            text = text.replace('[pme]\n', f'[pme]\ngrid = [{g[1]}, {g[2]}, {g[3]}]\n', 1)
        open(f'{dst}/run.toml', 'w').write(text)


def parse(row):
    """kernel, order, tolerance of a row of the report, or None."""
    m = re.match(r'g(\d)-([0-9.e+-]+)-\d+x\d+x\d+$', row['case'])
    if m:
        return 'gaussian', int(m[1]), float(m[2])
    m = re.match(r'p(\d)-([0-9.e+-]+)-auto$', row['case'])
    if m:
        return 'prolate', int(m[1]), float(m[2])
    if row['case'] == 'gaussian-default':
        return 'gaussian', 4, 1e-5
    return None


def pme_lines(kernel, order, tol, grid):
    out = [f'tolerance = {tol:g}', f'order = {order}', 'grid = [%d, %d, %d]' % tuple(grid)]
    return (['kernel = "PROLATE"'] if kernel == 'prolate' else []) + out


def pick(d, out):
    report = f'{d}/report.jsonl'
    edges = json.load(open(f'{d}/edges.json'))
    rows = [json.loads(l) for l in open(report)]
    rows = [r for r in rows if not r['case'].endswith('-mixed')]
    mixed_delta = {r['case'][:-6]: r['delta'] for r in map(json.loads, open(report))
                   if r['case'].endswith('-mixed')}
    tiers = {}
    for eps in EPSILONS:
        cases = []
        for r in rows:
            p = parse(r)
            if p and p[0] == 'prolate' and p[2] == eps:
                cases.append(dict(key=f'prolate-p{p[1]}', kernel='prolate', order=p[1], tolerance=eps,
                                  grid=r['grid'], delta=r['delta'], case=r['case']))
        target = next(c['delta'] for c in cases if c['order'] == 6)
        for order in GAUSSIAN:
            ok = [r for r in rows if (p := parse(r)) and p[0] == 'gaussian' and p[1] == order
                  and r['case'] != 'gaussian-default' and r['delta'] <= MATCH * target]
            if not ok:
                print(f'{eps:g}: no grid of order {order} reaches {MATCH * target:.3g}')
                continue
            best = min(ok, key=lambda r: (math.prod(r['grid']), r['delta']))
            _, _, tol = parse(best)
            cases.append(dict(key=f'gaussian-o{order}', kernel='gaussian', order=order, tolerance=tol,
                              grid=best['grid'], delta=best['delta'], case=best['case']))
        if eps == 1e-3:
            # MDIR's default: erfc 1e-5, order 4, max_spacing 1.2 Å
            r = next(r for r in rows if r['case'] == 'gaussian-default')
            cases.append(dict(key='gaussian-default', kernel='gaussian', order=4, tolerance=1e-5,
                              grid=[smooth(math.ceil(e / 1.2 - 1e-9)) for e in edges],
                              delta=r['delta'], case=r['case']))
        for c in cases:
            c['points'] = math.prod(c['grid']) if c['grid'] else None
            c['pme'] = pme_lines(c['kernel'], c['order'], c['tolerance'], c['grid']) if c['grid'] else []
            c['delta_mixed'] = mixed_delta.get(c['case'])
            c['target'] = target
        tiers[f'{eps:g}'] = cases
    # the spacing of each Gaussian pick, for the sweep of the next system
    for cases in tiers.values():
        for c in cases:
            c['spacing'] = max(e / k for e, k in zip(edges, c['grid']))
    json.dump(tiers, open(out, 'w'), indent=1)
    for t, cases in tiers.items():
        print(t)
        for c in cases:
            print(f"  {c['key']:18s} {c['case']:28s} {str(c['grid']):18s} {c['points'] or 0:>11,} "
                  f"Delta {c['delta']:.3g}  mixed {c['delta_mixed'] or float('nan'):.3g}")


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'accuracy':
        accuracy(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif mode == 'mixed':
        mixed(sys.argv[2], sys.argv[3:])
    elif mode == 'pick':
        pick(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
