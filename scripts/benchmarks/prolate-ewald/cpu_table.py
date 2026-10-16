"""The tables of the prolate kernel against the Gaussian on the CPU target
(issue #371), from the jobs of cpu_job.py.

    cpu_table.py NAME[:TAG,TAG...] ... [--cutoff R,R] [--doc]

For each system NAME (default tags t1e-3, t1e-4, t1e-5 under
$ESP_CPU_DIR/runs/NAME): the whole step of every case, the mean and sample
standard deviation over the rounds after the first, at each number of
threads, with its ratio to the prolate default (P = 6) of the tier; the
parts of the step of the sampled runs (scripts/benchmarks/cpu-breakdown/
prof.py --json, as parts.json in each run's directory: the share of the wall
clock of the main thread, times the mean rate; at one thread the loop over
pairs is not a parallel region and is counted in the rest); and an estimate of the step
with a shorter cutoff of the Coulomb sum alone (--cutoff, default 7 and 6 Å),
from the parts at 16 threads: the loop over pairs and the builds of the
list scale with the cube of the reach (cutoff + 1.5 Å), the points of the
grid with the inverse cube of the cutoff (both kernels: the spacing of the
Gaussian follows beta, which follows 1/r_c at a fixed erfc(beta r_c), and
that of the prolate kernel follows r_c by eq. 15 of the paper), the FFT
with M log M and the product with M, and the spreading, the window or the
B-splines, the gathering, and the rest do not change. That estimate is the
paper's Coulomb-only setting; with the Lennard-Jones at 9 Å the loop over
pairs would not shrink, and only the grid would grow.

--doc prints instead the compact tables of docs/pme-m1.md §7: for each
system and tier the prolate default, the fastest prolate case where it is
another, the Gaussian of order 4 (the paper's kind of baseline), and the
fastest Gaussian at 16 threads (the fair one); their parts at 16 threads;
and the ratios, with the estimate at a Coulomb-only cutoff.
"""
import json
import math
import os
import statistics
import sys

S = os.environ['ESP_CPU_DIR']
REACH = 1.5
CUT = 9.0
PARTS = [('spread', ['PME spread']), ('window', ['PME splines']), ('FFT', ['PME FFT']),
         ('product', ['PME convolution']), ('gather', ['PME gather']),
         ('pairs', ['pairs', 'excluded pairs']), ('list', ['list builds'])]
LABEL = {'prolate-p5': 'prolate, P = 5', 'prolate-p6': 'prolate, P = 6 (default)',
         'prolate-p7': 'prolate, P = 7', 'gaussian-o4': 'Gaussian, order 4',
         'gaussian-o6': 'Gaussian, order 6', 'gaussian-o8': 'Gaussian, order 8',
         'gaussian-default': "Gaussian, MDIR's default"}
TIER = {'0.001': '10⁻³', '0.0001': '10⁻⁴', '1e-05': '10⁻⁵'}


def sci(x):
    if not x:
        return '—'
    e = math.floor(math.log10(x))
    sup = str(e).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return f'{x / 10 ** e:.1f} × 10{sup}'


def load(name, tags):
    rows = []
    for tag in tags:
        f = f'{S}/runs/{name}/{tag}/results.jsonl'
        if os.path.exists(f):
            rows += [json.loads(l) for l in open(f)]
    return rows


def parts_of(row, rate):
    p = f"{row['dir']}/parts.json"
    if not os.path.exists(p):
        return None
    wall = json.load(open(p))['wall_part']
    total = sum(wall.values())
    out = {}
    for label, names in PARTS:
        out[label] = rate * sum(wall.get(n, 0) for n in names) / total
    out['rest'] = rate - sum(out.values())
    if not wall.get('pairs'):
        # One thread: the loops of the entry are not parallel regions, and
        # the loop over pairs falls in the serial code of the entry, with
        # the rest.
        out['pairs'] = None
    return out


def main():
    args = sys.argv[1:]
    cutoffs = [7.0, 6.0]
    systems = []
    doc = None
    while args:
        if args[0] == '--doc':
            doc = []; args = args[1:]
        elif args[0] == '--cutoff':
            cutoffs = [float(v) for v in args[1].split(',')]; args = args[2:]
        else:
            systems.append(args[0]); args = args[1:]
    summary = []
    for spec in systems:
        name, _, tags = spec.partition(':')
        tags = tags.split(',') if tags else ['t1e-3', 't1e-4', 't1e-5']
        rows = load(name, tags)
        for tier in ['0.001', '0.0001', '1e-05']:
            tr = [r for r in rows if r['tier'] == tier]
            if not tr:
                continue
            hosts = sorted({(r['host'], r['cpu']) for r in tr})
            threads = sorted({r['threads'] for r in tr}, reverse=True)
            keys = list(dict.fromkeys(r['key'] for r in tr))
            stat = {}
            for k in keys:
                for t in threads:
                    v = [r['rate_ms'] for r in tr if r['key'] == k and r['threads'] == t
                         and not r['sampled'] and r['round'] > 0 and r['rate_ms']]
                    if v:
                        stat[k, t] = (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0, len(v))
            first = {r['key']: r for r in tr}
            if doc is not None:
                out = sys.stdout
                sys.stdout = open(os.devnull, 'w')
            print(f'\n### {name}, ε = {TIER[tier]} ({", ".join(f"{h} ({c})" for h, c in hosts)})\n')
            head = '| Case | Grid | Points | Δ double | Δ mixed | ' + ' | '.join(
                f'{t} thr. (ms) | ratio' for t in threads) + ' |'
            print(head)
            print('|' + '---|' * (5 + 2 * len(threads)))
            for k in keys:
                r = first[k]
                cells = []
                for t in threads:
                    if (k, t) in stat:
                        m, s, n = stat[k, t]
                        base = stat.get(('prolate-p6', t))
                        cells += [f'{m:.2f} ± {s:.2f}', f'{m / base[0]:.2f}' if base else '—']
                    else:
                        cells += ['', '']
                g = r['grid']
                print(f"| {LABEL.get(k, k)} | {g[0]}×{g[1]}×{g[2]} | {r['points']:,} | {sci(r['delta'])} | "
                      f"{sci(r.get('delta_mixed'))} | " + ' | '.join(cells) + ' |')
            # the parts of the sampled runs
            print(f'\nParts (ms a step; the sampled run\'s shares times the mean rate):\n')
            print('| Case | Threads | ' + ' | '.join(p for p, _ in PARTS) + ' | rest | step |')
            print('|' + '---|' * (len(PARTS) + 4))
            parts16 = {}
            for k in keys:
                for t in threads:
                    sr = [r for r in tr if r['key'] == k and r['threads'] == t and r['sampled']]
                    if not sr or (k, t) not in stat:
                        continue
                    p = parts_of(sr[0], stat[k, t][0])
                    if not p:
                        continue
                    if t == threads[0]:
                        parts16[k] = p
                    print(f'| {LABEL.get(k, k)} | {t} | ' + ' | '.join(
                        '—' if p[x] is None else f'{p[x]:.2f}' for x, _ in PARTS)
                          + f" | {p['rest']:.2f} | {stat[k, t][0]:.2f} |")
            # a shorter cutoff of the Coulomb sum alone, from the parts at 16 threads
            estimates = {}
            for k, p in parts16.items():
                m = first[k]['points']
                estimates[k] = []
                for c in [CUT] + cutoffs:
                    pair = ((c + REACH) / (CUT + REACH)) ** 3
                    grid = (CUT / c) ** 3
                    fft = grid * math.log(m * grid) / math.log(m)
                    estimates[k].append((p['pairs'] + p['list']) * pair + p['FFT'] * fft + p['product'] * grid
                                        + p['spread'] + p['window'] + p['gather'] + p['rest'])
            if doc is not None:
                sys.stdout.close()
                sys.stdout = out
                t16 = threads[0]
                gs = {k: stat[k, t16][0] for k in keys if k.startswith('gaussian-o') and (k, t16) in stat}
                ps = {k: stat[k, t16][0] for k in keys if k.startswith('prolate') and (k, t16) in stat}
                fast, bestp = min(gs, key=gs.get), min(ps, key=ps.get)
                doc.append(dict(name=name, tier=tier, cpu=', '.join(sorted({c for _, c in hosts})),
                                threads=threads, stat=stat, first=first, parts=parts16, est=estimates,
                                show=list(dict.fromkeys(['prolate-p6', bestp, 'gaussian-o4', fast])),
                                fast=fast, bestp=bestp))
                continue
            if parts16:
                print(f'\nEstimate with a Coulomb-only cutoff (no Lennard-Jones), ms a step at '
                      f'{threads[0]} threads:\n')
                print('| Case | ' + ' | '.join(f'{c:g} Å' for c in [CUT] + cutoffs) + ' |')
                print('|' + '---|' * (2 + len(cutoffs)))
                for k, p in parts16.items():
                    m = first[k]['points']
                    cells = []
                    for c in [CUT] + cutoffs:
                        pair = ((c + REACH) / (CUT + REACH)) ** 3
                        grid = (CUT / c) ** 3
                        fft = grid * math.log(m * grid) / math.log(m)
                        est = (p['pairs'] + p['list']) * pair + p['FFT'] * fft + p['product'] * grid \
                            + p['spread'] + p['window'] + p['gather'] + p['rest']
                        cells.append(f'{est:.2f}')
                    print(f'| {LABEL.get(k, k)} | ' + ' | '.join(cells) + ' |')
            for t in threads:
                if ('prolate-p6', t) not in stat:
                    continue
                gs = {k: stat[k, t][0] for k in keys if k.startswith('gaussian-o') and (k, t) in stat}
                ps = {k: stat[k, t][0] for k in keys if k.startswith('prolate') and (k, t) in stat}
                if not gs:
                    continue
                fast = min(gs, key=gs.get)
                bestp = min(ps, key=ps.get)
                summary.append(dict(system=name, tier=tier, threads=t, p6=stat['prolate-p6', t][0],
                                    o4=gs.get('gaussian-o4'), fastest=fast, fastest_ms=gs[fast],
                                    best_prolate=bestp, best_prolate_ms=ps[bestp]))
    if doc is not None:
        print_doc(doc, cutoffs)
        return
    print('\n### Summary: the prolate default against the Gaussian\n')
    print('| System | ε | Threads | prolate P = 6 (ms) | / order 4 | fastest Gaussian | / fastest | '
          'fastest prolate / fastest Gaussian |')
    print('|---|---|---|---|---|---|---|---|')
    for s in summary:
        o4 = f"{s['p6'] / s['o4']:.2f}" if s['o4'] else '—'
        print(f"| {s['system']} | {TIER[s['tier']]} | {s['threads']} | {s['p6']:.2f} | {o4} | "
              f"{LABEL[s['fastest']]} ({s['fastest_ms']:.2f}) | {s['p6'] / s['fastest_ms']:.2f} | "
              f"{s['best_prolate_ms'] / s['fastest_ms']:.2f} ({LABEL[s['best_prolate']]}) |")


NAMES = {'jac': 'JAC', 'tip3p-32k': 'TIP3P 34k', 'tip3p-103k': 'TIP3P 103k', 'cel': 'Cellulose'}
SHORT = {'prolate-p5': 'Prolate, P = 5', 'prolate-p6': 'Prolate, P = 6', 'prolate-p7': 'Prolate, P = 7',
         'gaussian-o4': 'Gaussian, order 4', 'gaussian-o6': 'Gaussian, order 6',
         'gaussian-o8': 'Gaussian, order 8'}


def print_doc(doc, cutoffs):
    print('| System | ε | Kernel | Grid | Points | Δ | 16 threads | 4 threads | 1 thread |')
    print('|---|---|---|---|---|---|---|---|---|')
    for d in doc:
        for k in d['show']:
            r = d['first'][k]
            g = r['grid']
            cells = []
            for t in (16, 4, 1):
                cells.append(f"{d['stat'][k, t][0]:.1f}" if (k, t) in d['stat'] else '—')
            name = SHORT[k] + (' (fastest)' if k in (d['fast'], d['bestp'])
                               and k not in ('gaussian-o4', 'prolate-p6') else '')
            print(f"| {NAMES[d['name']]} | {TIER[d['tier']]} | {name} | {g[0]}×{g[1]}×{g[2]} | {r['points']:,} | "
                  f"{sci(r['delta'])} | " + ' | '.join(cells) + ' |')
    print('\nParts at 16 threads, ms a step:\n')
    print('| System | ε | Kernel | Spreading | Window or B-splines | FFT | Product | Gathering | Pairs | Rest | Step |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for d in doc:
        for k in d['show']:
            p = d['parts'].get(k)
            if not p:
                continue
            print(f"| {NAMES[d['name']]} | {TIER[d['tier']]} | {SHORT[k]} | {p['spread']:.2f} | {p['window']:.2f} | "
                  f"{p['FFT']:.2f} | {p['product']:.2f} | {p['gather']:.2f} | {p['pairs']:.2f} | "
                  f"{p['list'] + p['rest']:.2f} | {d['stat'][k, 16][0]:.2f} |")
    print('\nRatios of the step (prolate default over the Gaussian), and at a Coulomb-only cutoff '
          '(estimated, 16 threads):\n')
    print('| System | CPU | ε | / order 4, 16 / 4 / 1 threads | / fastest, 16 / 4 / 1 threads | '
          + ' | '.join(f'/ fastest, {c:g} Å' for c in cutoffs) + ' |')
    print('|---|---|---|---|---|' + '---|' * len(cutoffs))
    for d in doc:
        st = d['stat']
        def ratio(b, t):
            return f"{st['prolate-p6', t][0] / st[b, t][0]:.2f}" if ('prolate-p6', t) in st and (b, t) in st else '—'
        fastest = {t: min((k for k in st if k[1] == t and k[0].startswith('gaussian-o')), key=lambda k: st[k][0])[0]
                   for t in d['threads']}
        e = d['est']
        # against the fastest Gaussian order at each cutoff, which may change
        gk = [k for k in e if k.startswith('gaussian-o')]
        est = [f"{e['prolate-p6'][i + 1] / min(e[k][i + 1] for k in gk):.2f}" if 'prolate-p6' in e and gk else '—'
               for i in range(len(cutoffs))]
        print(f"| {NAMES[d['name']]} | {d['cpu'].replace(' 32-Core Processor', '')} | {TIER[d['tier']]} | "
              + ' / '.join(ratio('gaussian-o4', t) for t in (16, 4, 1)) + ' | '
              + ' / '.join(ratio(fastest[t], t) if t in fastest else '—' for t in (16, 4, 1)) + ' | '
              + ' | '.join(est) + ' |')


if __name__ == '__main__':
    main()
