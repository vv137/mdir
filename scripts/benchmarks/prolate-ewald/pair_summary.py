"""Tables of #380 from the jobs of pair_job.py: the step (mean of the rounds
after the first, its standard deviation), the loop over pairs at 16 threads
from the sampler (its share of the main thread's wall clock times the rate
of the sampled run), and the ratios prolate / Gaussian, for main and the
branch. A round beyond 1.5 times the median of its case is dropped and
listed.

    pair_summary.py CPU_DIR [GPU_DIR]

CPU_DIR and GPU_DIR hold the outputs of pair_job.py (a directory a job).
The parts of the sampled runs come from cpu-breakdown/prof.py, which emits
the program with $MDIR_MAIN or $MDIR_BRANCH (the programs of the builds
`main` and `br`) and translates it with $MLIR_TRANSLATE.
"""
import glob
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict

PROF = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cpu-breakdown', 'prof.py')
BINS = {'main': os.environ.get('MDIR_MAIN', 'mdir'), 'br': os.environ.get('MDIR_BRANCH', 'mdir')}
ORDER = ['jac', 'tip3p-34k', 'tip3p-103k', 'cellulose']
NAMES = {'jac': 'JAC', 'tip3p-34k': 'TIP3P 34k', 'tip3p-103k': 'TIP3P 103k', 'cellulose': 'Cellulose'}
TIERS = {'1e-03': '10⁻³', '1e-04': '10⁻⁴', '1e-05': '10⁻⁵'}


srate = {}
DROPPED = []


def clean(k, v):
    """The rounds within 1.5 times the median of their variant; a round of
    another job's interference is dropped and listed."""
    m = statistics.median(v)
    kept = [x for x in v if x <= 1.5 * m]
    for x in v:
        if x > 1.5 * m:
            DROPPED.append((k, x, m))
    return kept


def load(d):
    rates = defaultdict(list)
    sampled = {}
    cpu = {}
    compile_s = defaultdict(list)
    global srate
    for f in glob.glob(f'{d}/*/results.jsonl'):
        for line in open(f):
            r = json.loads(line)
            k = (r['system'], r['tier'], r['key'], r['build'], r['threads'])
            cpu[(r['system'], r['tier'])] = r['cpu']
            if r['mode'] == 'sampled':
                sampled[k] = r['dir']
                srate[k] = r['rate_ms']
            elif r['mode'] == 'profile':
                sampled[k] = r['dir']
            elif r['round'] > 0 and r['rate_ms']:
                rates[k].append(r['rate_ms'])
                log = open(r['dir'] + '/run.log').read()
                import re
                m = re.search(r'compiled in ([0-9.]+) s', log)
                if m:
                    compile_s[(r['system'], r['key'], r['build'])].append(float(m.group(1)))
    rates = defaultdict(list, {k: clean(k, v) for k, v in rates.items()})
    return rates, sampled, cpu, compile_s


def pairs_share(d, build):
    out = f'{d}/parts.json'
    if not os.path.exists(out):
        env = dict(os.environ, MDIR=BINS[build])
        subprocess.run([sys.executable, PROF, f'{d}/prof', '--program', f'{d}/mdir.toml', '--json', out],
                       cwd=d, env=env, capture_output=True, text=True)
    if not os.path.exists(out):
        return None
    w = json.load(open(out))['wall_part']
    total = sum(w.values())
    return w.get('pairs', 0) / total if total else None


def gpu_pairs(d):
    import re
    log = f'{d}/run.log'
    ks = []
    for line in open(log):
        m = re.match(r'mdrt: (\S*pair_for\S*)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)', line)
        if m:
            ks.append((int(m.group(2)), int(m.group(5)), float(m.group(4))))
    if not ks:
        return None
    top = max(k[0] for k in ks)
    return max((k for k in ks if k[0] == top), key=lambda k: k[1])[2]


def mean(v):
    return statistics.mean(v) if v else float('nan')


def main():
    rates, sampled, cpu, compile_s = load(sys.argv[1])
    print('## CPU: the step, ms (mean ± sd of the rounds after the first)\n')
    print('| System | CPU | ε | Kernel | Build | 16 threads | 4 threads | 1 thread | Pairs at 16 (ms) |')
    print('|---|---|---|---|---|---|---|---|---|')
    ratios = []
    for s in ORDER:
        for t in TIERS:
            if (s, t) not in cpu:
                continue
            for key in ('prolate-p6', 'gaussian'):
                for b in ('main', 'br'):
                    cells = []
                    for th in (16, 4, 1):
                        v = rates.get((s, t, key, b, th), [])
                        cells.append(f'{mean(v):.2f} ± {statistics.stdev(v):.2f}' if len(v) > 1 else '—')
                    pp = ''
                    d = sampled.get((s, t, key, b, 16))
                    if d:
                        share = pairs_share(d, b)
                        if share is not None:
                            pp = f'{share * srate[(s, t, key, b, 16)]:.2f}'
                    print(f'| {NAMES[s]} | {cpu[(s, t)].replace(" 32-Core Processor", "").replace(" 16-Core Processor", "")} | {TIERS[t]} | {key} | {b} | ' + ' | '.join(cells) + f' | {pp} |')
            row = [NAMES[s], TIERS[t]]
            for b in ('main', 'br'):
                rr = []
                for th in (16, 4, 1):
                    p = mean(rates.get((s, t, 'prolate-p6', b, th), []))
                    g = mean(rates.get((s, t, 'gaussian', b, th), []))
                    rr.append(f'{p / g:.2f}')
                row.append(' / '.join(rr))
            pr = []
            for b in ('main', 'br'):
                dp, dg = sampled.get((s, t, 'prolate-p6', b, 16)), sampled.get((s, t, 'gaussian', b, 16))
                if dp and dg:
                    sp, sg = pairs_share(dp, b), pairs_share(dg, b)
                    if sp and sg:
                        pr.append(f'{sp * srate[(s, t, "prolate-p6", b, 16)] / (sg * srate[(s, t, "gaussian", b, 16)]):.2f}')
                        continue
                pr.append('—')
            row += pr
            p_main = mean(rates.get((s, t, 'prolate-p6', 'main', 16), []))
            p_br = mean(rates.get((s, t, 'prolate-p6', 'br', 16), []))
            g_main = mean(rates.get((s, t, 'gaussian', 'main', 16), []))
            g_br = mean(rates.get((s, t, 'gaussian', 'br', 16), []))
            row += [f'{p_br / p_main:.3f}', f'{g_br / g_main:.3f}']
            ratios.append(row)
    print('\n## CPU: prolate / Gaussian\n')
    print('| System | ε | step, main: 16 / 4 / 1 | step, branch: 16 / 4 / 1 | pairs at 16, main | pairs at 16, branch | prolate branch / main at 16 | Gaussian branch / main at 16 |')
    print('|---|---|---|---|---|---|---|---|')
    for r in ratios:
        print('| ' + ' | '.join(r) + ' |')
    print('\nDropped rounds:', DROPPED)
    print('\n## Compile time, s (mean over the timed runs)\n')
    for k in sorted(compile_s):
        print(k, f'{mean(compile_s[k]):.2f}', len(compile_s[k]))
    if len(sys.argv) > 2:
        rates, profiled, gpu, _ = load(sys.argv[2])
        print('\n## RTX 4090: the step, ms, and the loop over pairs, µs\n')
        print('| System | ε | Kernel | main step | branch step | main pairs | branch pairs |')
        print('|---|---|---|---|---|---|---|')
        for s in ORDER:
            for t in TIERS:
                if (s, t) not in gpu:
                    continue
                for key in ('prolate-p6', 'gaussian'):
                    c = []
                    for b in ('main', 'br'):
                        v = rates.get((s, t, key, b, 1), [])
                        c.append(f'{mean(v):.3f} ± {statistics.stdev(v):.3f}' if len(v) > 1 else '—')
                    for b in ('main', 'br'):
                        d = profiled.get((s, t, key, b, 1))
                        c.append(f'{gpu_pairs(d):.1f}' if d else '—')
                    print(f'| {NAMES[s]} | {TIERS[t]} | {key} | ' + ' | '.join(c) + ' |')


if __name__ == '__main__':
    main()
