"""GPU job: MDIR's step on the control files of a system, alternated in
rounds (the first dropped), then one profiled run of each.
mdir_job.py RUNS_JSON OUT_JSONL ROUNDS [MDIR]"""
import json, os, re, subprocess, sys, statistics

runs = json.load(open(sys.argv[1]))
out = open(sys.argv[2], 'w')
rounds = int(sys.argv[3])
mdir = sys.argv[4]
subprocess.run(['nvidia-smi', '--query-gpu=name,clocks.sm,power.limit', '--format=csv'], check=True)
fail = 0
res = {r['tag']: [] for r in runs}


def one(r, env=None, log='mdir.out'):
    p = subprocess.run([mdir, 'run', 'mdir.toml'], cwd=r['dir'], capture_output=True, text=True, env=env)
    open(os.path.join(r['dir'], log), 'w').write(p.stdout + p.stderr)
    m = re.search(r'from step \d+ to step \d+, ([0-9.]+) ms per step', p.stdout + p.stderr) or \
        re.search(r'ran in [0-9.]+ s, ([0-9.]+) ms per step', p.stdout + p.stderr)
    return p.returncode, (float(m.group(1)) if m else None), p.stdout + p.stderr


for k in range(rounds):
    for r in runs:
        rc, ms, _ = one(r)
        if rc != 0 or ms is None:
            print('FAILED', r['tag'], flush=True)
            fail += 1
            continue
        print('round', k, r['tag'], ms, flush=True)
        if k > 0:
            res[r['tag']].append(ms)
for r in runs:
    v = res[r['tag']]
    d = dict(tag=r['tag'], case=r['case'], order=r['order'], ms=v,
             mean=statistics.mean(v) if v else None, sd=statistics.stdev(v) if len(v) > 1 else None)
    env = dict(os.environ, MDRT_PROFILE='1', MDRT_WAIT='1')
    rc, ms, text = one(r, env, 'profile.out')
    d['profile_ms'] = ms
    d['profile'] = [l for l in text.splitlines() if 'pme' in l.lower() or 'fft' in l.lower() or 'recip' in l.lower()][:80]
    out.write(json.dumps(d) + '\n')
    out.flush()
sys.exit(1 if fail else 0)
