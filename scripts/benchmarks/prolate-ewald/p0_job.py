"""The measurement P0 of #337: the gathering of the B-spline and the PSWF
windows at the same P (4, 5, 6, 8) on one grid of a system, in one
program; then with the weights precomputed; alternated in rounds, the
first dropped; then Nsight Compute on the gathering kernels if it runs.

p0_job.py make NAME K1 K2 K3      (on the host: writes the cases)
p0_job.py run NAME ROUNDS OUT      (on the GPU)
"""
import json, os, statistics, subprocess, sys

D = os.path.join(os.environ.get('ESP_WORK', '.'), '')
HERE = os.path.dirname(os.path.abspath(__file__))
PS = [4, 5, 6, 8]

if sys.argv[1] == 'make':
    name, K = sys.argv[2], sys.argv[3:6]
    import pswf
    c = pswf.c_of_eps(1e-5)
    c1 = pswf.c_of_eps(5e-6)
    os.makedirs(D + 'p0', exist_ok=True)
    for P in PS:
        subprocess.run([sys.executable, HERE + '/gen_case.py', name, 'gauss', str(P), '3.4'] + K +
                       [f'{D}p0/{name}-bspline-{P}.bin'], check=True)
        subprocess.run([sys.executable, HERE + '/gen_case.py', name, 'esp', str(P), repr(c), repr(c1)] + K +
                       [f'{D}p0/{name}-pswf-{P}.bin'], check=True)
    sys.exit(0)

name, rounds, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]
bench = D + os.environ.get('BENCH', 'esp_bench')
subprocess.run(['nvidia-smi', '--query-gpu=name,clocks.sm,power.limit', '--format=csv'], check=True)
print('node', os.uname().nodename, flush=True)
env = dict(os.environ, ESP_P0='1')
keys = ['p0_gather_us', 'p0_gather_tiled_us', 'p0_precompute_us', 'p0_gather_pre_us', 'p0_gather_pre_tiled_us',
        'p0_gather_tiled_padded_us', 'p0_gather_pre_tiled_padded_us', 'gather_us']
res = {}
fail = 0
for r in range(rounds):
    for P in PS:
        for window in ['bspline', 'pswf']:
            for lanes in [8, 32]:
                case = f'{D}p0/{name}-{window}-{P}.bin'
                p = subprocess.run([bench, case, 'f', '100', 'tiled4', str(lanes), 'sorted', '-'], capture_output=True, text=True, env=env)
                if p.returncode != 0:
                    print('FAILED', case, p.stderr, flush=True)
                    fail += 1
                    continue
                try:
                    d = json.loads(p.stdout.replace('-nan', 'NaN').replace('nan', 'NaN'))
                except ValueError:
                    print('UNPARSED', case, repr(p.stdout), flush=True)
                    fail += 1
                    continue
                print(r, window, P, lanes, {k: d[k] for k in keys}, 'D', d['D'], flush=True)
                if r > 0:
                    res.setdefault(f'{window}-{P}-{lanes}', []).append({k: d[k] for k in keys} | {'D': d['D'], 'K': d['K']})
summary = {}
for key, v in res.items():
    summary[key] = {k: (statistics.mean(x[k] for x in v), statistics.stdev(x[k] for x in v) if len(v) > 1 else 0.0) for k in keys}
    summary[key]['D'] = v[0]['D']
    summary[key]['K'] = v[0]['K']
json.dump(summary, open(out, 'w'), indent=1)
# Nsight Compute on the gathering kernels of P = 5, both windows, 32 lanes.
for window in ['bspline', 'pswf']:
    case = f'{D}p0/{name}-{window}-5.bin'
    rep = f'{out}-ncu-{window}-5.txt'
    p = subprocess.run(['/usr/local/cuda-13.2/bin/ncu', '--kernel-name', 'regex:gather', '--launch-count', '6',
                        '--section', 'SpeedOfLight', '--section', 'LaunchStats', '--section', 'Occupancy',
                        '--section', 'MemoryWorkloadAnalysis', bench, case, 'f', '3', 'tiled4', '32', 'sorted', '-'],
                       capture_output=True, text=True, env=env)
    open(rep, 'w').write(p.stdout + p.stderr)
    print('ncu', window, p.returncode, flush=True)
sys.exit(1 if fail else 0)
