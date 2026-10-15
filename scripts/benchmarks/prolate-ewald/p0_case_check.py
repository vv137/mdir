"""P0 (#337): the same tiled gathering (P = 5, 32 lanes, tiles of 4) on two
case files and two builds of the prototype, alternated in rounds, the first
dropped: which of the case (its D and its c1) or the build makes the
gathering of the earlier measurement take 253 µs.
p0_case_check.py ROUNDS OUT CASE... -- BINARY..."""
import json, os, statistics, subprocess, sys

rounds, out = int(sys.argv[1]), sys.argv[2]
split = sys.argv.index('--')
cases, binaries = sys.argv[3:split], sys.argv[split + 1:]
subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv'], check=True)
print('node', os.uname().nodename, flush=True)
res = {}
for r in range(rounds):
    for b in binaries:
        for c in cases:
            p = subprocess.run([b, c, 'f', '100', 'tiled4', '32', 'sorted', '-'], capture_output=True, text=True)
            if p.returncode != 0:
                print('skipped', b, c, p.stderr.strip(), flush=True)
                continue
            text = p.stdout
            for bad in ['-nan', 'nan', '-inf', 'inf']:
                text = text.replace(': ' + bad, ': 0').replace(bad + ',', '0,').replace(bad + ']', '0]')
            d = json.loads(text)
            key = f'{os.path.basename(b)} {os.path.basename(c)} D {d["D"]}'
            print(r, key, d['gather_us'], flush=True)
            if r > 0:
                res.setdefault(key, []).append(d['gather_us'])
summary = {k: (statistics.mean(v), statistics.stdev(v)) for k, v in res.items()}
json.dump(summary, open(out, 'w'), indent=1)
for k, v in summary.items():
    print(f'{k}: {v[0]:.1f} ± {v[1]:.1f} µs')
