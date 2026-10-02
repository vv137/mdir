#!/usr/bin/env python3
"""Compare the complete MD stage with/without the analytic one-bond solve.

This synthetic, force-free gas isolates the favorable one-bond topology;
its throughput is NOT a protein/water benchmark. Use --control for a real
system. GPU runs require an explicit single CUDA_VISIBLE_DEVICES value.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def prepare(work, side, steps, target, precision):
    count = side**3
    (work / 'dimers.top').write_text(f'''[ defaults ]
1 2 no 1.0 1.0
[ atomtypes ]
C 6 12.0 0 A 0.0 0.0
H 1 1.0 0 A 0.0 0.0
[ moleculetype ]
CH 1
[ atoms ]
1 C 1 CH C 1 0 12.0
2 H 1 CH H 1 0 1.0
[ bonds ]
1 2 1 0.109 284512.0
[ system ]
Noninteracting rigid diatomics
[ molecules ]
CH {count}
''')
    lines = ['Rigid diatomics', str(2*count)]
    for i in range(count):
        x, y, z = (i % side) * .8, ((i // side) % side) * .8, (i // side**2) * .8
        for j, atom in enumerate(['C', 'H']):
            lines.append(f'{(i+1)%100000:5d}{"CH":<5}{atom:>5}{(2*i+j+1)%100000:5d}{x+.2+.109*j:8.3f}{y+.2:8.3f}{z+.2:8.3f}')
    lines.append(' '.join([str(.8*side)]*3))
    (work / 'dimers.gro').write_text('\n'.join(lines)+'\n')
    return f'''[input]
topology = "{work / 'dimers.top'}"
coordinates = "{work / 'dimers.gro'}"
[output]
energy_interval = {max(1, steps//5)}
[energy]
cutoff = 2.0
pairlist_distance = 3.0
electrostatics = "CUTOFF"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.002
steps = {steps}
[ensemble]
ensemble = "NVE"
temperature = 100.0
[constraints]
hydrogen_bonds = true
[boundary]
type = "PERIODIC"
[execution]
target = "{target}"
precision = "{precision}"
threads = 1
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mdir', required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--control', type=Path)
    p.add_argument('--side', type=int, default=12)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--target', choices=['CPU', 'GPU'], default='CPU')
    p.add_argument('--precision', choices=['DOUBLE', 'MIXED', 'SINGLE'], default='DOUBLE')
    args = p.parse_args()
    if min(args.side, args.steps, args.repeat) < 1:
        p.error('side, steps and repeat must be positive')
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.control:
        source = args.control.resolve()
        text = source.read_text()
        # Keep input paths relative to the source; disable file output.
        # In particular, [input].checkpoint must not be confused with
        # [output].checkpoint, and its interval must be removed as well.
        lines, section = [], ''
        for line in text.splitlines():
            if line.strip().startswith('['):
                section = line.strip().split('#')[0].strip()
            key = line.split('=', 1)[0].strip()
            if section == '[input]' and key in ['topology', 'coordinates', 'checkpoint']:
                line = re.sub(r'"([^"]+)"', lambda m: f'"{source.parent / m[1]}"', line, count=1)
            if section == '[output]' and key in ['trajectory', 'checkpoint', 'trajectory_interval', 'checkpoint_interval']:
                continue
            lines.append(line)
        text = '\n'.join(lines) + '\n'
        text = re.sub(r'(?m)^(steps\s*=).*', rf'\g<1> {args.steps}', text)
        text = re.sub(r'(?m)^(energy_interval\s*=).*', rf'\g<1> {max(1,args.steps//5)}', text)
    else:
        text = prepare(work, args.side, args.steps, args.target, args.precision)
    if re.search(r'target\s*=\s*"gpu"', text, re.I):
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not visible or ',' in visible or visible == '-1':
            p.error('GPU benchmarks require one explicit CUDA_VISIBLE_DEVICES device')
    text = re.sub(r'(?m)^\s*analytic_bonds\s*=.*\n', '', text)
    if '[constraints]' not in text:
        p.error('control needs a [constraints] section')
    timings = {name: [] for name in ['newton', 'analytic']}
    for name in timings:
        content = text.replace('[constraints]', f'[constraints]\nanalytic_bonds = {str(name == "analytic").lower()}')
        (work / f'{name}.toml').write_text(content)
    for i in range(args.repeat):
        for name in (['newton', 'analytic'] if i % 2 == 0 else ['analytic', 'newton']):
            run = subprocess.run([args.mdir, 'run', str(work / f'{name}.toml')], cwd=work, capture_output=True, text=True)
            (work / f'{name}-{i}.log').write_text(run.stdout + run.stderr)
            if run.returncode:
                raise RuntimeError(run.stderr)
            matches = re.findall(r'MDIR: from step .*?, ([0-9.]+) ms per step', run.stdout)
            if not matches:
                raise RuntimeError('no steady-state timing in log')
            timings[name].append(float(matches[-1]))
            print(name, timings[name][-1], 'ms/step', flush=True)
    result = {'ms_per_step': timings,
              'median_speedup': statistics.median(timings['newton']) / statistics.median(timings['analytic']),
              'synthetic': not bool(args.control),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'note': 'Complete MD stage, excluding compilation; verify device contention separately.'}
    (work / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
