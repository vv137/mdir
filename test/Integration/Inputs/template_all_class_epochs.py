"""All-class native coverage across prune/build thresholds and cell changes.

Reuse the native fixture's epoch schedule and launch plumbing, but add
two mobile solutes and independently enumerate every site pair and image.
The quadratic sum has only nonnegative dyadic terms and is checked by bits.
This exercises real search/pruning/consumption, not seeded work records.
"""
import argparse
from fractions import Fraction as F
import itertools
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('program', type=Path)
p.add_argument('reference', type=Path)
p.add_argument('--triclinic', action='store_true')
a = p.parse_args()
command = [sys.executable, str(Path(__file__).with_name('template_native_oracle.py')),
           'generate', str(a.program), str(a.reference)]
if a.triclinic:
    command.append('--triclinic')
subprocess.run(command, check=True)
s = a.program.read_text()
old = re.search(r'@positions : memref<16x3xf64> = dense<(.*)>', s).group(1)
x = [[F(v) for v in row] for row in json.loads(old)]
x += [[F(5, 4), F(1, 2), F(0)], [F(15, 8), F(1, 2), F(0)]]
new = json.dumps([[float(v) for v in row] for row in x])
s = s.replace('dense<' + old + '>', 'dense<' + new + '>')
weights = re.search(r'@weights : memref<16xf64> = dense<(.*)>', s).group(1)
s = s.replace('dense<' + weights + '>',
              'dense<' + json.dumps(json.loads(weights) + [1., 1.]) + '>')
s = s.replace('16x3xf64', '18x3xf64').replace('16xf64', '18xf64')
s = s.replace('%n = arith.constant 16 : index', '%n = arith.constant 18 : index')
s = s.replace('pairs [1, 0, 0]', 'pairs [1, 1, 1]')
mixed = '''        %alpha = arith.constant dense<0.75> : vector<3xf32>
        %hh = arith.addf %h1, %h2 : vector<3xf32>
        %site = arith.mulf %alpha, %hh : vector<3xf32>
        %dm = arith.subf %d, %site : vector<3xf32>
        %square = arith.mulf %dm, %dm : vector<3xf32>
        %distance2 = vector.reduction <add>, %square : vector<3xf32> into f32
        %wide = arith.extf %distance2 : f32 to f64
        md_exec.yield %wide, %dm : f64, vector<3xf32>'''
atoms = '''        %square = arith.mulf %d, %d : vector<3xf32>
        %distance2 = vector.reduction <add>, %square : vector<3xf32> into f32
        %wide = arith.extf %distance2 : f32 to f64
        md_exec.yield %wide, %d : f64, vector<3xf32>'''
s = s.replace('''%h2: vector<3xf32>):
        md_exec.yield''', '%h2: vector<3xf32>):\n' + mixed)
s = s.replace('''^bb0(%d: vector<3xf32>):
        md_exec.yield''', '^bb0(%d: vector<3xf32>):\n' + atoms)
assert s.count(mixed) == s.count(atoms) == 7
# The base fixture separately validates the published metadata layout.
for slot in [4, 5, 6]:
    s = s.replace(f'  func.call @dump_epoch(%view{slot}) : (memref<?x3xf64, 1>) -> ()', '')

reference = []
for phase, (delta, edge) in enumerate([(F(0), 4), (F(1, 64), 4), (F(1, 64), 4),
                                     (F(1, 8), 4), (F(4), 4), (F(0), 8), (F(0), 4)]):
    for i in [0, 1, 2, 12, 16, 17]:
        x[i][0] += delta
    sites = [[x[o][k] + F(3, 4) * (x[o+1][k] + x[o+2][k] - 2*x[o][k])
              for k in range(3)] for o in [0, 3, 6, 9]] + x[16:18]
    h = [[F(edge), F(edge, 4) if a.triclinic else F(0), F(edge, 8) if a.triclinic else F(0)],
         [F(0), F(edge), F(edge, 4) if a.triclinic else F(0)], [F(0), F(0), F(edge)]]
    energies, counts = [F(0)] * 3, [0] * 3
    for i, j in itertools.combinations(range(6), 2):
        tag = 0 if j < 4 else 1 if i < 4 else 2
        for image in itertools.product(range(-2, 3), repeat=3):
            shift = [sum(v*n for v, n in zip(row, image)) for row in h]
            d = [u-v-w for u, v, w in zip(sites[i], sites[j], shift)]
            r2 = sum(v*v for v in d)
            if r2 < 1:
                energies[tag] += r2
                counts[tag] += 1
    assert all(counts), (phase, counts)
    print(f'phase {phase}: independent WW/SW/SS counts {counts}; sums {energies}')
    reference.append(int.from_bytes(struct.pack('>d', float(sum(energies))), 'big', signed=True))
a.program.write_text(s)
a.reference.write_text(json.dumps(reference) + '\n')
