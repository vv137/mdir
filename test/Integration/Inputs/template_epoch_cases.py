"""Fresh segmented lists must preserve function-level constraint history.

Reuse the native fixture's independent rational oracle. An identity tuple
writer stands for a completed constraint event: this tests provenance,
not a constraint algorithm. A marked writer precedes a fresh allocation;
an unmarked writer must not change the initial-input widening policy.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('case', choices=['initial-fresh', 'constrained-fresh', 'within-fresh'])
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

def replace(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)

replace('md.particle_set @atoms', '''md.particle_set @atoms
memref.global "private" constant @constraint : memref<3xi32> = dense<[0, 1, 2]>''')
replace('func.func @evaluate(%types:', 'func.func @evaluate(%constraint: memref<?xi32, 1>, %types:')
replace('  %parts = arith.constant 1024 : index', '''  %parts = arith.constant 1024 : index
  %three = arith.constant 3 : index
  %constraint = gpu.alloc(%three) : memref<?xi32, 1>
  %constraint_host = memref.get_global @constraint : memref<3xi32>
  %constraint_view = memref.cast %constraint_host : memref<3xi32> to memref<?xi32>
  %constraint_ready = gpu.wait async
  %constraint_copy = gpu.memcpy async [%constraint_ready] %constraint, %constraint_view : memref<?xi32, 1>, memref<?xi32>
  gpu.wait [%constraint_copy]''')
replace('func.call @evaluate(%types,', 'func.call @evaluate(%constraint, %types,')
replace('      : (memref<?xi32, 1>, memref<?x4xi32>,',
        '      : (memref<?xi32, 1>, memref<?xi32, 1>, memref<?x4xi32>,')
mark = ' {md.template_constraint_epoch}' if a.case != 'initial-fresh' else ''
event = '''  md_exec.tuple_for %constraint, %x, %cell1 coordinates(displacement(1, 0))
      ins(%x : memref<?x3xf64, 1>) outs(%x : memref<?x3xf64, 1>)
      arity(3) overwrite [true] disjoint {
  ^bb0(%d: vector<3xf64>, %p0: vector<3xf64>, %p1: vector<3xf64>, %p2: vector<3xf64>):
    md_exec.yield %p0, %p1, %p2 : vector<3xf64>, vector<3xf64>, vector<3xf64>
  }''' + mark + ''' : memref<?xi32, 1>, memref<?x3xf64, 1>
  %fresh = md_exec.empty_segmented_pairs %types, %members
      cutoff(1.0) skin(0.2) extent(0.2) alpha(0.75)
      : memref<?xi32, 1>, memref<?x4xi32> -> !list
'''
# The event must happen after the initial widened evaluation, but before
# the next writer and the new allocation's first refresh.
needle = '  md_exec.particle_for ins(%x, %weights'
assert s.count(needle) == 6
s = s.replace(needle, event + needle, 1)
replace('%r1 = md_exec.refresh_segmented_pairs %r0',
        '%r1 = md_exec.refresh_segmented_pairs %fresh')
if a.case == 'within-fresh':
    assert s.count('extent(0.2)') == 16
    s = s.replace('extent(0.2)', 'extent(1.0)')
# Only the seven independent energy values are this fixture's numerical
# contract; the base fixture separately checks all epoch metadata.
for slot in [4, 5, 6]:
    replace(f'  func.call @dump_epoch(%view{slot}) : (memref<?x3xf64, 1>) -> ()', '')
a.program.write_text(s)
a.reference.write_text(json.dumps(json.loads(a.reference.read_text())[:7]) + '\n')
