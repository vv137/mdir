"""D107 poisoned real particles, before a cold build or after a writer.

No numerical comparison is performed. The generated native list owns
four waters and two solutes; poison selection counts particles, not units.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('program', type=Path)
p.add_argument('phase', choices=['cold', 'writer'])
p.add_argument('member', choices=['oxygen', 'hydrogen', 'solute', 'two-hydrogens'])
p.add_argument('value', choices=['nan', 'infinity', 'huge'])
a = p.parse_args()
subprocess.run([sys.executable, str(Path(__file__).with_name('template_all_class_epochs.py')),
                str(a.program), str(a.program) + '.reference'], check=True,
               stdout=subprocess.DEVNULL)
s = a.program.read_text()
ids = {'oxygen': [0], 'hydrogen': [1], 'solute': [16], 'two-hydrogens': [1, 2]}[a.member]
value = {'nan': '0x7FF8000000000000', 'infinity': '0x7FF0000000000000', 'huge': '1.0e110'}[a.value]
if a.phase == 'cold':
    match = re.search(r'@positions : memref<18x3xf64> = dense<(.*)>', s)
    positions = json.loads(match.group(1))
    # Textual hexadecimal float constants preserve NaN/infinity exactly.
    rows = []
    for i, row in enumerate(positions):
        components = [repr(v) for v in row]
        if i in ids:
            components[1] = value
        rows.append('[' + ', '.join(components) + ']')
    s = s.replace(match.group(0), '@positions : memref<18x3xf64> = dense<[' + ', '.join(rows) + ']>')
else:
    mask = [int(i in ids) for i in range(18)]
    s = s.replace('md.particle_set @atoms', 'md.particle_set @atoms\n'
                  'memref.global "private" constant @poison_mask : memref<18xi32> = dense<' + json.dumps(mask) + '>')
    s = s.replace('func.func @evaluate(%types:', 'func.func @evaluate(%mask: memref<?xi32, 1>, %types:')
    s = s.replace('  %parts = arith.constant 1024 : index', '''  %parts = arith.constant 1024 : index
  %mask_host = memref.get_global @poison_mask : memref<18xi32>
  %mask_view = memref.cast %mask_host : memref<18xi32> to memref<?xi32>
  %mask = gpu.alloc(%n) : memref<?xi32, 1>
  %mask_ready = gpu.wait async
  %mask_copy = gpu.memcpy async [%mask_ready] %mask, %mask_view : memref<?xi32, 1>, memref<?xi32>
  gpu.wait [%mask_copy]''')
    s = s.replace('func.call @evaluate(%types,', 'func.call @evaluate(%mask, %types,')
    s = s.replace('      : (memref<?xi32, 1>, memref<?x4xi32>,',
                  '      : (memref<?xi32, 1>, memref<?xi32, 1>, memref<?x4xi32>,')
    s = s.replace('ins(%x, %weights : memref<?x3xf64, 1>, memref<?xf64, 1>)',
                  'ins(%x, %weights, %mask : memref<?x3xf64, 1>, memref<?xf64, 1>, memref<?xi32, 1>)', 1)
    s = s.replace('^bb0(%old: vector<3xf64>, %weight: f64):',
                  '^bb0(%old: vector<3xf64>, %weight: f64, %poison: i32):', 1)
    s = s.replace('    md_exec.yield %next : vector<3xf64>', f'''    %poison_zero = arith.constant 0 : i32
    %poisoned = arith.cmpi ne, %poison, %poison_zero : i32
    %bad = arith.constant dense<[0.0, {value}, 0.0]> : vector<3xf64>
    %result = arith.select %poisoned, %bad, %next : vector<3xf64>
    md_exec.yield %result : vector<3xf64>''', 1)
a.program.write_text(s)
