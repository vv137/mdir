"""Native list/writer/pruning energy against an exact all-images oracle.

Four unconstrained OHHM units have formed-site extent 3/4, exceeding the
declared 1/5. One O separation is 7/4: the initial nominal covering
radius would miss those interacting formed sites. The first build also
requires arena growth, which the fixture asserts. Later translations,
one full winding, and changing/restoring the cell exercise both epochs.
The independent reference enumerates formed-site pairs in rational
arithmetic. This is a quadratic integration fixture, not physical PME.
"""
import argparse
from fractions import Fraction as F
import json
import pathlib
import struct

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['generate', 'check'])
p.add_argument('program', type=pathlib.Path)
p.add_argument('reference', type=pathlib.Path)
p.add_argument('--triclinic', action='store_true')
a = p.parse_args()
if a.mode == 'check':
    got = [int(v) for v in a.program.read_text().split()]
    reference = json.loads(a.reference.read_text())
    assert got == reference, {'reported_bits': got, 'reference_bits': reference}
    print(f'native template epochs: 7 bitwise energies and {len(reference) - 7} exact epoch components; bound 0; initial widening, growth, motion, winding and cell changes')
    raise SystemExit(0)

positions = [[F(1, 8), F(0), F(0)], [F(5, 8), F(0), F(0)], [F(5, 8), F(0), F(0)],
             [F(15, 8), F(0), F(0)], [F(11, 8), F(0), F(0)], [F(11, 8), F(0), F(0)],
             [F(19, 8), F(0), F(0)], [F(23, 8), F(0), F(0)], [F(23, 8), F(0), F(0)],
             [F(29, 8), F(0), F(0)], [F(25, 8), F(0), F(0)], [F(25, 8), F(0), F(0)],
             [F(42), F(0), F(0)], [F(-42), F(0), F(0)],
             [F(84), F(0), F(0)], [F(-84), F(0), F(0)]]
phases = [(F(0), 4), (F(1, 64), 4), (F(1, 64), 4), (F(1, 8), 4),
          (F(4), 4), (F(0), 8), (F(0), 4)]
current = [r[:] for r in positions]
reference = []
for delta, edge in phases:
    for i in [0, 1, 2, 12]:
        current[i][0] += delta
    sites = []
    for oxygen in [0, 3, 6, 9]:
        sites.append([current[oxygen][k] + F(3, 4) *
                      (current[oxygen + 1][k] + current[oxygen + 2][k] - 2 * current[oxygen][k])
                      for k in range(3)])
    energy = F(0)
    for first in range(len(sites)):
        for second in range(first + 1, len(sites)):
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        image = [i, j, k]
                        matrix = [[F(edge), F(edge, 4) if a.triclinic else F(0), F(edge, 8) if a.triclinic else F(0)],
                                  [F(0), F(edge), F(edge, 4) if a.triclinic else F(0)],
                                  [F(0), F(0), F(edge)]]
                        shift = [sum(v * n for v, n in zip(row, image)) for row in matrix]
                        d = [x - y - z for x, y, z in zip(sites[first], sites[second], shift)]
                        squared = sum(v * v for v in d)
                        if squared < 1:
                            energy += squared
    reference.append(int.from_bytes(struct.pack('>d', float(energy)), 'big', signed=True))

# At the final restored cell every incident class has these published
# cells/reaches. Mirror the explicitly declared binary64 addition order,
# rather than accepting a decimal rendering of this internal ABI.
metadata = (([4., 4., 4., 1., .5, 1.] if a.triclinic else [4., 4., 4., 0., 0., 0.]) * 3) * 2
cutoff = 1. + (14.5 if a.triclinic else 12.) * 3.e-6
physical = [cutoff + 2. * .75, cutoff + .75, cutoff]
metadata += [r + .2 for r in physical] + [r + .0625 for r in physical]
reference += [int.from_bytes(struct.pack('>d', x), 'big', signed=True) for x in metadata]

lines = ['md.particle_set @atoms', '!list = !mdrt.segmented_pairs<@atoms>',
         'memref.global "private" constant @water : memref<4x4xi32> = dense<[[0, 1, 2, 12], [3, 4, 5, 13], [6, 7, 8, 14], [9, 10, 11, 15]]>',
         'memref.global "private" constant @positions : memref<16x3xf64> = dense<' +
         json.dumps([[float(v) for v in row] for row in positions]) + '>',
         'memref.global "private" constant @weights : memref<16xf64> = dense<[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]>',
         'func.func private @printI64(i64)', 'func.func private @printNewline()',
         'func.func private @mdrtDeviceActivationOpen() -> !llvm.ptr',
         'func.func private @mdrtDeviceActivationEnter(!llvm.ptr)',
         'func.func private @mdrtDeviceActivationClose(!llvm.ptr)',
         'func.func private @mdrtTemplatePairsCapacity(!list) -> i64',
         'func.func private @mdrtTemplatePairsFrameState(!list, i64) -> memref<?x3xf64, 1> attributes {llvm.emit_c_interface}',
         '''func.func @dump_epoch(%device: memref<?x3xf64, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %n = memref.dim %device, %c0 : memref<?x3xf64, 1>
  %host = memref.alloc(%n) : memref<?x3xf64>
  %ready = gpu.wait async
  %done = gpu.memcpy async [%ready] %host, %device : memref<?x3xf64>, memref<?x3xf64, 1>
  gpu.wait [%done]
  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %c3 step %c1 {
      %value = memref.load %host[%i, %j] : memref<?x3xf64>
      %bits = arith.bitcast %value : f64 to i64
      func.call @printI64(%bits) : (i64) -> ()
      func.call @printNewline() : () -> ()
    }
  }
  memref.dealloc %host : memref<?x3xf64>
  return
}''',
         '''func.func @evaluate(%types: memref<?xi32, 1>, %members: memref<?x4xi32>,
    %x: memref<?x3xf64, 1>, %weights: memref<?xf64, 1>,
    %sums: memref<?xf64, 1>, %partials: memref<?xf64, 1>) {
  %zero = arith.constant 0.0 : f64
  %list = md_exec.empty_segmented_pairs %types, %members
      cutoff(1.0) skin(0.2) extent(0.2) alpha(0.75)
      : memref<?xi32, 1>, memref<?x4xi32> -> !list
''']
previous = '%list'
for phase, (delta, edge) in enumerate(phases):
    lines.append(f'  %edge{phase} = arith.constant {float(edge)} : f64')
    if a.triclinic:
        lines.extend([f'  %bx{phase} = arith.constant {edge / 4.} : f64',
                      f'  %cx{phase} = arith.constant {edge / 8.} : f64',
                      f'  %cy{phase} = arith.constant {edge / 4.} : f64',
                      f'  %cell{phase} = md.triclinic_cell %edge{phase}, %edge{phase}, %edge{phase}, %bx{phase}, %cx{phase}, %cy{phase}'])
    else:
        lines.append(f'  %cell{phase} = md.orthorhombic_cell %edge{phase}, %edge{phase}, %edge{phase}')
    if phase:
        lines.append(f'''  md_exec.particle_for ins(%x, %weights : memref<?x3xf64, 1>, memref<?xf64, 1>)
      outs(%x : memref<?x3xf64, 1>) {{
  ^bb0(%old: vector<3xf64>, %weight: f64):
    %step = arith.constant dense<[{float(delta)}, 0.0, 0.0]> : vector<3xf64>
    %w = vector.broadcast %weight : f64 to vector<3xf64>
    %change = arith.mulf %step, %w : vector<3xf64>
    %next = arith.addf %old, %change : vector<3xf64>
    md_exec.yield %next : vector<3xf64>
  }}''')
    lines.append(f'''  %r{phase} = md_exec.refresh_segmented_pairs {previous} at(%x, %cell{phase})
      cutoff(1.0) skin(0.2) prune_skin(0.0625) extent(0.2) alpha(0.75)
      : !list, memref<?x3xf64, 1>''')
    if phase == 0:
        lines.append('''  %capacity = func.call @mdrtTemplatePairsCapacity(%r0) : (!list) -> i64
  %initial_capacity = arith.constant 4 : i64
  %grew = arith.cmpi ugt, %capacity, %initial_capacity : i64
  cf.assert %grew, "native fixture did not exercise automatic arena growth"''')
    lines.append(f'''  %energy{phase} = md_exec.template_for %r{phase}, %x, %cell{phase}
      reduce(%zero : f64) scratch(%sums, %partials : memref<?xf64, 1>, memref<?xf64, 1>)
      pairs [1, 0, 0] cutoff(1.0) extent(0.2) alpha(0.75)
      water {{
      ^bb0(%d: vector<3xf32>, %a1: vector<3xf32>, %a2: vector<3xf32>,
           %b1: vector<3xf32>, %b2: vector<3xf32>):
        %alpha = arith.constant dense<0.75> : vector<3xf32>
        %ha = arith.addf %a1, %a2 : vector<3xf32>
        %hb = arith.addf %b1, %b2 : vector<3xf32>
        %ma = arith.mulf %alpha, %ha : vector<3xf32>
        %mb = arith.mulf %alpha, %hb : vector<3xf32>
        %da = arith.addf %d, %ma : vector<3xf32>
        %dm = arith.subf %da, %mb : vector<3xf32>
        %square = arith.mulf %dm, %dm : vector<3xf32>
        %distance2 = vector.reduction <add>, %square : vector<3xf32> into f32
        %wide = arith.extf %distance2 : f32 to f64
        md_exec.yield %wide, %dm : f64, vector<3xf32>
      }} mixed {{
      ^bb0(%d: vector<3xf32>, %h1: vector<3xf32>, %h2: vector<3xf32>):
        md_exec.yield
      }} atoms {{
      ^bb0(%d: vector<3xf32>):
        md_exec.yield
      }} : !list, memref<?x3xf64, 1> -> f64
  %bits{phase} = arith.bitcast %energy{phase} : f64 to i64
  func.call @printI64(%bits{phase}) : (i64) -> ()
  func.call @printNewline() : () -> ()''')
    previous = f'%r{phase}'
for slot in [4, 5, 6]:
    lines.extend([f'  %slot{slot} = arith.constant {slot} : i64',
                  f'  %view{slot} = func.call @mdrtTemplatePairsFrameState({previous}, %slot{slot}) : (!list, i64) -> memref<?x3xf64, 1>',
                  f'  func.call @dump_epoch(%view{slot}) : (memref<?x3xf64, 1>) -> ()'])
lines.extend(['  return', '}', '''func.func @main() {
  %activation = func.call @mdrtDeviceActivationOpen() : () -> !llvm.ptr
  func.call @mdrtDeviceActivationEnter(%activation) : (!llvm.ptr) -> ()
  %n = arith.constant 16 : index
  %parts = arith.constant 1024 : index
  %types = gpu.alloc(%n) : memref<?xi32, 1>
  %x = gpu.alloc(%n) : memref<?x3xf64, 1>
  %w = gpu.alloc(%n) : memref<?xf64, 1>
  %sums = gpu.alloc(%n) : memref<?xf64, 1>
  %partials = gpu.alloc(%parts) : memref<?xf64, 1>
  %host = memref.get_global @positions : memref<16x3xf64>
  %hv = memref.cast %host : memref<16x3xf64> to memref<?x3xf64>
  %wh = memref.get_global @weights : memref<16xf64>
  %wv = memref.cast %wh : memref<16xf64> to memref<?xf64>
  %ready = gpu.wait async
  %copied = gpu.memcpy async [%ready] %x, %hv : memref<?x3xf64, 1>, memref<?x3xf64>
  %copiedw = gpu.memcpy async [%copied] %w, %wv : memref<?xf64, 1>, memref<?xf64>
  gpu.wait [%copiedw]
  %water = memref.get_global @water : memref<4x4xi32>
  %members = memref.cast %water : memref<4x4xi32> to memref<?x4xi32>
  func.call @evaluate(%types, %members, %x, %w, %sums, %partials)
      : (memref<?xi32, 1>, memref<?x4xi32>, memref<?x3xf64, 1>, memref<?xf64, 1>, memref<?xf64, 1>, memref<?xf64, 1>) -> ()
  func.call @mdrtDeviceActivationClose(%activation) : (!llvm.ptr) -> ()
  return
}'''])
a.program.write_text('\n'.join(lines) + '\n')
a.reference.write_text(json.dumps(reference) + '\n')
