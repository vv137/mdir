"""Execute frozen forward/VJP bodies on the CPU, independent of tile size."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import struct

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('root', type=Path)
p.add_argument('scratch', type=Path)
p.add_argument('mdir_opt')
p.add_argument('mlir_opt')
p.add_argument('runner')
p.add_argument('runner_utils')
p.add_argument('openmp')
a = p.parse_args()
a.scratch.mkdir(parents=True, exist_ok=True)
results = []
for rows in [1, 64]:
    name = 'painn-unit-rows1.mlir' if rows == 1 else 'painn-scalar-export.mlir'
    source = a.root / 'test/Driver/Inputs/model-stages' / name
    lowered = subprocess.check_output(
        [a.mdir_opt, '-', '--md-exec-lower-stage-bodies=device=false'],
        input=('md.particle_set @atoms\n' + source.read_text()).encode()).decode()
    # Only the outlined buffer functions and their globals are needed here.
    start = min(match.start() for match in re.finditer(r'  (?:memref.global|func.func)', lowered))
    bodies = lowered[start:lowered.rfind('}')]
    bodies = re.sub(r' attributes \{md.stage_body, md.stage_rows = \d+ : i64\}', '', bodies)
    main = f'''
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func @main() {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %n = arith.constant {rows} : index
    %z = arith.constant 8 : i32
    %one = arith.constant 1.0 : f32
    %numbers = memref.alloc() : memref<{rows}xi32>
    %features = memref.alloc() : memref<{rows}x4xf32>
    %energy = memref.alloc() : memref<{rows}xf32>
    %seed = memref.alloc() : memref<{rows}xf32>
    %gradient = memref.alloc() : memref<{rows}x4xf32>
    scf.for %i = %c0 to %n step %c1 {{
      memref.store %z, %numbers[%i] : memref<{rows}xi32>
      memref.store %one, %seed[%i] : memref<{rows}xf32>
    }}
    func.call @md_stage_embed_f(%numbers, %features) : (memref<{rows}xi32>, memref<{rows}x4xf32>) -> ()
    func.call @md_stage_readout_f(%features, %energy) : (memref<{rows}x4xf32>, memref<{rows}xf32>) -> ()
    func.call @md_stage_readout_b(%features, %seed, %gradient) : (memref<{rows}x4xf32>, memref<{rows}xf32>, memref<{rows}x4xf32>) -> ()
    %e = memref.load %energy[%c0] : memref<{rows}xf32>
    %eb = arith.bitcast %e : f32 to i32
    %ef = arith.extui %eb : i32 to i64
    func.call @printI64(%ef) : (i64) -> ()
    func.call @printNewline() : () -> ()
    scf.for %j = %c0 to %c4 step %c1 {{
      %g = memref.load %gradient[%c0, %j] : memref<{rows}x4xf32>
      %gb = arith.bitcast %g : f32 to i32
      %gf = arith.extui %gb : i32 to i64
      func.call @printI64(%gf) : (i64) -> ()
      func.call @printNewline() : () -> ()
    }}
    memref.dealloc %numbers : memref<{rows}xi32>
    memref.dealloc %features : memref<{rows}x4xf32>
    memref.dealloc %energy : memref<{rows}xf32>
    memref.dealloc %seed : memref<{rows}xf32>
    memref.dealloc %gradient : memref<{rows}x4xf32>
    return
  }}
'''
    module = 'module {\n' + bodies + main + '}\n'
    path = a.scratch / f'rows{rows}.mlir'
    path.write_text(module)
    passes = ['--convert-scf-to-openmp', '--canonicalize', '--convert-scf-to-cf',
              '--convert-math-to-llvm', '--convert-math-to-libm', '--convert-vector-to-llvm',
              '--expand-strided-metadata', '--finalize-memref-to-llvm', '--convert-arith-to-llvm',
              '--convert-func-to-llvm', '--convert-cf-to-llvm', '--convert-openmp-to-llvm',
              '--reconcile-unrealized-casts']
    llvm = subprocess.check_output([a.mlir_opt, str(path), *passes])
    output = subprocess.check_output(
        [a.runner, '-e', 'main', '--entry-point-result=void',
         '--shared-libs=' + a.runner_utils + ',' + a.openmp],
        input=llvm, env=dict(os.environ, OMP_NUM_THREADS='2')).decode()
    values = [struct.unpack('<f', struct.pack('<I', int(value)))[0]
              for value in output.split()]
    assert len(values) == 5, output
    results.append(values)
energy_difference = abs(results[0][0] - results[1][0])
gradient_difference = max(abs(x-y) for x,y in zip(results[0][1:], results[1][1:]))
print(json.dumps(dict(cpu_energy=results[0][0], energy_difference=energy_difference,
                      feature_vjp_difference=gradient_difference, tolerance=1e-5)))
assert energy_difference <= 1e-5 and gradient_difference <= 1e-5
