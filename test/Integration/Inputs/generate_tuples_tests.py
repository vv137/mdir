"""Writes the tests of terms over tuples from tuples_reference.py:

  test/Integration/tuples.mlir, tuples-mixed.mlir, tuples-dynamics.mlir,
  tuples-nonbonded.mlir, tables.mlir, and the tests in test/Integration/GPU
  that run
  them on a device.

Run it from the root of the repository after a change of the reference:

  python3 test/Integration/Inputs/generate_tuples_tests.py

The environment variable MIXED sets the tolerance of the test in mixed
precision (1.0e-5 by default).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tuples_reference as ref  # noqa: E402

#===------------------------------------------------------------------------===
# Energy, forces, and virial: tuples.mlir and tuples-mixed.mlir
#===------------------------------------------------------------------------===

x = ref.place()
tuples = ref.topology()
energy, forces, virial = ref.evaluate(x, tuples)
worst = ref.check(x, tuples)
assert worst < 1e-6

def f(v):
    return repr(float(v))

def dense2(rows, fmt):
    return "[" + ", ".join("[" + ", ".join(fmt(c) for c in row) + "]" for row in rows) + "]"

def dense1(values):
    return "[" + ", ".join(f(v) for v in values) + "]"

globals_ = []
globals_.append("memref.global \"private\" constant @positions : memref<%dx3xf64> =\n    dense<%s>"
                % (len(x), dense2(x, f)))
names = ["bonds", "angles", "dihedrals"]
for kind, name in enumerate(names):
    members = [m for m, _ in tuples[kind]]
    params = [p for _, p in tuples[kind]]
    arity = len(members[0])
    globals_.append("memref.global \"private\" constant @%s_members : memref<%dx%dxi32> =\n    dense<%s>"
                    % (name, len(members), arity, dense2(members, str)))
    for j, suffix in enumerate(["k", "0"]):
        globals_.append("memref.global \"private\" constant @%s_%s : memref<%dxf64> =\n    dense<%s>"
                        % (name, suffix, len(params), dense1([p[j] for p in params])))

def gpu_wrapper(name, mode):
    precision = ""
    if mode:
        precision = '// RUN:     --md-exec-assign-precision="mode=%s" \\\n' % mode
    return """// The test of ../%s on a GPU.
//
// REQUIRES: cuda
//
// RUN: mdir-opt %%S/../%s %%md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %%md_exec_transforms \\
%s// RUN:     --md-exec-assign-storage="memory=device" --convert-md-exec-to-gpu \\
// RUN: | mlir-opt %%lower_gpu_to_llvm \\
// RUN: | mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%%mlir_c_runner_utils,%%mdrt,%%mdrt_cuda \\
// RUN: | FileCheck %%S/../%s
""" % (name, name, precision, name)

def header_mixed():
    return """// The test of tuples.mlir in mixed precision: the positions and the
// parameters are held in f64, and the kernels compute in f32. The reference
// values are those of tuples.mlir. This file is generated from
// Inputs/tuples_reference.py by Inputs/generate_tuples_tests.py.
//
// RUN: mdir-opt %s %md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_transforms \\
// RUN:     --md-exec-assign-precision="mode=mixed" \\
// RUN:     --md-exec-assign-storage --convert-md-exec-to-loops \\
// RUN: | mlir-opt %lower_loops_to_llvm \\
// RUN: | mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \\
// RUN: | FileCheck %s
//
// Each check prints 1 if the value agrees with the reference to a relative
// tolerance of TOLERANCE_TEXT, or 0 if it does not.
"""

def header(gpu):
    if gpu:
        run = '''// REQUIRES: cuda
//
// RUN: mdir-opt %s %md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_gpu_passes \\
// RUN: | mlir-opt %lower_gpu_to_llvm \\
// RUN: | mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%mlir_c_runner_utils,%mlir_runner_utils,%mdrt,%mdrt_cuda \\
// RUN: | FileCheck %s
'''
        what = "on a device"
    else:
        run = '''// RUN: mdir-opt %s %md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \\
// RUN: | mlir-opt %lower_loops_to_llvm \\
// RUN: | mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt \\
// RUN: | FileCheck %s

// RUN: mdir-opt %s %md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_passes \\
// RUN: | mlir-opt %lower_loops_to_openmp \\
// RUN: | env OMP_NUM_THREADS=4 mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%openmp \\
// RUN: | FileCheck %s
'''
        what = "on the CPU"
    return '''// Energy, forces, and virial of bonds, angles, and dihedrals, compiled and
// run %s, compared with Inputs/tuples_reference.py, which evaluates the
// same terms from their definitions and checks its forces against finite
// differences. This file is generated from that script by
// Inputs/generate_tuples_tests.py.
//
%s
// The system: %d chains of %d particles in a periodic cube with the edge
// %s. Bonds k (r - r0)^2 / 2, angles k (cos(theta) - c0)^2 / 2, and
// dihedrals k (1 + cos(2 phi - phi0)), with parameters for each tuple.
//
// Each check prints 1 if the value agrees with the reference to a relative
// tolerance of 1e-10, or 0 if it does not.
''' % (what, run.rstrip("\n"), ref.CHAINS, ref.LENGTH, f(ref.EDGE))

BODY = '''
!vec       = !md.field<@atoms, 3 x f64>
!bonds     = !md.relation<@atoms, 2, unordered, @bonds>
!angles    = !md.relation<@atoms, 3, reversal, @angles>
!dihedrals = !md.relation<@atoms, 4, reversal, @dihedrals>
!of_bond     = !md.field<@bonds, f64>
!of_angle    = !md.field<@angles, f64>
!of_dihedral = !md.field<@dihedrals, f64>

md.particle_set @atoms
md.tuple_set @bonds on(@atoms) arity(2) orientation(unordered)
md.tuple_set @angles on(@atoms) arity(3) orientation(reversal)
md.tuple_set @dihedrals on(@atoms) arity(4) orientation(reversal)

md.potential @bonded(%x: !vec, %cell: !md.cell,
                     %bonds: !bonds, %kb: !of_bond, %r0: !of_bond,
                     %angles: !angles, %ka: !of_angle, %c0: !of_angle,
                     %dihedrals: !dihedrals, %kd: !of_dihedral,
                     %phi0: !of_dihedral) -> f64 {
  %ub = md.sum_tuples %bonds, %x, %cell coordinates(distance(0, 1))
          tuple(%kb, %r0 : !of_bond, !of_bond) {
  ^bb0(%r: f64, %k_t: f64, %r0_t: f64):
    %half = arith.constant 0.5 : f64
    %dr = arith.subf %r, %r0_t : f64
    %dr2 = arith.mulf %dr, %dr : f64
    %kh = arith.mulf %half, %k_t : f64
    %e = arith.mulf %kh, %dr2 : f64
    md.yield %e : f64
  } : !bonds, !vec -> f64

  %ua = md.sum_tuples %angles, %x, %cell coordinates(cosine(0, 1, 2))
          tuple(%ka, %c0 : !of_angle, !of_angle) {
  ^bb0(%c: f64, %k_t: f64, %c0_t: f64):
    %half = arith.constant 0.5 : f64
    %dc = arith.subf %c, %c0_t : f64
    %dc2 = arith.mulf %dc, %dc : f64
    %kh = arith.mulf %half, %k_t : f64
    %e = arith.mulf %kh, %dc2 : f64
    md.yield %e : f64
  } : !angles, !vec -> f64

  %ud = md.sum_tuples %dihedrals, %x, %cell coordinates(dihedral(0, 1, 2, 3))
          tuple(%kd, %phi0 : !of_dihedral, !of_dihedral) {
  ^bb0(%phi: f64, %k_t: f64, %phi0_t: f64):
    %one = arith.constant 1.0 : f64
    %two = arith.constant 2.0 : f64
    %p2 = arith.mulf %two, %phi : f64
    %a = arith.subf %p2, %phi0_t : f64
    %cos = math.cos %a : f64
    %s = arith.addf %one, %cos : f64
    %e = arith.mulf %k_t, %s : f64
    md.yield %e : f64
  } : !dihedrals, !vec -> f64

  %u0 = arith.addf %ub, %ua : f64
  %u = arith.addf %u0, %ud : f64
  md.return %u : f64
}

func.func private @printF64(f64)
func.func private @printNewline()

func.func @check(%value: f64, %reference: f64, %magnitude: f64) {
  %tolerance = arith.constant TOLERANCE : f64
  %difference = arith.subf %value, %reference : f64
  %error = math.absf %difference : f64
  %scale = math.absf %magnitude : f64
  %bound = arith.mulf %tolerance, %scale : f64
  %agrees = arith.cmpf ole, %error, %bound : f64
  %flag = arith.uitofp %agrees : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

GLOBALS

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c20 = arith.constant 20 : index

  %xs = memref.get_global @positions : memref<NATOMSx3xf64>
  %xd = memref.cast %xs : memref<NATOMSx3xf64> to memref<?x3xf64>
  %n = memref.dim %xd, %c0 : memref<?x3xf64>
  %buffer = memref.alloc(%n) : memref<?x3xf64>
  memref.copy %xd, %buffer : memref<?x3xf64> to memref<?x3xf64>
  %x = mdrt.from_buffer %buffer : memref<?x3xf64> to !vec

MEMBERS
  %edge = arith.constant EDGE : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge

  %u, %f, %w = md.evaluate @bonded(%x, %cell, %bonds, %bonds_k, %bonds_0,
                                   %angles, %angles_k, %angles_0,
                                   %dihedrals, %dihedrals_k, %dihedrals_0)
      request [energy, forces, virial]
      : (!vec, !md.cell, !bonds, !of_bond, !of_bond, !angles, !of_angle,
         !of_angle, !dihedrals, !of_dihedral, !of_dihedral)
        -> (f64, !vec, vector<9xf64>)
  %forces = mdrt.to_buffer %f : !vec to memref<?x3xf64>

  // Energy.
  // CHECK:      1
  %u_ref = arith.constant U_REF : f64
  call @check(%u, %u_ref, %u_ref) : (f64, f64, f64) -> ()

  // The force on particle 0, on the scale of the largest component.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  %f0_scale = arith.constant F0_SCALE : f64
  %f0x = memref.load %forces[%c0, %c0] : memref<?x3xf64>
  %f0y = memref.load %forces[%c0, %c1] : memref<?x3xf64>
  %f0z = memref.load %forces[%c0, %c2] : memref<?x3xf64>
  %f0x_ref = arith.constant F0X : f64
  %f0y_ref = arith.constant F0Y : f64
  %f0z_ref = arith.constant F0Z : f64
  call @check(%f0x, %f0x_ref, %f0_scale) : (f64, f64, f64) -> ()
  call @check(%f0y, %f0y_ref, %f0_scale) : (f64, f64, f64) -> ()
  call @check(%f0z, %f0z_ref, %f0_scale) : (f64, f64, f64) -> ()

  // The y component of the force on particle 20.
  // CHECK-NEXT: 1
  %f20y = memref.load %forces[%c20, %c1] : memref<?x3xf64>
  %f20y_ref = arith.constant F20Y : f64
  call @check(%f20y, %f20y_ref, %f20y_ref) : (f64, f64, f64) -> ()

  // The sum over all particles of the squared force.
  // CHECK-NEXT: 1
  %zero = arith.constant 0.0 : f64
  %f2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> (f64) {
    %fx = memref.load %forces[%i, %c0] : memref<?x3xf64>
    %fy = memref.load %forces[%i, %c1] : memref<?x3xf64>
    %fz = memref.load %forces[%i, %c2] : memref<?x3xf64>
    %xx = arith.mulf %fx, %fx : f64
    %yy = arith.mulf %fy, %fy : f64
    %zz = arith.mulf %fz, %fz : f64
    %xy = arith.addf %xx, %yy : f64
    %sq = arith.addf %xy, %zz : f64
    %next = arith.addf %acc, %sq : f64
    scf.yield %next : f64
  }
  %f2_ref = arith.constant F2_REF : f64
  call @check(%f2, %f2_ref, %f2_ref) : (f64, f64, f64) -> ()

  // The virial W = sum over the tuples and their members m of d_m0 (x) F_m,
  // on the scale of its largest component: every component.
  // CHECK-COUNT-9: 1
VIRIAL
  // CHECK-NOT: 0
  return
}
'''

members_ir = []
for kind, name in enumerate(names):
    count = len(tuples[kind])
    arity = len(tuples[kind][0][0])
    rel = {"bonds": "!bonds", "angles": "!angles", "dihedrals": "!dihedrals"}[name]
    fld = {"bonds": "!of_bond", "angles": "!of_angle", "dihedrals": "!of_dihedral"}[name]
    members_ir.append('''  %%%s_ms = memref.get_global @%s_members : memref<%dx%dxi32>
  %%%s_md = memref.cast %%%s_ms : memref<%dx%dxi32> to memref<?x%dxi32>
  %%%s = mdrt.from_buffer %%%s_md : memref<?x%dxi32> to %s''' % (
        name, name, count, arity, name, name, count, arity, arity, name, name, arity, rel))
    for suffix in ["k", "0"]:
        members_ir.append('''  %%%s_%s_s = memref.get_global @%s_%s : memref<%dxf64>
  %%%s_%s_d = memref.cast %%%s_%s_s : memref<%dxf64> to memref<?xf64>
  %%%s_%s = mdrt.from_buffer %%%s_%s_d : memref<?xf64> to %s''' % (
            name, suffix, name, suffix, count, name, suffix, name, suffix, count,
            name, suffix, name, suffix, fld))

scale9 = max(abs(v) for row in virial for v in row)
virial_ir = ["  %%w_scale = arith.constant %s : f64" % f(scale9)]
for a in range(3):
    for b in range(3):
        k = 3 * a + b
        virial_ir.append("  %%w%d = vector.extract %%w[%d] : f64 from vector<9xf64>" % (k, k))
        virial_ir.append("  %%w%d_ref = arith.constant %s : f64" % (k, f(virial[a][b])))
        virial_ir.append("  call @check(%%w%d, %%w%d_ref, %%w_scale) : (f64, f64, f64) -> ()" % (k, k))

f0 = forces[0]
body = BODY
for key, val in [("GLOBALS", "\n\n".join(globals_)), ("NATOMS", str(len(x))),
                 ("MEMBERS", "\n".join(members_ir)), ("EDGE", f(ref.EDGE)),
                 ("U_REF", f(sum(energy))), ("F0_SCALE", f(max(abs(c) for c in f0))),
                 ("F0X", f(f0[0])), ("F0Y", f(f0[1])), ("F0Z", f(f0[2])),
                 ("F20Y", f(forces[20][1])),
                 ("F2_REF", f(sum(c * c for v in forces for c in v))),
                 ("VIRIAL", "\n".join(virial_ir))]:
    body = body.replace(key, val)

mixed_tolerance = os.environ.get("MIXED", "1.0e-5")
double = body.replace("TOLERANCE", "1.0e-10")
open("test/Integration/tuples.mlir", "w").write(header(False) + double)
open("test/Integration/tuples-mixed.mlir", "w").write(
    header_mixed().replace("TOLERANCE_TEXT", mixed_tolerance)
    + body.replace("TOLERANCE", mixed_tolerance))
open("test/Integration/GPU/tuples.mlir", "w").write(gpu_wrapper("tuples.mlir", None))
open("test/Integration/GPU/tuples-mixed.mlir", "w").write(gpu_wrapper("tuples-mixed.mlir", "mixed"))

#===------------------------------------------------------------------------===
# Dynamics: tuples-dynamics.mlir
#===------------------------------------------------------------------------===

source = open("test/Integration/tuples.mlir").read()
types = source[source.index("!vec       ="):source.index("func.func private @printF64")]
globals_ = source[source.index('memref.global "private" constant @positions'):source.index("func.func @main()")]
members = source[source.index("  %bonds_ms ="):source.index("  %edge = arith.constant")]

x, v, u, k, start, worst = ref.integrate(ref.STEPS, ref.DT)
v0 = ref.velocities()
m = ref.masses()

def f(value):
    return repr(float(value))

def dense2(rows):
    return "[" + ", ".join("[" + ", ".join(f(c) for c in row) + "]" for row in rows) + "]"

text = '''// Chains of particles with bonds, angles, and dihedrals integrated for %(steps)d
// steps of %(dt)s with velocity Verlet, compiled and run, compared with the
// same integration in Inputs/tuples_reference.py. This file is generated
// from that script by Inputs/generate_tuples_tests.py.
//
// RUN: mdir-opt %%s %%md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %%md_exec_passes \\
// RUN: | mlir-opt %%lower_loops_to_llvm \\
// RUN: | mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%%mlir_c_runner_utils,%%mdrt \\
// RUN: | FileCheck %%s

// RUN: mdir-opt %%s %%md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %%md_exec_passes \\
// RUN: | mlir-opt %%lower_loops_to_openmp \\
// RUN: | env OMP_NUM_THREADS=4 mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%%mlir_c_runner_utils,%%mdrt,%%openmp \\
// RUN: | FileCheck %%s

// The system is that of tuples.mlir, with the masses 1, 1.5, and 2 in
// turn and velocities from a linear congruential generator. Each check
// prints 1 if the value agrees with the reference to a relative tolerance
// of 1e-9, or 0 if it does not. The total energy deviates from its start by
// at most %(worst).1e along the way, %(relative).1e of its value.

%(types)s
dyn.program @velocity_verlet(%%x: !vec, %%v: !vec, %%f: !vec, %%m: !md.field<@atoms, f64>,
                             %%cell: !md.cell, %%dt: f64,
                             %%bonds: !bonds, %%kb: !of_bond, %%r0: !of_bond,
                             %%angles: !angles, %%ka: !of_angle, %%c0: !of_angle,
                             %%dihedrals: !dihedrals, %%kd: !of_dihedral,
                             %%phi0: !of_dihedral) -> (!vec, !vec, !vec)
    attributes {provides = ["symplectic", "time_reversible"]} {
  %%c    = arith.constant 0.5 : f64
  %%half = arith.mulf %%c, %%dt : f64
  %%v1 = dyn.kick %%v, %%f, %%m, %%half : !vec
  %%x1 = dyn.drift %%x, %%v1, %%dt : !vec
  %%f1 = md.evaluate @bonded(%%x1, %%cell, %%bonds, %%kb, %%r0, %%angles, %%ka, %%c0,
                            %%dihedrals, %%kd, %%phi0) request [forces]
      : (!vec, !md.cell, !bonds, !of_bond, !of_bond, !angles, !of_angle,
         !of_angle, !dihedrals, !of_dihedral, !of_dihedral) -> !vec
  %%v2 = dyn.kick %%v1, %%f1, %%m, %%half : !vec
  dyn.return %%x1, %%v2, %%f1 : !vec, !vec, !vec
}

func.func private @printF64(f64)
func.func private @printNewline()

func.func @check(%%value: f64, %%reference: f64) {
  %%tolerance = arith.constant 1.0e-9 : f64
  %%difference = arith.subf %%value, %%reference : f64
  %%error = math.absf %%difference : f64
  %%scale = math.absf %%reference : f64
  %%bound = arith.mulf %%tolerance, %%scale : f64
  %%agrees = arith.cmpf ole, %%error, %%bound : f64
  %%flag = arith.uitofp %%agrees : i1 to f64
  call @printF64(%%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

%(globals)smemref.global "private" constant @velocities : memref<%(n)dx3xf64> =
    dense<%(v0)s>

memref.global "private" constant @masses : memref<%(n)dxf64> =
    dense<[%(m)s]>

func.func @main() {
  %%c0 = arith.constant 0 : index
  %%c1 = arith.constant 1 : index
  %%c2 = arith.constant 2 : index
  %%c20 = arith.constant 20 : index
  %%steps = arith.constant %(steps)d : index

  %%xs = memref.get_global @positions : memref<%(n)dx3xf64>
  %%xd = memref.cast %%xs : memref<%(n)dx3xf64> to memref<?x3xf64>
  %%n = memref.dim %%xd, %%c0 : memref<?x3xf64>
  %%positions = memref.alloc(%%n) : memref<?x3xf64>
  memref.copy %%xd, %%positions : memref<?x3xf64> to memref<?x3xf64>
  %%x0 = mdrt.from_buffer %%positions : memref<?x3xf64> to !vec

  %%vs = memref.get_global @velocities : memref<%(n)dx3xf64>
  %%vd = memref.cast %%vs : memref<%(n)dx3xf64> to memref<?x3xf64>
  %%velocities = memref.alloc(%%n) : memref<?x3xf64>
  memref.copy %%vd, %%velocities : memref<?x3xf64> to memref<?x3xf64>
  %%v0 = mdrt.from_buffer %%velocities : memref<?x3xf64> to !vec

  %%ms = memref.get_global @masses : memref<%(n)dxf64>
  %%md = memref.cast %%ms : memref<%(n)dxf64> to memref<?xf64>
  %%m = mdrt.from_buffer %%md : memref<?xf64> to !md.field<@atoms, f64>

%(members)s  %%edge = arith.constant %(edge)s : f64
  %%cell = md.orthorhombic_cell %%edge, %%edge, %%edge
  %%dt = arith.constant %(dt)s : f64

  %%f0 = md.evaluate @bonded(%%x0, %%cell, %%bonds, %%bonds_k, %%bonds_0,
                            %%angles, %%angles_k, %%angles_0,
                            %%dihedrals, %%dihedrals_k, %%dihedrals_0)
      request [forces]
      : (!vec, !md.cell, !bonds, !of_bond, !of_bond, !angles, !of_angle,
         !of_angle, !dihedrals, !of_dihedral, !of_dihedral) -> !vec

  %%x, %%v, %%f = scf.for %%s = %%c0 to %%steps step %%c1
      iter_args(%%xa = %%x0, %%va = %%v0, %%fa = %%f0) -> (!vec, !vec, !vec) {
    %%xb, %%vb, %%fb = dyn.step @velocity_verlet(
        %%xa, %%va, %%fa, %%m, %%cell, %%dt, %%bonds, %%bonds_k, %%bonds_0,
        %%angles, %%angles_k, %%angles_0, %%dihedrals, %%dihedrals_k, %%dihedrals_0)
        : (!vec, !vec, !vec, !md.field<@atoms, f64>, !md.cell, f64, !bonds,
           !of_bond, !of_bond, !angles, !of_angle, !of_angle, !dihedrals,
           !of_dihedral, !of_dihedral) -> (!vec, !vec, !vec)
    scf.yield %%xb, %%vb, %%fb : !vec, !vec, !vec
  }

  %%u = md.evaluate @bonded(%%x, %%cell, %%bonds, %%bonds_k, %%bonds_0,
                           %%angles, %%angles_k, %%angles_0,
                           %%dihedrals, %%dihedrals_k, %%dihedrals_0)
      request [energy]
      : (!vec, !md.cell, !bonds, !of_bond, !of_bond, !angles, !of_angle,
         !of_angle, !dihedrals, !of_dihedral, !of_dihedral) -> f64
  %%k = md.sum_particles gather(%%v, %%m : !vec, !md.field<@atoms, f64>) {
  ^bb0(%%v_i: vector<3xf64>, %%m_i: f64):
    %%half = arith.constant 0.5 : f64
    %%sq   = arith.mulf %%v_i, %%v_i : vector<3xf64>
    %%v2   = vector.reduction <add>, %%sq : vector<3xf64> into f64
    %%mv2  = arith.mulf %%m_i, %%v2 : f64
    %%ke   = arith.mulf %%half, %%mv2 : f64
    md.yield %%ke : f64
  } : f64

  // The potential and the kinetic energy at the end.
  // CHECK:      1
  // CHECK-NEXT: 1
  %%u_ref = arith.constant %(u)s : f64
  %%k_ref = arith.constant %(k)s : f64
  call @check(%%u, %%u_ref) : (f64, f64) -> ()
  call @check(%%k, %%k_ref) : (f64, f64) -> ()

  // The position of particle 0 and the velocity of particle 20.
  // CHECK-COUNT-6: 1
  %%xf = mdrt.to_buffer %%x : !vec to memref<?x3xf64>
  %%vf = mdrt.to_buffer %%v : !vec to memref<?x3xf64>
%(finals)s
  // CHECK-NOT: 0
  return
}
'''

finals = []
for name, buffer, row, values in [("x0", "xf", "c0", x[0]), ("v20", "vf", "c20", v[20])]:
    for c in range(3):
        finals.append("  %%%s_%d = memref.load %%%s[%%%s, %%c%d] : memref<?x3xf64>" % (name, c, buffer, row, c))
        finals.append("  %%%s_%d_ref = arith.constant %s : f64" % (name, c, f(values[c])))
        finals.append("  call @check(%%%s_%d, %%%s_%d_ref) : (f64, f64) -> ()" % (name, c, name, c))

out = text % {
    "steps": ref.STEPS, "dt": f(ref.DT), "worst": worst, "relative": worst / abs(start),
    "types": types.rstrip("\n") + "\n\n", "globals": globals_, "n": ref.COUNT,
    "v0": dense2(v0), "m": ", ".join(f(c) for c in m), "members": members,
    "edge": f(ref.EDGE), "u": f(u), "k": f(k), "finals": "\n".join(finals)}
open("test/Integration/tuples-dynamics.mlir", "w").write(out)
open("test/Integration/GPU/tuples-dynamics.mlir", "w").write('''// The test of ../tuples-dynamics.mlir on a GPU.
//
// REQUIRES: cuda
//
// RUN: mdir-opt %S/../tuples-dynamics.mlir %md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_gpu_passes \\
// RUN: | mlir-opt %lower_gpu_to_llvm \\
// RUN: | mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \\
// RUN: | FileCheck %S/../tuples-dynamics.mlir
''')

#===------------------------------------------------------------------------===
# Nonbonded terms with exclusions: tuples-nonbonded.mlir
#===------------------------------------------------------------------------===

def write_nonbonded():
    source = open("test/Integration/tuples.mlir").read()
    positions = source[source.index('memref.global "private" constant @positions'):
                       source.index('memref.global "private" constant @bonds_members')]

    x = ref.place()
    tuples = ref.topology()
    excluded = ref.excluded_pairs(tuples)
    pairs14 = ref.pairs14(tuples)
    energy, forces, virial = ref.nonbonded(x, tuples)

    def f(value):
        return repr(float(value))

    def dense_pairs(pairs):
        return "[" + ", ".join("[%d, %d]" % p for p in pairs) + "]"

    scale9 = max(abs(v) for row in virial for v in row)
    checks = []
    for a in range(3):
        for b in range(3):
            k = 3 * a + b
            checks.append("  %%w%d = vector.extract %%w[%d] : f64 from vector<9xf64>" % (k, k))
            checks.append("  %%w%d_ref = arith.constant %s : f64" % (k, f(virial[a][b])))
            checks.append("  call @check(%%w%d, %%w%d_ref, %%w_scale) : (f64, f64, f64) -> ()" % (k, k))
    f0 = forces[0]
    f0_scale = max(abs(c) for c in f0)
    f2 = sum(c * c for v in forces for c in v)

    text = """// Lennard-Jones between the particles of the chains of tuples.mlir, with
// the pairs one, two, and three bonds apart excluded from the neighborhood
// and the pairs three bonds apart added back with the factor %(scale14)s,
// compiled and run, compared with Inputs/tuples_reference.py. This file is
// generated from that script by Inputs/generate_tuples_tests.py.
//
// RUN: mdir-opt %%s %%md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %%md_exec_passes \\
// RUN: | mlir-opt %%lower_loops_to_llvm \\
// RUN: | mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%%mlir_c_runner_utils,%%mdrt \\
// RUN: | FileCheck %%s

// RUN: mdir-opt %%s %%md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %%md_exec_passes \\
// RUN: | mlir-opt %%lower_loops_to_openmp \\
// RUN: | env OMP_NUM_THREADS=4 mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%%mlir_c_runner_utils,%%mdrt,%%openmp \\
// RUN: | FileCheck %%s

// eps = %(eps)s, sigma = %(sigma)s, cutoff %(cutoff)s, with no truncation. Each
// check prints 1 if the value agrees with the reference to a relative
// tolerance of 1e-10, or 0 if it does not.

!vec      = !md.field<@atoms, 3 x f64>
!pairs    = !md.relation<@atoms, 2, unordered>
!excluded = !md.relation<@atoms, 2, unordered, @excluded>
!pairs14  = !md.relation<@atoms, 2, unordered, @pairs14>

md.particle_set @atoms
md.tuple_set @excluded on(@atoms) arity(2) orientation(unordered)
md.tuple_set @pairs14 on(@atoms) arity(2) orientation(unordered)

md.potential @nonbonded(%%x: !vec, %%cell: !md.cell, %%e: !excluded,
                        %%p: !pairs14, %%eps: f64, %%sigma: f64,
                        %%scale: f64) -> f64 {
  %%n = md.neighborhood %%x, %%cell cutoff(%(cutoff)s) exclude(%%e : !excluded)
         : !vec -> !pairs
  %%u = md.sum_relation %%n, %%x, %%cell exchange(symmetric) {
  ^bb0(%%r: f64, %%d: vector<3xf64>):
    %%c4  = arith.constant 4.0 : f64
    %%i6  = arith.constant 6 : i32
    %%sr  = arith.divf %%sigma, %%r : f64
    %%s6  = math.fpowi %%sr, %%i6 : f64, i32
    %%s12 = arith.mulf %%s6, %%s6 : f64
    %%t   = arith.subf %%s12, %%s6 : f64
    %%e4  = arith.mulf %%c4, %%eps : f64
    %%k   = arith.mulf %%e4, %%t : f64
    md.yield %%k : f64
  } : !pairs, !vec -> f64
  %%u14 = md.sum_tuples %%p, %%x, %%cell coordinates(distance(0, 1)) {
  ^bb0(%%r: f64):
    %%c4  = arith.constant 4.0 : f64
    %%i6  = arith.constant 6 : i32
    %%sr  = arith.divf %%sigma, %%r : f64
    %%s6  = math.fpowi %%sr, %%i6 : f64, i32
    %%s12 = arith.mulf %%s6, %%s6 : f64
    %%t   = arith.subf %%s12, %%s6 : f64
    %%e4  = arith.mulf %%c4, %%eps : f64
    %%k   = arith.mulf %%e4, %%t : f64
    %%ks  = arith.mulf %%scale, %%k : f64
    md.yield %%ks : f64
  } : !pairs14, !vec -> f64
  %%s = arith.addf %%u, %%u14 : f64
  md.return %%s : f64
}

func.func private @printF64(f64)
func.func private @printNewline()

func.func @check(%%value: f64, %%reference: f64, %%magnitude: f64) {
  %%tolerance = arith.constant 1.0e-10 : f64
  %%difference = arith.subf %%value, %%reference : f64
  %%error = math.absf %%difference : f64
  %%scale = math.absf %%magnitude : f64
  %%bound = arith.mulf %%tolerance, %%scale : f64
  %%agrees = arith.cmpf ole, %%error, %%bound : f64
  %%flag = arith.uitofp %%agrees : i1 to f64
  call @printF64(%%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

%(positions)smemref.global "private" constant @excluded_members : memref<%(ne)dx2xi32> =
    dense<%(excluded)s>

memref.global "private" constant @pairs14_members : memref<%(np)dx2xi32> =
    dense<%(pairs14)s>

func.func @main() {
  %%c0 = arith.constant 0 : index
  %%c1 = arith.constant 1 : index
  %%c2 = arith.constant 2 : index

  %%xs = memref.get_global @positions : memref<%(n)dx3xf64>
  %%xd = memref.cast %%xs : memref<%(n)dx3xf64> to memref<?x3xf64>
  %%n = memref.dim %%xd, %%c0 : memref<?x3xf64>
  %%buffer = memref.alloc(%%n) : memref<?x3xf64>
  memref.copy %%xd, %%buffer : memref<?x3xf64> to memref<?x3xf64>
  %%x = mdrt.from_buffer %%buffer : memref<?x3xf64> to !vec

  %%es = memref.get_global @excluded_members : memref<%(ne)dx2xi32>
  %%ed = memref.cast %%es : memref<%(ne)dx2xi32> to memref<?x2xi32>
  %%e = mdrt.from_buffer %%ed : memref<?x2xi32> to !excluded
  %%ps = memref.get_global @pairs14_members : memref<%(np)dx2xi32>
  %%pd = memref.cast %%ps : memref<%(np)dx2xi32> to memref<?x2xi32>
  %%p = mdrt.from_buffer %%pd : memref<?x2xi32> to !pairs14

  %%edge = arith.constant %(edge)s : f64
  %%cell = md.orthorhombic_cell %%edge, %%edge, %%edge
  %%eps = arith.constant %(eps)s : f64
  %%sigma = arith.constant %(sigma)s : f64
  %%scale = arith.constant %(scale14)s : f64

  %%u, %%f, %%w = md.evaluate @nonbonded(%%x, %%cell, %%e, %%p, %%eps, %%sigma, %%scale)
      request [energy, forces, virial]
      : (!vec, !md.cell, !excluded, !pairs14, f64, f64, f64)
        -> (f64, !vec, vector<9xf64>)
  %%forces = mdrt.to_buffer %%f : !vec to memref<?x3xf64>

  // Energy.
  // CHECK:      1
  %%u_ref = arith.constant %(u)s : f64
  call @check(%%u, %%u_ref, %%u_ref) : (f64, f64, f64) -> ()

  // The force on particle 0, on the scale of its largest component.
  // CHECK-COUNT-3: 1
  %%f0_scale = arith.constant %(f0_scale)s : f64
  %%f0x = memref.load %%forces[%%c0, %%c0] : memref<?x3xf64>
  %%f0y = memref.load %%forces[%%c0, %%c1] : memref<?x3xf64>
  %%f0z = memref.load %%forces[%%c0, %%c2] : memref<?x3xf64>
  %%f0x_ref = arith.constant %(f0x)s : f64
  %%f0y_ref = arith.constant %(f0y)s : f64
  %%f0z_ref = arith.constant %(f0z)s : f64
  call @check(%%f0x, %%f0x_ref, %%f0_scale) : (f64, f64, f64) -> ()
  call @check(%%f0y, %%f0y_ref, %%f0_scale) : (f64, f64, f64) -> ()
  call @check(%%f0z, %%f0z_ref, %%f0_scale) : (f64, f64, f64) -> ()

  // The sum over all particles of the squared force.
  // CHECK-NEXT: 1
  %%zero = arith.constant 0.0 : f64
  %%f2 = scf.for %%i = %%c0 to %%n step %%c1 iter_args(%%acc = %%zero) -> (f64) {
    %%fx = memref.load %%forces[%%i, %%c0] : memref<?x3xf64>
    %%fy = memref.load %%forces[%%i, %%c1] : memref<?x3xf64>
    %%fz = memref.load %%forces[%%i, %%c2] : memref<?x3xf64>
    %%xx = arith.mulf %%fx, %%fx : f64
    %%yy = arith.mulf %%fy, %%fy : f64
    %%zz = arith.mulf %%fz, %%fz : f64
    %%xy = arith.addf %%xx, %%yy : f64
    %%sq = arith.addf %%xy, %%zz : f64
    %%next = arith.addf %%acc, %%sq : f64
    scf.yield %%next : f64
  }
  %%f2_ref = arith.constant %(f2)s : f64
  call @check(%%f2, %%f2_ref, %%f2_ref) : (f64, f64, f64) -> ()

  // The virial, every component on the scale of the largest.
  // CHECK-COUNT-9: 1
  %%w_scale = arith.constant %(w_scale)s : f64
%(checks)s
  // CHECK-NOT: 0
  return
}
""" % {"scale14": f(ref.SCALE14), "eps": f(ref.EPSILON), "sigma": f(ref.SIGMA),
       "cutoff": f(ref.CUTOFF), "positions": positions, "n": ref.COUNT,
       "ne": len(excluded), "np": len(pairs14),
       "excluded": dense_pairs(excluded), "pairs14": dense_pairs(pairs14),
       "edge": f(ref.EDGE), "u": f(sum(energy)), "f0_scale": f(f0_scale),
       "f0x": f(f0[0]), "f0y": f(f0[1]), "f0z": f(f0[2]), "f2": f(f2),
       "w_scale": f(scale9), "checks": "\n".join(checks)}
    open("test/Integration/tuples-nonbonded.mlir", "w").write(text)
    open("test/Integration/GPU/tuples-nonbonded.mlir", "w").write(
        """// The test of ../tuples-nonbonded.mlir on a GPU.
//
// REQUIRES: cuda
//
// RUN: mdir-opt %S/../tuples-nonbonded.mlir %md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_gpu_passes \\
// RUN: | mlir-opt %lower_gpu_to_llvm \\
// RUN: | mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \\
// RUN: | FileCheck %S/../tuples-nonbonded.mlir
""")



write_nonbonded()


#===------------------------------------------------------------------------===
# Tables of pairs of types: tables.mlir
#===------------------------------------------------------------------------===


def write_tables():
    source = open("test/Integration/tuples.mlir").read()
    positions = source[source.index('memref.global "private" constant @positions'):
                       source.index('memref.global "private" constant @bonds_members')]
    x = ref.place()
    energy, forces, virial = ref.tabulated(x)
    sigma, epsilon = ref.pair_tables()
    n = len(sigma)

    def f(value):
        return repr(float(value))

    def matrix(rows):
        return "[" + ", ".join("[" + ", ".join(f(c) for c in row) + "]" for row in rows) + "]"

    scale9 = max(abs(v) for row in virial for v in row)
    checks = []
    for a in range(3):
        for b in range(3):
            k = 3 * a + b
            checks.append("  %%w%d = vector.extract %%w[%d] : f64 from vector<9xf64>" % (k, k))
            checks.append("  %%w%d_ref = arith.constant %s : f64" % (k, f(virial[a][b])))
            checks.append("  call @check(%%w%d, %%w%d_ref, %%w_scale) : (f64, f64, f64) -> ()" % (k, k))
    f0 = forces[0]
    f2 = sum(c * c for v in forces for c in v)

    text = """// Lennard-Jones with sigma and epsilon from tables of pairs of types, one
// pair of which the mixing rule does not give (NBFIX), and Coulomb with the
// product of the charges, both cut at %(cutoff)s nm with no shift, compiled
// and run, compared with Inputs/tuples_reference.py. The kernel is proved
// symmetric: the lookups are in symmetric tables. This file is generated by
// Inputs/generate_tuples_tests.py.
//
// RUN: mdir-opt %%s %%md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %%md_exec_passes \\
// RUN: | mlir-opt %%lower_loops_to_llvm \\
// RUN: | mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%%mlir_c_runner_utils,%%mdrt \\
// RUN: | FileCheck %%s

// RUN: mdir-opt %%s %%md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %%md_exec_passes \\
// RUN: | mlir-opt %%lower_loops_to_openmp \\
// RUN: | env OMP_NUM_THREADS=4 mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%%mlir_c_runner_utils,%%mdrt,%%openmp \\
// RUN: | FileCheck %%s

// In mixed precision, with the tolerance of the checks raised to 1e-5.
//
// RUN: sed 's/%%%%tolerance = arith.constant 1.0e-10/%%%%tolerance = arith.constant 1.0e-5/' %%s \\
// RUN: | mdir-opt %%md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %%md_exec_transforms \\
// RUN:     --md-exec-assign-precision="mode=mixed" \\
// RUN:     --md-exec-assign-storage --convert-md-exec-to-loops \\
// RUN: | mlir-opt %%lower_loops_to_llvm \\
// RUN: | mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%%mlir_c_runner_utils,%%mdrt \\
// RUN: | FileCheck %%s

// Each check prints 1 if the value agrees with the reference to a relative
// tolerance of 1e-10, or 0 if it does not.

!vec   = !md.field<@atoms, 3 x f64>
!real  = !md.field<@atoms, f64>
!kinds = !md.field<@atoms, i32>
!pairs = !md.relation<@atoms, 2, unordered>
!table = !md.table<2, f64, symmetric>

md.particle_set @atoms

md.potential @tabulated(%%x: !vec, %%cell: !md.cell, %%type: !kinds, %%q: !real,
                        %%sigma: !table, %%epsilon: !table, %%f: f64) -> f64 {
  %%n = md.neighborhood %%x, %%cell cutoff(%(cutoff)s) : !vec -> !pairs
  %%u = md.sum_relation %%n, %%x, %%cell gather(%%type, %%q : !kinds, !real)
         exchange(symmetric) {
  ^bb0(%%r: f64, %%d: vector<3xf64>, %%t_i: i32, %%t_j: i32, %%q_i: f64, %%q_j: f64):
    %%s   = md.lookup %%sigma[%%t_i, %%t_j] : !table, i32, i32 -> f64
    %%e   = md.lookup %%epsilon[%%t_i, %%t_j] : !table, i32, i32 -> f64
    %%c4  = arith.constant 4.0 : f64
    %%i6  = arith.constant 6 : i32
    %%sr  = arith.divf %%s, %%r : f64
    %%s6  = math.fpowi %%sr, %%i6 : f64, i32
    %%s12 = arith.mulf %%s6, %%s6 : f64
    %%t   = arith.subf %%s12, %%s6 : f64
    %%e4  = arith.mulf %%c4, %%e : f64
    %%lj  = arith.mulf %%e4, %%t : f64
    %%qq  = arith.mulf %%q_i, %%q_j : f64
    %%fqq = arith.mulf %%f, %%qq : f64
    %%c   = arith.divf %%fqq, %%r : f64
    %%k   = arith.addf %%lj, %%c : f64
    md.yield %%k : f64
  } : !pairs, !vec -> f64
  md.return %%u : f64
}

func.func private @printF64(f64)
func.func private @printNewline()

func.func @check(%%value: f64, %%reference: f64, %%magnitude: f64) {
  %%tolerance = arith.constant 1.0e-10 : f64
  %%difference = arith.subf %%value, %%reference : f64
  %%error = math.absf %%difference : f64
  %%scale = math.absf %%magnitude : f64
  %%bound = arith.mulf %%tolerance, %%scale : f64
  %%agrees = arith.cmpf ole, %%error, %%bound : f64
  %%flag = arith.uitofp %%agrees : i1 to f64
  call @printF64(%%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

%(positions)smemref.global "private" constant @types : memref<%(count)dxi32> =
    dense<[%(types)s]>

memref.global "private" constant @charges : memref<%(count)dxf64> =
    dense<[%(charges)s]>

memref.global "private" constant @sigma_table : memref<%(n)dx%(n)dxf64> =
    dense<%(sigma)s>

memref.global "private" constant @epsilon_table : memref<%(n)dx%(n)dxf64> =
    dense<%(epsilon)s>

func.func @main() {
  %%c0 = arith.constant 0 : index
  %%c1 = arith.constant 1 : index
  %%c2 = arith.constant 2 : index

  %%xs = memref.get_global @positions : memref<%(count)dx3xf64>
  %%xd = memref.cast %%xs : memref<%(count)dx3xf64> to memref<?x3xf64>
  %%n = memref.dim %%xd, %%c0 : memref<?x3xf64>
  %%buffer = memref.alloc(%%n) : memref<?x3xf64>
  memref.copy %%xd, %%buffer : memref<?x3xf64> to memref<?x3xf64>
  %%x = mdrt.from_buffer %%buffer : memref<?x3xf64> to !vec

  %%ts = memref.get_global @types : memref<%(count)dxi32>
  %%td = memref.cast %%ts : memref<%(count)dxi32> to memref<?xi32>
  %%type = mdrt.from_buffer %%td : memref<?xi32> to !kinds
  %%qs = memref.get_global @charges : memref<%(count)dxf64>
  %%qd = memref.cast %%qs : memref<%(count)dxf64> to memref<?xf64>
  %%q = mdrt.from_buffer %%qd : memref<?xf64> to !real
  %%ss = memref.get_global @sigma_table : memref<%(n)dx%(n)dxf64>
  %%sd = memref.cast %%ss : memref<%(n)dx%(n)dxf64> to memref<?x?xf64>
  %%sigma = mdrt.from_buffer %%sd : memref<?x?xf64> to !table
  %%es = memref.get_global @epsilon_table : memref<%(n)dx%(n)dxf64>
  %%ed = memref.cast %%es : memref<%(n)dx%(n)dxf64> to memref<?x?xf64>
  %%epsilon = mdrt.from_buffer %%ed : memref<?x?xf64> to !table

  %%edge = arith.constant %(edge)s : f64
  %%cell = md.orthorhombic_cell %%edge, %%edge, %%edge
  %%f = arith.constant %(coulomb)s : f64

  %%u, %%forces_field, %%w = md.evaluate @tabulated(%%x, %%cell, %%type, %%q, %%sigma, %%epsilon, %%f)
      request [energy, forces, virial]
      : (!vec, !md.cell, !kinds, !real, !table, !table, f64)
        -> (f64, !vec, vector<9xf64>)
  %%forces = mdrt.to_buffer %%forces_field : !vec to memref<?x3xf64>

  // Energy.
  // CHECK:      1
  %%u_ref = arith.constant %(u)s : f64
  call @check(%%u, %%u_ref, %%u_ref) : (f64, f64, f64) -> ()

  // The force on particle 0, on the scale of its largest component.
  // CHECK-COUNT-3: 1
  %%f0_scale = arith.constant %(f0_scale)s : f64
  %%f0x = memref.load %%forces[%%c0, %%c0] : memref<?x3xf64>
  %%f0y = memref.load %%forces[%%c0, %%c1] : memref<?x3xf64>
  %%f0z = memref.load %%forces[%%c0, %%c2] : memref<?x3xf64>
  %%f0x_ref = arith.constant %(f0x)s : f64
  %%f0y_ref = arith.constant %(f0y)s : f64
  %%f0z_ref = arith.constant %(f0z)s : f64
  call @check(%%f0x, %%f0x_ref, %%f0_scale) : (f64, f64, f64) -> ()
  call @check(%%f0y, %%f0y_ref, %%f0_scale) : (f64, f64, f64) -> ()
  call @check(%%f0z, %%f0z_ref, %%f0_scale) : (f64, f64, f64) -> ()

  // The sum over all particles of the squared force.
  // CHECK-NEXT: 1
  %%zero = arith.constant 0.0 : f64
  %%f2 = scf.for %%i = %%c0 to %%n step %%c1 iter_args(%%acc = %%zero) -> (f64) {
    %%fx = memref.load %%forces[%%i, %%c0] : memref<?x3xf64>
    %%fy = memref.load %%forces[%%i, %%c1] : memref<?x3xf64>
    %%fz = memref.load %%forces[%%i, %%c2] : memref<?x3xf64>
    %%xx = arith.mulf %%fx, %%fx : f64
    %%yy = arith.mulf %%fy, %%fy : f64
    %%zz = arith.mulf %%fz, %%fz : f64
    %%xy = arith.addf %%xx, %%yy : f64
    %%sq = arith.addf %%xy, %%zz : f64
    %%next = arith.addf %%acc, %%sq : f64
    scf.yield %%next : f64
  }
  %%f2_ref = arith.constant %(f2)s : f64
  call @check(%%f2, %%f2_ref, %%f2_ref) : (f64, f64, f64) -> ()

  // The virial, every component on the scale of the largest.
  // CHECK-COUNT-9: 1
  %%w_scale = arith.constant %(w_scale)s : f64
%(checks)s
  // CHECK-NOT: 0
  return
}
""" % {"cutoff": f(ref.CUTOFF), "positions": positions, "count": ref.COUNT,
       "types": ", ".join(str(t) for t in ref.particle_types()),
       "charges": ", ".join(f(q) for q in ref.charges()),
       "n": n, "sigma": matrix(sigma), "epsilon": matrix(epsilon),
       "edge": f(ref.EDGE), "coulomb": f(ref.COULOMB), "u": f(sum(energy)),
       "f0_scale": f(max(abs(c) for c in f0)), "f0x": f(f0[0]),
       "f0y": f(f0[1]), "f0z": f(f0[2]), "f2": f(f2),
       "w_scale": f(scale9), "checks": "\n".join(checks)}
    open("test/Integration/tables.mlir", "w").write(text)
    open("test/Integration/GPU/tables.mlir", "w").write(
        """// The test of ../tables.mlir on a GPU.
//
// REQUIRES: cuda
//
// RUN: mdir-opt %S/../tables.mlir %md_passes \\
// RUN:     --convert-md-to-md-exec="skin=0.2 width=64" %md_exec_gpu_passes \\
// RUN: | mlir-opt %lower_gpu_to_llvm \\
// RUN: | mlir-runner -e main --entry-point-result=void \\
// RUN:     --shared-libs=%mlir_c_runner_utils,%mdrt,%mdrt_cuda \\
// RUN: | FileCheck %S/../tables.mlir
""")


write_tables()
