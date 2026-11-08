"""Forward mode in the arguments of a reciprocal sum
(D389, #726).

Appends to the program that tangent_pme.py writes the potential Psi along
directions of its arguments, with its forces: along lambda (a number that
the charges q = q0 + lambda w depend on), along q0 (a field, with the
direction w, the same change of the charges), and along lambda and the
constant eps of the pair term of @both. Psi along lambda is dE/dlambda
(D161), and -t . F_Psi is the derivative of the tangent of D379 in lambda:
the transpose identity. Each check prints the relative error and then 1 if
it is within the tolerance of the program.

  direction_pme.py IN.mlir OUT.mlir

  direction_pme.py IN.mlir OUT.mlir fd

With `fd`, OUT.mlir keeps of `main` the inputs and the three evaluations of
Psi alone, for a run under md-check-derivatives, which then checks
`direction` and `direction_forces` by central differences along the
directions. The energy and the forces are quadratic in the charges and
linear in eps, so that the differences are exact but for the rounding at
any step.
"""
import sys

BLOCK = """  // Forward mode: Psi along lambda (a number), along q0 (a field, with the
  // direction w, the same change of the charges), and along lambda and eps of @both.
  %one = arith.constant 1.0 : f64
  %mark = arith.constant -726.0 : f64
  %fc0 = arith.constant 0 : index
  %fc1 = arith.constant 1 : index
  %fc3 = arith.constant 3 : index
  %fcn = arith.constant 10 : index
  %fzero = arith.constant 0.0 : f64
  call @printF64(%mark) : (f64) -> ()
  call @printNewline() : () -> ()
  %deps = arith.constant 0.7 : f64
  %psi1, %fpsi1 = md.evaluate @mesh(%x, %cell, %q0, %w, %lam, %moduli, %one)
      request [direction, direction_forces] {directions = array<i64: 4>}
      : (!vec, !md.cell, !real, !real, f64, !grid, f64) -> (f64, !vec)
  %psi2, %fpsi2 = md.evaluate @mesh(%x, %cell, %q0, %w, %lam, %moduli, %w)
      request [direction, direction_forces] {directions = array<i64: 2>}
      : (!vec, !md.cell, !real, !real, f64, !grid, !real) -> (f64, !vec)
  %psi3, %fpsi3 = md.evaluate @both(%x, %cell, %q0, %w, %lam, %moduli, %eps, %one, %deps)
      request [direction, direction_forces] {directions = array<i64: 4, 6>}
      : (!vec, !md.cell, !real, !real, f64, !grid, f64, f64, f64) -> (f64, !vec)
  %tt = mdrt.to_buffer %t : !vec to memref<?x3xf64>
  %b1 = mdrt.to_buffer %fpsi1 : !vec to memref<?x3xf64>
  %b2 = mdrt.to_buffer %fpsi2 : !vec to memref<?x3xf64>
  %b3 = mdrt.to_buffer %fpsi3 : !vec to memref<?x3xf64>
  %r:3 = scf.for %i = %fc0 to %fcn step %fc1 iter_args(%a1 = %fzero, %a2 = %fzero, %a3 = %fzero) -> (f64, f64, f64) {
    %in:3 = scf.for %k = %fc0 to %fc3 step %fc1 iter_args(%e1 = %a1, %e2 = %a2, %e3 = %a3) -> (f64, f64, f64) {
      %tk = memref.load %tt[%i, %k] : memref<?x3xf64>
      %f1 = memref.load %b1[%i, %k] : memref<?x3xf64>
      %f2 = memref.load %b2[%i, %k] : memref<?x3xf64>
      %f3 = memref.load %b3[%i, %k] : memref<?x3xf64>
      %m1 = arith.mulf %f1, %tk : f64
      %m2 = arith.mulf %f2, %tk : f64
      %m3 = arith.mulf %f3, %tk : f64
      %s1 = arith.subf %e1, %m1 : f64
      %s2 = arith.subf %e2, %m2 : f64
      %s3 = arith.subf %e3, %m3 : f64
      scf.yield %s1, %s2, %s3 : f64, f64, f64
    }
    scf.yield %in#0, %in#1, %in#2 : f64, f64, f64
  }
  // Psi = dE/dlambda, and -t . F_Psi = dPhi/dlambda (the transpose identity).
  call @check(%psi1, %du) : (f64, f64) -> ()
  call @check(%r#0, %dp) : (f64, f64) -> ()
  call @check(%psi2, %du) : (f64, f64) -> ()
  call @check(%r#1, %dp) : (f64, f64) -> ()
  %le = arith.mulf %deps, %dpe : f64
  %lhs = arith.addf %dpb, %le : f64
  call @check(%r#2, %lhs) : (f64, f64) -> ()
"""

with open(sys.argv[1]) as stream:
    program = stream.read()
end = program.rindex("  return\n}")
block = BLOCK
if len(sys.argv) > 3 and sys.argv[3] == "fd":
    # The evaluations of Psi, without the comparisons; and none of the
    # program's own, whose tangents in the positions want a smaller step.
    block = BLOCK[:BLOCK.index("  %tt = mdrt.to_buffer")]
    end_inputs = program.index("  %u, %p, ")
    program = program[:end_inputs] + program[end:]
    end = end_inputs
with open(sys.argv[2], "w") as stream:
    stream.write(program[:end] + block + program[end:])
