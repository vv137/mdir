"""Put a migration boundary before the existing distributed LJ oracle.

The original positions remain the force oracle. Three earlier positions
cross a face, an edge, and a corner on their way back to that snapshot.
This is a state-transfer test, not an integration algorithm.
"""

from pathlib import Path
import sys


source = Path(sys.argv[1]).read_text().replace("GRID", sys.argv[2])

move = """
func.func @move_old(%x: memref<?x3xf64>) {
  %boundary = arith.constant 2.4 : f64
  %shift_left = arith.constant -0.4 : f64
  %shift_right = arith.constant 0.4 : f64
"""
for particle, axes in ((2, (0,)), (10, (0, 1)), (42, (0, 1, 2))):
    for axis in axes:
        stem = f"p{particle}a{axis}"
        move += f"  %{stem}i = arith.constant {particle} : index\n"
        move += f"  %{stem}k = arith.constant {axis} : index\n"
        move += f"  %{stem}x = memref.load %x[%{stem}i, %{stem}k] : memref<?x3xf64>\n"
        move += f"  %{stem}left = arith.cmpf olt, %{stem}x, %boundary : f64\n"
        move += f"  %{stem}delta = arith.select %{stem}left, %shift_right, %shift_left : f64\n"
        move += f"  %{stem}old = arith.addf %{stem}x, %{stem}delta : f64\n"
        move += f"  memref.store %{stem}old, %x[%{stem}i, %{stem}k] : memref<?x3xf64>\n"
move += "  return\n}\n\n"
move += """
func.func @check_exact_field(%a: memref<?x3xf64>, %b: memref<?x3xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %yes = arith.constant true
  %n = memref.dim %a, %c0 : memref<?x3xf64>
  %all = scf.for %i = %c0 to %n step %c1 iter_args(%ok = %yes) -> i1 {
    %row = scf.for %k = %c0 to %c3 step %c1 iter_args(%r = %ok) -> i1 {
      %va = memref.load %a[%i, %k] : memref<?x3xf64>
      %vb = memref.load %b[%i, %k] : memref<?x3xf64>
      %equal = arith.cmpf oeq, %va, %vb : f64
      %both = arith.andi %r, %equal : i1
      scf.yield %both : i1
    }
    scf.yield %row : i1
  }
  %flag = arith.uitofp %all : i1 to f64
  call @printF64(%flag) : (f64) -> ()
  call @printNewline() : () -> ()
  return
}

func.func @check_periodic(%a: memref<?x3xf64>, %b: memref<?x3xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %edge = arith.constant 4.8 : f64
  %half = arith.constant 0.5 : f64
  %zero = arith.constant 0.0 : f64
  %tol = arith.constant 1.0e-10 : f64
  %n = memref.dim %a, %c0 : memref<?x3xf64>
  %error = scf.for %i = %c0 to %n step %c1 iter_args(%e = %zero) -> f64 {
    %row = scf.for %k = %c0 to %c3 step %c1 iter_args(%r = %e) -> f64 {
      %va = memref.load %a[%i, %k] : memref<?x3xf64>
      %vb = memref.load %b[%i, %k] : memref<?x3xf64>
      %raw = arith.subf %va, %vb : f64
      %q = arith.divf %raw, %edge : f64
      %qh = arith.addf %q, %half : f64
      %whole = math.floor %qh : f64
      %shift = arith.mulf %whole, %edge : f64
      %delta = arith.subf %raw, %shift : f64
      %ad = math.absf %delta : f64
      %next = arith.maximumf %r, %ad : f64
      scf.yield %next : f64
    }
    scf.yield %row : f64
  }
  call @check(%error, %zero, %edge, %tol) : (f64, f64, f64, f64) -> ()
  return
}

"""

source = source.replace("func.func @main() {", move + "func.func @main() {", 1)
source = source.replace(
    "  memref.copy %buffer, %copy : memref<?x3xf64> to memref<?x3xf64>",
    "  memref.copy %buffer, %copy : memref<?x3xf64> to memref<?x3xf64>\n"
    "  call @move_old(%copy) : (memref<?x3xf64>) -> ()",
    1,
)
source = source.replace(
    "  %xs = mdrt.from_buffer %copy : memref<?x3xf64> to !vec\n"
    "  %lay = md_dist.partition %xs, %cell : !vec -> !md_dist.layout<@dd>\n"
    "  %xo = md_dist.distribute %xs, %lay image(%cell) : !vec -> !own",
    "  %oldx = mdrt.from_buffer %copy : memref<?x3xf64> to !vec\n"
    "  %xs = mdrt.from_buffer %buffer : memref<?x3xf64> to !vec\n"
    "  %oldlay = md_dist.partition %oldx, %cell : !vec -> !md_dist.layout<@dd>\n"
    "  %oldowned = md_dist.distribute %xs, %oldlay : !vec -> !own\n"
    "  %vold = md_dist.distribute %oldx, %oldlay : !vec -> !own\n"
    "  %lay1 = md_dist.migrate %oldlay, %oldowned, %cell : "
    "!md_dist.layout<@dd>, !own -> !md_dist.layout<@dd>\n"
    "  %x1 = md_dist.redistribute %oldowned, %oldlay, %lay1 image(%cell) "
    ": !own, !md_dist.layout<@dd>, !md_dist.layout<@dd> -> !own\n"
    "  %v1 = md_dist.redistribute %vold, %oldlay, %lay1 : "
    "!own, !md_dist.layout<@dd>, !md_dist.layout<@dd> -> !own\n"
    "  %back0 = md_dist.distribute %oldx, %lay1 : !vec -> !own\n"
    "  %lay2 = md_dist.migrate %lay1, %back0, %cell : "
    "!md_dist.layout<@dd>, !own -> !md_dist.layout<@dd>\n"
    "  %back1 = md_dist.redistribute %back0, %lay1, %lay2 image(%cell) "
    ": !own, !md_dist.layout<@dd>, !md_dist.layout<@dd> -> !own\n"
    "  %v2 = md_dist.redistribute %v1, %lay1, %lay2 : "
    "!own, !md_dist.layout<@dd>, !md_dist.layout<@dd> -> !own\n"
    "  %forward0 = md_dist.distribute %xs, %lay2 : !vec -> !own\n"
    "  %lay = md_dist.migrate %lay2, %forward0, %cell : "
    "!md_dist.layout<@dd>, !own -> !md_dist.layout<@dd>\n"
    "  %xo = md_dist.redistribute %forward0, %lay2, %lay image(%cell) "
    ": !own, !md_dist.layout<@dd>, !md_dist.layout<@dd> -> !own\n"
    "  %vnew = md_dist.redistribute %v2, %lay2, %lay : "
    "!own, !md_dist.layout<@dd>, !md_dist.layout<@dd> -> !own\n"
    "  %vglobal = md_dist.collect %vnew, %lay : !own -> !vec\n"
    "  %x1global = md_dist.collect %x1, %lay1 : !own -> !vec\n"
    "  %backglobal = md_dist.collect %back1, %lay2 : !own -> !vec",
    1,
)
source = source.replace(
    "  // The adjoint identity.\n  // CHECK-NEXT: 1\n"
    "  call @check(%lhs, %rhs, %rhs, %fine) : (f64, f64, f64, f64) -> ()",
    "  // The adjoint identity.\n  // CHECK-NEXT: 1\n"
    "  call @check(%lhs, %rhs, %rhs, %fine) : (f64, f64, f64, f64) -> ()\n"
    "  %vb = mdrt.to_buffer %vglobal : !vec to memref<?x3xf64>\n"
    "  // CHECK-NEXT: 1\n"
    "  call @check_exact_field(%vb, %copy) : "
    "(memref<?x3xf64>, memref<?x3xf64>) -> ()\n"
    "  %x1b = mdrt.to_buffer %x1global : !vec to memref<?x3xf64>\n"
    "  %backb = mdrt.to_buffer %backglobal : !vec to memref<?x3xf64>\n"
    "  // CHECK-NEXT: 1\n"
    "  call @check_periodic(%x1b, %buffer) : "
    "(memref<?x3xf64>, memref<?x3xf64>) -> ()\n"
    "  // CHECK-NEXT: 1\n"
    "  call @check_periodic(%backb, %copy) : "
    "(memref<?x3xf64>, memref<?x3xf64>) -> ()",
    1,
)
if "mixed" in sys.argv[3:]:
    source = source.replace("1.0e-10", "1.0e-5")
    begin = source.index("  // The adjoint identity <G")
    end = source.index("  // The reference, on one domain.", begin)
    source = source[:begin] + source[end:]
    source = source.replace(
        "  // The adjoint identity.\n  // CHECK-NEXT: 1\n"
        "  call @check(%lhs, %rhs, %rhs, %fine) : (f64, f64, f64, f64) -> ()\n",
        "",
    )
if "report" in sys.argv[3:]:
    needle = "  call @check_periodic(%backb, %copy) : (memref<?x3xf64>, memref<?x3xf64>) -> ()"
    assert needle in source
    source = source.replace(
        needle,
        needle
        + "\n  %migration_energy_difference = arith.subf %ud, %u : f64"
        + "\n  call @printF64(%migration_energy_difference) : (f64) -> ()"
        + "\n  call @printNewline() : () -> ()",
        1,
    )
print(source)
