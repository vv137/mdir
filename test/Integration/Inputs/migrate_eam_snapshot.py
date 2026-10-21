"""Insert an earlier ownership snapshot before the distributed EAM force path.

The EAM one-domain oracle remains the original configuration. Only the
ownership epoch that feeds @derived is changed, after md-distribute has
formed its semantic primitives and before their physical lowering.
"""

from pathlib import Path
import sys


if sys.argv[1] == "source":
    source = Path(sys.argv[2]).read_text()
    if "mixed" in sys.argv[3:]:
        source = source.replace("1.0e-10", "1.0e-5")
        begin = source.index(
            "  // Central differences of the distributed energy in positions:"
        )
        end = source.index("  // The adjoint identity of the density:", begin)
        source = source[:begin] + source[end:]
    if "report" in sys.argv[3:]:
        needle = "  call @check(%ud, %u, %u, %tol) : (f64, f64, f64, f64) -> ()"
        assert needle in source
        source = source.replace(
            needle,
            needle
            + "\n  %migration_energy_difference = arith.subf %ud, %u : f64"
            + "\n  call @printF64(%migration_energy_difference) : (f64) -> ()"
            + "\n  call @printNewline() : () -> ()",
            1,
        )
    move = """
func.func @move_old(%x: memref<?x3xf64>) {
  %i = arith.constant 42 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %boundary = arith.constant 2.4 : f64
  %left = arith.constant -0.4 : f64
  %right = arith.constant 0.4 : f64
  scf.for %k = %c0 to %c3 step %c1 {
    %value = memref.load %x[%i, %k] : memref<?x3xf64>
    %low = arith.cmpf olt, %value, %boundary : f64
    %shift = arith.select %low, %right, %left : f64
    %old = arith.addf %value, %shift : f64
    memref.store %old, %x[%i, %k] : memref<?x3xf64>
  }
  return
}

"""
    assert "func.func @main() {" in source
    print(source.replace("func.func @main() {", move + "func.func @main() {", 1))
elif sys.argv[1] == "inject":
    source = sys.stdin.read()
    start = source.index("  func.func @derived(")
    end = source.index("  func.func @energy(", start)
    derived = source[start:end]
    old = (
        "    %3 = md_dist.partition %1, %2 : !md.field<@atoms, 3 x f64> "
        "-> !md_dist.layout<@eam.domains>\n"
        "    %4 = md_dist.distribute %1, %3 image(%2) : "
        "!md.field<@atoms, 3 x f64> -> "
        "!md_dist.field<@eam.domains, owned, 3 x f64>"
    )
    new = (
        "    %migration_buffer = call @copy(%0, %cst, %cst, %cst) : "
        "(memref<?x3xf64>, f64, f64, f64) -> memref<?x3xf64>\n"
        "    call @move_old(%migration_buffer) : (memref<?x3xf64>) -> ()\n"
        "    %migration_start = mdrt.from_buffer %migration_buffer : "
        "memref<?x3xf64> to !md.field<@atoms, 3 x f64>\n"
        "    %migration_old_layout = md_dist.partition %migration_start, %2 : "
        "!md.field<@atoms, 3 x f64> -> !md_dist.layout<@eam.domains>\n"
        "    %migration_old_field = md_dist.distribute %1, %migration_old_layout "
        ": !md.field<@atoms, 3 x f64> -> "
        "!md_dist.field<@eam.domains, owned, 3 x f64>\n"
        "    %3 = md_dist.migrate %migration_old_layout, %migration_old_field, %2 "
        ": !md_dist.layout<@eam.domains>, "
        "!md_dist.field<@eam.domains, owned, 3 x f64> "
        "-> !md_dist.layout<@eam.domains>\n"
        "    %4 = md_dist.redistribute %migration_old_field, "
        "%migration_old_layout, %3 image(%2) : "
        "!md_dist.field<@eam.domains, owned, 3 x f64>, "
        "!md_dist.layout<@eam.domains>, !md_dist.layout<@eam.domains> "
        "-> !md_dist.field<@eam.domains, owned, 3 x f64>"
    )
    assert old in derived, "the generated EAM partition contract changed"
    derived = derived.replace(old, new, 1)
    print(source[:start] + derived + source[end:])
else:
    raise SystemExit("expected source or inject")
