// RUN: mdir-opt %s %md_exec_passes | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s --check-prefix=ANCHOR
// RUN: sed '/^ *anchors(/d' %s | mdir-opt %md_exec_passes | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s --check-prefix=SITE
// RUN: sed -e 's/0.2001/0.0001/g' %s | mdir-opt %md_exec_passes | mlir-opt %lower_loops_to_llvm \
// RUN: | mlir-runner -e main --entry-point-result=void --shared-libs=%mlir_c_runner_utils,%mdrt \
// RUN: | FileCheck %s --check-prefix=NO-RADIUS

// The unit at x = 0.70 has a bound of 0.2001 on its member's reach.
// The member swings from x = 0.50 to x = 0.90 while every anchor is still.
// The only unordered pairs within the cutoff of 1.0 on the second
// configuration are (0, 1), (1, 2), and (2, 3), at 0.20, 0.95, and 0.85.
// The periodic images give no additional pairs. The anchors of (1, 2)
// remain 1.15 apart, which is outside cutoff + skin without the radii.
// ANCHOR:      {{^3$}}
// ANCHOR-NEXT: {{^1$}}
// SITE:        {{^3$}}
// SITE-NEXT:   {{^2$}}
// NO-RADIUS:   {{^2$}}
// NO-RADIUS-NEXT: {{^1$}}

!vec = !md.field<@atoms, 3 x f64>
!radii = !md.field<@atoms, f64>
!members = !md.relation<@atoms, 2, ordered, @members>
!nl = !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
md.particle_set @atoms

memref.global "private" constant @old_x : memref<4x3xf64> =
  dense<[[0.70, 0.0, 0.0], [0.50, 0.0, 0.0], [1.85, 0.0, 0.0], [2.70, 0.0, 0.0]]>
memref.global "private" constant @new_x : memref<4x3xf64> =
  dense<[[0.70, 0.0, 0.0], [0.90, 0.0, 0.0], [1.85, 0.0, 0.0], [2.70, 0.0, 0.0]]>
memref.global "private" constant @r : memref<4xf64> = dense<[0.2001, -0.2001, 0.0, 0.0]>
memref.global "private" constant @tuples : memref<4x2xi32> = dense<[[0, 0], [1, 0], [2, 2], [3, 3]]>

func.func private @printF64(f64)
func.func private @printI64(i64)
func.func private @printNewline()

func.func @main() {
  %edge = arith.constant 4.0 : f64
  %none = arith.constant 0.0 : f64
  %cell = md.orthorhombic_cell %edge, %edge, %edge

  %old = memref.get_global @old_x : memref<4x3xf64>
  %new = memref.get_global @new_x : memref<4x3xf64>
  %radii = memref.get_global @r : memref<4xf64>
  %tuples = memref.get_global @tuples : memref<4x2xi32>
  %old_buffer = memref.cast %old : memref<4x3xf64> to memref<?x3xf64>
  %new_buffer = memref.cast %new : memref<4x3xf64> to memref<?x3xf64>
  %radius_buffer = memref.cast %radii : memref<4xf64> to memref<?xf64>
  %member_buffer = memref.cast %tuples : memref<4x2xi32> to memref<?x2xi32>
  %x0 = mdrt.from_buffer %old_buffer : memref<?x3xf64> to !vec
  %x1 = mdrt.from_buffer %new_buffer : memref<?x3xf64> to !vec
  %radius = mdrt.from_buffer %radius_buffer : memref<?xf64> to !radii
  %members = mdrt.from_buffer %member_buffer : memref<?x2xi32> to !members

  %empty = md_exec.empty_neighbors kind(matrix) width(2) : !nl
  %first = md_exec.refresh_neighbors %empty, %x0, %cell
      cutoff(1.0) skin(0.1) cell_width(1.1)
      anchors(%members, %radius, 0.2001 : !members, !radii)
      policy(check) : !nl, !vec
  %second = md_exec.refresh_neighbors %first, %x1, %cell
      cutoff(1.0) skin(0.1) cell_width(1.1)
      anchors(%members, %radius, 0.2001 : !members, !radii)
      policy(check) : !nl, !vec
  %pairs = md_exec.pair_for %second, %x1, %cell reduce(%none : f64)
      cutoff(1.0) weights [0.5] policy(directed, owner_only) {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    %one = arith.constant 1.0 : f64
    md_exec.yield %one : f64
  } : !nl, !vec -> f64

  %builds = md_exec.rebuild_count %second : !nl
  call @printF64(%pairs) : (f64) -> ()
  call @printNewline() : () -> ()
  call @printI64(%builds) : (i64) -> ()
  call @printNewline() : () -> ()
  return
}
