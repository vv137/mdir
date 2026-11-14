// A list of groups written by hand for the serial definition of
// md_alg.traverse_pairs (D[md-alg-traverse]): one group of 16 places, three
// of them holding particles 0, 1, and 2, in a cubic cell of edge 4.
// Block 0 of the group's list holds three entries:
// - place 1, bit 0: the pair of places 0 and 1, shift 0;
// - place 2, bit 0: places 0 and 2, the entry moved by -1 cell in x, so
//   that x_2 = 3.9 is seen at -0.1;
// - place 2, bit 1: places 1 and 2, shift 0, r = 3.4, beyond the cutoff.
// The inner list of a dual list keeps the first entry alone.
//
// The mask of an entry: its bits of the places, then the cells of its
// shift in 4 bits an axis from bit 16, each the shift plus 4. The shift of
// a place: 10 bits an axis, each the shift plus 512.

!i = memref<?xi32>
memref.global "private" constant @order : memref<16xi32> =
    dense<[0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]>
memref.global "private" constant @shifts : memref<16xi32> = dense<537395712>
// 71565313 = 1 + 4 << 16 + 4 << 20 + 4 << 24; 71499777 has 3 << 16;
// 71565314 has bit 1.
memref.global "private" constant @entries : memref<64xi32> =
    dense<[1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
memref.global "private" constant @masks : memref<64xi32> =
    dense<[71565313, 71499777, 71565314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
memref.global "private" constant @counts : memref<1xi32> = dense<3>
memref.global "private" constant @zero : memref<1xi32> = dense<0>
memref.global "private" constant @one : memref<1xi32> = dense<1>
memref.global "private" constant @positions : memref<3x3xf64> =
    dense<[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.9, 0.0, 0.0]]>

func.func @get(%g: memref<16xi32>) -> !i {
  %d = memref.cast %g : memref<16xi32> to !i
  return %d : !i
}

func.func @forces() -> memref<?x3xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %f = memref.alloc(%c3) : memref<?x3xf64>
  // What the destination held, which the op overwrites.
  %seven = arith.constant 7.0 : f64
  scf.for %p = %c0 to %c3 step %c1 {
    scf.for %k = %c0 to %c3 step %c1 {
      memref.store %seven, %f[%p, %k] : memref<?x3xf64>
    }
  }
  return %f : memref<?x3xf64>
}

func.func private @printF64(f64)
func.func private @printNewline()

func.func @print(%f: memref<?x3xf64>, %u: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %p = %c0 to %c3 step %c1 {
    scf.for %k = %c0 to %c3 step %c1 {
      %v = memref.load %f[%p, %k] : memref<?x3xf64>
      func.call @printF64(%v) : (f64) -> ()
      func.call @printNewline() : () -> ()
    }
  }
  func.call @printF64(%u) : (f64) -> ()
  func.call @printNewline() : () -> ()
  return
}

func.func @main() {
  %o16 = memref.get_global @order : memref<16xi32>
  %s16 = memref.get_global @shifts : memref<16xi32>
  %e64 = memref.get_global @entries : memref<64xi32>
  %m64 = memref.get_global @masks : memref<64xi32>
  %c1g = memref.get_global @counts : memref<1xi32>
  %z1 = memref.get_global @zero : memref<1xi32>
  %one1 = memref.get_global @one : memref<1xi32>
  %x33 = memref.get_global @positions : memref<3x3xf64>
  %order = memref.cast %o16 : memref<16xi32> to !i
  %shifts = memref.cast %s16 : memref<16xi32> to !i
  %entries = memref.cast %e64 : memref<64xi32> to !i
  %masks = memref.cast %m64 : memref<64xi32> to !i
  %counts = memref.cast %c1g : memref<1xi32> to !i
  %units = memref.cast %z1 : memref<1xi32> to !i
  %ordinals = memref.cast %z1 : memref<1xi32> to !i
  %active = memref.cast %z1 : memref<1xi32> to !i
  %active_count = memref.cast %one1 : memref<1xi32> to !i
  %inner_counts = memref.cast %one1 : memref<1xi32> to !i
  %x = memref.cast %x33 : memref<3x3xf64> to memref<?x3xf64>
  %cell = arith.constant dense<4.0> : vector<3xf64>
  %places = arith.constant 16 : index
  %nunits = arith.constant 1 : index
  %u0 = arith.constant 0.0 : f64

  // The outer list.
  %f = func.call @forces() : () -> memref<?x3xf64>
  %u = md_alg.traverse_pairs
      lists(%entries, %masks, %counts, %units, %ordinals, %order, %shifts
            : !i, !i, !i, !i, !i, !i, !i) sizes(%places, %nunits)
      contract(!mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>)
      reach(1.5)
      positions(%x : memref<?x3xf64>) cell(%cell : vector<3xf64>)
      outs(%f : memref<?x3xf64>) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(1.0) weights [0.5] overwrite [true] {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_alg.yield %d, %r2 : vector<3xf64>, f64
  } -> f64
  func.call @print(%f, %u) : (memref<?x3xf64>, f64) -> ()

  // The inner list of a dual list, through its active units.
  %g = func.call @forces() : () -> memref<?x3xf64>
  %w = md_alg.traverse_pairs
      lists(%entries, %masks, %counts, %units, %ordinals, %order, %shifts,
            %entries, %masks, %inner_counts, %active, %active_count
            : !i, !i, !i, !i, !i, !i, !i, !i, !i, !i, !i, !i)
      sizes(%places, %nunits) inner
      contract(!mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>)
      reach(1.5)
      positions(%x : memref<?x3xf64>) cell(%cell : vector<3xf64>)
      outs(%g : memref<?x3xf64>) reduce(%u0 : f64)
      unique visit unordered combine atomic
      algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
      cutoff(1.0) weights [0.5] overwrite [true] {
  ^bb0(%r2: f64, %d: vector<3xf64>):
    md_alg.yield %d, %r2 : vector<3xf64>, f64
  } -> f64
  func.call @print(%g, %w) : (memref<?x3xf64>, f64) -> ()
  return
}
