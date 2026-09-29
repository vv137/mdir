// Builds a neighbor matrix by binning the particles into cells.
//
// This is a template. The compiler adds it to a module that builds neighbor
// structures, where it is specialized like any other code.
//
// The simulation cell is orthorhombic and periodic, with edge lengths `box`.
// On return, row `i` of `index` holds the particles within `reach` of
// particle `i`, and `counts[i]` holds their number. A row holds at most as
// many particles as `index` is wide; `counts[i]` is not limited, so that the
// caller can tell that a row was too narrow. The result is the largest
// count.

func.func private @mdrt.cell_count(%length: f64, %width: f64) -> index {
  %c1 = arith.constant 1 : index
  %ratio = arith.divf %length, %width : f64
  %floor = math.floor %ratio : f64
  %wide = arith.fptosi %floor : f64 to i64
  %count = arith.index_cast %wide : i64 to index
  %result = arith.maxsi %count, %c1 : index
  return %result : index
}

// The cell that the coordinate `x` lies in, along a direction with `count`
// cells and the edge length `length`.
func.func private @mdrt.cell_coordinate(%x: f64, %length: f64, %count: index)
    -> index {
  %c1 = arith.constant 1 : index
  %images = arith.divf %x, %length : f64
  %whole = math.floor %images : f64
  %shift = arith.mulf %whole, %length : f64
  %wrapped = arith.subf %x, %shift : f64
  %fraction = arith.divf %wrapped, %length : f64
  %wide = arith.index_cast %count : index to i64
  %scale = arith.sitofp %wide : i64 to f64
  %position = arith.mulf %fraction, %scale : f64
  %integer = arith.fptosi %position : f64 to i64
  %coordinate = arith.index_cast %integer : i64 to index
  // Rounding can place a particle at the upper edge one cell too far.
  %last = arith.subi %count, %c1 : index
  %result = arith.minsi %coordinate, %last : index
  return %result : index
}

// One component of the minimum-image displacement. `inverse` is one over
// `length`. The number of images is a whole number, so the result is that
// of a division by `length`, except where `d` is within rounding of half
// of `length`, which is beyond every reach.
func.func private @mdrt.minimum_image(%d: f64, %length: f64, %inverse: f64)
    -> f64 {
  %images = arith.mulf %d, %inverse : f64
  %nearest = math.roundeven %images : f64
  %shift = arith.mulf %nearest, %length : f64
  %result = arith.subf %d, %shift : f64
  return %result : f64
}

func.func private @mdrt.build_neighbors_matrix(
    %x: memref<?x3xf64>, %box: vector<3xf64>, %reach: f64, %cell_width: f64,
    %counts: memref<?xi32>, %index: memref<?x?xi32>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %n = memref.dim %x, %c0 : memref<?x3xf64>
  %row_width = memref.dim %index, %c1 : memref<?x?xi32>
  %reach2 = arith.mulf %reach, %reach : f64

  %lx = vector.extract %box[0] : f64 from vector<3xf64>
  %ly = vector.extract %box[1] : f64 from vector<3xf64>
  %lz = vector.extract %box[2] : f64 from vector<3xf64>
  %unit = arith.constant 1.0 : f64
  %ilx = arith.divf %unit, %lx : f64
  %ily = arith.divf %unit, %ly : f64
  %ilz = arith.divf %unit, %lz : f64
  %nx = call @mdrt.cell_count(%lx, %cell_width) : (f64, f64) -> index
  %ny = call @mdrt.cell_count(%ly, %cell_width) : (f64, f64) -> index
  %nz = call @mdrt.cell_count(%lz, %cell_width) : (f64, f64) -> index
  %nxy = arith.muli %nx, %ny : index
  %cells = arith.muli %nxy, %nz : index
  %cells1 = arith.addi %cells, %c1 : index

  //===--------------------------------------------------------------------===//
  // Counting sort of the particles by cell
  //===--------------------------------------------------------------------===//

  %key = memref.alloc(%n) : memref<?xindex>
  %start = memref.alloc(%cells1) : memref<?xindex>
  %cursor = memref.alloc(%cells) : memref<?xindex>
  %order = memref.alloc(%n) : memref<?xindex>

  scf.for %c = %c0 to %cells1 step %c1 {
    memref.store %c0, %start[%c] : memref<?xindex>
  }

  scf.for %i = %c0 to %n step %c1 {
    %xi = memref.load %x[%i, %c0] : memref<?x3xf64>
    %yi = memref.load %x[%i, %c1] : memref<?x3xf64>
    %zi = memref.load %x[%i, %c2] : memref<?x3xf64>
    %cx = func.call @mdrt.cell_coordinate(%xi, %lx, %nx)
        : (f64, f64, index) -> index
    %cy = func.call @mdrt.cell_coordinate(%yi, %ly, %ny)
        : (f64, f64, index) -> index
    %cz = func.call @mdrt.cell_coordinate(%zi, %lz, %nz)
        : (f64, f64, index) -> index
    %zy = arith.muli %cz, %ny : index
    %row = arith.addi %zy, %cy : index
    %rows = arith.muli %row, %nx : index
    %k = arith.addi %rows, %cx : index
    memref.store %k, %key[%i] : memref<?xindex>

    %next = arith.addi %k, %c1 : index
    %old = memref.load %start[%next] : memref<?xindex>
    %new = arith.addi %old, %c1 : index
    memref.store %new, %start[%next] : memref<?xindex>
  }

  scf.for %c = %c0 to %cells step %c1 {
    %next = arith.addi %c, %c1 : index
    %before = memref.load %start[%c] : memref<?xindex>
    %here = memref.load %start[%next] : memref<?xindex>
    %sum = arith.addi %before, %here : index
    memref.store %sum, %start[%next] : memref<?xindex>
    memref.store %before, %cursor[%c] : memref<?xindex>
  }

  scf.for %i = %c0 to %n step %c1 {
    %k = memref.load %key[%i] : memref<?xindex>
    %p = memref.load %cursor[%k] : memref<?xindex>
    memref.store %i, %order[%p] : memref<?xindex>
    %q = arith.addi %p, %c1 : index
    memref.store %q, %cursor[%k] : memref<?xindex>
  }

  //===--------------------------------------------------------------------===//
  // Neighbors
  //===--------------------------------------------------------------------===//

  // With fewer than three cells along a direction, the cells before and
  // after a cell coincide. Visit each cell once: the number of offsets is
  // the number of cells, up to three.
  %span_x = arith.minsi %nx, %c3 : index
  %span_y = arith.minsi %ny, %c3 : index
  %span_z = arith.minsi %nz, %c3 : index

  // The first offset is -1 with three offsets and 0 otherwise. To stay in
  // unsigned arithmetic, add the number of cells before taking the
  // remainder.
  %three_x = arith.cmpi eq, %span_x, %c3 : index
  %three_y = arith.cmpi eq, %span_y, %c3 : index
  %three_z = arith.cmpi eq, %span_z, %c3 : index
  %nx1 = arith.subi %nx, %c1 : index
  %ny1 = arith.subi %ny, %c1 : index
  %nz1 = arith.subi %nz, %c1 : index
  %first_x = arith.select %three_x, %nx1, %c0 : index
  %first_y = arith.select %three_y, %ny1, %c0 : index
  %first_z = arith.select %three_z, %nz1, %c0 : index

  scf.parallel (%i) = (%c0) to (%n) step (%c1) {
    %xi = memref.load %x[%i, %c0] : memref<?x3xf64>
    %yi = memref.load %x[%i, %c1] : memref<?x3xf64>
    %zi = memref.load %x[%i, %c2] : memref<?x3xf64>

    %k = memref.load %key[%i] : memref<?xindex>
    %cx = arith.remui %k, %nx : index
    %rest = arith.divui %k, %nx : index
    %cy = arith.remui %rest, %ny : index
    %cz = arith.divui %rest, %ny : index

    %found = scf.for %oz = %c0 to %span_z step %c1
        iter_args(%count_z = %c0) -> (index) {
      %sz0 = arith.addi %cz, %first_z : index
      %sz1 = arith.addi %sz0, %oz : index
      %nz_cell = arith.remui %sz1, %nz : index

      %after_y = scf.for %oy = %c0 to %span_y step %c1
          iter_args(%count_y = %count_z) -> (index) {
        %sy0 = arith.addi %cy, %first_y : index
        %sy1 = arith.addi %sy0, %oy : index
        %ny_cell = arith.remui %sy1, %ny : index

        %after_x = scf.for %ox = %c0 to %span_x step %c1
            iter_args(%count_x = %count_y) -> (index) {
          %sx0 = arith.addi %cx, %first_x : index
          %sx1 = arith.addi %sx0, %ox : index
          %nx_cell = arith.remui %sx1, %nx : index

          %zy = arith.muli %nz_cell, %ny : index
          %row = arith.addi %zy, %ny_cell : index
          %rows = arith.muli %row, %nx : index
          %cell = arith.addi %rows, %nx_cell : index
          %cell1 = arith.addi %cell, %c1 : index
          %begin = memref.load %start[%cell] : memref<?xindex>
          %end = memref.load %start[%cell1] : memref<?xindex>

          %after_cell = scf.for %p = %begin to %end step %c1
              iter_args(%count = %count_x) -> (index) {
            %j = memref.load %order[%p] : memref<?xindex>
            %xj = memref.load %x[%j, %c0] : memref<?x3xf64>
            %yj = memref.load %x[%j, %c1] : memref<?x3xf64>
            %zj = memref.load %x[%j, %c2] : memref<?x3xf64>
            %dx0 = arith.subf %xi, %xj : f64
            %dy0 = arith.subf %yi, %yj : f64
            %dz0 = arith.subf %zi, %zj : f64
            %dx = func.call @mdrt.minimum_image(%dx0, %lx, %ilx)
                : (f64, f64, f64) -> f64
            %dy = func.call @mdrt.minimum_image(%dy0, %ly, %ily)
                : (f64, f64, f64) -> f64
            %dz = func.call @mdrt.minimum_image(%dz0, %lz, %ilz)
                : (f64, f64, f64) -> f64
            %dx2 = arith.mulf %dx, %dx : f64
            %dy2 = arith.mulf %dy, %dy : f64
            %dz2 = arith.mulf %dz, %dz : f64
            %dxy2 = arith.addf %dx2, %dy2 : f64
            %r2 = arith.addf %dxy2, %dz2 : f64

            %near = arith.cmpf olt, %r2, %reach2 : f64
            %other = arith.cmpi ne, %i, %j : index
            %neighbor = arith.andi %near, %other : i1

            %fits = arith.cmpi ult, %count, %row_width : index
            %keep = arith.andi %neighbor, %fits : i1
            scf.if %keep {
              %wide = arith.index_cast %j : index to i64
              %narrow = arith.trunci %wide : i64 to i32
              memref.store %narrow, %index[%i, %count] : memref<?x?xi32>
            }

            %more = arith.addi %count, %c1 : index
            %next = arith.select %neighbor, %more, %count : index
            scf.yield %next : index
          }
          scf.yield %after_cell : index
        }
        scf.yield %after_x : index
      }
      scf.yield %after_y : index
    }

    %wide = arith.index_cast %found : index to i64
    %narrow = arith.trunci %wide : i64 to i32
    memref.store %narrow, %counts[%i] : memref<?xi32>
  }

  memref.dealloc %key : memref<?xindex>
  memref.dealloc %start : memref<?xindex>
  memref.dealloc %cursor : memref<?xindex>
  memref.dealloc %order : memref<?xindex>

  // The largest count, and the counts limited to the width of a row.
  %narrow_width = arith.index_cast %row_width : index to i32
  %largest = scf.for %i = %c0 to %n step %c1
      iter_args(%max = %c0) -> (index) {
    %count = memref.load %counts[%i] : memref<?xi32>
    %limited = arith.minsi %count, %narrow_width : i32
    memref.store %limited, %counts[%i] : memref<?xi32>
    %wide = arith.index_cast %count : i32 to index
    %larger = arith.maxsi %max, %wide : index
    scf.yield %larger : index
  }
  return %largest : index
}
