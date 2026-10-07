// Builds a neighbor matrix by binning the particles into cells.
//
// This is a template. The compiler adds it to a module that builds neighbor
// structures, where it is specialized like any other code.
//
// Binning the particles into cells to find their neighbors is the method of
// linked cells [Quentrec1973]; a structure built with a skin and reused
// while no particle has moved more than half of it is the list of Verlet
// [Verlet1967]; displacements are taken in the minimum image
// [AllenTildesley2017]. The keys are those of docs/references.md.
//
// The simulation cell is orthorhombic and periodic, with edge lengths `box`.
// On return, row `i` of `index` holds the particles within `reach` of
// particle `i`, and `counts[i]` holds their number. A row holds at most as
// many particles as `index` is wide; `counts[i]` is not limited, so that the
// caller can tell that a row was too narrow. The result is the largest
// count.
//
// The cells are `cell_width` wide or a little wider. A particle has its
// neighbors in the cells that are within `reach` of its own, which with
// cells of half the reach are 5 x 5 x 5 cells.
//
// The search works on a copy of the positions that is made for it: the
// positions in the cell, in f32, in the order of the cells. The particles
// of a cell are next to one another there, so the search reads memory in
// runs. A distance in f32 is off by rounding. The search therefore takes a
// pair that is within `reach` plus a margin: a row may hold particles that
// are a little beyond `reach`, and it holds every particle within.
//
// A row has its particles in the order of the cells that are searched, and
// within a cell in the order of their indices.

// The number of cells along an edge of the length `length`. The cells are
// at least `width` wide and a little more, so that the cells within reach
// are still the same when the ratio of the two is a whole number.
func.func private @mdrt.cell_count(%length: f64, %width: f64) -> index {
  %c1 = arith.constant 1 : index
  %slack = arith.constant 1.0001 : f64
  %padded = arith.mulf %width, %slack : f64
  %ratio = arith.divf %length, %padded : f64
  %floor = math.floor %ratio : f64
  // At least one cell and at most 256: a cell that is not a number, or one
  // that a run that has failed has blown up, gives a count of cells that is
  // defined and fits memory; the cells are then wider than the reach needs,
  // and the range of the search follows from their count (#168).
  %one_f = arith.constant 1.0 : f64
  %most_f = arith.constant 256.0 : f64
  %least = arith.maxnumf %floor, %one_f : f64
  %bounded = arith.minnumf %least, %most_f : f64
  %wide = arith.fptosi %bounded : f64 to i64
  %count = arith.index_cast %wide : i64 to index
  %result = arith.maxsi %count, %c1 : index
  return %result : index
}

// The number of cells on each side of a cell that may hold particles within
// `reach` of a particle of the cell, along an edge with `count` cells. The
// reach is taken a little longer than it is: rounding may put a particle
// that is at the edge of a cell into the cell next to it.
func.func private @mdrt.cell_range(%length: f64, %count: index, %reach: f64)
    -> index {
  %c1 = arith.constant 1 : index
  %slack = arith.constant 1.00005 : f64
  %wide = arith.index_cast %count : index to i64
  %cells = arith.sitofp %wide : i64 to f64
  %width = arith.divf %length, %cells : f64
  %ratio = arith.divf %reach, %width : f64
  %padded = arith.mulf %ratio, %slack : f64
  %ceil = math.ceil %padded : f64
  // Defined for a cell that is not a number, as the count is (#168).
  %one_f = arith.constant 1.0 : f64
  %most_f = arith.constant 256.0 : f64
  %least = arith.maxnumf %ceil, %one_f : f64
  %bounded = arith.minnumf %least, %most_f : f64
  %whole = arith.fptosi %bounded : f64 to i64
  %range = arith.index_cast %whole : i64 to index
  %result = arith.maxsi %range, %c1 : index
  return %result : index
}

// The width of the cells for `count` particles in the cell `box`: the
// reach, half of it, or a third, and `least` or more.
//
// A search visits rows of cells and tests the particles in them. Narrow
// cells have fewer particles to test and more rows to visit, and a row
// costs as much as 12 particles. The width is the one with the least cost
// for the cells that it gives in this cell.
func.func private @mdrt.cell_width(%count: index, %box: vector<3xf64>,
    %reach: f64, %least: f64) -> f64 {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %lx = vector.extract %box[0] : f64 from vector<3xf64>
  %ly = vector.extract %box[1] : f64 from vector<3xf64>
  %lz = vector.extract %box[2] : f64 from vector<3xf64>
  %wide_count = arith.index_cast %count : index to i64
  %particles = arith.sitofp %wide_count : i64 to f64
  %row_cost = arith.constant 12.0 : f64
  %none = arith.constant 1.0e30 : f64

  %cost, %width = scf.for %parts = %c1 to %c4 step %c1
      iter_args(%best = %none, %chosen = %least) -> (f64, f64) {
    %wide_parts = arith.index_cast %parts : index to i64
    %divisor = arith.sitofp %wide_parts : i64 to f64
    %narrow = arith.divf %reach, %divisor : f64
    %tried = arith.maximumf %narrow, %least : f64

    %nx = func.call @mdrt.cell_count(%lx, %tried) : (f64, f64) -> index
    %ny = func.call @mdrt.cell_count(%ly, %tried) : (f64, f64) -> index
    %nz = func.call @mdrt.cell_count(%lz, %tried) : (f64, f64) -> index
    %rx = func.call @mdrt.cell_range(%lx, %nx, %reach)
        : (f64, index, f64) -> index
    %ry = func.call @mdrt.cell_range(%ly, %ny, %reach)
        : (f64, index, f64) -> index
    %rz = func.call @mdrt.cell_range(%lz, %nz, %reach)
        : (f64, index, f64) -> index
    %tx = arith.addi %rx, %rx : index
    %ty = arith.addi %ry, %ry : index
    %tz = arith.addi %rz, %rz : index
    %fx = arith.addi %tx, %c1 : index
    %fy = arith.addi %ty, %c1 : index
    %fz = arith.addi %tz, %c1 : index
    %sx = arith.minsi %nx, %fx : index
    %sy = arith.minsi %ny, %fy : index
    %sz = arith.minsi %nz, %fz : index

    // The rows that a search visits, and the particles that it tests: the
    // particles of the cells that it visits, out of all cells.
    %rows = arith.muli %sy, %sz : index
    %visited = arith.muli %rows, %sx : index
    %nxy = arith.muli %nx, %ny : index
    %cells = arith.muli %nxy, %nz : index
    %wide_rows = arith.index_cast %rows : index to i64
    %wide_visited = arith.index_cast %visited : index to i64
    %wide_cells = arith.index_cast %cells : index to i64
    %real_rows = arith.sitofp %wide_rows : i64 to f64
    %real_visited = arith.sitofp %wide_visited : i64 to f64
    %real_cells = arith.sitofp %wide_cells : i64 to f64
    %share = arith.divf %real_visited, %real_cells : f64
    %tested = arith.mulf %particles, %share : f64
    %visits = arith.mulf %row_cost, %real_rows : f64
    %total = arith.addf %visits, %tested : f64

    %better = arith.cmpf olt, %total, %best : f64
    %next_best = arith.select %better, %total, %best : f64
    %next_chosen = arith.select %better, %tried, %chosen : f64
    scf.yield %next_best, %next_chosen : f64, f64
  }
  return %width : f64
}

// The coordinate `x` in the cell: between 0 and `length`, up to rounding.
// `inverse` is one over `length`.
func.func private @mdrt.wrap(%x: f64, %length: f64, %inverse: f64) -> f64 {
  %images = arith.mulf %x, %inverse : f64
  %whole = math.floor %images : f64
  %shift = arith.mulf %whole, %length : f64
  %result = arith.subf %x, %shift : f64
  return %result : f64
}

// The cell that the coordinate `wrapped` lies in, along a direction with
// `count` cells. `inverse` is one over the edge length.
func.func private @mdrt.cell_coordinate(%wrapped: f64, %inverse: f64,
                                        %count: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %fraction = arith.mulf %wrapped, %inverse : f64
  %wide = arith.index_cast %count : index to i64
  %scale = arith.sitofp %wide : i64 to f64
  %position = arith.mulf %fraction, %scale : f64
  %integer = arith.fptosi %position : f64 to i64
  %coordinate = arith.index_cast %integer : i64 to index
  // Rounding can place a particle at an edge one cell too far.
  %last = arith.subi %count, %c1 : index
  %below = arith.minsi %coordinate, %last : index
  %result = arith.maxsi %below, %c0 : index
  return %result : index
}

// One component of the minimum-image displacement. `inverse` is one over
// `length`. The number of images is a whole number, so the result is that
// of a division by `length`, except where `d` is within rounding of half
// of `length`, which is beyond every reach.
func.func private @mdrt.minimum_image(%d: f32, %length: f32, %inverse: f32)
    -> f32 {
  %images = arith.mulf %d, %inverse : f32
  %nearest = math.roundeven %images : f32
  %shift = arith.mulf %nearest, %length : f32
  %result = arith.subf %d, %shift : f32
  return %result : f32
}

func.func private @mdrt.build_neighbors_matrix(
    %x: memref<?x3xf64>, %box: vector<3xf64>, %reach: f64, %cell_width: f64,
    %counts: memref<?xi32>, %index: memref<?x?xi32>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %n = memref.dim %x, %c0 : memref<?x3xf64>
  %row_width = memref.dim %index, %c1 : memref<?x?xi32>

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

  // What the search computes with, in f32. The margin is more than the
  // rounding of a distance between positions in the cell.
  %narrow_lx = arith.truncf %lx : f64 to f32
  %narrow_ly = arith.truncf %ly : f64 to f32
  %narrow_lz = arith.truncf %lz : f64 to f32
  %narrow_unit = arith.constant 1.0 : f32
  %narrow_ilx = arith.divf %narrow_unit, %narrow_lx : f32
  %narrow_ily = arith.divf %narrow_unit, %narrow_ly : f32
  %narrow_ilz = arith.divf %narrow_unit, %narrow_lz : f32
  %lxy = arith.addf %lx, %ly : f64
  %lxyz = arith.addf %lxy, %lz : f64
  %tiny = arith.constant 3.0e-6 : f64
  %margin = arith.mulf %lxyz, %tiny : f64
  %far = arith.addf %reach, %margin : f64
  %far2 = arith.mulf %far, %far : f64
  %limit2 = arith.truncf %far2 : f64 to f32

  //===--------------------------------------------------------------------===//
  // Counting sort of the particles by cell
  //===--------------------------------------------------------------------===//

  %key = memref.alloc(%n) : memref<?xindex>
  %start = memref.alloc(%cells1) : memref<?xindex>
  %cursor = memref.alloc(%cells) : memref<?xindex>
  %order = memref.alloc(%n) : memref<?xindex>
  %wrapped = memref.alloc(%n) : memref<?x3xf32>
  %sorted = memref.alloc(%n) : memref<?x3xf32>

  scf.for %c = %c0 to %cells1 step %c1 {
    memref.store %c0, %start[%c] : memref<?xindex>
  }

  // The position of each particle in the cell, its cell, and the number of
  // particles of each cell: start[k + 1] counts the particles of cell k.
  scf.for %i = %c0 to %n step %c1 {
    %xi = memref.load %x[%i, %c0] : memref<?x3xf64>
    %yi = memref.load %x[%i, %c1] : memref<?x3xf64>
    %zi = memref.load %x[%i, %c2] : memref<?x3xf64>
    %wx = func.call @mdrt.wrap(%xi, %lx, %ilx) : (f64, f64, f64) -> f64
    %wy = func.call @mdrt.wrap(%yi, %ly, %ily) : (f64, f64, f64) -> f64
    %wz = func.call @mdrt.wrap(%zi, %lz, %ilz) : (f64, f64, f64) -> f64
    %narrow_x = arith.truncf %wx : f64 to f32
    %narrow_y = arith.truncf %wy : f64 to f32
    %narrow_z = arith.truncf %wz : f64 to f32
    memref.store %narrow_x, %wrapped[%i, %c0] : memref<?x3xf32>
    memref.store %narrow_y, %wrapped[%i, %c1] : memref<?x3xf32>
    memref.store %narrow_z, %wrapped[%i, %c2] : memref<?x3xf32>

    %cx = func.call @mdrt.cell_coordinate(%wx, %ilx, %nx)
        : (f64, f64, index) -> index
    %cy = func.call @mdrt.cell_coordinate(%wy, %ily, %ny)
        : (f64, f64, index) -> index
    %cz = func.call @mdrt.cell_coordinate(%wz, %ilz, %nz)
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

  // The particles in the order of the cells, and their positions in that
  // order. The particles of a cell are in the order of their indices.
  scf.for %i = %c0 to %n step %c1 {
    %k = memref.load %key[%i] : memref<?xindex>
    %p = memref.load %cursor[%k] : memref<?xindex>
    memref.store %i, %order[%p] : memref<?xindex>
    %q = arith.addi %p, %c1 : index
    memref.store %q, %cursor[%k] : memref<?xindex>
    %wx = memref.load %wrapped[%i, %c0] : memref<?x3xf32>
    %wy = memref.load %wrapped[%i, %c1] : memref<?x3xf32>
    %wz = memref.load %wrapped[%i, %c2] : memref<?x3xf32>
    memref.store %wx, %sorted[%p, %c0] : memref<?x3xf32>
    memref.store %wy, %sorted[%p, %c1] : memref<?x3xf32>
    memref.store %wz, %sorted[%p, %c2] : memref<?x3xf32>
  }

  //===--------------------------------------------------------------------===//
  // Neighbors
  //===--------------------------------------------------------------------===//

  // The cells within reach: `range` on each side of the cell of the
  // particle. Where these are more than the cells that there are, the cells
  // on the two sides coincide; each cell is visited once.
  %range_x = call @mdrt.cell_range(%lx, %nx, %reach)
      : (f64, index, f64) -> index
  %range_y = call @mdrt.cell_range(%ly, %ny, %reach)
      : (f64, index, f64) -> index
  %range_z = call @mdrt.cell_range(%lz, %nz, %reach)
      : (f64, index, f64) -> index
  %twice_x = arith.addi %range_x, %range_x : index
  %twice_y = arith.addi %range_y, %range_y : index
  %twice_z = arith.addi %range_z, %range_z : index
  %full_x = arith.addi %twice_x, %c1 : index
  %full_y = arith.addi %twice_y, %c1 : index
  %full_z = arith.addi %twice_z, %c1 : index
  %span_x = arith.minsi %nx, %full_x : index
  %span_y = arith.minsi %ny, %full_y : index
  %span_z = arith.minsi %nz, %full_z : index

  // The first cell is `range` cells before the cell of the particle, and
  // that cell itself where every cell is visited. To stay in unsigned
  // arithmetic, add the number of cells before taking the remainder.
  %all_x = arith.cmpi eq, %span_x, %full_x : index
  %all_y = arith.cmpi eq, %span_y, %full_y : index
  %all_z = arith.cmpi eq, %span_z, %full_z : index
  %back_x = arith.subi %nx, %range_x : index
  %back_y = arith.subi %ny, %range_y : index
  %back_z = arith.subi %nz, %range_z : index
  %first_x = arith.select %all_x, %back_x, %c0 : index
  %first_y = arith.select %all_y, %back_y, %c0 : index
  %first_z = arith.select %all_z, %back_z, %c0 : index

  scf.parallel (%p) = (%c0) to (%n) step (%c1) {
    %i = memref.load %order[%p] : memref<?xindex>
    %xi = memref.load %sorted[%p, %c0] : memref<?x3xf32>
    %yi = memref.load %sorted[%p, %c1] : memref<?x3xf32>
    %zi = memref.load %sorted[%p, %c2] : memref<?x3xf32>

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

        // The cells of the row are next to one another in the order of
        // the cells, so they are read as one run, or as two where the row
        // goes around the edge of the cell.
        %zy = arith.muli %nz_cell, %ny : index
        %row = arith.addi %zy, %ny_cell : index
        %row_first = arith.muli %row, %nx : index
        %sx0 = arith.addi %cx, %first_x : index
        %x_begin = arith.remui %sx0, %nx : index
        %x_end = arith.addi %x_begin, %span_x : index
        %around = arith.cmpi ugt, %x_end, %nx : index
        %x_stop = arith.select %around, %nx, %x_end : index
        %x_rest0 = arith.subi %x_end, %nx : index
        %x_rest = arith.select %around, %x_rest0, %c0 : index

        %after_x = scf.for %part_x = %c0 to %c2 step %c1
            iter_args(%count_x = %count_y) -> (index) {
          %second = arith.cmpi ne, %part_x, %c0 : index
          %from_x = arith.select %second, %c0, %x_begin : index
          %to_x = arith.select %second, %x_rest, %x_stop : index
          %from_cell = arith.addi %row_first, %from_x : index
          %to_cell = arith.addi %row_first, %to_x : index
          %begin = memref.load %start[%from_cell] : memref<?xindex>
          %end = memref.load %start[%to_cell] : memref<?xindex>

          %after_cell = scf.for %q = %begin to %end step %c1
              iter_args(%count = %count_x) -> (index) {
            %xj = memref.load %sorted[%q, %c0] : memref<?x3xf32>
            %yj = memref.load %sorted[%q, %c1] : memref<?x3xf32>
            %zj = memref.load %sorted[%q, %c2] : memref<?x3xf32>
            %dx0 = arith.subf %xi, %xj : f32
            %dy0 = arith.subf %yi, %yj : f32
            %dz0 = arith.subf %zi, %zj : f32
            %dx = func.call @mdrt.minimum_image(%dx0, %narrow_lx, %narrow_ilx)
                : (f32, f32, f32) -> f32
            %dy = func.call @mdrt.minimum_image(%dy0, %narrow_ly, %narrow_ily)
                : (f32, f32, f32) -> f32
            %dz = func.call @mdrt.minimum_image(%dz0, %narrow_lz, %narrow_ilz)
                : (f32, f32, f32) -> f32
            %dx2 = arith.mulf %dx, %dx : f32
            %dy2 = arith.mulf %dy, %dy : f32
            %dz2 = arith.mulf %dz, %dz : f32
            %dxy2 = arith.addf %dx2, %dy2 : f32
            %r2 = arith.addf %dxy2, %dz2 : f32

            %near = arith.cmpf olt, %r2, %limit2 : f32
            %other = arith.cmpi ne, %p, %q : index
            %neighbor = arith.andi %near, %other : i1

            %fits = arith.cmpi ult, %count, %row_width : index
            %keep = arith.andi %neighbor, %fits : i1
            scf.if %keep {
              %j = memref.load %order[%q] : memref<?xindex>
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
  memref.dealloc %wrapped : memref<?x3xf32>
  memref.dealloc %sorted : memref<?x3xf32>

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

// The minimum-image displacement in a triclinic cell, in one pass along c,
// b, and a (docs/triclinic-m2.md, Section 2): exact for a displacement
// whose nearest image is within half of the least of a_x, b_y, c_z.
func.func private @mdrt.minimum_image_triclinic(
    %dx: f32, %dy: f32, %dz: f32, %ax: f32, %by: f32, %cz: f32, %bx: f32,
    %cx: f32, %cy: f32, %iax: f32, %iby: f32, %icz: f32) -> (f32, f32, f32) {
  %nz0 = arith.mulf %dz, %icz : f32
  %nz = math.roundeven %nz0 : f32
  %sx1 = arith.mulf %nz, %cx : f32
  %sy1 = arith.mulf %nz, %cy : f32
  %sz1 = arith.mulf %nz, %cz : f32
  %x1 = arith.subf %dx, %sx1 : f32
  %y1 = arith.subf %dy, %sy1 : f32
  %z1 = arith.subf %dz, %sz1 : f32
  %ny0 = arith.mulf %y1, %iby : f32
  %ny = math.roundeven %ny0 : f32
  %sx2 = arith.mulf %ny, %bx : f32
  %sy2 = arith.mulf %ny, %by : f32
  %x2 = arith.subf %x1, %sx2 : f32
  %y2 = arith.subf %y1, %sy2 : f32
  %nx0 = arith.mulf %x2, %iax : f32
  %nx = math.roundeven %nx0 : f32
  %sx3 = arith.mulf %nx, %ax : f32
  %x3 = arith.subf %x2, %sx3 : f32
  return %x3, %y2, %z1 : f32, f32, f32
}

// The neighbor matrix of a triclinic cell, `box` = (a_x, b_y, c_z, b_x,
// c_x, c_y), with the widths of the cell between its faces, `widths`: the
// build of the orthorhombic cell in the fractional coordinates, which the
// positions are wrapped in, with the minimum image of the triclinic cell.
// It holds every pair within `reach` when `reach` is at most half of the
// least of a_x, b_y, c_z.
func.func private @mdrt.build_neighbors_matrix_triclinic(
    %x: memref<?x3xf64>, %box: vector<6xf64>, %widths: vector<3xf64>,
    %reach: f64, %cell_width: f64,
    %counts: memref<?xi32>, %index: memref<?x?xi32>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %n = memref.dim %x, %c0 : memref<?x3xf64>
  %row_width = memref.dim %index, %c1 : memref<?x?xi32>

  // The cell: its diagonal a_x, b_y, c_z and its tilts b_x, c_x, c_y; the
  // cells of the search are those of the fractional coordinates, and their
  // numbers follow from the widths of the cell between its faces.
  %lx = vector.extract %box[0] : f64 from vector<6xf64>
  %ly = vector.extract %box[1] : f64 from vector<6xf64>
  %lz = vector.extract %box[2] : f64 from vector<6xf64>
  %tbx = vector.extract %box[3] : f64 from vector<6xf64>
  %tcx = vector.extract %box[4] : f64 from vector<6xf64>
  %tcy = vector.extract %box[5] : f64 from vector<6xf64>
  %width_x = vector.extract %widths[0] : f64 from vector<3xf64>
  %width_y = vector.extract %widths[1] : f64 from vector<3xf64>
  %width_z = vector.extract %widths[2] : f64 from vector<3xf64>
  %unit = arith.constant 1.0 : f64
  %ilx = arith.divf %unit, %lx : f64
  %ily = arith.divf %unit, %ly : f64
  %ilz = arith.divf %unit, %lz : f64
  %nx = call @mdrt.cell_count(%width_x, %cell_width) : (f64, f64) -> index
  %ny = call @mdrt.cell_count(%width_y, %cell_width) : (f64, f64) -> index
  %nz = call @mdrt.cell_count(%width_z, %cell_width) : (f64, f64) -> index
  %nxy = arith.muli %nx, %ny : index
  %cells = arith.muli %nxy, %nz : index
  %cells1 = arith.addi %cells, %c1 : index

  // What the search computes with, in f32. The margin is more than the
  // rounding of a distance between positions in the cell.
  %narrow_lx = arith.truncf %lx : f64 to f32
  %narrow_ly = arith.truncf %ly : f64 to f32
  %narrow_lz = arith.truncf %lz : f64 to f32
  %narrow_unit = arith.constant 1.0 : f32
  %narrow_ilx = arith.divf %narrow_unit, %narrow_lx : f32
  %narrow_ily = arith.divf %narrow_unit, %narrow_ly : f32
  %narrow_ilz = arith.divf %narrow_unit, %narrow_lz : f32
  %narrow_bx = arith.truncf %tbx : f64 to f32
  %narrow_cx = arith.truncf %tcx : f64 to f32
  %narrow_cy = arith.truncf %tcy : f64 to f32
  %abs_bx = math.absf %tbx : f64
  %abs_cx = math.absf %tcx : f64
  %abs_cy = math.absf %tcy : f64
  %lxy0 = arith.addf %lx, %ly : f64
  %lxy1 = arith.addf %lxy0, %abs_bx : f64
  %lxy2 = arith.addf %lxy1, %abs_cx : f64
  %lxy = arith.addf %lxy2, %abs_cy : f64
  %lxyz = arith.addf %lxy, %lz : f64
  %tiny = arith.constant 3.0e-6 : f64
  %margin = arith.mulf %lxyz, %tiny : f64
  %far = arith.addf %reach, %margin : f64
  %far2 = arith.mulf %far, %far : f64
  %limit2 = arith.truncf %far2 : f64 to f32

  //===--------------------------------------------------------------------===//
  // Counting sort of the particles by cell
  //===--------------------------------------------------------------------===//

  %key = memref.alloc(%n) : memref<?xindex>
  %start = memref.alloc(%cells1) : memref<?xindex>
  %cursor = memref.alloc(%cells) : memref<?xindex>
  %order = memref.alloc(%n) : memref<?xindex>
  %wrapped = memref.alloc(%n) : memref<?x3xf32>
  %sorted = memref.alloc(%n) : memref<?x3xf32>

  scf.for %c = %c0 to %cells1 step %c1 {
    memref.store %c0, %start[%c] : memref<?xindex>
  }

  // The position of each particle in the cell, its cell, and the number of
  // particles of each cell: start[k + 1] counts the particles of cell k.
  scf.for %i = %c0 to %n step %c1 {
    %xi = memref.load %x[%i, %c0] : memref<?x3xf64>
    %yi = memref.load %x[%i, %c1] : memref<?x3xf64>
    %zi = memref.load %x[%i, %c2] : memref<?x3xf64>
    // s = x H⁻¹, taken into [0, 1), and the position x = s H in the cell.
    %sz = arith.mulf %zi, %ilz : f64
    %sz_cy = arith.mulf %sz, %tcy : f64
    %y_rest = arith.subf %yi, %sz_cy : f64
    %sy = arith.mulf %y_rest, %ily : f64
    %sy_bx = arith.mulf %sy, %tbx : f64
    %sz_cx = arith.mulf %sz, %tcx : f64
    %x_rest0 = arith.subf %xi, %sy_bx : f64
    %x_rest = arith.subf %x_rest0, %sz_cx : f64
    %sx = arith.mulf %x_rest, %ilx : f64
    %fx = func.call @mdrt.wrap(%sx, %unit, %unit) : (f64, f64, f64) -> f64
    %fy = func.call @mdrt.wrap(%sy, %unit, %unit) : (f64, f64, f64) -> f64
    %fz = func.call @mdrt.wrap(%sz, %unit, %unit) : (f64, f64, f64) -> f64
    %wz = arith.mulf %fz, %lz : f64
    %fy_by = arith.mulf %fy, %ly : f64
    %fz_cy = arith.mulf %fz, %tcy : f64
    %wy = arith.addf %fy_by, %fz_cy : f64
    %fx_ax = arith.mulf %fx, %lx : f64
    %fy_bx = arith.mulf %fy, %tbx : f64
    %fz_cx = arith.mulf %fz, %tcx : f64
    %wx0 = arith.addf %fx_ax, %fy_bx : f64
    %wx = arith.addf %wx0, %fz_cx : f64
    %narrow_x = arith.truncf %wx : f64 to f32
    %narrow_y = arith.truncf %wy : f64 to f32
    %narrow_z = arith.truncf %wz : f64 to f32
    memref.store %narrow_x, %wrapped[%i, %c0] : memref<?x3xf32>
    memref.store %narrow_y, %wrapped[%i, %c1] : memref<?x3xf32>
    memref.store %narrow_z, %wrapped[%i, %c2] : memref<?x3xf32>

    %cx = func.call @mdrt.cell_coordinate(%fx, %unit, %nx)
        : (f64, f64, index) -> index
    %cy = func.call @mdrt.cell_coordinate(%fy, %unit, %ny)
        : (f64, f64, index) -> index
    %cz = func.call @mdrt.cell_coordinate(%fz, %unit, %nz)
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

  // The particles in the order of the cells, and their positions in that
  // order. The particles of a cell are in the order of their indices.
  scf.for %i = %c0 to %n step %c1 {
    %k = memref.load %key[%i] : memref<?xindex>
    %p = memref.load %cursor[%k] : memref<?xindex>
    memref.store %i, %order[%p] : memref<?xindex>
    %q = arith.addi %p, %c1 : index
    memref.store %q, %cursor[%k] : memref<?xindex>
    %wx = memref.load %wrapped[%i, %c0] : memref<?x3xf32>
    %wy = memref.load %wrapped[%i, %c1] : memref<?x3xf32>
    %wz = memref.load %wrapped[%i, %c2] : memref<?x3xf32>
    memref.store %wx, %sorted[%p, %c0] : memref<?x3xf32>
    memref.store %wy, %sorted[%p, %c1] : memref<?x3xf32>
    memref.store %wz, %sorted[%p, %c2] : memref<?x3xf32>
  }

  //===--------------------------------------------------------------------===//
  // Neighbors
  //===--------------------------------------------------------------------===//

  // The cells within reach: `range` on each side of the cell of the
  // particle. Where these are more than the cells that there are, the cells
  // on the two sides coincide; each cell is visited once.
  %range_x = call @mdrt.cell_range(%width_x, %nx, %reach)
      : (f64, index, f64) -> index
  %range_y = call @mdrt.cell_range(%width_y, %ny, %reach)
      : (f64, index, f64) -> index
  %range_z = call @mdrt.cell_range(%width_z, %nz, %reach)
      : (f64, index, f64) -> index
  %twice_x = arith.addi %range_x, %range_x : index
  %twice_y = arith.addi %range_y, %range_y : index
  %twice_z = arith.addi %range_z, %range_z : index
  %full_x = arith.addi %twice_x, %c1 : index
  %full_y = arith.addi %twice_y, %c1 : index
  %full_z = arith.addi %twice_z, %c1 : index
  %span_x = arith.minsi %nx, %full_x : index
  %span_y = arith.minsi %ny, %full_y : index
  %span_z = arith.minsi %nz, %full_z : index

  // The first cell is `range` cells before the cell of the particle, and
  // that cell itself where every cell is visited. To stay in unsigned
  // arithmetic, add the number of cells before taking the remainder.
  %all_x = arith.cmpi eq, %span_x, %full_x : index
  %all_y = arith.cmpi eq, %span_y, %full_y : index
  %all_z = arith.cmpi eq, %span_z, %full_z : index
  %back_x = arith.subi %nx, %range_x : index
  %back_y = arith.subi %ny, %range_y : index
  %back_z = arith.subi %nz, %range_z : index
  %first_x = arith.select %all_x, %back_x, %c0 : index
  %first_y = arith.select %all_y, %back_y, %c0 : index
  %first_z = arith.select %all_z, %back_z, %c0 : index

  scf.parallel (%p) = (%c0) to (%n) step (%c1) {
    %i = memref.load %order[%p] : memref<?xindex>
    %xi = memref.load %sorted[%p, %c0] : memref<?x3xf32>
    %yi = memref.load %sorted[%p, %c1] : memref<?x3xf32>
    %zi = memref.load %sorted[%p, %c2] : memref<?x3xf32>

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

        // The cells of the row are next to one another in the order of
        // the cells, so they are read as one run, or as two where the row
        // goes around the edge of the cell.
        %zy = arith.muli %nz_cell, %ny : index
        %row = arith.addi %zy, %ny_cell : index
        %row_first = arith.muli %row, %nx : index
        %sx0 = arith.addi %cx, %first_x : index
        %x_begin = arith.remui %sx0, %nx : index
        %x_end = arith.addi %x_begin, %span_x : index
        %around = arith.cmpi ugt, %x_end, %nx : index
        %x_stop = arith.select %around, %nx, %x_end : index
        %x_rest0 = arith.subi %x_end, %nx : index
        %x_rest = arith.select %around, %x_rest0, %c0 : index

        %after_x = scf.for %part_x = %c0 to %c2 step %c1
            iter_args(%count_x = %count_y) -> (index) {
          %second = arith.cmpi ne, %part_x, %c0 : index
          %from_x = arith.select %second, %c0, %x_begin : index
          %to_x = arith.select %second, %x_rest, %x_stop : index
          %from_cell = arith.addi %row_first, %from_x : index
          %to_cell = arith.addi %row_first, %to_x : index
          %begin = memref.load %start[%from_cell] : memref<?xindex>
          %end = memref.load %start[%to_cell] : memref<?xindex>

          %after_cell = scf.for %q = %begin to %end step %c1
              iter_args(%count = %count_x) -> (index) {
            %xj = memref.load %sorted[%q, %c0] : memref<?x3xf32>
            %yj = memref.load %sorted[%q, %c1] : memref<?x3xf32>
            %zj = memref.load %sorted[%q, %c2] : memref<?x3xf32>
            %dx0 = arith.subf %xi, %xj : f32
            %dy0 = arith.subf %yi, %yj : f32
            %dz0 = arith.subf %zi, %zj : f32
            %dx, %dy, %dz = func.call @mdrt.minimum_image_triclinic(
                %dx0, %dy0, %dz0, %narrow_lx, %narrow_ly, %narrow_lz,
                %narrow_bx, %narrow_cx, %narrow_cy, %narrow_ilx, %narrow_ily,
                %narrow_ilz)
                : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
                  -> (f32, f32, f32)
            %dx2 = arith.mulf %dx, %dx : f32
            %dy2 = arith.mulf %dy, %dy : f32
            %dz2 = arith.mulf %dz, %dz : f32
            %dxy2 = arith.addf %dx2, %dy2 : f32
            %r2 = arith.addf %dxy2, %dz2 : f32

            %near = arith.cmpf olt, %r2, %limit2 : f32
            %other = arith.cmpi ne, %p, %q : index
            %neighbor = arith.andi %near, %other : i1

            %fits = arith.cmpi ult, %count, %row_width : index
            %keep = arith.andi %neighbor, %fits : i1
            scf.if %keep {
              %j = memref.load %order[%q] : memref<?xindex>
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
  memref.dealloc %wrapped : memref<?x3xf32>
  memref.dealloc %sorted : memref<?x3xf32>

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

// The order of the particles by cell: `order[k]` is the particle that comes
// to place `k`. The cells are `width` wide or a little wider and numbered
// along x first. The particles of a cell are in the order of `ids`.
//
// The order depends on the positions and on `ids` only, not on the order
// that the particles are in.
func.func private @mdrt.spatial_order(
    %x: memref<?x3xf64>, %box: vector<3xf64>, %width: f64,
    %ids: memref<?xi32>, %order: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %n = memref.dim %x, %c0 : memref<?x3xf64>
  %lx = vector.extract %box[0] : f64 from vector<3xf64>
  %ly = vector.extract %box[1] : f64 from vector<3xf64>
  %lz = vector.extract %box[2] : f64 from vector<3xf64>
  %unit = arith.constant 1.0 : f64
  %ilx = arith.divf %unit, %lx : f64
  %ily = arith.divf %unit, %ly : f64
  %ilz = arith.divf %unit, %lz : f64
  %nx = call @mdrt.cell_count(%lx, %width) : (f64, f64) -> index
  %ny = call @mdrt.cell_count(%ly, %width) : (f64, f64) -> index
  %nz = call @mdrt.cell_count(%lz, %width) : (f64, f64) -> index
  %nxy = arith.muli %nx, %ny : index
  %cells = arith.muli %nxy, %nz : index
  %cells1 = arith.addi %cells, %c1 : index

  %key = memref.alloc(%n) : memref<?xindex>
  %start = memref.alloc(%cells1) : memref<?xindex>
  %cursor = memref.alloc(%cells) : memref<?xindex>

  scf.for %c = %c0 to %cells1 step %c1 {
    memref.store %c0, %start[%c] : memref<?xindex>
  }

  scf.for %i = %c0 to %n step %c1 {
    %xi = memref.load %x[%i, %c0] : memref<?x3xf64>
    %yi = memref.load %x[%i, %c1] : memref<?x3xf64>
    %zi = memref.load %x[%i, %c2] : memref<?x3xf64>
    %wx = func.call @mdrt.wrap(%xi, %lx, %ilx) : (f64, f64, f64) -> f64
    %wy = func.call @mdrt.wrap(%yi, %ly, %ily) : (f64, f64, f64) -> f64
    %wz = func.call @mdrt.wrap(%zi, %lz, %ilz) : (f64, f64, f64) -> f64
    %cx = func.call @mdrt.cell_coordinate(%wx, %ilx, %nx)
        : (f64, f64, index) -> index
    %cy = func.call @mdrt.cell_coordinate(%wy, %ily, %ny)
        : (f64, f64, index) -> index
    %cz = func.call @mdrt.cell_coordinate(%wz, %ilz, %nz)
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
    %wide = arith.index_cast %i : index to i64
    %narrow = arith.trunci %wide : i64 to i32
    memref.store %narrow, %order[%p] : memref<?xi32>
    %q = arith.addi %p, %c1 : index
    memref.store %q, %cursor[%k] : memref<?xindex>
  }

  // The particles of each cell in the order of `ids`, by insertion.
  scf.parallel (%c) = (%c0) to (%cells) step (%c1) {
    %next = arith.addi %c, %c1 : index
    %begin = memref.load %start[%c] : memref<?xindex>
    %end = memref.load %start[%next] : memref<?xindex>
    %first = arith.addi %begin, %c1 : index
    scf.for %p = %first to %end step %c1 {
      %value = memref.load %order[%p] : memref<?xi32>
      %particle = arith.index_cast %value : i32 to index
      %id = memref.load %ids[%particle] : memref<?xi32>
      // Move the entries before p with a larger number up by one.
      %hole = scf.while (%q = %p) : (index) -> index {
        %more = arith.cmpi ugt, %q, %begin : index
        %before = arith.subi %q, %c1 : index
        %safe = arith.select %more, %before, %begin : index
        %left = memref.load %order[%safe] : memref<?xi32>
        %left_particle = arith.index_cast %left : i32 to index
        %left_id = memref.load %ids[%left_particle] : memref<?xi32>
        %larger = arith.cmpi sgt, %left_id, %id : i32
        %go = arith.andi %more, %larger : i1
        scf.condition(%go) %q : index
      } do {
      ^bb0(%q: index):
        %before = arith.subi %q, %c1 : index
        %left = memref.load %order[%before] : memref<?xi32>
        memref.store %left, %order[%q] : memref<?xi32>
        scf.yield %before : index
      }
      memref.store %value, %order[%hole] : memref<?xi32>
    }
  }

  memref.dealloc %key : memref<?xindex>
  memref.dealloc %start : memref<?xindex>
  memref.dealloc %cursor : memref<?xindex>
  return
}
