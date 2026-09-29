// Builds a neighbor matrix on a device by binning the particles into cells.
//
// This is a template. The compiler adds it to a module that builds neighbor
// structures on a device, where it is specialized like any other code.
//
// The buffers are on the device. The simulation cell is orthorhombic and
// periodic, with edge lengths `box`. On return, row `i` of `index` holds the
// particles within `reach` of particle `i`, and `counts[i]` holds their
// number, limited to the width of a row. The result is the largest number
// of neighbors that a particle has, which may exceed the width; the caller
// can then tell that a row was too narrow.
//
// The matrix is that of the template for the host, entry by entry. See
// there for the cells, for the copy of the positions that the search works
// on, and for the margin of the search.
//
// A kernel runs one thread per item, in blocks of 128 threads. The threads
// beyond the last item do nothing.
//
// The particles of a cell are sorted by index after they are binned: the
// order in which threads take their slots is not fixed, and the order of
// the neighbors decides the order in which forces are added up.
//
// The search has one thread for a particle, or one for every row of cells
// of a particle. A thread takes as long as the particles that it tests, so
// the second is faster as long as the device has threads to spare. It
// searches twice: the threads of a particle count what they find, the
// counts become the places of the threads in the row of the particle, and
// the threads search again and fill the row. `split_limit` is the largest
// number of threads that the search is split into.

func.func private @mdrt_gpu_cell_count(%length: f64, %width: f64) -> index {
  %c1 = arith.constant 1 : index
  %slack = arith.constant 1.0001 : f64
  %padded = arith.mulf %width, %slack : f64
  %ratio = arith.divf %length, %padded : f64
  %floor = math.floor %ratio : f64
  %wide = arith.fptosi %floor : f64 to i64
  %count = arith.index_cast %wide : i64 to index
  %result = arith.maxsi %count, %c1 : index
  return %result : index
}

func.func private @mdrt_gpu_cell_range(%length: f64, %count: index,
                                       %reach: f64) -> index {
  %c1 = arith.constant 1 : index
  %slack = arith.constant 1.00005 : f64
  %wide = arith.index_cast %count : index to i64
  %cells = arith.sitofp %wide : i64 to f64
  %width = arith.divf %length, %cells : f64
  %ratio = arith.divf %reach, %width : f64
  %padded = arith.mulf %ratio, %slack : f64
  %ceil = math.ceil %padded : f64
  %whole = arith.fptosi %ceil : f64 to i64
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
func.func private @mdrt_gpu_cell_width(%count: index, %box: vector<3xf64>,
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

    %nx = func.call @mdrt_gpu_cell_count(%lx, %tried) : (f64, f64) -> index
    %ny = func.call @mdrt_gpu_cell_count(%ly, %tried) : (f64, f64) -> index
    %nz = func.call @mdrt_gpu_cell_count(%lz, %tried) : (f64, f64) -> index
    %rx = func.call @mdrt_gpu_cell_range(%lx, %nx, %reach)
        : (f64, index, f64) -> index
    %ry = func.call @mdrt_gpu_cell_range(%ly, %ny, %reach)
        : (f64, index, f64) -> index
    %rz = func.call @mdrt_gpu_cell_range(%lz, %nz, %reach)
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

// The number of blocks of `block` threads that hold `count` threads.
func.func private @mdrt_gpu_grid(%count: index, %block: index) -> index {
  %c1 = arith.constant 1 : index
  %last = arith.subi %block, %c1 : index
  %padded = arith.addi %count, %last : index
  %result = arith.divui %padded, %block : index
  return %result : index
}

func.func private @mdrt_gpu_build_neighbors_matrix(
    %x: memref<?x3xf64, 1>, %box: vector<3xf64>, %reach: f64,
    %cell_width: f64, %split_limit: index, %counts: memref<?xi32, 1>,
    %index: memref<?x?xi32, 1>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %block = arith.constant 128 : index
  %chunk = arith.constant 256 : index

  %n = memref.dim %x, %c0 : memref<?x3xf64, 1>
  %row_width = memref.dim %index, %c1 : memref<?x?xi32, 1>

  %lx = vector.extract %box[0] : f64 from vector<3xf64>
  %ly = vector.extract %box[1] : f64 from vector<3xf64>
  %lz = vector.extract %box[2] : f64 from vector<3xf64>
  %unit = arith.constant 1.0 : f64
  %ilx = arith.divf %unit, %lx : f64
  %ily = arith.divf %unit, %ly : f64
  %ilz = arith.divf %unit, %lz : f64
  %nx = call @mdrt_gpu_cell_count(%lx, %cell_width) : (f64, f64) -> index
  %ny = call @mdrt_gpu_cell_count(%ly, %cell_width) : (f64, f64) -> index
  %nz = call @mdrt_gpu_cell_count(%lz, %cell_width) : (f64, f64) -> index
  %nxy = arith.muli %nx, %ny : index
  %cells = arith.muli %nxy, %nz : index
  %chunks = call @mdrt_gpu_grid(%n, %chunk) : (index, index) -> index
  %cell_chunks = call @mdrt_gpu_grid(%cells, %chunk)
      : (index, index) -> index

  // What the search computes with, in f32.
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

  // The cells within reach.
  %range_x = call @mdrt_gpu_cell_range(%lx, %nx, %reach)
      : (f64, index, f64) -> index
  %range_y = call @mdrt_gpu_cell_range(%ly, %ny, %reach)
      : (f64, index, f64) -> index
  %range_z = call @mdrt_gpu_cell_range(%lz, %nz, %reach)
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
  %all_x = arith.cmpi eq, %span_x, %full_x : index
  %all_y = arith.cmpi eq, %span_y, %full_y : index
  %all_z = arith.cmpi eq, %span_z, %full_z : index
  %back_x = arith.subi %nx, %range_x : index
  %back_y = arith.subi %ny, %range_y : index
  %back_z = arith.subi %nz, %range_z : index
  %first_x = arith.select %all_x, %back_x, %c0 : index
  %first_y = arith.select %all_y, %back_y, %c0 : index
  %first_z = arith.select %all_z, %back_z, %c0 : index

  // The threads of the search: one for a particle, or one for each row of
  // cells of a particle.
  %rows_within = arith.muli %span_y, %span_z : index
  %split_threads = arith.muli %n, %rows_within : index
  %splits = arith.cmpi ule, %split_threads, %split_limit : index
  %split = arith.select %splits, %c1, %c0 : index
  %groups = arith.select %splits, %rows_within, %c1 : index
  %threads = arith.muli %n, %groups : index

  %grid_n = call @mdrt_gpu_grid(%n, %block) : (index, index) -> index
  %grid_cells = call @mdrt_gpu_grid(%cells, %block) : (index, index) -> index
  %grid_chunks = call @mdrt_gpu_grid(%chunks, %block)
      : (index, index) -> index
  %grid_cell_chunks = call @mdrt_gpu_grid(%cell_chunks, %block)
      : (index, index) -> index
  %grid_threads = call @mdrt_gpu_grid(%threads, %block)
      : (index, index) -> index

  %cells1 = arith.addi %cells, %c1 : index
  %key = gpu.alloc (%n) : memref<?xi32, 1>
  %order = gpu.alloc (%n) : memref<?xi32, 1>
  %wrapped = gpu.alloc (%n) : memref<?x3xf32, 1>
  %sorted = gpu.alloc (%n) : memref<?x3xf32, 1>
  %held = gpu.alloc (%cells) : memref<?xi32, 1>
  %start = gpu.alloc (%cells1) : memref<?xi32, 1>
  %cursor = gpu.alloc (%cells) : memref<?xi32, 1>
  %cell_sums = gpu.alloc (%cell_chunks) : memref<?xi32, 1>
  %part = gpu.alloc (%threads) : memref<?xi32, 1>
  %partial = gpu.alloc (%chunks) : memref<?xi32, 1>
  %result = gpu.alloc () : memref<1xi32, 1>
  %host = memref.alloca() : memref<1xi32>

  //===--------------------------------------------------------------------===//
  // Counting sort of the particles by cell
  //===--------------------------------------------------------------------===//

  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_cells, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
    %base = arith.muli %bx, %block : index
    %c = arith.addi %base, %tx : index
    %inside = arith.cmpi ult, %c, %cells : index
    scf.if %inside {
      %none = arith.constant 0 : i32
      memref.store %none, %held[%c] : memref<?xi32, 1>
    }
    gpu.terminator
  }

  // The position of each particle in the cell, its cell, and the number of
  // particles that each cell holds.
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_n, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
    %base = arith.muli %bx, %block : index
    %i = arith.addi %base, %tx : index
    %inside = arith.cmpi ult, %i, %n : index
    scf.if %inside {
      %i0 = arith.constant 0 : index
      %i1 = arith.constant 1 : index
      %i2 = arith.constant 2 : index
      %one = arith.constant 1 : i32
      %xi = memref.load %x[%i, %i0] : memref<?x3xf64, 1>
      %yi = memref.load %x[%i, %i1] : memref<?x3xf64, 1>
      %zi = memref.load %x[%i, %i2] : memref<?x3xf64, 1>

      // Rounding can place a particle at an edge one cell too far.
      %x0 = arith.mulf %xi, %ilx : f64
      %x1 = math.floor %x0 : f64
      %x2 = arith.mulf %x1, %lx : f64
      %wx = arith.subf %xi, %x2 : f64
      %x3 = arith.mulf %wx, %ilx : f64
      %x4 = arith.index_cast %nx : index to i64
      %x5 = arith.sitofp %x4 : i64 to f64
      %x6 = arith.mulf %x3, %x5 : f64
      %x7 = arith.fptosi %x6 : f64 to i64
      %x8 = arith.index_cast %x7 : i64 to index
      %xl = arith.subi %nx, %i1 : index
      %xb = arith.minsi %x8, %xl : index
      %cx = arith.maxsi %xb, %i0 : index

      %y0 = arith.mulf %yi, %ily : f64
      %y1 = math.floor %y0 : f64
      %y2 = arith.mulf %y1, %ly : f64
      %wy = arith.subf %yi, %y2 : f64
      %y3 = arith.mulf %wy, %ily : f64
      %y4 = arith.index_cast %ny : index to i64
      %y5 = arith.sitofp %y4 : i64 to f64
      %y6 = arith.mulf %y3, %y5 : f64
      %y7 = arith.fptosi %y6 : f64 to i64
      %y8 = arith.index_cast %y7 : i64 to index
      %yl = arith.subi %ny, %i1 : index
      %yb = arith.minsi %y8, %yl : index
      %cy = arith.maxsi %yb, %i0 : index

      %z0 = arith.mulf %zi, %ilz : f64
      %z1 = math.floor %z0 : f64
      %z2 = arith.mulf %z1, %lz : f64
      %wz = arith.subf %zi, %z2 : f64
      %z3 = arith.mulf %wz, %ilz : f64
      %z4 = arith.index_cast %nz : index to i64
      %z5 = arith.sitofp %z4 : i64 to f64
      %z6 = arith.mulf %z3, %z5 : f64
      %z7 = arith.fptosi %z6 : f64 to i64
      %z8 = arith.index_cast %z7 : i64 to index
      %zl = arith.subi %nz, %i1 : index
      %zb = arith.minsi %z8, %zl : index
      %cz = arith.maxsi %zb, %i0 : index

      %narrow_x = arith.truncf %wx : f64 to f32
      %narrow_y = arith.truncf %wy : f64 to f32
      %narrow_z = arith.truncf %wz : f64 to f32
      memref.store %narrow_x, %wrapped[%i, %i0] : memref<?x3xf32, 1>
      memref.store %narrow_y, %wrapped[%i, %i1] : memref<?x3xf32, 1>
      memref.store %narrow_z, %wrapped[%i, %i2] : memref<?x3xf32, 1>

      %zy = arith.muli %cz, %ny : index
      %row = arith.addi %zy, %cy : index
      %rows = arith.muli %row, %nx : index
      %k = arith.addi %rows, %cx : index
      %k32 = arith.index_cast %k : index to i32
      memref.store %k32, %key[%i] : memref<?xi32, 1>
      %old = memref.atomic_rmw addi %one, %held[%k]
          : (i32, memref<?xi32, 1>) -> i32
    }
    gpu.terminator
  }

  // The counts become the offsets of the cells: the sums of chunks of 256
  // cells, the offsets of the chunks in one thread, and the offsets of the
  // cells of each chunk.
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_cell_chunks, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
    %base = arith.muli %bx, %block : index
    %b = arith.addi %base, %tx : index
    %inside = arith.cmpi ult, %b, %cell_chunks : index
    scf.if %inside {
      %i1 = arith.constant 1 : index
      %none = arith.constant 0 : i32
      %begin = arith.muli %b, %chunk : index
      %full = arith.addi %begin, %chunk : index
      %short = arith.cmpi ult, %cells, %full : index
      %end = arith.select %short, %cells, %full : index
      %sum = scf.for %c = %begin to %end step %i1
          iter_args(%partial_sum = %none) -> (i32) {
        %count = memref.load %held[%c] : memref<?xi32, 1>
        %next = arith.addi %partial_sum, %count : i32
        scf.yield %next : i32
      }
      memref.store %sum, %cell_sums[%b] : memref<?xi32, 1>
    }
    gpu.terminator
  }

  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %none = arith.constant 0 : i32
    %total = scf.for %b = %i0 to %cell_chunks step %i1
        iter_args(%before = %none) -> (i32) {
      %sum = memref.load %cell_sums[%b] : memref<?xi32, 1>
      memref.store %before, %cell_sums[%b] : memref<?xi32, 1>
      %next = arith.addi %before, %sum : i32
      scf.yield %next : i32
    }
    memref.store %total, %start[%cells] : memref<?xi32, 1>
    gpu.terminator
  }

  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_cell_chunks, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
    %base = arith.muli %bx, %block : index
    %b = arith.addi %base, %tx : index
    %inside = arith.cmpi ult, %b, %cell_chunks : index
    scf.if %inside {
      %i1 = arith.constant 1 : index
      %begin = arith.muli %b, %chunk : index
      %full = arith.addi %begin, %chunk : index
      %short = arith.cmpi ult, %cells, %full : index
      %end = arith.select %short, %cells, %full : index
      %first = memref.load %cell_sums[%b] : memref<?xi32, 1>
      %last = scf.for %c = %begin to %end step %i1
          iter_args(%before = %first) -> (i32) {
        %count = memref.load %held[%c] : memref<?xi32, 1>
        memref.store %before, %start[%c] : memref<?xi32, 1>
        memref.store %before, %cursor[%c] : memref<?xi32, 1>
        %next = arith.addi %before, %count : i32
        scf.yield %next : i32
      }
    }
    gpu.terminator
  }

  // Each particle takes the next slot of its cell.
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_n, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
    %base = arith.muli %bx, %block : index
    %i = arith.addi %base, %tx : index
    %inside = arith.cmpi ult, %i, %n : index
    scf.if %inside {
      %one = arith.constant 1 : i32
      %k32 = memref.load %key[%i] : memref<?xi32, 1>
      %k = arith.index_cast %k32 : i32 to index
      %slot32 = memref.atomic_rmw addi %one, %cursor[%k]
          : (i32, memref<?xi32, 1>) -> i32
      %slot = arith.index_cast %slot32 : i32 to index
      %narrow = arith.index_cast %i : index to i32
      memref.store %narrow, %order[%slot] : memref<?xi32, 1>
    }
    gpu.terminator
  }

  // The particles of each cell, sorted by index, by insertion, and their
  // positions in that order.
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_cells, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
    %base = arith.muli %bx, %block : index
    %c = arith.addi %base, %tx : index
    %inside = arith.cmpi ult, %c, %cells : index
    scf.if %inside {
      %i0 = arith.constant 0 : index
      %i1 = arith.constant 1 : index
      %i2 = arith.constant 2 : index
      %next = arith.addi %c, %i1 : index
      %begin32 = memref.load %start[%c] : memref<?xi32, 1>
      %end32 = memref.load %start[%next] : memref<?xi32, 1>
      %begin = arith.index_cast %begin32 : i32 to index
      %end = arith.index_cast %end32 : i32 to index
      %first = arith.addi %begin, %i1 : index
      scf.for %p = %first to %end step %i1 {
        %value = memref.load %order[%p] : memref<?xi32, 1>
        // Move the larger entries before p up by one.
        %hole = scf.while (%q = %p) : (index) -> index {
          %more = arith.cmpi ugt, %q, %begin : index
          %before = arith.subi %q, %i1 : index
          %safe = arith.select %more, %before, %begin : index
          %left = memref.load %order[%safe] : memref<?xi32, 1>
          %larger = arith.cmpi sgt, %left, %value : i32
          %go = arith.andi %more, %larger : i1
          scf.condition(%go) %q : index
        } do {
        ^bb0(%q: index):
          %before = arith.subi %q, %i1 : index
          %left = memref.load %order[%before] : memref<?xi32, 1>
          memref.store %left, %order[%q] : memref<?xi32, 1>
          scf.yield %before : index
        }
        memref.store %value, %order[%hole] : memref<?xi32, 1>
      }
      scf.for %p = %begin to %end step %i1 {
        %j32 = memref.load %order[%p] : memref<?xi32, 1>
        %j = arith.index_cast %j32 : i32 to index
        %wx = memref.load %wrapped[%j, %i0] : memref<?x3xf32, 1>
        %wy = memref.load %wrapped[%j, %i1] : memref<?x3xf32, 1>
        %wz = memref.load %wrapped[%j, %i2] : memref<?x3xf32, 1>
        memref.store %wx, %sorted[%p, %i0] : memref<?x3xf32, 1>
        memref.store %wy, %sorted[%p, %i1] : memref<?x3xf32, 1>
        memref.store %wz, %sorted[%p, %i2] : memref<?x3xf32, 1>
      }
    }
    gpu.terminator
  }

  //===--------------------------------------------------------------------===//
  // Neighbors
  //===--------------------------------------------------------------------===//

  // Where the search is split, its threads count what they find, and the
  // counts become the places of the threads in the rows.
  scf.if %splits {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_threads, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
      %base = arith.muli %bx, %block : index
      %t = arith.addi %base, %tx : index
      %inside = arith.cmpi ult, %t, %threads : index
      scf.if %inside {
        %i0 = arith.constant 0 : index
        %i1 = arith.constant 1 : index
        %i2 = arith.constant 2 : index
        %p = arith.divui %t, %groups : index
        %g = arith.remui %t, %groups : index
        %i32 = memref.load %order[%p] : memref<?xi32, 1>
        %i = arith.index_cast %i32 : i32 to index
        %xi = memref.load %sorted[%p, %i0] : memref<?x3xf32, 1>
        %yi = memref.load %sorted[%p, %i1] : memref<?x3xf32, 1>
        %zi = memref.load %sorted[%p, %i2] : memref<?x3xf32, 1>

        %k32 = memref.load %key[%i] : memref<?xi32, 1>
        %k = arith.index_cast %k32 : i32 to index
        %cx = arith.remui %k, %nx : index
        %rest = arith.divui %k, %nx : index
        %cy = arith.remui %rest, %ny : index
        %cz = arith.divui %rest, %ny : index

        // The rows of cells that the thread searches: one, or all.
        %is_split = arith.cmpi ne, %split, %i0 : index
        %g1 = arith.addi %g, %i1 : index
        %row_begin = arith.select %is_split, %g, %i0 : index
        %row_end = arith.select %is_split, %g1, %rows_within : index
        %none = arith.constant 0 : index

        %found = scf.for %r = %row_begin to %row_end step %i1
            iter_args(%count_r = %none) -> (index) {
          %oz = arith.divui %r, %span_y : index
          %oy = arith.remui %r, %span_y : index
          %sz0 = arith.addi %cz, %first_z : index
          %sz1 = arith.addi %sz0, %oz : index
          %nz_cell = arith.remui %sz1, %nz : index
          %sy0 = arith.addi %cy, %first_y : index
          %sy1 = arith.addi %sy0, %oy : index
          %ny_cell = arith.remui %sy1, %ny : index

          // The cells of the row are next to one another in the order of
          // the cells, so the thread reads them as one run, or as two where
          // the row goes around the edge of the cell.
          %zy = arith.muli %nz_cell, %ny : index
          %row = arith.addi %zy, %ny_cell : index
          %row_first = arith.muli %row, %nx : index
          %sx0 = arith.addi %cx, %first_x : index
          %x_begin = arith.remui %sx0, %nx : index
          %x_end = arith.addi %x_begin, %span_x : index
          %around = arith.cmpi ugt, %x_end, %nx : index
          %x_stop = arith.select %around, %nx, %x_end : index
          %x_rest0 = arith.subi %x_end, %nx : index
          %x_rest = arith.select %around, %x_rest0, %i0 : index

          %after_x = scf.for %part_x = %i0 to %i2 step %i1
              iter_args(%count_x = %count_r) -> (index) {
            %second = arith.cmpi ne, %part_x, %i0 : index
            %from_x = arith.select %second, %i0, %x_begin : index
            %to_x = arith.select %second, %x_rest, %x_stop : index
            %from_cell = arith.addi %row_first, %from_x : index
            %to_cell = arith.addi %row_first, %to_x : index
            %begin32 = memref.load %start[%from_cell] : memref<?xi32, 1>
            %end32 = memref.load %start[%to_cell] : memref<?xi32, 1>
            %begin = arith.index_cast %begin32 : i32 to index
            %end = arith.index_cast %end32 : i32 to index

            %after_cell = scf.for %q = %begin to %end step %i1
                iter_args(%count = %count_x) -> (index) {
              %xj = memref.load %sorted[%q, %i0] : memref<?x3xf32, 1>
              %yj = memref.load %sorted[%q, %i1] : memref<?x3xf32, 1>
              %zj = memref.load %sorted[%q, %i2] : memref<?x3xf32, 1>

              // The minimum-image displacement, with one over the edge
              // lengths: see the template for the host.
                %dx0 = arith.subf %xi, %xj : f32
                %dx1 = arith.mulf %dx0, %narrow_ilx : f32
                %dx2 = math.roundeven %dx1 : f32
                %dx3 = arith.mulf %dx2, %narrow_lx : f32
                %dx = arith.subf %dx0, %dx3 : f32
                %dy0 = arith.subf %yi, %yj : f32
                %dy1 = arith.mulf %dy0, %narrow_ily : f32
                %dy2 = math.roundeven %dy1 : f32
                %dy3 = arith.mulf %dy2, %narrow_ly : f32
                %dy = arith.subf %dy0, %dy3 : f32
                %dz0 = arith.subf %zi, %zj : f32
                %dz1 = arith.mulf %dz0, %narrow_ilz : f32
                %dz2 = math.roundeven %dz1 : f32
                %dz3 = arith.mulf %dz2, %narrow_lz : f32
                %dz = arith.subf %dz0, %dz3 : f32

              %dx_2 = arith.mulf %dx, %dx : f32
              %dy_2 = arith.mulf %dy, %dy : f32
              %dz_2 = arith.mulf %dz, %dz : f32
              %dxy_2 = arith.addf %dx_2, %dy_2 : f32
              %r2 = arith.addf %dxy_2, %dz_2 : f32

              %near = arith.cmpf olt, %r2, %limit2 : f32
              %other = arith.cmpi ne, %p, %q : index
              %neighbor = arith.andi %near, %other : i1

              %more = arith.addi %count, %i1 : index
              %next = arith.select %neighbor, %more, %count : index
              scf.yield %next : index
            }
            scf.yield %after_cell : index
          }
          scf.yield %after_x : index
        }

        %found32 = arith.index_cast %found : index to i32
        memref.store %found32, %part[%t] : memref<?xi32, 1>
      }
      gpu.terminator
    }

    gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_n, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
      %base = arith.muli %bx, %block : index
      %p = arith.addi %base, %tx : index
      %inside = arith.cmpi ult, %p, %n : index
      scf.if %inside {
        %i0 = arith.constant 0 : index
        %i1 = arith.constant 1 : index
        %none = arith.constant 0 : i32
        %begin = arith.muli %p, %groups : index
        %end = arith.addi %begin, %groups : index
        %total = scf.for %t = %begin to %end step %i1
            iter_args(%before = %none) -> (i32) {
          %found = memref.load %part[%t] : memref<?xi32, 1>
          memref.store %before, %part[%t] : memref<?xi32, 1>
          %next = arith.addi %before, %found : i32
          scf.yield %next : i32
        }
        %i32 = memref.load %order[%p] : memref<?xi32, 1>
        %i = arith.index_cast %i32 : i32 to index
        memref.store %total, %counts[%i] : memref<?xi32, 1>
      }
      gpu.terminator
    }
  }

  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_threads, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
    %base = arith.muli %bx, %block : index
    %t = arith.addi %base, %tx : index
    %inside = arith.cmpi ult, %t, %threads : index
    scf.if %inside {
      %i0 = arith.constant 0 : index
      %i1 = arith.constant 1 : index
      %i2 = arith.constant 2 : index
      %p = arith.divui %t, %groups : index
      %g = arith.remui %t, %groups : index
      %i32 = memref.load %order[%p] : memref<?xi32, 1>
      %i = arith.index_cast %i32 : i32 to index
      %xi = memref.load %sorted[%p, %i0] : memref<?x3xf32, 1>
      %yi = memref.load %sorted[%p, %i1] : memref<?x3xf32, 1>
      %zi = memref.load %sorted[%p, %i2] : memref<?x3xf32, 1>

      %k32 = memref.load %key[%i] : memref<?xi32, 1>
      %k = arith.index_cast %k32 : i32 to index
      %cx = arith.remui %k, %nx : index
      %rest = arith.divui %k, %nx : index
      %cy = arith.remui %rest, %ny : index
      %cz = arith.divui %rest, %ny : index

      // The rows of cells that the thread searches: one, or all.
      %is_split = arith.cmpi ne, %split, %i0 : index
      %g1 = arith.addi %g, %i1 : index
      %row_begin = arith.select %is_split, %g, %i0 : index
      %row_end = arith.select %is_split, %g1, %rows_within : index
      // The first entry of the thread in the row.
      %offset32 = memref.load %part[%t] : memref<?xi32, 1>
      %offset = arith.index_cast %offset32 : i32 to index
      %none = arith.select %is_split, %offset, %i0 : index

      %found = scf.for %r = %row_begin to %row_end step %i1
          iter_args(%count_r = %none) -> (index) {
        %oz = arith.divui %r, %span_y : index
        %oy = arith.remui %r, %span_y : index
        %sz0 = arith.addi %cz, %first_z : index
        %sz1 = arith.addi %sz0, %oz : index
        %nz_cell = arith.remui %sz1, %nz : index
        %sy0 = arith.addi %cy, %first_y : index
        %sy1 = arith.addi %sy0, %oy : index
        %ny_cell = arith.remui %sy1, %ny : index

        // The cells of the row are next to one another in the order of
        // the cells, so the thread reads them as one run, or as two where
        // the row goes around the edge of the cell.
        %zy = arith.muli %nz_cell, %ny : index
        %row = arith.addi %zy, %ny_cell : index
        %row_first = arith.muli %row, %nx : index
        %sx0 = arith.addi %cx, %first_x : index
        %x_begin = arith.remui %sx0, %nx : index
        %x_end = arith.addi %x_begin, %span_x : index
        %around = arith.cmpi ugt, %x_end, %nx : index
        %x_stop = arith.select %around, %nx, %x_end : index
        %x_rest0 = arith.subi %x_end, %nx : index
        %x_rest = arith.select %around, %x_rest0, %i0 : index

        %after_x = scf.for %part_x = %i0 to %i2 step %i1
            iter_args(%count_x = %count_r) -> (index) {
          %second = arith.cmpi ne, %part_x, %i0 : index
          %from_x = arith.select %second, %i0, %x_begin : index
          %to_x = arith.select %second, %x_rest, %x_stop : index
          %from_cell = arith.addi %row_first, %from_x : index
          %to_cell = arith.addi %row_first, %to_x : index
          %begin32 = memref.load %start[%from_cell] : memref<?xi32, 1>
          %end32 = memref.load %start[%to_cell] : memref<?xi32, 1>
          %begin = arith.index_cast %begin32 : i32 to index
          %end = arith.index_cast %end32 : i32 to index

          %after_cell = scf.for %q = %begin to %end step %i1
              iter_args(%count = %count_x) -> (index) {
            %xj = memref.load %sorted[%q, %i0] : memref<?x3xf32, 1>
            %yj = memref.load %sorted[%q, %i1] : memref<?x3xf32, 1>
            %zj = memref.load %sorted[%q, %i2] : memref<?x3xf32, 1>

            // The minimum-image displacement, with one over the edge
            // lengths: see the template for the host.
              %dx0 = arith.subf %xi, %xj : f32
              %dx1 = arith.mulf %dx0, %narrow_ilx : f32
              %dx2 = math.roundeven %dx1 : f32
              %dx3 = arith.mulf %dx2, %narrow_lx : f32
              %dx = arith.subf %dx0, %dx3 : f32
              %dy0 = arith.subf %yi, %yj : f32
              %dy1 = arith.mulf %dy0, %narrow_ily : f32
              %dy2 = math.roundeven %dy1 : f32
              %dy3 = arith.mulf %dy2, %narrow_ly : f32
              %dy = arith.subf %dy0, %dy3 : f32
              %dz0 = arith.subf %zi, %zj : f32
              %dz1 = arith.mulf %dz0, %narrow_ilz : f32
              %dz2 = math.roundeven %dz1 : f32
              %dz3 = arith.mulf %dz2, %narrow_lz : f32
              %dz = arith.subf %dz0, %dz3 : f32

            %dx_2 = arith.mulf %dx, %dx : f32
            %dy_2 = arith.mulf %dy, %dy : f32
            %dz_2 = arith.mulf %dz, %dz : f32
            %dxy_2 = arith.addf %dx_2, %dy_2 : f32
            %r2 = arith.addf %dxy_2, %dz_2 : f32

            %near = arith.cmpf olt, %r2, %limit2 : f32
            %other = arith.cmpi ne, %p, %q : index
            %neighbor = arith.andi %near, %other : i1

              %fits = arith.cmpi ult, %count, %row_width : index
              %keep = arith.andi %neighbor, %fits : i1
              scf.if %keep {
                %j32 = memref.load %order[%q] : memref<?xi32, 1>
                memref.store %j32, %index[%i, %count] : memref<?x?xi32, 1>
              }

            %more = arith.addi %count, %i1 : index
            %next = arith.select %neighbor, %more, %count : index
            scf.yield %next : index
          }
          scf.yield %after_cell : index
        }
        scf.yield %after_x : index
      }

      // A thread that searches for the whole particle has counted its
      // neighbors.
      %whole = arith.cmpi eq, %split, %i0 : index
      scf.if %whole {
        %found32 = arith.index_cast %found : index to i32
        memref.store %found32, %counts[%i] : memref<?xi32, 1>
      }
    }
    gpu.terminator
  }

  //===--------------------------------------------------------------------===//
  // The largest count, and the counts limited to the width of a row
  //===--------------------------------------------------------------------===//

  // One thread for each chunk of 256 particles.
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_chunks, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
    %base = arith.muli %bx, %block : index
    %b = arith.addi %base, %tx : index
    %inside = arith.cmpi ult, %b, %chunks : index
    scf.if %inside {
      %i1 = arith.constant 1 : index
      %none = arith.constant 0 : i32
      %width32 = arith.index_cast %row_width : index to i32
      %begin = arith.muli %b, %chunk : index
      %full = arith.addi %begin, %chunk : index
      %short = arith.cmpi ult, %n, %full : index
      %end = arith.select %short, %n, %full : index
      %largest = scf.for %i = %begin to %end step %i1
          iter_args(%max = %none) -> (i32) {
        %count = memref.load %counts[%i] : memref<?xi32, 1>
        %over = arith.cmpi sgt, %count, %width32 : i32
        %limited = arith.select %over, %width32, %count : i32
        memref.store %limited, %counts[%i] : memref<?xi32, 1>
        %more = arith.cmpi sgt, %count, %max : i32
        %larger = arith.select %more, %count, %max : i32
        scf.yield %larger : i32
      }
      memref.store %largest, %partial[%b] : memref<?xi32, 1>
    }
    gpu.terminator
  }

  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %none = arith.constant 0 : i32
    %largest = scf.for %b = %i0 to %chunks step %i1
        iter_args(%max = %none) -> (i32) {
      %value = memref.load %partial[%b] : memref<?xi32, 1>
      %more = arith.cmpi sgt, %value, %max : i32
      %larger = arith.select %more, %value, %max : i32
      scf.yield %larger : i32
    }
    memref.store %largest, %result[%i0] : memref<1xi32, 1>
    gpu.terminator
  }

  %t0 = gpu.wait async
  %t1 = gpu.memcpy async [%t0] %host, %result : memref<1xi32>, memref<1xi32, 1>
  gpu.wait [%t1]
  %largest32 = memref.load %host[%c0] : memref<1xi32>
  %largest = arith.index_cast %largest32 : i32 to index

  // The lowering of `gpu.dealloc` takes a buffer without a memory space.
  %key0 = memref.memory_space_cast %key
      : memref<?xi32, 1> to memref<?xi32>
  gpu.dealloc %key0 : memref<?xi32>
  %order0 = memref.memory_space_cast %order
      : memref<?xi32, 1> to memref<?xi32>
  gpu.dealloc %order0 : memref<?xi32>
  %wrapped0 = memref.memory_space_cast %wrapped
      : memref<?x3xf32, 1> to memref<?x3xf32>
  gpu.dealloc %wrapped0 : memref<?x3xf32>
  %sorted0 = memref.memory_space_cast %sorted
      : memref<?x3xf32, 1> to memref<?x3xf32>
  gpu.dealloc %sorted0 : memref<?x3xf32>
  %held0 = memref.memory_space_cast %held
      : memref<?xi32, 1> to memref<?xi32>
  gpu.dealloc %held0 : memref<?xi32>
  %start0 = memref.memory_space_cast %start
      : memref<?xi32, 1> to memref<?xi32>
  gpu.dealloc %start0 : memref<?xi32>
  %cursor0 = memref.memory_space_cast %cursor
      : memref<?xi32, 1> to memref<?xi32>
  gpu.dealloc %cursor0 : memref<?xi32>
  %cell_sums0 = memref.memory_space_cast %cell_sums
      : memref<?xi32, 1> to memref<?xi32>
  gpu.dealloc %cell_sums0 : memref<?xi32>
  %part0 = memref.memory_space_cast %part
      : memref<?xi32, 1> to memref<?xi32>
  gpu.dealloc %part0 : memref<?xi32>
  %partial0 = memref.memory_space_cast %partial
      : memref<?xi32, 1> to memref<?xi32>
  gpu.dealloc %partial0 : memref<?xi32>
  %result0 = memref.memory_space_cast %result
      : memref<1xi32, 1> to memref<1xi32>
  gpu.dealloc %result0 : memref<1xi32>
  return %largest : index
}
