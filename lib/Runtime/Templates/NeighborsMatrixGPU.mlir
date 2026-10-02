// Builds a neighbor matrix on a device by binning the particles into cells.
//
// This is a template. The compiler adds it to a module that builds neighbor
// structures on a device, where it is specialized like any other code.
//
// Binning the particles into cells to find their neighbors is the method of
// linked cells [Quentrec1973]; a structure built with a skin and reused
// while no particle has moved more than half of it is the list of Verlet
// [Verlet1967]; displacements are taken in the minimum image
// [AllenTildesley2017]. The keys are those of docs/references.md.
//
// The buffers are on the device. The simulation cell is periodic: for
// @mdrt_gpu_build_neighbors_matrix orthorhombic, with edge lengths `box`,
// and for @mdrt_gpu_build_neighbors_matrix_triclinic the lower-triangular
// cell of docs/triclinic-m2.md. On return, `order[p]` is the particle
// at the place `p` of the order of the cells, row `p` of `index` holds the
// places of the particles within `reach` of that particle, and `counts[p]`
// holds their number, limited to the width of a row. The loops over pairs
// run in this order, whose places are next to one another in space (D86).
// The results are the largest number of neighbors that a particle has,
// which may exceed the width (the caller can then tell that a row was too
// narrow), and the number of positions that are not numbers, or are beyond
// any cell, which get no cell and no place (D107): the caller stops the
// run.
//
// The matrix is that of the template for the host in the order of the
// cells: row `p` is its row `order[p]`, and an entry `q` is its entry
// `order[q]`. See there for the cells, for the copy of the positions that
// the search works on, and for the margin of the search.
//
// A kernel runs one thread per item, in blocks of 128 threads. The threads
// beyond the last item do nothing.
//
// The particles of a cell are sorted by index after they are binned: the
// order in which threads take their slots is not fixed, and the order of
// the neighbors decides the order in which forces are added up.
//
// The search has a warp of 32 threads for each particle. The lanes test 32
// particles of a run of cells at once, which a thread for each particle
// would test one after the other, and a ballot gives each neighbor its
// place in the row in the order in which that thread would find it. The
// excluded pairs of `excluded`, an incidence structure of pairs (a buffer
// with no rows where there are none), are entered as the particle itself,
// which the loops over pairs skip.

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
// A search visits rows of cells and tests the particles in them, 32 at a
// time, a warp for each particle. Narrow cells have fewer particles to
// test and more rows to visit; a row costs as much as one step of 32
// tests, and the last step of a row is half empty on average. The width is
// the one with the least cost for the cells that it gives in this cell.
func.func private @mdrt_gpu_cell_width(%count: index, %box: vector<3xf64>,
    %reach: f64, %least: f64) -> f64 {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %lx = vector.extract %box[0] : f64 from vector<3xf64>
  %ly = vector.extract %box[1] : f64 from vector<3xf64>
  %lz = vector.extract %box[2] : f64 from vector<3xf64>
  %wide_count = arith.index_cast %count : index to i64
  %particles = arith.sitofp %wide_count : i64 to f64
  // A row: the visit, and the half-empty last step.
  %row_cost = arith.constant 1.5 : f64
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
    // The particles of a row of cells, and the steps of 32 that it takes.
    %real_span = arith.divf %real_visited, %real_rows : f64
    %share = arith.divf %real_span, %real_cells : f64
    %per_row = arith.mulf %particles, %share : f64
    %lanes = arith.constant 32.0 : f64
    %steps = arith.divf %per_row, %lanes : f64
    %row_steps = arith.addf %steps, %row_cost : f64
    %total = arith.mulf %real_rows, %row_steps : f64

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

// The other member of the excluded pair `k` of the particle `i` in the
// incidence structure `excluded`: the member at place 1 − s, for the place
// s of the particle. A row holds the number of pairs, then for each its
// number, the place of the particle, and its two members.
func.func private @mdrt_gpu_excluded_partner(%excluded: memref<?x?xi32, 1>,
    %i: index, %k: index) -> i32 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0 : i32
  %entry0 = arith.muli %k, %c4 : index
  %entry = arith.addi %entry0, %c1 : index
  %place_column = arith.addi %entry, %c1 : index
  %place = memref.load %excluded[%i, %place_column] : memref<?x?xi32, 1>
  %at_first = arith.cmpi eq, %place, %zero : i32
  %first = arith.addi %entry, %c2 : index
  %second = arith.addi %entry, %c3 : index
  %column = arith.select %at_first, %second, %first : index
  %partner = memref.load %excluded[%i, %column] : memref<?x?xi32, 1>
  return %partner : i32
}

// The offsets of the cells from the numbers of particles that `held` counts,
// the particles in the order of the cells, sorted by index within each, and
// their positions `wrapped` in that order, in `sorted`. `key` holds the cell
// of each particle, or −1 for none.
func.func private @mdrt_gpu_matrix_sort(
    %key: memref<?xi32, 1>, %wrapped: memref<?x3xf32, 1>,
    %sorted: memref<?x3xf32, 1>, %held: memref<?xi32, 1>,
    %start: memref<?xi32, 1>, %cursor: memref<?xi32, 1>,
    %cell_sums: memref<?xi32, 1>, %order: memref<?xi32, 1>, %n: index,
    %cells: index, %cell_chunks: index) {
  %c1 = arith.constant 1 : index
  %block = arith.constant 128 : index
  %chunk = arith.constant 256 : index
  %grid_n = call @mdrt_gpu_grid(%n, %block) : (index, index) -> index
  %grid_cells = call @mdrt_gpu_grid(%cells, %block) : (index, index) -> index
  %grid_cell_chunks = call @mdrt_gpu_grid(%cell_chunks, %block)
      : (index, index) -> index

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
      %zero_key = arith.constant 0 : i32
      %has_cell = arith.cmpi sge, %k32, %zero_key : i32
      scf.if %has_cell {
      %k = arith.index_cast %k32 : i32 to index
      // A relaxed atomic at the scope of the device (see PMEGPU.mlir).
      %rx2_base = memref.extract_aligned_pointer_as_index %cursor : memref<?xi32, 1> -> index
      %rx2_bi = arith.index_cast %rx2_base : index to i64
      %rx2_ki = arith.index_cast %k : index to i64
      %rx2_four = arith.constant 4 : i64
      %rx2_off = arith.muli %rx2_ki, %rx2_four : i64
      %rx2_addr = arith.addi %rx2_bi, %rx2_off : i64
      %rx2_ptr = llvm.inttoptr %rx2_addr : i64 to !llvm.ptr<1>
      %slot32 = llvm.atomicrmw add %rx2_ptr, %one syncscope("device") monotonic : !llvm.ptr<1>, i32
      %slot = arith.index_cast %slot32 : i32 to index
      %narrow = arith.index_cast %i : index to i32
      memref.store %narrow, %order[%slot] : memref<?xi32, 1>
      }
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
  return
}

func.func private @mdrt_gpu_build_neighbors_matrix(
    %x: memref<?x3xf64, 1>, %box: vector<3xf64>, %reach: f64,
    %cell_width: f64, %excluded: memref<?x?xi32, 1>,
    %counts: memref<?xi32, 1>, %index: memref<?x?xi32, 1>,
    %order: memref<?xi32, 1>) -> (index, index) {
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

  // The search has a warp of 32 threads for each particle.
  %rows_within = arith.muli %span_y, %span_z : index
  %warp_threads = arith.constant 32 : index
  %threads = arith.muli %n, %warp_threads : index
  %false = arith.constant false

  %grid_n = call @mdrt_gpu_grid(%n, %block) : (index, index) -> index
  %grid_cells = call @mdrt_gpu_grid(%cells, %block) : (index, index) -> index
  %grid_cell_chunks = call @mdrt_gpu_grid(%cell_chunks, %block)
      : (index, index) -> index
  %grid_warps = call @mdrt_gpu_grid(%threads, %block)
      : (index, index) -> index

  %cells1 = arith.addi %cells, %c1 : index
  %key = gpu.alloc (%n) : memref<?xi32, 1>
  %wrapped = gpu.alloc (%n) : memref<?x3xf32, 1>
  %sorted = gpu.alloc (%n) : memref<?x3xf32, 1>
  %held = gpu.alloc (%cells) : memref<?xi32, 1>
  %start = gpu.alloc (%cells1) : memref<?xi32, 1>
  %cursor = gpu.alloc (%cells) : memref<?xi32, 1>
  %cell_sums = gpu.alloc (%cell_chunks) : memref<?xi32, 1>
  %result = gpu.alloc () : memref<2xi32, 1>
  %host = memref.alloca() : memref<2xi32>

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
      // The largest count, which the search raises.
      %c0_first = arith.constant 0 : index
      %first_cell = arith.cmpi eq, %c, %c0_first : index
      scf.if %first_cell {
        memref.store %none, %result[%c0_first] : memref<2xi32, 1>
        %c1_second = arith.constant 1 : index
        memref.store %none, %result[%c1_second] : memref<2xi32, 1>
      }
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
      // No particle at any place yet: a place that stays empty, for want
      // of a particle with a position, is skipped by the search.
      %no_particle = arith.constant -1 : i32
      memref.store %no_particle, %order[%i] : memref<?xi32, 1>
      // A position that is not a number, or is beyond any cell, gets no
      // cell, and is counted (D107).
      %sum_xy = arith.addf %xi, %yi : f64
      %sum_xyz = arith.addf %sum_xy, %zi : f64
      %not_number = arith.cmpf uno, %sum_xyz, %sum_xyz : f64
      %magnitude = math.absf %sum_xyz : f64
      %far_off = arith.constant 1.0e100 : f64
      %huge = arith.cmpf ogt, %magnitude, %far_off : f64
      %bad = arith.ori %not_number, %huge : i1
      scf.if %bad {
        memref.store %no_particle, %key[%i] : memref<?xi32, 1>
        %nn_base = memref.extract_aligned_pointer_as_index %result : memref<2xi32, 1> -> index
        %nn_bi = arith.index_cast %nn_base : index to i64
        %nn_four = arith.constant 4 : i64
        %nn_addr = arith.addi %nn_bi, %nn_four : i64
        %nn_ptr = llvm.inttoptr %nn_addr : i64 to !llvm.ptr<1>
        %nn_old = llvm.atomicrmw add %nn_ptr, %one syncscope("device") monotonic : !llvm.ptr<1>, i32
      } else {
        memref.store %k32, %key[%i] : memref<?xi32, 1>
        // A relaxed atomic at the scope of the device (see PMEGPU.mlir).
        %rx1_base = memref.extract_aligned_pointer_as_index %held : memref<?xi32, 1> -> index
        %rx1_bi = arith.index_cast %rx1_base : index to i64
        %rx1_ki = arith.index_cast %k : index to i64
        %rx1_four = arith.constant 4 : i64
        %rx1_off = arith.muli %rx1_ki, %rx1_four : i64
        %rx1_addr = arith.addi %rx1_bi, %rx1_off : i64
        %rx1_ptr = llvm.inttoptr %rx1_addr : i64 to !llvm.ptr<1>
        %old = llvm.atomicrmw add %rx1_ptr, %one syncscope("device") monotonic : !llvm.ptr<1>, i32
      }
    }
    gpu.terminator
  }

  // The offsets of the cells, the particles in the order of the cells, and
  // their positions in that order.
  call @mdrt_gpu_matrix_sort(%key, %wrapped, %sorted, %held, %start, %cursor,
                             %cell_sums, %order, %n, %cells, %cell_chunks)
      : (memref<?xi32, 1>, memref<?x3xf32, 1>, memref<?x3xf32, 1>,
         memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>,
         memref<?xi32, 1>, memref<?xi32, 1>, index, index, index) -> ()

  //===--------------------------------------------------------------------===//
  // Neighbors
  //===--------------------------------------------------------------------===//

  // A warp for each particle. Its lanes test 32 particles of a run of
  // cells at once, and a ballot gives each neighbor its place in the row,
  // in the order in which one thread would find them. A neighbor that is
  // an excluded pair of `excluded` is entered as the particle itself,
  // which the loops over pairs skip. The count goes on beyond the width
  // of the row, without writing.
  //
  // The kernel counts in i32, which holds every number of a particle, a
  // cell, or an entry of a row: with 64-bit indices it needed more
  // registers than a thread has and spilled.
  %n32 = arith.index_cast %n : index to i32
  %nx32 = arith.index_cast %nx : index to i32
  %ny32 = arith.index_cast %ny : index to i32
  %nz32 = arith.index_cast %nz : index to i32
  %span_x32 = arith.index_cast %span_x : index to i32
  %span_y32 = arith.index_cast %span_y : index to i32
  %rows32 = arith.index_cast %rows_within : index to i32
  %first_x32 = arith.index_cast %first_x : index to i32
  %first_y32 = arith.index_cast %first_y : index to i32
  %first_z32 = arith.index_cast %first_z : index to i32
  %row_width32 = arith.index_cast %row_width : index to i32
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_warps, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
    %base = arith.muli %bx, %block : index
    %t = arith.addi %base, %tx : index
    %t32 = arith.index_cast %t : index to i32
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %zero32 = arith.constant 0 : i32
    %one32 = arith.constant 1 : i32
    %two32 = arith.constant 2 : i32
    %lanes32 = arith.constant 32 : i32
    %none32 = arith.constant -1 : i32
    %p32 = arith.divui %t32, %lanes32 : i32
    %lane32 = arith.remui %t32, %lanes32 : i32
    // The warps past the last particle stop together.
    %inside = arith.cmpi ult, %p32, %n32 : i32
    scf.if %inside {
      %p = arith.index_cast %p32 : i32 to index
      %i32 = memref.load %order[%p] : memref<?xi32, 1>
      %i = arith.index_cast %i32 : i32 to index
      %zero_placed = arith.constant 0 : i32
      %placed = arith.cmpi sge, %i32, %zero_placed : i32
      scf.if %placed {
      %xi = memref.load %sorted[%p, %i0] : memref<?x3xf32, 1>
      %yi = memref.load %sorted[%p, %i1] : memref<?x3xf32, 1>
      %zi = memref.load %sorted[%p, %i2] : memref<?x3xf32, 1>

      %k32 = memref.load %key[%i] : memref<?xi32, 1>
      %cx = arith.remui %k32, %nx32 : i32
      %rest = arith.divui %k32, %nx32 : i32
      %cy = arith.remui %rest, %ny32 : i32
      %cz = arith.divui %rest, %ny32 : i32

      // The excluded pairs of the particle, if the structure has any.
      %rows_excluded = memref.dim %excluded, %i0 : memref<?x?xi32, 1>
      %has_excluded = arith.cmpi ult, %i, %rows_excluded : index
      %num_excluded = scf.if %has_excluded -> (i32) {
        %e32 = memref.load %excluded[%i, %i0] : memref<?x?xi32, 1>
        scf.yield %e32 : i32
      } else {
        scf.yield %zero32 : i32
      }

      // The other member of each excluded pair of the particle: lane k
      // holds that of the pair k, for the first 32; a neighbor is compared
      // with them by shuffles, and with the rest, if any, in memory.
      %lane_excluded = arith.cmpi ult, %lane32, %num_excluded : i32
      %my_partner = scf.if %lane_excluded -> (i32) {
        %lane = arith.index_cast %lane32 : i32 to index
        %mine = func.call @mdrt_gpu_excluded_partner(%excluded, %i, %lane)
            : (memref<?x?xi32, 1>, index, index) -> i32
        scf.yield %mine : i32
      } else {
        scf.yield %none32 : i32
      }
      %in_registers = arith.minui %num_excluded, %lanes32 : i32
      // The least and the greatest number of a partner: a neighbor outside
      // them is not excluded, and needs no comparison. With more than 32
      // pairs every neighbor is compared.
      %all_low = arith.constant 0 : i32
      %all_high = arith.constant 2147483647 : i32
      %low0, %high0 = scf.for %e = %zero32 to %in_registers step %one32
          iter_args(%lo = %all_high, %hi = %all_low) -> (i32, i32) : i32 {
        %partner, %valid = gpu.shuffle idx %my_partner, %e, %lanes32 : i32
        %lo1 = arith.minsi %lo, %partner : i32
        %hi1 = arith.maxsi %hi, %partner : i32
        scf.yield %lo1, %hi1 : i32, i32
      }
      %overflow = arith.cmpi ugt, %num_excluded, %lanes32 : i32
      %low = arith.select %overflow, %all_low, %low0 : i32
      %high = arith.select %overflow, %all_high, %high0 : i32

      %lane_bit = arith.shli %one32, %lane32 : i32
      %below_mask = arith.subi %lane_bit, %one32 : i32

      %found = scf.for %r = %zero32 to %rows32 step %one32
          iter_args(%count_r = %zero32) -> (i32) : i32 {
        %oz = arith.divui %r, %span_y32 : i32
        %oy = arith.remui %r, %span_y32 : i32
        %sz0 = arith.addi %cz, %first_z32 : i32
        %sz1 = arith.addi %sz0, %oz : i32
        %nz_cell = arith.remui %sz1, %nz32 : i32
        %sy0 = arith.addi %cy, %first_y32 : i32
        %sy1 = arith.addi %sy0, %oy : i32
        %ny_cell = arith.remui %sy1, %ny32 : i32

        // The cells of the row are next to one another in the order of
        // the cells, so the warp reads them as one run, or as two where
        // the row goes around the edge of the cell.
        %zy = arith.muli %nz_cell, %ny32 : i32
        %row = arith.addi %zy, %ny_cell : i32
        %row_first = arith.muli %row, %nx32 : i32
        %sx0 = arith.addi %cx, %first_x32 : i32
        %x_begin = arith.remui %sx0, %nx32 : i32
        %x_end = arith.addi %x_begin, %span_x32 : i32
        %around = arith.cmpi ugt, %x_end, %nx32 : i32
        %x_stop = arith.select %around, %nx32, %x_end : i32
        %x_rest0 = arith.subi %x_end, %nx32 : i32
        %x_rest = arith.select %around, %x_rest0, %zero32 : i32

        %after_x = scf.for %part_x = %zero32 to %two32 step %one32
            iter_args(%count_x = %count_r) -> (i32) : i32 {
          %second = arith.cmpi ne, %part_x, %zero32 : i32
          %from_x = arith.select %second, %zero32, %x_begin : i32
          %to_x = arith.select %second, %x_rest, %x_stop : i32
          %from_cell32 = arith.addi %row_first, %from_x : i32
          %to_cell32 = arith.addi %row_first, %to_x : i32
          %from_cell = arith.index_cast %from_cell32 : i32 to index
          %to_cell = arith.index_cast %to_cell32 : i32 to index
          %begin = memref.load %start[%from_cell] : memref<?xi32, 1>
          %end = memref.load %start[%to_cell] : memref<?xi32, 1>
          %last = arith.subi %end, %one32 : i32

          %after_cell = scf.for %q0 = %begin to %end step %lanes32
              iter_args(%count = %count_x) -> (i32) : i32 {
            %q32 = arith.addi %q0, %lane32 : i32
            %in_run = arith.cmpi slt, %q32, %end : i32
            %qc32 = arith.minsi %q32, %last : i32
            %qc = arith.index_cast %qc32 : i32 to index
            %xj = memref.load %sorted[%qc, %i0] : memref<?x3xf32, 1>
            %yj = memref.load %sorted[%qc, %i1] : memref<?x3xf32, 1>
            %zj = memref.load %sorted[%qc, %i2] : memref<?x3xf32, 1>

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
            %other = arith.cmpi ne, %p32, %q32 : i32
            %near_other = arith.andi %near, %other : i1
            %neighbor = arith.andi %near_other, %in_run : i1

            %ballot = gpu.ballot %neighbor : i32
            %before_bits = arith.andi %ballot, %below_mask : i32
            %before = math.ctpop %before_bits : i32
            %slot = arith.addi %count, %before : i32
            %fits = arith.cmpi ult, %slot, %row_width32 : i32
            %keep = arith.andi %neighbor, %fits : i1

            // Whether the neighbor of each lane is excluded: all lanes
            // shuffle when any lane has found one within the numbers of the
            // partners.
            %j32 = scf.if %neighbor -> (i32) {
              %q = arith.index_cast %q32 : i32 to index
              %j_found = memref.load %order[%q] : memref<?xi32, 1>
              scf.yield %j_found : i32
            } else {
              scf.yield %none32 : i32
            }
            %above = arith.cmpi sge, %j32, %low : i32
            %under = arith.cmpi sle, %j32, %high : i32
            %between = arith.andi %above, %under : i1
            %maybe = arith.andi %neighbor, %between : i1
            %maybe_bits = gpu.ballot %maybe : i32
            %any = arith.cmpi ne, %maybe_bits, %zero32 : i32
            %shuffled = scf.if %any -> (i1) {
              %hit = scf.for %e = %zero32 to %in_registers step %one32
                  iter_args(%found_e = %false) -> (i1) : i32 {
                %partner, %valid = gpu.shuffle idx %my_partner, %e, %lanes32 : i32
                %same = arith.cmpi eq, %partner, %j32 : i32
                %or = arith.ori %found_e, %same : i1
                scf.yield %or : i1
              }
              scf.yield %hit : i1
            } else {
              scf.yield %false : i1
            }
            scf.if %keep {
              %is_excluded = scf.for %e = %in_registers to %num_excluded
                  step %one32 iter_args(%found_e = %shuffled) -> (i1) : i32 {
                %e_index = arith.index_cast %e : i32 to index
                %partner = func.call @mdrt_gpu_excluded_partner(%excluded, %i, %e_index)
                    : (memref<?x?xi32, 1>, index, index) -> i32
                %same = arith.cmpi eq, %partner, %j32 : i32
                %any_e = arith.ori %found_e, %same : i1
                scf.yield %any_e : i1
              }
              %entered = arith.select %is_excluded, %p32, %q32 : i32
              %slot_index = arith.index_cast %slot : i32 to index
              memref.store %entered, %index[%p, %slot_index] : memref<?x?xi32, 1>
            }

            %found_here = math.ctpop %ballot : i32
            %next = arith.addi %count, %found_here : i32
            scf.yield %next : i32
          }
          scf.yield %after_cell : i32
        }
        scf.yield %after_x : i32
      }

      // The count limited to the width of a row, and the largest count,
      // which tells the caller that a row was too narrow. The largest of
      // the counts does not depend on the order of the atomics.
      %writer = arith.cmpi eq, %lane32, %zero32 : i32
      scf.if %writer {
        %limited = arith.minui %found, %row_width32 : i32
        memref.store %limited, %counts[%p] : memref<?xi32, 1>
        // A relaxed atomic at the scope of the device (see PMEGPU.mlir).
        %rm_base = memref.extract_aligned_pointer_as_index %result : memref<2xi32, 1> -> index
        %rm_addr = arith.index_cast %rm_base : index to i64
        %rm_ptr = llvm.inttoptr %rm_addr : i64 to !llvm.ptr<1>
        %rm_old = llvm.atomicrmw umax %rm_ptr, %found syncscope("device") monotonic : !llvm.ptr<1>, i32
      }
      }
    }
    gpu.terminator
  }

  %t0 = gpu.wait async
  %t1 = gpu.memcpy async [%t0] %host, %result : memref<2xi32>, memref<2xi32, 1>
  gpu.wait [%t1]
  %largest32 = memref.load %host[%c0] : memref<2xi32>
  %largest = arith.index_cast %largest32 : i32 to index
  %not_numbers32 = memref.load %host[%c1] : memref<2xi32>
  %not_numbers = arith.index_cast %not_numbers32 : i32 to index

  // The lowering of `gpu.dealloc` takes a buffer without a memory space.
  %key0 = memref.memory_space_cast %key
      : memref<?xi32, 1> to memref<?xi32>
  gpu.dealloc %key0 : memref<?xi32>
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
  %result0 = memref.memory_space_cast %result
      : memref<2xi32, 1> to memref<2xi32>
  gpu.dealloc %result0 : memref<2xi32>
  return %largest, %not_numbers : index, index
}

// The neighbor matrix of a triclinic cell, `box` = (a_x, b_y, c_z, b_x,
// c_x, c_y), with the widths of the cell between its faces, `widths`: the
// build of the orthorhombic cell in the fractional coordinates, which the
// positions are wrapped in, with the minimum image of the triclinic cell
// (docs/triclinic-m2.md). It holds every pair within `reach` when `reach`
// is at most half of the least of a_x, b_y, c_z.
func.func private @mdrt_gpu_build_neighbors_matrix_triclinic(
    %x: memref<?x3xf64, 1>, %box: vector<6xf64>, %widths: vector<3xf64>,
    %reach: f64, %cell_width: f64, %excluded: memref<?x?xi32, 1>,
    %counts: memref<?xi32, 1>, %index: memref<?x?xi32, 1>,
    %order: memref<?xi32, 1>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %block = arith.constant 128 : index
  %chunk = arith.constant 256 : index

  %n = memref.dim %x, %c0 : memref<?x3xf64, 1>
  %row_width = memref.dim %index, %c1 : memref<?x?xi32, 1>

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
  %nx = call @mdrt_gpu_cell_count(%width_x, %cell_width) : (f64, f64) -> index
  %ny = call @mdrt_gpu_cell_count(%width_y, %cell_width) : (f64, f64) -> index
  %nz = call @mdrt_gpu_cell_count(%width_z, %cell_width) : (f64, f64) -> index
  %nxy = arith.muli %nx, %ny : index
  %cells = arith.muli %nxy, %nz : index
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
  %narrow_bx = arith.truncf %tbx : f64 to f32
  %narrow_cx = arith.truncf %tcx : f64 to f32
  %narrow_cy = arith.truncf %tcy : f64 to f32
  // The margin is more than the rounding of a distance between positions in
  // the cell.
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

  // The cells within reach.
  %range_x = call @mdrt_gpu_cell_range(%width_x, %nx, %reach)
      : (f64, index, f64) -> index
  %range_y = call @mdrt_gpu_cell_range(%width_y, %ny, %reach)
      : (f64, index, f64) -> index
  %range_z = call @mdrt_gpu_cell_range(%width_z, %nz, %reach)
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

  // The search has a warp of 32 threads for each particle.
  %rows_within = arith.muli %span_y, %span_z : index
  %warp_threads = arith.constant 32 : index
  %threads = arith.muli %n, %warp_threads : index
  %false = arith.constant false

  %grid_n = call @mdrt_gpu_grid(%n, %block) : (index, index) -> index
  %grid_cells = call @mdrt_gpu_grid(%cells, %block) : (index, index) -> index
  %grid_cell_chunks = call @mdrt_gpu_grid(%cell_chunks, %block)
      : (index, index) -> index
  %grid_warps = call @mdrt_gpu_grid(%threads, %block)
      : (index, index) -> index

  %cells1 = arith.addi %cells, %c1 : index
  %key = gpu.alloc (%n) : memref<?xi32, 1>
  %wrapped = gpu.alloc (%n) : memref<?x3xf32, 1>
  %sorted = gpu.alloc (%n) : memref<?x3xf32, 1>
  %held = gpu.alloc (%cells) : memref<?xi32, 1>
  %start = gpu.alloc (%cells1) : memref<?xi32, 1>
  %cursor = gpu.alloc (%cells) : memref<?xi32, 1>
  %cell_sums = gpu.alloc (%cell_chunks) : memref<?xi32, 1>
  %result = gpu.alloc () : memref<2xi32, 1>
  %host = memref.alloca() : memref<2xi32>

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
      // The largest count, which the search raises.
      %c0_first = arith.constant 0 : index
      %first_cell = arith.cmpi eq, %c, %c0_first : index
      scf.if %first_cell {
        memref.store %none, %result[%c0_first] : memref<2xi32, 1>
        %c1_second = arith.constant 1 : index
        memref.store %none, %result[%c1_second] : memref<2xi32, 1>
      }
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

      // s = x H⁻¹, taken into [0, 1); its cell, of the fractional
      // coordinates; and the position s H in the cell, which the search
      // takes: see the template for the host. Rounding can place a
      // particle at an edge one cell too far.
      %frac_sz = arith.mulf %zi, %ilz : f64
      %frac_sz_cy = arith.mulf %frac_sz, %tcy : f64
      %y_rest = arith.subf %yi, %frac_sz_cy : f64
      %frac_sy = arith.mulf %y_rest, %ily : f64
      %frac_sy_bx = arith.mulf %frac_sy, %tbx : f64
      %frac_sz_cx = arith.mulf %frac_sz, %tcx : f64
      %x_rest0 = arith.subf %xi, %frac_sy_bx : f64
      %x_rest = arith.subf %x_rest0, %frac_sz_cx : f64
      %frac_sx = arith.mulf %x_rest, %ilx : f64
      %frac_sx_floor = math.floor %frac_sx : f64
      %fx = arith.subf %frac_sx, %frac_sx_floor : f64
      %frac_sy_floor = math.floor %frac_sy : f64
      %fy = arith.subf %frac_sy, %frac_sy_floor : f64
      %frac_sz_floor = math.floor %frac_sz : f64
      %fz = arith.subf %frac_sz, %frac_sz_floor : f64
      %x4 = arith.index_cast %nx : index to i64
      %x5 = arith.sitofp %x4 : i64 to f64
      %x6 = arith.mulf %fx, %x5 : f64
      %x7 = arith.fptosi %x6 : f64 to i64
      %x8 = arith.index_cast %x7 : i64 to index
      %xl = arith.subi %nx, %i1 : index
      %xb = arith.minsi %x8, %xl : index
      %cx = arith.maxsi %xb, %i0 : index
      %y4 = arith.index_cast %ny : index to i64
      %y5 = arith.sitofp %y4 : i64 to f64
      %y6 = arith.mulf %fy, %y5 : f64
      %y7 = arith.fptosi %y6 : f64 to i64
      %y8 = arith.index_cast %y7 : i64 to index
      %yl = arith.subi %ny, %i1 : index
      %yb = arith.minsi %y8, %yl : index
      %cy = arith.maxsi %yb, %i0 : index
      %z4 = arith.index_cast %nz : index to i64
      %z5 = arith.sitofp %z4 : i64 to f64
      %z6 = arith.mulf %fz, %z5 : f64
      %z7 = arith.fptosi %z6 : f64 to i64
      %z8 = arith.index_cast %z7 : i64 to index
      %zl = arith.subi %nz, %i1 : index
      %zb = arith.minsi %z8, %zl : index
      %cz = arith.maxsi %zb, %i0 : index
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
      memref.store %narrow_x, %wrapped[%i, %i0] : memref<?x3xf32, 1>
      memref.store %narrow_y, %wrapped[%i, %i1] : memref<?x3xf32, 1>
      memref.store %narrow_z, %wrapped[%i, %i2] : memref<?x3xf32, 1>

      %zy = arith.muli %cz, %ny : index
      %row = arith.addi %zy, %cy : index
      %rows = arith.muli %row, %nx : index
      %k = arith.addi %rows, %cx : index
      %k32 = arith.index_cast %k : index to i32
      // No particle at any place yet: a place that stays empty, for want
      // of a particle with a position, is skipped by the search.
      %no_particle = arith.constant -1 : i32
      memref.store %no_particle, %order[%i] : memref<?xi32, 1>
      // A position that is not a number, or is beyond any cell, gets no
      // cell, and is counted (D107).
      %sum_xy = arith.addf %xi, %yi : f64
      %sum_xyz = arith.addf %sum_xy, %zi : f64
      %not_number = arith.cmpf uno, %sum_xyz, %sum_xyz : f64
      %magnitude = math.absf %sum_xyz : f64
      %far_off = arith.constant 1.0e100 : f64
      %huge = arith.cmpf ogt, %magnitude, %far_off : f64
      %bad = arith.ori %not_number, %huge : i1
      scf.if %bad {
        memref.store %no_particle, %key[%i] : memref<?xi32, 1>
        %nn_base = memref.extract_aligned_pointer_as_index %result : memref<2xi32, 1> -> index
        %nn_bi = arith.index_cast %nn_base : index to i64
        %nn_four = arith.constant 4 : i64
        %nn_addr = arith.addi %nn_bi, %nn_four : i64
        %nn_ptr = llvm.inttoptr %nn_addr : i64 to !llvm.ptr<1>
        %nn_old = llvm.atomicrmw add %nn_ptr, %one syncscope("device") monotonic : !llvm.ptr<1>, i32
      } else {
        memref.store %k32, %key[%i] : memref<?xi32, 1>
        // A relaxed atomic at the scope of the device (see PMEGPU.mlir).
        %rx1_base = memref.extract_aligned_pointer_as_index %held : memref<?xi32, 1> -> index
        %rx1_bi = arith.index_cast %rx1_base : index to i64
        %rx1_ki = arith.index_cast %k : index to i64
        %rx1_four = arith.constant 4 : i64
        %rx1_off = arith.muli %rx1_ki, %rx1_four : i64
        %rx1_addr = arith.addi %rx1_bi, %rx1_off : i64
        %rx1_ptr = llvm.inttoptr %rx1_addr : i64 to !llvm.ptr<1>
        %old = llvm.atomicrmw add %rx1_ptr, %one syncscope("device") monotonic : !llvm.ptr<1>, i32
      }
    }
    gpu.terminator
  }

  // The offsets of the cells, the particles in the order of the cells, and
  // their positions in that order.
  call @mdrt_gpu_matrix_sort(%key, %wrapped, %sorted, %held, %start, %cursor,
                             %cell_sums, %order, %n, %cells, %cell_chunks)
      : (memref<?xi32, 1>, memref<?x3xf32, 1>, memref<?x3xf32, 1>,
         memref<?xi32, 1>, memref<?xi32, 1>, memref<?xi32, 1>,
         memref<?xi32, 1>, memref<?xi32, 1>, index, index, index) -> ()

  //===--------------------------------------------------------------------===//
  // Neighbors
  //===--------------------------------------------------------------------===//

  // A warp for each particle. Its lanes test 32 particles of a run of
  // cells at once, and a ballot gives each neighbor its place in the row,
  // in the order in which one thread would find them. A neighbor that is
  // an excluded pair of `excluded` is entered as the particle itself,
  // which the loops over pairs skip. The count goes on beyond the width
  // of the row, without writing.
  //
  // The kernel counts in i32, which holds every number of a particle, a
  // cell, or an entry of a row: with 64-bit indices it needed more
  // registers than a thread has and spilled.
  %n32 = arith.index_cast %n : index to i32
  %nx32 = arith.index_cast %nx : index to i32
  %ny32 = arith.index_cast %ny : index to i32
  %nz32 = arith.index_cast %nz : index to i32
  %span_x32 = arith.index_cast %span_x : index to i32
  %span_y32 = arith.index_cast %span_y : index to i32
  %rows32 = arith.index_cast %rows_within : index to i32
  %first_x32 = arith.index_cast %first_x : index to i32
  %first_y32 = arith.index_cast %first_y : index to i32
  %first_z32 = arith.index_cast %first_z : index to i32
  %row_width32 = arith.index_cast %row_width : index to i32
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_warps, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
    %base = arith.muli %bx, %block : index
    %t = arith.addi %base, %tx : index
    %t32 = arith.index_cast %t : index to i32
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %zero32 = arith.constant 0 : i32
    %one32 = arith.constant 1 : i32
    %two32 = arith.constant 2 : i32
    %lanes32 = arith.constant 32 : i32
    %none32 = arith.constant -1 : i32
    %p32 = arith.divui %t32, %lanes32 : i32
    %lane32 = arith.remui %t32, %lanes32 : i32
    // The warps past the last particle stop together.
    %inside = arith.cmpi ult, %p32, %n32 : i32
    scf.if %inside {
      %p = arith.index_cast %p32 : i32 to index
      %i32 = memref.load %order[%p] : memref<?xi32, 1>
      %i = arith.index_cast %i32 : i32 to index
      %zero_placed = arith.constant 0 : i32
      %placed = arith.cmpi sge, %i32, %zero_placed : i32
      scf.if %placed {
      %xi = memref.load %sorted[%p, %i0] : memref<?x3xf32, 1>
      %yi = memref.load %sorted[%p, %i1] : memref<?x3xf32, 1>
      %zi = memref.load %sorted[%p, %i2] : memref<?x3xf32, 1>

      %k32 = memref.load %key[%i] : memref<?xi32, 1>
      %cx = arith.remui %k32, %nx32 : i32
      %rest = arith.divui %k32, %nx32 : i32
      %cy = arith.remui %rest, %ny32 : i32
      %cz = arith.divui %rest, %ny32 : i32

      // The excluded pairs of the particle, if the structure has any.
      %rows_excluded = memref.dim %excluded, %i0 : memref<?x?xi32, 1>
      %has_excluded = arith.cmpi ult, %i, %rows_excluded : index
      %num_excluded = scf.if %has_excluded -> (i32) {
        %e32 = memref.load %excluded[%i, %i0] : memref<?x?xi32, 1>
        scf.yield %e32 : i32
      } else {
        scf.yield %zero32 : i32
      }

      // The other member of each excluded pair of the particle: lane k
      // holds that of the pair k, for the first 32; a neighbor is compared
      // with them by shuffles, and with the rest, if any, in memory.
      %lane_excluded = arith.cmpi ult, %lane32, %num_excluded : i32
      %my_partner = scf.if %lane_excluded -> (i32) {
        %lane = arith.index_cast %lane32 : i32 to index
        %mine = func.call @mdrt_gpu_excluded_partner(%excluded, %i, %lane)
            : (memref<?x?xi32, 1>, index, index) -> i32
        scf.yield %mine : i32
      } else {
        scf.yield %none32 : i32
      }
      %in_registers = arith.minui %num_excluded, %lanes32 : i32
      // The least and the greatest number of a partner: a neighbor outside
      // them is not excluded, and needs no comparison. With more than 32
      // pairs every neighbor is compared.
      %all_low = arith.constant 0 : i32
      %all_high = arith.constant 2147483647 : i32
      %low0, %high0 = scf.for %e = %zero32 to %in_registers step %one32
          iter_args(%lo = %all_high, %hi = %all_low) -> (i32, i32) : i32 {
        %partner, %valid = gpu.shuffle idx %my_partner, %e, %lanes32 : i32
        %lo1 = arith.minsi %lo, %partner : i32
        %hi1 = arith.maxsi %hi, %partner : i32
        scf.yield %lo1, %hi1 : i32, i32
      }
      %overflow = arith.cmpi ugt, %num_excluded, %lanes32 : i32
      %low = arith.select %overflow, %all_low, %low0 : i32
      %high = arith.select %overflow, %all_high, %high0 : i32

      %lane_bit = arith.shli %one32, %lane32 : i32
      %below_mask = arith.subi %lane_bit, %one32 : i32

      %found = scf.for %r = %zero32 to %rows32 step %one32
          iter_args(%count_r = %zero32) -> (i32) : i32 {
        %oz = arith.divui %r, %span_y32 : i32
        %oy = arith.remui %r, %span_y32 : i32
        %sz0 = arith.addi %cz, %first_z32 : i32
        %sz1 = arith.addi %sz0, %oz : i32
        %nz_cell = arith.remui %sz1, %nz32 : i32
        %sy0 = arith.addi %cy, %first_y32 : i32
        %sy1 = arith.addi %sy0, %oy : i32
        %ny_cell = arith.remui %sy1, %ny32 : i32

        // The cells of the row are next to one another in the order of
        // the cells, so the warp reads them as one run, or as two where
        // the row goes around the edge of the cell.
        %zy = arith.muli %nz_cell, %ny32 : i32
        %row = arith.addi %zy, %ny_cell : i32
        %row_first = arith.muli %row, %nx32 : i32
        %sx0 = arith.addi %cx, %first_x32 : i32
        %x_begin = arith.remui %sx0, %nx32 : i32
        %x_end = arith.addi %x_begin, %span_x32 : i32
        %around = arith.cmpi ugt, %x_end, %nx32 : i32
        %x_stop = arith.select %around, %nx32, %x_end : i32
        %x_rest0 = arith.subi %x_end, %nx32 : i32
        %x_rest = arith.select %around, %x_rest0, %zero32 : i32

        %after_x = scf.for %part_x = %zero32 to %two32 step %one32
            iter_args(%count_x = %count_r) -> (i32) : i32 {
          %second = arith.cmpi ne, %part_x, %zero32 : i32
          %from_x = arith.select %second, %zero32, %x_begin : i32
          %to_x = arith.select %second, %x_rest, %x_stop : i32
          %from_cell32 = arith.addi %row_first, %from_x : i32
          %to_cell32 = arith.addi %row_first, %to_x : i32
          %from_cell = arith.index_cast %from_cell32 : i32 to index
          %to_cell = arith.index_cast %to_cell32 : i32 to index
          %begin = memref.load %start[%from_cell] : memref<?xi32, 1>
          %end = memref.load %start[%to_cell] : memref<?xi32, 1>
          %last = arith.subi %end, %one32 : i32

          %after_cell = scf.for %q0 = %begin to %end step %lanes32
              iter_args(%count = %count_x) -> (i32) : i32 {
            %q32 = arith.addi %q0, %lane32 : i32
            %in_run = arith.cmpi slt, %q32, %end : i32
            %qc32 = arith.minsi %q32, %last : i32
            %qc = arith.index_cast %qc32 : i32 to index
            %xj = memref.load %sorted[%qc, %i0] : memref<?x3xf32, 1>
            %yj = memref.load %sorted[%qc, %i1] : memref<?x3xf32, 1>
            %zj = memref.load %sorted[%qc, %i2] : memref<?x3xf32, 1>

            // The minimum-image displacement of the triclinic cell, in one
            // pass along c, b, and a: see the template for the host.
            %dx0 = arith.subf %xi, %xj : f32
            %dy0 = arith.subf %yi, %yj : f32
            %dz0 = arith.subf %zi, %zj : f32
            %image_z0 = arith.mulf %dz0, %narrow_ilz : f32
            %image_z = math.roundeven %image_z0 : f32
            %shift_zx = arith.mulf %image_z, %narrow_cx : f32
            %shift_zy = arith.mulf %image_z, %narrow_cy : f32
            %shift_zz = arith.mulf %image_z, %narrow_lz : f32
            %dx1 = arith.subf %dx0, %shift_zx : f32
            %dy1 = arith.subf %dy0, %shift_zy : f32
            %dz = arith.subf %dz0, %shift_zz : f32
            %image_y0 = arith.mulf %dy1, %narrow_ily : f32
            %image_y = math.roundeven %image_y0 : f32
            %shift_yx = arith.mulf %image_y, %narrow_bx : f32
            %shift_yy = arith.mulf %image_y, %narrow_ly : f32
            %dx2 = arith.subf %dx1, %shift_yx : f32
            %dy = arith.subf %dy1, %shift_yy : f32
            %image_x0 = arith.mulf %dx2, %narrow_ilx : f32
            %image_x = math.roundeven %image_x0 : f32
            %shift_xx = arith.mulf %image_x, %narrow_lx : f32
            %dx = arith.subf %dx2, %shift_xx : f32

            %dx_2 = arith.mulf %dx, %dx : f32
            %dy_2 = arith.mulf %dy, %dy : f32
            %dz_2 = arith.mulf %dz, %dz : f32
            %dxy_2 = arith.addf %dx_2, %dy_2 : f32
            %r2 = arith.addf %dxy_2, %dz_2 : f32

            %near = arith.cmpf olt, %r2, %limit2 : f32
            %other = arith.cmpi ne, %p32, %q32 : i32
            %near_other = arith.andi %near, %other : i1
            %neighbor = arith.andi %near_other, %in_run : i1

            %ballot = gpu.ballot %neighbor : i32
            %before_bits = arith.andi %ballot, %below_mask : i32
            %before = math.ctpop %before_bits : i32
            %slot = arith.addi %count, %before : i32
            %fits = arith.cmpi ult, %slot, %row_width32 : i32
            %keep = arith.andi %neighbor, %fits : i1

            // Whether the neighbor of each lane is excluded: all lanes
            // shuffle when any lane has found one within the numbers of the
            // partners.
            %j32 = scf.if %neighbor -> (i32) {
              %q = arith.index_cast %q32 : i32 to index
              %j_found = memref.load %order[%q] : memref<?xi32, 1>
              scf.yield %j_found : i32
            } else {
              scf.yield %none32 : i32
            }
            %above = arith.cmpi sge, %j32, %low : i32
            %under = arith.cmpi sle, %j32, %high : i32
            %between = arith.andi %above, %under : i1
            %maybe = arith.andi %neighbor, %between : i1
            %maybe_bits = gpu.ballot %maybe : i32
            %any = arith.cmpi ne, %maybe_bits, %zero32 : i32
            %shuffled = scf.if %any -> (i1) {
              %hit = scf.for %e = %zero32 to %in_registers step %one32
                  iter_args(%found_e = %false) -> (i1) : i32 {
                %partner, %valid = gpu.shuffle idx %my_partner, %e, %lanes32 : i32
                %same = arith.cmpi eq, %partner, %j32 : i32
                %or = arith.ori %found_e, %same : i1
                scf.yield %or : i1
              }
              scf.yield %hit : i1
            } else {
              scf.yield %false : i1
            }
            scf.if %keep {
              %is_excluded = scf.for %e = %in_registers to %num_excluded
                  step %one32 iter_args(%found_e = %shuffled) -> (i1) : i32 {
                %e_index = arith.index_cast %e : i32 to index
                %partner = func.call @mdrt_gpu_excluded_partner(%excluded, %i, %e_index)
                    : (memref<?x?xi32, 1>, index, index) -> i32
                %same = arith.cmpi eq, %partner, %j32 : i32
                %any_e = arith.ori %found_e, %same : i1
                scf.yield %any_e : i1
              }
              %entered = arith.select %is_excluded, %p32, %q32 : i32
              %slot_index = arith.index_cast %slot : i32 to index
              memref.store %entered, %index[%p, %slot_index] : memref<?x?xi32, 1>
            }

            %found_here = math.ctpop %ballot : i32
            %next = arith.addi %count, %found_here : i32
            scf.yield %next : i32
          }
          scf.yield %after_cell : i32
        }
        scf.yield %after_x : i32
      }

      // The count limited to the width of a row, and the largest count,
      // which tells the caller that a row was too narrow. The largest of
      // the counts does not depend on the order of the atomics.
      %writer = arith.cmpi eq, %lane32, %zero32 : i32
      scf.if %writer {
        %limited = arith.minui %found, %row_width32 : i32
        memref.store %limited, %counts[%p] : memref<?xi32, 1>
        // A relaxed atomic at the scope of the device (see PMEGPU.mlir).
        %rm_base = memref.extract_aligned_pointer_as_index %result : memref<2xi32, 1> -> index
        %rm_addr = arith.index_cast %rm_base : index to i64
        %rm_ptr = llvm.inttoptr %rm_addr : i64 to !llvm.ptr<1>
        %rm_old = llvm.atomicrmw umax %rm_ptr, %found syncscope("device") monotonic : !llvm.ptr<1>, i32
      }
      }
    }
    gpu.terminator
  }

  %t0 = gpu.wait async
  %t1 = gpu.memcpy async [%t0] %host, %result : memref<2xi32>, memref<2xi32, 1>
  gpu.wait [%t1]
  %largest32 = memref.load %host[%c0] : memref<2xi32>
  %largest = arith.index_cast %largest32 : i32 to index
  %not_numbers32 = memref.load %host[%c1] : memref<2xi32>
  %not_numbers = arith.index_cast %not_numbers32 : i32 to index

  // The lowering of `gpu.dealloc` takes a buffer without a memory space.
  %key0 = memref.memory_space_cast %key
      : memref<?xi32, 1> to memref<?xi32>
  gpu.dealloc %key0 : memref<?xi32>
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
  %result0 = memref.memory_space_cast %result
      : memref<2xi32, 1> to memref<2xi32>
  gpu.dealloc %result0 : memref<2xi32>
  return %largest, %not_numbers : index, index
}

// The order of the particles by cell: `order[k]` is the particle that comes
// to place `k`. See the template for the host.
func.func private @mdrt_gpu_spatial_order(
    %x: memref<?x3xf64, 1>, %box: vector<3xf64>, %width: f64,
    %ids: memref<?xi32, 1>, %order: memref<?xi32, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %block = arith.constant 128 : index
  %chunk = arith.constant 256 : index

  %n = memref.dim %x, %c0 : memref<?x3xf64, 1>
  %lx = vector.extract %box[0] : f64 from vector<3xf64>
  %ly = vector.extract %box[1] : f64 from vector<3xf64>
  %lz = vector.extract %box[2] : f64 from vector<3xf64>
  %unit = arith.constant 1.0 : f64
  %ilx = arith.divf %unit, %lx : f64
  %ily = arith.divf %unit, %ly : f64
  %ilz = arith.divf %unit, %lz : f64
  %nx = call @mdrt_gpu_cell_count(%lx, %width) : (f64, f64) -> index
  %ny = call @mdrt_gpu_cell_count(%ly, %width) : (f64, f64) -> index
  %nz = call @mdrt_gpu_cell_count(%lz, %width) : (f64, f64) -> index
  %nxy = arith.muli %nx, %ny : index
  %cells = arith.muli %nxy, %nz : index
  %cells1 = arith.addi %cells, %c1 : index
  %cell_chunks = call @mdrt_gpu_grid(%cells, %chunk)
      : (index, index) -> index

  %grid_n = call @mdrt_gpu_grid(%n, %block) : (index, index) -> index
  %grid_cells = call @mdrt_gpu_grid(%cells, %block) : (index, index) -> index
  %grid_cell_chunks = call @mdrt_gpu_grid(%cell_chunks, %block)
      : (index, index) -> index

  %key = gpu.alloc (%n) : memref<?xi32, 1>
  %held = gpu.alloc (%cells) : memref<?xi32, 1>
  %start = gpu.alloc (%cells1) : memref<?xi32, 1>
  %cursor = gpu.alloc (%cells) : memref<?xi32, 1>
  %cell_sums = gpu.alloc (%cell_chunks) : memref<?xi32, 1>

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

      %zy = arith.muli %cz, %ny : index
      %row = arith.addi %zy, %cy : index
      %rows = arith.muli %row, %nx : index
      %k = arith.addi %rows, %cx : index
      %k32 = arith.index_cast %k : index to i32
      memref.store %k32, %key[%i] : memref<?xi32, 1>
      // A relaxed atomic at the scope of the device (see PMEGPU.mlir).
      %rx3_base = memref.extract_aligned_pointer_as_index %held : memref<?xi32, 1> -> index
      %rx3_bi = arith.index_cast %rx3_base : index to i64
      %rx3_ki = arith.index_cast %k : index to i64
      %rx3_four = arith.constant 4 : i64
      %rx3_off = arith.muli %rx3_ki, %rx3_four : i64
      %rx3_addr = arith.addi %rx3_bi, %rx3_off : i64
      %rx3_ptr = llvm.inttoptr %rx3_addr : i64 to !llvm.ptr<1>
      %old = llvm.atomicrmw add %rx3_ptr, %one syncscope("device") monotonic : !llvm.ptr<1>, i32
    }
    gpu.terminator
  }

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

  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_n, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
    %base = arith.muli %bx, %block : index
    %i = arith.addi %base, %tx : index
    %inside = arith.cmpi ult, %i, %n : index
    scf.if %inside {
      %one = arith.constant 1 : i32
      %k32 = memref.load %key[%i] : memref<?xi32, 1>
      %k = arith.index_cast %k32 : i32 to index
      // A relaxed atomic at the scope of the device (see PMEGPU.mlir).
      %rx4_base = memref.extract_aligned_pointer_as_index %cursor : memref<?xi32, 1> -> index
      %rx4_bi = arith.index_cast %rx4_base : index to i64
      %rx4_ki = arith.index_cast %k : index to i64
      %rx4_four = arith.constant 4 : i64
      %rx4_off = arith.muli %rx4_ki, %rx4_four : i64
      %rx4_addr = arith.addi %rx4_bi, %rx4_off : i64
      %rx4_ptr = llvm.inttoptr %rx4_addr : i64 to !llvm.ptr<1>
      %slot32 = llvm.atomicrmw add %rx4_ptr, %one syncscope("device") monotonic : !llvm.ptr<1>, i32
      %slot = arith.index_cast %slot32 : i32 to index
      %narrow = arith.index_cast %i : index to i32
      memref.store %narrow, %order[%slot] : memref<?xi32, 1>
    }
    gpu.terminator
  }

  // The particles of each cell in the order of `ids`, by insertion.
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_cells, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
    %base = arith.muli %bx, %block : index
    %c = arith.addi %base, %tx : index
    %inside = arith.cmpi ult, %c, %cells : index
    scf.if %inside {
      %i1 = arith.constant 1 : index
      %next = arith.addi %c, %i1 : index
      %begin32 = memref.load %start[%c] : memref<?xi32, 1>
      %end32 = memref.load %start[%next] : memref<?xi32, 1>
      %begin = arith.index_cast %begin32 : i32 to index
      %end = arith.index_cast %end32 : i32 to index
      %first = arith.addi %begin, %i1 : index
      scf.for %p = %first to %end step %i1 {
        %value = memref.load %order[%p] : memref<?xi32, 1>
        %particle = arith.index_cast %value : i32 to index
        %id = memref.load %ids[%particle] : memref<?xi32, 1>
        %hole = scf.while (%q = %p) : (index) -> index {
          %more = arith.cmpi ugt, %q, %begin : index
          %before = arith.subi %q, %i1 : index
          %safe = arith.select %more, %before, %begin : index
          %left = memref.load %order[%safe] : memref<?xi32, 1>
          %left_particle = arith.index_cast %left : i32 to index
          %left_id = memref.load %ids[%left_particle] : memref<?xi32, 1>
          %larger = arith.cmpi sgt, %left_id, %id : i32
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
    }
    gpu.terminator
  }

  %key0 = memref.memory_space_cast %key
      : memref<?xi32, 1> to memref<?xi32>
  gpu.dealloc %key0 : memref<?xi32>
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
  return
}
