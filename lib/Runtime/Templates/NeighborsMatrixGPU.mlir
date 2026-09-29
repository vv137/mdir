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
// A kernel runs one thread per item, in blocks of 128 threads. The threads
// beyond the last item do nothing.
//
// The particles of a cell are sorted by index after they are binned: the
// order in which threads take their slots is not fixed, and the order of
// the neighbors decides the order in which forces are added up.

func.func private @mdrt_gpu_cell_count(%length: f64, %width: f64) -> index {
  %c1 = arith.constant 1 : index
  %ratio = arith.divf %length, %width : f64
  %floor = math.floor %ratio : f64
  %wide = arith.fptosi %floor : f64 to i64
  %count = arith.index_cast %wide : i64 to index
  %result = arith.maxsi %count, %c1 : index
  return %result : index
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
    %cell_width: f64, %counts: memref<?xi32, 1>,
    %index: memref<?x?xi32, 1>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %block = arith.constant 128 : index
  %chunk = arith.constant 256 : index

  %n = memref.dim %x, %c0 : memref<?x3xf64, 1>
  %row_width = memref.dim %index, %c1 : memref<?x?xi32, 1>
  %reach2 = arith.mulf %reach, %reach : f64

  %lx = vector.extract %box[0] : f64 from vector<3xf64>
  %ly = vector.extract %box[1] : f64 from vector<3xf64>
  %lz = vector.extract %box[2] : f64 from vector<3xf64>
  %nx = call @mdrt_gpu_cell_count(%lx, %cell_width) : (f64, f64) -> index
  %ny = call @mdrt_gpu_cell_count(%ly, %cell_width) : (f64, f64) -> index
  %nz = call @mdrt_gpu_cell_count(%lz, %cell_width) : (f64, f64) -> index
  %nxy = arith.muli %nx, %ny : index
  %cells = arith.muli %nxy, %nz : index
  %cells1 = arith.addi %cells, %c1 : index
  %chunks = call @mdrt_gpu_grid(%n, %chunk) : (index, index) -> index

  %grid_n = call @mdrt_gpu_grid(%n, %block) : (index, index) -> index
  %grid_cells = call @mdrt_gpu_grid(%cells, %block) : (index, index) -> index
  %grid_cells1 = call @mdrt_gpu_grid(%cells1, %block)
      : (index, index) -> index
  %grid_chunks = call @mdrt_gpu_grid(%chunks, %block)
      : (index, index) -> index

  %key = gpu.alloc (%n) : memref<?xi32, 1>
  %order = gpu.alloc (%n) : memref<?xi32, 1>
  %start = gpu.alloc (%cells1) : memref<?xi32, 1>
  %cursor = gpu.alloc (%cells) : memref<?xi32, 1>
  %partial = gpu.alloc (%chunks) : memref<?xi32, 1>
  %result = gpu.alloc () : memref<1xi32, 1>
  %host = memref.alloca() : memref<1xi32>

  //===--------------------------------------------------------------------===//
  // Counting sort of the particles by cell
  //===--------------------------------------------------------------------===//

  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_cells1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
    %base = arith.muli %bx, %block : index
    %c = arith.addi %base, %tx : index
    %inside = arith.cmpi ult, %c, %cells1 : index
    scf.if %inside {
      %none = arith.constant 0 : i32
      memref.store %none, %start[%c] : memref<?xi32, 1>
    }
    gpu.terminator
  }

  // The cell of each particle, and the number of particles of each cell:
  // start[k + 1] counts the particles of cell k.
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

      // The cell that a coordinate lies in, along each direction. Rounding
      // can place a particle at the upper edge one cell too far.
      %ax0 = arith.divf %xi, %lx : f64
      %ax1 = math.floor %ax0 : f64
      %ax2 = arith.mulf %ax1, %lx : f64
      %ax3 = arith.subf %xi, %ax2 : f64
      %ax4 = arith.divf %ax3, %lx : f64
      %ax5 = arith.index_cast %nx : index to i64
      %ax6 = arith.sitofp %ax5 : i64 to f64
      %ax7 = arith.mulf %ax4, %ax6 : f64
      %ax8 = arith.fptosi %ax7 : f64 to i64
      %ax9 = arith.index_cast %ax8 : i64 to index
      %axl = arith.subi %nx, %i1 : index
      %axc = arith.cmpi slt, %ax9, %axl : index
      %cx = arith.select %axc, %ax9, %axl : index

      %ay0 = arith.divf %yi, %ly : f64
      %ay1 = math.floor %ay0 : f64
      %ay2 = arith.mulf %ay1, %ly : f64
      %ay3 = arith.subf %yi, %ay2 : f64
      %ay4 = arith.divf %ay3, %ly : f64
      %ay5 = arith.index_cast %ny : index to i64
      %ay6 = arith.sitofp %ay5 : i64 to f64
      %ay7 = arith.mulf %ay4, %ay6 : f64
      %ay8 = arith.fptosi %ay7 : f64 to i64
      %ay9 = arith.index_cast %ay8 : i64 to index
      %ayl = arith.subi %ny, %i1 : index
      %ayc = arith.cmpi slt, %ay9, %ayl : index
      %cy = arith.select %ayc, %ay9, %ayl : index

      %az0 = arith.divf %zi, %lz : f64
      %az1 = math.floor %az0 : f64
      %az2 = arith.mulf %az1, %lz : f64
      %az3 = arith.subf %zi, %az2 : f64
      %az4 = arith.divf %az3, %lz : f64
      %az5 = arith.index_cast %nz : index to i64
      %az6 = arith.sitofp %az5 : i64 to f64
      %az7 = arith.mulf %az4, %az6 : f64
      %az8 = arith.fptosi %az7 : f64 to i64
      %az9 = arith.index_cast %az8 : i64 to index
      %azl = arith.subi %nz, %i1 : index
      %azc = arith.cmpi slt, %az9, %azl : index
      %cz = arith.select %azc, %az9, %azl : index

      %zy = arith.muli %cz, %ny : index
      %row = arith.addi %zy, %cy : index
      %rows = arith.muli %row, %nx : index
      %k = arith.addi %rows, %cx : index
      %k32 = arith.index_cast %k : index to i32
      memref.store %k32, %key[%i] : memref<?xi32, 1>

      %next = arith.addi %k, %i1 : index
      %old = memref.atomic_rmw addi %one, %start[%next]
          : (i32, memref<?xi32, 1>) -> i32
    }
    gpu.terminator
  }

  // The counts become the offsets of the cells, in one thread.
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    scf.for %c = %i0 to %cells step %i1 {
      %next = arith.addi %c, %i1 : index
      %before = memref.load %start[%c] : memref<?xi32, 1>
      %here = memref.load %start[%next] : memref<?xi32, 1>
      %sum = arith.addi %before, %here : i32
      memref.store %sum, %start[%next] : memref<?xi32, 1>
      memref.store %before, %cursor[%c] : memref<?xi32, 1>
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

  // The particles of each cell, sorted by index, by insertion.
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
    }
    gpu.terminator
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

  gpu.launch blocks(%bx, %by, %bz) in (%gx = %grid_n, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {
    %base = arith.muli %bx, %block : index
    %i = arith.addi %base, %tx : index
    %inside = arith.cmpi ult, %i, %n : index
    scf.if %inside {
      %i0 = arith.constant 0 : index
      %i1 = arith.constant 1 : index
      %i2 = arith.constant 2 : index
      %xi = memref.load %x[%i, %i0] : memref<?x3xf64, 1>
      %yi = memref.load %x[%i, %i1] : memref<?x3xf64, 1>
      %zi = memref.load %x[%i, %i2] : memref<?x3xf64, 1>

      %k32 = memref.load %key[%i] : memref<?xi32, 1>
      %k = arith.index_cast %k32 : i32 to index
      %cx = arith.remui %k, %nx : index
      %rest = arith.divui %k, %nx : index
      %cy = arith.remui %rest, %ny : index
      %cz = arith.divui %rest, %ny : index

      %found = scf.for %oz = %i0 to %span_z step %i1
          iter_args(%count_z = %i0) -> (index) {
        %sz0 = arith.addi %cz, %first_z : index
        %sz1 = arith.addi %sz0, %oz : index
        %nz_cell = arith.remui %sz1, %nz : index

        %after_y = scf.for %oy = %i0 to %span_y step %i1
            iter_args(%count_y = %count_z) -> (index) {
          %sy0 = arith.addi %cy, %first_y : index
          %sy1 = arith.addi %sy0, %oy : index
          %ny_cell = arith.remui %sy1, %ny : index

          %after_x = scf.for %ox = %i0 to %span_x step %i1
              iter_args(%count_x = %count_y) -> (index) {
            %sx0 = arith.addi %cx, %first_x : index
            %sx1 = arith.addi %sx0, %ox : index
            %nx_cell = arith.remui %sx1, %nx : index

            %zy = arith.muli %nz_cell, %ny : index
            %row = arith.addi %zy, %ny_cell : index
            %rows = arith.muli %row, %nx : index
            %cell = arith.addi %rows, %nx_cell : index
            %cell1 = arith.addi %cell, %i1 : index
            %begin32 = memref.load %start[%cell] : memref<?xi32, 1>
            %end32 = memref.load %start[%cell1] : memref<?xi32, 1>
            %begin = arith.index_cast %begin32 : i32 to index
            %end = arith.index_cast %end32 : i32 to index

            %after_cell = scf.for %p = %begin to %end step %i1
                iter_args(%count = %count_x) -> (index) {
              %j32 = memref.load %order[%p] : memref<?xi32, 1>
              %j = arith.index_cast %j32 : i32 to index
              %xj = memref.load %x[%j, %i0] : memref<?x3xf64, 1>
              %yj = memref.load %x[%j, %i1] : memref<?x3xf64, 1>
              %zj = memref.load %x[%j, %i2] : memref<?x3xf64, 1>

              // The minimum-image displacement.
              %dx0 = arith.subf %xi, %xj : f64
              %dx1 = arith.divf %dx0, %lx : f64
              %dx2 = math.roundeven %dx1 : f64
              %dx3 = arith.mulf %dx2, %lx : f64
              %dx = arith.subf %dx0, %dx3 : f64
              %dy0 = arith.subf %yi, %yj : f64
              %dy1 = arith.divf %dy0, %ly : f64
              %dy2 = math.roundeven %dy1 : f64
              %dy3 = arith.mulf %dy2, %ly : f64
              %dy = arith.subf %dy0, %dy3 : f64
              %dz0 = arith.subf %zi, %zj : f64
              %dz1 = arith.divf %dz0, %lz : f64
              %dz2 = math.roundeven %dz1 : f64
              %dz3 = arith.mulf %dz2, %lz : f64
              %dz = arith.subf %dz0, %dz3 : f64

              %dx_2 = arith.mulf %dx, %dx : f64
              %dy_2 = arith.mulf %dy, %dy : f64
              %dz_2 = arith.mulf %dz, %dz : f64
              %dxy_2 = arith.addf %dx_2, %dy_2 : f64
              %r2 = arith.addf %dxy_2, %dz_2 : f64

              %near = arith.cmpf olt, %r2, %reach2 : f64
              %other = arith.cmpi ne, %i, %j : index
              %neighbor = arith.andi %near, %other : i1

              %fits = arith.cmpi ult, %count, %row_width : index
              %keep = arith.andi %neighbor, %fits : i1
              scf.if %keep {
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
        scf.yield %after_y : index
      }

      %found32 = arith.index_cast %found : index to i32
      memref.store %found32, %counts[%i] : memref<?xi32, 1>
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
  %key0 = memref.memory_space_cast %key : memref<?xi32, 1> to memref<?xi32>
  %order0 = memref.memory_space_cast %order
      : memref<?xi32, 1> to memref<?xi32>
  %start0 = memref.memory_space_cast %start
      : memref<?xi32, 1> to memref<?xi32>
  %cursor0 = memref.memory_space_cast %cursor
      : memref<?xi32, 1> to memref<?xi32>
  %partial0 = memref.memory_space_cast %partial
      : memref<?xi32, 1> to memref<?xi32>
  %result0 = memref.memory_space_cast %result
      : memref<1xi32, 1> to memref<1xi32>
  gpu.dealloc %key0 : memref<?xi32>
  gpu.dealloc %order0 : memref<?xi32>
  gpu.dealloc %start0 : memref<?xi32>
  gpu.dealloc %cursor0 : memref<?xi32>
  gpu.dealloc %partial0 : memref<?xi32>
  gpu.dealloc %result0 : memref<1xi32>
  return %largest : index
}
