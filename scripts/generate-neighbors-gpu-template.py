# Writes lib/Runtime/Templates/NeighborsMatrixGPU.mlir, the template that
# builds a neighbor matrix on a device. The kernels that search have their
# body more than once; this script keeps the copies in step. Edit the
# script, not the template, and run
#
#   python3 scripts/generate-neighbors-gpu-template.py \
#       lib/Runtime/Templates/NeighborsMatrixGPU.mlir
import sys

def launch(count_grid, count, body, var="%t"):
    return f'''  gpu.launch blocks(%bx, %by, %bz) in (%gx = {count_grid}, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %block, %sy = %c1, %sz = %c1) {{
    %base = arith.muli %bx, %block : index
    {var} = arith.addi %base, %tx : index
    %inside = arith.cmpi ult, {var}, {count} : index
    scf.if %inside {{
{body}    }}
    gpu.terminator
  }}
'''

def coordinate(a, length, inverse, count, out):
    # The position in the cell and the cell, along one direction.
    return f'''      %{a}0 = arith.mulf %{a}i, {inverse} : f64
      %{a}1 = math.floor %{a}0 : f64
      %{a}2 = arith.mulf %{a}1, {length} : f64
      %w{a} = arith.subf %{a}i, %{a}2 : f64
      %{a}3 = arith.mulf %w{a}, {inverse} : f64
      %{a}4 = arith.index_cast {count} : index to i64
      %{a}5 = arith.sitofp %{a}4 : i64 to f64
      %{a}6 = arith.mulf %{a}3, %{a}5 : f64
      %{a}7 = arith.fptosi %{a}6 : f64 to i64
      %{a}8 = arith.index_cast %{a}7 : i64 to index
      %{a}l = arith.subi {count}, %i1 : index
      %{a}b = arith.minsi %{a}8, %{a}l : index
      {out} = arith.maxsi %{a}b, %i0 : index
'''

def image(a):
    return f'''              %d{a}0 = arith.subf %{a}i, %{a}j : f32
              %d{a}1 = arith.mulf %d{a}0, %narrow_il{a} : f32
              %d{a}2 = math.roundeven %d{a}1 : f32
              %d{a}3 = arith.mulf %d{a}2, %narrow_l{a} : f32
              %d{a} = arith.subf %d{a}0, %d{a}3 : f32
'''

def search(fills):
    first = '''      %none = arith.constant 0 : index
'''
    if fills:
        first = '''      // The first entry of the thread in the row.
      %offset32 = memref.load %part[%t] : memref<?xi32, 1>
      %offset = arith.index_cast %offset32 : i32 to index
      %none = arith.select %is_split, %offset, %i0 : index
'''
    keep = ''
    if fills:
        keep = '''
              %fits = arith.cmpi ult, %count, %row_width : index
              %keep = arith.andi %neighbor, %fits : i1
              scf.if %keep {
                %j32 = memref.load %order[%q] : memref<?xi32, 1>
                memref.store %j32, %index[%i, %count] : memref<?x?xi32, 1>
              }
'''
    last = '''      %found32 = arith.index_cast %found : index to i32
      memref.store %found32, %part[%t] : memref<?xi32, 1>
'''
    if fills:
        last = '''      // A thread that searches for the whole particle has counted its
      // neighbors.
      %whole = arith.cmpi eq, %split, %i0 : index
      scf.if %whole {
        %found32 = arith.index_cast %found : index to i32
        memref.store %found32, %counts[%i] : memref<?xi32, 1>
      }
'''
    return f'''      %i0 = arith.constant 0 : index
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
{first}
      %found = scf.for %r = %row_begin to %row_end step %i1
          iter_args(%count_r = %none) -> (index) {{
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
            iter_args(%count_x = %count_r) -> (index) {{
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
              iter_args(%count = %count_x) -> (index) {{
            %xj = memref.load %sorted[%q, %i0] : memref<?x3xf32, 1>
            %yj = memref.load %sorted[%q, %i1] : memref<?x3xf32, 1>
            %zj = memref.load %sorted[%q, %i2] : memref<?x3xf32, 1>

            // The minimum-image displacement, with one over the edge
            // lengths: see the template for the host.
{image("x")}{image("y")}{image("z")}
            %dx_2 = arith.mulf %dx, %dx : f32
            %dy_2 = arith.mulf %dy, %dy : f32
            %dz_2 = arith.mulf %dz, %dz : f32
            %dxy_2 = arith.addf %dx_2, %dy_2 : f32
            %r2 = arith.addf %dxy_2, %dz_2 : f32

            %near = arith.cmpf olt, %r2, %limit2 : f32
            %other = arith.cmpi ne, %p, %q : index
            %neighbor = arith.andi %near, %other : i1
{keep}
            %more = arith.addi %count, %i1 : index
            %next = arith.select %neighbor, %more, %count : index
            scf.yield %next : index
          }}
          scf.yield %after_cell : index
        }}
        scf.yield %after_x : index
      }}

{last}'''

def indent(text, by):
    return ''.join((by + line if line.strip() else line)
                   for line in text.splitlines(True))

header = '''// Builds a neighbor matrix on a device by binning the particles into cells.
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

'''

body = header
body += launch('%grid_cells', '%cells', '''      %none = arith.constant 0 : i32
      memref.store %none, %held[%c] : memref<?xi32, 1>
''', var='%c')
body += '''
  // The position of each particle in the cell, its cell, and the number of
  // particles that each cell holds.
'''
body += launch('%grid_n', '%n', '''      %i0 = arith.constant 0 : index
      %i1 = arith.constant 1 : index
      %i2 = arith.constant 2 : index
      %one = arith.constant 1 : i32
      %xi = memref.load %x[%i, %i0] : memref<?x3xf64, 1>
      %yi = memref.load %x[%i, %i1] : memref<?x3xf64, 1>
      %zi = memref.load %x[%i, %i2] : memref<?x3xf64, 1>

      // Rounding can place a particle at an edge one cell too far.
''' + coordinate('x', '%lx', '%ilx', '%nx', '%cx') + '\n'
    + coordinate('y', '%ly', '%ily', '%ny', '%cy') + '\n'
    + coordinate('z', '%lz', '%ilz', '%nz', '%cz') + '''
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
''', var='%i')
body += '''
  // The counts become the offsets of the cells: the sums of chunks of 256
  // cells, the offsets of the chunks in one thread, and the offsets of the
  // cells of each chunk.
'''
body += launch('%grid_cell_chunks', '%cell_chunks', '''      %i1 = arith.constant 1 : index
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
''', var='%b')
body += '''
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

'''
body += launch('%grid_cell_chunks', '%cell_chunks', '''      %i1 = arith.constant 1 : index
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
''', var='%b')
body += '''
  // Each particle takes the next slot of its cell.
'''
body += launch('%grid_n', '%n', '''      %one = arith.constant 1 : i32
      %k32 = memref.load %key[%i] : memref<?xi32, 1>
      %k = arith.index_cast %k32 : i32 to index
      %slot32 = memref.atomic_rmw addi %one, %cursor[%k]
          : (i32, memref<?xi32, 1>) -> i32
      %slot = arith.index_cast %slot32 : i32 to index
      %narrow = arith.index_cast %i : index to i32
      memref.store %narrow, %order[%slot] : memref<?xi32, 1>
''', var='%i')
body += '''
  // The particles of each cell, sorted by index, by insertion, and their
  // positions in that order.
'''
body += launch('%grid_cells', '%cells', '''      %i0 = arith.constant 0 : index
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
''', var='%c')
body += '''
  //===--------------------------------------------------------------------===//
  // Neighbors
  //===--------------------------------------------------------------------===//

  // Where the search is split, its threads count what they find, and the
  // counts become the places of the threads in the rows.
  scf.if %splits {
'''
body += indent(launch('%grid_threads', '%threads', indent(search(False), '')), '  ')
body += '''
'''
body += indent(launch('%grid_n', '%n', '''      %i0 = arith.constant 0 : index
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
''', var='%p'), '  ')
body += '''  }

'''
body += launch('%grid_threads', '%threads', search(True))
body += '''
  //===--------------------------------------------------------------------===//
  // The largest count, and the counts limited to the width of a row
  //===--------------------------------------------------------------------===//

  // One thread for each chunk of 256 particles.
'''
body += launch('%grid_chunks', '%chunks', '''      %i1 = arith.constant 1 : index
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
''', var='%b')
body += '''
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
'''
for name, type in [('key', '?xi32'), ('order', '?xi32'),
                   ('wrapped', '?x3xf32'), ('sorted', '?x3xf32'),
                   ('held', '?xi32'), ('start', '?xi32'),
                   ('cursor', '?xi32'), ('cell_sums', '?xi32'),
                   ('part', '?xi32'), ('partial', '?xi32'),
                   ('result', '1xi32')]:
    body += f'''  %{name}0 = memref.memory_space_cast %{name}
      : memref<{type}, 1> to memref<{type}>
  gpu.dealloc %{name}0 : memref<{type}>
'''
body += '''  return %largest : index
}
'''

#===------------------------------------------------------------------------===
# The order of the particles
#===------------------------------------------------------------------------===

body += '''
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

'''
body += launch('%grid_cells', '%cells', '''      %none = arith.constant 0 : i32
      memref.store %none, %held[%c] : memref<?xi32, 1>
''', var='%c')
body += '''
'''
body += launch('%grid_n', '%n', '''      %i0 = arith.constant 0 : index
      %i1 = arith.constant 1 : index
      %i2 = arith.constant 2 : index
      %one = arith.constant 1 : i32
      %xi = memref.load %x[%i, %i0] : memref<?x3xf64, 1>
      %yi = memref.load %x[%i, %i1] : memref<?x3xf64, 1>
      %zi = memref.load %x[%i, %i2] : memref<?x3xf64, 1>

''' + coordinate('x', '%lx', '%ilx', '%nx', '%cx') + '\n'
    + coordinate('y', '%ly', '%ily', '%ny', '%cy') + '\n'
    + coordinate('z', '%lz', '%ilz', '%nz', '%cz') + '''
      %zy = arith.muli %cz, %ny : index
      %row = arith.addi %zy, %cy : index
      %rows = arith.muli %row, %nx : index
      %k = arith.addi %rows, %cx : index
      %k32 = arith.index_cast %k : index to i32
      memref.store %k32, %key[%i] : memref<?xi32, 1>
      %old = memref.atomic_rmw addi %one, %held[%k]
          : (i32, memref<?xi32, 1>) -> i32
''', var='%i')
body += '''
'''
body += launch('%grid_cell_chunks', '%cell_chunks', '''      %i1 = arith.constant 1 : index
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
''', var='%b')
body += '''
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

'''
body += launch('%grid_cell_chunks', '%cell_chunks', '''      %i1 = arith.constant 1 : index
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
''', var='%b')
body += '''
'''
body += launch('%grid_n', '%n', '''      %one = arith.constant 1 : i32
      %k32 = memref.load %key[%i] : memref<?xi32, 1>
      %k = arith.index_cast %k32 : i32 to index
      %slot32 = memref.atomic_rmw addi %one, %cursor[%k]
          : (i32, memref<?xi32, 1>) -> i32
      %slot = arith.index_cast %slot32 : i32 to index
      %narrow = arith.index_cast %i : index to i32
      memref.store %narrow, %order[%slot] : memref<?xi32, 1>
''', var='%i')
body += '''
  // The particles of each cell in the order of `ids`, by insertion.
'''
body += launch('%grid_cells', '%cells', '''      %i1 = arith.constant 1 : index
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
''', var='%c')
body += '''
'''
for name, type in [('key', '?xi32'), ('held', '?xi32'), ('start', '?xi32'),
                   ('cursor', '?xi32'), ('cell_sums', '?xi32')]:
    body += f'''  %{name}0 = memref.memory_space_cast %{name}
      : memref<{type}, 1> to memref<{type}>
  gpu.dealloc %{name}0 : memref<{type}>
'''
body += '''  return
}
'''
open(sys.argv[1], 'w').write(body)

