// GPU neighbor matrix build. Cell binning follows [Quentrec1973]; rows
// retain the particle order within each cell and the ballot rank within
// each warp. See docs/neighbors.md for the periodic search and skin.
#include "mdir/Conversion/MatrixBuild.h"
#include "NeighborBuildCode.h"
#include "mdir/Conversion/CompactionEmitter.h"
#include "mdir/Conversion/Passes.h"
#include "mdir/Dialect/MDAlg/MDAlgOps.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels::detail;
namespace {
class MatrixBuild : public NeighborBuildCode {
public:
  using NeighborBuildCode::NeighborBuildCode;
  MemRefType rowType =
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, i32,
                      MemRefLayoutAttrInterface(), b.getI64IntegerAttr(1));
  MemRefType resultType = MemRefType::get({2}, i32, MemRefLayoutAttrInterface(),
                                          b.getI64IntegerAttr(1));
  MemRefType hostType = MemRefType::get({2}, i32);
  Type boxType = VectorType::get({3}, f64),
       tiltedType = VectorType::get({6}, f64);
  md_alg::ScanStrategyAttr chunked =
      md_alg::ScanStrategyAttr::get(context, md_alg::ScanStrategy::Chunked);
  void emitCellCount();
  void emitCellRange();
  void emitCellWidth();
  void emitMatrixSort();
  void emitMatrixImageWithin();
  void emitBuildNeighborsMatrixTriclinic();
  void emitSpatialOrder();
  void emitSearch(StringRef symbol, bool triclinic, bool testAllImages,
                  bool useExplicitImages = false, bool useUniqueOwned = false);

private:
  bool tilted, allImages, explicitImages, uniqueOwned;
  // Common arguments and SSA values captured by the build's launches.
  Value x, box, widths, reach, cellWidth, excluded, counts, entries, order;
  Value lx, ly, lz, ilx, ily, ilz, tbx, tcx, tcy;
  Value narrowLx, narrowLy, narrowLz, narrowIlx, narrowIly, narrowIlz;
  Value narrowBx, narrowCx, narrowCy, narrowFar, mostImages, limit2;
  Value nx, ny, nz, cells, cellChunks, rowsWithin, spanX, spanY;
  Value firstX, firstY, firstZ, rowWidth, n, c0, c1, falseValue;
  Value threads, gridCells, gridN, gridWarps;
  Value key, wrapped, sorted, held, start, cursor, cellSums, result, host;

  Value wrapCoordinate(Value position, Value length, Value inverseLength) {
    Value periods = mulf(position, inverseLength);
    Value wholePeriods = floor(periods);
    Value displacement = mulf(wholePeriods, length);
    return subf(position, displacement);
  }
  // A position rounded onto a face still belongs to a valid bin.
  Value binCoordinate(Value fraction, Value bins, Value zero, Value one) {
    Value count64 = indexCast(i64, bins);
    Value countReal = sitofp(f64, count64);
    Value scaled = mulf(fraction, countReal);
    Value bin64 = fptosi(i64, scaled);
    Value bin = indexCast(index, bin64);
    Value last = subi(bins, one);
    Value bounded = minsi(bin, last);
    return maxsi(bounded, zero);
  }

  void emitSearchGeometry();
  void emitClearCellCounts();
  void emitParticleBins();
  void emitSortedPositions();
  // Partners held by this warp, plus the tail in the ragged row.
  struct ExclusionRow {
    Value first, count, inRegisters, partner, lowest, highest;
  };
  struct WarpConstants {
    Value zero, one, width, missing;
  };
  void emitEntryWithExclusions(const ExclusionRow &partners,
                               const WarpConstants &warp, Value neighbor,
                               Value keep, Value p, Value p32, Value q32,
                               Value slot);
  void emitSearchRows();
  void emitBuildResult();
  Value emitCandidateWithinReach(Value xi, Value yi, Value zi, Value xj,
                                 Value yj, Value zj, Value inRun);
};
void MatrixBuild::emitCellCount() {
  function("mdrt_gpu_cell_count", b.getFunctionType({f64, f64}, {index}),
           [&](ValueRange arguments) {
             Value length = arguments[0];
             Value width = arguments[1];
             Value c1 = cx(1);
             Value slack = cd(1.0001);
             Value padded = mulf(width, slack);
             // At least one cell and at most 256 (md_alg.grid_plan): a cell
             // that is not a number, or one that a run that has failed has
             // blown up, gives a count of cells that is defined and fits
             // memory; the cells are then wider than the reach needs, and the
             // range of the search follows from their count (#168).
             Value wide =
                 md_alg::GridPlanOp::create(
                     b, loc, TypeRange{i64}, ValueRange{length}, padded,
                     Value(), FloatAttr(), Value(), b.getF64FloatAttr(1.0),
                     b.getF64FloatAttr(256.0), nullptr)
                     .getCounts()[0];
             Value count = indexCast(index, wide);
             Value result = maxsi(count, c1);
             func::ReturnOp::create(b, loc, ValueRange{result});
           });
}

void MatrixBuild::emitCellRange() {
  function("mdrt_gpu_cell_range", b.getFunctionType({f64, index, f64}, {index}),
           [&](ValueRange arguments) {
             Value length = arguments[0];
             Value count = arguments[1];
             Value reach = arguments[2];
             Value c1 = cx(1);
             Value slack = cd(1.00005);
             Value wide = indexCast(i64, count);
             Value cells = sitofp(f64, wide);
             Value width = divf(length, cells);
             Value ratio = divf(reach, width);
             Value padded = mulf(ratio, slack);
             Value roundedUp = math::CeilOp::create(b, loc, padded);
             // Defined for a cell that is not a number, as the count is (#168).
             Value oneF = cd(1.0);
             Value mostF = cd(256.0);
             Value least = maxnumf(roundedUp, oneF);
             Value bounded = minnumf(least, mostF);
             Value whole = fptosi(i64, bounded);
             Value range = indexCast(index, whole);
             Value result = maxsi(range, c1);
             func::ReturnOp::create(b, loc, ValueRange{result});
           });
}

void MatrixBuild::emitCellWidth() {
  function(
      "mdrt_gpu_cell_width",
      b.getFunctionType({index, boxType, f64, f64}, {f64}),
      [&](ValueRange arguments) {
        Value count = arguments[0];
        Value box = arguments[1];
        Value reach = arguments[2];
        Value least = arguments[3];
        Value c1 = cx(1);
        Value c4 = cx(4);
        Value lx = extract(box, 0);
        Value ly = extract(box, 1);
        Value lz = extract(box, 2);
        Value wideCount = indexCast(i64, count);
        Value particles = sitofp(f64, wideCount);
        // A row: the visit, and the half-empty last step.
        Value rowCost = cd(1.5);
        Value none = cd(1.0e30);
        auto costValues = forLoop(
            c1, c4, c1, {none, least},
            [&](Value parts, ValueRange state) -> Values {
              Value best = state[0];
              Value chosen = state[1];
              Value wideParts = indexCast(i64, parts);
              Value divisor = sitofp(f64, wideParts);
              Value narrow = divf(reach, divisor);
              Value tried = maximumf(narrow, least);
              Value nx = call("mdrt_gpu_cell_count", {index}, {lx, tried})[0];
              Value ny = call("mdrt_gpu_cell_count", {index}, {ly, tried})[0];
              Value nz = call("mdrt_gpu_cell_count", {index}, {lz, tried})[0];
              Value rx =
                  call("mdrt_gpu_cell_range", {index}, {lx, nx, reach})[0];
              Value ry =
                  call("mdrt_gpu_cell_range", {index}, {ly, ny, reach})[0];
              Value rz =
                  call("mdrt_gpu_cell_range", {index}, {lz, nz, reach})[0];
              Value tx = addi(rx, rx);
              Value ty = addi(ry, ry);
              Value tz = addi(rz, rz);
              Value fx = addi(tx, c1);
              Value fy = addi(ty, c1);
              Value fz = addi(tz, c1);
              Value sx = minsi(nx, fx);
              Value sy = minsi(ny, fy);
              Value sz = minsi(nz, fz);
              // The rows that a search visits, and the particles that it tests:
              // the particles of the cells that it visits, out of all cells.
              Value rows = muli(sy, sz);
              Value visited = muli(rows, sx);
              Value nxy = muli(nx, ny);
              Value cells = muli(nxy, nz);
              Value wideRows = indexCast(i64, rows);
              Value wideVisited = indexCast(i64, visited);
              Value wideCells = indexCast(i64, cells);
              Value realRows = sitofp(f64, wideRows);
              Value realVisited = sitofp(f64, wideVisited);
              Value realCells = sitofp(f64, wideCells);
              // The particles of a row of cells, and the steps of 32 that it
              // takes.
              Value realSpan = divf(realVisited, realRows);
              Value share = divf(realSpan, realCells);
              Value perRow = mulf(particles, share);
              Value lanes = cd(32.0);
              Value steps = divf(perRow, lanes);
              Value rowSteps = addf(steps, rowCost);
              Value total = mulf(realRows, rowSteps);
              Value better = cmpf(CmpFPredicate::OLT, total, best);
              Value nextBest = select(better, total, best);
              Value nextChosen = select(better, tried, chosen);
              return {nextBest, nextChosen};
            });
        Value width = costValues[1];
        func::ReturnOp::create(b, loc, ValueRange{width});
      });
}

void MatrixBuild::emitMatrixSort() {
  function("mdrt_gpu_matrix_sort",
           b.getFunctionType({buffer, wrappedType, wrappedType, buffer, buffer,
                              buffer, buffer, buffer, index, index, index},
                             {}),
           [&](ValueRange arguments) {
             Value key = arguments[0];
             Value wrapped = arguments[1];
             Value sorted = arguments[2];
             Value held = arguments[3];
             Value start = arguments[4];
             Value cursor = arguments[5];
             Value cellSums = arguments[6];
             Value order = arguments[7];
             Value n = arguments[8];
             Value cells = arguments[9];
             Value cellChunks = arguments[10];
             Value c1 = cx(1);
             Value threads = cx(128);
             Value chunk = cx(256);
             Value gridN = call("mdrt_gpu_grid", {index}, {n, threads})[0];
             call("mdrt_gpu_grid", {index}, {cells, threads});
             Value gridCellChunks =
                 call("mdrt_gpu_grid", {index}, {cellChunks, threads})[0];
             // The counts become the offsets of the cells: the sums of chunks
             // of 256 cells, the offsets of the chunks in one thread, and the
             // offsets of the cells of each chunk.
             md_alg::ScanOp::create(
                 b, loc, held, cells, start, cursor, chunked, cellSums,
                 ValueRange{cellChunks, chunk, gridCellChunks, threads, c1},
                 nullptr);
             // Each particle takes the next slot of its cell.
             md_alg::ScatterToBinsOp::create(
                 b, loc, key, n, cursor, order, false,
                 ValueRange{gridN, threads, c1}, nullptr);
             // The particles of each cell, sorted by index, and their positions
             // in that order. The place of a particle within its cell is the
             // number of the particles of the cell with a smaller index, which
             // each particle counts for itself: a thread for each particle
             // rather than one for each cell, which sorted its cell by
             // insertion. @mdrt_gpu_cell_width gives a small system few cells
             // of many particles: 1,410 particles in a simulation cell of 25 Å
             // with a reach of 10 Å have 8 cells of 176, which one thread each
             // sorted in 0.85 ms of an RTX 3090 at every build (#231).
             Value ranked = alloc(buffer, {n});
             md_alg::SortBinsOp::create(
                 b, loc, start, cells, order, Value(), ranked, key,
                 md_alg::SortStrategyAttr::get(
                     context, md_alg::SortStrategy::StableRank),
                 ValueRange{gridN, threads, c1}, nullptr);
             {
               auto launch = gpu::LaunchOp::create(b, loc, gridN, c1, c1,
                                                   threads, c1, c1);
               OpBuilder::InsertionGuard scope(b);
               b.setInsertionPointToEnd(&launch.getBody().front());
               Value bx = launch.getBlockIds().x;
               Value tx = launch.getThreadIds().x;
               Value base = muli(bx, threads);
               Value p = addi(base, tx);
               Value total32 = load(start, {cells});
               Value total = indexCast(index, total32);
               Value inside = cmpi(CmpIPredicate::ult, p, total);
               ifThen(inside, [&] {
                 Value zeroIndex = cx(0);
                 Value firstIndex = cx(1);
                 Value secondIndex = cx(2);
                 Value j32 = load(ranked, {p});
                 Value j = indexCast(index, j32);
                 store(j32, order, {p});
                 Value wx = load(wrapped, {j, zeroIndex});
                 Value wy = load(wrapped, {j, firstIndex});
                 Value wz = load(wrapped, {j, secondIndex});
                 store(wx, sorted, {p, zeroIndex});
                 store(wy, sorted, {p, firstIndex});
                 store(wz, sorted, {p, secondIndex});
               });
               gpu::TerminatorOp::create(b, loc);
             }
             // The lowering of `gpu.dealloc` takes a buffer without a memory
             // space.
             dealloc(ranked);
             func::ReturnOp::create(b, loc, ValueRange{});
           });
}

void MatrixBuild::emitSearchGeometry() {
  c0 = cx(0);
  c1 = cx(1);
  threads = cx(128);
  Value chunk = cx(256);
  n = dim(x, c0);
  rowWidth = dim(entries, c1);
  // The cell: its diagonal a_x, b_y, c_z and its tilts b_x, c_x, c_y; the
  // cells of the search are those of the fractional coordinates, and their
  // numbers follow from the widths of the cell between its faces.
  lx = extract(box, 0);
  ly = extract(box, 1);
  lz = extract(box, 2);
  Value widthX, widthY, widthZ;
  if (tilted) {
    tbx = extract(box, 3);
    tcx = extract(box, 4);
    tcy = extract(box, 5);
    widthX = extract(widths, 0);
    widthY = extract(widths, 1);
    widthZ = extract(widths, 2);
  } else {
    widthX = lx;
    widthY = ly;
    widthZ = lz;
  }
  Value unit = cd(1.0);
  ilx = divf(unit, lx);
  ily = divf(unit, ly);
  ilz = divf(unit, lz);
  nx = call("mdrt_gpu_cell_count", {index}, {widthX, cellWidth})[0];
  ny = call("mdrt_gpu_cell_count", {index}, {widthY, cellWidth})[0];
  nz = call("mdrt_gpu_cell_count", {index}, {widthZ, cellWidth})[0];
  Value nxy = muli(nx, ny);
  cells = muli(nxy, nz);
  cellChunks = call("mdrt_gpu_grid", {index}, {cells, chunk})[0];
  // What the search computes with, in f32.
  narrowLx = truncf(lx);
  narrowLy = truncf(ly);
  narrowLz = truncf(lz);
  Value narrowUnit = cf(1.0);
  narrowIlx = divf(narrowUnit, narrowLx);
  narrowIly = divf(narrowUnit, narrowLy);
  narrowIlz = divf(narrowUnit, narrowLz);
  Value lxy;
  if (tilted) {
    narrowBx = truncf(tbx);
    narrowCx = truncf(tcx);
    narrowCy = truncf(tcy);
    // The margin is more than the rounding of a distance between positions in
    // the cell.
    Value absBx = absf(tbx);
    Value absCx = absf(tcx);
    Value absCy = absf(tcy);
    Value lxy0 = addf(lx, ly);
    Value lxy1 = addf(lxy0, absBx);
    Value lxy2 = addf(lxy1, absCx);
    lxy = addf(lxy2, absCy);
  } else {
    lxy = addf(lx, ly);
  }
  Value lxyz = addf(lxy, lz);
  Value tiny = cd(3.0e-6);
  Value margin = mulf(lxyz, tiny);
  Value far = addf(reach, margin);
  Value far2 = mulf(far, far);
  limit2 = truncf(far2);
  if (allImages) {
    narrowFar = truncf(far);
    mostImages = cf(64.0);
  }
  // The cells within reach.
  Value rangeX = call("mdrt_gpu_cell_range", {index}, {widthX, nx, reach})[0];
  Value rangeY = call("mdrt_gpu_cell_range", {index}, {widthY, ny, reach})[0];
  Value rangeZ = call("mdrt_gpu_cell_range", {index}, {widthZ, nz, reach})[0];
  Value twiceX = addi(rangeX, rangeX);
  Value twiceY = addi(rangeY, rangeY);
  Value twiceZ = addi(rangeZ, rangeZ);
  Value fullX = addi(twiceX, c1);
  Value fullY = addi(twiceY, c1);
  Value fullZ = addi(twiceZ, c1);
  spanX = minsi(nx, fullX);
  spanY = minsi(ny, fullY);
  Value spanZ = minsi(nz, fullZ);
  Value allX = cmpi(CmpIPredicate::eq, spanX, fullX);
  Value allY = cmpi(CmpIPredicate::eq, spanY, fullY);
  Value allZ = cmpi(CmpIPredicate::eq, spanZ, fullZ);
  Value backX = subi(nx, rangeX);
  Value backY = subi(ny, rangeY);
  Value backZ = subi(nz, rangeZ);
  firstX = select(allX, backX, c0);
  firstY = select(allY, backY, c0);
  firstZ = select(allZ, backZ, c0);
  // The search has a warp of 32 threads for each particle.
  rowsWithin = muli(spanY, spanZ);
  Value warpThreads = cx(32);
  Value totalThreads = muli(n, warpThreads);
  falseValue = cb(false);
  gridN = call("mdrt_gpu_grid", {index}, {n, threads})[0];
  gridCells = call("mdrt_gpu_grid", {index}, {cells, threads})[0];
  call("mdrt_gpu_grid", {index}, {cellChunks, threads});
  gridWarps = call("mdrt_gpu_grid", {index}, {totalThreads, threads})[0];
  Value cells1 = addi(cells, c1);
  key = alloc(buffer, {n});
  wrapped = alloc(wrappedType, {n});
  sorted = alloc(wrappedType, {n});
  held = alloc(buffer, {cells});
  start = alloc(buffer, {cells1});
  cursor = alloc(buffer, {cells});
  cellSums = alloc(buffer, {cellChunks});
  result = alloc(resultType, {});
  host = memref::AllocaOp::create(b, loc, hostType);
}

void MatrixBuild::emitClearCellCounts() {
  {
    auto launch =
        gpu::LaunchOp::create(b, loc, gridCells, c1, c1, threads, c1, c1);
    OpBuilder::InsertionGuard scope(b);
    b.setInsertionPointToEnd(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value base = muli(bx, threads);
    Value c = addi(base, tx);
    Value inside = cmpi(CmpIPredicate::ult, c, cells);
    ifThen(inside, [&] {
      Value none = ci(0);
      store(none, held, {c});
      // The largest count, which the search raises.
      Value c0First = cx(0);
      Value firstCell = cmpi(CmpIPredicate::eq, c, c0First);
      ifThen(firstCell, [&] {
        store(none, result, {c0First});
        Value c1Second = cx(1);
        store(none, result, {c1Second});
      });
    });
    gpu::TerminatorOp::create(b, loc);
  }
}

void MatrixBuild::emitParticleBins() {
  // The position of each particle in the cell, its cell, and the number of
  // particles that each cell holds.
  {
    auto launch = gpu::LaunchOp::create(b, loc, gridN, c1, c1, threads, c1, c1);
    OpBuilder::InsertionGuard scope(b);
    b.setInsertionPointToEnd(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value base = muli(bx, threads);
    Value i = addi(base, tx);
    Value inside = cmpi(CmpIPredicate::ult, i, n);
    ifThen(inside, [&] {
      Value zeroIndex = cx(0);
      Value firstIndex = cx(1);
      Value secondIndex = cx(2);
      Value one = ci(1);
      Value xi = load(x, {i, zeroIndex});
      Value yi = load(x, {i, firstIndex});
      Value zi = load(x, {i, secondIndex});
      Value wx, wy, wz, cellX, cy, cz;
      if (tilted) {
        // s = x H⁻¹, taken into [0, 1); its cell, of the fractional
        // coordinates; and the position s H in the cell, which the search
        // takes: see the template for the host. Rounding can place a
        // particle at an edge one cell too far.
        Value fracSz = mulf(zi, ilz);
        Value fracSzCy = mulf(fracSz, tcy);
        Value yRest = subf(yi, fracSzCy);
        Value fracSy = mulf(yRest, ily);
        Value fracSyBx = mulf(fracSy, tbx);
        Value fracSzCx = mulf(fracSz, tcx);
        Value xRest0 = subf(xi, fracSyBx);
        Value xRest = subf(xRest0, fracSzCx);
        Value fracSx = mulf(xRest, ilx);
        Value fracSxFloor = floor(fracSx);
        Value fx = subf(fracSx, fracSxFloor);
        Value fracSyFloor = floor(fracSy);
        Value fy = subf(fracSy, fracSyFloor);
        Value fracSzFloor = floor(fracSz);
        Value fz = subf(fracSz, fracSzFloor);
        cellX = binCoordinate(fx, nx, zeroIndex, firstIndex);
        cy = binCoordinate(fy, ny, zeroIndex, firstIndex);
        cz = binCoordinate(fz, nz, zeroIndex, firstIndex);
        wz = mulf(fz, lz);
        Value fyBy = mulf(fy, ly);
        Value fzCy = mulf(fz, tcy);
        wy = addf(fyBy, fzCy);
        Value fxAx = mulf(fx, lx);
        Value fyBx = mulf(fy, tbx);
        Value fzCx = mulf(fz, tcx);
        Value wx0 = addf(fxAx, fyBx);
        wx = addf(wx0, fzCx);
      } else {
        // Rounding can place a particle at an edge one cell too far.
        wx = wrapCoordinate(xi, lx, ilx);
        Value x3 = mulf(wx, ilx);
        cellX = binCoordinate(x3, nx, zeroIndex, firstIndex);
        wy = wrapCoordinate(yi, ly, ily);
        Value y3 = mulf(wy, ily);
        cy = binCoordinate(y3, ny, zeroIndex, firstIndex);
        wz = wrapCoordinate(zi, lz, ilz);
        Value z3 = mulf(wz, ilz);
        cz = binCoordinate(z3, nz, zeroIndex, firstIndex);
      }
      // An explicit-image neighborhood still uses periodic bin keys. Its
      // stored coordinates already identify a particular periodic copy.
      Value narrowX = truncf(explicitImages ? xi : wx);
      Value narrowY = truncf(explicitImages ? yi : wy);
      Value narrowZ = truncf(explicitImages ? zi : wz);
      store(narrowX, wrapped, {i, zeroIndex});
      store(narrowY, wrapped, {i, firstIndex});
      store(narrowZ, wrapped, {i, secondIndex});
      Value zy = muli(cz, ny);
      Value row = addi(zy, cy);
      Value rows = muli(row, nx);
      Value k = addi(rows, cellX);
      Value k32 = indexCast(i32, k);
      // No particle at any place yet: a place that stays empty, for want
      // of a particle with a position, is skipped by the search.
      Value noParticle = ci(-1);
      store(noParticle, order, {i});
      // A position that is not a number, or is beyond any cell, gets no
      // cell, and is counted (D107).
      // The largest coordinate, not their sum, which opposite coordinates
      // could cancel; a NaN in any of them propagates through the maximum.
      Value absX = absf(xi);
      Value absY = absf(yi);
      Value absZ = absf(zi);
      Value absXy = maximumf(absX, absY);
      Value magnitude = maximumf(absXy, absZ);
      Value notNumber = cmpf(CmpFPredicate::UNO, magnitude, magnitude);
      Value farOff = cd(1.0e100);
      Value huge = cmpf(CmpFPredicate::OGT, magnitude, farOff);
      Value bad = ori(notNumber, huge);
      ifThenElse(
          bad,
          [&] {
            store(noParticle, key, {i});
            Value nnBase =
                memref::ExtractAlignedPointerAsIndexOp::create(b, loc, result);
            Value nnBi = indexCast(i64, nnBase);
            Value nnFour = cl(4);
            Value nnAddr = addi(nnBi, nnFour);
            Value nnPtr = LLVM::IntToPtrOp::create(b, loc, pointer, nnAddr);
            atomic(LLVM::AtomicBinOp::add, nnPtr, one);
          },
          [&]() {
            store(k32, key, {i});
            // A relaxed atomic at the scope of the device (D70).
            Value rx1Base =
                memref::ExtractAlignedPointerAsIndexOp::create(b, loc, held);
            Value rx1Bi = indexCast(i64, rx1Base);
            Value rx1Ki = indexCast(i64, k);
            Value rx1Four = cl(4);
            Value rx1Off = muli(rx1Ki, rx1Four);
            Value rx1Addr = addi(rx1Bi, rx1Off);
            Value rx1Ptr = LLVM::IntToPtrOp::create(b, loc, pointer, rx1Addr);
            atomic(LLVM::AtomicBinOp::add, rx1Ptr, one);
          });
    });
    gpu::TerminatorOp::create(b, loc);
  }
}

void MatrixBuild::emitSortedPositions() {
  // The offsets of the cells, the particles in the order of the cells, and
  // their positions in that order.
  call("mdrt_gpu_matrix_sort", {},
       {key, wrapped, sorted, held, start, cursor, cellSums, order, n, cells,
        cellChunks});
}

void MatrixBuild::emitSearchRows() {
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
  Value n32 = indexCast(i32, n);
  Value nx32 = indexCast(i32, nx);
  Value ny32 = indexCast(i32, ny);
  Value nz32 = indexCast(i32, nz);
  Value spanX32 = indexCast(i32, spanX);
  Value spanY32 = indexCast(i32, spanY);
  Value rows32 = indexCast(i32, rowsWithin);
  Value firstX32 = indexCast(i32, firstX);
  Value firstY32 = indexCast(i32, firstY);
  Value firstZ32 = indexCast(i32, firstZ);
  Value rowWidth32 = indexCast(i32, rowWidth);
  {
    auto launch =
        gpu::LaunchOp::create(b, loc, gridWarps, c1, c1, threads, c1, c1);
    OpBuilder::InsertionGuard scope(b);
    b.setInsertionPointToEnd(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value base = muli(bx, threads);
    Value t = addi(base, tx);
    Value t32 = indexCast(i32, t);
    Value zeroIndex = cx(0);
    Value firstIndex = cx(1);
    Value secondIndex = cx(2);
    Value zero32 = ci(0);
    Value one32 = ci(1);
    Value two32 = ci(2);
    Value lanes32 = ci(32);
    Value none32 = ci(-1);
    Value p32 = divui(t32, lanes32);
    Value lane32 = remui(t32, lanes32);
    // The warps past the last particle stop together.
    Value inside = cmpi(CmpIPredicate::ult, p32, n32);
    ifThen(inside, [&] {
      Value p = indexCast(index, p32);
      Value particle32 = load(order, {p});
      Value i = indexCast(index, particle32);
      Value zeroPlaced = ci(0);
      Value placed = cmpi(CmpIPredicate::sge, particle32, zeroPlaced);
      ifThen(placed, [&] {
        Value xi = load(sorted, {p, zeroIndex});
        Value yi = load(sorted, {p, firstIndex});
        Value zi = load(sorted, {p, secondIndex});
        Value k32 = load(key, {i});
        Value cellX = remui(k32, nx32);
        Value rest = divui(k32, nx32);
        Value cy = remui(rest, ny32);
        Value cz = divui(rest, ny32);
        // The excluded pairs of the particle, if the structure has any.
        Value entryExcluded = cx(4);
        auto firstExcludedValues = md_alg::RaggedRowOp::create(
                                       b, loc, excluded, i, entryExcluded, true)
                                       .getResults();
        Value firstExcluded = firstExcludedValues[0];
        Value countExcluded = firstExcludedValues[1];
        Value numExcluded = indexCast(i32, countExcluded);
        // The other member of each excluded pair of the particle: lane k
        // holds that of the pair k, for the first 32; a neighbor is compared
        // with them by shuffles, and with the rest, if any, in memory.
        Value laneExcluded = cmpi(CmpIPredicate::ult, lane32, numExcluded);
        auto myPartnerValues = ifElse(
            {i32}, laneExcluded,
            [&]() -> Values {
              Value lane = indexCast(index, lane32);
              Value mine = call("mdrt_gpu_excluded_partner", {i32},
                                {excluded, firstExcluded, lane})[0];
              return {mine};
            },
            [&]() -> Values { return {none32}; });
        Value myPartner = myPartnerValues[0];
        Value inRegisters = minui(numExcluded, lanes32);
        // The least and the greatest number of a partner: a neighbor outside
        // them is not excluded, and needs no comparison. With more than 32
        // pairs every neighbor is compared.
        Value allLow = ci(0);
        Value allHigh = ci(2147483647);
        auto low0Values =
            forLoop(zero32, inRegisters, one32, {allHigh, allLow},
                    [&](Value e, ValueRange state) -> Values {
                      Value lo = state[0];
                      Value hi = state[1];
                      auto [partner, valid] =
                          shuffle(gpu::ShuffleMode::IDX, myPartner, e, lanes32);
                      Value lo1 = minsi(lo, partner);
                      Value hi1 = maxsi(hi, partner);
                      return {lo1, hi1};
                    });
        Value low0 = low0Values[0];
        Value high0 = low0Values[1];
        Value overflow = cmpi(CmpIPredicate::ugt, numExcluded, lanes32);
        Value low = select(overflow, allLow, low0);
        Value high = select(overflow, allHigh, high0);
        ExclusionRow partners{firstExcluded, numExcluded, inRegisters,
                              myPartner,     low,         high};
        WarpConstants warp{zero32, one32, lanes32, none32};
        Value belowMask = kernels::emitLanesBelow(b, loc, lane32, one32);
        auto foundValues = forLoop(
            zero32, rows32, one32, {zero32},
            [&](Value r, ValueRange state) -> Values {
              Value countR = state[0];
              Value oz = divui(r, spanY32);
              Value oy = remui(r, spanY32);
              Value sz0 = addi(cz, firstZ32);
              Value sz1 = addi(sz0, oz);
              Value nzCell = remui(sz1, nz32);
              Value sy0 = addi(cy, firstY32);
              Value sy1 = addi(sy0, oy);
              Value nyCell = remui(sy1, ny32);
              // The cells of the row are next to one another in the order of
              // the cells, so the warp reads them as one run, or as two where
              // the row goes around the edge of the cell.
              Value zy = muli(nzCell, ny32);
              Value row = addi(zy, nyCell);
              Value rowFirst = muli(row, nx32);
              Value sx0 = addi(cellX, firstX32);
              Value xBegin = remui(sx0, nx32);
              Value xEnd = addi(xBegin, spanX32);
              Value around = cmpi(CmpIPredicate::ugt, xEnd, nx32);
              Value xStop = select(around, nx32, xEnd);
              Value xRest0 = subi(xEnd, nx32);
              Value xRest = select(around, xRest0, zero32);
              auto afterXValues = forLoop(
                  zero32, two32, one32, {countR},
                  [&](Value partX, ValueRange state) -> Values {
                    Value countX = state[0];
                    Value second = cmpi(CmpIPredicate::ne, partX, zero32);
                    Value fromX = select(second, zero32, xBegin);
                    Value toX = select(second, xRest, xStop);
                    Value fromCell32 = addi(rowFirst, fromX);
                    Value toCell32 = addi(rowFirst, toX);
                    Value fromCell = indexCast(index, fromCell32);
                    Value toCell = indexCast(index, toCell32);
                    Value begin = load(start, {fromCell});
                    Value end = load(start, {toCell});
                    Value last = subi(end, one32);
                    auto afterCellValues = forLoop(
                        begin, end, lanes32, {countX},
                        [&](Value q0, ValueRange state) -> Values {
                          Value count = state[0];
                          Value q32 = addi(q0, lane32);
                          Value inRun = cmpi(CmpIPredicate::slt, q32, end);
                          Value qc32 = minsi(q32, last);
                          Value qc = indexCast(index, qc32);
                          Value xj = load(sorted, {qc, zeroIndex});
                          Value yj = load(sorted, {qc, firstIndex});
                          Value zj = load(sorted, {qc, secondIndex});
                          Value nearby = emitCandidateWithinReach(
                              xi, yi, zi, xj, yj, zj, inRun);
                          // A full single-domain layout is in global identity order.
                          // Compare particle identities before compacting the list.
                          Value other = uniqueOwned
                              ? cmpi(CmpIPredicate::slt, particle32, load(order, {qc}))
                              : cmpi(CmpIPredicate::ne, p32, q32);
                          Value nearOther = andi(nearby, other);
                          Value neighbor = andi(nearOther, inRun);
                          Value ballot =
                              kernels::emitWarpBallot(b, loc, neighbor);
                          Value before =
                              kernels::emitWarpRank(b, loc, ballot, belowMask);
                          Value slot = addi(count, before);
                          Value fits =
                              cmpi(CmpIPredicate::ult, slot, rowWidth32);
                          Value keep = andi(neighbor, fits);
                          emitEntryWithExclusions(partners, warp, neighbor,
                                                  keep, p, p32, q32, slot);
                          Value foundHere =
                              kernels::emitWarpCount(b, loc, ballot);
                          Value next = addi(count, foundHere);
                          return {next};
                        });
                    Value afterCell = afterCellValues[0];
                    return {afterCell};
                  });
              Value afterX = afterXValues[0];
              return {afterX};
            });
        Value found = foundValues[0];
        // The count limited to the width of a row, and the largest count,
        // which tells the caller that a row was too narrow. The largest of
        // the counts does not depend on the order of the atomics.
        Value writer = cmpi(CmpIPredicate::eq, lane32, zero32);
        ifThen(writer, [&] {
          Value limited = minui(found, rowWidth32);
          store(limited, counts, {p});
          // A relaxed atomic at the scope of the device (D70).
          Value rmBase =
              memref::ExtractAlignedPointerAsIndexOp::create(b, loc, result);
          Value rmAddr = indexCast(i64, rmBase);
          Value rmPtr = LLVM::IntToPtrOp::create(b, loc, pointer, rmAddr);
          atomic(LLVM::AtomicBinOp::umax, rmPtr, found);
        });
      });
    });
    gpu::TerminatorOp::create(b, loc);
  }
}

void MatrixBuild::emitEntryWithExclusions(const ExclusionRow &partners,
                                          const WarpConstants &warp,
                                          Value neighbor, Value keep, Value p,
                                          Value p32, Value q32, Value slot) {
  // Whether the neighbor of each lane is excluded: all
  // lanes shuffle when any lane has found one within
  // the numbers of the partners.
  auto j32Values = ifElse(
      {i32}, neighbor,
      [&]() -> Values {
        Value q = indexCast(index, q32);
        Value jFound = load(order, {q});
        return {jFound};
      },
      [&]() -> Values { return {warp.missing}; });
  Value j32 = j32Values[0];
  Value above = cmpi(CmpIPredicate::sge, j32, partners.lowest);
  Value under = cmpi(CmpIPredicate::sle, j32, partners.highest);
  Value between = andi(above, under);
  Value maybe = andi(neighbor, between);
  Value maybeBits = kernels::emitWarpBallot(b, loc, maybe);
  Value any = cmpi(CmpIPredicate::ne, maybeBits, warp.zero);
  auto shuffledValues = ifElse(
      {i1}, any,
      [&]() -> Values {
        auto hitValues =
            forLoop(warp.zero, partners.inRegisters, warp.one, {falseValue},
                    [&](Value e, ValueRange state) -> Values {
                      Value foundE = state[0];
                      auto [partner, valid] =
                          shuffle(gpu::ShuffleMode::IDX, partners.partner, e,
                                  warp.width);
                      Value same = cmpi(CmpIPredicate::eq, partner, j32);
                      Value matched = ori(foundE, same);
                      return {matched};
                    });
        Value hit = hitValues[0];
        return {hit};
      },
      [&]() -> Values { return {falseValue}; });
  Value shuffled = shuffledValues[0];
  ifThen(keep, [&] {
    auto isExcludedValues =
        forLoop(partners.inRegisters, partners.count, warp.one, {shuffled},
                [&](Value e, ValueRange state) -> Values {
                  Value foundE = state[0];
                  Value eIndex = indexCast(index, e);
                  Value partner = call("mdrt_gpu_excluded_partner", {i32},
                                       {excluded, partners.first, eIndex})[0];
                  Value same = cmpi(CmpIPredicate::eq, partner, j32);
                  Value anyE = ori(foundE, same);
                  return {anyE};
                });
    Value isExcluded = isExcludedValues[0];
    Value entered = select(isExcluded, p32, q32);
    Value slotIndex = indexCast(index, slot);
    store(entered, entries, {p, slotIndex});
  });
}

void MatrixBuild::emitBuildResult() {
  Value t0 = gpu::WaitOp::create(b, loc, gpu::AsyncTokenType::get(context),
                                 ValueRange{})
                 .getAsyncToken();
  Value t1 = gpu::MemcpyOp::create(b, loc, gpu::AsyncTokenType::get(context),
                                   ValueRange{t0}, host, result)
                 .getAsyncToken();
  gpu::WaitOp::create(b, loc, Type(), ValueRange{t1});
  Value largest32 = load(host, {c0});
  Value largest = indexCast(index, largest32);
  Value notNumbers32 = load(host, {c1});
  Value notNumbers = indexCast(index, notNumbers32);
  // The lowering of `gpu.dealloc` takes a buffer without a memory space.
  dealloc(key);
  dealloc(wrapped);
  dealloc(sorted);
  dealloc(held);
  dealloc(start);
  dealloc(cursor);
  dealloc(cellSums);
  dealloc(result);
  func::ReturnOp::create(b, loc, ValueRange{largest, notNumbers});
}

Value MatrixBuild::emitCandidateWithinReach(Value xi, Value yi, Value zi,
                                            Value xj, Value yj, Value zj,
                                            Value inRun) {
  Value dx, dy, dz;
  if (explicitImages) {
    dx = subf(xi, xj);
    dy = subf(yi, yj);
    dz = subf(zi, zj);
  } else if (tilted) {
    // The minimum-image displacement of the triclinic cell, in one
    // pass along c, b, and a: see the template for the host.
    Value dx0 = subf(xi, xj);
    Value dy0 = subf(yi, yj);
    Value dz0 = subf(zi, zj);
    Value imageZ0 = mulf(dz0, narrowIlz);
    Value imageZ = roundeven(imageZ0);
    Value shiftZx = mulf(imageZ, narrowCx);
    Value shiftZy = mulf(imageZ, narrowCy);
    Value shiftZz = mulf(imageZ, narrowLz);
    Value dx1 = subf(dx0, shiftZx);
    Value dy1 = subf(dy0, shiftZy);
    dz = subf(dz0, shiftZz);
    Value imageY0 = mulf(dy1, narrowIly);
    Value imageY = roundeven(imageY0);
    Value shiftYx = mulf(imageY, narrowBx);
    Value shiftYy = mulf(imageY, narrowLy);
    Value dx2 = subf(dx1, shiftYx);
    dy = subf(dy1, shiftYy);
    Value imageX0 = mulf(dx2, narrowIlx);
    Value imageX = roundeven(imageX0);
    Value shiftXx = mulf(imageX, narrowLx);
    dx = subf(dx2, shiftXx);
  } else {
    // The minimum-image displacement, with one over the edge
    // lengths: see the template for the host.
    Value dx0 = subf(xi, xj);
    Value dx1 = mulf(dx0, narrowIlx);
    Value dx2 = roundeven(dx1);
    Value dx3 = mulf(dx2, narrowLx);
    dx = subf(dx0, dx3);
    Value dy0 = subf(yi, yj);
    Value dy1 = mulf(dy0, narrowIly);
    Value dy2 = roundeven(dy1);
    Value dy3 = mulf(dy2, narrowLy);
    dy = subf(dy0, dy3);
    Value dz0 = subf(zi, zj);
    Value dz1 = mulf(dz0, narrowIlz);
    Value dz2 = roundeven(dz1);
    Value dz3 = mulf(dz2, narrowLz);
    dz = subf(dz0, dz3);
  }
  Value dx_2 = mulf(dx, dx);
  Value dy_2 = mulf(dy, dy);
  Value dz_2 = mulf(dz, dz);
  Value dxy_2 = addf(dx_2, dy_2);
  Value r2 = addf(dxy_2, dz_2);
  Value nearPass = cmpf(CmpFPredicate::OLT, r2, limit2);
  if (!allImages)
    return nearPass;

  Value trueW = cb(true);
  Value farPass = xori(nearPass, trueW);
  Value testPair = andi(farPass, inRun);
  auto nearbyValues = ifElse(
      {i1}, testPair,
      [&]() -> Values {
        Value nearImage =
            call("mdrt_gpu_matrix_image_within", {i1},
                 {dx, dy, dz, narrowLx, narrowLy, narrowLz, narrowBx, narrowCx,
                  narrowCy, narrowIlx, narrowIly, narrowIlz, narrowFar, limit2,
                  mostImages})[0];
        return {nearImage};
      },
      [&]() -> Values { return {nearPass}; });
  Value nearby = nearbyValues[0];
  return nearby;
}

void MatrixBuild::emitSearch(StringRef symbol, bool triclinic,
                             bool testAllImages, bool useExplicitImages,
                             bool useUniqueOwned) {
  uniqueOwned = useUniqueOwned;
  explicitImages = useExplicitImages;
  tilted = triclinic;
  allImages = testAllImages;
  SmallVector<Type> inputs{positions, tilted ? tiltedType : boxType};
  if (tilted)
    inputs.push_back(boxType);
  inputs.append({f64, f64, buffer, buffer, rowType, buffer});
  function(symbol, b.getFunctionType(inputs, {index, index}),
           [&](ValueRange arguments) {
             unsigned arg = 0;
             x = arguments[arg++];
             box = arguments[arg++];
             if (tilted)
               widths = arguments[arg++];
             reach = arguments[arg++];
             cellWidth = arguments[arg++];
             excluded = arguments[arg++];
             counts = arguments[arg++];
             entries = arguments[arg++];
             order = arguments[arg++];
             emitSearchGeometry();
             emitClearCellCounts();
             emitParticleBins();
             emitSortedPositions();
             emitSearchRows();
             emitBuildResult();
           });
}

void MatrixBuild::emitMatrixImageWithin() {
  function("mdrt_gpu_matrix_image_within",
           b.getFunctionType({f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
                              f32, f32, f32, f32, f32},
                             {i1}),
           [&](ValueRange arguments) {
             Value dx = arguments[0];
             Value dy = arguments[1];
             Value dz = arguments[2];
             Value ax = arguments[3];
             Value by = arguments[4];
             Value cz = arguments[5];
             Value bx = arguments[6];
             Value cellX = arguments[7];
             Value cy = arguments[8];
             Value iax = arguments[9];
             Value iby = arguments[10];
             Value icz = arguments[11];
             Value far = arguments[12];
             Value limit2 = arguments[13];
             Value most = arguments[14];
             Value one = ci(1);
             Value falseValue = cb(false);
             Value least = arith::NegFOp::create(b, loc, most);
             Value zl0 = subf(dz, far);
             Value zh0 = addf(dz, far);
             Value zl1 = mulf(zl0, icz);
             Value zh1 = mulf(zh0, icz);
             Value zl2 = math::CeilOp::create(b, loc, zl1);
             Value zh2 = floor(zh1);
             Value zl3 = maxnumf(zl2, least);
             Value zh3 = minnumf(zh2, most);
             Value zl = fptosi(i32, zl3);
             Value zh4 = fptosi(i32, zh3);
             Value zh = addi(zh4, one);
             auto foundValues =
                 forLoop(zl, zh, one, {falseValue},
                         [&](Value nc, ValueRange state) -> Values {
                           Value foundC = state[0];
                           Value fc = sitofp(f32, nc);
                           Value sz = mulf(fc, cz);
                           Value sy = mulf(fc, cy);
                           Value sx = mulf(fc, cellX);
                           Value z = subf(dz, sz);
                           Value y1 = subf(dy, sy);
                           Value x1 = subf(dx, sx);
                           Value z2 = mulf(z, z);
                           Value yl0 = subf(y1, far);
                           Value yh0 = addf(y1, far);
                           Value yl1 = mulf(yl0, iby);
                           Value yh1 = mulf(yh0, iby);
                           Value yl2 = math::CeilOp::create(b, loc, yl1);
                           Value yh2 = floor(yh1);
                           Value yl3 = maxnumf(yl2, least);
                           Value yh3 = minnumf(yh2, most);
                           Value yl = fptosi(i32, yl3);
                           Value yh4 = fptosi(i32, yh3);
                           Value yh = addi(yh4, one);
                           auto afterBValues = forLoop(
                               yl, yh, one, {foundC},
                               [&](Value nb, ValueRange state) -> Values {
                                 Value foundB = state[0];
                                 Value fb = sitofp(f32, nb);
                                 Value ty = mulf(fb, by);
                                 Value tx = mulf(fb, bx);
                                 Value y = subf(y1, ty);
                                 Value x2 = subf(x1, tx);
                                 Value y2 = mulf(y, y);
                                 Value zy2 = addf(z2, y2);
                                 Value xl0 = subf(x2, far);
                                 Value xh0 = addf(x2, far);
                                 Value xl1 = mulf(xl0, iax);
                                 Value xh1 = mulf(xh0, iax);
                                 Value xl2 = math::CeilOp::create(b, loc, xl1);
                                 Value xh2 = floor(xh1);
                                 Value xl3 = maxnumf(xl2, least);
                                 Value xh3 = minnumf(xh2, most);
                                 Value xl = fptosi(i32, xl3);
                                 Value xh4 = fptosi(i32, xh3);
                                 Value xh = addi(xh4, one);
                                 auto afterAValues = forLoop(
                                     xl, xh, one, {foundB},
                                     [&](Value na, ValueRange state) -> Values {
                                       Value foundA = state[0];
                                       Value fa = sitofp(f32, na);
                                       Value ux = mulf(fa, ax);
                                       Value x = subf(x2, ux);
                                       Value xx = mulf(x, x);
                                       Value r2 = addf(zy2, xx);
                                       Value nearby =
                                           cmpf(CmpFPredicate::OLT, r2, limit2);
                                       Value next = ori(foundA, nearby);
                                       return {next};
                                     });
                                 Value afterA = afterAValues[0];
                                 return {afterA};
                               });
                           Value afterB = afterBValues[0];
                           return {afterB};
                         });
             Value found = foundValues[0];
             func::ReturnOp::create(b, loc, ValueRange{found});
           });
}

void MatrixBuild::emitBuildNeighborsMatrixTriclinic() {
  function("mdrt_gpu_build_neighbors_matrix_triclinic",
           b.getFunctionType({positions, tiltedType, boxType, f64, f64, buffer,
                              buffer, rowType, buffer},
                             {index, index}),
           [&](ValueRange arguments) {
             Value x = arguments[0];
             Value box = arguments[1];
             Value widths = arguments[2];
             Value reach = arguments[3];
             Value cellWidth = arguments[4];
             Value excluded = arguments[5];
             Value counts = arguments[6];
             Value entries = arguments[7];
             Value order = arguments[8];
             Value lx = extract(box, 0);
             Value ly = extract(box, 1);
             Value lz = extract(box, 2);
             Value leastXy = minimumf(lx, ly);
             Value least = minimumf(leastXy, lz);
             Value part = cd(0.49);
             Value bound = mulf(least, part);
             Value within = cmpf(CmpFPredicate::OLE, reach, bound);
             auto largestValues = ifElse(
                 {index, index}, within,
                 [&]() -> Values {
                   auto oneValues =
                       call("mdrt_gpu_build_neighbors_matrix_triclinic_pass",
                            {index, index},
                            {x, box, widths, reach, cellWidth, excluded, counts,
                             entries, order});
                   Value one = oneValues[0];
                   Value oneNot = oneValues[1];
                   return {one, oneNot};
                 },
                 [&]() -> Values {
                   auto allValues =
                       call("mdrt_gpu_build_neighbors_matrix_triclinic_images",
                            {index, index},
                            {x, box, widths, reach, cellWidth, excluded, counts,
                             entries, order});
                   Value all = allValues[0];
                   Value allNot = allValues[1];
                   return {all, allNot};
                 });
             Value largest = largestValues[0];
             Value notNumbers = largestValues[1];
             func::ReturnOp::create(b, loc, ValueRange{largest, notNumbers});
           });
}

void MatrixBuild::emitSpatialOrder() {
  function(
      "mdrt_gpu_spatial_order",
      b.getFunctionType({positions, boxType, f64, buffer, buffer}, {}),
      [&](ValueRange arguments) {
        Value x = arguments[0];
        Value box = arguments[1];
        Value width = arguments[2];
        Value ids = arguments[3];
        Value order = arguments[4];
        Value c0 = cx(0);
        Value c1 = cx(1);
        Value threads = cx(128);
        Value chunk = cx(256);
        Value n = dim(x, c0);
        Value lx = extract(box, 0);
        Value ly = extract(box, 1);
        Value lz = extract(box, 2);
        Value unit = cd(1.0);
        Value ilx = divf(unit, lx);
        Value ily = divf(unit, ly);
        Value ilz = divf(unit, lz);
        Value nx = call("mdrt_gpu_cell_count", {index}, {lx, width})[0];
        Value ny = call("mdrt_gpu_cell_count", {index}, {ly, width})[0];
        Value nz = call("mdrt_gpu_cell_count", {index}, {lz, width})[0];
        Value nxy = muli(nx, ny);
        Value cells = muli(nxy, nz);
        Value cells1 = addi(cells, c1);
        Value cellChunks = call("mdrt_gpu_grid", {index}, {cells, chunk})[0];
        Value gridN = call("mdrt_gpu_grid", {index}, {n, threads})[0];
        Value gridCells = call("mdrt_gpu_grid", {index}, {cells, threads})[0];
        Value gridCellChunks =
            call("mdrt_gpu_grid", {index}, {cellChunks, threads})[0];
        Value key = alloc(buffer, {n});
        Value held = alloc(buffer, {cells});
        Value start = alloc(buffer, {cells1});
        Value cursor = alloc(buffer, {cells});
        Value cellSums = alloc(buffer, {cellChunks});
        {
          auto launch =
              gpu::LaunchOp::create(b, loc, gridCells, c1, c1, threads, c1, c1);
          OpBuilder::InsertionGuard scope(b);
          b.setInsertionPointToEnd(&launch.getBody().front());
          Value bx = launch.getBlockIds().x;
          Value tx = launch.getThreadIds().x;
          Value base = muli(bx, threads);
          Value c = addi(base, tx);
          Value inside = cmpi(CmpIPredicate::ult, c, cells);
          ifThen(inside, [&] {
            Value none = ci(0);
            store(none, held, {c});
          });
          gpu::TerminatorOp::create(b, loc);
        }
        // Each particle's cell, its key; the counts of the cells, their offsets
        // (the sums of chunks of 256 cells, the offsets of the chunks in one
        // thread, and the offsets of the cells of each chunk), each particle at
        // the next slot of its cell, and the particles of each cell in the
        // order of `ids`, by insertion: a counting sort (md_alg.sort_by_key),
        // whose counts the launch above cleared.
        {
          auto sort = md_alg::SortByKeyOp::create(
              b, loc, n, cells, md_alg::Order::Stable, key, held, start, cursor,
              order, ids, ValueRange{}, ValueRange{}, true, false,
              ValueRange{gridN, threads, c1}, chunked, cellSums,
              ValueRange{cellChunks, chunk, gridCellChunks, threads, c1},
              md_alg::SortStrategyAttr::get(context,
                                            md_alg::SortStrategy::Insertion),
              Value(), ValueRange{gridCells, threads, c1}, nullptr);
          OpBuilder::InsertionGuard scope(b);
          Block *keyBody = b.createBlock(&sort.getKey());
          Value i = keyBody->addArgument(index, loc);
          Value zeroIndex = cx(0);
          Value firstIndex = cx(1);
          Value secondIndex = cx(2);
          Value xi = load(x, {i, zeroIndex});
          Value yi = load(x, {i, firstIndex});
          Value zi = load(x, {i, secondIndex});
          Value wx = wrapCoordinate(xi, lx, ilx);
          Value x3 = mulf(wx, ilx);
          Value cellX = binCoordinate(x3, nx, zeroIndex, firstIndex);
          Value wy = wrapCoordinate(yi, ly, ily);
          Value y3 = mulf(wy, ily);
          Value cy = binCoordinate(y3, ny, zeroIndex, firstIndex);
          Value wz = wrapCoordinate(zi, lz, ilz);
          Value z3 = mulf(wz, ilz);
          Value cz = binCoordinate(z3, nz, zeroIndex, firstIndex);
          Value zy = muli(cz, ny);
          Value row = addi(zy, cy);
          Value rows = muli(row, nx);
          Value k = addi(rows, cellX);
          md_alg::BinOp::create(b, loc, k);
        }
        dealloc(key);
        dealloc(held);
        dealloc(start);
        dealloc(cursor);
        dealloc(cellSums);
        func::ReturnOp::create(b, loc, ValueRange{});
      });
}

} // namespace

void kernels::emitMatrixBuild(ModuleOp module, bool explicitImages, bool uniqueOwned) {
  OpBuilder builder = OpBuilder::atBlockEnd(module.getBody());
  MatrixBuild matrix(builder, module.getLoc());
  matrix.emitCellCount();
  matrix.emitCellRange();
  matrix.emitCellWidth();
  matrix.emitGrid("mdrt_gpu_grid");
  matrix.emitExcludedPartner("mdrt_gpu_excluded_partner");
  matrix.emitMatrixSort();
  matrix.emitSearch("mdrt_gpu_build_neighbors_matrix", false, false);
  if (uniqueOwned)
    matrix.emitSearch("mdrt_gpu_build_neighbors_matrix_unique", false, false, false, true);
  if (explicitImages)
    matrix.emitSearch("mdrt_gpu_build_neighbors_matrix_explicit", false, false, true);
  matrix.emitSearch("mdrt_gpu_build_neighbors_matrix_triclinic_pass", true,
                    false);
  matrix.emitMatrixImageWithin();
  matrix.emitSearch("mdrt_gpu_build_neighbors_matrix_triclinic_images", true,
                    true);
  matrix.emitBuildNeighborsMatrixTriclinic();
  matrix.emitSpatialOrder();
}

namespace mdir {
#define GEN_PASS_DEF_ADDMATRIXBUILD
#include "mdir/Conversion/Passes.h.inc"
} // namespace mdir

namespace {
class AddMatrixBuild : public mdir::impl::AddMatrixBuildBase<AddMatrixBuild> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    OwningOpRef<ModuleOp> definitions = ModuleOp::create(module.getLoc());
    kernels::emitMatrixBuild(*definitions);
    for (func::FuncOp function : definitions->getOps<func::FuncOp>()) {
      if (auto existing =
              module.lookupSymbol<func::FuncOp>(function.getName())) {
        if (!existing.isDeclaration() ||
            existing.getFunctionType() != function.getFunctionType()) {
          existing.emitOpError("conflicts with the GPU matrix build");
          return signalPassFailure();
        }
        existing.erase();
      }
    }
    module.getBody()->getOperations().splice(
        module.getBody()->end(), definitions->getBody()->getOperations());
  }
};
} // namespace
