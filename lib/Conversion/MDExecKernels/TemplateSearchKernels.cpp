// Periodic covering search [AllenTildesley2017], with the image retained
// for the scalar window and the covering bound of D242.
#include "mdir/Conversion/TemplateSearchKernels.h"
#include "mdir/Conversion/TemplateStageKernels.h"
#include "mdir/Conversion/TemplateImageSearchKernels.h"
#include "mdir/Conversion/TemplateImagePolicy.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mdir/Conversion/TemplateSearchPreparation.h"
#include "mdir/Conversion/ReductionEmitter.h"
#include "mdir/Conversion/TemplateSearchGeometry.h"
#include "mdir/Conversion/SpatialFrameEmitter.h"
#include "mdir/Conversion/CompactionEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/STLExtras.h"
#include <string>
#include <limits>

using namespace mlir;
using namespace mdir::kernels;
namespace {
static Value integer(OpBuilder &b, Location loc, int64_t n) {
  return arith::ConstantIntOp::create(b, loc, n, 64);
}
static Value word(OpBuilder &b, Location loc, int64_t n) {
  return arith::ConstantIntOp::create(b, loc, n, 32);
}
static Value asIndex(OpBuilder &b, Location loc, Value n) {
  return arith::IndexCastUIOp::create(b, loc, b.getIndexType(), n);
}
static Value asWide(OpBuilder &b, Location loc, Value n) {
  return arith::IndexCastUIOp::create(b, loc, b.getI64Type(), n);
}
static Value load(OpBuilder &b, Location loc, Value buffer, Value index) {
  return memref::LoadOp::create(b, loc, buffer, ValueRange{index});
}
static Value record(OpBuilder &b, Location loc, Value buffer, Value row, unsigned column) {
  Value at = arith::AddIOp::create(b, loc,
      arith::MulIOp::create(b, loc, row, createIndex(b, loc, 4)), createIndex(b, loc, column));
  return load(b, loc, buffer, at);
}
static Value floorInteger(OpBuilder &b, Location loc, Value real) {
  return arith::FPToSIOp::create(b, loc, b.getI64Type(), math::FloorOp::create(b, loc, real));
}
static Value floorDivide(OpBuilder &b, Location loc, Value a, Value positive) {
  Value quotient = arith::DivSIOp::create(b, loc, a, positive);
  Value remainder = arith::RemSIOp::create(b, loc, a, positive);
  Value negative = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                          remainder, integer(b, loc, 0));
  return arith::SelectOp::create(b, loc, negative,
      arith::SubIOp::create(b, loc, quotient, integer(b, loc, 1)), quotient);
}
// Search-local cache view: the group plan is complete before these four
// i32 words per unit are reused as Cartesian f32 coordinates and an ID.
static Value cartesianAnchor(OpBuilder &b, Location loc,
    const TemplateSearchBuffers &v, Value unit) {
  SmallVector<Value, 3> coordinates;
  for (int64_t axis = 0; axis != 3; ++axis) {
    Value index = arith::AddIOp::create(b, loc,
        arith::MulIOp::create(b, loc, unit, createIndex(b, loc, 4)),
        createIndex(b, loc, axis));
    coordinates.push_back(arith::BitcastOp::create(b, loc, b.getF32Type(),
        load(b, loc, v.spatial.sortScratch, index)));
  }
  return vector::FromElementsOp::create(b, loc,
      VectorType::get({3}, b.getF32Type()), coordinates);
}
static SpatialBoundingBox cartesianWarpBox(OpBuilder &b, Location loc,
    Value point, Value valid) {
  Value infinity = vector::BroadcastOp::create(b, loc, point.getType(),
      createReal(b, loc, b.getF32Type(), std::numeric_limits<float>::infinity()));
  Value minimum = arith::NegFOp::create(b, loc, emitWarpReduce(b, loc,
      arith::NegFOp::create(b, loc, arith::SelectOp::create(b, loc,
          valid, point, infinity)), Combiner::Max));
  Value maximum = emitWarpReduce(b, loc, arith::SelectOp::create(b, loc,
      valid, point, arith::NegFOp::create(b, loc, infinity)), Combiner::Max);
  return {minimum, maximum};
}
// IDX with the same scalar splitting as the common XOR reduction emitter.
static Value shuffle(OpBuilder &b, Location loc, Value value, Value lane) {
  Type type = value.getType();
  if (auto vector = dyn_cast<VectorType>(type)) {
    SmallVector<Value> elements;
    for (int64_t i = 0; i != vector.getNumElements(); ++i)
      elements.push_back(shuffle(b, loc, vector::ExtractOp::create(b, loc, value, i), lane));
    return vector::FromElementsOp::create(b, loc, type, elements);
  }
  auto take = [&](Value part) -> Value {
    return gpu::ShuffleOp::create(b, loc, part, lane, word(b, loc, 32),
                                   gpu::ShuffleMode::IDX).getShuffleResult();
  };
  if (type.getIntOrFloatBitWidth() == 32) return take(value);
  Value bits = arith::BitcastOp::create(b, loc, b.getI64Type(), value);
  Value shift = integer(b, loc, 32);
  Value low = arith::TruncIOp::create(b, loc, b.getI32Type(), bits);
  Value high = arith::TruncIOp::create(b, loc, b.getI32Type(),
      arith::ShRUIOp::create(b, loc, bits, shift));
  Value joined = arith::OrIOp::create(b, loc,
      arith::ExtUIOp::create(b, loc, b.getI64Type(), take(low)),
      arith::ShLIOp::create(b, loc,
          arith::ExtUIOp::create(b, loc, b.getI64Type(), take(high)), shift));
  return arith::BitcastOp::create(b, loc, type, joined);
}
static TemplateStageState forState(OpBuilder &b, Location loc, Value begin, Value end,
    Value step, TemplateStageState initial,
    llvm::function_ref<TemplateStageState(OpBuilder &, Value, TemplateStageState)> body) {
  auto loop = scf::ForOp::create(b, loc, begin, end, step, ValueRange{initial.count, initial.chunk},
      [&](OpBuilder &at, Location, Value i, ValueRange state) {
        auto next = body(at, i, {state[0], state[1]});
        scf::YieldOp::create(at, loc, ValueRange{next.count, next.chunk});
      });
  return {loop.getResult(0), loop.getResult(1)};
}
// The before region never loads: an empty row can end at the buffer tail.
static Value bound(OpBuilder &b, Location loc, Value begin, Value end, Value wanted,
    bool upper, llvm::function_ref<Value(OpBuilder &, Value)> keyAt) {
  auto loop = scf::WhileOp::create(b, loc, TypeRange{b.getIndexType(), b.getIndexType()},
      ValueRange{begin, end},
      [&](OpBuilder &at, Location, ValueRange range) {
        Value more = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, range[0], range[1]);
        scf::ConditionOp::create(at, loc, more, range);
      }, [&](OpBuilder &at, Location, ValueRange range) {
        Value middle = arith::AddIOp::create(at, loc, range[0],
            arith::DivUIOp::create(at, loc,
                arith::SubIOp::create(at, loc, range[1], range[0]), createIndex(at, loc, 2)));
        Value key = keyAt(at, middle);
        Value before = arith::CmpIOp::create(at, loc,
            upper ? arith::CmpIPredicate::ule : arith::CmpIPredicate::ult, key, wanted);
        Value left = arith::SelectOp::create(at, loc, before,
            arith::AddIOp::create(at, loc, middle, createIndex(at, loc, 1)), range[0]);
        Value right = arith::SelectOp::create(at, loc, before, range[1], middle);
        scf::YieldOp::create(at, loc, ValueRange{left, right});
      });
  return loop.getResult(0);
}
} // namespace

static void emitGeneralTemplateCandidates(OpBuilder &b, Location loc,
    const TemplateSearchBuffers &v) {
  auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
  auto real = [&](OpBuilder &at, double n) { return createReal(at, loc, at.getF64Type(), n); };
  auto boolean = [&](OpBuilder &at, bool yes) -> Value {
    return arith::ConstantOp::create(at, loc, at.getBoolAttr(yes));
  };
  Value one = ix(b, 1), block = ix(b, 64);
  Value waterRows = v.build.groups.waterGroups;
  Value atomRows = v.build.groups.atomGroups;
  Value ssBegin = arith::AddIOp::create(b, loc, waterRows, atomRows);
  // One prototype-shaped launch per class. Compile-time class constants
  // remove the other ownership/exclusion paths from each kernel. Global
  // row IDs and row-local ordinal compaction preserve the existing order.
  for (unsigned searchClass = 0; searchClass != 3; ++searchClass) {
  Value classRows = searchClass == 0 ? waterRows : atomRows;
  Value rowOffset = searchClass == 0 ? ix(b, 0) :
      (searchClass == 1 ? waterRows : ssBegin);
  Value threads = arith::MulIOp::create(b, loc, classRows, ix(b, 32));
  Value grid = arith::MaxUIOp::create(b, loc, one,
      arith::DivUIOp::create(b, loc,
          arith::AddIOp::create(b, loc, threads, ix(b, 63)), block));
  auto sharedSpace = gpu::AddressSpaceAttr::get(b.getContext(),
      gpu::GPUDialect::getWorkgroupAddressSpace());
  auto pointCacheType = MemRefType::get({32, 3}, b.getF64Type(),
      MemRefLayoutAttrInterface(), sharedSpace);
  auto cartesianCacheType = MemRefType::get({32, 3}, b.getF32Type(),
      MemRefLayoutAttrInterface(), sharedSpace);
  auto idCacheType = MemRefType::get({32}, b.getI32Type(),
      MemRefLayoutAttrInterface(), sharedSpace);
  auto launch = gpu::LaunchOp::create(b, loc, grid, one, one, block, one, one,
      /*dynamicSharedMemorySize=*/nullptr, /*asyncTokenType=*/nullptr,
      /*asyncDependencies=*/{},
      /*workgroupAttributions=*/TypeRange{pointCacheType, idCacheType, cartesianCacheType});
  Value pointCache = launch.getWorkgroupAttributionBBArgs()[0];
  Value idCache = launch.getWorkgroupAttributionBBArgs()[1];
  Value cartesianCache = launch.getWorkgroupAttributionBBArgs()[2];
  OpBuilder k = OpBuilder::atBlockEnd(&launch.getBody().front());
  Value thread = arith::AddIOp::create(k, loc,
      arith::MulIOp::create(k, loc, launch.getBlockIds().x, block), launch.getThreadIds().x);
  Value localRow = arith::DivUIOp::create(k, loc, thread, ix(k, 32));
  Value row = arith::AddIOp::create(k, loc, localRow, rowOffset);
  Value validRow = arith::CmpIOp::create(k, loc, arith::CmpIPredicate::ult, localRow, classRows);
  scf::IfOp::create(k, loc, validRow, [&](OpBuilder &at, Location) {
    Value lane = arith::IndexCastOp::create(at, loc, at.getI32Type(),
        arith::RemUIOp::create(at, loc, thread, ix(at, 32)));
    Value laneIndex = asIndex(at, loc, lane);
    Value rowWord = arith::IndexCastOp::create(at, loc, at.getI32Type(), row);
    TemplateStageBuffers stage{v.build.stageUnits, v.build.stageEntries, v.counters,
        v.build.counts, asWide(at, loc, v.build.unitCapacity), rowWord, lane};
    TemplateStageState initial{integer(at, loc, 0), integer(at, loc, -1)};
    Value ww = boolean(at, searchClass == 0);
    Value sw = boolean(at, searchClass == 1);
    Value ss = boolean(at, searchClass == 2);
    Value active = arith::SelectOp::create(at, loc, ww, v.spatial.waterActive,
        arith::SelectOp::create(at, loc, sw,
            arith::OrIOp::create(at, loc, v.spatial.waterActive, v.spatial.atomActive), v.spatial.atomActive));
    auto chosen = scf::IfOp::create(at, loc, active,
        [&](OpBuilder &search, Location) {
          Value classRow = localRow;
          Value firstGroup = arith::SelectOp::create(search, loc, ww, row,
              arith::AddIOp::create(search, loc, v.spatial.waters, classRow));
          Value first = asIndex(search, loc,
              load(search, loc, v.build.groups.starts, firstGroup));
          Value firstLength = asIndex(search, loc,
              load(search, loc, v.build.groups.lengths, firstGroup));
          Value firstLocal = asIndex(search, loc,
              arith::AndIOp::create(search, loc, lane, word(search, loc, 15)));
          Value firstPlace = arith::AddIOp::create(search, loc, first,
              arith::MinUIOp::create(search, loc, firstLocal,
                  arith::SubIOp::create(search, loc, firstLength, ix(search, 1))));
          Value firstUnit = load(search, loc, v.spatial.order, firstPlace);
          Value firstValid = arith::CmpIOp::create(search, loc, arith::CmpIPredicate::ult,
              firstLocal, firstLength);
          Value safeUnit = arith::SelectOp::create(search, loc, firstValid,
                                                    firstUnit, word(search, loc, 0));
          Value firstPoint = loadElement(search, loc, v.spatial.points, asIndex(search, loc, safeUnit));
          auto pointType = cast<VectorType>(firstPoint.getType());
          auto fastType = VectorType::get({3}, search.getF32Type());
          Value firstCartesian = cartesianAnchor(search, loc, v, asIndex(search, loc, safeUnit));
          Value fastCell = arith::TruncFOp::create(search, loc,
              VectorType::get(cast<VectorType>(v.cell.getType()).getShape(), search.getF32Type()), v.cell);
          // Immutable per-row anchors, following the shared-anchor search
          // structure of the measured prototype while retaining native f64.
          // Only the lower half writes; duplicate upper-half stores would race.
          Value cacheBase = arith::MulIOp::create(search, loc,
              arith::DivUIOp::create(search, loc,
                  launch.getThreadIds().x, ix(search, 32)), ix(search, 16));
          Value writesCache = arith::CmpIOp::create(search, loc,
              arith::CmpIPredicate::ult, lane, word(search, loc, 16));
          scf::IfOp::create(search, loc, writesCache, [&](OpBuilder &store, Location) {
            Value slot = arith::AddIOp::create(store, loc, cacheBase, laneIndex);
            for (int64_t axis = 0; axis != 3; ++axis) {
              memref::StoreOp::create(store, loc,
                  vector::ExtractOp::create(store, loc, firstPoint, axis),
                  pointCache, ValueRange{slot, ix(store, axis)});
              memref::StoreOp::create(store, loc,
                  vector::ExtractOp::create(store, loc, firstCartesian, axis),
                  cartesianCache, ValueRange{slot, ix(store, axis)});
            }
            memref::StoreOp::create(store, loc, safeUnit, idCache, ValueRange{slot});
            scf::YieldOp::create(store, loc);
          });
          // The sibling row may be absent/inactive: synchronize only this warp.
          NVVM::SyncWarpOp::create(search, loc, word(search, loc, -1));
          auto firstFrame = emitSpatialReferenceFrame(search, loc, firstPoint);
          // Query each distinct frame of this at-most-sixteen-unit group:
          // lattice-frame subset in the wrapped key frame. The stored image
          // is translated back before the unchanged f64 pair predicate.
          auto framed = forState(search, loc, ix(search, 0), ix(search, 16),
              ix(search, 1), initial,
              [&](OpBuilder &frameBody, Value representative, TemplateStageState frameState) {
            Value frame = shuffle(frameBody, loc, firstFrame.lattice,
                arith::IndexCastOp::create(frameBody, loc, frameBody.getI32Type(), representative));
            Value member = firstValid;
            for (int64_t axis = 0; axis != 3; ++axis) {
              Value same = arith::CmpFOp::create(frameBody, loc, arith::CmpFPredicate::OEQ,
                  vector::ExtractOp::create(frameBody, loc, firstFrame.lattice, axis),
                  vector::ExtractOp::create(frameBody, loc, frame, axis));
              member = arith::AndIOp::create(frameBody, loc, member, same);
            }
            Value lowerHalf = arith::CmpIOp::create(frameBody, loc, arith::CmpIPredicate::ult,
                lane, word(frameBody, loc, 16));
            Value members = emitWarpBallot(frameBody, loc,
                arith::AndIOp::create(frameBody, loc, member, lowerHalf));
            Value bit = arith::ShLIOp::create(frameBody, loc, word(frameBody, loc, 1),
                arith::IndexCastOp::create(frameBody, loc, frameBody.getI32Type(), representative));
            Value present = arith::CmpIOp::create(frameBody, loc, arith::CmpIPredicate::ne,
                arith::AndIOp::create(frameBody, loc, members, bit), word(frameBody, loc, 0));
            Value earlier = arith::AndIOp::create(frameBody, loc, members,
                arith::SubIOp::create(frameBody, loc, bit, word(frameBody, loc, 1)));
            Value firstOfFrame = arith::AndIOp::create(frameBody, loc, present,
                arith::CmpIOp::create(frameBody, loc, arith::CmpIPredicate::eq,
                    earlier, word(frameBody, loc, 0)));
            auto query = scf::IfOp::create(frameBody, loc, firstOfFrame,
                [&](OpBuilder &search, Location) {
                auto box = emitSpatialWarpBox(search, loc, firstFrame.wrapped, member);
                Value minimum = box.minimum, maximum = box.maximum;
                auto cartesianBox = cartesianWarpBox(search, loc, firstCartesian, member);
                Value cellNorm = real(search, 0);
                for (int64_t axis = 0; axis != (isTriclinic(v.cell) ? 6 : 3); ++axis)
                  cellNorm = arith::AddFOp::create(search, loc, cellNorm,
                      math::AbsFOp::create(search, loc,
                          vector::ExtractOp::create(search, loc, v.cell, axis)));

                Value reach;
                if (v.reaches) {
                  Value wwReach = vector::ExtractOp::create(search, loc, v.reaches, 0);
                  Value swReach = vector::ExtractOp::create(search, loc, v.reaches, 1);
                  Value ssReach = vector::ExtractOp::create(search, loc, v.reaches, 2);
                  reach = arith::SelectOp::create(search, loc, ww, wwReach,
                      arith::SelectOp::create(search, loc, sw, swReach, ssReach));
                } else {
                  Value multiplier = arith::SelectOp::create(search, loc, ww, real(search, 2),
                      arith::SelectOp::create(search, loc, sw, real(search, 1), real(search, 0)));
                  reach = arith::AddFOp::create(search, loc, v.baseReach,
                      arith::MulFOp::create(search, loc, multiplier, v.extent));
                }
                Value reach2 = arith::MulFOp::create(search, loc, reach, reach);
                Value widths = isTriclinic(v.cell) ? emitFaceWidths(search, loc, v.cell) : v.cell;
                Value radius = arith::DivFOp::create(search, loc,
                    vector::BroadcastOp::create(search, loc, pointType, reach), widths);
                // This widens only the fractional candidate box, not the radial
                // predicate. It covers the rounding of reciprocal face widths.
                radius = arith::MulFOp::create(search, loc, radius,
                    vector::BroadcastOp::create(search, loc, pointType, real(search, 1.0 + 0x1p-48)));
                Value low = arith::SubFOp::create(search, loc, minimum, radius);
                Value high = arith::AddFOp::create(search, loc, maximum, radius);
                auto component = [&](OpBuilder &in, Value vector, unsigned axis) -> Value {
                  return vector::ExtractOp::create(in, loc, vector, axis);
                };
                Value dimensions[] = {v.spatial.columnsX, v.spatial.columnsY};
                Value lower[2], upper[2], wideDimensions[2];
                for (unsigned axis = 0; axis != 2; ++axis) {
                  Value n = asWide(search, loc, dimensions[axis]);
                  wideDimensions[axis] = n;
                  Value scale = arith::UIToFPOp::create(search, loc, search.getF64Type(), n);
                  // f32 point/scale/product rounding can move a column key.
                  // One neighboring bin suffices below 2^22 columns; this bound
                  // also covers the largest admitted i32 unit domain.
                  Value halo = arith::AddIOp::create(search, loc, integer(search, loc, 1),
                      arith::DivUIOp::create(search, loc, n, integer(search, loc, 1 << 22)));
                  lower[axis] = arith::SubIOp::create(search, loc,
                      floorInteger(search, loc, arith::MulFOp::create(search, loc,
                          component(search, low, axis), scale)), halo);
                  upper[axis] = arith::AddIOp::create(search, loc,
                      floorInteger(search, loc, arith::MulFOp::create(search, loc,
                          component(search, high, axis), scale)),
                      arith::AddIOp::create(search, loc, halo, integer(search, loc, 1)));
                }
                Value zBegin = floorInteger(search, loc, component(search, low, 2));
                Value zEnd = arith::AddIOp::create(search, loc,
                    floorInteger(search, loc, component(search, high, 2)), integer(search, loc, 1));
                // Cover the entire original query range, not just this image.
                // Integer magnitudes are converted only for the certificate;
                // image accounting and the literal shell retain their f64/i64 graph.
                SmallVector<Value, 3> imageBounds;
                for (unsigned axis = 0; axis != 2; ++axis) {
                  Value loImage = floorDivide(search, loc, lower[axis], wideDimensions[axis]);
                  Value hiImage = floorDivide(search, loc,
                      arith::SubIOp::create(search, loc, upper[axis], integer(search, loc, 1)),
                      wideDimensions[axis]);
                  Value loReal = arith::SIToFPOp::create(search, loc, search.getF64Type(), loImage);
                  Value hiReal = arith::SIToFPOp::create(search, loc, search.getF64Type(), hiImage);
                  imageBounds.push_back(arith::MaximumFOp::create(search, loc,
                      math::AbsFOp::create(search, loc, loReal), math::AbsFOp::create(search, loc, hiReal)));
                }
                imageBounds.push_back(arith::MaximumFOp::create(search, loc,
                    math::AbsFOp::create(search, loc,
                        arith::SIToFPOp::create(search, loc, search.getF64Type(), zBegin)),
                    math::AbsFOp::create(search, loc,
                        arith::SIToFPOp::create(search, loc, search.getF64Type(),
                            arith::SubIOp::create(search, loc, zEnd, integer(search, loc, 1))))));
                Value imageEnvelope = vector::FromElementsOp::create(search, loc, pointType, imageBounds);
                auto certificate = emitTemplateSearchCertificate(search, loc,
                    v.cell, frame, imageEnvelope, reach2);
                Value allColumns = arith::MulIOp::create(search, loc, v.spatial.columnsX, v.spatial.columnsY);
                auto result = forState(search, loc, lower[0], upper[0], integer(search, loc, 1), frameState,
                    [&](OpBuilder &xloop, Value x, TemplateStageState stateX) {
                      Value sx = floorDivide(xloop, loc, x, wideDimensions[0]);
                      Value cx = asIndex(xloop, loc, arith::SubIOp::create(xloop, loc, x,
                          arith::MulIOp::create(xloop, loc, sx, wideDimensions[0])));
                      return forState(xloop, loc, lower[1], upper[1], integer(xloop, loc, 1), stateX,
                          [&](OpBuilder &yloop, Value y, TemplateStageState stateY) {
                            Value sy = floorDivide(yloop, loc, y, wideDimensions[1]);
                            Value cy = asIndex(yloop, loc, arith::SubIOp::create(yloop, loc, y,
                                arith::MulIOp::create(yloop, loc, sy, wideDimensions[1])));
                            Value odd = arith::CmpIOp::create(yloop, loc, arith::CmpIPredicate::ne,
                                arith::AndIOp::create(yloop, loc, cx, ix(yloop, 1)), ix(yloop, 0));
                            Value column = arith::AddIOp::create(yloop, loc,
                                arith::MulIOp::create(yloop, loc, cx, v.spatial.columnsY),
                                arith::SelectOp::create(yloop, loc, odd,
                                    arith::SubIOp::create(yloop, loc,
                                        arith::SubIOp::create(yloop, loc, v.spatial.columnsY, ix(yloop, 1)), cy), cy));
                            column = arith::AddIOp::create(yloop, loc, column,
                                arith::SelectOp::create(yloop, loc, ss, allColumns, ix(yloop, 0)));
                            // Prefix sums count compact descriptors; the SS partition
                            // still begins at the reserved water-unit capacity.
                            Value offset = arith::SelectOp::create(yloop, loc, ss,
                                arith::SubIOp::create(yloop, loc,
                                    v.spatial.waters, waterRows), ix(yloop, 0));
                            Value begin = arith::AddIOp::create(yloop, loc, offset,
                                asIndex(yloop, loc, load(yloop, loc, v.spatial.starts, column)));
                            Value end = arith::AddIOp::create(yloop, loc, begin,
                                asIndex(yloop, loc, load(yloop, loc, v.spatial.counts, column)));
                            return forState(yloop, loc, zBegin, zEnd, integer(yloop, loc, 1), stateY,
                                [&](OpBuilder &zloop, Value sz, TemplateStageState stateZ) {
                                  Value forwardBegin = begin;
                                  if (searchClass != 1)
                                    forwardBegin = arith::MinUIOp::create(zloop, loc, end,
                                        arith::MaxUIOp::create(zloop, loc, begin, firstGroup));
                                  // Scan direct column endpoints, as in the prototype.
                                  // The Cartesian group test rejects z nonreach before
                                  // units; no dependent group/order/z binary searches.
                                  Value firstCandidate = forwardBegin;
                                  Value lastCandidate = end;
                                  Value queryImages[] = {sx, sy, sz};
                                  SmallVector<Value, 3> queryComponents;
                                  Value magnitude = real(zloop, 4);
                                  for (int64_t axis = 0; axis != 3; ++axis) {
                                    Value query = arith::SIToFPOp::create(zloop, loc,
                                        zloop.getF64Type(), queryImages[axis]);
                                    queryComponents.push_back(query);
                                    magnitude = arith::AddFOp::create(zloop, loc, magnitude,
                                        arith::AddFOp::create(zloop, loc,
                                            math::AbsFOp::create(zloop, loc, query),
                                            math::AbsFOp::create(zloop, loc,
                                                vector::ExtractOp::create(zloop, loc, frame, axis))));
                                  }
                                  Value queryVector = vector::FromElementsOp::create(zloop, loc, pointType, queryComponents);
                                  Value fastQueryShift = emitLatticeShift(zloop, loc,
                                      arith::TruncFOp::create(zloop, loc, fastType, queryVector), fastCell);
                                  // 64 eps times the cell 1-norm and frame/image scale
                                  // covers the <32 rounded operations in box conversion,
                                  // image addition and the original displacement. Only
                                  // the prefilter widens; the exact pair test below does not.
                                  Value boxRoundoff = arith::MulFOp::create(zloop, loc,
                                      arith::MulFOp::create(zloop, loc, cellNorm, magnitude), real(zloop, 0x1p-46));

                                  SmallVector<Value, 3> imageComponents, imageIntegers;
                                  for (int64_t axis = 0; axis != 3; ++axis) {
                                    Value absolute = arith::AddIOp::create(zloop, loc, queryImages[axis],
                                        arith::FPToSIOp::create(zloop, loc, zloop.getI64Type(),
                                            vector::ExtractOp::create(zloop, loc, frame, axis)));
                                    imageIntegers.push_back(absolute);
                                    imageComponents.push_back(arith::SIToFPOp::create(
                                        zloop, loc, zloop.getF64Type(), absolute));
                                  }
                                  Value image = vector::FromElementsOp::create(zloop, loc, pointType, imageComponents);
                                  return forState(zloop, loc, firstCandidate, lastCandidate, ix(zloop, 32), stateZ,
                                      [&](OpBuilder &batch, Value cursor, TemplateStageState state) {
                                        Value position = arith::AddIOp::create(batch, loc, cursor, laneIndex);
                                        Value has = arith::CmpIOp::create(batch, loc, arith::CmpIPredicate::ult,
                                                                           position, lastCandidate);
                                        Value safe = arith::SelectOp::create(batch, loc, has, position, cursor);
                                        Value partnerGroup = safe;
                                        SmallVector<Value, 3> boxSides[2], fractionSides[2];
                                        for (int64_t side = 0; side != 2; ++side) {
                                          Value row = arith::AddIOp::create(batch, loc,
                                              arith::MulIOp::create(batch, loc, partnerGroup, ix(batch, 2)), ix(batch, side));
                                          for (int64_t axis = 0; axis != 3; ++axis) {
                                            boxSides[side].push_back(memref::LoadOp::create(batch, loc,
                                                v.groupBoxes, ValueRange{row, ix(batch, axis)}));
                                            fractionSides[side].push_back(memref::LoadOp::create(batch, loc,
                                                v.fractionalBoxes, ValueRange{row, ix(batch, axis)}));
                                          }
                                        }
                                        SpatialBoundingBox partnerBox{
                                            vector::FromElementsOp::create(batch, loc, fastType, boxSides[0]),
                                            vector::FromElementsOp::create(batch, loc, fastType, boxSides[1])};
                                        Value lowGap = arith::SubFOp::create(batch, loc, cartesianBox.minimum,
                                            arith::AddFOp::create(batch, loc, partnerBox.maximum, fastQueryShift));
                                        Value highGap = arith::SubFOp::create(batch, loc,
                                            arith::AddFOp::create(batch, loc, partnerBox.minimum, fastQueryShift), cartesianBox.maximum);
                                        Value gap = arith::MaximumFOp::create(batch, loc,
                                            arith::MaximumFOp::create(batch, loc, lowGap, highGap),
                                            vector::BroadcastOp::create(batch, loc, fastType,
                                                createReal(batch, loc, batch.getF32Type(), 0)));
                                        Value boxSquare = arith::MulFOp::create(batch, loc, gap, gap);
                                        Value boxDistance = vector::ReductionOp::create(batch, loc,
                                            vector::CombiningKind::ADD, boxSquare);
                                        Value finiteBox = arith::CmpFOp::create(batch, loc, arith::CmpFPredicate::OLT,
                                            math::AbsFOp::create(batch, loc, boxDistance),
                                            createReal(batch, loc, batch.getF32Type(), std::numeric_limits<float>::infinity()));
                                        Value boxReject = arith::AndIOp::create(batch, loc,
                                            arith::AndIOp::create(batch, loc, certificate.enabled, finiteBox),
                                            arith::CmpFOp::create(batch, loc, arith::CmpFPredicate::OGT,
                                                boxDistance, certificate.boxUpper));
                                        Value boxMayReach = arith::XOrIOp::create(batch, loc,
                                            boxReject, boolean(batch, true));
                                        SpatialBoundingBox partnerFraction{
                                            vector::FromElementsOp::create(batch, loc, pointType, fractionSides[0]),
                                            vector::FromElementsOp::create(batch, loc, pointType, fractionSides[1])};
                                        Value fractionRoundoff = arith::DivFOp::create(batch, loc,
                                            vector::BroadcastOp::create(batch, loc, pointType,
                                                arith::MulFOp::create(batch, loc, boxRoundoff, real(batch, 4))), widths);
                                        auto single = emitSpatialSingleImage(batch, loc,
                                            box, partnerFraction, radius, fractionRoundoff);
                                        Value sameImage = boolean(batch, true);
                                        for (int64_t axis = 0; axis != 3; ++axis)
                                          sameImage = arith::AndIOp::create(batch, loc, sameImage,
                                              arith::CmpFOp::create(batch, loc, arith::CmpFPredicate::OEQ,
                                                  vector::ExtractOp::create(batch, loc, single.image, axis),
                                                  queryComponents[axis]));
                                        Value imageMayReach = arith::OrIOp::create(batch, loc, sameImage,
                                            arith::XOrIOp::create(batch, loc, single.proved, boolean(batch, true)));
                                        has = arith::AndIOp::create(batch, loc, has,
                                            arith::AndIOp::create(batch, loc, boxMayReach, imageMayReach));
                                        // Reject entire groups before any unit loads or broadcasts.
                                        // The ballot and survivor loop are uniform across the warp.
                                        Value survivors = emitWarpBallot(batch, loc, has);
                                        auto selected = scf::WhileOp::create(batch, loc,
                                            TypeRange{batch.getI32Type(), batch.getI64Type(), batch.getI64Type()},
                                            ValueRange{survivors, state.count, state.chunk},
                                            [&](OpBuilder &test, Location, ValueRange carried) {
                                              Value more = arith::CmpIOp::create(test, loc,
                                                  arith::CmpIPredicate::ne, carried[0], word(test, loc, 0));
                                              scf::ConditionOp::create(test, loc, more, carried);
                                            }, [&](OpBuilder &units, Location, ValueRange carried) {
                                          Value firstLane = math::CountTrailingZerosOp::create(units, loc, carried[0]);
                                          Value rest = arith::AndIOp::create(units, loc, carried[0],
                                              arith::SubIOp::create(units, loc, carried[0], word(units, loc, 1)));
                                          Value two = arith::CmpIOp::create(units, loc,
                                              arith::CmpIPredicate::ne, rest, word(units, loc, 0));
                                          // cttz receives a nonzero value even for a singleton batch.
                                          Value secondLane = math::CountTrailingZerosOp::create(units, loc,
                                              arith::SelectOp::create(units, loc, two, rest, carried[0]));
                                          Value remaining = arith::AndIOp::create(units, loc, rest,
                                              arith::SubIOp::create(units, loc, rest, word(units, loc, 1)));
                                          Value lower = arith::CmpIOp::create(units, loc,
                                              arith::CmpIPredicate::ult, lane, word(units, loc, 16));
                                          Value sourceLane = arith::SelectOp::create(units, loc,
                                              lower, firstLane, secondLane);
                                          Value selectedGroup = asIndex(units, loc, shuffle(units, loc,
                                              arith::IndexCastOp::create(units, loc, units.getI32Type(), partnerGroup), sourceLane));
                                          Value start = asIndex(units, loc,
                                              load(units, loc, v.build.groups.starts, selectedGroup));
                                          Value length = asIndex(units, loc,
                                              load(units, loc, v.build.groups.lengths, selectedGroup));
                                          Value local = asIndex(units, loc,
                                              arith::AndIOp::create(units, loc, lane, word(units, loc, 15)));
                                          Value partnerPlace = arith::AddIOp::create(units, loc, start,
                                              arith::MinUIOp::create(units, loc, local,
                                                  arith::SubIOp::create(units, loc, length, ix(units, 1))));
                                          Value partner = load(units, loc, v.spatial.order, partnerPlace);
                                          Value partnerIndex = asIndex(units, loc, partner);
                                          Value partnerPoint = loadElement(units, loc, v.spatial.points, partnerIndex);
                                          Value partnerCartesian = cartesianAnchor(units, loc, v, partnerIndex);
                                          Value shiftedCartesian = arith::AddFOp::create(units, loc,
                                              partnerCartesian, fastQueryShift);
                                          Value has = arith::AndIOp::create(units, loc,
                                              arith::OrIOp::create(units, loc, lower, two),
                                              arith::CmpIOp::create(units, loc, arith::CmpIPredicate::ult, local, length));
                                          TemplateStageState state{carried[1], carried[2]};
                                        SmallVector<Value, 3> wrappedPartner;
                                        Value code = integer(units, loc, 0);
                                        Value imageFits = boolean(units, true);
                                        for (unsigned axis = 0; axis != 3; ++axis) {
                                          Value coordinate = component(units, partnerPoint, axis);
                                          Value frame = math::FloorOp::create(units, loc, coordinate);
                                          wrappedPartner.push_back(arith::SubFOp::create(units, loc, coordinate, frame));
                                          Value offset = arith::SubIOp::create(units, loc, imageIntegers[axis],
                                              arith::FPToSIOp::create(units, loc, units.getI64Type(), frame));
                                          Value lower = arith::CmpIOp::create(units, loc, arith::CmpIPredicate::sge,
                                              offset, integer(units, loc, -MDRT_TEMPLATE_IMAGE_OFFSET));
                                          Value upper = arith::CmpIOp::create(units, loc, arith::CmpIPredicate::slt,
                                              offset, integer(units, loc, MDRT_TEMPLATE_IMAGE_OFFSET));
                                          imageFits = arith::AndIOp::create(units, loc, imageFits,
                                              arith::AndIOp::create(units, loc, lower, upper));
                                          code = arith::OrIOp::create(units, loc, code,
                                              arith::ShLIOp::create(units, loc,
                                                  arith::AddIOp::create(units, loc, offset,
                                                      integer(units, loc, MDRT_TEMPLATE_IMAGE_OFFSET)),
                                                  integer(units, loc, MDRT_TEMPLATE_IMAGE_BITS * axis)));
                                        }
                                        Value imageCode = arith::TruncIOp::create(units, loc, units.getI32Type(), code);
                                        Value normalizedPartner = vector::FromElementsOp::create(units, loc, pointType, wrappedPartner);
                                        Value shifted = arith::AddFOp::create(units, loc, normalizedPartner, image);
                                        Value mask = word(units, loc, 0);
                                        for (unsigned i = 0; i != 16; ++i) {
                                          Value slot = arith::AddIOp::create(units, loc,
                                              cacheBase, ix(units, i));
                                          SmallVector<Value, 3> coordinates, fastCoordinates;
                                          for (int64_t axis = 0; axis != 3; ++axis) {
                                            coordinates.push_back(memref::LoadOp::create(units, loc,
                                                pointCache, ValueRange{slot, ix(units, axis)}));
                                            fastCoordinates.push_back(memref::LoadOp::create(units, loc,
                                                cartesianCache, ValueRange{slot, ix(units, axis)}));
                                          }
                                          Value cartesianI = vector::FromElementsOp::create(units, loc,
                                              fastType, fastCoordinates);
                                          Value pointI = vector::FromElementsOp::create(units, loc,
                                              pointType, coordinates);
                                          Value unitI = load(units, loc, idCache, slot);
                                          Value validI = arith::CmpIOp::create(units, loc,
                                              arith::CmpIPredicate::ne,
                                              arith::AndIOp::create(units, loc, members,
                                                  word(units, loc, int64_t(1) << i)), word(units, loc, 0));
                                          Value owns = arith::OrIOp::create(units, loc, sw,
                                              arith::CmpIOp::create(units, loc, arith::CmpIPredicate::ult,
                                                  arith::AddIOp::create(units, loc, first, ix(units, i)), partnerPlace));
                                          Value eligible = arith::AndIOp::create(units, loc, has,
                                              arith::AndIOp::create(units, loc, validI, owns));
                                          auto accepted = scf::IfOp::create(units, loc, eligible,
                                              [&](OpBuilder &pair, Location) {
                                                Value fastDisplacement = arith::SubFOp::create(pair, loc, cartesianI, shiftedCartesian);
                                                Value fastSquare = arith::MulFOp::create(pair, loc, fastDisplacement, fastDisplacement);
                                                Value fastDistance = vector::ReductionOp::create(pair, loc,
                                                    vector::CombiningKind::ADD, fastSquare);
                                                Value finiteDistance = arith::CmpFOp::create(pair, loc, arith::CmpFPredicate::OLT,
                                                    math::AbsFOp::create(pair, loc, fastDistance),
                                                    createReal(pair, loc, pair.getF32Type(), std::numeric_limits<float>::infinity()));
                                                Value fastEnabled = arith::AndIOp::create(pair, loc,
                                                    certificate.enabled, finiteDistance);
                                                Value fastAccept = arith::AndIOp::create(pair, loc,
                                                    arith::AndIOp::create(pair, loc, fastEnabled, certificate.acceptEnabled),
                                                    arith::CmpFOp::create(pair, loc, arith::CmpFPredicate::OLE,
                                                        fastDistance, certificate.lower));
                                                Value fastReject = arith::AndIOp::create(pair, loc, fastEnabled,
                                                    arith::CmpFOp::create(pair, loc, arith::CmpFPredicate::OGT,
                                                        fastDistance, certificate.upper));
                                                Value classified = arith::OrIOp::create(pair, loc, fastAccept, fastReject);
                                                auto decision = scf::IfOp::create(pair, loc, classified,
                                                    [&](OpBuilder &quick, Location) {
                                                      scf::YieldOp::create(quick, loc, ValueRange{fastAccept});
                                                    }, [&](OpBuilder &exact, Location) {
                                                Value raw = arith::SubFOp::create(exact, loc, pointI, shifted);
                                                Value displacement = emitLatticeShift(exact, loc, raw, v.cell);
                                                Value square = arith::MulFOp::create(exact, loc, displacement, displacement);
                                                Value distance = vector::ReductionOp::create(exact, loc,
                                                    vector::CombiningKind::ADD, square);
                                                Value inside = arith::CmpFOp::create(exact, loc,
                                                    arith::CmpFPredicate::OLE, distance, reach2);
                                                scf::YieldOp::create(exact, loc, ValueRange{inside});
                                                    });
                                                Value inside = decision.getResult(0);
                                                auto kept = scf::IfOp::create(pair, loc, inside,
                                                    [&](OpBuilder &near, Location) {
                                                      auto allowed = scf::IfOp::create(near, loc, ss,
                                                          [&](OpBuilder &atom, Location) {
                                                            Value w = arith::IndexCastOp::create(atom, loc, atom.getI32Type(), v.spatial.waters);
                                                            Value a = asIndex(atom, loc, arith::SubIOp::create(atom, loc, unitI, w));
                                                            Value other = arith::SubIOp::create(atom, loc, partner, w);
                                                            Value start = asIndex(atom, loc, load(atom, loc, v.excludedRows, a));
                                                            Value stop = asIndex(atom, loc, load(atom, loc, v.excludedRows,
                                                                arith::AddIOp::create(atom, loc, a, ix(atom, 1))));
                                                            Value found = bound(atom, loc, start, stop, other, false,
                                                                [&](OpBuilder &in, Value j) { return load(in, loc, v.excludedNeighbors, j); });
                                                            Value exists = arith::CmpIOp::create(atom, loc, arith::CmpIPredicate::ult, found, stop);
                                                            auto excluded = scf::IfOp::create(atom, loc, exists,
                                                                [&](OpBuilder &in, Location) {
                                                                  Value same = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::eq,
                                                                      load(in, loc, v.excludedNeighbors, found), other);
                                                                  scf::YieldOp::create(in, loc, ValueRange{same});
                                                                }, [&](OpBuilder &in, Location) {
                                                                  scf::YieldOp::create(in, loc, ValueRange{boolean(in, false)});
                                                                });
                                                            Value notExcluded = arith::XOrIOp::create(atom, loc, excluded.getResult(0), boolean(atom, true));
                                                            scf::YieldOp::create(atom, loc, ValueRange{notExcluded});
                                                          }, [&](OpBuilder &in, Location) {
                                                            scf::YieldOp::create(in, loc, ValueRange{boolean(in, true)});
                                                          });
                                                      scf::YieldOp::create(near, loc, allowed.getResults());
                                                    }, [&](OpBuilder &in, Location) {
                                                      scf::YieldOp::create(in, loc, ValueRange{boolean(in, false)});
                                                    });
                                                scf::YieldOp::create(pair, loc, kept.getResults());
                                              }, [&](OpBuilder &in, Location) {
                                                scf::YieldOp::create(in, loc, ValueRange{boolean(in, false)});
                                              });
                                          mask = arith::OrIOp::create(units, loc, mask,
                                              arith::SelectOp::create(units, loc, accepted.getResult(0),
                                                  word(units, loc, int64_t(1) << i), word(units, loc, 0)));
                                        }
                                        Value keep = arith::CmpIOp::create(units, loc, arith::CmpIPredicate::ne,
                                                                             mask, word(units, loc, 0));
                                        Value badImage = arith::AndIOp::create(units, loc, keep,
                                            arith::XOrIOp::create(units, loc, imageFits, boolean(units, true)));
                                        scf::IfOp::create(units, loc, badImage, [&](OpBuilder &bad, Location) {
                                          memref::AtomicRMWOp::create(bad, loc, arith::AtomicRMWKind::ori,
                                              integer(bad, loc, TEMPLATE_BUILD_INDEX_OVERFLOW), v.counters,
                                              ValueRange{ix(bad, TEMPLATE_COUNTER_FLAGS)});
                                          scf::YieldOp::create(bad, loc);
                                        });
                                        keep = arith::AndIOp::create(units, loc, keep, imageFits);
                                        Value fields[] = {partner, imageCode, mask, word(units, loc, 0)};
                                        auto next = emitAppendTemplateStage(units, loc, stage, state, keep, fields);
                                        scf::YieldOp::create(units, loc,
                                            ValueRange{remaining, next.count, next.chunk});
                                        });
                                        return TemplateStageState{selected.getResult(1), selected.getResult(2)};
                                      });
                                });
                          });
                    });
                scf::YieldOp::create(search, loc, ValueRange{result.count, result.chunk});
                }, [&](OpBuilder &skip, Location) {
                  scf::YieldOp::create(skip, loc, ValueRange{frameState.count, frameState.chunk});
                });
            return TemplateStageState{query.getResult(0), query.getResult(1)};
          });
          scf::YieldOp::create(search, loc, ValueRange{framed.count, framed.chunk});
        }, [&](OpBuilder &copy, Location) {
          Value tag = arith::SelectOp::create(copy, loc, ww, ix(copy, 0),
              arith::SelectOp::create(copy, loc, sw, ix(copy, 1), ix(copy, 2)));
          Value oldRange = arith::AddIOp::create(copy, loc,
              ix(copy, TEMPLATE_PAIR_OUTER_RANGES),
              arith::MulIOp::create(copy, loc, tag, ix(copy, 6)));
          Value oldBegin = asIndex(copy, loc,
              load(copy, loc, v.build.arena, oldRange));
          Value localRow = arith::SubIOp::create(copy, loc, row,
              arith::SelectOp::create(copy, loc, ww, ix(copy, 0),
                  arith::SelectOp::create(copy, loc, sw, waterRows, ssBegin)));
          Value oldRow = arith::AddIOp::create(copy, loc, oldBegin, localRow);
          Value begin = asIndex(copy, loc, record(copy, loc, v.build.outerRows, oldRow, 2));
          Value count = asIndex(copy, loc, record(copy, loc, v.build.outerRows, oldRow, 3));
          auto result = forState(copy, loc, ix(copy, 0), count, ix(copy, 32), initial,
              [&](OpBuilder &batch, Value cursor, TemplateStageState state) {
                Value ordinal = arith::AddIOp::create(batch, loc, cursor, laneIndex);
                Value has = arith::CmpIOp::create(batch, loc, arith::CmpIPredicate::ult, ordinal, count);
                Value safe = arith::SelectOp::create(batch, loc, has, ordinal, cursor);
                Value source = arith::AddIOp::create(batch, loc, begin, safe);
                SmallVector<Value, 4> fields;
                for (unsigned column = 0; column != 4; ++column)
                  fields.push_back(record(batch, loc, v.build.outerEntries, source, column));
                return emitAppendTemplateStage(batch, loc, stage, state, has, fields);
              });
          scf::YieldOp::create(copy, loc, ValueRange{result.count, result.chunk});
        });
    emitFinishTemplateStage(at, loc, stage, {chosen.getResult(0), chosen.getResult(1)});
    scf::YieldOp::create(at, loc);
  });
  gpu::TerminatorOp::create(k, loc);
  } // WW, SW, SS
}

// The outlined host functions own distinct GPU kernels. Mode selection
// therefore never adds an image-policy branch inside a candidate kernel.
static SmallVector<Value *> searchFields(TemplateSearchBuffers &v, bool includeAbsent = false) {
  SmallVector<Value *> fields;
  auto add = [&](Value &field) { if (field || includeAbsent) fields.push_back(&field); };
  auto &p = v.spatial; auto &a = v.build; auto &g = a.groups;
  for (Value *field : {&p.points,&p.keys,&p.zKeys,&p.counts,&p.starts,&p.cursors,
       &p.items,&p.scanScratch,&p.order,&p.places,&p.sortScratch,&p.waters,&p.atoms,
       &p.columnsX,&p.columnsY,&p.scanBlocks,&p.waterActive,&p.atomActive,
       &a.arena,&a.stageUnits,&a.stageEntries,&a.counts,&a.entryOffsets,&a.workCounts,
       &a.workOffsets,&a.scanScratch,&a.outerRows,&a.innerRows,&a.outerUnits,&a.innerUnits,
       &a.outerEntries,&a.innerEntries,&a.waters,&a.atoms,&a.rows,&a.unitCapacity,&a.scanBlocks,
       &g.starts,&g.lengths,&g.columns,&g.atStart,&g.waterGroups,&g.atomGroups,&g.rows,
       &v.counters,&v.excludedRows,&v.excludedNeighbors,&v.handle,&v.imageMargin,
       &v.cell,&v.baseReach,&v.extent,&v.reaches,&v.classCells,&v.classReaches,
       &v.groupBoxes,&v.fractionalBoxes}) add(*field);
  return fields;
}

static func::FuncOp outlineSearch(OpBuilder &b, Location loc, ModuleOp module,
    const TemplateSearchBuffers &values, unsigned path) {
  TemplateSearchBuffers args = values;
  auto fields = searchFields(args);
  SmallVector<Type> types;
  for (Value *field : fields) types.push_back(field->getType());
  auto type = b.getFunctionType(types, {});
  std::string stem = std::string("__mdir_template_search_") +
      (path == 0 ? "nearest" : path == 1 ? "conditional" : "general") +
      (isTriclinic(values.cell) ? "_triclinic" : "_orthorhombic");
  auto schemaFields = searchFields(args, true);
  assert(schemaFields.size() <= 64);
  uint64_t schema = 0;
  for (unsigned i = 0; i != schemaFields.size(); ++i)
    if (*schemaFields[i]) schema |= uint64_t(1) << i;
  stem += "_s" + std::to_string(schema);
  std::string name = stem;
  unsigned suffix = 0;
  while (auto found = module.lookupSymbol<func::FuncOp>(name)) {
    if (found.getFunctionType() == type) return found;
    name = stem + "_" + std::to_string(++suffix);
  }
  OpBuilder symbols(module.getBodyRegion());
  auto function = func::FuncOp::create(symbols, loc, name, type);
  function.setPrivate();
  Block *entry = function.addEntryBlock();
  for (auto [field, argument] : llvm::zip(fields, entry->getArguments())) *field = argument;
  OpBuilder at = OpBuilder::atBlockEnd(entry);
  if (path == 2) emitGeneralTemplateCandidates(at, loc, args);
  else emitSpecializedTemplateSearch(at, loc, args, path == 1);
  func::ReturnOp::create(at, loc);
  return function;
}

Value mdir::kernels::emitSearchTemplatePairs(OpBuilder &b, Location loc,
    ModuleOp module, const TemplateSearchBuffers &v) {
  auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
  auto real = [&](OpBuilder &at, double n) { return createReal(at, loc, at.getF64Type(), n); };
  Value one = ix(b, 1);
  // Spatial sort scratch is dead after group publication. Reuse it for
  // dense descriptor-column endpoints without changing the arena capacity.
  Value bins = arith::MulIOp::create(b, loc, ix(b, 2),
      arith::MulIOp::create(b, loc, v.spatial.columnsX, v.spatial.columnsY));
  Value columnBlock = ix(b, 256);
  Value columnGrid = arith::MaxUIOp::create(b, loc, one,
      arith::DivUIOp::create(b, loc,
          arith::AddIOp::create(b, loc, bins, ix(b, 255)), columnBlock));
  auto clearColumns = gpu::LaunchOp::create(b, loc,
      columnGrid, one, one, columnBlock, one, one);
  {
    OpBuilder at = OpBuilder::atBlockEnd(&clearColumns.getBody().front());
    Value column = arith::AddIOp::create(at, loc,
        arith::MulIOp::create(at, loc, clearColumns.getBlockIds().x, columnBlock),
        clearColumns.getThreadIds().x);
    Value inside = arith::CmpIOp::create(at, loc,
        arith::CmpIPredicate::ult, column, bins);
    scf::IfOp::create(at, loc, inside, [&](OpBuilder &in, Location) {
      memref::StoreOp::create(in, loc, word(in, loc, 0),
          v.spatial.counts, ValueRange{column});
      Value first = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::eq,
          column, ix(in, 0));
      scf::IfOp::create(in, loc, first, [&](OpBuilder &clear, Location) {
        memref::StoreOp::create(clear, loc, integer(clear, loc, 0),
            v.counters, ValueRange{ix(clear, 0)});
        scf::YieldOp::create(clear, loc);
      });
      scf::YieldOp::create(in, loc);
    });
    gpu::TerminatorOp::create(at, loc);
  }
  emitPrepareTemplateSearch(b, loc, v);
  emitDeviceScan(b, loc, v.spatial.counts, v.spatial.starts, v.spatial.cursors,
      bins, v.spatial.scanScratch, v.spatial.scanBlocks, columnBlock, one);
  // Read only the initialized maximum-frame scalar, not other control
  // words that a standalone fixture need not have initialized yet.
  auto scalarDevice = MemRefType::get({1}, b.getI64Type(),
      MemRefLayoutAttrInterface(), cast<MemRefType>(v.counters.getType()).getMemorySpace());
  Value frameDevice = memref::ReinterpretCastOp::create(b, loc, scalarDevice,
      v.counters, b.getIndexAttr(0), ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
  Value frameHost = memref::AllocaOp::create(b, loc, MemRefType::get({1}, b.getI64Type()));
  Type token = gpu::AsyncTokenType::get(b.getContext());
  Value ready = gpu::WaitOp::create(b, loc, token, ValueRange()).getAsyncToken();
  Value copied = gpu::MemcpyOp::create(b, loc, token, ValueRange{ready},
      frameHost, frameDevice).getAsyncToken();
  gpu::WaitOp::create(b, loc, Type(), ValueRange{copied});
  Value maxFrame = arith::BitcastOp::create(b, loc, b.getF64Type(),
      load(b, loc, frameHost, ix(b, 0)));
  Value maxReach;
  if (v.reaches) {
    maxReach = vector::ExtractOp::create(b, loc, v.reaches, 0);
    for (int64_t kind = 1; kind != 3; ++kind)
      maxReach = arith::MaximumFOp::create(b, loc, maxReach,
          vector::ExtractOp::create(b, loc, v.reaches, kind));
  } else {
    maxReach = arith::AddFOp::create(b, loc, v.baseReach,
        arith::MulFOp::create(b, loc, real(b, 2), v.extent));
  }
  auto policy = emitTemplateImagePolicy(b, loc, v.cell, maxReach, maxFrame);
  Value path = v.forcedImagePath < 0 ? policy.path : integer(b, loc, v.forcedImagePath);
  TemplateSearchBuffers selected = v;
  selected.imageMargin = policy.margin;
  auto clear = gpu::LaunchOp::create(b, loc, one, one, one, ix(b, 32), one, one);
  {
    OpBuilder at = OpBuilder::atBlockEnd(&clear.getBody().front());
    Value thread = clear.getThreadIds().x;
    Value leader = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::eq, thread, ix(at, 0));
    scf::IfOp::create(at, loc, leader, [&](OpBuilder &in, Location) {
      // The shared scan eagerly reads element zero even for empty input.
      for (Value buffer : {v.build.counts, v.build.workCounts})
        memref::StoreOp::create(in, loc, word(in, loc, 0), buffer, ValueRange{ix(in, 0)});
      scf::YieldOp::create(in, loc);
    });
    Value inside = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, thread, ix(at, 3));
    scf::IfOp::create(at, loc, inside, [&](OpBuilder &in, Location) {
      memref::StoreOp::create(in, loc, integer(in, loc, 0), v.counters, ValueRange{thread});
      if (v.classCells) {
        assert(v.classReaches && v.reaches);
        Value ww = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::eq, thread, ix(in, 0));
        Value ss = arith::CmpIOp::create(in, loc, arith::CmpIPredicate::eq, thread, ix(in, 2));
        Value active = arith::SelectOp::create(in, loc, ww, v.spatial.waterActive,
            arith::SelectOp::create(in, loc, ss, v.spatial.atomActive,
                arith::OrIOp::create(in, loc, v.spatial.waterActive, v.spatial.atomActive)));
        scf::IfOp::create(in, loc, active, [&](OpBuilder &publish, Location) {
          int width = isTriclinic(v.cell) ? 6 : 3;
          for (int64_t axis = 0; axis != 6; ++axis) {
            Value component = axis < width
                ? Value(vector::ExtractOp::create(publish, loc, v.cell, axis)) : real(publish, 0);
            Value row = arith::AddIOp::create(publish, loc,
                arith::MulIOp::create(publish, loc, thread, ix(publish, 2)), ix(publish, axis / 3));
            memref::StoreOp::create(publish, loc, component, v.classCells,
                ValueRange{row, ix(publish, axis % 3)});
          }
          Value radius = vector::ExtractOp::create(publish, loc, v.reaches, OpFoldResult(thread));
          memref::StoreOp::create(publish, loc, radius, v.classReaches,
              ValueRange{ix(publish, 0), thread});
          scf::YieldOp::create(publish, loc);
        });
      }
      scf::YieldOp::create(in, loc);
    });
    gpu::TerminatorOp::create(at, loc);
  }
  SmallVector<Value> metadata{v.handle, path,
      arith::ExtUIOp::create(b, loc, b.getI64Type(), v.spatial.waterActive),
      arith::ExtUIOp::create(b, loc, b.getI64Type(), v.spatial.atomActive)};
  auto metadataType = b.getFunctionType(ValueRange(metadata).getTypes(), TypeRange{b.getI64Type()});
  auto setter = module.lookupSymbol<func::FuncOp>("mdrtTemplatePairsSetImagePath");
  if (!setter) {
    OpBuilder symbols(module.getBodyRegion());
    setter = func::FuncOp::create(symbols, loc, "mdrtTemplatePairsSetImagePath", metadataType);
    setter.setPrivate();
  }
  func::CallOp::create(b, loc, setter, metadata);
  auto call = [&](OpBuilder &at, unsigned mode) {
    auto function = outlineSearch(at, loc, module, selected, mode);
    auto fields = searchFields(selected);
    SmallVector<Value> operands;
    for (Value *field : fields) operands.push_back(*field);
    func::CallOp::create(at, loc, function, operands);
  };
  Value nearest = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, path, integer(b, loc, 0));
  scf::IfOp::create(b, loc, nearest, [&](OpBuilder &at, Location) {
    call(at, 0); scf::YieldOp::create(at, loc);
  }, [&](OpBuilder &at, Location) {
    Value conditional = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::eq,
        path, integer(at, loc, 1));
    scf::IfOp::create(at, loc, conditional, [&](OpBuilder &chosen, Location) {
      call(chosen, 1); scf::YieldOp::create(chosen, loc);
    }, [&](OpBuilder &fallback, Location) {
      call(fallback, 2); scf::YieldOp::create(fallback, loc);
    });
    scf::YieldOp::create(at, loc);
  });
  return path;
}
