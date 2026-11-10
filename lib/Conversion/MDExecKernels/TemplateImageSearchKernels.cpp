// Periodic covering search [AllenTildesley2017], with the image retained
// for the scalar window and the covering bound of D242.
#include "mdir/Conversion/TemplateImageSearchKernels.h"
#include "mdir/Conversion/ImageEnumerationEmitter.h"
#include "mdir/Conversion/TemplateStageKernels.h"
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

namespace {
// Pair-local image enumeration. The one-pass image is evaluated first;
// every conditional prefix recomputes the following triangular coordinates.
static Value pairImages(OpBuilder &b, Location loc, const TemplateSearchBuffers &v,
    bool conditional, const TemplateSearchCertificate &cert, Value reach2, Value reach,
    Value pointI, Value cartesianI, Value normalizedJ, Value cartesianJ, Value frame,
    Value partnerFrame) {
  auto fastType=VectorType::get({3},b.getF32Type());
  auto frameI=emitSpatialReferenceFrame(b,loc,pointI);
  Value raw=arith::SubFOp::create(b,loc,frameI.wrapped,normalizedJ);
  Value d=emitLatticeShift(b,loc,raw,v.cell);
  Value limit=arith::AddFOp::create(b,loc,reach,v.imageMargin);
  auto strategy = conditional ? PairImageStrategy::ConditionalNeighbors : PairImageStrategy::Pass;
  auto results = emitPairImageChoices(b, loc, d, v.cell, limit, strategy,
      word(b, loc, -1), [&](OpBuilder &pair, Value query) -> Value {
        auto both=[&](Value x,Value y)->Value{return arith::AndIOp::create(pair,loc,x,y);};
        auto sub=[&](Value x,Value y)->Value{return arith::SubFOp::create(pair,loc,x,y);};
        auto elt=[&](Value x,unsigned i)->Value{return vector::ExtractOp::create(pair,loc,x,i);};
        Value yes=arith::ConstantIntOp::create(pair,loc,1,1);
        Value absolute=arith::AddFOp::create(pair,loc,frame,query);
        Value shifted=arith::AddFOp::create(pair,loc,normalizedJ,absolute);
        Value fastCell=arith::TruncFOp::create(pair,loc,
            VectorType::get(cast<VectorType>(v.cell.getType()).getShape(),pair.getF32Type()),v.cell);
        Value fastShift=emitLatticeShift(pair,loc,arith::TruncFOp::create(pair,loc,fastType,query),fastCell);
        Value shiftedFast=arith::AddFOp::create(pair,loc,cartesianJ,fastShift);
        Value fd=arith::SubFOp::create(pair,loc,cartesianI,shiftedFast);
        Value fq=vector::ReductionOp::create(pair,loc,vector::CombiningKind::ADD,
            arith::MulFOp::create(pair,loc,fd,fd));
        Value finite=arith::CmpFOp::create(pair,loc,arith::CmpFPredicate::OLT,
            math::AbsFOp::create(pair,loc,fq),createReal(pair,loc,pair.getF32Type(),std::numeric_limits<float>::infinity()));
        Value enabled=both(cert.enabled,finite);
        Value accept=both(both(enabled,cert.acceptEnabled),
            arith::CmpFOp::create(pair,loc,arith::CmpFPredicate::OLE,fq,cert.lower));
        Value reject=both(enabled,arith::CmpFOp::create(pair,loc,arith::CmpFPredicate::OGT,fq,cert.upper));
        Value classified=arith::OrIOp::create(pair,loc,accept,reject);
        auto decision=scf::IfOp::create(pair,loc,classified,
            [&](OpBuilder &quick,Location){scf::YieldOp::create(quick,loc,ValueRange{accept});},
            [&](OpBuilder &exact,Location){
              Value r=arith::SubFOp::create(exact,loc,pointI,shifted);
              Value displacement=emitLatticeShift(exact,loc,r,v.cell);
              Value square=arith::MulFOp::create(exact,loc,displacement,displacement);
              Value norm=vector::ReductionOp::create(exact,loc,vector::CombiningKind::ADD,square);
              Value inside=arith::CmpFOp::create(exact,loc,arith::CmpFPredicate::OLE,norm,reach2);
              scf::YieldOp::create(exact,loc,ValueRange{inside});
            });
        Value kept=decision.getResult(0);
        Value bits=integer(pair,loc,0),fits=yes;
        for(unsigned axis=0;axis<3;++axis){
          Value offset=arith::FPToSIOp::create(pair,loc,pair.getI64Type(),sub(elt(absolute,axis),elt(partnerFrame,axis)));
          fits=both(fits,both(arith::CmpIOp::create(pair,loc,arith::CmpIPredicate::sge,offset,integer(pair,loc,-MDRT_TEMPLATE_IMAGE_OFFSET)),
              arith::CmpIOp::create(pair,loc,arith::CmpIPredicate::slt,offset,integer(pair,loc,MDRT_TEMPLATE_IMAGE_OFFSET))));
          bits=arith::OrIOp::create(pair,loc,bits,arith::ShLIOp::create(pair,loc,
              arith::AddIOp::create(pair,loc,offset,integer(pair,loc,MDRT_TEMPLATE_IMAGE_OFFSET)),integer(pair,loc,axis*MDRT_TEMPLATE_IMAGE_BITS)));
        }
        Value bad=both(kept,arith::XOrIOp::create(pair,loc,fits,yes));
        scf::IfOp::create(pair,loc,bad,[&](OpBuilder &error,Location){
          memref::AtomicRMWOp::create(error,loc,arith::AtomicRMWKind::ori,
              integer(error,loc,TEMPLATE_BUILD_INDEX_OVERFLOW),v.counters,ValueRange{createIndex(error,loc,TEMPLATE_COUNTER_FLAGS)});
          scf::YieldOp::create(error,loc);
        });
        return arith::SelectOp::create(pair,loc,both(kept,fits),
            arith::TruncIOp::create(pair,loc,pair.getI32Type(),bits),word(pair,loc,-1));

      });
  return vector::FromElementsOp::create(b,loc,
      VectorType::get({int64_t(results.size())},b.getI32Type()),results);
}

}
void mdir::kernels::emitSpecializedTemplateSearch(OpBuilder &b, Location loc,
    const TemplateSearchBuffers &v, bool conditional) {
  auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
  auto real = [&](OpBuilder &at, double n) { return createReal(at, loc, at.getF64Type(), n); };
  auto boolean = [&](OpBuilder &at, bool value) -> Value {
    return arith::ConstantOp::create(at, loc, at.getBoolAttr(value));
  };
  Value one=ix(b,1),block=ix(b,64);
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
                Value columnLower[2], upper[2], wideDimensions[2];
                for (unsigned axis = 0; axis != 2; ++axis) {
                  Value n = asWide(search, loc, dimensions[axis]);
                  wideDimensions[axis] = n;
                  Value scale = arith::UIToFPOp::create(search, loc, search.getF64Type(), n);
                  // f32 point/scale/product rounding can move a column key.
                  // One neighboring bin suffices below 2^22 columns; this bound
                  // also covers the largest admitted i32 unit domain.
                  Value halo = arith::AddIOp::create(search, loc, integer(search, loc, 1),
                      arith::DivUIOp::create(search, loc, n, integer(search, loc, 1 << 22)));
                  columnLower[axis] = arith::SubIOp::create(search, loc,
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
                Value envelope=vector::BroadcastOp::create(search,loc,pointType,real(search,8));
                auto certificate=emitTemplateSearchCertificate(search,loc,v.cell,frame,envelope,reach2);
                // WW visits each wrapped covering column exactly once. SW and
                // SS use their direct descriptor partitions without image loops.
                Value nx=asWide(search,loc,v.spatial.columnsX),ny=asWide(search,loc,v.spatial.columnsY);
                Value xCount=searchClass==0?Value(arith::MinUIOp::create(search,loc,nx,arith::SubIOp::create(search,loc,upper[0],columnLower[0]))):integer(search,loc,1);
                Value yCount=searchClass==0?Value(arith::MinUIOp::create(search,loc,ny,arith::SubIOp::create(search,loc,upper[1],columnLower[1]))):integer(search,loc,1);
                auto result=forState(search,loc,integer(search,loc,0),xCount,integer(search,loc,1),frameState,
                    [&](OpBuilder &xloop,Value xi,TemplateStageState xs){
                  Value rawX=arith::AddIOp::create(xloop,loc,columnLower[0],xi);
                  Value x=arith::SubIOp::create(xloop,loc,rawX,
                      arith::MulIOp::create(xloop,loc,floorDivide(xloop,loc,rawX,nx),nx));
                  return forState(xloop,loc,integer(xloop,loc,0),yCount,integer(xloop,loc,1),xs,
                      [&](OpBuilder &yloop,Value yi,TemplateStageState ys){
                    Value rawY=arith::AddIOp::create(yloop,loc,columnLower[1],yi);
                    Value y=arith::SubIOp::create(yloop,loc,rawY,
                        arith::MulIOp::create(yloop,loc,floorDivide(yloop,loc,rawY,ny),ny));
                    Value begin,end;
                    if(searchClass==0){
                      Value odd=arith::CmpIOp::create(yloop,loc,arith::CmpIPredicate::ne,
                          arith::AndIOp::create(yloop,loc,x,integer(yloop,loc,1)),integer(yloop,loc,0));
                      Value serp=arith::SelectOp::create(yloop,loc,odd,
                          arith::SubIOp::create(yloop,loc,arith::SubIOp::create(yloop,loc,ny,integer(yloop,loc,1)),y),y);
                      Value column=asIndex(yloop,loc,arith::AddIOp::create(yloop,loc,
                          arith::MulIOp::create(yloop,loc,x,ny),serp));
                      begin=asIndex(yloop,loc,load(yloop,loc,v.spatial.starts,column));
                      end=arith::AddIOp::create(yloop,loc,begin,asIndex(yloop,loc,load(yloop,loc,v.spatial.counts,column)));
                    } else {
                      begin=searchClass==1?ix(yloop,0):v.spatial.waters;
                      end=searchClass==1?waterRows:Value(arith::AddIOp::create(yloop,loc,v.spatial.waters,atomRows));
                    }
                    if(searchClass!=1)begin=arith::MinUIOp::create(yloop,loc,end,arith::MaxUIOp::create(yloop,loc,begin,firstGroup));
                    return forState(yloop,loc,begin,end,ix(yloop,32),ys,
                        [&](OpBuilder &batch,Value cursor,TemplateStageState state){
                      Value position=arith::AddIOp::create(batch,loc,cursor,laneIndex);
                      Value has=arith::CmpIOp::create(batch,loc,arith::CmpIPredicate::ult,position,end);
                      Value partnerGroup=arith::SelectOp::create(batch,loc,has,position,firstGroup);
                      SmallVector<Value> fractions[2],cartesians[2];
                      for(unsigned side=0;side<2;++side)for(unsigned axis=0;axis<3;++axis){
                        Value r=arith::AddIOp::create(batch,loc,arith::MulIOp::create(batch,loc,partnerGroup,ix(batch,2)),ix(batch,side));
                        fractions[side].push_back(memref::LoadOp::create(batch,loc,v.fractionalBoxes,ValueRange{r,ix(batch,axis)}));
                        cartesians[side].push_back(memref::LoadOp::create(batch,loc,v.groupBoxes,ValueRange{r,ix(batch,axis)}));
                      }
                      SpatialBoundingBox partnerFraction{vector::FromElementsOp::create(batch,loc,pointType,fractions[0]),vector::FromElementsOp::create(batch,loc,pointType,fractions[1])};
                      Value fractionMargin=arith::DivFOp::create(batch,loc,
                          vector::BroadcastOp::create(batch,loc,pointType,v.imageMargin),widths);
                      auto single=emitSpatialSingleImage(batch,loc,box,partnerFraction,radius,fractionMargin);
                      Value fastShift=emitLatticeShift(batch,loc,arith::TruncFOp::create(batch,loc,fastType,single.image),fastCell);
                      SpatialBoundingBox partnerBox{vector::FromElementsOp::create(batch,loc,fastType,cartesians[0]),vector::FromElementsOp::create(batch,loc,fastType,cartesians[1])};
                      Value lowGap=arith::SubFOp::create(batch,loc,cartesianBox.minimum,arith::AddFOp::create(batch,loc,partnerBox.maximum,fastShift));
                      Value highGap=arith::SubFOp::create(batch,loc,arith::AddFOp::create(batch,loc,partnerBox.minimum,fastShift),cartesianBox.maximum);
                      Value gap=arith::MaximumFOp::create(batch,loc,arith::MaximumFOp::create(batch,loc,lowGap,highGap),vector::BroadcastOp::create(batch,loc,fastType,createReal(batch,loc,batch.getF32Type(),0)));
                      Value q=vector::ReductionOp::create(batch,loc,vector::CombiningKind::ADD,arith::MulFOp::create(batch,loc,gap,gap));
                      Value finite=arith::CmpFOp::create(batch,loc,arith::CmpFPredicate::OLT,math::AbsFOp::create(batch,loc,q),createReal(batch,loc,batch.getF32Type(),std::numeric_limits<float>::infinity()));
                      Value reject=arith::AndIOp::create(batch,loc,single.proved,arith::AndIOp::create(batch,loc,certificate.enabled,arith::AndIOp::create(batch,loc,finite,arith::CmpFOp::create(batch,loc,arith::CmpFPredicate::OGT,q,certificate.boxUpper))));
                      has=arith::AndIOp::create(batch,loc,has,arith::XOrIOp::create(batch,loc,reject,boolean(batch,true)));
                      Value survivors=emitWarpBallot(batch,loc,has);
                      auto selected=scf::WhileOp::create(batch,loc,TypeRange{batch.getI32Type(),batch.getI64Type(),batch.getI64Type()},ValueRange{survivors,state.count,state.chunk},
                        [&](OpBuilder &test,Location,ValueRange c){scf::ConditionOp::create(test,loc,arith::CmpIOp::create(test,loc,arith::CmpIPredicate::ne,c[0],word(test,loc,0)),c);},
                        [&](OpBuilder &units,Location,ValueRange carried){
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
                          Value has=arith::AndIOp::create(units,loc,
                              arith::OrIOp::create(units,loc,lower,two),arith::CmpIOp::create(units,loc,arith::CmpIPredicate::ult,local,length));
                          auto partnerFrame=emitSpatialReferenceFrame(units,loc,partnerPoint);
                          // Preserve the exact domain and order of the retained
                          // iterator, including unusually large reference frames.
                          // This only filters candidates; it never changes R.
                          Value allColumns = arith::MulIOp::create(units, loc,
                              v.spatial.columnsX, v.spatial.columnsY);
                          Value logicalColumn = arith::RemUIOp::create(units, loc,
                              asIndex(units, loc, load(units, loc, v.spatial.keys, partnerIndex)), allColumns);
                          Value partnerCX = arith::DivUIOp::create(units, loc,
                              logicalColumn, v.spatial.columnsY);
                          Value partnerSerp = arith::RemUIOp::create(units, loc,
                              logicalColumn, v.spatial.columnsY);
                          Value oddColumn = arith::CmpIOp::create(units, loc, arith::CmpIPredicate::ne,
                              arith::AndIOp::create(units, loc, partnerCX, ix(units, 1)), ix(units, 0));
                          Value partnerCY = arith::SelectOp::create(units, loc, oddColumn,
                              arith::SubIOp::create(units, loc,
                                  arith::SubIOp::create(units, loc, v.spatial.columnsY, ix(units, 1)), partnerSerp), partnerSerp);

                          int leaves=conditional?8:1;
                          auto codesType=VectorType::get({16*leaves},units.getI32Type());
                          Value codes=vector::BroadcastOp::create(units,loc,codesType,word(units,loc,-1));
                          for(unsigned i=0;i<16;++i){
                            Value slot=arith::AddIOp::create(units,loc,cacheBase,ix(units,i));
                            SmallVector<Value> original,fast;
                            for(unsigned axis=0;axis<3;++axis){
                              original.push_back(memref::LoadOp::create(units,loc,pointCache,ValueRange{slot,ix(units,axis)}));
                              fast.push_back(memref::LoadOp::create(units,loc,cartesianCache,ValueRange{slot,ix(units,axis)}));
                            }
                            Value pointI=vector::FromElementsOp::create(units,loc,pointType,original);
                            Value cartesianI=vector::FromElementsOp::create(units,loc,fastType,fast);
                            Value unitI=load(units,loc,idCache,slot);
                            Value validI=arith::CmpIOp::create(units,loc,arith::CmpIPredicate::ne,
                                arith::AndIOp::create(units,loc,members,word(units,loc,int64_t(1)<<i)),word(units,loc,0));
                            Value owns=arith::OrIOp::create(units,loc,sw,arith::CmpIOp::create(units,loc,arith::CmpIPredicate::ult,
                                arith::AddIOp::create(units,loc,first,ix(units,i)),partnerPlace));
                            Value eligible=arith::AndIOp::create(units,loc,has,arith::AndIOp::create(units,loc,validI,owns));
                            auto accepted=scf::IfOp::create(units,loc,eligible,
                              [&](OpBuilder &pair,Location){
                                Value allowed=boolean(pair,true);
                                if(searchClass==2){
                                  Value w=arith::IndexCastOp::create(pair,loc,pair.getI32Type(),v.spatial.waters);
                                  Value a=asIndex(pair,loc,arith::SubIOp::create(pair,loc,unitI,w));
                                  Value other=arith::SubIOp::create(pair,loc,partner,w);
                                  Value start=asIndex(pair,loc,load(pair,loc,v.excludedRows,a));
                                  Value stop=asIndex(pair,loc,load(pair,loc,v.excludedRows,arith::AddIOp::create(pair,loc,a,ix(pair,1))));
                                  Value found=bound(pair,loc,start,stop,other,false,[&](OpBuilder &q,Value j){return load(q,loc,v.excludedNeighbors,j);});
                                  auto excluded=scf::IfOp::create(pair,loc,arith::CmpIOp::create(pair,loc,arith::CmpIPredicate::ult,found,stop),
                                    [&](OpBuilder &q,Location){scf::YieldOp::create(q,loc,ValueRange{arith::CmpIOp::create(q,loc,arith::CmpIPredicate::eq,load(q,loc,v.excludedNeighbors,found),other)});},
                                    [&](OpBuilder &q,Location){scf::YieldOp::create(q,loc,ValueRange{boolean(q,false)});});
                                  allowed=arith::XOrIOp::create(pair,loc,excluded.getResult(0),boolean(pair,true));
                                }
                                auto images=scf::IfOp::create(pair,loc,allowed,
                                  [&](OpBuilder &q,Location){scf::YieldOp::create(q,loc,ValueRange{pairImages(q,loc,v,conditional,certificate,reach2,reach,pointI,cartesianI,partnerFrame.wrapped,partnerCartesian,frame,partnerFrame.lattice)});},
                                  [&](OpBuilder &q,Location){scf::YieldOp::create(q,loc,ValueRange{vector::BroadcastOp::create(q,loc,VectorType::get({leaves},q.getI32Type()),word(q,loc,-1))});});
                                scf::YieldOp::create(pair,loc,images.getResults());
                              },[&](OpBuilder &q,Location){scf::YieldOp::create(q,loc,ValueRange{vector::BroadcastOp::create(q,loc,VectorType::get({leaves},q.getI32Type()),word(q,loc,-1))});});
                            for(int leaf=0;leaf<leaves;++leaf)codes=vector::InsertOp::create(units,loc,
                                vector::ExtractOp::create(units,loc,accepted.getResult(0),leaf),codes,int64_t(i*leaves+leaf));
                          }
                          auto pending=scf::WhileOp::create(units,loc,TypeRange{codesType,units.getI64Type(),units.getI64Type()},ValueRange{codes,carried[1],carried[2]},
                            [&](OpBuilder &test,Location,ValueRange c){
                              Value any=boolean(test,false);
                              for(int j=0;j<16*leaves;++j)any=arith::OrIOp::create(test,loc,any,
                                  arith::CmpIOp::create(test,loc,arith::CmpIPredicate::ne,vector::ExtractOp::create(test,loc,c[0],j),word(test,loc,-1)));
                              Value votes=emitWarpBallot(test,loc,any);
                              scf::ConditionOp::create(test,loc,arith::CmpIOp::create(test,loc,arith::CmpIPredicate::ne,votes,word(test,loc,0)),c);
                            },[&](OpBuilder &emit,Location,ValueRange c){
                              Value chosen=word(emit,loc,-1);
                              for(int j=0;j<16*leaves;++j)chosen=arith::MinUIOp::create(emit,loc,chosen,vector::ExtractOp::create(emit,loc,c[0],j));
                              Value rest=c[0],mask=word(emit,loc,0);
                              for(unsigned i=0;i<16;++i){
                                Value hit=boolean(emit,false);
                                for(int leaf=0;leaf<leaves;++leaf){
                                  int j=i*leaves+leaf;
                                  Value item=vector::ExtractOp::create(emit,loc,c[0],j);
                                  Value same=arith::CmpIOp::create(emit,loc,arith::CmpIPredicate::eq,item,chosen);
                                  hit=arith::OrIOp::create(emit,loc,hit,same);
                                  rest=vector::InsertOp::create(emit,loc,arith::SelectOp::create(emit,loc,same,word(emit,loc,-1),item),rest,int64_t(j));
                                }
                                mask=arith::OrIOp::create(emit,loc,mask,arith::SelectOp::create(emit,loc,hit,word(emit,loc,int64_t(1)<<i),word(emit,loc,0)));
                              }
                              Value keep=arith::CmpIOp::create(emit,loc,arith::CmpIPredicate::ne,chosen,word(emit,loc,-1));
                              mask=arith::SelectOp::create(emit,loc,keep,mask,word(emit,loc,0));
                              Value tag=arith::ShLIOp::create(emit,loc,arith::IndexCastOp::create(emit,loc,emit.getI32Type(),representative),word(emit,loc,24));
                              for(unsigned axis=0;axis<3;++axis){
                                Value packed=arith::AndIOp::create(emit,loc,arith::ShRUIOp::create(emit,loc,chosen,word(emit,loc,axis*MDRT_TEMPLATE_IMAGE_BITS)),word(emit,loc,1023));
                                Value offset=arith::SubIOp::create(emit,loc,
                                    arith::ExtUIOp::create(emit,loc,emit.getI64Type(),packed),integer(emit,loc,MDRT_TEMPLATE_IMAGE_OFFSET));
                                Value f=arith::FPToSIOp::create(emit,loc,emit.getI64Type(),vector::ExtractOp::create(emit,loc,frame,axis));
                                Value jf=arith::FPToSIOp::create(emit,loc,emit.getI64Type(),vector::ExtractOp::create(emit,loc,partnerFrame.lattice,axis));
                                Value query=arith::AddIOp::create(emit,loc,arith::SubIOp::create(emit,loc,offset,f),jf);
                                Value ordinal = query;
                                Value lo = zBegin, hi = zEnd;
                                if (axis != 2) {
                                  Value dimension = axis == 0 ? nx : ny;
                                  Value coordinate = asWide(emit, loc, axis == 0 ? partnerCX : partnerCY);
                                  ordinal = arith::AddIOp::create(emit, loc,
                                      arith::MulIOp::create(emit, loc, query, dimension), coordinate);
                                  lo = columnLower[axis]; hi = upper[axis];
                                }
                                Value inRange = arith::AndIOp::create(emit, loc,
                                    arith::CmpIOp::create(emit, loc, arith::CmpIPredicate::sge, ordinal, lo),
                                    arith::CmpIOp::create(emit, loc, arith::CmpIPredicate::slt, ordinal, hi));
                                keep = arith::AndIOp::create(emit, loc, keep, inRange);
                                Value biased=arith::AndIOp::create(emit,loc,arith::TruncIOp::create(emit,loc,emit.getI32Type(),arith::AddIOp::create(emit,loc,query,integer(emit,loc,128))),word(emit,loc,255));
                                tag=arith::OrIOp::create(emit,loc,tag,arith::ShLIOp::create(emit,loc,biased,word(emit,loc,8*axis)));
                              }
                              Value fields[]={partner,chosen,mask,tag};
                              auto next=emitAppendTemplateStage(emit,loc,stage,{c[1],c[2]},keep,fields);
                              scf::YieldOp::create(emit,loc,ValueRange{rest,next.count,next.chunk});
                            });
                          scf::YieldOp::create(units,loc,ValueRange{remaining,pending.getResult(1),pending.getResult(2)});
                        });
                      return TemplateStageState{selected.getResult(1),selected.getResult(2)};
                    });
                  });
                });
                scf::YieldOp::create(search,loc,ValueRange{result.count,result.chunk});
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
