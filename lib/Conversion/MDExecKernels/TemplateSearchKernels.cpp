// Periodic covering search [AllenTildesley2017], with the image retained
// for the scalar window and the covering bound of D242.
#include "mdir/Conversion/TemplateSearchKernels.h"
#include "mdir/Conversion/TemplateStageKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Conversion/ReductionEmitter.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
// IDX with the same scalar splitting as the common XOR reduction emitter.
static Value shuffle(OpBuilder &b, Location loc, Value value, unsigned lane) {
  Type type = value.getType();
  if (auto vector = dyn_cast<VectorType>(type)) {
    SmallVector<Value> elements;
    for (int64_t i = 0; i != vector.getNumElements(); ++i)
      elements.push_back(shuffle(b, loc, vector::ExtractOp::create(b, loc, value, i), lane));
    return vector::FromElementsOp::create(b, loc, type, elements);
  }
  auto take = [&](Value part) -> Value {
    return gpu::ShuffleOp::create(b, loc, part, word(b, loc, lane), word(b, loc, 32),
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

void mdir::kernels::emitSearchTemplatePairs(OpBuilder &b, Location loc,
                                            const TemplateSearchBuffers &v) {
  auto ix = [&](OpBuilder &at, int64_t n) { return createIndex(at, loc, n); };
  auto real = [&](OpBuilder &at, double n) { return createReal(at, loc, at.getF64Type(), n); };
  auto boolean = [&](OpBuilder &at, bool value) -> Value {
    return arith::ConstantOp::create(at, loc, at.getBoolAttr(value));
  };
  Value one = ix(b, 1), block = ix(b, 256);
  auto clear = gpu::LaunchOp::create(b, loc, one, one, one, ix(b, 32), one, one);
  {
    OpBuilder at = OpBuilder::atBlockEnd(&clear.getBody().front());
    Value thread = clear.getThreadIds().x;
    Value inside = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, thread, ix(at, 3));
    scf::IfOp::create(at, loc, inside, [&](OpBuilder &in, Location) {
      memref::StoreOp::create(in, loc, integer(in, loc, 0), v.counters, ValueRange{thread});
      scf::YieldOp::create(in, loc);
    });
    gpu::TerminatorOp::create(at, loc);
  }
  Value waterRows = arith::DivUIOp::create(b, loc,
      arith::AddIOp::create(b, loc, v.spatial.waters, ix(b, 15)), ix(b, 16));
  Value atomRows = arith::DivUIOp::create(b, loc,
      arith::AddIOp::create(b, loc, v.spatial.atoms, ix(b, 15)), ix(b, 16));
  Value ssBegin = arith::AddIOp::create(b, loc, waterRows, atomRows);
  Value threads = arith::MulIOp::create(b, loc, v.build.rows, ix(b, 32));
  Value grid = arith::MaxUIOp::create(b, loc, one,
      arith::DivUIOp::create(b, loc,
          arith::AddIOp::create(b, loc, threads, ix(b, 255)), block));
  auto launch = gpu::LaunchOp::create(b, loc, grid, one, one, block, one, one);
  OpBuilder k = OpBuilder::atBlockEnd(&launch.getBody().front());
  Value thread = arith::AddIOp::create(k, loc,
      arith::MulIOp::create(k, loc, launch.getBlockIds().x, block), launch.getThreadIds().x);
  Value row = arith::DivUIOp::create(k, loc, thread, ix(k, 32));
  Value validRow = arith::CmpIOp::create(k, loc, arith::CmpIPredicate::ult, row, v.build.rows);
  scf::IfOp::create(k, loc, validRow, [&](OpBuilder &at, Location) {
    Value lane = arith::IndexCastOp::create(at, loc, at.getI32Type(),
        arith::RemUIOp::create(at, loc, thread, ix(at, 32)));
    Value laneIndex = asIndex(at, loc, lane);
    Value rowWord = arith::IndexCastOp::create(at, loc, at.getI32Type(), row);
    TemplateStageBuffers stage{v.build.stageUnits, v.build.stageEntries, v.counters,
        v.build.counts, asWide(at, loc, v.build.unitCapacity), rowWord, lane};
    TemplateStageState initial{integer(at, loc, 0), integer(at, loc, -1)};
    Value ww = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, row, waterRows);
    Value ss = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::uge, row, ssBegin);
    Value sw = arith::AndIOp::create(at, loc,
        arith::CmpIOp::create(at, loc, arith::CmpIPredicate::uge, row, waterRows),
        arith::CmpIOp::create(at, loc, arith::CmpIPredicate::ult, row, ssBegin));
    Value active = arith::SelectOp::create(at, loc, ww, v.spatial.waterActive,
        arith::SelectOp::create(at, loc, sw,
            arith::OrIOp::create(at, loc, v.spatial.waterActive, v.spatial.atomActive), v.spatial.atomActive));
    auto chosen = scf::IfOp::create(at, loc, active,
        [&](OpBuilder &search, Location) {
          Value classRow = arith::SubIOp::create(search, loc, row,
              arith::SelectOp::create(search, loc, sw, waterRows, ssBegin));
          Value first = arith::MulIOp::create(search, loc, ix(search, 16),
              arith::SelectOp::create(search, loc, ww, row,
                  arith::AddIOp::create(search, loc, waterRows, classRow)));
          Value firstPlace = arith::AddIOp::create(search, loc, first,
              asIndex(search, loc, arith::AndIOp::create(search, loc, lane, word(search, loc, 15))));
          Value firstUnit = load(search, loc, v.spatial.order, firstPlace);
          Value firstValid = arith::CmpIOp::create(search, loc, arith::CmpIPredicate::sge,
                                                    firstUnit, word(search, loc, 0));
          Value safeUnit = arith::SelectOp::create(search, loc, firstValid,
                                                    firstUnit, word(search, loc, 0));
          Value firstPoint = loadElement(search, loc, v.spatial.points, asIndex(search, loc, safeUnit));
          auto pointType = cast<VectorType>(firstPoint.getType());
          Value infinity = vector::BroadcastOp::create(search, loc, pointType,
              real(search, std::numeric_limits<double>::infinity()));
          Value negativeInfinity = arith::NegFOp::create(search, loc, infinity);
          Value minInput = arith::SelectOp::create(search, loc, firstValid, firstPoint, infinity);
          Value maxInput = arith::SelectOp::create(search, loc, firstValid, firstPoint, negativeInfinity);
          Value minimum = arith::NegFOp::create(search, loc,
              emitWarpReduce(search, loc, arith::NegFOp::create(search, loc, minInput), Combiner::Max));
          Value maximum = emitWarpReduce(search, loc, maxInput, Combiner::Max);
          Value multiplier = arith::SelectOp::create(search, loc, ww, real(search, 2),
              arith::SelectOp::create(search, loc, sw, real(search, 1), real(search, 0)));
          Value reach = arith::AddFOp::create(search, loc, v.baseReach,
              arith::MulFOp::create(search, loc, multiplier, v.extent));
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
          Value allColumns = arith::MulIOp::create(search, loc, v.spatial.columnsX, v.spatial.columnsY);
          auto result = forState(search, loc, lower[0], upper[0], integer(search, loc, 1), initial,
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
                      Value begin = asIndex(yloop, loc, load(yloop, loc, v.spatial.starts, column));
                      Value end = asIndex(yloop, loc, load(yloop, loc, v.spatial.starts,
                          arith::AddIOp::create(yloop, loc, column, ix(yloop, 1))));
                      return forState(yloop, loc, zBegin, zEnd, integer(yloop, loc, 1), stateY,
                          [&](OpBuilder &zloop, Value sz, TemplateStageState stateZ) {
                            Value zReal = arith::SIToFPOp::create(zloop, loc, zloop.getF64Type(), sz);
                            auto zKey = [&](Value limit, bool upper) -> Value {
                              Value scaled = arith::MulFOp::create(zloop, loc,
                                  arith::SubFOp::create(zloop, loc, limit, zReal), real(zloop, 1 << 21));
                              Value rounded = upper ? Value(math::CeilOp::create(zloop, loc, scaled))
                                                    : Value(math::FloorOp::create(zloop, loc, scaled));
                              Value key = arith::FPToSIOp::create(zloop, loc, zloop.getI64Type(), rounded);
                              key = arith::AddIOp::create(zloop, loc, key, integer(zloop, loc, upper ? 2 : -2));
                              key = arith::MaxSIOp::create(zloop, loc, key, integer(zloop, loc, 0));
                              key = arith::MinSIOp::create(zloop, loc, key, integer(zloop, loc, (1 << 21) - 1));
                              return arith::TruncIOp::create(zloop, loc, zloop.getI32Type(), key);
                            };
                            auto keyAt = [&](OpBuilder &in, Value position) -> Value {
                              Value unit = load(in, loc, v.spatial.items, position);
                              return load(in, loc, v.spatial.zKeys, asIndex(in, loc, unit));
                            };
                            Value firstCandidate = bound(zloop, loc, begin, end,
                                zKey(component(zloop, low, 2), false), false, keyAt);
                            Value lastCandidate = bound(zloop, loc, firstCandidate, end,
                                zKey(component(zloop, high, 2), true), true, keyAt);
                            Value imageComponents[] = {arith::SIToFPOp::create(zloop, loc, zloop.getF64Type(), sx),
                                arith::SIToFPOp::create(zloop, loc, zloop.getF64Type(), sy), zReal};
                            Value image = vector::FromElementsOp::create(zloop, loc, pointType, imageComponents);
                            Value imageIntegers[] = {sx, sy, sz};
                            return forState(zloop, loc, firstCandidate, lastCandidate, ix(zloop, 32), stateZ,
                                [&](OpBuilder &batch, Value cursor, TemplateStageState state) {
                                  Value position = arith::AddIOp::create(batch, loc, cursor, laneIndex);
                                  Value has = arith::CmpIOp::create(batch, loc, arith::CmpIPredicate::ult,
                                                                     position, lastCandidate);
                                  Value safe = arith::SelectOp::create(batch, loc, has, position, cursor);
                                  Value partner = load(batch, loc, v.spatial.items, safe);
                                  Value partnerIndex = asIndex(batch, loc, partner);
                                  Value partnerPlace = asIndex(batch, loc, load(batch, loc, v.spatial.places, partnerIndex));
                                  Value partnerPoint = loadElement(batch, loc, v.spatial.points, partnerIndex);
                                  SmallVector<Value, 3> wrappedPartner;
                                  Value code = integer(batch, loc, 0);
                                  Value imageFits = boolean(batch, true);
                                  for (unsigned axis = 0; axis != 3; ++axis) {
                                    Value coordinate = component(batch, partnerPoint, axis);
                                    Value frame = math::FloorOp::create(batch, loc, coordinate);
                                    wrappedPartner.push_back(arith::SubFOp::create(batch, loc, coordinate, frame));
                                    Value offset = arith::SubIOp::create(batch, loc, imageIntegers[axis],
                                        arith::FPToSIOp::create(batch, loc, batch.getI64Type(), frame));
                                    Value lower = arith::CmpIOp::create(batch, loc, arith::CmpIPredicate::sge,
                                        offset, integer(batch, loc, -MDRT_TEMPLATE_IMAGE_OFFSET));
                                    Value upper = arith::CmpIOp::create(batch, loc, arith::CmpIPredicate::slt,
                                        offset, integer(batch, loc, MDRT_TEMPLATE_IMAGE_OFFSET));
                                    imageFits = arith::AndIOp::create(batch, loc, imageFits,
                                        arith::AndIOp::create(batch, loc, lower, upper));
                                    code = arith::OrIOp::create(batch, loc, code,
                                        arith::ShLIOp::create(batch, loc,
                                            arith::AddIOp::create(batch, loc, offset,
                                                integer(batch, loc, MDRT_TEMPLATE_IMAGE_OFFSET)),
                                            integer(batch, loc, MDRT_TEMPLATE_IMAGE_BITS * axis)));
                                  }
                                  Value imageCode = arith::TruncIOp::create(batch, loc, batch.getI32Type(), code);
                                  Value normalizedPartner = vector::FromElementsOp::create(batch, loc, pointType, wrappedPartner);
                                  Value shifted = arith::AddFOp::create(batch, loc, normalizedPartner, image);
                                  Value mask = word(batch, loc, 0);
                                  for (unsigned i = 0; i != 16; ++i) {
                                    Value pointI = shuffle(batch, loc, firstPoint, i);
                                    Value unitI = shuffle(batch, loc, safeUnit, i);
                                    Value validI = shuffle(batch, loc,
                                        arith::ExtUIOp::create(batch, loc, batch.getI32Type(), firstValid), i);
                                    validI = arith::CmpIOp::create(batch, loc, arith::CmpIPredicate::ne,
                                                                     validI, word(batch, loc, 0));
                                    Value owns = arith::OrIOp::create(batch, loc, sw,
                                        arith::CmpIOp::create(batch, loc, arith::CmpIPredicate::ult,
                                            arith::AddIOp::create(batch, loc, first, ix(batch, i)), partnerPlace));
                                    Value eligible = arith::AndIOp::create(batch, loc, has,
                                        arith::AndIOp::create(batch, loc, validI, owns));
                                    auto accepted = scf::IfOp::create(batch, loc, eligible,
                                        [&](OpBuilder &pair, Location) {
                                          Value raw = arith::SubFOp::create(pair, loc, pointI, shifted);
                                          Value displacement = emitLatticeShift(pair, loc, raw, v.cell);
                                          Value square = arith::MulFOp::create(pair, loc, displacement, displacement);
                                          Value distance = vector::ReductionOp::create(pair, loc,
                                              vector::CombiningKind::ADD, square);
                                          Value inside = arith::CmpFOp::create(pair, loc,
                                              arith::CmpFPredicate::OLE, distance, reach2);
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
                                    mask = arith::OrIOp::create(batch, loc, mask,
                                        arith::SelectOp::create(batch, loc, accepted.getResult(0),
                                            word(batch, loc, int64_t(1) << i), word(batch, loc, 0)));
                                  }
                                  Value keep = arith::CmpIOp::create(batch, loc, arith::CmpIPredicate::ne,
                                                                       mask, word(batch, loc, 0));
                                  Value badImage = arith::AndIOp::create(batch, loc, keep,
                                      arith::XOrIOp::create(batch, loc, imageFits, boolean(batch, true)));
                                  scf::IfOp::create(batch, loc, badImage, [&](OpBuilder &bad, Location) {
                                    memref::AtomicRMWOp::create(bad, loc, arith::AtomicRMWKind::ori,
                                        integer(bad, loc, TEMPLATE_BUILD_INDEX_OVERFLOW), v.counters,
                                        ValueRange{ix(bad, TEMPLATE_COUNTER_FLAGS)});
                                    scf::YieldOp::create(bad, loc);
                                  });
                                  keep = arith::AndIOp::create(batch, loc, keep, imageFits);
                                  Value fields[] = {partner, imageCode, mask, word(batch, loc, 0)};
                                  return emitAppendTemplateStage(batch, loc, stage, state, keep, fields);
                                });
                          });
                    });
              });
          scf::YieldOp::create(search, loc, ValueRange{result.count, result.chunk});
        }, [&](OpBuilder &copy, Location) {
          Value begin = asIndex(copy, loc, record(copy, loc, v.build.outerRows, row, 2));
          Value count = asIndex(copy, loc, record(copy, loc, v.build.outerRows, row, 3));
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
}
