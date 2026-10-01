// What the lowerings of md_exec have in common.

#include "mdir/Conversion/MDExecKernels.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mdir/Dialect/MD/MDOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

Value kernels::createIndex(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

Value kernels::createZero(OpBuilder &builder, Location loc, Type type) {
  return arith::ConstantOp::create(
      builder, loc, type, cast<TypedAttr>(builder.getZeroAttr(type)));
}

Value kernels::createReal(OpBuilder &builder, Location loc, Type real,
                          double value) {
  return arith::ConstantOp::create(builder, loc, real,
                                   builder.getFloatAttr(real, value));
}

Value kernels::createInverse(OpBuilder &builder, Location loc, Value box) {
  auto type = cast<VectorType>(box.getType());
  FloatAttr one = builder.getFloatAttr(type.getElementType(), 1.0);
  Value ones = arith::ConstantOp::create(
      builder, loc, type, DenseElementsAttr::get(type, one.getValue()));
  return arith::DivFOp::create(builder, loc, ones, box);
}

Value kernels::convertReal(OpBuilder &builder, Location loc, Value value,
                           Type real) {
  Type source = value.getType();
  Type target = real;
  if (auto vector = dyn_cast<VectorType>(source))
    target = VectorType::get(vector.getShape(), real);
  if (source == target)
    return value;
  if (getElementTypeOrSelf(source).getIntOrFloatBitWidth() <
      real.getIntOrFloatBitWidth())
    return arith::ExtFOp::create(builder, loc, target, value);
  return arith::TruncFOp::create(builder, loc, target, value);
}

Value kernels::loadElement(OpBuilder &builder, Location loc, Value buffer,
                           Value particle) {
  auto type = cast<MemRefType>(buffer.getType());
  if (type.getRank() == 1)
    return memref::LoadOp::create(builder, loc, buffer, ValueRange{particle});

  int64_t components = type.getDimSize(1);
  SmallVector<Value, 3> elements;
  for (int64_t c = 0; c < components; ++c)
    elements.push_back(memref::LoadOp::create(
        builder, loc, buffer,
        ValueRange{particle, createIndex(builder, loc, c)}));
  return vector::FromElementsOp::create(
      builder, loc, VectorType::get({components}, type.getElementType()),
      elements);
}

void kernels::storeElement(OpBuilder &builder, Location loc, Value value,
                           Value buffer, Value particle) {
  auto type = cast<MemRefType>(buffer.getType());
  if (type.getRank() == 1) {
    memref::StoreOp::create(builder, loc, value, buffer,
                            ValueRange{particle});
    return;
  }
  for (int64_t c = 0, e = type.getDimSize(1); c < e; ++c) {
    Value element = vector::ExtractOp::create(builder, loc, value, c);
    memref::StoreOp::create(builder, loc, element, buffer,
                            ValueRange{particle, createIndex(builder, loc, c)});
  }
}

//===----------------------------------------------------------------------===//
// Loops over particles
//===----------------------------------------------------------------------===//

SmallVector<Value> kernels::emitParticleKernel(OpBuilder &builder,
                                               md_exec::ParticleForOp op,
                                               Value particle,
                                               IRMapping &local) {
  Location loc = op.getLoc();
  Block &kernel = op.getKernel().front();

  for (auto [index, buffer] : llvm::enumerate(op.getIns()))
    local.map(kernel.getArgument(index),
              loadElement(builder, loc, buffer, particle));
  for (Operation &nested : kernel.without_terminator())
    builder.clone(nested, local);

  Operation *yield = kernel.getTerminator();
  unsigned numOuts = op.getOuts().size();
  for (unsigned i = 0; i != numOuts; ++i)
    storeElement(builder, loc, local.lookupOrDefault(yield->getOperand(i)),
                 op.getOuts()[i], particle);

  SmallVector<Value> contributions;
  for (unsigned i = numOuts, e = yield->getNumOperands(); i != e; ++i)
    contributions.push_back(local.lookupOrDefault(yield->getOperand(i)));
  return contributions;
}

//===----------------------------------------------------------------------===//
// Loops over pairs
//===----------------------------------------------------------------------===//

SmallVector<Value> kernels::emitPairKernel(OpBuilder &builder,
                                           md_exec::PairForOp op,
                                           Value counts, Value index,
                                           Value box, Value inverse,
                                           Value central,
                                           IRMapping &local,
                                           const RowLanes *lanes,
                                           SmallVectorImpl<Value> *outTotals,
                                           const PairLayout *layout) {
  Location loc = op.getLoc();
  Value positions = layout ? layout->positions : op.getPositions();
  SmallVector<Value> insBuffers(op.getIns().begin(), op.getIns().end());
  if (layout)
    insBuffers.assign(layout->ins.begin(), layout->ins.end());
  Block &kernel = op.getKernel().front();
  Operation *yield = kernel.getTerminator();
  unsigned numIns = op.getIns().size();
  unsigned numOuts = op.getOuts().size();
  unsigned numYields = yield->getNumOperands();

  // The difference of the positions is taken in their type and then
  // converted to the type that the kernel computes in: the subtraction is
  // the step that loses precision. The minimum image and the squared
  // length follow in the type of the kernel; a device computes in double
  // precision at a small fraction of the rate of single.
  Type computed = kernel.getArgument(0).getType();
  double cutoff = op.getCutoff().convertToDouble();
  Value cutoff2 = createReal(builder, loc, computed, cutoff * cutoff);
  Value boxComputed = convertReal(builder, loc, box, computed);
  Value inverseComputed = convertReal(builder, loc, inverse, computed);
  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);

  Value centralPosition = loadElement(builder, loc, positions, central);
  SmallVector<Value> centralValues;
  for (Value buffer : insBuffers)
    centralValues.push_back(loadElement(builder, loc, buffer, central));

  Value count =
      memref::LoadOp::create(builder, loc, counts, ValueRange{central});
  Value end =
      arith::IndexCastOp::create(builder, loc, builder.getIndexType(), count);
  Value first = zero, step = one;
  if (lanes) {
    end = arith::SelectOp::create(builder, loc, lanes->valid, end, zero);
    first = lanes->lane;
    step = createIndex(builder, loc, lanes->lanes);
  }

  // The contributions of the neighbors of one particle are summed in the
  // order of its list, or, with lanes, in that order within each lane and
  // then over the lanes.
  // A contribution to a global sum that the kernel widens, as the mixed
  // mode widens the energy and the virial to f64, is summed over the row
  // in the type the kernel computed it in and widened once for the row:
  // a device adds in f64 at a small fraction of the rate of f32.
  auto getNarrow = [&](unsigned i) -> Value {
    if (i < numOuts)
      return Value();
    auto widen = yield->getOperand(i).getDefiningOp<arith::ExtFOp>();
    if (!widen || widen->getBlock() != &kernel)
      return Value();
    return widen.getIn();
  };
  SmallVector<Value> sums;
  for (unsigned i = 0; i != numYields; ++i) {
    Value narrow = getNarrow(i);
    Type type = narrow ? narrow.getType() : yield->getOperand(i).getType();
    sums.push_back(createZero(builder, loc, type));
  }

  auto inner = scf::ForOp::create(
      builder, loc, first, end, step, sums,
      [&](OpBuilder &pair, Location, Value entry, ValueRange partial) {
        Value narrow = memref::LoadOp::create(pair, loc, index,
                                              ValueRange{central, entry});
        Value other = arith::IndexCastOp::create(pair, loc,
                                                 pair.getIndexType(), narrow);

        // The minimum-image displacement [AllenTildesley2017] and its
        // squared length.
        Value otherPosition = loadElement(pair, loc, positions, other);
        Value raw = convertReal(
            pair, loc,
            arith::SubFOp::create(pair, loc, centralPosition, otherPosition),
            computed);
        Value images =
            arith::MulFOp::create(pair, loc, raw, inverseComputed);
        Value nearest = math::RoundEvenOp::create(pair, loc, images);
        Value shift = arith::MulFOp::create(pair, loc, nearest, boxComputed);
        Value d = arith::SubFOp::create(pair, loc, raw, shift);
        Value squares = arith::MulFOp::create(pair, loc, d, d);
        Value r2 = vector::ReductionOp::create(
            pair, loc, vector::CombiningKind::ADD, squares);

        IRMapping inside = local;
        inside.map(kernel.getArgument(0), r2);
        inside.map(kernel.getArgument(1), d);
        for (unsigned i = 0; i != numIns; ++i) {
          inside.map(kernel.getArgument(2 + 2 * i), centralValues[i]);
          inside.map(kernel.getArgument(3 + 2 * i),
                     loadElement(pair, loc, insBuffers[i], other));
        }
        for (Operation &nested : kernel.without_terminator())
          pair.clone(nested, inside);

        // A pair beyond the cutoff contributes nothing, and neither does
        // an entry that stands for the particle itself, which is how a
        // device marks an excluded pair (the build of the neighbor matrix,
        // lib/Runtime/Templates/NeighborsMatrixGPU.mlir).
        Value within = arith::CmpFOp::create(
            pair, loc, arith::CmpFPredicate::OLT, r2, cutoff2);
        Value distinct = arith::CmpIOp::create(
            pair, loc, arith::CmpIPredicate::ne, other, central);
        within = arith::AndIOp::create(pair, loc, within, distinct);
        SmallVector<Value> updated;
        for (unsigned i = 0; i != numYields; ++i) {
          Value narrow = getNarrow(i);
          Value contribution =
              inside.lookupOrDefault(narrow ? narrow : yield->getOperand(i));
          Value nothing = createZero(pair, loc, contribution.getType());
          Value masked = arith::SelectOp::create(pair, loc, within,
                                                 contribution, nothing);
          updated.push_back(
              arith::AddFOp::create(pair, loc, partial[i], masked));
        }
        scf::YieldOp::create(pair, loc, updated);
      });

  SmallVector<Value> totals(inner.getResults());
  for (unsigned i = 0; i != numYields; ++i)
    if (getNarrow(i))
      totals[i] = arith::ExtFOp::create(
          builder, loc, yield->getOperand(i).getType(), totals[i]);
  if (lanes)
    for (Value &total : totals)
      total = lanes->combine(builder, loc, total);

  auto emitStores = [&](OpBuilder &writer) {
    // The destinations are in the order of the particles.
    Value particle = central;
    if (layout)
      particle = arith::IndexCastOp::create(
          writer, loc, writer.getIndexType(),
          memref::LoadOp::create(writer, loc, layout->order,
                                 ValueRange{central}));
    for (unsigned i = 0; i != numOuts; ++i) {
      Value destination = op.getOuts()[i];
      Value total = totals[i];
      if (!op.overwrites(i))
        total = arith::AddFOp::create(
            writer, loc, loadElement(writer, loc, destination, particle),
            total);
      storeElement(writer, loc, total, destination, particle);
    }
  };
  if (outTotals) {
    outTotals->assign(totals.begin(), totals.begin() + numOuts);
  } else if (lanes && numOuts != 0) {
    Value leader = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         lanes->lane, zero);
    Value writes = arith::AndIOp::create(builder, loc, leader, lanes->valid);
    scf::IfOp::create(builder, loc, writes, [&](OpBuilder &then, Location) {
      emitStores(then);
      scf::YieldOp::create(then, loc);
    });
  } else {
    emitStores(builder);
  }

  SmallVector<Value> contributions;
  for (unsigned i = numOuts; i != numYields; ++i) {
    Value total = totals[i];
    if (auto weights = op.getWeights()) {
      double weight = (*weights)[i - numOuts];
      if (weight != 1.0) {
        Type type = total.getType();
        FloatAttr scalar =
            builder.getFloatAttr(getElementTypeOrSelf(type), weight);
        Value factor;
        if (auto vector = dyn_cast<VectorType>(type))
          factor = arith::ConstantOp::create(
              builder, loc, type,
              DenseElementsAttr::get(vector, scalar.getValue()));
        else
          factor = arith::ConstantOp::create(builder, loc, type, scalar);
        total = arith::MulFOp::create(builder, loc, factor, total);
      }
    }
    contributions.push_back(total);
  }
  return contributions;
}

/// The components of `value` (a number or a vector) as numbers.
static SmallVector<Value> getComponents(OpBuilder &b, Location loc,
                                        Value value) {
  auto vector = dyn_cast<VectorType>(value.getType());
  if (!vector)
    return {value};
  SmallVector<Value> parts;
  for (int64_t c = 0, e = vector.getNumElements(); c != e; ++c)
    parts.push_back(vector::ExtractOp::create(b, loc, value, c));
  return parts;
}

/// A value of `type` (a number or a vector) from its components.
static Value fromComponents(OpBuilder &b, Location loc, Type type,
                            ArrayRef<Value> parts) {
  if (auto vector = dyn_cast<VectorType>(type))
    return vector::FromElementsOp::create(b, loc, vector, parts);
  return parts.front();
}

static void emitAtomicAdd(OpBuilder &b, Location loc, Value value,
                          Value buffer, Value row);

SmallVector<Value> kernels::emitGroupPairKernel(
    OpBuilder &builder, md_exec::PairForOp op, const GroupLists &lists,
    const PairLayout &layout, Value box, Value inverse, Value unit,
    Value lane, IRMapping &local) {
  Location loc = op.getLoc();
  Block &kernel = op.getKernel().front();
  Operation *yield = kernel.getTerminator();
  unsigned numIns = op.getIns().size();
  unsigned numOuts = op.getOuts().size();
  unsigned numYields = yield->getNumOperands();
  Type computed = kernel.getArgument(0).getType();
  double cutoff = op.getCutoff().convertToDouble();
  Value cutoff2 = createReal(builder, loc, computed, cutoff * cutoff);
  (void)inverse;
  Type i32 = builder.getI32Type();
  auto constant32 = [&](OpBuilder &b, int64_t v) -> Value {
    return arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(v));
  };
  auto toIndex = [&](OpBuilder &b, Value v) -> Value {
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), v);
  };

  // The unit: its group, and its entries.
  Value group = memref::LoadOp::create(builder, loc, lists.units,
                                       ValueRange{unit});
  Value k = memref::LoadOp::create(builder, loc, lists.ordinals,
                                   ValueRange{unit});
  Value groupIndex = toIndex(builder, group);
  Value count = memref::LoadOp::create(builder, loc, lists.counts,
                                       ValueRange{groupIndex});
  Value begin = arith::MulIOp::create(builder, loc, k, constant32(builder, 64));
  Value end = arith::MinSIOp::create(
      builder, loc, count,
      arith::AddIOp::create(builder, loc, begin, constant32(builder, 64)));

  // The particle of this lane: place 16 g + u.
  Value lane32 = arith::IndexCastOp::create(builder, loc, i32, lane);
  Value u = arith::AndIOp::create(builder, loc, lane32, constant32(builder, 15));
  Value half = arith::AndIOp::create(builder, loc, lane32,
                                     constant32(builder, 16));
  Value mine32 = arith::AddIOp::create(
      builder, loc,
      arith::MulIOp::create(builder, loc, group, constant32(builder, 16)), u);
  Value mine = toIndex(builder, mine32);
  Value myPosition = loadElement(builder, loc, layout.positions, mine);
  SmallVector<Value> myValues;
  for (Value buffer : layout.ins)
    myValues.push_back(loadElement(builder, loc, buffer, mine));
  // The lane that the values come from at each step: the one before in the
  // half-warp.
  Value source = arith::OrIOp::create(
      builder, loc, half,
      arith::AndIOp::create(
          builder, loc,
          arith::AddIOp::create(builder, loc, u, constant32(builder, 15)),
          constant32(builder, 15)));
  Value width = constant32(builder, 32);
  auto shuffle = [&](OpBuilder &b, Value value) -> Value {
    SmallVector<Value> parts = getComponents(b, loc, value);
    for (Value &part : parts)
      part = gpu::ShuffleOp::create(b, loc, part, source, width,
                                    gpu::ShuffleMode::IDX)
                 .getShuffleResult();
    return fromComponents(b, loc, value.getType(), parts);
  };

  // The values of the kernel go to the particle of the group in the type
  // the kernel computes them in; the sums likewise, widened at the end.
  auto getNarrow = [&](unsigned i) -> Value {
    if (i < numOuts)
      return Value();
    auto widen = yield->getOperand(i).getDefiningOp<arith::ExtFOp>();
    if (!widen || widen->getBlock() != &kernel)
      return Value();
    return widen.getIn();
  };
  auto yieldType = [&](unsigned i) {
    Value narrow = getNarrow(i);
    return narrow ? narrow.getType() : yield->getOperand(i).getType();
  };
  SmallVector<Value> initial;
  for (unsigned i = 0; i != numYields; ++i)
    initial.push_back(createZero(builder, loc, yieldType(i)));

  // The entries, 32 at a time: at most two rounds for a unit of 64. The
  // unit is block `unit` of the entries.
  Value blockBase = arith::MulIOp::create(builder, loc, unit,
                                          createIndex(builder, loc, 64));
  Value first = toIndex(builder, begin);
  Value last = toIndex(builder, end);
  Value step32 = createIndex(builder, loc, 32);
  auto rounds = scf::ForOp::create(
      builder, loc, first, last, step32, initial,
      [&](OpBuilder &b, Location, Value e0, ValueRange acc) {
        Value e = arith::AddIOp::create(b, loc, e0, lane);
        Value has = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                          e, last);
        Value safe = arith::SelectOp::create(b, loc, has, e, first);
        // Entry e of the list is entry e - 64 k of the block.
        Value at = arith::AddIOp::create(
            b, loc, arith::SubIOp::create(b, loc, safe, first), blockBase);
        Value place32 =
            memref::LoadOp::create(b, loc, lists.entries, ValueRange{at});
        Value mask = arith::SelectOp::create(
            b, loc, has,
            memref::LoadOp::create(b, loc, lists.masks, ValueRange{at}),
            constant32(b, 0));
        Value place = toIndex(b, place32);
        // The entry in the frame of the group: its position, in the frame
        // of its own group, moved by the whole cells in bits 16 to 24 of
        // its mask, (e + 2) in three bits an axis (D95).
        Value position = loadElement(b, loc, layout.positions, place);
        {
          auto positionType = cast<VectorType>(position.getType());
          Type element = positionType.getElementType();
          SmallVector<Value> cells;
          for (int64_t axis = 0; axis != 3; ++axis) {
            Value bits = arith::AndIOp::create(
                b, loc,
                arith::ShRUIOp::create(b, loc, mask,
                                       constant32(b, 16 + 3 * axis)),
                constant32(b, 7));
            cells.push_back(arith::SIToFPOp::create(
                b, loc, element,
                arith::SubIOp::create(b, loc, bits, constant32(b, 2))));
          }
          Value shift = arith::MulFOp::create(
              b, loc, vector::FromElementsOp::create(b, loc, positionType, cells),
              convertReal(b, loc, box, element));
          position = arith::AddFOp::create(b, loc, position, shift);
        }
        SmallVector<Value> values;
        for (Value buffer : layout.ins)
          values.push_back(loadElement(b, loc, buffer, place));

        // The bits of the pairs of this lane's particle with the 16 entries
        // of its half-warp, as they come by: a ballot for each particle of
        // the group gives the bits of every entry, and each lane keeps
        // those of its own, rotated so that step k tests bit 15 - k and
        // each step shifts by one. The mask then does not turn with the
        // entry: a shuffle a pair less, on the pipe of loads and shuffles
        // that bounds the loop (D97).
        Value own16 = constant32(b, 0);
        for (int64_t p = 0; p != 16; ++p) {
          Value set = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::ne,
              arith::AndIOp::create(b, loc, mask, constant32(b, 1 << p)),
              constant32(b, 0));
          Value word = gpu::BallotOp::create(b, loc, b.getI32Type(), set);
          own16 = arith::SelectOp::create(
              b, loc,
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, u,
                                    constant32(b, p)),
              word, own16);
        }
        Value low16 = constant32(b, 0xFFFF);
        Value halfBits = arith::AndIOp::create(
            b, loc, arith::ShRUIOp::create(b, loc, own16, half), low16);
        Value turn = arith::SubIOp::create(b, loc, constant32(b, 15), u);
        Value pairBits = arith::AndIOp::create(
            b, loc,
            arith::OrIOp::create(
                b, loc, arith::ShLIOp::create(b, loc, halfBits, turn),
                arith::ShRUIOp::create(
                    b, loc, halfBits,
                    arith::SubIOp::create(b, loc, constant32(b, 16), turn))),
            low16);

        // 16 steps, the entry's values and what it receives turning.
        SmallVector<Value> carried = {position, pairBits};
        carried.append(values.begin(), values.end());
        for (unsigned i = 0; i != numOuts; ++i)
          carried.push_back(createZero(b, loc, yieldType(i)));
        carried.append(acc.begin(), acc.end());
        // The counter in i32: an index is 64 bits on the device, two
        // additions and two comparisons a step.
        auto steps = scf::ForOp::create(
            b, loc, constant32(b, 0), constant32(b, 16), constant32(b, 1),
            carried,
            [&](OpBuilder &s, Location, Value, ValueRange state) {
              Value otherPosition = state[0];
              Value bits = state[1];
              ValueRange otherValues = state.slice(2, numIns);
              ValueRange received = state.slice(2 + numIns, numOuts);
              ValueRange own = state.slice(2 + numIns + numOuts, numYields);

              Value bit = arith::AndIOp::create(s, loc, bits,
                                                constant32(s, 0x8000));
              Value paired = arith::CmpIOp::create(
                  s, loc, arith::CmpIPredicate::ne, bit, constant32(s, 0));
              // Both are in the frame of the group: the displacement is the
              // minimum image without a rounding (D95).
              Value d = convertReal(
                  s, loc,
                  arith::SubFOp::create(s, loc, myPosition, otherPosition),
                  computed);
              Value squares = arith::MulFOp::create(s, loc, d, d);
              Value distance2 = vector::ReductionOp::create(
                  s, loc, vector::CombiningKind::ADD, squares);
              // A slot whose bit is clear takes a squared distance beyond
              // the cutoff, so that one comparison masks it: the kernel
              // sees a finite distance and its value is not taken.
              Value r2 = arith::SelectOp::create(
                  s, loc, paired, distance2,
                  createReal(s, loc, computed, 4.0 * cutoff * cutoff));
              IRMapping inside = local;
              inside.map(kernel.getArgument(0), r2);
              inside.map(kernel.getArgument(1), d);
              for (unsigned i = 0; i != numIns; ++i) {
                inside.map(kernel.getArgument(2 + 2 * i), myValues[i]);
                inside.map(kernel.getArgument(3 + 2 * i), otherValues[i]);
              }
              for (Operation &nested : kernel.without_terminator())
                s.clone(nested, inside);
              Value within = arith::CmpFOp::create(
                  s, loc, arith::CmpFPredicate::OLT, r2, cutoff2);

              SmallVector<Value> nextOwn, nextReceived;
              for (unsigned i = 0; i != numYields; ++i) {
                Value narrow = getNarrow(i);
                Value value =
                    inside.lookupOrDefault(narrow ? narrow : yield->getOperand(i));
                Value nothing = createZero(s, loc, value.getType());
                Value masked =
                    arith::SelectOp::create(s, loc, within, value, nothing);
                nextOwn.push_back(arith::AddFOp::create(s, loc, own[i], masked));
                if (i < numOuts) {
                  // k(j, i) = s k(i, j), s the sign of the contract.
                  bool antisymmetric =
                      op.getExchange(i) == md::Exchange::Antisymmetric;
                  nextReceived.push_back(
                      antisymmetric
                          ? arith::SubFOp::create(s, loc, received[i], masked)
                                .getResult()
                          : arith::AddFOp::create(s, loc, received[i], masked)
                                .getResult());
                }
              }
              SmallVector<Value> next = {
                  shuffle(s, otherPosition),
                  arith::ShLIOp::create(s, loc, bits, constant32(s, 1))};
              for (Value value : otherValues)
                next.push_back(shuffle(s, value));
              for (Value value : nextReceived)
                next.push_back(shuffle(s, value));
              next.append(nextOwn.begin(), nextOwn.end());
              scf::YieldOp::create(s, loc, next);
            });
        // After 16 steps each entry is back at its lane: its values go to
        // its particle.
        ValueRange results = steps.getResults();
        Value particle32 = memref::LoadOp::create(b, loc, lists.order,
                                                  ValueRange{place});
        Value real = arith::AndIOp::create(
            b, loc, has,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                  particle32, constant32(b, 0)));
        scf::IfOp::create(b, loc, real, [&](OpBuilder &then, Location) {
          Value particle = toIndex(then, particle32);
          for (unsigned i = 0; i != numOuts; ++i) {
            Value destination = op.getOuts()[i];
            Type stored = getElementTypeOrSelf(destination.getType());
            emitAtomicAdd(then, loc,
                          convertReal(then, loc, results[2 + numIns + i],
                                      stored),
                          destination, particle);
          }
          scf::YieldOp::create(then, loc);
        });
        scf::YieldOp::create(
            b, loc,
            ValueRange(results.slice(2 + numIns + numOuts, numYields)));
      });

  // The values of the particles of the group: the two half-warps add up,
  // and the first adds them to the destinations.
  SmallVector<Value> totals(rounds.getResults());
  Value sixteen = constant32(builder, 16);
  for (unsigned i = 0; i != numOuts; ++i) {
    SmallVector<Value> parts = getComponents(builder, loc, totals[i]);
    for (Value &part : parts)
      part = arith::AddFOp::create(
          builder, loc, part,
          gpu::ShuffleOp::create(builder, loc, part, sixteen, width,
                                 gpu::ShuffleMode::XOR)
              .getShuffleResult());
    totals[i] = fromComponents(builder, loc, totals[i].getType(), parts);
  }
  Value particle32 = memref::LoadOp::create(builder, loc, lists.order,
                                            ValueRange{mine});
  Value writes = arith::AndIOp::create(
      builder, loc,
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, half,
                            constant32(builder, 0)),
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                            particle32, constant32(builder, 0)));
  if (numOuts != 0)
    scf::IfOp::create(builder, loc, writes, [&](OpBuilder &then, Location) {
      Value particle = toIndex(then, particle32);
      for (unsigned i = 0; i != numOuts; ++i) {
        Value destination = op.getOuts()[i];
        Type stored = getElementTypeOrSelf(destination.getType());
        emitAtomicAdd(then, loc, convertReal(then, loc, totals[i], stored),
                      destination, particle);
      }
      scf::YieldOp::create(then, loc);
    });

  // A weight is that of the sum over both orders of each pair; the
  // kernel of a sum is symmetric, so each pair once counts twice.
  SmallVector<Value> contributions;
  for (unsigned i = numOuts; i != numYields; ++i) {
    Value total = totals[i];
    if (getNarrow(i))
      total = arith::ExtFOp::create(builder, loc,
                                    yield->getOperand(i).getType(), total);
    double weight = 2.0;
    if (auto weights = op.getWeights())
      weight *= (*weights)[i - numOuts];
    if (weight != 1.0) {
      Type type = total.getType();
      FloatAttr scalar =
          builder.getFloatAttr(getElementTypeOrSelf(type), weight);
      Value factor;
      if (auto vector = dyn_cast<VectorType>(type))
        factor = arith::ConstantOp::create(
            builder, loc, type,
            DenseElementsAttr::get(vector, scalar.getValue()));
      else
        factor = arith::ConstantOp::create(builder, loc, type, scalar);
      total = arith::MulFOp::create(builder, loc, factor, total);
    }
    contributions.push_back(total);
  }
  return contributions;
}

//===----------------------------------------------------------------------===//
// Loops over tuples
//===----------------------------------------------------------------------===//

/// The value of `buffer`, a buffer of i32, at `row` and `column`, as an
/// index.
static Value loadIndex(OpBuilder &builder, Location loc, Value buffer,
                       Value row, Value column) {
  Value narrow =
      memref::LoadOp::create(builder, loc, buffer, ValueRange{row, column});
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                    narrow);
}

namespace {
/// A tuple of the row of a particle in an incidence structure, evaluated:
/// its number, the place of the particle, its members, and the mapping
/// from the values of the kernel to those that it computes.
struct EvaluatedTuple {
  Value tuple, place;
  SmallVector<Value, 4> members;
  IRMapping inside;
};
} // namespace

/// How a kernel loads the value of a member from a buffer, in `type` if
/// it is not null (the kernel narrows the value to it at once), and
/// whether the value is the position whose differences the kernel takes:
/// `loadElement`, or a value held in registers (emitIntegrationThread),
/// where a position may be one relative to the tuple, in the type of the
/// kernel.
using LoadMember = function_ref<Value(OpBuilder &, Value, Value, Type, bool)>;

/// The type that every use of `argument` narrows it to, if each is an
/// arith.truncf to one type; otherwise null.
static Type getNarrowedType(BlockArgument argument) {
  Type narrowed;
  for (Operation *user : argument.getUsers()) {
    auto truncate = dyn_cast<arith::TruncFOp>(user);
    if (!truncate || truncate.getRoundingmodeAttr() ||
        (narrowed && truncate.getType() != narrowed))
      return Type();
    narrowed = truncate.getType();
  }
  return narrowed;
}

static void evaluateMembers(OpBuilder &b, md_exec::TupleForOp op,
                            EvaluatedTuple &result, Value box, Value inverse,
                            const IRMapping &local,
                            LoadMember loadMember = nullptr);

/// Loads the entry `number` of the row of `particle` in `incidence`: the
/// tuple, the place of the particle in it, and its members.
static EvaluatedTuple loadTuple(OpBuilder &b, md_exec::TupleForOp op,
                                Value incidence, Value particle,
                                Value number) {
  Location loc = op.getLoc();
  int64_t arity = op.getArity();
  int64_t entry = md_exec::getIncidenceEntrySize(arity);
  Value one = createIndex(b, loc, 1);
  Value offset =
      arith::MulIOp::create(b, loc, number, createIndex(b, loc, entry));
  Value base = arith::AddIOp::create(b, loc, offset, one);
  auto column = [&](int64_t c) -> Value {
    return arith::AddIOp::create(b, loc, base, createIndex(b, loc, c));
  };
  EvaluatedTuple result;
  result.tuple = loadIndex(b, loc, incidence, particle, column(0));
  result.place = loadIndex(b, loc, incidence, particle, column(1));
  for (int64_t q = 0; q != arity; ++q)
    result.members.push_back(
        loadIndex(b, loc, incidence, particle, column(2 + q)));
  return result;
}

/// Loads the entry `number` of the row of `particle` in `incidence` and
/// emits the kernel of `op` for it: the displacements in the minimum image,
/// taken in the type of the positions and imaged in `computed`, the values
/// of `ins` for each member and of `parameters` for the tuple.
static EvaluatedTuple evaluateTuple(OpBuilder &b, md_exec::TupleForOp op,
                                    Value incidence, Value particle,
                                    Value number, Value box, Value inverse,
                                    const IRMapping &local) {
  EvaluatedTuple result = loadTuple(b, op, incidence, particle, number);
  evaluateMembers(b, op, result, box, inverse, local);
  return result;
}

/// Emits the kernel of `op` for the tuple `result.tuple` of the members
/// `result.members`, as `evaluateTuple` does.
static void evaluateMembers(OpBuilder &b, md_exec::TupleForOp op,
                            EvaluatedTuple &result, Value box, Value inverse,
                            const IRMapping &local, LoadMember loadMember) {
  Location loc = op.getLoc();
  Block &kernel = op.getKernel().front();
  int64_t arity = op.getArity();
  IRMapping &inside = result.inside;
  inside = local;
  Value positions = op.getPositions();
  auto load = [&](Value buffer, Value member, Type type,
                  bool position = false) -> Value {
    if (loadMember)
      return loadMember(b, buffer, member, type, position);
    Value value = loadElement(b, loc, buffer, member);
    if (type)
      value = arith::TruncFOp::create(b, loc, type, value);
    return value;
  };
  SmallVector<Value, 4> memberPositions(arity);
  auto positionOf = [&](int64_t q) {
    if (!memberPositions[q])
      memberPositions[q] =
          load(positions, result.members[q], Type(), /*position=*/true);
    return memberPositions[q];
  };
  Type computed = getElementTypeOrSelf(box.getType());
  for (auto [index, coordinate] : llvm::enumerate(op.getCoordinates())) {
    Value raw = convertReal(
        b, loc,
        arith::SubFOp::create(b, loc, positionOf(coordinate.members[0]),
                              positionOf(coordinate.members[1])),
        computed);
    Value images = arith::MulFOp::create(b, loc, raw, inverse);
    Value nearest = math::RoundEvenOp::create(b, loc, images);
    Value shift = arith::MulFOp::create(b, loc, nearest, box);
    Value d = arith::SubFOp::create(b, loc, raw, shift);
    inside.map(kernel.getArgument(index), d);
  }
  // A value that the kernel narrows at once is loaded narrowed, and the
  // ops that narrow it are not cloned.
  unsigned argument = op.getCoordinateKinds().size();
  for (Value buffer : op.getIns())
    for (int64_t q = 0; q != arity; ++q) {
      BlockArgument in = kernel.getArgument(argument++);
      if (in.use_empty())
        continue;
      Type narrowed = getNarrowedType(in);
      if (!narrowed) {
        inside.map(in, load(buffer, result.members[q], Type()));
        continue;
      }
      Value value = load(buffer, result.members[q], narrowed);
      for (Operation *user : in.getUsers())
        inside.map(user->getResult(0), value);
    }
  for (Value buffer : op.getParameters())
    inside.map(kernel.getArgument(argument++),
               loadElement(b, loc, buffer, result.tuple));
  for (Operation &nested : kernel.without_terminator()) {
    if (nested.getNumResults() == 1 && isa<arith::TruncFOp>(nested) &&
        inside.contains(nested.getResult(0)))
      continue;
    b.clone(nested, inside);
  }
}

/// The loop of `emitTupleKernel` for a set whose tuples share no particle:
/// the member at place 0 evaluates the tuple once and writes the values of
/// all members; a particle in no tuple writes 0 where the loop overwrites.
static SmallVector<Value> emitOwnedTupleKernel(OpBuilder &builder,
                                               md_exec::TupleForOp op,
                                               Value incidence, Value box,
                                               Value inverse, Value particle,
                                               IRMapping &local) {
  Location loc = op.getLoc();
  Block &kernel = op.getKernel().front();
  Operation *yield = kernel.getTerminator();
  int64_t arity = op.getArity();
  unsigned numOuts = op.getOuts().size();
  unsigned numYields = yield->getNumOperands();
  Value zero = createIndex(builder, loc, 0);

  SmallVector<Type> sumTypes;
  for (unsigned i = numOuts * arity; i != numYields; ++i)
    sumTypes.push_back(yield->getOperand(i).getType());
  auto zeros = [&](OpBuilder &b) {
    SmallVector<Value> values;
    for (Type type : sumTypes)
      values.push_back(createZero(b, loc, type));
    return values;
  };

  Value count = loadIndex(builder, loc, incidence, particle, zero);
  Value member = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                       count, zero);
  auto branch = scf::IfOp::create(
      builder, loc, member,
      [&](OpBuilder &b, Location) {
        EvaluatedTuple evaluated = evaluateTuple(
            b, op, incidence, particle, zero, box, inverse, local);
        Value owner = arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::eq, evaluated.place, zero);
        auto owned = scf::IfOp::create(
            b, loc, owner,
            [&](OpBuilder &c, Location) {
              for (unsigned i = 0; i != numOuts; ++i) {
                Value destination = op.getOuts()[i];
                for (int64_t q = 0; q != arity; ++q) {
                  Value target = evaluated.members[q];
                  Value value = evaluated.inside.lookupOrDefault(
                      yield->getOperand(i * arity + q));
                  if (!op.overwrites(i))
                    value = arith::AddFOp::create(
                        c, loc, loadElement(c, loc, destination, target),
                        value);
                  storeElement(c, loc, value, destination, target);
                }
              }
              SmallVector<Value> contributions;
              for (unsigned i = numOuts * arity; i != numYields; ++i)
                contributions.push_back(
                    evaluated.inside.lookupOrDefault(yield->getOperand(i)));
              scf::YieldOp::create(c, loc, contributions);
            },
            [&](OpBuilder &c, Location) {
              scf::YieldOp::create(c, loc, zeros(c));
            });
        scf::YieldOp::create(b, loc, owned.getResults());
      },
      [&](OpBuilder &b, Location) {
        for (unsigned i = 0; i != numOuts; ++i)
          if (op.overwrites(i))
            storeElement(
                b, loc,
                createZero(b, loc, yield->getOperand(i * arity).getType()),
                op.getOuts()[i], particle);
        scf::YieldOp::create(b, loc, zeros(b));
      });
  return SmallVector<Value>(branch.getResults());
}

SmallVector<Value> kernels::emitTupleKernel(OpBuilder &builder,
                                            md_exec::TupleForOp op,
                                            Value incidence, Value box,
                                            Value inverse, Value particle,
                                            IRMapping &local,
                                            const RowLanes *lanes,
                                            SmallVectorImpl<Value> *outTotals) {
  Location loc = op.getLoc();
  Block &kernel = op.getKernel().front();
  Operation *yield = kernel.getTerminator();
  int64_t arity = op.getArity();
  unsigned numOuts = op.getOuts().size();
  unsigned numYields = yield->getNumOperands();

  // As for pairs, the differences of the positions are taken in their type
  // and the minimum image in the type that the kernel computes in.
  Type computed =
      cast<VectorType>(kernel.getArgument(0).getType()).getElementType();
  Value boxComputed = convertReal(builder, loc, box, computed);
  Value inverseComputed = convertReal(builder, loc, inverse, computed);
  if (op.getDisjoint() && !lanes)
    return emitOwnedTupleKernel(builder, op, incidence, boxComputed,
                                inverseComputed, particle, local);
  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);

  // What the particle accumulates: one value for each destination, then
  // one for each global sum.
  SmallVector<Value> sums;
  for (unsigned i = 0; i != numOuts; ++i)
    sums.push_back(createZero(builder, loc, yield->getOperand(i * arity).getType()));
  // A contribution to a global sum that the kernel widens is summed over
  // the row in the type it was computed in and widened once (as in
  // emitPairKernel).
  auto getNarrow = [&](unsigned i) -> Value {
    auto widen = yield->getOperand(i).getDefiningOp<arith::ExtFOp>();
    if (!widen || widen->getBlock() != &kernel)
      return Value();
    return widen.getIn();
  };
  for (unsigned i = numOuts * arity; i != numYields; ++i) {
    Value narrow = getNarrow(i);
    sums.push_back(createZero(
        builder, loc,
        narrow ? narrow.getType() : yield->getOperand(i).getType()));
  }

  Value count = loadIndex(builder, loc, incidence, particle, zero);
  Value first = zero, step = one;
  if (lanes) {
    count = arith::SelectOp::create(builder, loc, lanes->valid, count, zero);
    first = lanes->lane;
    step = createIndex(builder, loc, lanes->lanes);
  }
  auto loop = scf::ForOp::create(
      builder, loc, first, count, step, sums,
      [&](OpBuilder &b, Location, Value number, ValueRange partial) {
        EvaluatedTuple evaluated =
            evaluateTuple(b, op, incidence, particle, number, boxComputed,
                          inverseComputed, local);
        IRMapping &inside = evaluated.inside;
        Value place = evaluated.place;

        // The value for the place of the particle, of each destination.
        SmallVector<Value> updated;
        for (unsigned i = 0; i != numOuts; ++i) {
          Value value = inside.lookupOrDefault(yield->getOperand(i * arity));
          for (int64_t q = 1; q != arity; ++q) {
            Value here = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::eq, place,
                createIndex(b, loc, q));
            value = arith::SelectOp::create(
                b, loc, here,
                inside.lookupOrDefault(yield->getOperand(i * arity + q)),
                value);
          }
          updated.push_back(arith::AddFOp::create(b, loc, partial[i], value));
        }

        // A tuple adds to a sum once, through its member at place 0.
        Value atFirst = arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::eq, place, zero);
        for (unsigned i = numOuts * arity, j = numOuts; i != numYields;
             ++i, ++j) {
          Value narrow = getNarrow(i);
          Value contribution =
              inside.lookupOrDefault(narrow ? narrow : yield->getOperand(i));
          Value nothing = createZero(b, loc, contribution.getType());
          Value masked =
              arith::SelectOp::create(b, loc, atFirst, contribution, nothing);
          updated.push_back(arith::AddFOp::create(b, loc, partial[j], masked));
        }
        scf::YieldOp::create(b, loc, updated);
      });

  SmallVector<Value> totals(loop.getResults());
  for (unsigned i = numOuts * arity, j = numOuts; i != numYields; ++i, ++j)
    if (getNarrow(i))
      totals[j] = arith::ExtFOp::create(
          builder, loc, yield->getOperand(i).getType(), totals[j]);
  if (lanes)
    for (Value &total : totals)
      total = lanes->combine(builder, loc, total);

  auto emitStores = [&](OpBuilder &writer) {
    for (unsigned i = 0; i != numOuts; ++i) {
      Value destination = op.getOuts()[i];
      Value total = totals[i];
      if (!op.overwrites(i))
        total = arith::AddFOp::create(
            writer, loc, loadElement(writer, loc, destination, particle),
            total);
      storeElement(writer, loc, total, destination, particle);
    }
  };
  if (outTotals) {
    outTotals->assign(totals.begin(), totals.begin() + numOuts);
  } else if (lanes && numOuts != 0) {
    Value leader = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         lanes->lane, zero);
    Value writes = arith::AndIOp::create(builder, loc, leader, lanes->valid);
    scf::IfOp::create(builder, loc, writes, [&](OpBuilder &then, Location) {
      emitStores(then);
      scf::YieldOp::create(then, loc);
    });
  } else {
    emitStores(builder);
  }

  return SmallVector<Value>(totals.begin() + numOuts, totals.end());
}

/// Adds `value`, a scalar or a vector, atomically to the element `row` of
/// the buffer `buffer` on a device, with no result: a relaxed atomic at the
/// scope of the device. f32 takes PTX's own reduction, as the NVPTX backend
/// of LLVM expands atomicrmw fadd of f32 into a loop of compare-and-swap.
static void emitAtomicAdd(OpBuilder &b, Location loc, Value value,
                          Value buffer, Value row) {
  auto type = cast<MemRefType>(buffer.getType());
  Type element = type.getElementType();
  int64_t components = type.getRank() == 1 ? 1 : type.getDimSize(1);
  int64_t bytes = element.getIntOrFloatBitWidth() / 8;
  Type wide = b.getI64Type();
  Value base = arith::IndexCastOp::create(
      b, loc, wide,
      memref::ExtractAlignedPointerAsIndexOp::create(b, loc, buffer));
  Value first = arith::MulIOp::create(
      b, loc, arith::IndexCastOp::create(b, loc, wide, row),
      arith::ConstantOp::create(b, loc, wide,
                                b.getI64IntegerAttr(components * bytes)));
  for (int64_t c = 0; c != components; ++c) {
    Value part = components == 1
                     ? value
                     : vector::ExtractOp::create(b, loc, value, c).getResult();
    Value offset = arith::AddIOp::create(
        b, loc, first,
        arith::ConstantOp::create(b, loc, wide, b.getI64IntegerAttr(c * bytes)));
    Value address = arith::AddIOp::create(b, loc, base, offset);
    Value pointer = LLVM::IntToPtrOp::create(
        b, loc, LLVM::LLVMPointerType::get(b.getContext(), 1), address);
    if (element.isF32()) {
      LLVM::InlineAsmOp::create(
          b, loc, TypeRange(), ValueRange{pointer, part},
          "red.relaxed.gpu.global.add.f32 [$0], $1;", "l,f",
          /*has_side_effects=*/true, /*is_align_stack=*/false,
          LLVM::TailCallKind::None, /*asm_dialect=*/LLVM::AsmDialectAttr(),
          /*operand_attrs=*/ArrayAttr());
    } else {
      LLVM::AtomicRMWOp::create(b, loc, LLVM::AtomicBinOp::fadd, pointer,
                                part, LLVM::AtomicOrdering::monotonic,
                                StringRef("device"));
    }
  }
}

void kernels::emitTupleOnce(OpBuilder &builder, md_exec::TupleForOp op,
                            Value members, Value tuple, Value box,
                            Value inverse, IRMapping &local) {
  Location loc = op.getLoc();
  Block &kernel = op.getKernel().front();
  Operation *yield = kernel.getTerminator();
  int64_t arity = op.getArity();
  Type computed =
      cast<VectorType>(kernel.getArgument(0).getType()).getElementType();
  Value boxComputed = convertReal(builder, loc, box, computed);
  Value inverseComputed = convertReal(builder, loc, inverse, computed);

  EvaluatedTuple evaluated;
  evaluated.tuple = tuple;
  for (int64_t q = 0; q != arity; ++q)
    evaluated.members.push_back(
        loadIndex(builder, loc, members, tuple, createIndex(builder, loc, q)));
  evaluateMembers(builder, op, evaluated, boxComputed, inverseComputed,
                  local);
  for (unsigned i = 0, e = op.getOuts().size(); i != e; ++i) {
    Value destination = op.getOuts()[i];
    Type stored = getElementTypeOrSelf(destination.getType());
    for (int64_t q = 0; q != arity; ++q) {
      Value value = evaluated.inside.lookupOrDefault(
          yield->getOperand(i * arity + q));
      emitAtomicAdd(builder, loc, convertReal(builder, loc, value, stored),
                    destination, evaluated.members[q]);
    }
  }
}

SmallVector<Value>
kernels::emitTuplesOnceWithSums(OpBuilder &builder, md_exec::TupleForOp op,
                                Value members, Value first, Value stride,
                                Value count, Value box, Value inverse) {
  Location loc = op.getLoc();
  Block &kernel = op.getKernel().front();
  Operation *yield = kernel.getTerminator();
  int64_t arity = op.getArity();
  unsigned numOuts = op.getOuts().size();
  unsigned numYields = yield->getNumOperands();
  Type computed =
      cast<VectorType>(kernel.getArgument(0).getType()).getElementType();
  Value boxComputed = convertReal(builder, loc, box, computed);
  Value inverseComputed = convertReal(builder, loc, inverse, computed);
  // A contribution that the kernel widens is summed in the type it was
  // computed in (as in emitTupleKernel).
  auto getNarrow = [&](unsigned i) -> Value {
    auto widen = yield->getOperand(i).getDefiningOp<arith::ExtFOp>();
    if (!widen || widen->getBlock() != &kernel)
      return Value();
    return widen.getIn();
  };
  SmallVector<Value> sums;
  for (unsigned i = numOuts * arity; i != numYields; ++i) {
    Value narrow = getNarrow(i);
    sums.push_back(createZero(
        builder, loc,
        narrow ? narrow.getType() : yield->getOperand(i).getType()));
  }
  auto loop = scf::ForOp::create(
      builder, loc, first, count, stride, sums,
      [&](OpBuilder &b, Location, Value tuple, ValueRange partial) {
        EvaluatedTuple evaluated;
        evaluated.tuple = tuple;
        for (int64_t q = 0; q != arity; ++q)
          evaluated.members.push_back(
              loadIndex(b, loc, members, tuple, createIndex(b, loc, q)));
        evaluateMembers(b, op, evaluated, boxComputed, inverseComputed,
                        IRMapping());
        IRMapping &inside = evaluated.inside;
        for (unsigned i = 0; i != numOuts; ++i) {
          Value destination = op.getOuts()[i];
          Type stored = getElementTypeOrSelf(destination.getType());
          for (int64_t q = 0; q != arity; ++q) {
            Value value =
                inside.lookupOrDefault(yield->getOperand(i * arity + q));
            emitAtomicAdd(b, loc, convertReal(b, loc, value, stored),
                          destination, evaluated.members[q]);
          }
        }
        SmallVector<Value> updated;
        for (unsigned i = numOuts * arity, j = 0; i != numYields; ++i, ++j) {
          Value narrow = getNarrow(i);
          Value contribution =
              inside.lookupOrDefault(narrow ? narrow : yield->getOperand(i));
          updated.push_back(
              arith::AddFOp::create(b, loc, partial[j], contribution));
        }
        scf::YieldOp::create(b, loc, updated);
      });
  SmallVector<Value> totals(loop.getResults());
  for (unsigned i = numOuts * arity, j = 0; i != numYields; ++i, ++j)
    if (getNarrow(i))
      totals[j] = arith::ExtFOp::create(
          builder, loc, yield->getOperand(i).getType(), totals[j]);
  return totals;
}

int64_t kernels::getIntegrationStride(const IntegrationRun &run) {
  int64_t widest = 1;
  for (md_exec::TupleForOp loop : run.loops)
    widest = std::max<int64_t>(widest, loop.getArity());
  return 32 - (widest - 1);
}

void kernels::emitIntegrationThread(
    OpBuilder &builder, const IntegrationRun &run, ArrayRef<Value> boxes,
    ArrayRef<Value> inverses, Value thread, Value span, Value acrossList,
    Value acrossCount, bool across,
    function_ref<void(OpBuilder &, Value, ArrayRef<Value>)> storeAfter) {
  md_exec::ParticleForOp before = run.before, after = run.after;
  SmallVector<md_exec::TupleForOp, 4> loops(run.loops.begin(),
                                            run.loops.end());
  Location loc = after.getLoc();
  Type i32 = builder.getI32Type();
  Value zero = createIndex(builder, loc, 0);
  Value width = arith::ConstantOp::create(builder, loc, i32,
                                          builder.getI32IntegerAttr(32));
  unsigned beforeOuts = before.getOuts().size();
  unsigned afterOuts = after.getOuts().size();
  auto toI32 = [&](OpBuilder &b, Value index) -> Value {
    return arith::IndexCastOp::create(b, loc, i32, index);
  };
  auto True = [&](OpBuilder &b) -> Value {
    return arith::ConstantOp::create(b, loc, b.getI1Type(),
                                     b.getBoolAttr(true));
  };

  // `value` from the lane `source` of the warp; numbers of 64 bits travel
  // as two of 32.
  auto shuffleFrom = [&](OpBuilder &b, Value value, Value source) -> Value {
    SmallVector<Value> parts = getComponents(b, loc, value);
    for (Value &part : parts) {
      Type type = part.getType();
      if (type.getIntOrFloatBitWidth() == 64) {
        Type i64 = b.getI64Type();
        Value bits = arith::BitcastOp::create(b, loc, i64, part);
        Value low = arith::TruncIOp::create(b, loc, i32, bits);
        Value high = arith::TruncIOp::create(
            b, loc, i32,
            arith::ShRUIOp::create(
                b, loc, bits,
                arith::ConstantOp::create(b, loc, i64,
                                          b.getI64IntegerAttr(32))));
        low = gpu::ShuffleOp::create(b, loc, low, source, width,
                                     gpu::ShuffleMode::IDX)
                  .getShuffleResult();
        high = gpu::ShuffleOp::create(b, loc, high, source, width,
                                      gpu::ShuffleMode::IDX)
                   .getShuffleResult();
        Value joined = arith::OrIOp::create(
            b, loc, arith::ExtUIOp::create(b, loc, i64, low),
            arith::ShLIOp::create(
                b, loc, arith::ExtUIOp::create(b, loc, i64, high),
                arith::ConstantOp::create(b, loc, i64,
                                          b.getI64IntegerAttr(32))));
        part = arith::BitcastOp::create(b, loc, type, joined);
      } else {
        part = gpu::ShuffleOp::create(b, loc, part, source, width,
                                      gpu::ShuffleMode::IDX)
                   .getShuffleResult();
      }
    }
    return fromComponents(b, loc, value.getType(), parts);
  };

  // The values of the loop before for a member, from its inputs as they
  // are stored, before the thread writes anything.
  auto evalBefore = [&](OpBuilder &b, Value member) {
    IRMapping local;
    Block &kernel = before.getKernel().front();
    for (auto [index, buffer] : llvm::enumerate(before.getIns()))
      local.map(kernel.getArgument(index), loadElement(b, loc, buffer, member));
    for (Operation &nested : kernel.without_terminator())
      b.clone(nested, local);
    Operation *yield = kernel.getTerminator();
    SmallVector<Value> values;
    for (unsigned i = 0; i != beforeOuts; ++i)
      values.push_back(local.lookupOrDefault(yield->getOperand(i)));
    return values;
  };
  // The values of the loop before for a member, as the first kernel wrote
  // them.
  auto loadBefore = [&](OpBuilder &b, Value member) {
    SmallVector<Value> values;
    for (Value buffer : before.getOuts())
      values.push_back(loadElement(b, loc, buffer, member));
    return values;
  };
  auto fromBefore = [&](Value buffer) -> int {
    for (unsigned j = 0; j != beforeOuts; ++j)
      if (before.getOuts()[j] == buffer)
        return j;
    return -1;
  };
  auto fromLoop = [&](Value buffer) -> std::pair<int, int> {
    for (auto [i, loop] : llvm::enumerate(loops))
      for (auto [o, out] : llvm::enumerate(loop.getOuts()))
        if (out == buffer)
          return {int(i), int(o)};
    return {-1, -1};
  };
  // The types of the values of a loop over tuples for one member: one for
  // each destination.
  auto correctionTypes = [&](md_exec::TupleForOp loop) {
    Operation *yield = loop.getKernel().front().getTerminator();
    int64_t arity = loop.getArity();
    SmallVector<Type> types;
    for (unsigned o = 0, e = loop.getOuts().size(); o != e; ++o)
      types.push_back(yield->getOperand(o * arity).getType());
    return types;
  };
  // The cell of a loop in the type its kernel computes in.
  auto cellOf = [&](OpBuilder &, unsigned i) -> std::pair<Value, Value> {
    return {boxes[i], inverses[i]};
  };

  // The loop after for a member, its values of the loop before from
  // registers and those of the loops over tuples from `corrections` (one
  // for each destination of each loop), and its contributions handed
  // over; the destinations of the loops over tuples for the member, and
  // those of the loop before that the loop after does not write, too.
  auto evalAfter = [&](OpBuilder &b, Value member, ArrayRef<Value> values,
                       ArrayRef<SmallVector<Value>> corrections) {
    for (auto [i, loop] : llvm::enumerate(loops))
      for (auto [o, out] : llvm::enumerate(loop.getOuts())) {
        if (!run.keepOuts[i][o])
          continue;
        Value value = corrections[i][o];
        if (!loop.overwrites(o))
          value = arith::AddFOp::create(
              b, loc, loadElement(b, loc, out, member), value);
        storeElement(b, loc, value, out, member);
      }
    IRMapping local;
    Block &kernel = after.getKernel().front();
    for (auto [index, buffer] : llvm::enumerate(after.getIns())) {
      Type type = kernel.getArgument(index).getType();
      Type real = getElementTypeOrSelf(type);
      Value value;
      if (int j = fromBefore(buffer); j >= 0)
        value = convertReal(b, loc, values[j], real);
      else if (auto [i, o] = fromLoop(buffer); i >= 0)
        value = convertReal(b, loc, corrections[i][o], real);
      else
        value = loadElement(b, loc, buffer, member);
      local.map(kernel.getArgument(index), value);
    }
    for (Operation &nested : kernel.without_terminator())
      b.clone(nested, local);
    Operation *yield = kernel.getTerminator();
    for (unsigned i = 0; i != afterOuts; ++i)
      storeElement(b, loc, local.lookupOrDefault(yield->getOperand(i)),
                   after.getOuts()[i], member);
    for (unsigned j = 0; j != beforeOuts; ++j)
      if (!llvm::is_contained(after.getOuts(), before.getOuts()[j]))
        storeElement(b, loc, values[j], before.getOuts()[j], member);
    SmallVector<Value> contributions;
    for (unsigned i = afterOuts, e = yield->getNumOperands(); i != e; ++i)
      contributions.push_back(local.lookupOrDefault(yield->getOperand(i)));
    storeAfter(b, member, contributions);
  };

  // The tuple of a particle in each loop, if any (at most one over the
  // loops), and whether its members all lie in the warp of the particle.
  struct Row {
    EvaluatedTuple tuple;
    Value inTuple, inWarp, useWarp, leaderLane, leadsAcross;
  };
  auto loadRows = [&](OpBuilder &b, Value particle, Value lane32,
                      Value warpBase, Value warpEnd, Value valid) {
    SmallVector<Row> rows;
    for (auto [i, loop] : llvm::enumerate(loops)) {
      Row row;
      Value count = loadIndex(b, loc, loop.getIncidence(), particle, zero);
      row.inTuple = arith::AndIOp::create(
          b, loc, valid,
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, count,
                                zero));
      row.tuple = loadTuple(b, loop, loop.getIncidence(), particle, zero);
      Value inWarp;
      for (Value member : row.tuple.members) {
        Value in = arith::AndIOp::create(
            b, loc,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, member,
                                  warpBase),
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, member,
                                  warpEnd));
        inWarp = inWarp ? arith::AndIOp::create(b, loc, inWarp, in) : in;
      }
      row.inWarp = inWarp;
      row.useWarp = arith::AndIOp::create(b, loc, row.inTuple, inWarp);
      row.leaderLane = arith::SelectOp::create(
          b, loc, row.useWarp,
          toI32(b, arith::SubIOp::create(b, loc, row.tuple.members[0],
                                         warpBase)),
          lane32);
      Value leads = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                          row.tuple.place, zero);
      row.leadsAcross = arith::AndIOp::create(
          b, loc,
          arith::AndIOp::create(
              b, loc, row.inTuple,
              arith::XOrIOp::create(b, loc, inWarp, True(b))),
          leads);
      rows.push_back(row);
    }
    return rows;
  };

  // The second kernel: the tuples that lie across warps, from the list
  // the first kernel made, `span` threads in one block. Their member at
  // place 0 takes every member: the values of the loop before that the
  // first kernel wrote, the tuple, and the loop after for each. The block
  // then clears the count for the next run.
  if (across) {
    Value count = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(),
        memref::LoadOp::create(builder, loc, acrossCount, ValueRange{zero}));
    scf::ForOp::create(
        builder, loc, thread, count, span, ValueRange(),
        [&](OpBuilder &b, Location, Value entry, ValueRange) {
      Value particle = arith::IndexCastOp::create(
          b, loc, b.getIndexType(),
          memref::LoadOp::create(b, loc, acrossList, ValueRange{entry}));
      SmallVector<Row> rows =
          loadRows(b, particle, toI32(b, zero), particle, particle, True(b));
      for (auto [i, loop] : llvm::enumerate(loops)) {
        Row &row = rows[i];
        Value leads = arith::AndIOp::create(
            b, loc, row.inTuple,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                  row.tuple.place, zero));
        scf::IfOp::create(b, loc, leads, [&](OpBuilder &c, Location) {
          int64_t arity = loop.getArity();
          EvaluatedTuple evaluated = row.tuple;
          SmallVector<SmallVector<Value>, 4> values;
          for (int64_t q = 0; q != arity; ++q)
            values.push_back(loadBefore(c, evaluated.members[q]));
          auto loadMember = [&](OpBuilder &d, Value buffer, Value member,
                                Type type, bool) -> Value {
            Value value;
            if (int j = fromBefore(buffer); j < 0) {
              value = loadElement(d, loc, buffer, member);
            } else {
              for (int64_t q = 0; q != arity && !value; ++q)
                if (evaluated.members[q] == member)
                  value = values[q][j];
              assert(value && "a member of the tuple");
            }
            if (type)
              value = arith::TruncFOp::create(d, loc, type, value);
            return value;
          };
          auto [box, inverse] = cellOf(c, i);
          evaluateMembers(c, loop, evaluated, box, inverse, IRMapping(),
                          loadMember);
          Operation *yield = loop.getKernel().front().getTerminator();
          for (int64_t q = 0; q != arity; ++q) {
            SmallVector<SmallVector<Value>> corrections;
            for (auto [k, other] : llvm::enumerate(loops)) {
              SmallVector<Value> ofLoop;
              SmallVector<Type> types = correctionTypes(other);
              for (unsigned o = 0, e = types.size(); o != e; ++o)
                ofLoop.push_back(
                    k == i ? evaluated.inside.lookupOrDefault(
                                 yield->getOperand(o * arity + q))
                           : createZero(c, loc, types[o]));
              corrections.push_back(ofLoop);
            }
            evalAfter(c, evaluated.members[q], values[q], corrections);
          }
          scf::YieldOp::create(c, loc);
        });
      }
      scf::YieldOp::create(b, loc);
    });
    gpu::BarrierOp::create(builder, loc);
    Value first = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                        thread, zero);
    scf::IfOp::create(builder, loc, first, [&](OpBuilder &b, Location) {
      memref::StoreOp::create(
          b, loc, arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(0)),
          acrossCount, ValueRange{zero});
      scf::YieldOp::create(b, loc);
    });
    return;
  }

  // The first kernel, over `span` particles. Warp w takes the particles
  // from b(S w) to b(S (w + 1)), where S is 32 less the widest arity less
  // one and b(n) is n moved up past the tuple that n is in if its members
  // follow one another and n is not the first: a tuple whose members
  // follow one another then lies in one warp, and a warp takes 32
  // particles at most. The driver keeps the members of a group together
  // (an order by anchors); the second kernel takes the other tuples that
  // lie across warps.
  Value c32 = createIndex(builder, loc, 32);
  Value warp = arith::DivUIOp::create(builder, loc, thread, c32);
  Value lane = arith::RemUIOp::create(builder, loc, thread, c32);
  Value stride = createIndex(builder, loc, getIntegrationStride(run));
  Value last = arith::SubIOp::create(builder, loc, span,
                                     createIndex(builder, loc, 1));
  auto boundary = [&](OpBuilder &b, Value n) -> Value {
    Value inside =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, n, span);
    Value at = arith::SelectOp::create(b, loc, inside, n, last);
    Value moved = at;
    for (md_exec::TupleForOp loop : loops) {
      Value count = loadIndex(b, loc, loop.getIncidence(), at, zero);
      EvaluatedTuple tuple =
          loadTuple(b, loop, loop.getIncidence(), at, zero);
      Value low = tuple.members[0], high = tuple.members[0];
      for (Value member : ArrayRef(tuple.members).drop_front()) {
        low = arith::MinUIOp::create(b, loc, low, member);
        high = arith::MaxUIOp::create(b, loc, high, member);
      }
      Value end = arith::AddIOp::create(b, loc, high,
                                        createIndex(b, loc, 1));
      Value together = arith::CmpIOp::create(
          b, loc, arith::CmpIPredicate::eq,
          arith::SubIOp::create(b, loc, end, low),
          createIndex(b, loc, loop.getArity()));
      Value straddles = arith::AndIOp::create(
          b, loc,
          arith::AndIOp::create(
              b, loc,
              arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, count,
                                    zero),
              together),
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, low, at));
      moved = arith::SelectOp::create(b, loc, straddles, end, moved);
    }
    return arith::SelectOp::create(b, loc, inside, moved, span);
  };
  Value warpBase =
      boundary(builder, arith::MulIOp::create(builder, loc, warp, stride));
  Value warpEnd = boundary(
      builder,
      arith::MulIOp::create(
          builder, loc,
          arith::AddIOp::create(builder, loc, warp,
                                createIndex(builder, loc, 1)),
          stride));
  auto nonempty = scf::IfOp::create(
      builder, loc,
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult, warpBase,
                            warpEnd),
      /*withElseRegion=*/false);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(nonempty.thenBlock()->getTerminator());

  // Every lane: the loop before for its particle. A lane past the end of
  // the particles of its warp takes the first one, takes part in the
  // shuffles, and writes nothing.
  Value candidate = arith::AddIOp::create(builder, loc, warpBase, lane);
  Value valid = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                                      candidate, warpEnd);
  Value particle =
      arith::SelectOp::create(builder, loc, valid, candidate, warpBase);
  SmallVector<Value> own = evalBefore(builder, particle);
  Value lane32 = toI32(builder, lane);
  SmallVector<Row> rows =
      loadRows(builder, particle, lane32, warpBase, warpEnd, valid);

  // The tuples whose members are in one warp: the member at place 0
  // gathers the values of every member from its lane, computes the
  // tuple, and every member takes its values from the lane of the member
  // at place 0. A particle is in one tuple at most over the loops, the
  // sets being disjoint, so one set of shuffles serves every loop; a lane
  // loads its own values, narrowed where the kernels narrow them at once.
  // The shuffles take every lane of the warp, so they are outside any
  // branch; a lane in no tuple shuffles from itself.
  int64_t widest = 0;
  Value useWarp, leaderLane = lane32;
  for (auto [i, loop] : llvm::enumerate(loops)) {
    Row &row = rows[i];
    widest = std::max<int64_t>(widest, loop.getArity());
    useWarp = useWarp ? arith::OrIOp::create(builder, loc, useWarp, row.useWarp)
                      : row.useWarp;
    leaderLane = arith::SelectOp::create(builder, loc, row.useWarp,
                                         row.leaderLane, leaderLane);
  }
  SmallVector<Value, 4> sources;
  for (int64_t q = 0; q != widest; ++q) {
    Value source = lane32;
    for (auto [i, loop] : llvm::enumerate(loops)) {
      if (q >= loop.getArity())
        continue;
      Row &row = rows[i];
      source = arith::SelectOp::create(
          builder, loc, row.useWarp,
          toI32(builder, arith::SubIOp::create(builder, loc,
                                               row.tuple.members[q],
                                               warpBase)),
          source);
    }
    sources.push_back(source);
  }
  // The values that the kernels take of each member: the positions, and
  // each of their inputs in the type they narrow it to, if any. Where a
  // kernel computes in a type narrower than that of the positions, a lane
  // takes its position relative to that of the member at place 0, in f64,
  // and narrows that: the differences of the kernel are then those of
  // small numbers, and the member at place 0 narrows nothing.
  struct Request {
    Value buffer;
    Type type;
    bool position;
    bool operator==(const Request &other) const {
      return buffer == other.buffer && type == other.type &&
             position == other.position;
    }
  };
  SmallVector<Request> requests;
  auto request = [&](Request wanted) {
    if (!llvm::is_contained(requests, wanted))
      requests.push_back(wanted);
  };
  auto positionRequest = [&](unsigned i) {
    Value positions = loops[i].getPositions();
    Type real = cast<MemRefType>(positions.getType()).getElementType();
    Type computed = getElementTypeOrSelf(boxes[i].getType());
    if (computed.getIntOrFloatBitWidth() < real.getIntOrFloatBitWidth())
      return Request{positions, VectorType::get({3}, computed), true};
    return Request{positions, Type(), true};
  };
  for (auto [i, loop] : llvm::enumerate(loops)) {
    Block &kernel = loop.getKernel().front();
    request(positionRequest(i));
    unsigned argument = loop.getCoordinateKinds().size();
    for (Value buffer : loop.getIns())
      for (int64_t q = 0, arity = loop.getArity(); q != arity; ++q) {
        BlockArgument in = kernel.getArgument(argument++);
        if (!in.use_empty())
          request({buffer, getNarrowedType(in), false});
      }
  }
  SmallVector<SmallVector<Value, 4>> gathered;
  for (auto [buffer, type, position] : requests) {
    Value value;
    if (int j = fromBefore(buffer); j >= 0)
      value = own[j];
    else
      value = loadElement(builder, loc, buffer, particle);
    if (position && type) {
      Value origin = shuffleFrom(builder, value, leaderLane);
      value = arith::TruncFOp::create(
          builder, loc, type,
          arith::SubFOp::create(builder, loc, value, origin));
    } else if (type) {
      value = arith::TruncFOp::create(builder, loc, type, value);
    }
    SmallVector<Value, 4> ofMembers;
    for (Value source : sources)
      ofMembers.push_back(shuffleFrom(builder, value, source));
    gathered.push_back(ofMembers);
  }

  // The values of each loop for each member and destination, computed by
  // the member at place 0 of its tuple.
  SmallVector<SmallVector<Value>> computedOf;
  for (auto [i, loop] : llvm::enumerate(loops)) {
    Row &row = rows[i];
    int64_t arity = loop.getArity();
    SmallVector<Type> types = correctionTypes(loop);
    Value leads = arith::AndIOp::create(
        builder, loc, row.useWarp,
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                              row.tuple.place, zero));
    auto computed = scf::IfOp::create(
        builder, loc, leads,
        [&](OpBuilder &b, Location) {
          EvaluatedTuple evaluated = row.tuple;
          Request place = positionRequest(i);
          auto loadMember = [&](OpBuilder &, Value buffer, Value member,
                                Type type, bool position) -> Value {
            auto it = llvm::find(requests, position
                                               ? place
                                               : Request{buffer, type, false});
            assert(it != requests.end() && "a value that was gathered");
            for (int64_t q = 0; q != arity; ++q)
              if (evaluated.members[q] == member)
                return gathered[it - requests.begin()][q];
            llvm_unreachable("a member of the tuple");
          };
          auto [box, inverse] = cellOf(b, i);
          evaluateMembers(b, loop, evaluated, box, inverse, IRMapping(),
                          loadMember);
          Operation *yield = loop.getKernel().front().getTerminator();
          SmallVector<Value> values;
          for (int64_t q = 0; q != arity; ++q)
            for (unsigned o = 0, e = types.size(); o != e; ++o)
              values.push_back(evaluated.inside.lookupOrDefault(
                  yield->getOperand(o * arity + q)));
          scf::YieldOp::create(b, loc, values);
        },
        [&](OpBuilder &b, Location) {
          SmallVector<Value> values;
          for (int64_t q = 0; q != arity; ++q)
            for (Type type : types)
              values.push_back(createZero(b, loc, type));
          scf::YieldOp::create(b, loc, values);
        });
    computedOf.push_back(SmallVector<Value>(computed.getResults()));
  }

  // Every member takes its values from the lane of the member at place 0:
  // the loops whose destinations have the same types share the shuffles,
  // the lane at place 0 holding the values of the one loop it leads.
  SmallVector<SmallVector<Value>> mine(loops.size());
  SmallVector<bool> done(loops.size(), false);
  for (unsigned first = 0, e = loops.size(); first != e; ++first) {
    if (done[first])
      continue;
    SmallVector<Type> types = correctionTypes(loops[first]);
    SmallVector<unsigned> group;
    for (unsigned i = first; i != e; ++i)
      if (!done[i] && correctionTypes(loops[i]) == types) {
        group.push_back(i);
        done[i] = true;
      }
    int64_t arity = 0;
    for (unsigned i : group)
      arity = std::max<int64_t>(arity, loops[i].getArity());
    SmallVector<Value> received(arity * types.size());
    for (int64_t q = 0; q != arity; ++q)
      for (unsigned o = 0, n = types.size(); o != n; ++o) {
        Value held = createZero(builder, loc, types[o]);
        for (unsigned i : group)
          if (q < loops[i].getArity())
            held = arith::SelectOp::create(builder, loc, rows[i].useWarp,
                                           computedOf[i][q * n + o], held);
        received[q * n + o] = shuffleFrom(builder, held, leaderLane);
      }
    for (unsigned i : group) {
      Row &row = rows[i];
      for (unsigned o = 0, n = types.size(); o != n; ++o) {
        Value chosen = createZero(builder, loc, types[o]);
        for (int64_t q = 0, a = loops[i].getArity(); q != a; ++q) {
          Value here = arith::CmpIOp::create(
              builder, loc, arith::CmpIPredicate::eq, row.tuple.place,
              createIndex(builder, loc, q));
          chosen = arith::SelectOp::create(builder, loc, here,
                                           received[q * n + o], chosen);
        }
        mine[i].push_back(
            arith::SelectOp::create(builder, loc, row.useWarp, chosen,
                                    createZero(builder, loc, types[o])));
      }
    }
  }

  // A particle in a tuple that lies across warps is left to the second
  // kernel, and its member at place 0 goes on the list; every other
  // particle takes the loop after here.
  Value elsewhere, leadsAcross;
  for (Row &row : rows) {
    Value across = arith::AndIOp::create(
        builder, loc, row.inTuple,
        arith::XOrIOp::create(builder, loc, row.inWarp, True(builder)));
    elsewhere = elsewhere ? arith::OrIOp::create(builder, loc, elsewhere,
                                                 across)
                          : across;
    leadsAcross = leadsAcross ? arith::OrIOp::create(builder, loc,
                                                     leadsAcross,
                                                     row.leadsAcross)
                              : row.leadsAcross;
  }
  scf::IfOp::create(builder, loc, leadsAcross, [&](OpBuilder &b, Location) {
    Value address = arith::IndexCastOp::create(
        b, loc, b.getI64Type(),
        memref::ExtractAlignedPointerAsIndexOp::create(b, loc, acrossCount));
    Value pointer = LLVM::IntToPtrOp::create(
        b, loc, LLVM::LLVMPointerType::get(b.getContext(), 1), address);
    Value slot = LLVM::AtomicRMWOp::create(
        b, loc, LLVM::AtomicBinOp::add, pointer,
        arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(1)),
        LLVM::AtomicOrdering::monotonic, StringRef("device"));
    memref::StoreOp::create(
        b, loc, toI32(b, particle), acrossList,
        ValueRange{arith::IndexCastOp::create(b, loc, b.getIndexType(), slot)});
    scf::YieldOp::create(b, loc);
  });
  // A particle left to the second kernel writes its values of the loop
  // before, which that kernel takes.
  scf::IfOp::create(
      builder, loc, elsewhere,
      [&](OpBuilder &b, Location) {
        for (unsigned j = 0; j != beforeOuts; ++j)
          storeElement(b, loc, own[j], before.getOuts()[j], particle);
        scf::YieldOp::create(b, loc);
      },
      [&](OpBuilder &b, Location) {
        scf::IfOp::create(b, loc, valid, [&](OpBuilder &c, Location) {
          evalAfter(c, particle, own, mine);
          scf::YieldOp::create(c, loc);
        });
        scf::YieldOp::create(b, loc);
      });
}

void kernels::emitExclusionFilter(OpBuilder &builder, Location loc,
                                  Value counts, Value index, Value excluded,
                                  Value particle) {
  Type narrow = builder.getI32Type();
  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  Value width = memref::DimOp::create(builder, loc, index, one);
  int64_t entry = md_exec::getIncidenceEntrySize(2);

  Value found = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(),
      memref::LoadOp::create(builder, loc, counts, ValueRange{particle}));
  Value fits =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ule, found,
                            width);
  Value numExcluded = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(),
      memref::LoadOp::create(builder, loc, excluded,
                             ValueRange{particle, zero}));
  Value any = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                    numExcluded, zero);
  Value filter = arith::AndIOp::create(builder, loc, fits, any);

  scf::IfOp::create(builder, loc, filter, [&](OpBuilder &b, Location) {
    // The other member of each excluded pair of the particle: the member at
    // place 1 − s, for the place s of the particle.
    auto partnerOf = [&](OpBuilder &c, Value k) -> Value {
      Value offset =
          arith::MulIOp::create(c, loc, k, createIndex(c, loc, entry));
      Value base = arith::AddIOp::create(c, loc, offset, one);
      Value placeColumn =
          arith::AddIOp::create(c, loc, base, createIndex(c, loc, 1));
      Value place = memref::LoadOp::create(c, loc, excluded,
                                           ValueRange{particle, placeColumn});
      Value isFirst = arith::CmpIOp::create(
          c, loc, arith::CmpIPredicate::eq, place,
          arith::ConstantOp::create(c, loc, narrow, c.getI32IntegerAttr(0)));
      Value first =
          arith::AddIOp::create(c, loc, base, createIndex(c, loc, 2));
      Value second =
          arith::AddIOp::create(c, loc, base, createIndex(c, loc, 3));
      Value column = arith::SelectOp::create(c, loc, isFirst, second, first);
      return memref::LoadOp::create(c, loc, excluded,
                                    ValueRange{particle, column});
    };

    auto kept = scf::ForOp::create(
        b, loc, zero, found, one, ValueRange{zero},
        [&](OpBuilder &c, Location, Value r, ValueRange written) {
          Value neighbor =
              memref::LoadOp::create(c, loc, index, ValueRange{particle, r});
          auto search = scf::ForOp::create(
              c, loc, zero, numExcluded, one,
              ValueRange{arith::ConstantOp::create(c, loc, c.getI1Type(),
                                                   c.getBoolAttr(false))},
              [&](OpBuilder &d, Location, Value k, ValueRange isExcluded) {
                Value same =
                    arith::CmpIOp::create(d, loc, arith::CmpIPredicate::eq,
                                          partnerOf(d, k), neighbor);
                scf::YieldOp::create(
                    d, loc,
                    ValueRange{arith::OrIOp::create(d, loc, isExcluded[0],
                                                    same)});
              });
          Value keep = arith::XOrIOp::create(
              c, loc, search.getResult(0),
              arith::ConstantOp::create(c, loc, c.getI1Type(),
                                        c.getBoolAttr(true)));
          scf::IfOp::create(c, loc, keep, [&](OpBuilder &d, Location) {
            memref::StoreOp::create(d, loc, neighbor, index,
                                    ValueRange{particle, written[0]});
            scf::YieldOp::create(d, loc);
          });
          Value more = arith::AddIOp::create(c, loc, written[0], one);
          scf::YieldOp::create(
              c, loc,
              ValueRange{arith::SelectOp::create(c, loc, keep, more,
                                                 written[0])});
        });
    memref::StoreOp::create(
        b, loc, arith::IndexCastOp::create(b, loc, narrow, kept.getResult(0)),
        counts, ValueRange{particle});
    scf::YieldOp::create(b, loc);
  });
}

void kernels::lowerLookups(Operation *root) {
  SmallVector<md::LookupOp> lookups;
  root->walk([&](md::LookupOp op) { lookups.push_back(op); });
  for (md::LookupOp op : lookups) {
    OpBuilder builder(op);
    Location loc = op.getLoc();
    SmallVector<Value, 2> indices;
    for (Value index : op.getIndices())
      indices.push_back(index.getType().isIndex()
                            ? index
                            : arith::IndexCastOp::create(
                                  builder, loc, builder.getIndexType(), index)
                                  .getResult());
    Value value =
        memref::LoadOp::create(builder, loc, op.getTable(), indices);
    op.getResult().replaceAllUsesWith(convertReal(
        builder, loc, value, getElementTypeOrSelf(op.getResult().getType())));
    op.erase();
  }
}

Value kernels::emitRenumber(OpBuilder &builder, Location loc, Value members,
                            Value ids) {
  Type narrow = builder.getI32Type();
  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  auto toIndex = [&](OpBuilder &b, Value value) -> Value {
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), value);
  };
  Value count = memref::DimOp::create(builder, loc, ids, zero);
  Value place = memref::AllocOp::create(
      builder, loc, MemRefType::get({ShapedType::kDynamic}, narrow),
      ValueRange{count});
  scf::ForOp::create(
      builder, loc, zero, count, one, ValueRange(),
      [&](OpBuilder &b, Location, Value p, ValueRange) {
        Value id = memref::LoadOp::create(b, loc, ids, ValueRange{p});
        memref::StoreOp::create(
            b, loc, arith::IndexCastOp::create(b, loc, narrow, p), place,
            ValueRange{toIndex(b, id)});
        scf::YieldOp::create(b, loc);
      });

  auto type = cast<MemRefType>(members.getType());
  int64_t arity = type.getDimSize(1);
  Value numTuples = memref::DimOp::create(builder, loc, members, zero);
  Value result = memref::AllocOp::create(
      builder, loc, MemRefType::get(type.getShape(), narrow),
      ValueRange{numTuples});
  scf::ForOp::create(
      builder, loc, zero, numTuples, one, ValueRange(),
      [&](OpBuilder &b, Location, Value t, ValueRange) {
        for (int64_t q = 0; q != arity; ++q) {
          Value column = createIndex(b, loc, q);
          Value member =
              memref::LoadOp::create(b, loc, members, ValueRange{t, column});
          Value now =
              memref::LoadOp::create(b, loc, place, ValueRange{toIndex(b, member)});
          memref::StoreOp::create(b, loc, now, result, ValueRange{t, column});
        }
        scf::YieldOp::create(b, loc);
      });
  memref::DeallocOp::create(builder, loc, place);
  return result;
}

void kernels::freeAtEndOfBlock(Operation *op, Value buffer) {
  OpBuilder builder(op->getBlock()->getTerminator());
  memref::DeallocOp::create(builder, op->getLoc(), buffer);
}

Value kernels::emitBuildIncidence(OpBuilder &builder, Location loc,
                                  Value members, Value size) {
  Type narrow = builder.getI32Type();
  auto membersType = cast<MemRefType>(members.getType());
  int64_t arity = membersType.getDimSize(1);
  int64_t entry = md_exec::getIncidenceEntrySize(arity);
  Value zero = createIndex(builder, loc, 0);
  Value one = createIndex(builder, loc, 1);
  Value numTuples = memref::DimOp::create(builder, loc, members, zero);
  auto toIndex = [&](OpBuilder &b, Value value) -> Value {
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), value);
  };
  auto toNarrow = [&](OpBuilder &b, Value value) -> Value {
    return arith::IndexCastOp::create(b, loc, narrow, value);
  };

  // The number of tuples of each particle.
  Value counts = memref::AllocOp::create(
      builder, loc, MemRefType::get({ShapedType::kDynamic}, narrow),
      ValueRange{size});
  Value none = arith::ConstantOp::create(builder, loc, narrow,
                                         builder.getI32IntegerAttr(0));
  Value increment = arith::ConstantOp::create(builder, loc, narrow,
                                              builder.getI32IntegerAttr(1));
  scf::ForOp::create(builder, loc, zero, size, one, ValueRange(),
                     [&](OpBuilder &b, Location, Value i, ValueRange) {
                       memref::StoreOp::create(b, loc, none, counts,
                                               ValueRange{i});
                       scf::YieldOp::create(b, loc);
                     });
  scf::ForOp::create(
      builder, loc, zero, numTuples, one, ValueRange(),
      [&](OpBuilder &b, Location, Value t, ValueRange) {
        for (int64_t q = 0; q != arity; ++q) {
          Value member = toIndex(
              b, memref::LoadOp::create(b, loc, members,
                                        ValueRange{t, createIndex(b, loc, q)}));
          Value count =
              memref::LoadOp::create(b, loc, counts, ValueRange{member});
          Value more = arith::AddIOp::create(b, loc, count, increment);
          memref::StoreOp::create(b, loc, more, counts, ValueRange{member});
        }
        scf::YieldOp::create(b, loc);
      });

  // The width of a row.
  auto widest = scf::ForOp::create(
      builder, loc, zero, size, one, ValueRange{zero},
      [&](OpBuilder &b, Location, Value i, ValueRange largest) {
        Value count =
            toIndex(b, memref::LoadOp::create(b, loc, counts, ValueRange{i}));
        Value larger = arith::MaxUIOp::create(b, loc, count, largest[0]);
        scf::YieldOp::create(b, loc, larger);
      });
  Value slots = arith::MulIOp::create(builder, loc, widest.getResult(0),
                                      createIndex(builder, loc, entry));
  Value width = arith::AddIOp::create(builder, loc, slots, one);
  Value incidence = memref::AllocOp::create(
      builder, loc,
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, narrow),
      ValueRange{size, width});

  // The rows, in the order of the tuples.
  scf::ForOp::create(builder, loc, zero, size, one, ValueRange(),
                     [&](OpBuilder &b, Location, Value i, ValueRange) {
                       memref::StoreOp::create(b, loc, none, incidence,
                                               ValueRange{i, zero});
                       scf::YieldOp::create(b, loc);
                     });
  scf::ForOp::create(
      builder, loc, zero, numTuples, one, ValueRange(),
      [&](OpBuilder &b, Location, Value t, ValueRange) {
        SmallVector<Value, 4> tupleMembers;
        for (int64_t q = 0; q != arity; ++q)
          tupleMembers.push_back(memref::LoadOp::create(
              b, loc, members, ValueRange{t, createIndex(b, loc, q)}));
        for (int64_t s = 0; s != arity; ++s) {
          Value particle = toIndex(b, tupleMembers[s]);
          Value filled = memref::LoadOp::create(b, loc, incidence,
                                                ValueRange{particle, zero});
          Value offset = arith::MulIOp::create(
              b, loc, toIndex(b, filled), createIndex(b, loc, entry));
          Value base = arith::AddIOp::create(b, loc, offset, one);
          auto store = [&](Value value, int64_t c) {
            Value column =
                arith::AddIOp::create(b, loc, base, createIndex(b, loc, c));
            memref::StoreOp::create(b, loc, value, incidence,
                                    ValueRange{particle, column});
          };
          store(toNarrow(b, t), 0);
          store(arith::ConstantOp::create(b, loc, narrow,
                                          b.getI32IntegerAttr(s)),
                1);
          for (int64_t q = 0; q != arity; ++q)
            store(tupleMembers[q], 2 + q);
          Value more = arith::AddIOp::create(b, loc, filled, increment);
          memref::StoreOp::create(b, loc, more, incidence,
                                  ValueRange{particle, zero});
        }
        scf::YieldOp::create(b, loc);
      });

  memref::DeallocOp::create(builder, loc, counts);
  return incidence;
}

//===----------------------------------------------------------------------===//
// Templates
//===----------------------------------------------------------------------===//

std::string kernels::getInstanceName(StringRef name, Type real) {
  return real.isF64() ? name.str() : (name + "_f32").str();
}

static std::string getPMESuffix(Type position, Type charge, Type force,
                                int64_t order) {
  auto bits = [](Type type) {
    return std::to_string(type.getIntOrFloatBitWidth());
  };
  return "_p" + bits(position) + "c" + bits(charge) + "f" + bits(force) +
         "o" + std::to_string(order);
}

std::string kernels::getPMEInstanceName(StringRef name, Type position,
                                        Type charge, Type force,
                                        int64_t order) {
  return (name + getPMESuffix(position, charge, force, order)).str();
}

std::string kernels::instantiatePMETemplates(StringRef text, Type position,
                                             Type charge, Type force,
                                             int64_t order) {
  auto spell = [](Type type) { return type.isF64() ? "f64" : "f32"; };
  // The conversion from and to f64 of a type that is f64 is a cast of the
  // bits, which changes nothing.
  auto extend = [](Type type) {
    return type.isF64() ? "arith.bitcast" : "arith.extf";
  };
  // The conversion between two of f32 and f64.
  auto convert = [](Type from, Type to) -> const char * {
    if (from == to)
      return "arith.bitcast";
    return from.isF64() ? "arith.truncf" : "arith.extf";
  };
  std::string suffix = getPMESuffix(position, charge, force, order);
  std::string instance;
  while (!text.empty()) {
    if (text.consume_front("!pme_pos = f64")) {
      instance += "!pme_pos = " + std::string(spell(position));
    } else if (text.consume_front("!pme_chg = f64")) {
      instance += "!pme_chg = " + std::string(spell(charge));
    } else if (text.consume_front("!pme_frc = f64")) {
      instance += "!pme_frc = " + std::string(spell(force));
    } else if (text.consume_front("PME_EXTEND_POS")) {
      instance += extend(position);
    } else if (text.consume_front("PME_EXTEND_CHG")) {
      instance += extend(charge);
    } else if (text.consume_front("PME_NARROW_FRC")) {
      instance += force.isF64() ? "arith.bitcast" : "arith.truncf";
    } else if (text.consume_front("!pme_real = f64")) {
      instance += "!pme_real = " + std::string(spell(force));
    } else if (text.consume_front("PME_CHG_TO_REAL")) {
      instance += convert(charge, force);
    } else if (text.consume_front("PME_F64_TO_REAL")) {
      instance += convert(Float64Type::get(force.getContext()), force);
    } else if (text.consume_front("PME_REAL_TO_F64")) {
      instance += convert(force, Float64Type::get(force.getContext()));
    } else if (text.consume_front("PME_ATOMIC_ADD %pointer, %value")) {
      // An atomic addition of `!pme_real` to global memory at the scope of
      // the device, with no result. The NVPTX backend of LLVM expands
      // `atomicrmw fadd` of f32 into a loop of compare-and-swap, at four
      // times the time of PTX's own reduction on an RTX 3090, so f32 takes
      // that (NVIDIA only, as the template is); f64 is native.
      if (force.isF64())
        instance += "%old = llvm.atomicrmw fadd %pointer, %value "
                    "syncscope(\"device\") monotonic : !llvm.ptr<1>, f64";
      else
        instance += "llvm.inline_asm has_side_effects "
                    "\"red.relaxed.gpu.global.add.f32 [$0], $1;\", \"l,f\" "
                    "%pointer, %value : (!llvm.ptr<1>, f32) -> ()";
    } else if (text.consume_front("PME_REAL_BYTES")) {
      // The bytes of a value of `!pme_real`, the type of the forces.
      instance += force.isF64() ? "8" : "4";
    } else if (text.consume_front("PME_LANES")) {
      // The threads of a particle in the gather: the least power of 2 not
      // below the order, so that they are aligned within a warp.
      instance += order <= 4 ? "4" : "8";
    } else if (text.consume_front("PME_ORDER")) {
      // The order of the splines, a constant in the kernels.
      instance += std::to_string(order);
    } else if (text.consume_front("PME_REAL_TO_FRC")) {
      instance += "arith.bitcast";
    } else if (text.consume_front("@mdrt")) {
      // The functions of the template, `@mdrt.` or `@mdrt_`; the functions
      // of the runtime, such as `@mdrtCudaFFTForward3D`, keep their names.
      StringRef name = text.take_while([](char c) {
        return llvm::isAlnum(c) || c == '_' || c == '.';
      });
      text = text.drop_front(name.size());
      bool local = name.starts_with(".") || name.starts_with("_");
      instance += "@mdrt" + name.str() + (local ? suffix : "");
    } else {
      instance += text.front();
      text = text.drop_front();
    }
  }
  return instance;
}

std::string kernels::instantiateTemplates(StringRef text, Type real) {
  if (real.isF64())
    return text.str();

  // The functions of the templates have names that begin with `mdrt`. A
  // conversion from f64 to f32 becomes one from f32 to f32, which is a
  // cast of the bits.
  std::string instance;
  StringRef prefix = "@mdrt";
  while (!text.empty()) {
    if (text.consume_front("f64")) {
      instance += "f32";
    } else if (text.consume_front("arith.truncf")) {
      instance += "arith.bitcast";
    } else if (text.consume_front(prefix)) {
      StringRef name = text.take_while([](char c) {
        return llvm::isAlnum(c) || c == '_' || c == '.';
      });
      text = text.drop_front(name.size());
      instance += getInstanceName((prefix + name).str(), real);
    } else {
      instance += text.front();
      text = text.drop_front();
    }
  }
  return instance;
}
