// What the lowerings of md_exec have in common.

#include "mdir/Conversion/MDExecKernels.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mdir/Dialect/MD/MDOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
                                           SmallVectorImpl<Value> *outTotals) {
  Location loc = op.getLoc();
  Value positions = op.getPositions();
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
  for (Value buffer : op.getIns())
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
  SmallVector<Value> sums;
  for (Value value : yield->getOperands())
    sums.push_back(createZero(builder, loc, value.getType()));

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
                     loadElement(pair, loc, op.getIns()[i], other));
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
          Value contribution = inside.lookupOrDefault(yield->getOperand(i));
          Value nothing = createZero(pair, loc, contribution.getType());
          Value masked = arith::SelectOp::create(pair, loc, within,
                                                 contribution, nothing);
          updated.push_back(
              arith::AddFOp::create(pair, loc, partial[i], masked));
        }
        scf::YieldOp::create(pair, loc, updated);
      });

  SmallVector<Value> totals(inner.getResults());
  if (lanes)
    for (Value &total : totals)
      total = lanes->combine(builder, loc, total);

  auto emitStores = [&](OpBuilder &writer) {
    for (unsigned i = 0; i != numOuts; ++i) {
      Value destination = op.getOuts()[i];
      Value total = totals[i];
      if (!op.overwrites(i))
        total = arith::AddFOp::create(
            writer, loc, loadElement(writer, loc, destination, central),
            total);
      storeElement(writer, loc, total, destination, central);
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

static void evaluateMembers(OpBuilder &b, md_exec::TupleForOp op,
                            EvaluatedTuple &result, Value box, Value inverse,
                            const IRMapping &local);

/// Loads the entry `number` of the row of `particle` in `incidence` and
/// emits the kernel of `op` for it: the displacements in the minimum image,
/// taken in the type of the positions and imaged in `computed`, the values
/// of `ins` for each member and of `parameters` for the tuple.
static EvaluatedTuple evaluateTuple(OpBuilder &b, md_exec::TupleForOp op,
                                    Value incidence, Value particle,
                                    Value number, Value box, Value inverse,
                                    const IRMapping &local) {
  Location loc = op.getLoc();
  Block &kernel = op.getKernel().front();
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
  evaluateMembers(b, op, result, box, inverse, local);
  return result;
}

/// Emits the kernel of `op` for the tuple `result.tuple` of the members
/// `result.members`, as `evaluateTuple` does.
static void evaluateMembers(OpBuilder &b, md_exec::TupleForOp op,
                            EvaluatedTuple &result, Value box, Value inverse,
                            const IRMapping &local) {
  Location loc = op.getLoc();
  Block &kernel = op.getKernel().front();
  int64_t arity = op.getArity();
  IRMapping &inside = result.inside;
  inside = local;
  Value positions = op.getPositions();
  SmallVector<Value, 4> memberPositions(arity);
  auto positionOf = [&](int64_t q) {
    if (!memberPositions[q])
      memberPositions[q] = loadElement(b, loc, positions, result.members[q]);
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
  unsigned argument = op.getCoordinateKinds().size();
  for (Value buffer : op.getIns())
    for (int64_t q = 0; q != arity; ++q)
      inside.map(kernel.getArgument(argument++),
                 loadElement(b, loc, buffer, result.members[q]));
  for (Value buffer : op.getParameters())
    inside.map(kernel.getArgument(argument++),
               loadElement(b, loc, buffer, result.tuple));
  for (Operation &nested : kernel.without_terminator())
    b.clone(nested, inside);
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
  for (unsigned i = numOuts * arity; i != numYields; ++i)
    sums.push_back(createZero(builder, loc, yield->getOperand(i).getType()));

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
          Value contribution = inside.lookupOrDefault(yield->getOperand(i));
          Value nothing = createZero(b, loc, contribution.getType());
          Value masked =
              arith::SelectOp::create(b, loc, atFirst, contribution, nothing);
          updated.push_back(arith::AddFOp::create(b, loc, partial[j], masked));
        }
        scf::YieldOp::create(b, loc, updated);
      });

  SmallVector<Value> totals(loop.getResults());
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
    op.getResult().replaceAllUsesWith(
        convertReal(builder, loc, value, op.getResult().getType()));
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
