// What the lowerings of md_exec have in common.

#include "mdir/Conversion/MDExecKernels.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mdir;

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
                                           IRMapping &local) {
  Location loc = op.getLoc();
  Value positions = op.getPositions();
  Block &kernel = op.getKernel().front();
  Operation *yield = kernel.getTerminator();
  unsigned numIns = op.getIns().size();
  unsigned numOuts = op.getOuts().size();
  unsigned numYields = yield->getNumOperands();

  // The displacement is computed in the type of the positions and then
  // converted to the type that the kernel computes in: the subtraction is
  // the step that loses precision.
  Type real = cast<MemRefType>(positions.getType()).getElementType();
  Type computed = kernel.getArgument(0).getType();
  double cutoff = op.getCutoff().convertToDouble();
  Value cutoff2 = createReal(builder, loc, real, cutoff * cutoff);
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

  // The contributions of the neighbors of one particle are summed in the
  // order of its list.
  SmallVector<Value> sums;
  for (Value value : yield->getOperands())
    sums.push_back(createZero(builder, loc, value.getType()));

  auto inner = scf::ForOp::create(
      builder, loc, zero, end, one, sums,
      [&](OpBuilder &pair, Location, Value entry, ValueRange partial) {
        Value narrow = memref::LoadOp::create(pair, loc, index,
                                              ValueRange{central, entry});
        Value other = arith::IndexCastOp::create(pair, loc,
                                                 pair.getIndexType(), narrow);

        // The minimum-image displacement and its squared length.
        Value otherPosition = loadElement(pair, loc, positions, other);
        Value raw =
            arith::SubFOp::create(pair, loc, centralPosition, otherPosition);
        Value images = arith::MulFOp::create(pair, loc, raw, inverse);
        Value nearest = math::RoundEvenOp::create(pair, loc, images);
        Value shift = arith::MulFOp::create(pair, loc, nearest, box);
        Value d = arith::SubFOp::create(pair, loc, raw, shift);
        Value squares = arith::MulFOp::create(pair, loc, d, d);
        Value r2 = vector::ReductionOp::create(
            pair, loc, vector::CombiningKind::ADD, squares);

        IRMapping inside = local;
        inside.map(kernel.getArgument(0),
                   convertReal(pair, loc, r2, computed));
        inside.map(kernel.getArgument(1),
                   convertReal(pair, loc, d, computed));
        for (unsigned i = 0; i != numIns; ++i) {
          inside.map(kernel.getArgument(2 + 2 * i), centralValues[i]);
          inside.map(kernel.getArgument(3 + 2 * i),
                     loadElement(pair, loc, op.getIns()[i], other));
        }
        for (Operation &nested : kernel.without_terminator())
          pair.clone(nested, inside);

        // A pair beyond the cutoff contributes nothing.
        Value within = arith::CmpFOp::create(
            pair, loc, arith::CmpFPredicate::OLT, r2, cutoff2);
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

  for (unsigned i = 0; i != numOuts; ++i) {
    Value destination = op.getOuts()[i];
    Value total = inner.getResult(i);
    if (!op.overwrites(i))
      total = arith::AddFOp::create(
          builder, loc, loadElement(builder, loc, destination, central),
          total);
    storeElement(builder, loc, total, destination, central);
  }

  SmallVector<Value> contributions;
  for (unsigned i = numOuts; i != numYields; ++i) {
    Value total = inner.getResult(i);
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
// Templates
//===----------------------------------------------------------------------===//

std::string kernels::getInstanceName(StringRef name, Type real) {
  return real.isF64() ? name.str() : (name + "_f32").str();
}

std::string kernels::instantiateTemplates(StringRef text, Type real) {
  if (real.isF64())
    return text.str();

  // The functions of the templates have names that begin with `mdrt`.
  std::string instance;
  StringRef prefix = "@mdrt";
  while (!text.empty()) {
    if (text.consume_front("f64")) {
      instance += "f32";
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
