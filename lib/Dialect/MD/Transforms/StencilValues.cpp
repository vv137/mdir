// Explicit reference traversal of a structured particle-grid relation.
#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mdir/Dialect/MD/Transforms/StencilGeometry.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDRT/MDRTOps.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include <limits>

using namespace mlir;
using namespace mdir::md;

namespace mdir::md {
#define GEN_PASS_DEF_EXPANDSTENCILREDUCTIONS
#define GEN_PASS_DEF_MATERIALIZESTENCILVALUES
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"
namespace {
Value index(OpBuilder &b, Location loc, int64_t n) {
  return arith::ConstantIndexOp::create(b, loc, n);
}
Value zero(OpBuilder &b, Location loc, Type type) {
  return arith::ConstantOp::create(b, loc, type,
                                  cast<TypedAttr>(b.getZeroAttr(type)));
}
MemRefType fieldBuffer(FieldType field) {
  SmallVector<int64_t> shape{ShapedType::kDynamic};
  if (field.getNumComponents() != 1)
    shape.push_back(field.getNumComponents());
  return MemRefType::get(shape, field.getElementType());
}
Value view(OpBuilder &b, Location loc, Value field) {
  return mdir::mdrt::ToBufferOp::create(b, loc,
      fieldBuffer(cast<FieldType>(field.getType())), field);
}
Value load(OpBuilder &b, Location loc, Value buffer, Value row, Type element) {
  if (element.isF64())
    return memref::LoadOp::create(b, loc, buffer, row);
  auto vector = cast<VectorType>(element);
  SmallVector<Value> values;
  for (int64_t i = 0; i != vector.getNumElements(); ++i)
    values.push_back(memref::LoadOp::create(b, loc, buffer,
        ValueRange{row, index(b, loc, i)}));
  return vector::FromElementsOp::create(b, loc, vector, values);
}
void store(OpBuilder &b, Location loc, Value value, Value buffer, Value row) {
  if (value.getType().isF64()) {
    memref::StoreOp::create(b, loc, value, buffer, row);
    return;
  }
  auto vector = cast<VectorType>(value.getType());
  for (int64_t i = 0; i != vector.getNumElements(); ++i) {
    Value scalar = vector::ExtractOp::create(b, loc, value, i);
    memref::StoreOp::create(b, loc, scalar, buffer,
        ValueRange{row, index(b, loc, i)});
  }
}

class ExpandStencilReductions
    : public impl::ExpandStencilReductionsBase<ExpandStencilReductions> {
  void runOnOperation() final {
    SmallVector<Operation *> operations;
    getOperation()->walk([&](Operation *op) {
      if (isa<ScatterStencilOp, GatherStencilOp>(op))
        operations.push_back(op);
    });
    for (Operation *source : operations) {
      OpBuilder b(source);
      Location loc = source->getLoc();
      bool scatter = isa<ScatterStencilOp>(source);
      auto reduced = ReduceStencilOp::create(b, loc, source->getResult(0).getType(),
          source->getOperand(0), source->getOperand(1), source->getOperand(2),
          scatter ? ValueRange{source->getOperand(3)} : ValueRange(),
          scatter ? ValueRange() : ValueRange{source->getOperand(3)},
          ValueRange(), b.getStringAttr(scatter ? "grid" : "particles"));
      Block *block = new Block();
      reduced.getKernel().push_back(block);
      block->addArgument(VectorType::get({3}, b.getF64Type()), loc);
      block->addArgument(VectorType::get({3}, b.getF64Type()), loc);
      block->addArgument(VectorType::get({9}, b.getF64Type()), loc);
      block->addArgument(b.getF64Type(), loc);
      OpBuilder kernel = OpBuilder::atBlockEnd(block);
      IRMapping mapping;
      Block &old = source->getRegion(0).front();
      mapping.map(old.getArgument(0), block->getArgument(0));
      for (Operation &inner : old.without_terminator())
        kernel.clone(inner, mapping);
      Value weight = mapping.lookupOrDefault(old.getTerminator()->getOperand(0));
      Value value = arith::MulFOp::create(kernel, loc, weight, block->getArgument(3));
      YieldOp::create(kernel, loc, ValueRange{value});
      source->getResult(0).replaceAllUsesWith(reduced.getResult());
      source->erase();
    }
  }
};

class MaterializeStencilValues
    : public impl::MaterializeStencilValuesBase<MaterializeStencilValues> {
  void runOnOperation() final {
    SmallVector<CheckedRectangularCellOp> checks;
    getOperation()->walk([&](CheckedRectangularCellOp op) { checks.push_back(op); });
    for (auto op : checks) {
      if (!op->getParentOfType<func::FuncOp>()) {
        op.emitOpError("materialize rectangular preconditions after md-inline");
        signalPassFailure();
        return;
      }
      OpBuilder b(op);
      Location loc = op.getLoc();
      Value matrix = emitStencilCellMatrix(b, loc, op.getInput());
      ScalarEmitter e(b, loc);
      Value valid;
      for (int64_t i = 0; i != 9; ++i) {
        Value h = vector::ExtractOp::create(b, loc, matrix, i);
        Value condition = arith::CmpFOp::create(b, loc,
            i % 4 == 0 ? arith::CmpFPredicate::OGT : arith::CmpFPredicate::OEQ,
            h, e.constant(0, b.getF64Type()));
        if (i % 4 == 0) {
          Value finite = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
              h, e.constant(std::numeric_limits<double>::max(), b.getF64Type()));
          condition = arith::AndIOp::create(b, loc, condition, finite);
        }
        valid = valid ? Value(arith::AndIOp::create(b, loc, valid, condition))
                      : condition;
      }
      cf::AssertOp::create(b, loc, valid,
          "selected Gaussian reciprocal chain requires a finite positive rectangular cell");
      op.replaceAllUsesWith(op.getInput());
      op.erase();
    }
    SmallVector<ReduceStencilOp> operations;
    getOperation()->walk([&](ReduceStencilOp op) { operations.push_back(op); });
    for (ReduceStencilOp op : operations) {
      if (!op->getParentOfType<func::FuncOp>()) {
        op.emitOpError("materialize stencil values after semantic differentiation and md-inline");
        signalPassFailure();
        return;
      }
      materialize(op);
    }
    // The relation is indices-only; no storage for its active payload exists.
    SmallVector<GridStencilOp> relations;
    getOperation()->walk([&](GridStencilOp op) {
      if (op.getResult().use_empty()) relations.push_back(op);
    });
    for (auto relation : relations)
      relation.erase();
  }

  void materialize(ReduceStencilOp op) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto stencil = cast<GridStencilType>(op.getStencil().getType());
    auto shape = stencil.getShape().asArrayRef();
    int64_t order = stencil.getOrder();
    Value positions = view(b, loc, op.getPositions());
    SmallVector<Value> fields;
    for (Value field : op.getParticles())
      fields.push_back(view(b, loc, field));
    Value matrix = emitStencilCellMatrix(b, loc, op.getCell());
    auto inverse = emitStencilCellInverse(b, loc, matrix);
    ScalarEmitter e(b, loc);
    Value positive = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT,
        inverse.determinant, e.constant(0, b.getF64Type()));
    Value finite = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT,
        inverse.determinant, e.constant(std::numeric_limits<double>::max(), b.getF64Type()));
    cf::AssertOp::create(b, loc, arith::AndIOp::create(b, loc, positive, finite),
                        "stencil requires a finite positive nonsingular periodic cell");
    Type resultType = op.getResult().getType();
    bool grid = op.getDestination() == "grid";
    bool particles = op.getDestination() == "particles";
    Type element = resultType;
    Value output;
    if (grid) {
      auto tensor = cast<RankedTensorType>(resultType);
      element = tensor.getElementType();
      output = tensor::GenerateOp::create(b, loc, tensor, ValueRange(),
          [&](OpBuilder &inner, Location where, ValueRange) {
            tensor::YieldOp::create(inner, where, zero(inner, where, element));
          });
    } else if (particles) {
      auto field = cast<FieldType>(resultType);
      element = field.getKernelValueType();
      Value count = memref::DimOp::create(b, loc, positions, 0);
      output = memref::AllocOp::create(b, loc, fieldBuffer(field), ValueRange{count});
    }
    Value count = memref::DimOp::create(b, loc, positions, 0);
    Value initial = grid ? output : zero(b, loc, element);
    auto outer = scf::ForOp::create(b, loc, index(b, loc, 0), count,
        index(b, loc, 1), ValueRange{initial},
        [&](OpBuilder &at, Location where, Value row, ValueRange state) {
          auto v3 = VectorType::get({3}, at.getF64Type());
          Value position = load(at, where, positions, row, v3);
          SmallVector<Value> scaled, bases, particleValues;
          ScalarEmitter emit(at, where);
          for (int a = 0; a != 3; ++a) {
            Value fraction;
            for (int c = 0; c != 3; ++c) {
              Value x = vector::ExtractOp::create(at, where, position, c);
              fraction = emit.add(fraction, emit.mul(x, inverse.entries[3*c+a]));
            }
            Value turns = math::FloorOp::create(at, where, fraction);
            Value wrapped = emit.sub(fraction, turns);
            Value u = emit.scale(shape[a], wrapped);
            Value lo = arith::CmpFOp::create(at, where, arith::CmpFPredicate::OGE,
                u, emit.constant(0, at.getF64Type()));
            Value hi = arith::CmpFOp::create(at, where, arith::CmpFPredicate::OLE,
                u, emit.constant(shape[a], at.getF64Type()));
            cf::AssertOp::create(at, where, arith::AndIOp::create(at, where, lo, hi),
                                "stencil position has nonfinite fractional coordinates");
            Value floor = math::FloorOp::create(at, where, u);
            Value integer = arith::FPToSIOp::create(at, where, at.getI64Type(), floor);
            Value shift = arith::ConstantIntOp::create(at, where, order-1, 64);
            bases.push_back(arith::SubIOp::create(at, where, integer, shift));
            scaled.push_back(u);
          }
          for (auto [buffer, field] : llvm::zip(fields, op.getParticles()))
            particleValues.push_back(load(at, where, buffer, row,
                cast<FieldType>(field.getType()).getKernelValueType()));
          Value start = grid || !particles ? state[0] : zero(at, where, element);
          auto entries = scf::ForOp::create(at, where, index(at, where, 0),
              index(at, where, order*order*order), index(at, where, 1),
              ValueRange{start},
              [&](OpBuilder &inner, Location entryLoc, Value flat, ValueRange sums) {
                SmallVector<Value> xyz, distances;
                Value offsets[3];
                offsets[0] = arith::DivUIOp::create(inner, entryLoc, flat,
                                                   index(inner, entryLoc, order*order));
                Value rest = arith::RemUIOp::create(inner, entryLoc, flat,
                                                   index(inner, entryLoc, order*order));
                offsets[1] = arith::DivUIOp::create(inner, entryLoc, rest,
                                                   index(inner, entryLoc, order));
                offsets[2] = arith::RemUIOp::create(inner, entryLoc, rest,
                                                   index(inner, entryLoc, order));
                ScalarEmitter scalar(inner, entryLoc);
                for (int a = 0; a != 3; ++a) {
                  Value offset = arith::IndexCastOp::create(inner, entryLoc,
                      inner.getI64Type(), offsets[a]);
                  Value unwrapped = arith::AddIOp::create(inner, entryLoc, bases[a], offset);
                  Value real = arith::SIToFPOp::create(inner, entryLoc,
                      inner.getF64Type(), unwrapped);
                  distances.push_back(scalar.sub(scaled[a], real));
                  Value extent = arith::ConstantIntOp::create(inner, entryLoc, shape[a], 64);
                  Value remainder = arith::RemSIOp::create(inner, entryLoc, unwrapped, extent);
                  Value positive = arith::AddIOp::create(inner, entryLoc, remainder, extent);
                  Value wrapped = arith::RemSIOp::create(inner, entryLoc, positive, extent);
                  xyz.push_back(arith::IndexCastOp::create(inner, entryLoc,
                      inner.getIndexType(), wrapped));
                }
                Block &kernel = op.getKernel().front();
                IRMapping mapping;
                mapping.map(kernel.getArgument(0), vector::FromElementsOp::create(
                    inner, entryLoc, v3, distances));
                mapping.map(kernel.getArgument(1), position);
                mapping.map(kernel.getArgument(2), matrix);
                unsigned argument = 3;
                for (Value value : particleValues)
                  mapping.map(kernel.getArgument(argument++), value);
                for (Value value : op.getGrids())
                  mapping.map(kernel.getArgument(argument++),
                      tensor::ExtractOp::create(inner, entryLoc, value, xyz));
                for (Value value : op.getUniforms())
                  mapping.map(kernel.getArgument(argument++), value);
                for (Operation &nested : kernel.without_terminator())
                  inner.clone(nested, mapping);
                Value value = mapping.lookupOrDefault(kernel.getTerminator()->getOperand(0));
                Value previous = grid ? Value(tensor::ExtractOp::create(
                    inner, entryLoc, sums[0], xyz)) : sums[0];
                Value next = arith::AddFOp::create(inner, entryLoc, previous, value);
                if (grid)
                  next = tensor::InsertOp::create(inner, entryLoc, next, sums[0], xyz);
                scf::YieldOp::create(inner, entryLoc, next);
              });
          if (particles)
            store(at, where, entries.getResult(0), output, row);
          scf::YieldOp::create(at, where, entries.getResult(0));
        });
    Value result = particles ? Value(mdir::mdrt::FromBufferOp::create(
        b, loc, resultType, output)) : outer.getResult(0);
    op.replaceAllUsesWith(result);
    op.erase();
  }
};
}
}
