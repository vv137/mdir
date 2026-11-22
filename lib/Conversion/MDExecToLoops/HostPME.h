// Builders for the serial PME stages [Essmann1995, Liang2026].
#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mdir::kernels {
using namespace mlir;
class HostPME {
public:
  HostPME(ModuleOp module, Type position, Type charge, Type force)
      : module(module), b(module.getContext()), loc(b.getUnknownLoc()),
        positionType(position), chargeType(charge), realType(force) {
    b.setInsertionPointToEnd(module.getBody());
  }
  void emit(bool potential, bool listed);

private:
  ModuleOp module;
  OpBuilder b;
  Location loc;
  Type positionType, chargeType, realType;
  Type idxType = b.getIndexType(), i32Type = b.getI32Type(),
       i64Type = b.getI64Type(), boolType = b.getI1Type(),
       f64Type = b.getF64Type();

  func::FuncOp declare(StringRef name, FunctionType type) {
    auto fn = func::FuncOp::create(b, loc, name, type);
    fn.setPrivate();
    return fn;
  }
  Value convert(Type type, Value input) {
    if (type == input.getType())
      return arith::BitcastOp::create(b, loc, type, input);
    auto width = [](Type t) {
      return getElementTypeOrSelf(t).getIntOrFloatBitWidth();
    };
    if (width(type) > width(input.getType()))
      return arith::ExtFOp::create(b, loc, type, input);
    return arith::TruncFOp::create(b, loc, type, input);
  }
  Value load(Value mem, ValueRange at) {
    return memref::LoadOp::create(b, loc, mem, at, BoolAttr(), IntegerAttr(),
                                  BoolAttr());
  }
  void store(Value value, Value mem, ValueRange at) {
    memref::StoreOp::create(b, loc, value, mem, at, BoolAttr(), IntegerAttr());
  }
  DenseElementsAttr dense(VectorType type, ArrayRef<double> values) {
    SmallVector<Attribute> elements;
    for (double value : values)
      elements.push_back(b.getFloatAttr(type.getElementType(), value));
    return DenseElementsAttr::get(type, elements);
  }
  void sumReductions(ValueRange values) {
    auto reduce = scf::ReduceOp::create(b, loc, values);
    OpBuilder::InsertionGuard guard(b);
    for (Region &region : reduce->getRegions()) {
      Block &block = region.front();
      b.setInsertionPointToEnd(&block);
      Value sum = arith::AddFOp::create(b, loc, block.getArgument(0),
                                        block.getArgument(1));
      scf::ReduceReturnOp::create(b, loc, sum);
    }
  }
  void emitRaise();
  void emitBspline();
  void emitWindow();
  void emitClenshaw();
  void emitSpline();
  void emitPlace();
  void emitInverseCell();
  void emitZero();
  void emitSpreadPoint();
  void emitSpreadOne();
  void emitSpreadOneTriclinic();
  void emitSpread();
  void emitSpreadTriclinic();
  void emitReal();
  void emitWave();
  void emitTables();
  void emitInfluence();
  void emitPointSums();
  void emitPointSumsP3m();
  void emitConvolveRow();
  void emitConvolveRowTriclinic();
  void emitConvolve();
  void emitScale();
  void emitConvolveTriclinic();
  void emitScaleTriclinic();
  void emitGatherSums();
  void emitPotentialSum();
  void emitGatherOne();
  void emitGather();
  void emitPotentialOne();
  void emitPotential();
  void emitSplinesTriclinic();
  void emitGatherOneTriclinic();
  void emitGatherTriclinic();
  void emitPotentialOneTriclinic();
  void emitPotentialTriclinic();
  void emitCountListed();
  void emitSpreadListed();
  void emitGatherListed();
  void emitSpreadListedTriclinic();
  void emitGatherListedTriclinic();
};
} // namespace mdir::kernels
