// Builders for the serial PME stages [Essmann1995, Liang2026].
#pragma once

#include "../MDExecKernels/PMECode.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mdir::kernels {
using namespace mlir;
class HostPME : public detail::PMECode {
public:
  HostPME(OpBuilder &builder, ModuleOp module, Type position, Type charge,
          Type force)
      : PMECode(builder, position, charge, force), module(module) {
    b.setInsertionPointToEnd(module.getBody());
  }
  void emit(bool potential, bool listed);

private:
  ModuleOp module;
  Type boolType = b.getI1Type();
  func::FuncOp declare(StringRef name, FunctionType type) {
    auto fn = func::FuncOp::create(b, loc, name, type);
    fn.setPrivate();
    return fn;
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
  enum class Cell { Orthorhombic, Triclinic };
  enum class ParticleStage { Spread, Gather, Potential };
  template <ParticleStage stage, Cell cell, bool listed = false>
  void emitParticles();
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
  void emitReal();
  void emitWave();
  void emitTables();
  void emitInfluence();
  void emitPointSums();
  void emitPointSumsP3m();
  void emitConvolveRow();
  void emitConvolveRowTriclinic();
  template <Cell cell, bool observables> void emitConvolve();
  void emitGatherSums();
  void emitPotentialSum();
  void emitGatherOne();
  void emitPotentialOne();
  void emitSplinesTriclinic();
  void emitGatherOneTriclinic();
  void emitPotentialOneTriclinic();
  void emitCountListed();
};
} // namespace mdir::kernels
