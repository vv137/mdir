// Stochastic cell rescaling [Bernetti2020] at the original insertion point.
#include "BarostatIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace mdir::driver::builder;
namespace {
class Kernel {
public:
  explicit Kernel(OpBuilder &builder)
      : builder(builder), loc(builder.getUnknownLoc()),
        vec(VectorType::get({3}, builder.getF64Type())) {}
  Value add(Value a, Value b) { return arith::AddFOp::create(builder, loc, a, b); }
  Value mul(Value a, Value b) { return arith::MulFOp::create(builder, loc, a, b); }
  Value sub(Value a, Value b) { return arith::SubFOp::create(builder, loc, a, b); }
  Value sum(Value a) { return vector::ReductionOp::create(builder, loc, vector::CombiningKind::ADD, a); }
  Value div(Value a, Value b) { return arith::DivFOp::create(builder, loc, a, b); }
  Value broadcast(Value a) { return vector::BroadcastOp::create(builder, loc, vec, a); }
  Value extract(Value a, int64_t slot) { return vector::ExtractOp::create(builder, loc, a, slot); }
  Value elements(ValueRange values) { return vector::FromElementsOp::create(builder, loc, vec, values); }
  Value call(StringRef name, ValueRange values) {
    return func::CallOp::create(builder, loc, name, TypeRange{builder.getF64Type()}, values).getResult(0);
  }
private:
  OpBuilder &builder;
  Location loc;
  VectorType vec;
};
} // namespace
StrainIRResult mdir::driver::builder::buildBarostatStrain(
    OpBuilder &builder, const StrainIRInputs &in, StrainCoupling coupling) {
  auto loc = builder.getUnknownLoc();
  Kernel k(builder);
  StrainIRResult result;
  SmallVector<Value> edges;
  for (Value index : in.edgeIndices)
    edges.push_back(memref::LoadOp::create(builder, loc, in.boxMemory, ValueRange{index}));
  Value xy = k.mul(edges[0], edges[1]);
  result.volume = k.mul(xy, edges[2]);
  Value twiceKinetic = k.add(in.kinetic, in.kinetic);
  Value rawVirial = k.add(twiceKinetic, in.diagonal);
  result.constantTrace = k.div(in.constantVirial, result.volume);
  Value thirdVirial = k.mul(result.constantTrace, in.third);
  result.constantDiagonal = k.broadcast(thirdVirial);
  Value virial = k.add(rawVirial, result.constantDiagonal);
  Value units = k.div(in.barUnit, result.volume);
  Value unitsVector = k.broadcast(units);
  Value pressure = k.mul(virial, unitsVector);
  if (coupling == StrainCoupling::Anisotropic) {
    SmallVector<Value> strains;
    for (int64_t axis = 0; axis != 3; ++axis) {
      Value component = k.extract(pressure, axis);
      Value number = arith::ConstantIntOp::create(builder, loc, axis, 64);
      strains.push_back(k.call("mdrtBarostatStrainAxis",
          {in.seed, in.step, number, component, in.target, result.volume,
           in.thermal, in.beta[axis], in.rate}));
    }
    result.logScale = k.elements(strains);
  } else if (coupling == StrainCoupling::Isotropic) {
    Value sum = vector::ReductionOp::create(builder, loc, vector::CombiningKind::ADD, pressure);
    Value mean = k.mul(sum, in.third);
    Value strain = k.call("mdrtBarostatStrain",
        {in.seed, in.step, mean, in.target, result.volume, in.thermal, in.beta[0], in.rate});
    Value axisStrain = k.mul(strain, in.third);
    result.logScale = k.broadcast(axisStrain);
  } else {
    Value px = k.extract(pressure, 0), py = k.extract(pressure, 1), pz = k.extract(pressure, 2);
    Value sum = k.add(px, py);
    Value mean = k.mul(sum, in.half);
    Value area = k.call("mdrtBarostatStrainArea",
        {in.seed, in.step, mean, in.target, result.volume, edges[2],
         in.thermal, in.beta[0], in.rate, in.tension});
    Value height = k.call("mdrtBarostatStrainHeight",
        {in.seed, in.step, pz, in.target, result.volume, in.thermal, in.beta[2], in.rate});
    Value xyStrain = k.mul(area, in.half);
    result.logScale = k.elements({xyStrain, xyStrain, height});
  }
  result.scale = math::ExpOp::create(builder, loc, result.logScale);
  result.inverse = k.div(in.unitVector, result.scale);
  static const unsigned columns[] = {0, 1, 2, 0, 0, 1};
  std::array<Value, 3> scales;
  for (size_t index = 0; index != edges.size(); ++index) {
    if (index < 3)
      scales[index] = k.extract(result.scale, index);
    Value edge = k.mul(edges[index], scales[columns[index]]);
    result.edges.push_back(edge);
    memref::StoreOp::create(builder, loc, edge, in.boxMemory, ValueRange{in.edgeIndices[index]});
  }
  func::CallOp::create(builder, loc, "mdrtSetBox", TypeRange{}, ValueRange(result.edges).take_front(3));
  if (edges.size() == 6)
    func::CallOp::create(builder, loc, "mdrtSetTilt", TypeRange{}, ValueRange(result.edges).drop_front(3));
  return result;
}
void mdir::driver::builder::buildStoreBarostatState(
    OpBuilder &builder, Value memory, Value diagonal, Value groups, Value kinetic) {
  auto loc = builder.getUnknownLoc();
  Kernel k(builder);
  SmallVector<Value> values;
  for (Value vector : {diagonal, groups, kinetic})
    for (int64_t component = 0; component != 3; ++component) {
      Value value = k.extract(vector, component);
      Value slot = arith::ConstantIndexOp::create(builder, loc, values.size());
      memref::StoreOp::create(builder, loc, value, memory, ValueRange{slot});
      values.push_back(value);
    }
  func::CallOp::create(builder, loc, "mdrtSetBarostatState", TypeRange{}, values);
}
Value mdir::driver::builder::buildCell(OpBuilder &builder, ValueRange edges) {
  auto loc = builder.getUnknownLoc();
  if (edges.size() == 6)
    return mdir::md::TriclinicCellOp::create(builder, loc,
        edges[0], edges[1], edges[2], edges[3], edges[4], edges[5]);
  return mdir::md::OrthorhombicCellOp::create(builder, loc, edges[0], edges[1], edges[2]);
}
Value mdir::driver::builder::buildReferenceScale(
    OpBuilder &builder, ValueRange edges, Value referenceEdges) {
  Kernel k(builder);
  Value vector = k.elements(edges.take_front(3));
  return k.div(vector, referenceEdges);
}

VirialDiagonalIR mdir::driver::builder::buildVirialDiagonal(
    OpBuilder &builder, Value virial) {
  Kernel k(builder);
  Value xx = k.extract(virial, 0), yy = k.extract(virial, 4), zz = k.extract(virial, 8);
  return {k.elements({xx, yy, zz}), xx, yy, zz};
}

Value mdir::driver::builder::buildVirialTrace(OpBuilder &builder, Value virial) {
  Kernel k(builder);
  Value xx = k.extract(virial, 0), yy = k.extract(virial, 4), zz = k.extract(virial, 8);
  Value xy = k.add(xx, yy);
  return k.add(xy, zz);
}

Value mdir::driver::builder::buildSum3(OpBuilder &builder, Value vector) {
  return vector::ReductionOp::create(builder, builder.getUnknownLoc(),
                                       vector::CombiningKind::ADD, vector);
}

// Work ordering of stochastic cell rescaling [Bernetti2020], D92/D116.
Value mdir::driver::builder::buildScaleKinetic(OpBuilder &builder, Value factor, Value kinetic) {
  Kernel k(builder);
  Value broadcast = k.broadcast(factor);
  return k.mul(broadcast, kinetic);
}
Value mdir::driver::builder::buildScaledGroupKinetic(
    OpBuilder &builder, Value factor, Value kinetic, Value forceTerm) {
  Value scaled = buildScaleKinetic(builder, factor, kinetic);
  return Kernel(builder).add(scaled, forceTerm);
}
Value mdir::driver::builder::buildVirialWork(OpBuilder &builder, Value virial, Value logScale) {
  Kernel k(builder);
  Value product = k.mul(virial, logScale);
  return k.sum(product);
}
Value mdir::driver::builder::buildMeanVirialWork(
    OpBuilder &builder, const MeanVirialWorkInputs &in) {
  Kernel k(builder);
  Value constant = k.div(in.constantVirial, in.newVolume);
  Value third = k.mul(constant, in.third);
  Value broadcast = k.broadcast(third);
  Value after = k.add(in.groupsAfter, broadcast);
  Value sum = k.add(in.workBefore, after);
  Value mean = k.mul(sum, in.half);
  Value product = k.mul(mean, in.logScale);
  return k.sum(product);
}
Value mdir::driver::builder::buildKineticScaleChange(
    OpBuilder &builder, Value inverse, Value unit, Value kinetic) {
  Kernel k(builder);
  Value squared = k.mul(inverse, inverse);
  Value difference = k.sub(squared, unit);
  Value product = k.mul(difference, kinetic);
  return k.sum(product);
}
Value mdir::driver::builder::buildTrotterBath(
    OpBuilder &builder, Value inverse, Value unit, Value kinetic, Value work, Value bath) {
  Value change = buildKineticScaleChange(builder, inverse, unit, kinetic);
  Kernel k(builder);
  Value difference = k.sub(change, work);
  return k.sub(bath, difference);
}
Value mdir::driver::builder::buildOneMinusScale(OpBuilder &builder, Value unit, Value scale) {
  return Kernel(builder).sub(unit, scale);
}
Value mdir::driver::builder::buildFirstOrderVirialWork(
    OpBuilder &builder, Value groups, Value constantTrace, Value third, Value oneMinus) {
  Kernel k(builder);
  Value trace = k.sum(groups);
  Value whole = k.add(trace, constantTrace);
  Value axis = k.mul(whole, third);
  Value broadcast = k.broadcast(axis);
  Value product = k.mul(oneMinus, broadcast);
  return k.sum(product);
}
Value mdir::driver::builder::buildBathAfterWork(
    OpBuilder &builder, Value work, Value change, Value bath) {
  Kernel k(builder);
  Value gain = k.add(work, change);
  return k.sub(bath, gain);
}
Value mdir::driver::builder::buildExactBath(OpBuilder &builder, const ExactBathInputs &in) {
  Kernel k(builder);
  Value xy = k.mul(in.edges[0], in.edges[1]);
  Value volume = k.mul(xy, in.edges[2]);
  Value nextInverse = k.div(in.unit, volume);
  Value oldInverse = k.div(in.unit, in.oldVolume);
  Value inverseChange = k.sub(nextInverse, oldInverse);
  Value constantChange = k.mul(in.constantEnergy, inverseChange);
  Value rawChange = k.sub(in.newEnergy, in.oldEnergy);
  Value energyChange = k.add(rawChange, constantChange);
  Value gain = k.add(energyChange, in.kineticChange);
  return k.sub(in.bath, gain);
}
