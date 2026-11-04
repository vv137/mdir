// Constraint-anchor order, field permutation, and semantic derived lists.
#include "ReorderIR.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/MDRTTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include <cassert>
using namespace mlir;
using namespace mdir::driver::builder;
Value mdir::driver::builder::buildRenumber(OpBuilder &builder, Value relation, Value ids) {
  return mdir::md_exec::RenumberOp::create(builder, builder.getUnknownLoc(), relation.getType(), relation, ids);
}
ReorderIRResult mdir::driver::builder::buildReorder(
    OpBuilder &builder, const ReorderIRInputs &in, const ReorderIRConfig &config) {
  auto loc = builder.getUnknownLoc();
  Value ordered = in.positions;
  if (config.anchorRelation) {
    Value relation = buildRenumber(builder, in.originalRelations[*config.anchorRelation], in.ids);
    DenseI32ArrayAttr kinds;
    DenseI64ArrayAttr members;
    SmallVector<mdir::md::Coordinate> coordinates{{mdir::md::CoordinateKind::Displacement, {1, 0}}};
    mdir::md::getCoordinateAttrs(builder, coordinates, kinds, members);
    auto gather = mdir::md::GatherTuplesOp::create(
        builder, loc, in.positions.getType(), Type{}, relation, in.positions, in.cell,
        ValueRange{in.positions}, ValueRange{}, kinds, members);
    {
      OpBuilder::InsertionGuard guard(builder);
      auto vec = VectorType::get({3}, builder.getF64Type());
      auto args = builder.createBlock(&gather.getKernel(), {}, {vec, vec, vec}, {loc, loc, loc})->getArguments();
      Value none = arith::ConstantOp::create(builder, loc, DenseElementsAttr::get(vec, 0.0));
      mdir::md::YieldOp::create(builder, loc, ValueRange{args[2], none});
    }
    ordered = gather.getResult();
  }
  auto atoms = cast<mdir::md::FieldType>(in.ids.getType()).getParticleSet();
  auto permutation = mdir::mdrt::PermutationType::get(builder.getContext(), atoms);
  Value order = mdir::md_exec::SpatialOrderOp::create(
      builder, loc, permutation, ordered, in.cell, in.ids, Value{}, builder.getF64FloatAttr(config.width)).getResult();
  auto permute = [&](Value field) -> Value {
    return mdir::md_exec::PermuteOp::create(builder, loc, field.getType(), field, order, Value{}).getResult();
  };
  ReorderIRResult result;
  result.positions = permute(in.positions);
  result.velocities = permute(in.velocities);
  if (in.forces)
    result.forces = permute(in.forces);
  result.masses = permute(in.masses);
  for (const auto &field : in.fields) {
    if (field.regenerateFrom) {
      assert(*field.regenerateFrom < result.fields.size() && "derived list must follow its source field");
      result.fields.push_back(mdir::md_exec::NonzeroOp::create(
          builder, loc, in.ids.getType(), result.fields[*field.regenerateFrom], Value{}).getResult());
    } else {
      result.fields.push_back(permute(field.value));
    }
  }
  result.ids = permute(in.ids);
  for (Value relation : in.originalRelations)
    result.relations.push_back(buildRenumber(builder, relation, result.ids));
  return result;
}
