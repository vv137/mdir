// SETTLE, SHAKE and RATTLE, the virtual sites, and the tests and stops
// of the constraints (docs/driver.md, Section 2.12).

#include "BuilderImpl.h"
#include "ConstraintIR.h"
#include "DynamicsIR.h"
#include "SiteIR.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

void Builder::emitPlaceSites(StringRef indent, StringRef x, StringRef result,
                             StringRef relations) {
  std::string current = x.str();
  std::vector<const Program::TupleSet *> sets = getSiteSets();
  for (const Program::TupleSet *each : sets) {
    const Program::TupleSet &set = *each;
    bool amber = set.name == "sites_amber";
    bool last = each == sets.back();
    std::string relation = (relations + set.name).str();
    std::string a = "%f_" + set.name + "_a", b = "%f_" + set.name + "_b";
    std::string relationType = "!rel_" + set.name, fieldType = "!of_" + set.name;
    std::string next =
        last ? result.str() : (result + "_" + set.name + "_x").str();
    emitIRFragment(
        indent, {next}, {"!vec"}, {relation, current, "%cell", a, b},
        {relationType, "!vec", "!md.cell", fieldType, fieldType},
        [amber](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          using namespace mlir;
          auto loc = builder.getUnknownLoc();
          auto vec = VectorType::get({3}, builder.getF64Type());
          DenseI32ArrayAttr kinds;
          DenseI64ArrayAttr members;
          mdir::md::getCoordinateAttrs(
              builder, {{mdir::md::CoordinateKind::Displacement, {2, 1}},
                        {mdir::md::CoordinateKind::Displacement, {3, 1}},
                        {mdir::md::CoordinateKind::Displacement, {0, 1}}},
              kinds, members);
          auto delta = mdir::md::GatherTuplesOp::create(
              builder, loc, values[1].getType(), Type{}, values[0], values[1],
              values[2], ValueRange{}, values.slice(3, 2), kinds, members);
          {
            OpBuilder::InsertionGuard guard(builder);
            SmallVector<Type> types = {vec, vec, vec, builder.getF64Type(),
                                       builder.getF64Type()};
            SmallVector<Location> locations(types.size(), loc);
            Block *block = builder.createBlock(&delta.getKernel(), {}, types,
                                                locations);
            auto args = block->getArguments();
            Value placed = buildSitePlacement(builder, amber, args[0], args[1],
                                               args[3], args[4]);
            Value change = arith::SubFOp::create(builder, loc, placed, args[2]);
            Value zero = arith::ConstantOp::create(
                builder, loc, DenseElementsAttr::get(vec, 0.0));
            mdir::md::YieldOp::create(builder, loc,
                                      ValueRange{change, zero, zero, zero});
          }
          return {buildAddChange(builder, values[1], delta.getResult())};
        });
    current = next;
  }
}

std::string Builder::emitSpreadSites(StringRef indent, StringRef x,
                                     StringRef f, StringRef result,
                                     StringRef relations, StringRef virial,
                                     StringRef virialResult) {
  std::string current = f.str(), currentVirial = virial.str();
  std::vector<const Program::TupleSet *> sets = getSiteSets();
  for (const Program::TupleSet *each : sets) {
    const Program::TupleSet &set = *each;
    bool amber = set.name == "sites_amber", derived = spreadsSitesDerived();
    bool last = each == sets.back();
    bool changesVirial = !currentVirial.empty() && (amber || derived);
    std::string next =
        last ? result.str() : (result + "_" + set.name + "_f").str();
    std::string changed = last ? virialResult.str()
                              : (virialResult + "_" + set.name + "_w").str();
    std::string relation = (relations + set.name).str();
    std::string a = "%f_" + set.name + "_a", b = "%f_" + set.name + "_b";
    std::string relationType = "!rel_" + set.name, fieldType = "!of_" + set.name;
    llvm::SmallVector<StringRef> operands = {relation, x, "%cell", a, b, current};
    llvm::SmallVector<StringRef> types = {relationType, "!vec", "!md.cell", fieldType,
                                    fieldType, "!vec"};
    llvm::SmallVector<StringRef> outputs = {next}, outputTypes = {"!vec"};
    if (changesVirial) {
      operands.push_back(currentVirial);
      types.push_back("vector<9xf64>");
      outputs.push_back(changed);
      outputTypes.push_back("vector<9xf64>");
    }
    emitIRFragment(
        indent, outputs, outputTypes, operands, types,
        [amber, derived, changesVirial, symbol = "spread_" + set.name](
            mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          SiteSpreadInputs input{values[0], values[1], values[2], values[3],
                                  values[4], values[5], {}};
          if (changesVirial)
            input.virial = values[6];
          auto result = buildSiteSpread(builder, input, amber, derived, symbol);
          TermIRValues outputs = {result.forces};
          if (changesVirial)
            outputs.push_back(result.virial);
          return outputs;
        });
    current = next;
    if (changesVirial)
      currentVirial = changed;
  }
  return currentVirial;
}

bool Builder::spreadsSitesDerived() const {
  const char *value = std::getenv("MDIR_SITE_SPREAD");
  return value && StringRef(value) == "derived";
}


void Builder::emitSettlePositions(StringRef indent, StringRef old,
                                  StringRef x, StringRef change,
                                  StringRef result, StringRef failed) {
  // Carry small M-SHAKE changes below double precision [Jung2026] (D112).
  // Double precision keeps the SETTLE triangle rotation [Miyamoto1992].
  if (control.precision != Precision::Double) {
    emitBondConstraints(indent, "settles", 3,
                        {{0, 1, 0}, {0, 2, 0}, {1, 2, 1}},
                        {"%f_settles_doh", "%f_settles_dhh"}, old, x, change, failed);
    emitAddChange(indent, x, change, result);
    return;
  }
  bool withFailure = !failed.empty();
  llvm::SmallVector<StringRef> outputs = {change}, types = {"!vec"};
  if (withFailure) {
    outputs.push_back(failed);
    types.push_back("i1");
  }
  emitIRFragment(
      indent, outputs, types,
      {"%r_settles", x, "%cell", old, "%m", "%f_settles_doh", "%f_settles_dhh"},
      {"!rel_settles", "!vec", "!md.cell", "!vec", "!real", "!of_settles", "!of_settles"},
      [withFailure](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        ConstraintPositionInputs input{values[0], values[1], values[2], values[3],
                                         values[4], values.drop_front(5)};
        auto result = buildSettlePositions(builder, input, withFailure, 1.0e-8);
        TermIRValues outputs = {result.change};
        if (withFailure)
          outputs.push_back(result.failed);
        return outputs;
      });
  emitAddChange(indent, x, change, result);
}

std::string Builder::emitConstraintVirial(StringRef indent,
                                          const Program::TupleSet &set,
                                          StringRef old, StringRef change,
                                          StringRef virial,
                                          StringRef virialResult) {
  // The forces of the constraints over the first half of the step: the
  // drift took x + dt v + dt² F / 2m to x + Δ, so G = 2 m Δ / dt², along
  // the bonds before the drift. Their virial Σ (x_j − x_0) ⊗ G_j, taken at
  // those positions, with the weight ½: with that of the second half, the
  // virial of the constraints is the mean of the two, as the kinetic
  // energy of the pressure is (D45).
  std::string relation = "%r_" + set.name, relationType = "!rel_" + set.name;
  emitIRFragment(
      indent, {virialResult}, {"vector<9xf64>"},
      {relation, old, "%cell", change, "%m", "%dt", virial},
      {relationType, "!vec", "!md.cell", "!vec", "!real", "f64", "vector<9xf64>"},
      [arity = set.arity](mlir::OpBuilder &builder,
                           mlir::ValueRange values) -> TermIRValues {
        ConstraintVirialInputs input{values[0], values[1], values[2], values[3],
                                      values[4], values[5], values[6]};
        return {buildConstraintVirial(builder, input, arity)};
      });
  return virialResult.str();
}

void Builder::emitShakePositions(StringRef indent, StringRef old,
                                 StringRef x, const Program::TupleSet &set,
                                 StringRef result, StringRef failed) {
  unsigned count = set.arity - 1;
  std::string change = (result + "_change").str();
  // The bonds of the heavy atom 0 to its hydrogens.
  std::vector<ConstrainedBond> bonds;
  std::vector<std::string> tuple;
  for (unsigned k = 1; k <= count; ++k) {
    bonds.push_back({0, k, k - 1});
    tuple.push_back("%f_" + set.name + "_d" + std::to_string(k));
  }
  emitBondConstraints(indent, set.name, set.arity, bonds, tuple, old, x,
                      change, failed);
  emitAddChange(indent, x, change, result);
}

void Builder::emitAddChange(StringRef indent, StringRef x, StringRef change,
                            StringRef result) {
  emitIRFragment(
      indent, {result}, {"!vec"}, {x, change}, {"!vec", "!vec"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildAddChange(builder, values[0], values[1])};
      });
}

void Builder::emitBondConstraints(StringRef indent, StringRef setName,
                                  unsigned arity,
                                  llvm::ArrayRef<ConstrainedBond> bonds,
                                  llvm::ArrayRef<std::string> tuple, StringRef old,
                                  StringRef x, StringRef change,
                                  StringRef failed) {
  std::string relation = ("%r_" + setName).str();
  std::string relationType = ("!rel_" + setName).str();
  std::string fieldType = ("!of_" + setName).str();
  llvm::SmallVector<StringRef> inputs = {relation, x, "%cell", old, "%m"};
  llvm::SmallVector<StringRef> types = {relationType, "!vec", "!md.cell", "!vec", "!real"};
  for (const auto &field : tuple) {
    inputs.push_back(field);
    types.push_back(fieldType);
  }
  ConstraintPositionConfig config{
      !failed.empty(), control.analyticBonds,
      control.precision == Precision::Double ? 1.0e-8 : 1.0e-4,
      control.precision == Precision::Double ? 1.0e-12 : 1.0e-6};
  llvm::SmallVector<StringRef> outputs = {change}, resultTypes = {"!vec"};
  if (config.withFailure) {
    outputs.push_back(failed);
    resultTypes.push_back("i1");
  }
  emitIRFragment(
      indent, outputs, resultTypes, inputs, types,
      [arity, config, bonds = std::vector<ConstraintBond>(bonds.begin(), bonds.end())](
          mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        ConstraintPositionInputs input{values[0], values[1], values[2], values[3],
                                         values[4], values.drop_front(5)};
        auto result = buildBondConstraints(builder, input, arity, bonds, config);
        TermIRValues outputs = {result.change};
        if (config.withFailure)
          outputs.push_back(result.failed);
        return outputs;
      });
}

std::string Builder::emitShakeVelocities(StringRef indent, StringRef x,
                                         StringRef v,
                                         const Program::TupleSet &set,
                                         StringRef result, StringRef virial,
                                         StringRef virialResult) {
  // RATTLE removes each internal bond velocity by the original small
  // mass-weighted solve; the second half-step virial reads its stored changes.
  std::string relation = "%r_" + set.name, relationType = "!rel_" + set.name;
  std::string fieldType = "!of_" + set.name;
  std::vector<std::string> parameters;
  for (unsigned j = 1; j < set.arity; ++j)
    parameters.push_back("%f_" + set.name + "_d" + std::to_string(j));
  llvm::SmallVector<StringRef> inputs = {relation, x, "%cell", v, "%m"};
  llvm::SmallVector<StringRef> types = {relationType, "!vec", "!md.cell", "!vec", "!real"};
  for (const auto &parameter : parameters) {
    inputs.push_back(parameter);
    types.push_back(fieldType);
  }
  bool withVirial = !virial.empty();
  llvm::SmallVector<StringRef> outputs = {result}, resultTypes = {"!vec"};
  if (withVirial) {
    inputs.append({"%dt", virial});
    types.append({"f64", "vector<9xf64>"});
    outputs.push_back(virialResult);
    resultTypes.push_back("vector<9xf64>");
  }
  emitIRFragment(
      indent, outputs, resultTypes, inputs, types,
      [arity = set.arity, withVirial](mlir::OpBuilder &builder,
                                      mlir::ValueRange values) -> TermIRValues {
        unsigned fields = arity - 1;
        ConstraintVelocityInputs input{values[0], values[1], values[2], values[3],
                                         values[4], values.slice(5, fields), {}, {}};
        if (withVirial) {
          input.timeStep = values[5 + fields];
          input.virial = values[6 + fields];
        }
        auto result = buildShakeVelocities(builder, input, arity);
        TermIRValues outputs = {result.velocities};
        if (withVirial)
          outputs.push_back(result.virial);
        return outputs;
      });
  return withVirial ? virialResult.str() : "";
}

std::string Builder::emitConstraintForceVirial(StringRef indent,
                                               StringRef x, StringRef v,
                                               StringRef f,
                                               StringRef virial) {
  std::string current = virial.str();
  unsigned count = 0;
  // A group and its bonds, with member displacements measured from the first.
  auto emitGroup = [&](StringRef setName, unsigned arity,
                       llvm::ArrayRef<std::pair<unsigned, unsigned>> bonds) {
    std::string relation = ("%r_" + setName).str();
    std::string relationType = ("!rel_" + setName).str();
    std::string next = "%cfw" + std::to_string(count++) + "_total";
    emitIRFragment(
        indent, {next}, {"vector<9xf64>"},
        {relation, x, "%cell", v, f, "%m", current},
        {relationType, "!vec", "!md.cell", "!vec", "!vec", "!real", "vector<9xf64>"},
        [arity, bonds = std::vector<std::pair<unsigned, unsigned>>(bonds.begin(), bonds.end())](
            mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          ConstraintForceInputs input{values[0], values[1], values[2], values[3],
                                       values[4], values[5], values[6]};
          return {buildConstraintForceVirial(builder, input, arity, bonds)};
        });
    current = next;
  };
  for (const Program::TupleSet &set : program.tupleSets)
    if (set.name == "settles")
      emitGroup(set.name, 3, {{0, 1}, {0, 2}, {1, 2}});
  for (const Program::TupleSet *set : getShakeSets()) {
    std::vector<std::pair<unsigned, unsigned>> bonds;
    for (unsigned j = 1; j != set->arity; ++j)
      bonds.push_back({0, j});
    emitGroup(set->name, set->arity, bonds);
  }
  return current;
}

std::string Builder::emitSettleVelocities(StringRef indent, StringRef x,
                                          StringRef v, StringRef result,
                                          StringRef virial,
                                          StringRef virialResult) {
  // Keep the water-specific determinant solve and read its stored velocity
  // changes for the virial, rather than solving them again (#97).
  llvm::SmallVector<StringRef> inputs = {
      "%r_settles", x, "%cell", v, "%m", "%f_settles_doh", "%f_settles_dhh"};
  llvm::SmallVector<StringRef> types = {
      "!rel_settles", "!vec", "!md.cell", "!vec", "!real", "!of_settles", "!of_settles"};
  bool withVirial = !virial.empty();
  llvm::SmallVector<StringRef> outputs = {result}, resultTypes = {"!vec"};
  if (withVirial) {
    inputs.append({"%dt", virial});
    types.append({"f64", "vector<9xf64>"});
    outputs.push_back(virialResult);
    resultTypes.push_back("vector<9xf64>");
  }
  emitIRFragment(
      indent, outputs, resultTypes, inputs, types,
      [withVirial](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        ConstraintVelocityInputs input{values[0], values[1], values[2], values[3],
                                         values[4], values.slice(5, 2), {}, {}};
        if (withVirial) {
          input.timeStep = values[7];
          input.virial = values[8];
        }
        auto result = buildSettleVelocities(builder, input);
        TermIRValues outputs = {result.velocities};
        if (withVirial)
          outputs.push_back(result.virial);
        return outputs;
      });
  return withVirial ? virialResult.str() : "";
}

void Builder::emitConstrainedDescent(StringRef indent, StringRef x,
                                     StringRef f, StringRef result) {
  bool settles = hasSettles();
  std::vector<const Program::TupleSet *> shakeSets = getShakeSets();
  unsigned steps = (settles ? 1 : 0) + shakeSets.size(), step = 0;
  std::string current = (result + "_all").str();
  if (steps == 0)
    current = result.str();
  emitIRFragment(
      indent, {current}, {"!vec"}, {f, "%m"}, {"!vec", "!real"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildDescentDirection(builder, values[0], values[1])};
      });
  // As RATTLE takes the velocities off the constraints.
  auto next = [&]() {
    ++step;
    return step == steps ? result.str()
                         : (result + "_c" + std::to_string(step)).str();
  };
  if (settles) {
    std::string projected = next();
    emitSettleVelocities(indent, x, current, projected);
    current = projected;
  }
  for (const Program::TupleSet *set : shakeSets) {
    std::string projected = next();
    emitShakeVelocities(indent, x, current, *set, projected, "", "");
    current = projected;
  }
}

std::string Builder::emitStartConstraintTrace(StringRef x, StringRef f,
                                              StringRef v, StringRef trace,
                                              bool axes) {
  // With `axes`, the diagonal instead of the trace, each axis with twice
  // its own internal kinetic energy: the identity holds for the trace, and
  // for each axis only in the mean, which serves the first scaling of a
  // run (D119).
  std::string name = axes ? "%dgc0" : "%trc0";
  std::string type = axes ? "vector<3xf64>" : "f64";
  std::string descent = axes ? "%gc0d" : "%gc0";
  emitConstrainedDescent("  ", x, f, descent);
  std::vector<const Program::TupleSet *> groups = getShakeSets();
  for (const Program::TupleSet &set : program.tupleSets)
    if (set.name == "settles")
      groups.insert(groups.begin(), &set);
  std::string current = trace.str();
  for (const Program::TupleSet *set : groups) {
    std::string relation = "%r_" + set->name, relationType = "!rel_" + set->name;
    std::string next = name + "_" + set->name + "_sum";
    emitIRFragment(
        "  ", {next}, {type},
        {relation, x, "%cell", descent, f, v, "%m", current},
        {relationType, "!vec", "!md.cell", "!vec", "!vec", "!vec", "!real", type},
        [arity = set->arity, axes](mlir::OpBuilder &builder,
                                    mlir::ValueRange values) -> TermIRValues {
          ConstraintTraceInputs input{values[0], values[1], values[2], values[3],
                                       values[4], values[5], values[6], values[7]};
          return {buildConstraintTrace(builder, input, arity, axes)};
        });
    current = next;
  }
  return current;
}

void Builder::emitConstraintFailures(StringRef indent, StringRef x,
                                     StringRef result) {
  auto counts = emitConstraintTests(indent, x, result, /*flag=*/false);
  llvm::SmallVector<StringRef> inputs, types;
  for (const auto &count : counts) {
    inputs.push_back(count.second);
    types.push_back("f64");
  }
  emitIRFragment(
      indent, {result}, {"f64"}, inputs, types,
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        using namespace mlir;
        auto loc = builder.getUnknownLoc();
        Value current;
        for (Value value : values)
          current = current ? arith::AddFOp::create(builder, loc, current, value)
                            : value;
        Value none = arith::ConstantOp::create(builder, loc,
                                               builder.getF64FloatAttr(0.0));
        if (!current)
          current = none;
        return {arith::AddFOp::create(builder, loc, current, none)};
      });
}

void Builder::emitConstraintStops(
    StringRef indent,
    const std::vector<std::pair<int, std::string>> &failures) {
  if (failures.empty())
    return;
  // The preceding step evaluation has already waited for the stream's flag;
  // reading it here adds no wait (D118). Stops keep their existing effects.
  llvm::SmallVector<StringRef> inputs = {"%noise_step"}, types = {"i64"};
  std::vector<int> sets;
  for (const auto &[set, failed] : failures) {
    sets.push_back(set);
    inputs.push_back(failed);
    types.push_back("i1");
  }
  emitIRFragment(
      indent, {}, {}, inputs, types,
      [sets](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        using namespace mlir;
        auto loc = builder.getUnknownLoc();
        for (auto [index, set] : llvm::enumerate(sets)) {
          auto branch = scf::IfOp::create(builder, loc, values[index + 1], false);
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(branch.thenBlock());
          Value number = arith::ConstantIntOp::create(builder, loc, set, 64);
          func::CallOp::create(builder, loc, "mdrtStopConstraintFailure",
                               TypeRange{}, ValueRange{values[0], number});
        }
        func::CallOp::create(builder, loc, "mdrtStopPending", TypeRange{},
                             ValueRange{});
        return {};
      });
}

std::vector<std::pair<int, std::string>>
Builder::emitConstraintTests(StringRef indent, StringRef x, StringRef prefix,
                             bool flag) {
  // A length fails if |r²/d² − 1| is above the tolerance or is not a
  // number (an unordered comparison). Below double precision the lengths of
  // intact waters of a box of 150 Å, stored in f32, are off by up to
  // 4·10⁻⁷ (measured, #343), and a water that the constraints did not take
  // back to its shape by 10⁻¹ or more: 10⁻⁴ lies between them. In double
  // precision SETTLE and SHAKE meet them to the rounding of f64.
  double tolerance = control.precision == Precision::Double ? 1.0e-8 : 1.0e-4;
  struct TestBond {
    unsigned first, second, length;
  };
  struct Group {
    std::string set;
    std::vector<TestBond> bonds;
    std::vector<std::string> fields;
    int number;
  };
  std::vector<Group> groups;
  if (hasSettles())
    groups.push_back({"settles", {{0, 1, 0}, {0, 2, 0}, {1, 2, 1}},
                      {"%f_settles_doh", "%f_settles_dhh"}, 0});
  for (const Program::TupleSet *set : getShakeSets()) {
    Group group{set->name, {}, {}, static_cast<int>(set->arity) - 1};
    for (unsigned k = 1; k < set->arity; ++k) {
      group.bonds.push_back({0, k, k - 1});
      group.fields.push_back("%f_" + set->name + "_d" + std::to_string(k));
    }
    groups.push_back(std::move(group));
  }
  std::vector<std::pair<int, std::string>> results;
  for (const Group &group : groups) {
    std::string relation = "%r_" + group.set;
    std::string relationType = "!rel_" + group.set;
    std::string fieldType = "!of_" + group.set;
    std::string sum = (prefix + "_" + group.set).str();
    llvm::SmallVector<StringRef> inputs = {relation, x, "%cell"};
    llvm::SmallVector<StringRef> inputTypes = {relationType, "!vec", "!md.cell"};
    for (const auto &field : group.fields) {
      inputs.push_back(field);
      inputTypes.push_back(fieldType);
    }
    emitIRFragment(
        indent, {sum}, {flag ? "i1" : "f64"}, inputs, inputTypes,
        [bonds = group.bonds, flag, tolerance](mlir::OpBuilder &builder,
                                               mlir::ValueRange values) -> TermIRValues {
          using namespace mlir;
          auto loc = builder.getUnknownLoc();
          auto f64 = builder.getF64Type();
          auto vec = VectorType::get({3}, f64);
          llvm::SmallVector<mdir::md::Coordinate> coordinates;
          for (const auto &bond : bonds)
            coordinates.push_back({mdir::md::CoordinateKind::Displacement,
                                    {bond.second, bond.first}});
          DenseI32ArrayAttr kinds;
          DenseI64ArrayAttr members;
          mdir::md::getCoordinateAttrs(builder, coordinates, kinds, members);
          Type resultType = flag ? Type(builder.getI1Type()) : Type(f64);
          auto sum = mdir::md::SumTuplesOp::create(
              builder, loc, resultType, values[0], values[1], values[2],
              ValueRange{}, values.drop_front(3), kinds, members);
          {
            OpBuilder::InsertionGuard guard(builder);
            llvm::SmallVector<Type> types(bonds.size(), vec);
            types.append(values.size() - 3, f64);
            llvm::SmallVector<Location> locations(types.size(), loc);
            auto args = builder.createBlock(&sum.getKernel(), {}, types,
                                              locations)->getArguments();
            auto constant = [&](double value) -> Value {
              return arith::ConstantOp::create(builder, loc,
                                                builder.getF64FloatAttr(value));
            };
            Value one = constant(1.0), zero = constant(0.0), limit = constant(tolerance);
            Value count;
            for (auto [index, bond] : llvm::enumerate(bonds)) {
              Value length = args[bonds.size() + bond.length];
              Value d2 = arith::MulFOp::create(builder, loc, length, length);
              Value r = args[index];
              Value square = arith::MulFOp::create(builder, loc, r, r);
              Value norm = vector::ReductionOp::create(
                  builder, loc, vector::CombiningKind::ADD, square);
              Value ratio = arith::DivFOp::create(builder, loc, norm, d2);
              Value off = arith::SubFOp::create(builder, loc, ratio, one);
              Value magnitude = math::AbsFOp::create(builder, loc, off);
              Value bad = arith::CmpFOp::create(builder, loc,
                                                arith::CmpFPredicate::UGT,
                                                magnitude, limit);
              if (flag) {
                count = count ? arith::OrIOp::create(builder, loc, count, bad) : bad;
              } else {
                Value failed = arith::SelectOp::create(builder, loc, bad, one, zero);
                count = count ? arith::AddFOp::create(builder, loc, count, failed)
                              : failed;
              }
            }
            mdir::md::YieldOp::create(builder, loc, ValueRange{count});
          }
          return {sum.getResult()};
        });
    results.push_back({group.number, sum});
  }
  return results;
}
