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
          return {buildPlacedSites(builder,
              {values[0], values[1], values[2], values[3], values[4]}, amber)};
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
  using TestBond = ConstraintBond;
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
          return {buildConstraintTest(builder,
              {values[0], values[1], values[2], values.drop_front(3)}, bonds, flag, tolerance)};
        });
    results.push_back({group.number, sum});
  }
  return results;
}
