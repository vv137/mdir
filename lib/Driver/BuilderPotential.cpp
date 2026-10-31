// Shared potential assembly, centroid geometry, observed selections, and
// site placement. Term-specific contributions are owned by the adapters.

#include "PotentialIR.h"
#include "BuilderImpl.h"
#include "SiteIR.h"
#include "PotentialIR.h"
#include "mlir/IR/BuiltinOps.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

llvm::StringMap<std::string> Builder::emitCentroidCoordinates(
    const TupleTerm &term, StringRef indent, StringRef x, StringRef cell,
    StringRef relations, StringRef prefix) {
  const size_t groups = term.centers.size();
  const unsigned arity = term.arity;
  std::string set = "cg_" + term.name;
  std::string relationType = "!rel_" + set, fieldType = "!of_" + set;
  std::vector<std::string> names = {(relations + set).str(), x.str(),
                                    cell.str()};
  llvm::SmallVector<StringRef> types = {relationType, "!vec", "!md.cell"};
  for (size_t slot = 0; slot + 1 < 2 * groups; ++slot) {
    names.push_back("%f_" + set + "_w" + std::to_string(slot));
    types.push_back(fieldType);
  }
  llvm::SmallVector<StringRef> operands;
  for (const auto &name : names)
    operands.push_back(name);
  std::vector<std::string> results;
  llvm::StringMap<std::string> mapping;
  llvm::SmallVector<StringRef> keys;
  if (arity == 2)
    keys = {"r", "dx", "dy", "dz"};
  else
    keys = {"theta"};
  for (StringRef key : keys) {
    results.push_back((prefix + key).str());
    mapping[key] = results.back();
  }
  llvm::SmallVector<StringRef> resultNames, resultTypes;
  for (const auto &result : results) {
    resultNames.push_back(result);
    resultTypes.push_back("f64");
  }
  emitIRFragment(
      indent, resultNames, resultTypes, operands, types,
      [groups, arity](mlir::OpBuilder &builder,
                      mlir::ValueRange values) -> TermIRValues {
        return buildCentroidCoordinates(builder, values, groups, arity);
      });
  return mapping;
}


unsigned Builder::getNumPotentialArguments() const {
  unsigned count = 2 + program.fields.size() + program.tables.size();
  for (const Program::TupleSet &set : program.tupleSets)
    count += 1 + set.fields.size();
  return count + (control.usesTime ? 1 : 0);
}

std::vector<Builder::ObservedTerm> Builder::getObservedTerms() const {
  // The terms in the order of their first column; Control has checked the
  // names and the constants.
  std::vector<ObservedTerm> terms;
  if (control.observables.empty() || !system.topology)
    return terms;
  for (auto [column, observable] : llvm::enumerate(control.observables)) {
    auto it = llvm::find_if(terms, [&](const ObservedTerm &term) {
      return term.name == observable.term;
    });
    if (it == terms.end()) {
      ObservedTerm term;
      term.name = observable.term;
      for (const TermBuilder *builder : getTermBuilders()) {
        auto reference = builder->findObservedTerm(*this, term.name);
        if (reference.kind) {
          term.kind = reference.kind;
          term.index = reference.index;
          term.potential = reference.potential;
        }
      }
      terms.push_back(std::move(term));
      it = std::prev(terms.end());
    }
    int place = -1;
    if (!observable.constant.empty()) {
      place = static_cast<int>(it->constants.size());
      it->constants.push_back(observable.constant);
    }
    it->columns.push_back({place, column});
  }
  return terms;
}

llvm::Error Builder::emitObservedPotentials() {
  // Each term alone, with its observed constants as arguments, whose
  // derivatives the differentiation of parameters gives as it gives
  // dH/dλ (D161): exactly 0 where the energy provably does not depend on
  // them, an error where a dependence has no rule.
  for (auto [k, term] : llvm::enumerate(getObservedTerms())) {
    observedTerm = term.name;
    observedConstants = term.constants;
    if (llvm::Error error = emitTopologyPotential("observe" + std::to_string(k), term.potential.terms,
                          term.potential.tupleTerm, term.potential.pairTerm,
                          term.potential.externalTerm)) return error;
    observedTerm.clear();
    observedConstants.clear();
  }
  return llvm::Error::success();
}

llvm::Error Builder::emitTopologyPotential(StringRef name, unsigned terms,
                                    int tupleTerm, int pairTerm,
                                    int externalTerm) {
  bool alchemicalOnly = terms & Alchemical;
  unsigned active = 0;
  for (const TermBuilder *builder : getTermBuilders())
    active |= builder->getPotentialMask(
        *this, dependent ? &*dependent : nullptr, alchemicalOnly);
  terms &= active;

    auto &context = termIR.getContext();
    mlir::OpBuilder ir(&context);
    ir.setInsertionPointToEnd(termIR.getModule().getBody());
    auto loc = ir.getUnknownLoc();
    auto atoms = mlir::FlatSymbolRefAttr::get(&context, "atoms");
    auto vec = mdir::md::FieldType::get(&context, atoms, 3, ir.getF64Type());
    auto real = mdir::md::FieldType::get(&context, atoms, 1, ir.getF64Type());
    auto parameters = buildParameterTypes(context, program);
    llvm::SmallVector<mlir::Type> inputs{vec, mdir::md::CellType::get(&context)};
    llvm::append_range(inputs, parameters);
    if (control.usesTime) inputs.push_back(ir.getF64Type());
    if (lambdaArguments) inputs.append(control.freeEnergy.lambdas.size(), ir.getF64Type());
    inputs.append(observedConstants.size(), ir.getF64Type());
    if (gradientArguments) {
      inputs.append(control.tunableConstants.size(), ir.getF64Type());
      inputs.append(getNumGradientSeeds(), real);
    }
    auto builtPotential = buildScalarPotential(ir, name, inputs,
        [&](mlir::OpBuilder &ir, mlir::ValueRange args) -> llvm::Expected<mlir::Value> {
      unsigned nextExtra = 2 + parameters.size() + unsigned(control.usesTime);
      llvm::SmallVector<mlir::Value> lambdaValues, observedValues;
      for (const auto &[name, values] : control.freeEnergy.lambdas)
        lambdaValues.push_back(lambdaArguments ? args[nextExtra++]
            : mlir::Value(mlir::arith::ConstantOp::create(ir, loc, ir.getF64FloatAttr(getLambda(name)))));
      for (size_t i = 0; i != observedConstants.size(); ++i) observedValues.push_back(args[nextExtra++]);
      auto field = [&](StringRef name) -> mlir::Value {
        auto found = llvm::find_if(program.fields, [&](const auto &field) { return field.name == name; });
        assert(found != program.fields.end() && "potential field must exist");
        return args[2 + std::distance(program.fields.begin(), found)];
      };
      auto table = [&](StringRef name) -> mlir::Value {
        auto found = llvm::find_if(program.tables, [&](const auto &table) { return table.name == name; });
        assert(found != program.tables.end() && "potential table must exist");
        return args[2 + program.fields.size() + std::distance(program.tables.begin(), found)];
      };
      auto tuple = [&](StringRef name) -> std::pair<const Program::TupleSet *, unsigned> {
        unsigned slot = program.fields.size() + program.tables.size();
        for (const auto &set : program.tupleSets) {
          if (set.name == name) return {&set, slot};
          slot += 1 + set.fields.size();
        }
        return {nullptr, 0};
      };
      auto relation = [&](StringRef name) -> mlir::Value {
        auto [set, slot] = tuple(name);
        return set ? args[2 + slot] : mlir::Value{};
      };
      auto tupleField = [&](StringRef setName, StringRef name) -> mlir::Value {
        auto [set, slot] = tuple(setName);
        assert(set && "potential tuple set must exist");
        auto found = llvm::find_if(set->fields, [&](const auto &field) { return field.name == name; });
        assert(found != set->fields.end() && "potential tuple field must exist");
        return args[2 + slot + 1 + std::distance(set->fields.begin(), found)];
      };
      auto bindLambdas = [&](llvm::StringMap<mlir::Value> &variables) {
        unsigned index = 0;
        for (const auto &[name, values] : control.freeEnergy.lambdas)
          variables["lambda_" + name] = lambdaValues[index++];
      };
      auto bindObserved = [&](StringRef term, llvm::StringMap<mlir::Value> &variables) {
        if (term != observedTerm) return;
        for (auto [index, name] : llvm::enumerate(observedConstants)) variables[name] = observedValues[index];
      };
      auto gradientSeed = [&](unsigned index) -> mlir::Value {
        assert(gradientArguments && index < getNumGradientSeeds());
        return args[nextExtra + control.tunableConstants.size() + index];
      };
      TermBuilder::PotentialSelection selection{terms, tupleTerm, pairTerm, externalTerm, alchemicalOnly};
      auto gradientConstant = [&](unsigned index) -> mlir::Value {
        assert(gradientArguments && index < control.tunableConstants.size());
        return args[nextExtra + index];
      };
      mlir::Value neighborhood;
      if (llvm::any_of(getTermBuilders(), [&](const TermBuilder *term) {
            return term->needsPairNeighborhood(*this, selection);
          })) {
        mlir::Value excluded;
        for (const TermBuilder *term : getTermBuilders()) {
          StringRef set = term->getPairExclusion();
          if (set.empty()) continue;
          if (auto value = relation(set)) {
            assert(!excluded && "a neighborhood has one exclusion relation");
            excluded = value;
          }
        }
        auto pairs = mdir::md::RelationType::get(&context, atoms, 2,
            mdir::md::Orientation::Unordered, mlir::FlatSymbolRefAttr{});
        neighborhood = mdir::md::NeighborhoodOp::create(ir, loc, pairs,
            args[0], args[1], excluded,
            ir.getF64FloatAttr(control.cutoffDistance * units::length));
      }
      TermNativeTopology state{{args[0], args[1], neighborhood,
          control.usesTime ? args[2 + parameters.size()] : mlir::Value{}, field, table}, relation, tupleField, bindLambdas, bindObserved, gradientSeed, gradientConstant, getPairTruncation(), lambdaArguments, shiftsAtCutoff};
      mlir::Value total;
      auto add = [&](mlir::Value energy) {
        total = total ? mlir::arith::AddFOp::create(ir, loc, total, energy).getResult() : energy;
      };
      for (const TermBuilder *term : mdir::driver::builder::termBuildersByTopologyRank(getTermBuilders()))
        if (llvm::Error error = term->buildTopologyPotential(*this, ir, state, selection, add))
          return std::move(error);
      if (!total) total = mlir::arith::ConstantOp::create(ir, loc, ir.getF64FloatAttr(0.0));
      return total;
    });
    if (!builtPotential) return builtPotential.takeError();
    auto potential = *builtPotential;
    emitIRFragment("", {}, {}, {}, {},
        [potential](mlir::OpBuilder &where, mlir::ValueRange) -> TermIRValues {
          potential->moveBefore(where.getInsertionBlock(), where.getInsertionPoint());
          return {};
        });
    return llvm::Error::success();

}

llvm::Error Builder::emitPotential() {

  if (control.truncation == Truncation::PowerForceSwitch)
    return makeError("the power force switch is for the Lennard-Jones of a "
                     "topology");

  auto &context = termIR.getContext();
  mlir::OpBuilder ir(&context);
  ir.setInsertionPointToEnd(termIR.getModule().getBody());
  auto loc = ir.getUnknownLoc();
  auto atoms = mlir::FlatSymbolRefAttr::get(&context, "atoms");
  auto vec = mdir::md::FieldType::get(&context, atoms, 3, ir.getF64Type());
  auto parameters = buildParameterTypes(context, program);
  llvm::SmallVector<mlir::Type> inputs{vec, mdir::md::CellType::get(&context)};
  llvm::append_range(inputs, parameters);
  if (control.usesTime) inputs.push_back(ir.getF64Type());
  auto builtPotential = buildScalarPotential(ir, "energy", inputs,
      [&](mlir::OpBuilder &ir, mlir::ValueRange args) -> mlir::Value {
    auto pairs = mdir::md::RelationType::get(&context, atoms, 2,
        mdir::md::Orientation::Unordered, mlir::FlatSymbolRefAttr{});
    mlir::Value neighborhood = mdir::md::NeighborhoodOp::create(ir, loc, pairs,
        args[0], args[1], mlir::Value{}, ir.getF64FloatAttr(control.cutoffDistance * units::length));
    auto field = [&](StringRef name) -> mlir::Value {
      auto found = llvm::find_if(program.fields, [&](const auto &field) { return field.name == name; });
      assert(found != program.fields.end() && "bare potential field must exist");
      return args[2 + std::distance(program.fields.begin(), found)];
    };
    auto table = [&](StringRef name) -> mlir::Value {
      auto found = llvm::find_if(program.tables, [&](const auto &table) { return table.name == name; });
      assert(found != program.tables.end() && "bare potential table must exist");
      return args[2 + program.fields.size() + std::distance(program.tables.begin(), found)];
    };
    TermNativePotential state{args[0], args[1], neighborhood,
        control.usesTime ? args.back() : mlir::Value{}, field, table};
    mlir::Value total;
    for (const TermBuilder *term : termBuildersByPotentialRank())
      total = term->buildBareContributions(*this, ir, state, total);
    return total;
  });
  if (!builtPotential) return builtPotential.takeError();
  auto potential = *builtPotential;
  emitIRFragment("", {}, {}, {}, {},
      [potential](mlir::OpBuilder &where, mlir::ValueRange) -> TermIRValues {
        potential->moveBefore(where.getInsertionBlock(), where.getInsertionPoint());
        return {};
      });
  return llvm::Error::success();
}

void Builder::emitSitePotentials() {
  if (!spreadsSitesDerived())
    return;
  // x_s = x_o + p(d_j, d_k) places a site from its owner o and two atoms j
  // and k, d_j = x_j − x_o. The force f on the site moves to the atoms by
  // the transpose of the derivative of the placement, Jᵀ f, which is the
  // gradient in the positions of Σ_t f_t · x_s(t) at a fixed f: the forces
  // of this potential, −∇(Σ_t f_t · p(t)), are −Jᵀ f less its part along
  // the owner's translation, which buildSiteSpread adds. md-differentiate
  // takes them from the declared transpose of md.sum_tuples and the
  // reverse derivative of the kernel (D295).
  for (const Program::TupleSet *each : getSiteSets()) {
    const Program::TupleSet &set = *each;
    bool amber = set.name == "sites_amber";
    std::string symbol = "spread_" + set.name;
    std::string setName = set.name;
    unsigned arity = set.arity;
    auto orientation = set.arity == 2 && !set.oriented
                           ? mdir::md::Orientation::Unordered
                           : set.reversible ? mdir::md::Orientation::Reversal
                                            : mdir::md::Orientation::Ordered;
    emitIRFragment(
        "", {}, {}, {}, {},
        [symbol, setName, arity, orientation, amber](mlir::OpBuilder &builder,
                                                     mlir::ValueRange) -> TermIRValues {
          using namespace mlir;
          auto ctx = builder.getContext();
          auto loc = builder.getUnknownLoc();
          auto f64 = builder.getF64Type();
          auto vec = VectorType::get({3}, f64);
          auto atoms = FlatSymbolRefAttr::get(ctx, "atoms");
          auto field = mdir::md::FieldType::get(ctx, atoms, 3, f64);
          auto tuples = FlatSymbolRefAttr::get(ctx, setName);
          auto parameter = mdir::md::FieldType::get(ctx, tuples, 1, f64);
          auto relation = mdir::md::RelationType::get(ctx, atoms, arity,
                                                      orientation, tuples);
          SmallVector<Type> inputs = {field, mdir::md::CellType::get(ctx),
                                      relation, parameter, parameter, field};
          auto potential = mdir::md::PotentialOp::create(
              builder, loc, symbol, builder.getFunctionType(inputs, {f64}),
              StringAttr{}, ArrayAttr{}, ArrayAttr{});
          OpBuilder::InsertionGuard potentialGuard(builder);
          SmallVector<Location> locations(inputs.size(), loc);
          Block *body = builder.createBlock(&potential.getBody(), {}, inputs,
                                             locations);
          auto args = body->getArguments();
          DenseI32ArrayAttr kinds;
          DenseI64ArrayAttr members;
          mdir::md::getCoordinateAttrs(
              builder, {{mdir::md::CoordinateKind::Displacement, {2, 1}},
                        {mdir::md::CoordinateKind::Displacement, {3, 1}}},
              kinds, members);
          auto sum = mdir::md::SumTuplesOp::create(
              builder, loc, f64, args[2], args[0], args[1], ValueRange{args[5]},
              ValueRange{args[3], args[4]}, kinds, members);
          {
            OpBuilder::InsertionGuard guard(builder);
            SmallVector<Type> types(6, vec);
            types.append(2, f64);
            SmallVector<Location> locations(types.size(), loc);
            Block *block = builder.createBlock(&sum.getKernel(), {}, types,
                                                locations);
            auto values = block->getArguments();
            Value placed = buildSitePlacement(builder, amber, values[0], values[1],
                                               values[6], values[7]);
            Value product = arith::MulFOp::create(builder, loc, values[2], placed);
            Value work = vector::ReductionOp::create(
                builder, loc, vector::CombiningKind::ADD, product);
            mdir::md::YieldOp::create(builder, loc, ValueRange{work});
          }
          mdir::md::ReturnOp::create(builder, loc, ValueRange{sum.getResult()});
          return {};
        });
  }
}

std::string Builder::emitHostEvaluation(StringRef indent, StringRef x,
    StringRef cell, StringRef prefix, StringRef f, StringRef fResult,
    StringRef u, StringRef uResult, StringRef w, StringRef tag) {
  TermEvaluation point{indent, x, cell, prefix, f, fResult, u, uResult, w, tag};
  auto emitIR = [this](StringRef indent, llvm::ArrayRef<StringRef> results,
                       llvm::ArrayRef<StringRef> resultTypes,
                       llvm::ArrayRef<StringRef> operands,
                       llvm::ArrayRef<StringRef> operandTypes,
                       TermIRRecipe recipe) {
    emitIRFragment(indent, results, resultTypes, operands, operandTypes,
                   std::move(recipe));
  };
  std::string virial = w.str();
  for (const TermBuilder *builder : getTermBuilders())
    if (const TermHostEvaluation *role = builder->getRoles().hostEvaluation)
      if (role->applies(*this))
        virial = role->emit(*this, emitIR, point);
  return virial;
}
