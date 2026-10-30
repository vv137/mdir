// Shared potential assembly, centroid geometry, observed selections, and
// site placement. Term-specific contributions are owned by the adapters.

#include "BuilderImpl.h"
#include "SiteIR.h"
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
        using namespace mlir;
        auto loc = builder.getUnknownLoc();
        auto constant = [&](double value) -> Value {
          return arith::ConstantOp::create(builder, loc,
                                           builder.getF64FloatAttr(value));
        };
        auto add = [&](Value a, Value b) -> Value {
          return arith::AddFOp::create(builder, loc, a, b);
        };
        auto sub = [&](Value a, Value b) -> Value {
          return arith::SubFOp::create(builder, loc, a, b);
        };
        auto mul = [&](Value a, Value b) -> Value {
          return arith::MulFOp::create(builder, loc, a, b);
        };
        auto sum = [&](size_t slot, int component) -> Value {
          DenseI32ArrayAttr kinds;
          DenseI64ArrayAttr members;
          mdir::md::Coordinate coordinate{
              mdir::md::CoordinateKind::Displacement, {0, 1}};
          mdir::md::getCoordinateAttrs(builder, {coordinate}, kinds, members);
          auto sum = mdir::md::SumTuplesOp::create(
              builder, loc, builder.getF64Type(), values[0], values[1],
              values[2], ValueRange{}, ValueRange{values[3 + slot]}, kinds,
              members);
          {
            OpBuilder::InsertionGuard guard(builder);
            llvm::SmallVector<Type> types = {
                VectorType::get({3}, builder.getF64Type()),
                builder.getF64Type()};
            llvm::SmallVector<Location> locations(types.size(), loc);
            Block *block =
                builder.createBlock(&sum.getKernel(), {}, types, locations);
            Value displacement = vector::ExtractOp::create(
                builder, loc, block->getArgument(0), component);
            Value energy = mul(block->getArgument(1), displacement);
            mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
          }
          return sum.getResult();
        };
        // Minimum-image member and reference links retain their original
        // loop order, allowing the same sums to fuse after inlining.
        using Vector = std::array<Value, 3>;
        Value angstrom = constant(1.0 / units::length);
        std::vector<Vector> centers;
        for (size_t g = 0; g != groups; ++g) {
          Vector position;
          for (int c = 0; c != 3; ++c) {
            Value value = sum(g, c);
            if (g > 0) {
              Value link = sum(groups + g - 1, c);
              value = add(link, value);
            }
            position[c] = mul(value, angstrom);
          }
          centers.push_back(position);
        }
        auto subtract = [&](const Vector &u, const Vector &v) {
          Vector w;
          for (int c = 0; c != 3; ++c)
            w[c] = sub(u[c], v[c]);
          return w;
        };
        auto dot = [&](const Vector &u, const Vector &v) {
          Value total = mul(u[0], v[0]);
          for (int c = 1; c != 3; ++c) {
            Value product = mul(u[c], v[c]);
            total = add(total, product);
          }
          return total;
        };
        auto cross = [&](const Vector &u, const Vector &v) {
          Vector w;
          for (int c = 0; c != 3; ++c) {
            int i = (c + 1) % 3, j = (c + 2) % 3;
            Value a = mul(u[i], v[j]);
            Value b = mul(u[j], v[i]);
            w[c] = sub(a, b);
          }
          return w;
        };
        auto scale = [&](Value factor, const Vector &u) {
          Vector w;
          for (int c = 0; c != 3; ++c)
            w[c] = mul(factor, u[c]);
          return w;
        };
        auto norm = [&](const Vector &u) -> Value {
          Value square = dot(u, u);
          return math::SqrtOp::create(builder, loc, square);
        };
        if (arity == 2) {
          Vector d = subtract(centers[1], centers[0]);
          Value r = norm(d);
          return {r, d[0], d[1], d[2]};
        }
        if (arity == 3) {
          Vector u = subtract(centers[0], centers[1]);
          Vector v = subtract(centers[2], centers[1]);
          Vector product = cross(u, v);
          Value a = norm(product);
          Value b = dot(u, v);
          return {math::Atan2Op::create(builder, loc, a, b)};
        }
        Vector b0 = subtract(centers[0], centers[1]);
        Vector b1 = subtract(centers[2], centers[1]);
        Vector b2 = subtract(centers[3], centers[2]);
        Value one = constant(1.0);
        Value length = norm(b1);
        Value inverse = arith::DivFOp::create(builder, loc, one, length);
        Vector axis = scale(inverse, b1);
        Value vProjection = dot(b0, axis);
        Vector vAxis = scale(vProjection, axis);
        Vector v = subtract(b0, vAxis);
        Value wProjection = dot(b2, axis);
        Vector wAxis = scale(wProjection, axis);
        Vector w = subtract(b2, wAxis);
        Vector product = cross(axis, v);
        Value a = dot(product, w);
        Value b = dot(v, w);
        return {math::Atan2Op::create(builder, loc, a, b)};
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

void Builder::emitObservedPotentials() {
  // Each term alone, with its observed constants as arguments, whose
  // derivatives the differentiation of parameters gives as it gives
  // dH/dλ (D161): exactly 0 where the energy provably does not depend on
  // them, an error where a dependence has no rule.
  for (auto [k, term] : llvm::enumerate(getObservedTerms())) {
    observedTerm = term.name;
    observedConstants = term.constants;
    emitTopologyPotential("observe" + std::to_string(k), term.potential.terms,
                          term.potential.tupleTerm, term.potential.pairTerm,
                          term.potential.externalTerm);
    observedTerm.clear();
    observedConstants.clear();
  }
}

void Builder::emitTopologyPotential(StringRef name, unsigned terms,
                                    int tupleTerm, int pairTerm,
                                    int externalTerm) {
  bool alchemicalOnly = terms & Alchemical;
  unsigned active = 0;
  for (const TermBuilder *builder : getTermBuilders())
    active |= builder->getPotentialMask(
        *this, dependent ? &*dependent : nullptr, alchemicalOnly);
  terms &= active;
  os << "md.potential @" << name << "(%x: !vec, %cell: !md.cell"
     << getFieldParameters() << getTimeParameter() << getLambdaParameters()
     << getObservedParameters() << getGradientParameters()
     << ") -> f64 {\n";
  emitLambdaConstants("  ");
  std::string total;
  auto add = [&](StringRef term) {
    std::string value = ("%u_" + term).str();
    if (total.empty()) {
      total = value;
      return;
    }
    std::string sum = ("%total_" + term).str();
    emitIRFragment(
        "  ", {sum}, {"f64"}, {total, value}, {"f64", "f64"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          return {mlir::arith::AddFOp::create(builder, builder.getUnknownLoc(),
                                               values[0], values[1])};
        });
    total = sum;
  };

  auto addResult = [&](StringRef value, StringRef sum) {
    if (total.empty()) {
      total = value.str();
      return;
    }
    emitIRFragment(
        "  ", {sum}, {"f64"}, {total, value}, {"f64", "f64"},
        [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          return {mlir::arith::AddFOp::create(builder, builder.getUnknownLoc(),
                                               values[0], values[1])};
        });
    total = sum.str();
  };
  TermPotentialAddition addition{add, addResult};

  // Lennard-Jones and Coulomb, both cut at the cutoff, over the pairs that
  // are not excluded: as the modifiers say, and in `@alchemical` and
  // `@observe<k>` shifted to 0 at the cutoff (D210).
  bool pairTerms = llvm::any_of(getTermBuilders(), [&](const TermBuilder *builder) {
    return builder->needsPairNeighborhood(
        *this, {terms, tupleTerm, pairTerm, externalTerm, alchemicalOnly});
  });
  if (pairTerms) {
    std::string excluded;
    for (const TermBuilder *builder : getTermBuilders()) {
      StringRef set = builder->getPairExclusion();
      if (!set.empty() && llvm::any_of(program.tupleSets,
              [&](const auto &data) { return data.name == set; })) {
        assert(excluded.empty() && "a neighborhood has one exclusion relation");
        excluded = set.str();
      }
    }
    emitPairNeighborhood("  ", "%n", "%x", "%cell", excluded);
  }
  for (const TermBuilder *builder :
       mdir::driver::builder::termBuildersByTopologyRank(getTermBuilders())) {
    TermBuilder::PotentialSelection selection{terms, tupleTerm, pairTerm,
                                              externalTerm, alchemicalOnly};
    if (const auto *role = builder->getRoles().nonbonded)
      role->emit(*this, getNonbondedData(), selection, addition);
    else
      builder->emitTopologyPotential(*this, selection, addition);
  }
  if (total.empty()) {
    emitIRFragment(
        "  ", {"%zero"}, {"f64"}, {}, {},
        [](mlir::OpBuilder &builder, mlir::ValueRange) -> TermIRValues {
          return {mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(),
                                                   builder.getF64FloatAttr(0.0))};
        });
    total = "%zero";
  }
  emitIRFragment(
      "  ", {}, {}, {total}, {"f64"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        mdir::md::ReturnOp::create(builder, builder.getUnknownLoc(), values);
        return {};
      });
  os << "}\n\n";
}

llvm::Error Builder::emitPotential() {

  if (control.truncation == Truncation::PowerForceSwitch)
    return makeError("the power force switch is for the Lennard-Jones of a "
                     "topology");

  os << "md.potential @energy(%x: !vec, %cell: !md.cell"
     << getFieldParameters() << getTimeParameter() << ") -> f64 {\n";
  emitPairNeighborhood("  ", "%n", "%x", "%cell", "");

  std::string total;
  for (const TermBuilder *builder : termBuildersByPotentialRank())
    builder->emitBareContributions(*this, total);
  emitIRFragment(
      "  ", {}, {}, {total}, {"f64"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        mdir::md::ReturnOp::create(builder, builder.getUnknownLoc(), values);
        return {};
      });
  os << "}\n\n";
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

void Builder::emitPairNeighborhood(StringRef indent, StringRef result,
                                     StringRef positions, StringRef cell,
                                     StringRef excluded) {
  llvm::SmallVector<StringRef> inputs{positions, cell}, types{"!vec", "!md.cell"};
  std::string relation = ("%r_" + excluded).str(), type = ("!rel_" + excluded).str();
  if (!excluded.empty()) {
    inputs.push_back(relation);
    types.push_back(type);
  }
  double cutoff = control.cutoffDistance * units::length;
  emitIRFragment(
      indent, {result}, {"!pairs"}, inputs, types,
      [cutoff](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        auto atoms = mlir::cast<mdir::md::FieldType>(values[0].getType()).getParticleSet();
        auto pairs = mdir::md::RelationType::get(builder.getContext(), atoms, 2,
            mdir::md::Orientation::Unordered, mlir::FlatSymbolRefAttr{});
        return {mdir::md::NeighborhoodOp::create(builder, builder.getUnknownLoc(),
            pairs, values[0], values[1], values.size() == 3 ? values[2] : mlir::Value{},
            builder.getF64FloatAttr(cutoff))};
      });
}

void Builder::emitLambdaConstants(StringRef indent) {
  if (lambdaArguments || control.freeEnergy.lambdas.empty())
    return;
  std::vector<std::string> names;
  std::vector<double> constants;
  for (const auto &[name, values] : control.freeEnergy.lambdas) {
    names.push_back("%lambda_" + name);
    constants.push_back(getLambda(name));
  }
  llvm::SmallVector<StringRef> outputs(names.begin(), names.end()), types(names.size(), "f64");
  emitIRFragment(
      indent, outputs, types, {}, {},
      [constants = std::move(constants)](mlir::OpBuilder &builder,
                                        mlir::ValueRange) -> TermIRValues {
        TermIRValues values;
        for (double constant : constants)
          values.push_back(mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(),
                                                          builder.getF64FloatAttr(constant)));
        return values;
      });
}
