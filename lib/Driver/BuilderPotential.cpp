// Shared potential assembly, centroid geometry, observed selections, and
// site placement. Term-specific contributions are owned by the adapters.

#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

llvm::StringMap<std::string> Builder::emitCentroidCoordinates(
    const TupleTerm &term, StringRef indent, StringRef x, StringRef cell,
    StringRef relations, StringRef prefix) {
  unsigned counter = 0;
  auto next = [&]() { return (prefix + llvm::Twine(counter++)).str(); };
  auto op = [&](StringRef name, std::initializer_list<std::string> operands) {
    std::string result = next();
    os << indent << result << " = " << name << " "
       << llvm::join(operands, ", ") << " : f64\n";
    return result;
  };
  auto constant = [&](double value) {
    std::string result = next();
    os << indent << result << " = arith.constant " << formatReal(value)
       << " : f64\n";
    return result;
  };
  // One component of the displacements of the pairs of the set of the
  // term, weighted by the field `w<slot>`, summed. The sums over the one
  // set become one loop (md-exec-fuse-loops).
  std::string set = "cg_" + term.name;
  auto sum = [&](size_t slot, int component) {
    std::string result = next();
    std::string field = "%f_" + set + "_w" + std::to_string(slot);
    os << indent << result << " = md.sum_tuples " << relations << set << ", "
       << x << ", " << cell << " coordinates(displacement(0, 1))\n"
       << indent << "    tuple(" << field << " : !of_" << set << ") {\n"
       << indent << "^bb0(%gd: vector<3xf64>, %gw: f64):\n"
       << indent << "  %gdc = vector.extract %gd[" << component
       << "] : f64 from vector<3xf64>\n"
       << indent << "  %ge = arith.mulf %gw, %gdc : f64\n"
       << indent << "  md.yield %ge : f64\n"
       << indent << "} : !rel_" << set << ", !vec -> f64\n";
    return result;
  };

  // The center of each group from the reference of the first, in Å: the
  // displacement of its reference from that one, and the weighted
  // displacements of its members from its reference. Both are minimum
  // images; their sum is the center while each group lies within half the
  // least width of the cell around its reference (System.cpp).
  using Vector = std::array<std::string, 3>;
  std::string angstrom = constant(1.0 / units::length);
  std::vector<Vector> centers;
  size_t groups = term.centers.size();
  for (size_t g = 0; g != groups; ++g) {
    Vector position;
    for (int c = 0; c != 3; ++c) {
      std::string value = sum(g, c);
      if (g > 0)
        value = op("arith.addf", {sum(groups + g - 1, c), value});
      position[c] = op("arith.mulf", {value, angstrom});
    }
    centers.push_back(position);
  }
  auto subtract = [&](const Vector &u, const Vector &v) {
    Vector w;
    for (int c = 0; c != 3; ++c)
      w[c] = op("arith.subf", {u[c], v[c]});
    return w;
  };
  auto dot = [&](const Vector &u, const Vector &v) {
    std::string total = op("arith.mulf", {u[0], v[0]});
    for (int c = 1; c != 3; ++c)
      total = op("arith.addf", {total, op("arith.mulf", {u[c], v[c]})});
    return total;
  };
  auto cross = [&](const Vector &u, const Vector &v) {
    Vector w;
    for (int c = 0; c != 3; ++c) {
      int i = (c + 1) % 3, j = (c + 2) % 3;
      w[c] = op("arith.subf", {op("arith.mulf", {u[i], v[j]}),
                               op("arith.mulf", {u[j], v[i]})});
    }
    return w;
  };
  auto scale = [&](const std::string &factor, const Vector &u) {
    Vector w;
    for (int c = 0; c != 3; ++c)
      w[c] = op("arith.mulf", {factor, u[c]});
    return w;
  };
  auto norm = [&](const Vector &u) { return op("math.sqrt", {dot(u, u)}); };

  // The coordinate, as the terms over particles take it: the distance and
  // the vector from the first center to the second; the angle at the
  // second center, as atan2(|u × v|, u · v); the dihedral with trans at π.
  llvm::StringMap<std::string> values;
  if (term.arity == 2) {
    Vector d = subtract(centers[1], centers[0]);
    values["dx"] = d[0];
    values["dy"] = d[1];
    values["dz"] = d[2];
    values["r"] = norm(d);
  } else if (term.arity == 3) {
    Vector u = subtract(centers[0], centers[1]);
    Vector v = subtract(centers[2], centers[1]);
    values["theta"] = op("math.atan2", {norm(cross(u, v)), dot(u, v)});
  } else {
    Vector b0 = subtract(centers[0], centers[1]);
    Vector b1 = subtract(centers[2], centers[1]);
    Vector b2 = subtract(centers[3], centers[2]);
    Vector axis = scale(op("arith.divf", {constant(1.0), norm(b1)}), b1);
    Vector v = subtract(b0, scale(dot(b0, axis), axis));
    Vector w = subtract(b2, scale(dot(b2, axis), axis));
    values["theta"] =
        op("math.atan2", {dot(cross(axis, v), w), dot(v, w)});
  }
  return values;
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
  double cutoff = control.cutoffDistance * units::length;
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
    os << "  " << sum << " = arith.addf " << total << ", " << value
       << " : f64\n";
    total = sum;
  };

  auto addResult = [&](StringRef value, StringRef sum) {
    if (total.empty()) {
      total = value.str();
      return;
    }
    os << "  " << sum << " = arith.addf " << total << ", " << value
       << " : f64\n";
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
    os << "  %n = md.neighborhood %x, %cell cutoff(" << formatReal(cutoff)
       << ")";
    for (const TermBuilder *builder : getTermBuilders()) {
      StringRef set = builder->getPairExclusion();
      if (!set.empty() &&
          llvm::any_of(program.tupleSets,
                       [&](const auto &data) { return data.name == set; }))
        os << " exclude(%r_" << set << " : !rel_" << set << ")";
    }
    os << " : !vec -> !pairs\n";
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
    os << "  %zero = arith.constant 0.0 : f64\n";
    total = "%zero";
  }
  os << "  md.return " << total << " : f64\n}\n\n";
}

llvm::Error Builder::emitPotential() {
  double cutoff = control.cutoffDistance * units::length;

  if (control.truncation == Truncation::PowerForceSwitch)
    return makeError("the power force switch is for the Lennard-Jones of a "
                     "topology");

  os << "md.potential @energy(%x: !vec, %cell: !md.cell"
     << getFieldParameters() << getTimeParameter() << ") -> f64 {\n";
  os << "  %n = md.neighborhood %x, %cell cutoff(" << formatReal(cutoff)
     << ") : !vec -> !pairs\n";

  std::string total;
  for (const TermBuilder *builder : termBuildersByPotentialRank())
    builder->emitBareContributions(*this, total);
  os << "  md.return " << total << " : f64\n}\n\n";
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
  // the owner's translation, which emitDerivedSpread adds. md-differentiate
  // takes them from the declared transpose of md.sum_tuples and the
  // reverse derivative of the kernel (D295).
  for (const Program::TupleSet *each : getSiteSets()) {
    const Program::TupleSet &set = *each;
    bool amber = set.name == "sites_amber";
    os << "md.potential @spread_" << set.name
       << "(%x: !vec, %cell: !md.cell, %rel: !rel_" << set.name
       << ", %a: !of_" << set.name << ", %b: !of_" << set.name
       << ", %fs: !vec) -> f64 {\n"
       << "  %u = md.sum_tuples %rel, %x, %cell\n"
       << "      coordinates(displacement(2, 1), displacement(3, 1))\n"
       << "      gather(%fs : !vec) tuple(%a, %b : !of_" << set.name
       << ", !of_" << set.name << ") {\n"
       << "  ^bb0(%vs_j: vector<3xf64>, %vs_k: vector<3xf64>, "
          "%vs_f0: vector<3xf64>, %vs_f1: vector<3xf64>, "
          "%vs_f2: vector<3xf64>, %vs_f3: vector<3xf64>, "
          "%vs_a: f64, %vs_b: f64):\n";
    SiteKernel k(os, "    ");
    std::string placed = emitPlacement(k, amber);
    std::string work = k.dot("%vs_f0", placed);
    os << "    md.yield " << work << " : f64\n"
       << "  } : !rel_" << set.name << ", !vec -> f64\n"
       << "  md.return %u : f64\n}\n\n";
  }
}

std::string Builder::emitHostEvaluation(StringRef indent, StringRef x,
    StringRef cell, StringRef prefix, StringRef f, StringRef fResult,
    StringRef u, StringRef uResult, StringRef w, StringRef tag) {
  TermEvaluation point{indent, x, cell, prefix, f, fResult, u, uResult, w, tag};
  std::string virial = w.str();
  for (const TermBuilder *builder : getTermBuilders())
    if (const TermHostEvaluation *role = builder->getRoles().hostEvaluation)
      if (role->applies(*this))
        virial = role->emit(*this, os, point);
  return virial;
}

void Builder::emitTerms(StringRef x) {
  if (!system.topology)
    return;
  // The restraints, if any, last: their energy at the start is that of the
  // evaluation before the terms.
  // The terms given by expressions follow those of the topology, those
  // over tuples (D136) and then those over pairs (D137).
  const auto definitions = getTermDefinitions();
  unsigned storedTermCount = 0;
  for (const TermBuilder *builder : getTermBuilders())
    if (const auto *role = builder->getRoles().postEvaluation)
      storedTermCount += role->getStoredTermCount(*this);
  int size =
      definitions.size() + storedTermCount;
  std::string type = "memref<" + std::to_string(size) + "xf64>";
  os << "  %terms = memref.alloca() : " << type << "\n";
  int index = 0;
  for (const TermDefinition &definition : definitions) {
    StringRef name = definition.name;
    os << "  %" << name << " = md.evaluate @" << name << "(" << x << ", %cell"
       << getFieldValues() << getTimeValue("%time0") << ") request [energy]\n"
       << "      : (!vec, !md.cell" << getFieldTypes() << getTimeType()
       << ") -> f64\n"
       << "  %i_" << name << " = arith.constant " << index++ << " : index\n"
       << "  memref.store %" << name << ", %terms[%i_" << name << "] : "
       << type << "\n";
  }
  for (const TermBuilder *builder : getTermBuilders())
    if (const auto *role = builder->getRoles().postEvaluation)
      role->emitStoredTerms(*this, type, index);
  os << "  %terms_cast = memref.cast %terms : " << type << " to "
        "memref<?xf64>\n"
     << "  func.call @mdrtWriteTerms(%terms_cast) : (memref<?xf64>) -> ()\n";
}
