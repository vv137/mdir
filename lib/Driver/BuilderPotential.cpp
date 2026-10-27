// The potentials of the program: the terms of pairs, of tuples, and of
// the topology, the Born and external terms, the centroids, the pulls,
// the observables, the staged models, and the restraints
// (docs/driver.md, Section 2.12).

#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

/// The attribute of an md.sum_relation that truncates its energy at the
/// cutoff as `truncation` says, with the switching distance of `control`;
/// the power force switch is for the Lennard-Jones of a topology only, and
/// the reader rejects it elsewhere.
static std::string getTruncation(const Control &control,
                                 Truncation truncation) {
  double from = control.switchDistance * units::length;
  switch (truncation) {
  case Truncation::None:
  case Truncation::PowerForceSwitch:
  case Truncation::SquaredDistanceSwitch:
    return "";
  case Truncation::Shift:
    return " truncation(shift)";
  case Truncation::Switch:
    return " truncation(switch, from = " + formatReal(from) + ")";
  case Truncation::ForceSwitch:
    return " truncation(force_switch, from = " + formatReal(from) + ")";
  }
  llvm_unreachable("unknown truncation");
}

static std::string getTruncation(const Control &control) {
  return getTruncation(control, control.truncation);
}

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

void Builder::emitCentroidTerm(size_t index, const TupleTerm &term) {
  // Numbers outside the kernels, named after the term.
  std::string prefix = "%cb" + std::to_string(index) + "_";
  llvm::StringMap<std::string> values =
      emitCentroidCoordinates(term, "  ", "%x", "%cell", "%r_", prefix);
  for (auto [k, parameter] : llvm::enumerate(term.parameters)) {
    std::string name = prefix + "p" + std::to_string(k);
    os << "  " << name << " = arith.constant "
       << formatReal(parameter.second.front()) << " : f64\n";
    values[parameter.first] = name;
  }
  if (control.usesTime)
    values["t"] = "%time";
  bindLambdas(values);
  bindObserved(term.name, values);
  Expression expression = llvm::cantFail(
      Expression::parse(term.expression, control.functions));
  std::string energy = expression.emit(os, values, prefix + "e", "  ", term.name);
  os << "  " << prefix << "kj = arith.constant " << formatReal(units::energy)
     << " : f64\n"
     << "  %u_centroid_" << term.name << " = arith.mulf " << energy << ", "
     << prefix << "kj : f64\n";
}

void Builder::emitPullPotentials() {
  // The energy of each term as a function of its coordinates, which follow
  // the other arguments, so that the derivatives with respect to them give
  // the forces along the coordinates.
  std::vector<std::pair<size_t, std::string>> columns = getPullColumns();
  if (columns.empty())
    return;
  for (size_t index = 0, e = system.topology->tupleTerms.size(); index != e;
       ++index) {
    const TupleTerm &term = system.topology->tupleTerms[index];
    if (!term.isCentroid())
      continue;
    std::vector<std::string> quantities;
    for (const auto &[k, quantity] : columns)
      if (k == index)
        quantities.push_back(quantity);
    os << "md.potential @pullterm" << index << "(%x: !vec, %cell: !md.cell"
       << getFieldParameters() << getTimeParameter();
    for (size_t q = 0; q != quantities.size(); ++q)
      os << ", %pq" << q << ": f64";
    os << ") -> f64 {\n";
    emitLambdaConstants("  ");
    llvm::StringMap<std::string> values;
    for (auto [q, quantity] : llvm::enumerate(quantities))
      values[quantity] = "%pq" + std::to_string(q);
    for (auto [p, parameter] : llvm::enumerate(term.parameters)) {
      std::string name = "%pp" + std::to_string(p);
      os << "  " << name << " = arith.constant "
         << formatReal(parameter.second.front()) << " : f64\n";
      values[parameter.first] = name;
    }
    if (control.usesTime)
      values["t"] = "%time";
    bindLambdas(values);
    Expression expression = llvm::cantFail(
        Expression::parse(term.expression, control.functions));
    std::string energy = expression.emit(os, values, "%pe", "  ", term.name);
    os << "  md.return " << energy << " : f64\n}\n\n";
  }
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
      for (auto [k, pair] : llvm::enumerate(control.pairs))
        if (pair.name == term.name)
          term.kind = PairTerms, term.index = static_cast<int>(k);
      for (auto [k, tuple] : llvm::enumerate(system.topology->tupleTerms))
        if (tuple.name == term.name)
          term.kind = TupleTerms, term.index = static_cast<int>(k);
      for (auto [k, external] :
           llvm::enumerate(system.topology->externalTerms))
        if (external.name == term.name)
          term.kind = ExternalTerms, term.index = static_cast<int>(k);
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
    emitTopologyPotential("observe" + std::to_string(k), term.kind,
                          term.kind == TupleTerms ? term.index : -1,
                          term.kind == PairTerms ? term.index : -1,
                          term.kind == ExternalTerms ? term.index : -1);
    observedTerm.clear();
    observedConstants.clear();
  }
}

void Builder::emitBorn(unsigned terms,
                       llvm::function_ref<void(StringRef)> add) {
  // Every pair within the cutoff, those that the topology excludes as
  // well.
  double cutoff = control.cutoffDistance * units::length;
  os << "  %ngb = md.neighborhood %x, %cell cutoff(" << formatReal(cutoff)
     << ") : !vec -> !pairs\n";
  // With a cutoff R of the radii (D152), the descreening reaches R and the
  // largest screened radius beyond it, which System checked is within the
  // cutoff.
  double radiusCutoff = control.bornRadiusCutoff * units::length;
  std::string reach = "%ngb";
  if (radiusCutoff > 0.0) {
    double largest = 0.0;
    for (size_t i = 0, e = system.topology->bornRadii.size(); i != e; ++i)
      largest = std::max(largest, system.topology->bornScreens[i] *
                                      (system.topology->bornRadii[i] - 0.009));
    reach = "%ngb_radii";
    os << "  %ngb_radii = md.neighborhood %x, %cell cutoff("
       << formatReal(std::min(radiusCutoff + largest, cutoff))
       << ") : !vec -> !pairs\n";
  }
  // The integral of the descreening of i by the sphere of j of the radius
  // s_j [Hawkins1996]: with L = max(ρ̃_i, |r − s_j|) and U = r + s_j,
  //   ½ (1/L − 1/U + r/4 (1/U² − 1/L²) + ln(L/U) / (2r) + s_j²/(4r) (1/L² − 1/U²)),
  // and 1/ρ̃_i − 1/L more where i lies within the sphere of j, 0 where the
  // sphere does not reach it, L ≥ U. A cutoff R of the radii takes the
  // integral over the shells within R alone, U = min(r + s_j, R), so that
  // the radii do not jump where a sphere crosses it: the smooth cutoff
  // `rgbmax` of Amber's pmemd (gb_ene.F90), written here from the integral
  // rather than its series (D152).
  os << "  %gb_integral = md.gather_relation " << reach << ", %x, %cell gather("
        "%p_gb_offset, %p_gb_scaled : !real, !real)\n"
     << "      exchange(none) {\n"
     << "  ^bb0(%r: f64, %d: vector<3xf64>, %ri: f64, %rj: f64, %si: f64, "
        "%sj: f64):\n"
     << "    %zero = arith.constant 0.0 : f64\n"
     << "    %one = arith.constant 1.0 : f64\n"
     << "    %two = arith.constant 2.0 : f64\n"
     << "    %quarter = arith.constant 0.25 : f64\n"
     << "    %half = arith.constant 0.5 : f64\n"
     << "    %upper" << (radiusCutoff > 0.0 ? "_sphere" : "")
     << " = arith.addf %r, %sj : f64\n";
  if (radiusCutoff > 0.0)
    os << "    %rgb = arith.constant " << formatReal(radiusCutoff)
       << " : f64\n"
       << "    %upper = arith.minimumf %upper_sphere, %rgb : f64\n";
  os << "    %gap = arith.subf %r, %sj : f64\n"
     << "    %agap = math.absf %gap : f64\n"
     << "    %lower = arith.maximumf %ri, %agap : f64\n"
     << "    %l = arith.divf %one, %lower : f64\n"
     << "    %u = arith.divf %one, %upper : f64\n"
     << "    %l2 = arith.mulf %l, %l : f64\n"
     << "    %u2 = arith.mulf %u, %u : f64\n"
     << "    %t0 = arith.subf %l, %u : f64\n"
     << "    %du = arith.subf %u2, %l2 : f64\n"
     << "    %qr = arith.mulf %quarter, %r : f64\n"
     << "    %t1 = arith.mulf %qr, %du : f64\n"
     << "    %ratio = arith.divf %u, %l : f64\n"
     << "    %log = math.log %ratio : f64\n"
     << "    %hr = arith.divf %half, %r : f64\n"
     << "    %t2 = arith.mulf %hr, %log : f64\n"
     << "    %sj2 = arith.mulf %sj, %sj : f64\n"
     << "    %qs = arith.mulf %quarter, %sj2 : f64\n"
     << "    %qsr = arith.divf %qs, %r : f64\n"
     << "    %dl = arith.subf %l2, %u2 : f64\n"
     << "    %t3 = arith.mulf %qsr, %dl : f64\n"
     << "    %a0 = arith.addf %t0, %t1 : f64\n"
     << "    %a1 = arith.addf %a0, %t2 : f64\n"
     << "    %a2 = arith.addf %a1, %t3 : f64\n"
     << "    %inner = arith.subf %sj, %r : f64\n"
     << "    %within = arith.cmpf olt, %ri, %inner : f64\n"
     << "    %iri = arith.divf %one, %ri : f64\n"
     << "    %e0 = arith.subf %iri, %l : f64\n"
     << "    %e1 = arith.mulf %two, %e0 : f64\n"
     << "    %a3 = arith.addf %a2, %e1 : f64\n"
     << "    %a4 = arith.select %within, %a3, %a2 : f64\n"
     << "    %reach = arith.cmpf olt, %lower, %upper : f64\n"
     << "    %k = arith.select %reach, %a4, %zero : f64\n"
     << "    md.yield %k : f64\n"
     << "  } : !pairs, !vec -> !real\n";
  // The Born radii of Hawkins, Cramer, and Truhlar [Hawkins1996],
  // B = 1 / (1/ρ̃ − I/2) (the factor ½ of the integral).
  if (control.implicitSolvent == Control::ImplicitSolvent::HCT)
    os << "  %gb_born = md.map_particles gather(%gb_integral, %p_gb_offset "
          ": !real, !real) {\n"
       << "  ^bb0(%i: f64, %ri: f64):\n"
       << "    %half = arith.constant 0.5 : f64\n"
       << "    %one = arith.constant 1.0 : f64\n"
       << "    %hi = arith.mulf %half, %i : f64\n"
       << "    %iri = arith.divf %one, %ri : f64\n"
       << "    %den = arith.subf %iri, %hi : f64\n"
       << "    %bi = arith.divf %one, %den : f64\n"
       << "    md.yield %bi : f64\n"
       << "  } : !real\n";
  // Those of Onufriev, Bashford, and Case [Onufriev2004]: with ψ = I ρ̃ / 2,
  // B = 1 / (1/ρ̃ − tanh(α ψ − β ψ² + γ ψ³) / ρ).
  bool first = control.implicitSolvent == Control::ImplicitSolvent::OBC1;
  double alpha = first ? 0.8 : 1.0, beta = first ? 0.0 : 0.8,
         gamma = first ? 2.909125 : 4.85;
  if (control.implicitSolvent != Control::ImplicitSolvent::HCT)
    os << "  %gb_born = md.map_particles gather(%gb_integral, %p_gb_offset, "
          "%p_gb_radius : !real, !real, !real) {\n"
       << "  ^bb0(%i: f64, %ri: f64, %r0: f64):\n"
       << "    %half = arith.constant 0.5 : f64\n"
       << "    %one = arith.constant 1.0 : f64\n"
       << "    %alpha = arith.constant " << formatReal(alpha) << " : f64\n"
       << "    %beta = arith.constant " << formatReal(beta) << " : f64\n"
       << "    %gamma = arith.constant " << formatReal(gamma) << " : f64\n"
       << "    %hi = arith.mulf %half, %i : f64\n"
       << "    %psi = arith.mulf %hi, %ri : f64\n"
       << "    %psi2 = arith.mulf %psi, %psi : f64\n"
       << "    %psi3 = arith.mulf %psi2, %psi : f64\n"
       << "    %a = arith.mulf %alpha, %psi : f64\n"
       << "    %b = arith.mulf %beta, %psi2 : f64\n"
       << "    %c = arith.mulf %gamma, %psi3 : f64\n"
       << "    %s0 = arith.subf %a, %b : f64\n"
       << "    %s1 = arith.addf %s0, %c : f64\n"
       << "    %t = math.tanh %s1 : f64\n"
       << "    %tr = arith.divf %t, %r0 : f64\n"
       << "    %iri = arith.divf %one, %ri : f64\n"
       << "    %den = arith.subf %iri, %tr : f64\n"
       << "    %bi = arith.divf %one, %den : f64\n"
       << "    md.yield %bi : f64\n"
       << "  } : !real\n";
  if (terms & GeneralizedBorn) {
    // −τ f q_i q_j / f_GB over the pairs and −τ f q_i² / (2 B_i) for each
    // particle, f_GB = sqrt(r² + B_i B_j exp(−r² / (4 B_i B_j))) and
    // τ = 1/ε_solute − 1/ε_solvent. A salt screens the solvent with the
    // Debye length 1/κ, τ = 1/ε_solute − exp(−κ f_GB)/ε_solvent, with κ
    // scaled by 0.73 for the layer around the solute that excludes the ions
    // [Srinivasan1999] (D152); f_GB is B_i for a particle.
    double tau = 1.0 / control.soluteDielectric -
                 1.0 / control.solventDielectric;
    double kappa = getDebyeKappa();
    // Emits `%ct`, f τ with the factor `factor` of f, for the distance
    // `distance`.
    auto emitScreening = [&](double factor, StringRef distance) {
      if (kappa == 0.0) {
        os << "    %ct = arith.constant "
           << formatReal(-factor * tau * coulombInternal) << " : f64\n";
        return;
      }
      os << "    %cf = arith.constant " << formatReal(-factor * coulombInternal)
         << " : f64\n"
         << "    %ein = arith.constant "
         << formatReal(1.0 / control.soluteDielectric) << " : f64\n"
         << "    %eout = arith.constant "
         << formatReal(1.0 / control.solventDielectric) << " : f64\n"
         << "    %nkappa = arith.constant " << formatReal(-kappa) << " : f64\n"
         << "    %kd = arith.mulf %nkappa, " << distance << " : f64\n"
         << "    %ekd = math.exp %kd : f64\n"
         << "    %screened = arith.mulf %ekd, %eout : f64\n"
         << "    %tau = arith.subf %ein, %screened : f64\n"
         << "    %ct = arith.mulf %cf, %tau : f64\n";
    };
    os << "  %u_gb_pairs = md.sum_relation %ngb, %x, %cell gather(%p_q, "
          "%gb_born : !real, !real)\n"
       << "      exchange(symmetric) {\n"
       << "  ^bb0(%r: f64, %d: vector<3xf64>, %qi: f64, %qj: f64, %bi: f64, "
          "%bj: f64):\n"
       << "    %quarter = arith.constant 0.25 : f64\n"
       << "    %bb = arith.mulf %bi, %bj : f64\n"
       << "    %r2 = arith.mulf %r, %r : f64\n"
       << "    %bb4 = arith.divf %quarter, %bb : f64\n"
       << "    %x0 = arith.mulf %r2, %bb4 : f64\n"
       << "    %nx = arith.negf %x0 : f64\n"
       << "    %ex = math.exp %nx : f64\n"
       << "    %be = arith.mulf %bb, %ex : f64\n"
       << "    %s = arith.addf %r2, %be : f64\n"
       << "    %fgb = math.sqrt %s : f64\n";
    emitScreening(1.0, "%fgb");
    os << "    %qq = arith.mulf %qi, %qj : f64\n"
       << "    %cq = arith.mulf %ct, %qq : f64\n"
       << "    %e = arith.divf %cq, %fgb : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !pairs, !vec -> f64\n"
       << "  %u_gb_self = md.sum_particles gather(%p_q, %gb_born : !real, "
          "!real) {\n"
       << "  ^bb0(%qi: f64, %bi: f64):\n";
    emitScreening(0.5, "%bi");
    os << "    %qq = arith.mulf %qi, %qi : f64\n"
       << "    %cq = arith.mulf %ct, %qq : f64\n"
       << "    %e = arith.divf %cq, %bi : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : f64\n"
       << "  %u_born = arith.addf %u_gb_pairs, %u_gb_self : f64\n";
    add("born");
  }
  if ((terms & Surface) && control.surfaceAreaEnergy > 0.0) {
    // The nonpolar term of Schaefer et al., 4π γ (ρ + 0.14 nm)² (ρ/B)⁶,
    // with γ from kcal/(mol Å²) in kJ/(mol nm²).
    double tension = control.surfaceAreaEnergy * units::energy /
                     (units::length * units::length);
    os << "  %u_surface = md.sum_particles gather(%p_gb_radius, %gb_born : "
          "!real, !real) {\n"
       << "  ^bb0(%r0: f64, %bi: f64):\n"
       << "    %c = arith.constant " << formatReal(4.0 * M_PI * tension)
       << " : f64\n"
       << "    %probe = arith.constant 0.14 : f64\n"
       << "    %six = arith.constant 6 : i32\n"
       << "    %rp = arith.addf %r0, %probe : f64\n"
       << "    %rp2 = arith.mulf %rp, %rp : f64\n"
       << "    %ratio = arith.divf %r0, %bi : f64\n"
       << "    %r6 = math.fpowi %ratio, %six : f64, i32\n"
       << "    %a = arith.mulf %c, %rp2 : f64\n"
       << "    %e = arith.mulf %a, %r6 : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : f64\n";
    add("surface");
  }
}

void Builder::emitExternalTerm(size_t index, const ExternalTerm &term,
                                llvm::function_ref<void(StringRef)> add) {
  // Σ_i k(x_i) over the particles of the term, the custom external force of
  // OpenMM [Eastman2017] (D148), the flag `ext<k>` 1 for
  // them and 0 for the others, which the kernel selects on: the positions
  // in Å, the energy in kcal/mol. The positions are never wrapped (D74), so
  // that the term is continuous along a trajectory.
  Expression expression = llvm::cantFail(
      Expression::parse(term.expression, control.functions));
  std::vector<std::string> names = expression.getNames();
  bool charges = llvm::is_contained(names, "q");
  std::string flag = "ext" + std::to_string(index);
  std::string prefix = "%xe" + std::to_string(index) + "_";
  // In the frame of the cell (D154), the positions scaled by the edges of
  // the cell of the input over those of the cell.
  bool frame = term.scaling == ExternalTerm::Scaling::Cell;
  if (frame) {
    os << "  " << prefix << "edges = md_exec.cell_edges %cell : vector<3xf64>\n"
       << "  " << prefix << "input = arith.constant dense<[";
    for (int k = 0; k != 3; ++k)
      os << (k ? ", " : "")
         << formatReal(system.inputBox[k] > 0.0 ? system.inputBox[k]
                                                 : system.box[k]);
    os << "]> : vector<3xf64>\n"
       << "  " << prefix << "frame = arith.divf " << prefix << "input, "
       << prefix << "edges : vector<3xf64>\n";
  }
  os << "  %u_external_" << term.name << " = md.sum_particles gather(%x, %p_"
     << flag;
  if (charges)
    os << ", %p_q";
  for (const auto &parameter : term.parameters)
    os << ", %p_" << flag << "_" << parameter.first;
  os << " : !vec, !real";
  if (charges)
    os << ", !real";
  for (size_t k = 0; k != term.parameters.size(); ++k)
    os << ", !real";
  os << ") {\n  ^bb0(" << prefix << "pos: vector<3xf64>, " << prefix
     << "on: f64";
  if (charges)
    os << ", " << prefix << "q: f64";
  for (size_t k = 0; k != term.parameters.size(); ++k)
    os << ", " << prefix << "p" << k << ": f64";
  os << "):\n";
  llvm::StringMap<std::string> values;
  os << "    " << prefix << "a = arith.constant "
     << formatReal(1.0 / units::length) << " : f64\n";
  std::string position = prefix + "pos";
  if (frame) {
    os << "    " << prefix << "posf = arith.mulf " << prefix << "pos, "
       << prefix << "frame : vector<3xf64>\n";
    position = prefix + "posf";
  }
  for (int c = 0; c != 3; ++c) {
    std::string component = prefix + "xyz"[c];
    os << "    " << component << "n = vector.extract " << position << "["
       << c << "] : f64 from vector<3xf64>\n"
       << "    " << component << " = arith.mulf " << component << "n, "
       << prefix << "a : f64\n";
    values[std::string(1, "xyz"[c])] = component;
  }
  if (charges)
    values["q"] = prefix + "q";
  for (auto [k, parameter] : llvm::enumerate(term.parameters))
    values[parameter.first] = prefix + "p" + std::to_string(k);
  for (auto [k, constant] : llvm::enumerate(term.constants)) {
    std::string name = prefix + "c" + std::to_string(k);
    os << "    " << name << " = arith.constant "
       << formatReal(constant.second) << " : f64\n";
    values[constant.first] = name;
  }
  if (control.usesTime)
    values["t"] = "%time";
  bindLambdas(values);
  bindObserved(term.name, values);
  std::string energy = expression.emit(os, values, prefix + "e", "    ", term.name);
  os << "    " << prefix << "kj = arith.constant " << formatReal(units::energy)
     << " : f64\n"
     << "    " << prefix << "u = arith.mulf " << energy << ", " << prefix
     << "kj : f64\n"
     << "    " << prefix << "zero = arith.constant 0.0 : f64\n"
     << "    " << prefix << "in = arith.cmpf one, " << prefix << "on, "
     << prefix << "zero : f64\n"
     << "    " << prefix << "k = arith.select " << prefix << "in, " << prefix
     << "u, " << prefix << "zero : f64\n"
     << "    md.yield " << prefix << "k : f64\n"
     << "  } : f64\n";
  add("external_" + term.name);
}

/// A part of the splitting of the prolate kernel at `x` = r / r_c, from the
/// table `table` (prolate::getSplittingTable): the polynomial of the cell
/// of x in t = 2 (x cells − k) − 1, the cell clamped to the table. The
/// place in the cell carries the derivative; the index and the lookups do
/// not, as in a tabulated function of the control file (D138).
static std::string emitProlateSplitting(llvm::raw_ostream &os,
                                        StringRef table, StringRef x,
                                        StringRef p, Precision precision) {
  auto [prolateCells, prolateTerms] = getProlateTableShape(precision);
  os << "    " << p << "cells = arith.constant "
     << formatReal(static_cast<double>(prolateCells)) << " : f64\n"
     << "    " << p << "top = arith.constant "
     << formatReal(static_cast<double>(prolateCells - 1)) << " : f64\n"
     << "    " << p << "nil = arith.constant 0.0 : f64\n"
     << "    " << p << "one = arith.constant 1.0 : f64\n"
     << "    " << p << "two = arith.constant 2.0 : f64\n"
     << "    " << p << "u = arith.mulf " << x << ", " << p << "cells : f64\n"
     << "    " << p << "floor = math.floor " << p << "u : f64\n"
     << "    " << p << "low = arith.maximumf " << p << "floor, " << p
     << "nil : f64\n"
     << "    " << p << "place = arith.minimumf " << p << "low, " << p
     << "top : f64\n"
     << "    " << p << "s = arith.subf " << p << "u, " << p << "place : f64\n"
     << "    " << p << "s2 = arith.mulf " << p << "two, " << p << "s : f64\n"
     << "    " << p << "t = arith.subf " << p << "s2, " << p << "one : f64\n"
     << "    " << p << "row = arith.fptosi " << p << "place : f64 to i32\n";
  for (int64_t j = 0; j != prolateTerms; ++j)
    os << "    " << p << "col" << j << " = arith.constant " << j << " : i32\n"
       << "    " << p << "a" << j << " = md.lookup %t_" << table << "[" << p
       << "row, " << p << "col" << j << "] : !grid, i32, i32 -> f64\n";
  std::string value = (p + "a" + llvm::Twine(prolateTerms - 1)).str();
  for (int64_t j = prolateTerms - 2; j >= 0; --j) {
    os << "    " << p << "m" << j << " = arith.mulf " << value << ", " << p
       << "t : f64\n"
       << "    " << p << "h" << j << " = arith.addf " << p << "m" << j << ", "
       << p << "a" << j << " : f64\n";
    value = (p + "h" + llvm::Twine(j)).str();
  }
  return value;
}

void Builder::emitTopologyPotential(StringRef name, unsigned terms,
                                    int tupleTerm, int pairTerm,
                                    int externalTerm) {
  double cutoff = control.cutoffDistance * units::length;
  // A program of the dependent terms has no other
  // (D243).
  if (dependent) {
    unsigned kept = 0;
    if (dependent->lennardJones)
      kept |= LennardJones | LennardJonesExcluded | LennardJonesReciprocal;
    if (dependent->coulomb)
      kept |= Coulomb | Coulomb14 | CoulombExcluded | CoulombReciprocal |
              CoulombWithin;
    if (!dependent->pairs.empty())
      kept |= PairTerms;
    if (!dependent->tuples.empty())
      kept |= TupleTerms;
    terms &= kept | Alchemical;
  }
  // The truncation of the pair terms, shifted in `@alchemical` and
  // `@observe<k>` (D210).
  Truncation truncation = getPairTruncation();
  auto has = [&](StringRef set) {
    return llvm::any_of(program.tupleSets, [&](const Program::TupleSet &s) {
      return s.name == set;
    });
  };
  // 4ε (σ¹² Φ₁₂(r) − σ⁶ Φ₆(r)), with Φₙ = r⁻ⁿ truncated as `modifier`
  // says, with rc the cutoff and rs the switching distance:
  //
  //   none             r⁻ⁿ
  //   potential shift  r⁻ⁿ − rc⁻ⁿ
  //   force switch     r⁻ⁿ − Cₙ − [r > rs] (Aₙ/3 (r − rs)³ + Bₙ/4 (r − rs)⁴),
  //                    a cubic polynomial added to the force, so that it and
  //                    its derivative vanish at rc [GromacsManual2025]
  //   power force      r⁻ⁿ − (rs rc)^(−n/2)               for r <= rs
  //   switch           kₙ (r^(−n/2) − rc^(−n/2))²          otherwise,
  //                    kₙ = rc^(n/2) / (rc^(n/2) − rs^(n/2)): the force of
  //                    each power times a switch linear in r^(n/2)
  //                    [Steinbach1994]
  //
  // σⁿ Φₙ is written with s = σ/r and g = σ, so that no power of r alone
  // leaves the range of f32.
  double from = control.switchDistance * units::length;
  // `distance` is r, or the soft-core distance of [free_energy] (D161).
  auto emitLennardJones = [&](StringRef sigma, StringRef epsilon,
                              StringRef result, Truncation modifier,
                              StringRef distance = "%r") {
    os << "    %c4 = arith.constant 4.0 : f64\n"
       << "    %e4 = arith.mulf %c4, " << epsilon << " : f64\n"
       << "    %sr = arith.divf " << sigma << ", " << distance << " : f64\n";
    if (modifier == Truncation::None ||
        modifier == Truncation::SquaredDistanceSwitch) {
      os << "    %i6 = arith.constant 6 : i32\n"
         << "    %s6 = math.fpowi %sr, %i6 : f64, i32\n"
         << "    %s12 = arith.mulf %s6, %s6 : f64\n"
         << "    %t = arith.subf %s12, %s6 : f64\n"
         << "    " << (modifier == Truncation::SquaredDistanceSwitch
                            ? StringRef("%lj_raw") : result)
         << " = arith.mulf %e4, %t : f64\n";
      if (modifier == Truncation::SquaredDistanceSwitch) {
        // The cubic switching potential in r² [Brooks1983]. Write it
        // as (1-y)²(1+2y), y=(r²-rs²)/(rc²-rs²), to avoid cancellation
        // near rc. Clamping y gives the constant branches and a zero
        // derivative at both ends. AD differentiates the full product.
        os << "    %sw_rs2 = arith.constant " << formatReal(from * from)
           << " : f64\n"
           << "    %sw_width_inv = arith.constant "
           << formatReal(1.0 / (cutoff * cutoff - from * from)) << " : f64\n"
           << "    %sw_zero = arith.constant 0.0 : f64\n"
           << "    %sw_one = arith.constant 1.0 : f64\n"
           << "    %sw_two = arith.constant 2.0 : f64\n"
           << "    %sw_r2 = arith.mulf %r, %r : f64\n"
           << "    %sw_dr2 = arith.subf %sw_r2, %sw_rs2 : f64\n"
           << "    %sw_y_raw = arith.mulf %sw_dr2, %sw_width_inv : f64\n"
           << "    %sw_below = arith.cmpf olt, %sw_y_raw, %sw_zero : f64\n"
           << "    %sw_y_lo = arith.select %sw_below, %sw_zero, %sw_y_raw : f64\n"
           << "    %sw_above = arith.cmpf ogt, %sw_y_lo, %sw_one : f64\n"
           << "    %sw_y = arith.select %sw_above, %sw_one, %sw_y_lo : f64\n"
           << "    %sw_complement = arith.subf %sw_one, %sw_y : f64\n"
           << "    %sw_square = arith.mulf %sw_complement, %sw_complement : f64\n"
           << "    %sw_twice = arith.mulf %sw_two, %sw_y : f64\n"
           << "    %sw_factor = arith.addf %sw_one, %sw_twice : f64\n"
           << "    %sw_s = arith.mulf %sw_square, %sw_factor : f64\n"
           << "    " << result << " = arith.mulf %lj_raw, %sw_s : f64\n";
      }
      return;
    }
    os << "    %i3 = arith.constant 3 : i32\n"
       << "    %s3 = math.fpowi %sr, %i3 : f64, i32\n"
       << "    %s6 = arith.mulf %s3, %s3 : f64\n"
       << "    %s12 = arith.mulf %s6, %s6 : f64\n"
       << "    %g3 = math.fpowi " << sigma << ", %i3 : f64, i32\n"
       << "    %g6 = arith.mulf %g3, %g3 : f64\n"
       << "    %g12 = arith.mulf %g6, %g6 : f64\n";
    auto constant = [&](StringRef name, double value) {
      os << "    %" << name << " = arith.constant " << formatReal(value)
         << " : f64\n";
    };
    // a = σ¹² Φ₁₂ and b = σ⁶ Φ₆, from what each power loses, g¹² c₁₂ and
    // g⁶ c₆.
    auto emitSubtract = [&](StringRef c12, StringRef c6) {
      os << "    %gc12 = arith.mulf %g12, " << c12 << " : f64\n"
         << "    %a = arith.subf %s12, %gc12 : f64\n"
         << "    %gc6 = arith.mulf %g6, " << c6 << " : f64\n"
         << "    %b = arith.subf %s6, %gc6 : f64\n";
    };
    std::string a = "%a", b = "%b";
    if (modifier == Truncation::Shift) {
      constant("rc12", std::pow(cutoff, -12.0));
      constant("rc6", std::pow(cutoff, -6.0));
      emitSubtract("%rc12", "%rc6");
    } else if (modifier == Truncation::ForceSwitch) {
      double width = cutoff - from;
      os << "    %rs = arith.constant " << formatReal(from) << " : f64\n"
         << "    %zero = arith.constant 0.0 : f64\n"
         << "    %dr = arith.subf " << distance << ", %rs : f64\n"
         << "    %beyond = arith.cmpf ogt, " << distance << ", %rs : f64\n"
         << "    %u = arith.select %beyond, %dr, %zero : f64\n"
         << "    %u2 = arith.mulf %u, %u : f64\n"
         << "    %u3 = arith.mulf %u2, %u : f64\n";
      for (int n : {12, 6}) {
        double an = -n * ((n + 4) * cutoff - (n + 1) * from) /
                    (std::pow(cutoff, n + 2) * width * width);
        double bn = n * ((n + 3) * cutoff - (n + 1) * from) /
                    (std::pow(cutoff, n + 2) * width * width * width);
        double cn = std::pow(cutoff, -n) - an / 3.0 * std::pow(width, 3) -
                    bn / 4.0 * std::pow(width, 4);
        std::string k = std::to_string(n);
        constant("a" + k, an / 3.0);
        constant("b" + k, bn / 4.0);
        constant("c" + k, cn);
        // Cₙ + u³ (Aₙ/3 + (Bₙ/4) u)
        os << "    %pb" << k << " = arith.mulf %b" << k << ", %u : f64\n"
           << "    %pa" << k << " = arith.addf %a" << k << ", %pb" << k
           << " : f64\n"
           << "    %pu" << k << " = arith.mulf %u3, %pa" << k << " : f64\n"
           << "    %p" << k << " = arith.addf %c" << k << ", %pu" << k
           << " : f64\n";
      }
      emitSubtract("%p12", "%p6");
    } else {
      // Below rs the shifted powers; above, the switched squares.
      constant("in12", std::pow(from * cutoff, -6.0));
      constant("in6", std::pow(from * cutoff, -3.0));
      emitSubtract("%in12", "%in6");
      constant("k12", std::pow(cutoff, 6.0) /
                          (std::pow(cutoff, 6.0) - std::pow(from, 6.0)));
      constant("k6", std::pow(cutoff, 3.0) /
                         (std::pow(cutoff, 3.0) - std::pow(from, 3.0)));
      constant("rc6", std::pow(cutoff, -6.0));
      constant("rc3", std::pow(cutoff, -3.0));
      os << "    %gr6 = arith.mulf %g6, %rc6 : f64\n"
         << "    %h6 = arith.subf %s6, %gr6 : f64\n"
         << "    %h6s = arith.mulf %h6, %h6 : f64\n"
         << "    %a_out = arith.mulf %k12, %h6s : f64\n"
         << "    %gr3 = arith.mulf %g3, %rc3 : f64\n"
         << "    %h3 = arith.subf %s3, %gr3 : f64\n"
         << "    %h3s = arith.mulf %h3, %h3 : f64\n"
         << "    %b_out = arith.mulf %k6, %h3s : f64\n"
         << "    %rs = arith.constant " << formatReal(from) << " : f64\n"
         << "    %inside = arith.cmpf ole, " << distance << ", %rs : f64\n"
         << "    %a_sel = arith.select %inside, %a, %a_out : f64\n"
         << "    %b_sel = arith.select %inside, %b, %b_out : f64\n";
      a = "%a_sel";
      b = "%b_sel";
    }
    os << "    %t = arith.subf " << a << ", " << b << " : f64\n"
       << "    " << result << " = arith.mulf %e4, %t : f64\n";
  };

  os << "md.potential @" << name << "(%x: !vec, %cell: !md.cell"
     << getFieldParameters() << getTimeParameter() << getLambdaParameters()
     << getObservedParameters() << getGradientParameters()
     << ") -> f64 {\n";
  emitLambdaConstants("  ");
  // [free_energy] (D161): `@alchemical` has only what depends on λ.
  bool alchemicalOnly = terms & Alchemical;
  if (alchemicalOnly) {
    unsigned kept = LennardJones | Coulomb | CoulombExcluded |
                    CoulombWithin | CoulombReciprocal | TupleTerms |
                    PairTerms | ExternalTerms;
    if (!decouples())
      kept &= ~(LennardJones | Coulomb | CoulombExcluded | CoulombWithin |
                CoulombReciprocal);
    terms &= kept;
  }
  // Which λ the kernels of the pairs take: as arguments, or as constants
  // of the state of the run that are not 0.
  bool scalesCoulomb =
      decouples() && (lambdaArguments || getLambda("coulomb") != 0.0);
  bool scalesVdw = decouples() && (lambdaArguments || getLambda("vdw") != 0.0);
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

  // Lennard-Jones and Coulomb, both cut at the cutoff, over the pairs that
  // are not excluded: as the modifiers say, and in `@alchemical` and
  // `@observe<k>` shifted to 0 at the cutoff (D210).
  bool lj = terms & LennardJones, coulomb = terms & Coulomb;
  bool pairTerms = (terms & PairTerms) && !control.pairs.empty();
  if (lj || coulomb || pairTerms) {
    os << "  %n = md.neighborhood %x, %cell cutoff(" << formatReal(cutoff)
       << ")";
    if (has("excluded"))
      os << " exclude(%r_excluded : !rel_excluded)";
    os << " : !vec -> !pairs\n";
  }
  // The pair terms given by expressions (D137), over the pairs of the
  // topology that are not excluded, truncated at the cutoff as the
  // Lennard-Jones is, in the units of the control file: r, sigma in Å, q
  // in e, epsilon and the energy in kcal/mol. The reader has tested that
  // the expression is symmetric in the two particles, which the kernel
  // then asserts; sigma and
  // epsilon those of the pair, with the pairs of types set apart, and
  // sigma1, epsilon1 those of the type of each particle with itself.
  for (auto [k, term] : llvm::enumerate(control.pairs)) {
    if (!pairTerms || (pairTerm >= 0 && static_cast<int>(k) != pairTerm))
      continue;
    if (dependent && !dependent->hasPair(static_cast<unsigned>(k)))
      continue;
    if (alchemicalOnly && !usesLambda(term.expression))
      continue;
    bool grouped = k < system.pairGroups.size() && !system.pairGroups[k].empty();
    std::string g = std::to_string(k), set = "pair_" + term.name;
    Expression expression = llvm::cantFail(Expression::parse(term.expression, control.functions));
    const std::vector<std::string> &used = expression.getNames();
    auto uses = [&](StringRef name) { return llvm::is_contained(used, name); };
    // The parameters of each particle that the term takes (D165).
    std::vector<std::string> stems;
    for (const auto &[name, values] : system.particleParameters)
      if (uses(name + "1") || uses(name + "2"))
        stems.push_back(name);
    os << "  %u_" << set << " = md.sum_relation %n, %x, %cell gather(%p_type, "
       << "%p_q" << (grouped ? ", %p_pg" + g + "a, %p_pg" + g + "b" : "");
    for (const std::string &stem : stems)
      os << ", %p_pp_" << stem;
    os << " : !ids, !real" << (grouped ? ", !real, !real" : "");
    for (size_t n = 0; n != stems.size(); ++n)
      os << ", !real";
    os << ")\n"
       << "      exchange(symmetric, asserted)"
       << getTruncation(control, truncation)
       << " {\n"
       << "  ^bb0(%r: f64, %d: vector<3xf64>, %type_i: i32, %type_j: i32, "
       << "%q_i: f64, %q_j: f64"
       << (grouped ? ", %ga_i: f64, %ga_j: f64, %gb_i: f64, %gb_j: f64" : "");
    for (const std::string &stem : stems)
      os << ", %pp_" << stem << "_i: f64, %pp_" << stem << "_j: f64";
    os << "):\n"
       << "    %pt_angstrom = arith.constant " << formatReal(1.0 / units::length)
       << " : f64\n"
       << "    %pt_kcal = arith.constant " << formatReal(1.0 / units::energy)
       << " : f64\n"
       << "    %pt_r = arith.mulf %r, %pt_angstrom : f64\n";
    llvm::StringMap<std::string> values;
    values["r"] = "%pt_r";
    if (control.usesTime)
      values["t"] = "%time";
    bindLambdas(values);
    values["q1"] = "%q_i";
    values["q2"] = "%q_j";
    for (const std::string &stem : stems) {
      values[stem + "1"] = "%pp_" + stem + "_i";
      values[stem + "2"] = "%pp_" + stem + "_j";
    }
    auto lookup = [&](StringRef variable, StringRef table, StringRef a,
                      StringRef b, StringRef factor) {
      if (!uses(variable))
        return;
      std::string raw = ("%pt_" + variable + "_raw").str();
      std::string value = ("%pt_" + variable).str();
      os << "    " << raw << " = md.lookup %t_" << table << "[" << a << ", "
         << b << "] : !table, i32, i32 -> f64\n"
         << "    " << value << " = arith.mulf " << raw << ", " << factor
         << " : f64\n";
      values[variable] = value;
    };
    lookup("sigma", "lj_sigma", "%type_i", "%type_j", "%pt_angstrom");
    lookup("epsilon", "lj_epsilon", "%type_i", "%type_j", "%pt_kcal");
    lookup("sigma1", "lj_sigma", "%type_i", "%type_i", "%pt_angstrom");
    lookup("sigma2", "lj_sigma", "%type_j", "%type_j", "%pt_angstrom");
    lookup("epsilon1", "lj_epsilon", "%type_i", "%type_i", "%pt_kcal");
    lookup("epsilon2", "lj_epsilon", "%type_j", "%type_j", "%pt_kcal");
    if (uses("coulomb")) {
      os << "    %pt_coulomb = arith.constant " << formatReal(coulombConstant)
         << " : f64\n";
      values["coulomb"] = "%pt_coulomb";
    }
    for (auto [c, constant] : llvm::enumerate(term.constants)) {
      std::string value = "%pt_c" + std::to_string(c);
      // A tunable constant is a value of the program's table, which an
      // update changes without compiling (D213).
      auto tunable = llvm::find(control.tunableConstants,
                                std::pair<unsigned, std::string>(
                                    static_cast<unsigned>(k), constant.first));
      if (tunable != control.tunableConstants.end() && gradientArguments) {
        values[constant.first] =
            "%tg_c" +
            std::to_string(tunable - control.tunableConstants.begin());
        continue;
      }
      if (tunable != control.tunableConstants.end()) {
        os << "    " << value << "_row = arith.constant 0 : i32\n"
           << "    " << value << "_column = arith.constant "
           << (tunable - control.tunableConstants.begin()) << " : i32\n"
           << "    " << value << " = md.lookup %t_tunable_constants[" << value
           << "_row, " << value << "_column] : !grid, i32, i32 -> f64\n";
      } else {
        os << "    " << value << " = arith.constant "
           << formatReal(constant.second) << " : f64\n";
      }
      values[constant.first] = value;
    }
    bindObserved(term.name, values);
    std::string energy = expression.emit(os, values, "%pte", "    ", term.name);
    os << "    %pt_kj = arith.constant " << formatReal(units::energy)
       << " : f64\n"
       << "    %pt_e = arith.mulf " << energy << ", %pt_kj : f64\n";
    std::string value = "%pt_e";
    if (grouped) {
      // A pair of a particle of each group, counted once even if both are
      // in both.
      os << "    %pt_ab = arith.mulf %ga_i, %gb_j : f64\n"
         << "    %pt_ba = arith.mulf %ga_j, %gb_i : f64\n"
         << "    %pt_either = arith.addf %pt_ab, %pt_ba : f64\n"
         << "    %pt_one = arith.constant 1.0 : f64\n"
         << "    %pt_mask = arith.minimumf %pt_either, %pt_one : f64\n"
         << "    %pt_masked = arith.mulf %pt_e, %pt_mask : f64\n";
      value = "%pt_masked";
    }
    os << "    md.yield " << value << " : f64\n"
       << "  } : !pairs, !vec -> f64\n";
    add(set);
  }
  if ((terms & (GeneralizedBorn | Surface)) &&
      control.implicitSolvent != Control::ImplicitSolvent::None)
    emitBorn(terms, add);
  if (terms & ExternalTerms)
    for (auto [index, term] : llvm::enumerate(system.topology->externalTerms))
      if ((externalTerm < 0 || static_cast<int>(index) == externalTerm) &&
          !(alchemicalOnly && !usesLambda(term.expression)))
        emitExternalTerm(index, term, add);
  if (lj || coulomb) {
    // [free_energy] (D161): a pair with one particle in the selection,
    // `%al_x` = 1, is decoupled. Its Lennard-Jones is that of the
    // soft-core distance r_A, r_A^6 = alpha sigma^6 lambda^p + r^6, times
    // 1 - lambda [Beutler1994], and its Coulomb is times 1 - lambda: the
    // charge of the particle in the selection scaled. The pairs within the
    // selection stay.
    bool flags = (lj && scalesVdw) || (coulomb && scalesCoulomb) ||
                 (alchemicalOnly && decouples());
    // In `@tunable`, σ and ε of a pair take the seeds of the derivatives
    // in their tables (D230, Control::TunableTableSeed).
    size_t seeds =
        gradientArguments && lj ? control.tunableTableSeeds.size() : 0;
    std::string seedFields, seedTypes, seedArguments;
    for (size_t c = 0; c != seeds; ++c) {
      seedFields += ", %tg_d" + std::to_string(c);
      seedTypes += ", !real";
      seedArguments += ", %tgd" + std::to_string(c) + "_i: f64, %tgd" +
                       std::to_string(c) + "_j: f64";
    }
    os << "  %u_nonbonded = md.sum_relation %n, %x, %cell gather(%p_type, "
          "%p_q"
       << (flags ? ", %p_alch" : "") << seedFields << " : !ids, !real"
       << (flags ? ", !real" : "") << seedTypes << ")\n"
       << "      exchange(symmetric" << (seeds ? ", asserted" : "") << ") {\n"
       << "  ^bb0(%r: f64, %d: vector<3xf64>, %type_i: i32, %type_j: i32, "
          "%q_i: f64, %q_j: f64"
       << (flags ? ", %al_i: f64, %al_j: f64" : "") << seedArguments
       << "):\n";
    if (flags)
      os << "    %al_ij = arith.mulf %al_i, %al_j : f64\n"
         << "    %al_sum = arith.addf %al_i, %al_j : f64\n"
         << "    %al_twice = arith.addf %al_ij, %al_ij : f64\n"
         << "    %cross = arith.subf %al_sum, %al_twice : f64\n"
         << "    %al_one = arith.constant 1.0 : f64\n";
    std::string value;
    if (lj) {
      bool seedsSigma = false, seedsEpsilon = false;
      for (size_t c = 0; c != seeds; ++c)
        (control.tunableTableSeeds[c].sigma ? seedsSigma : seedsEpsilon) =
            true;
      os << "    %sigma" << (seedsSigma ? "_table" : "")
         << " = md.lookup %t_lj_sigma[%type_i, %type_j] : !table, "
            "i32, i32 -> f64\n"
         << "    %epsilon" << (seedsEpsilon ? "_table" : "")
         << " = md.lookup %t_lj_epsilon[%type_i, %type_j] : "
            "!table, i32, i32 -> f64\n";
      // The value of the table plus d_i w[a, b] + d_j w[b, a] for each
      // seed: the value itself, d being zero.
      for (bool sigma : {true, false}) {
        std::string value = sigma ? "%sigma_table" : "%epsilon_table";
        size_t left = 0;
        for (size_t c = 0; c != seeds; ++c)
          left += control.tunableTableSeeds[c].sigma == sigma;
        for (size_t c = 0; c != seeds; ++c) {
          if (control.tunableTableSeeds[c].sigma != sigma)
            continue;
          std::string n = std::to_string(c);
          std::string next = --left ? "%tgs" + n
                                    : std::string(sigma ? "%sigma" : "%epsilon");
          os << "    %tgw" << n << "_ij = md.lookup %t_tgw" << n
             << "[%type_i, %type_j] : !grid, i32, i32 -> f64\n"
             << "    %tgw" << n << "_ji = md.lookup %t_tgw" << n
             << "[%type_j, %type_i] : !grid, i32, i32 -> f64\n"
             << "    %tgs" << n << "_i = arith.mulf %tgd" << n << "_i, %tgw"
             << n << "_ij : f64\n"
             << "    %tgs" << n << "_j = arith.mulf %tgd" << n << "_j, %tgw"
             << n << "_ji : f64\n"
             << "    %tgs" << n << "_ij = arith.addf %tgs" << n << "_i, %tgs"
             << n << "_j : f64\n"
             << "    " << next << " = arith.addf " << value << ", %tgs" << n
             << "_ij : f64\n";
          value = next;
        }
      }
      if (scalesVdw) {
        // The distance r_A of a decoupled pair, r of the others. A pair
        // without a σ, which has no Lennard-Jones, takes 0.3 nm in r_A,
        // so that r_A stays above 0 where its particles overlap.
        double alpha = control.freeEnergy.softCoreAlpha;
        std::string distance = "%r";
        if (alpha > 0.0) {
          os << "    %sc_alpha = arith.constant " << formatReal(alpha)
             << " : f64\n"
             << "    %sc_zero = arith.constant 0.0 : f64\n"
             << "    %sc_floor = arith.constant 3.0e-01 : f64\n"
             << "    %sc_has = arith.cmpf ogt, %sigma, %sc_zero : f64\n"
             << "    %sc_sigma = arith.select %sc_has, %sigma, %sc_floor : f64\n"
             << "    %sc_i3 = arith.constant 3 : i32\n"
             << "    %sc_s3 = math.fpowi %sc_sigma, %sc_i3 : f64, i32\n"
             << "    %sc_s6 = arith.mulf %sc_s3, %sc_s3 : f64\n"
             << "    %sc_r3 = math.fpowi %r, %sc_i3 : f64, i32\n"
             << "    %sc_r6 = arith.mulf %sc_r3, %sc_r3 : f64\n";
          std::string power = "%lambda_vdw";
          if (control.freeEnergy.softCorePower == 2) {
            os << "    %sc_lp = arith.mulf %lambda_vdw, %lambda_vdw : f64\n";
            power = "%sc_lp";
          }
          os << "    %sc_as = arith.mulf %sc_alpha, %sc_s6 : f64\n"
             << "    %sc_asl = arith.mulf %sc_as, " << power << " : f64\n"
             << "    %sc_x = arith.addf %sc_r6, %sc_asl : f64\n"
             ;
          if (truncation != Truncation::None &&
              truncation != Truncation::Shift) {
            os << "    %sc_sixth = arith.constant "
               << formatReal(1.0 / 6.0) << " : f64\n"
               << "    %sc_ra = math.powf %sc_x, %sc_sixth : f64\n"
               << "    %sc_half = arith.constant 5.0e-01 : f64\n"
               << "    %sc_is = arith.cmpf ogt, %cross, %sc_half : f64\n"
               << "    %sc_r = arith.select %sc_is, %sc_ra, %r : f64\n";
            distance = "%sc_r";
          }
        }
        bool plain = truncation == Truncation::None ||
                     truncation == Truncation::Shift;
        if (alpha > 0.0 && plain) {
          // Without a switch, (σ/r_A)⁶ = σ⁶ / x needs no root: one
          // expression for every pair, x = r⁶ for those not decoupled.
          os << "    %sc_xs = arith.mulf %sc_asl, %cross : f64\n"
             << "    %sc_xx = arith.addf %sc_r6, %sc_xs : f64\n"
             << "    %sc_g3 = math.fpowi %sigma, %sc_i3 : f64, i32\n"
             << "    %sc_g6 = arith.mulf %sc_g3, %sc_g3 : f64\n"
             << "    %sc_q6 = arith.divf %sc_g6, %sc_xx : f64\n"
             << "    %sc_q12 = arith.mulf %sc_q6, %sc_q6 : f64\n";
          std::string a = "%sc_q12", b = "%sc_q6";
          if (truncation == Truncation::Shift) {
            double cut6 = std::pow(cutoff, -6.0);
            os << "    %sc_rc6 = arith.constant " << formatReal(cut6)
               << " : f64\n"
               << "    %sc_rc12 = arith.constant " << formatReal(cut6 * cut6)
               << " : f64\n"
               << "    %sc_g12 = arith.mulf %sc_g6, %sc_g6 : f64\n"
               << "    %sc_c12 = arith.mulf %sc_g12, %sc_rc12 : f64\n"
               << "    %sc_c6 = arith.mulf %sc_g6, %sc_rc6 : f64\n"
               << "    %sc_a = arith.subf %sc_q12, %sc_c12 : f64\n"
               << "    %sc_b = arith.subf %sc_q6, %sc_c6 : f64\n";
            a = "%sc_a";
            b = "%sc_b";
          }
          os << "    %sc_c4 = arith.constant 4.0 : f64\n"
             << "    %sc_e4 = arith.mulf %sc_c4, %epsilon : f64\n"
             << "    %sc_t = arith.subf " << a << ", " << b << " : f64\n"
             << "    %lj_full = arith.mulf %sc_e4, %sc_t : f64\n";
        } else {
          emitLennardJones("%sigma", "%epsilon", "%lj_full", truncation,
                           distance);
        }
        os << "    %lj_cl = arith.mulf %cross, %lambda_vdw : f64\n"
           << "    %lj_scale = arith.subf %al_one, %lj_cl : f64\n"
           << "    %lj = arith.mulf %lj_scale, %lj_full : f64\n";
      } else {
        emitLennardJones("%sigma", "%epsilon", "%lj", truncation);
      }
      value = "%lj";
      if (program.ljpme) {
        // The dispersion that the grid sums within the cutoff,
        // −c_i c_j (1 − g(β r)) / r⁶, taken out again, so that the pair
        // has its own Lennard-Jones there [Essmann1995, Wennberg2013]
        // (D162); shifted with it to 0 at the cutoff if it is.
        double beta = program.ljpmeBeta;
        os << "    %lp_c6 = md.lookup %t_ljpme_c6[%type_i, %type_j] : "
              "!table, i32, i32 -> f64\n"
           << "    %lp_beta = arith.constant " << formatReal(beta)
           << " : f64\n"
           << "    %lp_x = arith.mulf %lp_beta, %r : f64\n"
           << "    %lp_x2 = arith.mulf %lp_x, %lp_x : f64\n"
           << "    %lp_nx2 = arith.negf %lp_x2 : f64\n"
           << "    %lp_e = math.exp %lp_nx2 : f64\n"
           << "    %lp_x4 = arith.mulf %lp_x2, %lp_x2 : f64\n"
           << "    %lp_half = arith.constant 0.5 : f64\n"
           << "    %lp_h4 = arith.mulf %lp_half, %lp_x4 : f64\n"
           << "    %lp_one = arith.constant 1.0 : f64\n"
           << "    %lp_p1 = arith.addf %lp_one, %lp_x2 : f64\n"
           << "    %lp_p = arith.addf %lp_p1, %lp_h4 : f64\n"
           << "    %lp_g = arith.mulf %lp_e, %lp_p : f64\n"
           << "    %lp_kept = arith.subf %lp_one, %lp_g : f64\n"
           << "    %lp_r2 = arith.mulf %r, %r : f64\n"
           << "    %lp_r4 = arith.mulf %lp_r2, %lp_r2 : f64\n"
           << "    %lp_r6 = arith.mulf %lp_r4, %lp_r2 : f64\n"
           << "    %lp_cr = arith.divf %lp_c6, %lp_r6 : f64\n"
           << "    %lp_d = arith.mulf %lp_cr, %lp_kept : f64\n";
        std::string correction = "%lp_d";
        if (truncation == Truncation::Shift) {
          double shift = (1.0 - getDispersionScreen(beta * cutoff)) /
                         std::pow(cutoff, 6.0);
          os << "    %lp_shift = arith.constant " << formatReal(shift)
             << " : f64\n"
             << "    %lp_cs = arith.mulf %lp_c6, %lp_shift : f64\n"
             << "    %lp_ds = arith.subf %lp_d, %lp_cs : f64\n";
          correction = "%lp_ds";
        }
        os << "    %ljp = arith.addf %lj, " << correction << " : f64\n";
        value = "%ljp";
      }
    }
    if (coulomb) {
      os << "    %f = arith.constant " << formatReal(coulombInternal)
         << " : f64\n";
      if (scalesCoulomb)
        os << "    %qq0 = arith.mulf %q_i, %q_j : f64\n"
           << "    %qq_cl = arith.mulf %cross, %lambda_coulomb : f64\n"
           << "    %qq_scale = arith.subf %al_one, %qq_cl : f64\n"
           << "    %qq = arith.mulf %qq0, %qq_scale : f64\n";
      else
        os << "    %qq = arith.mulf %q_i, %q_j : f64\n";
      os << "    %fqq = arith.mulf %f, %qq : f64\n";
      // A decoupled pair may overlap once its Lennard-Jones is off; its
      // Coulomb, then 0, takes a distance of at least 1e-4 nm, which no
      // pair that interacts comes near, so that it is not 0 times ∞.
      std::string rq = "%r";
      if (scalesCoulomb) {
        os << "    %rq_least = arith.constant 1.0e-04 : f64\n"
           << "    %rq = arith.maximumf %r, %rq_least : f64\n";
        rq = "%rq";
      }
      if (program.pme && program.pmeProlate) {
        // The direct sum of the prolate kernel, f q q (1 − Φ(r/r_c)) / r
        // [Liang2026, eq. 3]; Φ(1) = 1, so a shift at the cutoff is 0.
        os << "    %rc_inverse = arith.constant "
           << formatReal(1.0 / cutoff) << " : f64\n"
           << "    %xr = arith.mulf " << rq << ", %rc_inverse : f64\n";
        if (takesProlateSeries(control) && !scalesCoulomb) {
          // f q q (1 − x) P(t) / r, P = (1 − Φ(x))/(1 − x) a Chebyshev
          // series in t = 2x − 1: a function of r² alone, which a table of
          // r² then holds as it holds erfc(βr)/r (D94,
          // D280).
          os << "    %pf_two = arith.constant 2.0 : f64\n"
             << "    %pf_one = arith.constant 1.0 : f64\n"
             << "    %pf_2x = arith.mulf %pf_two, %xr : f64\n"
             << "    %pf_t = arith.subf %pf_2x, %pf_one : f64\n"
             << "    %pf_mean = md.chebyshev %pf_t [";
          llvm::interleaveComma(program.pmeFarSeries, os, [&](double c) {
            os << formatReal(c);
          });
          os << "] : f64\n"
             << "    %pf_rest = arith.subf %pf_one, %xr : f64\n"
             << "    %pf_far = arith.mulf %pf_rest, %pf_mean : f64\n"
             << "    %screened = arith.divf %pf_far, " << rq << " : f64\n"
             << "    %coulomb = arith.mulf %fqq, %screened : f64\n";
        } else {
          std::string far = emitProlateSplitting(os, "pme_far", "%xr", "%pf_",
                                                 control.precision);
          os << "    %screened = arith.divf " << far << ", " << rq
             << " : f64\n"
             << "    %coulomb = arith.mulf %fqq, %screened : f64\n";
        }
      } else if (program.pme) {
        // The direct sum of particle mesh Ewald, f q q erfc(β r) / r,
        // shifted to 0 at the cutoff if the control file asks.
        double beta = program.pmeBeta;
        os << "    %beta = arith.constant " << formatReal(beta) << " : f64\n"
           << "    %br = arith.mulf %beta, " << rq << " : f64\n"
           << "    %erfc = math.erfc %br : f64\n"
           << "    %screened = arith.divf %erfc, " << rq << " : f64\n";
        std::string kernel = "%screened";
        if (control.pmeShift || shiftsAtCutoff) {
          os << "    %shift = arith.constant "
             << formatReal(std::erfc(beta * cutoff) / cutoff) << " : f64\n"
             << "    %shifted = arith.subf %screened, %shift : f64\n";
          kernel = "%shifted";
        }
        os << "    %coulomb = arith.mulf %fqq, " << kernel << " : f64\n";
      } else if (program.reactionField) {
        // f q q (1/r + k r² − c), 0 at the cutoff (D140).
        auto [k, c] = getReactionField(control);
        os << "    %one = arith.constant 1.0 : f64\n"
           << "    %krf = arith.constant " << formatReal(k) << " : f64\n"
           << "    %crf = arith.constant " << formatReal(c) << " : f64\n"
           << "    %inverse = arith.divf %one, " << rq << " : f64\n"
           << "    %rr = arith.mulf %r, %r : f64\n"
           << "    %field = arith.mulf %krf, %rr : f64\n"
           << "    %near = arith.addf %inverse, %field : f64\n"
           << "    %rf = arith.subf %near, %crf : f64\n"
           << "    %coulomb = arith.mulf %fqq, %rf : f64\n";
      } else if (shiftsAtCutoff) {
        // f q q (1/r − 1/r_c), the cutoff shifted (D210).
        os << "    %one = arith.constant 1.0 : f64\n"
           << "    %inverse = arith.divf %one, " << rq << " : f64\n"
           << "    %rc_inverse = arith.constant " << formatReal(1.0 / cutoff)
           << " : f64\n"
           << "    %cut = arith.subf %inverse, %rc_inverse : f64\n"
           << "    %coulomb = arith.mulf %fqq, %cut : f64\n";
      } else {
        os << "    %coulomb = arith.divf %fqq, " << rq << " : f64\n";
      }
      if (value.empty()) {
        value = "%coulomb";
      } else {
        os << "    %k = arith.addf " << value << ", %coulomb : f64\n";
        value = "%k";
      }
    }
    // In `@alchemical` the pairs that λ leaves alone add nothing.
    if (alchemicalOnly) {
      os << "    %masked = arith.mulf " << value << ", %cross : f64\n";
      value = "%masked";
    }
    os << "    md.yield " << value << " : f64\n"
       << "  } : !pairs, !vec -> f64\n";
    add("nonbonded");
  }

  // %shielded, the share of 1/r that the reciprocal sum holds at %r:
  // erf(β r)/r, or of the prolate kernel Φ(r/r_c)/r, which is Φ(x)/x / r_c
  // within the cutoff and 1/r beyond it (D266).
  auto emitShielded = [&](llvm::raw_ostream &os) {
    if (!program.pmeProlate) {
      os << "    %br = arith.mulf %beta, %r : f64\n"
         << "    %erf = math.erf %br : f64\n"
         << "    %shielded = arith.divf %erf, %r : f64\n";
      return;
    }
    os << "    %rc_inverse = arith.constant " << formatReal(1.0 / cutoff)
       << " : f64\n"
       << "    %xr = arith.mulf %r, %rc_inverse : f64\n";
    std::string near = emitProlateSplitting(os, "pme_near", "%xr", "%pn_", control.precision);
    os << "    %near = arith.mulf " << near << ", %rc_inverse : f64\n"
       << "    %r_one = arith.constant 1.0 : f64\n"
       << "    %r_inverse = arith.divf %r_one, %r : f64\n"
       << "    %r_inside = arith.cmpf olt, %xr, %r_one : f64\n"
       << "    %shielded = arith.select %r_inside, %near, %r_inverse : f64\n";
  };
  if (program.pme && (terms & CoulombExcluded) && has("excluded")) {
    // The excluded pairs take their share of the reciprocal sum out again:
    // −f q_i q_j erf(β r) / r, with the charges that the reciprocal sum
    // takes, those of the selection of [free_energy] times 1 − λ (D161).
    bool flags = scalesCoulomb || (alchemicalOnly && decouples());
    os << "  %u_excluded = md.sum_tuples %r_excluded, %x, %cell coordinates("
          "distance(0, 1))\n"
       << "      gather(%p_q" << (flags ? ", %p_alch : !real, !real" : " : !real")
       << ") {\n"
       << "  ^bb0(%r: f64, %q_i: f64, %q_j: f64"
       << (flags ? ", %al_i: f64, %al_j: f64" : "") << "):\n"
       << "    %f = arith.constant " << formatReal(-coulombInternal)
       << " : f64\n"
       << "    %beta = arith.constant " << formatReal(program.pmeBeta)
       << " : f64\n"
       << "    %qq0 = arith.mulf %q_i, %q_j : f64\n";
    std::string qq = "%qq0";
    if (flags) {
      os << "    %al_one = arith.constant 1.0 : f64\n"
         << "    %al_li = arith.mulf %al_i, %lambda_coulomb : f64\n"
         << "    %al_si = arith.subf %al_one, %al_li : f64\n"
         << "    %al_lj = arith.mulf %al_j, %lambda_coulomb : f64\n"
         << "    %al_sj = arith.subf %al_one, %al_lj : f64\n"
         << "    %al_s = arith.mulf %al_si, %al_sj : f64\n"
         << "    %qq = arith.mulf %qq0, %al_s : f64\n";
      qq = "%qq";
    }
    os << "    %fqq = arith.mulf %f, " << qq << " : f64\n";
    emitShielded(os);
    os << "    %e = arith.mulf %fqq, %shielded : f64\n";
    std::string e = "%e";
    if (alchemicalOnly) {
      // Only the pairs within the selection depend on λ.
      os << "    %al_both = arith.mulf %al_i, %al_j : f64\n"
         << "    %masked = arith.mulf %e, %al_both : f64\n";
      e = "%masked";
    }
    os << "    md.yield " << e << " : f64\n"
       << "  } : !rel_excluded, !vec -> f64\n";
    add("excluded");
  }
  if (program.pme && (terms & CoulombWithin) && scalesCoulomb &&
      has("alchemical_pairs")) {
    // The pairs within the selection that are not excluded keep their
    // Coulomb, which the reciprocal sum of the scaled charges takes times
    // (1 − λ)²: (1 − (1 − λ)²) f q_i q_j erf(β r) / r adds the rest (D161).
    os << "  %u_within = md.sum_tuples %r_alchemical_pairs, %x, %cell "
          "coordinates(distance(0, 1))\n"
       << "      gather(%p_q : !real) {\n"
       << "  ^bb0(%r: f64, %q_i: f64, %q_j: f64):\n"
       << "    %f = arith.constant " << formatReal(coulombInternal)
       << " : f64\n"
       << "    %beta = arith.constant " << formatReal(program.pmeBeta)
       << " : f64\n"
       << "    %one = arith.constant 1.0 : f64\n"
       << "    %s = arith.subf %one, %lambda_coulomb : f64\n"
       << "    %s2 = arith.mulf %s, %s : f64\n"
       << "    %w = arith.subf %one, %s2 : f64\n"
       << "    %qq = arith.mulf %q_i, %q_j : f64\n"
       << "    %fqq = arith.mulf %f, %qq : f64\n"
       << "    %fw = arith.mulf %fqq, %w : f64\n";
    emitShielded(os);
    os << "    %e = arith.mulf %fw, %shielded : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_alchemical_pairs, !vec -> f64\n";
    add("within");
  }
  if (program.reactionField && (terms & CoulombExcluded) &&
      !alchemicalOnly && has("excluded")) {
    // The field acts on the excluded pairs within the cutoff as well:
    // f q_i q_j (k r² − c), as GROMACS has it (D140).
    auto [k, c] = getReactionField(control);
    os << "  %u_excluded = md.sum_tuples %r_excluded, %x, %cell coordinates("
          "distance(0, 1))\n"
       << "      gather(%p_q : !real) {\n"
       << "  ^bb0(%r: f64, %q_i: f64, %q_j: f64):\n"
       << "    %f = arith.constant " << formatReal(coulombInternal)
       << " : f64\n"
       << "    %krf = arith.constant " << formatReal(k) << " : f64\n"
       << "    %crf = arith.constant " << formatReal(c) << " : f64\n"
       << "    %rc = arith.constant " << formatReal(cutoff) << " : f64\n"
       << "    %zero = arith.constant 0.0 : f64\n"
       << "    %qq = arith.mulf %q_i, %q_j : f64\n"
       << "    %fqq = arith.mulf %f, %qq : f64\n"
       << "    %rr = arith.mulf %r, %r : f64\n"
       << "    %field = arith.mulf %krf, %rr : f64\n"
       << "    %rf = arith.subf %field, %crf : f64\n"
       << "    %all = arith.mulf %fqq, %rf : f64\n"
       << "    %inside = arith.cmpf olt, %r, %rc : f64\n"
       << "    %e = arith.select %inside, %all, %zero : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_excluded, !vec -> f64\n";
    add("excluded");
  }
  if (program.ljpme && (terms & LennardJonesExcluded) && has("excluded")) {
    // The excluded pairs take their share of the grid of the dispersion
    // out again: c_i c_j P(3, β² r²) / r⁶ with P(3, y) = 1 − exp(−y) (1 +
    // y + y²/2) = 1 − g(β r) (D162). Below y = 1, P is taken from its
    // series exp(−y) y³ Σ_k y^k / (k + 3)!, which has no cancellation;
    // the pairs of a molecule are close, where 1 − g is small.
    os << "  %u_lj_excluded = md.sum_tuples %r_excluded, %x, %cell "
          "coordinates(distance(0, 1))\n"
       << "      gather(%p_type : !ids) {\n"
       << "  ^bb0(%r: f64, %type_i: i32, %type_j: i32):\n"
       << "    %c6 = md.lookup %t_ljpme_c6[%type_i, %type_j] : !table, i32, "
          "i32 -> f64\n"
       << "    %beta = arith.constant " << formatReal(program.ljpmeBeta)
       << " : f64\n"
       << "    %bx = arith.mulf %beta, %r : f64\n"
       << "    %y = arith.mulf %bx, %bx : f64\n"
       << "    %ny = arith.negf %y : f64\n"
       << "    %ey = math.exp %ny : f64\n"
       << "    %one = arith.constant 1.0 : f64\n"
       << "    %half = arith.constant 0.5 : f64\n";
    // The series, by Horner's rule, to k = 10: the rest is below 1e-10 of
    // it.
    double factorial = 6.0;
    std::vector<double> coefficients;
    for (int k = 0; k <= 10; ++k) {
      coefficients.push_back(1.0 / factorial);
      factorial *= k + 4;
    }
    os << "    %s10 = arith.constant " << formatReal(coefficients[10])
       << " : f64\n";
    for (int k = 9; k >= 0; --k)
      os << "    %a" << k << " = arith.constant "
         << formatReal(coefficients[k]) << " : f64\n"
         << "    %sy" << k << " = arith.mulf %s" << k + 1 << ", %y : f64\n"
         << "    %s" << k << " = arith.addf %sy" << k << ", %a" << k
         << " : f64\n";
    os << "    %y2 = arith.mulf %y, %y : f64\n"
       << "    %y3 = arith.mulf %y2, %y : f64\n"
       << "    %ey3 = arith.mulf %ey, %y3 : f64\n"
       << "    %series = arith.mulf %ey3, %s0 : f64\n"
       << "    %hy2 = arith.mulf %half, %y2 : f64\n"
       << "    %p1 = arith.addf %one, %y : f64\n"
       << "    %p = arith.addf %p1, %hy2 : f64\n"
       << "    %g = arith.mulf %ey, %p : f64\n"
       << "    %closed = arith.subf %one, %g : f64\n"
       << "    %small = arith.cmpf olt, %y, %one : f64\n"
       << "    %kept = arith.select %small, %series, %closed : f64\n"
       << "    %r2 = arith.mulf %r, %r : f64\n"
       << "    %r4 = arith.mulf %r2, %r2 : f64\n"
       << "    %r6 = arith.mulf %r4, %r2 : f64\n"
       << "    %cr = arith.divf %c6, %r6 : f64\n"
       << "    %e = arith.mulf %cr, %kept : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_excluded, !vec -> f64\n";
    add("lj_excluded");
  }
  if (program.ljpme && (terms & LennardJonesReciprocal)) {
    // The grid of the dispersion (D162), of the coefficients c_i.
    bool listed = hasField("ljpme_listed");
    os << "  %u_lj_reciprocal, %f_lj_reciprocal, %w_lj_reciprocal = "
          "md.reciprocal %x, %p_ljpme_c, %cell, %t_ljpme_moduli"
       << (listed ? " listed(%p_ljpme_listed)" : "") << "\n"
       << "      grid([" << program.ljpmeGrid[0] << ", "
       << program.ljpmeGrid[1] << ", " << program.ljpmeGrid[2]
       << "]) order(" << control.ljpmeOrder << ") beta("
       << formatReal(program.ljpmeBeta) << ") coulomb(1.0) dispersion"
       << (listed && gathersAll("ljpme_listed") ? " gather_all" : "") << "\n"
       << "      : !vec, !real, !grid" << (listed ? ", !ids" : "")
       << " -> f64, !vec, vector<9xf64>\n";
    add("lj_reciprocal");
  }
  if (program.pme && (terms & CoulombReciprocal)) {
    // The charges of the selection of [free_energy] times 1 − λ (D161).
    std::string charges = "%p_q";
    if (scalesCoulomb) {
      os << "  %q_scaled = md.map_particles gather(%p_q, %p_alch : !real, "
            "!real) {\n"
         << "  ^bb0(%q: f64, %al: f64):\n"
         << "    %one = arith.constant 1.0 : f64\n"
         << "    %al_l = arith.mulf %al, %lambda_coulomb : f64\n"
         << "    %al_s = arith.subf %one, %al_l : f64\n"
         << "    %qs = arith.mulf %q, %al_s : f64\n"
         << "    md.yield %qs : f64\n"
         << "  } : !real\n";
      charges = "%q_scaled";
    }
    // The particles of a charge that can be nonzero alone
    // (D262).
    bool listed = hasField("pme_listed");
    os << "  %u_reciprocal, %f_reciprocal, %w_reciprocal = md.reciprocal %x, "
       << charges << ", %cell, %t_pme_moduli"
       << (listed ? " listed(%p_pme_listed)" : "") << "\n"
       << "      grid([" << program.pmeGrid[0] << ", " << program.pmeGrid[1]
       << ", " << program.pmeGrid[2] << "]) order(" << program.pmeOrder
       << ") beta(" << formatReal(program.pmeBeta) << ") coulomb("
       << formatReal(coulombInternal) << ")"
       << (program.pmeProlate ? " prolate" : "")
       << (program.pmeP3M ? " p3m" : "")
       << (listed && gathersAll("pme_listed") ? " gather_all" : "") << "\n"
       << "      : !vec, !real, !grid" << (listed ? ", !ids" : "")
       << " -> f64, !vec, vector<9xf64>\n";
    add("reciprocal");
  }
  if ((terms & Bonds) && has("bonds")) {
    os << "  %u_bonds = md.sum_tuples %r_bonds, %x, %cell coordinates("
          "distance(0, 1))\n"
       << "      tuple(%f_bonds_k, %f_bonds_r0 : !of_bonds, !of_bonds) {\n"
       << "  ^bb0(%r: f64, %k: f64, %r0: f64):\n"
       << "    %half = arith.constant 0.5 : f64\n"
       << "    %dr = arith.subf %r, %r0 : f64\n"
       << "    %dr2 = arith.mulf %dr, %dr : f64\n"
       << "    %hk = arith.mulf %half, %k : f64\n"
       << "    %e = arith.mulf %hk, %dr2 : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_bonds, !vec -> f64\n";
    add("bonds");
  }
  if ((terms & Angles) && has("angles")) {
    os << "  %u_angles = md.sum_tuples %r_angles, %x, %cell coordinates("
          "angle(0, 1, 2))\n"
       << "      tuple(%f_angles_k, %f_angles_theta0 : !of_angles, "
          "!of_angles) {\n"
       << "  ^bb0(%theta: f64, %k: f64, %theta0: f64):\n"
       << "    %half = arith.constant 0.5 : f64\n"
       << "    %dt = arith.subf %theta, %theta0 : f64\n"
       << "    %dt2 = arith.mulf %dt, %dt : f64\n"
       << "    %hk = arith.mulf %half, %k : f64\n"
       << "    %e = arith.mulf %hk, %dt2 : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_angles, !vec -> f64\n";
    add("angles");
  }
  if ((terms & UreyBradleys) && has("urey_bradley")) {
    os << "  %u_urey_bradley = md.sum_tuples %r_urey_bradley, %x, %cell "
          "coordinates(distance(0, 1))\n"
       << "      tuple(%f_urey_bradley_k, %f_urey_bradley_r0 : "
          "!of_urey_bradley, !of_urey_bradley) {\n"
       << "  ^bb0(%r: f64, %k: f64, %r0: f64):\n"
       << "    %half = arith.constant 0.5 : f64\n"
       << "    %dr = arith.subf %r, %r0 : f64\n"
       << "    %dr2 = arith.mulf %dr, %dr : f64\n"
       << "    %hk = arith.mulf %half, %k : f64\n"
       << "    %e = arith.mulf %hk, %dr2 : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_urey_bradley, !vec -> f64\n";
    add("urey_bradley");
  }
  if ((terms & Dihedrals) && has("dihedrals")) {
    os << "  %u_dihedrals = md.sum_tuples %r_dihedrals, %x, %cell "
          "coordinates(dihedral(0, 1, 2, 3))\n"
       << "      tuple(%f_dihedrals_k, %f_dihedrals_n, %f_dihedrals_phase : "
          "!of_dihedrals, !of_dihedrals, !of_dihedrals) {\n"
       << "  ^bb0(%phi: f64, %k: f64, %period: f64, %phase: f64):\n"
       << "    %one = arith.constant 1.0 : f64\n"
       << "    %nphi = arith.mulf %period, %phi : f64\n"
       << "    %a = arith.subf %nphi, %phase : f64\n"
       << "    %cos = math.cos %a : f64\n"
       << "    %s = arith.addf %one, %cos : f64\n"
       << "    %e = arith.mulf %k, %s : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_dihedrals, !vec -> f64\n";
    add("dihedrals");
  }
  if ((terms & HarmonicImpropers) && has("impropers")) {
    // ½ k (ξ − ξ0)², with ξ − ξ0 brought into [−π, π).
    os << "  %u_impropers = md.sum_tuples %r_impropers, %x, %cell "
          "coordinates(dihedral(0, 1, 2, 3))\n"
       << "      tuple(%f_impropers_k, %f_impropers_xi0 : !of_impropers, "
          "!of_impropers) {\n"
       << "  ^bb0(%xi: f64, %k: f64, %xi0: f64):\n"
       << "    %half = arith.constant 0.5 : f64\n"
       << "    %pi = arith.constant " << formatReal(M_PI) << " : f64\n"
       << "    %turn = arith.constant " << formatReal(2.0 * M_PI) << " : f64\n"
       << "    %per_turn = arith.constant " << formatReal(0.5 / M_PI)
       << " : f64\n"
       << "    %d = arith.subf %xi, %xi0 : f64\n"
       << "    %dp = arith.addf %d, %pi : f64\n"
       << "    %turns = arith.mulf %dp, %per_turn : f64\n"
       << "    %whole = math.floor %turns : f64\n"
       << "    %back = arith.mulf %whole, %turn : f64\n"
       << "    %w = arith.subf %d, %back : f64\n"
       << "    %w2 = arith.mulf %w, %w : f64\n"
       << "    %hk = arith.mulf %half, %k : f64\n"
       << "    %e = arith.mulf %hk, %w2 : f64\n"
       << "    md.yield %e : f64\n"
       << "  } : !rel_impropers, !vec -> f64\n";
    add("impropers");
  }
  // The terms given by expressions (D136), in the units of the control
  // file: the distance in Å, the angles in radians, the energy in kcal/mol.
  for (auto [index, term] : llvm::enumerate(system.topology->tupleTerms)) {
    if (!(terms & TupleTerms) ||
        (tupleTerm >= 0 && static_cast<int>(index) != tupleTerm))
      continue;
    if (dependent && !dependent->hasTuple(static_cast<unsigned>(index)))
      continue;
    if (alchemicalOnly && !usesLambda(term.expression))
      continue;
    if (term.isCentroid()) {
      emitCentroidTerm(index, term);
      add("centroid_" + term.name);
      continue;
    }
    static const char *const coordinates[] = {
        "", "", "distance(0, 1)", "angle(0, 1, 2)", "dihedral(0, 1, 2, 3)"};
    std::string set = "custom_" + term.name;
    // A compound term (D165) takes the coordinates that its expression
    // names, of any members of its tuples.
    std::string list = term.isCompound() ? "" : coordinates[term.arity];
    for (auto [k, coordinate] : llvm::enumerate(term.coordinates)) {
      list += (k ? ", " : "") + coordinate.kind + "(";
      for (auto [p, place] : llvm::enumerate(coordinate.places))
        list += (p ? ", " : "") + std::to_string(place);
      list += ")";
    }
    os << "  %u_" << set << " = md.sum_tuples %r_" << set
       << ", %x, %cell coordinates(" << list << ")";
    // In `@tunable`, the tunable parameters of the term take the fields of
    // zeros through which their derivatives leave (D230).
    std::vector<const TupleGradient *> gradients;
    if (gradientArguments)
      for (const TupleGradient &gradient : tupleGradients)
        if (gradient.term == index)
          gradients.push_back(&gradient);
    if (!gradients.empty()) {
      std::string fields, types;
      for (const TupleGradient *gradient : gradients)
        for (unsigned c = 0; c != gradient->count; ++c) {
          fields += (fields.empty() ? "%tg_d" : ", %tg_d") +
                    std::to_string(gradient->first + c);
          types += types.empty() ? "!real" : ", !real";
        }
      os << "\n      gather(" << fields << " : " << types << ")";
    }
    if (!term.parameters.empty()) {
      os << "\n      tuple(";
      for (auto [k, parameter] : llvm::enumerate(term.parameters))
        os << (k ? ", " : "") << "%f_" << set << "_" << parameter.first;
      for (const TupleGradient *gradient : gradients)
        os << ", %f_" << set << "_tgc_" << gradient->parameter;
      os << " : ";
      for (size_t k = 0; k != term.parameters.size() + gradients.size(); ++k)
        os << (k ? ", " : "") << "!of_" << set;
      os << ")";
    }
    llvm::StringMap<std::string> values;
    if (term.isCompound()) {
      os << " {\n  ^bb0(";
      for (size_t k = 0; k != term.coordinates.size(); ++k)
        os << (k ? ", " : "") << "%c" << k << ": f64";
    } else {
      os << " {\n  ^bb0(%c: f64";
    }
    for (const TupleGradient *gradient : gradients)
      for (unsigned c = 0; c != gradient->count; ++c)
        for (unsigned member = 0; member != term.arity; ++member)
          os << ", %tgd" << gradient->first + c << "_" << member << ": f64";
    for (const auto &parameter : term.parameters)
      os << ", %cp_" << parameter.first << ": f64";
    for (const TupleGradient *gradient : gradients)
      os << ", %cp_tgc_" << gradient->parameter << ": f64";
    os << "):\n";
    if (term.isCompound() || term.arity == 2)
      os << "    %c_scale = arith.constant " << formatReal(1.0 / units::length)
         << " : f64\n";
    if (term.isCompound()) {
      // Distances in Å, angles in radians.
      for (auto [k, coordinate] : llvm::enumerate(term.coordinates)) {
        std::string variable = "%c" + std::to_string(k);
        if (coordinate.kind == "distance") {
          os << "    " << variable << "_a = arith.mulf " << variable
             << ", %c_scale : f64\n";
          variable += "_a";
        }
        values[coordinate.name] = variable;
      }
    } else {
      std::string variable = "%c";
      if (term.arity == 2) {
        os << "    %c_a = arith.mulf %c, %c_scale : f64\n";
        variable = "%c_a";
      }
      values[term.getVariable()] = variable;
    }
    if (control.usesTime)
      values["t"] = "%time";
    bindLambdas(values);
    for (const auto &parameter : term.parameters)
      values[parameter.first] = "%cp_" + parameter.first;
    for (const TupleGradient *gradient : gradients) {
      // p + d_c[member] for the field c of the tuple: p, since d is zero.
      std::string stem = "%tgp_" + gradient->parameter;
      std::string value = "%cp_" + gradient->parameter;
      os << "    " << stem << "_zero = arith.constant 0.0 : f64\n";
      for (unsigned c = 0; c != gradient->count; ++c) {
        std::string at = stem + "_" + std::to_string(c);
        os << "    " << at << "_n = arith.constant " << formatReal(c)
           << " : f64\n"
           << "    " << at << "_is = arith.cmpf oeq, %cp_tgc_"
           << gradient->parameter << ", " << at << "_n : f64\n"
           << "    " << at << "_d = arith.select " << at << "_is, %tgd"
           << gradient->first + c << "_" << gradient->place << ", " << stem
           << "_zero : f64\n"
           << "    " << at << " = arith.addf " << value << ", " << at
           << "_d : f64\n";
        value = at;
      }
      values[gradient->parameter] = value;
    }
    bindObserved(term.name, values);
    Expression expression = llvm::cantFail(Expression::parse(term.expression, control.functions));
    std::string energy = expression.emit(os, values, "%ce", "    ", term.name);
    os << "    %c_kj = arith.constant " << formatReal(units::energy)
       << " : f64\n"
       << "    %c_e = arith.mulf " << energy << ", %c_kj : f64\n"
       << "    md.yield %c_e : f64\n"
       << "  } : !rel_" << set << ", !vec -> f64\n";
    add(set);
  }
  bool lj14 = terms & LennardJones14, coulomb14 = terms & Coulomb14;
  if ((lj14 || coulomb14) && has("pairs14")) {
    // In `@tunable` with the derivative in the charges, the product of
    // the charges is formed here, f s_C q_i q_j, from the charges
    // themselves (D230).
    bool products = gradientArguments && gradientCharges && coulomb14;
    os << "  %u_pairs14 = md.sum_tuples %r_pairs14, %x, %cell coordinates("
          "distance(0, 1))\n"
       << (products ? "      gather(%p_q : !real)\n" : "")
       << "      tuple(%f_pairs14_sigma, %f_pairs14_epsilon, %f_pairs14_qq"
       << (products ? ", %f_pairs14_tgs" : "")
       << " : !of_pairs14, !of_pairs14, !of_pairs14"
       << (products ? ", !of_pairs14" : "") << ") {\n"
       << "  ^bb0(%r: f64"
       << (products ? ", %q14_i: f64, %q14_j: f64" : "")
       << ", %sigma: f64, %epsilon: f64, %qq" << (products ? "_held" : "")
       << ": f64" << (products ? ", %tgs: f64" : "") << "):\n";
    if (products)
      os << "    %q14 = arith.mulf %q14_i, %q14_j : f64\n"
         << "    %qq = arith.mulf %tgs, %q14 : f64\n";
    std::string value;
    if (lj14) {
      // The pairs three bonds apart as they are, except under the power
      // force or squared-distance switch, which also switches these pairs.
      emitLennardJones("%sigma", "%epsilon", "%lj",
                       (control.truncation == Truncation::PowerForceSwitch ||
                        control.truncation == Truncation::SquaredDistanceSwitch)
                           ? control.truncation
                           : Truncation::None);
      value = "%lj";
    }
    if (coulomb14) {
      os << "    %coulomb = arith.divf %qq, %r : f64\n";
      if (value.empty()) {
        value = "%coulomb";
      } else {
        os << "    %k = arith.addf " << value << ", %coulomb : f64\n";
        value = "%k";
      }
    }
    os << "    md.yield " << value << " : f64\n"
       << "  } : !rel_pairs14, !vec -> f64\n";
    add("pairs14");
  }
  if ((terms & CMaps) && has("cmap")) {
    // The cell of φ and ψ, each from −180° in steps of 360° / n, and the
    // places t and u in it; φ = 180° is the first cell again. The patch
    // is E = Σ c_ij t^i u^j [MacKerell2004].
    unsigned n = system.topology->cmapResolution;
    os << "  %u_cmap = md.sum_tuples %r_cmap, %x, %cell coordinates("
          "dihedral(0, 1, 2, 3), dihedral(1, 2, 3, 4))\n"
       << "      tuple(%f_cmap_map : !of_cmap) {\n"
       << "  ^bb0(%phi: f64, %psi: f64, %map: f64):\n"
       << "    %scale = arith.constant " << formatReal(n / (2.0 * M_PI))
       << " : f64\n"
       << "    %shift = arith.constant " << formatReal(n / 2.0) << " : f64\n"
       << "    %cmap_n = arith.constant " << n << " : i32\n";
    for (StringRef angle : {"phi", "psi"}) {
      std::string a = angle.str();
      os << "    %" << a << "_s = arith.mulf %" << a << ", %scale : f64\n"
         << "    %" << a << "_x = arith.addf %" << a << "_s, %shift : f64\n"
         << "    %" << a << "_f = math.floor %" << a << "_x : f64\n"
         << "    %" << a << "_t = arith.subf %" << a << "_x, %" << a
         << "_f : f64\n"
         << "    %" << a << "_i = arith.fptosi %" << a
         << "_f : f64 to i32\n"
         << "    %" << a << "_j = arith.addi %" << a
         << "_i, %cmap_n : i32\n"
         << "    %" << a << "_k = arith.remsi %" << a
         << "_j, %cmap_n : i32\n";
    }
    os << "    %m = arith.fptosi %map : f64 to i32\n"
       << "    %mn = arith.muli %m, %cmap_n : i32\n"
       << "    %row = arith.addi %mn, %phi_k : i32\n"
       << "    %rown = arith.muli %row, %cmap_n : i32\n"
       << "    %cellid = arith.addi %rown, %psi_k : i32\n";
    for (int k = 0; k != 16; ++k)
      os << "    %k" << k << " = arith.constant " << k << " : i32\n"
         << "    %c" << k << " = md.lookup %t_cmap[%cellid, %k" << k
         << "] : !grid, i32, i32 -> f64\n";
    // Horner in u for each power of t, then in t.
    for (int i = 0; i != 4; ++i) {
      std::string last = "%c" + std::to_string(4 * i + 3);
      for (int j = 2; j >= 0; --j) {
        std::string name = "%r" + std::to_string(i) + std::to_string(j);
        os << "    " << name << "m = arith.mulf " << last
           << ", %psi_t : f64\n"
           << "    " << name << " = arith.addf " << name << "m, %c"
           << 4 * i + j << " : f64\n";
        last = name;
      }
    }
    std::string last = "%r30";
    for (int i = 2; i >= 0; --i) {
      std::string name = "%e" + std::to_string(i);
      os << "    " << name << "m = arith.mulf " << last << ", %phi_t : f64\n"
         << "    " << name << " = arith.addf " << name << "m, %r" << i
         << "0 : f64\n";
      last = name;
    }
    os << "    md.yield %e0 : f64\n"
       << "  } : !rel_cmap, !vec -> f64\n";
    add("cmap");
  }
  // The models in stages, which md-inline brings in and md-differentiate
  // differentiates with the rest (D294).
  if (terms == AllTerms)
    for (size_t k = 0; k != stagedPotentials.size(); ++k) {
      os << stagedPotentials[k];
      if (total.empty()) {
        total = stagedResults[k];
        continue;
      }
      std::string sum = "%total_staged" + std::to_string(k);
      os << "  " << sum << " = arith.addf " << total << ", "
         << stagedResults[k] << " : f64\n";
      total = sum;
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
  std::string truncation = getTruncation(control);

  os << "md.potential @energy(%x: !vec, %cell: !md.cell"
     << getFieldParameters() << getTimeParameter() << ") -> f64 {\n";
  os << "  %n = md.neighborhood %x, %cell cutoff(" << formatReal(cutoff)
     << ") : !vec -> !pairs\n";

  std::string total;
  for (auto [index, term] : llvm::enumerate(control.pairs)) {
    const Expression &expression = expressions[index];

    // The fields that the kernel reads, and the values of the two
    // particles of a pair.
    const llvm::StringMap<unsigned> &tables = termTables[index];
    std::string gathered, types, arguments;
    if (!tables.empty()) {
      gathered = "%p_type";
      types = "!ids";
      arguments = ", %type_i: i32, %type_j: i32";
    }
    for (const Parameter &parameter : parameters) {
      if (!tables.empty() || parameter.isUniform ||
          !llvm::is_contained(expression.getNames(), parameter.name))
        continue;
      gathered += (gathered.empty() ? "" : ", ") + ("%p_" + parameter.name);
      types += std::string(types.empty() ? "" : ", ") + "!real";
      arguments += ", %" + parameter.name + "_i: f64, %" + parameter.name +
                   "_j: f64";
    }

    std::string result = "%u" + std::to_string(index);
    os << "  " << result << " = md.sum_relation %n, %x, %cell";
    if (!gathered.empty())
      os << " gather(" << gathered << " : " << types << ")";
    os << "\n      exchange(symmetric)" << truncation << " {\n";
    os << "  ^bb0(%r: f64, %d: vector<3xf64>" << arguments << "):\n";

    // The expression is evaluated in the units of the control file.
    llvm::StringMap<std::string> values;
    os << "    %to_length = arith.constant "
       << formatReal(1.0 / units::length) << " : f64\n";
    os << "    %r_in = arith.mulf %r, %to_length : f64\n";
    values["r"] = "%r_in";

    for (auto &[name, value] : term.constants) {
      os << "    %c_" << name << " = arith.constant " << formatReal(value)
         << " : f64\n";
      values[name] = "%c_" + name;
    }
    if (llvm::is_contained(expression.getNames(), "coulomb")) {
      os << "    %coulomb = arith.constant " << formatReal(coulombConstant)
         << " : f64\n";
      values["coulomb"] = "%coulomb";
    }
    for (auto &[name, table] : tables) {
      os << "    %" << name.str() << " = md.lookup %t_"
         << program.tables[table].name
         << "[%type_i, %type_j] : !table, i32, i32 -> f64\n";
      values[name] = "%" + name.str();
    }
    for (const Parameter &parameter : parameters) {
      if (!tables.empty() ||
          !llvm::is_contained(expression.getNames(), parameter.name))
        continue;
      const std::string &name = parameter.name;
      if (parameter.isUniform) {
        os << "    %" << name << " = arith.constant "
           << formatReal(parameter.value) << " : f64\n";
      } else if (term.mixing.lookup(name) == Mixing::Arithmetic) {
        os << "    %half_" << name << " = arith.constant 5.0e-01 : f64\n";
        os << "    %sum_" << name << " = arith.addf %" << name << "_i, %"
           << name << "_j : f64\n";
        os << "    %" << name << " = arith.mulf %half_" << name << ", %sum_"
           << name << " : f64\n";
      } else if (term.mixing.lookup(name) == Mixing::Product) {
        os << "    %" << name << " = arith.mulf %" << name << "_i, %" << name
           << "_j : f64\n";
      } else {
        os << "    %product_" << name << " = arith.mulf %" << name
           << "_i, %" << name << "_j : f64\n";
        os << "    %" << name << " = math.sqrt %product_" << name
           << " : f64\n";
      }
      values[name] = "%" + name;
    }

    std::string value = expression.emit(os, values, "%e", "    ", term.name);
    os << "    %to_energy = arith.constant " << formatReal(units::energy)
       << " : f64\n";
    os << "    %u = arith.mulf " << value << ", %to_energy : f64\n";
    os << "    md.yield %u : f64\n";
    os << "  } : !pairs, !vec -> f64\n";

    if (total.empty()) {
      total = result;
    } else {
      std::string sum = "%total" + std::to_string(index);
      os << "  " << sum << " = arith.addf " << total << ", " << result
         << " : f64\n";
      total = sum;
    }
  }
  for (auto [index, term] : llvm::enumerate(control.triplets)) {
    std::string result = emitTripletTerm(index, term);
    std::string sum = "%total_t" + std::to_string(index);
    os << "  " << sum << " = arith.addf " << total << ", " << result
       << " : f64\n";
    total = sum;
  }
  os << "  md.return " << total << " : f64\n}\n\n";
  return llvm::Error::success();
}

std::string Builder::emitTripletTerm(size_t index, const TripletTerm &term) {
  // The triplets of the neighborhood within the cutoff of the term, each
  // center with each unordered pair of its legs once (D160); the center is
  // particle 1 of the expression and place 1 of the IR, so the ends 2 and
  // 3 are places 0 and 2.
  std::string k = std::to_string(index);
  std::string relation = "%triplets" + k, result = "%u_triplet" + k;
  os << "  " << relation << " = md.triplets %n cutoff("
     << formatReal(term.cutoff * units::length)
     << ") : !pairs -> !md.relation<@atoms, 3, reversal>\n";

  Expression expression =
      llvm::cantFail(Expression::parse(term.expression, control.functions));
  // cos(theta) is the coordinate cosine(0, 1, 2), whose derivative has no
  // 1/sin(theta), which a collinear triplet makes infinite
  // (tuples.md, Section 3.1); theta alone is the angle.
  expression.replaceCall("cos", "theta", "cos(theta)");
  const std::vector<std::string> &names = expression.getNames();
  struct Variable {
    const char *name, *coordinate;
    bool isLength;
  };
  static const Variable variables[] = {
      {"r12", "distance(0, 1)", true},
      {"r13", "distance(2, 1)", true},
      {"r23", "distance(0, 2)", true},
      {"theta", "angle(0, 1, 2)", false},
      {"cos(theta)", "cosine(0, 1, 2)", false}};
  // A loop over tuples takes at least one coordinate: r12, if the
  // expression uses none.
  std::vector<const Variable *> used;
  for (const Variable &variable : variables)
    if (llvm::is_contained(names, variable.name))
      used.push_back(&variable);
  if (used.empty())
    used.push_back(&variables[0]);

  // The parameters of the particles that differ between the types, one
  // value for each place.
  std::vector<std::string> gathered;
  for (const std::string &name : names) {
    std::string stem = getTripletParameter(term, name);
    if (stem.empty() || llvm::is_contained(gathered, stem))
      continue;
    const Parameter &parameter = *llvm::find_if(
        parameters, [&](const Parameter &p) { return p.name == stem; });
    if (!parameter.isUniform)
      gathered.push_back(stem);
  }

  std::string p = "%t" + k + "_";
  os << "  " << result << " = md.sum_tuples " << relation
     << ", %x, %cell coordinates(";
  llvm::interleaveComma(used, os,
                        [&](const Variable *v) { os << v->coordinate; });
  os << ")";
  if (!gathered.empty()) {
    os << "\n      gather(";
    llvm::interleaveComma(gathered, os,
                          [&](const std::string &s) { os << "%p_" << s; });
    os << " : ";
    llvm::interleaveComma(gathered, os,
                          [&](const std::string &) { os << "!real"; });
    os << ")";
  }
  os << " {\n  ^bb0(";
  llvm::StringMap<std::string> values;
  for (auto [i, v] : llvm::enumerate(used))
    os << (i ? ", " : "") << p << "v" << i << ": f64";
  for (const std::string &stem : gathered)
    for (int place = 0; place != 3; ++place)
      os << ", " << p << stem << "_" << place << ": f64";
  os << "):\n";

  // The expression is evaluated in the units of the control file.
  os << "    " << p << "to_length = arith.constant "
     << formatReal(1.0 / units::length) << " : f64\n";
  for (auto [i, v] : llvm::enumerate(used)) {
    std::string argument = p + "v" + std::to_string(i);
    if (v->isLength) {
      os << "    " << argument << "_in = arith.mulf " << argument << ", " << p
         << "to_length : f64\n";
      argument += "_in";
    }
    values[v->name] = argument;
  }
  for (auto &[name, value] : term.constants) {
    os << "    " << p << "c_" << name << " = arith.constant "
       << formatReal(value) << " : f64\n";
    values[name] = p + "c_" + name;
  }
  // Particle 1 is the center, place 1; 2 and 3 are places 0 and 2.
  static const int places[] = {1, 0, 2};
  for (const std::string &name : names) {
    std::string stem = getTripletParameter(term, name);
    if (stem.empty())
      continue;
    if (llvm::is_contained(gathered, stem)) {
      values[name] = p + stem + "_" +
                     std::to_string(places[name.back() - '1']);
      continue;
    }
    const Parameter &parameter = *llvm::find_if(
        parameters, [&](const Parameter &q) { return q.name == stem; });
    os << "    " << p << "u_" << name << " = arith.constant "
       << formatReal(parameter.value) << " : f64\n";
    values[name] = p + "u_" + name;
  }
  if (control.usesTime)
    values["t"] = "%time";
  std::string value = expression.emit(os, values, p + "e", "    ", term.name);
  os << "    " << p << "to_energy = arith.constant "
     << formatReal(units::energy) << " : f64\n";
  os << "    " << p << "u = arith.mulf " << value << ", " << p
     << "to_energy : f64\n";
  os << "    md.yield " << p << "u : f64\n";
  os << "  } : !md.relation<@atoms, 3, reversal>, !vec -> f64\n";
  return result;
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

llvm::Error Builder::emitStagedModels() {
  stagedPotentials.clear();
  stagedResults.clear();
  for (const Control::ModelTerm &term : control.stagedModels) {
    auto buffer = llvm::MemoryBuffer::getFile(term.path);
    if (!buffer)
      return makeError("cannot read the model in stages '" + term.name +
                       "' from '" + term.path +
                       "': " + buffer.getError().message());
    StringRef text = (*buffer)->getBuffer();
    // The module's one potential takes the positions, the cell, and the
    // atomic numbers (scripts/mlip/painn-stages.py export).
    static const char *const head = "md.potential @";
    size_t at = text.find(head);
    if (at == StringRef::npos || text.find(head, at + 1) != StringRef::npos)
      return makeError("the model in stages '" + term.name +
                       "' must hold one 'md.potential'");
    // The potential's body goes into `@energy`, where md-differentiate
    // differentiates it with the other terms; its values are renamed apart,
    // and its arguments are the positions, the cell, and the atomic numbers
    // of the program. The rest of the module (the stages and their bodies)
    // goes to the top of the program.
    size_t open = text.find("{\n", at);
    size_t close = text.find("\n}", open);
    llvm::SmallVector<StringRef, 3> parameters;
    StringRef header = text.slice(text.find('(', at) + 1, open);
    for (StringRef part : llvm::split(header, ','))
      if (part.trim().starts_with("%"))
        parameters.push_back(part.trim().take_until([](char c) {
          return c == ':' || c == ' ';
        }));
    if (open == StringRef::npos || close == StringRef::npos ||
        parameters.size() != 3)
      return makeError("the potential of the model in stages '" + term.name +
                       "' must take the positions, the cell, and the atomic "
                       "numbers");
    std::string prefix = "%sm" + std::to_string(stagedPotentials.size()) + "_";
    std::string body;
    StringRef rest = text.slice(open + 2, close + 1);
    std::string result;
    while (!rest.empty()) {
      size_t percent = rest.find('%');
      body += rest.take_front(percent).str();
      if (percent == StringRef::npos)
        break;
      rest = rest.drop_front(percent);
      size_t length = 1;
      while (length < rest.size() &&
             (llvm::isAlnum(rest[length]) || rest[length] == '_' ||
              rest[length] == '.' || rest[length] == '$'))
        ++length;
      StringRef value = rest.take_front(length);
      rest = rest.drop_front(length);
      if (value == parameters[0])
        body += "%x";
      else if (value == parameters[1])
        body += "%cell";
      else if (value == parameters[2])
        body += "%p_model_z";
      else
        body += prefix + value.drop_front().str();
    }
    size_t ret = body.rfind("md.return ");
    if (ret == std::string::npos)
      return makeError("the potential of the model in stages '" + term.name +
                       "' has no 'md.return'");
    StringRef returned = StringRef(body).drop_front(ret + 10);
    result = returned.take_until([](char c) { return c == ' '; }).str();
    body.erase(body.rfind('\n', ret) + 1);
    os << "// The model in stages '" << term.name << "' (D294)\n"
       << text.take_front(at) << text.drop_front(close + 2) << "\n";
    stagedPotentials.push_back(body);
    stagedResults.push_back(result);
  }
  return llvm::Error::success();
}

std::string Builder::emitModelTerms(StringRef indent, StringRef x,
                                    StringRef cell, StringRef prefix,
                                    StringRef f, StringRef fResult,
                                    StringRef u, StringRef uResult,
                                    StringRef w, StringRef tag) {
  std::string t = ("_m" + tag).str();
  bool numbers = !u.empty() || !w.empty();
  // The cell, as the diagonal and the tilts.
  os << indent << "%mcell" << t << " = md_exec.cell_numbers " << cell
     << " : vector<6xf64>\n";
  for (int k = 0; k != 6; ++k)
    os << indent << "%mcell" << t << "_" << k << " = vector.extract %mcell"
       << t << "[" << k << "] : f64 from vector<6xf64>\n";
  int bits = program.state == Element::F64 ? 64 : 32;
  int forceBits = program.force == Element::F64 ? 64 : 32;
  os << indent << "%mbits" << t << " = arith.constant " << bits << " : i64\n"
     << indent << "%mfbits" << t << " = arith.constant " << forceBits
     << " : i64\n";
  std::string forces = f.str(), energy = u.str(), virial = w.str();
  for (size_t index = 0; index != control.models.size(); ++index) {
    std::string k = t + "_" + std::to_string(index);
    bool last = index + 1 == control.models.size();
    std::string next = last ? fResult.str() : "%mf" + k;
    // The model's pair list (D277): the pairs of images
    // within its cutoff, from a neighborhood at that cutoff, which the
    // program refreshes with the skin of the run.
    if (modelEdges) {
      std::string cutoff = formatReal(modelCutoffs[index]);
      os << indent << "%mn" << k << " = md.neighborhood " << x << ", " << cell
         << " cutoff(" << cutoff << ") : !vec -> !pairs\n"
         << indent << "%me" << k << " = md.edges %mn" << k << ", " << x
         << ", " << cell << " cutoff(" << cutoff
         << ") : !pairs, !vec -> !md.edges<@atoms>\n";
    }
    os << indent << "%mi" << k << " = arith.constant " << index << " : i64\n"
       << indent << next;
    if (numbers)
      os << ", %mu" << k << ", %mw" << k;
    os << " = mdrt.host_call @mdrtModelEvaluate" << (modelEdges ? "Edges" : "")
       << (numbers ? "Energy" : "") << "(%mi" << k << ", " << x << ", "
       << prefix << "model_z";
    if (modelEdges)
      os << ", %me" << k;
    for (int c = 0; c != 6; ++c)
      os << ", %mcell" << t << "_" << c;
    os << ", %mbits" << t << ", %mfbits" << t << ")\n"
       << indent << "    outs(" << forces << " : !vec) {in_place}\n"
       << indent << "    : (i64, !vec, !ids, "
       << (modelEdges ? "!md.edges<@atoms>, " : "")
       << "f64, f64, f64, f64, f64, f64, i64, i64)\n"
       << indent << "    -> (!vec" << (numbers ? ", f64, vector<9xf64>" : "")
       << ")\n";
    forces = next;
    if (!u.empty()) {
      std::string sum = last ? uResult.str() : "%mus" + k;
      os << indent << sum << " = arith.addf " << energy << ", %mu" << k
         << " : f64\n";
      energy = sum;
    }
    if (!w.empty()) {
      std::string sum = "%mws" + k;
      os << indent << sum << " = arith.addf " << virial << ", %mw" << k
         << " : vector<9xf64>\n";
      virial = sum;
    }
  }
  return virial;
}

void Builder::emitRestraints(StringRef indent, StringRef x,
                             StringRef prefix, StringRef f,
                             StringRef fResult, StringRef u,
                             StringRef uResult, StringRef w,
                             StringRef wResult) {
  std::string fields = (prefix + "rest_k, " + prefix + "rest_x, " + prefix +
                        "rest_y, " + prefix + "rest_z")
                           .str();
  std::string types = "!real, !real, !real, !real";
  std::string arguments = "%k_i: f64, %rx_i: f64, %ry_i: f64, %rz_i: f64";
  if (scalesReference()) {
    fields += (", " + prefix + "rest_cx, " + prefix + "rest_cy, " + prefix +
               "rest_cz")
                  .str();
    types += ", !real, !real, !real";
    arguments += ", %rcx_i: f64, %rcy_i: f64, %rcz_i: f64";
  }
  // d = x − x_ref, and k d. Under a barostat the fields hold the offsets o
  // of the references from their centers c, and the center follows the cell
  // while the offset stays, x_ref = s ⊙ c + o (D124).
  auto emitOffset = [&](StringRef inner) {
    std::string reference = "%ref_i";
    os << inner << reference << " = vector.from_elements %rx_i, %ry_i, %rz_i "
                   ": vector<3xf64>\n";
    if (scalesReference()) {
      os << inner << "%rc_i = vector.from_elements %rcx_i, %rcy_i, %rcz_i "
                     ": vector<3xf64>\n"
         << inner << "%rcs_i = arith.mulf " << scaleName
         << ", %rc_i : vector<3xf64>\n"
         << inner << "%refs_i = arith.addf %rcs_i, %ref_i : vector<3xf64>\n";
      reference = "%refs_i";
    }
    os << inner << "%d_i = arith.subf %x_i, " << reference
       << " : vector<3xf64>\n"
       << inner << "%kb_i = vector.broadcast %k_i : f64 to vector<3xf64>\n"
       << inner << "%kd_i = arith.mulf %kb_i, %d_i : vector<3xf64>\n";
  };
  std::string inner = (indent + "  ").str();
  os << indent << fResult << " = md.map_particles gather(" << x << ", " << f
     << ", " << fields << " : !vec, !vec, " << types << ") {\n"
     << indent << "^bb0(%x_i: vector<3xf64>, %f_i: vector<3xf64>, "
     << arguments << "):\n";
  emitOffset(inner);
  os << inner << "%two_i = arith.constant dense<2.0> : vector<3xf64>\n"
     << inner << "%pull_i = arith.mulf %two_i, %kd_i : vector<3xf64>\n"
     << inner << "%fr_i = arith.subf %f_i, %pull_i : vector<3xf64>\n"
     << inner << "md.yield %fr_i : vector<3xf64>\n"
     << indent << "} : !vec\n";
  if (u.empty() && w.empty())
    return;
  // Σ k d ⊙ d, whose sum is the energy and whose elements times −2 are the
  // diagonal of the virial. Under a barostat the virial is Σ F ⊙ (x − s ⊙ c)
  // = −2 Σ k d ⊙ (d + o), the derivative of the energy when the positions
  // and the centers of the references scale and their offsets do not.
  std::string sum = ((u.empty() ? wResult : uResult) + "_parts").str();
  os << indent << sum << " = md.sum_particles gather(" << x << ", " << fields
     << " : !vec, " << types << ") {\n"
     << indent << "^bb0(%x_i: vector<3xf64>, " << arguments << "):\n";
  emitOffset(inner);
  os << inner << "%kdd_i = arith.mulf %kd_i, %d_i : vector<3xf64>\n"
     << inner << "md.yield %kdd_i : vector<3xf64>\n"
     << indent << "} : vector<3xf64>\n";
  std::string virialSum = sum;
  if (!w.empty() && scalesReference()) {
    virialSum = (wResult + "_vparts").str();
    os << indent << virialSum << " = md.sum_particles gather(" << x << ", "
       << fields << " : !vec, " << types << ") {\n"
       << indent << "^bb0(%x_i: vector<3xf64>, " << arguments << "):\n";
    emitOffset(inner);
    os << inner << "%dpo_i = arith.addf %d_i, %ref_i : vector<3xf64>\n"
       << inner << "%kdo_i = arith.mulf %kd_i, %dpo_i : vector<3xf64>\n"
       << inner << "md.yield %kdo_i : vector<3xf64>\n"
       << indent << "} : vector<3xf64>\n";
  }
  if (!u.empty())
    os << indent << uResult << "_r = vector.reduction <add>, " << sum
       << " : vector<3xf64> into f64\n"
       << indent << uResult << " = arith.addf " << u << ", " << uResult
       << "_r : f64\n";
  if (w.empty())
    return;
  std::string zero = (wResult + "_zero").str();
  os << indent << zero << " = arith.constant 0.0 : f64\n"
     << indent << wResult << "_m2 = arith.constant -2.0 : f64\n";
  std::string diagonal[3];
  for (int c = 0; c != 3; ++c) {
    diagonal[c] = (wResult + "_d" + std::to_string(c)).str();
    os << indent << diagonal[c] << "_s = vector.extract " << virialSum
       << "[" << c << "] : f64 from vector<3xf64>\n"
       << indent << diagonal[c] << " = arith.mulf " << diagonal[c] << "_s, "
       << wResult << "_m2 : f64\n";
  }
  os << indent << wResult << "_r = vector.from_elements " << diagonal[0]
     << ", " << zero << ", " << zero << ", " << zero << ", " << diagonal[1]
     << ", " << zero << ", " << zero << ", " << zero << ", " << diagonal[2]
     << " : vector<9xf64>\n"
     << indent << wResult << " = arith.addf " << w << ", " << wResult
     << "_r : vector<9xf64>\n";
}

void Builder::emitTerms(StringRef x) {
  if (!system.topology)
    return;
  // The restraints, if any, last: their energy at the start is that of the
  // evaluation before the terms.
  // The terms given by expressions follow those of the topology, those
  // over tuples (D136) and then those over pairs (D137).
  int custom = static_cast<int>(system.topology->tupleTerms.size());
  int pairs = static_cast<int>(control.pairs.size());
  int born = static_cast<int>(system.bornTermNames.size());
  int external = static_cast<int>(system.topology->externalTerms.size());
  int size =
      14 + custom + pairs + born + external + (hasRestraints() ? 1 : 0);
  std::string type = "memref<" + std::to_string(size) + "xf64>";
  os << "  %terms = memref.alloca() : " << type << "\n";
  int index = 0;
  for (StringRef name :
       {"term_lj", "term_coulomb", "term_bonds", "term_angles",
        "term_dihedrals", "term_lj14", "term_coulomb14", "term_cmap",
        "term_excluded", "term_reciprocal", "term_urey_bradley",
        "term_impropers", "term_lj_excluded", "term_lj_reciprocal"}) {
    os << "  %" << name << " = md.evaluate @" << name << "(" << x << ", %cell"
       << getFieldValues() << getTimeValue("%time0") << ") request [energy]\n"
       << "      : (!vec, !md.cell" << getFieldTypes() << getTimeType()
       << ") -> f64\n"
       << "  %i_" << name << " = arith.constant " << index++
       << " : index\n"
       << "  memref.store %" << name << ", %terms[%i_" << name
       << "] : " << type << "\n";
  }
  for (int k = 0; k != custom + pairs + born + external; ++k) {
    std::string name =
        k < custom ? "term_custom" + std::to_string(k)
        : k < custom + pairs ? "term_pair" + std::to_string(k - custom)
        : k == custom + pairs && born > 0 ? "term_born"
        : k < custom + pairs + born
            ? "term_surface"
            : "term_external" + std::to_string(k - custom - pairs - born);
    os << "  %" << name << " = md.evaluate @" << name << "(" << x << ", %cell"
       << getFieldValues() << getTimeValue("%time0") << ") request [energy]\n"
       << "      : (!vec, !md.cell" << getFieldTypes() << getTimeType()
       << ") -> f64\n"
       << "  %i_" << name << " = arith.constant " << index++ << " : index\n"
       << "  memref.store %" << name << ", %terms[%i_" << name << "] : "
       << type << "\n";
  }
  if (hasRestraints())
    os << "  %i_restraints = arith.constant " << index << " : index\n"
       << "  memref.store %u0_r, %terms[%i_restraints] : " << type << "\n";
  os << "  %terms_cast = memref.cast %terms : " << type << " to "
        "memref<?xf64>\n"
     << "  func.call @mdrtWriteTerms(%terms_cast) : (memref<?xf64>) -> ()\n";
}
