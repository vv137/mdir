// The potentials of the program: the terms of pairs, of tuples, and of
// the topology, the Born and external terms, the centroids, the pulls,
// the observables, the staged models, and the restraints
// (docs/driver.md, Section 2.12).

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
        auto [kind, index] = builder->findObservedTerm(*this, term.name);
        if (kind)
          term.kind = kind, term.index = index;
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
    emitTopologyPotential("observe" + std::to_string(k), term.kind,
                          term.kind == TupleTerms ? term.index : -1,
                          term.kind == PairTerms ? term.index : -1,
                          term.kind == ExternalTerms ? term.index : -1);
    observedTerm.clear();
    observedConstants.clear();
  }
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
  bool lj = terms & LennardJones, coulomb = terms & Coulomb;
  bool pairTerms = llvm::any_of(getTermBuilders(), [&](const TermBuilder *builder) {
    return builder->needsPairNeighborhood(
        *this, {terms, tupleTerm, pairTerm, externalTerm, alchemicalOnly});
  });
  if (lj || coulomb || pairTerms) {
    os << "  %n = md.neighborhood %x, %cell cutoff(" << formatReal(cutoff)
       << ")";
    if (has("excluded"))
      os << " exclude(%r_excluded : !rel_excluded)";
    os << " : !vec -> !pairs\n";
  }
  for (const TermBuilder *builder : termBuildersByPotentialRank())
    if (builder->topologyRank() == 0)
      builder->emitTopologyPotential(
        *this, {terms, tupleTerm, pairTerm, externalTerm, alchemicalOnly}, addition);
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
  for (const TermBuilder *builder : termBuildersByPotentialRank())
    if (builder->topologyRank() == 1)
      builder->emitTopologyPotential(
          *this, {terms, tupleTerm, pairTerm, externalTerm, alchemicalOnly}, addition);
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
  for (const TermBuilder *builder : termBuildersByPotentialRank())
    if (builder->topologyRank() == 2)
      builder->emitTopologyPotential(
          *this, {terms, tupleTerm, pairTerm, externalTerm, alchemicalOnly},
          addition);
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
  std::vector<std::string> registeredNames;
  for (const TermBuilder *builder : termBuildersByPotentialRank())
    builder->appendTermNames(*this, registeredNames);
  unsigned storedTermCount = 0;
  for (const TermBuilder *builder : getTermBuilders())
    storedTermCount += builder->getStoredTermCount(*this);
  int size =
      14 + registeredNames.size() + storedTermCount;
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
  for (const std::string &name : registeredNames) {
    os << "  %" << name << " = md.evaluate @" << name << "(" << x << ", %cell"
       << getFieldValues() << getTimeValue("%time0") << ") request [energy]\n"
       << "      : (!vec, !md.cell" << getFieldTypes() << getTimeType()
       << ") -> f64\n"
       << "  %i_" << name << " = arith.constant " << index++ << " : index\n"
       << "  memref.store %" << name << ", %terms[%i_" << name << "] : "
       << type << "\n";
  }
  for (const TermBuilder *builder : getTermBuilders())
    builder->emitStoredTerms(*this, type, index);
  os << "  %terms_cast = memref.cast %terms : " << type << " to "
        "memref<?xf64>\n"
     << "  func.call @mdrtWriteTerms(%terms_cast) : (memref<?xf64>) -> ()\n";
}
