// Custom tuple and centroid terms (D359, D361, #683).
#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
using TupleGradient = BuildKept::TupleGradient;

namespace mdir::driver::builder {
class TupleTermBuilder final : public TermBuilder {
public:
  int potentialRank() const override { return -3; }
  int topologyRank() const override { return 1; }
  void collectTupleSets(TermParameters &context) const override;
  void refillTupleSet(TermParameters &context, Program::TupleSet &set) const override {
    const Topology &topology = context.getTopology();
    for (const TupleTerm &term : topology.tupleTerms)
      if (!term.isCentroid() && set.name == "custom_" + term.name)
        for (Program::Field &field : set.fields)
          if (field.values.empty())
            fillTupleParameter(field, term);
  }
  void emitTopologyPotential(TermEmission &context,
      const PotentialSelection &selection,
      llvm::function_ref<void(StringRef)> add) const override;
  void emitTermPotentials(TermEmission &context) const override {
    for (size_t k = 0, e = context.getTopology().tupleTerms.size(); k != e; ++k)
      context.emitTopologyPotential("term_custom" + std::to_string(k),
                                    TermInputs::TupleTerms, static_cast<int>(k));
  }
  void appendTermNames(const TermInputs &context,
                       std::vector<std::string> &names) const override {
    for (size_t k = 0, e = context.getTopology().tupleTerms.size(); k != e; ++k)
      names.push_back("term_custom" + std::to_string(k));
  }
  std::pair<unsigned, int> findObservedTerm(const TermInputs &context,
                                           StringRef name) const override {
    for (auto [k, term] : llvm::enumerate(context.getTopology().tupleTerms))
      if (term.name == name)
        return {TermInputs::TupleTerms, static_cast<int>(k)};
    return {0, -1};
  }
  std::optional<double> getObservedValue(const TermInputs &context, unsigned kind,
                                        int index, StringRef name) const override {
    if (kind == TermInputs::TupleTerms)
      for (const auto &[key, values] : context.getTopology().tupleTerms[index].parameters)
        if (key == name)
          return values.front();
    return std::nullopt;
  }
  llvm::Expected<std::optional<Program::GradientOutcome>>
  getTunableOutcome(const TermInputs &context, unsigned k) const override {
    const Control &control = context.getControl();
    const auto &declaration = control.tunableDeclarations[k];
    if (declaration.kind != Control::TunableKind::TupleParameter)
      return std::optional<Program::GradientOutcome>();
    // A parameter that the expression of its term reads has its fields
    // (collectTopology); one that it does not read enters no value.
    const TupleTerm &term =
        context.getTopology().tupleTerms[declaration.termIndex];
    if (term.isCentroid())
      return makeError(
          "System.tunable_gradient: the derivative of the energy in the "
          "tunable '" + declaration.name + "', a parameter of the term '" +
          term.name + "' over centers of groups, is not implemented yet");
    if (llvm::any_of(context.getTupleGradients(), [&](const TupleGradient &gradient) {
          return gradient.tunable == k;
        }))
      return std::optional(Program::GradientOutcome::Rule);

    return std::optional(Program::GradientOutcome::Zero);
  }
private:
  static void fillTupleParameter(Program::Field &, const TupleTerm &);
  static void emitCentroidTerm(TermEmission &, size_t, const TupleTerm &);
};
} // namespace mdir::driver::builder

const TermBuilder &mdir::driver::builder::getTupleTermBuilder() {
  static const TupleTermBuilder builder;
  return builder;
}

void TupleTermBuilder::fillTupleParameter(Program::Field &field,
                                 const TupleTerm &term) {
  for (const auto &[name, values] : term.parameters)
    if (name == field.name)
      field.values = values;
}


void TupleTermBuilder::collectTupleSets(TermParameters &context) const {
  const Topology &topology = context.getTopology();
  const Control &control = context.getControl();
  auto addSet = [&](StringRef name, unsigned arity) -> Program::TupleSet & {
    Program::TupleSet set;
    set.name = name.str();
    set.arity = arity;
    context.getTupleSets().push_back(std::move(set));
    return context.getTupleSets().back();
  };
  auto addField = [&](Program::TupleSet &set, StringRef name) {
    Program::Field field;
    field.name = name.str();
    set.fields.push_back(std::move(field));
    return set.fields.size() - 1;
  };
  // The terms given by expressions (D136), with their parameters as fields
  // of their tuples, in the units of the control file.
  for (const TupleTerm &term : topology.tupleTerms) {
    // A term over the centers of groups (D139), one set of pairs: for
    // each group the pairs of a member and the reference of the group, and
    // for each group after the first the pair of its reference and that of
    // the first. A field for each group and each such link, `w<k>`, holds
    // the weight of the member in its pairs, 1 in the pair of the link,
    // and 0 elsewhere, so that one loop gives all the sums.
    if (term.isCentroid()) {
      size_t groups = term.centers.size();
      Program::TupleSet &set = addSet("cg_" + term.name, 2);
      set.oriented = true;
      set.reversible = false;
      for (size_t k = 0; k != 2 * groups - 1; ++k)
        addField(set, "w" + std::to_string(k));
      auto addPair = [&](unsigned i, unsigned j, size_t slot, double weight) {
        set.members.push_back(i);
        set.members.push_back(j);
        for (size_t k = 0; k != 2 * groups - 1; ++k)
          set.fields[k].values.push_back(k == slot ? weight : 0.0);
      };
      for (auto [g, center] : llvm::enumerate(term.centers)) {
        for (auto [i, weight] :
             llvm::zip_equal(center.members, center.weights))
          if (i != center.reference)
            addPair(i, center.reference, g, weight);
        if (g > 0)
          addPair(center.reference, term.centers.front().reference,
                  groups + g - 1, 1.0);
      }
      continue;
    }
    Program::TupleSet &set = addSet("custom_" + term.name, term.arity);
    for (unsigned member : term.particles)
      set.members.push_back(member);
    for (const auto &[name, values] : term.parameters)
      fillTupleParameter(set.fields[addField(set, name)], term);
    // The tunable parameters of the term whose derivatives the program
    // carries (D230): for each, the place whose members are
    // shared by the fewest tuples, and for each tuple its number among the
    // tuples of its member there.
    if (context.hasTunableGradient())
      for (auto [k, declaration] :
           llvm::enumerate(control.tunableDeclarations)) {
        if (declaration.kind != Control::TunableKind::TupleParameter ||
            &topology.tupleTerms[declaration.termIndex] != &term)
          continue;
        Expression expression = llvm::cantFail(
            Expression::parse(term.expression, control.functions));
        if (!llvm::is_contained(expression.getNames(), declaration.parameter))
          continue;
        size_t tuples = term.particles.size() / term.arity;
        unsigned place = 0, count = 0;
        std::vector<double> colors;
        for (unsigned candidate = 0; candidate != term.arity; ++candidate) {
          std::map<unsigned, unsigned> seen;
          std::vector<double> numbers;
          unsigned most = 0;
          for (size_t t = 0; t != tuples; ++t) {
            unsigned number = seen[term.particles[t * term.arity + candidate]]++;
            numbers.push_back(number);
            most = std::max(most, number + 1);
          }
          if (candidate == 0 || most < count) {
            place = candidate;
            count = most;
            colors = std::move(numbers);
          }
        }
        TupleGradient gradient;
        gradient.tunable = static_cast<unsigned>(k);
        gradient.term = declaration.termIndex;
        gradient.parameter = declaration.parameter;
        gradient.place = place;
        gradient.count = count;
        gradient.first = static_cast<unsigned>(context.getGradientFields().size());
        for (unsigned c = 0; c != count; ++c) {
          Program::GradientField field;
          field.tunable = gradient.tunable;
          for (size_t t = 0; t != tuples; ++t)
            if (colors[t] == c)
              field.sites.push_back(
                  {static_cast<uint32_t>(t),
                   term.particles[t * term.arity + place]});
          context.getGradientFields().push_back(std::move(field));
        }
        size_t field = addField(set, "tgc_" + declaration.parameter);
        set.fields[field].values = std::move(colors);
        context.getMutableTupleGradients().push_back(std::move(gradient));
      }
  }

}

void TupleTermBuilder::emitCentroidTerm(TermEmission &context, size_t index, const TupleTerm &term) {
  const Control &control = context.getControl();
  llvm::raw_ostream &os = context.getOutput();
  // Numbers outside the kernels, named after the term.
  std::string prefix = "%cb" + std::to_string(index) + "_";
  llvm::StringMap<std::string> values =
      context.emitCentroidCoordinates(term, "  ", "%x", "%cell", "%r_", prefix);
  for (auto [k, parameter] : llvm::enumerate(term.parameters)) {
    std::string name = prefix + "p" + std::to_string(k);
    os << "  " << name << " = arith.constant "
       << formatReal(parameter.second.front()) << " : f64\n";
    values[parameter.first] = name;
  }
  if (control.usesTime)
    values["t"] = "%time";
  context.bindLambdas(values);
  context.bindObserved(term.name, values);
  Expression expression = llvm::cantFail(
      Expression::parse(term.expression, control.functions));
  std::string energy = expression.emit(os, values, prefix + "e", "  ", term.name);
  os << "  " << prefix << "kj = arith.constant " << formatReal(units::energy)
     << " : f64\n"
     << "  %u_centroid_" << term.name << " = arith.mulf " << energy << ", "
     << prefix << "kj : f64\n";
}


void TupleTermBuilder::emitTopologyPotential(TermEmission &context,
    const PotentialSelection &selection,
    llvm::function_ref<void(StringRef)> add) const {
  const System &system = context.getSystem();
  const Control &control = context.getControl();
  llvm::raw_ostream &os = context.getOutput();
  unsigned terms = selection.terms;
  int tupleTerm = selection.tupleTerm;
  bool alchemicalOnly = selection.alchemicalOnly;
  // The terms given by expressions (D136), in the units of the control
  // file: the distance in Å, the angles in radians, the energy in kcal/mol.
  for (auto [index, term] : llvm::enumerate(system.topology->tupleTerms)) {
    if (!(terms & TermInputs::TupleTerms) ||
        (tupleTerm >= 0 && static_cast<int>(index) != tupleTerm))
      continue;
    if (!context.includesTuple(static_cast<unsigned>(index)))
      continue;
    if (alchemicalOnly && !context.usesLambda(term.expression))
      continue;
    if (term.isCentroid()) {
      emitCentroidTerm(context, index, term);
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
    if (context.hasGradientArguments())
      for (const TupleGradient &gradient : context.getTupleGradients())
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
    context.bindLambdas(values);
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
    context.bindObserved(term.name, values);
    Expression expression = llvm::cantFail(Expression::parse(term.expression, control.functions));
    std::string energy = expression.emit(os, values, "%ce", "    ", term.name);
    os << "    %c_kj = arith.constant " << formatReal(units::energy)
       << " : f64\n"
       << "    %c_e = arith.mulf " << energy << ", %c_kj : f64\n"
       << "    md.yield %c_e : f64\n"
       << "  } : !rel_" << set << ", !vec -> f64\n";
    add(set);
  }

}
