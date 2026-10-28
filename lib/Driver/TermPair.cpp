// Custom pair terms and their bounded scalar tail services (D359, D361, #683).
#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

/// The overrides of the pair term `term`: those that name it, and those
/// that name no term where there is one term.
static std::vector<const PairOverride *>
getOverrides(const Control &control, const PairTerm &term) {
  std::vector<const PairOverride *> found;
    for (const PairOverride &entry : control.overrides)
      if (entry.term == term.name ||
          (entry.term.empty() && control.pairs.size() == 1))
        found.push_back(&entry);
    return found;
  }

  /// The index of the type named `name`, or -1.
  static int findType(const Control &control, StringRef name) {
    for (auto [index, type] : llvm::enumerate(control.types))
      if (type.name == name)
        return static_cast<int>(index);
    return -1;
  }

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


  namespace mdir::driver::builder {
  class PairTermBuilder final : public TermBuilder {
  public:
    int parameterRank() const override { return 1; }
    int potentialRank() const override { return -2; }
    llvm::Error collectBareParameters(TermParameters &context) const override {
      const Control &control = context.getControl();
      const System &system = context.getSystem();
      auto &parameters = context.getMutableTypeParameters();
      auto &termTables = context.getMutablePairPlan().tables;
      auto &expressions = context.getMutablePairPlan().expressions;
    for (const PairOverride &entry : control.overrides) {
      if (entry.term.empty() && control.pairs.size() != 1)
        return makeError("[[energy.nbfix]] needs 'pair' to name the pair term "
                         "when there is more than one");
      if (!entry.term.empty() &&
          llvm::none_of(control.pairs, [&](const PairTerm &term) {
            return term.name == entry.term;
          }))
        return makeError("[[energy.nbfix]] names the pair term '" + entry.term +
                         "', which does not exist");
      for (const std::string &name : {entry.first, entry.second})
        if (findType(control, name) < 0)
          return makeError("[[energy.nbfix]] names the type '" + name +
                           "', which does not exist");
    }

    for (const PairTerm &term : control.pairs) {
      auto expression = Expression::parse(term.expression, control.functions);
      if (!expression)
        return expression.takeError();
      std::vector<const PairOverride *> overrides = getOverrides(control, term);
      llvm::StringMap<unsigned> tables;

      for (const std::string &name : expression->getNames()) {
        if (name == "r" || name == "coulomb")
          continue;
        if (llvm::any_of(term.constants,
                         [&](auto &constant) { return constant.first == name; }))
          continue;

        // A parameter of the types. Every type must have it.
        std::vector<double> values;
        for (const ParticleType &type : control.types) {
          auto found = llvm::find_if(type.parameters, [&](auto &entry) {
            return entry.first == name;
          });
          if (found == type.parameters.end())
            return makeError("the expression of '" + term.name + "' uses '" +
                             name + "', which is neither 'r', nor 'coulomb', "
                             "nor a number of the term, nor a parameter of the "
                             "type '" + type.name + "'");
          values.push_back(found->second);
        }
        bool isUniform = llvm::all_of(
            values, [&](double value) { return value == values.front(); });
        if (!isUniform && !term.mixing.count(name))
          return makeError("'" + name + "' differs between the types, but the "
                           "term '" + term.name + "' has no rule of 'mixing' "
                           "for it");

        // A term with overrides looks every parameter up in a table of the
        // pairs of types (D57).
        if (!overrides.empty()) {
          Program::Table table;
          table.name = term.name + "_" + name;
          table.count = control.types.size();
          table.values.resize(table.count * table.count);
          for (unsigned a = 0; a != table.count; ++a)
            for (unsigned b = 0; b != table.count; ++b)
              table.values[a * table.count + b] =
                  isUniform ? values.front()
                            : mix(term.mixing.lookup(name), values[a], values[b]);
          for (const PairOverride *entry : overrides)
            for (auto &[parameter, value] : entry->parameters)
              if (parameter == name) {
                unsigned a = findType(control, entry->first);
                unsigned b = findType(control, entry->second);
                table.values[a * table.count + b] = value;
                table.values[b * table.count + a] = value;
              }
          tables[name] = context.getTables().size();
          context.getTables().push_back(std::move(table));
          continue;
        }

        if (llvm::any_of(parameters,
                         [&](Parameter &known) { return known.name == name; }))
          continue;
        Parameter parameter;
        parameter.name = name;
        parameter.value = values.front();
        parameter.isUniform = isUniform;
        if (!parameter.isUniform) {
          parameter.field = context.getFields().size();
          Program::Field field;
          field.name = name;
          for (unsigned type : system.types)
            field.values.push_back(values[type]);
          context.getFields().push_back(std::move(field));
        }
        parameters.push_back(std::move(parameter));
      }

      // Every parameter that an override sets is one that the term uses.
      for (const PairOverride *entry : overrides)
        for (auto &[parameter, value] : entry->parameters)
          if (!tables.count(parameter))
            return makeError("[[energy.nbfix]] sets '" + parameter +
                             "', which is not a parameter of the types that "
                             "the term '" + term.name + "' uses");

      // The types of the particles, which the lookups take.
      if (!tables.empty() &&
          llvm::none_of(context.getFields(),
                        [](const Program::Field &field) {
                          return field.isInteger;
                        })) {
        Program::Field field;
        field.name = "type";
        field.isInteger = true;
        for (unsigned type : system.types)
          field.values.push_back(type);
        context.getFields().push_back(std::move(field));
      }
      termTables.push_back(std::move(tables));
      expressions.push_back(std::move(*expression));
    }


    return llvm::Error::success();
  }
  llvm::Error collectBareConstants(TermParameters &context) const override {
    return computeDispersion(context);
  }
  void collectParameters(TermParameters &context) const override {
    const Control &control = context.getControl();
    const System &system = context.getSystem();
    // The interaction groups of the pair terms, a flag for each particle
    // (D137).
    for (auto [k, groups] : llvm::enumerate(system.pairGroups))
      for (auto [g, flags] : llvm::enumerate(groups)) {
        Program::Field field;
        field.name = "pg" + std::to_string(k) + (g == 0 ? "a" : "b");
        for (bool flag : flags)
          field.values.push_back(flag ? 1.0 : 0.0);
        context.getFields().push_back(std::move(field));
      }

    // The parameters of each particle that the pair terms take, `w1` and
    // `w2` of `w`, a field gathered for both particles (D165).
    for (const auto &[name, values] : system.particleParameters) {
      bool used = llvm::any_of(control.pairs, [&](const PairTerm &term) {
        Expression expression = llvm::cantFail(
            Expression::parse(term.expression, control.functions));
        return llvm::is_contained(expression.getNames(), name + "1") ||
               llvm::is_contained(expression.getNames(), name + "2");
      });
      if (!used)
        continue;
      Program::Field field;
      field.name = "pp_" + name;
      field.values = values;
      context.getFields().push_back(std::move(field));
    }


  }
  void collectTopologyTables(TermParameters &context) const override {
    const Control &control = context.getControl();
    // The tunable constants of the pair terms (D213), a row that
    // the kernels read in place of constants of their text.
    if (!control.tunableConstants.empty()) {
      Program::Table table;
      table.name = "tunable_constants";
      table.count = 1;
      table.columns = control.tunableConstants.size();
      refillTable(context, table);
      context.getTables().push_back(std::move(table));
    }


  }
  bool refillTable(const TermInputs &context, Program::Table &table) const override {
    if (table.name != "tunable_constants")
      return false;
    const Control &control = context.getControl();

    // The tunable constants of the pair terms (D213), a row that the
    // kernels read in place of constants of their text.
    table.values.clear();
    for (const auto &[term, name] : control.tunableConstants)
      for (const auto &[key, value] : control.pairs[term].constants)
        if (key == name)
          table.values.push_back(value);

    return true;
  }
  bool needsPairNeighborhood(const TermInputs &context,
                            const PotentialSelection &selection) const override {
    return (selection.terms & TermInputs::PairTerms) && !context.getControl().pairs.empty();
  }
  void emitTopologyPotential(TermEmission &context,
      const PotentialSelection &selection,
      llvm::function_ref<void(StringRef)> add) const override {
    const Control &control = context.getControl();
    const System &system = context.getSystem();
    llvm::raw_ostream &os = context.getOutput();
    bool pairTerms = (selection.terms & TermInputs::PairTerms) && !control.pairs.empty();
    int pairTerm = selection.pairTerm;
    bool alchemicalOnly = selection.alchemicalOnly;
    Truncation truncation = context.getPairTruncation();
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
      if (!context.includesPair(static_cast<unsigned>(k)))
        continue;
      if (alchemicalOnly && !context.usesLambda(term.expression))
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
      context.bindLambdas(values);
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
        if (tunable != control.tunableConstants.end() && context.hasGradientArguments()) {
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
      context.bindObserved(term.name, values);
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

  }
  void emitBareContributions(TermEmission &context, std::string &total) const override {
    const Control &control = context.getControl();
    const auto &expressions = context.getPairPlan().expressions;
    const auto &termTables = context.getPairPlan().tables;
    const auto &parameters = context.getTypeParameters();
    llvm::raw_ostream &os = context.getOutput();
    std::string truncation = getTruncation(control);
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
           << context.getParameterTables()[table].name
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

  }
  void emitTermPotentials(TermEmission &context) const override {
    for (size_t k = 0, e = context.getControl().pairs.size(); k != e; ++k)
      context.emitTopologyPotential("term_pair" + std::to_string(k),
                                    TermInputs::PairTerms, -1, static_cast<int>(k));
  }
  void appendTermNames(const TermInputs &context, std::vector<std::string> &names) const override {
    for (size_t k = 0, e = context.getControl().pairs.size(); k != e; ++k)
      names.push_back("term_pair" + std::to_string(k));
  }
  std::pair<unsigned, int> findObservedTerm(const TermInputs &context, StringRef name) const override {
    for (auto [k, term] : llvm::enumerate(context.getControl().pairs))
      if (term.name == name)
        return {TermInputs::PairTerms, static_cast<int>(k)};
    return {0, -1};
  }
  std::optional<double> getObservedValue(const TermInputs &context, unsigned kind,
                                        int index, StringRef name) const override {
    if (kind == TermInputs::PairTerms)
      for (const auto &[key, value] : context.getControl().pairs[index].constants)
        if (key == name)
          return value;
    return std::nullopt;
  }
  llvm::Error collectDispersionConstants(TermParameters &context) const override {
    const System &system = context.getSystem();
    bool shifted = context.getControl().truncation == Truncation::Shift;
    // The tails of the pair terms (D209).
    if (llvm::Error error = collectPairTails(context))
      return error;
    for (unsigned index = 0, e = system.pairTails.size(); index != e;
         ++index) {
      PairTermTail tail = llvm::cantFail(getCustomPairTail(context, index, {}));
      context.addDispersion(tail.energy + (shifted ? tail.shift : 0.0),
                            shifted ? tail.shiftFixed : 0.0, tail.virial);
    }

    return llvm::Error::success();
  }
  void collectObservedConstants(TermParameters &context) const override {
    const Control &control = context.getControl();
    const System &system = context.getSystem();
    for (auto [column, observable] : llvm::enumerate(control.observables))
      for (auto [index, term] : llvm::enumerate(control.pairs)) {
        if (term.name != observable.term || index >= system.pairTails.size())
          continue;
        if (observable.constant.empty()) {
          PairTermTail tail = llvm::cantFail(getCustomPairTail(context, index, {}));
          context.getObservableVolumeConstants()[column] = tail.energy + tail.shift;
          context.getObservableFixedConstants()[column] = tail.shiftFixed;
          continue;
        }
        double value = 0.0;
        for (const auto &[name, v] : term.constants)
          if (name == observable.constant)
            value = v;
        context.getObservableVolumeConstants()[column] =
            getCustomPairTailDerivative(context, index, observable.constant, value, true, false);
        context.getObservableFixedConstants()[column] = getCustomPairTailDerivative(context,
            index, observable.constant, value, /*shift=*/true, /*fixed=*/true);
      }
  }
  void collectGradientShiftConstants(TermParameters &context) const override {
    const System &system = context.getSystem();
    for (unsigned index = 0, e = system.pairTails.size(); index != e;
         ++index) {
      PairTermTail pair = llvm::cantFail(getCustomPairTail(context, index, {}));
      context.addGradientShift(pair.shift, pair.shiftFixed);
    }
  }
  llvm::Expected<std::optional<Program::GradientOutcome>>
  collectTunableOutcome(TermParameters &context, unsigned k) const override {
    const Control &control = context.getControl();
    const System &system = context.getSystem();
    const auto &declaration = control.tunableDeclarations[k];
    if (declaration.kind != Control::TunableKind::PairConstant)
      return getTunableOutcome(context, k);
    bool dispersion = control.topologyDispersion != DispersionCorrection::None;
    // A constant that the expression of its term does not read enters
    // no value of the program.
    unsigned index = declaration.termIndex;
    const PairTerm &term = control.pairs[index];
    Expression expression =
        llvm::cantFail(Expression::parse(term.expression, control.functions));
    if (!llvm::is_contained(expression.getNames(), declaration.parameter))
      return std::optional(Program::GradientOutcome::Zero);

    auto column = llvm::find(
        control.tunableConstants,
        std::pair<unsigned, std::string>(index, declaration.parameter));
    double value = 0.0;
    for (const auto &[name, v] : term.constants)
      if (name == declaration.parameter)
        value = v;
    // The tail of the term and the estimate of its shift, which the
    // host adds to the energy, by the quadrature that computes them
    // (D209, D210).
    Program::GradientSlot slot;
    slot.tunable = static_cast<unsigned>(k);
    if (dispersion && index < context.getPairPlan().tailExpressions.size() &&
        index < system.pairTails.size())
      slot.volumeConstant =
          getCustomPairTailDerivative(context, index, declaration.parameter, value, true, false);
    if (dispersion && index < context.getPairPlan().tailExpressions.size() &&
        index < system.pairTails.size())
      slot.fixedConstant =
          getCustomPairTailDerivative(context, index, declaration.parameter, value,
                                /*shift=*/true, /*fixed=*/true);
    if (!std::isfinite(slot.volumeConstant) ||
        !std::isfinite(slot.fixedConstant))
      return makeError("the derivative of the tail of the pair term '" +
                       term.name + "' in its constant '" +
                       declaration.parameter +
                       "' cannot be integrated, for the tunable '" +
                       declaration.name + "'");
    context.getGradientSlots().push_back(slot);
    context.getGradientColumns().push_back(
        static_cast<unsigned>(column - control.tunableConstants.begin()));

    return std::optional(Program::GradientOutcome::Rule);
  }
private:
  static llvm::Error collectPairTails(TermParameters &);
  static llvm::Error collectPairValues(const TermInputs &, unsigned,
      std::vector<std::vector<llvm::StringMap<double>>> &);
  static llvm::Error computeDispersion(TermParameters &);
};
} // namespace mdir::driver::builder

const TermBuilder &mdir::driver::builder::getPairTermBuilder() {
  static const PairTermBuilder builder;
  return builder;
}

llvm::Error PairTermBuilder::collectPairValues(const TermInputs &context,
    unsigned index, std::vector<std::vector<llvm::StringMap<double>>> &values) {
  const Control &control = context.getControl();
  const auto &parameters = context.getTypeParameters();
  const auto &termTables = context.getPairPlan().tables;
  const PairTerm &term = control.pairs[index];
  unsigned count = control.types.size();
  values.assign(count, std::vector<llvm::StringMap<double>>(count));
  for (unsigned a = 0; a != count; ++a)
    for (unsigned b = 0; b != count; ++b) {
      llvm::StringMap<double> &pair = values[a][b];
      pair["coulomb"] = coulombConstant;
      for (auto &[name, value] : term.constants)
        pair[name] = value;
      for (const Parameter &parameter : parameters) {
        auto valueOf = [&](unsigned type) {
          for (auto &[name, value] : control.types[type].parameters)
            if (name == parameter.name)
              return value;
          return 0.0;
        };
        pair[parameter.name] =
            parameter.isUniform
                ? parameter.value
                : mix(term.mixing.lookup(parameter.name), valueOf(a),
                      valueOf(b));
      }
      for (auto &[name, table] : termTables[index])
        pair[name] = context.getParameterTables()[table].values[a * count + b];
    }
  return llvm::Error::success();
}

llvm::Error PairTermBuilder::computeDispersion(TermParameters &context) {
  const Control &control = context.getControl();
  const System &system = context.getSystem();
  const auto &expressions = context.getPairPlan().expressions;
  bool corrects = llvm::any_of(control.pairs, [](const PairTerm &term) {
    return term.dispersion != DispersionCorrection::None;
  });
  if (!corrects)
    return llvm::Error::success();
  if (control.truncation != Truncation::None)
    return makeError("'dispersion_correction' needs a plain cutoff: "
                     "'switch_distance' equal to 'cutoff', and no "
                     "'lennard_jones_modifier'");

  // Beyond the cutoff the term is taken to be its dispersion, −C6 / r⁶,
  // and the density to be uniform [AllenTildesley2017, GromacsManual2025]:
  //
  //   E = −(2π / 3V) N² ⟨C6⟩ / rc³        W = 6 E
  //
  // with ⟨C6⟩ the mean over the pairs of distinct particles, as GROMACS
  // takes it, so that the pressure changes by 2E / V.
  // The repulsion beyond the cutoff is left out, as the engines leave it
  // out. C6 is the limit of −r⁶ u(r); a term that does not decay as 1/r⁶ is
  // an error. The expression is in the units of the control file.
  unsigned count = control.types.size();
  std::vector<double> numbers(count, 0.0);
  for (unsigned type : system.types)
    numbers[type] += 1.0;
  double rc = control.cutoffDistance;
  double volume = system.box[0] * system.box[1] * system.box[2] /
                  (units::length * units::length * units::length);

  double energy = 0.0;
  double n = static_cast<double>(system.getNumParticles());
  double scale = n > 1.0 ? n / (n - 1.0) : 0.0;
  for (unsigned index = 0, e = control.pairs.size(); index != e; ++index) {
    const PairTerm &term = control.pairs[index];
    if (term.dispersion == DispersionCorrection::None)
      continue;
    std::vector<std::vector<llvm::StringMap<double>>> values;
    if (llvm::Error error = collectPairValues(context, index, values))
      return error;
    const Expression &expression = expressions[index];
    for (unsigned a = 0; a != count; ++a)
      for (unsigned b = 0; b != count; ++b) {
        llvm::StringMap<double> pair = values[a][b];
        auto c6At = [&](double r) {
          pair["r"] = r;
          return -std::pow(r, 6) * expression.evaluate(pair);
        };
        double near = c6At(1.0e3 * rc), far = c6At(1.0e4 * rc);
        if (!std::isfinite(near) || !std::isfinite(far) ||
            std::fabs(near - far) > 1.0e-9 * std::max(std::fabs(far), 1.0e-300))
          return makeError("'dispersion_corr' of the term '" + term.name +
                           "' needs a term that decays as 1/r^6 beyond the "
                           "cutoff");
        double pairs = numbers[a] * (numbers[b] - (a == b ? 1.0 : 0.0));
        energy += -2.0 * M_PI / (3.0 * volume) * scale * pairs * far /
                  (rc * rc * rc);
      }
  }
  context.setBareDispersion(energy * units::energy, 6.0 * energy * units::energy);
  return llvm::Error::success();
}

/// The integral of `f` over [0, 1] by adaptive Gauss–Kronrod quadrature
/// of 7 and 15 points [Piessens1983], to an error of 1e-13 of the whole or
/// of the integral of `size`, the size of the terms that `f` sums (#155);
/// NaN if it reaches neither. A difference of two terms that nearly cancel,
/// as an NBFIX correction at the force field's own parameters, has a value
/// at the level of their rounding, which no relative error of the value
/// could reach. The rule of 15 points is exact for polynomials of degree 22
/// and that of 7 for degree 13, so a polynomial of degree 13 or less takes
/// one interval and its value to round-off.
static double integrateUnit(llvm::function_ref<double(double)> f,
                            llvm::function_ref<double(double)> size) {
  static const double xk[8] = {
      0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
      0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
      0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
      0.207784955007898467600689403773245, 0.0};
  static const double wk[8] = {
      0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
      0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
      0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
      0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
  static const double wg[4] = {
      0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
      0.381830050505118944950369775488975, 0.417959183673469387755102040816327};
  struct Interval {
    double a, b, value, error, size;
  };
  auto rule = [&](double a, double b) {
    double center = 0.5 * (a + b), half = 0.5 * (b - a);
    double fc = f(center);
    double kronrod = wk[7] * fc, gauss = wg[3] * fc;
    double sized = wk[7] * size(center);
    for (int k = 0; k != 7; ++k) {
      double left = center - half * xk[k], right = center + half * xk[k];
      double sum = f(left) + f(right);
      kronrod += wk[k] * sum;
      sized += wk[k] * (size(left) + size(right));
      if (k % 2 == 1)
        gauss += wg[k / 2] * sum;
    }
    return Interval{a, b, kronrod * half,
                    std::fabs((kronrod - gauss) * half), sized * half};
  };
  std::vector<Interval> intervals = {rule(0.0, 1.0)};
  for (int iteration = 0; iteration != 4000; ++iteration) {
    double value = 0.0, error = 0.0, sized = 0.0;
    for (const Interval &interval : intervals) {
      value += interval.value;
      error += interval.error;
      sized += interval.size;
    }
    if (!std::isfinite(value) || !std::isfinite(error))
      return std::nan("");
    if (error <= 1.0e-13 * std::fmax(std::fabs(value), sized) ||
        error == 0.0)
      return value;
    auto worst = std::max_element(
        intervals.begin(), intervals.end(),
        [](const Interval &x, const Interval &y) { return x.error < y.error; });
    double a = worst->a, b = worst->b, middle = 0.5 * (a + b);
    if (!(middle > a && middle < b))
      return std::nan("");
    *worst = rule(a, middle);
    intervals.push_back(rule(middle, b));
  }
  return std::nan("");
}

llvm::Error PairTermBuilder::collectPairTails(TermParameters &context) {
  const Control &control = context.getControl();
  auto &tailExpressions = context.getMutablePairPlan().tailExpressions;
  // The pairs that each term counts, from the system
  // (D209); a term that the correction leaves out
  // has none.
  for (const PairTerm &term : control.pairs)
    tailExpressions.push_back(llvm::cantFail(
        Expression::parse(term.expression, control.functions)));
  for (auto [index, term] : llvm::enumerate(control.pairs)) {
    auto tail = getCustomPairTail(context, index, {});
    if (!tail) {
      llvm::consumeError(tail.takeError());
      return makeError("the correction for the dispersion integrates the "
                       "pair term '" + term.name + "' beyond the cutoff, "
                       "and its tail cannot be integrated; give "
                       "'dispersion_correction = \"NONE\"' in the term to "
                       "leave it out");
    }
  }
  return llvm::Error::success();
}

llvm::Expected<PairTermTail>
mdir::driver::builder::getCustomPairTail(const TermInputs &context, unsigned index,
                     const llvm::StringMap<double> &changes) {
  const Control &control = context.getControl();
  const System &system = context.getSystem();
  const auto &tailExpressions = context.getPairPlan().tailExpressions;
  auto &tailIntegrals = context.getPairPlan().tailIntegrals;
  // The tail of a pair term goes with the term
  // (D243).
  if (!context.includesPair(index))
    return PairTermTail();
  // With a uniform density beyond the cutoff [AllenTildesley2017], each
  // unordered pair adds (4π / V) I, I = ∫_rc^∞ r² u(r) dr, to the energy
  // and (4π / V)(3 I + rc³ u(rc)) to the trace of the virial, which is
  // −(4π / V) ∫_rc^∞ r³ u'(r) dr integrated by parts; for −C6 / r⁶ that is
  // 6 times the energy, as for the topology's correction. The sum takes
  // the factor N² / P of the topology's correction, P the ordered pairs
  // that are not excluded, so that both follow one convention. In
  // s = rc / r, I = rc³ ∫_0^1 u(rc / s) s⁻⁴ ds, whose integrand is a
  // polynomial in s for a sum of powers r⁻ᵏ, k ≥ 4, which the quadrature
  // integrates exactly. A shift to 0 at the cutoff takes u(r_c) from each
  // pair within it, at a uniform density (4π / 3V) r_c³ u(r_c) for each
  // pair, with the same factor, times getShiftFactor() for the particle
  // itself (D210).
  const Expression &expression = tailExpressions[index];
  double rc = control.cutoffDistance;
  double energy = 0.0, virial = 0.0, shift = 0.0;
  if (tailIntegrals.size() <= index)
    tailIntegrals.resize(index + 1);
  for (const System::TailPair &pair : system.pairTails[index]) {
    llvm::StringMap<double> values = pair.values;
    for (const auto &entry : changes)
      if (values.count(entry.getKey()))
        values[entry.getKey()] = entry.getValue();
    // The integral is a function of the values of the names that the
    // expression reads, in their order, and of nothing else of the pair:
    // one that was computed for these values is taken, the same number to
    // the bit.
    std::vector<uint64_t> key;
    for (const std::string &name : expression.getNames()) {
      auto found = values.find(name);
      uint64_t bits = 0;
      if (found != values.end())
        std::memcpy(&bits, &found->second, sizeof bits);
      key.push_back(bits);
      key.push_back(found != values.end());
    }
    auto [memo, computes] = tailIntegrals[index].try_emplace(std::move(key));
    PairTermPlan::TailIntegral &tail = memo->second;
    if (computes && hasFiniteTail(expression, values, rc)) {
      auto u = [&](double r) {
        values["r"] = r;
        return expression.evaluate(values);
      };
      auto size = [&](double r) {
        values["r"] = r;
        return expression.evaluateMagnitude(values);
      };
      tail.finite = true;
      tail.integral = integrateUnit(
          [&](double x) {
            if (x == 0.0)
              return 0.0;
            return rc * rc * rc * u(rc / x) / (x * x * x * x);
          },
          [&](double x) {
            if (x == 0.0)
              return 0.0;
            return rc * rc * rc * size(rc / x) / (x * x * x * x);
          });
      tail.edge = rc * rc * rc * u(rc);
    }
    if (!tail.finite)
      return makeError("diverges");
    double integral = tail.integral, edge = tail.edge;
    if (!std::isfinite(integral) || !std::isfinite(edge))
      return makeError("cannot be integrated");
    energy += pair.count * integral;
    virial += pair.count * (3.0 * integral + edge);
    shift += pair.count * edge / 3.0;
  }
  const Topology &topology = *system.topology;
  double n = static_cast<double>(topology.getNumParticles());
  double ordered = n * (n - 1.0) - 2.0 * topology.exclusions.size();
  double scale = ordered > 0.0 ? n * n / ordered : 0.0;
  double volume = system.box[0] * system.box[1] * system.box[2] /
                  (units::length * units::length * units::length);
  double factor = 4.0 * M_PI / volume * scale * units::energy;
  PairTermTail tail;
  tail.energy = factor * energy;
  tail.virial = factor * virial;
  tail.shift = factor * shift;
  tail.shiftFixed = factor * shift * context.getShiftFixedFactor();
  return tail;
}

double mdir::driver::builder::getCustomPairTails(const TermInputs &context, const llvm::StringMap<double> &changes,
                             bool fixed) {
  const System &system = context.getSystem();
  // A tail that a state makes diverge is NaN in the output, not 0.
  double sum = 0.0;
  for (unsigned index = 0, e = system.pairTails.size(); index != e; ++index) {
    auto tail = getCustomPairTail(context, index, changes);
    if (!tail) {
      llvm::consumeError(tail.takeError());
      return std::nan("");
    }
    sum += fixed ? tail->shiftFixed : tail->energy + tail->shift;
  }
  return sum;
}

double mdir::driver::builder::getCustomPairTailDerivative(const TermInputs &context, unsigned index, StringRef name,
                                      double value, bool shift,
                                      bool fixed) {
  const auto &tailExpressions = context.getPairPlan().tailExpressions;
  if (!llvm::is_contained(tailExpressions[index].getNames(), name))
    return 0.0;
  // Central differences extrapolated to h → 0 (Richardson); where a side
  // cannot be evaluated (a λ at 0 of an expression in √λ), the one-sided
  // differences of second order.
  auto at = [&](double x) {
    llvm::StringMap<double> changes;
    changes[name] = x;
    auto tail = getCustomPairTail(context, index, changes);
    if (!tail) {
      llvm::consumeError(tail.takeError());
      return std::nan("");
    }
    if (fixed)
      return tail->shiftFixed;
    return tail->energy + (shift ? tail->shift : 0.0);
  };
  double h = 1.0e-3 * std::max(std::fabs(value), 1.0);
  auto central = [&](double step) {
    return (at(value + step) - at(value - step)) / (2.0 * step);
  };
  double d = (4.0 * central(0.5 * h) - central(h)) / 3.0;
  if (std::isfinite(d))
    return d;
  for (double sign : {1.0, -1.0}) {
    double step = sign * 1.0e-4 * std::max(std::fabs(value), 1.0);
    double one = (-3.0 * at(value) + 4.0 * at(value + step) -
                  at(value + 2.0 * step)) / (2.0 * step);
    if (std::isfinite(one))
      return one;
  }
  return std::nan("");
}
