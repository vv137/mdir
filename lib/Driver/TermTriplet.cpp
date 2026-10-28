// Custom triplet terms and particle-place parameters (D359, D361, #683).
#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace mdir::driver::builder {
class TripletTermBuilder final : public TermBuilder {
public:
  int potentialRank() const override { return 1; }
  llvm::Error collectBareParameters(TermParameters &context) const override;
  void emitBareContributions(TermEmission &context,
                             std::string &total) const override;

private:
  static std::string getTripletParameter(const Control &, const TripletTerm &,
                                         StringRef);
  static std::string emitTripletTerm(TermEmission &, size_t,
                                     const TripletTerm &);
};
} // namespace mdir::driver::builder

std::unique_ptr<TermBuilder> mdir::driver::builder::createTripletTermBuilder() {
  return std::make_unique<TripletTermBuilder>();
}

llvm::Error
TripletTermBuilder::collectBareParameters(TermParameters &context) const {
  const Control &control = context.getControl();
  const System &system = context.getSystem();
  auto &parameters = context.getMutableTypeParameters();
  auto &fields = context.getFields();
  // The parameters of the types that the terms over triplets take for a
  // particle by its place, `sigma1` for the center (D160): a field where
  // they differ between the types, a number where they do not.
  for (const TripletTerm &term : control.triplets) {
    Expression expression =
        llvm::cantFail(Expression::parse(term.expression, control.functions));
    for (const std::string &name : expression.getNames()) {
      std::string stem = getTripletParameter(control, term, name);
      if (stem.empty() || llvm::any_of(parameters, [&](Parameter &known) {
            return known.name == stem;
          }))
        continue;
      std::vector<double> values;
      for (const ParticleType &type : control.types)
        values.push_back(llvm::find_if(type.parameters, [&](auto &entry) {
                           return entry.first == stem;
                         })->second);
      Parameter parameter;
      parameter.name = stem;
      parameter.value = values.front();
      parameter.isUniform = llvm::all_of(
          values, [&](double value) { return value == values.front(); });
      if (!parameter.isUniform) {
        parameter.field = fields.size();
        Program::Field field;
        field.name = stem;
        for (unsigned type : system.types)
          field.values.push_back(values[type]);
        fields.push_back(std::move(field));
      }
      parameters.push_back(std::move(parameter));
    }
  }
  return llvm::Error::success();
}

std::string TripletTermBuilder::getTripletParameter(const Control &control,
                                                    const TripletTerm &term,
                                                    StringRef name) {
  if (name.size() < 2 || !llvm::is_contained("123", name.back()) ||
      llvm::any_of(term.constants,
                   [&](const auto &c) { return c.first == name; }))
    return "";
  StringRef stem = name.drop_back();
  if (control.types.empty() ||
      !llvm::all_of(control.types, [&](const ParticleType &type) {
        return llvm::any_of(type.parameters,
                            [&](const auto &p) { return p.first == stem; });
      }))
    return "";
  return stem.str();
}

void TripletTermBuilder::emitBareContributions(TermEmission &context,
                                               std::string &total) const {
  const Control &control = context.getControl();
  auto &os = context.getOutput();
  for (auto [index, term] : llvm::enumerate(control.triplets)) {
    std::string result = emitTripletTerm(context, index, term);
    std::string sum = "%total_t" + std::to_string(index);
    os << "  " << sum << " = arith.addf " << total << ", " << result
       << " : f64\n";
    total = sum;
  }
}

std::string TripletTermBuilder::emitTripletTerm(TermEmission &context,
                                                size_t index,
                                                const TripletTerm &term) {
  const Control &control = context.getControl();
  auto parameters = context.getTypeParameters();
  auto &os = context.getOutput();
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
    std::string stem = getTripletParameter(control, term, name);
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
    std::string stem = getTripletParameter(control, term, name);
    if (stem.empty())
      continue;
    if (llvm::is_contained(gathered, stem)) {
      values[name] = p + stem + "_" + std::to_string(places[name.back() - '1']);
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
