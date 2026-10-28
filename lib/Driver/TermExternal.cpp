// The external terms: parameter fields, potentials, and energy output
// (D359). Entry buffers follow the shared field list.

#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace mdir {
namespace driver {
namespace builder {

class ExternalTermBuilder final : public TermBuilder {
public:
  void collectParameters(TermParameters &context) const override {
    const Topology &topology = context.getTopology();
    size_t count = topology.getNumParticles();
    auto &fields = context.getFields();
    // The terms of the absolute positions (D148): a flag for each particle,
    // 1 for those of the term, and each parameter given as a list.
    for (auto [k, term] : llvm::enumerate(topology.externalTerms)) {
      std::string flag = context.indexedName("ext", k);
      Program::Field on;
      on.name = flag;
      on.values.assign(count, 0.0);
      for (unsigned particle : term.particles)
        on.values.edit()[particle] = 1.0;
      fields.push_back(std::move(on));
      for (const auto &[name, values] : term.parameters) {
        Program::Field field;
        field.name = flag + "_" + name;
        field.values.assign(count, 0.0);
        for (auto [place, particle] : llvm::enumerate(term.particles))
          field.values.edit()[particle] = values[place];
        fields.push_back(std::move(field));
      }
    }
  }

  void emitTopologyPotential(
      TermEmission &context, const PotentialSelection &selection,
      const TermPotentialAddition & add) const override {
    if (selection.terms & TermInputs::ExternalTerms)
      for (auto [index, term] :
           llvm::enumerate(context.getTopology().externalTerms))
        if ((selection.externalTerm < 0 ||
             static_cast<int>(index) == selection.externalTerm) &&
            !(selection.alchemicalOnly && !context.usesLambda(term.expression)))
          emitExternalTerm(context, index, term, add);
  }

  void emitTermPotentials(TermEmission &context) const override {
    for (size_t k = 0, e = context.getTopology().externalTerms.size();
         k != e; ++k)
      context.emitTopologyPotential("term_external" + std::to_string(k),
                                     TermInputs::ExternalTerms, -1, -1,
                                     static_cast<int>(k));
  }

  void appendTermNames(const TermInputs &context,
                       std::vector<std::string> &names) const override {
    for (size_t k = 0, e = context.getTopology().externalTerms.size();
         k != e; ++k)
      names.push_back("term_external" + std::to_string(k));
  }

  std::pair<unsigned, int> findObservedTerm(const TermInputs &context,
                                           StringRef name) const override {
    for (auto [index, term] :
         llvm::enumerate(context.getTopology().externalTerms))
      if (term.name == name)
        return {TermInputs::ExternalTerms, static_cast<int>(index)};
    return {0, -1};
  }

  std::optional<double> getObservedValue(const TermInputs &context, unsigned kind,
                                        int index, StringRef name) const override {
    if (kind == TermInputs::ExternalTerms)
      for (const auto &[key, value] :
           context.getTopology().externalTerms[index].constants)
        if (key == name)
          return value;
    return std::nullopt;
  }

private:
  static void emitExternalTerm(TermEmission &context, size_t index,
                              const ExternalTerm &term,
                              const TermPotentialAddition & add);
};

} // namespace builder
} // namespace driver
} // namespace mdir

std::unique_ptr<TermBuilder> mdir::driver::builder::createExternalTermBuilder() {
  return std::make_unique<ExternalTermBuilder>();
}

void ExternalTermBuilder::emitExternalTerm(
    TermEmission &context, size_t index, const ExternalTerm &term,
    const TermPotentialAddition & add) {
  const Control &control = context.getControl();
  const System &system = context.getSystem();
  llvm::raw_ostream &os = context.getOutput();
  // Σ_i k(x_i) over the particles of the term [Eastman2017] (D148),
  // the flag `ext<k>` 1 for
  // them and 0 for the others, which the kernel selects on: the positions
  // in Å, the energy in kcal/mol. The positions are never wrapped (D74), so
  // that the term is continuous along a trajectory.
  Expression expression = llvm::cantFail(
      Expression::parse(term.expression, control.functions));
  std::vector<std::string> names = expression.getNames();
  bool charges = llvm::is_contained(names, "q");
  std::string flag = context.indexedName("ext", index);
  std::string prefix = context.indexedName("%xe", index) + "_";
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
  context.bindLambdas(values);
  context.bindObserved(term.name, values);
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

