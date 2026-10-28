// Staged and hosted model terms, with private preparation per build
// (D359, D361, D[builder-term-roles], #683).
#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace mdir::driver::builder {
class ModelTermBuilder final : public TermPreparation,
                               public TermHostEvaluation,
                               public TermBuilder {
  struct State {
    bool edges = false;
    std::vector<double> cutoffs;
    std::vector<std::string> potentials, results;
  };
  std::unique_ptr<State> state;

public:
  ModelTermBuilder() : state(std::make_unique<State>()) {
    roles.preparation = this;
    roles.hostEvaluation = this;
  }
  int parameterRank() const override { return -3; }
  int topologyRank() const override { return 2; }
  llvm::Error validate(const TermInputs &context) const override;
  llvm::Error prepare(const TermInputs &context,
                      llvm::raw_ostream &os) const override;
  bool applies(const TermInputs &context) const override {
    return !context.getControl().models.empty();
  }
  std::string emit(const TermInputs &context, llvm::raw_ostream &os,
                   const TermEvaluation &point) const override;
  void collectParameters(TermParameters &context) const override {
    const Control &control = context.getControl();
    const Topology &topology = context.getTopology();
    size_t count = topology.getNumParticles();
    // The atomic numbers that a learned potential takes
    // (D270), in the order of the particles that the program
    // keeps, as the other fields; a model in stages takes them too
    // (D294).
    if (!control.models.empty() || !control.stagedModels.empty()) {
      Program::Field numbers;
      numbers.name = "model_z";
      numbers.isInteger = true;
      for (size_t i = 0; i != count; ++i)
        numbers.values.push_back(
            i < topology.atomicNumbers.size() ? topology.atomicNumbers[i] : 0);
      context.getFields().push_back(std::move(numbers));
    }
  }
  void
  emitTopologyPotential(TermEmission &context,
                        const PotentialSelection &selection,
                        const TermPotentialAddition &addition) const override {
    if (selection.terms != TermInputs::AllTerms)
      return;
    for (size_t k = 0; k != state->potentials.size(); ++k) {
      context.getOutput() << state->potentials[k];
      addition.append(state->results[k], "%total_staged" + std::to_string(k));
    }
  }

private:
  void emitDeclarations(const TermInputs &, llvm::raw_ostream &) const override;
};
} // namespace mdir::driver::builder

std::unique_ptr<TermBuilder> mdir::driver::builder::createModelTermBuilder() {
  return std::make_unique<ModelTermBuilder>();
}

llvm::Error ModelTermBuilder::validate(const TermInputs &context) const {
  const Control &control = context.getControl();
  const System &system = context.getSystem();
  // A learned potential takes every particle with its element, and ML1 has
  // no constraints and no virtual sites beside it
  // (docs/metatomic-ml1.md, Section 6.9).
  if (!control.stagedModels.empty()) {
    if (!system.topology)
      return makeError("a model in stages of [[energy.model]] needs a "
                       "topology, which gives the elements of the particles");
    if (context.hasSites() || !system.topology->virtualSites.empty())
      return makeError("a model in stages takes no virtual sites yet");
  }
  if (!control.models.empty()) {
    if (!system.topology)
      return makeError("a term of [[energy.model]] needs a topology, which "
                       "gives the elements of the particles");
    if (context.hasConstraints())
      return makeError("a term of [[energy.model]] takes no constraints "
                       "yet: give a flexible model (no SETTLE or SHAKE)");
    if (context.hasSites() || !system.topology->virtualSites.empty())
      return makeError("a term of [[energy.model]] takes no virtual sites "
                       "yet");
    if (control.precision == Precision::Single)
      return makeError("a term of [[energy.model]] takes the mixed or the "
                       "double precision");
    // A frame evaluator is chosen when a Python simulation is made, after
    // the checks of the control: its forces and virial come from
    // evaluations of their own, which do not take the model.
    if (control.frames)
      return makeError("a term of [[energy.model]] is not evaluated by the "
                       "frame evaluator yet");
    if (control.tunables || control.tunableGradient)
      return makeError("a term of [[energy.model]] has no derivative in a "
                       "tunable parameter: tunables are refused");
    // The pair list is a value of the program, at the model's cutoff
    // (D277).
    llvm::Expected<int> mode = getPairsMode();
    if (!mode)
      return mode.takeError();
    state->edges = *mode != MDIR_MLIP_PAIRS_SEARCH;
    if (state->edges)
      for (const Control::ModelTerm &term : control.models) {
        llvm::Expected<double> cutoff = getModelCutoff(term);
        if (!cutoff)
          return cutoff.takeError();
        state->cutoffs.push_back(*cutoff);
      }
  }
  return llvm::Error::success();
}

llvm::Error ModelTermBuilder::prepare(const TermInputs &context,
                                      llvm::raw_ostream &os) const {
  const Control &control = context.getControl();
  state->potentials.clear();
  state->results.clear();
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
        parameters.push_back(part.trim().take_until(
            [](char c) { return c == ':' || c == ' '; }));
    if (open == StringRef::npos || close == StringRef::npos ||
        parameters.size() != 3)
      return makeError("the potential of the model in stages '" + term.name +
                       "' must take the positions, the cell, and the atomic "
                       "numbers");
    std::string prefix = "%sm" + std::to_string(state->potentials.size()) + "_";
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
    state->potentials.push_back(body);
    state->results.push_back(result);
  }
  return llvm::Error::success();
}

std::string ModelTermBuilder::emit(const TermInputs &context,
                                   llvm::raw_ostream &os,
                                   const TermEvaluation &point) const {
  const Control &control = context.getControl();
  StringRef indent = point.indent, x = point.x, cell = point.cell,
            prefix = point.prefix, f = point.f, fResult = point.fResult,
            u = point.u, uResult = point.uResult, w = point.w, tag = point.tag;

  std::string t = ("_m" + tag).str();
  bool numbers = !u.empty() || !w.empty();
  // The cell, as the diagonal and the tilts.
  os << indent << "%mcell" << t << " = md_exec.cell_numbers " << cell
     << " : vector<6xf64>\n";
  for (int k = 0; k != 6; ++k)
    os << indent << "%mcell" << t << "_" << k << " = vector.extract %mcell" << t
       << "[" << k << "] : f64 from vector<6xf64>\n";
  int bits = context.getStateElement() == Element::F64 ? 64 : 32;
  int forceBits = context.getForceElement() == Element::F64 ? 64 : 32;
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
    if (state->edges) {
      std::string cutoff = formatReal(state->cutoffs[index]);
      os << indent << "%mn" << k << " = md.neighborhood " << x << ", " << cell
         << " cutoff(" << cutoff << ") : !vec -> !pairs\n"
         << indent << "%me" << k << " = md.edges %mn" << k << ", " << x << ", "
         << cell << " cutoff(" << cutoff
         << ") : !pairs, !vec -> !md.edges<@atoms>\n";
    }
    os << indent << "%mi" << k << " = arith.constant " << index << " : i64\n"
       << indent << next;
    if (numbers)
      os << ", %mu" << k << ", %mw" << k;
    os << " = mdrt.host_call @mdrtModelEvaluate"
       << (state->edges ? "Edges" : "") << (numbers ? "Energy" : "") << "(%mi"
       << k << ", " << x << ", " << prefix << "model_z";
    if (state->edges)
      os << ", %me" << k;
    for (int c = 0; c != 6; ++c)
      os << ", %mcell" << t << "_" << c;
    os << ", %mbits" << t << ", %mfbits" << t << ")\n"
       << indent << "    outs(" << forces << " : !vec) {in_place}\n"
       << indent << "    : (i64, !vec, !ids, "
       << (state->edges ? "!md.edges<@atoms>, " : "")
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

void ModelTermBuilder::emitDeclarations(const TermInputs &context,
                                        llvm::raw_ostream &os) const {
  const Control &control = context.getControl();
  StringRef stateType = getName(context.getStateElement());
  StringRef force = getName(context.getForceElement());
  // A learned potential (D270): the host evaluates it at the
  // positions where they are, in the cell of the step, and adds its forces
  // to those of the program where they are; with `Energy`, it gives its
  // energy and its virial as well.
  if (!control.models.empty()) {
    // With the pair list of the program, its slots, counts, offsets, and
    // the number of their build (D277).
    std::string edges =
        state->edges ? "memref<?x5xi32> {mdrt.read_only}, memref<?xi32> "
                       "{mdrt.read_only}, memref<?xi32> {mdrt.read_only}, i64, "
                     : "";
    std::string arguments =
        (llvm::Twine("i64, memref<?x3x") + stateType +
         "> {mdrt.read_only}, memref<?xi32> {mdrt.read_only}, " + edges +
         "f64, f64, f64, f64, f64, f64, i64, i64, memref<?x3x" + force + ">")
            .str();
    std::string name =
        state->edges ? "@mdrtModelEvaluateEdges" : "@mdrtModelEvaluate";
    os << "func.func private " << name << "(" << arguments << ")\n"
       << "    attributes {llvm.emit_c_interface}\n"
       << "func.func private " << name << "Energy(" << arguments
       << ", memref<1xf64>, memref<9xf64>)\n"
       << "    attributes {llvm.emit_c_interface}\n";
  }
}
