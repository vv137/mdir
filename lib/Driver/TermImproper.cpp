// Built-in bonded family: tuple data, potential, and logged columns (#683).
#include "BuilderImpl.h"
using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace mdir::driver::builder {
class ImproperTermBuilder final : public TermBuilder {
public:
  ImproperTermBuilder() {
    potentialMask = TermInputs::HarmonicImpropers;
    tupleRank = -1;
  }
  int topologyRank() const override { return 2; }
  void collectTupleSets(TermParameters &context) const override {
    const Topology &topology = context.getTopology();
    if (!topology.harmonicImpropers.empty()) {
      Program::TupleSet &set =
          addTupleSet(context.getTupleSets(), "impropers", 4);
      size_t k = addTupleField(set, "k"), xi0 = addTupleField(set, "xi0");
      for (const Topology::HarmonicImproper &term :
           topology.harmonicImpropers) {
        for (unsigned member : {term.i, term.j, term.k, term.l})
          set.members.push_back(member);
        set.fields[k].values.push_back(term.force);
        set.fields[xi0].values.push_back(term.xi0);
      }
    }
  }
  void appendTermDefinitions(
      const TermInputs &,
      std::vector<TermDefinition> &definitions) const override {
    definitions.push_back(
        {11, 0, "term_impropers", TermInputs::HarmonicImpropers});
  }
  void emitTopologyPotential(TermEmission &context,
                             const PotentialSelection &selection,
                             const TermPotentialAddition &add) const override {
    auto &os = context.getOutput();
    unsigned terms = selection.terms;
    auto has = [&](StringRef name) {
      return llvm::any_of(context.getTupleSets(),
                          [&](const auto &set) { return set.name == name; });
    };
    if ((terms & TermInputs::HarmonicImpropers) && has("impropers")) {
      // ½ k (ξ − ξ0)², with ξ − ξ0 brought into [−π, π).
      os << "  %u_impropers = md.sum_tuples %r_impropers, %x, %cell "
            "coordinates(dihedral(0, 1, 2, 3))\n"
         << "      tuple(%f_impropers_k, %f_impropers_xi0 : !of_impropers, "
            "!of_impropers) {\n"
         << "  ^bb0(%xi: f64, %k: f64, %xi0: f64):\n"
         << "    %half = arith.constant 0.5 : f64\n"
         << "    %pi = arith.constant " << formatReal(M_PI) << " : f64\n"
         << "    %turn = arith.constant " << formatReal(2.0 * M_PI)
         << " : f64\n"
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
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder>
mdir::driver::builder::createImproperTermBuilder() {
  return std::make_unique<ImproperTermBuilder>();
}
