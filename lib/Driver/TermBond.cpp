// Harmonic bond terms: tuple data, potential, and logged energy (#683).
#include "BuilderImpl.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace mdir::driver::builder {
class BondTermBuilder final : public TermBuilder {
public:
  BondTermBuilder() {
    potentialMask = TermInputs::Bonds;
    tupleRank = -1;
  }
  int topologyRank() const override { return 2; }
  void collectTupleSets(TermParameters &context) const override {
    const Topology &topology = context.getTopology();
    if (!topology.bonds.empty()) {
      Program::TupleSet &set = addTupleSet(context.getTupleSets(), "bonds", 2);
      size_t k = addTupleField(set, "k"), r0 = addTupleField(set, "r0");
      for (const Topology::Bond &bond : topology.bonds) {
        set.members.push_back(bond.i);
        set.members.push_back(bond.j);
        set.fields[k].values.push_back(bond.k);
        set.fields[r0].values.push_back(bond.r0);
      }
    }
  }
  void appendTermDefinitions(
      const TermInputs &,
      std::vector<TermDefinition> &definitions) const override {
    definitions.push_back({2, 0, "term_bonds", TermInputs::Bonds});
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
    if ((terms & TermInputs::Bonds) && has("bonds")) {
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
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder> mdir::driver::builder::createBondTermBuilder() {
  return std::make_unique<BondTermBuilder>();
}
