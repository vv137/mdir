// Built-in bonded family: tuple data, potential, and logged columns (#683).
#include "BuilderImpl.h"
using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace mdir::driver::builder {
class UreyBradleyTermBuilder final : public TermBuilder {
public:
  UreyBradleyTermBuilder() { tupleRank = -1; }
  int topologyRank() const override { return 3; }
  void collectTupleSets(TermParameters &context) const override {
    const Topology &topology = context.getTopology();
    if (!topology.ureyBradleys.empty()) {
      Program::TupleSet &set =
          addTupleSet(context.getTupleSets(), "urey_bradley", 2);
      size_t k = addTupleField(set, "k"), r0 = addTupleField(set, "r0");
      for (const Topology::UreyBradley &term : topology.ureyBradleys) {
        set.members.push_back(term.i);
        set.members.push_back(term.k);
        set.fields[k].values.push_back(term.force);
        set.fields[r0].values.push_back(term.r0);
      }
    }
  }
  void appendTermDefinitions(
      const TermInputs &,
      std::vector<TermDefinition> &definitions) const override {
    definitions.push_back(
        {10, 0, "term_urey_bradley", TermInputs::UreyBradleys});
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
    if ((terms & TermInputs::UreyBradleys) && has("urey_bradley")) {
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
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder>
mdir::driver::builder::createUreyBradleyTermBuilder() {
  return std::make_unique<UreyBradleyTermBuilder>();
}
