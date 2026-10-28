// Built-in bonded family: tuple data, potential, and logged columns (#683).
#include "BuilderImpl.h"
using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace mdir::driver::builder {
class DihedralTermBuilder final : public TermBuilder {
public:
  DihedralTermBuilder() {
    potentialMask = TermInputs::Dihedrals;
    tupleRank = -1;
  }
  int topologyRank() const override { return 2; }
  void collectTupleSets(TermParameters &context) const override {
    const Topology &topology = context.getTopology();
    if (!topology.dihedrals.empty()) {
      Program::TupleSet &set =
          addTupleSet(context.getTupleSets(), "dihedrals", 4);
      size_t k = addTupleField(set, "k"), n = addTupleField(set, "n"),
             phase = addTupleField(set, "phase");
      for (const Topology::Dihedral &dihedral : topology.dihedrals) {
        for (unsigned member : {dihedral.i, dihedral.j, dihedral.k, dihedral.l})
          set.members.push_back(member);
        set.fields[k].values.push_back(dihedral.force);
        set.fields[n].values.push_back(dihedral.n);
        set.fields[phase].values.push_back(dihedral.phase);
      }
    }
  }
  void appendTermDefinitions(
      const TermInputs &,
      std::vector<TermDefinition> &definitions) const override {
    definitions.push_back({4, 0, "term_dihedrals", TermInputs::Dihedrals});
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
    if ((terms & TermInputs::Dihedrals) && has("dihedrals")) {
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
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder>
mdir::driver::builder::createDihedralTermBuilder() {
  return std::make_unique<DihedralTermBuilder>();
}
