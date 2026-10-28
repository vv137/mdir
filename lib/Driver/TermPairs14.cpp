// Built-in bonded family: tuple data, potential, and logged columns (#683).
#include "BuilderImpl.h"
using namespace mdir::driver;
using namespace mdir::driver::builder;

static void fillPairProducts(const TermInputs &context, Program::Field &field) {
  // f q_i q_j s_C of each 1-4 pair.
  const Topology &topology = context.getTopology();
  std::vector<double> products;
  products.reserve(topology.pairs.size());
  for (const Topology::Pair &pair : topology.pairs)
    products.push_back(coulombInternal * topology.charges[pair.i] *
                       topology.charges[pair.j] * pair.scaleCoulomb);
  field.values = std::move(products);
}

namespace mdir::driver::builder {
class Pairs14TermBuilder final : public TermBuilder {
public:
  Pairs14TermBuilder() {
    potentialMask = TermInputs::LennardJones14 | TermInputs::Coulomb14;
    tupleRank = 1;
  }
  unsigned getPotentialMask(const TermInputs &, const DependentTerms *dependent,
                            bool alchemicalOnly) const override {
    if (alchemicalOnly)
      return 0;
    return dependent
               ? (dependent->coulomb ? unsigned(TermInputs::Coulomb14) : 0u)
               : potentialMask;
  }
  int topologyRank() const override { return 4; }
  void collectTupleSets(TermParameters &context) const override {
    const Topology &topology = context.getTopology();
    if (!topology.pairs.empty()) {
      // The factors enter the parameters: ε s_LJ, and f q_i q_j s_C.
      Program::TupleSet &set =
          addTupleSet(context.getTupleSets(), "pairs14", 2);
      size_t sigma = addTupleField(set, "sigma"),
             epsilon = addTupleField(set, "epsilon"),
             qq = addTupleField(set, "qq");
      // In `@tunable` the pairs take f s_C and the charges of their members,
      // so that the derivative in the charges covers them.
      if (context.hasGradientCharges()) {
        size_t scale = addTupleField(set, "tgs");
        for (const Topology::Pair &pair : topology.pairs)
          set.fields[scale].values.push_back(coulombInternal *
                                             pair.scaleCoulomb);
      }
      for (const Topology::Pair &pair : topology.pairs) {
        set.members.push_back(pair.i);
        set.members.push_back(pair.j);
        set.fields[sigma].values.push_back(pair.sigma);
        set.fields[epsilon].values.push_back(pair.epsilon * pair.scaleLJ);
      }
      fillPairProducts(context, set.fields[qq]);
    }
  }
  void refillArrays(TermParameters &context) const override {
    for (Program::TupleSet &set : context.getTupleSets()) {

      if (set.name == "pairs14")
        for (Program::Field &field : set.fields)
          if (field.values.empty() && field.name == "qq")
            fillPairProducts(context, field);
    }
  }
  void appendTermDefinitions(
      const TermInputs &,
      std::vector<TermDefinition> &definitions) const override {
    definitions.push_back({5, 0, "term_lj14", TermInputs::LennardJones14});
    definitions.push_back({6, 0, "term_coulomb14", TermInputs::Coulomb14});
  }
  void emitTopologyPotential(TermEmission &context,
                             const PotentialSelection &selection,
                             const TermPotentialAddition &add) const override {
    auto &os = context.getOutput();
    const auto &control = context.getControl();
    unsigned terms = selection.terms;
    auto has = [&](StringRef name) {
      return llvm::any_of(context.getTupleSets(),
                          [&](const auto &set) { return set.name == name; });
    };
    bool lj14 = terms & TermInputs::LennardJones14,
         coulomb14 = terms & TermInputs::Coulomb14;
    if ((lj14 || coulomb14) && has("pairs14")) {
      // In `@tunable` with the derivative in the charges, the product of
      // the charges is formed here, f s_C q_i q_j, from the charges
      // themselves (D230).
      bool products = context.hasGradientArguments() &&
                      context.hasGradientCharges() && coulomb14;
      os << "  %u_pairs14 = md.sum_tuples %r_pairs14, %x, %cell coordinates("
            "distance(0, 1))\n"
         << (products ? "      gather(%p_q : !real)\n" : "")
         << "      tuple(%f_pairs14_sigma, %f_pairs14_epsilon, %f_pairs14_qq"
         << (products ? ", %f_pairs14_tgs" : "")
         << " : !of_pairs14, !of_pairs14, !of_pairs14"
         << (products ? ", !of_pairs14" : "") << ") {\n"
         << "  ^bb0(%r: f64" << (products ? ", %q14_i: f64, %q14_j: f64" : "")
         << ", %sigma: f64, %epsilon: f64, %qq" << (products ? "_held" : "")
         << ": f64" << (products ? ", %tgs: f64" : "") << "):\n";
      if (products)
        os << "    %q14 = arith.mulf %q14_i, %q14_j : f64\n"
           << "    %qq = arith.mulf %tgs, %q14 : f64\n";
      std::string value;
      if (lj14) {
        // The pairs three bonds apart as they are, except under the power
        // force or squared-distance switch, which also switches these pairs.
        emitLennardJones(
            os, control, "%sigma", "%epsilon", "%lj",
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
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder> mdir::driver::builder::createPairs14TermBuilder() {
  return std::make_unique<Pairs14TermBuilder>();
}
