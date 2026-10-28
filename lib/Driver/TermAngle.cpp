// Built-in bonded family: tuple data, potential, and logged columns (#683).
#include "BuilderImpl.h"
using namespace mdir::driver;
using namespace mdir::driver::builder;

namespace mdir::driver::builder {
class AngleTermBuilder final : public TermBuilder {
public:
  AngleTermBuilder() {
    potentialMask = TermInputs::Angles;
    tupleRank = -1;
  }
  int topologyRank() const override { return 2; }
  void collectTupleSets(TermParameters &context) const override {
    const Topology &topology = context.getTopology();
    if (!topology.angles.empty()) {
      Program::TupleSet &set = addTupleSet(context.getTupleSets(), "angles", 3);
      size_t k = addTupleField(set, "k"), t0 = addTupleField(set, "theta0");
      for (const Topology::Angle &angle : topology.angles) {
        for (unsigned member : {angle.i, angle.j, angle.k})
          set.members.push_back(member);
        set.fields[k].values.push_back(angle.force);
        set.fields[t0].values.push_back(angle.theta0);
      }
    }
  }
  void appendTermDefinitions(
      const TermInputs &,
      std::vector<TermDefinition> &definitions) const override {
    definitions.push_back({3, 0, "term_angles", TermInputs::Angles});
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
    if ((terms & TermInputs::Angles) && has("angles")) {
      os << "  %u_angles = md.sum_tuples %r_angles, %x, %cell coordinates("
            "angle(0, 1, 2))\n"
         << "      tuple(%f_angles_k, %f_angles_theta0 : !of_angles, "
            "!of_angles) {\n"
         << "  ^bb0(%theta: f64, %k: f64, %theta0: f64):\n"
         << "    %half = arith.constant 0.5 : f64\n"
         << "    %dt = arith.subf %theta, %theta0 : f64\n"
         << "    %dt2 = arith.mulf %dt, %dt : f64\n"
         << "    %hk = arith.mulf %half, %k : f64\n"
         << "    %e = arith.mulf %hk, %dt2 : f64\n"
         << "    md.yield %e : f64\n"
         << "  } : !rel_angles, !vec -> f64\n";
      add("angles");
    }
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder> mdir::driver::builder::createAngleTermBuilder() {
  return std::make_unique<AngleTermBuilder>();
}
