// Built-in bonded family: tuple data, potential, and logged columns (#683).
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
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

static mlir::Value buildPairs14Contribution(mlir::OpBuilder &builder, mlir::ValueRange values, bool lj14, bool coulomb14, bool products, double cutoff, double from, Truncation modifier) {
  using namespace mlir;
  auto loc = builder.getUnknownLoc();
  llvm::SmallVector<mdir::md::Coordinate> coordinates = {
      {mdir::md::CoordinateKind::Distance, {0, 1}}};
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  mdir::md::getCoordinateAttrs(builder, coordinates, kinds, members);
  size_t gathered = products ? 1 : 0;
  auto sum = mdir::md::SumTuplesOp::create(
      builder, loc, builder.getF64Type(), values[0], values[1],
      values[2], values.slice(3, gathered),
      values.drop_front(3 + gathered), kinds, members);
  {
    OpBuilder::InsertionGuard guard(builder);
    unsigned count = products ? 7 : 4;
    llvm::SmallVector<Type> types(count, builder.getF64Type());
    llvm::SmallVector<Location> locations(count, loc);
    Block *block =
        builder.createBlock(&sum.getKernel(), {}, types, locations);
    auto args = block->getArguments();
    unsigned tuple = products ? 3 : 1;
    Value qq = args[tuple + 2];
    if (products) {
      Value charges =
          arith::MulFOp::create(builder, loc, args[1], args[2]);
      qq = arith::MulFOp::create(builder, loc, args[tuple + 3],
                                 charges);
    }
    Value energy;
    if (lj14)
      energy = buildLennardJonesIR(
          builder,
          ValueRange{args[tuple], args[tuple + 1], args[0], args[0]},
          cutoff, from, modifier);
    if (coulomb14) {
      Value coulomb =
          arith::DivFOp::create(builder, loc, qq, args[0]);
      if (energy)
        energy = arith::AddFOp::create(builder, loc, energy, coulomb);
      else
        energy = coulomb;
    }
    mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
  }
  return sum.getResult();
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
  llvm::Error buildTopologyPotential(const TermInputs &context, mlir::OpBuilder &builder,
      const TermNativeTopology &state, const PotentialSelection &selection,
      llvm::function_ref<void(mlir::Value)> add) const override {
    bool lj14 = selection.terms & TermInputs::LennardJones14;
    bool coulomb14 = selection.terms & TermInputs::Coulomb14;
    if (!lj14 && !coulomb14) return llvm::Error::success();
    auto relation = state.relation("pairs14");
    if (!relation) return llvm::Error::success();
    bool products = context.hasGradientArguments() && context.hasGradientCharges() && coulomb14;
    const auto &control = context.getControl();
    Truncation modifier =
        (control.truncation == Truncation::PowerForceSwitch ||
         control.truncation == Truncation::SquaredDistanceSwitch)
            ? control.truncation : Truncation::None;
    TermIRValues values{relation, state.physical.positions, state.physical.cell};
    if (products) values.push_back(state.physical.field("q"));
    for (StringRef field : {"sigma", "epsilon", "qq"})
      values.push_back(state.tupleField("pairs14", field));
    if (products) values.push_back(state.tupleField("pairs14", "tgs"));
    add(buildPairs14Contribution(builder, values, lj14, coulomb14, products,
        control.cutoffDistance * units::length,
        control.switchDistance * units::length, modifier));

    return llvm::Error::success();
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder> mdir::driver::builder::createPairs14TermBuilder() {
  return std::make_unique<Pairs14TermBuilder>();
}
