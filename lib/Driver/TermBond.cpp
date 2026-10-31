// Harmonic bond terms: tuple data, potential, and logged energy (#683).
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;

static mlir::Value buildBondsContribution(mlir::OpBuilder &builder, mlir::ValueRange values) {
            using namespace mlir;
            auto loc = builder.getUnknownLoc();
            llvm::SmallVector<mdir::md::Coordinate> coordinates = {
                {mdir::md::CoordinateKind::Distance, {0, 1}}};
            DenseI32ArrayAttr kinds;
            DenseI64ArrayAttr members;
            mdir::md::getCoordinateAttrs(builder, coordinates, kinds, members);
            auto sum = mdir::md::SumTuplesOp::create(
                builder, loc, builder.getF64Type(), values[0], values[1],
                values[2], ValueRange{}, values.drop_front(3), kinds, members);
            {
              OpBuilder::InsertionGuard guard(builder);
              llvm::SmallVector<Type> types(3, builder.getF64Type());
              llvm::SmallVector<Location> locations(3, loc);
              Block *block =
                  builder.createBlock(&sum.getKernel(), {}, types, locations);
              auto args = block->getArguments();
              Value half = arith::ConstantOp::create(
                  builder, loc, builder.getF64FloatAttr(0.5));
              Value delta =
                  arith::SubFOp::create(builder, loc, args[0], args[2]);
              Value square = arith::MulFOp::create(builder, loc, delta, delta);
              Value hk = arith::MulFOp::create(builder, loc, half, args[1]);
              Value energy = arith::MulFOp::create(builder, loc, hk, square);
              mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
            }
            return sum.getResult();
}

namespace mdir::driver::builder {
class BondTermBuilder final : public TermBuilder {
public:
  BondTermBuilder() {
    potentialMask = TermInputs::Bonds;
    nativePotentialMask = potentialMask;
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
  void buildTopologyPotential(const TermInputs &, mlir::OpBuilder &builder,
      const TermNativeTopology &state, const PotentialSelection &selection,
      llvm::function_ref<void(mlir::Value)> add) const override {
    if (!(selection.terms & TermInputs::Bonds)) return;
    auto relation = state.relation("bonds");
    if (!relation) return;
    add(buildBondsContribution(builder,
        mlir::ValueRange{relation, state.physical.positions, state.physical.cell,
                        state.tupleField("bonds", "k"), state.tupleField("bonds", "r0")}));
  }
  void emitTopologyPotential(TermEmission &context,
                             const PotentialSelection &selection,
                             const TermPotentialAddition &add) const override {
    unsigned terms = selection.terms;
    auto has = [&](StringRef name) {
      return llvm::any_of(context.getTupleSets(),
                          [&](const auto &set) { return set.name == name; });
    };
    if ((terms & TermInputs::Bonds) && has("bonds")) {
      context.emitIRFragment(
          "  ", {"%u_bonds"}, {"f64"},
          {"%r_bonds", "%x", "%cell", "%f_bonds_k", "%f_bonds_r0"},
          {"!rel_bonds", "!vec", "!md.cell", "!of_bonds", "!of_bonds"},
          [](mlir::OpBuilder &builder,
             mlir::ValueRange values) -> TermIRValues {
            return {buildBondsContribution(builder, values)};
          });
      add("bonds");
    }
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder> mdir::driver::builder::createBondTermBuilder() {
  return std::make_unique<BondTermBuilder>();
}
