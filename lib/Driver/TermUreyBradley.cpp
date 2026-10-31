// Built-in bonded family: tuple data, potential, and logged columns (#683).
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
using namespace mdir::driver;
using namespace mdir::driver::builder;

static mlir::Value buildUreyBradleysContribution(mlir::OpBuilder &builder, mlir::ValueRange values) {
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
class UreyBradleyTermBuilder final : public TermBuilder {
public:
  UreyBradleyTermBuilder() {
    potentialMask = TermInputs::UreyBradleys;
    nativePotentialMask = potentialMask;
    tupleRank = -1;
  }
  int topologyRank() const override { return 2; }
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
  llvm::Error buildTopologyPotential(const TermInputs &context, mlir::OpBuilder &builder,
      const TermNativeTopology &state, const PotentialSelection &selection,
      llvm::function_ref<void(mlir::Value)> add) const override {
    if (!(selection.terms & TermInputs::UreyBradleys)) return llvm::Error::success();
    auto relation = state.relation("urey_bradley");
    if (!relation) return llvm::Error::success();
    add(buildUreyBradleysContribution(builder, mlir::ValueRange{relation, state.physical.positions, state.physical.cell, state.tupleField("urey_bradley", "k"), state.tupleField("urey_bradley", "r0")}));

    return llvm::Error::success();
  }
  void emitTopologyPotential(TermEmission &context,
                             const PotentialSelection &selection,
                             const TermPotentialAddition &add) const override {
    unsigned terms = selection.terms;
    auto has = [&](StringRef name) {
      return llvm::any_of(context.getTupleSets(),
                          [&](const auto &set) { return set.name == name; });
    };
    if ((terms & TermInputs::UreyBradleys) && has("urey_bradley")) {
      context.emitIRFragment(
          "  ", {"%u_urey_bradley"}, {"f64"},
          {"%r_urey_bradley", "%x", "%cell", "%f_urey_bradley_k",
           "%f_urey_bradley_r0"},
          {"!rel_urey_bradley", "!vec", "!md.cell", "!of_urey_bradley",
           "!of_urey_bradley"},
          [](mlir::OpBuilder &builder,
             mlir::ValueRange values) -> TermIRValues {
            return {buildUreyBradleysContribution(builder, values)};
          });
      add("urey_bradley");
    }
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder>
mdir::driver::builder::createUreyBradleyTermBuilder() {
  return std::make_unique<UreyBradleyTermBuilder>();
}
