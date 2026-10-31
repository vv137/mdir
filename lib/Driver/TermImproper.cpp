// Built-in bonded family: tuple data, potential, and logged columns (#683).
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
using namespace mdir::driver;
using namespace mdir::driver::builder;

static mlir::Value buildHarmonicImpropersContribution(mlir::OpBuilder &builder, mlir::ValueRange values) {
            using namespace mlir;
            auto loc = builder.getUnknownLoc();
            llvm::SmallVector<mdir::md::Coordinate> coordinates = {
                {mdir::md::CoordinateKind::Dihedral, {0, 1, 2, 3}}};
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
              Value pi = arith::ConstantOp::create(
                  builder, loc, builder.getF64FloatAttr(M_PI));
              Value turn = arith::ConstantOp::create(
                  builder, loc, builder.getF64FloatAttr(2.0 * M_PI));
              Value perTurn = arith::ConstantOp::create(
                  builder, loc, builder.getF64FloatAttr(0.5 / M_PI));
              Value delta =
                  arith::SubFOp::create(builder, loc, args[0], args[2]);
              Value shifted = arith::AddFOp::create(builder, loc, delta, pi);
              Value turns =
                  arith::MulFOp::create(builder, loc, shifted, perTurn);
              Value whole = math::FloorOp::create(builder, loc, turns);
              Value back = arith::MulFOp::create(builder, loc, whole, turn);
              Value wrapped = arith::SubFOp::create(builder, loc, delta, back);
              Value square =
                  arith::MulFOp::create(builder, loc, wrapped, wrapped);
              Value hk = arith::MulFOp::create(builder, loc, half, args[1]);
              Value energy = arith::MulFOp::create(builder, loc, hk, square);
              mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
            }
            return sum.getResult();
}

namespace mdir::driver::builder {
class ImproperTermBuilder final : public TermBuilder {
public:
  ImproperTermBuilder() {
    potentialMask = TermInputs::HarmonicImpropers;
    nativePotentialMask = potentialMask;
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
  llvm::Error buildTopologyPotential(const TermInputs &context, mlir::OpBuilder &builder,
      const TermNativeTopology &state, const PotentialSelection &selection,
      llvm::function_ref<void(mlir::Value)> add) const override {
    if (!(selection.terms & TermInputs::HarmonicImpropers)) return llvm::Error::success();
    auto relation = state.relation("impropers");
    if (!relation) return llvm::Error::success();
    add(buildHarmonicImpropersContribution(builder, mlir::ValueRange{relation, state.physical.positions, state.physical.cell, state.tupleField("impropers", "k"), state.tupleField("impropers", "xi0")}));

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
    if ((terms & TermInputs::HarmonicImpropers) && has("impropers")) {
      // ½ k (ξ − ξ0)², with ξ − ξ0 brought into [−π, π).
      context.emitIRFragment(
          "  ", {"%u_impropers"}, {"f64"},
          {"%r_impropers", "%x", "%cell", "%f_impropers_k", "%f_impropers_xi0"},
          {"!rel_impropers", "!vec", "!md.cell", "!of_impropers",
           "!of_impropers"},
          [](mlir::OpBuilder &builder,
             mlir::ValueRange values) -> TermIRValues {
            return {buildHarmonicImpropersContribution(builder, values)};
          });
      add("impropers");
    }
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder>
mdir::driver::builder::createImproperTermBuilder() {
  return std::make_unique<ImproperTermBuilder>();
}
