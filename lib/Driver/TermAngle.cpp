// Built-in bonded family: tuple data, potential, and logged columns (#683).
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
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
    unsigned terms = selection.terms;
    auto has = [&](StringRef name) {
      return llvm::any_of(context.getTupleSets(),
                          [&](const auto &set) { return set.name == name; });
    };
    if ((terms & TermInputs::Angles) && has("angles")) {
      context.emitIRFragment(
          "  ", {"%u_angles"}, {"f64"},
          {"%r_angles", "%x", "%cell", "%f_angles_k", "%f_angles_theta0"},
          {"!rel_angles", "!vec", "!md.cell", "!of_angles", "!of_angles"},
          [](mlir::OpBuilder &builder,
             mlir::ValueRange values) -> TermIRValues {
            using namespace mlir;
            auto loc = builder.getUnknownLoc();
            llvm::SmallVector<mdir::md::Coordinate> coordinates = {
                {mdir::md::CoordinateKind::Angle, {0, 1, 2}}};
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
            return {sum.getResult()};
          });
      add("angles");
    }
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder> mdir::driver::builder::createAngleTermBuilder() {
  return std::make_unique<AngleTermBuilder>();
}
