// Built-in bonded family: tuple data, potential, and logged columns (#683).
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
using namespace mdir::driver;
using namespace mdir::driver::builder;

static mlir::Value buildDihedralsContribution(mlir::OpBuilder &builder, mlir::ValueRange values) {
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
              llvm::SmallVector<Type> types(4, builder.getF64Type());
              llvm::SmallVector<Location> locations(4, loc);
              Block *block =
                  builder.createBlock(&sum.getKernel(), {}, types, locations);
              auto args = block->getArguments();
              Value one = arith::ConstantOp::create(
                  builder, loc, builder.getF64FloatAttr(1.0));
              Value nphi =
                  arith::MulFOp::create(builder, loc, args[2], args[0]);
              Value angle = arith::SubFOp::create(builder, loc, nphi, args[3]);
              Value cosine = math::CosOp::create(builder, loc, angle);
              Value sum = arith::AddFOp::create(builder, loc, one, cosine);
              Value energy = arith::MulFOp::create(builder, loc, args[1], sum);
              mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
            }
            return sum.getResult();
}

namespace mdir::driver::builder {
class DihedralTermBuilder final : public TermBuilder {
public:
  DihedralTermBuilder() {
    potentialMask = TermInputs::Dihedrals;
    nativePotentialMask = potentialMask;
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
  llvm::Error buildTopologyPotential(const TermInputs &context, mlir::OpBuilder &builder,
      const TermNativeTopology &state, const PotentialSelection &selection,
      llvm::function_ref<void(mlir::Value)> add) const override {
    if (!(selection.terms & TermInputs::Dihedrals)) return llvm::Error::success();
    auto relation = state.relation("dihedrals");
    if (!relation) return llvm::Error::success();
    add(buildDihedralsContribution(builder, mlir::ValueRange{relation, state.physical.positions, state.physical.cell, state.tupleField("dihedrals", "k"), state.tupleField("dihedrals", "n"), state.tupleField("dihedrals", "phase")}));

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
    if ((terms & TermInputs::Dihedrals) && has("dihedrals")) {
      context.emitIRFragment(
          "  ", {"%u_dihedrals"}, {"f64"},
          {"%r_dihedrals", "%x", "%cell", "%f_dihedrals_k", "%f_dihedrals_n",
           "%f_dihedrals_phase"},
          {"!rel_dihedrals", "!vec", "!md.cell", "!of_dihedrals",
           "!of_dihedrals", "!of_dihedrals"},
          [](mlir::OpBuilder &builder,
             mlir::ValueRange values) -> TermIRValues {
            return {buildDihedralsContribution(builder, values)};
          });
      add("dihedrals");
    }
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder>
mdir::driver::builder::createDihedralTermBuilder() {
  return std::make_unique<DihedralTermBuilder>();
}
