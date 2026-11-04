// Built-in bonded family: tuple data, potential, and logged columns (#683).
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
using namespace mdir::driver;
using namespace mdir::driver::builder;

static mlir::Value buildDihedralsContribution(
    mlir::OpBuilder &builder, mlir::ValueRange values,
    const BuildKept::TupleGradient *gradient) {
  using namespace mlir;
  auto loc = builder.getUnknownLoc();
  llvm::SmallVector<mdir::md::Coordinate> coordinates = {
      {mdir::md::CoordinateKind::Dihedral, {0, 1, 2, 3}}};
  DenseI32ArrayAttr kinds;
  DenseI64ArrayAttr members;
  mdir::md::getCoordinateAttrs(builder, coordinates, kinds, members);
  unsigned gathered = gradient ? gradient->count : 0;
  auto sum = mdir::md::SumTuplesOp::create(
      builder, loc, builder.getF64Type(), values[0], values[1],
      values[2], values.slice(3, gathered), values.drop_front(3 + gathered),
      kinds, members);
  {
    OpBuilder::InsertionGuard guard(builder);
    unsigned parameter = 1 + 4 * gathered;
    llvm::SmallVector<Type> types(parameter + 3 + (gradient ? 1 : 0),
                                 builder.getF64Type());
    llvm::SmallVector<Location> locations(types.size(), loc);
    Block *block =
        builder.createBlock(&sum.getKernel(), {}, types, locations);
    auto args = block->getArguments();
    Value one = arith::ConstantOp::create(
        builder, loc, builder.getF64FloatAttr(1.0));
    Value force = args[parameter];
    if (gradient)
      force = buildTupleGradientParameter(builder, force, args[parameter + 3],
          args.slice(1, 4 * gathered), 4, *gradient);
    Value nphi =
        arith::MulFOp::create(builder, loc, args[parameter + 1], args[0]);
    Value angle = arith::SubFOp::create(builder, loc, nphi, args[parameter + 2]);
    Value cosine = math::CosOp::create(builder, loc, angle);
    Value sum = arith::AddFOp::create(builder, loc, one, cosine);
    Value energy = arith::MulFOp::create(builder, loc, force, sum);
    mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
  }
  return sum.getResult();
}

namespace mdir::driver::builder {
class DihedralTermBuilder final : public TermBuilder {
public:
  DihedralTermBuilder() {
    potentialMask = TermInputs::Dihedrals;
    tupleRank = -1;
  }
  int topologyRank() const override { return 2; }
  unsigned getPotentialMask(const TermInputs &, const DependentTerms *dependent,
                            bool alchemicalOnly) const override {
    return !alchemicalOnly && (!dependent || dependent->dihedrals)
               ? potentialMask : 0;
  }
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
      if (context.hasTunableGradient())
        for (auto [index, declaration] :
             llvm::enumerate(context.getControl().tunableDeclarations))
          if (declaration.kind == Control::TunableKind::DihedralK)
            collectTupleGradient(context, set, static_cast<unsigned>(index), "k");
    }
  }
  void refillArrays(TermParameters &context) const override {
    for (auto &set : context.getTupleSets())
      if (set.name == "dihedrals")
        for (auto &field : set.fields)
          if (field.name == "k" && field.values.empty())
            for (const auto &row : context.getTopology().dihedrals)
              field.values.push_back(row.force);
  }
  llvm::Expected<std::optional<Program::GradientOutcome>>
  collectTunableOutcome(TermParameters &context, unsigned index) const override {
    if (context.getControl().tunableDeclarations[index].kind !=
        Control::TunableKind::DihedralK)
      return std::optional<Program::GradientOutcome>();
    return std::optional(Program::GradientOutcome::Rule);
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
    const BuildKept::TupleGradient *gradient = nullptr;
    if (context.hasGradientArguments())
      for (const auto &candidate : context.getTupleGradients())
        if (candidate.tupleSet == "dihedrals") {
          assert(!gradient && "one dihedral coefficient declaration");
          gradient = &candidate;
        }
    TermIRValues values{relation, state.physical.positions, state.physical.cell};
    if (gradient)
      for (unsigned c = 0; c != gradient->count; ++c)
        values.push_back(state.gradientSeed(gradient->first + c));
    for (StringRef name : {"k", "n", "phase"})
      values.push_back(state.tupleField("dihedrals", name));
    if (gradient)
      values.push_back(state.tupleField("dihedrals", "tgc_k"));
    add(buildDihedralsContribution(builder, values, gradient));

    return llvm::Error::success();
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder>
mdir::driver::builder::createDihedralTermBuilder() {
  return std::make_unique<DihedralTermBuilder>();
}
