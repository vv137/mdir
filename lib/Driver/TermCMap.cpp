// Built-in bonded family: tuple data, potential, and logged columns (#683).
#include "BuilderImpl.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
using namespace mdir::driver;
using namespace mdir::driver::builder;

static mlir::Value buildCMapsContribution(mlir::OpBuilder &builder, mlir::ValueRange values, unsigned n) {
            using namespace mlir;
            auto loc = builder.getUnknownLoc();
            llvm::SmallVector<mdir::md::Coordinate> coordinates = {
                {mdir::md::CoordinateKind::Dihedral, {0, 1, 2, 3}},
                {mdir::md::CoordinateKind::Dihedral, {1, 2, 3, 4}}};
            DenseI32ArrayAttr kinds;
            DenseI64ArrayAttr members;
            mdir::md::getCoordinateAttrs(builder, coordinates, kinds, members);
            auto sum = mdir::md::SumTuplesOp::create(
                builder, loc, builder.getF64Type(), values[0], values[1],
                values[2], ValueRange{}, values.slice(3, 1), kinds, members);
            {
              OpBuilder::InsertionGuard guard(builder);
              llvm::SmallVector<Type> types(3, builder.getF64Type());
              llvm::SmallVector<Location> locations(3, loc);
              Block *block =
                  builder.createBlock(&sum.getKernel(), {}, types, locations);
              auto args = block->getArguments();
              Value scale = arith::ConstantOp::create(
                  builder, loc, builder.getF64FloatAttr(n / (2.0 * M_PI)));
              Value shift = arith::ConstantOp::create(
                  builder, loc, builder.getF64FloatAttr(n / 2.0));
              Value count = arith::ConstantIntOp::create(builder, loc, n, 32);
              llvm::SmallVector<Value> fractions, indices;
              for (unsigned angle : {0u, 1u}) {
                Value scaled =
                    arith::MulFOp::create(builder, loc, args[angle], scale);
                Value x = arith::AddFOp::create(builder, loc, scaled, shift);
                Value floor = math::FloorOp::create(builder, loc, x);
                fractions.push_back(
                    arith::SubFOp::create(builder, loc, x, floor));
                Value i = arith::FPToSIOp::create(builder, loc,
                                                  builder.getI32Type(), floor);
                Value j = arith::AddIOp::create(builder, loc, i, count);
                indices.push_back(
                    arith::RemSIOp::create(builder, loc, j, count));
              }
              Value map = arith::FPToSIOp::create(
                  builder, loc, builder.getI32Type(), args[2]);
              Value mn = arith::MulIOp::create(builder, loc, map, count);
              Value row = arith::AddIOp::create(builder, loc, mn, indices[0]);
              Value rown = arith::MulIOp::create(builder, loc, row, count);
              Value cell =
                  arith::AddIOp::create(builder, loc, rown, indices[1]);
              llvm::SmallVector<Value> coefficients;
              for (unsigned k = 0; k != 16; ++k) {
                Value column =
                    arith::ConstantIntOp::create(builder, loc, k, 32);
                coefficients.push_back(mdir::md::LookupOp::create(
                    builder, loc, builder.getF64Type(), values[4],
                    ValueRange{cell, column}));
              }
              llvm::SmallVector<Value> rows;
              for (unsigned i = 0; i != 4; ++i) {
                Value last = coefficients[4 * i + 3];
                for (int j = 2; j >= 0; --j) {
                  Value product =
                      arith::MulFOp::create(builder, loc, last, fractions[1]);
                  last = arith::AddFOp::create(builder, loc, product,
                                               coefficients[4 * i + j]);
                }
                rows.push_back(last);
              }
              Value energy = rows[3];
              for (int i = 2; i >= 0; --i) {
                Value product =
                    arith::MulFOp::create(builder, loc, energy, fractions[0]);
                energy = arith::AddFOp::create(builder, loc, product, rows[i]);
              }
              mdir::md::YieldOp::create(builder, loc, ValueRange{energy});
            }
            return sum.getResult();
}

namespace mdir::driver::builder {
class CMapTermBuilder final : public TermBuilder {
public:
  CMapTermBuilder() {
    potentialMask = TermInputs::CMaps;
    nativePotentialMask = potentialMask;
    tupleRank = 2;
  }
  int topologyRank() const override { return 4; }
  void collectTupleSets(TermParameters &context) const override {
    const Topology &topology = context.getTopology();
    if (!topology.cmaps.empty()) {
      // The map of each term, and the bicubic patches of every map: a row
      // of 16 coefficients for each cell.
      Program::TupleSet &set = addTupleSet(context.getTupleSets(), "cmap", 5);
      set.reversible = false;
      size_t map = addTupleField(set, "map");
      for (const Topology::CMap &cmap : topology.cmaps) {
        for (unsigned member : {cmap.i, cmap.j, cmap.k, cmap.l, cmap.m})
          set.members.push_back(member);
        set.fields[map].values.push_back(cmap.map);
      }
      Program::Table table;
      table.name = "cmap";
      table.values = getCMapCoefficients(topology);
      table.columns = 16;
      table.count = table.values.size() / 16;
      context.getTables().push_back(std::move(table));
    }
  }
  void appendTermDefinitions(
      const TermInputs &,
      std::vector<TermDefinition> &definitions) const override {
    definitions.push_back({7, 0, "term_cmap", TermInputs::CMaps});
  }
  llvm::Error buildTopologyPotential(const TermInputs &context, mlir::OpBuilder &builder,
      const TermNativeTopology &state, const PotentialSelection &selection,
      llvm::function_ref<void(mlir::Value)> add) const override {
    if (!(selection.terms & TermInputs::CMaps)) return llvm::Error::success();
    auto relation = state.relation("cmap");
    if (!relation) return llvm::Error::success();
    add(buildCMapsContribution(builder, mlir::ValueRange{relation, state.physical.positions, state.physical.cell, state.tupleField("cmap", "map"), state.physical.table("cmap")}, context.getTopology().cmapResolution));

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
    if ((terms & TermInputs::CMaps) && has("cmap")) {
      // The cell of φ and ψ, each from −180° in steps of 360° / n, and the
      // places t and u in it; φ = 180° is the first cell again. The patch
      // is E = Σ c_ij t^i u^j [MacKerell2004].
      unsigned n = context.getTopology().cmapResolution;
      context.emitIRFragment(
          "  ", {"%u_cmap"}, {"f64"},
          {"%r_cmap", "%x", "%cell", "%f_cmap_map", "%t_cmap"},
          {"!rel_cmap", "!vec", "!md.cell", "!of_cmap", "!grid"},
          [n](mlir::OpBuilder &builder,
              mlir::ValueRange values) -> TermIRValues {
            return {buildCMapsContribution(builder, values, n)};
          });
      add("cmap");
    }
  }
};
} // namespace mdir::driver::builder
std::unique_ptr<TermBuilder> mdir::driver::builder::createCMapTermBuilder() {
  return std::make_unique<CMapTermBuilder>();
}
