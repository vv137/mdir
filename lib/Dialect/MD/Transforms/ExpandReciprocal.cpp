// Selected Gaussian reciprocal composition in semantic primitives (D388).
#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MD/Transforms/GridKernels.h"
#include "mdir/Dialect/MD/Transforms/ReciprocalGrid.h"
#include "mdir/Dialect/MD/Transforms/ScalarDerivative.h"
#include "mdir/Dialect/MD/Transforms/StencilGeometry.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/OperationSupport.h"
#include <cmath>
#include <limits>

using namespace mlir;
using namespace mdir::md;

namespace mdir::md {
#define GEN_PASS_DEF_EXPANDRECIPROCAL
#include "mdir/Dialect/MD/Transforms/Passes.h.inc"
namespace {
Value influence(OpBuilder &b, Location loc, Value index, Value matrix,
                Value moduli, ArrayRef<int64_t> grid, double beta,
                double coulomb) {
  ScalarEmitter e(b, loc);
  auto inverse = emitStencilCellInverse(b, loc, matrix);
  SmallVector<Value> waveNumbers, indices;
  for (int64_t a = 0; a != 3; ++a) {
    Value k = vector::ExtractOp::create(b, loc, index, a);
    indices.push_back(k);
    Value half = arith::ConstantIndexOp::create(b, loc, grid[a]/2);
    Value beyond = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, k, half);
    Value number = arith::IndexCastOp::create(b, loc, b.getI64Type(), k);
    Value extent = arith::ConstantIntOp::create(b, loc, grid[a], 64);
    Value shifted = arith::SubIOp::create(b, loc, number, extent);
    number = arith::SelectOp::create(b, loc, beyond, shifted, number);
    waveNumbers.push_back(arith::SIToFPOp::create(b, loc, b.getF64Type(), number));
  }
  // A physical reciprocal vector is H^{-1} n in the row-cell convention.
  SmallVector<Value> wave;
  for (int a = 0; a != 3; ++a) {
    Value component;
    for (int c = 0; c != 3; ++c)
      component = e.add(component, e.mul(inverse.entries[3*a+c], waveNumbers[c]));
    wave.push_back(component);
  }
  Value squared, factor;
  double pi = std::acos(-1.0);
  Value gauss = e.constant(pi*pi/(beta*beta), b.getF64Type());
  Value prefactor = e.div(e.constant(coulomb/pi, b.getF64Type()), inverse.determinant);
  for (int a = 0; a != 3; ++a) {
    Value square = e.mul(wave[a], wave[a]);
    squared = e.add(squared, square);
    Value exponential = math::ExpOp::create(b, loc, e.neg(e.mul(gauss, square)));
    Value axis = arith::ConstantIndexOp::create(b, loc, a);
    Value modulus = LookupOp::create(b, loc, b.getF64Type(), moduli,
                                    ValueRange{axis, indices[a]});
    Value edge = e.mul(exponential, modulus);
    if (a == 0)
      edge = e.mul(edge, prefactor);
    factor = factor ? e.mul(factor, edge) : edge;
  }
  Value nil = e.constant(0, b.getF64Type());
  Value origin = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OEQ, squared, nil);
  Value safe = arith::SelectOp::create(b, loc, origin,
      e.constant(1, b.getF64Type()), squared);
  Value multiplier = e.div(factor, safe);
  return arith::SelectOp::create(b, loc, origin, nil, multiplier);
}

void productKernel(Block &block, Location loc, ArrayRef<int64_t> shape,
                   double beta, double coulomb) {
  OpBuilder b = OpBuilder::atBlockEnd(&block);
  Value multiplier = influence(b, loc, block.getArgument(0),
      block.getArgument(2), block.getArgument(3), shape, beta, coulomb);
  auto v2 = VectorType::get({2}, b.getF64Type());
  Value broadcast = vector::BroadcastOp::create(b, loc, v2, multiplier);
  Value multiplied = arith::MulFOp::create(b, loc, block.getArgument(1), broadcast);
  YieldOp::create(b, loc, ValueRange{multiplied});
}

void energyKernel(Block &block, Location loc) {
  OpBuilder b = OpBuilder::atBlockEnd(&block);
  Value pairs = arith::MulFOp::create(b, loc, block.getArgument(1), block.getArgument(2));
  Value inner = vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD, pairs);
  ScalarEmitter e(b, loc);
  YieldOp::create(b, loc, ValueRange{e.scale(0.5, inner)});
}

template <typename Populate>
bool equivalentKernel(Operation *source, Populate populate) {
  Operation *expected = source->cloneWithoutRegions();
  Block *body = new Block();
  expected->getRegion(0).push_back(body);
  Block &actual = source->getRegion(0).front();
  for (BlockArgument argument : actual.getArguments())
    body->addArgument(argument.getType(), source->getLoc());
  populate(*body);
  bool same = OperationEquivalence::isRegionEquivalentTo(
      &source->getRegion(0), &expected->getRegion(0),
      OperationEquivalence::IgnoreLocations);
  expected->destroy();
  return same;
}
}

LogicalResult verifyGaussianReciprocalKernel(MapGridOp product) {
  auto beta = product->getAttrOfType<FloatAttr>(kReciprocalBeta);
  auto coulomb = product->getAttrOfType<FloatAttr>(kReciprocalCoulomb);
  auto v9 = VectorType::get({9}, Float64Type::get(product.getContext()));
  if (!beta || !coulomb || !beta.getType().isF64() || !coulomb.getType().isF64() ||
      !std::isfinite(beta.getValueAsDouble()) || beta.getValueAsDouble() <= 0 ||
      !std::isfinite(coulomb.getValueAsDouble()) || product.getInputs().size() != 3 ||
      product.getInputs()[1].getType() != v9 ||
      !isa<TableType>(product.getInputs()[2].getType()))
    return product.emitOpError("expected explicit Gaussian product parameters and operands");
  if (!equivalentKernel(product, [&](Block &body) {
        productKernel(body, product.getLoc(), product.getGrid(),
                      beta.getValueAsDouble(), coulomb.getValueAsDouble());
      }))
    return product.emitOpError("Gaussian product kernel does not match its declared arithmetic; metadata is not a proof");
  return success();
}

LogicalResult verifyReciprocalEnergyKernel(SumGridOp energy) {
  if (energy.getInputs().size() != 2 || !energy.getResult().getType().isF64() ||
      energy.getPairing() != "real_hermitian_half" ||
      !equivalentKernel(energy, [&](Block &body) { energyKernel(body, energy.getLoc()); }))
    return energy.emitOpError("expected the Hermitian spectral half-inner-product energy kernel");
  return success();
}

namespace {
class ExpandReciprocal : public impl::ExpandReciprocalBase<ExpandReciprocal> {
public:
  using impl::ExpandReciprocalBase<ExpandReciprocal>::ExpandReciprocalBase;

private:
  void runOnOperation() final {
    SmallVector<ReciprocalOp> selected;
    getOperation()->walk([&](ReciprocalOp op) {
      auto function = op->getParentOfType<FunctionOpInterface>();
      if (!functions.empty()) {
        auto symbol = function ? SymbolTable::getSymbolName(function) : StringAttr();
        if (!symbol || !llvm::is_contained(functions, symbol.getValue().str()))
          return;
      }
      selected.push_back(op);
    });
    for (auto op : selected) {
      if (op.getDispersion() || op.getProlate() || op.getP3m() || op.getListed() ||
          op.getOrder() != 4 || !std::isfinite(op.getBeta().convertToDouble()) ||
          op.getBeta().convertToDouble() <= 0 ||
          !std::isfinite(op.getCoulomb().convertToDouble())) {
        op.emitOpError("selected reciprocal expansion requires unlisted order-four rectangular Gaussian Coulomb SPME/OPTIMAL");
        signalPassFailure();
        return;
      }
      int64_t count = 1;
      for (int64_t n : op.getGrid()) {
        if (n < 4 || n > std::numeric_limits<int32_t>::max() ||
            count > std::numeric_limits<int64_t>::max()/n) {
          op.emitOpError("selected reciprocal grid extents must be at least four, fit i32 and have a signed-i64 product");
          signalPassFailure();
          return;
        }
        count *= n;
      }
    }
    for (auto op : selected)
      if (failed(expand(op))) {
        signalPassFailure();
        return;
      }
  }

  LogicalResult expand(ReciprocalOp op) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto positions = cast<FieldType>(op.getPositions().getType());
    auto grid = op.getGridAttr();
    auto shape = grid.asArrayRef();
    auto stencilType = GridStencilType::get(b.getContext(),
        positions.getParticleSet(), grid, 4, b.getStringAttr("bspline"));
    Value cell = CheckedRectangularCellOp::create(b, loc,
        op.getCell().getType(), op.getCell());
    Value matrix = emitStencilCellMatrix(b, loc, cell);
    Value relation = GridStencilOp::create(b, loc, stencilType, op.getPositions(), cell);
    auto real = RankedTensorType::get(shape, b.getF64Type());
    auto spread = ScatterStencilOp::create(b, loc, real, relation,
        op.getPositions(), cell, op.getCharges());
    Block *weight = new Block();
    spread.getKernel().push_back(weight);
    weight->addArgument(VectorType::get({3}, b.getF64Type()), loc);
    OpBuilder kernel = OpBuilder::atBlockEnd(weight);
    Value w = BSplineWindowOp::create(kernel, loc, weight->getArgument(0), 4);
    YieldOp::create(kernel, loc, ValueRange{w});
    SmallVector<int64_t> halfShape(shape);
    halfShape[2] = shape[2]/2+1;
    auto spectrumType = RankedTensorType::get(halfShape, ComplexType::get(b.getF64Type()));
    Value spectrum = LinearTransformOp::create(b, loc, spectrumType, spread.getResult(),
        b.getStringAttr("fft3d"), b.getStringAttr("forward"),
        b.getStringAttr("backward"), b.getStringAttr("real_hermitian_half"));
    Operation *product = createGridKernel(b, loc, false, spectrumType,
        ValueRange{spectrum, matrix, op.getModuli()}, grid,
        b.getStringAttr("real_hermitian_half"));
    Block &productBody = product->getRegion(0).front();
    product->setAttr(kReciprocalBeta, op.getBetaAttr());
    product->setAttr(kReciprocalCoulomb, op.getCoulombAttr());
    productKernel(productBody, loc, shape, op.getBeta().convertToDouble(),
                  op.getCoulomb().convertToDouble());
    auto energy = cast<SumGridOp>(createGridKernel(b, loc, true, b.getF64Type(),
        ValueRange{spectrum, product->getResult(0)}, grid,
        b.getStringAttr("real_hermitian_half")));
    energyKernel(energy.getKernel().front(), loc);
    bool derivatives = !op.getForces().use_empty() || !op.getVirial().use_empty() ||
                       (op.getPotential() && !op.getPotential().use_empty());
    if (derivatives) {
      auto reversed = reverseReciprocalGrid(b, energy);
      if (failed(reversed))
        return failure();
      auto forces = MapParticlesOp::create(b, loc, positions,
                                           ValueRange{reversed->positionGradient});
      Block *body = new Block();
      forces.getKernel().push_back(body);
      body->addArgument(positions.getKernelValueType(), loc);
      OpBuilder at = OpBuilder::atBlockEnd(body);
      Value negative = arith::NegFOp::create(at, loc, body->getArgument(0));
      YieldOp::create(at, loc, ValueRange{negative});
      op.getForces().replaceAllUsesWith(forces.getResult());
      op.getVirial().replaceAllUsesWith(reversed->virial);
      if (op.getPotential())
        op.getPotential().replaceAllUsesWith(reversed->chargeGradient);
    }
    op.getEnergy().replaceAllUsesWith(energy.getResult());
    op.erase();
    return success();
  }
};
}
}
