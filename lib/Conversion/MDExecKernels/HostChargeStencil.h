// Shared orthorhombic CPU charge assignment [Essmann1995]. The ordinary
// grid and a periodic domain subgrid differ in their starting indices
// and in the accumulation performed at each point.
#pragma once
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mdir::kernels {
inline void emitHostChargeStencil(
    mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange first,
    mlir::ValueRange shape, mlir::Value order, mlir::Value charge,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, unsigned, mlir::Value)>
        weight,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Value, mlir::Value)> add) {
  using namespace mlir;
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  auto wrap = [&](OpBuilder &at, unsigned axis, Value delta) -> Value {
    Value i = arith::AddIOp::create(at, loc, first[axis], delta);
    Value wrapped = arith::SubIOp::create(at, loc, i, shape[axis]);
    Value beyond = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::uge, i,
                                         shape[axis]);
    return arith::SelectOp::create(at, loc, beyond, wrapped, i);
  };
  scf::ForOp::create(
      b, loc, zero, order, one, ValueRange{},
      [&](OpBuilder &x, Location, Value i, ValueRange) {
        Value gx = wrap(x, 0, i);
        Value qx = arith::MulFOp::create(x, loc, charge, weight(x, 0, i));
        scf::ForOp::create(
            x, loc, zero, order, one, ValueRange{},
            [&](OpBuilder &y, Location, Value j, ValueRange) {
              Value gy = wrap(y, 1, j);
              Value qxy = arith::MulFOp::create(y, loc, qx, weight(y, 1, j));
              Value row = arith::MulIOp::create(y, loc, gx, shape[1]);
              row = arith::AddIOp::create(y, loc, row, gy);
              Value base = arith::MulIOp::create(y, loc, row, shape[2]);
              scf::ForOp::create(
                  y, loc, zero, order, one, ValueRange{},
                  [&](OpBuilder &z, Location, Value k, ValueRange) {
                    Value gz = wrap(z, 2, k);
                    Value value =
                        arith::MulFOp::create(z, loc, qxy, weight(z, 2, k));
                    add(z, arith::AddIOp::create(z, loc, base, gz), value);
                    scf::YieldOp::create(z, loc);
                  });
              scf::YieldOp::create(y, loc);
            });
        scf::YieldOp::create(x, loc);
      });
}
} // namespace mdir::kernels
