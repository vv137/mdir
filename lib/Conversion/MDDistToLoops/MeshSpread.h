// The mesh runtime has the bounds of the periodic particle subgrid. It
// passes views of the cached windows and the contiguous grid to this
// specialization of the ordinary CPU spreading loop once per domain.
#pragma once
#include "../MDExecKernels/HostChargeStencil.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

namespace mdir {
inline mlir::func::FuncOp getMeshSpread(mlir::ModuleOp module, int64_t order,
                                        bool mixed) {
  using namespace mlir;
  std::string name = "mdir_pme_spread_subgrid_" +
                     std::string(mixed ? "f32_" : "f64_") +
                     std::to_string(order);
  if (auto fn = module.lookupSymbol<func::FuncOp>(name))
    return fn;
  OpBuilder b(module.getContext());
  b.setInsertionPointToEnd(module.getBody());
  Location loc = module.getLoc();
  auto i64 = b.getI64Type();
  auto f64 = b.getF64Type();
  auto index = b.getIndexType();
  auto unranked = [&](Type t) {
    return UnrankedMemRefType::get(t, Attribute{});
  };
  auto fn = func::FuncOp::create(
      b, loc, name,
      b.getFunctionType({unranked(i64), unranked(f64), unranked(f64),
                         unranked(f64), index, index, index},
                        {}));
  fn.setPrivate();
  auto args = fn.addEntryBlock()->getArguments();
  b.setInsertionPointToStart(&fn.front());
  constexpr int64_t dyn = ShapedType::kDynamic;
  auto view = [&](unsigned arg, Type type, ArrayRef<int64_t> shape,
                  ArrayRef<int64_t> strides) -> Value {
    auto layout = StridedLayoutAttr::get(b.getContext(), 0, strides);
    return memref::CastOp::create(b, loc, MemRefType::get(shape, type, layout),
                                  args[arg]);
  };
  Value starts = view(0, i64, {dyn, 3}, {dyn, 1});
  Value weights = view(1, f64, {dyn, 3 * order}, {3 * order, 1});
  Value charges = view(2, f64, {dyn}, {dyn});
  Value grid = view(3, f64, {dyn}, {1});
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value n = arith::ConstantIndexOp::create(b, loc, order);
  Value count = memref::DimOp::create(b, loc, charges, zero);
  scf::ParallelOp::create(
      b, loc, ValueRange{zero}, ValueRange{count}, ValueRange{one},
      [&](OpBuilder &at, Location, ValueRange iv) {
        Value q = memref::LoadOp::create(at, loc, charges, iv);
        Value none = arith::ConstantOp::create(at, loc, at.getF64FloatAttr(0));
        Value active =
            arith::CmpFOp::create(at, loc, arith::CmpFPredicate::UNE, q, none);
        scf::IfOp::create(at, loc, active, [&](OpBuilder &body, Location) {
          auto real = [&](OpBuilder &op, Value v) -> Value {
            return mixed ? Value(arith::TruncFOp::create(op, loc,
                                                         op.getF32Type(), v))
                         : v;
          };
          SmallVector<Value, 3> first;
          for (unsigned a = 0; a < 3; ++a) {
            Value axis = arith::ConstantIndexOp::create(body, loc, a);
            Value v = memref::LoadOp::create(body, loc, starts,
                                             ValueRange{iv[0], axis});
            first.push_back(arith::IndexCastOp::create(body, loc, index, v));
          }
          kernels::emitHostChargeStencil(
              body, loc, first, args.slice(4, 3), n, real(body, q),
              [&](OpBuilder &op, unsigned a, Value k) -> Value {
                Value base = arith::ConstantIndexOp::create(op, loc, order * a);
                Value column = arith::AddIOp::create(op, loc, base, k);
                return real(op,
                            memref::LoadOp::create(op, loc, weights,
                                                   ValueRange{iv[0], column}));
              },
              [&](OpBuilder &op, Value i, Value value) {
                if (mixed)
                  value = arith::ExtFOp::create(op, loc, f64, value);
                memref::AtomicRMWOp::create(op, loc, arith::AtomicRMWKind::addf,
                                            value, grid, ValueRange{i});
              });
          scf::YieldOp::create(body, loc);
        });
        scf::ReduceOp::create(at, loc);
      });
  func::ReturnOp::create(b, loc);
  return fn;
}
} // namespace mdir
