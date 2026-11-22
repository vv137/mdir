// Scalar operations and memref accesses shared by kernel builders.
#pragma once
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mdir::kernels::detail {
using namespace mlir;
class KernelCode {
protected:
  KernelCode(OpBuilder &builder, Location location)
      : b(builder), loc(location) {}

public:
  OpBuilder &b;
  Location loc;
  // Arithmetic.
#define MDIR_BINARY(name, Op)                                                  \
  Value name(Value x, Value y) { return arith::Op::create(b, loc, x, y); }
  MDIR_BINARY(addi, AddIOp)
  MDIR_BINARY(subi, SubIOp)
  MDIR_BINARY(muli, MulIOp)
  MDIR_BINARY(divui, DivUIOp)
  MDIR_BINARY(divsi, DivSIOp)
  MDIR_BINARY(remui, RemUIOp)
  MDIR_BINARY(remsi, RemSIOp)
  MDIR_BINARY(andi, AndIOp)
  MDIR_BINARY(ori, OrIOp)
  MDIR_BINARY(xori, XOrIOp)
  MDIR_BINARY(shli, ShLIOp)
  MDIR_BINARY(shrui, ShRUIOp)
  MDIR_BINARY(shrsi, ShRSIOp)
  MDIR_BINARY(minsi, MinSIOp)
  MDIR_BINARY(maxsi, MaxSIOp)
  MDIR_BINARY(minui, MinUIOp)
  MDIR_BINARY(maxui, MaxUIOp)
  MDIR_BINARY(addf, AddFOp)
  MDIR_BINARY(subf, SubFOp)
  MDIR_BINARY(mulf, MulFOp)
  MDIR_BINARY(divf, DivFOp)
  MDIR_BINARY(maximumf, MaximumFOp)
  MDIR_BINARY(minimumf, MinimumFOp)
  MDIR_BINARY(maxnumf, MaxNumFOp)
  MDIR_BINARY(minnumf, MinNumFOp)
#undef MDIR_BINARY
  Value cmpi(arith::CmpIPredicate p, Value x, Value y) {
    return arith::CmpIOp::create(b, loc, p, x, y);
  }
  Value cmpf(arith::CmpFPredicate p, Value x, Value y) {
    return arith::CmpFOp::create(b, loc, p, x, y);
  }
  Value select(Value c, Value x, Value y) {
    return arith::SelectOp::create(b, loc, c, x, y);
  }
  Value indexCast(Type type, Value v) {
    return arith::IndexCastOp::create(b, loc, type, v);
  }
  Value sitofp(Type type, Value v) {
    return arith::SIToFPOp::create(b, loc, type, v);
  }
  Value fptosi(Type type, Value v) {
    return arith::FPToSIOp::create(b, loc, type, v);
  }
  Value extract(Value vector, int64_t at) {
    return vector::ExtractOp::create(b, loc, vector, ArrayRef<int64_t>{at});
  }

  // Memory, with no attributes (the parser of a template sets none).
  Value load(Value memref, ValueRange indices) {
    return memref::LoadOp::create(b, loc, memref, indices,
                                  /*nontemporal=*/BoolAttr(),
                                  /*alignment=*/IntegerAttr(),
                                  /*invariant=*/BoolAttr());
  }
  void store(Value v, Value memref, ValueRange indices) {
    memref::StoreOp::create(b, loc, v, memref, indices,
                            /*nontemporal=*/BoolAttr(),
                            /*alignment=*/IntegerAttr());
  }
};
} // namespace mdir::kernels::detail
