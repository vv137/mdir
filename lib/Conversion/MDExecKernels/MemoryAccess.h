// Unadorned memref accesses shared by kernel builders.
#pragma once
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mdir::kernels::detail {
using namespace mlir;
class MemoryAccess {
protected:
  MemoryAccess(OpBuilder &builder, Location location)
      : b(builder), loc(location) {}

public:
  OpBuilder &b;
  Location loc;
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
