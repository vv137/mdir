// Set identity on storage-form buffers and its lowering boundary.

#include "mdir/Dialect/MDExec/StorageSet.h"

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace mdir::md_exec;

static FlatSymbolRefAttr getSet(Type type) {
  auto buffer = dyn_cast<MemRefType>(type);
  if (!buffer)
    return {};
  if (auto layout = dyn_cast_if_present<SetLayoutAttr>(buffer.getLayout()))
    return layout.getSet();
  return {};
}

static Type eraseSet(Type type) {
  auto buffer = dyn_cast<MemRefType>(type);
  if (!buffer || !getSet(type))
    return type;
  return MemRefType::get(buffer.getShape(), buffer.getElementType(),
                         MemRefLayoutAttrInterface(), buffer.getMemorySpace());
}

static FlatSymbolRefAttr getProvenance(Value value) {
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return getProvenance(cast.getSource());
  if (auto cast = value.getDefiningOp<memref::MemorySpaceCastOp>())
    return getProvenance(cast.getSource());
  return getSet(value.getType());
}

LogicalResult mdir::md_exec::eraseStorageSets(ModuleOp module) {
  SmallVector<memref::CastOp> casts;
  bool valid = true;
  module.walk([&](memref::CastOp cast) {
    FlatSymbolRefAttr source = getProvenance(cast.getSource());
    FlatSymbolRefAttr target = getSet(cast.getType());
    if (source && target && source != target) {
      cast.emitOpError() << "cannot reinterpret storage of set " << source
                         << " as set " << target;
      valid = false;
      return;
    }
    if ((source || target) &&
        eraseSet(cast.getSource().getType()) == eraseSet(cast.getType()))
      casts.push_back(cast);
  });
  if (!valid)
    return failure();
  for (memref::CastOp cast : casts) {
    cast.getResult().replaceAllUsesWith(cast.getSource());
    cast.erase();
  }

  module.walk([&](Operation *op) {
    for (Value result : op->getResults())
      result.setType(eraseSet(result.getType()));
    SmallVector<NamedAttribute> attributes(op->getAttrs());
    for (NamedAttribute named : attributes)
      if (auto type = dyn_cast<TypeAttr>(named.getValue()))
        if (Type erased = eraseSet(type.getValue()); erased != type.getValue())
          op->setAttr(named.getName(), TypeAttr::get(erased));
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          argument.setType(eraseSet(argument.getType()));
  });
  module.walk([&](func::FuncOp function) {
    FunctionType old = function.getFunctionType();
    SmallVector<Type> inputs, results;
    for (Type type : old.getInputs())
      inputs.push_back(eraseSet(type));
    for (Type type : old.getResults())
      results.push_back(eraseSet(type));
    function.setType(FunctionType::get(module.getContext(), inputs, results));
  });
  module.walk([](Operation *op) { op->removeAttr("md_exec.tuple_set"); });
  module->removeAttr("md_exec.typed_storage");
  return success();
}
