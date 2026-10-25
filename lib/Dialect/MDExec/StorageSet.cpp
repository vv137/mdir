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
  if (auto cast = value.getDefiningOp<memref::CastOp>()) {
    if (hasInputSetBoundary(cast))
      return getSet(cast.getType());
    return getProvenance(cast.getSource());
  }
  if (auto cast = value.getDefiningOp<memref::MemorySpaceCastOp>())
    return getProvenance(cast.getSource());
  if (auto view = value.getDefiningOp<memref::SubViewOp>())
    return getProvenance(view.getSource());
  if (auto view = value.getDefiningOp<memref::ReinterpretCastOp>())
    return getProvenance(view.getSource());
  if (auto view = value.getDefiningOp<memref::ViewOp>())
    return getProvenance(view.getSource());
  return getSet(value.getType());
}

static LogicalResult verifySetCast(memref::CastOp cast) {
  FlatSymbolRefAttr source = getProvenance(cast.getSource());
  FlatSymbolRefAttr target = getSet(cast.getType());
  Attribute boundary = cast->getAttr(kInputSetBoundaryAttrName);
  bool inputBoundary = static_cast<bool>(boundary);
  if (inputBoundary && !isa<UnitAttr>(boundary))
    return cast.emitOpError()
           << "input set boundary must be a unit attribute";
  if (inputBoundary && (source || !target))
    return cast.emitOpError() << "input set boundary must tag a plain buffer";
  if (target && !source && !inputBoundary)
    return cast.emitOpError()
           << "cannot infer the source set of a cast to set " << target;
  if (source && target && source != target)
    return cast.emitOpError() << "cannot reinterpret storage of set " << source
                              << " as set " << target;
  return success();
}

bool mdir::md_exec::hasInputSetBoundary(memref::CastOp cast) {
  return cast->hasAttr(kInputSetBoundaryAttrName);
}

void mdir::md_exec::markInputSetBoundary(memref::CastOp cast) {
  cast->setAttr(kInputSetBoundaryAttrName, UnitAttr::get(cast.getContext()));
}

LogicalResult mdir::md_exec::verifyInputSetBoundaryAttribute(Operation *op,
                                                              Attribute value) {
  auto cast = dyn_cast<memref::CastOp>(op);
  if (!cast || !isa<UnitAttr>(value))
    return op->emitError()
           << "expects '" << kInputSetBoundaryAttrName
           << "' as a unit attribute on memref.cast";
  return verifySetCast(cast);
}

LogicalResult mdir::md_exec::verifyStorageSetCasts(ModuleOp module) {
  WalkResult result = module.walk([&](memref::CastOp cast) {
    return failed(verifySetCast(cast)) ? WalkResult::interrupt()
                                       : WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult mdir::md_exec::eraseStorageSets(ModuleOp module) {
  if (failed(verifyStorageSetCasts(module)))
    return failure();
  SmallVector<memref::CastOp> casts;
  module.walk([&](memref::CastOp cast) {
    FlatSymbolRefAttr source = getProvenance(cast.getSource());
    FlatSymbolRefAttr target = getSet(cast.getType());
    if ((source || target) &&
        eraseSet(cast.getSource().getType()) == eraseSet(cast.getType()))
      casts.push_back(cast);
  });
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
