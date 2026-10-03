#include "mdir/Dialect/MDDist/MDDistDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace mdir::md_dist;
#include "mdir/Dialect/MDDist/MDDistDialect.cpp.inc"
#define GET_OP_CLASSES
#include "mdir/Dialect/MDDist/MDDistOps.cpp.inc"
void MDDistDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mdir/Dialect/MDDist/MDDistOps.cpp.inc"
      >();
}
LogicalResult PlanOp::verifyRegions() {
  if (getGrid().size() != 3 ||
      llvm::any_of(getGrid(), [](int64_t n) { return n <= 0; }))
    return emitOpError("requires three positive grid dimensions");
  if (getBody().empty())
    return emitOpError("requires a nonempty plan body");
  auto &block = getBody().front();
  if ((block.getNumArguments() != 2 && block.getNumArguments() != 3) ||
      !isa<mdir::mdrt::LayoutType>(block.getArgument(0).getType()) ||
      !isa<mdir::mdrt::TransferMapType>(block.getArgument(1).getType()))
    return emitOpError("requires layout and transfer-map block arguments");
  auto layout = cast<mdir::mdrt::LayoutType>(block.getArgument(0).getType());
  auto map = cast<mdir::mdrt::TransferMapType>(block.getArgument(1).getType());
  if (layout.getParticleSet() != map.getParticleSet())
    return emitOpError("layout and map must name the same particle set");
  bool topology = block.getNumArguments() == 3;
  if (topology) {
    auto type =
        dyn_cast<mdir::mdrt::TransferMapType>(block.getArgument(2).getType());
    if (!type || type.getParticleSet() != layout.getParticleSet())
      return emitOpError("topology map must name the layout's particle set");
  }
  Value topologyEvent, reverseEvent;
  bool topologyReady = false, bonds = false, applied = false;
  Value event;
  bool waited = false, interior = false, boundary = false;
  FlatSymbolRefAttr kernel;
  for (Operation &op : getBody().front()) {
    if (auto start = dyn_cast<HaloStartOp>(op)) {
      if (start.getLayout() != block.getArgument(0) ||
          start.getMap() != block.getArgument(1))
        return start.emitOpError(
            "requires the layout and map bound by this plan");
      if (event)
        return start.emitOpError("duplicate halo start");
      event = start.getEvent();
      if (!event.hasOneUse())
        return start.emitOpError("event requires exactly one wait");
    } else if (auto wait = dyn_cast<HaloWaitOp>(op)) {
      if (!event || waited || wait.getEvent() != event)
        return wait.emitOpError("requires the unmatched event from this plan");
      waited = true;
    } else if (auto dispatch = dyn_cast<DispatchOp>(op)) {
      if (!event || boundary)
        return dispatch.emitOpError("invalid dispatch order");
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          getOperation(), dispatch.getCalleeAttr());
      auto ctx = getContext();
      auto i32 = IntegerType::get(ctx, 32);
      auto f64 = Float64Type::get(ctx);
      SmallVector<Type> inputs{
          MemRefType::get({ShapedType::kDynamic}, i32),
          MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, i32),
          MemRefType::get({ShapedType::kDynamic, 3}, f64),
          MemRefType::get({ShapedType::kDynamic, 3}, f64),
          MemRefType::get({10}, f64)};
      if (!callee ||
          callee.getFunctionType() != FunctionType::get(ctx, inputs, {}))
        return dispatch.emitOpError(
            "requires the fixed-layout pair-kernel ABI");
      if (kernel && kernel != dispatch.getCalleeAttr())
        return dispatch.emitOpError("subsets must dispatch the same kernel");
      kernel = dispatch.getCalleeAttr();
      if (dispatch.getSubset() == "interior") {
        if (interior)
          return dispatch.emitOpError("duplicate interior dispatch");
        interior = true;
      } else if (dispatch.getSubset() == "boundary") {
        if (!waited || !interior)
          return dispatch.emitOpError(
              "boundary requires halo completion and interior dispatch");
        boundary = true;
      } else
        return dispatch.emitOpError("unknown center subset");
    } else if (auto start = dyn_cast<TopologyStartOp>(op)) {
      if (!topology || topologyEvent ||
          start.getLayout() != block.getArgument(0) ||
          start.getMap() != block.getArgument(2))
        return start.emitOpError(
            "requires the unused topology map bound by this plan");
      topologyEvent = start.getEvent();
      if (!topologyEvent.hasOneUse())
        return start.emitOpError("event requires exactly one wait");
    } else if (auto wait = dyn_cast<TopologyWaitOp>(op)) {
      if (!topologyEvent || topologyReady || wait.getEvent() != topologyEvent)
        return wait.emitOpError("requires the unmatched topology event");
      topologyReady = true;
    } else if (auto dispatch = dyn_cast<BondDispatchOp>(op)) {
      if (!topologyReady || bonds || !boundary)
        return dispatch.emitOpError(
            "bond dispatch requires completed topology and both pair subsets");
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          getOperation(), dispatch.getCalleeAttr());
      auto ctx = getContext();
      auto f64 = Float64Type::get(ctx);
      SmallVector<Type> inputs{
          MemRefType::get({ShapedType::kDynamic, 2}, IntegerType::get(ctx, 32)),
          MemRefType::get({ShapedType::kDynamic, 2}, f64),
          MemRefType::get({ShapedType::kDynamic, 3}, f64),
          MemRefType::get({ShapedType::kDynamic, 6}, f64),
          MemRefType::get({ShapedType::kDynamic, 10}, f64)};
      if (!callee ||
          callee.getFunctionType() != FunctionType::get(ctx, inputs, {}))
        return dispatch.emitOpError(
            "requires the per-bond contribution kernel ABI");
      bonds = true;
    } else if (auto start = dyn_cast<ReverseStartOp>(op)) {
      if (!bonds || reverseEvent || start.getLayout() != block.getArgument(0) ||
          start.getMap() != block.getArgument(2))
        return start.emitOpError("reverse requires completed bond dispatch and "
                                 "the same topology map");
      reverseEvent = start.getEvent();
      if (!reverseEvent.hasOneUse())
        return start.emitOpError("event requires exactly one wait");
    } else if (auto wait = dyn_cast<ReverseWaitOp>(op)) {
      if (!reverseEvent || applied || wait.getEvent() != reverseEvent)
        return wait.emitOpError("requires the unmatched reverse event");
      applied = true;
    } else
      return op.emitOpError("unsupported operation in fixed-layout plan");
  }
  if (!event || !waited || !interior || !boundary)
    return emitOpError(
        "requires start, wait, interior, and boundary exactly once");
  if (topology && (!topologyReady || !bonds || !applied))
    return emitOpError(
        "requires topology transfer, bond dispatch, and reverse completion");
  return success();
}
