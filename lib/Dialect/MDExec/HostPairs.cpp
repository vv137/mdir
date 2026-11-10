// The lanes of a row of a loop over pairs on the host and their
// precondition (D[host-pair-vector]); see
// include/mdir/Dialect/MDExec/HostPairs.h.

#include "mdir/Dialect/MDExec/HostPairs.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mdir/Dialect/MDExec/Algebra.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::md_exec;
using hostpairs::LaneMapping;

LaneMapping hostpairs::getLaneMapping(Operation *op) {
  if (op->getNumRegions() != 0)
    return LaneMapping::None;
  if (isa<YieldOp>(op))
    return LaneMapping::Yield;
  if (isa<arith::ConstantOp, memref::GetGlobalOp>(op))
    return LaneMapping::Uniform;
  if (auto power = dyn_cast<math::FPowIOp>(op))
    return power.getRhs().getDefiningOp<arith::ConstantOp>()
               ? LaneMapping::PowI
               : LaneMapping::None;
  if (isa<arith::ArithDialect, math::MathDialect>(op->getDialect()))
    return op->hasTrait<OpTrait::Elementwise>() ? LaneMapping::Elementwise
                                                : LaneMapping::None;
  if (auto extract = dyn_cast<vector::ExtractOp>(op))
    return extract.getSourceVectorType().getRank() == 1 &&
                   extract.getDynamicPosition().empty() &&
                   !isa<VectorType>(extract.getResult().getType())
               ? LaneMapping::Extract
               : LaneMapping::None;
  if (auto broadcast = dyn_cast<vector::BroadcastOp>(op))
    return !isa<VectorType>(broadcast.getSource().getType()) &&
                   broadcast.getResultVectorType().getRank() == 1
               ? LaneMapping::Broadcast
               : LaneMapping::None;
  if (auto elements = dyn_cast<vector::FromElementsOp>(op))
    return elements.getDest().getType().getRank() == 1
               ? LaneMapping::FromElements
               : LaneMapping::None;
  if (isa<md::LookupOp, memref::LoadOp>(op))
    return LaneMapping::Gather;
  if (auto load = dyn_cast<vector::LoadOp>(op))
    return load.getVectorType().getRank() == 1 ? LaneMapping::Gather
                                               : LaneMapping::None;
  return LaneMapping::None;
}

hostpairs::Refusal hostpairs::checkWidth(int64_t lanes) {
  if (lanes < 1 || lanes > kMaxLanes || !llvm::isPowerOf2_64(lanes))
    return std::string("the lanes are a power of two from 1 to ") +
           std::to_string(kMaxLanes) + ", not " + std::to_string(lanes);
  return std::nullopt;
}

/// Whether a value of `type` can be held in lanes: a number of an integer,
/// a float, or an index, or a vector of one dimension of these.
static bool isLaneType(Type type) {
  if (auto vector = dyn_cast<VectorType>(type))
    return vector.getRank() == 1 && !vector.isScalable() &&
           isLaneType(vector.getElementType());
  return type.isIntOrIndexOrFloat();
}

Type hostpairs::getComputedType(PairForOp op) {
  Block &kernel = op.getKernel().front();
  if (kernel.getNumArguments() == 0)
    return Type();
  return kernel.getArgument(0).getType();
}

hostpairs::Refusal hostpairs::checkLanes(PairForOp op, int64_t lanes) {
  if (Refusal refused = checkWidth(lanes))
    return refused;
  if (op.getTraversal() != Traversal::Directed ||
      op.getConflict() != Conflict::OwnerOnly)
    return std::string("the policy is not (directed, owner_only): the "
                       "lanes write at the centers alone");
  if (op.getCenters() || op.getCenterSet() || op.getGhostFree() ||
      op.getImages() == Images::Explicit)
    return std::string("the loop is over a domain (centers, center_set, "
                       "ghost_free, or images(explicit)), which the first "
                       "slice does not take");
  Type computed = getComputedType(op);
  if (!computed || !(computed.isF32() || computed.isF64()))
    return std::string("the kernel does not compute in f32 or f64");
  for (BlockArgument argument : op.getKernel().front().getArguments())
    if (!isLaneType(argument.getType()))
      return std::string("an argument of the kernel is neither a number "
                         "nor a vector of one dimension");
  for (Operation &nested : op.getKernel().front()) {
    LaneMapping mapping = getLaneMapping(&nested);
    if (mapping == LaneMapping::None) {
      if (auto power = dyn_cast<math::FPowIOp>(&nested))
        return std::string("the exponent of 'math.fpowi' is not a constant");
      return "the kernel holds '" + nested.getName().getStringRef().str() +
             "', which has no lane mapping";
    }
    for (Type type : nested.getResultTypes()) {
      if (mapping == LaneMapping::Uniform && isa<MemRefType>(type))
        continue;
      if (!isLaneType(type))
        return "a value of '" + nested.getName().getStringRef().str() +
               "' is neither a number nor a vector of one dimension";
    }
  }
  if (algebra::needsAlgebra(op) && !algebra::getAlgebras(op))
    return algebra::getMissingMessage();
  return std::nullopt;
}

int64_t hostpairs::getLanes(int64_t vectorBits, Type element) {
  return std::max<int64_t>(1, vectorBits / element.getIntOrFloatBitWidth());
}

int64_t hostpairs::getDeterministicLanes(Type element) {
  return element.isF64() ? 4 : 8;
}

std::optional<hostpairs::Policy> hostpairs::findPolicy(StringRef cpu) {
  // The hosts measured (D[host-pair-vector]); every other host keeps the
  // scalar row outside the deterministic mode.
  static const Policy policies[] = {
      // EPYC 7543 (Zen 3, AVX2): the NVE step of JAC, TIP3P 103k, and
      // Cellulose against main, 1 and 16 threads, mixed and double; the
      // lanes took 1.3 to 1.7 times less time a step, beyond the spread.
      {"znver3", true, "lanes measured faster on an EPYC 7543"},
  };
  for (const Policy &policy : policies)
    if (policy.cpu == cpu)
      return policy;
  return std::nullopt;
}
