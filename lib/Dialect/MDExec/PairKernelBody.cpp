// The scalar pair computations of the loops over pairs and over the
// segments of a list of units (PairKernelBody.td), for the rewritings of
// the distance (md-exec-simplify-distance, md-exec-expand-radial, D94).
// After #795's interface of the same name.

#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MDRT/SegmentRegions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mdir::md_exec;

#include "mdir/Dialect/MDExec/PairKernelBody.cpp.inc"

SmallVector<PairKernelBody> PairForOp::getPairKernelBodies() {
  Block &body = getKernel().front();
  return {{&getKernel(), body.getArgument(0),
           llvm::to_vector(body.getTerminator()->getOperands())}};
}
double PairForOp::getPairKernelCutoff() {
  return getCutoff().convertToDouble();
}

/// Each site pair of each segment's region is a computation of its own: its
/// squared distance is the norm that its slice forms of its displacement
/// (D405's generator writes the norm; the region's argument r^2 holds the
/// same value), and its outputs are the yields of its mask.
SmallVector<PairKernelBody> SegmentForOp::getPairKernelBodies() {
  SmallVector<PairKernelBody> bodies;
  for (unsigned r = 0, e = getKernels().size(); r != e; ++r) {
    Region &region = getKernels()[r];
    Block &block = region.front();
    Operation *yield = block.getTerminator();
    SmallVector<mdir::mdrt::SitePair> table =
        mdir::mdrt::getSitePairs(getSitesAttr(), r);
    unsigned begin = 0;
    for (unsigned m = 0, em = table.size(); m != em; ++m) {
      unsigned end = begin + table[m].values.size();
      Value displacement = block.getArgument(2 * m + 1);
      Value squared;
      for (Operation *user : displacement.getUsers()) {
        auto product = dyn_cast<arith::MulFOp>(user);
        if (!product || product.getLhs() != displacement ||
            product.getRhs() != displacement || product->getBlock() != &block)
          continue;
        for (Operation *use : product.getResult().getUsers())
          if (auto sum = dyn_cast<vector::ReductionOp>(use);
              sum && sum.getKind() == vector::CombiningKind::ADD &&
              sum->getBlock() == &block) {
            squared = sum.getResult();
            break;
          }
        if (squared)
          break;
      }
      if (squared) {
        PairKernelBody body{&region, squared, {}};
        for (unsigned n = begin; n != end; ++n)
          body.outputs.push_back(yield->getOperand(n));
        bodies.push_back(std::move(body));
      }
      begin = end;
    }
  }
  return bodies;
}
double SegmentForOp::getPairKernelCutoff() {
  return getCutoff().convertToDouble();
}
