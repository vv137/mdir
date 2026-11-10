#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mdir/Dialect/MD/MDTemplateWindow.h"
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
double PairForOp::getPairKernelCutoff() { return getCutoff().convertToDouble(); }

SmallVector<PairKernelBody> TemplateForOp::getPairKernelBodies() {
  SmallVector<PairKernelBody> bodies;
  unsigned columns = getOuts().size() + getReduce().size();
  constexpr unsigned arities[] = {6, 4, 2};
  for (unsigned tag = 0; tag != 3; ++tag) {
    Region &region = (*this)->getRegion(tag);
    Operation *yield = region.front().getTerminator();
    llvm::DenseMap<Value, unsigned> seen;
    unsigned offset = 0;
    for (unsigned column = 0; column != columns; ++column) {
      unsigned stride = column < getOuts().size()
          ? mdir::md::getTemplateForceArity(*this, arities[tag], column, getOuts().size()) : 1;
      unsigned count = getPairs()[tag * columns + column];
      for (unsigned pair = 0; pair != count; ++pair) {
        Value displacement = yield->getOperand(offset + stride * count + pair);
        Value squared;
        // The source-derived body retains the site's displacement beside
        // its outputs. Recognize its existing norm, without introducing
        // a distance or rewriting a body that has no scalar distance use.
        for (Operation *user : displacement.getUsers()) {
          auto product = dyn_cast<arith::MulFOp>(user);
          if (!product || product.getLhs() != displacement ||
              product.getRhs() != displacement || product->getBlock() != &region.front())
            continue;
          for (Operation *use : product.getResult().getUsers())
            if (auto sum = dyn_cast<vector::ReductionOp>(use);
                sum && sum.getKind() == vector::CombiningKind::ADD &&
                sum->getBlock() == &region.front()) {
              squared = sum.getResult();
              break;
            }
          if (squared) break;
        }
        if (!squared) continue;
        auto [found, inserted] = seen.try_emplace(squared, bodies.size());
        if (inserted) bodies.push_back({&region, squared, {}});
        for (unsigned member = 0; member != stride; ++member)
          bodies[found->second].outputs.push_back(
              yield->getOperand(offset + pair * stride + member));
      }
      offset += count * (stride + 1);
    }
  }
  return bodies;
}
double TemplateForOp::getPairKernelCutoff() { return getCutoff().convertToDouble(); }
