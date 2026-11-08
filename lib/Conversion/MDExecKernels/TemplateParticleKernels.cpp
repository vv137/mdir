#include "mdir/Conversion/TemplateParticleKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace mdir::kernels;

void mdir::kernels::emitTemplateParticleKernel(OpBuilder &b, Location loc,
    mdir::md_exec::ParticleForOp op, Value particle,
    Value slots, Value members, Value waters, const IRMapping &captures,
    llvm::function_ref<void(OpBuilder &, Value,
                            ArrayRef<TemplateParticleResult>)> finish) {
  auto word = [&](OpBuilder &at, int64_t n) -> Value {
    return arith::ConstantIntOp::create(at, loc, n, 32);
  };
  Value slot = memref::LoadOp::create(b, loc, slots, ValueRange{particle});
  Value waterRows = arith::MulIOp::create(b, loc,
      arith::IndexCastOp::create(b, loc, b.getI32Type(), waters), word(b, 3));
  Value real = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, slot, word(b, 0));
  Value water = arith::AndIOp::create(b, loc, real,
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, slot, waterRows));
  auto evaluate = [&](OpBuilder &at, Value member) {
    TemplateParticleResult result;
    result.particle = member;
    IRMapping inside(captures);
    Block &kernel = op.getKernel().front();
    for (auto [i, buffer] : llvm::enumerate(op.getIns()))
      inside.map(kernel.getArgument(i), loadElement(at, loc, buffer, member));
    for (Operation &nested : kernel.without_terminator()) at.clone(nested, inside);
    Operation *yield = kernel.getTerminator();
    for (auto [i, value] : llvm::enumerate(yield->getOperands())) {
      Value mapped = inside.lookupOrDefault(value);
      if (i < op.getOuts().size()) result.outputs.push_back(mapped);
      else result.reductions.push_back(mapped);
    }
    return result;
  };
  auto store = [&](OpBuilder &at, const TemplateParticleResult &result) {
    for (auto [buffer, value] : llvm::zip(op.getOuts(), result.outputs))
      storeElement(at, loc, value, buffer, result.particle);
  };
  scf::IfOp::create(b, loc, water, [&](OpBuilder &at, Location) {
    Value leader = arith::CmpIOp::create(at, loc, arith::CmpIPredicate::eq,
        arith::RemUIOp::create(at, loc, slot, word(at, 3)), word(at, 0));
    scf::IfOp::create(at, loc, leader, [&](OpBuilder &in, Location) {
      SmallVector<TemplateParticleResult, 3> results;
      for (int64_t site = 0; site != 3; ++site) {
        Value row = arith::IndexCastUIOp::create(in, loc, in.getIndexType(),
            arith::AddIOp::create(in, loc, slot, word(in, site)));
        Value member = arith::IndexCastOp::create(in, loc, in.getIndexType(),
            memref::LoadOp::create(in, loc, members, ValueRange{row}));
        results.push_back(evaluate(in, member));
      }
      // This separate store phase is essential when inputs alias outputs.
      for (const auto &result : results) store(in, result);
      Value unit = arith::DivUIOp::create(in, loc, slot, word(in, 3));
      finish(in, unit, results);
      scf::YieldOp::create(in, loc);
    });
    scf::YieldOp::create(at, loc);
  }, [&](OpBuilder &at, Location) {
    auto result = evaluate(at, particle);
    store(at, result);
    Value unit = arith::SelectOp::create(at, loc, real,
        arith::SubIOp::create(at, loc, slot,
            arith::IndexCastOp::create(at, loc, at.getI32Type(),
                arith::MulIOp::create(at, loc, waters, createIndex(at, loc, 2)))),
        word(at, -1));
    finish(at, unit, ArrayRef<TemplateParticleResult>{result});
    scf::YieldOp::create(at, loc);
  });
}
