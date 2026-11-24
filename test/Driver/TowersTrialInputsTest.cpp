// Interpret the host table initializer and print the trial buffer's f32 bits.
#include "TowersTrialInputs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Format.h"
#include <cstring>

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::gpu::GPUDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!module)
    return 1;
  llvm::SmallVector<mlir::Value> captures;
  module->walk(
      [&](mlir::gpu::MemcpyOp copy) { captures.push_back(copy.getDst()); });
  llvm::ExitOnError take;
  auto buffers = take(mdir::tool::bindTowersTrialInputs({}, {}, captures));
  for (const auto &buffer : buffers) {
    uint32_t bits;
    if (buffer.size() != sizeof(bits))
      return 1;
    std::memcpy(&bits, buffer.data(), sizeof(bits));
    llvm::outs() << llvm::format_hex(bits, 10) << "\n";
  }
}
