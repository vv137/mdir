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
  if (argc != 2 && !(argc == 3 && llvm::StringRef(argv[2]) == "--table-types"))
    return 1;
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::gpu::GPUDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!module)
    return 1;
  if (argc == 3) {
    mdir::driver::Program program;
    program.tables = {{"lj", 9, {}}, {"prolate", 128, {}, 4}};
    auto result = module->walk([&](mlir::gpu::AllocOp alloc) {
      auto type = mdir::tool::getTowersTableType(program, alloc.getMemref());
      if (!type)
        return mlir::WalkResult::interrupt();
      llvm::outs() << type << "\n";
      return mlir::WalkResult::advance();
    });
    return result.wasInterrupted();
  }
  llvm::SmallVector<mlir::Value> captures;
  module->walk(
      [&](mlir::gpu::MemcpyOp copy) { captures.push_back(copy.getDst()); });
  llvm::ExitOnError take;
  auto buffers = take(mdir::tool::bindTowersTrialInputs({}, {}, captures));
  for (const auto &buffer : buffers.values) {
    uint32_t bits;
    if (buffer.size() != sizeof(bits))
      return 1;
    std::memcpy(&bits, buffer.data(), sizeof(bits));
    llvm::outs() << llvm::format_hex(bits, 10) << "\n";
  }
}
