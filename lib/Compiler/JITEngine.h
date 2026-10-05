#ifndef MDIR_COMPILER_JITENGINE_H
#define MDIR_COMPILER_JITENGINE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Target/TargetMachine.h"

namespace mdir::compiler {
/// The simulation's final object boundary (D[jit-invariants]).
class JITEngine {
public:
  static llvm::Expected<std::unique_ptr<JITEngine>>
  create(mlir::ModuleOp module, std::unique_ptr<llvm::TargetMachine> target,
         llvm::ArrayRef<std::string> libraries, llvm::StringRef entry);
  ~JITEngine();
  llvm::Error registerSymbols(
      llvm::function_ref<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)> map);
  llvm::Error initialize();
  llvm::Expected<void (*)(void **)> lookupPacked(llvm::StringRef name);

private:
  std::unique_ptr<llvm::orc::LLJIT> jit;
  bool initialized = false;
};
} // namespace mdir::compiler
#endif
