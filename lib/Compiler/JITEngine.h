#ifndef MDIR_COMPILER_JITENGINE_H
#define MDIR_COMPILER_JITENGINE_H

#include "mdir/Compiler/CompileCache.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm { class JITEventListener; }

namespace mdir::compiler {
/// The simulation's final object boundary (D199).
class JITEngine {
public:
  /// `codegen` names settings outside the target machine that change the
  /// generated code; it is part of the key of the compile cache. Without
  /// `cache`, the cache is bypassed whatever the environment says
  /// (D[compile-cache-controls]).
  static llvm::Expected<std::unique_ptr<JITEngine>>
  create(mlir::ModuleOp module, std::unique_ptr<llvm::TargetMachine> target,
         llvm::ArrayRef<std::string> libraries, llvm::StringRef entry,
         llvm::StringRef codegen, bool cache = true);
  ~JITEngine();
  llvm::Error registerSymbols(
      llvm::function_ref<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)> map);
  llvm::Error initialize();
  llvm::Expected<void (*)(void **)> lookupPacked(llvm::StringRef name);
  /// What generating the host code cost, and what the cache saved.
  CompileStats getCompileStats() const { return cache->getStats(); }

private:
  // The compile cache (D212) outlives the compiler that uses
  // it. It holds no linked code: each engine links its own copy of an
  // object, as it does a generated one.
  std::unique_ptr<HostObjectCache> cache;
  // The owned listener outlives the object layer's notifications.
  std::unique_ptr<llvm::JITEventListener> perfListener;
  std::unique_ptr<llvm::orc::LLJIT> jit;
  bool initialized = false;
};
} // namespace mdir::compiler
#endif
