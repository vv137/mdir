set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MD)
add_public_tablegen_target(MDIRMDPassIncGen)

set(LLVM_TARGET_DEFINITIONS DerivativeInterface.td)
mlir_tablegen(DerivativeInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(DerivativeInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MDIRDerivativeInterfaceIncGen)
