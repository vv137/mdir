set(LLVM_TARGET_DEFINITIONS MDDistOps.td)
mlir_tablegen(MDDistOps.h.inc -gen-op-decls)
mlir_tablegen(MDDistOps.cpp.inc -gen-op-defs)
mlir_tablegen(MDDistTypes.h.inc -gen-typedef-decls -typedefs-dialect=md_dist)
mlir_tablegen(MDDistTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=md_dist)
mlir_tablegen(MDDistDialect.h.inc -gen-dialect-decls -dialect=md_dist)
mlir_tablegen(MDDistDialect.cpp.inc -gen-dialect-defs -dialect=md_dist)

# The enumerations come from their own file, since the ops include those of
# md as well.
set(LLVM_TARGET_DEFINITIONS MDDistEnums.td)
mlir_tablegen(MDDistEnums.h.inc -gen-enum-decls)
mlir_tablegen(MDDistEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect_tablegen_target(MDIRMDDistIncGen)
