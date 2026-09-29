# The driver compiles and runs in its own process, so it links the passes
# and the dialects of MLIR, the translation to LLVM IR, and the execution
# engine.
set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

add_llvm_executable(mdir
  mdir.cpp
  Check.cpp
  Checkpoint.cpp
  Run.cpp
  )
llvm_update_compile_flags(mdir)
target_link_libraries(mdir PRIVATE
  MDIRDriver
  MDIRDyn
  MDIRMD
  MDIRMDExec
  MDIRMDExecToGPU
  MDIRMDExecToLoops
  MDIRMDExecTransforms
  MDIRMDRT
  MDIRMDToMDExec
  MDIRMDTransforms
  MLIRExecutionEngine
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRToLLVMIRTranslationRegistration
  )
# Compiled code calls functions of the driver.
set_target_properties(mdir PROPERTIES ENABLE_EXPORTS ON)

target_compile_definitions(mdir PRIVATE
  MDIR_LLVM_LIBRARY_DIR="${LLVM_LIBRARY_DIR}"
  MDIR_CUDA_ROOT="${MDIR_CUDA_ROOT}"
  MDIR_VERSION="${PROJECT_VERSION}"
  )
