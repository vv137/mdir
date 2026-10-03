set(LLVM_LINK_COMPONENTS Core Support nativecodegen native)
add_llvm_executable(mdir-cpu-lj main.cpp)
llvm_update_compile_flags(mdir-cpu-lj)
target_link_libraries(mdir-cpu-lj PRIVATE MPI::MPI_CXX MDIRMD MDIRMDExec
  MDIRMDDist MDIRMDRT MDIRMDExecToLoops MDIRMDExecToGPU MDIRMDToMDExec
  MLIRExecutionEngine MLIRRegisterAllDialects MLIRRegisterAllExtensions
  MLIRRegisterAllPasses MLIRToLLVMIRTranslationRegistration)
target_compile_definitions(mdir-cpu-lj PRIVATE
  MDIR_LLVM_LIBRARY_DIR="${LLVM_LIBRARY_DIR}")
