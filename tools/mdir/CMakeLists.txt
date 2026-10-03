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
  BugReport.cpp
  Check.cpp
  Checkpoint.cpp
  Manifest.cpp
  Doctor.cpp
  Run.cpp
  )

# The commit that was built, refreshed at every build.
set(MDIR_BUILD_INFO "${CMAKE_CURRENT_BINARY_DIR}/BuildInfo.h")
add_custom_target(mdir-build-info
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -DOUTPUT=${MDIR_BUILD_INFO}
          -P ${PROJECT_SOURCE_DIR}/cmake/BuildInfo.cmake
  BYPRODUCTS ${MDIR_BUILD_INFO}
  COMMENT "Recording the commit of the source"
  )
add_dependencies(mdir mdir-build-info)
target_include_directories(mdir PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
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
  MDIR_CUDA_VERSION="${MDIR_CUDA_VERSION}"
  MDIR_CUDA_ROOT="${MDIR_CUDA_ROOT}"
  MDIR_HAS_CUDA=$<BOOL:${MDIR_CUDA_FOUND}>
  MDIR_VERSION="${PROJECT_VERSION}"
  )

# Installed as <prefix>/bin/mdir beside <prefix>/lib, where the driver looks
# for its runtime (Run.cpp). The libraries found at link time outside the
# build tree, such as HDF5, stay on the search path of the installed binary.
set_target_properties(mdir PROPERTIES
  INSTALL_RPATH "$ORIGIN/../lib"
  INSTALL_RPATH_USE_LINK_PATH ON
  )
install(TARGETS mdir RUNTIME DESTINATION bin COMPONENT mdir)
