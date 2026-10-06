add_subdirectory(lib/Dialect/MD)

# Optional Python tests use the interpreter that owns the extension ABI.
set(MDIR_TEST_PYTHON 0)
if(MDIR_ENABLE_PYTHON)
  set(MDIR_TEST_PYTHON 1)
endif()
# For the site configuration of lit.
if(MDIR_CUDA_FOUND)
  set(MDIR_TEST_CUDA 1)
else()
  set(MDIR_TEST_CUDA 0)
endif()

get_property(MDIR_HDF5_FOUND GLOBAL PROPERTY MDIR_HDF5_FOUND)
if(MDIR_HDF5_FOUND)
  set(MDIR_TEST_HDF5 1)
else()
  set(MDIR_TEST_HDF5 0)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

add_llvm_executable(mdir-model-test Driver/ModelTest.cpp)
llvm_update_compile_flags(mdir-model-test)
target_link_libraries(mdir-model-test PRIVATE MDIRDriver)

add_llvm_executable(mdir-compile-test Driver/CompileTest.cpp)
llvm_update_compile_flags(mdir-compile-test)
target_link_libraries(mdir-compile-test PRIVATE MDIRCompiler)

add_llvm_executable(mdir-jit-memory-test Driver/JITMemoryTest.cpp)
llvm_update_compile_flags(mdir-jit-memory-test)
target_compile_options(mdir-jit-memory-test PRIVATE -fexceptions)
target_link_libraries(mdir-jit-memory-test PRIVATE MDIRSimulation MLIRParser)

add_llvm_executable(mdir-compile-cache-test Driver/CompileCacheTest.cpp)
llvm_update_compile_flags(mdir-compile-cache-test)
target_link_libraries(mdir-compile-cache-test PRIVATE MDIRSimulation)

set(MDIR_TEST_DEPENDS
  mdir-jit-memory-test
  mdir-compile-cache-test
  mdir-compile-test
  mdir-model-test
  mdir
  mdir-opt
  mdrt
  )
if(MDIR_CUDA_FOUND)
  list(APPEND MDIR_TEST_DEPENDS mdrt_cuda)
endif()

add_lit_testsuite(check-mdir "Running the MDIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MDIR_TEST_DEPENDS}
  )
if(MDIR_ENABLE_PYTHON)
  add_dependencies(check-mdir mdir_python)
endif()
set_target_properties(check-mdir PROPERTIES FOLDER "Tests")
