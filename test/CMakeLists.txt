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

# The plugin of learned potentials and the interpreter of its Python
# environment (torch, metatomic, NequIP), which builds the test's artifact.
if(MDIR_ENABLE_METATOMIC)
  set(MDIR_TEST_METATOMIC 1)
else()
  set(MDIR_TEST_METATOMIC 0)
endif()
set(MDIR_MLIP_PYTHON "" CACHE FILEPATH
  "The Python of the environment that the plugin was built against")

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

add_llvm_executable(mdir-term-ir-test Driver/TermIRTest.cpp)
llvm_update_compile_flags(mdir-term-ir-test)
target_link_libraries(mdir-term-ir-test PRIVATE MDIRDriver MLIRParser)

add_llvm_executable(mdir-prolate-test Driver/ProlateTest.cpp)
llvm_update_compile_flags(mdir-prolate-test)
target_link_libraries(mdir-prolate-test PRIVATE MDIRDriver MDIRMD)

add_llvm_executable(mdir-neighbor-count-test Driver/NeighborCountTest.cpp)
llvm_update_compile_flags(mdir-neighbor-count-test)
target_link_libraries(mdir-neighbor-count-test PRIVATE MDIRDriver)

add_llvm_executable(mdir-influence-test Driver/InfluenceTest.cpp)
llvm_update_compile_flags(mdir-influence-test)
target_link_libraries(mdir-influence-test PRIVATE MDIRDriver)

add_llvm_executable(mdir-output-rows-test Driver/OutputRowsTest.cpp)
llvm_update_compile_flags(mdir-output-rows-test)
target_link_libraries(mdir-output-rows-test PRIVATE MDIRDriver)

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
  mdir-term-ir-test
  mdir-jit-memory-test
  mdir-compile-cache-test
  mdir-compile-test
  mdir-model-test
  mdir-output-rows-test
  mdir-prolate-test
  mdir-influence-test
  mdir-neighbor-count-test
  mdir
  mdir-opt
  mdrt
  )
if(MDIR_CUDA_FOUND)
  find_package(CUDAToolkit REQUIRED)
  add_executable(mdir-cuda-copy-scope-test Integration/Inputs/cuda_scoped_copy.c)
  target_link_libraries(mdir-cuda-copy-scope-test PRIVATE mdrt_cuda CUDA::cuda_driver)
  set_target_properties(mdir-cuda-copy-scope-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  list(APPEND MDIR_TEST_DEPENDS mdrt_cuda)
  list(APPEND MDIR_TEST_DEPENDS mdir-cuda-copy-scope-test)
endif()

add_lit_testsuite(check-mdir "Running the MDIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MDIR_TEST_DEPENDS}
  )
if(MDIR_ENABLE_PYTHON)
  add_dependencies(check-mdir mdir_python)
endif()
set_target_properties(check-mdir PROPERTIES FOLDER "Tests")
