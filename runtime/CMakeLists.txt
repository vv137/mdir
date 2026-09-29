# The runtime is plain C, so that it builds without the C++ standard library
# and loads into any host process.
add_library(mdrt SHARED mdrt.c)
set_target_properties(mdrt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  C_VISIBILITY_PRESET default
  )

# The runtime for NVIDIA GPUs is a library of its own, so that the runtime
# loads on machines without the driver. It needs the header of the CUDA
# toolkit and the library of the driver.
set(MDIR_ENABLE_CUDA "AUTO" CACHE STRING
  "Build the runtime for NVIDIA GPUs: ON, OFF, or AUTO")
set(MDIR_CUDA_FOUND OFF)
if(MDIR_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(MDIR_CUDA_FOUND ON)
  elseif(NOT MDIR_ENABLE_CUDA STREQUAL "AUTO")
    message(FATAL_ERROR
      "MDIR_ENABLE_CUDA is ON, but the CUDA toolkit was not found. "
      "Set CUDAToolkit_ROOT.")
  endif()
endif()

if(MDIR_CUDA_FOUND)
  message(STATUS "CUDA toolkit: ${CUDAToolkit_TARGET_DIR}")
  add_library(mdrt_cuda SHARED mdrt_cuda.c)
  target_link_libraries(mdrt_cuda PRIVATE CUDA::cuda_driver)
  set_target_properties(mdrt_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    C_VISIBILITY_PRESET default
    )
else()
  message(STATUS "CUDA toolkit: not found; the GPU runtime is not built")
endif()

set(MDIR_CUDA_FOUND ${MDIR_CUDA_FOUND} PARENT_SCOPE)
set(MDIR_CUDA_ROOT "${CUDAToolkit_TARGET_DIR}" PARENT_SCOPE)
