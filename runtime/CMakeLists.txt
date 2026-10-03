# The runtime is plain C, so that it builds without the C++ standard library
# and loads into any host process.
add_library(mdrt SHARED mdrt.c
  ${PROJECT_SOURCE_DIR}/third_party/pocketfft/pocketfft.c)
target_include_directories(mdrt PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/pocketfft)
# The thermostat takes the math library.
target_link_libraries(mdrt PRIVATE m)
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
  # The FFT of particle mesh Ewald takes cuFFT of the toolkit.
  target_link_libraries(mdrt_cuda PRIVATE CUDA::cuda_driver CUDA::cufft)
  set_target_properties(mdrt_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    C_VISIBILITY_PRESET default
    )
else()
  message(STATUS "CUDA toolkit: not found; the GPU runtime is not built")
endif()

# The runtime goes to <prefix>/lib, beside the driver's <prefix>/bin, and so
# does the OpenMP runtime of the LLVM build, which threaded runs on the CPU
# load; the build tree of LLVM is then not needed at run time.
install(TARGETS mdrt LIBRARY DESTINATION lib COMPONENT mdir)
if(MDIR_CUDA_FOUND)
  install(TARGETS mdrt_cuda LIBRARY DESTINATION lib COMPONENT mdir)
endif()
if(EXISTS "${LLVM_LIBRARY_DIR}/libomp.so")
  install(FILES "${LLVM_LIBRARY_DIR}/libomp.so" DESTINATION lib COMPONENT mdir)
endif()

set(MDIR_CUDA_FOUND ${MDIR_CUDA_FOUND} PARENT_SCOPE)
set(MDIR_CUDA_ROOT "${CUDAToolkit_TARGET_DIR}" PARENT_SCOPE)

set(MDIR_CUDA_VERSION "${CUDAToolkit_VERSION}" PARENT_SCOPE)
