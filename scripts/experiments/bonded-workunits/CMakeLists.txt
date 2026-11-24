# Offline experiment over the driver's prepared topology.
add_executable(bonded-partition partition.cpp)
target_link_libraries(bonded-partition PRIVATE MDIRDriver)
set_target_properties(bonded-partition PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
llvm_update_compile_flags(bonded-partition)

if(MDIR_CUDA_FOUND AND CUDAToolkit_NVCC_EXECUTABLE)
  find_package(CUDAToolkit REQUIRED)
  set(bonded_sources ${CMAKE_CURRENT_SOURCE_DIR})
  set(bonded_check_obj ${CMAKE_CURRENT_BINARY_DIR}/bonded-device-check.o)
  add_custom_command(OUTPUT ${bonded_check_obj}
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -std=c++17 -O3 -arch=compute_89
            -code=compute_89 -c ${bonded_sources}/device-check.cu -o ${bonded_check_obj}
    DEPENDS ${bonded_sources}/device-check.cu ${bonded_sources}/partition.h
    VERBATIM)
  add_executable(bonded-device-check partition.cpp ${bonded_check_obj})
  target_compile_definitions(bonded-device-check PRIVATE MDIR_CHECK_BONDED_DEVICE)
  target_link_libraries(bonded-device-check PRIVATE MDIRDriver CUDA::cudart)
  set_target_properties(bonded-device-check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  llvm_update_compile_flags(bonded-device-check)
endif()
