# The runtime is plain C, so that it builds without the C++ standard library
# and loads into any host process.
add_library(mdrt SHARED mdrt.c mdrt_dist.c mdrt_fft.c
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
  # Keep older toolkits and header-only installations on the separate pass.
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13 AND CUDAToolkit_NVCC_EXECUTABLE)
    set(MDIR_PME_CALLBACK 1)
    set(MDIR_PME_CALLBACK 1 PARENT_SCOPE)
    add_custom_command(OUTPUT pme_callback.inc
      BYPRODUCTS pme_callback.fatbin
      COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -dc -fatbin
              -gencode arch=compute_75,code=lto_75
              -gencode arch=compute_89,code=lto_89
              ${CMAKE_CURRENT_SOURCE_DIR}/pme_callback.cu -o pme_callback.fatbin
      COMMAND ${CUDAToolkit_BIN_DIR}/bin2c --name pmeCallback --type char
              --static --const pme_callback.fatbin > pme_callback.inc
      DEPENDS pme_callback.cu pme_callback.h VERBATIM)
    target_sources(mdrt_cuda PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/pme_callback.inc)
    target_include_directories(mdrt_cuda PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    foreach(target mdrt_cuda MDIRMDExecToGPU obj.MDIRMDExecToGPU)
      if(TARGET ${target})
        target_compile_definitions(${target} PRIVATE MDIR_PME_CALLBACK=1)
      endif()
    endforeach()
  endif()
  # The graph certificate kernels are scalar. Embed a second PTX image
  # for Volta when the toolkit can generate it. The driver chooses the
  # highest compatible image at load, retaining the original image on
  # Turing and later devices.
  set(step_targets -arch=compute_75 -code=compute_75)
  if(CUDAToolkit_VERSION VERSION_LESS 13.0)
    set(step_targets -gencode arch=compute_70,code=compute_70
                     -gencode arch=compute_75,code=compute_75)
  endif()
  set(step_object "${CMAKE_CURRENT_BINARY_DIR}/mdrt_step_graph.o")
  add_custom_command(OUTPUT "${step_object}"
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" -std=c++17 -O2 ${step_targets}
      -Xcompiler=-fPIC -c "${CMAKE_CURRENT_SOURCE_DIR}/mdrt_step_graph.cu" -o "${step_object}"
    DEPENDS mdrt_step_graph.cu VERBATIM)
  target_sources(mdrt_cuda PRIVATE "${step_object}")
  set_target_properties(mdrt_cuda PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(mdrt_cuda PRIVATE CUDA::cudart_static mdrt)
  if(SKBUILD AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Like the Python extension, the shipped runtime must load without a
    # dependency on the host's version of libstdc++.
    target_link_options(mdrt_cuda PRIVATE -static-libstdc++)
  endif()
  # The FFT of particle mesh Ewald takes cuFFT of the toolkit.
  # The products of the bodies of neural stages take cuBLAS
  # (D302).
  target_link_libraries(mdrt_cuda PRIVATE CUDA::cuda_driver CUDA::cufft
    CUDA::cublas m)
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

# An installed tree holds code of third parties in binary form: pocketfft
# (compiled into libmdrt), toml++ and LLVM with MLIR (in mdir, and in the
# extension of the Python package), and the OpenMP runtime of LLVM beside
# them. When the tree is distributed (the container image, the release
# tarball, a wheel), their licenses ask for their notices, so every install
# carries them (packaging/licenses/README.md says which clause asks).
function(mdir_install_notices destination component)
  install(FILES ${PROJECT_SOURCE_DIR}/third_party/pocketfft/LICENSE.md
    DESTINATION "${destination}" RENAME pocketfft-LICENSE
    COMPONENT ${component})
  install(FILES ${PROJECT_SOURCE_DIR}/third_party/tomlplusplus/LICENSE
    DESTINATION "${destination}" RENAME tomlplusplus-LICENSE
    COMPONENT ${component})
  install(FILES ${PROJECT_SOURCE_DIR}/packaging/licenses/LLVM-LICENSE.TXT
    DESTINATION "${destination}" COMPONENT ${component})
  if(EXISTS "${LLVM_LIBRARY_DIR}/libomp.so")
    install(FILES ${PROJECT_SOURCE_DIR}/packaging/licenses/OpenMP-LICENSE.TXT
      DESTINATION "${destination}" COMPONENT ${component})
  endif()
endfunction()
mdir_install_notices(share/mdir/licenses mdir)

# A relocatable release (D177): the shared libraries that the
# installed tree needs beyond the base system go into <prefix>/lib under
# their sonames, and libdevice of the toolkit into share/mdir/cuda, where
# the driver finds it when CUDA_ROOT is not set. cuFFT and libdevice are
# redistributable under Attachment A of the CUDA EULA, installed with it.
#
# The tree then also distributes HDF5 in binary form, so HDF5's notice
# joins the others in share/mdir/licenses.
option(MDIR_BUNDLE_LIBRARIES
  "Install HDF5, cuFFT, and libdevice with mdir, for a relocatable release" OFF)
# HDF5's notice, in the root or in share/ of its install tree.
function(mdir_install_hdf5_notice destination component)
  if(NOT MDIR_HDF5_LIBRARY)
    return()
  endif()
  get_filename_component(hdf5_root "${MDIR_HDF5_INCLUDE_DIR}" DIRECTORY)
  foreach(notice COPYING share/COPYING)
    if(EXISTS "${hdf5_root}/${notice}")
      install(FILES "${hdf5_root}/${notice}" DESTINATION "${destination}"
        RENAME HDF5-COPYING COMPONENT ${component})
      return()
    endif()
  endforeach()
  if(MDIR_BUNDLE_LIBRARIES)
    message(FATAL_ERROR "MDIR_BUNDLE_LIBRARIES bundles libhdf5, but HDF5's "
      "COPYING is neither in ${hdf5_root} nor in ${hdf5_root}/share")
  endif()
endfunction()
function(mdir_install_soname library)
  get_filename_component(real "${library}" REALPATH)
  get_filename_component(name "${real}" NAME)
  string(REGEX MATCH "^lib[^.]+\\.so\\.[0-9]+" soname "${name}")
  if(NOT soname)
    set(soname "${name}")
  endif()
  install(FILES "${real}" DESTINATION lib RENAME "${soname}" COMPONENT mdir)
endfunction()
if(MDIR_BUNDLE_LIBRARIES)
  if(MDIR_HDF5_LIBRARY)
    mdir_install_soname("${MDIR_HDF5_LIBRARY}")
  endif()
  mdir_install_hdf5_notice(share/mdir/licenses mdir)
  if(MDIR_CUDA_FOUND)
    set_target_properties(mdrt_cuda PROPERTIES INSTALL_RPATH "$ORIGIN")
    get_target_property(cufft CUDA::cufft IMPORTED_LOCATION)
    mdir_install_soname("${cufft}")
    if(MDIR_PME_CALLBACK)
      # cuFFT loads both JIT libraries from its own directory. NVRTC
      # loads the builtins with the toolkit major and minor in the name.
      get_filename_component(cuda_lib "${cufft}" DIRECTORY)
      mdir_install_soname("${cuda_lib}/libnvJitLink.so")
      mdir_install_soname("${cuda_lib}/libnvrtc.so")
      set(builtins "libnvrtc-builtins.so.${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")
      get_filename_component(real_builtins "${cuda_lib}/${builtins}" REALPATH)
      install(FILES "${real_builtins}" DESTINATION lib RENAME "${builtins}" COMPONENT mdir)
    endif()
    # cuBLAS and the cuBLASLt that it loads (D302).
    get_target_property(cublas CUDA::cublas IMPORTED_LOCATION)
    mdir_install_soname("${cublas}")
    get_target_property(cublaslt CUDA::cublasLt IMPORTED_LOCATION)
    mdir_install_soname("${cublaslt}")
    install(FILES "${CUDAToolkit_LIBRARY_ROOT}/nvvm/libdevice/libdevice.10.bc"
      DESTINATION share/mdir/cuda/nvvm/libdevice COMPONENT mdir)
    install(FILES "${CUDAToolkit_LIBRARY_ROOT}/EULA.txt"
      DESTINATION share/mdir/cuda COMPONENT mdir)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/version.txt"
      "CUDA Version ${CUDAToolkit_VERSION}\n")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.txt"
      DESTINATION share/mdir/cuda COMPONENT mdir)
  endif()
endif()

# A wheel of the Python package (pyproject.toml, scikit-build-core sets
# SKBUILD; D228): the component `python` carries the runtime
# in mdir/lib and libdevice with the CUDA EULA in mdir/cuda, where the
# extension looks beside itself. cuFFT and cuBLAS are not carried: the GPU
# runtime takes them from NVIDIA's wheels (nvidia-cufft, nvidia-cublas) in
# the same site-packages, else from the system's search path.
if(SKBUILD)
  install(TARGETS mdrt LIBRARY DESTINATION mdir/lib COMPONENT python)
  if(EXISTS "${LLVM_LIBRARY_DIR}/libomp.so")
    install(FILES "${LLVM_LIBRARY_DIR}/libomp.so" DESTINATION mdir/lib
      COMPONENT python)
  endif()
  if(MDIR_CUDA_FOUND)
    set_target_properties(mdrt_cuda PROPERTIES
      INSTALL_RPATH "$ORIGIN;$ORIGIN/../../nvidia/cu13/lib")
    install(TARGETS mdrt_cuda LIBRARY DESTINATION mdir/lib COMPONENT python)
    # The root of the toolkit: CMake before 3.27 may leave
    # CUDAToolkit_LIBRARY_ROOT empty; nvcc's bin is then beside nvvm.
    set(cuda_root "${CUDAToolkit_LIBRARY_ROOT}")
    if(NOT cuda_root)
      get_filename_component(cuda_root "${CUDAToolkit_BIN_DIR}" DIRECTORY)
    endif()
    install(FILES "${cuda_root}/nvvm/libdevice/libdevice.10.bc"
      DESTINATION mdir/cuda/nvvm/libdevice COMPONENT python)
    install(FILES "${cuda_root}/EULA.txt"
      DESTINATION mdir/cuda COMPONENT python)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/version.txt"
      "CUDA Version ${CUDAToolkit_VERSION}\n")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.txt"
      DESTINATION mdir/cuda COMPONENT python)
  endif()
  # pybind11 3.0.1, compiled into the extension (BSD-3-Clause, whose
  # notice goes with binaries); the copy is python/pybind11-LICENSE.
  install(FILES ${PROJECT_SOURCE_DIR}/python/pybind11-LICENSE
    DESTINATION mdir/licenses COMPONENT python)
  install(FILES ${PROJECT_SOURCE_DIR}/LICENSE DESTINATION mdir/licenses
    COMPONENT python)
  # HDF5's notice, which a binary distribution reproduces (auditwheel
  # grafts libhdf5 into mdir.libs), and those of pocketfft (in libmdrt),
  # toml++ and LLVM with MLIR (in the extension), and the OpenMP runtime
  # (mdir/lib/libomp.so).
  mdir_install_hdf5_notice(mdir/licenses python)
  mdir_install_notices(mdir/licenses python)
endif()

set(MDIR_CUDA_FOUND ${MDIR_CUDA_FOUND} PARENT_SCOPE)
set(MDIR_CUDA_ROOT "${CUDAToolkit_TARGET_DIR}" PARENT_SCOPE)

set(MDIR_CUDA_VERSION "${CUDAToolkit_VERSION}" PARENT_SCOPE)
