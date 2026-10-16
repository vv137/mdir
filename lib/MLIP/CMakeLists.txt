# The plugin of learned potentials (D270,
# docs/metatomic-ml1.md, Section 6.1): C++20 with RTTI and exceptions, which
# the headers of libtorch need, linking libtorch, metatomic-torch, and
# metatensor-torch. Their CMake packages come from the Python environment
# that the plugin is built against: CMAKE_PREFIX_PATH lists
# torch.utils.cmake_prefix_path, metatomic.torch.utils.cmake_prefix_path,
# metatensor.torch.utils.cmake_prefix_path, and
# metatensor.utils.cmake_prefix_path. The RUNPATH names their libraries, so
# that `mdir run` loads the libtorch of the build; a Python process that has
# imported torch already has it (Section 6.2).
find_package(Torch REQUIRED CONFIG)
find_package(metatensor_torch REQUIRED CONFIG)
find_package(metatomic_torch REQUIRED CONFIG)
message(STATUS "metatomic plugin: torch ${Torch_VERSION}, "
               "metatomic-torch ${metatomic_torch_VERSION}")

add_library(mdir_metatomic SHARED MetatomicPlugin.cpp)
set_target_properties(mdir_metatomic PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  INSTALL_RPATH_USE_LINK_PATH ON
  )
# LLVM's options turn RTTI and exceptions off for the whole tree; the
# plugin needs both, and catches every exception at the C boundary.
target_compile_options(mdir_metatomic PRIVATE -frtti -fexceptions)
target_link_libraries(mdir_metatomic PRIVATE
  ${TORCH_LIBRARIES} metatomic_torch metatensor_torch)
install(TARGETS mdir_metatomic LIBRARY DESTINATION lib COMPONENT mdir)
