# The floating-point std::from_chars of C++17 (<charconv>, std::chars_format),
# which the reader of GROMACS topologies uses (lib/Driver/Gromacs.cpp).
# libstdc++ has it from GCC 11; an older one, such as that of GCC 9, fails
# deep in the build with "'std::chars_format' has not been declared", so it
# is refused here, at configure (#654). The probe tests the feature, not a
# version number, so that any standard library that has it passes.
include(CheckCXXSourceCompiles)

function(mdir_check_float_from_chars)
  check_cxx_source_compiles([=[
#include <charconv>
int main() {
  const char text[] = "1.5e-3";
  double value = 0.0;
  auto result = std::from_chars(text, text + sizeof(text) - 1, value,
                                std::chars_format::general);
  return result.ec == std::errc() ? 0 : 1;
}
]=] MDIR_HAVE_FLOAT_FROM_CHARS)
  if(NOT MDIR_HAVE_FLOAT_FROM_CHARS)
    message(FATAL_ERROR
      "MDIR needs a C++ standard library with the floating-point "
      "std::from_chars of C++17 (std::chars_format), which the compiler "
      "${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} "
      "${CMAKE_CXX_COMPILER_VERSION}) does not have: use GCC 11 or later, or "
      "a Clang whose libc++ or libstdc++ has it (docs/building.md, "
      "Requirements). Set CMAKE_CXX_COMPILER, or CC and CXX, in a new build "
      "tree: CMake keeps the compiler of an existing one.")
  endif()
endfunction()
