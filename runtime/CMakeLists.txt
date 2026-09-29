# The runtime is plain C, so that it builds without the C++ standard library
# and loads into any host process.
add_library(mdrt SHARED mdrt.c)
set_target_properties(mdrt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  C_VISIBILITY_PRESET default
  )
