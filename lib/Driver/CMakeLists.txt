add_mlir_library(MDIRDriver
  Builder.cpp
  Control.cpp
  Expression.cpp
  Output.cpp
  System.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Driver

  LINK_COMPONENTS
  Support
  )
target_include_directories(MDIRDriver SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/tomlplusplus)
