# The template is code that the compiler adds to the modules it lowers. It is
# kept as a file of its own and compiled into the library as a string.
set(TEMPLATE_FILE
  ${PROJECT_SOURCE_DIR}/lib/Runtime/Templates/NeighborsMatrix.mlir)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${TEMPLATE_FILE})
file(READ ${TEMPLATE_FILE} NEIGHBORS_MATRIX_TEMPLATE)
set(PME_TEMPLATE_FILE ${PROJECT_SOURCE_DIR}/lib/Runtime/Templates/PME.mlir)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${PME_TEMPLATE_FILE})
file(READ ${PME_TEMPLATE_FILE} PME_TEMPLATE)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Templates.cpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/Templates.cpp
  @ONLY)

add_mlir_conversion_library(MDIRMDExecToLoops
  CPUVectorPairs.cpp
  FixedOrderReductions.cpp
  HoistAllocas.cpp
  MDExecToLoops.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/Templates.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Conversion

  DEPENDS
  MDIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MDIRMD
  MDIRMDExecKernels
  MDIRMDExec
  MDIRMDRT
  MLIRControlFlowDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRVectorDialect
  )
