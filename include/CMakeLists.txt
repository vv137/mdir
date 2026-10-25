add_subdirectory(mdir/Dialect/MD)
add_subdirectory(mdir/Dialect/MD/Transforms)
add_subdirectory(mdir/Dialect/Dyn)
add_subdirectory(mdir/Dialect/MDRT)
add_subdirectory(mdir/Dialect/MDExec)
add_subdirectory(mdir/Dialect/MDDist)
add_subdirectory(mdir/Conversion)
add_subdirectory(mdir/Dialect/MDExec/Transforms)

# The reference of the dialects in docs/dialects/, from their TableGen
# definitions (#610): `ninja mdir-dialect-docs` writes it into the source
# tree, and test/Docs/dialect-docs.test fails when it is out of date.
if(TARGET ${MLIR_TABLEGEN_EXE})
  set(MDIR_DOC_TBLGEN $<TARGET_FILE:${MLIR_TABLEGEN_EXE}>)
else()
  find_program(MDIR_DOC_TBLGEN ${MLIR_TABLEGEN_EXE} HINTS ${LLVM_TOOLS_BINARY_DIR})
endif()
add_custom_target(mdir-dialect-docs
  COMMAND ${CMAKE_SOURCE_DIR}/scripts/generate-dialect-docs.sh
          ${MDIR_DOC_TBLGEN} ${MLIR_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/docs/dialects
  COMMENT "Writing docs/dialects/ from the TableGen definitions"
  VERBATIM)
