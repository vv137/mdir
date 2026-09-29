add_subdirectory(lib/Dialect/MD)

# For the site configuration of lit.
if(MDIR_CUDA_FOUND)
  set(MDIR_TEST_CUDA 1)
else()
  set(MDIR_TEST_CUDA 0)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(MDIR_TEST_DEPENDS
  mdir-opt
  mdrt
  )
if(MDIR_CUDA_FOUND)
  list(APPEND MDIR_TEST_DEPENDS mdrt_cuda)
endif()

add_lit_testsuite(check-mdir "Running the MDIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MDIR_TEST_DEPENDS}
  )
set_target_properties(check-mdir PROPERTIES FOLDER "Tests")
