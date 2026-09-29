add_subdirectory(lib/Dialect/MD)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(MDIR_TEST_DEPENDS
  mdir-opt
  )

add_lit_testsuite(check-mdir "Running the MDIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MDIR_TEST_DEPENDS}
  )
set_target_properties(check-mdir PROPERTIES FOLDER "Tests")
