add_subdirectory(mdir)
# mdir-opt links the test passes of test/lib (MDIRTestPasses).
if(MDIR_BUILD_TESTS)
  add_subdirectory(mdir-opt)
endif()
