find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
if(Python_VERSION VERSION_GREATER_EQUAL 3.14)
  message(FATAL_ERROR "The M2 Python interface supports Python 3.10 through 3.13")
endif()
# Host arrays have one required dependency and one validated interchange path.
execute_process(COMMAND "${Python_EXECUTABLE}" -c
  "import numpy as np; assert np.lib.NumpyVersion(np.__version__) >= '1.23.0', np.__version__"
  RESULT_VARIABLE MDIR_NUMPY_STATUS OUTPUT_VARIABLE MDIR_NUMPY_OUTPUT
  ERROR_VARIABLE MDIR_NUMPY_ERROR)
if(NOT MDIR_NUMPY_STATUS STREQUAL "0")
  message(FATAL_ERROR
    "The MDIR Python interface requires NumPy >=1.23 with ${Python_EXECUTABLE}.\n${MDIR_NUMPY_ERROR}")
endif()
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 3.0.1 EXACT CONFIG REQUIRED)

# The package `mdir`: `mdir/__init__.py` and the extension `mdir._core`, in
# <build>/python/mdir as in site-packages, so that PYTHONPATH=<build>/python
# imports the build (D228).
set(MDIR_PYTHON_PACKAGE_DIR ${CMAKE_BINARY_DIR}/python/mdir)
pybind11_add_module(mdir_python NO_EXTRAS Bindings.cpp)
set_target_properties(mdir_python PROPERTIES OUTPUT_NAME _core
  LIBRARY_OUTPUT_DIRECTORY ${MDIR_PYTHON_PACKAGE_DIR})
target_compile_options(mdir_python PRIVATE -fexceptions)
target_compile_definitions(mdir_python PRIVATE MDIR_VERSION="${PROJECT_VERSION}")
target_link_libraries(mdir_python PRIVATE MDIRCompiler MDIRSimulation)
# The extension exports its module init alone: the symbols of the static
# libraries are not exported (--exclude-libs), so that --gc-sections removes
# the sections nothing reaches (#655). Compiled code finds the functions of
# the driver through the symbols that Simulation.cpp gives the engine.
target_link_options(mdir_python PRIVATE LINKER:--exclude-libs,ALL LINKER:--gc-sections)
configure_file(mdir/__init__.py ${MDIR_PYTHON_PACKAGE_DIR}/__init__.py COPYONLY)
# The frame evaluator and its PyTorch adapter (docs/python-frames.md);
# `mdir.torch` imports torch and is imported only by a script that asks.
configure_file(mdir/_frames.py ${MDIR_PYTHON_PACKAGE_DIR}/_frames.py COPYONLY)
configure_file(mdir/torch.py ${MDIR_PYTHON_PACKAGE_DIR}/torch.py COPYONLY)

if(SKBUILD)
  # A wheel (pyproject.toml): the package is the root of the wheel's
  # platlib, with the runtime in mdir/lib (runtime/CMakeLists.txt). The
  # extension also takes HDF5 from the directory where the build found it,
  # so that a wheel that auditwheel has not repaired imports without
  # LD_LIBRARY_PATH (#634). auditwheel's repair grafts libhdf5 into
  # mdir.libs and keeps only the entries of the RPATH that lie in the wheel
  # (append_rpath_within_wheel), so a repaired wheel does not carry this
  # directory; check-wheel.sh refuses an entry outside the wheel.
  set(MDIR_PYTHON_INSTALL_DIR mdir)
  set(MDIR_PYTHON_RPATH "$ORIGIN/lib")
  get_property(MDIR_PYTHON_HDF5 GLOBAL PROPERTY MDIR_HDF5_FOUND)
  if(MDIR_PYTHON_HDF5)
    get_filename_component(MDIR_PYTHON_HDF5_DIR "${MDIR_HDF5_LIBRARY}" DIRECTORY)
    list(APPEND MDIR_PYTHON_RPATH "${MDIR_PYTHON_HDF5_DIR}")
  endif()
  set_target_properties(mdir_python PROPERTIES INSTALL_RPATH "${MDIR_PYTHON_RPATH}")
else()
  set(MDIR_PYTHON_INSTALL_DIR
    "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/mdir"
    CACHE STRING "Python package install destination")
endif()
install(TARGETS mdir_python LIBRARY DESTINATION ${MDIR_PYTHON_INSTALL_DIR}
  COMPONENT python)
install(FILES mdir/__init__.py mdir/_frames.py mdir/torch.py
  DESTINATION ${MDIR_PYTHON_INSTALL_DIR} COMPONENT python)
# What a wheel builds (pyproject.toml): the extension and the runtime.
add_custom_target(mdir_wheel)
add_dependencies(mdir_wheel mdir_python mdrt)
if(TARGET mdrt_cuda)
  add_dependencies(mdir_wheel mdrt_cuda)
endif()
set(MDIR_PYTHON_EXECUTABLE "${Python_EXECUTABLE}" PARENT_SCOPE)
