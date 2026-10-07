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
# imports the build (D[python-package]).
set(MDIR_PYTHON_PACKAGE_DIR ${CMAKE_BINARY_DIR}/python/mdir)
pybind11_add_module(mdir_python NO_EXTRAS Bindings.cpp)
set_target_properties(mdir_python PROPERTIES OUTPUT_NAME _core
  LIBRARY_OUTPUT_DIRECTORY ${MDIR_PYTHON_PACKAGE_DIR})
target_compile_options(mdir_python PRIVATE -fexceptions)
target_compile_definitions(mdir_python PRIVATE MDIR_VERSION="${PROJECT_VERSION}")
target_link_libraries(mdir_python PRIVATE MDIRCompiler MDIRSimulation)
configure_file(mdir/__init__.py ${MDIR_PYTHON_PACKAGE_DIR}/__init__.py COPYONLY)

if(SKBUILD)
  # A wheel (pyproject.toml): the package is the root of the wheel's
  # platlib, with the runtime in mdir/lib (runtime/CMakeLists.txt), where
  # the extension also finds HDF5 until auditwheel moves it to mdir.libs.
  set(MDIR_PYTHON_INSTALL_DIR mdir)
  set_target_properties(mdir_python PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
else()
  set(MDIR_PYTHON_INSTALL_DIR
    "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/mdir"
    CACHE STRING "Python package install destination")
endif()
install(TARGETS mdir_python LIBRARY DESTINATION ${MDIR_PYTHON_INSTALL_DIR}
  COMPONENT python)
install(FILES mdir/__init__.py DESTINATION ${MDIR_PYTHON_INSTALL_DIR}
  COMPONENT python)
set(MDIR_PYTHON_EXECUTABLE "${Python_EXECUTABLE}" PARENT_SCOPE)
