// The one lock of the process around the HDF5 library (#262). It is
// defined in a build without the library as well, where nothing takes it.

#include "mdir/Driver/HDF5.h"

std::mutex &mdir::driver::getHDF5Mutex() {
  static std::mutex mutex;
  return mutex;
}
