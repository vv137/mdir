// The one lock of the process around the HDF5 library (#262).

#ifndef MDIR_DRIVER_HDF5_H
#define MDIR_DRIVER_HDF5_H

#include <mutex>

namespace mdir {
namespace driver {

/// The mutex that every use of the HDF5 library in MDIR holds, from its
/// first call to the release of its last identifier: the writers and the
/// readers of checkpoints (`writeCheckpoint`, `readCheckpoint`) and of
/// trajectories in H5MD (`H5MDWriter`, `H5MDReader`). The library is
/// thread-safe only if it was built so (`H5is_library_threadsafe`), which
/// MDIR does not ask of a build, and a simulation writes a checkpoint
/// outside the mutex of the runs and without the GIL: two threads would
/// otherwise be inside the library at once. It is not recursive: no code
/// that holds it calls another that takes it. Code that holds it takes
/// neither the mutex of the runs nor the GIL (a frame is written inside a
/// part, with the mutex of the runs held: the order is that one, then this).
std::mutex &getHDF5Mutex();

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_HDF5_H
