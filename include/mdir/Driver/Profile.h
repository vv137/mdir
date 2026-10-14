// Marks of the time that the host spends outside the code of a program: in
// an update of the tunables and at the start of an activation (#324). For
// the measurements and the tests; nothing reads them to decide anything.

#ifndef MDIR_DRIVER_PROFILE_H
#define MDIR_DRIVER_PROFILE_H

#include <cstdint>
#include <map>
#include <string>

#include <sys/resource.h>
#include <time.h>

namespace mdir {
namespace driver {

/// What the marks of one owner (a simulation) have summed: for each name
/// the seconds, the pages of memory that were faulted in (a block that the C
/// library maps anew is paid for there, #294), and the number of marks.
struct HostProfile {
  struct Slice {
    double seconds = 0.0;
    int64_t faults = 0, calls = 0;
  };
  std::map<std::string, Slice> slices;

  /// Makes `profile` the one that the marks of this thread add to, until it
  /// is destroyed.
  class Scope {
  public:
    explicit Scope(HostProfile &profile);
    ~Scope();
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

  private:
    HostProfile *before;
  };
};

/// The profile that the marks of this thread add to; none outside a scope.
inline thread_local HostProfile *hostProfile = nullptr;

inline HostProfile::Scope::Scope(HostProfile &profile) : before(hostProfile) {
  hostProfile = &profile;
}
inline HostProfile::Scope::~Scope() { hostProfile = before; }

/// A mark: from its construction to `end` or its destruction. Marks nest;
/// each name sums its own time, so that of an outer mark holds the inner.
class HostMark {
public:
  explicit HostMark(const char *name) : name(name) {
    if (!hostProfile)
      return;
    start = now();
    faults = getFaults();
  }
  ~HostMark() { end(); }
  HostMark(const HostMark &) = delete;
  HostMark &operator=(const HostMark &) = delete;

  void end() {
    if (!hostProfile || !name)
      return;
    HostProfile::Slice &slice = hostProfile->slices[name];
    slice.seconds += now() - start;
    slice.faults += getFaults() - faults;
    slice.calls += 1;
    name = nullptr;
  }
  /// Ends the mark and begins one named `next`.
  void next(const char *next) {
    end();
    name = next;
    if (!hostProfile)
      return;
    start = now();
    faults = getFaults();
  }

private:
  static double now() {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return static_cast<double>(time.tv_sec) +
           1.0e-9 * static_cast<double>(time.tv_nsec);
  }
  static int64_t getFaults() {
    struct rusage usage;
    getrusage(RUSAGE_THREAD, &usage);
    return usage.ru_minflt;
  }
  const char *name;
  double start = 0.0;
  int64_t faults = 0;
};

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_PROFILE_H
