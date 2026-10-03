// What defined a run, which its checkpoint records and a run that takes the
// checkpoint compares (D[checkpoint-fingerprint]).
//
// See docs/driver-m0.md, Section 2.6.

#ifndef MDIR_DRIVER_FINGERPRINT_H
#define MDIR_DRIVER_FINGERPRINT_H

#include "mdir/Driver/Checkpoint.h"
#include "mdir/Driver/Control.h"
#include "mdir/Driver/System.h"

#include "llvm/Support/Error.h"

namespace mdir {
namespace driver {

/// The fingerprint of the run of `controlFile`:
///   - physics: the keys of [energy], [pme], [lj_pme], [constraints],
///     [restraints], [boundary], and [free_energy] as the file writes them,
///     and SHA-256 of the files of the topology, of the masses, and of the
///     reference of positional restraints;
///   - coupling: the keys of [dynamics] but `steps`, and of [ensemble],
///     [thermostat], and [barostat];
///   - execution: the keys of [execution].
llvm::Expected<Fingerprint> getRunFingerprint(llvm::StringRef controlFile,
                                              const Control &control,
                                              const System &system);

/// An entry that differs between two fingerprints. A side without it has
/// "(none)".
struct FingerprintChange {
  std::string group;
  std::string name;
  std::string before;
  std::string after;
};

/// The entries of `group` that differ between `stored` and `current`, in
/// the order of their names.
std::vector<FingerprintChange>
compareFingerprints(const Fingerprint &stored, const Fingerprint &current,
                    llvm::StringRef group);

} // namespace driver
} // namespace mdir

#endif // MDIR_DRIVER_FINGERPRINT_H
