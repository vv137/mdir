// The elements of a topology as `mdir check` reports them, where each comes
// from, and the warnings where other evidence disagrees with them (#469,
// D[gromacs-hmr-elements]). A diagnosis: no run reads it.

#ifndef MDIR_TOOLS_MDIR_ELEMENTS_H
#define MDIR_TOOLS_MDIR_ELEMENTS_H

#include "mdir/Driver/Control.h"
#include "mdir/Driver/Topology.h"

#include "llvm/Support/JSON.h"

#include <string>
#include <utility>
#include <vector>

namespace mdir {
namespace tool {

struct ElementReport {
  /// The particles by the element that the run takes them for and by
  /// where it comes from. `number` is the atomic number, 0 for a particle
  /// without an element (a virtual site), and -1 for a particle without
  /// one that the run does not take for a hydrogen.
  struct Count {
    int number;
    driver::Topology::ElementSource source;
    size_t count;
  };
  std::vector<Count> counts;
  /// The particles by source: the file, the name and the mass, the mass,
  /// none.
  size_t sources[4] = {0, 0, 0, 0};
  /// The particles whose element comes from a fallback, not the file.
  size_t getFallbacks() const { return sources[1] + sources[2] + sources[3]; }
  /// The constraints by kind: the bonds of hydrogen that SHAKE keeps, in
  /// their groups, and the rigid waters of SETTLE.
  size_t hydrogenBonds = 0, hydrogenGroups = 0, rigidWaters = 0;
  /// The warnings: a code and a message.
  std::vector<std::pair<std::string, std::string>> warnings;
};

/// The report of the elements of `topology` as the system was read for
/// `control`.
ElementReport diagnoseElements(const driver::Control &control,
                               const driver::Topology &topology);

/// "H", "C", ... for an atomic number, "?" for -1, "none" for 0.
std::string getElementSymbol(int number);

/// The key of a source in the JSON report.
const char *getElementSourceKey(driver::Topology::ElementSource source);

/// The two lines of the text report, without their labels.
std::string describeElementCounts(const ElementReport &report);
std::string describeElementSources(const ElementReport &report);
std::string describeConstraintKinds(const ElementReport &report);

llvm::json::Object makeElementsJSON(const ElementReport &report);

} // namespace tool
} // namespace mdir

#endif
