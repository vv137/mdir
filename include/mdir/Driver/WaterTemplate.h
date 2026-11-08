// Topology proof for the rigid-water relation (D[rigid-water-template]).
#ifndef MDIR_DRIVER_WATERTEMPLATE_H
#define MDIR_DRIVER_WATERTEMPLATE_H

#include "mdir/Driver/System.h"
#include <array>
#include <map>

namespace mdir::driver {

/// Constants shared by every instance of one template. Lengths are in nm,
/// charges in e, and the oxygen LJ parameters in nm and kJ/mol. Type identity
/// is retained for the general solute--water mixing table.
struct WaterTemplateModel {
  unsigned sites = 0;
  std::array<double, 4> charges{};
  std::array<unsigned, 4> types{};
  double distanceOH = 0.0, distanceHH = 0.0, distanceOS = 0.0;
  double alpha = 0.0, sigma = 0.0, epsilon = 0.0;
  double extent = 0.0;
};

struct WaterTemplatePlan {
  WaterTemplateModel model;
  /// O, H1, H2, and (for four sites) the stored extra point. Three-site
  /// instances use their oxygen in the unused fourth slot.
  std::vector<std::array<unsigned, 4>> instances;
  std::vector<unsigned> solute;
  /// Counts of water molecules, not particles. Ties for the predominant
  /// model are resolved by the first oxygen in topology order.
  std::map<std::string, size_t> mismatches;
  size_t candidates = 0;

  /// WW, SW and SS are ranges in one buffer, not independent lists.
  /// Refresh a segment when either participating mobility group needs it:
  /// bit 0 is anchors and bit 1 atoms. Thus adding atom classes does not
  /// add a displacement test (P(P+1)/2 segments, C <= P tests).
  static constexpr std::array<unsigned, 3> mobilityMasks = {1, 3, 2};
};

/// Match an already validated, prepared topology. This proves only the
/// topology; execution-mode selection and the measured AUTO floor are
/// separate. It does not change the topology, parameters or ordinary IR.
llvm::Expected<WaterTemplatePlan> matchWaterTemplate(const Control &control,
                                                    const System &system);

/// Execution modes whose parameters or derivative contracts require the
/// ordinary path. Empty means that topology matching may be attempted.
std::string getWaterTemplateModeReason(const Control &control);

struct WaterTemplateSelection {
  WaterTemplatePlan plan;
  bool enabled = false;
  std::string reason;
};

/// The AUTO floor is supplied by the end-to-end measurement record; ON
/// bypasses that floor and the initial four-site-only AUTO policy.
llvm::Expected<WaterTemplateSelection>
selectWaterTemplate(const Control &control, const System &system,
                    size_t measuredParticleFloor);

} // namespace mdir::driver
#endif
