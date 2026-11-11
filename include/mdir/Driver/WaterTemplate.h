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

  /// Solute units (#799): each solute particle that a constrained bond
  /// keeps near a heavy atom is listed as (member, anchor); every other
  /// solute particle anchors itself. The units partition the solute.
  std::vector<std::array<unsigned, 2>> anchored;
  /// The longest constrained member-anchor length, in nm.
  double anchorLength = 0.0;
  /// Solute hydrogens with no constrained bond to an anchor. Any makes
  /// the template inapplicable: the run takes the general path.
  size_t unconstrainedHydrogens = 0;

  /// The class-segmented pair list holds WW, SW and SS in one buffer.
  /// One mobility group, the anchors of the waters and of the solute
  /// units, refreshes every segment: P(P+1)/2 segments, one test.
  static constexpr std::array<unsigned, 3> mobilityMasks = {1, 1, 1};
};

/// The radius of the solute units: the longest constrained length with
/// the constraints' relative tolerance, rho = d (1 + tol), which their
/// failure test enforces after every position constraint. Zero without
/// anchored members.
double getAnchorRadius(const WaterTemplatePlan &, bool doublePrecision);

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

/// The least of the three face widths (the distances between opposite
/// faces) of the cell with the diagonal `box` and the tilts `tilt`.
double getLeastFaceWidth(const double box[3], const double tilt[3]);

/// The AUTO floor is supplied by the end-to-end measurement record; ON
/// bypasses that floor and the initial four-site-only AUTO policy.
llvm::Expected<WaterTemplateSelection>
selectWaterTemplate(const Control &control, const System &system,
                    size_t measuredParticleFloor);

} // namespace mdir::driver
#endif
