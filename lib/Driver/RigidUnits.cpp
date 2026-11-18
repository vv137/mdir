//===- RigidUnits.cpp - The route of the rigid units -----------------------===//
//
// `[performance] rigid_units` (D[rigid-units-route]; D417, slice S4b): the
// choice of the kinds, the refusals of `ON`, the covering radii at the
// input, and the words of the records.
//
//===----------------------------------------------------------------------===//

#include "mdir/Driver/RigidUnits.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace mdir::driver;

namespace {
/// A displacement of positions.
struct Vec {
  double x = 0, y = 0, z = 0;
};

Vec getPosition(const System &system, unsigned particle) {
  const double *p = &system.positions[3 * size_t(particle)];
  return {p[0], p[1], p[2]};
}

/// The image of `d` nearest the origin for a short `d` (a bond): reduced
/// along c, b, and a in turn, as the reduced cell of docs/triclinic.md
/// allows.
Vec reduce(Vec d, const System &system, bool periodic) {
  if (!periodic)
    return d;
  const double ax = system.box[0], by = system.box[1], cz = system.box[2];
  const double bx = system.tilt[0], cx = system.tilt[1], cy = system.tilt[2];
  double n = std::nearbyint(d.z / cz);
  d.x -= n * cx, d.y -= n * cy, d.z -= n * cz;
  n = std::nearbyint(d.y / by);
  d.x -= n * bx, d.y -= n * by;
  n = std::nearbyint(d.x / ax);
  d.x -= n * ax;
  return d;
}

double norm(Vec v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

/// Half the least face width of the cell: the volume over the area of
/// each face.
double getHalfLeastWidth(const System &system) {
  const double ax = system.box[0], by = system.box[1], cz = system.box[2];
  const double bx = system.tilt[0], cx = system.tilt[1], cy = system.tilt[2];
  auto cross = [](Vec u, Vec v) {
    return Vec{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z,
               u.x * v.y - u.y * v.x};
  };
  Vec a{ax, 0, 0}, b{bx, by, 0}, c{cx, cy, cz};
  double volume = ax * by * cz;
  double least = std::min({volume / norm(cross(b, c)),
                           volume / norm(cross(c, a)),
                           volume / norm(cross(a, b))});
  return 0.5 * least;
}

/// The covering radius of a kind about its anchor, the first frame atom,
/// from the constrained distances of the frame, without the margin. A site
/// at sum_k w_k x_k is at |sum_k w_k h_k| with h_k = x_k - x_0, and
/// h_j . h_k = (d_0j^2 + d_0k^2 - d_jk^2) / 2.
double getConstrainedExtent(const RigidKind &kind) {
  unsigned m = kind.frame.size();
  auto distance = [&](unsigned a, unsigned b) -> double {
    if (a == b)
      return 0.0;
    if (a > b)
      std::swap(a, b);
    // The pairs (a, b), a < b, in lexicographic order.
    unsigned index = a * m - a * (a + 1) / 2 + (b - a - 1);
    return kind.distances[index];
  };
  auto dot = [&](unsigned j, unsigned k) {
    double dj = distance(0, j), dk = distance(0, k), djk = distance(j, k);
    return 0.5 * (dj * dj + dk * dk - djk * djk);
  };
  double largest = 0.0;
  for (unsigned k = 1; k < m; ++k)
    largest = std::max(largest, distance(0, k));
  for (const RigidKind::Site &site : kind.sites) {
    double square = 0.0;
    for (unsigned j = 1; j < m; ++j)
      for (unsigned k = 1; k < m; ++k)
        square += site.weights[j] * site.weights[k] * dot(j, k);
    largest = std::max(largest, std::sqrt(std::max(square, 0.0)));
  }
  return largest;
}

/// The largest distance of a place of a copy of `kind` from its anchor at
/// the positions of `system`: each frame atom in the image of its bond
/// nearest the anchor, each site placed by its weights.
double getMeasuredExtent(const RigidKind &kind, const System &system,
                         bool periodic) {
  double largest = 0.0;
  unsigned m = kind.frame.size();
  std::vector<Vec> offsets(m);
  for (size_t copy = 0, e = kind.getCount(); copy != e; ++copy) {
    const unsigned *members = &kind.members[copy * kind.size];
    Vec anchor = getPosition(system, members[kind.frame[0]]);
    for (unsigned k = 1; k < m; ++k) {
      Vec x = getPosition(system, members[kind.frame[k]]);
      offsets[k] = reduce({x.x - anchor.x, x.y - anchor.y, x.z - anchor.z},
                          system, periodic);
      largest = std::max(largest, norm(offsets[k]));
    }
    for (const RigidKind::Site &site : kind.sites) {
      Vec c;
      for (unsigned k = 1; k < m; ++k) {
        c.x += site.weights[k] * offsets[k].x;
        c.y += site.weights[k] * offsets[k].y;
        c.z += site.weights[k] * offsets[k].z;
      }
      largest = std::max(largest, norm(c));
    }
  }
  return largest;
}

/// The anchors of the rest (#799): a hydrogen whose bond to a heavy atom
/// the constraints keep, and the hydrogens of a rigid water that no taken
/// kind holds, belong to the unit of their heavy atom. Gives the longest
/// of those constrained lengths.
double findAnchored(const Topology &topology, const std::vector<bool> &inKind,
                    std::vector<std::array<unsigned, 2>> &anchored) {
  double longest = 0.0;
  auto anchor = [&](unsigned member, unsigned center, double length) {
    if (inKind[member] || inKind[center])
      return;
    anchored.push_back({member, center});
    longest = std::max(longest, length);
  };
  for (const Topology::Shake &shake : topology.shakes)
    for (size_t k = 0; k != shake.hydrogens.size(); ++k)
      anchor(shake.hydrogens[k], shake.center, shake.lengths[k]);
  for (const Topology::Settle &settle : topology.settles) {
    anchor(settle.oxygen + 1, settle.oxygen, settle.distanceOH);
    anchor(settle.oxygen + 2, settle.oxygen, settle.distanceOH);
  }
  return longest;
}

/// Why the program cannot take the units, or "" where it can.
std::string getProgramReason(const Control &control, const System &system) {
  if (control.target != Target::GPU)
    return "the CPU target: lists of units run on a GPU";
  if (control.precision != Precision::Mixed)
    return std::string("precision ") +
           (control.precision == Precision::Double ? "DOUBLE" : "SINGLE") +
           ": the units take MIXED";
  if (control.deterministic)
    return "the deterministic mode";
  if (control.minimize)
    return "a minimization";
  if (!control.periodic)
    return "a cell that is not periodic";
  if (!system.alchemical.empty() || control.hasFreeEnergy)
    return "[free_energy]";
  if (control.ljpme)
    return "lennard_jones = \"PME\"";
  if (control.pme && control.pmeKernel != PMEKernel::Gaussian)
    return "a kernel of PME other than GAUSSIAN";
  if (control.frames || control.frameProduct || control.tunableGradient ||
      control.dependentTerms || control.tunables ||
      !control.observedTunables.empty() ||
      llvm::any_of(control.observables, [](const Control::Observable &o) {
        return !o.constant.empty();
      }))
    return "frame, tunable, or derivative programs";
  return "";
}

/// The covering radius that the run takes: the nominal one, which holds
/// the margin, unless a place at the input lies beyond it; then the
/// measured distance plus the margin. A constrained input keeps the
/// nominal radius, and so the bytes of its program, whatever the rounding
/// of its positions within the margin.
double getRadius(double nominal, double measured) {
  return measured > nominal ? measured + rigidUnitsMargin : nominal;
}

std::string angstrom(double nm, const char *format = "%.4g") {
  return llvm::formatv("{0}", llvm::format(format, nm / units::length)).str();
}
} // namespace

llvm::Error mdir::driver::resolveRigidUnits(Control &control,
                                           const System &system) {
  RigidUnitsRoute route;
  route.resolved = true;
  auto refuse = [&](const llvm::Twine &reason) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "'rigid_units = \"ON\"' cannot apply: " + reason.str());
  };
  if (control.rigidUnits == RigidUnits::Off) {
    route.reason = control.rigidUnitsGiven ? "OFF was given" : "OFF";
    control.rigidUnitsRoute = std::move(route);
    return llvm::Error::success();
  }
  bool on = control.rigidUnits == RigidUnits::On;
  std::string program = getProgramReason(control, system);
  if (on && !program.empty())
    return refuse(program);
  // The kinds, which `mdir check` reports under AUTO as what ON would take.
  if (system.topology) {
    const Topology &topology = *system.topology;
    RigidGroups groups = findRigidGroups(topology);
    unsigned takenCount = 0;
    std::vector<bool> inKind(topology.getNumParticles(), false);
    // Two or more particles, by count, then by first particle: the order
    // of D403's kinds, with the kinds of one particle left out (the
    // maintainer's ruling on #915).
    for (auto [index, kind] : llvm::enumerate(groups.kinds)) {
      RigidUnitsRoute::Kind entry;
      entry.number = index + 1;
      entry.molecules = kind.getCount();
      entry.places = kind.size;
      entry.frame = kind.frame.size();
      entry.sites = kind.sites.size();
      auto plural = [](size_t n) { return n == 1 ? "" : "s"; };
      if (kind.size < 2)
        entry.reason = "one particle";
      else if (kind.incompleteExclusions)
        entry.reason = llvm::formatv("{0} molecule{1} with a pair of their "
                                     "own that is not excluded",
                                     kind.incompleteExclusions,
                                     plural(kind.incompleteExclusions));
      else if (kind.outsideExclusions)
        entry.reason = llvm::formatv("{0} molecule{1} with an excluded pair "
                                     "outside the molecule",
                                     kind.outsideExclusions,
                                     plural(kind.outsideExclusions));
      else if (takenCount == 2)
        entry.reason = "beyond the two most populous";
      else {
        entry.taken = true;
        ++takenCount;
        for (unsigned particle : kind.members)
          inKind[particle] = true;
        entry.constrained = getConstrainedExtent(kind) + rigidUnitsMargin;
        entry.measured = getMeasuredExtent(kind, system, control.periodic);
        entry.radius = getRadius(entry.constrained, entry.measured);
      }
      route.kinds.push_back(std::move(entry));
    }
    if (takenCount) {
      std::vector<std::array<unsigned, 2>> anchored;
      double longest = findAnchored(topology, inKind, anchored);
      route.restParticles = llvm::count(inKind, false);
      route.anchored = anchored.size();
      if (!anchored.empty()) {
        double measured = 0.0;
        for (auto [member, center] : anchored) {
          Vec x = getPosition(system, member), a = getPosition(system, center);
          measured = std::max(
              measured, norm(reduce({x.x - a.x, x.y - a.y, x.z - a.z}, system,
                                    control.periodic)));
        }
        route.restConstrained = longest + rigidUnitsMargin;
        route.restMeasured = measured;
        route.restRadius = getRadius(route.restConstrained, measured);
      }
      double widest = route.restRadius;
      for (const RigidUnitsRoute::Kind &kind : route.kinds)
        widest = std::max(widest, kind.radius);
      route.reach = control.pairlistDistance * units::length + 2.0 * widest;
      if (control.periodic)
        route.halfWidth = getHalfLeastWidth(system);
    }
    if (on) {
      if (route.kinds.empty())
        return refuse("no rigid kind: the system has no rigid group (D403)");
      if (!takenCount) {
        std::string reasons;
        for (const RigidUnitsRoute::Kind &kind : route.kinds)
          reasons += llvm::formatv("{0}kind {1}: {2}",
                                   reasons.empty() ? "" : "; ", kind.number,
                                   kind.reason)
                         .str();
        return refuse("no kind is eligible (" + reasons + ")");
      }
      if (control.periodic && route.reach > route.halfWidth) {
        double widest = route.restRadius;
        for (const RigidUnitsRoute::Kind &kind : route.kinds)
          widest = std::max(widest, kind.radius);
        return refuse("the covering reach, " + angstrom(route.reach) +
                      " Ang (the pair list distance plus twice " +
                      angstrom(widest) +
                      " Ang, the largest covering radius of the units), "
                      "exceeds half the least face width of the cell, " +
                      angstrom(route.halfWidth) + " Ang");
      }
    }
  } else if (on) {
    return refuse("a system without a topology");
  }
  if (on) {
    route.units = true;
    route.reason = "ON was given";
  } else {
    route.reason = "AUTO takes the general route until its rule is measured";
  }
  control.rigidUnitsRoute = std::move(route);
  return llvm::Error::success();
}

RigidUnitsPlan mdir::driver::getRigidUnitsPlan(const RigidUnitsRoute &route,
                                               const Topology &topology) {
  RigidUnitsPlan plan;
  plan.groups = findRigidGroups(topology);
  std::vector<bool> inKind(topology.getNumParticles(), false);
  for (const RigidUnitsRoute::Kind &kind : route.kinds) {
    if (!kind.taken)
      continue;
    unsigned index = kind.number - 1;
    plan.taken.push_back(index);
    plan.radii.push_back(kind.radius);
    for (unsigned particle : plan.groups.kinds[index].members)
      inKind[particle] = true;
  }
  findAnchored(topology, inKind, plan.anchored);
  plan.restRadius = plan.anchored.empty() ? 0.0 : route.restRadius;
  return plan;
}

std::string mdir::driver::describeRigidUnitsRoute(const Control &control) {
  const RigidUnitsRoute &route = control.rigidUnitsRoute;
  return llvm::formatv("{0}: {1} ({2})", getRigidUnitsName(control.rigidUnits),
                       route.units ? "the units" : "general", route.reason)
      .str();
}

std::vector<std::string>
mdir::driver::describeRigidUnitsKinds(const Control &control) {
  const RigidUnitsRoute &route = control.rigidUnitsRoute;
  std::vector<std::string> lines;
  auto plural = [](size_t n) { return n == 1 ? "" : "s"; };
  auto radius = [&](double radius, double constrained, double measured) {
    std::string text = "radius " + angstrom(radius, "%.4f") + " Ang";
    if (radius > constrained)
      text += " (" + angstrom(constrained, "%.4f") +
              " constrained, widened by " +
              angstrom(radius - constrained, "%.2g") +
              " Ang to the input's " + angstrom(measured, "%.4f") + ")";
    return text;
  };
  bool any = false;
  for (const RigidUnitsRoute::Kind &kind : route.kinds) {
    std::string head = llvm::formatv("kind {0}:", kind.number).str();
    head.resize(std::max<size_t>(head.size() + 1, 10), ' ');
    if (!kind.taken) {
      lines.push_back(head + "not taken: " + kind.reason);
      continue;
    }
    any = true;
    lines.push_back(
        head +
        llvm::formatv("{0}{1} units of {2} place{3} ({4} in the frame, {5} "
                      "site{6}), ",
                      route.units ? "taken, " : "ON would take: ",
                      kind.molecules, kind.places, plural(kind.places),
                      kind.frame, kind.sites, plural(kind.sites))
            .str() +
        radius(kind.radius, kind.constrained, kind.measured));
  }
  if (any) {
    std::string rest = llvm::formatv("the rest: {0} particle{1}, {2} anchored",
                                     route.restParticles,
                                     plural(route.restParticles),
                                     route.anchored)
                           .str();
    if (route.anchored)
      rest += ", " + radius(route.restRadius, route.restConstrained,
                            route.restMeasured);
    lines.push_back(rest);
    if (route.halfWidth > 0.0)
      lines.push_back("reach:    " + angstrom(route.reach) +
                      " Ang, half the least face width " +
                      angstrom(route.halfWidth) + " Ang");
  }
  return lines;
}

llvm::json::Object mdir::driver::getRigidUnitsJSON(const Control &control) {
  using llvm::json::Array;
  using llvm::json::Object;
  const RigidUnitsRoute &route = control.rigidUnitsRoute;
  auto a = [](double nm) { return nm / units::length; };
  Array kinds;
  for (const RigidUnitsRoute::Kind &kind : route.kinds) {
    Object entry{{"kind", kind.number},
                 {"taken", route.units && kind.taken},
                 {"taken_by_on", kind.taken},
                 {"reason", kind.reason},
                 {"units", static_cast<int64_t>(kind.molecules)},
                 {"places", kind.places},
                 {"frame", kind.frame},
                 {"sites", kind.sites}};
    if (kind.taken) {
      entry["radius_angstrom"] = a(kind.radius);
      entry["constrained_radius_angstrom"] = a(kind.constrained);
      entry["input_distance_angstrom"] = a(kind.measured);
      entry["widened_angstrom"] = a(kind.radius - kind.constrained);
    }
    kinds.push_back(std::move(entry));
  }
  Object result{{"requested", getRigidUnitsName(control.rigidUnits)},
                {"route", route.units ? "units" : "general"},
                {"reason", route.reason},
                {"kinds", std::move(kinds)}};
  if (llvm::any_of(route.kinds, [](const auto &k) { return k.taken; })) {
    result["rest"] = Object{
        {"particles", static_cast<int64_t>(route.restParticles)},
        {"anchored", static_cast<int64_t>(route.anchored)},
        {"radius_angstrom", a(route.restRadius)},
        {"constrained_radius_angstrom", a(route.restConstrained)},
        {"input_distance_angstrom", a(route.restMeasured)},
        {"widened_angstrom", a(route.restRadius - route.restConstrained)}};
    result["reach_angstrom"] = a(route.reach);
    result["half_least_face_width_angstrom"] =
        route.halfWidth > 0.0 ? llvm::json::Value(a(route.halfWidth))
                              : llvm::json::Value(nullptr);
  }
  return result;
}
