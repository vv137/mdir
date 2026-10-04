// Validation shared by file and object front ends (D[python-model]).
#include "mdir/Driver/Topology.h"
#include "mdir/Driver/Cell.h"
#include "llvm/ADT/STLExtras.h"
#include <cmath>
#include <set>
using namespace mdir::driver;

llvm::Error mdir::driver::validateTopology(const Topology &t) {
  auto fail = [](const llvm::Twine &message) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "topology: " + message);
  };
  size_t n = t.getNumParticles(), nt = t.getNumTypes();
  if (!n || !nt)
    return fail("expected particles and Lennard-Jones types");
  for (auto [name, size] : {
           std::pair<const char *, size_t>{"atom names", t.atomNames.size()},
           {"atomic numbers", t.atomicNumbers.size()}, {"charges", t.charges.size()},
           {"types", t.types.size()}, {"positions", t.positions.size() / 3},
           {"residue identities", t.residueOf.size()}})
    if (size != n)
      return fail(llvm::Twine(name) + " do not match the particle count");
  if (t.positions.size() != 3 * n ||
      (!t.velocities.empty() && t.velocities.size() != 3 * n))
    return fail("positions and optional velocities must have shape (N, 3)");
  if (t.sigma.size() != nt * nt || t.epsilon.size() != nt * nt)
    return fail("Lennard-Jones tables must have shape (types, types)");
  for (const auto *values : {&t.masses, &t.charges, &t.positions, &t.velocities,
                             &t.sigma, &t.epsilon, &t.bornRadii, &t.bornScreens})
    if (llvm::any_of(*values, [](double v) { return !std::isfinite(v); }))
      return fail("particle arrays and parameters must be finite");
  for (double m : t.masses)
    if (m < 0.0)
      return fail("masses must be nonnegative");
  for (size_t a = 0; a != nt; ++a)
    for (size_t b = 0; b != nt; ++b)
      if (t.sigma[a*nt+b] < 0 || t.epsilon[a*nt+b] < 0 ||
          t.sigma[a*nt+b] != t.sigma[b*nt+a] ||
          t.epsilon[a*nt+b] != t.epsilon[b*nt+a])
        return fail("Lennard-Jones tables must be nonnegative and symmetric");
  for (unsigned type : t.types)
    if (type >= nt)
      return fail("a particle type is out of range");
  if (t.residueNames.empty() || t.residueStarts.size() != t.residueNames.size() ||
      t.residueStarts.front() != 0)
    return fail("residue names and starts must cover all particles");
  for (size_t k = 0; k != t.residueStarts.size(); ++k) {
    unsigned begin = t.residueStarts[k];
    unsigned end = k+1 < t.residueStarts.size() ? t.residueStarts[k+1] : n;
    if (begin >= end || end > n)
      return fail("residue starts must be increasing and in range");
    for (unsigned i = begin; i != end; ++i)
      if (t.residueOf[i] != k)
        return fail("residue identities disagree with residue starts");
  }
  auto ids = [&](std::initializer_list<unsigned> members) {
    std::set<unsigned> unique;
    for (unsigned i : members)
      if (i >= n || !unique.insert(i).second)
        return false;
    return true;
  };
  auto positive = [](double v) { return std::isfinite(v) && v > 0; };
  auto nonnegative = [](double v) { return std::isfinite(v) && v >= 0; };
  for (const auto &b : t.bonds)
    if (!ids({b.i,b.j}) || !positive(b.r0) || !nonnegative(b.k))
      return fail("invalid bond identities or parameters");
  for (const auto &a : t.angles)
    if (!ids({a.i,a.j,a.k}) || !nonnegative(a.force) ||
        !std::isfinite(a.theta0) || a.theta0 < 0 || a.theta0 > M_PI)
      return fail("invalid angle identities or parameters");
  for (const auto &d : t.dihedrals)
    if (!ids({d.i,d.j,d.k,d.l}) || !std::isfinite(d.force) ||
        !std::isfinite(d.phase) || d.n < 0)
      return fail("invalid dihedral identities or parameters");
  for (const auto &u : t.ureyBradleys)
    if (!ids({u.i,u.k}) || !positive(u.r0) || !nonnegative(u.force))
      return fail("invalid Urey-Bradley identities or parameters");
  for (const auto &h : t.harmonicImpropers)
    if (!ids({h.i,h.j,h.k,h.l}) || !nonnegative(h.force) || !std::isfinite(h.xi0))
      return fail("invalid harmonic improper identities or parameters");
  for (const auto &p : t.pairs)
    if (!ids({p.i,p.j}) || !nonnegative(p.sigma) || !nonnegative(p.epsilon) ||
        !nonnegative(p.scaleLJ) || !nonnegative(p.scaleCoulomb))
      return fail("invalid special pair identities or parameters");
  for (auto [i,j] : t.exclusions)
    if (!ids({i,j}) || i >= j)
      return fail("exclusions must have ordered, distinct particle identities");
  if (!std::is_sorted(t.exclusions.begin(), t.exclusions.end()) ||
      std::adjacent_find(t.exclusions.begin(), t.exclusions.end()) != t.exclusions.end())
    return fail("exclusions must be sorted and unique");
  for (const auto &c : t.cmaps)
    if (!ids({c.i,c.j,c.k,c.l,c.m}) || c.map >= t.cmapGrids.size())
      return fail("invalid CMAP particle or map identity");
  for (const auto &grid : t.cmapGrids)
    if (t.cmapResolution < 2 || grid.size() != size_t(t.cmapResolution)*t.cmapResolution ||
        llvm::any_of(grid, [](double v) { return !std::isfinite(v); }))
      return fail("CMAP grids must be finite square arrays matching their resolution");
  std::set<unsigned> constrained;
  for (const auto &s : t.settles) {
    if (s.oxygen >= n || n-s.oxygen < 3 || !positive(s.distanceOH) ||
        !positive(s.distanceHH) || s.distanceHH >= 2*s.distanceOH)
      return fail("invalid SETTLE particles or distances");
    for (unsigned i = s.oxygen; i != s.oxygen+3; ++i)
      if (!positive(t.masses[i]) || !constrained.insert(i).second)
        return fail("SETTLE groups must have positive masses and be disjoint");
  }
  for (const auto &s : t.shakes) {
    if (s.center >= n || s.hydrogens.empty() || s.hydrogens.size() > 3 ||
        s.lengths.size() != s.hydrogens.size() || !positive(t.masses[s.center]) ||
        !constrained.insert(s.center).second)
      return fail("invalid SHAKE group");
    for (size_t k = 0; k != s.hydrogens.size(); ++k) {
      unsigned i = s.hydrogens[k];
      if (i >= n || !positive(t.masses[i]) || !positive(s.lengths[k]) ||
          !constrained.insert(i).second)
        return fail("SHAKE groups must have positive masses/distances and be disjoint");
    }
  }
  std::set<unsigned> sites;
  for (const auto &v : t.virtualSites) {
    if (!ids({v.site,v.i,v.j,v.k}) || t.masses[v.site] != 0 ||
        !std::isfinite(v.a) || !std::isfinite(v.b) || !sites.insert(v.site).second ||
        (v.kind != Topology::VirtualSite::AmberWater && v.kind != Topology::VirtualSite::Linear))
      return fail("invalid virtual site identities or parameters");
  }
  for (const auto &v : t.virtualSites)
    if (sites.count(v.i) || sites.count(v.j) || sites.count(v.k))
      return fail("virtual-site parents must be massive particles");
  for (size_t i = 0; i != n; ++i)
    if (t.masses[i] == 0 && !sites.count(i))
      return fail("a zero-mass particle must be a virtual site");
  for (int k = 0; k != 3; ++k)
    if (!std::isfinite(t.box[k]) || !std::isfinite(t.tilt[k]))
      return fail("cell entries must be finite");
  return llvm::Error::success();
}
