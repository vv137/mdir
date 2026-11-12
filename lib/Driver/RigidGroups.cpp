// The rigid groups of a topology and their kinds (D403), from
// the constraints and the site definitions alone: no atomic number, name,
// or order of elements is read.

#include "mdir/Driver/RigidGroups.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <tuple>

using namespace mdir::driver;

const char *mdir::driver::getRigidReasonText(RigidReason reason) {
  switch (reason) {
  case RigidReason::FlexibleFrame:
    return "a pair of massive atoms without a constrained distance";
  case RigidReason::SiteNotConstantWeight:
    return "a site that is no constant-weight combination of the frame";
  }
  return "";
}

const char *mdir::driver::getRigidReasonKey(RigidReason reason) {
  switch (reason) {
  case RigidReason::FlexibleFrame:
    return "flexible_frame";
  case RigidReason::SiteNotConstantWeight:
    return "site_not_constant_weight";
  }
  return "";
}

std::optional<std::array<double, 3>>
mdir::driver::getBisectorWeights(double a, double ij, double ik, double jk) {
  // |u + v|^2 = 2 + 2 u.v, and the law of cosines gives
  // 2 ij ik u.v = ij^2 + ik^2 - jk^2, so
  // ij ik |u + v|^2 = (ij + ik)^2 - jk^2 = (ij + ik - jk) (ij + ik + jk).
  // The product form loses no digits where jk is near ij + ik.
  if (!(ij > 0.0 && ik > 0.0 && jk >= 0.0) || !std::isfinite(a) ||
      !std::isfinite(ij) || !std::isfinite(ik) || !std::isfinite(jk))
    return std::nullopt;
  double q = (ij + ik - jk) * (ij + ik + jk);
  if (!(q > 0.0))
    return std::nullopt;
  double root = std::sqrt(q);
  double wj = a * std::sqrt(ik / ij) / root;
  double wk = a * std::sqrt(ij / ik) / root;
  if (!std::isfinite(wj) || !std::isfinite(wk))
    return std::nullopt;
  return std::array<double, 3>{1.0 - wj - wk, wj, wk};
}

namespace {
struct Components {
  std::vector<unsigned> parent;
  explicit Components(size_t n) : parent(n) {
    std::iota(parent.begin(), parent.end(), 0u);
  }
  unsigned root(unsigned i) {
    while (i != parent[i]) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  }
  /// The root of a component is its least particle.
  void join(unsigned a, unsigned b) {
    a = root(a);
    b = root(b);
    if (a != b)
      parent[std::max(a, b)] = std::min(a, b);
  }
};

/// A constrained distance, `i < j`.
struct Edge {
  unsigned i, j;
  double length;
  bool operator<(const Edge &other) const {
    return std::tie(i, j) < std::tie(other.i, other.j);
  }
};

/// What makes two rigid groups copies of one kind (R1), compared as
/// numbers: see the note on the tolerance in docs/constraints.md.
struct Key {
  std::vector<unsigned> integers;
  std::vector<double> reals;
  bool operator<(const Key &other) const {
    return std::tie(integers, reals) < std::tie(other.integers, other.reals);
  }
};
} // namespace

RigidGroups mdir::driver::findRigidGroups(const Topology &t) {
  RigidGroups result;
  const size_t n = t.getNumParticles();
  if (n == 0)
    return result;

  // The constrained distances, whatever record holds them.
  std::vector<Edge> edges;
  auto constrain = [&](unsigned a, unsigned b, double length) {
    edges.push_back({std::min(a, b), std::max(a, b), length});
  };
  for (const Topology::Settle &s : t.settles) {
    constrain(s.oxygen, s.oxygen + 1, s.distanceOH);
    constrain(s.oxygen, s.oxygen + 2, s.distanceOH);
    constrain(s.oxygen + 1, s.oxygen + 2, s.distanceHH);
  }
  for (const Topology::Shake &s : t.shakes)
    for (size_t k = 0; k != s.hydrogens.size(); ++k)
      constrain(s.center, s.hydrogens[k], s.lengths[k]);
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end(),
                          [](const Edge &a, const Edge &b) {
                            return a.i == b.i && a.j == b.j;
                          }),
              edges.end());

  // The molecules.
  Components components(n);
  for (const Topology::Bond &b : t.bonds)
    components.join(b.i, b.j);
  for (const Edge &e : edges)
    components.join(e.i, e.j);
  std::vector<const Topology::VirtualSite *> definition(n, nullptr);
  for (const Topology::VirtualSite &s : t.virtualSites) {
    components.join(s.site, s.i);
    components.join(s.site, s.j);
    components.join(s.site, s.k);
    definition[s.site] = &s;
  }

  // Per molecule, at its root: its particles, its massive atoms, its
  // constrained pairs, and its excluded pairs.
  std::vector<unsigned> size(n, 0), massive(n, 0), constrained(n, 0),
      excluded(n, 0);
  std::vector<bool> outside(n, false);
  for (unsigned i = 0; i != n; ++i) {
    unsigned r = components.root(i);
    ++size[r];
    massive[r] += t.masses[i] != 0.0;
  }
  for (const Edge &e : edges)
    ++constrained[components.root(e.i)];
  for (auto [i, j] : t.exclusions) {
    unsigned ri = components.root(i), rj = components.root(j);
    if (ri == rj)
      ++excluded[ri];
    else
      outside[ri] = outside[rj] = true;
  }
  // The particles of each molecule, ascending, in one array.
  std::vector<size_t> start(n + 1, 0);
  for (unsigned i = 0; i != n; ++i)
    start[components.root(i) + 1] += 1;
  for (size_t r = 0; r != n; ++r)
    start[r + 1] += start[r];
  std::vector<unsigned> particles(n);
  {
    std::vector<size_t> next(start.begin(), start.end() - 1);
    for (unsigned i = 0; i != n; ++i)
      particles[next[components.root(i)]++] = i;
  }

  auto distance = [&](unsigned a, unsigned b) -> const double * {
    auto found = std::lower_bound(edges.begin(), edges.end(),
                                  Edge{std::min(a, b), std::max(a, b), 0.0});
    if (found == edges.end() || found->i != std::min(a, b) ||
        found->j != std::max(a, b))
      return nullptr;
    return &found->length;
  };

  std::map<Key, size_t> kindOf;
  for (unsigned r = 0; r != n; ++r) {
    if (components.root(r) != r)
      continue;
    ++result.molecules;
    auto leaveOut = [&](RigidReason reason) {
      ++result.leftOut[static_cast<unsigned>(reason)];
    };
    // R1: every pair of the frame has a constrained distance. Constraints
    // join massive atoms of one molecule, and each pair is counted once,
    // so the count decides it without visiting the pairs of a large
    // flexible molecule.
    size_t m = massive[r];
    if (constrained[r] != m * (m - 1) / 2) {
      leaveOut(RigidReason::FlexibleFrame);
      continue;
    }
    const unsigned *members = &particles[start[r]];
    RigidKind kind;
    kind.size = size[r];
    std::vector<unsigned> frameAtoms;
    for (unsigned p = 0; p != kind.size; ++p) {
      unsigned i = members[p];
      kind.charges.push_back(t.charges[i]);
      kind.types.push_back(t.types[i]);
      if (t.masses[i] != 0.0) {
        kind.frame.push_back(p);
        frameAtoms.push_back(i);
      }
    }
    bool complete = true;
    for (size_t a = 0; a != m && complete; ++a)
      for (size_t b = a + 1; b != m && complete; ++b) {
        const double *length = distance(frameAtoms[a], frameAtoms[b]);
        if (length)
          kind.distances.push_back(*length);
        else
          complete = false;
      }
    if (!complete) {
      leaveOut(RigidReason::FlexibleFrame);
      continue;
    }
    // R2: the weights of each site over the frame.
    auto placeInFrame = [&](unsigned atom) {
      return static_cast<size_t>(
          std::lower_bound(frameAtoms.begin(), frameAtoms.end(), atom) -
          frameAtoms.begin());
    };
    bool constant = true;
    for (unsigned p = 0; p != kind.size && constant; ++p) {
      unsigned i = members[p];
      if (t.masses[i] != 0.0)
        continue;
      const Topology::VirtualSite *site = definition[i];
      std::optional<std::array<double, 3>> weights;
      if (site && site->kind == Topology::VirtualSite::Linear) {
        weights = std::array<double, 3>{1.0 - site->a - site->b, site->a,
                                        site->b};
      } else if (site && site->kind == Topology::VirtualSite::AmberWater) {
        const double *ij = distance(site->i, site->j);
        const double *ik = distance(site->i, site->k);
        const double *jk = distance(site->j, site->k);
        if (ij && ik && jk)
          weights = getBisectorWeights(site->a, *ij, *ik, *jk);
      }
      // A form that this rule does not know is not assumed to be one.
      if (!weights) {
        constant = false;
        break;
      }
      RigidKind::Site placed;
      placed.place = p;
      placed.weights.assign(m, 0.0);
      const unsigned atoms[3] = {site->i, site->j, site->k};
      for (unsigned k = 0; k != 3; ++k)
        placed.weights[placeInFrame(atoms[k])] += (*weights)[k];
      kind.sites.push_back(std::move(placed));
    }
    if (!constant) {
      leaveOut(RigidReason::SiteNotConstantWeight);
      continue;
    }

    Key key;
    key.integers.push_back(kind.size);
    key.integers.insert(key.integers.end(), kind.frame.begin(),
                        kind.frame.end());
    key.integers.insert(key.integers.end(), kind.types.begin(),
                        kind.types.end());
    key.reals = kind.charges;
    key.reals.insert(key.reals.end(), kind.distances.begin(),
                     kind.distances.end());
    for (const RigidKind::Site &site : kind.sites)
      key.reals.insert(key.reals.end(), site.weights.begin(),
                       site.weights.end());
    auto [entry, inserted] = kindOf.insert({std::move(key), result.kinds.size()});
    if (inserted)
      result.kinds.push_back(std::move(kind));
    RigidKind &copies = result.kinds[entry->second];
    copies.members.insert(copies.members.end(), members,
                          members + copies.size);
    size_t pairs = size_t(copies.size) * (copies.size - 1) / 2;
    copies.scattered += members[copies.size - 1] - members[0] + 1 != copies.size;
    copies.incompleteExclusions += excluded[r] != pairs;
    copies.outsideExclusions += outside[r];
  }
  // Molecules were visited in the order of their first particle, so a
  // stable sort keeps that order among kinds of equal count.
  std::stable_sort(result.kinds.begin(), result.kinds.end(),
                   [](const RigidKind &a, const RigidKind &b) {
                     return a.getCount() > b.getCount();
                   });
  return result;
}
