// The rigid groups of made topologies (D403): the two rules on
// molecules of several kinds, one case for each reason that leaves a
// molecule out, the facts of a kind, and the weights of the sites against
// the placement of docs/constraints.md, Section 2, in f64.
//
// Usage: mdir-rigid-groups-test
#include "mdir/Driver/RigidGroups.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>

using namespace mdir::driver;

#define REQUIRE(condition)                                                     \
  do {                                                                         \
    if (!(condition)) {                                                        \
      llvm::errs() << __FILE__ << ":" << __LINE__                              \
                   << ": failed: " #condition "\n";                            \
      std::exit(1);                                                            \
    }                                                                          \
  } while (false)

namespace {
using Vec = std::array<double, 3>;
constexpr double Epsilon = std::numeric_limits<double>::epsilon();

/// A topology built a molecule at a time. Atomic numbers and names are
/// given on purpose values that tell nothing: the rules read neither.
struct Made {
  Topology t;
  Made() {
    t.typeNames = {"A", "B", "C", "D"};
    t.sigma.assign(16, 0.0);
    t.epsilon.assign(16, 0.0);
    // Every type but C has Lennard-Jones with itself.
    for (unsigned a : {0u, 1u, 3u}) {
      t.sigma[a * 4 + a] = 0.3;
      t.epsilon[a * 4 + a] = 0.5;
    }
  }
  unsigned add(double mass, double charge, unsigned type) {
    unsigned i = t.getNumParticles();
    t.atomNames.push_back("X");
    t.atomicNumbers.push_back(79);
    t.masses.push_back(mass);
    t.charges.push_back(charge);
    t.types.push_back(type);
    t.residueOf.push_back(0);
    t.positions.insert(t.positions.end(), {0.0, 0.0, 0.0});
    return i;
  }
  void exclude(std::initializer_list<unsigned> members) {
    for (unsigned a : members)
      for (unsigned b : members)
        if (a < b)
          t.exclusions.emplace_back(a, b);
  }
  /// Three massive atoms that one triple of constraints holds.
  unsigned triple(double q0 = -0.8, double q = 0.4, double first = 0.1,
                  double across = 0.16, unsigned type = 1) {
    unsigned o = add(16.0, q0, 0);
    add(1.0, q, type);
    add(1.0, q, type);
    t.settles.push_back({o, first, across});
    exclude({o, o + 1, o + 2});
    return o;
  }
  /// A triple with a massless site of the bisector form, or of the linear
  /// one with the same weights.
  unsigned withSite(bool linear, double a = 0.015, bool swapped = false) {
    unsigned o = add(16.0, 0.0, 0);
    add(1.0, 0.5, 1);
    add(1.0, 0.5, 1);
    unsigned s = add(0.0, -1.0, 2);
    t.settles.push_back({o, 0.1, 0.16});
    unsigned j = swapped ? o + 2 : o + 1, k = swapped ? o + 1 : o + 2;
    if (linear) {
      // The weights of the bisector site as numbers, or one unit in the
      // last place away.
      double w = (*getBisectorWeights(0.015, 0.1, 0.1, 0.16))[1];
      if (a != 0.015)
        w = std::nextafter(w, 1.0);
      t.virtualSites.push_back({Topology::VirtualSite::Linear, s, o, j, k, w, w});
    } else {
      t.virtualSites.push_back(
          {Topology::VirtualSite::AmberWater, s, o, j, k, a, 0.0});
    }
    exclude({o, o + 1, o + 2, s});
    return o;
  }
  /// A center and `count` lighter atoms, their bonds constrained or not.
  unsigned star(unsigned count, bool constrained) {
    unsigned c = add(12.0, -0.1 * count, 3);
    Topology::Shake group{c, {}, {}};
    for (unsigned k = 0; k != count; ++k) {
      unsigned h = add(1.0, 0.1, 3);
      if (constrained) {
        group.hydrogens.push_back(h);
        group.lengths.push_back(0.109);
      } else {
        t.bonds.push_back({c, h, 1000.0, 0.109, true});
      }
    }
    if (constrained)
      t.shakes.push_back(group);
    for (unsigned a = c; a != c + count + 1; ++a)
      for (unsigned b = a + 1; b != c + count + 1; ++b)
        t.exclusions.emplace_back(a, b);
    return c;
  }
  Topology &finish(bool validate = true) {
    t.residueNames = {"R"};
    t.residueStarts = {0};
    std::sort(t.exclusions.begin(), t.exclusions.end());
    if (validate)
      if (llvm::Error error = validateTopology(t)) {
        llvm::errs() << llvm::toString(std::move(error)) << "\n";
        std::exit(1);
      }
    return t;
  }
};

size_t leftOut(const RigidGroups &groups, RigidReason reason) {
  return groups.leftOut[static_cast<unsigned>(reason)];
}

double sum(const std::vector<double> &values) {
  double total = 0.0;
  for (double v : values)
    total += v;
  return total;
}

//===----------------------------------------------------------------------===//
// R1 and R2 on molecules of several kinds
//===----------------------------------------------------------------------===//

void testKinds() {
  Made made;
  // Three triples, two triples with a bisector site, one with the linear
  // site at the same place, an atom alone, a constrained pair, and a
  // triple with Lennard-Jones on every atom.
  unsigned three[3];
  for (unsigned &o : three)
    o = made.triple();
  unsigned four[2] = {made.withSite(false), made.withSite(false)};
  unsigned linear = made.withSite(true);
  unsigned atom = made.add(39.0, 1.0, 3);
  unsigned pair = made.star(1, true);
  unsigned allLJ = made.triple(-0.8, 0.4, 0.1, 0.16, /*type=*/3);
  RigidGroups groups = findRigidGroups(made.finish());

  REQUIRE(groups.molecules == 9);
  REQUIRE(leftOut(groups, RigidReason::FlexibleFrame) == 0);
  REQUIRE(leftOut(groups, RigidReason::SiteNotConstantWeight) == 0);
  // The bisector site and the linear site of the same weights are one
  // kind (R2): three copies, as many as the plain triple, which comes
  // first by its first particle.
  REQUIRE(groups.kinds.size() == 5);
  const RigidKind &k3 = groups.kinds[0], &k4 = groups.kinds[1];
  REQUIRE(k3.getCount() == 3 && k3.size == 3 && k3.sites.empty());
  REQUIRE(k3.members ==
          (std::vector<unsigned>{three[0], three[0] + 1, three[0] + 2,
                                 three[1], three[1] + 1, three[1] + 2,
                                 three[2], three[2] + 1, three[2] + 2}));
  REQUIRE(k3.frame == (std::vector<unsigned>{0, 1, 2}));
  REQUIRE(k3.charges == (std::vector<double>{-0.8, 0.4, 0.4}));
  REQUIRE(k3.types == (std::vector<unsigned>{0, 1, 1}));
  REQUIRE(k3.distances == (std::vector<double>{0.1, 0.1, 0.16}));
  REQUIRE(k4.getCount() == 3 && k4.size == 4 && k4.sites.size() == 1);
  REQUIRE(k4.members.front() == four[0] && k4.members[4] == four[1] &&
          k4.members[8] == linear);
  REQUIRE(k4.frame == (std::vector<unsigned>{0, 1, 2}));
  REQUIRE(k4.sites[0].place == 3 && k4.sites[0].weights.size() == 3);
  // The weight of each of the two outer atoms, in the form of an
  // isosceles frame: a over twice the height of the triangle.
  double height = std::sqrt(0.1 * 0.1 - 0.25 * 0.16 * 0.16);
  double expected = 0.015 / (2.0 * height);
  REQUIRE(std::fabs(k4.sites[0].weights[1] - expected) <= 4 * Epsilon);
  REQUIRE(std::fabs(k4.sites[0].weights[2] - expected) <= 4 * Epsilon);
  REQUIRE(std::fabs(sum(k4.sites[0].weights) - 1.0) <= 2 * Epsilon);
  // The kinds of one copy, in the order of their first particle.
  REQUIRE(groups.kinds[2].size == 1 && groups.kinds[2].members[0] == atom);
  REQUIRE(groups.kinds[2].distances.empty());
  REQUIRE(groups.kinds[3].size == 2 && groups.kinds[3].members[0] == pair);
  REQUIRE(groups.kinds[3].distances == (std::vector<double>{0.109}));
  REQUIRE(groups.kinds[4].size == 3 && groups.kinds[4].members[0] == allLJ);
  REQUIRE(groups.kinds[4].types == (std::vector<unsigned>{0, 3, 3}));
  for (const RigidKind &kind : groups.kinds)
    REQUIRE(!kind.scattered && !kind.incompleteExclusions &&
            !kind.outsideExclusions);
  llvm::outs() << "kinds: 9 molecules in 5 kinds, none left out\n";
}

/// What makes two molecules one kind is compared as numbers: a charge
/// one unit in the last place away, another type, another length, or
/// another weight is another kind; the order of a site's atoms is not.
void testSameness() {
  auto count = [](Made &made) {
    return findRigidGroups(made.finish()).kinds.size();
  };
  {
    Made made;
    made.triple();
    made.triple();
    REQUIRE(count(made) == 1);
  }
  {
    Made made;
    made.triple();
    made.triple(std::nextafter(-0.8, 0.0));
    REQUIRE(count(made) == 2);
  }
  {
    Made made;
    made.triple();
    made.triple(-0.8, 0.4, 0.1, 0.16, /*type=*/3);
    REQUIRE(count(made) == 2);
  }
  {
    Made made;
    made.triple();
    made.triple(-0.8, 0.4, 0.1, std::nextafter(0.16, 1.0));
    REQUIRE(count(made) == 2);
  }
  {
    Made made;
    made.withSite(false);
    made.withSite(false, std::nextafter(0.015, 1.0));
    REQUIRE(count(made) == 2);
  }
  {
    Made made;
    made.withSite(false);
    made.withSite(false, 0.015, /*swapped=*/true);
    REQUIRE(count(made) == 1);
  }
  {
    // Two forms of one site are one kind where their weights are the same
    // numbers, and two where they differ in the last place.
    Made made;
    made.withSite(false);
    made.withSite(true);
    REQUIRE(count(made) == 1);
  }
  {
    Made made;
    made.withSite(false);
    made.withSite(true, 0.0151);
    REQUIRE(count(made) == 2);
  }
  {
    // A triple with a site and one without are two kinds, whatever else.
    Made made;
    made.triple(0.0, 0.5);
    made.withSite(false);
    REQUIRE(count(made) == 2);
  }
  llvm::outs() << "sameness: equal numbers, no tolerance\n";
}

//===----------------------------------------------------------------------===//
// The reasons that leave a molecule out, one case each
//===----------------------------------------------------------------------===//

void testLeftOut() {
  {
    // No constraint at all: three atoms with their bonds.
    Made made;
    unsigned o = made.add(16.0, -0.8, 0);
    made.add(1.0, 0.4, 1);
    made.add(1.0, 0.4, 1);
    made.t.bonds.push_back({o, o + 1, 1000.0, 0.1, true});
    made.t.bonds.push_back({o, o + 2, 1000.0, 0.1, true});
    RigidGroups groups = findRigidGroups(made.finish());
    REQUIRE(groups.molecules == 1 && groups.kinds.empty());
    REQUIRE(leftOut(groups, RigidReason::FlexibleFrame) == 1);
    REQUIRE(leftOut(groups, RigidReason::SiteNotConstantWeight) == 0);
  }
  for (unsigned outer : {2u, 3u}) {
    // Constrained in part: a center and two or three atoms whose bonds
    // are held, 2 of 3 and 3 of 6 pairs. The angles between them are free.
    Made made;
    made.star(outer, true);
    RigidGroups groups = findRigidGroups(made.finish());
    REQUIRE(groups.molecules == 1 && groups.kinds.empty());
    REQUIRE(leftOut(groups, RigidReason::FlexibleFrame) == 1);
  }
  {
    // A rigid triple bonded to one more atom: the frame is the four.
    Made made;
    unsigned o = made.triple();
    unsigned x = made.add(12.0, 0.0, 3);
    made.t.bonds.push_back({o, x, 1000.0, 0.15, false});
    RigidGroups groups = findRigidGroups(made.finish());
    REQUIRE(groups.molecules == 1 && groups.kinds.empty());
    REQUIRE(leftOut(groups, RigidReason::FlexibleFrame) == 1);
  }
  {
    // A site on a flexible frame: the frame's reason, not the site's.
    Made made;
    unsigned o = made.add(16.0, 0.0, 0);
    made.add(1.0, 0.5, 1);
    made.add(1.0, 0.5, 1);
    unsigned s = made.add(0.0, -1.0, 2);
    made.t.bonds.push_back({o, o + 1, 1000.0, 0.1, true});
    made.t.bonds.push_back({o, o + 2, 1000.0, 0.1, true});
    made.t.virtualSites.push_back(
        {Topology::VirtualSite::Linear, s, o, o + 1, o + 2, 0.1, 0.1});
    RigidGroups groups = findRigidGroups(made.finish());
    REQUIRE(groups.molecules == 1 && groups.kinds.empty());
    REQUIRE(leftOut(groups, RigidReason::FlexibleFrame) == 1);
    REQUIRE(leftOut(groups, RigidReason::SiteNotConstantWeight) == 0);
  }
  {
    // A site whose form is not one of the constant-weight forms: a rigid
    // triple with two sites of a kind that the topology does not have
    // yet, as the sites out of the plane of a molecule of five would be.
    // validateTopology refuses the kind, so the topology is not validated.
    Made made;
    unsigned o = made.triple(0.0, 0.25);
    for (unsigned k = 0; k != 2; ++k) {
      unsigned s = made.add(0.0, -0.25, 2);
      made.t.virtualSites.push_back({static_cast<Topology::VirtualSite::Kind>(7),
                                     s, o, o + 1, o + 2, 0.1, 0.1});
    }
    made.triple();
    RigidGroups groups = findRigidGroups(made.finish(/*validate=*/false));
    REQUIRE(groups.molecules == 2 && groups.kinds.size() == 1);
    REQUIRE(groups.kinds[0].size == 3 && groups.kinds[0].getCount() == 1);
    REQUIRE(leftOut(groups, RigidReason::SiteNotConstantWeight) == 1);
    REQUIRE(leftOut(groups, RigidReason::FlexibleFrame) == 0);
  }
  {
    // The bisector form without a direction: the two outer atoms opposite
    // each other, 0.2 = 0.1 + 0.1. validateTopology refuses the lengths.
    Made made;
    unsigned o = made.withSite(false);
    made.t.settles.back().distanceHH = 0.2;
    (void)o;
    RigidGroups groups = findRigidGroups(made.finish(/*validate=*/false));
    REQUIRE(groups.molecules == 1 && groups.kinds.empty());
    REQUIRE(leftOut(groups, RigidReason::SiteNotConstantWeight) == 1);
  }
  llvm::outs() << "left out: flexible frame, site not of constant weights\n";
}

/// The facts of a kind are counts of its copies; none of them leaves a
/// molecule out or makes another kind.
void testFacts() {
  Made made;
  unsigned a = made.triple();
  unsigned b = made.triple();
  unsigned c = made.triple();
  // The site of a fourth molecule comes after another molecule.
  unsigned o = made.add(16.0, 0.0, 0);
  made.add(1.0, 0.5, 1);
  made.add(1.0, 0.5, 1);
  made.t.settles.push_back({o, 0.1, 0.16});
  unsigned d = made.triple();
  unsigned s = made.add(0.0, -1.0, 2);
  made.t.virtualSites.push_back(
      {Topology::VirtualSite::AmberWater, s, o, o + 1, o + 2, 0.015, 0.0});
  made.exclude({o, o + 1, o + 2, s});
  // b lacks one of its own pairs; c has a pair with a particle of d.
  auto &exclusions = made.t.exclusions;
  exclusions.erase(std::find(exclusions.begin(), exclusions.end(),
                             std::pair<unsigned, unsigned>{b + 1, b + 2}));
  exclusions.emplace_back(c, d + 1);
  RigidGroups groups = findRigidGroups(made.finish());
  REQUIRE(groups.molecules == 5 && groups.kinds.size() == 2);
  const RigidKind &plain = groups.kinds[0], &sited = groups.kinds[1];
  REQUIRE(plain.getCount() == 4 && plain.members.front() == a);
  REQUIRE(plain.scattered == 0);
  REQUIRE(plain.incompleteExclusions == 1);
  REQUIRE(plain.outsideExclusions == 2);
  REQUIRE(sited.getCount() == 1 && sited.scattered == 1);
  REQUIRE(sited.members == (std::vector<unsigned>{o, o + 1, o + 2, s}));
  REQUIRE(sited.incompleteExclusions == 0 && sited.outsideExclusions == 0);
  llvm::outs() << "facts: scattered, incomplete exclusions, outside "
                  "exclusions counted\n";
}

//===----------------------------------------------------------------------===//
// The weights against the placement, in f64
//===----------------------------------------------------------------------===//

Vec combine(const std::vector<double> &weights, const Vec *atoms) {
  Vec r{0.0, 0.0, 0.0};
  for (size_t k = 0; k != weights.size(); ++k)
    for (int c = 0; c != 3; ++c)
      r[c] += weights[k] * atoms[k][c];
  return r;
}

/// The placement of the extra point of Amber as the engine forms it
/// (docs/constraints.md, Section 2; SiteIR.cpp): x_i + a (u + v) / |u + v|.
/// `norm` receives |u + v|.
Vec placeBisector(const Vec &xi, const Vec &xj, const Vec &xk, double a,
                  double &norm) {
  Vec toJ, toK, total;
  double lengthJ = 0.0, lengthK = 0.0;
  for (int c = 0; c != 3; ++c) {
    toJ[c] = xj[c] - xi[c];
    toK[c] = xk[c] - xi[c];
    lengthJ += toJ[c] * toJ[c];
    lengthK += toK[c] * toK[c];
  }
  lengthJ = std::sqrt(lengthJ);
  lengthK = std::sqrt(lengthK);
  norm = 0.0;
  for (int c = 0; c != 3; ++c) {
    total[c] = toJ[c] / lengthJ + toK[c] / lengthK;
    norm += total[c] * total[c];
  }
  norm = std::sqrt(norm);
  Vec r;
  for (int c = 0; c != 3; ++c)
    r[c] = xi[c] + a / norm * total[c];
  return r;
}

double largest(const Vec *atoms, size_t count) {
  double most = 0.0;
  for (size_t k = 0; k != count; ++k)
    for (double x : atoms[k])
      most = std::max(most, std::fabs(x));
  return most;
}

double absoluteSum(const std::vector<double> &values) {
  double total = 0.0;
  for (double v : values)
    total += std::fabs(v);
  return total;
}

/// The distance of two points, rounded once from extended precision.
double measure(const Vec &a, const Vec &b) {
  long double total = 0.0L;
  for (int c = 0; c != 3; ++c) {
    long double d = static_cast<long double>(a[c]) - b[c];
    total += d * d;
  }
  return static_cast<double>(std::sqrt(total));
}

/// getBisectorWeights on frames of three different lengths, which no
/// record of main gives yet: any three points, the three distances
/// measured from them, so that the frame holds those distances to a
/// rounding. Each coordinate of sum_k w_k x_k must equal the placement
/// within
///   eps (4 (1 + sum |w_k|) X + 64 |a| / |u + v|),
/// X the largest coordinate: the first term bounds the rounding of the two
/// sums over coordinates of size X, the second that of the weights (the
/// rounded distances, two square roots, and three divisions each) and of
/// the unit vectors, which act on the offset a / |u + v| of the site along
/// each bond. The largest error over bound is printed: a bound that the
/// errors come nowhere near would hide a bug.
void testBisectorWeights() {
  std::mt19937_64 random(20261112);
  std::uniform_real_distribution<double> coordinate(-5.0, 5.0),
      offset(-0.2, 0.2), length(-0.1, 0.1);
  double worst = 0.0;
  unsigned cases = 0;
  while (cases != 20000) {
    Vec atoms[3];
    for (double &x : atoms[0])
      x = coordinate(random);
    for (int k = 1; k != 3; ++k)
      for (int c = 0; c != 3; ++c)
        atoms[k][c] = atoms[0][c] + offset(random);
    double a = length(random);
    double ij = measure(atoms[0], atoms[1]), ik = measure(atoms[0], atoms[2]),
           jk = measure(atoms[1], atoms[2]);
    double norm;
    Vec placed = placeBisector(atoms[0], atoms[1], atoms[2], a, norm);
    // Away from the frames without a direction, where the bound grows.
    if (norm < 0.05 || ij < 0.01 || ik < 0.01)
      continue;
    auto weights = getBisectorWeights(a, ij, ik, jk);
    REQUIRE(weights);
    std::vector<double> w(weights->begin(), weights->end());
    REQUIRE(std::fabs(sum(w) - 1.0) <= 2 * Epsilon * absoluteSum(w));
    Vec combined = combine(w, atoms);
    double bound = Epsilon * (4.0 * (1.0 + absoluteSum(w)) * largest(atoms, 3) +
                              64.0 * std::fabs(a) / norm);
    for (int c = 0; c != 3; ++c) {
      double error = std::fabs(combined[c] - placed[c]);
      REQUIRE(error <= bound);
      worst = std::max(worst, error / bound);
    }
    ++cases;
  }
  REQUIRE(worst > 0.01);
  llvm::outs() << "bisector weights: 20000 frames of three lengths within "
                  "the bound, largest error "
               << llvm::format("%.2f", worst) << " of it\n";
  // No direction, and lengths that are none.
  REQUIRE(!getBisectorWeights(0.015, 0.1, 0.1, 0.2));
  REQUIRE(!getBisectorWeights(0.015, 0.1, 0.1, 0.25));
  REQUIRE(!getBisectorWeights(0.015, 0.0, 0.1, 0.1));
  REQUIRE(!getBisectorWeights(0.015, 0.1, -0.1, 0.1));
  REQUIRE(!getBisectorWeights(0.015, 0.1, 0.1,
                              std::numeric_limits<double>::quiet_NaN()));
  // The frame nearly straight still has one.
  REQUIRE(getBisectorWeights(0.015, 0.1, 0.1, std::nextafter(0.2, 0.0)));
}

/// The kinds of findRigidGroups against the placement, on copies of a
/// molecule turned and moved: the frame of the constraint's lengths
/// (0.08724, 0.08724, 0.137116 nm, a = 0.01594 nm) built in a plane, then
/// rotated and shifted in f64, so that it holds the lengths only to the
/// rounding of coordinates of size X: at most 2 eps X in a coordinate
/// (three products, their sum, and the shift, half a unit each), 4 eps X
/// in a component of a bond, and tau = 4 sqrt(3) eps X / d in the
/// relative length of a bond of length d. Off the rigid frame the weights
/// and the bisector differ in the first order of tau, by at most
/// a tau (2 / S + 2 (4 - S^2) / S^2) for two equal lengths, S = |u + v|
/// (docs/constraints.md, Section 3.3): 4.85 a tau at S = 1.237 here. Beside
/// the terms of testBisectorWeights the bound therefore has
/// 40 eps a X / d, which is above 4.85 * 4 sqrt(3) = 33.6 of them.
void testPlacedKinds() {
  const double first = 0.08724, across = 0.137116, a = 0.01594;
  std::mt19937_64 random(855);
  std::uniform_real_distribution<double> coordinate(-5.0, 5.0),
      unit(-1.0, 1.0);
  for (bool linear : {false, true}) {
    Made made;
    unsigned o = made.add(16.0, 0.0, 0);
    made.add(1.0, 0.5, 1);
    made.add(1.0, 0.5, 1);
    unsigned s = made.add(0.0, -1.0, 2);
    made.t.settles.push_back({o, first, across});
    // A linear site with two different weights, so that their order
    // shows; the bisector site of the lengths above.
    const double wj = 0.11, wk = 0.23;
    if (linear)
      made.t.virtualSites.push_back(
          {Topology::VirtualSite::Linear, s, o, o + 2, o + 1, wk, wj});
    else
      made.t.virtualSites.push_back(
          {Topology::VirtualSite::AmberWater, s, o, o + 1, o + 2, a, 0.0});
    RigidGroups groups = findRigidGroups(made.finish());
    REQUIRE(groups.kinds.size() == 1 && groups.kinds[0].sites.size() == 1);
    const std::vector<double> &w = groups.kinds[0].sites[0].weights;
    REQUIRE(w.size() == 3);
    REQUIRE(std::fabs(sum(w) - 1.0) <= 2 * Epsilon * absoluteSum(w));

    double worst = 0.0;
    for (unsigned sample = 0; sample != 20000; ++sample) {
      // The frame in its plane.
      double half = 0.5 * across;
      double height = std::sqrt(first * first - half * half);
      Vec local[3] = {{0.0, 0.0, 0.0}, {height, half, 0.0}, {height, -half, 0.0}};
      // A rotation from a random unit quaternion, and a shift.
      double q[4], total = 0.0;
      for (double &x : q) {
        x = unit(random);
        total += x * x;
      }
      for (double &x : q)
        x /= std::sqrt(total);
      double R[3][3] = {
          {1 - 2 * (q[2] * q[2] + q[3] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]),
           2 * (q[1] * q[3] + q[0] * q[2])},
          {2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] * q[1] + q[3] * q[3]),
           2 * (q[2] * q[3] - q[0] * q[1])},
          {2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]),
           1 - 2 * (q[1] * q[1] + q[2] * q[2])}};
      Vec shift{coordinate(random), coordinate(random), coordinate(random)};
      Vec atoms[3];
      for (int k = 0; k != 3; ++k)
        for (int c = 0; c != 3; ++c)
          atoms[k][c] = shift[c] + R[c][0] * local[k][0] +
                        R[c][1] * local[k][1] + R[c][2] * local[k][2];
      Vec placed;
      double norm = 2.0;
      if (linear) {
        // (1 - a - b) x_i + a x_j + b x_k as the engine forms it, from
        // the displacements, with j the third atom and k the second.
        for (int c = 0; c != 3; ++c)
          placed[c] = atoms[0][c] + (wk * (atoms[2][c] - atoms[0][c]) +
                                     wj * (atoms[1][c] - atoms[0][c]));
      } else {
        placed = placeBisector(atoms[0], atoms[1], atoms[2], a, norm);
      }
      Vec combined = combine(w, atoms);
      double X = largest(atoms, 3);
      double bound = Epsilon * (4.0 * (1.0 + absoluteSum(w)) * X +
                                (linear ? 0.0
                                        : 64.0 * a / norm + 40.0 * a * X / first));
      for (int c = 0; c != 3; ++c) {
        double error = std::fabs(combined[c] - placed[c]);
        REQUIRE(error <= bound);
        worst = std::max(worst, error / bound);
      }
    }
    REQUIRE(worst > 0.01);
    llvm::outs() << (linear ? "linear site" : "bisector site")
                 << ": 20000 placements within the bound, largest error "
                 << llvm::format("%.2f", worst) << " of it\n";
  }
}
} // namespace

int main() {
  testKinds();
  testSameness();
  testLeftOut();
  testFacts();
  testBisectorWeights();
  testPlacedKinds();
  return 0;
}
