//===- InfluenceTest.cpp - Influence functions against direct alias sums -===//
//
// Checks the builder of influence functions (lib/Driver/Influence.cpp,
// D[p3m-influence]) against sums over the aliases written out here, a
// different arrangement of the same series:
//
//  1. influence "OPTIMAL" (|b|² λ², getSplineModuli × getAliasFactors) is
//     eq. 4.2 of [Ballenegger2012], per edge, Û²(n) / (Σ_j Û²(n + jK))²,
//     to the truncation of its sums at |j| ≤ 50; and how far eq. 4.2 with
//     the sums of "P3M" (|j| ≤ 2) is from it.
//  2. eq. 2.28 in a rectangular cell, as the kernels compute it from the
//     rows of the edges (getEdgeSums: a product of factors over a sum of
//     mean squares), equals the direct sum over J ∈ ℤ³ with max|J_a| ≤ 2
//     to rounding, and its sums for the virial too.
//  3. the virial of a point, E (δ − 2 (π²/β²) A − 2 B / D), is −∂E/∂ε of
//     the direct sum under a strain of the cell, by central differences
//     extrapolated to a step of 0, on the diagonal and off it.
//  4. the truncation: eq. 2.28 with |j| ≤ 2 against |j| ≤ 12; and,
//     reported, how far eq. 4.1 (eq. 2.28 without the aliases of its
//     numerator) is from eq. 2.28.
//
// Prints each quantity, its reference, and "FAIL" beyond its tolerance.
//
//===----------------------------------------------------------------------===//

#include "mdir/Driver/Influence.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace mdir::influence;

namespace {
int failures = 0;

void report(const char *what, double difference, double tolerance) {
  bool ok = difference <= tolerance;
  failures += !ok;
  std::printf("%s: %.1e (tolerance %.0e) %s\n", what, difference, tolerance,
              ok ? "ok" : "FAIL");
}

Basis rectangular(double lx, double ly, double lz) {
  Basis basis{};
  basis[0][0] = 1.0 / lx;
  basis[1][1] = 1.0 / ly;
  basis[2][2] = 1.0 / lz;
  return basis;
}

/// The cell strained by ε, x → (1 + ε) x: the reciprocal basis, the rows
/// of H⁻¹, goes to H⁻¹ (1 + ε)⁻¹, each row m → (1 + ε)⁻ᵀ m.
Basis strained(const Basis &basis, int a, int b, double e) {
  // (1 + ε)⁻¹ with one element ε_ab = e: 1 − e E_ab (+ e² for a = b).
  double inverse[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  if (a == b)
    inverse[a][a] = 1.0 / (1.0 + e);
  else
    inverse[a][b] = -e;
  Basis out{};
  for (int r = 0; r != 3; ++r)
    for (int c = 0; c != 3; ++c)
      for (int k = 0; k != 3; ++k)
        out[r][c] += basis[r][k] * inverse[k][c];
  return out;
}

double volume(const Basis &basis) {
  const auto &m = basis;
  double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  return 1.0 / std::fabs(det);
}
} // namespace

int main() {
  // 1. OPTIMAL against eq. 4.2, per edge.
  for (auto [count, order] : {std::pair<int64_t, int64_t>{64, 4}, {54, 4},
                              {25, 4}, {70, 6}, {70, 8}}) {
    std::vector<double> moduli = getSplineModuli(count, order);
    std::vector<double> factors = getAliasFactors(count, order);
    Window window = getBSpline(order);
    double worst50 = 0.0, worst2 = 0.0;
    for (int64_t k = 0; k != count; ++k) {
      int64_t n = getSigned(k, count);
      double optimal = moduli[k] * factors[k];
      for (int64_t aliases : {50, 2}) {
        double u0 = window.transform(double(n) / count), sum = 0.0;
        for (int64_t j = -aliases; j <= aliases; ++j) {
          double u = window.transform(double(n + j * count) / count);
          sum += u * u;
        }
        double eq42 = u0 * u0 / (sum * sum);
        double difference = std::fabs(optimal / eq42 - 1.0);
        double &worst = aliases == 50 ? worst50 : worst2;
        worst = std::max(worst, difference);
      }
    }
    char what[128];
    std::snprintf(what, sizeof what,
                  "K %lld order %lld: max |OPTIMAL / eq. 4.2 - 1|, |j| <= 50",
                  (long long)count, (long long)order);
    report(what, worst50, order == 4 ? 5e-7 : 1e-9);
    std::snprintf(what, sizeof what,
                  "K %lld order %lld: max |OPTIMAL / eq. 4.2 - 1|, |j| <= 2",
                  (long long)count, (long long)order);
    report(what, worst2, 1e-3);
  }

  // 2.-4. Eq. 4.1 in a rectangular cell: JAC's edge with 64 points, a
  // coarse grid of 2 Å, and an odd grid; β of MDIR's tolerance 1e-5 at
  // 8 Å (0.3904 / Å) and pmemd's at 1e-6 (0.39467 / Å).
  struct Case {
    double lx, ly, lz;
    int64_t kx, ky, kz, order;
    double beta;
  };
  const Case cases[] = {
      {62.23, 62.23, 62.23, 64, 64, 64, 4, 0.39467},
      {62.23, 62.23, 62.23, 32, 32, 32, 4, 0.3904},
      {40.0, 35.0, 30.0, 25, 21, 27, 6, 0.3904},
      {40.0, 35.0, 30.0, 20, 18, 16, 8, 0.3904},
  };
  const int64_t aliases = 2;
  for (const Case &c : cases) {
    Window window = getBSpline(c.order);
    Splitting gaussian = getGaussian(c.beta);
    Basis basis = rectangular(c.lx, c.ly, c.lz);
    std::array<int64_t, 3> grid = {c.kx, c.ky, c.kz};
    double lengths[3] = {c.lx, c.ly, c.lz};
    EdgeWeights rows[3];
    for (int a = 0; a != 3; ++a)
      rows[a] = getEdgeWeights(window, grid[a], aliases);
    double worstValue = 0.0, worstSums = 0.0, worstVirial = 0.0,
           worstTruncation = 0.0, worstNumerator = 0.0, largest = 0.0;
    // Every point of a grid of the coarse cases, a lattice of points of
    // the others, the half-complex part (k3 ≤ K3 / 2), m = 0 left out.
    for (int64_t k0 = 0; k0 < grid[0]; k0 += grid[0] > 32 ? 9 : 2)
      for (int64_t k1 = 0; k1 < grid[1]; k1 += grid[1] > 32 ? 7 : 3)
        for (int64_t k2 = 0; k2 <= grid[2] / 2; k2 += grid[2] > 32 ? 3 : 1) {
          std::array<int64_t, 3> ks = {k0, k1, k2};
          std::array<int64_t, 3> n;
          for (int a = 0; a != 3; ++a)
            n[a] = getSigned(ks[a], grid[a]);
          if (n[0] == 0 && n[1] == 0 && n[2] == 0)
            continue;
          Point direct =
              getForceOptimal(window, gaussian, basis, grid, n, aliases);
          largest = std::max(largest, direct.value);
          // 2. The separable form of the kernels.
          EdgeSums e[3];
          double product = 1.0, squares = 0.0;
          for (int a = 0; a != 3; ++a) {
            e[a] = getEdgeSums(rows[a], grid[a], aliases, ks[a], lengths[a],
                               gaussian.gauss);
            product *= e[a].factor;
            squares += e[a].square;
          }
          double separable = product / squares;
          // Eq. 4.1: eq. 2.28 without the sum over the aliases in its
          // numerator, Û²(n) m² φ̂(m) at J = 0 alone.
          double numerator = 1.0;
          for (int a = 0; a != 3; ++a) {
            double mu = double(n[a]) / lengths[a];
            numerator *= rows[a].weights[aliases * grid[a] + ks[a]] *
                         std::exp(-gaussian.gauss * mu * mu) *
                         rows[a].inverse[ks[a]];
          }
          double eq41 = numerator / squares;
          // Relative where the value matters: against the largest value of
          // the function on the grid, 2e-16 per operation, and relative
          // to the value itself beyond 1e-12 of the largest.
          worstValue = std::max(
              worstValue, std::fabs(separable - direct.value) /
                              std::max(direct.value, 1e-12 * largest));
          for (int a = 0; a != 3; ++a)
            for (int b = 0; b != 3; ++b) {
              double ga = a == b ? e[a].gaussSquare
                                 : e[a].gaussMean * e[b].gaussMean;
              double pa = a == b ? e[a].square : e[a].mean * e[b].mean;
              double scale = squares;
              worstSums = std::max(
                  {worstSums,
                   std::fabs(ga - direct.gaussian[a][b]) / scale,
                   std::fabs(pa - direct.plain[a][b]) / scale});
            }
          // 3. The virial against central differences of V⁻¹ G under a
          // strain, at a point where the value matters.
          if (direct.value > 1e-6 * largest && (k0 + k1 + k2) % 4 == 0) {
            double D = direct.plain[0][0] + direct.plain[1][1] +
                       direct.plain[2][2];
            for (int a = 0; a != 3; ++a)
              for (int b = 0; b != 3; ++b) {
                double h = 1e-4;
                auto energy = [&](double s) {
                  Basis strainedBasis = strained(basis, a, b, s);
                  return getForceOptimal(window, gaussian, strainedBasis, grid,
                                         n, aliases)
                             .value /
                         volume(strainedBasis);
                };
                // Central differences at h and h / 2, extrapolated
                // (Richardson): an error of O(h⁴).
                double coarse = (energy(h) - energy(-h)) / (2.0 * h);
                double fine = (energy(h / 2) - energy(-h / 2)) / h;
                double derivative = fine + (fine - coarse) / 3.0;
                double e0 = direct.value / volume(basis);
                double virial =
                    e0 * ((a == b ? 1.0 : 0.0) -
                          2.0 * gaussian.gauss * direct.gaussian[a][b] -
                          2.0 * direct.plain[a][b] / D);
                worstVirial = std::max(
                    worstVirial, std::fabs(virial + derivative) / e0);
              }
          }
          // 4. The truncation of the sums over the aliases.
          Point wide = getForceOptimal(window, gaussian, basis, grid, n, 12);
          if (direct.value > 1e-6 * largest)
            worstTruncation = std::max(
                worstTruncation, std::fabs(direct.value / wide.value - 1.0));
          if (direct.value > 1e-6 * largest)
            worstNumerator =
                std::max(worstNumerator, std::fabs(eq41 / direct.value - 1.0));
        }
    char what[160];
    std::snprintf(what, sizeof what,
                  "%lldx%lldx%lld order %lld beta %.4f: eq. 2.28, edges "
                  "against the direct sum",
                  (long long)c.kx, (long long)c.ky, (long long)c.kz,
                  (long long)c.order, c.beta);
    report(what, worstValue, 1e-13);
    std::snprintf(what, sizeof what,
                  "%lldx%lldx%lld order %lld: sums of the virial, edges "
                  "against the direct sum",
                  (long long)c.kx, (long long)c.ky, (long long)c.kz,
                  (long long)c.order);
    report(what, worstSums, 1e-13);
    std::snprintf(what, sizeof what,
                  "%lldx%lldx%lld order %lld: virial against -dE/de",
                  (long long)c.kx, (long long)c.ky, (long long)c.kz,
                  (long long)c.order);
    report(what, worstVirial, 1e-9);
    std::snprintf(what, sizeof what,
                  "%lldx%lldx%lld order %lld: max |G(|j|<=2) / G(|j|<=12) - "
                  "1| where G > 1e-6 of its largest",
                  (long long)c.kx, (long long)c.ky, (long long)c.kz,
                  (long long)c.order);
    report(what, worstTruncation, 1e-4);
    // Not a tolerance: how far eq. 4.1 is from eq. 2.28 (reported).
    std::snprintf(what, sizeof what,
                  "%lldx%lldx%lld order %lld: max |eq. 4.1 / eq. 2.28 - 1| "
                  "where G > 1e-6 of its largest",
                  (long long)c.kx, (long long)c.ky, (long long)c.kz,
                  (long long)c.order);
    std::printf("%s: %.1e (reported)\n", what, worstNumerator);
  }
  std::printf("%d failures\n", failures);
  return failures != 0;
}
