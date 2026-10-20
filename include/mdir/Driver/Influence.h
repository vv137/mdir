//===- Influence.h - Influence functions of mesh Ewald ----------*- C++ -*-===//
//
// The influence functions (lattice Green functions) of particle mesh Ewald
// with analytic differentiation (AD), built from a splitting and a window
// by sums over the aliases of each point of the grid [Ballenegger2012]
// (docs/pme-m1.md, Section 1.1; D[p3m-influence]).
//
// The notation is that of docs/pme-m1.md: a wave vector is m = H⁻ᵀ ν, with
// ν = n + J K the signed index n of a point of the grid shifted by J ∈ ℤ³
// times the numbers of points K, so that m is in reciprocal length (no
// factor 2π). The window enters through Û(ν_a / K_a) along each edge, the
// splitting through m² φ̂(m) up to the constant f / (π V), which for the
// Gaussian splitting is exp(−π² m² / β²). With the sums over the aliases
// J taken over max_a |J_a| ≤ J_max:
//
//   eq. 4.1 (AD, optimal for the forces)
//     G(n) = Σ_J Û²(ν) m² φ̂(m) / ([Σ_J Û²(ν)] [Σ_J Û²(ν) m²])
//   eq. 4.2 (separable, = influence "OPTIMAL")
//     G(n) = Û²(n) φ̂(m₀) / [Σ_J Û²(ν)]²
//
// (equation numbers of [Ballenegger2012] as cited in issue #468.) The
// functions here are on the host, in f64. The direct sums over J ∈ ℤ³ are
// the reference; the separable forms are what the driver tabulates for the
// kernels.
//
//===----------------------------------------------------------------------===//

#ifndef MDIR_DRIVER_INFLUENCE_H
#define MDIR_DRIVER_INFLUENCE_H

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace mdir {
namespace influence {

/// The window of the spreading: Û(t) at t = ν / K, the Fourier transform
/// of the assignment function over one spacing of the grid, Û(0) = 1.
struct Window {
  std::function<double(double t)> transform;
};

/// The cardinal B-spline of order `order`: Û(t) = (sin(π t) / (π t))ⁿ,
/// exactly 0 at every nonzero integer t.
Window getBSpline(int64_t order);

/// The splitting: m² φ̂(m) as a function of m², up to the constant
/// f / (π V); `separable` when it is a product over the axes of a
/// rectangular cell, with `gauss` its exponent per unit m² then.
struct Splitting {
  std::function<double(double msq)> weighted;
  bool separable = false;
  double gauss = 0.0;
};

/// The Gaussian of Ewald: m² φ̂(m) = exp(−π² m² / β²).
Splitting getGaussian(double beta);

/// A cell by its reciprocal basis: m = Σ_a ν_a recip[a], with recip[a]
/// the row a of H⁻¹ (a rectangular cell: recip[a] = e_a / L_a).
using Basis = std::array<std::array<double, 3>, 3>;

/// The signed index of point k of an edge of `count` points, as the
/// kernels number them: k for k ≤ count / 2, else k − count.
int64_t getSigned(int64_t k, int64_t count);

/// The influence function at the point `n` (signed indices) of a grid of
/// `grid` points, and the sums the virial takes: with weights
/// w_J = Û²(ν), a_J = w_J m² φ̂(m),
///   value = Σ a / (Σ w · Σ w m²),
///   gaussian[a][b] = Σ a m_a m_b / Σ a,  plain[a][b] = Σ w m_a m_b / Σ w,
/// so that, for the Gaussian, the virial of the point is
///   W_ab = E_n (δ_ab − 2 (π²/β²) gaussian_ab − 2 plain_ab / D),
///   D = tr plain. 0 at n = 0.
struct Point {
  double value = 0.0;
  std::array<std::array<double, 3>, 3> gaussian{}, plain{};
};
Point getForceOptimal(const Window &window, const Splitting &splitting,
                      const Basis &basis, const std::array<int64_t, 3> &grid,
                      const std::array<int64_t, 3> &n, int64_t aliases);

/// Eq. 4.2 at the point `n`: Û²(n) φ̂(m₀) / (Σ_J Û²(ν))², the Û
/// a product over the axes. 0 at n = 0.
double getEnergyOptimal(const Window &window, const Splitting &splitting,
                        const Basis &basis, const std::array<int64_t, 3> &grid,
                        const std::array<int64_t, 3> &n, int64_t aliases);

/// The rows of the table of an edge of `count` points for eq. 4.1 with a
/// separable splitting in a rectangular cell, which do not depend on the
/// cell: `inverse`[k] = 1 / Σ_j Û²(ν_j) and `weights`[(j + aliases) count
/// + k] = Û²(ν_j) / Σ_j Û²(ν_j), ν_j = getSigned(k) + j count, for
/// |j| ≤ aliases.
struct EdgeWeights {
  std::vector<double> inverse, weights;
};
EdgeWeights getEdgeWeights(const Window &window, int64_t count,
                           int64_t aliases);

/// What a kernel computes for point k of an edge of length `length` from
/// those rows (docs/pme-m1.md, Section 1.1): with μ_j = ν_j / L, g_j =
/// exp(−gauss μ_j²), u_j the weights,
///   factor = Σ u g / S, mean = Σ u μ, square = Σ u μ²,
///   gaussMean = Σ u g μ / Σ u g, gaussSquare = Σ u g μ² / Σ u g
/// (the last two 0 where Σ u g is 0).
struct EdgeSums {
  double factor = 0.0, mean = 0.0, square = 0.0, gaussMean = 0.0,
         gaussSquare = 0.0;
};
EdgeSums getEdgeSums(const EdgeWeights &rows, int64_t count, int64_t aliases,
                     int64_t k, double length, double gauss);

/// |b(m)|² of the Euler exponential spline of order `order` for each index
/// of a grid of `count` points [Essmann1995].
std::vector<double> getSplineModuli(int64_t count, int64_t order);

/// The square of the factor λ of influence "OPTIMAL" for each index
/// (docs/pme-m1.md, Section 1.1), with the sums over |j| ≤ 50.
std::vector<double> getAliasFactors(int64_t count, int64_t order);

} // namespace influence
} // namespace mdir

#endif // MDIR_DRIVER_INFLUENCE_H
