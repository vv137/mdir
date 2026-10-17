//===- Prolate.h - The prolate spheroidal wave function of order zero ----===//
//
// ψ₀ᶜ on [−1, 1], the eigenfunction of largest eigenvalue of
// φ ↦ ∫₋₁¹ e^{icxt} φ(t) dt [Slepian1961], for the splitting and the
// spreading of Ewald summation with prolates [Liang2026] (D266).
// It is a series of Legendre polynomials of even degree whose coefficients
// are the eigenvector of smallest eigenvalue of the symmetric tridiagonal
// matrix of the prolate differential operator in the normalized Legendre
// basis [Xiao2001], computed in f64 on the host. ψ is normalized to
// ∫₋₁¹ ψ² = 1 with ψ(0) > 0, the normalization in which ψ₀ᶜ(1) = ε gives the
// bandwidths of Table 2 of [Liang2026].
//
//===----------------------------------------------------------------------===//

#ifndef MDIR_DRIVER_PROLATE_H
#define MDIR_DRIVER_PROLATE_H

#include <cstdint>
#include <functional>
#include <vector>

namespace mdir {
namespace prolate {

class Function {
public:
  /// ψ₀ᶜ of bandwidth `c` > 0.
  explicit Function(double c);

  double bandwidth() const { return c; }
  /// ψ(x), for |x| ≤ 1.
  double value(double x) const;
  /// ψ'(x), for |x| ≤ 1.
  double derivative(double x) const;
  /// ∫₀ˣ ψ, for |x| ≤ 1.
  double integral(double x) const;
  /// ∫₋₁¹ ψ(t) cos(k t) dt, by Gauss–Legendre quadrature, for any k.
  double transform(double k) const;
  /// d/dk of `transform`.
  double transformDerivative(double k) const;
  /// λ₀ of ∫₋₁¹ e^{icxt} ψ(t) dt = λ₀ ψ(x).
  double eigenvalue() const { return lambda; }
  /// C₀ = ∫₀¹ ψ.
  double halfIntegral() const { return c0; }
  /// ∫₋₁¹ x² ψ / ∫₋₁¹ ψ.
  double secondMoment() const;

private:
  double c;
  std::vector<double> coefficients; // of P_k, k even; odd entries zero
  double lambda = 0.0, c0 = 0.0;
  std::vector<double> nodes, weights, values; // Gauss–Legendre on [−1, 1]
};

/// The bandwidth c such that ψ₀ᶜ(1) = `epsilon` (0 < ε < 1), by bisection.
double bandwidthForTolerance(double epsilon);

/// The coefficients a_0 .. a_{n−1} of the Chebyshev series Σ a_k T_k(t) on
/// [−1, 1] that interpolates `f` at 64 Chebyshev points, cut after the
/// last coefficient above `tolerance` times the largest, or times `scale`
/// if that is larger.

std::vector<double>
getChebyshevSeries(const std::function<double(double)> &f, double tolerance,
                   double scale = 0.0);

/// The window of the spreading, φ₁(d) = ψ₀^{c₁}(2d / P) − ψ₀^{c₁}(1) for
/// |d| ≤ P/2 in points of the grid: the window of [Liang2026, eq. 14] less
/// its value at the edge of its support, so that it is continuous there. A
/// particle that crosses a plane of the grid moves the point of its edge
/// from one end of its P points to the other; with the window of the paper
/// that moves a weight of ψ(1) ≈ ε/2, the energy of the mesh jumps, and
/// the total energy of a run drifts (#334: 75 kcal/mol/ns on JAC at
/// ε = 1e-3). As the kernels evaluate it: a
/// particle at the scaled coordinate u adds to the P points first + j,
/// first = ⌊u − h⌋ − ⌊P/2⌋ + 1 with h = ½ for odd P and 0 for even P, with
/// the weight φ₁(s + h + ⌊P/2⌋ − 1 − j) where s = u − h − ⌊u − h⌋. Each
/// weight, and its derivative in u, is a Chebyshev series in t = 2s − 1 of
/// `degree` terms, row j of `values` and of `slopes` (degree × P each).
struct Window {
  int64_t points = 0, degree = 0;
  /// The terms that f32 takes: those above 1e-8 of ψ(0) in any weight or
  /// slope, a window within the rounding of f32 of the full one.
  int64_t degreeSingle = 0;
  std::vector<double> values, slopes;
  /// φ̂₁(κ) = ∫ φ₁(d) e^{−iκd} dd, of κ in radians a point.
  double transform(double kappa) const;
  /// The same of the window of the paper, ψ₀^{c₁}(2d / P) without the
  /// value at the edge taken off: what the aliasing estimate of eq. 10 is
  /// calibrated with (getAliasing).
  double paperTransform(double kappa) const;
  double c1 = 0.0, edge = 0.0;
  std::vector<double> transformNodes, transformWeights;
};
Window getWindow(double c1, int64_t points);

/// The spectral factor of the splitting, χ̂(k) = λ₀ ψ₀ᶜ(k/c) / (2C₀) for
/// k ≤ c and 0 beyond, with χ̂(0) = 1 [Liang2026, eq. 5], and k χ̂'(k),
/// each a Chebyshev series in y = 2x² − 1 of x = k/c, of `terms` terms.
struct Influence {
  int64_t terms = 0;
  std::vector<double> value, slope;
};
Influence getInfluence(const Function &psi);

/// The parts of the splitting that the pair terms take, as functions of
/// x = r / r_c on [0, 1] (1 beyond): `Far`, 1 − Φ(x), which multiplies
/// f q q / r in the direct sum; `Near`, Φ(x)/x = ∫₀¹ χ(xτ) dτ, which
/// multiplies f q q / r_c in the excluded pairs. Each a table of `cells`
/// rows, the polynomial of the cell [k, k + 1)/cells in t = 2(x cells − k)
/// − 1 with `terms` coefficients of t⁰ upwards, within 1e-14 of the
/// largest value.
enum class Part { Far, Near };
std::vector<double> getSplittingTable(const Function &psi, Part part,
                                      int64_t cells, int64_t terms);
double getSplitting(const Function &psi, Part part, double x);

/// (1 − Φ(x))/(1 − x), the mean of χ over [x, 1], as a Chebyshev series in
/// t = 2x − 1 for x in [0, 1], to 10⁻¹⁴ of its largest term. The direct
/// sum f q q (1 − Φ(x))/r is then f q q (1 − x) Σ a_k T_k(t) / r, a
/// function of r² whose value near the cutoff, of the order of ψ(1)(1 − x),
/// carries the relative accuracy of (1 − x) and of the series, not the
/// cancellation of 1 − Φ.
std::vector<double> getFarSeries(const Function &psi);

/// The aliasing error of the spreading relative to the spectral sum that
/// it perturbs, the estimate of eq. 10 of [Liang2026] divided by the same
/// sum without the factor of the alias: for each axis a,
/// 6 Σ_k w(k) |φ̂₁(2π(k_a + K_a)/K_a) / φ̂₁(2πk_a/K_a)| / Σ_k w(k) of the paper's window (Window::paperTransform), with
/// w(k) = χ̂(2π r_c |m|) / |m|² over the wave vectors m = k/L within the
/// band of χ̂; the largest of the three. Lengths in any one unit.
double getAliasing(const Function &psi, const Window &window,
                   const double (&edges)[3], const int64_t (&grid)[3],
                   double cutoff);

} // namespace prolate
} // namespace mdir

#endif // MDIR_DRIVER_PROLATE_H
