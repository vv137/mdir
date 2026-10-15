//===- Prolate.h - The prolate spheroidal wave function of order zero ----===//
//
// ψ₀ᶜ on [−1, 1], the eigenfunction of largest eigenvalue of
// φ ↦ ∫₋₁¹ e^{icxt} φ(t) dt [Slepian1961], for the splitting and the
// spreading of Ewald summation with prolates [Liang2026] (D[prolate-ewald]).
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

} // namespace prolate
} // namespace mdir

#endif // MDIR_DRIVER_PROLATE_H
