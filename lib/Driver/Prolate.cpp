//===- Prolate.cpp - The prolate spheroidal wave function of order zero --===//
//
// See Prolate.h. The method: the prolate operator
// L = −d/dx (1 − x²) d/dx + c²x² commutes with the integral operator of
// band limit c and has the same eigenfunctions [Slepian1961]; in the basis
// of the normalized Legendre polynomials P̄_k = √(k + ½) P_k its matrix is
// tridiagonal and, for even k, couples k with k ± 2 [Xiao2001]:
//
//   L_kk     = k(k + 1) + c² (2k(k + 1) − 1) / ((2k + 3)(2k − 1)),
//   L_k,k+2  = c² (k + 2)(k + 1) / ((2k + 3) √((2k + 1)(2k + 5))).
//
// ψ₀ᶜ is the eigenvector of the smallest eigenvalue, found by Sturm
// bisection and refined by inverse iteration below it, where the shifted
// matrix is positive definite and its LDLᵀ needs no pivoting.
//
//===----------------------------------------------------------------------===//

#include "mdir/Driver/Prolate.h"

#include <algorithm>
#include <cmath>

using namespace mdir::prolate;

namespace {

/// The number of eigenvalues of the symmetric tridiagonal (d, e) below x.
int countBelow(const std::vector<double> &d, const std::vector<double> &e,
               double x) {
  int count = 0;
  double q = d[0] - x;
  if (q < 0)
    ++count;
  for (size_t i = 1; i != d.size(); ++i) {
    if (q == 0.0)
      q = 1e-300;
    q = d[i] - x - e[i - 1] * e[i - 1] / q;
    if (q < 0)
      ++count;
  }
  return count;
}

/// Gauss–Legendre nodes and weights on [−1, 1], by Newton's method on P_n.
void gaussLegendre(int n, std::vector<double> &x, std::vector<double> &w) {
  x.resize(n);
  w.resize(n);
  for (int i = 0; i != n; ++i) {
    double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), dp = 1.0;
    for (int iteration = 0; iteration != 100; ++iteration) {
      double p0 = 1.0, p1 = z;
      for (int k = 2; k <= n; ++k) {
        double p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      dp = n * (z * p1 - p0) / (z * z - 1.0);
      double dz = p1 / dp;
      z -= dz;
      if (std::fabs(dz) < 1e-16)
        break;
    }
    x[i] = z;
    w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
  }
}

} // namespace

Function::Function(double bandwidth) : c(bandwidth) {
  // Enough terms that the last coefficients are below the rounding: the
  // coefficients decay superexponentially beyond k ≈ c.
  int n = static_cast<int>(std::ceil(c)) + 40;
  std::vector<double> d(n), e(n - 1);
  double c2 = c * c;
  for (int i = 0; i != n; ++i) {
    double k = 2.0 * i;
    d[i] = k * (k + 1) + c2 * (2 * k * (k + 1) - 1) / ((2 * k + 3) * (2 * k - 1));
    if (i + 1 != n)
      e[i] = c2 * (k + 2) * (k + 1) /
             ((2 * k + 3) * std::sqrt((2 * k + 1) * (2 * k + 5)));
  }
  // The smallest eigenvalue by bisection between Gershgorin bounds.
  double low = d[0], high = d[0];
  for (int i = 0; i != n; ++i) {
    double radius = (i ? std::fabs(e[i - 1]) : 0.0) +
                    (i + 1 != n ? std::fabs(e[i]) : 0.0);
    low = std::min(low, d[i] - radius);
    high = std::max(high, d[i] + radius);
  }
  for (int iteration = 0; iteration != 200; ++iteration) {
    double middle = 0.5 * (low + high);
    if (middle == low || middle == high)
      break;
    if (countBelow(d, e, middle) >= 1)
      high = middle;
    else
      low = middle;
  }
  // Inverse iteration with a shift just below it: (T − μ) is positive
  // definite, so its LDLᵀ is stable without pivoting.
  double mu = low - 1e-9 * std::max(1.0, std::fabs(low));
  std::vector<double> v(n, 1.0), diag(n), sub(n);
  for (int iteration = 0; iteration != 4; ++iteration) {
    diag[0] = d[0] - mu;
    for (int i = 1; i != n; ++i) {
      sub[i] = e[i - 1] / diag[i - 1];
      diag[i] = d[i] - mu - sub[i] * e[i - 1];
    }
    for (int i = 1; i != n; ++i)
      v[i] -= sub[i] * v[i - 1];
    v[n - 1] /= diag[n - 1];
    for (int i = n - 2; i >= 0; --i)
      v[i] = (v[i] - e[i] * v[i + 1]) / diag[i];
    double norm = 0.0;
    for (double a : v)
      norm += a * a;
    norm = std::sqrt(norm);
    for (double &a : v)
      a /= norm;
  }
  // Σ β_k² = 1 is ∫ψ² = 1; plain Legendre coefficients a_k = β_k √(k + ½).
  coefficients.assign(2 * n, 0.0);
  for (int i = 0; i != n; ++i)
    coefficients[2 * i] = v[i] * std::sqrt(2.0 * i + 0.5);
  if (value(0.0) < 0.0)
    for (double &a : coefficients)
      a = -a;
  // λ₀ at x = 0: ∫ψ = λ₀ ψ(0), and ∫P_k = 2 δ_k0.
  lambda = 2.0 * coefficients[0] / value(0.0);
  c0 = integral(1.0);
  int points = std::max(200, static_cast<int>(4 * c) + 100);
  gaussLegendre(points, nodes, weights);
  values.resize(points);
  for (int i = 0; i != points; ++i)
    values[i] = value(nodes[i]);
}

double Function::value(double x) const {
  double p0 = 1.0, p1 = x, sum = coefficients[0];
  for (size_t k = 1; k + 1 < coefficients.size(); ++k) {
    double p2 = ((2 * k + 1) * x * p1 - k * p0) / (k + 1);
    p0 = p1;
    p1 = p2;
    if ((k + 1) % 2 == 0)
      sum += coefficients[k + 1] * p1;
  }
  return sum;
}

double Function::derivative(double x) const {
  // P'_{k+1} = P'_{k−1} + (2k + 1) P_k, regular at ±1.
  double p0 = 1.0, p1 = x, dp0 = 0.0, dp1 = 1.0, sum = 0.0;
  for (size_t k = 1; k + 1 < coefficients.size(); ++k) {
    double p2 = ((2 * k + 1) * x * p1 - k * p0) / (k + 1);
    double dp2 = dp0 + (2 * k + 1) * p1;
    p0 = p1;
    p1 = p2;
    dp0 = dp1;
    dp1 = dp2;
    if ((k + 1) % 2 == 0)
      sum += coefficients[k + 1] * dp1;
  }
  return sum;
}

double Function::integral(double x) const {
  // ∫₀ˣ P_0 = x; ∫₀ˣ P_k = (P_{k+1} − P_{k−1}) / (2k + 1) for even k ≥ 2,
  // the odd polynomials vanishing at 0.
  std::vector<double> p(coefficients.size() + 1);
  p[0] = 1.0;
  p[1] = x;
  for (size_t k = 1; k + 1 < p.size(); ++k)
    p[k + 1] = ((2 * k + 1) * x * p[k] - k * p[k - 1]) / (k + 1);
  double sum = coefficients[0] * x;
  for (size_t k = 2; k < coefficients.size(); k += 2)
    sum += coefficients[k] * (p[k + 1] - p[k - 1]) / (2 * k + 1);
  return sum;
}

double Function::transform(double k) const {
  double sum = 0.0;
  for (size_t i = 0; i != nodes.size(); ++i)
    sum += weights[i] * values[i] * std::cos(k * nodes[i]);
  return sum;
}

double Function::transformDerivative(double k) const {
  double sum = 0.0;
  for (size_t i = 0; i != nodes.size(); ++i)
    sum -= weights[i] * values[i] * nodes[i] * std::sin(k * nodes[i]);
  return sum;
}

double Function::secondMoment() const {
  double moment = 0.0, mass = 0.0;
  for (size_t i = 0; i != nodes.size(); ++i) {
    moment += weights[i] * values[i] * nodes[i] * nodes[i];
    mass += weights[i] * values[i];
  }
  return moment / mass;
}

double mdir::prolate::bandwidthForTolerance(double epsilon) {
  // ψ₀ᶜ(1) falls with c.
  double low = 0.5, high = 80.0;
  for (int iteration = 0; iteration != 60; ++iteration) {
    double middle = 0.5 * (low + high);
    if (Function(middle).value(1.0) > epsilon)
      low = middle;
    else
      high = middle;
  }
  return 0.5 * (low + high);
}
