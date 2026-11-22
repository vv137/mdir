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
#include "mdir/Driver/ProlateC2.h"
#include <array>
#include <numeric>

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

// Project the band-concentration operator onto the null space of value
// and slope at x=1 in the orthonormal even Legendre basis on [0,1].
using Basis = std::array<double, C2Basis>;
using Matrix = std::array<Basis, C2Basis>;
double dot(const Basis &a, const Basis &b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

std::vector<double> constrainedKernel(double bandwidth) {
  Matrix basis{}, concentration{}, eigenvectors{};
  for (int i = 0; i < C2Basis; ++i) {
    basis[0][i] = std::sqrt(4 * i + 1);
    basis[1][i] = i * (2 * i + 1) * basis[0][i];
    if (i >= 2)
      basis[i][i - 2] = 1;
    eigenvectors[i][i] = 1;
  }
  for (int i = 0; i < C2Basis; ++i) {
    for (int pass = 0; pass < 2; ++pass)
      for (int j = 0; j < i; ++j) {
        double projection = dot(basis[i], basis[j]);
        for (int k = 0; k < C2Basis; ++k)
          basis[i][k] -= projection * basis[j][k];
      }
    double norm = std::sqrt(dot(basis[i], basis[i]));
    for (double &v : basis[i])
      v /= norm;
  }

  std::vector<double> nodes, weights;
  gaussLegendre(std::max(128, static_cast<int>(bandwidth) + 64), nodes,
                weights);
  std::vector<Basis> values(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    double x = (1 + nodes[i]) / 2, prev = 1, cur = x;
    values[i][0] = weights[i] / 2;
    for (int k = 2; k <= 2 * (C2Basis - 1); ++k) {
      double next = ((2 * k - 1) * x * cur - (k - 1) * prev) / k;
      if (k % 2 == 0)
        values[i][k / 2] = std::sqrt(2 * k + 1) * next * weights[i] / 2;
      prev = cur;
      cur = next;
    }
  }
  for (size_t f = 0; f < nodes.size(); ++f) {
    double frequency = bandwidth * (1 + nodes[f]) / 2;
    Basis spectrum{}, projected{};
    for (size_t i = 0; i < nodes.size(); ++i) {
      double cosine = std::cos(frequency * (1 + nodes[i]) / 2);
      for (int k = 0; k < C2Basis; ++k)
        spectrum[k] += cosine * values[i][k];
    }
    for (int i = 2; i < C2Basis; ++i)
      projected[i] = dot(basis[i], spectrum);
    for (int i = 2; i < C2Basis; ++i)
      for (int j = 2; j < C2Basis; ++j)
        concentration[i][j] += weights[f] * projected[i] * projected[j];
  }
  // Symmetric Jacobi diagonalization in the constrained subspace.
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0;
    for (int i = 2; i < C2Basis; ++i)
      for (int j = i + 1; j < C2Basis; ++j) {
        double cross = concentration[i][j];
        off = std::max(off, std::fabs(cross));
        if (std::fabs(cross) < 1e-19)
          continue;
        double tau = (concentration[j][j] - concentration[i][i]) / (2 * cross);
        double t =
            std::copysign(1 / (std::fabs(tau) + std::hypot(1.0, tau)), tau);
        double c = 1 / std::hypot(1.0, t), s = t * c;
        concentration[i][i] -= t * cross;
        concentration[j][j] += t * cross;
        concentration[i][j] = concentration[j][i] = 0;
        for (int k = 2; k < C2Basis; ++k) {
          if (k != i && k != j) {
            double x = concentration[k][i], y = concentration[k][j];
            concentration[i][k] = concentration[k][i] = c * x - s * y;
            concentration[j][k] = concentration[k][j] = s * x + c * y;
          }
          double x = eigenvectors[k][i], y = eigenvectors[k][j];
          eigenvectors[k][i] = c * x - s * y;
          eigenvectors[k][j] = s * x + c * y;
        }
      }
    if (off < 2e-18)
      break;
  }
  int best = 2;
  for (int i = 3; i < C2Basis; ++i)
    if (concentration[i][i] > concentration[best][best])
      best = i;
  std::vector<double> coefficients(2 * C2Basis);
  for (int k = 0; k < C2Basis; ++k)
    for (int i = 2; i < C2Basis; ++i)
      coefficients[2 * k] +=
          std::sqrt(4 * k + 1) * basis[i][k] * eigenvectors[i][best];
  return coefficients;
}

} // namespace

Function::Function(double bandwidth, bool constrained) : c(bandwidth) {
  if (constrained) {
    coefficients = constrainedKernel(c);
  } else {
    // Enough terms that the last coefficients are below the rounding: the
    // coefficients decay superexponentially beyond k ≈ c.
    int n = static_cast<int>(std::ceil(c)) + 40;
    std::vector<double> d(n), e(n - 1);
    double c2 = c * c;
    for (int i = 0; i != n; ++i) {
      double k = 2.0 * i;
      d[i] = k * (k + 1) +
             c2 * (2 * k * (k + 1) - 1) / ((2 * k + 3) * (2 * k - 1));
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
  }
  if (value(0.0) < 0.0)
    for (double &a : coefficients)
      a = -a;
  // λ₀ at x = 0: ∫ψ = λ₀ ψ(0), and ∫P_k = 2 δ_k0.
  lambda = 2.0 * coefficients[0] / value(0.0);
  c0 = integral(1.0);
  int points = std::max(200, static_cast<int>(4 * c) + 100);
  if (constrained)
    points = std::max(points, 384);
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

std::vector<double>
mdir::prolate::getChebyshevSeries(const std::function<double(double)> &f,
                                  double tolerance, double scale) {
  // Interpolation at the points of Chebyshev of the first kind: a_k =
  // (2/n) Σ_j f(t_j) T_k(t_j), halved for k = 0.
  const int n = 64;
  std::vector<double> samples(n), a(n);
  for (int j = 0; j != n; ++j)
    samples[j] = f(std::cos(M_PI * (j + 0.5) / n));
  double largest = 0.0;
  for (int k = 0; k != n; ++k) {
    double sum = 0.0;
    for (int j = 0; j != n; ++j)
      sum += samples[j] * std::cos(M_PI * k * (j + 0.5) / n);
    a[k] = (k == 0 ? 1.0 : 2.0) * sum / n;
    largest = std::max(largest, std::fabs(a[k]));
  }
  int terms = 1;
  for (int k = 0; k != n; ++k)
    if (std::fabs(a[k]) > tolerance * std::max(largest, scale))
      terms = k + 1;
  a.resize(terms);
  return a;
}

Window mdir::prolate::getWindow(double c1, int64_t points) {
  Function psi(c1);
  Window window;
  window.points = points;
  window.c1 = c1;
  double half = 0.5 * static_cast<double>(points);
  double shift = points % 2 ? 0.5 : 0.0;
  int64_t below = points / 2;
  // The series to 1e-14 of ψ(0), above the rounding of the Legendre series
  // (2e-15): the kernels in f32 and in f64
  // evaluate the same window, so that the factor B of the transform is
  // that of what they spread.
  std::vector<std::vector<double>> values, slopes;
  // The window less its value at the edge of its support, continuous there
  // (Prolate.h).
  double edge = psi.value(1.0);
  window.edge = edge;
  size_t degree = 1;
  for (int64_t j = 0; j != points; ++j) {
    double offset = shift + static_cast<double>(below - 1 - j);
    auto at = [&](double t) {
      double d = 0.5 * (t + 1.0) + offset;
      double x = d / half;
      return std::fabs(x) <= 1.0 ? psi.value(x) - edge : 0.0;
    };
    auto slope = [&](double t) {
      double d = 0.5 * (t + 1.0) + offset;
      double x = d / half;
      return std::fabs(x) <= 1.0 ? psi.derivative(x) / half : 0.0;
    };
    values.push_back(getChebyshevSeries(at, 1e-14, psi.value(0.0)));
    slopes.push_back(getChebyshevSeries(slope, 1e-14, 4.0 * psi.value(0.0) / half));
    degree = std::max({degree, values.back().size(), slopes.back().size()});
  }
  window.degree = static_cast<int64_t>(degree);
  for (int64_t j = 0; j != points; ++j)
    for (size_t k = 0; k != degree; ++k) {
      double v = k < values[j].size() ? std::fabs(values[j][k]) : 0.0;
      double s = k < slopes[j].size() ? std::fabs(slopes[j][k]) * half / 4.0 : 0.0;
      if (std::max(v, s) > 1e-8 * psi.value(0.0))
        window.degreeSingle = std::max<int64_t>(window.degreeSingle, k + 1);
    }
  for (int64_t j = 0; j != points; ++j) {
    values[j].resize(degree, 0.0);
    slopes[j].resize(degree, 0.0);
    window.values.insert(window.values.end(), values[j].begin(),
                         values[j].end());
    window.slopes.insert(window.slopes.end(), slopes[j].begin(),
                         slopes[j].end());
  }
  // φ̂₁(κ) = (P/2) ∫₋₁¹ (ψ(t) − ψ(1)) cos(κ P t / 2) dt.
  int count = std::max(200, static_cast<int>(4 * c1) + 100);
  std::vector<double> x, w;
  gaussLegendre(count, x, w);
  for (int i = 0; i != count; ++i) {
    window.transformNodes.push_back(x[i] * half);
    window.transformWeights.push_back(w[i] * (psi.value(x[i]) - edge) * half);
  }
  return window;
}

double Window::transform(double kappa) const {
  double sum = 0.0;
  for (size_t i = 0; i != transformNodes.size(); ++i)
    sum += transformWeights[i] * std::cos(kappa * transformNodes[i]);
  return sum;
}

double Window::paperTransform(double kappa) const {
  // (P/2) ∫₋₁¹ ψ(1) cos(κ P t / 2) dt = 2 ψ(1) sin(κ P / 2) / κ.
  double half = 0.5 * static_cast<double>(points);
  double box = kappa == 0.0 ? 2.0 * half : 2.0 * std::sin(kappa * half) / kappa;
  return transform(kappa) + edge * box;
}

Influence mdir::prolate::getInfluence(const Function &psi) {
  double scale = psi.eigenvalue() / (2.0 * psi.halfIntegral());
  Influence influence;
  auto x = [](double y) { return std::sqrt(std::max(0.0, 0.5 * (y + 1.0))); };
  influence.value = getChebyshevSeries(
      [&](double y) { return scale * psi.value(x(y)); }, 1e-14);
  influence.slope = getChebyshevSeries(
      [&](double y) { return scale * x(y) * psi.derivative(x(y)); }, 1e-14,
      scale * psi.value(0.0));
  size_t terms = std::max(influence.value.size(), influence.slope.size());
  influence.value.resize(terms, 0.0);
  influence.slope.resize(terms, 0.0);
  influence.terms = static_cast<int64_t>(terms);
  return influence;
}

double mdir::prolate::getAliasing(const Function &psi, const Window &window,
                                  const double (&edges)[3],
                                  const int64_t (&grid)[3], double cutoff) {
  double c = psi.bandwidth();
  double scale = psi.eigenvalue() / (2.0 * psi.halfIntegral());
  int64_t reach[3];
  std::vector<double> ratio[3];
  for (int a = 0; a != 3; ++a) {
    reach[a] = static_cast<int64_t>(
        std::floor(c * edges[a] / (2.0 * M_PI * cutoff)));
    for (int64_t k = -reach[a]; k <= reach[a]; ++k) {
      double kappa = 2.0 * M_PI * k / grid[a];
      double alias = 2.0 * M_PI * (k + grid[a]) / grid[a];
      ratio[a].push_back(std::fabs(window.paperTransform(alias) /
                                   window.paperTransform(kappa)));
    }
  }
  double total = 0.0, aliased[3] = {0.0, 0.0, 0.0};
  for (int64_t i = -reach[0]; i <= reach[0]; ++i)
    for (int64_t j = -reach[1]; j <= reach[1]; ++j)
      for (int64_t k = -reach[2]; k <= reach[2]; ++k) {
        double mx = i / edges[0], my = j / edges[1], mz = k / edges[2];
        double m2 = mx * mx + my * my + mz * mz;
        double x = 2.0 * M_PI * cutoff * std::sqrt(m2) / c;
        if (m2 == 0.0 || x >= 1.0)
          continue;
        double weight = scale * psi.value(x) / m2;
        total += weight;
        aliased[0] += weight * ratio[0][i + reach[0]];
        aliased[1] += weight * ratio[1][j + reach[1]];
        aliased[2] += weight * ratio[2][k + reach[2]];
      }
  if (total == 0.0)
    return 0.0;
  return 6.0 * std::max({aliased[0], aliased[1], aliased[2]}) / total;
}

double mdir::prolate::getSplitting(const Function &psi, Part part, double x) {
  x = std::min(std::fabs(x), 1.0);
  // Φ(x)/x = (1/C₀) ∫₀¹ ψ(xτ) dτ, by Gauss–Legendre in τ, which has no
  // cancellation at small x.
  static thread_local std::vector<double> nodes, weights;
  if (nodes.empty())
    gaussLegendre(48, nodes, weights);
  double sum = 0.0;
  for (size_t i = 0; i != nodes.size(); ++i)
    sum += 0.5 * weights[i] * psi.value(0.5 * x * (nodes[i] + 1.0));
  double near = sum / psi.halfIntegral();
  return part == Part::Near ? near : 1.0 - x * near;
}

std::vector<double> mdir::prolate::getFarSeries(const Function &psi) {
  // (1 − Φ(x))/(1 − x) = (1/C₀) ∫₀¹ ψ(x + (1 − x)τ) dτ, the mean of χ over
  // [x, 1], by Gauss–Legendre in τ: no cancellation near x = 1, where
  // 1 − Φ is of the order of ψ(1)(1 − x). Its series in t = 2x − 1 falls
  // below 10⁻¹⁴ of its largest term after 30 to 40 terms, where those of
  // the 64 points reach the noise of the quadrature; the cut keeps those
  // above it.
  static thread_local std::vector<double> nodes, weights;
  if (nodes.empty())
    gaussLegendre(64, nodes, weights);
  return getChebyshevSeries(
      [&](double t) {
        double x = 0.5 * (t + 1.0), sum = 0.0;
        for (size_t i = 0; i != nodes.size(); ++i)
          sum += 0.5 * weights[i] *
                 psi.value(x + (1.0 - x) * 0.5 * (nodes[i] + 1.0));
        return sum / psi.halfIntegral();
      },
      1e-14);
}

std::vector<double> mdir::prolate::getSplittingTable(const Function &psi,
                                                     Part part, int64_t cells,
                                                     int64_t terms) {
  std::vector<double> table(cells * terms, 0.0);
  for (int64_t k = 0; k != cells; ++k) {
    // The Chebyshev series of the cell, then its monomials in t.
    std::vector<double> a = getChebyshevSeries(
        [&](double t) {
          return getSplitting(psi, part,
                              (static_cast<double>(k) + 0.5 * (t + 1.0)) /
                                  static_cast<double>(cells));
        },
        0.0);
    a.resize(terms, 0.0);
    // T_j as monomials: T_0 = 1, T_1 = t, T_{j+1} = 2t T_j − T_{j−1}.
    std::vector<double> previous(terms, 0.0), current(terms, 0.0);
    previous[0] = 1.0;
    if (terms > 1)
      current[1] = 1.0;
    for (int64_t j = 0; j != terms; ++j) {
      const std::vector<double> &tj = j == 0 ? previous : current;
      for (int64_t i = 0; i != terms; ++i)
        table[k * terms + i] += a[j] * tj[i];
      if (j >= 1) {
        std::vector<double> next(terms, 0.0);
        for (int64_t i = 0; i + 1 < terms; ++i)
          next[i + 1] += 2.0 * current[i];
        for (int64_t i = 0; i != terms; ++i)
          next[i] -= previous[i];
        previous = current;
        current = next;
      }
    }
  }
  return table;
}

double Function::endpointDerivative(int derivative) const {
  long double result = 0.0L;
  for (int k = derivative; k < static_cast<int>(coefficients.size()); ++k) {
    long double factor = 1;
    for (int j = 1; j <= derivative; ++j)
      factor *= static_cast<long double>((k + j) * (k - j + 1)) / (2 * j);
    result += coefficients[k] * factor;
  }
  return static_cast<double>(result);
}

Influence mdir::prolate::getC2Influence(const Function &psi) {
  Influence influence;
  influence.terms = C2Terms;
  influence.value.resize(C2Terms);
  influence.slope.resize(C2Terms);
  double norm = 2 * psi.halfIntegral();
  for (int i = 0; i < C2Knots; ++i) {
    double k = static_cast<double>(i) / C2SamplesPerUnit;
    influence.value[i] = psi.transform(k) / norm;
    influence.slope[i] = psi.transformDerivative(k) / (norm * C2SamplesPerUnit);
  }
  // Integration by parts terminates at the degree of the polynomial.
  // Values multiply sin(k)/k, slopes multiply cos(k)/k².
  for (int i = 0; i < C2Basis; ++i) {
    double scale = (i % 2 ? -1 : 1) /
                   (psi.halfIntegral() * std::pow(double(C2Transition), 2 * i));
    influence.value[C2Knots + i] = scale * psi.endpointDerivative(2 * i);
    influence.slope[C2Knots + i] = scale * psi.endpointDerivative(2 * i + 1);
  }
  return influence;
}

Window mdir::prolate::getCubicWindow() {
  Window window;
  window.points = window.degree = window.degreeSingle = 4;
  for (int j = 0; j < 4; ++j) {
    auto polynomial = [j](double t, bool slope) {
      double u = 0.5 * (t + 1), v = 1 - u;
      switch (j) {
      case 0:
        return slope ? -v * v / 2 : v * v * v / 6;
      case 1:
        return slope ? (3 * u * u - 4 * u) / 2
                     : (4 - 6 * u * u + 3 * u * u * u) / 6;
      case 2:
        return slope ? (4 * v - 3 * v * v) / 2
                     : (4 - 6 * v * v + 3 * v * v * v) / 6;
      default:
        return slope ? u * u / 2 : u * u * u / 6;
      }
    };
    for (bool slope : {false, true}) {
      auto coeff =
          getChebyshevSeries([&](double t) { return polynomial(t, slope); }, 0);
      coeff.resize(4);
      auto &out = slope ? window.slopes : window.values;
      out.insert(out.end(), coeff.begin(), coeff.begin() + 4);
    }
  }
  return window;
}

std::vector<double> Function::squaredDistanceTable() const {
  // The cubic B-spline for L(t) = Phi(sqrt(t))/sqrt(t) is C² at every
  // interior knot. At t=1 it matches the value, slope and curvature of
  // 1/sqrt(t), so the direct term vanishes through its second derivative.
  constexpr int n = C2PairCells;
  constexpr double h = 1.0 / n;
  std::vector<double> controls(n + 3), upper(n);
  auto at = [&](int i) -> double & { return controls[i + 1]; };
  auto endpoint = [&](int i, double f, double df, double ddf) {
    double curvature = h * h * ddf;
    at(i - 1) = f - h * df + curvature / 3;
    at(i) = f - curvature / 6;
    at(i + 1) = f + h * df + curvature / 3;
  };
  // The second and fourth derivatives at zero follow from Legendre's equation.
  double second = 0, fourth = 0, legendreAtZero = 1;
  for (size_t k = 2; k < coefficients.size(); k += 2) {
    legendreAtZero *= -(double(k) - 1) / k;
    double lambda = double(k) * (k + 1);
    double term = lambda * coefficients[k] * legendreAtZero;
    second -= term;
    fourth += (lambda - 6) * term;
  }
  endpoint(0, value(0) / c0, second / (6 * c0), fourth / (60 * c0));
  endpoint(n, 1, -0.5, 0.75);

  // Interpolate the free interior knots: b[i-1]+4b[i]+b[i+1]=6L(i h).
  // The two nearest the endpoints are fixed by the boundary derivatives.
  for (int i = 2; i <= n - 2; ++i) {
    double x = std::sqrt(i * h);
    double diagonal = 4 - upper[i - 1];
    at(i) = (6 * integral(x) / (x * c0) - at(i - 1)) / diagonal;
    upper[i] = 1 / diagonal;
  }
  for (int i = n - 2; i >= 2; --i)
    at(i) -= upper[i] * at(i + 1);

  std::vector<double> result(4 * n);
  for (int i = 0; i < n; ++i) {
    double before = at(i) - at(i - 1), here = at(i + 1) - at(i);
    double after = at(i + 2) - at(i + 1);
    double f = at(i) + (here - before) / 6;
    double slope = (here + before) / 2;
    double quadratic = (here - before) / 2;
    double cubic = (after - 2 * here + before) / 6;
    // Recenter the monomials from [0,1] to the pair lookup's [-1,1].
    result[4 * i] = f + slope / 2 + quadratic / 4 + cubic / 8;
    result[4 * i + 1] = slope / 2 + quadratic / 2 + 3 * cubic / 8;
    result[4 * i + 2] = quadratic / 4 + 3 * cubic / 8;
    result[4 * i + 3] = cubic / 8;
  }
  return result;
}
