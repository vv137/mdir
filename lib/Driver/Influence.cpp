//===- Influence.cpp - Influence functions of mesh Ewald ------------------===//
//
// See Influence.h and docs/pme-m1.md, Section 1.1 (D[p3m-influence]).
//
//===----------------------------------------------------------------------===//

#include "mdir/Driver/Influence.h"

#include <algorithm>
#include <cmath>

using namespace mdir::influence;

Window mdir::influence::getBSpline(int64_t order) {
  return {[order](double t) {
    if (t == 0.0)
      return 1.0;
    // sin(π t) is not exactly 0 at an integer t in f64; Û is.
    if (t == std::nearbyint(t))
      return 0.0;
    double x = M_PI * t;
    return std::pow(std::sin(x) / x, static_cast<double>(order));
  }};
}

Splitting mdir::influence::getGaussian(double beta) {
  double gauss = M_PI * M_PI / (beta * beta);
  return {[gauss](double msq) { return std::exp(-gauss * msq); },
          /*separable=*/true, gauss};
}

int64_t mdir::influence::getSigned(int64_t k, int64_t count) {
  return k <= count / 2 ? k : k - count;
}

namespace {
/// The wave vector of the alias ν = n + J K.
std::array<double, 3> getWave(const Basis &basis,
                              const std::array<double, 3> &nu) {
  std::array<double, 3> m{};
  for (int a = 0; a != 3; ++a)
    for (int c = 0; c != 3; ++c)
      m[c] += nu[a] * basis[a][c];
  return m;
}
} // namespace

Point mdir::influence::getForceOptimal(const Window &window,
                                       const Splitting &splitting,
                                       const Basis &basis,
                                       const std::array<int64_t, 3> &grid,
                                       const std::array<int64_t, 3> &n,
                                       int64_t aliases) {
  Point point;
  if (n[0] == 0 && n[1] == 0 && n[2] == 0)
    return point;
  double sw = 0.0, swm2 = 0.0, sa = 0.0;
  std::array<std::array<double, 3>, 3> am{}, wm{};
  for (int64_t j0 = -aliases; j0 <= aliases; ++j0)
    for (int64_t j1 = -aliases; j1 <= aliases; ++j1)
      for (int64_t j2 = -aliases; j2 <= aliases; ++j2) {
        std::array<int64_t, 3> j = {j0, j1, j2};
        std::array<double, 3> nu{};
        double w = 1.0;
        for (int a = 0; a != 3; ++a) {
          nu[a] = static_cast<double>(n[a] + j[a] * grid[a]);
          double u = window.transform(nu[a] / static_cast<double>(grid[a]));
          w *= u * u;
        }
        if (w == 0.0)
          continue;
        std::array<double, 3> m = getWave(basis, nu);
        double msq = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
        double a = w * splitting.weighted(msq);
        sw += w;
        swm2 += w * msq;
        sa += a;
        for (int p = 0; p != 3; ++p)
          for (int q = 0; q != 3; ++q) {
            am[p][q] += a * m[p] * m[q];
            wm[p][q] += w * m[p] * m[q];
          }
      }
  point.value = sa / (sw * swm2);
  for (int p = 0; p != 3; ++p)
    for (int q = 0; q != 3; ++q) {
      point.gaussian[p][q] = sa > 0.0 ? am[p][q] / sa : 0.0;
      point.plain[p][q] = wm[p][q] / sw;
    }
  return point;
}

double mdir::influence::getEnergyOptimal(const Window &window,
                                         const Splitting &splitting,
                                         const Basis &basis,
                                         const std::array<int64_t, 3> &grid,
                                         const std::array<int64_t, 3> &n,
                                         int64_t aliases) {
  if (n[0] == 0 && n[1] == 0 && n[2] == 0)
    return 0.0;
  // The sum over J ∈ ℤ³ of a product over the axes is the product of the
  // sums along each axis.
  double ratio = 1.0;
  for (int a = 0; a != 3; ++a) {
    double k = static_cast<double>(grid[a]);
    double u0 = window.transform(static_cast<double>(n[a]) / k);
    double sum = 0.0;
    for (int64_t j = -aliases; j <= aliases; ++j) {
      double u = window.transform(static_cast<double>(n[a] + j * grid[a]) / k);
      sum += u * u;
    }
    ratio *= u0 * u0 / (sum * sum);
  }
  std::array<double, 3> nu = {static_cast<double>(n[0]),
                              static_cast<double>(n[1]),
                              static_cast<double>(n[2])};
  std::array<double, 3> m = getWave(basis, nu);
  double msq = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
  return ratio * splitting.weighted(msq) / msq;
}

EdgeWeights mdir::influence::getEdgeWeights(const Window &window,
                                            int64_t count, int64_t aliases) {
  EdgeWeights rows;
  rows.inverse.assign(count, 0.0);
  rows.weights.assign((2 * aliases + 1) * count, 0.0);
  std::vector<double> squares(2 * aliases + 1);
  for (int64_t k = 0; k != count; ++k) {
    int64_t n = getSigned(k, count);
    double sum = 0.0;
    for (int64_t j = -aliases; j <= aliases; ++j) {
      double u = window.transform(static_cast<double>(n + j * count) /
                                  static_cast<double>(count));
      squares[j + aliases] = u * u;
      sum += u * u;
    }
    rows.inverse[k] = 1.0 / sum;
    for (int64_t j = -aliases; j <= aliases; ++j)
      rows.weights[(j + aliases) * count + k] = squares[j + aliases] / sum;
  }
  return rows;
}

EdgeSums mdir::influence::getEdgeSums(const EdgeWeights &rows, int64_t count,
                                      int64_t aliases, int64_t k,
                                      double length, double gauss) {
  EdgeSums sums;
  int64_t n = getSigned(k, count);
  double sg = 0.0, sgm = 0.0, sgm2 = 0.0;
  for (int64_t j = -aliases; j <= aliases; ++j) {
    double u = rows.weights[(j + aliases) * count + k];
    double mu = static_cast<double>(n + j * count) / length;
    double g = std::exp(-gauss * mu * mu);
    sums.mean += u * mu;
    sums.square += u * mu * mu;
    sg += u * g;
    sgm += u * g * mu;
    sgm2 += u * g * mu * mu;
  }
  sums.factor = sg * rows.inverse[k];
  if (sg > 0.0) {
    sums.gaussMean = sgm / sg;
    sums.gaussSquare = sgm2 / sg;
  }
  return sums;
}

std::vector<double> mdir::influence::getSplineModuli(int64_t count,
                                                     int64_t order) {
  // M_n at 1 .. n − 1, from the recursion M_n(x) = (x M_{n−1}(x) + (n − x)
  // M_{n−1}(x − 1)) / (n − 1), with M_2(x) = 1 − |x − 1|.
  std::vector<double> m(order + 1, 0.0), next(order + 1, 0.0);
  m[1] = 1.0;
  for (int64_t n = 3; n <= order; ++n) {
    std::fill(next.begin(), next.end(), 0.0);
    for (int64_t x = 1; x < n; ++x)
      next[x] = (x * m[x] + (n - x) * m[x - 1]) / (n - 1);
    m.swap(next);
  }
  std::vector<double> moduli(count);
  for (int64_t k = 0; k != count; ++k) {
    double re = 0.0, im = 0.0;
    for (int64_t j = 0; j <= order - 2; ++j) {
      double angle = 2.0 * M_PI * k * j / count;
      re += m[j + 1] * std::cos(angle);
      im += m[j + 1] * std::sin(angle);
    }
    moduli[k] = 1.0 / (re * re + im * im);
  }
  return moduli;
}

std::vector<double> mdir::influence::getAliasFactors(int64_t count,
                                                     int64_t order) {
  std::vector<double> factors(count, 1.0);
  for (int64_t k = 0; k != count; ++k) {
    int64_t m = k <= count / 2 ? k : k - count;
    if (m == 0)
      continue;
    double x = M_PI * static_cast<double>(m) / static_cast<double>(count);
    double single = 1.0, twice = 1.0;
    for (int64_t j = 1; j <= 50; ++j)
      for (double sign : {1.0, -1.0}) {
        double s = x / (x + sign * M_PI * j);
        single += std::pow(s, order);
        twice += std::pow(s, 2 * order);
      }
    double lambda = single / twice;
    factors[k] = lambda * lambda;
  }
  return factors;
}
