// The bicubic patches of correction maps (CMAP) of two dihedrals
// [MacKerell2004]. The derivatives at the points of a grid, and the patch
// of each cell, follow what sander and GROMACS compute
// (docs/design-m1.md, Section 20).

#include "mdir/Driver/Topology.h"

#include <array>

using namespace mdir::driver;

/// The derivatives at the n points of a periodic line of values `y`, with
/// the step `h`: those of the natural cubic spline through the line
/// repeated to 2n points, from −n/2 to 3n/2, at its central n points. The
/// spline is not periodic; its ends are far enough for the difference to
/// be small, but it is the one that the engines take.
static std::vector<double> getSlopes(const std::vector<double> &y,
                                     double h) {
  int n = static_cast<int>(y.size());
  int m = 2 * n;
  std::vector<double> z(m), s(m, 0.0), w(m, 0.0);
  for (int k = 0; k != m; ++k)
    z[k] = y[((k - n / 2) % n + n) % n];
  // Second derivatives, by the forward sweep and the back substitution of
  // the tridiagonal system of equal steps.
  for (int k = 1; k != m - 1; ++k) {
    double p = 0.5 * s[k - 1] + 2.0;
    s[k] = -1.0 / (2.0 * p);
    double q = (z[k + 1] - 2.0 * z[k] + z[k - 1]) / h;
    w[k] = (3.0 * q / h - 0.5 * w[k - 1]) / p;
  }
  s[m - 1] = 0.0;
  for (int k = m - 2; k >= 0; --k)
    s[k] = s[k] * s[k + 1] + w[k];
  std::vector<double> slopes(n);
  for (int j = 0; j != n; ++j) {
    int k = j + n / 2;
    slopes[j] =
        (z[k + 1] - z[k]) / h - h * s[k] / 3.0 - h * s[k + 1] / 6.0;
  }
  return slopes;
}

std::vector<double> mdir::driver::getCMapCoefficients(
    const Topology &topology) {
  int n = static_cast<int>(topology.cmapResolution);
  double h = 360.0 / n;
  std::vector<double> coefficients;
  coefficients.reserve(topology.cmapGrids.size() * n * n * 16);
  for (const std::vector<double> &grid : topology.cmapGrids) {
    auto at = [&](const std::vector<double> &values, int a, int b) {
      return values[(a % n) * n + (b % n)];
    };
    // dE/dφ along each column, dE/dψ along each row, and the cross
    // derivative along the rows of dE/dφ, per degree.
    std::vector<double> phi(n * n), psi(n * n), cross(n * n);
    std::vector<double> line(n);
    for (int b = 0; b != n; ++b) {
      for (int a = 0; a != n; ++a)
        line[a] = grid[a * n + b];
      std::vector<double> slopes = getSlopes(line, h);
      for (int a = 0; a != n; ++a)
        phi[a * n + b] = slopes[a];
    }
    for (int a = 0; a != n; ++a) {
      std::vector<double> row(grid.begin() + a * n, grid.begin() + a * n + n);
      std::vector<double> slopes = getSlopes(row, h);
      std::vector<double> phiRow(phi.begin() + a * n,
                                 phi.begin() + a * n + n);
      std::vector<double> crossSlopes = getSlopes(phiRow, h);
      for (int b = 0; b != n; ++b) {
        psi[a * n + b] = slopes[b];
        cross[a * n + b] = crossSlopes[b];
      }
    }

    // The Hermite bicubic of each cell, C = M F Mᵀ, from the values and
    // the derivatives per cell at its corners (t, u) in {0, 1}².
    static const double M[4][4] = {
        {1, 0, 0, 0}, {0, 0, 1, 0}, {-3, 3, -2, -1}, {2, -2, 1, 1}};
    for (int a = 0; a != n; ++a)
      for (int b = 0; b != n; ++b) {
        double F[4][4];
        for (int dt = 0; dt != 2; ++dt)
          for (int du = 0; du != 2; ++du) {
            F[dt][du] = at(grid, a + dt, b + du);
            F[dt][2 + du] = h * at(psi, a + dt, b + du);
            F[2 + dt][du] = h * at(phi, a + dt, b + du);
            F[2 + dt][2 + du] = h * h * at(cross, a + dt, b + du);
          }
        double MF[4][4];
        for (int i = 0; i != 4; ++i)
          for (int j = 0; j != 4; ++j) {
            MF[i][j] = 0.0;
            for (int k = 0; k != 4; ++k)
              MF[i][j] += M[i][k] * F[k][j];
          }
        for (int i = 0; i != 4; ++i)
          for (int j = 0; j != 4; ++j) {
            double c = 0.0;
            for (int k = 0; k != 4; ++k)
              c += MF[i][k] * M[j][k];
            coefficients.push_back(c);
          }
      }
  }
  return coefficients;
}
