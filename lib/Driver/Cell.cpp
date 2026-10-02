// The periodic cell of a run (docs/triclinic-m2.md, Section 1).

#include "mdir/Driver/Cell.h"

#include <cmath>

using namespace mdir::driver;

namespace {

constexpr double radiansPerDegree = M_PI / 180.0;
/// How far past a bound of the reduced form a cell may be before it is
/// reduced: the rounding of the numbers that describe it.
constexpr double tolerance = 1.0e-6;

using Matrix = std::array<std::array<double, 3>, 3>;

Matrix multiply(const Matrix &a, const Matrix &b) {
  Matrix c{};
  for (int i = 0; i != 3; ++i)
    for (int j = 0; j != 3; ++j)
      for (int k = 0; k != 3; ++k)
        c[i][j] += a[i][k] * b[k][j];
  return c;
}

Matrix transpose(const Matrix &a) {
  Matrix t{};
  for (int i = 0; i != 3; ++i)
    for (int j = 0; j != 3; ++j)
      t[i][j] = a[j][i];
  return t;
}

/// The eigenvalues and the eigenvectors (columns of `v`) of the symmetric
/// matrix `a`, by Jacobi rotations.
void diagonalize(Matrix a, std::array<double, 3> &values, Matrix &v) {
  v = Matrix{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  for (int sweep = 0; sweep != 50; ++sweep) {
    double off = std::fabs(a[0][1]) + std::fabs(a[0][2]) + std::fabs(a[1][2]);
    if (off < 1e-300)
      break;
    for (int p = 0; p != 2; ++p)
      for (int q = p + 1; q != 3; ++q) {
        if (a[p][q] == 0.0)
          continue;
        double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        double t = (theta >= 0 ? 1.0 : -1.0) /
                   (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (int k = 0; k != 3; ++k) {
          double akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k != 3; ++k) {
          double apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k != 3; ++k) {
          double vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
  }
  for (int i = 0; i != 3; ++i)
    values[i] = a[i][i];
}

} // namespace

std::array<std::array<double, 3>, 3> Cell::getVectors() const {
  return {{{diagonal[0], 0.0, 0.0},
           {tilt[0], diagonal[1], 0.0},
           {tilt[1], tilt[2], diagonal[2]}}};
}

std::array<double, 6> Cell::getLengthsAndAngles() const {
  Matrix h = getVectors();
  auto dot = [&](int i, int j) {
    return h[i][0] * h[j][0] + h[i][1] * h[j][1] + h[i][2] * h[j][2];
  };
  double a = std::sqrt(dot(0, 0)), b = std::sqrt(dot(1, 1)),
         c = std::sqrt(dot(2, 2));
  auto angle = [&](int i, int j, double li, double lj) {
    return std::acos(dot(i, j) / (li * lj)) / radiansPerDegree;
  };
  return {a, b, c, angle(1, 2, b, c), angle(0, 2, a, c), angle(0, 1, a, b)};
}

llvm::Expected<Cell> mdir::driver::makeCell(double a, double b, double c,
                                            double alpha, double beta,
                                            double gamma) {
  auto fail = [](const llvm::Twine &message) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                   message.str().c_str());
  };
  if (!(a > 0.0 && b > 0.0 && c > 0.0))
    return fail("the lengths of a cell must be positive");
  for (double angle : {alpha, beta, gamma})
    if (!(angle > 0.0 && angle < 180.0))
      return fail("the angles of a cell must be between 0 and 180 degrees");
  Cell cell;
  // Right angles are exact, so that a rectangular cell has no tilt at all.
  auto cosine = [](double degrees) {
    return degrees == 90.0 ? 0.0 : std::cos(degrees * radiansPerDegree);
  };
  double ca = cosine(alpha), cb = cosine(beta), cg = cosine(gamma);
  double sg = std::sqrt(1.0 - cg * cg);
  cell.diagonal[0] = a;
  cell.tilt[0] = b * cg;
  cell.diagonal[1] = b * sg;
  cell.tilt[1] = c * cb;
  cell.tilt[2] = c * (ca - cb * cg) / sg;
  double z2 = c * c - cell.tilt[1] * cell.tilt[1] - cell.tilt[2] * cell.tilt[2];
  // Angles whose cell has no volume, or none to the rounding, make none.
  if (!(z2 > 1.0e-12 * c * c))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the angles %g, %g, %g do not make a cell",
                                   alpha, beta, gamma);
  cell.diagonal[2] = std::sqrt(z2);
  if (llvm::Error error = reduceCell(cell))
    return std::move(error);
  return cell;
}

llvm::Error mdir::driver::reduceCell(Cell &cell) {
  double &bx = cell.tilt[0], &cx = cell.tilt[1], &cy = cell.tilt[2];
  double ax = cell.diagonal[0], by = cell.diagonal[1];
  auto past = [](double value, double bound) {
    return std::fabs(value) > 0.5 * bound * (1.0 + tolerance);
  };
  // c by b, then c by a, then b by a, as OpenMM and GROMACS reduce.
  if (past(cy, by)) {
    double n = std::round(cy / by);
    cx -= n * bx;
    cy -= n * by;
  }
  if (past(cx, ax))
    cx -= std::round(cx / ax) * ax;
  if (past(bx, ax))
    bx -= std::round(bx / ax) * ax;
  if (past(cy, by) || past(cx, ax) || past(bx, ax))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the cell cannot be reduced in one pass");
  return llvm::Error::success();
}

std::array<double, 9> mdir::driver::getSymmetricFrameRotation(
    const Cell &cell) {
  if (cell.isOrthorhombic())
    return {1, 0, 0, 0, 1, 0, 0, 0, 1};
  // G = H Hᵀ; H_s = G^(1/2) = V Λ^(1/2) Vᵀ; R = H⁻¹ H_s.
  Matrix h = cell.getVectors();
  Matrix g = multiply(h, transpose(h));
  std::array<double, 3> lambda;
  Matrix v;
  diagonalize(g, lambda, v);
  Matrix root{};
  for (int i = 0; i != 3; ++i)
    for (int j = 0; j != 3; ++j)
      for (int k = 0; k != 3; ++k)
        root[i][j] += v[i][k] * std::sqrt(lambda[k]) * v[j][k];
  // The inverse of the lower-triangular H by substitution.
  Matrix inverse{};
  double ax = h[0][0], bx = h[1][0], by = h[1][1], cx = h[2][0], cy = h[2][1],
         cz = h[2][2];
  inverse[0][0] = 1.0 / ax;
  inverse[1][0] = -bx / (ax * by);
  inverse[1][1] = 1.0 / by;
  inverse[2][0] = (bx * cy - by * cx) / (ax * by * cz);
  inverse[2][1] = -cy / (by * cz);
  inverse[2][2] = 1.0 / cz;
  Matrix r = multiply(inverse, root);
  return {r[0][0], r[0][1], r[0][2], r[1][0], r[1][1],
          r[1][2], r[2][0], r[2][1], r[2][2]};
}
