//===- ProlateTest.cpp - ψ₀ᶜ of the host against an independent reference ===//
//
// The reference is the eigenfunction of the integral operator itself,
// discretized by Gauss–Legendre at 96 nodes in mpmath at 40 digits
// (scripts/benchmarks/prolate-ewald/pswf_reference.py), a method other than
// the Legendre series of the host. Prints each quantity, its reference, and
// its difference, and "FAIL" where the difference exceeds 1e-12 relative to
// the largest value of ψ (D266).
//
//===----------------------------------------------------------------------===//

#include "mdir/Driver/Prolate.h"
#include "mdir/Dialect/MD/MDOps.h"

#include <cmath>
#include <cstdio>
#include <vector>

using mdir::prolate::Function;

namespace {
struct Reference {
  double c, lambda, c0, chihatHalf;
  double psi[5]; // at 0, 0.25, 0.5, 0.75, 1
};
// pswf_reference.py 9.5392 14.4712
const Reference references[] = {
    {9.5392, 0.81158485406718964, 0.52984405028539672, 0.31101324207689962,
     {1.3057021644258824, 0.98878186952717609, 0.40609066334491876,
      0.070503363839893041, 0.00099995621449380474}},
    {14.4712, 0.6589275429652098, 0.47933652511970996, 0.16041053206671305,
     {1.4548990408343519, 0.94178154243812036, 0.23338112924358885,
      0.014726312797244225, 1.000023715139208e-5}},
};
int failures = 0;
void check(const char *what, double c, double value, double reference,
           double scale) {
  double difference = std::fabs(value - reference) / scale;
  bool ok = difference <= 1e-12;
  failures += !ok;
  std::printf("c %.4f %s %.17g reference %.17g difference %.1e %s\n", c, what,
              value, reference, difference, ok ? "ok" : "FAIL");
}
} // namespace

int main() {
  for (const Reference &r : references) {
    Function psi(r.c);
    check("lambda", r.c, psi.eigenvalue(), r.lambda, 1.0);
    check("C0", r.c, psi.halfIntegral(), r.c0, 1.0);
    const double xs[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
    for (int i = 0; i != 5; ++i) {
      char name[32];
      std::snprintf(name, sizeof name, "psi(%.2f)", xs[i]);
      check(name, r.c, psi.value(xs[i]), r.psi[i], r.psi[0]);
    }
    // χ̂(c/2) = λ₀ ψ(1/2) / (2 C₀), and by quadrature of the transform.
    check("chihat(c/2)", r.c,
          psi.eigenvalue() * psi.value(0.5) / (2 * psi.halfIntegral()),
          r.chihatHalf, 1.0);
    check("chihat(c/2) by quadrature", r.c,
          psi.transform(0.5 * r.c) / (2 * psi.halfIntegral()), r.chihatHalf,
          1.0);
    // The derivative against a central difference, and the integral.
    double h = 1e-5, x = 0.6;
    check("psi'(0.6) against a difference", r.c, psi.derivative(x),
          (psi.value(x + h) - psi.value(x - h)) / (2 * h),
          r.psi[0] * 1e3); // the difference is good to 1e-10 of ψ(0)
  }
  // The bandwidths of Table 2 of Liang et al. (2026), to their digits, at
  // the tolerances of whole decades (its other columns are those of a
  // table in steps of 0.1 decade, 10.290 at 10^-3.3 rather than 5e-4).
  const double tolerances[] = {1e-3, 1e-4, 1e-5};
  const double table[] = {9.5392, 12.024, 14.471};
  for (int i = 0; i != 3; ++i) {
    double c = mdir::prolate::bandwidthForTolerance(tolerances[i]);
    double digits = table[i] < 10 ? 1e-4 : 1e-3;
    bool ok = std::fabs(c - table[i]) <= 0.5 * digits;
    failures += !ok;
    std::printf("epsilon %g bandwidth %.6f table %.4f %s\n", tolerances[i], c,
                table[i], ok ? "ok" : "FAIL");
  }
  // The window and the influence as Chebyshev series, against ψ itself;
  // and the aliasing estimate on a cube of JAC's edge, 6.223 nm, with
  // r_c = 0.8 nm, ε = 1e-5, P = 6, against alias.py of the measurement of
  // #334 (5.20e-6 at 48 points, 6.82e-5 at 36).
  {
    double c = mdir::prolate::bandwidthForTolerance(1e-5);
    double c1 = mdir::prolate::bandwidthForTolerance(5e-6);
    Function psi(c), window1(c1);
    mdir::prolate::Window window = mdir::prolate::getWindow(c1, 6);
    double worst = 0.0;
    for (int j = 0; j != 6; ++j)
      for (double t : {-1.0, -0.3, 0.4, 1.0}) {
        // T_k(t) by the recurrence.
        double t0 = 1.0, t1 = t, sum = window.values[j * window.degree];
        for (int64_t k = 1; k < window.degree; ++k) {
          sum += window.values[j * window.degree + k] * t1;
          double t2 = 2 * t * t1 - t0;
          t0 = t1;
          t1 = t2;
        }
        // The window less its value at the edge of its support.
        double x = (0.5 * (t + 1.0) + 2.0 - j) / 3.0;
        double expected =
            std::fabs(x) <= 1.0 ? window1.value(x) - window1.value(1.0) : 0.0;
        worst = std::max(worst, std::fabs(sum - expected));
      }
    bool ok = worst <= 1e-14 * window1.value(0.0) && window.degree <= 20;
    failures += !ok;
    std::printf("window of %lld terms, largest difference %.1e %s\n",
                static_cast<long long>(window.degree), worst,
                ok ? "ok" : "FAIL");
    const double edges[3] = {6.223, 6.223, 6.223};
    const int64_t fine[3] = {48, 48, 48}, coarse[3] = {36, 36, 36};
    double a48 = mdir::prolate::getAliasing(psi, window, edges, fine, 0.8);
    double a36 = mdir::prolate::getAliasing(psi, window, edges, coarse, 0.8);
    ok = std::fabs(a48 / 5.20e-6 - 1) < 0.01 && std::fabs(a36 / 6.82e-5 - 1) < 0.01;
    failures += !ok;
    std::printf("aliasing %.3g at 48, %.3g at 36 %s\n", a48, a36,
                ok ? "ok" : "FAIL");
  }
  // The tables of the splitting, 64 cells of 10 terms, against the
  // functions at 1000 points; Φ by quadrature against the Legendre
  // integral.
  {
    using mdir::prolate::Part;
    Function psi(mdir::prolate::bandwidthForTolerance(1e-5));
    double phi = 0.5 * mdir::prolate::getSplitting(psi, Part::Near, 0.5);
    check("Phi(0.5)", psi.bandwidth(), phi,
          psi.integral(0.5) / psi.halfIntegral(), 1.0);
    for (Part part : {Part::Far, Part::Near}) {
      std::vector<double> table =
          mdir::prolate::getSplittingTable(psi, part, 64, 10);
      double worst = 0.0;
      for (int i = 0; i != 1000; ++i) {
        double x = (i + 0.37) / 1000.0, u = x * 64;
        int k = static_cast<int>(u);
        double t = 2 * (u - k) - 1, p = 0.0;
        for (int j = 9; j >= 0; --j)
          p = p * t + table[k * 10 + j];
        worst = std::max(worst,
                         std::fabs(p - mdir::prolate::getSplitting(psi, part, x)));
      }
      bool ok = worst <= 1e-14 * (part == Part::Far ? 1.0 : 3.0);
      failures += !ok;
      std::printf("table of %s: largest difference %.1e %s\n",
                  part == Part::Far ? "1 - Phi" : "Phi/x", worst,
                  ok ? "ok" : "FAIL");
    }
  }
  // Φ(x)/x as a Chebyshev series in y = 2x² − 1 (D[prolate-pair-table]),
  // by Clenshaw's recurrence of md.chebyshev, against the quadrature at
  // 1000 points, to 2e-14 (its cut at 1e-14 of the largest term, and the
  // noise of the quadrature); the series of its derivative in y, times
  // dy/dx = 4x, against (χ(x) − Φ(x)/x)/x, to 2e-11 (the first term cut,
  // times k² of its derivative), of values of order 1.
  for (double epsilon : {1e-3, 1e-5, 1e-9}) {
    using mdir::prolate::Part;
    Function psi(mdir::prolate::bandwidthForTolerance(epsilon));
    std::vector<double> series = mdir::prolate::getNearSeries(psi);
    llvm::SmallVector<double> slope =
        mdir::md::getChebyshevDerivative(series);
    double worst = 0.0, worstSlope = 0.0;
    for (int i = 0; i != 1000; ++i) {
      double x = (i + 0.37) / 1000.0, y = 2 * x * x - 1;
      double near = mdir::prolate::getSplitting(psi, Part::Near, x);
      worst = std::max(
          worst, std::fabs(mdir::md::evaluateChebyshev(series, y) - near));
      if (x < 0.05)
        continue;
      double chi = psi.value(x) / psi.halfIntegral();
      worstSlope = std::max(
          worstSlope,
          std::fabs(4 * x * mdir::md::evaluateChebyshev(slope, y) -
                    (chi - near) / x));
    }
    bool ok = worst <= 2e-14 && worstSlope <= 2e-11;
    failures += !ok;
    std::printf("series of Phi/x at epsilon %g: %zu terms, largest difference "
                "%.1e, of the slope %.1e %s\n",
                epsilon, series.size(), worst, worstSlope, ok ? "ok" : "FAIL");
  }
  std::printf("%d failures\n", failures);
  return failures ? 1 : 0;
}
