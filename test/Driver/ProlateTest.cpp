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
#include <cstring>
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
  // The B-spline window (D444): χ̂ of the constrained
  // density as the kernels evaluate it, against the quadrature of the
  // transform below the transition and continuous at it; and the mesh error
  // estimate against `spline_grid.py functional`
  // (scripts/benchmarks/prolate-ewald), which sums the whole grid with a
  // density and a transform of its own, to 1e-6.
  {
    Function psi(9.5392, 2, 8);
    mdir::prolate::Influence influence =
        mdir::prolate::getPolynomialInfluence(psi);
    double worst = 0.0;
    for (double k = 0.01; k < 300.0; k *= 1.01)
      worst = std::max(
          worst, std::fabs(mdir::prolate::evaluatePolynomialInfluence(influence, k) -
                           psi.transform(k) / (2.0 * psi.halfIntegral())));
    double below = mdir::prolate::evaluatePolynomialInfluence(influence, 512.0 - 1e-9);
    double above = mdir::prolate::evaluatePolynomialInfluence(influence, 512.0);
    bool ok = worst < 1e-8 && std::fabs(below - above) < 1e-15;
    failures += !ok;
    std::printf("spline spectrum: largest difference %.1e, step at the transition "
                "%.1e %s\n",
                worst, std::fabs(below - above), ok ? "ok" : "FAIL");
    struct Case {
      double c;
      int continuity, terms;
      int64_t grid[3];
      double edges[3], cutoff, reference;
    };
    const Case cases[] = {
        {9.5392, 2, 8, {24, 24, 24}, {62.23, 62.23, 62.23}, 8.0, 9.776661733777e-03},
        {14.471, 2, 8, {20, 18, 16}, {30.1, 27.4, 24.9}, 9.0, 1.575745593247e-03},
        {12.024, 3, 8, {16, 16, 16}, {30.0, 30.0, 30.0}, 9.0, 3.157652485010e-03},
        {9.5392, 1, 8, {20, 20, 18}, {40.0, 40.0, 36.0}, 8.0, 3.540370751036e-03},
        {9.5392, 2, 40, {20, 20, 18}, {40.0, 40.0, 36.0}, 8.0, 3.508506844097e-03},
    };
    for (const Case &k : cases) {
      Function split(k.c, k.continuity, k.terms);
      mdir::prolate::Influence table = mdir::prolate::getPolynomialInfluence(split);
      double value = mdir::prolate::getSplineMeshError(
          [&](double x) { return mdir::prolate::evaluatePolynomialInfluence(table, x); },
          4, k.edges, k.grid, k.cutoff);
      bool close = std::fabs(value / k.reference - 1.0) < 1e-6;
      failures += !close;
      std::printf("spline mesh error C%d N=%d c %g %lldx%lldx%lld: %.10e reference "
                  "%.10e %s\n",
                  k.continuity, k.terms, k.c, static_cast<long long>(k.grid[0]),
                  static_cast<long long>(k.grid[1]),
                  static_cast<long long>(k.grid[2]), value, k.reference,
                  close ? "ok" : "FAIL");
    }
    // A grid large enough to be summed on threads: the same bits on one
    // thread and on four.
    Function split(9.5392, 2, 8);
    mdir::prolate::Influence table = mdir::prolate::getPolynomialInfluence(split);
    auto spectrum = [&](double x) {
      return mdir::prolate::evaluatePolynomialInfluence(table, x);
    };
    const double edges[3] = {62.23, 62.23, 62.23};
    const int64_t grid[3] = {160, 160, 160};
    double one = mdir::prolate::getSplineMeshError(spectrum, 4, edges, grid, 8.0, 1);
    double four = mdir::prolate::getSplineMeshError(spectrum, 4, edges, grid, 8.0, 4);
    bool same = std::memcmp(&one, &four, sizeof one) == 0;
    failures += !same;
    std::printf("spline mesh error 160x160x160 on 1 and 4 threads: %.17g %.17g %s\n",
                one, four, same ? "ok" : "FAIL");
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
  // (1 − Φ(x))/(1 − x) as a Chebyshev series in t = 2x − 1
  // (D280), by Clenshaw's recurrence of md.chebyshev:
  // (1 − x) times it against 1 − Φ by quadrature at 1000 points, to 2e-14
  // (its cut at 1e-14 of the largest term, and the noise of the
  // quadrature); at x = 1 against its limit χ(1) = ψ(1)/C₀ to 1e-14, the
  // noise of ψ (relative to χ(1), 4e-11 at ε = 1e-5 and 3e-7 at 1e-9,
  // printed); and the derivative of
  // (1 − x) P, −P + (1 − x) 2 P'(t), against −χ(x), to 2e-11 (the first
  // term cut, times k² of its derivative), of values of order 1.
  for (double epsilon : {1e-3, 1e-5, 1e-9}) {
    using mdir::prolate::Part;
    Function psi(mdir::prolate::bandwidthForTolerance(epsilon));
    std::vector<double> series = mdir::prolate::getFarSeries(psi);
    llvm::SmallVector<double> slope =
        mdir::md::getChebyshevDerivative(series);
    double worst = 0.0, worstSlope = 0.0;
    for (int i = 0; i != 1000; ++i) {
      double x = (i + 0.37) / 1000.0, t = 2 * x - 1;
      double mean = mdir::md::evaluateChebyshev(series, t);
      double far = mdir::prolate::getSplitting(psi, Part::Far, x);
      worst = std::max(worst, std::fabs((1 - x) * mean - far));
      double chi = psi.value(x) / psi.halfIntegral();
      worstSlope = std::max(
          worstSlope,
          std::fabs(-mean + (1 - x) * 2 * mdir::md::evaluateChebyshev(slope, t) +
                    chi));
    }
    double edge = psi.value(1.0) / psi.halfIntegral();
    double atEdge = std::fabs(mdir::md::evaluateChebyshev(series, 1.0) - edge);
    bool ok = worst <= 2e-14 && atEdge <= 1e-14 && worstSlope <= 2e-11;
    failures += !ok;
    std::printf("series of (1 - Phi)/(1 - x) at epsilon %g: %zu terms, "
                "largest difference %.1e, at x = 1 %.1e (%.1e of chi(1)), of "
                "the slope %.1e %s\n",
                epsilon, series.size(), worst, atEdge, atEdge / edge, worstSlope,
                ok ? "ok" : "FAIL");
  }
  for (double c : {9.5392, 12.024, 14.471}) {
    Function original(c), defaultSplit(c, 0, 0), finiteSplit(c, 0, 40);
    double exact = 0, finite = 0;
    for (int i = 0; i <= 512; ++i) {
      double x = i / 512.;
      double reference = original.value(x) / original.halfIntegral();
      exact = std::max(exact, std::abs(defaultSplit.value(x) / defaultSplit.halfIntegral() - reference));
      finite = std::max(finite, std::abs(finiteSplit.value(x) / finiteSplit.halfIntegral() - reference));
    }
    std::printf("k=0 c=%g default max error %.8g, finite N=40 %.8g\n", c, exact, finite);
    failures += exact != 0 || finite > 1e-7;
  }
  for (double c : {9.5392, 12.024, 14.471}) {
    for (auto [k, terms] : {std::pair{1, 8}, {2, 8}, {2, 40}, {3, 8}, {3, 40}}) {
      Function split(c, k, terms);
      auto polynomial = split.squaredDistanceSeries();
      llvm::SmallVector<double> series(polynomial.begin(), polynomial.end());
      double derivatives[4];
      for (int i = 0; i < 4; ++i) {
        derivatives[i] = mdir::md::evaluateChebyshev(series, 1);
        series = mdir::md::getChebyshevDerivative(series);
      }
      // Interior limits of 1/r - L(2(r/rc)^2-1)/rc at rc = 8 angstrom.
      constexpr double rc = 8;
      auto [v, d, dd, ddd] = derivatives;
      double jumps[] = {(1 - v) / rc, (-1 - 4 * d) / (rc * rc),
                        (2 - 4 * d - 16 * dd) / std::pow(rc, 3),
                        (-6 - 48 * dd - 64 * ddd) / std::pow(rc, 4)};
      bool ok = std::abs(jumps[0]) < 1e-14 && std::abs(jumps[1]) < 1e-13 &&
                (k < 2 || std::abs(jumps[2]) < 1e-12) && (k < 3 || std::abs(jumps[3]) < 1e-10);
      failures += !ok;
      std::printf("C%d cutoff N=%d at c %g: %.2e %.2e %.2e %.2e %s\n",
                  k, terms, c, jumps[0], jumps[1], jumps[2], jumps[3], ok ? "ok" : "FAIL");
    }
  }
  std::printf("%d failures\n", failures);
  return failures ? 1 : 0;
}
