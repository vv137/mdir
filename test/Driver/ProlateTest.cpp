//===- ProlateTest.cpp - ψ₀ᶜ of the host against an independent reference ===//
//
// The reference is the eigenfunction of the integral operator itself,
// discretized by Gauss–Legendre at 96 nodes in mpmath at 40 digits
// (scripts/benchmarks/prolate-ewald/pswf_reference.py), a method other than
// the Legendre series of the host. Prints each quantity, its reference, and
// its difference, and "FAIL" where the difference exceeds 1e-12 relative to
// the largest value of ψ (D[prolate-ewald]).
//
//===----------------------------------------------------------------------===//

#include "mdir/Driver/Prolate.h"

#include <cmath>
#include <cstdio>

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
  std::printf("%d failures\n", failures);
  return failures ? 1 : 0;
}
