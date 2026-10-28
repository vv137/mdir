// Shared scalar Lennard-Jones emitter for primary and special 1-4 pairs.
#include "BuilderImpl.h"
using namespace mdir::driver;
using namespace mdir::driver::builder;

// 4ε (σ¹² Φ₁₂(r) − σ⁶ Φ₆(r)), with Φₙ = r⁻ⁿ truncated as `modifier`
// says, with rc the cutoff and rs the switching distance:
//
//   none             r⁻ⁿ
//   potential shift  r⁻ⁿ − rc⁻ⁿ
//   force switch     r⁻ⁿ − Cₙ − [r > rs] (Aₙ/3 (r − rs)³ + Bₙ/4 (r − rs)⁴),
//                    a cubic polynomial added to the force, so that it and
//                    its derivative vanish at rc [GromacsManual2025]
//   power force      r⁻ⁿ − (rs rc)^(−n/2)               for r <= rs
//   switch           kₙ (r^(−n/2) − rc^(−n/2))²          otherwise,
//                    kₙ = rc^(n/2) / (rc^(n/2) − rs^(n/2)): the force of
//                    each power times a switch linear in r^(n/2)
//                    [Steinbach1994]
//
// σⁿ Φₙ is written with s = σ/r and g = σ, so that no power of r alone
// leaves the range of f32.
void mdir::driver::builder::emitLennardJones(llvm::raw_ostream &os,
                                             const Control &control,
                                             StringRef sigma, StringRef epsilon,
                                             StringRef result,
                                             Truncation modifier,
                                             StringRef distance) {
  double cutoff = control.cutoffDistance * units::length;
  double from = control.switchDistance * units::length;
  os << "    %c4 = arith.constant 4.0 : f64\n"
     << "    %e4 = arith.mulf %c4, " << epsilon << " : f64\n"
     << "    %sr = arith.divf " << sigma << ", " << distance << " : f64\n";
  if (modifier == Truncation::None ||
      modifier == Truncation::SquaredDistanceSwitch) {
    os << "    %i6 = arith.constant 6 : i32\n"
       << "    %s6 = math.fpowi %sr, %i6 : f64, i32\n"
       << "    %s12 = arith.mulf %s6, %s6 : f64\n"
       << "    %t = arith.subf %s12, %s6 : f64\n"
       << "    "
       << (modifier == Truncation::SquaredDistanceSwitch ? StringRef("%lj_raw")
                                                         : result)
       << " = arith.mulf %e4, %t : f64\n";
    if (modifier == Truncation::SquaredDistanceSwitch) {
      // The cubic switching potential in r² [Brooks1983]. Write it
      // as (1-y)²(1+2y), y=(r²-rs²)/(rc²-rs²), to avoid cancellation
      // near rc. Clamping y gives the constant branches and a zero
      // derivative at both ends. AD differentiates the full product.
      os << "    %sw_rs2 = arith.constant " << formatReal(from * from)
         << " : f64\n"
         << "    %sw_width_inv = arith.constant "
         << formatReal(1.0 / (cutoff * cutoff - from * from)) << " : f64\n"
         << "    %sw_zero = arith.constant 0.0 : f64\n"
         << "    %sw_one = arith.constant 1.0 : f64\n"
         << "    %sw_two = arith.constant 2.0 : f64\n"
         << "    %sw_r2 = arith.mulf %r, %r : f64\n"
         << "    %sw_dr2 = arith.subf %sw_r2, %sw_rs2 : f64\n"
         << "    %sw_y_raw = arith.mulf %sw_dr2, %sw_width_inv : f64\n"
         << "    %sw_below = arith.cmpf olt, %sw_y_raw, %sw_zero : f64\n"
         << "    %sw_y_lo = arith.select %sw_below, %sw_zero, %sw_y_raw : f64\n"
         << "    %sw_above = arith.cmpf ogt, %sw_y_lo, %sw_one : f64\n"
         << "    %sw_y = arith.select %sw_above, %sw_one, %sw_y_lo : f64\n"
         << "    %sw_complement = arith.subf %sw_one, %sw_y : f64\n"
         << "    %sw_square = arith.mulf %sw_complement, %sw_complement : f64\n"
         << "    %sw_twice = arith.mulf %sw_two, %sw_y : f64\n"
         << "    %sw_factor = arith.addf %sw_one, %sw_twice : f64\n"
         << "    %sw_s = arith.mulf %sw_square, %sw_factor : f64\n"
         << "    " << result << " = arith.mulf %lj_raw, %sw_s : f64\n";
    }
    return;
  }
  os << "    %i3 = arith.constant 3 : i32\n"
     << "    %s3 = math.fpowi %sr, %i3 : f64, i32\n"
     << "    %s6 = arith.mulf %s3, %s3 : f64\n"
     << "    %s12 = arith.mulf %s6, %s6 : f64\n"
     << "    %g3 = math.fpowi " << sigma << ", %i3 : f64, i32\n"
     << "    %g6 = arith.mulf %g3, %g3 : f64\n"
     << "    %g12 = arith.mulf %g6, %g6 : f64\n";
  auto constant = [&](StringRef name, double value) {
    os << "    %" << name << " = arith.constant " << formatReal(value)
       << " : f64\n";
  };
  // a = σ¹² Φ₁₂ and b = σ⁶ Φ₆, from what each power loses, g¹² c₁₂ and
  // g⁶ c₆.
  auto emitSubtract = [&](StringRef c12, StringRef c6) {
    os << "    %gc12 = arith.mulf %g12, " << c12 << " : f64\n"
       << "    %a = arith.subf %s12, %gc12 : f64\n"
       << "    %gc6 = arith.mulf %g6, " << c6 << " : f64\n"
       << "    %b = arith.subf %s6, %gc6 : f64\n";
  };
  std::string a = "%a", b = "%b";
  if (modifier == Truncation::Shift) {
    constant("rc12", std::pow(cutoff, -12.0));
    constant("rc6", std::pow(cutoff, -6.0));
    emitSubtract("%rc12", "%rc6");
  } else if (modifier == Truncation::ForceSwitch) {
    double width = cutoff - from;
    os << "    %rs = arith.constant " << formatReal(from) << " : f64\n"
       << "    %zero = arith.constant 0.0 : f64\n"
       << "    %dr = arith.subf " << distance << ", %rs : f64\n"
       << "    %beyond = arith.cmpf ogt, " << distance << ", %rs : f64\n"
       << "    %u = arith.select %beyond, %dr, %zero : f64\n"
       << "    %u2 = arith.mulf %u, %u : f64\n"
       << "    %u3 = arith.mulf %u2, %u : f64\n";
    for (int n : {12, 6}) {
      double an = -n * ((n + 4) * cutoff - (n + 1) * from) /
                  (std::pow(cutoff, n + 2) * width * width);
      double bn = n * ((n + 3) * cutoff - (n + 1) * from) /
                  (std::pow(cutoff, n + 2) * width * width * width);
      double cn = std::pow(cutoff, -n) - an / 3.0 * std::pow(width, 3) -
                  bn / 4.0 * std::pow(width, 4);
      std::string k = std::to_string(n);
      constant("a" + k, an / 3.0);
      constant("b" + k, bn / 4.0);
      constant("c" + k, cn);
      // Cₙ + u³ (Aₙ/3 + (Bₙ/4) u)
      os << "    %pb" << k << " = arith.mulf %b" << k << ", %u : f64\n"
         << "    %pa" << k << " = arith.addf %a" << k << ", %pb" << k
         << " : f64\n"
         << "    %pu" << k << " = arith.mulf %u3, %pa" << k << " : f64\n"
         << "    %p" << k << " = arith.addf %c" << k << ", %pu" << k
         << " : f64\n";
    }
    emitSubtract("%p12", "%p6");
  } else {
    // Below rs the shifted powers; above, the switched squares.
    constant("in12", std::pow(from * cutoff, -6.0));
    constant("in6", std::pow(from * cutoff, -3.0));
    emitSubtract("%in12", "%in6");
    constant("k12", std::pow(cutoff, 6.0) /
                        (std::pow(cutoff, 6.0) - std::pow(from, 6.0)));
    constant("k6", std::pow(cutoff, 3.0) /
                       (std::pow(cutoff, 3.0) - std::pow(from, 3.0)));
    constant("rc6", std::pow(cutoff, -6.0));
    constant("rc3", std::pow(cutoff, -3.0));
    os << "    %gr6 = arith.mulf %g6, %rc6 : f64\n"
       << "    %h6 = arith.subf %s6, %gr6 : f64\n"
       << "    %h6s = arith.mulf %h6, %h6 : f64\n"
       << "    %a_out = arith.mulf %k12, %h6s : f64\n"
       << "    %gr3 = arith.mulf %g3, %rc3 : f64\n"
       << "    %h3 = arith.subf %s3, %gr3 : f64\n"
       << "    %h3s = arith.mulf %h3, %h3 : f64\n"
       << "    %b_out = arith.mulf %k6, %h3s : f64\n"
       << "    %rs = arith.constant " << formatReal(from) << " : f64\n"
       << "    %inside = arith.cmpf ole, " << distance << ", %rs : f64\n"
       << "    %a_sel = arith.select %inside, %a, %a_out : f64\n"
       << "    %b_sel = arith.select %inside, %b, %b_out : f64\n";
    a = "%a_sel";
    b = "%b_sel";
  }
  os << "    %t = arith.subf " << a << ", " << b << " : f64\n"
     << "    " << result << " = arith.mulf %e4, %t : f64\n";
}
