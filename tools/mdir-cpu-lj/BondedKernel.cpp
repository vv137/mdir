#include "Bonded.h"
#include <iomanip>
#include <sstream>

// Harmonic energy and analytic gradient [AllenTildesley2017]. Each parallel
// iteration owns its output row; endpoint scatter happens after this kernel.
std::string bondedKernel(const double *box, bool mixed) {
  std::ostringstream s;
  s << std::scientific << std::setprecision(17);
  std::string t = mixed ? "f32" : "f64";
  s << "func.func @evaluate_bonds(%ends: memref<?x2xi32>, %params: "
       "memref<?x2xf64>, "
       "%x: memref<?x3xf64>, %f: memref<?x6xf64>, %out: memref<?x10xf64>) "
       "attributes {llvm.emit_c_interface} {\n";
  for (int k = 0; k < 10; ++k)
    s << "%c" << k << " = arith.constant " << k << " : index\n";
  for (int k = 0; k < 3; ++k)
    s << "%box" << k << " = arith.constant " << box[k] << " : f64\n";
  s << "%half = arith.constant 5.000000e-01 : " << t
    << "\n"
       "%nb = memref.dim %ends, %c0 : memref<?x2xi32>\n"
       "scf.parallel (%b) = (%c0) to (%nb) step (%c1) {\n"
       "%i32 = memref.load %ends[%b,%c0] : memref<?x2xi32>\n"
       "%j32 = memref.load %ends[%b,%c1] : memref<?x2xi32>\n"
       "%i = arith.index_cast %i32 : i32 to index\n"
       "%j = arith.index_cast %j32 : i32 to index\n";
  for (int k = 0; k < 3; ++k) {
    s << "%xi" << k << " = memref.load %x[%i,%c" << k << "] : memref<?x3xf64>\n"
      << "%xj" << k << " = memref.load %x[%j,%c" << k << "] : memref<?x3xf64>\n"
      << "%raw" << k << " = arith.subf %xi" << k << ", %xj" << k << " : f64\n"
      << "%ratio" << k << " = arith.divf %raw" << k << ", %box" << k
      << " : f64\n"
      << "%image" << k << " = math.roundeven %ratio" << k << " : f64\n"
      << "%shift" << k << " = arith.mulf %image" << k << ", %box" << k
      << " : f64\n"
      << (mixed ? "%wide" : "%d") << k << " = arith.subf %raw" << k
      << ", %shift" << k << " : f64\n";
    if (mixed)
      s << "%d" << k << " = arith.truncf %wide" << k << " : f64 to f32\n";
    s << "%sq" << k << " = arith.mulf %d" << k << ", %d" << k << " : " << t
      << '\n';
  }
  s << "%xy = arith.addf %sq0, %sq1 : " << t << "\n"
    << "%r2 = arith.addf %xy, %sq2 : " << t << "\n"
    << "%r = math.sqrt %r2 : " << t << "\n";
  for (int k = 0; k < 2; ++k) {
    s << (mixed ? "%pw" : "%p") << k << " = memref.load %params[%b,%c" << k
      << "] : memref<?x2xf64>\n";
    if (mixed)
      s << "%p" << k << " = arith.truncf %pw" << k << " : f64 to f32\n";
  }
  s << "%extension = arith.subf %r, %p1 : " << t << "\n"
    << "%derivative = arith.mulf %p0, %extension : " << t << "\n"
    << "%negative = arith.negf %derivative : " << t << "\n"
    << "%scale = arith.divf %negative, %r : " << t << "\n"
    << "%twiceE = arith.mulf %derivative, %extension : " << t << "\n"
    << "%energy = arith.mulf %half, %twiceE : " << t << "\n";
  auto store = [&](std::string name, std::string buffer, int col, int width) {
    if (mixed)
      s << '%' << name << "64 = arith.extf %" << name << " : f32 to f64\n";
    s << "memref.store %" << name << (mixed ? "64" : "") << ", %" << buffer
      << "[%b,%c" << col << "] : memref<?x" << width << "xf64>\n";
  };
  store("energy", "out", 0, 10);
  for (int k = 0; k < 3; ++k) {
    s << "%fi" << k << " = arith.mulf %scale, %d" << k << " : " << t << "\n"
      << "%fj" << k << " = arith.negf %fi" << k << " : " << t << "\n";
    store("fi" + std::to_string(k), "f", k, 6);
    store("fj" + std::to_string(k), "f", k + 3, 6);
  }
  for (int k = 0; k < 9; ++k) {
    s << "%vir" << k << " = arith.mulf %d" << k / 3 << ", %fi" << k % 3 << " : "
      << t << "\n";
    store("vir" + std::to_string(k), "out", k + 1, 10);
  }
  s << "scf.reduce\n}\nreturn\n}\n";
  return s.str();
}
