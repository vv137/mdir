// SETTLE, SHAKE and RATTLE, the virtual sites, and the tests and stops
// of the constraints (docs/driver.md, Section 2.12).

#include "BuilderImpl.h"
#include "SiteIR.h"
#include "mdir/Dialect/MD/MDCoordinates.h"
#include "mdir/Dialect/MD/MDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"

using namespace mdir::driver;
using namespace mdir::driver::builder;
namespace prolate = mdir::prolate;
namespace influence = mdir::influence;

void Builder::emitPlaceSites(StringRef indent, StringRef x, StringRef result,
                             StringRef relations) {
  std::string current = x.str();
  std::vector<const Program::TupleSet *> sets = getSiteSets();
  for (const Program::TupleSet *each : sets) {
    const Program::TupleSet &set = *each;
    bool amber = set.name == "sites_amber";
    bool last = each == sets.back();
    std::string relation = (relations + set.name).str();
    std::string a = "%f_" + set.name + "_a", b = "%f_" + set.name + "_b";
    std::string relationType = "!rel_" + set.name, fieldType = "!of_" + set.name;
    std::string next =
        last ? result.str() : (result + "_" + set.name + "_x").str();
    emitIRFragment(
        indent, {next}, {"!vec"}, {relation, current, "%cell", a, b},
        {relationType, "!vec", "!md.cell", fieldType, fieldType},
        [amber](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          using namespace mlir;
          auto loc = builder.getUnknownLoc();
          auto vec = VectorType::get({3}, builder.getF64Type());
          DenseI32ArrayAttr kinds;
          DenseI64ArrayAttr members;
          mdir::md::getCoordinateAttrs(
              builder, {{mdir::md::CoordinateKind::Displacement, {2, 1}},
                        {mdir::md::CoordinateKind::Displacement, {3, 1}},
                        {mdir::md::CoordinateKind::Displacement, {0, 1}}},
              kinds, members);
          auto delta = mdir::md::GatherTuplesOp::create(
              builder, loc, values[1].getType(), Type{}, values[0], values[1],
              values[2], ValueRange{}, values.slice(3, 2), kinds, members);
          {
            OpBuilder::InsertionGuard guard(builder);
            SmallVector<Type> types = {vec, vec, vec, builder.getF64Type(),
                                       builder.getF64Type()};
            SmallVector<Location> locations(types.size(), loc);
            Block *block = builder.createBlock(&delta.getKernel(), {}, types,
                                                locations);
            auto args = block->getArguments();
            Value placed = buildSitePlacement(builder, amber, args[0], args[1],
                                               args[3], args[4]);
            Value change = arith::SubFOp::create(builder, loc, placed, args[2]);
            Value zero = arith::ConstantOp::create(
                builder, loc, DenseElementsAttr::get(vec, 0.0));
            mdir::md::YieldOp::create(builder, loc,
                                      ValueRange{change, zero, zero, zero});
          }
          return {buildAddChange(builder, values[1], delta.getResult())};
        });
    current = next;
  }
}

std::string Builder::emitSpreadSites(StringRef indent, StringRef x,
                                     StringRef f, StringRef result,
                                     StringRef relations, StringRef virial,
                                     StringRef virialResult) {
  std::string current = f.str(), currentVirial = virial.str();
  std::vector<const Program::TupleSet *> sets = getSiteSets();
  for (const Program::TupleSet *each : sets) {
    const Program::TupleSet &set = *each;
    bool amber = set.name == "sites_amber", derived = spreadsSitesDerived();
    bool last = each == sets.back();
    bool changesVirial = !currentVirial.empty() && (amber || derived);
    std::string next =
        last ? result.str() : (result + "_" + set.name + "_f").str();
    std::string changed = last ? virialResult.str()
                              : (virialResult + "_" + set.name + "_w").str();
    std::string relation = (relations + set.name).str();
    std::string a = "%f_" + set.name + "_a", b = "%f_" + set.name + "_b";
    std::string relationType = "!rel_" + set.name, fieldType = "!of_" + set.name;
    llvm::SmallVector<StringRef> operands = {relation, x, "%cell", a, b, current};
    llvm::SmallVector<StringRef> types = {relationType, "!vec", "!md.cell", fieldType,
                                    fieldType, "!vec"};
    llvm::SmallVector<StringRef> outputs = {next}, outputTypes = {"!vec"};
    if (changesVirial) {
      operands.push_back(currentVirial);
      types.push_back("vector<9xf64>");
      outputs.push_back(changed);
      outputTypes.push_back("vector<9xf64>");
    }
    emitIRFragment(
        indent, outputs, outputTypes, operands, types,
        [amber, derived, changesVirial, symbol = "spread_" + set.name](
            mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
          SiteSpreadInputs input{values[0], values[1], values[2], values[3],
                                  values[4], values[5], {}};
          if (changesVirial)
            input.virial = values[6];
          auto result = buildSiteSpread(builder, input, amber, derived, symbol);
          TermIRValues outputs = {result.forces};
          if (changesVirial)
            outputs.push_back(result.virial);
          return outputs;
        });
    current = next;
    if (changesVirial)
      currentVirial = changed;
  }
  return currentVirial;
}

bool Builder::spreadsSitesDerived() const {
  const char *value = std::getenv("MDIR_SITE_SPREAD");
  return value && StringRef(value) == "derived";
}


void Builder::emitSettlePositions(StringRef indent, StringRef old,
                                  StringRef x, StringRef change,
                                  StringRef result, StringRef failed) {
  // Below double precision, the three bonds of the water by M-SHAKE: the
  // positions that SETTLE turns are of the size of the water, and their
  // rounding in f32 over the change of a step is an error of the
  // velocities, which heated JAC by 7 K in 2 ns; M-SHAKE carries the
  // change itself, a small number [Jung2026] (D112).
  if (control.precision != Precision::Double) {
    emitBondConstraints(indent, "settles", 3,
                        {{0, 1, "%vs_doh"}, {0, 2, "%vs_doh"},
                         {1, 2, "%vs_dhh"}},
                        {"%f_settles_doh", "%f_settles_dhh"}, old, x, change,
                        failed);
    emitAddChange(indent, x, change, result);
    return;
  }
  std::string inner = (indent + "  ").str();
  // In the frame of the old plane of the water, with the origin at the new
  // center of mass, the new triangle is the rigid one turned by three
  // angles (Miyamoto and Kollman 1992, Appendix A). The positions are
  // taken relative to the new oxygen, in the minimum image; the old ones
  // are moved by the same periods of the cell.
  os << indent << change << (failed.empty() ? "" : ", " + failed.str())
     << " = md.gather_tuples %r_settles, " << x << ", %cell\n"
     << indent << "    coordinates(displacement(1, 0), displacement(2, 0))\n"
     << indent << "    gather(" << old << ", " << x
     << ", %m : !vec, !vec, !real)\n"
     << indent << "    tuple(%f_settles_doh, %f_settles_dhh : !of_settles, "
                  "!of_settles) {\n"
     << indent << "^bb0(%vs_b1: vector<3xf64>, %vs_c1: vector<3xf64>, "
                  "%vs_o0: vector<3xf64>, %vs_o1: vector<3xf64>, "
                  "%vs_o2: vector<3xf64>, %vs_n0: vector<3xf64>, "
                  "%vs_n1: vector<3xf64>, %vs_n2: vector<3xf64>, "
                  "%vs_m0: f64, %vs_m1: f64, %vs_m2: f64, %vs_doh: f64, "
                  "%vs_dhh: f64):\n";
  SiteKernel k(os, inner);
  // The old bonds, in the periods of the new ones.
  auto oldBond = [&](StringRef bond, StringRef o, StringRef n) {
    std::string raw = k.vector("subf", n, "%vs_n0");
    std::string shift = k.vector("subf", bond, raw);
    return k.vector("addf", k.vector("subf", o, "%vs_o0"), shift);
  };
  std::string b0 = oldBond("%vs_b1", "%vs_o1", "%vs_n1");
  std::string c0 = oldBond("%vs_c1", "%vs_o2", "%vs_n2");
  // The rigid triangle: the oxygen at ra from the center of mass along the
  // bisector, the hydrogens at rb behind it and rc to each side.
  std::string two = k.constant(2.0), half = k.constant(0.5),
              one = k.constant(1.0);
  std::string total =
      k.real("addf", "%vs_m0", k.real("mulf", two, "%vs_m1"));
  std::string rc = k.real("mulf", half, "%vs_dhh");
  std::string height = k.root(k.real(
      "subf", k.real("mulf", "%vs_doh", "%vs_doh"), k.real("mulf", rc, rc)));
  std::string ra = k.real(
      "divf", k.real("mulf", k.real("mulf", two, "%vs_m1"), height), total);
  std::string rb = k.real("subf", height, ra);
  // The new positions about their center of mass.
  std::string center = k.scale(k.real("divf", "%vs_m1", total),
                               k.vector("addf", "%vs_b1", "%vs_c1"));
  std::string a1 = k.negate(center);
  std::string b1 = k.vector("subf", "%vs_b1", center);
  std::string c1 = k.vector("subf", "%vs_c1", center);
  std::string z = k.unit(k.cross(b0, c0));
  std::string xAxis = k.unit(k.cross(a1, z));
  std::string y = k.cross(z, xAxis);
  std::string xb0 = k.dot(b0, xAxis), yb0 = k.dot(b0, y);
  std::string xc0 = k.dot(c0, xAxis), yc0 = k.dot(c0, y);
  std::string za1 = k.dot(a1, z);
  std::string xb1 = k.dot(b1, xAxis), yb1 = k.dot(b1, y), zb1 = k.dot(b1, z);
  std::string xc1 = k.dot(c1, xAxis), yc1 = k.dot(c1, y), zc1 = k.dot(c1, z);
  auto complement = [&](StringRef sine) {
    return k.root(k.real("subf", one, k.real("mulf", sine, sine)));
  };
  std::string sinPhi = k.real("divf", za1, ra);
  std::string cosPhi = complement(sinPhi);
  std::string sinPsi =
      k.real("divf", k.real("subf", zb1, zc1),
             k.real("mulf", k.real("mulf", two, rc), cosPhi));
  std::string cosPsi = complement(sinPsi);
  std::string ya2 = k.real("mulf", ra, cosPhi);
  std::string xb2 = k.real("mulf", k.real("subf", k.constant(0.0), rc),
                           cosPsi);
  std::string t1 = k.real("mulf", k.real("subf", k.constant(0.0), rb),
                          cosPhi);
  std::string t2 = k.real("mulf", k.real("mulf", rc, sinPsi), sinPhi);
  std::string yb2 = k.real("subf", t1, t2), yc2 = k.real("addf", t1, t2);
  auto sum3 = [&](StringRef a, StringRef b, StringRef c) {
    return k.real("addf", k.real("addf", a, b), c);
  };
  std::string alpha = sum3(k.real("mulf", xb2, k.real("subf", xb0, xc0)),
                           k.real("mulf", yb0, yb2),
                           k.real("mulf", yc0, yc2));
  std::string beta = sum3(k.real("mulf", xb2, k.real("subf", yc0, yb0)),
                          k.real("mulf", xb0, yb2),
                          k.real("mulf", xc0, yc2));
  std::string gamma = k.real(
      "addf",
      k.real("subf", k.real("mulf", xb0, yb1), k.real("mulf", xb1, yb0)),
      k.real("subf", k.real("mulf", xc0, yc1), k.real("mulf", xc1, yc0)));
  std::string both = k.real("addf", k.real("mulf", alpha, alpha),
                            k.real("mulf", beta, beta));
  std::string sinTheta = k.real(
      "divf",
      k.real("subf", k.real("mulf", alpha, gamma),
             k.real("mulf", beta,
                    k.root(k.real("subf", both,
                                  k.real("mulf", gamma, gamma))))),
      both);
  std::string cosTheta = complement(sinTheta);
  std::string zero = k.constant(0.0);
  std::string xa3 = k.real("subf", zero, k.real("mulf", ya2, sinTheta));
  std::string ya3 = k.real("mulf", ya2, cosTheta);
  std::string xb3 = k.real("subf", k.real("mulf", xb2, cosTheta),
                           k.real("mulf", yb2, sinTheta));
  std::string yb3 = k.real("addf", k.real("mulf", xb2, sinTheta),
                           k.real("mulf", yb2, cosTheta));
  std::string xc3 = k.real("subf", k.real("subf", zero,
                                          k.real("mulf", xb2, cosTheta)),
                           k.real("mulf", yc2, sinTheta));
  std::string yc3 = k.real("addf", k.real("subf", zero,
                                          k.real("mulf", xb2, sinTheta)),
                           k.real("mulf", yc2, cosTheta));
  // Back to the cell, relative to the new oxygen.
  auto back = [&](StringRef px, StringRef py, StringRef pz) {
    return k.vector("addf", k.combine(px, xAxis, py, y, pz, z), center);
  };
  std::string a3 = back(xa3, ya3, za1);
  std::string b3 = back(xb3, yb3, zb1);
  std::string c3 = back(xc3, yc3, zc1);
  std::string db = k.vector("subf", b3, "%vs_b1");
  std::string dc = k.vector("subf", c3, "%vs_c1");
  // The bonds of the new triangle, which the test of the constraints takes
  // (D278).
  std::string test;
  if (!failed.empty())
    test = ", " + emitLengthTest(k, {{k.vector("subf", b3, a3), "%vs_doh"},
                                     {k.vector("subf", c3, a3), "%vs_doh"},
                                     {k.vector("subf", c3, b3), "%vs_dhh"}});
  os << inner << "md.yield " << a3 << ", " << db << ", " << dc << test
     << " : vector<3xf64>, vector<3xf64>, vector<3xf64>"
     << (failed.empty() ? "" : ", i1") << "\n"
     << indent << "} : !rel_settles, !vec -> !vec"
     << (failed.empty() ? "" : ", i1") << "\n";
  os << indent << result << " = md.map_particles gather(" << x << ", "
     << change << " : !vec, !vec) {\n"
     << indent << "^bb0(%vs_x: vector<3xf64>, %vs_d: vector<3xf64>):\n"
     << inner << "%vs_sum = arith.addf %vs_x, %vs_d : vector<3xf64>\n"
     << inner << "md.yield %vs_sum : vector<3xf64>\n"
     << indent << "} : !vec\n";
}

std::string Builder::emitConstraintVirial(StringRef indent,
                                          const Program::TupleSet &set,
                                          StringRef old, StringRef change,
                                          StringRef virial,
                                          StringRef virialResult) {
  // The forces of the constraints over the first half of the step: the
  // drift took x + dt v + dt² F / 2m to x + Δ, so G = 2 m Δ / dt², along
  // the bonds before the drift. Their virial Σ (x_j − x_0) ⊗ G_j, taken at
  // those positions, with the weight ½: with that of the second half, the
  // virial of the constraints is the mean of the two, as the kinetic
  // energy of the pressure is (D45).
  unsigned count = set.arity - 1;
  std::string inner = (indent + "  ").str();
  std::string coordinates, arguments;
  for (unsigned k = 1; k <= count; ++k) {
    coordinates += (k == 1 ? "" : ", ") +
                   ("displacement(" + std::to_string(k) + ", 0)");
    arguments += "%vs_r" + std::to_string(k) + ": vector<3xf64>, ";
  }
  for (unsigned k = 0; k <= count; ++k)
    arguments += "%vs_c" + std::to_string(k) + ": vector<3xf64>, ";
  for (unsigned k = 0; k <= count; ++k)
    arguments += "%vs_m" + std::to_string(k) + ": f64" +
                 (k == count ? "" : ", ");
  std::string sum = (virialResult + "_first").str();
  os << indent << sum << " = md.sum_tuples %r_" << set.name << ", " << old
     << ", %cell\n"
     << indent << "    coordinates(" << coordinates << ")\n"
     << indent << "    gather(" << change << ", %m : !vec, !real) {\n"
     << indent << "^bb0(" << arguments << "):\n";
  SiteKernel w(os, inner);
  std::string rate = w.real("divf", w.constant(1.0),
                            w.real("mulf", "%dt", "%dt"));
  std::vector<std::string> forces;
  for (unsigned j = 1; j <= count; ++j)
    forces.push_back(w.scale(
        w.real("mulf", "%vs_m" + std::to_string(j), rate),
        "%vs_c" + std::to_string(j)));
  std::string elements;
  for (int a = 0; a != 3; ++a) {
    std::string row;
    for (unsigned j = 1; j <= count; ++j) {
      std::string term = w.scale(w.component("%vs_r" + std::to_string(j), a),
                                 forces[j - 1]);
      row = row.empty() ? term : w.vector("addf", row, term);
    }
    for (int b = 0; b != 3; ++b)
      elements += (elements.empty() ? "" : ", ") + w.component(row, b);
  }
  os << inner << "%vs_w = vector.from_elements " << elements
     << " : vector<9xf64>\n"
     << inner << "md.yield %vs_w : vector<9xf64>\n"
     << indent << "} : !rel_" << set.name << ", !vec -> vector<9xf64>\n";
  os << indent << virialResult << " = arith.addf " << virial << ", " << sum
     << " : vector<9xf64>\n";
  return virialResult.str();
}

/// Emits the solution of `A x = b` for a system of at most 3 × 3, by
/// Gaussian elimination without pivoting: the matrices of the constraints
/// of a group are dominated by their diagonals.
/// The iterations of Newton of the positions of SHAKE: from the error of
/// a step of 4 fs with repartitioned masses, about 1e-2 of a bond, four
/// reach the rounding of f64; two more are a margin.
static const int shakeIterations = 6;

static std::vector<std::string>
emitSmallSolve(SiteKernel &k, std::vector<std::vector<std::string>> A,
               std::vector<std::string> b) {
  unsigned count = b.size();
  for (unsigned c = 0; c != count; ++c)
    for (unsigned r = c + 1; r != count; ++r) {
      std::string factor = k.real("divf", A[r][c], A[c][c]);
      for (unsigned d = c; d != count; ++d)
        A[r][d] = k.real("subf", A[r][d], k.real("mulf", factor, A[c][d]));
      b[r] = k.real("subf", b[r], k.real("mulf", factor, b[c]));
    }
  std::vector<std::string> x(count);
  for (unsigned c = count; c-- != 0;) {
    std::string sum = b[c];
    for (unsigned d = c + 1; d != count; ++d)
      sum = k.real("subf", sum, k.real("mulf", A[c][d], x[d]));
    x[c] = k.real("divf", sum, A[c][c]);
  }
  return x;
}

void Builder::emitShakePositions(StringRef indent, StringRef old,
                                 StringRef x, const Program::TupleSet &set,
                                 StringRef result, StringRef failed) {
  unsigned count = set.arity - 1;
  std::string change = (result + "_change").str();
  // The bonds of the heavy atom 0 to its hydrogens.
  std::vector<ConstrainedBond> bonds;
  std::vector<std::string> tuple;
  for (unsigned k = 1; k <= count; ++k) {
    bonds.push_back({0, k, "%vs_d" + std::to_string(k)});
    tuple.push_back("%f_" + set.name + "_d" + std::to_string(k));
  }
  emitBondConstraints(indent, set.name, set.arity, bonds, tuple, old, x,
                      change, failed);
  emitAddChange(indent, x, change, result);
}

void Builder::emitAddChange(StringRef indent, StringRef x, StringRef change,
                            StringRef result) {
  emitIRFragment(
      indent, {result}, {"!vec"}, {x, change}, {"!vec", "!vec"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        return {buildAddChange(builder, values[0], values[1])};
      });
}

void Builder::emitBondConstraints(StringRef indent, StringRef setName,
                                  unsigned arity,
                                  llvm::ArrayRef<ConstrainedBond> bonds,
                                  llvm::ArrayRef<std::string> tuple, StringRef old,
                                  StringRef x, StringRef change,
                                  StringRef failed) {
  unsigned count = bonds.size();
  std::string inner = (indent + "  ").str();
  std::string coordinates, tupleTypes, arguments;
  for (auto [k, bond] : llvm::enumerate(bonds)) {
    coordinates += (k == 0 ? "" : ", ") +
                   ("displacement(" + std::to_string(bond.second) + ", " +
                    std::to_string(bond.first) + ")");
    arguments += "%vs_r" + std::to_string(k) + ": vector<3xf64>, ";
  }
  for (StringRef field : {"o", "n"})
    for (unsigned j = 0; j != arity; ++j)
      arguments += ("%vs_" + field + std::to_string(j) + ": vector<3xf64>, ")
                       .str();
  for (unsigned j = 0; j != arity; ++j)
    arguments += "%vs_m" + std::to_string(j) + ": f64, ";
  std::string tupleList;
  for (auto [k, name] : llvm::enumerate(tuple)) {
    tupleList += (k == 0 ? "" : ", ") + name;
    tupleTypes += (k == 0 ? "" : ", ") + ("!of_" + setName).str();
  }
  // The arguments of the tuple fields: the lengths the bonds name.
  std::vector<std::string> fieldArguments;
  for (const ConstrainedBond &bond : bonds)
    if (!llvm::is_contained(fieldArguments, bond.length))
      fieldArguments.push_back(bond.length);
  for (auto [k, name] : llvm::enumerate(fieldArguments))
    arguments += name + ": f64" +
                 (k + 1 == fieldArguments.size() ? "" : ", ");
  os << indent << change << (failed.empty() ? "" : ", " + failed.str())
     << " = md.gather_tuples %r_" << setName << ", " << x << ", %cell\n"
     << indent << "    coordinates(" << coordinates << ")\n"
     << indent << "    gather(" << old << ", " << x
     << ", %m : !vec, !vec, !real)\n"
     << indent << "    tuple(" << tupleList << " : " << tupleTypes << ") {\n"
     << indent << "^bb0(" << arguments << "):\n";
  SiteKernel k(os, inner);
  // The old bonds, in the periods of the new ones.
  std::vector<std::string> olds, inverse;
  std::string one = k.constant(1.0), two = k.constant(2.0);
  for (unsigned j = 0; j != arity; ++j)
    inverse.push_back(k.real("divf", one, "%vs_m" + std::to_string(j)));
  for (auto [i, bond] : llvm::enumerate(bonds)) {
    std::string a = std::to_string(bond.first), b = std::to_string(bond.second);
    std::string raw = k.vector("subf", "%vs_n" + b, "%vs_n" + a);
    std::string shift = k.vector("subf", "%vs_r" + std::to_string(i), raw);
    olds.push_back(k.vector(
        "addf", k.vector("subf", "%vs_o" + b, "%vs_o" + a), shift));
  }
  std::string zero = k.zero();
  // For one bond the fixed-old-direction projection is a quadratic.
  // With r the predicted bond, s the old bond, and r' = r + t s,
  // t = (d²-r²)/(r·s + sqrt((r·s)² + s²(d²-r²))). Rationalizing the
  // near root avoids subtracting nearly equal numbers. Keep the same
  // mass-weighted impulses, periodic image, and small-change storage as
  // M-SHAKE. A checked fast path falls back to the existing Newton solve.
  bool analytic = control.analyticBonds && count == 1;
  if (analytic) {
    const auto &bond = bonds.front();
    std::string r = "%vs_r0", s = olds.front();
    std::string d2 = k.real("mulf", bond.length, bond.length);
    std::string error = k.real("subf", d2, k.dot(r, r));
    std::string rs = k.dot(r, s);
    std::string discriminant = k.real(
        "addf", k.real("mulf", rs, rs), k.real("mulf", k.dot(s, s), error));
    std::string t = k.real("divf", error,
                           k.real("addf", rs, k.root(discriminant)));
    std::string lambda = k.real(
        "divf", t, k.real("addf", inverse[bond.first], inverse[bond.second]));
    std::string push = k.scale(lambda, s);
    std::string da = k.negate(k.scale(inverse[bond.first], push));
    std::string db = k.scale(inverse[bond.second], push);
    std::string corrected = k.vector("subf", k.vector("addf", r, db), da);
    std::string residual = k.real("subf", k.dot(corrected, corrected), d2);
    std::string tolerance = k.real(
        "mulf", d2, k.constant(control.precision == Precision::Double
                                   ? 1.0e-12 : 1.0e-6));
    std::string scalarZero = k.constant(0.0);
    os << inner << "%vs_residual = math.absf " << residual << " : f64\n"
       << inner << "%vs_accurate = arith.cmpf ole, %vs_residual, "
       << tolerance << " : f64\n"
       << inner << "%vs_forward = arith.cmpf ogt, " << rs << ", "
       << scalarZero << " : f64\n"
       << inner << "%vs_analytic = arith.andi %vs_accurate, %vs_forward : i1\n"
       << inner << "%vs_checked0, %vs_checked1 = scf.if %vs_analytic -> "
                   "(vector<3xf64>, vector<3xf64>) {\n"
       << inner << "  scf.yield " << da << ", " << db
       << " : vector<3xf64>, vector<3xf64>\n"
       << inner << "} else {\n";
    inner += "  ";
  }
  // The members move along the old bonds s_l: for each bond l = (a, b),
  // b by λ_l s_l / m_b and a by −λ_l s_l / m_a. Each iteration of Newton
  // solves the constraints linearized at the current bonds r_k exactly,
  //   Σ_l 2 (r_k · s_l) w_kl λ_l = d_k² − r_k²,
  // with w_kl what bond l moves bond k by per unit of λ_l s_l: the inverse
  // mass of each member that the two share, with its sign; the error
  // squares with each iteration whatever the masses (M-SHAKE
  // [Krautler2001]). The atoms carry what they moved, small numbers, so
  // that the change keeps its digits in f32: its rounding becomes an
  // error of the velocities a step divides by [Jung2026].
  os << inner << "%vs_c0 = arith.constant 0 : index\n"
     << inner << "%vs_c1 = arith.constant 1 : index\n"
     << inner << "%vs_iterations = arith.constant " << shakeIterations
     << " : index\n";
  std::string results, inits, types;
  for (unsigned j = 0; j != arity; ++j) {
    std::string n = std::to_string(j);
    results += (j == 0 ? "" : ", ") + ("%vs_a" + n);
    inits += (j == 0 ? "" : ", ") + ("%vs_s" + n + " = " + zero);
    types += (j == 0 ? "" : ", ") + std::string("vector<3xf64>");
  }
  os << inner << results << " = scf.for %vs_sweep = %vs_c0 to "
                            "%vs_iterations step %vs_c1\n"
     << inner << "    iter_args(" << inits << ") -> (" << types << ") {\n";
  SiteKernel l(os, inner + "  ", "%vsl");
  std::vector<std::string> moved;
  for (unsigned j = 0; j != arity; ++j)
    moved.push_back("%vs_s" + std::to_string(j));
  std::vector<std::string> current, error;
  for (auto [i, bond] : llvm::enumerate(bonds)) {
    current.push_back(l.vector(
        "subf",
        l.vector("addf", "%vs_r" + std::to_string(i), moved[bond.second]),
        moved[bond.first]));
    error.push_back(l.real("subf", l.real("mulf", bond.length, bond.length),
                           l.dot(current.back(), current.back())));
  }
  // w_kl: what bond l moves bond k by, per unit of λ_l s_l: the inverse
  // masses of the members the two share, with their signs.
  auto weightOf = [&](const ConstrainedBond &k,
                      const ConstrainedBond &bond) -> std::string {
    std::string sum;
    auto add = [&](bool negative, unsigned j) {
      if (sum.empty())
        sum = negative ? l.real("subf", l.constant(0.0), inverse[j])
                       : inverse[j];
      else
        sum = l.real(negative ? "subf" : "addf", sum, inverse[j]);
    };
    if (k.second == bond.second)
      add(false, k.second);
    if (k.second == bond.first)
      add(true, k.second);
    if (k.first == bond.second)
      add(true, k.first);
    if (k.first == bond.first)
      add(false, k.first);
    return sum.empty() ? l.constant(0.0) : sum;
  };
  std::vector<std::vector<std::string>> A(count,
                                          std::vector<std::string>(count));
  for (unsigned i = 0; i != count; ++i)
    for (unsigned j = 0; j != count; ++j)
      A[i][j] = l.real("mulf", l.real("mulf", two, l.dot(current[i],
                                                          olds[j])),
                       weightOf(bonds[i], bonds[j]));
  std::vector<std::string> lambda = emitSmallSolve(l, A, error);
  for (auto [j, bond] : llvm::enumerate(bonds)) {
    std::string push = l.scale(lambda[j], olds[j]);
    moved[bond.second] = l.vector("addf", moved[bond.second],
                                  l.scale(inverse[bond.second], push));
    moved[bond.first] = l.vector("subf", moved[bond.first],
                                 l.scale(inverse[bond.first], push));
  }
  std::string yielded;
  for (unsigned j = 0; j != arity; ++j)
    yielded += (j == 0 ? "" : ", ") + moved[j];
  os << inner << "  scf.yield " << yielded << " : " << types << "\n"
     << inner << "}\n";
  if (analytic) {
    os << inner << "scf.yield " << results << " : " << types << "\n";
    inner.resize(inner.size() - 2);
    os << inner << "}\n";
    results = "%vs_checked0, %vs_checked1";
  }
  // The bonds that the changes give, which the test of the constraints
  // takes (D278).
  std::string test;
  if (!failed.empty()) {
    SiteKernel t(os, inner, "%vst");
    std::vector<std::string> changes;
    StringRef rest = results;
    while (!rest.empty()) {
      auto [name, tail] = rest.split(", ");
      changes.push_back(name.str());
      rest = tail;
    }
    std::vector<std::pair<std::string, std::string>> lengths;
    for (auto [i, bond] : llvm::enumerate(bonds))
      lengths.push_back(
          {t.vector("subf",
                    t.vector("addf", "%vs_r" + std::to_string(i),
                             changes[bond.second]),
                    changes[bond.first]),
           bond.length});
    test = ", " + emitLengthTest(t, lengths);
  }
  os << inner << "md.yield " << results << test << " : " << types
     << (failed.empty() ? "" : ", i1") << "\n"
     << indent << "} : !rel_" << setName << ", !vec -> !vec"
     << (failed.empty() ? "" : ", i1") << "\n";
}

std::string Builder::emitLengthTest(
    SiteKernel &k, llvm::ArrayRef<std::pair<std::string, std::string>> bonds) {
  // The test of emitConstraintFailures: |r²/d² − 1| above the tolerance,
  // or not a number (an unordered comparison).
  double tolerance = control.precision == Precision::Double ? 1.0e-8 : 1.0e-4;
  std::string one = k.constant(1.0), limit = k.constant(tolerance);
  std::string any;
  for (const auto &[r, length] : bonds) {
    std::string d2 = k.real("mulf", length, length);
    std::string off = k.real("subf", k.real("divf", k.dot(r, r), d2), one);
    std::string magnitude = k.unary("math.absf", off);
    std::string bad = k.compare("ugt", magnitude, limit);
    any = any.empty() ? bad : k.boolean("ori", any, bad);
  }
  return any;
}

std::string Builder::emitShakeVelocities(StringRef indent, StringRef x,
                                         StringRef v,
                                         const Program::TupleSet &set,
                                         StringRef result, StringRef virial,
                                         StringRef virialResult) {
  unsigned count = set.arity - 1;
  std::string inner = (indent + "  ").str();
  std::string coordinates, tuple, tupleTypes, arguments;
  for (unsigned k = 1; k <= count; ++k) {
    coordinates += (k == 1 ? "" : ", ") + ("displacement(" +
                                           std::to_string(k) + ", 0)");
    tuple += (k == 1 ? "" : ", ") + ("%f_" + set.name + "_d" +
                                     std::to_string(k));
    tupleTypes += (k == 1 ? "" : ", ") + ("!of_" + set.name);
    arguments += "%vs_r" + std::to_string(k) + ": vector<3xf64>, ";
  }
  for (unsigned k = 0; k <= count; ++k)
    arguments += "%vs_v" + std::to_string(k) + ": vector<3xf64>, ";
  for (unsigned k = 0; k <= count; ++k)
    arguments += "%vs_m" + std::to_string(k) + ": f64, ";
  for (unsigned k = 1; k <= count; ++k)
    arguments += "%vs_d" + std::to_string(k) + ": f64" +
                 (k == count ? "" : ", ");
  auto header = [&](StringRef name, StringRef op, StringRef field) {
    os << indent << name << " = md." << op << " %r_" << set.name << ", " << x
       << ", %cell\n"
       << indent << "    coordinates(" << coordinates << ")\n"
       << indent << "    gather(" << field << ", %m : !vec, !real)\n"
       << indent << "    tuple(" << tuple << " : " << tupleTypes << ") {\n"
       << indent << "^bb0(" << arguments << "):\n";
  };
  // The impulses τ along the bonds that leave no velocity along them:
  // A τ = −b, A_ij = δ_ij / m_j + e_i · e_j / m_0, b_i = e_i · (v_i − v_0),
  // by Gaussian elimination. Returns the changes of the velocities.
  auto solve = [&](SiteKernel &k) {
    std::vector<std::string> units, rhs;
    std::string one = k.constant(1.0), zero = k.constant(0.0);
    std::string inverseCenter = k.real("divf", one, "%vs_m0");
    std::vector<std::string> inverse;
    for (unsigned j = 1; j <= count; ++j) {
      std::string n = std::to_string(j);
      units.push_back(k.unit("%vs_r" + n));
      inverse.push_back(k.real("divf", one, "%vs_m" + n));
      rhs.push_back(k.real(
          "subf", zero,
          k.dot(units.back(), k.vector("subf", "%vs_v" + n, "%vs_v0"))));
    }
    std::vector<std::vector<std::string>> A(count,
                                            std::vector<std::string>(count));
    for (unsigned i = 0; i != count; ++i)
      for (unsigned j = 0; j != count; ++j) {
        std::string coupling =
            k.real("mulf", k.dot(units[i], units[j]), inverseCenter);
        A[i][j] = i == j ? k.real("addf", coupling, inverse[i]) : coupling;
      }
    std::vector<std::string> impulse = emitSmallSolve(k, A, rhs);
    std::vector<std::string> changes(count + 1);
    std::string center = k.zero();
    for (unsigned j = 0; j != count; ++j) {
      std::string push = k.scale(impulse[j], units[j]);
      changes[j + 1] = k.scale(inverse[j], push);
      center = k.vector("subf", center, k.scale(inverseCenter, push));
    }
    changes[0] = center;
    return changes;
  };

  std::string change = (result + "_change").str();
  header(change, "gather_tuples", v);
  SiteKernel k(os, inner);
  std::vector<std::string> changes = solve(k);
  std::string yielded, types;
  for (unsigned j = 0; j <= count; ++j) {
    yielded += (j == 0 ? "" : ", ") + changes[j];
    types += (j == 0 ? "" : ", ") + std::string("vector<3xf64>");
  }
  os << inner << "md.yield " << yielded << " : " << types << "\n"
     << indent << "} : !rel_" << set.name << ", !vec -> !vec\n";

  std::string virialName = virial.str();
  if (!virial.empty()) {
    // The forces of the constraints over the second half of the step,
    // G = 2 m Δv / dt, and their virial Σ (x_j − x_0) ⊗ G_j, with the
    // weight ½: the virial of the constraints is the mean of those of the
    // two halves (emitConstraintVirial). The changes are read, not solved
    // again, as for SETTLE (#97); the clusters are disjoint.
    std::string sum = (virialResult + "_sum").str();
    header(sum, "sum_tuples", change);
    SiteKernel w(os, inner);
    std::vector<std::string> dv(count + 1);
    for (unsigned j = 0; j <= count; ++j)
      dv[j] = "%vs_v" + std::to_string(j);
    std::string two = w.constant(1.0);
    std::string elements;
    std::vector<std::string> forces;
    for (unsigned j = 1; j <= count; ++j)
      forces.push_back(w.scale(
          w.real("divf", w.real("mulf", two, "%vs_m" + std::to_string(j)),
                 "%dt"),
          dv[j]));
    for (int a = 0; a != 3; ++a) {
      std::string row;
      for (unsigned j = 1; j <= count; ++j) {
        std::string term = w.scale(
            w.component("%vs_r" + std::to_string(j), a), forces[j - 1]);
        row = row.empty() ? term : w.vector("addf", row, term);
      }
      for (int b = 0; b != 3; ++b)
        elements += (elements.empty() ? "" : ", ") + w.component(row, b);
    }
    os << inner << "%vs_w = vector.from_elements " << elements
       << " : vector<9xf64>\n"
       << inner << "md.yield %vs_w : vector<9xf64>\n"
       << indent << "} : !rel_" << set.name << ", !vec -> vector<9xf64>\n";
    os << indent << virialResult << " = arith.addf " << virial << ", " << sum
       << " : vector<9xf64>\n";
    virialName = virialResult.str();
  }
  os << indent << result << " = md.map_particles gather(" << v << ", "
     << change << " : !vec, !vec) {\n"
     << indent << "^bb0(%vs_x: vector<3xf64>, %vs_d: vector<3xf64>):\n"
     << inner << "%vs_sum = arith.addf %vs_x, %vs_d : vector<3xf64>\n"
     << inner << "md.yield %vs_sum : vector<3xf64>\n"
     << indent << "} : !vec\n";
  return virialName;
}

std::string Builder::emitConstraintForceVirial(StringRef indent,
                                               StringRef x, StringRef v,
                                               StringRef f,
                                               StringRef virial) {
  std::string inner = (indent + "  ").str();
  std::string current = virial.str();
  unsigned count = 0;
  // A group of `members` particles and its bonds (p, q), with the
  // displacements of the members from the first.
  auto emitGroup = [&](StringRef setName, unsigned members,
                       llvm::ArrayRef<std::pair<unsigned, unsigned>> bonds,
                       StringRef tupleFields, StringRef tupleTypes,
                       unsigned numTupleFields) {
    std::string coordinates, arguments;
    for (unsigned j = 1; j != members; ++j) {
      coordinates += (j == 1 ? "" : ", ") +
                     ("displacement(" + std::to_string(j) + ", 0)");
      arguments += "%cf_r" + std::to_string(j) + ": vector<3xf64>, ";
    }
    for (StringRef field : {"v", "f"})
      for (unsigned j = 0; j != members; ++j)
        arguments += ("%cf_" + field + std::to_string(j) + ": vector<3xf64>, ").str();
    for (unsigned j = 0; j != members; ++j)
      arguments += "%cf_m" + std::to_string(j) + ": f64, ";
    for (unsigned j = 0; j != numTupleFields; ++j)
      arguments += "%cf_t" + std::to_string(j) + ": f64, ";
    arguments.resize(arguments.size() - 2);
    std::string sum = "%cfw" + std::to_string(count++);
    os << indent << sum << " = md.sum_tuples %r_" << setName << ", " << x
       << ", %cell\n"
       << indent << "    coordinates(" << coordinates << ")\n"
       << indent << "    gather(" << v << ", " << f << ", %m : !vec, !vec, "
       << "!real)\n";
    if (numTupleFields)
      os << indent << "    tuple(" << tupleFields << " : " << tupleTypes
         << ")";
    os << " {\n" << indent << "^bb0(" << arguments << "):\n";
    SiteKernel k(os, inner, "%cfk");
    std::string zero = k.constant(0.0), one = k.constant(1.0);
    // The position of member j relative to the first, and its velocity,
    // acceleration without the constraints, and inverse mass.
    auto position = [&](unsigned j) { return "%cf_r" + std::to_string(j); };
    std::vector<std::string> u, inverse;
    for (unsigned j = 0; j != members; ++j) {
      std::string n = std::to_string(j);
      inverse.push_back(k.real("divf", one, "%cf_m" + n));
      u.push_back(k.scale(inverse.back(), "%cf_f" + n));
    }
    std::vector<std::string> units, rhs;
    for (auto [p, q] : bonds) {
      std::string d = p == 0 ? position(q)
                             : k.vector("subf", position(q), position(p));
      std::string length = k.norm(d);
      units.push_back(k.unit(d));
      std::string relative = k.vector("subf", "%cf_v" + std::to_string(q),
                                      "%cf_v" + std::to_string(p));
      std::string centripetal =
          k.real("divf", k.dot(relative, relative), length);
      std::string along =
          k.dot(units.back(), k.vector("subf", u[q], u[p]));
      rhs.push_back(k.real("subf", zero, k.real("addf", along, centripetal)));
    }
    // A_bc: what a unit force along bond c, +e_c on q_c and -e_c on p_c,
    // does to the acceleration along bond b.
    size_t n = bonds.size();
    std::vector<std::vector<std::string>> A(n, std::vector<std::string>(n));
    for (size_t b = 0; b != n; ++b)
      for (size_t c = 0; c != n; ++c) {
        auto [pb, qb] = bonds[b];
        auto [pc, qc] = bonds[c];
        auto on = [&](unsigned x) {
          return (x == qc ? 1 : 0) - (x == pc ? 1 : 0);
        };
        std::string coefficient = zero;
        if (on(qb) != 0)
          coefficient = k.real(on(qb) > 0 ? "addf" : "subf", coefficient,
                               inverse[qb]);
        if (on(pb) != 0)
          coefficient = k.real(on(pb) > 0 ? "subf" : "addf", coefficient,
                               inverse[pb]);
        A[b][c] = k.real("mulf", k.dot(units[b], units[c]), coefficient);
      }
    std::vector<std::string> tension = emitSmallSolve(k, A, rhs);
    // The force on each member but the first, and the virial.
    std::string elements;
    std::vector<std::string> forces(members, "");
    for (unsigned j = 1; j != members; ++j) {
      std::string force = k.zero();
      for (size_t c = 0; c != n; ++c) {
        auto [pc, qc] = bonds[c];
        if (j != pc && j != qc)
          continue;
        std::string push = k.scale(tension[c], units[c]);
        force = k.vector(j == qc ? "addf" : "subf", force, push);
      }
      forces[j] = force;
    }
    for (int a = 0; a != 3; ++a) {
      std::string row;
      for (unsigned j = 1; j != members; ++j) {
        std::string term = k.scale(k.component(position(j), a), forces[j]);
        row = row.empty() ? term : k.vector("addf", row, term);
      }
      for (int b = 0; b != 3; ++b)
        elements += (elements.empty() ? "" : ", ") + k.component(row, b);
    }
    os << inner << "%cf_w = vector.from_elements " << elements
       << " : vector<9xf64>\n"
       << inner << "md.yield %cf_w : vector<9xf64>\n"
       << indent << "} : !rel_" << setName << ", !vec -> vector<9xf64>\n";
    std::string next = sum + "_total";
    os << indent << next << " = arith.addf " << current << ", " << sum
       << " : vector<9xf64>\n";
    current = next;
  };
  for (const Program::TupleSet &set : program.tupleSets) {
    if (set.name == "settles") {
      static const std::pair<unsigned, unsigned> bonds[] = {
          {0, 1}, {0, 2}, {1, 2}};
      emitGroup(set.name, 3, bonds, "", "", 0);
    }
  }
  for (const Program::TupleSet *set : getShakeSets()) {
    std::vector<std::pair<unsigned, unsigned>> bonds;
    for (unsigned j = 1; j != set->arity; ++j)
      bonds.push_back({0, j});
    emitGroup(set->name, set->arity, bonds, "", "", 0);
  }
  return current;
}

std::string Builder::emitSettleVelocities(StringRef indent, StringRef x,
                                          StringRef v, StringRef result,
                                          StringRef virial,
                                          StringRef virialResult) {
  std::string inner = (indent + "  ").str();
  // The impulses along the three bonds that leave no velocity along them:
  // A τ = −b, with b the velocities along the bonds and A what a unit
  // impulse along one bond does to the velocity along another.
  auto emitKernel = [&](SiteKernel &k) {
    std::string e0 = k.unit("%vs_d10"), e1 = k.unit("%vs_d20"),
                e2 = k.unit("%vs_d21");
    std::string one = k.constant(1.0);
    std::string io = k.real("divf", one, "%vs_m0");
    std::string ih = k.real("divf", one, "%vs_m1");
    std::string zero = k.constant(0.0);
    auto minus = [&](StringRef a) { return k.real("subf", zero, a); };
    std::string r0 =
        minus(k.dot(e0, k.vector("subf", "%vs_v1", "%vs_v0")));
    std::string r1 =
        minus(k.dot(e1, k.vector("subf", "%vs_v2", "%vs_v0")));
    std::string r2 =
        minus(k.dot(e2, k.vector("subf", "%vs_v2", "%vs_v1")));
    std::string a00 = k.real("addf", io, ih), a11 = a00;
    std::string a22 = k.real("addf", ih, ih);
    std::string a01 = k.real("mulf", k.dot(e0, e1), io);
    std::string a02 = minus(k.real("mulf", k.dot(e0, e2), ih));
    std::string a12 = k.real("mulf", k.dot(e1, e2), ih);
    auto det3 = [&](StringRef m00, StringRef m01, StringRef m02,
                    StringRef m10, StringRef m11, StringRef m12,
                    StringRef m20, StringRef m21, StringRef m22) {
      auto minor = [&](StringRef p, StringRef q, StringRef r, StringRef s) {
        return k.real("subf", k.real("mulf", p, q), k.real("mulf", r, s));
      };
      return k.real(
          "addf",
          k.real("subf", k.real("mulf", m00, minor(m11, m22, m12, m21)),
                 k.real("mulf", m01, minor(m10, m22, m12, m20))),
          k.real("mulf", m02, minor(m10, m21, m11, m20)));
    };
    std::string det = det3(a00, a01, a02, a01, a11, a12, a02, a12, a22);
    std::string t0 = k.real(
        "divf", det3(r0, a01, a02, r1, a11, a12, r2, a12, a22), det);
    std::string t1 = k.real(
        "divf", det3(a00, r0, a02, a01, r1, a12, a02, r2, a22), det);
    std::string t2 = k.real(
        "divf", det3(a00, a01, r0, a01, a11, r1, a02, a12, r2), det);
    // Impulse c along e_c from p to q: +τ e / m_q on q, −τ e / m_p on p.
    std::string i0 = k.scale(t0, e0), i1 = k.scale(t1, e1),
                i2 = k.scale(t2, e2);
    std::string dv0 = k.negate(k.scale(io, k.vector("addf", i0, i1)));
    std::string dv1 = k.scale(ih, k.vector("subf", i0, i2));
    std::string dv2 = k.scale(ih, k.vector("addf", i1, i2));
    return std::array<std::string, 3>{dv0, dv1, dv2};
  };
  auto header = [&](StringRef op, StringRef field) {
    os << indent << " = md." << op << " %r_settles, " << x << ", %cell\n"
       << indent << "    coordinates(displacement(1, 0), displacement(2, 0), "
                    "displacement(2, 1))\n"
       << indent << "    gather(" << field << ", %m : !vec, !real)\n"
       << indent << "    tuple(%f_settles_doh, %f_settles_dhh : !of_settles, "
                    "!of_settles) {\n"
       << indent << "^bb0(%vs_d10: vector<3xf64>, %vs_d20: vector<3xf64>, "
                    "%vs_d21: vector<3xf64>, %vs_v0: vector<3xf64>, "
                    "%vs_v1: vector<3xf64>, %vs_v2: vector<3xf64>, "
                    "%vs_m0: f64, %vs_m1: f64, %vs_m2: f64, %vs_doh: f64, "
                    "%vs_dhh: f64):\n";
  };
  std::string change = (result + "_settle").str();
  os << indent << change;
  header("gather_tuples", v);
  SiteKernel k(os, inner);
  std::array<std::string, 3> dv = emitKernel(k);
  os << inner << "md.yield " << dv[0] << ", " << dv[1] << ", " << dv[2]
     << " : vector<3xf64>, vector<3xf64>, vector<3xf64>\n"
     << indent << "} : !rel_settles, !vec -> !vec\n";

  std::string virialName = virial.str();
  if (!virial.empty()) {
    // The forces of the constraints over the second half of the step,
    // G = 2 m Δv / dt, and their virial Σ (x_i − x_O) ⊗ G_i, with the
    // weight ½ (emitConstraintVirial). The changes Δv are those that the
    // step applies, read rather than solved again: a solve of its own
    // could round otherwise once fused with the step's (#97), and the
    // dynamics of a step would depend on whether it writes energies. The
    // groups are disjoint, so a particle's change is its group's.
    std::string sum = (virialResult + "_settle").str();
    os << indent << sum;
    header("sum_tuples", change);
    SiteKernel w(os, inner);
    std::string factor = w.real("divf", "%vs_m1", "%dt");
    std::array<std::string, 2> arms = {"%vs_d10", "%vs_d20"};
    std::array<std::string, 2> forces = {w.scale(factor, "%vs_v1"),
                                         w.scale(factor, "%vs_v2")};
    std::string elements;
    for (int a = 0; a != 3; ++a) {
      std::string row;
      for (int m = 0; m != 2; ++m) {
        std::string term = w.scale(w.component(arms[m], a), forces[m]);
        row = row.empty() ? term : w.vector("addf", row, term);
      }
      for (int b = 0; b != 3; ++b)
        elements += (elements.empty() ? "" : ", ") + w.component(row, b);
    }
    os << inner << "%vs_w = vector.from_elements " << elements
       << " : vector<9xf64>\n"
       << inner << "md.yield %vs_w : vector<9xf64>\n"
       << indent << "} : !rel_settles, !vec -> vector<9xf64>\n";
    os << indent << virialResult << " = arith.addf " << virial << ", "
       << sum << " : vector<9xf64>\n";
    virialName = virialResult.str();
  }
  os << indent << result << " = md.map_particles gather(" << v << ", "
     << change << " : !vec, !vec) {\n"
     << indent << "^bb0(%vs_x: vector<3xf64>, %vs_d: vector<3xf64>):\n"
     << inner << "%vs_sum = arith.addf %vs_x, %vs_d : vector<3xf64>\n"
     << inner << "md.yield %vs_sum : vector<3xf64>\n"
     << indent << "} : !vec\n";
  return virialName;
}

void Builder::emitConstrainedDescent(StringRef indent, StringRef x,
                                     StringRef f, StringRef result) {
  bool settles = hasSettles();
  std::vector<const Program::TupleSet *> shakeSets = getShakeSets();
  unsigned steps = (settles ? 1 : 0) + shakeSets.size(), step = 0;
  std::string current = (result + "_all").str();
  if (steps == 0)
    current = result.str();
  emitIRFragment(
      indent, {current}, {"!vec"}, {f, "%m"}, {"!vec", "!real"},
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        using namespace mlir;
        auto loc = builder.getUnknownLoc();
        auto f64 = builder.getF64Type();
        auto vec = VectorType::get({3}, f64);
        auto map = mdir::md::MapParticlesOp::create(
            builder, loc, values[0].getType(), values);
        {
          OpBuilder::InsertionGuard guard(builder);
          auto args = builder.createBlock(&map.getKernel(), {}, {vec, f64},
                                            {loc, loc})->getArguments();
          Value zero = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(0.0));
          Value unit = arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(1.0));
          Value zeroVector = arith::ConstantOp::create(
              builder, loc, DenseElementsAttr::get(vec, 0.0));
          Value massive = arith::CmpFOp::create(builder, loc,
                                                arith::CmpFPredicate::OGT,
                                                args[1], zero);
          Value safe = arith::SelectOp::create(builder, loc, massive, args[1], unit);
          Value mass = vector::BroadcastOp::create(builder, loc, vec, safe);
          Value ratio = arith::DivFOp::create(builder, loc, args[0], mass);
          Value result = arith::SelectOp::create(builder, loc, massive, ratio,
                                                 zeroVector);
          mdir::md::YieldOp::create(builder, loc, ValueRange{result});
        }
        return {map.getResult()};
      });
  // As RATTLE takes the velocities off the constraints.
  auto next = [&]() {
    ++step;
    return step == steps ? result.str()
                         : (result + "_c" + std::to_string(step)).str();
  };
  if (settles) {
    std::string projected = next();
    emitSettleVelocities(indent, x, current, projected);
    current = projected;
  }
  for (const Program::TupleSet *set : shakeSets) {
    std::string projected = next();
    emitShakeVelocities(indent, x, current, *set, projected, "", "");
    current = projected;
  }
}

std::string Builder::emitStartConstraintTrace(StringRef x, StringRef f,
                                              StringRef v, StringRef trace,
                                              bool axes) {
  // With `axes`, the diagonal instead of the trace, each axis with twice
  // its own internal kinetic energy: the identity holds for the trace, and
  // for each axis only in the mean, which serves the first scaling of a
  // run (D119).
  std::string name = axes ? "%dgc0" : "%trc0";
  std::string type = axes ? "vector<3xf64>" : "f64";
  std::string descent = axes ? "%gc0d" : "%gc0";
  emitConstrainedDescent("  ", x, f, descent);
  std::vector<const Program::TupleSet *> groups = getShakeSets();
  for (const Program::TupleSet &set : program.tupleSets)
    if (set.name == "settles")
      groups.insert(groups.begin(), &set);
  std::string current = trace.str();
  for (const Program::TupleSet *set : groups) {
    unsigned count = set->arity - 1;
    std::string coordinates, arguments;
    for (unsigned k = 1; k <= count; ++k) {
      coordinates += (k == 1 ? "" : ", ") +
                     ("displacement(" + std::to_string(k) + ", 0)");
      arguments += "%vs_r" + std::to_string(k) + ": vector<3xf64>, ";
    }
    for (StringRef field : {"g", "f", "v"})
      for (unsigned k = 0; k <= count; ++k)
        arguments += ("%vs_" + field + std::to_string(k) + ": vector<3xf64>, ")
                         .str();
    for (unsigned k = 0; k <= count; ++k)
      arguments += "%vs_m" + std::to_string(k) + ": f64" +
                   (k == count ? "" : ", ");
    std::string sum = name + "_" + set->name;
    os << "  " << sum << " = md.sum_tuples %r_" << set->name << ", " << x
       << ", %cell\n"
       << "    coordinates(" << coordinates << ")\n"
       << "    gather(" << descent << ", " << f << ", " << v
       << ", %m : !vec, !vec, !vec, !real) {\n"
       << "  ^bb0(" << arguments << "):\n";
    SiteKernel k(os, "    ");
    // Σ (x_k − x_0) · G⁰_k over the particles but the first, whose arm is
    // 0.
    std::string arms;
    for (unsigned j = 1; j <= count; ++j) {
      std::string n = std::to_string(j);
      std::string force = k.vector(
          "subf", k.scale("%vs_m" + n, "%vs_g" + n), "%vs_f" + n);
      std::string term = axes ? k.vector("mulf", "%vs_r" + n, force)
                              : k.dot("%vs_r" + n, force);
      arms = arms.empty() ? term
                          : axes ? k.vector("addf", arms, term)
                                 : k.real("addf", arms, term);
    }
    // 2 K_int = Σ m |v − V|².
    std::string total = "%vs_m0", moment = k.scale("%vs_m0", "%vs_v0");
    for (unsigned j = 1; j <= count; ++j) {
      std::string n = std::to_string(j);
      total = k.real("addf", total, "%vs_m" + n);
      moment = k.vector("addf", moment, k.scale("%vs_m" + n, "%vs_v" + n));
    }
    std::string center =
        k.scale(k.real("divf", k.constant(1.0), total), moment);
    std::string twice;
    for (unsigned j = 0; j <= count; ++j) {
      std::string n = std::to_string(j);
      std::string relative = k.vector("subf", "%vs_v" + n, center);
      std::string term =
          axes ? k.scale("%vs_m" + n, k.vector("mulf", relative, relative))
               : k.real("mulf", "%vs_m" + n, k.dot(relative, relative));
      twice = twice.empty() ? term
                            : axes ? k.vector("addf", twice, term)
                                   : k.real("addf", twice, term);
    }
    std::string virial = axes ? k.vector("subf", arms, twice)
                              : k.real("subf", arms, twice);
    os << "    md.yield " << virial << " : " << type << "\n"
       << "  } : !rel_" << set->name << ", !vec -> " << type << "\n";
    std::string next = name + "_" + set->name + "_sum";
    os << "  " << next << " = arith.addf " << current << ", " << sum
       << " : " << type << "\n";
    current = next;
  }
  return current;
}

void Builder::emitConstraintFailures(StringRef indent, StringRef x,
                                     StringRef result) {
  auto counts = emitConstraintTests(indent, x, result, /*flag=*/false);
  llvm::SmallVector<StringRef> inputs, types;
  for (const auto &count : counts) {
    inputs.push_back(count.second);
    types.push_back("f64");
  }
  emitIRFragment(
      indent, {result}, {"f64"}, inputs, types,
      [](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        using namespace mlir;
        auto loc = builder.getUnknownLoc();
        Value current;
        for (Value value : values)
          current = current ? arith::AddFOp::create(builder, loc, current, value)
                            : value;
        Value none = arith::ConstantOp::create(builder, loc,
                                               builder.getF64FloatAttr(0.0));
        if (!current)
          current = none;
        return {arith::AddFOp::create(builder, loc, current, none)};
      });
}

void Builder::emitConstraintStops(
    StringRef indent,
    const std::vector<std::pair<int, std::string>> &failures) {
  if (failures.empty())
    return;
  // The preceding step evaluation has already waited for the stream's flag;
  // reading it here adds no wait (D118). Stops keep their existing effects.
  llvm::SmallVector<StringRef> inputs = {"%noise_step"}, types = {"i64"};
  std::vector<int> sets;
  for (const auto &[set, failed] : failures) {
    sets.push_back(set);
    inputs.push_back(failed);
    types.push_back("i1");
  }
  emitIRFragment(
      indent, {}, {}, inputs, types,
      [sets](mlir::OpBuilder &builder, mlir::ValueRange values) -> TermIRValues {
        using namespace mlir;
        auto loc = builder.getUnknownLoc();
        for (auto [index, set] : llvm::enumerate(sets)) {
          auto branch = scf::IfOp::create(builder, loc, values[index + 1], false);
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(branch.thenBlock());
          Value number = arith::ConstantIntOp::create(builder, loc, set, 64);
          func::CallOp::create(builder, loc, "mdrtStopConstraintFailure",
                               TypeRange{}, ValueRange{values[0], number});
        }
        func::CallOp::create(builder, loc, "mdrtStopPending", TypeRange{},
                             ValueRange{});
        return {};
      });
}

std::vector<std::pair<int, std::string>>
Builder::emitConstraintTests(StringRef indent, StringRef x, StringRef prefix,
                             bool flag) {
  // A length fails if |r²/d² − 1| is above the tolerance or is not a
  // number (an unordered comparison). Below double precision the lengths of
  // intact waters of a box of 150 Å, stored in f32, are off by up to
  // 4·10⁻⁷ (measured, #343), and a water that the constraints did not take
  // back to its shape by 10⁻¹ or more: 10⁻⁴ lies between them. In double
  // precision SETTLE and SHAKE meet them to the rounding of f64.
  double tolerance = control.precision == Precision::Double ? 1.0e-8 : 1.0e-4;
  struct TestBond {
    unsigned first, second, length;
  };
  struct Group {
    std::string set;
    std::vector<TestBond> bonds;
    std::vector<std::string> fields;
    int number;
  };
  std::vector<Group> groups;
  if (hasSettles())
    groups.push_back({"settles", {{0, 1, 0}, {0, 2, 0}, {1, 2, 1}},
                      {"%f_settles_doh", "%f_settles_dhh"}, 0});
  for (const Program::TupleSet *set : getShakeSets()) {
    Group group{set->name, {}, {}, static_cast<int>(set->arity) - 1};
    for (unsigned k = 1; k < set->arity; ++k) {
      group.bonds.push_back({0, k, k - 1});
      group.fields.push_back("%f_" + set->name + "_d" + std::to_string(k));
    }
    groups.push_back(std::move(group));
  }
  std::vector<std::pair<int, std::string>> results;
  for (const Group &group : groups) {
    std::string relation = "%r_" + group.set;
    std::string relationType = "!rel_" + group.set;
    std::string fieldType = "!of_" + group.set;
    std::string sum = (prefix + "_" + group.set).str();
    llvm::SmallVector<StringRef> inputs = {relation, x, "%cell"};
    llvm::SmallVector<StringRef> inputTypes = {relationType, "!vec", "!md.cell"};
    for (const auto &field : group.fields) {
      inputs.push_back(field);
      inputTypes.push_back(fieldType);
    }
    emitIRFragment(
        indent, {sum}, {flag ? "i1" : "f64"}, inputs, inputTypes,
        [bonds = group.bonds, flag, tolerance](mlir::OpBuilder &builder,
                                               mlir::ValueRange values) -> TermIRValues {
          using namespace mlir;
          auto loc = builder.getUnknownLoc();
          auto f64 = builder.getF64Type();
          auto vec = VectorType::get({3}, f64);
          llvm::SmallVector<mdir::md::Coordinate> coordinates;
          for (const auto &bond : bonds)
            coordinates.push_back({mdir::md::CoordinateKind::Displacement,
                                    {bond.second, bond.first}});
          DenseI32ArrayAttr kinds;
          DenseI64ArrayAttr members;
          mdir::md::getCoordinateAttrs(builder, coordinates, kinds, members);
          Type resultType = flag ? Type(builder.getI1Type()) : Type(f64);
          auto sum = mdir::md::SumTuplesOp::create(
              builder, loc, resultType, values[0], values[1], values[2],
              ValueRange{}, values.drop_front(3), kinds, members);
          {
            OpBuilder::InsertionGuard guard(builder);
            llvm::SmallVector<Type> types(bonds.size(), vec);
            types.append(values.size() - 3, f64);
            llvm::SmallVector<Location> locations(types.size(), loc);
            auto args = builder.createBlock(&sum.getKernel(), {}, types,
                                              locations)->getArguments();
            auto constant = [&](double value) -> Value {
              return arith::ConstantOp::create(builder, loc,
                                                builder.getF64FloatAttr(value));
            };
            Value one = constant(1.0), zero = constant(0.0), limit = constant(tolerance);
            Value count;
            for (auto [index, bond] : llvm::enumerate(bonds)) {
              Value length = args[bonds.size() + bond.length];
              Value d2 = arith::MulFOp::create(builder, loc, length, length);
              Value r = args[index];
              Value square = arith::MulFOp::create(builder, loc, r, r);
              Value norm = vector::ReductionOp::create(
                  builder, loc, vector::CombiningKind::ADD, square);
              Value ratio = arith::DivFOp::create(builder, loc, norm, d2);
              Value off = arith::SubFOp::create(builder, loc, ratio, one);
              Value magnitude = math::AbsFOp::create(builder, loc, off);
              Value bad = arith::CmpFOp::create(builder, loc,
                                                arith::CmpFPredicate::UGT,
                                                magnitude, limit);
              if (flag) {
                count = count ? arith::OrIOp::create(builder, loc, count, bad) : bad;
              } else {
                Value failed = arith::SelectOp::create(builder, loc, bad, one, zero);
                count = count ? arith::AddFOp::create(builder, loc, count, failed)
                              : failed;
              }
            }
            mdir::md::YieldOp::create(builder, loc, ValueRange{count});
          }
          return {sum.getResult()};
        });
    results.push_back({group.number, sum});
  }
  return results;
}
