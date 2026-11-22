#include "HostPME.h"

using namespace mlir;
using namespace mdir::kernels;

// Raises the weights `a` of the B-spline at the fraction `w` from order
// k − 1 to order k, in place (the recursion of Essmann et al., Eq. 2.28).
void HostPME::emitRaise() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_raise",
              b.getFunctionType(
                  {realType, idxType, MemRefType::get({8}, realType)}, {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value w = entry->getArgument(0);
  Value k = entry->getArgument(1);
  Value a = entry->getArgument(2);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
  Value one = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
  raiseSpline(w, k, a, {}, c1, c2, one);
  func::ReturnOp::create(b, loc);
}

// The weights `values` of the B-spline of order `n` at the fraction `w`,
// and their derivatives `slopes` with respect to the fractional coordinate.
// The fraction is in f64, and is rounded once to the grid precision.
void HostPME::emitBspline() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_bspline",
              b.getFunctionType(
                  {f64Type, idxType, MemRefType::get({8}, realType),
                   MemRefType::get({8}, realType),
                   MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   f64Type)},
                  {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value w = entry->getArgument(0);
  Value n = entry->getArgument(1);
  Value values = entry->getArgument(2);
  Value slopes = entry->getArgument(3);
  Value moduli = entry->getArgument(4);
  Value r0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value r5 = arith::ConstantIndexOp::create(b, loc, 5);
  Value rows = memref::DimOp::create(b, loc, moduli, r0);
  Value prolate =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, rows, r5);
  auto ifOp1 = scf::IfOp::create(b, loc, TypeRange{}, prolate, true, true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(ifOp1.thenBlock());
    func::CallOp::create(b, loc, "mdrt.pme_window", TypeRange{},
                         ValueRange{w, n, values, slopes, moduli});
    scf::YieldOp::create(b, loc);
  }
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(ifOp1.elseBlock());
    Value wr = convert(realType, w);
    func::CallOp::create(b, loc, "mdrt.pme_spline", TypeRange{},
                         ValueRange{wr, n, values, slopes});
    scf::YieldOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

// The weights of the window of the prolate kernel at the fraction `w`
// of the place (Builder::collectProlate, D266): weight j is the
// Chebyshev series of row 3 of `moduli`, at offset j D, in t = 2w − 1,
// and its slope in the scaled coordinate the series at offset (n + j) D,
// by Clenshaw's recurrence [Liang2026, eq. 14]. Computed in f64 and
// rounded once to the grid precision.
void HostPME::emitWindow() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_window",
              b.getFunctionType(
                  {f64Type, idxType, MemRefType::get({8}, realType),
                   MemRefType::get({8}, realType),
                   MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   f64Type)},
                  {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value w = entry->getArgument(0);
  Value n = entry->getArgument(1);
  Value values = entry->getArgument(2);
  Value slopes = entry->getArgument(3);
  Value moduli = entry->getArgument(4);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c3 = arith::ConstantIndexOp::create(b, loc, 3);
  Value c4 = arith::ConstantIndexOp::create(b, loc, 4);
  Value one = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
  Value two = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 2.0));
  Value df = load(moduli, {c4, c0});
  Value di = arith::FPToSIOp::create(b, loc, i64Type, df);
  Value d = arith::IndexCastOp::create(b, loc, idxType, di);
  Value tw = arith::MulFOp::create(b, loc, two, w);
  Value t = arith::SubFOp::create(b, loc, tw, one);
  Value t2 = arith::MulFOp::create(b, loc, two, t);
  Value nd = arith::MulIOp::create(b, loc, n, d);
  scf::ForOp::create(
      b, loc, c0, n, c1, ValueRange{},
      [&](OpBuilder &b, Location, Value j, ValueRange carried) {
        Value jd = arith::MulIOp::create(b, loc, j, d);
        Value sd = arith::AddIOp::create(b, loc, nd, jd);
        Value v = func::CallOp::create(b, loc, "mdrt.pme_clenshaw",
                                       TypeRange{f64Type},
                                       ValueRange{moduli, c3, jd, d, t2, t})
                      .getResult(0);
        Value sl = func::CallOp::create(b, loc, "mdrt.pme_clenshaw",
                                        TypeRange{f64Type},
                                        ValueRange{moduli, c3, sd, d, t2, t})
                       .getResult(0);
        Value vr = convert(realType, v);
        Value slr = convert(realType, sl);
        store(vr, values, {j});
        store(slr, slopes, {j});
        scf::YieldOp::create(b, loc);
      });
  func::ReturnOp::create(b, loc);
}

// Σ_k a_k T_k(t) of the `count` coefficients from `moduli`[row, at], by
// Clenshaw's recurrence; `t2` = 2t.
void HostPME::emitClenshaw() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_clenshaw",
              b.getFunctionType(
                  {MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   f64Type),
                   idxType, idxType, idxType, f64Type, f64Type},
                  {f64Type}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value moduli = entry->getArgument(0);
  Value row = entry->getArgument(1);
  Value at = entry->getArgument(2);
  Value count = entry->getArgument(3);
  Value t2 = entry->getArgument(4);
  Value t = entry->getArgument(5);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value zero = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 0.0));
  Value v = evaluateSeries(count, c1, t, t2, zero, [&](Value k) {
    Value column = k ? arith::AddIOp::create(b, loc, at, k).getResult() : at;
    return load(moduli, {row, column});
  });
  func::ReturnOp::create(b, loc, ValueRange{v});
}

// The weights of the B-spline, as pme_bspline gives them for a table of
// the Gaussian kernel.
void HostPME::emitSpline() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare(
      "mdrt.pme_spline",
      b.getFunctionType({realType, idxType, MemRefType::get({8}, realType),
                         MemRefType::get({8}, realType)},
                        {}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value w = entry->getArgument(0);
  Value n = entry->getArgument(1);
  Value values = entry->getArgument(2);
  Value slopes = entry->getArgument(3);
  splineWindow<false>(
      w, n, values, slopes, [&](Value k, Value, Value, Value, Value) {
        func::CallOp::create(b, loc, "mdrt.pme_raise", TypeRange{},
                             ValueRange{w, k, values});
      });
  func::ReturnOp::create(b, loc);
}

// The first point of the grid that a particle at `x` along an edge of the
// length `length` with `k` points adds to, plus k so that it is not
// negative, and the fraction of its coordinate.
void HostPME::emitPlace() {
  OpBuilder::InsertionGuard guard(b);
  auto fn =
      declare("mdrt.pme_place",
              b.getFunctionType(
                  {f64Type, f64Type, idxType, idxType,
                   MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                   f64Type)},
                  {idxType, f64Type}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value x = entry->getArgument(0);
  Value length = entry->getArgument(1);
  Value k = entry->getArgument(2);
  Value n = entry->getArgument(3);
  Value moduli = entry->getArgument(4);
  Value r0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value r2 = arith::ConstantIndexOp::create(b, loc, 2);
  Value r5 = arith::ConstantIndexOp::create(b, loc, 5);
  Value rows = memref::DimOp::create(b, loc, moduli, r0);
  Value prolate =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, rows, r5);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value s = arith::DivFOp::create(b, loc, x, length);
  Value fs = math::FloorOp::create(b, loc, s);
  Value frac = arith::SubFOp::create(b, loc, s, fs);
  Value ki = arith::IndexCastOp::create(b, loc, i64Type, k);
  Value kf = arith::SIToFPOp::create(b, loc, f64Type, ki);
  Value u = arith::MulFOp::create(b, loc, frac, kf);
  Value fu0 = math::FloorOp::create(b, loc, u);
  Value nil = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 0.0));
  Value fu = arith::MaxNumFOp::create(b, loc, fu0, nil);
  Value w = arith::SubFOp::create(b, loc, u, fu);
  Value bi = arith::FPToSIOp::create(b, loc, i64Type, fu);
  Value accB = arith::IndexCastOp::create(b, loc, idxType, bi);
  Value bk = arith::AddIOp::create(b, loc, accB, k);
  Value bkn = arith::SubIOp::create(b, loc, bk, n);
  Value start = arith::AddIOp::create(b, loc, bkn, c1);
  Value nodd = arith::RemUIOp::create(b, loc, n, r2);
  Value odd = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, nodd, r0);
  Value halfp = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 0.5));
  Value h = arith::SelectOp::create(b, loc, odd, halfp, nil);
  Value uk = arith::AddFOp::create(b, loc, u, kf);
  Value uh = arith::SubFOp::create(b, loc, uk, h);
  Value fp0 = math::FloorOp::create(b, loc, uh);
  Value fp = arith::MaxNumFOp::create(b, loc, fp0, nil);
  Value wp = arith::SubFOp::create(b, loc, uh, fp);
  Value pi = arith::FPToSIOp::create(b, loc, i64Type, fp);
  Value pb = arith::IndexCastOp::create(b, loc, idxType, pi);
  Value below = arith::DivUIOp::create(b, loc, n, r2);
  Value pbn = arith::SubIOp::create(b, loc, pb, below);
  Value pstart = arith::AddIOp::create(b, loc, pbn, c1);
  Value first = arith::SelectOp::create(b, loc, prolate, pstart, start);
  Value fraction = arith::SelectOp::create(b, loc, prolate, wp, w);
  func::ReturnOp::create(b, loc, ValueRange{first, fraction});
}

// The rows of H⁻¹ of a triclinic cell `box` = (a_x, b_y, c_z, b_x, c_x,
// c_y): h00 = 1/a_x; h10 = −b_x/(a_x b_y), h11 = 1/b_y;
// h20 = (b_x c_y − b_y c_x)/(a_x b_y c_z), h21 = −c_y/(b_y c_z),
// h22 = 1/c_z.
void HostPME::emitInverseCell() {
  OpBuilder::InsertionGuard guard(b);
  auto fn = declare("mdrt.pme_inverse_cell",
                    b.getFunctionType({VectorType::get({6}, f64Type)},
                                      {f64Type, f64Type, f64Type, f64Type,
                                       f64Type, f64Type}));
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value box = entry->getArgument(0);
  Value lx = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{0});
  Value ly = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{1});
  Value lz = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{2});
  auto [h00, h10, h11, h20, h21, h22] = inverseCell(box, lx, ly, lz);
  func::ReturnOp::create(b, loc, ValueRange{h00, h10, h11, h20, h21, h22});
}
