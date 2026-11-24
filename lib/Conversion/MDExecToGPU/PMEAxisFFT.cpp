// Gaussian spectral product on the last forward axis [Essmann1995].
#include "DevicePME.h"
#include "mdir/Conversion/AxisFFTEmitter.h"

using namespace mlir;
using namespace mdir::kernels;

func::FuncOp DevicePME::emitAxisFFT(StringRef name, ArrayRef<int64_t> grid,
                                    int64_t pitch, bool observed) {
  auto buffer = globalMem({ShapedType::kDynamic}, realType);
  SmallVector<Type> inputs{buffer, buffer, buffer}, results;
  if (observed) {
    inputs.append({globalMem({ShapedType::kDynamic}, f64Type), f64Type});
    results.append({f64Type, VectorType::get({9}, f64Type)});
  }
  auto fn =
      func::FuncOp::create(b, loc, name, b.getFunctionType(inputs, results));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value tables = entry->getArgument(2);
  Value rows, gauss;
  if (observed) {
    rows = entry->getArgument(3);
    Value beta = entry->getArgument(4);
    Value pi2 =
        arith::ConstantOp::create(b, loc, b.getF64FloatAttr(9.869604401089358));
    gauss = convert(realType, divf(pi2, mulf(beta, beta)));
  }
  auto pointwise = [&](OpBuilder &, Value x, Value y, Value z, FFTCoefficient c,
                       llvm::MutableArrayRef<Value> sums) {
    SmallVector<Value, 3> wave;
    Value square, factor;
    int64_t offset = 0;
    Value indices[] = {x, y, z};
    for (auto [k, length] : llvm::zip(indices, grid)) {
      auto component = [&](int field) {
        Value start =
            arith::ConstantIndexOp::create(b, loc, offset + field * length);
        return load(tables, {addi(start, k)});
      };
      if (observed)
        wave.push_back(component(0));
      Value s = component(1), d = component(2);
      square = square ? addf(square, s) : s;
      factor = factor ? mulf(factor, d) : d;
      offset += 3 * length;
    }
    Value zero = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0));
    Value one = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1));
    Value origin = cmpf(arith::CmpFPredicate::OEQ, square, zero);
    Value inverse = divf(one, select(origin, one, square));
    Value influence = select(origin, zero, mulf(factor, inverse));
    if (observed) {
      Value last = arith::ConstantIndexOp::create(b, loc, grid[2] / 2);
      Value first = arith::ConstantIndexOp::create(b, loc, 0);
      Value edge = ori(cmpi(arith::CmpIPredicate::eq, z, first),
                       cmpi(arith::CmpIPredicate::eq, z, last));
      Value half = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.5));
      Value e = mulf(mulf(select(edge, half, one), influence),
                     addf(mulf(c.re, c.re), mulf(c.im, c.im)));
      sums[0] = addf(sums[0], e);
      Value h = addf(inverse, gauss);
      Value eh = mulf(e, addf(h, h));
      int col = 1;
      for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j, ++col)
          sums[col] = addf(sums[col], subf(i == j ? e : zero,
                                           mulf(eh, mulf(wave[i], wave[j]))));
    }
    return FFTCoefficient{mulf(c.re, influence), mulf(c.im, influence)};
  };
  auto partial = [&](Value block, ValueRange sums) {
    Value base = muli(block, arith::ConstantIndexOp::create(b, loc, 10));
    for (auto [i, sum] : llvm::enumerate(sums))
      store(convert(f64Type, sum), rows,
            {addi(base, arith::ConstantIndexOp::create(b, loc, i))});
  };
  int blocks = emitAxisConvolution(b, loc, grid, pitch, entry->getArgument(0),
                                   entry->getArgument(1), pointwise,
                                   observed ? 7 : 0, partial);
  if (observed)
    finishConvolution(rows, arith::ConstantIndexOp::create(b, loc, blocks));
  else
    func::ReturnOp::create(b, loc);
  return fn;
}
