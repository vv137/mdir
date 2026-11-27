// Spectral product on the last forward axis [Essmann1995, Liang2026].
#include "DevicePME.h"
#include "mdir/Conversion/AxisFFTEmitter.h"

using namespace mlir;
using namespace mdir::kernels;

func::FuncOp DevicePME::emitAxisFFT(StringRef name, ArrayRef<int64_t> grid,
                                    int64_t pitch, bool observed, bool tilted,
                                    bool radial) {
  auto buffer = globalMem({ShapedType::kDynamic}, realType);
  SmallVector<Type> inputs{buffer, buffer, buffer}, results;
  if (observed) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, f64Type));
    results.append({f64Type, VectorType::get({9}, f64Type)});
  }
  if (!radial && (observed || tilted))
    inputs.push_back(f64Type);
  if (tilted)
    inputs.push_back(VectorType::get({6}, f64Type));
  auto fn =
      func::FuncOp::create(b, loc, name, b.getFunctionType(inputs, results));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value tables = entry->getArgument(2);
  Value rows, gauss;
  if (observed)
    rows = entry->getArgument(3);
  if (!radial && (observed || tilted)) {
    Value beta = entry->getArgument(observed ? 4 : 3);
    Value pi2 =
        arith::ConstantOp::create(b, loc, b.getF64FloatAttr(9.869604401089358));
    gauss = convert(realType, divf(pi2, mulf(beta, beta)));
  }
  SmallVector<Value, 6> inverseBox;
  if (tilted) {
    Value box = entry->getArguments().back();
    auto [xx, yx, yy, zx, zy, zz] =
        inverseCell(box, extract(box, 0), extract(box, 1), extract(box, 2));
    for (Value v : {xx, yx, yy, zx, zy, zz})
      inverseBox.push_back(convert(realType, v));
  }
  auto pointwise = [&](OpBuilder &, Value x, Value y, Value z, FFTCoefficient c,
                       llvm::MutableArrayRef<Value> sums) {
    SmallVector<Value, 3> wave;
    Value square;
    SmallVector<Value, 3> factors;
    int64_t offset = 0;
    Value indices[] = {x, y, z};
    for (auto [k, length] : llvm::zip(indices, grid)) {
      auto component = [&](int field) {
        Value start =
            arith::ConstantIndexOp::create(b, loc, offset + field * length);
        return load(tables, {addi(start, k)});
      };
      if (observed || tilted)
        wave.push_back(component(0));
      if (!tilted) {
        Value s = component(1);
        square = square ? addf(square, s) : s;
      }
      factors.push_back(component(2));
      offset += 3 * length;
    }
    Value zero = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0));
    Value one = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1));
    if (tilted) {
      auto [i, j, k] = std::make_tuple(wave[0], wave[1], wave[2]);
      wave[0] = mulf(inverseBox[0], i);
      wave[1] = addf(mulf(inverseBox[1], i), mulf(inverseBox[2], j));
      wave[2] = addf(addf(mulf(inverseBox[3], i), mulf(inverseBox[4], j)),
                     mulf(inverseBox[5], k));
      for (Value v : wave)
        square = square ? addf(square, mulf(v, v)) : mulf(v, v);
    }
    Influence mode;
    if (radial) {
      Value metadata = arith::ConstantIndexOp::create(b, loc, offset);
      Value stride = arith::ConstantIndexOp::create(b, loc, 1);
      mode = radialInfluence(tables, metadata, stride, square, one, factors[0],
                             factors[1], factors[2]);
    } else if (tilted) {
      mode = gaussianInfluenceTriclinic(square, zero, one, factors[0],
                                        factors[1], factors[2], gauss);
    } else {
      mode = gaussianInfluence(square, zero, one, factors[0], factors[1], factors[2]);
    }
    Value influence = mode.factor;
    if (observed) {
      Value last = arith::ConstantIndexOp::create(b, loc, grid[2] / 2);
      Value first = arith::ConstantIndexOp::create(b, loc, 0);
      Value edge = ori(cmpi(arith::CmpIPredicate::eq, z, first),
                       cmpi(arith::CmpIPredicate::eq, z, last));
      Value half =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.5));
      Value e = mulf(mulf(select(edge, half, one), influence),
                     addf(mulf(c.re, c.re), mulf(c.im, c.im)));
      sums[0] = addf(sums[0], e);
      Value eh;
      if (radial) {
        Value magnitude = addf(mulf(c.re, c.re), mulf(c.im, c.im));
        eh = mulf(mulf(select(edge, half, one), mode.virialTerm), magnitude);
      } else {
        Value h = addf(mode.virialTerm, gauss);
        eh = mulf(e, addf(h, h));
      }
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
                                   observed ? 7 : 0, partial,
                                   /*preservesEndPlaneSymmetry=*/!tilted);
  if (observed)
    finishConvolution(rows, arith::ConstantIndexOp::create(b, loc, blocks));
  else
    func::ReturnOp::create(b, loc);
  return fn;
}
