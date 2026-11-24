// Gaussian spectral product on the last forward axis [Essmann1995].
#include "DevicePME.h"
#include "mdir/Conversion/AxisFFTEmitter.h"

using namespace mlir;
using namespace mdir::kernels;

func::FuncOp DevicePME::emitAxisFFT(StringRef name, ArrayRef<int64_t> grid,
                                    int64_t pitch) {
  auto buffer = globalMem({ShapedType::kDynamic}, realType);
  auto fn = func::FuncOp::create(
      b, loc, name, b.getFunctionType({buffer, buffer, buffer}, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value tables = entry->getArgument(2);
  auto pointwise = [&](OpBuilder &, Value x, Value y, Value z,
                       FFTCoefficient c) {
    Value square, factor;
    int64_t offset = 0;
    Value indices[] = {x, y, z};
    for (auto [k, length] : llvm::zip(indices, grid)) {
      auto component = [&](int field) {
        Value start =
            arith::ConstantIndexOp::create(b, loc, offset + field * length);
        return load(tables, {addi(start, k)});
      };
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
    return FFTCoefficient{mulf(c.re, influence), mulf(c.im, influence)};
  };
  emitAxisConvolution(b, loc, grid, pitch, entry->getArgument(0),
                      entry->getArgument(1), pointwise);
  func::ReturnOp::create(b, loc);
  return fn;
}
