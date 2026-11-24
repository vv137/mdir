// Hierarchical register FFTs and coalesced line tiles [Govindaraju2008].
// Fuse the pointwise operation between the last forward and first inverse
// axis [Tolmachev2023].
#include "mdir/Conversion/AxisFFTEmitter.h"
#include "KernelCode.h"
#include "mdir/Conversion/ReductionEmitter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <cmath>

using namespace mlir;
using namespace mdir::kernels;

namespace {
using C = FFTCoefficient;
using Line = SmallVector<C, 32>;
constexpr double tau = 6.2831853071795864769;

class Axis : public mdir::kernels::detail::KernelCode {
public:
  Axis(OpBuilder &b, Location loc, ArrayRef<int64_t> grid, int pitch,
       Value real, Value spectrum, int axis, bool inverse)
      : KernelCode(b, loc), grid(grid), pitch(pitch), real(real),
        spectrum(spectrum), axis(axis), inverse(inverse),
        n(axis == 2 ? grid[2] / 2 : grid[axis]), f(*factorAxis(n)) {}

  int emit(FFTPointwise pointwise, int count, FFTPartial partial,
           bool preservesEndPlaneSymmetry) {
    const bool z = axis == 2;
    const int h = grid[2] / 2 + 1;
    int lines = 8;
    const int exchange = f.first * (f.second + (f.second % 2 == 0));
    const int sharedPerLine = exchange + (z && !inverse ? n : 0);
    while (lines * sharedPerLine * 8 > (count ? 47 : 48) * 1024)
      lines /= 2;
    const int tiles = (h + lines - 1) / lines;
    int blocks = z ? (grid[0] * grid[1] + lines - 1) / lines
                   : (axis == 0 ? grid[1] : grid[0]) * tiles;
    auto space = gpu::AddressSpaceAttr::get(b.getContext(),
                                            gpu::AddressSpace::Workgroup);
    Type shared = MemRefType::get({2 * lines * sharedPerLine}, b.getF32Type(),
                                  MemRefLayoutAttrInterface(), space);
    SmallVector<Type> sharedTypes{shared};
    int digits = f.second;
    if (count) {
      digits = llvm::alignTo(digits, 32 / lines);
      sharedTypes.push_back(
          MemRefType::get({digits * lines / 32 * count}, b.getF64Type(),
                          MemRefLayoutAttrInterface(), space));
    }
    auto launch = gpu::LaunchOp::create(
        b, loc, idx(blocks), idx(1), idx(1), idx(z ? f.second : lines),
        idx(z ? lines : digits), idx(1), nullptr, nullptr, ValueRange{},
        sharedTypes);
    launch->setAttr("mdir.fft_stage",
                    b.getStringAttr(z           ? (inverse ? "z_c2r" : "z_r2c")
                                    : axis == 0 ? "x_convolve"
                                    : inverse   ? "y_inverse"
                                                : "y_forward"));
    launch->setAttr("mdir.fft_layout",
                    b.getDenseI64ArrayAttr({f.first, f.second, lines, pitch}));
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value scratch = launch.getWorkgroupAttributionBBArgs()[0];
    Value block = launch.getBlockIds().x;
    Value digit = z ? launch.getThreadIds().x : launch.getThreadIds().y;
    Value lane = z ? launch.getThreadIds().y : launch.getThreadIds().x;
    Value line = z ? addi(muli(block, idx(lines)), lane)
                   : addi(muli(remui(block, idx(tiles)), idx(lines)), lane);
    Value plane = z ? line : divui(block, idx(tiles));
    Value valid = less(line, z ? grid[0] * grid[1] : h);
    Value stride = idx(axis == 0 ? grid[1] * pitch : pitch);
    Value base =
        z ? muli(line, idx(pitch))
          : addi(muli(plane, idx(axis == 0 ? pitch : grid[1] * pitch)), line);
    auto address = [&](Value k) {
      return z ? addi(base, k) : addi(base, muli(k, stride));
    };
    auto digitIndex = [&](int step, int i) {
      return addi(digit, idx(step * i));
    };
    auto slot = [&](Value p, Value t) {
      Value row = addi(muli(p, idx(f.second + (f.second % 2 == 0))), t);
      return addi(muli(row, idx(lines)), lane);
    };
    auto packed = [&](Value k) {
      return addi(idx(exchange * lines), addi(muli(lane, idx(n)), k));
    };
    auto rotate = [&](C v, int p, bool backward) {
      return p ? product(v, phase(muli(digit, idx(p)), n, backward)) : v;
    };
    when(count ? andi(valid, less(digit, f.second)) : valid, [&] {
      Line reg;
      for (int a = 0; a < f.first; ++a) {
        Value k = digitIndex(f.second, a);
        if (!z)
          reg.push_back(read(spectrum, address(k)));
        else if (!inverse)
          reg.push_back(read(real, addi(muli(line, idx(n)), k)));
        else {
          C r = read(spectrum, address(k));
          C s = conjugate(read(spectrum, address(subi(idx(n), k))));
          // After inverse X and Y, a triclinic multiplier can leave an
          // imaginary part on the two end planes. A real inverse uses
          // only their real parts for the packed zero frequency.
          if (a == 0 && !preservesEndPlaneSymmetry) {
            Value end = equal(k, idx(0));
            r.im = select(end, constant(0), r.im);
            s.im = select(end, constant(0), s.im);
          }
          C d = product(phase(k, 2 * n, true), minus(r, s));
          reg.push_back(plus(plus(r, s), timesI(d)));
        }
      }
      transform(reg, inverse);
      for (int p = 0; p < f.first; ++p)
        write(scratch, slot(idx(p), digit), rotate(reg[p], p, inverse));
    });
    gpu::BarrierOp::create(b, loc);
    SmallVector<Value> zeros(count, constant(0));
    auto second =
        scf::IfOp::create(b, loc, TypeRange(zeros),
                          andi(valid, less(digit, f.first)), true, count != 0);
    {
      OpBuilder::InsertionGuard stage(b);
      b.setInsertionPointToStart(second.thenBlock());
      SmallVector<Value> sums(zeros);

      Line reg;
      for (int j = 0; j < f.second; ++j)
        reg.push_back(read(scratch, slot(digit, idx(j))));
      transform(reg, inverse);
      for (int q = 0; q < f.second; ++q) {
        Value k = digitIndex(f.first, q);
        if (axis == 0)
          reg[q] = pointwise(b, k, plane, line, reg[q], sums);
        else if (!z)
          write(spectrum, address(k), reg[q]);
        else if (inverse)
          write(real, addi(muli(line, idx(n)), k), reg[q]);
        else
          write(scratch, packed(k), reg[q]);
      }
      if (axis == 0) {
        transform(reg, true);
        for (int j = 0; j < f.second; ++j)
          write(scratch, slot(digit, idx(j)), reg[j]);
      }
      scf::YieldOp::create(b, loc, sums);
      if (count) {
        b.setInsertionPointToStart(second.elseBlock());
        scf::YieldOp::create(b, loc, zeros);
      }
    }
    if (count) {
      Value tid = addi(muli(digit, idx(lines)), lane);
      auto totals = emitBlockReduce(b, loc, second.getResults(), tid,
                                    launch.getWorkgroupAttributionBBArgs()[1],
                                    Combiner::Add, digits * lines);
      when(equal(tid, idx(0)), [&] { partial(block, totals); });
    }
    if (axis == 0 || (z && !inverse)) {
      gpu::BarrierOp::create(b, loc);
      when(count ? andi(valid, less(digit, f.second)) : valid, [&] {
        if (axis == 0) {
          Line reg;
          for (int p = 0; p < f.first; ++p)
            reg.push_back(rotate(read(scratch, slot(idx(p), digit)), p, true));
          transform(reg, true);
          for (int a = 0; a < f.first; ++a)
            write(spectrum, address(digitIndex(f.second, a)), reg[a]);
        } else {
          for (int a = 0; a < f.first; ++a) {
            Value k = digitIndex(f.second, a);
            C r = read(scratch, packed(k));
            Value mirror = select(equal(k, idx(0)), idx(0), subi(idx(n), k));
            C s = conjugate(read(scratch, packed(mirror)));
            C d = product(phase(k, 2 * n, false), minus(r, s));
            C folded = minus(plus(r, s), timesI(d));
            write(spectrum, address(k), scale(folded, constant(0.5)));
            if (a == 0)
              when(equal(digit, idx(0)), [&] {
                write(spectrum, address(idx(n)),
                      {subf(r.re, r.im), constant(0)});
              });
          }
        }
      });
    }
    gpu::TerminatorOp::create(b, loc);
    return blocks;
  }

private:
  ArrayRef<int64_t> grid;
  int pitch;
  Value real, spectrum;
  int axis;
  bool inverse;
  int n;
  AxisFactors f;
  Value idx(int v) { return arith::ConstantIndexOp::create(b, loc, v); }
  Value constant(double v) {
    return arith::ConstantOp::create(b, loc, b.getF32FloatAttr(v));
  }
  Value less(Value x, int y) {
    return cmpi(arith::CmpIPredicate::ult, x, idx(y));
  }
  Value equal(Value x, Value y) { return cmpi(arith::CmpIPredicate::eq, x, y); }
  void when(Value pred, llvm::function_ref<void()> body) {
    auto op = scf::IfOp::create(b, loc, TypeRange{}, pred, true, false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(op.thenBlock());
    body();
    scf::YieldOp::create(b, loc);
  }
  C read(Value mem, Value i) {
    Value at = muli(i, idx(2));
    return {load(mem, {at}), load(mem, {addi(at, idx(1))})};
  }
  void write(Value mem, Value i, C v) {
    Value at = muli(i, idx(2));
    store(v.re, mem, {at});
    store(v.im, mem, {addi(at, idx(1))});
  }
  C plus(C a, C c) { return {addf(a.re, c.re), addf(a.im, c.im)}; }
  C minus(C a, C c) { return {subf(a.re, c.re), subf(a.im, c.im)}; }
  C scale(C a, Value s) { return {mulf(a.re, s), mulf(a.im, s)}; }
  Value neg(Value v) { return arith::NegFOp::create(b, loc, v); }
  C conjugate(C a) { return {a.re, neg(a.im)}; }
  C timesI(C a) { return {neg(a.im), a.re}; }
  C product(C a, C c) {
    return {subf(mulf(a.re, c.re), mulf(a.im, c.im)),
            addf(mulf(a.re, c.im), mulf(a.im, c.re))};
  }
  C phase(Value k, int length, bool backward) {
    Value angle = mulf(sitofp(b.getF32Type(), indexCast(b.getI32Type(), k)),
                       constant((backward ? tau : -tau) / length));
    return {NVVM::CosOp::create(b, loc, angle),
            NVVM::SinOp::create(b, loc, angle)};
  }
  C root(C v, int k, int length, bool backward) {
    k %= length;
    if (!k)
      return v;
    if (2 * k == length)
      return {neg(v.re), neg(v.im)};
    if (4 * k == length || 4 * k == 3 * length) {
      C w = timesI(v);
      return (4 * k == length) == backward ? w : C{neg(w.re), neg(w.im)};
    }
    double angle = (backward ? tau : -tau) * k / length;
    return product(v, {constant(std::cos(angle)), constant(std::sin(angle))});
  }

  // Stockham autosort. A stage inserts q above the s resolved low digits.
  void transform(Line &v, bool backward) {
    int length = v.size();
    for (int s = 1; s < length;) {
      int r = *llvm::find_if(axisRadices, [&](int radix) {
        return (length / s) % radix == 0;
      });
      Line out(length);
      for (int j = 0; j < length / r; ++j) {
        Line terms;
        for (int u = 0; u < r; ++u)
          terms.push_back(
              root(v[j + u * (length / r)], u * (j % s), r * s, backward));
        for (int q = 0; q < r; ++q) {
          C sum = terms[0];
          for (int u = 1; u < r; ++u)
            sum = plus(sum, root(terms[u], u * q, r, backward));
          out[(j / s) * r * s + j % s + s * q] = sum;
        }
      }
      v.swap(out);
      s *= r;
    }
  }
};
} // namespace

int mdir::kernels::emitAxisConvolution(OpBuilder &b, Location loc,
                                       ArrayRef<int64_t> grid, int64_t pitch,
                                       Value real, Value spectrum,
                                       FFTPointwise pointwise, int count,
                                       FFTPartial partial,
                                       bool preservesEndPlaneSymmetry) {
  int xBlocks = 0;
  for (int pass = 0; pass < 5; ++pass) {
    int blocks =
        Axis(b, loc, grid, pitch, real, spectrum, std::abs(2 - pass), pass > 2)
            .emit(pointwise, pass == 2 ? count : 0, partial, preservesEndPlaneSymmetry);
    if (pass == 2)
      xBlocks = blocks;
  }
  return xBlocks;
}
