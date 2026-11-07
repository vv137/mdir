#include "FrameBatchTailIR.h"
#include "FrameBatchOutputIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include <cmath>
#include <limits>
using namespace mlir;
using namespace mdir::driver::builder;
namespace {
struct Arithmetic {
  OpBuilder &b;
  Location loc;
  explicit Arithmetic(OpBuilder &b) : b(b), loc(b.getUnknownLoc()) {}
  Value real(double x) { return arith::ConstantOp::create(b, loc, b.getF64FloatAttr(x)); }
  Value index(int64_t x) { return arith::ConstantIndexOp::create(b, loc, x); }
  Value bit(bool x) { return arith::ConstantIntOp::create(b, loc, x, 1); }
  Value add(Value x, Value y) { return arith::AddFOp::create(b, loc, x, y); }
  Value sub(Value x, Value y) { return arith::SubFOp::create(b, loc, x, y); }
  Value mul(Value x, Value y) { return arith::MulFOp::create(b, loc, x, y); }
  Value div(Value x, Value y) { return arith::DivFOp::create(b, loc, x, y); }
  Value abs(Value x) { return math::AbsFOp::create(b, loc, x); }
  Value both(Value x, Value y) { return arith::AndIOp::create(b, loc, x, y); }
  Value either(Value x, Value y) { return arith::OrIOp::create(b, loc, x, y); }
  Value cmp(arith::CmpFPredicate p, Value x, Value y) { return arith::CmpFOp::create(b, loc, p, x, y); }
  Value select(Value p, Value x, Value y) { return arith::SelectOp::create(b, loc, p, x, y); }
  Value load(Value memory, int64_t slot) { return memref::LoadOp::create(b, loc, memory, index(slot)); }
  void store(Value value, Value memory, int64_t slot) { memref::StoreOp::create(b, loc, value, memory, index(slot)); }
};
using Names = llvm::StringMap<Value>;
// Same 7/15-point adaptive rule and cancellation scale as the host
// correction [Piessens1983]. Intervals retain insertion order; the first
// interval of greatest error is split, preserving the host's sums.
struct Integrator {
  const FrameBatchTailPlan::Term &term;
  const FrameBatchTailPlan &plan;
  const Names &tables;
  Value workspace;
  std::array<Value, 2> sample(OpBuilder &b, Names values, Value x) {
    Arithmetic a(b);
    values["r"] = a.div(a.real(plan.cutoff), x);
    Value denominator = a.mul(a.mul(a.mul(x, x), x), x);
    double rc3 = plan.cutoff * plan.cutoff * plan.cutoff;
    return {a.div(a.mul(a.real(rc3), term.expression.build(b, values, tables)), denominator),
            a.div(a.mul(a.real(rc3), term.expression.buildMagnitude(b, values, tables)), denominator)};
  }
  void rule(OpBuilder &b, const Names &values, Value lo, Value hi, Value row) {
    static constexpr double xk[] = {
      .991455371120812639206854697526329, .949107912342758524526189684047851,
      .864864423359769072789712788640926, .741531185599394439863864773280788,
      .586087235467691130294144845693013, .405845151377397166906606412076961,
      .207784955007898467600689403773245};
    static constexpr double wk[] = {
      .022935322010529224963732008058970, .063092092629978553290700663189204,
      .104790010322250183839876322541518, .140653259715525918745189590510238,
      .169004726639267902826583426598550, .190350578064785409913256402421014,
      .204432940075298892414161999234649, .209482141084727828012999174891714};
    static constexpr double wg[] = {
      .129484966168869693270611432679082, .279705391489276667901467771423780,
      .381830050505118944950369775488975, .417959183673469387755102040816327};
    Arithmetic a(b);
    Value center = a.mul(a.real(.5), a.add(lo, hi));
    Value half = a.mul(a.real(.5), a.sub(hi, lo));
    auto fc = sample(b, values, center);
    Value kronrod = a.mul(a.real(wk[7]), fc[0]);
    Value gauss = a.mul(a.real(wg[3]), fc[0]);
    Value sized = a.mul(a.real(wk[7]), fc[1]);
    for (unsigned k = 0; k != 7; ++k) {
      Value offset = a.mul(half, a.real(xk[k]));
      auto left = sample(b, values, a.sub(center, offset));
      auto right = sample(b, values, a.add(center, offset));
      Value sum = a.add(left[0], right[0]);
      kronrod = a.add(kronrod, a.mul(a.real(wk[k]), sum));
      sized = a.add(sized, a.mul(a.real(wk[k]), a.add(left[1], right[1])));
      if (k % 2) gauss = a.add(gauss, a.mul(a.real(wg[k / 2]), sum));
    }
    Value columns[] = {lo, hi, a.mul(kronrod, half),
        a.abs(a.mul(a.sub(kronrod, gauss), half)), a.mul(sized, half)};
    for (unsigned c = 0; c != 5; ++c)
      memref::StoreOp::create(b, a.loc, columns[c], workspace, ValueRange{row, a.index(c)});
  }
  Value integrate(OpBuilder &b, Names values) {
    Arithmetic a(b);
    Value nil = a.real(0), nan = a.real(std::numeric_limits<double>::quiet_NaN());
    Value finiteTail = a.bit(true), previous = nil;
    for (unsigned k = 3; k <= 6; ++k) {
      // These powers are fixed host metadata, not device pow operations.
      double radius = plan.cutoff * std::pow(10.0, k);
      values["r"] = a.real(radius);
      Value cube = a.real(radius * radius * radius);
      Value h = a.abs(a.mul(cube, term.expression.build(b, values, tables)));
      Value size = a.abs(a.mul(cube, term.expression.buildMagnitude(b, values, tables)));
      h = a.select(a.cmp(arith::CmpFPredicate::OLE, h, a.mul(a.real(1e-12), size)), nil, h);
      finiteTail = a.both(finiteTail, buildFrameBatchIsFinite(b, h));
      if (k > 3) finiteTail = a.both(finiteTail,
          a.either(a.cmp(arith::CmpFPredicate::OEQ, h, nil),
                   a.cmp(arith::CmpFPredicate::OLE, h, a.mul(a.real(.1), previous))));
      previous = h;
    }
    rule(b, values, nil, a.real(1), a.index(0));
    auto iterations = scf::ForOp::create(b, a.loc, a.index(0), a.index(4000), a.index(1),
        ValueRange{a.index(1), a.bit(false), nan},
        [&](OpBuilder &at, Location, Value, ValueRange carried) {
      Arithmetic x(at);
      Value active = arith::XOrIOp::create(at, x.loc, carried[1], x.bit(true));
      auto next = scf::IfOp::create(at, x.loc,
          TypeRange{at.getIndexType(), at.getI1Type(), at.getF64Type()}, active, true);
      {
        OpBuilder work = OpBuilder::atBlockEnd(next.thenBlock());
        Arithmetic w(work);
        auto sum = scf::ForOp::create(work, w.loc, w.index(0), carried[0], w.index(1),
            ValueRange{nil, nil, nil, w.index(0), w.real(-1)},
            [&](OpBuilder &row, Location, Value i, ValueRange old) {
          Arithmetic r(row);
          auto load = [&](unsigned c) -> Value {
            return memref::LoadOp::create(row, r.loc, workspace, ValueRange{i, r.index(c)});
          };
          Value error = load(3);
          Value greater = r.cmp(arith::CmpFPredicate::OGT, error, old[4]);
          scf::YieldOp::create(row, r.loc, ValueRange{r.add(old[0], load(2)),
              r.add(old[1], error), r.add(old[2], load(4)),
              r.select(greater, i, old[3]), r.select(greater, error, old[4])});
        });
        Value value = sum.getResult(0), error = sum.getResult(1), sized = sum.getResult(2);
        Value maximum = arith::MaxNumFOp::create(work, w.loc, w.abs(value), sized);
        Value converged = w.either(w.cmp(arith::CmpFPredicate::OLE, error, w.mul(w.real(1e-13), maximum)),
                                  w.cmp(arith::CmpFPredicate::OEQ, error, nil));
        Value finite = w.both(buildFrameBatchIsFinite(work, value), buildFrameBatchIsFinite(work, error));
        Value worst = sum.getResult(3);
        Value lo = memref::LoadOp::create(work, w.loc, workspace, ValueRange{worst, w.index(0)});
        Value hi = memref::LoadOp::create(work, w.loc, workspace, ValueRange{worst, w.index(1)});
        Value middle = w.mul(w.real(.5), w.add(lo, hi));
        Value split = w.both(w.cmp(arith::CmpFPredicate::OGT, middle, lo),
                             w.cmp(arith::CmpFPredicate::OLT, middle, hi));
        Value valid = w.both(finite, split);
        Value done = w.either(converged, arith::XOrIOp::create(work, w.loc, valid, w.bit(true)));
        auto step = scf::IfOp::create(work, w.loc,
            TypeRange{work.getIndexType(), work.getI1Type(), work.getF64Type()}, done, true);
        {
          OpBuilder end = OpBuilder::atBlockEnd(step.thenBlock());
          Arithmetic e(end);
          scf::YieldOp::create(end, e.loc, ValueRange{carried[0], e.bit(true),
              e.select(e.both(finite, converged), value, nan)});
        }
        {
          OpBuilder refine = OpBuilder::atBlockEnd(step.elseBlock());
          Arithmetic r(refine);
          rule(refine, values, lo, middle, worst);
          rule(refine, values, middle, hi, carried[0]);
          Value count = arith::AddIOp::create(refine, r.loc, carried[0], r.index(1));
          scf::YieldOp::create(refine, r.loc, ValueRange{count, r.bit(false), nan});
        }
        scf::YieldOp::create(work, w.loc, step.getResults());
      }
      {
        OpBuilder skip = OpBuilder::atBlockEnd(next.elseBlock());
        scf::YieldOp::create(skip, skip.getUnknownLoc(), carried);
      }
      scf::YieldOp::create(at, x.loc, next.getResults());
    });
    return a.select(finiteTail, iterations.getResult(2), nan);
  }
};
}

llvm::Error mdir::driver::builder::buildFrameBatchTails(OpBuilder &builder,
    const FrameBatchTailPlan &plan, Value theta, Value workspace, Value corrections,
    Value rawVolume, Value rawFixed, Value observableVolume, Value observableFixed,
    const Names &tables) {
  if (plan.terms.empty()) return llvm::Error::success();
  Arithmetic root(builder);
  Value one = root.index(1);
  auto launch = gpu::LaunchOp::create(builder, root.loc, one, one, one, one, one, one);
  launch->setAttr("mdir.frame_host_arithmetic", builder.getUnitAttr());
  OpBuilder kernel = OpBuilder::atBlockEnd(&launch.getBody().front());
  Arithmetic a(kernel);
  if (plan.resetCorrections)
    for (unsigned i = 0; i != 5; ++i) a.store(a.real(plan.initialCorrections[i]), corrections, i);
  for (const auto &term : plan.terms) {
    Names current;
    for (const auto &parameter : term.parameters)
      current[parameter.name] = a.load(theta, parameter.theta);
    auto tail = [&](OpBuilder &b, const Names &changes) {
      Arithmetic x(b);
      Value energy = x.real(0), virial = x.real(0), shift = x.real(0);
      Integrator quadrature{term, plan, tables, workspace};
      for (const auto &pair : term.pairs) {
        Names values;
        for (const auto &entry : pair.values) values[entry.getKey()] = x.real(entry.getValue());
        for (const auto &entry : changes)
          if (values.count(entry.getKey())) values[entry.getKey()] = entry.getValue();
        Value integral = quadrature.integrate(b, values);
        values["r"] = x.real(plan.cutoff);
        Value edge = x.mul(x.real(plan.cutoff * plan.cutoff * plan.cutoff),
            term.expression.build(b, values, tables));
        Value count = x.real(pair.count);
        energy = x.add(energy, x.mul(count, integral));
        virial = x.add(virial, x.mul(count, x.add(x.mul(x.real(3), integral), edge)));
        shift = x.add(shift, x.div(x.mul(count, edge), x.real(3)));
      }
      Value factor = x.real(plan.factor);
      return std::array<Value, 4>{x.mul(factor, energy), x.mul(factor, virial),
          x.mul(factor, shift), x.mul(x.mul(factor, shift), x.real(plan.fixedFactor))};
    };
    auto actual = tail(kernel, current);
    Value delta[4];
    for (unsigned i = 0; i != 4; ++i) delta[i] = a.sub(actual[i], a.real(term.initial[i]));
    Value additions[] = {plan.shifted ? a.add(delta[0], delta[2]) : delta[0],
        plan.shifted ? delta[3] : a.real(0), delta[2], delta[3], delta[1]};
    for (unsigned i = 0; i != 5; ++i) a.store(a.add(a.load(corrections, i), additions[i]), corrections, i);
    auto derivative = [&](llvm::StringRef name) -> std::array<Value, 2> {
      Value value = current.lookup(name);
      if (!value) value = a.real(term.pairs.front().values.lookup(name));
      Value magnitude = arith::MaxNumFOp::create(kernel, a.loc, a.abs(value), a.real(1));
      Value h = a.mul(a.real(1e-3), magnitude);
      auto at = [&](OpBuilder &b, Value argument) {
        Names changed(current);
        changed[name] = argument;
        auto result = tail(b, changed);
        Arithmetic x(b);
        return std::array<Value, 2>{x.add(result[0], result[2]), result[3]};
      };
      auto central = [&](Value step) {
        auto plus = at(kernel, a.add(value, step));
        auto minus = at(kernel, a.sub(value, step));
        Value twice = a.mul(a.real(2), step);
        return std::array<Value, 2>{a.div(a.sub(plus[0], minus[0]), twice),
                                    a.div(a.sub(plus[1], minus[1]), twice)};
      };
      auto fine = central(a.mul(a.real(.5), h));
      auto coarse = central(h);
      std::array<Value, 2> d;
      for (unsigned i = 0; i != 2; ++i)
        d[i] = a.div(a.sub(a.mul(a.real(4), fine[i]), coarse[i]), a.real(3));
      Value valid = a.both(buildFrameBatchIsFinite(kernel, d[0]), buildFrameBatchIsFinite(kernel, d[1]));
      auto result = scf::IfOp::create(kernel, a.loc,
          TypeRange{kernel.getF64Type(), kernel.getF64Type()}, valid, true);
      {
        OpBuilder yes = OpBuilder::atBlockEnd(result.thenBlock());
        scf::YieldOp::create(yes, yes.getUnknownLoc(), ValueRange{d[0], d[1]});
      }
      {
        OpBuilder no = OpBuilder::atBlockEnd(result.elseBlock());
        Arithmetic x(no);
        auto base = at(no, value);
        for (double sign : {1., -1.}) {
          Value step = x.mul(x.real(sign * 1e-4), magnitude);
          auto first = at(no, x.add(value, step));
          auto second = at(no, x.add(value, x.mul(x.real(2), step)));
          for (unsigned i = 0; i != 2; ++i) {
            Value numerator = x.sub(x.add(x.mul(x.real(-3), base[i]),
                x.mul(x.real(4), first[i])), second[i]);
            Value fallback = x.div(numerator, x.mul(x.real(2), step));
            d[i] = x.select(buildFrameBatchIsFinite(no, d[i]), d[i], fallback);
          }
        }
        scf::YieldOp::create(no, x.loc, ValueRange{d[0], d[1]});
      }
      return {result.getResult(0), result.getResult(1)};
    };
    for (const auto &parameter : term.parameters)
      if (parameter.raw >= 0) {
        auto slope = derivative(parameter.name);
        a.store(slope[0], rawVolume, parameter.raw);
        a.store(slope[1], rawFixed, parameter.raw);
      }
    for (const auto &observed : term.observed) {
      auto values = observed.derivative.empty()
          ? std::array<Value, 2>{a.add(actual[0], actual[2]), actual[3]}
          : derivative(observed.derivative);
      a.store(values[0], observableVolume, observed.column);
      a.store(values[1], observableFixed, observed.column);
    }
  }
  gpu::TerminatorOp::create(kernel, a.loc);
  return llvm::Error::success();
}
