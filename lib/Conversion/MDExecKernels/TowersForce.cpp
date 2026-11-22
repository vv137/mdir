// Rectangular particle traversal [Shaw2005]. The quantized screen, prefix
// records and rank partition are the stage 4n GPU implementation of D[towers].
#include "TowersForce.h"
#include "NeighborBuildCode.h"
#include "mdir/Conversion/CompactionEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include <cmath>
#include <limits>

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;
using namespace mdir::kernels::detail;

namespace {
class Force : public NeighborBuildCode {
public:
  Force(OpBuilder &b, TowersLayout g, md_exec::PairForOp term,
        TowersForceShape shape, bool probe)
      : NeighborBuildCode(b, term.getLoc()), g(g), term(term),
        panel(shape.panel), stripe(shape.stripe), workers(shape.workers),
        recordCap(shape.records), residency(shape.residency),
        groups(panel / 32), probe(probe), v3(VectorType::get({3}, f32)),
        v4(VectorType::get({4}, f32)) {}
  LogicalResult emit();

private:
  TowersLayout g;
  md_exec::PairForOp term;
  int panel, stripe, workers, recordCap, residency, groups;
  bool probe;
  Value trace, owner;
  void mark(int phase);
  Value recordPtr(Value rec, int col) {
    Value at = addi(shli(rec, ci(1)), ci(col));
    return LLVM::GEPOp::create(b, loc, records.getType(), i32, records,
                               ValueRange{at});
  }
  Value recordWord(Value rec, int col) {
    return LLVM::LoadOp::create(b, loc, i32, recordPtr(rec, col));
  }
  VectorType v3, v4;
  SmallVector<Type> types;
  llvm::SetVector<Value> captures;
  IRMapping mapping;
  Value descriptors, packed, atomIds, geometry, words, invalid, wordScale;
  Values parameters, tileParameters;
  int positionParameter = -1;
  int64_t fixedBytes = 0;
  void stageParameters(gpu::GPUFuncOp fn);
  Value pos, id, xy, zn, bias, records, totals, keptSlots, accum, coarse;
  Value tid, lane, scale, limit, cutoff2, filterLimit;
  void bind(ValueRange args);
  void withValidGeometry(function_ref<void()> body);
  void host();
  void device();
  Value point(Value slot, Value field);
  Value displacement(Value target, Value source);
  Value squared(Value d);
  Values pair(Value t, Value s, Value ti, Value si,
              ValueRange targetParameters = {});
  Value sharedAdd(Value dst, Value byteOffset, Value word);
  void addWord(Value dst, Value i, Value axis, Value w);
  Value shared(gpu::GPUFuncOp fn, ArrayRef<int64_t> shape, Type t);
  void stage(ArrayRef<Value> starts, ArrayRef<Value> counts, Value begin,
             Value size, bool source);
  Value mask(Value target, Value group, Value sourceCount);
  Value indexed(Value v, Value from) {
    return shuffle(gpu::ShuffleMode::IDX, v, from, ci(32)).first;
  }
  Value visitTile(Value sc, Value tc, Value sb, Value tb, Value home,
                  Value lower, Value lastTile);
  void traversal(Value owner);
};

Value Force::point(Value slot, Value field) {
  Value p = load(packed, {slot});
  int lateral = 2 * g.reach + 1;
  auto is = [&](int f) { return cmpi(CmpIPredicate::eq, field, cx(f)); };
  auto sign = [&](Value neg, Value positive) {
    return select(neg, cf(-1), select(positive, cf(1), cf(0)));
  };
  Values shift{
      sign(ori(is(lateral), is(lateral + 1)), is(lateral + 3)),
      select(cmpi(CmpIPredicate::ugt, field, cx(lateral)), cf(-1), cf(0))};
  Value dz =
      select(cmpi(CmpIPredicate::ule, field, cx(g.reach)), subi(cx(0), field),
             select(cmpi(CmpIPredicate::ult, field, cx(lateral)),
                    subi(field, cx(g.reach)), cx(0)));
  shift.push_back(sitofp(f32, indexCast(i32, dz)));
  Values coords;
  for (int a = 0; a < 3; ++a)
    coords.push_back(addf(
        extract(p, a), mulf(shift[a], truncf(load(geometry, {cx(a), cx(1)})))));
  coords.push_back(extract(p, 3));
  return vector::FromElementsOp::create(b, loc, v4, coords);
}

Value Force::displacement(Value t, Value s) {
  Values d;
  for (int a = 0; a < 3; ++a)
    d.push_back(subf(extract(t, a), extract(s, a)));
  return vector::FromElementsOp::create(b, loc, v3, d);
}

Value Force::squared(Value d) {
  auto x = extract(d, 0), y = extract(d, 1), z = extract(d, 2);
  Value yz = math::FmaOp::create(b, loc, y, y, mulf(z, z));
  return math::FmaOp::create(b, loc, x, x, yz);
}

Values Force::pair(Value t, Value s, Value ti, Value si,
                   ValueRange targetParameters) {
  Value d = displacement(t, s), r2 = squared(d);
  auto wt = g.host ? i64 : i32;
  return ifElse(
      {wt, wt, wt}, cmpf(CmpFPredicate::OLT, r2, cutoff2),
      [&] {
        Block &body = term.getKernel().front();
        IRMapping inside = mapping;
        inside.map(body.getArgument(0), r2);
        inside.map(body.getArgument(1), d);
        for (auto [i, param] :
             llvm::enumerate(g.host ? parameters : tileParameters)) {
          inside.map(body.getArgument(2 + 2 * i),
                     targetParameters.empty()
                         ? (param ? load(param, {ti}) : extract(t, 3))
                         : targetParameters[i]);
          inside.map(body.getArgument(3 + 2 * i),
                     param ? load(param, {si}) : extract(s, 3));
        }
        for (auto &op : body.without_terminator())
          b.clone(op, inside);
        Value f = inside.lookup(body.getTerminator()->getOperand(0));
        Values scaled;
        Value narrow = cb(true);
        for (int a = 0; a < 3; ++a) {
          Value x = mulf(extract(f, a), wordScale);
          scaled.push_back(x);
          narrow = andi(narrow, cmpf(CmpFPredicate::OLT, absf(x), cf(0x1p31)));
        }
        auto wide = [&] {
          Values result;
          Value valid = cb(true);
          for (Value x : scaled) {
            Value fits = andi(cmpf(CmpFPredicate::OGT, x, cf(-0x1p63)),
                              cmpf(CmpFPredicate::OLT, x, cf(0x1p63)));
            valid = andi(valid, fits);
            Value safe = select(fits, x, cf(0));
            if (g.host)
              result.push_back(fptosi(i64, roundeven(safe)));
            else
              result.push_back(LLVM::InlineAsmOp::create(
                                   b, loc, TypeRange{i64}, ValueRange{safe},
                                   "cvt.rni.s64.f32 $0, $1;", "=l,f", false,
                                   false, LLVM::TailCallKind::None,
                                   LLVM::AsmDialectAttr(), ArrayAttr())
                                   .getResult(0));
          }
          ifThen(notNarrow(valid), [&] {
            if (g.host)
              store(ci(1), invalid, {cx(0)});
            else
              atomic(LLVM::AtomicBinOp::_or,
                     pointerTo(baseAddress(invalid), cl(0)), ci(1));
          });
          return result;
        };
        if (g.host)
          return wide();
        Values result = ifElse(
            {i32, i32, i32}, narrow,
            [&] {
              Values words;
              for (Value x : scaled) {
                Value q = LLVM::InlineAsmOp::create(
                              b, loc, TypeRange{i32}, ValueRange{x},
                              "cvt.rni.s32.f32 $0, $1;", "=r,f", false, false,
                              LLVM::TailCallKind::None, LLVM::AsmDialectAttr(),
                              ArrayAttr())
                              .getResult(0);
                words.push_back(q);
              }
              return words;
            },
            [&] {
              // A blind bonded interaction can exceed i32 before exclusion
              // replay. Keep its actual word for this stage's direct-only
              // oracle.
              Values f = wide();
              Value target = indexCast(index, load(id, {ti})),
                    source = indexCast(index, load(id, {si}));
              for (int a = 0; a < 3; ++a) {
                addWord(words, target, cx(a), f[a]);
                addWord(words, source, cx(a), subi(cl(0), f[a]));
              }
              return Values(3, ci(0));
            });
        return result;
      },
      [&] { return Values(3, g.host ? cl(0) : ci(0)); });
}

// Only lanes of this CTA access the tile. The barriers protect its reuse.
Value Force::sharedAdd(Value dst, Value byteOffset, Value word) {
  Value ptr =
      LLVM::IntToPtrOp::create(b, loc, LLVM::LLVMPointerType::get(context, 3),
                               addi(baseAddress(dst), byteOffset));
  return LLVM::AtomicRMWOp::create(b, loc, LLVM::AtomicBinOp::add, ptr, word,
                                   LLVM::AtomicOrdering::monotonic, "block");
}

void Force::addWord(Value dst, Value i, Value axis, Value w) {
  if (g.host) {
    store(addi(load(dst, {i, axis}), w), dst, {i, axis});
  } else if (dst == words) {
    Value offset = muli(indexCast(i64, addi(muli(i, cx(3)), axis)), cl(8));
    atomic(LLVM::AtomicBinOp::add, pointerTo(baseAddress(dst), offset), w);
  } else {
    sharedAdd(
        dst,
        muli(indexCast(i64, addi(muli(axis, cx(panel + stripe)), i)), cl(4)),
        w);
  }
}

void Force::host() {
  forLoop(cx(0), cx(g.cells()), cx(1), {}, [&](Value owner, ValueRange) {
    emitOwnedTowersSpans(
        b, loc, g.reach, owner, descriptors,
        [&](TowersSpan s, TowersSpan t, Value triangular) {
          forLoop(cx(0), s.count, cx(1), {}, [&](Value i, ValueRange) {
            Value ss = addi(s.first, i), source = point(ss, s.field),
                  si = indexCast(index, load(atomIds, {ss}));
            forLoop(cx(0), select(triangular, i, t.count), cx(1), {},
                    [&](Value j, ValueRange) {
                      Value ts = addi(t.first, j),
                            ti = indexCast(index, load(atomIds, {ts}));
                      auto force = pair(point(ts, t.field), source, ti, si);
                      for (auto [axis, w] : llvm::enumerate(force)) {
                        addWord(words, ti, cx(axis), w);
                        addWord(words, si, cx(axis), subi(cl(0), w));
                      }
                      return Values{};
                    });
            return Values{};
          });
        });
    return Values{};
  });
}

Value Force::shared(gpu::GPUFuncOp fn, ArrayRef<int64_t> shape, Type t) {
  auto space =
      gpu::AddressSpaceAttr::get(context, gpu::AddressSpace::Workgroup);
  return fn.addWorkgroupAttribution(
      MemRefType::get(shape, t, MemRefLayoutAttrInterface(), space), loc);
}

void Force::stage(ArrayRef<Value> starts, ArrayRef<Value> counts, Value begin,
                  Value size, bool source) {
  // Each lane imports one slot. Resolve its descriptor in registers instead
  // of reusing the front of every warp for each (usually short) span.
  forLoop(tid, size, cx(workers), {}, [&](Value laneSlot, ValueRange) {
    Value logical = addi(begin, laneSlot), first = cx(0), field = cx(0);
    Value addr = addi(starts[0], logical);
    int columns = 2 * g.reach + 1;
    for (int k = 0; k < (source ? columns : 5); ++k) {
      int j = source || k == 0 ? k : columns + k - 1;
      if (k) {
        Value inside = cmpi(CmpIPredicate::uge, logical, first);
        addr = select(inside, addi(starts[j], subi(logical, first)), addr);
        field = select(inside, cx(j), field);
      }
      first = addi(first, counts[j]);
    }
    Value slot = addi(cx(source ? 0 : panel), laneSlot);
    Value p = point(addr, field);
    Value atom = load(atomIds, {addr});
    store(atom, id, {slot});
    for (auto [src, dst] : llvm::zip(parameters, tileParameters)) {
      Value value = load(src, {indexCast(index, atom)});
      if (dst)
        store(value, dst, {slot});
      else
        p = vector::InsertOp::create(b, loc, value, p, 3);
    }
    store(p, pos, {slot});
    Values q;
    for (int a = 0; a < 3; ++a)
      q.push_back(roundeven(mulf(extract(p, a), scale)));
    Value norm =
        addf(addf(mulf(q[0], q[0]), mulf(q[1], q[1])), mulf(q[2], q[2]));
    Value fourth = cf(64);
    if (source) {
      // Round the source norm downward. This can only add screen positives.
      Value bits =
          LLVM::InlineAsmOp::create(b, loc, TypeRange{b.getI16Type()},
                                    ValueRange{mulf(norm, cf(1.0 / 64))},
                                    "cvt.rm.f16.f32 $0, $1;", "=h,f", false,
                                    false, LLVM::TailCallKind::None,
                                    LLVM::AsmDialectAttr(), ArrayAttr())
              .getResult(0);
      Value half = arith::BitcastOp::create(b, loc, b.getF16Type(), bits);
      fourth =
          arith::ExtFOp::create(b, loc, f32, half, arith::FastMathFlagsAttr());
      for (Value &v : q)
        v = mulf(cf(-2), v);
    } else
      store(norm, bias, {subi(slot, cx(panel))});
    auto halfPair = [&](Value a, Value c) {
      auto ht = b.getF16Type();
      Value x = arith::TruncFOp::create(b, loc, ht, a),
            y = arith::TruncFOp::create(b, loc, ht, c);
      Value v = vector::FromElementsOp::create(b, loc, VectorType::get({2}, ht),
                                               ValueRange{x, y});
      return LLVM::BitcastOp::create(b, loc, i32, v).getResult();
    };
    store(halfPair(q[0], q[1]), xy, {slot});
    store(halfPair(q[2], fourth), zn, {slot});
    return Values{};
  });
}

// Each warp supplies 32 targets. Lane r returns the 32 source signs for
// its target. Each m16n8k8 fragment covers 16 target rows and 8 sources.
Value Force::mask(Value target, Value group, Value sc) {
  Value row = shrui(lane, ci(2)), k = andi(lane, ci(3)), bits = ci(0);
  Value tx = load(xy, {addi(cx(panel), target)}),
        tz = load(zn, {addi(cx(panel), target)});
  Value c = subf(load(bias, {target}), limit);
  auto operand = [&](Value x, Value z) {
    return select(cmpi(CmpIPredicate::eq, k, ci(0)), x,
                  select(cmpi(CmpIPredicate::eq, k, ci(1)), z, ci(0)));
  };
  auto resultTy =
      LLVM::LLVMStructType::getLiteral(context, {f32, f32, f32, f32});
  for (int h = 0; h < 2; ++h) {
    Values ops, initial;
    for (int j = 0; j < 2; ++j) {
      Value col = addi(shli(row, ci(2)), ci(h * 2 + j));
      ops.push_back(operand(indexed(tx, col), indexed(tz, col)));
      initial.push_back(indexed(c, col));
    }
    for (int q = 0; q < 4; ++q) {
      Value src = minui(subi(sc, cx(1)),
                        addi(group, addi(cx(q * 8), indexCast(index, row))));
      Value srcOp = operand(load(xy, {src}), load(zn, {src}));
      auto mma =
          LLVM::InlineAsmOp::create(
              b, loc, TypeRange{resultTy},
              ValueRange{ops[0], ops[1], srcOp, initial[0], initial[1]},
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, "
              "{$4,$5}, {$6}, {$7,$7,$8,$8};",
              "=f,=f,=f,=f,r,r,r,f,f", false, false, LLVM::TailCallKind::None,
              LLVM::AsmDialectAttr(), ArrayAttr())
              .getResult(0);
      for (int j = 0; j < 4; ++j) {
        Value v =
            LLVM::ExtractValueOp::create(b, loc, mma, ArrayRef<int64_t>{j});
        bits = ori(bits,
                   shli(shrui(bitcast(i32, v), ci(31)), ci(8 * q + 4 * h + j)));
      }
    }
  }
  for (int d : {1, 2}) {
    Value lo = ci(d == 1 ? 0x33333333 : 0x0f0f0f0f), hi = xori(lo, ci(-1));
    Value other = shuffle(gpu::ShuffleMode::XOR, bits, ci(d), ci(32)).first;
    bits = select(cmpi(CmpIPredicate::ne, andi(lane, ci(d)), ci(0)),
                  ori(andi(bits, hi), shrui(andi(other, hi), ci(2 * d))),
                  ori(andi(bits, lo), shli(andi(other, lo), ci(2 * d))));
  }
  return bits;
}

// One observer per tower. Each row contains the eight phase cycle sums,
// the wall-clock span in ns, the last cycle timestamp, and the tile count.
void Force::mark(int phase) {
  if (!probe)
    return;
  ifThen(cmpi(CmpIPredicate::eq, tid, cx(0)), [&] {
    Value now = NVVM::Clock64Op::create(b, loc, i64);
    Value last = load(trace, {owner, cx(9)});
    Value sum = load(trace, {owner, cx(phase)});
    store(addi(sum, subi(now, last)), trace, {owner, cx(phase)});
    store(now, trace, {owner, cx(9)});
    if (phase == 0)
      store(addi(load(trace, {owner, cx(10)}), cl(1)), trace, {owner, cx(10)});
  });
}

Value Force::visitTile(Value sc, Value tc, Value sb, Value tb, Value home,
                       Value lower, Value lastTile) {
  auto barrier = [&] { gpu::BarrierOp::create(b, loc); };
  ifThen(cmpi(CmpIPredicate::eq, tid, cx(0)), [&] {
    store(ci(0), totals, {cx(0)});
    store(ci(0), totals, {cx(1)});
  });
  barrier();
  mark(0);
  Value target = minui(tid, subi(tc, cx(1)));
  Value p = load(pos, {addi(cx(panel), target)});
  Value gap2 = cf(0);
  for (int a = 0; a < 2; ++a) {
    Value half = mulf(cf(0.5), truncf(load(geometry, {cx(a), cx(1)})));
    Value gap = maxnumf(cf(0), subf(absf(extract(p, a)), half));
    gap2 = addf(gap2, mulf(gap, gap));
  }
  Value keep = andi(cmpi(CmpIPredicate::ult, tid, tc),
                    cmpf(CmpFPredicate::OLE, gap2, filterLimit));
  Value votes = emitWarpBallot(b, loc, keep);
  Value reserve = ifElse(
      {i32},
      andi(cmpi(CmpIPredicate::eq, lane, ci(0)),
           cmpi(CmpIPredicate::ne, votes, ci(0))),
      [&] {
        return Values{sharedAdd(totals, cl(4), emitWarpCount(b, loc, votes))};
      },
      [&] { return Values{ci(0)}; })[0];
  reserve = indexed(reserve, ci(0));
  ifThen(keep, [&] {
    Value place = addi(
        reserve, emitWarpRank(b, loc, votes, emitLanesBelow(b, loc, lane)));
    store(indexCast(i32, target), keptSlots, {indexCast(index, place)});
  });
  barrier();
  Value kept = indexCast(index, load(totals, {cx(1)}));
  keep = cmpi(CmpIPredicate::ult, tid, kept);
  target = ifElse(
      {index}, keep,
      [&] { return Values{indexCast(index, load(keptSlots, {tid}))}; },
      [&] { return Values{cx(0)}; })[0];
  Value warpActive =
      cmpi(CmpIPredicate::ult, subi(tid, indexCast(index, lane)), kept);
  Value orig = addi(tb, target);
  Value atHome = cmpi(CmpIPredicate::ult, orig, home);
  Value high = select(atHome, minui(addi(sb, sc), lower), addi(sb, sc));
  Value low = select(atHome, maxui(addi(orig, cx(1)), sb), sb);
  mark(1);
  Values masks;
  Value counts = ci(0);
  for (int j = 0; j < groups; ++j) {
    Value first = addi(sb, cx(32 * j));
    Value l = indexCast(i32, minui(cx(32), subi(maxui(first, low), first)));
    Value h = indexCast(i32, minui(cx(32), subi(maxui(first, high), first)));
    Value active = cmpi(CmpIPredicate::ult, l, h);
    Value m = ifElse(
        {i32}, andi(warpActive, cmpi(CmpIPredicate::ult, cx(32 * j), sc)),
        [&] { return Values{mask(target, cx(32 * j), sc)}; },
        [&] { return Values{ci(0)}; })[0];
    // Mask the shift amounts too: an empty interval may begin or end at 32.
    Value range = andi(shli(ci(-1), andi(l, ci(31))),
                       shrui(ci(-1), andi(subi(ci(32), h), ci(31))));
    m = select(andi(keep, active), andi(m, range), ci(0));
    masks.push_back(m);
    counts =
        addi(counts,
             addi(select(cmpi(CmpIPredicate::ne, m, ci(0)), ci(1 << 18), ci(0)),
                  math::CtPopOp::create(b, loc, m)));
  }
  mark(2);
  Value inclusive = emitWarpScan(b, loc, counts, lane, ci(0), ci(32), 32, true);
  Value base = ifElse(
      {i32}, cmpi(CmpIPredicate::eq, lane, ci(31)),
      [&] { return Values{sharedAdd(totals, cl(0), inclusive)}; },
      [&] { return Values{ci(0)}; })[0];
  base = addi(indexed(base, ci(31)), subi(inclusive, counts));
  barrier();
  Value total = load(totals, {cx(0)});
  Value fits = cmpi(CmpIPredicate::ule, shrui(total, ci(18)), ci(recordCap));
  mark(3);
  ifThen(fits, [&] {
    for (auto [j, m] : llvm::enumerate(masks)) {
      ifThen(cmpi(CmpIPredicate::ne, m, ci(0)), [&] {
        Value rec = shrui(base, ci(18));
        Value header =
            ori(shli(base, ci(14)), ori(ci(j << 10), indexCast(i32, target)));
        LLVM::StoreOp::create(b, loc, header, recordPtr(rec, 0));
        LLVM::StoreOp::create(b, loc, m, recordPtr(rec, 1));
      });
      base = addi(base, addi(select(cmpi(CmpIPredicate::ne, m, ci(0)),
                                    ci(1 << 18), ci(0)),
                             math::CtPopOp::create(b, loc, m)));
    }
    forLoop(tid, cx(panel + stripe), cx(workers), {}, [&](Value i, ValueRange) {
      for (int a = 0; a < 3; ++a) {
        store(ci(0), accum, {cx(a), i});
        store(ci(0), coarse, {cx(a), i});
      }
      return Values{};
    });
    barrier();
    mark(4);
    Value nc = andi(total, ci(0x3ffff));
    Value worker = indexCast(i32, tid);
    Value begin = divui(muli(nc, worker), ci(workers));
    Value end = divui(muli(nc, addi(worker, ci(1))), ci(workers));
    ifThen(cmpi(CmpIPredicate::ult, begin, end), [&] {
      Value found = whileLoop(
          {i32, i32}, {ci(0), shrui(total, ci(18))},
          [&](ValueRange x) {
            return std::make_pair(
                cmpi(CmpIPredicate::ult, addi(x[0], ci(1)), x[1]), Values(x));
          },
          [&](ValueRange x) {
            Value mid = shrui(addi(x[0], x[1]), ci(1));
            Value h = recordWord(mid, 0);
            Value valid = cmpi(CmpIPredicate::ule, shrui(h, ci(14)), begin);
            return Values{select(valid, mid, x[0]), select(valid, x[1], mid)};
          })[0];
      auto read = [&](Value rec, int col) { return recordWord(rec, col); };
      Value header = read(found, 0), m = read(found, 1);
      Value skip = subi(begin, shrui(header, ci(14)));
      m = forLoop(cx(0), indexCast(index, skip), cx(1), {m},
                  [&](Value, ValueRange x) {
                    return Values{andi(x[0], subi(x[0], ci(1)))};
                  })[0];
      auto targetSlot = [&](Value h) {
        return addi(cx(panel), indexCast(index, andi(h, ci(1023))));
      };
      auto targetValues = [&](Value slot) {
        Values v{load(pos, {slot})};
        for (Value p : tileParameters)
          v.push_back(p ? load(p, {slot}) : extract(v[0], 3));
        return v;
      };
      Values initial{found, header, m,     ci(0), ci(0),
                     ci(0), ci(0),  ci(0), ci(0)};
      Values cached = targetValues(targetSlot(header));
      initial.append(cached);
      auto endState = forLoop(
          indexCast(index, begin), indexCast(index, end), cx(1), initial,
          [&](Value, ValueRange x) {
            auto v = ifElse(
                {i32, i32, i32}, cmpi(CmpIPredicate::eq, x[2], ci(0)),
                [&] {
                  Value next = addi(x[0], ci(1));
                  return Values{next, read(next, 0), read(next, 1)};
                },
                [&] { return Values{x[0], x[1], x[2]}; });
            Value prev = targetSlot(x[1]), t = targetSlot(v[1]),
                  changed = cmpi(CmpIPredicate::ne, prev, t);
            ifThen(changed, [&] {
              for (int a = 0; a < 3; ++a) {
                addWord(accum, prev, cx(a), x[3 + a]);
                addWord(coarse, prev, cx(a), x[6 + a]);
              }
            });
            Value s = indexCast(
                index, addi(shli(andi(shrui(v[1], ci(10)), ci(15)), ci(5)),
                            math::CountTrailingZerosOp::create(b, loc, v[2])));
            auto targetData = ifElse(
                TypeRange(cached), changed, [&] { return targetValues(t); },
                [&] { return Values(x.drop_front(9)); });
            Values f = pair(targetData[0], load(pos, {s}), t, s,
                            targetData.drop_front());
            Values next{v[0], v[1], andi(v[2], subi(v[2], ci(1)))};
            // Separate the rounded quotient at a power of two. At most
            // max(panel,stripe) half-unit residuals reach either endpoint.
            int shift = std::max(panel, stripe) <= 512 ? 22 : 21;
            Values quotients;
            Value any = ci(0);
            for (int a = 0; a < 3; ++a) {
              Value q = shrsi(addi(shrsi(f[a], ci(shift - 1)), ci(1)), ci(1));
              quotients.push_back(q);
              any = ori(any, q);
              addWord(accum, s, cx(a), subi(ci(0), f[a]));
              next.push_back(addi(select(changed, ci(0), x[3 + a]), f[a]));
            }
            ifThen(cmpi(CmpIPredicate::ne, any, ci(0)), [&] {
              for (int a = 0; a < 3; ++a)
                addWord(coarse, s, cx(a), subi(ci(0), quotients[a]));
            });
            for (int a = 0; a < 3; ++a)
              next.push_back(
                  addi(select(changed, ci(0), x[6 + a]), quotients[a]));
            next.append(targetData.begin(), targetData.end());
            return next;
          });
      for (int a = 0; a < 3; ++a) {
        addWord(accum, targetSlot(endState[1]), cx(a), endState[3 + a]);
        addWord(coarse, targetSlot(endState[1]), cx(a), endState[6 + a]);
      }
    });
    barrier();
    mark(5);
    forLoop(tid, cx(panel + stripe), cx(workers), {}, [&](Value i, ValueRange) {
      ifThen(ori(cmpi(CmpIPredicate::ult, i, sc),
                 andi(cmpi(CmpIPredicate::uge, i, cx(panel)),
                      cmpi(CmpIPredicate::ult, i, addi(cx(panel), tc)))),
             [&] {
               Value atom = indexCast(index, load(id, {i}));
               for (int a = 0; a < 3; ++a) {
                 int shift = std::max(panel, stripe) <= 512 ? 22 : 21;
                 Value q = load(coarse, {cx(a), i});
                 Value residual =
                     subi(load(accum, {cx(a), i}), shli(q, ci(shift)));
                 Value w = addi(
                     shli(arith::ExtSIOp::create(b, loc, i64, q), cl(shift)),
                     arith::ExtSIOp::create(b, loc, i64, residual));
                 ifThen(cmpi(CmpIPredicate::ne, w, cl(0)),
                        [&] { addWord(words, atom, cx(a), w); });
               }
             });
      return Values{};
    });
    mark(6);
  });
  ifThen(notNarrow(andi(lastTile, fits)), barrier);
  return fits;
}

void Force::traversal(Value owner) {
  int columns = 1 + 2 * g.reach;
  Values starts, counts;
  for (int field = 0; field < columns + 4; ++field) {
    starts.push_back(
        indexCast(index, load(descriptors, {owner, cx(field), cx(0)})));
    counts.push_back(
        indexCast(index, load(descriptors, {owner, cx(field), cx(1)})));
  }
  Value ns = cx(0), nt = counts[0], lower;
  for (int i = 0; i < columns; ++i) {
    ns = addi(ns, counts[i]);
    if (i == g.reach)
      lower = ns;
  }
  for (int i = columns; i < columns + 4; ++i)
    nt = addi(nt, counts[i]);
  // A tower that fits in one panel and stripe needs no live loop state.
  // On record overflow visitTile has written no forces and synchronized the
  // block, so the streaming path may restart the entire tower.
  Value oneTile = andi(cmpi(CmpIPredicate::ule, ns, cx(panel)),
                       cmpi(CmpIPredicate::ule, nt, cx(stripe)));
  Value complete = ifElse(
      {i1}, oneTile,
      [&] {
        Value nonempty = andi(cmpi(CmpIPredicate::ne, ns, cx(0)),
                              cmpi(CmpIPredicate::ne, nt, cx(0)));
        return Values{ifElse(
            {i1}, nonempty,
            [&] {
              stage(starts, counts, cx(0), ns, true);
              stage(starts, counts, cx(0), nt, false);
              return Values{
                  visitTile(ns, nt, cx(0), cx(0), counts[0], lower, cb(true))};
            },
            [&] { return Values{cb(true)}; })[0]};
      },
      [&] { return Values{cb(false)}; })[0];
  ifThen(notNarrow(complete), [&] {
    forLoop(cx(0), ns, cx(panel), {}, [&](Value sb, ValueRange) {
      Value sc = minui(cx(panel), subi(ns, sb));
      whileLoop(
          {index, index}, {cx(0), cx(stripe)},
          [&](ValueRange x) {
            return std::make_pair(cmpi(CmpIPredicate::ult, x[0], nt),
                                  Values(x));
          },
          [&](ValueRange x) {
            Value tb = x[0], tc = minui(x[1], subi(nt, tb));
            // The preceding stripe may have reused the source features for
            // sums.
            stage(starts, counts, sb, sc, true);
            stage(starts, counts, tb, tc, false);
            Value last = andi(cmpi(CmpIPredicate::eq, addi(sb, sc), ns),
                              cmpi(CmpIPredicate::eq, addi(tb, tc), nt));
            Value fits = visitTile(sc, tc, sb, tb, counts[0], lower, last);
            return Values{
                select(fits, addi(tb, tc), tb),
                select(fits, cx(stripe), divui(addi(tc, cx(1)), cx(2)))};
          });
      return Values{};
    });
  });
}

void Force::bind(ValueRange args) {
  descriptors = args[0];
  packed = args[1];
  atomIds = args[2];
  geometry = args[3];
  words = args[4];
  invalid = args[5];
  wordScale = args[6];
  cutoff2 = createCutoff2(b, loc, f32, term.getCutoff().convertToDouble());
  unsigned n = 7;
  for (unsigned j = 0; j < term.getIns().size(); ++j)
    parameters.push_back(args[n++]);
  for (Value v : captures) {
    if (matchPattern(v, m_Constant())) {
      if (!mapping.contains(v))
        b.clone(*v.getDefiningOp(), mapping);
    } else
      mapping.map(v, args[n++]);
  }
}

// Put the pair region's atom fields next to its positions. Small immutable
// rectangular tables are read once per CTA. Other captured values retain
// their original addresses and types.
void Force::stageParameters(gpu::GPUFuncOp fn) {
  int64_t fixed = fixedBytes;
  for (auto [i, v] : llvm::enumerate(parameters)) {
    Type t = cast<MemRefType>(v.getType()).getElementType();
    tileParameters.push_back(static_cast<int>(i) == positionParameter
                                 ? Value()
                                 : shared(fn, {panel + stripe}, t));
  }
  for (Value v : captures) {
    auto t = dyn_cast<MemRefType>(v.getType());
    if (!t || !t.hasStaticShape() || t.getRank() != 2 || !t.getNumElements())
      continue;
    // Remapping is local to simple loads. A call or a view could require
    // the original address space or memory layout.
    if (!llvm::all_of(v.getUsers(), [&](Operation *op) {
          return !term->isAncestor(op) || isa<memref::LoadOp>(op);
        }))
      continue;
    auto element = t.getElementType();
    auto vec = dyn_cast<VectorType>(element);
    auto scalar = getElementTypeOrSelf(element);
    if (!scalar.isIntOrFloat())
      continue;
    int64_t bytes = t.getNumElements() * (vec ? vec.getNumElements() : 1) *
                    ((scalar.getIntOrFloatBitWidth() + 7) / 8);
    if (bytes > 8192 || fixed + bytes > 48 * 1024)
      continue;
    fixed += bytes;
    Value src = mapping.lookup(v), dst = shared(fn, t.getShape(), element);
    forLoop(tid, cx(t.getNumElements()), cx(workers), {},
            [&](Value i, ValueRange) {
              Values at{divui(i, cx(t.getDimSize(1))),
                        remui(i, cx(t.getDimSize(1)))};
              store(load(src, at), dst, at);
              return Values{};
            });
    mapping.map(v, dst);
  }
}

void Force::withValidGeometry(function_ref<void()> body) {
  double rc = term.getCutoff().convertToDouble();
  Value valid = andi(cmpf(CmpFPredicate::OGT, wordScale, cf(0)),
                     cmpf(CmpFPredicate::OLE, wordScale,
                          cf(std::numeric_limits<float>::max())));
  for (int a = 0; a < 3; ++a) {
    Value width = load(geometry, {cx(a), cx(1)});
    valid = andi(valid, cmpf(CmpFPredicate::OGE, width,
                             cd(rc / (a == 2 ? g.reach : 1))));
    valid = andi(valid,
                 cmpf(CmpFPredicate::OLE, width,
                      cd(std::numeric_limits<float>::max() / (g.reach + 1.0))));
  }
  ifThenElse(valid, body, [&] {
    if (g.host)
      store(ci(2), invalid, {cx(0)});
    else
      ifThen(andi(cmpi(CmpIPredicate::eq, owner, cx(0)),
                  cmpi(CmpIPredicate::eq, tid, cx(0))),
             [&] { store(ci(2), invalid, {cx(0)}); });
  });
}

void Force::device() {
  // An external unsized array denotes the launch's dynamic shared memory.
  // A zero-element internal definition can alias the fixed tile buffers.
  auto dyn = LLVM::GlobalOp::create(b, loc, LLVM::LLVMArrayType::get(i32, 0),
                                    false, LLVM::Linkage::External,
                                    "towers_records", Attribute(), 8, 3);

  auto fn = gpu::GPUFuncOp::create(b, loc, "towers_force",
                                   b.getFunctionType(types, {}));
  fn.setKernel(true);
  fn.setKnownBlockSizeAttr(b.getDenseI32ArrayAttr({workers, 1, 1}));
  if (residency > 1)
    fn->setAttr(NVVM::NVVMDialect::getMinctasmAttrName(),
                b.getI32IntegerAttr(residency));
  pos = shared(fn, {panel + stripe}, v4);
  id = shared(fn, {panel + stripe}, i32);
  totals = shared(fn, {2}, i32);
  int slots = panel + stripe;
  // Screen features and the compacted target list die at the prefix barrier.
  // Six force planes cover all four arrays, even when the stripe dominates.
  Value scratch = shared(fn, {6 * slots * 4}, b.getI8Type());
  fn.setWorkgroupAttributionAttr(fn.getNumWorkgroupAttributions() - 1,
                                 "llvm.align", b.getI64IntegerAttr(16));
  Block &entry = fn.getBody().front();
  b.setInsertionPointToStart(&entry);
  auto view = [&](int word, ArrayRef<int64_t> shape, Type element) {
    auto type =
        MemRefType::get(shape, element, MemRefLayoutAttrInterface(),
                        cast<MemRefType>(scratch.getType()).getMemorySpace());
    return memref::ViewOp::create(b, loc, type, scratch, cx(4 * word),
                                  ValueRange{});
  };
  xy = view(0, {slots}, i32);
  zn = view(slots, {slots}, i32);
  bias = view(2 * slots, {stripe}, f32);
  keptSlots = view(2 * slots + stripe, {stripe}, i32);
  accum = view(0, {3, slots}, i32);
  coarse = view(3 * slots, {3, slots}, i32);
  bind(entry.getArguments());
  records = LLVM::AddressOfOp::create(b, loc, dyn);
  tid = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  lane = andi(indexCast(i32, tid), ci(31));
  Value bound = cf(0);
  for (int a = 0; a < 3; ++a) {
    Value width = truncf(load(geometry, {cx(a), cx(1)}));
    bound = maxnumf(bound, mulf(width, cf(a == 2 ? g.reach + 0.5 : 1.5)));
  }
  // Round the maximum coordinate magnitude up to a power of two.
  Value bits = bitcast(i32, bound);
  Value ceilPow = bitcast(f32, andi(addi(bits, ci(0x7fffff)), ci(0x7f800000)));
  scale = divf(cf(1024), ceilPow);
  Value padded =
      addf(cf(term.getCutoff().convertToDouble()), mulf(cf(0x1p-18), bound));
  filterLimit = mulf(mulf(padded, padded), cf(1 + 0x1p-19));
  // One lattice unit per axis for the two endpoint roundings, plus the
  // staging error in the common Cartesian image. Round the shell outward.
  Value margin = addf(cf(1.733), mulf(mulf(cf(0x1p-18), bound), scale));
  Value r = addf(mulf(cf(term.getCutoff().convertToDouble()), scale), margin);
  limit = addf(math::CeilOp::create(b, loc, mulf(r, r)), cf(1));
  owner = gpu::BlockIdOp::create(b, loc, gpu::Dimension::x);
  if (probe) {
    trace = entry.getArgument(types.size() - 1);
    ifThen(cmpi(CmpIPredicate::eq, tid, cx(0)), [&] {
      for (int i = 0; i < 8; ++i)
        store(cl(0), trace, {owner, cx(i)});
      store(cl(0), trace, {owner, cx(10)});
      store(NVVM::GlobalTimerOp::create(b, loc, i64), trace, {owner, cx(8)});
      store(NVVM::Clock64Op::create(b, loc, i64), trace, {owner, cx(9)});
    });
  }
  withValidGeometry([&] {
    stageParameters(fn);
    traversal(owner);
  });
  mark(7);
  if (probe)
    ifThen(cmpi(CmpIPredicate::eq, tid, cx(0)), [&] {
      store(subi(NVVM::GlobalTimerOp::create(b, loc, i64),
                 load(trace, {owner, cx(8)})),
            trace, {owner, cx(8)});
    });
  gpu::ReturnOp::create(b, loc);
}

LogicalResult Force::emit() {
  if (panel < 32 || panel > 512 || panel % 32 || workers < 32 ||
      workers > 1024 || workers % 32 || stripe < 1 || stripe > workers ||
      stripe * panel >= (1 << 18) || recordCap < groups ||
      recordCap > groups * stripe || residency < 1 || residency > 2)
    return term.emitError("infeasible TOWERS panel, stripe, worker count, "
                          "record count or residency");
  double rc = term.getCutoff().convertToDouble();
  if (!(rc > 0) || !std::isnormal(static_cast<float>(rc * rc)))
    return term.emitError(
        "TOWERS requires a positive cutoff with a normal binary32 square");
  Block &body = term.getKernel().front();
  if (term.getCenters() || term.getImages() != md_exec::Images::Minimum ||
      term.getOuts().size() != 1 || !term.getReduce().empty() ||
      term.getExchange(0) != md::Exchange::Antisymmetric ||
      body.getArgument(0).getType() != f32 ||
      body.getTerminator()->getOperand(0).getType() != v3)
    return term.emitError(
        "TOWERS requires one mixed antisymmetric force contribution");
  for (Operation &op : body.without_terminator())
    if (op.walk([](Operation *nested) {
            return isa<memref::LoadOp>(nested) || isMemoryEffectFree(nested) ||
                           nested
                               ->hasTrait<OpTrait::HasRecursiveMemoryEffects>()
                       ? WalkResult::advance()
                       : WalkResult::interrupt();
          }).wasInterrupted())
      return term.emitError("TOWERS requires a read-only pair region");
  auto mem = [&](ArrayRef<int64_t> shape, Type t) {
    return MemRefType::get(shape, t);
  };
  int slots = g.cells() * g.capacity;
  types = {mem({g.cells(), 5 + 2 * g.reach, 2}, i32),
           mem({slots}, v4),
           mem({slots}, i32),
           mem({3, 3}, f64),
           mem({g.atoms, 3}, i64),
           mem({1}, i32),
           f32};
  fixedBytes = (panel + stripe) * 44 + 16;
  for (unsigned j = 0; j < term.getIns().size(); ++j) {
    Type t = body.getArgument(2 + 2 * j).getType();
    auto scalar = getElementTypeOrSelf(t);
    if (!scalar.isIntOrFloat())
      return term.emitError("unsupported TOWERS particle input type");
    auto vec = dyn_cast<VectorType>(t);
    if (positionParameter < 0 && t == f32)
      positionParameter = j;
    else
      fixedBytes += (panel + stripe) * (vec ? vec.getNumElements() : 1) *
                    ((scalar.getIntOrFloatBitWidth() + 7) / 8);
    types.push_back(mem({g.atoms}, t));
  }
  if (!g.host && fixedBytes > 48 * 1024)
    return term.emitError("TOWERS force tile exceeds static workgroup storage");
  getUsedValuesDefinedAbove(term.getKernel(), term.getKernel(), captures);
  for (Value v : captures)
    if (!matchPattern(v, m_Constant()))
      types.push_back(v.getType());
  if (probe) {
    if (g.host)
      return term.emitError("TOWERS force phase counters require a GPU");
    types.push_back(mem({g.cells(), 11}, i64));
  }
  if (g.host) {
    auto fn = func::FuncOp::create(b, loc, "towers_force",
                                   b.getFunctionType(types, {}));
    Block *entry = fn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    bind(entry->getArguments());
    withValidGeometry([&] { host(); });
    func::ReturnOp::create(b, loc);
  } else
    device();
  return success();
}
} // namespace

LogicalResult mdir::kernels::emitTowersForce(ModuleOp module, TowersLayout g,
                                             md_exec::PairForOp term,
                                             TowersForceShape shape,
                                             bool probe) {
  OpBuilder b = OpBuilder::atBlockEnd(module.getBody());
  if (!g.host)
    b.setInsertionPointToStart(
        module.lookupSymbol<gpu::GPUModuleOp>("towers").getBody());
  return Force(b, g, term, shape, probe).emit();
}
