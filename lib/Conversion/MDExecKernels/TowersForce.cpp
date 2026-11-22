// Rectangular particle traversal [Shaw2005]. The quantized screen, prefix
// records and rank partition are the stage 4n GPU implementation of D[towers].
#include "TowersForce.h"
#include "NeighborBuildCode.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;
using namespace mdir::kernels::detail;

namespace {
constexpr int panel = 128, stripe = 128, workers = 128, groups = panel / 32;
class Force : public NeighborBuildCode {
public:
  Force(OpBuilder &b, TowersLayout g, md_exec::PairForOp term, bool probe)
      : NeighborBuildCode(b, term.getLoc()), g(g), term(term), probe(probe),
        v3(VectorType::get({3}, f32)), v4(VectorType::get({4}, f32)) {}
  LogicalResult emit();

private:
  TowersLayout g;
  md_exec::PairForOp term;
  bool probe;
  Value trace, owner;
  void mark(int phase);
  VectorType v3, v4;
  SmallVector<Type> types;
  llvm::SetVector<Value> captures;
  IRMapping mapping;
  Value descriptors, packed, atomIds, geometry, words, invalid, wordScale;
  Values parameters;
  Value pos, id, xy, zn, bias, records, totals, accum;
  Value tid, lane, scale, limit, cutoff2;
  void bind(ValueRange args);
  void host();
  void device();
  Value point(Value slot, Value field);
  Value displacement(Value target, Value source);
  Value squared(Value d);
  void pair(Value t, Value s, Value ti, Value si,
            function_ref<void(int, Value)> add);
  Value sharedAdd(Value dst, Value byteOffset, Value word);
  void addWord(Value dst, Value i, Value axis, Value w);
  Value shared(gpu::GPUFuncOp fn, ArrayRef<int64_t> shape, Type t);
  void stage(Value first, Value count, Value tileFirst, Value field,
             Value begin, Value size, bool source);
  Value mask(Value target, Value group, Value sourceCount);
  Value indexed(Value v, Value from) {
    return shuffle(gpu::ShuffleMode::IDX, v, from, ci(32)).first;
  }
  void visitTile(Value sc, Value tc, Value sb, Value tb, Value home,
                 Value lower);
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

void Force::pair(Value t, Value s, Value ti, Value si,
                 function_ref<void(int, Value)> add) {
  Value d = displacement(t, s), r2 = squared(d);
  ifThen(cmpf(CmpFPredicate::OLT, r2, cutoff2), [&] {
    Block &body = term.getKernel().front();
    IRMapping inside = mapping;
    inside.map(body.getArgument(0), r2);
    inside.map(body.getArgument(1), d);
    for (auto [i, param] : llvm::enumerate(parameters)) {
      inside.map(body.getArgument(2 + 2 * i), load(param, {ti}));
      inside.map(body.getArgument(3 + 2 * i), load(param, {si}));
    }
    for (auto &op : body.without_terminator())
      b.clone(op, inside);
    Value f = inside.lookup(body.getTerminator()->getOperand(0));
    for (int a = 0; a < 3; ++a) {
      Value scaled = mulf(extract(f, a), wordScale);
      Value fits = andi(cmpf(CmpFPredicate::OGE, scaled, cf(-0x1p63)),
                        cmpf(CmpFPredicate::OLT, scaled, cf(0x1p63)));
      ifThen(notNarrow(fits), [&] {
        if (g.host)
          store(ci(1), invalid, {cx(0)});
        else
          atomic(LLVM::AtomicBinOp::_or, pointerTo(baseAddress(invalid), cl(0)),
                 ci(1));
      });
      add(a, fptosi(i64, roundeven(select(fits, scaled, cf(0)))));
    }
  });
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
    sharedAdd(dst, muli(indexCast(i64, addi(muli(i, cx(3)), axis)), cl(8)), w);
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
                      pair(point(ts, t.field), source, ti, si,
                           [&](int a, Value w) {
                             addWord(words, ti, cx(a), w);
                             addWord(words, si, cx(a), subi(cl(0), w));
                           });
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

void Force::stage(Value first, Value count, Value tileFirst, Value field,
                  Value begin, Value size, bool source) {
  Value start = maxui(first, begin),
        end = minui(addi(first, count), addi(begin, size));
  forLoop(addi(start, tid), end, cx(workers), {}, [&](Value i, ValueRange) {
    Value addr = addi(tileFirst, subi(i, first)),
          slot = addi(cx(source ? 0 : panel), subi(i, begin));
    Value p = point(addr, field);
    store(p, pos, {slot});
    store(load(atomIds, {addr}), id, {slot});
    Values q;
    for (int a = 0; a < 3; ++a)
      q.push_back(roundeven(mulf(extract(p, a), scale)));
    Value norm =
        addf(addf(mulf(q[0], q[0]), mulf(q[1], q[1])), mulf(q[2], q[2]));
    Value fourth = cf(64);
    if (source) {
      // Round the source norm downward. This can only add screen positives.
      auto half = arith::TruncFOp::create(b, loc, b.getF16Type(),
                                          mulf(norm, cf(1.0 / 64)));
      half.setRoundingmode(arith::RoundingMode::toward_zero);
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

void Force::visitTile(Value sc, Value tc, Value sb, Value tb, Value home,
                      Value lower) {
  auto barrier = [&] { gpu::BarrierOp::create(b, loc); };
  ifThen(cmpi(CmpIPredicate::eq, tid, cx(0)),
         [&] { store(ci(0), totals, {cx(0)}); });
  forLoop(tid, cx(panel + stripe), cx(workers), {}, [&](Value i, ValueRange) {
    for (int a = 0; a < 3; ++a)
      store(cl(0), accum, {i, cx(a)});
    return Values{};
  });
  barrier();
  mark(0);
  Value target = minui(tid, subi(tc, cx(1)));
  Value orig = addi(tb, target);
  Value atHome = cmpi(CmpIPredicate::ult, orig, home);
  Value high = select(atHome, minui(addi(sb, sc), lower), addi(sb, sc));
  Value low = select(atHome, maxui(addi(orig, cx(1)), sb), sb);
  Value p = load(pos, {addi(cx(panel), target)});
  Value gap2 = cf(0);
  for (int a = 0; a < 2; ++a) {
    Value half = mulf(cf(0.5), truncf(load(geometry, {cx(a), cx(1)})));
    Value gap = maxnumf(cf(0), subf(absf(extract(p, a)), half));
    gap2 = addf(gap2, mulf(gap, gap));
  }
  Value keep =
      andi(cmpi(CmpIPredicate::ult, tid, tc),
           cmpf(CmpFPredicate::OLE, gap2, mulf(cutoff2, cf(1 + 0x1p-19))));
  mark(1);
  Values masks;
  Value counts = ci(0);
  for (int j = 0; j < groups; ++j) {
    Value first = addi(sb, cx(32 * j));
    Value l = indexCast(i32, minui(cx(32), subi(maxui(first, low), first)));
    Value h = indexCast(i32, minui(cx(32), subi(maxui(first, high), first)));
    Value active = cmpi(CmpIPredicate::ult, l, h);
    Value m = ifElse(
        {i32}, cmpi(CmpIPredicate::ult, cx(32 * j), sc),
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
  Value inclusive = counts;
  for (int d = 1; d < 32; d *= 2) {
    Value v = shuffle(gpu::ShuffleMode::UP, inclusive, ci(d), ci(32)).first;
    inclusive = addi(inclusive,
                     select(cmpi(CmpIPredicate::uge, lane, ci(d)), v, ci(0)));
  }
  Value base = ifElse(
      {i32}, cmpi(CmpIPredicate::eq, lane, ci(31)),
      [&] { return Values{sharedAdd(totals, cl(0), inclusive)}; },
      [&] { return Values{ci(0)}; })[0];
  base = addi(indexed(base, ci(31)), subi(inclusive, counts));
  mark(3);
  for (auto [j, m] : llvm::enumerate(masks)) {
    ifThen(cmpi(CmpIPredicate::ne, m, ci(0)), [&] {
      Value rec = indexCast(index, shrui(base, ci(18)));
      Value header =
          ori(shli(base, ci(14)), ori(ci(j << 10), indexCast(i32, tid)));
      store(header, records, {rec, cx(0)});
      store(m, records, {rec, cx(1)});
    });
    base =
        addi(base,
             addi(select(cmpi(CmpIPredicate::ne, m, ci(0)), ci(1 << 18), ci(0)),
                  math::CtPopOp::create(b, loc, m)));
  }
  barrier();
  mark(4);
  Value total = load(totals, {cx(0)}), nc = andi(total, ci(0x3ffff));
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
          Value h = load(records, {indexCast(index, mid), cx(0)});
          Value valid = cmpi(CmpIPredicate::ule, shrui(h, ci(14)), begin);
          return Values{select(valid, mid, x[0]), select(valid, x[1], mid)};
        })[0];
    auto read = [&](Value rec, int col) {
      return load(records, {indexCast(index, rec), cx(col)});
    };
    Value header = read(found, 0), m = read(found, 1);
    Value skip = subi(begin, shrui(header, ci(14)));
    m = forLoop(cx(0), indexCast(index, skip), cx(1), {m},
                [&](Value, ValueRange x) {
                  return Values{andi(x[0], subi(x[0], ci(1)))};
                })[0];
    forLoop(
        indexCast(index, begin), indexCast(index, end), cx(1),
        {found, header, m}, [&](Value, ValueRange x) {
          auto v = ifElse(
              {i32, i32, i32}, cmpi(CmpIPredicate::eq, x[2], ci(0)),
              [&] {
                Value next = addi(x[0], ci(1));
                return Values{next, read(next, 0), read(next, 1)};
              },
              [&] { return Values(x); });
          Value t = addi(cx(panel), indexCast(index, andi(v[1], ci(1023))));
          Value s = indexCast(
              index, addi(shli(andi(shrui(v[1], ci(10)), ci(15)), ci(5)),
                          math::CountTrailingZerosOp::create(b, loc, v[2])));
          pair(load(pos, {t}), load(pos, {s}), indexCast(index, load(id, {t})),
               indexCast(index, load(id, {s})), [&](int a, Value w) {
                 addWord(accum, t, cx(a), w);
                 addWord(accum, s, cx(a), subi(cl(0), w));
               });
          return Values{v[0], v[1], andi(v[2], subi(v[2], ci(1)))};
        });
  });
  barrier();
  mark(5);
  forLoop(tid, cx(panel + stripe), cx(workers), {}, [&](Value i, ValueRange) {
    ifThen(ori(cmpi(CmpIPredicate::ult, i, sc),
               andi(cmpi(CmpIPredicate::uge, i, cx(panel)),
                    cmpi(CmpIPredicate::ult, i, addi(cx(panel), tc)))),
           [&] {
             Value atom = indexCast(index, load(id, {i}));
             for (int a = 0; a < 3; ++a)
               addWord(words, atom, cx(a), load(accum, {i, cx(a)}));
           });
    return Values{};
  });
  barrier();
  mark(6);
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
  forLoop(cx(0), ns, cx(panel), {}, [&](Value sb, ValueRange) {
    Value sc = minui(cx(panel), subi(ns, sb)), first = cx(0);
    for (int i = 0; i < columns; ++i) {
      stage(first, counts[i], starts[i], cx(i), sb, sc, true);
      first = addi(first, counts[i]);
    }
    forLoop(cx(0), nt, cx(stripe), {}, [&](Value tb, ValueRange) {
      Value tc = minui(cx(stripe), subi(nt, tb)), begin = cx(0);
      for (int k = 0; k < 5; ++k) {
        int i = k ? columns + k - 1 : 0;
        stage(begin, counts[i], starts[i], cx(i), tb, tc, false);
        begin = addi(begin, counts[i]);
      }
      visitTile(sc, tc, sb, tb, counts[0], lower);
      return Values{};
    });
    return Values{};
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

void Force::device() {
  auto fn = gpu::GPUFuncOp::create(b, loc, "towers_force",
                                   b.getFunctionType(types, {}));
  fn.setKernel(true);
  fn.setKnownBlockSizeAttr(b.getDenseI32ArrayAttr({workers, 1, 1}));
  pos = shared(fn, {panel + stripe}, v4);
  id = shared(fn, {panel + stripe}, i32);
  xy = shared(fn, {panel + stripe}, i32);
  zn = shared(fn, {panel + stripe}, i32);
  bias = shared(fn, {stripe}, f32);
  records = shared(fn, {groups * stripe, 2}, i32);
  totals = shared(fn, {1}, i32);
  accum = shared(fn, {panel + stripe, 3}, i64);
  Block &entry = fn.getBody().front();
  b.setInsertionPointToStart(&entry);
  bind(entry.getArguments());
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
  traversal(owner);
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
  Block &body = term.getKernel().front();
  if (term.getCenters() || term.getOuts().size() != 1 ||
      !term.getReduce().empty() ||
      term.getExchange(0) != md::Exchange::Antisymmetric ||
      body.getArgument(0).getType() != f32 ||
      body.getTerminator()->getOperand(0).getType() != v3)
    return term.emitError(
        "TOWERS requires one mixed antisymmetric force contribution");
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
  for (unsigned j = 0; j < term.getIns().size(); ++j) {
    Type t = body.getArgument(2 + 2 * j).getType();
    types.push_back(mem({g.atoms}, t));
  }
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
    host();
    func::ReturnOp::create(b, loc);
  } else
    device();
  return success();
}
} // namespace

LogicalResult mdir::kernels::emitTowersForce(ModuleOp module, TowersLayout g,
                                             md_exec::PairForOp term,
                                             bool probe) {
  OpBuilder b = OpBuilder::atBlockEnd(module.getBody());
  if (!g.host)
    b.setInsertionPointToStart(
        module.lookupSymbol<gpu::GPUModuleOp>("towers").getBody());
  return Force(b, g, term, probe).emit();
}
