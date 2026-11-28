// Rectangular particle traversal [Shaw2005]. The quantized screen, prefix
// records and rank partition are the stage 4n GPU implementation of D428.
#include "mdir/Conversion/TowersForce.h"
#include "NeighborBuildCode.h"
#include "mdir/Conversion/CompactionEmitter.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Support/ForceWords.h"
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
// The fatal and replay words have disjoint writers at each launch boundary.
enum StateWord { Fatal, Replay };
enum ForceError {
  InvalidContribution = 1,
  InvalidGeometry = 2,
  InvalidTotal = 4
};
class Force : public NeighborBuildCode {
public:
  Force(OpBuilder &b, TowersLayout g, md_exec::PairForOp term,
        TowersForceShape shape, bool probe, md_exec::PairForOp correction)
      : NeighborBuildCode(b, term.getLoc()), g(g), term(term),
        correction(correction), output(shape.output),
        yieldIndex(shape.yieldIndex), virialYield(shape.virialYield),
        exclusions(shape.exclusions), split(shape.split),
        exclusionRows(shape.exclusionRowsPerThread()), panel(shape.panel),
        stripe(shape.stripe), workers(shape.workers), recordCap(shape.records),
        residency(shape.residency), groups(panel / 32),
        packedChargeType(shape.packedChargeType), certified(shape.certified),
        movingDispatch(shape.movingDispatch), scalarScreen(shape.scalarScreen),
        probe(probe), keepForceWords(shape.keepForceWords),
        indexedExclusions(shape.indexedExclusions),
        cartesianPositions(shape.cartesianPositions),
        typeCount(shape.typeCount), emitRetryCounter(shape.emitRetryCounter),
        v3(VectorType::get({3}, f32)), v4(VectorType::get({4}, f32)) {}
  LogicalResult emit();

private:
  TowersLayout g;
  md_exec::PairForOp term, correction;
  TowersOutput output;
  unsigned yieldIndex;
  int virialYield;
  Value fusedSum;
  bool fused() const { return virialYield >= 0 && !observer() && !recovering; }
  bool virial() const {
    return output != TowersOutput::Force && output != TowersOutput::Energy;
  }
  int virialComponents() {
    return (output == TowersOutput::TensorVirial ||
            (fused() && cast<VectorType>(term.getKernel()
                                             .front()
                                             .getTerminator()
                                             ->getOperand(virialYield)
                                             .getType())
                                .getNumElements() == 9))
               ? 9
               : 6;
  }
  bool observer() const { return output != TowersOutput::Force; }
  bool energy() const { return output == TowersOutput::Energy; }
  StringRef observerName() const {
    return energy() ? "towers_energy" : "towers_virial";
  }
  Value localSum;
  Value clonePair(Value t, Value s, Value ti, Value si, Value r2, Value d,
                  ValueRange targetParameters, bool physical);
  void observe(Value result, Value d, bool physical);
  void startObservation();
  void flushObservation();
  Value warpDown(Value v, int d);
  void observationEntries();
  int exclusions, split, exclusionRows;
  int64_t towerBlocks() const { return int64_t(g.cells()) * split; }
  Value bonds, inverse, raw, box, exclusionIndex;
  bool excluding = false;
  Value isExcluded(Value a, Value c);
  Value searchExclusions(Value first, Value second);
  void exclusion(Value row);

  int panel, stripe, workers, recordCap, residency, groups;
  bool packedChargeType, certified, movingDispatch, scalarScreen, probe,
      keepForceWords, indexedExclusions;
  Type cartesianPositions;
  int64_t typeCount;
  bool emitRetryCounter, counting = false;
  Value retryCounts;
  bool recovering = false;
  Value failedConversions, highWords, localVotes;
  bool shortVote() const {
    // A warp's subtotal fits whenever the whole tile's Cartesian product does.
    int bound = g.growCells ? g.atoms : g.capacity;
    return bound <= INT_MAX / (5 * (2 * g.reach + 1)) / bound;
  }
  void flag(Value state, int slot, int bits);
  void addWide(Value atom, Value axis, Value word, bool negative);
  void ownedPairs(Value owner);
  void recoveryEntries();
  Value planeWord(Value provisional, Value atom, int axis);
  Value certifyGrid(Value provisional, Value arrivals, Value grid,
                    Value stride);
  void publishGrid(Value provisional, Value destination, Value retry,
                   Value stride);
  void checkWide(Value state, Value lowWords, Value highWords, Value atom);

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
  int positionParameter = -1, typeParameter = -1;
  Value slotTypes;
  int64_t fixedBytes = 0;
  void stageParameters(gpu::GPUFuncOp fn);
  Value pos, id, features, bias, records, totals, keptSlots, accum, coarse;
  Value tid, lane, scale, limit, cutoff2, filterLimit;
  // The split parts of a tower take its sources in turn by chunks of a warp
  // (D[towers-split-interleave]): part p takes the logical chunks p,
  // p + split, ... of 32 consecutive sources, so that each warp stages one
  // contiguous chunk. A part-local index l = 32 c + r stands for the logical
  // index 32 (split c + p) + r.
  Value sourcePart;
  Value logicalSource(Value local) {
    if (split == 1)
      return local;
    Value chunk = divui(local, cx(32)), rest = remui(local, cx(32));
    return addi(muli(addi(muli(chunk, cx(split)), sourcePart), cx(32)), rest);
  }
  // The number of part-local sources whose logical index is below x.
  Value localSourcesBelow(Value x) {
    if (split == 1)
      return x;
    Value chunk = divui(x, cx(32)), rest = remui(x, cx(32));
    // Whole chunks of this part below the chunk of x, then the part of that
    // chunk below x if it belongs to this part.
    Value whole = select(
        cmpi(CmpIPredicate::ule, chunk, sourcePart), cx(0),
        divui(addi(subi(chunk, sourcePart), cx(split - 1)), cx(split)));
    Value own = cmpi(CmpIPredicate::eq, remui(chunk, cx(split)), sourcePart);
    return addi(muli(whole, cx(32)), select(own, rest, cx(0)));
  }
  void bind(ValueRange args);
  Value width(int axis);
  Value lateralWord(Value info, int word, bool packedInfo);
  // A triclinic device entry that reads the binary64 geometry converts the
  // field offsets, the face normals and the lateral words once per
  // traversal into workgroup memory (D[towers-triclinic-filter]). Layout:
  // the field offsets (3 per field), n0, n1, then the ten lateral words.
  Value blockGeometry;
  bool stagesGeometry() const {
    return g.triclinic && !g.host && !packedChargeType;
  }
  int blockGeometryWords() const { return 3 * (5 + 2 * g.reach) + 16; }
  void stageBlockGeometry();
  Value faceNormal(int axis, int component);
  Value lateral(int word);
  Value validWidths(Value raw);
  void screenConstants(Value raw);
  void prepareGeometry();
  void withValidGeometry(function_ref<void()> body);
  void host();
  void device(LLVM::GlobalOp recordsGlobal, TypeRange entryTypes,
              bool singleBlock = false);
  Values fieldOffset(Value field);
  Value offsetComponent(ValueRange offset, int component, Value info,
                        bool packedInfo);
  Values closestImage(Value a, Value c);
  Value point(Value slot, Value field);
  Value displacement(Value target, Value source);
  Value squared(Value d);
  Values pair(Value t, Value s, Value ti, Value si,
              ValueRange targetParameters = {}, bool physical = false);
  Value sharedAdd(Value dst, Value byteOffset, Value word);
  void addWord(Value dst, Value i, Value axis, Value w);
  Value shared(gpu::GPUFuncOp fn, ArrayRef<int64_t> shape, Type t);
  void stage(ArrayRef<Value> starts, ArrayRef<Value> counts, Value begin,
             Value size, bool source);
  Value screenOperand(Value x, Value z);
  Values unpackFeatures(Value pair);
  Values targetOperands(Value target);
  Value mask(ValueRange operands, Value group, Value sourceCount);
  Value indexed(Value v, Value from) {
    return shuffle(gpu::ShuffleMode::IDX, v, from, ci(32)).first;
  }
  Value visitTile(Value sc, Value tc, Value sb, Value tb, Value home,
                  Value lower, Value lastTile);
  void traversal(Value owner);
};

Values Force::fieldOffset(Value field) {
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
  return shift;
}

Value Force::offsetComponent(ValueRange offset, int component, Value info,
                             bool packedInfo) {
  auto term = [&](int axis) {
    Value step = packedInfo
                     ? load(info, {cx(7 + 3 * axis + component)})
                     : truncf(load(info, {cx(axis),
                                          cx(TowersLayout::step + component)}));
    return mulf(offset[axis], step);
  };
  return addf(addf(term(0), term(1)), term(2));
}

Value Force::point(Value slot, Value field) {
  Value p = load(packed, {slot});
  bool precomputed = g.triclinic && !g.host;
  Values shift, coords;
  if (!precomputed)
    shift = fieldOffset(field);
  for (int a = 0; a < 3; ++a) {
    Value offset;
    if (precomputed)
      offset = packedChargeType
                   ? load(geometry, {addi(cx(22 + a), muli(cx(3), field))})
                   : load(blockGeometry, {addi(cx(a), muli(cx(3), field))});
    else if (g.triclinic)
      offset = offsetComponent(shift, a, geometry, false);
    else
      offset = mulf(shift[a], width(a));
    coords.push_back(addf(extract(p, a), offset));
  }
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

Value Force::clonePair(Value t, Value s, Value ti, Value si, Value r2, Value d,
                       ValueRange targetParameters, bool physical) {
  Block &body = (physical ? correction : term).getKernel().front();
  IRMapping inside = mapping;
  inside.map(body.getArgument(0), r2);
  inside.map(body.getArgument(1), d);
  for (auto [i, param] :
       llvm::enumerate(g.host || excluding ? parameters : tileParameters)) {
    inside.map(body.getArgument(2 + 2 * i),
               targetParameters.empty()
                   ? (param ? load(param, {ti}) : extract(t, 3))
                   : targetParameters[i]);
    inside.map(body.getArgument(3 + 2 * i),
               param ? load(param, {si}) : extract(s, 3));
  }
  for (auto &op : body.without_terminator()) {
    auto div = dyn_cast<arith::DivFOp>(op);
    if (!g.host && div && div.getType() == f32 &&
        matchPattern(div.getLhs(), m_OneFloat())) {
      bool approximate = arith::bitEnumContainsAny(
          div.getFastmath(), arith::FastMathFlags::arcp | arith::FastMathFlags::afn);
      // For the inverse powers in the LJ term, even the largest subnormal
      // squared distance has a reciprocal whose square overflows. Every
      // consumer sees +infinity with either reciprocal. The cutoff bounds
      // the other end so the reciprocal of a normal input cannot underflow.
      bool ftz = approximate && !physical && div.getRhs() == body.getArgument(0) &&
                 term.getCutoff().convertToDouble() <= 0x1p63 &&
                 llvm::all_of(div->getUsers(), [](Operation *user) {
                   auto power = dyn_cast<math::FPowIOp>(user);
                   APInt exponent;
                   return power &&
                          matchPattern(power.getRhs(), m_ConstantInt(&exponent)) &&
                          exponent.sge(2);
                 });
      if (ftz)
        inside.map(div.getResult(), NVVM::RcpApproxFtzF32Op::create(b, loc, f32, r2));
      else if (arith::bitEnumContainsAny(div.getFastmath(), arith::FastMathFlags::arcp))
        inside.map(div.getResult(), LLVM::InlineAsmOp::create(
                       b, loc, TypeRange{f32}, ValueRange{inside.lookup(div.getRhs())},
                       "rcp.approx.f32 $0,$1;", "=f,f", false, false,
                       LLVM::TailCallKind::None, LLVM::AsmDialectAttr(), ArrayAttr())
                       .getResult(0));
      else
        b.clone(op, inside);
    } else
      b.clone(op, inside);
  }
  if (fused() && !excluding) {
    auto accumulate = [&] {
      Value v =
          inside.lookupOrDefault(body.getTerminator()->getOperand(virialYield));
      int size = cast<VectorType>(v.getType()).getNumElements();
      for (int a = 0; a < size; ++a) {
        int j = size == 3 ? std::array{0, 3, 5}[a] : a;
        store(addf(load(localSum, {cx(j)}), extract(v, a)), localSum, {cx(j)});
      }
    };
    if (exclusions)
      ifThen(notNarrow(isExcluded(load(id, {ti}), load(id, {si}))), accumulate);
    else
      accumulate();
  }
  return inside.lookupOrDefault(body.getTerminator()->getOperand(physical ? 0 : yieldIndex));
}

// A rejected pair gets a legal radius, but its displacement still belongs
// to the original pair. A user region that computes an address from that
// displacement or has another non-speculatable operation needs the branch.
static bool canSelectCutoff(Block &body) {
  llvm::SmallDenseSet<Value> fromDisplacement{body.getArgument(1)};
  for (Operation &op : body.without_terminator()) {
    bool depends = llvm::any_of(op.getOperands(), [&](Value operand) {
      return fromDisplacement.contains(operand);
    });
    if (isa<memref::LoadOp, vector::LoadOp>(op)) {
      if (depends)
        return false;
    } else if (!isSpeculatable(&op) || !isMemoryEffectFree(&op)) {
      return false;
    }
    if (depends)
      fromDisplacement.insert(op.result_begin(), op.result_end());
  }
  return true;
}

Values Force::pair(Value t, Value s, Value ti, Value si,
                   ValueRange targetParameters, bool physical) {
  Value d = displacement(t, s), r2 = squared(d);
  auto wt = g.host || (excluding && recovering) ? i64 : i32;
  Value take = physical ? cb(true) : cmpf(CmpFPredicate::OLT, r2, cutoff2);
  if ((recovering || virial()) && exclusions &&
      !excluding) {
    auto atom = [&](Value i) {
      if (!g.host)
        return load(id, {i});
      return packedChargeType ? load(atomIds, {i}) : indexCast(i32, i);
    };
    take = andi(take, notNarrow(isExcluded(atom(ti), atom(si))));
  }
  if (observer()) {
    ifThen(take, [&] {
      observe(clonePair(t, s, ti, si, r2, d, targetParameters, physical), d,
              physical);
    });
    return {};
  }
  // The moving force region reads a type table and a radial table. Give a
  // rejected candidate an in-range radius for the radial lookup. Mask the
  // scaled components before certification so even a NaN force contributes
  // zero words and zero failed conversions. Fused virials have side effects.
  bool selectCutoff = movingDispatch && !fused() && !excluding && !recovering &&
                      canSelectCutoff(term.getKernel().front());
  if (selectCutoff)
    r2 = select(take, r2, mulf(cutoff2, cf(0.5)));
  auto convert = [&]() -> Values {
    Value f = clonePair(t, s, ti, si, r2, d, targetParameters, physical);
    Values scaled;
    Value narrow = cb(true);
    for (int a = 0; a < 3; ++a) {
      Value x = mulf(extract(f, a), wordScale);
      if (selectCutoff)
        x = select(take, x, cf(0));
      scaled.push_back(x);
      narrow = andi(narrow, andi(cmpf(CmpFPredicate::OGE, x, cf(-0x1p31)),
                                 cmpf(CmpFPredicate::OLT, x, cf(0x1p31))));
    }
    auto wide = [&] {
      Values result;
      Value valid = cb(true);
      for (Value x : scaled) {
        Value fits = andi(cmpf(CmpFPredicate::OGE, x, cf(-0x1p63)),
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
      ifThen(notNarrow(valid),
             [&] { flag(invalid, Fatal, InvalidContribution); });
      return result;
    };
    if ((g.host || excluding) && (recovering || !certified))
      return wide();
    auto failed = [&] {
      Value vote = cl(excluding && !physical ? -1 : 1);
      if (g.host)
        store(addi(load(failedConversions, {cx(0)}), vote),
              failedConversions, {cx(0)});
      else
        store(addi(load(localVotes, {}), shortVote() ? trunci(vote) : vote),
              localVotes, {});
      return Values(3, g.host ? cl(0) : ci(0));
    };
    if (g.host) {
      Values converted = ifElse(
          {i64, i64, i64}, narrow,
          [&] {
            Values result;
            for (Value x : scaled)
              result.push_back(fptosi(i64, roundeven(x)));
            return result;
          },
          failed);
      return converted;
    }
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
          if (certified && !recovering)
            return failed();
          // A blind bonded interaction can exceed i32 before exclusion
          // replay. Keep its actual word for this stage's direct-only
          // oracle.
          Values f = wide();
          Value target = indexCast(index, load(id, {ti})),
                source = indexCast(index, load(id, {si}));
          for (int a = 0; a < 3; ++a) {
            if (recovering) {
              addWide(target, cx(a), f[a], false);
              addWide(source, cx(a), f[a], true);
            } else {
              addWord(words, target, cx(a), f[a]);
              addWord(words, source, cx(a), subi(cl(0), f[a]));
            }
          }
          return Values(3, ci(0));
        });
    return result;
  };
  if (selectCutoff)
    return convert();
  return ifElse({wt, wt, wt}, take, convert,
                [&] { return Values(3, wt == i64 ? cl(0) : ci(0)); });
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
    Value linear = movingDispatch ? addi(muli(axis, cx(g.atoms)), i)
                                  : addi(muli(i, cx(3)), axis);
    Value offset = muli(indexCast(i64, linear), cl(8));
    atomic(LLVM::AtomicBinOp::add, pointerTo(baseAddress(dst), offset), w);
  } else {
    sharedAdd(
        dst,
        muli(indexCast(i64, addi(muli(axis, cx(panel + stripe)), i)), cl(4)),
        w);
  }
}

// Virial and wide replay evaluate only physical pairs.
Value Force::isExcluded(Value a, Value c) {
  Value first = minui(a, c), second = maxui(a, c);
  if (!indexedExclusions)
    return searchExclusions(first, second);
  Value row = indexCast(index, first);
  Value distance = subi(second, first);
  // A bit represents each of the next 32 permanent IDs. For a longer
  // edge only the owner's maximum can require the exact sorted search.
  return ifElse(
      {i1}, cmpi(CmpIPredicate::ule, distance, ci(32)),
      [&] {
        Value bit = andi(subi(distance, ci(1)), ci(31));
        Value mask = load(exclusionIndex, {row, cx(0)});
        return Values{andi(
            cmpi(CmpIPredicate::ne, distance, ci(0)),
            cmpi(CmpIPredicate::ne, andi(shrui(mask, bit), ci(1)), ci(0)))};
      },
      [&] {
        Value maximum = load(exclusionIndex, {row, cx(1)});
        return ifElse(
            {i1}, cmpi(CmpIPredicate::ule, second, maximum),
            [&] { return Values{searchExclusions(first, second)}; },
            [&] { return Values{cb(false)}; });
      })[0];
}

Value Force::searchExclusions(Value first, Value second) {
  auto bounds = whileLoop(
      {index, index}, {cx(0), cx(exclusions)},
      [&](ValueRange x) {
        return std::make_pair(cmpi(CmpIPredicate::ult, x[0], x[1]), Values(x));
      },
      [&](ValueRange x) {
        Value mid = addi(x[0], divui(subi(x[1], x[0]), cx(2)));
        Value u = load(bonds, {mid, cx(0)}), v = load(bonds, {mid, cx(1)});
        Value less = ori(cmpi(CmpIPredicate::ult, u, first),
                         andi(cmpi(CmpIPredicate::eq, u, first),
                              cmpi(CmpIPredicate::ult, v, second)));
        return Values{select(less, addi(mid, cx(1)), x[0]),
                      select(less, x[1], mid)};
      });
  Value row = minui(bounds[0], cx(exclusions - 1));
  return andi(cmpi(CmpIPredicate::eq, load(bonds, {row, cx(0)}), first),
              cmpi(CmpIPredicate::eq, load(bonds, {row, cx(1)}), second));
}

// The perpendicular face height bounds each fractional component of every
// candidate closer than the component-rounded image. Keep the rounded image
// on ties, then enumerate in lexicographic order as in the reference.
Values Force::closestImage(Value a, Value c) {
  Values delta, image, v, heights;
  const int dims[] = {g.nx, g.ny, g.nz};
  Value minimum;
  for (int k = 0; k < 3; ++k) {
    auto fraction = [&](Value atom) {
      Value x = load(raw, {cx(k), atom});
      return subf(x, floor(x));
    };
    Value d = subf(fraction(a), fraction(c)), n = roundeven(d);
    delta.push_back(d);
    image.push_back(fptosi(i32, n));
    v.push_back(subf(d, n));
    // Packed slab widths were rounded to float. Round the search bound
    // outward, including the multiplication back to a whole face height.
    Value h = mulf(arith::ExtFOp::create(b, loc, f64, width(k)),
                   cd(dims[k] * (1.0 - 0x1p-20)));
    heights.push_back(h);
    minimum = k ? select(cmpf(CmpFPredicate::OLT, h, minimum), h, minimum) : h;
  }
  auto norm2 = [&](ValueRange u) {
    Value total = cd(0);
    for (int k = 0; k < 3; ++k) {
      auto part = [&](int row) {
        return mulf(u[row], load(box, {cx(3 * row + k)}));
      };
      Value x = addf(addf(part(0), part(1)), part(2));
      total = addf(total, mulf(x, x));
    }
    return total;
  };
  Value distance = norm2(v);
  auto best = ifElse(
      {i32, i32, i32},
      cmpf(CmpFPredicate::OLT, distance,
           mulf(cd(0.25), mulf(minimum, minimum))),
      [&] { return image; },
      [&] {
        Values lower, upper;
        Value radius = sqrt(distance);
        for (int k = 0; k < 3; ++k) {
          Value span = addf(divf(radius, heights[k]), cd(0x1p-45));
          lower.push_back(
              fptosi(i32, math::CeilOp::create(b, loc, subf(delta[k], span))));
          upper.push_back(
              addi(ci(1), fptosi(i32, floor(addf(delta[k], span)))));
        }
        Values init{distance, image[0], image[1], image[2]};
        auto result = forLoop(
            lower[0], upper[0], ci(1), init, [&](Value i, ValueRange first) {
              auto y =
                  forLoop(lower[1], upper[1], ci(1), first,
                          [&](Value j, ValueRange second) {
                            auto z = forLoop(
                                lower[2], upper[2], ci(1), second,
                                [&](Value k, ValueRange prior) {
                                  Values u{subf(delta[0], sitofp(f64, i)),
                                           subf(delta[1], sitofp(f64, j)),
                                           subf(delta[2], sitofp(f64, k))};
                                  Value d = norm2(u),
                                        improved = cmpf(CmpFPredicate::OLT, d,
                                                        prior[0]);
                                  return Values{select(improved, d, prior[0]),
                                                select(improved, i, prior[1]),
                                                select(improved, j, prior[2]),
                                                select(improved, k, prior[3])};
                                });
                            return Values(z.begin(), z.end());
                          });
              return Values(y.begin(), y.end());
            });
        return Values{result[1], result[2], result[3]};
      });
  return Values(best.begin(), best.end());
}

void Force::exclusion(Value row) {
  // The simulation keeps its topology tuple loops. In that case this
  // entry only cancels provisional direct words on excluded pairs.
  if (!correction && (recovering || virial()))
    return;
  excluding = true;
  Value a = indexCast(index, load(bonds, {row, cx(0)}));
  Value c = indexCast(index, load(bonds, {row, cx(1)}));
  Value sa = load(inverse, {a}), sc = load(inverse, {c});
  auto owner = [&](Value atom, Value slot) {
    return g.growCells ? load(inverse, {addi(cx(g.atoms), atom)})
                       : divui(slot, ci(g.capacity));
  };
  Value ca = owner(a, sa), cc = owner(c, sc);
  Values delta, image;
  if (g.triclinic)
    image = closestImage(a, c);
  int stride = g.cells();
  for (auto [axis, bins] : llvm::enumerate(std::array{g.nx, g.ny, g.nz})) {
    stride /= bins;
    auto coordinate = [&](Value cell) {
      return remui(divui(cell, ci(stride)), ci(bins));
    };
    Value gap = subi(coordinate(ca), coordinate(cc));
    if (g.triclinic) {
      delta.push_back(subi(gap, muli(image[axis], ci(bins))));
      continue;
    }
    Value mag = maxsi(gap, subi(ci(0), gap));
    Value wrap =
        select(cmpi(CmpIPredicate::slt, gap, ci(0)), ci(-bins), ci(bins));
    Value shift = ifElse(
        {i32},
        cmpi(CmpIPredicate::slt, muli(addi(mag, ci(1)), ci(2)), ci(bins)),
        [&] { return Values{ci(0)}; },
        [&] {
          return ifElse(
              {i32},
              cmpi(CmpIPredicate::sgt, muli(subi(mag, ci(1)), ci(2)), ci(bins)),
              [&] { return Values{wrap}; },
              [&] {
                Value length = load(box, {cx(axis)}),
                      half = mulf(length, cd(0.5));
                auto key = [&](Value atom) {
                  Value v = cartesianPositions ? load(raw, {atom, cx(axis)})
                                               : load(raw, {cx(axis), atom});
                  Value outside = ori(cmpf(CmpFPredicate::OLT, v, cd(0)),
                                      cmpf(CmpFPredicate::OGE, v, length));
                  v = ifElse(
                      {f64}, outside,
                      [&] {
                        Value rem =
                            g.host
                                ? Value(
                                      arith::RemFOp::create(b, loc, v, length))
                                : LLVM::CallOp::create(b, loc, TypeRange{f64},
                                                       "__nv_fmod",
                                                       ValueRange{v, length})
                                      .getResult();
                        return Values{
                            select(cmpf(CmpFPredicate::OLT, rem, cd(0)),
                                   addf(rem, length), rem)};
                      },
                      [&] { return Values{v}; })[0];
                  Value upper = cmpf(CmpFPredicate::OGE, v, half);
                  return Values{upper, select(upper, subf(v, half), v)};
                };
                auto x = key(a), y = key(c);
                Value forward = andi(andi(x[0], notNarrow(y[0])),
                                     cmpf(CmpFPredicate::OGT, x[1], y[1]));
                Value back = andi(andi(y[0], notNarrow(x[0])),
                                  cmpf(CmpFPredicate::OLT, x[1], y[1]));
                return Values{
                    select(forward, ci(bins), select(back, ci(-bins), ci(0)))};
              });
        })[0];
    delta.push_back(subi(gap, shift));
  }
  Value forward = cmpi(CmpIPredicate::ult, sa, sc);
  for (int axis : {2, 0, 1}) {
    Value d = delta[axis];
    forward = select(
        cmpi(CmpIPredicate::eq, d, ci(0)), forward,
        cmpi(axis == 2 ? CmpIPredicate::sgt : CmpIPredicate::slt, d, ci(0)));
  }
  Value ts = indexCast(index, select(forward, sa, sc)),
        ss = indexCast(index, select(forward, sc, sa));
  Value ti = select(forward, a, c), si = select(forward, c, a);
  Value t = load(packed, {ts}), s = load(packed, {ss});
  Values shifts;
  for (int axis = 0; axis < 3; ++axis) {
    Value d = select(forward, delta[axis], subi(ci(0), delta[axis]));
    shifts.push_back(sitofp(f32, axis == 2 ? subi(ci(0), d) : d));
  }
  if (g.triclinic) {
    Values target{shifts[0], shifts[1], cf(0)}, source{cf(0), cf(0), shifts[2]};
    for (int k = 0; k < 3; ++k) {
      for (auto [p, offset] :
           {std::pair{&t, &target}, std::pair{&s, &source}}) {
        Value move =
            offsetComponent(*offset, k, geometry, !g.host && packedChargeType);
        *p =
            vector::InsertOp::create(b, loc, addf(extract(*p, k), move), *p, k);
      }
    }
  } else {
    for (int axis = 0; axis < 3; ++axis) {
      Value &p = axis == 2 ? s : t;
      p = vector::InsertOp::create(
          b, loc, addf(extract(p, axis), mulf(shifts[axis], width(axis))), p,
          axis);
    }
  }
  auto eval = [&](bool physical) {
    return pair(t, s, packedChargeType ? ts : ti, packedChargeType ? ss : si,
                {}, physical);
  };
  // The common radial arithmetic of the unconditional correction dominates
  // the cutoff branch, allowing CSE with the cloned direct pair region.
  Values corrected;
  if (correction)
    corrected = eval(true);
  else if (!observer())
    corrected.assign(3, g.host ? cl(0) : ci(0));
  Values direct;
  if (!recovering && !virial())
    direct = eval(false);
  if (observer()) {
    excluding = false;
    return;
  }
  for (int axis = 0; axis < 3; ++axis) {
    Value word = corrected[axis];
    if (recovering) {
      addWide(ti, cx(axis), word, false);
      addWide(si, cx(axis), word, true);
    } else {
      auto extend = [&](Value v) -> Value {
        return g.host ? v : arith::ExtSIOp::create(b, loc, i64, v);
      };
      word = subi(extend(word), extend(direct[axis]));
      addWord(words, ti, cx(axis), word);
      addWord(words, si, cx(axis), subi(cl(0), word));
    }
  }
  excluding = false;
}

void Force::host() {
  forLoop(cx(0), cx(g.cells()), cx(1), {}, [&](Value owner, ValueRange) {
    ownedPairs(owner);
    return Values{};
  });
  if (exclusions)
    forLoop(cx(0), cx(exclusions), cx(1), {}, [&](Value row, ValueRange) {
      exclusion(row);
      return Values{};
    });
}

void Force::ownedPairs(Value owner) {
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
                    auto force = pair(point(ts, t.field), source,
                                      packedChargeType ? ts : ti,
                                      packedChargeType ? ss : si);
                    for (auto [axis, w] : llvm::enumerate(force)) {
                      if (recovering) {
                        addWide(ti, cx(axis), w, false);
                        addWide(si, cx(axis), w, true);
                      } else {
                        addWord(words, ti, cx(axis), w);
                        addWord(words, si, cx(axis), subi(cl(0), w));
                      }
                    }
                    return Values{};
                  });
          return Values{};
        });
      });
}

// Each lane accumulates one exact signed Q32 energy in two limbs. With fewer
// than 2^26 atoms even the sum of all signed i64 pair words fits in 128 bits.
void Force::startObservation() {
  localSum = memref::AllocaOp::create(
      b, loc, cast<MemRefType>(fused() ? fusedSum.getType() : types[4]));
  for (int j = 0; j < (energy() ? 2 : virialComponents()); ++j)
    store(energy() ? cl(0) : cd(0), localSum, {cx(j)});
}

void Force::observe(Value value, Value d, bool physical) {
  if (output == TowersOutput::DiagonalVirial || output == TowersOutput::TensorVirial) {
    int size = cast<VectorType>(value.getType()).getNumElements();
    for (int a = 0; a < size; ++a) {
      int j = size == 3 ? std::array{0, 3, 5}[a] : a;
      Value part = extract(value, a);
      store(addf(load(localSum, {cx(j)}), part), localSum, {cx(j)});
    }
    return;
  }
  if (!energy()) {
    int j = 0;
    for (int a = 0; a < 3; ++a)
      for (int k = a; k < 3; ++k) {
        Value part =
            mulf(arith::ExtFOp::create(b, loc, f64, extract(d, a)),
                 arith::ExtFOp::create(b, loc, f64, extract(value, k)));
        store(addf(load(localSum, {cx(j)}), part), localSum, {cx(j)});
        ++j;
      }
    return;
  }
  if (value.getType() == f64)
    value = truncf(value);
  Value x = mulf(value, wordScale);
  Value fits = andi(cmpf(CmpFPredicate::OGE, x, cf(-0x1p63)),
                    cmpf(CmpFPredicate::OLT, x, cf(0x1p63)));
  bool subtract = excluding && !physical;
  Value safe = select(fits, x, cf(0));
  Value word =
      g.host
          ? fptosi(i64, roundeven(safe))
          : LLVM::InlineAsmOp::create(b, loc, TypeRange{i64}, ValueRange{safe},
                                      "cvt.rni.s64.f32 $0,$1;", "=l,f", false,
                                      false, LLVM::TailCallKind::None,
                                      LLVM::AsmDialectAttr(), ArrayAttr())
                .getResult(0);
  Value upper = shrsi(word, cl(63));
  if (subtract) {
    upper = addi(xori(upper, cl(-1)),
                 arith::ExtUIOp::create(b, loc, i64,
                                        cmpi(CmpIPredicate::eq, word, cl(0))));
    word = subi(cl(0), word);
  }
  Value low = load(localSum, {cx(0)}), next = addi(low, word);
  Value carry =
      arith::ExtUIOp::create(b, loc, i64, cmpi(CmpIPredicate::ult, next, low));
  store(next, localSum, {cx(0)});
  store(addi(load(localSum, {cx(1)}), addi(upper, carry)), localSum, {cx(1)});
  ifThen(notNarrow(fits), [&] {
    if (recovering)
      flag(invalid, Fatal, InvalidContribution);
    else if (g.host)
      store(addi(load(failedConversions, {cx(0)}), cl(subtract ? -1 : 1)),
            failedConversions, {cx(0)});
    else {
      Value vote = shortVote() ? ci(subtract ? -1 : 1) : cl(subtract ? -1 : 1);
      store(addi(load(localVotes, {}), vote), localVotes, {});
    }
  });
}

Value Force::warpDown(Value v, int d) {
  return LLVM::InlineAsmOp::create(
             b, loc, TypeRange{i64}, ValueRange{v, ci(d)},
             "{ .reg .b32 l,h; mov.b64 {l,h},$1; "
             "shfl.sync.down.b32 l,l,$2,31,0xffffffff; "
             "shfl.sync.down.b32 h,h,$2,31,0xffffffff; mov.b64 $0,{l,h}; }",
             "=l,l,r", false, false, LLVM::TailCallKind::None,
             LLVM::AsmDialectAttr(), ArrayAttr())
      .getResult(0);
}

void Force::flushObservation() {
  if (!energy()) {
    for (int j = 0; j < virialComponents(); ++j) {
      Value part = load(localSum, {cx(j)});
      if (g.host)
        store(addf(load(words, {cx(j)}), part), words, {cx(j)});
      else {
        for (int delta : {16, 8, 4, 2, 1}) {
          Value bits = arith::BitcastOp::create(b, loc, i64, part);
          part = addf(part, arith::BitcastOp::create(b, loc, f64,
                                                     warpDown(bits, delta)));
        }
        ifThen(cmpi(CmpIPredicate::eq, lane, ci(0)), [&] {
          atomic(LLVM::AtomicBinOp::fadd,
                 pointerTo(baseAddress(fused() ? fusedSum : words), cl(8 * j)),
                 part);
        });
      }
    }
    return;
  }
  Value low = load(localSum, {cx(0)}), high = load(localSum, {cx(1)});
  if (!g.host) {
    // Reduce the signed two-limb sums in each warp before issuing the
    // contended low-word atomic whose return value supplies its carry.
    for (int d : {16, 8, 4, 2, 1}) {
      Value other = warpDown(low, d), next = addi(low, other);
      high = addi(addi(high, warpDown(high, d)),
                  arith::ExtUIOp::create(b, loc, i64,
                                         cmpi(CmpIPredicate::ult, next, low)));
      low = next;
    }
  }
  auto add = [&](int slot, Value delta) {
    if (!g.host)
      return atomic(LLVM::AtomicBinOp::add,
                    pointerTo(baseAddress(words), cl(8 * slot)), delta);
    Value old = load(words, {cx(slot)});
    store(addi(old, delta), words, {cx(slot)});
    return old;
  };
  Value writes = cmpi(CmpIPredicate::ne, ori(low, high), cl(0));
  if (!g.host)
    writes = andi(writes, cmpi(CmpIPredicate::eq, lane, ci(0)));
  ifThen(writes, [&] {
    Value old = add(0, low);
    Value carry = arith::ExtUIOp::create(
        b, loc, i64, cmpi(CmpIPredicate::ult, addi(old, low), old));
    Value delta = addi(high, carry);
    ifThen(cmpi(CmpIPredicate::ne, delta, cl(0)), [&] { add(1, delta); });
  });
}

void Force::observationEntries() {
  Type sum = types[4], state = types[5];
  if (!energy()) {
    emitTowersEntry(
        b, loc, g.host, "towers_virial_check", {sum, state}, virialComponents(),
        [&](Value j, ValueRange a) {
          Value v = absf(load(a[0], {j}));
          ifThen(notNarrow(cmpf(CmpFPredicate::OLE, v,
                                cd(std::numeric_limits<double>::max()))),
                 [&] { flag(a[1], Fatal, InvalidContribution); });
        });
    emitTowersEntry(
        b, loc, g.host, "towers_publish_virial", {state, sum, sum}, virialComponents(),
        [&](Value j, ValueRange a) {
          ifThen(cmpi(CmpIPredicate::eq, load(a[0], {cx(Fatal)}), ci(0)),
                 [&] { store(load(a[1], {j}), a[2], {j}); });
        });
    return;
  }
  auto overflow = [&](Value sums) {
    return cmpi(CmpIPredicate::ne, load(sums, {cx(1)}),
                shrsi(load(sums, {cx(0)}), cl(63)));
  };
  auto retry = [&](Value flags) {
    return cmpi(CmpIPredicate::ne, load(flags, {cx(Replay)}), ci(0));
  };
  // An output step certifies both provisional results before either is
  // published. The force and energy entries debit the same failure counter.
  Type forces = MemRefType::get({g.atoms, 3}, i64),
       result = MemRefType::get({1}, i64);
  emitTowersEntry(
      b, loc, g.host, "towers_certify_outputs", {forces, sum, types[7], state},
      g.atoms, [&](Value i, ValueRange a) {
        Value bad = andi(cmpi(CmpIPredicate::eq, i, cx(0)),
                         ori(overflow(a[1]), cmpi(CmpIPredicate::ne,
                                                  load(a[2], {cx(0)}), cl(0))));
        for (int k = 0; k < 3; ++k) {
          Value word = load(a[0], {i, cx(k)});
          bad =
              ori(bad, cmpi(CmpIPredicate::ne, word,
                            arith::ExtSIOp::create(b, loc, i64, trunci(word))));
        }
        ifThen(bad, [&] { flag(a[3], Replay, 1); });
      });
  emitTowersEntry(
      b, loc, g.host, "towers_publish_outputs",
      {state, forces, forces, forces, sum, sum, result}, g.atoms,
      [&](Value i, ValueRange a) {
        ifThen(cmpi(CmpIPredicate::eq, load(a[0], {cx(Fatal)}), ci(0)), [&] {
          Value replay = retry(a[0]);
          Value src = select(replay, a[2], a[1]);
          for (int k = 0; k < 3; ++k)
            store(load(src, {i, cx(k)}), a[3], {i, cx(k)});
          ifThen(cmpi(CmpIPredicate::eq, i, cx(0)), [&] {
            store(load(select(replay, a[5], a[4]), {cx(0)}), a[6], {cx(0)});
          });
        });
      });
  emitTowersEntry(
      b, loc, g.host, "towers_certify_energy", {sum, types[7], state}, 1,
      [&](Value, ValueRange a) {
        ifThen(ori(overflow(a[0]),
                   cmpi(CmpIPredicate::ne, load(a[1], {cx(0)}), cl(0))),
               [&] { flag(a[2], Replay, 1); });
      });
  if (g.host) {
    OpBuilder::InsertionGuard guard(b);
    auto fn = func::FuncOp::create(b, loc, "towers_energy_recover",
                                   b.getFunctionType(types, {}));
    Block *entry = fn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    bind(entry->getArguments());
    recovering = true;
    ifThen(retry(invalid), [&] {
      startObservation();
      host();
      flushObservation();
    });
    recovering = false;
    func::ReturnOp::create(b, loc);
  }
  emitTowersEntry(b, loc, g.host, "towers_check_recovered_energy", {sum, state},
                  1, [&](Value, ValueRange a) {
                    ifThen(andi(retry(a[1]), overflow(a[0])),
                           [&] { flag(a[1], Fatal, InvalidTotal); });
                  });
  emitTowersEntry(
      b, loc, g.host, "towers_publish_energy",
      {state, sum, sum, MemRefType::get({1}, i64)}, 1,
      [&](Value, ValueRange a) {
        ifThen(cmpi(CmpIPredicate::eq, load(a[0], {cx(Fatal)}), ci(0)), [&] {
          store(load(select(retry(a[0]), a[2], a[1]), {cx(0)}), a[3], {cx(0)});
        });
      });
}

void Force::flag(Value state, int slot, int bits) {
  if (g.host)
    store(ori(load(state, {cx(slot)}), ci(bits)), state, {cx(slot)});
  else
    atomic(LLVM::AtomicBinOp::_or, pointerTo(baseAddress(state), cl(4 * slot)),
           ci(bits));
}

// Sign extend before taking the opposite endpoint, including -2^63.
// The old low limb returned by this atomic supplies the carry for this add.
void Force::addWide(Value atom, Value axis, Value w, bool negative) {
  Value hi = shrsi(w, cl(63));
  if (negative) {
    hi = addi(
        xori(hi, cl(-1)),
        arith::ExtUIOp::create(b, loc, i64, cmpi(CmpIPredicate::eq, w, cl(0))));
    w = subi(cl(0), w);
  }
  Value offset = muli(indexCast(i64, addi(muli(atom, cx(3)), axis)), cl(8));
  auto add = [&](Value mem, Value x) -> Value {
    if (!g.host)
      return atomic(LLVM::AtomicBinOp::add, pointerTo(baseAddress(mem), offset),
                    x);
    Value old = load(mem, {atom, axis});
    store(addi(old, x), mem, {atom, axis});
    return old;
  };
  Value old = add(words, w);
  Value carry = cmpi(CmpIPredicate::ult, addi(old, w), old);
  Value delta = addi(hi, arith::ExtUIOp::create(b, loc, i64, carry));
  ifThen(cmpi(CmpIPredicate::ne, delta, cl(0)), [&] { add(highWords, delta); });
}

void Force::checkWide(Value state, Value lowWords, Value upperWords,
                      Value atom) {
  Value bad = cb(false);
  for (int axis = 0; axis < 3; ++axis) {
    Value low = load(lowWords, {atom, cx(axis)}),
          hi = load(upperWords, {atom, cx(axis)});
    bad = ori(bad, cmpi(CmpIPredicate::ne, hi, shrsi(low, cl(63))));
  }
  ifThen(bad, [&] { flag(state, Fatal, InvalidTotal); });
}

void Force::recoveryEntries() {
  Type sums = types[4], state = types[5], failed = types[7];
  auto retry = [&](Value flags) {
    return cmpi(CmpIPredicate::ne, load(flags, {cx(Replay)}), ci(0));
  };
  // Each other atom can contribute at most once to this atom. Setup
  // requires n < 2^26, so signed i32 pair words sum exactly in i64.
  // A zero failed count and an inclusive i32 endpoint test certify the
  // provisional force. Exclusion replay debits the same failed direct
  // conversions.
  emitTowersEntry(
      b, loc, g.host, "towers_certify", {sums, failed, state}, g.atoms,
      [&](Value i, ValueRange a) {
        Value bad = andi(cmpi(CmpIPredicate::eq, i, cx(0)),
                         cmpi(CmpIPredicate::ne, load(a[1], {cx(0)}), cl(0)));
        for (int axis = 0; axis < 3; ++axis) {
          Value w = load(a[0], {i, cx(axis)});
          Value low = arith::TruncIOp::create(b, loc, i32, w);
          bad = ori(bad, cmpi(CmpIPredicate::ne, w,
                              arith::ExtSIOp::create(b, loc, i64, low)));
        }
        ifThen(
            andi(bad, cmpi(CmpIPredicate::eq, load(a[2], {cx(Fatal)}), ci(0))),
            [&] { flag(a[2], Replay, 1); });
      });
  emitTowersEntry(b, loc, g.host, "towers_clear_wide", {state, sums, sums},
                  g.atoms, [&](Value i, ValueRange a) {
                    ifThen(retry(a[0]), [&] {
                      for (int axis = 0; axis < 3; ++axis) {
                        store(cl(0), a[1], {i, cx(axis)});
                        store(cl(0), a[2], {i, cx(axis)});
                      }
                    });
                  });
  if (g.host) {
    SmallVector<Type> replay(types);
    replay.push_back(sums);
    recovering = true;
    emitTowersEntry(b, loc, g.host, "towers_recover", replay,
                    g.cells() + int64_t(exclusions),
                    [&](Value i, ValueRange a) {
                      bind(a);
                      highWords = a.back();
                      ifThen(retry(invalid), [&] {
                        if (exclusions)
                          ifThenElse(
                              cmpi(CmpIPredicate::ult, i, cx(g.cells())),
                              [&] { ownedPairs(i); },
                              [&] { exclusion(subi(i, cx(g.cells()))); });
                        else
                          ownedPairs(i);
                      });
                    });
    recovering = false;
  }
  emitTowersEntry(b, loc, g.host, "towers_check_wide", {state, sums, sums},
                  g.atoms, [&](Value i, ValueRange a) {
                    ifThen(retry(a[0]),
                           [&] { checkWide(a[0], a[1], a[2], i); });
                  });
  emitTowersEntry(
      b, loc, g.host, "towers_publish", {state, sums, sums, sums}, g.atoms,
      [&](Value i, ValueRange a) {
        ifThen(cmpi(CmpIPredicate::eq, load(a[0], {cx(Fatal)}), ci(0)), [&] {
          Value source = select(retry(a[0]), a[2], a[1]);
          for (int axis = 0; axis < 3; ++axis) {
            Values at{i, cx(axis)};
            store(load(source, at), a[3], at);
          }
        });
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
  auto import = [&](Value laneSlot) {
    Value local = addi(begin, laneSlot);
    Value logical = indexCast(i32, source ? logicalSource(local) : local),
          first = ci(0), field = ci(0);
    Value addr32 = addi(indexCast(i32, starts[0]), logical);
    int columns = 2 * g.reach + 1;
    for (int k = 0; k < (source ? columns : 5); ++k) {
      int j = source || k == 0 ? k : columns + k - 1;
      if (k) {
        Value inside = cmpi(CmpIPredicate::uge, logical, first);
        addr32 = select(inside,
                        addi(indexCast(i32, starts[j]), subi(logical, first)),
                        addr32);
        field = select(inside, ci(j), field);
      }
      first = addi(first, indexCast(i32, counts[j]));
    }
    Value addr = indexCast(index, addr32);
    Value slot = addi(cx(source ? 0 : panel), laneSlot);
    Value p = point(addr, indexCast(index, field));
    Value atom = load(atomIds, {addr});
    store(atom, id, {slot});
    for (auto [j, src, dst] : llvm::enumerate(parameters, tileParameters)) {
      if (!src || (movingDispatch && int(j) == positionParameter))
        continue;
      Value value =
          movingDispatch && int(j) == typeParameter
              ? load(slotTypes, {addr})
              : load(src, {packedChargeType ? addr : indexCast(index, atom)});
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
    auto halfPair = [&](Value a, Value c) -> Value {
      // Vector truncation selects packed conversion when the target has it.
      Value floats = vector::FromElementsOp::create(
          b, loc, VectorType::get({2}, f32), ValueRange{a, c});
      Value halves = arith::TruncFOp::create(
          b, loc, VectorType::get({2}, b.getF16Type()), floats);
      return LLVM::BitcastOp::create(b, loc, i32, halves);
    };
    Value words[] = {halfPair(q[0], q[1]), halfPair(q[2], fourth)};
    store(vector::FromElementsOp::create(b, loc, VectorType::get({2}, i32),
                                         words),
          features, {slot});
  };
  for (int offset = 0; offset < (source ? panel : stripe); offset += workers) {
    // Start the target import on the warps following the source panel.
    Value thread =
        source ? tid
               : remui(addi(tid, cx(workers - panel % workers)), cx(workers));
    Value slot = addi(thread, cx(offset));
    ifThen(cmpi(CmpIPredicate::ult, slot, size), [&] { import(slot); });
  }
}

// Each warp supplies 32 targets. Lane r returns the 32 source signs for
// its target. Each m16n8k8 fragment covers 16 target rows and 8 sources.
Value Force::screenOperand(Value x, Value z) {
  Value k = andi(lane, ci(3));
  return select(cmpi(CmpIPredicate::eq, k, ci(0)), x,
                select(cmpi(CmpIPredicate::eq, k, ci(1)), z, ci(0)));
}

Values Force::unpackFeatures(Value pair) {
  Values result;
  for (int j = 0; j < 2; ++j) {
    Value halves = LLVM::BitcastOp::create(
        b, loc, VectorType::get({2}, b.getF16Type()), extract(pair, j));
    for (int k : {0, 1})
      result.push_back(arith::ExtFOp::create(b, loc, f32, extract(halves, k)));
  }
  return result;
}

Values Force::targetOperands(Value target) {
  Value pair = load(features, {addi(cx(panel), target)});
  Value c = subf(load(bias, {target}), limit);
  if (scalarScreen) {
    Values operands = unpackFeatures(pair);
    operands.push_back(c);
    return operands;
  }
  Value tx = extract(pair, 0), tz = extract(pair, 1);
  Values operands;
  for (int j = 0; j < 4; ++j) {
    Value col = addi(andi(lane, ci(-4)), ci(j));
    operands.push_back(screenOperand(indexed(tx, col), indexed(tz, col)));
    operands.push_back(indexed(c, col));
  }
  return operands;
}

Value Force::mask(ValueRange operands, Value group, Value sc) {
  if (scalarScreen) {
    Value n = minui(cx(32), subi(sc, group));
    return forLoop(cx(0), n, cx(1), {ci(0)}, [&](Value j, ValueRange mask) {
      Values source = unpackFeatures(load(features, {addi(group, j)}));
      Value residual = operands[4];
      for (int a = 0; a < 4; ++a)
        residual = math::FmaOp::create(b, loc, operands[a], source[a], residual);
      Value bit = shli(ci(1), indexCast(i32, j));
      Value inside = cmpf(CmpFPredicate::OLT, residual, cf(0));
      return Values{ori(mask[0], select(inside, bit, ci(0)))};
    })[0];
  }
  Value row = shrui(lane, ci(2)), bits = ci(0);
  auto resultTy =
      LLVM::LLVMStructType::getLiteral(context, {f32, f32, f32, f32});
  for (int q = 0; q < 4; ++q) {
    Value src = indexCast(
        index, minui(subi(indexCast(i32, sc), ci(1)),
                     addi(indexCast(i32, group), addi(ci(q * 8), row))));
    Value pair = load(features, {src});
    Value srcOp = screenOperand(extract(pair, 0), extract(pair, 1));
    for (int h = 0; h < 2; ++h) {
      auto op = operands.slice(4 * h, 4);
      auto mma =
          LLVM::InlineAsmOp::create(
              b, loc, TypeRange{resultTy},
              ValueRange{op[0], op[2], srcOp, op[1], op[3]},
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, "
              "{$4,$5}, {$6}, {$7,$7,$8,$8};",
              "=f,=f,=f,=f,r,r,r,f,f", false, false, LLVM::TailCallKind::None,
              LLVM::AsmDialectAttr(), ArrayAttr())
              .getResult(0);
      Values result;
      for (int j = 0; j < 4; ++j)
        result.push_back(
            LLVM::ExtractValueOp::create(b, loc, mma, ArrayRef<int64_t>{j}));
      // Select and sign-extend the high byte of each residual, then sum
      // four disjoint bit weights to form this fragment's nibble.
      Value nibble = LLVM::InlineAsmOp::create(
                         b, loc, TypeRange{i32}, result,
                         "{ .reg .b32 a,b,c; prmt.b32 a,$1,$2,0xfb; prmt.b32 "
                         "b,$3,$4,0xfb; "
                         "prmt.b32 c,a,b,0x5410; and.b32 c,c,0x08040201; "
                         "dp4a.u32.u32 $0,c,0x01010101,0; }",
                         "=r,f,f,f,f", false, false, LLVM::TailCallKind::None,
                         LLVM::AsmDialectAttr(), ArrayAttr())
                         .getResult(0);
      bits = ori(bits, shli(nibble, ci(8 * q + 4 * h)));
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
  if (!probe || recovering)
    return;
  ifThen(cmpi(CmpIPredicate::eq, tid, cx(0)), [&] {
    Value now = NVVM::Clock64Op::create(b, loc, i64);
    Value last = load(trace, {cx(9)});
    Value sum = load(trace, {cx(phase)});
    store(addi(sum, subi(now, last)), trace, {cx(phase)});
    store(now, trace, {cx(9)});
    if (phase == 0)
      store(addi(load(trace, {cx(10)}), cl(1)), trace, {cx(10)});
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
  if (g.triclinic) {
    // The cross-section P of the column, centered at the bin, is a
    // parallelogram in the plane normal to the tower axis. A projection onto
    // a unit vector u of that plane never lengthens a distance, so
    // max(0, |u.p| - s(u)) <= d(p, P) for the support s(u) of P. The bound
    // takes the largest over the two face normals and the two diagonal axes
    // (n0 +- n1)/|n0 +- n1|, whose supports cover the corners that the faces
    // alone leave open. A sum of the squared face gaps would overestimate
    // the distance where the normals are not orthogonal.
    Value bound = cf(0);
    for (int a = 0; a < 4; ++a) {
      auto axis = [&](int k) {
        return a >= 2 ? lateral(3 * (a - 2) + k) : faceNormal(a, k);
      };
      Value projected = math::FmaOp::create(
          b, loc, axis(0), extract(p, 0),
          math::FmaOp::create(b, loc, axis(1), extract(p, 1),
                              mulf(axis(2), extract(p, 2))));
      bound = maxnumf(bound, subf(absf(projected), lateral(6 + a)));
    }
    // Ties at the filter radius are kept.
    bound = maxnumf(cf(0), bound);
    gap2 = mulf(bound, bound);
  } else {
    // Orthogonal face normals: the sum of the squared gaps is the distance.
    for (int a = 0; a < 2; ++a) {
      Value half = mulf(cf(0.5), width(a));
      Value gap = maxnumf(cf(0), subf(absf(extract(p, a)), half));
      gap2 = addf(gap2, mulf(gap, gap));
    }
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
  // A home target takes the sources of the lower layers and of home above
  // its own slot: logical indices in (orig, lower), a contiguous range of
  // part-local indices.
  Value high = select(atHome, minui(addi(sb, sc), localSourcesBelow(lower)),
                      addi(sb, sc));
  Value low = select(atHome,
                     maxui(localSourcesBelow(addi(orig, cx(1))), sb), sb);
  mark(1);
  // Empty warps take no target loads, matrix instructions or mask counts.
  auto screen = [&]() {
    Values operands = targetOperands(target);
    Values masks;
    Value counts = ci(0);
    for (int j = 0; j < groups; ++j) {
      Value first = addi(indexCast(i32, sb), ci(32 * j));
      Value left = maxsi(ci(0), subi(indexCast(i32, low), first));
      Value right =
          maxsi(ci(0), subi(addi(first, ci(32)), indexCast(i32, high)));
      Value m = ifElse(
          {i32},
          observer()
              ? cmpi(CmpIPredicate::ult, cx(32 * j), sc)
              : andi(warpActive, cmpi(CmpIPredicate::ult, cx(32 * j), sc)),
          [&] { return Values{mask(operands, cx(32 * j), sc)}; },
          [&] { return Values{ci(0)}; })[0];
      // PTX shifts by the word width or more return zero for an empty interval.
      Value range =
          LLVM::InlineAsmOp::create(
              b, loc, TypeRange{i32}, ValueRange{left, right},
              "{ .reg .b32 l,r; shl.b32 l, -1, $1; shr.u32 r, -1, $2; "
              "and.b32 $0,l,r; }",
              "=r,r,r", false, false, LLVM::TailCallKind::None,
              LLVM::AsmDialectAttr(), ArrayAttr())
              .getResult(0);
      m = select(keep, andi(m, range), ci(0));
      masks.push_back(m);
      counts = addi(counts, addi(select(cmpi(CmpIPredicate::ne, m, ci(0)),
                                        ci(1 << 18), ci(0)),
                                 math::CtPopOp::create(b, loc, m)));
    }
    masks.push_back(counts);
    return masks;
  };
  Values masks = observer()
                     ? ifElse(SmallVector<Type>(groups + 1, i32), warpActive,
                              screen, [&] { return Values(groups + 1, ci(0)); })
                     : screen();
  Value counts = masks.pop_back_val();
  mark(2);
  auto prefix = [&] {
    Value inclusive = counts;
    for (int offset : {1, 2, 4, 8, 16}) {
      inclusive = LLVM::InlineAsmOp::create(
                      b, loc, TypeRange{i32}, ValueRange{inclusive, ci(offset)},
                      "{ .reg .pred p; .reg .b32 up; shfl.sync.up.b32 "
                      "up|p,$1,$2,0,0xffffffff; "
                      "mov.b32 $0,$1; @p add.u32 $0,up,$1; }",
                      "=r,r,r", false, false, LLVM::TailCallKind::None,
                      LLVM::AsmDialectAttr(), ArrayAttr())
                      .getResult(0);
    }
    Value base = ifElse(
        {i32},
        andi(cmpi(CmpIPredicate::eq, lane, ci(31)),
             cmpi(CmpIPredicate::ne, inclusive, ci(0))),
        [&] { return Values{sharedAdd(totals, cl(0), inclusive)}; },
        [&] { return Values{ci(0)}; })[0];
    return Values{addi(indexed(base, ci(31)), subi(inclusive, counts))};
  };
  Value base = observer() ? ifElse({i32}, warpActive, prefix,
                                   [&] { return Values{ci(0)}; })[0]
                          : prefix()[0];
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
    // The feature storage is reused for force planes after screening.
    // Only imported source and target slots can receive a contribution.
    Value occupied = addi(sc, tc);
    auto activeSlot = [&](Value i) {
      return select(cmpi(CmpIPredicate::ult, i, sc), i,
                    addi(cx(panel), subi(i, sc)));
    };
    if (!observer())
      forLoop(tid, occupied, cx(workers), {}, [&](Value i, ValueRange) {
        Value slot = activeSlot(i);
        for (int a = 0; a < 3; ++a) {
          store(ci(0), accum, {cx(a), slot});
          store(ci(0), coarse, {cx(a), slot});
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
      Value firstBit =
          LLVM::InlineAsmOp::create(
              b, loc, TypeRange{i32}, ValueRange{m, addi(skip, ci(1))},
              "fns.b32 $0, $1, 0, $2;", "=r,r,r", false, false,
              LLVM::TailCallKind::None, LLVM::AsmDialectAttr(), ArrayAttr())
              .getResult(0);
      m = andi(m, shli(ci(-1), firstBit));
      auto targetSlot = [&](Value h) {
        return addi(cx(panel), indexCast(index, andi(h, ci(1023))));
      };
      auto targetValues = [&](Value slot) {
        Values v{load(pos, {slot})};
        for (Value p : tileParameters)
          v.push_back(p ? load(p, {slot}) : extract(v[0], 3));
        return v;
      };
      Values initial{found, header, m, ci(0), ci(0), ci(0)};
      Values cached = targetValues(targetSlot(header));
      initial.append(cached);
      auto endState =
          forLoop(begin, end, ci(1), initial, [&](Value, ValueRange x) {
            // All remaining bits of a record use the same target. Carry its
            // words and fields across those bits; compare slots only on refill.
            auto v = ifElse(
                TypeRange(initial), cmpi(CmpIPredicate::eq, x[2], ci(0)),
                [&] {
                  Value next = addi(x[0], ci(1));
                  Value h = read(next, 0), mask = read(next, 1);
                  Value prev = targetSlot(x[1]), t = targetSlot(h);
                  auto state = ifElse(
                      TypeRange(x.drop_front(3)), cmpi(CmpIPredicate::ne, prev, t),
                      [&] {
                        if (!observer())
                          for (int a = 0; a < 3; ++a)
                            addWord(accum, prev, cx(a), x[3 + a]);
                        Values fresh(3, ci(0));
                        fresh.append(targetValues(t));
                        return fresh;
                      },
                      [&] { return Values(x.drop_front(3)); });
                  Values refill{next, h, mask};
                  refill.append(state.begin(), state.end());
                  return refill;
                },
                [&] { return Values(x); });
            Value t = targetSlot(v[1]);
            Value s = indexCast(
                index, addi(shli(andi(shrui(v[1], ci(10)), ci(15)), ci(5)),
                            math::CountTrailingZerosOp::create(b, loc, v[2])));
            ValueRange targetData = v.drop_front(6);
            Values f = pair(targetData[0], load(pos, {s}), t, s,
                            targetData.drop_front());
            Values next{v[0], v[1], andi(v[2], subi(v[2], ci(1)))};
            if (observer()) {
              next.append(x.begin() + 3, x.begin() + 6);
              next.append(targetData.begin(), targetData.end());
              return next;
            }
            // Separate the rounded quotient at a power of two. At most
            // max(panel,stripe) half-unit residuals reach either endpoint.
            int shift = std::max(panel, stripe) <= 512 ? 22 : 21;
            // In biased unsigned form every component with zero quotient
            // has only its low shift bits set, including the negative tie.
            Value biased = ci(0);
            for (int a = 0; a < 3; ++a) {
              biased = ori(biased, addi(f[a], ci(1 << (shift - 1))));
              addWord(accum, s, cx(a), subi(ci(0), f[a]));
              next.push_back(addi(v[3 + a], f[a]));
            }
            ifThen(cmpi(CmpIPredicate::uge, biased, ci(1 << shift)), [&] {
              for (int a = 0; a < 3; ++a) {
                Value q = shrsi(addi(shrsi(f[a], ci(shift - 1)), ci(1)), ci(1));
                addWord(coarse, s, cx(a), subi(ci(0), q));
                addWord(coarse, t, cx(a), q);
              }
            });
            next.append(targetData.begin(), targetData.end());
            return next;
          });
      if (!observer())
        for (int a = 0; a < 3; ++a) {
          addWord(accum, targetSlot(endState[1]), cx(a), endState[3 + a]);
        }
    });
    barrier();
    mark(5);
    if (!observer())
      forLoop(tid, occupied, cx(workers), {}, [&](Value thread, ValueRange) {
        Value i = activeSlot(thread);
        Value atom = indexCast(index, load(id, {i}));
        for (int a = 0; a < 3; ++a) {
          int shift = std::max(panel, stripe) <= 512 ? 22 : 21;
          Value q = load(coarse, {cx(a), i});
          Value residual = subi(load(accum, {cx(a), i}), shli(q, ci(shift)));
          Value w =
              addi(shli(arith::ExtSIOp::create(b, loc, i64, q), cl(shift)),
                   arith::ExtSIOp::create(b, loc, i64, residual));
          ifThen(cmpi(CmpIPredicate::ne, w, cl(0)), [&] {
            if (recovering)
              addWide(atom, cx(a), w, false);
            else
              addWord(words, atom, cx(a), w);
          });
        }
        return Values{};
      });
    mark(6);
  });
  ifThen(notNarrow(andi(lastTile, fits)), barrier);
  return fits;
}

void Force::traversal(Value block) {
  // Entries that traverse no tower, such as a finish without a replay, do
  // not convert the geometry.
  stageBlockGeometry();
  Value owner = divui(block, cx(split)), part = remui(block, cx(split));
  int columns = 1 + 2 * g.reach;
  Values starts, counts;
  Value ns, nt, lower, begin, end;
  for (int field = 0; field < columns + 4; ++field) {
    Value at = addi(muli(owner, cx(columns + 4)), cx(field));
    Value ptr =
        pointerTo(baseAddress(descriptors), muli(indexCast(i64, at), cl(8)));
    auto type = VectorType::get({2}, i32);
    auto span = LLVM::LoadOp::create(b, loc, type, ptr, 4);
    starts.push_back(indexCast(index, extract(span, 0)));
    counts.push_back(indexCast(index, extract(span, 1)));
  }
  ns = cx(0);
  nt = counts[0];
  for (int i = 0; i < columns; ++i) {
    ns = addi(ns, counts[i]);
    if (i == g.reach)
      lower = ns;
  }
  // Contiguous halves of the column would give the first part the home and
  // lower layers, the only sources that meet home targets: up to nine times
  // the screen passes of the second on a Trp-cage cell. Taking chunks of
  // sources in turn gives each part the same mix of layers.
  sourcePart = part;
  begin = cx(0);
  end = localSourcesBelow(ns);
  for (int i = columns; i < columns + 4; ++i)
    nt = addi(nt, counts[i]);
  // Keep a single compiled copy of the pair body for both single and
  // multi-tile towers. Advance the source panel only after its last stripe.
  whileLoop(
      {index, index, index}, {begin, cx(0), cx(stripe)},
      [&](ValueRange x) {
        return std::make_pair(andi(cmpi(CmpIPredicate::ult, x[0], end),
                                   cmpi(CmpIPredicate::ult, x[1], nt)),
                              Values(x));
      },
      [&](ValueRange x) {
        Value sb = x[0], tb = x[1];
        Value sc = minui(cx(panel), subi(end, sb)),
              tc = minui(x[2], subi(nt, tb));
        Value nextSource = addi(sb, sc), nextTarget = addi(tb, tc);
        Value lastStripe = cmpi(CmpIPredicate::eq, nextTarget, nt);
        Value last = andi(lastStripe, cmpi(CmpIPredicate::eq, nextSource, end));
        stage(starts, counts, sb, sc, true);
        stage(starts, counts, tb, tc, false);
        Value fits = visitTile(sc, tc, sb, tb, counts[0], lower, last);
        if (counting)
          ifThen(andi(cmpi(CmpIPredicate::eq, tid, cx(0)), notNarrow(fits)),
                 [&] {
                   store(addi(load(retryCounts, {block}), cl(1)), retryCounts,
                         {block});
                 });
        Value advance = andi(fits, lastStripe);
        return Values{select(advance, nextSource, sb),
                      select(advance, cx(0), select(fits, nextTarget, tb)),
                      select(fits, cx(stripe), divui(addi(tc, cx(1)), cx(2)))};
      });
}

void Force::bind(ValueRange args) {
  mapping.clear();
  parameters.clear();
  descriptors = args[0];
  packed = args[1];
  atomIds = args[2];
  geometry = args[3];
  words = args[4];
  invalid = args[5];
  wordScale = args[6];
  double cutoff = term.getCutoff().convertToDouble();
  cutoff2 = createCutoff2(b, loc, f32, cutoff);
  unsigned n = 7;
  if (certified)
    failedConversions = args[n++];
  if (exclusions) {
    bonds = args[n++];
    inverse = args[n++];
    raw = args[n++];
    box = args[n++];
    if (indexedExclusions)
      exclusionIndex = args[n++];
  }
  for (unsigned j = 0; j < term.getIns().size(); ++j)
    parameters.push_back(packedChargeType && j == 0 ? Value() : args[n++]);
  if (movingDispatch)
    slotTypes = args[n++];
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
    Type t = term.getKernel().front().getArgument(2 + 2 * i).getType();
    tileParameters.push_back(static_cast<int>(i) == positionParameter
                                 ? Value()
                                 : shared(fn, {panel + stripe}, t));
  }
  for (Value v : captures) {
    auto t = dyn_cast<MemRefType>(v.getType());
    if (!t || t.getRank() != 2)
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
    int64_t rows = t.getDimSize(0), columns = t.getDimSize(1);
    if (!t.hasStaticShape()) {
      // Folded LJ coefficients have a dynamic descriptor. Only specialize a
      // table whose two subscripts come directly from particle type fields.
      if (!movingDispatch || typeCount <= 0 || typeCount > 32 ||
          typeParameter < 0 || element != VectorType::get({2}, f32))
        continue;
      Block &body = term.getKernel().front();
      auto fromType = [&](Value index) {
        auto cast = index.getDefiningOp<arith::IndexCastOp>();
        if (!cast)
          return false;
        Value value = cast.getIn();
        return value == body.getArgument(2 + 2 * typeParameter) ||
               value == body.getArgument(3 + 2 * typeParameter);
      };
      if (!llvm::all_of(v.getUsers(), [&](Operation *op) {
            if (!term->isAncestor(op))
              return true;
            auto indices = cast<memref::LoadOp>(op).getIndices();
            return llvm::all_of(indices, fromType);
          }))
        continue;
      rows = columns = typeCount;
    }
    int64_t entries = rows * columns;
    if (!entries)
      continue;
    int64_t bytes = entries * (vec ? vec.getNumElements() : 1) *
                    ((scalar.getIntOrFloatBitWidth() + 7) / 8);
    if (bytes > 8192 || fixed + bytes > 48 * 1024)
      continue;
    fixed += bytes;
    Value src = mapping.lookup(v), dst = shared(fn, {rows, columns}, element);
    forLoop(tid, cx(entries), cx(workers), {},
            [&](Value i, ValueRange) {
              Values at{divui(i, cx(columns)),
                        remui(i, cx(columns))};
              auto copy = [&] { store(load(src, at), dst, at); };
              if (t.hasStaticShape())
                copy();
              else {
                Value nr = memref::DimOp::create(b, loc, src, 0);
                Value nc = memref::DimOp::create(b, loc, src, 1);
                ifThen(andi(cmpi(CmpIPredicate::ult, at[0], nr),
                            cmpi(CmpIPredicate::ult, at[1], nc)), copy);
              }
              return Values{};
            });
    mapping.map(v, dst);
  }
}

Value Force::width(int axis) {
  if (!g.host && packedChargeType)
    return load(geometry, {cx(axis)});
  return truncf(
      load(geometry, {cx(axis), cx(g.triclinic ? TowersLayout::slab : 1)}));
}

// The diagonal lateral axes (words 0 to 5) and the supports of n0, n1, the
// first and the second diagonal (words 6 to 9) of the column's cross-section
// (TowersBuild::geometry). The components of the axes round to nearest; the
// supports round up to binary32, so that the bound stays below the distance
// computed with the same axes up to the f32 error of the dot product, which
// the padding of the filter radius covers (screenConstants).
Value Force::lateralWord(Value info, int word, bool packedInfo) {
  if (packedInfo)
    return load(info, {cx(g.lateralAxesOffset() + word)});
  if (word < 6)
    return truncf(
        load(info, {cx(word / 3), cx(TowersLayout::diagonal + word % 3)}));
  Value x = load(info, {cx(word % 2),
                        cx(TowersLayout::support + (word - 6) / 2)});
  Value nearest = truncf(x);
  Value below = cmpf(CmpFPredicate::OLT,
                     arith::ExtFOp::create(b, loc, f64, nearest), x);
  // Supports are positive: the next binary32 is one unit of the last place up.
  Value next = bitcast(f32, addi(bitcast(i32, nearest), ci(1)));
  return select(below, next, nearest);
}

Value Force::faceNormal(int axis, int component) {
  if (stagesGeometry())
    return load(blockGeometry,
                {cx(3 * (5 + 2 * g.reach) + 3 * axis + component)});
  if (!g.host && packedChargeType)
    return load(geometry, {cx(16 + 3 * axis + component)});
  return truncf(
      load(geometry, {cx(axis), cx(TowersLayout::normal + component)}));
}

Value Force::lateral(int word) {
  if (stagesGeometry())
    return load(blockGeometry, {cx(3 * (5 + 2 * g.reach) + 6 + word)});
  return lateralWord(geometry, word, !g.host && packedChargeType);
}

// Every word is converted from the binary64 geometry by one thread at the
// start of a traversal, so that staging a particle and filtering a target
// read binary32 workgroup memory instead of converting binary64 rows at each
// use.
void Force::stageBlockGeometry() {
  if (!stagesGeometry())
    return;
  int fields = 5 + 2 * g.reach;
  for (int w = 0; w < blockGeometryWords(); ++w)
    ifThen(cmpi(CmpIPredicate::eq, tid, cx(w % workers)), [&] {
      Value v;
      if (w < 3 * fields)
        v = offsetComponent(fieldOffset(cx(w / 3)), w % 3, geometry, false);
      else if (w < 3 * fields + 6)
        v = truncf(load(geometry,
                        {cx((w - 3 * fields) / 3),
                         cx(TowersLayout::normal + (w - 3 * fields) % 3)}));
      else
        v = lateralWord(geometry, w - 3 * fields - 6, false);
      store(v, blockGeometry, {cx(w)});
    });
  gpu::BarrierOp::create(b, loc);
}

Value Force::validWidths(Value raw) {
  double rc = term.getCutoff().convertToDouble();
  Value valid = cb(true);
  for (int a = 0; a < 3; ++a) {
    Value width = load(raw, {cx(a), cx(g.triclinic ? TowersLayout::slab : 1)});
    valid = andi(valid, cmpf(CmpFPredicate::OGE, width,
                             cd(rc / (a == 2 ? g.reach : 1))));
    valid = andi(valid,
                 cmpf(CmpFPredicate::OLE, width,
                      cd(std::numeric_limits<float>::max() / (g.reach + 1.0))));
  }
  return valid;
}

void Force::screenConstants(Value raw) {
  Value bound = cf(0);
  for (int a = 0; a < 3; ++a) {
    if (g.triclinic) {
      Value extent = cf(0);
      for (int k = 0; k < 3; ++k) {
        Value step =
            absf(truncf(load(raw, {cx(k), cx(TowersLayout::step + a)})));
        extent = addf(extent, mulf(step, cf(k == 2 ? g.reach + 0.5 : 1.5)));
      }
      bound = maxnumf(bound, extent);
      continue;
    }
    Value width = truncf(load(raw, {cx(a), cx(1)}));
    bound = maxnumf(bound, mulf(width, cf(a == 2 ? g.reach + 0.5 : 1.5)));
  }
  // Round the maximum coordinate magnitude up to a power of two.
  Value bits = bitcast(i32, bound);
  Value ceilPow = bitcast(f32, andi(addi(bits, ci(0x7fffff)), ci(0x7f800000)));
  scale = divf(cf(1024), ceilPow);
  Value padded = addf(cf(term.getCutoff().convertToDouble()),
                      mulf(cf(g.triclinic ? 0x1p-15 : 0x1p-18), bound));
  filterLimit = mulf(mulf(padded, padded), cf(1 + 0x1p-19));
  // One lattice unit per axis for the two endpoint roundings, plus the
  // staging error in the common Cartesian image. Round the shell outward.
  Value margin = addf(
      cf(1.733), mulf(mulf(cf(g.triclinic ? 0x1p-15 : 0x1p-18), bound), scale));
  Value r = addf(mulf(cf(term.getCutoff().convertToDouble()), scale), margin);
  limit = addf(math::CeilOp::create(b, loc, mulf(r, r)), cf(1));
}

void Force::prepareGeometry() {
  OpBuilder::InsertionGuard guard(b);
  auto raw = MemRefType::get({3, g.rawColumns()}, f64);
  auto fn = gpu::GPUFuncOp::create(b, loc, "towers_prepare_force",
                                   b.getFunctionType({raw, types[3]}, {}));
  fn.setKernel(true);
  Block &entry = fn.getBody().front();
  b.setInsertionPointToStart(&entry);
  Value thread = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  ifThen(cmpi(CmpIPredicate::eq, thread, cx(0)), [&] {
    Value input = entry.getArgument(0), output = entry.getArgument(1);
    screenConstants(input);
    for (int a = 0; a < 3; ++a)
      store(truncf(
                load(input, {cx(a), cx(g.triclinic ? TowersLayout::slab : 1)})),
            output, {cx(a)});
    if (g.triclinic) {
      for (int a = 0; a < 3; ++a)
        for (int k = 0; k < 3; ++k) {
          store(truncf(load(input, {cx(a), cx(TowersLayout::step + k)})),
                output, {cx(7 + 3 * a + k)});
          if (a < 2)
            store(truncf(load(input, {cx(a), cx(TowersLayout::normal + k)})),
                  output, {cx(16 + 3 * a + k)});
        }
      for (int f = 0; f < 5 + 2 * g.reach; ++f) {
        Values shift = fieldOffset(cx(f));
        for (int k = 0; k < 3; ++k)
          store(offsetComponent(shift, k, input, false), output,
                {cx(22 + 3 * f + k)});
      }
      for (int word = 0; word < TowersLayout::lateralAxesWords; ++word)
        store(lateralWord(input, word, false), output,
              {cx(g.lateralAxesOffset() + word)});
    }
    store(scale, output, {cx(3)});
    store(filterLimit, output, {cx(4)});
    store(limit, output, {cx(5)});
    store(select(validWidths(input), cf(1), cf(0)), output, {cx(6)});
  });
  gpu::ReturnOp::create(b, loc);
}

void Force::withValidGeometry(function_ref<void()> body) {
  Value widths = !g.host && packedChargeType
                     ? cmpf(CmpFPredicate::OEQ, load(geometry, {cx(6)}), cf(1))
                     : validWidths(geometry);
  Value valid = andi(widths, andi(cmpf(CmpFPredicate::OGT, wordScale, cf(0)),
                                  cmpf(CmpFPredicate::OLE, wordScale,
                                       cf(std::numeric_limits<float>::max()))));
  ifThenElse(valid, body, [&] {
    if (g.host)
      store(ci(InvalidGeometry), invalid, {cx(Fatal)});
    else
      ifThen(andi(cmpi(CmpIPredicate::eq, owner, cx(0)),
                  cmpi(CmpIPredicate::eq, tid, cx(0))),
             [&] { store(ci(InvalidGeometry), invalid, {cx(Fatal)}); });
  });
}

Value Force::planeWord(Value provisional, Value atom, int axis) {
  Value linear = addi(muli(cx(axis), cx(g.atoms)), atom);
  return load(provisional, {divui(linear, cx(3)), remui(linear, cx(3))});
}

Value Force::certifyGrid(Value provisional, Value arrivals, Value grid,
                         Value stride) {
  Value first = addi(muli(owner, cx(workers)), tid);
  forLoop(first, cx(g.atoms), stride, {}, [&](Value atom, ValueRange) {
    Value bad = cb(false);
    for (int a = 0; a < 3; ++a) {
      Value w = planeWord(provisional, atom, a);
      bad = ori(bad, cmpi(CmpIPredicate::ne, w,
                          arith::ExtSIOp::create(b, loc, i64, trunci(w))));
    }
    ifThen(bad, [&] { flag(invalid, Replay, 1); });
    return Values{};
  });
  gpu::BarrierOp::create(b, loc);
  ifThen(cmpi(CmpIPredicate::eq, tid, cx(0)), [&] {
    // All blocks are resident. Release this block's votes, then acquire
    // the final arrival before reading the frame-wide replay decision.
    Value ptr = pointerTo(baseAddress(arrivals), cl(0));
    LLVM::AtomicRMWOp::create(b, loc, LLVM::AtomicBinOp::add, ptr, ci(1),
                              LLVM::AtomicOrdering::acq_rel, "device");
    whileLoop(
        {}, {},
        [&](ValueRange) {
          auto arrived = LLVM::LoadOp::create(b, loc, i32, ptr, 4, true);
          arrived.setOrdering(LLVM::AtomicOrdering::acquire);
          arrived.setSyncscopeAttr(b.getStringAttr("device"));
          return std::make_pair(
              cmpi(CmpIPredicate::ult, arrived, indexCast(i32, grid)),
              Values{});
        },
        [&](ValueRange) { return Values{}; });
    Value failed =
        cmpi(CmpIPredicate::ne, load(failedConversions, {cx(0)}), cl(0));
    Value retry = cmpi(CmpIPredicate::ne, load(invalid, {cx(Replay)}), ci(0));
    store(select(ori(failed, retry), ci(1), ci(0)), totals, {cx(0)});
  });
  gpu::BarrierOp::create(b, loc);
  return cmpi(CmpIPredicate::ne, load(totals, {cx(0)}), ci(0));
}

void Force::publishGrid(Value provisional, Value destination, Value retry,
                        Value stride) {
  gpu::BarrierOp::create(b, loc);
  Value lead = cmpi(CmpIPredicate::eq, owner, cx(0));
  ifThen(ori(notNarrow(retry), lead), [&] {
    Value bad = cmpi(CmpIPredicate::ne, load(invalid, {cx(Fatal)}), ci(0));
    ifThenElse(
        bad,
        [&] {
          LLVM::InlineAsmOp::create(b, loc, TypeRange{}, ValueRange{}, "trap;",
                                    "", true, false, LLVM::TailCallKind::None,
                                    LLVM::AsmDialectAttr(), ArrayAttr());
        },
        [&] {
          // On replay the first block owns every atom and has checked both
          // limbs.
          Value first = select(retry, tid, addi(muli(owner, cx(workers)), tid));
          Value step = select(retry, cx(workers), stride);
          forLoop(first, cx(g.atoms), step, {}, [&](Value atom, ValueRange) {
            for (int a = 0; a < 3; ++a) {
              Value v = ifElse(
                  {i64}, retry,
                  [&] { return Values{load(words, {atom, cx(a)})}; },
                  [&] { return Values{planeWord(provisional, atom, a)}; })[0];
              if (term.getNumResults() == 0 && !keepForceWords) {
                Type t =
                    cast<MemRefType>(destination.getType()).getElementType();
                v = mulf(sitofp(t, v),
                         createReal(b, loc, t, 1.0 / forceWordScale));
                if (!term.overwrites(0))
                  v = addf(v, load(destination, {atom, cx(a)}));
              }
              store(v, destination, {atom, cx(a)});
            }
            return Values{};
          });
        });
  });
}

void Force::device(LLVM::GlobalOp recordsGlobal, TypeRange entryTypes,
                   bool singleBlock) {
  OpBuilder::InsertionGuard guard(b);
  tileParameters.clear();
  StringRef entryName =
      counting      ? "towers_count_stripe_retries"
      : observer()  ? (recovering ? (singleBlock ? "towers_energy_recover_once"
                                                 : "towers_energy_recover")
                                  : observerName())
      : singleBlock ? (movingDispatch ? "towers_finish" : "towers_recover_once")
      : recovering  ? "towers_recover"
      : fused()     ? "towers_force_virial"
                    : "towers_force";
  auto fn = gpu::GPUFuncOp::create(b, loc, entryName,
                                   b.getFunctionType(entryTypes, {}));
  fn.setKernel(true);
  fn.setKnownBlockSizeAttr(b.getDenseI32ArrayAttr({workers, 1, 1}));
  if (residency > 1)
    fn->setAttr(NVVM::NVVMDialect::getMinctasmAttrName(),
                b.getI32IntegerAttr(
                    !observer() && !recovering && !fused() && workers == 256 &&
                            fixedBytes + recordCap * 8 <= 32 * 1024
                        ? std::max(3, residency)
                        : residency));
  pos = shared(fn, {panel + stripe}, v4);
  id = shared(fn, {panel + stripe}, i32);
  totals = shared(fn, {2}, i32);
  int slots = panel + stripe;
  // Screen features and the compacted target list die at the prefix barrier.
  // Six force planes cover all four arrays, even when the stripe dominates.
  Value scratch =
      shared(fn, {(observer() ? 2 * slots + 2 * stripe : 6 * slots) * 4},
             b.getI8Type());
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
  features = view(0, {slots}, VectorType::get({2}, i32));
  bias = view(2 * slots, {stripe}, f32);
  keptSlots = view(2 * slots + stripe, {stripe}, i32);
  if (!observer()) {
    accum = view(0, {3, slots}, i32);
    coarse = view(3 * slots, {3, slots}, i32);
  }
  bind(entry.getArguments());
  if (counting)
    retryCounts = entry.getArgument(entryTypes.size() - 1);
  records = LLVM::AddressOfOp::create(b, loc, recordsGlobal);
  tid = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  lane = andi(indexCast(i32, tid), ci(31));
  if (packedChargeType) {
    scale = load(geometry, {cx(3)});
    filterLimit = load(geometry, {cx(4)});
    limit = load(geometry, {cx(5)});
  } else
    screenConstants(geometry);
  if (stagesGeometry())
    blockGeometry = shared(fn, {blockGeometryWords()}, f32);
  owner = gpu::BlockIdOp::create(b, loc, gpu::Dimension::x);
  if (probe && !recovering) {
    trace = shared(fn, {11}, i64);
    ifThen(cmpi(CmpIPredicate::eq, tid, cx(0)), [&] {
      for (int i = 0; i < 8; ++i)
        store(cl(0), trace, {cx(i)});
      store(cl(0), trace, {cx(10)});
      store(NVVM::GlobalTimerOp::create(b, loc, i64), trace, {cx(8)});
      store(NVVM::Clock64Op::create(b, loc, i64), trace, {cx(9)});
    });
  }
  if (certified && !recovering) {
    localVotes = memref::AllocaOp::create(
        b, loc, MemRefType::get({}, shortVote() ? i32 : i64));
    store(shortVote() ? ci(0) : cl(0), localVotes, {});
  }
  if (fused())
    fusedSum = entry.getArgument(types.size());
  // Begin with tower work to warm the packed positions. Insert the compact
  // exclusion grid before the last group of towers so both can finish together.
  if (exclusionRows > 1 && !singleBlock) {
    int64_t groups = llvm::divideCeil(exclusions, exclusionRows * workers);
    int64_t first = towerBlocks() - std::min(towerBlocks() / 2, 4 * groups);
    // Unsigned subtraction also puts blocks before first outside the interval.
    Value after = subi(owner, cx(first));
    Value shifted = select(cmpi(CmpIPredicate::ult, after, cx(groups)),
                           addi(cx(towerBlocks()), after),
                           subi(owner, cx(groups)));
    Value middle =
        cmpi(CmpIPredicate::ult, after, cx(towerBlocks() + groups - first));
    owner = select(middle, shifted, owner);
  }
  if (observer() || fused())
    startObservation();
  auto evaluate = [&] {
    auto tower = [&] {
      stageParameters(fn);
      traversal(owner);
    };
    if (!exclusions) {
      tower();
      return;
    }
    auto globalMapping = mapping;
    ifThenElse(cmpi(CmpIPredicate::ult, owner, cx(towerBlocks())), tower, [&] {
      mapping = globalMapping;
      int width = exclusionRows * workers;
      Value first = muli(subi(owner, cx(towerBlocks())), cx(width));
      Value row = addi(first, tid);
      if (exclusionRows == 1) {
        ifThen(cmpi(CmpIPredicate::ult, row, cx(exclusions)),
               [&] { exclusion(row); });
      } else {
        Value end = minui(addi(first, cx(width)), cx(exclusions));
        forLoop(row, end, cx(workers), {}, [&](Value i, ValueRange) {
          exclusion(i);
          return Values{};
        });
      }
    });
  };
  if (recovering) {
    bool finish = movingDispatch && singleBlock && !observer();
    if (!observer())
      highWords = entry.getArgument(entryTypes.size() - (finish ? 4 : 1));
    Value replay, retry, provisional, destination, stride;
    if (finish) {
      provisional = entry.getArgument(entryTypes.size() - 3);
      Value arrivals = entry.getArgument(entryTypes.size() - 2);
      destination = entry.getArgument(entryTypes.size() - 1);
      Value grid = gpu::GridDimOp::create(b, loc, gpu::Dimension::x);
      stride = muli(grid, cx(workers));
      retry = certifyGrid(provisional, arrivals, grid, stride);
      replay = andi(retry, cmpi(CmpIPredicate::eq, owner, cx(0)));
    } else
      replay = cmpi(CmpIPredicate::ne, load(invalid, {cx(Replay)}), ci(0));
    ifThen(replay, [&] {
      if (!singleBlock) {
        evaluate();
        return;
      }
      // One CTA owns the whole fallback. Its barriers order clearing,
      // traversal and range checks without a grid-wide rendezvous.
      auto atoms = [&](auto visit) {
        forLoop(tid, cx(g.atoms), cx(workers), {}, [&](Value i, ValueRange) {
          visit(i);
          return Values{};
        });
      };
      if (energy()) {
        ifThen(cmpi(CmpIPredicate::eq, tid, cx(0)), [&] {
          store(cl(0), words, {cx(0)});
          store(cl(0), words, {cx(1)});
        });
      } else
        atoms([&](Value i) {
          for (int axis = 0; axis < 3; ++axis) {
            store(cl(0), words, {i, cx(axis)});
            store(cl(0), highWords, {i, cx(axis)});
          }
        });
      gpu::BarrierOp::create(b, loc);
      auto globalMapping = mapping;
      stageParameters(fn);
      forLoop(cx(0), cx(towerBlocks()), cx(1), {}, [&](Value cell, ValueRange) {
        traversal(cell);
        gpu::BarrierOp::create(b, loc);
        return Values{};
      });
      if (exclusions) {
        mapping = globalMapping;
        forLoop(tid, cx(exclusions), cx(workers), {},
                [&](Value row, ValueRange) {
                  exclusion(row);
                  return Values{};
                });
        gpu::BarrierOp::create(b, loc);
      }
      if (energy()) {
        flushObservation();
        gpu::BarrierOp::create(b, loc);
        ifThen(andi(cmpi(CmpIPredicate::eq, tid, cx(0)),
                    cmpi(CmpIPredicate::ne, load(words, {cx(1)}),
                         shrsi(load(words, {cx(0)}), cl(63)))),
               [&] { flag(invalid, Fatal, InvalidTotal); });
      } else
        atoms([&](Value i) { checkWide(invalid, words, highWords, i); });
    });
    if (finish)
      publishGrid(provisional, destination, retry, stride);
  } else
    withValidGeometry(evaluate);
  if (certified && !recovering) {
    Value count = load(localVotes, {});
    if (shortVote()) {
      Value sum = gpu::SubgroupReduceOp::create(
          b, loc, count, gpu::AllReduceOperation::ADD, true);
      count = arith::ExtSIOp::create(
          b, loc, i64,
          select(cmpi(CmpIPredicate::eq, lane, ci(0)), sum, ci(0)));
    }
    ifThen(cmpi(CmpIPredicate::ne, count, cl(0)), [&] {
      atomic(LLVM::AtomicBinOp::add,
             pointerTo(baseAddress(failedConversions), cl(0)), count);
    });
  }
  if ((observer() || fused()) && !singleBlock)
    flushObservation();
  mark(7);
  if (probe && !recovering)
    ifThen(cmpi(CmpIPredicate::eq, tid, cx(0)), [&] {
      Value start = load(trace, {cx(8)});
      store(subi(NVVM::GlobalTimerOp::create(b, loc, i64), start), trace,
            {cx(8)});
      Value output = entry.getArgument(types.size() - 1);
      if (observer()) {
        ifThen(cmpi(CmpIPredicate::ult, owner, cx(towerBlocks())), [&] {
          store(start, output, {owner, cx(11)});
          store(arith::ExtUIOp::create(b, loc, i64,
                                       NVVM::SmIdOp::create(b, loc, i32)),
                output, {owner, cx(12)});
        });
      }
      for (int i = 0; i < 11; ++i)
        ifThen(cmpi(CmpIPredicate::ult, owner, cx(towerBlocks())),
               [&] { store(load(trace, {cx(i)}), output, {owner, cx(i)}); });
    });
  gpu::ReturnOp::create(b, loc);
}

LogicalResult Force::emit() {
  if ((split != 1 && split != 2) || panel < 32 || panel > 512 || panel % 32 ||
      workers < 32 || workers > 1024 || workers % 32 || stripe < 1 ||
      stripe > workers || stripe * panel >= (1 << 18) || recordCap < groups ||
      recordCap > groups * stripe || residency < 1 || residency > 4 ||
      workers * residency > 2048)
    return term.emitError("infeasible TOWERS panel, stripe, worker count, "
                          "record count or residency");
  double rc = term.getCutoff().convertToDouble();
  if (!(rc > 0) || !std::isnormal(static_cast<float>(rc * rc)))
    return term.emitError(
        "TOWERS requires a positive cutoff with a normal binary32 square");
  Block &body = term.getKernel().front();
  auto supported = [&](md_exec::PairForOp op) {
    if (op.getCenters() || op.getImages() != md_exec::Images::Minimum ||
        op.getKernel().front().getArgument(0).getType() != f32)
      return false;
    auto yield = op.getKernel().front().getTerminator();
    unsigned i = op == term ? yieldIndex : 0;
    if (i >= yield->getNumOperands())
      return false;
    Type t = yield->getOperand(i).getType();
    if (observer() && output != TowersOutput::Virial) {
      if (i < op.getOuts().size() || op.getExchange(i) != md::Exchange::Symmetric ||
          !op.getWeights() || (*op.getWeights())[i - op.getOuts().size()] != 0.5)
        return false;
      return energy() ? t == f32 || t == f64 : t == VectorType::get({output == TowersOutput::TensorVirial ? 9 : 3}, f64);
    }
    return i < op.getOuts().size() && op.getExchange(i) == md::Exchange::Antisymmetric && t == v3;
  };
  if (!supported(term) || (observer() && !certified))
    return term.emitError("TOWERS requires a mixed antisymmetric force or "
                          "checked symmetric energy region");
  for (auto region : {term, correction}) {
    if (!region)
      continue;
    for (Operation &op : region.getKernel().front().without_terminator())
      if (op.walk([](Operation *nested) {
              return isa<memref::LoadOp, vector::LoadOp>(nested) ||
                             isMemoryEffectFree(nested) ||
                             nested->hasTrait<
                                 OpTrait::HasRecursiveMemoryEffects>()
                         ? WalkResult::advance()
                         : WalkResult::interrupt();
            }).wasInterrupted())
        return region.emitError("TOWERS requires a read-only pair region");
  }
  auto mem = [&](ArrayRef<int64_t> shape, Type t) {
    return MemRefType::get(shape, t);
  };
  int64_t slots = g.slots();
  types = {mem({g.cells(), 5 + 2 * g.reach, 2}, i32),
           mem({slots}, v4),
           mem({slots}, i32),
           packedChargeType && !g.host ? mem({g.packedGeometrySize()}, f32)
                                       : mem({3, g.rawColumns()}, f64),
           observer() ? mem({energy() ? 2 : virialComponents()}, energy() ? i64 : f64)
                      : mem({g.atoms, 3}, i64),
           mem({certified ? 2 : 1}, i32),
           f32};
  if (certified)
    types.push_back(mem({1}, i64));
  if (exclusions) {
    types.append(
        {mem({exclusions, 2}, i32), mem({g.inverseWords()}, i32),
         cartesianPositions ? cartesianPositions : mem({3, g.atoms}, f64),
         mem({g.boxSize()}, f64)});
    if (indexedExclusions)
      types.push_back(mem({g.atoms, 2}, i32));
    if (correction && (!supported(correction) ||
        !llvm::equal(correction.getIns(), term.getIns()) ||
        correction.getKernel().front().getArgumentTypes() !=
            body.getArgumentTypes()))
      return correction.emitError(
          "expected matching direct and exclusion force signatures");
  }
  if (packedChargeType &&
      (term.getIns().size() != 2 || body.getArgument(2).getType() != f32 ||
       body.getArgument(4).getType() != i32))
    return term.emitError("packed-charge-type requires charge (f32) and type "
                          "(i32) inputs in that order");
  fixedBytes = (panel + stripe) * 44 + 16 +
               (stagesGeometry() ? 4 * blockGeometryWords() : 0);
  for (unsigned j = 0; j < term.getIns().size(); ++j) {
    Type t = body.getArgument(2 + 2 * j).getType();
    if (typeParameter < 0 && t == i32)
      typeParameter = j;
    auto scalar = getElementTypeOrSelf(t);
    if (!scalar.isIntOrFloat())
      return term.emitError("unsupported TOWERS particle input type");
    auto vec = dyn_cast<VectorType>(t);
    if (positionParameter < 0 && t == f32)
      positionParameter = j;
    else
      fixedBytes += (panel + stripe) * (vec ? vec.getNumElements() : 1) *
                    ((scalar.getIntOrFloatBitWidth() + 7) / 8);
    if (!packedChargeType || j != 0)
      types.push_back(movingDispatch
                          ? term.getIns()[j].getType()
                          : mem({packedChargeType ? slots : g.atoms}, t));
  }
  if (movingDispatch)
    types.push_back(mem({slots}, i32));
  if (!g.host && fixedBytes > 48 * 1024)
    return term.emitError("TOWERS force tile exceeds static workgroup storage");
  getUsedValuesDefinedAbove(term.getKernel(), term.getKernel(), captures);
  if (correction)
    getUsedValuesDefinedAbove(correction.getKernel(), correction.getKernel(),
                              captures);
  for (Value v : captures)
    if (!matchPattern(v, m_Constant()))
      types.push_back(v.getType());
  if (probe) {
    if (g.host)
      return term.emitError("TOWERS force phase counters require a GPU");
    types.push_back(mem({towerBlocks(), observer() ? 13 : 11}, i64));
  }
  if (observer())
    observationEntries();
  else if (certified && !movingDispatch)
    recoveryEntries();
  if (g.host) {
    auto fn = func::FuncOp::create(b, loc,
                                   observer() ? observerName() : "towers_force",
                                   b.getFunctionType(types, {}));
    Block *entry = fn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    bind(entry->getArguments());
    if (observer())
      startObservation();
    withValidGeometry([&] { host(); });
    if (observer())
      flushObservation();
    func::ReturnOp::create(b, loc);
  } else {
    if (packedChargeType && !observer())
      prepareGeometry();
    auto dyn = b.getInsertionBlock()
                   ->getParentOp()
                   ->getRegion(0)
                   .getOps<LLVM::GlobalOp>();
    LLVM::GlobalOp recordsGlobal;
    for (auto var : dyn)
      if (var.getSymName() == "towers_records")
        recordsGlobal = var;
    if (!recordsGlobal)
      recordsGlobal = LLVM::GlobalOp::create(
          b, loc, LLVM::LLVMArrayType::get(i32, 0), false,
          LLVM::Linkage::External, "towers_records", Attribute(), 8, 3);
    if (fused()) {
      if (!movingDispatch) {
        int saved = virialYield;
        virialYield = -1;
        device(recordsGlobal, types);
        virialYield = saved;
      }
      auto fusedTypes = types;
      fusedTypes.push_back(mem({virialComponents()}, f64));
      device(recordsGlobal, fusedTypes);
    } else
      device(recordsGlobal, types);
    if (emitRetryCounter && recordCap < groups * stripe) {
      auto diagnosticTypes = types;
      if (fused())
        diagnosticTypes.push_back(mem({virialComponents()}, f64));
      diagnosticTypes.push_back(mem({towerBlocks()}, i64));
      counting = true;
      device(recordsGlobal, diagnosticTypes);
      counting = false;
    }
    if (energy()) {
      recovering = true;
      auto replay = types;
      if (probe)
        replay.pop_back();
      device(recordsGlobal, replay);
      device(recordsGlobal, replay, true);
    } else if (!observer() && certified) {
      SmallVector<Type> replay(types);
      if (probe)
        replay.pop_back();
      replay.push_back(types[4]);
      recovering = true;
      if (!movingDispatch)
        device(recordsGlobal, replay);
      if (movingDispatch) {
        replay.push_back(types[4]);
        replay.push_back(mem({1}, i32));
        replay.push_back(term.getNumResults() == 0 && !keepForceWords
                             ? term.getOuts().front().getType()
                             : types[4]);
      }
      device(recordsGlobal, replay, true);
      recovering = false;
    }
  }
  return success();
}
} // namespace

LogicalResult mdir::kernels::emitTowersForce(ModuleOp module, TowersLayout g,
                                             md_exec::PairForOp term,
                                             TowersForceShape shape, bool probe,
                                             md_exec::PairForOp correction) {
  OpBuilder b = OpBuilder::atBlockEnd(module.getBody());
  if (!g.host)
    b.setInsertionPointToStart(
        module.lookupSymbol<gpu::GPUModuleOp>("towers").getBody());
  return Force(b, g, term, shape, probe, correction).emit();
}
