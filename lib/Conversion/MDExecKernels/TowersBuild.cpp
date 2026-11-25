// Rectangular Neutral Territory ownership [Shaw2005], D428.
// The geometry is fixed at setup. All positions and memberships are rebuilt.
#include "mdir/Conversion/TowersBuild.h"
#include "NeighborBuildCode.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Conversion/Passes.h"
#include "mdir/Conversion/TowersForce.h"
#include "mdir/Dialect/MDExec/MDExecOps.h"
#include <array>
#include <climits>

using namespace mlir;
using namespace mdir::kernels::detail;

namespace mdir {
#define GEN_PASS_DEF_ADDTOWERSBUILD
#include "mdir/Conversion/Passes.h.inc"
} // namespace mdir

namespace {
using mdir::kernels::TowersLayout;

// Separate entries let the evaluation reuse storage across steps and the
// timing harness measure precisely the phases being lowered.
class TowersBuild : public NeighborBuildCode {
public:
  TowersBuild(OpBuilder &b, Location loc, TowersLayout layout)
      : NeighborBuildCode(b, loc), g(layout) {
    offsets.push_back({0, 0, 0});
    for (int sign : {-1, 1})
      for (int dz = 1; dz <= g.reach; ++dz)
        offsets.push_back({0, 0, sign * dz});
    for (Offset lateral :
         {Offset{-1, 0, 0}, {-1, -1, 0}, {0, -1, 0}, {1, -1, 0}})
      offsets.push_back(lateral);
  }

  void emit(bool audit);
  void initializeCell(Value cell, Value map, Value counts);
  void geometry(Value box, Value status, Value info);
  void insert(Value item, Value atom, ValueRange xyzq, ValueRange binGeometry,
              ValueRange lattice, Value counts, Value keys, Value local);

private:
  using Offset = std::array<int, 3>;
  TowersLayout g;
  SmallVector<Offset> offsets;

  MemRefType mem(ArrayRef<int64_t> shape, Type type) {
    return MemRefType::get(shape, type);
  }
  void phase(StringRef name, TypeRange types, int64_t items,
             function_ref<void(Value, ValueRange)> body);
  Value neighbor(Value cell, Offset shift);
  void increment(Value mem, Value at);
  void sortWarp(Value keys, Value base, Value lane, Value count, int batches,
                function_ref<void(Value, Value)> place);
  Value wrap(Value position, Value length);
  Value quarterCell(Value v, Value width, Value reciprocal);
  Values locate(Value x, Value length, Value width, Value reciprocal, int bins);
  void emitOwnership(MemRefType spans, MemRefType ids);
};

void TowersBuild::phase(StringRef name, TypeRange types, int64_t items,
                        function_ref<void(Value, ValueRange)> body) {
  mdir::kernels::emitTowersEntry(b, loc, g.host, name, types, items, body);
}

Value TowersBuild::neighbor(Value cell, Offset shift) {
  Value result = cx(0);
  int stride = g.cells();
  for (auto [axis, bins] : llvm::enumerate(std::array{g.nx, g.ny, g.nz})) {
    stride /= bins;
    Value n = cx(bins);
    Value component = remui(divui(cell, cx(stride)), n);
    Value shifted = remui(addi(component, cx(bins + shift[axis])), n);
    result = addi(muli(result, n), shifted);
  }
  return result;
}

void TowersBuild::increment(Value mem, Value at) {
  memref::AtomicRMWOp::create(b, loc, arith::AtomicRMWKind::addi, ci(1), mem,
                              ValueRange{at});
}

Value TowersBuild::wrap(Value x, Value length) {
  Value negative = cmpf(CmpFPredicate::OLT, x, cd(0));
  Value above = cmpf(CmpFPredicate::OGE, x, length);
  Value single =
      select(negative, addf(x, length), select(above, subf(x, length), x));
  Value multiple = ori(cmpf(CmpFPredicate::OLT, x, subf(cd(0), length)),
                       cmpf(CmpFPredicate::OGE, x, addf(length, length)));
  return ifElse(
      {f64}, multiple,
      [&] {
        // The general remainder also covers windings too large to represent
        // the integer quotient in binary64.
        Value rem;
        if (g.host)
          rem = arith::RemFOp::create(b, loc, x, length);
        else
          rem = LLVM::CallOp::create(b, loc, TypeRange{f64}, "__nv_fmod",
                                     ValueRange{x, length})
                    .getResult();
        return Values{select(cmpf(CmpFPredicate::OLT, rem, cd(0)),
                             addf(rem, length), rem)};
      },
      [&] { return Values{single}; })[0];
}

Value TowersBuild::quarterCell(Value v, Value width, Value reciprocal) {
  // An FMA resolves points on either side of a rounded reciprocal product.
  Value estimate = fptosi(i32, mulf(v, reciprocal));
  Value residual = math::FmaOp::create(
      b, loc, subf(cd(0), sitofp(f64, estimate)), width, mulf(v, cd(4)));
  return select(cmpf(CmpFPredicate::OLT, residual, cd(0)),
                subi(estimate, ci(1)),
                select(cmpf(CmpFPredicate::OGE, residual, width),
                       addi(estimate, ci(1)), estimate));
}

Values TowersBuild::locate(Value x, Value length, Value width, Value reciprocal,
                           int bins) {
  auto precise = [&] {
    Value v = wrap(x, length);
    Value q = quarterCell(v, width, reciprocal);
    return Values{v, minsi(maxsi(q, ci(0)), ci(4 * bins - 1))};
  };
  if (g.host || bins > (1 << 20)) {
    return precise();
  }
  // Classify in the input image. Away from a quarter-cell face,
  // two f32 conversions and the product perturb the scaled
  // coordinate by less than eight unit roundoffs. The guard
  // also covers rounding of a single f64 box translation.
  Value q = mulf(truncf(x), truncf(reciprocal));
  Value nearest = math::RoundEvenOp::create(b, loc, q);
  Value gap = math::AbsFOp::create(b, loc, subf(q, nearest));
  Value guard =
      mulf(cf(0x1p-20f), arith::MaximumFOp::create(
                             b, loc, cf(1), math::AbsFOp::create(b, loc, q)));
  Value interior = andi(cmpf(CmpFPredicate::OGT, gap, guard),
                        andi(cmpf(CmpFPredicate::OGT, q, cf(-4 * bins)),
                             cmpf(CmpFPredicate::OLT, q, cf(8 * bins))));
  return ifElse(
      {f64, i32}, interior,
      [&] {
        Value k = fptosi(i32, math::FloorOp::create(b, loc, q));
        Value below = cmpi(CmpIPredicate::slt, k, ci(0));
        Value above = cmpi(CmpIPredicate::sge, k, ci(4 * bins));
        Value shift =
            select(below, length, select(above, subf(cd(0), length), cd(0)));
        Value delta =
            select(below, ci(4 * bins), select(above, ci(-4 * bins), ci(0)));
        return Values{addf(x, shift), addi(k, delta)};
      },
      precise);
}

void TowersBuild::sortWarp(Value keys, Value base, Value lane, Value count,
                           int batches,
                           function_ref<void(Value, Value)> place) {
  // A bitonic network sorts one, two, four or eight registers per lane. The
  // sentinel compares after every legal compound key.
  SmallVector<Value> values(batches);
  for (int j = 0; j < batches; ++j) {
    Value slot = addi(lane, cx(32 * j));
    values[j] = ifElse(
        {i32}, cmpi(CmpIPredicate::ult, slot, count),
        [&] { return Values{load(keys, {addi(base, slot)})}; },
        [&] { return Values{ci(-1)}; })[0];
  }
  for (int width = 2; width <= 32 * batches; width *= 2) {
    for (int stride = width / 2; stride; stride /= 2) {
      for (int j = 0; j < batches; ++j) {
        if (stride >= 32) {
          int partner = j ^ (stride / 32);
          if (partner < j)
            continue;
          Value lo = minui(values[j], values[partner]),
                hi = maxui(values[j], values[partner]);
          bool ascending = (32 * j & width) == 0;
          values[j] = ascending ? lo : hi;
          values[partner] = ascending ? hi : lo;
        } else {
          Value slot = addi(lane, cx(32 * j));
          Value flip = andi(
              xori(divui(slot, cx(width)), divui(slot, cx(stride))), cx(1));
          Value other =
              shuffle(gpu::ShuffleMode::XOR, values[j], ci(stride), ci(32))
                  .first;
          values[j] = select(cmpi(CmpIPredicate::eq, flip, cx(0)),
                             minui(values[j], other), maxui(values[j], other));
        }
      }
    }
  }
  for (int j = 0; j < batches; ++j) {
    Value slot = addi(lane, cx(32 * j));
    ifThen(cmpi(CmpIPredicate::ult, slot, count),
           [&] { place(values[j], slot); });
  }
}

void TowersBuild::geometry(Value box, Value status, Value info) {
  store(ci(0), status, {cx(0)});
  for (auto [axis, bins] : llvm::enumerate(std::array{g.nx, g.ny, g.nz})) {
    Value length = g.triclinic ? cd(1) : load(box, {cx(axis)});
    if (g.triclinic) {
      auto basis = [&](int row, int col) {
        return load(box, {cx(3 * row + col)});
      };
      // The cross product of the other two rows is normal to the
      // face. Its dot product with this row gives the slab height.
      Values cross;
      Value norm2 = cd(0);
      int next = (axis + 1) % 3, last = (axis + 2) % 3;
      for (int k = 0; k < 3; ++k) {
        int u = (k + 1) % 3, v = (k + 2) % 3;
        Value c = subf(mulf(basis(next, u), basis(last, v)),
                       mulf(basis(next, v), basis(last, u)));
        cross.push_back(c);
        norm2 = addf(norm2, mulf(c, c));
      }
      Value norm = sqrt(norm2), height = cd(0);
      for (int k = 0; k < 3; ++k) {
        Value h = basis(axis, k), normal = divf(cross[k], norm);
        height = addf(height, mulf(h, normal));
        store(h, info, {cx(axis), cx(TowersLayout::lattice + k)});
        store(divf(h, cd(bins)), info, {cx(axis), cx(TowersLayout::step + k)});
        store(normal, info, {cx(axis), cx(TowersLayout::normal + k)});
      }
      store(divf(height, cd(bins)), info, {cx(axis), cx(TowersLayout::slab)});
    }
    Value width = divf(length, cd(bins));
    store(length, info, {cx(axis), cx(0)});
    store(width, info, {cx(axis), cx(1)});
    store(divf(cd(4), width), info, {cx(axis), cx(2)});
  }
}

void TowersBuild::initializeCell(Value cell, Value map, Value counts) {
  store(ci(0), counts, {cell});
  if (g.growCells)
    store(ci(-1), counts, {addi(cx(g.cells()), cell)});
  for (auto [field, delta] : llvm::enumerate(offsets)) {
    Offset back{-delta[0], -delta[1], -delta[2]};
    store(indexCast(i32, neighbor(cell, back)), map, {cell, cx(field)});
  }
}

void TowersBuild::insert(Value item, Value atom, ValueRange xyzq,
                         ValueRange binGeometry, ValueRange lattice,
                         Value counts, Value keys, Value local) {
  auto xyzw = VectorType::get({4}, f32);
  Value cell = cx(0), sub = ci(0);
  Values coordinates;
  for (auto [axis, bins] : llvm::enumerate(std::array{g.nx, g.ny, g.nz})) {
    Value length = binGeometry[3 * axis];
    Value x = xyzq[axis];
    Value width = binGeometry[3 * axis + 1];
    Value reciprocal = binGeometry[3 * axis + 2];
    Values located = locate(x, length, width, reciprocal, bins);
    Value v = located[0], quarter = located[1];
    Value bin = shrui(quarter, ci(2));
    cell = addi(muli(cell, cx(bins)), indexCast(index, bin));
    sub = addi(shli(sub, ci(2)), andi(quarter, ci(3)));
    Value center = mulf(addf(sitofp(f64, bin), cd(0.5)), width);
    Value local = subf(v, center);
    coordinates.push_back(g.triclinic ? local : truncf(local));
  }
  if (g.triclinic) {
    Values cartesian;
    for (int k = 0; k < 3; ++k) {
      auto component = [&](int axis) {
        return mulf(coordinates[axis], lattice[3 * axis + k]);
      };
      cartesian.push_back(
          truncf(addf(addf(component(0), component(1)), component(2))));
    }
    coordinates = std::move(cartesian);
  }
  coordinates.push_back(xyzq[3]);
  store(vector::FromElementsOp::create(b, loc, xyzw, coordinates), local,
        {atom});
  Value rank;
  if (g.host) {
    rank = load(counts, {cell});
    store(addi(rank, ci(1)), counts, {cell});
  } else {
    // The mask excludes inactive lanes in the last particle warp.
    Value lane = indexCast(i32, remui(item, cx(32)));
    Value left = subi(ci(g.atoms), subi(indexCast(i32, item), lane));
    Value active = shrui(ci(-1), subi(ci(32), minui(left, ci(32))));
    Value group = NVVM::MatchSyncOp::create(
        b, loc, i32, active, indexCast(i32, cell), NVVM::MatchSyncKind::any);
    Value first = math::CountTrailingZerosOp::create(b, loc, group);
    Value base = ifElse(
        {i32}, cmpi(CmpIPredicate::eq, lane, first),
        [&] {
          Value ptr =
              pointerTo(baseAddress(counts), muli(indexCast(i64, cell), cl(4)));
          Value members = math::CtPopOp::create(b, loc, group);
          return Values{atomic(LLVM::AtomicBinOp::add, ptr, members)};
        },
        [&] { return Values{ci(0)}; })[0];
    base = NVVM::ShflOp::create(b, loc, i32, group, base, first, ci(31),
                                NVVM::ShflKind::idx, nullptr);
    Value preceding = andi(group, subi(shli(ci(1), lane), ci(1)));
    rank = addi(base, math::CtPopOp::create(b, loc, preceding));
  }
  Value key = ori(shli(sub, ci(26)), indexCast(i32, atom));
  if (g.growCells) {
    ifThen(cmpi(CmpIPredicate::eq, item, cx(0)), [&] {
      store(ci(int64_t(g.cells()) * g.capacity), counts, {cx(2LL * g.cells())});
    });
    // Each unplaced key belongs to one list. Both the key and its
    // local coordinates come from this insertion, before any motion.
    ifThen(cmpi(CmpIPredicate::uge, rank, ci(g.capacity)), [&] {
      Value head = addi(cx(g.cells()), cell);
      Value previous = memref::AtomicRMWOp::create(
          b, loc, arith::AtomicRMWKind::assign, key, counts, ValueRange{head});
      store(previous, keys, {addi(cx(g.slots()), atom)});
    });
  }
  ifThen(cmpi(CmpIPredicate::ult, rank, ci(g.capacity)), [&] {
    store(key, keys,
          {addi(muli(cell, cx(g.capacity)), indexCast(index, rank))});
  });
}

void TowersBuild::emit(bool audit) {
  if (!g.host)
    LLVM::LLVMFuncOp::create(b, loc, "__nv_fmod",
                             LLVM::LLVMFunctionType::get(f64, {f64, f64}));
  int cells = g.cells(), fields = offsets.size(), n = g.atoms;
  auto counts = mem({g.countWords()}, i32), keys = mem({g.keyWords()}, i32);
  auto slotIds = mem({g.slots()}, i32);
  auto atomInverse = mem({g.inverseWords()}, i32);
  auto status = mem({1}, i32), inverse = mem({cells, fields}, i32);
  auto spans = mem({cells, fields, 2}, i32);
  auto xyzw = VectorType::get({4}, f32);
  auto raw = mem({3, n}, f64), local = mem({n}, xyzw);
  auto packed = mem({g.slots()}, xyzw), types = mem({n}, i32);
  auto charges = mem({n}, f32), box = mem({g.boxSize()}, f64),
       geometry = mem({3, g.rawColumns()}, f64);

  phase("towers_setup", {inverse, counts, status, box, geometry}, cells,
        [&](Value cell, ValueRange a) {
          initializeCell(cell, a[0], a[1]);
          ifThen(cmpi(CmpIPredicate::eq, cell, cx(0)),
                 [&] { TowersBuild::geometry(a[3], a[2], a[4]); });
        });
  if (!g.compact)
    phase("towers_clear", {counts}, cells,
          [&](Value c, ValueRange a) { store(ci(0), a[0], {c}); });

  // Keys are unique even when all atoms lie on the same quarter-cell tie.
  phase("towers_insert", {raw, charges, geometry, counts, keys, local, i32}, n,
        [&](Value item, ValueRange a) {
          Value atom = select(cmpi(CmpIPredicate::ne, a[6], ci(0)),
                              subi(cx(n - 1), item), item);
          Values xyzq, binGeometry, lattice;
          for (int axis = 0; axis < 3; ++axis) {
            xyzq.push_back(load(a[0], {cx(axis), atom}));
            for (int k = 0; k < 3; ++k) {
              binGeometry.push_back(load(a[2], {cx(axis), cx(k)}));
              if (g.triclinic)
                lattice.push_back(
                    load(a[2], {cx(axis), cx(TowersLayout::lattice + k)}));
            }
          }
          xyzq.push_back(load(a[1], {atom}));
          insert(item, atom, xyzq, binGeometry, lattice, a[3], a[4], a[5]);
        });

  // One warp consumes each cell on the device. Each key has a unique
  // rank, so even a dense cell can be packed without a shared sort buffer.
  int lanes = g.host ? 1 : 32;
  phase(
      "towers_pack",
      {counts, keys, local, types, inverse, packed, slotIds, slotIds,
       atomInverse, spans, status},
      int64_t(cells) * lanes, [&](Value item, ValueRange a) {
        Value cell = divui(item, cx(lanes)), lane = remui(item, cx(lanes));
        Value size = load(a[0], {cell});
        Value count = indexCast(index, minui(size, ci(g.capacity)));
        Value base = muli(cell, cx(g.capacity));
        if (g.growCells) {
          // Only a dense cell takes space from the pool. The cells are
          // disjoint, so their total request cannot exceed the atom count.
          base = ifElse(
              {index}, cmpi(CmpIPredicate::ugt, size, ci(g.capacity)),
              [&] {
                Value first = ifElse(
                    {i32}, cmpi(CmpIPredicate::eq, lane, cx(0)),
                    [&] {
                      Value slot = memref::AtomicRMWOp::create(
                          b, loc, arith::AtomicRMWKind::addi, size, a[0],
                          ValueRange{cx(2LL * cells)});
                      Value head = load(a[0], {addi(cx(cells), cell)});
                      forLoop(cx(g.capacity), indexCast(index, size), cx(1),
                              ValueRange{head}, [&](Value i, ValueRange chain) {
                                Value key = chain[0];
                                Value atom =
                                    indexCast(index, andi(key, ci(0x3ffffff)));
                                store(key, a[1],
                                      {addi(indexCast(index, slot), i)});
                                return Values{
                                    load(a[1], {addi(cx(g.slots()), atom)})};
                              });
                      return Values{slot};
                    },
                    [&] { return Values{ci(0)}; })[0];
                if (!g.host)
                  first = shuffle(gpu::ShuffleMode::IDX, first, ci(0), ci(32))
                              .first;
                Value grown = indexCast(index, first);
                forLoop(lane, cx(g.capacity), cx(lanes), {},
                        [&](Value i, ValueRange) {
                          store(load(a[1], {addi(base, i)}), a[1],
                                {addi(grown, i)});
                          return Values{};
                        });
                if (!g.host)
                  warpSync(ci(-1));
                return Values{grown};
              },
              [&] { return Values{base}; })[0];
          count = indexCast(index, size);
        }
        auto place = [&](Value key, Value rank) {
          Value atom = indexCast(index, andi(key, ci(0x3ffffff)));
          Value dest = addi(base, rank);
          store(load(a[2], {atom}), a[5], {dest});
          store(load(a[3], {atom}), a[6], {dest});
          store(indexCast(i32, atom), a[7], {dest});
          store(indexCast(i32, dest), a[8], {atom});
          if (g.growCells)
            store(indexCast(i32, cell), a[8], {addi(cx(n), atom)});
        };
        auto rankSort = [&] {
          forLoop(lane, count, cx(lanes), {}, [&](Value j, ValueRange) {
            Value key = load(a[1], {addi(base, j)});
            Value rank = forLoop(
                cx(0), count, cx(1), ValueRange{cx(0)},
                [&](Value k, ValueRange acc) {
                  Value earlier = cmpi(CmpIPredicate::ult,
                                       load(a[1], {addi(base, k)}), key);
                  return Values{addi(acc[0], select(earlier, cx(1), cx(0)))};
                })[0];
            place(key, rank);
            return Values{};
          });
        };
        if (g.host)
          rankSort();
        else {
          auto sort = [&](int batches) {
            sortWarp(a[1], base, lane, count, batches, place);
          };
          ifThenElse(
              cmpi(CmpIPredicate::ule, count, cx(32)), [&] { sort(1); },
              [&] {
                ifThenElse(
                    cmpi(CmpIPredicate::ule, count, cx(64)), [&] { sort(2); },
                    [&] {
                      ifThenElse(
                          cmpi(CmpIPredicate::ule, count, cx(128)),
                          [&] { sort(4); }, [&] {
                            if (g.widePack)
                              ifThenElse(cmpi(CmpIPredicate::ule, count, cx(256)),
                                         [&] { sort(8); }, rankSort);
                            else
                              rankSort();
                          });
                    });
              });
        }
        forLoop(lane, cx(fields), cx(lanes), {}, [&](Value field, ValueRange) {
          Value owner = indexCast(index, load(a[4], {cell, field}));
          store(indexCast(i32, base), a[9], {owner, field, cx(0)});
          store(indexCast(i32, count), a[9], {owner, field, cx(1)});
          return Values{};
        });
        if (!g.host && g.compact)
          warpSync(ci(-1));
        ifThen(cmpi(CmpIPredicate::eq, lane, cx(0)), [&] {
          if (g.compact)
            store(ci(0), a[0], {cell});
          if (g.growCells) {
            if (g.compact)
              store(ci(-1), a[0], {addi(cx(cells), cell)});
            return;
          }
          ifThen(cmpi(CmpIPredicate::ugt, size, ci(g.capacity)), [&] {
            memref::AtomicRMWOp::create(b, loc, arith::AtomicRMWKind::maxu,
                                        size, a[10], ValueRange{cx(0)});
          });
        });
      });
  if (audit)
    emitOwnership(spans, slotIds);
}

// Enumerate candidate multiplicities by stable ID. The caller applies the
// physical cutoff independently. No dense pair table is used in a rebuild.
void TowersBuild::emitOwnership(MemRefType spans, MemRefType ids) {
  auto pairs = mem({int64_t(g.atoms) * g.atoms}, i32);
  phase(
      "towers_ownership", {spans, ids, pairs}, g.cells(),
      [&](Value owner, ValueRange a) {
        mdir::kernels::emitOwnedTowersSpans(
            b, loc, g.reach, owner, a[0],
            [&](auto source, auto target, Value triangular) {
              forLoop(cx(0), source.count, cx(1), {}, [&](Value i, ValueRange) {
                Value id =
                    indexCast(index, load(a[1], {addi(source.first, i)}));
                Value end = select(triangular, i, target.count);
                forLoop(cx(0), end, cx(1), {}, [&](Value j, ValueRange) {
                  Value other =
                      indexCast(index, load(a[1], {addi(target.first, j)}));
                  increment(a[2], addi(muli(minui(id, other), cx(g.atoms)),
                                       maxui(id, other)));
                  return Values{};
                });
                return Values{};
              });
            });
      });
}
} // namespace

void mdir::kernels::emitTowersGeometry(OpBuilder &b, Location loc,
                                       TowersLayout g, Value box, Value status,
                                       Value info) {
  TowersBuild(b, loc, g).geometry(box, status, info);
}

void mdir::kernels::emitTowersInitRow(OpBuilder &b, Location loc,
                                      TowersLayout g, Value cell, Value map,
                                      Value counts) {
  TowersBuild(b, loc, g).initializeCell(cell, map, counts);
}

void mdir::kernels::emitTowersInsertRow(OpBuilder &b, Location loc,
                                        TowersLayout g, Value atom,
                                        ValueRange xyzq, ValueRange binGeometry,
                                        ValueRange lattice, Value counts,
                                        Value keys, Value local) {
  TowersBuild(b, loc, g).insert(atom, atom, xyzq, binGeometry, lattice, counts,
                                keys, local);
}

SmallVector<Value> mdir::kernels::towersMovingBinGeometry(OpBuilder &b,
                                                          Location loc,
                                                          TowersLayout g,
                                                          Value cell) {
  auto dbl = b.getF64Type();
  SmallVector<Value> binsInfo;
  if (g.cells() <= g.atoms) {
    for (auto [a, bins] : llvm::enumerate(std::array{g.nx, g.ny, g.nz})) {
      Value length =
          g.triclinic ? createReal(b, loc, dbl, 1)
                      : vector::ExtractOp::create(b, loc, cell, a).getResult();
      Value width =
          arith::DivFOp::create(b, loc, length, createReal(b, loc, dbl, bins));
      binsInfo.push_back(length);
      binsInfo.push_back(width);
      binsInfo.push_back(
          arith::DivFOp::create(b, loc, createReal(b, loc, dbl, 4), width));
    }
  }
  return binsInfo;
}

void mdir::kernels::TowersMovingRow::emit(OpBuilder &k, Location loc, Value i,
                                          ValueRange binsInfo,
                                          ValueRange fields) const {
  Type dbl = k.getF64Type(), real = k.getF32Type(), narrow = k.getI32Type(),
       wide = k.getI64Type();
  auto axis = [&](int a) { return createIndex(k, loc, a); };
  if (initialWords) {
    for (int axisIndex = 0; axisIndex < 3; ++axisIndex)
      memref::StoreOp::create(k, loc, createZero(k, loc, wide), initialWords,
                              ValueRange{i, axis(axisIndex)});
    scf::IfOp::create(
        k, loc,
        arith::CmpIOp::create(k, loc, arith::CmpIPredicate::eq, i, axis(0)),
        [&](OpBuilder &at, Location l) {
          Value zero = createIndex(at, l, 0);
          for (auto [dest, slot] :
               {std::pair{initialState, 0}, std::pair{initialState, 1},
                std::pair{arrivals, 0}})
            memref::StoreOp::create(at, l, createZero(at, l, narrow), dest,
                                    ValueRange{createIndex(at, l, slot)});
          memref::StoreOp::create(at, l, createZero(at, l, wide), initialFailed,
                                  ValueRange{zero});
          scf::YieldOp::create(at, l);
        });
  }
  SmallVector<Value> xyz, lattice;
  for (int a = 0; a < 3; ++a)
    xyz.push_back(
        memref::LoadOp::create(k, loc, positions, ValueRange{i, axis(a)}));
  auto edge = [&](int a) {
    return vector::ExtractOp::create(k, loc, cell, a).getResult();
  };
  if (g.triclinic) {
    Value zero = createReal(k, loc, dbl, 0);
    lattice = {edge(0), zero,    zero,    edge(3), edge(1),
               zero,    edge(4), edge(5), edge(2)};
    for (int a = 2; a >= 0; --a) {
      for (int j = 2; j > a; --j)
        xyz[a] = arith::SubFOp::create(
            k, loc, xyz[a],
            arith::MulFOp::create(k, loc, xyz[j], lattice[3 * j + a]));
      xyz[a] = arith::DivFOp::create(k, loc, xyz[a], lattice[3 * a + a]);
    }
  } else
    lattice = {edge(0), edge(1), edge(2)};
  auto first =
      arith::CmpIOp::create(k, loc, arith::CmpIPredicate::eq, i, axis(0));
  scf::IfOp::create(k, loc, first, [&](OpBuilder &at, Location l) {
    for (auto [j, v] : llvm::enumerate(lattice))
      memref::StoreOp::create(at, l, v, box, ValueRange{createIndex(at, l, j)});
    if (g.cells() <= g.atoms)
      emitTowersGeometry(at, l, g, box, status, geometry);
    scf::YieldOp::create(at, l);
  });
  if (!cartesianPositions)
    for (int a = 0; a < 3; ++a)
      memref::StoreOp::create(k, loc, xyz[a], raw, ValueRange{axis(a), i});
  Value chargeValue, typeValue;
  for (Value from : fields) {
    Value v = memref::LoadOp::create(k, loc, from, ValueRange{i});
    if (!chargeValue && v.getType() == real)
      chargeValue = v;
    if (!typeValue && v.getType() == narrow)
      typeValue = v;
  }
  if (!chargeValue)
    chargeValue = createZero(k, loc, real);
  if (g.cells() > g.atoms)
    memref::StoreOp::create(k, loc, chargeValue, charge, ValueRange{i});
  memref::StoreOp::create(k, loc,
                          typeValue ? typeValue : createZero(k, loc, narrow),
                          types, ValueRange{i});
  if (g.cells() <= g.atoms) {
    xyz.push_back(chargeValue);
    emitTowersInsertRow(k, loc, g, i, xyz, binsInfo, lattice, counts, keys,
                        local);
  }
}

void mdir::kernels::emitTowersEntry(
    OpBuilder &b, Location loc, bool host, StringRef name, TypeRange types,
    int64_t items, function_ref<void(Value, ValueRange)> body) {
  OpBuilder::InsertionGuard guard(b);
  FunctionType type = b.getFunctionType(types, {});
  NeighborBuildCode ir(b, loc);
  if (host) {
    auto fn = func::FuncOp::create(b, loc, name, type);
    Block *entry = fn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    ir.forLoop(ir.cx(0), ir.cx(items), ir.cx(1), {}, [&](Value i, ValueRange) {
      body(i, entry->getArguments());
      return SmallVector<Value>{};
    });
    func::ReturnOp::create(b, loc);
    return;
  }
  auto fn = gpu::GPUFuncOp::create(b, loc, name, type);
  fn.setKernel(true);
  Block &entry = fn.getBody().front();
  b.setInsertionPointToStart(&entry);
  Value block = gpu::BlockIdOp::create(b, loc, gpu::Dimension::x);
  Value size = gpu::BlockDimOp::create(b, loc, gpu::Dimension::x);
  Value lane = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  Value item = ir.addi(ir.muli(block, size), lane);
  ir.ifThen(ir.cmpi(arith::CmpIPredicate::ult, item, ir.cx(items)),
            [&] { body(item, entry.getArguments()); });
  gpu::ReturnOp::create(b, loc);
}

void mdir::kernels::emitOwnedTowersSpans(
    OpBuilder &builder, Location loc, int reach, Value owner, Value descriptors,
    llvm::function_ref<void(TowersSpan, TowersSpan, Value)> visit) {
  NeighborBuildCode ir(builder, loc);
  Value zero = ir.cx(0), one = ir.cx(1);
  auto span = [&](Value field) {
    return TowersSpan{
        field,
        ir.indexCast(ir.index, ir.load(descriptors, {owner, field, zero})),
        ir.indexCast(ir.index, ir.load(descriptors, {owner, field, one}))};
  };
  int columns = 2 * reach + 1;
  ir.forLoop(zero, ir.cx(columns), one, {}, [&](Value s, ValueRange) {
    TowersSpan source = span(s);
    // Home targets take the preceding layers and the upper triangle of home.
    ir.ifThen(ir.cmpi(CmpIPredicate::ule, s, ir.cx(reach)), [&] {
      visit(source, span(zero), ir.cmpi(CmpIPredicate::eq, s, zero));
    });
    // Each lateral half-plate target takes the entire axial column.
    ir.forLoop(ir.cx(columns), ir.cx(columns + 4), one, {},
               [&](Value t, ValueRange) {
                 visit(source, span(t), ir.cb(false));
                 return Values{};
               });
    return Values{};
  });
}

LogicalResult mdir::kernels::emitTowersBuild(ModuleOp module, TowersLayout g,
                                             bool audit) {
  auto [atoms, nx, ny, nz, reach, capacity, host, compact, triclinic,
        growCells, widePack] = g;
  if (atoms < 1 || atoms >= (1 << 26) || (reach != 1 && reach != 2) || nx < 3 ||
      ny < 3 || nz < 2 * reach + 1 || capacity < 1 || nx > INT_MAX / 4 ||
      ny > INT_MAX / 4 || nz > INT_MAX / 4 ||
      INT_MAX / capacity / nx / ny < nz || g.slots() > INT_MAX) {
    return module.emitError("infeasible rectangular TOWERS layout");
  }
  OpBuilder builder = OpBuilder::atBlockEnd(module.getBody());
  if (!host) {
    auto device = gpu::GPUModuleOp::create(builder, module.getLoc(), "towers");
    builder.setInsertionPointToStart(device.getBody());
  }
  TowersBuild(builder, module.getLoc(), g).emit(audit);
  return success();
}

namespace {
class AddTowersBuild : public mdir::impl::AddTowersBuildBase<AddTowersBuild> {
public:
  using Base::Base;
  void runOnOperation() final {
    auto module = getOperation();
    TowersLayout layout{atoms,    nx,   ny,      nz,        reach,
                        capacity, host, compact, triclinic, growCells};
    if (failed(mdir::kernels::emitTowersBuild(module, layout, audit))) {
      signalPassFailure();
      return;
    }
    if (forceBody.empty()) {
      if (!energyBody.empty() || directVirial) {
        module.emitError("energy and virial entries require force-body");
        signalPassFailure();
      }
      return;
    }
    auto fn = module.lookupSymbol<func::FuncOp>(forceBody);
    SmallVector<mdir::md_exec::PairForOp> terms;
    if (fn)
      fn.walk([&](mdir::md_exec::PairForOp term) { terms.push_back(term); });
    if (exclusions < 0 || (exclusions && !certifyForce) ||
        terms.size() != (exclusions ? 2u : 1u)) {
      module.emitError(
          "TOWERS force-body requires one direct pair loop and, with "
          "exclusions, certified force and one correction pair loop");
      signalPassFailure();
      return;
    }
    using namespace mdir::kernels;
    TowersForceShape shape{forcePanel,   forceStripe,    forceWorkers,
                           forceRecords, forceResidency, packedChargeType,
                           certifyForce, exclusions,     forceSplit};
    auto emit = [&](ArrayRef<mdir::md_exec::PairForOp> regions,
                    TowersOutput output) {
      shape.output = output;
      return emitTowersForce(module, layout, regions.front(), shape, probeForce,
                             exclusions ? regions.back()
                                        : mdir::md_exec::PairForOp{});
    };
    if (!layout.host && directVirial && !exclusions &&
        terms.front().getNumResults() == 1) {
      auto tensor = dyn_cast<VectorType>(terms.front().getResult(0).getType());
      if (tensor && tensor.getElementType().isF64() &&
          (tensor.getNumElements() == 3 || tensor.getNumElements() == 9))
        shape.virialYield = 1;
    }
    if (failed(emit(terms, TowersOutput::Force)) ||
        (directVirial && failed(emit(terms, TowersOutput::Virial)))) {
      signalPassFailure();
      return;
    }
    if (!energyBody.empty()) {
      SmallVector<mdir::md_exec::PairForOp> energy;
      if (auto body = module.lookupSymbol<func::FuncOp>(energyBody))
        body.walk([&](mdir::md_exec::PairForOp op) { energy.push_back(op); });
      if (energy.size() != terms.size()) {
        module.emitError(
            "expected matching direct and correction energy regions");
        signalPassFailure();
      } else if (failed(emit(energy, TowersOutput::Energy)))
        signalPassFailure();
    }
  }
};
} // namespace
