// Rectangular Neutral Territory ownership [Shaw2005], D[towers].
// The geometry is fixed at setup. All positions and memberships are rebuilt.
#include "mdir/Conversion/TowersBuild.h"
#include "NeighborBuildCode.h"
#include "mdir/Conversion/Passes.h"
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
  void emitOwnership(MemRefType spans, MemRefType ids);
};

void TowersBuild::phase(StringRef name, TypeRange types, int64_t items,
                        function_ref<void(Value, ValueRange)> body) {
  OpBuilder::InsertionGuard guard(b);
  FunctionType type = b.getFunctionType(types, {});
  if (g.host) {
    auto fn = func::FuncOp::create(b, loc, name, type);
    Block *entry = fn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    forLoop(cx(0), cx(items), cx(1), {}, [&](Value i, ValueRange) {
      body(i, entry->getArguments());
      return Values{};
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
  Value item = addi(muli(block, size), lane);
  ifThen(cmpi(CmpIPredicate::ult, item, cx(items)),
         [&] { body(item, entry.getArguments()); });
  gpu::ReturnOp::create(b, loc);
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

void TowersBuild::emit(bool audit) {
  int cells = g.cells(), fields = offsets.size(), n = g.atoms;
  int slots = g.capacity * cells;
  auto counts = mem({cells}, i32), keys = mem({slots}, i32);
  auto status = mem({1}, i32), inverse = mem({cells, fields}, i32);
  auto spans = mem({cells, fields, 2}, i32);
  auto raw = mem({3, n}, f64), local = mem({n, 4}, f32);
  auto packed = mem({slots, 4}, f32), types = mem({n}, i32);
  auto charges = mem({n}, f32), box = mem({3}, f64);

  phase("towers_setup", {inverse, counts, status}, cells,
        [&](Value cell, ValueRange a) {
          store(ci(0), a[1], {cell});
          ifThen(cmpi(CmpIPredicate::eq, cell, cx(0)),
                 [&] { store(ci(0), a[2], {cx(0)}); });
          for (auto [field, delta] : llvm::enumerate(offsets)) {
            Offset back{-delta[0], -delta[1], -delta[2]};
            store(indexCast(i32, neighbor(cell, back)), a[0],
                  {cell, cx(field)});
          }
        });
  if (!g.compact)
    phase("towers_clear", {counts}, cells,
          [&](Value c, ValueRange a) { store(ci(0), a[0], {c}); });

  // Keys are unique even when all atoms lie on the same quarter-cell tie.
  phase("towers_insert", {raw, charges, box, counts, keys, local, i32}, n,
        [&](Value item, ValueRange a) {
          Value atom = select(cmpi(CmpIPredicate::ne, a[6], ci(0)),
                              subi(cx(n - 1), item), item);
          Value cell = cx(0), sub = ci(0);
          for (auto [axis, bins] :
               llvm::enumerate(std::array{g.nx, g.ny, g.nz})) {
            Value length = load(a[2], {cx(axis)});
            Value x = load(a[0], {cx(axis), atom});
            Value outside = ori(cmpf(CmpFPredicate::OLT, x, cd(0)),
                                cmpf(CmpFPredicate::OGE, x, length));
            Value v = ifElse(
                {f64}, outside,
                [&] {
                  return Values{subf(x, mulf(floor(divf(x, length)), length))};
                },
                [&] { return Values{x}; })[0];
            Value width = divf(length, cd(bins));
            Value quarter = fptosi(i32, mulf(cd(4), divf(v, width)));
            quarter = minsi(maxsi(quarter, ci(0)), ci(4 * bins - 1));
            Value bin = shrui(quarter, ci(2));
            cell = addi(muli(cell, cx(bins)), indexCast(index, bin));
            sub = addi(shli(sub, ci(2)), andi(quarter, ci(3)));
            Value center = mulf(addf(sitofp(f64, bin), cd(0.5)), width);
            store(truncf(subf(v, center)), a[5], {atom, cx(axis)});
          }
          store(load(a[1], {atom}), a[5], {atom, cx(3)});
          Value rank =
              memref::AtomicRMWOp::create(b, loc, arith::AtomicRMWKind::addi,
                                          ci(1), a[3], ValueRange{cell});
          ifThen(cmpi(CmpIPredicate::ult, rank, ci(g.capacity)), [&] {
            Value key = ori(shli(sub, ci(26)), indexCast(i32, atom));
            store(key, a[4],
                  {addi(muli(cell, cx(g.capacity)), indexCast(index, rank))});
          });
        });

  // One warp consumes each cell on the device. Each key has a unique
  // rank, so even a dense cell can be packed without a shared sort buffer.
  int lanes = g.host ? 1 : 32;
  phase("towers_pack",
        {counts, keys, local, types, inverse, packed, keys, keys, types, spans,
         status},
        int64_t(cells) * lanes, [&](Value item, ValueRange a) {
          Value cell = divui(item, cx(lanes)), lane = remui(item, cx(lanes));
          Value size = load(a[0], {cell});
          Value count = indexCast(index, minui(size, ci(g.capacity)));
          Value base = muli(cell, cx(g.capacity));
          auto place = [&](Value key, Value rank) {
            Value atom = indexCast(index, andi(key, ci(0x3ffffff)));
            Value dest = addi(base, rank);
            for (int k = 0; k < 4; ++k)
              store(load(a[2], {atom, cx(k)}), a[5], {dest, cx(k)});
            store(load(a[3], {atom}), a[6], {dest});
            store(indexCast(i32, atom), a[7], {dest});
            store(indexCast(i32, dest), a[8], {atom});
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
            // Most cells fit two registers per lane. Sort 64 keys in a warp;
            // absent keys compare last (the atom count excludes the sentinel).
            ifThenElse(
                cmpi(CmpIPredicate::ule, size, ci(64)),
                [&] {
                  Value key[2];
                  for (int j = 0; j < 2; ++j) {
                    Value slot = addi(lane, cx(32 * j));
                    key[j] = ifElse(
                        {i32}, cmpi(CmpIPredicate::ult, slot, count),
                        [&] { return Values{load(a[1], {addi(base, slot)})}; },
                        [&] { return Values{ci(-1)}; })[0];
                  }
                  for (int width = 2; width <= 64; width *= 2)
                    for (int stride = width / 2; stride; stride /= 2) {
                      if (stride == 32) {
                        Value lo = minui(key[0], key[1]),
                              hi = maxui(key[0], key[1]);
                        key[0] = lo;
                        key[1] = hi;
                        continue;
                      }
                      for (int j = 0; j < 2; ++j) {
                        Value slot = addi(lane, cx(32 * j));
                        Value flip = andi(xori(divui(slot, cx(width)),
                                               divui(slot, cx(stride))),
                                          cx(1));
                        Value other = shuffle(gpu::ShuffleMode::XOR, key[j],
                                              ci(stride), ci(32))
                                          .first;
                        key[j] =
                            select(cmpi(CmpIPredicate::eq, flip, cx(0)),
                                   minui(key[j], other), maxui(key[j], other));
                      }
                    }
                  for (int j = 0; j < 2; ++j) {
                    Value slot = addi(lane, cx(32 * j));
                    ifThen(cmpi(CmpIPredicate::ult, slot, count),
                           [&] { place(key[j], slot); });
                  }
                },
                rankSort);
          }
          forLoop(lane, cx(fields), cx(lanes), {},
                  [&](Value field, ValueRange) {
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
            ifThen(cmpi(CmpIPredicate::ugt, size, ci(g.capacity)), [&] {
              memref::AtomicRMWOp::create(b, loc, arith::AtomicRMWKind::maxu,
                                          size, a[10], ValueRange{cx(0)});
            });
          });
        });
  if (audit)
    emitOwnership(spans, keys);
}

// Enumerate candidate multiplicities by stable ID. The caller applies the
// physical cutoff independently. No dense pair table is used in a rebuild.
void TowersBuild::emitOwnership(MemRefType spans, MemRefType ids) {
  auto pairs = mem({int64_t(g.atoms) * g.atoms}, i32);
  phase("towers_ownership", {spans, ids, pairs}, g.cells(),
        [&](Value owner, ValueRange a) {
          int column = 2 * g.reach + 1;
          auto span = [&](Value field, int word) {
            return indexCast(index, load(a[0], {owner, field, cx(word)}));
          };
          forLoop(cx(0), cx(column), cx(1), {}, [&](Value s, ValueRange) {
            Value first = span(s, 0), n = span(s, 1);
            forLoop(cx(0), cx(5), cx(1), {}, [&](Value t, ValueRange) {
              Value self = cmpi(CmpIPredicate::eq, t, cx(0));
              Value field = select(self, cx(0), addi(t, cx(column - 1)));
              Value valid = ori(notNarrow(self),
                                cmpi(CmpIPredicate::ule, s, cx(g.reach)));
              ifThen(valid, [&] {
                Value target = span(field, 0), nt = span(field, 1);
                forLoop(cx(0), n, cx(1), {}, [&](Value i, ValueRange) {
                  Value same = andi(self, cmpi(CmpIPredicate::eq, s, cx(0)));
                  Value source = indexCast(index, load(a[1], {addi(first, i)}));
                  forLoop(cx(0), select(same, i, nt), cx(1), {},
                          [&](Value j, ValueRange) {
                            Value other =
                                indexCast(index, load(a[1], {addi(target, j)}));
                            increment(a[2], addi(muli(minui(source, other),
                                                      cx(g.atoms)),
                                                 maxui(source, other)));
                            return Values{};
                          });
                  return Values{};
                });
              });
              return Values{};
            });
            return Values{};
          });
        });
}
} // namespace

LogicalResult mdir::kernels::emitTowersBuild(ModuleOp module, TowersLayout g,
                                             bool audit) {
  auto [atoms, nx, ny, nz, reach, capacity, host, compact] = g;
  if (atoms < 1 || atoms >= (1 << 26) || (reach != 1 && reach != 2) || nx < 3 ||
      ny < 3 || nz < 2 * reach + 1 || capacity < 1 || nx > INT_MAX / 4 ||
      ny > INT_MAX / 4 || nz > INT_MAX / 4 ||
      INT_MAX / capacity / nx / ny < nz) {
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
    if (failed(mdir::kernels::emitTowersBuild(
            getOperation(), {atoms, nx, ny, nz, reach, capacity, host, compact},
            audit)))
      signalPassFailure();
  }
};
} // namespace
