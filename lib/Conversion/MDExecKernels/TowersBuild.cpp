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
  void sortWarp(Value keys, Value base, Value lane, Value count, int batches,
                function_ref<void(Value, Value)> place);
  Value wrap(Value position, Value length);
  Value quarterCell(Value v, Value width, Value reciprocal);
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

void TowersBuild::sortWarp(Value keys, Value base, Value lane, Value count,
                           int batches,
                           function_ref<void(Value, Value)> place) {
  // A bitonic network sorts one, two or four registers per lane. The
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

void TowersBuild::emit(bool audit) {
  if (!g.host)
    LLVM::LLVMFuncOp::create(b, loc, "__nv_fmod",
                             LLVM::LLVMFunctionType::get(f64, {f64, f64}));
  int cells = g.cells(), fields = offsets.size(), n = g.atoms;
  int slots = g.capacity * cells;
  auto counts = mem({cells}, i32), keys = mem({slots}, i32);
  auto status = mem({1}, i32), inverse = mem({cells, fields}, i32);
  auto spans = mem({cells, fields, 2}, i32);
  auto xyzw = VectorType::get({4}, f32);
  auto raw = mem({3, n}, f64), local = mem({n}, xyzw);
  auto packed = mem({slots}, xyzw), types = mem({n}, i32);
  auto charges = mem({n}, f32), box = mem({3}, f64),
       geometry = mem({3, 3}, f64);

  phase("towers_setup", {inverse, counts, status, box, geometry}, cells,
        [&](Value cell, ValueRange a) {
          store(ci(0), a[1], {cell});
          ifThen(cmpi(CmpIPredicate::eq, cell, cx(0)), [&] {
            store(ci(0), a[2], {cx(0)});
            for (auto [axis, bins] :
                 llvm::enumerate(std::array{g.nx, g.ny, g.nz})) {
              Value length = load(a[3], {cx(axis)});
              Value width = divf(length, cd(bins));
              store(length, a[4], {cx(axis), cx(0)});
              store(width, a[4], {cx(axis), cx(1)});
              store(divf(cd(4), width), a[4], {cx(axis), cx(2)});
            }
          });
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
  phase("towers_insert", {raw, charges, geometry, counts, keys, local, i32}, n,
        [&](Value item, ValueRange a) {
          Value atom = select(cmpi(CmpIPredicate::ne, a[6], ci(0)),
                              subi(cx(n - 1), item), item);
          Value cell = cx(0), sub = ci(0);
          Values coordinates;
          for (auto [axis, bins] :
               llvm::enumerate(std::array{g.nx, g.ny, g.nz})) {
            Value length = load(a[2], {cx(axis), cx(0)});
            Value x = load(a[0], {cx(axis), atom});
            Value v = wrap(x, length);
            Value width = load(a[2], {cx(axis), cx(1)});
            Value reciprocal = load(a[2], {cx(axis), cx(2)});
            Value quarter = quarterCell(v, width, reciprocal);
            quarter = minsi(maxsi(quarter, ci(0)), ci(4 * bins - 1));
            Value bin = shrui(quarter, ci(2));
            cell = addi(muli(cell, cx(bins)), indexCast(index, bin));
            sub = addi(shli(sub, ci(2)), andi(quarter, ci(3)));
            Value center = mulf(addf(sitofp(f64, bin), cd(0.5)), width);
            coordinates.push_back(truncf(subf(v, center)));
          }
          coordinates.push_back(load(a[1], {atom}));
          store(vector::FromElementsOp::create(b, loc, xyzw, coordinates), a[5],
                {atom});
          Value rank;
          if (g.host) {
            rank = load(a[3], {cell});
            store(addi(rank, ci(1)), a[3], {cell});
          } else {
            // The mask excludes inactive lanes in the last particle warp.
            Value lane = indexCast(i32, remui(item, cx(32)));
            Value left = subi(ci(n), subi(indexCast(i32, item), lane));
            Value active = shrui(ci(-1), subi(ci(32), minui(left, ci(32))));
            Value group = NVVM::MatchSyncOp::create(b, loc, i32, active,
                                                    indexCast(i32, cell),
                                                    NVVM::MatchSyncKind::any);
            Value first = math::CountTrailingZerosOp::create(b, loc, group);
            Value base = ifElse(
                {i32}, cmpi(CmpIPredicate::eq, lane, first),
                [&] {
                  Value ptr = pointerTo(baseAddress(a[3]),
                                        muli(indexCast(i64, cell), cl(4)));
                  Value members = math::CtPopOp::create(b, loc, group);
                  return Values{atomic(LLVM::AtomicBinOp::add, ptr, members)};
                },
                [&] { return Values{ci(0)}; })[0];
            base = NVVM::ShflOp::create(b, loc, i32, group, base, first, ci(31),
                                        NVVM::ShflKind::idx, nullptr);
            Value preceding = andi(group, subi(shli(ci(1), lane), ci(1)));
            rank = addi(base, math::CtPopOp::create(b, loc, preceding));
          }
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
            store(load(a[2], {atom}), a[5], {dest});
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
                            [&] { sort(4); }, rankSort);
                      });
                });
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
