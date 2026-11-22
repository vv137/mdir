// Per-frame PME counting sort. The consumer interface is the cell offsets,
// slot-to-atom permutation and aligned (mesh x,y,z, charge) records.
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

void DevicePME::forParticles(OpBuilder &b, Location loc, Value count, Value blocks,
                             Value threads, Value one,
                             function_ref<void(Value)> body) {
  auto launch = gpu::LaunchOp::create(b, loc, blocks, one, one, threads, one, one);
  OpBuilder::InsertionGuard scope(b);
  b.setInsertionPointToStart(&launch.getBody().front());
  Value first = arith::MulIOp::create(b, loc, launch.getBlockIds().x, threads);
  Value i = arith::AddIOp::create(b, loc, first, launch.getThreadIds().x);
  Value inside = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, i, count);
  auto active = scf::IfOp::create(b, loc, inside, false);
  b.setInsertionPointToStart(active.thenBlock());
  body(i);
  b.setInsertionPointAfter(active);
  gpu::TerminatorOp::create(b, loc);
}

template <bool tilted> void DevicePME::emitCellOrder(StringRef symbol) {
  OpBuilder::InsertionGuard scope(b);
  auto words = globalMem({ShapedType::kDynamic}, i32Type);
  SmallVector<Type> types{globalMem({ShapedType::kDynamic, 3}, positionType),
                          globalMem({ShapedType::kDynamic}, chargeType),
                          VectorType::get({tilted ? 6 : 3}, f64Type), records()};
  types.append(6, words);
  types.append(3, idxType);
  auto fn = func::FuncOp::create(b, loc, symbol, b.getFunctionType(types, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  auto a = entry->getArguments();
  Value x = a[0], q = a[1], box = a[2], raw = a[3];
  Value keys = a[4], perm = a[5], counts = a[6], offsets = a[7], cursor = a[8];
  Value sums = a[9];
  auto ix = [&](int64_t n) -> Value {
    return arith::ConstantIndexOp::create(b, loc, n);
  };
  Value zero = ix(0), one = ix(1), four = ix(4), threads = ix(128);
  Value n = memref::DimOp::create(b, loc, x, zero);
  auto blocks = [&](Value count, Value width) {
    return maxui(one, arith::CeilDivUIOp::create(b, loc, count, width));
  };
  Value grid = blocks(n, threads);
  Value unit = arith::ConstantOp::create(b, loc, b.getF64FloatAttr(1));
  Value lx = extract(box, 0), ly = extract(box, 1), lz = extract(box, 2);
  Value inverse[] = {divf(unit, lx), divf(unit, ly), divf(unit, lz)};
  Value h00, h10, h11, h20, h21, h22;
  if constexpr (tilted)
    std::tie(h00, h10, h11, h20, h21, h22) = inverseCell(box, lx, ly, lz);
  Value cells = memref::DimOp::create(b, loc, counts, zero);
  Value scanBlocks = blocks(cells, ix(1024));
  auto sort = md_alg::SortByKeyOp::create(
      b, loc, n, cells, md_alg::Order::Unordered, keys, counts, offsets, cursor,
      perm, Value{}, ValueRange{raw}, ValueRange{}, false, false,
      ValueRange{grid, threads, one},
      md_alg::ScanStrategyAttr::get(b.getContext(), md_alg::ScanStrategy::ReduceThenScan),
      sums, ValueRange{scanBlocks, ix(256), one}, md_alg::SortStrategyAttr{},
      Value{}, ValueRange{}, nullptr);
  {
    OpBuilder::InsertionGuard keyScope(b);
    Block *key = b.createBlock(&sort.getKey());
    Value i = key->addArgument(idxType, loc);
    Value xyz[3];
    for (int axis = 0; axis < 3; ++axis)
      xyz[axis] = convert(f64Type, load(x, {i, ix(axis)}));
    if constexpr (tilted) {
      auto f = fractionalPosition(xyz[0], xyz[1], xyz[2], h00, h10, h11, h20, h21, h22);
      xyz[0] = f.x;
      xyz[1] = f.y;
      xyz[2] = f.z;
    } else {
      for (int axis = 0; axis < 3; ++axis)
        xyz[axis] = mulf(xyz[axis], inverse[axis]);
    }
    SmallVector<Value> components;
    for (int axis = 0; axis < 3; ++axis)
      components.push_back(meshCoordinate(xyz[axis], a[10 + axis]));
    components.push_back(convert(realType, load(q, {i})));
    for (int axis = 0; axis != 4; ++axis)
      store(components[axis], raw, {i, ix(axis)});
    Value cell = zero;
    for (int axis = 0; axis < 3; ++axis) {
      Value k = a[10 + axis];
      Value start = splitMeshAxis(components[axis], k, four).first;
      cell = addi(muli(cell, blocks(k, four)), divui(start, four));
    }
    md_alg::BinOp::create(b, loc, cell);
  }
  func::ReturnOp::create(b, loc);
}

void DevicePME::emitPack(StringRef symbol) {
  OpBuilder::InsertionGuard scope(b);
  SmallVector<Type> types{records(), globalMem({ShapedType::kDynamic}, i32Type),
                          records()};
  auto fn = func::FuncOp::create(b, loc, symbol, b.getFunctionType(types, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  auto a = entry->getArguments();
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value threads = arith::ConstantIndexOp::create(b, loc, 128);
  Value n = memref::DimOp::create(b, loc, a[0], zero);
  Value grid = maxui(one, arith::CeilDivUIOp::create(b, loc, n, threads));
  forParticles(b, loc, n, grid, threads, one, [&](Value i) {
    Value atom = indexCast(idxType, load(a[1], {i}));
    writeRecord(readRecord(a[0], atom), a[2], i);
  });
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitCellOrder<false>(StringRef);
template void DevicePME::emitCellOrder<true>(StringRef);
