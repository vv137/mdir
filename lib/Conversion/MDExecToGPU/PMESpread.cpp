// Direct charge spreading, with floating or fixed-point accumulation.
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <bool tilted, bool useList, bool fixedGrid>
void DevicePME::emitSpread(StringRef symbol) {
  StringRef blockFunction =
      tilted ? "mdrt_gpu_pme_blocks_triclinic" : "mdrt_gpu_pme_blocks";
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(globalMem({ShapedType::kDynamic, 3}, positionType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, chargeType));
  if constexpr (useList) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, i32Type));
  }
  if constexpr (!tilted) {
    inputs.push_back(VectorType::get({3}, f64Type));
  } else {
    inputs.push_back(VectorType::get({6}, f64Type));
  }
  if constexpr (fixedGrid) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, i64Type));
  } else {
    inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  }
  inputs.append(4, idxType);
  auto fn = func::FuncOp::create(b, loc, symbol, b.getFunctionType(inputs, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  unsigned arg = 0;
  Value x = entry->getArgument(arg++);
  Value q = entry->getArgument(arg++);
  Value list;
  if constexpr (useList) {
    list = entry->getArgument(arg++);
  }
  Value box = entry->getArgument(arg++);
  Value grid = entry->getArgument(arg++);
  Value k1 = entry->getArgument(arg++);
  Value k2 = entry->getArgument(arg++);
  Value k3 = entry->getArgument(arg++);
  ++arg;
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c128 = arith::ConstantIndexOp::create(b, loc, 128);
  Value lx = extract(box, 0);
  Value ly = extract(box, 1);
  Value lz = extract(box, 2);
  Value h00;
  Value h11;
  Value h22;
  Value h10;
  Value h21;
  Value h20;
  if constexpr (tilted) {
    std::tie(h00, h10, h11, h20, h21, h22) = inverseCell(box, lx, ly, lz);
    convert(realType, h00);
    convert(realType, h10);
    convert(realType, h11);
    convert(realType, h20);
    convert(realType, h21);
    convert(realType, h22);
  }
  Value unit = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
  Value ilx = divf(unit, lx);
  Value ily = divf(unit, ly);
  Value ilz = divf(unit, lz);
  Value points = memref::DimOp::create(b, loc, grid, c0);
  Value count = memref::DimOp::create(b, loc, x, c0);
  Value spread_lanes =
      arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
  Value spread = muli(count, spread_lanes);
  Value blocks_points =
      func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                           ValueRange{points})
          .getResult(0);
  {
    auto launch =
        gpu::LaunchOp::create(b, loc, blocks_points, c1, c1, c128, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;

    Value tx = launch.getThreadIds().x;

    Value base = muli(bx, c128);
    Value item = addi(base, tx);
    Value inside = cmpi(arith::CmpIPredicate::ult, item, points);
    auto insideBranch =
        scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(insideBranch.thenBlock());
      Value none;
      if constexpr (fixedGrid) {
        none = arith::ConstantOp::create(b, loc, b.getIntegerAttr(i64Type, 0));
      } else {
        none = arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
      }
      store(none, grid, {item});
      scf::YieldOp::create(b, loc);
    }
    gpu::TerminatorOp::create(b, loc);
  }
  Value blocks_spread =
      func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                           ValueRange{spread})
          .getResult(0);
  {
    auto launch =
        gpu::LaunchOp::create(b, loc, blocks_spread, c1, c1, c128, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value base = muli(bx, c128);
    Value item = addi(base, tx);
    Value inside = cmpi(arith::CmpIPredicate::ult, item, spread);
    auto insideBranch2 =
        scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(insideBranch2.thenBlock());
      Value laneCount =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
      Value i;
      if constexpr (!useList) {
        i = divui(item, laneCount);
      }
      Value slot;
      if constexpr (useList) {
        slot = divui(item, laneCount);
      }
      Value j3 = remui(item, laneCount);
      scf::IfOp is_listedBranch;
      if constexpr (useList) {
        Value listed = load(list, {slot});
        Value listed_none =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
        Value is_listed = cmpi(arith::CmpIPredicate::sge, listed, listed_none);
        is_listedBranch =
            scf::IfOp::create(b, loc, TypeRange{}, is_listed, true, false);
        b.setInsertionPointToStart(is_listedBranch.thenBlock());
        i = indexCast(idxType, listed);
      }
      Value i0 = arith::ConstantIndexOp::create(b, loc, 0);
      Value i1 = arith::ConstantIndexOp::create(b, loc, 1);
      Value i2 = arith::ConstantIndexOp::create(b, loc, 2);
      Value pointCount =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
      Value wx =
          memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
      Value wy =
          memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
      Value wz =
          memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
      Value xs = load(x, {i, i0});
      Value ys = load(x, {i, i1});
      Value zs = load(x, {i, i2});
      Value qs = load(q, {i});
      Value xi = convert(f64Type, xs);
      Value yi = convert(f64Type, ys);
      Value zi = convert(f64Type, zs);
      Value qi = convert(realType, qs);
      std::pair<Value, Value> xPlacement;
      if constexpr (!tilted) {
        xPlacement = placeAxis(xi, ilx, k1, pointCount);
      }
      Value fr_one;
      Value fr_z;
      Value fr_y;
      if constexpr (tilted) {
        fr_one =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
        fr_z = mulf(zi, h22);
        Value fr_y1 = mulf(yi, h11);
        Value fr_y2 = mulf(zi, h21);
        fr_y = addf(fr_y1, fr_y2);
        Value fr_x1 = mulf(xi, h00);
        Value fr_x2 = mulf(yi, h10);
        Value fr_x3 = mulf(zi, h20);
        Value fr_x12 = addf(fr_x1, fr_x2);
        Value fr_x = addf(fr_x12, fr_x3);
        xPlacement = placeAxis(fr_x, fr_one, k1, pointCount);
      }
      Value px_start = xPlacement.first;
      Value px_w = xPlacement.second;
      splineWeights(px_w, pointCount, wx);
      std::pair<Value, Value> yPlacement;
      if constexpr (!tilted) {
        yPlacement = placeAxis(yi, ily, k2, pointCount);
      } else {
        yPlacement = placeAxis(fr_y, fr_one, k2, pointCount);
      }
      Value py_start = yPlacement.first;
      Value py_w = yPlacement.second;
      splineWeights(py_w, pointCount, wy);
      std::pair<Value, Value> zPlacement;
      if constexpr (!tilted) {
        zPlacement = placeAxis(zi, ilz, k3, pointCount);
      } else {
        zPlacement = placeAxis(fr_z, fr_one, k3, pointCount);
      }
      Value pz_start = zPlacement.first;
      Value pz_w = zPlacement.second;
      splineWeights(pz_w, pointCount, wz);
      Value w3p_value = pickSpline(wz, j3, pointCount);
      Value scale;
      if constexpr (fixedGrid) {
        scale = arith::ConstantOp::create(
            b, loc, b.getFloatAttr(realType, 1099511627776.0));
      } else {
        scale =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 1.0));
      }
      Value qscaled = mulf(qi, scale);
      Value eight;
      if constexpr (fixedGrid) {
        eight = arith::ConstantOp::create(b, loc, b.getIntegerAttr(i64Type, 8));
      } else {
        eight = arith::ConstantOp::create(b, loc,
                                          b.getIntegerAttr(i64Type, realBytes));
      }
      Value gridindex =
          memref::ExtractAlignedPointerAsIndexOp::create(b, loc, grid);
      Value gridbase = indexCast(i64Type, gridindex);
      Value j0 = arith::ConstantIndexOp::create(b, loc, 0);
      Value j1c = arith::ConstantIndexOp::create(b, loc, 1);
      Value g3s = addi(pz_start, j3);
      Value g3w = subi(g3s, k3);
      Value g3over = cmpi(arith::CmpIPredicate::uge, g3s, k3);
      Value g3 = select(g3over, g3w, g3s);
      Value q3 = mulf(qscaled, w3p_value);
      scf::ForOp::create(
          b, loc, j0, pointCount, j1c, ValueRange{},
          [&](OpBuilder &b, Location, Value j1, ValueRange carried) {
            Value g1s = addi(px_start, j1);
            Value g1w = subi(g1s, k1);
            Value g1over = cmpi(arith::CmpIPredicate::uge, g1s, k1);
            Value g1 = select(g1over, g1w, g1s);
            Value w1 = load(wx, {j1});
            Value q13 = mulf(q3, w1);
            scf::ForOp::create(
                b, loc, j0, pointCount, j1c, ValueRange{},
                [&](OpBuilder &b, Location, Value j2, ValueRange carried) {
                  Value g2s = addi(py_start, j2);
                  Value g2w = subi(g2s, k2);
                  Value g2over = cmpi(arith::CmpIPredicate::uge, g2s, k2);
                  Value g2 = select(g2over, g2w, g2s);
                  Value w2 = load(wy, {j2});
                  Value value = mulf(q13, w2);
                  Value row1 = muli(g1, k2);
                  Value row = addi(row1, g2);
                  Value gbase = muli(row, k3);
                  Value fixed;
                  if constexpr (fixedGrid) {
                    Value rounded = math::RoundEvenOp::create(b, loc, value);
                    fixed = fptosi(i64Type, rounded);
                  }
                  Value at = addi(gbase, g3);
                  Value ati = indexCast(i64Type, at);
                  Value offset = muli(ati, eight);
                  Value address = addi(gridbase, offset);
                  Value pointer =
                      LLVM::IntToPtrOp::create(b, loc, globalPointer, address);
                  if constexpr (fixedGrid) {
                    LLVM::AtomicRMWOp::create(
                        b, loc, LLVM::AtomicBinOp::add, pointer, fixed,
                        LLVM::AtomicOrdering::monotonic, "device");
                  } else {
                    addGlobal(pointer, value);
                  }
                  scf::YieldOp::create(b, loc);
                });
            scf::YieldOp::create(b, loc);
          });
      scf::YieldOp::create(b, loc);
      if constexpr (useList) {
        b.setInsertionPointAfter(is_listedBranch);
        scf::YieldOp::create(b, loc);
      }
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitSpread<false, false, true>(StringRef);
template void DevicePME::emitSpread<false, false, false>(StringRef);
template void DevicePME::emitSpread<false, true, true>(StringRef);
template void DevicePME::emitSpread<false, true, false>(StringRef);
template void DevicePME::emitSpread<true, false, true>(StringRef);
template void DevicePME::emitSpread<true, false, false>(StringRef);
template void DevicePME::emitSpread<true, true, true>(StringRef);
template void DevicePME::emitSpread<true, true, false>(StringRef);
