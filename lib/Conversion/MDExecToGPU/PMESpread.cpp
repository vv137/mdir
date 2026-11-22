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
  Value lx = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{0});
  Value ly = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{1});
  Value lz = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{2});
  Value h00;
  Value h11;
  Value h22;
  Value h10;
  Value h21;
  Value h20;
  if constexpr (tilted) {
    Value tbx = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{3});
    Value tcx = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{4});
    Value tcy = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{5});
    Value hunit =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
    h00 = arith::DivFOp::create(b, loc, hunit, lx);
    h11 = arith::DivFOp::create(b, loc, hunit, ly);
    h22 = arith::DivFOp::create(b, loc, hunit, lz);
    Value h10a = arith::MulFOp::create(b, loc, tbx, h00);
    Value h10b = arith::MulFOp::create(b, loc, h10a, h11);
    h10 = arith::NegFOp::create(b, loc, h10b);
    Value h21a = arith::MulFOp::create(b, loc, tcy, h11);
    Value h21b = arith::MulFOp::create(b, loc, h21a, h22);
    h21 = arith::NegFOp::create(b, loc, h21b);
    Value h20a = arith::MulFOp::create(b, loc, tbx, tcy);
    Value h20b = arith::MulFOp::create(b, loc, ly, tcx);
    Value h20c = arith::SubFOp::create(b, loc, h20a, h20b);
    Value h20d = arith::MulFOp::create(b, loc, h20c, h00);
    Value h20e = arith::MulFOp::create(b, loc, h20d, h11);
    h20 = arith::MulFOp::create(b, loc, h20e, h22);
    convert(realType, h00);
    convert(realType, h10);
    convert(realType, h11);
    convert(realType, h20);
    convert(realType, h21);
    convert(realType, h22);
  }
  Value unit = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
  Value ilx = arith::DivFOp::create(b, loc, unit, lx);
  Value ily = arith::DivFOp::create(b, loc, unit, ly);
  Value ilz = arith::DivFOp::create(b, loc, unit, lz);
  Value points = memref::DimOp::create(b, loc, grid, c0);
  Value count = memref::DimOp::create(b, loc, x, c0);
  Value spread_lanes =
      arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
  Value spread = arith::MulIOp::create(b, loc, count, spread_lanes);
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

    Value base = arith::MulIOp::create(b, loc, bx, c128);
    Value item = arith::AddIOp::create(b, loc, base, tx);
    Value inside =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, item, points);
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
    Value base = arith::MulIOp::create(b, loc, bx, c128);
    Value item = arith::AddIOp::create(b, loc, base, tx);
    Value inside =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, item, spread);
    auto insideBranch2 =
        scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(insideBranch2.thenBlock());
      Value laneCount =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
      Value i;
      if constexpr (!useList) {
        i = arith::DivUIOp::create(b, loc, item, laneCount);
      }
      Value slot;
      if constexpr (useList) {
        slot = arith::DivUIOp::create(b, loc, item, laneCount);
      }
      Value j3 = arith::RemUIOp::create(b, loc, item, laneCount);
      scf::IfOp is_listedBranch;
      if constexpr (useList) {
        Value listed = load(list, {slot});
        Value listed_none =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
        Value is_listed = arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::sge, listed, listed_none);
        is_listedBranch =
            scf::IfOp::create(b, loc, TypeRange{}, is_listed, true, false);
        b.setInsertionPointToStart(is_listedBranch.thenBlock());
        i = arith::IndexCastOp::create(b, loc, idxType, listed);
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
        fr_z = arith::MulFOp::create(b, loc, zi, h22);
        Value fr_y1 = arith::MulFOp::create(b, loc, yi, h11);
        Value fr_y2 = arith::MulFOp::create(b, loc, zi, h21);
        fr_y = arith::AddFOp::create(b, loc, fr_y1, fr_y2);
        Value fr_x1 = arith::MulFOp::create(b, loc, xi, h00);
        Value fr_x2 = arith::MulFOp::create(b, loc, yi, h10);
        Value fr_x3 = arith::MulFOp::create(b, loc, zi, h20);
        Value fr_x12 = arith::AddFOp::create(b, loc, fr_x1, fr_x2);
        Value fr_x = arith::AddFOp::create(b, loc, fr_x12, fr_x3);
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
      Value qscaled = arith::MulFOp::create(b, loc, qi, scale);
      Value eight;
      if constexpr (fixedGrid) {
        eight = arith::ConstantOp::create(b, loc, b.getIntegerAttr(i64Type, 8));
      } else {
        eight = arith::ConstantOp::create(b, loc,
                                          b.getIntegerAttr(i64Type, realBytes));
      }
      Value gridindex =
          memref::ExtractAlignedPointerAsIndexOp::create(b, loc, grid);
      Value gridbase = arith::IndexCastOp::create(b, loc, i64Type, gridindex);
      Value j0 = arith::ConstantIndexOp::create(b, loc, 0);
      Value j1c = arith::ConstantIndexOp::create(b, loc, 1);
      Value g3s = arith::AddIOp::create(b, loc, pz_start, j3);
      Value g3w = arith::SubIOp::create(b, loc, g3s, k3);
      Value g3over =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, g3s, k3);
      Value g3 = arith::SelectOp::create(b, loc, g3over, g3w, g3s);
      Value q3 = arith::MulFOp::create(b, loc, qscaled, w3p_value);
      scf::ForOp::create(
          b, loc, j0, pointCount, j1c, ValueRange{},
          [&](OpBuilder &b, Location, Value j1, ValueRange carried) {
            Value g1s = arith::AddIOp::create(b, loc, px_start, j1);
            Value g1w = arith::SubIOp::create(b, loc, g1s, k1);
            Value g1over = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::uge, g1s, k1);
            Value g1 = arith::SelectOp::create(b, loc, g1over, g1w, g1s);
            Value w1 = load(wx, {j1});
            Value q13 = arith::MulFOp::create(b, loc, q3, w1);
            scf::ForOp::create(
                b, loc, j0, pointCount, j1c, ValueRange{},
                [&](OpBuilder &b, Location, Value j2, ValueRange carried) {
                  Value g2s = arith::AddIOp::create(b, loc, py_start, j2);
                  Value g2w = arith::SubIOp::create(b, loc, g2s, k2);
                  Value g2over = arith::CmpIOp::create(
                      b, loc, arith::CmpIPredicate::uge, g2s, k2);
                  Value g2 = arith::SelectOp::create(b, loc, g2over, g2w, g2s);
                  Value w2 = load(wy, {j2});
                  Value value = arith::MulFOp::create(b, loc, q13, w2);
                  Value row1 = arith::MulIOp::create(b, loc, g1, k2);
                  Value row = arith::AddIOp::create(b, loc, row1, g2);
                  Value gbase = arith::MulIOp::create(b, loc, row, k3);
                  Value fixed;
                  if constexpr (fixedGrid) {
                    Value rounded = math::RoundEvenOp::create(b, loc, value);
                    fixed = arith::FPToSIOp::create(b, loc, i64Type, rounded);
                  }
                  Value at = arith::AddIOp::create(b, loc, gbase, g3);
                  Value ati = arith::IndexCastOp::create(b, loc, i64Type, at);
                  Value offset = arith::MulIOp::create(b, loc, ati, eight);
                  Value address =
                      arith::AddIOp::create(b, loc, gridbase, offset);
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
