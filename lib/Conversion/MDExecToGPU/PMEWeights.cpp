// Cache the order-four spline weights and particle placement (D109).
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <bool tilted> void DevicePME::emitWeights(StringRef symbol) {
  StringRef blockFunction =
      tilted ? "mdrt_gpu_pme_blocks_triclinic" : "mdrt_gpu_pme_blocks";
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(globalMem({ShapedType::kDynamic, 3}, positionType));
  if constexpr (!tilted) {
    inputs.push_back(VectorType::get({3}, f64Type));
  } else {
    inputs.push_back(VectorType::get({6}, f64Type));
  }
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.append(3, idxType);
  auto fn = func::FuncOp::create(b, loc, symbol, b.getFunctionType(inputs, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  unsigned arg = 0;
  Value x = entry->getArgument(arg++);
  Value box = entry->getArgument(arg++);
  Value weights = entry->getArgument(arg++);
  Value k1 = entry->getArgument(arg++);
  Value k2 = entry->getArgument(arg++);
  Value k3 = entry->getArgument(arg++);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c3 = arith::ConstantIndexOp::create(b, loc, 3);
  Value c4 = arith::ConstantIndexOp::create(b, loc, 4);
  Value c16 = arith::ConstantIndexOp::create(b, loc, 16);
  arith::ConstantIndexOp::create(b, loc, 32);
  Value c128 = arith::ConstantIndexOp::create(b, loc, 128);
  Value n_all = memref::DimOp::create(b, loc, x, c0);
  Value k1u = addi(k1, c3);
  Value b1 = divui(k1u, c4);
  Value k2u = addi(k2, c3);
  Value b2 = divui(k2u, c4);
  Value b12 = muli(b1, b2);
  Value b123 = muli(b12, k3);
  muli(b123, c16);
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
  Value blocks_n_all =
      func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                           ValueRange{n_all})
          .getResult(0);
  {
    auto launch =
        gpu::LaunchOp::create(b, loc, blocks_n_all, c1, c1, c128, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value base = muli(bx, c128);
    Value item = addi(base, tx);
    Value inside = cmpi(arith::CmpIPredicate::ult, item, n_all);
    auto insideBranch =
        scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(insideBranch.thenBlock());
      Value i = addi(item, c0);
      arith::ConstantIndexOp::create(b, loc, 0);
      arith::ConstantIndexOp::create(b, loc, 1);
      arith::ConstantIndexOp::create(b, loc, 2);
      Value pointCount =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
      Value xc = arith::ConstantIndexOp::create(b, loc, 0);
      Value xs = load(x, {i, xc});
      Value xp = convert(f64Type, xs);
      std::pair<Value, Value> xPlacement;
      if constexpr (!tilted) {
        xPlacement = placeAxis(xp, ilx, k1, pointCount);
      }
      Value yc;
      Value zc;
      Value fr_one;
      Value fr_z;
      Value fr_y;
      if constexpr (tilted) {
        yc = arith::ConstantIndexOp::create(b, loc, 1);
        Value ys = load(x, {i, yc});
        Value yp = convert(f64Type, ys);
        zc = arith::ConstantIndexOp::create(b, loc, 2);
        Value zs = load(x, {i, zc});
        Value zp = convert(f64Type, zs);
        fr_one =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
        fr_z = mulf(zp, h22);
        Value fr_y1 = mulf(yp, h11);
        Value fr_y2 = mulf(zp, h21);
        fr_y = addf(fr_y1, fr_y2);
        Value fr_x1 = mulf(xp, h00);
        Value fr_x2 = mulf(yp, h10);
        Value fr_x3 = mulf(zp, h20);
        Value fr_x12 = addf(fr_x1, fr_x2);
        Value fr_x = addf(fr_x12, fr_x3);
        xPlacement = placeAxis(fr_x, fr_one, k1, pointCount);
      }
      Value px_start = xPlacement.first;
      Value px_w = xPlacement.second;
      Value xfc = arith::ConstantIndexOp::create(b, loc, 3);
      Value xfn = muli(n_all, xfc);
      Value xfat = addi(xfn, i);
      store(px_w, weights, {xfat});
      Value xstart_i = indexCast(i32Type, px_start);
      Value xstart_r = sitofp(realType, xstart_i);
      Value xsn = muli(n_all, xc);
      Value xsat = addi(xsn, i);
      store(xstart_r, weights, {xsat});
      std::pair<Value, Value> yPlacement;
      if constexpr (!tilted) {
        yc = arith::ConstantIndexOp::create(b, loc, 1);
        Value ys = load(x, {i, yc});
        Value yp = convert(f64Type, ys);
        yPlacement = placeAxis(yp, ily, k2, pointCount);
      } else {
        yPlacement = placeAxis(fr_y, fr_one, k2, pointCount);
      }
      Value py_start = yPlacement.first;
      Value py_w = yPlacement.second;
      Value yfc = arith::ConstantIndexOp::create(b, loc, 4);
      Value yfn = muli(n_all, yfc);
      Value yfat = addi(yfn, i);
      store(py_w, weights, {yfat});
      Value ystart_i = indexCast(i32Type, py_start);
      Value ystart_r = sitofp(realType, ystart_i);
      Value ysn = muli(n_all, yc);
      Value ysat = addi(ysn, i);
      store(ystart_r, weights, {ysat});
      std::pair<Value, Value> zPlacement;
      if constexpr (!tilted) {
        zc = arith::ConstantIndexOp::create(b, loc, 2);
        Value zs = load(x, {i, zc});
        Value zp = convert(f64Type, zs);
        zPlacement = placeAxis(zp, ilz, k3, pointCount);
      } else {
        zPlacement = placeAxis(fr_z, fr_one, k3, pointCount);
      }
      Value pz_start = zPlacement.first;
      Value pz_w = zPlacement.second;
      Value zfc = arith::ConstantIndexOp::create(b, loc, 5);
      Value zfn = muli(n_all, zfc);
      Value zfat = addi(zfn, i);
      store(pz_w, weights, {zfat});
      Value zstart_i = indexCast(i32Type, pz_start);
      Value zstart_r = sitofp(realType, zstart_i);
      Value zsn = muli(n_all, zc);
      Value zsat = addi(zsn, i);
      store(zstart_r, weights, {zsat});
      scf::YieldOp::create(b, loc);
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitWeights<false>(StringRef);
template void DevicePME::emitWeights<true>(StringRef);
