// Shifted prolate placement on the periodic mesh [Liang2026].
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <bool tilted> void DevicePME::emitPlaces(StringRef symbol) {
  StringRef blockFunction = "mdrt_gpu_pme_prolate_blocks";
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
  Value places = entry->getArgument(arg++);
  Value k1 = entry->getArgument(arg++);
  Value k2 = entry->getArgument(arg++);
  Value k3 = entry->getArgument(arg++);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c128 = arith::ConstantIndexOp::create(b, loc, 128);
  Value n_all = memref::DimOp::create(b, loc, x, c0);
  Value lx = extract(box, 0);
  Value ly = extract(box, 1);
  Value lz = extract(box, 2);
  Value unit = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
  Value ilx = divf(unit, lx);
  Value ily = divf(unit, ly);
  Value ilz = divf(unit, lz);
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
  Value blocks = func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                                      ValueRange{n_all})
                     .getResult(0);
  {
    auto launch = gpu::LaunchOp::create(b, loc, blocks, c1, c1, c128, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value base = muli(bx, c128);
    Value i = addi(base, tx);
    Value inside = cmpi(arith::CmpIPredicate::ult, i, n_all);
    auto insideBranch =
        scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(insideBranch.thenBlock());
      Value points =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, order));
      Value two32 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 2));
      Value one32 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 1));
      Value half_points = divui(points, two32);
      Value odd = remui(points, two32);
      Value odd1 = cmpi(arith::CmpIPredicate::ne, odd, one32);
      Value halfr =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.5));
      Value zeror =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
      Value shift = select(odd1, zeror, halfr);
      Value below_one =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, -1.0));
      Value xc = arith::ConstantIndexOp::create(b, loc, 0);
      Value xs = load(x, {i, xc});
      Value xp = convert(f64Type, xs);
      Value yc = arith::ConstantIndexOp::create(b, loc, 1);
      Value ys = load(x, {i, yc});
      Value yp = convert(f64Type, ys);
      Value zc = arith::ConstantIndexOp::create(b, loc, 2);
      Value zs = load(x, {i, zc});
      Value zp = convert(f64Type, zs);
      Value x_s;
      Value y_s;
      Value z_s;
      if constexpr (!tilted) {
        x_s = mulf(xp, ilx);
        y_s = mulf(yp, ily);
        z_s = mulf(zp, ilz);
      } else {
        auto pos = fractionalPosition(xp, yp, zp, h00, h10, h11, h20, h21, h22);
        x_s = pos.x;
        y_s = pos.y;
        z_s = pos.z;
      }
      auto place = [&](Value coordinate, Value size, int axis) {
        Value fs = math::FloorOp::create(b, loc, coordinate);
        Value frac = subf(coordinate, fs);
        Value ki = indexCast(i32Type, size);
        Value kf = sitofp(realType, ki);
        Value fracr = convert(realType, frac);
        Value u = mulf(fracr, kf);
        Value v = subf(u, shift);
        Value fl0 = math::FloorOp::create(b, loc, v);
        Value fl = maxnumf(fl0, below_one);
        Value w = subf(v, fl);
        Value base = fptosi(i32Type, fl);
        Value s0 = addi(base, ki);
        Value s1 = addi(s0, ki);
        Value s2 = subi(s1, half_points);
        Value s3 = addi(s2, one32);
        Value start = remsi(s3, ki);
        Value startr = sitofp(realType, start);
        Value sat0 = arith::ConstantIndexOp::create(b, loc, axis);
        Value sn = muli(n_all, sat0);
        Value sat = addi(sn, i);
        store(startr, places, {sat});
        Value fat0 = arith::ConstantIndexOp::create(b, loc, axis + 3);
        Value fn = muli(n_all, fat0);
        Value fat = addi(fn, i);
        store(w, places, {fat});
      };
      place(x_s, k1, 0);
      place(y_s, k2, 1);
      place(z_s, k3, 2);
      scf::YieldOp::create(b, loc);
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitPlaces<false>(StringRef);
template void DevicePME::emitPlaces<true>(StringRef);
