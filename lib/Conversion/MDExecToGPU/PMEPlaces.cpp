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
        z_s = mulf(zp, h22);
        Value y_t1 = mulf(yp, h11);
        Value y_t2 = mulf(zp, h21);
        y_s = addf(y_t1, y_t2);
        Value x_t1 = mulf(xp, h00);
        Value x_t2 = mulf(yp, h10);
        Value x_t3 = mulf(zp, h20);
        Value x_t12 = addf(x_t1, x_t2);
        x_s = addf(x_t12, x_t3);
      }
      Value x_fs = math::FloorOp::create(b, loc, x_s);
      Value x_frac = subf(x_s, x_fs);
      Value x_ki = indexCast(i32Type, k1);
      Value x_kf = sitofp(realType, x_ki);
      Value x_fracr = convert(realType, x_frac);
      Value x_u = mulf(x_fracr, x_kf);
      Value x_v = subf(x_u, shift);
      Value x_fl0 = math::FloorOp::create(b, loc, x_v);
      Value x_fl = maxnumf(x_fl0, below_one);
      Value x_w = subf(x_v, x_fl);
      Value x_b = fptosi(i32Type, x_fl);
      Value x_s0 = addi(x_b, x_ki);
      Value x_s1 = addi(x_s0, x_ki);
      Value x_s2 = subi(x_s1, half_points);
      Value x_s3 = addi(x_s2, one32);
      Value x_start = remsi(x_s3, x_ki);
      Value x_startr = sitofp(realType, x_start);
      Value x_sat0 = arith::ConstantIndexOp::create(b, loc, 0);
      Value x_sn = muli(n_all, x_sat0);
      Value x_sat = addi(x_sn, i);
      store(x_startr, places, {x_sat});
      Value x_fat0 = arith::ConstantIndexOp::create(b, loc, 3);
      Value x_fn = muli(n_all, x_fat0);
      Value x_fat = addi(x_fn, i);
      store(x_w, places, {x_fat});
      Value y_fs = math::FloorOp::create(b, loc, y_s);
      Value y_frac = subf(y_s, y_fs);
      Value y_ki = indexCast(i32Type, k2);
      Value y_kf = sitofp(realType, y_ki);
      Value y_fracr = convert(realType, y_frac);
      Value y_u = mulf(y_fracr, y_kf);
      Value y_v = subf(y_u, shift);
      Value y_fl0 = math::FloorOp::create(b, loc, y_v);
      Value y_fl = maxnumf(y_fl0, below_one);
      Value y_w = subf(y_v, y_fl);
      Value y_b = fptosi(i32Type, y_fl);
      Value y_s0 = addi(y_b, y_ki);
      Value y_s1 = addi(y_s0, y_ki);
      Value y_s2 = subi(y_s1, half_points);
      Value y_s3 = addi(y_s2, one32);
      Value y_start = remsi(y_s3, y_ki);
      Value y_startr = sitofp(realType, y_start);
      Value y_sat0 = arith::ConstantIndexOp::create(b, loc, 1);
      Value y_sn = muli(n_all, y_sat0);
      Value y_sat = addi(y_sn, i);
      store(y_startr, places, {y_sat});
      Value y_fat0 = arith::ConstantIndexOp::create(b, loc, 4);
      Value y_fn = muli(n_all, y_fat0);
      Value y_fat = addi(y_fn, i);
      store(y_w, places, {y_fat});
      Value z_fs = math::FloorOp::create(b, loc, z_s);
      Value z_frac = subf(z_s, z_fs);
      Value z_ki = indexCast(i32Type, k3);
      Value z_kf = sitofp(realType, z_ki);
      Value z_fracr = convert(realType, z_frac);
      Value z_u = mulf(z_fracr, z_kf);
      Value z_v = subf(z_u, shift);
      Value z_fl0 = math::FloorOp::create(b, loc, z_v);
      Value z_fl = maxnumf(z_fl0, below_one);
      Value z_w = subf(z_v, z_fl);
      Value z_b = fptosi(i32Type, z_fl);
      Value z_s0 = addi(z_b, z_ki);
      Value z_s1 = addi(z_s0, z_ki);
      Value z_s2 = subi(z_s1, half_points);
      Value z_s3 = addi(z_s2, one32);
      Value z_start = remsi(z_s3, z_ki);
      Value z_startr = sitofp(realType, z_start);
      Value z_sat0 = arith::ConstantIndexOp::create(b, loc, 2);
      Value z_sn = muli(n_all, z_sat0);
      Value z_sat = addi(z_sn, i);
      store(z_startr, places, {z_sat});
      Value z_fat0 = arith::ConstantIndexOp::create(b, loc, 5);
      Value z_fn = muli(n_all, z_fat0);
      Value z_fat = addi(z_fn, i);
      store(z_w, places, {z_fat});
      scf::YieldOp::create(b, loc);
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitPlaces<false>(StringRef);
template void DevicePME::emitPlaces<true>(StringRef);
