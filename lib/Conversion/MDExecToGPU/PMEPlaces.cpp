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
  Value lx = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{0});
  Value ly = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{1});
  Value lz = vector::ExtractOp::create(b, loc, box, ArrayRef<int64_t>{2});
  Value unit = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
  Value ilx = arith::DivFOp::create(b, loc, unit, lx);
  Value ily = arith::DivFOp::create(b, loc, unit, ly);
  Value ilz = arith::DivFOp::create(b, loc, unit, lz);
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
  Value blocks = func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                                      ValueRange{n_all})
                     .getResult(0);
  {
    auto launch = gpu::LaunchOp::create(b, loc, blocks, c1, c1, c128, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value base = arith::MulIOp::create(b, loc, bx, c128);
    Value i = arith::AddIOp::create(b, loc, base, tx);
    Value inside =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, i, n_all);
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
      Value half_points = arith::DivUIOp::create(b, loc, points, two32);
      Value odd = arith::RemUIOp::create(b, loc, points, two32);
      Value odd1 =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, odd, one32);
      Value halfr =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.5));
      Value zeror =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
      Value shift = arith::SelectOp::create(b, loc, odd1, zeror, halfr);
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
        x_s = arith::MulFOp::create(b, loc, xp, ilx);
        y_s = arith::MulFOp::create(b, loc, yp, ily);
        z_s = arith::MulFOp::create(b, loc, zp, ilz);
      } else {
        z_s = arith::MulFOp::create(b, loc, zp, h22);
        Value y_t1 = arith::MulFOp::create(b, loc, yp, h11);
        Value y_t2 = arith::MulFOp::create(b, loc, zp, h21);
        y_s = arith::AddFOp::create(b, loc, y_t1, y_t2);
        Value x_t1 = arith::MulFOp::create(b, loc, xp, h00);
        Value x_t2 = arith::MulFOp::create(b, loc, yp, h10);
        Value x_t3 = arith::MulFOp::create(b, loc, zp, h20);
        Value x_t12 = arith::AddFOp::create(b, loc, x_t1, x_t2);
        x_s = arith::AddFOp::create(b, loc, x_t12, x_t3);
      }
      Value x_fs = math::FloorOp::create(b, loc, x_s);
      Value x_frac = arith::SubFOp::create(b, loc, x_s, x_fs);
      Value x_ki = arith::IndexCastOp::create(b, loc, i32Type, k1);
      Value x_kf = arith::SIToFPOp::create(b, loc, realType, x_ki);
      Value x_fracr = convert(realType, x_frac);
      Value x_u = arith::MulFOp::create(b, loc, x_fracr, x_kf);
      Value x_v = arith::SubFOp::create(b, loc, x_u, shift);
      Value x_fl0 = math::FloorOp::create(b, loc, x_v);
      Value x_fl = arith::MaxNumFOp::create(b, loc, x_fl0, below_one);
      Value x_w = arith::SubFOp::create(b, loc, x_v, x_fl);
      Value x_b = arith::FPToSIOp::create(b, loc, i32Type, x_fl);
      Value x_s0 = arith::AddIOp::create(b, loc, x_b, x_ki);
      Value x_s1 = arith::AddIOp::create(b, loc, x_s0, x_ki);
      Value x_s2 = arith::SubIOp::create(b, loc, x_s1, half_points);
      Value x_s3 = arith::AddIOp::create(b, loc, x_s2, one32);
      Value x_start = arith::RemSIOp::create(b, loc, x_s3, x_ki);
      Value x_startr = arith::SIToFPOp::create(b, loc, realType, x_start);
      Value x_sat0 = arith::ConstantIndexOp::create(b, loc, 0);
      Value x_sn = arith::MulIOp::create(b, loc, n_all, x_sat0);
      Value x_sat = arith::AddIOp::create(b, loc, x_sn, i);
      store(x_startr, places, {x_sat});
      Value x_fat0 = arith::ConstantIndexOp::create(b, loc, 3);
      Value x_fn = arith::MulIOp::create(b, loc, n_all, x_fat0);
      Value x_fat = arith::AddIOp::create(b, loc, x_fn, i);
      store(x_w, places, {x_fat});
      Value y_fs = math::FloorOp::create(b, loc, y_s);
      Value y_frac = arith::SubFOp::create(b, loc, y_s, y_fs);
      Value y_ki = arith::IndexCastOp::create(b, loc, i32Type, k2);
      Value y_kf = arith::SIToFPOp::create(b, loc, realType, y_ki);
      Value y_fracr = convert(realType, y_frac);
      Value y_u = arith::MulFOp::create(b, loc, y_fracr, y_kf);
      Value y_v = arith::SubFOp::create(b, loc, y_u, shift);
      Value y_fl0 = math::FloorOp::create(b, loc, y_v);
      Value y_fl = arith::MaxNumFOp::create(b, loc, y_fl0, below_one);
      Value y_w = arith::SubFOp::create(b, loc, y_v, y_fl);
      Value y_b = arith::FPToSIOp::create(b, loc, i32Type, y_fl);
      Value y_s0 = arith::AddIOp::create(b, loc, y_b, y_ki);
      Value y_s1 = arith::AddIOp::create(b, loc, y_s0, y_ki);
      Value y_s2 = arith::SubIOp::create(b, loc, y_s1, half_points);
      Value y_s3 = arith::AddIOp::create(b, loc, y_s2, one32);
      Value y_start = arith::RemSIOp::create(b, loc, y_s3, y_ki);
      Value y_startr = arith::SIToFPOp::create(b, loc, realType, y_start);
      Value y_sat0 = arith::ConstantIndexOp::create(b, loc, 1);
      Value y_sn = arith::MulIOp::create(b, loc, n_all, y_sat0);
      Value y_sat = arith::AddIOp::create(b, loc, y_sn, i);
      store(y_startr, places, {y_sat});
      Value y_fat0 = arith::ConstantIndexOp::create(b, loc, 4);
      Value y_fn = arith::MulIOp::create(b, loc, n_all, y_fat0);
      Value y_fat = arith::AddIOp::create(b, loc, y_fn, i);
      store(y_w, places, {y_fat});
      Value z_fs = math::FloorOp::create(b, loc, z_s);
      Value z_frac = arith::SubFOp::create(b, loc, z_s, z_fs);
      Value z_ki = arith::IndexCastOp::create(b, loc, i32Type, k3);
      Value z_kf = arith::SIToFPOp::create(b, loc, realType, z_ki);
      Value z_fracr = convert(realType, z_frac);
      Value z_u = arith::MulFOp::create(b, loc, z_fracr, z_kf);
      Value z_v = arith::SubFOp::create(b, loc, z_u, shift);
      Value z_fl0 = math::FloorOp::create(b, loc, z_v);
      Value z_fl = arith::MaxNumFOp::create(b, loc, z_fl0, below_one);
      Value z_w = arith::SubFOp::create(b, loc, z_v, z_fl);
      Value z_b = arith::FPToSIOp::create(b, loc, i32Type, z_fl);
      Value z_s0 = arith::AddIOp::create(b, loc, z_b, z_ki);
      Value z_s1 = arith::AddIOp::create(b, loc, z_s0, z_ki);
      Value z_s2 = arith::SubIOp::create(b, loc, z_s1, half_points);
      Value z_s3 = arith::AddIOp::create(b, loc, z_s2, one32);
      Value z_start = arith::RemSIOp::create(b, loc, z_s3, z_ki);
      Value z_startr = arith::SIToFPOp::create(b, loc, realType, z_start);
      Value z_sat0 = arith::ConstantIndexOp::create(b, loc, 2);
      Value z_sn = arith::MulIOp::create(b, loc, n_all, z_sat0);
      Value z_sat = arith::AddIOp::create(b, loc, z_sn, i);
      store(z_startr, places, {z_sat});
      Value z_fat0 = arith::ConstantIndexOp::create(b, loc, 5);
      Value z_fn = arith::MulIOp::create(b, loc, n_all, z_fat0);
      Value z_fat = arith::AddIOp::create(b, loc, z_fn, i);
      store(z_w, places, {z_fat});
      scf::YieldOp::create(b, loc);
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitPlaces<false>(StringRef);
template void DevicePME::emitPlaces<true>(StringRef);
