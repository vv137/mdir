// Interpolate from the potential grid, using derivatives for forces
// [Essmann1995].
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <bool tilted, bool useList, bool potentialOnly, bool cachedWeights>
void DevicePME::emitGather(StringRef symbol) {
  StringRef blockFunction =
      tilted ? "mdrt_gpu_pme_blocks_triclinic" : "mdrt_gpu_pme_blocks";
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(globalMem({ShapedType::kDynamic, 3}, positionType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, chargeType));
  if constexpr (useList) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, i32Type));
  }
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  if constexpr (!tilted) {
    inputs.push_back(VectorType::get({3}, f64Type));
  } else {
    inputs.push_back(VectorType::get({6}, f64Type));
  }
  if constexpr ((!potentialOnly && !cachedWeights) ||
                (!tilted && potentialOnly)) {
    inputs.push_back(idxType);
  }
  if constexpr (cachedWeights) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  }
  inputs.append(3, idxType);
  if constexpr (!potentialOnly) {
    inputs.push_back(globalMem({ShapedType::kDynamic, 3}, realType));
  }
  if constexpr (tilted && potentialOnly) {
    inputs.push_back(idxType);
  }
  if constexpr (potentialOnly) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  }
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
  Value phi = entry->getArgument(arg++);
  Value box = entry->getArgument(arg++);
  Value weights;
  if constexpr (cachedWeights) {
    weights = entry->getArgument(arg++);
  }
  Value k1 = entry->getArgument(arg++);
  Value k2 = entry->getArgument(arg++);
  Value k3 = entry->getArgument(arg++);
  if constexpr (!cachedWeights) {
    ++arg;
  }
  Value f = entry->getArgument(arg++);
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
  Value h00r;
  Value h10r;
  Value h11r;
  Value h20r;
  Value h21r;
  Value h22r;
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
    h00r = convert(realType, h00);
    h10r = convert(realType, h10);
    h11r = convert(realType, h11);
    h20r = convert(realType, h20);
    h21r = convert(realType, h21);
    h22r = convert(realType, h22);
  }
  Value k1i = arith::IndexCastOp::create(b, loc, i64Type, k1);
  Value k2i = arith::IndexCastOp::create(b, loc, i64Type, k2);
  Value k3i = arith::IndexCastOp::create(b, loc, i64Type, k3);
  Value k1f = arith::SIToFPOp::create(b, loc, f64Type, k1i);
  Value k2f = arith::SIToFPOp::create(b, loc, f64Type, k2i);
  Value k3f = arith::SIToFPOp::create(b, loc, f64Type, k3i);
  Value rx = arith::DivFOp::create(b, loc, k1f, lx);
  Value ry = arith::DivFOp::create(b, loc, k2f, ly);
  Value rz = arith::DivFOp::create(b, loc, k3f, lz);
  Value rxr = convert(realType, rx);
  Value ryr = convert(realType, ry);
  Value rzr = convert(realType, rz);
  Value k1fr;
  Value k2fr;
  Value k3fr;
  if constexpr (tilted) {
    k1fr = convert(realType, k1f);
    k2fr = convert(realType, k2f);
    k3fr = convert(realType, k3f);
  }
  Value ilx;
  Value ily;
  Value ilz;
  if constexpr (!cachedWeights) {
    Value unit =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
    ilx = arith::DivFOp::create(b, loc, unit, lx);
    ily = arith::DivFOp::create(b, loc, unit, ly);
    ilz = arith::DivFOp::create(b, loc, unit, lz);
  }
  Value count = memref::DimOp::create(b, loc, x, c0);
  Value lanes_all =
      arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, lanes));
  Value points = arith::MulIOp::create(b, loc, count, lanes_all);
  Value last = arith::SubIOp::create(b, loc, count, c1);
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
    Value item0 = arith::AddIOp::create(b, loc, base, tx);
    Value inside =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, item0, points);
    Value lanes_l =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, lanes));
    Value c4l = arith::ConstantIndexOp::create(b, loc, 4);
    Value lane_of = arith::RemUIOp::create(b, loc, item0, lanes_l);
    Value last_first = arith::MulIOp::create(b, loc, last, lanes_l);
    Value last_item = arith::AddIOp::create(b, loc, last_first, lane_of);
    Value item = arith::SelectOp::create(b, loc, inside, item0, last_item);
    Value laneCount =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, lanes));
    Value i;
    if constexpr (!useList) {
      i = arith::DivUIOp::create(b, loc, item, laneCount);
    }
    Value slot;
    if constexpr (useList) {
      slot = arith::DivUIOp::create(b, loc, item, laneCount);
    }
    Value j3 = arith::RemUIOp::create(b, loc, item, laneCount);
    Value listed;
    Value is_listed;
    scf::IfOp warp_activeBranch;
    if constexpr (useList) {
      listed = load(list, {slot});
      Value listed_none =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      is_listed = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                        listed, listed_none);
      Value c32w = arith::ConstantIndexOp::create(b, loc, 32);
      Value lane32 = arith::RemUIOp::create(b, loc, item0, c32w);
      Value wfirst = arith::SubIOp::create(b, loc, item0, lane32);
      Value wslot = arith::DivUIOp::create(b, loc, wfirst, lanes_l);
      Value wslot_in = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                             wslot, count);
      Value wslot_c = arith::SelectOp::create(b, loc, wslot_in, wslot, c0);
      Value warp_listed = load(list, {wslot_c});
      Value wnone =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      Value wlisted = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                            warp_listed, wnone);
      Value warp_active = arith::AndIOp::create(b, loc, wslot_in, wlisted);
      warp_activeBranch =
          scf::IfOp::create(b, loc, TypeRange{}, warp_active, true, false);
      b.setInsertionPointToStart(warp_activeBranch.thenBlock());
      Value listed_or =
          arith::SelectOp::create(b, loc, is_listed, listed, warp_listed);
      i = arith::IndexCastOp::create(b, loc, idxType, listed_or);
    }
    Value i0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value i1 = arith::ConstantIndexOp::create(b, loc, 1);
    Value i2 = arith::ConstantIndexOp::create(b, loc, 2);
    Value pointCount;
    if constexpr (!cachedWeights) {
      pointCount =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
    }
    if constexpr (cachedWeights) {
      pointCount = arith::ConstantIndexOp::create(b, loc, 4);
    }
    Value wx = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
    Value wy = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
    Value wz = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
    Value dx = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
    Value dy = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
    Value dz = memref::AllocaOp::create(b, loc, MemRefType::get({8}, realType));
    Value xs;
    Value ys;
    Value zs;
    if constexpr (!cachedWeights) {
      xs = load(x, {i, i0});
      ys = load(x, {i, i1});
      zs = load(x, {i, i2});
    }
    Value qs = load(q, {i});
    Value xi;
    Value yi;
    Value zi;
    if constexpr (!cachedWeights) {
      xi = convert(f64Type, xs);
      yi = convert(f64Type, ys);
      zi = convert(f64Type, zs);
    }
    Value qi = convert(realType, qs);
    std::pair<Value, Value> xPlacement;
    if constexpr (!tilted && !cachedWeights) {
      xPlacement = placeAxis(xi, ilx, k1, pointCount);
    }
    Value fr_one;
    Value fr_z;
    Value fr_y;
    if constexpr (tilted && !cachedWeights) {
      fr_one = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
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
    Value px_start;
    if constexpr (!cachedWeights) {
      px_start = xPlacement.first;
      Value px_w = xPlacement.second;
      splineWeights(px_w, pointCount, wx, dx);
    }
    std::pair<Value, Value> yPlacement;
    if constexpr (!tilted && !cachedWeights) {
      yPlacement = placeAxis(yi, ily, k2, pointCount);
    }
    if constexpr (tilted && !cachedWeights) {
      yPlacement = placeAxis(fr_y, fr_one, k2, pointCount);
    }
    Value py_start;
    if constexpr (!cachedWeights) {
      py_start = yPlacement.first;
      Value py_w = yPlacement.second;
      splineWeights(py_w, pointCount, wy, dy);
    }
    std::pair<Value, Value> zPlacement;
    if constexpr (!tilted && !cachedWeights) {
      zPlacement = placeAxis(zi, ilz, k3, pointCount);
    }
    if constexpr (tilted && !cachedWeights) {
      zPlacement = placeAxis(fr_z, fr_one, k3, pointCount);
    }
    Value pz_start;
    if constexpr (!cachedWeights) {
      pz_start = zPlacement.first;
      Value pz_w = zPlacement.second;
      splineWeights(pz_w, pointCount, wz, dz);
    }
    Value xstart;
    Value ystart;
    Value zstart;
    if constexpr (cachedWeights) {
      Value xsc = arith::ConstantIndexOp::create(b, loc, 0);
      Value xsn = arith::MulIOp::create(b, loc, count, xsc);
      Value xsat = arith::AddIOp::create(b, loc, xsn, i);
      Value xstart_r = load(weights, {xsat});
      xstart = arith::FPToSIOp::create(b, loc, i32Type, xstart_r);
      Value xfc = arith::ConstantIndexOp::create(b, loc, 3);
      Value xfn = arith::MulIOp::create(b, loc, count, xfc);
      Value xfat = arith::AddIOp::create(b, loc, xfn, i);
      Value xfrac = load(weights, {xfat});
      splineWeights(xfrac, pointCount, wx, dx);
      Value ysc = arith::ConstantIndexOp::create(b, loc, 1);
      Value ysn = arith::MulIOp::create(b, loc, count, ysc);
      Value ysat = arith::AddIOp::create(b, loc, ysn, i);
      Value ystart_r = load(weights, {ysat});
      ystart = arith::FPToSIOp::create(b, loc, i32Type, ystart_r);
      Value yfc = arith::ConstantIndexOp::create(b, loc, 4);
      Value yfn = arith::MulIOp::create(b, loc, count, yfc);
      Value yfat = arith::AddIOp::create(b, loc, yfn, i);
      Value yfrac = load(weights, {yfat});
      splineWeights(yfrac, pointCount, wy, dy);
      Value zsc = arith::ConstantIndexOp::create(b, loc, 2);
      Value zsn = arith::MulIOp::create(b, loc, count, zsc);
      Value zsat = arith::AddIOp::create(b, loc, zsn, i);
      Value zstart_r = load(weights, {zsat});
      zstart = arith::FPToSIOp::create(b, loc, i32Type, zstart_r);
      Value zfc = arith::ConstantIndexOp::create(b, loc, 5);
      Value zfn = arith::MulIOp::create(b, loc, count, zfc);
      Value zfat = arith::AddIOp::create(b, loc, zfn, i);
      Value zfrac = load(weights, {zfat});
      splineWeights(zfrac, pointCount, wz, dz);
    }
    Value w3p_value = pickSpline(wz, j3, pointCount);
    Value d3p_value = pickSpline(dz, j3, pointCount);
    Value gzero =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
    Value j0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value j1c = arith::ConstantIndexOp::create(b, loc, 1);
    Value g3;
    if constexpr (!cachedWeights) {
      g3 = md_alg::WrapOp::create(b, loc, pz_start, j3, k3, false);
    }
    Value k1t;
    Value k2t;
    Value k3t;
    if constexpr (cachedWeights) {
      k1t = arith::IndexCastOp::create(b, loc, i32Type, k1);
      k2t = arith::IndexCastOp::create(b, loc, i32Type, k2);
      k3t = arith::IndexCastOp::create(b, loc, i32Type, k3);
      Value j3t = arith::IndexCastOp::create(b, loc, i32Type, j3);
      g3 = md_alg::WrapOp::create(b, loc, zstart, j3t, k3t, false);
    }
    auto sumxOp = scf::ForOp::create(
        b, loc, j0, pointCount, j1c, ValueRange{gzero, gzero, gzero},
        [&](OpBuilder &b, Location, Value j1, ValueRange carried) {
          Value ax = carried[0];
          Value ay = carried[1];
          Value az = carried[2];
          Value g1s;
          Value g1w;
          Value g1over;
          if constexpr (!cachedWeights) {
            g1s = arith::AddIOp::create(b, loc, px_start, j1);
            g1w = arith::SubIOp::create(b, loc, g1s, k1);
            g1over = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge,
                                           g1s, k1);
          }
          if constexpr (cachedWeights) {
            Value j1t = arith::IndexCastOp::create(b, loc, i32Type, j1);
            g1s = arith::AddIOp::create(b, loc, xstart, j1t);
            g1w = arith::SubIOp::create(b, loc, g1s, k1t);
            g1over = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge,
                                           g1s, k1t);
          }
          Value g1 = arith::SelectOp::create(b, loc, g1over, g1w, g1s);
          Value row1;
          if constexpr (cachedWeights) {
            row1 = arith::MulIOp::create(b, loc, g1, k2t);
          }
          Value w1 = load(wx, {j1});
          Value d1 = load(dx, {j1});
          auto bx2Op = scf::ForOp::create(
              b, loc, j0, pointCount, j1c, ValueRange{ax, ay, az},
              [&](OpBuilder &b, Location, Value j2, ValueRange carried) {
                Value cx = carried[0];
                Value cy = carried[1];
                Value cz = carried[2];
                Value g2s;
                Value g2w;
                Value g2over;
                if constexpr (!cachedWeights) {
                  g2s = arith::AddIOp::create(b, loc, py_start, j2);
                  g2w = arith::SubIOp::create(b, loc, g2s, k2);
                  g2over = arith::CmpIOp::create(
                      b, loc, arith::CmpIPredicate::uge, g2s, k2);
                }
                if constexpr (cachedWeights) {
                  Value j2t = arith::IndexCastOp::create(b, loc, i32Type, j2);
                  g2s = arith::AddIOp::create(b, loc, ystart, j2t);
                  g2w = arith::SubIOp::create(b, loc, g2s, k2t);
                  g2over = arith::CmpIOp::create(
                      b, loc, arith::CmpIPredicate::uge, g2s, k2t);
                }
                Value g2 = arith::SelectOp::create(b, loc, g2over, g2w, g2s);
                Value w2 = load(wy, {j2});
                Value d2 = load(dy, {j2});
                if constexpr (!cachedWeights) {
                  row1 = arith::MulIOp::create(b, loc, g1, k2);
                }
                Value row = arith::AddIOp::create(b, loc, row1, g2);
                Value gbase;
                if constexpr (!cachedWeights) {
                  gbase = arith::MulIOp::create(b, loc, row, k3);
                }
                if constexpr (cachedWeights) {
                  gbase = arith::MulIOp::create(b, loc, row, k3t);
                }
                Value p =
                    md_alg::GridReadOp::create(b, loc, phi, gbase, g3, Value{});
                Value d1w2 = arith::MulFOp::create(b, loc, d1, w2);
                Value w1d2 = arith::MulFOp::create(b, loc, w1, d2);
                Value w1w2 = arith::MulFOp::create(b, loc, w1, w2);
                Value px;
                if constexpr (!potentialOnly) {
                  px = arith::MulFOp::create(b, loc, d1w2, w3p_value);
                } else {
                  px = arith::MulFOp::create(b, loc, w1w2, w3p_value);
                }
                Value py = arith::MulFOp::create(b, loc, w1d2, w3p_value);
                Value pz = arith::MulFOp::create(b, loc, w1w2, d3p_value);
                Value vx = arith::MulFOp::create(b, loc, p, px);
                Value vy = arith::MulFOp::create(b, loc, p, py);
                Value vz = arith::MulFOp::create(b, loc, p, pz);
                Value nx = arith::AddFOp::create(b, loc, cx, vx);
                Value ny = arith::AddFOp::create(b, loc, cy, vy);
                Value nz = arith::AddFOp::create(b, loc, cz, vz);
                scf::YieldOp::create(b, loc, ValueRange{nx, ny, nz});
              });
          Value bx2 = bx2Op.getResult(0);
          Value by2 = bx2Op.getResult(1);
          Value bz2 = bx2Op.getResult(2);
          scf::YieldOp::create(b, loc, ValueRange{bx2, by2, bz2});
        });
    Value sumx = sumxOp.getResult(0);
    Value sumy = sumxOp.getResult(1);
    Value sumz = sumxOp.getResult(2);
    Value width =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
    Value one32 =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 1));
    Value two32 =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 2));
    auto sx1Shuffle = gpu::ShuffleOp::create(b, loc, sumx, one32, width,
                                             gpu::ShuffleMode::XOR);
    Value sx1 = sx1Shuffle.getShuffleResult();

    auto sy1Shuffle = gpu::ShuffleOp::create(b, loc, sumy, one32, width,
                                             gpu::ShuffleMode::XOR);
    Value sy1 = sy1Shuffle.getShuffleResult();

    auto sz1Shuffle = gpu::ShuffleOp::create(b, loc, sumz, one32, width,
                                             gpu::ShuffleMode::XOR);
    Value sz1 = sz1Shuffle.getShuffleResult();

    Value tx1 = arith::AddFOp::create(b, loc, sumx, sx1);
    Value ty1 = arith::AddFOp::create(b, loc, sumy, sy1);
    Value tz1 = arith::AddFOp::create(b, loc, sumz, sz1);
    auto sx2Shuffle = gpu::ShuffleOp::create(b, loc, tx1, two32, width,
                                             gpu::ShuffleMode::XOR);
    Value sx2 = sx2Shuffle.getShuffleResult();

    auto sy2Shuffle = gpu::ShuffleOp::create(b, loc, ty1, two32, width,
                                             gpu::ShuffleMode::XOR);
    Value sy2 = sy2Shuffle.getShuffleResult();

    auto sz2Shuffle = gpu::ShuffleOp::create(b, loc, tz1, two32, width,
                                             gpu::ShuffleMode::XOR);
    Value sz2 = sz2Shuffle.getShuffleResult();

    Value ux = arith::AddFOp::create(b, loc, tx1, sx2);
    Value uy = arith::AddFOp::create(b, loc, ty1, sy2);
    Value uz = arith::AddFOp::create(b, loc, tz1, sz2);
    Value four32 =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 4));
    Value eight_lanes =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, lanes_l, c4l);
    auto sx4Shuffle = gpu::ShuffleOp::create(b, loc, ux, four32, width,
                                             gpu::ShuffleMode::XOR);
    Value sx4 = sx4Shuffle.getShuffleResult();

    auto sy4Shuffle = gpu::ShuffleOp::create(b, loc, uy, four32, width,
                                             gpu::ShuffleMode::XOR);
    Value sy4 = sy4Shuffle.getShuffleResult();

    auto sz4Shuffle = gpu::ShuffleOp::create(b, loc, uz, four32, width,
                                             gpu::ShuffleMode::XOR);
    Value sz4 = sz4Shuffle.getShuffleResult();

    Value zero_r =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
    Value hx = arith::SelectOp::create(b, loc, eight_lanes, sx4, zero_r);
    Value hy = arith::SelectOp::create(b, loc, eight_lanes, sy4, zero_r);
    Value hz = arith::SelectOp::create(b, loc, eight_lanes, sz4, zero_r);
    Value allx = arith::AddFOp::create(b, loc, ux, hx);
    Value ally = arith::AddFOp::create(b, loc, uy, hy);
    Value allz = arith::AddFOp::create(b, loc, uz, hz);
    Value first =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, j3, c0);
    Value stores;
    if constexpr (!useList) {
      stores = arith::AndIOp::create(b, loc, inside, first);
    } else {
      Value stores_any = arith::AndIOp::create(b, loc, inside, first);
      stores = arith::AndIOp::create(b, loc, stores_any, is_listed);
    }
    auto storesBranch =
        scf::IfOp::create(b, loc, TypeRange{}, stores, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(storesBranch.thenBlock());
      Value mq;
      if constexpr (!potentialOnly) {
        mq = arith::NegFOp::create(b, loc, qi);
      }
      Value fx64;
      Value fy64;
      Value fz64;
      if constexpr (!tilted && !potentialOnly) {
        Value sxq = arith::MulFOp::create(b, loc, mq, rxr);
        Value syq = arith::MulFOp::create(b, loc, mq, ryr);
        Value szq = arith::MulFOp::create(b, loc, mq, rzr);
        fx64 = arith::MulFOp::create(b, loc, sxq, allx);
        fy64 = arith::MulFOp::create(b, loc, syq, ally);
        fz64 = arith::MulFOp::create(b, loc, szq, allz);
      }
      if constexpr (tilted && !potentialOnly) {
        Value ka1 = arith::MulFOp::create(b, loc, k1fr, allx);
        Value ka2 = arith::MulFOp::create(b, loc, k2fr, ally);
        Value ka3 = arith::MulFOp::create(b, loc, k3fr, allz);
        Value fxh = arith::MulFOp::create(b, loc, h00r, ka1);
        Value fyh1 = arith::MulFOp::create(b, loc, h10r, ka1);
        Value fyh2 = arith::MulFOp::create(b, loc, h11r, ka2);
        Value fyh = arith::AddFOp::create(b, loc, fyh1, fyh2);
        Value fzh1 = arith::MulFOp::create(b, loc, h20r, ka1);
        Value fzh2 = arith::MulFOp::create(b, loc, h21r, ka2);
        Value fzh3 = arith::MulFOp::create(b, loc, h22r, ka3);
        Value fzh12 = arith::AddFOp::create(b, loc, fzh1, fzh2);
        Value fzh = arith::AddFOp::create(b, loc, fzh12, fzh3);
        fx64 = arith::MulFOp::create(b, loc, mq, fxh);
        fy64 = arith::MulFOp::create(b, loc, mq, fyh);
        fz64 = arith::MulFOp::create(b, loc, mq, fzh);
      }
      if constexpr (!potentialOnly) {
        Value fxs = convert(realType, fx64);
        Value fys = convert(realType, fy64);
        Value fzs = convert(realType, fz64);
        store(fxs, f, {i, i0});
        store(fys, f, {i, i1});
        store(fzs, f, {i, i2});
      } else {
        Value pots = convert(realType, allx);
        store(pots, f, {i});
      }
      scf::YieldOp::create(b, loc);
    }
    if constexpr (useList) {
      scf::YieldOp::create(b, loc);
      b.setInsertionPointAfter(warp_activeBranch);
      Value ztrue = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
      Value zunlisted = arith::XOrIOp::create(b, loc, is_listed, ztrue);
      Value zfirst =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane_of, c0);
      Value zin = arith::AndIOp::create(b, loc, inside, zfirst);
      Value zstores = arith::AndIOp::create(b, loc, zin, zunlisted);
      auto zstoresBranch =
          scf::IfOp::create(b, loc, TypeRange{}, zstores, true, false);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(zstoresBranch.thenBlock());
        Value zminus =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, -1));
        Value zplace32 = arith::SubIOp::create(b, loc, zminus, listed);
        Value zplace = arith::IndexCastOp::create(b, loc, idxType, zplace32);
        Value zc1 = arith::ConstantIndexOp::create(b, loc, 1);
        Value zc2 = arith::ConstantIndexOp::create(b, loc, 2);
        Value zero_f =
            arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
        store(zero_f, f, {zplace, c0});
        store(zero_f, f, {zplace, zc1});
        store(zero_f, f, {zplace, zc2});
        scf::YieldOp::create(b, loc);
      }
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitGather<false, false, false, false>(StringRef);
template void DevicePME::emitGather<false, false, false, true>(StringRef);
template void DevicePME::emitGather<false, false, true, false>(StringRef);
template void DevicePME::emitGather<false, true, false, false>(StringRef);
template void DevicePME::emitGather<false, true, false, true>(StringRef);
template void DevicePME::emitGather<true, false, false, false>(StringRef);
template void DevicePME::emitGather<true, false, false, true>(StringRef);
template void DevicePME::emitGather<true, false, true, false>(StringRef);
template void DevicePME::emitGather<true, true, false, false>(StringRef);
template void DevicePME::emitGather<true, true, false, true>(StringRef);
