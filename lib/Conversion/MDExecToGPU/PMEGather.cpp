// Interpolate from the potential grid, using derivatives for forces
// [Essmann1995].
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <bool tilted, bool useList, bool potentialOnly, bool cachedWeights>
void DevicePME::emitGather(StringRef symbol, bool ordered, int orderedLanes) {
  const int gatherLanes = ordered ? orderedLanes : lanes;
  StringRef blockFunction =
      tilted ? "mdrt_gpu_pme_blocks_triclinic" : "mdrt_gpu_pme_blocks";
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(ordered ? records() : globalMem({ShapedType::kDynamic, 3}, positionType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, ordered ? i32Type : chargeType));
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
  if (cachedWeights && !ordered) {
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
  if (ordered)
    inputs.push_back(idxType);
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
  if (cachedWeights && !ordered) {
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
  Value lx = extract(box, 0);
  Value ly = extract(box, 1);
  Value lz = extract(box, 2);
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
    std::tie(h00, h10, h11, h20, h21, h22) = inverseCell(box, lx, ly, lz);
    h00r = convert(realType, h00);
    h10r = convert(realType, h10);
    h11r = convert(realType, h11);
    h20r = convert(realType, h20);
    h21r = convert(realType, h21);
    h22r = convert(realType, h22);
  }
  Value k1i = indexCast(i64Type, k1);
  Value k2i = indexCast(i64Type, k2);
  Value k3i = indexCast(i64Type, k3);
  Value k1f = sitofp(f64Type, k1i);
  Value k2f = sitofp(f64Type, k2i);
  Value k3f = sitofp(f64Type, k3i);
  Value rx = divf(k1f, lx);
  Value ry = divf(k2f, ly);
  Value rz = divf(k3f, lz);
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
    ilx = divf(unit, lx);
    ily = divf(unit, ly);
    ilz = divf(unit, lz);
  }
  Value count = memref::DimOp::create(b, loc, x, c0);
  Value lanes_all =
      arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, gatherLanes));
  Value points = muli(count, lanes_all);
  Value last = subi(count, c1);
  Value threads = ordered ? entry->getArguments().back() : c128;
  Value blocks_points = ordered
      ? maxui(c1, arith::CeilDivUIOp::create(b, loc, points, threads))
      : func::CallOp::create(b, loc, blockFunction, TypeRange{idxType},
                           ValueRange{points})
          .getResult(0);
  {
    auto launch =
        gpu::LaunchOp::create(b, loc, blocks_points, c1, c1, threads, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value base = muli(bx, threads);
    Value item0 = addi(base, tx);
    Value inside = cmpi(arith::CmpIPredicate::ult, item0, points);
    Value lanes_l =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, gatherLanes));
    Value lane_of = remui(item0, lanes_l);
    Value last_first = muli(last, lanes_l);
    Value last_item = addi(last_first, lane_of);
    Value item = select(inside, item0, last_item);
    Value laneCount =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, gatherLanes));
    Value i;
    if constexpr (!useList) {
      i = divui(item, laneCount);
    }
    Value slot;
    if constexpr (useList) {
      slot = divui(item, laneCount);
    }
    Value j3 = remui(item, laneCount);
    if (ordered && gatherLanes == 2)
      j3 = addi(j3, j3);
    Value listed;
    Value is_listed;
    scf::IfOp warp_activeBranch;
    if constexpr (useList) {
      listed = load(list, {slot});
      Value listed_none =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      is_listed = cmpi(arith::CmpIPredicate::sge, listed, listed_none);
      Value c32w = arith::ConstantIndexOp::create(b, loc, 32);
      Value lane32 = remui(item0, c32w);
      Value wfirst = subi(item0, lane32);
      Value wslot = divui(wfirst, lanes_l);
      Value wslot_in = cmpi(arith::CmpIPredicate::ult, wslot, count);
      Value wslot_c = select(wslot_in, wslot, c0);
      Value warp_listed = load(list, {wslot_c});
      Value wnone =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      Value wlisted = cmpi(arith::CmpIPredicate::sge, warp_listed, wnone);
      Value warp_active = andi(wslot_in, wlisted);
      warp_activeBranch =
          scf::IfOp::create(b, loc, TypeRange{}, warp_active, true, false);
      b.setInsertionPointToStart(warp_activeBranch.thenBlock());
      Value listed_or = select(is_listed, listed, warp_listed);
      i = indexCast(idxType, listed_or);
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
    const bool cubicForce = order == 4 && !potentialOnly;
    std::array<Value, 4> cubicX, cubicY, cubicZ, slopeX, slopeY, slopeZ;
    auto prepareAxis = [&](Value fraction, Value weights, Value slopes,
                           std::array<Value, 4> &w, std::array<Value, 4> &dw) {
      if (cubicForce)
        w = cubicWeights(fraction, &dw);
      else
        splineWeights(fraction, pointCount, weights, slopes);
    };
    Value xs;
    Value ys;
    Value zs;
    if constexpr (!cachedWeights) {
      xs = load(x, {i, i0});
      ys = load(x, {i, i1});
      zs = load(x, {i, i2});
    }
    Value packed = ordered ? readRecord(x, i) : Value();
    Value qs = ordered ? extract(packed, 3) : load(q, {i});
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
    FractionalPosition fractional;
    if constexpr (tilted && !cachedWeights) {
      fr_one = arith::ConstantOp::create(b, loc, b.getFloatAttr(f64Type, 1.0));
      fractional = fractionalPosition(xi, yi, zi, h00, h10, h11, h20, h21, h22);
      xPlacement = placeAxis(fractional.x, fr_one, k1, pointCount);
    }
    Value px_start;
    if constexpr (!cachedWeights) {
      px_start = xPlacement.first;
      Value px_w = xPlacement.second;
      prepareAxis(px_w, wx, dx, cubicX, slopeX);
    }
    std::pair<Value, Value> yPlacement;
    if constexpr (!tilted && !cachedWeights) {
      yPlacement = placeAxis(yi, ily, k2, pointCount);
    }
    if constexpr (tilted && !cachedWeights) {
      yPlacement = placeAxis(fractional.y, fr_one, k2, pointCount);
    }
    Value py_start;
    if constexpr (!cachedWeights) {
      py_start = yPlacement.first;
      Value py_w = yPlacement.second;
      prepareAxis(py_w, wy, dy, cubicY, slopeY);
    }
    std::pair<Value, Value> zPlacement;
    if constexpr (!tilted && !cachedWeights) {
      zPlacement = placeAxis(zi, ilz, k3, pointCount);
    }
    if constexpr (tilted && !cachedWeights) {
      zPlacement = placeAxis(fractional.z, fr_one, k3, pointCount);
    }
    Value pz_start;
    if constexpr (!cachedWeights) {
      pz_start = zPlacement.first;
      Value pz_w = zPlacement.second;
      prepareAxis(pz_w, wz, dz, cubicZ, slopeZ);
    }
    Value xstart;
    Value ystart;
    Value zstart;
    if constexpr (cachedWeights) {
      auto axis = [&](int a, Value size, auto &w, auto &dw, Value ws, Value ds) {
        Value start, fraction;
        if (ordered) {
          std::tie(start, fraction) = splitMeshAxis(extract(packed, a), size, pointCount, true);
          start = indexCast(i32Type, start);
        } else {
          start = placementStart(weights, count, i, a);
          Value axis = arith::ConstantIndexOp::create(b, loc, a + 3);
          fraction = load(weights, {addi(muli(axis, count), i)});
        }
        prepareAxis(fraction, ws, ds, w, dw);
        return start;
      };
      xstart = axis(0, k1, cubicX, slopeX, wx, dx);
      ystart = axis(1, k2, cubicY, slopeY, wy, dy);
      zstart = axis(2, k3, cubicZ, slopeZ, wz, dz);
    }
    Value w3p_value =
        cubicForce ? cubicLane(cubicZ, j3) : pickSpline(wz, j3, pointCount);
    Value d3p_value =
        cubicForce ? cubicLane(slopeZ, j3) : pickSpline(dz, j3, pointCount);
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
      k1t = indexCast(i32Type, k1);
      k2t = indexCast(i32Type, k2);
      k3t = indexCast(i32Type, k3);
      Value j3t = indexCast(i32Type, j3);
      g3 = md_alg::WrapOp::create(b, loc, zstart, j3t, k3t, false);
    }
    Value sumx, sumy, sumz;
    if (cubicForce) {
      // Contract each z plane along y first so the three derivatives
      // share each row of potential samples.
      auto wrapped = [&](Value start, int offset, Value size) {
        Value j = arith::ConstantOp::create(
            b, loc, b.getIntegerAttr(size.getType(), offset));
        return md_alg::WrapOp::create(b, loc, start, j, size, false).getResult();
      };
      Value nx = cachedWeights ? k1t : k1;
      Value ny = cachedWeights ? k2t : k2;
      Value nz = cachedWeights ? k3t : k3;
      Value sx = cachedWeights ? xstart : px_start;
      Value sy = cachedWeights ? ystart : py_start;
      auto contract = [&](Value z, Value w, Value dw) {
        Value dxPlane = gzero, dyPlane = gzero, plane = gzero;
        for (int xj = 0; xj < 4; ++xj) {
          Value xrow = muli(wrapped(sx, xj, nx), ny);
          Value row = gzero, dyRow = gzero;
          for (int yj = 0; yj < 4; ++yj) {
            Value base = muli(addi(xrow, wrapped(sy, yj, ny)), nz);
            Value sample =
                md_alg::GridReadOp::create(b, loc, phi, base, z, Value{});
            row = addf(row, mulf(cubicY[yj], sample));
            dyRow = addf(dyRow, mulf(slopeY[yj], sample));
          }
          dxPlane = addf(dxPlane, mulf(slopeX[xj], row));
          dyPlane = addf(dyPlane, mulf(cubicX[xj], dyRow));
          plane = addf(plane, mulf(cubicX[xj], row));
        }
        return std::array<Value, 3>{mulf(dxPlane, w), mulf(dyPlane, w),
                                    mulf(plane, dw)};
      };
      auto force = contract(g3, w3p_value, d3p_value);
      if (ordered && gatherLanes == 2) {
        Value next = addi(j3, c1);
        auto second = contract(wrapped(g3, 1, nz), cubicLane(cubicZ, next),
                               cubicLane(slopeZ, next));
        for (int axis = 0; axis < 3; ++axis)
          force[axis] = addf(force[axis], second[axis]);
      }
      sumx = force[0];
      sumy = force[1];
      sumz = force[2];
    } else {
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
              g1s = addi(px_start, j1);
              g1w = subi(g1s, k1);
              g1over = cmpi(arith::CmpIPredicate::uge, g1s, k1);
            }
            if constexpr (cachedWeights) {
              Value j1t = indexCast(i32Type, j1);
              g1s = addi(xstart, j1t);
              g1w = subi(g1s, k1t);
              g1over = cmpi(arith::CmpIPredicate::uge, g1s, k1t);
            }
            Value g1 = select(g1over, g1w, g1s);
            Value row1;
            if constexpr (cachedWeights) {
              row1 = muli(g1, k2t);
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
                    g2s = addi(py_start, j2);
                    g2w = subi(g2s, k2);
                    g2over = cmpi(arith::CmpIPredicate::uge, g2s, k2);
                  }
                  if constexpr (cachedWeights) {
                    Value j2t = indexCast(i32Type, j2);
                    g2s = addi(ystart, j2t);
                    g2w = subi(g2s, k2t);
                    g2over = cmpi(arith::CmpIPredicate::uge, g2s, k2t);
                  }
                  Value g2 = select(g2over, g2w, g2s);
                  Value w2 = load(wy, {j2});
                  Value d2 = load(dy, {j2});
                  if constexpr (!cachedWeights) {
                    row1 = muli(g1, k2);
                  }
                  Value row = addi(row1, g2);
                  Value gbase;
                  if constexpr (!cachedWeights) {
                    gbase = muli(row, k3);
                  }
                  if constexpr (cachedWeights) {
                    gbase = muli(row, k3t);
                  }
                  Value p =
                      md_alg::GridReadOp::create(b, loc, phi, gbase, g3, Value{});
                  Value d1w2 = mulf(d1, w2);
                  Value w1d2 = mulf(w1, d2);
                  Value w1w2 = mulf(w1, w2);
                  Value px;
                  if constexpr (!potentialOnly) {
                    px = mulf(d1w2, w3p_value);
                  } else {
                    px = mulf(w1w2, w3p_value);
                  }
                  Value py = mulf(w1d2, w3p_value);
                  Value pz = mulf(w1w2, d3p_value);
                  Value vx = mulf(p, px);
                  Value vy = mulf(p, py);
                  Value vz = mulf(p, pz);
                  Value nx = addf(cx, vx);
                  Value ny = addf(cy, vy);
                  Value nz = addf(cz, vz);
                  scf::YieldOp::create(b, loc, ValueRange{nx, ny, nz});
                });
            Value bx2 = bx2Op.getResult(0);
            Value by2 = bx2Op.getResult(1);
            Value bz2 = bx2Op.getResult(2);
            scf::YieldOp::create(b, loc, ValueRange{bx2, by2, bz2});
          });
      sumx = sumxOp.getResult(0);
      sumy = sumxOp.getResult(1);
      sumz = sumxOp.getResult(2);
    }
    Value width =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
    Value zero_r = gzero;
    auto reduce = [&](Value value) {
      for (int offset = 1; offset < gatherLanes; offset *= 2) {
        Value hop = arith::ConstantIntOp::create(b, loc, offset, 32);
        value = addf(value, laneValue(gpu::ShuffleMode::XOR, value, hop, width));
      }
      return !ordered && gatherLanes == 4 ? addf(value, zero_r) : value;
    };
    Value allx = reduce(sumx), ally = reduce(sumy), allz = reduce(sumz);
    Value first = cmpi(arith::CmpIPredicate::eq, j3, c0);
    Value stores;
    if constexpr (!useList) {
      stores = andi(inside, first);
    } else {
      Value stores_any = andi(inside, first);
      stores = andi(stores_any, is_listed);
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
        Value sxq = mulf(mq, rxr);
        Value syq = mulf(mq, ryr);
        Value szq = mulf(mq, rzr);
        fx64 = mulf(sxq, allx);
        fy64 = mulf(syq, ally);
        fz64 = mulf(szq, allz);
      }
      if constexpr (tilted && !potentialOnly) {
        Value ka1 = mulf(k1fr, allx);
        Value ka2 = mulf(k2fr, ally);
        Value ka3 = mulf(k3fr, allz);
        Value fxh = mulf(h00r, ka1);
        Value fyh1 = mulf(h10r, ka1);
        Value fyh2 = mulf(h11r, ka2);
        Value fyh = addf(fyh1, fyh2);
        Value fzh1 = mulf(h20r, ka1);
        Value fzh2 = mulf(h21r, ka2);
        Value fzh3 = mulf(h22r, ka3);
        Value fzh12 = addf(fzh1, fzh2);
        Value fzh = addf(fzh12, fzh3);
        fx64 = mulf(mq, fxh);
        fy64 = mulf(mq, fyh);
        fz64 = mulf(mq, fzh);
      }
      if constexpr (!potentialOnly) {
        Value fxs = convert(realType, fx64);
        Value fys = convert(realType, fy64);
        Value fzs = convert(realType, fz64);
        Value atom = ordered ? indexCast(idxType, load(q, {i})) : i;
        store(fxs, f, {atom, i0});
        store(fys, f, {atom, i1});
        store(fzs, f, {atom, i2});
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
      Value zunlisted = xori(is_listed, ztrue);
      Value zfirst = cmpi(arith::CmpIPredicate::eq, lane_of, c0);
      Value zin = andi(inside, zfirst);
      Value zstores = andi(zin, zunlisted);
      auto zstoresBranch =
          scf::IfOp::create(b, loc, TypeRange{}, zstores, true, false);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(zstoresBranch.thenBlock());
        Value zminus =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, -1));
        Value zplace32 = subi(zminus, listed);
        Value zplace = indexCast(idxType, zplace32);
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

template void DevicePME::emitGather<false, false, false, false>(StringRef, bool, int);
template void DevicePME::emitGather<false, false, false, true>(StringRef, bool, int);
template void DevicePME::emitGather<false, false, true, false>(StringRef, bool, int);
template void DevicePME::emitGather<false, true, false, false>(StringRef, bool, int);
template void DevicePME::emitGather<false, true, false, true>(StringRef, bool, int);
template void DevicePME::emitGather<true, false, false, false>(StringRef, bool, int);
template void DevicePME::emitGather<true, false, false, true>(StringRef, bool, int);
template void DevicePME::emitGather<true, false, true, false>(StringRef, bool, int);
template void DevicePME::emitGather<true, true, false, false>(StringRef, bool, int);
template void DevicePME::emitGather<true, true, false, true>(StringRef, bool, int);
