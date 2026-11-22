// Interpolate the prolate window with one warp per particle [Liang2026].
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <bool tilted, bool useList, bool potentialOnly>
void DevicePME::emitWarpGather(StringRef symbol) {
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(globalMem({ShapedType::kDynamic}, chargeType));
  if constexpr (useList) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, i32Type));
  }
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  if constexpr (!tilted) {
    inputs.push_back(VectorType::get({3}, f64Type));
  } else {
    inputs.push_back(VectorType::get({6}, f64Type));
  }
  inputs.append(3, idxType);
  if constexpr (!potentialOnly) {
    inputs.push_back(globalMem({ShapedType::kDynamic, 3}, realType));
  } else {
    inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  }
  auto fn = func::FuncOp::create(b, loc, symbol, b.getFunctionType(inputs, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  unsigned arg = 0;
  Value q = entry->getArgument(arg++);
  Value list;
  if constexpr (useList) {
    list = entry->getArgument(arg++);
  }
  Value places = entry->getArgument(arg++);
  Value tables = entry->getArgument(arg++);
  Value phi = entry->getArgument(arg++);
  Value box = entry->getArgument(arg++);
  Value k1 = entry->getArgument(arg++);
  Value k2 = entry->getArgument(arg++);
  Value k3 = entry->getArgument(arg++);
  Value f = entry->getArgument(arg++);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value c3 = arith::ConstantIndexOp::create(b, loc, 3);
  Value c4 = arith::ConstantIndexOp::create(b, loc, 4);
  arith::ConstantIndexOp::create(b, loc, 16);
  arith::ConstantIndexOp::create(b, loc, 32);
  Value c128 = arith::ConstantIndexOp::create(b, loc, 128);
  Value places_length = memref::DimOp::create(b, loc, places, c0);
  Value c6 = arith::ConstantIndexOp::create(b, loc, 6);
  Value n_all = divui(places_length, c6);
  Value k12 = addi(k1, k2);
  Value k123 = addi(k12, k3);
  Value pe = muli(k123, c3);
  Value warps = addi(n_all, c3);
  Value blocks = divui(warps, c4);
  Value lx = extract(box, 0);
  Value ly = extract(box, 1);
  Value lz = extract(box, 2);
  Value h00r;
  Value h10r;
  Value h11r;
  Value h20r;
  Value h21r;
  Value h22r;
  if constexpr (tilted) {
    auto [h00, h10, h11, h20, h21, h22] = inverseCell(box, lx, ly, lz);
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
  {
    auto launch = gpu::LaunchOp::create(b, loc, blocks, c1, c1, c128, c1, c1);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value c0w = arith::ConstantIndexOp::create(b, loc, 0);
    Value c32w = arith::ConstantIndexOp::create(b, loc, 32);
    Value two_i = arith::ConstantIndexOp::create(b, loc, 2);
    Value three_i = arith::ConstantIndexOp::create(b, loc, 3);
    Value four_w = arith::ConstantIndexOp::create(b, loc, 4);
    Value warp = divui(tx, c32w);
    Value lane_x = remui(tx, c32w);
    Value lane = indexCast(i32Type, lane_x);
    Value wb = muli(bx, four_w);
    Value i;
    Value inside;
    if constexpr (!useList) {
      i = addi(wb, warp);
      inside = cmpi(arith::CmpIPredicate::ult, i, n_all);
    }
    Value slot;
    if constexpr (useList) {
      slot = addi(wb, warp);
      inside = cmpi(arith::CmpIPredicate::ult, slot, n_all);
    }
    auto insideBranch =
        scf::IfOp::create(b, loc, TypeRange{}, inside, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(insideBranch.thenBlock());
      Value listed;
      scf::IfOp is_listedBranch;
      if constexpr (useList) {
        listed = load(list, {slot});
        Value listed_none =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
        Value is_listed = cmpi(arith::CmpIPredicate::sge, listed, listed_none);
        is_listedBranch =
            scf::IfOp::create(b, loc, TypeRange{}, is_listed, true, true);
        b.setInsertionPointToStart(is_listedBranch.thenBlock());
        i = indexCast(idxType, listed);
      }
      auto [wvalue, wslope, axis, zero_w, one] =
          laneWindow(lane, i, places, tables, n_all, pe, c0w, two_i, three_i,
                     !potentialOnly);
      Value xstart = placementStart(places, n_all, i, 0);
      Value ystart = placementStart(places, n_all, i, 1);
      Value zstart = placementStart(places, n_all, i, 2);
      Value k1t = indexCast(i32Type, k1);
      Value k2t = indexCast(i32Type, k2);
      Value k3t = indexCast(i32Type, k3);
      Value pt =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, order));
      Value ppt = muli(pt, pt);
      Value pt2 = addi(pt, pt);
      Value thirty_two =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
      Value zero32 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      Value width =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
      Value zr =
          arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
      Value offsets[8], weights[8], slopes[8];
      for (int j = 0; j < 8; ++j) {
        offsets[j] =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, j));
        Value at = addi(pt2, offsets[j]);
        weights[j] = laneValue(gpu::ShuffleMode::IDX, wvalue, at, width);
        if constexpr (!potentialOnly)
          slopes[j] = laneValue(gpu::ShuffleMode::IDX, wslope, at, width);
      }
      Value passes_n =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
      Value passes_n2 = muli(passes_n, passes_n);
      Value c31p = arith::ConstantIndexOp::create(b, loc, 31);
      Value passes_u = addi(passes_n2, c31p);
      Value passes = divui(passes_u, c32w);
      constexpr unsigned components = potentialOnly ? 1 : 3;
      SmallVector<Value> initial(components, zr);
      auto sums = scf::ForOp::create(
          b, loc, c0w, passes, one, initial,
          [&](OpBuilder &b, Location, Value pass, ValueRange carried) {
            Value pass32 = indexCast(i32Type, pass);
            Value pass_base = muli(pass32, thirty_two);
            Value ab = addi(pass_base, lane);
            Value in = cmpi(arith::CmpIPredicate::slt, ab, ppt);
            Value abc = select(in, ab, zero32);
            Value pa = divui(abc, pt);
            Value pb = remui(abc, pt);
            Value pbs = addi(pb, pt);
            Value wx = laneValue(gpu::ShuffleMode::IDX, wvalue, pa, width);

            Value wy = laneValue(gpu::ShuffleMode::IDX, wvalue, pbs, width);

            Value dx, dy;
            if constexpr (!potentialOnly) {
              dx = laneValue(gpu::ShuffleMode::IDX, wslope, pa, width);
              dy = laneValue(gpu::ShuffleMode::IDX, wslope, pbs, width);
            }
            auto row_sumsResult0Op = scf::IfOp::create(
                b, loc, TypeRange{realType, realType}, in, true, true);
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(row_sumsResult0Op.thenBlock());
              Value g1s = addi(xstart, pa);
              Value g1w = subi(g1s, k1t);
              Value g1over = cmpi(arith::CmpIPredicate::sge, g1s, k1t);
              Value g1 = select(g1over, g1w, g1s);
              Value g2s = addi(ystart, pb);
              Value g2w = subi(g2s, k2t);
              Value g2over = cmpi(arith::CmpIPredicate::sge, g2s, k2t);
              Value g2 = select(g2over, g2w, g2s);
              Value row1 = muli(g1, k2t);
              Value row = addi(row1, g2);
              Value rowk = muli(row, k3t);
              Value weighted = zr, derivative = zr;
              for (int j = 0; j < 8; ++j) {
                Value active = cmpi(arith::CmpIPredicate::slt, offsets[j], pt);
                Value z = md_alg::WrapOp::create(b, loc, zstart, offsets[j],
                                                 k3t, true);
                Value p =
                    md_alg::GridReadOp::create(b, loc, phi, rowk, z, active);
                auto accumulate = [&](Value sum, Value w) -> Value {
                  Value product = mulf(p, w);
                  Value term = select(active, product, zr);
                  return addf(sum, term);
                };
                weighted = accumulate(weighted, weights[j]);
                if constexpr (!potentialOnly)
                  derivative = accumulate(derivative, slopes[j]);
              }
              scf::YieldOp::create(b, loc, ValueRange{weighted, derivative});
            }
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(row_sumsResult0Op.elseBlock());
              scf::YieldOp::create(b, loc, ValueRange{zr, zr});
            }
            Value row_sumsResult0 = row_sumsResult0Op.getResult(0);
            Value row_sumsResult1 = row_sumsResult0Op.getResult(1);
            if constexpr (potentialOnly) {
              Value wxy = mulf(wx, wy);
              Value term = mulf(wxy, row_sumsResult0);
              Value next = addf(carried[0], term);
              scf::YieldOp::create(b, loc, ValueRange{next});
            } else {
              Value dxwy = mulf(dx, wy);
              Value wxdy = mulf(wx, dy);
              Value wxwy = mulf(wx, wy);
              Value tx_ = mulf(dxwy, row_sumsResult0);
              Value ty_ = mulf(wxdy, row_sumsResult0);
              Value tz_ = mulf(wxwy, row_sumsResult1);
              Value nx = addf(carried[0], tx_);
              Value ny = addf(carried[1], ty_);
              Value nz = addf(carried[2], tz_);
              scf::YieldOp::create(b, loc, ValueRange{nx, ny, nz});
            }
          });
      SmallVector<Value> reduced(sums.getResults());
      for (int mask = 16; mask; mask /= 2) {
        Value offset =
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, mask));
        for (Value &component : reduced) {
          Value peer =
              laneValue(gpu::ShuffleMode::XOR, component, offset, width);
          component = addf(component, peer);
        }
      }
      Value lane0 = cmpi(arith::CmpIPredicate::eq, lane, zero32);
      auto lane0Branch =
          scf::IfOp::create(b, loc, TypeRange{}, lane0, true, false);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(lane0Branch.thenBlock());
        Value mq;
        if constexpr (!potentialOnly) {
          Value qs = load(q, {i});
          Value qi = convert(realType, qs);
          mq = arith::NegFOp::create(b, loc, qi);
        }
        Value fxv;
        Value fyv;
        Value fzv;
        if constexpr (!tilted && !potentialOnly) {
          Value sxq = mulf(mq, rxr);
          Value syq = mulf(mq, ryr);
          Value szq = mulf(mq, rzr);
          fxv = mulf(sxq, reduced[0]);
          fyv = mulf(syq, reduced[1]);
          fzv = mulf(szq, reduced[2]);
        }
        if constexpr (tilted && !potentialOnly) {
          Value k1fr0 = sitofp(realType, k1t);
          Value k2fr0 = sitofp(realType, k2t);
          Value k3fr0 = sitofp(realType, k3t);
          Value ka1 = mulf(k1fr0, reduced[0]);
          Value ka2 = mulf(k2fr0, reduced[1]);
          Value ka3 = mulf(k3fr0, reduced[2]);
          Value fxh = mulf(h00r, ka1);
          Value fyh1 = mulf(h10r, ka1);
          Value fyh2 = mulf(h11r, ka2);
          Value fyh = addf(fyh1, fyh2);
          Value fzh1 = mulf(h20r, ka1);
          Value fzh2 = mulf(h21r, ka2);
          Value fzh3 = mulf(h22r, ka3);
          Value fzh12 = addf(fzh1, fzh2);
          Value fzh = addf(fzh12, fzh3);
          fxv = mulf(mq, fxh);
          fyv = mulf(mq, fyh);
          fzv = mulf(mq, fzh);
        }
        if constexpr (!potentialOnly) {
          Value fxs = convert(realType, fxv);
          Value fys = convert(realType, fyv);
          Value fzs = convert(realType, fzv);
          Value i0 = arith::ConstantIndexOp::create(b, loc, 0);
          Value i1 = arith::ConstantIndexOp::create(b, loc, 1);
          Value i2 = arith::ConstantIndexOp::create(b, loc, 2);
          store(fxs, f, {i, i0});
          store(fys, f, {i, i1});
          store(fzs, f, {i, i2});
        } else {
          Value stored = convert(realType, reduced[0]);
          store(stored, f, {i});
        }
        scf::YieldOp::create(b, loc);
      }
      scf::YieldOp::create(b, loc);
      if constexpr (useList) {
        b.setInsertionPointAfter(is_listedBranch);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(is_listedBranch.elseBlock());
          Value zfirst = cmpi(arith::CmpIPredicate::eq, lane_x, c0w);
          auto zfirstBranch =
              scf::IfOp::create(b, loc, TypeRange{}, zfirst, true, false);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(zfirstBranch.thenBlock());
            Value zminus = arith::ConstantOp::create(
                b, loc, b.getIntegerAttr(i32Type, -1));
            Value zplace32 = subi(zminus, listed);
            Value zplace = indexCast(idxType, zplace32);
            Value zc1 = arith::ConstantIndexOp::create(b, loc, 1);
            Value zc2 = arith::ConstantIndexOp::create(b, loc, 2);
            Value zero_f = arith::ConstantOp::create(
                b, loc, b.getFloatAttr(realType, 0.0));
            store(zero_f, f, {zplace, c0w});
            store(zero_f, f, {zplace, zc1});
            store(zero_f, f, {zplace, zc2});
            scf::YieldOp::create(b, loc);
          }
          scf::YieldOp::create(b, loc);
        }
        scf::YieldOp::create(b, loc);
      }
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitWarpGather<false, false, false>(StringRef);
template void DevicePME::emitWarpGather<false, false, true>(StringRef);
template void DevicePME::emitWarpGather<true, false, false>(StringRef);
template void DevicePME::emitWarpGather<true, false, true>(StringRef);
template void DevicePME::emitWarpGather<false, true, false>(StringRef);
template void DevicePME::emitWarpGather<true, true, false>(StringRef);
