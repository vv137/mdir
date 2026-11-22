// Bin-local charge spreading, staged in shared memory, with a global overflow
// (D282).
#include "DevicePME.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;

template <Window window, int tileEdge>
void DevicePME::emitSpreadTile(StringRef symbol) {
  constexpr int tileThreads = tileEdge == 16 ? 512 : 128;
  OpBuilder::InsertionGuard functionGuard(b);
  SmallVector<Type> inputs;
  inputs.push_back(globalMem({ShapedType::kDynamic}, chargeType));
  if constexpr (window == Window::Prolate) {
    inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  }
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.push_back(globalMem({ShapedType::kDynamic}, i32Type));
  inputs.push_back(globalMem({ShapedType::kDynamic}, realType));
  inputs.append(3, idxType);
  auto fn = func::FuncOp::create(b, loc, symbol, b.getFunctionType(inputs, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  unsigned arg = 0;
  Value q = entry->getArgument(arg++);
  Value places = entry->getArgument(arg++);
  Value tables;
  if constexpr (window == Window::Prolate) {
    tables = entry->getArgument(arg++);
  }
  Value bins = entry->getArgument(arg++);
  Value realBuffer = entry->getArgument(arg++);
  Value k1 = entry->getArgument(arg++);
  Value k2 = entry->getArgument(arg++);
  Value k3 = entry->getArgument(arg++);
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  arith::ConstantIndexOp::create(b, loc, 128);
  Value n_all = memref::DimOp::create(b, loc, q, c0);
  Value edge = arith::ConstantIndexOp::create(b, loc, tileEdge);
  arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, tileEdge));
  Value edge_m1 = arith::ConstantIndexOp::create(b, loc, tileEdge - 1);
  Value k1e = addi(k1, edge_m1);
  Value nb1 = divui(k1e, edge);
  Value k2e = addi(k2, edge_m1);
  Value nb2 = divui(k2e, edge);
  Value k3e = addi(k3, edge_m1);
  Value nb3 = divui(k3e, edge);
  Value nb12 = muli(nb1, nb2);
  Value nbins = muli(nb12, nb3);
  Value nbins_m1 = subi(nbins, c1);
  Value mean_u = addi(n_all, nbins_m1);
  Value mean = divui(mean_u, nbins);
  Value mean2 = addi(mean, mean);
  Value slack = arith::ConstantIndexOp::create(b, loc, 8);
  Value capacity = addi(mean2, slack);
  Value of_lists = addi(nbins, c1);
  Value lists = muli(nbins, capacity);
  Value of_overflow = addi(of_lists, lists);
  Value cthreads = arith::ConstantIndexOp::create(b, loc, tileThreads);
  Value blocks_tiles = addi(nbins, c1);
  {
    auto launch = gpu::LaunchOp::create(
        b, loc, blocks_tiles, c1, c1, cthreads, c1, c1, Value{}, Type{},
        ValueRange{},
        TypeRange{tileType(tileEdge), shared(4 * tileThreads, realType),
                  shared(3 * tileThreads, i32Type)});
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;
    Value tx = launch.getThreadIds().x;
    Value tile = launch.getWorkgroupAttributionBBArgs()[0];
    Value staged = launch.getWorkgroupAttributionBBArgs()[1];
    Value starts = launch.getWorkgroupAttributionBBArgs()[2];
    Value c3w = arith::ConstantIndexOp::create(b, loc, 3);
    Value c32w = arith::ConstantIndexOp::create(b, loc, 32);
    Value cwarps = arith::ConstantIndexOp::create(b, loc, tileThreads / 32);
    Value cthreads2 = arith::ConstantIndexOp::create(b, loc, 2 * tileThreads);
    Value zero_r =
        arith::ConstantOp::create(b, loc, b.getFloatAttr(realType, 0.0));
    Value true_w = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
    Value side32 = arith::ConstantOp::create(
        b, loc, b.getIntegerAttr(i32Type, tileSide(tileEdge)));
    Value volume = arith::ConstantOp::create(
        b, loc, b.getIntegerAttr(idxType, tilePoints(tileEdge)));
    Value overflows = cmpi(arith::CmpIPredicate::eq, bx, nbins);
    Value bins_block = xori(overflows, true_w);
    Value bin = select(overflows, c0, bx);
    Value b3 = remui(bin, nb3);
    Value b12 = divui(bin, nb3);
    Value b2 = remui(b12, nb2);
    Value b1 = divui(b12, nb2);
    Value o1 = muli(b1, edge);
    Value o2 = muli(b2, edge);
    Value o3 = muli(b3, edge);
    Value o1t = indexCast(i32Type, o1);
    Value o2t = indexCast(i32Type, o2);
    Value o3t = indexCast(i32Type, o3);
    Value counted32 = load(bins, {bx});
    Value counted = indexCast(idxType, counted32);
    Value in_run = minui(counted, capacity);
    Value count = select(overflows, counted, in_run);
    Value run = muli(bin, capacity);
    Value run_at = addi(of_lists, run);
    Value list_at = select(overflows, of_overflow, run_at);
    Value some = cmpi(arith::CmpIPredicate::ugt, count, c0);
    auto someBranch = scf::IfOp::create(b, loc, TypeRange{}, some, true, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(someBranch.thenBlock());
      scf::ForOp::create(
          b, loc, tx, volume, cthreads, ValueRange{},
          [&](OpBuilder &b, Location, Value t, ValueRange carried) {
            store(zero_r, tile, {t});
            scf::YieldOp::create(b, loc);
          });
      Value warp = divui(tx, c32w);
      Value lane_x = remui(tx, c32w);
      Value lane = indexCast(i32Type, lane_x);
      Value pt =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, order));
      Value pidx =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
      Value ppt = muli(pt, pt);
      Value pppt = muli(ppt, pt);
      Value pt2 = addi(pt, pt);
      Value thirty_two =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
      Value zero32 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      Value width =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
      Value passes_n2 = muli(pidx, pidx);
      Value passes_n3 = muli(passes_n2, pidx);
      Value c31p = arith::ConstantIndexOp::create(b, loc, 31);
      Value passes_u = addi(passes_n3, c31p);
      Value passes = divui(passes_u, c32w);
      Value three_p = muli(pidx, c3w);
      Value active = cmpi(arith::CmpIPredicate::ult, lane_x, three_p);
      Value lane_a = select(active, lane_x, c0);
      Value axis_l = divui(lane_a, pidx);
      Value j_l = remui(lane_a, pidx);
      Value axis_at = muli(axis_l, cthreads);
      Value k1t = indexCast(i32Type, k1);
      Value k2t = indexCast(i32Type, k2);
      Value k3t = indexCast(i32Type, k3);
      Value k3w = arith::ExtUIOp::create(b, loc, i64Type, k3t);
      Value bytes = arith::ConstantOp::create(
          b, loc, b.getIntegerAttr(i64Type, realBytes));
      Value gindex =
          memref::ExtractAlignedPointerAsIndexOp::create(b, loc, realBuffer);
      Value gbase = indexCast(i64Type, gindex);
      scf::ForOp::create(
          b, loc, c0, count, cthreads, ValueRange{},
          [&](OpBuilder &b, Location, Value round, ValueRange carried) {
            gpu::BarrierOp::create(b, loc);
            Value left = subi(count, round);
            Value in_round = minui(left, cthreads);
            Value stages = cmpi(arith::CmpIPredicate::ult, tx, in_round);
            auto stagesBranch =
                scf::IfOp::create(b, loc, TypeRange{}, stages, true, false);
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(stagesBranch.thenBlock());
              Value entry = addi(round, tx);
              Value entry_at = addi(list_at, entry);
              Value particle = load(bins, {entry_at});
              Value i = indexCast(idxType, particle);
              Value qs = load(q, {i});
              Value qi = convert(realType, qs);
              store(qi, staged, {tx});
              scf::ForOp::create(
                  b, loc, c0, c3w, c1, ValueRange{},
                  [&](OpBuilder &b, Location, Value ax, ValueRange carried) {
                    Value at_n = muli(n_all, ax);
                    Value at_s = addi(at_n, i);
                    Value start_r = load(places, {at_s});
                    Value start = fptosi(i32Type, start_r);
                    Value fax = addi(ax, c3w);
                    Value at_fn = muli(n_all, fax);
                    Value at_f = addi(at_fn, i);
                    Value frac_g = load(places, {at_f});
                    Value slot0 = muli(ax, cthreads);
                    Value slot = addi(slot0, tx);
                    store(start, starts, {slot});
                    Value fslot = addi(slot, cthreads);
                    store(frac_g, staged, {fslot});
                    scf::YieldOp::create(b, loc);
                  });
              scf::YieldOp::create(b, loc);
            }
            gpu::BarrierOp::create(b, loc);
            scf::ForOp::create(
                b, loc, warp, in_round, cwarps, ValueRange{},
                [&](OpBuilder &b, Location, Value k, ValueRange carried) {
                  Value qk = load(staged, {k});
                  Value fk0 = addi(axis_at, k);
                  Value fk = addi(fk0, cthreads);
                  Value frac = load(staged, {fk});
                  Value spline;
                  if constexpr (window == Window::Spline) {
                    Value pointCount = arith::ConstantOp::create(
                        b, loc, b.getIntegerAttr(idxType, order));
                    Value c4o = arith::ConstantIndexOp::create(b, loc, 4);
                    Value closed =
                        cmpi(arith::CmpIPredicate::eq, pointCount, c4o);
                    Value splines = memref::AllocaOp::create(
                        b, loc, MemRefType::get({8}, realType));
                    splineWeights(frac, pointCount, splines);
                    Value tp_value = pickSpline(splines, j_l, pointCount);
                    Value w_sixth = arith::ConstantOp::create(
                        b, loc, b.getFloatAttr(realType, 0.16666666666666666));
                    Value w_one = arith::ConstantOp::create(
                        b, loc, b.getFloatAttr(realType, 1.0));
                    Value w_three = arith::ConstantOp::create(
                        b, loc, b.getFloatAttr(realType, 3.0));
                    Value w_four = arith::ConstantOp::create(
                        b, loc, b.getFloatAttr(realType, 4.0));
                    Value w_six = arith::ConstantOp::create(
                        b, loc, b.getFloatAttr(realType, 6.0));
                    Value cw_w2 = mulf(frac, frac);
                    Value cw_w3 = mulf(cw_w2, frac);
                    Value cw_omw = subf(w_one, frac);
                    Value cw_omw2 = mulf(cw_omw, cw_omw);
                    Value cw_m0 = mulf(cw_omw2, cw_omw);
                    Value cw_a1 = mulf(w_three, cw_w3);
                    Value cw_b1 = mulf(w_six, cw_w2);
                    Value cw_c1 = subf(cw_a1, cw_b1);
                    Value cw_m1 = addf(cw_c1, w_four);
                    Value cw_a2 = mulf(w_three, cw_w2);
                    Value cw_b2 = mulf(w_three, frac);
                    Value cw_c2 = subf(cw_a2, cw_a1);
                    Value cw_d2 = addf(cw_c2, cw_b2);
                    Value cw_m2 = addf(cw_d2, w_one);
                    Value cw_j0 = arith::ConstantIndexOp::create(b, loc, 0);
                    Value cw_j1 = arith::ConstantIndexOp::create(b, loc, 1);
                    Value cw_j2 = arith::ConstantIndexOp::create(b, loc, 2);
                    Value cw_is0 = cmpi(arith::CmpIPredicate::eq, j_l, cw_j0);
                    Value cw_is1 = cmpi(arith::CmpIPredicate::eq, j_l, cw_j1);
                    Value cw_is2 = cmpi(arith::CmpIPredicate::eq, j_l, cw_j2);
                    Value cw_s2 = select(cw_is2, cw_m2, cw_w3);
                    Value cw_s1 = select(cw_is1, cw_m1, cw_s2);
                    Value cw_s0 = select(cw_is0, cw_m0, cw_s1);
                    Value cw_value = mulf(cw_s0, w_sixth);
                    spline = select(closed, cw_value, tp_value);
                  }
                  Value wvalue;
                  if constexpr (window == Window::Prolate) {
                    Value two_i = arith::ConstantIndexOp::create(b, loc, 2);
                    Value three_i = arith::ConstantIndexOp::create(b, loc, 3);
                    Value one = arith::ConstantIndexOp::create(b, loc, 1);
                    Value points = arith::ConstantOp::create(
                        b, loc, b.getIntegerAttr(idxType, order));
                    Value k12t = addi(k1, k2);
                    Value k123t = addi(k12t, k3);
                    Value pe = muli(k123t, three_i);
                    Value j = addi(j_l, c0);
                    Value twor = arith::ConstantOp::create(
                        b, loc, b.getFloatAttr(realType, 2.0));
                    Value oner = arith::ConstantOp::create(
                        b, loc, b.getFloatAttr(realType, 1.0));
                    Value frac2 = mulf(frac, twor);
                    Value t = subf(frac2, oner);
                    Value meta_d = addi(pe, two_i);
                    Value dr = load(tables, {meta_d});
                    Value meta_n = load(tables, {pe});
                    Value terms_i = fptosi(i32Type, dr);
                    Value terms = indexCast(idxType, terms_i);
                    Value n_i = fptosi(i32Type, meta_n);
                    Value n_s = indexCast(idxType, n_i);
                    Value n_2 = addi(n_s, n_s);
                    Value wb0 = addi(pe, three_i);
                    Value wbase = addi(wb0, n_2);
                    Value wvalue0 =
                        clenshaw(wbase, t, one, terms, points, j, tables);
                    wvalue = select(active, wvalue0, zero_r);
                  }
                  Value first_axis = cmpi(arith::CmpIPredicate::eq, axis_l, c0);
                  Value weight;
                  if constexpr (window == Window::Spline) {
                    Value charged = mulf(qk, spline);
                    Value along = select(first_axis, charged, spline);
                    weight = select(active, along, zero_r);
                  }
                  if constexpr (window == Window::Prolate) {
                    Value qv = mulf(wvalue, qk);
                    weight = select(first_axis, qv, wvalue);
                  }
                  Value xstart = load(starts, {k});
                  Value ys_at = addi(k, cthreads);
                  Value ystart = load(starts, {ys_at});
                  Value zs_at = addi(k, cthreads2);
                  Value zstart = load(starts, {zs_at});
                  Value l1 = subi(xstart, o1t);
                  Value l2 = subi(ystart, o2t);
                  Value l3 = subi(zstart, o3t);
                  scf::ForOp::create(
                      b, loc, c0, passes, c1, ValueRange{},
                      [&](OpBuilder &b, Location, Value pass,
                          ValueRange carried) {
                        Value pass32 = indexCast(i32Type, pass);
                        Value pass_base = muli(pass32, thirty_two);
                        Value kk = addi(pass_base, lane);
                        Value in = cmpi(arith::CmpIPredicate::slt, kk, pppt);
                        Value kc = select(in, kk, zero32);
                        Value pa = divui(kc, ppt);
                        Value pbc = remui(kc, ppt);
                        Value pb = divui(pbc, pt);
                        Value pc = remui(pbc, pt);
                        Value wx =
                            laneValue(gpu::ShuffleMode::IDX, weight, pa, width);

                        Value pbs = addi(pb, pt);
                        Value wy = laneValue(gpu::ShuffleMode::IDX, weight, pbs,
                                             width);

                        Value pcs = addi(pc, pt2);
                        Value wz = laneValue(gpu::ShuffleMode::IDX, weight, pcs,
                                             width);

                        auto inBranch = scf::IfOp::create(b, loc, TypeRange{},
                                                          in, true, false);
                        {
                          OpBuilder::InsertionGuard guard(b);
                          b.setInsertionPointToStart(inBranch.thenBlock());
                          Value wxy = mulf(wx, wy);
                          Value value = mulf(wxy, wz);
                          auto overflowsBranch = scf::IfOp::create(
                              b, loc, TypeRange{}, overflows, true, true);
                          {
                            OpBuilder::InsertionGuard guard(b);
                            b.setInsertionPointToStart(
                                overflowsBranch.thenBlock());
                            Value g1s = addi(xstart, pa);
                            Value g1w = subi(g1s, k1t);
                            Value g1over =
                                cmpi(arith::CmpIPredicate::sge, g1s, k1t);
                            Value g1 = select(g1over, g1w, g1s);
                            Value g2s = addi(ystart, pb);
                            Value g2w = subi(g2s, k2t);
                            Value g2over =
                                cmpi(arith::CmpIPredicate::sge, g2s, k2t);
                            Value g2 = select(g2over, g2w, g2s);
                            Value g3s = addi(zstart, pc);
                            Value g3w = subi(g3s, k3t);
                            Value g3over =
                                cmpi(arith::CmpIPredicate::sge, g3s, k3t);
                            Value g3 = select(g3over, g3w, g3s);
                            Value row1 = muli(g1, k2t);
                            Value row = addi(row1, g2);
                            Value row64 =
                                arith::ExtUIOp::create(b, loc, i64Type, row);
                            Value rowbase = muli(row64, k3w);
                            Value g364 =
                                arith::ExtUIOp::create(b, loc, i64Type, g3);
                            Value at = addi(rowbase, g364);
                            Value offset = muli(at, bytes);
                            Value address = addi(gbase, offset);
                            Value pointer = LLVM::IntToPtrOp::create(
                                b, loc, globalPointer, address);
                            addGlobal(pointer, value);
                            scf::YieldOp::create(b, loc);
                          }
                          {
                            OpBuilder::InsertionGuard guard(b);
                            b.setInsertionPointToStart(
                                overflowsBranch.elseBlock());
                            Value t1 = addi(l1, pa);
                            Value t2 = addi(l2, pb);
                            Value t3 = addi(l3, pc);
                            Value t1s = muli(t1, side32);
                            Value t12 = addi(t1s, t2);
                            Value t12s = muli(t12, side32);
                            Value t123 = addi(t12s, t3);
                            Value tat = indexCast(idxType, t123);
                            addShared(tile, tat, value);
                            scf::YieldOp::create(b, loc);
                          }
                          scf::YieldOp::create(b, loc);
                        }
                        scf::YieldOp::create(b, loc);
                      });
                  scf::YieldOp::create(b, loc);
                });
            scf::YieldOp::create(b, loc);
          });
      gpu::BarrierOp::create(b, loc);
      auto bins_blockBranch =
          scf::IfOp::create(b, loc, TypeRange{}, bins_block, true, false);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(bins_blockBranch.thenBlock());
        scf::ForOp::create(
            b, loc, tx, volume, cthreads, ValueRange{},
            [&](OpBuilder &b, Location, Value t, ValueRange carried) {
              Value value = load(tile, {t});
              Value nonzero = cmpf(arith::CmpFPredicate::UNE, value, zero_r);
              auto nonzeroBranch =
                  scf::IfOp::create(b, loc, TypeRange{}, nonzero, true, false);
              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(nonzeroBranch.thenBlock());
                Value t32 = indexCast(i32Type, t);
                Value c = remui(t32, side32);
                Value ab = divui(t32, side32);
                Value accB = remui(ab, side32);
                Value a = divui(ab, side32);
                Value g1s = addi(o1t, a);
                Value g1w = subi(g1s, k1t);
                Value g1over = cmpi(arith::CmpIPredicate::sge, g1s, k1t);
                Value g1 = select(g1over, g1w, g1s);
                Value g2s = addi(o2t, accB);
                Value g2w = subi(g2s, k2t);
                Value g2over = cmpi(arith::CmpIPredicate::sge, g2s, k2t);
                Value g2 = select(g2over, g2w, g2s);
                Value g3s = addi(o3t, c);
                Value g3w = subi(g3s, k3t);
                Value g3over = cmpi(arith::CmpIPredicate::sge, g3s, k3t);
                Value g3 = select(g3over, g3w, g3s);
                Value row1 = muli(g1, k2t);
                Value row = addi(row1, g2);
                Value row64 = arith::ExtUIOp::create(b, loc, i64Type, row);
                Value rowbase = muli(row64, k3w);
                Value g364 = arith::ExtUIOp::create(b, loc, i64Type, g3);
                Value at = addi(rowbase, g364);
                Value offset = muli(at, bytes);
                Value address = addi(gbase, offset);
                Value pointer =
                    LLVM::IntToPtrOp::create(b, loc, globalPointer, address);
                addGlobal(pointer, value);
                scf::YieldOp::create(b, loc);
              }
              scf::YieldOp::create(b, loc);
            });
        scf::YieldOp::create(b, loc);
      }
      scf::YieldOp::create(b, loc);
    }
    gpu::TerminatorOp::create(b, loc);
  }
  func::ReturnOp::create(b, loc);
}

template void DevicePME::emitSpreadTile<Window::Spline, 4>(StringRef);
template void DevicePME::emitSpreadTile<Window::Spline, 8>(StringRef);
template void DevicePME::emitSpreadTile<Window::Spline, 16>(StringRef);
template void DevicePME::emitSpreadTile<Window::Prolate, 4>(StringRef);
template void DevicePME::emitSpreadTile<Window::Prolate, 8>(StringRef);
template void DevicePME::emitSpreadTile<Window::Prolate, 16>(StringRef);
