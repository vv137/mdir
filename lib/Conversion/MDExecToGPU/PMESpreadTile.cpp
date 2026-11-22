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
  Value k1e = arith::AddIOp::create(b, loc, k1, edge_m1);
  Value nb1 = arith::DivUIOp::create(b, loc, k1e, edge);
  Value k2e = arith::AddIOp::create(b, loc, k2, edge_m1);
  Value nb2 = arith::DivUIOp::create(b, loc, k2e, edge);
  Value k3e = arith::AddIOp::create(b, loc, k3, edge_m1);
  Value nb3 = arith::DivUIOp::create(b, loc, k3e, edge);
  Value nb12 = arith::MulIOp::create(b, loc, nb1, nb2);
  Value nbins = arith::MulIOp::create(b, loc, nb12, nb3);
  Value nbins_m1 = arith::SubIOp::create(b, loc, nbins, c1);
  Value mean_u = arith::AddIOp::create(b, loc, n_all, nbins_m1);
  Value mean = arith::DivUIOp::create(b, loc, mean_u, nbins);
  Value mean2 = arith::AddIOp::create(b, loc, mean, mean);
  Value slack = arith::ConstantIndexOp::create(b, loc, 8);
  Value capacity = arith::AddIOp::create(b, loc, mean2, slack);
  Value of_lists = arith::AddIOp::create(b, loc, nbins, c1);
  Value lists = arith::MulIOp::create(b, loc, nbins, capacity);
  Value of_overflow = arith::AddIOp::create(b, loc, of_lists, lists);
  Value cthreads = arith::ConstantIndexOp::create(b, loc, tileThreads);
  Value blocks_tiles = arith::AddIOp::create(b, loc, nbins, c1);
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
    Value overflows =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, bx, nbins);
    Value bins_block = arith::XOrIOp::create(b, loc, overflows, true_w);
    Value bin = arith::SelectOp::create(b, loc, overflows, c0, bx);
    Value b3 = arith::RemUIOp::create(b, loc, bin, nb3);
    Value b12 = arith::DivUIOp::create(b, loc, bin, nb3);
    Value b2 = arith::RemUIOp::create(b, loc, b12, nb2);
    Value b1 = arith::DivUIOp::create(b, loc, b12, nb2);
    Value o1 = arith::MulIOp::create(b, loc, b1, edge);
    Value o2 = arith::MulIOp::create(b, loc, b2, edge);
    Value o3 = arith::MulIOp::create(b, loc, b3, edge);
    Value o1t = arith::IndexCastOp::create(b, loc, i32Type, o1);
    Value o2t = arith::IndexCastOp::create(b, loc, i32Type, o2);
    Value o3t = arith::IndexCastOp::create(b, loc, i32Type, o3);
    Value counted32 = load(bins, {bx});
    Value counted = arith::IndexCastOp::create(b, loc, idxType, counted32);
    Value in_run = arith::MinUIOp::create(b, loc, counted, capacity);
    Value count = arith::SelectOp::create(b, loc, overflows, counted, in_run);
    Value run = arith::MulIOp::create(b, loc, bin, capacity);
    Value run_at = arith::AddIOp::create(b, loc, of_lists, run);
    Value list_at =
        arith::SelectOp::create(b, loc, overflows, of_overflow, run_at);
    Value some =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt, count, c0);
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
      Value warp = arith::DivUIOp::create(b, loc, tx, c32w);
      Value lane_x = arith::RemUIOp::create(b, loc, tx, c32w);
      Value lane = arith::IndexCastOp::create(b, loc, i32Type, lane_x);
      Value pt =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, order));
      Value pidx =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, order));
      Value ppt = arith::MulIOp::create(b, loc, pt, pt);
      Value pppt = arith::MulIOp::create(b, loc, ppt, pt);
      Value pt2 = arith::AddIOp::create(b, loc, pt, pt);
      Value thirty_two =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
      Value zero32 =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 0));
      Value width =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(i32Type, 32));
      Value passes_n2 = arith::MulIOp::create(b, loc, pidx, pidx);
      Value passes_n3 = arith::MulIOp::create(b, loc, passes_n2, pidx);
      Value c31p = arith::ConstantIndexOp::create(b, loc, 31);
      Value passes_u = arith::AddIOp::create(b, loc, passes_n3, c31p);
      Value passes = arith::DivUIOp::create(b, loc, passes_u, c32w);
      Value three_p = arith::MulIOp::create(b, loc, pidx, c3w);
      Value active = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                           lane_x, three_p);
      Value lane_a = arith::SelectOp::create(b, loc, active, lane_x, c0);
      Value axis_l = arith::DivUIOp::create(b, loc, lane_a, pidx);
      Value j_l = arith::RemUIOp::create(b, loc, lane_a, pidx);
      Value axis_at = arith::MulIOp::create(b, loc, axis_l, cthreads);
      Value k1t = arith::IndexCastOp::create(b, loc, i32Type, k1);
      Value k2t = arith::IndexCastOp::create(b, loc, i32Type, k2);
      Value k3t = arith::IndexCastOp::create(b, loc, i32Type, k3);
      Value k3w = arith::ExtUIOp::create(b, loc, i64Type, k3t);
      Value bytes = arith::ConstantOp::create(
          b, loc, b.getIntegerAttr(i64Type, realBytes));
      Value gindex =
          memref::ExtractAlignedPointerAsIndexOp::create(b, loc, realBuffer);
      Value gbase = arith::IndexCastOp::create(b, loc, i64Type, gindex);
      scf::ForOp::create(
          b, loc, c0, count, cthreads, ValueRange{},
          [&](OpBuilder &b, Location, Value round, ValueRange carried) {
            gpu::BarrierOp::create(b, loc);
            Value left = arith::SubIOp::create(b, loc, count, round);
            Value in_round = arith::MinUIOp::create(b, loc, left, cthreads);
            Value stages = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::ult, tx, in_round);
            auto stagesBranch =
                scf::IfOp::create(b, loc, TypeRange{}, stages, true, false);
            {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPointToStart(stagesBranch.thenBlock());
              Value entry = arith::AddIOp::create(b, loc, round, tx);
              Value entry_at = arith::AddIOp::create(b, loc, list_at, entry);
              Value particle = load(bins, {entry_at});
              Value i = arith::IndexCastOp::create(b, loc, idxType, particle);
              Value qs = load(q, {i});
              Value qi = convert(realType, qs);
              store(qi, staged, {tx});
              scf::ForOp::create(
                  b, loc, c0, c3w, c1, ValueRange{},
                  [&](OpBuilder &b, Location, Value ax, ValueRange carried) {
                    Value at_n = arith::MulIOp::create(b, loc, n_all, ax);
                    Value at_s = arith::AddIOp::create(b, loc, at_n, i);
                    Value start_r = load(places, {at_s});
                    Value start =
                        arith::FPToSIOp::create(b, loc, i32Type, start_r);
                    Value fax = arith::AddIOp::create(b, loc, ax, c3w);
                    Value at_fn = arith::MulIOp::create(b, loc, n_all, fax);
                    Value at_f = arith::AddIOp::create(b, loc, at_fn, i);
                    Value frac_g = load(places, {at_f});
                    Value slot0 = arith::MulIOp::create(b, loc, ax, cthreads);
                    Value slot = arith::AddIOp::create(b, loc, slot0, tx);
                    store(start, starts, {slot});
                    Value fslot = arith::AddIOp::create(b, loc, slot, cthreads);
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
                  Value fk0 = arith::AddIOp::create(b, loc, axis_at, k);
                  Value fk = arith::AddIOp::create(b, loc, fk0, cthreads);
                  Value frac = load(staged, {fk});
                  Value spline;
                  if constexpr (window == Window::Spline) {
                    Value pointCount = arith::ConstantOp::create(
                        b, loc, b.getIntegerAttr(idxType, order));
                    Value c4o = arith::ConstantIndexOp::create(b, loc, 4);
                    Value closed = arith::CmpIOp::create(
                        b, loc, arith::CmpIPredicate::eq, pointCount, c4o);
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
                    Value cw_w2 = arith::MulFOp::create(b, loc, frac, frac);
                    Value cw_w3 = arith::MulFOp::create(b, loc, cw_w2, frac);
                    Value cw_omw = arith::SubFOp::create(b, loc, w_one, frac);
                    Value cw_omw2 =
                        arith::MulFOp::create(b, loc, cw_omw, cw_omw);
                    Value cw_m0 =
                        arith::MulFOp::create(b, loc, cw_omw2, cw_omw);
                    Value cw_a1 = arith::MulFOp::create(b, loc, w_three, cw_w3);
                    Value cw_b1 = arith::MulFOp::create(b, loc, w_six, cw_w2);
                    Value cw_c1 = arith::SubFOp::create(b, loc, cw_a1, cw_b1);
                    Value cw_m1 = arith::AddFOp::create(b, loc, cw_c1, w_four);
                    Value cw_a2 = arith::MulFOp::create(b, loc, w_three, cw_w2);
                    Value cw_b2 = arith::MulFOp::create(b, loc, w_three, frac);
                    Value cw_c2 = arith::SubFOp::create(b, loc, cw_a2, cw_a1);
                    Value cw_d2 = arith::AddFOp::create(b, loc, cw_c2, cw_b2);
                    Value cw_m2 = arith::AddFOp::create(b, loc, cw_d2, w_one);
                    Value cw_j0 = arith::ConstantIndexOp::create(b, loc, 0);
                    Value cw_j1 = arith::ConstantIndexOp::create(b, loc, 1);
                    Value cw_j2 = arith::ConstantIndexOp::create(b, loc, 2);
                    Value cw_is0 = arith::CmpIOp::create(
                        b, loc, arith::CmpIPredicate::eq, j_l, cw_j0);
                    Value cw_is1 = arith::CmpIOp::create(
                        b, loc, arith::CmpIPredicate::eq, j_l, cw_j1);
                    Value cw_is2 = arith::CmpIOp::create(
                        b, loc, arith::CmpIPredicate::eq, j_l, cw_j2);
                    Value cw_s2 =
                        arith::SelectOp::create(b, loc, cw_is2, cw_m2, cw_w3);
                    Value cw_s1 =
                        arith::SelectOp::create(b, loc, cw_is1, cw_m1, cw_s2);
                    Value cw_s0 =
                        arith::SelectOp::create(b, loc, cw_is0, cw_m0, cw_s1);
                    Value cw_value =
                        arith::MulFOp::create(b, loc, cw_s0, w_sixth);
                    spline = arith::SelectOp::create(b, loc, closed, cw_value,
                                                     tp_value);
                  }
                  Value wvalue;
                  if constexpr (window == Window::Prolate) {
                    Value two_i = arith::ConstantIndexOp::create(b, loc, 2);
                    Value three_i = arith::ConstantIndexOp::create(b, loc, 3);
                    Value one = arith::ConstantIndexOp::create(b, loc, 1);
                    Value points = arith::ConstantOp::create(
                        b, loc, b.getIntegerAttr(idxType, order));
                    Value k12t = arith::AddIOp::create(b, loc, k1, k2);
                    Value k123t = arith::AddIOp::create(b, loc, k12t, k3);
                    Value pe = arith::MulIOp::create(b, loc, k123t, three_i);
                    Value j = arith::AddIOp::create(b, loc, j_l, c0);
                    Value twor = arith::ConstantOp::create(
                        b, loc, b.getFloatAttr(realType, 2.0));
                    Value oner = arith::ConstantOp::create(
                        b, loc, b.getFloatAttr(realType, 1.0));
                    Value frac2 = arith::MulFOp::create(b, loc, frac, twor);
                    Value t = arith::SubFOp::create(b, loc, frac2, oner);
                    Value meta_d = arith::AddIOp::create(b, loc, pe, two_i);
                    Value dr = load(tables, {meta_d});
                    Value meta_n = load(tables, {pe});
                    Value terms_i =
                        arith::FPToSIOp::create(b, loc, i32Type, dr);
                    Value terms =
                        arith::IndexCastOp::create(b, loc, idxType, terms_i);
                    Value n_i =
                        arith::FPToSIOp::create(b, loc, i32Type, meta_n);
                    Value n_s =
                        arith::IndexCastOp::create(b, loc, idxType, n_i);
                    Value n_2 = arith::AddIOp::create(b, loc, n_s, n_s);
                    Value wb0 = arith::AddIOp::create(b, loc, pe, three_i);
                    Value wbase = arith::AddIOp::create(b, loc, wb0, n_2);
                    Value wvalue0 =
                        clenshaw(wbase, t, one, terms, points, j, tables);
                    wvalue = arith::SelectOp::create(b, loc, active, wvalue0,
                                                     zero_r);
                  }
                  Value first_axis = arith::CmpIOp::create(
                      b, loc, arith::CmpIPredicate::eq, axis_l, c0);
                  Value weight;
                  if constexpr (window == Window::Spline) {
                    Value charged = arith::MulFOp::create(b, loc, qk, spline);
                    Value along = arith::SelectOp::create(b, loc, first_axis,
                                                          charged, spline);
                    weight =
                        arith::SelectOp::create(b, loc, active, along, zero_r);
                  }
                  if constexpr (window == Window::Prolate) {
                    Value qv = arith::MulFOp::create(b, loc, wvalue, qk);
                    weight =
                        arith::SelectOp::create(b, loc, first_axis, qv, wvalue);
                  }
                  Value xstart = load(starts, {k});
                  Value ys_at = arith::AddIOp::create(b, loc, k, cthreads);
                  Value ystart = load(starts, {ys_at});
                  Value zs_at = arith::AddIOp::create(b, loc, k, cthreads2);
                  Value zstart = load(starts, {zs_at});
                  Value l1 = arith::SubIOp::create(b, loc, xstart, o1t);
                  Value l2 = arith::SubIOp::create(b, loc, ystart, o2t);
                  Value l3 = arith::SubIOp::create(b, loc, zstart, o3t);
                  scf::ForOp::create(
                      b, loc, c0, passes, c1, ValueRange{},
                      [&](OpBuilder &b, Location, Value pass,
                          ValueRange carried) {
                        Value pass32 =
                            arith::IndexCastOp::create(b, loc, i32Type, pass);
                        Value pass_base =
                            arith::MulIOp::create(b, loc, pass32, thirty_two);
                        Value kk =
                            arith::AddIOp::create(b, loc, pass_base, lane);
                        Value in = arith::CmpIOp::create(
                            b, loc, arith::CmpIPredicate::slt, kk, pppt);
                        Value kc =
                            arith::SelectOp::create(b, loc, in, kk, zero32);
                        Value pa = arith::DivUIOp::create(b, loc, kc, ppt);
                        Value pbc = arith::RemUIOp::create(b, loc, kc, ppt);
                        Value pb = arith::DivUIOp::create(b, loc, pbc, pt);
                        Value pc = arith::RemUIOp::create(b, loc, pbc, pt);
                        auto wxShuffle = gpu::ShuffleOp::create(
                            b, loc, weight, pa, width, gpu::ShuffleMode::IDX);
                        Value wx = wxShuffle.getShuffleResult();

                        Value pbs = arith::AddIOp::create(b, loc, pb, pt);
                        auto wyShuffle = gpu::ShuffleOp::create(
                            b, loc, weight, pbs, width, gpu::ShuffleMode::IDX);
                        Value wy = wyShuffle.getShuffleResult();

                        Value pcs = arith::AddIOp::create(b, loc, pc, pt2);
                        auto wzShuffle = gpu::ShuffleOp::create(
                            b, loc, weight, pcs, width, gpu::ShuffleMode::IDX);
                        Value wz = wzShuffle.getShuffleResult();

                        auto inBranch = scf::IfOp::create(b, loc, TypeRange{},
                                                          in, true, false);
                        {
                          OpBuilder::InsertionGuard guard(b);
                          b.setInsertionPointToStart(inBranch.thenBlock());
                          Value wxy = arith::MulFOp::create(b, loc, wx, wy);
                          Value value = arith::MulFOp::create(b, loc, wxy, wz);
                          auto overflowsBranch = scf::IfOp::create(
                              b, loc, TypeRange{}, overflows, true, true);
                          {
                            OpBuilder::InsertionGuard guard(b);
                            b.setInsertionPointToStart(
                                overflowsBranch.thenBlock());
                            Value g1s =
                                arith::AddIOp::create(b, loc, xstart, pa);
                            Value g1w = arith::SubIOp::create(b, loc, g1s, k1t);
                            Value g1over = arith::CmpIOp::create(
                                b, loc, arith::CmpIPredicate::sge, g1s, k1t);
                            Value g1 = arith::SelectOp::create(b, loc, g1over,
                                                               g1w, g1s);
                            Value g2s =
                                arith::AddIOp::create(b, loc, ystart, pb);
                            Value g2w = arith::SubIOp::create(b, loc, g2s, k2t);
                            Value g2over = arith::CmpIOp::create(
                                b, loc, arith::CmpIPredicate::sge, g2s, k2t);
                            Value g2 = arith::SelectOp::create(b, loc, g2over,
                                                               g2w, g2s);
                            Value g3s =
                                arith::AddIOp::create(b, loc, zstart, pc);
                            Value g3w = arith::SubIOp::create(b, loc, g3s, k3t);
                            Value g3over = arith::CmpIOp::create(
                                b, loc, arith::CmpIPredicate::sge, g3s, k3t);
                            Value g3 = arith::SelectOp::create(b, loc, g3over,
                                                               g3w, g3s);
                            Value row1 = arith::MulIOp::create(b, loc, g1, k2t);
                            Value row = arith::AddIOp::create(b, loc, row1, g2);
                            Value row64 =
                                arith::ExtUIOp::create(b, loc, i64Type, row);
                            Value rowbase =
                                arith::MulIOp::create(b, loc, row64, k3w);
                            Value g364 =
                                arith::ExtUIOp::create(b, loc, i64Type, g3);
                            Value at =
                                arith::AddIOp::create(b, loc, rowbase, g364);
                            Value offset =
                                arith::MulIOp::create(b, loc, at, bytes);
                            Value address =
                                arith::AddIOp::create(b, loc, gbase, offset);
                            Value pointer = LLVM::IntToPtrOp::create(
                                b, loc, globalPointer, address);
                            addGlobal(pointer, value);
                            scf::YieldOp::create(b, loc);
                          }
                          {
                            OpBuilder::InsertionGuard guard(b);
                            b.setInsertionPointToStart(
                                overflowsBranch.elseBlock());
                            Value t1 = arith::AddIOp::create(b, loc, l1, pa);
                            Value t2 = arith::AddIOp::create(b, loc, l2, pb);
                            Value t3 = arith::AddIOp::create(b, loc, l3, pc);
                            Value t1s =
                                arith::MulIOp::create(b, loc, t1, side32);
                            Value t12 = arith::AddIOp::create(b, loc, t1s, t2);
                            Value t12s =
                                arith::MulIOp::create(b, loc, t12, side32);
                            Value t123 =
                                arith::AddIOp::create(b, loc, t12s, t3);
                            Value tat = arith::IndexCastOp::create(
                                b, loc, idxType, t123);
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
              Value nonzero = arith::CmpFOp::create(
                  b, loc, arith::CmpFPredicate::UNE, value, zero_r);
              auto nonzeroBranch =
                  scf::IfOp::create(b, loc, TypeRange{}, nonzero, true, false);
              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(nonzeroBranch.thenBlock());
                Value t32 = arith::IndexCastOp::create(b, loc, i32Type, t);
                Value c = arith::RemUIOp::create(b, loc, t32, side32);
                Value ab = arith::DivUIOp::create(b, loc, t32, side32);
                Value accB = arith::RemUIOp::create(b, loc, ab, side32);
                Value a = arith::DivUIOp::create(b, loc, ab, side32);
                Value g1s = arith::AddIOp::create(b, loc, o1t, a);
                Value g1w = arith::SubIOp::create(b, loc, g1s, k1t);
                Value g1over = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::sge, g1s, k1t);
                Value g1 = arith::SelectOp::create(b, loc, g1over, g1w, g1s);
                Value g2s = arith::AddIOp::create(b, loc, o2t, accB);
                Value g2w = arith::SubIOp::create(b, loc, g2s, k2t);
                Value g2over = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::sge, g2s, k2t);
                Value g2 = arith::SelectOp::create(b, loc, g2over, g2w, g2s);
                Value g3s = arith::AddIOp::create(b, loc, o3t, c);
                Value g3w = arith::SubIOp::create(b, loc, g3s, k3t);
                Value g3over = arith::CmpIOp::create(
                    b, loc, arith::CmpIPredicate::sge, g3s, k3t);
                Value g3 = arith::SelectOp::create(b, loc, g3over, g3w, g3s);
                Value row1 = arith::MulIOp::create(b, loc, g1, k2t);
                Value row = arith::AddIOp::create(b, loc, row1, g2);
                Value row64 = arith::ExtUIOp::create(b, loc, i64Type, row);
                Value rowbase = arith::MulIOp::create(b, loc, row64, k3w);
                Value g364 = arith::ExtUIOp::create(b, loc, i64Type, g3);
                Value at = arith::AddIOp::create(b, loc, rowbase, g364);
                Value offset = arith::MulIOp::create(b, loc, at, bytes);
                Value address = arith::AddIOp::create(b, loc, gbase, offset);
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
