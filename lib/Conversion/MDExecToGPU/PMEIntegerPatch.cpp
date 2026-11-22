// Frame-certified cubic charge deposition (D[pme-tiles]).
#include "DevicePME.h"
#include "mdir/Dialect/MDExec/PMECellOrder.h"

using namespace mlir;
using namespace mdir::kernels;

void DevicePME::emitIntegerPatch(StringRef symbol) {
  OpBuilder::InsertionGuard guard(b);
  auto words = globalMem({ShapedType::kDynamic}, i32Type);
  auto wideWords = globalMem({ShapedType::kDynamic}, i64Type);
  auto mesh = globalMem({ShapedType::kDynamic}, realType);
  SmallVector<Type> args{records(), words, words, wideWords, words, mesh,
                         idxType, idxType, idxType, idxType, idxType};
  auto fn = func::FuncOp::create(b, loc, symbol, b.getFunctionType(args, {}));
  fn.setPrivate();
  Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  auto a = entry->getArguments();
  Value particles = a[0], offsets = a[1], grid = a[2], wide = a[3];
  Value flag = a[4], out = a[5], threads = a[9], patchX = a[10];
  Value dims[] = {a[6], a[7], a[8]};
  auto ix = [&](int n) -> Value { return arith::ConstantIndexOp::create(b, loc, n); };
  auto integer = [&](Type type, int n) -> Value {
    return arith::ConstantOp::create(b, loc, b.getIntegerAttr(type, n));
  };
  auto fp = [&](Type type, double x) -> Value {
    return arith::ConstantOp::create(b, loc, b.getFloatAttr(type, x));
  };
  constexpr int edge = mdir::md_exec::PMECellEdge, side = edge + 3;
  Value zero = ix(0), one = ix(1), four = ix(4);
  Value cellEdge = ix(edge), patchSide = ix(side);
  Value zero32 = integer(i32Type, 0), one32 = integer(i32Type, 1);
  Value zero64 = integer(i64Type, 0);
  Value points = memref::DimOp::create(b, loc, grid, zero);
  Value atoms = memref::DimOp::create(b, loc, particles, zero);
  Value cells = divui(subi(memref::DimOp::create(b, loc, offsets, zero), one), ix(8));
  auto blocks = [&](Value n) { return maxui(one, arith::CeilDivUIOp::create(b, loc, n, threads)); };
  auto when = [&](Value pred, auto body) {
    auto op = scf::IfOp::create(b, loc, pred, false);
    OpBuilder::InsertionGuard scope(b);
    b.setInsertionPointToStart(op.thenBlock());
    body();
  };
  auto either = [&](Value pred, auto yes, auto no) {
    auto op = scf::IfOp::create(b, loc, pred, true);
    OpBuilder::InsertionGuard scope(b);
    b.setInsertionPointToStart(op.thenBlock());
    yes();
    b.setInsertionPointToStart(op.elseBlock());
    no();
  };
  auto loop = [&](Value first, Value end, auto body) {
    auto op = scf::ForOp::create(b, loc, first, end, threads);
    OpBuilder::InsertionGuard scope(b);
    b.setInsertionPointToStart(op.getBody());
    body(op.getInductionVar());
  };
  auto atomic = [&](LLVM::AtomicBinOp kind, Value word, Value buf, Value at) -> Value {
    bool local = isa<gpu::AddressSpaceAttr>(cast<MemRefType>(buf.getType()).getMemorySpace());
    Value base = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, buf);
    Value bytes = ix(word.getType().getIntOrFloatBitWidth() / 8);
    Value address = indexCast(i64Type, addi(base, muli(at, bytes)));
    Value ptr = LLVM::IntToPtrOp::create(
        b, loc, LLVM::LLVMPointerType::get(b.getContext(), local ? 3 : 1), address);
    return LLVM::AtomicRMWOp::create(b, loc, kind, ptr, word,
                                     LLVM::AtomicOrdering::monotonic,
                                     local ? "block" : "device");
  };
  auto linear = [&](Value x, Value y, Value z) {
    return addi(muli(addi(muli(x, dims[1]), y), dims[2]), z);
  };
  auto wrap = [&](Value i, Value n) {
    return select(cmpi(arith::CmpIPredicate::uge, i, n), subi(i, n), i);
  };
  // The pack is frame-local. Recheck charge and voxel occupancy on every
  // evaluation, including changes of cell and alchemical charges.
  const bool mixed = realType.isF32();
  const double scale = mixed ? 0x1p23 : 0x1p40;
  Value gridBlocks = blocks(points);
  forParticles(b, loc, points, gridBlocks, threads, one, [&](Value i) {
    if (mixed)
      store(zero32, grid, {i});
    when(cmpi(arith::CmpIPredicate::eq, i, zero), [&] {
      store(mixed ? zero32 : one32, flag, {zero});
    });
  });
  Value count = sitofp(f64Type, indexCast(i64Type, atoms));
  forParticles(b, loc, atoms, blocks(atoms), threads, one, [&](Value i) {
    Value p = readRecord(particles, i);
    Value q = math::AbsFOp::create(b, loc, convert(f64Type, extract(p, 3)));
    // Every node receives at most one word per atom. The factor two
    // bounds the cubic weight product and its evaluation error; the added
    // unit covers nearest-even rounding. Leave one full bit of headroom.
    Value bound = mulf(count, addf(mulf(q, fp(f64Type, 2 * scale)), fp(f64Type, 1)));
    when(cmpf(arith::CmpFPredicate::UGE, bound, fp(f64Type, 0x1p62)), [&] {
      LLVM::Trap::create(b, loc);
    });
    if (mixed) {
      Value base[3];
      for (int axis = 0; axis < 3; ++axis)
        base[axis] = splitMeshAxis(extract(p, axis), dims[axis], four, true).first;
      Value prior = atomic(LLVM::AtomicBinOp::add, one32, grid,
                           linear(base[0], base[1], base[2]));
      // 15 * (8.01 * 2^23 * 2 + 64) < 2^31. A failure switches the
      // entire mesh to certified i64 before its first deposit.
      Value tooMany = cmpi(arith::CmpIPredicate::uge, prior, integer(i32Type, 15));
      Value tooLarge = cmpf(arith::CmpFPredicate::OGT, q, fp(f64Type, 2));
      when(ori(tooMany, tooLarge), [&] {
        atomic(LLVM::AtomicBinOp::_or, one32, flag, zero);
      });
    }
  });
  auto narrow = [&] { return cmpi(arith::CmpIPredicate::eq, load(flag, {zero}), zero32); };
  // One vector store per lane amortizes address calculation and the load
  // of the frame certificate. Odd-sized meshes have a short scalar tail.
  Value groups = arith::CeilDivUIOp::create(b, loc, points, four);
  forParticles(b, loc, groups, blocks(groups), threads, one, [&](Value group) {
    Value at = muli(group, four);
    auto clear = [&](Value buf, Value value) {
      either(cmpi(arith::CmpIPredicate::ule, addi(at, four), points), [&] {
        Type vec = VectorType::get({4}, value.getType());
        Value zeros = vector::BroadcastOp::create(b, loc, vec, value);
        vector::StoreOp::create(b, loc, zeros, buf, ValueRange{at},
                                false, llvm::Align(4 * value.getType().getIntOrFloatBitWidth() / 8));
      }, [&] {
        for (int lane = 0; lane < 3; ++lane) {
          Value i = addi(at, ix(lane));
          when(cmpi(arith::CmpIPredicate::ult, i, points), [&] { store(value, buf, {i}); });
        }
      });
    };
    either(narrow(), [&] { clear(grid, zero32); }, [&] { clear(wide, zero64); });
  });
  Value ny = arith::CeilDivUIOp::create(b, loc, dims[1], cellEdge);
  Value nz = arith::CeilDivUIOp::create(b, loc, dims[2], cellEdge);
  auto launch = gpu::LaunchOp::create(b, loc, cells, one, one, threads, one, one,
      Value{}, Type{}, ValueRange{}, TypeRange{shared(side * side * side, i32Type)});
  {
    OpBuilder::InsertionGuard scope(b);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value tile = launch.getWorkgroupAttributionBBArgs()[0];
    Value tx = launch.getThreadIds().x, cell = launch.getBlockIds().x;
    Value first = indexCast(idxType, load(offsets, {muli(cell, ix(8))}));
    Value end = indexCast(idxType, load(offsets, {muli(addi(cell, one), ix(8))}));
    Value yz = divui(cell, nz);
    Value origin[] = {muli(divui(yz, ny), cellEdge), muli(remui(yz, ny), cellEdge),
                      muli(remui(cell, nz), cellEdge)};
    auto spread = [&](bool usePatch) {
      Value volume = muli(patchX, ix(side * side));
      if (usePatch) {
        loop(tx, volume, [&](Value j) { store(zero32, tile, {j}); });
        gpu::BarrierOp::create(b, loc);
      }
      loop(addi(first, tx), end, [&](Value i) {
        Value p = readRecord(particles, i);
        Value start[3], local[3];
        std::array<Value, 4> weights[3];
        Value fits = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
        for (int axis = 0; axis < 3; ++axis) {
          Value fraction;
          std::tie(start[axis], fraction) = splitMeshAxis(extract(p, axis), dims[axis], four, true);
          weights[axis] = cubicWeights(fraction);
          local[axis] = subi(start[axis], origin[axis]);
          Value extent = axis == 0 ? patchX : patchSide;
          fits = andi(fits, andi(cmpi(arith::CmpIPredicate::uge, start[axis], origin[axis]),
                                 cmpi(arith::CmpIPredicate::ule, addi(local[axis], four), extent)));
        }
        Value q = mulf(extract(p, 3), fp(realType, scale));
        auto deposit = [&](bool patch) {
          for (int x = 0; x < 4; ++x) {
            Value qx = mulf(q, weights[0][x]);
            for (int y = 0; y < 4; ++y) {
              Value qxy = mulf(qx, weights[1][y]);
              for (int z = 0; z < 4; ++z) {
                Value rounded = math::RoundEvenOp::create(b, loc, mulf(qxy, weights[2][z]));
                Value at = patch
                    ? addi(muli(addi(muli(addi(local[0], ix(x)), patchSide),
                                     addi(local[1], ix(y))), patchSide), addi(local[2], ix(z)))
                    : linear(wrap(addi(start[0], ix(x)), dims[0]),
                             wrap(addi(start[1], ix(y)), dims[1]),
                             wrap(addi(start[2], ix(z)), dims[2]));
                atomic(LLVM::AtomicBinOp::add, fptosi(usePatch ? i32Type : i64Type, rounded),
                       patch ? tile : (usePatch ? grid : wide), at);
              }
            }
          }
        };
        if (usePatch)
          either(fits, [&] { deposit(true); }, [&] { deposit(false); });
        else
          deposit(false);
      });
      if (usePatch) {
        gpu::BarrierOp::create(b, loc);
        loop(tx, volume, [&](Value j) {
          Value value = load(tile, {j});
          when(cmpi(arith::CmpIPredicate::ne, value, zero32), [&] {
            Value row = divui(j, patchSide);
            Value at = linear(wrap(addi(origin[0], divui(row, patchSide)), dims[0]),
                              wrap(addi(origin[1], remui(row, patchSide)), dims[1]),
                              wrap(addi(origin[2], remui(j, patchSide)), dims[2]));
            atomic(LLVM::AtomicBinOp::add, value, grid, at);
          });
        });
      }
    };
    when(cmpi(arith::CmpIPredicate::ult, first, end), [&] {
      either(narrow(), [&] { spread(true); }, [&] { spread(false); });
    });
    gpu::TerminatorOp::create(b, loc);
  }
  forParticles(b, loc, points, gridBlocks, threads, one, [&](Value i) {
    auto convertGrid = [&](Value input) {
      store(mulf(sitofp(realType, load(input, {i})), fp(realType, 1 / scale)), out, {i});
    };
    either(narrow(), [&] { convertGrid(grid); }, [&] { convertGrid(wide); });
  });
  func::ReturnOp::create(b, loc);
}
