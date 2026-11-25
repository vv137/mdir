// Atom-local evaluation of the application's differentiated tuple region.
#include "mdir/Conversion/BondedUnits.h"
#include "NeighborBuildCode.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Support/ForceWords.h"

using namespace mlir;
using namespace mdir;
using namespace mdir::kernels;
using namespace mdir::kernels::detail;
namespace {
struct UnitKernel : NeighborBuildCode {
  using NeighborBuildCode::NeighborBuildCode;
  void failIf(Value bad) {
    ifThen(bad, [&] {
      LLVM::InlineAsmOp::create(b, loc, TypeRange{}, ValueRange{}, "trap;", "",
                                true, false, LLVM::TailCallKind::None,
                                LLVM::AsmDialectAttr(), ArrayAttr());
    });
  }
  Value quantize(Value value, double scale) {
    Type type = value.getType();
    Value scaled = mulf(value, createReal(b, loc, type, scale));
    Value fits = andi(
        cmpf(CmpFPredicate::OGE, scaled, createReal(b, loc, type, -0x1p63)),
        cmpf(CmpFPredicate::OLT, scaled, createReal(b, loc, type, 0x1p63)));
    failIf(notNarrow(fits));
    return fptosi(i64,
                  roundeven(select(fits, scaled, createReal(b, loc, type, 0))));
  }
  Value atomicAdd(Value dest, ValueRange indices, Value value) {
    // The add returns the preceding word for the signed overflow check.
    // Phase barriers and kernel completion order the consumers of the sum.
    auto layout = memref::ExtractStridedMetadataOp::create(b, loc, dest);
    Value element = layout.getOffset();
    for (auto [i, stride] : llvm::zip_equal(indices, layout.getStrides()))
      element = addi(element, muli(i, stride));
    Value address = addi(baseAddress(dest),
                         muli(indexCast(i64, element),
                              cl(value.getType().getIntOrFloatBitWidth() / 8)));
    auto type = cast<MemRefType>(dest.getType());
    bool shared =
        type.getMemorySpace() ==
        gpu::AddressSpaceAttr::get(context, gpu::AddressSpace::Workgroup);
    auto ptr = LLVM::LLVMPointerType::get(context, shared ? 3 : 1);
    return LLVM::AtomicRMWOp::create(
        b, loc, LLVM::AtomicBinOp::add,
        LLVM::IntToPtrOp::create(b, loc, ptr, address), value,
        LLVM::AtomicOrdering::monotonic, shared ? "block" : "device");
  }
  void addWord(Value dest, ValueRange indices, Value value) {
    Value old = atomicAdd(dest, indices, value);
    Value sum = addi(old, value);
    failIf(cmpi(CmpIPredicate::slt, andi(xori(sum, old), xori(sum, value)),
                cl(0)));
  }
  void tryNarrowAdd(Value narrow, ValueRange indices, Value word, Value retry) {
    Value v = trunci(word), old = atomicAdd(narrow, indices, v);
    Value sum = addi(old, v);
    Value truncated =
        cmpi(CmpIPredicate::ne, arith::ExtSIOp::create(b, loc, i64, v), word);
    Value overflow =
        cmpi(CmpIPredicate::slt, andi(xori(sum, old), xori(sum, v)), ci(0));
    ifThen(ori(truncated, overflow), [&] {
      memref::AtomicRMWOp::create(b, loc, arith::AtomicRMWKind::ori, ci(1),
                                  retry, ValueRange{cx(0)});
    });
  }
  void publish(Value words, Value row, Value dest, Value atom,
               function_ref<void(int, Value)> afterLoad) {
    Type real = getElementTypeOrSelf(dest.getType());
    Value total = createZero(b, loc, VectorType::get({3}, real));
    for (int c = 0; c < 3; ++c) {
      Value word = load(words, {row, cx(c)});
      afterLoad(c, word);
      Value decoded = mulf(sitofp(real, word),
                           createReal(b, loc, real, 1.0 / forceWordScale));
      total = vector::InsertOp::create(b, loc, decoded, total, c);
    }
    storeElement(b, loc, addf(loadElement(b, loc, dest, atom), total), dest,
                 atom);
  }
  void emit(ArrayRef<md_exec::TupleForOp> ops, Value block, Value lane,
            Value stream, Value place, Value box, Value inverse, Value global,
            ValueRange outputs, ValueRange shared, int mode, Value arrivals,
            bool checkWords) {
    Value positions = shared[0], liveIds = shared[1], owned = shared[2],
          forces = shared[3];
    auto word = [&](Value p) { return load(stream, {p}); };
    auto offset = [&](Value p) { return indexCast(index, word(p)); };
    Value base = offset(block), atoms = offset(base);
    Value imports = addi(base, cx(3 + ops.size()));
    Value group = offset(addi(base, cx(1))),
          contributors = word(addi(base, cx(2)));
    Value single = cmpi(CmpIPredicate::eq, contributors, ci(1));
    auto firstOp = ops.front();
    auto lanes = [&](Value n, function_ref<void(Value)> body) {
      return scf::ForOp::create(
          b, loc, lane, n, cx(128), ValueRange{},
          [&](OpBuilder &inner, Location l, Value k, ValueRange) {
            body(k);
            scf::YieldOp::create(inner, l);
          });
    };
    bool speculate = arrivals && ops.size() == 4;
    Value narrow, retry;
    if (speculate) {
      narrow = shared[4];
      retry = shared[5];
      ifThen(cmpi(CmpIPredicate::eq, lane, cx(0)),
             [&] { store(ci(0), retry, {cx(0)}); });
    }
    lanes(atoms, [&](Value i) {
      Value rep = word(addi(imports, i)),
            here = cmpi(CmpIPredicate::sge, rep, ci(0));
      Value slot = indexCast(index, select(here, rep, xori(rep, ci(-1))));
      Value atom = place ? indexCast(index, load(place, {slot})) : slot;
      store(atom, liveIds, {i});
      store(select(here, ci(1), ci(0)), owned, {i});
      storeElement(b, loc, loadElement(b, loc, firstOp.getPositions(), atom),
                   positions, i);
      if (mode != 3) {
        for (int c = 0; c < 3; ++c)
          if (speculate)
            store(ci(0), narrow, {i, cx(c)});
          else
            store(cl(0), forces, {i, cx(c)});
      }
    });
    gpu::BarrierOp::create(b, loc);
    auto isOwned = [&](Value i) {
      return cmpi(CmpIPredicate::ne, load(owned, {i}), ci(0));
    };
    auto evaluate = [&](bool small) {
      Value instructions = addi(imports, atoms);
      for (unsigned set = 0; set < ops.size(); ++set) {
        auto op = ops[set];
        Value terms = offset(addi(base, cx(set + 3)));
        int arity = op.getArity();
        int stride = mode == 1 ? (arity > 3 ? 3 : 2) : 1 + arity;
        auto termLoop = lanes(terms, [&](Value k) {
          Value first = addi(instructions, muli(k, cx(stride))),
                t = offset(first);
          Value packed;
          if (mode == 1) {
            auto extend = [&](Value x) {
              return arith::ExtUIOp::create(b, loc, i64, x).getResult();
            };
            packed = extend(word(addi(first, cx(1))));
            if (arity > 3)
              packed =
                  ori(packed, shli(extend(word(addi(first, cx(2)))), cl(32)));
          }
          Values slots;
          for (int j = 0; j < arity; ++j)
            slots.push_back(
                mode == 1 ? indexCast(index,
                                      andi(shrui(packed, cl(10 * j)), cl(1023)))
                          : offset(addi(first, cx(j + 1))));
          auto yields = emitWorkUnitTuple(b, op, t, slots, positions, liveIds,
                                          box, inverse);
          for (int j = 0; j < arity; ++j)
            ifThen(isOwned(slots[j]), [&] {
              Value dest = mode == 3 ? global : forces;
              Value row = mode == 3 ? load(liveIds, {slots[j]}) : slots[j];
              for (int c = 0; c < 3; ++c) {
                Value value = quantize(
                    convertReal(
                        b, loc, vector::ExtractOp::create(b, loc, yields[j], c),
                        getElementTypeOrSelf(op.getOuts()[0].getType())),
                    forceWordScale);
                if (small) {
                  tryNarrowAdd(narrow, {row, cx(c)}, value, retry);
                } else {
                  addWord(dest, {row, cx(c)}, value);
                }
              }
              if (j == 0)
                for (auto [i, buffer] : llvm::enumerate(outputs)) {
                  Value v = yields[arity + i];
                  if (isa<FloatType>(v.getType()))
                    addWord(buffer, {cx(0)}, quantize(v, bondedEnergyScale));
                  else
                    storeElement(b, loc, v, buffer, t);
                }
            });
        });
        if (op->hasAttr("mdir.bonded_cmap"))
          termLoop->setAttr("mdir.fixed_slot_rounding", b.getUnitAttr());
        instructions = addi(instructions, muli(terms, cx(stride)));
      }
    };
    evaluate(speculate);
    if (speculate) {
      gpu::BarrierOp::create(b, loc);
      ifThenElse(
          cmpi(CmpIPredicate::ne, load(retry, {cx(0)}), ci(0)),
          [&] {
            lanes(atoms, [&](Value i) {
              for (int c = 0; c < 3; ++c)
                store(cl(0), forces, {i, cx(c)});
            });
            gpu::BarrierOp::create(b, loc);
            evaluate(false);
          },
          [&] {
            lanes(atoms, [&](Value i) {
              for (int c = 0; c < 3; ++c)
                store(arith::ExtSIOp::create(b, loc, i64,
                                             load(narrow, {i, cx(c)})),
                      forces, {i, cx(c)});
            });
          });
    }
    if (mode != 3) {
      gpu::BarrierOp::create(b, loc);
      lanes(atoms, [&](Value i) {
        ifThen(isOwned(i), [&] {
          Value atom = load(liveIds, {i});
          auto accumulate = [&] {
            for (int c = 0; c < 3; ++c)
              addWord(global, {atom, cx(c)}, load(forces, {i, cx(c)}));
          };
          if (arrivals) {
            ifThenElse(
                single,
                [&] {
                  publish(forces, i, firstOp.getOuts()[0], atom,
                          [&](int component, Value word) {
                            if (checkWords)
                              store(word, global, {atom, cx(component)});
                          });
                },
                accumulate);
          } else {
            accumulate();
          }
        });
      });
      // All producers enter the barrier before lane zero joins the device-wide
      // release sequence. Only the last block reads and recycles this interval.
      if (arrivals) {
        ifThen(notNarrow(single), [&] {
          gpu::BarrierOp::create(b, loc);
          ifThen(cmpi(CmpIPredicate::eq, lane, cx(0)), [&] {
            Value address =
                addi(baseAddress(arrivals), muli(indexCast(i64, group), cl(4)));
            auto ptr = LLVM::LLVMPointerType::get(context, 1);
            Value old = LLVM::AtomicRMWOp::create(
                b, loc, LLVM::AtomicBinOp::add,
                LLVM::IntToPtrOp::create(b, loc, ptr, address), ci(1),
                LLVM::AtomicOrdering::acq_rel, "device");
            Value last =
                cmpi(CmpIPredicate::eq, addi(old, ci(1)), contributors);
            store(select(last, ci(1), ci(0)), owned, {cx(0)});
            ifThen(last, [&] { store(ci(0), arrivals, {group}); });
          });
          gpu::BarrierOp::create(b, loc);
          Value file = addi(muli(group, cx(128)), lane);
          Value n = memref::DimOp::create(b, loc, global, 0);
          ifThen(andi(cmpi(CmpIPredicate::ne, load(owned, {cx(0)}), ci(0)),
                      cmpi(CmpIPredicate::ult, file, n)),
                 [&] {
                   Value atom =
                       place ? indexCast(index, load(place, {file})) : file;
                   publish(global, atom, firstOp.getOuts()[0], atom,
                           [&](int component, Value) {
                             if (!checkWords)
                               store(cl(0), global, {atom, cx(component)});
                           });
                 });
        });
      }
    }
  }
};
} // namespace
gpu::LaunchOp kernels::emitBondedUnits(
    OpBuilder &b, ArrayRef<md_exec::TupleForOp> ops, Value blocks,
    Value instructions, Value place, Value box, Value inverse, Value forceWords,
    ValueRange observerTerms, int mode, Value arrivals, bool checkWords) {
  auto first = ops.front();
  Location loc = first.getLoc();
  auto space =
      gpu::AddressSpaceAttr::get(b.getContext(), gpu::AddressSpace::Workgroup);
  auto mem = [&](ArrayRef<int64_t> shape, Type type) {
    return MemRefType::get(shape, type, MemRefLayoutAttrInterface(), space);
  };
  Type real = cast<MemRefType>(first.getPositions().getType()).getElementType();
  SmallVector<Type> shared{mem({256, 3}, real), mem({256}, b.getIndexType()),
                           mem({256}, b.getI32Type()),
                           mem({256, 3}, b.getI64Type())};
  if (arrivals && ops.size() == 4) {
    shared.push_back(mem({256, 3}, b.getI32Type()));
    shared.push_back(mem({1}, b.getI32Type()));
  }
  Value one = createIndex(b, loc, 1);
  auto launch =
      gpu::LaunchOp::create(b, loc, blocks, one, one, createIndex(b, loc, 128),
                            one, one, nullptr, nullptr, ValueRange{}, shared);
  OpBuilder at = OpBuilder::atBlockEnd(&launch.getBody().front());
  UnitKernel(at, loc).emit(
      ops, launch.getBlockIds().x, launch.getThreadIds().x, instructions, place,
      box, inverse, forceWords, observerTerms,
      launch.getWorkgroupAttributionBBArgs(), mode, arrivals, checkWords);
  gpu::TerminatorOp::create(at, loc);
  return launch;
}
