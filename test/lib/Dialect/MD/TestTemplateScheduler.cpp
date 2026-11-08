// One-warp execution of the actual source-derived body and scheduler.
#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mdir/Conversion/MDExecKernels.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mdir::kernels;
namespace {
static void transfer(OpBuilder &b, Location loc, Value to, Value from) {
  Type token = gpu::AsyncTokenType::get(b.getContext());
  Value ready = gpu::WaitOp::create(b, loc, token, ValueRange()).getAsyncToken();
  Value done = gpu::MemcpyOp::create(b, loc, token, ValueRange{ready}, to, from).getAsyncToken();
  gpu::WaitOp::create(b, loc, Type(), ValueRange{done});
}
struct TestTemplateScheduler
    : PassWrapper<TestTemplateScheduler, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTemplateScheduler)
  TestTemplateScheduler() = default;
  TestTemplateScheduler(const TestTemplateScheduler &pass) : PassWrapper(pass) {}
  Option<bool> rotateFrames{*this, "rotate-frames",
      llvm::cl::desc("Use different dyadic quarter-turn frames per unit"), llvm::cl::init(false)};
  Option<bool> arena{*this, "arena",
      llvm::cl::desc("Read physical work records, unit order and packed frames"), llvm::cl::init(false)};
  StringRef getArgument() const final { return "test-md-template-scheduler"; }
  StringRef getDescription() const final {
    return "Executes the derived WW window with padded I units and a partial J batch";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
        LLVM::LLVMDialect, memref::MemRefDialect, scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<mdir::md_exec::TemplateForOp> loops;
    module.walk([&](mdir::md_exec::TemplateForOp loop) { loops.push_back(loop); });
    if (loops.size() != 1) {
      module.emitError("scheduler fixture expects one fused template loop");
      return signalPassFailure();
    }
    auto loop = loops.front();
    if (loop.getOuts().size() != 1 || loop.getReduce().size() != 2 ||
        !loop.getIns().empty() || !loop.getReduce()[0].getType().isF64() ||
        loop.getReduce()[1].getType() != VectorType::get({9}, Float64Type::get(&getContext())) ||
        !llvm::equal(loop.getPairs(), ArrayRef<int32_t>{1, 1, 1, 0, 0, 0, 0, 0, 0})) {
      loop.emitError("scheduler fixture expects WW forces, scalar energy and strain only");
      return signalPassFailure();
    }
    OpBuilder b(&getContext());
    Location loc = loop.getLoc();
    Type f32 = b.getF32Type(), f64 = b.getF64Type();
    auto hostForce = MemRefType::get({153, 3}, f32);
    auto hostSums = MemRefType::get({32, 10}, f64);
    auto type = b.getFunctionType({hostForce, hostSums}, {});
    auto probe = module.lookupSymbol<func::FuncOp>("template_scheduler_probe");
    if (probe && (!probe.isExternal() || probe.getFunctionType() != type)) {
      loop.emitError("scheduler probe declaration has the wrong type");
      return signalPassFailure();
    }
    if (!probe) {
      probe = func::FuncOp::create(loc, "template_scheduler_probe", type);
      module.push_back(probe);
    }
    Block *entry = probe.addEntryBlock();
    b.setInsertionPointToEnd(entry);
    auto allocate = [&](MemRefType host) -> Value {
      auto device = MemRefType::get(host.getShape(), host.getElementType(),
          MemRefLayoutAttrInterface(), b.getI64IntegerAttr(1));
      return gpu::AllocOp::create(b, loc, device, Type(), ValueRange(),
                                  ValueRange(), ValueRange()).getMemref();
    };
    Value forces = allocate(hostForce), sums = allocate(hostSums);
    transfer(b, loc, forces, entry->getArgument(0));
    Value one = createIndex(b, loc, 1), threads = createIndex(b, loc, 32);
    auto launch = gpu::LaunchOp::create(b, loc, one, one, one, threads, one, one);
    OpBuilder k = OpBuilder::atBlockEnd(&launch.getBody().front());
    auto word = [&](OpBuilder &body, int64_t v) -> Value {
      return arith::ConstantIntOp::create(body, loc, v, 32);
    };
    auto index = [&](OpBuilder &body, Value value) -> Value {
      return arith::IndexCastOp::create(body, loc, body.getIndexType(), value);
    };
    auto makeVector = [&](OpBuilder &body, Value x, Value y, Value z) -> Value {
      return vector::FromElementsOp::create(body, loc, VectorType::get({3}, f32),
                                            ValueRange{x, y, z});
    };
    auto frame = [&](OpBuilder &body, Value unit, bool partner) {
      Value zero = createReal(body, loc, f32, 0.0);
      Value x = arith::MulFOp::create(body, loc,
          arith::UIToFPOp::create(body, loc, f32, unit),
          createReal(body, loc, f32, partner ? 1.0 / 128.0 : 1.0 / 64.0));
      if (partner) x = arith::AddFOp::create(body, loc, x, createReal(body, loc, f32, 0.5));
      Value anchor = makeVector(body, x, zero, zero);
      Value offset = createReal(body, loc, f32, 1.0 / 16.0);
      Value h1 = makeVector(body, offset, zero, zero);
      Value h2 = makeVector(body, zero, offset, zero);
      if (rotateFrames) {
        Value even = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::eq,
            arith::AndIOp::create(body, loc, unit, word(body, 1)), word(body, 0));
        Value negative = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::ne,
            arith::AndIOp::create(body, loc, unit, word(body, 2)), word(body, 0));
        Value signedOffset = arith::SelectOp::create(body, loc, negative,
            createReal(body, loc, f32, -1.0 / 16.0), offset);
        Value opposite = arith::NegFOp::create(body, loc, signedOffset);
        h1 = makeVector(body, arith::SelectOp::create(body, loc, even, signedOffset, zero),
            arith::SelectOp::create(body, loc, even, zero, signedOffset), zero);
        h2 = makeVector(body, arith::SelectOp::create(body, loc, even, zero, opposite),
            arith::SelectOp::create(body, loc, even, signedOffset, zero), zero);
      }
      return SmallVector<Value, 3>{anchor,
          arith::AddFOp::create(body, loc, anchor, h1),
          arith::AddFOp::create(body, loc, anchor, h2)};
    };
    Value lane = arith::IndexCastOp::create(k, loc, k.getI32Type(), launch.getThreadIds().x);
    Value first = arith::AndIOp::create(k, loc, lane, word(k, 15));
    Value valid = arith::CmpIOp::create(k, loc, arith::CmpIPredicate::ult, first, word(k, 13));
    TemplateUnit unit{TemplatePairClass::WaterWater, lane, word(k, 0), word(k, 35), valid};
    unit.first = frame(k, first, false);
    unit.hasSite = true; unit.alpha = loop.getAlpha().convertToDouble();
    unit.sumTypes = {f64, VectorType::get({9}, f64)};
    Block &source = loop.getWater().front();
    IRMapping captures;
    bool badCapture = false;
    std::function<void(Value)> bring = [&](Value value) {
      if (captures.contains(value)) return;
      Operation *definition = value.getDefiningOp();
      if (!definition || !isPure(definition) || definition->getNumRegions()) {
        badCapture = true; return;
      }
      for (Value operand : definition->getOperands()) bring(operand);
      if (!badCapture) k.clone(*definition, captures);
    };
    source.walk([&](Operation *op) {
      for (Value operand : op->getOperands())
        if (!source.getParent()->isAncestor(operand.getParentRegion())) bring(operand);
    });
    if (badCapture) {
      loop.emitError("scheduler fixture expects constant scalar captures");
      return signalPassFailure();
    }
    SmallVector<Value, 2> result;
    SmallVector<Value> arenaAllocations;
    if (arena) {
      // Fixed data only seeds the physical adapter; the same independent
      // world-position oracle checks its outputs. No build or search is
      // claimed by this isolated record-consumption fixture.
      OpBuilder setup(launch);
      auto hostWords = MemRefType::get({144}, b.getI32Type());
      auto hostOrder = MemRefType::get({16}, b.getI32Type());
      auto hostMembers = MemRefType::get({153}, b.getI32Type());
      auto hostFrames = MemRefType::get({153, 3}, f32);
      Value pairHost = memref::AllocOp::create(setup, loc, hostWords);
      Value orderHost = memref::AllocOp::create(setup, loc, hostOrder);
      Value memberHost = memref::AllocOp::create(setup, loc, hostMembers);
      Value frameHost = memref::AllocOp::create(setup, loc, hostFrames);
      auto put = [&](Value buffer, unsigned at, int32_t value) {
        memref::StoreOp::create(setup, loc, word(setup, value), buffer,
                                ValueRange{createIndex(setup, loc, at)});
      };
      put(pairHost, 0, 0); put(pairHost, 1, 0);
      put(pairHost, 2, 0); put(pairHost, 3, 35);
      for (unsigned j = 0; j != 35; ++j) {
        unsigned mask = 0;
        for (unsigned i = 0; i != 16; ++i) if ((i + j) % 3) mask |= 1 << i;
        put(pairHost, 4 + 4 * j, 16 + j);
        put(pairHost, 5 + 4 * j,
            (MDRT_TEMPLATE_IMAGE_OFFSET + (j % 5 == 0)) |
            (MDRT_TEMPLATE_IMAGE_OFFSET << MDRT_TEMPLATE_IMAGE_BITS) |
            (MDRT_TEMPLATE_IMAGE_OFFSET << (2 * MDRT_TEMPLATE_IMAGE_BITS)));
        put(pairHost, 6 + 4 * j, mask);
        put(pairHost, 7 + 4 * j, 0);
      }
      for (unsigned i = 0; i != 16; ++i) put(orderHost, i, i < 13 ? i : -1);
      for (unsigned i = 0; i != 153; ++i) put(memberHost, i, i);
      scf::ForOp::create(setup, loc, createIndex(setup, loc, 0),
          createIndex(setup, loc, 51), createIndex(setup, loc, 1), ValueRange(),
          [&](OpBuilder &body, Location, Value i, ValueRange) {
            Value initial = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::ult,
                i, createIndex(body, loc, 16));
            auto positions = scf::IfOp::create(body, loc, initial,
                [&](OpBuilder &inside, Location) {
                  Value u = arith::IndexCastOp::create(inside, loc, inside.getI32Type(), i);
                  scf::YieldOp::create(inside, loc, frame(inside, u, false));
                }, [&](OpBuilder &inside, Location) {
                  Value u = arith::IndexCastOp::create(inside, loc, inside.getI32Type(),
                      arith::SubIOp::create(inside, loc, i, createIndex(inside, loc, 16)));
                  scf::YieldOp::create(inside, loc, frame(inside, u, true));
                });
            for (unsigned site = 0; site != 3; ++site)
              storeElement(body, loc, positions.getResult(site), frameHost,
                  arith::AddIOp::create(body, loc,
                      arith::MulIOp::create(body, loc, i, createIndex(body, loc, 3)),
                      createIndex(body, loc, site)));
            scf::YieldOp::create(body, loc);
          });
      auto upload = [&](Value host) -> Value {
        auto hostType = cast<MemRefType>(host.getType());
        auto device = MemRefType::get(hostType.getShape(), hostType.getElementType(),
            MemRefLayoutAttrInterface(), setup.getI64IntegerAttr(1));
        Value buffer = gpu::AllocOp::create(setup, loc, device, Type(), ValueRange(),
                                            ValueRange(), ValueRange()).getMemref();
        transfer(setup, loc, buffer, host);
        memref::DeallocOp::create(setup, loc, host);
        arenaAllocations.push_back(buffer);
        return buffer;
      };
      Value pairs = upload(pairHost), order = upload(orderHost);
      Value members = upload(memberHost), frames = upload(frameHost);
      TemplateWorkBuffers buffers{pairs, createIndex(k, loc, 0), createIndex(k, loc, 4),
          order, members, frames, word(k, 51),
          arith::ConstantOp::create(k, loc,
              DenseElementsAttr::get(VectorType::get({3}, f64), ArrayRef<double>{2, 2, 2})), true};
      IRMapping mapped = captures;
      mapped.map(loop.getOuts()[0], forces);
      result = emitTemplateWorkUnit(k, loc, loop, buffers, word(k, 0), lane, mapped);
    } else {
      result = emitTemplateClassUnit(k, loc, unit,
        [&](OpBuilder &body, Value partner) {
          TemplatePartner loaded;
          loaded.frame = frame(body, partner, true);
          Value image = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::eq,
              arith::RemUIOp::create(body, loc, partner, word(body, 5)), word(body, 0));
          Value zero = createReal(body, loc, f32, 0.0);
          loaded.shift = makeVector(body, arith::SelectOp::create(body, loc, image,
              createReal(body, loc, f32, 2.0), zero), zero, zero);
          int32_t masks[3] = {};
          for (unsigned residue = 0; residue != 3; ++residue)
            for (unsigned i = 0; i != 16; ++i)
              if ((i + residue) % 3) masks[residue] |= 1 << i;
          Value residue = arith::RemUIOp::create(body, loc, partner, word(body, 3));
          Value a = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::eq, residue, word(body, 0));
          Value c = arith::CmpIOp::create(body, loc, arith::CmpIPredicate::eq, residue, word(body, 1));
          loaded.mask = arith::SelectOp::create(body, loc, a, word(body, masks[0]),
              arith::SelectOp::create(body, loc, c, word(body, masks[1]), word(body, masks[2])));
          return loaded;
        }, [&](OpBuilder &body, ValueRange coords, ValueRange, ValueRange, ValueRange sites) {
          IRMapping mapping = captures;
          mapping.map(source.getArguments(), coords);
          mapTemplateSiteOffsets(loop, 0, sites, mapping);
          return emitTemplateWindowContributions(body, loc, loop, 0, mapping);
        }, [&](OpBuilder &body, ValueRange values) {
          Value base = arith::MulIOp::create(body, loc, first, word(body, 3));
          for (auto [site, force] : llvm::enumerate(values))
            emitAtomicAdd(body, loc, force, forces,
                index(body, arith::AddIOp::create(body, loc, base, word(body, site))));
        }, [&](OpBuilder &body, Value partner, ValueRange values) {
          Value base = arith::AddIOp::create(body, loc, word(body, 48),
              arith::MulIOp::create(body, loc, partner, word(body, 3)));
          for (auto [site, force] : llvm::enumerate(values))
            emitAtomicAdd(body, loc, force, forces,
                index(body, arith::AddIOp::create(body, loc, base, word(body, site))));
        });
    }
    Value row = index(k, lane);
    memref::StoreOp::create(k, loc, result[0], sums, ValueRange{row, createIndex(k, loc, 0)});
    for (unsigned c = 0; c != 9; ++c)
      memref::StoreOp::create(k, loc, vector::ExtractOp::create(k, loc, result[1], c),
          sums, ValueRange{row, createIndex(k, loc, c + 1)});
    gpu::TerminatorOp::create(k, loc);
    b.setInsertionPointAfter(launch);
    transfer(b, loc, entry->getArgument(0), forces);
    transfer(b, loc, entry->getArgument(1), sums);
    // The runtime's free accepts a generic pointer to device storage.
    Value plainForces = memref::MemorySpaceCastOp::create(b, loc, hostForce, forces);
    Value plainSums = memref::MemorySpaceCastOp::create(b, loc, hostSums, sums);
    gpu::DeallocOp::create(b, loc, Type(), ValueRange(), plainForces);
    gpu::DeallocOp::create(b, loc, Type(), ValueRange(), plainSums);
    for (Value buffer : arenaAllocations) {
      auto device = cast<MemRefType>(buffer.getType());
      auto host = MemRefType::get(device.getShape(), device.getElementType());
      Value generic = memref::MemorySpaceCastOp::create(b, loc, host, buffer);
      gpu::DeallocOp::create(b, loc, Type(), ValueRange(), generic);
    }
    func::ReturnOp::create(b, loc);
    loop->getParentOfType<func::FuncOp>().erase();
    for (Operation &op : llvm::make_early_inc_range(module))
      if (isa<mdir::md::MDDialect>(op.getDialect())) op.erase();
  }
};
} // namespace
namespace mdir::test {
void registerTestTemplateScheduler() { PassRegistration<TestTemplateScheduler>(); }
}
