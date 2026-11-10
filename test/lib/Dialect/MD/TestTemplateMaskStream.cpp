// Exact-integer execution fixture for the production mask-stream emitter.
#include "mdir/Conversion/TemplateWindowKernels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace {
static void transfer(OpBuilder &b, Location loc, Value to, Value from) {
  Type token = gpu::AsyncTokenType::get(b.getContext());
  Value ready = gpu::WaitOp::create(b, loc, token, ValueRange()).getAsyncToken();
  Value done = gpu::MemcpyOp::create(b, loc, token, ValueRange{ready}, to, from).getAsyncToken();
  gpu::WaitOp::create(b, loc, Type(), ValueRange{done});
}
struct TestTemplateMaskStream
    : PassWrapper<TestTemplateMaskStream, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTemplateMaskStream)
  StringRef getArgument() const final { return "test-md-template-mask-stream"; }
  StringRef getDescription() const final { return "Executes the production mask transpose against exact integer fixtures"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
             LLVM::LLVMDialect, memref::MemRefDialect>();
  }
  void runOnOperation() final {
    auto f = getOperation().lookupSymbol<func::FuncOp>("mask_stream_probe");
    if (!f || !f.isExternal() || f.getNumArguments() != 3) {
      getOperation().emitError("mask stream fixture expects a three-argument declaration");
      return signalPassFailure();
    }
    auto words = dyn_cast<MemRefType>(f.getArgumentTypes()[0]);
    auto lengths = dyn_cast<MemRefType>(f.getArgumentTypes()[1]);
    if (!words || !lengths || words.getRank()!=2 || words.getDimSize(1)!=32 ||
        !words.hasStaticShape() || !words.getElementType().isInteger(32) ||
        lengths != MemRefType::get({words.getDimSize(0)}, words.getElementType()) ||
        f.getArgumentTypes()[2] != words) {
      f.emitError("expected static memref<Nx32xi32>, memref<Nxi32>, memref<Nx32xi32>");
      return signalPassFailure();
    }
    Location loc=f.getLoc();
    Block *entry=f.addEntryBlock();
    OpBuilder b=OpBuilder::atBlockEnd(entry);
    auto idx=[&](OpBuilder &at,int64_t n)->Value {return arith::ConstantIndexOp::create(at,loc,n);};
    auto word=[&](OpBuilder &at,int64_t n)->Value {return arith::ConstantIntOp::create(at,loc,n,32);};
    auto alloc=[&](MemRefType host)->Value {
      auto type=MemRefType::get(host.getShape(),host.getElementType(),
          MemRefLayoutAttrInterface(),b.getI64IntegerAttr(1));
      return gpu::AllocOp::create(b,loc,type,Type(),ValueRange(),ValueRange(),ValueRange()).getMemref();
    };
    Value masks=alloc(words), counts=alloc(lengths), outputs=alloc(words);
    transfer(b,loc,masks,entry->getArgument(0));
    transfer(b,loc,counts,entry->getArgument(1));
    Value one=idx(b,1);
    auto launch=gpu::LaunchOp::create(b,loc,idx(b,words.getDimSize(0)),one,one,idx(b,32),one,one);
    OpBuilder k=OpBuilder::atBlockEnd(&launch.getBody().front());
    Value row=launch.getBlockIds().x, thread=launch.getThreadIds().x;
    Value lane=arith::IndexCastOp::create(k,loc,k.getI32Type(),thread);
    Value mask=memref::LoadOp::create(k,loc,masks,ValueRange{row,thread});
    Value count=memref::LoadOp::create(k,loc,counts,ValueRange{row});
    Value has=arith::CmpIOp::create(k,loc,arith::CmpIPredicate::ult,lane,count);
    mask=arith::SelectOp::create(k,loc,has,mask,word(k,0));
    Value stream=mdir::kernels::emitTemplateMaskStream(k,loc,mask,lane);
    stream=arith::AndIOp::create(k,loc,stream,word(k,65535));
    memref::StoreOp::create(k,loc,stream,outputs,ValueRange{row,thread});
    gpu::TerminatorOp::create(k,loc);
    b.setInsertionPointAfter(launch);
    transfer(b,loc,entry->getArgument(2),outputs);
    // Runtime free takes a generic pointer, matching TestTemplateScheduler.
    for (Value buffer : {masks, counts, outputs}) {
      auto device=cast<MemRefType>(buffer.getType());
      auto host=MemRefType::get(device.getShape(),device.getElementType());
      Value plain=memref::MemorySpaceCastOp::create(b,loc,host,buffer);
      gpu::DeallocOp::create(b,loc,Type(),ValueRange(),plain);
    }
    func::ReturnOp::create(b,loc);
  }
};
}
namespace mdir::test {
void registerTestTemplateMaskStream() { PassRegistration<TestTemplateMaskStream>(); }
}
