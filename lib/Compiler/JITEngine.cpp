#include "JITEngine.h"
#include "JITMemory.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/DynamicLibrary.h"

using namespace llvm;
using namespace llvm::orc;
using namespace mdir::compiler;
namespace {
Error ownedError(Error error) {
  if (!error)
    return Error::success();
  // ORC errors may retain references to session storage. Copy before teardown.
  return createStringError(inconvertibleErrorCode(), toString(std::move(error)));
}
}
Expected<std::unique_ptr<JITEngine>> JITEngine::create(
    mlir::ModuleOp input, std::unique_ptr<TargetMachine> target,
    ArrayRef<std::string> libraries, StringRef entry, StringRef codegen,
    bool cache) {
  Triple triple = target->getTargetTriple();
  // Fail closed: this validation decodes ELF .eh_frame, not COFF/Mach-O.
  if (!triple.isOSBinFormatELF() ||
      (triple.getArch() != Triple::x86_64 && triple.getArch() != Triple::aarch64))
    return createStringError(inconvertibleErrorCode(),
                             "JIT ownership: unsupported host unwind format");
  auto context = std::make_unique<LLVMContext>();
  auto module = mlir::translateModuleToLLVMIR(input, *context);
  if (!module)
    return createStringError(inconvertibleErrorCode(), "cannot translate host module");
  module->setTargetTriple(triple);
  module->setDataLayout(target->createDataLayout());
  auto *function = module->getFunction(entry);
  if (!function || !function->getReturnType()->isVoidTy())
    return createStringError(inconvertibleErrorCode(), "invalid simulation entry");
  // Match MLIR's packed-entry ABI, with a wrapper only for our void entry.
  // The ABI is described by MLIR ExecutionEngine (LLVM 23.1.2).
  IRBuilder<> builder(*context);
  auto *packed = Function::Create(
      FunctionType::get(builder.getVoidTy(), builder.getPtrTy(), false),
      GlobalValue::ExternalLinkage, "_mlir_" + entry.str(), *module);
  builder.SetInsertPoint(BasicBlock::Create(*context, "entry", packed));
  SmallVector<Value *> arguments;
  for (auto [index, argument] : enumerate(function->args())) {
    auto *slot = builder.CreateGEP(builder.getPtrTy(), packed->getArg(0),
                                   builder.getInt64(index));
    arguments.push_back(builder.CreateLoad(
        argument.getType(), builder.CreateLoad(builder.getPtrTy(), slot)));
  }
  builder.CreateCall(function, arguments);
  builder.CreateRetVoid();
  // Section placement is retained for locality, but validity depends on the
  // allocated code and relocated FDEs, including functions added later by ORC.
  StringRef section = triple.getArch() == Triple::x86_64 &&
                              target->getCodeModel() == CodeModel::Large
                          ? ".ltext" : ".text";
  for (auto &f : *module)
    if (!f.isDeclaration())
      f.setSection(section);
  auto layout = module->getDataLayout();
  auto engine = std::unique_ptr<JITEngine>(new JITEngine);
  engine->cache = std::make_unique<HostObjectCache>(
      cache ? CompileCacheConfig::fromEnvironment() : std::nullopt,
      describeMachine(*target, codegen),
      module->getModuleIdentifier());
  engine->perfListener.reset(JITEventListener::createPerfJITEventListener());
  if (!engine->perfListener)
    engine->perfListener.reset(JITEventListener::createIntelJITEventListener());
  auto created = LLJITBuilder()
      .setDataLayout(layout)
      .setCompileFunctionCreator([&](JITTargetMachineBuilder)
          -> Expected<std::unique_ptr<IRCompileLayer::IRCompiler>> {
        return std::make_unique<TMOwningSimpleCompiler>(std::move(target),
                                                        engine->cache.get());
      })
      .setObjectLinkingLayerCreator([triple, &engine](ExecutionSession &session,
                                            jitlink::JITLinkMemoryManager &)
          -> Expected<std::unique_ptr<ObjectLayer>> {
        auto layer = std::make_unique<RTDyldObjectLinkingLayer>(
            session, [triple](const MemoryBuffer &) {
              return std::make_unique<JITMemory>(triple);
            });
        if (auto *listener = JITEventListener::createGDBRegistrationListener())
          layer->registerJITEventListener(*listener);
        if (engine->perfListener)
          layer->registerJITEventListener(*engine->perfListener);
        return layer;
      }).create();
  if (!created)
    return ownedError(created.takeError());
  engine->jit = std::move(*created);
  auto &main = engine->jit->getMainJITDylib();
  for (auto &path : libraries) {
    // Match the runtime's permanent in-process loading: device selection and
    // stop-handler symbols also need to be visible through DynamicLibrary.
    std::string message;
    auto library = sys::DynamicLibrary::getPermanentLibrary(path.c_str(), &message);
    if (!library.isValid()) {
      // cuFFT comes from NVIDIA's wheel of the extra `cuda` of the Python
      // package, or from a CUDA toolkit (D228).
      if (StringRef(message).contains("libcufft"))
        message += "; the GPU runtime needs cuFFT of CUDA 13: install the "
                   "Python package with its extra, pip install 'mdir[cuda]', "
                   "or put the lib64 of a CUDA 13 toolkit on LD_LIBRARY_PATH";
      return createStringError(inconvertibleErrorCode(), "cannot load runtime: " + message);
    }
    auto generator = DynamicLibrarySearchGenerator::Load(path.c_str(), layout.getGlobalPrefix());
    if (!generator)
      return ownedError(generator.takeError());
    main.addGenerator(std::move(*generator));
  }
  auto process = DynamicLibrarySearchGenerator::GetForCurrentProcess(layout.getGlobalPrefix());
  if (!process)
    return ownedError(process.takeError());
  main.addGenerator(std::move(*process));
  if (auto error = engine->jit->addIRModule(ThreadSafeModule(std::move(module), std::move(context))))
    return ownedError(std::move(error));
  return std::move(engine);
}
JITEngine::~JITEngine() {
  if (initialized)
    if (auto error = jit->deinitialize(jit->getMainJITDylib()))
      errs() << "JIT deinitialization failed: " << toString(std::move(error)) << '\n';
  // LLJIT ends the session; object removal deregisters frames. The manager's
  // own destructor also deregisters, covering failed materialization.
}
Error JITEngine::registerSymbols(function_ref<SymbolMap(MangleAndInterner)> map) {
  return ownedError(jit->getMainJITDylib().define(absoluteSymbols(
      map(MangleAndInterner(jit->getExecutionSession(), jit->getDataLayout())))));
}
Error JITEngine::initialize() {
  if (initialized)
    return Error::success();
  if (auto error = jit->initialize(jit->getMainJITDylib()))
    return ownedError(std::move(error));
  initialized = true;
  return Error::success();
}
Expected<void (*)(void **)> JITEngine::lookupPacked(StringRef name) {
  auto symbol = jit->lookup("_mlir_" + name.str());
  if (!symbol)
    return ownedError(symbol.takeError());
  return symbol->toPtr<void (*)(void **)>();
}
