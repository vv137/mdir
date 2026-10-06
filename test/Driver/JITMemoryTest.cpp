// Deterministic tests at the production memory-manager boundary.
#include "../../lib/Compiler/JITMemory.h"
#include "../../lib/Compiler/JITEngine.h"
#include "mdir/Compiler/Compile.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Endian.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <stdexcept>

using namespace llvm;
using namespace mdir::compiler;
static void require(bool condition, StringRef message) {
  if (!condition)
    report_fatal_error(message);
}
static void reject(Error error, StringRef reason) {
  require(bool(error), "invalid layout was accepted");
  auto message = toString(std::move(error));
  require(StringRef(message).contains(reason), message);
}
// DWARF/LSB .eh_frame: zR CIE, pcrel/sdata4 start, udata4 range.
static uint8_t *frame(JITMemory &memory, uint8_t *code, uint32_t range = 64) {
  auto *bytes = memory.allocateDataSection(44, 8, 1, ".eh_frame", true);
  const uint8_t cie[] = {16,0,0,0, 0,0,0,0, 1,'z','R',0, 1,0x78,16,1,
                         0x1b,0x0c,7,8};
  std::memcpy(bytes, cie, sizeof(cie));
  std::memset(bytes + 20, 0, 24);
  support::endian::write32le(bytes + 20, 16);
  support::endian::write32le(bytes + 24, 24);
  auto displacement = reinterpret_cast<intptr_t>(code) -
                      reinterpret_cast<intptr_t>(bytes + 28);
  require(displacement >= INT32_MIN && displacement <= INT32_MAX,
          "test fixture cannot encode PC-relative address");
  support::endian::write32le(bytes + 28, displacement);
  support::endian::write32le(bytes + 32, range);
  memory.registerEHFrames(bytes, reinterpret_cast<uintptr_t>(bytes), 44);
  return bytes;
}
static void throwFromJIT() { throw std::runtime_error("through JIT"); }
int main(int argc, char **) {
  if (argc > 1) {
    JITMemoryMapper mapper;
    mapper.registered = true;
    sys::MemoryBlock block;
    mapper.releaseMappedMemory(block);
    report_fatal_error("unsafe release was accepted");
  }
  reject(validateJITRanges({}, {}, {}), "missing");
  reject(validateJITRanges({{100,200}}, {{190,210}}, {}), "outside");
  reject(validateJITRanges({{100,200},{400,500}}, {{110,130}}, {{250,300}}), "overlaps");
  reject(validateJITRanges({{100,100}}, {{100,110}}, {}), "invalid");
  reject(validateJITRanges({{100,200}}, {{190,180}}, {}), "outside");
  require(!validateJITRanges({{100,200}}, {{110,130}}, {{200,300}}),
          "adjacent interval rejected");
  Triple triple(sys::getDefaultTargetTriple());
  for (int mode = 0; mode != 6; ++mode) {
    JITMemory memory(triple);
    memory.reserveAllocationSpace(64, Align(16), 64, Align(8), 0, Align(8));
    auto *code = memory.allocateCodeSection(64, 16, 0, ".text.late");
    uint8_t *bytes = nullptr;
    if (mode != 0)
      bytes = frame(memory, code, mode == 1 ? 65 : 64);
    if (mode == 2)
      bytes[0] = 255; // Malformed record length must be rejected before registration.
    if (mode == 5)
      memory.allocateCodeSection(1024 * 1024, 16, 2, ".text.separate");
    std::string message;
    bool failed = memory.finalizeMemory(&message);
    require(failed == (mode < 3 || mode == 5), "unexpected finalization result");
    if (mode == 0)
      require(StringRef(message).contains("one unwind table"), message);
    if (mode == 1)
      require(StringRef(message).contains("outside"), message);
    if (mode == 2)
      require(StringRef(message).contains("parse"), message);
    if (mode == 5)
      require(StringRef(message).contains("owned allocation"), message);
    if (mode == 3) {
      require(memory.finalizeMemory(&message), "duplicate finalization accepted");
      require(StringRef(message).contains("more than once"), message);
    }
    if (mode == 4) {
      memory.deregisterEHFrames();
      memory.registerEHFrames(bytes, reinterpret_cast<uintptr_t>(bytes), 44);
      require(memory.finalizeMemory(&message), "registration after release accepted");
    }
    memory.deregisterEHFrames();
    memory.deregisterEHFrames();
    // An exception after rejection/removal checks that no stale entry is visible.
    try { throw std::runtime_error("delayed"); }
    catch (const std::runtime_error &) {}
  }
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();
  mlir::MLIRContext context(mdir::compiler::getRegistry());
  for (auto configuration : {0, 1, 2}) {
    bool fails = configuration == 0;
    auto module = mlir::parseSourceString<mlir::ModuleOp>(fails ? R"mlir(
      module {
        llvm.func @missing_jit_ownership_test_symbol()
        llvm.func @entry() attributes {uwtable_kind = #llvm.uwtableKind<sync>} {
          llvm.call @missing_jit_ownership_test_symbol() : () -> ()
          llvm.return
        }
        llvm.mlir.global_ctors ctors = [@entry], priorities = [0 : i32], data = [#llvm.zero]
      }
    )mlir" : R"mlir(
      module {
        llvm.func @jit_ownership_throw()
        llvm.func @entry() attributes {uwtable_kind = #llvm.uwtableKind<sync>} {
          llvm.call @jit_ownership_throw() : () -> ()
          llvm.return
        }
        llvm.func @init() attributes {uwtable_kind = #llvm.uwtableKind<sync>} { llvm.return }
        llvm.mlir.global_ctors ctors = [@init], priorities = [0 : i32], data = [#llvm.zero]
        llvm.mlir.global_dtors dtors = [@init], priorities = [0 : i32], data = [#llvm.zero]
      }
    )mlir", &context);
    require(bool(module), "cannot parse JIT fixture");
    auto builder = cantFail(orc::JITTargetMachineBuilder::detectHost());
    builder.setCodeModel(configuration == 2 ? CodeModel::Small : CodeModel::Large);
    auto target = cantFail(builder.createTargetMachine());
    auto engine = cantFail(JITEngine::create(*module, std::move(target), {}, "entry", ""));
    cantFail(engine->registerSymbols([](orc::MangleAndInterner interner) {
      orc::SymbolMap symbols;
      symbols[interner("jit_ownership_throw")] = {
          orc::ExecutorAddr::fromPtr(&throwFromJIT), JITSymbolFlags::Exported};
      return symbols;
    }));
    auto initialized = engine->initialize();
    require(bool(initialized) == fails, "unexpected initialization result");
    if (fails) {
      require(StringRef(toString(std::move(initialized))).contains("Failed to materialize"),
              "missing-symbol error was not propagated");
    } else {
      auto function = cantFail(engine->lookupPacked("entry"));
      bool caught = false;
      try { function(nullptr); }
      catch (const std::runtime_error &error) {
        caught = std::strcmp(error.what(), "through JIT") == 0;
      }
      require(caught, "exception did not unwind through the JIT entry");
    }
    engine.reset();
    try { throw std::runtime_error("after engine destruction"); }
    catch (const std::runtime_error &) {}
  }
  outs() << "JIT memory negative and lifecycle checks passed\n";
  return 0;
}
