// The keys, entries, and bound of the compile cache of host objects
// (D212): a changed key part misses, a damaged entry is
// rejected, and eviction removes the entries used least recently.
#include "mdir/Compiler/CompileCache.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include <chrono>

using namespace llvm;
using namespace mdir::compiler;

static int failures = 0;
static void check(bool ok, const char *what) {
  outs() << (ok ? "ok: " : "FAILED: ") << what << "\n";
  failures += !ok;
}

static std::unique_ptr<TargetMachine> machine(bool hostFeatures) {
  auto builder = cantFail(orc::JITTargetMachineBuilder::detectHost());
  builder.setCodeGenOptLevel(CodeGenOptLevel::Default);
  if (!hostFeatures) {
    // A baseline CPU of the same triple: another node of a cluster.
    builder.setCPU(builder.getTargetTriple().isX86() ? "x86-64" : "generic");
    builder.getFeatures() = SubtargetFeatures();
  }
  return cantFail(builder.createTargetMachine());
}

/// A module named `program` with one function that returns `value`.
static std::unique_ptr<Module> program(LLVMContext &context,
                                       const TargetMachine &target, int value) {
  auto module = std::make_unique<Module>("program", context);
  module->setDataLayout(target.createDataLayout());
  module->setTargetTriple(target.getTargetTriple());
  auto *type = FunctionType::get(Type::getInt32Ty(context), false);
  auto *function =
      Function::Create(type, Function::ExternalLinkage, "entry", *module);
  IRBuilder<> builder(BasicBlock::Create(context, "body", function));
  builder.CreateRet(builder.getInt32(value));
  return module;
}

/// The relocatable object of `module`, as the JIT's compiler makes it.
static std::string objectOf(Module &module, TargetMachine &target) {
  SmallVector<char, 0> object;
  raw_svector_ostream os(object);
  legacy::PassManager passes;
  MCContext *context = nullptr;
  if (target.addPassesToEmitMC(passes, context, os))
    report_fatal_error("cannot emit an object");
  passes.run(module);
  return std::string(object.begin(), object.end());
}

static std::string read(StringRef path) {
  auto file = MemoryBuffer::getFile(path, /*IsText=*/false,
                                    /*RequiresNullTerminator=*/false);
  return file ? (*file)->getBuffer().str() : std::string();
}

static void write(StringRef path, StringRef data) {
  std::error_code error;
  raw_fd_ostream os(path, error);
  os << data;
}

static void age(StringRef path, std::chrono::seconds seconds) {
  cantFail(errorCodeToError(sys::fs::setLastAccessAndModificationTime(
      path, std::chrono::system_clock::now() - seconds)));
}

int main(int argc, char **argv) {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  if (argc < 2)
    return 2;
  SmallString<256> directory(argv[1]);

  // Keys: the CPU and its features, the scheduler, and the module.
  auto host = machine(true);
  std::string hostKey = describeMachine(*host, "pre-RA-sched=fast");
  check(hostKey != describeMachine(*machine(false), "pre-RA-sched=fast"),
        "another CPU and features change the key");
  check(hostKey != describeMachine(*host, ""),
        "another scheduler changes the key");
  check(hostKey == describeMachine(*machine(true), "pre-RA-sched=fast"),
        "the same machine gives the same key");
  LLVMContext context;
  auto one = program(context, *host, 1), two = program(context, *host, 2);
  HostObjectCache cache(std::nullopt, hostKey, "program");
  HostObjectCache generic(std::nullopt,
                          describeMachine(*machine(false), "pre-RA-sched=fast"),
                          "program");
  std::string key = cache.getKey(*one);
  check(key != cache.getKey(*two), "another module changes the key");
  check(key == cache.getKey(*program(context, *host, 1)),
        "the same module gives the same key");
  check(key != generic.getKey(*one), "another machine changes the module's key");
  check(HostObjectCache::getEntryName(key) !=
            HostObjectCache::getEntryName(cache.getKey(*two)),
        "another key names another entry");

  // Entries: the key, the length, the hash, and the object are checked.
  std::string objects = HostObjectCache::getObjectDirectory(directory);
  cantFail(errorCodeToError(sys::fs::create_directories(objects)));
  std::string object = objectOf(*one, *host);
  auto entry = [&](StringRef name) {
    SmallString<256> path(objects);
    sys::path::append(path, name);
    return std::string(path);
  };
  std::string path = entry(HostObjectCache::getEntryName(key));
  cantFail(HostObjectCache::writeEntry(
      path, key, MemoryBufferRef(object, "program"), 1.5));
  double seconds = 0.0;
  bool rejected = true;
  auto found = HostObjectCache::readEntry(path, key, seconds, rejected);
  check(found && found->getBuffer() == object && seconds == 1.5 && !rejected,
        "an entry gives back its object and its time");
  found = HostObjectCache::readEntry(entry("absent.o"), key, seconds, rejected);
  check(!found && !rejected, "an absent entry is a miss, not a rejection");
  found = HostObjectCache::readEntry(path, cache.getKey(*two), seconds,
                                     rejected);
  check(!found && rejected, "an entry under another key is rejected");
  std::string intact = read(path);
  write(path, StringRef(intact).take_front(intact.size() / 2));
  found = HostObjectCache::readEntry(path, key, seconds, rejected);
  check(!found && rejected, "a truncated entry is rejected");
  std::string flipped = intact;
  flipped[flipped.size() - 100] ^= 0x40;
  write(path, flipped);
  found = HostObjectCache::readEntry(path, key, seconds, rejected);
  check(!found && rejected, "an object with a byte flipped is rejected");
  flipped = intact;
  flipped[7] ^= 0x01;
  write(path, flipped);
  found = HostObjectCache::readEntry(path, key, seconds, rejected);
  check(!found && rejected, "an entry of another format is rejected");
  std::string text = "not an object";
  cantFail(HostObjectCache::writeEntry(
      path, key, MemoryBufferRef(text, "program"), 1.0));
  found = HostObjectCache::readEntry(path, key, seconds, rejected);
  check(!found && rejected, "an intact entry that holds no object is rejected");
  cantFail(HostObjectCache::writeEntry(
      path, key, MemoryBufferRef(object, "program"), 1.5));
  found = HostObjectCache::readEntry(path, key, seconds, rejected);
  check(found && !rejected, "a rewritten entry is read again");

  // The bound: the entries used least recently go first, and temporary
  // files go once they are older than an hour.
  std::string paths[3];
  for (int i = 0; i < 3; ++i) {
    std::string name = cache.getKey(*program(context, *host, 10 + i));
    paths[i] = entry(HostObjectCache::getEntryName(name));
    cantFail(HostObjectCache::writeEntry(
        paths[i], name, MemoryBufferRef(object, "program"), 1.0));
  }
  cantFail(errorCodeToError(sys::fs::remove(path)));
  age(paths[0], std::chrono::seconds(30));
  age(paths[1], std::chrono::seconds(10));
  age(paths[2], std::chrono::seconds(20));
  std::string stale = entry("x.o.tmp-stale"), fresh = entry("y.o.tmp-fresh");
  write(stale, "partial");
  write(fresh, "partial");
  age(stale, std::chrono::hours(2));
  uint64_t size = 0;
  cantFail(errorCodeToError(sys::fs::file_size(paths[0], size)));
  HostObjectCache::evict(objects, 2 * size);
  check(!sys::fs::exists(paths[0]) && sys::fs::exists(paths[1]) &&
            sys::fs::exists(paths[2]),
        "the entry used least recently is evicted first");
  check(!sys::fs::exists(stale) && sys::fs::exists(fresh),
        "a stale temporary file is removed, a recent one is kept");
  HostObjectCache::evict(objects, 0);
  check(!sys::fs::exists(paths[1]) && !sys::fs::exists(paths[2]),
        "a bound of zero keeps no entry");

  outs() << (failures ? "compile cache checks FAILED\n"
                      : "compile cache checks passed\n");
  return failures ? 1 : 0;
}
