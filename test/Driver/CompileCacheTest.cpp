// The keys, entries, and bound of the compile cache of host objects
// (D212): a changed key part misses, a damaged entry is
// rejected, and eviction removes the entries used least recently.
// clearCache (D217) removes the entries of this format
// and nothing else, and a process that clears while another writes leaves
// only intact entries.
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
#include <sys/wait.h>
#include <unistd.h>

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
  evictCache(directory, 2 * size);
  check(!sys::fs::exists(paths[0]) && sys::fs::exists(paths[1]) &&
            sys::fs::exists(paths[2]),
        "the entry used least recently is evicted first");
  check(!sys::fs::exists(stale) && sys::fs::exists(fresh),
        "a stale temporary file is removed, a recent one is kept");
  evictCache(directory, 0);
  check(!sys::fs::exists(paths[1]) && !sys::fs::exists(paths[2]),
        "a bound of zero keeps no entry");

  // The entries of the GPU modules (D214): a check of the
  // data rejects an entry, and one bound covers the entries of every kind.
  SmallString<256> gpu(directory);
  sys::path::append(gpu, "gpu");
  cantFail(errorCodeToError(sys::fs::create_directories(gpu)));
  auto gpuEntry = [&](StringRef key, StringRef extension) {
    SmallString<256> path(gpu);
    sys::path::append(path, getCacheEntryName(key, extension));
    return std::string(path);
  };
  std::string ptxKey = "ptx of a module", cubinKey = "cubin of a module";
  std::string ptx = gpuEntry(ptxKey, ".ptx"), cubin = gpuEntry(cubinKey, ".cubin");
  check(getCacheEntryName(ptxKey, ".ptx") != getCacheEntryName(cubinKey, ".ptx"),
        "another GPU key names another entry");
  cantFail(writeCacheEntry(ptx, ptxKey, ".version 8.0\n.target sm_86\n", 1.0));
  auto isPtx = [](StringRef data) { return data.contains(".version"); };
  found = readCacheEntry(ptx, ptxKey, seconds, rejected, isPtx);
  check(found && !rejected && seconds == 1.0, "a GPU entry gives back its data");
  cantFail(writeCacheEntry(ptx, ptxKey, "not PTX", 1.0));
  found = readCacheEntry(ptx, ptxKey, seconds, rejected, isPtx);
  check(!found && rejected, "a GPU entry whose data fails its check is rejected");
  cantFail(writeCacheEntry(ptx, ptxKey, object, 1.0));
  cantFail(writeCacheEntry(cubin, cubinKey, object, 1.0));
  std::string hostKey20 = cache.getKey(*program(context, *host, 20));
  std::string hostEntry = entry(HostObjectCache::getEntryName(hostKey20));
  cantFail(HostObjectCache::writeEntry(
      hostEntry, hostKey20, MemoryBufferRef(object, "program"), 1.0));
  age(ptx, std::chrono::seconds(30));
  age(hostEntry, std::chrono::seconds(20));
  age(cubin, std::chrono::seconds(10));
  uint64_t hostSize = 0, cubinSize = 0;
  cantFail(errorCodeToError(sys::fs::file_size(hostEntry, hostSize)));
  cantFail(errorCodeToError(sys::fs::file_size(cubin, cubinSize)));
  evictCache(directory, hostSize + cubinSize);
  check(!sys::fs::exists(ptx) && sys::fs::exists(hostEntry) &&
            sys::fs::exists(cubin),
        "one bound covers the host and GPU entries, least recent first");

  // Clearing (D217): the entries of this format go,
  // of every kind; another format, other names, the temporary file of a
  // writer under way, and the directories stay; a stale temporary file goes.
  std::string foreign = entry("foreign.o"), other = entry("notes.txt");
  write(foreign, "OTHERFMT and more bytes than a tag");
  write(other, intact);
  write(stale, "partial");
  age(stale, std::chrono::hours(2));
  ClearedCache cleared = clearCache(directory);
  check(cleared.hostEntries == 1 && cleared.gpuEntries == 1 &&
            cleared.bytes == hostSize + cubinSize,
        "a clear removes the host and GPU entries and counts their bytes");
  check(!sys::fs::exists(hostEntry) && !sys::fs::exists(cubin),
        "a cleared entry is gone");
  check(sys::fs::exists(foreign) && sys::fs::exists(other) &&
            sys::fs::exists(fresh) && !sys::fs::exists(stale) &&
            sys::fs::is_directory(objects) && sys::fs::is_directory(gpu),
        "a clear leaves other formats, other names, a writer's temporary "
        "file, and the directories, and removes a stale temporary file");
  cleared = clearCache(directory);
  check(cleared.hostEntries == 0 && cleared.gpuEntries == 0 &&
            cleared.bytes == 0,
        "a second clear removes nothing");
  SmallString<256> absent(directory);
  sys::path::append(absent, "absent");
  cleared = clearCache(absent);
  check(cleared.hostEntries == 0 && cleared.gpuEntries == 0 &&
            !sys::fs::exists(absent),
        "a clear of a directory that does not exist removes nothing and "
        "makes nothing");
  for (StringRef file : {foreign, other, fresh})
    sys::fs::remove(file);

  // A process that writes entries while this one clears: every write
  // succeeds, and every entry that remains is intact.
  constexpr int kKeys = 50, kWrites = 3000;
  auto raceKey = [](int i) { return "race key " + std::to_string(i % kKeys); };
  auto racePath = [&](int i) {
    return i % 2 ? gpuEntry(raceKey(i), ".cubin")
                 : entry(getCacheEntryName(raceKey(i), ".o"));
  };
  outs().flush();
  pid_t writer = fork();
  if (writer == 0) {
    int failed = 0;
    for (int i = 0; i < kWrites; ++i)
      if (auto error = writeCacheEntry(racePath(i), raceKey(i), object, 1.0)) {
        consumeError(std::move(error));
        ++failed;
      }
    _exit(failed ? 1 : 0);
  }
  check(writer > 0, "the writer starts");
  int clears = 0, status = 0;
  uint64_t removed = 0;
  while (writer > 0 && waitpid(writer, &status, WNOHANG) == 0) {
    ClearedCache c = clearCache(directory);
    removed += c.hostEntries + c.gpuEntries;
    ++clears;
  }
  check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "every write succeeds while another process clears");
  int remaining = 0, intactEntries = 0;
  for (int i = 0; i < kKeys; ++i) {
    if (!sys::fs::exists(racePath(i)))
      continue;
    ++remaining;
    found = readCacheEntry(racePath(i), raceKey(i), seconds, rejected);
    intactEntries += found && !rejected && found->getBuffer() == object;
  }
  int temporaries = 0;
  for (StringRef kind : {StringRef(objects), StringRef(gpu)}) {
    std::error_code error;
    for (sys::fs::directory_iterator it(kind, error), end; it != end && !error;
         it.increment(error))
      temporaries += StringRef(it->path()).contains(".tmp-");
  }
  outs() << "clear race: writes " << kWrites << ", clears " << clears
         << ", removed " << removed << ", remaining " << remaining
         << ", intact " << intactEntries << ", temporary files "
         << temporaries << "\n";
  check(clears > 0 && removed > 0, "the clears removed entries meanwhile");
  check(intactEntries == remaining && temporaries == 0,
        "every entry that remains is intact, and no temporary file is left");

  // The total of the cache (D[compile-cache-size-file], issue #196): a
  // store adds its bytes to `<directory>/size` and lists the directory
  // only when the bound may be exceeded, when no total is known, or when
  // the last listing is a day old. A cache of its own, with entries of one
  // size.
  SmallString<256> sized(directory);
  sys::path::append(sized, "sized");
  SmallString<256> sizedGpu(sized), sizeFile(sized);
  sys::path::append(sizedGpu, "gpu");
  sys::path::append(sizeFile, "size");
  cantFail(errorCodeToError(sys::fs::create_directories(sizedGpu)));
  const uint64_t kUnbounded = uint64_t(1) << 40;
  // Keys of one length, so that the entries have one size.
  auto sizedKey = [](int i) {
    return "sized key " + std::to_string(10000 + i);
  };
  auto sizedEntry = [&](int i) {
    SmallString<256> path(sizedGpu);
    sys::path::append(path, getCacheEntryName(
                                sizedKey(i), ".ptx"));
    return std::string(path);
  };
  // Stores entry `i` as a compile does, and returns its bytes.
  auto store = [&](int i, uint64_t bound) {
    uint64_t written = 0;
    cantFail(writeCacheEntry(sizedEntry(i), sizedKey(i),
                             object, 1.0, &written));
    noteCacheStores(sized, written, bound);
    return written;
  };
  auto recorded = [&] {
    std::string content = read(sizeFile);
    StringRef text(content);
    uint64_t total = ~uint64_t(0);
    text.split('\n').second.split(' ').first.getAsInteger(10, total);
    return total;
  };
  auto onDisk = [&] {
    uint64_t total = 0;
    std::error_code error;
    for (sys::fs::directory_iterator it(sizedGpu, error), end;
         it != end && !error; it.increment(error)) {
      uint64_t size = 0;
      if (!sys::fs::file_size(it->path(), size))
        total += size;
    }
    return total;
  };
  const int kStores = 300;
  uint64_t before = getCacheListingCount(), entrySize = 0;
  for (int i = 0; i < kStores; ++i)
    entrySize = store(i, kUnbounded);
  uint64_t listed = getCacheListingCount() - before;
  outs() << "size file: " << kStores << " stores, listings " << listed
         << "\n";
  check(listed == 1, "of many stores below the bound only the first, which "
                     "finds no total, lists the directory");
  check(recorded() == kStores * entrySize && onDisk() == recorded(),
        "the total kept is the bytes of the entries");

  // At the bound: no listing while the total fits; the store that exceeds
  // it lists and removes the entry used least recently, and the total kept
  // is what the listing found.
  uint64_t bound = (kStores + 2) * entrySize;
  age(sizedEntry(7), std::chrono::seconds(60));
  before = getCacheListingCount();
  store(kStores, bound);
  store(kStores + 1, bound);
  check(getCacheListingCount() == before && recorded() == bound,
        "stores up to the bound do not list");
  store(kStores + 2, bound);
  check(getCacheListingCount() == before + 1 &&
            !sys::fs::exists(sizedEntry(7)) &&
            sys::fs::exists(sizedEntry(kStores + 2)) && onDisk() == bound &&
            recorded() == bound,
        "the store that exceeds the bound lists once and evicts the entry "
        "used least recently");

  // An entry removed by hand leaves the total too large, never too small:
  // the next store over the bound lists, finds room, and evicts nothing.
  cantFail(errorCodeToError(sys::fs::remove(sizedEntry(8))));
  before = getCacheListingCount();
  store(kStores + 3, bound);
  check(getCacheListingCount() == before + 1 && onDisk() == bound &&
            recorded() == bound && sys::fs::exists(sizedEntry(9)),
        "a total that is too large is corrected by a listing");

  // A total whose listing is older than a day, one without a readable
  // total, and a file that cannot be used each list.
  auto setSizeFile = [&](StringRef text) {
    cantFail(errorCodeToError(sys::fs::remove(sizeFile)));
    write(sizeFile, text);
  };
  int64_t current = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
  // The total here is too small on purpose: the listing replaces it.
  setSizeFile("mdir-cache-size 1\n1 " + std::to_string(current - 2 * 86400) +
              "\n");
  before = getCacheListingCount();
  store(kStores + 4, kUnbounded);
  check(getCacheListingCount() == before + 1 && recorded() == onDisk(),
        "a total listed more than a day ago is listed again");
  store(kStores + 5, kUnbounded);
  check(getCacheListingCount() == before + 1 && recorded() == onDisk(),
        "and the store after it does not list");
  setSizeFile("something else\n");
  store(kStores + 6, kUnbounded);
  check(getCacheListingCount() == before + 2 && recorded() == onDisk(),
        "a file without a readable total is replaced by a listing");
  cantFail(errorCodeToError(sys::fs::remove(sizeFile)));
  cantFail(errorCodeToError(sys::fs::create_directory(sizeFile)));
  before = getCacheListingCount();
  uint64_t held = onDisk();
  store(kStores + 7, kUnbounded);
  store(kStores + 8, held);
  check(getCacheListingCount() == before + 2 && onDisk() == held,
        "without a usable file every store lists, and the bound holds");
  cantFail(errorCodeToError(sys::fs::remove(sizeFile)));

  // Processes that store at once add under the file's lock: the total is
  // the bytes on disk, and none of them lists after the first.
  store(kStores + 9, kUnbounded);
  const int kProcesses = 4, kEach = 100;
  pid_t children[kProcesses];
  for (int p = 0; p < kProcesses; ++p) {
    children[p] = fork();
    if (children[p] == 0) {
      uint64_t start = getCacheListingCount();
      for (int i = 0; i < kEach; ++i)
        store(1000 + p * kEach + i, kUnbounded);
      _exit(getCacheListingCount() == start ? 0 : 1);
    }
  }
  bool quiet = true;
  for (pid_t child : children) {
    int status = 0;
    waitpid(child, &status, 0);
    quiet &= WIFEXITED(status) && WEXITSTATUS(status) == 0;
  }
  check(quiet && recorded() == onDisk(),
        "concurrent processes keep one exact total and do not list");

  // A bound of zero lists at every store and keeps nothing; a clear takes
  // what it removes off the total.
  before = getCacheListingCount();
  store(2000, 0);
  store(2001, 0);
  check(getCacheListingCount() == before + 2 && onDisk() == 0 &&
            recorded() == 0,
        "a bound of zero lists at every store and keeps no entry");
  store(2002, kUnbounded);
  store(2003, kUnbounded);
  cleared = clearCache(sized);
  before = getCacheListingCount();
  check(cleared.gpuEntries == 2 && recorded() == 0,
        "a clear takes the bytes it removes off the total");
  store(2004, kUnbounded);
  check(getCacheListingCount() == before && recorded() == entrySize,
        "and the store after a clear does not list");

  outs() << (failures ? "compile cache checks FAILED\n"
                      : "compile cache checks passed\n");
  return failures ? 1 : 0;
}
