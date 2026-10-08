//===- CompileCache.cpp - The compile cache (D212) ----------------------===//
//
// The object cache follows the interface that LLVM's ORC JIT offers for it
// (llvm::ObjectCache, as in LLVM's LLJITWithObjectCache example); entries
// are content-addressed by the module they were generated from.
//
//===----------------------------------------------------------------------===//

#include "mdir/Compiler/CompileCache.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <unistd.h>
#include <vector>

using namespace llvm;
using namespace mdir::compiler;

namespace {
// The layout of an entry; a change of it changes this tag, so old entries
// are rejected (and replaced) rather than misread.
constexpr char kMagic[8] = {'M', 'D', 'I', 'R', 'O', 'B', 'J', '1'};
constexpr StringLiteral kFormat = "mdir-object-cache 1";
// A temporary file older than this is left by a process that died.
constexpr auto kStaleTemporary = std::chrono::hours(1);
// The file of the cache's total, `<directory>/size`: this tag, then the
// bytes of the entries and the time of the last listing, in seconds since
// the epoch.
constexpr StringLiteral kSizeTag = "mdir-cache-size 1";
// The total is taken from a listing again once the last one is this old,
// which removes what processes that died left: bytes they stored and did
// not add to the total, and their temporary files.
constexpr auto kListingPeriod = std::chrono::hours(24);
std::atomic<uint64_t> listings{0};

double now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/// A stream that hashes what is written to it.
class HashStream : public raw_ostream {
public:
  HashStream() { SetUnbuffered(); }
  BLAKE3 hasher;

private:
  uint64_t position = 0;
  void write_impl(const char *data, size_t size) override {
    hasher.update(ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(data),
                                    size));
    position += size;
  }
  uint64_t current_pos() const override { return position; }
};

std::string hashObject(StringRef bytes) {
  return toHex(BLAKE3::hash(arrayRefFromStringRef(bytes)),
               /*LowerCase=*/true);
}

template <typename T> void append(std::string &out, T value) {
  out.append(reinterpret_cast<const char *>(&value), sizeof(T));
}
template <typename T> bool take(StringRef &in, T &value) {
  if (in.size() < sizeof(T))
    return false;
  std::memcpy(&value, in.data(), sizeof(T));
  in = in.drop_front(sizeof(T));
  return true;
}
} // namespace

CompileStats &CompileStats::operator+=(const CompileStats &other) {
  programs += other.programs;
  pipelineSeconds += other.pipelineSeconds;
  engineSeconds += other.engineSeconds;
  compiled += other.compiled;
  compileSeconds += other.compileSeconds;
  hits += other.hits;
  savedSeconds += other.savedSeconds;
  rejected += other.rejected;
  stored += other.stored;
  unstored += other.unstored;
  lookupSeconds += other.lookupSeconds;
  gpuModules += other.gpuModules;
  gpuSerializeSeconds += other.gpuSerializeSeconds;
  gpuPtxCompiled += other.gpuPtxCompiled;
  gpuPtxHits += other.gpuPtxHits;
  gpuCubinCompiled += other.gpuCubinCompiled;
  gpuCubinHits += other.gpuCubinHits;
  gpuCompileSeconds += other.gpuCompileSeconds;
  gpuSavedSeconds += other.gpuSavedSeconds;
  gpuLookupSeconds += other.gpuLookupSeconds;
  gpuRejected += other.gpuRejected;
  gpuStored += other.gpuStored;
  gpuUnstored += other.gpuUnstored;
  bypassed += other.bypassed;
  reused += other.reused;
  reuseSavedSeconds += other.reuseSavedSeconds;
  return *this;
}

std::shared_ptr<const KeptCode> CodeStore::find(StringRef key) const {
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto &entry : entries)
    if (entry.first == key)
      return entry.second;
  return nullptr;
}

void CodeStore::insert(StringRef key, std::shared_ptr<const KeptCode> code) {
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto &entry : entries)
    if (entry.first == key)
      return;
  entries.emplace_back(key.str(), std::move(code));
  if (entries.size() > capacity)
    entries.erase(entries.begin());
}

size_t CodeStore::size() const {
  std::lock_guard<std::mutex> lock(mutex);
  return entries.size();
}

std::string mdir::compiler::getCodeKey(StringRef module, StringRef pipeline,
                                       StringRef entry, StringRef machine) {
  // Each part with its length, so that no two lists of parts give the same
  // bytes.
  BLAKE3 hasher;
  for (StringRef part : {module, pipeline, entry, machine}) {
    uint64_t size = part.size();
    hasher.update(StringRef(reinterpret_cast<const char *>(&size), sizeof size));
    hasher.update(part);
  }
  return toHex(hasher.final(), /*LowerCase=*/true);
}

std::optional<CompileCacheConfig> CompileCacheConfig::fromEnvironment() {
  if (auto off = sys::Process::GetEnv("MDIR_COMPILE_CACHE"))
    if (StringRef(*off).equals_insensitive("off"))
      return std::nullopt;
  auto directory = sys::Process::GetEnv("MDIR_COMPILE_CACHE_DIR");
  if (!directory || directory->empty())
    return std::nullopt;
  CompileCacheConfig config;
  config.directory = *directory;
  if (auto size = sys::Process::GetEnv("MDIR_COMPILE_CACHE_MAX_MB")) {
    uint64_t megabytes = 0;
    if (!StringRef(*size).trim().getAsInteger(10, megabytes))
      config.maxBytes = megabytes << 20;
  }
  return config;
}

std::string mdir::compiler::describeMachine(const TargetMachine &machine,
                                            StringRef extra) {
  std::string text;
  raw_string_ostream os(text);
  const TargetOptions &o = machine.Options;
  os << kFormat << '\n'
     << "llvm " << LLVM_VERSION_STRING << '\n'
     << "triple " << machine.getTargetTriple().str() << '\n'
     << "cpu " << machine.getTargetCPU() << '\n'
     << "features " << machine.getTargetFeatureString() << '\n'
     << "opt-level " << static_cast<int>(machine.getOptLevel()) << '\n'
     << "code-model " << static_cast<int>(machine.getCodeModel()) << '\n'
     << "relocation " << static_cast<int>(machine.getRelocationModel()) << '\n'
     << "data-layout " << machine.createDataLayout().getStringRepresentation()
     << '\n'
     << "options " << static_cast<int>(o.FloatABIType) << ' '
     << static_cast<int>(o.AllowFPOpFusion) << ' '
     << static_cast<int>(o.ExceptionModel) << ' ' << o.EnableFastISel << ' '
     << o.EnableGlobalISel << ' ' << o.FunctionSections << ' '
     << o.DataSections << ' ' << o.EmulatedTLS << ' ' << o.UseInitArray << ' '
     << o.TrapUnreachable << ' ' << o.NoTrappingFPMath << '\n'
     << "extra " << extra << '\n';
  return text;
}

HostObjectCache::HostObjectCache(std::optional<CompileCacheConfig> config,
                                 std::string machine, std::string module)
    : config(std::move(config)), machine(std::move(machine)),
      module(std::move(module)) {}

HostObjectCache::~HostObjectCache() = default;

std::string HostObjectCache::getKey(const Module &module) const {
  HashStream stream;
  WriteBitcodeToFile(module, stream, /*ShouldPreserveUseListOrder=*/true);
  return machine + "module " +
         toHex(stream.hasher.final(), /*LowerCase=*/true) + '\n';
}

std::string HostObjectCache::getEntryName(StringRef key) {
  return getCacheEntryName(key, ".o");
}

std::string HostObjectCache::getObjectDirectory(StringRef directory) {
  SmallString<256> path(directory);
  sys::path::append(path, "host");
  return std::string(path);
}

std::unique_ptr<MemoryBuffer>
mdir::compiler::readCacheEntry(StringRef path, StringRef key, double &seconds,
                               bool &rejected,
                               function_ref<bool(StringRef)> isValid) {
  rejected = false;
  // Read, not mapped: another process may replace the file by a rename.
  auto file = MemoryBuffer::getFile(path, /*IsText=*/false,
                                    /*RequiresNullTerminator=*/false,
                                    /*IsVolatile=*/true);
  if (!file)
    return nullptr;
  rejected = true;
  StringRef in = (*file)->getBuffer();
  if (!in.consume_front(StringRef(kMagic, sizeof(kMagic))))
    return nullptr;
  uint64_t keyLength = 0, objectLength = 0;
  if (!take(in, keyLength) || in.size() < keyLength ||
      in.take_front(keyLength) != key)
    return nullptr;
  in = in.drop_front(keyLength);
  if (!take(in, objectLength) || !take(in, seconds) || in.size() < 64)
    return nullptr;
  StringRef hash = in.take_front(64);
  in = in.drop_front(64);
  if (in.size() != objectLength || hashObject(in) != hash)
    return nullptr;
  // A copy, aligned, which a parser of objects needs.
  auto data = MemoryBuffer::getMemBufferCopy(in, path);
  if (isValid && !isValid(data->getBuffer()))
    return nullptr;
  rejected = false;
  return data;
}

std::unique_ptr<MemoryBuffer>
HostObjectCache::readEntry(StringRef path, StringRef key, double &seconds,
                           bool &rejected) {
  // An object LLVM cannot read is rejected here rather than at linking.
  return readCacheEntry(path, key, seconds, rejected, [](StringRef data) {
    auto parsed = object::ObjectFile::createObjectFile(
        MemoryBufferRef(data, "cached object"));
    if (!parsed) {
      consumeError(parsed.takeError());
      return false;
    }
    return true;
  });
}

Error HostObjectCache::writeEntry(StringRef path, StringRef key,
                                  MemoryBufferRef object, double seconds,
                                  uint64_t *written) {
  return writeCacheEntry(path, key, object.getBuffer(), seconds, written);
}

Error mdir::compiler::writeCacheEntry(StringRef path, StringRef key,
                                      StringRef object, double seconds,
                                      uint64_t *written) {
  std::string data(kMagic, sizeof(kMagic));
  append<uint64_t>(data, key.size());
  data += key;
  append<uint64_t>(data, object.size());
  append<double>(data, seconds);
  data += hashObject(object);
  data += object;
  // A unique temporary file in the same directory, then a rename: a
  // reader sees the old entry, the new one, or none, never a part.
  SmallString<256> model(path);
  model += ".tmp-%%%%%%%%%%%%";
  int fd = -1;
  SmallString<256> temporary;
  if (auto error = sys::fs::createUniqueFile(model, fd, temporary))
    return errorCodeToError(error);
  {
    raw_fd_ostream os(fd, /*shouldClose=*/true);
    os << data;
    os.close();
    if (os.has_error()) {
      std::error_code error = os.error();
      os.clear_error();
      sys::fs::remove(temporary);
      return errorCodeToError(error);
    }
  }
  if (auto error = sys::fs::rename(temporary, path)) {
    sys::fs::remove(temporary);
    return errorCodeToError(error);
  }
  if (written)
    *written = data.size();
  return Error::success();
}

std::string mdir::compiler::getCacheEntryName(StringRef key,
                                              StringRef extension) {
  return hashObject(key) + extension.str();
}

void mdir::compiler::touchCacheEntry(StringRef path) {
  sys::fs::setLastAccessAndModificationTime(path,
                                            std::chrono::system_clock::now());
}

namespace {
/// Lists the entries of every kind, removes the temporary files left by
/// processes that died and, least recently used first, entries until at
/// most `maxBytes` are left. Returns the bytes left.
uint64_t listAndEvict(StringRef directory, uint64_t maxBytes) {
  ++listings;
  struct Entry {
    std::string path;
    uint64_t size;
    sys::TimePoint<> used;
  };
  std::vector<Entry> entries;
  uint64_t total = 0;
  auto current = std::chrono::system_clock::now();
  // One bound for the entries of every kind.
  for (StringRef kind : {"host", "gpu"}) {
    SmallString<256> subdirectory(directory);
    sys::path::append(subdirectory, kind);
    std::error_code error;
    for (sys::fs::directory_iterator it(subdirectory, error), end;
         it != end && !error; it.increment(error)) {
      sys::fs::file_status status;
      // Another process may have removed it since the listing.
      if (sys::fs::status(it->path(), status) ||
          status.type() != sys::fs::file_type::regular_file)
        continue;
      StringRef name = sys::path::filename(it->path());
      if (name.contains(".tmp-")) {
        if (current - status.getLastModificationTime() > kStaleTemporary)
          sys::fs::remove(it->path());
        continue;
      }
      if (!name.ends_with(".o") && !name.ends_with(".ptx") &&
          !name.ends_with(".cubin"))
        continue;
      entries.push_back({it->path(), status.getSize(),
                         status.getLastModificationTime()});
      total += status.getSize();
    }
  }
  if (total <= maxBytes)
    return total;
  std::sort(entries.begin(), entries.end(),
            [](const Entry &a, const Entry &b) { return a.used < b.used; });
  for (const Entry &entry : entries) {
    if (total <= maxBytes)
      break;
    // A file another process removed first counts as removed.
    sys::fs::remove(entry.path);
    total -= entry.size;
  }
  return total;
}

/// The file of the total of the cache at `directory`, open and locked
/// against other processes and other threads until it is destroyed. A
/// process holds it only while it adds to the total or lists the directory.
class SizeFile {
public:
  SizeFile(StringRef directory, bool create) : lock(getMutex()) {
    SmallString<256> path(directory);
    sys::path::append(path, "size");
    auto opened = sys::fs::openNativeFileForReadWrite(
        path, create ? sys::fs::CD_OpenAlways : sys::fs::CD_OpenExisting,
        sys::fs::OF_None);
    if (!opened) {
      consumeError(opened.takeError());
      return;
    }
    fd = *opened;
    // A lock of the file between processes (fcntl), which a network file
    // system carries; a file system without locks leaves the file unused.
    if (sys::fs::lockFile(fd)) {
      ::close(fd);
      fd = -1;
    }
  }
  ~SizeFile() {
    // Closing the file gives the lock back.
    if (fd >= 0)
      ::close(fd);
  }
  bool usable() const { return fd >= 0; }

  /// The total and the time of the last listing, if the file holds them.
  bool read(uint64_t &total, int64_t &listed) const {
    char buffer[128];
    ssize_t size = ::pread(fd, buffer, sizeof(buffer), 0);
    if (size <= 0)
      return false;
    StringRef text(buffer, size);
    if (!text.consume_front(kSizeTag) || !text.consume_front("\n"))
      return false;
    auto [first, second] = text.split('\n').first.split(' ');
    return !first.getAsInteger(10, total) && !second.getAsInteger(10, listed);
  }
  void write(uint64_t total, int64_t listed) const {
    std::string text =
        (kSizeTag + "\n" + Twine(total) + " " + Twine(listed) + "\n").str();
    if (::pwrite(fd, text.data(), text.size(), 0) !=
            static_cast<ssize_t>(text.size()) ||
        ::ftruncate(fd, text.size()) != 0)
      // A total that cannot be written is taken from a listing next time.
      (void)!::ftruncate(fd, 0);
  }

private:
  // A lock by fcntl belongs to the process, and closing any descriptor of
  // the file gives it back: one thread at a time opens the file.
  static std::mutex &getMutex() {
    static std::mutex mutex;
    return mutex;
  }
  std::lock_guard<std::mutex> lock;
  int fd = -1;
};

int64_t secondsSinceEpoch() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
} // namespace

void mdir::compiler::evictCache(StringRef directory, uint64_t maxBytes) {
  SizeFile file(directory, /*create=*/false);
  uint64_t total = listAndEvict(directory, maxBytes);
  if (file.usable())
    file.write(total, secondsSinceEpoch());
}

void mdir::compiler::noteCacheStores(StringRef directory, uint64_t bytes,
                                     uint64_t maxBytes) {
  SizeFile file(directory, /*create=*/true);
  if (!file.usable()) {
    // No total to rely on: the directory is listed, as after every store
    // before the total was kept.
    listAndEvict(directory, maxBytes);
    return;
  }
  uint64_t total = 0;
  int64_t listed = 0, current = secondsSinceEpoch();
  int64_t period = std::chrono::seconds(kListingPeriod).count();
  // The total is exact after a listing and never too small afterwards,
  // but for the bytes of a process that died between its stores and this
  // call: an entry that replaces another, or that another process removed,
  // is still counted. The directory is listed when the total may exceed
  // the bound, when no total is known, and once in a period (the clocks of
  // two nodes may differ, so a time ahead by less than a period is taken).
  bool known = file.read(total, listed) && current - listed < period &&
               listed - current < period;
  total += bytes;
  if (known && total <= maxBytes) {
    file.write(total, listed);
    return;
  }
  file.write(listAndEvict(directory, maxBytes), current);
}

uint64_t mdir::compiler::getCacheListingCount() { return listings; }

ClearedCache mdir::compiler::clearCache(StringRef directory) {
  ClearedCache cleared;
  auto current = std::chrono::system_clock::now();
  for (StringRef kind : {"host", "gpu"}) {
    SmallString<256> subdirectory(directory);
    sys::path::append(subdirectory, kind);
    std::error_code error;
    for (sys::fs::directory_iterator it(subdirectory, error), end;
         it != end && !error; it.increment(error)) {
      sys::fs::file_status status;
      // Another process may have removed it since the listing.
      if (sys::fs::status(it->path(), status) ||
          status.type() != sys::fs::file_type::regular_file)
        continue;
      StringRef name = sys::path::filename(it->path());
      // The temporary file of a writer under way is left, so that its
      // rename succeeds; one older than an hour was left by a process
      // that died.
      if (name.contains(".tmp-")) {
        if (current - status.getLastModificationTime() > kStaleTemporary)
          sys::fs::remove(it->path());
        continue;
      }
      if (!name.ends_with(".o") && !name.ends_with(".ptx") &&
          !name.ends_with(".cubin"))
        continue;
      // Only entries of this format: the magic tag of the layout.
      auto head = MemoryBuffer::getFileSlice(it->path(), sizeof(kMagic), 0,
                                             /*IsVolatile=*/true);
      if (!head || (*head)->getBuffer() != StringRef(kMagic, sizeof(kMagic)))
        continue;
      // A file another process removed first counts as left to it.
      if (sys::fs::remove(it->path(), /*IgnoreNonExisting=*/false))
        continue;
      ++(kind == "host" ? cleared.hostEntries : cleared.gpuEntries);
      cleared.bytes += status.getSize();
    }
  }
  // The total loses what was removed. An entry that its writer had not yet
  // added leaves the total too large, which the next listing corrects.
  if (cleared.bytes) {
    SizeFile file(directory, /*create=*/false);
    uint64_t total = 0;
    int64_t listed = 0;
    if (file.usable() && file.read(total, listed))
      file.write(total - std::min(total, cleared.bytes), listed);
  }
  return cleared;
}

std::unique_ptr<MemoryBuffer> HostObjectCache::getObject(const Module *module) {
  if (module->getModuleIdentifier() != this->module)
    return nullptr;
  // The code that the program keeps: a copy of its object, which the
  // engine links into memory of its own (D[program-reuse]).
  if (kept)
    return MemoryBuffer::getMemBufferCopy(kept->object, this->module);
  // The key is the bitcode of the module before code generation, which
  // changes the module; without a directory, only the generation is timed.
  Pending request;
  double start = now();
  if (config) {
    request.key = getKey(*module);
    SmallString<256> path(getObjectDirectory(config->directory));
    sys::path::append(path, getEntryName(request.key));
    double seconds = 0.0;
    bool rejected = false;
    if (auto object = readEntry(path, request.key, seconds, rejected)) {
      // Marks the entry as used, for the eviction of the least recent.
      touchCacheEntry(path);
      std::lock_guard<std::mutex> lock(mutex);
      ++stats.hits;
      stats.savedSeconds += seconds;
      stats.lookupSeconds += now() - start;
      if (keeps)
        this->object = object->getBuffer().str();
      return object;
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (rejected)
      ++stats.rejected;
  }
  request.start = now();
  std::lock_guard<std::mutex> lock(mutex);
  stats.lookupSeconds += request.start - start;
  pending[module] = std::move(request);
  return nullptr;
}

void HostObjectCache::notifyObjectCompiled(const Module *module,
                                       MemoryBufferRef object) {
  Pending request;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto found = pending.find(module);
    if (found == pending.end())
      return;
    request = std::move(found->second);
    pending.erase(found);
  }
  double seconds = now() - request.start;
  bool stored = false;
  uint64_t written = 0;
  if (config) {
    std::string directory = getObjectDirectory(config->directory);
    SmallString<256> path(directory);
    sys::path::append(path, getEntryName(request.key));
    // A cache that cannot be written is no cache; the run goes on.
    if (sys::fs::create_directories(directory)) {
      stored = false;
    } else if (auto error =
                   writeEntry(path, request.key, object, seconds, &written)) {
      consumeError(std::move(error));
    } else {
      stored = true;
      noteCacheStores(config->directory, written, config->maxBytes);
    }
  }
  std::lock_guard<std::mutex> lock(mutex);
  ++stats.compiled;
  stats.compileSeconds += seconds;
  if (config)
    ++(stored ? stats.stored : stats.unstored);
  if (keeps)
    this->object = object.getBuffer().str();
}

void HostObjectCache::setKeptCode(std::shared_ptr<const KeptCode> code) {
  kept = std::move(code);
}

std::string HostObjectCache::takeObject() {
  std::lock_guard<std::mutex> lock(mutex);
  return std::move(object);
}

CompileStats HostObjectCache::getStats() const {
  std::lock_guard<std::mutex> lock(mutex);
  return stats;
}
