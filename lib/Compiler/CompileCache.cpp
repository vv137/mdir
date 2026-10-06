//===- CompileCache.cpp - The cache of host objects (D[compile-cache]) ----===//
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
#include <chrono>
#include <cstring>
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
  return *this;
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
  return hashObject(key) + ".o";
}

std::string HostObjectCache::getObjectDirectory(StringRef directory) {
  SmallString<256> path(directory);
  sys::path::append(path, "host");
  return std::string(path);
}

std::unique_ptr<MemoryBuffer>
HostObjectCache::readEntry(StringRef path, StringRef key, double &seconds,
                       bool &rejected) {
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
  auto object = MemoryBuffer::getMemBufferCopy(in, path);
  // An object LLVM cannot read is rejected here rather than at linking.
  auto parsed = object::ObjectFile::createObjectFile(object->getMemBufferRef());
  if (!parsed) {
    consumeError(parsed.takeError());
    return nullptr;
  }
  rejected = false;
  return object;
}

Error HostObjectCache::writeEntry(StringRef path, StringRef key,
                              MemoryBufferRef object, double seconds) {
  std::string data(kMagic, sizeof(kMagic));
  append<uint64_t>(data, key.size());
  data += key;
  append<uint64_t>(data, object.getBufferSize());
  append<double>(data, seconds);
  data += hashObject(object.getBuffer());
  data += object.getBuffer();
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
  return Error::success();
}

void HostObjectCache::evict(StringRef directory, uint64_t maxBytes) {
  struct Entry {
    std::string path;
    uint64_t size;
    sys::TimePoint<> used;
  };
  std::vector<Entry> entries;
  uint64_t total = 0;
  auto current = std::chrono::system_clock::now();
  std::error_code error;
  for (sys::fs::directory_iterator it(directory, error), end;
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
    if (!name.ends_with(".o"))
      continue;
    entries.push_back({it->path(), status.getSize(),
                       status.getLastModificationTime()});
    total += status.getSize();
  }
  if (total <= maxBytes)
    return;
  std::sort(entries.begin(), entries.end(),
            [](const Entry &a, const Entry &b) { return a.used < b.used; });
  for (const Entry &entry : entries) {
    if (total <= maxBytes)
      break;
    // A file another process removed first counts as removed.
    sys::fs::remove(entry.path);
    total -= entry.size;
  }
}

std::unique_ptr<MemoryBuffer> HostObjectCache::getObject(const Module *module) {
  if (module->getModuleIdentifier() != this->module)
    return nullptr;
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
      sys::fs::setLastAccessAndModificationTime(path,
                                                std::chrono::system_clock::now());
      std::lock_guard<std::mutex> lock(mutex);
      ++stats.hits;
      stats.savedSeconds += seconds;
      stats.lookupSeconds += now() - start;
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
  if (config) {
    std::string directory = getObjectDirectory(config->directory);
    SmallString<256> path(directory);
    sys::path::append(path, getEntryName(request.key));
    // A cache that cannot be written is no cache; the run goes on.
    if (sys::fs::create_directories(directory)) {
      stored = false;
    } else if (auto error = writeEntry(path, request.key, object, seconds)) {
      consumeError(std::move(error));
    } else {
      stored = true;
      evict(directory, config->maxBytes);
    }
  }
  std::lock_guard<std::mutex> lock(mutex);
  ++stats.compiled;
  stats.compileSeconds += seconds;
  if (config)
    ++(stored ? stats.stored : stats.unstored);
}

CompileStats HostObjectCache::getStats() const {
  std::lock_guard<std::mutex> lock(mutex);
  return stats;
}
